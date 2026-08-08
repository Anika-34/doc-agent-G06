"""Stage 3 — OCR/HTR (BASELINE = pretrained TrOCR / Tesseract)

Transcribes page regions detected in Stage 2 using an OCR engine.

Strategy (baseline):
  1. Load OCR model from cfg['ocr']['model'] (e.g., "microsoft/trocr-base-printed")
  2. For each region (except figures), crop the image and run OCR
  3. Return transcribed text as Chunk objects with:
     - text: extracted transcription
     - page_ids: list of pages this chunk came from
     - doc_id: document identifier
  4. Optionally fine-tune on domain-specific data (cfg['ocr']['finetune'])

Reference: EDA §7 "OCR tool bake-off" — tool selection and performance metrics
          EDA §8 "Preprocessing sweep" — per-book preprocessing before OCR
          Config: cfg['ocr']['model'] (HF model ID or path), cfg['ocr']['finetune'] (bool)
"""
from __future__ import annotations
from pathlib import Path
from PIL import Image
import cv2
from ..contracts import *  # noqa
from ..logging_conf import get_logger

logger = get_logger(__name__)


class Reader:
    """OCR reader using pretrained TrOCR or Tesseract.
    
    Model selection via cfg['ocr']['model']:
      - HF model ID: "microsoft/trocr-base-printed" (default, Transformer-based)
      - Fallback: Tesseract with Bangla language pack (lang="ben")
    
    Optional fine-tuning via cfg['ocr']['finetune']:
      - true: load fine-tuned weights if available
      - false: use pretrained weights as-is
    """
    
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg.get("ocr", {})
        self.model_id = self.cfg.get("model", "microsoft/trocr-base-printed")
        self.use_finetune = self.cfg.get("finetune", False)
        self.language = "ben"  # Bangla
        
        self._reader = None
        self._use_tesseract = False
        
        # Determine which backend to use
        if "trocr" in self.model_id.lower():
            self._use_tesseract = False
            logger.info(f"OCR Reader: TrOCR model (lazy-loaded on first use)")
        else:
            self._use_tesseract = True
            logger.info(f"OCR Reader: Tesseract fallback (lang={self.language})")
    
    def transcribe_region(self, region: Region) -> str:
        """Transcribe a single region (crop image and run OCR).
        
        Args:
            region: Region object with bbox and kind
            
        Returns:
            Extracted text string (empty if OCR fails or region is non-text)
        """
        # Get page image path
        page_img_path = self._get_page_image_path(region.page_id)
        
        if not page_img_path or not Path(page_img_path).exists():
            logger.debug(f"Page image not found: {page_img_path} (region {region.page_id})")
            return ""
        
        try:
            # Crop region from page
            full_img = cv2.imread(str(page_img_path), cv2.IMREAD_COLOR)
            if full_img is None:
                return ""
            
            x1, y1, x2, y2 = region.bbox
            cropped = full_img[y1:y2, x1:x2]
            
            # Skip empty crops
            if cropped.size == 0:
                return ""
            
            # Convert to PIL
            pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            
            # Run OCR
            if self._use_tesseract:
                text = self._transcribe_tesseract(pil_img)
            else:
                text = self._transcribe_trocr(pil_img)
            
            return text.strip()
        
        except Exception as e:
            logger.debug(f"OCR failed for region {region.page_id}: {e}")
            return ""
    
    def _get_page_image_path(self, page_id: str) -> str | None:
        """Reconstruct page image path from page_id.
        
        Format: "BookName:page-191" → data/pages/BookName/page-191.png
        """
        if ":" not in page_id:
            return None
        
        try:
            book, page_part = page_id.split(":", 1)
            if page_part.startswith("page-"):
                page_num_str = page_part.split("-", 1)[1]
                
                # Try multiple path patterns
                paths = [
                    f"data/pages/{book}/{page_part}.png",
                    f"data/pages/{book}/page-{int(page_num_str):05d}.png",
                ]
                
                for path in paths:
                    if Path(path).exists():
                        return path
                
                # Return first pattern as default
                return paths[0]
        except (ValueError, IndexError):
            pass
        
        return None
    
    def _transcribe_tesseract(self, pil_img: Image.Image) -> str:
        """Tesseract OCR (lightweight baseline)."""
        try:
            import pytesseract
            return pytesseract.image_to_string(pil_img, lang=self.language)
        except Exception as e:
            logger.debug(f"Tesseract OCR error: {e}")
            return ""
    
    def _transcribe_trocr(self, pil_img: Image.Image) -> str:
        """TrOCR OCR using transformers library (Transformer-based)."""
        try:
            if self._reader is None:
                from transformers import VisionEncoderDecoderModel, TrOCRProcessor
                
                # Load model and processor
                logger.debug(f"Loading TrOCR model: {self.model_id}")
                self._reader = {
                    "model": VisionEncoderDecoderModel.from_pretrained(self.model_id),
                    "processor": TrOCRProcessor.from_pretrained(self.model_id),
                }
            
            processor = self._reader["processor"]
            model = self._reader["model"]
            
            # Process image and run inference
            pixel_values = processor(pil_img, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return text
        
        except Exception as e:
            logger.debug(f"TrOCR error: {e}. Falling back to Tesseract.")
            self._use_tesseract = True
            return self._transcribe_tesseract(pil_img)


def transcribe(regions: list[Region], cfg: dict) -> list[Chunk]:
    """Transcribe all regions to text chunks.
    
    Args:
        regions: List of Region objects from Stage 2 (layout detection)
        cfg: Config dict with ocr parameters
        
    Returns:
        List of Chunk objects with transcribed text
    """
    if not regions:
        logger.info("No regions to transcribe")
        return []
    
    reader = Reader(cfg)
    chunks = []
    
    # Group regions by page
    regions_by_page = {}
    for region in regions:
        if region.page_id not in regions_by_page:
            regions_by_page[region.page_id] = []
        regions_by_page[region.page_id].append(region)
    
    logger.info(f"Transcribing {len(regions)} regions from {len(regions_by_page)} pages...")
    
    chunk_id = 0
    for page_id, page_regions in regions_by_page.items():
        # Extract document ID from page_id format "DocName:page-N"
        doc_id = page_id.split(":")[0] if ":" in page_id else page_id
        
        # Sort regions top-to-bottom, left-to-right for reading order
        page_regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        
        for region in page_regions:
            # Skip figure-only regions (images/diagrams)
            if region.kind == "figure":
                continue
            
            # Transcribe region
            text = reader.transcribe_region(region)
            
            if text:  # Only create chunk if text was extracted
                chunk = Chunk(
                    id=f"chunk_{chunk_id:06d}",
                    doc_id=doc_id,
                    text=text,
                    page_ids=[region.page_id],
                    score=0.0  # Will be set by retrieval
                )
                chunks.append(chunk)
                chunk_id += 1
    
    logger.info(f"OCR transcription complete: {len(chunks)} chunks created from {len(regions)} regions")
    return chunks

