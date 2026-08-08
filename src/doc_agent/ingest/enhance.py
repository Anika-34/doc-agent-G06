"""Stage 1 — ENHANCEMENT (VAE / diffusion) — generative denoise / super-resolution of degraded scans"""
from __future__ import annotations
from ..contracts import *  # noqa

from pathlib import Path
import cv2
import numpy as np
from ..logging_conf import get_logger
 
logger = get_logger(__name__)


_PER_BOOK_RECIPE: dict[str, list[str]] = {
    "Bharater-Krishi-Babyasthar-Parichay": ["denoise", "clahe"],
    "Krishi-Bigyan": ["clahe", "unsharp"],
    "Krishi-Darpan": ["stroke_repair", "unsharp"],
}
_DEFAULT_RECIPE = ["denoise"]  # safe fallback for any doc_id not in the table above
 
 
def _prep_denoise(img: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(img, (3, 3), 0)

def _prep_clahe(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
 
 
def _prep_stroke_repair(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    repaired = 255 - closed  # back to black-text-on-white for OCR
    return cv2.cvtColor(repaired, cv2.COLOR_GRAY2BGR)
 
 
def _prep_unsharp(img: np.ndarray, radius: float = 1.5, strength: float = 0.8) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    return cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)


_STEP_FUNCS = {
    "denoise": _prep_denoise,
    "clahe": _prep_clahe,
    "stroke_repair": _prep_stroke_repair,
    "unsharp": _prep_unsharp,
}
 
 
class Enhancer:
    """Model set by cfg['enhance']. IMPLEMENT train() and apply()."""
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["enhance"]
    def train(self, pages: list[Page]) -> None:
        # raise NotImplementedError("Stage 1: train VAE/diffusion enhancer")
        logger.info("Stage 1: train enhancer (placeholder, no actual training implemented)")
    def apply(self, pages: list[Page]) -> list[Page]:
        # raise NotImplementedError("Stage 1: apply enhancer")
        enhanced_pages: list[Page] = []
 
        for page in pages:
            src_path = Path(page.image_path)
            steps = _PER_BOOK_RECIPE.get(page.doc_id, _DEFAULT_RECIPE)
 
            img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
            if img is None:
                logger.warning(f"Enhancer: could not load {src_path}, passing page through unchanged")
                enhanced_pages.append(page)
                continue
 
            for step_name in steps:
                img = _STEP_FUNCS[step_name](img)
 
            # Mirror data/pages/<book>/<file>.png -> data/enhanced/<book>/<file>.png
            parts = list(src_path.parts)
            try:
                pages_idx = parts.index("pages")
                parts[pages_idx] = "enhanced"
                out_path = Path(*parts)
            except ValueError:
                out_path = src_path.parent.parent / "enhanced" / src_path.parent.name / src_path.name
 
            # out_path.parent.mkdir(parents=True, exist_ok=True)
            # cv2.imwrite(str(out_path), img)
 
            enhanced_pages.append(
                Page(id=page.id, image_path=str(out_path), doc_id=page.doc_id)
            )
 
        logger.info(f"Enhancer.apply(): enhanced {len(enhanced_pages)} pages using per-book recipe")
        return enhanced_pages
 

def run(pages: list[Page], cfg: dict) -> list[Page]:
    if not cfg["enhance"]["enabled"]:
        return pages
    return Enhancer(cfg).apply(pages)

