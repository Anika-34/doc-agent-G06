from __future__ import annotations
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
"""Stage 3 — OCR/HTR (BASELINE = pretrained foundation, fine-tuned)"""

import re
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import pytesseract
import torch
from jiwer import cer as _cer, wer as _wer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

from ..contracts import *  # noqa


def _cfg_get(cfg, section, key, default):
    return cfg.get(section, {}).get(key, default)


# ---------------------------------------------------------------------------
# Tesseract reading
# ---------------------------------------------------------------------------

def _crop(image_path: str, bbox: tuple[int, int, int, int]) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"OpenCV could not read {image_path}")
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return img[y1:y2, x1:x2]


def _tesseract_region(img_bgr: np.ndarray, lang: str, psm: int) -> str:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return pytesseract.image_to_string(rgb, lang=lang, config=f"--psm {psm}").strip()


def _split_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# mT5 corrector -- lazy singleton, mirrors vision/layout.py's _get_model()
# ---------------------------------------------------------------------------

_CORRECTOR = None


class _Corrector:
    def __init__(self, checkpoint_path: str, max_length: int, num_threads: int | None):
        if num_threads:
            torch.set_num_threads(num_threads)
        start = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path, torch_dtype=torch.float32)
        self.model.to("cpu")
        self.model.eval()
        print(f"[timing] mT5 corrector loaded in {time.time() - start:.1f}s "
              f"(one-time cost -- excluded from per-line rate below)")
        self.max_length = max_length
        self.prefix = "ocr_fix: "

    @torch.no_grad()
    def correct_batch(self, lines: list[str], batch_size: int = 32) -> list[str]:
        if not lines:
            return []
        start = time.time()
        out: list[str] = []
        n_batches = (len(lines) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(lines), batch_size), total=n_batches, desc="mt5 correct"):
            batch = lines[i:i + batch_size]
            inputs = self.tokenizer(
                [self.prefix + ln for ln in batch],
                return_tensors="pt", truncation=True, padding=True,
                max_length=self.max_length,
            )
            gen_ids = self.model.generate(**inputs, max_length=self.max_length, num_beams=1)
            decoded = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            out.extend(decoded)
        elapsed = time.time() - start
        rate = len(lines) / elapsed if elapsed > 0 else float("inf")
        print(f"[timing] correction: {len(lines)} lines in {elapsed:.1f}s ({rate:.1f} lines/sec)")
        return out


def _get_corrector(cfg: dict) -> "_Corrector":
    global _CORRECTOR
    if _CORRECTOR is None:
        checkpoint_path = cfg["ocr"]["corrector_checkpoint_path"]
        max_length = int(_cfg_get(cfg, "ocr", "corrector_max_length", 96))
        num_threads = _cfg_get(cfg, "ocr", "corrector_cpu_threads", None)
        _CORRECTOR = _Corrector(checkpoint_path, max_length, num_threads)
    return _CORRECTOR


# ---------------------------------------------------------------------------
# Ground-truth lookup -- used ONLY as the reference string for scoring, never
# to decide correspondence between predicted and reference text
# ---------------------------------------------------------------------------

def _parse_page_id(page_id: str) -> tuple[str, str]:
    """loader.py builds Page.id as '<doc_id>:<page_stem>' -- split it back
    apart to locate the source image and the matching GT file."""
    doc_id, _, page_stem = page_id.partition(":")
    return doc_id, page_stem


def _load_gt_lines(gt_root: Path, doc_id: str, page_stem: str) -> list[str] | None:
    gt_path = gt_root / doc_id / f"{page_stem}.txt"
    if not gt_path.exists():
        return None
    with open(gt_path, "r", encoding="utf-8") as f:
        return _split_lines(f.read())


def _region_chunk_id(region: Region) -> str:
    """Deterministic id from page_id + bbox -- stable across re-runs on the
    same layout output, since Region carries no id of its own."""
    x1, y1, x2, y2 = region.bbox
    return f"{region.page_id}:{x1}-{y1}-{x2}-{y2}"


class _MetricAccumulator:
    """Accumulates one (hypothesis, reference) pair PER PAGE -- not per line
    and not via any GT-informed matching. Each pair is: the system's full
    output for that page (regions concatenated in bbox reading order) vs.
    that page's full GT text. jiwer's own edit-distance computation is the
    only alignment performed, and it never sees the other side's identity
    while deciding correspondence -- it just diffs two fixed strings."""
    def __init__(self, held_out_docs: set[str]):
        self.held_out_docs = held_out_docs
        self.all_hyp, self.all_ref = [], []
        self.held_hyp, self.held_ref = [], []

    def add(self, doc_id: str, hyp: str, ref: str):
        if not ref.strip():
            return
        self.all_hyp.append(hyp); self.all_ref.append(ref)
        if doc_id in self.held_out_docs:
            self.held_hyp.append(hyp); self.held_ref.append(ref)

    def report(self, label: str):
        def _scores(hyps, refs):
            return (_cer(refs, hyps), _wer(refs, hyps)) if refs else None
        corpus, held = _scores(self.all_hyp, self.all_ref), _scores(self.held_hyp, self.held_ref)
        print(f"\n=== {label} ===")
        print(f"corpus-wide : cer={corpus[0]:.4f}  wer={corpus[1]:.4f}  (n={len(self.all_ref)} pages)"
              if corpus else "corpus-wide : no ground truth found -- skipped")
        print(f"held-out    : cer={held[0]:.4f}  wer={held[1]:.4f}  (n={len(self.held_ref)} pages)"
              if held else "held-out    : no ground truth found -- skipped")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

class Reader:
    """Kept for interface compatibility -- transcribe() below does the real work."""
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["ocr"]

    def transcribe_region(self, region: Region) -> str:
        raise NotImplementedError("use transcribe()")


def transcribe(regions: list[Region], cfg: dict) -> list[Chunk]:
    ocr_cfg = cfg.get("ocr", {})
    lang = ocr_cfg.get("tesseract_lang", "ben")  # ben+eng regressed corpus-wide CER 0.16->0.25 -- stay ben-only
    use_corrector = ocr_cfg.get("use_corrector", True)
    batch_size = int(ocr_cfg.get("corrector_batch_size", 32))
    data_root = Path(_cfg_get(cfg, "ingest", "data_root", "data"))
    pages_dir = _cfg_get(cfg, "ingest", "pages_dir", "pages")
    gt_root = data_root / ocr_cfg.get("gt_dir", "ground-truth-ocr")
    held_out_docs = set(ocr_cfg.get("held_out_docs", ["Krishi-Darpan"]))

    raw_metrics = _MetricAccumulator(held_out_docs)
    corrected_metrics = _MetricAccumulator(held_out_docs) if use_corrector else None

    # Pass 1: Tesseract per region, split into lines, flatten across the
    # WHOLE corpus so correction (pass 2) runs as large batches instead of
    # tiny per-region calls -- this is the main CPU-throughput lever.
    # Regions are ordered by SPATIAL POSITION ONLY (page, then top-to-bottom,
    # then left-to-right) -- this ordering is what both the emitted Chunks
    # and the scoring below use as "reading order". No ground truth is
    # consulted anywhere in this pass or in how regions are arranged.
    region_lines: list[list[str]] = []
    region_meta: list[tuple[Region, str, str]] = []
    flat_lines: list[str] = []
    flat_owner: list[tuple[int, int]] = []
    t_pass1 = time.time()
    regions = sorted(regions, key=lambda r: (r.page_id, r.bbox[1], r.bbox[0]))
    for region in tqdm(regions, desc="tesseract"):
        if region.kind == "figure":
            continue
        doc_id, page_stem = _parse_page_id(region.page_id)
        image_path = str(data_root / pages_dir / doc_id / f"{page_stem}.png")
        psm = 7 if region.kind == "heading" else 6
        crop = _crop(image_path, region.bbox)
        raw_text = _tesseract_region(crop, lang=lang, psm=psm)
        lines = [_normalize(ln) for ln in _split_lines(raw_text)]

        region_idx = len(region_lines)
        region_lines.append(lines)
        region_meta.append((region, doc_id, page_stem))
        for j, ln in enumerate(lines):
            flat_lines.append(ln)
            flat_owner.append((region_idx, j))

    print(f"[timing] tesseract pass: {len(region_meta)} regions, "
          f"{len(flat_lines)} lines in {time.time() - t_pass1:.1f}s")

    # Pass 2: one batched correction sweep over every line in the corpus.
    if use_corrector and flat_lines:
        corrected_flat = _get_corrector(cfg).correct_batch(flat_lines, batch_size=batch_size)
    else:
        corrected_flat = flat_lines

    corrected_region_lines = [list(lines) for lines in region_lines]
    for (region_idx, line_idx), corrected in zip(flat_owner, corrected_flat):
        corrected_region_lines[region_idx][line_idx] = corrected

    # Pass 2.5: page-level scoring. For each page, concatenate every
    # region's lines IN THE SPATIAL READING ORDER established above into
    # one block of text -- this is exactly what a real downstream consumer
    # of this pipeline would see, with no knowledge of the GT. That block
    # is compared directly against the page's full GT text via jiwer's
    # own edit-distance CER/WER. No custom alignment step, no GT-informed
    # line matching -- the comparison never uses GT content to decide what
    # corresponds to what, only to measure how far off the final result is.
    pages_order: list[tuple[str, str]] = []
    page_region_idxs: dict[tuple[str, str], list[int]] = {}
    for idx, (region, doc_id, page_stem) in enumerate(region_meta):
        key = (doc_id, page_stem)
        if key not in page_region_idxs:
            page_region_idxs[key] = []
            pages_order.append(key)
        page_region_idxs[key].append(idx)

    for doc_id, page_stem in pages_order:
        idxs = page_region_idxs[(doc_id, page_stem)]
        page_raw_text = "\n".join(ln for i in idxs for ln in region_lines[i])
        page_corrected_text = "\n".join(ln for i in idxs for ln in corrected_region_lines[i])

        gt_lines = _load_gt_lines(gt_root, doc_id, page_stem)
        if not gt_lines:
            continue
        page_gt_text = "\n".join(_normalize(g) for g in gt_lines)

        raw_metrics.add(doc_id, page_raw_text, page_gt_text)
        if corrected_metrics is not None:
            corrected_metrics.add(doc_id, page_corrected_text, page_gt_text)

    # Pass 3: build chunks. Same spatial reading order as above.
    chunks: list[Chunk] = []
    kind_counts: dict[str, int] = {}
    skipped_figures = 0

    for region in regions:
        kind_counts[region.kind] = kind_counts.get(region.kind, 0) + 1
        if region.kind == "figure":
            skipped_figures += 1

    for region_idx, (region, doc_id, page_stem) in enumerate(region_meta):
        lines = corrected_region_lines[region_idx]
        chunks.append(Chunk(
            id=_region_chunk_id(region),
            doc_id=doc_id,
            text="\n".join(lines),
            page_ids=[region.page_id],
            score=0.0,
        ))

    print(f"\n=== Region accounting ===")
    print(f"input regions by kind : {kind_counts}")
    print(f"figures skipped (no text to extract) : {skipped_figures}")
    print(f"chunks emitted : {len(chunks)}  "
        f"(expected = input regions - figures = {sum(kind_counts.values()) - skipped_figures})")
    if len(chunks) != sum(kind_counts.values()) - skipped_figures:
        print("WARNING: chunk count doesn't match expected -- a region was dropped somewhere above")

    raw_metrics.report("Tesseract (raw)")
    if corrected_metrics is not None:
        corrected_metrics.report("Tesseract + mT5 corrector")

    return chunks