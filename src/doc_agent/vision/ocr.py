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
import json  # add to imports

def _cfg_get(cfg, section, key, default):
    return cfg.get(section, {}).get(key, default)


def _ocr_region_cache_path(ocr_dir: Path, doc_id: str, page_stem: str) -> Path:
    return ocr_dir / "regions" / doc_id / f"{page_stem}.json"

def _save_ocr_regions(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

def _load_ocr_regions(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

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


def _tesseract_region(img_bgr: np.ndarray, lang: str, psm: int, oem: int = 1) -> str:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return pytesseract.image_to_string(rgb, lang=lang, config=f"--psm {psm} --oem {oem}").strip()


def _split_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# NEW: word-level dual-lang (ben/eng) resolution
# ---------------------------------------------------------------------------

_BANGLA_RANGE = (0x0980, 0x09FF)


def _word_boxes(img_bgr: np.ndarray, lang: str, psm: int, oem: int = 1) -> list[dict]:
    """Word-level boxes + text + confidence for one region crop."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(
        rgb, lang=lang, config=f"--psm {psm} --oem {oem}", output_type=pytesseract.Output.DICT
    )
    words = []
    for i, text in enumerate(data["text"]):
        if not text.strip():
            continue
        conf = float(data["conf"][i])
        if conf == -1:
            continue
        words.append({
            "text": text, "conf": conf,
            "left": data["left"][i], "top": data["top"][i],
            "width": data["width"][i], "height": data["height"][i],
            "line_num": data["line_num"][i], "word_num": data["word_num"][i],
        })
    return words

def _line_boxes(img_bgr: np.ndarray, lang: str, psm: int, oem: int = 1) -> list[dict]:
    """Group word boxes by line_num and compute each line's bbox + plain-OCR
    text, so dual-lang can be gated PER LINE -- a region can genuinely mix
    Bangla prose with one English citation line, and a single region-wide
    check can't serve both without either missing the citation or risking
    the Bangla lines."""
    words = _word_boxes(img_bgr, lang=lang, psm=psm, oem=oem)
    lines: dict[int, list[dict]] = {}
    for w in words:
        lines.setdefault(w["line_num"], []).append(w)

    out = []
    for line_num, ws in sorted(lines.items()):
        xs1 = [w["left"] for w in ws]
        ys1 = [w["top"] for w in ws]
        xs2 = [w["left"] + w["width"] for w in ws]
        ys2 = [w["top"] + w["height"] for w in ws]
        text = " ".join(w["text"] for w in sorted(ws, key=lambda w: w["word_num"]))
        out.append({"line_num": line_num, "bbox": (min(xs1), min(ys1), max(xs2), max(ys2)), "text": text})
    return out


def _resolve_mixed_script_line(
    img_bgr: np.ndarray, psm: int, oem: int = 1,
    per_word_conf_thr: float = 55.0,
    swap_margin: float = 15.0,
) -> str:
    """Word-level gate: trust ben's own confidence per word. Only pay for the
    expensive eng re-check on words ben is already unsure about, and only
    accept the eng reading if it's decisively better -- not marginally.
    Fixes English tokens being force-mapped into garbled Bangla glyphs
    (e.g. 'equivalent inches of mercury' -> random Bangla digit-glyphs)."""
    ben_words = _word_boxes(img_bgr, lang="ben", psm=psm, oem=oem)
    resolved: list[tuple[int, int, str]] = []

    for w in ben_words:
        if w["conf"] >= per_word_conf_thr:
            resolved.append((w["line_num"], w["word_num"], w["text"]))
            continue

        x, y, cw, ch = w["left"], w["top"], w["width"], w["height"]
        pad = 2
        crop = img_bgr[max(0, y - pad):y + ch + pad, max(0, x - pad):x + cw + pad]
        if crop.size == 0:
            resolved.append((w["line_num"], w["word_num"], w["text"]))
            continue

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        eng_txt = pytesseract.image_to_string(rgb, lang="eng", config="--psm 8").strip()
        eng_data = pytesseract.image_to_data(rgb, lang="eng", config="--psm 8", output_type=pytesseract.Output.DICT)
        eng_conf = max([c for c in eng_data["conf"] if c != -1], default=-1)

        if eng_txt and eng_conf > w["conf"] + swap_margin:
            resolved.append((w["line_num"], w["word_num"], eng_txt))
        else:
            resolved.append((w["line_num"], w["word_num"], w["text"]))

    lines: dict[int, list[tuple[int, str]]] = {}
    for line_num, word_num, txt in resolved:
        lines.setdefault(line_num, []).append((word_num, txt))
    out_lines = [" ".join(t for _, t in sorted(words)) for _, words in sorted(lines.items())]
    return "\n".join(out_lines)


# def _bangla_char_ratio(text: str) -> float:
#     """Fraction of alphabetic characters that are Bangla. Low ratio flags
#     citation lines / force-mapped garbage that the corrector shouldn't
#     'fix' -- it has no training signal for that content and tends to
#     hallucinate (e.g. repetition loops on chemistry-sounding tokens)."""
#     letters = [ch for ch in text if ch.isalpha()]
#     if not letters:
#         return 1.0  # pure punctuation/numbers -- don't block correction on this alone
#     bangla = sum(1 for ch in letters if _BANGLA_RANGE[0] <= ord(ch) <= _BANGLA_RANGE[1])
#     return bangla / len(letters)

# vision/ocr.py — replace _bangla_char_ratio (this is the one function used
# by BOTH the region-level correction gate and the line-level dual-lang gate)

_SAFE_PUNCT = set("।,.!?()[]—;:'\"-–০১২৩৪৫৬৭৮৯")  # Bangla digits are legitimate; ASCII digits are not

def _bangla_char_ratio(text: str) -> float:
    """Fraction of SIGNIFICANT characters (letters, digits, symbols -- not
    just alphabetic ones) that are Bangla. Counting only isalpha() chars was
    the bug: a line dominated by garbled Latin digit-glyphs but containing
    a couple of real Bangla words scored 1.0, since digits/symbols were
    invisible to the ratio either way -- exactly the citation/garbled-number
    content this gate exists to catch."""
    significant = [ch for ch in text if not ch.isspace() and ch not in _SAFE_PUNCT]
    if not significant:
        return 1.0
    bangla = sum(1 for ch in significant if _BANGLA_RANGE[0] <= ord(ch) <= _BANGLA_RANGE[1])
    return bangla / len(significant)

def _is_degenerate(text: str, max_word_repeat: int = 4) -> bool:
    """Catch seq2seq outputs stuck repeating the same short word run --
    a generation failure, not a real correction."""
    words = text.split()
    if len(words) < max_word_repeat:
        return False
    for i in range(len(words) - max_word_repeat + 1):
        window = words[i:i + max_word_repeat]
        if len(set(window)) == 1:
            return True
    return False


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
            gen_ids = self.model.generate(
                **inputs, max_length=self.max_length, num_beams=1,
                repetition_penalty=1.3,       # MODIFIED: penalize repeated tokens
                no_repeat_ngram_size=3,       # MODIFIED: hard-block repeated 3-grams
                early_stopping=True,          # MODIFIED
            )
            decoded = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            # MODIFIED: fall back to the pre-correction line if generation still degenerated
            decoded = [orig if _is_degenerate(corr) else corr for orig, corr in zip(batch, decoded)]
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
    lang = ocr_cfg.get("tesseract_lang", "ben")
    oem = int(ocr_cfg.get("tesseract_oem", 1))                                   # MODIFIED
    use_corrector = ocr_cfg.get("use_corrector", True)
    batch_size = int(ocr_cfg.get("corrector_batch_size", 32))
    use_dual_lang = ocr_cfg.get("dual_lang_word_level", True)                    # MODIFIED
    per_word_conf_thr = float(ocr_cfg.get("per_word_conf_thr", 55.0))            # MODIFIED
    swap_margin = float(ocr_cfg.get("swap_margin", 15.0))                        # MODIFIED
    min_bangla_ratio = float(ocr_cfg.get("min_bangla_ratio_for_correction", 0.6))# MODIFIED
    data_root = Path(_cfg_get(cfg, "ingest", "data_root", "data"))
    pages_dir = _cfg_get(cfg, "ingest", "pages_dir", "pages")
    ocr_dir = data_root / _cfg_get(cfg, "ingest", "ocr_dir", "interim/ocr")
    gt_root = data_root / ocr_cfg.get("gt_dir", "ground-truth-ocr")
    held_out_docs = set(ocr_cfg.get("held_out_docs", ["Krishi-Darpan"]))
    is_debug = bool(cfg.get("debug"))

    raw_metrics = _MetricAccumulator(held_out_docs)
    corrected_metrics = _MetricAccumulator(held_out_docs) if use_corrector else None

    all_input_regions = regions
    regions = sorted([r for r in regions if r.kind != "figure"],
                      key=lambda r: (r.page_id, r.bbox[1], r.bbox[0]))

    page_regions: dict[tuple[str, str], list[Region]] = {}
    for region in regions:
        doc_id, page_stem = _parse_page_id(region.page_id)
        page_regions.setdefault((doc_id, page_stem), []).append(region)

    # (page_key -> list of {"region", "raw_text", "corrected_text"}), sourced from
    # EITHER cache or fresh OCR below -- scoring and Chunk emission don't care which.
    page_entries: dict[tuple[str, str], list[dict]] = {}
    pages_to_process: list[tuple[str, str]] = []

    for key in page_regions:
        doc_id, page_stem = key
        cache_path = _ocr_region_cache_path(ocr_dir, doc_id, page_stem)
        if cache_path.exists():
            page_entries[key] = _load_ocr_regions(cache_path)
        else:
            pages_to_process.append(key)

    # Pass 1: Tesseract per region, only for uncached pages.
    region_raw_lines: dict[tuple[str, str], list[list[str]]] = {}
    flat_lines: list[str] = []
    flat_owner: list[tuple[tuple[str, str], int, int]] = []
    t_pass1 = time.time()
    for key in tqdm(pages_to_process, desc="tesseract"):
        doc_id, page_stem = key
        image_path = str(data_root / pages_dir / doc_id / f"{page_stem}.png")
        per_region_lines: list[list[str]] = []
        for region in page_regions[key]:
            psm = 7 if region.kind == "heading" else 6
            crop = _crop(image_path, region.bbox)
            # MODIFIED: line-level gate -- only run the risky word-level dual-lang
            # pass on lines that actually look mixed/garbled; ordinary Bangla
            # lines go straight through untouched.
            if use_dual_lang:
                lines_info = _line_boxes(crop, lang=lang, psm=psm, oem=oem)
                # print(f"[ocr] line-gate check on region {region.bbox}: "
                #     f"{[(round(_bangla_char_ratio(l['text']), 2), l['text'][:20]) for l in lines_info]}")
                
                resolved_lines = []
                line_gate_ratio = float(ocr_cfg.get("whole_bangla_ratio_thr", 0.85))         # NEW
                for line in lines_info:
                    if _bangla_char_ratio(line["text"]) < line_gate_ratio:
                        lx1, ly1, lx2, ly2 = line["bbox"]
                        pad = 3
                        line_crop = crop[max(0, ly1 - pad):ly2 + pad, max(0, lx1 - pad):lx2 + pad]
                        resolved = _resolve_mixed_script_line(
                            line_crop, psm=7, oem=oem,  # single-line crop -> psm 7
                            per_word_conf_thr=per_word_conf_thr, swap_margin=swap_margin,
                        ) if line_crop.size else line["text"]
                        resolved_lines.append(resolved)
                    else:
                        resolved_lines.append(line["text"])
                raw_text = "\n".join(resolved_lines)
            else:
                raw_text = _tesseract_region(crop, lang=lang, psm=psm, oem=oem)
            lines = [_normalize(ln) for ln in _split_lines(raw_text)]
            region_idx = len(per_region_lines)
            per_region_lines.append(lines)
            for j, ln in enumerate(lines):
                flat_lines.append(ln)
                flat_owner.append((key, region_idx, j))
        region_raw_lines[key] = per_region_lines
    print(f"[timing] tesseract pass: {len(pages_to_process)} pages, "
          f"{len(flat_lines)} lines in {time.time() - t_pass1:.1f}s")

    # print(f"raw lines to see for debug : {flat_lines}")

    # Pass 2: batched correction -- MODIFIED: only over lines that are
    # Bangla-heavy enough to be worth correcting. Citation lines / garbled
    # force-mapped content bypass the corrector and keep Pass 1's output.
    if use_corrector and flat_lines:
        correctable_idx = [i for i, ln in enumerate(flat_lines) if _bangla_char_ratio(ln) >= min_bangla_ratio]
        correctable_lines = [flat_lines[i] for i in correctable_idx]
        corrected_subset = _get_corrector(cfg).correct_batch(correctable_lines, batch_size=batch_size)
        corrected_flat = list(flat_lines)
        for i, corrected in zip(correctable_idx, corrected_subset):
            corrected_flat[i] = corrected
        print(f"[ocr] sent {len(correctable_lines)}/{len(flat_lines)} lines to corrector "
              f"(bangla_ratio >= {min_bangla_ratio}); rest kept as-is")
    else:
        corrected_flat = flat_lines

    region_corrected_lines = {k: [list(lines) for lines in v] for k, v in region_raw_lines.items()}
    for (key, region_idx, line_idx), corrected in zip(flat_owner, corrected_flat):
        region_corrected_lines[key][region_idx][line_idx] = corrected

    # print(f"[ocr] corrected lines to see for debug : {corrected_flat}")

    # Pass 2.5: build region entries for freshly-processed pages, write region cache.
    for key in pages_to_process:
        doc_id, page_stem = key
        entries = []
        for region_idx, region in enumerate(page_regions[key]):
            raw_text = "\n".join(region_raw_lines[key][region_idx])
            corrected_text = "\n".join(region_corrected_lines[key][region_idx]) if use_corrector else raw_text
            entries.append({
                "region": region.model_dump(),
                "raw_text": raw_text,
                "corrected_text": corrected_text,
            })
        page_entries[key] = entries
        if not is_debug:
            _save_ocr_regions(_ocr_region_cache_path(ocr_dir, doc_id, page_stem), entries)

    # Pass 3: page-level scoring -- concatenate each page's cached/fresh region
    # entries in list order (== spatial order, preserved by the cache) and diff
    # against GT text. Cache now carries both raw and corrected text per region,
    # so raw_metrics is scoreable on cache-hit pages too.
    for key, entries in page_entries.items():
        doc_id, page_stem = key
        gt_lines = _load_gt_lines(gt_root, doc_id, page_stem)
        if not gt_lines:
            continue
        page_gt_text = "\n".join(_normalize(g) for g in gt_lines)
        page_raw_text = "\n".join(e["raw_text"] for e in entries)
        page_corrected_text = "\n".join(e["corrected_text"] for e in entries)
        raw_metrics.add(doc_id, page_raw_text, page_gt_text)
        if corrected_metrics is not None:
            corrected_metrics.add(doc_id, page_corrected_text, page_gt_text)

    # Pass 4: one Chunk PER REGION, built uniformly from page_entries regardless
    # of cache vs. fresh-OCR source.
    chunks: list[Chunk] = []
    kind_counts: dict[str, int] = {}
    for entries in page_entries.values():
        for e in entries:
            region = Region.model_validate(e["region"])
            kind_counts[region.kind] = kind_counts.get(region.kind, 0) + 1
            chunks.append(Chunk(
                id=_region_chunk_id(region),
                doc_id=_parse_page_id(region.page_id)[0],
                text=e["corrected_text"] if use_corrector else e["raw_text"],
                page_ids=[region.page_id],
                score=0.0,
            ))

    skipped_figures = sum(1 for r in all_input_regions if r.kind == "figure")

    print(f"\n=== Region accounting ===")
    print(f"input regions by kind (chunked) : {kind_counts}")
    print(f"figures skipped (no text to extract) : {skipped_figures}")
    print(f"pages cached (skipped OCR) : {len(page_entries) - len(pages_to_process)}")
    print(f"pages processed this run   : {len(pages_to_process)}")
    print(f"chunks emitted : {len(chunks)}")

    raw_metrics.report("Tesseract (raw)")
    if corrected_metrics is not None:
        corrected_metrics.report("Tesseract + mT5 corrector")

    return chunks