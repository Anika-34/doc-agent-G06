"""Stage 4 — chunk text"""
from __future__ import annotations

import json
import re
from pathlib import Path

from transformers import AutoTokenizer

from ..contracts import *  # noqa

_TOKENIZER = None

def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    return _TOKENIZER

def _tok_len(text: str) -> int:
    return len(_get_tokenizer().encode(text, add_special_tokens=False))

def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=।)\s*", text)
    return [p.strip() for p in parts if p.strip()]

def _cfg_get(cfg, section, key, default):
    return cfg.get(section, {}).get(key, default)

def _parse_page_id(page_id: str) -> tuple[str, str]:
    doc_id, _, page_stem = page_id.partition(":")
    return doc_id, page_stem

def _chunk_cache_path(chunk_dir: Path, doc_id: str, page_stem: str) -> Path:
    return chunk_dir / doc_id / f"{page_stem}.json"

def _save_chunks(path: Path, chunks: list[Chunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in chunks], f, ensure_ascii=False, indent=2)

def _load_chunks(path: Path) -> list[Chunk]:
    with open(path, "r", encoding="utf-8") as f:
        return [Chunk.model_validate(c) for c in json.load(f)]

def _load_ocr_regions(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# Merge: greedily pack consecutive same-page regions (already in spatial
# reading order in the OCR cache) up to `target_tokens`. Tables are kept
# standalone -- structurally different from prose, shouldn't get blurred
# into a surrounding paragraph chunk even if small.
# ---------------------------------------------------------------------------

def _merge_page_regions(entries: list[dict], target_tokens: int) -> list[list[dict]]:
    groups: list[list[dict]] = []
    cur: list[dict] = []
    cur_len = 0
    for e in entries:
        if e["region"]["kind"] == "table":
            if cur:
                groups.append(cur)
                cur, cur_len = [], 0
            groups.append([e])
            continue
        n = _tok_len(e["corrected_text"])
        if cur and cur_len + n > target_tokens:
            groups.append(cur)
            cur, cur_len = [], 0
        cur.append(e)
        cur_len += n
    if cur:
        groups.append(cur)
    return groups

# ---------------------------------------------------------------------------
# Split: only reached when a merged group exceeds the hard cap (rare -- 0.6%
# of raw regions exceed 512 tokens, and merging is capped at target_tokens
# so this mainly fires on single oversized regions). Pack sentences up to
# max_tokens, carrying trailing sentences into the next piece as overlap so
# no sentence is ever boundary-orphaned.
# ---------------------------------------------------------------------------

def _split_long_text(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    sentences = _split_sentences(text) or [text]
    pieces: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for s in sentences:
        s_len = _tok_len(s)
        if cur and cur_len + s_len > max_tokens:
            pieces.append(" ".join(cur))
            overlap_sents: list[str] = []
            overlap_len = 0
            for prev in reversed(cur):
                prev_len = _tok_len(prev)
                if overlap_sents and overlap_len + prev_len > overlap_tokens:
                    break
                overlap_sents.insert(0, prev)
                overlap_len += prev_len
            cur, cur_len = list(overlap_sents), overlap_len
        cur.append(s)
        cur_len += s_len
    if cur:
        pieces.append(" ".join(cur))
    return pieces

def split(chunks: list[Chunk], cfg: dict) -> list[Chunk]:
    index_cfg = cfg.get("index", {})
    target_tokens = int(index_cfg.get("chunk_tokens", 256))
    max_tokens = int(index_cfg.get("chunk_max_tokens", 512))
    overlap_tokens = int(index_cfg.get("overlap", 32))
    data_root = Path(_cfg_get(cfg, "ingest", "data_root", "data"))
    ocr_dir = data_root / _cfg_get(cfg, "ingest", "ocr_dir", "interim/ocr")
    chunk_dir = data_root / _cfg_get(cfg, "ingest", "chunk_dir", "interim/chunks")
    is_debug = bool(cfg.get("debug"))

    page_keys = sorted({_parse_page_id(pid) for c in chunks for pid in c.page_ids})

    out: list[Chunk] = []
    for doc_id, page_stem in page_keys:
        cache_path = _chunk_cache_path(chunk_dir, doc_id, page_stem)
        if cache_path.exists():
            out.extend(_load_chunks(cache_path))
            continue

        region_cache_path = ocr_dir / doc_id / f"{page_stem}.json"
        entries = sorted(
            (e for e in _load_ocr_regions(region_cache_path) if e["region"]["kind"] != "heading"),
            key=lambda e: (e["region"]["bbox"][1], e["region"]["bbox"][0]),
        )

        page_chunks: list[Chunk] = []
        for group_idx, group in enumerate(_merge_page_regions(entries, target_tokens)):
            text = "\n".join(e["corrected_text"] for e in group)
            page_id = group[0]["region"]["page_id"]
            if _tok_len(text) > max_tokens:
                pieces = _split_long_text(text, max_tokens, overlap_tokens)
            else:
                pieces = [text]
            for piece_idx, piece in enumerate(pieces):
                page_chunks.append(Chunk(
                    id=f"{doc_id}:{page_stem}:{group_idx}-{piece_idx}",
                    doc_id=doc_id,
                    text=piece,
                    page_ids=[page_id],
                    score=0.0,
                ))

        if not is_debug:
            _save_chunks(cache_path, page_chunks)
        out.extend(page_chunks)

    print(f"[chunk] {len(page_keys)} pages -> {len(out)} chunks "
          f"(target={target_tokens} cap={max_tokens} overlap={overlap_tokens})")
    return out