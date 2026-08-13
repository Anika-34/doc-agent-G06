"""FIXED end-to-end order (Stages 0-9) + cross-cutting seams.
Do not reorder stages or remove hooks.run()/register_all() calls."""
from __future__ import annotations
import time
from . import config, hooks, wiring  # noqa: F401
from .ingest import loader, preprocess, enhance
from .vision import layout, ocr
from .index import chunk, embed, store
from .retrieval import retriever
from .agent import agent


def _apply_debug_limits(pages: list[Page], cfg: dict) -> list[Page]:
    """Dev/debug speed knob -- caps how many pages flow into layout + OCR
    so you can smoke-test the pipeline without waiting on the full corpus.
    No-op unless cfg['debug']['max_pages_per_doc'] is set. Optionally
    restrict to specific books via cfg['debug']['doc_ids'] (useful for
    testing specifically on the held-out book, e.g. Krishi-Darpan)."""
    debug_cfg = cfg.get("debug", {})
    max_per_doc = debug_cfg.get("max_pages_per_doc")
    doc_ids = debug_cfg.get("doc_ids")

    if doc_ids:
        pages = [p for p in pages if p.doc_id in doc_ids]

    if not max_per_doc:
        if doc_ids:
            print(f"[debug] doc_ids={doc_ids} active -- {len(pages)} pages")
        return pages

    by_doc: dict[str, list[Page]] = {}
    for p in pages:
        by_doc.setdefault(p.doc_id, []).append(p)
    limited = [p for doc_pages in by_doc.values() for p in doc_pages[:max_per_doc]]
    print(f"[debug] max_pages_per_doc={max_per_doc} doc_ids={doc_ids or 'all'} "
          f"-- using {len(limited)}/{len(pages)} pages")
    return limited

def _timed(label: str, fn, *args, **kwargs):
    start = time.time()
    result = fn(*args, **kwargs)
    print(f"[timing] {label}: {time.time() - start:.1f}s")
    return result


def build_knowledge_base(cfg: dict) -> None:
    wiring.register_all(cfg)                        # wire cross-cutting features
    pages = _timed("loader.load_pages", loader.load_pages, cfg)
    pages = _timed("preprocess.run", preprocess.run, pages, cfg)
    pages = _apply_debug_limits(pages, cfg)          # <-- debug cap, applied after blank-page filtering
    pages = _timed("enhance.run", enhance.run, pages, cfg)
    hooks.run(hooks.AFTER_INGEST, {"pages": pages})
    regions = _timed("layout.detect", layout.detect, pages, cfg)             # Stage 2
    text = _timed("ocr.transcribe", ocr.transcribe, regions, cfg)             # Stage 3
    hooks.run(hooks.AFTER_OCR, {"chunks": text})    # e.g. PII redaction on extracted text
    chunks = _timed("chunk.split", chunk.split, text, cfg)                 # Stage 4
    hooks.run(hooks.BEFORE_INDEX, {"chunks": chunks})
    vectors = _timed("embed.encode", embed.encode, chunks, cfg)
    store.build(chunks, vectors, cfg)


def answer(query_text: str, cfg: dict):
    wiring.register_all(cfg)
    r = retriever.Retriever(cfg)                    # Stage 5
    return agent.Agent(cfg, r).run(query_text)      # Stage 6 (seams run inside the loop)
