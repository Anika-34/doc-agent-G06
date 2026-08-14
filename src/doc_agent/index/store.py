"""Stage 4 — vector store"""
from __future__ import annotations
from ..contracts import *  # noqa
import json
from pathlib import Path
import numpy as np
import faiss

def _index_dir(cfg: dict) -> Path:
    data_root = Path(cfg.get("ingest", {}).get("data_root", "data"))
    index_cfg = cfg.get("index", {})
    return data_root / index_cfg.get("index_dir", "interim/index")

def build(chunks: list[Chunk], vectors: np.ndarray, cfg: dict) -> None:
    out_dir = _index_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product — matches normalize_embeddings=True in embed.py
    index.add(vectors)

    faiss.write_index(index, str(out_dir / "index.faiss"))

    # persist chunk metadata in the SAME order vectors were added, so a
    # retrieved vector's row index maps directly back to its Chunk
    meta = [c.model_dump() for c in chunks]
    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for row in meta:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[store] wrote {index.ntotal} vectors (dim={dim}) + {len(meta)} chunk records to {out_dir}")