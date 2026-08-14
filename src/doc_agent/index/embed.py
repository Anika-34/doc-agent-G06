"""Stage 4 — embed chunks"""
from __future__ import annotations
from ..contracts import *  # noqa
import numpy as np
from sentence_transformers import SentenceTransformer


_MODEL = None

def _get_model(device: str) -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("BAAI/bge-m3", device=device)
    return _MODEL

def encode(chunks: list[Chunk], cfg: dict) -> np.ndarray:
    embed_cfg = cfg.get("embed", {})
    device = cfg.get("device", "cpu")
    batch_size = int(embed_cfg.get("batch_size", 16))

    model = _get_model(device)
    texts = [c.text for c in chunks]

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,   # BGE models are trained for cosine sim on normalized vectors
        show_progress_bar=True,
    )
    return np.asarray(vectors, dtype=np.float32)
