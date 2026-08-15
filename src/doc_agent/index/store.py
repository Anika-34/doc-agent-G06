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

def _index_path(out_dir: Path) -> Path:
    return out_dir / "index.faiss"

def _meta_path(out_dir: Path) -> Path:
    return out_dir / "chunks.jsonl"

def _load_existing(out_dir: Path) -> tuple[faiss.Index | None, list[dict]]:
    """Load a previously-built index + its chunk metadata, if present.
    Returns (None, []) if nothing has been built yet."""
    index_path = _index_path(out_dir)
    meta_path = _meta_path(out_dir)
    if not (index_path.exists() and meta_path.exists()):
        return None, []

    index = faiss.read_index(str(index_path))
    meta: list[dict] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                meta.append(json.loads(line))
    return index, meta


def build(chunks: list[Chunk], vectors: np.ndarray, cfg: dict) -> None:
    out_dir = _index_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    if vectors.shape[0] != len(chunks):
        raise ValueError(
            f"vectors/chunks length mismatch: {vectors.shape[0]} vectors vs {len(chunks)} chunks -- "
            "these must stay aligned (vectors[i] corresponds to chunks[i])."
        )

    existing_index, existing_meta = _load_existing(out_dir)
    existing_ids = {row["id"] for row in existing_meta}

    # Split incoming chunks into "already indexed" vs "missing" by id.
    missing_positions = [i for i, c in enumerate(chunks) if c.id not in existing_ids]

    if existing_index is not None and not missing_positions:
        print(f"[store] cache hit: all {len(chunks)} chunks already indexed "
              f"({existing_index.ntotal} vectors in {out_dir}) -- nothing to build")
        return

    dim = vectors.shape[1]

    if existing_index is None:
        # Nothing cached yet -- build fresh, exactly as before.
        index = faiss.IndexFlatIP(dim)
        new_vectors = vectors
        new_chunks = chunks
        print(f"[store] no existing index found -- building fresh from {len(chunks)} chunks")
    else:
        if existing_index.d != dim:
            raise ValueError(
                f"cached index dim ({existing_index.d}) != incoming vector dim ({dim}) -- "
                "embedding model/config likely changed; delete the cached index to rebuild from scratch."
            )
        index = existing_index
        new_vectors = vectors[missing_positions]
        new_chunks = [chunks[i] for i in missing_positions]
        print(f"[store] cache partial hit: {len(existing_ids)} chunks already indexed, "
              f"adding {len(new_chunks)} missing chunk(s)")

    if len(new_chunks) > 0:
        index.add(new_vectors)

    faiss.write_index(index, str(_index_path(out_dir)))

    if len(new_chunks) > 0:
        with open(_meta_path(out_dir), "a", encoding="utf-8") as f:
            for c in new_chunks:
                f.write(json.dumps(c.model_dump(), ensure_ascii=False) + "\n")

    print(f"[store] index now has {index.ntotal} vectors "
          f"({len(new_chunks)} newly added) + matching chunk records in {out_dir}")