# """Data — corpus versioning (which corpus version -> which result)"""
# from __future__ import annotations
# from ..contracts import *  # noqa

# def snapshot(corpus_dir: str) -> str:
#     """Hash + record a corpus version id. IMPLEMENT (or wire DVC)."""
#     raise NotImplementedError("Data: version snapshot")

"""Data — corpus versioning (which corpus version -> which result)"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
from datetime import datetime
from ..contracts import *  # noqa


def snapshot(corpus_dir: str) -> str:
    """Hash + record a corpus version id for reproducibility.

    Computes a deterministic version id from a SHA256 hash of every file
    under corpus_dir, plus the timestamp the snapshot was taken. Writes
    the metadata to .corpus_version at the project root.

    Returns:
        Version id string, format "v-<hash-prefix>-<timestamp>"
    """
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    corpus_hash = _compute_directory_hash(corpus_path)
    timestamp = datetime.now().isoformat(timespec="seconds")
    version_id = f"v-{corpus_hash[:16]}-{timestamp.replace(':', '')}"

    snapshot_file = Path(__file__).parent.parent.parent.parent / ".corpus_version"
    metadata = {
        "version_id": version_id,
        "corpus_dir": str(corpus_path),
        "hash": corpus_hash,
        "timestamp": timestamp,
    }
    with open(snapshot_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return version_id


def _compute_directory_hash(directory: Path, algorithm: str = "sha256") -> str:
    """Compute a deterministic hash of every file in directory (sorted, so
    the result is stable across runs regardless of filesystem ordering)."""
    hash_func = hashlib.new(algorithm)
    ignore_names = {".DS_Store", ".gitkeep", ".corpus_version"}

    for filepath in sorted(directory.rglob("*")):
        if not filepath.is_file():
            continue
        if filepath.name in ignore_names or filepath.name.startswith("."):
            continue
        with open(filepath, "rb") as f:
            file_hash = hashlib.new(algorithm, f.read()).digest()
        hash_func.update(file_hash)

    return hash_func.hexdigest()