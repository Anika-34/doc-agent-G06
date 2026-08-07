"""Stage 1 — deskew / denoise / binarize / augment"""
from __future__ import annotations
from ..contracts import *  # noqa
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _cfg_get(cfg: dict, key: str, default):
    return cfg.get("ingest", {}).get(key, default)


def _page_stats(image_path: Path, min_blob_area: int) -> dict:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"OpenCV could not read {image_path}")
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    _num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    real_blobs = areas[areas >= min_blob_area]
    total_pixels = img.shape[0] * img.shape[1]
    ink_ratio = float(real_blobs.sum()) / total_pixels if total_pixels else 0.0
    return {"blob_count": int(len(real_blobs)), "ink_ratio": ink_ratio}


def _is_blank(stats: dict, min_blob_count: int, ink_ratio_threshold: float) -> bool:
    return stats["blob_count"] < min_blob_count or stats["ink_ratio"] < ink_ratio_threshold

def run(pages: list[Page], cfg: dict) -> list[Page]:
    """Filter likely-blank page-images; splitting/rendering is handled by loader.load_pages()."""
    data_root = Path(_cfg_get(cfg, "data_root", "data"))
    blank_root = data_root / _cfg_get(cfg, "blank_dir", "blank")

    min_blob_area = int(_cfg_get(cfg, "min_blob_area", 15))
    ink_ratio_threshold = float(_cfg_get(cfg, "ink_ratio_threshold", 0.002))
    min_blob_count = int(_cfg_get(cfg, "min_blob_count", 3))

    blank_root.mkdir(parents=True, exist_ok=True)

    processed: list[Page] = []
    for page in pages:
        doc_id = page.doc_id
        image_path = Path(page.image_path)
        stats = _page_stats(image_path, min_blob_area=min_blob_area)
        if _is_blank(stats, min_blob_count=min_blob_count, ink_ratio_threshold=ink_ratio_threshold):
            dest_dir = blank_root / doc_id
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / image_path.name
            image_path.replace(dest_path)
            continue

        processed.append(page)

    return processed
