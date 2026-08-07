"""Stage 1 — load scanned page-images"""
from __future__ import annotations
from ..contracts import *  # noqa
from pathlib import Path
from pdf2image import convert_from_path



IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _cfg_get(cfg: dict, key: str, default):
    return cfg.get("ingest", {}).get(key, default)


def _render_pdfs_to_pages(raw_root: Path, pages_root: Path, dpi: int) -> None:
    pages_root.mkdir(parents=True, exist_ok=True)
    for pdf_path in sorted(raw_root.glob("*.pdf")):
        doc_id = pdf_path.stem
        out_dir = pages_root / doc_id
        out_dir.mkdir(parents=True, exist_ok=True)

        if any(out_dir.glob("*.png")):
            continue

        images = convert_from_path(str(pdf_path), dpi=dpi)
        for i, img in enumerate(images, start=1):
            out_path = out_dir / f"page-{i:03d}.png"
            if not out_path.exists():
                img.save(out_path, "PNG")


def _iter_pages(pages_root: Path) -> list[Page]:
    pages: list[Page] = []
    if not pages_root.exists():
        return pages
    for doc_dir in sorted(p for p in pages_root.iterdir() if p.is_dir()):
        for image_path in sorted(p for p in doc_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
            page_id = f"{doc_dir.name}:{image_path.stem}"
            pages.append(Page(id=page_id, image_path=str(image_path), doc_id=doc_dir.name))
    return pages

def load_pages(cfg: dict) -> list[Page]:
    """Read PDFs from data/raw, render to data/pages, then return page-image records."""
    data_root = Path(_cfg_get(cfg, "data_root", "data"))
    raw_root = data_root / _cfg_get(cfg, "raw_dir", "raw")
    pages_root = data_root / _cfg_get(cfg, "pages_dir", "pages")
    dpi = int(_cfg_get(cfg, "render_dpi", 150))

    _render_pdfs_to_pages(raw_root, pages_root, dpi)
    pages = _iter_pages(pages_root)
    if not pages:
        raise ValueError(f"No page images found under {pages_root}. Ensure PDFs exist in {raw_root}.")
    return pages
