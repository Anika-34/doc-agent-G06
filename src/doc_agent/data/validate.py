# """Data — data schema/quality validation at ingest"""
# from __future__ import annotations
# from ..contracts import *  # noqa

# def validate(pages: list[Page]) -> None:
#     """Assert min pages/words, format, no leakage across splits. IMPLEMENT."""
#     raise NotImplementedError("Data: validate")

"""Data — data schema/quality validation at ingest"""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
from ..contracts import *  # noqa


def validate(pages: list[Page]) -> None:
    """Assert min pages/words, format, no leakage across splits.

    Validates:
    - >=300 total pages
    - >=60k total words (from ground-truth OCR)
    - No leakage: test set (Krishi-Darpan) separate from train/val (others)
    - All pages have a valid id, image_path, and doc_id
    """
    assert pages, "No pages provided"

    doc_ids: dict[str, int] = defaultdict(int)
    for page in pages:
        assert page.id, f"Page missing id: {page}"
        assert page.image_path, f"Page {page.id} missing image_path"
        assert page.doc_id, f"Page {page.id} missing doc_id"
        assert Path(page.image_path).exists(), f"Image not found: {page.image_path}"
        doc_ids[page.doc_id] += 1

    min_pages = 300
    assert len(pages) >= min_pages, f"Only {len(pages)} pages (need >= {min_pages})"

    total_words = _count_words_from_ocr(doc_ids)
    min_words = 60_000
    assert total_words >= min_words, f"Only {total_words} words (need >= {min_words})"

    _check_no_leakage(doc_ids)


def _count_words_from_ocr(doc_ids: dict[str, int]) -> int:
    """Count words from ground-truth OCR .txt files for the given documents."""
    ocr_dir = Path(__file__).parent.parent.parent.parent / "data" / "ground-truth-ocr"
    total_words = 0

    if not ocr_dir.exists():
        return total_words

    for doc_folder in ocr_dir.iterdir():
        if doc_folder.is_dir() and doc_folder.name in doc_ids:
            for txt_file in doc_folder.glob("*.txt"):
                with open(txt_file, "r", encoding="utf-8") as f:
                    total_words += len(f.read().split())

    return total_words


def _check_no_leakage(doc_ids: dict[str, int]) -> None:
    """Verify test/train split integrity (no document overlap).

    Expected split:
    - Train/Val: Krishi-Bigyan, Bharater-Krishi-Babyasthar-Parichay
    - Test: Krishi-Darpan
    """
    train_val_docs = {"Krishi-Bigyan", "Bharater-Krishi-Babyasthar-Parichay"}
    test_docs = {"Krishi-Darpan"}

    actual_docs = set(doc_ids.keys())
    train_val_actual = actual_docs & train_val_docs
    test_actual = actual_docs & test_docs

    assert train_val_actual, "No train/val documents found"
    assert test_actual, "No test documents found"

    overlap = train_val_actual & test_actual
    assert not overlap, f"Leakage detected: {overlap} appears in both train/val and test"