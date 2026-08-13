# """Stage 2 — layout detection / segmentation"""
# from __future__ import annotations
# from ..contracts import *  # noqa

# def detect(pages: list[Page], cfg: dict) -> list[Region]:
#     """Detect text/table/figure/heading regions. IMPLEMENT."""
#     raise NotImplementedError("Stage 2: layout detection")

# vision/layout.py
from __future__ import annotations
import cv2
import numpy as np
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from ..contracts import *  # noqa
import os

_COLOR_MAP = {
    "text": (0, 255, 0),       # Green
    "figure": (0, 0, 255),     # Red
    "table": (255, 0, 0),      # Blue
    "heading": (0, 255, 255),  # Yellow
}

def visualize_regions(pages: list[Page], regions: list[Region], output_dir: str = "debug_layout") -> None:
    """Saves page images with color-coded bounding boxes and labels for debugging."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group detected regions by page_id
    regions_by_page: dict[str, list[Region]] = {}
    for r in regions:
        regions_by_page.setdefault(r.page_id, []).append(r)

    for page in pages:
        img = cv2.imread(page.image_path)
        if img is None:
            continue
            
        page_regions = regions_by_page.get(page.id, [])
        for r in page_regions:
            x1, y1, x2, y2 = r.bbox
            color = _COLOR_MAP.get(r.kind, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)
            
            # Draw region label above box
            cv2.putText(
                img, 
                r.kind, 
                (x1, max(y1 - 5, 15)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                color, 
                2, 
                cv2.LINE_AA
            )

        out_path = os.path.join(output_dir, f"debug_{page.id}.png")
        cv2.imwrite(out_path, img)


# DocStructBench's 10 fine-grained classes, collapsed onto your 4-way Region.kind
_ID_TO_NAME = {
    0: "title", 1: "plain text", 2: "abandon", 3: "figure", 4: "figure_caption",
    5: "table", 6: "table_caption", 7: "table_footnote", 8: "isolate_formula", 9: "formula_caption",
}
_NAME_TO_KIND = {
    "title": "heading",
    "plain text": "text",
    "abandon": "text",          # headers/footers/page numbers/marginalia -> still text, just low-value
    "figure": "figure",
    "figure_caption": "text",   # captions are readable prose, not the image itself
    "table": "table",
    "table_caption": "text",
    "table_footnote": "text",
    "isolate_formula": "figure",  # not real text for a Bangla-OCR reader; treat like a figure
    "formula_caption": "text",
}

_MODEL = None

def _get_model():
    global _MODEL
    if _MODEL is None:
        weights_path = hf_hub_download(
            repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
            filename="doclayout_yolo_docstructbench_imgsz1024.pt",
        )
        _MODEL = YOLOv10(weights_path)
    return _MODEL

def detect(pages: list[Page], cfg: dict) -> list[Region]:
    model = _get_model()
    conf_thr = cfg["layout"].get("score_thr", 0.25)
    device = cfg.get("device", "cpu")

    regions: list[Region] = []
    for page in pages:
        det = model.predict(page.image_path, imgsz=1024, conf=conf_thr, device=device)[0]
        boxes = det.boxes.xyxy.cpu().numpy()
        classes = det.boxes.cls.cpu().numpy()

        for box, cls_id in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box)
            name = _ID_TO_NAME.get(int(cls_id), "plain text")
            kind = _NAME_TO_KIND.get(name, "text")
            regions.append(Region(page_id=page.id, bbox=(x1, y1, x2, y2), kind=kind))
    visualize_regions(pages, regions, output_dir="data/debug_layout")
    return regions
