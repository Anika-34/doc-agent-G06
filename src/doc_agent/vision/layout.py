# """Stage 2 — layout detection / segmentation"""
# from __future__ import annotations
# from ..contracts import *  # noqa

# def detect(pages: list[Page], cfg: dict) -> list[Region]:
#     """Detect text/table/figure/heading regions. IMPLEMENT."""
#     raise NotImplementedError("Stage 2: layout detection")

# vision/layout.py

from __future__ import annotations
# import cv2
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from ..contracts import *  # noqa
# import os

# # Color palette (BGR) for region visualization
# _COLOR_MAP = {
#     "text": (0, 255, 0),       # Green
#     "figure": (0, 0, 255),     # Red
#     "table": (255, 0, 0),      # Blue
#     "heading": (0, 0, 0),      # Black
# }

# def visualize_regions(pages: list[Page], regions: list[Region], output_dir: str = "data/debug_layout_2") -> None:
#     """Saves page images with color-coded bounding boxes and labels for debugging."""
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Group detected regions by page_id
#     regions_by_page: dict[str, list[Region]] = {}
#     for r in regions:
#         regions_by_page.setdefault(r.page_id, []).append(r)

#     for page in pages:
#         img = cv2.imread(page.image_path)
#         if img is None:
#             continue
            
#         page_regions = regions_by_page.get(page.id, [])
#         for r in page_regions:
#             x1, y1, x2, y2 = r.bbox
#             color = _COLOR_MAP.get(r.kind, (255, 255, 255))
            
#             # Draw bounding box
#             cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)
            
#             # Draw region label above box
#             cv2.putText(
#                 img, 
#                 r.kind, 
#                 (x1, max(y1 - 5, 15)), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 
#                 0.6, 
#                 color, 
#                 2, 
#                 cv2.LINE_AA
#             )

#         out_path = os.path.join(output_dir, f"debug_{page.id}.png")
#         cv2.imwrite(out_path, img)

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
    "isolate_formula": "text",  # not real text for a Bangla-OCR reader; treat like a figure
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

# def detect(pages: list[Page], cfg: dict) -> list[Region]:
#     model = _get_model()
#     # conf_thr = cfg["layout"].get("score_thr", 0.25)
#     conf_thr = cfg["layout"].get("score_thr", 0.07)   # down from 0.25
#     device = cfg.get("device", "cpu")

#     regions: list[Region] = []
#     for page in pages:
#         # till now best 1144
#         # det = model.predict(page.image_path, imgsz=1024, conf=conf_thr, device=device)[0]
#         det = model.predict(page.image_path, imgsz=1184, conf=conf_thr, device=device)[0]  # up from 1024

#         boxes = det.boxes.xyxy.cpu().numpy()
#         classes = det.boxes.cls.cpu().numpy()

#         for box, cls_id in zip(boxes, classes):
#             x1, y1, x2, y2 = map(int, box)
#             name = _ID_TO_NAME.get(int(cls_id), "plain text")
#             kind = _NAME_TO_KIND.get(name, "text")
#             regions.append(Region(page_id=page.id, bbox=(x1, y1, x2, y2), kind=kind))
#     visualize_regions(pages, regions)
#     return regions
# vision/layout.py

def _iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0

def _dedup_boxes(
    detections: list[tuple[tuple[int, int, int, int], str, float]],  # (bbox, kind, confidence)
    iou_thr: float = 0.5,
) -> list[tuple[tuple[int, int, int, int], str]]:
    """Class-agnostic dedup: when two boxes of ANY kind overlap heavily, they're almost
    certainly the same physical region detected twice — keep only the higher-confidence one.
    This is what class-aware NMS inside the model doesn't catch."""
    detections = sorted(detections, key=lambda d: d[2], reverse=True)  # highest confidence first
    kept: list[tuple[tuple[int, int, int, int], str, float]] = []
    for box, kind, conf in detections:
        if all(_iou(box, kb) < iou_thr for kb, _, _ in kept):
            kept.append((box, kind, conf))
    return [(box, kind) for box, kind, _ in kept]


# vision/layout.py

def _containment_ratio(inner: tuple, outer: tuple) -> float:
    """What fraction of `inner`'s area sits inside `outer`."""
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    x1, y1 = max(ix1, ox1), max(iy1, oy1)
    x2, y2 = min(ix2, ox2), min(iy2, oy2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    inner_area = (ix2 - ix1) * (iy2 - iy1)
    return inter / inner_area if inner_area > 0 else 0

def _drop_contained_boxes(
    boxes_kinds: list[tuple[tuple[int, int, int, int], str]],
    containment_thr: float = 0.85,
) -> list[tuple[tuple[int, int, int, int], str]]:
    """Drop boxes that sit almost entirely inside a larger box — the bigger
    box's OCR already covers that content, so keeping both duplicates it."""
    areas = [((x2 - x1) * (y2 - y1), box, kind) for (box, kind) in boxes_kinds for x1, y1, x2, y2 in [box]]
    areas.sort(key=lambda t: t[0], reverse=True)  # largest first

    kept: list[tuple[tuple, str]] = []
    for area, box, kind in areas:
        if any(_containment_ratio(box, kb) > containment_thr for kb, _ in kept):
            continue  # this box is basically inside an already-kept larger box — skip it
        kept.append((box, kind))
    return kept

def detect(pages: list[Page], cfg: dict) -> list[Region]:
    model = _get_model()
    conf_thr = cfg["layout"].get("score_thr", 0.085)
    imgsz = cfg["layout"].get("imgsz", 1184)
    device = cfg.get("device", "cpu")
    dedup_iou = cfg["layout"].get("dedup_iou_thr", 0.5)

    regions: list[Region] = []
    for page in pages:
        det = model.predict(page.image_path, imgsz=imgsz, conf=conf_thr, device=device)[0]
        boxes = det.boxes.xyxy.cpu().numpy()
        classes = det.boxes.cls.cpu().numpy()
        confs = det.boxes.conf.cpu().numpy()   # need this now — kept alongside kind for dedup

        raw = []
        for box, cls_id, conf in zip(boxes, classes, confs):
            x1, y1, x2, y2 = map(int, box)
            name = _ID_TO_NAME.get(int(cls_id), "plain text")
            kind = _NAME_TO_KIND.get(name, "text")
            raw.append(((x1, y1, x2, y2), kind, float(conf)))

        deduped = _dedup_boxes(raw, iou_thr=dedup_iou)
        deduped = _drop_contained_boxes(deduped, containment_thr=0.85)
        for (x1, y1, x2, y2), kind in deduped:
            regions.append(Region(page_id=page.id, bbox=(x1, y1, x2, y2), kind=kind))

    # visualize_regions(pages, regions)
    return regions