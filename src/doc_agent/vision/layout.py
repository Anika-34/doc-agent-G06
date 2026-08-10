"""Stage 2 — layout detection / segmentation

Detects and segments a page into regions of different types:
  - text: main body paragraphs
  - table: tabular data (detected via aligned rows / Markdown structure)
  - figure: images, diagrams, charts
  - heading: section headers, chapter titles

Strategy (baseline):
  1. Binarize and denoise the page
  2. Detect connected components (text blocks)
  3. Use heuristics to classify each region:
     - Figure: isolated large blobs, caption with figure keywords (চিত্র, Fig, etc.)
     - Table: multiple aligned columns, rows with consistent spacing
     - Heading: isolated text at top/section starts, large glyphs
     - Text: everything else

"""
from __future__ import annotations
import re
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from ..contracts import *  # noqa
from ..logging_conf import get_logger
from collections import defaultdict

logger = get_logger(__name__)


def detect(pages: list[Page], cfg: dict) -> list[Region]:
    """Detect text/table/figure/heading regions on all pages.
    
    Args:
        pages: List of Page objects with valid image_path
        cfg: Config dict with layout detection parameters
        
    Returns:
        List of Region objects with bounding boxes and kind labels
    """
    regions = []
    detector = LayoutDetector(cfg)
    
    for page in pages:
        try:
            page_regions = detector.detect_page(page)
            regions.extend(page_regions)
            logger.debug(f"Detected {len(page_regions)} regions on page {page.id}")
        except Exception as e:
            logger.warning(f"Layout detection failed for page {page.id}: {e}")
    
    logger.info(f"Layout detection complete: {len(regions)} total regions")
    return regions


class LayoutDetector:
    """Classical layout detection using connected components + heuristics.
    
    Configurable via cfg['layout'] with parameters:
      - min_text_height, max_text_height: glyph height range
      - min_region_area: minimum pixels to be a region (skip noise)
      - table_column_threshold: min columns for a region to be classified as table
      - heading_min_iou: IoU threshold for heading-like isolated regions
    """
    
    def __init__(self, cfg: dict) -> None:
        layout_cfg = cfg.get("layout", {})
        self.min_text_height = layout_cfg.get("min_text_height", 8)
        self.max_text_height = layout_cfg.get("max_text_height", 100)
        # self.min_region_area = layout_cfg.get("min_region_area", 20)
        self.min_region_area = layout_cfg.get(
            "min_region_area",
            cfg.get("ingest", {}).get("min_blob_area", 15),
        )
        self.table_column_threshold = layout_cfg.get("table_column_threshold", 3)
        self.heading_isolated_threshold = layout_cfg.get("heading_isolated_threshold", 0.15)
        self.photo_blob_count_threshold = layout_cfg.get("photo_blob_count_threshold", 40)
        self.photo_min_area_ratio = layout_cfg.get("photo_min_area_ratio", 0.01)
        self.photo_max_median_component_area = layout_cfg.get(
            "photo_max_median_component_area", 12
        )
    
    def detect_page(self, page: Page) -> list[Region]:
        """Detect layout regions on a single page."""
        img = cv2.imread(page.image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {page.image_path}")
        
        h, w = img.shape
        
        # Preprocessing: threshold + denoise
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Connected components
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cleaned, connectivity=8
        )

        photo_regions, absorbed_ids = self._detect_photo_regions(
            cleaned, n_labels, stats, centroids, h, w, page.id
        )


        
        # Extract regions from each component
        # regions = []
        regions = list(photo_regions)
        for label_id in range(1, n_labels):  # skip background (0)
            if label_id in absorbed_ids:
                continue

            stat = stats[label_id]
            x, y, w_comp, h_comp, area = (
                stat[cv2.CC_STAT_LEFT],
                stat[cv2.CC_STAT_TOP],
                stat[cv2.CC_STAT_WIDTH],
                stat[cv2.CC_STAT_HEIGHT],
                stat[cv2.CC_STAT_AREA],
            )
            
            # Skip tiny noise
            if area < self.min_region_area:
                continue
            
            # Skip regions outside glyph height (too big = figures/tables, too small = noise)
            if h_comp < self.min_text_height or h_comp > self.max_text_height:
                kind = _classify_large_region(x, y, w_comp, h_comp, area, h, w)
            else:
                # Text-sized regions: classify as text unless evidence of heading
                kind = _classify_text_region(label_id, labels, x, y, w_comp, h_comp, cleaned)
            
            bbox = (x, y, x + w_comp, y + h_comp)
            regions.append(Region(
                page_id=page.id,
                bbox=bbox,
                kind=kind
            ))
        
        # Post-process: merge nearby regions of the same kind (optional, for cleaner output)
        regions = _merge_nearby_regions(regions)
        
        return regions

    def _detect_photo_regions(
            self,
            cleaned: np.ndarray,
            n_labels: int,
            stats: np.ndarray,
            centroids: np.ndarray,
            img_h: int,
            img_w: int,
            page_id: str,
        ) -> tuple[list[Region], set[int]]:
        """Find areas with an unusually high density of tiny connected
        components (halftone dot-screen noise) and collapse each into one
        'figure' Region, rather than emitting each dot as its own text-sized
        speck.
 
        Returns (figure_regions, absorbed_component_ids) -- the latter tells
        the caller which raw component label_ids were folded into a figure
        so they aren't also emitted individually as text/heading regions.
        """
        # Merge components that sit close together into coarse blobs. A large
        # kernel is intentional: halftone dots are packed tightly, so this
        # fuses an entire photo into one blob while leaving normal,
        # well-spaced text lines mostly separate.
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        dilated = cv2.dilate(cleaned, dilate_kernel, iterations=1)
        n_blobs, blob_labels, blob_stats, _ = cv2.connectedComponentsWithStats(
            dilated, connectivity=8
        )
 
        members: dict[int, list[int]] = defaultdict(list)
        for label_id in range(1, n_labels):
            cx, cy = centroids[label_id]
            blob_id = blob_labels[int(cy), int(cx)]
            if blob_id != 0:
                members[blob_id].append(label_id)
 
        figure_regions: list[Region] = []
        absorbed_ids: set[int] = set()
 
        for blob_id, member_ids in members.items():
            area_ratio = blob_stats[blob_id][cv2.CC_STAT_AREA] / (img_h * img_w)
            member_areas = [stats[i][cv2.CC_STAT_AREA] for i in member_ids]
            median_component_area = float(np.median(member_areas)) if member_areas else 0.0
 
            is_dense_enough = len(member_ids) >= self.photo_blob_count_threshold
            is_big_enough = area_ratio >= self.photo_min_area_ratio
            is_dot_sized = median_component_area <= self.photo_max_median_component_area

        #     if (
        #         len(member_ids) >= self.photo_blob_count_threshold
        #         and area_ratio >= self.photo_min_area_ratio
        #     ):
        #         bx, by, bw, bh = (
        #             blob_stats[blob_id][cv2.CC_STAT_LEFT],
        #             blob_stats[blob_id][cv2.CC_STAT_TOP],
        #             blob_stats[blob_id][cv2.CC_STAT_WIDTH],
        #             blob_stats[blob_id][cv2.CC_STAT_HEIGHT],
        #         )
        #         figure_regions.append(Region(
        #             page_id=page_id,
        #             bbox=(bx, by, bx + bw, by + bh),
        #             kind="figure",
        #         ))
        #         absorbed_ids.update(member_ids)
 
        # return figure_regions, absorbed_ids
        if is_dense_enough and is_big_enough and is_dot_sized:
            bx, by, bw, bh = (
                blob_stats[blob_id][cv2.CC_STAT_LEFT],
                blob_stats[blob_id][cv2.CC_STAT_TOP],
                blob_stats[blob_id][cv2.CC_STAT_WIDTH],
                blob_stats[blob_id][cv2.CC_STAT_HEIGHT],
            )
            figure_regions.append(Region(
                page_id=page_id,
                bbox=(bx, by, bx + bw, by + bh),
                kind="figure",
            ))
            absorbed_ids.update(member_ids)
            print(f"Detected figure region on page {page_id}: bbox=({bx},{by},{bx+bw},{by+bh}), "
                  f"member_count={len(member_ids)}, area_ratio={area_ratio:.4f}, median_component_area={median_component_area:.2f}")
 
        return figure_regions, absorbed_ids




def _classify_large_region(x: int, y: int, w: int, h: int, area: int, 
                           img_h: int, img_w: int) -> str:
    """Classify large regions (figures, tables, or figure captions).
    
    Heuristics:
      - If very wide and multiple lines: table
      - If isolated and large: figure
      - If small width relative to image: caption (treat as text for now)
    """
    aspect_ratio = w / h if h > 0 else 0
    area_ratio = area / (img_h * img_w)
    
    # Wide, multi-line regions suggest tables
    if aspect_ratio > 1.5 and h > 50:
        return "table"
    
    # Large isolated blobs (figures, maps, diagrams)
    if area_ratio > 0.05:
        return "figure"
    
    # Default to text (could be a large heading or caption)
    return "text"


def _classify_text_region(label_id: int, labels: np.ndarray, 
                          x: int, y: int, w: int, h: int, 
                          binary: np.ndarray) -> str:
    """Classify text-sized regions.
    
    Heuristics:
      - If isolated at top/isolated layout: heading
      - Otherwise: text
    """
    # Isolation heuristic: if region is at top 20% of page and isolated, likely heading
    img_h = labels.shape[0]
    if y < 0.2 * img_h:
        mask = (labels == label_id).astype(np.uint8)
        neighbor_area = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (50, 20)), iterations=1)
        neighbor_pixels = np.sum(neighbor_area * binary)
        own_pixels = np.sum(mask * binary)
        if own_pixels > 0 and neighbor_pixels / own_pixels < 2.0:
            return "heading"
    
    return "text"


def _merge_nearby_regions(regions: list[Region], distance_threshold: int = 20) -> list[Region]:
    """Merge regions that are close and the same kind (optional, for cleaner output).
    
    Args:
        regions: List of detected regions
        distance_threshold: Max distance between bboxes to merge
        
    Returns:
        Merged region list
    """
    if not regions:
        return regions
    
    # Group by kind
    by_kind = {}
    for region in regions:
        if region.kind not in by_kind:
            by_kind[region.kind] = []
        by_kind[region.kind].append(region)
    
    merged = []
    for kind, kind_regions in by_kind.items():
        # Simple greedy merging: merge overlapping or nearby bboxes
        kind_regions.sort(key=lambda r: r.bbox[0])  # sort by x
        
        current_merged = []
        for region in kind_regions:
            if not current_merged:
                current_merged.append(region)
            else:
                last = current_merged[-1]
                x1_1, y1_1, x2_1, y2_1 = last.bbox
                x1_2, y1_2, x2_2, y2_2 = region.bbox
                
                # Check overlap or proximity
                x_overlap = not (x2_1 < x1_2 - distance_threshold or x2_2 < x1_1 - distance_threshold)
                y_overlap = not (y2_1 < y1_2 - distance_threshold or y2_2 < y1_1 - distance_threshold)
                
                if x_overlap and y_overlap:
                    # Merge: union of bboxes
                    merged_bbox = (
                        min(x1_1, x1_2),
                        min(y1_1, y1_2),
                        max(x2_1, x2_2),
                        max(y2_1, y2_2),
                    )
                    current_merged[-1] = Region(
                        page_id=last.page_id,
                        bbox=merged_bbox,
                        kind=kind
                    )
                else:
                    current_merged.append(region)
        
        merged.extend(current_merged)
    
    return merged

