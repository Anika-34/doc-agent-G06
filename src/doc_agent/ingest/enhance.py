"""Stage 1 — ENHANCEMENT (VAE / diffusion) — generative denoise / super-resolution of degraded scans"""
from __future__ import annotations
from ..contracts import *  # noqa

import cv2
import numpy as np
from PIL import Image


def _to_cv(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _to_pil(cv_img: np.ndarray) -> Image.Image:
    if cv_img.ndim == 2:
        return Image.fromarray(cv_img)
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


def _upscale_2x(pil_img: Image.Image) -> Image.Image:
    cv_img = _to_cv(pil_img)
    h, w = cv_img.shape[:2]
    return _to_pil(cv2.resize(cv_img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC))


class Enhancer:
    """No learned (VAE/diffusion) enhancer implemented -- the classical
    sweep found no transform worth deploying, learned or otherwise."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["enhance"]
        self._apply_upscale_for: set[str] = set() 

    def train(self, pages: list[Page]) -> None:
        raise NotImplementedError("No learned enhancer implemented")

    def apply(self, pages: list[Page]) -> list[Page]:
        if not self._apply_upscale_for:
            return pages
        for page in pages:
            if page.doc_id in self._apply_upscale_for:
                img = Image.open(page.image_path).convert("RGB")
                _upscale_2x(img).save(page.image_path)
        return pages


def run(pages: list[Page], cfg: dict) -> list[Page]:
    if not cfg["enhance"]["enabled"]:
        return pages
    return Enhancer(cfg).apply(pages)