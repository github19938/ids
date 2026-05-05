"""render.py – layered alpha compositing.

The pipeline produces a final image by stacking the following layers,
in this exact bottom-to-top order:

    1. background (the input photo's background, behind the person)
    2. body       (warped suit body, masked to NOT touch face/hair/neck)
    3. collar     (warped suit collar, optional, also occlusion-masked)
    4. neck       (the original portrait pixels in the neck strip — these
                   are preserved exactly so the collar/skin transition is
                   the subject's real skin, not a synthesized blob)
    5. face       (the original face pixels)
    6. hair       (the original hair pixels — drawn LAST so flyaway hair
                   correctly overlaps the collar)

Every blend uses linear (premultiplied) alpha and a Gaussian-feathered
mask boundary to guarantee no hard cut lines.

Public API
----------
    composite = layered_render(
        portrait_bgr, masks, body_rgba, collar_rgba, render_cfg
    )
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .types import PortraitMasks

logger = logging.getLogger(__name__)


@dataclass
class RenderConfig:
    """All knobs that influence the final composite."""

    feather_px: int = 6
    # How aggressively to dilate the "do not paint over" region (face+hair+neck)
    # before subtracting it from the suit alpha.  Larger value = safer, but the
    # collar may sit lower on the neck.
    protect_dilate_px: int = 2
    # When the suit alpha is very low, we let the original photo bleed through;
    # this keeps the background unchanged outside the garment.
    suit_alpha_floor: float = 0.0
    # Optional final unsharp mask to recover edge crispness lost during blending.
    final_sharpen: float = 0.15


def layered_render(
    portrait_bgr: np.ndarray,
    masks: PortraitMasks,
    body_rgba: np.ndarray,
    collar_rgba: Optional[np.ndarray],
    cfg: Optional[RenderConfig] = None,
    background_bgr: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Composite the warped garment with the original portrait.

    Parameters
    ----------
    portrait_bgr:
        Original input image (HxWx3 uint8 BGR).
    masks:
        :class:`PortraitMasks` from `segmentation.get_masks`.
    body_rgba, collar_rgba:
        RGBA garment layers, ALREADY warped into portrait coordinates
        (i.e. same HxW as `portrait_bgr`).  `collar_rgba` may be None.
    cfg:
        Render parameters.
    background_bgr:
        Optional replacement background, same size as portrait.  If None,
        the original portrait's background is preserved.

    Returns
    -------
    Final HxWx3 uint8 BGR image.
    """
    cfg = cfg or RenderConfig()
    h, w = portrait_bgr.shape[:2]
    if body_rgba.shape[:2] != (h, w):
        raise ValueError("body_rgba size must match portrait")
    if collar_rgba is not None and collar_rgba.shape[:2] != (h, w):
        raise ValueError("collar_rgba size must match portrait")

    # ------------------------------------------------------------------
    # Step 0. Build the "protect" mask: pixels we are not allowed to paint over.
    # face + hair + neck are the no-go zones for the suit.
    # ------------------------------------------------------------------
    protect = np.clip(masks.face + masks.hair + masks.neck, 0.0, 1.0)
    if cfg.protect_dilate_px > 0:
        k = cfg.protect_dilate_px
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        protect = cv2.dilate(protect, kernel)
    protect = _feather(protect, cfg.feather_px)

    # ------------------------------------------------------------------
    # Step 1. Background layer.
    # ------------------------------------------------------------------
    if background_bgr is None:
        canvas = portrait_bgr.astype(np.float32) / 255.0
    else:
        if background_bgr.shape[:2] != (h, w):
            raise ValueError("background_bgr size must match portrait")
        # Composite person onto new background using the soft `person` mask.
        bg = background_bgr.astype(np.float32) / 255.0
        fg = portrait_bgr.astype(np.float32) / 255.0
        person = masks.person[..., None]
        canvas = fg * person + bg * (1.0 - person)

    # ------------------------------------------------------------------
    # Step 2. Body layer.
    # ------------------------------------------------------------------
    body_rgb, body_a = _split_rgba(body_rgba)
    # Suit is only allowed where the *person's* upper body is, AND not where
    # protect says no.  This is what eliminates "suit pasted over the chin".
    body_alpha = body_a * (1.0 - protect)
    # Confine to person silhouette + a bit of slack outward (the lapel may
    # legitimately extend a few px past the original shoulder).
    person_dilated = _dilate(masks.person, 8)
    body_alpha = body_alpha * person_dilated
    body_alpha = _feather(body_alpha, cfg.feather_px)
    body_alpha = np.clip(body_alpha, cfg.suit_alpha_floor, 1.0)
    canvas = _alpha_over(body_rgb, body_alpha, canvas)

    # ------------------------------------------------------------------
    # Step 3. Collar layer.
    # ------------------------------------------------------------------
    if collar_rgba is not None:
        collar_rgb, collar_a = _split_rgba(collar_rgba)
        collar_alpha = collar_a * (1.0 - protect)
        collar_alpha = _feather(collar_alpha, max(2, cfg.feather_px // 2))
        canvas = _alpha_over(collar_rgb, collar_alpha, canvas)

    # ------------------------------------------------------------------
    # Step 4. Neck layer (re-paste original neck pixels).
    # ------------------------------------------------------------------
    neck_alpha = _feather(masks.neck, max(2, cfg.feather_px // 2))
    canvas = _alpha_over(portrait_bgr.astype(np.float32) / 255.0, neck_alpha, canvas)

    # ------------------------------------------------------------------
    # Step 5. Face layer.
    # ------------------------------------------------------------------
    face_alpha = _feather(masks.face, max(2, cfg.feather_px // 2))
    canvas = _alpha_over(portrait_bgr.astype(np.float32) / 255.0, face_alpha, canvas)

    # ------------------------------------------------------------------
    # Step 6. Hair layer (last — flyaways must sit on top of the collar).
    # ------------------------------------------------------------------
    hair_alpha = _feather(masks.hair, max(2, cfg.feather_px // 2))
    canvas = _alpha_over(portrait_bgr.astype(np.float32) / 255.0, hair_alpha, canvas)

    # ------------------------------------------------------------------
    # Optional sharpening pass to crisp up edges blurred by feathering.
    # ------------------------------------------------------------------
    if cfg.final_sharpen > 0.0:
        canvas = _unsharp(canvas, amount=cfg.final_sharpen, radius=1.0)

    out = np.clip(canvas * 255.0, 0.0, 255.0).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _split_rgba(rgba: np.ndarray):
    """Return (rgb_float, alpha_float) in BGR order, [0,1]."""
    if rgba.dtype != np.uint8:
        raise TypeError("RGBA layers must be uint8")
    if rgba.shape[2] != 4:
        raise ValueError("RGBA layers must have 4 channels")
    rgb = rgba[..., :3].astype(np.float32) / 255.0
    a = rgba[..., 3].astype(np.float32) / 255.0
    return rgb, a


def _alpha_over(src_rgb: np.ndarray, src_alpha: np.ndarray, dst_rgb: np.ndarray) -> np.ndarray:
    """Standard `src` over `dst` Porter-Duff blend.  All inputs float [0,1]."""
    a = src_alpha[..., None]
    return src_rgb * a + dst_rgb * (1.0 - a)


def _feather(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    k = max(3, radius * 2 + 1)
    return cv2.GaussianBlur(mask, (k, k), sigmaX=radius)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    # Threshold to binary, dilate, then return as float.
    binary = (mask > 0.05).astype(np.uint8) * 255
    dil = cv2.dilate(binary, kernel)
    return dil.astype(np.float32) / 255.0


def _unsharp(img: np.ndarray, amount: float, radius: float) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=radius)
    sharp = cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0.0)
    return np.clip(sharp, 0.0, 1.0)
