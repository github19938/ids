"""color_match.py – luminance / color harmonization between portrait and suit.

A flat-rendered template will look "stuck on" if the portrait was lit warmly
or under-exposed.  We perform a *gentle* statistical color transfer of mean
brightness (and optionally chroma) from a reference region of the portrait
(default: the neck/shoulder area) into the body+collar layer, in LAB space.

The transfer is regulated by `strength ∈ [0, 1]`.  At 0 the suit is untouched;
at 1 the suit's mean L (and a/b) are fully replaced.  In practice 0.3–0.5 is
a tasteful default — strong enough to blend, weak enough to keep the suit's
designed color.

This is intentionally NOT a full Reinhard transfer (which can muddy bright
white shirts).  We match means but only optionally pull stds.

Public API
----------
    out_rgba = match_color(suit_rgba, portrait_bgr, ref_mask, strength=0.4)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ColorMatchConfig:
    enable: bool = True
    strength: float = 0.4         # 0..1, applied to L mean shift
    match_chroma: bool = False    # if True, also nudge a/b means
    match_chroma_strength: float = 0.15
    # Optional standard-deviation matching (kept off by default; can wash out fabric).
    match_std: bool = False


def match_color(
    suit_rgba: np.ndarray,
    portrait_bgr: np.ndarray,
    ref_mask: np.ndarray,
    config: Optional[ColorMatchConfig] = None,
) -> np.ndarray:
    """Return a color-harmonized copy of `suit_rgba`.

    Parameters
    ----------
    suit_rgba:
        HxWx4 uint8 garment image (already warped into portrait coords).
    portrait_bgr:
        Original HxWx3 uint8 portrait.
    ref_mask:
        HxW float32 mask in [0, 1] selecting the *portrait* pixels to use as
        the reference (typically the neck or upper-body skin area).
    """
    cfg = config or ColorMatchConfig()
    if not cfg.enable:
        return suit_rgba
    if suit_rgba.shape[:2] != portrait_bgr.shape[:2] or suit_rgba.shape[:2] != ref_mask.shape[:2]:
        raise ValueError("match_color: shape mismatch among inputs")

    # Build a binary-ish reference mask (portrait pixels we trust) and a
    # garment-validity mask (where the suit actually exists).
    ref = ref_mask
    if ref.sum() < 50:  # not enough samples
        logger.info("color_match: reference region too small (%d), skipping.", int(ref.sum()))
        return suit_rgba
    suit_alpha = suit_rgba[..., 3].astype(np.float32) / 255.0
    if suit_alpha.sum() < 50:
        return suit_rgba

    portrait_lab = cv2.cvtColor(portrait_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    suit_lab = cv2.cvtColor(suit_rgba[..., :3], cv2.COLOR_BGR2LAB).astype(np.float32)

    p_mean, p_std = _weighted_mean_std(portrait_lab, ref)
    s_mean, s_std = _weighted_mean_std(suit_lab, suit_alpha)

    out = suit_lab.copy()

    # L channel mean shift, weighted by `strength`.
    delta_L = (p_mean[0] - s_mean[0]) * cfg.strength
    out[..., 0] = out[..., 0] + delta_L

    if cfg.match_chroma:
        out[..., 1] = out[..., 1] + (p_mean[1] - s_mean[1]) * cfg.match_chroma_strength
        out[..., 2] = out[..., 2] + (p_mean[2] - s_mean[2]) * cfg.match_chroma_strength

    if cfg.match_std:
        # Avoid division blow-up; clamp std ratio to a sane band.
        ratio_L = np.clip((p_std[0] / max(s_std[0], 1e-3)), 0.7, 1.4)
        out[..., 0] = (out[..., 0] - s_mean[0]) * (
            1.0 + (ratio_L - 1.0) * cfg.strength
        ) + s_mean[0] + delta_L

    out[..., 0] = np.clip(out[..., 0], 0.0, 255.0)
    out[..., 1] = np.clip(out[..., 1], 0.0, 255.0)
    out[..., 2] = np.clip(out[..., 2], 0.0, 255.0)

    bgr = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_LAB2BGR)
    matched = suit_rgba.copy()
    matched[..., :3] = bgr
    return matched


# ---------------------------------------------------------------------------
def _weighted_mean_std(img: np.ndarray, weights: np.ndarray):
    """Per-channel weighted mean / std for an HxWxC image and HxW weights."""
    w = weights.astype(np.float32)
    wsum = float(w.sum())
    if wsum < 1.0:
        return np.zeros(img.shape[2], np.float32), np.ones(img.shape[2], np.float32)
    flat = img.reshape(-1, img.shape[2])
    wf = w.reshape(-1)
    mean = (flat * wf[:, None]).sum(axis=0) / wsum
    var = ((flat - mean) ** 2 * wf[:, None]).sum(axis=0) / wsum
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)
