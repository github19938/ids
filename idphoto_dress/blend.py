"""Alpha blending utilities.

All functions here work on ``uint8`` BGR images and ``uint8`` masks /
alpha channels with values in ``[0, 255]``.
"""

from __future__ import annotations

import cv2
import numpy as np


def feather_mask(mask: np.ndarray, ksize: int = 21) -> np.ndarray:
    """Apply a gaussian blur to soften mask edges.

    ``ksize`` must be odd and >= 3. Values smaller than 15 are clamped to
    15 to satisfy the project requirement.
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    ksize = max(15, int(ksize) | 1)
    return cv2.GaussianBlur(mask, (ksize, ksize), 0)


def alpha_blend(
    background: np.ndarray, foreground: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    """Standard ``out = fg * a + bg * (1 - a)`` blend.

    Parameters
    ----------
    background, foreground:
        ``HxWx3`` uint8 BGR images.
    alpha:
        ``HxW`` uint8 mask in ``[0, 255]``.
    """
    if alpha.ndim == 2:
        a = alpha.astype(np.float32) / 255.0
        a = a[:, :, None]
    else:
        a = alpha.astype(np.float32) / 255.0
    bg = background.astype(np.float32)
    fg = foreground.astype(np.float32)
    out = fg * a + bg * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def composite_layers(
    background: np.ndarray, layers
) -> np.ndarray:
    """Composite a list of ``(rgb, alpha)`` layers on top of ``background``.

    Layers are applied in order (first layer is bottom-most). Each alpha
    is automatically feathered with a gaussian blur before blending.
    """
    out = background.copy()
    for rgb, alpha in layers:
        if alpha is None:
            continue
        soft = feather_mask(alpha, ksize=21)
        out = alpha_blend(out, rgb, soft)
    return out
