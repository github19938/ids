"""Alpha-aware compositing primitives.

All functions in this module operate on float32 images in ``[0, 1]``.
There is exactly one blending primitive (``alpha_blend``) and one
high-level layered compositor (``composite_pipeline``). No hard
overlay, no mode switches, no thresholding.
"""

from __future__ import annotations

import cv2
import numpy as np

EPS = 1e-6


# ---------------------------------------------------------------------------
# Color space helpers (sRGB <-> linear)
# ---------------------------------------------------------------------------


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.power(np.clip(x, 0.0, 1.0), 2.2).astype(np.float32)


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    return np.power(np.clip(x, 0.0, 1.0), 1.0 / 2.2).astype(np.float32)


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------


def alpha_blend(
    fg: np.ndarray,
    bg: np.ndarray,
    alpha: np.ndarray,
    use_linear: bool = False,
) -> np.ndarray:
    """Standard *over* compositing: ``out = fg * a + bg * (1 - a)``.

    Parameters
    ----------
    fg, bg : np.ndarray
        Foreground / background images. ``float32`` in ``[0, 1]``,
        either ``(H, W)`` or ``(H, W, C)``. Shapes must broadcast.
    alpha : np.ndarray
        Continuous alpha matte. ``float32`` in ``[0, 1]``. Shape
        ``(H, W)`` or ``(H, W, 1)``; broadcast against the colour
        channels automatically.
    use_linear : bool, optional
        If ``True``, blend in linear light (sRGB->linear, blend,
        linear->sRGB). Defaults to plain sRGB blending.

    Returns
    -------
    np.ndarray
        Blended image, ``float32`` in ``[0, 1]``, same shape as ``fg``
        (after broadcasting).
    """
    fg_f = np.asarray(fg, dtype=np.float32)
    bg_f = np.asarray(bg, dtype=np.float32)
    a = np.asarray(alpha, dtype=np.float32)

    if a.ndim == 2 and fg_f.ndim == 3:
        a = a[..., None]
    a = np.clip(a, 0.0, 1.0)

    if use_linear:
        fg_lin = _srgb_to_linear(fg_f)
        bg_lin = _srgb_to_linear(bg_f)
        out_lin = fg_lin * a + bg_lin * (1.0 - a)
        out = _linear_to_srgb(out_lin)
    else:
        out = fg_f * a + bg_f * (1.0 - a)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def decontaminate(hair_rgb: np.ndarray, alpha_hair: np.ndarray) -> np.ndarray:
    """Recover an un-premultiplied (clean) foreground colour.

    Given a *premultiplied* foreground ``hair_rgb = clean * alpha`` and
    its alpha, reconstruct the clean RGB by dividing out alpha. This is
    the standard cure for the "white / grey halo" around hair when
    blending with a different background.

    Parameters
    ----------
    hair_rgb : np.ndarray
        Premultiplied hair colour, ``float32`` in ``[0, 1]``, shape
        ``(H, W, 3)``.
    alpha_hair : np.ndarray
        Hair alpha, ``float32`` in ``[0, 1]``, shape ``(H, W)`` or
        ``(H, W, 1)``.

    Returns
    -------
    np.ndarray
        Decontaminated (un-premultiplied) hair colour, ``float32`` in
        ``[0, 1]``, shape ``(H, W, 3)``.
    """
    a = np.asarray(alpha_hair, dtype=np.float32)
    if a.ndim == 2:
        a = a[..., None]
    a = np.clip(a, 0.0, 1.0)

    fg = np.asarray(hair_rgb, dtype=np.float32)
    clean = fg / (a + EPS)
    return np.clip(clean, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# High-level layered compositor
# ---------------------------------------------------------------------------


def composite_pipeline(
    body_rgb: np.ndarray,
    neck_rgb: np.ndarray,
    face_rgb: np.ndarray,
    hair_rgb_clean: np.ndarray,
    alpha_visible_hair: np.ndarray,
    cloth_rgb: np.ndarray,
    cloth_alpha: np.ndarray,
    alpha_neck: np.ndarray,
    alpha_face: np.ndarray,
    use_linear: bool = False,
) -> np.ndarray:
    """Layered alpha compositor with the fixed render order.

    Render order (bottom to top):

        1. body  (already-warped try-on body image)
        2. neck  (from the original photo)
        3. face  (from the original photo)
        4. hair  (clean hair colour, modulated by ``alpha_visible_hair``)
        5. collar / cloth  (cloth_rgb modulated by ``cloth_alpha``)

    Every layer is composited with :func:`alpha_blend`. There is no
    hard overlay anywhere in this function.
    """
    canvas = np.asarray(body_rgb, dtype=np.float32)
    canvas = alpha_blend(neck_rgb, canvas, alpha_neck, use_linear=use_linear)
    canvas = alpha_blend(face_rgb, canvas, alpha_face, use_linear=use_linear)
    canvas = alpha_blend(hair_rgb_clean, canvas, alpha_visible_hair, use_linear=use_linear)
    canvas = alpha_blend(cloth_rgb, canvas, cloth_alpha, use_linear=use_linear)
    return canvas
