"""Single-entry virtual try-on pipeline (matting-only).

This is the only public entry point of the project. Everything else
(matting, decontamination, alpha blending) is a helper used by this
function. There is no mode switch, no rule-based occlusion, no
binary masks.
"""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

from .compositing import alpha_blend, decontaminate
from .matting import run_matting


def _to_float01(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return np.clip(image.astype(np.float32), 0.0, 1.0)


def _ensure_alpha01(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Coerce an arbitrary mask input into a continuous ``float32`` alpha.

    The mask is *not* thresholded; values are merely scaled to ``[0, 1]``
    and resized to the target shape.
    """
    a = np.asarray(mask)
    if a.ndim == 3:
        a = a[..., 0]

    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    else:
        a = a.astype(np.float32)

    a = np.clip(a, 0.0, 1.0)

    h, w = target_hw
    if a.shape[:2] != (h, w):
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)

    return a.astype(np.float32)


def apply_virtual_tryon(
    image: np.ndarray,
    cloth_rgba: np.ndarray,
    face_mask: np.ndarray,
    keypoints: Optional[Any] = None,
    use_linear: bool = False,
    matting_model: Optional[object] = None,
) -> np.ndarray:
    """Virtual try-on with matting-driven hair occlusion.

    The full algorithm (no branches, no modes):

        1. ``alpha_hair = run_matting(image)``
        2. ``hair_rgb = image * alpha_hair`` ; ``background = image * (1 - alpha_hair)``
        3. ``clean_hair = decontaminate(hair_rgb, alpha_hair)``
        4. ``cloth_alpha = cloth_rgba[..., 3] / 255`` resized + slightly blurred
        5. ``cloth_alpha *= (1 - face_mask)``
        6. ``alpha_visible = alpha_hair * (1 - cloth_alpha)``
        7. ``result = alpha_blend(clean_hair, background, alpha_visible)``
        8. ``result = alpha_blend(cloth_rgb, result, cloth_alpha)``

    Parameters
    ----------
    image : np.ndarray
        Source portrait, RGB, shape ``(H, W, 3)``. ``uint8`` or
        ``float32`` in ``[0, 1]``.
    cloth_rgba : np.ndarray
        Garment image with alpha channel, shape ``(Hc, Wc, 4)``,
        ``uint8`` or ``float32``. Will be resized to ``(H, W)``.
    face_mask : np.ndarray
        Continuous face protection mask in ``[0, 1]``, shape
        ``(H, W)`` or ``(H, W, 1)``. Will be resized as needed and
        clipped, but not thresholded.
    keypoints : Any, optional
        Reserved for downstream consumers. Currently unused by the
        pipeline (no rule-based geometry is allowed). Accepted for
        API stability.
    use_linear : bool, optional
        If ``True``, perform every alpha-blend in linear light. The
        default (``False``) blends in sRGB, which is the simple mode
        called out in the spec.
    matting_model : object, optional
        Custom matting backend (object exposing
        ``predict(rgb_uint8) -> alpha``). Defaults to the placeholder
        bundled in :mod:`virtual_tryon.matting`.

    Returns
    -------
    np.ndarray
        The composited image, ``float32`` in ``[0, 1]``, shape ``(H, W, 3)``.
    """
    del keypoints  # accepted for API parity; not used

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be (H, W, 3); got {image.shape}")
    if cloth_rgba.ndim != 3 or cloth_rgba.shape[2] != 4:
        raise ValueError(f"cloth_rgba must be (H, W, 4); got {cloth_rgba.shape}")

    h, w = image.shape[:2]

    # ---- 1) Matting ----
    alpha_hair = run_matting(image, model=matting_model)  # float32 in [0, 1]
    alpha_hair_3 = alpha_hair[..., None]

    # ---- 2) Premultiplied hair / complement background ----
    image_f = _to_float01(image)
    hair_rgb = image_f * alpha_hair_3
    background = image_f * (1.0 - alpha_hair_3)

    # ---- 3) De-contamination (un-premultiply) ----
    clean_hair = decontaminate(hair_rgb, alpha_hair)

    # ---- 4) Cloth alpha ----
    cloth_resized = cv2.resize(cloth_rgba, (w, h), interpolation=cv2.INTER_LINEAR)
    cloth_rgb = _to_float01(cloth_resized[..., :3])

    cloth_alpha_raw = cloth_resized[..., 3]
    if cloth_alpha_raw.dtype == np.uint8:
        cloth_alpha = cloth_alpha_raw.astype(np.float32) / 255.0
    else:
        cloth_alpha = np.clip(cloth_alpha_raw.astype(np.float32), 0.0, 1.0)
    cloth_alpha = cv2.GaussianBlur(cloth_alpha, (0, 0), sigmaX=0.8, sigmaY=0.8)
    cloth_alpha = np.clip(cloth_alpha, 0.0, 1.0).astype(np.float32)

    # ---- 5) Face protection ----
    face_alpha = _ensure_alpha01(face_mask, (h, w))
    cloth_alpha = cloth_alpha * (1.0 - face_alpha)

    # ---- 6) Visible hair alpha ----
    alpha_visible = alpha_hair * (1.0 - cloth_alpha)
    alpha_visible = np.clip(alpha_visible, 0.0, 1.0).astype(np.float32)

    # ---- 7) Hair over background ----
    result = alpha_blend(clean_hair, background, alpha_visible, use_linear=use_linear)

    # ---- 8) Cloth on top (collar layer) ----
    result = alpha_blend(cloth_rgb, result, cloth_alpha, use_linear=use_linear)

    return result.astype(np.float32)
