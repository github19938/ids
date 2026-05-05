"""Lightweight brightness matching and shadow enhancement.

These steps are optional but help the swapped clothing feel like it
belongs in the original photograph.
"""

from __future__ import annotations

import cv2
import numpy as np


def match_brightness(
    template_bgr: np.ndarray,
    template_alpha: np.ndarray,
    reference_bgr: np.ndarray,
    reference_mask: np.ndarray,
    strength: float = 0.6,
) -> np.ndarray:
    """Shift the template's V channel toward the reference's mean V.

    We use HSV's V (value) channel to keep hue/saturation untouched and
    blend partially (``strength``) so the template still preserves its own
    visual identity.
    """
    if template_bgr.size == 0:
        return template_bgr

    template_hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV).astype(
        np.float32
    )
    ref_hsv = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    tpl_v = template_hsv[:, :, 2]
    ref_v = ref_hsv[:, :, 2]

    tpl_alpha = template_alpha > 16
    if not tpl_alpha.any():
        return template_bgr
    ref_valid = reference_mask > 16
    if not ref_valid.any():
        return template_bgr

    tpl_mean = float(tpl_v[tpl_alpha].mean())
    ref_mean = float(ref_v[ref_valid].mean())
    delta = (ref_mean - tpl_mean) * float(np.clip(strength, 0.0, 1.0))

    template_hsv[:, :, 2] = np.clip(tpl_v + delta, 0, 255)
    out = cv2.cvtColor(template_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def add_shadow(
    image_bgr: np.ndarray,
    template_alpha: np.ndarray,
    neck_center: tuple,
    radius: int = 60,
    intensity: float = 0.35,
) -> np.ndarray:
    """Darken a soft halo just below the chin to fake a neck-cast shadow.

    The shadow is a gaussian-blurred ellipse multiplied into the V channel
    only inside the clothing region (so we never darken the face).
    """
    h, w = image_bgr.shape[:2]
    cx, cy = int(neck_center[0]), int(neck_center[1])
    cy = min(h - 1, cy + radius // 3)

    shadow_mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(
        shadow_mask,
        (cx, cy),
        (radius, radius // 2),
        0,
        0,
        360,
        1.0,
        -1,
    )
    shadow_mask = cv2.GaussianBlur(
        shadow_mask, (radius | 1, radius | 1), radius / 3
    )

    cloth_alpha = template_alpha.astype(np.float32) / 255.0
    shadow_mask *= cloth_alpha
    shadow_mask *= float(np.clip(intensity, 0.0, 1.0))

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] *= 1.0 - shadow_mask
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
