"""End-to-end clothing swap pipeline.

Layer order (bottom -> top), per the project requirements:

1. Background           -- the original photo's background
2. Clothing template    -- warped, alpha blended, masked by person
3. Original neck        -- pasted back from the source image
4. Original face        -- pasted back from the source image
5. Original hair        -- pasted back from the source image (top-most)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .blend import alpha_blend, feather_mask
from .color_match import add_shadow, match_brightness
from .keypoints import PersonKeypoints, detect_keypoints
from .segmentation import PersonMasks, compute_masks
from .transform import (
    TemplateAnchors,
    WarpResult,
    clip_above_chin,
    default_anchors_for,
    warp_template,
)


@dataclass
class ClothingTemplate:
    """An RGBA clothing template plus its anchor metadata."""

    rgba: np.ndarray
    anchors: TemplateAnchors

    @classmethod
    def from_path(
        cls,
        path: str,
        anchors: Optional[TemplateAnchors] = None,
    ) -> "ClothingTemplate":
        rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if rgba is None:
            raise FileNotFoundError(f"Could not read template at {path!r}")
        if rgba.ndim == 2 or rgba.shape[2] == 3:
            raise ValueError(
                "Clothing template must be an RGBA PNG with transparency."
            )
        if anchors is None:
            anchors = default_anchors_for(rgba)
        return cls(rgba=rgba, anchors=anchors)


# ---------------------------------------------------------------------------
# Pipeline steps (kept as small free functions for testability)
# ---------------------------------------------------------------------------

def _restrict_to_person(alpha: np.ndarray, person_mask: np.ndarray) -> np.ndarray:
    """Multiply the template alpha by the soft person mask.

    "上衣区域 = 人体mask" — clothing must never extend beyond the body.
    """
    pm = person_mask.astype(np.float32) / 255.0
    a = alpha.astype(np.float32) / 255.0
    out = (a * pm * 255.0).clip(0, 255).astype(np.uint8)
    return out


def _exclude(alpha: np.ndarray, *masks: np.ndarray) -> np.ndarray:
    """Subtract one or more masks from an alpha channel.

    Used to enforce 防穿模 rules: clothing must not cover face / hair /
    neck regions.
    """
    out = alpha.astype(np.float32)
    for m in masks:
        if m is None:
            continue
        m_f = m.astype(np.float32) / 255.0
        out *= 1.0 - m_f
    return out.clip(0, 255).astype(np.uint8)


def _build_clothing_layer(
    image_bgr: np.ndarray,
    template: ClothingTemplate,
    keypoints: PersonKeypoints,
    masks: PersonMasks,
    enable_color_match: bool,
    enable_shadow: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(clothing_rgb, clothing_alpha)`` ready for compositing."""
    h, w = image_bgr.shape[:2]
    warped: WarpResult = warp_template(
        template_rgba=template.rgba,
        anchors=template.anchors,
        dst_size=(h, w),
        dst_left_shoulder=keypoints.left_shoulder,
        dst_right_shoulder=keypoints.right_shoulder,
        dst_neck_center=keypoints.neck_center,
    )

    alpha = warped.alpha
    alpha = clip_above_chin(alpha, keypoints.chin_y, feather=11)
    alpha = _restrict_to_person(alpha, masks.person)
    # Don't paint over face / hair / neck — those layers will be added back
    # later, but keeping the alpha low here prevents bleed-through if the
    # subsequent layers' soft alpha doesn't fully cover.
    alpha = _exclude(alpha, masks.face, masks.hair)

    rgb = warped.rgb
    if enable_color_match:
        rgb = match_brightness(
            template_bgr=rgb,
            template_alpha=alpha,
            reference_bgr=image_bgr,
            reference_mask=masks.person,
            strength=0.55,
        )

    if enable_shadow:
        rgb = add_shadow(
            image_bgr=rgb,
            template_alpha=alpha,
            neck_center=keypoints.neck_center,
            radius=max(20, int(keypoints.shoulder_width * 0.25)),
            intensity=0.30,
        )

    alpha = feather_mask(alpha, ksize=21)
    return rgb, alpha


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def swap_clothing(
    image_bgr: np.ndarray,
    template: ClothingTemplate,
    enable_color_match: bool = True,
    enable_shadow: bool = True,
) -> np.ndarray:
    """Swap the clothing in ``image_bgr`` for ``template``.

    Parameters
    ----------
    image_bgr:
        Source ID-style portrait in OpenCV BGR ordering.
    template:
        :class:`ClothingTemplate` (RGBA + anchors).
    enable_color_match, enable_shadow:
        Toggles for the two optional bonus stages.

    Returns
    -------
    np.ndarray
        BGR uint8 image with the swapped clothing.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must be a 3-channel BGR image.")

    keypoints = detect_keypoints(image_bgr)
    masks = compute_masks(image_bgr, keypoints)

    cloth_rgb, cloth_alpha = _build_clothing_layer(
        image_bgr=image_bgr,
        template=template,
        keypoints=keypoints,
        masks=masks,
        enable_color_match=enable_color_match,
        enable_shadow=enable_shadow,
    )

    # Layer 1: background (original image, unchanged)
    output = image_bgr.copy()

    # Layer 2: clothing template
    output = alpha_blend(output, cloth_rgb, cloth_alpha)

    # Layer 3: original neck region — re-paste from the source image so
    # the neck is preserved exactly.
    neck_alpha = feather_mask(masks.neck, ksize=15)
    output = alpha_blend(output, image_bgr, neck_alpha)

    # Layer 4: original face
    face_alpha = feather_mask(masks.face, ksize=15)
    output = alpha_blend(output, image_bgr, face_alpha)

    # Layer 5: original hair (top-most)
    hair_alpha = feather_mask(masks.hair, ksize=21)
    output = alpha_blend(output, image_bgr, hair_alpha)

    return output
