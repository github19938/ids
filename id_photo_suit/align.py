"""align.py – compute the affine transform that warps a template into the photo.

We solve for a 2×3 similarity transform (uniform scale + rotation + translation)
using three corresponding fiducials:

    template_anchors          ->  portrait_keypoints
        left_shoulder         ->  left_shoulder
        right_shoulder        ->  right_shoulder
        neck                  ->  neck

A similarity (rather than full affine or homography) is the right model
because we don't want shear distortions that would make a suit look melted.
We compute it in closed form by least-squares (`cv2.estimateAffinePartial2D`).
The returned matrix is then *shared* by both `body.png` and `collar.png`
so they remain registered to each other.

Public API
----------
    M, info = compute_transform(kp, template, scale_bias=1.0, y_offset_ratio=0.0)
    M : 2x3 float32 numpy array, suitable for cv2.warpAffine
    info: AlignmentInfo with debug fields (scale, angle, translation, etc.)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from .types import GarmentTemplate, PoseKeypoints

logger = logging.getLogger(__name__)


@dataclass
class AlignmentInfo:
    matrix: np.ndarray             # 2x3 float32
    scale: float                   # uniform scale factor applied
    rotation_deg: float            # degrees
    translation: Tuple[float, float]
    src_pts: np.ndarray            # template anchor points (3, 2) float32
    dst_pts: np.ndarray            # portrait key points    (3, 2) float32


def compute_transform(
    kp: PoseKeypoints,
    template: GarmentTemplate,
    scale_bias: float = 1.0,
    y_offset_ratio: float = 0.0,
    extra_scale: float = 1.0,
    extra_angle_deg: float = 0.0,
    extra_offset_px: Tuple[float, float] = (0.0, 0.0),
) -> AlignmentInfo:
    """Compute the affine transform for a template.

    Parameters
    ----------
    kp:
        Pose key-points extracted from the portrait.
    template:
        Loaded :class:`GarmentTemplate` with ``anchor_points``.
    scale_bias:
        Multiplier applied to the auto-computed scale.  Use values
        slightly > 1 (e.g. 1.05) to make the suit a touch wider than
        the bare shoulder distance, which tends to look more natural.
    y_offset_ratio:
        Vertical offset, expressed as a fraction of the shoulder width,
        applied AFTER the similarity solve.  Positive = move suit down.
    extra_scale, extra_angle_deg, extra_offset_px:
        Manual overrides for fine-tuning at the CLI level.
    """
    if kp.shoulder_width <= 1.0:
        raise ValueError(
            "Cannot align: detected shoulder width is degenerate "
            f"({kp.shoulder_width:.3f} px)."
        )

    src = np.array(
        [
            template.anchor_points["left_shoulder"],
            template.anchor_points["right_shoulder"],
            template.anchor_points["neck"],
        ],
        dtype=np.float32,
    )
    dst = np.array(
        [kp.left_shoulder, kp.right_shoulder, kp.neck], dtype=np.float32
    )

    # 1. Closed-form similarity (uniform scale + rotation + translation).
    M, _ = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.LMEDS, refineIters=10
    )
    if M is None:
        raise RuntimeError("estimateAffinePartial2D failed to find a transform.")
    M = M.astype(np.float32)

    # 2. Extract scale/angle from M for diagnostics and biasing.
    scale = float(np.hypot(M[0, 0], M[0, 1]))
    angle = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))

    # 3. Apply user biases.
    if scale_bias != 1.0 or extra_scale != 1.0:
        M = _rescale_about_dst_centroid(M, src, dst, scale_bias * extra_scale)
        scale *= scale_bias * extra_scale

    if extra_angle_deg != 0.0:
        cx, cy = float(dst.mean(axis=0)[0]), float(dst.mean(axis=0)[1])
        R = cv2.getRotationMatrix2D((cx, cy), extra_angle_deg, 1.0).astype(np.float32)
        M = _compose_2x3(R, M)
        angle += extra_angle_deg

    if y_offset_ratio != 0.0:
        M[1, 2] += y_offset_ratio * kp.shoulder_width
    if extra_offset_px != (0.0, 0.0):
        M[0, 2] += float(extra_offset_px[0])
        M[1, 2] += float(extra_offset_px[1])

    return AlignmentInfo(
        matrix=M,
        scale=scale,
        rotation_deg=angle,
        translation=(float(M[0, 2]), float(M[1, 2])),
        src_pts=src,
        dst_pts=dst,
    )


def warp_rgba(
    rgba: np.ndarray, M: np.ndarray, out_shape: Tuple[int, int]
) -> np.ndarray:
    """Affine-warp an RGBA image using bilinear interpolation, transparent border.

    `out_shape` is (h, w) of the destination canvas (= portrait size).
    """
    h, w = out_shape
    return cv2.warpAffine(
        rgba,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


# ---------------------------------------------------------------------------
def _rescale_about_dst_centroid(
    M: np.ndarray, src: np.ndarray, dst: np.ndarray, factor: float
) -> np.ndarray:
    """Multiply the scale of `M` by `factor`, keeping the dst centroid fixed."""
    cx, cy = float(dst.mean(axis=0)[0]), float(dst.mean(axis=0)[1])
    S = np.array(
        [[factor, 0.0, (1.0 - factor) * cx],
         [0.0, factor, (1.0 - factor) * cy]],
        dtype=np.float32,
    )
    return _compose_2x3(S, M)


def _compose_2x3(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return the 2x3 representing A ∘ B (apply B first, then A)."""
    A3 = np.vstack([A, [0.0, 0.0, 1.0]]).astype(np.float32)
    B3 = np.vstack([B, [0.0, 0.0, 1.0]]).astype(np.float32)
    C3 = A3 @ B3
    return C3[:2, :].astype(np.float32)
