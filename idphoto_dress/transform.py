"""Affine warp of the clothing template.

The clothing template is an RGBA PNG whose neck region has already been
cut out. The template carries metadata describing where its anchor points
sit (shoulders + neck center) so that we can compute the affine transform
that aligns it with the person.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class TemplateAnchors:
    """Anchor points on the clothing template (in template pixel space)."""

    left_shoulder: Tuple[float, float]
    right_shoulder: Tuple[float, float]
    neck_center: Tuple[float, float]

    @property
    def shoulder_width(self) -> float:
        lx, ly = self.left_shoulder
        rx, ry = self.right_shoulder
        return float(np.hypot(lx - rx, ly - ry))

    @property
    def shoulder_angle_deg(self) -> float:
        lx, ly = self.left_shoulder
        rx, ry = self.right_shoulder
        return float(np.degrees(np.arctan2(ry - ly, rx - lx)))


def default_anchors_for(template_rgba: np.ndarray) -> TemplateAnchors:
    """Best-effort default anchors when the template has no metadata.

    We assume the template is centered, fills the canvas horizontally, and
    that the shoulder line sits at roughly 22% of the template height with
    shoulders 8% inset from the left/right edges. The neck center sits on
    the same vertical line as the canvas center, slightly above the
    shoulder line.

    These constants are sane defaults for typical "整体服装模板" PNGs but
    callers should pass real anchors via :class:`TemplateAnchors` whenever
    possible.
    """
    h, w = template_rgba.shape[:2]
    sy = 0.22 * h
    lx = 0.08 * w
    rx = 0.92 * w
    return TemplateAnchors(
        left_shoulder=(lx, sy),
        right_shoulder=(rx, sy),
        neck_center=(0.5 * w, sy - 0.05 * h),
    )


@dataclass
class WarpResult:
    """Output of :func:`warp_template`.

    ``rgb`` and ``alpha`` are the same spatial size as the destination
    image (i.e. the original photo).
    """

    rgb: np.ndarray  # uint8, HxWx3
    alpha: np.ndarray  # uint8, HxW
    matrix: np.ndarray = field(default_factory=lambda: np.eye(2, 3))


def compute_affine(
    src_anchors: TemplateAnchors,
    dst_left_shoulder: Tuple[float, float],
    dst_right_shoulder: Tuple[float, float],
    dst_neck_center: Tuple[float, float],
) -> np.ndarray:
    """Compute a 2x3 affine matrix mapping template -> destination.

    The mapping is built from three correspondence points (left shoulder,
    right shoulder, neck center) which jointly encode scale + rotation +
    translation.
    """
    src = np.array(
        [
            src_anchors.left_shoulder,
            src_anchors.right_shoulder,
            src_anchors.neck_center,
        ],
        dtype=np.float32,
    )
    dst = np.array(
        [dst_left_shoulder, dst_right_shoulder, dst_neck_center],
        dtype=np.float32,
    )
    return cv2.getAffineTransform(src, dst)


def warp_template(
    template_rgba: np.ndarray,
    anchors: TemplateAnchors,
    dst_size: Tuple[int, int],
    dst_left_shoulder: Tuple[float, float],
    dst_right_shoulder: Tuple[float, float],
    dst_neck_center: Tuple[float, float],
) -> WarpResult:
    """Warp the RGBA template onto a canvas matching the input photo.

    Parameters
    ----------
    template_rgba:
        4-channel RGBA template image.
    anchors:
        Anchor coordinates in template pixel space.
    dst_size:
        ``(height, width)`` of the destination canvas.
    dst_left_shoulder, dst_right_shoulder, dst_neck_center:
        Target positions in destination pixel space.
    """
    if template_rgba.ndim != 3 or template_rgba.shape[2] != 4:
        raise ValueError("template_rgba must be an RGBA image (HxWx4).")

    h, w = dst_size
    matrix = compute_affine(
        anchors, dst_left_shoulder, dst_right_shoulder, dst_neck_center
    )

    rgb = template_rgba[:, :, :3]
    alpha = template_rgba[:, :, 3]

    warped_rgb = cv2.warpAffine(
        rgb,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped_alpha = cv2.warpAffine(
        alpha,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return WarpResult(rgb=warped_rgb, alpha=warped_alpha, matrix=matrix)


def clip_above_chin(
    alpha: np.ndarray, chin_y: Optional[float], feather: int = 9
) -> np.ndarray:
    """Force the template alpha to zero above ``chin_y``.

    This is the "防穿模" rule: clothing must never rise above the chin,
    even if the warped template extends further up.
    """
    if chin_y is None:
        return alpha
    out = alpha.copy()
    cut = max(0, min(out.shape[0], int(chin_y)))
    out[:cut] = 0
    if feather > 0 and cut < out.shape[0]:
        # Soft fade for the few rows just below the cut to avoid a hard
        # horizontal seam if the template happens to peek through.
        fade_h = min(feather * 2, out.shape[0] - cut)
        ramp = np.linspace(0, 1, fade_h, dtype=np.float32)[:, None]
        out[cut:cut + fade_h] = (
            out[cut:cut + fade_h].astype(np.float32) * ramp
        ).astype(np.uint8)
    return out
