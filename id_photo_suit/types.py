"""Shared dataclasses / type aliases for the id_photo_suit pipeline.

Keeping these in a single module avoids cyclic imports between the
pose / segmentation / align / render layers and gives every stage
a stable, documented data contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


Point2D = Tuple[float, float]


@dataclass
class PoseKeypoints:
    """Pixel-space landmarks extracted from a portrait.

    All coordinates are in image pixels (origin = top-left).
    `visibility` ∈ [0, 1] is MediaPipe's confidence score.

    `neck` is *derived* (not a raw landmark) as the midpoint of the two
    shoulders, optionally lifted toward the chin to better match where a
    collar should sit.  We expose it as a first-class field so downstream
    code never has to recompute it.
    """

    left_shoulder: Point2D
    right_shoulder: Point2D
    neck: Point2D
    chin: Optional[Point2D] = None
    nose: Optional[Point2D] = None
    visibility: float = 1.0
    image_size: Tuple[int, int] = (0, 0)  # (h, w)

    @property
    def shoulder_width(self) -> float:
        lx, ly = self.left_shoulder
        rx, ry = self.right_shoulder
        return float(np.hypot(lx - rx, ly - ry))

    @property
    def shoulder_angle_deg(self) -> float:
        """Rotation of the shoulder line in degrees (CCW positive)."""
        lx, ly = self.left_shoulder
        rx, ry = self.right_shoulder
        # We want the angle of the vector right→left (subject's perspective is mirrored,
        # but only the line orientation matters for our affine fit).
        return float(np.degrees(np.arctan2(ly - ry, lx - rx)))


@dataclass
class PortraitMasks:
    """Soft (float32, 0..1) masks describing portrait regions.

    Every mask is HxW float32 in [0, 1].  They are NOT mutually exclusive:
    `person` is the union of everything human; `face`, `hair`, `neck`,
    `upper_body` are sub-regions.
    """

    person: np.ndarray
    face: np.ndarray
    hair: np.ndarray
    neck: np.ndarray
    upper_body: np.ndarray

    def shape(self) -> Tuple[int, int]:
        return self.person.shape[:2]


@dataclass
class GarmentTemplate:
    """A loaded suit template.

    `body` and `collar` are RGBA uint8 images.  `anchor_points` describes
    where, in the template's own pixel coordinates, the left/right shoulder
    and neck sit.  These are the fiducials used by `align.py` to compute
    the affine transform that warps the template into the portrait.
    """

    name: str
    body: np.ndarray  # HxWx4 uint8
    collar: Optional[np.ndarray]  # HxWx4 uint8 or None
    anchor_points: Dict[str, Point2D]
    config: Dict = field(default_factory=dict)

    def required_anchors(self) -> Tuple[Point2D, Point2D, Point2D]:
        ls = self.anchor_points["left_shoulder"]
        rs = self.anchor_points["right_shoulder"]
        nk = self.anchor_points["neck"]
        return ls, rs, nk
