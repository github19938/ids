"""segmentation.py – portrait region masks.

For a commercial-quality system we ultimately want a dedicated face-parsing
network (BiSeNet, MODNet, etc.) for hair/skin/clothes labels.  To keep the
out-of-the-box dependency footprint small *and* still produce usable masks
on CPU within the <1s budget, we combine two MediaPipe models:

* **SelfieSegmentation** – binary person / background mask.
* **FaceMesh**            – dense facial landmarks → face polygon, neck strip.

Hair is approximated as `person ∩ ¬face ∩ above_chin`.  Upper-body is
`person ∩ below_neck`.  These approximations are documented limitations
(see README); the module is structured so that `MaskBackend` is pluggable –
swap in an ONNX BiSeNet later without touching the rest of the pipeline.

Public API
----------
    seg = Segmenter()
    masks = seg.get_masks(bgr_image)   # PortraitMasks
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .types import PortraitMasks

logger = logging.getLogger(__name__)


# FaceMesh contour landmark indices (subset – the silhouette of the face).
# Source: MediaPipe FACEMESH_FACE_OVAL.
_FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]


@dataclass
class SegmenterConfig:
    person_threshold: float = 0.5
    feather_radius_px: int = 5
    # Vertical extent of the synthesized "neck strip" below the face oval,
    # expressed as a fraction of the face bounding-box height.
    neck_strip_height_ratio: float = 0.35
    # Horizontal width of the neck strip relative to face width.
    neck_strip_width_ratio: float = 0.55


class Segmenter:
    """Composite portrait segmenter (selfie + face-mesh)."""

    def __init__(self, config: Optional[SegmenterConfig] = None):
        self.cfg = config or SegmenterConfig()
        import mediapipe as mp

        self._mp = mp
        # model_selection=1 = "landscape" model, more accurate at portrait scale.
        self._selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=1
        )
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self) -> None:
        for obj in (self._selfie, self._face_mesh):
            try:
                obj.close()
            except Exception:  # pragma: no cover
                pass

    # ------------------------------------------------------------------
    def get_masks(self, bgr: np.ndarray) -> PortraitMasks:
        """Return :class:`PortraitMasks` for the given BGR image."""
        if bgr is None or bgr.size == 0:
            raise ValueError("segmentation.get_masks: empty image")
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(
                f"segmentation.get_masks: expected HxWx3 BGR, got {bgr.shape}"
            )

        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        person = self._person_mask(rgb)
        face_polygon = self._face_polygon(rgb)
        face_mask = self._polygon_mask((h, w), face_polygon) if face_polygon is not None else np.zeros((h, w), np.float32)

        # Hair = person ∧ ¬face ∧ above the chin line.
        chin_y = self._chin_y(face_polygon, default=h)
        above_chin = np.zeros((h, w), np.float32)
        if chin_y > 0:
            above_chin[:int(chin_y), :] = 1.0
        hair = np.clip(person * (1.0 - face_mask) * above_chin, 0.0, 1.0)

        # Neck = small trapezoidal strip directly below the face oval, intersected
        # with `person`.  Acts as a safety zone the renderer must NOT overwrite.
        neck_mask = self._neck_strip_mask((h, w), face_polygon)
        neck_mask = np.clip(neck_mask * person, 0.0, 1.0)

        # Upper body = person below the chin minus face/neck.
        below_chin = np.zeros((h, w), np.float32)
        below_chin[int(chin_y):, :] = 1.0
        upper_body = np.clip(
            person * below_chin * (1.0 - face_mask), 0.0, 1.0
        )

        if self.cfg.feather_radius_px > 0:
            r = self.cfg.feather_radius_px
            person = _feather(person, r)
            face_mask = _feather(face_mask, r)
            hair = _feather(hair, r)
            neck_mask = _feather(neck_mask, r)
            upper_body = _feather(upper_body, r)

        return PortraitMasks(
            person=person,
            face=face_mask,
            hair=hair,
            neck=neck_mask,
            upper_body=upper_body,
        )

    # ------------------------------------------------------------------
    def _person_mask(self, rgb: np.ndarray) -> np.ndarray:
        res = self._selfie.process(rgb)
        if res.segmentation_mask is None:
            logger.warning("SelfieSegmentation returned no mask; falling back to full frame.")
            return np.ones(rgb.shape[:2], np.float32)
        m = res.segmentation_mask.astype(np.float32)
        # MediaPipe returns per-pixel probability already in [0,1]; we still
        # apply a soft sigmoid-ish remap around the configured threshold for
        # cleaner edges.
        t = self.cfg.person_threshold
        m = np.clip((m - (t - 0.15)) / 0.30, 0.0, 1.0)
        return m

    def _face_polygon(self, rgb: np.ndarray):
        h, w = rgb.shape[:2]
        res = self._face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lms = res.multi_face_landmarks[0].landmark
        pts = np.array(
            [(lms[i].x * w, lms[i].y * h) for i in _FACE_OVAL],
            dtype=np.float32,
        )
        return pts

    @staticmethod
    def _polygon_mask(shape: Tuple[int, int], polygon: np.ndarray) -> np.ndarray:
        h, w = shape
        mask = np.zeros((h, w), np.float32)
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 1.0)
        return mask

    @staticmethod
    def _chin_y(polygon: Optional[np.ndarray], default: int) -> int:
        if polygon is None:
            return default
        return int(polygon[:, 1].max())

    def _neck_strip_mask(self, shape: Tuple[int, int], polygon: Optional[np.ndarray]) -> np.ndarray:
        h, w = shape
        if polygon is None:
            return np.zeros((h, w), np.float32)
        x0 = float(polygon[:, 0].min())
        x1 = float(polygon[:, 0].max())
        y0 = float(polygon[:, 1].min())
        y1 = float(polygon[:, 1].max())
        face_w = x1 - x0
        face_h = y1 - y0
        cx = 0.5 * (x0 + x1)
        # Trapezoid: a bit narrower than the face at the top, narrower still at the bottom.
        top_half_w = 0.5 * face_w * self.cfg.neck_strip_width_ratio
        bot_half_w = top_half_w * 0.85
        top_y = y1 - 0.05 * face_h            # slight overlap into the chin
        bot_y = y1 + face_h * self.cfg.neck_strip_height_ratio
        bot_y = min(bot_y, h - 1)
        pts = np.array(
            [
                (cx - top_half_w, top_y),
                (cx + top_half_w, top_y),
                (cx + bot_half_w, bot_y),
                (cx - bot_half_w, bot_y),
            ],
            dtype=np.int32,
        )
        mask = np.zeros((h, w), np.float32)
        cv2.fillPoly(mask, [pts], 1.0)
        return mask


def _feather(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    k = max(3, radius * 2 + 1)
    return cv2.GaussianBlur(mask, (k, k), sigmaX=radius)
