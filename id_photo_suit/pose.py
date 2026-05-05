"""pose.py – key-point extraction from a portrait image.

We use MediaPipe Pose for shoulder landmarks and (optionally) MediaPipe
FaceMesh for a chin reference.  Pose alone gives reliable shoulders;
the chin landmark from FaceMesh helps stabilize the synthesized "neck"
anchor when the subject's head is tilted.

Public API
----------
    extractor = PoseExtractor()
    kp = extractor.extract(bgr_image)   # -> PoseKeypoints | None
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .types import PoseKeypoints

logger = logging.getLogger(__name__)


# MediaPipe Pose landmark indices we care about.
_LM_NOSE = 0
_LM_LEFT_SHOULDER = 11
_LM_RIGHT_SHOULDER = 12
_LM_LEFT_EAR = 7
_LM_RIGHT_EAR = 8


@dataclass
class PoseExtractorConfig:
    model_complexity: int = 1            # 0=fastest, 1=balanced, 2=accurate
    min_detection_confidence: float = 0.5
    static_image_mode: bool = True
    use_face_mesh_for_chin: bool = True
    # Where to place the synthesized "neck" anchor along the segment from
    # shoulder-midpoint toward the chin.  0 = at shoulders, 1 = at chin.
    neck_lift_ratio: float = 0.35


class PoseExtractor:
    """Thin, lazy wrapper around MediaPipe Pose / FaceMesh.

    MediaPipe solution objects are heavyweight; we keep them alive for the
    lifetime of the extractor so repeated `extract()` calls amortize the
    construction cost.  Use `close()` (or a context manager) to release
    them when done.
    """

    def __init__(self, config: Optional[PoseExtractorConfig] = None):
        self.cfg = config or PoseExtractorConfig()
        # Imported lazily so `import id_photo_suit` doesn't pay MediaPipe's
        # startup cost just for, say, reading help text.
        import mediapipe as mp

        self._mp = mp
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=self.cfg.static_image_mode,
            model_complexity=self.cfg.model_complexity,
            enable_segmentation=False,
            min_detection_confidence=self.cfg.min_detection_confidence,
        )
        self._face_mesh = None
        if self.cfg.use_face_mesh_for_chin:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=self.cfg.static_image_mode,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=self.cfg.min_detection_confidence,
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self) -> None:
        try:
            self._pose.close()
        except Exception:  # pragma: no cover – defensive
            pass
        if self._face_mesh is not None:
            try:
                self._face_mesh.close()
            except Exception:  # pragma: no cover
                pass

    # ------------------------------------------------------------------
    def extract(self, bgr: np.ndarray) -> Optional[PoseKeypoints]:
        """Return PoseKeypoints in pixel coordinates, or None on failure."""
        if bgr is None or bgr.size == 0:
            raise ValueError("pose.extract: empty image")
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(f"pose.extract: expected HxWx3 BGR, got {bgr.shape}")

        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            logger.warning("MediaPipe Pose did not detect a person.")
            return None

        lms = result.pose_landmarks.landmark
        ls = lms[_LM_LEFT_SHOULDER]
        rs = lms[_LM_RIGHT_SHOULDER]
        nose = lms[_LM_NOSE]
        # MediaPipe is normalized (0..1); convert to pixel coords.
        ls_px = (ls.x * w, ls.y * h)
        rs_px = (rs.x * w, rs.y * h)
        nose_px = (nose.x * w, nose.y * h)

        chin_px = self._estimate_chin(rgb)
        neck_px = self._compute_neck(ls_px, rs_px, chin_px or nose_px)

        # Average visibility of the two shoulders is our overall confidence.
        vis = float((ls.visibility + rs.visibility) * 0.5)

        return PoseKeypoints(
            left_shoulder=ls_px,
            right_shoulder=rs_px,
            neck=neck_px,
            chin=chin_px,
            nose=nose_px,
            visibility=vis,
            image_size=(h, w),
        )

    # ------------------------------------------------------------------
    def _estimate_chin(self, rgb: np.ndarray):
        if self._face_mesh is None:
            return None
        h, w = rgb.shape[:2]
        res = self._face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None
        # Landmark 152 is the chin tip in MediaPipe FaceMesh.
        chin = res.multi_face_landmarks[0].landmark[152]
        return (chin.x * w, chin.y * h)

    def _compute_neck(self, ls, rs, head_ref):
        """Synthesize a neck anchor between shoulders, lifted toward the head."""
        mid = ((ls[0] + rs[0]) * 0.5, (ls[1] + rs[1]) * 0.5)
        if head_ref is None:
            return mid
        t = self.cfg.neck_lift_ratio
        return (
            mid[0] + (head_ref[0] - mid[0]) * t,
            mid[1] + (head_ref[1] - mid[1]) * t,
        )
