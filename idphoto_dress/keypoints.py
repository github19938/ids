"""Human keypoint detection.

Wraps MediaPipe Pose to extract the shoulder points and a synthetic
``neck_center`` keypoint, plus a chin Y coordinate that is used by the
anti-clipping rules.

The module is intentionally thin so that the rest of the pipeline can be
unit-tested with hand-crafted keypoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import mediapipe as mp  # type: ignore
except ImportError:  # pragma: no cover - mediapipe is a hard runtime dep
    mp = None  # type: ignore


@dataclass
class PersonKeypoints:
    """Keypoints needed for clothing alignment.

    All coordinates are in pixel space of the original input image.
    """

    left_shoulder: Tuple[float, float]
    right_shoulder: Tuple[float, float]
    neck_center: Tuple[float, float]
    chin_y: Optional[float] = None

    @property
    def shoulder_width(self) -> float:
        """Euclidean distance between the two shoulders, in pixels."""
        lx, ly = self.left_shoulder
        rx, ry = self.right_shoulder
        return float(np.hypot(lx - rx, ly - ry))

    @property
    def shoulder_angle_deg(self) -> float:
        """Angle of the shoulder line in degrees.

        Positive values mean the right shoulder is below the left shoulder
        in image coordinates (cv2-friendly).
        """
        lx, ly = self.left_shoulder
        rx, ry = self.right_shoulder
        return float(np.degrees(np.arctan2(ry - ly, rx - lx)))


def _to_pixels(landmark, w: int, h: int) -> Tuple[float, float]:
    return float(landmark.x * w), float(landmark.y * h)


def detect_keypoints(image_bgr: np.ndarray) -> PersonKeypoints:
    """Detect shoulder + neck keypoints from a BGR image.

    Parameters
    ----------
    image_bgr:
        Input image in OpenCV BGR ordering.

    Returns
    -------
    PersonKeypoints
        The detected keypoints in pixel coordinates.

    Raises
    ------
    RuntimeError
        If MediaPipe is not installed or no person is found.
    """
    if mp is None:
        raise RuntimeError(
            "mediapipe is required for keypoint detection. "
            "Install it via `pip install mediapipe`."
        )

    h, w = image_bgr.shape[:2]
    image_rgb = image_bgr[:, :, ::-1]

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.3,
    ) as pose:
        result = pose.process(image_rgb)

    if not result.pose_landmarks:
        raise RuntimeError("No person detected by MediaPipe Pose.")

    lm = result.pose_landmarks.landmark
    left = _to_pixels(lm[mp_pose.PoseLandmark.LEFT_SHOULDER], w, h)
    right = _to_pixels(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER], w, h)

    # MediaPipe Pose does not expose a neck landmark directly; we synthesise
    # one as the midpoint of the two shoulders, then nudge it slightly upward
    # toward the mouth landmark so it sits at the base of the neck rather
    # than the collarbone line.
    shoulder_mid_x = (left[0] + right[0]) / 2.0
    shoulder_mid_y = (left[1] + right[1]) / 2.0

    mouth_left = _to_pixels(lm[mp_pose.PoseLandmark.MOUTH_LEFT], w, h)
    mouth_right = _to_pixels(lm[mp_pose.PoseLandmark.MOUTH_RIGHT], w, h)
    mouth_mid_y = (mouth_left[1] + mouth_right[1]) / 2.0

    # Neck center sits ~30% of the way from the shoulder midpoint to the
    # mouth, which empirically lands around the bottom of the chin/top of
    # the neck for upright frontal portraits.
    neck_y = shoulder_mid_y + 0.30 * (mouth_mid_y - shoulder_mid_y)
    neck_center = (shoulder_mid_x, neck_y)

    chin_y = shoulder_mid_y + 0.55 * (mouth_mid_y - shoulder_mid_y)

    return PersonKeypoints(
        left_shoulder=left,
        right_shoulder=right,
        neck_center=neck_center,
        chin_y=chin_y,
    )
