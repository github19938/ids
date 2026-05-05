"""Person / face / hair segmentation utilities.

* ``person_mask`` is produced by MediaPipe Selfie Segmentation. It is used
  to constrain the clothing template so it never covers the background.
* ``face_mask`` and ``hair_mask`` come from MediaPipe FaceMesh + a hair
  heuristic on top of the person mask. They are used to re-composite the
  original face / hair on top of the swapped clothing.
* ``neck_mask`` is the area between the chin line and the shoulders that
  must be re-pasted from the original image so the neck stays natural.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

try:
    import mediapipe as mp  # type: ignore
except ImportError:  # pragma: no cover
    mp = None  # type: ignore

from .keypoints import PersonKeypoints


@dataclass
class PersonMasks:
    """A bundle of binary / soft masks for the subject.

    All masks are ``uint8`` arrays with values in ``[0, 255]`` and the same
    spatial size as the source image.
    """

    person: np.ndarray
    face: np.ndarray
    hair: np.ndarray
    neck: np.ndarray


# ---------------------------------------------------------------------------
# Selfie segmentation
# ---------------------------------------------------------------------------

def _person_mask(image_bgr: np.ndarray) -> np.ndarray:
    if mp is None:
        raise RuntimeError(
            "mediapipe is required for selfie segmentation. "
            "Install it via `pip install mediapipe`."
        )

    image_rgb = image_bgr[:, :, ::-1]
    mp_selfie = mp.solutions.selfie_segmentation
    with mp_selfie.SelfieSegmentation(model_selection=1) as seg:
        result = seg.process(image_rgb)
    if result.segmentation_mask is None:
        raise RuntimeError("Selfie segmentation failed to produce a mask.")

    soft = result.segmentation_mask  # float32 in [0, 1]
    mask = (soft * 255.0).clip(0, 255).astype(np.uint8)
    # Clean up isolated pixels but keep a soft boundary for blending.
    _, binary = cv2.threshold(mask, 80, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    return binary


# ---------------------------------------------------------------------------
# Face mesh based face / hair masks
# ---------------------------------------------------------------------------

# Indices on the MediaPipe FaceMesh "FACEMESH_FACE_OVAL" loop.
_FACE_OVAL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]


def _face_oval_polygon(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    if mp is None:
        return None
    h, w = image_bgr.shape[:2]
    image_rgb = image_bgr[:, :, ::-1]
    mp_face = mp.solutions.face_mesh
    with mp_face.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.3,
    ) as fm:
        result = fm.process(image_rgb)
    if not result.multi_face_landmarks:
        return None
    lm = result.multi_face_landmarks[0].landmark
    pts = np.array(
        [[lm[i].x * w, lm[i].y * h] for i in _FACE_OVAL_IDX],
        dtype=np.float32,
    )
    return pts


def _polygon_mask(shape, polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
    return mask


def _hair_mask_heuristic(
    person_mask: np.ndarray,
    face_mask: np.ndarray,
    chin_y: Optional[float],
) -> np.ndarray:
    r"""Approximate the hair region as ``person \ face`` above the chin.

    This isn't perfect but is enough to keep the hair on top of the
    swapped clothing (which is the only constraint we need).
    """
    h, w = person_mask.shape[:2]
    above = np.zeros_like(person_mask)
    cut_y = int(chin_y) if chin_y is not None else h // 2
    cut_y = max(0, min(h, cut_y))
    above[:cut_y] = 255

    hair = cv2.bitwise_and(person_mask, above)
    hair = cv2.subtract(hair, face_mask)
    hair = cv2.morphologyEx(hair, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return hair


def _neck_mask(
    person_mask: np.ndarray,
    face_mask: np.ndarray,
    keypoints: PersonKeypoints,
) -> np.ndarray:
    """Strip of the original image between chin and shoulder line.

    This is what we re-paste on top of the swapped clothing so the neck
    doesn't get covered by the template.
    """
    h, w = person_mask.shape[:2]
    chin_y = int(keypoints.chin_y if keypoints.chin_y is not None else 0)
    shoulder_y = int(
        max(keypoints.left_shoulder[1], keypoints.right_shoulder[1])
    )
    chin_y = max(0, min(h, chin_y))
    shoulder_y = max(0, min(h, shoulder_y))
    if shoulder_y <= chin_y:
        shoulder_y = min(h, chin_y + 5)

    band = np.zeros_like(person_mask)
    band[chin_y:shoulder_y, :] = 255

    neck = cv2.bitwise_and(person_mask, band)
    neck = cv2.subtract(neck, face_mask)
    return neck


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_masks(
    image_bgr: np.ndarray, keypoints: PersonKeypoints
) -> PersonMasks:
    """Compute every mask the pipeline needs in a single pass."""
    person = _person_mask(image_bgr)

    polygon = _face_oval_polygon(image_bgr)
    if polygon is not None:
        face = _polygon_mask(image_bgr.shape, polygon)
    else:
        # Fallback: synthesise a face ellipse from keypoints.
        face = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        cx = int((keypoints.left_shoulder[0] + keypoints.right_shoulder[0]) / 2)
        cy = int(keypoints.chin_y or keypoints.neck_center[1] - 30)
        rx = int(keypoints.shoulder_width * 0.35)
        ry = int(keypoints.shoulder_width * 0.5)
        cv2.ellipse(face, (cx, cy - ry // 2), (rx, ry), 0, 0, 360, 255, -1)

    hair = _hair_mask_heuristic(person, face, keypoints.chin_y)
    neck = _neck_mask(person, face, keypoints)

    return PersonMasks(person=person, face=face, hair=hair, neck=neck)
