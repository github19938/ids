"""Minimal runnable example for the matting-only virtual try-on pipeline.

This script fabricates synthetic inputs (a portrait-like image, a
T-shirt-like RGBA garment, and a face protection mask), runs
``apply_virtual_tryon``, and writes the result to ``output.png``.

Usage:
    python examples/run_minimal_example.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from virtual_tryon import apply_virtual_tryon  # noqa: E402


def _make_portrait(h: int = 512, w: int = 384) -> np.ndarray:
    """Synthetic 'portrait': pale background, oval face, dark hair on top."""
    img = np.full((h, w, 3), (235, 220, 210), dtype=np.uint8)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h * 0.42, w * 0.5
    ry, rx = h * 0.20, w * 0.18
    face = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 < 1.0
    img[face] = (225, 195, 175)

    hair_top = ((yy - (cy - ry * 0.3)) / (ry * 1.05)) ** 2 + ((xx - cx) / (rx * 1.25)) ** 2 < 1.0
    hair_top &= yy < cy + ry * 0.4
    img[hair_top] = (40, 30, 25)

    img = cv2.GaussianBlur(img, (0, 0), sigmaX=1.2, sigmaY=1.2)
    return img


def _make_garment(h: int = 512, w: int = 384) -> np.ndarray:
    """Synthetic RGBA T-shirt covering the lower 55% of the canvas."""
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]

    body = (yy > h * 0.45) & (np.abs(xx - w / 2) < w * 0.32)
    sleeves = (yy > h * 0.48) & (yy < h * 0.62) & (np.abs(xx - w / 2) < w * 0.46)
    cloth = body | sleeves

    rgba[cloth, :3] = (60, 110, 200)
    rgba[cloth, 3] = 255

    rgba[..., 3] = cv2.GaussianBlur(rgba[..., 3], (0, 0), sigmaX=1.5, sigmaY=1.5)
    return rgba


def _make_face_mask(h: int = 512, w: int = 384) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h * 0.42, w * 0.5
    ry, rx = h * 0.22, w * 0.20
    d = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2

    mask = np.clip(1.0 - d, 0.0, 1.0).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=4.0, sigmaY=4.0)
    return np.clip(mask, 0.0, 1.0)


def main() -> None:
    h, w = 512, 384
    image = _make_portrait(h, w)
    cloth_rgba = _make_garment(h, w)
    face_mask = _make_face_mask(h, w)

    result = apply_virtual_tryon(
        image=image,
        cloth_rgba=cloth_rgba,
        face_mask=face_mask,
        keypoints=None,
        use_linear=False,
    )

    out_uint8 = np.clip(result * 255.0, 0.0, 255.0).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_uint8, cv2.COLOR_RGB2BGR)
    out_path = os.path.join(os.path.dirname(__file__), "output.png")
    cv2.imwrite(out_path, out_bgr)
    print(f"Wrote {out_path} ({out_uint8.shape}, dtype={out_uint8.dtype})")


if __name__ == "__main__":
    main()
