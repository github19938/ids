"""tools/smoke_test.py – verify the pipeline end-to-end.

This script has two modes:

1. **Synthetic (default).**  Builds a fake portrait + manually crafted masks
   and key-points, then exercises *every deterministic stage* of the pipeline
   (align → warp → color_match → layered_render) WITHOUT relying on
   MediaPipe.  This is the right thing to run in CI: it doesn't depend on
   stochastic neural-network behaviour, and it validates that imports work
   and the math doesn't crash.

2. **Real (--real PATH).**  Runs the full pipeline on a real photograph.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))

from id_photo_suit.align import compute_transform, warp_rgba  # noqa: E402
from id_photo_suit.color_match import ColorMatchConfig, match_color  # noqa: E402
from id_photo_suit.pipeline import SuitPipeline  # noqa: E402
from id_photo_suit.render import RenderConfig, layered_render  # noqa: E402
from id_photo_suit.template_loader import load_template  # noqa: E402
from id_photo_suit.types import PortraitMasks, PoseKeypoints  # noqa: E402


def make_synthetic_portrait(w=720, h=960):
    img = np.full((h, w, 3), 220, np.uint8)
    cv2.rectangle(img, (w // 2 - 220, int(h * 0.55)), (w // 2 + 220, h), (180, 80, 80), -1)
    cv2.rectangle(img, (w // 2 - 50, int(h * 0.45)), (w // 2 + 50, int(h * 0.60)), (180, 175, 200), -1)
    cv2.ellipse(img, (w // 2, int(h * 0.32)), (140, 175), 0, 0, 360, (190, 195, 220), -1)
    cv2.ellipse(img, (w // 2, int(h * 0.22)), (150, 100), 0, 180, 360, (40, 40, 50), -1)
    cv2.circle(img, (w // 2 - 50, int(h * 0.32)), 10, (30, 30, 30), -1)
    cv2.circle(img, (w // 2 + 50, int(h * 0.32)), 10, (30, 30, 30), -1)
    cv2.line(img, (w // 2, int(h * 0.34)), (w // 2, int(h * 0.40)), (90, 90, 110), 3)
    cv2.ellipse(img, (w // 2, int(h * 0.44)), (35, 12), 0, 0, 180, (60, 30, 30), 3)
    return img


def make_synthetic_masks(w, h):
    """Build PortraitMasks consistent with `make_synthetic_portrait`."""
    person = np.zeros((h, w), np.float32)
    cv2.rectangle(person, (w // 2 - 230, int(h * 0.50)), (w // 2 + 230, h), 1.0, -1)
    cv2.ellipse(person, (w // 2, int(h * 0.32)), (160, 200), 0, 0, 360, 1.0, -1)
    person = cv2.GaussianBlur(person, (15, 15), 5)

    face = np.zeros((h, w), np.float32)
    cv2.ellipse(face, (w // 2, int(h * 0.34)), (110, 145), 0, 0, 360, 1.0, -1)

    hair = np.zeros((h, w), np.float32)
    cv2.ellipse(hair, (w // 2, int(h * 0.22)), (150, 100), 0, 180, 360, 1.0, -1)
    hair = np.clip(hair * (1.0 - face), 0, 1)

    neck = np.zeros((h, w), np.float32)
    cv2.rectangle(neck, (w // 2 - 55, int(h * 0.45)), (w // 2 + 55, int(h * 0.55)), 1.0, -1)

    upper = np.zeros((h, w), np.float32)
    cv2.rectangle(upper, (w // 2 - 230, int(h * 0.55)), (w // 2 + 230, h), 1.0, -1)

    return PortraitMasks(
        person=person,
        face=cv2.GaussianBlur(face, (9, 9), 3),
        hair=cv2.GaussianBlur(hair, (9, 9), 3),
        neck=cv2.GaussianBlur(neck, (9, 9), 3),
        upper_body=cv2.GaussianBlur(upper, (9, 9), 3),
    )


def make_synthetic_keypoints(w, h) -> PoseKeypoints:
    return PoseKeypoints(
        left_shoulder=(w // 2 - 180, int(h * 0.55)),
        right_shoulder=(w // 2 + 180, int(h * 0.55)),
        neck=(w // 2, int(h * 0.50)),
        chin=(w // 2, int(h * 0.46)),
        nose=(w // 2, int(h * 0.34)),
        visibility=1.0,
        image_size=(h, w),
    )


def run_deterministic() -> int:
    out_dir = os.path.normpath(os.path.join(HERE, "..", "examples"))
    os.makedirs(out_dir, exist_ok=True)
    portrait = make_synthetic_portrait()
    h, w = portrait.shape[:2]
    cv2.imwrite(os.path.join(out_dir, "synthetic_input.png"), portrait)

    template_path = os.path.normpath(os.path.join(HERE, "..", "templates", "default_suit"))
    template = load_template(template_path)
    print(
        f"Template '{template.name}' loaded: body={template.body.shape}, "
        f"collar={None if template.collar is None else template.collar.shape}"
    )

    kp = make_synthetic_keypoints(w, h)
    masks = make_synthetic_masks(w, h)

    t0 = time.perf_counter()
    align = compute_transform(kp, template, scale_bias=1.05, y_offset_ratio=0.02)
    body_warp = warp_rgba(template.body, align.matrix, (h, w))
    collar_warp = warp_rgba(template.collar, align.matrix, (h, w))

    body_warp = match_color(body_warp, portrait, masks.neck, ColorMatchConfig(strength=0.4))
    collar_warp = match_color(collar_warp, portrait, masks.neck, ColorMatchConfig(strength=0.2))

    out = layered_render(portrait, masks, body_warp, collar_warp, RenderConfig())
    dt = (time.perf_counter() - t0) * 1000.0

    out_path = os.path.join(out_dir, "synthetic_output.png")
    cv2.imwrite(out_path, out)
    print(f"OK: deterministic pipeline ran in {dt:.1f} ms.  Output: {out_path}")
    print(
        f"  shoulder_width={kp.shoulder_width:.1f} px  "
        f"scale={align.scale:.3f}  angle={align.rotation_deg:.2f}°  "
        f"translation=({align.translation[0]:.1f},{align.translation[1]:.1f})"
    )
    # Sanity checks
    assert out.shape == portrait.shape, "Output shape mismatch."
    assert out.dtype == np.uint8
    # The face region should be (almost) untouched.  Sample a center face pixel.
    fy, fx = int(h * 0.34), w // 2 - 30
    diff = int(np.abs(int(out[fy, fx, 0]) - int(portrait[fy, fx, 0])))
    print(f"  face-pixel BGR delta: {diff} (should be small; <40)")
    assert diff < 40, "Face region was painted over!"
    return 0


def run_real(image_path: str) -> int:
    template_path = os.path.normpath(os.path.join(HERE, "..", "templates", "default_suit"))
    out_path = os.path.splitext(image_path)[0] + "_suit.jpg"
    with SuitPipeline() as p:
        result = p.process_path(image_path, template_path, out_path=out_path)
    print(f"OK: real pipeline output -> {out_path}, shape={result.shape}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", help="Path to a real portrait to process instead of synthetic.")
    args = ap.parse_args()
    if args.real:
        return run_real(args.real)
    return run_deterministic()


if __name__ == "__main__":
    sys.exit(main())
