"""tools/build_default_template.py – generate a synthetic default suit template.

We ship a programmatically-generated template so the project is *immediately*
runnable without licensed garment artwork.  It produces:

    templates/default_suit/body.png      # dark navy suit jacket, RGBA
    templates/default_suit/collar.png    # white shirt collar + V-neck, RGBA
    templates/default_suit/config.json   # anchor points + render config

The art is intentionally simple but has correctly-placed lapels, a V-neck
shirt, and a centered collar so the alignment math has meaningful fiducials
to lock onto.  Replace these PNGs with real photographic assets to reach
production quality without changing any code.
"""
from __future__ import annotations

import json
import os
from typing import Tuple

import cv2
import numpy as np


TEMPLATE_W = 1024
TEMPLATE_H = 1024

# All anchor points are in template pixel space.  Keep them consistent
# between body.png and collar.png — the renderer applies the SAME affine
# to both layers.
ANCHOR_LEFT_SHOULDER = (260, 360)
ANCHOR_RIGHT_SHOULDER = (764, 360)
ANCHOR_NECK = (512, 320)


def _new_canvas(w: int = TEMPLATE_W, h: int = TEMPLATE_H) -> np.ndarray:
    return np.zeros((h, w, 4), dtype=np.uint8)


def _draw_polygon(canvas: np.ndarray, pts, bgr, alpha: int = 255) -> None:
    pts = np.asarray(pts, dtype=np.int32)
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [pts], (*bgr, alpha))
    # Use the polygon's own alpha as the merge mask so we don't write over
    # already-drawn pixels with zero where we shouldn't.
    mask = np.zeros(canvas.shape[:2], np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    canvas[mask == 255] = overlay[mask == 255]


def _vignette(canvas: np.ndarray, polygon, dark_factor: float = 0.55) -> None:
    """Subtle inner shadow inside `polygon` for a touch of depth."""
    h, w = canvas.shape[:2]
    mask = np.zeros((h, w), np.float32)
    cv2.fillPoly(mask, [np.asarray(polygon, np.int32)], 1.0)
    edge = mask - cv2.erode(mask, np.ones((25, 25), np.uint8))
    edge = cv2.GaussianBlur(edge, (0, 0), sigmaX=10)
    edge = np.clip(edge, 0.0, 1.0)
    rgb = canvas[..., :3].astype(np.float32)
    rgb = rgb * (1.0 - edge[..., None] * (1.0 - dark_factor))
    canvas[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)


def build_body() -> np.ndarray:
    canvas = _new_canvas()

    # Suit color: charcoal-navy.  BGR.
    suit_bgr = (50, 38, 30)
    # Slightly lighter highlight for the lapel inside.
    lapel_bgr = (62, 46, 36)

    # Main jacket polygon.  Anchors at (260,360) and (764,360) define the
    # shoulder line; the body fans out below.
    jacket = [
        (ANCHOR_LEFT_SHOULDER[0] - 80, ANCHOR_LEFT_SHOULDER[1] + 0),   # outer left shoulder
        (ANCHOR_LEFT_SHOULDER[0] + 60, ANCHOR_LEFT_SHOULDER[1] - 40),  # inner left shoulder (top of lapel)
        (480, 470),                                                    # left chest, V notch
        (512, 700),                                                    # bottom mid
        (544, 470),                                                    # right chest, V notch
        (ANCHOR_RIGHT_SHOULDER[0] - 60, ANCHOR_RIGHT_SHOULDER[1] - 40),
        (ANCHOR_RIGHT_SHOULDER[0] + 80, ANCHOR_RIGHT_SHOULDER[1] + 0),
        (ANCHOR_RIGHT_SHOULDER[0] + 200, 1000),                        # bottom-right
        (ANCHOR_LEFT_SHOULDER[0] - 200, 1000),                         # bottom-left
    ]
    _draw_polygon(canvas, jacket, suit_bgr)

    # Left lapel (lighter notch).
    left_lapel = [
        (ANCHOR_LEFT_SHOULDER[0] + 60, ANCHOR_LEFT_SHOULDER[1] - 40),
        (480, 470),
        (510, 700),
        (470, 720),
        (380, 600),
        (380, 470),
    ]
    _draw_polygon(canvas, left_lapel, lapel_bgr)

    right_lapel = [
        (ANCHOR_RIGHT_SHOULDER[0] - 60, ANCHOR_RIGHT_SHOULDER[1] - 40),
        (544, 470),
        (514, 700),
        (554, 720),
        (644, 600),
        (644, 470),
    ]
    _draw_polygon(canvas, right_lapel, lapel_bgr)

    _vignette(canvas, jacket, dark_factor=0.7)

    # Soften the very bottom (it'll usually be cropped, but feather anyway).
    a = canvas[..., 3].astype(np.float32)
    fade = np.ones_like(a)
    fade[900:] = np.linspace(1.0, 0.0, a.shape[0] - 900)[:, None]
    canvas[..., 3] = (a * fade).astype(np.uint8)

    return canvas


def build_collar() -> np.ndarray:
    canvas = _new_canvas()

    shirt_bgr = (245, 245, 245)
    shadow_bgr = (210, 210, 210)
    tie_bgr = (40, 28, 90)  # burgundy-ish

    # White shirt V (visible in the lapel V).
    shirt = [
        (470, 360),
        (554, 360),
        (560, 480),
        (512, 560),
        (464, 480),
    ]
    _draw_polygon(canvas, shirt, shirt_bgr)

    # Collar wings (left + right) – two small triangles framing the V.
    left_collar = [
        (440, 350),
        (510, 360),
        (475, 470),
        (455, 460),
        (430, 400),
    ]
    right_collar = [
        (584, 350),
        (514, 360),
        (549, 470),
        (569, 460),
        (594, 400),
    ]
    _draw_polygon(canvas, left_collar, shirt_bgr)
    _draw_polygon(canvas, right_collar, shirt_bgr)

    # Subtle collar shadow on the inside edge.
    inside_shadow = [
        (490, 365),
        (534, 365),
        (524, 420),
        (500, 420),
    ]
    _draw_polygon(canvas, inside_shadow, shadow_bgr)

    # A simple knotted tie peeking out of the V.
    tie = [
        (498, 470),
        (526, 470),
        (532, 540),
        (492, 540),
    ]
    _draw_polygon(canvas, tie, tie_bgr)
    knot = [
        (496, 455),
        (528, 455),
        (524, 478),
        (500, 478),
    ]
    _draw_polygon(canvas, knot, (30, 22, 70))

    return canvas


def build_template(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    body = build_body()
    collar = build_collar()
    cv2.imwrite(os.path.join(out_dir, "body.png"), body)
    cv2.imwrite(os.path.join(out_dir, "collar.png"), collar)
    config = {
        "name": "default_suit",
        "description": "Procedurally-generated charcoal suit with white shirt and tie.",
        "anchor_points": {
            "left_shoulder": list(ANCHOR_LEFT_SHOULDER),
            "right_shoulder": list(ANCHOR_RIGHT_SHOULDER),
            "neck": list(ANCHOR_NECK),
        },
        "color_match": {
            "enable": True,
            "strength": 0.35,
            "match_chroma": False,
        },
        "render": {
            "feather_px": 6,
            "protect_dilate_px": 2,
            "scale_bias": 1.05,
            "y_offset_ratio": 0.02,
            "final_sharpen": 0.15,
        },
    }
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote template to {out_dir}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "templates", "default_suit")
    build_template(os.path.normpath(out))
