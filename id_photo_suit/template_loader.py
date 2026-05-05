"""template_loader.py – load and validate suit templates.

Template directory layout
-------------------------
    <template_root>/
        body.png        # required, RGBA
        collar.png      # optional, RGBA  (may be merged into body.png)
        config.json     # required

`config.json` schema (all coordinates in pixels of the template image)::

    {
      "name": "default_suit",
      "anchor_points": {
        "left_shoulder":  [x, y],
        "right_shoulder": [x, y],
        "neck":           [x, y]
      },
      "color_match": {
        "enable": true,
        "strength": 0.4
      },
      "render": {
        "feather_px": 6,
        "scale_bias": 1.0,
        "y_offset_ratio": 0.0
      }
    }

Public API
----------
    tpl = load_template("templates/default_suit")
    tpl.body         # HxWx4 uint8
    tpl.anchor_points["left_shoulder"]
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

import cv2
import numpy as np

from .types import GarmentTemplate

logger = logging.getLogger(__name__)


REQUIRED_ANCHORS = ("left_shoulder", "right_shoulder", "neck")


class TemplateError(ValueError):
    """Raised when a template directory is malformed."""


def load_template(path: str) -> GarmentTemplate:
    """Load a garment template from a directory.

    Parameters
    ----------
    path:
        Path to a directory containing ``body.png`` and ``config.json``.
    """
    if not os.path.isdir(path):
        raise TemplateError(f"Template path is not a directory: {path}")

    cfg_path = os.path.join(path, "config.json")
    body_path = os.path.join(path, "body.png")
    collar_path = os.path.join(path, "collar.png")

    if not os.path.isfile(cfg_path):
        raise TemplateError(f"Missing config.json in {path}")
    if not os.path.isfile(body_path):
        raise TemplateError(f"Missing body.png in {path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    name = cfg.get("name") or os.path.basename(os.path.normpath(path))
    anchors_raw = cfg.get("anchor_points") or {}
    for key in REQUIRED_ANCHORS:
        if key not in anchors_raw:
            raise TemplateError(
                f"config.json missing anchor_points['{key}'] for template '{name}'"
            )
    anchors = {k: (float(v[0]), float(v[1])) for k, v in anchors_raw.items()}

    body = _read_rgba(body_path)
    collar = _read_rgba(collar_path) if os.path.isfile(collar_path) else None

    if body.shape[2] != 4:
        raise TemplateError("body.png must be RGBA (4 channels).")
    if collar is not None and collar.shape[2] != 4:
        raise TemplateError("collar.png must be RGBA (4 channels).")

    # Sanity: anchors should fall inside the body image.
    h, w = body.shape[:2]
    for k, (x, y) in anchors.items():
        if not (0 <= x <= w and 0 <= y <= h):
            logger.warning(
                "Template '%s' anchor '%s' = (%.1f, %.1f) is outside body image (%d x %d)",
                name, k, x, y, w, h,
            )

    return GarmentTemplate(
        name=name,
        body=body,
        collar=collar,
        anchor_points=anchors,
        config=cfg,
    )


def list_templates(root: str) -> list:
    """Return every immediate sub-directory of `root` that looks like a template."""
    if not os.path.isdir(root):
        return []
    out = []
    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)
        if os.path.isdir(full) and os.path.isfile(os.path.join(full, "config.json")):
            out.append(full)
    return out


# ---------------------------------------------------------------------------
def _read_rgba(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise TemplateError(f"cv2.imread failed for {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img
