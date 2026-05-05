"""template_loader.py – load and validate suit templates.

Template directory layout
-------------------------
    <template_root>/
        body.png        # required, RGBA  (filename overridable via config.json)
        collar.png      # optional, RGBA  (may be merged into body.png)
        config.json     # required

`config.json` schema (all coordinates in pixels of the template image)::

    {
      "name": "default_suit",
      "body_image":   "21.png",     # optional override of body image filename
      "collar_image": "collar.png", # optional override; null/absent disables it
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

The two image filename keys (``body_image`` and ``collar_image``) are
optional.  When absent, the loader falls back to the historical
``body.png`` / ``collar.png`` filenames, so existing templates continue
to work unchanged.  Setting ``collar_image`` to ``null`` (or to an empty
string) explicitly disables the collar layer — useful when the body PNG
already contains the collar / tie / accessories baked in.

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
    if not os.path.isfile(cfg_path):
        raise TemplateError(f"Missing config.json in {path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    name = cfg.get("name") or os.path.basename(os.path.normpath(path))

    # Resolve the body / collar image filenames.  config.json may override
    # the historical defaults; when absent we fall back to body.png /
    # collar.png so existing templates keep working unchanged.
    body_filename = cfg.get("body_image") or "body.png"
    collar_filename = cfg.get("collar_image", "collar.png")

    body_path = os.path.join(path, body_filename)
    if not os.path.isfile(body_path):
        raise TemplateError(
            f"Missing body image '{body_filename}' in {path} "
            f"(set 'body_image' in config.json or place a body.png next to it)"
        )

    collar_path = (
        os.path.join(path, collar_filename) if collar_filename else None
    )

    anchors_raw = cfg.get("anchor_points") or {}
    for key in REQUIRED_ANCHORS:
        if key not in anchors_raw:
            raise TemplateError(
                f"config.json missing anchor_points['{key}'] for template '{name}'"
            )
    anchors = {k: (float(v[0]), float(v[1])) for k, v in anchors_raw.items()}

    body = _read_rgba(body_path)
    collar = (
        _read_rgba(collar_path)
        if collar_path and os.path.isfile(collar_path)
        else None
    )

    if body.shape[2] != 4:
        raise TemplateError(f"{body_filename} must be RGBA (4 channels).")
    if collar is not None and collar.shape[2] != 4:
        raise TemplateError(f"{collar_filename} must be RGBA (4 channels).")

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
