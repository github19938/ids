"""Command-line entry point for the ID-photo clothing swap.

Example
-------
::

    python main.py \
        --photo path/to/portrait.jpg \
        --template path/to/outfit.png \
        --output result.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from idphoto_dress import ClothingTemplate, swap_clothing
from idphoto_dress.transform import TemplateAnchors, default_anchors_for


def _load_anchors(path: str | None, template_rgba) -> TemplateAnchors:
    if path is None:
        return default_anchors_for(template_rgba)
    data = json.loads(Path(path).read_text())
    return TemplateAnchors(
        left_shoulder=tuple(data["left_shoulder"]),
        right_shoulder=tuple(data["right_shoulder"]),
        neck_center=tuple(data["neck_center"]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Swap the clothing in an ID-photo with an RGBA template.",
    )
    parser.add_argument("--photo", required=True, help="Input portrait image.")
    parser.add_argument("--template", required=True, help="RGBA PNG outfit template.")
    parser.add_argument("--output", required=True, help="Where to write the result.")
    parser.add_argument(
        "--anchors",
        default=None,
        help="Optional JSON file with template anchors "
        "(keys: left_shoulder, right_shoulder, neck_center).",
    )
    parser.add_argument(
        "--no-color-match",
        action="store_true",
        help="Disable automatic brightness matching.",
    )
    parser.add_argument(
        "--no-shadow",
        action="store_true",
        help="Disable simulated neck shadow.",
    )
    args = parser.parse_args(argv)

    photo = cv2.imread(args.photo, cv2.IMREAD_COLOR)
    if photo is None:
        print(f"Could not read photo: {args.photo}", file=sys.stderr)
        return 2

    template_rgba = cv2.imread(args.template, cv2.IMREAD_UNCHANGED)
    if template_rgba is None:
        print(f"Could not read template: {args.template}", file=sys.stderr)
        return 2
    if template_rgba.ndim != 3 or template_rgba.shape[2] != 4:
        print(
            "Template must be an RGBA PNG with a transparent neck region.",
            file=sys.stderr,
        )
        return 2

    anchors = _load_anchors(args.anchors, template_rgba)
    template = ClothingTemplate(rgba=template_rgba, anchors=anchors)

    result = swap_clothing(
        image_bgr=photo,
        template=template,
        enable_color_match=not args.no_color_match,
        enable_shadow=not args.no_shadow,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), result):
        print(f"Failed to write output: {out_path}", file=sys.stderr)
        return 3
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
