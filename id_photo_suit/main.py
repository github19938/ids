"""main.py – CLI entry point for the id_photo_suit pipeline.

Examples
--------
    python -m id_photo_suit.main \
        --input ./examples/input.jpg \
        --template ./templates/default_suit \
        --output ./examples/output.jpg

    python -m id_photo_suit.main --list-templates ./templates
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from .pipeline import PipelineConfig, SuitPipeline
from .template_loader import list_templates


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="id_photo_suit",
        description="Automatic ID-photo suit replacement.",
    )
    p.add_argument("-i", "--input", help="Path to portrait image.")
    p.add_argument("-t", "--template", help="Path to template directory.")
    p.add_argument("-o", "--output", help="Path to write the final image.")
    p.add_argument("--background", help="Optional replacement background image.")

    p.add_argument("--scale-bias", type=float, default=None,
                   help="Multiplier on the auto-computed garment scale (default per-template).")
    p.add_argument("--y-offset", type=float, default=None,
                   help="Vertical offset as fraction of shoulder width (positive = down).")
    p.add_argument("--extra-scale", type=float, default=1.0)
    p.add_argument("--extra-angle", type=float, default=0.0,
                   help="Extra rotation in degrees applied around the shoulder midpoint.")
    p.add_argument("--extra-offset-x", type=float, default=0.0)
    p.add_argument("--extra-offset-y", type=float, default=0.0)

    p.add_argument("--max-edge", type=int, default=1280,
                   help="Down-scale long edge before processing (0 to disable).")
    p.add_argument("--no-color-match", action="store_true",
                   help="Disable luminance/color matching.")

    p.add_argument("--list-templates", metavar="ROOT",
                   help="List templates under ROOT and exit.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.list_templates:
        for path in list_templates(args.list_templates):
            print(path)
        return 0

    if not args.input or not args.template:
        print("error: --input and --template are required (or use --list-templates).", file=sys.stderr)
        return 2
    if not os.path.isfile(args.input):
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 2

    cfg = PipelineConfig()
    cfg.max_long_edge = args.max_edge
    if args.no_color_match:
        cfg.color.enable = False

    with SuitPipeline(cfg) as pipeline:
        out = pipeline.process_path(
            image_path=args.input,
            template_path=args.template,
            out_path=args.output,
            background_path=args.background,
            scale_bias=args.scale_bias,
            y_offset_ratio=args.y_offset,
            extra_scale=args.extra_scale,
            extra_angle_deg=args.extra_angle,
            extra_offset_px=(args.extra_offset_x, args.extra_offset_y),
        )

    if not args.output:
        # If no output path, drop next to the input.
        base, ext = os.path.splitext(args.input)
        fallback = f"{base}_suit{ext}"
        import cv2
        cv2.imwrite(fallback, out)
        print(fallback)
    else:
        print(args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
