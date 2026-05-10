"""id_photo_suit – automatic ID photo suit replacement system.

Public API:
    from id_photo_suit.pipeline import SuitPipeline, run

The package is organized as:
    pose.py            – MediaPipe key-point extraction
    segmentation.py    – person / hair / face / neck masks
    template_loader.py – garment template loading & validation
    align.py           – affine transform from key-points → template
    render.py          – layered alpha-blended compositing
    color_match.py     – brightness / color harmonization
    pipeline.py        – orchestration
    main.py            – CLI entry point
"""

from .pipeline import SuitPipeline, run  # noqa: F401

__all__ = ["SuitPipeline", "run"]
__version__ = "0.1.0"
