"""ID-photo clothing swap pipeline.

A modular implementation that combines MediaPipe Pose / FaceMesh /
SelfieSegmentation with classic OpenCV affine warping and alpha blending
to produce a natural looking ID-photo with a swapped outfit.

Public entry point::

    from idphoto_dress import swap_clothing
"""

from .pipeline import swap_clothing, ClothingTemplate

__all__ = ["swap_clothing", "ClothingTemplate"]
