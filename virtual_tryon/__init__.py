"""Virtual try-on package (matting-only pipeline).

Public API:
    apply_virtual_tryon -- the single entry point for the entire pipeline.
"""

from .pipeline import apply_virtual_tryon

__all__ = ["apply_virtual_tryon"]
