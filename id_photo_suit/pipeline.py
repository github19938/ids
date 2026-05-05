"""pipeline.py – orchestrates pose → segmentation → align → render.

End-to-end usage::

    from id_photo_suit.pipeline import SuitPipeline
    pipeline = SuitPipeline()
    out_bgr = pipeline.process_path(
        "in.jpg",
        template_path="templates/default_suit",
        out_path="out.jpg",
    )

A class-based API is exposed so MediaPipe solutions are constructed once
and reused across calls (important for batch / server use).  A free
function ``run(image_path, template_path, out_path)`` is provided for
quick one-shot scripts and matches the spec in the task description.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

from .align import AlignmentInfo, compute_transform, warp_rgba
from .color_match import ColorMatchConfig, match_color
from .pose import PoseExtractor, PoseExtractorConfig
from .render import RenderConfig, layered_render
from .segmentation import Segmenter, SegmenterConfig
from .template_loader import load_template
from .types import GarmentTemplate, PortraitMasks, PoseKeypoints

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    pose: PoseExtractorConfig = field(default_factory=PoseExtractorConfig)
    seg: SegmenterConfig = field(default_factory=SegmenterConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    color: ColorMatchConfig = field(default_factory=ColorMatchConfig)
    # Garment placement biases (overridable per-call).
    scale_bias: float = 1.05
    y_offset_ratio: float = 0.02
    # If True, the pipeline downscales very large inputs to `max_long_edge`
    # before processing, then up-scales the result.  Helps hold the <1s budget.
    max_long_edge: int = 1280


@dataclass
class PipelineResult:
    image: np.ndarray              # final BGR uint8
    keypoints: PoseKeypoints
    masks: PortraitMasks
    alignment: AlignmentInfo
    elapsed_ms: float


class SuitPipeline:
    """Reusable orchestrator.  Construct once, call many times."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.cfg = config or PipelineConfig()
        self._pose = PoseExtractor(self.cfg.pose)
        self._seg = Segmenter(self.cfg.seg)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self) -> None:
        self._pose.close()
        self._seg.close()

    # ------------------------------------------------------------------
    def process(
        self,
        portrait_bgr: np.ndarray,
        template: GarmentTemplate,
        scale_bias: Optional[float] = None,
        y_offset_ratio: Optional[float] = None,
        extra_scale: float = 1.0,
        extra_angle_deg: float = 0.0,
        extra_offset_px: Tuple[float, float] = (0.0, 0.0),
        background_bgr: Optional[np.ndarray] = None,
    ) -> PipelineResult:
        """Run the full pipeline on an in-memory portrait."""
        t0 = time.perf_counter()

        portrait, scale_back = self._maybe_downscale(portrait_bgr)

        kp = self._pose.extract(portrait)
        if kp is None:
            raise RuntimeError("Pose detection failed: no person found in image.")

        masks = self._seg.get_masks(portrait)

        # Merge per-template render config overrides into the pipeline default.
        tcfg = template.config or {}
        render_cfg = self._compose_render_cfg(tcfg)
        color_cfg = self._compose_color_cfg(tcfg)

        sb = scale_bias if scale_bias is not None else self.cfg.scale_bias
        yor = y_offset_ratio if y_offset_ratio is not None else self.cfg.y_offset_ratio
        # Allow per-template defaults from config.json -> render.scale_bias / y_offset_ratio.
        rcfg = tcfg.get("render", {}) or {}
        if "scale_bias" in rcfg and scale_bias is None:
            sb = float(rcfg["scale_bias"])
        if "y_offset_ratio" in rcfg and y_offset_ratio is None:
            yor = float(rcfg["y_offset_ratio"])

        align = compute_transform(
            kp,
            template,
            scale_bias=sb,
            y_offset_ratio=yor,
            extra_scale=extra_scale,
            extra_angle_deg=extra_angle_deg,
            extra_offset_px=extra_offset_px,
        )

        h, w = portrait.shape[:2]
        body_warp = warp_rgba(template.body, align.matrix, (h, w))
        collar_warp = (
            warp_rgba(template.collar, align.matrix, (h, w))
            if template.collar is not None
            else None
        )

        # Color harmonize the warped suit toward the portrait's neck region.
        if color_cfg.enable:
            ref_mask = masks.neck if masks.neck.sum() > 50 else masks.upper_body
            body_warp = match_color(body_warp, portrait, ref_mask, color_cfg)
            if collar_warp is not None:
                collar_warp = match_color(collar_warp, portrait, ref_mask, color_cfg)

        composite = layered_render(
            portrait_bgr=portrait,
            masks=masks,
            body_rgba=body_warp,
            collar_rgba=collar_warp,
            cfg=render_cfg,
            background_bgr=background_bgr,
        )

        if scale_back is not None:
            composite = cv2.resize(
                composite, scale_back, interpolation=cv2.INTER_CUBIC
            )

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Pipeline done in %.1f ms (template=%s, shoulder_w=%.1f px, scale=%.3f, angle=%.2f°)",
            elapsed, template.name, kp.shoulder_width, align.scale, align.rotation_deg,
        )
        return PipelineResult(
            image=composite,
            keypoints=kp,
            masks=masks,
            alignment=align,
            elapsed_ms=elapsed,
        )

    # ------------------------------------------------------------------
    def process_path(
        self,
        image_path: str,
        template_path: str,
        out_path: Optional[str] = None,
        background_path: Optional[str] = None,
        **kwargs,
    ) -> np.ndarray:
        portrait = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if portrait is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        template = load_template(template_path)

        bg = None
        if background_path:
            bg = cv2.imread(background_path, cv2.IMREAD_COLOR)
            if bg is None:
                raise FileNotFoundError(f"Cannot read background: {background_path}")
            bg = cv2.resize(bg, (portrait.shape[1], portrait.shape[0]))

        result = self.process(portrait, template, background_bgr=bg, **kwargs)

        if out_path:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
            cv2.imwrite(out_path, result.image)
        return result.image

    # ------------------------------------------------------------------
    def _maybe_downscale(self, img: np.ndarray):
        h, w = img.shape[:2]
        long_edge = max(h, w)
        if self.cfg.max_long_edge <= 0 or long_edge <= self.cfg.max_long_edge:
            return img, None
        scale = self.cfg.max_long_edge / float(long_edge)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return small, (w, h)  # original size to up-scale back to

    def _compose_render_cfg(self, tcfg: dict) -> RenderConfig:
        rc = self.cfg.render
        rcfg = tcfg.get("render", {}) or {}
        return RenderConfig(
            feather_px=int(rcfg.get("feather_px", rc.feather_px)),
            protect_dilate_px=int(rcfg.get("protect_dilate_px", rc.protect_dilate_px)),
            suit_alpha_floor=float(rcfg.get("suit_alpha_floor", rc.suit_alpha_floor)),
            final_sharpen=float(rcfg.get("final_sharpen", rc.final_sharpen)),
        )

    def _compose_color_cfg(self, tcfg: dict) -> ColorMatchConfig:
        cc = self.cfg.color
        ccfg = tcfg.get("color_match", {}) or {}
        return ColorMatchConfig(
            enable=bool(ccfg.get("enable", cc.enable)),
            strength=float(ccfg.get("strength", cc.strength)),
            match_chroma=bool(ccfg.get("match_chroma", cc.match_chroma)),
            match_chroma_strength=float(
                ccfg.get("match_chroma_strength", cc.match_chroma_strength)
            ),
            match_std=bool(ccfg.get("match_std", cc.match_std)),
        )


# ---------------------------------------------------------------------------
def run(image_path: str, template_path: str, out_path: Optional[str] = None, **kwargs) -> np.ndarray:
    """One-shot convenience wrapper matching the original task spec."""
    with SuitPipeline() as p:
        return p.process_path(image_path, template_path, out_path=out_path, **kwargs)
