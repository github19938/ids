"""Hair / portrait matting inference.

This module exposes a single function ``run_matting`` that returns a
continuous (float32, range [0, 1]) alpha matte the same size as the
input image.

The default implementation here is a *placeholder* so the pipeline can
be exercised end-to-end without a real model. Replace
``_PlaceholderMattingModel`` (or the body of ``run_matting``) with a
real ONNXRuntime / PyTorch session that loads your trained matting
network. The contract of ``run_matting`` is what the pipeline relies on.

Replaceable parts:
    * ``_PlaceholderMattingModel.predict`` -- swap with a real ONNX or
      PyTorch forward pass that produces an alpha matte. The expected
      output is a float32 array shaped ``(H, W)`` in ``[0, 1]`` (no
      thresholding, no binarisation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Real-model adapter (ONNX). Kept here as documentation / drop-in slot.
# ---------------------------------------------------------------------------


@dataclass
class OnnxMattingModel:
    """Thin wrapper around an ONNXRuntime matting session.

    Replace ``model_path`` with a real matting model (e.g. MODNet,
    RobustVideoMatting, BiSeNet hair-parsing turned matting, etc.).
    The wrapper assumes the network ingests an RGB float tensor in
    ``[0, 1]`` shaped ``(1, 3, H, W)`` and returns a single-channel
    alpha map in ``[0, 1]``.
    """

    model_path: str
    input_size: int = 512

    def __post_init__(self) -> None:
        import onnxruntime as ort  # local import so the placeholder works without ORT

        self._session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def predict(self, image_rgb_uint8: np.ndarray) -> np.ndarray:
        h, w = image_rgb_uint8.shape[:2]
        resized = cv2.resize(
            image_rgb_uint8, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA
        )
        tensor = resized.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        out = self._session.run([self._output_name], {self._input_name: tensor})[0]
        alpha = np.squeeze(out).astype(np.float32)
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(alpha, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Placeholder model so the pipeline runs without a real network.
# ---------------------------------------------------------------------------


class _PlaceholderMattingModel:
    """Deterministic, model-free placeholder.

    Produces a soft alpha matte from a luminance + saturation heuristic.
    It is *not* a real matting model; it exists so that
    ``apply_virtual_tryon`` is runnable end-to-end. Replace by an
    ``OnnxMattingModel`` (or any other real network) in production.
    """

    def predict(self, image_rgb_uint8: np.ndarray) -> np.ndarray:
        img = image_rgb_uint8.astype(np.float32) / 255.0
        hsv = cv2.cvtColor(image_rgb_uint8, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0

        darkness = 1.0 - img.mean(axis=2)
        saturation = hsv[..., 1]

        raw = 0.7 * darkness + 0.3 * saturation
        raw = cv2.GaussianBlur(raw, (0, 0), sigmaX=2.0, sigmaY=2.0)

        lo, hi = float(raw.min()), float(raw.max())
        if hi - lo < 1e-6:
            alpha = np.zeros_like(raw, dtype=np.float32)
        else:
            alpha = (raw - lo) / (hi - lo)

        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.5, sigmaY=1.5)
        return np.clip(alpha.astype(np.float32), 0.0, 1.0)


_DEFAULT_MODEL: Optional[object] = None


def _get_default_model() -> _PlaceholderMattingModel:
    global _DEFAULT_MODEL
    if _DEFAULT_MODEL is None:
        _DEFAULT_MODEL = _PlaceholderMattingModel()
    return _DEFAULT_MODEL  # type: ignore[return-value]


def run_matting(image: np.ndarray, model: Optional[object] = None) -> np.ndarray:
    """Run hair / portrait matting on ``image``.

    Parameters
    ----------
    image : np.ndarray
        RGB image, shape ``(H, W, 3)``, dtype ``uint8`` or float in
        ``[0, 1]``. The image is not modified.
    model : object, optional
        An object exposing ``predict(image_rgb_uint8) -> np.ndarray``.
        Defaults to the bundled placeholder. Pass an
        :class:`OnnxMattingModel` or any compatible adapter to swap in
        a real network.

    Returns
    -------
    np.ndarray
        Alpha matte, dtype ``float32``, shape ``(H, W)``, range ``[0, 1]``.
        The matte is *not* thresholded.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"run_matting expects an RGB image, got shape {image.shape}")

    if image.dtype == np.uint8:
        rgb_uint8 = image
    else:
        rgb_uint8 = np.clip(image.astype(np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)

    runner = model if model is not None else _get_default_model()
    alpha = runner.predict(rgb_uint8)
    alpha = np.asarray(alpha, dtype=np.float32)

    if alpha.shape[:2] != image.shape[:2]:
        alpha = cv2.resize(
            alpha, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR
        )

    return np.clip(alpha, 0.0, 1.0).astype(np.float32)
