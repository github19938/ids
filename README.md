# Virtual Try-On (Matting-Only Pipeline)

A structurally-refactored virtual try-on project. The old rule-based
occlusion / ROI / direction-protection branches are gone; the only
implementation path now is **matting-driven real-alpha occlusion**.

## Layout

```
virtual_tryon/
    __init__.py          # exports apply_virtual_tryon
    pipeline.py          # the single entry point
    matting/
        __init__.py
        matting.py       # run_matting + ONNX/placeholder adapters
    compositing/
        __init__.py
        compositing.py   # alpha_blend, decontaminate, composite_pipeline
examples/
    run_minimal_example.py
requirements.txt
```

## Module responsibilities

| Module | Responsibility |
| --- | --- |
| `virtual_tryon.matting.matting` | `run_matting(image) -> alpha_hair`. Returns a continuous `float32` alpha matte in `[0, 1]`. The bundled `_PlaceholderMattingModel` is a heuristic stand-in; replace it with `OnnxMattingModel(model_path=...)` (also in this file) or any object exposing `predict(rgb_uint8) -> alpha`. |
| `virtual_tryon.compositing.compositing` | The three primitives required by the spec: `alpha_blend(fg, bg, alpha)`, `decontaminate(hair_rgb, alpha_hair)`, and `composite_pipeline(...)` for the layered render order. All blending is `out = fg*a + bg*(1-a)` with optional sRGB↔linear conversion. |
| `virtual_tryon.pipeline` | The single entry point `apply_virtual_tryon(image, cloth_rgba, face_mask, keypoints)`. Implements the fixed nine-step flow, no branching, no rule-based occlusion. |

## Public API

```python
from virtual_tryon import apply_virtual_tryon

result = apply_virtual_tryon(
    image=image,            # (H, W, 3) RGB uint8 or float32 in [0, 1]
    cloth_rgba=cloth_rgba,  # (Hc, Wc, 4) RGBA garment, resized inside
    face_mask=face_mask,    # (H, W) continuous face protection mask
    keypoints=None,         # accepted for API parity, currently unused
    use_linear=False,       # True => blend in linear light
)
# result: (H, W, 3) float32 in [0, 1]
```

## Algorithm (no branches, no modes)

1. `alpha_hair = run_matting(image)` — continuous float32 in `[0, 1]`.
2. `hair_rgb = image * alpha_hair` ; `background = image * (1 - alpha_hair)`.
3. `clean_hair = decontaminate(hair_rgb, alpha_hair)` — un-premultiply.
4. `cloth_alpha = cloth_rgba[..., 3] / 255`, resize + slight Gaussian blur.
5. `cloth_alpha *= (1 - face_mask)` (face protection).
6. `alpha_visible = alpha_hair * (1 - cloth_alpha)` — no thresholding, no `mask - mask`.
7. `result = alpha_blend(clean_hair, background, alpha_visible)`.
8. `result = alpha_blend(cloth_rgb, result, cloth_alpha)`.

Render order (fixed, used by `composite_pipeline`):

```
body -> neck -> face -> hair (alpha_visible) -> collar/cloth
```

## Replaceable parts (real model slots)

- `virtual_tryon/matting/matting.py`:
  - `OnnxMattingModel` — already wired; pass an ONNX file path to use a real matting net (MODNet, RVM, …).
  - `_PlaceholderMattingModel.predict` — replace with any real model that returns a `(H, W)` float alpha in `[0, 1]`.
- `virtual_tryon/pipeline.py` accepts a `matting_model=` argument so you can inject any object exposing `predict(rgb_uint8) -> alpha` without editing pipeline code.
- `face_mask` and `cloth_rgba` are inputs; the pipeline does **not** estimate them. Wire them up to your face parser / garment alpha producer of choice.

## Running the example

```bash
pip install -r requirements.txt
python examples/run_minimal_example.py
# writes examples/output.png
```

The example uses synthetic inputs and the placeholder matting model;
it exists to prove the wiring runs end-to-end.

## Numerical / colour conventions

- All alphas are `float32` in `[0, 1]`. Never thresholded.
- All composites use `alpha_blend` (`out = fg*a + bg*(1-a)`). No hard overlay anywhere.
- `use_linear=False` is the default (sRGB blend, simple mode).
- `use_linear=True` performs sRGB→linear, blend, linear→sRGB around every blend.
- No `pow(alpha, x)` "gamma fix" for grey halos — halos are handled by `decontaminate`.

## Acceptance / sanity checks

- No `roi`, `direction_weight`, `scheme_A`, mode-switch booleans.
- No `if/else` selecting a "scheme".
- Every composite goes through `alpha_blend`.
- No `alpha > 0.5` style thresholding.
- Halos around the hair are eliminated by un-premultiplying (`decontaminate`), not by gamma tweaks.
