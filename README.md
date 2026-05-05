# ID-Photo Clothing Swap

A modular Python pipeline for swapping clothing in ID-style portraits using
an **integral garment template** (a single transparent-background RGBA PNG
with the neck area cut out). The implementation focuses on **natural**
results — it is not a flat overlay.

## Pipeline

The end-to-end flow follows the spec exactly:

1. **Keypoints** — `MediaPipe Pose` provides `left_shoulder`,
   `right_shoulder`, and a synthesised `neck_center`.
2. **Alignment** — shoulder width (Euclidean distance) drives `scale`,
   shoulder line drives rotation, and `neck_center` drives translation.
3. **Template preprocessing** — the template's neck region is assumed to
   be already cut out; anchors are loaded from JSON or inferred.
4. **Affine warp** — a single `cv2.getAffineTransform` from three
   correspondence points encodes scale + rotation + translation.
5. **Person segmentation** — `MediaPipe Selfie Segmentation` produces the
   body mask. The clothing alpha is multiplied by it so it can never
   leak into the background.
6. **Alpha blending** — every mask edge is gaussian-blurred (`ksize ≥ 15`)
   before blending. No hard `np.where` overlays.
7. **Layer order** — background → clothing → original neck → original
   face → original hair (top-most).
8. **Anti-clipping** — the clothing alpha is forced to zero above the
   detected chin line, with a soft fade.
9. **Output** — naturally composited BGR image.

Optional bonus stages:

- **Brightness matching** (`color_match.match_brightness`) — shifts the
  template's HSV `V` toward the photo's mean.
- **Shadow enhancement** (`color_match.add_shadow`) — soft elliptical
  shadow under the chin to fake ambient occlusion.

## Layout

```
idphoto_dress/
├── __init__.py        # public API
├── keypoints.py       # MediaPipe Pose -> shoulders + neck
├── segmentation.py    # selfie + face + hair + neck masks
├── transform.py       # template anchors and affine warp
├── blend.py           # gaussian-feathered alpha compositing
├── color_match.py     # brightness + shadow bonus stages
└── pipeline.py        # orchestration
main.py                # CLI entry point
requirements.txt
```

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py \
    --photo portrait.jpg \
    --template outfit.png \
    --output result.png
```

Provide template anchors when available (recommended) — without them the
code falls back to a centered shoulder line at 22 % canvas height:

```bash
python main.py \
    --photo portrait.jpg \
    --template outfit.png \
    --anchors outfit.anchors.json \
    --output result.png
```

`outfit.anchors.json`:

```json
{
  "left_shoulder":  [120, 220],
  "right_shoulder": [560, 220],
  "neck_center":    [340, 170]
}
```

Coordinates are in template pixel space.

## Programmatic API

```python
import cv2
from idphoto_dress import ClothingTemplate, swap_clothing

photo = cv2.imread("portrait.jpg")
template = ClothingTemplate.from_path("outfit.png")
result = swap_clothing(photo, template)
cv2.imwrite("result.png", result)
```

## Notes

- Built with `OpenCV` + `NumPy` + `MediaPipe` only.
- All masks are uint8 in `[0, 255]`, all images are BGR uint8.
- The chin-line clip plus the person mask together guarantee the
  clothing template can never cover the face or escape the body
  silhouette.
