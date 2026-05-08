# -*- coding: utf-8 -*-
"""
add_neck_source.py
==================
完全独立的「真实脖子移植」类。

从原图（带真实脖子的完整人像）把脖子区域 affine 对齐后像素级移植到
抠图头像（RGBA，透明背景）上，最大限度还原原图的脖子颜色 + 纹理 + 光照。

**零项目依赖**：仅需 ``numpy``, ``opencv-python``, ``mediapipe``，可直接复制到
任何项目独立运行。

用法::

    from add_neck_source import NeckSourceTransplant

    nst = NeckSourceTransplant()
    result = nst.run("head_rgba.png", "source_photo.jpg")
    nst.save("output.png")

    # 或传入内存中的 BGRA 数组
    head_bgra = cv2.imread("head.png", cv2.IMREAD_UNCHANGED)
    source_bgr = cv2.imread("source.jpg")
    result = nst.run(head_bgra, source_bgr)

与 ``add_neckv2.py --source-image`` 的对应关系：
    本类是 transplant 成功路径的独立实现（affine warp + 像素替换 + 轻度颜色对齐），
    不含 Phase B（纹理合成）/ Phase D（Laplacian 金字塔）/ Phase E 完整 CDF 匹配，
    因为它直接用原图真实像素替换合成结果，无需纹理合成步骤。
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from collections import deque
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

# =====================================================================================
# 常量
# =====================================================================================

LANDMARK_CHIN_BOTTOM = 152
LANDMARK_LEFT_JAW_ON_OVAL = 172
LANDMARK_RIGHT_JAW_ON_OVAL = 397

SKIN_PATCH_CHEEK_PX = 10
SKIN_PATCH_OTHER_PX = 10

SKIN_SAMPLE_REGIONS: List[Tuple[int, float, str, int]] = [
    (205, 0.25, "L-cheek", SKIN_PATCH_CHEEK_PX),
    (425, 0.25, "R-cheek", SKIN_PATCH_CHEEK_PX),
    (164, 0.20, "philtrum", SKIN_PATCH_OTHER_PX),
    (10, 0.20, "forehead", SKIN_PATCH_OTHER_PX),
    (200, 0.10, "above-chin", SKIN_PATCH_OTHER_PX),
]

SKIN_YCRCB_CR_MIN, SKIN_YCRCB_CR_MAX = 133, 173
SKIN_YCRCB_CB_MIN, SKIN_YCRCB_CB_MAX = 77, 127
SKIN_FILTER_MIN_COUNT = 4

NECK_SLIM_SCALE_DEFAULT = 0.87
NECK_TOP_INSET_DEFAULT = 0.78
NECK_MASK_SUPERSAMPLE = 2

JAW_SPAN_OVERLAP_FRAC = 0.06
JAW_SPAN_OVERLAP_MIN_PX = 6.0
JAW_SPAN_OVERLAP_MAX_PX = 60.0
JAW_SPAN_DEPTH_FRAC = 1.4
JAW_SPAN_DEPTH_MIN_PX = 36.0

NECK_FEATHER_FRAC = 0.025
NECK_FEATHER_MIN_PX = 1.5
NECK_FEATHER_MAX_PX = 16.0
NECK_SUPPRESS_DECAY_FRAC = 0.06
NECK_SUPPRESS_DECAY_MIN_PX = 4.0
NECK_SUPPRESS_ALPHA_GAMMA = 0.55

POSE_PITCH_CHIN_OVERLAP_GAIN = 0.55
POSE_PITCH_CHIN_OVERLAP_MIN = 0.55
POSE_PITCH_CHIN_OVERLAP_MAX = 1.55
POSE_YAW_ASYM_INSET_GAIN = 0.30
POSE_ROLL_MAX_RAD = 0.78

HEAD_POSE_LANDMARKS = (1, 152, 33, 263, 61, 291)
HEAD_POSE_CANONICAL_3D = np.array([
    [0.0,    0.0,    0.0],
    [0.0,   63.6,  -12.5],
    [-43.3, -32.7, -26.0],
    [43.3,  -32.7, -26.0],
    [-28.9,  28.9, -24.1],
    [28.9,   28.9, -24.1],
], dtype=np.float64)

TRANSPLANT_ANCHOR_LANDMARKS = (152, 172, 397)
TRANSPLANT_TONE_MATCH_STRENGTH = 0.30

COLLAR_DETECT_MIN_FRAC = 0.30
COLLAR_DEPTH_SHRINK_MIN_PX = 12.0
COLLAR_PROBE_HALF_WIDTH_FRAC = 0.18
COLLAR_PROBE_MAX_FRAC = 2.0
COLLAR_ALPHA_THRESHOLD = 128

# MediaPipe Selfie Multiclass Segmenter（CPU 上的"皮肤/衣服/头发"语义分割）
# 类别：0=background, 1=hair, 2=body-skin, 3=face-skin, 4=clothes, 5=others/accessories
SELFIE_MULTICLASS_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)
SELFIE_MULTICLASS_FILENAME = "selfie_multiclass_256x256.tflite"
SELFIE_CLASS_BACKGROUND = 0
SELFIE_CLASS_HAIR = 1
SELFIE_CLASS_BODY_SKIN = 2
SELFIE_CLASS_FACE_SKIN = 3
SELFIE_CLASS_CLOTHES = 4
SELFIE_CLASS_OTHERS = 5


# =====================================================================================
# IO
# =====================================================================================

def imread_unicode(path: str, flags: int = cv2.IMREAD_UNCHANGED) -> Optional[np.ndarray]:
    path = os.path.normpath(path)
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as e:
        raise FileNotFoundError(f"无法打开文件: {path}") from e
    if not raw:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, flags)


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    path = os.path.normpath(path)
    ext = os.path.splitext(path)[1].lower() or ".png"
    if not ext.startswith("."):
        ext = "." + ext
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True


def load_rgba(path: str) -> np.ndarray:
    img = imread_unicode(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"需要彩色图（3 或 4 通道），当前 shape={img.shape}")
    if img.shape[2] == 3:
        bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = 255
        return bgra
    return img


def bgra_to_rgb(bgra: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2RGB)


def alpha_over(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    b = bottom.astype(np.float32) / 255.0
    t = top.astype(np.float32) / 255.0
    ba = b[..., 3:4]
    ta = t[..., 3:4]
    out_a = ta + ba * (1.0 - ta)
    out_a_safe = np.maximum(out_a, 1e-6)
    out_rgb = t[..., :3] * ta + b[..., :3] * ba * (1.0 - ta)
    out_rgb = out_rgb / out_a_safe
    out = np.dstack([out_rgb, out_a])
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)


# =====================================================================================
# Landmark / Graph
# =====================================================================================

def landmark_xy(landmark, width: int, height: int) -> Tuple[float, float]:
    return float(landmark.x * width), float(landmark.y * height)


def _face_oval_graph() -> dict:
    graph: dict = {}
    for a, b in mp.solutions.face_mesh.FACEMESH_FACE_OVAL:
        graph.setdefault(a, []).append(b)
        graph.setdefault(b, []).append(a)
    return graph


def bfs_path_on_graph(graph: dict, start: int, end: int) -> Optional[List[int]]:
    if start == end:
        return [start]
    q = deque([start])
    parent: dict = {start: None}
    while q:
        u = q.popleft()
        if u == end:
            break
        for v in graph.get(u, ()):
            if v not in parent:
                parent[v] = u
                q.append(v)
    if end not in parent:
        return None
    path: List[int] = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def jaw_index_path_through_chin() -> List[int]:
    g = _face_oval_graph()
    p_l = bfs_path_on_graph(g, LANDMARK_LEFT_JAW_ON_OVAL, LANDMARK_CHIN_BOTTOM)
    p_r = bfs_path_on_graph(g, LANDMARK_CHIN_BOTTOM, LANDMARK_RIGHT_JAW_ON_OVAL)
    if p_l is None or p_r is None:
        return [172, 136, 148, 152, 377, 397]
    if p_l[-1] != LANDMARK_CHIN_BOTTOM or p_r[0] != LANDMARK_CHIN_BOTTOM:
        return [172, 136, 148, 152, 377, 397]
    return p_l[:-1] + p_r


# =====================================================================================
# 肤色采样
# =====================================================================================

def skin_patch_rect_at(cx: float, cy: float, ih: int, iw: int, patch: int) -> Tuple[int, int, int, int]:
    patch = int(patch)
    half = patch // 2
    ix, iy = int(round(cx)), int(round(cy))
    x0 = int(np.clip(ix - half, 0, max(0, iw - patch)))
    y0 = int(np.clip(iy - half, 0, max(0, ih - patch)))
    pw = min(patch, iw - x0)
    ph = min(patch, ih - y0)
    return x0, y0, pw, ph


def _patch_skin_alpha_mask(roi_bgra: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    am = roi_bgra[:, :, 3].astype(np.float32) > 40.0
    bgr = roi_bgra[:, :, :3]
    if bgr.size == 0:
        return am, np.zeros_like(am, dtype=bool)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    skin = (
        (cr >= SKIN_YCRCB_CR_MIN) & (cr <= SKIN_YCRCB_CR_MAX) &
        (cb >= SKIN_YCRCB_CB_MIN) & (cb <= SKIN_YCRCB_CB_MAX)
    )
    return am, am & skin


def median_bgr_in_patch(bgra: np.ndarray, x0: int, y0: int, pw: int, ph: int) -> Optional[np.ndarray]:
    if pw <= 0 or ph <= 0:
        return None
    roi = bgra[y0: y0 + ph, x0: x0 + pw]
    am, sm = _patch_skin_alpha_mask(roi)
    if not np.any(am):
        return None
    use = sm if int(np.sum(sm)) >= SKIN_FILTER_MIN_COUNT else am
    flat = roi[:, :, :3][use].astype(np.float64)
    if flat.shape[0] == 0:
        return None
    return np.median(flat, axis=0)


def _apply_neck_skin_tone(bgr: np.ndarray, v_scale: float, h_shift: float = 0.0, s_scale: float = 1.0) -> np.ndarray:
    px = np.clip(np.round(bgr).astype(np.uint8).reshape(1, 1, 3), 0, 255)
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[0, 0, 0] = (hsv[0, 0, 0] + float(h_shift)) % 180.0
    hsv[0, 0, 1] = float(np.clip(hsv[0, 0, 1] * float(s_scale), 0.0, 255.0))
    hsv[0, 0, 2] = float(np.clip(hsv[0, 0, 2] * float(v_scale), 0.0, 255.0))
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)[0, 0].astype(np.float64)


def sample_skin_color_bgra(
    bgra: np.ndarray, landmarks, w: int, h: int,
    skin_v_scale: float = 0.92, skin_h_shift: float = 0.0, skin_s_scale: float = 1.0,
) -> np.ndarray:
    skin_v_scale = float(np.clip(skin_v_scale, 0.85, 1.05))
    skin_h_shift = float(np.clip(skin_h_shift, -6.0, 6.0))
    skin_s_scale = float(np.clip(skin_s_scale, 0.90, 1.20))
    lm = landmarks.landmark
    weighted: List[Tuple[float, np.ndarray]] = []
    for lid, wt, _, psize in SKIN_SAMPLE_REGIONS:
        cx, cy = landmark_xy(lm[lid], w, h)
        x0, y0, pw, ph = skin_patch_rect_at(cx, cy, h, w, psize)
        med = median_bgr_in_patch(bgra, x0, y0, pw, ph)
        if med is not None:
            weighted.append((float(wt), med.astype(np.float64)))
    if not weighted:
        return _apply_neck_skin_tone(
            np.array([180.0, 200.0, 220.0], dtype=np.float64),
            skin_v_scale, skin_h_shift, skin_s_scale,
        )
    sw = sum(w for w, _ in weighted)
    raw = sum(w * c for w, c in weighted) / max(sw, 1e-9)
    return _apply_neck_skin_tone(raw, skin_v_scale, skin_h_shift, skin_s_scale)


def _gather_face_skin_pixels_bgr(bgra: np.ndarray, landmarks, ih: int, iw: int, patch: int = 22) -> np.ndarray:
    """从 5 个肤色采样点附近收集肤色像素池，供颜色匹配作 ref。"""
    lm = landmarks.landmark
    chunks: List[np.ndarray] = []
    for lid, _, _, _ in SKIN_SAMPLE_REGIONS:
        cx, cy = landmark_xy(lm[lid], iw, ih)
        x0, y0, pw, ph = skin_patch_rect_at(cx, cy, ih, iw, patch)
        if pw <= 0 or ph <= 0:
            continue
        roi = bgra[y0: y0 + ph, x0: x0 + pw]
        am, sm = _patch_skin_alpha_mask(roi)
        use = sm if int(np.sum(sm)) >= SKIN_FILTER_MIN_COUNT else am
        if np.any(use):
            chunks.append(roi[:, :, :3][use])
    if not chunks:
        return np.empty((0, 3), dtype=np.uint8)
    return np.vstack(chunks)


# =====================================================================================
# 几何 / 多边形
# =====================================================================================

def odd_kernel(size: int) -> int:
    return size if size % 2 == 1 else size + 1


def smooth_polyline_xy(pts: np.ndarray, win: int = 5) -> np.ndarray:
    n = pts.shape[0]
    if n < 3:
        return pts
    win = odd_kernel(min(max(3, win), max(3, n - (1 - (n % 2)))))
    pad = win // 2
    k = np.ones(win, dtype=np.float64) / win
    xs = np.convolve(np.pad(pts[:, 0], (pad, pad), mode="edge"), k, mode="valid")
    ys = np.convolve(np.pad(pts[:, 1], (pad, pad), mode="edge"), k, mode="valid")
    return np.column_stack([xs, ys])


def build_jaw_top_polyline(
    landmarks, w: int, h: int, chin_overlap_px: float, top_inset: float = 1.0,
) -> np.ndarray:
    lm = landmarks.landmark
    idxs = jaw_index_path_through_chin()
    top = np.array([landmark_xy(lm[i], w, h) for i in idxs], dtype=np.float64)
    top = smooth_polyline_xy(top, win=5)
    if float(top_inset) < 0.999:
        cx_top = float(np.mean(top[:, 0]))
        top[:, 0] = cx_top + (top[:, 0] - cx_top) * float(top_inset)
    top[:, 1] -= float(chin_overlap_px)
    top[:, 0] = np.clip(top[:, 0], 0.0, float(w - 1))
    top[:, 1] = np.clip(top[:, 1], 0.0, float(h - 1))
    return top


def build_jaw_guided_neck_polygon(
    landmarks, w: int, h: int, chin_overlap_px: float, neck_depth_px: float,
    bottom_flare: float, top_inset: float = 1.0,
) -> np.ndarray:
    lm = landmarks.landmark
    idxs = jaw_index_path_through_chin()
    top = np.array([landmark_xy(lm[i], w, h) for i in idxs], dtype=np.float64)
    top = smooth_polyline_xy(top, win=5)
    if float(top_inset) < 0.999:
        cx_top = float(np.mean(top[:, 0]))
        top[:, 0] = cx_top + (top[:, 0] - cx_top) * float(top_inset)
    top[:, 1] -= float(chin_overlap_px)
    cx = float(np.mean(top[:, 0]))
    bot_y = float(np.max(top[:, 1])) + float(neck_depth_px)
    bottom = np.column_stack([
        cx + (top[:, 0] - cx) * float(bottom_flare),
        np.full(len(top), bot_y, dtype=np.float64),
    ])
    n = len(top)
    u = np.linspace(-1.0, 1.0, n)
    sag = min(6.0, neck_depth_px * 0.08)
    bottom[:, 1] += sag * (1.0 - u * u)
    poly = np.vstack([top, bottom[::-1]])
    poly[:, 0] = np.clip(poly[:, 0], 0.0, float(w - 1))
    poly[:, 1] = np.clip(poly[:, 1], 0.0, float(h - 1))
    return poly


def fill_polygon_mask(h: int, w: int, poly: np.ndarray, supersample: int = NECK_MASK_SUPERSAMPLE) -> np.ndarray:
    if supersample <= 1:
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.round(poly).astype(np.int32).reshape(1, -1, 2)
        cv2.fillPoly(mask, pts, 255, lineType=cv2.LINE_AA)
        return mask
    H2 = h * int(supersample)
    W2 = w * int(supersample)
    mask_big = np.zeros((H2, W2), dtype=np.uint8)
    pts_big = np.round(poly * float(supersample)).astype(np.int32).reshape(1, -1, 2)
    cv2.fillPoly(mask_big, pts_big, 255, lineType=cv2.LINE_AA)
    return cv2.resize(mask_big, (w, h), interpolation=cv2.INTER_AREA)


def rotate_xy(pts: np.ndarray, cx: float, cy: float, angle: float) -> np.ndarray:
    if abs(float(angle)) < 1e-4:
        return pts
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    dx = pts[:, 0] - float(cx)
    dy = pts[:, 1] - float(cy)
    nx = float(cx) + dx * c - dy * s
    ny = float(cy) + dx * s + dy * c
    return np.column_stack([nx, ny])


def distance_map_to_polyline(h: int, w: int, pts: np.ndarray) -> np.ndarray:
    if pts.shape[0] < 2:
        return np.full((h, w), 1e6, dtype=np.float64)
    img = np.ones((h, w), dtype=np.uint8) * 255
    pi = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pi], isClosed=False, color=0, thickness=2, lineType=cv2.LINE_AA)
    return cv2.distanceTransform(img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE).astype(np.float64)


# =====================================================================================
# Pose
# =====================================================================================

def estimate_head_pose(landmarks, w: int, h: int) -> Tuple[float, float, float]:
    lm = landmarks.landmark
    le = landmark_xy(lm[33], w, h)
    re = landmark_xy(lm[263], w, h)
    eye_dx = re[0] - le[0]
    eye_dy = re[1] - le[1]
    simple_roll = float(np.arctan2(eye_dy, eye_dx))

    image_pts = np.array([landmark_xy(lm[i], w, h) for i in HEAD_POSE_LANDMARKS], dtype=np.float64)
    K = np.array([
        [float(w), 0.0, float(w) * 0.5],
        [0.0, float(w), float(h) * 0.5],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)
    yaw = 0.0
    pitch = 0.0
    try:
        success, rvec, _tvec = cv2.solvePnP(
            HEAD_POSE_CANONICAL_3D, image_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        success = False
    if success:
        Rmat, _ = cv2.Rodrigues(rvec)
        sy = float(np.sqrt(Rmat[0, 0] ** 2 + Rmat[1, 0] ** 2))
        if sy > 1e-6:
            pnp_pitch = float(np.arctan2(Rmat[2, 1], Rmat[2, 2]))
            pnp_yaw = float(np.arctan2(-Rmat[2, 0], sy))
            pnp_roll = float(np.arctan2(Rmat[1, 0], Rmat[0, 0]))
        else:
            pnp_pitch = float(np.arctan2(-Rmat[1, 2], Rmat[1, 1]))
            pnp_yaw = float(np.arctan2(-Rmat[2, 0], sy))
            pnp_roll = 0.0
        yaw = pnp_yaw
        pitch = -pnp_pitch
        if abs(pnp_roll - simple_roll) > 0.175:
            yaw *= 0.5
            pitch *= 0.5

    return (
        float(np.clip(yaw, -POSE_ROLL_MAX_RAD, POSE_ROLL_MAX_RAD)),
        float(np.clip(pitch, -POSE_ROLL_MAX_RAD, POSE_ROLL_MAX_RAD)),
        float(np.clip(simple_roll, -POSE_ROLL_MAX_RAD, POSE_ROLL_MAX_RAD)),
    )


# =====================================================================================
# 已有脖子/衣领检测
# =====================================================================================

def estimate_existing_neck_extent_px(bgra: np.ndarray, chin_x: float, chin_y: float, jaw_span_px: float) -> float:
    h, w = bgra.shape[:2]
    half = max(int(jaw_span_px * COLLAR_PROBE_HALF_WIDTH_FRAC), 4)
    x0 = max(0, int(round(chin_x - half)))
    x1 = min(w, int(round(chin_x + half)))
    y_top = max(0, int(round(chin_y)))
    y_bot = min(h, int(round(chin_y + jaw_span_px * COLLAR_PROBE_MAX_FRAC)))
    if x1 <= x0 or y_bot <= y_top + 1:
        return 0.0
    strip_alpha = bgra[y_top:y_bot, x0:x1, 3]
    row_med = np.median(strip_alpha, axis=1)
    above = row_med > float(COLLAR_ALPHA_THRESHOLD)
    if above.size == 0 or not bool(above[0]):
        return 0.0
    if bool(above.all()):
        return float(above.size)
    return float(int(np.argmin(above)))


# =====================================================================================
# Face detection
# =====================================================================================

def _make_face_mesh(
    static_image_mode: bool = True, max_num_faces: int = 1,
    refine_landmarks: bool = True, min_detection_confidence: float = 0.4,
    min_tracking_confidence: float = 0.4,
):
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=static_image_mode,
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def _detect_first_face(face_mesh, rgb: np.ndarray):
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None
    return res.multi_face_landmarks[0]


def detect_face_with_retry(rgb: np.ndarray):
    h_img, w_img = rgb.shape[:2]
    with _make_face_mesh() as fm:
        landmarks = _detect_first_face(fm, rgb)
    if landmarks is not None:
        return landmarks

    longer = max(h_img, w_img)
    if longer < 1024:
        scale = 1024.0 / float(longer)
        rh = int(round(h_img * scale))
        rw = int(round(w_img * scale))
        rgb_up = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_LINEAR)
        with _make_face_mesh() as fm:
            landmarks = _detect_first_face(fm, rgb_up)
        if landmarks is not None:
            return landmarks

    with _make_face_mesh(min_detection_confidence=0.2) as fm:
        landmarks = _detect_first_face(fm, rgb)
    if landmarks is not None:
        return landmarks

    return None


# =====================================================================================
# Transplant 核心
# =====================================================================================

def _load_source_image_and_detect(source_image_path: str):
    """加载 source 图并检测人脸；返回 (bgra, landmarks) 或 (None, None)。"""
    if not source_image_path:
        return None, None
    try:
        src = imread_unicode(source_image_path)
    except (FileNotFoundError, OSError, ValueError):
        return None, None
    if src is None or src.ndim != 3:
        return None, None
    if src.shape[2] == 3:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2BGRA)
        src[:, :, 3] = 255
    elif src.shape[2] != 4:
        return None, None
    src_rgb = cv2.cvtColor(src[:, :, :3], cv2.COLOR_BGR2RGB)
    src_landmarks = detect_face_with_retry(src_rgb)
    if src_landmarks is None:
        return None, None
    return src, src_landmarks


def _compute_affine_matrix(
    src_landmarks, src_w: int, src_h: int,
    head_landmarks, head_w: int, head_h: int,
) -> Optional[np.ndarray]:
    head_pts = np.array(
        [landmark_xy(head_landmarks.landmark[i], head_w, head_h) for i in TRANSPLANT_ANCHOR_LANDMARKS],
        dtype=np.float32,
    )
    src_pts = np.array(
        [landmark_xy(src_landmarks.landmark[i], src_w, src_h) for i in TRANSPLANT_ANCHOR_LANDMARKS],
        dtype=np.float32,
    )
    v1 = head_pts[1] - head_pts[0]
    v2 = head_pts[2] - head_pts[0]
    if abs(v1[0] * v2[1] - v1[1] * v2[0]) < 1.0:
        return None
    v1s = src_pts[1] - src_pts[0]
    v2s = src_pts[2] - src_pts[0]
    if abs(v1s[0] * v2s[1] - v1s[1] * v2s[0]) < 1.0:
        return None
    try:
        return cv2.getAffineTransform(src_pts, head_pts)
    except cv2.error:
        return None


def _affine_pts(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=pts.dtype)])
    return (M @ pts_h.T).T


CLOTHING_DETECT_TOP_SKIP_FRAC = 0.12
CLOTHING_DETECT_LAB_THRESHOLD = 70.0
CLOTHING_DETECT_CONSECUTIVE_ROWS = 3


def detect_clothing_boundary(
    source_bgra: np.ndarray, source_landmarks,
    top_polyline_s: np.ndarray, neck_depth_s: float,
    chin_s_x: float, chin_s_y: float,
) -> float:
    sH, sW = source_bgra.shape[:2]
    lm = source_landmarks.landmark

    ref_pixels = _gather_face_skin_pixels_bgr(source_bgra, source_landmarks, sH, sW, patch=22)
    if ref_pixels.shape[0] < 8:
        return neck_depth_s
    ref_lab = cv2.cvtColor(ref_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)
    ref_mean = ref_lab.mean(axis=0)

    top_y = float(np.min(top_polyline_s[:, 1]))
    bot_y = top_y + neck_depth_s
    half_w = max(float(np.max(top_polyline_s[:, 0]) - np.min(top_polyline_s[:, 0])) * 0.40, sW * 0.02)

    y_start = int(top_y + (bot_y - top_y) * CLOTHING_DETECT_TOP_SKIP_FRAC)
    y_end = int(bot_y)
    x_left = max(int(chin_s_x - half_w), 0)
    x_right = min(int(chin_s_x + half_w), sW)

    if y_start >= y_end - 2 or x_right <= x_left + 5:
        return neck_depth_s

    roi = source_bgra[y_start:y_end, x_left:x_right]
    roi_lab = cv2.cvtColor(roi[:, :, :3], cv2.COLOR_BGR2LAB).astype(np.float64)
    am = roi[:, :, 3] > 40

    rows = roi_lab.shape[0]
    dists = np.zeros(rows, dtype=np.float64)
    for r in range(rows):
        row_pix = roi_lab[r, am[r]]
        if row_pix.shape[0] < 5:
            dists[r] = 999.0
            continue
        row_mean = row_pix.mean(axis=0)
        dL = row_mean[0] - ref_mean[0]
        da = row_mean[1] - ref_mean[1]
        db = row_mean[2] - ref_mean[2]
        dists[r] = np.sqrt(dL * dL + da * da + db * db)

    # 找到第一段连续 >= CONSECUTIVE_ROWS 行都超过阈值的位置（衣领开始处）
    # 把多边形截断到这段连续区间的"起点"，而不是结束点；并按"相对 top_y 的距离"计算
    # 真实的脖子高度（depth = 检测到的 y - top_y），避免把衣物像素卷入 warp 区。
    span = CLOTHING_DETECT_CONSECUTIVE_ROWS
    for r in range(span - 1, rows):
        if np.all(dists[r - span + 1: r + 1] > CLOTHING_DETECT_LAB_THRESHOLD):
            boundary_y = float(y_start + r - (span - 1))   # 衣领过渡起点 y
            truncated_depth = max(boundary_y - top_y, 20.0)
            return min(truncated_depth, neck_depth_s)

    return neck_depth_s


def _reinhard_lab_mean_shift(
    layer_bgra: np.ndarray,
    mask_bool: np.ndarray,
    ref_pixels_bgr: np.ndarray,
    strength: float,
) -> None:
    """简单 Lab mean-shift 颜色迁移：仅做 L 通道均值对齐 + a/b 通道均值对齐。"""
    if ref_pixels_bgr.shape[0] < 8 or not np.any(mask_bool):
        return
    bgr = layer_bgra[:, :, :3]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    src = lab[mask_bool]
    if src.shape[0] < 8:
        return
    ref_lab = cv2.cvtColor(
        ref_pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB
    ).reshape(-1, 3).astype(np.float32)
    s = float(np.clip(strength, 0.0, 1.0))
    L_shift = (ref_lab[:, 0].mean() - src[:, 0].mean()) * s
    a_shift = (ref_lab[:, 1].mean() - src[:, 1].mean()) * s
    b_shift = (ref_lab[:, 2].mean() - src[:, 2].mean()) * s
    lab[mask_bool, 0] = np.clip(src[:, 0] + L_shift, 0.0, 255.0)
    lab[mask_bool, 1] = np.clip(src[:, 1] + a_shift, 0.0, 255.0)
    lab[mask_bool, 2] = np.clip(src[:, 2] + b_shift, 0.0, 255.0)
    layer_bgra[:, :, :3] = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


# =====================================================================================
# Selfie Multiclass Segmenter — 用 CPU 模型做精准的皮肤/衣服分割
# =====================================================================================

def _default_model_cache_dir() -> str:
    """模型缓存目录：脚本同目录的 ``models/`` 优先；否则 ``~/.cache/add_neck_source``。"""
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, "models")
    try:
        os.makedirs(candidate, exist_ok=True)
        if os.access(candidate, os.W_OK):
            return candidate
    except OSError:
        pass
    home = os.path.expanduser("~")
    fallback = os.path.join(home, ".cache", "add_neck_source")
    os.makedirs(fallback, exist_ok=True)
    return fallback


def _ensure_selfie_multiclass_model(
    model_path: Optional[str] = None, timeout: float = 30.0,
) -> Optional[str]:
    """确保 ``selfie_multiclass_256x256.tflite`` 模型存在，必要时联网下载。

    返回模型本地绝对路径；下载失败时返回 ``None``。"""
    if model_path:
        if os.path.exists(model_path):
            return os.path.abspath(model_path)
    target_dir = _default_model_cache_dir()
    target = os.path.join(target_dir, SELFIE_MULTICLASS_FILENAME)
    if os.path.exists(target) and os.path.getsize(target) > 100 * 1024:
        return target
    try:
        print(f"[selfie] 下载模型到 {target} ...", file=sys.stderr)
        with urllib.request.urlopen(SELFIE_MULTICLASS_URL, timeout=timeout) as resp:
            data = resp.read()
        if not data or len(data) < 100 * 1024:
            return None
        tmp = target + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, target)
        return target
    except Exception as e:  # noqa: BLE001
        print(f"[selfie] 下载失败：{e}", file=sys.stderr)
        return None


def _make_selfie_segmenter(model_path: str):
    """创建 MediaPipe Tasks ImageSegmenter 实例（IMAGE 模式，category mask）。

    通过 ``model_asset_buffer`` 传入模型字节，规避 MediaPipe 在 Windows 上
    将绝对路径误当成相对路径处理的问题。
    """
    from mediapipe.tasks.python import BaseOptions  # type: ignore
    from mediapipe.tasks.python.vision import (  # type: ignore
        ImageSegmenter, ImageSegmenterOptions, RunningMode,
    )
    with open(model_path, "rb") as f:
        model_bytes = f.read()
    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_buffer=model_bytes),
        running_mode=RunningMode.IMAGE,
        output_category_mask=True,
        output_confidence_masks=False,
    )
    return ImageSegmenter.create_from_options(options)


def segment_skin_mask(
    bgr_or_bgra: np.ndarray, model_path: Optional[str] = None,
    include_face: bool = True, include_body: bool = True,
) -> Optional[np.ndarray]:
    """对单张图执行 Selfie Multiclass Segmentation，返回 ``(H,W) bool`` 皮肤掩码。

    - ``include_face=True`` 时把 face-skin (类别 3) 算作皮肤
    - ``include_body=True`` 时把 body-skin (类别 2) 算作皮肤
    - 模型不可用时返回 ``None``，调用方应回退到颜色启发式。
    """
    mp_path = _ensure_selfie_multiclass_model(model_path)
    if mp_path is None:
        return None
    if bgr_or_bgra.ndim != 3:
        return None
    if bgr_or_bgra.shape[2] == 4:
        bgr = bgr_or_bgra[:, :, :3]
    else:
        bgr = bgr_or_bgra
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    try:
        with _make_selfie_segmenter(mp_path) as seg:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = seg.segment(mp_image)
        cat = result.category_mask.numpy_view()  # uint8 (H,W)
    except Exception as e:  # noqa: BLE001
        print(f"[selfie] 分割失败：{e}", file=sys.stderr)
        return None
    skin = np.zeros(cat.shape, dtype=bool)
    if include_face:
        skin |= cat == SELFIE_CLASS_FACE_SKIN
    if include_body:
        skin |= cat == SELFIE_CLASS_BODY_SKIN
    return skin


# =====================================================================================
# NeckSourceTransplant — 主类
# =====================================================================================

class NeckSourceTransplant:
    """
    独立脖子移植类。

    Parameters
    ----------
    neck_top_inset : float
        脖顶内收比例（0.70-1.0）。<1 让脖子顶部比下颌窄，默认 0.78。
    neck_slim_scale : float
        脖子整体变细比例（0.72-1.0）。默认 0.87。
    neck_bottom_flare : float
        底边外扩系数（>=1.02）。默认 1.05。
    neck_depth_frac : float
        脖子深度系数（越大脖子越长）。默认 1.4。
    pose_correction : bool
        是否启用姿态修正（俯仰/yaw 影响 overlap）。默认 True。
    auto_scale_by_jaw : bool
        是否按下颌跨度自适应 neck_depth / overlap。默认 True。
    skin_v_scale : float
        肤色 V 通道缩放（0.85-1.05）。默认 0.92。
    skin_h_shift : float
        肤色 H 偏移（-6~6）。默认 0.0。
    skin_s_scale : float
        肤色 S 缩放（0.90-1.20）。默认 1.0。
    tone_match_strength : float
        transplant 后颜色对齐强度（0-1）。默认 0.30。
    use_segmenter : bool
        是否使用 MediaPipe Selfie Multiclass Segmenter（CPU 模型）做精准的皮肤分割。
        默认 True；模型不可用 / 下载失败时自动回退到颜色启发式。
    segmenter_model_path : str or None
        显式指定 ``selfie_multiclass_256x256.tflite`` 的本地路径；
        留空时自动从 ``models/`` 或 ``~/.cache/add_neck_source/`` 加载，
        若都不存在则联网下载（仅首次）。
    debug_dir : str or None
        调试输出目录。非空时会在该目录下保存每一步的中间图片。
    """

    def __init__(
        self,
        neck_top_inset: float = NECK_TOP_INSET_DEFAULT,
        neck_slim_scale: float = NECK_SLIM_SCALE_DEFAULT,
        neck_bottom_flare: float = 1.05,
        neck_depth_frac: float = 1.4,
        pose_correction: bool = True,
        auto_scale_by_jaw: bool = True,
        skin_v_scale: float = 0.92,
        skin_h_shift: float = 0.0,
        skin_s_scale: float = 1.0,
        tone_match_strength: float = TRANSPLANT_TONE_MATCH_STRENGTH,
        use_segmenter: bool = True,
        segmenter_model_path: Optional[str] = None,
        debug_dir: Optional[str] = None,
    ):
        self.neck_top_inset = float(np.clip(neck_top_inset, 0.70, 1.0))
        self.neck_slim_scale = float(np.clip(neck_slim_scale, 0.72, 1.0))
        self.neck_bottom_flare = max(1.02, float(neck_bottom_flare))
        self.neck_depth_frac = float(neck_depth_frac)
        self.pose_correction = pose_correction
        self.auto_scale_by_jaw = auto_scale_by_jaw
        self.skin_v_scale = skin_v_scale
        self.skin_h_shift = skin_h_shift
        self.skin_s_scale = skin_s_scale
        self.tone_match_strength = float(np.clip(tone_match_strength, 0.0, 1.0))
        self.use_segmenter = bool(use_segmenter)
        self.segmenter_model_path = (
            os.path.abspath(segmenter_model_path) if segmenter_model_path else None
        )
        self.debug_dir: Optional[str] = None
        self._debug_step: int = 0
        if debug_dir:
            self.debug_dir = os.path.abspath(debug_dir)
        self._result: Optional[np.ndarray] = None

    def run(
        self,
        head_input,
        source_input,
    ) -> np.ndarray:
        """
        执行脖子移植主流程。

        Parameters
        ----------
        head_input : str or np.ndarray
            抠图头像。str 时为 RGBA PNG 路径；np.ndarray 时为 (H,W,4) BGRA uint8 数组。
        source_input : str or np.ndarray
            原图（完整人像，带真实脖子）。str 时为图片路径；np.ndarray 时为 (H,W,3/4) 数组。

        Returns
        -------
        np.ndarray
            (H,W,4) BGRA uint8 合成结果。

        Raises
        ------
        RuntimeError
            未检测到人脸时抛出。
        """
        if isinstance(head_input, str):
            head_bgra = load_rgba(head_input)
        else:
            head_bgra = head_input.copy()
            if head_bgra.shape[2] == 3:
                head_bgra = cv2.cvtColor(head_bgra, cv2.COLOR_BGR2BGRA)
                head_bgra[:, :, 3] = 255

        if isinstance(source_input, str):
            source_tuple = _load_source_image_and_detect(source_input)
            if source_tuple[0] is None:
                raise RuntimeError(f"无法从 source 图检测到人脸: {source_input}")
            source_bgra, source_landmarks = source_tuple
        else:
            src = source_input.copy()
            if src.ndim != 3:
                raise ValueError("source_input 数组需要 3 维")
            if src.shape[2] == 3:
                src = cv2.cvtColor(src, cv2.COLOR_BGR2BGRA)
                src[:, :, 3] = 255
            elif src.shape[2] != 4:
                raise ValueError(f"source_input 通道数需为 3 或 4，当前 {src.shape[2]}")
            src_rgb = cv2.cvtColor(src[:, :, :3], cv2.COLOR_BGR2RGB)
            source_landmarks = detect_face_with_retry(src_rgb)
            if source_landmarks is None:
                raise RuntimeError("无法从 source 数组检测到人脸")
            source_bgra = src

        result = self._transplant(head_bgra, source_bgra, source_landmarks)
        self._result = result
        return result

    def save(self, output_path: str) -> bool:
        """保存结果到文件（支持中文路径）。"""
        if self._result is None:
            raise RuntimeError("尚未执行 run()，无结果可保存。")
        path = os.path.normpath(output_path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        return imwrite_unicode(path, self._result)

    @property
    def result(self) -> Optional[np.ndarray]:
        """最后一次 run() 的结果 BGRA 数组。"""
        return self._result

    # ---- internal: debug helpers ----------------------------------------------

    LANDMARK_LABELS: dict = {
        152: "chin", 172: "jaw-L", 397: "jaw-R",
        33: "eye-L", 263: "eye-R", 1: "nose",
    }

    def _debug_img(self, name: str, img: np.ndarray):
        if self.debug_dir is None:
            return
        os.makedirs(self.debug_dir, exist_ok=True)
        self._debug_step += 1
        step = f"{self._debug_step:02d}"
        path = os.path.join(self.debug_dir, f"{step}_{name}.png")
        imwrite_unicode(path, img)

    def _debug_alpha(self, name: str, alpha: np.ndarray):
        if self.debug_dir is None:
            return
        v = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
        bgra = np.dstack([rgb, np.full_like(v, 255)])
        self._debug_img(name, bgra)

    def _draw_lm(self, bgra: np.ndarray, landmarks, h: int, w: int, labels: Optional[set] = None):
        out = bgra.copy()
        lm = landmarks.landmark
        for lid, label in self.LANDMARK_LABELS.items():
            if labels is not None and label not in labels:
                continue
            x, y = landmark_xy(lm[lid], w, h)
            cv2.circle(out, (int(round(x)), int(round(y))), max(3, w // 300), (0, 255, 255), -1)
            cv2.putText(out, label, (int(round(x)) + 5, int(round(y)) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.35, w / 1200), (0, 255, 255), 1, cv2.LINE_AA)
        return out

    def _draw_poly(self, bgra: np.ndarray, poly: np.ndarray, color=(0, 255, 0)):
        out = bgra.copy()
        pts = np.round(poly).astype(np.int32).reshape(1, -1, 2)
        cv2.polylines(out, pts, True, color, 2, cv2.LINE_AA)
        return out

    # ---- internal ------------------------------------------------------------

    def _transplant(
        self,
        head_bgra: np.ndarray,
        source_bgra: np.ndarray,
        source_landmarks,
    ) -> np.ndarray:
        h, w = head_bgra.shape[:2]
        sH, sW = source_bgra.shape[:2]
        self._debug_step = 0
        self._debug_img("input_head", head_bgra)
        self._debug_img("input_source", source_bgra)

        rgb = bgra_to_rgb(head_bgra)
        head_landmarks = detect_face_with_retry(rgb)
        if head_landmarks is None:
            raise RuntimeError("未在 head 图中检测到人脸")
        head_lm_viz = self._draw_lm(head_bgra, head_landmarks, h, w)
        self._debug_img("head_landmarks", head_lm_viz)

        src_lm_viz = self._draw_lm(source_bgra, source_landmarks, sH, sW)
        self._debug_img("source_landmarks", src_lm_viz)

        lm_h = head_landmarks.landmark
        head_jaw_l = landmark_xy(lm_h[LANDMARK_LEFT_JAW_ON_OVAL], w, h)
        head_jaw_r = landmark_xy(lm_h[LANDMARK_RIGHT_JAW_ON_OVAL], w, h)
        head_jaw_span = max(float(np.hypot(head_jaw_l[0] - head_jaw_r[0], head_jaw_l[1] - head_jaw_r[1])), 12.0)

        lm_s = source_landmarks.landmark
        src_jaw_l = landmark_xy(lm_s[LANDMARK_LEFT_JAW_ON_OVAL], sW, sH)
        src_jaw_r = landmark_xy(lm_s[LANDMARK_RIGHT_JAW_ON_OVAL], sW, sH)
        src_jaw_span = max(float(np.hypot(src_jaw_l[0] - src_jaw_r[0], src_jaw_l[1] - src_jaw_r[1])), 12.0)

        if self.auto_scale_by_jaw:
            overlap_s = float(np.clip(
                src_jaw_span * JAW_SPAN_OVERLAP_FRAC, JAW_SPAN_OVERLAP_MIN_PX, JAW_SPAN_OVERLAP_MAX_PX,
            ))
        else:
            overlap_s = 17.0
        neck_depth_s = max(
            src_jaw_span * float(self.neck_depth_frac), JAW_SPAN_DEPTH_MIN_PX,
        ) if self.auto_scale_by_jaw else 80.0

        slim = float(np.clip(self.neck_slim_scale, 0.72, 1.0))
        flare = max(1.02, self.neck_bottom_flare)
        flare_slim = 1.0 + (flare - 1.0) * slim
        neck_depth_s *= slim

        chin_s_x, chin_s_y = landmark_xy(lm_s[LANDMARK_CHIN_BOTTOM], sW, sH)

        if self.pose_correction:
            yaw, pitch, roll = estimate_head_pose(head_landmarks, w, h)
            pitch_scale = float(np.clip(
                1.0 + POSE_PITCH_CHIN_OVERLAP_GAIN * pitch,
                POSE_PITCH_CHIN_OVERLAP_MIN, POSE_PITCH_CHIN_OVERLAP_MAX,
            ))
            overlap_s = float(np.clip(
                overlap_s * pitch_scale, JAW_SPAN_OVERLAP_MIN_PX, JAW_SPAN_OVERLAP_MAX_PX,
            ))
        else:
            yaw, pitch, roll = 0.0, 0.0, 0.0

        skin_bgr = sample_skin_color_bgra(
            head_bgra, head_landmarks, w, h,
            skin_v_scale=self.skin_v_scale,
            skin_h_shift=self.skin_h_shift,
            skin_s_scale=self.skin_s_scale,
        )

        affine_M = _compute_affine_matrix(source_landmarks, sW, sH, head_landmarks, w, h)
        if affine_M is None:
            raise RuntimeError("affine 变换失败（三点共线或 opencv 错误）")
        affine_M32 = affine_M.astype(np.float32)

        top_s = build_jaw_top_polyline(
            source_landmarks, sW, sH, overlap_s, top_inset=self.neck_top_inset,
        )

        neck_depth_s = detect_clothing_boundary(
            source_bgra, source_landmarks, top_s, neck_depth_s,
            chin_s_x, chin_s_y,
        )

        top_h = _affine_pts(top_s, affine_M32)

        top_h[:, 0] = np.clip(top_h[:, 0], 0.0, float(w - 1))
        top_h[:, 1] = np.clip(top_h[:, 1], 0.0, float(h - 1))

        pts_vert = np.array([[chin_s_x, chin_s_y], [chin_s_x, chin_s_y + neck_depth_s]], dtype=np.float64)
        neck_end_h = _affine_pts(pts_vert, affine_M32)
        neck_depth_h = float(np.hypot(
            neck_end_h[1, 0] - neck_end_h[0, 0],
            neck_end_h[1, 1] - neck_end_h[0, 1],
        ))
        cx_h = float(np.mean(top_h[:, 0]))
        bot_y_h = float(np.max(top_h[:, 1])) + neck_depth_h
        bottom_h = np.column_stack([
            cx_h + (top_h[:, 0] - cx_h) * flare_slim,
            np.full(len(top_h), bot_y_h, dtype=np.float64),
        ])
        n_top = len(top_h)
        u = np.linspace(-1.0, 1.0, n_top)
        sag = min(6.0, neck_depth_h * 0.08)
        bottom_h[:, 1] += sag * (1.0 - u * u)

        poly = np.vstack([top_h, bottom_h[::-1]])
        poly[:, 0] = np.clip(poly[:, 0], 0.0, float(w - 1))
        poly[:, 1] = np.clip(poly[:, 1], 0.0, float(h - 1))

        top_s_viz = self._draw_poly(src_lm_viz, top_s, color=(0, 255, 0))
        self._debug_img("neck_top_source", top_s_viz)

        mask_u8 = fill_polygon_mask(h, w, poly, supersample=NECK_MASK_SUPERSAMPLE)
        mask_bool = mask_u8 >= 1

        poly_viz = self._draw_poly(head_lm_viz, poly, color=(0, 255, 0))
        self._debug_img("neck_polygon_head", poly_viz)
        self._debug_img("neck_mask", mask_u8)

        warped = cv2.warpAffine(source_bgra, affine_M32, (w, h),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        self._debug_img("source_warped", warped)

        ref_pixels = _gather_face_skin_pixels_bgr(head_bgra, head_landmarks, h, w, patch=22)
        if ref_pixels.shape[0] < 8:
            ref_pixels = np.array([skin_bgr], dtype=np.uint8).reshape(1, 3)

        src_ref_pixels = _gather_face_skin_pixels_bgr(source_bgra, source_landmarks, sH, sW, patch=22)
        if src_ref_pixels.shape[0] < 8:
            src_ref_pixels = ref_pixels

        # ============ 主皮肤检测：MediaPipe Selfie Multiclass Segmenter ============
        # 直接对 source 做语义分割（face-skin + body-skin = 真皮肤），
        # 然后用 affine 变换到 head 空间，作为 skin_w_map 的主信号。
        skin_w_map: Optional[np.ndarray] = None
        seg_used = False
        if self.use_segmenter:
            seg_skin_src = segment_skin_mask(
                source_bgra, model_path=self.segmenter_model_path,
            )
            if seg_skin_src is not None:
                seg_u8_src = (seg_skin_src.astype(np.uint8) * 255)
                # 用同一份 affine_M32 把 mask 也 warp 到 head 空间
                seg_warp = cv2.warpAffine(
                    seg_u8_src, affine_M32, (w, h),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                skin_w_map = (seg_warp.astype(np.float32) / 255.0)
                # 边缘平滑
                skin_w_map = cv2.GaussianBlur(skin_w_map, (5, 5), 1.2)
                seg_used = True
                if self.debug_dir is not None:
                    # 保存 source 上的原始分割结果，便于排查
                    seg_dbg = np.zeros((sH, sW, 4), dtype=np.uint8)
                    seg_dbg[..., :3] = source_bgra[..., :3]
                    overlay = source_bgra[..., :3].astype(np.float32)
                    pink = np.array([200, 100, 255], dtype=np.float32)
                    overlay[seg_skin_src] = (
                        overlay[seg_skin_src] * 0.45 + pink * 0.55
                    )
                    seg_dbg[..., :3] = np.clip(overlay, 0, 255).astype(np.uint8)
                    seg_dbg[..., 3] = 255
                    self._debug_img("segmenter_source", seg_dbg)

        # 若 segmenter 不可用 / 失败，回退到 Lab 距离启发式
        if skin_w_map is None:
            skin_w_map = self._skin_similarity_map(
                warped[:, :, :3], src_ref_pixels,
            )

        if self.debug_dir is not None:
            self._debug_alpha(
                "skin_similarity_seg" if seg_used else "skin_similarity_lab",
                skin_w_map,
            )

        # 把 warped 多边形内部的"非皮肤"像素（衣领/头发等）用周围真实皮肤纹理 inpaint 出去，
        # 让左右两侧都拥有 source 真皮肤的色彩 + 纹理，避免一侧被 procedural 平面色填充而显得"缺一块"。
        warped_filled = self._inpaint_non_skin(
            warped[:, :, :3], skin_w_map, mask_bool,
        )
        if self.debug_dir is not None:
            self._debug_img("warped_skin_filled",
                             np.dstack([warped_filled, mask_u8]))

        # 以 warped 真皮肤中位色作为 procedural 基础色，使无 warped 信息的边缘也协调一致。
        proc_skin_bgr = skin_bgr
        sw_in_mask = (skin_w_map > 0.7) & mask_bool
        if int(sw_in_mask.sum()) >= 60:
            warp_skin_pixels = warped[sw_in_mask][:, :3]
            med = np.median(warp_skin_pixels.astype(np.float64), axis=0)
            proc_skin_bgr = (med * 0.65 + np.asarray(skin_bgr, dtype=np.float64) * 0.35)

        proc_bgr = self._procedural_neck_layer(h, w, poly, mask_bool, proc_skin_bgr)
        proc_bgra = np.dstack([proc_bgr, mask_u8])
        self._debug_img("procedural_neck", proc_bgra)

        # 用 inpaint 后的 warped 替代原 warped 进行融合：左右两侧都有"真皮肤"风格的像素。
        neck_bgra = np.zeros((h, w, 4), dtype=np.uint8)
        neck_bgr = proc_bgr.copy()
        neck_bgr = self._blend_transplant(
            neck_bgr, warped_filled, mask_bool, None,
        )
        neck_blended_bgra = np.dstack([neck_bgr, mask_u8])
        self._debug_img("transplant_blended", neck_blended_bgra)

        neck_bgra[:, :, :3] = neck_bgr
        neck_bgra[:, :, 3] = mask_u8
        _reinhard_lab_mean_shift(neck_bgra, mask_bool, ref_pixels, strength=self.tone_match_strength)
        self._debug_img("color_matched", neck_bgra)

        refined_alpha = self._feather_alpha(mask_u8, head_jaw_span)
        self._debug_alpha("alpha_feather", refined_alpha)

        refined_alpha = self._suppress_overlap_alpha(refined_alpha, head_bgra, poly, head_jaw_span)
        self._debug_alpha("alpha_suppress", refined_alpha)

        refined_alpha = self._fade_out_alpha(refined_alpha, poly, h)
        self._debug_alpha("alpha_fadeout", refined_alpha)

        neck_bgra[:, :, 3] = np.clip(np.round(refined_alpha * 255.0), 0, 255).astype(np.uint8)
        self._debug_img("neck_final_layer", neck_bgra)

        result = alpha_over(neck_bgra, head_bgra)
        self._debug_img("result", result)
        return result

    # ---- internal: procedural neck layer (flat shading) ----------------------

    def _procedural_neck_layer(
        self, h: int, w: int, poly: np.ndarray, mask_bool: np.ndarray,
        skin_bgr: np.ndarray,
    ) -> np.ndarray:
        xx = np.arange(w, dtype=np.float64)[None, :]
        n_up = max(poly.shape[0] // 2, 2)
        upper = poly[:n_up].astype(np.float64)
        dt_top = distance_map_to_polyline(h, w, upper)
        depth_poly = max(float(np.max(poly[:, 1]) - np.min(poly[:, 1])), 1.0)
        tau_ao = max(depth_poly * 0.22, 8.0)
        ao_w = np.exp(-dt_top / tau_ao)
        x_center = float(np.mean(poly[:, 0]))
        span_x = float(np.max(poly[:, 0]) - np.min(poly[:, 0]))
        R_cyl = max(span_x * 0.5, 4.0)
        radial = np.clip((xx - x_center) / R_cyl, -1.4, 1.4)
        rad_curve = 1.0 - np.tanh(np.abs(radial) * 1.4) ** 1.6
        light_dir = np.tanh(radial * 0.9)
        L_lat = 1.0 + 0.06 * (rad_curve - 0.5) + 0.03 * light_dir
        L_ao = 1.0 - 0.08 * ao_w
        L = L_lat * L_ao
        if np.any(mask_bool):
            mu_L = float(np.mean(L[mask_bool]))
            L = L / max(mu_L, 1e-6)
        L = np.clip(L, 0.85, 1.15)
        wm_f = mask_bool.astype(np.float64)
        L = L * wm_f + (1.0 - wm_f)
        skin = np.array(skin_bgr, dtype=np.float64)
        bgr = np.tile(skin[None, None, :], (h, w, 1)) * L[:, :, np.newaxis]
        return np.clip(np.round(bgr), 0, 255).astype(np.uint8)

    # ---- internal: transplant blend ------------------------------------------

    @staticmethod
    def _inpaint_non_skin(
        warped_bgr: np.ndarray, skin_w_map: np.ndarray, mask_bool: np.ndarray,
        skin_thresh: float = 0.55, dilate_iters: int = 1, radius: int = 5,
    ) -> np.ndarray:
        """对 warped 图中位于 mask 内、但被判定为"非皮肤"的像素，用 OpenCV INPAINT
        从周围真皮肤区域延伸像素填上去，保留真实皮肤的纹理与微变化。"""
        non_skin = (skin_w_map < skin_thresh) & mask_bool
        if dilate_iters > 0:
            kernel = np.ones((3, 3), np.uint8)
            non_skin = cv2.dilate(
                non_skin.astype(np.uint8), kernel, iterations=int(dilate_iters)
            ).astype(bool)
        if not np.any(non_skin):
            return warped_bgr.copy()
        # mask=255 表示需要 inpaint 的位置
        m = (non_skin.astype(np.uint8)) * 255
        # 为了避免从多边形外（夹克/背景）取像素，先把多边形外区域也标记为 inpaint 的"已知 = 不是这里"——
        # 即只允许从多边形内的真皮肤区域取像素。把外部填成"未知"会扩大问题，所以反过来用：把外部区域
        # 用 mask 内 known 像素先临时填好（用 NS 法 + 较小 radius），再做最终 inpaint 即可获得平滑结果。
        return cv2.inpaint(warped_bgr, m, int(radius), cv2.INPAINT_TELEA)

    @staticmethod
    def _skin_similarity_map(
        warped_bgr: np.ndarray, ref_pixels_bgr: np.ndarray,
        d_low: float = 25.0, d_high: float = 55.0,
    ) -> np.ndarray:
        """以 warped 像素到参考肤色（source 脸部肤色池）的 Lab 距离推算"皮肤可信度"。

        距离 <= d_low ：完全可信（权重 1，warped 真实像素胜出）
        距离 >= d_high：完全不可信（权重 0，被 procedural 纯肤色替换）
        中间线性过渡。返回 (H,W) float32 in [0,1]。
        """
        warped_lab = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(
            ref_pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB,
        ).reshape(-1, 3).astype(np.float32)
        ref_mean = ref_lab.mean(axis=0)
        diff = warped_lab - ref_mean
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        w = np.clip((d_high - dist) / max(d_high - d_low, 1e-3), 0.0, 1.0)
        # 微调平滑避免硬边
        w = cv2.GaussianBlur(w.astype(np.float32), (5, 5), 1.0)
        return w

    def _blend_transplant(
        self, proc_bgr: np.ndarray, warped_bgr: np.ndarray,
        mask_bool: np.ndarray,
        skin_w_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        eroded_mask = cv2.erode(
            mask_bool.astype(np.uint8) * 255,
            np.ones((3, 3), np.uint8), iterations=1,
        )
        em = eroded_mask > 0
        edge_w = em.astype(np.float32)
        if skin_w_map is None:
            blend_w = edge_w[..., None]
        else:
            # 同时受边缘衰减和"皮肤可信度"约束：非皮肤区域的 warped 像素被 procedural 替换
            blend_w = (edge_w * skin_w_map.astype(np.float32))[..., None]
        proc_f = proc_bgr.astype(np.float32)
        warp_f = warped_bgr.astype(np.float32)
        blended_f = warp_f * blend_w + proc_f * (1.0 - blend_w)
        return np.clip(blended_f, 0, 255).astype(np.uint8)

    # ---- internal: alpha -----------------------------------------------------

    def _feather_alpha(self, mask_u8: np.ndarray, jaw_span: float) -> np.ndarray:
        sigma = float(np.clip(
            jaw_span * NECK_FEATHER_FRAC, NECK_FEATHER_MIN_PX, NECK_FEATHER_MAX_PX,
        ))
        k = odd_kernel(int(round(sigma * 3.0)) + 1)
        alpha = cv2.GaussianBlur(mask_u8.astype(np.float32), (k, k), sigma) / 255.0
        return alpha.astype(np.float64)

    def _suppress_overlap_alpha(
        self, alpha: np.ndarray, head_bgra: np.ndarray,
        poly: np.ndarray, jaw_span: float,
    ) -> np.ndarray:
        h, w = alpha.shape
        n_up = max(poly.shape[0] // 2, 2)
        upper = poly[:n_up].astype(np.float64)
        dt_upper = distance_map_to_polyline(h, w, upper)
        decay = float(max(jaw_span * NECK_SUPPRESS_DECAY_FRAC, NECK_SUPPRESS_DECAY_MIN_PX))
        boundary_envelope = np.exp(-dt_upper / decay)
        oa = head_bgra[:, :, 3].astype(np.float64) / 255.0
        suppress = np.power(oa, NECK_SUPPRESS_ALPHA_GAMMA) * boundary_envelope
        return np.clip(alpha * (1.0 - suppress), 0.0, 1.0)

    @staticmethod
    def _fade_out_alpha(alpha: np.ndarray, poly: np.ndarray, h: int) -> np.ndarray:
        y_min_poly = float(np.min(poly[:, 1]))
        y_max_poly = float(np.max(poly[:, 1]))
        poly_height = max(y_max_poly - y_min_poly, 1.0)
        yy = np.arange(h, dtype=np.float64)[:, None]
        yn_poly = np.clip((yy - y_min_poly) / poly_height, 0.0, 1.0)
        fade_start = 0.50
        fade = np.where(
            yn_poly < fade_start, 1.0,
            np.clip(1.0 - (yn_poly - fade_start) / (1.0 - fade_start), 0.0, 1.0) ** 1.2,
        )
        return alpha * fade


# =====================================================================================
# CLI
# =====================================================================================


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="独立脖子移植：从原图（带真实脖子）把脖子像素 affine 对齐后移植到抠图头像上。",
    )
    parser.add_argument("input", help="输入 RGBA PNG 路径（抠图头像）")
    parser.add_argument("--source-image", required=True, help="原图路径（同一人的完整人像，带真实脖子）")
    parser.add_argument("-o", "--output", default=None, help="输出路径；默认 *_with_neck_source.png")
    parser.add_argument("--neck-top-inset", type=float, default=NECK_TOP_INSET_DEFAULT)
    parser.add_argument("--neck-slim", type=float, default=NECK_SLIM_SCALE_DEFAULT)
    parser.add_argument("--neck-bottom-flare", type=float, default=1.05)
    parser.add_argument("--neck-depth-frac", type=float, default=1.4)
    parser.add_argument("--skin-v-scale", type=float, default=0.92)
    parser.add_argument("--skin-h-shift", type=float, default=0.0)
    parser.add_argument("--skin-s-scale", type=float, default=1.0)
    parser.add_argument("--tone-match-strength", type=float, default=TRANSPLANT_TONE_MATCH_STRENGTH)
    parser.add_argument("--no-pose-correction", dest="pose_correction", action="store_false")
    parser.add_argument("--no-auto-scale", dest="auto_scale_by_jaw", action="store_false")
    parser.add_argument(
        "--no-segmenter", dest="use_segmenter", action="store_false",
        help="关闭 MediaPipe Selfie 分割模型，回退到颜色启发式。",
    )
    parser.add_argument(
        "--segmenter-model", default=None,
        help="可选：本地 selfie_multiclass_256x256.tflite 路径；不填则自动下载到 models/ 缓存。",
    )
    parser.add_argument(
        "--debug", default=None,
        help="启用调试输出：将每一步中间图片保存到指定目录",
    )
    parser.set_defaults(pose_correction=True, auto_scale_by_jaw=True, use_segmenter=True)
    args = parser.parse_args(argv)

    if args.output:
        out = os.path.abspath(args.output)
    else:
        dname, fname = os.path.split(os.path.abspath(args.input))
        stem, _ = os.path.splitext(fname)
        if stem.lower().endswith("_rgba"):
            stem = stem[:-5]
        out = os.path.join(dname, f"{stem}_with_neck_source.png") if dname else f"{stem}_with_neck_source.png"

    nst = NeckSourceTransplant(
        neck_top_inset=args.neck_top_inset,
        neck_slim_scale=args.neck_slim,
        neck_bottom_flare=args.neck_bottom_flare,
        neck_depth_frac=args.neck_depth_frac,
        pose_correction=args.pose_correction,
        auto_scale_by_jaw=args.auto_scale_by_jaw,
        skin_v_scale=args.skin_v_scale,
        skin_h_shift=args.skin_h_shift,
        skin_s_scale=args.skin_s_scale,
        tone_match_strength=args.tone_match_strength,
        use_segmenter=args.use_segmenter,
        segmenter_model_path=args.segmenter_model,
        debug_dir=args.debug,
    )

    try:
        result = nst.run(args.input, args.source_image)
    except Exception as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1

    if not nst.save(out):
        print(f"[错误] 无法写入 {out}", file=sys.stderr)
        return 1
    print(f"已保存: {out}")
    if args.debug:
        print(f"调试图片已保存到: {args.debug}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())