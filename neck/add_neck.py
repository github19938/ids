# -*- coding: utf-8 -*-
"""
add_neck.py
-----------
为已抠图（透明背景）的 RGBA 头像在下巴下方生成下颌引导的「假脖子」
（多边形 + 渐变与柔边，替代整块椭圆），
使用 MediaPipe FaceMesh 定位下巴与人脸宽度，采样肤色并做高斯模糊与 Alpha 合成。

依赖：opencv-python、mediapipe、numpy
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from typing import List, NamedTuple, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# MediaPipe FaceMesh 常用关键点索引（468 点模型）
# 参考：https://github.com/google/mediapipe/wiki/Attention-Mesh
# ---------------------------------------------------------------------------
# 下巴最下端（颏部正中）
LANDMARK_CHIN_BOTTOM = 152
# 肤色：5 区按权重混合中值；左右颊取样 **10×10**，其余三区边长不变（仍为 10×10，见 SKIN_PATCH_OTHER_PX）
SKIN_PATCH_CHEEK_PX = 10
SKIN_PATCH_OTHER_PX = 10
# (关键点索引, 权重, 标注简称, 取样边长 px) — 左右颊合计 50%，人中 20%，额 20%，下巴上 10%
SKIN_SAMPLE_REGIONS: List[Tuple[int, float, str, int]] = [
    (205, 0.25, "L-cheek", SKIN_PATCH_CHEEK_PX),
    (425, 0.25, "R-cheek", SKIN_PATCH_CHEEK_PX),
    (164, 0.20, "philtrum", SKIN_PATCH_OTHER_PX),
    (10, 0.20, "forehead", SKIN_PATCH_OTHER_PX),
    (200, 0.10, "above-chin", SKIN_PATCH_OTHER_PX),
]
# 脸外轮廓上、靠近下颌角/下缘的点：用于「脖子顶」折线，避免从 234/454 经颞颊绕到耳侧过宽
LANDMARK_LEFT_JAW_ON_OVAL = 172
LANDMARK_RIGHT_JAW_ON_OVAL = 397
# 下巴向上插入像素范围（脖子顶 y = chin_y - overlap），与需求一致夹在 [15, 20]
CHIN_OVERLAP_MIN_PX = 15.0
CHIN_OVERLAP_MAX_PX = 20.0

# 脖子整体相对几何「变细」比例（<1）：同时缩小 **垂直深度** 与 **底边相对下颌的外扩量**，上沿仍贴下颌
NECK_SLIM_SCALE_DEFAULT = 0.87

# Film grain：脸部采样块灰度 std → 脖子 ``cv2.randn`` 标准差（自适应「包浆」）
GRAIN_REF_GAIN_DEFAULT = 1.08
GRAIN_SIGMA_MIN = 0.55
GRAIN_SIGMA_MAX = 4.0

# 脖子圆柱体明暗（在 mask 内乘在肤色 albedo 上；mask 内会再做均值归一，避免整体变暗）
# 主光从画面左侧来：迎光侧提亮略强于背光侧压暗（非对称），立体感更强且不易发灰
NECK_CYLINDER_K_LIT = 0.125
NECK_CYLINDER_K_SHADOW = 0.072
# tanh(radial * scale)：越大圆柱转折越「圆」
NECK_CYLINDER_TANH_SCALE = 1.52
# 上沿 AO：沿 **下颌上折线** 的距离变换（近颌线压暗），替代整条竖直线性带
NECK_CYLINDER_AO_TOP = 0.028
# 距离变换衰减长度（相对脖子深度 depth 的比例，下限像素）
NECK_AO_DT_TAU_FRAC = 0.24
NECK_AO_DT_TAU_MIN_PX = 6.0
# 左右颊 V 通道差 → 圆柱轴 / 高光横向偏移（左亮则轴与高光略向左）
NECK_LIGHT_AXIS_GAIN = 0.88
NECK_SHINE_TRACK_GAIN = 0.58
NECK_SHINE_K = 0.042
NECK_SHINE_SIGMA_FRAC = 0.19
# 默认主光略偏左时的高光基线（相对轴再往左）
NECK_SHINE_BASE_OFFSET_FRAC = 0.15
# 脸部 V 对比度 → 圆柱 K / 高光强度缩放（平光弱、强侧光强）
NECK_LIGHT_STRENGTH_LR_COEF = 2.0
NECK_LIGHT_STRENGTH_RNG_COEF = 1.05
NECK_LIGHT_STRENGTH_STD_COEF = 0.35
NECK_LIGHT_STRENGTH_MIN = 0.52
NECK_LIGHT_STRENGTH_MAX = 1.38
NECK_SHINE_STRENGTH_LR_COEF = 0.95
NECK_SHINE_STRENGTH_RNG_COEF = 0.62
NECK_SHINE_K_MIN_FRAC = 0.28
NECK_SHINE_K_MAX_FRAC = 1.22
# 脖子 albedo 上叠加的线性肤色渐变（由额/颏与左右颊 V 差驱动，mask 内归一）
NECK_SKIN_GRAD_GAIN_LR = 0.38
NECK_SKIN_GRAD_GAIN_FB = 0.22
# 下巴区明显暗于双颊时略加强颌下 AO
NECK_AO_CHIN_DARK_BOOST_COEF = 0.25
NECK_AO_CHIN_DARK_DIV = 80.0
NECK_AO_TOP_MIN_FRAC = 0.78
NECK_AO_TOP_MAX_FRAC = 1.48
# 归一化后允许的相对亮度范围（相对 mask 内均值 1.0）
NECK_CYLINDER_L_MIN = 0.86
NECK_CYLINDER_L_MAX = 1.14


def imread_unicode(path: str, flags: int = cv2.IMREAD_UNCHANGED) -> Optional[np.ndarray]:
    """
    读取任意路径下的图像（支持中文等非 ASCII 路径）。

    Windows 上 cv2.imread 走底层 ANSI API，路径含中文时常返回 None；
    先用 Python 以二进制打开文件，再用 cv2.imdecode 解码即可规避。
    """
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
    """写入图像（支持中文路径）。使用 imencode + 二进制写入。"""
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
    """
    读取 PNG（支持透明通道）。
    OpenCV 以 BGRA 顺序存储（与常见 RGBA 文件名含义对应为：通道顺序为 B,G,R,A）。
    使用 imread_unicode，避免 Windows 下中文路径无法读取。
    """
    img = imread_unicode(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取图像（文件不存在、损坏或格式不受支持）: {path}")
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"需要彩色图（3 或 4 通道），当前 shape={img.shape}")
    if img.shape[2] == 3:
        # 无 alpha 时补全为不透明，便于统一后续逻辑
        bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = 255
        return bgra
    return img


def bgra_to_rgb(bgra: np.ndarray) -> np.ndarray:
    """供 MediaPipe 使用的 RGB uint8。"""
    bgr = bgra[:, :, :3]
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def landmark_xy(landmark, width: int, height: int) -> Tuple[float, float]:
    """
    MediaPipe 输出为「归一化坐标」x,y ∈ [0,1]，需乘以图像宽高得到像素坐标。
    z 为相对深度，本程序绘制 2D 脖子只用 x,y。
    """
    return float(landmark.x * width), float(landmark.y * height)


def skin_patch_rect_at(
    cx: float,
    cy: float,
    ih: int,
    iw: int,
    patch: int,
) -> Tuple[int, int, int, int]:
    """以 (cx,cy) 为中心取 patch×patch 矩形（贴图边界裁切），返回 x0,y0,pw,ph。"""
    patch = int(patch)
    half = patch // 2
    ix, iy = int(round(cx)), int(round(cy))
    x0 = int(np.clip(ix - half, 0, max(0, iw - patch)))
    y0 = int(np.clip(iy - half, 0, max(0, ih - patch)))
    pw = min(patch, iw - x0)
    ph = min(patch, ih - y0)
    return x0, y0, pw, ph


def median_bgr_in_patch(bgra: np.ndarray, x0: int, y0: int, pw: int, ph: int) -> Optional[np.ndarray]:
    """patch 内 alpha>40 的像素 BGR 各通道取中值；无有效像素返回 None。"""
    if pw <= 0 or ph <= 0:
        return None
    roi = bgra[y0 : y0 + ph, x0 : x0 + pw]
    m = roi[:, :, 3].astype(np.float32) > 40.0
    if not np.any(m):
        return None
    flat = roi[:, :, :3][m].astype(np.float64)
    if flat.shape[0] == 0:
        return None
    return np.median(flat, axis=0)


def mean_hsv_v_in_patch(
    bgra: np.ndarray, x0: int, y0: int, pw: int, ph: int
) -> Optional[float]:
    """patch 内 alpha>40 像素 HSV 的 V 通道均值；无有效像素返回 None。"""
    if pw <= 0 or ph <= 0:
        return None
    roi = bgra[y0 : y0 + ph, x0 : x0 + pw]
    m = roi[:, :, 3].astype(np.float32) > 40.0
    if not np.any(m):
        return None
    hsv = cv2.cvtColor(roi[:, :, :3], cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float64)[m]
    return float(np.mean(v))


class FaceNeckLightParams(NamedTuple):
    """由脸部局部 V 推断脖子光照：轴偏移、圆柱强度、高光、AO、肤色线性渐变系数。"""

    x_axis_shift: float
    spec_x_shift: float
    k_lit: float
    k_shadow: float
    shine_k: float
    ao_top: float
    skin_grad_gx: float
    skin_grad_gy: float


def estimate_face_lighting_for_neck(
    bgra: np.ndarray,
    landmarks,
    ih: int,
    iw: int,
    R: float,
) -> FaceNeckLightParams:
    """
    用与肤色采样一致的关键点小块 **HSV-V 均值** 估计照在脸上的光：

    - 左右颊(205/425) 推断 **横向** 来光（轴与高光带平移）；
    - 额(10) 与下巴上(200) 推断 **上下** 分量，驱动脖子 albedo 的弱竖直渐变；
    - ``|V_L-V_R|``、多块 V 的极差与标准差 → **对比度强度**，自适应缩放 ``K_LIT`` / ``K_SHADOW``、
      高光强度（平光时减弱、强侧光或高反差时增强）；
    - 双颊平均明显亮于下巴上区时略 **加强 AO**（颌下更贴重阴影照片）。
    """
    lm = landmarks.landmark
    Rf = float(max(R, 4.0))

    def patch_v(lid: int, psize: int) -> Optional[float]:
        cx, cy = landmark_xy(lm[lid], iw, ih)
        x0, y0, pw, ph = skin_patch_rect_at(cx, cy, ih, iw, psize)
        return mean_hsv_v_in_patch(bgra, x0, y0, pw, ph)

    vl = patch_v(205, SKIN_PATCH_CHEEK_PX)
    vr = patch_v(425, SKIN_PATCH_CHEEK_PX)
    vf = patch_v(10, SKIN_PATCH_OTHER_PX)
    vu = patch_v(200, SKIN_PATCH_OTHER_PX)

    vals = [float(v) for v in (vl, vr, vf, vu) if v is not None]
    v_std = float(np.std(np.array(vals, dtype=np.float64))) if len(vals) >= 2 else 0.0
    v_rng = (max(vals) - min(vals)) / 255.0 if len(vals) >= 2 else 0.0

    lr_asym = 0.0
    if vl is not None and vr is not None:
        lr_asym = abs(float(vl) - float(vr)) / (float(vl) + float(vr) + 1e-3)

    strength = float(
        NECK_LIGHT_STRENGTH_MIN
        + float(NECK_LIGHT_STRENGTH_LR_COEF) * lr_asym
        + float(NECK_LIGHT_STRENGTH_RNG_COEF) * v_rng
        + float(NECK_LIGHT_STRENGTH_STD_COEF) * (v_std / 40.0)
    )
    strength = float(np.clip(strength, NECK_LIGHT_STRENGTH_MIN, NECK_LIGHT_STRENGTH_MAX))

    if vl is None or vr is None:
        axis_shift = 0.0
        spec_shift = 0.0
    else:
        denom = float(vl + vr) + 1e-3
        dv = float(np.clip((float(vl) - float(vr)) / denom, -0.28, 0.28))
        axis_shift = -float(NECK_LIGHT_AXIS_GAIN) * dv * Rf
        spec_shift = -float(NECK_SHINE_TRACK_GAIN) * dv * Rf

    k_lit = float(NECK_CYLINDER_K_LIT) * strength
    k_shadow = float(NECK_CYLINDER_K_SHADOW) * strength

    shine_scale = float(
        NECK_SHINE_K_MIN_FRAC
        + float(NECK_SHINE_STRENGTH_LR_COEF) * lr_asym
        + float(NECK_SHINE_STRENGTH_RNG_COEF) * v_rng
    )
    shine_scale = float(np.clip(shine_scale, NECK_SHINE_K_MIN_FRAC, NECK_SHINE_K_MAX_FRAC))
    shine_k = float(NECK_SHINE_K) * shine_scale

    ao_top = float(NECK_CYLINDER_AO_TOP)
    if vu is not None and vl is not None and vr is not None:
        v_mid = 0.5 * (float(vl) + float(vr))
        if v_mid > float(vu) + 4.0:
            ao_top *= float(
                np.clip(
                    1.0
                    + float(NECK_AO_CHIN_DARK_BOOST_COEF)
                    * ((v_mid - float(vu)) / float(NECK_AO_CHIN_DARK_DIV)),
                    1.0,
                    1.45,
                )
            )
    ao_top = float(
        np.clip(
            ao_top,
            float(NECK_CYLINDER_AO_TOP) * float(NECK_AO_TOP_MIN_FRAC),
            float(NECK_CYLINDER_AO_TOP) * float(NECK_AO_TOP_MAX_FRAC),
        )
    )

    skin_gx = 0.0
    skin_gy = 0.0
    if vl is not None and vr is not None:
        skin_gx = float(
            np.clip((float(vl) - float(vr)) / 255.0, -0.22, 0.22) * float(NECK_SKIN_GRAD_GAIN_LR)
        )
    if vf is not None and vu is not None:
        skin_gy = float(
            np.clip((float(vf) - float(vu)) / 255.0, -0.22, 0.22) * float(NECK_SKIN_GRAD_GAIN_FB)
        )

    return FaceNeckLightParams(
        x_axis_shift=axis_shift,
        spec_x_shift=spec_shift,
        k_lit=k_lit,
        k_shadow=k_shadow,
        shine_k=shine_k,
        ao_top=ao_top,
        skin_grad_gx=skin_gx,
        skin_grad_gy=skin_gy,
    )


def _apply_bgr_value_scale(bgr: np.ndarray, v_scale: float) -> np.ndarray:
    """在 HSV 中缩放 V 通道（OpenCV H∈[0,180], S,V∈[0,255]）。"""
    px = np.clip(np.round(bgr).astype(np.uint8).reshape(1, 1, 3), 0, 255)
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[0, 0, 2] *= float(v_scale)
    hsv = np.clip(hsv, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)[0, 0].astype(np.float64)
    return out


def sample_skin_color_bgra(
    bgra: np.ndarray,
    landmarks,
    w: int,
    h: int,
    skin_v_scale: float = 0.925,
) -> np.ndarray:
    """
    肤色采样：5 个核心区域，各取 ``patch×patch`` 子块（左右颊 ``SKIN_PATCH_CHEEK_PX``，
    其余 ``SKIN_PATCH_OTHER_PX``）内 **alpha>40** 像素的 **BGR 中值**，
    再按权重做向量加权平均；某块无有效像素时丢弃该块权重并**重归一化**其余权重。

    权重：左颊 25% + 右颊 25% + 人中 20% + 额头 20% + 下巴上 10% = 100%。

    最后对合成 BGR 做 HSV 的 **V 乘以 skin_v_scale**（默认约压暗 5%～10% 量级）。
    """
    skin_v_scale = float(np.clip(skin_v_scale, 0.90, 0.95))
    ih, iw = bgra.shape[:2]
    lm = landmarks.landmark
    weighted: List[Tuple[float, np.ndarray]] = []
    for lid, wt, _, psize in SKIN_SAMPLE_REGIONS:
        cx, cy = landmark_xy(lm[lid], iw, ih)
        x0, y0, pw, ph = skin_patch_rect_at(cx, cy, ih, iw, psize)
        med = median_bgr_in_patch(bgra, x0, y0, pw, ph)
        if med is not None:
            weighted.append((float(wt), med.astype(np.float64)))
    if not weighted:
        return _apply_bgr_value_scale(np.array([180.0, 200.0, 220.0], dtype=np.float64), skin_v_scale)
    sw = sum(w for w, _ in weighted)
    raw = sum(w * c for w, c in weighted) / max(sw, 1e-9)
    return _apply_bgr_value_scale(raw, skin_v_scale)


def estimate_face_luminance_grain_std(
    bgra: np.ndarray,
    landmarks,
    ih: int,
    iw: int,
) -> float:
    """
    在肤色采样的各子块内，对 **灰度** 计算标准差，取 **最大值** 作为脸部局部「颗粒/细节」强度参考。
    用于自适应脖子 film grain 强度，使噪点与原图画质大致同档。
    """
    lm = landmarks.landmark
    stds: List[float] = []
    for lid, _, _, psize in SKIN_SAMPLE_REGIONS:
        cx, cy = landmark_xy(lm[lid], iw, ih)
        x0, y0, pw, ph = skin_patch_rect_at(cx, cy, ih, iw, psize)
        if pw <= 0 or ph <= 0:
            continue
        roi = bgra[y0 : y0 + ph, x0 : x0 + pw]
        m = roi[:, :, 3].astype(np.float32) > 40.0
        if not np.any(m):
            continue
        gray = cv2.cvtColor(roi[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
        stds.append(float(np.std(gray[m])))
    if not stds:
        return 2.2
    return max(stds)


def apply_film_grain_to_neck_bgra(
    neck_bgra: np.ndarray,
    face_grain_std: float,
    grain_gain: float = GRAIN_REF_GAIN_DEFAULT,
) -> np.ndarray:
    """
    在脖子图层 **RGB** 上叠加微弱高斯白噪声（``cv2.randn``），**不改 alpha**。
    标准差 ``sigma = clip(face_grain_std * grain_gain, GRAIN_SIGMA_MIN, GRAIN_SIGMA_MAX)``，
    仅在 alpha>8 的像素上叠加，避免污染全透明区。
    """
    gain = float(np.clip(grain_gain, 0.35, 2.5))
    sigma = float(np.clip(face_grain_std * gain, GRAIN_SIGMA_MIN, GRAIN_SIGMA_MAX))
    h, w = neck_bgra.shape[:2]
    noise = np.zeros((h, w, 3), dtype=np.float32)
    cv2.randn(noise, (0.0, 0.0, 0.0), (sigma, sigma, sigma))
    m = (neck_bgra[:, :, 3].astype(np.float32) > 8.0)[:, :, np.newaxis]
    rgb = neck_bgra[:, :, :3].astype(np.float32) + noise * m
    out = neck_bgra.copy()
    out[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return out


def render_skin_sample_marked_preview(bgra: np.ndarray, landmarks) -> np.ndarray:
    """
    在输入图副本上标注 5 处肤色采样：各 10×10 矩形（异色描边）、
    块内 alpha>40 像素红点、关键点十字；角标注明权重。
    """
    h, w = bgra.shape[:2]
    out = bgra.copy()
    lm = landmarks.landmark
    # 每区不同描边颜色（BGRA）
    rect_colors = [
        (255, 255, 0, 255),
        (0, 255, 255, 255),
        (255, 0, 255, 255),
        (0, 255, 0, 255),
        (200, 120, 255, 255),
    ]
    for idx, (lid, wt, tag, psize) in enumerate(SKIN_SAMPLE_REGIONS):
        cx, cy = landmark_xy(lm[lid], w, h)
        x0, y0, pw, ph = skin_patch_rect_at(cx, cy, h, w, psize)
        col = rect_colors[idx % len(rect_colors)]
        if pw > 0 and ph > 0:
            cv2.rectangle(
                out,
                (x0, y0),
                (x0 + pw - 1, y0 + ph - 1),
                col,
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            roi_a = bgra[y0 : y0 + ph, x0 : x0 + pw, 3]
            ys, xs = np.where(roi_a > 40)
            for yi, xi in zip(ys.tolist(), xs.tolist()):
                gx, gy = x0 + int(xi), y0 + int(yi)
                cv2.circle(out, (gx, gy), 1, (0, 0, 255, 220), -1, lineType=cv2.LINE_AA)
        ax, ay = int(round(cx)), int(round(cy))
        cv2.drawMarker(
            out,
            (ax, ay),
            (0, 165, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=12,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
        label = f"{tag} {int(wt * 100)}%"
        cv2.putText(
            out,
            label,
            (max(2, x0), max(14, y0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    tip = (
        f"skin cheek{SKIN_PATCH_CHEEK_PX}x{SKIN_PATCH_CHEEK_PX} "
        f"other{SKIN_PATCH_OTHER_PX}x{SKIN_PATCH_OTHER_PX} median x5 "
        f"(L+R 50%, phil20%, fore20%, chin10%)"
    )
    cv2.putText(
        out,
        tip,
        (4, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def skin_sample_marked_output_path(result_path: str) -> str:
    """与主输出同目录，文件名加后缀 ``_skin_sample_marked.png``。"""
    dname, fname = os.path.split(result_path)
    stem, _ = os.path.splitext(fname)
    out_name = f"{stem}_skin_sample_marked.png"
    return os.path.join(dname, out_name) if dname else out_name


def odd_kernel(size: int) -> int:
    """高斯核大小必须为正奇数。"""
    size = max(3, int(size))
    if size % 2 == 0:
        size += 1
    return size


def _face_oval_graph() -> dict[int, list[int]]:
    """MediaPipe FACE_OVAL 无向邻接表。"""
    graph: dict[int, list[int]] = {}
    for a, b in mp.solutions.face_mesh.FACEMESH_FACE_OVAL:
        graph.setdefault(a, []).append(b)
        graph.setdefault(b, []).append(a)
    return graph


def bfs_path_on_graph(graph: dict[int, list[int]], start: int, end: int) -> Optional[List[int]]:
    """在图上做 BFS 得到最短（边数最少）简单路径。"""
    if start == end:
        return [start]
    q: deque[int] = deque([start])
    parent: dict[int, Optional[int]] = {start: None}
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
    cur: Optional[int] = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def jaw_index_path_through_chin() -> List[int]:
    """
    沿 FACE_OVAL 的 **下颌下缘** 走线：**172 → 152 → 397**（BFS 最短边路径）。

    若用 234→152→454，最短路径会先沿脸颊高处向颞侧，脖子上沿会从「耳/颊」起笔过宽；
    172 / 397 在轮廓上更靠近下颌角一带，顶宽更接近「喉-颌」宽度。
    """
    g = _face_oval_graph()
    p_l = bfs_path_on_graph(g, LANDMARK_LEFT_JAW_ON_OVAL, LANDMARK_CHIN_BOTTOM)
    p_r = bfs_path_on_graph(g, LANDMARK_CHIN_BOTTOM, LANDMARK_RIGHT_JAW_ON_OVAL)
    if p_l is None or p_r is None:
        return [172, 136, 148, 152, 377, 397]
    if p_l[-1] != LANDMARK_CHIN_BOTTOM or p_r[0] != LANDMARK_CHIN_BOTTOM:
        return [172, 136, 148, 152, 377, 397]
    return p_l[:-1] + p_r


def smooth_polyline_xy(pts: np.ndarray, win: int = 5) -> np.ndarray:
    """沿轮廓对 x/y 分别一维平滑（边界 replicate），减轻折线锯齿。"""
    n = pts.shape[0]
    if n < 3:
        return pts
    win = odd_kernel(min(max(3, win), max(3, n - (1 - (n % 2)))))
    pad = win // 2
    k = np.ones(win, dtype=np.float64) / win
    xs = np.convolve(np.pad(pts[:, 0], (pad, pad), mode="edge"), k, mode="valid")
    ys = np.convolve(np.pad(pts[:, 1], (pad, pad), mode="edge"), k, mode="valid")
    return np.column_stack([xs, ys])


def build_jaw_guided_neck_polygon(
    landmarks,
    w: int,
    h: int,
    chin_overlap_px: float,
    neck_depth_px: float,
    bottom_flare: float,
) -> np.ndarray:
    """
    构造闭合多边形：上边界 = 下颌下缘（FACE_OVAL 上 172—152—397 链），整体向上平移以插入下巴；
    下边界 = 上边界各点水平按 bottom_flare 从中心外扩、向下平移 neck_depth，形成上窄下宽。
    返回 shape (N, 2) float64，闭合顺序为「上边界（下颌）左→右 + 下底边右→左」。
    """
    lm = landmarks.landmark
    idxs = jaw_index_path_through_chin()
    top = np.array([landmark_xy(lm[i], w, h) for i in idxs], dtype=np.float64)
    top = smooth_polyline_xy(top, win=5)
    top[:, 1] -= float(chin_overlap_px)
    cx = float(np.mean(top[:, 0]))
    bot_y = float(np.max(top[:, 1])) + float(neck_depth_px)
    bottom = np.column_stack(
        [cx + (top[:, 0] - cx) * float(bottom_flare), np.full(len(top), bot_y, dtype=np.float64)]
    )
    # 轻微下凹的底边（更像领口弧线）：对 bottom y 做抛物线调整
    n = len(top)
    u = np.linspace(-1.0, 1.0, n)
    sag = min(6.0, neck_depth_px * 0.08)
    bottom[:, 1] += sag * (1.0 - u * u)
    poly = np.vstack([top, bottom[::-1]])
    poly[:, 0] = np.clip(poly[:, 0], 0.0, float(w - 1))
    poly[:, 1] = np.clip(poly[:, 1], 0.0, float(h - 1))
    return poly


def fill_polygon_mask(h: int, w: int, poly: np.ndarray) -> np.ndarray:
    """多边形内部 255，外部 0。"""
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(poly).astype(np.int32).reshape(1, -1, 2)
    cv2.fillPoly(mask, pts, 255, lineType=cv2.LINE_AA)
    return mask


def distance_map_to_polyline(h: int, w: int, pts: np.ndarray) -> np.ndarray:
    """
    各像素到折线（非闭合）的欧氏距离。折线栅格为 0，其余为 255，``distanceTransform`` 即到最近 0 的距离。
    """
    if pts.shape[0] < 2:
        return np.full((h, w), 1e6, dtype=np.float64)
    img = np.ones((h, w), dtype=np.uint8) * 255
    pi = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(
        img,
        [pi],
        isClosed=False,
        color=0,
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    dt = cv2.distanceTransform(img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return dt.astype(np.float64)


def neck_cylinder_shade_map(
    h: int,
    w: int,
    poly: np.ndarray,
    mask: np.ndarray,
    x_axis_shift: float = 0.0,
    spec_x_shift: float = 0.0,
    k_lit: Optional[float] = None,
    k_shadow: Optional[float] = None,
    shine_k: Optional[float] = None,
    ao_top: Optional[float] = None,
) -> np.ndarray:
    """
    在整幅图上生成圆柱侧面亮度乘子 (h, w)。

    - 水平：多边形 x 均值为轴，可叠加 ``x_axis_shift``（由左右颊亮度推断侧光）；``tanh`` 柔化径向；
      **迎光侧**（画面左侧）用 ``k_lit`` 提亮，**背光侧**用 ``k_shadow`` 压暗（可由脸部对比度自适应）。
    - 颌下 AO：到 **上沿折线**（``poly`` 前半链，对应下颌引导边）的距离变换，近颌弧压暗，
      而非整条竖直线性带。
    - 微弱水平高光带（``shine_k``），中心随 ``spec_x_shift`` 与轴一起平移。
    - **mask 内对 L 做均值归一化到 1.0**，再 clip，避免整块脖子比采样肤色偏暗。
    """
    kL = float(NECK_CYLINDER_K_LIT if k_lit is None else k_lit)
    kS = float(NECK_CYLINDER_K_SHADOW if k_shadow is None else k_shadow)
    k_spec = float(NECK_SHINE_K if shine_k is None else shine_k)
    k_ao = float(NECK_CYLINDER_AO_TOP if ao_top is None else ao_top)
    xx = np.arange(w, dtype=np.float64)[np.newaxis, :]
    yy = np.arange(h, dtype=np.float64)[:, np.newaxis]
    x_axis = float(np.mean(poly[:, 0])) + float(x_axis_shift)
    span = float(np.max(poly[:, 0]) - np.min(poly[:, 0]))
    R = max(span * 0.5, 4.0)
    radial = (xx - x_axis) / R
    radial = np.clip(radial, -1.45, 1.45)
    rad_s = np.tanh(radial * float(NECK_CYLINDER_TANH_SCALE))
    # rad_s<0 → 画面左侧：迎光；rad_s>0 → 背光
    lit = np.maximum(0.0, -rad_s)
    sh = np.maximum(0.0, rad_s)
    L = 1.0 + kL * lit - kS * sh
    y_min = float(np.min(poly[:, 1]))
    y_max = float(np.max(poly[:, 1]))
    depth = max(y_max - y_min, 1.0)
    # 窄条高光（乘性），中心在轴左侧一点并叠加 spec_x_shift
    sig = max(R * float(NECK_SHINE_SIGMA_FRAC), 2.5)
    x_spec = (
        x_axis
        - R * float(NECK_SHINE_BASE_OFFSET_FRAC)
        + float(spec_x_shift)
    )
    shine = k_spec * np.exp(-0.5 * np.square((xx - x_spec) / sig))
    L = L * (1.0 + shine)
    n_up = max(poly.shape[0] // 2, 2)
    upper = poly[:n_up].astype(np.float64)
    dt = distance_map_to_polyline(h, w, upper)
    tau = max(depth * float(NECK_AO_DT_TAU_FRAC), float(NECK_AO_DT_TAU_MIN_PX))
    ao_w = np.exp(-dt / tau)
    L = L * (1.0 - k_ao * np.power(ao_w, 0.95))
    wm = mask > 0
    if np.any(wm):
        mu = float(np.mean(L[wm]))
        L = L / max(mu, 1e-6)
    L = np.clip(L, float(NECK_CYLINDER_L_MIN), float(NECK_CYLINDER_L_MAX))
    wm_f = wm.astype(np.float64)
    return L * wm_f + (1.0 - wm_f)


def alpha_over(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    """
    Porter-Duff「over」合成：top 叠在 bottom 之上，均为 BGRA uint8。
    bottom、top 形状相同。
    """
    b = bottom.astype(np.float32) / 255.0
    t = top.astype(np.float32) / 255.0
    ba = b[..., 3:4]
    ta = t[..., 3:4]
    # 结果 alpha
    out_a = ta + ba * (1.0 - ta)
    out_a_safe = np.maximum(out_a, 1e-6)
    # 结果颜色（非预乘形式下的标准公式）
    out_rgb = t[..., :3] * ta + b[..., :3] * ba * (1.0 - ta)
    out_rgb = out_rgb / out_a_safe
    out = np.dstack([out_rgb, out_a])
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)


def build_natural_neck_layer(
    h: int,
    w: int,
    landmarks,
    bgra: np.ndarray,
    chin_y: float,
    overlap: float,
    neck_height_ratio: float,
    neck_width_scale: float,
    neck_bottom_flare: float,
    skin_bgr: np.ndarray,
    neck_slim_scale: float = NECK_SLIM_SCALE_DEFAULT,
) -> np.ndarray:
    """
    下颌引导多边形；mask 内 BGR = 采样肤色 × **局部线性肤色渐变** × **圆柱体明暗乘子**
    （径向 + 距离型 AO；K/高光/AO 由脸部 V 对比度自适应），alpha 恒为 255；
    原图 alpha 抑制（不做高斯模糊）。

    ``neck_slim_scale``：整体缩放脖子「粗细」（<1 变细），垂直深度与底边喇叭外扩同比缩小，上沿仍贴合下颌。
    """
    slim = float(np.clip(neck_slim_scale, 0.72, 1.0))
    neck_depth = max(8.0, float(h) * neck_height_ratio * 2.0) * slim
    effective_flare = float(neck_bottom_flare) * (float(neck_width_scale) / 1.08)
    effective_flare = max(1.02, min(effective_flare, 1.45))
    # 只压缩相对中心的「额外宽度」：flare=1 无额外，>1 的部分乘以 slim
    flare_slim = 1.0 + (effective_flare - 1.0) * slim

    poly = build_jaw_guided_neck_polygon(
        landmarks, w, h, overlap, neck_depth, flare_slim
    )
    mask = fill_polygon_mask(h, w, poly)

    yy = np.arange(h, dtype=np.float64)[:, np.newaxis]
    alpha_f = 255.0 * (mask.astype(np.float64) > 0.0)

    skin = np.array(skin_bgr, dtype=np.float64)
    wm = (mask > 0).astype(np.float64)
    span_x = float(np.max(poly[:, 0]) - np.min(poly[:, 0]))
    R_est = max(span_x * 0.5, 4.0)
    y_min = float(np.min(poly[:, 1]))
    y_max = float(np.max(poly[:, 1]))
    depth_n = max(y_max - y_min, 1.0)
    cxn = float(np.mean(poly[:, 0]))
    cyn = float(np.mean(poly[:, 1]))

    light = estimate_face_lighting_for_neck(bgra, landmarks, h, w, R_est)
    L = neck_cylinder_shade_map(
        h,
        w,
        poly,
        mask,
        x_axis_shift=light.x_axis_shift,
        spec_x_shift=light.spec_x_shift,
        k_lit=light.k_lit,
        k_shadow=light.k_shadow,
        shine_k=light.shine_k,
        ao_top=light.ao_top,
    )
    xx = np.arange(w, dtype=np.float64)[np.newaxis, :]
    yy = np.arange(h, dtype=np.float64)[:, np.newaxis]
    G = (
        1.0
        + light.skin_grad_gx * (xx - cxn) / max(R_est, 1.0)
        + light.skin_grad_gy * (cyn - yy) / max(depth_n, 1.0)
    )
    G = np.clip(G, 0.9, 1.12)
    wm_b = wm > 0
    if np.any(wm_b):
        G = G / max(float(np.mean(G[wm_b])), 1e-6)
    bgr = skin * L[:, :, np.newaxis] * G[:, :, np.newaxis] * wm[:, :, np.newaxis]

    layer = np.zeros((h, w, 4), dtype=np.float64)
    layer[:, :, 0] = bgr[:, :, 0]
    layer[:, :, 1] = bgr[:, :, 1]
    layer[:, :, 2] = bgr[:, :, 2]
    layer[:, :, 3] = alpha_f
    layer_u8 = np.clip(np.round(layer), 0, 255).astype(np.uint8)

    # 原图不透明处减弱脖子（避免「贴纸」进脸；抠图透明区仍显示脖子）
    oa = bgra[:, :, 3].astype(np.float64) / 255.0
    y_above_chin = np.clip((float(chin_y) - yy.astype(np.float64)) / max(float(overlap) * 3.0, 1.0), 0.0, 1.0)
    suppress = oa * (0.70 + 0.28 * y_above_chin)
    layer_u8[:, :, 3] = np.clip(
        layer_u8[:, :, 3].astype(np.float64) * (1.0 - suppress), 0, 255
    ).astype(np.uint8)

    return layer_u8


def add_fake_neck(
    bgra: np.ndarray,
    neck_width_scale: float = 1.08,
    neck_height_ratio: float = 0.26,
    chin_overlap_px: float = 17.0,
    neck_bottom_flare: float = 1.18,
    skin_v_scale: float = 0.925,
    neck_slim_scale: float = NECK_SLIM_SCALE_DEFAULT,
    grain_gain: float = GRAIN_REF_GAIN_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    核心流程（自然衔接版）：
    1) FaceMesh 检测人脸；下巴 **152**；
    2) **下颌路径**：在 FACE_OVAL 上 BFS **172→152→397**（下颌下缘），得到上边界折线，
       整体上移 `chin_overlap_px` 以插入下巴后缘；
    3) **上窄下宽**：下边界相对水平中心外扩（``neck_bottom_flare``）后再下移；经 ``neck_slim_scale``
       同比缩小「深度 + 外扩量」，脖子整体变细，上沿仍贴下颌；
    4) **颜色**：5 区采样 + HSV 压 V 得 albedo；mask 内乘 **关键点推断的肤色渐变 + 圆柱明暗**
      （V 对比度自适应 K/高光/AO，颌线距离型 AO）；alpha 满值；
    5) **Film grain**：按脸部采样块灰度标准差自适应强度，对脖子 RGB 叠加 ``cv2.randn`` 弱噪点；
    6) **原图 alpha**：脸部不透明区域按比例压低脖子 alpha，减少穿帮；
    7) **层级**：`alpha_over(脖子, 头像)`，脖子在下。

    :return: ``(合成图 BGRA, 肤色采样标注图 BGRA)``，后者与输入同尺寸，便于核对取样区域。

    :param neck_width_scale: 与 `neck_bottom_flare` 联动微调整体外扩（默认 1.08 为基准）。
    :param neck_height_ratio: 控制脖子区域垂直深度（相对图高）。
    :param neck_bottom_flare: 下颌底相对顶宽的水平放大（喇叭），约 1.12–1.25。
    :param chin_overlap_px: 顶边相对 152 向上偏移像素，钳制在 15–20。
    :param skin_v_scale: 采样肤色后在 HSV 中对 V 的乘子，建议 0.90–0.95（默认 0.925）。
    :param neck_slim_scale: 脖子整体变细比例（0.72–1），默认 0.87；同比缩小深度与底边外扩。
    :param grain_gain: 脸部灰度 std → 脖子噪点 std 的倍率，默认见 ``GRAIN_REF_GAIN_DEFAULT``。
    """
    h, w = bgra.shape[:2]
    rgb = bgra_to_rgb(bgra)

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    ) as face_mesh:
        results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        raise RuntimeError("未检测到人脸，请确认图中包含完整面部且对比度正常。")

    lm = results.multi_face_landmarks[0].landmark
    landmarks = results.multi_face_landmarks[0]
    _, chin_y = landmark_xy(lm[LANDMARK_CHIN_BOTTOM], w, h)

    overlap = float(np.clip(chin_overlap_px, CHIN_OVERLAP_MIN_PX, CHIN_OVERLAP_MAX_PX))

    skin_bgr = sample_skin_color_bgra(
        bgra, landmarks, w, h, skin_v_scale=skin_v_scale
    )

    neck_layer = build_natural_neck_layer(
        h,
        w,
        landmarks,
        bgra,
        chin_y,
        overlap,
        neck_height_ratio,
        neck_width_scale,
        neck_bottom_flare,
        skin_bgr,
        neck_slim_scale=neck_slim_scale,
    )
    grain_ref = estimate_face_luminance_grain_std(bgra, landmarks, h, w)
    neck_layer = apply_film_grain_to_neck_bgra(neck_layer, grain_ref, grain_gain=grain_gain)

    composed = alpha_over(neck_layer, bgra)
    skin_marked = render_skin_sample_marked_preview(bgra, landmarks)
    return composed, skin_marked


def default_output_path(input_path: str) -> str:
    """
    命名示例（与题目一致）：
    - 输入 person_rgba.png -> 输出 person_with_neck.png（去掉文件名中的 _rgba 后缀再加 _with_neck）
    - 其他 stem.png -> stem_with_neck.png
    """
    dname, fname = os.path.split(input_path)
    stem, ext = os.path.splitext(fname)
    if stem.lower().endswith("_rgba"):
        stem = stem[:-5]
    out_name = f"{stem}_with_neck.png"
    return os.path.join(dname, out_name) if dname else out_name


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="为 RGBA 抠图头像在下巴下生成假脖子并导出 PNG。")
    parser.add_argument("input", help="输入 RGBA PNG 路径（透明背景头像）")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="输出路径；默认在输入同目录生成 *_with_neck.png",
    )
    parser.add_argument(
        "--neck-width-scale",
        type=float,
        default=1.08,
        help="脖子宽度相对「脸轮廓顶端实测宽」的倍数，默认 1.08",
    )
    parser.add_argument(
        "--neck-height-ratio",
        type=float,
        default=0.26,
        help="脖子区域垂直深度相对图像高度的比例，默认 0.26",
    )
    parser.add_argument(
        "--chin-overlap",
        type=float,
        default=17.0,
        help="脖子顶相对下巴点(152)向上偏移像素，建议 15–20，会钳制到该范围，默认 17",
    )
    parser.add_argument(
        "--neck-bottom-flare",
        type=float,
        default=1.18,
        help="下颌底相对顶宽的水平外扩比例（喇叭），约 1.12–1.25，默认 1.18",
    )
    parser.add_argument(
        "--skin-v-scale",
        type=float,
        default=0.925,
        help="肤色采样后在 HSV 中 V 通道乘子，约 0.90–0.95（略压暗以贴合颌下阴影），默认 0.925",
    )
    parser.add_argument(
        "--neck-slim",
        type=float,
        default=NECK_SLIM_SCALE_DEFAULT,
        help="脖子整体变细比例（0.72–1），同比缩小垂直深度与底边外扩，默认 0.87",
    )
    parser.add_argument(
        "--grain-gain",
        type=float,
        default=GRAIN_REF_GAIN_DEFAULT,
        help="脸部灰度标准差映射到脖子 film grain 强度的倍率，默认 1.08",
    )
    args = parser.parse_args(argv)

    inp = os.path.abspath(args.input)
    out = os.path.abspath(args.output) if args.output else os.path.abspath(default_output_path(inp))

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    bgra = load_rgba(inp)
    try:
        out_bgra, skin_marked = add_fake_neck(
            bgra,
            neck_width_scale=args.neck_width_scale,
            neck_height_ratio=args.neck_height_ratio,
            chin_overlap_px=args.chin_overlap,
            neck_bottom_flare=args.neck_bottom_flare,
            skin_v_scale=args.skin_v_scale,
            neck_slim_scale=float(np.clip(args.neck_slim, 0.72, 1.0)),
            grain_gain=float(np.clip(args.grain_gain, 0.35, 2.5)),
        )
    except Exception as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1

    ok = imwrite_unicode(out, out_bgra)
    if not ok:
        print(f"[错误] 无法写入: {out}", file=sys.stderr)
        return 1
    print(f"已保存: {out}")

    dbg_path = skin_sample_marked_output_path(out)
    if not imwrite_unicode(dbg_path, skin_marked):
        print(f"[错误] 无法写入采样标注图: {dbg_path}", file=sys.stderr)
        return 1
    print(f"已保存采样标注: {dbg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
