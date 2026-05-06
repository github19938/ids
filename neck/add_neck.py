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

# 上沿向中心收缩比例（<1）：让脖子在颌下「内收」一段，避免顶宽=下颌宽形成「双下巴」感。
# 端点（172/397，下颌角附近）处保留 ~1.0（不动），中段最大收缩到该值，过渡用 |u|^0.7。
NECK_TOP_INSET_DEFAULT = 0.86

# 下颌跨度归一化：chin_overlap 与 neck_depth 默认按 172↔397 跨度自适应，避免硬编码像素带来分辨率敏感
JAW_SPAN_OVERLAP_FRAC = 0.06
JAW_SPAN_OVERLAP_MIN_PX = 6.0
JAW_SPAN_OVERLAP_MAX_PX = 60.0
JAW_SPAN_DEPTH_FRAC = 1.85
JAW_SPAN_DEPTH_MIN_PX = 36.0

# 沿「到上沿折线」的距离做的羽化/抑制（替代原来基于 chin_y 的硬过渡）
NECK_FEATHER_FRAC = 0.025
NECK_FEATHER_MIN_PX = 1.5
NECK_FEATHER_MAX_PX = 8.0
NECK_SUPPRESS_DECAY_FRAC = 0.06
NECK_SUPPRESS_DECAY_MIN_PX = 4.0

# 脖子相对脸部「色调偏暖偏暗」微调：在 HSV 上额外移 H、缩 S（V 已由 skin_v_scale 处理）
NECK_HUE_SHIFT_DEFAULT = 1.6
NECK_SAT_SCALE_DEFAULT = 1.05
# Reinhard Lab 颜色统计匹配（仅 mean shift）强度：把脖子 BGR 整体均值朝脸部肤色像素均值拉近
NECK_TONE_MATCH_STRENGTH = 0.40

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
# 多点估光（L/R 颊、额左右、下颌左右、额中、下巴上）→ 最小二乘拟合 2D V 梯度，
# 比单一「左右颊 V 差」对侧光/平光/俯仰光更鲁棒，并能避免「默认主光偏左」的硬编码偏置。
LIGHT_SAMPLE_LANDMARKS: List[Tuple[int, str, int]] = [
    (205, "L-cheek", SKIN_PATCH_CHEEK_PX),
    (425, "R-cheek", SKIN_PATCH_CHEEK_PX),
    (103, "L-forehead", SKIN_PATCH_OTHER_PX),
    (332, "R-forehead", SKIN_PATCH_OTHER_PX),
    (172, "L-jaw", SKIN_PATCH_OTHER_PX),
    (397, "R-jaw", SKIN_PATCH_OTHER_PX),
    (10, "forehead-mid", SKIN_PATCH_OTHER_PX),
    (200, "above-chin", SKIN_PATCH_OTHER_PX),
]
# V 梯度（lstsq 拟合 V = a + b*x_norm + c*y_norm，x/y 已用 R 归一化）→ 圆柱轴 / 高光横向偏移
NECK_LIGHT_AXIS_GAIN = 0.88
NECK_SHINE_TRACK_GAIN = 0.58
NECK_SHINE_K = 0.042
# 双层高光：窄 specular 峰（高频反射）+ 宽漫射 roll-off（皮肤次表面散射使高光柔化）
NECK_SHINE_SIGMA_FRAC = 0.10
NECK_SHINE_DIFFUSE_SIGMA_FRAC = 0.45
NECK_SHINE_NARROW_SHARE = 0.55  # 0~1：能量分配到窄峰的比例，剩下给宽 roll-off
# 「默认主光偏左」的硬编码偏置已移除，spec 中心位置完全由数据驱动；
# 仅保留一个 SAFETY 偏移：当数据里的 |gx| < 阈值（基本平光）时把 spec 略偏到 lit 侧
NECK_SHINE_FLAT_LIGHT_FALLBACK_FRAC = 0.04
# 圆柱朝光面 SSS 软化系数（0=纯线性，1=完全 sqrt）：模拟皮肤次表面散射使光「绕过」曲面
NECK_CYLINDER_SSS_ALPHA = 0.40

# 解剖弱阴影：胸锁乳突肌（SCM）双侧斜向阴影带 + 中轴弱凸起（喉结/喉体），整体强度都很小，
# 用 sin^2 余弦窗让它们在顶/底渐隐，避免穿过下颌或脖子底。
NECK_ANATOMY_SCM_OFFSET_FRAC = 0.42  # SCM 中心相对圆柱半径 R 的横向偏移
NECK_ANATOMY_SCM_SIGMA_FRAC = 0.10
NECK_ANATOMY_SCM_DEPTH = 0.025  # 多大幅度的「压暗」（multiplier 减量）
# 顶部/底部 SCM 的 V 形偏移（向外 spread）：模拟肌肉束从乳突到锁骨的走向
NECK_ANATOMY_SCM_FLARE_FRAC = 0.18  # 自顶到底，SCM 中心向外多偏移 R 的多少
NECK_ANATOMY_RIDGE_SIGMA_FRAC = 0.06
NECK_ANATOMY_RIDGE_BRIGHT = 0.012  # 中轴凸起亮度提升（multiplier 增量），弱以适配男女
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


# 经典 YCrCb 肤色范围（多数文献：Cr∈[133,173], Cb∈[77,127]）；用于过滤 patch 内的非肤色像素
# （眉毛、睫毛、痣、刘海阴影等），让中值/V 均值更稳定。过滤后若有效像素 < 阈值则回退到 alpha 过滤
# 以避免极端肤色或高光被错误剔除。
SKIN_YCRCB_CR_MIN, SKIN_YCRCB_CR_MAX = 133, 173
SKIN_YCRCB_CB_MIN, SKIN_YCRCB_CB_MAX = 77, 127
SKIN_FILTER_MIN_COUNT = 4


def _patch_skin_alpha_mask(roi_bgra: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回 (alpha_mask, skin_mask)：均为 (h,w) bool。
    - alpha_mask: roi 内 alpha > 40 的像素；
    - skin_mask: alpha_mask & YCrCb 肤色范围内的像素。
    若 skin_mask 像素数 < SKIN_FILTER_MIN_COUNT，调用方应回退用 alpha_mask。
    """
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
    """
    patch 内 **alpha>40 ∩ YCrCb 肤色范围内** 的像素 BGR 各通道取中值；
    若肤色过滤后像素数 < ``SKIN_FILTER_MIN_COUNT``，回退到仅 alpha 过滤；
    仍无有效像素则返回 None。
    """
    if pw <= 0 or ph <= 0:
        return None
    roi = bgra[y0 : y0 + ph, x0 : x0 + pw]
    am, sm = _patch_skin_alpha_mask(roi)
    if not np.any(am):
        return None
    use = sm if int(np.sum(sm)) >= SKIN_FILTER_MIN_COUNT else am
    flat = roi[:, :, :3][use].astype(np.float64)
    if flat.shape[0] == 0:
        return None
    return np.median(flat, axis=0)


def mean_hsv_v_in_patch(
    bgra: np.ndarray, x0: int, y0: int, pw: int, ph: int
) -> Optional[float]:
    """
    patch 内 **alpha>40 ∩ YCrCb 肤色范围内** 像素 HSV 的 V 均值；
    肤色过滤过严时回退到仅 alpha；无有效像素返回 None。
    """
    if pw <= 0 or ph <= 0:
        return None
    roi = bgra[y0 : y0 + ph, x0 : x0 + pw]
    am, sm = _patch_skin_alpha_mask(roi)
    if not np.any(am):
        return None
    use = sm if int(np.sum(sm)) >= SKIN_FILTER_MIN_COUNT else am
    hsv = cv2.cvtColor(roi[:, :, :3], cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float64)[use]
    if v.size == 0:
        return None
    return float(np.mean(v))


class FaceNeckLightParams(NamedTuple):
    """由脸部局部 V 推断脖子光照：轴偏移、圆柱强度、高光、AO、肤色线性渐变系数、光源方向。"""

    x_axis_shift: float
    spec_x_shift: float
    k_lit: float
    k_shadow: float
    shine_k: float
    ao_top: float
    skin_grad_gx: float
    skin_grad_gy: float
    # +1 表示主光从画面右侧来（lit 在右侧）；-1 表示从左侧；数据弱（平光）时回退 -1（与原硬编码兼容）。
    light_sign: float = -1.0


def estimate_face_lighting_for_neck(
    bgra: np.ndarray,
    landmarks,
    ih: int,
    iw: int,
    R: float,
) -> FaceNeckLightParams:
    """
    多点估光：在 ``LIGHT_SAMPLE_LANDMARKS`` 列出的左右颊 / 左右额 / 左右下颌 / 额中 / 下巴上
    一共 8 个点采小块的 **HSV-V 均值**，对位置做 R 归一化后**最小二乘拟合 V = a + b*xn + c*yn**，
    得到 2D 光梯度 ``(gx_norm, gy_norm)``：

    - ``gx_norm > 0``：右侧亮（光从右），``light_sign = +1``；反之 ``-1``；
    - 圆柱轴 / 高光带 横向偏移：均与 ``gx_norm`` 同向（替代原"左右颊 V 差"单点近似）；
    - 强度：用 8 点 V 的极差 + 标准差 + ``|gx_norm|`` 综合算 ``strength``；
    - **取消** 原"默认主光偏左"硬偏置 ``NECK_SHINE_BASE_OFFSET_FRAC=0.15``；平光时 spec 居中，
      仅当 ``|gx|`` 太小且 ``light_sign`` 仍 fallback 时给一点点 ``FLAT_LIGHT_FALLBACK`` 偏移；
    - 上下分量 ``gy_norm`` 驱动脖子 albedo 的弱竖直渐变；
    - 双颊平均明显亮于下巴上区时略加强 AO（颌下更贴重阴影照片）。
    """
    lm = landmarks.landmark
    Rf = float(max(R, 4.0))

    samples: List[Tuple[int, float, float, float]] = []
    for lid, _, psize in LIGHT_SAMPLE_LANDMARKS:
        cx, cy = landmark_xy(lm[lid], iw, ih)
        x0, y0, pw, ph = skin_patch_rect_at(cx, cy, ih, iw, psize)
        v = mean_hsv_v_in_patch(bgra, x0, y0, pw, ph)
        if v is not None:
            samples.append((int(lid), float(cx), float(cy), float(v)))
    by_lid = {s[0]: (s[1], s[2], s[3]) for s in samples}

    if len(samples) >= 4:
        xs = np.array([s[1] for s in samples], dtype=np.float64)
        ys = np.array([s[2] for s in samples], dtype=np.float64)
        vs = np.array([s[3] for s in samples], dtype=np.float64)
        cx_face = float(np.mean(xs))
        cy_face = float(np.mean(ys))
        xn = (xs - cx_face) / Rf
        yn = (ys - cy_face) / Rf
        A = np.column_stack([np.ones_like(xn), xn, yn])
        coef, *_ = np.linalg.lstsq(A, vs, rcond=None)
        v_mean = float(np.mean(vs)) + 1e-3
        gx_norm = float(np.clip(coef[1] / v_mean, -0.45, 0.45))
        gy_norm = float(np.clip(coef[2] / v_mean, -0.45, 0.45))
        v_std = float(np.std(vs))
        v_rng = (float(np.max(vs)) - float(np.min(vs))) / 255.0
    else:
        # 数据太少，回退老路径（仅依赖左右颊 + 额/下巴上 V 差）
        vl_t = by_lid.get(205, (None, None, None))[2]
        vr_t = by_lid.get(425, (None, None, None))[2]
        vf_t = by_lid.get(10, (None, None, None))[2]
        vu_t = by_lid.get(200, (None, None, None))[2]
        vals = [v for v in (vl_t, vr_t, vf_t, vu_t) if v is not None]
        v_std = float(np.std(np.array(vals, dtype=np.float64))) if len(vals) >= 2 else 0.0
        v_rng = (max(vals) - min(vals)) / 255.0 if len(vals) >= 2 else 0.0
        if vl_t is not None and vr_t is not None:
            denom = float(vl_t + vr_t) + 1e-3
            gx_norm = float(np.clip((float(vr_t) - float(vl_t)) / denom, -0.28, 0.28))
        else:
            gx_norm = 0.0
        if vf_t is not None and vu_t is not None:
            gy_norm = float(np.clip((float(vu_t) - float(vf_t)) / 255.0, -0.22, 0.22))
        else:
            gy_norm = 0.0

    lr_asym = abs(gx_norm)
    strength = float(
        NECK_LIGHT_STRENGTH_MIN
        + float(NECK_LIGHT_STRENGTH_LR_COEF) * lr_asym
        + float(NECK_LIGHT_STRENGTH_RNG_COEF) * v_rng
        + float(NECK_LIGHT_STRENGTH_STD_COEF) * (v_std / 40.0)
    )
    strength = float(np.clip(strength, NECK_LIGHT_STRENGTH_MIN, NECK_LIGHT_STRENGTH_MAX))

    # 主光方向：gx_norm > 0 表示右侧亮（光从右）→ light_sign = +1；反之 -1。
    # 数据极弱（|gx| < 0.025）时 fallback 到 -1（保持与原硬编码"主光默认偏左"的视觉风格）。
    if abs(gx_norm) >= 0.025:
        light_sign = 1.0 if gx_norm > 0.0 else -1.0
    else:
        light_sign = -1.0

    # axis_shift / spec_shift 跟随 gx_norm 同向（光从右 gx_norm>0 → 偏移到右侧）
    axis_shift = float(NECK_LIGHT_AXIS_GAIN) * gx_norm * Rf
    spec_shift = float(NECK_SHINE_TRACK_GAIN) * gx_norm * Rf

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
    vl_e = by_lid.get(205, (None, None, None))[2]
    vr_e = by_lid.get(425, (None, None, None))[2]
    vu_e = by_lid.get(200, (None, None, None))[2]
    if vu_e is not None and vl_e is not None and vr_e is not None:
        v_mid = 0.5 * (float(vl_e) + float(vr_e))
        if v_mid > float(vu_e) + 4.0:
            ao_top *= float(
                np.clip(
                    1.0
                    + float(NECK_AO_CHIN_DARK_BOOST_COEF)
                    * ((v_mid - float(vu_e)) / float(NECK_AO_CHIN_DARK_DIV)),
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

    skin_gx = float(np.clip(gx_norm, -0.22, 0.22) * float(NECK_SKIN_GRAD_GAIN_LR))
    skin_gy = float(np.clip(-gy_norm, -0.22, 0.22) * float(NECK_SKIN_GRAD_GAIN_FB))

    return FaceNeckLightParams(
        x_axis_shift=axis_shift,
        spec_x_shift=spec_shift,
        k_lit=k_lit,
        k_shadow=k_shadow,
        shine_k=shine_k,
        ao_top=ao_top,
        skin_grad_gx=skin_gx,
        skin_grad_gy=skin_gy,
        light_sign=light_sign,
    )


def _apply_neck_skin_tone(
    bgr: np.ndarray,
    v_scale: float,
    h_shift: float = 0.0,
    s_scale: float = 1.0,
) -> np.ndarray:
    """
    在 HSV 中调整 V/S/H：
    - V: ``*= v_scale``（默认 0.925 略压暗）；
    - S: ``*= s_scale``（>1 略加饱和，颌下次表面散射使脖子色比脸略饱和）；
    - H: ``+= h_shift``（OpenCV H∈[0,180]，正值向橙/暖偏，模拟皮肤 SSS 偏暖）。
    """
    px = np.clip(np.round(bgr).astype(np.uint8).reshape(1, 1, 3), 0, 255)
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[0, 0, 0] = (hsv[0, 0, 0] + float(h_shift)) % 180.0
    hsv[0, 0, 1] = float(np.clip(hsv[0, 0, 1] * float(s_scale), 0.0, 255.0))
    hsv[0, 0, 2] = float(np.clip(hsv[0, 0, 2] * float(v_scale), 0.0, 255.0))
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)[0, 0].astype(np.float64)
    return out


def _apply_bgr_value_scale(bgr: np.ndarray, v_scale: float) -> np.ndarray:
    """向后兼容包装：等价于 ``_apply_neck_skin_tone(bgr, v_scale, 0.0, 1.0)``。"""
    return _apply_neck_skin_tone(bgr, v_scale, 0.0, 1.0)


def sample_skin_color_bgra(
    bgra: np.ndarray,
    landmarks,
    w: int,
    h: int,
    skin_v_scale: float = 0.925,
    skin_h_shift: float = NECK_HUE_SHIFT_DEFAULT,
    skin_s_scale: float = NECK_SAT_SCALE_DEFAULT,
) -> np.ndarray:
    """
    肤色采样：5 个核心区域，各取 ``patch×patch`` 子块（左右颊 ``SKIN_PATCH_CHEEK_PX``，
    其余 ``SKIN_PATCH_OTHER_PX``）内 **alpha>40** 像素的 **BGR 中值**，
    再按权重做向量加权平均；某块无有效像素时丢弃该块权重并**重归一化**其余权重。

    权重：左颊 25% + 右颊 25% + 人中 20% + 额头 20% + 下巴上 10% = 100%。

    最后对合成 BGR 做 HSV 调整：V*=skin_v_scale（默认略压暗），S*=skin_s_scale（略加饱和），
    H+=skin_h_shift（向橙偏），共同模拟脖子相对脸部的「次表面散射偏暖偏暗略饱和」特征。
    """
    skin_v_scale = float(np.clip(skin_v_scale, 0.90, 0.95))
    skin_h_shift = float(np.clip(skin_h_shift, -6.0, 6.0))
    skin_s_scale = float(np.clip(skin_s_scale, 0.90, 1.20))
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
        return _apply_neck_skin_tone(
            np.array([180.0, 200.0, 220.0], dtype=np.float64),
            skin_v_scale, skin_h_shift, skin_s_scale,
        )
    sw = sum(w for w, _ in weighted)
    raw = sum(w * c for w, c in weighted) / max(sw, 1e-9)
    return _apply_neck_skin_tone(raw, skin_v_scale, skin_h_shift, skin_s_scale)


def gather_face_skin_pixels_bgr(
    bgra: np.ndarray,
    landmarks,
    ih: int,
    iw: int,
    patch: int = 22,
) -> np.ndarray:
    """
    在 5 个肤色采样点附近收集 **alpha>40** 的 BGR 像素，作为肤色统计参考池
    （供 Reinhard Lab 颜色匹配使用，比单一加权均值更能表达分布）。
    """
    lm = landmarks.landmark
    chunks: List[np.ndarray] = []
    for lid, _, _, _ in SKIN_SAMPLE_REGIONS:
        cx, cy = landmark_xy(lm[lid], iw, ih)
        x0, y0, pw, ph = skin_patch_rect_at(cx, cy, ih, iw, patch)
        if pw <= 0 or ph <= 0:
            continue
        roi = bgra[y0 : y0 + ph, x0 : x0 + pw]
        am, sm = _patch_skin_alpha_mask(roi)
        use = sm if int(np.sum(sm)) >= SKIN_FILTER_MIN_COUNT else am
        if np.any(use):
            chunks.append(roi[:, :, :3][use])
    if not chunks:
        return np.empty((0, 3), dtype=np.uint8)
    return np.vstack(chunks)


def reinhard_lab_mean_shift_bgra_inplace(
    layer_u8: np.ndarray,
    mask_bool: np.ndarray,
    ref_pixels_bgr: np.ndarray,
    strength: float = NECK_TONE_MATCH_STRENGTH,
) -> None:
    """
    对 ``layer_u8`` 在 ``mask_bool`` 内做 Reinhard Lab **均值偏移**（仅 mean shift，不改 std），
    把脖子区域的整体色调朝 ``ref_pixels_bgr`` 的肤色均值拉近 ``strength`` 比例。

    仅 shift mean、不改 std 的原因：脖子层已经包含我们故意做出的圆柱明暗与肤色渐变（std 信息），
    若同时归一化 std 会把这些着色 wash 掉。
    """
    if ref_pixels_bgr.shape[0] < 8 or not np.any(mask_bool):
        return
    bgr = layer_u8[:, :, :3]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    src = lab[mask_bool]
    if src.shape[0] < 8:
        return
    src_mean = src.mean(axis=0)
    ref_lab = cv2.cvtColor(
        ref_pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB
    ).reshape(-1, 3).astype(np.float32)
    ref_mean = ref_lab.mean(axis=0)
    s = float(np.clip(strength, 0.0, 1.0))
    shift = (ref_mean - src_mean) * s
    src_adj = src + shift
    src_adj = np.clip(src_adj, 0.0, 255.0)
    lab[mask_bool] = src_adj
    out_bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    layer_u8[:, :, :3] = out_bgr


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


# 1/f 噪声 / Laplacian 锐度参考阈值。SHARP_REF 是经验值：
# 大部分自然 8-bit 人像在 16x16 patch 上 |Laplacian| 均值 ~3-5；过强意味着图像很锐，过弱意味着糊。
SHARP_REF = 4.0
SHARP_FACTOR_MIN = 0.45
SHARP_FACTOR_MAX = 1.55
PINK_NOISE_ALPHA = 1.0  # 功率谱 ∝ 1/f^alpha；alpha=1 (1/f noise) 接近自然皮肤纹理频谱


def generate_pink_noise_2d(
    h: int,
    w: int,
    sigma: float,
    alpha: float = PINK_NOISE_ALPHA,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    生成 (h, w) float32 的 1/f^alpha 噪声场（功率谱按频率倒数衰减），
    最后归一化到目标标准差 ``sigma``。

    与白噪声相比，pink noise 在中低频能量更强、空间相关性显著，
    上叠在脖子皮肤上更接近真实皮肤"毛孔+细纹"的统计特性，而不是干净相机噪声。
    """
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((h, w)).astype(np.float32)
    F = np.fft.fft2(white)
    fx = np.fft.fftfreq(w).astype(np.float32)
    fy = np.fft.fftfreq(h).astype(np.float32)
    Fx, Fy = np.meshgrid(fx, fy)
    radius = np.sqrt(Fx * Fx + Fy * Fy)
    # 避免 DC 分量被无限放大；DC（频率=0）直接置 0
    safe_r = np.where(radius > 0, radius, 1.0)
    scale = (1.0 / safe_r) ** (alpha / 2.0)
    scale = scale.astype(np.float32)
    scale[0, 0] = 0.0
    pink = np.real(np.fft.ifft2(F * scale)).astype(np.float32)
    s = float(pink.std())
    if s > 1e-6:
        pink *= float(sigma) / s
    return pink


def estimate_face_high_freq_energy(
    bgra: np.ndarray,
    landmarks,
    ih: int,
    iw: int,
) -> float:
    """
    在肤色采样块内对灰度做 Laplacian，取 ``|Lap|`` 在 alpha&肤色掩码下的均值，
    各块之间取 **中位数**（比 max 更鲁棒，不会被一个极端 patch 拉偏）。

    返回值越大表示原图越锐；用作 film grain σ 的额外缩放因子，
    避免在被美颜/平滑过的图上把噪点叠得比脸还粗。
    """
    lm = landmarks.landmark
    vals: List[float] = []
    for lid, _, _, psize in SKIN_SAMPLE_REGIONS:
        cx, cy = landmark_xy(lm[lid], iw, ih)
        psize_use = max(int(psize), 12)
        x0, y0, pw, ph = skin_patch_rect_at(cx, cy, ih, iw, psize_use)
        if pw <= 0 or ph <= 0:
            continue
        roi = bgra[y0 : y0 + ph, x0 : x0 + pw]
        am, sm = _patch_skin_alpha_mask(roi)
        use = sm if int(np.sum(sm)) >= SKIN_FILTER_MIN_COUNT else am
        if not np.any(use):
            continue
        gray = cv2.cvtColor(roi[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        vals.append(float(np.mean(np.abs(lap[use]))))
    if not vals:
        return SHARP_REF
    return float(np.median(vals))


def apply_film_grain_to_neck_bgra(
    neck_bgra: np.ndarray,
    face_grain_std: float,
    grain_gain: float = GRAIN_REF_GAIN_DEFAULT,
    sharp_factor: float = 1.0,
    use_pink_noise: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    在脖子图层 **RGB** 上叠加微弱噪声（默认 **1/f pink noise**，3 通道独立），不改 alpha。

    ``sigma = clip(face_grain_std * grain_gain * sharp_factor,
                   GRAIN_SIGMA_MIN, GRAIN_SIGMA_MAX)``

    其中 ``sharp_factor`` 由脸部 Laplacian 高频能量驱动（``estimate_face_high_freq_energy``），
    平滑/美颜过的脸 sharp_factor 自动小于 1，避免脖子上叠出比脸更粗的颗粒。
    """
    gain = float(np.clip(grain_gain, 0.35, 2.5))
    sf = float(np.clip(sharp_factor, SHARP_FACTOR_MIN, SHARP_FACTOR_MAX))
    sigma = float(np.clip(face_grain_std * gain * sf, GRAIN_SIGMA_MIN, GRAIN_SIGMA_MAX))
    h, w = neck_bgra.shape[:2]
    if use_pink_noise:
        seed_val = seed if seed is not None else 0
        noise = np.stack([
            generate_pink_noise_2d(h, w, sigma, seed=seed_val + i * 7919)
            for i in range(3)
        ], axis=-1)
    else:
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


# 用于 solvePnP 的标准 3D 人脸点（毫米单位，常见参考模型）：鼻尖、下巴、左右眼外角、左右嘴角
HEAD_POSE_LANDMARKS = (1, 152, 33, 263, 61, 291)
# 注意：image y 轴向下，所以 canonical Y 也用「向下为正」（与原文献的"向上为正"取负），
# 这样 solvePnP 出来的 Euler 角与"图像 y-down"一致，正脸时 pitch≈0。
HEAD_POSE_CANONICAL_3D = np.array([
    [0.0,    0.0,    0.0],     # 鼻尖 (1)
    [0.0,   63.6,  -12.5],     # 下巴 (152)，下巴在画面下方 → +Y
    [-43.3, -32.7, -26.0],     # 左眼外角 (33)，眼睛在画面上方 → -Y
    [43.3,  -32.7, -26.0],     # 右眼外角 (263)
    [-28.9,  28.9, -24.1],     # 左嘴角 (61)
    [28.9,   28.9, -24.1],     # 右嘴角 (291)
], dtype=np.float64)

# 姿态修正强度常量（弧度）
POSE_PITCH_CHIN_OVERLAP_GAIN = 0.55     # chin_overlap *= clip(1 + gain * pitch, 0.55, 1.55)
POSE_PITCH_CHIN_OVERLAP_MIN = 0.55
POSE_PITCH_CHIN_OVERLAP_MAX = 1.55
POSE_YAW_ASYM_INSET_GAIN = 0.30         # 顶边不对称 inset：远侧多 inset、近侧少 inset
POSE_ROLL_MAX_RAD = 0.78                # ~45°，保护性 clip


def estimate_head_pose(
    landmarks,
    w: int,
    h: int,
) -> Tuple[float, float, float]:
    """
    返回 ``(yaw, pitch, roll)``（弧度）。约定：
    - yaw>0：头朝画面右侧转；
    - pitch>0：仰头（颌下露出更多）；
    - roll>0：头向画面右侧倾斜（眼线右端低于左端）。

    采用混合策略：
    1. **roll 必走"眼线角度"**——左眼 33 → 右眼 263 的方向角，几何上确定可靠，
       不受 solvePnP 在 6 点退化情况下的影响；
    2. **yaw / pitch 走 solvePnP**：6 个标准点 + 默认相机内参（fx=fy=w，主点=图像中心）；
    3. 若 solvePnP 算出的 roll 与"眼线角度"差 >10°，认为求解器不稳定，
       同步把 yaw / pitch 的幅度衰减 50%（保留方向）。

    失败/退化时尽量返回 (0,0,roll_from_eye)，最差也能保证 roll 正确。
    """
    lm = landmarks.landmark
    le = landmark_xy(lm[33], w, h)
    re = landmark_xy(lm[263], w, h)
    eye_dx = re[0] - le[0]
    eye_dy = re[1] - le[1]
    simple_roll = float(np.arctan2(eye_dy, eye_dx))

    image_pts = np.array(
        [landmark_xy(lm[i], w, h) for i in HEAD_POSE_LANDMARKS],
        dtype=np.float64,
    )
    K = np.array(
        [
            [float(w), 0.0, float(w) * 0.5],
            [0.0, float(w), float(h) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist = np.zeros((4, 1), dtype=np.float64)
    yaw = 0.0
    pitch = 0.0
    pnp_roll = simple_roll
    try:
        success, rvec, _tvec = cv2.solvePnP(
            HEAD_POSE_CANONICAL_3D, image_pts, K, dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
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
        # image-y-down 下 solvePnP 的 pitch 是"低头为正"，反一下符号到"仰头为正"
        yaw = pnp_yaw
        pitch = -pnp_pitch
        # 求解器与眼线角度差异 > 10° 视为退化，衰减 yaw / pitch
        if abs(pnp_roll - simple_roll) > 0.175:
            yaw *= 0.5
            pitch *= 0.5

    return (
        float(np.clip(yaw, -POSE_ROLL_MAX_RAD, POSE_ROLL_MAX_RAD)),
        float(np.clip(pitch, -POSE_ROLL_MAX_RAD, POSE_ROLL_MAX_RAD)),
        float(np.clip(simple_roll, -POSE_ROLL_MAX_RAD, POSE_ROLL_MAX_RAD)),
    )


def rotate_xy(pts: np.ndarray, cx: float, cy: float, angle: float) -> np.ndarray:
    """绕 (cx, cy) 旋转 (N,2) 点集；angle 弧度，正 = 画面 CCW（注意 OpenCV y 朝下，所以正向 = 顺时针视觉）。"""
    if abs(float(angle)) < 1e-4:
        return pts
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    dx = pts[:, 0] - float(cx)
    dy = pts[:, 1] - float(cy)
    nx = float(cx) + dx * c - dy * s
    ny = float(cy) + dx * s + dy * c
    return np.column_stack([nx, ny])


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
    top_inset: float = 1.0,
) -> np.ndarray:
    """
    构造闭合多边形：上边界 = 下颌下缘（FACE_OVAL 上 172—152—397 链），整体向上平移以插入下巴；
    下边界 = 上边界各点水平按 bottom_flare 从中心外扩、向下平移 neck_depth，形成上窄下宽。

    ``top_inset`` (<1) 让上沿在颌下整体「内收」：所有上沿点（含端点 172/397）均按比例向中心收缩。
    端点收缩后位于下颌轮廓**内侧**，上方被脸 alpha 遮蔽，不产生可见接缝；而多边形的左右边
    （端点→底边）整体内移，**正下方可见的脖子侧宽 < 下颌宽**，符合"脖子比下颌窄"的解剖事实。
    返回 shape (N, 2) float64，闭合顺序为「上边界（下颌）左→右 + 下底边右→左」。
    """
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
    light_sign: float = -1.0,
    sss_alpha: float = NECK_CYLINDER_SSS_ALPHA,
    roll: float = 0.0,
) -> np.ndarray:
    """
    在整幅图上生成圆柱侧面亮度乘子 (h, w)。

    - **方向感知** 的 lit/sh：``light_sign=+1`` 表示主光从画面右侧（lit 在右），``-1`` 从左；
      圆柱中线 = 多边形 x 均值 + ``x_axis_shift``；``tanh`` 柔化径向。
    - **SSS 软化朝光面**：朝光侧 ``lit`` 不再是纯线性，而是 ``mix(lin, sqrt(lin), sss_alpha)``，
      模拟皮肤次表面散射使光"绕过"曲面更多，过渡更软；背光侧保持线性（光不会绕到背面）。
    - **双层高光**：窄 specular 峰（高频反射）+ 宽漫射 roll-off（皮肤主要还是漫反射），
      能量按 ``NECK_SHINE_NARROW_SHARE`` 分配。中心位置为 ``x_axis + spec_x_shift``，
      平光时再叠一点 ``light_sign * FLAT_LIGHT_FALLBACK_FRAC * R`` 偏置避免完全居中。
    - 颌下 AO：到上沿折线的距离变换，近颌弧压暗。
    - mask 内对 L 做均值归一化到 1.0，再 clip。
    """
    kL = float(NECK_CYLINDER_K_LIT if k_lit is None else k_lit)
    kS = float(NECK_CYLINDER_K_SHADOW if k_shadow is None else k_shadow)
    k_spec = float(NECK_SHINE_K if shine_k is None else shine_k)
    k_ao = float(NECK_CYLINDER_AO_TOP if ao_top is None else ao_top)
    sign = float(np.sign(light_sign)) if abs(light_sign) > 1e-3 else -1.0
    sss_a = float(np.clip(sss_alpha, 0.0, 1.0))
    xx = np.arange(w, dtype=np.float64)[np.newaxis, :]
    yy_g = np.arange(h, dtype=np.float64)[:, np.newaxis]
    x_axis = float(np.mean(poly[:, 0])) + float(x_axis_shift)
    y_center = float(np.mean(poly[:, 1]))
    span = float(np.max(poly[:, 0]) - np.min(poly[:, 0]))
    R = max(span * 0.5, 4.0)
    # roll 把圆柱整体绕 (x_axis, y_center) 倾斜：径向 / 高光的「水平」方向变成 (cos, sin) 矢量。
    cos_r = float(np.cos(roll))
    sin_r = float(np.sin(roll))
    x_perp = (xx - x_axis) * cos_r + (yy_g - y_center) * sin_r
    radial = np.clip(x_perp / R, -1.45, 1.45)
    rad_s = np.tanh(radial * float(NECK_CYLINDER_TANH_SCALE))
    # 方向感知：sign=+1 → lit 在右（rad_s>0）；sign=-1 → lit 在左（rad_s<0）
    lit_lin = np.maximum(0.0, sign * rad_s)
    sh_lin = np.maximum(0.0, -sign * rad_s)
    # 朝光面 SSS 软化（sqrt-like 让光绕过曲面更多）；背光面保线性
    lit = (1.0 - sss_a) * lit_lin + sss_a * np.sqrt(np.maximum(lit_lin, 0.0))
    L = 1.0 + kL * lit - kS * sh_lin
    y_min = float(np.min(poly[:, 1]))
    y_max = float(np.max(poly[:, 1]))
    depth = max(y_max - y_min, 1.0)

    # 双层高光：窄 specular + 宽 diffuse roll-off；中心位置在「旋转后径向 = 偏移量 / R」处
    sig_n = max(R * float(NECK_SHINE_SIGMA_FRAC), 2.0)
    sig_w = max(R * float(NECK_SHINE_DIFFUSE_SIGMA_FRAC), 4.0)
    narrow_share = float(np.clip(NECK_SHINE_NARROW_SHARE, 0.0, 1.0))
    # 平光（spec_x_shift 极小）时给一点点方向偏置，避免高光卡在轴中心显假
    flat_fallback = 0.0
    if abs(spec_x_shift) < 0.5 * R * float(NECK_SHINE_FLAT_LIGHT_FALLBACK_FRAC):
        flat_fallback = sign * R * float(NECK_SHINE_FLAT_LIGHT_FALLBACK_FRAC)
    spec_offset = float(spec_x_shift) + flat_fallback
    spec_dist = x_perp - spec_offset
    shine_n = (k_spec * narrow_share) * np.exp(-0.5 * np.square(spec_dist / sig_n))
    shine_w = (k_spec * (1.0 - narrow_share)) * np.exp(-0.5 * np.square(spec_dist / sig_w))
    shine = shine_n + shine_w
    L = L * (1.0 + shine)

    n_up = max(poly.shape[0] // 2, 2)
    upper = poly[:n_up].astype(np.float64)
    dt = distance_map_to_polyline(h, w, upper)
    tau = max(depth * float(NECK_AO_DT_TAU_FRAC), float(NECK_AO_DT_TAU_MIN_PX))
    ao_w = np.exp(-dt / tau)
    L = L * (1.0 - k_ao * np.power(ao_w, 0.95))

    # 解剖弱修饰：双侧 SCM（胸锁乳突肌）阴影带 + 中轴弱凸起；与圆柱一同绕 roll 旋转。
    # 沿"垂直"方向（旋转坐标 y_perp = -(xx-x_axis)*sin + (yy-y_center)*cos）计算 yn，
    # 让 SCM 与中轴的"上下"始终对齐脖子主轴而不是图像 y 轴。
    y_perp = -(xx - x_axis) * sin_r + (yy_g - y_center) * cos_r
    half_h = max(depth * 0.5, 1.0)
    yn = np.clip((y_perp / half_h) * 0.5 + 0.5, 0.0, 1.0)
    vert_win = np.sin(np.pi * yn) ** 2
    # SCM 横向位置随旋转后深度自顶到底向外略 flare（V 形）
    scm_offset = R * (
        float(NECK_ANATOMY_SCM_OFFSET_FRAC)
        + float(NECK_ANATOMY_SCM_FLARE_FRAC) * yn
    )
    sig_scm = max(R * float(NECK_ANATOMY_SCM_SIGMA_FRAC), 1.5)
    g_l = np.exp(-0.5 * np.square((x_perp - (-scm_offset)) / sig_scm))
    g_r = np.exp(-0.5 * np.square((x_perp - (+scm_offset)) / sig_scm))
    L = L * (1.0 - float(NECK_ANATOMY_SCM_DEPTH) * (g_l + g_r) * vert_win)
    # 中轴凸起
    sig_ridge = max(R * float(NECK_ANATOMY_RIDGE_SIGMA_FRAC), 1.0)
    g_c = np.exp(-0.5 * np.square(x_perp / sig_ridge))
    L = L * (1.0 + float(NECK_ANATOMY_RIDGE_BRIGHT) * g_c * vert_win)

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
    neck_top_inset: float = NECK_TOP_INSET_DEFAULT,
    jaw_span_px: Optional[float] = None,
    neck_depth_override_px: Optional[float] = None,
    tone_match_strength: float = NECK_TONE_MATCH_STRENGTH,
    head_yaw: float = 0.0,
    head_pitch: float = 0.0,
    head_roll: float = 0.0,
) -> np.ndarray:
    """
    下颌引导多边形；mask 内 BGR = 采样肤色 × **局部线性肤色渐变** × **圆柱体明暗乘子**
    （径向 + 距离型 AO；K/高光/AO 由脸部 V 对比度自适应）；
    再做 **Reinhard Lab 均值偏移** 把整体色调朝脸部肤色均值拉近 ``tone_match_strength`` 比例。

    边缘处理：
    - **mask 高斯羽化**：消硬边，过渡半径按 ``jaw_span_px`` 自适应（无该值则回退按图高近似）；
    - **沿到上沿折线的距离做 sigmoid 抑制**：脸部不透明区在「靠近下颌弧」处压低脖子 alpha，
      远离下颌弧（即靠近脖子腹部）保持满 alpha——比原 ``oa*(0.70+0.28*y_above_chin)`` 的硬过渡更自然。

    ``neck_slim_scale``：整体缩放脖子「粗细」（<1 变细），垂直深度与底边喇叭外扩同比缩小，上沿仍贴合下颌。
    ``neck_top_inset``：上沿端点保持下颌角位置，中段向中心收缩，让脖子可见侧窄于下颌。
    ``jaw_span_px``：172↔397 跨度（像素），用于把羽化/抑制衰减常数标定到与人脸大小同尺度。
    ``neck_depth_override_px``：若提供则直接使用该值作脖子垂直深度（自适应路径），忽略 ``neck_height_ratio``。
    """
    slim = float(np.clip(neck_slim_scale, 0.72, 1.0))
    if neck_depth_override_px is not None:
        neck_depth = max(8.0, float(neck_depth_override_px)) * slim
    else:
        neck_depth = max(8.0, float(h) * neck_height_ratio * 2.0) * slim
    effective_flare = float(neck_bottom_flare) * (float(neck_width_scale) / 1.08)
    effective_flare = max(1.02, min(effective_flare, 1.45))
    flare_slim = 1.0 + (effective_flare - 1.0) * slim
    top_inset = float(np.clip(neck_top_inset, 0.70, 1.0))

    poly = build_jaw_guided_neck_polygon(
        landmarks, w, h, overlap, neck_depth, flare_slim, top_inset=top_inset
    )

    # 姿态修正之 yaw：远离相机的一侧（与 yaw 同号）顶边再多 inset 一点（更窄），
    # 近相机一侧少 inset，让脖子的"近-远"对比与脸部 yaw 一致。
    if abs(head_yaw) > 1e-3:
        n_top_y = poly.shape[0] // 2
        top = poly[:n_top_y]
        bot = poly[n_top_y:]
        cx_top = float(np.mean(top[:, 0]))
        # yaw>0（头朝右转）：左半（x<cx）更靠近相机 → 少 inset；右半（x>cx）远 → 多 inset
        sign_dir = np.sign(top[:, 0] - cx_top)  # +1 right, -1 left
        extra_factor = 1.0 - float(POSE_YAW_ASYM_INSET_GAIN) * float(head_yaw) * sign_dir
        extra_factor = np.clip(extra_factor, 0.7, 1.3)
        top[:, 0] = cx_top + (top[:, 0] - cx_top) * extra_factor
        # 底边按同样比例做（保持顶底连续）
        sign_b = np.sign(bot[:, 0] - cx_top)
        extra_b = 1.0 - float(POSE_YAW_ASYM_INSET_GAIN) * float(head_yaw) * sign_b
        extra_b = np.clip(extra_b, 0.7, 1.3)
        bot[:, 0] = cx_top + (bot[:, 0] - cx_top) * extra_b
        poly = np.vstack([top, bot])

    # 姿态修正之 roll：绕下巴点旋转整个多边形
    if abs(head_roll) > 1e-3:
        chin_x_lm, chin_y_lm = landmark_xy(landmarks.landmark[LANDMARK_CHIN_BOTTOM], w, h)
        poly = rotate_xy(poly, chin_x_lm, chin_y_lm, head_roll)
        poly[:, 0] = np.clip(poly[:, 0], 0.0, float(w - 1))
        poly[:, 1] = np.clip(poly[:, 1], 0.0, float(h - 1))

    mask = fill_polygon_mask(h, w, poly)

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
        light_sign=light.light_sign,
        roll=float(head_roll),
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

    # 上沿折线（poly 前半段）—— 用作距离场，驱动 alpha 抑制 / 羽化
    n_up = max(poly.shape[0] // 2, 2)
    upper_pts = poly[:n_up].astype(np.float64)
    dt_upper = distance_map_to_polyline(h, w, upper_pts)

    # mask 羽化：边缘有平滑过渡而非硬切边
    span_for_scale = float(jaw_span_px) if jaw_span_px is not None else float(span_x)
    feather_sigma = float(np.clip(
        span_for_scale * NECK_FEATHER_FRAC, NECK_FEATHER_MIN_PX, NECK_FEATHER_MAX_PX
    ))
    feather_k = odd_kernel(int(round(feather_sigma * 3.0)) + 1)
    mask_soft = cv2.GaussianBlur(
        mask.astype(np.float32), (feather_k, feather_k), feather_sigma
    ).astype(np.float64)
    alpha_f = np.clip(mask_soft, 0.0, 255.0)

    layer = np.zeros((h, w, 4), dtype=np.float64)
    layer[:, :, :3] = bgr
    layer[:, :, 3] = alpha_f
    layer_u8 = np.clip(np.round(layer), 0, 255).astype(np.uint8)

    # 颜色统计匹配：Reinhard Lab mean shift（仅迁移 mean，不动 std；保留圆柱明暗细节）
    if tone_match_strength > 0.0:
        ref_pixels = gather_face_skin_pixels_bgr(bgra, landmarks, h, w, patch=22)
        match_mask = alpha_f > 8.0
        reinhard_lab_mean_shift_bgra_inplace(
            layer_u8, match_mask, ref_pixels, strength=float(tone_match_strength)
        )

    # 距离驱动的 alpha 抑制：在脸部不透明区，越靠近下颌折线越压低脖子 alpha；越往脖子腹部越保留
    oa = bgra[:, :, 3].astype(np.float64) / 255.0
    decay = float(max(span_for_scale * NECK_SUPPRESS_DECAY_FRAC, NECK_SUPPRESS_DECAY_MIN_PX))
    boundary_envelope = np.exp(-dt_upper / decay)
    suppress = oa * boundary_envelope
    layer_u8[:, :, 3] = np.clip(
        layer_u8[:, :, 3].astype(np.float64) * (1.0 - suppress), 0, 255
    ).astype(np.uint8)

    return layer_u8


def add_fake_neck(
    bgra: np.ndarray,
    neck_width_scale: float = 1.08,
    neck_height_ratio: float = 0.26,
    chin_overlap_px: Optional[float] = None,
    neck_bottom_flare: float = 1.18,
    skin_v_scale: float = 0.925,
    neck_slim_scale: float = NECK_SLIM_SCALE_DEFAULT,
    grain_gain: float = GRAIN_REF_GAIN_DEFAULT,
    neck_top_inset: float = NECK_TOP_INSET_DEFAULT,
    skin_h_shift: float = NECK_HUE_SHIFT_DEFAULT,
    skin_s_scale: float = NECK_SAT_SCALE_DEFAULT,
    tone_match_strength: float = NECK_TONE_MATCH_STRENGTH,
    auto_scale_by_jaw: bool = True,
    pose_correction: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    核心流程（自然衔接版）：
    1) FaceMesh 检测人脸；下巴 **152**；并计算 **下颌跨度** ``jaw_span = ||172-397||``；
    2) **下颌路径**：在 FACE_OVAL 上 BFS **172→152→397**（下颌下缘），得到上边界折线，
       整体上移 `chin_overlap_px` 以插入下巴后缘；``auto_scale_by_jaw`` 时该值与脖子深度均按
       jaw_span 自适应（替代硬编码相对图高）；
    3) **上窄下宽**：上沿端点处保留下颌角形态、中段按 ``neck_top_inset`` 向中心收缩，
       下边界相对水平中心外扩（``neck_bottom_flare``）后再下移；经 ``neck_slim_scale`` 同比缩小
       「深度 + 外扩量」；
    4) **颜色**：5 区采样 + HSV 调整（V 压暗 + S 略加饱和 + H 略偏暖）得 albedo；
       mask 内乘 **关键点推断的肤色渐变 + 圆柱明暗**（V 对比度自适应 K/高光/AO，颌线距离型 AO）；
       再做 **Reinhard Lab 均值偏移** 把脖子整体色调朝脸部肤色均值拉近 ``tone_match_strength`` 比例；
    5) **边缘**：mask 高斯羽化（消硬边）+ 沿到上沿折线的距离做 sigmoid 抑制（替代基于 chin_y 的硬过渡）；
    6) **Film grain**：按脸部采样块灰度标准差自适应强度，对脖子 RGB 叠加 ``cv2.randn`` 弱噪点；
    7) **层级**：`alpha_over(脖子, 头像)`，脖子在下。

    :return: ``(合成图 BGRA, 肤色采样标注图 BGRA)``，后者与输入同尺寸，便于核对取样区域。

    :param neck_width_scale: 与 `neck_bottom_flare` 联动微调整体外扩（默认 1.08 为基准）。
    :param neck_height_ratio: 控制脖子区域垂直深度（相对图高）。``auto_scale_by_jaw=True`` 时被忽略。
    :param neck_bottom_flare: 下颌底相对顶宽的水平放大（喇叭），约 1.12–1.25。
    :param chin_overlap_px: 顶边相对 152 向上偏移像素；``None`` + ``auto_scale_by_jaw=True`` 时按
        jaw_span 自适应（``jaw_span * 0.06``，钳 6–60 px）；显式传值则使用该值（钳 6–60）。
    :param skin_v_scale: 采样肤色后在 HSV 中对 V 的乘子，建议 0.90–0.95（默认 0.925）。
    :param neck_slim_scale: 脖子整体变细比例（0.72–1），默认 0.87；同比缩小深度与底边外扩。
    :param grain_gain: 脸部灰度 std → 脖子噪点 std 的倍率，默认见 ``GRAIN_REF_GAIN_DEFAULT``。
    :param neck_top_inset: 上沿中段向中心收缩比例（0.70–1.0），默认 0.86，让脖子可见侧窄于下颌。
    :param skin_h_shift: 采样肤色 HSV 中 H 偏移（OpenCV H∈[0,180]，正向橙偏），默认 1.6。
    :param skin_s_scale: 采样肤色 HSV 中 S 缩放，默认 1.05（略加饱和）。
    :param tone_match_strength: Reinhard Lab 均值偏移强度（0=关，1=完全对齐脸部均值），默认 0.40。
    :param auto_scale_by_jaw: 若 True 且 ``chin_overlap_px is None``，按 jaw_span 自动定 overlap、
        depth 与边缘羽化/抑制衰减常数，使行为不随分辨率变化；False 时退回原相对图高的硬编码尺寸。
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

    jaw_l_xy = landmark_xy(lm[LANDMARK_LEFT_JAW_ON_OVAL], w, h)
    jaw_r_xy = landmark_xy(lm[LANDMARK_RIGHT_JAW_ON_OVAL], w, h)
    jaw_span = float(np.hypot(jaw_l_xy[0] - jaw_r_xy[0], jaw_l_xy[1] - jaw_r_xy[1]))
    jaw_span = max(jaw_span, 12.0)

    if chin_overlap_px is None:
        if auto_scale_by_jaw:
            overlap = float(np.clip(
                jaw_span * JAW_SPAN_OVERLAP_FRAC,
                JAW_SPAN_OVERLAP_MIN_PX, JAW_SPAN_OVERLAP_MAX_PX,
            ))
        else:
            overlap = float(np.clip(17.0, JAW_SPAN_OVERLAP_MIN_PX, JAW_SPAN_OVERLAP_MAX_PX))
    else:
        overlap = float(np.clip(
            chin_overlap_px, JAW_SPAN_OVERLAP_MIN_PX, JAW_SPAN_OVERLAP_MAX_PX
        ))

    if auto_scale_by_jaw:
        depth_override: Optional[float] = max(
            jaw_span * JAW_SPAN_DEPTH_FRAC, JAW_SPAN_DEPTH_MIN_PX
        )
    else:
        depth_override = None

    if pose_correction:
        yaw_rad, pitch_rad, roll_rad = estimate_head_pose(landmarks, w, h)
        # pitch 调 chin_overlap：仰头 (pitch>0) 颌下露出更多 → overlap 加大；俯首减小
        pitch_scale = float(np.clip(
            1.0 + POSE_PITCH_CHIN_OVERLAP_GAIN * pitch_rad,
            POSE_PITCH_CHIN_OVERLAP_MIN, POSE_PITCH_CHIN_OVERLAP_MAX,
        ))
        overlap = float(np.clip(
            overlap * pitch_scale,
            JAW_SPAN_OVERLAP_MIN_PX, JAW_SPAN_OVERLAP_MAX_PX,
        ))
    else:
        yaw_rad = pitch_rad = roll_rad = 0.0

    skin_bgr = sample_skin_color_bgra(
        bgra, landmarks, w, h,
        skin_v_scale=skin_v_scale,
        skin_h_shift=skin_h_shift,
        skin_s_scale=skin_s_scale,
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
        neck_top_inset=neck_top_inset,
        jaw_span_px=jaw_span,
        neck_depth_override_px=depth_override,
        tone_match_strength=tone_match_strength,
        head_yaw=yaw_rad,
        head_pitch=pitch_rad,
        head_roll=roll_rad,
    )
    grain_ref = estimate_face_luminance_grain_std(bgra, landmarks, h, w)
    sharp_energy = estimate_face_high_freq_energy(bgra, landmarks, h, w)
    sharp_factor = float(np.clip(sharp_energy / SHARP_REF, SHARP_FACTOR_MIN, SHARP_FACTOR_MAX))
    neck_layer = apply_film_grain_to_neck_bgra(
        neck_layer, grain_ref, grain_gain=grain_gain, sharp_factor=sharp_factor
    )

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
        default=-1.0,
        help="脖子顶相对下巴点(152)向上偏移像素；<0 表示按下颌跨度自适应（默认）。"
        "显式传值会被钳到 6–60 px",
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
    parser.add_argument(
        "--neck-top-inset",
        type=float,
        default=NECK_TOP_INSET_DEFAULT,
        help="上沿中段向中心收缩比例（0.70–1.0），默认 0.86；"
        "端点保留下颌角形态，让脖子可见侧窄于下颌（避免双下巴感）",
    )
    parser.add_argument(
        "--skin-h-shift",
        type=float,
        default=NECK_HUE_SHIFT_DEFAULT,
        help="肤色 HSV 中 H 偏移（OpenCV H∈[0,180]，正向橙偏），默认 1.6",
    )
    parser.add_argument(
        "--skin-s-scale",
        type=float,
        default=NECK_SAT_SCALE_DEFAULT,
        help="肤色 HSV 中 S 缩放（>1 略加饱和），默认 1.05",
    )
    parser.add_argument(
        "--tone-match",
        type=float,
        default=NECK_TONE_MATCH_STRENGTH,
        help="Reinhard Lab 均值偏移强度（0=关，1=完全对齐脸部肤色均值），默认 0.40",
    )
    parser.add_argument(
        "--no-auto-scale",
        dest="auto_scale_by_jaw",
        action="store_false",
        help="禁用按下颌跨度自适应；改用 --neck-height-ratio 等基于图高的硬编码相对量",
    )
    parser.set_defaults(auto_scale_by_jaw=True)
    args = parser.parse_args(argv)

    inp = os.path.abspath(args.input)
    out = os.path.abspath(args.output) if args.output else os.path.abspath(default_output_path(inp))

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    bgra = load_rgba(inp)
    chin_overlap_arg: Optional[float] = (
        None if args.chin_overlap is None or args.chin_overlap < 0 else float(args.chin_overlap)
    )
    try:
        out_bgra, skin_marked = add_fake_neck(
            bgra,
            neck_width_scale=args.neck_width_scale,
            neck_height_ratio=args.neck_height_ratio,
            chin_overlap_px=chin_overlap_arg,
            neck_bottom_flare=args.neck_bottom_flare,
            skin_v_scale=args.skin_v_scale,
            neck_slim_scale=float(np.clip(args.neck_slim, 0.72, 1.0)),
            grain_gain=float(np.clip(args.grain_gain, 0.35, 2.5)),
            neck_top_inset=float(np.clip(args.neck_top_inset, 0.70, 1.0)),
            skin_h_shift=float(np.clip(args.skin_h_shift, -6.0, 6.0)),
            skin_s_scale=float(np.clip(args.skin_s_scale, 0.90, 1.20)),
            tone_match_strength=float(np.clip(args.tone_match, 0.0, 1.0)),
            auto_scale_by_jaw=bool(args.auto_scale_by_jaw),
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
