# -*- coding: utf-8 -*-
"""
add_neckv1.py
=============
真实感重构版本（Phase A + B + C + D + E，独立于 add_neck.py 不影响其行为）。

设计要点：
  - **不删旧代码**：``add_neck.py`` 完整保留作为 fallback；本文件复用其几何/光照/姿态/估光等
    底层助手，并在它生成的"过程化先验"上跑：
  - **Phase B（纹理迁移）** 三种可选方案：
      - ``noise``（默认）：1/f pink noise，频谱近似皮肤毛孔，无脸结构污染。
      - ``tile``：从脸颊取单块肤色 ROI 镜像平铺，保留真实毛孔但可能有微弱对称感。
      - ``patchquilt``：Efros-Freeman image quilting + min-cut seam 拼接（旧方案，对平滑皮肤接缝可见）。
  - **Phase C（边缘羽化）** 用 ``pymatting`` closed-form alpha matting 在 chin 弧线区域
    解 ``I = α·F + (1-α)·B``，得到结构感知的 sub-pixel alpha，替代旧的 GaussianBlur 羽化。
  - **Phase D（纹理叠加）** 将高频 detail（毛孔纹理）叠加到过程化层（光影正确）之上：
      low-freq（光影/AO）← procedural，high-freq（毛孔）← quilted/noise。
  - **Phase E（颜色迁移）** 分通道 Lab 处理：L 通道 mean shift（保留 detail std），
      a/b 通道 1D CDF match（色相分布形状完整迁移）。
  - **Phase A（重构）** ``procedural_neck_init`` + ``realism_pipeline`` 两段拆分，
    每个 Phase 都可独立开关 + 自带 fallback。

入口与 ``add_neck.py`` 的 ``add_fake_neck`` 对等：``add_fake_neck_v1``。CLI 全部新参数齐备。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, NamedTuple, Optional, Tuple

import cv2
import numpy as np

# === 复用 add_neck.py 的成熟基础模块（不重复实现）======================================
# 优先按包导入；若本脚本被直接运行（cwd 在 neck/ 内），fallback 同目录导入。
try:
    from neck.add_neck import (  # type: ignore
        bgra_to_rgb, load_rgba, imwrite_unicode, landmark_xy, skin_patch_rect_at,
        build_jaw_guided_neck_polygon, fill_polygon_mask, distance_map_to_polyline, smooth_polyline_xy,
        estimate_face_lighting_for_neck, neck_cylinder_shade_map, estimate_head_pose, rotate_xy,
        sample_skin_color_bgra, gather_lower_face_skin_pixels_bgr, _patch_skin_alpha_mask,
        SKIN_FILTER_MIN_COUNT, SKIN_SAMPLE_REGIONS,
        NECK_TOP_INSET_DEFAULT, NECK_SLIM_SCALE_DEFAULT, NECK_MASK_SUPERSAMPLE,
        NECK_FEATHER_FRAC, NECK_FEATHER_MIN_PX, NECK_FEATHER_MAX_PX,
        NECK_SUPPRESS_DECAY_FRAC, NECK_SUPPRESS_DECAY_MIN_PX, NECK_SUPPRESS_ALPHA_GAMMA,
        JAW_SPAN_OVERLAP_FRAC, JAW_SPAN_OVERLAP_MIN_PX, JAW_SPAN_OVERLAP_MAX_PX,
        JAW_SPAN_DEPTH_FRAC, JAW_SPAN_DEPTH_MIN_PX,
        POSE_PITCH_CHIN_OVERLAP_GAIN, POSE_PITCH_CHIN_OVERLAP_MIN, POSE_PITCH_CHIN_OVERLAP_MAX,
        POSE_YAW_ASYM_INSET_GAIN, LANDMARK_CHIN_BOTTOM, LANDMARK_LEFT_JAW_ON_OVAL, LANDMARK_RIGHT_JAW_ON_OVAL,
        NECK_HUE_SHIFT_DEFAULT, NECK_SAT_SCALE_DEFAULT, odd_kernel,
        estimate_existing_neck_extent_px, COLLAR_DETECT_MIN_FRAC, COLLAR_DEPTH_SHRINK_MIN_PX,
        detect_face_with_retry, _detect_first_face, alpha_over, render_skin_sample_marked_preview,
        generate_pink_noise_2d, GRAIN_SIGMA_MIN, GRAIN_SIGMA_MAX,
    )
except ImportError:  # 直接 `python add_neckv1.py` 时 cwd 是 neck/，无 neck 包
    from add_neck import (  # type: ignore
        bgra_to_rgb, load_rgba, imwrite_unicode, landmark_xy, skin_patch_rect_at,
        build_jaw_guided_neck_polygon, fill_polygon_mask, distance_map_to_polyline, smooth_polyline_xy,
        estimate_face_lighting_for_neck, neck_cylinder_shade_map, estimate_head_pose, rotate_xy,
        sample_skin_color_bgra, gather_lower_face_skin_pixels_bgr, _patch_skin_alpha_mask,
        SKIN_FILTER_MIN_COUNT, SKIN_SAMPLE_REGIONS,
        NECK_TOP_INSET_DEFAULT, NECK_SLIM_SCALE_DEFAULT, NECK_MASK_SUPERSAMPLE,
        NECK_FEATHER_FRAC, NECK_FEATHER_MIN_PX, NECK_FEATHER_MAX_PX,
        NECK_SUPPRESS_DECAY_FRAC, NECK_SUPPRESS_DECAY_MIN_PX, NECK_SUPPRESS_ALPHA_GAMMA,
        JAW_SPAN_OVERLAP_FRAC, JAW_SPAN_OVERLAP_MIN_PX, JAW_SPAN_OVERLAP_MAX_PX,
        JAW_SPAN_DEPTH_FRAC, JAW_SPAN_DEPTH_MIN_PX,
        POSE_PITCH_CHIN_OVERLAP_GAIN, POSE_PITCH_CHIN_OVERLAP_MIN, POSE_PITCH_CHIN_OVERLAP_MAX,
        POSE_YAW_ASYM_INSET_GAIN, LANDMARK_CHIN_BOTTOM, LANDMARK_LEFT_JAW_ON_OVAL, LANDMARK_RIGHT_JAW_ON_OVAL,
        NECK_HUE_SHIFT_DEFAULT, NECK_SAT_SCALE_DEFAULT, odd_kernel,
        estimate_existing_neck_extent_px, COLLAR_DETECT_MIN_FRAC, COLLAR_DEPTH_SHRINK_MIN_PX,
        detect_face_with_retry, _detect_first_face, alpha_over, render_skin_sample_marked_preview,
        generate_pink_noise_2d, GRAIN_SIGMA_MIN, GRAIN_SIGMA_MAX,
    )


# =====================================================================================
# 下颌锚定肤色采样（替代 sample_skin_color_bgra 默认）
# =====================================================================================
# 当前 sample_skin_color_bgra 用 5 区（含额头 20% + 双颊 50%）权重，对自然光人像
# 输出的 skin albedo 与「下颌-颈连接处」的真实肤色 **能差 30+ Lab L**——脖子绘制
# 出来普遍偏亮偏粉。改为只用下颌附近的 4 个点：下巴上 / 下巴尖 / 双下颌角，
# 完全忽略脸颊、额头、人中。
NECK_ANCHOR_LANDMARKS: Tuple[Tuple[int, float, str, int], ...] = (
    # 实测 imagev2 脖子色 ≈ 双颊色而非下颌色（AI 生成倾向脖子与脸主体一致）。
    # 把权重重平衡到「双颊主导 + 下巴辅助」，颌角去掉（颌角通常偏暗有阴影）。
    (205, 0.32, "L-cheek",    14),
    (425, 0.32, "R-cheek",    14),
    (164, 0.16, "philtrum",   12),
    (200, 0.10, "above-chin", 12),
    (152, 0.10, "chin-tip",   10),
)


def sample_neck_anchor_skin_color(
    bgra: np.ndarray,
    landmarks,
    w: int,
    h: int,
    skin_v_scale: float = 0.92,
    skin_h_shift: float = 0.0,
    skin_s_scale: float = 1.0,
) -> np.ndarray:
    """
    从下颌锚定区（``NECK_ANCHOR_LANDMARKS``）采样肤色，权重和：
    下巴上 30% + 下巴尖 30% + 左下颌角 20% + 右下颌角 20% = 100%。
    每块取 patch 内 alpha>40 ∩ YCrCb 肤色的 BGR 中值；无效块自动剔除并归一化权重。
    最后做轻 HSV 调整（V 略压暗、H/S 默认不动）。

    与 ``sample_skin_color_bgra`` 的区别：完全不取额头 / 双颊 / 人中。
    脖子物理上接的是下巴，色调应该和下颌-颈连接处一致而不是脸均值。
    """
    try:
        from neck.add_neck import median_bgr_in_patch, _apply_neck_skin_tone
    except ImportError:
        from add_neck import median_bgr_in_patch, _apply_neck_skin_tone

    skin_v_scale = float(np.clip(skin_v_scale, 0.85, 1.05))
    skin_h_shift = float(np.clip(skin_h_shift, -6.0, 6.0))
    skin_s_scale = float(np.clip(skin_s_scale, 0.90, 1.20))

    lm = landmarks.landmark
    weighted: List[Tuple[float, np.ndarray]] = []
    for lid, wt, _, psize in NECK_ANCHOR_LANDMARKS:
        cx, cy = landmark_xy(lm[lid], w, h)
        x0, y0, pw, ph = skin_patch_rect_at(cx, cy, h, w, psize)
        med = median_bgr_in_patch(bgra, x0, y0, pw, ph)
        if med is not None:
            weighted.append((float(wt), med.astype(np.float64)))
    if not weighted:
        return _apply_neck_skin_tone(
            np.array([170.0, 175.0, 200.0], dtype=np.float64),
            skin_v_scale, skin_h_shift, skin_s_scale,
        )
    sw = sum(w for w, _ in weighted)
    raw = sum(w * c for w, c in weighted) / max(sw, 1e-9)
    return _apply_neck_skin_tone(raw, skin_v_scale, skin_h_shift, skin_s_scale)


# 方案 5：胡茬检测 + 颌下暗调
# 实测 imagev2 chin+15% R=224 比 chin+50% R=240 暗 16 单位——这就是胡茬投影。
# 我们当前 procedural 只有 -8% 的 AO，缺这个色调偏移。
# 检测：比较下巴上(200) 与双颊(205, 425) 的 V 通道差。颊比下巴亮 → 有胡茬阴影。
STUBBLE_DETECT_LANDMARK_CHIN = 200          # 下巴上（胡茬区中心）
STUBBLE_DETECT_LANDMARKS_CHEEK = (205, 425) # 双颊（无胡茬区）
STUBBLE_DETECT_PATCH_PX = 14
STUBBLE_DETECT_THRESHOLD = 4.0              # V 差 > 4 视为有胡茬
STUBBLE_DETECT_FULL_RANGE = 24.0            # V 差 ≥ 24 → 强度 1.0
STUBBLE_TOP_BAND_FRAC = 0.40                # 暗调影响脖子顶端 40% 区域
STUBBLE_DECAY_FRAC = 0.13                   # 颌线距离指数衰减（小=暗调集中在顶端不外溢）
STUBBLE_INTENSITY_MAX = 32.0                # max BGR 减量（强度=1时）
STUBBLE_NOISE_SIGMA = 3.0                   # 胡茬区随机斑点 sigma（让阴影不均匀）


def _stubble_patch_v(bgra, lid, lm, h, w):
    cx, cy = landmark_xy(lm[lid], w, h)
    x0, y0, pw, ph = skin_patch_rect_at(cx, cy, h, w, STUBBLE_DETECT_PATCH_PX)
    if pw <= 0 or ph <= 0:
        return None
    roi = bgra[y0 : y0 + ph, x0 : x0 + pw]
    am = roi[:, :, 3] > 40
    if not am.any():
        return None
    hsv = cv2.cvtColor(roi[:, :, :3], cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 2][am].mean())


def detect_stubble_strength(bgra: np.ndarray, landmarks, h: int, w: int) -> float:
    """
    估计胡茬阴影强度（0~1）。
    比较下巴上（lid 200，胡茬区中心）与双颊（lid 205/425，无胡茬区）的 V 通道差：
    颊比下巴亮 N 单位 → strength = clip((N - 4) / 20, 0, 1)。
    """
    lm = landmarks.landmark
    v_chin = _stubble_patch_v(bgra, STUBBLE_DETECT_LANDMARK_CHIN, lm, h, w)
    v_cheek_vals = []
    for lid in STUBBLE_DETECT_LANDMARKS_CHEEK:
        v = _stubble_patch_v(bgra, lid, lm, h, w)
        if v is not None:
            v_cheek_vals.append(v)
    if v_chin is None or not v_cheek_vals:
        return 0.0
    v_cheek = float(np.mean(v_cheek_vals))
    diff = v_cheek - v_chin   # 颊亮于下巴的程度
    if diff < STUBBLE_DETECT_THRESHOLD:
        return 0.0
    return float(np.clip(
        (diff - STUBBLE_DETECT_THRESHOLD) / (STUBBLE_DETECT_FULL_RANGE - STUBBLE_DETECT_THRESHOLD),
        0.0, 1.0,
    ))


def apply_stubble_shadow_inplace(
    layer_bgra: np.ndarray,
    polygon_mask_u8: np.ndarray,
    poly: np.ndarray,
    jaw_span: float,
    strength: float,
    rng_seed: int = 12345,
) -> None:
    """
    在脖子顶端区域（chin 接缝处）叠加暗调，模拟胡茬投影。
    暗调强度沿 chin 折线距离指数衰减：颌下最深，向下渐隐。
    叠加微弱噪声让阴影不均匀（更像真实胡茬）。
    """
    if strength <= 0.01:
        return
    h, w = polygon_mask_u8.shape
    n_up = max(poly.shape[0] // 2, 2)
    upper = poly[:n_up].astype(np.float64)
    dt_top = distance_map_to_polyline(h, w, upper)
    depth = max(float(np.max(poly[:, 1]) - np.min(poly[:, 1])), 1.0)
    tau = max(depth * STUBBLE_DECAY_FRAC, 6.0)
    decay = np.exp(-dt_top / tau)
    decay = decay.astype(np.float32)

    # 微弱噪声让暗调不均匀
    rng = np.random.default_rng(rng_seed + 1)
    noise = rng.standard_normal((h, w)).astype(np.float32) * STUBBLE_NOISE_SIGMA
    noise = cv2.GaussianBlur(noise, (5, 5), 1.0)

    mask_bool = polygon_mask_u8 >= 1
    intensity = (decay * (STUBBLE_INTENSITY_MAX * float(strength)) + noise * float(strength))
    intensity = np.clip(intensity, 0.0, 32.0)
    bgr = layer_bgra[:, :, :3].astype(np.float32)
    # 非等量减：B 多减 5%，R 少减 5%，模拟真实胡茬阴影的微暖色调
    stubble_tint = np.array([1.05, 1.0, 0.95], dtype=np.float32)  # BGR tint
    bgr -= intensity[..., None] * stubble_tint * mask_bool[..., None].astype(np.float32)
    layer_bgra[:, :, :3] = np.clip(bgr, 0, 255).astype(np.uint8)


# Reinhard tone-match 的 ref pool 用「双颊主导 + 人中」，**不含下巴**（下巴常处于阴影会
# 拉低 ref mean）。让 ref mean 尽量接近脸主体亮度，从而脖子色匹配真正的脸色而非下颌阴影色。
NECK_REF_POOL_LANDMARKS: Tuple[int, ...] = (10, 205, 425, 280, 164)
# 实测脸最亮 5 点：额中 / 双颊 / 右颊上 / 人中。去掉了 50/67/297 等被头发阴影污染的点。
# 注：仅作 fallback；默认 add_fake_neck_v1 会调 gather_face_oval_skin_pixels 取整脸全像素。


def _build_polygon_from_edges(
    edges, lm, w: int, h: int
) -> Optional[np.ndarray]:
    """
    从 mediapipe 的 FACEMESH_* edge set 构造闭合 polygon 的 (N, 2) 像素坐标。
    edge set 是 (start, end) 集合，可能多组无序，返回单一闭环（最大连通分量）。
    """
    if not edges:
        return None
    g: dict = {}
    for a, b in edges:
        g.setdefault(a, set()).add(b)
        g.setdefault(b, set()).add(a)
    # 找一个起点，沿邻接走环
    start = min(g.keys())
    visited = {start}
    order = [start]
    cur = start
    while True:
        nxts = [n for n in g[cur] if n not in visited]
        if not nxts:
            break
        # 走能延续环的下一点（任意取一个邻居即可）
        nx = nxts[0]
        visited.add(nx)
        order.append(nx)
        cur = nx
    if len(order) < 3:
        return None
    pts = np.array([landmark_xy(lm[i], w, h) for i in order], dtype=np.float64)
    return pts


FACE_OVAL_BRIGHTEST_FRAC = 0.40
# 整脸 mean 包含 oval 边缘的鬓角/颌底等偏暗像素 → 与 5-landmark 选最亮点比反而暗。
# 改取整脸最亮 40% 像素的 mean 作 ref：
#   - 仍保留大样本（30k 中取 12k）的统计稳定性，
#   - 又自动避开鬓角/颌底/局部阴影，对齐脸亮区。


def gather_face_oval_skin_pixels(
    bgra: np.ndarray,
    landmarks,
    h: int,
    w: int,
    brightest_frac: float = FACE_OVAL_BRIGHTEST_FRAC,
) -> np.ndarray:
    """
    **方案 2（CPU 纯算法路径优化）**：从整张脸的「最亮 brightest_frac」肤色像素采样，
    作为 Reinhard tone-match 的 ref pool。

    步骤：
      1. 用 ``FACEMESH_FACE_OVAL`` 构造闭合 polygon mask（脸轮廓内部）
      2. 减去 ``FACEMESH_LEFT_EYE / RIGHT_EYE / LIPS / LEFT_EYEBROW /
         RIGHT_EYEBROW / NOSE`` 各自 polygon
      3. 与 alpha>40 ∩ YCrCb 肤色范围相交
      4. 按 luminance（Rec.601 灰度）排序，取最亮 ``brightest_frac`` (默认 40%)
      5. 返回该子集像素 (N, 3)

    样本量从原 (5 landmarks patches) ≈ ~1000 px 提升到 ~12,000-20,000 px，
    mean 估计标准误降 ~4×；同时通过亮度过滤避开鬓角/颌底/局部阴影，
    比"5-landmark 手挑亮点"更不依赖 landmark 选位。
    """
    try:
        import mediapipe as _mp
    except ImportError:
        return np.empty((0, 3), dtype=np.uint8)
    fm = _mp.solutions.face_mesh
    lm = landmarks.landmark

    oval_pts = _build_polygon_from_edges(fm.FACEMESH_FACE_OVAL, lm, w, h)
    if oval_pts is None:
        return np.empty((0, 3), dtype=np.uint8)

    oval_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(oval_mask, [np.round(oval_pts).astype(np.int32).reshape(1, -1, 2)], 255)

    # 减去眼/眉/嘴/鼻
    feature_attrs = (
        "FACEMESH_LEFT_EYE", "FACEMESH_RIGHT_EYE", "FACEMESH_LIPS",
        "FACEMESH_LEFT_EYEBROW", "FACEMESH_RIGHT_EYEBROW", "FACEMESH_NOSE",
    )
    for attr in feature_attrs:
        edges = getattr(fm, attr, None)
        if edges is None:
            continue
        feat_pts = _build_polygon_from_edges(edges, lm, w, h)
        if feat_pts is None or feat_pts.shape[0] < 3:
            continue
        cv2.fillPoly(
            oval_mask,
            [np.round(feat_pts).astype(np.int32).reshape(1, -1, 2)],
            0,
        )

    # 与 alpha & YCrCb 肤色相交
    am = bgra[:, :, 3] > 40
    ycrcb = cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]; cb = ycrcb[:, :, 2]
    skin = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    final = (oval_mask > 0) & am & skin
    if not np.any(final):
        return np.empty((0, 3), dtype=np.uint8)
    pixels = bgra[:, :, :3][final]
    if pixels.shape[0] < 100:
        return pixels
    # 取最亮 brightest_frac 子集（避开鬓角/颌底/局部阴影）
    frac = float(np.clip(brightest_frac, 0.05, 1.0))
    if frac < 0.999:
        # Rec.601 luminance: 0.114*B + 0.587*G + 0.299*R
        lum = (
            0.114 * pixels[:, 0].astype(np.float32)
            + 0.587 * pixels[:, 1].astype(np.float32)
            + 0.299 * pixels[:, 2].astype(np.float32)
        )
        thr = float(np.quantile(lum, 1.0 - frac))
        keep = lum >= thr
        if int(np.sum(keep)) >= 100:
            pixels = pixels[keep]
    return pixels


def _gather_neck_ref_pixels(
    bgra: np.ndarray,
    landmarks,
    h: int,
    w: int,
    patch: int = 24,
) -> np.ndarray:
    """
    收集「脖子色应当对齐」的参考像素池：颊主导 + 人中，不含下巴/下颌角。
    每块 patch×patch ROI 内取 alpha>40 ∩ YCrCb 肤色的像素。
    """
    lm = landmarks.landmark
    chunks: List[np.ndarray] = []
    for lid in NECK_REF_POOL_LANDMARKS:
        cx, cy = landmark_xy(lm[lid], w, h)
        x0, y0, pw, ph = skin_patch_rect_at(cx, cy, h, w, patch)
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


# 下巴过渡区采样：取下巴线正上方的窄条肤色像素，作为脖子顶部需要直接衔接的颜色参考。
CHIN_STRIP_HEIGHT_FRAC = 0.08   # 窄条高度 = jaw_span × 8%
CHIN_STRIP_WIDTH_FRAC = 0.70    # 窄条宽度 = jaw_span × 70%（居中，避开两侧颌角阴影）
CHIN_STRIP_REF_WEIGHT = 0.15    # 在 ref pool 中所占的比例（之前 50% 让脖子整体偏暗 12 单位）

# 原图脖子区域采样（source_image 路径）：在 chin 下方提取真实肤色像素作 ref
SOURCE_NECK_Y_START_FRAC = 0.10   # 起始 y = chin_y + jaw_span × 0.10（避开 chin AA 边界）
SOURCE_NECK_Y_END_FRAC = 0.55     # 结束 y = chin_y + jaw_span × 0.55（避开衣领）
SOURCE_NECK_HALF_WIDTH_FRAC = 0.20  # 横向 ±jaw_span × 0.20
SOURCE_NECK_MIN_PIXELS = 200      # 至少 200 个肤色像素才认为"原图有脖子"


def gather_chin_strip_skin_pixels(
    bgra: np.ndarray,
    landmarks,
    h: int,
    w: int,
    jaw_span: float,
) -> np.ndarray:
    """
    在下巴线正上方取一条窄带肤色像素。这是脖子顶部需要直接衔接的区域——
    颜色迁移的参考如果只用脸部最亮像素，脖子会比下巴过渡区偏亮，形成亮带。

    窄带定义：以下巴尖 (landmark 152) 为中心，高度 = jaw_span × 8%，
    宽度 = jaw_span × 70%，只取 YCrCb 肤色 + alpha>40 的像素。

    返回 (N, 3) BGR uint8。
    """
    lm = landmarks.landmark
    strip_h = max(int(round(jaw_span * CHIN_STRIP_HEIGHT_FRAC)), 4)
    strip_w = max(int(round(jaw_span * CHIN_STRIP_WIDTH_FRAC)), 8)

    # 以下巴尖为中心
    chin_x, chin_y = landmark_xy(lm[152], w, h)
    x0 = int(np.clip(chin_x - strip_w // 2, 0, w - 1))
    y0 = int(np.clip(chin_y - strip_h, 0, h - 1))  # 正上方
    x1 = int(np.clip(x0 + strip_w, 0, w))
    y1 = int(np.clip(y0 + strip_h, 0, h))

    roi = bgra[y0:y1, x0:x1]
    if roi.size == 0:
        return np.empty((0, 3), dtype=np.uint8)

    am = roi[:, :, 3] > 40
    ycrcb = cv2.cvtColor(roi[:, :, :3], cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]; cb = ycrcb[:, :, 2]
    skin = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    valid = am & skin

    if not np.any(valid):
        # fallback：放宽到 alpha>40 即可
        valid = am
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.uint8)

    return roi[:, :, :3][valid]


# Transplant 模式的 affine 锚点：用 chin / 双下颌角三点做 affine 变换（保 scale + rotation + translate）
TRANSPLANT_ANCHOR_LANDMARKS = (152, 172, 397)
TRANSPLANT_TONE_MATCH_STRENGTH = 0.30   # 颜色微调强度：transplant 后再做轻度 Reinhard 适应 head 脸色
TRANSPLANT_INNER_SHRINK_PX = 1.0        # 内部 mask 微缩，避免 warp 边界残影


def transplant_source_neck(
    head_bgra: np.ndarray,
    head_landmarks,
    source_image_path: Optional[str],
) -> Optional[np.ndarray]:
    """
    **真实色彩还原最直接的方法**：把原图（``source_image_path``）的脖子区域用
    affine 变换 warp 到 head 坐标系，**像素级 1:1 复刻**原图脖子的颜色 + 纹理 + 光照。

    流程：
      1. 加载 source 图，FaceMesh 检测人脸（独立于 head 的 landmarks）
      2. 用 chin (152) + 左/右下颌角 (172/397) 三点构造 affine 变换：
         ``cv2.getAffineTransform(src_pts, head_pts)``
         三点 affine 同时对齐 scale + rotation + translate
      3. ``cv2.warpAffine(source, M, (head_w, head_h))`` 把 source 整图 warp 到 head 坐标系
      4. 返回 warped BGRA 图（与 head 同尺寸）；mask 外像素由 BORDER_REPLICATE 填充

    返回：
      - (h, w, 4) BGRA uint8: warped 后的 source 图
      - None: source 路径无效 / FaceMesh 检测失败 / 三点 affine 退化（共线）
    """
    if source_image_path is None or not source_image_path:
        return None
    try:
        from neck.add_neck import imread_unicode
    except ImportError:
        from add_neck import imread_unicode
    try:
        src = imread_unicode(source_image_path)
    except (FileNotFoundError, OSError, ValueError):
        return None
    if src is None or src.ndim != 3:
        return None
    if src.shape[2] == 3:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2BGRA)
        src[:, :, 3] = 255
    elif src.shape[2] != 4:
        return None

    sH, sW = src.shape[:2]
    src_rgb = cv2.cvtColor(src[:, :, :3], cv2.COLOR_BGR2RGB)
    src_landmarks = detect_face_with_retry(src_rgb)
    if src_landmarks is None:
        return None

    h, w = head_bgra.shape[:2]
    head_pts = np.array(
        [landmark_xy(head_landmarks.landmark[i], w, h) for i in TRANSPLANT_ANCHOR_LANDMARKS],
        dtype=np.float32,
    )
    src_pts = np.array(
        [landmark_xy(src_landmarks.landmark[i], sW, sH) for i in TRANSPLANT_ANCHOR_LANDMARKS],
        dtype=np.float32,
    )
    # 检查三点不共线（否则 affine 退化）
    v1 = head_pts[1] - head_pts[0]
    v2 = head_pts[2] - head_pts[0]
    cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
    if cross < 1.0:
        return None
    v1s = src_pts[1] - src_pts[0]
    v2s = src_pts[2] - src_pts[0]
    cross_s = abs(v1s[0] * v2s[1] - v1s[1] * v2s[0])
    if cross_s < 1.0:
        return None

    try:
        M = cv2.getAffineTransform(src_pts, head_pts)
    except cv2.error:
        return None

    warped = cv2.warpAffine(
        src, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped


def gather_source_neck_skin_pixels(
    source_image_path: Optional[str],
) -> Optional[np.ndarray]:
    """
    从「原图」（带真实脖子的完整人像，可以是 RGB JPG / PNG / RGBA PNG）的脖子区域
    提取真实肤色像素作为 ground-truth ref pool。

    流程：
      1. imread 加载原图（支持中文路径），无 alpha 时自动加 alpha=255
      2. 在原图上跑 FaceMesh 检测人脸（不依赖 head.png 的检测结果，因为是另一张图）
      3. 取 chin 下方 [+10% jaw_span, +55% jaw_span] × ±20% jaw_span 的矩形 ROI
         （避开 chin AA 边界 + 避开衣领）
      4. alpha>40 ∩ YCrCb 肤色范围过滤
      5. 至少 SOURCE_NECK_MIN_PIXELS=200 个肤色像素才认为"原图有脖子"

    返回：
      - (N, 3) BGR uint8：原图脖子真实肤色像素池，N ≥ 200
      - None：source_image_path 为空 / 文件不存在 / 无人脸 / 脖子区域无足够肤色
    """
    if source_image_path is None or not source_image_path:
        return None
    try:
        from neck.add_neck import imread_unicode
    except ImportError:
        from add_neck import imread_unicode

    try:
        src = imread_unicode(source_image_path)
    except (FileNotFoundError, OSError, ValueError):
        return None
    if src is None or src.ndim != 3:
        return None
    if src.shape[2] == 3:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2BGRA)
        src[:, :, 3] = 255
    elif src.shape[2] != 4:
        return None

    sH, sW = src.shape[:2]
    src_rgb = cv2.cvtColor(src[:, :, :3], cv2.COLOR_BGR2RGB)

    src_landmarks = detect_face_with_retry(src_rgb)
    if src_landmarks is None:
        return None

    lm = src_landmarks.landmark
    chin_x = float(lm[152].x * sW)
    chin_y = float(lm[152].y * sH)
    jaw_l = (lm[172].x * sW, lm[172].y * sH)
    jaw_r = (lm[397].x * sW, lm[397].y * sH)
    jaw_span = float(np.hypot(jaw_l[0] - jaw_r[0], jaw_l[1] - jaw_r[1]))
    if jaw_span < 12.0:
        return None

    # 脖子 ROI: chin 下方一段 + 中央 ±0.2 jaw_span 宽
    y_start = int(np.clip(chin_y + jaw_span * SOURCE_NECK_Y_START_FRAC, 0, sH))
    y_end = int(np.clip(chin_y + jaw_span * SOURCE_NECK_Y_END_FRAC, 0, sH))
    half_w = int(jaw_span * SOURCE_NECK_HALF_WIDTH_FRAC)
    x_start = int(np.clip(chin_x - half_w, 0, sW))
    x_end = int(np.clip(chin_x + half_w, 0, sW))
    if y_end <= y_start + 5 or x_end <= x_start + 5:
        return None

    roi = src[y_start:y_end, x_start:x_end]
    am = roi[:, :, 3] > 40
    ycrcb = cv2.cvtColor(roi[:, :, :3], cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    skin = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    valid = am & skin

    n_skin = int(np.sum(valid))
    if n_skin < SOURCE_NECK_MIN_PIXELS:
        return None

    return roi[:, :, :3][valid]


# =====================================================================================
# Phase B: Image Quilting （从脸部真实皮肤拼接到脖子区域）
# =====================================================================================

# 经典 Efros-Freeman 2001。本实现要点：
#   1) 从 SKIN_SAMPLE_REGIONS 周围更大的 ROI（~3x patch size）抽 patch；
#   2) 每个候选 patch 必须 ≥80% 落在 YCrCb 肤色范围；
#   3) 在 neck 多边形 bbox 内 raster scan，patch 间留 overlap≈patch/3；
#   4) 选 patch 时按 SSD on overlap 取 top-K，再随机选一个，避免肉眼可见的重复；
#   5) 在 overlap 区做 min-cut seam（DP）缝合，避免直边接缝。

# --- Image Quilting 旧方案保留（实测在小尺寸 portrait + 平滑皮肤上 patch 接缝可见）------
QUILT_PATCH_FRAC_OF_JAW = 0.10
QUILT_PATCH_MIN_PX = 16
QUILT_PATCH_MAX_PX = 56
QUILT_OVERLAP_FRAC = 0.38
QUILT_K_BEST_CANDIDATES = 6
QUILT_LIBRARY_TARGET_SIZE = 240
QUILT_LIBRARY_SAMPLE_BOX_FRAC = 3.0
QUILT_MIN_SKIN_RATIO = 0.80
QUILT_RNG_SEED = 12345

# --- 新默认方案：1/f Pink Noise Detail（频谱接近皮肤毛孔，无脸结构污染）-------------
# face_detail_tile 取脸颊 ROI 镜像平铺会把脸上的眼角/鼻翼阴影等大结构也带进 detail，
# 在脖子上显出"漏斗形阴影"。改用 1/f pink noise 做 detail：频谱与真实皮肤毛孔接近
# 但是纯合成纹理，**完全没有脸的结构污染**。强度参考 imagev2 实测 std≈11.7。
PINK_NOISE_SIGMA = 4.5                    # imagev2 实测高频 std≈1，对应 sigma≈4-5
PINK_NOISE_ALPHA = 0.3                    # 1/f^0.3 接近白噪声，能量集中在高频（细毛孔）
PINK_NOISE_BLUR_SIGMA = 1.3               # 1.3px 让噪点细腻不显砂砾


def _compute_noise_bgr_scale(
    bgra: np.ndarray,
    landmarks,
    h: int,
    w: int,
) -> np.ndarray:
    """
    从脸部锚定肤色 Lab a 通道自适应计算 pink noise 的 BGR 通道比例。
    返回 (3,) float32，[B_scale, G_scale, R_scale]，中心=1.0。

    - a_ref > 0（暖肤色，红润）：R 偏强、B 偏弱
    - a_ref < 0（冷肤色，偏绿/蓝）：B 偏强、R 偏弱
    - a_ref ≈ 0（中性）：三通道等比

    比例幅度 = clip(|a_ref| / 10, 0.01, 0.08)，自动缩放。
    """
    try:
        from neck.add_neck import median_bgr_in_patch
    except ImportError:
        from add_neck import median_bgr_in_patch

    lm = landmarks.landmark
    # 取双颊 (205, 425) 的中值 BGR → Lab a
    bgr_vals: List[np.ndarray] = []
    for lid in (205, 425):
        cx, cy = landmark_xy(lm[lid], w, h)
        x0, y0, pw, ph = skin_patch_rect_at(cx, cy, h, w, 14)
        med = median_bgr_in_patch(bgra, x0, y0, pw, ph)
        if med is not None:
            bgr_vals.append(med.astype(np.float32))

    if not bgr_vals:
        # fallback：中性比例
        return np.array([1.0, 1.0, 1.0], dtype=np.float32)

    avg_bgr = np.mean(bgr_vals, axis=0).reshape(1, 1, 3).astype(np.uint8)
    lab = cv2.cvtColor(avg_bgr, cv2.COLOR_BGR2LAB)
    a_ref = float(lab[0, 0, 1]) - 128.0  # OpenCV Lab a 中心=128

    # 比例幅度：a_ref 绝对值越大，通道偏差越大
    amp = float(np.clip(abs(a_ref) / 10.0, 0.01, 0.08))
    if a_ref > 0:
        # 暖肤色：R 偏强，B 偏弱
        scale = np.array([1.0 - amp, 1.0, 1.0 + amp], dtype=np.float32)
    else:
        # 冷肤色：B 偏强，R 偏弱
        scale = np.array([1.0 + amp, 1.0, 1.0 - amp], dtype=np.float32)
    return scale

# --- Face Detail Tiling（旧路径，保留作可选）-----------------------------------------
DETAIL_TILE_SIZE_FRAC = 0.20              # 0.45→0.20：更小 tile 减少非纹理结构混入
DETAIL_TILE_MIN_PX = 24
DETAIL_TILE_MAX_PX = 80
DETAIL_TILE_LANDMARKS = (205, 425, 10)
DETAIL_INTENSITY_CLIP = 18.0
DETAIL_EDGE_FEATHER_PX = 3                # 8→3：减少边缘"模糊感"，让 detail 接近边缘也保留

# Mean-Match σ 必须 **远大于** detail tile 大小才能消除 tile 级色彩漂移；
# 实测 σ ≥ tile_size 即可干净消除任何 tile 级低频差异。
QUILT_MEAN_MATCH_SIGMA_FRAC = 4.0


def _masked_gaussian_blur(
    img: np.ndarray, mask_f: np.ndarray, sigma: float, ksize: int
) -> np.ndarray:
    """
    Mask-aware Gaussian blur：只考虑 mask 内像素的卷积。返回 (h,w,3) float32。
    img_blur(p) = sum(img(q)*mask(q)*g(p,q)) / sum(mask(q)*g(p,q))
    避免 mask 外 0 值让 mask 边缘的 lf 偏低（以及 detail 假性偏高）。
    """
    img_f = img.astype(np.float32)
    if img_f.ndim == 2:
        img_f = img_f[..., None]
    mw_2d = mask_f if mask_f.ndim == 2 else mask_f[..., 0]
    mw_2d = mw_2d.astype(np.float32)
    mw = mw_2d[..., None]
    weighted_img = img_f * mw
    blur_img = cv2.GaussianBlur(weighted_img, (ksize, ksize), sigma)
    # 注意 OpenCV 对 (h,w,1) 输入可能返回 (h,w)，统一转 (h,w,1)
    blur_w_2d = cv2.GaussianBlur(mw_2d, (ksize, ksize), sigma)
    blur_w = np.maximum(blur_w_2d, 1e-3)[..., None]
    out = blur_img / blur_w
    return out


def build_face_skin_patch_library(
    bgra: np.ndarray,
    landmarks,
    patch_size: int,
    target_n: int = QUILT_LIBRARY_TARGET_SIZE,
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """
    在 ``SKIN_SAMPLE_REGIONS`` 各关键点周围 ``QUILT_LIBRARY_SAMPLE_BOX_FRAC * patch_size``
    大小的方形 ROI 内做随机滑窗，提取 ``patch_size × patch_size`` 的 BGR uint8 patch。
    每个 patch 必须 ``alpha>40`` 且 ≥``QUILT_MIN_SKIN_RATIO`` 落在 YCrCb 肤色范围。
    返回 patch 列表（≤ target_n）。
    """
    if rng is None:
        rng = np.random.default_rng(QUILT_RNG_SEED)
    ih, iw = bgra.shape[:2]
    lm = landmarks.landmark
    box = max(int(round(patch_size * QUILT_LIBRARY_SAMPLE_BOX_FRAC)), patch_size + 4)
    patches: List[np.ndarray] = []
    per_region_target = max(1, target_n // max(len(SKIN_SAMPLE_REGIONS), 1))
    for lid, _, _, _ in SKIN_SAMPLE_REGIONS:
        cx, cy = landmark_xy(lm[lid], iw, ih)
        bx0, by0, bw, bh = skin_patch_rect_at(cx, cy, ih, iw, box)
        if bw < patch_size or bh < patch_size:
            continue
        roi = bgra[by0 : by0 + bh, bx0 : bx0 + bw]
        am, sm = _patch_skin_alpha_mask(roi)
        if int(np.sum(am)) < patch_size * patch_size:
            continue
        for _ in range(per_region_target * 4):  # 试 4x，过滤掉不合格的
            x = int(rng.integers(0, bw - patch_size + 1))
            y = int(rng.integers(0, bh - patch_size + 1))
            cand = roi[y : y + patch_size, x : x + patch_size]
            cand_am, cand_sm = _patch_skin_alpha_mask(cand)
            if int(np.sum(cand_am)) < patch_size * patch_size * 0.95:
                continue
            if int(np.sum(cand_sm)) < patch_size * patch_size * QUILT_MIN_SKIN_RATIO:
                continue
            patches.append(cand[:, :, :3].copy())
            if len(patches) >= target_n:
                return patches
    return patches


def _min_cut_path(cost: np.ndarray, axis: int) -> np.ndarray:
    """
    在 cost (H,W) 上沿 ``axis`` 找最小累积代价路径（DP），返回每行/列的切点位置。
    axis=0：top→bottom 走，输出 shape=(H,) 的列号；
    axis=1：left→right 走，输出 shape=(W,) 的行号。
    """
    if axis == 1:
        cost = cost.T
    H, W = cost.shape
    acc = cost.astype(np.float64).copy()
    for i in range(1, H):
        prev = acc[i - 1]
        left = np.concatenate([[np.inf], prev[:-1]])
        right = np.concatenate([prev[1:], [np.inf]])
        acc[i] += np.minimum(np.minimum(prev, left), right)
    cuts = np.zeros(H, dtype=np.int64)
    cuts[-1] = int(np.argmin(acc[-1]))
    for i in range(H - 2, -1, -1):
        j = cuts[i + 1]
        cands = [(j, acc[i, j])]
        if j > 0:
            cands.append((j - 1, acc[i, j - 1]))
        if j < W - 1:
            cands.append((j + 1, acc[i, j + 1]))
        cuts[i] = min(cands, key=lambda kv: kv[1])[0]
    return cuts


def _seam_mask(
    canvas_overlap: np.ndarray,
    new_overlap: np.ndarray,
    canvas_filled: np.ndarray,
    direction: str,
) -> np.ndarray:
    """
    对 overlap 区计算 min-cut seam，返回 bool mask：True 表示用 new patch 像素，False 用旧 canvas。
    direction: 'left'（new 从左侧来，纵向 seam）/ 'top'（new 从上侧来，横向 seam）。
    canvas_filled 是 (h,w) bool 表示 canvas overlap 内的有效像素（已被前面填过）；
    未填过的列/行 cost 设为 0（强制选 new）。
    """
    diff = np.sum((canvas_overlap.astype(np.float32) - new_overlap.astype(np.float32)) ** 2, axis=-1)
    diff[~canvas_filled] = 0.0  # 没有旧内容的位置，强制选 new（cost 0）
    if direction == "left":
        cuts = _min_cut_path(diff, axis=0)
        H, W = diff.shape
        cols = np.arange(W)[None, :]
        return cols >= cuts[:, None]
    H, W = diff.shape
    cuts = _min_cut_path(diff, axis=1)
    rows = np.arange(H)[:, None]
    return rows >= cuts[None, :]


def face_detail_tile_into_mask(
    bgra: np.ndarray,
    landmarks,
    target_h: int,
    target_w: int,
    target_mask_bool: np.ndarray,
    jaw_span_px: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    新方案（替代 image quilting）：从脸部 ``DETAIL_TILE_LANDMARKS`` 关键点周围取
    **单块** 最大连续肤色 ROI（tile），用 ``cv2.copyMakeBorder(BORDER_REFLECT_101)``
    镜像平铺到 mask 区域内。

    优势：tile 内部本来就连续无接缝；BORDER_REFLECT 在 tile 边界做镜像反射
    （像撑开的羽翼）也是无接缝的，只是会有一个微弱的"反射对称感"——但因为后续
    Mean-Match 会把低频抹掉只留 detail，反射对称的低频部分被消除，只留下毛孔级
    的真实纹理。这是经典图形学「detail texture overlay」做法。

    返回 (h, w, 3) BGR uint8，mask 外像素 = 0。
    """
    if rng is None:
        rng = np.random.default_rng(QUILT_RNG_SEED)
    out = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    if not np.any(target_mask_bool):
        return out

    tile_size = int(np.clip(round(jaw_span_px * DETAIL_TILE_SIZE_FRAC), DETAIL_TILE_MIN_PX, DETAIL_TILE_MAX_PX))
    half = tile_size // 2
    lm = landmarks.landmark
    ih, iw = bgra.shape[:2]
    tile: Optional[np.ndarray] = None
    for lid in DETAIL_TILE_LANDMARKS:
        cx, cy = landmark_xy(lm[lid], iw, ih)
        x0 = int(round(cx - half))
        y0 = int(round(cy - half))
        x1 = x0 + tile_size
        y1 = y0 + tile_size
        if x0 < 0 or y0 < 0 or x1 > iw or y1 > ih:
            continue
        cand = bgra[y0:y1, x0:x1]
        am, sm = _patch_skin_alpha_mask(cand)
        if int(np.sum(sm)) >= tile_size * tile_size * 0.85:
            tile = cand[:, :, :3].copy()
            break
    if tile is None:
        # 退而求其次：取任一点位置足够大的 alpha 全开 patch
        for lid in DETAIL_TILE_LANDMARKS:
            cx, cy = landmark_xy(lm[lid], iw, ih)
            x0, y0, pw, ph = skin_patch_rect_at(cx, cy, ih, iw, tile_size)
            if pw == tile_size and ph == tile_size:
                tile = bgra[y0:y0+ph, x0:x0+pw, :3].copy()
                break
    if tile is None:
        return out

    # 找 mask bbox
    ys, xs = np.where(target_mask_bool)
    y0_m, y1_m = int(ys.min()), int(ys.max()) + 1
    x0_m, x1_m = int(xs.min()), int(xs.max()) + 1
    mh = y1_m - y0_m
    mw = x1_m - x0_m
    if mh <= 0 or mw <= 0:
        return out

    # 用 cv2.copyMakeBorder + BORDER_REFLECT_101 把 tile 镜像扩展到 mw x mh（不少于）
    pad_top = (mh + tile_size) // 2
    pad_left = (mw + tile_size) // 2
    big = cv2.copyMakeBorder(tile, pad_top, pad_top, pad_left, pad_left, cv2.BORDER_REFLECT_101)
    # 在 big 中随机选一个 mh x mw 的窗口（带轻微抖动）作为 detail tile
    bh, bw = big.shape[:2]
    rx = int(rng.integers(0, max(bw - mw, 1) + 1))
    ry = int(rng.integers(0, max(bh - mh, 1) + 1))
    crop = big[ry:ry + mh, rx:rx + mw]
    if crop.shape[0] != mh or crop.shape[1] != mw:
        crop = cv2.resize(crop, (mw, mh), interpolation=cv2.INTER_LINEAR)

    # 写入 mask 区域；mask 外保持 0
    region_mask = target_mask_bool[y0_m:y1_m, x0_m:x1_m]
    region_out = out[y0_m:y1_m, x0_m:x1_m]
    region_out[region_mask] = crop[region_mask]
    return out


def image_quilt_in_mask(
    target_h: int,
    target_w: int,
    target_mask_bool: np.ndarray,
    library: List[np.ndarray],
    patch_size: int,
    overlap: int,
    rng: Optional[np.random.Generator] = None,
    k_best: int = QUILT_K_BEST_CANDIDATES,
) -> np.ndarray:
    """
    在 ``target_mask_bool`` 内用 image quilting 拼接出一张 (h,w,3) BGR uint8 图。
    target_mask_bool 外像素值 = 0。
    """
    if rng is None:
        rng = np.random.default_rng(QUILT_RNG_SEED + 1)
    out = np.zeros((target_h, target_w, 3), dtype=np.float32)
    filled = np.zeros((target_h, target_w), dtype=bool)
    if not library or not np.any(target_mask_bool):
        return out.astype(np.uint8)

    step = max(patch_size - overlap, 1)
    lib_arr = np.stack(library, axis=0).astype(np.float32)  # (L, p, p, 3)
    L = lib_arr.shape[0]

    last_chosen = -1
    for y in range(0, target_h - patch_size + 1, step):
        for x in range(0, target_w - patch_size + 1, step):
            cell_mask = target_mask_bool[y : y + patch_size, x : x + patch_size]
            if int(np.sum(cell_mask)) < patch_size * patch_size * 0.20:
                continue

            # 计算 overlap 区与候选库 patch 的 SSD（顶/左方向各取一段）
            canvas_patch = out[y : y + patch_size, x : x + patch_size]
            canvas_filled = filled[y : y + patch_size, x : x + patch_size]
            costs = np.zeros(L, dtype=np.float64)
            n_top = 1 if y > 0 else 0
            n_left = 1 if x > 0 else 0
            if n_top:
                co_t = canvas_patch[:overlap, :, :]
                cf_t = canvas_filled[:overlap, :]
                if np.any(cf_t):
                    diff_t = (lib_arr[:, :overlap, :, :] - co_t[None, ...]) ** 2
                    diff_t = diff_t.sum(axis=-1)
                    diff_t = np.where(cf_t[None, ...], diff_t, 0.0)
                    costs += diff_t.sum(axis=(1, 2))
            if n_left:
                co_l = canvas_patch[:, :overlap, :]
                cf_l = canvas_filled[:, :overlap]
                if np.any(cf_l):
                    diff_l = (lib_arr[:, :, :overlap, :] - co_l[None, ...]) ** 2
                    diff_l = diff_l.sum(axis=-1)
                    diff_l = np.where(cf_l[None, ...], diff_l, 0.0)
                    costs += diff_l.sum(axis=(1, 2))
            # 不允许连续两次同一 patch（去重复感）
            if last_chosen >= 0:
                costs[last_chosen] = costs.max() + 1.0
            top_idx = np.argpartition(costs, min(k_best, L - 1))[:k_best]
            chosen_idx = int(rng.choice(top_idx))
            last_chosen = chosen_idx
            chosen = lib_arr[chosen_idx]

            # min-cut 缝合 + 写入
            patch_mask = np.ones((patch_size, patch_size), dtype=bool)
            if n_top and np.any(canvas_filled[:overlap, :]):
                top_seam = _seam_mask(
                    canvas_patch[:overlap, :, :],
                    chosen[:overlap, :, :],
                    canvas_filled[:overlap, :],
                    "top",
                )
                patch_mask[:overlap, :] &= top_seam
            if n_left and np.any(canvas_filled[:, :overlap]):
                left_seam = _seam_mask(
                    canvas_patch[:, :overlap, :],
                    chosen[:, :overlap, :],
                    canvas_filled[:, :overlap],
                    "left",
                )
                patch_mask[:, :overlap] &= left_seam

            # 只在 target_mask 内写入；target_mask 外像素保持 0
            write = patch_mask & cell_mask
            for c in range(3):
                slab = out[y : y + patch_size, x : x + patch_size, c]
                slab[write] = chosen[:, :, c][write]
            filled[y : y + patch_size, x : x + patch_size] |= write

    return np.clip(np.round(out), 0, 255).astype(np.uint8)


# =====================================================================================
# Phase D: Laplacian 金字塔多频段融合
# =====================================================================================
# 把过程化层（光影对）和 quilted 层（纹理真）按频段交叉混合：
#   低频 → 过程化（保留圆柱明暗、AO、SCM、Pitié 之后的色调）
#   高频 → quilted（保留真实毛孔、细纹）
# 等价于 Poisson 频域近似，但更稳定 / 可控 / 实现简单。

LAPLACIAN_LEVELS_DEFAULT = 5


# =====================================================================================
# Phase C: Alpha Matting （pymatting closed-form）
# =====================================================================================

MATTING_TRIMAP_ERODE_FRAC = 0.012   # 0.025→0.012：减小边缘软过渡带，让两侧不显"模糊"
MATTING_TRIMAP_DILATE_FRAC = 0.018  # 0.035→0.018
MATTING_BAND_PAD_PX = 24
MATTING_MIN_BAND_PX = 6


def refine_alpha_via_matting(
    original_bgra: np.ndarray,
    polygon_mask_uint8: np.ndarray,
    jaw_span_px: float,
) -> np.ndarray:
    """
    用 ``pymatting.estimate_alpha_cf`` 在 polygon_mask 边界带内解 closed-form alpha matting，
    输出 (h,w) float64 in [0,1]。
    优化：只在 trimap 中"未知"区域的 bbox + ``MATTING_BAND_PAD_PX`` 内做求解，
    剩下的位置直接用 erode/dilate 决定（fg=1, bg=0）。这把 CPU 耗时从全图 ~3s 降到典型 ~200ms。
    失败时回退到 GaussianBlur 羽化。
    """
    H, W = polygon_mask_uint8.shape[:2]
    erode_px = max(int(jaw_span_px * MATTING_TRIMAP_ERODE_FRAC), 2)
    dilate_px = max(int(jaw_span_px * MATTING_TRIMAP_DILATE_FRAC), 2)
    bin_mask = (polygon_mask_uint8 > 64).astype(np.uint8)
    fg = cv2.erode(bin_mask, np.ones((erode_px * 2 + 1, erode_px * 2 + 1), np.uint8))
    bg_inv = cv2.dilate(bin_mask, np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8))
    bg = (bg_inv == 0).astype(np.uint8)

    trimap = np.full((H, W), 0.5, dtype=np.float64)
    trimap[fg > 0] = 1.0
    trimap[bg > 0] = 0.0
    unknown = (trimap == 0.5)

    if not np.any(unknown):
        return polygon_mask_uint8.astype(np.float64) / 255.0

    # bbox of unknown band + padding
    ys, xs = np.where(unknown)
    pad = MATTING_BAND_PAD_PX
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(H, int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(W, int(xs.max()) + pad + 1)
    if y1 - y0 < MATTING_MIN_BAND_PX or x1 - x0 < MATTING_MIN_BAND_PX:
        return polygon_mask_uint8.astype(np.float64) / 255.0

    image_rgb = cv2.cvtColor(original_bgra[y0:y1, x0:x1, :3], cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    trimap_crop = trimap[y0:y1, x0:x1]

    try:
        import pymatting  # 仅当走 matting 路径才 import，便于无 pymatting 环境 fallback
        alpha_crop = pymatting.estimate_alpha_cf(image_rgb, trimap_crop)
    except Exception:
        # fallback：GaussianBlur 羽化
        sigma = max(jaw_span_px * NECK_FEATHER_FRAC, NECK_FEATHER_MIN_PX)
        sigma = min(sigma, NECK_FEATHER_MAX_PX)
        k = odd_kernel(int(round(sigma * 3.0)) + 1)
        return (cv2.GaussianBlur(polygon_mask_uint8.astype(np.float32), (k, k), sigma) / 255.0).astype(np.float64)

    alpha_full = np.where(fg > 0, 1.0, 0.0).astype(np.float64)
    alpha_full[bg > 0] = 0.0
    alpha_full[y0:y1, x0:x1] = np.clip(alpha_crop, 0.0, 1.0)
    return alpha_full


# =====================================================================================
# Phase E: 颜色迁移 — L mean-shift + a/b CDF match
# =====================================================================================


def _match_1d_cdf(src: np.ndarray, ref_sorted: np.ndarray) -> np.ndarray:
    """把 src 的 1D 分布按 rank 映射到 ref_sorted 上。ref 已排序。"""
    n_src = src.shape[0]
    n_ref = ref_sorted.shape[0]
    if n_src == 0 or n_ref == 0:
        return src.copy()
    src_sorted_idx = np.argsort(src)
    target_pos = np.round(np.arange(n_src) * (n_ref - 1) / max(n_src - 1, 1)).astype(np.int64)
    target_vals = ref_sorted[target_pos]
    out = np.empty_like(src)
    out[src_sorted_idx] = target_vals
    return out


def _apply_L_meanshift_ab_cdf_inplace(
    layer_bgra: np.ndarray,
    mask_bool: np.ndarray,
    ref_pixels_bgr: np.ndarray,
    strength: float = 1.0,
    alpha_map: Optional[np.ndarray] = None,
) -> None:
    """
    分通道处理 Lab 空间颜色迁移，支持 alpha 加权。
    - L 通道：mean shift（保留 detail std，亮度 noise 完整透过）
    - a, b 通道：1D CDF match（让色相分布完整迁移到 ref 分布形状）

    ``alpha_map`` 为 (h,w) float64 in [0,1] 时，每个像素的实际校正强度 =
    ``strength × alpha_map[i,j]``。这使边缘过渡带（alpha 0.05~0.3）也得到
    部分颜色校正，避免硬阈值导致的色差接缝。``alpha_map=None`` 时退化为
    统一 strength（兼容旧调用）。
    """
    if ref_pixels_bgr.shape[0] < 8 or not np.any(mask_bool):
        return
    h, w = layer_bgra.shape[:2]
    bgr = layer_bgra[:, :, :3]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    has_alpha = alpha_map is not None
    if has_alpha:
        alpha_f = np.clip(alpha_map.astype(np.float32), 0.0, 1.0)
        # 用 alpha > 0.02 确定参与计算的像素（避免除零和无效运算）
        work_mask = alpha_f > 0.02
    else:
        work_mask = mask_bool
        alpha_f = None

    if not np.any(work_mask):
        return

    src = lab[work_mask]
    if src.shape[0] < 8:
        return
    ref_lab = cv2.cvtColor(
        ref_pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB
    ).reshape(-1, 3).astype(np.float32)
    s = float(np.clip(strength, 0.0, 1.0))

    # L 通道：mean shift
    L_shift = (ref_lab[:, 0].mean() - src[:, 0].mean()) * s
    new_L = src[:, 0] + L_shift

    # a, b 通道：1D CDF match (rank-based 分位数映射)
    ref_a_sorted = np.sort(ref_lab[:, 1])
    ref_b_sorted = np.sort(ref_lab[:, 2])
    cdf_a = _match_1d_cdf(src[:, 1].astype(np.float64), ref_a_sorted.astype(np.float64))
    cdf_b = _match_1d_cdf(src[:, 2].astype(np.float64), ref_b_sorted.astype(np.float64))
    new_a = src[:, 1] * (1.0 - s) + cdf_a.astype(np.float32) * s
    new_b = src[:, 2] * (1.0 - s) + cdf_b.astype(np.float32) * s

    src_new = np.stack([new_L, new_a, new_b], axis=1)
    src_new = np.clip(src_new, 0.0, 255.0)

    if has_alpha:
        # per-pixel 混合：final = original × (1 - s×alpha_i) + corrected × (s×alpha_i)
        pixel_alpha = alpha_f[work_mask]  # (N,)
        blend = pixel_alpha[:, None].astype(np.float32)  # (N,1)
        blended = src * (1.0 - s * blend) + src_new * (s * blend)
        blended = np.clip(blended, 0.0, 255.0)
        lab[work_mask] = blended
    else:
        lab[work_mask] = src_new

    out_bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    layer_bgra[:, :, :3] = out_bgr


# =====================================================================================
# Phase A: 重构 — procedural_neck_init + realism_pipeline + add_fake_neck_v1
# =====================================================================================


class V1DebugInfo(NamedTuple):
    procedural_layer: np.ndarray
    quilted_neck: np.ndarray
    blended_layer: np.ndarray
    refined_alpha: np.ndarray
    final_layer: np.ndarray
    skin_marked: np.ndarray
    library_size: int
    patch_size: int
    overlap_px: int
    pose: Tuple[float, float, float]


def _build_polygon_and_mask(
    landmarks,
    h: int,
    w: int,
    overlap: float,
    neck_depth: float,
    flare: float,
    top_inset: float,
    yaw: float,
    roll: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """构造 jaw-guided 多边形（含姿态修正），返回 (poly, mask_uint8 AA-edged)。"""
    poly = build_jaw_guided_neck_polygon(
        landmarks, w, h, overlap, neck_depth, flare, top_inset=top_inset
    )
    if abs(yaw) > 1e-3:
        n_top = poly.shape[0] // 2
        top = poly[:n_top]
        bot = poly[n_top:]
        cx_top = float(np.mean(top[:, 0]))
        sign_dir = np.sign(top[:, 0] - cx_top)
        extra_factor = np.clip(1.0 - POSE_YAW_ASYM_INSET_GAIN * yaw * sign_dir, 0.7, 1.3)
        top[:, 0] = cx_top + (top[:, 0] - cx_top) * extra_factor
        sign_b = np.sign(bot[:, 0] - cx_top)
        extra_b = np.clip(1.0 - POSE_YAW_ASYM_INSET_GAIN * yaw * sign_b, 0.7, 1.3)
        bot[:, 0] = cx_top + (bot[:, 0] - cx_top) * extra_b
        poly = np.vstack([top, bot])
    if abs(roll) > 1e-3:
        chin_x_lm, chin_y_lm = landmark_xy(landmarks.landmark[LANDMARK_CHIN_BOTTOM], w, h)
        poly = rotate_xy(poly, chin_x_lm, chin_y_lm, roll)
        poly[:, 0] = np.clip(poly[:, 0], 0.0, float(w - 1))
        poly[:, 1] = np.clip(poly[:, 1], 0.0, float(h - 1))
    mask = fill_polygon_mask(h, w, poly, supersample=NECK_MASK_SUPERSAMPLE)
    return poly, mask


def procedural_neck_init(
    bgra: np.ndarray,
    landmarks,
    skin_bgr: np.ndarray,
    overlap: float,
    neck_depth: float,
    flare: float,
    top_inset: float,
    yaw: float,
    pitch: float,
    roll: float,
    jaw_span: float,
    flat_shading: bool = True,
    cylinder_strength: float = 0.30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Phase A 之"过程化先验"：调用 add_neck.py 的成熟模块产出多边形、肤色 albedo，
    可选地叠加圆柱明暗 / AO / SCM 解剖。**不**做 Reinhard / 不叠 grain / 不做 alpha 抑制。

    ``flat_shading=True``（默认）：脖子用均匀的肤色 + 轻微肤色渐变，不做圆柱阴影。
    适合 portrait + 简单背景，避免脖子两侧出现"S 形阴影线"显假。
    ``flat_shading=False``：调用 ``neck_cylinder_shade_map``，按 ``cylinder_strength``
    线性衰减 lit/shadow 强度（0=完全平，1=原 add_neck.py 默认强度）。

    返回 (procedural_layer_bgra, polygon_mask_uint8, polygon_xy)。
    """
    h, w = bgra.shape[:2]
    poly, mask = _build_polygon_and_mask(
        landmarks, h, w, overlap, neck_depth, flare, top_inset, yaw, roll
    )
    skin = np.array(skin_bgr, dtype=np.float64)
    span_x = float(np.max(poly[:, 0]) - np.min(poly[:, 0]))
    R_est = max(span_x * 0.5, 4.0)
    light = estimate_face_lighting_for_neck(bgra, landmarks, h, w, R_est)
    if flat_shading:
        # v1 自实现的轻度 cylinder shading（不带 SCM/ridge，避免显假）：
        # (a) 颌下 AO：到上沿折线距离指数衰减（imagev2 实测顶部比中段暗 ~10%）
        # (b) 横向 cylinder：中心比两侧亮 ~5%，模拟真实脖子的圆柱微立体感
        # (c) 主光方向偏移：让 lit 一侧偏亮（imagev2 实测主光从右偏，亮度差~15）
        n_up_pre = max(poly.shape[0] // 2, 2)
        upper_pre = poly[:n_up_pre].astype(np.float64)
        dt_top = distance_map_to_polyline(h, w, upper_pre)
        depth_for_ao = max(float(np.max(poly[:, 1]) - np.min(poly[:, 1])), 1.0)
        tau_ao = max(depth_for_ao * 0.22, 8.0)
        ao_w = np.exp(-dt_top / tau_ao)

        # 横向 cylinder：用 tanh 让中央亮、两侧渐暗
        xx = np.arange(w, dtype=np.float64)[None, :]
        x_axis = float(np.mean(poly[:, 0]))
        span_x = float(np.max(poly[:, 0]) - np.min(poly[:, 0]))
        R_cyl = max(span_x * 0.5, 4.0)
        radial = (xx - x_axis) / R_cyl  # 中央 0，两边 ±1
        radial = np.clip(radial, -1.4, 1.4)
        # 中央亮（绝对值小→1），两侧暗（绝对值大→0）；rad_curve ∈ [0, 1]
        rad_curve = 1.0 - np.tanh(np.abs(radial) * 1.4) ** 1.6
        # 主光偏右：右侧再加 ~3% 亮，左侧减 ~3%
        light_dir = np.tanh(radial * 0.9)  # 右 +1，左 -1（保持 ndarray）
        cyl_intensity = 0.06  # 中央比两侧亮 6%
        light_dir_intensity = 0.03  # 右侧再加 3%
        # 形成 lateral L: [0.97 - 3%, 0.97 + 3% + cyl_at_center]
        L_lat = 1.0 + cyl_intensity * (rad_curve - 0.5) + light_dir_intensity * light_dir

        # AO: 颌下指数衰减
        L_ao = 1.0 - 0.08 * ao_w

        # 合成
        L = L_lat * L_ao
        # mask 内归一化到 mean=1，避免整体变暗/变亮
        wm_bool_local = mask >= 1
        if np.any(wm_bool_local):
            mu_L = float(np.mean(L[wm_bool_local]))
            L = L / max(mu_L, 1e-6)
        L = np.clip(L, 0.85, 1.15)
        wm_f = wm_bool_local.astype(np.float64)
        L = L * wm_f + (1.0 - wm_f)
    else:
        cs = float(np.clip(cylinder_strength, 0.0, 1.0))
        L = neck_cylinder_shade_map(
            h, w, poly, mask,
            x_axis_shift=light.x_axis_shift,
            spec_x_shift=light.spec_x_shift,
            k_lit=light.k_lit * cs, k_shadow=light.k_shadow * cs,
            shine_k=light.shine_k * cs, ao_top=light.ao_top,
            light_sign=light.light_sign,
            roll=float(roll),
        )
    wm_bool = mask >= 1
    if flat_shading:
        # flat 模式：彻底关掉肤色横向/纵向渐变 G（它源自脸部估光，叠到脖子上会形成可见亮带）。
        # 脖子是均匀肤色 albedo + 极弱颌下 AO（已包含在 L 中）。
        bgr = skin * L[:, :, np.newaxis]
        bgr = np.broadcast_to(bgr, (h, w, 3)).copy() if bgr.shape[:2] == (1, 1) else bgr
        if bgr.ndim == 3 and bgr.shape[:2] != (h, w):
            bgr = np.tile(skin, (h, w, 1)) * L[:, :, np.newaxis]
        else:
            # skin 是 (3,)，L 是 (h,w)；通过广播得到 (h,w,3)
            bgr = np.tile(skin[None, None, :], (h, w, 1)) * L[:, :, np.newaxis]
    else:
        xx = np.arange(w, dtype=np.float64)[np.newaxis, :]
        yy = np.arange(h, dtype=np.float64)[:, np.newaxis]
        cxn = float(np.mean(poly[:, 0]))
        cyn = float(np.mean(poly[:, 1]))
        y_min = float(np.min(poly[:, 1]))
        y_max = float(np.max(poly[:, 1]))
        depth_n = max(y_max - y_min, 1.0)
        G = (
            1.0
            + light.skin_grad_gx * (xx - cxn) / max(R_est, 1.0)
            + light.skin_grad_gy * (cyn - yy) / max(depth_n, 1.0)
        )
        G = np.clip(G, 0.9, 1.12)
        if np.any(wm_bool):
            G = G / max(float(np.mean(G[wm_bool])), 1e-6)
        bgr = np.tile(skin[None, None, :], (h, w, 1)) * L[:, :, np.newaxis] * G[:, :, np.newaxis]

    layer = np.zeros((h, w, 4), dtype=np.float64)
    layer[:, :, :3] = bgr
    # 临时 alpha：直接用 mask（uint8 0~255），后面 Phase C 会重算
    layer[:, :, 3] = mask.astype(np.float64)
    layer_u8 = np.clip(np.round(layer), 0, 255).astype(np.uint8)
    return layer_u8, mask, poly


def realism_pipeline(
    bgra: np.ndarray,
    landmarks,
    procedural_layer: np.ndarray,
    polygon_mask_u8: np.ndarray,
    poly: np.ndarray,
    jaw_span: float,
    enable_quilt: bool = True,
    enable_matting: bool = True,
    enable_lap_blend: bool = True,
    enable_pitie: bool = True,
    enable_stubble: bool = True,
    pitie_strength: float = 0.6,
    laplacian_levels: int = LAPLACIAN_LEVELS_DEFAULT,
    rng_seed: int = QUILT_RNG_SEED,
    quilt_method: str = "tile",
    source_neck_pixels: Optional[np.ndarray] = None,
    source_warped_bgra: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, V1DebugInfo]:
    """
    Phase A 之"真实感主流程"：在 procedural 层之上跑 B/C/D/E。
    """
    h, w = bgra.shape[:2]
    mask_bool = polygon_mask_u8 >= 1

    # ---------- Phase B: 真实皮肤纹理迁移（默认走 face_detail_tile，可选 quilting）-----
    # face_detail_tile：取脸部单块连续肤色 ROI，BORDER_REFLECT 镜像平铺，无接缝。
    # image_quilt（旧）：min-cut seam 拼接，对低频皮肤会暴露接缝（不推荐，作 advanced）。
    quilted = np.zeros((h, w, 3), dtype=np.uint8)
    have_texture = False
    patch_size = int(np.clip(round(jaw_span * QUILT_PATCH_FRAC_OF_JAW), QUILT_PATCH_MIN_PX, QUILT_PATCH_MAX_PX))
    overlap = max(int(round(patch_size * QUILT_OVERLAP_FRAC)), 4)
    library: List[np.ndarray] = []
    library_size = 0
    detail = np.zeros((h, w, 3), dtype=np.float32)
    edge_dist = cv2.distanceTransform(
        mask_bool.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    edge_weight = np.clip(edge_dist / float(DETAIL_EDGE_FEATHER_PX), 0.0, 1.0)
    if enable_quilt:
        rng = np.random.default_rng(rng_seed)
        if quilt_method == "noise":
            # 1/f pink noise 频谱接近真实皮肤毛孔。
            # 关键：3 通道共享同一**灰度** noise（不是各自独立 RGB noise，否则产生彩虹色斑）。
            # 真实皮肤纹理本质是 luminance 变化（毛孔深浅 / 皮下血管），各 RGB 通道同步变。
            sigma = float(np.clip(PINK_NOISE_SIGMA, 1.0, 60.0))
            noise_gray = generate_pink_noise_2d(h, w, sigma, alpha=PINK_NOISE_ALPHA, seed=rng_seed)
            if PINK_NOISE_BLUR_SIGMA > 0:
                bk = odd_kernel(int(round(PINK_NOISE_BLUR_SIGMA * 3.0)) + 1)
                noise_gray = cv2.GaussianBlur(noise_gray, (bk, bk), PINK_NOISE_BLUR_SIGMA)
            # 3 通道共享，色相波动自适应：从脸部锚定肤色 Lab a 通道决定方向和幅度
            # a > 0 → 暖肤色（R 偏强），a < 0 → 冷肤色（B 偏强）
            noise_bgr_scale = _compute_noise_bgr_scale(bgra, landmarks, h, w)
            noise = np.stack([
                noise_gray * noise_bgr_scale[0],  # B
                noise_gray * noise_bgr_scale[1],  # G
                noise_gray * noise_bgr_scale[2],  # R
            ], axis=-1).astype(np.float32)
            detail = noise * edge_weight[..., None] * mask_bool[..., None].astype(np.float32)
            have_texture = True
        elif quilt_method == "tile":
            quilted = face_detail_tile_into_mask(
                bgra, landmarks, h, w, mask_bool, jaw_span, rng=rng,
            )
            have_texture = bool(np.any(quilted))
        elif quilt_method == "patchquilt":
            library = build_face_skin_patch_library(bgra, landmarks, patch_size, rng=rng)
            library_size = len(library)
            if library:
                quilted = image_quilt_in_mask(
                    h, w, mask_bool, library, patch_size, overlap, rng=rng
                )
                have_texture = True

    # ---------- Mean-Match Texture Transfer（仅 tile/patchquilt 路径需要）-------------
    # 1/f pink noise 路径已经天然 high-pass（频谱在低频较弱），不需要 mean-match。
    if have_texture and quilt_method in ("tile", "patchquilt"):
        if quilt_method == "tile":
            tile_size_eff = int(np.clip(round(jaw_span * DETAIL_TILE_SIZE_FRAC), DETAIL_TILE_MIN_PX, DETAIL_TILE_MAX_PX))
        else:
            tile_size_eff = patch_size
        sigma_match = max(tile_size_eff * QUILT_MEAN_MATCH_SIGMA_FRAC, 12.0)
        k_match = odd_kernel(int(round(sigma_match * 3.0)) + 1)
        mask_f = mask_bool.astype(np.float32)
        quilted_lf = _masked_gaussian_blur(quilted, mask_f, sigma_match, k_match)
        detail = quilted.astype(np.float32) - quilted_lf
        detail = np.clip(detail, -DETAIL_INTENSITY_CLIP, DETAIL_INTENSITY_CLIP)
        detail = detail * edge_weight[..., None] * mask_bool[..., None].astype(np.float32)

    # ---------- Phase D: 把 detail（高频纹理）叠加到 procedural ----------------------
    if enable_lap_blend and have_texture:
        blended_bgr = np.clip(
            procedural_layer[:, :, :3].astype(np.float32) + detail, 0, 255
        ).astype(np.uint8)
    elif have_texture and not enable_lap_blend:
        blended_bgr = quilted.copy()
    else:
        blended_bgr = procedural_layer[:, :, :3].copy()

    # ---------- Transplant 替换：用原图脖子像素 1:1 替换合成脖子 BGR -----------------
    # 保留 procedural 的"形状/光照框架"和 mask，仅在 mask 内**用原图真实像素覆盖** BGR。
    # 边缘做一像素级羽化避免 warp 残影。
    if source_warped_bgra is not None and source_warped_bgra.shape[:2] == (h, w):
        # 用 mask_bool 内部稍微 erode 一圈避免 warp 边界混色
        eroded_mask = cv2.erode(
            mask_bool.astype(np.uint8) * 255,
            np.ones((3, 3), np.uint8),
            iterations=1,
        )
        em = eroded_mask > 0
        # 中央用原图，mask 边缘 1px 平滑过渡到 procedural detail（避免硬接缝）
        blended_bgr_arr = blended_bgr.astype(np.float32)
        warped_bgr = source_warped_bgra[:, :, :3].astype(np.float32)
        blended_bgr = np.where(em[..., None], warped_bgr, blended_bgr_arr)
        blended_bgr = np.clip(blended_bgr, 0, 255).astype(np.uint8)

    blended_bgra = np.dstack([blended_bgr, polygon_mask_u8])

    # ---------- Phase C 先做 alpha 求解（把范围扩到 feather 全境）---------------------
    if enable_matting:
        refined_alpha = refine_alpha_via_matting(bgra, polygon_mask_u8, jaw_span)
    else:
        sigma = float(np.clip(jaw_span * NECK_FEATHER_FRAC, NECK_FEATHER_MIN_PX, NECK_FEATHER_MAX_PX))
        k = odd_kernel(int(round(sigma * 3.0)) + 1)
        refined_alpha = (cv2.GaussianBlur(polygon_mask_u8.astype(np.float32), (k, k), sigma) / 255.0).astype(np.float64)

    # ---------- Phase E: 颜色迁移 ----------------------------------------------------
    # 优先级：
    #   0. 若 transplant 模式（source_warped_bgra 已成功 warp 过来）→ BGR 已是原图真实
    #      色，仅需要轻度（strength=0.30）Reinhard 适应 head 脸色光照差异，避免脖子
    #      偏离 head 整体色调。
    #   1. 若 source_neck_pixels 非空（提供原图但 transplant 没启用 / 几何对齐失败）→
    #      严格按原图脖子色作 ref pool（不混合 chin/oval）
    #   2. 否则走「下巴过渡区 + 脸部 oval 最亮 40%」混合（chin 仅 15% 权重，避免拉暗）
    ref_pixels = np.empty((0, 3), dtype=np.uint8)
    ref_source = "none"
    transplant_active = source_warped_bgra is not None and source_warped_bgra.shape[:2] == (h, w)
    if enable_pitie:
        if transplant_active:
            # transplant 模式：仅用 head 自身脸色作 ref，做轻度光照对齐
            ref_pixels = gather_face_oval_skin_pixels(bgra, landmarks, h, w)
            if ref_pixels.shape[0] < 200:
                ref_pixels = _gather_neck_ref_pixels(bgra, landmarks, h, w)
            ref_source = "transplant_light_match"
        elif source_neck_pixels is not None and source_neck_pixels.shape[0] >= SOURCE_NECK_MIN_PIXELS:
            # 路径 1：原图有脖子但没 transplant，严格用原图脖子色
            ref_pixels = source_neck_pixels
            ref_source = "source_image_neck"
        else:
            # 路径 2：原图没提供 / 没检测到脖子 → 走 chin strip + oval 混合
            # 修复 1: chin strip 权重从 50% 降到 15%（之前实测让 ref 偏暗 17 单位 R）
            chin_pixels = gather_chin_strip_skin_pixels(bgra, landmarks, h, w, jaw_span)
            oval_pixels = gather_face_oval_skin_pixels(bgra, landmarks, h, w)
            if chin_pixels.shape[0] >= 200 and oval_pixels.shape[0] >= 200:
                # chin 占 CHIN_STRIP_REF_WEIGHT (15%)，oval 占 (85%)
                n_oval = oval_pixels.shape[0]
                target_oval = n_oval
                target_chin = max(int(round(n_oval * CHIN_STRIP_REF_WEIGHT / (1.0 - CHIN_STRIP_REF_WEIGHT))), 200)
                target_chin = min(target_chin, chin_pixels.shape[0])
                rng_ref = np.random.default_rng(42)
                if chin_pixels.shape[0] > target_chin:
                    idx = rng_ref.choice(chin_pixels.shape[0], target_chin, replace=False)
                    chin_pixels = chin_pixels[idx]
                ref_pixels = np.vstack([chin_pixels, oval_pixels])
                ref_source = "chin15_oval85"
            elif oval_pixels.shape[0] >= 8:
                ref_pixels = oval_pixels
                ref_source = "oval_only"
            else:
                ref_pixels = _gather_neck_ref_pixels(bgra, landmarks, h, w)
                ref_source = "5_landmark_fallback"

        if ref_pixels.shape[0] >= 8:
            pitie_mask = polygon_mask_u8 >= 1
            # transplant 模式用更低强度（仅适应光照），其他路径用原 strength
            effective_strength = (
                TRANSPLANT_TONE_MATCH_STRENGTH if transplant_active else pitie_strength
            )
            _apply_L_meanshift_ab_cdf_inplace(
                blended_bgra, pitie_mask, ref_pixels,
                strength=effective_strength, alpha_map=refined_alpha,
            )

    # ---------- 方案 5: 胡茬检测 + 颌下暗调叠加 -------------------------------------
    # 检测脸是否有胡茬（下巴上 V < 颊 V 多少），若 strength>0 则在脖子顶端区域叠加
    # 衰减暗调（带微弱噪声让阴影不均匀），模拟胡茬向脖子的投影。
    if enable_stubble:
        stubble_strength = detect_stubble_strength(bgra, landmarks, h, w)
        if stubble_strength > 0.01:
            apply_stubble_shadow_inplace(
                blended_bgra, polygon_mask_u8, poly, jaw_span,
                strength=stubble_strength, rng_seed=rng_seed,
            )

    # 应用 alpha 抑制（仅在脸部不透明区域 + 距上沿近的 pixels 抑制脖子 alpha）
    n_up = max(poly.shape[0] // 2, 2)
    upper = poly[:n_up].astype(np.float64)
    dt_upper = distance_map_to_polyline(h, w, upper)
    decay = float(max(jaw_span * NECK_SUPPRESS_DECAY_FRAC, NECK_SUPPRESS_DECAY_MIN_PX))
    boundary_envelope = np.exp(-dt_upper / decay)
    oa = bgra[:, :, 3].astype(np.float64) / 255.0
    suppress = np.power(oa, NECK_SUPPRESS_ALPHA_GAMMA) * boundary_envelope
    final_alpha = np.clip(refined_alpha * (1.0 - suppress), 0.0, 1.0)

    # ---------- 底端 vertical fade-out（衣领过渡）-----------------------------------
    # imagev2 中脖子在 chin+50% jaw_span 处就过渡到衣领（白底图里直接是白）。
    # 让多边形下 50% 高度的 alpha 渐隐到 0，模拟「脖子下端→衣领→背景」的自然过渡。
    y_min_poly = float(np.min(poly[:, 1]))
    y_max_poly = float(np.max(poly[:, 1]))
    poly_height = max(y_max_poly - y_min_poly, 1.0)
    yy = np.arange(h, dtype=np.float64)[:, None]
    yn_poly = np.clip((yy - y_min_poly) / poly_height, 0.0, 1.0)
    fade_start = 0.50  # 0.50 → 1.0 段做 fade（之前 0.65，现在更早过渡）
    fade = np.where(
        yn_poly < fade_start,
        1.0,
        np.clip(1.0 - (yn_poly - fade_start) / (1.0 - fade_start), 0.0, 1.0) ** 1.2,
    )
    final_alpha = final_alpha * fade

    blended_bgra[:, :, 3] = np.clip(np.round(final_alpha * 255.0), 0, 255).astype(np.uint8)

    info = V1DebugInfo(
        procedural_layer=procedural_layer,
        quilted_neck=quilted,
        blended_layer=blended_bgra.copy(),
        refined_alpha=refined_alpha,
        final_layer=blended_bgra,
        skin_marked=np.zeros_like(bgra),
        library_size=len(library),
        patch_size=patch_size,
        overlap_px=overlap,
        pose=(0.0, 0.0, 0.0),
    )
    return blended_bgra, info


def add_fake_neck_v1(
    bgra: np.ndarray,
    *,
    neck_top_inset: float = 0.78,             # 0.86→0.78：脖子可见侧明显窄于下颌
    neck_slim_scale: float = NECK_SLIM_SCALE_DEFAULT,
    neck_bottom_flare: float = 1.05,          # 1.18→1.05：减少底部外扩，配合 fade-out
    neck_width_scale: float = 1.08,
    chin_overlap_px: Optional[float] = None,
    skin_v_scale: float = 1.0,                # 1.0：直接用实测下颌色，让 Reinhard 100% 拉到 ref
    skin_h_shift: float = 0.0,                # 取消向暖偏，让 Reinhard/Pitié 决定方向
    skin_s_scale: float = 1.0,                # 取消加饱和
    pose_correction: bool = True,
    auto_scale_by_jaw: bool = True,
    use_neck_anchor_skin: bool = True,        # True：用下颌锚定肤色采样（v1 默认）
    enable_quilt: bool = True,
    enable_matting: bool = True,
    enable_lap_blend: bool = True,
    enable_pitie: bool = True,
    enable_stubble: bool = True,              # 方案 5：检测胡茬 + 颌下暗调
    pitie_strength: float = 1.0,              # 1.0：完全对齐脸部下半部分均值（实测匹配 imagev2）
    quilt_method: str = "noise",              # "noise"(默认)=1/f pink noise; "tile"=镜像平铺; "patchquilt"=旧 quilting
    flat_shading: bool = True,                # True=不画圆柱 lit/shadow, 适合 portrait/无衣领
    cylinder_strength: float = 0.30,          # flat_shading=False 时圆柱明暗强度系数
    neck_depth_frac: float = 1.4,             # neck_depth = jaw_span * neck_depth_frac（1.85→1.4 更短）
    source_image_path: Optional[str] = None,  # 原图（带真实脖子）路径；非空且检测到脖子 → 严格按原图脖子色
    laplacian_levels: int = LAPLACIAN_LEVELS_DEFAULT,
    rng_seed: int = QUILT_RNG_SEED,
    face_mesh: Optional[object] = None,
    return_debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    重构版 add_fake_neck（Phase A-E）。返回 (合成 BGRA, 肤色采样标注图)；
    ``return_debug=True`` 时返回 (合成图, V1DebugInfo)。
    """
    h, w = bgra.shape[:2]
    rgb = bgra_to_rgb(bgra)

    if face_mesh is not None:
        landmarks = _detect_first_face(face_mesh, rgb)
        if landmarks is None:
            landmarks = detect_face_with_retry(rgb)
    else:
        landmarks = detect_face_with_retry(rgb)
    if landmarks is None:
        raise RuntimeError("未检测到人脸；尝试更大尺寸或更清晰的输入图。")

    lm = landmarks.landmark
    jaw_l_xy = landmark_xy(lm[LANDMARK_LEFT_JAW_ON_OVAL], w, h)
    jaw_r_xy = landmark_xy(lm[LANDMARK_RIGHT_JAW_ON_OVAL], w, h)
    jaw_span = max(float(np.hypot(jaw_l_xy[0] - jaw_r_xy[0], jaw_l_xy[1] - jaw_r_xy[1])), 12.0)

    if chin_overlap_px is None:
        if auto_scale_by_jaw:
            overlap = float(np.clip(jaw_span * JAW_SPAN_OVERLAP_FRAC, JAW_SPAN_OVERLAP_MIN_PX, JAW_SPAN_OVERLAP_MAX_PX))
        else:
            overlap = 17.0
    else:
        overlap = float(np.clip(chin_overlap_px, JAW_SPAN_OVERLAP_MIN_PX, JAW_SPAN_OVERLAP_MAX_PX))
    neck_depth = max(jaw_span * float(neck_depth_frac), JAW_SPAN_DEPTH_MIN_PX) if auto_scale_by_jaw else 80.0
    slim = float(np.clip(neck_slim_scale, 0.72, 1.0))
    flare = max(1.02, min(neck_bottom_flare * (neck_width_scale / 1.08), 1.45))
    flare_slim = 1.0 + (flare - 1.0) * slim
    neck_depth *= slim

    chin_x_px, chin_y_px = landmark_xy(lm[LANDMARK_CHIN_BOTTOM], w, h)
    existing_extent = estimate_existing_neck_extent_px(bgra, chin_x_px, chin_y_px, jaw_span)
    if existing_extent > jaw_span * COLLAR_DETECT_MIN_FRAC:
        neck_depth = max(COLLAR_DEPTH_SHRINK_MIN_PX, neck_depth - existing_extent * 0.7)
        overlap = float(np.clip(overlap * 0.7, JAW_SPAN_OVERLAP_MIN_PX, JAW_SPAN_OVERLAP_MAX_PX))

    if pose_correction:
        yaw, pitch, roll = estimate_head_pose(landmarks, w, h)
        pitch_scale = float(np.clip(
            1.0 + POSE_PITCH_CHIN_OVERLAP_GAIN * pitch,
            POSE_PITCH_CHIN_OVERLAP_MIN, POSE_PITCH_CHIN_OVERLAP_MAX,
        ))
        overlap = float(np.clip(overlap * pitch_scale, JAW_SPAN_OVERLAP_MIN_PX, JAW_SPAN_OVERLAP_MAX_PX))
    else:
        yaw, pitch, roll = 0.0, 0.0, 0.0

    if use_neck_anchor_skin:
        skin_bgr = sample_neck_anchor_skin_color(
            bgra, landmarks, w, h,
            skin_v_scale=skin_v_scale,
            skin_h_shift=skin_h_shift,
            skin_s_scale=skin_s_scale,
        )
    else:
        skin_bgr = sample_skin_color_bgra(
            bgra, landmarks, w, h,
            skin_v_scale=skin_v_scale,
            skin_h_shift=skin_h_shift,
            skin_s_scale=skin_s_scale,
        )

    proc_layer, mask_u8, poly = procedural_neck_init(
        bgra, landmarks, skin_bgr,
        overlap, neck_depth, flare_slim, neck_top_inset,
        yaw, pitch, roll, jaw_span,
        flat_shading=flat_shading,
        cylinder_strength=cylinder_strength,
    )

    # 若提供了原图路径，尝试做 transplant（最大限度还原真实脖子色彩）
    source_warped = transplant_source_neck(bgra, landmarks, source_image_path)
    # 同时（作 fallback ref pool）提取像素池
    source_neck_pixels = gather_source_neck_skin_pixels(source_image_path)

    # transplant 成功时禁用胡茬阴影（原图脖子已包含真实胡茬阴影，再叠加会变双倍）
    effective_enable_stubble = enable_stubble and source_warped is None

    final_layer, info = realism_pipeline(
        bgra, landmarks, proc_layer, mask_u8, poly, jaw_span,
        enable_quilt=enable_quilt,
        enable_matting=enable_matting,
        enable_lap_blend=enable_lap_blend,
        enable_pitie=enable_pitie,
        enable_stubble=effective_enable_stubble,
        pitie_strength=pitie_strength,
        laplacian_levels=laplacian_levels,
        rng_seed=rng_seed,
        quilt_method=quilt_method,
        source_neck_pixels=source_neck_pixels,
        source_warped_bgra=source_warped,
    )
    info = info._replace(pose=(yaw, pitch, roll))

    composed = alpha_over(final_layer, bgra)
    skin_marked = render_skin_sample_marked_preview(bgra, landmarks)
    if return_debug:
        info = info._replace(skin_marked=skin_marked)
        return composed, info  # type: ignore
    return composed, skin_marked


# =====================================================================================
# CLI
# =====================================================================================


def default_output_path(input_path: str) -> str:
    dname, fname = os.path.split(input_path)
    stem, _ = os.path.splitext(fname)
    if stem.lower().endswith("_rgba"):
        stem = stem[:-5]
    return os.path.join(dname, f"{stem}_with_neck_v1.png") if dname else f"{stem}_with_neck_v1.png"


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "为 RGBA 抠图头像添加真实感脖子（Phase A-E：quilting + matting + Laplacian + Pitié）。"
            " 与 add_neck.py 并存；本程序使用 add_neck.py 的几何/光照基础但替换纹理/边缘/色调子模块。"
        )
    )
    parser.add_argument("input", help="输入 RGBA PNG 路径")
    parser.add_argument("-o", "--output", default=None, help="输出路径；默认 *_with_neck_v1.png")
    parser.add_argument("--neck-top-inset", type=float, default=0.78)
    parser.add_argument("--neck-slim", type=float, default=NECK_SLIM_SCALE_DEFAULT)
    parser.add_argument("--neck-bottom-flare", type=float, default=1.05)
    parser.add_argument("--skin-v-scale", type=float, default=0.92)
    parser.add_argument("--skin-h-shift", type=float, default=0.0)
    parser.add_argument("--skin-s-scale", type=float, default=1.0)
    parser.add_argument("--pitie-strength", type=float, default=0.85)
    parser.add_argument("--laplacian-levels", type=int, default=LAPLACIAN_LEVELS_DEFAULT)
    parser.add_argument("--seed", type=int, default=QUILT_RNG_SEED)
    parser.add_argument("--no-quilt", dest="enable_quilt", action="store_false")
    parser.add_argument("--no-matting", dest="enable_matting", action="store_false")
    parser.add_argument("--no-lap-blend", dest="enable_lap_blend", action="store_false")
    parser.add_argument("--no-pitie", dest="enable_pitie", action="store_false")
    parser.add_argument("--no-pose-correction", dest="pose_correction", action="store_false")
    parser.add_argument("--no-auto-scale", dest="auto_scale_by_jaw", action="store_false")
    parser.add_argument(
        "--no-stubble", dest="enable_stubble", action="store_false",
        help="禁用胡茬检测 + 颌下暗调（默认开；女性头像无影响因为检测不出胡茬）",
    )
    parser.set_defaults(enable_stubble=True)
    parser.add_argument(
        "--source-image", default=None,
        help="原图路径（同一人的完整人像，带真实脖子）；提供后**严格按原图脖子肤色**渲染；"
        "为空 / 检测不到脖子时走默认 chin+oval 混合 ref 路径",
    )
    parser.add_argument(
        "--quilt-method", choices=["noise", "tile", "patchquilt"], default="noise",
        help="纹理方法：noise（默认，1/f pink noise，无脸结构污染）/ tile（镜像平铺）/ patchquilt（旧）",
    )
    parser.add_argument(
        "--cylinder-shading", dest="flat_shading", action="store_false",
        help="启用圆柱 lit/shadow 立体感（默认关；portrait/简单背景下圆柱阴影会显假）",
    )
    parser.set_defaults(flat_shading=True)
    parser.add_argument(
        "--cylinder-strength", type=float, default=0.30,
        help="圆柱明暗强度（仅当 --cylinder-shading 时生效）；0=平，1=add_neck.py 默认强度",
    )
    parser.set_defaults(
        enable_quilt=True, enable_matting=True, enable_lap_blend=True,
        enable_pitie=True, pose_correction=True, auto_scale_by_jaw=True,
    )
    args = parser.parse_args(argv)

    inp = os.path.abspath(args.input)
    out = os.path.abspath(args.output) if args.output else os.path.abspath(default_output_path(inp))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    bgra = load_rgba(inp)
    try:
        composed, skin_marked = add_fake_neck_v1(
            bgra,
            neck_top_inset=args.neck_top_inset,
            neck_slim_scale=args.neck_slim,
            neck_bottom_flare=args.neck_bottom_flare,
            skin_v_scale=args.skin_v_scale,
            skin_h_shift=args.skin_h_shift,
            skin_s_scale=args.skin_s_scale,
            pitie_strength=args.pitie_strength,
            laplacian_levels=args.laplacian_levels,
            rng_seed=args.seed,
            enable_quilt=args.enable_quilt,
            enable_matting=args.enable_matting,
            enable_lap_blend=args.enable_lap_blend,
            enable_pitie=args.enable_pitie,
            pose_correction=args.pose_correction,
            auto_scale_by_jaw=args.auto_scale_by_jaw,
            quilt_method=args.quilt_method,
            flat_shading=args.flat_shading,
            cylinder_strength=args.cylinder_strength,
            enable_stubble=args.enable_stubble,
            source_image_path=args.source_image,
        )
    except Exception as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1

    if not imwrite_unicode(out, composed):
        print(f"[错误] 无法写入 {out}", file=sys.stderr)
        return 1
    print(f"已保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
