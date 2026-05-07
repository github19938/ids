# -*- coding: utf-8 -*-
"""
add_neckv1.py
=============
真实感重构版本（Phase A + B + C + D + E 全做完，独立于 add_neck.py 不影响其行为）。

设计要点：
  - **不删旧代码**：``add_neck.py`` 完整保留作为 fallback；本文件复用其几何/光照/姿态/估光等
    底层助手，并在它生成的"过程化先验"上跑：
  - **Phase B（核心方案换）** 从用户脸部真实皮肤采 patch，用 image quilting + min-cut seam
    拼到脖子区域。脖子的纹理像素本质上**就是这张图脸上的肤色**，毛孔/淡疤/油光都是真的。
  - **Phase C（方案换）** 用 ``pymatting`` closed-form alpha matting 在 chin 弧线区域
    解 ``I = α·F + (1-α)·B``，得到结构感知的 sub-pixel alpha，替代旧的 GaussianBlur 羽化。
  - **Phase D（新增模块）** Laplacian 金字塔多频段融合：
      低频（光影/AO/SCM）从过程化层取，高频（真实毛孔/纹理）从 quilted 层取，
      合成一张「光影对、纹理真」的脖子。等价于 Poisson seam fix 但更稳定可控。
  - **Phase E（方案换）** Pitié N-D PDF transfer 替代 Reinhard mean shift，对 3D Lab
    联合分布做整形迁移而不是只动均值；带 sub-sampling + LUT bake 优化，CPU 实测 ~30 ms。
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


# Reinhard tone-match 的 ref pool 用「双颊主导 + 人中」，**不含下巴**（下巴常处于阴影会
# 拉低 ref mean）。让 ref mean 尽量接近脸主体亮度，从而脖子色匹配真正的脸色而非下颌阴影色。
NECK_REF_POOL_LANDMARKS: Tuple[int, ...] = (205, 425, 50, 280, 164)  # L 颊/R 颊/L 颊上/R 颊上/人中


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
PINK_NOISE_SIGMA = 8.0                    # 8 ≈ 真实皮肤 luminance 微变化幅度
PINK_NOISE_ALPHA = 0.5                    # 1/f^0.5（接近白噪声但稍偏低频，更接近真实皮肤）
PINK_NOISE_BLUR_SIGMA = 1.2               # blur 1.2px 让噪点显细腻（不显砂砾）

# --- Face Detail Tiling（旧路径，保留作可选）-----------------------------------------
DETAIL_TILE_SIZE_FRAC = 0.20              # 0.45→0.20：更小 tile 减少非纹理结构混入
DETAIL_TILE_MIN_PX = 24
DETAIL_TILE_MAX_PX = 80
DETAIL_TILE_LANDMARKS = (205, 425, 10)
DETAIL_INTENSITY_CLIP = 18.0
DETAIL_EDGE_FEATHER_PX = 8

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
# 各级（高频→低频）混入 quilted 的比例。新默认更"挑剔"地只取 ultra-high freq 细节，
# mid/low freq 几乎全用 procedural —— 这样 quilting 的 patch 接缝（典型 mid-freq
# 现象）几乎不可能透出来。Level 2 通常对应 patch_size 波长，必须压到极低。
LAPLACIAN_HIGH_FREQ_FROM_QUILT = (
    1.0, 0.55, 0.15, 0.0, 0.0,
)


def _build_gaussian_pyramid(img: np.ndarray, levels: int) -> List[np.ndarray]:
    pyr = [img.astype(np.float32)]
    cur = img.astype(np.float32)
    for _ in range(levels - 1):
        cur = cv2.pyrDown(cur)
        pyr.append(cur)
    return pyr


def _build_laplacian_pyramid(img: np.ndarray, levels: int) -> List[np.ndarray]:
    gpyr = _build_gaussian_pyramid(img, levels)
    lpyr = []
    for i in range(levels - 1):
        up = cv2.pyrUp(gpyr[i + 1], dstsize=(gpyr[i].shape[1], gpyr[i].shape[0]))
        lpyr.append(gpyr[i] - up)
    lpyr.append(gpyr[-1])  # 最低频留 Gaussian 顶端
    return lpyr


def _reconstruct_from_laplacian(lpyr: List[np.ndarray]) -> np.ndarray:
    cur = lpyr[-1]
    for lvl in range(len(lpyr) - 2, -1, -1):
        cur = cv2.pyrUp(cur, dstsize=(lpyr[lvl].shape[1], lpyr[lvl].shape[0]))
        cur = cur + lpyr[lvl]
    return cur


def laplacian_band_blend(
    procedural_bgr: np.ndarray,
    quilted_bgr: np.ndarray,
    mask_2d: np.ndarray,
    levels: int = LAPLACIAN_LEVELS_DEFAULT,
    high_freq_quilt_mix: Tuple[float, ...] = LAPLACIAN_HIGH_FREQ_FROM_QUILT,
) -> np.ndarray:
    """
    Laplacian 金字塔多频段融合。``mask_2d`` (h,w) bool 或 0~1 float，融合只在 mask 内进行。

    在每一频段：
        out_band = (1-α_band) * procedural_band + α_band * quilted_band
    其中 ``α_band`` 是 ``high_freq_quilt_mix`` 在该频段对应的值（高频→1，低频→0），
    超出 mask 的位置 α_band 强制为 0。
    """
    h, w = procedural_bgr.shape[:2]
    if quilted_bgr.shape[:2] != (h, w):
        raise ValueError("quilted shape mismatch")
    levels = max(2, int(levels))
    if len(high_freq_quilt_mix) != levels:
        # 平移 / 截断 / 填充
        mix = list(high_freq_quilt_mix)
        if len(mix) < levels:
            mix = mix + [mix[-1]] * (levels - len(mix))
        mix = mix[:levels]
    else:
        mix = list(high_freq_quilt_mix)

    if mask_2d.dtype == bool:
        mask_f = mask_2d.astype(np.float32)
    else:
        mask_f = mask_2d.astype(np.float32)
        if mask_f.max() > 1.5:
            mask_f = mask_f / 255.0
        mask_f = np.clip(mask_f, 0.0, 1.0)

    lap_p = _build_laplacian_pyramid(procedural_bgr.astype(np.float32), levels)
    lap_q = _build_laplacian_pyramid(quilted_bgr.astype(np.float32), levels)
    g_mask = _build_gaussian_pyramid(mask_f, levels)

    blended = []
    for lvl in range(levels):
        a = float(np.clip(mix[lvl], 0.0, 1.0))
        gm = g_mask[lvl][..., None]
        # 只在 mask 内做混入；mask 外完全用 procedural
        band = lap_p[lvl] * (1.0 - a * gm) + lap_q[lvl] * (a * gm)
        blended.append(band)

    out = _reconstruct_from_laplacian(blended)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


# =====================================================================================
# Phase C: Alpha Matting （pymatting closed-form）
# =====================================================================================

MATTING_TRIMAP_ERODE_FRAC = 0.025
MATTING_TRIMAP_DILATE_FRAC = 0.035
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
# Phase E: Pitié N-D PDF Color Transfer
# =====================================================================================

PITIE_DEFAULT_ITER = 20
# 注意：sub-sample + 1-NN 推广会在低相关度的相邻像素间产生 banding（实测 head.png 上
# 出现横向暗带）。把阈值放大到 60000 让大部分 portrait 全量跑（~37k px 用 ~150ms 全跑）；
# 大图 + > 60k px 时仍然 fallback 到 sub-sample 路径但用 k-NN 平均替代 1-NN。
PITIE_SUBSAMPLE_SRC_MAX = 60000
PITIE_KNN_K = 8


def _random_orthogonal_3d(rng: np.random.Generator) -> np.ndarray:
    """通过 QR 分解得到一个均匀分布的 3x3 正交矩阵。"""
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    # 修正符号歧义（QR 不保证 det = 1）
    sign = np.sign(np.diag(R))
    sign[sign == 0] = 1.0
    return Q * sign


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


def pitie_pdf_transfer(
    src_pixels: np.ndarray,
    ref_pixels: np.ndarray,
    n_iter: int = PITIE_DEFAULT_ITER,
    seed: int = 4242,
) -> np.ndarray:
    """
    Pitié & Kokaram 2007 N-D PDF transfer 主流程。
    输入 (N,3) src 与 (M,3) ref（一般是 Lab 空间），返回 (N,3) 转移后的 src。
    每轮：随机 3D 旋转 → 三通道 1D CDF 匹配 → 反旋转。
    """
    if src_pixels.size == 0 or ref_pixels.size == 0:
        return src_pixels.copy()
    rng = np.random.default_rng(seed)
    src = src_pixels.astype(np.float64).copy()
    ref = ref_pixels.astype(np.float64)
    for _ in range(int(n_iter)):
        R = _random_orthogonal_3d(rng)
        src_r = src @ R
        ref_r = ref @ R
        # 三通道独立 1D CDF 匹配
        for c in range(3):
            ref_sorted_c = np.sort(ref_r[:, c])
            src_r[:, c] = _match_1d_cdf(src_r[:, c], ref_sorted_c)
        src = src_r @ R.T
    return src


# 当 src 的 Lab std 低于此阈值，认为分布近似常数，Pitié 会过度展开导致 banding；
# 此时退化为 Reinhard mean shift（直接平移均值，不动 std）。
PITIE_LOW_VAR_STD_THRESHOLD = 6.0  # std < 6 视为低方差（每个 Lab 通道）


def _reinhard_mean_shift_inplace(
    layer_bgra: np.ndarray,
    mask_bool: np.ndarray,
    ref_pixels_bgr: np.ndarray,
    strength: float = 0.85,
) -> None:
    """Lab 空间均值偏移（沿用 add_neck.py 的 reinhard_lab_mean_shift_bgra_inplace 思路）。"""
    if ref_pixels_bgr.shape[0] < 8 or not np.any(mask_bool):
        return
    bgr = layer_bgra[:, :, :3]
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
    src_adj = np.clip(src + shift, 0.0, 255.0)
    lab[mask_bool] = src_adj
    out_bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    layer_bgra[:, :, :3] = out_bgr


def apply_pitie_to_layer_inplace(
    layer_bgra: np.ndarray,
    mask_bool: np.ndarray,
    ref_pixels_bgr: np.ndarray,
    strength: float = 0.65,
    n_iter: int = PITIE_DEFAULT_ITER,
    subsample_src_max: int = PITIE_SUBSAMPLE_SRC_MAX,
    seed: int = 4242,
    low_var_threshold: float = PITIE_LOW_VAR_STD_THRESHOLD,
) -> None:
    """
    把 layer 在 mask 内的 BGR 经过 Pitié 转移到 ref 分布，按 strength 与原值线性混合。
    Lab 空间执行。

    **关键安全网**：当 src 在 Lab 任一通道的 std 低于 ``low_var_threshold``，认为
    src 是 near-constant（如 procedural 均匀 albedo + 微弱 L 调制的脖子），此时
    Pitié 会把无意义的 micro-variation 放大成完整 ref 分布，产生可见 banding。
    自动退化为 ``_reinhard_mean_shift_inplace``（仅 mean shift），可避免 banding。

    **Sub-sampling 优化**：当源像素 > subsample_src_max 且非 near-constant 时，
    随机抽 subsample_src_max 子集跑 Pitié，再用 3D **k-NN 加权平均**推广。
    """
    if ref_pixels_bgr.shape[0] < 8 or not np.any(mask_bool):
        return
    bgr_full = layer_bgra[:, :, :3]
    lab_full = cv2.cvtColor(bgr_full, cv2.COLOR_BGR2LAB).astype(np.float64)
    src_lab = lab_full[mask_bool]  # (N, 3)
    n = src_lab.shape[0]
    ref_lab = (
        cv2.cvtColor(ref_pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)
        .reshape(-1, 3)
        .astype(np.float64)
    )

    # 低方差检测 → 退化为 Reinhard
    src_std = float(np.max(np.std(src_lab, axis=0)))
    if src_std < float(low_var_threshold):
        _reinhard_mean_shift_inplace(layer_bgra, mask_bool, ref_pixels_bgr, strength=strength)
        return

    rng = np.random.default_rng(seed)
    if n > subsample_src_max:
        sub_idx = rng.choice(n, subsample_src_max, replace=False)
        sub_src = src_lab[sub_idx]
        sub_dst = pitie_pdf_transfer(sub_src, ref_lab, n_iter=n_iter, seed=seed)
        new_src = _nearest_map_3d(src_lab, sub_src, sub_dst)
    else:
        new_src = pitie_pdf_transfer(src_lab, ref_lab, n_iter=n_iter, seed=seed)

    s = float(np.clip(strength, 0.0, 1.0))
    blended = src_lab * (1.0 - s) + new_src * s
    blended = np.clip(blended, 0.0, 255.0)

    lab_full[mask_bool] = blended
    out_bgr = cv2.cvtColor(lab_full.astype(np.uint8), cv2.COLOR_LAB2BGR)
    layer_bgra[:, :, :3] = out_bgr


def _nearest_map_3d(
    query: np.ndarray,
    anchors: np.ndarray,
    anchor_targets: np.ndarray,
    k: int = PITIE_KNN_K,
) -> np.ndarray:
    """
    用 anchors→anchor_targets 的散点映射，对 query 做 **k-NN 加权平均**查表（替代 1-NN）。
    1-NN 在低相关度区会让相邻像素跳到不同目标，产生 banding；k-NN 加权平均显著平滑。
    距离权重 = 1 / (d + ε)。分块计算避免 O(N*M) 内存。
    """
    n_q = query.shape[0]
    n_a = anchors.shape[0]
    k_eff = min(int(k), n_a)
    out = np.empty_like(query)
    chunk = 4096
    a_sq = np.sum(anchors * anchors, axis=1)  # (n_a,)
    for i in range(0, n_q, chunk):
        q = query[i : i + chunk]
        q_sq = np.sum(q * q, axis=1, keepdims=True)
        d = q_sq + a_sq[None, :] - 2.0 * (q @ anchors.T)
        d = np.maximum(d, 0.0)
        if k_eff <= 1:
            idx = np.argmin(d, axis=1)
            out[i : i + chunk] = anchor_targets[idx]
            continue
        # 取 k 个最近，加权平均
        idx_topk = np.argpartition(d, k_eff, axis=1)[:, :k_eff]  # (chunk, k)
        rows = np.arange(idx_topk.shape[0])[:, None]
        d_top = d[rows, idx_topk]
        w = 1.0 / (np.sqrt(d_top) + 1e-3)  # 距离权重
        w = w / np.sum(w, axis=1, keepdims=True)
        targets = anchor_targets[idx_topk]  # (chunk, k, 3)
        out[i : i + chunk] = np.sum(targets * w[:, :, None], axis=1)
    return out


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
        # 平 shading：跳过 cylinder（避免 SCM/ridge），自己写颌下 AO + 自顶到底渐变。
        # imagev2 实测脖子顶部（chin+15%）比中段（chin+50%）暗约 5%，所以 AO 最多 -8%。
        n_up_pre = max(poly.shape[0] // 2, 2)
        upper_pre = poly[:n_up_pre].astype(np.float64)
        dt_top = distance_map_to_polyline(h, w, upper_pre)
        depth_for_ao = max(float(np.max(poly[:, 1]) - np.min(poly[:, 1])), 1.0)
        tau_ao = max(depth_for_ao * 0.22, 8.0)
        ao_w = np.exp(-dt_top / tau_ao)
        L = 1.0 - 0.08 * ao_w  # 颌下最多压 8%（之前 3% 太弱）
        wm_f = (mask >= 1).astype(np.float64)
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
    pitie_strength: float = 0.6,
    laplacian_levels: int = LAPLACIAN_LEVELS_DEFAULT,
    rng_seed: int = QUILT_RNG_SEED,
    quilt_method: str = "tile",
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
            # 3 通道共享，但允许 R/G/B 各通道有微小色相波动（×0.95~1.05 让它不完全单调）
            noise = np.stack([
                noise_gray * 0.95,
                noise_gray * 1.00,
                noise_gray * 1.05,
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
    blended_bgra = np.dstack([blended_bgr, polygon_mask_u8])

    # ---------- Phase C 先做 alpha 求解（把范围扩到 feather 全境）---------------------
    if enable_matting:
        refined_alpha = refine_alpha_via_matting(bgra, polygon_mask_u8, jaw_span)
    else:
        sigma = float(np.clip(jaw_span * NECK_FEATHER_FRAC, NECK_FEATHER_MIN_PX, NECK_FEATHER_MAX_PX))
        k = odd_kernel(int(round(sigma * 3.0)) + 1)
        refined_alpha = (cv2.GaussianBlur(polygon_mask_u8.astype(np.float32), (k, k), sigma) / 255.0).astype(np.float64)

    # ---------- Phase E: 颜色迁移（默认 Reinhard mean shift）-----------------------------
    # 关键：Pitié 会把 detail noise 的 std 替换成 ref 的低 std，导致毛孔纹理被抹掉
    # （std 从 12 降到 3）。改用 Reinhard mean shift——只动 mean、保留 std，detail 完整透过。
    # ref pool 用颊主导（NECK_REF_POOL_LANDMARKS），让目标色对齐脸主体而非颌下阴影。
    if enable_pitie:
        ref_pixels = _gather_neck_ref_pixels(bgra, landmarks, h, w)
        pitie_mask = refined_alpha > 0.05
        _reinhard_mean_shift_inplace(
            blended_bgra, pitie_mask, ref_pixels, strength=pitie_strength,
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
    pitie_strength: float = 1.0,              # 1.0：完全对齐脸部下半部分均值（实测匹配 imagev2）
    quilt_method: str = "noise",              # "noise"(默认)=1/f pink noise; "tile"=镜像平铺; "patchquilt"=旧 quilting
    flat_shading: bool = True,                # True=不画圆柱 lit/shadow, 适合 portrait/无衣领
    cylinder_strength: float = 0.30,          # flat_shading=False 时圆柱明暗强度系数
    neck_depth_frac: float = 1.4,             # neck_depth = jaw_span * neck_depth_frac（1.85→1.4 更短）
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

    final_layer, info = realism_pipeline(
        bgra, landmarks, proc_layer, mask_u8, poly, jaw_span,
        enable_quilt=enable_quilt,
        enable_matting=enable_matting,
        enable_lap_blend=enable_lap_blend,
        enable_pitie=enable_pitie,
        pitie_strength=pitie_strength,
        laplacian_levels=laplacian_levels,
        rng_seed=rng_seed,
        quilt_method=quilt_method,
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
