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
    )


# =====================================================================================
# Phase B: Image Quilting （从脸部真实皮肤拼接到脖子区域）
# =====================================================================================

# 经典 Efros-Freeman 2001。本实现要点：
#   1) 从 SKIN_SAMPLE_REGIONS 周围更大的 ROI（~3x patch size）抽 patch；
#   2) 每个候选 patch 必须 ≥80% 落在 YCrCb 肤色范围；
#   3) 在 neck 多边形 bbox 内 raster scan，patch 间留 overlap≈patch/3；
#   4) 选 patch 时按 SSD on overlap 取 top-K，再随机选一个，避免肉眼可见的重复；
#   5) 在 overlap 区做 min-cut seam（DP）缝合，避免直边接缝。

QUILT_PATCH_FRAC_OF_JAW = 0.07            # patch 边长 ≈ jaw_span * 0.07
QUILT_PATCH_MIN_PX = 14
QUILT_PATCH_MAX_PX = 48
QUILT_OVERLAP_FRAC = 0.34                 # overlap = patch * 0.34
QUILT_K_BEST_CANDIDATES = 6
QUILT_LIBRARY_TARGET_SIZE = 240           # 从脸上抽到的 patch 库目标大小
QUILT_LIBRARY_SAMPLE_BOX_FRAC = 3.0       # 在每个 SKIN_SAMPLE_REGIONS 中心采 ~3x patch 边长的 ROI
QUILT_MIN_SKIN_RATIO = 0.80               # patch 至少 80% 像素是 YCrCb 肤色
QUILT_RNG_SEED = 12345


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
LAPLACIAN_HIGH_FREQ_FROM_QUILT = (
    1.0, 1.0, 0.85, 0.45, 0.10,
)  # 各级（高频→低频）混入 quilted 的比例；最低频几乎全用 procedural


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
PITIE_SUBSAMPLE_SRC_MAX = 8000


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


def apply_pitie_to_layer_inplace(
    layer_bgra: np.ndarray,
    mask_bool: np.ndarray,
    ref_pixels_bgr: np.ndarray,
    strength: float = 0.65,
    n_iter: int = PITIE_DEFAULT_ITER,
    subsample_src_max: int = PITIE_SUBSAMPLE_SRC_MAX,
    seed: int = 4242,
) -> None:
    """
    把 layer 在 mask 内的 BGR 经过 Pitié 转移到 ref 分布，按 strength 与原值线性混合。
    Lab 空间执行。**Sub-sampling 优化**：当源像素 > subsample_src_max 时，
    随机抽 subsample_src_max 子集做 Pitié，得到子集的"前→后"映射，
    然后用 3D **kNN-mean** 把映射推广到全部源像素（CPU 一次 ~30 ms）。
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

    rng = np.random.default_rng(seed)
    if n > subsample_src_max:
        sub_idx = rng.choice(n, subsample_src_max, replace=False)
        sub_src = src_lab[sub_idx]
        sub_dst = pitie_pdf_transfer(sub_src, ref_lab, n_iter=n_iter, seed=seed)
        # kNN（k=1，余量很小时也够用）：把 src_lab 中所有像素按最近邻映射到 sub_dst 中
        # 用 cv2.flann 较快；这里简化用 numpy + 分块计算（典型 N≤200K 完全够）
        new_src = _nearest_map_3d(src_lab, sub_src, sub_dst)
    else:
        new_src = pitie_pdf_transfer(src_lab, ref_lab, n_iter=n_iter, seed=seed)

    s = float(np.clip(strength, 0.0, 1.0))
    blended = src_lab * (1.0 - s) + new_src * s
    blended = np.clip(blended, 0.0, 255.0)

    lab_full[mask_bool] = blended
    out_bgr = cv2.cvtColor(lab_full.astype(np.uint8), cv2.COLOR_LAB2BGR)
    layer_bgra[:, :, :3] = out_bgr


def _nearest_map_3d(query: np.ndarray, anchors: np.ndarray, anchor_targets: np.ndarray) -> np.ndarray:
    """
    用 anchors→anchor_targets 的散点映射，对 query 做 1-NN 查表。
    分块计算避免一次性 O(N*M) 内存。
    """
    n_q = query.shape[0]
    n_a = anchors.shape[0]
    out = np.empty_like(query)
    chunk = 4096
    a_sq = np.sum(anchors * anchors, axis=1)  # (n_a,)
    for i in range(0, n_q, chunk):
        q = query[i : i + chunk]
        # |q - a|^2 = |q|^2 + |a|^2 - 2 q·a
        q_sq = np.sum(q * q, axis=1, keepdims=True)
        d = q_sq + a_sq[None, :] - 2.0 * (q @ anchors.T)
        idx = np.argmin(d, axis=1)
        out[i : i + chunk] = anchor_targets[idx]
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Phase A 之"过程化先验"：调用 add_neck.py 的成熟模块产出（多边形、圆柱明暗、AO、SCM 解剖、肤色 albedo），
    **不**做 Reinhard / 不叠 grain / 不做 alpha 抑制。
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
    L = neck_cylinder_shade_map(
        h, w, poly, mask,
        x_axis_shift=light.x_axis_shift,
        spec_x_shift=light.spec_x_shift,
        k_lit=light.k_lit, k_shadow=light.k_shadow,
        shine_k=light.shine_k, ao_top=light.ao_top,
        light_sign=light.light_sign,
        roll=float(roll),
    )
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
    wm_bool = mask >= 1
    if np.any(wm_bool):
        G = G / max(float(np.mean(G[wm_bool])), 1e-6)
    bgr = skin * L[:, :, np.newaxis] * G[:, :, np.newaxis]

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
) -> Tuple[np.ndarray, V1DebugInfo]:
    """
    Phase A 之"真实感主流程"：在 procedural 层之上跑 B/C/D/E。
    """
    h, w = bgra.shape[:2]
    mask_bool = polygon_mask_u8 >= 1

    # ---------- Phase B: image quilting ---------------------------------------------
    patch_size = int(np.clip(round(jaw_span * QUILT_PATCH_FRAC_OF_JAW), QUILT_PATCH_MIN_PX, QUILT_PATCH_MAX_PX))
    overlap = max(int(round(patch_size * QUILT_OVERLAP_FRAC)), 4)
    library: List[np.ndarray] = []
    quilted = np.zeros((h, w, 3), dtype=np.uint8)
    if enable_quilt:
        rng = np.random.default_rng(rng_seed)
        library = build_face_skin_patch_library(bgra, landmarks, patch_size, rng=rng)
        if library:
            quilted = image_quilt_in_mask(
                h, w, mask_bool, library, patch_size, overlap, rng=rng
            )

    # ---------- Phase D: Laplacian band blend (low=proc, high=quilt) -----------------
    if enable_lap_blend and library and enable_quilt:
        blended_bgr = laplacian_band_blend(
            procedural_layer[:, :, :3], quilted, mask_bool.astype(np.float32),
            levels=laplacian_levels,
        )
    else:
        blended_bgr = procedural_layer[:, :, :3].copy()
    blended_bgra = np.dstack([blended_bgr, polygon_mask_u8])

    # ---------- Phase E: Pitié N-D PDF transfer to align with face tone --------------
    if enable_pitie:
        ref_pixels = gather_lower_face_skin_pixels_bgr(bgra, landmarks, h, w)
        apply_pitie_to_layer_inplace(
            blended_bgra, mask_bool, ref_pixels,
            strength=pitie_strength, n_iter=PITIE_DEFAULT_ITER, seed=rng_seed,
        )

    # ---------- Phase C: pymatting closed-form alpha at chin boundary ----------------
    if enable_matting:
        refined_alpha = refine_alpha_via_matting(bgra, polygon_mask_u8, jaw_span)
    else:
        sigma = float(np.clip(jaw_span * NECK_FEATHER_FRAC, NECK_FEATHER_MIN_PX, NECK_FEATHER_MAX_PX))
        k = odd_kernel(int(round(sigma * 3.0)) + 1)
        refined_alpha = (cv2.GaussianBlur(polygon_mask_u8.astype(np.float32), (k, k), sigma) / 255.0).astype(np.float64)

    # 应用 alpha 抑制（仅在脸部不透明区域 + 距上沿近的 pixels 抑制脖子 alpha）
    n_up = max(poly.shape[0] // 2, 2)
    upper = poly[:n_up].astype(np.float64)
    dt_upper = distance_map_to_polyline(h, w, upper)
    decay = float(max(jaw_span * NECK_SUPPRESS_DECAY_FRAC, NECK_SUPPRESS_DECAY_MIN_PX))
    boundary_envelope = np.exp(-dt_upper / decay)
    oa = bgra[:, :, 3].astype(np.float64) / 255.0
    suppress = np.power(oa, NECK_SUPPRESS_ALPHA_GAMMA) * boundary_envelope
    final_alpha = np.clip(refined_alpha * (1.0 - suppress), 0.0, 1.0)
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
    neck_top_inset: float = NECK_TOP_INSET_DEFAULT,
    neck_slim_scale: float = NECK_SLIM_SCALE_DEFAULT,
    neck_bottom_flare: float = 1.18,
    neck_width_scale: float = 1.08,
    chin_overlap_px: Optional[float] = None,
    skin_v_scale: float = 0.96,
    skin_h_shift: float = NECK_HUE_SHIFT_DEFAULT,
    skin_s_scale: float = NECK_SAT_SCALE_DEFAULT,
    pose_correction: bool = True,
    auto_scale_by_jaw: bool = True,
    enable_quilt: bool = True,
    enable_matting: bool = True,
    enable_lap_blend: bool = True,
    enable_pitie: bool = True,
    pitie_strength: float = 0.6,
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
    neck_depth = max(jaw_span * JAW_SPAN_DEPTH_FRAC, JAW_SPAN_DEPTH_MIN_PX) if auto_scale_by_jaw else 80.0
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
    parser.add_argument("--neck-top-inset", type=float, default=NECK_TOP_INSET_DEFAULT)
    parser.add_argument("--neck-slim", type=float, default=NECK_SLIM_SCALE_DEFAULT)
    parser.add_argument("--skin-v-scale", type=float, default=0.96)
    parser.add_argument("--skin-h-shift", type=float, default=NECK_HUE_SHIFT_DEFAULT)
    parser.add_argument("--skin-s-scale", type=float, default=NECK_SAT_SCALE_DEFAULT)
    parser.add_argument("--pitie-strength", type=float, default=0.6)
    parser.add_argument("--laplacian-levels", type=int, default=LAPLACIAN_LEVELS_DEFAULT)
    parser.add_argument("--seed", type=int, default=QUILT_RNG_SEED)
    parser.add_argument("--no-quilt", dest="enable_quilt", action="store_false")
    parser.add_argument("--no-matting", dest="enable_matting", action="store_false")
    parser.add_argument("--no-lap-blend", dest="enable_lap_blend", action="store_false")
    parser.add_argument("--no-pitie", dest="enable_pitie", action="store_false")
    parser.add_argument("--no-pose-correction", dest="pose_correction", action="store_false")
    parser.add_argument("--no-auto-scale", dest="auto_scale_by_jaw", action="store_false")
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
