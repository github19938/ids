# -*- coding: utf-8 -*-
"""
add_clothes.py
==============
用 抠图头像 + 原图 + 衣服模板 合成蓝底证件照。

流程：
  1. 用 ``add_neckv1.add_fake_neck_v1`` 在头像下方延伸真实脖子（``--source-image``
     transplant 模式 1:1 复刻原图脖子色彩）。
  2. 衣服模板按头像 jaw_span 比例缩放，V 领顶点对齐到 chin 下方一段距离（让
     脖子从衣领里露出来）。
  3. 蓝底画布 + 合成层级：
        蓝底 ← 衣服模板 ← 带脖子的头像（最上层）

入口：
  - ``add_clothes(...)`` 主函数（可被 import 调用）
  - CLI: ``python add_clothes.py --head head.png --source src.png --clothes c.png -o out.png``
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

# 复用 add_neck.py / add_neckv1.py 的助手
try:
    from neck.add_neck import (  # type: ignore
        load_rgba, imread_unicode, imwrite_unicode, bgra_to_rgb,
        landmark_xy, alpha_over, detect_face_with_retry,
        LANDMARK_CHIN_BOTTOM, LANDMARK_LEFT_JAW_ON_OVAL, LANDMARK_RIGHT_JAW_ON_OVAL,
    )
    from neck.add_neckv1 import add_fake_neck_v1  # type: ignore
except ImportError:
    from add_neck import (  # type: ignore
        load_rgba, imread_unicode, imwrite_unicode, bgra_to_rgb,
        landmark_xy, alpha_over, detect_face_with_retry,
        LANDMARK_CHIN_BOTTOM, LANDMARK_LEFT_JAW_ON_OVAL, LANDMARK_RIGHT_JAW_ON_OVAL,
    )
    from add_neckv1 import add_fake_neck_v1  # type: ignore


# =====================================================================================
# 默认参数
# =====================================================================================

# 标准证件照背景色（中国大陆居民身份证 / 证件照常见蓝色 BGR）。
# 注意 OpenCV 是 BGR 顺序：标准证件照蓝 ≈ R=67, G=142, B=219
ID_PHOTO_BLUE_BGR: Tuple[int, int, int] = (219, 142, 67)
# 输出尺寸（W×H）。常见证件照比例 295×413 (1 英寸放大 4 倍 = 实际 295×413 px @ 300 DPI)。
DEFAULT_OUTPUT_SIZE_WH: Tuple[int, int] = (590, 826)
# 头像顶端到画布顶端的 margin 比例（头顶之上留多少空间）
HEAD_TOP_MARGIN_FRAC: float = 0.06
# 头部高度占画布高度的目标比例（头顶到下巴）
HEAD_HEIGHT_FRAC: float = 0.50
# 衣服 V 领顶点相对 chin 下方的位置 = chin_y + jaw_span × CLOTHES_VNECK_ANCHOR_FRAC
# 0.30 让 V 领紧贴 chin 下方，避开脖子 fade-out 区，无视觉缝隙
CLOTHES_VNECK_ANCHOR_FRAC: float = 0.30
# 衣服宽度相对 jaw_span 的比例
CLOTHES_WIDTH_TO_JAW_RATIO: float = 5.5
# 衣服顶部 V 领顶点的 x 坐标在衣服图中的位置（默认假定居中）
CLOTHES_TEMPLATE_VNECK_X_FRAC: float = 0.50
# V 领顶点的 y 坐标在衣服图中的位置（多数模板顶点在最上方一带，取顶部 5% 处的 alpha 中心）
CLOTHES_TEMPLATE_VNECK_Y_FRAC: float = 0.18
# 头部检测+延长脖子的临时 PNG（中间产物）
TEMP_HEAD_WITH_NECK_NAME: str = "_tmp_head_with_neck.png"


# =====================================================================================
# 主函数
# =====================================================================================


def _detect_chin_and_jaw_span(bgra: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """检测人脸，返回 (chin_x, chin_y, jaw_span) 像素坐标；失败返回 None。"""
    h, w = bgra.shape[:2]
    rgb = bgra_to_rgb(bgra)
    landmarks = detect_face_with_retry(rgb)
    if landmarks is None:
        return None
    lm = landmarks.landmark
    chin_x, chin_y = landmark_xy(lm[LANDMARK_CHIN_BOTTOM], w, h)
    jaw_l = landmark_xy(lm[LANDMARK_LEFT_JAW_ON_OVAL], w, h)
    jaw_r = landmark_xy(lm[LANDMARK_RIGHT_JAW_ON_OVAL], w, h)
    js = float(np.hypot(jaw_l[0] - jaw_r[0], jaw_l[1] - jaw_r[1]))
    return float(chin_x), float(chin_y), float(max(js, 12.0))


def _compute_head_bbox(bgra: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """根据 alpha 通道找头像主体的 bbox (x0, y0, x1, y1)。"""
    if bgra.shape[2] != 4:
        return None
    a = bgra[:, :, 3]
    ys, xs = np.where(a > 32)
    if len(ys) == 0 or len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _resize_with_alpha(im: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """带 alpha 通道的图像 resize；使用 INTER_AREA（缩小）/ INTER_CUBIC（放大）。"""
    src_h, src_w = im.shape[:2]
    flag = cv2.INTER_AREA if (new_w * new_h < src_w * src_h) else cv2.INTER_CUBIC
    return cv2.resize(im, (max(1, int(new_w)), max(1, int(new_h))), interpolation=flag)


def _alpha_paste(
    canvas: np.ndarray, layer: np.ndarray, x0: int, y0: int
) -> np.ndarray:
    """
    把 RGBA layer 用 alpha 合成贴到 RGBA canvas 的 (x0, y0) 起点。canvas 与 layer
    的 alpha 都参与 over 合成。返回新 canvas（不修改输入）。
    """
    H, W = canvas.shape[:2]
    lh, lw = layer.shape[:2]
    # 裁切到 canvas 范围
    cx0 = max(x0, 0)
    cy0 = max(y0, 0)
    cx1 = min(x0 + lw, W)
    cy1 = min(y0 + lh, H)
    if cx1 <= cx0 or cy1 <= cy0:
        return canvas
    lx0 = cx0 - x0
    ly0 = cy0 - y0
    lx1 = lx0 + (cx1 - cx0)
    ly1 = ly0 + (cy1 - cy0)
    sub_canvas = canvas[cy0:cy1, cx0:cx1].astype(np.float32)
    sub_layer = layer[ly0:ly1, lx0:lx1].astype(np.float32)
    a_layer = sub_layer[:, :, 3:4] / 255.0
    a_canvas = sub_canvas[:, :, 3:4] / 255.0
    out_a = a_layer + a_canvas * (1.0 - a_layer)
    out_a_safe = np.maximum(out_a, 1e-6)
    out_rgb = (
        sub_layer[:, :, :3] * a_layer
        + sub_canvas[:, :, :3] * a_canvas * (1.0 - a_layer)
    ) / out_a_safe
    new_canvas = canvas.copy()
    new_canvas[cy0:cy1, cx0:cx1, :3] = np.clip(np.round(out_rgb), 0, 255).astype(np.uint8)
    new_canvas[cy0:cy1, cx0:cx1, 3] = np.clip(np.round(out_a[:, :, 0] * 255.0), 0, 255).astype(np.uint8)
    return new_canvas


def _find_clothes_vneck_anchor(
    clothes_bgra: np.ndarray,
    fallback_x_frac: float = CLOTHES_TEMPLATE_VNECK_X_FRAC,
    fallback_y_frac: float = CLOTHES_TEMPLATE_VNECK_Y_FRAC,
) -> Tuple[int, int]:
    """
    自动找衣服模板里 V 领顶点的位置（脖子穿过去的入口）。
    策略：
      - 沿 alpha 通道扫描每一列：找出该列最上方 alpha>32 的 y 像素位置。
      - V 领顶点 = 中央 30% 列里 y 最大（最低位置）的列；这是 V 领最深处。
      - 失败 fallback 到 (fallback_x_frac × W, fallback_y_frac × H)。
    """
    h, w = clothes_bgra.shape[:2]
    if clothes_bgra.shape[2] != 4:
        return int(w * fallback_x_frac), int(h * fallback_y_frac)
    alpha = clothes_bgra[:, :, 3]
    # 中央 30% 列范围
    x_lo = int(w * 0.35)
    x_hi = int(w * 0.65)
    if x_hi <= x_lo + 1:
        return int(w * fallback_x_frac), int(h * fallback_y_frac)
    best_x = int(w * fallback_x_frac)
    best_y = -1
    for x in range(x_lo, x_hi):
        col = alpha[:, x]
        ys = np.where(col > 32)[0]
        if ys.size == 0:
            continue
        top_y = int(ys[0])  # 该列最上方不透明像素的 y
        if top_y > best_y:
            best_y = top_y
            best_x = x
    if best_y < 0:
        return int(w * fallback_x_frac), int(h * fallback_y_frac)
    return best_x, best_y


def add_clothes(
    head_image_path: str,
    clothes_template_path: str,
    output_path: str,
    source_image_path: Optional[str] = None,
    bg_color_bgr: Tuple[int, int, int] = ID_PHOTO_BLUE_BGR,
    output_size_wh: Tuple[int, int] = DEFAULT_OUTPUT_SIZE_WH,
    head_top_margin_frac: float = HEAD_TOP_MARGIN_FRAC,
    head_height_frac: float = HEAD_HEIGHT_FRAC,
    clothes_vneck_anchor_frac: float = CLOTHES_VNECK_ANCHOR_FRAC,
    clothes_width_to_jaw_ratio: float = CLOTHES_WIDTH_TO_JAW_RATIO,
    keep_temp_files: bool = False,
) -> int:
    """
    用抠图头像 + 原图 + 衣服模板合成蓝底证件照。

    :param head_image_path: 抠图好的头像路径（RGBA 透明背景）
    :param clothes_template_path: 衣服模板路径（RGBA 衣服图，背景透明）
    :param output_path: 输出蓝底证件照路径
    :param source_image_path: 同人完整原图路径（用于 add_fake_neck_v1 transplant 模式
        1:1 复刻真实脖子色彩）；为 None 走 add_fake_neck_v1 的 fallback 合成路径。
    :param bg_color_bgr: 背景色（默认证件照蓝）
    :param output_size_wh: 输出图尺寸 (宽, 高)，默认 590×826（证件照 5:7 比例）
    :param head_top_margin_frac: 头顶到画布顶端 margin / 画布高
    :param head_height_frac: 头部（头顶到下巴）高度 / 画布高
    :param clothes_vneck_anchor_frac: 衣服 V 领顶点相对 chin 的纵向偏移（jaw_span 倍数）
    :param clothes_width_to_jaw_ratio: 衣服整体宽度 / jaw_span
    :param keep_temp_files: 是否保留中间产物（带脖子的头像 PNG）
    :return: 0 成功, 非零失败
    """
    # ============ 1. 在头像下方加真实脖子 ============
    out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmp_head_path = os.path.join(out_dir, TEMP_HEAD_WITH_NECK_NAME)

    head_bgra = load_rgba(head_image_path)
    detected = _detect_chin_and_jaw_span(head_bgra)
    if detected is None:
        print(f"[错误] 头像中未检测到人脸: {head_image_path}", file=sys.stderr)
        return 1

    try:
        head_with_neck, _skin_marked = add_fake_neck_v1(
            head_bgra,
            source_image_path=source_image_path,
        )
    except Exception as e:
        print(f"[错误] 添加脖子失败: {e}", file=sys.stderr)
        return 1

    if keep_temp_files:
        imwrite_unicode(tmp_head_path, head_with_neck)

    # 重新检测 chin / jaw_span（脖子加完后位置不变，但用最新的 BGRA 做后续合成）
    detected2 = _detect_chin_and_jaw_span(head_with_neck)
    if detected2 is None:
        print("[错误] 加完脖子后人脸检测失败（这通常不会发生）", file=sys.stderr)
        return 1
    chin_x, chin_y, jaw_span = detected2

    # ============ 2. 计算头像在输出 canvas 中的目标位置和缩放 ============
    out_w, out_h = int(output_size_wh[0]), int(output_size_wh[1])
    head_bbox = _compute_head_bbox(head_with_neck)
    if head_bbox is None:
        print("[错误] 无法获取头像 bbox", file=sys.stderr)
        return 1
    bx0, by0, bx1, by1 = head_bbox
    src_head_h = by1 - by0  # 头像 alpha bbox 高度（含脖子）

    # 注意：bbox 含整个 alpha 范围（脸 + 脖子）。我们想"头部高度" = 头顶到 chin = chin_y - by0。
    head_top_to_chin = max(int(chin_y) - by0, 1)
    target_head_top_to_chin = int(out_h * head_height_frac)
    scale = target_head_top_to_chin / float(head_top_to_chin)

    # 缩放整个 head_with_neck（保持原比例）
    src_h, src_w = head_with_neck.shape[:2]
    scaled_w = max(1, int(round(src_w * scale)))
    scaled_h = max(1, int(round(src_h * scale)))
    scaled_head = _resize_with_alpha(head_with_neck, scaled_w, scaled_h)

    # 缩放后的几何
    scaled_chin_x = int(round(chin_x * scale))
    scaled_chin_y = int(round(chin_y * scale))
    scaled_jaw_span = jaw_span * scale
    scaled_bx0 = int(round(bx0 * scale))
    scaled_by0 = int(round(by0 * scale))

    # 头像在 canvas 中的放置位置：让 by0 (头顶) 距 canvas 顶端 = head_top_margin_frac × out_h
    paste_y = int(out_h * head_top_margin_frac) - scaled_by0
    # 水平居中：让 chin_x 落在 canvas 中央
    paste_x = (out_w // 2) - scaled_chin_x

    # ============ 3. 衣服模板缩放 + 对齐 V 领顶点到 chin 下方 ============
    clothes_bgra = imread_unicode(clothes_template_path, cv2.IMREAD_UNCHANGED)
    if clothes_bgra is None or clothes_bgra.ndim != 3:
        print(f"[错误] 衣服模板加载失败: {clothes_template_path}", file=sys.stderr)
        return 1
    if clothes_bgra.shape[2] == 3:
        clothes_bgra = cv2.cvtColor(clothes_bgra, cv2.COLOR_BGR2BGRA)
        clothes_bgra[:, :, 3] = 255

    # 自动找衣服模板的 V 领锚点
    cl_anchor_x, cl_anchor_y = _find_clothes_vneck_anchor(clothes_bgra)
    cl_h, cl_w = clothes_bgra.shape[:2]

    # 衣服目标宽度 = scaled_jaw_span × ratio
    target_clothes_w = scaled_jaw_span * clothes_width_to_jaw_ratio
    cl_scale = target_clothes_w / float(cl_w)
    cl_new_w = max(1, int(round(cl_w * cl_scale)))
    cl_new_h = max(1, int(round(cl_h * cl_scale)))
    scaled_clothes = _resize_with_alpha(clothes_bgra, cl_new_w, cl_new_h)
    scaled_anchor_x = int(round(cl_anchor_x * cl_scale))
    scaled_anchor_y = int(round(cl_anchor_y * cl_scale))

    # V 领顶点应对齐到（在 canvas 坐标系下）：
    #   ax = canvas 中央
    #   ay = canvas 中 chin_y 位置 + scaled_jaw_span × clothes_vneck_anchor_frac
    canvas_chin_y = paste_y + scaled_chin_y
    target_anchor_x_canvas = out_w // 2
    target_anchor_y_canvas = int(round(canvas_chin_y + scaled_jaw_span * clothes_vneck_anchor_frac))

    # 衣服图层在 canvas 中的左上角放置坐标
    clothes_paste_x = target_anchor_x_canvas - scaled_anchor_x
    clothes_paste_y = target_anchor_y_canvas - scaled_anchor_y

    # ============ 4. 合成 ============
    bg_b, bg_g, bg_r = bg_color_bgr
    canvas = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    canvas[:, :, 0] = bg_b
    canvas[:, :, 1] = bg_g
    canvas[:, :, 2] = bg_r
    canvas[:, :, 3] = 255

    # 顺序：bg ← clothes ← head（最上层）
    canvas = _alpha_paste(canvas, scaled_clothes, clothes_paste_x, clothes_paste_y)
    canvas = _alpha_paste(canvas, scaled_head, paste_x, paste_y)

    # ============ 5. 输出 ============
    if not imwrite_unicode(output_path, canvas):
        print(f"[错误] 写入输出失败: {output_path}", file=sys.stderr)
        return 1

    print(f"已保存: {output_path}")
    if keep_temp_files:
        print(f"中间产物（带脖子头像）: {tmp_head_path}")
    elif os.path.exists(tmp_head_path):
        try:
            os.remove(tmp_head_path)
        except OSError:
            pass
    return 0


# =====================================================================================
# CLI
# =====================================================================================


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="抠图头像 + 原图 + 衣服模板 → 蓝底证件照",
    )
    parser.add_argument("--head", required=True, help="抠图好的头像路径（RGBA）")
    parser.add_argument("--clothes", required=True, help="衣服模板路径（RGBA）")
    parser.add_argument(
        "--source", default=None,
        help="同人完整原图路径，用于真实脖子色彩 transplant；为空时走合成 fallback",
    )
    parser.add_argument("-o", "--output", required=True, help="输出证件照路径")
    parser.add_argument(
        "--bg-color", default="219,142,67",
        help="背景色 BGR（逗号分隔），默认 219,142,67（标准证件照蓝）",
    )
    parser.add_argument(
        "--size", default=f"{DEFAULT_OUTPUT_SIZE_WH[0]}x{DEFAULT_OUTPUT_SIZE_WH[1]}",
        help="输出尺寸 WxH，默认 590x826（5:7 证件照比例）",
    )
    parser.add_argument(
        "--head-top-margin", type=float, default=HEAD_TOP_MARGIN_FRAC,
        help="头顶到画布顶端 margin 占画布高的比例，默认 0.06",
    )
    parser.add_argument(
        "--head-height-frac", type=float, default=HEAD_HEIGHT_FRAC,
        help="头部（头顶到下巴）高度占画布高的比例，默认 0.50",
    )
    parser.add_argument(
        "--clothes-anchor-frac", type=float, default=CLOTHES_VNECK_ANCHOR_FRAC,
        help="衣服 V 领顶点对齐到 chin 下方的距离（jaw_span 倍数），默认 0.30",
    )
    parser.add_argument(
        "--clothes-width-ratio", type=float, default=CLOTHES_WIDTH_TO_JAW_RATIO,
        help="衣服宽度 / jaw_span 比例，默认 5.5",
    )
    parser.add_argument(
        "--keep-temp", action="store_true",
        help="保留中间产物（带脖子的头像 PNG）",
    )
    args = parser.parse_args(argv)

    bg_parts = [int(x.strip()) for x in args.bg_color.split(",")]
    if len(bg_parts) != 3:
        print("[错误] --bg-color 必须为 'B,G,R' 三个 0-255 整数", file=sys.stderr)
        return 1
    bg_color = tuple(int(np.clip(v, 0, 255)) for v in bg_parts)  # type: ignore

    size_parts = args.size.lower().split("x")
    if len(size_parts) != 2:
        print("[错误] --size 必须是 WxH（如 590x826）", file=sys.stderr)
        return 1
    out_size = (int(size_parts[0]), int(size_parts[1]))

    return add_clothes(
        head_image_path=os.path.abspath(args.head),
        clothes_template_path=os.path.abspath(args.clothes),
        output_path=os.path.abspath(args.output),
        source_image_path=os.path.abspath(args.source) if args.source else None,
        bg_color_bgr=bg_color,  # type: ignore
        output_size_wh=out_size,
        head_top_margin_frac=args.head_top_margin,
        head_height_frac=args.head_height_frac,
        clothes_vneck_anchor_frac=args.clothes_anchor_frac,
        clothes_width_to_jaw_ratio=args.clothes_width_ratio,
        keep_temp_files=args.keep_temp,
    )


if __name__ == "__main__":
    raise SystemExit(main())
