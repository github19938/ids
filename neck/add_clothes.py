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

# === 新版几何定位策略：以衣服模板为基准，不缩放衣服 ===
# canvas 宽度 = 衣服模板原始宽度（保留衣服细节，避免缩放损失）
# canvas 高度 = 头顶 margin + 头部 + 脖子 + 衣服（自动推导）

# (a) 头部 jaw_span 在 canvas 中的目标宽度 = clothes_w × 此比例
#     1 寸证件照里下颌宽通常占画面宽度的 17-22%；取 0.20。
HEAD_JAW_SPAN_TO_CANVAS_W_RATIO: float = 0.20
# (b) 头顶到 canvas 顶端 margin = clothes_w × 此比例
#     头顶留白通常 ~10-12% 画布宽度。
HEAD_TOP_MARGIN_TO_CANVAS_W_RATIO: float = 0.11
# (c) 脖子可见高度（chin 到 V 领顶点的经验距离）= scaled_jaw_span × 此比例
#     经验值：脖子在证件照里露出 0.70 × jaw_span 高度后被衣领遮住，比例自然。
#     注意：此值是 scaled_jaw_span 的倍数（与头部缩放联动），不是绝对像素。
NECK_VISIBLE_TO_JAW_SPAN_RATIO: float = 0.70

# 衣服顶部 V 领顶点的 fallback 位置（自动检测失败时用）
CLOTHES_TEMPLATE_VNECK_X_FRAC: float = 0.50
CLOTHES_TEMPLATE_VNECK_Y_FRAC: float = 0.18
# 头部检测+延长脖子的临时 PNG（中间产物）
TEMP_HEAD_WITH_NECK_NAME: str = "_tmp_head_with_neck.png"


# =====================================================================================
# 主函数
# =====================================================================================


# 调用 add_fake_neck_v1 前，确保 chin 下方有足够空间画脖子。
# add_neckv1 的脖子 polygon 高度 = jaw_span × JAW_SPAN_DEPTH_FRAC × slim ≈ jaw_span × 1.4 × 0.87 ≈ 1.22。
# 留 1.5x 余量。
NECK_DRAWING_BOTTOM_PAD_FACTOR: float = 1.5


def _ensure_neck_drawing_space(bgra: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    若 chin 下方在画布内剩余像素 < jaw_span × NECK_DRAWING_BOTTOM_PAD_FACTOR，
    在画布底部加透明 padding，让 add_fake_neck_v1 有足够空间绘制脖子。

    返回 (扩展后 bgra, 加了多少 px 的底部 pad)。
    """
    detected = _detect_chin_and_jaw_span(bgra)
    if detected is None:
        return bgra, 0
    chin_x, chin_y, jaw_span = detected
    h, w = bgra.shape[:2]
    needed = int(round(jaw_span * NECK_DRAWING_BOTTOM_PAD_FACTOR))
    available = h - int(chin_y)
    if available >= needed:
        return bgra, 0
    pad = needed - available
    out = np.zeros((h + pad, w, 4), dtype=np.uint8)
    out[:h, :, :] = bgra
    # 底部 padding 区 alpha=0（透明），不会影响合成
    return out, pad


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
    head_jaw_to_canvas_ratio: float = HEAD_JAW_SPAN_TO_CANVAS_W_RATIO,
    head_top_margin_to_canvas_ratio: float = HEAD_TOP_MARGIN_TO_CANVAS_W_RATIO,
    neck_visible_to_jaw_ratio: float = NECK_VISIBLE_TO_JAW_SPAN_RATIO,
    keep_temp_files: bool = False,
) -> int:
    """
    用抠图头像 + 原图 + 衣服模板合成蓝底证件照。

    **几何定位策略**（以衣服模板为基准 + 经验值像素定位）：
      - canvas 宽度 = 衣服模板原始宽度（衣服不缩放，保留细节）
      - canvas 高度 = 自动推导 = 头顶 margin + 头部高 + 脖子可见 + 衣服尚下部分
      - 头部缩放比例：让 scaled_jaw_span = canvas_w × ``head_jaw_to_canvas_ratio``
      - 脖子可见高度：scaled_jaw_span × ``neck_visible_to_jaw_ratio``
      - 衣服 V 领顶点位置 = chin_y_in_canvas + 脖子可见高度（让脖子从衣领里露出来）

    :param head_image_path: 抠图好的头像路径（RGBA 透明背景）
    :param clothes_template_path: 衣服模板路径（RGBA 衣服图，**不缩放**）
    :param output_path: 输出蓝底证件照路径
    :param source_image_path: 同人完整原图路径（用于 transplant 1:1 真实脖子色彩）
    :param bg_color_bgr: 背景色（默认证件照蓝 BGR=(219,142,67)）
    :param head_jaw_to_canvas_ratio: jaw_span / canvas_w，默认 0.20（脸宽约 1/5 画布）
    :param head_top_margin_to_canvas_ratio: 头顶 margin / canvas_w，默认 0.11
    :param neck_visible_to_jaw_ratio: 脖子可见高度 / scaled_jaw_span，默认 0.70
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

    # ★ 关键：image0 等抠图常常画布卡到 chin（chin 离底部仅 0-5px），
    # 加脖子前必须先扩展底部画布，否则 polygon 没空间画。
    head_bgra_expanded, pad_added = _ensure_neck_drawing_space(head_bgra)
    if pad_added > 0:
        print(f"[信息] 原画布 chin 下空间不足，已在底部加 {pad_added}px 透明 padding")

    try:
        head_with_neck, _skin_marked = add_fake_neck_v1(
            head_bgra_expanded,
            source_image_path=source_image_path,
            for_clothes_compositing=True,  # ★ 衣领合成模式：脖子塞满 V 领、不 fade、色调更主动适应
        )
    except Exception as e:
        print(f"[错误] 添加脖子失败: {e}", file=sys.stderr)
        return 1

    if keep_temp_files:
        imwrite_unicode(tmp_head_path, head_with_neck)

    # 重新检测脖子图上的 chin / jaw_span（用于位置对齐）
    detected2 = _detect_chin_and_jaw_span(head_with_neck)
    if detected2 is None:
        print("[错误] 加完脖子后人脸检测失败", file=sys.stderr)
        return 1
    chin_x, chin_y, jaw_span = detected2

    head_bbox = _compute_head_bbox(head_with_neck)
    if head_bbox is None:
        print("[错误] 无法获取头像 bbox", file=sys.stderr)
        return 1
    bx0, by0, bx1, by1 = head_bbox  # by0 = 头顶 y

    # ============ 2. 加载衣服模板（不缩放）+ 自动检测 V 领锚点 ============
    clothes_bgra = imread_unicode(clothes_template_path, cv2.IMREAD_UNCHANGED)
    if clothes_bgra is None or clothes_bgra.ndim != 3:
        print(f"[错误] 衣服模板加载失败: {clothes_template_path}", file=sys.stderr)
        return 1
    if clothes_bgra.shape[2] == 3:
        clothes_bgra = cv2.cvtColor(clothes_bgra, cv2.COLOR_BGR2BGRA)
        clothes_bgra[:, :, 3] = 255
    elif clothes_bgra.shape[2] != 4:
        print(f"[错误] 衣服模板格式不支持: shape={clothes_bgra.shape}", file=sys.stderr)
        return 1

    cl_h, cl_w = clothes_bgra.shape[:2]
    cl_anchor_x, cl_anchor_y = _find_clothes_vneck_anchor(clothes_bgra)

    # ============ 3. 几何定位（按经验值）============
    canvas_w = cl_w  # canvas 宽度 = 衣服宽度

    # 头部缩放：让 scaled_jaw_span = canvas_w × ratio
    target_jaw_span_in_canvas = canvas_w * head_jaw_to_canvas_ratio
    scale = target_jaw_span_in_canvas / float(jaw_span)

    src_h, src_w = head_with_neck.shape[:2]
    scaled_w = max(1, int(round(src_w * scale)))
    scaled_h = max(1, int(round(src_h * scale)))
    scaled_head = _resize_with_alpha(head_with_neck, scaled_w, scaled_h)
    scaled_chin_x = int(round(chin_x * scale))
    scaled_chin_y = int(round(chin_y * scale))
    scaled_by0 = int(round(by0 * scale))  # 头顶 y in scaled_head
    scaled_jaw_span = jaw_span * scale     # = target_jaw_span_in_canvas

    # 头顶距 canvas 顶端 margin（经验像素值）
    head_top_margin_px = int(round(canvas_w * head_top_margin_to_canvas_ratio))

    # head 在 canvas 中的位置
    head_paste_y = head_top_margin_px - scaled_by0
    head_paste_x = canvas_w // 2 - scaled_chin_x

    # chin 在 canvas 中的 y 位置
    chin_y_in_canvas = head_paste_y + scaled_chin_y

    # 脖子可见高度（chin 到 V 领顶点的距离，经验值）
    neck_visible_h = int(round(scaled_jaw_span * neck_visible_to_jaw_ratio))

    # V 领顶点在 canvas 中的目标位置：chin_y + 脖子可见高度
    target_vneck_y_in_canvas = chin_y_in_canvas + neck_visible_h
    target_vneck_x_in_canvas = canvas_w // 2  # 衣服 V 领居中

    # 衣服在 canvas 中的左上角放置位置
    clothes_paste_x = target_vneck_x_in_canvas - cl_anchor_x
    clothes_paste_y = target_vneck_y_in_canvas - cl_anchor_y

    # canvas 高度 = 衣服底部位置（衣服贴底）
    clothes_bottom_y = clothes_paste_y + cl_h
    canvas_h = max(clothes_bottom_y, target_vneck_y_in_canvas + 10)

    # ============ 4. 合成 ============
    bg_b, bg_g, bg_r = bg_color_bgr
    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    canvas[:, :, 0] = bg_b
    canvas[:, :, 1] = bg_g
    canvas[:, :, 2] = bg_r
    canvas[:, :, 3] = 255

    # 顺序：bg ← clothes ← head（最上层）
    canvas = _alpha_paste(canvas, clothes_bgra, clothes_paste_x, clothes_paste_y)
    canvas = _alpha_paste(canvas, scaled_head, head_paste_x, head_paste_y)

    # ============ 5. 输出 ============
    if not imwrite_unicode(output_path, canvas):
        print(f"[错误] 写入输出失败: {output_path}", file=sys.stderr)
        return 1

    print(f"已保存: {output_path}  (canvas {canvas_w}x{canvas_h})")
    print(
        f"  几何: scaled_jaw_span={scaled_jaw_span:.0f}px, "
        f"head_top_margin={head_top_margin_px}px, "
        f"neck_visible={neck_visible_h}px, "
        f"vneck_at=({target_vneck_x_in_canvas},{target_vneck_y_in_canvas})"
    )
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
        "--head-jaw-ratio", type=float, default=HEAD_JAW_SPAN_TO_CANVAS_W_RATIO,
        help="头部 jaw_span / canvas_w 比例，默认 0.20（脸宽约 1/5 画布）。"
        "调大 → 头部更大；调小 → 头部更小",
    )
    parser.add_argument(
        "--head-top-margin", type=float, default=HEAD_TOP_MARGIN_TO_CANVAS_W_RATIO,
        help="头顶到 canvas 顶 margin / canvas_w 比例，默认 0.11",
    )
    parser.add_argument(
        "--neck-visible-ratio", type=float, default=NECK_VISIBLE_TO_JAW_SPAN_RATIO,
        help="脖子可见高度 / scaled_jaw_span 比例，默认 0.70。"
        "调大 → 脖子更长（V 领更远）；调小 → V 领更近 chin",
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

    return add_clothes(
        head_image_path=os.path.abspath(args.head),
        clothes_template_path=os.path.abspath(args.clothes),
        output_path=os.path.abspath(args.output),
        source_image_path=os.path.abspath(args.source) if args.source else None,
        bg_color_bgr=bg_color,  # type: ignore
        head_jaw_to_canvas_ratio=args.head_jaw_ratio,
        head_top_margin_to_canvas_ratio=args.head_top_margin,
        neck_visible_to_jaw_ratio=args.neck_visible_ratio,
        keep_temp_files=args.keep_temp,
    )


if __name__ == "__main__":
    raise SystemExit(main())
