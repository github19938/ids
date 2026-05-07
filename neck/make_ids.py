# -*- coding: utf-8 -*-
"""
make_ids.py
===========
独立的证件照生成器（self-contained，不依赖项目其他文件）。

只做一件事：抠图头像 + 原图 + 衣服模板 + 底色 → 输出蓝底证件照。

合成顺序（与之前 add_clothes 不同的关键点）：
    底色画布 ← 带脖子的头像 ← 衣服模板（最上层）
即衣服盖在头像上方，V 领开口区（衣服 alpha=0）露出脸/脖子，
衣领布料区（衣服 alpha>0）遮住超出 V 领的脖子。

依赖：numpy, opencv-python, mediapipe（仅这三个）。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


class MakeIDPhoto:
    """证件照生成器（self-contained）。"""

    # ============ FaceMesh 关键点索引 ============
    LANDMARK_CHIN = 152          # 下巴尖
    LANDMARK_LEFT_JAW = 172      # 左下颌角
    LANDMARK_RIGHT_JAW = 397     # 右下颌角

    # ============ 脖子几何参数（经验值）============
    NECK_TOP_INSET = 0.95        # 脖子顶宽 / 下颌宽（接近 1 = 几乎=下颌宽）
    NECK_BOTTOM_FLARE = 1.0      # 脖子底宽 / 顶宽（不外扩）
    NECK_DEPTH_FRAC = 1.10       # 脖子高度 = jaw_span × 此值
    CHIN_OVERLAP_FRAC = 0.06     # 脖子上沿向上插入下巴的距离 / jaw_span
    NECK_DRAWING_PAD_FACTOR = 1.5  # 加脖子前底部预留 padding = jaw_span × 此值

    # ============ 衣服 / 画布几何参数（经验值）============
    HEAD_JAW_TO_CANVAS_RATIO = 0.20   # canvas 上 scaled_jaw_span / canvas_w
    HEAD_TOP_MARGIN_RATIO = 0.11      # 头顶 margin / canvas_w
    NECK_VISIBLE_TO_JAW_RATIO = 0.70  # 脖子可见高度 / scaled_jaw_span
    CLOTHES_VNECK_X_FRAC = 0.50       # 衣服 V 领 fallback x 位置 / clothes_w
    CLOTHES_VNECK_Y_FRAC = 0.18       # 衣服 V 领 fallback y 位置 / clothes_h

    # ============ 默认背景色（标准证件照蓝 #4382DB）============
    DEFAULT_BG_HEX = "#4382DB"

    # ============ MediaPipe 检测参数 ============
    FACE_MIN_DETECTION_CONFIDENCE = 0.4

    # ============ 多边形 mask 超采样倍数（消除两侧锯齿）============
    POLYGON_SUPERSAMPLE = 2

    def __init__(
        self,
        head_path: str,
        source_path: str,
        clothes_path: str,
        output_path: str,
        bg_hex: str = DEFAULT_BG_HEX,
    ):
        """
        :param head_path: 抠图头像路径（RGBA PNG，背景透明）
        :param source_path: 原图路径（同人完整照片，含真实脖子；RGB / RGBA 都可）
        :param clothes_path: 衣服模板路径（RGBA PNG，衣服外背景透明）
        :param output_path: 输出证件照路径
        :param bg_hex: 16 进制底色，支持 '#RGB' / '#RRGGBB' / 'RRGGBB' / '0xRRGGBB'
        """
        self.head_path = head_path
        self.source_path = source_path
        self.clothes_path = clothes_path
        self.output_path = output_path
        self.bg_bgr = self._hex_to_bgr(bg_hex)

    # =================================================================================
    # 主流程
    # =================================================================================

    def run(self) -> int:
        """执行完整 pipeline，返回 0 成功，非零失败。"""
        # 1. 加载所有图（自动转 RGBA）
        head = self._load_rgba(self.head_path)
        source = self._load_rgba(self.source_path)
        clothes = self._load_rgba(self.clothes_path)

        # 2. 检测原图人脸（用于 transplant 几何对齐）
        source_face = self._detect_face_landmarks(source)
        if source_face is None:
            raise RuntimeError(f"原图未检测到人脸: {self.source_path}")

        # 3. 头像加底部 padding，给脖子留绘制空间
        head_padded = self._ensure_neck_drawing_space(head)
        head_face = self._detect_face_landmarks(head_padded)
        if head_face is None:
            raise RuntimeError(f"头像未检测到人脸: {self.head_path}")

        # 4. 给头像加脖子：原图 affine warp + 多边形 mask 替换
        head_with_neck = self._add_neck(head_padded, head_face, source, source_face)

        # 5. 合成证件照（蓝底 + 头像 + 衣服，衣服最上层）
        canvas = self._compose(head_with_neck, clothes)

        # 6. 输出
        out_dir = os.path.dirname(os.path.abspath(self.output_path)) or "."
        os.makedirs(out_dir, exist_ok=True)
        self._imwrite(self.output_path, canvas)
        print(f"已保存: {self.output_path}  (canvas {canvas.shape[1]}x{canvas.shape[0]})")
        return 0

    # =================================================================================
    # 核心方法 1：加脖子（affine warp 原图脖子像素到头像坐标系）
    # =================================================================================

    def _add_neck(
        self,
        head_bgra: np.ndarray,
        head_face,
        source_bgra: np.ndarray,
        source_face,
    ) -> np.ndarray:
        """
        在头像下方加真实脖子：
          1. 头像和原图各自检测的下颌三点构造 affine 变换
          2. 把原图整体 warp 到头像坐标系
          3. 构造梯形脖子 polygon，在 polygon 内**用 warped 像素覆盖头像 BGR**
          4. polygon 内 alpha 强制满（与原 head alpha 取 max）

        polygon 边缘用 AA mask 0-255 灰度做软过渡，无可见接缝。
        """
        h, w = head_bgra.shape[:2]
        sH, sW = source_bgra.shape[:2]

        # 头像三点
        head_chin = self._landmark_xy(head_face.landmark[self.LANDMARK_CHIN], w, h)
        head_jl = self._landmark_xy(head_face.landmark[self.LANDMARK_LEFT_JAW], w, h)
        head_jr = self._landmark_xy(head_face.landmark[self.LANDMARK_RIGHT_JAW], w, h)
        # 原图三点
        src_chin = self._landmark_xy(source_face.landmark[self.LANDMARK_CHIN], sW, sH)
        src_jl = self._landmark_xy(source_face.landmark[self.LANDMARK_LEFT_JAW], sW, sH)
        src_jr = self._landmark_xy(source_face.landmark[self.LANDMARK_RIGHT_JAW], sW, sH)

        head_pts = np.array([head_chin, head_jl, head_jr], dtype=np.float32)
        src_pts = np.array([src_chin, src_jl, src_jr], dtype=np.float32)

        # 三点共线检查
        v1 = head_pts[1] - head_pts[0]
        v2 = head_pts[2] - head_pts[0]
        if abs(v1[0] * v2[1] - v1[1] * v2[0]) < 1.0:
            raise RuntimeError("头像下颌三点共线，无法计算 affine 变换")
        v1s = src_pts[1] - src_pts[0]
        v2s = src_pts[2] - src_pts[0]
        if abs(v1s[0] * v2s[1] - v1s[1] * v2s[0]) < 1.0:
            raise RuntimeError("原图下颌三点共线，无法计算 affine 变换")

        # affine warp 原图到头像坐标系
        M = cv2.getAffineTransform(src_pts, head_pts)
        warped = cv2.warpAffine(
            source_bgra, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # 构造脖子 polygon
        poly = self._build_neck_polygon(head_face, w, h)

        # AA mask（超采样 + INTER_AREA 下采样消锯齿）
        ss = self.POLYGON_SUPERSAMPLE
        big_mask = np.zeros((h * ss, w * ss), dtype=np.uint8)
        cv2.fillPoly(
            big_mask,
            [np.round(poly * ss).astype(np.int32).reshape(1, -1, 2)],
            255,
            lineType=cv2.LINE_AA,
        )
        mask = cv2.resize(big_mask, (w, h), interpolation=cv2.INTER_AREA)

        # 在 polygon 内用 warped 替换头像 BGR；polygon 边缘用 mask 0-255 做软过渡
        soft = (mask.astype(np.float32) / 255.0)[..., None]
        warped_bgr = warped[:, :, :3].astype(np.float32)
        head_bgr = head_bgra[:, :, :3].astype(np.float32)
        out_bgr = warped_bgr * soft + head_bgr * (1.0 - soft)

        # alpha：head 原 alpha 与 polygon mask 取 max（不缩小已有不透明区）
        new_alpha = np.maximum(head_bgra[:, :, 3], mask)

        out = np.dstack(
            [np.clip(np.round(out_bgr), 0, 255).astype(np.uint8), new_alpha]
        )
        return out

    def _build_neck_polygon(self, landmarks, w: int, h: int) -> np.ndarray:
        """
        构造脖子梯形 polygon：
          上沿：左下颌角 (172) → 下巴 (152) → 右下颌角 (397)，沿下颌弧线，
                整体上移 chin_overlap_px 让脖子顶部"陷"入下巴底
          下沿：上沿各点水平方向按 NECK_BOTTOM_FLARE 从中心外扩，
                整体下移 jaw_span × NECK_DEPTH_FRAC

        额外应用 NECK_TOP_INSET 让上沿整体内收（脖子接近下颌宽避免双下巴感）。
        返回 (N, 2) float64 像素坐标。
        """
        lm = landmarks.landmark
        chin = self._landmark_xy(lm[self.LANDMARK_CHIN], w, h)
        jl = self._landmark_xy(lm[self.LANDMARK_LEFT_JAW], w, h)
        jr = self._landmark_xy(lm[self.LANDMARK_RIGHT_JAW], w, h)
        jaw_span = float(np.hypot(jl[0] - jr[0], jl[1] - jr[1]))
        if jaw_span < 12.0:
            raise RuntimeError("jaw_span 过小，可能未检测到完整人脸")

        # 上沿用下颌弧线 5 个等距点（左→中→右），简化的 face oval 路径
        # 实际位置取在三点之间线性插值，效果上是个光滑的下颌弧
        n_top = 7
        ts = np.linspace(0.0, 1.0, n_top)
        top_xy = np.zeros((n_top, 2), dtype=np.float64)
        for i, t in enumerate(ts):
            if t < 0.5:
                tt = t / 0.5
                top_xy[i, 0] = jl[0] + tt * (chin[0] - jl[0])
                top_xy[i, 1] = jl[1] + tt * (chin[1] - jl[1])
            else:
                tt = (t - 0.5) / 0.5
                top_xy[i, 0] = chin[0] + tt * (jr[0] - chin[0])
                top_xy[i, 1] = chin[1] + tt * (jr[1] - chin[1])

        # 上移 chin_overlap 把脖子顶部"陷"入下巴
        chin_overlap = jaw_span * self.CHIN_OVERLAP_FRAC
        top_xy[:, 1] -= chin_overlap

        # NECK_TOP_INSET：所有上沿点向中心收缩
        cx_top = float(np.mean(top_xy[:, 0]))
        top_xy[:, 0] = cx_top + (top_xy[:, 0] - cx_top) * self.NECK_TOP_INSET

        # 下沿：上沿各点水平方向按 NECK_BOTTOM_FLARE 从中心外扩，y 统一下移 neck_depth
        cx = float(np.mean(top_xy[:, 0]))
        bot_y = float(np.max(top_xy[:, 1])) + jaw_span * self.NECK_DEPTH_FRAC
        bottom_xy = np.column_stack([
            cx + (top_xy[:, 0] - cx) * self.NECK_BOTTOM_FLARE,
            np.full(n_top, bot_y, dtype=np.float64),
        ])

        # 闭环：上沿左→右 + 下沿右→左
        poly = np.vstack([top_xy, bottom_xy[::-1]])
        poly[:, 0] = np.clip(poly[:, 0], 0.0, float(w - 1))
        poly[:, 1] = np.clip(poly[:, 1], 0.0, float(h - 1))
        return poly

    # =================================================================================
    # 核心方法 2：合成证件照（bg ← head ← clothes）
    # =================================================================================

    def _compose(self, head_with_neck: np.ndarray, clothes: np.ndarray) -> np.ndarray:
        """
        合成最终证件照：
          - canvas 宽度 = 衣服模板原始宽度（衣服不缩放）
          - canvas 高度自动推导 = 头顶 margin + 头部 + 脖子 + 衣服底部
          - 头像缩放使 scaled_jaw_span = canvas_w × HEAD_JAW_TO_CANVAS_RATIO
          - 衣服 V 领顶点对齐到 chin 下方 scaled_jaw_span × NECK_VISIBLE_TO_JAW_RATIO

        ★ 合成顺序：底色 ← 头像 ← 衣服（最上层）
          衣服 V 领开口 alpha=0 → 露出头像（脸/脖子）
          衣服布料区 alpha>0 → 遮住超出 V 领的脖子
        """
        # 重新检测脖子图上的人脸
        face = self._detect_face_landmarks(head_with_neck)
        if face is None:
            raise RuntimeError("加脖子后人脸检测失败")
        h, w = head_with_neck.shape[:2]
        chin_x = float(face.landmark[self.LANDMARK_CHIN].x * w)
        chin_y = float(face.landmark[self.LANDMARK_CHIN].y * h)
        jl = self._landmark_xy(face.landmark[self.LANDMARK_LEFT_JAW], w, h)
        jr = self._landmark_xy(face.landmark[self.LANDMARK_RIGHT_JAW], w, h)
        jaw_span = float(np.hypot(jl[0] - jr[0], jl[1] - jr[1]))

        # 头像 alpha bbox（找头顶 y）
        ys, _ = np.where(head_with_neck[:, :, 3] > 32)
        if len(ys) == 0:
            raise RuntimeError("头像 alpha 全透明")
        by0 = int(ys.min())

        # 衣服 V 领锚点
        cl_h, cl_w = clothes.shape[:2]
        vn_x, vn_y = self._find_clothes_vneck_anchor(clothes)

        # canvas 宽度 = 衣服宽度
        canvas_w = cl_w

        # 头像缩放比例
        target_jaw = canvas_w * self.HEAD_JAW_TO_CANVAS_RATIO
        scale = target_jaw / jaw_span
        scaled_w = max(1, int(round(w * scale)))
        scaled_h = max(1, int(round(h * scale)))
        flag = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        scaled_head = cv2.resize(head_with_neck, (scaled_w, scaled_h), interpolation=flag)

        scaled_chin_x = int(round(chin_x * scale))
        scaled_chin_y = int(round(chin_y * scale))
        scaled_by0 = int(round(by0 * scale))
        scaled_jaw = jaw_span * scale

        # 头部位置：头顶距画布顶 = canvas_w × HEAD_TOP_MARGIN_RATIO，水平 chin 居中
        head_top_margin = int(round(canvas_w * self.HEAD_TOP_MARGIN_RATIO))
        head_paste_y = head_top_margin - scaled_by0
        head_paste_x = canvas_w // 2 - scaled_chin_x
        chin_y_in_canvas = head_paste_y + scaled_chin_y

        # 衣服 V 领位置：chin 下方 scaled_jaw × NECK_VISIBLE_TO_JAW_RATIO
        neck_visible_h = int(round(scaled_jaw * self.NECK_VISIBLE_TO_JAW_RATIO))
        target_vneck_y = chin_y_in_canvas + neck_visible_h
        target_vneck_x = canvas_w // 2

        clothes_paste_x = target_vneck_x - vn_x
        clothes_paste_y = target_vneck_y - vn_y
        clothes_bottom_y = clothes_paste_y + cl_h

        canvas_h = max(clothes_bottom_y, target_vneck_y + 10)

        # 创建底色画布（RGBA）
        bg_b, bg_g, bg_r = self.bg_bgr
        canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
        canvas[:, :, 0] = bg_b
        canvas[:, :, 1] = bg_g
        canvas[:, :, 2] = bg_r
        canvas[:, :, 3] = 255

        # ★ 合成顺序：底色 ← 头像 ← 衣服（衣服在最上层覆盖脖子）
        canvas = self._alpha_paste(canvas, scaled_head, head_paste_x, head_paste_y)
        canvas = self._alpha_paste(canvas, clothes, clothes_paste_x, clothes_paste_y)
        return canvas

    # =================================================================================
    # 内部辅助
    # =================================================================================

    @staticmethod
    def _hex_to_bgr(hex_str: str) -> Tuple[int, int, int]:
        """16 进制颜色 → BGR tuple。支持 '#RGB' / '#RRGGBB' / 'RRGGBB' / '0xRRGGBB'。"""
        s = hex_str.strip().lstrip("#")
        if s.lower().startswith("0x"):
            s = s[2:]
        if len(s) == 3:
            s = "".join(c * 2 for c in s)
        if len(s) != 6:
            raise ValueError(f"Invalid hex color: {hex_str!r}")
        try:
            r = int(s[0:2], 16)
            g = int(s[2:4], 16)
            b = int(s[4:6], 16)
        except ValueError as e:
            raise ValueError(f"Invalid hex color: {hex_str!r}") from e
        return (b, g, r)

    @staticmethod
    def _imread(path: str) -> np.ndarray:
        """读取图像（支持中文路径），返回 BGR / BGRA uint8。"""
        path = os.path.normpath(path)
        with open(path, "rb") as f:
            raw = f.read()
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim != 3:
            raise ValueError(f"Invalid image: {path}")
        return img

    @staticmethod
    def _imwrite(path: str, img: np.ndarray) -> None:
        """写入图像（支持中文路径）。"""
        path = os.path.normpath(path)
        ext = os.path.splitext(path)[1].lower() or ".png"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            raise IOError(f"Failed to encode: {path}")
        with open(path, "wb") as f:
            f.write(buf.tobytes())

    @classmethod
    def _load_rgba(cls, path: str) -> np.ndarray:
        """加载图像并保证 RGBA（RGB 自动加 alpha=255）。"""
        img = cls._imread(path)
        if img.shape[2] == 3:
            bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = 255
            return bgra
        if img.shape[2] == 4:
            return img
        raise ValueError(f"Unexpected channels in {path}: shape={img.shape}")

    @classmethod
    def _detect_face_landmarks(cls, bgra: np.ndarray):
        """用 FaceMesh 检测第一个人脸；返回 landmarks 或 None。"""
        rgb = cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2RGB)
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=cls.FACE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=cls.FACE_MIN_DETECTION_CONFIDENCE,
        ) as fm:
            res = fm.process(rgb)
        if not res.multi_face_landmarks:
            return None
        return res.multi_face_landmarks[0]

    @staticmethod
    def _landmark_xy(landmark, w: int, h: int) -> Tuple[float, float]:
        """归一化 landmark → 像素坐标。"""
        return float(landmark.x * w), float(landmark.y * h)

    @classmethod
    def _ensure_neck_drawing_space(cls, bgra: np.ndarray) -> np.ndarray:
        """
        若 chin 下方画布空间不足画脖子（chin 太靠近底端），
        在画布底部加透明 padding，留 jaw_span × NECK_DRAWING_PAD_FACTOR。
        """
        face = cls._detect_face_landmarks(bgra)
        if face is None:
            return bgra
        h, w = bgra.shape[:2]
        chin_y = float(face.landmark[cls.LANDMARK_CHIN].y * h)
        jl = cls._landmark_xy(face.landmark[cls.LANDMARK_LEFT_JAW], w, h)
        jr = cls._landmark_xy(face.landmark[cls.LANDMARK_RIGHT_JAW], w, h)
        jaw_span = float(np.hypot(jl[0] - jr[0], jl[1] - jr[1]))
        if jaw_span < 12.0:
            return bgra
        needed = int(round(jaw_span * cls.NECK_DRAWING_PAD_FACTOR))
        available = h - int(chin_y)
        if available >= needed:
            return bgra
        pad = needed - available
        out = np.zeros((h + pad, w, 4), dtype=np.uint8)
        out[:h, :, :] = bgra
        return out

    @classmethod
    def _find_clothes_vneck_anchor(cls, clothes_bgra: np.ndarray) -> Tuple[int, int]:
        """
        自动找衣服模板 V 领顶点：中央 30% 列里 alpha 顶端（最上不透明像素 y）最低
        （即 V 形最深处）的列即为 V 领顶点。失败回退到固定比例。
        """
        h, w = clothes_bgra.shape[:2]
        alpha = clothes_bgra[:, :, 3]
        x_lo = int(w * 0.35)
        x_hi = int(w * 0.65)
        if x_hi <= x_lo + 1:
            return int(w * cls.CLOTHES_VNECK_X_FRAC), int(h * cls.CLOTHES_VNECK_Y_FRAC)
        best_x = int(w * cls.CLOTHES_VNECK_X_FRAC)
        best_y = -1
        for x in range(x_lo, x_hi):
            ys_col = np.where(alpha[:, x] > 32)[0]
            if ys_col.size == 0:
                continue
            top_y = int(ys_col[0])
            if top_y > best_y:
                best_y = top_y
                best_x = x
        if best_y < 0:
            return int(w * cls.CLOTHES_VNECK_X_FRAC), int(h * cls.CLOTHES_VNECK_Y_FRAC)
        return best_x, best_y

    @staticmethod
    def _alpha_paste(canvas: np.ndarray, layer: np.ndarray, x0: int, y0: int) -> np.ndarray:
        """RGBA layer 用 alpha-over 贴到 RGBA canvas 的 (x0, y0) 起点；返回新 canvas。"""
        H, W = canvas.shape[:2]
        lh, lw = layer.shape[:2]
        cx0 = max(x0, 0)
        cy0 = max(y0, 0)
        cx1 = min(x0 + lw, W)
        cy1 = min(y0 + lh, H)
        if cx1 <= cx0 or cy1 <= cy0:
            return canvas
        lx0 = cx0 - x0
        ly0 = cy0 - y0
        sub_c = canvas[cy0:cy1, cx0:cx1].astype(np.float32)
        sub_l = layer[ly0:ly0 + (cy1 - cy0), lx0:lx0 + (cx1 - cx0)].astype(np.float32)
        a_l = sub_l[:, :, 3:4] / 255.0
        a_c = sub_c[:, :, 3:4] / 255.0
        out_a = a_l + a_c * (1.0 - a_l)
        out_a_safe = np.maximum(out_a, 1e-6)
        out_rgb = (
            sub_l[:, :, :3] * a_l + sub_c[:, :, :3] * a_c * (1.0 - a_l)
        ) / out_a_safe
        new_c = canvas.copy()
        new_c[cy0:cy1, cx0:cx1, :3] = np.clip(np.round(out_rgb), 0, 255).astype(np.uint8)
        new_c[cy0:cy1, cx0:cx1, 3] = np.clip(np.round(out_a[:, :, 0] * 255.0), 0, 255).astype(np.uint8)
        return new_c


# =====================================================================================
# CLI
# =====================================================================================


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="生成证件照：抠图头像 + 原图 + 衣服模板 + 底色 → 蓝底证件照",
    )
    parser.add_argument("--head", required=True, help="抠图头像（RGBA PNG）")
    parser.add_argument("--source", required=True, help="原图（同人完整照片，含真实脖子）")
    parser.add_argument("--clothes", required=True, help="衣服模板（RGBA PNG）")
    parser.add_argument("-o", "--output", required=True, help="输出证件照路径")
    parser.add_argument(
        "--bg", default=MakeIDPhoto.DEFAULT_BG_HEX,
        help=f"16 进制底色，默认 {MakeIDPhoto.DEFAULT_BG_HEX}（标准证件照蓝）",
    )
    args = parser.parse_args(argv)

    try:
        job = MakeIDPhoto(
            head_path=os.path.abspath(args.head),
            source_path=os.path.abspath(args.source),
            clothes_path=os.path.abspath(args.clothes),
            output_path=os.path.abspath(args.output),
            bg_hex=args.bg,
        )
        return job.run()
    except Exception as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
