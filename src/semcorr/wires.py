"""纳米线绿色标注的检测、映射与 wire 矩形生成。

SEM 原图（彩色）上用纯绿 ``#00FF00`` 描出的细线段是纳米线位置的人工标注。
本模块：

1. ``detect_green_segments`` —— 在原图上检出绿色线段（参数栏区域除外，
   栏内可能有绿色的界面图形，不是纳米线）；
2. ``map_segments_to_space`` —— 用校正模型 ``H_out_to_in`` 的逆把线段从
   原图像素映射到校正图像素（线段逐像素映射后重新拟合，吸收校正形变）；
3. ``segment_to_quad`` —— 把线段变成细长闭合矩形：长度比纳米线两端各外延
   ``extend_um``，宽度取实验室既有 wire 约定（默认 0.5 µm，与
   dxf-gds/wire_from_annotations.py 的 DEFAULT_WIRE_W 一致）。

像素→µm 用与 CAD 贴图完全相同的仿射（``fit_bottom_left`` 的 matrix/origin，
半像素约定 ``(x+0.5, H-y-0.5)``），保证 wire 与照片贴在同一坐标系。
"""

from __future__ import annotations

import math

import cv2
import numpy as np

# 纯绿标注判据：绿通道显著高于红/蓝通道（抗 JPEG/TIFF 轻微色偏）
GREEN_LEVEL_MIN = 100
GREEN_LEAD = 40
MIN_SEGMENT_AREA_PX = 8
# 低于该长度的绿色碎点不算线段（µm 级纳米线在 0.09 µm/px 下至少几十像素）
MIN_SEGMENT_LENGTH_PX = 8.0
WIRE_WIDTH_UM = 0.5
WIRE_EXTEND_UM = 0.5


def green_mask(color_bgr, crop_top=None):
    """纯绿像素掩膜；``crop_top`` 以下（参数栏区域）一律清零。"""
    if color_bgr is None or color_bgr.ndim != 3:
        raise ValueError("绿色标注检测需要彩色图像")
    b, g, r = [color_bgr[:, :, i].astype(np.int16) for i in range(3)]
    mask = ((g - r > GREEN_LEAD) & (g - b > GREEN_LEAD)
            & (g > GREEN_LEVEL_MIN)).astype(np.uint8)
    if crop_top is not None and 0 <= crop_top < mask.shape[0]:
        mask[crop_top:, :] = 0
    return mask


def detect_green_segments(color_bgr, crop_top=None):
    """检出绿色线段，返回 [{p1, p2, pixels, length_px, width_px, n_px}]。

    连通域做 PCA：主轴方向即线段方向，两端点取主轴投影的极值，
    短轴宽度取 面积/长度（细笔画的平均厚度）。``pixels`` 为域内像素
    坐标 (x, y)，供映射到其他坐标系后重新拟合。
    """
    mask = green_mask(color_bgr, crop_top=crop_top)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    segments = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_SEGMENT_AREA_PX:
            continue
        ys, xs = np.nonzero(labels == i)
        pts = np.column_stack((xs, ys)).astype(np.float64)
        center = pts.mean(axis=0)
        centered = pts - center
        cov = centered.T @ centered / max(len(pts) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, int(np.argmax(eigvals))]
        proj = centered @ axis
        p1 = center + axis * proj.min()
        p2 = center + axis * proj.max()
        length = float(proj.max() - proj.min())
        if length < MIN_SEGMENT_LENGTH_PX:
            continue
        width = float(area) / length
        segments.append({
            "p1": (float(p1[0]), float(p1[1])),
            "p2": (float(p2[0]), float(p2[1])),
            "pixels": pts,
            "length_px": length,
            "width_px": width,
            "n_px": area,
        })
    segments.sort(key=lambda s: -s["length_px"])
    return segments


def map_segment(segment, H_inv):
    """把线段用 ``H_inv``（原图→校正图的 3x3 单应）映射到校正图坐标。

    逐像素映射后重新做 PCA 拟合，端点取投影极值 —— 校正形变会把直线
    微微掰弯，重新拟合比只搬两个端点更稳。返回与 detect 相同结构。
    """
    H_inv = np.asarray(H_inv, np.float64)
    if H_inv.shape != (3, 3) or not np.isfinite(H_inv).all():
        raise ValueError("映射矩阵必须是有限的 3x3")
    pts = segment["pixels"]
    homo = np.column_stack((pts, np.ones(len(pts)))) @ H_inv.T
    w = homo[:, 2]
    if np.any(np.abs(w) < 1e-12):
        raise ValueError("映射退化：像素落在无穷远")
    mapped = homo[:, :2] / w[:, None]
    center = mapped.mean(axis=0)
    centered = mapped - center
    cov = centered.T @ centered / max(len(mapped) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    proj = centered @ axis
    p1 = center + axis * proj.min()
    p2 = center + axis * proj.max()
    length = float(proj.max() - proj.min())
    return {
        "p1": (float(p1[0]), float(p1[1])),
        "p2": (float(p2[0]), float(p2[1])),
        "pixels": mapped,
        "length_px": length,
        "width_px": segment["width_px"],
        "n_px": segment["n_px"],
    }


def px_to_um_factory(height_px, matrix, origin_um):
    """校正图像素→版图 µm 的闭包，与 cad.fit_bottom_left 同一约定。

    ``matrix`` 列为 u_um/v_um（每像素向量），``origin_um`` 为图像外边界
    左下角；像素中心先换到外角点局部坐标 ``(x+0.5, H-y-0.5)``。
    """
    A = np.asarray(matrix, np.float64)
    o = np.asarray(origin_um, np.float64)
    if A.shape != (2, 2) or o.shape != (2,) or not np.isfinite(A).all():
        raise ValueError("像素→µm 参数无效")

    def to_um(point):
        x, y = float(point[0]), float(point[1])
        local = np.array([x + 0.5, float(height_px) - y - 0.5])
        u = A @ local + o
        return (float(u[0]), float(u[1]))

    return to_um


def segment_to_quad(p1_um, p2_um, width_um=WIRE_WIDTH_UM,
                    extend_um=WIRE_EXTEND_UM):
    """线段→细长闭合矩形四角（µm）：两端各外延 extend_um，宽度 width_um。"""
    x1, y1 = map(float, p1_um)
    x2, y2 = map(float, p2_um)
    vals = [x1, y1, x2, y2, width_um, extend_um]
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("wire 坐标/宽度/外延必须有限")
    if width_um <= 0 or extend_um < 0:
        raise ValueError("wire 宽度必须为正、外延不能为负")
    dx, dy = x2 - x1, y2 - y1
    d = math.hypot(dx, dy)
    if d < 1e-9:
        raise ValueError("wire 线段长度为零")
    ux, uy = dx / d, dy / d
    nx, ny = -uy, ux
    h = width_um / 2.0
    a = (x1 - ux * extend_um, y1 - uy * extend_um)
    b = (x2 + ux * extend_um, y2 + uy * extend_um)
    return [(a[0] + nx * h, a[1] + ny * h),
            (b[0] + nx * h, b[1] + ny * h),
            (b[0] - nx * h, b[1] - ny * h),
            (a[0] - nx * h, a[1] - ny * h)]


def draw_wire_overlay(corrected_bgr, segments_px, width_um, extend_um,
                      scale_um_per_px):
    """在校正图副本上画出绿色线段（黄）与 wire 矩形（青），供人工复核。"""
    canvas = corrected_bgr.copy()
    width_px = width_um / scale_um_per_px
    extend_px = extend_um / scale_um_per_px
    for seg in segments_px:
        p1 = tuple(int(round(v)) for v in seg["p1"])
        p2 = tuple(int(round(v)) for v in seg["p2"])
        cv2.line(canvas, p1, p2, (0, 255, 255), 1, cv2.LINE_AA)
        quad = segment_to_quad(seg["p1"], seg["p2"], width_px, extend_px)
        pts = np.array([[int(round(x)), int(round(y))] for x, y in quad],
                       np.int32)
        cv2.polylines(canvas, [pts], True, (255, 255, 0), 1, cv2.LINE_AA)
    return canvas


def detect_wires_in_corrected(raw_path, report, height_px, matrix,
                              origin_um, *, width_um=WIRE_WIDTH_UM,
                              extend_um=WIRE_EXTEND_UM):
    """一条完整链：原图绿色标注 → 校正图像素 → 版图 µm → wire 矩形。

    返回 (wires, segments_corrected_px)；wires 项含端点/矩形/长度等，
    segments_corrected_px 供绘制校正图上的复核叠加层。
    """
    if report.get("method") != "exact-square" or not report.get(
            "exact_square_cells"):
        raise ValueError("wire 需要 exact-square 校正模型（--affine 不支持）")
    cells = report["exact_square_cells"]
    if len(cells) != 1:
        raise ValueError("wire 目前仅支持 2x2 单格校正模型")
    H_out_to_in = np.asarray(cells[0]["H_out_to_in"], np.float64)
    H_inv = np.linalg.inv(H_out_to_in)
    color = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
    if color is None:
        raise ValueError("无法读取原图: %s" % raw_path)
    crop_top = (report.get("info_bar") or {}).get("top")
    segments = detect_green_segments(color, crop_top=crop_top)
    to_um = px_to_um_factory(height_px, matrix, origin_um)
    wires, mapped_segs = [], []
    for seg in segments:
        m = map_segment(seg, H_inv)
        mapped_segs.append(m)
        p1_um, p2_um = to_um(m["p1"]), to_um(m["p2"])
        length_um = math.hypot(p2_um[0] - p1_um[0], p2_um[1] - p1_um[1])
        quad = segment_to_quad(p1_um, p2_um, width_um, extend_um)
        wires.append({
            "p1_um": [p1_um[0], p1_um[1]],
            "p2_um": [p2_um[0], p2_um[1]],
            "length_um": length_um,
            "width_um": float(width_um),
            "extend_um": float(extend_um),
            "quad_um": [list(q) for q in quad],
            "px": {"p1": list(m["p1"]), "p2": list(m["p2"]),
                   "length_px": m["length_px"], "n_px": m["n_px"]},
        })
    return wires, mapped_segs
