"""Diagnostic plots and machine-readable reports."""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse

from .io import write_image


def write_diagnostics(*, outdir, base, gray, names, ordered, ideal,
                      rejected, resid_used, rms_used, loo, affine):
    """Write detection/residual plots and center coordinate artifacts."""
    diag_dir = Path(outdir) / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    h_img, w_img = gray.shape
    small_scale = 1400.0 / max(w_img, h_img)
    small = cv2.resize(gray, None, fx=small_scale, fy=small_scale,
                       interpolation=cv2.INTER_AREA)

    fig1 = Figure(figsize=(12, 12 * small.shape[0] / small.shape[1]))
    ax1 = fig1.subplots()
    ax1.imshow(small, cmap="gray")
    for i, name in enumerate(names):
        x = ordered[i][0] * small_scale
        y = ordered[i][1] * small_scale
        ax1.plot(x, y, "o", mec="lime", mfc="none", ms=14, mew=1.5)
        ax1.annotate(name, (x, y), textcoords="offset points",
                     xytext=(10, 10), color="lime", fontsize=11)
    for candidate in rejected:
        ax1.plot(candidate["cx"] * small_scale,
                 candidate["cy"] * small_scale, "rx", ms=10, mew=2)
    ax1.set_title("Detected (green) / rejected (red)")
    ax1.axis("off")
    fig1.tight_layout()
    detection_path = diag_dir / f"{base}_detection.png"
    fig1.savefig(detection_path, dpi=150)

    fig2 = Figure(figsize=(14, 6))
    ax2a, ax2b = fig2.subplots(1, 2)
    ax2a.imshow(small, cmap="gray")
    quiver_scale = 20.0
    suspect_ids = set(loo["suspects"])
    for i, name in enumerate(names):
        x = ordered[i][0] * small_scale
        y = ordered[i][1] * small_scale
        ax2a.plot(x, y, "o", mec="lime", mfc="none", ms=10, mew=1.2)
        ax2a.annotate(name, (x, y), textcoords="offset points",
                      xytext=(8, 8), color="lime", fontsize=10)
        ax2a.arrow(
            x, y,
            resid_used[i][0] * quiver_scale * small_scale,
            resid_used[i][1] * quiver_scale * small_scale,
            color="red", width=1.2, head_width=5,
        )
        if name in suspect_ids:
            ax2a.plot(x, y, "o", mec="darkorange", mfc="none",
                      ms=22, mew=2.2)
    ax2a.set_title(f"Residual vectors (x{quiver_scale:.0f})")
    ax2a.axis("off")

    magnitudes = np.linalg.norm(resid_used, axis=1)
    colors = ["darkorange" if name in suspect_ids else "steelblue"
              for name in names]
    ax2b.bar(names, magnitudes, color=colors)
    ax2b.set_ylabel("Residual (px)")
    title = f"Per-mark residual  (RMS = {rms_used:.3f} px)"
    if len(names) < (4 if affine else 5):
        title += "  [no redundancy - round-off only, see self-check]"
    elif suspect_ids:
        title += "  orange = LOO-suspect"
    ax2b.set_title(title)
    ax2b.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    residual_path = diag_dir / f"{base}_residuals.png"
    fig2.savefig(residual_path, dpi=150)

    centers_csv = diag_dir / f"{base}_centers.csv"
    with centers_csv.open("w", encoding="utf-8") as handle:
        handle.write("id,x,y\n")
        for i, name in enumerate(names):
            handle.write(f"{name},{ordered[i][0]:.3f},{ordered[i][1]:.3f}\n")

    annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    annot_arm, annot_scale, annot_thick = 32, 1.1, 2
    annot_texts = [f"{names[i]} ({ordered[i][0]:.1f}, {ordered[i][1]:.1f})"
                   for i in range(len(names))]
    annot_layout = _layout_center_labels(
        annotated.shape, ordered, annot_texts,
        annot_arm, annot_scale, annot_thick)
    for i, name in enumerate(names):
        xi, yi = int(round(ordered[i][0])), int(round(ordered[i][1]))
        cv2.line(annotated, (xi - 22, yi), (xi + 22, yi), (0, 255, 0), 1)
        cv2.line(annotated, (xi, yi - 22), (xi, yi + 22), (0, 255, 0), 1)
        cv2.circle(annotated, (xi, yi), 32, (0, 255, 0), 2)
    for text, lx, ly in annot_layout:
        _draw_label(annotated, text, lx, ly, annot_scale, annot_thick,
                    (0, 255, 0))
    annotated_path = diag_dir / f"{base}_centers.png"
    write_image(annotated_path, annotated)

    return {
        "detection_overlay": os.fspath(detection_path),
        "residual_plot": os.fspath(residual_path),
        "centers_csv": os.fspath(centers_csv),
        "centers_annotated": os.fspath(annotated_path),
    }


_RED = (0, 0, 255)

# 校正图上中心标记的默认臂长（px，自中心向外）。固定 3，**不随图像尺寸缩放**
# —— 同一参数在任何图上都是同一个像素数，行为可预期。
# 臂长 3 → 总宽 7 px 的十字，由 13 个红色像素组成（7 横 + 7 竖 − 中心 1 个重叠）。
# 需要别的尺寸用 CLI 的 --mark-arm（同样是绝对像素）。
MARK_ARM_DEFAULT = 3
# 线宽（px）。恒为 1 —— 这是栅格图像的线宽下限，不可能更细。
# 它曾经随图像尺寸缩放到 2 px，于是出现"传了 --mark-arm 线更细、不传更粗"
# 这种自相矛盾的行为（同一张图线宽取决于有没有传参数），现已去掉缩放。
MARK_LINE_WIDTH = 1


def _load_bgr(image):
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = np.array(image, copy=True)
    if canvas.dtype != np.uint8:
        canvas = cv2.normalize(canvas, None, 0, 255,
                               cv2.NORM_MINMAX).astype(np.uint8)
    return canvas


def _rect_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersection(a, b):
    return _rect_area((max(a[0], b[0]), max(a[1], b[1]),
                       min(a[2], b[2]), min(a[3], b[3])))


def _layout_center_labels(shape, points, texts, arm, scale, thickness):
    """给每个中心坐标标签选一个位置：右 → 左 → 下 → 上 依次试放，
    以"压住标记/压住已放标签/越出画面"的重叠面积为代价取最优。
    四角布局下标签都朝同一侧必然互撞，必须逐点避让。"""
    h, w = shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    keepout = [(int(round(x)) - arm, int(round(y)) - arm,
                int(round(x)) + arm, int(round(y)) + arm)
               for x, y in points]
    placed, layout = [], []
    for (x, y), text in zip(points, texts):
        (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
        xi, yi = int(round(x)), int(round(y))
        gap = 8
        options = [
            (xi + arm + gap, yi + th // 2),
            (xi - arm - gap - tw, yi + th // 2),
            (xi - tw // 2, yi + arm + gap + th),
            (xi - tw // 2, yi - arm - gap),
        ]
        best = None
        for ox, oy in options:
            cx = int(np.clip(ox, 4, max(4, w - tw - 4)))
            cy = int(np.clip(oy, th + 6, h - 8))
            box = (cx, cy - th, cx + tw, cy + base)
            cost = (4.0 * sum(_intersection(box, k) for k in keepout)
                    + 6.0 * sum(_intersection(box, b) for b in placed)
                    + abs(cx - ox) + abs(cy - oy))
            if best is None or cost < best[0]:
                best = (cost, cx, cy, box)
        _, cx, cy, box = best
        placed.append(box)
        layout.append((text, cx, cy))
    return layout


def _draw_label(canvas, text, x, y, scale, thickness, color):
    """黑描边 + 彩色正文，保证在任何底色上都可读。"""
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def draw_center_marks(corrected, centers, arm_px=None):
    """在（内存中的）校正图上用红色小十字标出每个 mark 中心，返回带标记的
    3 通道图，由调用方写盘。

    只做定位指示：极小十字 + 无文字、无圆圈，避免遮挡样品细节。坐标本身
    不进图，需要数值请看 *_centers_corrected.csv 与报告 corrected_centers。

    arm_px=None  → 用默认臂长 MARK_ARM_DEFAULT = 3，即总宽 7 px、**13 个
                   红色像素**的正十字（7 横 + 7 竖 − 中心 1 个重叠）。
                   固定值，不随图像尺寸缩放。
    arm_px=N>0   → 用 N 作为臂长（自中心向外的像素数），同样是绝对像素；
    arm_px=0     → 只画中心那 1 个像素（栅格图像能做到的最小标记）。

    线宽恒为 MARK_LINE_WIDTH = 1 px —— 栅格图像的线宽下限。再"细"只能用
    抗锯齿把强度摊到相邻像素，那是**变淡**而不是变细。线宽与是否传 arm_px
    无关，只受这一个常数控制。

    注意：务必在 self_check 之后调用 —— 端到端自检必须在**干净**的校正图上
    重新检测，不能被这些红线干扰。"""
    canvas = _load_bgr(corrected)
    h, w = canvas.shape[:2]
    arm = MARK_ARM_DEFAULT if arm_px is None else max(0, int(arm_px))
    thickness = max(1, int(MARK_LINE_WIDTH))
    for x, y in centers:
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < w and 0 <= yi < h):
            continue
        if arm == 0:
            canvas[yi, xi] = _RED
            continue
        cv2.line(canvas, (max(0, xi - arm), yi), (min(w - 1, xi + arm), yi),
                 _RED, thickness)
        cv2.line(canvas, (xi, max(0, yi - arm)), (xi, min(h - 1, yi + arm)),
                 _RED, thickness)
    return canvas


def write_corrected_centers_csv(*, outdir, base, names, centers, sources, unit):
    """校正图上的 mark 中心坐标（机器可读）。"""
    diag_dir = Path(outdir) / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    path = diag_dir / f"{base}_centers_corrected.csv"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("id,x,y,source,unit\n")
        for i, name in enumerate(names):
            handle.write("%s,%.3f,%.3f,%s,%s\n"
                         % (name, centers[i][0], centers[i][1],
                            sources[i], unit))
    return os.fspath(path)


def write_failure_overlay(*, outdir, base, gray, accepted, rejected,
                          predicted=None, slot=None, spacing=None):
    """标记不齐全时的诊断图：绿=已接受，红=被图像门剔除，橙=缺失格位的
    几何预测位置（含搜索半径）。用于直接回答"是不是十字本身残缺"。
    图内文字一律英文 —— 与其余诊断图一致，避开 DejaVu 无中文字形的问题。"""
    diag_dir = Path(outdir) / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    small_scale = 1400.0 / max(gray.shape)
    small = cv2.resize(gray, None, fx=small_scale, fy=small_scale,
                       interpolation=cv2.INTER_AREA)
    fig = Figure(figsize=(12, 12 * small.shape[0] / small.shape[1]))
    ax = fig.subplots()
    ax.imshow(small, cmap="gray")
    for c in accepted:
        x, y = c["rx"] * small_scale, c["ry"] * small_scale
        ax.plot(x, y, "o", mec="lime", mfc="none", ms=16, mew=2)
        ax.annotate("accepted", (x, y), textcoords="offset points",
                    xytext=(12, 10), color="lime", fontsize=11)
    for c in rejected:
        ax.plot(c["cx"] * small_scale, c["cy"] * small_scale,
                "x", color="red", ms=11, mew=2)
    if predicted is not None:
        px, py = predicted[0] * small_scale, predicted[1] * small_scale
        ax.plot(px, py, "o", mec="darkorange", mfc="none", ms=30, mew=2.5)
        if spacing:
            ax.add_patch(Ellipse((px, py),
                                 0.4 * spacing * small_scale,
                                 0.4 * spacing * small_scale,
                                 fill=False, ec="darkorange",
                                 ls="--", lw=1.2))
        name = "missing slot" if slot is None else f"missing {slot}"
        ax.annotate(f"{name} (predicted)\n({predicted[0]:.1f}, {predicted[1]:.1f})",
                    (px, py), textcoords="offset points",
                    xytext=(0, 44), ha="center", color="darkorange",
                    fontsize=11)
    ax.set_title("Incomplete mark set  -  green: accepted | red: rejected | "
                 "orange: predicted missing slot", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    path = diag_dir / f"{base}_detection_failed.png"
    fig.savefig(path, dpi=150)
    return os.fspath(path)


def write_report(*, outdir, base, report):
    path = Path(outdir) / "diagnostics" / f"{base}_report.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return os.fspath(path)


def print_outputs(outputs, verbose=True):
    if not verbose:
        return
    print("\n输出：")
    print("  校正图       : %s   [红=mark 中心]" % outputs["corrected_image"])
    print("  校正后坐标   : %s" % outputs["centers_corrected_csv"])
    if outputs.get("info_bar_strip"):
        print("  参数栏条带   : %s   [已从画面裁下，参数备查]"
              % outputs["info_bar_strip"])
    print("  定位诊断     : %s" % outputs["detection_overlay"])
    print("  残差诊断     : %s" % outputs["residual_plot"])
    print("  原图中心     : %s" % outputs["centers_csv"])
    print("  原图标注     : %s" % outputs["centers_annotated"])
    print("  报告         : %s" % outputs["report"])
