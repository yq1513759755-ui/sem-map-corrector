"""Diagnostic plots and machine-readable reports."""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

    fig1, ax1 = plt.subplots(
        figsize=(12, 12 * small.shape[0] / small.shape[1]))
    try:
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
    finally:
        plt.close(fig1)

    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 6))
    try:
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
    finally:
        plt.close(fig2)

    centers_csv = diag_dir / f"{base}_centers.csv"
    with centers_csv.open("w", encoding="utf-8") as handle:
        handle.write("id,x,y\n")
        for i, name in enumerate(names):
            handle.write(f"{name},{ordered[i][0]:.3f},{ordered[i][1]:.3f}\n")

    annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for i, name in enumerate(names):
        x, y = ordered[i]
        xi, yi = int(round(x)), int(round(y))
        cv2.line(annotated, (xi - 22, yi), (xi + 22, yi), (0, 255, 0), 1)
        cv2.line(annotated, (xi, yi - 22), (xi, yi + 22), (0, 255, 0), 1)
        cv2.circle(annotated, (xi, yi), 32, (0, 255, 0), 2)
        label = f"{name} ({x:.1f}, {y:.1f})"
        cv2.putText(annotated, label, (xi + 40, yi - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 4,
                    cv2.LINE_AA)
        cv2.putText(annotated, label, (xi + 40, yi - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2,
                    cv2.LINE_AA)
    annotated_path = diag_dir / f"{base}_centers.png"
    write_image(annotated_path, annotated)

    return {
        "detection_overlay": os.fspath(detection_path),
        "residual_plot": os.fspath(residual_path),
        "centers_csv": os.fspath(centers_csv),
        "centers_annotated": os.fspath(annotated_path),
    }


def write_report(*, outdir, base, report):
    path = Path(outdir) / "diagnostics" / f"{base}_report.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return os.fspath(path)


def print_outputs(outputs):
    print("\n输出：")
    print("  校正图   : %s" % outputs["corrected_image"])
    print("  定位诊断 : %s" % outputs["detection_overlay"])
    print("  残差诊断 : %s" % outputs["residual_plot"])
    print("  中心坐标 : %s" % outputs["centers_csv"])
    print("  中心标注 : %s" % outputs["centers_annotated"])
    print("  报告     : %s" % outputs["report"])
