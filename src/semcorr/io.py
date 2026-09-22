"""Input validation and safe image output."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = (".tif", ".tiff", ".bmp", ".png", ".jpg", ".jpeg")
OUTPUT_SUFFIXES = (
    "_corrected.tif", "_detection.png", "_residuals.png",
    "_report.json", "_centers.csv", "_centers.png",
    "_corrected_marked.tif", "_corrected_marked.png",
    "_centers_corrected.csv", "_detection_failed.png",
    "_infobar.png",
)


def load_gray(path: str | os.PathLike[str]) -> np.ndarray:
    image = cv2.imread(os.fspath(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return image


def load_design(path: str | os.PathLike[str], names: list[str]):
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    missing = [name for name in names if name not in raw]
    if missing:
        raise RuntimeError("设计坐标缺少: " + ", ".join(missing))
    points = []
    for name in names:
        value = raw[name]
        if not isinstance(value, list) or len(value) != 2:
            raise RuntimeError(f"{name} 设计坐标必须是 [x, y]")
        x, y = map(float, value)
        if not np.isfinite([x, y]).all():
            raise RuntimeError(f"{name} 设计坐标必须是有限数值")
        points.append((x, y))
    return points


def list_images(directory: str | os.PathLike[str]) -> list[Path]:
    root = Path(directory)
    return sorted(
        path for path in root.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTS
        and not path.name.lower().endswith(OUTPUT_SUFFIXES)
    )


def write_image(path: str | os.PathLike[str], image: np.ndarray) -> None:
    if not cv2.imwrite(os.fspath(path), image):
        raise RuntimeError(f"图像写入失败: {path}")


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_passing_report(corrected_dir, stem, input_sha256):
    """Return (report, report_path, corrected_image) for a reusable PASS result.

    Prefers ``diagnostics/<stem>_report.json``; a renamed raw file can still
    reconnect through the unique matching ``input_sha256``. Requires PASS plus
    the on-disk corrected image — never a bare report.
    """
    corrected_dir = Path(corrected_dir)
    diag = corrected_dir / "diagnostics"
    if not diag.is_dir() or not input_sha256:
        return None
    candidates = []
    direct = diag / f"{stem}_report.json"
    if direct.is_file():
        candidates.append(direct)
    for path in sorted(diag.glob("*_report.json")):
        if path not in candidates:
            candidates.append(path)
    hits = []
    for report_path in candidates:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if report.get("input_sha256") != input_sha256:
            continue
        if report.get("quality_status") != "PASS":
            return None
        base = report_path.name[: -len("_report.json")]
        image = corrected_dir / f"{base}_corrected.tif"
        if image.is_file():
            hits.append((report, report_path, image))
        if report_path == direct:
            return (report, report_path, image) if image.is_file() else None
    return hits[0] if len(hits) == 1 else None


def pick_image_dialog() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError(
            "未提供图像路径，且当前环境缺少 tkinter；请直接传入图像路径。"
        ) from exc
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="选择要处理的 SE2 图像",
        filetypes=[("图像文件", "*.tif *.tiff *.bmp *.png *.jpg *.jpeg"),
                   ("所有文件", "*.*")],
    )
    root.destroy()
    if not path:
        raise RuntimeError("未选择图像，已取消。")
    return path
