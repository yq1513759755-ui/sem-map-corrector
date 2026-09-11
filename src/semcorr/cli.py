"""Command-line entry point for the SE2 correction pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from . import __version__
from .io import list_images, pick_image_dialog


def build_parser():
    parser = argparse.ArgumentParser(
        prog="semcorr",
        description="SE2 实心十字标记定位与 SEM 几何畸变校正",
    )
    parser.add_argument("image", nargs="?", help="输入图像路径")
    parser.add_argument("--batch", metavar="目录", help="批量处理目录")
    parser.add_argument("--design", help="M1..Mn 设计坐标 JSON")
    parser.add_argument("--grid", default="2x2",
                        help="标记网格 行x列，默认 2x2")
    parser.add_argument("--outdir", help="输出目录")
    parser.add_argument("--affine", action="store_true",
                        help="使用全局仿射而不是默认精确单应校正")
    parser.add_argument("--mark-arm", type=int, default=None, metavar="PX",
                        help="校正图上 mark 中心红色小十字的臂长"
                             "（像素，自中心向外的长度）："
                             "1 → 总宽 3 px / 5 个像素，"
                             "0 → 只画中心 1 个像素；"
                             "缺省 3 → 总宽 7 px / 13 个像素的十字")
    parser.add_argument("--keep-info-bar", action="store_true",
                        help="保留下方的 SEM 参数信息栏。默认自动检测并裁掉"
                             "（裁下的条带另存为 *_infobar.png 备查）")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    return parser


def _validate_grid(value):
    try:
        rows, cols = [int(part) for part in value.lower().split("x")]
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("--grid 必须写成 行x列，例如 2x2 或 2x3") from exc
    if rows < 1 or cols < 1 or rows * cols < 3:
        raise RuntimeError("--grid 至少需要 3 个标记")


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        from .pipeline import correct_image

        _validate_grid(args.grid)
        if args.mark_arm is not None and args.mark_arm < 0:
            raise RuntimeError("--mark-arm 不能为负数（0 = 只画中心 1 个像素）")
        if args.batch:
            root = Path(args.batch)
            if not root.is_dir():
                raise RuntimeError(f"批量目录不存在: {root}")
            files = list_images(root)
            if not files:
                raise RuntimeError(f"目录中没有可处理的图像: {root}")
            outdir = Path(args.outdir) if args.outdir else root / "corrected"
            print(f"批量处理 {len(files)} 张 SE2 图像 → {outdir}")
            failures = []
            for index, path in enumerate(files, 1):
                print(f"\n================ [{index}/{len(files)}] {path.name} ================")
                try:
                    correct_image(path, grid=args.grid, design=args.design,
                                  outdir=outdir, affine=args.affine,
                                  mark_arm=args.mark_arm,
                                  keep_info_bar=args.keep_info_bar)
                except (RuntimeError, FileNotFoundError, ValueError) as exc:
                    print(f"失败：{exc}")
                    failures.append((path.name, str(exc)))
            print(f"\n===== 批量完成：成功 {len(files) - len(failures)} / 失败 {len(failures)} =====")
            for name, error in failures:
                print(f"  [失败] {name}: {error}")
            return 1 if failures else 0

        image = args.image or pick_image_dialog()
        correct_image(image, grid=args.grid, design=args.design,
                      outdir=args.outdir, affine=args.affine,
                      mark_arm=args.mark_arm,
                      keep_info_bar=args.keep_info_bar)
        return 0
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"错误：{exc}")
        return 1
