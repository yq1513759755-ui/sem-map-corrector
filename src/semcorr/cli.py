"""Command-line entry point for the SE2 correction pipeline."""

from __future__ import annotations

import argparse
import json
import math
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
    parser.add_argument("--cad", action="store_true",
                        help="与 --batch 配合：校正后自动生成 AutoCAD 贴图包")
    parser.add_argument("--anchor-overrides", help="CAD 左下锚点覆盖 JSON，单位为 100 µm")
    parser.add_argument("--pitch-um", type=float, default=50., help="CAD 标记间距（µm，默认 50）")
    parser.add_argument("--max-residual-um", type=float, default=.05,
                        help="CAD 最大单点配准误差（µm，默认 0.05）")
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
        if args.cad:
            if not args.batch:
                raise RuntimeError("--cad 必须与 --batch 目录一起使用")
            if args.grid.lower() != "2x2" or args.design:
                raise RuntimeError("一键 CAD 流程要求 --grid 2x2，且不使用 --design；绝对坐标从文件名读取")
            if not all(math.isfinite(v) and v > 0 for v in (args.pitch_um, args.max_residual_um)):
                raise RuntimeError("CAD 间距和残差门槛必须为有限正数")
            overrides = Path(args.anchor_overrides) if args.anchor_overrides else Path(args.batch) / "cad_anchor_overrides.json"
            if args.anchor_overrides or overrides.exists():
                with overrides.open(encoding="utf-8") as handle:
                    if not isinstance(json.load(handle), dict):
                        raise RuntimeError("坐标覆盖 JSON 必须为对象")
        elif args.anchor_overrides or args.pitch_um != 50. or args.max_residual_um != .05:
            raise RuntimeError("CAD 参数需要同时指定 --cad")
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
            failures, reviews = [], []
            for index, path in enumerate(files, 1):
                print(f"\n================ [{index}/{len(files)}] {path.name} ================")
                try:
                    report = correct_image(path, grid=args.grid, design=args.design,
                                  outdir=outdir, affine=args.affine,
                                  mark_arm=args.mark_arm,
                                  keep_info_bar=args.keep_info_bar)
                    if report["quality_status"] != "PASS":
                        reviews.append((path.name, report["quality_warnings"]))
                except (RuntimeError, FileNotFoundError, ValueError) as exc:
                    print(f"失败：{exc}")
                    failures.append((path.name, str(exc)))
            print(f"\n===== 批量完成：通过 {len(files) - len(failures) - len(reviews)} / 需复核 {len(reviews)} / 失败 {len(failures)} =====")
            for name, warnings in reviews:
                print(f"  [需复核] {name}: {'; '.join(warnings)}")
            for name, error in failures:
                print(f"  [失败] {name}: {error}")
            if args.cad:
                from .cad import export_batch

                excluded = {name: "本次校正失败：" + error for name, error in failures}
                excluded.update({name: "本次校正需复核：" + "; ".join(warnings)
                                 for name, warnings in reviews})
                cad = export_batch(root, corrected_dir=outdir, excluded=excluded,
                                   overrides_path=args.anchor_overrides,
                                   pitch_um=args.pitch_um,
                                   max_residual_um=args.max_residual_um)
                exported = {row["name"] for row in cad["images"]}
                reasons = {row["name"]: row["reason"] for row in cad["skipped"]}
                failed_names = {name for name, _ in failures}
                review_names = {name for name, _ in reviews}
                entries = []
                for path in files:
                    status = ("correction_failed" if path.name in failed_names else
                              "review" if path.name in review_names else
                              "ready" if path.stem in exported else "cad_rejected")
                    entries.append({"image": path.name, "status": status,
                                    "reason": reasons.get(path.name)})
                summary = {"schema_version": 1, "input_dir": str(root.resolve()),
                           "output_dir": str(outdir.resolve()), "total": len(files),
                           "ready": len(cad["images"]), "skipped": len(cad["skipped"]),
                           "images": entries,
                           "cad_script": str((outdir / "cad" / "sem_map.lsp").resolve())}
                summary_path = outdir / "workflow_summary.json"
                summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"\n===== 一键流程完成：可贴图 {summary['ready']} / 未导出 {summary['skipped']} =====")
                for row in cad["skipped"]:
                    print(f"  [未导出] {row['name']}: {row['reason'].splitlines()[0]}")
                print(f"批次汇总：{summary_path.resolve()}")
                print(f"贴图程序：{summary['cad_script']}")
                if summary["ready"]:
                    print("在 AutoCAD 空闲状态 APPLOAD 加载上述程序，再输入 SEMMAPONE 试贴或 SEMMAP 批量贴图。")
                return 1 if cad["skipped"] else 0
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
