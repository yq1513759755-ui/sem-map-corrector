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
    parser.add_argument("--batch", metavar="目录",
                        help="批量处理目录（默认顺带生成 AutoCAD 贴图包）")
    parser.add_argument("--cad", action="store_true",
                        help="显式要求生成 AutoCAD 贴图包（--batch 已默认开启，一般不必写）")
    parser.add_argument("--no-cad", action="store_true",
                        help="批量时跳过 AutoCAD 贴图包（文件名可不按区域编号）")
    parser.add_argument("--pitch-um", type=float, default=50., help="区域网格固定间距（仅支持 50 µm）")
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
    parser.add_argument("--qc-only", action="store_true",
                        help="只基于已有 corrected/ 生成批次质检与邻格一致性报告，不重新校正")
    parser.add_argument("--neighbor-tol-um", type=float, default=0.05,
                        help="邻格共享 mark 允许偏差（µm，默认 0.05）")
    parser.add_argument("--force", action="store_true",
                        help="忽略已有 PASS 结果，强制重新校正每一张图")
    parser.add_argument("--dxf", action="store_true",
                        help="可选：把贴图结果合并进 DXF 模板并另存 DXF 2018"
                             "（个人工作流；默认模板 ~/PhD/dxf-gds/RAW/Marker.dxf）")
    parser.add_argument("--dxf-template", metavar="PATH",
                        help="DXF 模板路径（配合 --dxf）")
    parser.add_argument("--dxf-outdir", metavar="PATH",
                        help="DXF 输出目录（配合 --dxf，默认 ~/PhD/dxf-gds/DXF）")
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
    # `./semcorr DIR` is the common case: treat a directory argument as --batch.
    if args.image and not args.batch:
        candidate = Path(args.image)
        if candidate.is_dir():
            args.batch = args.image
            args.image = None
    try:
        from .pipeline import correct_image
        from .io import find_passing_report, sha256_file

        _validate_grid(args.grid)
        if args.cad and args.no_cad:
            raise RuntimeError("--cad 与 --no-cad 不能同时使用")
        if args.cad and not args.batch:
            raise RuntimeError("--cad 必须与 --batch 目录一起使用")
        use_cad = False
        if args.batch and not args.qc_only:
            if args.no_cad:
                use_cad = False
            elif args.grid.lower() == "2x2" and not args.design:
                use_cad = True
            elif args.cad:
                raise RuntimeError("一键 CAD 流程要求 --grid 2x2，且不使用 --design；绝对坐标从文件名读取")
            else:
                print("提示：非 2x2 或使用 --design，已跳过 AutoCAD 贴图包（可去掉 --design 并整理为 2x2 区域命名）")
        if use_cad:
            if not all(math.isfinite(v) and v > 0 for v in (args.pitch_um, args.max_residual_um)):
                raise RuntimeError("CAD 间距和残差门槛必须为有限正数")
            from .cad import parse_region, validate_region_pitch
            validate_region_pitch(args.pitch_um)
        elif args.pitch_um != 50. or args.max_residual_um != .05:
            raise RuntimeError("CAD 参数仅在贴图包流程中生效（默认开启；跳过请用 --no-cad）")
        if args.mark_arm is not None and args.mark_arm < 0:
            raise RuntimeError("--mark-arm 不能为负数（0 = 只画中心 1 个像素）")
        if args.dxf and not args.batch:
            raise RuntimeError("--dxf 必须与 --batch 目录一起使用")
        if args.dxf and not use_cad:
            raise RuntimeError("--dxf 需要贴图包流程（不要用 --no-cad / 非 2x2 / --design）")
        if not math.isfinite(args.neighbor_tol_um) or args.neighbor_tol_um <= 0:
            raise RuntimeError("--neighbor-tol-um 必须为有限正数")
        if args.qc_only:
            if not args.batch:
                raise RuntimeError("--qc-only 必须与 --batch 目录一起使用")
            from .batch_qc import run_batch_qc

            root = Path(args.batch)
            outdir = Path(args.outdir) if args.outdir else root / "corrected"
            payload = run_batch_qc(root, corrected_dir=outdir, outdir=outdir,
                                   neighbor_tol_um=args.neighbor_tol_um)
            summary = payload["summary"]
            print(f"质检汇总：{ (outdir / 'batch_qc.html').resolve() }")
            print(f"逐图表：{ (outdir / 'batch_qc.csv').resolve() }")
            print(f"邻格对比：{ (outdir / 'neighbor_checks.csv').resolve() }")
            print(f"图像 {summary['n_images']} / 对比 {summary['n_pair_checks']} / "
                  f"不一致 {summary['n_mismatches']} / 门限 {args.neighbor_tol_um:g} µm")
            return 1 if summary["n_mismatches"] else 0
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
            naming_errors = {}
            reused = []
            for index, path in enumerate(files, 1):
                print(f"\n================ [{index}/{len(files)}] {path.name} ================")
                if use_cad:
                    try:
                        parse_region(path.stem)
                    except ValueError as exc:
                        naming_errors[path.name] = str(exc)
                        print(f"命名无效：{path.name}: {exc}")
                        continue
                if not args.force:
                    try:
                        digest = sha256_file(path)
                    except OSError as exc:
                        failures.append((path.name, f"无法读取原图：{exc}"))
                        print(f"失败：无法读取原图：{exc}")
                        continue
                    hit = find_passing_report(outdir, path.stem, digest)
                    if hit is not None:
                        report, report_path, corrected = hit
                        reused.append((path.name, report_path.name))
                        print(f"跳过：哈希一致，沿用 PASS（{report_path.name}）")
                        continue
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
            print(f"\n===== 批量完成：增量复用 {len(reused)} / 新算通过 {len(files) - len(failures) - len(reviews) - len(naming_errors) - len(reused)} / 需复核 {len(reviews)} / 失败 {len(failures)} / 命名无效 {len(naming_errors)} =====")
            for name, report_name in reused:
                print(f"  [复用] {name}: {report_name}")
            for name, warnings in reviews:
                print(f"  [需复核] {name}: {'; '.join(warnings)}")
            for name, error in failures:
                print(f"  [失败] {name}: {error}")
            if use_cad:
                from .cad import export_batch

                excluded = {name: "本次校正失败：" + error for name, error in failures}
                excluded.update({name: "本次校正需复核：" + "; ".join(warnings)
                                 for name, warnings in reviews})
                excluded.update(naming_errors)
                cad = export_batch(root, corrected_dir=outdir, excluded=excluded,
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
                if summary["ready"] and not args.dxf:
                    print("在 AutoCAD 空闲状态 APPLOAD 加载上述程序，再输入 SEMMAPONE 试贴或 SEMMAP 批量贴图。")
                dxf_exit = 0
                if args.dxf:
                    try:
                        from .dxf_export import (
                            DEFAULT_OUTDIR, DEFAULT_TEMPLATE, export_dxf_from_cad,
                        )

                        template = (Path(args.dxf_template).expanduser()
                                    if args.dxf_template else DEFAULT_TEMPLATE)
                        dxf_outdir = (Path(args.dxf_outdir).expanduser()
                                      if args.dxf_outdir else DEFAULT_OUTDIR)
                        result = export_dxf_from_cad(
                            outdir / "cad", template=template, outdir=dxf_outdir,
                            name=root.name)
                        print(f"DXF 2018：{result['output']}（{result['n_images']} 张）")
                    except (OSError, ValueError, RuntimeError) as exc:
                        print(f"DXF 导出失败：{exc}")
                        dxf_exit = 1
                qc_exit = 0
                try:
                    from .batch_qc import run_batch_qc

                    payload = run_batch_qc(root, corrected_dir=outdir, outdir=outdir,
                                           neighbor_tol_um=args.neighbor_tol_um)
                    n_mis = payload["summary"]["n_mismatches"]
                    print(f"质检报告：{(outdir / 'batch_qc.html').resolve()}")
                    print(f"邻格不一致：{n_mis}（门限 {args.neighbor_tol_um:g} µm）")
                    qc_exit = 1 if n_mis else 0
                except (OSError, ValueError) as exc:
                    print(f"质检汇总失败：{exc}")
                return 1 if cad["skipped"] or qc_exit or dxf_exit else 0
            try:
                from .batch_qc import run_batch_qc

                payload = run_batch_qc(root, corrected_dir=outdir, outdir=outdir,
                                       neighbor_tol_um=args.neighbor_tol_um)
                print(f"质检报告：{(outdir / 'batch_qc.html').resolve()}")
                print(f"邻格不一致：{payload['summary']['n_mismatches']}")
            except (OSError, ValueError) as exc:
                print(f"质检汇总失败：{exc}")
            return 1 if failures else 0

        image = args.image or pick_image_dialog()
        image_path = Path(image)
        if image_path.is_dir():
            raise RuntimeError(
                f"这是目录，请用批量模式：./semcorr --batch {image}（或直接 ./semcorr {image}）")
        if not image_path.is_file():
            raise RuntimeError(f"无法读取图像: {image}")
        correct_image(image, grid=args.grid, design=args.design,
                      outdir=args.outdir, affine=args.affine,
                      mark_arm=args.mark_arm,
                      keep_info_bar=args.keep_info_bar)
        return 0
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"错误：{exc}")
        return 1
