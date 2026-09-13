#!/usr/bin/env python3
"""从 semcorr 校正结果生成 AutoCAD 精确贴图参数与 .scr 脚本。

工作流：把校正图按"插入点 + 比例 + 旋转"直接贴进版图坐标系（µm），
使 4 个 mark 中心与版图设计位置精确重合，免去手动 ALIGN 时人眼点击
mark 中心带来的 0.1~0.3 µm 误差。

每张图的处理：
  1. 文件名前 4 位解码大十字设计坐标（"0115" → x=100, y=1500 µm）；
  2. 在原图上检测 4 个 mark，按 span 找出大十字在哪个角（M1..M4）；
  3. 大十字锚定到解码坐标，三个小十字按 ±50 µm 格点分配设计坐标；
  4. 图像方向/镜像（共 4 种假设）自动裁决：只有真实方向能让
     原图像素→设计坐标的相似变换（Umeyama，无镜像）残差为零，
     错误方向组合必然需要镜像、残差会大到间距量级；
  5. 用报告 JSON 里 corrected_centers（即校正图上红十字的位置）拟合
     相似变换，输出 插入点/比例(µm每px)/旋转角(度，逆时针为正)。

用法（在本仓库根目录）：
    .venv/bin/python scripts/make_cad_align.py [图像文件夹]

    文件夹需含 原始tif + corrected/<名>_corrected.tif + 其报告 JSON。
    默认处理仓库内的 260910-3381-2 工作副本。

输出（<文件夹>/corrected/cad/）：
    cad_params.csv      每张图的三要素 + 拟合残差 + 方向结论
    attach_<名>.scr     单张图自动贴图脚本（AutoCAD 中 SCRIPT 命令运行）
    attach_all.scr      全部图按序贴好的合并脚本（整幅 map 一次成型）
"""

import csv
import glob
import json
import math
import os
import sys

import cv2
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))     # scripts/
_ROOT = os.path.dirname(_BASE)                         # 仓库根目录
sys.path.insert(0, os.path.join(_ROOT, "src"))

from semcorr.detectors.se2 import detect_marks, recover_missing   # noqa: E402
from semcorr.geometry import fit_similarity, rms_of               # noqa: E402
from semcorr.infobar import strip_info_bar                        # noqa: E402

# ===== 可调常数 =====
UM_PER_CODE = 100.0     # 编码每个数字 = 100 µm（"0301" → x=300, y=100）
PITCH_UM = 50.0         # 画面内 2x2 方格边长（1 大 + 3 小十字，唯一可能取法）
FIT_WARN_UM = 0.05      # 套准拟合残差警告阈值（µm）；检测噪声 ~0.01 µm
DEFAULT_FOLDER = os.path.join(_ROOT, "260910-3381-2")
# 方向自动裁决；如需手动强制，设为 ("+", "-") 之类的 (sx, sy)
#   sx=+1 图像向右=版图+x；sy=-1 图像向下=版图-y（常见非镜像情形）
FORCE_ORIENTATION = None

SLOT_NAME = ["左上M1", "右上M2", "左下M3", "右下M4"]


def decode_design(name):
    """文件名前 4 位 → 大十字设计坐标 (x_um, y_um)。"""
    code = name[:4]
    if not code.isdigit():
        raise ValueError("文件名前缀 %r 不是 4 位坐标编码" % code)
    return float(int(code[:2]) * UM_PER_CODE), float(int(code[2:]) * UM_PER_CODE)


def classify_corner(raw_path):
    """在原图上检测 + 必要时网格恢复，返回 (大十字角位 0..3, 4 中心)。
    返回 (None, None) 表示标记不全且恢复失败。"""
    gray, _ = strip_info_bar(cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE))
    accepted, rejected, blur = detect_marks(gray, verbose=False)
    pool = accepted + rejected
    diag = {}
    if len(accepted) < 4:
        rec = recover_missing(accepted, 2, 2, blur, verbose=False, diag=diag)
        if rec is None:
            return None, None
        d, best = min(((c["cx"] - rec["rx"]) ** 2 + (c["cy"] - rec["ry"]) ** 2,
                       c) for c in pool)
        rec = dict(rec)
        rec["span"] = best["span"] if d < 80 ** 2 else rec["span"]
        accepted = accepted + [rec]
    if len(accepted) != 4:
        return None, None
    rows = sorted(accepted, key=lambda c: c["ry"])
    tl, tr = sorted(rows[:2], key=lambda c: c["rx"])
    bl, br = sorted(rows[2:], key=lambda c: c["rx"])
    slots = [tl, tr, bl, br]                       # 行优先：M1..M4
    spans = [s["span"] for s in slots]
    return int(np.argmax(spans)), slots


def design_targets(big_slot, big_xy, sx, sy):
    """大十字锚定在解码坐标，其余三角按 ±PITCH 的格点展开。"""
    r_b, c_b = divmod(big_slot, 2)
    out = []
    for slot in range(4):
        r, c = divmod(slot, 2)
        out.append((big_xy[0] + sx * (c - c_b) * PITCH_UM,
                    big_xy[1] + sy * (r - r_b) * PITCH_UM))
    return out


def fit_image(points, img_h, big_slot, big_xy, sx, sy):
    """相似变换拟合 校正图像素(y向上) → 设计坐标(µm)。
    返回 dict(scale, rot_deg, ins_x, ins_y, rms_um)。"""
    src = np.array([[p["x"], img_h - p["y"]] for p in points], np.float64)
    dst = np.asarray(design_targets(big_slot, big_xy, sx, sy), np.float64)
    s_sim, R, t, resid = fit_similarity(src, dst)
    return {"scale": float(s_sim),
            "rot_deg": math.degrees(math.atan2(R[1, 0], R[0, 0])),
            "ins_x": float(t[0]), "ins_y": float(t[1]),
            "rms_um": rms_of(resid)}


def scr_block(name, tif_path, r):
    """单张图的 -IMAGE 脚本段（回答顺序：文件、插入点、比例、旋转）。"""
    return ("; %s  大十字%s  拟合残差 %.3f µm\n"
            "-IMAGE A \"%s\"\n"
            "%.3f,%.3f\n"
            "%.7f\n"
            "%.4f\n"
            % (name, SLOT_NAME[r["corner"]], r["rms_um"],
               tif_path.replace("\\", "/"),
               r["ins_x"], r["ins_y"], r["scale"], r["rot_deg"]))


def main(folder):
    out_dir = os.path.join(folder, "corrected", "cad")
    os.makedirs(out_dir, exist_ok=True)
    corrected = sorted(glob.glob(os.path.join(folder, "corrected",
                                              "*_corrected.tif")))
    if not corrected:
        sys.exit("未找到 corrected/*_corrected.tif: %s" % folder)

    rows, skipped = [], []
    for tif in corrected:
        name = os.path.basename(tif)[:-len("_corrected.tif")]
        raw_path = os.path.join(folder, name + ".tif")
        report_path = os.path.join(folder, "corrected", "diagnostics",
                                   name + "_report.json")
        if not (os.path.exists(raw_path) and os.path.exists(report_path)):
            skipped.append((name, "缺少原图或报告 JSON"))
            continue
        corner, _ = classify_corner(raw_path)
        if corner is None:
            skipped.append((name, "标记不全且恢复失败（1701/1903 一类）"))
            continue
        with open(report_path, encoding="utf-8") as fh:
            points = json.loads(fh.read())["corrected_centers"]["points"]
        if len(points) != 4:
            skipped.append((name, "报告内中心数 %d" % len(points)))
            continue
        h_img = cv2.imread(tif, cv2.IMREAD_UNCHANGED).shape[0]
        item = {"name": name, "corner": corner,
                "big_xy": decode_design(name), "points": points,
                "h": h_img}
        combos = FORCE_ORIENTATION or [(+1, -1), (-1, -1), (+1, +1), (-1, +1)]
        scored = [(fit_image(points, h_img, corner, item["big_xy"], sx, sy), (sx, sy))
                  for sx, sy in combos]
        best_fit, _ = min(scored, key=lambda e: e[0]["rms_um"] ** 2)
        rows.append({**item, **best_fit})

    if not rows:
        sys.exit("没有可处理的图像")

    # ---- 方向：自动排除镜像，报告 180° 二义性 ----
    # 4 个方向假设中，错误手性（纯镜像）的相似变换残差达间距量级，
    # 唯一被排除；剩下 (sx,sy) 与 (-sx,-sy) 两个解相差 180°，
    # 对"1 大 + 3 小"的正方星座残差同为零——这是星座的固有二义性，
    # mark 几何本身无法区分，需人工核对一次（全图共用同一结论）。
    if FORCE_ORIENTATION is None:
        totals = {}
        for sx, sy in [(+1, -1), (-1, -1), (+1, +1), (-1, +1)]:
            sq = 0.0
            for r in rows:
                f = fit_image(r["points"], r["h"], r["corner"], r["big_xy"], sx, sy)
                sq += f["rms_um"] ** 2
            totals[(sx, sy)] = sq
        best = min(totals.values())
        good = [c for c, v in totals.items() if v < best * 1e3 + 1e-9]
        if len(good) != 2:
            print("警告：零残差方向假设有 %d 个（预期 2 个），请人工核查！" % len(good))
        sx, sy = (+1, -1) if (+1, -1) in good else good[0]
        mirror_excluded = [c for c in totals if c not in good]
        if mirror_excluded:
            print("镜像手性已自动排除（%s 残差 %.0f µm 量级，正确方向 %.3f µm 量级）"
                  % (" / ".join("(%+d,%+d)" % c for c in mirror_excluded),
                     (totals[mirror_excluded[0]] / max(1, len(rows))) ** 0.5,
                     best ** 0.5))
        flip = [c for c in good if c != (sx, sy)]
        if flip:
            print("注意：存在与当前结论相差 180° 的等价解 (%+d,%+d) —— "
                  "四个 mark 无法区分这两者。贴图后请任选一张图，将画面内容"
                  "与样品/版图认知对照一次：若整体方向反了，在脚本顶部设 "
                  "FORCE_ORIENTATION = (%+d, %+d) 重跑（全图一起翻转，锚点不变）。"
                  % (flip[0][0], flip[0][1], flip[0][0], flip[0][1]))
    else:
        sx, sy = FORCE_ORIENTATION

    print("采用图像方向：图像向右 = 版图 %s，图像向下 = 版图 %s"
          % ("+x" if sx > 0 else "-x（水平镜像）",
             "-y" if sy < 0 else "+y（垂直翻转）"))
    for r in rows:   # 用统一方向重算
        r.update(fit_image(r["points"], r["h"], r["corner"],
                           r["big_xy"], sx, sy))

    bad = [r for r in rows if r["rms_um"] > FIT_WARN_UM]
    print("\n%-10s %-8s %-22s %10s %10s %10s %9s %8s" %
          ("图像", "大十字", "大十字设计坐标(µm)", "插入点X", "插入点Y",
           "比例µm/px", "残差µm", "旋转°"))
    for r in sorted(rows, key=lambda e: e["name"]):
        print("%-10s %-8s (%7.1f,%7.1f)        %10.3f %10.3f %10.7f %9.4f %8.4f%s" %
              (r["name"], SLOT_NAME[r["corner"]], r["big_xy"][0],
               r["big_xy"][1], r["ins_x"], r["ins_y"], r["scale"],
               r["rms_um"], r["rot_deg"],
               "  ←超阈值" if r["rms_um"] > FIT_WARN_UM else ""))
    for name, why in skipped:
        print("%-10s 跳过：%s" % (name, why))

    # ---- 写 CAD 输出 ----
    csv_path = os.path.join(out_dir, "cad_params.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "big_cross_slot", "big_x_um", "big_y_um",
                    "insert_x_um", "insert_y_um", "scale_um_per_px",
                    "rotation_deg_ccw", "fit_rms_um"])
        for r in sorted(rows, key=lambda e: e["name"]):
            w.writerow([r["name"], SLOT_NAME[r["corner"]], r["big_xy"][0],
                        r["big_xy"][1], "%.4f" % r["ins_x"],
                        "%.4f" % r["ins_y"], "%.7f" % r["scale"],
                        "%.4f" % r["rot_deg"], "%.4f" % r["rms_um"]])

    all_blocks = ["; ===== 全部校正图按设计坐标贴入（整幅 map）=====",
                  "FILEDIA 0"]
    for r in sorted(rows, key=lambda e: e["name"]):
        tif = os.path.abspath(os.path.join(folder, "corrected",
                                           r["name"] + "_corrected.tif"))
        block = scr_block(r["name"], tif, r)
        all_blocks.append(block)
        one = os.path.join(out_dir, "attach_%s.scr" % r["name"])
        with open(one, "w", encoding="utf-8") as fh:
            fh.write("; 在 AutoCAD 命令行输入 SCRIPT 并选择本文件\n"
                     "FILEDIA 0\n" + block + "FILEDIA 1\nZOOM E\n")
    all_blocks.append("FILEDIA 1\nZOOM E\n")
    with open(os.path.join(out_dir, "attach_all.scr"), "w",
              encoding="utf-8") as fh:
        fh.write("; 在 AutoCAD 命令行输入 SCRIPT 并选择本文件；"
                 "重复运行会重复贴图\n" + "\n".join(all_blocks))

    print("\n使用方法：在 AutoCAD 命令行输入 SCRIPT，选择 attach_<图像名>.scr"
          " 贴单张图，或 attach_all.scr 一次贴齐整幅 map；"
          "若 -IMAGE 在你的版本里不可用，按 cad_params.csv 的三列数字手动 "
          "IMAGEATTACH（插入点 / 比例 / 旋转）。")
    print("输出目录：%s" % out_dir)
    print("  cad_params.csv            每张图的插入点/比例/旋转/残差")
    print("  attach_<图像名>.scr        单张精确贴图脚本")
    print("  attach_all.scr             全部图一次贴齐（整幅 map）")
    if bad:
        print("注意：%s 残差超 %.2f µm，贴图前请核对该图的检测诊断图"
              % (", ".join(r["name"] for r in bad), FIT_WARN_UM))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FOLDER)
