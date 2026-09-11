"""Shared SE2 correction pipeline with a library-friendly API."""

from __future__ import annotations

import math
import os
import platform
from datetime import datetime, timezone
from types import SimpleNamespace

import cv2
import numpy as np

from .detectors.se2 import detect_marks, estimate_spacing, recover_missing
from .geometry import (
    assign_grid,
    build_ideal_grid,
    decompose_affine,
    fit_model,
    fit_similarity,
    rms_of,
)
from .io import load_design, load_gray, sha256_file, write_image
from .infobar import detect_info_bar, strip_info_bar
from .quality import leave_one_out, self_check
from .reporting import (
    draw_center_marks,
    print_outputs,
    write_corrected_centers_csv,
    write_diagnostics,
    write_failure_overlay,
    write_report,
)
from .warp import warp_exact_square

from . import __version__ as TOOL_VERSION

GRID_FIT_MAX_RMS = 5.0


def _nearby_rejected(rejected, predicted, radius):
    """预测位置附近的被剔除候选 —— 说明"那里有东西，但不是完整十字"。"""
    if predicted is None or not rejected:
        return []
    found = []
    for c in rejected:
        d = math.hypot(c["cx"] - predicted[0], c["cy"] - predicted[1])
        if d <= radius:
            found.append((d, c))
    found.sort(key=lambda e: e[0])
    return found


def _insufficient_marks_message(*, accepted, rejected, diag, n_rows, n_cols,
                                outdir, base, gray, detector_diag=None):
    """标记不齐全时的归因诊断：把"是不是 mark 十字本身残缺"直接答出来。"""
    n_total = n_rows * n_cols
    spacing = estimate_spacing([(c["rx"], c["ry"]) for c in accepted])
    if not spacing:
        spacing = 0.3 * float(min(gray.shape))

    lines = [
        "标记不齐全：只检出 %d 个，--grid %dx%d 要求 %d 个，"
        "网格几何预测恢复也失败了。"
        % (len(accepted), n_rows, n_cols, n_total),
        "",
    ]
    if accepted:
        lines.append("已接受的标记：")
        for c in sorted(accepted, key=lambda q: (q["ry"], q["rx"])):
            lines.append("  (%9.2f, %9.2f)   模板=%.3f 对称=%.3f 形状=%.3f"
                         % (c["rx"], c["ry"], c["score"], c["sym"],
                            c["shape"]))

    stage = diag.get("stage")
    predicted = diag.get("predicted_px")
    slot = diag.get("slot")
    gates = {g["name"]: g for g in diag.get("gates", [])}

    if stage == "gate_rejected" and predicted:
        lines.append("")
        lines.append("按其余标记的网格几何，缺的是 %s，它应当位于 "
                     "(%.1f, %.1f) px。" % (slot, predicted[0], predicted[1]))
        lines.append("该位置的局部模板搜索与几何验证结果：")
        for g in diag.get("gates", []):
            lines.append("   %-5s %.3f   （门槛 %.2f）   %s"
                         % (g["name"], g["value"], g["threshold"],
                            "通过" if g["passed"] else "未通过 ←"))
        lines.append("   对称性 %.3f（仅供参考：污染会天然破坏对称，"
                     "故不作为恢复门槛）" % diag.get("symmetry_score", 0.0))
        near = _nearby_rejected(rejected, predicted, 0.30 * spacing)
        if near:
            lines.append("")
            lines.append("该位置附近还有被图像门剔除的候选"
                         "（距预测位置 %.0f px 内）：" % (0.30 * spacing))
            for d, c in near:
                lines.append(
                    "   (%8.1f, %8.1f)  距 %.1f px   模板=%.3f 对称=%.3f 形状=%.3f"
                    % (c["cx"], c["cy"], d, c["score"], c["sym"], c["shape"]))
    elif stage == "insufficient_points":
        lines.append("")
        lines.append("可用标记少于 3 个，连“缺的是哪一个格位”都无法预测。")
    elif stage == "row_split":
        lines.append("")
        lines.append("检测点无法按 y 切成 %d 行 —— 已检出的标记分布本身就不像 "
                     "%dx%d 网格，通常是多数 mark 残缺或混入了干扰物。"
                     % (n_rows, n_rows, n_cols))
    elif stage == "no_hypothesis":
        lines.append("")
        lines.append("没有任何“缺失哪个格位”的假设能解释当前的检测分布。")

    overlay = None
    try:
        overlay = write_failure_overlay(
            outdir=outdir, base=base, gray=gray, accepted=accepted,
            rejected=rejected, predicted=predicted, slot=slot,
            spacing=spacing)
    except Exception:                                     # 绘图失败不掩盖真实错误
        overlay = None

    shape_bad = "臂角形状" in gates and not gates["臂角形状"]["passed"]
    tmpl_bad = "模板相关" in gates and not gates["模板相关"]["passed"]
    lines.append("")
    lines.append("→ 最可能的原因：这个 SEM mark 的十字形状本身不完整。")
    dd = detector_diag or {}
    if dd.get("n_dropped_by_area", 0) > 0:
        lines.append("   ⚠ 但先排除另一种可能：粗筛阶段有 %d 个亮结构因面积不足"
                     "（< %.1f px²，尺度基准 %.0f px²）被丢弃。"
                     % (dd["n_dropped_by_area"], dd.get("min_area_px2", 0.0),
                        dd.get("anchor_area_px2", 0.0)))
        lines.append("     若该格位本来有标记，那它多半是**变小了**（倍率偏低/"
                     "聚焦差/剂量不足），而不是十字残缺——请先确认成像倍率。")
    if shape_bad and not tmpl_bad:
        lines.append("   该位置的相关性是够的（确实有亮结构），但臂/角结构测试")
        lines.append("   判定它不是完整十字：典型成因是十字缺臂、被碎屑或套刻图形")
        lines.append("   覆盖、刻蚀不完整、剂量不足，或十字与旁边亮图形粘连成一体。")
    elif tmpl_bad and not shape_bad:
        lines.append("   该位置的相关性低于门槛：十字可能残缺到无法与模板对齐，")
        lines.append("   也可能该格位本来就没有标记（漏写、被剥离或被打掉）。")
    elif tmpl_bad and shape_bad:
        lines.append("   模板相关与臂角形状双双不达标：该处大概率没有可用的十字。")
    else:
        lines.append("   请对照诊断图确认该处的十字是否完整。")
    if predicted:
        lines.append("   请把原图放大到 (%.1f, %.1f) 附近，"
                     "逐个检查十字的四个臂是否都在。"
                     % (predicted[0], predicted[1]))
    else:
        lines.append("   请对照诊断图逐个确认各 mark 的十字是否完整。")
    if overlay:
        lines.append("")
        lines.append("诊断图（绿=接受  红=被剔除  橙=缺失格位的几何预测位置）已保存：")
        lines.append("  %s" % overlay)
    lines.append("")
    lines.append("注意：mark 十字残缺时，任何拟合都会把模型硬拽到错误的点上，"
                 "结果不可信；")
    lines.append("建议换器件或重拍该点，而不是靠调参绕过。")
    return "\n".join(lines)


def process_single(image_path, args, outdir=None, verbose=True):
    """单张图像完整流程；失败时抛出 RuntimeError。
    verbose=False 时静默运行（库/批量调用），完整结果看返回的 report。"""
    n_rows, n_cols = [int(v) for v in args.grid.lower().split("x")]
    outdir = outdir or args.outdir or os.path.dirname(os.path.abspath(image_path))
    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    def log(msg):
        if verbose:
            print(msg)

    # ---------- 1) 读取 + 信息栏裁切 ----------
    raw = load_gray(image_path)
    log("图像尺寸: %d x %d" % (raw.shape[1], raw.shape[0]))
    gray = raw
    info_bar = None
    infobar_path = None
    if not getattr(args, "keep_info_bar", False):
        gray, info_bar = strip_info_bar(raw)
        if info_bar is not None:
            # 裁下的条带单独存盘：参数（Mag/WD/EHT/日期/时间…）不能丢
            infobar_path = os.path.join(outdir, "diagnostics",
                                        base + "_infobar.png")
            os.makedirs(os.path.dirname(infobar_path), exist_ok=True)
            write_image(infobar_path, raw[info_bar["top"]:, :])
            log("已裁掉底部参数信息栏: y >= %d（%d 行，占图高 %.1f%%）"
                "→ %d x %d" %
                (info_bar["top"], info_bar["height"], info_bar["frac"] * 100,
                 gray.shape[1], gray.shape[0]))
    elif detect_info_bar(raw) is not None:
        log("检测到参数信息栏，但 --keep-info-bar 已指定，保持原样")

    # ---------- 2) 检测 ----------
    detector_diag = {}
    accepted, rejected, blur = detect_marks(gray, verbose=verbose,
                                            diag=detector_diag)

    # 网格不完整时：按已知网格结构预测并恢复缺失标记
    n_total = n_rows * n_cols
    recover_diag = {}
    if len(accepted) < n_total:
        rec = recover_missing(accepted, n_rows, n_cols, blur,
                              verbose=verbose, diag=recover_diag)
        if rec is not None:
            log("成功恢复: (%9.2f, %9.2f)  模板=%.3f 对称=%.3f" %
                  (rec["rx"], rec["ry"], rec["score"], rec["sym"]))
            accepted.append(rec)
    if len(accepted) < n_total:
        # 归因到"mark 十字是否残缺"，并落一张诊断图供人工核对
        raise RuntimeError(_insufficient_marks_message(
            accepted=accepted, rejected=rejected, diag=recover_diag,
            n_rows=n_rows, n_cols=n_cols, outdir=outdir, base=base,
            gray=gray, detector_diag=detector_diag))

    ordered = assign_grid(accepted, n_rows, n_cols)
    names = ["M%d" % (i + 1) for i in range(len(ordered))]

    # ---------- 2) 理想坐标 ----------
    pitch = None
    if args.design:
        ideal = load_design(args.design, names)
        ideal_source = "用户提供的设计坐标"
    else:
        ideal, (pitch, dx_med, dy_med) = build_ideal_grid(ordered, n_rows, n_cols)
        ideal_source = ("正方形网格推断（间距 p=%.1f px；实测横向间距中位数 %.1f、"
                        "纵向 %.1f。如需矩形网格或绝对尺度，请用 --design 传入"
                        "设计坐标）" % (pitch, dx_med, dy_med))
    log("理想坐标来源: %s" % ideal_source)

    # ---------- 3) 拟合（含网格几何自愈） ----------
    method_used = "affine" if args.affine else "homography"

    def _fit_all(ideal_pts, ordered_pts):
        (A_, t_), resid_ = fit_model(ideal_pts, ordered_pts, "affine")
        H_, resid_h_ = fit_model(ideal_pts, ordered_pts, "homography")
        return (A_, t_), resid_, H_, resid_h_

    def _cand_for(pt):
        for c in accepted:
            if abs(c["rx"] - pt[0]) < 1e-9 and abs(c["ry"] - pt[1]) < 1e-9:
                return c
        raise RuntimeError("内部错误：网格点 (%.1f, %.1f) 找不到对应候选"
                           % (pt[0], pt[1]))

    assigned_cands = [_cand_for(p) for p in ordered]
    (A, t), resid, H, resid_h = _fit_all(ideal, ordered)
    rms = rms_of(resid)
    rms_h = rms_of(resid_h)
    # 后续表格/诊断图/报告统一用"实际校正模型"的残差
    resid_used = resid if args.affine else resid_h
    rms_used = rms if args.affine else rms_h

    # 网格几何自愈：mark 网格"相邻等距、行列正交"是绝对的，模板分数
    # 只是证据。RMS 超限说明有入选点违反几何关系（碎屑压住真 mark 被
    # 误杀、干扰物挤占格位等）。处理：逐出"剔除后 RMS 最低"的点 →
    # 按剩余点的网格几何预测空位 → 图像局部重搜回填（回填门只看模板
    # +形状，不含对称分——污染恰好破坏对称性）→ 重拟合。至多两轮。
    repair_log = []
    while (rms_used > GRID_FIT_MAX_RMS and len(repair_log) < 2
           and n_rows * n_cols >= 4):
        loo_try = leave_one_out(ideal, ordered, method_used)
        if not loo_try["per_mark"]:
            log("网格自愈：标记数无冗余，无法定位违规点，停止自愈")
            break
        worst = min(loo_try["per_mark"], key=lambda e: e["rms_without_px"])
        wi = int(worst["id"][1:]) - 1
        log("\n网格自愈：%s 与网格几何不符（剔除后 RMS %.2f << 全体 %.2f）"
              "——逐出，按剩余点几何预测该格位并重新搜索" %
              (worst["id"], worst["rms_without_px"], rms_used))
        pool = assigned_cands[:wi] + assigned_cands[wi + 1:]
        rec = recover_missing(pool, n_rows, n_cols, blur, verbose=verbose)
        if rec is None:
            log("网格自愈失败：%s 格位在几何预测位置找不到可信标记。"
                  % worst["id"])
            break
        log("网格自愈：回填 %s ← (%.1f, %.1f)（模板 %.3f 形状 %.3f）"
              % (worst["id"], rec["rx"], rec["ry"], rec["score"],
                 rec["shape"]))
        repair_log.append({"slot": worst["id"],
                           "expelled_at": [float(ordered[wi][0]),
                                           float(ordered[wi][1])],
                           "recovered_at": [rec["rx"], rec["ry"]],
                           "template_score": rec["score"]})
        assigned_cands[wi] = rec
        ordered = [(c["rx"], c["ry"]) for c in assigned_cands]
        if not args.design:
            ideal, _ = build_ideal_grid(ordered, n_rows, n_cols)
        (A, t), resid, H, resid_h = _fit_all(ideal, ordered)
        rms = rms_of(resid)
        rms_h = rms_of(resid_h)
        resid_used = resid if args.affine else resid_h
        rms_used = rms if args.affine else rms_h
    if repair_log:
        log("网格自愈完成，当前 RMS = %.3f px" % rms_used)

    decomp = decompose_affine(A)

    log("\n===== 拟合结果（诊断模型: 全局%s；校正模型: %s）=====" %
          ("仿射" if args.affine else "单应",
           "仿射" if args.affine else
           ("分格精确单应（mark 中心严格正方形）"
            if (n_rows >= 2 and n_cols >= 2) else "单应/透视")))
    log("仿射 6 参数  RMS: %.4f px（最大 %.4f px）" %
          (rms, float(np.max(np.linalg.norm(resid, axis=1)))))
    log("单应 8 参数  RMS: %.4f px%s" %
          (rms_h, "（明显优于仿射，透视成分存在）"
           if rms_h < 0.7 * rms and args.affine else ""))
    log("仿射分解: scale_x=%.5f  scale_y=%.5f  旋转=%.4f°  正交偏差=%.4f°" %
          (decomp["scale_x"], decomp["scale_y"],
           decomp["rotation_deg"], decomp["non_orthogonal_deg"]))
    log("\n%-4s %12s %12s %12s %12s %10s" %
          ("编号", "检测x", "检测y", "理想x", "理想y", "残差(px)"))
    for i, nm in enumerate(names):
        d = ordered[i]
        m = ideal[i]
        r = float(np.linalg.norm(resid_used[i]))
        log("%-4s %12.3f %12.3f %12.3f %12.3f %10.3f" %
              (nm, d[0], d[1], m[0], m[1], r))

    # ---------- 3b) 留一交叉验证 ----------
    loo = leave_one_out(ideal, ordered, method_used)
    if loo.get("note"):
        log("留一交叉验证：%s" % loo["note"])
    elif loo["suspects"]:
        by_id = {e["id"]: e for e in loo["per_mark"]}
        for nm in loo["suspects"]:
            e = by_id[nm]
            log("留一验证警告：%s 可疑——剔除后 RMS %.4f << 全体 %.4f，"
                  "该点疑似离群（请对照残差诊断图核查）"
                  % (nm, e["rms_without_px"], loo["rms_full_px"]))
    else:
        log("留一交叉验证：无可疑标记（全体 RMS %.4f）" % loo["rms_full_px"])

    # ---------- 3c) 指派合理性校验（自愈完成后的最终裁决） ----------
    # 层 1：拟合 RMS 熔断——有冗余网格（≥5 点）下，混入干扰物会推高
    # 全局单应 RMS。层 2：相似变换校验——对 2x2 等无冗余网格这是唯一
    # 的全局几何防线（4 点拟合单应恒为零残差，层 1 形同虚设）：正确的
    # 方格指派在相似变换（旋转+等比缩放+平移）下残差极小；一旦混入
    # 冒名顶替者或行列错位组合（如平行四边形），残差达间距的百分之几十。
    if rms_used > GRID_FIT_MAX_RMS:
        hint = "网格自愈已尝试仍未通过，" if repair_log else ""
        raise RuntimeError(
            "拟合 RMS %.1f px 超过合理性上限 %.1f px——%s网格指派几乎肯定"
            "有误（干扰物以假乱真、mark 缺失/污染过重，或 --grid/--design "
            "与实际不符）。请对照上方\"接受候选\"列表逐个核查位置。"
            % (rms_used, GRID_FIT_MAX_RMS, hint))

    unit_sq = np.array([[float(c), float(r)]
                        for r in range(n_rows) for c in range(n_cols)])
    s_sim, _, _, resid_sim = fit_similarity(unit_sq,
                                            np.asarray(ordered, np.float64))
    sim_rms = rms_of(resid_sim)
    log("几何指派校验: 相似变换 RMS = %.3f px（间距 %.0f px 的 %.4f%%）"
          % (sim_rms, s_sim, sim_rms / s_sim * 100))
    if sim_rms > 0.25 * s_sim:
        raise RuntimeError(
            "网格指派不合理：理想方格→检测点的相似变换 RMS = %.1f px"
            "（间距 %.0f px 的 %.0f%%）——指派组合不构成正交等距网格，"
            "疑似混入干扰物或行列错位，请核查检测诊断图。"
            % (sim_rms, s_sim, sim_rms / s_sim * 100))

    # ---------- 4) 校正 ----------
    # 关键：拟合出的映射都是 理想坐标→检测坐标，即 输出(校正图)→输入(原图)
    # 的逆映射，必须按逆映射直接采样（WARP_INVERSE_MAP / remap）；
    # 否则校正会被反向施加（畸变越校越大）。
    # 默认模型：分格精确单应（每个 2x2 格 4 点精确解，零残差），
    # 保证 mark 中心在校正图中严格构成正方形（定位坐标系的硬性要求）。
    # 全局单应 H 仅作诊断（失真场、留一验证、自愈判据）。--affine 回退
    # 到全局仿射（平滑但 mark 中心只是最小二乘接近正方形）。
    h_img, w_img = gray.shape
    cells = None
    if args.affine:
        Mw = np.array([[A[0, 0], A[0, 1], t[0]],
                       [A[1, 0], A[1, 1], t[1]]], np.float64)
        corrected = cv2.warpAffine(gray, Mw, (w_img, h_img),
                                   flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_REPLICATE)
        method = "affine"
    elif n_rows >= 2 and n_cols >= 2:
        corrected, cells = warp_exact_square(gray, ideal, ordered,
                                             n_rows, n_cols)
        method = "exact-square"
    else:
        corrected = cv2.warpPerspective(gray, H, (w_img, h_img),
                                        flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                                        borderMode=cv2.BORDER_REPLICATE)
        method = "homography"
    out_img = os.path.join(outdir, base + "_corrected.tif")

    # ---------- 4b) 校正后自检（端到端验证） ----------
    # 在校正图上重新检测：mark 应精确落在理想格位上（分格精确模型下
    # 偏差 = 单点检测噪声水平）。未检出的格位在已知理想位置做定向
    # 验证。这是"校正后 mark 是否严格正方形"的直接检验。
    unit = "设计单位" if args.design else "px"
    span_hint = float(np.median([c["span"] for c in assigned_cands]))
    sc = self_check(corrected, ideal, n_rows, n_cols, span_hint=span_hint,
                    verbose=verbose)
    if sc["rms"] is not None:
        log("校正后自检：重新检测 %d/%d 个标记，残差 RMS = %.3f %s"
              "（拟合 RMS %.3f %s）" %
              (sc["n_detected"], sc["n_total"], sc["rms"], unit,
               rms_used, unit))
        if sc["n_detected"] < sc["n_total"]:
            log("  注意：%d 个标记在校正图上未检出（可能贴边或被移出画面）"
                  % (sc["n_total"] - sc["n_detected"]))
        for m in sc["marks"]:
            if m["residual"] is not None and m["residual"] > 1.0:
                log("  注意：%s 校正后偏离理想格位 %.2f %s"
                      "（mark 被污染/遮挡时其真实中心可能偏离设计位置，"
                      "定位时请勿以该点为锚）" % (m["id"], m["residual"], unit))
        if sc["rms"] > max(1.0, 3.0 * rms_used):
            log("  警告：校正后残差明显大于拟合残差，请检查残差诊断图！")
    else:
        log("校正后自检失败：校正图上未检测到任何标记")

    # ---------- 4c) 校正图上的红色中心标记 ----------
    # 刻意排在 4b 之后：自检必须在**干净**的校正图上重新检测，红线不能干扰它。
    # 坐标取自校正图上重新检测的结果（真实的校正后测量值）；个别未复检到
    # 的格位回退到理想格位，并在 source 字段标明来源，避免误读。
    sc_by_id = {m["id"]: m for m in sc["marks"]}
    corr_centers, corr_sources = [], []
    for i, nm in enumerate(names):
        m = sc_by_id.get(nm)
        if m and m.get("detected"):
            corr_centers.append((float(m["detected"][0]),
                                 float(m["detected"][1])))
            corr_sources.append("self-check")
        else:
            corr_centers.append((float(ideal[i][0]), float(ideal[i][1])))
            corr_sources.append("ideal-fallback")
    write_image(out_img, draw_center_marks(corrected, corr_centers,
                                           arm_px=getattr(args, "mark_arm",
                                                          None)))
    corrected_centers_csv = write_corrected_centers_csv(
        outdir=outdir, base=base, names=names, centers=corr_centers,
        sources=corr_sources, unit=unit)
    log("已在校正图上用红色小十字标出 %d 个 mark 中心" % len(names))

    # ---------- 5) 诊断输出与可追溯报告 ----------
    diagnostic_outputs = write_diagnostics(
        outdir=outdir,
        base=base,
        gray=gray,
        names=names,
        ordered=ordered,
        ideal=ideal,
        rejected=rejected,
        resid_used=resid_used,
        rms_used=rms_used,
        loo=loo,
        affine=args.affine,
    )
    report_path = os.path.join(outdir, "diagnostics", base + "_report.json")
    outputs = {
        "corrected_image": out_img,
        "centers_corrected_csv": corrected_centers_csv,
        **diagnostic_outputs,
        "report": report_path,
    }
    if infobar_path:
        outputs["info_bar_strip"] = infobar_path

    quality_warnings = []
    if sc["rms"] is None:
        quality_warnings.append("校正后未检测到可验证标记")
    if sc["n_detected"] < sc["n_total"]:
        quality_warnings.append(
            "校正后仅验证 %d/%d 个标记" %
            (sc["n_detected"], sc["n_total"]))
    if (sc["rms"] is not None
            and sc["rms"] > max(1.0, 3.0 * rms_used)):
        quality_warnings.append("校正后残差明显大于拟合残差")
    quality_status = "WARN_REVIEW" if quality_warnings else "PASS"

    report = {
        "schema_version": 2,
        "tool": {
            "name": "sem-map-corrector",
            "version": TOOL_VERSION,
            "profile": "se2-v1",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "image": os.path.abspath(image_path),
        "input_sha256": sha256_file(image_path),
        "quality_status": quality_status,
        "quality_warnings": quality_warnings,
        "method": method,
        "grid": [n_rows, n_cols],
        "detector": detector_diag,
        "info_bar": info_bar,
        "ideal_source": ideal_source,
        "ideal_pitch_px": pitch,
        "affine": {"A": A.tolist(), "t": t.tolist(),
                   "decomposed": decomp,
                   "rms_px": rms,
                   "max_residual_px": float(
                       np.max(np.linalg.norm(resid, axis=1)))},
        "homography": {"H": H.tolist(),
                       "rms_px": rms_h,
                       "max_residual_px": float(
                           np.max(np.linalg.norm(resid_h, axis=1)))},
        "used_model_rms_px": rms_used,
        "exact_square_cells": cells,
        "repair": repair_log,
        "leave_one_out": loo,
        "self_check": sc,
        "corrected_centers": {
            "unit": unit,
            "points": [
                {"id": names[i], "x": corr_centers[i][0],
                 "y": corr_centers[i][1], "source": corr_sources[i]}
                for i in range(len(names))
            ],
        },
        "marks": [
            {"id": names[i],
             "detected_px": list(ordered[i]),
             "ideal": list(map(float, ideal[i])),
             "residual_px": resid_used[i].tolist(),
             "detector": "se2-intensity",
             "recovered_from_grid_search": bool(
                 assigned_cands[i].get("recovered", False))}
            for i in range(len(names))
        ],
        "outputs": outputs,
    }
    write_report(outdir=outdir, base=base, report=report)
    print_outputs(outputs, verbose=verbose)
    return report


def correct_image(image_path, *, grid="2x2", design=None, outdir=None,
                  affine=False, verbose=True, mark_arm=None,
                  keep_info_bar=False):
    """Correct one SE2 image and return its structured report."""
    args = SimpleNamespace(
        image=os.fspath(image_path),
        batch=None,
        design=os.fspath(design) if design is not None else None,
        grid=grid,
        outdir=os.fspath(outdir) if outdir is not None else None,
        affine=bool(affine),
        mark_arm=None if mark_arm is None else int(mark_arm),
        keep_info_bar=bool(keep_info_bar),
    )
    return process_single(os.fspath(image_path), args, outdir=args.outdir,
                          verbose=verbose)
