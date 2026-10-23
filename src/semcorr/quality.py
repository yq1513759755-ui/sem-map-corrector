"""Post-fit and end-to-end quality checks."""

import cv2
import numpy as np

from .detectors.se2 import (
    ARM_CONTRAST_MIN,
    ARM_WIDTH_RATIO,
    RECOVER_SCORE,
    cross_shape_score,
    detect_marks,
    refine_center,
)
from .geometry import fit_model, rms_of

SUSPECT_RMS_DROP = 0.70
SELF_CHECK_GATE = 0.30
# 留一验证的"无冗余"判据：剔除一个点后若剩余点数刚好等于该模型的自由度，
# 重新拟合会精确穿过剩余点，rms_without 恒为 0，此时下降判据没有信息量。
_MIN_POINTS = {"affine": 3, "homography": 4}


def leave_one_out(ideal, ordered, method):
    """留一交叉验证：每次剔除一个标记、用其余点重新拟合同一模型。
    若剔除某标记后 RMS 明显下降（< SUSPECT_RMS_DROP × 全体 RMS），
    说明该点一直在拉歪模型 → 可疑。只报告，不自动剔除。

    冗余守卫：当 n-1 恰好等于模型所需最少点数时（2x2 用仿射、5 点用单应），
    重拟合是恰定的，rms_without 恒为 0 —— 用它作判据会把**所有**点都标成可疑。
    这种情形改用留出残差（该点相对其余点拟合结果的偏差）判定，并如实记录
    判定依据。"""
    n = len(ideal)
    src = np.asarray(ideal, np.float64)
    dst = np.asarray(ordered, np.float64)
    _, resid_full = fit_model(src, dst, method)
    rms_full = rms_of(resid_full)
    min_pts = _MIN_POINTS.get(method, 4)
    if n - 1 < min_pts:
        return {"rms_full_px": rms_full, "per_mark": [], "suspects": [],
                "criterion": "not-applicable",
                "note": "仅 %d 个标记：剔除后不足 %d 个，无法拟合%s，"
                        "留一验证不适用（无冗余可校验）"
                        % (n, min_pts, "仿射" if method == "affine" else "单应")}
    entries = []
    for i in range(n):
        keep = np.arange(n) != i
        params_i, resid_i = fit_model(src[keep], dst[keep], method)
        rms_i = rms_of(resid_i)
        if method == "affine":
            A_i, t_i = params_i
            pred_i = src[i] @ A_i.T + t_i
        else:
            pred_i = cv2.perspectiveTransform(
                src[i].reshape(1, 1, 2).astype(np.float32), params_i)
            pred_i = pred_i.reshape(2).astype(np.float64)
        hold = float(np.linalg.norm(dst[i] - pred_i))
        entries.append({"id": "M%d" % (i + 1),
                        "rms_without_px": rms_i,
                        "holdout_residual_px": hold})
    # 恰定重拟合：rms_without 恒为 0，下降判据失效，改用留出残差。
    exactly_determined = (n - 1) == min_pts
    suspects = []
    if exactly_determined:
        criterion = "holdout"
        if rms_full > 0.05:
            holds = np.array([e["holdout_residual_px"] for e in entries])
            med = float(np.median(holds))
            for e in entries:
                # 明显大于其余点的留出残差才算可疑；恰定模型下全体点
                # 共享同一个留出残差，此时不会有人被误标。
                if med > 0 and e["holdout_residual_px"] > 2.0 * med:
                    suspects.append(e["id"])
    else:
        criterion = "rms-drop"
        if rms_full > 0.05:
            holds = np.array([e["holdout_residual_px"] for e in entries])
            med_hold = float(np.median(holds))
            for e in entries:
                # 两个条件同时成立才可疑：剔除它能让全体 RMS 明显下降（说明
                # 它在拉歪模型），且它自身的留出残差高于中位数（说明偏差集中
                # 在它身上）。只看前者会把被离群点带偏的相邻点一并标出。
                if (e["rms_without_px"] < SUSPECT_RMS_DROP * rms_full
                        and e["holdout_residual_px"] > med_hold):
                    suspects.append(e["id"])
    return {"rms_full_px": rms_full, "per_mark": entries,
            "suspects": suspects, "criterion": criterion,
            "exactly_determined": exactly_determined}


def self_check(corrected, ideal, n_rows, n_cols, span_hint=None, verbose=False):
    """校正后自检（端到端验证）：对校正图重新跑完整检测，把检测到的
    标记与最近理想格位一一配对，残差即"校正后标记偏离理想网格多少"。
    拟合 RMS 只说明模型贴合检测点，这里的数值才反映校正的真实效果。"""
    accepted, _, _ = detect_marks(corrected, verbose=verbose)
    ideal_arr = np.asarray(ideal, np.float64)
    if accepted:
        det = np.array([[q["rx"], q["ry"]] for q in accepted], np.float64)
    else:
        det = np.zeros((0, 2), np.float64)
    # 配对门限按理想网格间距定标
    grid = ideal_arr.reshape(n_rows, n_cols, 2)
    spac = []
    for r in range(n_rows):
        for c in range(n_cols - 1):
            spac.append(np.linalg.norm(grid[r, c + 1] - grid[r, c]))
    for c in range(n_cols):
        for r in range(n_rows - 1):
            spac.append(np.linalg.norm(grid[r + 1, c] - grid[r, c]))
    gate = SELF_CHECK_GATE * float(np.median(spac))
    # 贪心最近邻配对（距离升序，一个检测点最多认领一个格位）
    pairs = []
    for i in range(len(ideal_arr)):
        for j in range(len(det)):
            pairs.append((float(np.linalg.norm(ideal_arr[i] - det[j])), i, j))
    pairs.sort()
    used_i, used_j, match = set(), set(), {}
    for d, i, j in pairs:
        if d > gate:
            break
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        match[i] = (det[j], d, accepted[j].get("center_refinement", {}))
    # 未配对格位的定向回退：校正后每个 mark 的位置是**已知**的（理想
    # 格位），不存在指派歧义——直接在理想位置做局部模板+形状验证。
    # 被污染/粘连的 mark（如图册数字旁的十字）常通不过常规验收门，
    # 但在已知位置搜索只需较低的证据门槛即可确认。
    span_med = span_hint
    if span_med is None and accepted:
        span_med = float(np.median([q["span"] for q in accepted]))
    recovered_slots = []
    if span_med is not None:
        for i in range(len(ideal_arr)):
            if i in match:
                continue
            cand = {"cx": float(ideal_arr[i][0]),
                    "cy": float(ideal_arr[i][1]),
                    "span": span_med, "area": 0.0,
                    "arm_ratio": ARM_WIDTH_RATIO,
                    "score": None, "rx": None, "ry": None}
            x, y, score = refine_center(corrected, cand)
            shape = cross_shape_score(corrected, x, y, span_med)
            if score >= RECOVER_SCORE and shape >= ARM_CONTRAST_MIN:
                match[i] = ((x, y),
                            float(np.linalg.norm(np.array([x, y])
                                                 - ideal_arr[i])),
                            cand.get("center_refinement", {}))
                recovered_slots.append(i)
    marks = []
    res = []
    for i in range(len(ideal_arr)):
        nm = "M%d" % (i + 1)
        if i in match:
            p, d, evidence = match[i]
            marks.append({"id": nm,
                          "detected": [float(p[0]), float(p[1])],
                          "residual": float(d),
                          "center_refinement": evidence,
                          "verified_at_ideal_position": i in recovered_slots})
            res.append(d)
        else:
            marks.append({"id": nm, "detected": None, "residual": None,
                          "verified_at_ideal_position": False})
    rms = float(np.sqrt(np.mean(np.square(res)))) if res else None
    return {"n_detected": len(res), "n_total": len(ideal_arr),
            "rms": rms, "gate": gate, "marks": marks}
