"""SE2 filled bright-cross detector (legacy-v1 compatible)."""

import math

import cv2
import numpy as np

from ..geometry import fit_affine

MIN_AREA_FLOOR = 30.0     # 面积下限的绝对兜底（px²）：再小就不可能是标记
MIN_AREA_RATIO = 12.0     # 面积门 = 图中最大可信连通域面积 / 该比值
MAX_AREA_FRAC = 0.25
BBOX_ASPECT_RANGE = (0.35, 3.0)
WINDOW_FACTOR = 2.0
TEMPLATE_SCALES = (0.80, 0.90, 1.00, 1.10, 1.25)
ACCEPT_SCORE = 0.50
SYM_MIN = 0.72
ARM_CONTRAST_MIN = 0.35
RECOVER_SCORE = 0.55
ROBUST_REFINE_SCORE = 0.85
ARM_WIDTH_RATIO = 0.16


def find_candidates(gray, diag=None):
    """Otsu 阈值 + 连通域，返回粗筛后的亮区候选列表。

    面积门是**尺度自适应**的，不再写死绝对像素：先收集"形态上像标记"的
    连通域（面积不超画面比例上限、长宽比在范围内），取其中面积最大者作尺度
    基准，阈值 = 基准 / MIN_AREA_RATIO，下限 MIN_AREA_FLOOR。

    依据：同一版图里 mark 尺寸是固定的一族（本项目大十字/小十字面积比约
    8–9 倍），所以"最大者 ÷ 常数"在任意倍率下都稳定落在小 mark 之下；而写死
    绝对像素则会在降倍率时把小 mark 整批丢掉 —— 那正是"标记不齐全"最常见的
    真实成因，且极易被误判成"十字形状不完整"。

    失败时把尺度基准/阈值/被丢弃数写入 `diag`（若提供），便于诊断。
    """
    blur = cv2.medianBlur(gray, 3)
    _, mask = cv2.threshold(blur, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    h_img, w_img = gray.shape
    area_cap = MAX_AREA_FRAC * h_img * w_img

    # 第一遍：只做形态筛查，收集所有"像标记"的连通域
    plausible = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area > area_cap:
            continue
        aspect = float(max(w, h)) / max(1, min(w, h))
        if not (BBOX_ASPECT_RANGE[0] <= aspect <= BBOX_ASPECT_RANGE[1]):
            continue
        plausible.append((i, int(x), int(y), int(w), int(h), float(area)))

    # 第二遍：用最大者的面积定尺度，再据此筛面积
    anchor = max((p[5] for p in plausible), default=0.0)
    min_area = (max(MIN_AREA_FLOOR, anchor / MIN_AREA_RATIO)
                if anchor > 0 else MIN_AREA_FLOOR)

    cands = []
    for i, x, y, w, h, area in plausible:
        if area < min_area:
            continue
        span_f = float(max(w, h))
        # 由连通域面积反推臂宽：十字面积 ≈ 2·span·w − w²  →  w = span − √(span²−area)
        # （模板臂宽若与真实臂宽失配，相关峰会变平并产生系统性定位偏差）
        disc = max(span_f * span_f - area, 0.0)
        w_est = span_f - float(np.sqrt(disc))
        arm_ratio = float(np.clip(w_est / max(span_f, 1e-6), 0.04, 0.5))
        cands.append({
            "cx": float(cents[i][0]), "cy": float(cents[i][1]),
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "span": span_f, "area": float(area), "arm_ratio": arm_ratio,
            "score": None, "rx": None, "ry": None,
        })
    if diag is not None:
        diag.clear()
        diag.update({
            "anchor_area_px2": anchor,
            "min_area_px2": min_area,
            "n_plausible": len(plausible),
            "n_candidates": len(cands),
            "n_dropped_by_area": len(plausible) - len(cands),
        })
    return cands, blur


def make_cross_template(span, arm_ratio=ARM_WIDTH_RATIO):
    """生成合成十字模板（零均值），尺寸取奇数。"""
    size = int(round(span))
    if size % 2 == 0:
        size += 1
    size = max(size, 9)
    t = np.zeros((size, size), np.float32)
    c = size // 2
    hw = max(1, int(round(size * arm_ratio)) // 2)
    t[c - hw: c + hw + 1, :] = 1.0    # 横臂
    t[:, c - hw: c + hw + 1] = 1.0    # 竖臂
    t -= t.mean()
    return t


def subpixel_peak(score_map, x, y):
    """3x3 抛物面拟合细化相关峰位置。"""
    H, W = score_map.shape
    if x <= 0 or y <= 0 or x >= W - 1 or y >= H - 1:
        return float(x), float(y)
    fx = score_map[y, x]
    dx = (score_map[y, x + 1] - score_map[y, x - 1])
    dxx = (2.0 * fx - score_map[y, x + 1] - score_map[y, x - 1])
    dy = (score_map[y + 1, x] - score_map[y - 1, x])
    dyy = (2.0 * fx - score_map[y + 1, x] - score_map[y - 1, x])
    ox = 0.5 * dx / dxx if abs(dxx) > 1e-9 else 0.0
    oy = 0.5 * dy / dyy if abs(dyy) > 1e-9 else 0.0
    return float(x) + float(np.clip(ox, -0.5, 0.5)), \
           float(y) + float(np.clip(oy, -0.5, 0.5))


def refine_center(image, cand):
    """在候选邻域做多尺度模板匹配，返回 (x, y, best_score)。"""
    H, W = image.shape
    win = int(cand["span"] * WINDOW_FACTOR)
    x0 = int(max(0, cand["cx"] - win / 2))
    y0 = int(max(0, cand["cy"] - win / 2))
    x1 = int(min(W, cand["cx"] + win / 2))
    y1 = int(min(H, cand["cy"] + win / 2))
    roi = image[y0:y1, x0:x1]
    if roi.shape[0] < 16 or roi.shape[1] < 16:
        return cand["cx"], cand["cy"], 0.0
    roi = cv2.medianBlur(roi.astype(np.uint8), 3).astype(np.float32)
    ratio = cand.get("arm_ratio", ARM_WIDTH_RATIO)

    best = (cand["cx"], cand["cy"], -1.0)
    best_span = cand["span"]
    for s in TEMPLATE_SCALES:
        tmpl = make_cross_template(cand["span"] * s, arm_ratio=ratio)
        if tmpl.shape[0] >= roi.shape[0] or tmpl.shape[1] >= roi.shape[1]:
            continue
        score = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(score)
        if max_val > best[2]:
            px, py = max_loc
            sx, sy = subpixel_peak(score, px, py)
            half = (tmpl.shape[1] // 2, tmpl.shape[0] // 2)
            best = (x0 + sx + half[0], y0 + sy + half[1], float(max_val))
            best_span = cand["span"] * s
    # 模板分低 → 疑似数字/碎屑干扰主导了评分窗口：用污染感知重定位
    # 重新求中心（剔除污染像素后只剩十字自身的干净结构参与评分）
    if 0.0 < best[2] < ROBUST_REFINE_SCORE:
        rx, ry, _ = robust_refine(image, best[0], best[1],
                                  best_span, ratio)
        best = (rx, ry, best[2])
    return best


def robust_refine(image, x, y, span, arm_ratio, search=5, iters=3):
    """污染感知重定位：数字/碎屑压在 mark 旁时，普通 matchTemplate
    的评分窗口被这些又大又亮的结构主导，相关峰被拉偏。这里先按当前
    中心渲染同尺寸十字模板，把"模板为背景、图像却亮"（数字笔画）与
    "模板为臂、图像却暗"（遮挡物）的像素标记为污染并从评分中剔除，
    只用干净像素在 ±search px 内做加权 NCC 网格搜索 + 抛物面亚像素。
    污染掩膜依赖中心位置，故交替迭代"精化→重算掩膜"至收敛，消除对
    初始猜测的依赖。干净 mark 无像素可剔，结果与普通定位一致。
    返回 (x, y, score)。"""
    s = int(round(span))
    if s % 2 == 0:
        s += 1
    s = max(s, 9)
    hw = max(1, int(round(s * arm_ratio)) // 2)
    t = np.zeros((s, s), np.float64)
    t[s // 2 - hw: s // 2 + hw + 1, :] = 1.0
    t[:, s // 2 - hw: s // 2 + hw + 1] = 1.0
    half = s // 2
    H, W = image.shape
    t_mean = float(t.mean())
    tc = t - t_mean
    t_norm = float(np.sqrt(np.sum(tc * tc)))
    n_g = 2 * search + 1

    def _pass(cx, cy):
        xi, yi = int(round(cx)), int(round(cy))
        if (xi - half - search < 0 or yi - half - search < 0
                or xi + half + search >= W or yi + half + search >= H):
            return cx, cy, 0.0
        win = image[yi - half - search: yi + half + search + 1,
                    xi - half - search: xi + half + search + 1].astype(np.float64)
        peak = float(win.max())
        if peak < 1e-9:
            return cx, cy, 0.0
        # 污染掩膜（窗口中心处与模板对齐）：
        # 亮污染 = 模板背景处图像 > 0.45×峰值；暗污染 = 模板臂处图像 < 0.45×峰值
        sl = np.s_[search: search + s, search: search + s]
        w = ~(((t < 0.5) & (win[sl] > 0.45 * peak))
              | ((t > 0.5) & (win[sl] < 0.45 * peak)))
        w = w.astype(np.float64)
        w_sum = float(np.sum(w))
        if w_sum < 0.2 * s * s:
            return cx, cy, 0.0            # 干净像素过少，放弃重定位
        tm = float(np.sum(w * t) / w_sum)
        tc_w = (t - tm) * w
        t_norm_w = float(np.sqrt(np.sum(tc_w * tc_w)))
        if t_norm_w < 1e-9:
            return cx, cy, 0.0
        score_map = np.zeros((n_g, n_g), np.float64)
        for dy in range(n_g):
            for dx in range(n_g):
                P = win[dy: dy + s, dx: dx + s]
                p_mean = float(np.sum(w * P) / w_sum)
                pc = (P - p_mean) * w
                denom = float(np.sqrt(np.sum(pc * pc))) * t_norm_w
                score_map[dy, dx] = (float(np.sum(pc * tc_w)) / denom
                                     if denom > 1e-9 else 0.0)
        ix = int(np.argmax(score_map)) % n_g
        iy = int(np.argmax(score_map)) // n_g
        sx, sy = subpixel_peak(score_map, ix, iy)
        sx = float(np.clip(sx - search, -search, search))
        sy = float(np.clip(sy - search, -search, search))
        # 网格以取整后的 (xi,yi) 为基准，偏移必须加回整数中心
        return xi + sx, yi + sy, float(score_map[iy, ix])

    cx, cy = float(x), float(y)
    score = 0.0
    for _ in range(iters):
        nx, ny, score = _pass(cx, cy)
        moved = math.hypot(nx - cx, ny - cy)
        if moved > 1.5:
            break            # 单轮跳变过大：掩膜与位置正反馈发散，保留上一轮
        cx, cy = nx, ny
        if moved < 0.02:
            break
    return cx, cy, score


def symmetry_score(gray, x, y, span):
    """90° 旋转自相关：十字标记在任意面内旋转下都是四重对称的，
    数字/斑点则不是。返回 [-1,1]，越接近 1 越像十字。"""
    L = max(16, int(round(span * 1.6)))
    if L % 2 == 1:
        L += 1
    half = L // 2
    pad = half + 2
    padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad,
                                cv2.BORDER_REFLECT_101)
    xi, yi = int(round(x)) + pad, int(round(y)) + pad
    patch = padded[yi - half: yi + half, xi - half: xi + half].astype(np.float64)
    if patch.size == 0:
        return 0.0
    rot = np.rot90(patch)
    p = patch - patch.mean()
    r = rot - rot.mean()
    denom = np.sqrt(np.sum(p * p) * np.sum(r * r))
    if denom < 1e-9:
        return 0.0
    return float(np.sum(p * r) / denom)


def cross_shape_score(gray, x, y, span):
    """十字形状验证：臂上采样点应亮、臂间对角采样点应暗。
    返回 (臂亮度中位数 − 角点亮度中位数) / (patch 峰值 − 角点亮度中位数)，
    真十字 ≈ 1；圆形颗粒/实心碎屑的"角点"与臂一样亮，得分 ≈ 0。
    与旋转对称门互补——圆斑天然四重对称、且与十字模板相关分可达 0.6+，
    只有臂/角结构测试能把它筛掉。"""
    L = max(16, int(round(span * 1.6)))
    half = L // 2
    pad = half + 2
    padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad,
                                cv2.BORDER_REFLECT_101)
    xi, yi = int(round(x)) + pad, int(round(y)) + pad
    patch = padded[yi - half: yi + half + 1,
                   xi - half: xi + half + 1].astype(np.float64)
    h, w = patch.shape[0] // 2, patch.shape[1] // 2
    o = max(1, int(round(span * 0.30)))   # 采样偏移：臂中段/空象限对角
    arm_pts = [(h, w - o), (h, w + o), (h - o, w), (h + o, w)]
    corner_pts = [(h - o, w - o), (h - o, w + o),
                  (h + o, w - o), (h + o, w + o)]
    if any(not (0 <= r < patch.shape[0] and 0 <= c < patch.shape[1])
           for r, c in arm_pts + corner_pts):
        return 0.0
    arm = float(np.median([patch[r, c] for r, c in arm_pts]))
    corner = float(np.median([patch[r, c] for r, c in corner_pts]))
    denom = float(patch.max()) - corner
    if denom < 1e-9:
        return 0.0
    return (arm - corner) / denom


def detect_marks(gray, verbose=True, diag=None):
    """完整检测链：粗筛 → 亚像素精定位 → 对称性验证。
    返回 (accepted, rejected, blur)；blur 供缺失恢复的局部重搜使用。
    diag 非空时，写入粗筛的尺度基准/面积门/丢弃数（供报告与诊断）。"""
    cands, blur = find_candidates(gray, diag=diag)
    if verbose:
        print("粗筛候选: %d 个" % len(cands))
        if diag:
            print("  面积门: 尺度基准 %.0f px² ÷ %.0f = 阈值 %.1f px²；"
                  "形态合格 %d 个，其中因面积不足丢弃 %d 个"
                  % (diag.get("anchor_area_px2", 0.0), MIN_AREA_RATIO,
                     diag.get("min_area_px2", 0.0),
                     diag.get("n_plausible", 0),
                     diag.get("n_dropped_by_area", 0)))
    accepted, rejected = [], []
    for c in cands:
        x, y, score = refine_center(blur, c)
        c["rx"], c["ry"], c["score"] = x, y, score
        sym = symmetry_score(blur, x, y, c["span"])
        c["sym"] = sym
        shape = cross_shape_score(blur, x, y, c["span"])
        c["shape"] = shape
        c["combined"] = 0.5 * score + 0.5 * max(0.0, sym)
        if (score >= ACCEPT_SCORE and sym >= SYM_MIN
                and shape >= ARM_CONTRAST_MIN):
            accepted.append(c)
        else:
            rejected.append(c)
    if verbose:
        print("通过模板+对称性+形状验证: %d 个；剔除干扰: %d 个" %
              (len(accepted), len(rejected)))
        for c in rejected:
            print("  剔除 @ (%.0f, %.0f)  模板=%.3f 对称=%.3f 形状=%.3f" %
                  (c["cx"], c["cy"], c["score"], c["sym"], c["shape"]))
        print("接受候选:")
        for c in sorted(accepted, key=lambda q: (q["ry"], q["rx"])):
            print("  (%9.2f, %9.2f)  模板=%.3f 对称=%.3f 形状=%.3f" %
                  (c["rx"], c["ry"], c["score"], c["sym"], c["shape"]))
    return accepted, rejected, blur


def _split_rows(cands, n_rows):
    """按 y 最大间隙把候选切成 n_rows 行，每行按 x 排序返回。"""
    pts = sorted(cands, key=lambda c: c["ry"])
    if n_rows <= 1:
        return [sorted(pts, key=lambda c: c["rx"])]
    ys = np.array([c["ry"] for c in pts])
    gaps = np.diff(ys)
    cut_idx = np.argsort(gaps)[-(n_rows - 1):]
    cuts = sorted([int(i) + 1 for i in cut_idx])
    return [sorted([pts[i] for i in g], key=lambda c: c["rx"])
            for g in np.split(np.arange(len(pts)), cuts)]


def estimate_spacing(points):
    """用标记点两两最小距离估计网格间距（正方形网格下即 pitch）。
    样本不足时返回 None，供诊断信息做尺度参照。"""
    if len(points) < 2:
        return None
    arr = np.asarray(points, np.float64)
    best = None
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            d = float(np.linalg.norm(arr[i] - arr[j]))
            if best is None or d < best:
                best = d
    return best


def _diag_fail(diag, stage, reason, **extra):
    """把恢复失败的结构化原因写回调用方的 diag 字典（若提供）。"""
    if diag is not None:
        diag.clear()
        diag.update({"stage": stage, "reason": reason})
        diag.update(extra)
    return None


def recover_missing(accepted, n_rows, n_cols, image, verbose=True, diag=None):
    """网格不完整时：枚举"缺哪个格位"的所有假设。对每个假设：
    按行切分检测点并与该行剩余格位按 x 顺序配对，拟合 格点→像素 仿射，
    预测缺失位置，再在预测点局部重搜模板。均匀网格缺角时存在多个近似
    等价的仿射解释，必须用图像证据（预测点处是否真有十字）来裁决，
    而不能只看拟合 RMS。

    失败时返回 None，并把结构化的失败原因写入 `diag`（若提供）：
    stage ∈ {insufficient_points, row_split, no_hypothesis, gate_rejected}，
    其中 gate_rejected 附带缺失格位、预测位置、逐项门槛判定 —— 供调用方
    给出"mark 十字本身是否残缺"的判断依据。"""
    n_total = n_rows * n_cols
    if len(accepted) < 3 or len(accepted) >= n_total:
        return _diag_fail(
            diag, "insufficient_points",
            "可用标记少于 3 个，无法用网格几何预测缺失位置",
            n_detected=len(accepted), n_total=n_total)
    cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    span_guess = float(np.median([q["span"] for q in accepted]))
    ratio_guess = float(np.median(
        [q.get("arm_ratio", ARM_WIDTH_RATIO) for q in accepted]))
    H, W = image.shape
    rows = _split_rows(accepted, n_rows)
    if len(rows) != n_rows:
        return _diag_fail(
            diag, "row_split",
            "检测点无法按 y 切分为 %d 行，无法定位缺失格位" % n_rows,
            n_detected=len(accepted), n_total=n_total)

    best = None  # (score, cand, missing_index, fit_rms)
    hypotheses = []
    for m in range(n_total):
        r_m, c_m = cells[m]
        # 该假设下每行应剩哪些列；行数/行内点数必须与检测一致
        cols_per_row = {rr: sorted(cc for (r2, cc) in cells
                                   if r2 == rr and (r2, cc) != (r_m, c_m))
                        for rr in range(n_rows)}
        if any(len(rows[rr]) != len(cols_per_row[rr]) for rr in range(n_rows)):
            continue
        src, dst = [], []
        for rr in range(n_rows):
            for cc, p in zip(cols_per_row[rr], rows[rr]):
                src.append([float(cc), float(rr)])
                dst.append([p["rx"], p["ry"]])
        A, t, resid = fit_affine(np.array(src), np.array(dst))
        rms = float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1))))
        if rms > 30.0:
            continue
        pred = A @ np.array([float(c_m), float(r_m)]) + t
        if not (0 <= pred[0] < W and 0 <= pred[1] < H):
            continue
        cand = {"cx": float(pred[0]), "cy": float(pred[1]),
                "span": span_guess, "area": 0.0, "arm_ratio": ratio_guess,
                "score": None, "rx": None, "ry": None}
        x, y, score = refine_center(image, cand)
        cand["rx"], cand["ry"], cand["score"] = x, y, score
        # 三个分数在选优前一次算齐：选优仍只看模板分（与旧行为一致），
        # 但失败时可以把完整证据链交给调用方判断"是不是十字残缺"。
        cand["sym"] = symmetry_score(image, x, y, span_guess)
        cand["shape"] = cross_shape_score(image, x, y, span_guess)
        cand["combined"] = 0.5 * score + 0.5 * max(0.0, cand["sym"])
        if verbose:
            print("  假设缺 M%d：指派RMS=%.2f px，预测 (%.1f, %.1f)，"
                  "模板=%.3f 对称=%.3f 形状=%.3f" %
                  (m + 1, rms, x, y, score, cand["sym"], cand["shape"]))
        hypotheses.append({"slot": "M%d" % (m + 1), "fit_rms_px": rms,
                           "predicted_px": [x, y],
                           "template_score": float(score),
                           "symmetry_score": float(cand["sym"]),
                           "shape_score": float(cand["shape"])})
        if best is None or score > best[0]:
            best = (score, cand, m, rms)

    if best is None:
        return _diag_fail(
            diag, "no_hypothesis",
            "没有任何“缺失哪个格位”的假设能解释当前的检测分布",
            n_detected=len(accepted), n_total=n_total)

    score, cand, m, rms = best
    cand["recovered"] = True
    gates = [
        {"name": "模板相关", "value": float(score),
         "threshold": RECOVER_SCORE, "passed": bool(score >= RECOVER_SCORE)},
        {"name": "臂角形状", "value": float(cand["shape"]),
         "threshold": ARM_CONTRAST_MIN,
         "passed": bool(cand["shape"] >= ARM_CONTRAST_MIN)},
    ]
    if diag is not None:
        diag.clear()
        diag.update({
            "stage": "recovered",
            "n_detected": len(accepted), "n_total": n_total,
            "slot": "M%d" % (m + 1), "fit_rms_px": rms,
            "predicted_px": [cand["rx"], cand["ry"]],
            "template_score": float(score),
            "symmetry_score": float(cand["sym"]),
            "shape_score": float(cand["shape"]),
            "gates": gates, "hypotheses": hypotheses,
        })
    if not all(g["passed"] for g in gates):
        if diag is not None:
            diag["stage"] = "gate_rejected"
            diag["reason"] = "预测位置未通过 %s" % "、".join(
                g["name"] for g in gates if not g["passed"])
        if verbose:
            print("警告：最佳假设（缺 M%d）预测位置模板=%.3f 形状=%.3f，"
                  "未达门槛（模板 %.2f / 形状 %.2f），恢复失败——"
                  "该位置可能确实没有标记" %
                  (m + 1, score, cand["shape"], RECOVER_SCORE, ARM_CONTRAST_MIN))
        return None
    if verbose:
        print("采纳假设：缺 M%d，恢复位置 (%.1f, %.1f)" % (m + 1, cand["rx"], cand["ry"]))
    return cand
