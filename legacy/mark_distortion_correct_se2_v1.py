#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mark_distortion_correct.py — 旧版 SE2 实心亮十字定位 + 几何畸变校正
================================================================
用途：
    SEM / 光学显微镜图像中曝光的十字（crosshair）标记自动定位，
    与设计坐标比对后估计几何畸变（单应/透视为主，仿射可选），
    输出校正后的图像、残差诊断与像素级中心坐标。

设计约定：
    版图上的 mark 中心构成正交网格（默认按正方形网格处理；
    矩形/非均匀网格请用 --design 提供设计坐标）。
    关键原则：相邻间距相等、行/列连线正交——这一几何关系是绝对的，
    是判别候选真伪的最终裁判；图像特征（模板相关/对称性/形状）只是
    证据。两者冲突时以几何为准（见"网格几何自愈"）。mark 被污染或
    部分遮挡时其模板/对称分会下降甚至被图像门误拒，自愈流程会按
    网格几何预测其位置并重新搜索回填。

流程：
    1) 灰度 + 中值滤波去椒盐噪声
    2) Otsu 阈值 + 连通域 → 亮区候选（面积/长宽比粗筛）
    3) 每个候选：按其尺度生成合成十字模板，局部归一化互相关
       → 相关峰抛物面拟合得到亚像素中心；三重验收门剔除干扰——
       相关分数 + 90°旋转自相关（数字/噪点无四重对称）
       + 臂-角形状对比（圆形颗粒与十字相关分可达 0.6+，
       但其"臂间对角"与臂一样亮，形状分≈0）
    4) 按 y 聚行、行内按 x 排列，分配为 行×列 网格（M1..Mn，先行后列）
    5) 理想坐标：--design 设计坐标；否则正方形网格推断
       （间距 = 全部行/列间距的中位数，网格中心 = 检测点质心）
    6) 最小二乘拟合仿射 + 单应（透视）。校正默认用单应——
       与 PS Camera Raw "过 mark 中心画四条参考线拉直"同类，
       但中心是亚像素精度且全部标记参与拟合；--affine 可强制仿射
    7) 网格几何自愈：拟合 RMS 超限 ⇒ 有入选点违反网格几何（碎屑
       压住真 mark、干扰物挤占格位等）——逐出"剔除后 RMS 最低"的点，
       按剩余点的网格几何预测空位，回图像局部重搜回填，重拟合
       （至多两轮；仍超限则熔断报错拒绝出图）
    8) 留一交叉验证：每次剔除一个标记重新拟合同一模型，剔除后
       RMS 明显下降者标记为可疑点（仅报告/高亮，不自动剔除）
    9) 逆映射重采样输出校正图。默认校正模型为"分格精确单应"：
       每个 2x2 格用 4 个角点 mark 做 4 点精确解（零残差），mark 中心
       在校正图中严格构成正方形（样品定位坐标系的硬性要求；全局单应
       8 自由度拟合 6 点必有残差，做不到）。格间共享边位置连续，接缝
       在格间中线、远离 mark。--affine 回退全局仿射（平滑但仅
       最小二乘接近正方形）；全局单应 H 保留用于失真诊断。
       注意坐标含义：校正图的像素坐标 = 理想网格坐标——不带 --design
       时是正方形网格的 px 坐标（尺度任意）；--design 给出真实设计
       坐标（如 µm）时，校正图像素坐标即设计坐标，可直接读样品位置
   10) 校正后自检：对校正图重新跑完整检测（未检出的格位在已知理想
       位置做定向模板验证），报告端到端残差——即"校正后 mark 是否
       严格正方形"的直接检验

用法：
    python mark_distortion_correct.py                 # 不带参数：弹出对话框选图
    python mark_distortion_correct.py image.tif       # 默认 --grid 2x2（4 mark）
    python mark_distortion_correct.py image.tif --grid 2x3              # 6 mark 图像
    python mark_distortion_correct.py image.tif --design design.json --outdir out
    #   输出（全部默认）：校正 TIFF 在输出目录外层；JSON 报告与
    #   诊断图/中心坐标统一收进输出目录下的 diagnostics/ 子文件夹
    python mark_distortion_correct.py --batch 某目录   # 批量处理目录内所有图像，
                                                      # 默认输出到 该目录/corrected

设计坐标文件格式（design.json，键名对应网格编号，单位任意、比例正确即可）：
    {"M1": [0, 0], "M2": [500, 0], "M3": [1000, 0],
     "M4": [0, 400], "M5": [500, 400], "M6": [1000, 400]}

说明：
    - 仅提供仿射校正时要求标记数 >= 3；非线性畸变的可靠判定需要更多标记。
    - 生产流程请使用显微镜导出的无损原图（TIFF/BMP），JPG 压缩会轻微
      影响亚像素定位精度。
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==================== 可调参数 ====================
MIN_AREA = 300.0               # 连通域最小面积（像素），过滤小噪点
MAX_AREA_FRAC = 0.25           # 连通域面积占全图比例上限
BBOX_ASPECT_RANGE = (0.35, 3.0)  # 候选包围盒长宽比范围
WINDOW_FACTOR = 2.0            # 局部搜索窗口 = 标记跨度 × 该系数
TEMPLATE_SCALES = (0.80, 0.90, 1.00, 1.10, 1.25)  # 模板尺度搜索
ACCEPT_SCORE = 0.50            # 归一化相关峰接受阈值（低于则视为干扰物）
SYM_MIN = 0.72                 # 90°旋转自相关下限（十字具有四重对称，数字/噪点没有）
ARM_CONTRAST_MIN = 0.35        # 十字形状验证：臂-角亮度对比下限（圆形颗粒此值≈0）
RECOVER_SCORE = 0.55           # 网格预测恢复时的模板分数门槛（不依赖对称门）
ROBUST_REFINE_SCORE = 0.85     # 模板分低于此值的候选做污染感知重定位
GRID_FIT_MAX_RMS = 5.0         # 拟合 RMS 合理性上限：超过即判定网格指派有误，拒绝出图
ARM_WIDTH_RATIO = 0.16         # 模板臂宽/臂长：仅作回退默认值，正常由面积自动反推
SUSPECT_RMS_DROP = 0.70        # 留一验证：剔除后 RMS < 该比例×全体 RMS → 可疑
SELF_CHECK_GATE = 0.30         # 自检配对门限：检测点与理想格位距离 × 理想间距
# ================================================


def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("无法读取图像: %s" % path)
    return img


def find_candidates(gray):
    """Otsu 阈值 + 连通域，返回粗筛后的亮区候选列表。"""
    blur = cv2.medianBlur(gray, 3)
    _, mask = cv2.threshold(blur, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    h_img, w_img = gray.shape
    cands = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < MIN_AREA:
            continue
        if area > MAX_AREA_FRAC * h_img * w_img:
            continue
        aspect = float(max(w, h)) / max(1, min(w, h))
        if not (BBOX_ASPECT_RANGE[0] <= aspect <= BBOX_ASPECT_RANGE[1]):
            continue
        span_f = float(max(w, h))
        # 由连通域面积反推臂宽：十字面积 ≈ 2·span·w − w²  →  w = span − √(span²−area)
        # （模板臂宽若与真实臂宽失配，相关峰会变平并产生系统性定位偏差）
        disc = max(span_f * span_f - float(area), 0.0)
        w_est = span_f - float(np.sqrt(disc))
        arm_ratio = float(np.clip(w_est / max(span_f, 1e-6), 0.04, 0.5))
        cands.append({
            "cx": float(cents[i][0]), "cy": float(cents[i][1]),
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "span": span_f, "area": float(area), "arm_ratio": arm_ratio,
            "score": None, "rx": None, "ry": None,
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


def detect_marks(gray, verbose=True):
    """完整检测链：粗筛 → 亚像素精定位 → 对称性验证。
    返回 (accepted, rejected, blur)；blur 供缺失恢复的局部重搜使用。"""
    cands, blur = find_candidates(gray)
    if verbose:
        print("粗筛候选: %d 个" % len(cands))
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


def recover_missing(accepted, n_rows, n_cols, image):
    """网格不完整时：枚举"缺哪个格位"的所有假设。对每个假设：
    按行切分检测点并与该行剩余格位按 x 顺序配对，拟合 格点→像素 仿射，
    预测缺失位置，再在预测点局部重搜模板。均匀网格缺角时存在多个近似
    等价的仿射解释，必须用图像证据（预测点处是否真有十字）来裁决，
    而不能只看拟合 RMS。"""
    n_total = n_rows * n_cols
    if len(accepted) < 3 or len(accepted) >= n_total:
        return None
    cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    span_guess = float(np.median([q["span"] for q in accepted]))
    ratio_guess = float(np.median(
        [q.get("arm_ratio", ARM_WIDTH_RATIO) for q in accepted]))
    H, W = image.shape
    rows = _split_rows(accepted, n_rows)
    if len(rows) != n_rows:
        print("警告：缺失标记恢复失败（行切分异常）")
        return None

    best = None  # (score, cand, missing_index, fit_rms)
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
        print("  假设缺 M%d：指派RMS=%.2f px，预测 (%.1f, %.1f)，模板=%.3f" %
              (m + 1, rms, x, y, score))
        if best is None or score > best[0]:
            best = (score, cand, m, rms)

    if best is None:
        print("警告：缺失标记恢复失败（没有通过检验的格位指派）")
        return None
    score, cand, m, rms = best
    cand["sym"] = symmetry_score(image, cand["rx"], cand["ry"], span_guess)
    cand["shape"] = cross_shape_score(image, cand["rx"], cand["ry"],
                                      span_guess)
    cand["combined"] = 0.5 * score + 0.5 * max(0.0, cand["sym"])
    cand["recovered"] = True
    if score < RECOVER_SCORE or cand["shape"] < ARM_CONTRAST_MIN:
        print("警告：最佳假设（缺 M%d）预测位置模板=%.3f 形状=%.3f，"
              "未达门槛（模板 %.2f / 形状 %.2f），恢复失败——"
              "该位置可能确实没有标记" %
              (m + 1, score, cand["shape"], RECOVER_SCORE, ARM_CONTRAST_MIN))
        return None
    print("采纳假设：缺 M%d，恢复位置 (%.1f, %.1f)" % (m + 1, cand["rx"], cand["ry"]))
    return cand


def assign_grid(cands, n_rows, n_cols):
    """按 y 分行（最大间隙切分），行内按 x 排序；
    若某行候选超员，按综合分数保留最高的 n_cols 个。"""
    pts = sorted(cands, key=lambda c: c["ry"])
    ys = np.array([c["ry"] for c in pts])
    if len(pts) < n_rows * n_cols:
        raise RuntimeError(
            "合格标记数不足：期望 %d，实际 %d" % (n_rows * n_cols, len(pts)))
    # 取 y 方向最大的 (n_rows-1) 个间隙切行
    gaps = np.diff(ys)
    cut_idx = np.argsort(gaps)[-(n_rows - 1):]
    cuts = sorted([int(i) + 1 for i in cut_idx])
    row_groups = np.split(np.arange(len(pts)), cuts)
    ordered = []
    for g in row_groups:
        row = [pts[i] for i in g]
        if len(row) > n_cols:
            print("警告：某行检测到 %d 个候选，按分数保留前 %d 个" %
                  (len(row), n_cols))
            row = sorted(row, key=lambda c: c["combined"], reverse=True)[:n_cols]
        row = sorted(row, key=lambda c: c["rx"])
        if len(row) < n_cols:
            raise RuntimeError("行内标记不足：期望每行 %d 个" % n_cols)
        ordered.extend((c["rx"], c["ry"]) for c in row)
    return ordered


def build_ideal_grid(detected, n_rows, n_cols):
    """无设计坐标时：构造正方形理想网格（设计约定 mark 中心为正交方阵）。
    间距取全部行/列相邻间距的中位数；网格中心 = 检测点集质心。"""
    arr = np.asarray(detected, np.float64)
    grid = arr.reshape(n_rows, n_cols, 2)
    dxs, dys = [], []
    for r in range(n_rows):
        for c in range(n_cols - 1):
            dxs.append(grid[r, c + 1, 0] - grid[r, c, 0])
    for c in range(n_cols):
        for r in range(n_rows - 1):
            dys.append(grid[r + 1, c, 1] - grid[r, c, 1])
    pitch = float(np.median(dxs + dys))
    cx, cy = arr[:, 0].mean(), arr[:, 1].mean()
    ideal = []
    for r in range(n_rows):
        for c in range(n_cols):
            ideal.append((cx + (c - (n_cols - 1) / 2.0) * pitch,
                          cy + (r - (n_rows - 1) / 2.0) * pitch))
    return ideal, (pitch, float(np.median(dxs)), float(np.median(dys)))


def fit_affine(src, dst):
    """最小二乘仿射：dst ≈ A·src + t。返回 A, t, 残差。"""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    n = len(src)
    M = np.zeros((2 * n, 6))
    for i in range(n):
        M[2 * i] = [src[i, 0], src[i, 1], 1, 0, 0, 0]
        M[2 * i + 1] = [0, 0, 0, src[i, 0], src[i, 1], 1]
    p, _, _, _ = np.linalg.lstsq(M, dst.reshape(-1), rcond=None)
    A = np.array([[p[0], p[1]], [p[3], p[4]]])
    t = np.array([p[2], p[5]])
    fitted = src @ A.T + t
    return A, t, dst - fitted


def fit_model(src, dst, method):
    """拟合 理想→检测 的几何模型，返回 (params, resid)。
    method="affine"     → params=(A, t)
    method="homography" → params=H（3x3）
    供主流程与留一验证共用，保证两者用同一个模型定义。"""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    if method == "affine":
        if len(src) < 3:
            raise RuntimeError("仿射拟合至少需要 3 个点，实际 %d" % len(src))
        A, t, resid = fit_affine(src, dst)
        return (A, t), resid
    if len(src) < 4:
        raise RuntimeError("单应拟合至少需要 4 个点，实际 %d" % len(src))
    H, _ = cv2.findHomography(src.astype(np.float32),
                              dst.astype(np.float32), method=0)
    proj = cv2.perspectiveTransform(
        src.reshape(-1, 1, 2).astype(np.float32), H)
    resid = dst - proj.reshape(-1, 2).astype(np.float64)
    return H, resid


def fit_similarity(src, dst):
    """Umeyama 相似变换（旋转 + 各向同性缩放 + 平移，无镜像）：
    dst ≈ s·R·src + t。返回 (s, R, t, resid)。"""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    sc, dc = src - mu_s, dst - mu_d
    H = sc.T @ dc
    U, D, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    var = float(np.sum(sc * sc))
    s = float(np.trace(np.diag(D)) / var) if var > 1e-12 else 1.0
    t = mu_d - s * (R @ mu_s)
    fitted = src @ (s * R).T + t
    return s, R, t, dst - fitted


def rms_of(resid):
    """残差向量的 RMS（各点位移模长的均方根）。"""
    r = np.asarray(resid, np.float64)
    return float(np.sqrt(np.mean(np.sum(r * r, axis=1))))


def decompose_affine(A):
    """把仿射矩阵分解为可读的几何量（SEM 扫描畸变诊断用）。"""
    col1, col2 = A[:, 0], A[:, 1]
    sx = float(np.linalg.norm(col1))
    sy = float(np.linalg.norm(col2))
    rot = float(np.degrees(np.arctan2(col1[1], col1[0])))
    ang2 = float(np.degrees(np.arctan2(col2[1], col2[0])))
    shear = (rot + 90.0) - ang2          # 偏离正交的角度
    shear = ((shear + 180.0) % 360.0) - 180.0
    return {"scale_x": sx, "scale_y": sy,
            "rotation_deg": rot, "non_orthogonal_deg": -shear}


def leave_one_out(ideal, ordered, method):
    """留一交叉验证：每次剔除一个标记、用其余点重新拟合同一模型。
    若剔除某标记后 RMS 明显下降（< SUSPECT_RMS_DROP × 全体 RMS），
    说明该点一直在拉歪模型 → 可疑。只报告，不自动剔除。
    冗余守卫：剔除后点数不足以拟合模型时（单应 <4、仿射 <3）无法
    验证（2x2 网格用单应正属此情形），如实返回"不适用"。"""
    n = len(ideal)
    src = np.asarray(ideal, np.float64)
    dst = np.asarray(ordered, np.float64)
    _, resid_full = fit_model(src, dst, method)
    rms_full = rms_of(resid_full)
    min_pts = 3 if method == "affine" else 4
    if n - 1 < min_pts:
        return {"rms_full_px": rms_full, "per_mark": [], "suspects": [],
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
    suspects = []
    if rms_full > 0.05:
        for e in entries:
            if e["rms_without_px"] < SUSPECT_RMS_DROP * rms_full:
                suspects.append(e["id"])
    return {"rms_full_px": rms_full, "per_mark": entries,
            "suspects": suspects}


def warp_exact_square(gray, ideal, ordered, n_rows, n_cols):
    """分格精确映射（默认校正模型）：每个 2x2 格用 4 个角点 mark 做
    4 点单应——4 对点唯一确定单应、零残差，因此全部 mark 中心在
    校正图中精确落在理想正方形格位上（全局单应 8 自由度拟合 6 点
    必有残差，做不到这一点）。格间共享边两端点在两格的单应下映射
    一致 → 位置连续，仅导数在格间中线（远离 mark）有细微跳变。
    返回 (corrected, cells)；cells 供 JSON 报告与精确定位换算使用
    （Hc: 输出坐标→原图像素；逆矩阵即 原图像素→输出坐标）。"""
    h_img, w_img = gray.shape
    grid = np.asarray(ideal, np.float64).reshape(n_rows, n_cols, 2)
    # 格间边界 = 相邻理想列/行坐标的中点（兼容 --design 非均匀网格）
    x_mids = [float(np.mean(grid[:, c, 0] + grid[:, c + 1, 0])) / 2
              for c in range(n_cols - 1)]
    y_mids = [float(np.mean(grid[r, :, 1] + grid[r + 1, :, 1])) / 2
              for r in range(n_rows - 1)]
    X, Y = np.meshgrid(np.arange(w_img, dtype=np.float64),
                       np.arange(h_img, dtype=np.float64))
    map_x = np.zeros((h_img, w_img), np.float32)
    map_y = np.zeros((h_img, w_img), np.float32)
    cells = []
    for r in range(n_rows - 1):
        for c in range(n_cols - 1):
            i = [(r + dr) * n_cols + (c + dc)
                 for dr in (0, 1) for dc in (0, 1)]
            src = np.asarray([ideal[k] for k in i], np.float64)
            dst = np.asarray([ordered[k] for k in i], np.float64)
            Hc, _ = cv2.findHomography(src.astype(np.float32),
                                       dst.astype(np.float32), method=0)
            back = cv2.perspectiveTransform(
                src.reshape(-1, 1, 2).astype(np.float32), Hc)
            res = float(np.max(np.linalg.norm(
                back.reshape(-1, 2).astype(np.float64) - dst, axis=1)))
            x0 = 0 if c == 0 else int(round(x_mids[c - 1]))
            x1 = w_img if c == n_cols - 2 else int(round(x_mids[c]))
            y0 = 0 if r == 0 else int(round(y_mids[r - 1]))
            y1 = h_img if r == n_rows - 2 else int(round(y_mids[r]))
            sub = np.stack([X[y0:y1, x0:x1], Y[y0:y1, x0:x1],
                            np.ones_like(X[y0:y1, x0:x1])], axis=-1)
            proj = sub @ Hc.astype(np.float64).T
            proj = proj[..., :2] / proj[..., 2:3]
            map_x[y0:y1, x0:x1] = proj[..., 0]
            map_y[y0:y1, x0:x1] = proj[..., 1]
            cells.append({"row": r, "col": c,
                          "marks": ["M%d" % (k + 1) for k in i],
                          "H_out_to_in": Hc.tolist(),
                          "max_corner_residual_px": res})
    corrected = cv2.remap(gray, map_x, map_y, cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_REPLICATE)
    return corrected, cells


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
        match[i] = (det[j], d)
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
                                                 - ideal_arr[i])))
                recovered_slots.append(i)
    marks = []
    res = []
    for i in range(len(ideal_arr)):
        nm = "M%d" % (i + 1)
        if i in match:
            p, d = match[i]
            marks.append({"id": nm,
                          "detected": [float(p[0]), float(p[1])],
                          "residual": float(d),
                          "verified_at_ideal_position": i in recovered_slots})
            res.append(d)
        else:
            marks.append({"id": nm, "detected": None, "residual": None,
                          "verified_at_ideal_position": False})
    rms = float(np.sqrt(np.mean(np.square(res)))) if res else None
    return {"n_detected": len(res), "n_total": len(ideal_arr),
            "rms": rms, "gate": gate, "marks": marks}


IMAGE_EXTS = (".tif", ".tiff", ".bmp", ".png", ".jpg", ".jpeg")
OUTPUT_SUFFIXES = ("_corrected.tif", "_detection.png",
                   "_residuals.png", "_report.json",
                   "_centers.csv", "_centers.png")


def pick_image_dialog():
    """不带参数运行时弹出文件选择对话框。"""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        raise RuntimeError("未提供图像路径，且当前环境缺少 tkinter，"
                           "无法弹出选择对话框；请在命令行直接传入图像路径。")
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)   # 确保对话框不被 PyCharm 窗口挡住
    path = filedialog.askopenfilename(
        title="选择要处理的图像",
        filetypes=[("图像文件", "*.tif *.tiff *.bmp *.png *.jpg *.jpeg"),
                   ("所有文件", "*.*")])
    root.destroy()
    if not path:
        raise RuntimeError("未选择图像，已取消。")
    return path


def process_single(image_path, args, outdir=None):
    """单张图像完整流程；失败时抛出 RuntimeError。"""
    n_rows, n_cols = [int(v) for v in args.grid.lower().split("x")]
    outdir = outdir or args.outdir or os.path.dirname(os.path.abspath(image_path))
    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    # ---------- 1) 检测 ----------
    gray = load_gray(image_path)
    print("图像尺寸: %d x %d" % (gray.shape[1], gray.shape[0]))
    accepted, rejected, blur = detect_marks(gray, verbose=True)

    # 网格不完整时：按已知网格结构预测并恢复缺失标记
    n_total = n_rows * n_cols
    if len(accepted) < n_total:
        rec = recover_missing(accepted, n_rows, n_cols, blur)
        if rec is not None:
            print("成功恢复: (%9.2f, %9.2f)  模板=%.3f 对称=%.3f" %
                  (rec["rx"], rec["ry"], rec["score"], rec["sym"]))
            accepted.append(rec)
    if len(accepted) < n_total:
        raise RuntimeError(
            "标记数不足（检出 %d，--grid %sx%s 要求 %d）且无法自动恢复。"
            "若该版图的标准配置与此不符（如 6 个 mark 的 2x3），"
            "请用 --grid 行x列 指定正确网格，例如 --grid 2x3。"
            % (len(accepted), n_rows, n_cols, n_total))

    ordered = assign_grid(accepted, n_rows, n_cols)
    names = ["M%d" % (i + 1) for i in range(len(ordered))]

    # ---------- 2) 理想坐标 ----------
    pitch = None
    if args.design:
        with open(args.design, "r", encoding="utf-8") as f:
            design = json.load(f)
        ideal = [tuple(design[nm]) for nm in names]
        ideal_source = "用户提供的设计坐标"
    else:
        ideal, (pitch, dx_med, dy_med) = build_ideal_grid(ordered, n_rows, n_cols)
        ideal_source = ("正方形网格推断（间距 p=%.1f px；实测横向间距中位数 %.1f、"
                        "纵向 %.1f。如需矩形网格或绝对尺度，请用 --design 传入"
                        "设计坐标）" % (pitch, dx_med, dy_med))
    print("理想坐标来源: %s" % ideal_source)

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
            print("网格自愈：标记数无冗余，无法定位违规点，停止自愈")
            break
        worst = min(loo_try["per_mark"], key=lambda e: e["rms_without_px"])
        wi = int(worst["id"][1:]) - 1
        print("\n网格自愈：%s 与网格几何不符（剔除后 RMS %.2f << 全体 %.2f）"
              "——逐出，按剩余点几何预测该格位并重新搜索" %
              (worst["id"], worst["rms_without_px"], rms_used))
        pool = assigned_cands[:wi] + assigned_cands[wi + 1:]
        rec = recover_missing(pool, n_rows, n_cols, blur)
        if rec is None:
            print("网格自愈失败：%s 格位在几何预测位置找不到可信标记。"
                  % worst["id"])
            break
        print("网格自愈：回填 %s ← (%.1f, %.1f)（模板 %.3f 形状 %.3f）"
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
        print("网格自愈完成，当前 RMS = %.3f px" % rms_used)

    decomp = decompose_affine(A)

    print("\n===== 拟合结果（诊断模型: 全局%s；校正模型: %s）=====" %
          ("仿射" if args.affine else "单应",
           "仿射" if args.affine else
           ("分格精确单应（mark 中心严格正方形）"
            if (n_rows >= 2 and n_cols >= 2) else "单应/透视")))
    print("仿射 6 参数  RMS: %.4f px（最大 %.4f px）" %
          (rms, float(np.max(np.linalg.norm(resid, axis=1)))))
    print("单应 8 参数  RMS: %.4f px%s" %
          (rms_h, "（明显优于仿射，透视成分存在）"
           if rms_h < 0.7 * rms and args.affine else ""))
    print("仿射分解: scale_x=%.5f  scale_y=%.5f  旋转=%.4f°  正交偏差=%.4f°" %
          (decomp["scale_x"], decomp["scale_y"],
           decomp["rotation_deg"], decomp["non_orthogonal_deg"]))
    print("\n%-4s %12s %12s %12s %12s %10s" %
          ("编号", "检测x", "检测y", "理想x", "理想y", "残差(px)"))
    for i, nm in enumerate(names):
        d = ordered[i]
        m = ideal[i]
        r = float(np.linalg.norm(resid_used[i]))
        print("%-4s %12.3f %12.3f %12.3f %12.3f %10.3f" %
              (nm, d[0], d[1], m[0], m[1], r))

    # ---------- 3b) 留一交叉验证 ----------
    loo = leave_one_out(ideal, ordered, method_used)
    if loo.get("note"):
        print("留一交叉验证：%s" % loo["note"])
    elif loo["suspects"]:
        by_id = {e["id"]: e for e in loo["per_mark"]}
        for nm in loo["suspects"]:
            e = by_id[nm]
            print("留一验证警告：%s 可疑——剔除后 RMS %.4f << 全体 %.4f，"
                  "该点疑似离群（请对照残差诊断图核查）"
                  % (nm, e["rms_without_px"], loo["rms_full_px"]))
    else:
        print("留一交叉验证：无可疑标记（全体 RMS %.4f）" % loo["rms_full_px"])

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
    print("几何指派校验: 相似变换 RMS = %.3f px（间距 %.0f px 的 %.4f%%）"
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
    cv2.imwrite(out_img, corrected)

    # ---------- 4b) 校正后自检（端到端验证） ----------
    # 在校正图上重新检测：mark 应精确落在理想格位上（分格精确模型下
    # 偏差 = 单点检测噪声水平）。未检出的格位在已知理想位置做定向
    # 验证。这是"校正后 mark 是否严格正方形"的直接检验。
    unit = "设计单位" if args.design else "px"
    span_hint = float(np.median([c["span"] for c in assigned_cands]))
    sc = self_check(corrected, ideal, n_rows, n_cols, span_hint=span_hint)
    if sc["rms"] is not None:
        print("校正后自检：重新检测 %d/%d 个标记，残差 RMS = %.3f %s"
              "（拟合 RMS %.3f %s）" %
              (sc["n_detected"], sc["n_total"], sc["rms"], unit,
               rms_used, unit))
        if sc["n_detected"] < sc["n_total"]:
            print("  注意：%d 个标记在校正图上未检出（可能贴边或被移出画面）"
                  % (sc["n_total"] - sc["n_detected"]))
        for m in sc["marks"]:
            if m["residual"] is not None and m["residual"] > 1.0:
                print("  注意：%s 校正后偏离理想格位 %.2f %s"
                      "（mark 被污染/遮挡时其真实中心可能偏离设计位置，"
                      "定位时请勿以该点为锚）" % (m["id"], m["residual"], unit))
        if sc["rms"] > max(1.0, 3.0 * rms_used):
            print("  警告：校正后残差明显大于拟合残差，请检查残差诊断图！")
    else:
        print("校正后自检失败：校正图上未检测到任何标记")

    # ---------- 5) 诊断图 / 报告（默认跳过，--diagnostics 时生成） ----------
    # 全部默认输出：校正 TIFF 在外层，报告与诊断文件统一收进
    # 输出目录下的 diagnostics/ 子文件夹
    diag_dir = os.path.join(outdir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    small_scale = 1400.0 / max(w_img, h_img)
    small = cv2.resize(gray, None, fx=small_scale, fy=small_scale,
                       interpolation=cv2.INTER_AREA)
    fig1, ax1 = plt.subplots(figsize=(12, 12 * small.shape[0] / small.shape[1]))
    ax1.imshow(small, cmap="gray")
    for i, nm in enumerate(names):
        x, y = ordered[i][0] * small_scale, ordered[i][1] * small_scale
        ax1.plot(x, y, "o", mec="lime", mfc="none", ms=14, mew=1.5)
        ax1.annotate(nm, (x, y), textcoords="offset points",
                     xytext=(10, 10), color="lime", fontsize=11)
    for c in rejected:
        ax1.plot(c["cx"] * small_scale, c["cy"] * small_scale,
                 "rx", ms=10, mew=2)
    ax1.set_title("Detected (green) / rejected (red)")
    ax1.axis("off")
    fig1.tight_layout()
    out_fig1 = os.path.join(diag_dir, base + "_detection.png")
    fig1.savefig(out_fig1, dpi=150)

    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 6))
    ax2a.imshow(small, cmap="gray")
    quiver_scale = 20.0
    suspect_ids = set(loo["suspects"])
    for i, nm in enumerate(names):
        x = ordered[i][0] * small_scale
        y = ordered[i][1] * small_scale
        ax2a.plot(x, y, "o", mec="lime", mfc="none", ms=10, mew=1.2)
        ax2a.annotate(nm, (x, y), textcoords="offset points",
                      xytext=(8, 8), color="lime", fontsize=10)
        ax2a.arrow(x, y,
                   resid_used[i][0] * quiver_scale * small_scale,
                   resid_used[i][1] * quiver_scale * small_scale,
                   color="red", width=1.2, head_width=5)
        if nm in suspect_ids:
            ax2a.plot(x, y, "o", mec="darkorange", mfc="none",
                      ms=22, mew=2.2)
    ax2a.set_title("Residual vectors (x%.0f)" % quiver_scale)
    ax2a.axis("off")
    r_mag = np.linalg.norm(resid_used, axis=1)
    bar_colors = ["darkorange" if nm in suspect_ids else "steelblue"
                  for nm in names]
    ax2b.bar(names, r_mag, color=bar_colors)
    ax2b.set_ylabel("Residual (px)")
    ttl = "Per-mark residual  (RMS = %.3f px)" % rms_used
    if len(names) < (4 if args.affine else 5):
        # 无冗余（2x2）：4 点单应可精确穿过任意 4 点，残差只剩
        # float32 舍入噪声（~1e-5 px），此诊断不反映畸变/校正质量
        ttl += "  [no redundancy - round-off only, see self-check]"
    elif suspect_ids:
        ttl += "  orange = LOO-suspect"
    ax2b.set_title(ttl)
    ax2b.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    out_fig2 = os.path.join(diag_dir, base + "_residuals.png")
    fig2.savefig(out_fig2, dpi=150)

    # ---------- 6) 像素级中心标注（供 PS 手动校正 / 核对） ----------
    centers_csv = os.path.join(diag_dir, base + "_centers.csv")
    with open(centers_csv, "w", encoding="utf-8") as f:
        f.write("id,x,y\n")
        for i, nm in enumerate(names):
            f.write("%s,%.3f,%.3f\n" % (nm, ordered[i][0], ordered[i][1]))
    annot = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for i, nm in enumerate(names):
        x, y = ordered[i]
        xi, yi = int(round(x)), int(round(y))
        cv2.line(annot, (xi - 22, yi), (xi + 22, yi), (0, 255, 0), 1)
        cv2.line(annot, (xi, yi - 22), (xi, yi + 22), (0, 255, 0), 1)
        cv2.circle(annot, (xi, yi), 32, (0, 255, 0), 2)
        label = "%s (%.1f, %.1f)" % (nm, x, y)
        cv2.putText(annot, label, (xi + 40, yi - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(annot, label, (xi + 40, yi - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2, cv2.LINE_AA)
    out_annot = os.path.join(diag_dir, base + "_centers.png")
    cv2.imwrite(out_annot, annot)

    # ---------- 7) 报告 ----------
    report = {
        "image": os.path.abspath(image_path),
        "method": method,
        "grid": [n_rows, n_cols],
        "ideal_source": ideal_source,
        "ideal_pitch_px": pitch,
        "affine": {"A": A.tolist(), "t": t.tolist(),
                   "decomposed": decomp,
                   "rms_px": rms,
                   "max_residual_px": float(np.max(np.linalg.norm(resid, axis=1)))},
        "homography": {"H": H.tolist(),
                       "rms_px": rms_h,
                       "max_residual_px": float(np.max(np.linalg.norm(resid_h, axis=1)))},
        "used_model_rms_px": rms_used,
        "exact_square_cells": cells,
        "repair": repair_log,
        "leave_one_out": loo,
        "self_check": sc,
        "marks": [
            {"id": names[i],
             "detected_px": list(ordered[i]),
             "ideal": list(map(float, ideal[i])),
             "residual_px": resid_used[i].tolist()}
            for i in range(len(names))
        ],
        "outputs": {"corrected_image": out_img,
                    "detection_overlay": out_fig1,
                    "residual_plot": out_fig2,
                    "centers_csv": centers_csv,
                    "centers_annotated": out_annot},
    }
    out_json = os.path.join(diag_dir, base + "_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n输出：")
    print("  校正图   : %s" % out_img)
    print("  定位诊断 : %s" % out_fig1)
    print("  残差诊断 : %s" % out_fig2)
    print("  中心坐标 : %s" % centers_csv)
    print("  中心标注 : %s" % out_annot)
    print("  报告     : %s" % out_json)


def main():
    ap = argparse.ArgumentParser(description="十字标记定位与几何畸变校正")
    ap.add_argument("image", nargs="?", default=None,
                    help="输入图像路径（可省略，省略时弹出选择对话框）")
    ap.add_argument("--batch", default=None, metavar="目录",
                    help="批量模式：处理目录内所有图像")
    ap.add_argument("--design", default=None,
                    help="设计坐标 JSON（键 M1..Mn，行优先编号）")
    ap.add_argument("--grid", default="2x2", help="标记网格 行x列，默认 2x2"
                                                 "（4 mark 标准版图；6 mark 请显式指定 --grid 2x3）")
    ap.add_argument("--outdir", default=None, help="输出目录（默认与输入同目录）")
    ap.add_argument("--diagnostics", action="store_true",
                    help="（已默认输出全部文件；保留此开关仅为兼容旧命令）")
    ap.add_argument("--affine", action="store_true",
                    help="强制用 6 参数仿射校正（默认用单应/透视模型，"
                         "与 Camera Raw 四参考线法同类）")
    args = ap.parse_args()

    # ---------- 批量模式 ----------
    if args.batch:
        if not os.path.isdir(args.batch):
            print("错误：批量目录不存在: %s" % args.batch)
            sys.exit(1)
        outdir = args.outdir or os.path.join(args.batch, "corrected")
        files = sorted(
            f for f in os.listdir(args.batch)
            if f.lower().endswith(IMAGE_EXTS)
            and not f.lower().endswith(OUTPUT_SUFFIXES))
        if not files:
            print("错误：目录中没有可处理的图像（支持 %s）"
                  % " ".join(IMAGE_EXTS))
            sys.exit(1)
        print("批量处理 %d 张图像 → %s" % (len(files), outdir))
        ok, failed = [], []
        for i, fn in enumerate(files, 1):
            print("\n================ [%d/%d] %s ================"
                  % (i, len(files), fn))
            try:
                process_single(os.path.join(args.batch, fn), args, outdir)
                ok.append(fn)
            except (RuntimeError, FileNotFoundError) as e:
                print("失败：%s" % e)
                failed.append((fn, str(e)))
        print("\n===== 批量完成：成功 %d / 失败 %d ====="
              % (len(ok), len(failed)))
        for fn, err in failed:
            print("  [失败] %s: %s" % (fn, err))
        sys.exit(1 if failed else 0)

    # ---------- 单张模式 ----------
    if not args.image:
        try:
            args.image = pick_image_dialog()
        except RuntimeError as e:
            print("错误：" + str(e))
            sys.exit(1)
        print("已选择图像: %s" % args.image)
    try:
        process_single(args.image, args)
    except (RuntimeError, FileNotFoundError) as e:
        print("错误：" + str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
