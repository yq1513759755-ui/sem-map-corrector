"""Grid assignment and geometric model fitting."""

import math

import cv2
import numpy as np


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
