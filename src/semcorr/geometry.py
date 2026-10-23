"""Grid assignment and geometric model fitting."""

import cv2
import numpy as np


def _row_major_order(px, n_rows, n_cols, row_angle_deg):
    """Order points row-major for one candidate row direction.

    Points are rotated by ``-row_angle_deg`` so rows become horizontal, then
    sorted by the row coordinate and by x within each row. Returns the
    similarity-fit residual of the resulting ordering, the fitted in-plane
    rotation (degrees, near 0 when the grid is axis aligned) and the order.
    """
    t = np.radians(row_angle_deg)
    c, s = np.cos(t), np.sin(t)
    qx = c * px[:, 0] + s * px[:, 1]
    qy = -s * px[:, 0] + c * px[:, 1]
    order = np.lexsort((qx, qy))
    rows = [order[k * n_cols:(k + 1) * n_cols] for k in range(n_rows)]
    rows = [row[np.argsort(qx[row])] for row in rows]
    linear = [int(i) for row in rows for i in row]
    unit = np.array([[float(col), float(row)]
                     for row in range(n_rows) for col in range(n_cols)])
    _, rot, _, resid = fit_similarity(unit, px[linear])
    rms = float(np.sqrt(np.mean(np.sum(resid * resid, axis=1))))
    fitted = float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))
    return rms, fitted, linear


def assign_grid(cands, n_rows, n_cols):
    """把候选指派为行优先的 M1..Mn，允许网格在画面内旋转。

    旧实现假设"y 方向最大的间隙就是行间间隙"，图像一旦旋转（2×3 约 20°、
    2×2 约 28°）就会把同一行切成两行，并误报成"行内标记不足 / 十字残缺"。
    现在改为：候选点两两位移的方向即网格轴方向，把这些方向对 90° 折叠后
    逐一代入，选相似变换残差最小、且行方向最接近水平的那个指派。对无旋转
    的 2×2 / 2×3 图像，结果与旧实现一致。"""
    n_total = n_rows * n_cols
    pts = list(cands)
    if len(pts) < n_total:
        raise RuntimeError(
            "合格标记数不足：期望 %d，实际 %d" % (n_total, len(pts)))
    if len(pts) > n_total:
        pts = sorted(pts, key=lambda c: c.get("combined", 0.0),
                     reverse=True)[:n_total]
        print("警告：检测到 %d 个候选，多于网格 %d 个，按综合分数保留前 %d 个"
              % (len(cands), n_total, n_total))
    px = np.asarray([[c["rx"], c["ry"]] for c in pts], np.float64)

    if n_rows == 1 or n_cols == 1:
        # 单行/单列：沿主轴排序（正常流程会拒绝这种网格，这里保持函数可用）。
        centred = px - px.mean(axis=0)
        if np.allclose(centred, 0.0):
            linear = list(range(len(pts)))
        else:
            _, _, vt = np.linalg.svd(centred, full_matrices=False)
            linear = [int(i) for i in np.argsort(centred @ vt[0])]
        return [(float(pts[i]["rx"]), float(pts[i]["ry"])) for i in linear]

    # 候选行方向 = 所有两两位移的角度，折叠到 (-45°, 45°]（垂直轴合并）。
    angles = []
    for i in range(len(px)):
        for j in range(i + 1, len(px)):
            d = px[j] - px[i]
            if float(np.hypot(d[0], d[1])) < 1e-9:
                continue
            a = float(np.degrees(np.arctan2(d[1], d[0])))
            angles.append(((a + 45.0) % 90.0) - 45.0)
    candidates = sorted({round(a, 3) for a in angles})
    if not candidates:
        raise RuntimeError("网格指派失败：候选点重合，无法确定行方向")

    scored = []
    for theta in candidates:
        # 折叠后的角度只代表"轴"，行方向可以是它本身或加 90°。
        for angle in (theta, theta + 90.0):
            rms, fitted, linear = _row_major_order(px, n_rows, n_cols, angle)
            scored.append((rms, abs(fitted), linear))
    rms_min = min(item[0] for item in scored)
    # 正方形网格（2×2、3×3）对 90° 旋转对称，多个指派的相似变换残差同为 0。
    # 此时按**拟合出的实际旋转角**绝对值最小者裁决，即保持网格尽量轴对齐
    # ——与旧实现的编号约定一致。只看"试算角度"会在旋转接近 45° 时选错。
    tolerance = 0.01
    _, _, best = min((item for item in scored if item[0] <= rms_min + tolerance),
                     key=lambda item: (item[1], item[0]))
    return [(float(pts[i]["rx"]), float(pts[i]["ry"])) for i in best]


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
