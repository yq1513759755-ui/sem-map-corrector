"""Image resampling and exact-square grid warp."""

import cv2
import numpy as np

from .geometry import fit_model


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
