"""合成 SE2 演示图：无实验数据时的全流程验证样本。

渲染正方形十字网格 + 水平照明梯度 + 已知几何畸变
（3° 旋转 × 各向异性缩放 1.02/0.98 + 平移 + 弱透视），
与回归测试 tests/test_regression.py 的合成场景同源。
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


def make_demo_image(path, n_rows=2, n_cols=3, img_w=640, img_h=420,
                    margin=70.0, pitch=160.0, span=60, arm=12):
    """渲染带照明梯度 + 已知几何畸变的合成 SE2 十字网格图并写盘。"""
    gradient = 45 + 25.0 * np.arange(img_w) / img_w
    img = np.repeat(gradient.astype(np.uint8)[None, :], img_h, axis=0)
    ideal = [(margin + c * pitch, margin + r * pitch)
             for r in range(n_rows) for c in range(n_cols)]
    for x, y in ideal:
        xi, yi = int(round(x)), int(round(y))
        h = arm // 2
        x0, y0 = xi - span // 2, yi - span // 2
        img[yi - h:yi + h + 1, x0:x0 + span] = 210
        img[y0:y0 + span, xi - h:xi + h + 1] = 210
    th = math.radians(3.0)
    A = (np.array([[math.cos(th), -math.sin(th)],
                   [math.sin(th), math.cos(th)]]) @ np.diag([1.02, 0.98]))
    H = np.eye(3)
    H[:2, :2] = A
    H[:2, 2] = [6.0, -4.0]
    H[2, 0], H[2, 1] = 2.0e-6, 1.5e-6
    distorted = cv2.warpPerspective(img, H, (img_w, img_h))
    if not cv2.imwrite(str(path), distorted):
        raise RuntimeError(f"演示图写入失败: {path}")
    return Path(path)
