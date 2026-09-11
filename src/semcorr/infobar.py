"""SEM 图像底部参数信息栏的检测与裁切。

多数 SEM（本项目场景为 Zeiss GeminiSEM）导出的图片底部带一条深色参数栏
（Mag / WD / EHT / I Probe / Signal A / Date / Time / Scan Speed …）。这条
栏会污染后续处理：

* Otsu 阈值被"灰噪声图像 + 纯黑栏"这对强双峰带偏，粗筛连通域整体错位；
* 栏内白色文字是又大又亮的块，会成为模板匹配的干扰候选；
* 画面不再是"纯样品区"，校正后自检的残差统计随之失去意义。

因此默认在读取后立即裁掉，并把裁下的条带单独存盘（保留实验参数备查）。

判据（四条同时成立才认定 —— 保守优先，误裁比不裁更糟）：

1. **底部存在连续的深色行**：该行近黑像素（< ``INFO_BAR_DARK_LEVEL``）占比
   超过 ``INFO_BAR_DARK_FRAC``。
   这里用**行内近黑像素占比**而不是行中位数/均值：栏内白字很密时（图越窄、
   文字占行宽比例越大）行中位数会被抬到 40 以上，用它判会把连续段打断。
   占比判据对文字密度不敏感 —— 文字行仍有 44%~70% 的像素是纯黑。
2. **条带贴底**：其下沿距画面底部不超过 ``INFO_BAR_BOTTOM_SLACK`` 比例的行，
   兼容"带窗口边缘/圆角的截图"（截图里条带下方还会留几行浅色）。
3. **高度占比合理**：``INFO_BAR_MIN_FRAC`` ~ ``INFO_BAR_MAX_FRAC``。
   上限同时挡掉"整幅图都很暗"这种把全图判成条带的退化情形。
4. **背景是近乎纯黑、上沿有陡降、且栏内有近白文字**：
   - 条带 60% 分位 < ``INFO_BAR_BG_MAX`` —— 信息栏底色是**纯黑**，
     而"样品本身的暗区"通常只是偏暗，这一条把它们区分开；
   - 上沿相对上一行有明显灰度落差（> ``INFO_BAR_MIN_STEP``）；
   - 条带内存在近白像素（参数文字占 ``INFO_BAR_TEXT_PCT`` 分位）。
"""

import numpy as np

INFO_BAR_DARK_LEVEL = 40       # 近黑判据（灰度）
INFO_BAR_DARK_FRAC = 0.25      # 行内近黑像素占比下限（对白字密度不敏感）
INFO_BAR_BG_PCT = 60           # 条带"背景"取该百分位
INFO_BAR_BG_MAX = 30           # 且必须低于此值（底色近乎纯黑）
INFO_BAR_MIN_FRAC = 0.015      # 条带高度占画面高度的下限
INFO_BAR_MAX_FRAC = 0.20       # 上限（同时挡掉"整幅偏暗"）
INFO_BAR_BOTTOM_SLACK = 0.03   # 条带下沿距底部允许的行数比例
INFO_BAR_MIN_STEP = 25.0       # 上沿相对上一行的最小灰度落差
INFO_BAR_TEXT_PCT = 99.9       # 近白像素的百分位
INFO_BAR_TEXT_MIN = 150.0      # 该百分位需超过的灰度值
INFO_BAR_MIN_ROWS = 6          # 绝对下限，防止极扁的图里误判


def detect_info_bar(gray):
    """检测底部参数信息栏。

    命中时返回 ``{"top", "bottom", "height", "frac", "step", "bg_pct"}``
    （``top`` 即应裁切到的行号，``bottom`` 为条带下沿）；未命中返回 ``None``。
    """
    if gray.ndim != 2:
        raise ValueError("detect_info_bar 需要单通道灰度图")
    h, w = gray.shape
    if h < 4 * INFO_BAR_MIN_ROWS or w < 32:
        return None

    row_median = np.median(gray, axis=1)
    dark_frac = (gray < INFO_BAR_DARK_LEVEL).mean(axis=1)
    bar_like = dark_frac > INFO_BAR_DARK_FRAC
    idx = np.nonzero(bar_like)[0]
    if idx.size == 0:
        return None

    # 自最后一条"像栏"的行往上走，取这条连续段
    bottom = int(idx[-1])
    present = set(idx.tolist())
    top = bottom
    while (top - 1) in present:
        top -= 1
    height = bottom - top + 1

    # 判据 2：贴底（下方容许少量非栏行 = 截图窗口边缘）
    if bottom < h - 1 - max(2, int(round(INFO_BAR_BOTTOM_SLACK * h))):
        return None

    # 判据 3：高度占比
    frac = height / float(h)
    if height < INFO_BAR_MIN_ROWS or not (INFO_BAR_MIN_FRAC <= frac
                                          <= INFO_BAR_MAX_FRAC):
        return None

    # 判据 4a：底色近乎纯黑（区别于"样品本身的暗区"）
    band = gray[top:bottom + 1, :]
    bg_pct = float(np.percentile(band, INFO_BAR_BG_PCT))
    if bg_pct >= INFO_BAR_BG_MAX:
        return None

    # 判据 4b：上沿陡降（拿条带整体中位行灰度作参照，避免单行受文字影响）
    step = float(row_median[top - 1] - np.median(row_median[top:bottom + 1])) \
        if top > 0 else 0.0
    if step < INFO_BAR_MIN_STEP:
        return None

    # 判据 4c：栏内存在近白的参数文字
    if float(np.percentile(band, INFO_BAR_TEXT_PCT)) < INFO_BAR_TEXT_MIN:
        return None

    return {"top": int(top), "bottom": int(bottom), "height": int(height),
            "frac": float(frac), "step": step, "bg_pct": bg_pct}


def strip_info_bar(gray):
    """裁掉底部参数信息栏。

    返回 ``(图, info)``；未检出信息栏时原图原样返回、``info`` 为 ``None``。
    """
    info = detect_info_bar(gray)
    if info is None:
        return gray, None
    return gray[:info["top"], :], info
