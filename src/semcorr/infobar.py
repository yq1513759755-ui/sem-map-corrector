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

    # 白色边框会把参数栏和最底部的一行样品/窗口像素隔开。
    # 从下往上检查每个暗行连续段，不能只检查最后一段。
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    for group in reversed(groups):
        top, bottom = int(group[0]), int(group[-1])
        height = bottom - top + 1
        if bottom < h - 1 - max(2, int(round(INFO_BAR_BOTTOM_SLACK * h))):
            continue
        frac = height / float(h)
        if height < INFO_BAR_MIN_ROWS or not (INFO_BAR_MIN_FRAC <= frac
                                              <= INFO_BAR_MAX_FRAC):
            continue
        band = gray[top:bottom + 1, :]
        bg_pct = float(np.percentile(band, INFO_BAR_BG_PCT))
        if bg_pct >= INFO_BAR_BG_MAX:
            continue
        step = float(row_median[top - 1] - np.median(row_median[top:bottom + 1])) \
            if top > 0 else 0.0
        if step < INFO_BAR_MIN_STEP:
            continue
        if float(np.percentile(band, INFO_BAR_TEXT_PCT)) < INFO_BAR_TEXT_MIN:
            continue

        dark_top = top
        # 参数栏上方的近白通栏分隔线也属于参数栏；最多回收 4 行。
        # 普通样品亮区或没有边框的旧导出图不受影响。
        for _ in range(4):
            if top > 0 and float((gray[top - 1] >= 245).mean()) >= 0.90:
                top -= 1
            else:
                break
        return {"top": top, "bottom": bottom, "height": bottom - top + 1,
                "frac": (bottom - top + 1) / float(h), "step": step,
                "bg_pct": bg_pct, "dark_top": dark_top,
                "removed_rows": h - top, "removed_frac": (h - top) / float(h)}
    return None


def strip_info_bar(gray):
    """裁掉底部参数信息栏。

    返回 ``(图, info)``；未检出信息栏时原图原样返回、``info`` 为 ``None``。
    """
    info = detect_info_bar(gray)
    if info is None:
        return gray, None
    return gray[:info["top"], :], info
