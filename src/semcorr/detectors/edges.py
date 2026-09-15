"""Locate a bright cross by the intersection of its measured arm midlines.

Template matching supplies the neighbourhood and scale. Across both ends of
each arm, measure the two half-contrast edges, fit their midpoints, then
intersect the vertical and horizontal midlines. Arm lengths and brightness
need not be equal. Unsupported/occluded profiles are rejected explicitly.
"""

import numpy as np


def _profile_center(profile, origin, expected, span):
    bg, bright = np.percentile(profile, [20, 95])
    if bright - bg < 8.0:
        return None
    level = (bg + bright) * 0.5
    above = profile > level
    starts = np.flatnonzero(np.diff(above.astype(int), prepend=0) == 1)
    stops = np.flatnonzero(np.diff(above.astype(int), append=0) == -1)
    choices = []
    for left, right in zip(starts, stops):
        if left == 0 or right == len(profile) - 1:
            continue
        lo = left - 1 + (level - profile[left - 1]) / (profile[left] - profile[left - 1])
        hi = right + (profile[right] - level) / (profile[right] - profile[right + 1])
        width = hi - lo
        center = origin + (lo + hi) / 2
        if 1.2 <= width <= 0.45 * span and abs(center - expected) <= max(3., .15 * span):
            choices.append((abs(center - expected), center, width))
    return min(choices)[1:] if choices else None


def _midline(image, x, y, span):
    """Vertical arm: fit x = a * (y - y0) + b with support on both ends."""
    h, w = image.shape
    radius = max(5, int(round(span * .32)))
    x0, x1 = max(0, int(round(x)) - radius), min(w, int(round(x)) + radius + 1)
    inner, outer = max(3, int(round(span * .20))), max(5, int(round(span * .40)))
    samples = []
    for side in (-1, 1):
        for offset in range(inner, outer + 1):
            row = int(round(y)) + side * offset
            if not 0 <= row < h:
                continue
            result = _profile_center(image[row, x0:x1].astype(float), x0, x, span)
            if result is not None:
                center, width = result
                samples.append((row - y, center, width, side))
    if len(samples) < 8:
        return None
    points = np.asarray(samples)
    widths = points[:, 2]
    med_width = float(np.median(widths))
    keep = abs(widths - med_width) <= max(1.5, .25 * med_width)
    # Initial slope from all pairs of opposite-arm samples, robust to debris.
    upper, lower = points[(points[:, 3] < 0) & keep], points[(points[:, 3] > 0) & keep]
    if len(upper) < 3 or len(lower) < 3:
        return None
    slopes = (lower[:, None, 1] - upper[None, :, 1]) / (lower[:, None, 0] - upper[None, :, 0])
    a = float(np.median(slopes))
    b = float(np.median(points[keep, 1] - a * points[keep, 0]))
    for _ in range(3):
        residual = points[:, 1] - (a * points[:, 0] + b)
        mad = float(np.median(abs(residual[keep] - np.median(residual[keep]))))
        keep &= abs(residual) <= max(.6, 3 * 1.4826 * mad)
        if any(np.count_nonzero(keep & (points[:, 3] == side)) < 3 for side in (-1, 1)):
            return None
        design = np.column_stack((points[keep, 0], np.ones(np.count_nonzero(keep))))
        a, b = np.linalg.lstsq(design, points[keep, 1], rcond=None)[0]
    rms = float(np.sqrt(np.mean((points[keep, 1] - (a * points[keep, 0] + b)) ** 2)))
    if abs(a) > .30 or rms > max(.7, .08 * med_width):
        return None
    return float(a), float(b), rms, int(np.count_nonzero(keep))


def refine_arm_edges(image, x, y, span):
    """Return (x, y, evidence), or None if four-arm evidence is insufficient."""
    vertical = _midline(image, x, y, span)
    horizontal = _midline(image.T, y, x, span)
    if vertical is None or horizontal is None:
        return None
    a, b, vrms, vn = vertical
    c, d, hrms, hn = horizontal
    # x' = a*(y'-y)+b; y' = c*(x'-x)+d.
    nx = (a * (d - c * x - y) + b) / (1 - a * c)
    ny = c * (nx - x) + d
    shift = float(np.hypot(nx - x, ny - y))
    if shift > max(2., .12 * span):
        return None
    return float(nx), float(ny), {
        "method": "arm-edges", "shift_px": shift,
        "vertical_rms_px": vrms, "horizontal_rms_px": hrms,
        "vertical_samples": vn, "horizontal_samples": hn,
    }
