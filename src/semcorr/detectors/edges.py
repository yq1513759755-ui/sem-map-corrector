"""Locate a bright cross by the intersection of its measured arm midlines.

Template matching supplies the neighbourhood and scale. Across both ends of
each arm, measure the two half-contrast edges, fit their midpoints, then
intersect the vertical and horizontal midlines. Arm lengths and brightness
need not be equal. Unsupported/occluded profiles are rejected explicitly.

Sub-pixel accuracy: each half-contrast edge uses a 3-point local model.
Near-linear flanks (the usual case at half-max of a smooth SEM edge) take a
3-point least-squares line — less noise-sensitive than 2-point interpolation.
Clearly curved / asymmetric flanks take a parabolic level crossing instead.
Midline coefficients come from a contrast- and width-weighted least squares
fit so one contaminated or abnormally wide profile cannot drag the axis.
"""

import numpy as np

# Soft width gate: profiles far from the median arm width contribute little.
_WIDTH_SOFT_SIGMA = 0.25
_WIDTH_HARD_SIGMA = 1.5
# Contrast scale (gray levels) above which a profile is fully trusted.
_CONTRAST_TRUST = 32.0
# |second difference| / contrast above this means the flank is really curved
# and a parabolic root is used instead of the 3-point line.
_CURVATURE_REL = 0.04


def _parabolic_crossing(p0, p1, p2, x_mid, level):
    """Sub-pixel x where the quadratic through p0,p1,p2 at x_mid-1..+1 equals level.

    Coefficients use the standard 3-point form with the middle sample at t=0.
    Returns None when the root leaves the sampled span.
    """
    g0, g1, g2 = float(p0) - level, float(p1) - level, float(p2) - level
    a = 0.5 * (g0 - 2.0 * g1 + g2)
    b = 0.5 * (g2 - g0)
    c = g1
    if abs(a) < 1e-9:
        if abs(b) < 1e-12:
            return None
        t = -c / b
    else:
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return None
        sqrt_d = float(np.sqrt(disc))
        roots = ((-b + sqrt_d) / (2.0 * a), (-b - sqrt_d) / (2.0 * a))
        t = min(roots, key=lambda r: abs(r))
    if not (-1.0 <= t <= 1.0):
        return None
    return float(x_mid) + float(t)


def _linear3_crossing(p0, p1, p2, x_mid, level):
    """3-point least-squares line through p0,p1,p2 at x_mid-1..+1, solve for level."""
    # t in {-1, 0, 1}: slope a = (p2 - p0) / 2, intercept b = mean(p).
    a = 0.5 * (float(p2) - float(p0))
    b = (float(p0) + float(p1) + float(p2)) / 3.0
    if abs(a) < 1e-12:
        return None
    t = (level - b) / a
    if not (-1.0 <= t <= 1.0):
        return None
    return float(x_mid) + float(t)


def _edge_crossing(profile, i, level, rising):
    """Sub-pixel crossing of ``level`` between samples i and i+1.

    ``rising`` True: profile goes from below to above (i last below).
    False: falling (i last above). Crossing lies in (i, i+1].

    Sharp (1-sample) steps use 2-point linear interpolation — a 3-point
    line or parabola is biased when the flank is almost a jump. Gradual
    SEM flanks use a 3-point least-squares line, or a parabolic root when
    that flank is clearly curved/asymmetric.
    """
    n = len(profile)
    lo, hi = float(profile[i]), float(profile[i + 1])
    den = hi - lo
    linear2 = None
    if abs(den) >= 1e-12:
        linear2 = i + (level - lo) / den

    window = profile[max(0, i - 2):min(n, i + 4)]
    local_span = float(np.max(window) - np.min(window)) if window.size else 0.0
    jump = abs(den)
    # Almost all of the flank happens in this one pixel pair → hard step.
    if local_span <= 1e-9 or jump >= 0.55 * local_span:
        return linear2 if linear2 is not None and i - 1e-9 <= linear2 <= i + 1 + 1e-9 else None

    contrast = max(local_span, 1.0)
    candidates = []
    for center in (i, i + 1):
        if not (0 < center < n - 1):
            continue
        p0, p1, p2 = float(profile[center - 1]), float(profile[center]), float(profile[center + 1])
        curv = abs(0.5 * (p0 - 2.0 * p1 + p2))
        if curv > _CURVATURE_REL * contrast:
            x = _parabolic_crossing(p0, p1, p2, center, level)
            if x is not None:
                candidates.append(x)
        x = _linear3_crossing(p0, p1, p2, center, level)
        if x is not None:
            candidates.append(x)
    if linear2 is not None and i - 1e-9 <= linear2 <= i + 1 + 1e-9:
        candidates.append(float(linear2))

    best, best_dist = None, None
    for x in candidates:
        if not (i - 1e-9 <= x <= i + 1 + 1e-9):
            continue
        dist = abs(x - (i + 0.5))
        if best_dist is None or dist < best_dist:
            best, best_dist = x, dist
    return best


def _profile_center(profile, origin, expected, span):
    bg, bright = np.percentile(profile, [20, 95])
    contrast = float(bright - bg)
    if contrast < 8.0:
        return None
    level = (bg + bright) * 0.5
    above = profile > level
    starts = np.flatnonzero(np.diff(above.astype(int), prepend=0) == 1)
    stops = np.flatnonzero(np.diff(above.astype(int), append=0) == -1)
    choices = []
    for left, right in zip(starts, stops):
        if left == 0 or right == len(profile) - 1:
            continue
        lo = _edge_crossing(profile, left - 1, level, rising=True)
        hi = _edge_crossing(profile, right, level, rising=False)
        if lo is None or hi is None:
            continue
        width = hi - lo
        center = origin + (lo + hi) / 2
        if 1.2 <= width <= 0.45 * span and abs(center - expected) <= max(3., .15 * span):
            choices.append((abs(center - expected), center, width, contrast))
    return min(choices)[1:] if choices else None


def _profile_weight(width, contrast, med_width):
    """Combine width consistency and edge contrast into a LSQ weight."""
    if med_width <= 0:
        return 0.0
    sigma = max(_WIDTH_SOFT_SIGMA * med_width, 0.35)
    dz = (width - med_width) / sigma
    if abs(dz) > _WIDTH_HARD_SIGMA / _WIDTH_SOFT_SIGMA:
        return 0.0
    width_w = float(np.exp(-0.5 * dz * dz))
    contrast_w = float(min(contrast, _CONTRAST_TRUST) / _CONTRAST_TRUST)
    return width_w * max(contrast_w, 0.05)


def _fit_weighted_line(t, x, w):
    """Weighted LSQ for x = a t + b. Returns (a, b) or None."""
    mask = w > 0
    if np.count_nonzero(mask) < 2:
        return None
    tw, xw, ww = t[mask], x[mask], w[mask]
    sw = float(np.sum(ww))
    if sw <= 1e-12:
        return None
    st = float(np.sum(ww * tw))
    sx = float(np.sum(ww * xw))
    stt = float(np.sum(ww * tw * tw))
    stx = float(np.sum(ww * tw * xw))
    det = stt * sw - st * st
    if abs(det) < 1e-12:
        return None
    a = (stx * sw - st * sx) / det
    b = (stt * sx - st * stx) / det
    return float(a), float(b)


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
                center, width, contrast = result
                samples.append((row - y, center, width, contrast, side))
    if len(samples) < 8:
        return None
    points = np.asarray(samples, dtype=float)
    widths = points[:, 2]
    med_width = float(np.median(widths))
    weights = np.array([
        _profile_weight(width, contrast, med_width)
        for width, contrast in zip(points[:, 2], points[:, 3])
    ], dtype=float)
    # Hard width band still required so a different bright bar cannot enter.
    keep = np.abs(widths - med_width) <= max(1.5, .25 * med_width)
    keep &= weights > 0
    # Initial slope from all pairs of opposite-arm samples, robust to debris.
    upper = points[(points[:, 4] < 0) & keep]
    lower = points[(points[:, 4] > 0) & keep]
    if len(upper) < 3 or len(lower) < 3:
        return None
    slopes = (lower[:, None, 1] - upper[None, :, 1]) / (lower[:, None, 0] - upper[None, :, 0])
    a = float(np.median(slopes))
    b = float(np.median(points[keep, 1] - a * points[keep, 0]))
    for _ in range(3):
        residual = points[:, 1] - (a * points[:, 0] + b)
        mad = float(np.median(np.abs(residual[keep] - np.median(residual[keep]))))
        keep &= np.abs(residual) <= max(.6, 3 * 1.4826 * mad)
        keep &= weights > 0
        if any(np.count_nonzero(keep & (points[:, 4] == side)) < 3 for side in (-1, 1)):
            return None
        fitted = _fit_weighted_line(points[keep, 0], points[keep, 1], weights[keep])
        if fitted is None:
            return None
        a, b = fitted
    keep &= weights > 0
    if np.count_nonzero(keep) < 8:
        return None
    rms = float(np.sqrt(np.mean((points[keep, 1] - (a * points[keep, 0] + b)) ** 2)))
    if abs(a) > .30 or rms > max(.7, .08 * med_width):
        return None
    mean_contrast = float(np.mean(points[keep, 3]))
    mean_weight = float(np.mean(weights[keep]))
    return float(a), float(b), rms, int(np.count_nonzero(keep)), mean_contrast, mean_weight


def refine_arm_edges(image, x, y, span):
    """Return (x, y, evidence), or None if four-arm evidence is insufficient."""
    vertical = _midline(image, x, y, span)
    horizontal = _midline(image.T, y, x, span)
    if vertical is None or horizontal is None:
        return None
    a, b, vrms, vn, v_contrast, v_weight = vertical
    c, d, hrms, hn, h_contrast, h_weight = horizontal
    # x' = a*(y'-y)+b; y' = c*(x'-x)+d.
    nx = (a * (d - c * x - y) + b) / (1 - a * c)
    ny = c * (nx - x) + d
    shift = float(np.hypot(nx - x, ny - y))
    if shift > max(2., .12 * span):
        return None
    return float(nx), float(ny), {
        "method": "arm-edges",
        "edge_interp": "hard-step-or-smooth-hybrid",
        "midline_fit": "contrast-width-weighted",
        "shift_px": shift,
        "vertical_rms_px": vrms, "horizontal_rms_px": hrms,
        "vertical_samples": vn, "horizontal_samples": hn,
        "vertical_contrast": v_contrast, "horizontal_contrast": h_contrast,
        "vertical_mean_weight": v_weight, "horizontal_mean_weight": h_weight,
    }
