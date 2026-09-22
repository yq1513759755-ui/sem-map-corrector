"""Regression cases for framed Zeiss footers and biased cross templates."""
import cv2
import numpy as np
import pytest

from semcorr.infobar import strip_info_bar
from semcorr.detectors.edges import refine_arm_edges
from semcorr.detectors.se2 import refine_center


@pytest.mark.parametrize('width,height,footer', [(1024,768,61), (640,480,40)])
@pytest.mark.parametrize('last_dark', [False,True])
def test_framed_footer_ignores_tail_and_removes_white_separator(width,height,footer,last_dark):
    top = height-footer
    rng = np.random.default_rng(1)
    sample = rng.integers(42,85,(top,width),dtype=np.uint8)
    image = np.zeros((height,width),np.uint8)
    image[:top] = sample
    image[top:top+2] = 255
    image[-3:-1] = 255
    image[-1] = 0 if last_dark else 60
    for x in range(10,width-20,40):
        image[top+12:top+18,x:x+16] = 255
    cropped,info = strip_info_bar(image)
    assert info['top'] == top
    assert info['dark_top'] == top+2
    assert info['removed_rows'] == footer
    np.testing.assert_array_equal(cropped,sample)
    # A second run must not crop the already clean sample.
    again,info = strip_info_bar(cropped)
    assert info is None
    np.testing.assert_array_equal(again,sample)


def _asymmetric_cross(angle=0):
    # Render at 4x and area-integrate, with known subpixel centre.
    scale=4; size=240; cx,cy=119.3,121.7
    yy,xx=np.mgrid[:size*scale,:size*scale]
    xx=xx/scale-cx; yy=yy/scale-cy
    theta=np.deg2rad(angle)
    u=np.cos(theta)*xx+np.sin(theta)*yy
    v=-np.sin(theta)*xx+np.cos(theta)*yy
    mask=((abs(u)<=5)&(v>=-55)&(v<=65))|((abs(v)<=6)&(u>=-65)&(u<=42))
    image=cv2.resize(np.where(mask,190,35).astype(np.float32),(size,size),interpolation=cv2.INTER_AREA)
    image=cv2.GaussianBlur(image,(5,5),.7)
    image+=np.random.default_rng(0).normal(0,2,image.shape)
    truth=np.array([cx,cy])-(scale-1)/(2*scale)
    return np.clip(image,0,255).astype(np.uint8),truth


@pytest.mark.parametrize('angle',[0,3,8])
def test_unequal_arms_with_noise_and_rotation_have_known_center(angle):
    image,truth=_asymmetric_cross(angle)
    candidate={'cx':truth[0]-2,'cy':truth[1]+1,'span':120,'arm_ratio':.1}
    x,y,_=refine_center(image,candidate)
    assert candidate['center_refinement']['method']=='arm-edges'
    assert candidate['center_refinement']['edge_interp']=='hard-step-or-smooth-hybrid'
    assert candidate['center_refinement']['midline_fit']=='contrast-width-weighted'
    assert np.linalg.norm([x-truth[0],y-truth[1]])<.12


def test_mismatched_small_template_does_not_shift_large_cross_center():
    yy,xx=np.mgrid[:240,:240];cx,cy=120.2,118.7
    coverage=lambda t:np.clip(t+.5,0,1)
    v=coverage(6-abs(xx-cx))*coverage(60-abs(yy-cy))
    h=coverage(6-abs(yy-cy))*coverage(xx-cx+61)*coverage(45-(xx-cx))
    image=np.clip(35+160*np.maximum(v,h)+np.random.default_rng(3).normal(0,1,(240,240)),0,255).astype(np.uint8)
    # Missing-mark recovery uses the median (small-mark) span on a big mark.
    candidate={'cx':cx-1,'cy':cy+1,'span':34,'arm_ratio':.16}
    x,y,_=refine_center(image,candidate)
    assert candidate['center_refinement']['method']=='arm-edges'
    assert np.hypot(x-cx,y-cy)<.2


def test_edge_crossing_noise_and_curvature():
    """Near-linear noisy flank: 3-point line beats 2-point. Curved flank: parabolic wins."""
    from semcorr.detectors.edges import (
        _edge_crossing, _linear3_crossing, _parabolic_crossing,
    )

    true_edge = 20.37
    level = 100.0
    slope = 12.0
    xs = np.arange(12, 30, dtype=float)
    # Level is attained exactly at true_edge.
    clean = level + slope * (xs - true_edge)
    i_true = int(np.floor(true_edge)) - int(xs[0])
    assert clean[i_true] < level <= clean[i_true + 1]
    # _edge_crossing works in profile-index space (xs[0] == 12).
    true_index = true_edge - xs[0]
    errs_h, errs_l2 = [], []
    for k in range(40):
        p = clean + np.random.default_rng(100 + k).normal(0, 3.0, clean.shape)
        x_h = _edge_crossing(p, i_true, level, rising=True)
        x_2 = i_true + (level - p[i_true]) / (p[i_true + 1] - p[i_true])
        errs_h.append(abs(x_h - true_index))
        errs_l2.append(abs(x_2 - true_index))
    assert np.mean(errs_h) < np.mean(errs_l2)
    assert np.mean(errs_h) < 0.15

    # Clearly curved (asymmetric) flank: parabolic root is closer than 3-pt line.
    p0, p1, p2 = 10.0, 30.0, 80.0
    level_c = 45.0
    x_par = _parabolic_crossing(p0, p1, p2, 1, level_c)
    x_lin = _linear3_crossing(p0, p1, p2, 1, level_c)
    # Mid-centered quadratic through the three samples: 15 t^2 + 35 t - 15 = 0
    # root t ≈ 0.3699 → x = 1.3699.
    assert abs(x_par - 1.3699) < 0.01
    assert abs(x_par - 1.3699) < abs(x_lin - 1.3699)

    # Hard step: 2-point linear must be used (3-point would be biased).
    step = np.array([30.0, 30.0, 30.0, 200.0, 200.0, 200.0])
    x_step = _edge_crossing(step, 2, 115.0, rising=True)
    assert abs(x_step - (2 + (115.0 - 30.0) / 170.0)) < 1e-9


def test_width_outlier_is_downweighted_in_midline():
    """A bright blob with abnormal width on one row must not drag the midline."""
    from semcorr.detectors.edges import _midline, _profile_weight

    size = 200
    cx, cy = 100.0, 100.0
    scale = 4
    yy, xx = np.mgrid[:size * scale, :size * scale]
    xx = xx / scale - cx
    yy = yy / scale - cy
    mask = ((np.abs(xx) <= 2.5) & (np.abs(yy) <= 55)) | ((np.abs(yy) <= 2.5) & (np.abs(xx) <= 55))
    image = cv2.resize(
        np.where(mask, 200.0, 35.0).astype(np.float32),
        (size, size), interpolation=cv2.INTER_AREA,
    )
    image = cv2.GaussianBlur(image, (5, 5), 0.7).astype(np.float64)
    clean = _midline(image, cx, cy, 110)
    assert clean is not None

    row = int(round(cy)) - 20
    contaminated = image.copy()
    contaminated[row, :] = 35.0
    contaminated[row, int(cx) + 8:int(cx) + 28] = 220.0  # width ~20 px vs median ~6
    assert _profile_weight(20.0, 185.0, 6.0) == 0.0

    dirty = _midline(contaminated, cx, cy, 110)
    assert dirty is not None
    # Contamination must not move the axis (width outlier gets zero weight).
    assert abs(dirty[1] - clean[1]) < 0.05


@pytest.mark.parametrize('kind',['uniform','disk','missing-arm'])
def test_edge_measurement_rejects_missing_four_arm_evidence(kind):
    image=np.full((160,160),40,np.uint8)
    if kind=='disk':
        cv2.circle(image,(80,80),26,220,-1)
    elif kind=='missing-arm':
        image[75:86,50:111]=220
        image[50:81,75:86]=220
    assert refine_arm_edges(image,80,80,60) is None


def test_template_only_centers_require_review(tmp_path,monkeypatch):
    from semcorr.demo import make_demo_image
    from semcorr.pipeline import correct_image
    from semcorr.detectors import se2
    monkeypatch.setattr(se2,'refine_arm_edges',lambda *args:None)
    path=make_demo_image(tmp_path/'scene.tif',n_cols=2)
    report=correct_image(path,outdir=tmp_path/'out',verbose=False)
    assert report['self_check']['n_detected']==4
    assert report['quality_status']=='WARN_REVIEW'
    assert any('四臂边缘' in warning for warning in report['quality_warnings'])
