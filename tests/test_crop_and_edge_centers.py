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
    assert np.linalg.norm([x-truth[0],y-truth[1]])<.2


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
