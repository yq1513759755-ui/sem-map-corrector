import json
import re
import numpy as np
import pytest
import cv2
from semcorr.cad import parse_anchor,fit_bottom_left,prepare_image,export_batch
from semcorr.io import sha256_file


@pytest.mark.parametrize('name,expected',[
 ('0303-02(3.5,2)',[350,200]),('any（-1，2.5）',[-100,250]),
 ('abc(.5,+2)',[50,200]),('3719-05(37.5,19.5)',[3750,1950])])
def test_filename_anchor(name,expected):
    actual,source=parse_anchor(name)
    np.testing.assert_allclose(actual,expected)
    assert source=='filename-bottom-left'


@pytest.mark.parametrize('name',['0115-01','3719-05(3.7.5,19.5)','a(1,2) junk','a(nan,2)'])
def test_no_silent_coordinate_guess(name):
    with pytest.raises(ValueError):parse_anchor(name)


def test_explicit_override():
    xy,source=parse_anchor('bad(3.7.5,2)',{'bad(3.7.5,2)':[37.5,2]})
    np.testing.assert_allclose(xy,[3750,200]);assert source=='explicit-override'


@pytest.mark.parametrize('angle',[-2,0,3])
def test_known_similarity_and_pixel_corner_convention(angle):
    th=np.deg2rad(angle);scale=.09
    mat=scale*np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
    origin=np.array([325.,170.]);height=707
    anchor=np.array([350.,200.]);target=anchor+50*np.array([[0,1],[1,1],[0,0],[1,0]])
    local=(target-origin)@np.linalg.inv(mat).T
    centers=np.column_stack((local[:,0]-.5,height-local[:,1]-.5))
    f=fit_bottom_left(centers,height,anchor)
    np.testing.assert_allclose(f['origin_um'],origin,atol=1e-10)
    np.testing.assert_allclose(f['u_um'],mat[:,0],atol=1e-10)
    np.testing.assert_allclose(f['v_um'],mat[:,1],atol=1e-10)
    assert f['max_residual_um']<1e-10


def test_anchor_is_exact_even_with_noisy_other_marks():
    centers=np.array([[200.,100.],[700.,100.],[200.,600.],[700.,600.]])
    centers[0]+=[.2,-.3]
    f=fit_bottom_left(centers,707,[1000,2000])
    np.testing.assert_allclose(f['fitted_um'][2],[1000,2000],atol=1e-12)
    assert 0<f['max_residual_um']<.05


@pytest.fixture
def batch(tmp_path):
    folder=tmp_path/'batch';folder.mkdir();out=folder/'corrected';out.mkdir()
    diag=out/'diagnostics';diag.mkdir();stem='0101-01(1,1)'
    raw=folder/(stem+'.tif');cv2.imwrite(str(raw),np.full((768,1024),30,np.uint8))
    points=[[200.,100.],[700.,100.],[200.,600.],[700.,600.]]
    image=np.full((707,1024,3),50,np.uint8)
    for x,y in points:image[int(y),int(x)]=[0,0,255]
    cv2.imwrite(str(out/(stem+'_corrected.tif')),image)
    records=[dict(id='M%d'%(i+1),x=p[0],y=p[1],source='self-check') for i,p in enumerate(points)]
    checked=[dict(id='M%d'%(i+1),detected=p,center_refinement={'method':'arm-edges'}) for i,p in enumerate(points)]
    r=dict(quality_status='PASS',grid=[2,2],input_sha256=sha256_file(raw),corrected_centers=dict(unit='px',points=records),self_check=dict(marks=checked))
    rp=diag/(stem+'_report.json');rp.write_text(json.dumps(r))
    return folder,raw,rp


@pytest.mark.parametrize('change',['warn','fallback','units','hash','edges'])
def test_reject_untrustworthy_reports(batch,change):
    folder,raw,rp=batch;r=json.loads(rp.read_text())
    if change=='warn':r['quality_status']='WARN_REVIEW'
    if change=='fallback':r['corrected_centers']['points'][0]['source']='ideal-fallback'
    if change=='units':r['corrected_centers']['unit']='设计单位'
    if change=='hash':r['input_sha256']='wrong'
    if change=='edges':r['self_check']['marks'][0]['center_refinement']={}
    rp.write_text(json.dumps(r))
    with pytest.raises(ValueError):prepare_image(raw,folder/'corrected',{},50,.05)


def test_renamed_raw_reconnects_by_content_hash(batch):
    folder,raw,rp=batch
    renamed=raw.with_name('0101-01(1.5,2).tif');raw.rename(renamed)
    row=prepare_image(renamed,folder/'corrected',{},50,.05)
    assert row['anchor_um']==[150,200]
    assert row['report']==str(rp)


def test_full_bundle_round_trip(batch):
    folder,raw,rp=batch
    r=export_batch(folder)
    assert len(r['images'])==1 and not r['skipped']
    row=r['images'][0];bundle=folder/'corrected/cad'
    assert sha256_file(bundle/row['bundle_image'])==row['corrected_sha256']
    text=(bundle/'sem_map.lsp').read_text()
    assert 'vl-load-com' not in text
    assert '(cons 11 (nth 5 row))' in text
    assert '(cons 12 (nth 6 row))' in text
    # Lisp syntax sanity, including quoted filenames and comment lines.
    stripped=re.sub(r'"(?:\\.|[^"\\])*"','""',text)
    stripped=re.sub(r';[^\n]*','',stripped);depth=0
    for ch in stripped:
        depth+=(ch=='(')-(ch==')')
        assert depth>=0
    assert depth==0
    saved=json.loads((bundle/'cad_manifest.json').read_text())
    assert saved['images'][0]['anchor_um']==[100,100]
