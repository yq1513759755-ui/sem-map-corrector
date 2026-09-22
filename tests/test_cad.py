import json
import re
import numpy as np
import pytest
import cv2
from semcorr.cad import parse_anchor,parse_region,fit_bottom_left,prepare_image,export_batch
from semcorr.io import sha256_file


@pytest.mark.parametrize('name,expected',[
 ('0719-12-03',[700,1950]),('0303-11-01',[350,350]),
 ('0101-41-01',[150,50]),('0101-22-01',[0,150]),
 ('0101-44-01',[150,0]),('0101-11',[150,150]),
 ('0303-11-99',[350,350]),('0719-12',[700,1950])])
def test_filename_anchor(name,expected):
    actual,source=parse_anchor(name)
    np.testing.assert_allclose(actual,expected)
    assert source=='filename-region'


def test_quadrant_digits_are_not_hyphen_separated_row_column():
    # The 0.4.1 row-column form must not be silently reinterpreted as quadrants.
    with pytest.raises(ValueError,match='象限'):parse_anchor('0303-1-4-01')


@pytest.mark.parametrize('name',[
 '0115-01','0719-01','0101(3,4)','0303-02(3.5,2)','0101_r3c4_01',
 '0303-1-4-01','0537-2-3-05','0501-3-1-02',
 '0303-0.1','0303-1.0','0303-5.1','0303-1.5','0303-1.40','0303-01.4','0303-1,4','0303-1.4',
 '0719-0-1','0719-5-1','0719-1-0','0719-1-5',
 '0719-15','0719-05','0719-55','0719-123','0719-1','0719-12-03-extra','0719-12-text',
 '719-12','0719-12junk','0719-12_00','0719_12','0719 12'])
def test_no_silent_coordinate_guess(name):
    with pytest.raises(ValueError):parse_anchor(name)


def test_region_metadata():
    r=parse_region('0719-12-03')
    assert r=={'marker_code':'0719','marker_center_um':[700.,1900.],
               'quadrant':1,'sub_quadrant':2,'sequence':'03',
               'bottom_left_um':[700.,1950.],'top_right_um':[750.,2000.]}


def test_quadrants_tile_the_200_um_block_without_gaps():
    cells={}
    for quadrant in range(1,5):
        for sub in range(1,5):
            region=parse_region(f'0303-{quadrant}{sub}-01')
            x,y=region['bottom_left_um']
            assert region['top_right_um']==[x+50,y+50]
            assert 200<=x<400 and 200<=y<400
            cells[(x,y)]=f'{quadrant}{sub}'
    assert len(cells)==16                      # 16 格互不重叠
    assert {x for x,_ in cells}=={200,250,300,350}
    assert {y for _,y in cells}=={200,250,300,350}


def test_quadrant_numbering_runs_counter_clockwise_from_plus_x_plus_y():
    def cell(name):return parse_region(name)['bottom_left_um']
    assert cell('0303-11')==[350,350]          # +X,+Y 象限的右上小格
    assert cell('0303-12')==[300,350]          # 同象限左上
    assert cell('0303-13')==[300,300]          # 同象限左下 = marker 中心
    assert cell('0303-14')==[350,300]          # 同象限右下
    assert cell('0303-21')==[250,350]          # −X,+Y 象限
    assert cell('0303-31')==[250,250]          # −X,−Y 象限
    assert cell('0303-41')==[350,250]          # +X,−Y 象限


def test_quadrant_codes_reach_the_same_cells_as_the_retired_row_column_grid():
    def legacy_anchor(row,column):            # 0.4.1: X0=Xc−100+50(c−1), Y0=Yc+100−50r
        return [300-100+50*(column-1),300+100-50*row]
    seen=set()
    for row in range(1,5):
        for column in range(1,5):
            want=legacy_anchor(row,column)
            hits=[f'{q}{s}' for q in range(1,5) for s in range(1,5)
                  if parse_region(f'0303-{q}{s}')['bottom_left_um']==want]
            assert len(hits)==1,(row,column,want,hits)
            seen.add(hits[0])
    assert len(seen)==16


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
    diag=out/'diagnostics';diag.mkdir();stem='0101-13-01'
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
    with pytest.raises(ValueError):prepare_image(raw,folder/'corrected',50,.05)


def test_renamed_raw_reconnects_by_content_hash(batch):
    folder,raw,rp=batch
    renamed=raw.with_name('0101-12-01.tif');raw.rename(renamed)
    row=prepare_image(renamed,folder/'corrected',50,.05)
    assert row['anchor_um']==[100,150]
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
    assert saved['schema_version']==3
    assert saved['naming_convention']=='marker-quadrant-subquadrant-sequence'
    assert saved['images'][0]['region']['quadrant']==1
    assert saved['images'][0]['region']['sub_quadrant']==3
    csv_header=(bundle/'cad_params.csv').read_text(encoding='utf-8-sig').splitlines()[0]
    assert csv_header.split(',')[2:4]==['quadrant','sub_quadrant']


def test_nonstandard_pitch_rejected_before_export(batch):
    folder,raw,rp=batch
    with pytest.raises(ValueError,match='50'):
        export_batch(folder,pitch_um=100)
    assert not (folder/'corrected/cad').exists()


def test_legacy_override_file_cannot_change_new_region(batch):
    folder,raw,rp=batch
    (folder/'cad_anchor_overrides.json').write_text(json.dumps({raw.stem:[99,99]}))
    report=export_batch(folder)
    assert report['images'][0]['anchor_um']==[100,100]


@pytest.mark.parametrize('suffix',['01','05','99','100','00','0'])
def test_sem_sequence_is_not_a_coordinate(suffix):
    r=parse_region('0719-12-'+suffix)
    assert r['quadrant']==1 and r['sub_quadrant']==2
    assert r['bottom_left_um']==[700.,1950.]
    assert r['sequence']==suffix


def test_sequence_optional():
    assert parse_region('0719-12')['bottom_left_um']==[700.,1950.]


def test_multiple_acquisitions_remain_distinct(batch):
    import shutil
    folder,raw,rp=batch
    other=raw.with_name('0101-13-05.tif');shutil.copy2(raw,other)
    r=export_batch(folder)
    assert len(r['images'])==2
    assert r['images'][0]['anchor_um']==r['images'][1]['anchor_um']
    assert r['images'][0]['id']!=r['images'][1]['id']
