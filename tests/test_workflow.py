import json
import pytest
from semcorr.cli import main
from semcorr.demo import make_demo_image


def scene(folder,name='sample(3.5,2).tif'):
    folder.mkdir(exist_ok=True)
    return make_demo_image(folder/name,n_rows=2,n_cols=2)


def test_one_command_custom_output_and_anchor(tmp_path):
    root=tmp_path/'input';scene(root);out=tmp_path/'custom'
    assert main(['--batch',str(root),'--cad','--outdir',str(out)])==0
    summary=json.loads((out/'workflow_summary.json').read_text())
    assert summary['ready']==1 and summary['images'][0]['status']=='ready'
    report=json.loads((out/'cad/cad_manifest.json').read_text())
    assert report['images'][0]['anchor_um']==[350.,200.]
    assert (out/'cad/sem_map.lsp').is_file()
    assert not (root/'corrected').exists()


def test_current_failure_cannot_reuse_old_pass(tmp_path,monkeypatch):
    root=tmp_path/'input';scene(root)
    args=['--batch',str(root),'--cad']
    assert main(args)==0
    from semcorr import pipeline
    def fail(*args,**kwargs):raise RuntimeError('current-run failure')
    monkeypatch.setattr(pipeline,'correct_image',fail)
    assert main(args)==1
    report=json.loads((root/'corrected/cad/cad_manifest.json').read_text())
    assert report['images']==[]
    assert 'current-run failure' in report['skipped'][0]['reason']
    summary=json.loads((root/'corrected/workflow_summary.json').read_text())
    assert summary['images'][0]['status']=='correction_failed'


def test_review_is_excluded_and_identified(tmp_path,monkeypatch):
    root=tmp_path/'input';scene(root)
    from semcorr import pipeline
    real=pipeline.correct_image
    def review(*a,**kw):
        report=real(*a,**kw)
        report['quality_status']='WARN_REVIEW';report['quality_warnings']=['test review']
        return report
    monkeypatch.setattr(pipeline,'correct_image',review)
    assert main(['--batch',str(root),'--cad'])==1
    s=json.loads((root/'corrected/workflow_summary.json').read_text())
    assert s['ready']==0 and s['images'][0]['status']=='review'


def test_partial_batch_exports_good_image(tmp_path):
    root=tmp_path/'input';scene(root);scene(root,'missing-coordinates.tif')
    assert main(['--batch',str(root),'--cad'])==1
    s=json.loads((root/'corrected/workflow_summary.json').read_text())
    assert s['ready']==1 and s['skipped']==1
    assert {x['status'] for x in s['images']}=={'ready','cad_rejected'}


@pytest.mark.parametrize('extra',[
    ['--grid','2x3'],['--design','not-read.json'],['--pitch-um','nan'],
    ['--max-residual-um','0'],['--anchor-overrides','missing.json']])
def test_invalid_options_fail_before_processing(tmp_path,extra):
    root=tmp_path/'input';scene(root)
    assert main(['--batch',str(root),'--cad',*extra])==1
    assert not (root/'corrected').exists()


def test_cad_requires_batch():
    assert main(['--cad'])==1


def test_original_correction_only_command_is_unchanged(tmp_path):
    root=tmp_path/'input';scene(root)
    assert main(['--batch',str(root)])==0
    assert (root/'corrected/sample(3.5,2)_corrected.tif').is_file()
    assert not (root/'corrected/cad').exists()
