import json
import pytest
from semcorr.cli import main
from semcorr.demo import make_demo_image


def scene(folder,name='0303-11-01.tif'):
    folder.mkdir(exist_ok=True)
    return make_demo_image(folder/name,n_rows=2,n_cols=2)


def test_one_command_custom_output_and_anchor(tmp_path):
    root=tmp_path/'input';scene(root);out=tmp_path/'custom'
    assert main(['--batch',str(root),'--outdir',str(out)])==0
    summary=json.loads((out/'workflow_summary.json').read_text())
    assert summary['ready']==1 and summary['images'][0]['status']=='ready'
    report=json.loads((out/'cad/cad_manifest.json').read_text())
    assert report['images'][0]['anchor_um']==[350.,350.]
    assert (out/'cad/sem_map.lsp').is_file()
    assert not (root/'corrected').exists()


def test_batch_defaults_to_cad_without_flag(tmp_path):
    root=tmp_path/'input';scene(root)
    assert main(['--batch',str(root)])==0
    assert (root/'corrected/cad/sem_map.lsp').is_file()
    assert (root/'corrected/workflow_summary.json').is_file()


def test_directory_positional_is_batch(tmp_path):
    root=tmp_path/'input';scene(root)
    assert main([str(root)])==0
    assert (root/'corrected/cad/sem_map.lsp').is_file()


def test_no_cad_skips_bundle(tmp_path):
    root=tmp_path/'input';scene(root)
    assert main(['--batch',str(root),'--no-cad'])==0
    assert (root/'corrected/0303-11-01_corrected.tif').is_file()
    assert not (root/'corrected/cad').exists()


def test_cad_and_no_cad_conflict(tmp_path):
    root=tmp_path/'input';scene(root)
    assert main(['--batch',str(root),'--cad','--no-cad'])==1


def test_current_failure_cannot_reuse_old_pass(tmp_path,monkeypatch):
    root=tmp_path/'input';scene(root)
    args=['--batch',str(root),'--force']
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


def test_incremental_skips_unchanged_pass(tmp_path,monkeypatch):
    root=tmp_path/'input';scene(root)
    assert main(['--batch',str(root)])==0
    from semcorr import pipeline
    def boom(*a,**k):
        raise AssertionError('should not re-correct unchanged PASS')
    monkeypatch.setattr(pipeline,'correct_image',boom)
    assert main(['--batch',str(root)])==0
    assert (root/'corrected/cad/sem_map.lsp').is_file()
    assert (root/'corrected/workflow_summary.json').is_file()


def test_force_reruns_even_when_pass_exists(tmp_path,monkeypatch):
    root=tmp_path/'input';scene(root)
    assert main(['--batch',str(root)])==0
    from semcorr import pipeline
    calls=[]
    real=pipeline.correct_image
    def counted(*a,**k):
        calls.append(1)
        return real(*a,**k)
    monkeypatch.setattr(pipeline,'correct_image',counted)
    assert main(['--batch',str(root),'--force'])==0
    assert len(calls)==1


def test_incremental_reuses_renamed_file_by_hash(tmp_path):
    root=tmp_path/'input';scene(root)
    assert main(['--batch',str(root)])==0
    src=root/'0303-11-01.tif'
    dst=root/'0303-12-01.tif'
    src.rename(dst)
    assert main(['--batch',str(root)])==0
    # New name still exports via hash reconnect of the old PASS report.
    report=json.loads((root/'corrected/cad/cad_manifest.json').read_text())
    assert report['images'] and report['images'][0]['raw_sha256']


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
    ['--max-residual-um','0'],['--pitch-um','100']])
def test_invalid_options_fail_before_processing(tmp_path,extra):
    root=tmp_path/'input';scene(root)
    assert main(['--batch',str(root),'--cad',*extra])==1
    assert not (root/'corrected').exists()


def test_cad_requires_batch():
    assert main(['--cad'])==1


def test_original_correction_only_command_is_unchanged(tmp_path):
    root=tmp_path/'input';scene(root)
    assert main(['--batch',str(root),'--no-cad'])==0
    assert (root/'corrected/0303-11-01_corrected.tif').is_file()
    assert not (root/'corrected/cad').exists()


@pytest.mark.parametrize('legacy',['0303-02(3.5,2).tif','0303-1-4-01.tif','0303-1.4.tif','0719-01.tif'])
def test_legacy_name_not_corrected_in_cad_mode(tmp_path,legacy):
    root=tmp_path/'input';scene(root,legacy)
    assert main(['--batch',str(root),'--cad'])==1
    assert not (root/'corrected'/(legacy[:-4]+'_corrected.tif')).exists()
    report=json.loads((root/'corrected/workflow_summary.json').read_text())
    assert report['ready']==0
    assert report['images'][0]['status']=='cad_rejected'
    assert '命名' in report['images'][0]['reason']


def test_coordinate_overrides_option_removed():
    with pytest.raises(SystemExit) as exc:
        main(['--batch','unused','--cad','--anchor-overrides','unused.json'])
    assert exc.value.code==2
