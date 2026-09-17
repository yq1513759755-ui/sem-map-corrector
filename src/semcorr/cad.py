"""AutoCAD raster placement from marker-row-column-sequence region filenames."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
import cv2
import numpy as np
from .io import list_images, sha256_file

REGION = re.compile(r'(?P<marker>[0-9]{4})-(?P<row>[1-4])-(?P<col>[1-4])(?:-(?P<sequence>[0-9]+))?')
IDS = ('M1','M2','M3','M4')
REGION_PITCH_UM = 50.0


def parse_region(stem):
    """Decode aabb-row-column[-sequence]; rows go down, columns go right.

    The aabb marker is the centre of a 200 um square, divided into 4x4 cells.
    Hyphens separate row and column. The SEM acquisition suffix is metadata only.
    """
    match = REGION.fullmatch(stem)
    if not match:
        raise ValueError('图片命名必须为 0303-1-4-01.tif（数字marker-行-列-序号，行列均为1–4）；'
                         '末尾数字序号不参与坐标换算。旧点号、括号坐标及 r3c4 格式已停用')
    marker = match['marker']
    row, column = int(match['row']), int(match['col'])
    xc, yc = 100.0 * int(marker[:2]), 100.0 * int(marker[2:])
    x0 = xc - 100.0 + 50.0 * (column - 1)
    y0 = yc + 100.0 - 50.0 * row
    return {'marker_code': marker, 'marker_center_um': [xc, yc],
            'row': row, 'column': column,
            'sequence': match['sequence'],
            'bottom_left_um': [x0, y0], 'top_right_um': [x0 + 50.0, y0 + 50.0]}


def parse_anchor(stem):
    region = parse_region(stem)
    return np.asarray(region['bottom_left_um'], dtype=float), 'filename-region'


def validate_region_pitch(pitch_um):
    if not math.isfinite(pitch_um) or pitch_um != REGION_PITCH_UM:
        raise ValueError('区域编号规范固定为 200×200 µm 内的 4×4 个 50×50 µm 小格；--pitch-um 必须为50')


def fit_bottom_left(points,height,anchor_um,pitch_um=50.):
    """Fix M3 exactly; fit scale/rotation with all three displacement vectors.

    Pixel centres (OpenCV) map to outer-corner local coords (x+.5,H-y-.5).
    IMAGE DXF 10/11/12 encode the outer origin and the two per-pixel vectors.
    """
    if not math.isfinite(pitch_um) or pitch_um<=0:
        raise ValueError('间距必须为正数')
    src=np.asarray(points,dtype=float)
    if src.shape!=(4,2) or not np.isfinite(src).all():
        raise ValueError('需要 M1..M4 四个有限中心坐标')
    if not (src[0,0]<src[1,0] and src[2,0]<src[3,0] and src[0,1]<src[2,1] and src[1,1]<src[3,1]):
        raise ValueError('标记必须按左上、右上、左下、右下编号')
    local=np.column_stack((src[:,0]+.5,height-src[:,1]-.5))
    anchor=np.asarray(anchor_um,dtype=float)
    dst=anchor+pitch_um*np.array([[0,1],[1,1],[0,0],[1,0]])
    d=local-local[2];t=dst-dst[2]
    z=d[:,0]+1j*d[:,1];w=t[:,0]+1j*t[:,1]
    denom=float(np.vdot(z,z).real)
    if denom<=1e-12:
        raise ValueError('标记间距退化')
    f=np.vdot(z,w)/denom
    matrix=np.array([[f.real,-f.imag],[f.imag,f.real]])
    origin=anchor-matrix@local[2];fitted=local@matrix.T+origin
    errors=np.linalg.norm(fitted-dst,axis=1)
    return dict(scale_um_per_px=float(abs(f)),rotation_deg=float(np.degrees(np.angle(f))),
                origin_um=origin.tolist(),u_um=matrix[:,0].tolist(),v_um=matrix[:,1].tolist(),
                targets_um=dst.tolist(),fitted_um=fitted.tolist(),residuals_um=errors.tolist(),
                rms_um=float(np.sqrt(np.mean(errors**2))),max_residual_um=float(errors.max()))


def prepare_image(raw,corrected_dir,pitch_um,max_residual_um):
    validate_region_pitch(pitch_um)
    region=parse_region(raw.stem)
    anchor,source=parse_anchor(raw.stem)
    rp=corrected_dir/'diagnostics'/(raw.stem+'_report.json')
    ip=corrected_dir/(raw.stem+'_corrected.tif')
    raw_hash=sha256_file(raw)
    if not rp.is_file() or not ip.is_file():
        # Renaming a raw image does not invalidate its correction. Reconnect by
        # file content, never by a guessed coordinate prefix or similar name.
        matches=[]
        for candidate in (corrected_dir/'diagnostics').glob('*_report.json'):
            record=json.loads(candidate.read_text(encoding='utf-8'))
            image=corrected_dir/(candidate.name[:-len('_report.json')]+'_corrected.tif')
            if record.get('input_sha256')==raw_hash and image.is_file():
                matches.append((candidate,image))
        if len(matches)!=1:
            raise ValueError('缺少唯一且成功的校正图或报告，请先完成校正')
        rp,ip=matches[0]
    r=json.loads(rp.read_text(encoding='utf-8'))
    if r.get('quality_status')!='PASS' or r.get('grid')!=[2,2]:
        raise ValueError('仅导出 PASS 的 2x2 校正结果')
    if r.get('input_sha256')!=raw_hash:
        raise ValueError('原图 SHA256 与报告不一致')
    cc=r.get('corrected_centers',{})
    if cc.get('unit')!='px':
        raise ValueError('校正中心必须为像素坐标，不能混用 --design 单位')
    records=cc.get('points',[])
    if len(records)!=4 or {p.get('id') for p in records}!=set(IDS):
        raise ValueError('报告必须恰好包含 M1..M4')
    byid={p['id']:p for p in records}
    checked={p['id']:p for p in r.get('self_check',{}).get('marks',[])}
    for name in IDS:
        if byid[name].get('source')!='self-check':
            raise ValueError('拒绝理想格位回退坐标：'+name)
        if checked.get(name,{}).get('center_refinement',{}).get('method')!='arm-edges':
            raise ValueError('缺少四臂边缘复核：'+name)
        if not np.allclose(checked[name]['detected'],[byid[name]['x'],byid[name]['y']],rtol=0,atol=1e-8):
            raise ValueError('报告中心字段不一致：'+name)
    image=cv2.imread(str(ip),cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('无法读取校正图')
    height,width=image.shape[:2]
    points=np.array([[byid[n]['x'],byid[n]['y']] for n in IDS],float)
    if not np.isfinite(points).all() or not ((points>=0).all() and (points[:,0]<width).all() and (points[:,1]<height).all()):
        raise ValueError('中心坐标越界')
    for x,y in points:
        ix,iy=int(round(x)),int(round(y))
        if ix>=width or iy>=height or not np.array_equal(image[iy,ix],[0,0,255]):
            raise ValueError('校正图红标与报告不对应')
    fit=fit_bottom_left(points,height,anchor,pitch_um)
    if fit['max_residual_um']>max_residual_um:
        raise ValueError('最大单点偏差 %.4f µm 超过 %.4f µm'%(fit['max_residual_um'],max_residual_um))
    if abs(fit['rotation_deg'])>5:
        raise ValueError('旋转超过 5°，请确认图像右=+X、图像上=+Y')
    image_hash=sha256_file(ip)
    identity=hashlib.sha256((raw.stem+image_hash+json.dumps(anchor.tolist())).encode()).hexdigest()[:12]
    return dict(name=raw.stem,id='SEM_'+identity,anchor_um=anchor.tolist(),anchor_source=source,region=region,
                pitch_um=pitch_um,image=str(ip.resolve()),report=str(rp.resolve()),
                raw_sha256=r['input_sha256'],corrected_sha256=image_hash,
                width_px=width,height_px=height,centers_px=points.tolist(),**fit)


def lisp_string(value):
    return '"'+str(value).replace('\\','/').replace('"','\\"').replace('\n',' ')+'"'


def lisp_point(value):
    return '('+' '.join('%.12g'%v for v in [*value,0.])+')'


def export_batch(folder,*,outdir=None,pitch_um=50.,max_residual_um=.05,
                 corrected_dir=None,excluded=None):
    folder=Path(folder).resolve()
    validate_region_pitch(pitch_um)
    if not all(math.isfinite(x) and x>0 for x in (pitch_um,max_residual_um)):
        raise ValueError('间距和误差门槛必须为正数')
    corrected_dir=Path(corrected_dir).resolve() if corrected_dir else folder/'corrected'
    outdir=Path(outdir).resolve() if outdir else corrected_dir/'cad'
    excluded=excluded or {}
    rows,skipped=[],[]
    for raw in list_images(folder):
        # A failed current run must never reuse an older PASS report.
        if raw.name in excluded:
            skipped.append(dict(name=raw.name,reason=excluded[raw.name]))
            continue
        try:
            rows.append(prepare_image(raw,corrected_dir,pitch_um,max_residual_um))
        except (ValueError,KeyError,TypeError,OSError) as exc:
            skipped.append(dict(name=raw.name,reason=str(exc)))
    outdir.mkdir(parents=True,exist_ok=True);(outdir/'images').mkdir(exist_ok=True)
    for row in rows:
        row['bundle_image']='images/'+row['id']+'.tif'
        dest=outdir/row['bundle_image'];shutil.copy2(row['image'],dest)
        if sha256_file(dest)!=row['corrected_sha256']:
            raise RuntimeError('图像复制校验失败')
    summary=dict(schema_version=2,naming_convention='marker-row-column-sequence',coordinate_unit='um',anchor='bottom-left M3',
                 orientation='image-right=+X,image-up=+Y',pixel_convention='x+0.5,H-y-0.5',
                 pitch_um=pitch_um,max_residual_um=max_residual_um,images=rows,skipped=skipped)
    (outdir/'cad_manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with (outdir/'cad_params.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['name','marker_code','region_row','region_column','anchor_x_um','anchor_y_um','origin_x_um','origin_y_um','um_per_px','rotation_deg','width_um','height_um','rms_um','max_residual_um','layer'])
        for r in rows:
            writer.writerow([r['name'],r['region']['marker_code'],r['region']['row'],r['region']['column'],*r['anchor_um'],*r['origin_um'],r['scale_um_per_px'],r['rotation_deg'],r['width_px']*r['scale_um_per_px'],r['height_px']*r['scale_um_per_px'],r['rms_um'],r['max_residual_um'],r['id']])
    data=[]
    for r in rows:
        data.append('  ('+' '.join([lisp_string(r['id']),lisp_string(r['bundle_image']),str(r['width_px']),str(r['height_px']),lisp_point(r['origin_um']),lisp_point(r['u_um']),lisp_point(r['v_um'])])+')')
    header='(setq semmap-root '+lisp_string(str(outdir)+'/')+')\n'
    header+="(setq semmap-data '(\n"+'\n'.join(data)+'\n))\n'
    runtime=Path(__file__).with_name('cad_runtime.lsp').read_text(encoding='utf-8')
    (outdir/'sem_map.lsp').write_text(header+runtime,encoding='utf-8')
    (outdir/'attach_all.scr').write_text('(load '+lisp_string(outdir/'sem_map.lsp')+')\nSEMMAP\n',encoding='utf-8')
    (outdir/'README.md').write_text(f'''# 自动贴图

可用 {len(rows)} 张；跳过 {len(skipped)} 张，详见 cad_manifest.json。

1. 在版图副本的模型空间工作。1 个绘图单位代表 1 µm。
2. APPLOAD 加载本目录 sem_map.lsp，先输入 SEMMAPONE 试贴第一张，再输入 SEMMAP 批量贴图；或 SCRIPT 选择 attach_all.scr。
3. 每张图独立 SEM_ 图层。重复运行跳过已存在的图层，不重复贴图。
4. SEMMAPCHECK 核查实际 IMAGE 的插入点、每像素向量和尺寸。
5. 核对后另存为 DWG。遇到贴图失败时停止后续贴图。脚本不自动保存。图像为外部参照，请保留整个 cad 文件夹。

文件名采用 0303-1-4-01.tif：0303 为数字 marker 编号，1-4 为第1行第4列，末尾01仅为SEM图片序号。
marker 中心 (300,300) µm，所属小格左下 (350,350)、右上 (400,400) µm。
行从上往下、列从左往右；图像右=+X、上=+Y，间距固定 {pitch_um:g} µm。
旧点号区域编号、括号坐标、r3c4 命名和坐标覆盖 JSON 不再使用。
左下锚点严格固定，其他点拟合比例和旋转；最大单点偏差门槛 {max_residual_um:g} µm。
像素中心转换为 (x+0.5,H-y-0.5)，插入点为图像外边界左下角。
IMAGE 的 DXF 10/11/12 控制插入点及每像素向量，不依赖 DPI 或 INSUNITS，不更改原版图单位设置。
已通过拟合不代表实际曝光套刻精度。

来源：
https://help.autodesk.com/cloudhelp/2018/ENU/AutoCAD-DXF/files/GUID-3A2FF847-BE14-4AC5-9BD4-BD3DCAEF2281.htm
https://help.autodesk.com/cloudhelp/2018/ENU/OARX-RefGuide/files/OREF-AcDbRasterImage__getOrientation_AcGePoint3d__AcGeVector3d__AcGeVector3d__const.html
''',encoding='utf-8')
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description='文件名 0303-1-4-01.tif（marker-行-列-序号）自动换算坐标并生成 AutoCAD 贴图包')
    p.add_argument('folder');p.add_argument('--outdir')
    p.add_argument('--pitch-um',type=float,default=50.)
    p.add_argument('--max-residual-um',type=float,default=.05)
    a=p.parse_args(argv)
    try:
        r=export_batch(a.folder,outdir=a.outdir,pitch_um=a.pitch_um,max_residual_um=a.max_residual_um)
    except (ValueError,OSError) as exc:
        p.exit(2,str(exc)+'\n')
    print('贴图包：可用 %d 张 / 跳过 %d 张'%(len(r['images']),len(r['skipped'])))
    for e in r['skipped']:print('[跳过] %(name)s: %(reason)s'%e)
    return 1 if r['skipped'] else 0
