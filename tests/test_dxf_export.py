"""Optional DXF merge export (--dxf)."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

# ezdxf is an optional extra (`.[dxf]`). Skip this module instead of aborting
# collection when it is absent — a bare top-level import makes pytest stop the
# whole session with "Interrupted: 1 error during collection".
ezdxf = pytest.importorskip("ezdxf")

from semcorr.dxf_export import export_dxf_from_cad
from semcorr.cli import main


def _fake_cad(root: Path):
    cad = root / "cad"
    (cad / "images").mkdir(parents=True)
    image = np.full((20, 30, 3), 180, np.uint8)
    cv2.imwrite(str(cad / "images" / "SEM_test.tif"), image)
    manifest = {
        "schema_version": 4,
        "images": [{
            "name": "0101-11-01",
            "id": "SEM_test",
            "bundle_image": "images/SEM_test.tif",
            "width_px": 30,
            "height_px": 20,
            "origin_um": [10.0, 20.0],
            "u_um": [0.1, 0.0],
            "v_um": [0.0, 0.1],
            "wires": [{
                "p1_um": [12.0, 22.0], "p2_um": [14.0, 22.0],
                "length_um": 2.0, "width_um": 0.5, "extend_um": 0.5,
                "quad_um": [[11.5, 22.25], [14.5, 22.25],
                            [14.5, 21.75], [11.5, 21.75]],
            }],
        }],
        "skipped": [],
    }
    (cad / "cad_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return cad


def test_export_dxf_writes_r2018_with_pixel_vectors(tmp_path):
    cad = _fake_cad(tmp_path)
    doc = ezdxf.new("R2018")
    template = tmp_path / "template.dxf"
    doc.saveas(template)
    result = export_dxf_from_cad(
        cad, template=template, outdir=tmp_path / "DXF", name="BATCH")
    out = Path(result["output"])
    assert out.is_file()
    assert result["n_images"] == 1
    loaded = ezdxf.readfile(out)
    assert loaded.dxfversion == "AC1032"
    images = list(loaded.modelspace().query("IMAGE"))
    assert len(images) == 1
    image = images[0]
    assert tuple(image.dxf.insert)[:2] == (10.0, 20.0)
    assert tuple(image.dxf.u_pixel)[:2] == pytest.approx((0.1, 0.0))
    assert tuple(image.dxf.v_pixel)[:2] == pytest.approx((0.0, 0.1))
    assert image.dxf.layer == "photo"
    assert "photo" in loaded.layers
    defs = list(loaded.objects.query("IMAGEDEF"))
    assert defs and "SEM_test.tif" in defs[0].dxf.filename
    assert result["n_wires"] == 1
    wires = list(loaded.modelspace().query('LWPOLYLINE[layer=="wire"]'))
    assert len(wires) == 1 and wires[0].closed
    pts = [(float(x), float(y)) for x, y in wires[0].get_points("xy")]
    assert pts[0] == pytest.approx((11.5, 22.25))


def test_cli_dxf_flag_exports_to_custom_dir(tmp_path):
    from semcorr.demo import make_demo_image

    root = tmp_path / "input"
    root.mkdir()
    make_demo_image(root / "0303-11-01.tif", n_rows=2, n_cols=2)
    template = tmp_path / "template.dxf"
    ezdxf.new("R2018").saveas(template)
    outdir = tmp_path / "DXF"
    code = main([
        "--batch", str(root), "--dxf",
        "--dxf-template", str(template), "--dxf-outdir", str(outdir),
    ])
    assert code == 0
    produced = list(outdir.glob("*.dxf"))
    assert produced and produced[0].name == f"{root.name}.dxf"


def test_cli_dxf_requires_batch():
    assert main(["--dxf"]) == 1
