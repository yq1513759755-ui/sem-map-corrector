"""纳米线绿色标注 → wire 矩形：检测、映射、端到端。"""
import math

import cv2
import numpy as np
import pytest

from semcorr.wires import (WIRE_EXTEND_UM, WIRE_WIDTH_UM, detect_green_segments,
                           detect_wires_in_corrected, green_mask, map_segment,
                           px_to_um_factory, segment_to_quad)


def canvas_with_line(p1, p2, shape=(200, 200), thickness=2):
    img = np.full((shape[0], shape[1], 3), 120, np.uint8)
    cv2.line(img, tuple(p1), tuple(p2), (0, 255, 0), thickness)
    return img


def test_green_segment_detection_geometry():
    img = canvas_with_line((30, 40), (150, 90))
    segs = detect_green_segments(img)
    assert len(segs) == 1
    seg = segs[0]
    ends = sorted([seg["p1"], seg["p2"]])
    assert ends[0] == pytest.approx((30, 40), abs=2.0)
    assert ends[1] == pytest.approx((150, 90), abs=2.0)
    assert seg["length_px"] == pytest.approx(
        math.hypot(120, 50), rel=0.05)
    assert 1.0 <= seg["width_px"] <= 4.0


def test_crop_top_excludes_info_bar_graphics():
    img = canvas_with_line((30, 150), (150, 150), shape=(200, 200))
    assert detect_green_segments(img)            # 不裁时检出
    assert detect_green_segments(img, crop_top=100) == []


def test_small_green_fragments_ignored():
    img = np.full((200, 200, 3), 120, np.uint8)
    img[50, 50] = (0, 255, 0)
    img[51, 50] = (0, 255, 0)
    assert detect_green_segments(img) == []


def test_gray_input_rejected():
    with pytest.raises(ValueError):
        green_mask(np.full((50, 50), 120, np.uint8))


def test_map_segment_applies_homography():
    img = canvas_with_line((30, 40), (150, 90))
    seg = detect_green_segments(img)[0]
    # 原图→校正图：x 放大 2 倍、y 平移 +10
    H_inv = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 10.0], [0.0, 0.0, 1.0]])
    mapped = map_segment(seg, H_inv)
    ends = sorted([mapped["p1"], mapped["p2"]])
    assert ends[0][0] == pytest.approx(60.0, abs=3.0)
    assert ends[1][0] == pytest.approx(300.0, abs=3.0)
    assert ends[0][1] == pytest.approx(50.0, abs=3.0)


def test_wire_quad_geometry():
    quad = segment_to_quad((0.0, 0.0), (10.0, 0.0), width_um=0.5, extend_um=0.5)
    xs = sorted({round(x, 6) for x, _ in quad})
    ys = sorted({round(y, 6) for _, y in quad})
    assert xs == pytest.approx([-0.5, 10.5])
    assert ys == pytest.approx([-0.25, 0.25])
    # 长边 = 线段 + 2×外延，短边 = 宽度
    assert math.dist(quad[0], quad[1]) == pytest.approx(11.0)
    assert math.dist(quad[1], quad[2]) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        segment_to_quad((0.0, 0.0), (0.0, 0.0))
    with pytest.raises(ValueError):
        segment_to_quad((0.0, 0.0), (1.0, 0.0), width_um=-0.5)
    with pytest.raises(ValueError):
        segment_to_quad((0.0, 0.0), (1.0, 0.0), extend_um=-0.1)


def test_px_to_um_factory_convention():
    # 像素中心 (x+0.5, H-y-0.5) → 局部坐标 → matrix@local + origin
    to_um = px_to_um_factory(100, np.eye(2) * 0.1, (350.0, 350.0))
    assert to_um((0.0, 0.0)) == pytest.approx((350.05, 359.95))
    assert to_um((10.0, 20.0)) == pytest.approx((351.05, 357.95))


def test_end_to_end_batch_wires(tmp_path):
    from semcorr.cli import main
    from semcorr.demo import make_demo_image

    root = tmp_path / "input"
    root.mkdir()
    path = root / "0303-11-01.tif"
    make_demo_image(path, n_rows=2, n_cols=2)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    cv2.line(img, (150, 150), (210, 180), (0, 255, 0), 2)
    cv2.imwrite(str(path), img)

    assert main(["--batch", str(root)]) == 0
    import json
    manifest = json.loads((root / "corrected/cad/cad_manifest.json").read_text())
    assert manifest["schema_version"] == 4
    assert manifest["n_wires"] == 1
    wires = manifest["images"][0]["wires"]
    assert len(wires) == 1
    w = wires[0]
    assert w["width_um"] == WIRE_WIDTH_UM
    assert w["extend_um"] == WIRE_EXTEND_UM
    # 长边 = 纳米线长 + 2×外延；短边 = 宽度
    quad = [tuple(q) for q in w["quad_um"]]
    sides = [math.dist(quad[i], quad[(i + 1) % 4]) for i in range(4)]
    assert max(sides) == pytest.approx(w["length_um"] + 2 * WIRE_EXTEND_UM,
                                       rel=1e-6)
    assert min(sides) == pytest.approx(WIRE_WIDTH_UM, rel=1e-6)
    # 中心应落在 0303-11-01 的 50 µm 小格（350..400, 350..400）内
    mid = ((w["p1_um"][0] + w["p2_um"][0]) / 2, (w["p1_um"][1] + w["p2_um"][1]) / 2)
    assert 350 <= mid[0] <= 400 and 350 <= mid[1] <= 400
    # 叠加层与 wires.csv
    assert (root / "corrected/diagnostics/0303-11-01_wires.png").is_file()
    csv_text = (root / "corrected/cad/wires.csv").read_text(encoding="utf-8-sig")
    assert len(csv_text.splitlines()) == 2          # 表头 + 1 根


def test_end_to_end_dxf_wire_layer(tmp_path):
    ezdxf = pytest.importorskip("ezdxf")
    from semcorr.cli import main
    from semcorr.demo import make_demo_image

    root = tmp_path / "input"
    root.mkdir()
    path = root / "0303-11-01.tif"
    make_demo_image(path, n_rows=2, n_cols=2)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    cv2.line(img, (150, 150), (210, 180), (0, 255, 0), 2)
    cv2.imwrite(str(path), img)
    template = tmp_path / "template.dxf"
    ezdxf.new("R2018").saveas(template)
    outdir = tmp_path / "DXF"

    assert main(["--batch", str(root), "--dxf",
                 "--dxf-template", str(template),
                 "--dxf-outdir", str(outdir)]) == 0
    doc = ezdxf.readfile(str(outdir / f"{root.name}.dxf"))
    wires = list(doc.modelspace().query('LWPOLYLINE[layer=="wire"]'))
    assert len(wires) == 1 and wires[0].closed
    assert len(list(wires[0].get_points("xy"))) == 4


def test_wire_params_rejected_outside_cad(tmp_path):
    from semcorr.cli import main
    from semcorr.demo import make_demo_image

    root = tmp_path / "input"
    root.mkdir()
    make_demo_image(root / "a.tif", n_rows=2, n_cols=2)
    assert main(["--batch", str(root), "--no-cad",
                 "--wire-width-um", "1.0"]) == 1
