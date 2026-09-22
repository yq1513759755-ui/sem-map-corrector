"""Batch QC summary and neighbour consistency checks."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from semcorr.batch_qc import (
    evaluate_pairs,
    find_mark_pairs,
    load_image_records,
    attach_regions_and_fits,
    run_batch_qc,
)
from semcorr.cad import parse_region


def _write_case(root, stem, height, width, points_px):
    """Minimal on-disk case: blank corrected image + PASS report."""
    folder = Path(root)
    (folder / "corrected" / "diagnostics").mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), np.uint8)
    for x, y in points_px:
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < width and 0 <= iy < height:
            image[iy, ix] = (0, 0, 255)
    cv2.imwrite(str(folder / "corrected" / f"{stem}_corrected.tif"), image)
    points = []
    for name, (x, y) in zip(("M1", "M2", "M3", "M4"), points_px):
        points.append({"id": name, "x": x, "y": y, "source": "self-check"})
    report = {
        "quality_status": "PASS",
        "quality_warnings": [],
        "method": "exact-square",
        "grid": [2, 2],
        "input_sha256": "0" * 64,
        "self_check": {
            "n_detected": 4, "n_total": 4, "rms": 0.05,
            "marks": [{"id": n, "detected": [x, y],
                       "center_refinement": {"method": "arm-edges"}}
                      for n, (x, y) in zip(("M1", "M2", "M3", "M4"), points_px)],
        },
        "corrected_centers": {"unit": "px", "points": points},
        "marks": [{"id": n, "detected_px": [x, y],
                   "center_refinement": {"method": "arm-edges"}}
                  for n, (x, y) in zip(("M1", "M2", "M3", "M4"), points_px)],
    }
    (folder / "corrected" / "diagnostics" / f"{stem}_report.json").write_text(
        json.dumps(report), encoding="utf-8")


def _points_for_cell(x0, y0, height, width, um_per_px=0.1, jitter_px=None):
    """Pixel centres for M1..M4 of cell (x0,y0) under image-up = +Y.

    ``jitter_px`` is an optional per-mark (dx, dy) offset — a global shift is
    absorbed by the similarity fit and cannot create cross-image disagreement.
    """
    scale = um_per_px
    jitter_px = jitter_px or [(0.0, 0.0)] * 4

    def px_from_um(xu, yu):
        px = (xu - x0) / scale + 20.0
        py = height - ((yu - y0) / scale) - 20.0
        return (px, py)

    corners = [
        (x0, y0 + 50),
        (x0 + 50, y0 + 50),
        (x0, y0),
        (x0 + 50, y0),
    ]
    out = []
    for (xu, yu), (jx, jy) in zip(corners, jitter_px):
        px, py = px_from_um(xu, yu)
        out.append((px + jx, py + jy))
    return out


def test_neighbor_pair_and_agreement(tmp_path):
    height, width = 400, 400
    left = parse_region("0101-11")
    right = parse_region("0101-12")
    assert left["bottom_left_um"] != right["bottom_left_um"]
    _write_case(tmp_path, "0101-11-01", height, width,
                _points_for_cell(*left["bottom_left_um"], height, width))
    # ~0.02 µm random jitter (0.2 px at 0.1 µm/px) stays under the 0.05 µm gate.
    _write_case(tmp_path, "0101-12-01", height, width,
                _points_for_cell(*right["bottom_left_um"], height, width,
                                 jitter_px=[(0.2, -0.1), (-0.1, 0.15),
                                            (0.05, 0.05), (-0.15, -0.1)]))

    payload = run_batch_qc(tmp_path, neighbor_tol_um=0.05)
    pairs = payload["pair_checks"]
    assert pairs, "应当找到共享 mark 对"
    assert all(row["kind"] in ("neighbor", "repeat") for row in pairs)
    assert payload["summary"]["n_mismatches"] == 0
    assert (tmp_path / "corrected" / "batch_qc.html").is_file()
    assert (tmp_path / "corrected" / "batch_qc.csv").is_file()
    assert (tmp_path / "corrected" / "neighbor_checks.csv").is_file()
    html = (tmp_path / "corrected" / "batch_qc.html").read_text(encoding="utf-8")
    assert "邻格" in html
    assert "0101-11-01" in html


def test_neighbor_mismatch_is_flagged(tmp_path):
    height, width = 400, 400
    left = parse_region("0101-11")
    right = parse_region("0101-12")
    _write_case(tmp_path, "0101-11-01", height, width,
                _points_for_cell(*left["bottom_left_um"], height, width))
    # Per-mark jitter of several pixels (~0.3 µm) must breach the 0.05 µm gate.
    _write_case(tmp_path, "0101-12-01", height, width,
                _points_for_cell(*right["bottom_left_um"], height, width,
                                 jitter_px=[(3.0, 0.0), (-3.0, 2.0),
                                            (2.5, -2.0), (0.0, 3.0)]))
    payload = run_batch_qc(tmp_path, neighbor_tol_um=0.05)
    assert payload["summary"]["n_mismatches"] >= 1
    assert any(row["status"] == "mismatch" for row in payload["pair_checks"])


def test_repeat_acquisitions_compare_all_marks(tmp_path):
    height, width = 400, 400
    cell = parse_region("0101-11")
    _write_case(tmp_path, "0101-11-01", height, width,
                _points_for_cell(*cell["bottom_left_um"], height, width))
    _write_case(tmp_path, "0101-11-02", height, width,
                _points_for_cell(*cell["bottom_left_um"], height, width,
                                 jitter_px=[(0.3, 0.0), (-0.2, 0.1),
                                            (0.1, -0.2), (0.0, 0.15)]))
    records = load_image_records(tmp_path / "corrected")
    attach_regions_and_fits(records)
    pairs = find_mark_pairs(records)
    assert len(pairs) == 1
    assert pairs[0]["kind"] == "repeat"
    assert len(pairs[0]["shared"]) == 4
    results = evaluate_pairs(pairs, records, neighbor_tol_um=0.05)
    assert len(results) == 4


def test_non_neighbor_cells_are_not_paired(tmp_path):
    height, width = 400, 400
    a = parse_region("0101-11")
    b = parse_region("0101-44")
    dx = abs(a["bottom_left_um"][0] - b["bottom_left_um"][0])
    dy = abs(a["bottom_left_um"][1] - b["bottom_left_um"][1])
    # Not side-adjacent (dx,dy) == (50,0) or (0,50).
    assert not ((abs(dx - 50) < 1e-6 and dy < 1e-6) or
                (abs(dy - 50) < 1e-6 and dx < 1e-6))
    _write_case(tmp_path, "0101-11-01", height, width,
                _points_for_cell(*a["bottom_left_um"], height, width))
    _write_case(tmp_path, "0101-44-01", height, width,
                _points_for_cell(*b["bottom_left_um"], height, width))
    records = load_image_records(tmp_path / "corrected")
    attach_regions_and_fits(records)
    assert find_mark_pairs(records) == []


def test_qc_only_cli(tmp_path):
    height, width = 400, 400
    cell = parse_region("0101-11")
    _write_case(tmp_path, "0101-11-01", height, width,
                _points_for_cell(*cell["bottom_left_um"], height, width))
    from semcorr.cli import main

    code = main(["--batch", str(tmp_path), "--qc-only"])
    assert code == 0
    assert (tmp_path / "corrected" / "batch_qc.json").is_file()


def test_invalid_filename_is_noted_not_crashing(tmp_path):
    height, width = 200, 200
    points = [(20.0, 20.0), (120.0, 20.0), (20.0, 120.0), (120.0, 120.0)]
    _write_case(tmp_path, "not-a-region", height, width, points)
    payload = run_batch_qc(tmp_path)
    assert payload["summary"]["n_images"] == 1
    assert payload["pair_checks"] == []
    assert payload["images"][0]["qc_note"]
