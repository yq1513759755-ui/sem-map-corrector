"""Batch QC summary and neighbour-cell consistency checks.

After a batch run (or on an existing ``corrected/`` tree) write one table of
per-image quality status plus cross-image checks on shared registration marks.

Two checks:

* **neighbour** — adjacent 50 µm cells share edge crosses. Each tile maps its
  four centres to µm with the same similarity as CAD export; the shared mark's
  two mapped positions must agree.
* **repeat** — two acquisitions of the same cell (same region, different SEM
  sequence) must agree on every mark.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import numpy as np

from .cad import IDS, CELL_UM, fit_bottom_left, parse_region

DEFAULT_NEIGHBOR_TOL_UM = 0.05


def _esc_path(path):
    return html.escape(str(path))


def _image_size(path):
    from PIL import Image

    with Image.open(path) as handle:
        width, height = handle.size
    return int(width), int(height)


def _design_marks(region):
    x0, y0 = (float(v) for v in region["bottom_left_um"])
    return {
        "M1": (x0, y0 + CELL_UM),
        "M2": (x0 + CELL_UM, y0 + CELL_UM),
        "M3": (x0, y0),
        "M4": (x0 + CELL_UM, y0),
    }


def _corner_key(point, ndigits=6):
    return (round(float(point[0]), ndigits), round(float(point[1]), ndigits))


def load_image_records(corrected_dir):
    """One record per ``diagnostics/*_report.json``."""
    corrected_dir = Path(corrected_dir)
    diag = corrected_dir / "diagnostics"
    records = []
    if not diag.is_dir():
        return records
    for report_path in sorted(diag.glob("*_report.json")):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stem = report_path.name[: -len("_report.json")]
        points = {}
        for item in report.get("corrected_centers", {}).get("points", []):
            points[item["id"]] = (float(item["x"]), float(item["y"]))
        if len(points) != 4:
            points = {}
            for mark in report.get("marks", []):
                x, y = mark.get("detected_px", (None, None))
                if x is not None:
                    points[mark["id"]] = (float(x), float(y))
        refinement = {}
        for mark in report.get("self_check", {}).get("marks", []) or report.get("marks", []):
            refinement[mark["id"]] = mark.get("center_refinement", {}).get("method")
        records.append({
            "stem": stem,
            "report_path": report_path,
            "quality_status": report.get("quality_status"),
            "quality_warnings": list(report.get("quality_warnings") or []),
            "self_check_rms_px": (report.get("self_check") or {}).get("rms"),
            "self_check_n": (report.get("self_check") or {}).get("n_detected"),
            "method": report.get("method"),
            "points_px": points,
            "refinement": refinement,
            "input_sha256": report.get("input_sha256"),
            "region": None,
            "fit": None,
            "marks_um": None,
            "qc_note": "",
        })
    return records


def attach_regions_and_fits(records, pitch_um=50.0):
    """Parse filenames and fit pixel→µm for records that look CAD-ready."""
    for record in records:
        try:
            region = parse_region(record["stem"])
        except ValueError as exc:
            record["qc_note"] = f"命名无效：{exc}"
            continue
        record["region"] = region
        if record["quality_status"] != "PASS":
            record["qc_note"] = "非 PASS，不参与邻格对比"
            continue
        if len(record["points_px"]) != 4 or set(record["points_px"]) != set(IDS):
            record["qc_note"] = "缺少 M1..M4 中心"
            continue
        corrected = record["report_path"].parent.parent / f"{record['stem']}_corrected.tif"
        if not corrected.is_file():
            record["qc_note"] = "缺少校正图"
            continue
        try:
            _, height = _image_size(corrected)
            points = np.array([record["points_px"][name] for name in IDS], dtype=float)
            fit = fit_bottom_left(points, height, region["bottom_left_um"], pitch_um)
        except (OSError, ValueError) as exc:
            record["qc_note"] = f"拟合失败：{exc}"
            continue
        record["fit"] = fit
        record["marks_um"] = {
            name: tuple(fit["fitted_um"][i]) for i, name in enumerate(IDS)
        }
        record["design_um"] = _design_marks(region)


def find_mark_pairs(records):
    """Pairs of images that share at least one design corner."""
    pairs = []
    usable = [r for r in records if r.get("marks_um") and r.get("design_um")]
    for i, a in enumerate(usable):
        for b in usable[i + 1:]:
            a_corners = {_corner_key(pt): name for name, pt in a["design_um"].items()}
            b_corners = {_corner_key(pt): name for name, pt in b["design_um"].items()}
            common = sorted(set(a_corners) & set(b_corners))
            if not common:
                continue
            xa = a["region"]["bottom_left_um"]
            xb = b["region"]["bottom_left_um"]
            dx = abs(xa[0] - xb[0])
            dy = abs(xa[1] - xb[1])
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                kind = "repeat"
            elif (abs(dx - CELL_UM) < 1e-6 and dy < 1e-6) or (
                    abs(dy - CELL_UM) < 1e-6 and dx < 1e-6):
                kind = "neighbor"
            else:
                continue
            shared = [(a_corners[key], b_corners[key], key) for key in common]
            pairs.append({"a": a["stem"], "b": b["stem"], "kind": kind,
                          "shared": shared})
    return pairs


def evaluate_pairs(pairs, records, neighbor_tol_um=DEFAULT_NEIGHBOR_TOL_UM):
    by_stem = {r["stem"]: r for r in records}
    results = []
    for pair in pairs:
        a = by_stem[pair["a"]]
        b = by_stem[pair["b"]]
        for name_a, name_b, corner in pair["shared"]:
            pa = np.asarray(a["marks_um"][name_a], dtype=float)
            pb = np.asarray(b["marks_um"][name_b], dtype=float)
            delta = float(np.linalg.norm(pa - pb))
            design = np.asarray(a["design_um"][name_a], dtype=float)
            results.append({
                "kind": pair["kind"],
                "image_a": pair["a"],
                "image_b": pair["b"],
                "mark_a": name_a,
                "mark_b": name_b,
                "corner_x_um": corner[0],
                "corner_y_um": corner[1],
                "a_x_um": float(pa[0]), "a_y_um": float(pa[1]),
                "b_x_um": float(pb[0]), "b_y_um": float(pb[1]),
                "delta_um": delta,
                "design_x_um": float(design[0]), "design_y_um": float(design[1]),
                "status": "ok" if delta <= neighbor_tol_um else "mismatch",
            })
    results.sort(key=lambda row: (-row["delta_um"], row["image_a"], row["image_b"]))
    return results


def summarize(records, pair_results, neighbor_tol_um=DEFAULT_NEIGHBOR_TOL_UM):
    counts = {"PASS": 0, "WARN_REVIEW": 0, "other": 0, "unusable": 0}
    for record in records:
        status = record.get("quality_status") or "other"
        if status in counts:
            counts[status] += 1
        else:
            counts["other"] += 1
        if record.get("qc_note") and record.get("marks_um") is None:
            counts["unusable"] += 1
    mismatches = [row for row in pair_results if row["status"] != "ok"]
    return {
        "n_images": len(records),
        "counts": counts,
        "n_pair_checks": len(pair_results),
        "n_mismatches": len(mismatches),
        "neighbor_tol_um": neighbor_tol_um,
        "max_delta_um": max((row["delta_um"] for row in pair_results), default=None),
    }


def write_summary_csv(path, records):
    path = Path(path)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "image", "quality_status", "self_check_rms_px", "self_check_n",
            "method", "region", "um_per_px", "rotation_deg", "rms_um",
            "max_residual_um", "warnings", "qc_note", "report",
        ])
        for record in records:
            region = record.get("region") or {}
            fit = record.get("fit") or {}
            writer.writerow([
                record["stem"],
                record.get("quality_status") or "",
                record.get("self_check_rms_px"),
                record.get("self_check_n"),
                record.get("method") or "",
                f"{region.get('marker_code', '')}-{region.get('quadrant', '')}{region.get('sub_quadrant', '')}"
                if region else "",
                fit.get("scale_um_per_px"),
                fit.get("rotation_deg"),
                fit.get("rms_um"),
                fit.get("max_residual_um"),
                "; ".join(record.get("quality_warnings") or []),
                record.get("qc_note") or "",
                record["report_path"].name,
            ])


def write_pairs_csv(path, pair_results):
    path = Path(path)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "kind", "image_a", "image_b", "mark_a", "mark_b",
            "corner_x_um", "corner_y_um", "a_x_um", "a_y_um",
            "b_x_um", "b_y_um", "delta_um", "status",
        ])
        for row in pair_results:
            writer.writerow([
                row["kind"], row["image_a"], row["image_b"], row["mark_a"], row["mark_b"],
                row["corner_x_um"], row["corner_y_um"], row["a_x_um"], row["a_y_um"],
                row["b_x_um"], row["b_y_um"], row["delta_um"], row["status"],
            ])


def write_html(path, records, pair_results, summary, neighbor_tol_um=DEFAULT_NEIGHBOR_TOL_UM,
               title="SEM 批次质检汇总", subtitle=""):
    path = Path(path)

    def esc(value):
        return html.escape("" if value is None else str(value))

    def fmt(value, digits=4):
        if value is None or value == "":
            return "—"
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return esc(value)

    def status_badge(status):
        mapping = {
            "PASS": ("ok", "PASS"),
            "WARN_REVIEW": ("warn", "需复核"),
            "ok": ("ok", "一致"),
            "mismatch": ("bad", "不一致"),
        }
        cls, label = mapping.get(status, ("warn", status or "—"))
        return f'<span class="badge {cls}">{esc(label)}</span>'

    def delta_bar(delta, tol):
        """Tiny bar: filled fraction = delta / (3*tol), red when over tol."""
        if delta is None:
            return "—"
        cap = max(float(tol) * 3.0, 1e-9)
        pct = max(2.0, min(100.0, 100.0 * float(delta) / cap))
        over = float(delta) > float(tol)
        cls = "over" if over else "ok"
        return (
            f'<div class="bar" title="{delta:.4f} µm / 门限 {tol:g} µm">'
            f'<i class="{cls}" style="width:{pct:.1f}%"></i></div>'
        )

    rows = []
    for record in records:
        fit = record.get("fit") or {}
        region = record.get("region") or {}
        region_label = (
            f"{region.get('marker_code')}-{region.get('quadrant')}{region.get('sub_quadrant')}"
            if region else "—"
        )
        note = "; ".join(record.get("quality_warnings") or []) or (record.get("qc_note") or "")
        status = record.get("quality_status") or "other"
        row_cls = {
            "PASS": "row-ok", "WARN_REVIEW": "row-warn",
        }.get(status, "row-bad" if status not in ("ok",) and record.get("qc_note") else "")
        residual = fit.get("max_residual_um")
        resid_cls = ""
        if residual is not None and residual > neighbor_tol_um:
            resid_cls = "num-bad"
        rows.append(
            f'<tr class="{row_cls}">'
            f"<td class=mono>{esc(record['stem'])}</td>"
            f"<td>{status_badge(status)}</td>"
            f"<td class=num>{fmt(record.get('self_check_rms_px'), 4)}</td>"
            f"<td class=mono>{esc(region_label)}</td>"
            f"<td class=num>{fmt(fit.get('scale_um_per_px'), 5)}</td>"
            f"<td class='num {resid_cls}'>{fmt(residual, 4)}</td>"
            f"<td class=note>{esc(note) or '—'}</td>"
            "</tr>"
        )

    pair_rows = []
    for row in pair_results:
        kind_label = "邻格" if row["kind"] == "neighbor" else "重复采集"
        pair_rows.append(
            f'<tr class="{"row-bad" if row["status"] != "ok" else "row-ok"}">'
            f"<td><span class='pill kind-{esc(row['kind'])}'>{esc(kind_label)}</span></td>"
            f"<td class=mono>{esc(row['image_a'])}<span class=dim> ↔ </span>{esc(row['image_b'])}</td>"
            f"<td class=mono>{esc(row['mark_a'])}<span class=dim>→</span>{esc(row['mark_b'])}</td>"
            f"<td class=num>({row['corner_x_um']:g}, {row['corner_y_um']:g})</td>"
            f"<td class=num>{row['delta_um']:.4f}</td>"
            f"<td class=barcell>{delta_bar(row['delta_um'], neighbor_tol_um)}</td>"
            f"<td>{status_badge(row['status'])}</td>"
            "</tr>"
        )

    max_delta = summary.get("max_delta_um")
    max_text = "—" if max_delta is None else f"{max_delta:.4f}"
    mismatches = summary.get("n_mismatches", 0)
    overall = "ok" if mismatches == 0 else "bad"
    overall_label = "全部通过" if mismatches == 0 else f"{mismatches} 项不一致"
    n_pass = summary.get("counts", {}).get("PASS", 0)
    n_warn = summary.get("counts", {}).get("WARN_REVIEW", 0)
    n_other = summary.get("counts", {}).get("other", 0)

    body = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{esc(title)}</title>
<style>
:root {{
  --bg: #f4f1ea; --paper: #fffdf8; --ink: #1c1b18; --muted: #6b6560;
  --line: #e4ddd2; --accent: #2f5d50; --ok: #2f6b4f; --ok-bg: #e5f2ea;
  --warn: #8a6d00; --warn-bg: #fff3cd; --bad: #9b2c2c; --bad-bg: #f8e5e5;
  --mono: ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; padding: 28px 20px 48px;
  font-family: -apple-system, 'PingFang SC', 'Microsoft YaHei', sans-serif;
  color: var(--ink); background:
    radial-gradient(circle at top left, #efe8da 0, transparent 40%),
    var(--bg);
  line-height: 1.45;
}}
.wrap {{ max-width: 1100px; margin: 0 auto; }}
header {{
  background: var(--paper); border: 1px solid var(--line); border-radius: 14px;
  padding: 22px 24px 18px; margin-bottom: 18px;
  box-shadow: 0 1px 0 rgba(28,27,24,.04);
}}
.eyebrow {{
  font-size: 12px; letter-spacing: .12em; text-transform: uppercase;
  color: var(--accent); font-weight: 600; margin-bottom: 6px;
}}
h1 {{ margin: 0 0 6px; font-size: 26px; font-weight: 650; letter-spacing: .01em; }}
.sub {{ color: var(--muted); font-size: 13px; }}
.sub .mono {{ color: var(--ink); }}
.verdict {{
  display: inline-flex; align-items: center; gap: 8px; margin-top: 14px;
  padding: 6px 12px; border-radius: 999px; font-size: 13px; font-weight: 600;
}}
.verdict.ok {{ background: var(--ok-bg); color: var(--ok); }}
.verdict.bad {{ background: var(--bad-bg); color: var(--bad); }}
.grid {{
  display: grid; grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 10px; margin: 0 0 22px;
}}
@media (max-width: 900px) {{ .grid {{ grid-template-columns: repeat(3, 1fr); }} }}
.card {{
  background: var(--paper); border: 1px solid var(--line); border-radius: 12px;
  padding: 14px 14px 12px; min-height: 84px;
}}
.card .n {{ font-size: 24px; font-weight: 650; font-variant-numeric: tabular-nums; }}
.card .l {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
.card.accent .n {{ color: var(--accent); }}
.card.warn .n {{ color: var(--warn); }}
.card.bad .n {{ color: var(--bad); }}
section {{
  background: var(--paper); border: 1px solid var(--line); border-radius: 14px;
  padding: 8px 0 4px; margin-bottom: 18px; overflow: hidden;
}}
section h2 {{
  margin: 0; padding: 14px 20px 10px; font-size: 15px; font-weight: 650;
  border-bottom: 1px solid var(--line);
  display: flex; justify-content: space-between; align-items: baseline; gap: 12px;
}}
section h2 span {{ font-size: 12px; color: var(--muted); font-weight: 400; }}
section p.hint {{ margin: 0; padding: 10px 20px 0; color: var(--muted); font-size: 12px; }}
.scroll {{ overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
th, td {{ padding: 9px 12px; text-align: left; vertical-align: middle; }}
thead th {{
  position: sticky; top: 0; background: #f7f2e8; color: var(--muted);
  font-weight: 600; font-size: 11.5px; letter-spacing: .02em;
  border-bottom: 1px solid var(--line);
}}
tbody tr {{ border-top: 1px solid var(--line); }}
tbody tr:hover {{ background: #faf6ee; }}
tr.row-ok td:first-child {{ box-shadow: inset 3px 0 0 var(--ok); }}
tr.row-warn td:first-child {{ box-shadow: inset 3px 0 0 var(--warn); }}
tr.row-bad td:first-child {{ box-shadow: inset 3px 0 0 var(--bad); }}
.mono {{ font-family: var(--mono); font-size: 12px; }}
.num {{ font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; }}
.num-bad {{ color: var(--bad); font-weight: 600; }}
.note {{ color: var(--muted); max-width: 280px; }}
.dim {{ color: #b0a89c; padding: 0 2px; }}
.badge {{
  display: inline-block; padding: 2px 8px; border-radius: 999px;
  font-size: 11px; font-weight: 650; letter-spacing: .02em;
}}
.badge.ok {{ background: var(--ok-bg); color: var(--ok); }}
.badge.warn {{ background: var(--warn-bg); color: var(--warn); }}
.badge.bad {{ background: var(--bad-bg); color: var(--bad); }}
.pill {{
  display: inline-block; padding: 2px 8px; border-radius: 999px;
  font-size: 11px; font-weight: 600; background: #eef2ef; color: var(--accent);
}}
.pill.kind-repeat {{ background: #ebe8f5; color: #4a3f7a; }}
.bar {{ width: 88px; height: 8px; background: #efe8da; border-radius: 99px; overflow: hidden; }}
.bar i {{ display: block; height: 100%; border-radius: 99px; }}
.bar i.ok {{ background: #2f6b4f; }}
.bar i.over {{ background: #9b2c2c; }}
.barcell {{ width: 100px; }}
footer {{
  color: var(--muted); font-size: 12px; text-align: center; margin-top: 8px;
}}
.empty {{ padding: 18px 20px; color: var(--muted); }}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div class="eyebrow">SEM-MAP · Batch QC</div>
  <h1>{esc(title)}</h1>
  <div class="sub">{esc(subtitle)}</div>
  <div class="verdict {overall}">{esc(overall_label)} · 共享 mark 门限 {neighbor_tol_um:g} µm</div>
</header>

<div class="grid">
  <div class="card accent"><div class="n">{summary['n_images']}</div><div class="l">图像</div></div>
  <div class="card"><div class="n">{n_pass}</div><div class="l">PASS</div></div>
  <div class="card {'warn' if n_warn else ''}"><div class="n">{n_warn}</div><div class="l">需复核</div></div>
  <div class="card"><div class="n">{summary['n_pair_checks']}</div><div class="l">共享 mark 对比</div></div>
  <div class="card {'bad' if mismatches else ''}"><div class="n">{mismatches}</div><div class="l">不一致</div></div>
  <div class="card"><div class="n">{esc(max_text)}</div><div class="l">最大偏差 (µm)</div></div>
</div>

<section>
  <h2>逐图状态 <span>PASS {n_pass} · 需复核 {n_warn} · 其他 {n_other}</span></h2>
  <div class="scroll">
  <table>
    <thead>
      <tr>
        <th>图像</th><th>状态</th><th class=num>self_check RMS</th>
        <th>区域</th><th class=num>µm/px</th><th class=num>最大残差 µm</th><th>备注</th>
      </tr>
    </thead>
    <tbody>
{chr(10).join(rows) if rows else '<tr><td colspan="7" class="empty">无报告</td></tr>'}
    </tbody>
  </table>
  </div>
</section>

<section>
  <h2>邻格 / 重复采集一致性 <span>Δ ≤ {neighbor_tol_um:g} µm 为通过</span></h2>
  <p class="hint">相邻 50 µm 小格共享边十字；同格不同序号为重复采集。Δ 为两图映射到 µm 后的偏差，条越短越好。</p>
  <div class="scroll">
  <table>
    <thead>
      <tr>
        <th>类型</th><th>图像</th><th>mark</th>
        <th class=num>设计角点 µm</th><th class=num>Δ µm</th><th>相对门限</th><th>状态</th>
      </tr>
    </thead>
    <tbody>
{chr(10).join(pair_rows) if pair_rows else '<tr><td colspan="7" class="empty">本批次没有可对比的共享 mark（邻格或同格重复）</td></tr>'}
    </tbody>
  </table>
  </div>
</section>

<footer>配准残差与邻格偏差不是实际曝光套刻精度 · 由 semcorr batch_qc 生成</footer>
</div>
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")


def run_batch_qc(folder, *, corrected_dir=None, outdir=None,
                 neighbor_tol_um=DEFAULT_NEIGHBOR_TOL_UM, pitch_um=50.0):
    """Write batch_qc.csv / neighbor_checks.csv / batch_qc.html under outdir."""
    folder = Path(folder)
    corrected_dir = Path(corrected_dir) if corrected_dir else folder / "corrected"
    outdir = Path(outdir) if outdir else corrected_dir
    outdir.mkdir(parents=True, exist_ok=True)

    records = load_image_records(corrected_dir)
    attach_regions_and_fits(records, pitch_um=pitch_um)
    pairs = find_mark_pairs(records)
    pair_results = evaluate_pairs(pairs, records, neighbor_tol_um=neighbor_tol_um)
    summary = summarize(records, pair_results, neighbor_tol_um=neighbor_tol_um)

    write_summary_csv(outdir / "batch_qc.csv", records)
    write_pairs_csv(outdir / "neighbor_checks.csv", pair_results)
    write_html(
        outdir / "batch_qc.html", records, pair_results, summary, neighbor_tol_um,
        title=f"SEM 批次质检 · {folder.name}",
        subtitle=(
            f'<span class="mono">{_esc_path(folder)}</span>'
            f' · {len(records)} 张 · 校正目录 <span class="mono">{_esc_path(corrected_dir)}</span>'
        ),
    )

    payload = {
        "schema_version": 1,
        "folder": str(folder.resolve()),
        "corrected_dir": str(corrected_dir.resolve()),
        "summary": summary,
        "images": [{
            "stem": r["stem"],
            "quality_status": r.get("quality_status"),
            "self_check_rms_px": r.get("self_check_rms_px"),
            "region": r.get("region"),
            "qc_note": r.get("qc_note"),
            "fit": ({k: r["fit"][k] for k in
                     ("scale_um_per_px", "rotation_deg", "rms_um", "max_residual_um")}
                    if r.get("fit") else None),
        } for r in records],
        "pair_checks": pair_results,
    }
    (outdir / "batch_qc.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
