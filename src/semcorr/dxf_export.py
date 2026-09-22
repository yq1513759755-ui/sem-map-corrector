"""Optional AutoCAD DXF merge: stamp SEM underlays into a fixed layout template.

Not part of the default lab workflow (``--dxf``). Reads a CAD bundle produced
by :mod:`semcorr.cad` and inserts each IMAGE with the same outer-origin /
per-pixel vectors as ``sem_map.lsp`` (DXF groups 10/11/12), then saves a new
DXF R2018 file. Marker template content is preserved.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import ezdxf

# Personal defaults (override with --dxf-template / --dxf-outdir).
DEFAULT_TEMPLATE = Path("~/PhD/dxf-gds/RAW/Marker.dxf").expanduser()
DEFAULT_OUTDIR = Path("~/PhD/dxf-gds/DXF").expanduser()


def _require_ezdxf():
    try:
        import ezdxf  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "需要 ezdxf 才能导出 DXF：uv pip install ezdxf（或 pip install 'sem-map-corrector[dxf]'）"
        ) from exc


def _relpath_for_dxf(image_path: Path, dxf_path: Path) -> str:
    """Relative path from the DXF file to the image (AutoCAD external ref)."""
    try:
        rel = os.path.relpath(image_path.resolve(), dxf_path.parent.resolve())
    except ValueError:
        return str(image_path.resolve())
    return rel.replace(os.sep, "/")


def export_dxf_from_cad(cad_dir, *, template, outdir, name=None):
    """Write ``outdir/<name>.dxf`` from ``cad_dir/cad_manifest.json``.

    Returns dict with output path and per-image handles. Requires a completed
    CAD bundle (``cad/images/*.tif`` + manifest).
    """
    _require_ezdxf()
    cad_dir = Path(cad_dir).resolve()
    template = Path(template).expanduser().resolve()
    outdir = Path(outdir).expanduser().resolve()
    manifest_path = cad_dir / "cad_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"缺少 CAD 贴图包清单: {manifest_path}")
    if not template.is_file():
        raise RuntimeError(f"DXF 模板不存在: {template}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    images = manifest.get("images") or []
    if not images:
        raise RuntimeError("CAD 贴图包中没有可插入的图像")

    outdir.mkdir(parents=True, exist_ok=True)
    stem = name or cad_dir.parent.name or "sem_map"
    out_path = outdir / f"{stem}.dxf"

    doc = ezdxf.readfile(template)
    # Template is already AC1032 (R2018); force the export version explicitly.
    if doc.dxfversion != ezdxf.DXF2018:
        try:
            doc.dxfversion = ezdxf.DXF2018
        except Exception as exc:
            raise RuntimeError(f"无法将模板升级为 DXF 2018: {exc}") from exc

    msp = doc.modelspace()
    inserted = []
    for row in images:
        bundle = cad_dir / row["bundle_image"]
        if not bundle.is_file():
            raise RuntimeError(f"缺少贴图图像: {bundle}")
        width = int(row["width_px"])
        height = int(row["height_px"])
        origin = list(row["origin_um"])
        u = list(row["u_um"])
        v = list(row["v_um"])
        rel = _relpath_for_dxf(bundle, out_path)
        image_def = doc.add_image_def(filename=rel, size_in_pixel=(width, height))
        scale = float((u[0] ** 2 + u[1] ** 2) ** 0.5) or 1.0
        rotation = math.degrees(math.atan2(u[1], u[0])) if abs(u[0]) + abs(u[1]) > 0 else 0.0
        image = msp.add_image(
            image_def=image_def,
            insert=(origin[0], origin[1], 0.0),
            size_in_units=(width * scale, height * scale),
            rotation=rotation,
            dxfattribs={"layer": "0"},
        )
        # Same as sem_map.lsp: explicit outer origin + per-pixel vectors (10/11/12).
        image.dxf.insert = (float(origin[0]), float(origin[1]), 0.0)
        image.dxf.u_pixel = (float(u[0]), float(u[1]), 0.0)
        image.dxf.v_pixel = (float(v[0]), float(v[1]), 0.0)
        inserted.append({
            "id": row.get("id"),
            "name": row.get("name"),
            "image": rel,
            "origin_um": origin,
            "u_um": u,
            "v_um": v,
        })

    doc.saveas(out_path)
    return {
        "output": str(out_path),
        "template": str(template),
        "n_images": len(inserted),
        "images": inserted,
        "dxf_version": "R2018",
    }
