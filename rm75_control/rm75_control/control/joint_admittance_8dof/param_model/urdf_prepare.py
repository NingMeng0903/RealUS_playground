"""Prepare RM75 genesis URDF: absolute mesh paths + strip white URDF materials (DAE colors)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from xml.etree import ElementTree as ET

from rm75_control.control.joint_admittance_8dof.param_model.paths import ASSETS_DIR, URDF_CACHE_DIR


def package_assets_dir() -> Path:
    return ASSETS_DIR


def _resolve_mesh_path(source_urdf: Path, mesh_filename: str) -> Path:
    raw = str(mesh_filename).strip()
    if raw.startswith("package://"):
        rest = raw[len("package://") :]
        idx = rest.find("/")
        package_name = rest[:idx] if idx >= 0 else ""
        raw = rest[idx + 1 :] if idx >= 0 else rest
        for root in (source_urdf.parent, source_urdf.parent.parent, source_urdf.parent.parent.parent):
            if package_name and root.name != package_name and not (root / "package.xml").is_file():
                continue
            candidate = (root / raw).resolve()
            if candidate.exists():
                return candidate
    return (source_urdf.parent / raw).resolve()


def prepare_genesis_urdf(
    source_urdf: Path,
    *,
    link_rgba: dict[str, tuple[float, float, float, float]] | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Rewrite mesh paths to absolute; strip visual materials so Collada shading wins."""
    cache_root = cache_dir if cache_dir is not None else URDF_CACHE_DIR
    link_rgba = dict(link_rgba or {})
    try:
        st = source_urdf.stat()
        stat_payload = (int(st.st_size), int(st.st_mtime_ns))
    except OSError:
        stat_payload = (0, 0)
    ordered = tuple(
        sorted(
            (_ln, tuple(round(float(x), 6) for x in rgba))
            for _ln, rgba in sorted(link_rgba.items())
        )
    )
    cache_key = hashlib.sha1(
        repr(("rm75_8dof_gvis", "keep_box_mat_v2", str(source_urdf.resolve()), stat_payload, ordered)).encode(
            "utf-8"
        )
    ).hexdigest()[:16]
    out = cache_root / f"{source_urdf.stem}_gvis_{cache_key}.urdf"
    if out.is_file():
        return out

    tree = ET.parse(source_urdf)
    root_el = tree.getroot()
    for mesh in root_el.findall(".//mesh"):
        fn = mesh.attrib.get("filename")
        if not fn:
            continue
        mesh_path = Path(fn)
        if not mesh_path.is_absolute() or str(fn).startswith("package://"):
            mesh.set("filename", str(_resolve_mesh_path(source_urdf, fn)))

    for link_el in root_el.findall("link"):
        link_name = str(link_el.attrib.get("name") or "").strip()
        if not link_name:
            continue
        for visual in link_el.findall("visual"):
            has_mesh = visual.find(".//mesh") is not None
            if has_mesh:
                for material_el in list(visual.findall("material")):
                    visual.remove(material_el)
            rgba_tpl = link_rgba.get(link_name)
            if rgba_tpl is None:
                continue
            r, g, b, a = (float(rgba_tpl[i]) for i in range(4))
            mat_el = ET.SubElement(visual, "material")
            mat_el.set("name", "")
            color_el = ET.SubElement(mat_el, "color")
            color_el.set("rgba", f"{r:.8f} {g:.8f} {b:.8f} {a:.8f}")

    cache_root.mkdir(parents=True, exist_ok=True)
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out
