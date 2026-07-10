"""Shared URDF rewrite for scene ``probe_collision`` (convex hull / cylinder) used by Genesis sims."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from common.project import project_paths

from projects.genesis_ue_sync.sim_platform.scenes.common_scene import SyncSceneSpec


def _repo_root() -> Path:
    return project_paths(__file__).root


def _collision_cache_key(
    *,
    source_urdf: Path,
    link_name: str,
    shape: str,
    radius: float,
    length: float,
    origin_xyz: tuple[float, float, float] | None,
    origin_rpy: tuple[float, float, float] | None,
    mesh_filename: str | None,
) -> str:
    payload = {
        "source_urdf": str(source_urdf.expanduser().resolve()),
        "link_name": str(link_name),
        "shape": str(shape),
        "radius": float(radius),
        "length": float(length),
        "origin_xyz": None if origin_xyz is None else [float(v) for v in origin_xyz],
        "origin_rpy": None if origin_rpy is None else [float(v) for v in origin_rpy],
        "mesh_filename": str(mesh_filename or ""),
    }
    try:
        st = source_urdf.stat()
        payload["source_urdf_stat"] = {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}
    except OSError:
        pass
    return hashlib.sha1(repr(sorted(payload.items())).encode("utf-8")).hexdigest()[:16]


def _resolve_urdf_mesh_path(source_urdf: Path, mesh_filename: str, repo_root: Path | None = None) -> Path:
    raw = str(mesh_filename).strip()
    if raw.startswith("package://"):
        rest = raw[len("package://") :]
        idx = rest.find("/")
        package_name = rest[:idx] if idx >= 0 else ""
        rel = rest[idx + 1 :] if idx >= 0 else rest
        for root in (source_urdf.parent, source_urdf.parent.parent, source_urdf.parent.parent.parent):
            if package_name and root.name != package_name and not (root / "package.xml").is_file():
                continue
            candidate = (root / rel).resolve()
            if candidate.is_file():
                return candidate
        root_out = repo_root if repo_root is not None else _repo_root()
        assets = root_out / "assets" / "robots"
        if package_name and assets.is_dir():
            for candidate in sorted(assets.glob(f"**/{package_name}/{rel}")):
                if candidate.is_file():
                    return candidate.resolve()
    candidate = Path(raw)
    if candidate.is_file():
        return candidate.resolve()
    return (source_urdf.parent / raw).resolve()


def apply_probe_collision_to_urdf(
    source_urdf: Path,
    *,
    link_name: str,
    shape: str = "cylinder",
    radius: float = 0.024,
    length: float = 0.14,
    origin_xyz: tuple[float, float, float] | None = None,
    origin_rpy: tuple[float, float, float] | None = None,
    mesh_filename: str | None = None,
    repo_root: Path | None = None,
) -> Path:
    root_out = repo_root if repo_root is not None else _repo_root()
    shape_norm = str(shape or "cylinder").strip().lower()
    cache_key = _collision_cache_key(
        source_urdf=source_urdf,
        link_name=link_name,
        shape=shape_norm,
        radius=radius,
        length=length,
        origin_xyz=origin_xyz,
        origin_rpy=origin_rpy,
        mesh_filename=mesh_filename,
    )
    if shape_norm == "convex_hull_mesh":
        out = (
            root_out
            / "outputs"
            / "genesis_robot_urdf_cache"
            / f"{source_urdf.stem}_{link_name}_col_mesh_{cache_key}.urdf"
        )
    else:
        out = (
            root_out
            / "outputs"
            / "genesis_robot_urdf_cache"
            / f"{source_urdf.stem}_{link_name}_col_cylinder_{cache_key}.urdf"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(source_urdf)
    root = tree.getroot()
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if not filename:
            continue
        mesh_path = Path(filename)
        if not mesh_path.is_absolute() or str(filename).startswith("package://"):
            mesh.set("filename", str(_resolve_urdf_mesh_path(source_urdf, filename)))

    target_link_el = None
    for link in root.findall("link"):
        if link.attrib.get("name") == link_name:
            target_link_el = link
            break
    if target_link_el is None:
        raise ValueError(f"probe_collision link_name {link_name!r} not found in {source_urdf}")
    if shape_norm == "convex_hull_mesh":
        rel_mesh = (mesh_filename or "").strip()
        visual_mesh_scale = None
        if not rel_mesh:
            visual = target_link_el.find("visual")
            vgeom = visual.find("geometry") if visual is not None else None
            vm = vgeom.find("mesh") if vgeom is not None else None
            if vm is not None and vm.attrib.get("filename"):
                rel_mesh = str(vm.attrib["filename"])
                visual_mesh_scale = vm.attrib.get("scale")
        if not rel_mesh:
            raise ValueError(
                f"convex_hull_mesh probe_collision requires mesh_filename or a visual mesh on link {link_name!r}."
            )
        mesh_abs = Path(rel_mesh)
        if not mesh_abs.is_absolute() or rel_mesh.startswith("package://"):
            mesh_abs = _resolve_urdf_mesh_path(source_urdf, rel_mesh)
        collision = target_link_el.find("collision")
        if collision is None:
            collision = ET.SubElement(target_link_el, "collision")
        geometry = collision.find("geometry")
        if geometry is None:
            geometry = ET.SubElement(collision, "geometry")
        for child in list(geometry):
            geometry.remove(child)
        mesh_el = ET.SubElement(geometry, "mesh")
        mesh_el.set("filename", str(mesh_abs))
        if visual_mesh_scale:
            mesh_el.set("scale", str(visual_mesh_scale))
        origin = collision.find("origin")
        if origin is None and (origin_xyz is not None or origin_rpy is not None):
            origin = ET.SubElement(collision, "origin")
        if origin is not None:
            if origin_xyz is not None:
                origin.set("xyz", " ".join(f"{float(v):.6f}" for v in origin_xyz))
            if origin_rpy is not None:
                origin.set("rpy", " ".join(f"{float(v):.6f}" for v in origin_rpy))
    else:
        for link in root.findall("link"):
            if link.attrib.get("name") != link_name:
                continue
            collision = link.find("collision")
            if collision is None:
                collision = ET.SubElement(link, "collision")
            geometry = collision.find("geometry")
            if geometry is None:
                geometry = ET.SubElement(collision, "geometry")
            for child in list(geometry):
                geometry.remove(child)
            cylinder = ET.SubElement(geometry, "cylinder")
            cylinder.set("radius", f"{float(radius):.6f}")
            cylinder.set("length", f"{float(length):.6f}")
            origin = collision.find("origin")
            if origin is None and (origin_xyz is not None or origin_rpy is not None):
                origin = ET.SubElement(collision, "origin")
            if origin is not None:
                if origin_xyz is not None:
                    origin.set("xyz", " ".join(f"{float(v):.6f}" for v in origin_xyz))
                if origin_rpy is not None:
                    origin.set("rpy", " ".join(f"{float(v):.6f}" for v in origin_rpy))
            break

    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out


def _urdf_uses_package_uri(path: Path) -> bool:
    try:
        return "package://" in path.read_text(encoding="utf-8")
    except OSError:
        return False


def _urdf_needs_absolute_mesh_materialization(path: Path) -> bool:
    """True when Genesis cannot resolve mesh paths (vendor package:// URDFs only)."""
    return _urdf_uses_package_uri(path)


def _parse_urdf_xyz(raw: str | None) -> tuple[float, float, float]:
    parts = str(raw or "0 0 0").split()
    vals = [float(parts[i]) if i < len(parts) else 0.0 for i in range(3)]
    return float(vals[0]), float(vals[1]), float(vals[2])


def apply_rm75_joint7_fk_correction(
    source_urdf: Path,
    *,
    repo_root: Path | None = None,
) -> Path:
    """Realman FK correction for RM75-6F joint_7 (vendor y=-0.1725 -> -0.1612 m)."""
    root_out = repo_root if repo_root is not None else _repo_root()
    try:
        st = source_urdf.stat()
        stat_payload = (int(st.st_size), int(st.st_mtime_ns))
    except OSError:
        stat_payload = (0, 0)
    cache_key = hashlib.sha1(
        repr(("rm75_joint7_fk", "j7_only_v4", str(source_urdf.resolve()), stat_payload)).encode("utf-8")
    ).hexdigest()[:16]
    out = root_out / "outputs" / "genesis_robot_urdf_cache" / f"{source_urdf.stem}_j7fk_{cache_key}.urdf"
    if out.is_file():
        return out

    tree = ET.parse(source_urdf)
    root_el = tree.getroot()
    for joint_el in root_el.findall("joint"):
        if str(joint_el.attrib.get("name") or "").strip() != "joint_7":
            continue
        origin_el = joint_el.find("origin")
        if origin_el is None:
            origin_el = ET.SubElement(joint_el, "origin")
        xyz = list(_parse_urdf_xyz(origin_el.get("xyz")))
        xyz[1] = -0.1612
        origin_el.set("xyz", " ".join(f"{float(v):.6f}" for v in xyz))
        break

    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out


def ensure_rm75_gripper_tcp_link(
    source_urdf: Path,
    *,
    repo_root: Path | None = None,
    offset_m: float = 0.220,
) -> Path:
    """Ensure link_7 -> tcp fixed joint exists (matches rm75_control / Realman gripper)."""
    root_out = repo_root if repo_root is not None else _repo_root()
    try:
        st = source_urdf.stat()
        stat_payload = (int(st.st_size), int(st.st_mtime_ns))
    except OSError:
        stat_payload = (0, 0)
    cache_key = hashlib.sha1(
        repr(("rm75_gripper_tcp", float(offset_m), str(source_urdf.resolve()), stat_payload)).encode("utf-8")
    ).hexdigest()[:16]
    out = root_out / "outputs" / "genesis_robot_urdf_cache" / f"{source_urdf.stem}_tcp_{cache_key}.urdf"
    if out.is_file():
        return out

    tree = ET.parse(source_urdf)
    root_el = tree.getroot()
    link_names = {str(link.attrib.get("name", "")) for link in root_el.findall("link")}
    if "tcp" in link_names:
        out.parent.mkdir(parents=True, exist_ok=True)
        tree.write(out, encoding="utf-8", xml_declaration=True)
        return out

    tcp_link = ET.SubElement(root_el, "link")
    tcp_link.set("name", "tcp")
    joint = ET.SubElement(root_el, "joint")
    joint.set("name", "link_7_to_tcp")
    joint.set("type", "fixed")
    origin = ET.SubElement(joint, "origin")
    origin.set("xyz", f"0 0 {float(offset_m):.6f}")
    origin.set("rpy", "0 0 0")
    parent = ET.SubElement(joint, "parent")
    parent.set("link", "link_7")
    child = ET.SubElement(joint, "child")
    child.set("link", "tcp")

    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out


def apply_genesis_link_visual_material_overrides(
    source_urdf: Path,
    link_rgba: dict[str, tuple[float, float, float, float]],
    *,
    repo_root: Path | None = None,
) -> Path:
    """Genesis-only URDF tweak: strip URDF `<visual><material>` on most links so COLLADA shading wins,
    except links listed in ``link_rgba``, which receive a URDF ``rgba``. Requires URDF morph
    ``prioritize_urdf_material=True`` for those overrides to supersede dense mesh visuals.
    """
    if not link_rgba:
        return source_urdf
    root_out = repo_root if repo_root is not None else _repo_root()
    try:
        st = source_urdf.stat()
        stat_payload = (int(st.st_size), int(st.st_mtime_ns))
    except OSError:
        stat_payload = (0, 0)
    ordered = tuple(
        sorted(
            (_ln, tuple(round(float(x), 6) for x in rgba)) for _ln, rgba in sorted(link_rgba.items())
        )
    )
    cache_key = hashlib.sha1(
        repr(("genesis_link_visual", str(source_urdf.resolve()), stat_payload, ordered)).encode("utf-8")
    ).hexdigest()[:16]
    out = root_out / "outputs" / "genesis_robot_urdf_cache" / f"{source_urdf.stem}_gvis_{cache_key}.urdf"
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
            mesh.set("filename", str(_resolve_urdf_mesh_path(source_urdf, fn)))

    for link_el in root_el.findall("link"):
        link_name = str(link_el.attrib.get("name") or "").strip()
        if not link_name:
            continue
        for visual in link_el.findall("visual"):
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
    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out


def append_fixed_tool_chain_to_urdf(
    source_urdf: Path,
    *,
    parent_link: str,
    sensor_link: str,
    tool_link: str,
    sensor_origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    sensor_origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tool_origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tool_origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    visual_mesh_filename: str | None = None,
    visual_origin_xyz: tuple[float, float, float] = (0.0, 0.0, -0.10385),
    visual_origin_rpy: tuple[float, float, float] = (0.0, 0.0, -1.57079632679),
    tool_mass_kg: float = 1.0,
    repo_root: Path | None = None,
) -> Path:
    """Append fixed flange -> virtual F/T sensor -> probe links for simulation-only tools."""

    root_out = repo_root if repo_root is not None else _repo_root()
    visual_abs = ""
    if visual_mesh_filename:
        candidate = Path(str(visual_mesh_filename))
        if not candidate.is_absolute():
            candidate = (root_out / candidate).resolve()
        visual_abs = str(candidate)
    try:
        st = source_urdf.stat()
        stat_payload = (int(st.st_size), int(st.st_mtime_ns))
    except OSError:
        stat_payload = (0, 0)
    payload = (
        "fixed_tool_chain",
        str(source_urdf.resolve()),
        stat_payload,
        parent_link,
        sensor_link,
        tool_link,
        tuple(float(v) for v in sensor_origin_xyz),
        tuple(float(v) for v in sensor_origin_rpy),
        tuple(float(v) for v in tool_origin_xyz),
        tuple(float(v) for v in tool_origin_rpy),
        visual_abs,
        tuple(float(v) for v in visual_origin_xyz),
        tuple(float(v) for v in visual_origin_rpy),
        float(tool_mass_kg),
    )
    cache_key = hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()[:16]
    out = root_out / "outputs" / "genesis_robot_urdf_cache" / f"{source_urdf.stem}_{tool_link}_tool_{cache_key}.urdf"
    if out.is_file():
        return out

    tree = ET.parse(source_urdf)
    root = tree.getroot()
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if not filename:
            continue
        mesh_path = Path(filename)
        if not mesh_path.is_absolute() or str(filename).startswith("package://"):
            mesh.set("filename", str(_resolve_urdf_mesh_path(source_urdf, filename)))

    link_names = {str(link.attrib.get("name", "")) for link in root.findall("link")}
    if parent_link not in link_names:
        raise ValueError(f"fixed_tool_chain parent_link {parent_link!r} not found in {source_urdf}")
    if sensor_link in link_names or tool_link in link_names:
        out.parent.mkdir(parents=True, exist_ok=True)
        tree.write(out, encoding="utf-8", xml_declaration=True)
        return out

    def _origin(parent: ET.Element, xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> None:
        origin = ET.SubElement(parent, "origin")
        origin.set("xyz", " ".join(f"{float(v):.6f}" for v in xyz))
        origin.set("rpy", " ".join(f"{float(v):.12f}" for v in rpy))

    sensor = ET.SubElement(root, "link")
    sensor.set("name", sensor_link)
    inertial = ET.SubElement(sensor, "inertial")
    _origin(inertial, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    ET.SubElement(inertial, "mass").set("value", "0.00010000")
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": "0.00000100",
            "ixy": "0.0",
            "ixz": "0.0",
            "iyy": "0.00000100",
            "iyz": "0.0",
            "izz": "0.00000100",
        },
    )

    tool = ET.SubElement(root, "link")
    tool.set("name", tool_link)
    inertial = ET.SubElement(tool, "inertial")
    _origin(inertial, (0.01, -0.01, 0.15), (0.0, 0.0, 0.0))
    ET.SubElement(inertial, "mass").set("value", f"{float(tool_mass_kg):.8f}")
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": "0.005358333333333334",
            "ixy": "0.0",
            "ixz": "0.0",
            "iyy": "0.005358333333333334",
            "iyz": "0.0",
            "izz": "0.00405",
        },
    )
    if visual_abs:
        visual = ET.SubElement(tool, "visual")
        _origin(visual, visual_origin_xyz, visual_origin_rpy)
        geometry = ET.SubElement(visual, "geometry")
        ET.SubElement(geometry, "mesh").set("filename", visual_abs)

    joint_sensor = ET.SubElement(root, "joint")
    joint_sensor.set("name", f"{parent_link}_to_{sensor_link}")
    joint_sensor.set("type", "fixed")
    _origin(joint_sensor, sensor_origin_xyz, sensor_origin_rpy)
    ET.SubElement(joint_sensor, "parent").set("link", parent_link)
    ET.SubElement(joint_sensor, "child").set("link", sensor_link)

    joint_tool = ET.SubElement(root, "joint")
    joint_tool.set("name", f"{sensor_link}_to_{tool_link}")
    joint_tool.set("type", "fixed")
    _origin(joint_tool, tool_origin_xyz, tool_origin_rpy)
    ET.SubElement(joint_tool, "parent").set("link", sensor_link)
    ET.SubElement(joint_tool, "child").set("link", tool_link)

    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out


def materialize_robot_urdf_absolute_mesh_paths(
    source_urdf: Path,
    *,
    repo_root: Path | None = None,
) -> Path:
    """Rewrite mesh `filename` entries to absolute paths so Genesis can load vendor `package://` URDFs."""
    root_out = repo_root if repo_root is not None else _repo_root()
    try:
        st = source_urdf.stat()
        stat_payload = (int(st.st_size), int(st.st_mtime_ns))
    except OSError:
        stat_payload = (0, 0)
    cache_key = hashlib.sha1(
        repr(("abs_mesh", str(source_urdf.resolve()), stat_payload)).encode("utf-8")
    ).hexdigest()[:16]
    out = (
        root_out
        / "outputs"
        / "genesis_robot_urdf_cache"
        / f"{source_urdf.stem}_abs_mesh_{cache_key}.urdf"
    )
    if out.is_file():
        return out
    tree = ET.parse(source_urdf)
    for mesh in tree.getroot().findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if not filename:
            continue
        mesh_path = Path(filename)
        if not mesh_path.is_absolute() or str(filename).startswith("package://"):
            mesh.set("filename", str(_resolve_urdf_mesh_path(source_urdf, filename)))
    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out


def strip_probe_collision_from_urdf(
    source_urdf: Path,
    *,
    link_name: str = "panda_probe",
    repo_root: Path | None = None,
) -> Path:
    """Remove native probe collision geometry so CI-MPC owns human-probe contact."""

    root_out = repo_root if repo_root is not None else _repo_root()
    try:
        st = source_urdf.stat()
        stat_payload = (int(st.st_size), int(st.st_mtime_ns))
    except OSError:
        stat_payload = (0, 0)
    cache_key = hashlib.sha1(
        repr(("strip_probe", str(source_urdf.resolve()), link_name, stat_payload)).encode("utf-8")
    ).hexdigest()[:16]
    out = (
        root_out
        / "outputs"
        / "genesis_robot_urdf_cache"
        / f"{source_urdf.stem}_{link_name}_no_collision_{cache_key}.urdf"
    )
    if out.is_file():
        return out
    tree = ET.parse(source_urdf)
    root = tree.getroot()
    for mesh in root.findall(".//mesh"):
        fn = mesh.attrib.get("filename")
        if not fn:
            continue
        mesh_path = Path(fn)
        if not mesh_path.is_absolute() or str(fn).startswith("package://"):
            mesh.set("filename", str(_resolve_urdf_mesh_path(source_urdf, fn)))
    for link in root.findall("link"):
        if link.attrib.get("name") != link_name:
            continue
        for collision in list(link.findall("collision")):
            link.remove(collision)
        break
    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out


def resolved_robot_urdf_for_robot_spec(
    robot_spec: Any,
    *,
    enable_collision: bool,
    repo_root: Path | None = None,
    suppress_probe_physics_collision: bool = False,
) -> Path:
    link_rgba = dict(getattr(robot_spec, "genesis_link_visual_urdf_rgba", None) or {})
    has_visual_override = bool(link_rgba)

    robot_urdf = Path(robot_spec.resolved_urdf_path)
    model_id = str(getattr(robot_spec, "model_id", "") or "").strip().lower()
    if model_id == "rm75_6f":
        robot_urdf = apply_rm75_joint7_fk_correction(robot_urdf, repo_root=repo_root)
        robot_urdf = ensure_rm75_gripper_tcp_link(robot_urdf, repo_root=repo_root)
    elif model_id == "rm75_6f_8dof":
        generated = (
            (repo_root or _repo_root())
            / "rm75_control"
            / "rm75_control"
            / "control"
            / "joint_admittance_8dof"
            / "assets"
            / "RM75-6F-8dof.slider.generated.urdf"
        )
        if generated.is_file():
            robot_urdf = generated
        if _urdf_needs_absolute_mesh_materialization(robot_urdf):
            robot_urdf = materialize_robot_urdf_absolute_mesh_paths(robot_urdf, repo_root=repo_root)
        if has_visual_override:
            robot_urdf = apply_genesis_link_visual_material_overrides(robot_urdf, link_rgba, repo_root=repo_root)
        return robot_urdf
    robot_meta = dict(getattr(robot_spec, "asset_metadata", None) or {})
    robot_meta.update(dict(getattr(robot_spec, "metadata", None) or {}))
    fixed_tool_chain = robot_meta.get("fixed_tool_chain")
    if isinstance(fixed_tool_chain, dict) and bool(fixed_tool_chain.get("enabled", True)):
        robot_urdf = append_fixed_tool_chain_to_urdf(
            robot_urdf,
            parent_link=str(fixed_tool_chain.get("parent_link", "link_7")),
            sensor_link=str(fixed_tool_chain.get("sensor_link", "rm75_ft_sensor")),
            tool_link=str(fixed_tool_chain.get("tool_link", "rm75_probe")),
            sensor_origin_xyz=tuple(float(v) for v in fixed_tool_chain.get("sensor_origin_xyz", [0.0, 0.0, 0.0])),
            sensor_origin_rpy=tuple(float(v) for v in fixed_tool_chain.get("sensor_origin_rpy", [0.0, 0.0, 0.0])),
            tool_origin_xyz=tuple(float(v) for v in fixed_tool_chain.get("tool_origin_xyz", [0.0, 0.0, 0.0])),
            tool_origin_rpy=tuple(float(v) for v in fixed_tool_chain.get("tool_origin_rpy", [0.0, 0.0, 0.0])),
            visual_mesh_filename=(
                str(fixed_tool_chain.get("visual_mesh_filename", ""))
                if fixed_tool_chain.get("visual_mesh_filename") not in (None, "")
                else None
            ),
            visual_origin_xyz=tuple(float(v) for v in fixed_tool_chain.get("visual_origin_xyz", [0.0, 0.0, -0.10385])),
            visual_origin_rpy=tuple(float(v) for v in fixed_tool_chain.get("visual_origin_rpy", [0.0, 0.0, -1.57079632679])),
            tool_mass_kg=float(fixed_tool_chain.get("tool_mass_kg", 1.0)),
            repo_root=repo_root,
        )
    probe_collision = robot_spec.probe_collision
    if (
        bool(enable_collision)
        and probe_collision is not None
        and bool(probe_collision.enabled)
        and not bool(suppress_probe_physics_collision)
    ):
        robot_urdf = apply_probe_collision_to_urdf(
            robot_urdf,
            link_name=str(probe_collision.link_name),
            shape=str(getattr(probe_collision, "shape", "cylinder") or "cylinder"),
            radius=float(probe_collision.radius),
            length=float(probe_collision.length),
            origin_xyz=(
                tuple(float(v) for v in probe_collision.origin_xyz)
                if probe_collision.origin_xyz is not None
                else None
            ),
            origin_rpy=(
                tuple(float(v) for v in probe_collision.origin_rpy)
                if probe_collision.origin_rpy is not None
                else None
            ),
            mesh_filename=getattr(probe_collision, "mesh_filename", None),
            repo_root=repo_root,
        )
        if has_visual_override:
            robot_urdf = apply_genesis_link_visual_material_overrides(robot_urdf, link_rgba, repo_root=repo_root)
        return robot_urdf

    if bool(suppress_probe_physics_collision) and probe_collision is not None:
        base_urdf = robot_urdf
        if _urdf_needs_absolute_mesh_materialization(base_urdf):
            base_urdf = materialize_robot_urdf_absolute_mesh_paths(base_urdf, repo_root=repo_root)
        robot_urdf = strip_probe_collision_from_urdf(
            base_urdf,
            link_name=str(probe_collision.link_name),
            repo_root=repo_root,
        )
        if has_visual_override:
            robot_urdf = apply_genesis_link_visual_material_overrides(robot_urdf, link_rgba, repo_root=repo_root)
        return robot_urdf

    if _urdf_needs_absolute_mesh_materialization(robot_urdf):
        robot_urdf = materialize_robot_urdf_absolute_mesh_paths(robot_urdf, repo_root=repo_root)
    if has_visual_override:
        robot_urdf = apply_genesis_link_visual_material_overrides(robot_urdf, link_rgba, repo_root=repo_root)
    return robot_urdf


def resolved_robot_urdf_for_scene_spec(
    scene_spec: SyncSceneSpec,
    *,
    enable_collision: bool,
    repo_root: Path | None = None,
    suppress_probe_physics_collision: bool = False,
) -> Path:
    return resolved_robot_urdf_for_robot_spec(
        scene_spec.robot,
        enable_collision=enable_collision,
        repo_root=repo_root,
        suppress_probe_physics_collision=suppress_probe_physics_collision,
    )
