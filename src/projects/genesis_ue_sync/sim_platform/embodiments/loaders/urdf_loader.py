from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import HumanMotionSequence
from projects.genesis_ue_sync.sim_platform.core.messages import CameraIntrinsics
from projects.genesis_ue_sync.sim_platform.core.specs import FrameSpec
from projects.genesis_ue_sync.sim_platform.embodiments.profiles import (
    CameraRigProfile,
    EmbodimentProfile,
    EndEffectorProfile,
    JointLimit,
    RobotProfile,
    SensorProfile,
    ToolProfile,
)
from projects.genesis_ue_sync.sim_platform.embodiments.smpl2urdf import ProxyGeometry, resolve_smpl_proxy_urdf


@dataclass
class URDFToolFrames:
    base_frame: str
    eef_link: str
    tool_frame: str
    tcp_frame: str
    ultrasound_image_frame: str | None = None


def parse_revolute_joint_limits(urdf_path: Path) -> tuple[list[str], dict[str, JointLimit]]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joint_names: list[str] = []
    joint_limits: dict[str, JointLimit] = {}
    for joint in root.findall("joint"):
        jtype = joint.attrib.get("type")
        if jtype not in {"revolute", "continuous"}:
            continue
        name = joint.attrib["name"]
        limit = joint.find("limit")
        if limit is None:
            if jtype == "continuous":
                joint_names.append(name)
                joint_limits[name] = JointLimit(
                    lower=-6.283185307179586,
                    upper=6.283185307179586,
                    effort=250.0,
                    velocity=50.0,
                )
            continue
        joint_names.append(name)
        joint_limits[name] = JointLimit(
            lower=float(limit.attrib["lower"]),
            upper=float(limit.attrib["upper"]),
            effort=float(limit.attrib.get("effort", "0.0")),
            velocity=float(limit.attrib.get("velocity", "0.0")),
        )
    return joint_names, joint_limits


def _scale_space_separated_xyz(value: str, scale: float) -> str:
    parts = value.replace(",", " ").split()
    if len(parts) < 3:
        return value
    x, y, z = (float(parts[0]), float(parts[1]), float(parts[2]))
    return f"{x * scale:.5f} {y * scale:.5f} {z * scale:.5f}"


def _scale_float_attr(elem: ET.Element, key: str, scale: float) -> None:
    if key not in elem.attrib:
        return
    elem.attrib[key] = f"{float(elem.attrib[key]) * scale:.5f}"


# Genesis (MuJoCo-style) parsers require mass and diagonal inertia above mjMINVAL; PyBullet humanoid.urdf uses zero-mass dummies.
_MIN_URDF_LINK_MASS_KG = 1e-4
_MIN_URDF_LINK_INERTIA_DIAG = 1e-6
_GENESIS_INERTIAL_PATCH_ID = b"genesis_inertial_v1"


def patch_urdf_zero_inertial_links(source_urdf: Path, dest_urdf: Path) -> None:
    tree = ET.parse(source_urdf)
    for link in tree.getroot().findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass_el = inertial.find("mass")
        inertia_el = inertial.find("inertia")
        if mass_el is None or inertia_el is None:
            continue
        m = float(mass_el.attrib.get("value", "0"))
        ixx = float(inertia_el.attrib.get("ixx", "0"))
        iyy = float(inertia_el.attrib.get("iyy", "0"))
        izz = float(inertia_el.attrib.get("izz", "0"))
        # Only touch massless URDF dummies (e.g. PyBullet humanoid link1_*). Real links keep authored mass/inertia.
        if m <= 0.0:
            mass_el.attrib["value"] = f"{_MIN_URDF_LINK_MASS_KG:.8f}"
        if ixx <= 0.0 and iyy <= 0.0 and izz <= 0.0:
            for k in ("ixx", "iyy", "izz"):
                inertia_el.attrib[k] = f"{_MIN_URDF_LINK_INERTIA_DIAG:.8f}"
            for k in ("ixy", "ixz", "iyz"):
                if k in inertia_el.attrib:
                    inertia_el.attrib[k] = "0.0"
    dest_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dest_urdf, encoding="unicode", xml_declaration=True)


def _cached_genesis_safe_urdf(source_urdf: Path, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(_GENESIS_INERTIAL_PATCH_ID + b"|" + str(source_urdf.resolve()).encode()).hexdigest()[:12]
    out = cache_dir / f"{source_urdf.stem}_gins_{digest}.urdf"
    if not out.is_file():
        patch_urdf_zero_inertial_links(source_urdf, out)
    return out


def write_uniform_scaled_urdf(source_urdf: Path, dest_urdf: Path, scale: float) -> None:
    """Scale URDF link/joint translations and primitive collision geometry by a uniform factor.

    Mass scales as scale**3 and inertia components as scale**5 (uniform density scaling).
    Common practice for subject-specific / SMPL-driven collision proxies in humanoid retargeting
    and video-to-sim pipelines.
    """
    s = float(scale)
    if s <= 0.0:
        raise ValueError(f"URDF scale must be positive, got {scale}")
    tree = ET.parse(source_urdf)
    root_el = tree.getroot()
    m_scale = s**3
    i_scale = s**5
    for elem in root_el.iter():
        tag = elem.tag.split("}")[-1]
        if tag == "origin" and "xyz" in elem.attrib:
            elem.attrib["xyz"] = _scale_space_separated_xyz(elem.attrib["xyz"], s)
        elif tag == "mass" and "value" in elem.attrib:
            mv = float(elem.attrib["value"])
            elem.attrib["value"] = f"{mv * m_scale:.8f}" if mv != 0.0 else elem.attrib["value"]
        elif tag == "inertia":
            for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
                if k in elem.attrib:
                    iv = float(elem.attrib[k])
                    elem.attrib[k] = f"{iv * i_scale:.8f}" if iv != 0.0 else elem.attrib[k]
        elif tag == "geometry":
            parent = elem
            for child in list(parent):
                ctag = child.tag.split("}")[-1]
                if ctag == "capsule":
                    _scale_float_attr(child, "radius", s)
                    _scale_float_attr(child, "length", s)
                elif ctag == "cylinder":
                    _scale_float_attr(child, "radius", s)
                    _scale_float_attr(child, "length", s)
                elif ctag == "sphere":
                    _scale_float_attr(child, "radius", s)
                elif ctag == "box":
                    if "size" in child.attrib:
                        child.attrib["size"] = _scale_space_separated_xyz(child.attrib["size"], s)
    dest_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dest_urdf, encoding="unicode", xml_declaration=True)


def cached_scaled_urdf_path(source_urdf: Path, scale: float, cache_dir: Path) -> Path:
    digest = hashlib.sha256(f"{source_urdf.resolve()}|{scale:.8f}".encode()).hexdigest()[:12]
    return cache_dir / f"{source_urdf.stem}_{digest}_u{scale:.4f}.urdf"


def resolve_urdf_with_uniform_scale(
    source_urdf: Path,
    scale: float,
    *,
    enabled: bool,
    cache_dir: Path,
) -> Path:
    src = source_urdf.expanduser().resolve()
    base = _cached_genesis_safe_urdf(src, cache_dir)
    s = float(scale)
    if not enabled or abs(s - 1.0) < 1e-6:
        return base
    out = cached_scaled_urdf_path(base, s, cache_dir)
    if not out.is_file():
        write_uniform_scaled_urdf(base, out, s)
    return out


def _link_limb_group(link_name: str) -> str:
    core = link_name.split("__")[0]
    if core == "Pelvis":
        return "pelvis"
    if core in ("Torso", "Spine", "Chest", "Neck", "Head", "L_Thorax", "R_Thorax"):
        return "torso"
    if core in ("L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle", "L_Toe", "R_Toe"):
        return "leg"
    if core in ("L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand"):
        return "arm"
    return "torso"


def _scale_for_link(link_name: str, limb_scales: dict[str, float]) -> float:
    g = _link_limb_group(link_name)
    v = limb_scales.get(g)
    if v is not None and float(v) > 0.0:
        return float(v)
    fb = limb_scales.get("torso")
    if fb is not None and float(fb) > 0.0:
        return float(fb)
    return 1.0


def write_limb_group_scaled_urdf(source_urdf: Path, dest_urdf: Path, limb_scales: dict[str, float]) -> None:
    """Scale joint origins, collision primitives, and diagonal inertia per limb group (SMPL-style link names).

    Mass scales as s**3 and diagonal inertia as s**5 for links with positive mass. Zero-mass dummies are unchanged.
    """
    tree = ET.parse(source_urdf)
    root_el = tree.getroot()
    for joint in root_el.findall("joint"):
        child = joint.find("child")
        if child is None or "link" not in child.attrib:
            continue
        cl = str(child.attrib["link"])
        s = _scale_for_link(cl, limb_scales)
        if abs(s - 1.0) < 1e-9:
            continue
        origin = joint.find("origin")
        if origin is not None and "xyz" in origin.attrib:
            origin.attrib["xyz"] = _scale_space_separated_xyz(origin.attrib["xyz"], s)

    for link in root_el.findall("link"):
        name = link.attrib.get("name")
        if not name:
            continue
        s = _scale_for_link(str(name), limb_scales)
        if abs(s - 1.0) < 1e-9:
            continue
        inertial = link.find("inertial")
        if inertial is not None:
            m_scale = s**3
            i_scale = s**5
            mass_el = inertial.find("mass")
            inertia_el = inertial.find("inertia")
            if mass_el is not None and "value" in mass_el.attrib:
                mv = float(mass_el.attrib["value"])
                if mv > 0.0:
                    mass_el.attrib["value"] = f"{mv * m_scale:.8f}"
            if inertia_el is not None:
                for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
                    if k in inertia_el.attrib:
                        iv = float(inertia_el.attrib[k])
                        if iv > 0.0:
                            inertia_el.attrib[k] = f"{iv * i_scale:.8f}"
            orig = inertial.find("origin")
            if orig is not None and "xyz" in orig.attrib:
                orig.attrib["xyz"] = _scale_space_separated_xyz(orig.attrib["xyz"], s)
        for col in link.findall("collision"):
            o = col.find("origin")
            if o is not None and "xyz" in o.attrib:
                o.attrib["xyz"] = _scale_space_separated_xyz(o.attrib["xyz"], s)
            geom = col.find("geometry")
            if geom is None:
                continue
            for gchild in list(geom):
                ctag = gchild.tag.split("}")[-1]
                if ctag == "capsule":
                    _scale_float_attr(gchild, "radius", s)
                    _scale_float_attr(gchild, "length", s)
                elif ctag == "cylinder":
                    _scale_float_attr(gchild, "radius", s)
                    _scale_float_attr(gchild, "length", s)
                elif ctag == "sphere":
                    _scale_float_attr(gchild, "radius", s)
                elif ctag == "box":
                    if "size" in gchild.attrib:
                        gchild.attrib["size"] = _scale_space_separated_xyz(gchild.attrib["size"], s)

    dest_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dest_urdf, encoding="unicode", xml_declaration=True)


def _cached_limb_group_scaled_urdf_path(source_urdf: Path, limb_scales: dict[str, float], cache_dir: Path) -> Path:
    ordered = tuple(sorted((str(k), round(float(v), 6)) for k, v in limb_scales.items()))
    digest = hashlib.sha256(str(source_urdf.resolve()).encode() + repr(ordered).encode()).hexdigest()[:14]
    return cache_dir / f"{source_urdf.stem}_limb_{digest}.urdf"


def resolve_urdf_with_limb_group_scale(source_urdf: Path, limb_scales: dict[str, float], *, cache_dir: Path) -> Path:
    """Genesis-safe URDF, then per-limb collision/inertia/origin scaling; cached by limb scale tuple."""
    src = source_urdf.expanduser().resolve()
    base = _cached_genesis_safe_urdf(src, cache_dir)
    ls = {str(k): float(v) for k, v in limb_scales.items() if float(v) > 0.0}
    if not ls:
        return base
    if all(abs(float(ls[k]) - 1.0) < 1e-6 for k in ls):
        return base
    out = _cached_limb_group_scaled_urdf_path(base, ls, cache_dir)
    if not out.is_file():
        write_limb_group_scaled_urdf(base, out, ls)
    return out


def resolve_shape_specific_proxy_urdf(
    sequence: HumanMotionSequence,
    *,
    cache_dir: Path | None = None,
    device: str | None = "cpu",
) -> tuple[Path, ProxyGeometry]:
    return resolve_smpl_proxy_urdf(sequence, cache_dir=cache_dir, device=device)


def parse_kinematic_edges(urdf_path: Path) -> list[tuple[str, str]]:
    """Parent/child link pairs for every URDF joint (any type), for debug skeleton drawing."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    edges: list[tuple[str, str]] = []
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        pl = parent.attrib.get("link")
        cl = child.attrib.get("link")
        if pl and cl:
            edges.append((str(pl), str(cl)))
    return edges


def parse_link_masses_kg(urdf_path: Path) -> dict[str, float]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    masses: dict[str, float] = {}
    for link in root.findall("link"):
        name = link.attrib.get("name")
        if not name:
            continue
        inertial = link.find("inertial")
        if inertial is None:
            masses[str(name)] = 0.0
            continue
        mass_el = inertial.find("mass")
        if mass_el is None:
            masses[str(name)] = 0.0
        else:
            masses[str(name)] = float(mass_el.attrib.get("value", "0"))
    return masses


def parse_collapsed_kinematic_edges(urdf_path: Path, *, min_mass_kg: float = 1e-6) -> list[tuple[str, str]]:
    """
    Edges between heavy links only, collapsing zero-mass intermediate links (e.g. PyBullet humanoid link1_*).

    Raw parent/child link origins often sit at inertial frames, not joint axes; collapsing reduces spurious
    segments. Lines still approximate bone direction (link-frame to link-frame), not exact joint axes.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    masses = parse_link_masses_kg(urdf_path)
    children: dict[str, list[str]] = {}
    parent_of: dict[str, str] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        pl = parent.attrib.get("link")
        cl = child.attrib.get("link")
        if not pl or not cl:
            continue
        pl_s, cl_s = str(pl), str(cl)
        children.setdefault(pl_s, []).append(cl_s)
        parent_of[cl_s] = pl_s

    def heavy_ancestor(link: str) -> str:
        cur = link
        for _ in range(512):
            if masses.get(cur, 0.0) >= min_mass_kg:
                return cur
            if cur not in parent_of:
                return cur
            cur = parent_of[cur]
        return cur

    def heavy_descendant(link: str) -> str:
        cur = link
        for _ in range(512):
            if masses.get(cur, 0.0) >= min_mass_kg:
                return cur
            kids = children.get(cur, [])
            if len(kids) != 1:
                return cur
            cur = kids[0]
        return cur

    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        pl = str(parent.attrib.get("link", ""))
        cl = str(child.attrib.get("link", ""))
        if not pl or not cl:
            continue
        a = heavy_ancestor(pl)
        d = heavy_descendant(cl)
        if a == d:
            continue
        key = (a, d)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def parse_root_link_name(urdf_path: Path) -> str:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    link_names = {link.attrib["name"] for link in root.findall("link") if "name" in link.attrib}
    child_links = {
        child.attrib["link"]
        for joint in root.findall("joint")
        for child in [joint.find("child")]
        if child is not None and "link" in child.attrib
    }
    root_links = sorted(link_names - child_links)
    if not root_links:
        raise ValueError(f"Failed to infer root link from URDF: {urdf_path}")
    return str(root_links[0])


def _build_camera_sensor(
    *,
    camera_name: str,
    frame_id: str,
    resolution: tuple[int, int],
    focal_length_px: float,
    mount_link: str | None = None,
) -> SensorProfile:
    width, height = resolution
    return SensorProfile(
        name=camera_name,
        modality="rgb",
        frame_id=frame_id,
        mount_link=mount_link,
        hz=30.0,
        encoding="rgb8",
        resolution=resolution,
        intrinsics=CameraIntrinsics(
            width=width,
            height=height,
            fx=focal_length_px,
            fy=focal_length_px,
            cx=float(width) / 2.0,
            cy=float(height) / 2.0,
        ),
    )


def build_embodiment_from_urdf(
    *,
    name: str,
    urdf_path: Path,
    tool_frames: URDFToolFrames,
    camera_names: Iterable[str] = (),
    image_resolution: tuple[int, int] = (1280, 720),
    camera_baseline_m: float = 0.12,
    fixed_base: bool = True,
    tool_name: str = "tool",
    max_contact_force_n: float = 15.0,
    workspace_limits: dict[str, tuple[float, float]] | None = None,
    safety_constraints: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
) -> EmbodimentProfile:
    urdf = Path(urdf_path)
    if not urdf.exists():
        raise FileNotFoundError(f"URDF not found: {urdf}")

    joint_names, joint_limits = parse_revolute_joint_limits(urdf)
    camera_name_list = list(camera_names)
    focal_length_px = float(max(image_resolution)) * 0.9
    frame_spec = FrameSpec(
        world_frame="world",
        robot_base_frame=tool_frames.base_frame,
        eef_link=tool_frames.eef_link,
        tool_frame=tool_frames.tool_frame,
        probe_contact_frame=tool_frames.tcp_frame,
        ultrasound_image_frame=tool_frames.ultrasound_image_frame or "ultrasound_image_frame",
        smpl_frame="smpl_world",
        patient_surface_local_frame="patient_surface_local_frame",
        camera_frames={camera_name: f"camera_frame/{camera_name}" for camera_name in camera_name_list},
    )

    robot = RobotProfile(
        name=name,
        urdf_path=urdf,
        base_frame=tool_frames.base_frame,
        eef_link=tool_frames.eef_link,
        joint_names=joint_names,
        joint_limits=joint_limits,
        fixed_base=fixed_base,
        default_control_space="joint_position",
        workspace_limits=workspace_limits or {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (0.0, 1.5)},
        safety_constraints=safety_constraints or {"collision_check": True},
        metadata=dict(metadata or {}),
    )

    tool = ToolProfile(
        name=tool_name,
        mount_frame=tool_frames.eef_link,
        tool_frame=tool_frames.tool_frame,
        contact_frame=tool_frames.tcp_frame,
        ultrasound_image_frame=frame_spec.ultrasound_image_frame,
        max_contact_force_n=max_contact_force_n,
    )

    end_effector = EndEffectorProfile(
        name=f"{name}_end_effector",
        mount_link=tool_frames.eef_link,
        tool_frame=tool_frames.tool_frame,
        tcp_frame=tool_frames.tcp_frame,
        command_frame=tool_frames.tcp_frame,
    )

    sensors = [
        _build_camera_sensor(
            camera_name=camera_name,
            frame_id=frame_spec.frame_for_camera(camera_name),
            resolution=image_resolution,
            focal_length_px=focal_length_px,
        )
        for camera_name in camera_name_list
    ]
    if tool_frames.ultrasound_image_frame is not None:
        sensors.append(
            SensorProfile(
                name="ultrasound",
                modality="ultrasound",
                frame_id=frame_spec.ultrasound_image_frame,
                mount_link=tool_frames.tool_frame,
                hz=30.0,
                encoding="rgb8",
                resolution=image_resolution,
            )
        )

    camera_rigs = []
    if camera_name_list:
        camera_rigs.append(
            CameraRigProfile(
                name=f"{name}_camera_rig",
                camera_names=camera_name_list,
                primary_camera=camera_name_list[0],
                baseline_m=camera_baseline_m,
                metadata={"supports_n_view_extension": True},
            )
        )

    return EmbodimentProfile(
        name=name,
        robot=robot,
        tool=tool,
        end_effector=end_effector,
        frame_spec=frame_spec,
        sensors=sensors,
        camera_rigs=camera_rigs,
        metadata=dict(metadata or {}),
    )
