"""PHC bundled SMPL humanoid MJCF for Genesis human proxy.

Default: copy gender templates from ``phc/data/assets/mjcf/``. Optional: ``AMONGUS_PHC_MJCF_SOURCE=smpl_robot``
generates beta-conditioned capsule MJCF via SMPLSim ``SMPL_Robot`` (same family as PHC training).

Motion → Genesis ``dofs``: ``phc_mjcf_retarget``; layout order follows Genesis BFS link order.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import HumanMotionSequence
from projects.genesis_ue_sync.sim_platform.embodiments.smpl2urdf import (
    ProxyBodyGeometry,
    ProxyGeometry,
    shape_key_from_params,
)


def phc_mjcf_library_dir(phc_root: Path) -> Path:
    return Path(phc_root).expanduser().resolve() / "phc" / "data" / "assets" / "mjcf"


def resolve_phc_bundle_mjcf_path(phc_root: Path, gender: str) -> Path:
    """Pick a PHC template MuJoCo file (bundled in the PHC repo)."""

    lib = phc_mjcf_library_dir(phc_root)
    g = str(gender).strip().lower()
    candidates: list[str]
    if g == "female":
        candidates = ["smpl_1_humanoid.xml", "smpl_humanoid.xml"]
    elif g == "male":
        candidates = ["smpl_2_humanoid.xml", "smpl_humanoid.xml"]
    else:
        candidates = ["smpl_0_humanoid.xml", "smpl_humanoid.xml"]
    for name in candidates:
        path = lib / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No PHC bundled MJCF found under {lib} (tried {candidates}). "
        "Clone PHC so phc/data/assets/mjcf/*.xml exists."
    )


def _geom_volume_m3(geom: ET.Element) -> float:
    typ = (geom.get("type") or "").strip().lower()
    size_attr = geom.get("size") or ""
    if typ == "sphere":
        r = float(size_attr.split()[0]) if size_attr else 0.0
        return float((4.0 / 3.0) * math.pi * max(r, 1e-9) ** 3)
    if typ == "box":
        parts = [float(x) for x in size_attr.split()]
        if len(parts) < 3:
            return 0.0
        sx, sy, sz = parts[0], parts[1], parts[2]
        return float((2. * sx) * (2. * sy) * (2. * sz))
    if typ == "capsule":
        fromto = geom.get("fromto") or ""
        parts = [float(x) for x in fromto.split()]
        if len(parts) < 6:
            return 0.0
        p0 = np.asarray(parts[:3], dtype=np.float64)
        p1 = np.asarray(parts[3:6], dtype=np.float64)
        seg = float(np.linalg.norm(p1 - p0))
        r = float(size_attr.split()[0]) if size_attr else 0.0
        r = max(r, 1e-9)
        cyl_h = max(seg - 2.0 * r, 0.0)
        v_cyl = math.pi * r * r * cyl_h
        v_sph = (4.0 / 3.0) * math.pi * r ** 3
        return float(v_cyl + v_sph)
    return 0.0


def _geom_mass_kg(geom: ET.Element) -> float:
    m_attr = geom.get("mass")
    if m_attr is not None:
        try:
            return max(float(m_attr), 1e-9)
        except ValueError:
            pass
    d_attr = geom.get("density")
    density = float(d_attr) if d_attr is not None else 1000.0
    vol = _geom_volume_m3(geom)
    return max(density * vol, 1e-9)


def _proxy_geometry_from_phc_mjcf(mjcf_path: Path, *, sequence: HumanMotionSequence) -> ProxyGeometry:
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    wb = root.find("worldbody")
    if wb is None:
        raise ValueError(f"No worldbody in {mjcf_path}")

    mass_by_body: dict[str, float] = {}

    def walk_body(body_el: ET.Element) -> None:
        bname = body_el.get("name") or "unnamed"
        for child in body_el:
            if child.tag == "geom":
                m = _geom_mass_kg(child)
                mass_by_body[bname] = mass_by_body.get(bname, 0.0) + m
            if child.tag == "body":
                walk_body(child)

    for b in wb.findall("body"):
        walk_body(b)

    shape_key = shape_key_from_params(
        model_type=sequence.model_type,
        gender=sequence.gender,
        betas=np.asarray(sequence.betas, dtype=np.float32).reshape(-1),
    )
    shape_key = f"{shape_key}_phc_bundled"

    bodies: list[ProxyBodyGeometry] = []
    for i, (name, mass) in enumerate(sorted(mass_by_body.items(), key=lambda x: x[0])):
        bodies.append(
            ProxyBodyGeometry(
                name=name,
                parent_name=None,
                joint_idx=i + 1,
                parent_joint_idx=None,
                end_joint_idx=None,
                joint_origin_xyz=(0.0, 0.0, 0.0),
                capsule_length_m=0.1,
                capsule_radius_m=0.05,
                capsule_axis_world=(0.0, 0.0, 1.0),
                mass_kg=float(mass),
                group="phc_mjcf_geom",
                primitive_type="capsule",
            )
        )

    if not bodies:
        bodies.append(
            ProxyBodyGeometry(
                name="Pelvis",
                parent_name=None,
                joint_idx=1,
                parent_joint_idx=None,
                end_joint_idx=None,
                joint_origin_xyz=(0.0, 0.0, 0.0),
                capsule_length_m=0.1,
                capsule_radius_m=0.05,
                capsule_axis_world=(0.0, 0.0, 1.0),
                mass_kg=70.0,
                group="phc_mjcf_geom",
                primitive_type="capsule",
            )
        )

    return ProxyGeometry(
        model_type=str(sequence.model_type).lower(),
        gender=str(sequence.gender).lower(),
        shape_key=shape_key,
        hip_width_m=0.0,
        shoulder_width_m=0.0,
        torso_height_m=0.0,
        torso_depth_m=0.0,
        bodies=tuple(bodies),
    )


def build_phc_bundled_mjcf_layout_dict(mjcf_path: Path) -> dict[str, Any]:
    """DOF segments in **Genesis MJCF import order** (BFS link order), not XML depth-first."""

    try:
        import mujoco
        from genesis.utils import urdf as uu
    except ModuleNotFoundError:
        return _build_phc_bundled_mjcf_layout_dict_from_xml(mjcf_path)

    mjcf_path = Path(mjcf_path).expanduser().resolve()
    mj = mujoco.MjModel.from_xml_path(str(mjcf_path))
    nbody = int(mj.nbody)
    l_infos: list[dict[str, Any]] = []
    j_placeholder: list[list] = [[] for _ in range(nbody)]
    for i_l in range(nbody):
        nm = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, i_l) or f"body_{i_l}"
        if int(mj.body_parentid[i_l]) == i_l:
            pidx = -1
        else:
            pidx = int(mj.body_parentid[i_l])
        l_infos.append({"name": nm, "parent_idx": pidx})
    *_, bfs_perm = uu._order_links(l_infos, j_placeholder, None)

    jnt_free = int(mujoco.mjtJoint.mjJNT_FREE)
    jnt_hinge = int(mujoco.mjtJoint.mjJNT_HINGE)
    jnt_ball = int(mujoco.mjtJoint.mjJNT_BALL)
    segs: list[dict[str, Any]] = []
    for old_i in bfs_perm:
        bname = str(l_infos[old_i]["name"])
        jnt_adr = int(mj.body_jntadr[old_i])
        jnt_num = int(mj.body_jntnum[old_i])
        if jnt_num <= 0:
            continue
        for k in range(jnt_num):
            i_j = jnt_adr + k
            mj_type = int(mj.jnt_type[i_j])
            jn = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, i_j) or "joint"
            if mj_type == jnt_free:
                segs.append(
                    {
                        "kind": "free_mujoco",
                        "body": bname,
                        "joint": str(jn),
                        "n": 6,
                        "labels": ["px", "py", "pz", "rx", "ry", "rz"],
                    }
                )
            elif mj_type == jnt_hinge:
                ax = mj.jnt_axis[i_j]
                axis = [float(ax[0]), float(ax[1]), float(ax[2])]
                seg_h: dict[str, Any] = {
                    "kind": "hinge",
                    "body": bname,
                    "joint": str(jn),
                    "n": 1,
                    "labels": ["angle"],
                    "axis_world_hint": axis,
                }
                if int(mj.jnt_limited[i_j]):
                    seg_h["range_rad"] = [float(mj.jnt_range[i_j][0]), float(mj.jnt_range[i_j][1])]
                segs.append(seg_h)
            elif mj_type == jnt_ball:
                segs.append(
                    {
                        "kind": "ball",
                        "body": bname,
                        "joint": str(jn),
                        "n": 3,
                        "labels": ["rx", "ry", "rz"],
                    }
                )
            else:
                raise ValueError(f"Unsupported MuJoCo joint type {mj_type} on {jn!r} / body {bname!r}")

    total = int(sum(int(s["n"]) for s in segs))
    return {
        "mjcf_layout_tag": "phc_bundled_mjcf",
        "segments": segs,
        "total_dofs": total,
        "source_mjcf": str(mjcf_path.resolve()),
        "notes": (
            "Segment order matches Genesis MJCF loader: breadth-first link order via "
            "genesis/utils/urdf._order_links (not XML DFS). Root FREE = 6 Genesis dofs. "
            "Hinge metadata from compiled mjModel."
        ),
    }


def _build_phc_bundled_mjcf_layout_dict_from_xml(mjcf_path: Path) -> dict[str, Any]:
    mjcf_path = Path(mjcf_path).expanduser().resolve()
    tree = ET.parse(mjcf_path)
    worldbody = tree.getroot().find("worldbody")
    if worldbody is None:
        raise ValueError(f"MJCF has no worldbody: {mjcf_path}")

    queue: list[ET.Element] = [body for body in list(worldbody) if body.tag == "body"]
    segs: list[dict[str, Any]] = []
    while queue:
        body = queue.pop(0)
        bname = str(body.attrib.get("name") or "body")
        for child in list(body):
            if child.tag == "freejoint":
                segs.append(
                    {
                        "kind": "free_mujoco",
                        "body": bname,
                        "joint": str(child.attrib.get("name") or bname),
                        "n": 6,
                        "labels": ["px", "py", "pz", "rx", "ry", "rz"],
                    }
                )
            elif child.tag == "joint":
                jtype = str(child.attrib.get("type", "hinge")).strip().lower()
                if jtype == "free":
                    segs.append(
                        {
                            "kind": "free_mujoco",
                            "body": bname,
                            "joint": str(child.attrib.get("name") or bname),
                            "n": 6,
                            "labels": ["px", "py", "pz", "rx", "ry", "rz"],
                        }
                    )
                elif jtype == "hinge":
                    axis_raw = str(child.attrib.get("axis", "1 0 0")).split()
                    axis = [float(v) for v in axis_raw[:3]]
                    seg_h: dict[str, Any] = {
                        "kind": "hinge",
                        "body": bname,
                        "joint": str(child.attrib.get("name") or "joint"),
                        "n": 1,
                        "labels": ["angle"],
                        "axis_world_hint": axis,
                    }
                    if "range" in child.attrib:
                        vals = [float(v) for v in str(child.attrib["range"]).split()[:2]]
                        seg_h["range_rad"] = [math.radians(vals[0]), math.radians(vals[1])]
                    segs.append(seg_h)
                elif jtype == "ball":
                    segs.append(
                        {
                            "kind": "ball",
                            "body": bname,
                            "joint": str(child.attrib.get("name") or "joint"),
                            "n": 3,
                            "labels": ["rx", "ry", "rz"],
                        }
                    )
                else:
                    raise ValueError(f"Unsupported MJCF joint type {jtype!r} on body {bname!r}")
        queue.extend([child for child in list(body) if child.tag == "body"])

    total = int(sum(int(s["n"]) for s in segs))
    return {
        "mjcf_layout_tag": "phc_bundled_mjcf",
        "segments": segs,
        "total_dofs": total,
        "source_mjcf": str(mjcf_path.resolve()),
        "notes": (
            "Fallback layout parsed directly from MJCF XML because mujoco/genesis layout helpers were unavailable. "
            "Body order is breadth-first; hinge ranges are converted from XML degrees to radians."
        ),
    }


_PLACEHOLDER_URDF = '''<?xml version="1.0"?>
<robot name="phc_bundled_proxy_placeholder">
  <link name="Pelvis">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1e-6"/>
      <inertia ixx="1e-9" ixy="0" ixz="0" iyy="1e-9" iyz="0" izz="1e-9"/>
    </inertial>
  </link>
</robot>
'''


def _ensure_placeholder_urdf(path: Path) -> None:
    path = Path(path)
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_PLACEHOLDER_URDF, encoding="utf-8")


def _shrink_foot_box_geoms(mjcf_path: Path, *, scale: float = 2.0 / 3.0) -> None:
    """Shrink PHC ankle/toe box geoms in cached MJCF copies.

    The bundled PHC foot boxes are too blocky for bed/contact visualization. This
    only mutates generated cache files. A sidecar marker avoids repeated scaling
    without adding non-MuJoCo attributes to the XML.
    """

    scale = float(scale)
    if not (0.0 < scale <= 1.0):
        return
    path = Path(mjcf_path)
    marker_path = path.with_suffix(path.suffix + f".foot_box_scale_{scale:.6f}")
    if marker_path.is_file():
        return
    tree = ET.parse(path)
    root = tree.getroot()
    changed = False
    foot_names = ("ankle", "toe", "foot")
    for body in root.iter("body"):
        body_name = str(body.get("name") or "").lower()
        if not any(token in body_name for token in foot_names):
            continue
        for geom in body.findall("geom"):
            if str(geom.get("type") or "").strip().lower() != "box":
                continue
            geom.attrib.pop("amongus_foot_box_scale", None)
            size_raw = geom.get("size") or ""
            parts = [float(v) for v in size_raw.split()]
            if len(parts) < 3:
                continue
            geom.set("size", " ".join(f"{v * scale:.6g}" for v in parts[:3]))
            changed = True
    if changed:
        tree.write(path, encoding="utf-8", xml_declaration=True)
    marker_path.write_text("ok\n", encoding="utf-8")


def sync_phc_bundled_proxy_to_cache(
    sequence: HumanMotionSequence,
    *,
    cache_dir: Path,
    phc_root: Path,
    force_rewrite: bool = False,
) -> tuple[Path, Path, Path, ProxyGeometry]:
    """Write PHC-style MJCF + layout JSON + placeholder URDF under ``cache_dir``."""

    cache_dir = Path(cache_dir).expanduser().resolve()
    phc_root = Path(phc_root).expanduser().resolve()
    sk = f"{shape_key_from_params(model_type=sequence.model_type, gender=sequence.gender, betas=np.asarray(sequence.betas, dtype=np.float32).reshape(-1))}_phc_bundled"

    out_dir = cache_dir / "phc_bundled_mjcf"
    out_dir.mkdir(parents=True, exist_ok=True)

    source = os.environ.get("AMONGUS_PHC_MJCF_SOURCE", "bundled").strip().lower()
    mjcf_out: Path
    regenerated_mjcf = False
    if source == "smpl_robot":
        mjcf_out = out_dir / f"{sk}_smpl_robot.xml"
        if force_rewrite or not mjcf_out.is_file():
            from projects.genesis_ue_sync.sim_platform.embodiments.phc_smpl_robot_mjcf import try_write_smpl_robot_mjcf

            regenerated_mjcf = bool(try_write_smpl_robot_mjcf(sequence, mjcf_out))
            if not regenerated_mjcf:
                template = resolve_phc_bundle_mjcf_path(phc_root, sequence.gender)
                warnings.warn(
                    "AMONGUS_PHC_MJCF_SOURCE=smpl_robot failed (deps or SMPL_Robot error); "
                    "falling back to bundled PHC template.",
                    stacklevel=2,
                )
                mjcf_out = out_dir / f"{sk}_{template.stem}.xml"
                if force_rewrite or not mjcf_out.is_file():
                    shutil.copy2(template, mjcf_out)
                    regenerated_mjcf = True
    else:
        template = resolve_phc_bundle_mjcf_path(phc_root, sequence.gender)
        mjcf_out = out_dir / f"{sk}_{template.stem}.xml"
        if force_rewrite or not mjcf_out.is_file():
            shutil.copy2(template, mjcf_out)
            regenerated_mjcf = True

    if regenerated_mjcf:
        for marker in mjcf_out.parent.glob(f"{mjcf_out.name}.foot_box_scale_*"):
            marker.unlink(missing_ok=True)
    _shrink_foot_box_geoms(mjcf_out)
    proxy_gr = _proxy_geometry_from_phc_mjcf(mjcf_out, sequence=sequence)
    layout_out = out_dir / f"{mjcf_out.stem}_dof_layout.json"
    layout_data = build_phc_bundled_mjcf_layout_dict(mjcf_out)
    layout_out.write_text(json.dumps(layout_data, indent=2), encoding="utf-8")
    placeholder_urdf = out_dir / "phc_bundled_proxy_placeholder.urdf"
    _ensure_placeholder_urdf(placeholder_urdf)

    return mjcf_out, layout_out, placeholder_urdf, proxy_gr
