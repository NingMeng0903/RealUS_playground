#!/usr/bin/env python3
"""Audit DAE→OBJ baking, optional UE import A/B checklist, and URDF visual vs mesh PCA."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

_THIS_FILE = Path(__file__).resolve()
SRC_ROOT = next(parent for parent in (_THIS_FILE.parent, *_THIS_FILE.parents) if parent.name == "src")
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.project import project_paths
from projects.genesis_ue_sync.cli.render.media.convert_collada_to_obj import bake_collada_mesh, read_collada_asset_meta
from projects.genesis_ue_sync.mesh_audit.stats import mesh_descriptor_from_vertices, parse_obj_vertices, sorted_vertex_max_residual_m
from projects.genesis_ue_sync.sim_platform.scenes import load_sync_scene_spec
from projects.genesis_ue_sync.urdf import compose_link_visual_world_transform, compute_link_world_transforms, parse_urdf_model


def _parse_visual_basis_rpy_deg_env() -> tuple[float, float, float]:
    raw = str(os.environ.get("AMONGUS_UE_ROBOT_VISUAL_BASIS_RPY_DEG", "0 0 0")).strip()
    parts = raw.replace(",", " ").split()
    if len(parts) != 3:
        return (0.0, 0.0, 0.0)
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError:
        return (0.0, 0.0, 0.0)


def _emit_ue_import_checklist(*, dae_path: str, obj_path: str) -> dict[str, object]:
    return {
        "purpose": "Isolate UE StaticMesh import vs COLLADA→OBJ converter",
        "steps": [
            "Import DAE via Content Browser (if your engine build lists .dae).",
            "Import OBJ from the same link (repo converter output or this tool's --write-obj).",
            "Place two StaticMeshActors at origin with identity scale; set mesh on each.",
            "Compare bounding box extent axes in the StaticMesh editor (local space).",
            "If DAE and OBJ match but Genesis differs, suspect FK/URDF visual origin or world bridge.",
            "If DAE and OBJ differ, inspect convert_collada_to_obj.py and UE OBJ import settings.",
        ],
        "env": {
            "AMONGUS_PANDA_MESH_SOURCE": "dae or obj (see ue_common_scene_loader._panda_mesh_import_preference)",
            "AMONGUS_UE_ROBOT_VISUAL_BASIS_RPY_DEG": "optional mesh-local correction; use only after data from this audit",
        },
        "paths": {"dae": dae_path, "obj": obj_path},
    }


def _resolve_urdf_mesh_path(urdf_path: Path, mesh_filename: str) -> Path:
    raw = str(mesh_filename).strip()
    if raw.startswith("package://"):
        rest = raw[len("package://") :]
        idx = rest.find("/")
        raw = rest[idx + 1 :] if idx >= 0 else rest
    return (urdf_path.parent / raw).resolve()


def parse_args() -> argparse.Namespace:
    paths = project_paths(__file__)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dae", type=Path, help="Input COLLADA .dae for baking stats / OBJ compare")
    p.add_argument("--obj", type=Path, help="OBJ from converter or reference tool")
    p.add_argument("--write-obj", type=Path, help="Write baked mesh to this OBJ path")
    p.add_argument("--scene-spec", type=Path, help="Scene YAML for FK + per-link mesh PCA report")
    p.add_argument("--emit-ue-checklist", action="store_true", help="Print JSON UE editor A/B procedure")
    p.add_argument("--output-json", type=Path, default=paths.tmp_root / "robot_visual_mesh_audit.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report: dict[str, object] = {"ok": True, "issues": []}

    if args.emit_ue_checklist:
        dae_s = str(args.dae.expanduser().resolve()) if args.dae else ""
        obj_s = str(args.obj.expanduser().resolve()) if args.obj else ""
        print(json.dumps(_emit_ue_import_checklist(dae_path=dae_s, obj_path=obj_s), indent=2))
        if args.dae is None and args.obj is None and args.scene_spec is None:
            return

    basis = _parse_visual_basis_rpy_deg_env()
    report["amongus_ue_robot_visual_basis_rpy_deg"] = list(basis)

    if args.dae is not None:
        dae_path = args.dae.expanduser().resolve()
        up_axis, unit_m = read_collada_asset_meta(ET.parse(dae_path).getroot())
        report["collada_asset"] = {"up_axis": up_axis, "unit_meter": float(unit_m)}
        if up_axis and up_axis.upper() != "Z_UP":
            report["issues"].append(
                f"COLLADA up_axis={up_axis!r}: converter does not remap axes; Y_UP assets may need explicit rotation."
            )
        baked = bake_collada_mesh(dae_path)
        v_dae = np.asarray(baked.positions, dtype=np.float64)
        report["dae_baked"] = mesh_descriptor_from_vertices(v_dae).__dict__

        if args.write_obj is not None:
            out = args.write_obj.expanduser().resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8") as fh:
                for x, y, z in baked.positions:
                    fh.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
                for a, b, c in baked.faces:
                    fh.write(f"f {a} {b} {c}\n")
            report["wrote_obj"] = str(out)

        if args.obj is not None:
            v_obj = parse_obj_vertices(args.obj.expanduser().resolve())
            desc_obj = mesh_descriptor_from_vertices(v_obj)
            report["obj_file"] = desc_obj.__dict__
            try:
                max_res = sorted_vertex_max_residual_m(v_dae, v_obj)
                report["sorted_vertex_max_residual_m"] = float(max_res)
                if max_res > 1e-4:
                    report["issues"].append(f"DAE vs OBJ sorted vertex residual {max_res:.6e} m (expect ~0 for same bake)")
            except ValueError as exc:
                report["issues"].append(f"DAE vs OBJ vertex count mismatch: {exc}")
            try:
                vm = np.asarray(v_dae, dtype=np.float64).copy()
                vm[:, 1] *= -1.0
                max_mirror = sorted_vertex_max_residual_m(vm, v_obj)
                report["dae_mirror_y_vs_obj_sorted_vertex_max_residual_m"] = float(max_mirror)
                if max_mirror < float(report.get("sorted_vertex_max_residual_m", 1e9)):
                    report["note_mirror_y"] = (
                        "Residual improved after local Y mirror on DAE vertices; OBJ likely exported with "
                        "--mirror-y-for-unreal-obj-import."
                    )
            except ValueError as exc:
                report["issues"].append(f"DAE mirror-Y vs OBJ vertex count mismatch: {exc}")

    if args.scene_spec is not None:
        scene = load_sync_scene_spec(args.scene_spec.expanduser().resolve())
        urdf_path = scene.robot.resolved_urdf_path
        model = parse_urdf_model(urdf_path)
        fk = compute_link_world_transforms(
            urdf_path=urdf_path,
            base_pos_m=tuple(scene.robot.base_pos),
            base_quat_xyzw=scene.robot.base_quat_xyzw,
            joint_positions=[float(v) for v in scene.robot.joint_positions],
        )
        visual_report: dict[str, object] = {}
        for link_name, link in model.links.items():
            mesh_rel = link.visual_mesh
            if not mesh_rel:
                continue
            dae_path = _resolve_urdf_mesh_path(Path(urdf_path), str(mesh_rel))
            if not dae_path.is_file():
                report["issues"].append(f"Missing visual mesh for link {link_name}: {dae_path}")
                continue
            baked = bake_collada_mesh(dae_path)
            v_local = np.asarray(baked.positions, dtype=np.float64)
            desc_local = mesh_descriptor_from_vertices(v_local)
            lw = fk.get(link_name)
            if lw is None:
                continue
            vw = compose_link_visual_world_transform(
                lw,
                visual_origin_xyz=link.visual_origin_xyz,
                visual_origin_rpy=link.visual_origin_rpy,
                visual_basis_rpy_deg=basis,
            )
            r = vw[:3, :3]
            t = vw[:3, 3]
            v_world = (r @ v_local.T).T + t.reshape(1, 3)
            desc_world = mesh_descriptor_from_vertices(v_world)
            u_local = np.asarray(desc_local.principal_axis_0_unit, dtype=np.float64)
            u_world_mesh = np.asarray(desc_world.principal_axis_0_unit, dtype=np.float64)
            u_world_pred = r @ u_local
            na = float(np.linalg.norm(u_world_pred))
            nb = float(np.linalg.norm(u_world_mesh))
            align = float(abs(np.dot(u_world_pred / na, u_world_mesh / nb))) if na > 1e-12 and nb > 1e-12 else 0.0
            entry = {
                "visual_mesh": str(mesh_rel),
                "principal_axis_align_cos": align,
                "mesh_vertex_count": desc_local.vertex_count,
                "mesh_extent_m": list(desc_local.extent_m),
                "mesh_centroid_link_frame_m": list(desc_local.centroid_m),
            }
            obj_sibling = dae_path.with_suffix(".obj")
            if obj_sibling.is_file():
                try:
                    v_sib = parse_obj_vertices(obj_sibling)
                    max_sib = sorted_vertex_max_residual_m(v_local, v_sib)
                    entry["dae_vs_sibling_obj_sorted_vertex_max_residual_m"] = float(max_sib)
                    vm = np.asarray(v_local, dtype=np.float64).copy()
                    vm[:, 1] *= -1.0
                    max_ms = sorted_vertex_max_residual_m(vm, v_sib)
                    entry["dae_mirror_y_vs_sibling_obj_sorted_vertex_max_residual_m"] = float(max_ms)
                except ValueError as exc:
                    entry["sibling_obj_issue"] = str(exc)
            visual_report[link_name] = entry
        report["urdf_visual_mesh_fk"] = visual_report

    report["ok"] = len(report["issues"]) == 0
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
