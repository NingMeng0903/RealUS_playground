"""Bone-first femur diagnostics: anatomical span vs mesh span vs skin outside."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    JOINT_SPECS,
    _measure_frames,
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.dynamic_main_chain_validation_v5 import (
    _area_inside_fraction,
    _tissue_ranges,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import easymocap_fit_to_smplx55
from projects.genesis_ue_sync.anatomy_retarget.pose_map_v1 import (
    build_pose_map_v1,
    pose_whole_chain_vertices,
)
from projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    load_smplx_model_v7,
    require_frozen_smplx_male_v7,
    smplx_body_surface_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
    materialize_subject,
)
from projects.genesis_ue_sync.anatomy_retarget.whole_chain_rest_fit_v1 import (
    load_whole_chain_rest_fit_v1,
)


def _mesh_axis_span(
    vertices: np.ndarray, ids: np.ndarray, origin: np.ndarray, axis: np.ndarray
) -> float:
    pts = np.asarray(vertices, dtype=np.float64)[np.asarray(ids, dtype=np.int64)]
    unit = np.asarray(axis, dtype=np.float64)
    unit = unit / np.linalg.norm(unit)
    proj = (pts - np.asarray(origin, dtype=np.float64)) @ unit
    return float(proj.max() - proj.min())


def _femur_patella_rows(asset: Any) -> list[tuple[str, int, int]]:
    rows = []
    for name, start, stop in _tissue_ranges(asset, {"bone"}):
        lower = name.lower()
        if "femur" in lower or "patella" in lower:
            rows.append((name, start, stop))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--smplx-model", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--shadow", type=Path, required=True)
    parser.add_argument("--subject", default="213328")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    operator = load_source_operator(args.operator.resolve(), mmap=True)
    calibration = load_anatomical_calibration_v1(
        args.calibration.resolve(),
        operator=operator,
        required_scope="full_main_chain",
    )
    model_path, model_sha = require_frozen_smplx_male_v7(args.smplx_model)
    model = load_smplx_model_v7(model_path)
    with np.load(args.capture.resolve(), allow_pickle=False) as data:
        betas = np.asarray(data["shapes"], dtype=np.float64).reshape(-1)[:10]
        pose = easymocap_fit_to_smplx55(
            data["Rh"], data["poses"], model_path=model_path
        )
    value = load_whole_chain_rest_fit_v1(
        args.shadow.resolve() / f"subject_{args.subject}",
        operator=operator,
        calibration=calibration,
        smplx_model=model,
        smplx_model_sha256=model_sha,
        recheck=False,
    )
    asset = materialize_subject(operator, betas=betas, gender="male").rigged_asset
    pose_map = build_pose_map_v1(
        value,
        asset=asset,
        calibration=calibration,
        oracle_path=args.oracle.resolve(),
        source_operator_digest=operator.runtime_digest(validate=False),
    )
    lookup = {spec.name: index for index, spec in enumerate(JOINT_SPECS)}
    prefit_frames, _, _ = _measure_frames(
        value.vertices_prefit,
        calibration.domains,
        calibration.joint_domain_bases,
        partition="fit",
    )
    final_frames, _, _ = _measure_frames(
        value.vertices_final,
        calibration.domains,
        calibration.joint_domain_bases,
        partition="fit",
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "FemurBoneFirstDiagV8",
        "subject": args.subject,
        "shadow": str(args.shadow.resolve()),
        "method": "anatomical_span_vs_mesh_span_vs_outside",
        "sides": {},
        "poses": {},
        "publishable": False,
    }
    for side in ("left", "right"):
        hip_i = lookup[f"{side}_hip"]
        knee_i = lookup[f"{side}_knee"]
        hip_p = prefit_frames[hip_i, :3, 3]
        knee_p = prefit_frames[knee_i, :3, 3]
        hip_f = final_frames[hip_i, :3, 3]
        knee_f = final_frames[knee_i, :3, 3]
        axis_p = knee_p - hip_p
        anatomical_span_prefit = float(np.linalg.norm(axis_p))
        anatomical_span_final = float(np.linalg.norm(knee_f - hip_f))
        femur_ids = []
        for name, start, stop in _tissue_ranges(asset, {"bone"}):
            if f"femur" in name.lower() and (
                name.endswith("_L") if side == "left" else name.endswith("_R")
            ):
                femur_ids.append(np.arange(start, stop, dtype=np.int64))
            if "femur" in name.lower() and side[0].upper() in name:
                # Femur_L / Femur_R naming
                if (side == "left" and name.upper().endswith("L")) or (
                    side == "right" and name.upper().endswith("R")
                ):
                    femur_ids.append(np.arange(start, stop, dtype=np.int64))
        # Prefer mesh names containing Femur and side suffix.
        ids = []
        for name, start, stop in _tissue_ranges(asset, {"bone"}):
            n = name.lower()
            if "femur" not in n:
                continue
            if side == "left" and n.endswith("_l"):
                ids = np.arange(start, stop, dtype=np.int64)
                break
            if side == "right" and n.endswith("_r"):
                ids = np.arange(start, stop, dtype=np.int64)
                break
        if len(ids) == 0:
            raise RuntimeError(f"missing femur mesh ids for {side}")
        mesh_span_prefit = _mesh_axis_span(
            value.vertices_prefit, ids, hip_p, axis_p
        )
        mesh_span_final = _mesh_axis_span(
            value.vertices_final, ids, hip_f, knee_f - hip_f
        )
        br = value.build_report.get("centerlines", {}).get(side, {})
        report["sides"][side] = {
            "anatomical_span_prefit_m": anatomical_span_prefit,
            "anatomical_span_final_m": anatomical_span_final,
            "mesh_span_prefit_m": mesh_span_prefit,
            "mesh_span_final_m": mesh_span_final,
            "mesh_minus_anatomical_prefit_m": mesh_span_prefit - anatomical_span_prefit,
            "femur_length_scale_reported": br.get("femur_length_scale"),
            "femur_requested_skin_scale_reported": br.get("femur_requested_skin_scale"),
            "policy": "bone_first_anatomical_segment_not_skin_proportional",
        }
    for pose_name, pose_aa in (
        ("tpose", np.zeros((55, 3), dtype=np.float32)),
        ("pose_capture", pose),
    ):
        verts, _ = pose_whole_chain_vertices(
            value, pose_map, source_asset=asset, pose_axis_angle=pose_aa
        )
        skin, skin_faces = smplx_body_surface_v7(
            model, betas=betas, pose_axis_angle=pose_aa
        )
        meshes = []
        for name, start, stop in _femur_patella_rows(asset):
            area, max_out = _area_inside_fraction(
                verts, asset.faces, skin, skin_faces, start, stop
            )
            meshes.append(
                {
                    "mesh_name": name,
                    "area_inside_fraction": area,
                    "max_outside_m": max_out,
                }
            )
        report["poses"][pose_name] = {"meshes": meshes}
    path = output / "femur_bone_first_diag.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"FemurBoneFirstDiagV8 -> {path}")
    for side, row in report["sides"].items():
        print(
            f"  {side}: anat={row['anatomical_span_prefit_m']:.4f} "
            f"mesh={row['mesh_span_prefit_m']:.4f} "
            f"delta={row['mesh_minus_anatomical_prefit_m']*1000:.1f}mm"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
