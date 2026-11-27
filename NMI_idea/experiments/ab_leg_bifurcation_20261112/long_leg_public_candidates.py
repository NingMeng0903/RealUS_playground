"""Read-only public-placement candidates for the long-leg view table.

This helper is separate from the recorded ``LONG_LEG_VIEWS_CHECKED`` run.  It
recreates that run's original inward-yaw-pi table under a new filename, then
evaluates one common public placement found by a bounded geometry/reachability
search.  No trial output or anatomy observation is read.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

import long_leg_views as llv


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "LONG_LEG_VIEWS.npz"
SURFACE = llv.SURFACE_PATH
PLACEMENT = llv.PLACEMENT_PATH
YAW_ONLY_OUT = HERE / "LONG_LEG_VIEWS_CHECKED_YAW_ONLY.npz"
YAW_ONLY_JSON = HERE / "LONG_LEG_VIEWS_CHECKED_YAW_ONLY.json"
COMMON_OUT = HERE / "LONG_LEG_VIEWS_COMMON.npz"
COMMON_JSON = HERE / "LONG_LEG_VIEWS_COMMON.json"


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _restore_yaw_only(
    *, source_path: Path = SOURCE, output_path: Path = YAW_ONLY_OUT, audit_path: Path = YAW_ONLY_JSON
) -> dict[str, Any]:
    """Recreate the original K=7 inward-yaw-pi table without touching its old name."""

    old = _load(source_path)
    with np.load(source_path, allow_pickle=False) as source:
        old_meta = json.loads(str(source["metadata_json"].item()))
    offsets = np.asarray([-24.0, -16.0, -8.0, 0.0, 8.0, 16.0, 24.0], dtype=float)
    surface = llv._load_surface_atlas(SURFACE)
    geom = llv._surface_views(surface, old["skin_m"], old["frames"], offsets_mm=offsets)
    sys.path.insert(0, str(llv.CAPABILITY_DIR))
    from capability import CapabilityEvaluator  # type: ignore

    placement = json.loads(PLACEMENT.read_text(encoding="utf-8"))
    q_seed = np.asarray(placement["q_seed"], dtype=float)
    anchor = np.asarray(placement["T_world_body"], dtype=float)
    yaw = np.diag([-1.0, -1.0, 1.0])
    checked_anchor = anchor.copy()
    checked_anchor[:3, :3] = anchor[:3, :3] @ yaw
    robot = llv._robot_views(
        geom.points_m, old["frames"], anchor=checked_anchor, q_seed=q_seed,
        capability=CapabilityEvaluator(), static=True,
    )
    metadata = dict(old_meta)
    metadata.update(
        {
            "schema": "long_leg_views_checked_v1",
            "status": "geometry_and_public_robot_static_ik_checked_public_inward_yaw_pi",
            "source_original_table": {"path": str(source_path.resolve()), "sha256": llv._sha256(source_path)},
            "checked_lateral_offsets_nominal_mm": offsets.tolist(),
            "checked_public_placement": {
                "base_placement": str(PLACEMENT.resolve()),
                "base_anchor": anchor.tolist(),
                "rotation_local_axis": "inward/TCP-z",
                "rotation_deg": 180.0,
                "checked_anchor": checked_anchor.tolist(),
                "interpretation": "contact-facing inward axis unchanged; long/lateral axes reversed",
            },
            "station_count": int(len(old["row_ids"])),
            "view_projection_residual_max_mm": float(np.max(geom.projection_residual_m) * llv.MM_PER_M),
            "view_projection_residual_median_mm": float(np.median(geom.projection_residual_m) * llv.MM_PER_M),
            "view_projection_within_tolerance_count": int(np.sum(geom.projection_feasible)),
            "view_projection_total_count": int(geom.projection_feasible.size),
            "public_robot": {
                "placement_rotation": "local inward/TCP-z yaw π",
                "static_requested": True,
                "candidate_count_per_view": int(robot["candidate_rail_m"].shape[-1]),
                "chosen_static_found_count": int(np.sum(robot["chosen_static_found"])),
                "chosen_view_count": int(robot["chosen_static_found"].size),
                "candidate_static_found_count": int(np.sum(robot["candidate_static_found"])),
                "candidate_total_count": int(robot["candidate_static_found"].size),
                "dird_margin_chosen_min": float(np.min(robot["chosen_dird_margin"])),
                "dird_margin_chosen_max": float(np.max(robot["chosen_dird_margin"])),
                "chosen_projection_masked_static_count": int(np.sum(np.asarray(robot["chosen_static_found"], dtype=bool) & geom.projection_feasible)),
                "chosen_projection_masked_view_count": int(np.sum(geom.projection_feasible)),
            },
        }
    )
    arrays = dict(old)
    arrays.pop("metadata_json", None)
    for key in (
        "views_m", "view_lat_mm", "view_projection_residual_m", "view_projection_feasible",
        "view_chart_ids", "view_coordinates_uv_m", "view_face_ids", "view_face",
        "view_barycentric", "view_bary", "dird_margin", "static_found", "q_solution",
        "chosen_candidate_index", "chosen_rail_m", "candidate_rail_m", "candidate_q0",
        "candidate_dird_margin", "candidate_proxy_pass", "candidate_static_found",
        "candidate_q_solution", "candidate_static_seconds", "robot_from_atlas", "target_robot",
    ):
        if key in arrays:
            arrays[f"original_{key}"] = np.asarray(arrays[key]).copy()
    arrays.update(
        {
            "views_m": geom.points_m,
            "view_lat_mm": geom.actual_lat_mm,
            "view_projection_residual_m": geom.projection_residual_m,
            "view_projection_feasible": geom.projection_feasible,
            "view_chart_ids": geom.chart_ids,
            "view_coordinates_uv_m": geom.coordinates_uv_m,
            "view_face_ids": geom.face_ids,
            "view_face": geom.face_ids,
            "view_barycentric": geom.barycentric,
            "view_bary": geom.barycentric,
            "dird_margin": np.asarray(robot["chosen_dird_margin"], dtype=float),
            "static_found": np.asarray(robot["chosen_static_found"], dtype=bool),
            "q_solution": np.asarray(robot["chosen_q_solution"], dtype=float),
            "chosen_candidate_index": np.asarray(robot["chosen_candidate_index"], dtype=np.int64),
            "chosen_rail_m": np.asarray(robot["chosen_rail_m"], dtype=float),
            "candidate_rail_m": np.asarray(robot["candidate_rail_m"], dtype=float),
            "candidate_q0": np.asarray(robot["candidate_rail_m"], dtype=float),
            "candidate_dird_margin": np.asarray(robot["candidate_dird_margin"], dtype=float),
            "candidate_proxy_pass": np.asarray(robot["candidate_proxy_pass"], dtype=bool),
            "candidate_static_found": np.asarray(robot["candidate_static_found"], dtype=bool),
            "candidate_q_solution": np.asarray(robot["candidate_q_solution"], dtype=float),
            "candidate_static_seconds": np.asarray(robot["candidate_static_seconds"], dtype=float),
            "robot_from_atlas": np.asarray(robot["robot_from_atlas"], dtype=float),
            "target_robot": np.asarray(robot["target_robot"], dtype=float),
        }
    )
    np.savez_compressed(output_path, metadata_json=np.asarray(json.dumps(llv._jsonable(metadata), sort_keys=True)), **arrays)
    metadata["output"] = str(output_path.resolve())
    metadata["audit_output"] = str(audit_path.resolve())
    audit_path.write_text(json.dumps(llv._jsonable(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def _common_anchor(anchor: np.ndarray) -> np.ndarray:
    """Best bounded geometry-only common placement from public score search."""

    from scipy.spatial.transform import Rotation

    # Local inward/TCP-z +60 degrees, followed by a -60 mm local-long and
    # -60 mm local-lateral translation.  The search used only public dIRD
    # margins; these values are fixed here as an auditable candidate.
    rotation = Rotation.from_rotvec(np.deg2rad(60.0) * np.asarray([0.0, 0.0, 1.0])).as_matrix()
    out = anchor.copy()
    out[:3, :3] = anchor[:3, :3] @ rotation
    out[:3, 3] += out[:3, :3] @ np.asarray([-0.06, -0.06, 0.0])
    return out


def build_common(*, source_path: Path = SOURCE, output_path: Path = COMMON_OUT, audit_path: Path = COMMON_JSON) -> dict[str, Any]:
    """Build a K=7 common-placement table and retain the base table fields."""

    old = _load(source_path)
    with np.load(source_path, allow_pickle=False) as source:
        old_meta = json.loads(str(source["metadata_json"].item()))
    offsets = np.asarray([-24.0, -16.0, -8.0, 0.0, 8.0, 16.0, 24.0], dtype=float)
    geom = llv._surface_views(llv._load_surface_atlas(SURFACE), old["skin_m"], old["frames"], offsets_mm=offsets)
    sys.path.insert(0, str(llv.CAPABILITY_DIR))
    from capability import CapabilityEvaluator  # type: ignore

    placement = json.loads(PLACEMENT.read_text(encoding="utf-8"))
    q_seed = np.asarray(placement["q_seed"], dtype=float)
    base_anchor = np.asarray(placement["T_world_body"], dtype=float)
    anchor = _common_anchor(base_anchor)
    robot = llv._robot_views(
        geom.points_m, old["frames"], anchor=anchor, q_seed=q_seed,
        capability=CapabilityEvaluator(), static=True,
    )
    metadata = dict(old_meta)
    metadata.update(
        {
            "schema": "long_leg_views_common_public_placement_v1",
            "status": "geometry_and_public_robot_static_ik_common_placement",
            "source_original_table": {"path": str(source_path.resolve()), "sha256": llv._sha256(source_path)},
            "checked_lateral_offsets_nominal_mm": offsets.tolist(),
            "common_public_placement": {
                "base_placement": str(PLACEMENT.resolve()),
                "base_anchor": base_anchor.tolist(),
                "local_rotation_axis": "inward/TCP-z",
                "rotation_deg": 60.0,
                "local_translation_m": [-0.06, -0.06, 0.0],
                "anchor": anchor.tolist(),
                "selection_basis": "bounded geometry-only dIRD margin search; no trial/anatomy outcome",
            },
            "view_projection_residual_max_mm": float(np.max(geom.projection_residual_m) * llv.MM_PER_M),
            "view_projection_residual_median_mm": float(np.median(geom.projection_residual_m) * llv.MM_PER_M),
            "view_projection_within_tolerance_count": int(np.sum(geom.projection_feasible)),
            "view_projection_total_count": int(geom.projection_feasible.size),
            "public_robot": {
                "static_requested": True,
                "candidate_count_per_view": int(robot["candidate_rail_m"].shape[-1]),
                "chosen_static_found_count": int(np.sum(robot["chosen_static_found"])),
                "chosen_view_count": int(robot["chosen_static_found"].size),
                "candidate_static_found_count": int(np.sum(robot["candidate_static_found"])),
                "candidate_total_count": int(robot["candidate_static_found"].size),
                "dird_margin_chosen_min": float(np.min(robot["chosen_dird_margin"])),
                "dird_margin_chosen_max": float(np.max(robot["chosen_dird_margin"])),
                "chosen_projection_masked_static_count": int(np.sum(np.asarray(robot["chosen_static_found"], dtype=bool) & geom.projection_feasible)),
                "chosen_projection_masked_view_count": int(np.sum(geom.projection_feasible)),
                "evaluator_manifest": llv._jsonable(CapabilityEvaluator().manifest()),
            },
        }
    )
    arrays = dict(old)
    arrays.pop("metadata_json", None)
    for key in ("views_m", "view_lat_mm", "view_projection_residual_m", "view_projection_feasible", "view_chart_ids", "view_coordinates_uv_m", "view_face_ids", "view_face", "view_barycentric", "view_bary", "dird_margin", "static_found", "q_solution", "chosen_candidate_index", "chosen_rail_m", "candidate_rail_m", "candidate_q0", "candidate_dird_margin", "candidate_proxy_pass", "candidate_static_found", "candidate_q_solution", "candidate_static_seconds", "robot_from_atlas", "target_robot"):
        if key in arrays:
            arrays[f"base_{key}"] = np.asarray(arrays[key]).copy()
    arrays.update(
        {
            "views_m": geom.points_m, "view_lat_mm": geom.actual_lat_mm,
            "view_projection_residual_m": geom.projection_residual_m, "view_projection_feasible": geom.projection_feasible,
            "view_chart_ids": geom.chart_ids, "view_coordinates_uv_m": geom.coordinates_uv_m,
            "view_face_ids": geom.face_ids, "view_face": geom.face_ids,
            "view_barycentric": geom.barycentric, "view_bary": geom.barycentric,
            "dird_margin": np.asarray(robot["chosen_dird_margin"], dtype=float),
            "static_found": np.asarray(robot["chosen_static_found"], dtype=bool),
            "q_solution": np.asarray(robot["chosen_q_solution"], dtype=float),
            "chosen_candidate_index": np.asarray(robot["chosen_candidate_index"], dtype=np.int64),
            "chosen_rail_m": np.asarray(robot["chosen_rail_m"], dtype=float),
            "candidate_rail_m": np.asarray(robot["candidate_rail_m"], dtype=float), "candidate_q0": np.asarray(robot["candidate_rail_m"], dtype=float),
            "candidate_dird_margin": np.asarray(robot["candidate_dird_margin"], dtype=float),
            "candidate_proxy_pass": np.asarray(robot["candidate_proxy_pass"], dtype=bool),
            "candidate_static_found": np.asarray(robot["candidate_static_found"], dtype=bool),
            "candidate_q_solution": np.asarray(robot["candidate_q_solution"], dtype=float),
            "candidate_static_seconds": np.asarray(robot["candidate_static_seconds"], dtype=float),
            "robot_from_atlas": np.asarray(robot["robot_from_atlas"], dtype=float), "target_robot": np.asarray(robot["target_robot"], dtype=float),
            "static_found_projection_masked": np.asarray(robot["chosen_static_found"], dtype=bool) & geom.projection_feasible,
            "dird_margin_projection_masked": np.where(geom.projection_feasible, np.asarray(robot["chosen_dird_margin"], dtype=float), np.nan),
            "q_solution_projection_masked": np.where(geom.projection_feasible[..., None], np.asarray(robot["chosen_q_solution"], dtype=float), np.nan),
        }
    )
    np.savez_compressed(output_path, metadata_json=np.asarray(json.dumps(llv._jsonable(metadata), sort_keys=True)), **arrays)
    metadata["output"] = str(output_path.resolve()); metadata["audit_output"] = str(audit_path.resolve())
    audit_path.write_text(json.dumps(llv._jsonable(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    print(json.dumps({"yaw_only": _restore_yaw_only(), "common": build_common()}, indent=2, default=llv._jsonable))

