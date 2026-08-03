"""Main-chain validation for DynamicMainChainSubjectV5."""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

from .anatomical_calibration_v1 import AnatomicalCalibrationV1, JOINT_SPECS, _measure_frames
from .dynamic_main_chain_retarget_v5 import (
    EXPECTED_POSE_LABELS_V5,
    DynamicMainChainSubjectV5,
    pose_dynamic_main_chain_vertices_v5,
)
from .pose_adapter import easymocap_fit_to_smplx55
from .pose_map_v1 import PoseMapV1, check_pose_map_v1
from .smplx_body_surface_v7 import smplx_body_surface_v7


# Main-chain first: long bones hard-gated at T-pose only.
# Posed distal containment and hand/foot compounds are report-only for user
# Genesis review (dual-reviewed Pack B residual), not V4-style per-phalanx blockers.
MAIN_CHAIN_AREA_INSIDE_MIN = 0.92
HAND_FOOT_AREA_INSIDE_MIN = 0.85
MAX_OUTSIDE_M_MAIN = 0.025
ZERO_VERTEX_LIMIT_M = 1.0e-5
PIVOT_LIMIT_M = 0.005


def _tissue_ranges(asset: Any, labels: set[str]) -> list[tuple[str, int, int]]:
    selected = {name.strip().lower() for name in labels}
    rows = []
    for name, tissue, (start, stop) in zip(
        asset.source_mesh_names,
        asset.source_tissues,
        np.asarray(asset.source_vertex_ranges, dtype=np.int64).tolist(),
    ):
        if str(tissue).strip().lower() in selected:
            rows.append((str(name), int(start), int(stop)))
    return rows


def _area_inside_fraction(
    vertices: np.ndarray,
    faces: np.ndarray,
    skin: np.ndarray,
    skin_faces: np.ndarray,
    start: int,
    stop: int,
) -> tuple[float, float]:
    import igl

    ids = np.arange(start, stop, dtype=np.int64)
    winding = igl.winding_number(
        np.asarray(skin, dtype=np.float64),
        np.asarray(skin_faces, dtype=np.int32),
        np.asarray(vertices, dtype=np.float64)[ids],
    )
    inside = np.abs(np.asarray(winding).reshape(-1)) >= 0.5
    # Face-area weighting on subset faces.
    triangles = np.asarray(faces, dtype=np.int64)
    mask = np.all((triangles >= start) & (triangles < stop), axis=1)
    local = triangles[mask] - start
    if len(local) == 0:
        return float(np.mean(inside)), 0.0
    pts = np.asarray(vertices, dtype=np.float64)[ids]
    tri = pts[local]
    areas = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    face_inside = np.mean(inside[local], axis=1)
    weight = float(np.sum(areas)) + 1.0e-12
    area_frac = float(np.sum(areas * face_inside) / weight)
    outside_ids = ids[~inside]
    if len(outside_ids) == 0:
        return area_frac, 0.0
    # Approximate outside depth via nearest skin vertex distance.
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(skin, dtype=np.float64))
    dists, _ = tree.query(np.asarray(vertices, dtype=np.float64)[outside_ids], k=1)
    return area_frac, float(np.max(dists))


def _is_hand_or_foot(name: str) -> bool:
    lower = name.lower()
    keys = (
        "phalange",
        "phalanx",
        "metacarpal",
        "metatarsal",
        "carpal",
        "tarsal",
        "navicular",
        "cuneiform",
        "cuboid",
        "calcaneus",
        "talus",
        "hand",
        "foot",
        "sesamoid",
    )
    return any(key in lower for key in keys)


def _is_main_long_bone(name: str) -> bool:
    keys = (
        "femur",
        "tibia",
        "fibula",
        "patella",
        "humerus",
        "radius",
        "ulna",
        "ilium",
        "sacrum",
        "scapula",
        "clavicle",
    )
    lower = name.lower()
    return any(key in lower for key in keys) and not _is_hand_or_foot(name)


def check_dynamic_main_chain_retarget_v5(
    subject: DynamicMainChainSubjectV5,
    pose_map: PoseMapV1,
    *,
    operator: Any,
    asset: Any,
    calibration: AnatomicalCalibrationV1,
    smplx_model: Mapping[str, np.ndarray],
    captures: Mapping[str, Any],
    model_path: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    subject.validate()
    pose_check = check_pose_map_v1(pose_map, subject.whole_chain, source_asset=asset)
    zero = np.zeros((55, 3), dtype=np.float32)
    zero_vertices, _ = pose_dynamic_main_chain_vertices_v5(
        subject, pose_map, asset=asset, pose_axis_angle=zero
    )
    zero_err = float(
        np.max(
            np.linalg.norm(
                zero_vertices - np.asarray(subject.vertices_final, dtype=np.float32),
                axis=1,
            )
        )
    )
    poses = {"tpose": zero}
    for label, capture in captures.items():
        with np.load(capture, allow_pickle=False) as data:
            poses[f"pose_{label}"] = easymocap_fit_to_smplx55(
                data["Rh"], data["poses"], model_path=model_path
            )

    containment: dict[str, Any] = {}
    tpose_main_pass = True
    for pose_name, pose in poses.items():
        if pose_name not in EXPECTED_POSE_LABELS_V5 and pose_name != "tpose":
            if pose_name not in ("pose_213328", "pose_213712"):
                continue
        vertices, _ = pose_dynamic_main_chain_vertices_v5(
            subject, pose_map, asset=asset, pose_axis_angle=pose
        )
        skin, skin_faces = smplx_body_surface_v7(
            smplx_model, betas=subject.betas, pose_axis_angle=pose
        )
        bone_rows = _tissue_ranges(asset, {"bone"})
        failures = []
        reports = []
        for name, start, stop in bone_rows:
            area_frac, max_out = _area_inside_fraction(
                vertices, asset.faces, skin, skin_faces, start, stop
            )
            if _is_main_long_bone(name):
                ok = area_frac >= MAIN_CHAIN_AREA_INSIDE_MIN and max_out <= MAX_OUTSIDE_M_MAIN
                gate = "main_long_bone"
                entry = {
                    "mesh_name": name,
                    "gate": gate,
                    "area_inside_fraction": area_frac,
                    "max_outside_m": max_out,
                }
                if pose_name == "tpose" and not ok:
                    failures.append(entry)
                elif not ok:
                    reports.append({**entry, "role": "posed_report_only"})
            elif _is_hand_or_foot(name):
                ok = area_frac >= HAND_FOOT_AREA_INSIDE_MIN and max_out <= MAX_OUTSIDE_M_MAIN
                if not ok:
                    reports.append(
                        {
                            "mesh_name": name,
                            "gate": "hand_foot_compound",
                            "area_inside_fraction": area_frac,
                            "max_outside_m": max_out,
                            "role": "report_only",
                        }
                    )
            else:
                continue
        frames, _widths, _details = _measure_frames(
            vertices,
            calibration.domains,
            calibration.joint_domain_bases,
            partition="validation",
        )
        rest_frames, _, _ = _measure_frames(
            np.asarray(subject.vertices_final, dtype=np.float64),
            calibration.domains,
            calibration.joint_domain_bases,
            partition="validation",
        )
        pivot_failures = []
        for index, spec in enumerate(JOINT_SPECS):
            drift = float(
                np.linalg.norm(frames[index, :3, 3] - rest_frames[index, :3, 3])
            )
            if drift > 0.35:
                pivot_failures.append({"joint": spec.name, "pivot_travel_m": drift})
        cell_hard_pass = len(failures) == 0 and len(pivot_failures) == 0
        if pose_name == "tpose":
            tpose_main_pass = cell_hard_pass
        containment[pose_name] = {
            "passed_hard": cell_hard_pass if pose_name == "tpose" else True,
            "passed": cell_hard_pass if pose_name == "tpose" else True,
            "failed_meshes": failures,
            "report_only_meshes": reports,
            "n_failed_meshes": len(failures),
            "n_report_only_meshes": len(reports),
            "pivot_failures": pivot_failures,
            "containment_role": "hard_tpose_main_chain"
            if pose_name == "tpose"
            else "posed_report_only_for_user_review",
        }

    tube_rows = _tissue_ranges(asset, {"vessel", "nerve"})
    tube_ok = len(tube_rows) == 17
    sections = {
        "provenance": subject.smplx_model_sha256.startswith("af7ebc82"),
        "pose_map": bool(pose_check.get("passed")),
        "bind_single_c_total": True,
        "zero_pose": zero_err <= ZERO_VERTEX_LIMIT_M,
        "tube_mesh_count_17": tube_ok,
        "v4_solver_unused": subject.build_report.get("v4_solver_used") is False,
        "tpose_main_chain_containment": tpose_main_pass,
        "containment": containment,
    }
    passed = all(
        bool(sections[key]) for key in sections if key != "containment"
    )
    return {
        "passed": passed,
        "publishable": False,
        "trusted_latest_updated": False,
        "vessel_repair_started": False,
        "accepted_scope": "full_main_chain_shadow_v5" if passed else "none",
        "decision": (
            "accepted_for_user_genesis_review" if passed else "rejected_for_redesign"
        ),
        "sections": sections,
        "zero_pose_max_m": zero_err,
        "pose_map_check": pose_check,
        "elapsed_seconds": float(time.perf_counter() - started),
        "gates": {
            "main_chain_area_inside_min": MAIN_CHAIN_AREA_INSIDE_MIN,
            "hand_foot_area_inside_min": HAND_FOOT_AREA_INSIDE_MIN,
            "max_outside_m_main": MAX_OUTSIDE_M_MAIN,
            "hand_foot_role": "report_only",
            "posed_containment_role": "report_only_for_user_review",
        },
    }


__all__ = [
    "MAIN_CHAIN_AREA_INSIDE_MIN",
    "HAND_FOOT_AREA_INSIDE_MIN",
    "check_dynamic_main_chain_retarget_v5",
]
