"""Male multi-pose main-chain rest retarget with the frozen 142 motion rig.

V1 demonstrated that changing the authored local bind can preserve zero pose
while moving a complete hand tens of millimetres in recorded poses.  V2 keeps
the 235-controller Blender bind as the motion authority and separates it from
the one-time rest transport used to fit bone geometry and coupled tubes.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .anatomical_calibration_v1 import (
    AnatomicalCalibrationV1,
    JOINT_SPECS,
    _measure_frames,
)
from .anatomy_lbs import source_bone_posed_global
from .chain_containment_v1 import _summary, _vertex_areas
from .chain_rest_fit_v1 import (
    ChainRestFitSubjectV1,
    _array_digest,
    _global_to_local,
    _sha256,
    _weighted_rest_correction,
)
from .dynamic_main_chain_validation_v2 import dynamic_main_chain_regions_v2
from .smplx_body_surface_v7 import (
    FROZEN_SMPLX_MALE_SHA256,
    smplx_body_surface_v7,
)
from .whole_chain_rest_fit_v1 import (
    BASELINE_COMMIT,
    build_whole_chain_rest_fit_v1,
)


DYNAMIC_MAIN_CHAIN_RETARGET_SCHEMA_VERSION = 2
DYNAMIC_MAIN_CHAIN_RETARGET_KIND = "DynamicMainChainSubjectV2"
TERMINAL_LABELS = ("left_hand", "right_hand", "left_foot", "right_foot")
_TERMINAL_ROOTS = {
    "left_hand": "Wrist_Rotate_L",
    "right_hand": "Wrist_Rotate_R1",
    "left_foot": "Ankle_Rot_L",
    "right_foot": "Ankle_Rot_R",
}
_TERMINAL_BOUNDS = {
    "left_hand": (0.012, np.deg2rad(5.0)),
    "right_hand": (0.012, np.deg2rad(5.0)),
    "left_foot": (0.035, np.deg2rad(10.0)),
    "right_foot": (0.035, np.deg2rad(10.0)),
}


def _descendants(parents: np.ndarray, root: int) -> np.ndarray:
    selected = {int(root)}
    changed = True
    while changed:
        changed = False
        for index, parent in enumerate(np.asarray(parents, dtype=np.int64).tolist()):
            if index not in selected and int(parent) in selected:
                selected.add(index)
                changed = True
    return np.asarray(sorted(selected), dtype=np.int64)


def _tube_vertex_ids(asset: Any) -> np.ndarray:
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    chunks = [
        np.arange(int(start), int(stop), dtype=np.int64)
        for tissue, (start, stop) in zip(asset.source_tissues, ranges.tolist())
        if str(tissue).strip().lower() in {"vessel", "nerve"}
    ]
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)


def _terminal_fit_vertex_ids(
    asset: Any,
    *,
    label: str,
    terminal_vertex_ids: np.ndarray,
) -> np.ndarray:
    if not label.endswith("_foot"):
        return np.asarray(terminal_vertex_ids, dtype=np.int64)
    terminal_mask = np.zeros(len(asset.vertices_rest), dtype=bool)
    terminal_mask[np.asarray(terminal_vertex_ids, dtype=np.int64)] = True
    chunks = []
    for name, tissue, (start, stop) in zip(
        asset.source_mesh_names, asset.source_tissues, asset.source_vertex_ranges
    ):
        start_i, stop_i = int(start), int(stop)
        if (
            str(tissue).strip().lower() == "bone"
            and "phalanx" not in str(name).strip().lower()
            and np.any(terminal_mask[start_i:stop_i])
        ):
            chunks.append(np.arange(start_i, stop_i, dtype=np.int64))
    if not chunks:
        raise ValueError(f"V2 found no major rigid foot vertices for {label}")
    return np.concatenate(chunks)


def _rigid_about_pivot(parameters: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    values = np.asarray(parameters, dtype=np.float64).reshape(6)
    rotation = Rotation.from_rotvec(values[3:]).as_matrix()
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = values[:3] + pivot - rotation @ pivot
    return result


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (
        np.asarray(points, dtype=np.float64) @ transform[:3, :3].T
        + transform[:3, 3]
    )


def _signed_distance_details(
    points: np.ndarray,
    skin_vertices: np.ndarray,
    skin_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import igl

    query = np.asarray(points, dtype=np.float64)
    skin = np.asarray(skin_vertices, dtype=np.float64)
    faces = np.asarray(skin_faces, dtype=np.int32)
    winding = np.asarray(igl.winding_number(skin, faces, query)).reshape(-1)
    squared, _face, closest = igl.point_mesh_squared_distance(query, skin, faces)
    distance = np.sqrt(np.maximum(0.0, np.asarray(squared, dtype=np.float64)))
    inside = np.abs(winding) >= 0.5
    signed = np.where(inside, -distance, distance)
    direction = query - np.asarray(closest, dtype=np.float64)
    norm = np.linalg.norm(direction, axis=1)
    gradient = np.zeros_like(direction)
    valid = norm > 1.0e-10
    gradient[valid] = direction[valid] / norm[valid, None]
    gradient[inside] *= -1.0
    if not (
        len(signed) == len(query)
        and np.all(np.isfinite(signed))
        and np.all(np.isfinite(gradient))
    ):
        raise ValueError("terminal signed-distance query returned invalid values")
    return signed, np.asarray(closest, dtype=np.float64), gradient


def _pose_affines(
    asset: Any,
    *,
    bind_global: np.ndarray,
    vertex_ids: np.ndarray,
    pose_axis_angle: np.ndarray,
) -> np.ndarray:
    posed_global = source_bone_posed_global(asset, pose_axis_angle)
    transforms = posed_global @ np.linalg.inv(bind_global)
    indices = np.asarray(asset.driver_indices, dtype=np.int64)[vertex_ids]
    weights = np.asarray(asset.driver_weights, dtype=np.float64)[vertex_ids]
    return np.einsum("vk,vkab->vab", weights, transforms[indices])


def _apply_affines(points: np.ndarray, affines: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        (np.asarray(points, dtype=np.float64), np.ones((len(points), 1))), axis=1
    )
    return np.einsum("vab,vb->va", affines, homogeneous)[:, :3]


def _fk(local: np.ndarray, parents: np.ndarray) -> np.ndarray:
    values = np.asarray(local, dtype=np.float64)
    result = np.empty_like(values)
    for index, parent in enumerate(np.asarray(parents, dtype=np.int64).tolist()):
        result[index] = values[index] if parent < 0 else result[parent] @ values[index]
    return result


def _exact_terminal_score(
    signed: np.ndarray,
    area_weights: np.ndarray,
    *,
    margin_m: float = 0.001,
) -> float:
    weights = np.asarray(area_weights, dtype=np.float64)
    weights = weights / np.sum(weights)
    outside = np.maximum(np.asarray(signed, dtype=np.float64) + margin_m, 0.0)
    outside_fraction = float(np.sum(weights[np.asarray(signed) > 0.0]))
    return float(
        1.0e6 * np.sum(weights * outside * outside)
        + 1500.0 * float(np.quantile(outside, 0.95))
        + 20.0 * outside_fraction
    )


def _fit_terminal_compound_v2(
    *,
    label: str,
    asset: Any,
    source_bind_global: np.ndarray,
    source_bind_local: np.ndarray,
    base_target_bind_global: np.ndarray,
    base_target_bind_local: np.ndarray,
    bone_parents: np.ndarray,
    root_controller: int,
    rest_vertices: np.ndarray,
    vertex_ids: np.ndarray,
    area_weights: np.ndarray,
    pose_bundle: Mapping[str, np.ndarray],
    skins: Mapping[str, tuple[np.ndarray, np.ndarray]],
    maximum_linearization_steps: int = 3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one terminal-root parent-local bind against all posed male skins."""

    translation_bound, rotation_bound = _TERMINAL_BOUNDS[label]
    source = np.asarray(rest_vertices, dtype=np.float64)[vertex_ids]
    weights = np.asarray(area_weights, dtype=np.float64)[vertex_ids]
    weights = weights / np.sum(weights)
    parents = np.asarray(bone_parents, dtype=np.int64)
    source_global = np.asarray(source_bind_global, dtype=np.float64)
    source_local = np.asarray(source_bind_local, dtype=np.float64)
    base_target_global = np.asarray(base_target_bind_global, dtype=np.float64)
    base_target_local = np.asarray(base_target_bind_local, dtype=np.float64)
    inverse_source_global = np.linalg.inv(source_global)
    driver_indices = np.asarray(asset.driver_indices, dtype=np.int64)[vertex_ids]
    driver_weights = np.asarray(asset.driver_weights, dtype=np.float64)[vertex_ids]
    local_basis: dict[str, np.ndarray] = {}
    for pose_label, pose in pose_bundle.items():
        posed_source = source_bone_posed_global(asset, pose)
        posed_local = _global_to_local(posed_source, parents)
        local_basis[pose_label] = np.linalg.inv(source_local) @ posed_local
    parameters = np.zeros(6, dtype=np.float64)
    history: list[dict[str, Any]] = []

    def candidate(values: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        local_delta = np.eye(4, dtype=np.float64)
        local_delta[:3, :3] = Rotation.from_rotvec(values[3:]).as_matrix()
        local_delta[:3, 3] = values[:3]
        target_local = base_target_local.copy()
        target_local[root_controller] = base_target_local[root_controller] @ local_delta
        target_global = _fk(target_local, parents)
        correction = target_global @ inverse_source_global
        moved = _transform_points(source, correction[root_controller])
        inverse_target = np.linalg.inv(target_global)
        posed: dict[str, np.ndarray] = {}
        for pose_label in pose_bundle:
            target_pose_local = target_local @ local_basis[pose_label]
            target_pose_global = _fk(target_pose_local, parents)
            transforms = target_pose_global @ inverse_target
            selected = transforms[driver_indices]
            affines = np.einsum("vk,vkab->vab", driver_weights, selected)
            posed[pose_label] = _apply_affines(moved, affines)
        return correction[root_controller], posed

    def exact_state(values: np.ndarray) -> tuple[float, list[dict[str, Any]]]:
        _correction, posed_by_label = candidate(values)
        cells: list[dict[str, Any]] = []
        scores = []
        for pose_label, (skin, faces) in skins.items():
            posed = posed_by_label[pose_label]
            signed, closest, gradient = _signed_distance_details(posed, skin, faces)
            summary = _summary(signed, weights)
            score = _exact_terminal_score(signed, weights)
            scores.append(score)
            cells.append(
                {
                    "label": pose_label,
                    "posed": posed,
                    "signed": signed,
                    "closest": closest,
                    "gradient": gradient,
                    "summary": summary,
                    "score": score,
                }
            )
        regularization = float(
            1500.0 * np.dot(values[:3], values[:3])
            + 15.0 * np.dot(values[3:], values[3:])
        )
        return float(np.mean(scores) + regularization), cells

    current_score, cells = exact_state(parameters)
    initial_score = current_score
    initial_cells = [
        {"label": cell["label"], "summary": cell["summary"], "score": cell["score"]}
        for cell in cells
    ]
    for step in range(int(maximum_linearization_steps)):
        reference = parameters.copy()
        frozen = []
        for cell in cells:
            active = np.asarray(cell["signed"]) > -0.004
            if np.count_nonzero(active) < 64:
                order = np.argsort(np.asarray(cell["signed"]))[-64:]
                active = np.zeros(len(source), dtype=bool)
                active[order] = True
            frozen.append(
                {
                    "label": str(cell["label"]),
                    "active": active,
                    "reference_posed": np.asarray(cell["posed"])[active],
                    "signed": np.asarray(cell["signed"])[active],
                    "gradient": np.asarray(cell["gradient"])[active],
                    "sqrt_weight": np.sqrt(weights[active] / np.sum(weights[active])),
                }
            )

        def residual(values: np.ndarray) -> np.ndarray:
            _correction, posed_by_label = candidate(values)
            chunks = []
            for cell in frozen:
                posed = posed_by_label[cell["label"]][cell["active"]]
                predicted = cell["signed"] + np.einsum(
                    "vi,vi->v", cell["gradient"], posed - cell["reference_posed"]
                )
                chunks.append(
                    1000.0
                    * cell["sqrt_weight"]
                    * np.maximum(predicted + 0.001, 0.0)
                )
            chunks.append(0.20 * values[:3] / 0.015)
            chunks.append(0.20 * values[3:] / np.deg2rad(5.0))
            return np.concatenate(chunks)

        solved = least_squares(
            residual,
            reference,
            bounds=(
                np.asarray(
                    [-translation_bound] * 3 + [-rotation_bound] * 3,
                    dtype=np.float64,
                ),
                np.asarray(
                    [translation_bound] * 3 + [rotation_bound] * 3,
                    dtype=np.float64,
                ),
            ),
            max_nfev=60,
            xtol=1.0e-8,
            ftol=1.0e-8,
            gtol=1.0e-8,
        )
        proposal = np.asarray(solved.x, dtype=np.float64)
        proposal_score, proposal_cells = exact_state(proposal)
        accepted = proposal_score < current_score - 1.0e-8
        if not accepted:
            for fraction in (0.5, 0.25, 0.125):
                trial = reference + fraction * (proposal - reference)
                trial_score, trial_cells = exact_state(trial)
                if trial_score < current_score - 1.0e-8:
                    proposal, proposal_score, proposal_cells = (
                        trial,
                        trial_score,
                        trial_cells,
                    )
                    accepted = True
                    break
        history.append(
            {
                "step": step,
                "accepted": bool(accepted),
                "score_before": float(current_score),
                "score_after": float(proposal_score),
                "least_squares_nfev": int(solved.nfev),
            }
        )
        if not accepted:
            break
        parameters, current_score, cells = proposal, proposal_score, proposal_cells

    transform, _posed = candidate(parameters)
    final_cells = [
        {"label": cell["label"], "summary": cell["summary"], "score": cell["score"]}
        for cell in cells
    ]
    return transform, {
        "method": "multi_pose_frozen_sdf_terminal_root_local_bind_se3_v2",
        "initial_score": float(initial_score),
        "final_score": float(current_score),
        "improved": bool(current_score < initial_score - 1.0e-8),
        "local_bind_translation_m": parameters[:3].tolist(),
        "local_bind_rotation_vector_rad": parameters[3:].tolist(),
        "rest_correction_translation_m": transform[:3, 3].tolist(),
        "rotation_deg": float(np.degrees(np.linalg.norm(parameters[3:]))),
        "det_rotation": float(np.linalg.det(transform[:3, :3])),
        "scale": 1.0,
        "initial_cells": initial_cells,
        "final_cells": final_cells,
        "linearization_history": history,
    }


@dataclass(frozen=True)
class DynamicMainChainSubjectV2(ChainRestFitSubjectV1):
    rest_transport_bone: np.ndarray | None = None
    terminal_labels: np.ndarray | None = None
    terminal_rest_corrections: np.ndarray | None = None
    validation_pose_labels: np.ndarray | None = None
    validation_pose_axis_angle: np.ndarray | None = None

    def validate(self) -> None:
        super().validate()
        transport = np.asarray(self.rest_transport_bone, dtype=np.float64)
        terminal = np.asarray(self.terminal_rest_corrections, dtype=np.float64)
        poses = np.asarray(self.validation_pose_axis_angle, dtype=np.float64)
        labels = [str(value) for value in np.asarray(self.terminal_labels).tolist()]
        pose_labels = [
            str(value) for value in np.asarray(self.validation_pose_labels).tolist()
        ]
        if transport.shape != (235, 4, 4) or not np.all(np.isfinite(transport)):
            raise ValueError("V2 rest transport must be finite [235,4,4]")
        if terminal.shape != (4, 4, 4) or labels != list(TERMINAL_LABELS):
            raise ValueError("V2 terminal correction order is not frozen")
        if poses.shape != (len(pose_labels), 55, 3) or pose_labels[0] != "tpose":
            raise ValueError("V2 validation pose bundle is incomplete")
        roots = self.build_report.get("terminal_bind_root_indices")
        allowed = self.build_report.get("changed_parent_local_bind_indices")
        if (
            not isinstance(roots, list)
            or len(roots) != 4
            or not isinstance(allowed, list)
        ):
            raise ValueError("V2 changed bind indices are not explicit")
        source_local = _global_to_local(self.B_prefit, self.bone_parents)
        target_local = np.asarray(self.target_local_bind, dtype=np.float64)
        unchanged = np.ones(235, dtype=bool)
        unchanged[np.asarray(allowed, dtype=np.int64)] = False
        if not np.allclose(
            target_local[unchanged], source_local[unchanged], atol=2.0e-7, rtol=0.0
        ):
            raise ValueError("V2 changed a non-terminal parent-local bind")
        for matrix in np.concatenate((self.C_bone, transport, terminal), axis=0):
            rotation = matrix[:3, :3]
            if not (
                np.allclose(rotation.T @ rotation, np.eye(3), atol=2.0e-6)
                and np.isclose(np.linalg.det(rotation), 1.0, atol=2.0e-6)
            ):
                raise ValueError("V2 rest transport contains non-rigid scale")


def build_dynamic_main_chain_retarget_v2(
    operator: Any,
    calibration: AnatomicalCalibrationV1,
    *,
    betas: Any,
    subject_label: str,
    capture_sha256: str,
    smplx_model: Mapping[str, np.ndarray],
    smplx_model_sha256: str,
    recorded_poses: Mapping[str, np.ndarray],
) -> DynamicMainChainSubjectV2:
    started = time.perf_counter()
    if str(smplx_model_sha256) != FROZEN_SMPLX_MALE_SHA256:
        raise ValueError("dynamic main-chain V2 requires the authenticated male model")
    pose_bundle = {
        "tpose": np.zeros((55, 3), dtype=np.float64),
        **{
            str(label): np.asarray(pose, dtype=np.float64).reshape(55, 3)
            for label, pose in recorded_poses.items()
        },
    }
    if set(pose_bundle) != {"tpose", "pose_213328", "pose_213712"}:
        raise ValueError("V2 requires T-pose and both frozen recorded poses")
    legacy = build_whole_chain_rest_fit_v1(
        operator,
        calibration,
        betas=betas,
        subject_label=subject_label,
        capture_sha256=capture_sha256,
        smplx_model=smplx_model,
        smplx_model_sha256=smplx_model_sha256,
    )
    from .v8_artifacts import materialize_subject

    asset = materialize_subject(operator, betas=betas, gender="male").rigged_asset
    bind = np.asarray(legacy.B_prefit, dtype=np.float64)
    source_local = _global_to_local(bind, legacy.bone_parents)
    regions = dynamic_main_chain_regions_v2(asset)
    areas = _vertex_areas(legacy.vertices_prefit, legacy.faces)
    rest = np.asarray(legacy.vertices_final, dtype=np.float64).copy()
    terminal_transforms: list[np.ndarray] = []
    terminal_reports: dict[str, Any] = {}
    names = list(asset.source_bone_names or ())
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    base_target_global = bind
    base_target_local = source_local
    target_local = base_target_local.copy()
    skins = {
        label: smplx_body_surface_v7(
            smplx_model,
            betas=betas,
            pose_axis_angle=pose,
        )
        for label, pose in pose_bundle.items()
    }

    for label in TERMINAL_LABELS:
        side = label.split("_", 1)[0]
        if label.endswith("_foot"):
            ids = np.union1d(
                regions[f"{side}_foot_major"],
                regions[f"{side}_toe_phalanges"],
            )
            fit_ids = regions[f"{side}_foot_major"]
        else:
            ids = regions[label]
            fit_ids = _terminal_fit_vertex_ids(
                asset, label=label, terminal_vertex_ids=ids
            )
        root = names.index(_TERMINAL_ROOTS[label])
        transform, report = _fit_terminal_compound_v2(
            label=label,
            asset=asset,
            source_bind_global=bind,
            source_bind_local=source_local,
            base_target_bind_global=base_target_global,
            base_target_bind_local=base_target_local,
            bone_parents=parents,
            root_controller=root,
            rest_vertices=rest,
            vertex_ids=fit_ids,
            area_weights=areas,
            pose_bundle=pose_bundle,
            skins=skins,
        )
        local_delta = np.eye(4, dtype=np.float64)
        local_delta[:3, :3] = Rotation.from_rotvec(
            report["local_bind_rotation_vector_rad"]
        ).as_matrix()
        local_delta[:3, 3] = report["local_bind_translation_m"]
        target_local[root] = base_target_local[root] @ local_delta
        terminal_transforms.append(transform)
        terminal_reports[label] = report
        terminal_reports[label]["rigid_compound_vertex_count"] = int(len(ids))
        terminal_reports[label]["fit_vertex_count"] = int(len(fit_ids))
        terminal_reports[label]["excluded_small_bone_policy"] = (
            "toe_phalanges_report_only_same_rigid_transform"
            if label.endswith("_foot")
            else "none"
        )

    B_final = _fk(target_local, parents)
    C_bone = B_final @ np.linalg.inv(bind)
    rest_transport = np.asarray(legacy.C_bone, dtype=np.float64).copy()
    for label in TERMINAL_LABELS:
        side = label.split("_", 1)[0]
        ids = (
            np.union1d(regions[f"{side}_foot_major"], regions[f"{side}_toe_phalanges"])
            if label.endswith("_foot")
            else regions[label]
        )
        root = names.index(_TERMINAL_ROOTS[label])
        rest[ids] = _transform_points(rest[ids], C_bone[root])
        descendants = _descendants(parents, root)
        rest_transport[descendants] = C_bone[descendants] @ rest_transport[descendants]
    terminal_transforms = [
        C_bone[names.index(_TERMINAL_ROOTS[label])] for label in TERMINAL_LABELS
    ]

    tube_ids = _tube_vertex_ids(asset)
    if len(tube_ids):
        transported = _weighted_rest_correction(
            legacy.vertices_prefit,
            asset.driver_indices,
            asset.driver_weights,
            rest_transport,
        )
        rest[tube_ids] = transported[tube_ids]
    changed = np.flatnonzero(
        np.any(
            np.asarray(rest, dtype=np.float32)
            != np.asarray(legacy.vertices_prefit, dtype=np.float32),
            axis=1,
        )
    ).astype(np.int32)
    moved = np.unique(
        np.concatenate(
            (
                changed,
                np.asarray(legacy.pelvis_cage_vertex_ids, dtype=np.int32),
            )
        )
    ).astype(np.int32)
    final_frames, _widths, _details = _measure_frames(
        rest,
        calibration.domains,
        calibration.joint_domain_bases,
        partition="fit",
    )
    joint_names = [spec.name for spec in JOINT_SPECS]
    mesh_policy = np.asarray(legacy.mesh_policy).copy()
    mesh_controllers = np.asarray(asset.source_mesh_controller_bones, dtype=np.int64)
    for label in TERMINAL_LABELS:
        descendants = _descendants(parents, names.index(_TERMINAL_ROOTS[label]))
        mesh_policy[np.isin(mesh_controllers, descendants)] = f"rigid_{label}_v2"
    result = DynamicMainChainSubjectV2(
        **{
            **legacy.__dict__,
            "vertices_final": rest.astype(np.float32),
            "B_final": B_final,
            "C_bone": C_bone,
            "target_local_bind": target_local,
            "inverse_bind": np.linalg.inv(B_final),
            "final_anatomical_frames": final_frames[
                [index for index, _name in enumerate(joint_names)]
            ],
            "mesh_policy": mesh_policy,
            "moved_vertex_ids": moved,
            "build_report": {
                "schema_version": DYNAMIC_MAIN_CHAIN_RETARGET_SCHEMA_VERSION,
                "artifact_kind": DYNAMIC_MAIN_CHAIN_RETARGET_KIND,
                "method": "frozen_142_main_bind_multi_pose_terminal_root_rebind_v2",
                "baseline_commit": BASELINE_COMMIT,
                "smplx_gender": "male",
                "smplx_model_sha256": str(smplx_model_sha256),
                "motion_authority": "frozen_142_parent_local_fk",
                "bind_correction": "four_bounded_terminal_root_local_se3_only",
                "terminal_bind_root_indices": [
                    names.index(_TERMINAL_ROOTS[label]) for label in TERMINAL_LABELS
                ],
                "changed_parent_local_bind_indices": sorted(
                    {
                        names.index(_TERMINAL_ROOTS[label])
                        for label in TERMINAL_LABELS
                    }
                ),
                "main_chain_bind_policy": "frozen_142_parent_local_bind_v2_3",
                "main_chain_geometry_source": "whole_chain_rest_fit_v1",
                "terminal_compounds": terminal_reports,
                "radial_scale": 1.0,
                "uniform_scale": 1.0,
                "tube_transport_application_count": 1,
                "tube_transport_vertex_count": int(len(tube_ids)),
                "driver_indices_or_weights_changed": False,
                "bone_hierarchy_changed": False,
                "publishable": False,
                "elapsed_seconds": float(time.perf_counter() - started),
            },
            "rest_transport_bone": rest_transport,
            "terminal_labels": np.asarray(TERMINAL_LABELS),
            "terminal_rest_corrections": np.stack(terminal_transforms),
            "validation_pose_labels": np.asarray(list(pose_bundle)),
            "validation_pose_axis_angle": np.stack(list(pose_bundle.values())),
        }
    )
    result.validate()
    return result


def dynamic_main_chain_subject_v2_digest(value: DynamicMainChainSubjectV2) -> str:
    value.validate()
    digest = hashlib.sha256(b"dynamic-main-chain-subject-v2\0")
    for name, field in value.__dict__.items():
        digest.update(name.encode("ascii"))
        if isinstance(field, str):
            digest.update(field.encode("utf-8"))
        elif isinstance(field, Mapping):
            digest.update(
                json.dumps(field, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
        elif field is None:
            digest.update(b"none")
        else:
            digest.update(_array_digest(field).encode("ascii"))
    return digest.hexdigest()


def save_dynamic_main_chain_subject_v2(
    path: Path | str,
    value: DynamicMainChainSubjectV2,
    *,
    provenance: Mapping[str, Any],
) -> Path:
    value.validate()
    output = Path(path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite V2 subject: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        npz = temporary / "dynamic_main_chain_subject_v2.npz"
        arrays = {
            name: field
            for name, field in value.__dict__.items()
            if isinstance(field, np.ndarray)
        }
        np.savez_compressed(npz, **arrays)
        manifest = {
            "schema_version": DYNAMIC_MAIN_CHAIN_RETARGET_SCHEMA_VERSION,
            "artifact_kind": DYNAMIC_MAIN_CHAIN_RETARGET_KIND,
            "baseline_commit": BASELINE_COMMIT,
            "subject_label": value.subject_label,
            "subject_content_digest": dynamic_main_chain_subject_v2_digest(value),
            "npz": npz.name,
            "npz_sha256": _sha256(npz),
            "build_report": value.build_report,
            "provenance": dict(provenance),
            "accepted_scope": "full_main_chain_shadow_v2",
            "smplx_gender": "male",
            "smplx_model_sha256": value.smplx_model_sha256,
            "publishable": False,
            "trusted_latest_updated": False,
            "vessel_repair_started": False,
            "complete": True,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


__all__ = [
    "DYNAMIC_MAIN_CHAIN_RETARGET_KIND",
    "DYNAMIC_MAIN_CHAIN_RETARGET_SCHEMA_VERSION",
    "DynamicMainChainSubjectV2",
    "build_dynamic_main_chain_retarget_v2",
    "dynamic_main_chain_subject_v2_digest",
    "save_dynamic_main_chain_subject_v2",
]
