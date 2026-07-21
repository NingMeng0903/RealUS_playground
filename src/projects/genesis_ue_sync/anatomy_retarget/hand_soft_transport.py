"""Offline smooth hand-end transport from fitted rigid anatomy."""

from __future__ import annotations

from typing import Any

import numpy as np

from .rigged_asset import AnatomyRiggedAsset


_HAND_BONE_TOKENS = (
    "metacarpal",
    "phalanx_hand",
    "phalanges_hand",
    "capitate",
    "hamate",
    "lunate",
    "pisiform",
    "scaphoid",
    "trapezium",
    "trapezoid",
    "triquetrum",
)


def transport_hand_soft_rbf(
    asset: AnatomyRiggedAsset,
    *,
    tissues: tuple[str, ...] = ("nerve",),
    controls_per_mesh: int = 64,
    radius_ratio: float = 0.535,
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Transport hand-end soft anatomy through the fitted hand bone field."""
    if asset.harmonic_reference_vertices is None:
        return asset, {"available": False, "reason": "harmonic_reference_missing"}
    from scipy.interpolate import RBFInterpolator
    from scipy.spatial import cKDTree

    reference = np.asarray(asset.harmonic_reference_vertices, dtype=np.float64)
    final = np.asarray(asset.vertices_rest, dtype=np.float64)
    output = final.copy()
    selected_tissues = {str(value).lower() for value in tissues}
    joint_id = {str(name): index for index, name in enumerate(asset.joint_names)}
    sides: dict[str, Any] = {}
    moved = np.zeros(len(output), dtype=bool)
    for side, suffix in (("left", "_l"), ("right", "_r")):
        source_controls: list[np.ndarray] = []
        target_controls: list[np.ndarray] = []
        control_meshes: list[str] = []
        for (start, stop), name, tissue in zip(
            asset.source_vertex_ranges, asset.source_mesh_names, asset.source_tissues
        ):
            lower = str(name).lower()
            if (
                str(tissue).lower() != "bone"
                or not lower.endswith(suffix)
                or not any(token in lower for token in _HAND_BONE_TOKENS)
            ):
                continue
            start_i, stop_i = int(start), int(stop)
            count = min(int(controls_per_mesh), stop_i - start_i)
            sample = np.linspace(start_i, stop_i - 1, count, dtype=np.int64)
            source_controls.append(reference[sample])
            target_controls.append(final[sample])
            control_meshes.append(str(name))
        if not source_controls:
            continue
        source_control = np.concatenate(source_controls, axis=0)
        target_control = np.concatenate(target_controls, axis=0)
        displacement = target_control - source_control
        wrist = np.asarray(asset.rest_joints[joint_id[f"{side}_wrist"]], dtype=np.float64)
        middle3 = np.asarray(asset.rest_joints[joint_id[f"{side}_middle3"]], dtype=np.float64)
        radius = float(radius_ratio) * float(np.linalg.norm(middle3 - wrist))
        candidates = np.zeros(len(output), dtype=bool)
        for (start, stop), tissue in zip(
            asset.source_vertex_ranges, asset.source_tissues
        ):
            if str(tissue).lower() in selected_tissues:
                candidates[int(start) : int(stop)] = True
        candidate_rows = np.flatnonzero(candidates)
        distance, _nearest = cKDTree(source_control).query(
            reference[candidate_rows], k=1, workers=-1
        )
        active_rows = candidate_rows[np.asarray(distance) <= radius]
        model = RBFInterpolator(
            source_control,
            displacement,
            kernel="thin_plate_spline",
            smoothing=1.0e-8,
            neighbors=min(96, len(source_control)),
        )
        predicted = np.asarray(model(reference[active_rows]), dtype=np.float64)
        output[active_rows] = reference[active_rows] + predicted
        moved[active_rows] = True
        control_error = np.linalg.norm(
            np.asarray(model(source_control)) - displacement, axis=1
        )
        sides[side] = {
            "control_mesh_count": int(len(control_meshes)),
            "control_count": int(len(source_control)),
            "active_vertices": int(len(active_rows)),
            "radius_m": radius,
            "control_rms_m": float(np.sqrt(np.mean(control_error**2))),
            "control_max_m": float(np.max(control_error)),
            "mapped_displacement_max_m": (
                float(np.max(np.linalg.norm(predicted, axis=1)))
                if len(predicted)
                else 0.0
            ),
        }
    result = type(asset)(
        **{**asset.__dict__, "vertices_rest": output.astype(np.float32)}
    )
    result.validate()
    return result, {
        "available": True,
        "backend": "bilateral_local_thin_plate_rbf_from_harmonic_hand_bones",
        "tissues": sorted(selected_tissues),
        "controls_per_mesh": int(controls_per_mesh),
        "radius_ratio": float(radius_ratio),
        "moved_vertex_count": int(np.count_nonzero(moved)),
        "sides": sides,
        "source_weights_preserved": True,
        "source_hierarchy_preserved": True,
    }
