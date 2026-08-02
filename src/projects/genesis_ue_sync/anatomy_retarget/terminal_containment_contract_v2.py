"""Frozen anatomical regions for dynamic terminal containment checks.

SMPL-X has no independent toe-volume authority.  The contract therefore gates
the complete rigid hands and the SMPL-X-expressible foot volume, while keeping
toe phalanges as rigid-integrity and visual-review evidence.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


TERMINAL_CONTAINMENT_CONTRACT_SCHEMA_VERSION = 2
TERMINAL_CONTAINMENT_CONTRACT_KIND = "TerminalContainmentContractV2"

_ROOTS = {
    "left_lower": "Femur_Rot_L",
    "right_lower": "Femur_Rot_R",
    "left_upper": "Shoulder_Rotate_L",
    "right_upper": "Shoulder_Rotate_R",
    "left_foot": "Ankle_Rot_L",
    "right_foot": "Ankle_Rot_R",
    "left_hand": "Wrist_Rotate_L",
    "right_hand": "Wrist_Rotate_R1",
}

_MAJOR_FOOT_TOKENS = (
    "talus",
    "calcaneus",
    "navicular",
    "cuboid",
    "cuneiform",
    "metatarsal",
)

_REGION_THRESHOLDS = {
    "lower_core": {
        "gate_type": "bounded_regression",
        "inside_fraction_min": 0.750,
        "inside_fraction_delta_min": -0.005,
        "max_outside_m": 0.040,
        "max_outside_regression_m": 0.002,
        "comparison_tolerance": 0.0,
    },
    "upper_core": {
        "gate_type": "absolute",
        "inside_fraction_min": 0.950,
        "max_outside_m": 0.020,
        "comparison_tolerance": 0.0,
    },
    "hand": {
        "gate_type": "absolute",
        "inside_fraction_min": 0.980,
        "max_outside_m": 0.006,
        "comparison_tolerance": 0.001,
    },
    "foot_major": {
        "gate_type": "absolute",
        "inside_fraction_min": 0.900,
        "max_outside_m": 0.015,
        "per_mesh_inside_fraction_min": 0.600,
        "comparison_tolerance": 0.0,
    },
    "toe_phalanges": {
        "gate_type": "report_only_rigid_integrity_and_genesis",
        "inside_fraction_min": None,
        "max_outside_m": None,
        "comparison_tolerance": 0.0,
    },
}


def _region_kind(label: str) -> str:
    for kind in _REGION_THRESHOLDS:
        if label == kind or label.endswith(f"_{kind}"):
            return kind
    raise KeyError(f"unknown terminal containment region: {label}")


def _descendants(parents: np.ndarray, root: int) -> set[int]:
    selected = {int(root)}
    changed = True
    while changed:
        changed = False
        for index, parent in enumerate(np.asarray(parents, dtype=np.int64).tolist()):
            if index not in selected and int(parent) in selected:
                selected.add(index)
                changed = True
    return selected


def _bone_vertices_for_controllers(asset: Any, controllers: set[int]) -> np.ndarray:
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    mesh_controllers = np.asarray(asset.source_mesh_controller_bones, dtype=np.int64)
    tissues = np.char.lower(np.char.strip(np.asarray(asset.source_tissues).astype(str)))
    selected = (tissues == "bone") & np.isin(
        mesh_controllers, np.asarray(sorted(controllers), dtype=np.int64)
    )
    chunks = [
        np.arange(int(start), int(stop), dtype=np.int64)
        for include, (start, stop) in zip(selected.tolist(), ranges.tolist())
        if bool(include)
    ]
    if not chunks:
        raise ValueError("terminal containment region contains no bone vertices")
    return np.unique(np.concatenate(chunks))


def _bone_mesh_vertices(
    asset: Any,
    *,
    side_suffix: str,
    include_tokens: tuple[str, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    chunks: list[np.ndarray] = []
    selected_names: list[str] = []
    for name, tissue, (start, stop) in zip(
        asset.source_mesh_names, asset.source_tissues, asset.source_vertex_ranges
    ):
        mesh_name = str(name)
        lower = mesh_name.strip().lower()
        if (
            str(tissue).strip().lower() == "bone"
            and mesh_name.endswith(side_suffix)
            and any(token in lower for token in include_tokens)
        ):
            chunks.append(np.arange(int(start), int(stop), dtype=np.int64))
            selected_names.append(mesh_name)
    if not chunks:
        raise ValueError(
            f"terminal containment found no {side_suffix} meshes for {include_tokens}"
        )
    return np.unique(np.concatenate(chunks)), tuple(selected_names)


def terminal_containment_regions_v2(asset: Any) -> dict[str, np.ndarray]:
    """Return disjoint core, rigid-hand, major-foot and toe-report regions."""

    names = list(asset.source_bone_names or ())
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    controllers = {
        label: _descendants(parents, names.index(root))
        for label, root in _ROOTS.items()
    }
    regions: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        lower = controllers[f"{side}_lower"] - controllers[f"{side}_foot"]
        upper = controllers[f"{side}_upper"] - controllers[f"{side}_hand"]
        regions[f"{side}_lower_core"] = _bone_vertices_for_controllers(asset, lower)
        regions[f"{side}_upper_core"] = _bone_vertices_for_controllers(asset, upper)
        regions[f"{side}_hand"] = _bone_vertices_for_controllers(
            asset, controllers[f"{side}_hand"]
        )
        suffix = "_L" if side == "left" else "_R"
        regions[f"{side}_foot_major"], _major_names = _bone_mesh_vertices(
            asset,
            side_suffix=suffix,
            include_tokens=_MAJOR_FOOT_TOKENS,
        )
        regions[f"{side}_toe_phalanges"], _toe_names = _bone_mesh_vertices(
            asset,
            side_suffix=suffix,
            include_tokens=("phalanx_foot",),
        )
        complete_foot = _bone_vertices_for_controllers(
            asset, controllers[f"{side}_foot"]
        )
        classified = np.union1d(
            regions[f"{side}_foot_major"], regions[f"{side}_toe_phalanges"]
        )
        if not np.array_equal(classified, complete_foot):
            missing = np.setdiff1d(complete_foot, classified)
            extra = np.setdiff1d(classified, complete_foot)
            raise ValueError(
                f"{side} foot contract is incomplete: missing={len(missing)} extra={len(extra)}"
            )

    for chain in ("lower_core", "upper_core", "hand", "foot_major", "toe_phalanges"):
        regions[chain] = np.union1d(
            regions[f"left_{chain}"], regions[f"right_{chain}"]
        )

    keys = list(regions)
    for index, first in enumerate(keys):
        if first in {"lower_core", "upper_core", "hand", "foot_major", "toe_phalanges"}:
            continue
        for second in keys[index + 1 :]:
            if second in {"lower_core", "upper_core", "hand", "foot_major", "toe_phalanges"}:
                continue
            if np.intersect1d(regions[first], regions[second]).size:
                raise ValueError(f"terminal regions overlap: {first} and {second}")
    return regions


def terminal_containment_foot_mesh_regions_v2(
    asset: Any, *, side: str
) -> dict[str, np.ndarray]:
    suffix = "_L" if side == "left" else "_R" if side == "right" else None
    if suffix is None:
        raise ValueError("side must be left or right")
    result: dict[str, np.ndarray] = {}
    for name, tissue, (start, stop) in zip(
        asset.source_mesh_names, asset.source_tissues, asset.source_vertex_ranges
    ):
        mesh_name = str(name)
        if (
            str(tissue).strip().lower() == "bone"
            and mesh_name.endswith(suffix)
            and any(token in mesh_name.lower() for token in _MAJOR_FOOT_TOKENS)
        ):
            result[mesh_name] = np.arange(int(start), int(stop), dtype=np.int64)
    if len(result) != 12:
        raise ValueError(f"expected 12 major foot meshes for {side}, got {len(result)}")
    return result


def terminal_containment_contract_v2(asset: Any) -> dict[str, Any]:
    """Describe and digest the exact region taxonomy shared by fit and checker."""

    regions = terminal_containment_regions_v2(asset)
    mesh_names = [str(name) for name in asset.source_mesh_names]
    mesh_ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    region_manifest: dict[str, Any] = {}
    for label, ids in regions.items():
        selected_meshes = [
            mesh_names[index]
            for index, (start, stop) in enumerate(mesh_ranges.tolist())
            if np.intersect1d(ids, np.arange(int(start), int(stop))).size
        ]
        region_manifest[label] = {
            "vertex_count": int(len(ids)),
            "vertex_ids_sha256": hashlib.sha256(
                np.ascontiguousarray(ids, dtype=np.int64).tobytes()
            ).hexdigest(),
            "mesh_names": selected_meshes,
        }
    gate_modes = {}
    baseline_roles = {}
    thresholds = {}
    for label in regions:
        kind = _region_kind(label)
        if label in {"lower_core", "upper_core", "hand", "foot_major", "toe_phalanges"}:
            gate_modes[label] = "report_only_bilateral_diagnostic"
            baseline_roles[label] = "report_only"
        else:
            gate_modes[label] = _REGION_THRESHOLDS[kind]["gate_type"]
            baseline_roles[label] = (
                "gate_reference" if kind == "lower_core" else "report_only"
            )
        thresholds[label] = dict(_REGION_THRESHOLDS[kind])
    payload = {
        "schema_version": TERMINAL_CONTAINMENT_CONTRACT_SCHEMA_VERSION,
        "artifact_kind": TERMINAL_CONTAINMENT_CONTRACT_KIND,
        "regions": region_manifest,
        "gate_modes": gate_modes,
        "thresholds": thresholds,
        "baseline_roles": baseline_roles,
        "baseline_142_is_report_only": False,
        "optimizer_checker_regions_are_identical": True,
        "toe_phalanges_follow_same_rigid_foot_transform": True,
        "publishable": False,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    payload["contract_digest"] = hashlib.sha256(canonical).hexdigest()
    return payload


__all__ = [
    "TERMINAL_CONTAINMENT_CONTRACT_KIND",
    "TERMINAL_CONTAINMENT_CONTRACT_SCHEMA_VERSION",
    "terminal_containment_contract_v2",
    "terminal_containment_foot_mesh_regions_v2",
    "terminal_containment_regions_v2",
]
