"""Posed vessel/nerve linkage invariants.

Tube linkage is only checked at rest today.  ``check_whole_chain_rest_fit_v1``
proves ``vertices_final[tube]`` reproduces the same frozen LBS transport and
that the topology digests hold, and ``blender_link_oracle_v7`` freezes the
authored digests -- but both are rest-space.  ``vessel_gates_v7`` does have
posed subgates and is not wired into any V7/V10/V11 shadow CLI.

Nothing anywhere checks the invariant the whole design rests on: that a
vessel keeps its position *relative to the bone it runs along* once the body
is posed.  If tubes were ever driven by anything other than the same frozen
14-slot LBS as the bones, that offset would drift.  This module measures it.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Mapping, Sequence

import numpy as np


# The authored contract, frozen in MD/todo_ana.md section 2.4.
EXPECTED_TUBE_MESH_COUNT = 17
EXPECTED_TUBE_VERTEX_COUNT = 55337

# Judged against the frozen 142 transport, not against zero.  Sharing a
# dominant controller does not mean sharing a weight vector, so blended tube
# vertices legitimately change their bone distance when a joint moves -- the
# raw 142 baseline itself drifts 22 mm at pose_213328.  What must not happen
# is drifting *more* than the authored rig does.
TUBE_BONE_OFFSET_REGRESSION_LIMIT_M = 0.001

_TUBE_TISSUES = ("vessel", "nerve")


def _tissue_ids(asset: Any, tissues: Sequence[str]) -> np.ndarray:
    wanted = {str(name).strip().lower() for name in tissues}
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    chunks = [
        np.arange(int(start), int(stop), dtype=np.int64)
        for label, (start, stop) in zip(asset.source_tissues, ranges.tolist())
        if str(label).strip().lower() in wanted
    ]
    if not chunks:
        raise ValueError(f"asset exposes no {sorted(wanted)} meshes")
    return np.concatenate(chunks)


def _tube_mesh_count(asset: Any) -> int:
    wanted = {name.lower() for name in _TUBE_TISSUES}
    return sum(
        1 for label in asset.source_tissues if str(label).strip().lower() in wanted
    )


def tube_topology_digest_v12(asset: Any) -> str:
    """Digest over the tube faces, so a reroute cannot pass unnoticed."""

    tube_ids = _tissue_ids(asset, _TUBE_TISSUES)
    member = np.zeros(int(np.max(np.asarray(asset.faces)) + 1), dtype=bool)
    member[tube_ids] = True
    faces = np.asarray(asset.faces, dtype=np.int64)
    mask = np.all(member[faces], axis=1)
    digest = hashlib.sha256(b"tube-topology-v12\0")
    digest.update(np.ascontiguousarray(faces[mask], dtype=np.int64).tobytes())
    return digest.hexdigest()


def _dominant_controller(asset: Any) -> np.ndarray:
    indices = np.asarray(asset.driver_indices, dtype=np.int64)
    weights = np.asarray(asset.driver_weights, dtype=np.float64)
    return indices[np.arange(len(indices)), np.argmax(weights, axis=1)]


def tube_bone_offset_metrics_v12(
    rest_vertices: np.ndarray,
    posed_vertices: np.ndarray,
    *,
    asset: Any,
) -> dict[str, Any]:
    """How far each tube vertex drifts from a frozen, co-driven bone partner.

    The partner has to share the tube vertex's dominant LBS controller.  A
    plain nearest-bone pairing does not work: a vessel crossing the knee has
    its nearest bone vertex on the other side of the joint, so flexing the
    knee legitimately changes that distance -- the raw 142 baseline drifts
    52.9 mm that way while its median stays at zero.  Co-driven pairs undergo
    exactly the same rigid transform, so any drift really is a second
    transport rather than articulation.
    """

    from scipy.spatial import cKDTree

    rest = np.asarray(rest_vertices, dtype=np.float64)
    posed = np.asarray(posed_vertices, dtype=np.float64)
    if rest.shape != posed.shape:
        raise ValueError("rest and posed vertex arrays disagree")
    tube_ids = _tissue_ids(asset, _TUBE_TISSUES)
    bone_ids = _tissue_ids(asset, ("bone",))
    controller = _dominant_controller(asset)

    bone_by_controller: dict[int, np.ndarray] = {}
    for value in np.unique(controller[bone_ids]):
        bone_by_controller[int(value)] = bone_ids[controller[bone_ids] == value]

    paired: list[np.ndarray] = []
    partners: list[np.ndarray] = []
    unpaired = 0
    for value in np.unique(controller[tube_ids]):
        members = tube_ids[controller[tube_ids] == value]
        bones = bone_by_controller.get(int(value))
        if bones is None or not len(bones):
            unpaired += int(len(members))
            continue
        _distance, neighbour = cKDTree(rest[bones]).query(rest[members], k=1)
        paired.append(members)
        partners.append(bones[np.asarray(neighbour, dtype=np.int64)])

    if not paired:
        raise ValueError("no tube vertex shares a controller with any bone vertex")
    members = np.concatenate(paired)
    partner = np.concatenate(partners)
    rest_distance = np.linalg.norm(rest[members] - rest[partner], axis=1)
    posed_distance = np.linalg.norm(posed[members] - posed[partner], axis=1)
    drift = np.abs(posed_distance - rest_distance)

    return {
        "tube_vertex_count": int(len(tube_ids)),
        "tube_mesh_count": int(_tube_mesh_count(asset)),
        "topology_digest": tube_topology_digest_v12(asset),
        "paired_vertex_count": int(len(members)),
        "unpaired_vertex_count": int(unpaired),
        "offset_drift_max_m": float(np.max(drift)),
        "offset_drift_p95_m": float(np.quantile(drift, 0.95)),
        "offset_drift_median_m": float(np.median(drift)),
        "rest_offset_median_m": float(np.median(rest_distance)),
    }


def evaluate_linkage_v12(
    metrics_by_pose: Mapping[str, Mapping[str, Any]],
    *,
    reference: Mapping[str, Mapping[str, Any]],
    reference_topology_digest: str | None = None,
    drift_regression_limit_m: float = TUBE_BONE_OFFSET_REGRESSION_LIMIT_M,
    expected_tube_mesh_count: int = EXPECTED_TUBE_MESH_COUNT,
    expected_tube_vertex_count: int = EXPECTED_TUBE_VERTEX_COUNT,
) -> dict[str, Any]:
    """Hard-gate the posed linkage invariants across every pose.

    Counts and the topology digest are absolute; the tube-to-bone offset is
    judged against ``reference`` (the frozen 142 transport), because that is
    the linkage the Blender rig actually authored.
    """

    started = time.perf_counter()
    cells: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    digests = {
        pose: str(metrics["topology_digest"])
        for pose, metrics in metrics_by_pose.items()
    }
    expected_digest = reference_topology_digest or next(iter(digests.values()), None)

    for pose_name, metrics in metrics_by_pose.items():
        if pose_name not in reference:
            raise KeyError(f"missing reference linkage metrics for pose {pose_name}")
        baseline = reference[pose_name]
        pose_failures: list[dict[str, Any]] = []
        if int(metrics["tube_mesh_count"]) != int(expected_tube_mesh_count):
            pose_failures.append(
                {
                    "reason": "tube_mesh_count_changed",
                    "pose": pose_name,
                    "tube_mesh_count": int(metrics["tube_mesh_count"]),
                    "expected": int(expected_tube_mesh_count),
                }
            )
        if int(metrics["tube_vertex_count"]) != int(expected_tube_vertex_count):
            pose_failures.append(
                {
                    "reason": "tube_vertex_count_changed",
                    "pose": pose_name,
                    "tube_vertex_count": int(metrics["tube_vertex_count"]),
                    "expected": int(expected_tube_vertex_count),
                }
            )
        if expected_digest is not None and digests[pose_name] != expected_digest:
            pose_failures.append(
                {
                    "reason": "tube_topology_digest_changed",
                    "pose": pose_name,
                    "digest": digests[pose_name],
                    "expected": expected_digest,
                }
            )
        for field in ("offset_drift_max_m", "offset_drift_p95_m"):
            regression = float(metrics[field] - baseline[field])
            if regression > drift_regression_limit_m:
                pose_failures.append(
                    {
                        "reason": "tube_bone_offset_regressed",
                        "pose": pose_name,
                        "field": field,
                        "value_m": float(metrics[field]),
                        "reference_m": float(baseline[field]),
                        "regression_m": regression,
                        "limit_m": float(drift_regression_limit_m),
                    }
                )
        cells[pose_name] = {
            "passed": len(pose_failures) == 0,
            "metrics": dict(metrics),
            "failures": pose_failures,
        }
        failures.extend(pose_failures)
    return {
        "schema_version": 12,
        "artifact_kind": "PosedLinkageV12",
        "passed": len(failures) == 0,
        "publishable": False,
        "invariant": "tubes_follow_bones_through_the_same_frozen_lbs",
        "gates": {
            "tube_bone_offset_regression_limit_m": float(drift_regression_limit_m),
            "expected_tube_mesh_count": int(expected_tube_mesh_count),
            "expected_tube_vertex_count": int(expected_tube_vertex_count),
            "reference_topology_digest": expected_digest,
        },
        "cells": cells,
        "failures": failures,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


__all__ = [
    "EXPECTED_TUBE_MESH_COUNT",
    "EXPECTED_TUBE_VERTEX_COUNT",
    "TUBE_BONE_OFFSET_REGRESSION_LIMIT_M",
    "evaluate_linkage_v12",
    "tube_bone_offset_metrics_v12",
    "tube_topology_digest_v12",
]
