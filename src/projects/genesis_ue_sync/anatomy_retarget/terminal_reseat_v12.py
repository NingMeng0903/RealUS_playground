"""Bounded rigid re-seat of the hand/foot clusters in rest space.

Every version from V7 through V11 froze the terminals to copy-142, so their
containment numbers are identical and the feet are the top poke source in
every pose -- both independent visual reviews block on exactly that.

Measured on subject 213328 at rest, a pure rigid move of the left foot cluster
(6 mm translation, 11 degrees rotation, no scale at all) takes its worst
poke-through from 16.4 mm to 2.4 mm and the outside vertex count from 815 to
93.  The foot is not too large for the SMPL-X envelope; it is rotated about
11 degrees out of the foot pocket.

So this module solves that rigid transform and applies it to both the terminal
bind matrices and the terminal geometry, blended by LBS weight so the wrist and
ankle seams do not tear.  No scaling, no vertex surgery, no weight edits: the
authored bone stays exactly the shape the Blender rig authored.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np

from .chain_containment_v1 import _signed_distance
from .pose_map_v10 import FOOT_ROOTS, HAND_ROOTS
from .whole_chain_rest_fit_v1 import _descendants


# The measured optimum is 6 mm / 11 degrees; these leave headroom for other
# betas without letting the solver walk a terminal off its joint.
MAX_TRANSLATION_M = 0.015
MAX_ROTATION_DEG = 15.0

# Objective sample size.  The clusters have 5k-10k vertices and the signed
# distance query dominates the solve, so the fit runs on a deterministic
# subsample and the reported metrics are recomputed on every vertex.
OBJECTIVE_SAMPLES = 1200

TERMINAL_ROOTS = (*HAND_ROOTS, *FOOT_ROOTS)


def _cluster_vertex_ids(asset: Any, controllers: Sequence[int]) -> np.ndarray:
    wanted = set(int(value) for value in controllers)
    owners = np.asarray(asset.source_mesh_controller_bones, dtype=np.int64)
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    tissues = [str(label).strip().lower() for label in asset.source_tissues]
    chunks = [
        np.arange(int(start), int(stop), dtype=np.int64)
        for owner, (start, stop), tissue in zip(owners.tolist(), ranges.tolist(), tissues)
        if tissue == "bone" and int(owner) in wanted
    ]
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)


def _rigid(parameters: np.ndarray, centre: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    matrix = np.eye(4, dtype=np.float64)
    rotation = Rotation.from_rotvec(np.asarray(parameters[3:], dtype=np.float64)).as_matrix()
    matrix[:3, :3] = rotation
    matrix[:3, 3] = centre + np.asarray(parameters[:3], dtype=np.float64) - rotation @ centre
    return matrix


def _apply(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ np.asarray(transform[:3, :3]).T + np.asarray(transform[:3, 3])


def solve_terminal_reseat_v12(
    vertices: np.ndarray,
    *,
    asset: Any,
    skin: np.ndarray,
    skin_faces: np.ndarray,
    bone_parents: np.ndarray,
    max_translation_m: float = MAX_TRANSLATION_M,
    max_rotation_deg: float = MAX_ROTATION_DEG,
    samples: int = OBJECTIVE_SAMPLES,
) -> dict[str, Any]:
    """Fit one bounded rigid transform per terminal cluster at rest."""

    from scipy.optimize import minimize

    rest = np.asarray(vertices, dtype=np.float64)
    names = [str(name) for name in asset.source_bone_names]
    parents = np.asarray(bone_parents, dtype=np.int64)
    skin_points = np.asarray(skin, dtype=np.float64)
    skin_triangles = np.asarray(skin_faces)
    bound = float(np.deg2rad(max_rotation_deg))

    result: dict[str, Any] = {}
    for root_name in TERMINAL_ROOTS:
        root = names.index(root_name)
        controllers = sorted({root, *(int(i) for i in _descendants(parents, root))})
        ids = _cluster_vertex_ids(asset, controllers)
        if not len(ids):
            raise ValueError(f"terminal cluster {root_name} has no bone vertices")
        points = rest[ids]
        centre = points.mean(axis=0)
        generator = np.random.default_rng(abs(hash(root_name)) % (2**32))
        pick = (
            generator.choice(len(points), size=samples, replace=False)
            if len(points) > samples
            else np.arange(len(points))
        )
        sampled = points[pick]

        def cost(parameters: np.ndarray) -> float:
            moved = _apply(_rigid(parameters, centre), sampled)
            signed = _signed_distance(moved, skin_points, skin_triangles)
            return float(np.sum(np.maximum(signed, 0.0) ** 2))

        best = minimize(
            cost,
            np.zeros(6),
            method="Powell",
            bounds=[(-max_translation_m, max_translation_m)] * 3 + [(-bound, bound)] * 3,
            options={"maxiter": 300, "xtol": 1.0e-4, "ftol": 1.0e-7},
        )
        transform = _rigid(np.asarray(best.x, dtype=np.float64), centre)
        before = _signed_distance(points, skin_points, skin_triangles)
        after = _signed_distance(_apply(transform, points), skin_points, skin_triangles)
        result[root_name] = {
            "controllers": controllers,
            "vertex_ids": ids,
            "transform": transform,
            "translation_m": float(np.linalg.norm(best.x[:3])),
            "rotation_deg": float(np.degrees(np.linalg.norm(best.x[3:]))),
            "max_outside_before_m": float(max(0.0, float(np.max(before)))),
            "max_outside_after_m": float(max(0.0, float(np.max(after)))),
            "outside_count_before": int(np.count_nonzero(before > 0.0)),
            "outside_count_after": int(np.count_nonzero(after > 0.0)),
        }
    return result


def apply_terminal_reseat_v12(
    vertices: np.ndarray,
    bind: np.ndarray,
    *,
    asset: Any,
    reseat: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Move terminal geometry and terminal binds by the same rigid transform.

    Geometry is blended by the fraction of each vertex's LBS mass that sits on
    the cluster, so a vertex straddling the wrist moves partway rather than
    tearing away from the forearm.
    """

    moved = np.asarray(vertices, dtype=np.float64).copy()
    matrices = np.asarray(bind, dtype=np.float64).copy()
    driver_indices = np.asarray(asset.driver_indices, dtype=np.int64)
    driver_weights = np.asarray(asset.driver_weights, dtype=np.float64)
    report: dict[str, Any] = {}

    for root_name, entry in reseat.items():
        transform = np.asarray(entry["transform"], dtype=np.float64)
        controllers = list(entry["controllers"])
        member = np.isin(driver_indices, controllers)
        alpha = np.clip(np.sum(driver_weights * member, axis=1), 0.0, 1.0)
        active = np.flatnonzero(alpha > 0.0)
        if not len(active):
            raise ValueError(f"terminal cluster {root_name} drives no vertices")
        target = _apply(transform, moved[active])
        weights = alpha[active][:, None]
        shift = weights * (target - moved[active])
        moved[active] += shift
        for controller in controllers:
            matrices[controller] = transform @ matrices[controller]
        report[root_name] = {
            "translation_m": float(entry["translation_m"]),
            "rotation_deg": float(entry["rotation_deg"]),
            "max_outside_before_m": float(entry["max_outside_before_m"]),
            "max_outside_after_m": float(entry["max_outside_after_m"]),
            "outside_count_before": int(entry["outside_count_before"]),
            "outside_count_after": int(entry["outside_count_after"]),
            "blended_vertex_count": int(len(active)),
            "max_vertex_shift_m": float(np.max(np.linalg.norm(shift, axis=1))),
        }
    return moved, matrices, report


def reseat_terminals_v12(
    vertices: np.ndarray,
    bind: np.ndarray,
    *,
    asset: Any,
    skin: np.ndarray,
    skin_faces: np.ndarray,
    bone_parents: np.ndarray,
    max_translation_m: float = MAX_TRANSLATION_M,
    max_rotation_deg: float = MAX_ROTATION_DEG,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Solve and apply the terminal re-seat in one call."""

    started = time.perf_counter()
    reseat = solve_terminal_reseat_v12(
        vertices,
        asset=asset,
        skin=skin,
        skin_faces=skin_faces,
        bone_parents=bone_parents,
        max_translation_m=max_translation_m,
        max_rotation_deg=max_rotation_deg,
    )
    moved, matrices, report = apply_terminal_reseat_v12(
        vertices, bind, asset=asset, reseat=reseat
    )
    return (
        moved,
        matrices,
        {
            "method": "bounded_rigid_terminal_reseat_v12",
            "scaling_applied": False,
            "max_translation_m": float(max_translation_m),
            "max_rotation_deg": float(max_rotation_deg),
            "clusters": report,
            "elapsed_seconds": float(time.perf_counter() - started),
        },
    )


def reseat_subject_terminals_v12(
    value: Any,
    *,
    asset: Any,
    skin: np.ndarray,
    skin_faces: np.ndarray,
    max_translation_m: float = MAX_TRANSLATION_M,
    max_rotation_deg: float = MAX_ROTATION_DEG,
) -> Any:
    """Return the subject with its terminal clusters rigidly re-seated."""

    from dataclasses import replace

    from .chain_rest_fit_v1 import _global_to_local

    moved, b_final, report = reseat_terminals_v12(
        np.asarray(value.vertices_final, dtype=np.float64),
        np.asarray(value.B_final, dtype=np.float64),
        asset=asset,
        skin=skin,
        skin_faces=skin_faces,
        bone_parents=np.asarray(value.bone_parents, dtype=np.int64),
        max_translation_m=max_translation_m,
        max_rotation_deg=max_rotation_deg,
    )
    b_prefit = np.asarray(value.B_prefit, dtype=np.float64)
    parents = np.asarray(value.bone_parents, dtype=np.int64)
    build_report = dict(value.build_report)
    build_report["terminal_reseat_v12"] = report
    build_report["terminal_policy_note"] = (
        "hand/foot clusters rigidly re-seated at rest; no scaling, weights and "
        "topology untouched"
    )
    return replace(
        value,
        vertices_final=np.asarray(moved, dtype=np.float32),
        B_final=b_final,
        C_bone=b_final @ np.linalg.inv(b_prefit),
        target_local_bind=_global_to_local(b_final, parents),
        inverse_bind=np.linalg.inv(b_final),
        build_report=build_report,
    )


__all__ = [
    "MAX_ROTATION_DEG",
    "MAX_TRANSLATION_M",
    "TERMINAL_ROOTS",
    "apply_terminal_reseat_v12",
    "reseat_subject_terminals_v12",
    "reseat_terminals_v12",
    "solve_terminal_reseat_v12",
]
