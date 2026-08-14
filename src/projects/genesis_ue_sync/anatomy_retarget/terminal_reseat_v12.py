"""Bounded rigid re-seat of the hand/foot clusters in rest space.

Every version from V7 through V11 froze the terminals to copy-142, so their
containment numbers are identical and the feet are the top poke source in
every pose -- both independent visual reviews block on exactly that.

A single rigid T on ``Ankle_Rot`` cannot seat every posed foot pocket: the
hindfoot, metatarsals and toes have to move together, and the 1 mm
non-regression wall then leaves a 10 mm-class residual.  V12c splits the
authored tree at ``Arch_Rot`` (the only child of the ankle) so the hindfoot
and the forefoot each get their own bounded rigid T, composed as
``T_arch @ T_a`` on the arch subtree.  The objective is the outside-area
fraction the blind reviewer sees as red, not a single max-mm spike.

No scaling, no vertex surgery, no weight edits: the authored bone stays
exactly the shape the Blender rig authored.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np

from .chain_containment_v1 import _signed_distance, _vertex_areas
from .pose_map_v10 import FOOT_ROOTS, HAND_ROOTS
from .whole_chain_rest_fit_v1 import _descendants


# Rotation is taken about the terminal's own joint centre so the swing is
# anatomical rather than about an arbitrary centroid.  Rotation alone is not
# enough: pivoting the foot about the ankle swings the toes through a long arc
# and measured *worse* than doing nothing (16.4 -> 17.1 mm), while rotation
# plus a small translation reaches 4.0 mm.  The translation does move the
# wrist/ankle bind origin, so ``evaluate_rest_anatomical_anchor_v11`` is what
# decides whether a given re-seat is allowed to stand.
MAX_TRANSLATION_M = 0.015
MAX_ROTATION_DEG = 15.0

# Arch_Rot is not in JOINT_SPECS, so it has no anatomical ball.  Keep the
# extra forefoot translation tighter than the ankle so the midfoot seam
# cannot walk away from the tarsals.
FOREFOOT_MAX_TRANSLATION_M = 0.008

# Objective sample size.  The clusters have 5k-10k vertices and the signed
# distance query dominates the solve, so the fit runs on a deterministic
# subsample and the reported metrics are recomputed on every vertex.
OBJECTIVE_SAMPLES = 800

# A frame may not get more than this much worse than the unmoved cluster.
# Same millimetre as the absolute-poke regression gate: one rigid T cannot
# seat every posed foot pocket, and mean-SSE will happily trade a 6 mm
# T-pose regression for a 11 mm flexed-foot win.
NON_REGRESSION_SLACK_M = 0.001

# Shadow terminal / body gates fail a candidate if mean area-inside drops
# more than 2%.  Max-outside can stay inside 1 mm while hundreds of vertices
# cross the skin; the accept step therefore also holds this area budget.
AREA_INSIDE_REGRESSION_MAX = 0.02

# Ankle / wrist roots keep the 142 terminal FK contract.  Arch_Rot is the
# authored midfoot root (owns every metatarsal and the toe chains).
FOREFOOT_ROOTS = ("Arch_Rot_L", "Arch_Rot_R")
TERMINAL_ROOTS = (*HAND_ROOTS, *FOOT_ROOTS)
RESEAT_ORDER = (*HAND_ROOTS, *FOOT_ROOTS, *FOREFOOT_ROOTS)

_ARCH_PARENT = {
    "Arch_Rot_L": "Ankle_Rot_L",
    "Arch_Rot_R": "Ankle_Rot_R",
}


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


def _blended_affine(
    asset: Any, vertex_ids: np.ndarray, transforms: np.ndarray
) -> np.ndarray:
    """LBS-blend per-controller transforms down to one affine per vertex."""

    indices = np.asarray(asset.driver_indices, dtype=np.int64)[vertex_ids]
    weights = np.asarray(asset.driver_weights, dtype=np.float64)[vertex_ids]
    selected = np.asarray(transforms, dtype=np.float64)[indices]
    return np.einsum("vk,vkab->vab", weights, selected)


def _controller_delta(
    n_bones: int, assignments: Mapping[int, np.ndarray]
) -> np.ndarray:
    """Identity on every bone, then the named per-controller transforms."""

    delta = np.tile(np.eye(4, dtype=np.float64), (int(n_bones), 1, 1))
    for controller, transform in assignments.items():
        delta[int(controller)] = np.asarray(transform, dtype=np.float64)
    return delta


def _lbs_moved(
    asset: Any, vertex_ids: np.ndarray, points: np.ndarray, delta: np.ndarray
) -> np.ndarray:
    """Apply an LBS-blended rest correction to a vertex subset."""

    affine = _blended_affine(asset, vertex_ids, delta)
    return np.einsum("vab,vb->va", affine[:, :3, :3], points) + affine[:, :3, 3]


def _controllers_for(
    root_name: str, names: Sequence[str], parents: np.ndarray
) -> list[int]:
    """Partition the ankle from its arch subtree.

    ``Ankle_Rot_*`` owns only itself (hindfoot meshes).  ``Arch_Rot_*`` and
    the wrists own the full descendant set, including the root.
    """

    root = names.index(root_name)
    if root_name in FOOT_ROOTS:
        return [int(root)]
    return sorted(int(i) for i in _descendants(parents, root))


def _parent_root(root_name: str) -> str | None:
    return _ARCH_PARENT.get(root_name)


def _cluster_mesh_areas(
    asset: Any,
    vertices: np.ndarray,
    skin: np.ndarray,
    skin_faces: np.ndarray,
    controllers: Sequence[int],
) -> list[float]:
    """Per-mesh area-inside of bone meshes owned by the cluster controllers."""

    from .terminal_pose_regression_v6 import area_inside_fraction

    wanted = {int(value) for value in controllers}
    owners = np.asarray(asset.source_mesh_controller_bones, dtype=np.int64)
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    tissues = [str(label).strip().lower() for label in asset.source_tissues]
    faces = np.asarray(asset.faces)
    fracs: list[float] = []
    for owner, (start, stop), tissue in zip(owners.tolist(), ranges.tolist(), tissues):
        if tissue != "bone" or int(owner) not in wanted:
            continue
        frac, _outside = area_inside_fraction(
            vertices, faces, skin, skin_faces, int(start), int(stop)
        )
        fracs.append(float(frac))
    return fracs


def _outside_area_fraction(
    points: np.ndarray,
    skin: np.ndarray,
    skin_faces: np.ndarray,
    weights: np.ndarray,
) -> float:
    signed = _signed_distance(points, skin, skin_faces)
    poke = signed > 0.0
    total = float(np.sum(weights))
    if not len(poke):
        return 0.0
    if total <= 0.0:
        return float(np.mean(poke))
    return float(np.sum(weights[poke]) / total)


def _frame_max_outside(
    asset: Any,
    vertex_ids: np.ndarray,
    rest_points: np.ndarray,
    delta: np.ndarray,
    frames: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray | None]],
) -> list[float]:
    """Max outside depth of a cluster on every fitted frame."""

    moved = _lbs_moved(asset, vertex_ids, rest_points, delta)
    out: list[float] = []
    for frame_skin, frame_faces, affine in frames:
        posed = (
            moved
            if affine is None
            else np.einsum("vab,vb->va", affine[:, :3, :3], moved)
            + affine[:, :3, 3]
        )
        signed = _signed_distance(posed, frame_skin, frame_faces)
        out.append(float(max(0.0, float(np.max(signed)))))
    return out


def _frame_outside_area(
    asset: Any,
    vertex_ids: np.ndarray,
    rest_points: np.ndarray,
    weights: np.ndarray,
    delta: np.ndarray,
    frames: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray | None]],
) -> list[float]:
    moved = _lbs_moved(asset, vertex_ids, rest_points, delta)
    out: list[float] = []
    for frame_skin, frame_faces, affine in frames:
        posed = (
            moved
            if affine is None
            else np.einsum("vab,vb->va", affine[:, :3, :3], moved)
            + affine[:, :3, 3]
        )
        out.append(_outside_area_fraction(posed, frame_skin, frame_faces, weights))
    return out


def _rigid(parameters: np.ndarray, centre: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    matrix = np.eye(4, dtype=np.float64)
    rotation = Rotation.from_rotvec(np.asarray(parameters[3:], dtype=np.float64)).as_matrix()
    matrix[:3, :3] = rotation
    matrix[:3, 3] = centre + np.asarray(parameters[:3], dtype=np.float64) - rotation @ centre
    return matrix


def _apply(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ np.asarray(transform[:3, :3]).T + np.asarray(transform[:3, 3])


def _cluster_weights(
    rest: np.ndarray, asset: Any, vertex_ids: np.ndarray
) -> np.ndarray:
    faces = getattr(asset, "faces", None)
    if faces is None:
        return np.ones(len(vertex_ids), dtype=np.float64)
    areas = _vertex_areas(rest, np.asarray(faces))
    weights = areas[vertex_ids]
    if float(np.sum(weights)) <= 0.0:
        return np.ones(len(vertex_ids), dtype=np.float64)
    return weights


def _assignments(
    *,
    n_bones: int,
    controllers: Sequence[int],
    transform: np.ndarray,
    parent_controllers: Sequence[int] | None,
    parent_transform: np.ndarray | None,
) -> dict[int, np.ndarray]:
    """Per-controller deltas: parent T on the proximal set, composed on distal."""

    assigned: dict[int, np.ndarray] = {}
    parent = (
        np.asarray(parent_transform, dtype=np.float64)
        if parent_transform is not None
        else np.eye(4, dtype=np.float64)
    )
    if parent_controllers:
        for controller in parent_controllers:
            assigned[int(controller)] = parent
    composed = np.asarray(transform, dtype=np.float64) @ parent
    for controller in controllers:
        assigned[int(controller)] = composed
    return assigned


def _fit_one_cluster(
    *,
    root_name: str,
    rest: np.ndarray,
    asset: Any,
    matrices: np.ndarray,
    names: Sequence[str],
    parents: np.ndarray,
    skin_points: np.ndarray,
    skin_triangles: np.ndarray,
    pose_frames: Sequence[Mapping[str, Any]] | None,
    anchor_targets: Mapping[str, np.ndarray] | None,
    anchor_budget_m: Mapping[str, float] | None,
    parent_transform: np.ndarray | None,
    parent_controllers: Sequence[int] | None,
    max_translation_m: float,
    max_rotation_deg: float,
    samples: int,
) -> dict[str, Any]:
    from scipy.optimize import minimize

    controllers = _controllers_for(root_name, names, parents)
    ids = _cluster_vertex_ids(asset, controllers)
    if not len(ids):
        raise ValueError(f"terminal cluster {root_name} has no bone vertices")
    cluster = rest[ids]
    centre = np.asarray(matrices[names.index(root_name), :3, 3], dtype=np.float64)
    if parent_transform is not None:
        centre = _apply(parent_transform, centre[None, :])[0]
    generator = np.random.default_rng(abs(hash(root_name)) % (2**32))
    pick = (
        generator.choice(len(cluster), size=samples, replace=False)
        if len(cluster) > samples
        else np.arange(len(cluster))
    )
    sampled = cluster[pick]
    weights = _cluster_weights(rest, asset, ids)
    sample_weights = weights[pick]
    bound = float(np.deg2rad(max_rotation_deg))
    target = None if anchor_targets is None else anchor_targets.get(root_name)
    budget = None if anchor_budget_m is None else anchor_budget_m.get(root_name)

    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]] = [
        (skin_points, skin_triangles, None)
    ]
    for entry in pose_frames or ():
        frames.append(
            (
                np.asarray(entry["skin"], dtype=np.float64),
                np.asarray(entry["skin_faces"]),
                _blended_affine(
                    asset, ids[pick], np.asarray(entry["source_transforms"])
                ),
            )
        )

    identity = np.eye(4, dtype=np.float64)
    parent = (
        np.asarray(parent_transform, dtype=np.float64)
        if parent_transform is not None
        else identity
    )

    def _delta_of(transform: np.ndarray) -> np.ndarray:
        return _controller_delta(
            len(matrices),
            _assignments(
                n_bones=len(matrices),
                controllers=controllers,
                transform=transform,
                parent_controllers=parent_controllers,
                parent_transform=parent_transform,
            ),
        )

    inherit_name = None
    inherit_controllers: list[int] = []
    inherit_ids = np.empty(0, dtype=np.int64)
    inherit_pick = np.empty(0, dtype=np.int64)
    inherit_sampled = np.empty((0, 3), dtype=np.float64)
    inherit_frames: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]] = []
    if root_name in FOOT_ROOTS:
        candidate = "Arch_Rot_L" if root_name.endswith("_L") else "Arch_Rot_R"
        if candidate in names:
            inherit_name = candidate
            inherit_controllers = _controllers_for(candidate, names, parents)
            inherit_ids = _cluster_vertex_ids(asset, inherit_controllers)
            if len(inherit_ids):
                inherit_cluster_pts = rest[inherit_ids]
                inherit_pick = (
                    generator.choice(len(inherit_ids), size=min(samples, len(inherit_ids)), replace=False)
                    if len(inherit_ids) > samples
                    else np.arange(len(inherit_ids))
                )
                inherit_sampled = inherit_cluster_pts[inherit_pick]
                inherit_frames = [(skin_points, skin_triangles, None)]
                for entry in pose_frames or ():
                    inherit_frames.append(
                        (
                            np.asarray(entry["skin"], dtype=np.float64),
                            np.asarray(entry["skin_faces"]),
                            _blended_affine(
                                asset,
                                inherit_ids[inherit_pick],
                                np.asarray(entry["source_transforms"]),
                            ),
                        )
                    )

    def _inherit_delta(transform: np.ndarray) -> np.ndarray:
        # Preview T_a on the arch subtree (T_rel = I) so an ankle T that
        # swings the toes 20 mm cannot hide behind a hindfoot-only wall.
        assigned = {int(c): transform for c in controllers}
        for controller in inherit_controllers:
            assigned[int(controller)] = transform
        return _controller_delta(len(matrices), assigned)

    true_identity = _controller_delta(len(matrices), {})
    baselines_mm: list[float] = []
    unmoved_cluster = sampled
    for frame_skin, frame_faces, affine in frames:
        unmoved = (
            unmoved_cluster
            if affine is None
            else np.einsum("vab,vb->va", affine[:, :3, :3], unmoved_cluster)
            + affine[:, :3, 3]
        )
        signed = _signed_distance(unmoved, frame_skin, frame_faces)
        baselines_mm.append(float(max(0.0, float(np.max(signed)))))
    inherit_baselines: list[float] = []
    for frame_skin, frame_faces, affine in inherit_frames:
        unmoved = (
            inherit_sampled
            if affine is None
            else np.einsum("vab,vb->va", affine[:, :3, :3], inherit_sampled)
            + affine[:, :3, 3]
        )
        signed = _signed_distance(unmoved, frame_skin, frame_faces)
        inherit_baselines.append(float(max(0.0, float(np.max(signed)))))

    def cost(parameters: np.ndarray) -> float:
        rot_norm = float(np.linalg.norm(parameters[3:]))
        if rot_norm > bound:
            return 1.0e6 * (1.0 + rot_norm - bound)
        transform = _rigid(parameters, centre)
        if target is not None and budget is not None:
            origin = _apply(transform @ parent, matrices[names.index(root_name), :3, 3][None, :])[0]
            excess = float(np.linalg.norm(origin - target)) - float(budget)
            if excess > 0.0:
                return 1.0e6 * (1.0 + excess)
        delta = _delta_of(transform)
        moved = _lbs_moved(asset, ids[pick], sampled, delta)
        worst_area = 0.0
        worst_mm = 0.0
        for baseline, (frame_skin, frame_faces, affine) in zip(baselines_mm, frames):
            posed = (
                moved
                if affine is None
                else np.einsum("vab,vb->va", affine[:, :3, :3], moved)
                + affine[:, :3, 3]
            )
            signed = _signed_distance(posed, frame_skin, frame_faces)
            poke = np.maximum(signed, 0.0)
            mx = float(np.max(poke)) if len(poke) else 0.0
            extra = mx - float(baseline) - NON_REGRESSION_SLACK_M
            if extra > 0.0:
                return 1.0e6 * (1.0 + extra)
            worst_mm = max(worst_mm, mx)
            worst_area = max(
                worst_area,
                _outside_area_fraction(posed, frame_skin, frame_faces, sample_weights),
            )
        if len(inherit_ids):
            inherited = _lbs_moved(
                asset, inherit_ids[inherit_pick], inherit_sampled, _inherit_delta(transform)
            )
            for baseline, (frame_skin, frame_faces, affine) in zip(
                inherit_baselines, inherit_frames
            ):
                posed = (
                    inherited
                    if affine is None
                    else np.einsum("vab,vb->va", affine[:, :3, :3], inherited)
                    + affine[:, :3, 3]
                )
                mx = float(max(0.0, float(np.max(_signed_distance(posed, frame_skin, frame_faces)))))
                if mx > float(baseline) + NON_REGRESSION_SLACK_M:
                    return 1.0e6 * (1.0 + mx - float(baseline))
        return worst_area * worst_area + 1.0e-4 * worst_mm * worst_mm

    translation_bound = max(float(max_translation_m), 1.0e-12)
    best = minimize(
        cost,
        np.zeros(6),
        method="Powell",
        bounds=[(-translation_bound, translation_bound)] * 3 + [(-bound, bound)] * 3,
        options={"maxiter": 300, "xtol": 1.0e-4, "ftol": 1.0e-7},
    )
    parameters = np.asarray(best.x, dtype=np.float64).copy()
    if max_translation_m <= 0.0:
        parameters[:3] = 0.0
    rot_norm = float(np.linalg.norm(parameters[3:]))
    if rot_norm > bound:
        parameters[3:] *= bound / rot_norm
    transform = _rigid(parameters, centre)

    full_frames: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]] = [
        (skin_points, skin_triangles, None)
    ]
    for entry in pose_frames or ():
        full_frames.append(
            (
                np.asarray(entry["skin"], dtype=np.float64),
                np.asarray(entry["skin_faces"]),
                _blended_affine(asset, ids, np.asarray(entry["source_transforms"])),
            )
        )
    has_faces = getattr(asset, "faces", None) is not None

    def _scatter_posed(delta: np.ndarray, affine: np.ndarray | None) -> np.ndarray:
        moved = _lbs_moved(asset, ids, cluster, delta)
        posed = (
            moved
            if affine is None
            else np.einsum("vab,vb->va", affine[:, :3, :3], moved)
            + affine[:, :3, 3]
        )
        full = np.asarray(rest, dtype=np.float64).copy()
        full[ids] = posed
        return full

    def _areas(delta: np.ndarray) -> list[list[float]]:
        if not has_faces:
            return [[] for _ in full_frames]
        return [
            _cluster_mesh_areas(
                asset, _scatter_posed(delta, affine), frame_skin, frame_faces, controllers
            )
            for frame_skin, frame_faces, affine in full_frames
        ]

    def _accept(candidate: np.ndarray) -> tuple[bool, np.ndarray, list[float], list[float]]:
        from .terminal_pose_regression_v6 import (
            HAND_FOOT_COLLAPSE_BASELINE_MIN,
            HAND_FOOT_COLLAPSE_CANDIDATE_MAX,
        )

        matrix = _rigid(candidate, centre)
        delta = _delta_of(matrix)
        after_m = _frame_max_outside(asset, ids, cluster, delta, full_frames)
        after_area = _frame_outside_area(
            asset, ids, cluster, weights, delta, full_frames
        )
        if any(
            later > earlier + NON_REGRESSION_SLACK_M + 1.0e-6
            for earlier, later in zip(original_before, after_m)
        ):
            return False, matrix, after_m, after_area
        if len(inherit_ids):
            inherit_after = _frame_max_outside(
                asset,
                inherit_ids,
                rest[inherit_ids],
                _inherit_delta(matrix),
                inherit_full_frames,
            )
            if any(
                later > earlier + NON_REGRESSION_SLACK_M + 1.0e-6
                for earlier, later in zip(inherit_original, inherit_after)
            ):
                return False, matrix, after_m, after_area
        if has_faces:
            for before_meshes, after_meshes in zip(area_before, _areas(delta)):
                if not before_meshes:
                    continue
                if (
                    float(np.mean(after_meshes))
                    < float(np.mean(before_meshes)) - AREA_INSIDE_REGRESSION_MAX
                ):
                    return False, matrix, after_m, after_area
                if any(
                    earlier > HAND_FOOT_COLLAPSE_BASELINE_MIN
                    and later < HAND_FOOT_COLLAPSE_CANDIDATE_MAX
                    for earlier, later in zip(before_meshes, after_meshes)
                ):
                    return False, matrix, after_m, after_area
        return True, matrix, after_m, after_area

    identity_delta = _delta_of(identity)
    original_before = _frame_max_outside(asset, ids, cluster, true_identity, full_frames)
    inherit_full_frames: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]] = []
    inherit_original: list[float] = []
    if len(inherit_ids):
        inherit_full_frames = [(skin_points, skin_triangles, None)]
        for entry in pose_frames or ():
            inherit_full_frames.append(
                (
                    np.asarray(entry["skin"], dtype=np.float64),
                    np.asarray(entry["skin_faces"]),
                    _blended_affine(
                        asset, inherit_ids, np.asarray(entry["source_transforms"])
                    ),
                )
            )
        inherit_original = _frame_max_outside(
            asset, inherit_ids, rest[inherit_ids], true_identity, inherit_full_frames
        )
    full_before = _frame_max_outside(asset, ids, cluster, identity_delta, full_frames)
    area_frac_before = _frame_outside_area(
        asset, ids, cluster, weights, identity_delta, full_frames
    )
    area_before = _areas(identity_delta)
    ok, transform, full_after, area_after = _accept(parameters)
    rejected = bool(best.fun >= 1.0e5) or not ok
    if rejected and float(np.linalg.norm(parameters)) > 0.0:
        lo, hi = 0.0, 1.0
        kept = np.zeros(6, dtype=np.float64)
        kept_transform = identity
        kept_after = list(full_before)
        kept_area = list(area_frac_before)
        for _ in range(8):
            mid = 0.5 * (lo + hi)
            trial = parameters * mid
            accepted, matrix, after_m, after_area = _accept(trial)
            if accepted:
                kept = trial
                kept_transform = matrix
                kept_after = after_m
                kept_area = after_area
                lo = mid
            else:
                hi = mid
        if lo > 0.0:
            parameters = kept
            transform = kept_transform
            full_after = kept_after
            area_after = kept_area
            rejected = False
        else:
            parameters[:] = 0.0
            transform = identity
            full_after = list(full_before)
            area_after = list(area_frac_before)

    composed = transform @ parent
    before = _signed_distance(
        _lbs_moved(asset, ids, cluster, identity_delta), skin_points, skin_triangles
    )
    after = _signed_distance(
        _lbs_moved(asset, ids, cluster, _delta_of(transform)),
        skin_points,
        skin_triangles,
    )
    bind_origin = np.asarray(matrices[names.index(root_name), :3, 3], dtype=np.float64)
    return {
        "controllers": controllers,
        "vertex_ids": ids,
        "transform": transform,
        "composed_transform": composed,
        "parent": _parent_root(root_name),
        "pivot": "terminal_root_bind_origin",
        "root_origin_shift_m": float(
            np.linalg.norm(_apply(composed, bind_origin[None, :])[0] - bind_origin)
        ),
        "translation_m": float(np.linalg.norm(parameters[:3])),
        "rotation_deg": float(np.degrees(np.linalg.norm(parameters[3:]))),
        "max_outside_before_m": float(max(0.0, float(np.max(before)))),
        "max_outside_after_m": float(max(0.0, float(np.max(after)))),
        "outside_area_before": float(
            _outside_area_fraction(
                _lbs_moved(asset, ids, cluster, identity_delta),
                skin_points,
                skin_triangles,
                weights,
            )
        ),
        "outside_area_after": float(
            _outside_area_fraction(
                _lbs_moved(asset, ids, cluster, _delta_of(transform)),
                skin_points,
                skin_triangles,
                weights,
            )
        ),
        "outside_count_before": int(np.count_nonzero(before > 0.0)),
        "outside_count_after": int(np.count_nonzero(after > 0.0)),
        "rejected_full_cluster_regression": bool(rejected),
        "frame_max_outside_before_m": [float(v) for v in full_before],
        "frame_max_outside_after_m": [float(v) for v in full_after],
        "frame_outside_area_before": [float(v) for v in area_frac_before],
        "frame_outside_area_after": [float(v) for v in area_after],
    }


def solve_terminal_reseat_v12(
    vertices: np.ndarray,
    *,
    asset: Any,
    skin: np.ndarray,
    skin_faces: np.ndarray,
    bone_parents: np.ndarray,
    bind: np.ndarray,
    anchor_targets: Mapping[str, np.ndarray] | None = None,
    anchor_budget_m: Mapping[str, float] | None = None,
    pose_frames: Sequence[Mapping[str, Any]] | None = None,
    max_translation_m: float = MAX_TRANSLATION_M,
    max_rotation_deg: float = MAX_ROTATION_DEG,
    samples: int = OBJECTIVE_SAMPLES,
) -> dict[str, Any]:
    """Fit one bounded rigid transform per terminal / arch cluster at rest.

    Ankles are fitted first on hindfoot meshes only.  ``Arch_Rot`` is then
    fitted relative to that ankle T, pivoted at the already-moved arch
    origin.  Hands stay a single descendant cluster.

    ``anchor_targets`` / ``anchor_budget_m`` apply only to wrist/ankle roots
    that sit in ``evaluate_rest_anatomical_anchor_v11``.
    """

    rest = np.asarray(vertices, dtype=np.float64)
    matrices = np.asarray(bind, dtype=np.float64)
    names = [str(name) for name in asset.source_bone_names]
    parents = np.asarray(bone_parents, dtype=np.int64)
    skin_points = np.asarray(skin, dtype=np.float64)
    skin_triangles = np.asarray(skin_faces)
    present = set(names)

    result: dict[str, Any] = {}
    for root_name in RESEAT_ORDER:
        if root_name not in present:
            continue
        parent_name = _parent_root(root_name)
        parent_transform = None
        parent_controllers = None
        translation_cap = float(max_translation_m)
        if parent_name is not None and parent_name in result:
            parent_transform = np.asarray(
                result[parent_name]["transform"], dtype=np.float64
            )
            parent_controllers = list(result[parent_name]["controllers"])
            translation_cap = min(translation_cap, FOREFOOT_MAX_TRANSLATION_M)
        elif parent_name is not None:
            translation_cap = min(translation_cap, FOREFOOT_MAX_TRANSLATION_M)
        result[root_name] = _fit_one_cluster(
            root_name=root_name,
            rest=rest,
            asset=asset,
            matrices=matrices,
            names=names,
            parents=parents,
            skin_points=skin_points,
            skin_triangles=skin_triangles,
            pose_frames=pose_frames,
            anchor_targets=anchor_targets,
            anchor_budget_m=anchor_budget_m,
            parent_transform=parent_transform,
            parent_controllers=parent_controllers,
            max_translation_m=translation_cap,
            max_rotation_deg=max_rotation_deg,
            samples=samples,
        )
    return result


def apply_terminal_reseat_v12(
    vertices: np.ndarray,
    bind: np.ndarray,
    *,
    asset: Any,
    reseat: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Move terminal geometry and terminal binds by the composed rigid Ts.

    Proximal clusters (wrist / ankle) write ``T`` onto their controllers.
    Distal ``Arch_Rot`` writes ``T_arch @ T_a`` so the forefoot inherits the
    ankle motion and then adds its own relative T.  Geometry still moves
    through the frozen LBS; a seam vertex weighted to both ankle and arch
    sees the blend of those two deltas.
    """

    from .chain_rest_fit_v1 import _weighted_rest_correction

    base = np.asarray(vertices, dtype=np.float64)
    matrices = np.asarray(bind, dtype=np.float64).copy()
    driver_indices = np.asarray(asset.driver_indices, dtype=np.int64)
    driver_weights = np.asarray(asset.driver_weights, dtype=np.float64)
    delta = np.tile(np.eye(4, dtype=np.float64), (len(matrices), 1, 1))
    report: dict[str, Any] = {}

    proximal = [
        (name, entry)
        for name, entry in reseat.items()
        if not entry.get("parent")
    ]
    distal = [
        (name, entry)
        for name, entry in reseat.items()
        if entry.get("parent")
    ]
    parent_ts = {
        name: np.asarray(entry["transform"], dtype=np.float64)
        for name, entry in proximal
    }
    for root_name, entry in proximal:
        transform = np.asarray(entry["transform"], dtype=np.float64)
        for controller in entry["controllers"]:
            delta[int(controller)] = transform
            matrices[int(controller)] = transform @ matrices[int(controller)]
    for root_name, entry in distal:
        relative = np.asarray(entry["transform"], dtype=np.float64)
        parent_name = str(entry["parent"])
        parent = parent_ts.get(parent_name, np.eye(4, dtype=np.float64))
        composed = relative @ parent
        for controller in entry["controllers"]:
            delta[int(controller)] = composed
            matrices[int(controller)] = composed @ matrices[int(controller)]

    moved = _weighted_rest_correction(base, driver_indices, driver_weights, delta)

    for root_name, entry in reseat.items():
        active = np.asarray(entry["vertex_ids"], dtype=np.int64)
        shift = moved[active] - base[active]
        report[root_name] = {
            "translation_m": float(entry["translation_m"]),
            "rotation_deg": float(entry["rotation_deg"]),
            "root_origin_shift_m": float(entry["root_origin_shift_m"]),
            "max_outside_before_m": float(entry["max_outside_before_m"]),
            "max_outside_after_m": float(entry["max_outside_after_m"]),
            "outside_area_before": float(entry.get("outside_area_before", 0.0)),
            "outside_area_after": float(entry.get("outside_area_after", 0.0)),
            "outside_count_before": int(entry["outside_count_before"]),
            "outside_count_after": int(entry["outside_count_after"]),
            "moved_vertex_count": int(len(active)),
            "max_vertex_shift_m": float(np.max(np.linalg.norm(shift, axis=1)))
            if len(active)
            else 0.0,
            "rejected_full_cluster_regression": bool(
                entry.get("rejected_full_cluster_regression", False)
            ),
            "parent": entry.get("parent"),
            "frame_max_outside_before_m": [
                float(v) for v in entry.get("frame_max_outside_before_m", ())
            ],
            "frame_max_outside_after_m": [
                float(v) for v in entry.get("frame_max_outside_after_m", ())
            ],
            "frame_outside_area_before": [
                float(v) for v in entry.get("frame_outside_area_before", ())
            ],
            "frame_outside_area_after": [
                float(v) for v in entry.get("frame_outside_area_after", ())
            ],
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
    anchor_targets: Mapping[str, np.ndarray] | None = None,
    anchor_budget_m: Mapping[str, float] | None = None,
    pose_frames: Sequence[Mapping[str, Any]] | None = None,
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
        bind=bind,
        anchor_targets=anchor_targets,
        anchor_budget_m=anchor_budget_m,
        pose_frames=pose_frames,
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
            "method": "bounded_rigid_terminal_reseat_v12c",
            "scaling_applied": False,
            "max_translation_m": float(max_translation_m),
            "forefoot_max_translation_m": float(FOREFOOT_MAX_TRANSLATION_M),
            "max_rotation_deg": float(max_rotation_deg),
            "non_regression_slack_m": float(NON_REGRESSION_SLACK_M),
            "area_inside_regression_max": float(AREA_INSIDE_REGRESSION_MAX),
            "fitted_pose_count": 1 + len(pose_frames or ()),
            "clusters": report,
            "elapsed_seconds": float(time.perf_counter() - started),
        },
    )


def _anchor_constraints(
    value: Any, *, asset: Any, calibration: Any
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Anatomical target and allowed radius for each wrist/ankle root."""

    from .anatomical_calibration_v1 import JOINT_SPECS
    from .segment_similarity_rest_v10 import subject_anatomical_pivots_v10

    names = [str(name) for name in asset.source_bone_names]
    pivots = subject_anatomical_pivots_v10(asset, calibration)
    prefit = np.asarray(value.B_prefit, dtype=np.float64)
    targets: dict[str, np.ndarray] = {}
    budget: dict[str, float] = {}
    for index, spec in enumerate(JOINT_SPECS):
        if spec.controller not in TERMINAL_ROOTS:
            continue
        origin = pivots[index, :3, 3]
        controller = names.index(spec.controller)
        targets[spec.controller] = origin
        budget[spec.controller] = float(
            np.linalg.norm(prefit[controller, :3, 3] - origin)
        )
    return targets, budget


def reseat_subject_terminals_v12(
    value: Any,
    *,
    asset: Any,
    calibration: Any,
    skin: np.ndarray,
    skin_faces: np.ndarray,
    pose_frames: Sequence[Mapping[str, Any]] | None = None,
    max_translation_m: float = MAX_TRANSLATION_M,
    max_rotation_deg: float = MAX_ROTATION_DEG,
) -> Any:
    """Return the subject with its terminal clusters rigidly re-seated."""

    from dataclasses import replace

    from .chain_rest_fit_v1 import _global_to_local

    targets, budget = _anchor_constraints(
        value, asset=asset, calibration=calibration
    )
    moved, b_final, report = reseat_terminals_v12(
        np.asarray(value.vertices_final, dtype=np.float64),
        np.asarray(value.B_final, dtype=np.float64),
        asset=asset,
        skin=skin,
        skin_faces=skin_faces,
        bone_parents=np.asarray(value.bone_parents, dtype=np.int64),
        anchor_targets=targets,
        anchor_budget_m=budget,
        pose_frames=pose_frames,
        max_translation_m=max_translation_m,
        max_rotation_deg=max_rotation_deg,
    )
    b_prefit = np.asarray(value.B_prefit, dtype=np.float64)
    parents = np.asarray(value.bone_parents, dtype=np.int64)
    build_report = dict(value.build_report)
    build_report["terminal_reseat_v12"] = report
    build_report["terminal_policy_note"] = (
        "hand clusters plus split ankle/arch foot clusters rigidly re-seated "
        "at rest; no scaling, weights and topology untouched"
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
    "AREA_INSIDE_REGRESSION_MAX",
    "FOREFOOT_MAX_TRANSLATION_M",
    "FOREFOOT_ROOTS",
    "MAX_ROTATION_DEG",
    "MAX_TRANSLATION_M",
    "NON_REGRESSION_SLACK_M",
    "RESEAT_ORDER",
    "TERMINAL_ROOTS",
    "apply_terminal_reseat_v12",
    "reseat_subject_terminals_v12",
    "reseat_terminals_v12",
    "solve_terminal_reseat_v12",
]
