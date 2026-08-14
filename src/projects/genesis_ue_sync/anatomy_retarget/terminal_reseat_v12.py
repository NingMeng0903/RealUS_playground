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


# Rotation is taken about the terminal's own joint centre so the swing is
# anatomical rather than about an arbitrary centroid.  Rotation alone is not
# enough: pivoting the foot about the ankle swings the toes through a long arc
# and measured *worse* than doing nothing (16.4 -> 17.1 mm), while rotation
# plus a small translation reaches 4.0 mm.  The translation does move the
# wrist/ankle bind origin, so ``evaluate_rest_anatomical_anchor_v11`` is what
# decides whether a given re-seat is allowed to stand.
MAX_TRANSLATION_M = 0.015
MAX_ROTATION_DEG = 15.0

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


def _blended_affine(
    asset: Any, vertex_ids: np.ndarray, transforms: np.ndarray
) -> np.ndarray:
    """LBS-blend per-controller transforms down to one affine per vertex."""

    indices = np.asarray(asset.driver_indices, dtype=np.int64)[vertex_ids]
    weights = np.asarray(asset.driver_weights, dtype=np.float64)[vertex_ids]
    selected = np.asarray(transforms, dtype=np.float64)[indices]
    return np.einsum("vk,vkab->vab", weights, selected)


def _controller_delta(
    n_bones: int, controllers: Sequence[int], transform: np.ndarray
) -> np.ndarray:
    """Identity on every bone, ``transform`` on the named controllers."""

    delta = np.tile(np.eye(4, dtype=np.float64), (int(n_bones), 1, 1))
    matrix = np.asarray(transform, dtype=np.float64)
    for controller in controllers:
        delta[int(controller)] = matrix
    return delta


def _lbs_moved(
    asset: Any, vertex_ids: np.ndarray, points: np.ndarray, delta: np.ndarray
) -> np.ndarray:
    """Apply an LBS-blended rest correction to a vertex subset."""

    affine = _blended_affine(asset, vertex_ids, delta)
    return np.einsum("vab,vb->va", affine[:, :3, :3], points) + affine[:, :3, 3]


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
    bind: np.ndarray,
    anchor_targets: Mapping[str, np.ndarray] | None = None,
    anchor_budget_m: Mapping[str, float] | None = None,
    pose_frames: Sequence[Mapping[str, Any]] | None = None,
    max_translation_m: float = MAX_TRANSLATION_M,
    max_rotation_deg: float = MAX_ROTATION_DEG,
    samples: int = OBJECTIVE_SAMPLES,
) -> dict[str, Any]:
    """Fit one bounded rigid transform per terminal cluster at rest.

    The rotation pivot is the terminal root's bind origin -- the wrist or
    ankle joint centre -- so the swing is anatomical rather than about an
    arbitrary centroid.

    ``anchor_targets`` / ``anchor_budget_m`` keep the solve inside
    ``evaluate_rest_anatomical_anchor_v11``: the re-seated root origin may not
    end further from its anatomical target than the prefit bind was.  Without
    that constraint the best skin fit walks the ankle 18.6 mm off the
    anatomical ankle and the anchor gate rejects it.
    """

    from scipy.optimize import minimize

    rest = np.asarray(vertices, dtype=np.float64)
    matrices = np.asarray(bind, dtype=np.float64)
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
        cluster = rest[ids]
        centre = matrices[root, :3, 3]
        generator = np.random.default_rng(abs(hash(root_name)) % (2**32))
        pick = (
            generator.choice(len(cluster), size=samples, replace=False)
            if len(cluster) > samples
            else np.arange(len(cluster))
        )
        sampled = cluster[pick]

        target = None if anchor_targets is None else anchor_targets.get(root_name)
        budget = None if anchor_budget_m is None else anchor_budget_m.get(root_name)

        # Under the terminal contract the posed transform of a terminal vertex
        # is ``G_src @ inv(B_src)`` blended by its frozen weights, which does
        # not depend on the re-seat at all.  So the posed cluster is an affine
        # image of the re-seated rest cluster and every pose can be scored for
        # the cost of one more signed-distance query.  Fitting rest only is not
        # enough: it wins 11 mm at rest and gives 3 mm back at 105 degrees of
        # knee flexion, because the posed foot pocket is a different shape.
        frames: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]] = [
            (skin_points, skin_triangles, None)
        ]
        for entry in pose_frames or ():
            affine = _blended_affine(
                asset, ids[pick], np.asarray(entry["source_transforms"])
            )
            frames.append(
                (
                    np.asarray(entry["skin"], dtype=np.float64),
                    np.asarray(entry["skin_faces"]),
                    affine,
                )
            )

        # T=0 baselines on the same samples the objective sees.  Mean-SSE
        # across frames is the wrong score: one rigid T cannot seat every
        # posed foot pocket, so the mean spends T-pose millimetres to buy
        # flexed-foot millimetres.  Minimax + a 1 mm non-regression wall
        # keeps any frame from paying for another.
        baselines: list[float] = []
        for frame_skin, frame_faces, affine in frames:
            unmoved = (
                sampled
                if affine is None
                else np.einsum("vab,vb->va", affine[:, :3, :3], sampled)
                + affine[:, :3, 3]
            )
            signed = _signed_distance(unmoved, frame_skin, frame_faces)
            baselines.append(float(max(0.0, float(np.max(signed)))))

        def cost(parameters: np.ndarray) -> float:
            rot_norm = float(np.linalg.norm(parameters[3:]))
            if rot_norm > bound:
                return 1.0e6 * (1.0 + rot_norm - bound)
            transform = _rigid(parameters, centre)
            if target is not None and budget is not None:
                origin = _apply(transform, centre[None, :])[0]
                excess = float(np.linalg.norm(origin - target)) - float(budget)
                if excess > 0.0:
                    # Outside the anatomical ball the anchor gate allows.
                    return 1.0e6 * (1.0 + excess)
            # Same LBS path as apply_terminal_reseat_v12: a seam vertex that
            # is only partly weighted to the ankle must not be scored as if
            # the whole rigid T landed on it.  The earlier full-T objective
            # fitted a motion the apply step then refused to realise.
            delta = _controller_delta(len(matrices), controllers, transform)
            moved = _lbs_moved(asset, ids[pick], sampled, delta)
            worst = 0.0
            sse = 0.0
            for baseline, (frame_skin, frame_faces, affine) in zip(baselines, frames):
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
                worst = max(worst, mx)
                sse += float(np.sum(poke * poke))
            return worst * worst + 1.0e-3 * sse / len(frames)

        translation_bound = max(float(max_translation_m), 1.0e-12)
        best = minimize(
            cost,
            np.zeros(6),
            method="Powell",
            bounds=[(-translation_bound, translation_bound)] * 3
            + [(-bound, bound)] * 3,
            options={"maxiter": 300, "xtol": 1.0e-4, "ftol": 1.0e-7},
        )
        parameters = np.asarray(best.x, dtype=np.float64).copy()
        if max_translation_m <= 0.0:
            parameters[:3] = 0.0
        rot_norm = float(np.linalg.norm(parameters[3:]))
        if rot_norm > bound:
            parameters[3:] *= bound / rot_norm
        transform = _rigid(parameters, centre)
        # Powell scores a subsample.  A 26 deg wrist swing can look fine on
        # 800 palm samples and throw the fingertips 60 mm out of the skin.
        # Reject any T that regresses the *full* cluster on any fitted frame.
        full_frames: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]] = [
            (skin_points, skin_triangles, None)
        ]
        for entry in pose_frames or ():
            full_frames.append(
                (
                    np.asarray(entry["skin"], dtype=np.float64),
                    np.asarray(entry["skin_faces"]),
                    _blended_affine(
                        asset, ids, np.asarray(entry["source_transforms"])
                    ),
                )
            )
        identity = np.eye(4, dtype=np.float64)
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

        def _accept(candidate: np.ndarray) -> tuple[bool, np.ndarray, list[float]]:
            from .terminal_pose_regression_v6 import (
                HAND_FOOT_COLLAPSE_BASELINE_MIN,
                HAND_FOOT_COLLAPSE_CANDIDATE_MAX,
            )

            matrix = _rigid(candidate, centre)
            delta = _controller_delta(len(matrices), controllers, matrix)
            after_m = _frame_max_outside(asset, ids, cluster, delta, full_frames)
            if any(
                later > earlier + NON_REGRESSION_SLACK_M + 1.0e-6
                for earlier, later in zip(full_before, after_m)
            ):
                return False, matrix, after_m
            if has_faces:
                for before_meshes, after_meshes in zip(area_before, _areas(delta)):
                    if not before_meshes:
                        continue
                    if (
                        float(np.mean(after_meshes))
                        < float(np.mean(before_meshes)) - AREA_INSIDE_REGRESSION_MAX
                    ):
                        return False, matrix, after_m
                    if any(
                        earlier > HAND_FOOT_COLLAPSE_BASELINE_MIN
                        and later < HAND_FOOT_COLLAPSE_CANDIDATE_MAX
                        for earlier, later in zip(before_meshes, after_meshes)
                    ):
                        return False, matrix, after_m
            return True, matrix, after_m

        identity_delta = _controller_delta(len(matrices), controllers, identity)
        full_before = _frame_max_outside(asset, ids, cluster, identity_delta, full_frames)
        area_before = _areas(identity_delta)
        ok, transform, full_after = _accept(parameters)
        rejected = bool(best.fun >= 1.0e5) or not ok
        if rejected and float(np.linalg.norm(parameters)) > 0.0:
            # Largest scaled T that still clears max-outside and area-inside
            # on every fitted frame, including T-pose.
            lo, hi = 0.0, 1.0
            kept = np.zeros(6, dtype=np.float64)
            kept_transform = identity
            kept_after = list(full_before)
            for _ in range(8):
                mid = 0.5 * (lo + hi)
                trial = parameters * mid
                accepted, matrix, after_m = _accept(trial)
                if accepted:
                    kept = trial
                    kept_transform = matrix
                    kept_after = after_m
                    lo = mid
                else:
                    hi = mid
            if lo > 0.0:
                parameters = kept
                transform = kept_transform
                full_after = kept_after
                rejected = False
            else:
                parameters[:] = 0.0
                transform = identity
                full_after = list(full_before)
        before = _signed_distance(cluster, skin_points, skin_triangles)
        after = _signed_distance(
            _lbs_moved(
                asset,
                ids,
                cluster,
                _controller_delta(len(matrices), controllers, transform),
            ),
            skin_points,
            skin_triangles,
        )
        result[root_name] = {
            "controllers": controllers,
            "vertex_ids": ids,
            "transform": transform,
            "pivot": "terminal_root_bind_origin",
            "root_origin_shift_m": float(
                np.linalg.norm(_apply(transform, centre[None, :])[0] - centre)
            ),
            "translation_m": float(np.linalg.norm(parameters[:3])),
            "rotation_deg": float(np.degrees(np.linalg.norm(parameters[3:]))),
            "max_outside_before_m": float(max(0.0, float(np.max(before)))),
            "max_outside_after_m": float(max(0.0, float(np.max(after)))),
            "outside_count_before": int(np.count_nonzero(before > 0.0)),
            "outside_count_after": int(np.count_nonzero(after > 0.0)),
            "rejected_full_cluster_regression": bool(rejected),
            "frame_max_outside_before_m": [float(v) for v in full_before],
            "frame_max_outside_after_m": [float(v) for v in full_after],
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

    The geometry has to move through the *same frozen LBS* that will later
    pose it, not by assigning the transform to a hand-picked vertex list.
    Changing a controller's bind changes the inverse bind for every vertex it
    drives, including distal tibia vertices that are only partly weighted to
    the ankle; if their rest position does not move by the matching weighted
    amount, posing amplifies the mismatch.  Both earlier attempts -- an
    LBS-weighted mesh blend against a full bind move, and a hard cluster
    assignment -- got that wrong and threw the feet 190 mm out of the skin.
    """

    from .chain_rest_fit_v1 import _weighted_rest_correction

    base = np.asarray(vertices, dtype=np.float64)
    matrices = np.asarray(bind, dtype=np.float64).copy()
    driver_indices = np.asarray(asset.driver_indices, dtype=np.int64)
    driver_weights = np.asarray(asset.driver_weights, dtype=np.float64)
    delta = np.tile(np.eye(4, dtype=np.float64), (len(matrices), 1, 1))
    report: dict[str, Any] = {}

    for root_name, entry in reseat.items():
        transform = np.asarray(entry["transform"], dtype=np.float64)
        controllers = list(entry["controllers"])
        for controller in controllers:
            delta[controller] = transform
            matrices[controller] = transform @ matrices[controller]

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
            "outside_count_before": int(entry["outside_count_before"]),
            "outside_count_after": int(entry["outside_count_after"]),
            "moved_vertex_count": int(len(active)),
            "max_vertex_shift_m": float(np.max(np.linalg.norm(shift, axis=1))),
            "rejected_full_cluster_regression": bool(
                entry.get("rejected_full_cluster_regression", False)
            ),
            "frame_max_outside_before_m": [
                float(v) for v in entry.get("frame_max_outside_before_m", ())
            ],
            "frame_max_outside_after_m": [
                float(v) for v in entry.get("frame_max_outside_after_m", ())
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
            "method": "bounded_rigid_terminal_reseat_v12",
            "scaling_applied": False,
            "max_translation_m": float(max_translation_m),
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
    """Anatomical target and allowed radius for each terminal root."""

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
    "AREA_INSIDE_REGRESSION_MAX",
    "MAX_ROTATION_DEG",
    "MAX_TRANSLATION_M",
    "NON_REGRESSION_SLACK_M",
    "TERMINAL_ROOTS",
    "apply_terminal_reseat_v12",
    "reseat_subject_terminals_v12",
    "reseat_terminals_v12",
    "solve_terminal_reseat_v12",
]
