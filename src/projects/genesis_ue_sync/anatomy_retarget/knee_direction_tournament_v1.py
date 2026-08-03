"""Multi-direction knee containment tournament (Phase K reset).

Distinct spikes vs V7 authority. Banned: SE3 coupled knee, Femur axial scale,
SKEL limb-length hard anchor, PCA corrective net.

Directions:
  - inward_shared_t: OSSO soft-tissue idea — shared femur+patella+shank translation only
  - patella_only: local Patella_Rotate SE(3); hinge bones untouched
  - delta_j_centerline: SKEL-J-style bounded knee-center nudge from skin centerline
  - weight_refit: Pinocchio-style distal Knee_Rotate mass (existing refit_weights_v1)
  - v8_existing: already-built bone-first V8 shadow (compare only)

Gates: contact_nonregress_v9 + flex femur/patella outside↓. Genesis slim for all.
Never promotes trusted/latest.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .anatomical_calibration_v1 import AnatomicalCalibrationV1, JOINT_SPECS, _measure_frames
from .anatomy_lbs import source_bone_posed_global
from .chain_containment_v1 import _signed_distance_details
from .chain_rest_fit_v1 import (
    LOWER_JOINT_NAMES,
    ChainRestFitSubjectV1,
    _global_to_local,
    _knee_gap_contact_violation_m,
    _weighted_rest_correction,
)
from .joint_contact_nonregress_v9 import (
    MAX_FLEX_CONTACT_ABS_M,
    MAX_FLEX_GAP_REGRESSION_M,
    MAX_GAP_REGRESSION_M,
    evaluate_joint_contact_nonregress_v9,
)
from .joint_contact_v7 import FrozenJointMaterialDomainsV7
from .pose_map_v1 import build_pose_map_v1, pose_whole_chain_vertices
from .smplx_body_surface_v7 import smplx_body_surface_v7
from .v8_artifacts import SourceOperatorV8, materialize_subject


TOURNAMENT_KIND = "KneeDirectionTournamentV1"
TRANSLATION_BOUND_M = 0.010
ROTATION_BOUND_RAD = float(np.deg2rad(8.0))
DELTA_J_BOUND_M = 0.012
FIT_GAP_SAFETY_M = 2.0e-4
INFEASIBLE = 1.0e3


def _mesh_ids(asset: Any, names: tuple[str, ...]) -> np.ndarray:
    lookup = {
        str(n): (int(s), int(e))
        for n, (s, e) in zip(asset.source_mesh_names, asset.source_vertex_ranges)
    }
    chunks = []
    for name in names:
        if name not in lookup:
            raise ValueError(f"missing mesh {name}")
        start, stop = lookup[name]
        chunks.append(np.arange(start, stop, dtype=np.int64))
    return np.unique(np.concatenate(chunks))


def _lbs_subset(vertices, driver_indices, driver_weights, transforms, vert_ids):
    ids = np.asarray(vert_ids, dtype=np.int64)
    points = np.asarray(vertices, dtype=np.float64)[ids]
    indices = np.asarray(driver_indices, dtype=np.int64)[ids]
    weights = np.asarray(driver_weights, dtype=np.float64)[ids]
    selected = np.asarray(transforms, dtype=np.float64)[indices]
    mapped = (
        np.einsum("nsij,nj->nsi", selected[:, :, :3, :3], points)
        + selected[:, :, :3, 3]
    )
    return np.sum(mapped * weights[:, :, None], axis=1)


def _write_lbs(base, *, driver_indices, driver_weights, transforms, vert_ids):
    out = np.asarray(base, dtype=np.float64).copy()
    ids = np.asarray(vert_ids, dtype=np.int64)
    if len(ids):
        out[ids] = _lbs_subset(base, driver_indices, driver_weights, transforms, ids)
    return out


def _translate(t: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return out


def _se3_about_pivot(params6: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    values = np.asarray(params6, dtype=np.float64).reshape(6)
    rotation = Rotation.from_rotvec(values[3:]).as_matrix()
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rotation
    out[:3, 3] = values[:3] + pivot - rotation @ pivot
    return out


def _gap_open_violation(*, domains, vertices, side, reference_gaps, gap_max_m, max_open_m):
    _v, gaps = _knee_gap_contact_violation_m(
        domains=domains, vertices=vertices, side=side, gap_max_m=gap_max_m
    )
    budget = max(0.0, float(max_open_m) - FIT_GAP_SAFETY_M)
    viol = 0.0
    for compartment in ("medial", "lateral"):
        gap = float(gaps[compartment])
        if gap > gap_max_m:
            viol += gap - gap_max_m
        grow = gap - float(reference_gaps[compartment])
        if grow > budget:
            viol += grow - budget
    return float(viol), {k: float(v) for k, v in gaps.items()}


def _outside_hinge(signed: np.ndarray, *, margin_m: float = 0.0015) -> float:
    outside = np.maximum(np.asarray(signed, dtype=np.float64) + margin_m, 0.0)
    if outside.size == 0:
        return 0.0
    return float(np.mean(outside * outside) + 0.2 * np.quantile(outside, 0.95))


def _finalize_subject(
    v7: ChainRestFitSubjectV1,
    *,
    c_total: np.ndarray,
    b_final: np.ndarray,
    rest_base: np.ndarray,
    driver_indices: np.ndarray,
    driver_weights: np.ndarray,
    calibration: AnatomicalCalibrationV1,
) -> ChainRestFitSubjectV1:
    rest = _weighted_rest_correction(rest_base, driver_indices, driver_weights, c_total)
    parents = np.asarray(v7.bone_parents, dtype=np.int64)
    moved = np.asarray(v7.moved_vertex_ids, dtype=np.int64)
    verts_final = rest_base.copy()
    verts_final[moved] = rest[moved]
    delta = np.linalg.norm(rest - rest_base, axis=1)
    extra = np.nonzero(delta > 1.0e-8)[0]
    verts_final[extra] = rest[extra]
    moved_ids = np.unique(np.concatenate([moved, extra])).astype(np.int32)
    final_frames, _w, _d = _measure_frames(
        verts_final,
        calibration.domains,
        calibration.joint_domain_bases,
        partition="fit",
    )
    lower_index = {spec.name: i for i, spec in enumerate(JOINT_SPECS)}
    lower_frames = np.stack(
        [final_frames[lower_index[n]] for n in LOWER_JOINT_NAMES], axis=0
    )
    return replace(
        v7,
        vertices_final=verts_final.astype(np.float64),
        B_final=b_final.astype(np.float64),
        C_bone=c_total.astype(np.float64),
        target_local_bind=_global_to_local(b_final, parents).astype(np.float64),
        inverse_bind=np.linalg.inv(b_final).astype(np.float64),
        final_anatomical_frames=lower_frames.astype(np.float64),
        moved_vertex_ids=moved_ids,
        build_report={},
    )


def _eval_gates(
    *,
    value: ChainRestFitSubjectV1,
    v7: ChainRestFitSubjectV1,
    asset: Any,
    calibration: AnatomicalCalibrationV1,
    domains: FrozenJointMaterialDomainsV7,
    oracle_path: Path,
    operator: SourceOperatorV8,
    flex_pose: np.ndarray,
    flex_key: str,
    model: Any,
    baseline_asset: Any | None = None,
) -> dict[str, Any]:
    asset_base = baseline_asset if baseline_asset is not None else asset
    pm_v7 = build_pose_map_v1(
        v7,
        asset=asset_base,
        calibration=calibration,
        oracle_path=oracle_path,
        source_operator_digest=operator.runtime_digest(validate=False),
    )
    pm_c = build_pose_map_v1(
        value,
        asset=asset,
        calibration=calibration,
        oracle_path=oracle_path,
        source_operator_digest=operator.runtime_digest(validate=False),
    )
    flex_v7, _ = pose_whole_chain_vertices(
        v7, pm_v7, source_asset=asset_base, pose_axis_angle=flex_pose
    )
    flex_c, _ = pose_whole_chain_vertices(
        value, pm_c, source_asset=asset, pose_axis_angle=flex_pose
    )
    contact = evaluate_joint_contact_nonregress_v9(
        value,
        domains=domains,
        baseline_vertices=v7.vertices_final,
        pose_map=pm_c,
        source_asset=asset,
        flex_pose_axis_angle=flex_pose,
        baseline_flex_vertices=flex_v7,
    )
    femur_ids = np.unique(
        np.concatenate(
            [
                _mesh_ids(asset, ("Femur_L", "Patella_L")),
                _mesh_ids(asset, ("Femur_R", "Patella_R")),
            ]
        )
    )
    skin, faces = smplx_body_surface_v7(
        model, betas=np.asarray(v7.betas), pose_axis_angle=flex_pose
    )
    out_v7 = float(
        np.maximum(_signed_distance_details(flex_v7[femur_ids], skin, faces)[0], 0.0).max()
    )
    out_c = float(
        np.maximum(_signed_distance_details(flex_c[femur_ids], skin, faces)[0], 0.0).max()
    )
    contact_ok = bool(contact.get("passed"))
    outside_ok = out_c <= out_v7 + 1.0e-4
    return {
        "contact_passed": contact_ok,
        "contact_failures": contact.get("failures", []),
        "flex_femur_outside_v7_m": out_v7,
        "flex_femur_outside_cand_m": out_c,
        "outside_improved": outside_ok and out_c < out_v7 - 1.0e-5,
        "outside_nonincrease": outside_ok,
        "numeric_feasible": contact_ok and outside_ok,
        "flex_pose_id": flex_key,
        "pose_map": pm_c,
        "flex_vertices": flex_c,
        "flex_v7_vertices": flex_v7,
    }


def _optimize_c_extra(
    *,
    v7: ChainRestFitSubjectV1,
    asset: Any,
    domains: FrozenJointMaterialDomainsV7,
    model: Any,
    pose_bundle: Mapping[str, np.ndarray],
    flex_key: str,
    params0: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    c_extra_fn: Callable[[np.ndarray], np.ndarray],
    max_nfev: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    names = list(asset.source_bone_names)
    b_prefit = np.asarray(v7.B_prefit, dtype=np.float64)
    c_v7 = np.asarray(v7.C_bone, dtype=np.float64)
    rest_base = np.asarray(v7.vertices_prefit, dtype=np.float64)
    inv_prefit = np.linalg.inv(b_prefit)
    driver_indices = np.asarray(asset.driver_indices, dtype=np.int64)
    driver_weights = np.asarray(asset.driver_weights, dtype=np.float64)
    limb_ids_u = np.unique(
        np.concatenate(
            [
                _mesh_ids(asset, ("Femur_L", "Patella_L", "Tibia_L")),
                _mesh_ids(asset, ("Femur_R", "Patella_R", "Tibia_R")),
            ]
        )
    )
    femur_sample = {}
    rng = np.random.default_rng(0)
    for side, meshes in (
        ("left", ("Femur_L", "Patella_L")),
        ("right", ("Femur_R", "Patella_R")),
    ):
        ids = _mesh_ids(asset, meshes)
        femur_sample[side] = (
            np.sort(rng.choice(ids, size=400, replace=False)) if len(ids) > 400 else ids
        )

    ref_rest = {}
    for side in ("left", "right"):
        _v, gaps = _knee_gap_contact_violation_m(
            domains=domains, vertices=v7.vertices_final, side=side
        )
        ref_rest[side] = {k: float(v) for k, v in gaps.items()}

    skins = {
        label: smplx_body_surface_v7(
            model, betas=np.asarray(v7.betas), pose_axis_angle=pose
        )
        for label, pose in pose_bundle.items()
    }
    source_posed = {
        label: source_bone_posed_global(asset, pose)
        for label, pose in pose_bundle.items()
    }
    # V7 flex gaps via full pose later; during fit use rest_final posed approx.
    # Prefetch flex ref from v7.vertices_final + flex LBS on limb (full limb).
    flex_pose = pose_bundle[flex_key]
    b_v7 = np.asarray(v7.B_final, dtype=np.float64)
    g_flex_v7 = source_posed[flex_key] @ inv_prefit @ b_v7
    tf_flex_v7 = g_flex_v7 @ np.linalg.inv(b_v7)
    flex_v7_buf = _write_lbs(
        np.asarray(v7.vertices_final, dtype=np.float64),
        driver_indices=driver_indices,
        driver_weights=driver_weights,
        transforms=tf_flex_v7,
        vert_ids=limb_ids_u,
    )
    ref_flex = {}
    for side in ("left", "right"):
        _v, gaps = _knee_gap_contact_violation_m(
            domains=domains, vertices=flex_v7_buf, side=side, gap_max_m=0.025
        )
        ref_flex[side] = {k: float(v) for k, v in gaps.items()}

    def evaluate(params: np.ndarray) -> tuple[float, dict[str, Any]]:
        c_extra = c_extra_fn(params)
        c_total = c_extra @ c_v7
        b_final = c_total @ b_prefit
        inv_final = np.linalg.inv(b_final)
        rest_buf = _write_lbs(
            rest_base,
            driver_indices=driver_indices,
            driver_weights=driver_weights,
            transforms=c_total,
            vert_ids=limb_ids_u,
        )
        gap_viol = 0.0
        for side in ("left", "right"):
            viol, _ = _gap_open_violation(
                domains=domains,
                vertices=rest_buf,
                side=side,
                reference_gaps=ref_rest[side],
                gap_max_m=0.003,
                max_open_m=MAX_GAP_REGRESSION_M,
            )
            gap_viol += viol
        g_flex = source_posed[flex_key] @ inv_prefit @ b_final
        tf_flex = g_flex @ inv_final
        flex_buf = _write_lbs(
            rest_buf,
            driver_indices=driver_indices,
            driver_weights=driver_weights,
            transforms=tf_flex,
            vert_ids=limb_ids_u,
        )
        for side in ("left", "right"):
            viol, _ = _gap_open_violation(
                domains=domains,
                vertices=flex_buf,
                side=side,
                reference_gaps=ref_flex[side],
                gap_max_m=MAX_FLEX_CONTACT_ABS_M,
                max_open_m=MAX_FLEX_GAP_REGRESSION_M,
            )
            gap_viol += viol
        feasible = gap_viol <= 1.0e-9
        chunks = []
        scores = []
        for label, (skin, faces) in skins.items():
            g_tgt = source_posed[label] @ inv_prefit @ b_final
            transforms = g_tgt @ inv_final
            for side, ids in femur_sample.items():
                posed = _lbs_subset(
                    rest_buf, driver_indices, driver_weights, transforms, ids
                )
                signed = _signed_distance_details(posed, skin, faces)[0]
                excess = np.maximum(signed + 0.0015, 0.0)
                chunks.append(excess)
                scores.append(_outside_hinge(signed))
        mean_out = float(np.mean(scores)) if scores else 0.0
        reg = float(50.0 * np.dot(params, params))
        if not feasible:
            total = INFEASIBLE * (1.0 + 1.0e3 * gap_viol) + reg
        else:
            total = 1.0e6 * mean_out + reg
        return total, {
            "feasible": feasible,
            "gap_violation_m": float(gap_viol),
            "mean_outside_score": mean_out,
            "outside_chunks": chunks,
            "c_total": c_total,
            "b_final": b_final,
        }

    x0 = np.asarray(params0, dtype=np.float64)
    init_score, init_info = evaluate(x0)
    n_out = sum(int(c.size) for c in init_info["outside_chunks"])

    def residual(params: np.ndarray) -> np.ndarray:
        _s, info = evaluate(params)
        if not info["feasible"]:
            base = np.full(max(n_out, 1), INFEASIBLE, dtype=np.float64)
            base[0] += 1.0e4 * float(info["gap_violation_m"])
            return np.concatenate([base, 0.25 * params / np.maximum(np.abs(hi), 1e-6)])
        chunks = [1000.0 * c for c in info["outside_chunks"]]
        chunks.append(0.25 * params / np.maximum(np.abs(hi), 1e-6))
        return np.concatenate(chunks)

    solved = least_squares(
        residual,
        x0,
        bounds=(lo, hi),
        max_nfev=int(max_nfev),
        xtol=1e-7,
        ftol=1e-7,
        gtol=1e-7,
        verbose=0,
    )
    best = np.asarray(solved.x, dtype=np.float64)
    best_score, best_info = evaluate(best)
    if (not best_info["feasible"]) or best_score > init_score - 1e-8:
        best = x0
        best_score, best_info = evaluate(x0)
    return best, {
        "nfev": int(solved.nfev),
        "init_score": float(init_score),
        "final_score": float(best_score),
        "final_feasible": bool(best_info["feasible"]),
        "params": best.tolist(),
        "c_total": best_info["c_total"],
        "b_final": best_info["b_final"],
    }


def build_inward_shared_t(
    v7, *, asset, domains, model, pose_bundle, flex_key, calibration, max_nfev=36
):
    """Shared translation only on femur cluster + shank (no rotation)."""

    names = list(asset.source_bone_names)

    def c_extra_fn(params: np.ndarray) -> np.ndarray:
        values = np.asarray(params, dtype=np.float64).reshape(6)
        c_extra = np.tile(np.eye(4, dtype=np.float64), (235, 1, 1))
        for side_i, side in enumerate(("left", "right")):
            suf = "L" if side == "left" else "R"
            t = values[side_i * 3 : side_i * 3 + 3]
            shared = _translate(t)
            for bone in (
                f"Femur_Rot_{suf}",
                f"Knee_Rotate_{suf}",
                f"Patella_Rotate_{suf}",
                f"Tibia_Bone_{suf}",
                f"Tibia_Twist_{suf}",
            ):
                c_extra[names.index(bone)] = shared
        return c_extra

    lo = np.full(6, -TRANSLATION_BOUND_M, dtype=np.float64)
    hi = -lo
    best, info = _optimize_c_extra(
        v7=v7,
        asset=asset,
        domains=domains,
        model=model,
        pose_bundle=pose_bundle,
        flex_key=flex_key,
        params0=np.zeros(6),
        lo=lo,
        hi=hi,
        c_extra_fn=c_extra_fn,
        max_nfev=max_nfev,
    )
    value = _finalize_subject(
        v7,
        c_total=info["c_total"],
        b_final=info["b_final"],
        rest_base=np.asarray(v7.vertices_prefit, dtype=np.float64),
        driver_indices=np.asarray(asset.driver_indices, dtype=np.int64),
        driver_weights=np.asarray(asset.driver_weights, dtype=np.float64),
        calibration=calibration,
    )
    return value, {"direction": "inward_shared_t", "fit": {k: v for k, v in info.items() if k not in {"c_total", "b_final"}}, "params": best.tolist()}


def build_patella_only(
    v7, *, asset, domains, model, pose_bundle, flex_key, calibration, max_nfev=36
):
    """Patella_Rotate SE(3) only — hinge femur/tibia untouched."""

    names = list(asset.source_bone_names)
    rest_base = np.asarray(v7.vertices_prefit, dtype=np.float64)
    pivots = {
        "left": rest_base[_mesh_ids(asset, ("Patella_L",))].mean(axis=0),
        "right": rest_base[_mesh_ids(asset, ("Patella_R",))].mean(axis=0),
    }

    def c_extra_fn(params: np.ndarray) -> np.ndarray:
        values = np.asarray(params, dtype=np.float64).reshape(12)
        c_extra = np.tile(np.eye(4, dtype=np.float64), (235, 1, 1))
        for side_i, side in enumerate(("left", "right")):
            suf = "L" if side == "left" else "R"
            p = values[side_i * 6 : side_i * 6 + 6]
            c_extra[names.index(f"Patella_Rotate_{suf}")] = _se3_about_pivot(
                p, pivots[side]
            )
        return c_extra

    lo = np.asarray(
        ([-TRANSLATION_BOUND_M] * 3 + [-ROTATION_BOUND_RAD] * 3) * 2, dtype=np.float64
    )
    hi = -lo
    best, info = _optimize_c_extra(
        v7=v7,
        asset=asset,
        domains=domains,
        model=model,
        pose_bundle=pose_bundle,
        flex_key=flex_key,
        params0=np.zeros(12),
        lo=lo,
        hi=hi,
        c_extra_fn=c_extra_fn,
        max_nfev=max_nfev,
    )
    value = _finalize_subject(
        v7,
        c_total=info["c_total"],
        b_final=info["b_final"],
        rest_base=rest_base,
        driver_indices=np.asarray(asset.driver_indices, dtype=np.int64),
        driver_weights=np.asarray(asset.driver_weights, dtype=np.float64),
        calibration=calibration,
    )
    return value, {"direction": "patella_only", "fit": {k: v for k, v in info.items() if k not in {"c_total", "b_final"}}, "params": best.tolist()}


def build_delta_j_centerline(
    v7, *, asset, domains, model, pose_bundle, flex_key, calibration, max_nfev=40
):
    """Bounded shared translation toward skin centerline knee (SKEL-J ΔJ proxy).

    Uses V7 centerline_points knee endpoint as target joint center; pulls current
    condyle+plateau midpoint toward that target with hard gap. No axial scale.
    """

    names = list(asset.source_bone_names)
    rest_final = np.asarray(v7.vertices_final, dtype=np.float64)
    # centerline_points: [2 sides, 2 segments, N, 3] — femur segment distal ≈ knee
    cl = np.asarray(v7.centerline_points, dtype=np.float64)
    targets = {}
    currents = {}
    for side_i, side in enumerate(("left", "right")):
        # femur centerline last sample ≈ knee station
        knee_tgt = cl[side_i, 0, -1]
        condyle_keys = []
        for compartment in ("medial", "lateral"):
            for key in (
                f"{side}/femoral_condyle_{compartment}.fit",
                f"{side}/femoral_condyle_{compartment}",
            ):
                if key in domains.domains:
                    condyle_keys.append(np.asarray(domains.require(key), dtype=np.int64))
                    break
        if not condyle_keys:
            raise RuntimeError(f"missing condyle domains for {side}")
        condyle_ids = np.unique(np.concatenate(condyle_keys))
        knee_cur = rest_final[condyle_ids].mean(axis=0)
        targets[side] = knee_tgt
        currents[side] = knee_cur
    # Seed: half-step toward target, clipped.
    t0 = []
    for side in ("left", "right"):
        delta = targets[side] - currents[side]
        step = 0.5 * delta
        nrm = float(np.linalg.norm(step))
        if nrm > DELTA_J_BOUND_M:
            step = step * (DELTA_J_BOUND_M / nrm)
        t0.extend(step.tolist())
    t0 = np.asarray(t0, dtype=np.float64)

    def c_extra_fn(params: np.ndarray) -> np.ndarray:
        values = np.asarray(params, dtype=np.float64).reshape(6)
        c_extra = np.tile(np.eye(4, dtype=np.float64), (235, 1, 1))
        for side_i, side in enumerate(("left", "right")):
            suf = "L" if side == "left" else "R"
            t = values[side_i * 3 : side_i * 3 + 3]
            shared = _translate(t)
            for bone in (
                f"Femur_Rot_{suf}",
                f"Knee_Rotate_{suf}",
                f"Patella_Rotate_{suf}",
                f"Tibia_Bone_{suf}",
                f"Tibia_Twist_{suf}",
            ):
                c_extra[names.index(bone)] = shared
        return c_extra

    lo = np.full(6, -DELTA_J_BOUND_M, dtype=np.float64)
    hi = -lo
    best, info = _optimize_c_extra(
        v7=v7,
        asset=asset,
        domains=domains,
        model=model,
        pose_bundle=pose_bundle,
        flex_key=flex_key,
        params0=t0,
        lo=lo,
        hi=hi,
        c_extra_fn=c_extra_fn,
        max_nfev=max_nfev,
    )
    value = _finalize_subject(
        v7,
        c_total=info["c_total"],
        b_final=info["b_final"],
        rest_base=np.asarray(v7.vertices_prefit, dtype=np.float64),
        driver_indices=np.asarray(asset.driver_indices, dtype=np.int64),
        driver_weights=np.asarray(asset.driver_weights, dtype=np.float64),
        calibration=calibration,
    )
    return value, {
        "direction": "delta_j_centerline",
        "fit": {k: v for k, v in info.items() if k not in {"c_total", "b_final"}},
        "params": best.tolist(),
        "seed_params": t0.tolist(),
        "targets_m": {s: targets[s].tolist() for s in targets},
        "currents_m": {s: currents[s].tolist() for s in currents},
    }


def save_tournament_subject(path: Path, value: ChainRestFitSubjectV1, *, report: Mapping) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        path / "subject.npz",
        betas=np.asarray(value.betas),
        vertices_prefit=np.asarray(value.vertices_prefit),
        vertices_final=np.asarray(value.vertices_final),
        faces=np.asarray(value.faces),
        bone_parents=np.asarray(value.bone_parents),
        B_prefit=np.asarray(value.B_prefit),
        B_final=np.asarray(value.B_final),
        C_bone=np.asarray(value.C_bone),
        target_local_bind=np.asarray(value.target_local_bind),
        inverse_bind=np.asarray(value.inverse_bind),
        prefit_anatomical_frames=np.asarray(value.prefit_anatomical_frames),
        final_anatomical_frames=np.asarray(value.final_anatomical_frames),
        smplx_joints_tpose=np.asarray(value.smplx_joints_tpose),
        station_frame_translation=np.asarray(value.station_frame_translation),
        centerline_points=np.asarray(value.centerline_points),
        mesh_policy=np.asarray(value.mesh_policy),
        moved_vertex_ids=np.asarray(value.moved_vertex_ids),
    )
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_kind": TOURNAMENT_KIND,
                "subject_label": value.subject_label,
                "publishable": False,
                "npz": "subject.npz",
                "report": dict(report),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def load_tournament_subject(path: Path, *, template: ChainRestFitSubjectV1) -> ChainRestFitSubjectV1:
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    with np.load(path / manifest["npz"]) as data:
        return replace(
            template,
            betas=np.asarray(data["betas"], dtype=np.float64),
            vertices_prefit=np.asarray(data["vertices_prefit"], dtype=np.float64),
            vertices_final=np.asarray(data["vertices_final"], dtype=np.float64),
            faces=np.asarray(data["faces"], dtype=np.int32),
            bone_parents=np.asarray(data["bone_parents"], dtype=np.int64),
            B_prefit=np.asarray(data["B_prefit"], dtype=np.float64),
            B_final=np.asarray(data["B_final"], dtype=np.float64),
            C_bone=np.asarray(data["C_bone"], dtype=np.float64),
            target_local_bind=np.asarray(data["target_local_bind"], dtype=np.float64),
            inverse_bind=np.asarray(data["inverse_bind"], dtype=np.float64),
            prefit_anatomical_frames=np.asarray(
                data["prefit_anatomical_frames"], dtype=np.float64
            ),
            final_anatomical_frames=np.asarray(
                data["final_anatomical_frames"], dtype=np.float64
            ),
            smplx_joints_tpose=np.asarray(data["smplx_joints_tpose"], dtype=np.float64),
            station_frame_translation=np.asarray(
                data["station_frame_translation"], dtype=np.float64
            ),
            centerline_points=np.asarray(data["centerline_points"], dtype=np.float64),
            mesh_policy=np.asarray(data["mesh_policy"]),
            moved_vertex_ids=np.asarray(data["moved_vertex_ids"], dtype=np.int32),
            build_report=manifest.get("report") or {},
            subject_label=str(manifest.get("subject_label", template.subject_label)),
        )


__all__ = [
    "TOURNAMENT_KIND",
    "build_delta_j_centerline",
    "build_inward_shared_t",
    "build_patella_only",
    "_eval_gates",
    "load_tournament_subject",
    "save_tournament_subject",
]
