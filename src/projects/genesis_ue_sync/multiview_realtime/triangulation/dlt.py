"""Robust per-joint multiview triangulation.

The old EasyMocap iterative path used one global ``min_view``.  That is a poor
fit for four cameras: a bad third observation can cause two mutually
consistent foot observations to be discarded.  This module instead evaluates
all 2--N view hypotheses per joint and records the actual inlier decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

# Only true BODY25 endpoints may fall back to two views.  Knees, elbows and
# central joints retain the three-view accuracy contract; standalone 21-joint
# hand schemas are endpoints by definition and are handled separately.
_TWO_VIEW_BODY25 = frozenset((4, 7, 11, 14, 19, 20, 21, 22, 23, 24))


@dataclass(frozen=True)
class TriangulationConfig:
    confidence_threshold: float = 0.28
    min_conf: float = 0.1
    min_view: int = 2
    min_joints: int = 3
    dist_max_px: float = 25.0
    dist_vel: float = 0.05
    thres_outlier_view: float = 0.4
    thres_outlier_joint: float = 0.4
    adaptive_views: bool = True
    core_min_view: int = 3
    two_view_max_reproj_px: float = 12.0
    two_view_min_ray_angle_deg: float = 3.0
    low_view_confidence_scale: float = 0.5
    simcc_probability_weight: float = 0.05
    simcc_uncertainty_weight: float = 0.05
    detector_confidence_weight: float = 0.05
    # DLT fits a two-view seed almost exactly by construction.  Treat very
    # small all-view cost differences as equivalent so that a consistent
    # third/fourth observation can still win the final view-count tie-break.
    hypothesis_cost_tie: float = 1e-2
    simcc_pair_hypotheses: int = 5

    @classmethod
    def from_legacy_dict(cls,
                         tri: dict[str, Any] | None) -> "TriangulationConfig":
        tri = dict(tri or {})
        dist = tri.get("dist_max_px", tri.get("max_reprojection_error_px",
                                              25.0))
        return cls(
            confidence_threshold=float(tri.get("confidence_threshold", 0.28)),
            min_conf=float(tri.get("min_conf", 0.1)),
            # Legacy min_view remains a compatibility fallback only.  Adaptive
            # fusion below decides view count per joint.
            min_view=max(
                2, int(tri.get("min_view", tri.get("min_views_per_joint",
                                                   2)))),
            min_joints=max(1, int(tri.get("min_joints", 3))),
            dist_max_px=float(dist),
            dist_vel=float(tri.get("dist_vel", 0.05)),
            thres_outlier_view=float(
                tri.get("thres_outlier_view", tri.get("view_outlier_ratio",
                                                      0.4))),
            thres_outlier_joint=float(tri.get("thres_outlier_joint", 0.4)),
            adaptive_views=bool(tri.get("adaptive_views", True)),
            core_min_view=max(2, int(tri.get("core_min_view", 3))),
            two_view_max_reproj_px=float(
                tri.get("two_view_max_reproj_px", min(float(dist), 12.0))),
            two_view_min_ray_angle_deg=float(
                tri.get("two_view_min_ray_angle_deg", 3.0)),
            low_view_confidence_scale=float(
                tri.get("low_view_confidence_scale", 0.5)),
            simcc_probability_weight=float(
                tri.get("simcc_probability_weight", 0.05)),
            simcc_uncertainty_weight=float(
                tri.get("simcc_uncertainty_weight", 0.05)),
            detector_confidence_weight=float(
                tri.get("detector_confidence_weight", 0.05)),
            hypothesis_cost_tie=float(tri.get("hypothesis_cost_tie", 1e-2)),
            simcc_pair_hypotheses=max(1,
                                      int(tri.get("simcc_pair_hypotheses",
                                                  5))),
        )


def _triangulate_linear(xy: np.ndarray, p: np.ndarray,
                        weights: np.ndarray) -> np.ndarray | None:
    """Weighted homogeneous DLT for one joint."""
    rows: list[np.ndarray] = []
    for (u, v), proj, weight in zip(xy, p, weights):
        if not np.isfinite(u) or not np.isfinite(v) or weight <= 0:
            continue
        w = float(np.sqrt(weight))
        rows.extend((w * (u * proj[2] - proj[0]), w * (v * proj[2] - proj[1])))
    if len(rows) < 4:
        return None
    _u, _s, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64),
                               full_matrices=False)
    h = vh[-1]
    if not np.isfinite(h[3]) or abs(float(h[3])) < 1e-10:
        return None
    xyz = h[:3] / h[3]
    return xyz if np.all(np.isfinite(xyz)) else None


def _project(xyz: np.ndarray, p: np.ndarray) -> np.ndarray:
    h = np.concatenate((np.asarray(xyz, dtype=np.float64), np.ones(1)))
    q = np.asarray(p, dtype=np.float64) @ h
    if q.ndim == 1:
        q = q.reshape(1, 3)
    valid = np.isfinite(q[:, 2]) & (q[:, 2] > 1e-9)
    out = np.full((len(q), 2), np.nan, dtype=np.float64)
    out[valid] = q[valid, :2] / q[valid, 2:3]
    return out


def _camera_center(p: np.ndarray) -> np.ndarray | None:
    m = np.asarray(p, dtype=np.float64)[:, :3]
    try:
        c = -np.linalg.solve(m, np.asarray(p, dtype=np.float64)[:, 3])
    except np.linalg.LinAlgError:
        return None
    return c if np.all(np.isfinite(c)) else None


def _minimum_ray_angle_deg(xyz: np.ndarray, p: np.ndarray) -> float:
    centers = [_camera_center(pi) for pi in p]
    vectors = []
    for center in centers:
        if center is None:
            continue
        direction = np.asarray(xyz, dtype=np.float64) - center
        norm = float(np.linalg.norm(direction))
        if norm > 1e-9:
            vectors.append(direction / norm)
    if len(vectors) < 2:
        return 0.0
    angles = []
    for a, b in combinations(vectors, 2):
        angles.append(
            float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))))
    return min(angles) if angles else 0.0


def _meta_array(
    observation_meta: dict[str, Any] | None,
    names: tuple[str, ...],
) -> np.ndarray | None:
    if not isinstance(observation_meta, dict):
        return None
    for name in names:
        value = observation_meta.get(name)
        if value is not None:
            return np.asarray(value)
    return None


def _joint_candidates(
    observations: np.ndarray,
    observation_meta: dict[str, Any] | None,
    joint_index: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Return per-view ``(xy, probability, variance_px2)`` candidates.

    The optional public metadata contract uses arrays shaped ``(V,J,K,2)``,
    ``(V,J,K)`` and ``(V,J,2)``.  Aliases are accepted so saved detector
    diagnostics can be passed without a conversion copy.  Missing metadata
    falls back to the decoded DWPose coordinate with probability one.
    """
    candidate_xy = _meta_array(observation_meta,
                               ("candidate_xy", "simcc_candidate_xy"))
    candidate_prob = _meta_array(
        observation_meta,
        ("candidate_probabilities", "candidate_probability", "candidate_prob"))
    variance = _meta_array(observation_meta,
                           ("variance_px2", "simcc_variance_px2"))
    std_xy = _meta_array(observation_meta, ("std_xy_px", "simcc_std_xy_px"))
    result_xy: list[np.ndarray] = []
    result_prob: list[np.ndarray] = []
    result_var: list[np.ndarray] = []
    for view in range(len(observations)):
        xy = np.asarray(observations[view, :2], dtype=np.float64).reshape(1, 2)
        prob = np.ones((1, ), dtype=np.float64)
        var = np.zeros((1, 2), dtype=np.float64)
        try:
            values = np.asarray(candidate_xy[view, joint_index],
                                dtype=np.float64).reshape(-1, 2)
            valid = np.isfinite(values).all(axis=1)
            if np.any(valid):
                xy = values[valid]
                if candidate_prob is not None:
                    p = np.asarray(candidate_prob[view, joint_index],
                                   dtype=np.float64).reshape(-1)
                    if len(p) == len(values):
                        prob = np.clip(p[valid], 1e-12, 1.0)
                    else:
                        prob = np.ones((len(xy), ), dtype=np.float64)
                else:
                    prob = np.ones((len(xy), ), dtype=np.float64)
        except (IndexError, TypeError, ValueError):
            pass
        try:
            if variance is not None:
                vv = np.asarray(variance[view, joint_index], dtype=np.float64)
            elif std_xy is not None:
                vv = np.square(
                    np.asarray(std_xy[view, joint_index], dtype=np.float64))
            else:
                vv = np.zeros((2, ), dtype=np.float64)
            if vv.ndim == 1:
                var = np.broadcast_to(vv.reshape(1, 2), (len(xy), 2)).copy()
            else:
                vv = vv.reshape(-1, 2)
                var = vv[:len(xy)].copy() if len(vv) >= len(xy) else np.zeros(
                    (len(xy), 2))
            var[~np.isfinite(var)] = 0.0
            var = np.maximum(var, 0.0)
        except (IndexError, TypeError, ValueError):
            var = np.zeros((len(xy), 2), dtype=np.float64)
        result_xy.append(xy)
        result_prob.append(prob)
        result_var.append(var)
    return result_xy, result_prob, result_var


def _observation_weight(confidence: float, probability: float,
                        variance_px2: np.ndarray) -> float:
    # High SimCC variance is a weak observation, but is never allowed to turn
    # into a zero weight solely because the distribution is broad.
    variance_scale = 1.0 + float(np.mean(np.maximum(variance_px2, 0.0))) / 16.0
    return max(float(confidence), 1e-6) * max(float(probability),
                                              1e-6) / variance_scale


def _geman_mcclure_unit(value: np.ndarray) -> np.ndarray:
    """Bounded robust cost so one occluded camera cannot pull a clean pair."""
    squared = np.square(np.asarray(value, dtype=np.float64))
    return squared / (1.0 + squared)


def _hypothesis_cost(
    errors: np.ndarray,
    confidences: np.ndarray,
    probabilities: np.ndarray,
    variances_px2: np.ndarray,
    config: TriangulationConfig,
) -> float:
    scale = max(float(config.two_view_max_reproj_px), 1.0)
    weights = np.maximum(confidences, 1e-6)
    # Every hypothesis is scored against the same set of observed cameras,
    # including cameras it rejects.  Otherwise a two-view DLT solution has
    # nearly zero training residual by construction and would beat a genuinely
    # consistent 3/4-view reconstruction.  Truncation at the gross-outlier
    # gate prevents one occluded camera from dominating the robust objective.
    clipped_errors = np.minimum(np.asarray(errors, dtype=np.float64),
                                float(config.dist_max_px))
    residual = float(
        np.average(_geman_mcclure_unit(clipped_errors / scale),
                   weights=weights))
    probability_cost = float(
        np.average(-np.log(np.clip(probabilities, 1e-12, 1.0)),
                   weights=weights))
    uncertainty = np.sqrt(np.maximum(np.mean(variances_px2, axis=1), 0.0))
    uncertainty_cost = float(
        np.average(np.log1p(uncertainty / scale), weights=weights))
    confidence_cost = float(np.mean(1.0 - np.clip(confidences, 0.0, 1.0)))
    return (residual +
            float(config.simcc_probability_weight) * probability_cost +
            float(config.simcc_uncertainty_weight) * uncertainty_cost +
            float(config.detector_confidence_weight) * confidence_cost)


def _hypothesis_is_better(
    new: dict[str, Any],
    best: dict[str, Any] | None,
    config: TriangulationConfig,
) -> bool:
    if best is None:
        return True
    delta = float(new["robust_cost"]) - float(best["robust_cost"])
    if abs(delta) > float(config.hypothesis_cost_tie):
        return delta < 0.0
    # View count is consulted only after robust all-observed-view costs are
    # indistinguishable.  This keeps a visibly worse third view from winning,
    # while allowing a genuinely consistent 3/4-view solution to be reported
    # as high-confidence instead of collapsing to an exact-fit DLT pair.
    new_tie = (
        len(new["used_views"]),
        float(new["min_ray_angle_deg"]),
        float(np.mean(new["candidate_probabilities_used"])),
        float(np.mean(new["confidences_used"])),
    )
    best_tie = (
        len(best["used_views"]),
        float(best["min_ray_angle_deg"]),
        float(np.mean(best["candidate_probabilities_used"])),
        float(np.mean(best["confidences_used"])),
    )
    return new_tie > best_tie


def _joint_solution(
    observations: np.ndarray,
    p_all: np.ndarray,
    config: TriangulationConfig,
    *,
    joint_index: int,
    is_body25: bool,
    observation_meta: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Find the lowest-cost geometrically valid hypothesis for one joint."""
    conf_gate = float(max(config.confidence_threshold, config.min_conf))
    observed = np.flatnonzero(
        np.isfinite(observations[:, :2]).all(axis=1)
        & (observations[:, 2] >= conf_gate))
    empty = np.zeros((4, ), dtype=np.float32)
    mask = np.zeros((len(observations), ), dtype=bool)
    base = {
        "observed_views": observed.astype(int).tolist(),
        "used_views": [],
        "rejected_views": [],
        "reprojection_error_px": None,
        "max_reprojection_error_px": None,
        "min_ray_angle_deg": None,
        "status": "missing",
        "geometry_ok": False,
        "selected_hypothesis_id": None,
        "selected_candidate_ranks": [-1] * len(observations),
        "selected_observations_xy": [None] * len(observations),
        "selected_reprojection_errors_px": [None] * len(observations),
        "selected_candidate_probabilities": [None] * len(observations),
        "selected_simcc_variance_px2": [None] * len(observations),
        "rejected_view_reasons": {},
        "robust_cost": None,
        "candidate_hypotheses": [],
    }
    is_core = bool(is_body25 and joint_index not in _TWO_VIEW_BODY25)
    required = int(config.core_min_view if is_core else config.min_view)
    if len(observed) < 2:
        base["rejected_views"] = observed.astype(int).tolist()
        return empty, mask, base

    candidates_xy, candidates_prob, candidates_var = _joint_candidates(
        observations, observation_meta, joint_index)
    hypotheses: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    # Candidate pairs are sufficient seeds.  Each seed chooses the nearest
    # SimCC mode in every other camera, then is re-fit on only those inliers.
    # This avoids the K**V explosion while still allowing a secondary peak to
    # rescue an occluded view.
    for seed_views_tuple in combinations(observed.tolist(), 2):
        seed_views = np.asarray(seed_views_tuple, dtype=np.int64)
        pair_probability = (candidates_prob[seed_views[0]][:, None] *
                            candidates_prob[seed_views[1]][None, :])
        flat_pair_probability = pair_probability.reshape(-1)
        pair_count = min(int(config.simcc_pair_hypotheses),
                         len(flat_pair_probability))
        seed_pairs = np.argpartition(flat_pair_probability,
                                     -pair_count)[-pair_count:]
        seed_pairs = seed_pairs[np.argsort(
            flat_pair_probability[seed_pairs])[::-1]]
        for flat_pair_index in seed_pairs:
            seed_rank0, seed_rank1 = np.unravel_index(int(flat_pair_index),
                                                      pair_probability.shape)
            hypothesis_id = len(hypotheses)
            ranks = np.full((len(observations), ), -1, dtype=np.int64)
            ranks[seed_views] = (seed_rank0, seed_rank1)
            seed_xy = np.asarray([
                candidates_xy[seed_views[0]][seed_rank0],
                candidates_xy[seed_views[1]][seed_rank1]
            ])
            seed_prob = np.asarray([
                candidates_prob[seed_views[0]][seed_rank0],
                candidates_prob[seed_views[1]][seed_rank1]
            ])
            seed_var = np.asarray([
                candidates_var[seed_views[0]][seed_rank0],
                candidates_var[seed_views[1]][seed_rank1]
            ])
            seed_weights = np.asarray([
                _observation_weight(observations[v, 2], p, var)
                for v, p, var in zip(seed_views, seed_prob, seed_var)
            ])
            xyz = _triangulate_linear(seed_xy, p_all[seed_views], seed_weights)
            diagnostic: dict[str, Any] = {
                "hypothesis_id": hypothesis_id,
                "xyz": None,
                "confidence": 0.0,
                "used_views": [],
                "inlier_mask": [False] * len(observations),
                "candidate_ranks": ranks.astype(int).tolist(),
                "candidate_probabilities": [None] * len(observations),
                "simcc_variance_px2": [None] * len(observations),
                "observation_xy": [None] * len(observations),
                "reprojection_errors_px": [None] * len(observations),
                "mean_reprojection_error_px": None,
                "max_reprojection_error_px": None,
                "robust_cost": None,
                "min_ray_angle_deg": None,
                "geometry_ok": False,
                "rejection_reason": None,
                "selected": False,
            }
            if xyz is None:
                diagnostic["rejection_reason"] = "dlt_failed"
                hypotheses.append(diagnostic)
                continue

            # Select the most geometrically compatible SimCC mode in each
            # observed view.  The seed ranks stay fixed for determinism.
            repro_all = _project(xyz, p_all[observed])
            for local, view in enumerate(observed):
                if ranks[view] >= 0:
                    continue
                distances = np.linalg.norm(candidates_xy[view] -
                                           repro_all[local],
                                           axis=1)
                quality = distances + float(
                    config.simcc_probability_weight) * float(
                        config.two_view_max_reproj_px) * (-np.log(
                            np.clip(candidates_prob[view], 1e-12, 1.0)))
                ranks[view] = int(np.argmin(quality))
            selected_xy = np.asarray(
                [candidates_xy[v][ranks[v]] for v in observed])
            errors_all = np.linalg.norm(repro_all - selected_xy, axis=1)
            gross_inliers = observed[np.isfinite(errors_all) &
                                     (errors_all <= float(config.dist_max_px))]
            if len(gross_inliers) < 2:
                diagnostic["candidate_ranks"] = ranks.astype(int).tolist()
                diagnostic["rejection_reason"] = "gross_reprojection_outlier"
                hypotheses.append(diagnostic)
                continue
            extras = [
                int(v) for v in gross_inliers
                if int(v) not in set(seed_views.tolist())
            ]
            # Enumerate explicit 2/3/4-view hypotheses.  The old code only
            # kept the maximal inlier set, which meant a merely plausible
            # bad third view could still destroy a clean two-view foot.
            inlier_subsets: list[np.ndarray] = []
            for count in range(len(extras) + 1):
                for extra_subset in combinations(extras, count):
                    inlier_subsets.append(
                        np.asarray(sorted(seed_views.tolist() +
                                          list(extra_subset)),
                                   dtype=np.int64))
            for inliers in inlier_subsets:
                candidate_diagnostic = dict(diagnostic)
                candidate_diagnostic["hypothesis_id"] = len(hypotheses)
                inlier_xy = np.asarray(
                    [candidates_xy[v][ranks[v]] for v in inliers])
                probabilities = np.asarray(
                    [candidates_prob[v][ranks[v]] for v in inliers])
                variances = np.asarray(
                    [candidates_var[v][ranks[v]] for v in inliers])
                confidences = observations[inliers, 2]
                weights = np.asarray([
                    _observation_weight(c, p, var)
                    for c, p, var in zip(confidences, probabilities, variances)
                ])
                xyz_refit = _triangulate_linear(inlier_xy, p_all[inliers],
                                                weights)
                if xyz_refit is None:
                    candidate_diagnostic["candidate_ranks"] = ranks.astype(
                        int).tolist()
                    candidate_diagnostic["rejection_reason"] = "refit_failed"
                    hypotheses.append(candidate_diagnostic)
                    continue
                repro_refit = _project(xyz_refit, p_all[inliers])
                err_refit = np.linalg.norm(repro_refit - inlier_xy, axis=1)
                mean_err = float(np.mean(err_refit))
                max_err = float(np.max(err_refit))
                angle = _minimum_ray_angle_deg(xyz_refit, p_all[inliers])
                # Compare every 2/3/4-view subset on the same observed-view
                # support.  Geometry gates below still use only the views
                # claimed as inliers.
                repro_evaluation = _project(xyz_refit, p_all[observed])
                evaluation_xy = np.asarray(
                    [candidates_xy[v][ranks[v]] for v in observed])
                evaluation_errors = np.linalg.norm(repro_evaluation -
                                                   evaluation_xy,
                                                   axis=1)
                evaluation_probabilities = np.asarray(
                    [candidates_prob[v][ranks[v]] for v in observed])
                evaluation_variances = np.asarray(
                    [candidates_var[v][ranks[v]] for v in observed])
                evaluation_confidences = observations[observed, 2]
                cost = _hypothesis_cost(
                    evaluation_errors,
                    evaluation_confidences,
                    evaluation_probabilities,
                    evaluation_variances,
                    config,
                )
                inlier_mask = np.zeros((len(observations), ), dtype=bool)
                inlier_mask[inliers] = True
                probabilities_all: list[float
                                        | None] = [None] * len(observations)
                variances_all: list[list[float]
                                    | None] = [None] * len(observations)
                errors_out: list[float | None] = [None] * len(observations)
                observations_out: list[list[float]
                                       | None] = [None] * len(observations)
                for view, probability, var, error in zip(
                        observed,
                        evaluation_probabilities,
                        evaluation_variances,
                        evaluation_errors,
                ):
                    probabilities_all[int(view)] = float(probability)
                    variances_all[int(view)] = np.asarray(
                        var, dtype=float).tolist()
                    errors_out[int(view)] = float(error)
                    observations_out[int(view)] = np.asarray(
                        candidates_xy[view][ranks[view]],
                        dtype=float).tolist()
                confidence = float(np.mean(confidences))
                if len(inliers) == 2:
                    confidence *= float(config.low_view_confidence_scale)
                candidate_diagnostic.update({
                    "xyz":
                    np.asarray(xyz_refit, dtype=float).tolist(),
                    "confidence":
                    confidence,
                    "used_views":
                    inliers.astype(int).tolist(),
                    "inlier_mask":
                    inlier_mask.tolist(),
                    "candidate_ranks":
                    ranks.astype(int).tolist(),
                    "candidate_probabilities":
                    probabilities_all,
                    "simcc_variance_px2":
                    variances_all,
                    "observation_xy":
                    observations_out,
                    "reprojection_errors_px":
                    errors_out,
                    "mean_reprojection_error_px":
                    mean_err,
                    "max_reprojection_error_px":
                    max_err,
                    "all_observed_mean_reprojection_error_px":
                    float(np.mean(evaluation_errors)),
                    "robust_cost":
                    float(cost),
                    "min_ray_angle_deg":
                    float(angle),
                    # Private comparison-only values are removed below.
                    "candidate_probabilities_used":
                    probabilities,
                    "confidences_used":
                    confidences,
                })
                if len(inliers) < required:
                    candidate_diagnostic[
                        "rejection_reason"] = "insufficient_views"
                elif len(inliers) == 2 and angle < float(
                        config.two_view_min_ray_angle_deg):
                    candidate_diagnostic[
                        "rejection_reason"] = "ray_angle_too_small"
                elif (len(inliers) == 2 or not is_core) and max_err > float(
                        config.two_view_max_reproj_px):
                    # The 12 px precision gate applies to every distal
                    # hypothesis, including hypotheses with 3+ cameras.
                    candidate_diagnostic[
                        "rejection_reason"] = "precision_reprojection_outlier"
                else:
                    candidate_diagnostic["geometry_ok"] = True
                    candidate_diagnostic["rejection_reason"] = None
                    if _hypothesis_is_better(candidate_diagnostic, best,
                                             config):
                        best = candidate_diagnostic
                hypotheses.append(candidate_diagnostic)

    for hypothesis in hypotheses:
        hypothesis.pop("candidate_probabilities_used", None)
        hypothesis.pop("confidences_used", None)
    base["candidate_hypotheses"] = hypotheses

    if best is None:
        base["rejected_views"] = observed.astype(int).tolist()
        if len(observed) >= 2:
            base["status"] = "missing_no_valid_hypothesis"
        return empty, mask, base
    selected_id = int(best["hypothesis_id"])
    hypotheses[selected_id]["selected"] = True
    for hypothesis in hypotheses:
        if hypothesis["geometry_ok"] and not hypothesis[
                "selected"] and hypothesis["rejection_reason"] is None:
            hypothesis["rejection_reason"] = "higher_robust_cost"
    xyz = np.asarray(best["xyz"], dtype=np.float64)
    inliers = np.asarray(best["used_views"], dtype=np.int64)
    errors = np.asarray([best["reprojection_errors_px"][v] for v in inliers],
                        dtype=np.float64)
    angle = float(best["min_ray_angle_deg"])
    mask[inliers] = True
    confidence = float(best["confidence"])
    status = "observed_high" if len(inliers) >= 3 else "observed_low_two_view"
    rejected_views = np.setdiff1d(observed, inliers).astype(int).tolist()
    rejected_reasons: dict[str, str] = {}
    for view in rejected_views:
        error = best["reprojection_errors_px"][view]
        rejected_reasons[str(view)] = (
            "gross_reprojection_outlier" if error is None or float(error)
            > float(config.dist_max_px) else "excluded_by_robust_hypothesis")
    out = np.asarray((xyz[0], xyz[1], xyz[2], confidence), dtype=np.float32)
    base.update({
        "used_views":
        inliers.astype(int).tolist(),
        "rejected_views":
        rejected_views,
        "rejected_view_reasons":
        rejected_reasons,
        "reprojection_error_px":
        float(np.mean(errors)),
        "max_reprojection_error_px":
        float(np.max(errors)),
        "min_ray_angle_deg":
        float(angle),
        "status":
        status,
        "geometry_ok":
        True,
        "selected_hypothesis_id":
        selected_id,
        "selected_candidate_ranks":
        list(best["candidate_ranks"]),
        "selected_observations_xy":
        list(best["observation_xy"]),
        "selected_reprojection_errors_px":
        list(best["reprojection_errors_px"]),
        "selected_candidate_probabilities":
        list(best["candidate_probabilities"]),
        "selected_simcc_variance_px2":
        list(best["simcc_variance_px2"]),
        "robust_cost":
        float(best["robust_cost"]),
    })
    return out, mask, base


def triangulate_multiview(
    keypoints2d: np.ndarray,
    P: np.ndarray,
    config: TriangulationConfig,
    *,
    previous: np.ndarray | None = None,
    observation_meta: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Triangulate all joints and expose the exact per-joint view decision.

    ``previous`` is retained for API compatibility; temporal completion is
    performed at the burst level and is never fabricated in this function.
    """
    del previous
    kp = np.asarray(keypoints2d, dtype=np.float64)
    rt = np.asarray(P, dtype=np.float64)
    if kp.ndim != 3 or kp.shape[-1] < 3:
        raise ValueError(
            f"keypoints2d must be (views,joints,3), got {kp.shape}")
    if rt.shape != (kp.shape[0], 3, 4):
        raise ValueError(f"P must be ({kp.shape[0]},3,4), got {rt.shape}")
    n_views, n_joints = kp.shape[:2]
    out = np.zeros((n_joints, 4), dtype=np.float32)
    masked = kp.copy()
    masked[..., 2] = 0.0
    details: list[dict[str, Any]] = []
    for joint in range(n_joints):
        point, inlier_mask, detail = _joint_solution(
            kp[:, joint, :3],
            rt,
            config,
            joint_index=joint,
            is_body25=n_joints == 25,
            observation_meta=observation_meta,
        )
        out[joint] = point
        masked[inlier_mask, joint, :3] = kp[inlier_mask, joint, :3]
        selected_xy = list(detail.get("selected_observations_xy") or [])
        for view in np.flatnonzero(inlier_mask):
            if view < len(selected_xy) and selected_xy[view] is not None:
                masked[view, joint, :2] = np.asarray(selected_xy[view],
                                                     dtype=np.float64)
        detail["joint_index"] = int(joint)
        details.append(detail)

    valid = out[:, 3] > 0.0
    used = np.asarray([len(d["used_views"]) for d in details], dtype=np.int32)
    diag: dict[str, Any] = {
        "algorithm": "adaptive_hypothesis_dlt",
        "confidence_threshold": float(config.confidence_threshold),
        "min_conf": float(config.min_conf),
        "min_view": int(config.min_view),
        "core_min_view": int(config.core_min_view),
        "dist_max_px": float(config.dist_max_px),
        "two_view_max_reproj_px": float(config.two_view_max_reproj_px),
        "two_view_min_ray_angle_deg": float(config.two_view_min_ray_angle_deg),
        "observation_meta_used": bool(observation_meta),
        "n_views": int(n_views),
        "n_joints": int(n_joints),
        "valid_joints": int(np.sum(valid)),
        "two_view_joints": int(np.sum(used == 2)),
        "three_plus_view_joints": int(np.sum(used >= 3)),
        "joint_details": details,
        # Consumers that previously used EasyMocap diagnostics can keep these
        # compact fields during migration.
        "used_views": used.tolist(),
        "kpts2d_inlier_mask": masked[..., 2] > 0.0,
    }
    return out, diag
