"""Same-first-window fixed-vs-feedback diagnostic at the posed 15° state.

This is deliberately separate from the long runner.  It freezes all adjacent
single-trunk pairs that are checked at the 15° pose and whose two stations
both admit the common lateral bias ``[-8, +8]`` mm in ``truth_support``.  For
each pair, one source window is selected from geometry alone and used by both
arms.  The tail is then evaluated exactly over the two coherent bias particles
at full FOV widths 14 and 50 mm:

    fixed  = max_b E_o[V(b,o)]
    B      = E_o max_b[V(b,o)].

The candidate tail set and measured transition costs are identical for both
arms.  The adaptive-greedy suffix is the same posterior expected-capture
choice as B, so this diagnostic isolates the fixed-suffix contrast rather than
adding another first-action advantage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_GEOMETRY = HERE / "DYNAMIC_GEOMETRY_FROZEN.npz"
DEFAULT_OUTPUT = HERE / "SAME_FIRST_WINDOW.json"
POSE_INDEX = 2
POSE_DEGREES = 15.0
BIAS_PARTICLES_MM = np.asarray([-8.0, 8.0], dtype=float)
FULL_WIDTHS_MM = (14.0, 50.0)
N_BINS = 7
SUPPORT_TOLERANCE_MM = 1.0e-9

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from dynamic_scan import DynamicScan  # noqa: E402
from stress_observation import (  # noqa: E402
    DEFAULT_NOISE_MIXTURE,
    DEFAULT_P_DETECT,
    bin_kernel,
    hit_probability,
)
from truth_support import build_support  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _support_rows(scan: DynamicScan, payload: dict[str, np.ndarray]) -> dict[int, np.ndarray]:
    """Map each trunk station to its pose-15 truth-support interval."""

    _, limits = build_support(scan, payload)
    locations = [(int(site), int(label)) for site, label in scan.targets if label < 2]
    rows: dict[int, np.ndarray] = {}
    for site in range(scan.n):
        key = (site, 0)
        if key not in locations:
            continue
        row = POSE_INDEX * len(locations) + locations.index(key)
        rows[site] = np.asarray(limits[row], dtype=float)
    return rows


def _single_trunk_pairs(scan: DynamicScan, support: dict[int, np.ndarray]) -> tuple[list[dict[str, Any]], list[tuple[int, int]]]:
    """Freeze and audit every adjacent pair before outcome calculations."""

    branch = {int(site) for site in scan.branch_indices}
    all_pairs: list[dict[str, Any]] = []
    selected: list[tuple[int, int]] = []
    for source in range(scan.n - 1):
        target = source + 1
        source_limit = support.get(source)
        target_limit = support.get(target)
        source_valid = np.flatnonzero(scan.eligible[POSE_INDEX, source]).astype(int)
        target_valid = np.flatnonzero(scan.eligible[POSE_INDEX, target]).astype(int)
        single_trunk = source not in branch and target not in branch
        support_ok = bool(
            source_limit is not None
            and target_limit is not None
            and source_limit[0] <= -8.0 + SUPPORT_TOLERANCE_MM
            and source_limit[1] >= 8.0 - SUPPORT_TOLERANCE_MM
            and target_limit[0] <= -8.0 + SUPPORT_TOLERANCE_MM
            and target_limit[1] >= 8.0 - SUPPORT_TOLERANCE_MM
        )
        checked_windows = bool(source_valid.size and target_valid.size)
        valid = bool(single_trunk and support_ok and checked_windows)
        if valid:
            reason = "single_trunk_checked_pair_with_common_pm8mm_support"
            selected.append((source, target))
        elif not single_trunk:
            reason = "branch_station_excluded"
        elif not support_ok:
            reason = "truth_support_does_not_cover_common_pm8mm_bias"
        else:
            reason = "one_or_both_stations_have_no_checked_window"
        all_pairs.append(
            {
                "source_station": source,
                "target_station": target,
                "single_trunk": single_trunk,
                "source_checked_windows": source_valid.tolist(),
                "target_checked_windows": target_valid.tolist(),
                "source_support_interval_mm": None if source_limit is None else source_limit.tolist(),
                "target_support_interval_mm": None if target_limit is None else target_limit.tolist(),
                "common_bias_supported": support_ok,
                "included": valid,
                "reason": reason,
            }
        )
    return all_pairs, selected


def _first_window(
    scan: DynamicScan,
    source: int,
    view_tilts_deg: np.ndarray,
) -> tuple[int, dict[str, Any], tuple[np.ndarray, float]]:
    """Select one source window by geometry-only predictive entropy.

    The score is computed for both declared FOV widths using the two coherent
    ±8-mm bias particles.  No sampled observation or policy result enters the
    ranking.  Ties prefer larger nominal detection mass, then smaller nominal
    absolute centroid, then canonical candidate order.
    """

    valid = np.flatnonzero(scan.eligible[POSE_INDEX, source]).astype(int)
    ranked: list[tuple[tuple[float, float, float, int], int, tuple[np.ndarray, float], np.ndarray]] = []
    theta = np.zeros((len(BIAS_PARTICLES_MM), scan.dim), dtype=float)
    theta[:, source] = BIAS_PARTICLES_MM
    for candidate in valid:
        model = scan.models(POSE_INDEX, (source, int(candidate)), False)[0]
        row, base = np.asarray(model[0], dtype=float), float(model[1])
        signals = float(base) + theta @ row
        entropy_bits = 0.0
        nominal_detect = 0.0
        for width in FULL_WIDTHS_MM:
            edges = np.linspace(-width / 2.0, width / 2.0, N_BINS + 1)
            kernel, _, _ = bin_kernel(
                signals[None, :],
                edges,
                noise_mixture=DEFAULT_NOISE_MIXTURE,
                false_rate=0.0,
                p_detect=DEFAULT_P_DETECT,
            )
            predictive = kernel[0].mean(axis=0)
            live = predictive > 1.0e-15
            entropy_bits += float(-np.sum(predictive[live] * np.log2(predictive[live])))
            nominal_detect += float(
                np.mean(hit_probability(float(base), width / 2.0, noise_mixture=DEFAULT_NOISE_MIXTURE,
                                         p_detect=DEFAULT_P_DETECT))
            )
        rank = (entropy_bits, nominal_detect, -abs(float(base)), -int(candidate))
        ranked.append((rank, int(candidate), (row, base), signals))
    if not ranked:
        raise ValueError(f"source station {source} has no checked window")
    ranked.sort(key=lambda item: item[0], reverse=True)
    rank, candidate, model, signals = ranked[0]
    return candidate, {
        "candidate": candidate,
        "offset_mm": float(scan.offsets[candidate]),
        "tilt_deg": float(view_tilts_deg[candidate]),
        "nominal_signal_mm": float(model[1]),
        "signals_by_bias_mm": signals.tolist(),
        "geometry_entropy_score_bits": float(rank[0]),
        "geometry_nominal_detection_score": float(rank[1]),
        "selection_rule": "max summed predictive entropy at 14/50 mm; nominal detection, centrality, canonical index ties",
        "all_source_candidates_ranked": [
            {
                "candidate": item[1],
                "offset_mm": float(scan.offsets[item[1]]),
                "tilt_deg": float(view_tilts_deg[item[1]]),
                "nominal_signal_mm": float(item[2][1]),
                "rank_tuple": [float(item[0][0]), float(item[0][1]), float(item[0][2]), int(item[0][3])],
            }
            for item in ranked
        ],
    }, model


def _evaluate_pair_width(
    scan: DynamicScan,
    source: int,
    target: int,
    first_candidate: int,
    first_model: tuple[np.ndarray, float],
    tails: np.ndarray,
    tail_cost: np.ndarray,
    width: float,
    theta: np.ndarray,
    view_tilts_deg: np.ndarray,
) -> dict[str, Any]:
    half_width = width / 2.0
    edges = np.linspace(-half_width, half_width, N_BINS + 1)
    first_row, first_base = first_model
    first_signals = first_base + theta @ first_row
    # ``multiple`` is retained as a zero-probability labelled outcome even
    # though this diagnostic uses genuine-only bins and no false centroids.
    kernel, genuine, labels = bin_kernel(
        first_signals[None, :],
        edges,
        noise_mixture=DEFAULT_NOISE_MIXTURE,
        false_rate=0.0,
        p_detect=DEFAULT_P_DETECT,
    )
    likelihood = kernel[0]
    genuine_single = genuine[0]
    predictive = likelihood.mean(axis=0)
    prior_likelihood = likelihood / float(theta.shape[0])
    posterior = np.divide(
        prior_likelihood,
        predictive[None, :],
        out=np.zeros_like(likelihood),
        where=predictive[None, :] > 1.0e-15,
    )
    live_outcomes = predictive > 1.0e-15
    if np.any(live_outcomes) and not np.allclose(
        posterior[:, live_outcomes].sum(axis=0), 1.0, rtol=1.0e-10, atol=1.0e-12
    ):
        raise AssertionError("discrete outcome posterior columns are not normalized")
    tail_capture = np.empty((theta.shape[0], tails.size), dtype=float)
    tail_signals = np.empty_like(tail_capture)
    for tail_index, candidate in enumerate(tails):
        row, base = scan.models(POSE_INDEX, (target, int(candidate)), False)[0]
        tail_signals[:, tail_index] = base + theta @ row
        tail_capture[:, tail_index] = hit_probability(
            tail_signals[:, tail_index],
            half_width,
            noise_mixture=DEFAULT_NOISE_MIXTURE,
            p_detect=DEFAULT_P_DETECT,
        )
    branch_values = posterior.T @ tail_capture - tail_cost[None, :]
    fixed_by_tail = predictive @ branch_values
    fixed_index = int(np.argmax(fixed_by_tail))
    fixed_value = float(fixed_by_tail[fixed_index])
    feedback_index = np.argmax(branch_values, axis=1).astype(int)
    feedback_branch = np.max(branch_values, axis=1)
    feedback_value = float(predictive @ feedback_branch)
    alternate = feedback_index != fixed_index
    alternate_mass = float(predictive @ alternate.astype(float))
    return {
        "width_mm": float(width),
        "half_width_mm": float(half_width),
        "edges_mm": edges.tolist(),
        "outcome_labels": list(labels),
        "p_outcome_exact": predictive.tolist(),
        "genuine_single_mass_by_particle": genuine_single.tolist(),
        "genuine_single_mass_exact": genuine_single.mean(axis=0).tolist(),
        "outcome_sum": float(predictive.sum()),
        "tail_candidates": tails.tolist(),
        "tail_offsets_mm": [float(scan.offsets[candidate]) for candidate in tails],
        "tail_tilts_deg": [float(view_tilts_deg[candidate]) for candidate in tails],
        "tail_costs": tail_cost.tolist(),
        "tail_capture_by_bias": tail_capture.tolist(),
        "tail_signal_by_bias_mm": tail_signals.tolist(),
        "branch_value_outcome_x_tail": branch_values.tolist(),
        "fixed_index": fixed_index,
        "fixed_tail_candidate": int(tails[fixed_index]),
        "max_expected_fixed_value": fixed_value,
        "feedback_indices_by_outcome": feedback_index.tolist(),
        "feedback_tail_candidates_by_outcome": [int(tails[index]) for index in feedback_index],
        "expected_max_feedback_value": feedback_value,
        "same_first_gap": float(feedback_value - fixed_value),
        "support_weighted_alternate_branch_mass": alternate_mass,
        "adaptive_greedy_indices_by_outcome": feedback_index.tolist(),
        "adaptive_greedy_value": feedback_value,
        "adaptive_greedy_equals_B": True,
        "common_tail_bank_and_costs": True,
        "integration": "exact discrete equal prior over coherent shared biases -8/+8 mm; no Monte Carlo",
    }


def build_report(geometry_path: Path) -> dict[str, Any]:
    with np.load(geometry_path, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    scan = DynamicScan(payload)
    support = _support_rows(scan, payload)
    all_pairs, selected_pairs = _single_trunk_pairs(scan, support)
    view_tilts_deg = np.asarray(payload["view_tilts_deg"], dtype=float)
    theta_template = np.zeros((len(BIAS_PARTICLES_MM), scan.dim), dtype=float)
    pair_reports: list[dict[str, Any]] = []
    for source, target in selected_pairs:
        first_candidate, selection, first_model = _first_window(scan, source, view_tilts_deg)
        tails = np.flatnonzero(scan.eligible[POSE_INDEX, target]).astype(int)
        tail_actions = [(target, int(candidate)) for candidate in tails]
        tail_cost = np.asarray(
            scan.costs((POSE_INDEX, source, first_candidate), POSE_INDEX, tail_actions),
            dtype=float,
        )
        first_cost = float(scan.costs(None, POSE_INDEX, [(source, first_candidate)])[0])
        theta = theta_template.copy()
        theta[:, source] = BIAS_PARTICLES_MM
        theta[:, target] = BIAS_PARTICLES_MM
        widths = {
            str(int(width)): _evaluate_pair_width(
                scan, source, target, first_candidate, first_model, tails, tail_cost, width, theta, view_tilts_deg
            )
            for width in FULL_WIDTHS_MM
        }
        pair_reports.append(
            {
                "source_station": source,
                "target_station": target,
                "source_arc_mm": float(scan.s[source]),
                "target_arc_mm": float(scan.s[target]),
                "source_support_interval_mm": support[source].tolist(),
                "target_support_interval_mm": support[target].tolist(),
                "first_cost": first_cost,
                "first_window": selection,
                "source_checked_windows": np.flatnonzero(scan.eligible[POSE_INDEX, source]).tolist(),
                "target_checked_windows": tails.tolist(),
                "widths": widths,
            }
        )
    return {
        "study": "same_first_window_fixed_vs_feedback",
        "pose_index": POSE_INDEX,
        "pose_degrees": POSE_DEGREES,
        "geometry": str(geometry_path.resolve()),
        "geometry_sha256": _sha256(geometry_path),
        "dynamic_scan_sha256": _sha256(HERE / "dynamic_scan.py"),
        "truth_support_sha256": _sha256(HERE / "truth_support.py"),
        "decision_scope": {
            "same_first_window": True,
            "first_selection_uses_outcomes": False,
            "first_selection_rule": "geometry-only entropy over two shared ±8 mm particles at both full widths",
            "tail_candidates": "all 15-degree checked target windows for each included pair",
            "tail_cost": "DynamicScan checked cost from selected source window to each target window",
            "A": "one fixed suffix maximizing expected branch value",
            "B": "outcome-feedback suffix maximizing each branch then averaging",
            "adaptive_greedy": "same branchwise posterior expected-capture suffix as B",
            "replay_performed": False,
        },
        "prior": {
            "bias_particles_mm": BIAS_PARTICLES_MM.tolist(),
            "weights": [0.5, 0.5],
            "shared_bias_across_source_target": True,
            "noise_mixture": [list(component) for component in DEFAULT_NOISE_MIXTURE],
            "p_detect": DEFAULT_P_DETECT,
            "false_rate": 0.0,
            "bins": N_BINS,
        },
        "support_gate": {
            "common_bias_mm": [-8.0, 8.0],
            "basis": "truth_support.build_support at pose index 2; both node intervals must contain both endpoints",
        },
        "all_adjacent_pairs": all_pairs,
        "included_pair_count": len(pair_reports),
        "included_pairs": pair_reports,
        "diagnostic_summary": {
            "widths_mm": list(FULL_WIDTHS_MM),
            "pairs_with_positive_same_first_gap_by_width": {
                str(int(width)): sum(
                    pair["widths"][str(int(width))]["same_first_gap"] > 1.0e-10
                    for pair in pair_reports
                )
                for width in FULL_WIDTHS_MM
            },
            "pairs_with_changed_feedback_tail_by_width": {
                str(int(width)): sum(
                    pair["widths"][str(int(width))]["support_weighted_alternate_branch_mass"] > 1.0e-10
                    for pair in pair_reports
                )
                for width in FULL_WIDTHS_MM
            },
            "mean_same_first_gap_by_width": {
                str(int(width)): float(np.mean([
                    pair["widths"][str(int(width))]["same_first_gap"] for pair in pair_reports
                ]))
                for width in FULL_WIDTHS_MM
            },
            "maximum_same_first_gap_by_width": {
                str(int(width)): float(max(
                    pair["widths"][str(int(width))]["same_first_gap"] for pair in pair_reports
                ))
                for width in FULL_WIDTHS_MM
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(args.geometry)
    args.output.write_text(json.dumps(_jsonable(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = report["diagnostic_summary"]
    print(json.dumps({"output": str(args.output), "included_pairs": report["included_pair_count"], "summary": summary}))


if __name__ == "__main__":
    main()

