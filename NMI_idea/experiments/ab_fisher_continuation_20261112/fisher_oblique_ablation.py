"""Synthetic oblique-response data ablation on the checked DynamicScan bank.

This is deliberately separate from the completion policy.  It asks whether a
bounded teacher set helps learn signed action responses, rather than whether a
probe changes a final B decision.  The candidate bank and public safety mask
come from the checked DynamicScan geometry.  Training excludes site indices
``mod 5 == 0`` and the checked 19-degree pose; evaluation uses exactly those
sites, the untouched 19-degree pose, and two untouched subjects.

The contrast family is fixed before any simulated outcomes are drawn:

* every admitted noncentral lateral offset and +/-12-degree physical tilt at
  held-out 19 degrees, compared with the same-site 19-degree center contact;
* one centered 19-degree-minus-0-degree pose contrast per held-out site.

The response model includes signed pose, signed lateral, signed physical tilt,
depth, and pose*lateral, pose*tilt, lateral*tilt interactions, plus an
unknown session zero-offset nuisance for each training subject.  Primary
methods use equal 24- or 48-attempt budgets: Fisher oblique teacher, uniform
diverse random, and center-only.  The 48-contact center rule is explicitly a
same-support quantity control.  All seeds, ranks, true contrast counts, and
paired uncertainty are written to ``FISHER_OBLIQUE_ABLATION.json``.

This synthetic experiment does not identify a mixture mode and does not claim
hardware or subject-validity beyond the declared simulator assumptions.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fisher_data_ablation import (  # noqa: E402
    BIN_EDGES,
    DETECT_PROBABILITY,
    GEOMETRY,
    OBSERVATION_SCALE,
    Contact,
    _load_contacts,
    _sha256,
    greedy_common_candidates,
    binned_signed_centroid_fisher,
)


SEED_SELECTION = 2026111217
SEED_DATA = 2026111218
REPLICATES = 24
TRAIN_SUBJECTS = 6
HELDOUT_SUBJECTS = 2
TRAIN_POSES = (0, 1, 2)  # 0, 11, 15 degrees; pose 3 = untouched 19 degrees
HELDOUT_POSE = 3
PRIMARY_BUDGETS = (24, 48)
SITE_HOLDOUT_MOD = 5
TASK_DIM = 8
PARAMETER_DIM = TASK_DIM + 1  # one representative session-zero nuisance
TILT_SCALE_DEG = 12.0


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _scales(contacts: Sequence[Contact]) -> tuple[float, float, float, float]:
    safe = [row for row in contacts if row.safe]
    lat = np.asarray([row.lateral_mm for row in safe], dtype=float)
    depth = np.asarray([row.depth_mm for row in safe], dtype=float)
    return (
        10.0,  # frozen pose-centre used by the predeclared design family
        max(1.0, float(np.percentile(np.abs(lat), 75))),
        float(np.mean(depth)),
        max(1.0, float(np.std(depth))),
    )


def _features(row: Contact, scales: tuple[float, float, float, float]) -> np.ndarray:
    pose_center, lateral_scale, depth_center, depth_scale = scales
    pose = (row.pose_deg - pose_center) / 15.0
    lateral = row.lateral_mm / lateral_scale
    depth = (row.depth_mm - depth_center) / depth_scale
    tilt = row.view_tilt_deg / TILT_SCALE_DEG
    return np.asarray([
        1.0,
        pose,
        lateral,
        depth,
        tilt,
        pose * lateral,
        pose * tilt,
        lateral * tilt,
    ], dtype=float)


def _safe_training_mask(contacts: Sequence[Contact]) -> np.ndarray:
    return np.asarray([
        row.safe and row.pose_index in TRAIN_POSES and row.site_index % SITE_HOLDOUT_MOD != 0
        for row in contacts
    ], dtype=bool)


def _center_view(contacts: Sequence[Contact], pose: int, site: int) -> Contact | None:
    rows = [row for row in contacts if row.pose_index == pose and row.site_index == site
            and abs(row.view_offset_mm) < 1e-9 and abs(row.view_tilt_deg) < 1e-9 and row.safe]
    return min(rows, key=lambda row: row.view_index) if rows else None


def _candidate_fishers(contacts: Sequence[Contact], feature_table: np.ndarray) -> np.ndarray:
    fishers = np.zeros((len(contacts), PARAMETER_DIM, PARAMETER_DIM), dtype=float)
    for i, row in enumerate(contacts):
        if not row.safe:
            continue
        jacobian = np.zeros((3, PARAMETER_DIM), dtype=float)
        jacobian[0, :TASK_DIM] = feature_table[i]
        jacobian[0, -1] = 1.0
        result = binned_signed_centroid_fisher(
            centroid=0.0,
            scale=OBSERVATION_SCALE,
            p_detect=DETECT_PROBABILITY,
            edges=BIN_EDGES,
            parameter_jacobian=jacobian,
        )
        fishers[i] = result.parameter_fisher
    return fishers


def _contrast_family(contacts: Sequence[Contact], feature_table: np.ndarray,
                     training_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build the fixed training-geometry oblique score directions."""

    directions: list[np.ndarray] = []
    weights: list[float] = []
    counts_by_kind = {"training_oblique": 0, "training_centered": 0}
    for row in contacts:
        if not training_mask[row.index] or not row.safe:
            continue
        center = _center_view(contacts, row.pose_index, row.site_index)
        if center is None or row.index == center.index:
            continue
        directions.append(np.r_[feature_table[row.index] - feature_table[center.index], 0.0])
        weights.append(1.0)
        counts_by_kind["training_oblique"] += 1
    # Include centered pose response in the score, with lower weight so it
    # cannot overwhelm action-response directions.
    for site in sorted({row.site_index for row in contacts if training_mask[row.index]}):
        low = _center_view(contacts, TRAIN_POSES[0], site)
        high = _center_view(contacts, TRAIN_POSES[-1], site)
        if low is None or high is None:
            continue
        directions.append(np.r_[feature_table[high.index] - feature_table[low.index], 0.0])
        weights.append(0.25)
        counts_by_kind["training_centered"] += 1
    if not directions:
        raise ValueError("no predeclared training contrast directions")
    direction_matrix = np.asarray(directions, dtype=float).T
    weight_array = np.asarray(weights, dtype=float)
    weight_array /= float(weight_array.sum())
    metadata = {
        "family_predeclared_before_simulation": True,
        "training_direction_count": int(direction_matrix.shape[1]),
        "training_direction_counts_by_kind": counts_by_kind,
        "direction_definition": "feature(noncentral checked contact) - feature(same-site same-pose checked center); plus 15deg center - 0deg center",
        "direction_parameter_order": [
            "intercept", "pose", "lateral", "depth", "physical_tilt",
            "pose*lateral", "pose*physical_tilt", "lateral*physical_tilt", "session_zero_offset",
        ],
    }
    return direction_matrix, weight_array, metadata


def _center_selection(contacts: Sequence[Contact], training_mask: np.ndarray, budget: int) -> list[int]:
    pool = [row.index for row in contacts if training_mask[row.index]
            and abs(row.view_offset_mm) < 1e-9 and abs(row.view_tilt_deg) < 1e-9]
    if len(pool) < budget:
        raise ValueError(f"center support {len(pool)} cannot fill budget {budget}")
    pool.sort(key=lambda i: (contacts[i].site_index, contacts[i].pose_index, contacts[i].view_index))
    return [pool[int(i)] for i in np.linspace(0, len(pool) - 1, budget, dtype=int)]


def _uniform_selection(contacts: Sequence[Contact], training_mask: np.ndarray, budget: int) -> list[int]:
    pool = np.flatnonzero(training_mask)
    rng = np.random.default_rng(SEED_SELECTION + budget)
    return [int(i) for i in rng.choice(pool, size=budget, replace=False)]


def _fisher_selection(contacts: Sequence[Contact], training_mask: np.ndarray,
                      fishers: np.ndarray, directions: np.ndarray,
                      weights: np.ndarray, budget: int) -> tuple[list[int], dict[str, Any]]:
    prior = np.diag(np.asarray([2.0, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.6]) ** 2)
    result = greedy_common_candidates(
        fishers,
        prior,
        budget,
        safe_mask=training_mask,
        candidate_ids=[row.index for row in contacts],
        decision_directions=directions,
        weights=weights,
        method="directional_trace",
        min_marginal_score=-1.0e-12,
    )
    if len(result.selected_indices) != budget:
        raise ValueError("Fisher teacher could not fill the requested budget")
    return list(result.selected_indices), {
        "marginal_scores": list(result.marginal_scores),
        "cumulative_scores": list(result.cumulative_scores),
        "safe_support_count": int(np.sum(result.safe_mask)),
    }


def _subject_parameters(rng: np.random.Generator, truth: np.ndarray) -> tuple[np.ndarray, float]:
    delta_scale = np.asarray([0.0, 0.06, 0.06, 0.045, 0.05, 0.07, 0.08, 0.08])
    return truth + rng.normal(0.0, delta_scale), float(rng.normal(0.0, 1.5))


def _make_replicate(contacts: Sequence[Contact], feature_table: np.ndarray,
                    training_mask: np.ndarray, truth: np.ndarray, replicate: int) -> dict[str, Any]:
    rng = np.random.default_rng(SEED_DATA + 1000 * replicate)
    subjects = [_subject_parameters(rng, truth) for _ in range(TRAIN_SUBJECTS + HELDOUT_SUBJECTS)]
    observations: dict[tuple[int, int], tuple[bool, float | None]] = {}
    for subject in range(TRAIN_SUBJECTS):
        coefficients, offset = subjects[subject]
        for row in contacts:
            if not training_mask[row.index]:
                continue
            value = float(feature_table[row.index] @ coefficients + offset + rng.normal(0.0, OBSERVATION_SCALE))
            detected = bool(rng.random() < DETECT_PROBABILITY)
            interior = detected and BIN_EDGES[1] <= value <= BIN_EDGES[-2]
            observations[(subject, row.index)] = (detected, value if interior else None)
    return {"subjects": subjects, "observations": observations}


def _fit_selected(contacts: Sequence[Contact], feature_table: np.ndarray, selected: Sequence[int],
                  replicate: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    rows: list[np.ndarray] = []
    values: list[float] = []
    detected = 0
    censored = 0
    for subject in range(TRAIN_SUBJECTS):
        for index in selected:
            hit, value = replicate["observations"][(subject, index)]
            if not hit:
                continue
            detected += 1
            if value is None:
                censored += 1
                continue
            nuisance = np.zeros(TRAIN_SUBJECTS - 1, dtype=float)
            if subject > 0:
                nuisance[subject - 1] = 1.0
            rows.append(np.r_[feature_table[index], nuisance])
            values.append(float(value))
    design = np.asarray(rows, dtype=float)
    response = np.asarray(values, dtype=float)
    if design.shape[0] < PARAMETER_DIM + TRAIN_SUBJECTS:
        raise ValueError("too few detected/uncensored observations for oblique fit")
    fit, _residuals, rank, _singular = np.linalg.lstsq(design, response, rcond=1.0e-10)
    return fit[:TASK_DIM], {
        "design_matrix_rows": int(design.shape[0]),
        "design_matrix_columns": int(design.shape[1]),
        "design_matrix_rank": int(rank),
        "detected_count": int(detected),
        "censored_detected_count": int(censored),
        "uncensored_count": int(len(values)),
    }


def _heldout_contrasts(contacts: Sequence[Contact], feature_table: np.ndarray,
                       coefficients: np.ndarray) -> dict[str, np.ndarray]:
    heldout_sites = sorted({row.site_index for row in contacts if row.site_index % SITE_HOLDOUT_MOD == 0})
    centered: list[np.ndarray] = []
    oblique: list[np.ndarray] = []
    for site in heldout_sites:
        low = _center_view(contacts, TRAIN_POSES[0], site)
        center19 = _center_view(contacts, HELDOUT_POSE, site)
        if low is None or center19 is None:
            continue
        centered.append(feature_table[center19.index] - feature_table[low.index])
        for row in contacts:
            if row.site_index != site or row.pose_index != HELDOUT_POSE or not row.safe:
                continue
            if row.index == center19.index:
                continue
            # Includes every admitted +/- lateral offset and +/- physical tilt
            # represented by the checked table, with no result-dependent gate.
            oblique.append(feature_table[row.index] - feature_table[center19.index])
    if not centered or not oblique:
        raise ValueError("heldout contrast family is empty")
    centered_array = np.asarray(centered, dtype=float)
    oblique_array = np.asarray(oblique, dtype=float)
    return {
        "centered": centered_array,
        "oblique": oblique_array,
        "centered_true": centered_array @ coefficients,
        "oblique_true": oblique_array @ coefficients,
        "heldout_site_count": np.asarray(heldout_sites, dtype=int),
    }


def _metric(fit: np.ndarray, family: Mapping[str, np.ndarray], subject_truth: Sequence[np.ndarray]) -> dict[str, float]:
    output: dict[str, float] = {}
    all_errors: list[float] = []
    for kind in ("centered", "oblique"):
        errors: list[float] = []
        design = family[kind]
        for truth in subject_truth:
            errors.extend((design @ fit - design @ truth).tolist())
        error = np.asarray(errors, dtype=float)
        output[f"{kind}_rmse"] = float(np.sqrt(np.mean(error ** 2)))
        output[f"{kind}_bias"] = float(np.mean(error))
        output[f"{kind}_abs_bias"] = float(np.mean(np.abs(error)))
        output[f"{kind}_true_count"] = int(error.size)
        all_errors.extend(error.tolist())
    all_error = np.asarray(all_errors, dtype=float)
    output["all_rmse"] = float(np.sqrt(np.mean(all_error ** 2)))
    output["all_bias"] = float(np.mean(all_error))
    output["all_true_count"] = int(all_error.size)
    return output


def _summary(rows: Sequence[Mapping[str, Any]], reference: Sequence[Mapping[str, Any]] | None) -> dict[str, Any]:
    keys = ("centered_rmse", "oblique_rmse", "all_rmse", "centered_bias", "oblique_bias", "all_bias")
    result: dict[str, Any] = {
        f"{key}_mean": float(np.mean([row[key] for row in rows])) for key in keys
    }
    result.update({
        f"{key}_sd": float(np.std([row[key] for row in rows], ddof=1)) for key in keys
    })
    result["design_matrix_rank_min"] = int(min(row["design_matrix_rank"] for row in rows))
    result["design_matrix_rank_max"] = int(max(row["design_matrix_rank"] for row in rows))
    result["detected_count_mean"] = float(np.mean([row["detected_count"] for row in rows]))
    result["censored_detected_count_mean"] = float(np.mean([row["censored_detected_count"] for row in rows]))
    result["true_contrast_count"] = int(rows[0]["oblique_true_count"] + rows[0]["centered_true_count"])
    if reference is not None:
        delta = np.asarray([row["oblique_rmse"] for row in rows]) - np.asarray([row["oblique_rmse"] for row in reference])
        result["paired_uncertainty"] = {
            "reference": "same-budget fisher_teacher",
            "oblique_rmse_delta_mean": float(np.mean(delta)),
            "oblique_rmse_delta_sd": float(np.std(delta, ddof=1)),
            "oblique_rmse_delta_ci95": [
                float(np.mean(delta) - 1.96 * np.std(delta, ddof=1) / np.sqrt(len(delta))),
                float(np.mean(delta) + 1.96 * np.std(delta, ddof=1) / np.sqrt(len(delta))),
            ],
            "paired_replicates": int(len(delta)),
        }
    return result


def run_ablation() -> dict[str, Any]:
    contacts, source = _load_contacts()
    scales = _scales(contacts)
    feature_table = np.asarray([_features(row, scales) for row in contacts], dtype=float)
    training_mask = _safe_training_mask(contacts)
    fishers = _candidate_fishers(contacts, feature_table)
    directions, weights, family_metadata = _contrast_family(contacts, feature_table, training_mask)
    truth = np.asarray([0.18, 0.70, -0.34, 0.16, 0.80, 0.46, 0.58, -0.43], dtype=float)
    heldout_family = _heldout_contrasts(contacts, feature_table, truth)
    selections: dict[str, list[int]] = {}
    audits: dict[str, Any] = {}
    for budget in PRIMARY_BUDGETS:
        fisher, fisher_audit = _fisher_selection(contacts, training_mask, fishers, directions, weights, budget)
        selections[f"fisher_teacher_{budget}"] = fisher
        audits[f"fisher_teacher_{budget}"] = fisher_audit
        selections[f"uniform_diverse_{budget}"] = _uniform_selection(contacts, training_mask, budget)
        selections[f"center_only_{budget}"] = _center_selection(contacts, training_mask, budget)
    if any(len(rows) != int(name.rsplit("_", 1)[-1]) for name, rows in selections.items()):
        raise AssertionError("selection methods do not share their declared budgets")

    replicate_data = [_make_replicate(contacts, feature_table, training_mask, truth, r)
                      for r in range(REPLICATES)]
    metrics: dict[str, Any] = {}
    raw_rows: dict[str, list[dict[str, Any]]] = {}
    for name, selected in selections.items():
        rows: list[dict[str, Any]] = []
        for replicate in replicate_data:
            fit, fit_diag = _fit_selected(contacts, feature_table, selected, replicate)
            family = _heldout_contrasts(contacts, feature_table, truth)
            # Rebuild truth contrasts for each held-out subject so response
            # heterogeneity remains part of the held-out task.
            subject_truth = [item[0] for item in replicate["subjects"][TRAIN_SUBJECTS:]]
            family_rows = _metric(fit, family, subject_truth)
            rows.append({
                **fit_diag,
                **family_rows,
                "centered_true_count": int(family["centered"].shape[0] * HELDOUT_SUBJECTS),
                "oblique_true_count": int(family["oblique"].shape[0] * HELDOUT_SUBJECTS),
            })
        raw_rows[name] = rows
        budget = int(name.rsplit("_", 1)[-1])
        reference = raw_rows.get(f"fisher_teacher_{budget}") if not name.startswith("fisher_teacher") else None
        metrics[name] = {
            "attempt_budget": budget,
            "accepted_checked_contacts": len(selected),
            "same_training_support": True,
            "selection_indices": [int(i) for i in selected],
            "selection_pose_degrees": [float(contacts[i].pose_deg) for i in selected],
            "selection_lateral_mm": [float(contacts[i].lateral_mm) for i in selected],
            "selection_tilt_degrees": [float(contacts[i].view_tilt_deg) for i in selected],
            "selection_pose_counts": {
                str(int(pose)): int(sum(contacts[i].pose_index == pose for i in selected))
                for pose in range(4)
            },
            "summary": _summary(rows, reference),
            "replicate_metrics": rows,
        }
    for budget in PRIMARY_BUDGETS:
        metrics[f"fisher_teacher_{budget}"]["selection_audit"] = audits[f"fisher_teacher_{budget}"]

    heldout_true_count = int(heldout_family["centered"].shape[0] + heldout_family["oblique"].shape[0]) * HELDOUT_SUBJECTS
    result = {
        "schema": "fisher_oblique_data_ablation_v1",
        "purpose": "bounded synthetic signed-action response learning ablation; not a B-completion result",
        "source": source,
        "simulator_assumptions": {
            "response_features": [
                "intercept", "signed_pose", "signed_lateral", "depth", "signed_physical_tilt",
                "pose*lateral", "pose*physical_tilt", "lateral*physical_tilt",
            ],
            "truth_coefficients": truth.tolist(),
            "subject_response_sd": [0.0, 0.06, 0.06, 0.045, 0.05, 0.07, 0.08, 0.08],
            "session_zero_offset_sd": 1.5,
            "observation_scale": OBSERVATION_SCALE,
            "p_detect": DETECT_PROBABILITY,
            "public_bins": list(BIN_EDGES),
            "fit": "least squares on detected uncensored values with one nuisance offset per training subject; misses/censored values are not imputed",
            "heldout": "site_index mod 5 == 0, checked 19-degree pose, and two subjects never enter selection/fitting",
            "mixture_mode_identification": "not_claimed",
        },
        "contrast_family": {
            **family_metadata,
            "heldout_pose_degree": 19.0,
            "heldout_oblique_definition": "all safe noncentral checked +/- lateral offsets and +/-12-degree tilts at 19 degrees minus same-site 19-degree center",
            "heldout_centered_definition": "same-site 19-degree center minus same-site 0-degree center",
            "heldout_oblique_true_count": int(heldout_family["oblique"].shape[0] * HELDOUT_SUBJECTS),
            "heldout_centered_true_count": int(heldout_family["centered"].shape[0] * HELDOUT_SUBJECTS),
            "heldout_total_true_count": heldout_true_count,
            "bias_cancellation": "session zero offset cancels within each same-site contrast",
        },
        "design": {
            "methods_by_budget": {
                "24": ["fisher_teacher_24", "uniform_diverse_24", "center_only_24"],
                "48": ["fisher_teacher_48", "uniform_diverse_48", "center_only_48"]
            },
            "common_training_support_count": int(np.sum(training_mask)),
            "primary_budgets": list(PRIMARY_BUDGETS),
            "same_support_center_2x": True,
            "replicates": REPLICATES,
            "selection_seed": SEED_SELECTION,
            "data_seed": SEED_DATA,
            "design_parameter_count": PARAMETER_DIM,
            "true_task_coefficient_count": int(np.sum(np.abs(truth) > 0.0)),
        },
        "metrics": metrics,
    }
    return _json_safe(result)


def main() -> None:
    output = HERE / "FISHER_OBLIQUE_ABLATION.json"
    payload = run_ablation()
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, row in payload["metrics"].items():
        print(name, row["summary"])


if __name__ == "__main__":
    main()

