"""Bounded synthetic data-collection ablation for the checked DynamicScan bank.

The script is a data-use diagnostic, not a completion-policy experiment.  It
loads the checked ``DYNAMIC_GEOMETRY.npz`` table, restricts teacher candidates
to retained/static/projection-feasible contacts, and compares three common
support rules at an equal contact-attempt budget:

``fisher_teacher``
    Greedy marginal covariance-reduction score from the public censored
    signed-centroid Fisher kernel.
``center_only``
    Central zero-lateral/zero-tilt contacts, spread over available stations
    and checked pose knots.
``uniform_random``
    Uniform random safe contacts, with a fixed selection seed.

The simulated response has five shared task coefficients (intercept, pose,
lateral, depth, and pose*lateral interaction) plus one session zero offset
per training subject.  A response observation is detected with fixed
probability and is binned/censored at the public layer; detected interior
values are used by a small least-squares fit.  Held-out sites, the 19-degree
pose knot, and held-out subjects never enter selection or fitting.  Evaluation
uses within-site contrasts, so the unknown session offset cancels by design.

All simulator choices are declared in ``FISHER_DATA_ABLATION.json``.  The
script does not claim that the Fisher score identifies a mixture mode or that
synthetic metrics predict hardware performance.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
GEOMETRY = HERE.parent / "ab_leg_dynamic_20261112" / "DYNAMIC_GEOMETRY.npz"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from probe_information import (  # noqa: E402
    binned_signed_centroid_fisher,
    greedy_common_candidates,
    task_design_score,
)


SEED_SELECTION = 2026111207
SEED_DATA = 2026111208
N_TRAIN_SUBJECTS = 6
N_HELDOUT_SUBJECTS = 2
N_REPLICATES = 24
ATTEMPT_BUDGET = 24
HIGH_QUANTITY_BUDGET = 48
DETECT_PROBABILITY = 0.88
OBSERVATION_SCALE = 0.85
BIN_EDGES = (-np.inf, -2.5, -0.75, 0.75, 2.5, np.inf)
HELDOUT_POSE_INDEX = 3  # checked 19 degrees; 0/11/15 are training poses
TASK_DIM = 5
PARAMETER_DIM = 6  # task coefficients + one representative session offset


@dataclass(frozen=True)
class Contact:
    """One public checked DynamicScan contact candidate."""

    index: int
    pose_index: int
    site_index: int
    view_index: int
    pose_deg: float
    lateral_mm: float
    depth_mm: float
    view_offset_mm: float
    view_tilt_deg: float
    safe: bool

    def features(self, pose_center: float, pose_scale: float, lat_scale: float,
                 depth_center: float, depth_scale: float) -> np.ndarray:
        pose = (self.pose_deg - pose_center) / pose_scale
        lateral = self.lateral_mm / lat_scale
        depth = (self.depth_mm - depth_center) / depth_scale
        return np.asarray([1.0, pose, lateral, depth, pose * lateral], dtype=float)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_contacts(path: Path = GEOMETRY) -> tuple[list[Contact], dict[str, Any]]:
    """Load only checked DynamicScan geometry and form a stable contact bank."""

    with np.load(path, allow_pickle=False) as data:
        poses = np.asarray(data["pose_knee_deg"], dtype=float)
        if not np.array_equal(poses, np.asarray([0.0, 11.0, 15.0, 19.0])):
            raise ValueError(f"unexpected checked pose knots: {poses.tolist()}")
        offsets = np.asarray(data["view_offsets_mm"], dtype=float)
        tilts = np.asarray(data["view_tilts_deg"], dtype=float)
        lateral = np.asarray(data["view_lat_mm"], dtype=float)
        views = np.asarray(data["posed_views_m"], dtype=float)
        frames = np.asarray(data["posed_frames"], dtype=float)
        vessel = np.asarray(data["posed_vessel_m"], dtype=float)
        retained = np.asarray(data["retained_candidate_mask"], dtype=bool)
        static = np.asarray(data["static_found"], dtype=bool)
        feasible = np.asarray(data["view_projection_feasible"], dtype=bool)
        if lateral.ndim == 2:
            lateral = np.broadcast_to(lateral[None, ...], (poses.size,) + lateral.shape)
        n_pose, n_site, n_view = lateral.shape
        if views.shape[:3] != (n_pose, n_site, n_view) or frames.shape[:3] != (n_pose, n_site, n_view):
            raise ValueError("DynamicScan pose/view shapes disagree")
        # Depth is computed from the checked vessel/contact frame rather than
        # copied from a nominal pose, retaining the actual 0/11/15/19 geometry.
        depth = 1000.0 * np.einsum("pskc,pskc->psk", vessel[:, :, None] - views, frames[..., 2])
        contacts: list[Contact] = []
        flat = 0
        for p in range(n_pose):
            for s in range(n_site):
                for v in range(n_view):
                    ok = bool(retained[p, s, v] and static[p, s, v] and feasible[s, v]
                              and np.isfinite(lateral[p, s, v]) and np.isfinite(depth[p, s, v]))
                    contacts.append(Contact(
                        index=flat, pose_index=p, site_index=s, view_index=v,
                        pose_deg=float(poses[p]), lateral_mm=float(lateral[p, s, v]),
                        depth_mm=float(depth[p, s, v]), view_offset_mm=float(offsets[v]),
                        view_tilt_deg=float(tilts[v]), safe=ok,
                    ))
                    flat += 1
        metadata = {
            "source_npz": str(path),
            "source_sha256": _sha256(path),
            "checked_pose_degrees": poses.tolist(),
            "checked_view_offsets_mm": offsets.tolist(),
            "checked_view_tilts_deg": tilts.tolist(),
            "contact_shape": [int(n_pose), int(n_site), int(n_view)],
            "safe_contact_count": int(sum(row.safe for row in contacts)),
            "safe_definition": "retained_candidate_mask & static_found & view_projection_feasible & finite signed-contact features",
        }
    return contacts, metadata


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_scales(contacts: Sequence[Contact]) -> tuple[float, float, float, float, float]:
    safe = [row for row in contacts if row.safe]
    poses = np.asarray([row.pose_deg for row in safe], dtype=float)
    lat = np.asarray([row.lateral_mm for row in safe], dtype=float)
    depth = np.asarray([row.depth_mm for row in safe], dtype=float)
    pose_center = float(np.mean([0.0, 11.0, 15.0]))
    pose_scale = max(1.0, float(np.ptp(poses)))
    lat_scale = max(1.0, float(np.percentile(np.abs(lat), 75)))
    depth_center = float(np.mean(depth))
    depth_scale = max(1.0, float(np.std(depth)))
    return pose_center, pose_scale, lat_scale, depth_center, depth_scale


def _candidate_features(contacts: Sequence[Contact], scales: tuple[float, float, float, float, float]) -> np.ndarray:
    return np.asarray([
        row.features(*scales) if row.safe else np.zeros(TASK_DIM, dtype=float)
        for row in contacts
    ], dtype=float)


def _candidate_fishers(contacts: Sequence[Contact], features: np.ndarray) -> np.ndarray:
    """Public censored Fisher per checked contact, with session nuisance row."""

    fishers = np.zeros((len(contacts), PARAMETER_DIM, PARAMETER_DIM), dtype=float)
    for i, row in enumerate(contacts):
        if not row.safe:
            continue
        # d(centroid)/d(task+nuisance) = [task feature, session offset].
        jacobian = np.zeros((3, PARAMETER_DIM), dtype=float)
        jacobian[0, :TASK_DIM] = features[i]
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


def _center_only_selection(contacts: Sequence[Contact], budget: int) -> list[int]:
    rows = [row.index for row in contacts if row.safe and abs(row.view_offset_mm) < 1e-9
            and abs(row.view_tilt_deg) < 1e-9 and row.pose_index != HELDOUT_POSE_INDEX]
    rows.sort(key=lambda i: (contacts[i].site_index, contacts[i].pose_index, contacts[i].view_index))
    if len(rows) < budget:
        raise ValueError("central rule has fewer safe contacts than the common budget")
    # Evenly spaced in the checked station order, preserving deterministic
    # common support without using response outcomes to choose sites.
    selected = np.linspace(0, len(rows) - 1, budget, dtype=int)
    return [rows[int(i)] for i in selected]


def _uniform_selection(contacts: Sequence[Contact], budget: int, seed: int) -> list[int]:
    rows = [row.index for row in contacts if row.safe and row.pose_index != HELDOUT_POSE_INDEX]
    rng = np.random.default_rng(seed)
    return [int(i) for i in rng.choice(rows, size=budget, replace=False)]


def _fisher_selection(contacts: Sequence[Contact], fishers: np.ndarray, features: np.ndarray,
                      budget: int) -> tuple[list[int], dict[str, Any]]:
    allowed = np.asarray([row.safe and row.pose_index != HELDOUT_POSE_INDEX for row in contacts], dtype=bool)
    prior = np.diag(np.asarray([2.0, 1.0, 1.0, 1.0, 1.0, 1.6], dtype=float) ** 2)
    # The target is a within-site pose contrast at an untouched 19-degree knot
    # and a within-site lateral contrast at the 15-degree knot.  The session
    # offset coordinate is exactly zero in both directions.
    high = np.asarray([1.0, (19.0 - 10.0) / 15.0, 0.0, 0.0, 0.0, 0.0])
    low = np.asarray([1.0, (0.0 - 10.0) / 15.0, 0.0, 0.0, 0.0, 0.0])
    d_pose = high - low
    d_lat = np.asarray([0.0, 0.0, 1.0, 0.0, (19.0 - 10.0) / 15.0, 0.0])
    directions = np.column_stack((d_pose, d_lat))
    selection = greedy_common_candidates(
        fishers,
        prior,
        budget,
        safe_mask=allowed,
        candidate_ids=[row.index for row in contacts],
        decision_directions=directions,
        weights=np.asarray([1.0, 0.5]),
        method="directional_trace",
        min_marginal_score=-1.0e-12,
    )
    if len(selection.selected_indices) != budget:
        raise ValueError("Fisher rule did not fill the common contact budget")
    return list(selection.selected_indices), {
        "score_directions": directions.tolist(),
        "marginal_scores": list(selection.marginal_scores),
        "cumulative_scores": list(selection.cumulative_scores),
        "safe_mask_count": int(np.sum(selection.safe_mask)),
    }


def _subject_coefficients(rng: np.random.Generator, subject: int, truth: np.ndarray) -> tuple[np.ndarray, float]:
    # Small subject response variation is deliberately independent of the
    # session zero offset.  The held-out subject split therefore tests transfer
    # of the response, not memorization of offsets.
    delta = rng.normal(0.0, np.asarray([0.0, 0.055, 0.055, 0.045, 0.06]))
    coefficients = truth + delta
    offset = float(rng.normal(0.0, 1.5))
    return coefficients, offset


def _draw_observation(rng: np.random.Generator, contact: Contact, features: np.ndarray,
                      coefficients: np.ndarray, session_offset: float) -> tuple[bool, float | None, str]:
    mean = float(features @ coefficients + session_offset)
    detected = bool(rng.random() < DETECT_PROBABILITY)
    if not detected:
        return False, None, "miss"
    value = mean + float(rng.normal(0.0, OBSERVATION_SCALE))
    if value < BIN_EDGES[1]:
        return True, None, "left_censored"
    if value > BIN_EDGES[-2]:
        return True, None, "right_censored"
    return True, value, "interior"


def _fit_selected(
    contacts: Sequence[Contact],
    features: np.ndarray,
    selected: Sequence[int],
    rng: np.random.Generator,
    truth: np.ndarray,
    *,
    n_train_subjects: int = N_TRAIN_SUBJECTS,
) -> tuple[np.ndarray, dict[str, Any]]:
    rows: list[np.ndarray] = []
    values: list[float] = []
    detected_count = 0
    censored_count = 0
    for subject in range(n_train_subjects):
        coefficients, offset = _subject_coefficients(rng, subject, truth)
        for index in selected:
            detected, value, kind = _draw_observation(rng, contacts[index], features[index], coefficients, offset)
            if not detected:
                continue
            detected_count += 1
            if value is None:
                censored_count += 1
                continue
            # Reference subject's offset is absorbed by the task intercept;
            # other subject offsets are one-hot nuisance coordinates.
            nuisance = np.zeros(n_train_subjects - 1, dtype=float)
            if subject > 0:
                nuisance[subject - 1] = 1.0
            rows.append(np.concatenate((features[index], nuisance)))
            values.append(float(value))
    if len(rows) < TASK_DIM + n_train_subjects - 1:
        raise ValueError("too few uncensored training observations")
    design = np.asarray(rows, dtype=float)
    response = np.asarray(values, dtype=float)
    fit, residuals, rank, singular = np.linalg.lstsq(design, response, rcond=1.0e-10)
    return fit[:TASK_DIM], {
        "fit_rank": int(rank),
        "uncensored_fit_count": int(len(values)),
        "detected_count": int(detected_count),
        "censored_detected_count": int(censored_count),
        "fit_rmse_uncensored": float(np.sqrt(np.mean((design @ fit - response) ** 2))),
    }


def _heldout_contrasts(contacts: Sequence[Contact], features: np.ndarray, coefficients: np.ndarray,
                       site_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return truth/prediction design pairs for untouched within-site targets."""

    contrasts: list[np.ndarray] = []
    for site in site_indices:
        # Central checked view at each pose (offset=0, tilt=0) is a public
        # contact; choose the first matching view deterministically.
        central = [row for row in contacts if row.site_index == site and row.view_index == 3]
        if not central:
            continue
        low = next(row for row in central if row.pose_index == 0)
        high = next(row for row in central if row.pose_index == HELDOUT_POSE_INDEX)
        contrasts.append(features[high.index] - features[low.index])
    if not contrasts:
        raise ValueError("heldout split has no central contacts")
    design = np.asarray(contrasts, dtype=float)
    truth_values = design @ coefficients
    return design, truth_values, np.asarray(site_indices, dtype=int)[:design.shape[0]]


def _evaluate_fit(contacts: Sequence[Contact], features: np.ndarray, fit: np.ndarray,
                  truth_subject: Sequence[np.ndarray], heldout_sites: Sequence[int]) -> dict[str, float]:
    errors: list[float] = []
    biases: list[float] = []
    for coefficients in truth_subject:
        design, truth_values, _sites = _heldout_contrasts(contacts, features, coefficients, heldout_sites)
        prediction = design @ fit
        errors.extend((prediction - truth_values).tolist())
        biases.extend((prediction - truth_values).tolist())
    error = np.asarray(errors, dtype=float)
    return {
        "within_site_pose_contrast_rmse": float(np.sqrt(np.mean(error ** 2))),
        "within_site_pose_contrast_bias": float(np.mean(error)),
        "within_site_pose_contrast_abs_bias": float(np.mean(np.abs(error))),
        "heldout_contrast_count": int(error.size),
    }


def run_ablation() -> dict[str, Any]:
    contacts, source = _load_contacts()
    scales = _feature_scales(contacts)
    features = _candidate_features(contacts, scales)
    fishers = _candidate_fishers(contacts, features)
    train_candidates = [row for row in contacts if row.safe and row.pose_index != HELDOUT_POSE_INDEX
                        and row.site_index % 5 != 0]
    if len(train_candidates) <= ATTEMPT_BUDGET:
        raise ValueError("training support is smaller than the common budget")
    train_mask = np.asarray([row.safe and row.pose_index != HELDOUT_POSE_INDEX and row.site_index % 5 != 0
                             for row in contacts], dtype=bool)
    center_selected = _center_only_selection(contacts, ATTEMPT_BUDGET)
    # Restrict the center rule to the same untouched site/pose support.
    if not all(train_mask[i] for i in center_selected):
        center_rows = [i for i in center_selected if train_mask[i]]
        central_pool = [row.index for row in contacts if train_mask[row.index]
                        and abs(row.view_offset_mm) < 1e-9 and abs(row.view_tilt_deg) < 1e-9]
        center_selected = [central_pool[int(i)] for i in np.linspace(0, len(central_pool) - 1, ATTEMPT_BUDGET, dtype=int)]
    random_selected = _uniform_selection([row if train_mask[row.index] else Contact(
        row.index, row.pose_index, row.site_index, row.view_index, row.pose_deg,
        row.lateral_mm, row.depth_mm, row.view_offset_mm, row.view_tilt_deg, False)
        for row in contacts], ATTEMPT_BUDGET, SEED_SELECTION)
    fisher_selected, fisher_audit = _fisher_selection(
        [row if train_mask[row.index] else Contact(
            row.index, row.pose_index, row.site_index, row.view_index, row.pose_deg,
            row.lateral_mm, row.depth_mm, row.view_offset_mm, row.view_tilt_deg, False)
         for row in contacts],
        fishers,
        features,
        ATTEMPT_BUDGET,
    )
    high_quantity_selected, high_quantity_audit = _fisher_selection(
        [row if train_mask[row.index] else Contact(
            row.index, row.pose_index, row.site_index, row.view_index, row.pose_deg,
            row.lateral_mm, row.depth_mm, row.view_offset_mm, row.view_tilt_deg, False)
         for row in contacts],
        fishers,
        features,
        HIGH_QUANTITY_BUDGET,
    )
    selected_by_method = {
        "fisher_teacher": fisher_selected,
        "center_only": center_selected,
        "uniform_random": random_selected,
    }
    if any(len(rows) != ATTEMPT_BUDGET for rows in selected_by_method.values()):
        raise AssertionError("methods do not share the accepted/attempt budget")

    truth = np.asarray([0.20, 0.95, -0.38, 0.17, 0.52], dtype=float)
    heldout_sites = tuple(sorted({row.site_index for row in contacts if row.site_index % 5 == 0}))
    rng_truth = np.random.default_rng(SEED_DATA)
    subject_truth = [_subject_coefficients(rng_truth, subject, truth)[0]
                     for subject in range(N_TRAIN_SUBJECTS + N_HELDOUT_SUBJECTS)]
    metrics: dict[str, Any] = {}
    for method, selected in selected_by_method.items():
        replicate_rows: list[dict[str, Any]] = []
        for replicate in range(N_REPLICATES):
            # Shared replicate seed makes the methods a paired comparison; the
            # method only changes which checked contacts are retained.
            fit_rng = np.random.default_rng(SEED_DATA + 1000 * replicate)
            fit, fit_diag = _fit_selected(contacts, features, selected, fit_rng, truth)
            eval_diag = _evaluate_fit(
                contacts, features, fit,
                subject_truth[N_TRAIN_SUBJECTS:], heldout_sites,
            )
            replicate_rows.append({**fit_diag, **eval_diag})
        metrics[method] = {
            "attempt_budget": ATTEMPT_BUDGET,
            "accepted_checked_contacts": len(selected),
            "selection_indices": [int(i) for i in selected],
            "selection_pose_degrees": [float(contacts[i].pose_deg) for i in selected],
            "selection_lateral_mm": [float(contacts[i].lateral_mm) for i in selected],
            "selection_site_indices": [int(contacts[i].site_index) for i in selected],
            "selection_pose_counts": {
                str(int(pose)): int(sum(contacts[i].pose_index == pose for i in selected))
                for pose in range(4)
            },
            "replicate_metrics": replicate_rows,
            "summary": {
                "within_site_pose_contrast_rmse_mean": float(np.mean([row["within_site_pose_contrast_rmse"] for row in replicate_rows])),
                "within_site_pose_contrast_rmse_sd": float(np.std([row["within_site_pose_contrast_rmse"] for row in replicate_rows], ddof=1)),
                "within_site_pose_contrast_bias_mean": float(np.mean([row["within_site_pose_contrast_bias"] for row in replicate_rows])),
                "detected_count_mean": float(np.mean([row["detected_count"] for row in replicate_rows])),
                "censored_detected_count_mean": float(np.mean([row["censored_detected_count"] for row in replicate_rows])),
            },
        }
    metrics["fisher_teacher"]["selection_audit"] = fisher_audit
    # Same checked support, same Fisher rule, and twice the accepted/attempt
    # quantity.  This is a quantity control, not part of the equal-budget
    # three-way method comparison above.
    high_rows: list[dict[str, Any]] = []
    for replicate in range(N_REPLICATES):
        fit_rng = np.random.default_rng(SEED_DATA + 1000 * replicate)
        fit, fit_diag = _fit_selected(contacts, features, high_quantity_selected, fit_rng, truth)
        eval_diag = _evaluate_fit(contacts, features, fit, subject_truth[N_TRAIN_SUBJECTS:], heldout_sites)
        high_rows.append({**fit_diag, **eval_diag})
    metrics["fisher_teacher_high_quantity"] = {
        "attempt_budget": HIGH_QUANTITY_BUDGET,
        "accepted_checked_contacts": len(high_quantity_selected),
        "same_support_as_primary": True,
        "selection_indices": [int(i) for i in high_quantity_selected],
        "selection_pose_degrees": [float(contacts[i].pose_deg) for i in high_quantity_selected],
        "selection_lateral_mm": [float(contacts[i].lateral_mm) for i in high_quantity_selected],
        "selection_site_indices": [int(contacts[i].site_index) for i in high_quantity_selected],
        "selection_pose_counts": {
            str(int(pose)): int(sum(contacts[i].pose_index == pose for i in high_quantity_selected))
            for pose in range(4)
        },
        "replicate_metrics": high_rows,
        "summary": {
            "within_site_pose_contrast_rmse_mean": float(np.mean([row["within_site_pose_contrast_rmse"] for row in high_rows])),
            "within_site_pose_contrast_rmse_sd": float(np.std([row["within_site_pose_contrast_rmse"] for row in high_rows], ddof=1)),
            "within_site_pose_contrast_bias_mean": float(np.mean([row["within_site_pose_contrast_bias"] for row in high_rows])),
            "detected_count_mean": float(np.mean([row["detected_count"] for row in high_rows])),
            "censored_detected_count_mean": float(np.mean([row["censored_detected_count"] for row in high_rows])),
        },
        "selection_audit": high_quantity_audit,
    }
    result = {
        "schema": "fisher_data_collection_ablation_v1",
        "purpose": "bounded synthetic data usefulness ablation; not a B-completion result",
        "source": source,
        "simulator_assumptions": {
            "response": "signed centroid = [intercept, pose, lateral, depth, pose*lateral] coefficients + subject session zero-offset + Gaussian noise",
            "truth_coefficients": truth.tolist(),
            "subject_response_sd": [0.0, 0.055, 0.055, 0.045, 0.06],
            "session_zero_offset_sd": 1.5,
            "observation_scale": OBSERVATION_SCALE,
            "p_detect": DETECT_PROBABILITY,
            "public_bins": list(BIN_EDGES),
            "fit": "least squares on detected uncensored values with per-training-subject offset columns; misses/censored values are not imputed",
            "heldout": "sites with site_index mod 5 == 0, checked 19-degree pose knot, and two subjects are excluded from selection/fitting",
            "within_site_target": "pose contrast at same site; session offset cancels",
            "mixture_mode_identification": "not_claimed",
        },
        "design": {
            "methods": ["fisher_teacher", "center_only", "uniform_random"],
            "quantity_control": "fisher_teacher_high_quantity",
            "common_support_count": int(np.sum(train_mask)),
            "common_attempt_budget": ATTEMPT_BUDGET,
            "higher_quantity_budget": HIGH_QUANTITY_BUDGET,
            "n_train_subjects": N_TRAIN_SUBJECTS,
            "n_heldout_subjects": N_HELDOUT_SUBJECTS,
            "replicates": N_REPLICATES,
            "selection_seed": SEED_SELECTION,
            "data_seed": SEED_DATA,
            "same_support_and_attempt_budget": True,
        },
        "heldout_sites": list(heldout_sites),
        "methods": metrics,
    }
    return _json_safe(result)


def main() -> None:
    output = HERE / "FISHER_DATA_ABLATION.json"
    payload = run_ablation()
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, row in payload["methods"].items():
        print(name, row["summary"])


if __name__ == "__main__":
    main()

