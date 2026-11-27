#!/usr/bin/env python3
"""Registered abstract local action-response identifiability experiment.

This module deliberately contains no ultrasound, robot, dIRD, or neural
network code.  It is a small mechanism test for whether a frozen local
response model can answer a future action query from public action/observation
pairs.  See PROTOCOL.txt for the registered design and interpretation limits.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parent
SEED = 720940
N_SUBJECTS = 512
N_NUISANCE = 6
THETA_DIM = 6 + N_NUISANCE
SIGMA = 0.02
RANK_TOL = 1e-10
N_COV_REPS = 4096
ALLOWED_ACTION_LIMIT = 2.0
CONTROLLER_TARGET_LIMIT = 0.3
CONTROLLER_THRESHOLD = 0.08

ACTIONS_NO_EXCITATION = np.asarray(
    [[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [0.5, 0.0]], dtype=float
)
ACTIONS_COMPLETED = np.asarray(
    [[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float
)
QUERIES = {
    "translation": np.asarray([0.5, 0.0], dtype=float),
    "rotation": np.asarray([0.0, 0.5], dtype=float),
    "combined": np.asarray([0.35, -0.40], dtype=float),
}


def rotation(angle: float) -> np.ndarray:
    """2-D active rotation matrix."""

    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.asarray([[c, -s], [s, c]], dtype=float)


def build_f(actions: np.ndarray) -> np.ndarray:
    """Build the public raw-y design F for theta=(c,J,zeta)."""

    actions = np.asarray(actions, dtype=float)
    rows = []
    for a0, a1 in actions:
        row0 = np.zeros(THETA_DIM, dtype=float)
        row0[0] = 1.0
        row0[2] = a0
        row0[3] = a1
        rows.append(row0)

        row1 = np.zeros(THETA_DIM, dtype=float)
        row1[1] = 1.0
        row1[4] = a0
        row1[5] = a1
        rows.append(row1)
    return np.asarray(rows, dtype=float)


def build_g(delta_action: np.ndarray) -> np.ndarray:
    """Build the query matrix G for Delta y=J delta_action."""

    d0, d1 = np.asarray(delta_action, dtype=float)
    rows = np.zeros((2, THETA_DIM), dtype=float)
    rows[0, 2] = d0
    rows[0, 3] = d1
    rows[1, 4] = d0
    rows[1, 5] = d1
    return rows


def sample_subjects(rng: np.random.Generator, n: int) -> Tuple[np.ndarray, ...]:
    """Draw allowed J, c, and nuisance parameters."""

    phi = rng.uniform(-0.8, 0.8, size=n)
    psi = rng.uniform(-0.8, 0.8, size=n)
    g1 = rng.uniform(0.7, 1.3, size=n)
    g2 = rng.uniform(0.7, 1.3, size=n)
    c = rng.uniform(-10.0, 10.0, size=(n, 2))
    nuisance = rng.uniform(-1.0, 1.0, size=(n, N_NUISANCE))

    jacobians = np.empty((n, 2, 2), dtype=float)
    for i in range(n):
        jacobians[i] = (
            rotation(phi[i])
            @ np.diag([g1[i], g2[i]])
            @ rotation(psi[i])
        )

    theta = np.zeros((n, THETA_DIM), dtype=float)
    theta[:, 0:2] = c
    theta[:, 2:6] = jacobians.reshape(n, 4)
    theta[:, 6:] = nuisance
    return theta, c, jacobians, nuisance


def generate_observations(
    theta: np.ndarray,
    actions: np.ndarray,
    noise: np.ndarray,
) -> np.ndarray:
    """Generate public raw observations, in (subject, action, output) order."""

    c = theta[:, 0:2]
    jacobians = theta[:, 2:6].reshape(-1, 2, 2)
    actions = np.asarray(actions, dtype=float)
    clean = c[:, None, :] + np.einsum("nij,kj->nki", jacobians, actions)
    return clean + np.asarray(noise, dtype=float)


def fit_minimum_norm_ls(actions: np.ndarray, observations: np.ndarray) -> np.ndarray:
    """Fit theta using only public actions and observations."""

    f_mat = build_f(actions)
    f_pinv = np.linalg.pinv(f_mat, rcond=1e-12)
    y_flat = np.asarray(observations, dtype=float).reshape(observations.shape[0], -1)
    return y_flat @ f_pinv.T


def extract_j(theta_hat: np.ndarray) -> np.ndarray:
    return np.asarray(theta_hat[:, 2:6], dtype=float).reshape(-1, 2, 2)


def query_predictions(jacobians: np.ndarray, delta_action: np.ndarray) -> np.ndarray:
    return np.einsum("nij,j->ni", jacobians, np.asarray(delta_action, dtype=float))


def metric_summary(error: np.ndarray) -> Dict[str, object]:
    """Summarize vector prediction errors without dropping any subject."""

    error = np.asarray(error, dtype=float)
    l2 = np.linalg.norm(error, axis=1)
    linf = np.max(np.abs(error), axis=1)
    return {
        "n": int(error.shape[0]),
        "component_mean_signed": error.mean(axis=0).tolist(),
        "component_rmse": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "component_mae": np.mean(np.abs(error), axis=0).tolist(),
        "l2_mean": float(l2.mean()),
        "l2_rmse": float(np.sqrt(np.mean(l2**2))),
        "l2_median": float(np.median(l2)),
        "l2_p95": float(np.quantile(l2, 0.95)),
        "l2_max": float(l2.max()),
        "linf_mean": float(linf.mean()),
        "linf_p95": float(np.quantile(linf, 0.95)),
        "linf_max": float(linf.max()),
    }


def rank(matrix: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(np.asarray(matrix), tol=RANK_TOL))


def query_geometry(f_mat: np.ndarray, g_mat: np.ndarray) -> Dict[str, object]:
    """Return the exact finite-dimensional query identifiability diagnostics."""

    f_pinv = np.linalg.pinv(f_mat, rcond=1e-12)
    p_row = f_pinv @ f_mat
    complement = np.eye(THETA_DIM) - p_row
    identifiable_part = g_mat @ p_row
    unknown_part = g_mat @ complement
    residual = g_mat - identifiable_part
    # F x=0 implies P_F x=0; therefore residual=0 is exactly the tested
    # ker(F) subseteq ker(G) condition up to floating point tolerance.
    return {
        "rank_F_full_theta": rank(f_mat),
        "rank_G": rank(g_mat),
        "rank_G_identifiable_part": rank(identifiable_part),
        "rank_G_unknown_part": rank(unknown_part),
        "ker_inclusion_residual_fro": float(np.linalg.norm(residual)),
        "ker_inclusion_holds_at_1e-10": bool(np.linalg.norm(residual) <= RANK_TOL),
        "f_shape": list(f_mat.shape),
        "g_shape": list(g_mat.shape),
    }


def covariance_validation(
    theta_fixed: np.ndarray,
    actions: np.ndarray,
    query_name: str,
    delta_action: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, object]:
    """Compare centered estimator-noise covariance with the registered formula."""

    f_mat = build_f(actions)
    g_mat = build_g(delta_action)
    f_pinv = np.linalg.pinv(f_mat, rcond=1e-12)
    # Independent noise replicates; truth is fixed, so this is a covariance
    # test of the estimator rather than a mixture over subjects.
    eps = rng.normal(0.0, SIGMA, size=(N_COV_REPS, f_mat.shape[0]))
    y_clean = f_mat @ theta_fixed
    theta_hats = (y_clean[None, :] + eps) @ f_pinv.T
    query_errors = theta_hats @ g_mat.T - (g_mat @ theta_fixed)[None, :]
    centered = query_errors - query_errors.mean(axis=0, keepdims=True)
    empirical_cov = centered.T @ centered / N_COV_REPS
    theoretical_cov = SIGMA**2 * g_mat @ np.linalg.pinv(
        f_mat.T @ f_mat, rcond=1e-12
    ) @ g_mat.T
    diff = empirical_cov - theoretical_cov
    return {
        "n_replicates": N_COV_REPS,
        "query": query_name,
        "empirical_centered_cov": empirical_cov.tolist(),
        "theoretical_cov": theoretical_cov.tolist(),
        "max_abs_cov_difference": float(np.max(np.abs(diff))),
        "fro_cov_difference": float(np.linalg.norm(diff)),
        "mean_query_error_including_bias": query_errors.mean(axis=0).tolist(),
        "note": (
            "For an unidentifiable query, the centered covariance tests only "
            "estimator noise; the nonzero mean is missing-direction bias."
        ),
    }


def controller_results(
    theta_hats: np.ndarray,
    true_jacobians: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, object]:
    """Score the registered task-increment controller using truth only here."""

    j_hats = extract_j(theta_hats)
    pseudo_inverses = np.linalg.pinv(j_hats, rcond=1e-12)
    proposed_actions = np.einsum("nij,nj->ni", pseudo_inverses, targets)
    clipped_actions = np.clip(
        proposed_actions, -ALLOWED_ACTION_LIMIT, ALLOWED_ACTION_LIMIT
    )
    realized = np.einsum("nij,nj->ni", true_jacobians, clipped_actions)
    residual = targets - realized
    successful = np.max(np.abs(residual), axis=1) <= CONTROLLER_THRESHOLD
    return {
        "n": int(targets.shape[0]),
        "target_limit": CONTROLLER_TARGET_LIMIT,
        "action_clip": ALLOWED_ACTION_LIMIT,
        "success_threshold_linf": CONTROLLER_THRESHOLD,
        "success_count": int(np.sum(successful)),
        "success_rate": float(np.mean(successful)),
        "clip_any_count": int(np.sum(np.any(proposed_actions != clipped_actions, axis=1))),
        "clip_any_rate": float(np.mean(np.any(proposed_actions != clipped_actions, axis=1))),
        "residual_metrics": metric_summary(residual),
        "residual_component_p95": np.quantile(np.abs(residual), 0.95, axis=0).tolist(),
    }


def alias_check() -> Dict[str, object]:
    """Construct the registered same-first-column alias pair."""

    actions_no = ACTIONS_NO_EXCITATION
    actions_done = ACTIONS_COMPLETED
    c = np.zeros(2, dtype=float)
    nuisance = np.zeros(N_NUISANCE, dtype=float)
    j_plus = np.diag([1.0, 0.8])
    j_minus = np.diag([1.0, 1.2])
    theta_plus = np.r_[c, j_plus.reshape(-1), nuisance]
    theta_minus = np.r_[c, j_minus.reshape(-1), nuisance]
    f_no = build_f(actions_no)
    f_done = build_f(actions_done)
    y_plus_no = f_no @ theta_plus
    y_minus_no = f_no @ theta_minus
    y_plus_done = f_done @ theta_plus
    y_minus_done = f_done @ theta_minus
    g_rotation = build_g(QUERIES["rotation"])
    future_plus = g_rotation @ theta_plus
    future_minus = g_rotation @ theta_minus
    return {
        "j_plus": j_plus.tolist(),
        "j_minus": j_minus.tolist(),
        "same_first_column": bool(np.allclose(j_plus[:, 0], j_minus[:, 0])),
        "no_excitation_noiseless_max_abs_difference": float(
            np.max(np.abs(y_plus_no - y_minus_no))
        ),
        "completed_noiseless_max_abs_difference": float(
            np.max(np.abs(y_plus_done - y_minus_done))
        ),
        "rotation_future_plus": future_plus.tolist(),
        "rotation_future_minus": future_minus.tolist(),
        "rotation_future_difference": (future_plus - future_minus).tolist(),
        "rotation_future_difference_l2": float(np.linalg.norm(future_plus - future_minus)),
        "statement": (
            "The no-excitation public distributions are identical under the "
            "same independent Gaussian noise law; the coupled noiseless check "
            "is exactly zero."
        ),
    }


def physical_component_errors(error: np.ndarray) -> np.ndarray:
    """Map normalized y errors to (mm, degrees) for readability."""

    return np.asarray(error) * np.asarray([2.0, 2.0])


def run() -> Dict[str, object]:
    rng = np.random.default_rng(SEED)
    theta, c, true_jacobians, nuisance = sample_subjects(rng, N_SUBJECTS)
    # Four observations per design.  The same noise draw is used at each
    # action index (paired design comparison), and remains public only as y.
    observation_noise = rng.normal(0.0, SIGMA, size=(N_SUBJECTS, 4, 2))
    targets = rng.uniform(
        -CONTROLLER_TARGET_LIMIT,
        CONTROLLER_TARGET_LIMIT,
        size=(N_SUBJECTS, 2),
    )
    obs_no = generate_observations(theta, ACTIONS_NO_EXCITATION, observation_noise)
    obs_done = generate_observations(theta, ACTIONS_COMPLETED, observation_noise)

    designs = {
        "no_excitation": (ACTIONS_NO_EXCITATION, obs_no),
        "completed_excitation": (ACTIONS_COMPLETED, obs_done),
    }
    fitted: Dict[str, np.ndarray] = {}
    predictions: Dict[str, Dict[str, np.ndarray]] = {}
    errors: Dict[str, Dict[str, np.ndarray]] = {}
    summaries: Dict[str, Dict[str, object]] = {}
    geometry: Dict[str, Dict[str, object]] = {}
    controllers: Dict[str, Dict[str, object]] = {}

    for design_name, (actions, observations) in designs.items():
        f_mat = build_f(actions)
        theta_hat = fit_minimum_norm_ls(actions, observations)
        fitted[design_name] = theta_hat
        geometry[design_name] = {}
        predictions[design_name] = {}
        errors[design_name] = {}
        summaries[design_name] = {}

        j_hat = extract_j(theta_hat)
        for query_name, delta_action in QUERIES.items():
            g_mat = build_g(delta_action)
            geometry[design_name][query_name] = query_geometry(f_mat, g_mat)
            pred = query_predictions(j_hat, delta_action)
            truth = query_predictions(true_jacobians, delta_action)
            err = pred - truth
            predictions[design_name][query_name] = pred
            errors[design_name][query_name] = err
            summary = metric_summary(err)
            summary["component_rmse_physical_mm_or_deg"] = physical_component_errors(
                np.sqrt(np.mean(err**2, axis=0))
            ).tolist()
            summaries[design_name][query_name] = summary

        controllers[design_name] = controller_results(theta_hat, true_jacobians, targets)

        # Fixed J=I baseline has no fitted response parameters.  It is scored
        # on the same truth and queries but is separate from the LS learner.
        fixed_errors = {}
        for query_name, delta_action in QUERIES.items():
            pred = np.asarray(delta_action, dtype=float)[None, :].repeat(N_SUBJECTS, axis=0)
            truth = query_predictions(true_jacobians, delta_action)
            fixed_errors[query_name] = pred - truth

        summaries[design_name]["fixed_J_identity"] = {}
        for query_name, err in fixed_errors.items():
            summary = metric_summary(err)
            summary["component_rmse_physical_mm_or_deg"] = physical_component_errors(
                np.sqrt(np.mean(err**2, axis=0))
            ).tolist()
            summaries[design_name]["fixed_J_identity"][query_name] = summary

    # The covariance validation uses the first registered subject as a fixed
    # allowed theta/c and a fresh independent RNG stream, not main test data.
    covariance_rng = np.random.default_rng(SEED + 1)
    covariance = {}
    for design_name, actions in {
        "no_excitation": ACTIONS_NO_EXCITATION,
        "completed_excitation": ACTIONS_COMPLETED,
    }.items():
        covariance[design_name] = {}
        for query_name, delta_action in QUERIES.items():
            covariance[design_name][query_name] = covariance_validation(
                theta[0], actions, query_name, delta_action, covariance_rng
            )

    result = {
        "protocol": {
            "seed": SEED,
            "n_subjects": N_SUBJECTS,
            "sigma": SIGMA,
            "n_nuisance": N_NUISANCE,
            "theta_dim": THETA_DIM,
            "actions_no_excitation": ACTIONS_NO_EXCITATION.tolist(),
            "actions_completed_excitation": ACTIONS_COMPLETED.tolist(),
            "queries": {k: v.tolist() for k, v in QUERIES.items()},
            "controller_threshold": CONTROLLER_THRESHOLD,
        },
        "geometry": geometry,
        "summaries": summaries,
        "controller": controllers,
        "covariance_validation": covariance,
        "alias": alias_check(),
        "sample_summary": {
            "c_min": c.min(axis=0).tolist(),
            "c_max": c.max(axis=0).tolist(),
            "nuisance_min": nuisance.min(axis=0).tolist(),
            "nuisance_max": nuisance.max(axis=0).tolist(),
            "jacobian_fro_min": np.linalg.norm(true_jacobians, axis=(1, 2)).min().item(),
            "jacobian_fro_max": np.linalg.norm(true_jacobians, axis=(1, 2)).max().item(),
        },
    }

    raw_arrays = {
        "theta_truth_audit_only": theta,
        "c_truth_audit_only": c,
        "jacobians_truth_audit_only": true_jacobians,
        "nuisance_truth_audit_only": nuisance,
        "observation_noise_audit_only": observation_noise,
        "targets_public": targets,
        "actions_no_excitation": ACTIONS_NO_EXCITATION,
        "actions_completed_excitation": ACTIONS_COMPLETED,
        "observations_no_excitation_public": obs_no,
        "observations_completed_excitation_public": obs_done,
        "F_no_excitation": build_f(ACTIONS_NO_EXCITATION),
        "F_completed_excitation": build_f(ACTIONS_COMPLETED),
        "theta_hat_no_excitation": fitted["no_excitation"],
        "theta_hat_completed_excitation": fitted["completed_excitation"],
        "true_jacobians_audit_only": true_jacobians,
    }
    for query_name, delta_action in QUERIES.items():
        raw_arrays[f"G_{query_name}"] = build_g(delta_action)
    for design_name in designs:
        j_hat = extract_j(fitted[design_name])
        raw_arrays[f"j_hat_{design_name}"] = j_hat
        raw_arrays[f"controller_actions_{design_name}"] = np.clip(
            np.einsum("nij,nj->ni", np.linalg.pinv(j_hat, rcond=1e-12), targets),
            -ALLOWED_ACTION_LIMIT,
            ALLOWED_ACTION_LIMIT,
        )
        for query_name in QUERIES:
            raw_arrays[f"prediction_{design_name}_{query_name}"] = predictions[design_name][query_name]
            raw_arrays[f"error_{design_name}_{query_name}"] = errors[design_name][query_name]

    np.savez_compressed(ROOT / "RAW_RESULTS.npz", **raw_arrays)
    with (ROOT / "RESULTS.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    report = make_report(result)
    (ROOT / "REPORT.txt").write_text(report, encoding="utf-8")
    return result


def fmt(x: float) -> str:
    return f"{x:.6g}"


def make_report(result: Dict[str, object]) -> str:
    lines = [
        "Local action-response query-identifiability report",
        "===================================================",
        "",
        "Scope: abstract y=c+J a mechanism only; not real ultrasound, complete A, B, dIRD, or robot validation.",
        f"Registered seed={SEED}, subjects={N_SUBJECTS}, sigma={SIGMA}, all cases retained.",
        "",
        "1. Nullspace/query geometry",
        "----------------------------",
    ]
    geometry = result["geometry"]
    for design_name, queries in geometry.items():
        lines.append(f"{design_name}:")
        for query_name, values in queries.items():
            lines.append(
                "  "
                + query_name
                + ": rank(F)="
                + str(values["rank_F_full_theta"])
                + ", rank(GP)="
                + str(values["rank_G_identifiable_part"])
                + ", rank(G(I-P))="
                + str(values["rank_G_unknown_part"])
                + ", residual="
                + fmt(values["ker_inclusion_residual_fro"])
                + ", ker inclusion="
                + str(values["ker_inclusion_holds_at_1e-10"])
            )

    lines.extend(["", "2. Frozen query prediction (normalized y units)", "-----------------------------------------------"])
    summaries = result["summaries"]
    for design_name, queries in summaries.items():
        lines.append(f"{design_name}:")
        lines.append("  minimum_norm_ls:")
        for query_name in QUERIES:
            values = queries[query_name]
            lines.append(
                f"    {query_name}: l2_rmse={fmt(values['l2_rmse'])}, "
                f"l2_p95={fmt(values['l2_p95'])}, linf_p95={fmt(values['linf_p95'])}, "
                f"component_rmse={','.join(fmt(v) for v in values['component_rmse'])}"
            )
        lines.append("  fixed_J_identity:")
        for query_name in QUERIES:
            values = queries["fixed_J_identity"][query_name]
            lines.append(
                f"    {query_name}: l2_rmse={fmt(values['l2_rmse'])}, "
                f"l2_p95={fmt(values['l2_p95'])}, linf_p95={fmt(values['linf_p95'])}, "
                f"component_rmse={','.join(fmt(v) for v in values['component_rmse'])}"
            )

    lines.extend(["", "3. Registered task-increment controller", "--------------------------------------"])
    for design_name, values in result["controller"].items():
        lines.append(
            f"{design_name}: success={values['success_count']}/{values['n']} "
            f"({fmt(values['success_rate'])}), "
            f"clip_any={values['clip_any_count']}/{values['n']}"
        )

    lines.extend(["", "4. Independent covariance check", "-------------------------------"])
    for design_name, queries in result["covariance_validation"].items():
        lines.append(f"{design_name}:")
        for query_name, values in queries.items():
            lines.append(
                f"  {query_name}: max_abs_cov_difference="
                f"{fmt(values['max_abs_cov_difference'])}, "
                f"mean_error(including bias)="
                f"{','.join(fmt(v) for v in values['mean_query_error_including_bias'])}"
            )

    alias = result["alias"]
    lines.extend(
        [
            "",
            "5. Alias-pair check",
            "-------------------",
            "J+=diag(1,0.8), J-=diag(1,1.2): same first column and both allowed.",
            "No-excitation noiseless public max difference="
            + fmt(alias["no_excitation_noiseless_max_abs_difference"])
            + "; completed-excitation max difference="
            + fmt(alias["completed_noiseless_max_abs_difference"])
            + ".",
            "Rotation future difference="
            + ",".join(fmt(v) for v in alias["rotation_future_difference"])
            + " (l2="
            + fmt(alias["rotation_future_difference_l2"])
            + ").",
        ]
    )
    lines.extend(
        [
            "",
            "Interpretation",
            "--------------",
            "The no-excitation design identifies translation but leaves the rotation query in an unobserved Jacobian column; the alias pair makes this failure unavoidable from its public data.  Adding one orthogonal action makes the registered rotation and combined queries identifiable and should reduce their frozen-query error.",
            "The completed design is a batch least-squares/RLS-equivalent mechanism, not evidence that a neural network is necessary.  The experiment establishes only a necessary local excitation/query condition; it does not establish cross-person transfer, belief updating, real ultrasound, or executable robot motion.",
        ]
    )
    return "\n".join(lines) + "\n"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest() -> None:
    files = [
        "PROTOCOL.txt",
        "run_identification.py",
        "test_identification.py",
        "RAW_RESULTS.npz",
        "RESULTS.json",
        "REPORT.txt",
    ]
    manifest = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "seed": SEED,
        "files_sha256": {name: file_sha256(ROOT / name) for name in files},
    }
    (ROOT / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    run()
    # Manifest is written only after results/report are complete.  The source
    # and protocol hashes therefore cover the exact registered implementation.
    write_manifest()
