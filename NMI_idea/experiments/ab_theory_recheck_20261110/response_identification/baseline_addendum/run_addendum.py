#!/usr/bin/env python3
"""Post-hoc conservative baselines for the registered response mechanism.

The parent directory is read-only from this script's perspective.  All new
artifacts are written below baseline_addendum/.  See PROTOCOL.txt: this is a
diagnostic added after the original minimum-norm result, not a prospective
replacement for it.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parent
if str(ORIGINAL) not in sys.path:
    sys.path.insert(0, str(ORIGINAL))
import run_identification as original  # noqa: E402


TRAIN_SEED = 720941
N_TRAIN = 4096
RIDGE_LAMBDA = original.SIGMA**2
J_PRIOR_IDENTITY = np.eye(2, dtype=float)
JITTER = 1e-8
C_PRIOR_VARIANCE = 1e12
Z95 = 1.959963984540054
CHI2_2_95 = -2.0 * np.log(0.05)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_original_hashes() -> Dict[str, str]:
    """Read the original manifest and verify every registered artifact first."""

    manifest_path = ORIGINAL / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed = {}
    for name, expected in manifest["files_sha256"].items():
        path = ORIGINAL / name
        got = sha256(path)
        if got != expected:
            raise RuntimeError(f"original artifact hash changed before addendum: {name}")
        observed[name] = got
    return observed


def build_f6(actions: np.ndarray) -> np.ndarray:
    """Independent six-parameter public design (c0,c1,J00,J01,J10,J11)."""

    rows = []
    for a0, a1 in np.asarray(actions, dtype=float):
        rows.append([1.0, 0.0, a0, a1, 0.0, 0.0])
        rows.append([0.0, 1.0, 0.0, 0.0, a0, a1])
    return np.asarray(rows, dtype=float)


def build_g6(delta_action: np.ndarray) -> np.ndarray:
    d0, d1 = np.asarray(delta_action, dtype=float)
    return np.asarray(
        [[0.0, 0.0, d0, d1, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, d0, d1]],
        dtype=float,
    )


def fit_ls6(actions: np.ndarray, observations: np.ndarray) -> np.ndarray:
    f_mat = build_f6(actions)
    f_pinv = np.linalg.pinv(f_mat, rcond=1e-12)
    y_flat = np.asarray(observations, dtype=float).reshape(observations.shape[0], -1)
    return y_flat @ f_pinv.T


def ridge_identity_fit(actions: np.ndarray, observations: np.ndarray) -> np.ndarray:
    """MAP estimator for the fixed J=I prior, with broad independent c."""

    f_mat = build_f6(actions)
    prior_mean = np.zeros(6, dtype=float)
    prior_mean[2] = 1.0
    prior_mean[5] = 1.0
    prior_precision = np.zeros((6, 6), dtype=float)
    prior_precision[2, 2] = 1.0
    prior_precision[3, 3] = 1.0
    prior_precision[4, 4] = 1.0
    prior_precision[5, 5] = 1.0
    # This is ||F x-y||^2 + sigma^2 ||J-I||^2 after multiplying the
    # registered objective by sigma^2.  c has no penalty.
    normal = f_mat.T @ f_mat + RIDGE_LAMBDA * prior_precision
    rhs_prior = RIDGE_LAMBDA * prior_precision @ prior_mean
    y_flat = np.asarray(observations, dtype=float).reshape(observations.shape[0], -1)
    rhs = y_flat @ f_mat + rhs_prior[None, :]
    return np.linalg.solve(normal.T, rhs.T).T


def fit_gaussian_posterior(
    actions: np.ndarray,
    observations: np.ndarray,
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Posterior means/covariance for a frozen shared Gaussian J prior."""

    f_mat = build_f6(actions)
    prior_precision = np.linalg.inv(prior_cov)
    normal = f_mat.T @ f_mat / original.SIGMA**2 + prior_precision
    covariance = np.linalg.inv(normal)
    y_flat = np.asarray(observations, dtype=float).reshape(observations.shape[0], -1)
    rhs = y_flat @ f_mat / original.SIGMA**2 + prior_mean @ prior_precision.T
    means = rhs @ covariance.T
    return means, covariance


def orthogonality_completion(theta_hat6: np.ndarray) -> np.ndarray:
    """Use first J column and a fixed positive 90-degree equal-norm column."""

    j_first = np.asarray(theta_hat6[:, 2:6], dtype=float).reshape(-1, 2, 2)[:, :, 0]
    j_second = np.stack([-j_first[:, 1], j_first[:, 0]], axis=1)
    j_hat = np.stack([j_first, j_second], axis=2)
    return j_hat


def extract_j(theta_hat6: np.ndarray) -> np.ndarray:
    return np.asarray(theta_hat6[:, 2:6], dtype=float).reshape(-1, 2, 2)


def prediction_metrics(error: np.ndarray) -> Dict[str, object]:
    return original.metric_summary(np.asarray(error, dtype=float))


def controller_score(
    j_hat: np.ndarray, true_j: np.ndarray, targets: np.ndarray
) -> Dict[str, object]:
    """Score only here with hidden truth; the estimator never receives true_j."""

    proposed = np.einsum("nij,nj->ni", np.linalg.pinv(j_hat, rcond=1e-12), targets)
    action = np.clip(proposed, -original.ALLOWED_ACTION_LIMIT, original.ALLOWED_ACTION_LIMIT)
    realized = np.einsum("nij,nj->ni", true_j, action)
    residual = targets - realized
    success = np.max(np.abs(residual), axis=1) <= original.CONTROLLER_THRESHOLD
    return {
        "n": int(len(targets)),
        "success_count": int(success.sum()),
        "success_rate": float(success.mean()),
        "clip_any_count": int(np.any(proposed != action, axis=1).sum()),
        "clip_any_rate": float(np.any(proposed != action, axis=1).mean()),
        "residual_metrics": prediction_metrics(residual),
        "actions": action,
        "proposed_actions": proposed,
        "residual": residual,
        "success": success,
    }


def coverage_for_query(
    posterior_mean: np.ndarray,
    posterior_cov: np.ndarray,
    true_j: np.ndarray,
    delta_action: np.ndarray,
) -> Dict[str, object]:
    """Evaluate marginal and 2-D ellipsoid coverage against hidden test truth."""

    g_mat = build_g6(delta_action)
    pred = posterior_mean @ g_mat.T
    truth = np.einsum("nij,j->ni", true_j, delta_action)
    error = pred - truth
    query_cov = g_mat @ posterior_cov @ g_mat.T
    query_cov = 0.5 * (query_cov + query_cov.T)
    std = np.sqrt(np.maximum(np.diag(query_cov), 0.0))
    lower = pred - Z95 * std
    upper = pred + Z95 * std
    marginal = (truth >= lower) & (truth <= upper)
    # The joint ellipsoid uses the nominal chi-square(2) 95% threshold.
    inv_cov = np.linalg.pinv(query_cov, rcond=1e-12)
    mahal = np.einsum("ni,ij,nj->n", error, inv_cov, error)
    joint_ellipsoid = mahal <= CHI2_2_95
    return {
        "posterior_query_mean_across_test": pred.mean(axis=0).tolist(),
        "posterior_query_cov": query_cov.tolist(),
        "posterior_query_std": std.tolist(),
        "marginal_95_coverage_by_component": marginal.mean(axis=0).tolist(),
        "joint_95_ellipsoid_coverage": float(joint_ellipsoid.mean()),
        "marginal_95_all_components_coverage": float(marginal.all(axis=1).mean()),
        "mahalanobis_p95": float(np.quantile(mahal, 0.95)),
        "query_error_metrics": prediction_metrics(error),
        "error": error,
        "prediction": pred,
        "truth": truth,
    }


def train_shared_prior() -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Learn only from public full-action training observations."""

    rng = np.random.default_rng(TRAIN_SEED)
    theta_train, c_train, j_train, nuisance_train = original.sample_subjects(rng, N_TRAIN)
    noise_train = rng.normal(
        0.0, original.SIGMA, size=(N_TRAIN, len(original.ACTIONS_COMPLETED), 2)
    )
    observations_train = original.generate_observations(
        theta_train, original.ACTIONS_COMPLETED, noise_train
    )
    theta_hat_train = fit_ls6(original.ACTIONS_COMPLETED, observations_train)
    j_hat_train = extract_j(theta_hat_train)
    j_flat = j_hat_train.reshape(N_TRAIN, 4)
    prior_mean_j = j_flat.mean(axis=0)
    prior_cov_j = np.cov(j_flat, rowvar=False, ddof=1)
    prior_cov_j = 0.5 * (prior_cov_j + prior_cov_j.T) + JITTER * np.eye(4)
    prior_mean = np.r_[0.0, 0.0, prior_mean_j]
    prior_cov = np.zeros((6, 6), dtype=float)
    prior_cov[0:2, 0:2] = C_PRIOR_VARIANCE * np.eye(2)
    prior_cov[2:6, 2:6] = prior_cov_j
    arrays = {
        "theta_train_truth_audit_only": theta_train,
        "c_train_truth_audit_only": c_train,
        "jacobians_train_truth_audit_only": j_train,
        "nuisance_train_truth_audit_only": nuisance_train,
        "noise_train_audit_only": noise_train,
        "observations_train_public": observations_train,
        "theta_hat_train_from_public": theta_hat_train,
        "j_hat_train_from_public": j_hat_train,
    }
    details = {
        "seed": TRAIN_SEED,
        "n_train": N_TRAIN,
        "prior_mean_j": prior_mean_j.tolist(),
        "prior_cov_j": prior_cov_j.tolist(),
        "prior_cov_eigenvalues": np.linalg.eigvalsh(prior_cov_j).tolist(),
        "prior_cov_jitter": JITTER,
        "c_prior_variance": C_PRIOR_VARIANCE,
        "learned_from": "public full-action observations via LS only; no GT J input",
    }
    arrays["shared_prior_mean"] = prior_mean
    arrays["shared_prior_cov"] = prior_cov
    return arrays, {"arrays": arrays, "prior_mean": prior_mean, "prior_cov": prior_cov, "details": details}


def load_fixed_test() -> Dict[str, np.ndarray]:
    """Load the exact original 512 cases without regenerating or modifying them."""

    raw = np.load(ORIGINAL / "RAW_RESULTS.npz")
    out = {
        "theta": raw["theta_truth_audit_only"],
        "true_j": raw["jacobians_truth_audit_only"],
        "obs_no": raw["observations_no_excitation_public"],
        "obs_done": raw["observations_completed_excitation_public"],
        "targets": raw["targets_public"],
    }
    if out["true_j"].shape[0] != original.N_SUBJECTS:
        raise RuntimeError("original test subject count changed")
    return out


def json_safe_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    return {k: v for k, v in metrics.items() if k not in {"actions", "proposed_actions", "residual", "success", "error", "prediction", "truth"}}


def run() -> Dict[str, object]:
    original_hashes_before = verify_original_hashes()
    test = load_fixed_test()
    train_arrays, train_info = train_shared_prior()
    prior_mean = train_info["prior_mean"]
    prior_cov = train_info["prior_cov"]

    designs = {
        "no_excitation": (original.ACTIONS_NO_EXCITATION, test["obs_no"]),
        "completed_excitation": (original.ACTIONS_COMPLETED, test["obs_done"]),
    }
    model_outputs: Dict[str, Dict[str, object]] = {}
    raw_arrays: Dict[str, np.ndarray] = {
        **train_arrays,
        "test_theta_truth_audit_only": test["theta"],
        "test_true_jacobians_audit_only": test["true_j"],
        "test_targets_public": test["targets"],
        "test_observations_no_excitation_public": test["obs_no"],
        "test_observations_completed_excitation_public": test["obs_done"],
    }

    for design_name, (actions, observations) in designs.items():
        ls6 = fit_ls6(actions, observations)
        ridge6 = ridge_identity_fit(actions, observations)
        gaussian_mean, gaussian_cov = fit_gaussian_posterior(
            actions, observations, prior_mean, prior_cov
        )
        ortho_j = orthogonality_completion(ls6)
        models = {
            "ridge_identity": (extract_j(ridge6), ridge6, None),
            "shared_gaussian": (extract_j(gaussian_mean), gaussian_mean, gaussian_cov),
            "orthogonality_completion": (ortho_j, ls6, None),
        }
        model_outputs[design_name] = {}
        raw_arrays[f"theta_hat_ls6_{design_name}"] = ls6
        raw_arrays[f"theta_hat_ridge_identity_{design_name}"] = ridge6
        raw_arrays[f"theta_hat_shared_gaussian_{design_name}"] = gaussian_mean
        raw_arrays[f"j_hat_orthogonality_completion_{design_name}"] = ortho_j

        for model_name, (j_hat, theta_hat, posterior_cov) in models.items():
            model_record: Dict[str, object] = {"queries": {}, "controller": None}
            for query_name, delta in original.QUERIES.items():
                prediction = original.query_predictions(j_hat, delta)
                truth = original.query_predictions(test["true_j"], delta)
                error = prediction - truth
                model_record["queries"][query_name] = {
                    "metrics": prediction_metrics(error),
                    "prediction": prediction,
                    "error": error,
                }
                raw_arrays[f"prediction_{model_name}_{design_name}_{query_name}"] = prediction
                raw_arrays[f"error_{model_name}_{design_name}_{query_name}"] = error

            controller = controller_score(j_hat, test["true_j"], test["targets"])
            model_record["controller"] = controller
            raw_arrays[f"controller_actions_{model_name}_{design_name}"] = controller["actions"]
            raw_arrays[f"controller_proposed_actions_{model_name}_{design_name}"] = controller[
                "proposed_actions"
            ]
            raw_arrays[f"controller_residual_{model_name}_{design_name}"] = controller[
                "residual"
            ]
            raw_arrays[f"controller_success_{model_name}_{design_name}"] = controller[
                "success"
            ]

            if model_name == "shared_gaussian":
                model_record["posterior_covariance"] = posterior_cov
                model_record["coverage"] = {}
                for query_name, delta in original.QUERIES.items():
                    coverage = coverage_for_query(
                        theta_hat, posterior_cov, test["true_j"], delta
                    )
                    model_record["coverage"][query_name] = coverage
                    raw_arrays[f"posterior_cov_{design_name}_{query_name}"] = (
                        build_g6(delta) @ posterior_cov @ build_g6(delta).T
                    )
                    raw_arrays[f"coverage_prediction_{design_name}_{query_name}"] = coverage[
                        "prediction"
                    ]
                    raw_arrays[f"coverage_error_{design_name}_{query_name}"] = coverage["error"]

            model_outputs[design_name][model_name] = model_record

    original_results = json.loads((ORIGINAL / "RESULTS.json").read_text(encoding="utf-8"))
    # Convert arrays into a compact JSON summary while preserving all arrays in
    # RAW_ADDENDUM.npz for audit/reproduction.
    summary_outputs = {}
    for design_name, models in model_outputs.items():
        summary_outputs[design_name] = {}
        for model_name, record in models.items():
            summary_outputs[design_name][model_name] = {
                "queries": {
                    q: {"metrics": m["metrics"]} for q, m in record["queries"].items()
                },
                "controller": json_safe_metrics(record["controller"]),
            }
            if "coverage" in record:
                summary_outputs[design_name][model_name]["coverage"] = {
                    q: {
                        k: v
                        for k, v in cov.items()
                        if k not in {"error", "prediction", "truth"}
                    }
                    for q, cov in record["coverage"].items()
                }

    result = {
        "status": "posthoc_baseline_addendum",
        "original_artifacts_unchanged_before_run": original_hashes_before,
        "protocol": {
            "original_seed": original.SEED,
            "n_fixed_test": original.N_SUBJECTS,
            "train_seed": TRAIN_SEED,
            "n_train": N_TRAIN,
            "ridge_lambda": RIDGE_LAMBDA,
            "shared_prior_jitter": JITTER,
            "c_prior_variance": C_PRIOR_VARIANCE,
            "controller_threshold": original.CONTROLLER_THRESHOLD,
            "queries": {k: v.tolist() for k, v in original.QUERIES.items()},
        },
        "shared_prior_training": train_info["details"],
        "models": summary_outputs,
        "original_registered_reference": {
            "summaries": original_results["summaries"],
            "controller": original_results["controller"],
        },
        "interpretation": {
            "ridge_identity": "Uses fitted observed directions and J=I only for unobserved directions; it is a conservative prior baseline.",
            "shared_gaussian": "Uses an independent 4096-subject public-data-fitted response prior; posterior coverage is evaluated against held-out test truth.",
            "orthogonality_completion": "Fixed +90 degree equal-norm geometric completion, not learned and not a proof of response transfer.",
            "limits": "All results are abstract local-linear diagnostics, not real ultrasound or complete A/B validation.",
        },
    }

    np.savez_compressed(ROOT / "RAW_ADDENDUM.npz", **raw_arrays)
    (ROOT / "RESULTS.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    (ROOT / "REPORT.txt").write_text(make_report(result), encoding="utf-8")

    # The original hashes are checked again after all writes, proving this
    # addendum did not modify any registered artifact.
    original_hashes_after = verify_original_hashes()
    result["original_artifacts_unchanged_after_run"] = original_hashes_after
    (ROOT / "RESULTS.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    (ROOT / "REPORT.txt").write_text(make_report(result), encoding="utf-8")
    return result


def fmt(value: float) -> str:
    return f"{float(value):.6g}"


def make_report(result: Dict[str, object]) -> str:
    lines = [
        "Post-hoc conservative baseline addendum",
        "=======================================",
        "",
        "Scope: abstract local y=c+J a mechanism; original registered artifacts are unchanged.",
        "This addendum was added after the original minimum-norm result and is not a prospective test.",
        f"Fixed test cases: {result['protocol']['n_fixed_test']}; independent prior training subjects: {result['protocol']['n_train']}; train seed={TRAIN_SEED}.",
        "",
        "1. Frozen query prediction (normalized y units)",
        "-----------------------------------------------",
    ]
    for design_name, models in result["models"].items():
        lines.append(f"{design_name}:")
        for model_name, record in models.items():
            lines.append(f"  {model_name}:")
            for query_name, q in record["queries"].items():
                metrics = q["metrics"]
                lines.append(
                    f"    {query_name}: l2_rmse={fmt(metrics['l2_rmse'])}, "
                    f"l2_p95={fmt(metrics['l2_p95'])}, linf_p95={fmt(metrics['linf_p95'])}"
                )
            ctrl = record["controller"]
            lines.append(
                f"    controller: success={ctrl['success_count']}/{ctrl['n']} "
                f"({fmt(ctrl['success_rate'])}), clip_any={ctrl['clip_any_count']}/{ctrl['n']}"
            )
            if "coverage" in record:
                lines.append("    shared-prior coverage:")
                for query_name, coverage in record["coverage"].items():
                    lines.append(
                        f"      {query_name}: marginal95="
                        f"{','.join(fmt(v) for v in coverage['marginal_95_coverage_by_component'])}, "
                        f"all-marginal={fmt(coverage['marginal_95_all_components_coverage'])}, "
                        f"joint-ellipsoid95={fmt(coverage['joint_95_ellipsoid_coverage'])}, "
                        f"std={','.join(fmt(v) for v in coverage['posterior_query_std'])}"
                    )
        lines.append("")

    prior = result["shared_prior_training"]
    lines.extend(
        [
            "2. Shared prior learned from public training observations",
            "--------------------------------------------------------",
            "mu_J=" + ",".join(fmt(v) for v in prior["prior_mean_j"]),
            "diag(Sigma_J)=" + ",".join(fmt(prior["prior_cov_j"][i][i]) for i in range(4)),
            "eigenvalues(Sigma_J)=" + ",".join(fmt(v) for v in prior["prior_cov_eigenvalues"]),
            "",
            "3. Interpretation",
            "-----------------",
            "The J=I ridge baseline tests whether the original no-excitation failure was partly caused by the minimum-norm zero completion. The shared Gaussian baseline tests whether a response prior learned from independent public data can fill an unobserved direction; held-out prediction and interval coverage, especially no-excitation rotation, are the relevant evidence.",
            "A high posterior confidence with poor held-out coverage is overconfidence, not success. Even good abstract baseline performance is only evidence about this linear mechanism and does not establish a complete A skill, B active observation, real ultrasound transfer, or executable robot behavior.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_manifest() -> None:
    names = [
        "PROTOCOL.txt",
        "run_addendum.py",
        "test_addendum.py",
        "RAW_ADDENDUM.npz",
        "RESULTS.json",
        "REPORT.txt",
    ]
    manifest = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "original_directory": str(ORIGINAL),
        "original_manifest_hashes_verified": True,
        "files_sha256": {name: sha256(ROOT / name) for name in names},
    }
    (ROOT / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    run()
    write_manifest()
