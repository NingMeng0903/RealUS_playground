#!/usr/bin/env python3
"""Independent checks for the post-hoc baseline addendum."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import run_addendum as add


ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parent
if str(ORIGINAL) not in sys.path:
    sys.path.insert(0, str(ORIGINAL))
import run_identification as original  # noqa: E402


def independent_f6(actions: np.ndarray) -> np.ndarray:
    rows = []
    for a0, a1 in np.asarray(actions, dtype=float):
        rows.append([1.0, 0.0, a0, a1, 0.0, 0.0])
        rows.append([0.0, 1.0, 0.0, 0.0, a0, a1])
    return np.asarray(rows, dtype=float)


def independent_g6(delta: np.ndarray) -> np.ndarray:
    d0, d1 = np.asarray(delta, dtype=float)
    return np.asarray(
        [[0.0, 0.0, d0, d1, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, d0, d1]],
        dtype=float,
    )


def check() -> None:
    results = json.loads((ROOT / "RESULTS.json").read_text(encoding="utf-8"))
    raw = np.load(ROOT / "RAW_ADDENDUM.npz")
    assert results["status"] == "posthoc_baseline_addendum"
    assert results["protocol"]["n_fixed_test"] == 512
    assert results["protocol"]["n_train"] == 4096

    # Original registered files are byte-for-byte unchanged.
    old_manifest = json.loads((ORIGINAL / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, expected in old_manifest["files_sha256"].items():
        digest = hashlib.sha256((ORIGINAL / name).read_bytes()).hexdigest()
        assert digest == expected, name

    assert np.array_equal(
        independent_f6(original.ACTIONS_NO_EXCITATION),
        add.build_f6(original.ACTIONS_NO_EXCITATION),
    )
    assert np.array_equal(
        independent_f6(original.ACTIONS_COMPLETED),
        add.build_f6(original.ACTIONS_COMPLETED),
    )
    for delta in original.QUERIES.values():
        assert np.array_equal(independent_g6(delta), add.build_g6(delta))

    # Public-data LS used to learn the shared prior is reproducible without
    # accessing the saved training truth arrays.
    f_train = independent_f6(original.ACTIONS_COMPLETED)
    y_train = raw["observations_train_public"]
    train_ls = np.stack(
        [np.linalg.lstsq(f_train, y.reshape(-1), rcond=1e-12)[0] for y in y_train],
        axis=0,
    )
    assert np.allclose(train_ls, raw["theta_hat_train_from_public"], atol=2e-12, rtol=2e-12)
    train_j = train_ls[:, 2:6]
    assert np.allclose(train_j.mean(axis=0), raw["shared_prior_mean"][2:6])
    cov = np.cov(train_j, rowvar=False, ddof=1) + add.JITTER * np.eye(4)
    assert np.allclose(cov, raw["shared_prior_cov"][2:6, 2:6])

    # Recompute the ridge normal equations independently for representative
    # subjects in each design.
    prior_precision = np.diag([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    prior_mean = np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    for design, actions, obs in (
        ("no_excitation", original.ACTIONS_NO_EXCITATION, raw["test_observations_no_excitation_public"]),
        ("completed_excitation", original.ACTIONS_COMPLETED, raw["test_observations_completed_excitation_public"]),
    ):
        f = independent_f6(actions)
        normal = f.T @ f + add.RIDGE_LAMBDA * prior_precision
        for idx in (0, 17, 511):
            y = obs[idx].reshape(-1)
            expected = np.linalg.solve(
                normal, f.T @ y + add.RIDGE_LAMBDA * prior_precision @ prior_mean
            )
            assert np.allclose(
                expected,
                raw[f"theta_hat_ridge_identity_{design}"][idx],
                atol=2e-12,
                rtol=2e-12,
            )

    # Recompute one shared-prior posterior independently and verify the fixed
    # orthogonal completion rule exactly.
    prior_mean_shared = raw["shared_prior_mean"]
    prior_cov_shared = raw["shared_prior_cov"]
    for design, actions, obs in (
        ("no_excitation", original.ACTIONS_NO_EXCITATION, raw["test_observations_no_excitation_public"]),
        ("completed_excitation", original.ACTIONS_COMPLETED, raw["test_observations_completed_excitation_public"]),
    ):
        f = independent_f6(actions)
        precision = np.linalg.inv(prior_cov_shared)
        normal = f.T @ f / original.SIGMA**2 + precision
        covariance = np.linalg.inv(normal)
        idx = 3
        expected = covariance @ (
            f.T @ obs[idx].reshape(-1) / original.SIGMA**2 + precision @ prior_mean_shared
        )
        assert np.allclose(
            expected,
            raw[f"theta_hat_shared_gaussian_{design}"][idx],
            atol=2e-11,
            rtol=2e-11,
        )

        first = raw[f"theta_hat_ls6_{design}"][idx, 2:6].reshape(2, 2)[:, 0]
        expected_ortho = np.column_stack([first, [-first[1], first[0]]])
        assert np.allclose(
            expected_ortho,
            raw[f"j_hat_orthogonality_completion_{design}"][idx],
            atol=2e-12,
            rtol=2e-12,
        )

    # Every baseline retained all 512 cases, and the unobserved-direction
    # coverage is present as a measured statistic rather than inferred from
    # posterior variance alone.
    for design in ("no_excitation", "completed_excitation"):
        for model in ("ridge_identity", "shared_gaussian", "orthogonality_completion"):
            assert results["models"][design][model]["controller"]["n"] == 512
        cov = results["models"][design]["shared_gaussian"]["coverage"]["rotation"]
        assert "marginal_95_coverage_by_component" in cov
        assert "joint_95_ellipsoid_coverage" in cov

    print("independent addendum checks: PASS")


if __name__ == "__main__":
    check()
