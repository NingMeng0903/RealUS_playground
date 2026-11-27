#!/usr/bin/env python3
"""Independent formula and artifact checks for the registered mechanism test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_identification as exp


ROOT = Path(__file__).resolve().parent


def independent_f(actions: np.ndarray) -> np.ndarray:
    """Reconstruct F row-wise without calling the implementation builder."""

    rows = []
    for action in np.asarray(actions, dtype=float):
        a0, a1 = action
        rows.append([1, 0, a0, a1, 0, 0, 0, 0, 0, 0, 0, 0])
        rows.append([0, 1, 0, 0, a0, a1, 0, 0, 0, 0, 0, 0])
    return np.asarray(rows, dtype=float)


def independent_g(delta: np.ndarray) -> np.ndarray:
    d0, d1 = np.asarray(delta, dtype=float)
    return np.asarray(
        [
            [0, 0, d0, d1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, d0, d1, 0, 0, 0, 0, 0, 0],
        ],
        dtype=float,
    )


def check() -> None:
    f_no = independent_f(exp.ACTIONS_NO_EXCITATION)
    f_done = independent_f(exp.ACTIONS_COMPLETED)
    assert np.array_equal(f_no, exp.build_f(exp.ACTIONS_NO_EXCITATION))
    assert np.array_equal(f_done, exp.build_f(exp.ACTIONS_COMPLETED))
    assert np.linalg.matrix_rank(f_no, tol=exp.RANK_TOL) == 4
    assert np.linalg.matrix_rank(f_done, tol=exp.RANK_TOL) == 6

    for query_name, delta in exp.QUERIES.items():
        g_ind = independent_g(delta)
        assert np.array_equal(g_ind, exp.build_g(delta))

        # The projection criterion is checked independently from the report.
        for actions, expected in (
            (exp.ACTIONS_NO_EXCITATION, None),
            (exp.ACTIONS_COMPLETED, True),
        ):
            f_mat = independent_f(actions)
            p_row = np.linalg.pinv(f_mat, rcond=1e-12) @ f_mat
            residual = np.linalg.norm(g_ind - g_ind @ p_row)
            if actions is exp.ACTIONS_NO_EXCITATION:
                if query_name == "translation":
                    assert residual <= exp.RANK_TOL
                elif query_name == "rotation":
                    assert residual > 1e-6
            else:
                assert residual <= exp.RANK_TOL

    results = json.loads((ROOT / "RESULTS.json").read_text(encoding="utf-8"))
    assert results["geometry"]["no_excitation"]["translation"][
        "ker_inclusion_holds_at_1e-10"
    ]
    assert not results["geometry"]["no_excitation"]["rotation"][
        "ker_inclusion_holds_at_1e-10"
    ]
    assert results["geometry"]["completed_excitation"]["rotation"][
        "ker_inclusion_holds_at_1e-10"
    ]
    for design in ("no_excitation", "completed_excitation"):
        assert all(
            values["rank_F_full_theta"] < exp.THETA_DIM
            for values in results["geometry"][design].values()
        )

    # Explicit row-wise least squares agrees with the saved fitted estimates.
    raw = np.load(ROOT / "RAW_RESULTS.npz")
    for actions, obs_key, theta_key in (
        (exp.ACTIONS_NO_EXCITATION, "observations_no_excitation_public", "theta_hat_no_excitation"),
        (exp.ACTIONS_COMPLETED, "observations_completed_excitation_public", "theta_hat_completed_excitation"),
    ):
        f_mat = independent_f(actions)
        obs = raw[obs_key]
        expected = np.stack(
            [np.linalg.lstsq(f_mat, y.reshape(-1), rcond=1e-12)[0] for y in obs],
            axis=0,
        )
        assert np.allclose(expected, raw[theta_key], atol=2e-12, rtol=2e-12)

    # Alias construction is exact in the noiseless public no-excitation data.
    alias = results["alias"]
    assert alias["same_first_column"]
    assert alias["no_excitation_noiseless_max_abs_difference"] == 0.0
    assert alias["completed_noiseless_max_abs_difference"] > 0.0
    assert np.isclose(alias["rotation_future_difference_l2"], 0.2)

    # The mechanism result must be visible without relying on a tuned absolute
    # threshold: the completed design reduces the frozen rotation p95 error.
    no_p95 = results["summaries"]["no_excitation"]["rotation"]["l2_p95"]
    done_p95 = results["summaries"]["completed_excitation"]["rotation"]["l2_p95"]
    assert done_p95 < no_p95

    # Independent Monte-Carlo covariance checks should be numerically close;
    # the no-excitation rotation covariance itself is zero while its bias is not.
    for design in results["covariance_validation"].values():
        for cov in design.values():
            assert cov["max_abs_cov_difference"] < 2e-3

    print("independent identifiability checks: PASS")


if __name__ == "__main__":
    check()
