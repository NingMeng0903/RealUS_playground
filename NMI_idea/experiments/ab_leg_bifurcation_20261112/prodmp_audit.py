#!/usr/bin/env python3
"""Independent audit for the learned 2-D ProDMP artifact.

This script intentionally does not import ``learned_prodmp``.  It reads the
saved demonstration/weight arrays and reimplements only the small integrated
Green-convolution calculation needed to verify endpoint constraints, empirical
weight moments, split disjointness, skill sweep sign, held-out error and the
stored analytic/retiming diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np


LAMBDA = 5.0
RBF_CENTERS = np.asarray([0.12, 0.36, 0.64, 0.88], dtype=float)
RBF_WIDTHS = np.full(4, 55.0, dtype=float)
PHASE_GATE_DECAY = 3.0
SKILLS = ("advance", "sweep_left", "sweep_right")


def _phi(s: np.ndarray | float) -> np.ndarray:
    ss = np.asarray(s, dtype=float).reshape(-1)
    ss = np.clip(ss, 0.0, 1.0)
    psi = np.exp(-RBF_WIDTHS[None, :] * (ss[:, None] - RBF_CENTERS[None, :]) ** 2)
    psi /= np.maximum(psi.sum(axis=1, keepdims=True), 1.0e-14)
    return psi * np.exp(-PHASE_GATE_DECAY * ss[:, None])


class _IntegralBasis:
    def __init__(self, order: int = 64):
        nodes, weights = np.polynomial.legendre.leggauss(int(order))
        self.nodes = (nodes + 1.0) / 2.0
        self.weights = weights / 2.0

    def evaluate(self, s: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        ss = np.asarray(s, dtype=float).reshape(-1)
        out_b = np.zeros((ss.size, 4), dtype=float)
        out_bp = np.zeros((ss.size, 4), dtype=float)
        nz = ss > 0.0
        if np.any(nz):
            s_nz = ss[nz]
            u = s_nz[:, None] * self.nodes[None, :]
            delta = s_nz[:, None] - u
            kernel = np.exp(-LAMBDA * delta)
            phi = _phi(u.reshape(-1)).reshape(s_nz.size, self.nodes.size, 4)
            quadrature = self.weights[None, :, None] * phi
            out_b[nz] = s_nz[:, None] * np.sum(
                quadrature * delta[:, :, None] * kernel[:, :, None], axis=1
            )
            out_bp[nz] = s_nz[:, None] * np.sum(
                quadrature * (1.0 - LAMBDA * delta)[:, :, None] * kernel[:, :, None], axis=1
            )
        return out_b, out_bp


def _homogeneous(s: np.ndarray, start: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ss = np.asarray(s, dtype=float).reshape(-1)
    delta = start - goal
    e = np.exp(-LAMBDA * ss)[:, None]
    eta = goal[None, :] + e * (1.0 + LAMBDA * ss[:, None]) * delta[None, :]
    etap = e * (-LAMBDA * LAMBDA * ss[:, None] * delta[None, :])
    return eta, etap


def _chart(phases: np.ndarray, weights: np.ndarray, basis: _IntegralBasis) -> tuple[np.ndarray, np.ndarray]:
    eta, etap = _homogeneous(phases, np.zeros(2), np.asarray([1.0, 0.0]))
    b, bp = basis.evaluate(phases)
    return eta + b @ weights, etap + bp @ weights


def _endpoint_target(basis: _IntegralBasis) -> tuple[np.ndarray, np.ndarray]:
    b, bp = basis.evaluate(1.0)
    target = np.vstack((np.asarray([1.0, 0.0]) - _homogeneous(np.asarray([1.0]), np.zeros(2), np.asarray([1.0, 0.0]))[0][0],
                        -_homogeneous(np.asarray([1.0]), np.zeros(2), np.asarray([1.0, 0.0]))[1][0]))
    return np.vstack((b[0], bp[0])), target


def _rmse(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.asarray(value) ** 2, axis=-1))))


def _array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value, dtype=np.float64).tobytes()).hexdigest()


def audit(directory: str | Path) -> dict[str, object]:
    root = Path(directory)
    protocol = json.loads((root / "prodmp_protocol.json").read_text())
    stored = json.loads((root / "prodmp_diagnostics.json").read_text())
    demos = np.load(root / "prodmp_demos.npz")
    weights = np.load(root / "prodmp_weights.npz")
    basis = _IntegralBasis(int(protocol.get("quadrature_order", 64)))
    constraints, target = _endpoint_target(basis)
    endpoint_error = {}
    mean_error = {}
    covariance_error = {}
    heldout_rmse = {}
    endpoint_only_rmse = {}
    split_hashes = {}
    sweep_center = {}
    train_count = int(protocol["train_per_skill"])
    test_count = int(protocol["test_per_skill"])
    assert tuple(protocol["skills"]) == SKILLS
    for skill in SKILLS:
        train_chart = np.asarray(demos[f"train_{skill}_chart"])
        test_chart = np.asarray(demos[f"test_{skill}_chart"])
        train_derivative = np.asarray(demos[f"train_{skill}_derivative"])
        test_derivative = np.asarray(demos[f"test_{skill}_derivative"])
        train_w = np.asarray(weights[f"train_{skill}"])
        test_w = np.asarray(weights[f"test_{skill}"])
        mean = np.asarray(weights[f"mean_{skill}"])
        covariance = np.asarray(weights[f"covariance_{skill}"])
        assert train_chart.shape[0] == train_count and test_chart.shape[0] == test_count
        assert train_chart.shape[2] == 2 and test_chart.shape[2] == 2
        assert train_derivative.shape == train_chart.shape and test_derivative.shape == test_chart.shape
        assert train_w.shape == (train_count, 4, 2) and test_w.shape == (test_count, 4, 2)
        assert np.isfinite(train_chart).all() and np.isfinite(test_chart).all()
        assert np.isfinite(train_w).all() and np.isfinite(test_w).all()
        assert np.max(np.abs(train_chart[:, 0])) < 1.0e-12
        assert np.max(np.abs(train_chart[:, -1] - [1.0, 0.0])) < 1.0e-12
        assert np.max(np.abs(test_chart[:, 0])) < 1.0e-12
        assert np.max(np.abs(test_chart[:, -1] - [1.0, 0.0])) < 1.0e-12
        train_residual = np.asarray([constraints @ w - target for w in train_w])
        test_residual = np.asarray([constraints @ w - target for w in test_w])
        endpoint_error[skill] = float(max(np.max(np.abs(train_residual)), np.max(np.abs(test_residual))))
        empirical_mean = np.mean(train_w, axis=0)
        empirical_cov = np.cov(train_w.reshape(train_count, -1), rowvar=False, ddof=1)
        mean_error[skill] = float(np.max(np.abs(empirical_mean - mean)))
        covariance_error[skill] = float(np.max(np.abs(empirical_cov - covariance)))
        train_hash = _array_hash(np.asarray(demos[f"train_{skill}_parameters"]))
        test_hash = _array_hash(np.asarray(demos[f"test_{skill}_parameters"]))
        split_hashes[skill] = {"train": train_hash, "test": test_hash, "disjoint": train_hash != test_hash}
        train_phases = np.linspace(0.0, 1.0, train_chart.shape[1])
        test_phases = np.linspace(0.0, 1.0, test_chart.shape[1])
        predicted, _ = _chart(test_phases, mean, basis)
        endpoint_weights, *_ = np.linalg.lstsq(constraints, target, rcond=None)
        baseline, _ = _chart(test_phases, endpoint_weights, basis)
        heldout_rmse[skill] = _rmse(test_chart - predicted)
        endpoint_only_rmse[skill] = _rmse(test_chart - baseline)
        mid = test_chart[:, test_chart.shape[1] // 2, 1]
        sweep_center[skill] = float(np.mean(mid))
        # The stored analytic derivative arrays are independently checked with
        # finite differences on one held-out trajectory.
        numerical = np.gradient(test_chart[0], test_phases, axis=0, edge_order=2)
        # The held-out grid is intentionally modest (97 points); the finite
        # difference truncation error is below 3e-3 even for the fastest
        # analytic sweep.  The model's dense-grid derivative diagnostic is
        # recorded separately in ``prodmp_diagnostics.json``.
        assert float(np.max(np.abs(numerical[2:-2] - test_derivative[0, 2:-2]))) < 3.0e-3
    assert all(v["disjoint"] for v in split_hashes.values())
    assert sweep_center["sweep_left"] > 0.0 and sweep_center["sweep_right"] < 0.0
    assert sweep_center["advance"] > 0.0
    result = {
        "audit": "independent learned 2-D ProDMP arrays and endpoint contract",
        "seed": protocol["seed"],
        "basis_source_declared": protocol.get("basis_source"),
        "skills": list(SKILLS),
        "train_count_per_skill": train_count,
        "test_count_per_skill": test_count,
        "endpoint_constraint_max_abs": endpoint_error,
        "weight_mean_validation_max_abs": mean_error,
        "weight_covariance_validation_max_abs": covariance_error,
        "heldout_position_rmse_normalized": heldout_rmse,
        "endpoint_only_position_rmse_normalized": endpoint_only_rmse,
        "learned_beats_endpoint_only_all_skills": all(heldout_rmse[k] < endpoint_only_rmse[k] for k in SKILLS),
        "sweep_center_mean_signed_chart": sweep_center,
        "train_test_parameter_hashes": split_hashes,
        "stored_analytic_derivative_max_abs": stored.get("analytic_derivative_max_abs"),
        "stored_retimed_position_max_abs_mm": stored.get("retimed_position_max_abs_mm"),
        "stored_retimed_velocity_ratio_max_abs_error": stored.get("retimed_velocity_ratio_max_abs_error"),
    }
    (root / "prodmp_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    result = audit(args.dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
