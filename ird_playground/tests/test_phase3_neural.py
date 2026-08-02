"""Phase-3 neural / calib wiring: splits, conformal, tolerance, input stats."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ird_playground.calib.conformal import (
    false_acceptance_report,
    fit_unreachable_safety_threshold,
    fit_zero_bias,
    calibrated_clearance,
    empirical_coverage,
    fit_split_conformal,
)
from ird_playground.ird.canonical import FLANGE_CANONICAL_DIM
from ird_playground.ird.metric import unit_speed_eikonal_loss
from ird_playground.neural.signed_field import (
    SignedReachabilityField,
    assert_fitted_normalization,
    compute_input_stats,
)
from ird_playground.neural.tolerance_field import (
    RHO_DIM,
    ToleranceConditionedField,
    build_rho_descriptor,
    rho_zero_consistency_loss,
)
from ird_playground.neural.train_signed import (
    EXTERNAL_SPLIT_REQUIRED_KEYS,
    _split_indices,
    evaluate_signed_field,
    external_holdout_indices,
    require_source_pose_id,
)


def test_compute_input_stats_rejects_identity_defaults():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, FLANGE_CANONICAL_DIM)).astype(np.float32)
    x[:, 0] += 0.4
    x[:, 1] = np.abs(x[:, 1]) + 0.05
    center, scale = compute_input_stats(x)
    assert center.shape == (FLANGE_CANONICAL_DIM,)
    assert scale.shape == (FLANGE_CANONICAL_DIM,)
    assert_fitted_normalization(center, scale)
    with pytest.raises(ValueError, match="forbidden"):
        assert_fitted_normalization(
            np.zeros(FLANGE_CANONICAL_DIM), np.ones(FLANGE_CANONICAL_DIM)
        )


def test_split_requires_source_pose_id():
    boundary = np.full(8, -1, dtype=np.int64)
    with pytest.raises(KeyError, match="source_pose_id"):
        require_source_pose_id({"boundary_id": boundary, "canonical": np.zeros((8, 9))})
    with pytest.raises(KeyError, match="source_pose_id"):
        _split_indices(boundary, 0.25, seed=1, source_pose_id=None)  # type: ignore[arg-type]


def test_external_holdout_by_block_and_orbit_docs():
    n = 40
    block = np.repeat(np.arange(8, dtype=np.int64), 5)
    arrays = {
        "block_id": block,
        "q_best": np.linspace(0, 1, n * 7).reshape(n, 7),
    }
    keep, hold = external_holdout_indices(arrays, mode="block", fraction=0.25, seed=3)
    assert set(block[keep]).isdisjoint(set(block[hold]))
    assert len(keep) + len(hold) == n

    keep2, hold2 = external_holdout_indices(arrays, mode="orbit", fraction=0.25, seed=3)
    assert len(keep2) + len(hold2) == n
    assert "block" in EXTERNAL_SPLIT_REQUIRED_KEYS
    assert "orbit" in EXTERNAL_SPLIT_REQUIRED_KEYS

    with pytest.raises(KeyError, match="block_id"):
        external_holdout_indices({"q_best": arrays["q_best"]}, mode="block", fraction=0.2, seed=0)


def test_split_conformal_threshold_and_coverage():
    rng = np.random.default_rng(11)
    # Separable scores with a little noise.
    y = rng.random(400) > 0.45
    scores = np.where(y, rng.normal(0.4, 0.15, size=y.size), rng.normal(-0.4, 0.15, size=y.size))
    result = fit_split_conformal(scores[:200], y[:200], alpha=0.1)
    assert np.isfinite(result.threshold)
    assert result.n_calib == 200
    cal = calibrated_clearance(scores[200:], result.threshold)
    assert cal.shape == (200,)
    cov = empirical_coverage(scores[200:], y[200:], result.threshold)
    assert cov["n"] == 200.0
    # With margin=0 the conservative rule should still recover most reachables.
    assert cov["coverage_reachable"] >= 0.7


def test_independent_zero_and_unreachable_calibration():
    zero = fit_zero_bias(np.array([0.09, 0.10, 0.11]), bootstrap_samples=50, seed=2)
    assert zero.zero_bias == pytest.approx(0.10)
    unreachable = np.linspace(-1.0, 0.2, 100)
    safety = fit_unreachable_safety_threshold(unreachable, alpha=0.05)
    assert safety.safety_threshold >= np.quantile(unreachable, 0.94)
    report = false_acceptance_report(
        np.r_[unreachable, [0.5, 0.6]],
        np.r_[np.zeros(100, dtype=bool), np.ones(2, dtype=bool)],
        safety.safety_threshold,
    )
    assert report["false_accept_rate_upper"] >= report["false_accept_rate"]


def test_tolerance_rho_zero_consistency_loss_finite():
    center = np.linspace(-0.2, 0.5, FLANGE_CANONICAL_DIM).astype(np.float32)
    scale = np.full(FLANGE_CANONICAL_DIM, 0.3, dtype=np.float32)
    base = SignedReachabilityField(
        width=32, depth=2, fourier_bands=1, input_center=center, input_scale=scale
    )
    tol = ToleranceConditionedField(
        width=32,
        depth=2,
        fourier_bands=1,
        input_center=center,
        input_scale=scale,
        require_fitted_chart_stats=True,
    )
    x = torch.randn(16, FLANGE_CANONICAL_DIM)
    with torch.no_grad():
        teacher = base(x)
    loss = rho_zero_consistency_loss(tol, teacher, x)
    assert torch.isfinite(loss)
    loss.backward()
    rho = build_rho_descriptor(box_btn=(0.01, 0.02, 0.03), cvar_level=0.1)
    assert rho.shape == (RHO_DIM,)
    y = tol(x, torch.as_tensor(rho).expand(16, -1))
    assert y.shape == (16,)
    assert torch.isfinite(y).all()


def test_empirical_boundary_slope_and_near_axis_eval_hook():
    g = torch.tensor([[3.0, 4.0], [0.0, 1.0]], requires_grad=False)
    target = torch.tensor([5.0, 1.0]) * 100.0
    loss = unit_speed_eikonal_loss(g * 100.0, target_slope=target)
    assert float(loss) == pytest.approx(0.0)

    center = np.linspace(-0.1, 0.4, FLANGE_CANONICAL_DIM).astype(np.float32)
    scale = np.full(FLANGE_CANONICAL_DIM, 0.25, dtype=np.float32)
    model = SignedReachabilityField(
        width=24, depth=2, fourier_bands=1, input_center=center, input_scale=scale
    )
    from ird_playground.neural.signed_field import ReachabilitySDF

    field = ReachabilitySDF(model, device="cpu")
    n = 40
    canonical = np.zeros((n, FLANGE_CANONICAL_DIM), dtype=np.float32)
    canonical[:, 1] = np.linspace(0.01, 0.2, n)  # r
    arrays = {
        "canonical": canonical,
        "reachable": (canonical[:, 1] < 0.1).astype(np.float32),
        "classification_weight": np.ones(n, dtype=np.float32),
        "boundary_id": np.full(n, -1, dtype=np.int64),
        "clearance_kind": np.full(n, -1, dtype=np.int8),
        "boundary_signed_m": np.full(n, np.nan, dtype=np.float32),
        "boundary_signed_rot_deg": np.full(n, np.nan, dtype=np.float32),
    }
    metrics = evaluate_signed_field(field, arrays, np.arange(n), report_near_axis=True)
    assert metrics["near_axis_r_lt_5cm_n"] > 0
    assert "near_axis_r_lt_5cm_accuracy" in metrics
