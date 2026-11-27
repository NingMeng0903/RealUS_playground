"""Focused checks for the context-conditioned Fisher ProDMP fit."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


MODULE_PATH = Path(__file__).with_name("demo_fit.py")
SPEC = importlib.util.spec_from_file_location("fisher_demo_fit_test_module", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_stationary_one_axis_prediction_keeps_neutral_endpoints() -> None:
    phases = np.linspace(0.0, 1.0, 61)
    context = np.linspace(-1.0, 1.0, 10)[:, None]
    trajectories = np.zeros((10, phases.size, 1), dtype=float)
    for row, value in enumerate(context[:, 0]):
        trajectories[row, :, 0] = 0.12 * value * np.sin(np.pi * phases) ** 2

    result = MODULE.fit_and_evaluate(
        trajectories[:8],
        context[:8],
        context[8:],
        heldout_trajectories=trajectories[8:],
        heldout_phases=phases,
    )

    assert result["trajectories"].shape == (2, phases.size, 1)
    np.testing.assert_allclose(result["trajectories"][:, 0, :], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(result["trajectories"][:, -1, :], 0.0, atol=1.0e-12)
    assert result["endpoint_errors"]["max_position_error"] == 0.0
    assert result["endpoint_only_endpoint_errors"]["max_position_error"] == 0.0


def test_three_axis_heldout_endpoints_and_endpoint_reference() -> None:
    phases = np.linspace(0.0, 1.0, 41)
    context = np.column_stack((np.linspace(-1.0, 1.0, 8), np.linspace(1.0, -1.0, 8)))
    starts = np.column_stack((np.linspace(0.0, 1.0, 8), np.zeros(8), np.full(8, -0.2)))
    goals = starts + np.column_stack((np.full(8, 0.8), np.full(8, -0.25), np.full(8, 0.4)))
    trajectories = starts[:, None, :] + (goals - starts)[:, None, :] * phases[None, :, None]
    trajectories += 0.04 * context[:, :1, None] * np.sin(np.pi * phases)[None, :, None] ** 2

    result = MODULE.fit_and_evaluate(
        trajectories[:6],
        context[:6],
        context[6:],
        heldout_starts=starts[6:],
        heldout_goals=goals[6:],
        heldout_trajectories=trajectories[6:],
        heldout_phases=phases,
    )

    assert result["trajectories"].shape == (2, phases.size, 3)
    np.testing.assert_allclose(result["trajectories"][:, 0, :], starts[6:], atol=1.0e-12)
    np.testing.assert_allclose(result["trajectories"][:, -1, :], goals[6:], atol=1.0e-12)
    np.testing.assert_allclose(
        result["endpoint_only_trajectories"][:, [0, -1], :],
        np.stack((starts[6:], goals[6:]), axis=1),
        atol=1.0e-12,
    )
    assert result["fit_checks"]["all_finite"]
    assert result["endpoint_errors"]["max_endpoint_velocity_error"] < 1.0e-8


def test_ten_basis_symmetric_diagnostic_meets_angle_tolerance() -> None:
    diagnostic = MODULE.symmetric_demo_diagnostic(n_basis=10)

    assert diagnostic["phase_1_3_error_deg"] <= 0.65
    assert diagnostic["phase_2_3_error_deg"] <= 0.65
    assert diagnostic["neutral_endpoint_error_deg"] == 0.0
    assert diagnostic["passes_0.65deg"]
