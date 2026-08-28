#!/usr/bin/env python3
"""Standalone source-string checks for the 2026-08-26 rail/WBC P0 patch."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def old_jointwise_projection(
    jacobian: np.ndarray,
    requested_compensation: np.ndarray,
    q: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    *,
    activation: float = 0.8,
) -> np.ndarray:
    arm_jacobian = jacobian[:, 1:]
    qdot_arm, *_ = np.linalg.lstsq(
        arm_jacobian, requested_compensation, rcond=None
    )
    half = np.maximum(0.5 * (q_hi[1:] - q_lo[1:]), 1.0e-9)
    mid = 0.5 * (q_hi[1:] + q_lo[1:])
    normalized_q = (q[1:] - mid) / half
    mask = (normalized_q * qdot_arm > 0.0) & (
        np.abs(normalized_q) >= activation
    )
    qdot_arm[mask] = 0.0
    return arm_jacobian @ qdot_arm


def update_hysteresis(
    latched: bool, value: float, *, enter: float = 0.15, exit_: float = 0.03
) -> bool:
    if not latched and value >= enter:
        return True
    if latched and value <= exit_:
        return False
    return latched


class ExactRailCompensationTest(unittest.TestCase):
    def test_exact_target_preserves_six_dimensional_task(self) -> None:
        rng = np.random.default_rng(7)
        jacobian = rng.normal(size=(6, 8))
        jacobian[:, 0] = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        v_cmd = np.array([0.04, -0.03, 0.02, 0.0, 0.0, 0.0])
        rail_exec = 0.112
        rail_contribution = jacobian[:, 0] * rail_exec
        arm_target = v_cmd - rail_contribution
        np.testing.assert_allclose(
            rail_contribution + arm_target, v_cmd, atol=1.0e-14
        )
        np.testing.assert_allclose(arm_target[3:], 0.0, atol=1.0e-14)

    def test_removed_projection_can_create_rotation(self) -> None:
        jacobian = np.zeros((6, 8))
        jacobian[:, 0] = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        jacobian[:, 1] = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        jacobian[:, 2] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        q_lo = -np.ones(8)
        q_hi = np.ones(8)
        q = np.zeros(8)
        q[1] = 0.9
        rail_exec = -0.1
        rail_contribution = jacobian[:, 0] * rail_exec
        projected = old_jointwise_projection(
            jacobian, -rail_contribution, q, q_lo, q_hi
        )
        physical_error = rail_contribution + projected
        self.assertGreater(abs(float(physical_error[5])), 0.05)


class SlackHoldTest(unittest.TestCase):
    def test_enter_exit_pair_does_not_chatter(self) -> None:
        values = [0.14, 0.151, 0.149, 0.10, 0.031, 0.029]
        states: list[bool] = []
        latched = False
        for value in values:
            latched = update_hysteresis(latched, value)
            states.append(latched)
        self.assertEqual(states, [False, True, True, True, True, False])


class PatchedSourceTest(unittest.TestCase):
    def test_native_uses_exact_arm_target(self) -> None:
        source = (ROOT / "native/wbc_rt/src/inner.cpp").read_text()
        solve = source[source.index("bool InnerLoop::solve_hqp") :]
        solve = solve[: solve.index("Vec8 lo_box")]
        self.assertIn("b_task = v_cmd - rail_contrib;", solve)
        self.assertNotIn("project_arm_compensation", solve)

    def test_python_uses_exact_arm_target(self) -> None:
        source = (
            ROOT
            / "rm75_control/control/joint_admittance_8dof/solver/qp_builder.py"
        ).read_text()
        self.assertIn("b_task = v_cmd0 - rail_exec_contrib", source)
        self.assertNotIn("project_arm_compensation(", source)

    def test_slack_hold_has_hysteresis_and_skips_planner_tick(self) -> None:
        cpp = (ROOT / "native/wbc_rt/src/inner.cpp").read_text()
        py = (
            ROOT / "rm75_control/control/joint_admittance_8dof/loop.py"
        ).read_text()
        self.assertIn("secondary_alpha=float(secondary_alpha)", py)
        self.assertNotIn("const bool posture_hold = slack_high || task_hold;", cpp)
        self.assertIn("slack_now <= slack_exit", py)
        self.assertNotIn(
            "quiescent=bool(self._quiescent or slack_high)", py
        )

    def test_stop_clears_slack_latch(self) -> None:
        cpp = (ROOT / "native/wbc_rt/src/inner.cpp").read_text()
        cpp_stop = cpp[cpp.index("void InnerLoop::stop()") :]
        cpp_stop = cpp_stop[: cpp_stop.index("void InnerLoop::reset")]
        py = (
            ROOT / "rm75_control/control/joint_admittance_8dof/loop.py"
        ).read_text()
        py_stop = py[py.index("    def stop(self) -> None:") :]
        py_stop = py_stop[: py_stop.index("    def _update_quiescent")]
        self.assertIn("slack_hold_latched_ = false", cpp_stop)
        self.assertIn("self._slack_hold_latched = False", py_stop)

    def test_sigma_barrier_uses_sigma_arm_argument(self) -> None:
        cpp = (ROOT / "native/wbc_rt/src/inner.cpp").read_text()
        self.assertIn("double sigma_arm,", cpp)
        self.assertIn(
            "pref_lo[pref_n] = -cfg_.sigma_gamma * (sigma_arm - cfg_.sigma_safe);",
            cpp,
        )
        self.assertIn("sigma_tick_ == 0 || sigma_activated_edge", cpp)
        reset = cpp[cpp.index("void InnerLoop::reset") :]
        reset = reset[: reset.index("void InnerLoop::begin_hybrid")]
        self.assertIn("sigma_row_active_ = false;", reset)
        self.assertIn("sigma_grad_.setZero();", reset)
        self.assertIn("sigma_tick_ = 0;", reset)


if __name__ == "__main__":
    unittest.main(verbosity=2)
