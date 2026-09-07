"""Regression checks for bounded, symmetric tool-y moment balance."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import unittest

import numpy as np
import yaml

from peirastic.realman8dof.force.protocol import ForceOutput
from peirastic.realman8dof.force.torque_tilt import (
    LegacyForceWithTilt,
    TorqueTilt,
    TorqueTiltConfig,
    apply_tilt_selection,
    estimate_contact_cop,
    remap_force_along_world_normal,
    rotation_from_pose,
)

DT = 0.005
DESIRED = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
THETA_MAX = TorqueTiltConfig().theta_max_rad
CAP20 = replace(TorqueTiltConfig(), theta_max_rad=math.radians(20.0), cop_stall_s=0.0)


def law(**kwargs) -> TorqueTilt:
    return TorqueTilt(replace(TorqueTiltConfig(), **kwargs))


def step(tilt, tau, *, fz=2.0, contact=True, desired=DESIRED, dt=DT, **kwargs):
    wrench = np.array([0.0, 0.0, fz, 0.0, tau, 0.0])
    return tilt.update(wrench, desired, dt_s=dt, contact=contact, **kwargs)


class TestTorqueTilt(unittest.TestCase):
    def test_small_noise_sticks_but_both_sides_above_threshold_yield(self):
        for tau in (-0.025, -0.01, 0.0, 0.01, 0.025):
            tilt = law()
            for _ in range(80):
                self.assertEqual(step(tilt, tau), 0.0)
        for tau in (-0.03, 0.03):
            tilt = law()
            output = [step(tilt, tau) for _ in range(100)]
            self.assertTrue(all(w * tau < 0.0 for w in output))
            self.assertGreater(abs(output[-1]), 0.01)

    def test_positive_and_negative_histories_are_exact_mirrors(self):
        positive, negative = law(), law()
        torques = np.repeat([0.01, 0.03, 0.08, -0.02, -0.06, 0.15, 0.0], 70)
        for tau in torques:
            plus = step(positive, tau)
            minus = step(negative, -tau)
            self.assertAlmostEqual(plus, -minus, places=14)
        self.assertAlmostEqual(positive.theta_tilt, -negative.theta_tilt, places=14)

    def test_normal_force_error_cannot_suppress_real_moment(self):
        nominal, excess, deficient = law(), law(), law()
        for _ in range(100):
            full = step(nominal, 0.12)
            self.assertEqual(full, step(excess, 0.12, fz=12.0))
            self.assertEqual(full, step(deficient, 0.12, fz=-1.0))
        self.assertLess(full, -0.1)

    def test_contact_rising_edge_zeros_integral_then_cap_blocks_that_sign(self):
        tilt = TorqueTilt(CAP20)
        for _ in range(400):
            output = step(tilt, 0.12)
        self.assertAlmostEqual(abs(tilt.theta_tilt), CAP20.theta_max_rad, places=6)
        self.assertTrue(tilt.tilt_capped)
        self.assertEqual(output, 0.0)
        step(tilt, 0.12, contact=False)
        output = step(tilt, 0.12, contact=True)
        self.assertLess(abs(tilt.theta_tilt), 0.05)
        self.assertLess(output, 0.0)

    def test_cap_is_one_sided_so_opposite_moment_can_unwind(self):
        tilt = TorqueTilt(CAP20)
        for _ in range(400):
            step(tilt, 0.12)
        self.assertLess(tilt.theta_tilt, -0.3)
        for _ in range(80):
            output = step(tilt, -0.12)
        self.assertGreater(output, 0.0)
        self.assertGreater(tilt.theta_tilt, -CAP20.theta_max_rad + 0.05)

    def test_rotational_contact_spring_balances_inside_cap(self):
        tilt = law()
        theta, surface_angle, stiffness = 0.0, 0.20, 0.25
        for _ in range(3000):
            tau = stiffness * (theta - surface_angle)
            theta += DT * step(tilt, tau)
        self.assertGreater(theta, 0.08)
        self.assertLessEqual(theta, surface_angle)
        self.assertLessEqual(abs(stiffness * (theta - surface_angle)), 0.025 + 1e-5)

    def test_rotational_contact_spring_stops_at_small_cap_before_far_surface(self):
        tilt = TorqueTilt(CAP20)
        theta, surface_angle, stiffness = 0.0, 1.0, 0.25
        for _ in range(3000):
            tau = stiffness * (theta - surface_angle)
            theta += DT * step(tilt, tau)
        self.assertLessEqual(abs(tilt.theta_tilt), CAP20.theta_max_rad + 1e-9)
        self.assertLess(theta, 0.40)
        self.assertGreater(abs(stiffness * (theta - surface_angle)), 0.10)

    def test_wrap_around_reaches_large_surface_angle_without_world_freeze(self):
        tilt = law()
        theta, surface_angle, stiffness = 0.0, 1.0, 0.25
        twisted = np.array([0.0, 0.0, 0.3, 0.0, -1.16, 0.0])
        for _ in range(4000):
            tau = stiffness * (theta - surface_angle)
            theta += DT * step(tilt, tau, pose=twisted, slack_norm=0.0)
        self.assertGreater(theta, 0.70)
        self.assertFalse(tilt.tilt_frozen)
        self.assertLessEqual(abs(stiffness * (theta - surface_angle)), 0.025 + 1e-5)

    def test_slack_freezes_omega_but_world_tilt_does_not(self):
        tilt = law()
        twisted = np.array([0.0, 0.0, 0.3, 0.0, -1.16, 0.0])
        self.assertLess(step(tilt, 0.12, pose=twisted), 0.0)
        self.assertFalse(tilt.tilt_frozen)
        tilt.reset()
        upright = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0])
        self.assertLess(step(tilt, 0.12, pose=upright, slack_norm=0.01), 0.0)
        self.assertFalse(tilt.tilt_frozen)
        tilt.reset()
        self.assertEqual(step(tilt, 0.12, pose=upright, slack_norm=0.12), 0.0)
        self.assertTrue(tilt.tilt_frozen)

    def test_reversal_brakes_immediately_and_changes_direction(self):
        tilt = law()
        for _ in range(3):
            step(tilt, 0.5)
        before = tilt.omega_y
        self.assertGreater(step(tilt, -0.04), before)
        for _ in range(100):
            output = step(tilt, -0.04)
        self.assertGreater(output, 0.02)

    def test_velocity_and_acceleration_limits_with_large_moments(self):
        cfg = replace(TorqueTiltConfig(), a_max=0.8, vmax_rad_s=0.12, cop_stall_s=0.0)
        tilt = TorqueTilt(cfg)
        previous = 0.0
        for tau in np.repeat([4.0, -4.0, 0.0, 4.0], 100):
            output = step(tilt, tau)
            self.assertLessEqual(abs(output), cfg.vmax_rad_s + 1e-12)
            self.assertLessEqual(abs(output - previous), cfg.a_max * DT + 1e-12)
            previous = output

    def test_default_vmax_is_fast_enough_for_wrap(self):
        tilt = law()
        for _ in range(40):
            step(tilt, 0.12)
        self.assertGreater(abs(tilt.omega_y), 0.15)
        self.assertLessEqual(abs(tilt.omega_y), tilt.cfg.vmax_rad_s + 1e-12)

    def test_contact_loss_stops_and_air_guidance_can_be_explicitly_enabled(self):
        tilt = law()
        self.assertEqual(step(tilt, 0.12, contact=False), 0.0)
        for _ in range(60):
            step(tilt, 0.12)
        for _ in range(20):
            output = step(tilt, 0.12, contact=False)
        self.assertEqual(output, 0.0)
        self.assertFalse(tilt.engaged)
        air = TorqueTilt(replace(tilt.cfg, contact_only=False))
        self.assertLess(step(air, 0.12, fz=0.0, contact=False), 0.0)

    def test_contact_inference_uses_desired_normal_sign(self):
        tilt = law()
        self.assertEqual(step(tilt, 0.12, fz=-2.0, contact=None), 0.0)
        self.assertLess(step(tilt, 0.12, fz=2.0, contact=None), 0.0)
        tilt.reset()
        self.assertLess(step(tilt, 0.12, fz=-2.0, contact=None, desired=-DESIRED), 0.0)

    def test_desired_moment_is_the_balance_point_without_bias_learning(self):
        tilt = law()
        desired = DESIRED.copy()
        desired[4] = 0.10
        for _ in range(500):
            self.assertEqual(step(tilt, 0.10, desired=desired), 0.0)
        self.assertLess(step(tilt, 0.14, desired=desired), 0.0)

    def test_default_selection_and_explicit_axis_ownership(self):
        np.testing.assert_array_equal(apply_tilt_selection(None, TorqueTiltConfig()), [1, 1, 0, 1, 0, 1])
        explicit = np.array([1, 1, 0, 1, 1, 1])
        np.testing.assert_array_equal(apply_tilt_selection(explicit, TorqueTiltConfig()), explicit)
        np.testing.assert_array_equal(apply_tilt_selection(None, TorqueTiltConfig(enabled=False)), explicit)

    def test_bad_parameters_and_samples_are_rejected(self):
        for config in (
            {"mass": 0.0},
            {"a_max": 0.0},
            {"damping": float("nan")},
            {"axis": 2},
            {"theta_max_rad": 0.0},
            {"align_min": -0.1},
            {"r_face_m": 0.0},
        ):
            with self.assertRaises(ValueError):
                TorqueTiltConfig(**config)
        for dt in (0.0, -1.0, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                step(law(), 0.12, dt=dt)
        with self.assertRaises(ValueError):
            step(law(), float("nan"))

    def test_yaml_degrees_map_to_the_wrap_limits(self):
        cfg = TorqueTiltConfig.from_dict(
            {
                "hybrid_motion": {
                    "torque_tilt": {
                        "theta_max_deg": 150.0,
                        "twist_align_deg": 0.0,
                        "vmax_rad_s": 0.45,
                    }
                }
            }
        )
        self.assertAlmostEqual(cfg.theta_max_rad, math.radians(150.0), places=12)
        self.assertEqual(cfg.align_min, 0.0)
        self.assertAlmostEqual(cfg.vmax_rad_s, 0.45, places=12)
        self.assertAlmostEqual(cfg.r_tube_m, 0.020, places=12)

    def test_contact_cop_is_the_face_moment_arm(self):
        cop_x, cop_y, cop_r, valid = estimate_contact_cop(
            np.array([0.0, 0.0, 4.0, 0.02, -0.04, 0.0]), f_min=0.8
        )
        self.assertTrue(valid)
        self.assertAlmostEqual(cop_x, 0.01, places=12)
        self.assertAlmostEqual(cop_y, 0.005, places=12)
        self.assertAlmostEqual(cop_r, math.hypot(0.01, 0.005), places=12)
        _, _, _, empty = estimate_contact_cop(np.zeros(6), f_min=0.8)
        self.assertFalse(empty)

    def test_explicit_legacy_cop_stall_is_reported(self):
        tilt = TorqueTilt(TorqueTiltConfig(cop_stall_s=0.35))
        # 4 N × 12.5 mm leftover My sits on the 20 mm tube and does not shrink.
        for _ in range(int(0.60 / DT)):
            step(tilt, -0.05, fz=4.0)
        self.assertTrue(tilt.cop_valid)
        self.assertTrue(tilt.on_tube)
        self.assertGreater(tilt.cop_r, 0.010)
        self.assertTrue(tilt.tilt_stalled)
        self.assertEqual(tilt.telemetry()["tilt_stop_reason"], "cop_stall")
        held = [step(tilt, -0.05, fz=4.0) for _ in range(80)]
        self.assertLess(abs(held[-1]), 0.02)
        # Even the opt-in legacy latch must not trap the opposite correction.
        self.assertLess(step(tilt, 0.05, fz=4.0), 0.0)
        self.assertFalse(tilt.tilt_stalled)

    def test_defaults_and_live_yaml_do_not_latch_slow_cop_motion(self):
        self.assertEqual(TorqueTiltConfig().cop_stall_s, 0.0)
        self.assertEqual(TorqueTiltConfig.from_dict({}).cop_stall_s, 0.0)
        for filename in ("force.yaml", "controller.yaml"):
            path = Path(__file__).resolve().parents[1] / "configs" / filename
            cfg = TorqueTiltConfig.from_dict(yaml.safe_load(path.read_text()))
            self.assertEqual(cfg.cop_stall_s, 0.0, filename)

    def test_soft_surface_continues_past_previous_1_degree_latch(self):
        for direction in (-1.0, 1.0):
            tilt = TorqueTilt()
            theta = 0.0
            for _ in range(1200):
                tau = 0.12 * (theta - direction * 0.5)
                theta += DT * step(tilt, tau, fz=8.0)
                self.assertFalse(tilt.tilt_stalled)
                self.assertTrue(tilt.on_tube)
            # Old 350-ms latch stopped at 1.46 degrees despite residual torque.
            self.assertGreater(direction * theta, math.radians(14.0))
            self.assertFalse(tilt.tilt_frozen)
            self.assertFalse(tilt.tilt_capped)

    def test_moving_surface_can_keep_nonzero_cop_while_continuing_to_turn(self):
        tilt = TorqueTilt()
        theta = 0.0
        for i in range(1200):
            surface = 0.4 + 0.1 * i * DT
            theta += DT * step(tilt, 0.12 * (theta - surface), fz=8.0)
            self.assertFalse(tilt.tilt_stalled)
        self.assertGreater(theta, 0.5)
        self.assertGreater(tilt.omega_y, 0.08)
        self.assertGreater(abs(tilt.cop_x), tilt.cfg.cop_stall_m)
        self.assertEqual(tilt.telemetry()["tilt_stop_reason"], "")

    def test_stop_telemetry_distinguishes_deadband_contact_slack_and_angle(self):
        tilt = TorqueTilt()
        step(tilt, 0.01)
        self.assertEqual(tilt.tilt_stop_reason, "torque_deadband")
        self.assertEqual(tilt.telemetry()["tilt_deadband_nm"], 0.025)
        step(tilt, 0.12, contact=False)
        self.assertEqual(tilt.tilt_stop_reason, "no_contact")
        step(tilt, 0.12, slack_norm=0.12)
        self.assertEqual(tilt.tilt_stop_reason, "qp_slack")
        tilt = TorqueTilt(CAP20)
        for _ in range(400):
            step(tilt, 0.12)
        self.assertEqual(tilt.tilt_stop_reason, "angle_limit")

    def test_cop_near_tcp_line_does_not_stall_a_real_wrap(self):
        tilt = TorqueTilt()
        theta = 0.0
        for _ in range(800):
            tau = 0.25 * (theta - 0.8)
            if abs(tau) > 0.04:
                tau = math.copysign(0.04, tau)
            w = step(tilt, tau, fz=8.0)
            theta += DT * w
        self.assertTrue(tilt.on_face or tilt.on_tube)
        self.assertLess(tilt.cop_r, 0.006)
        self.assertFalse(tilt.tilt_stalled)
        self.assertGreater(abs(theta), 0.15)

    def test_wrapper_preserves_normal_command_and_contact_source(self):
        class ZLaw:
            def update(self, **kwargs):
                return ForceOutput(np.array([0, 0, -0.017, 0, 0, 0], dtype=float), -0.017, True, 2.0)

        wrapper = LegacyForceWithTilt(ZLaw(), law())
        out = wrapper.update(
            dt_s=DT,
            f_ext=np.array([0.0, 0.0, 2.0, 0.0, 0.12, 0.0]),
            f_des=DESIRED,
            contact=True,
        )
        self.assertEqual(out.v_force_z, -0.017)
        self.assertEqual(out.v_force[2], -0.017)
        self.assertLess(out.v_force[4], 0.0)
        np.testing.assert_array_equal(out.v_force[[0, 1, 3, 5]], np.zeros(4))
        self.assertEqual(out.telemetry["tau_y"], 0.12)

    def test_wrapper_remaps_retract_when_slack_frozen(self):
        class ZLaw:
            def update(self, **kwargs):
                return ForceOutput(np.array([0, 0, -0.08, 0, 0, 0], dtype=float), -0.08, True, 4.0)

        wrapper = LegacyForceWithTilt(ZLaw(), law())
        twisted = np.array([0.0, 0.0, 0.3, 0.0, -1.16, 0.0])
        out = wrapper.update(
            dt_s=DT,
            pose=twisted,
            f_ext=np.array([0.0, 0.0, 4.0, 0.0, 0.12, 0.0]),
            f_des=np.array([0.0, 0.0, 4.0, 0.0, 0.0, 0.0]),
            contact=True,
            slack_norm=0.12,
        )
        self.assertEqual(out.v_force[4], 0.0)
        self.assertTrue(out.telemetry["tilt_frozen"])
        self.assertGreater(abs(out.v_force[0]) + abs(out.v_force[1]), 0.01)
        world_v = rotation_from_pose(twisted) @ out.v_force[:3]
        self.assertGreater(world_v[2], 0.04)

    def test_remap_along_world_up_uses_latched_normal(self):
        r_mat = np.eye(3)
        v = remap_force_along_world_normal(
            np.array([0.0, 0.0, -0.08, 0.0, 0.0, 0.0]),
            rotation=r_mat,
            n_world=np.array([0.0, 0.0, 1.0]),
            v_force_z=-0.08,
        )
        np.testing.assert_allclose(v[:3], [0.0, 0.0, 0.08], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
