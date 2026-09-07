"""Regression checks for unbounded, symmetric tool-y moment balance."""

from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from peirastic.realman8dof.force.protocol import ForceOutput
from peirastic.realman8dof.force.torque_tilt import (
    LegacyForceWithTilt,
    TorqueTilt,
    TorqueTiltConfig,
    apply_tilt_selection,
)

DT = 0.005
DESIRED = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])


def step(law, tau, *, fz=2.0, contact=True, desired=DESIRED, dt=DT):
    wrench = np.array([0.0, 0.0, fz, 0.0, tau, 0.0])
    return law.update(wrench, desired, dt_s=dt, contact=contact)


class TestTorqueTilt(unittest.TestCase):
    def test_small_noise_sticks_but_both_sides_above_threshold_yield(self):
        for tau in (-0.025, -0.01, 0.0, 0.01, 0.025):
            law = TorqueTilt()
            for _ in range(80):
                self.assertEqual(step(law, tau), 0.0)
        for tau in (-0.03, 0.03):
            law = TorqueTilt()
            output = [step(law, tau) for _ in range(100)]
            self.assertTrue(all(w * tau < 0.0 for w in output))
            self.assertGreater(abs(output[-1]), 0.01)

    def test_positive_and_negative_histories_are_exact_mirrors(self):
        positive, negative = TorqueTilt(), TorqueTilt()
        torques = np.repeat([0.01, 0.03, 0.08, -0.02, -0.06, 0.15, 0.0], 70)
        for tau in torques:
            plus = step(positive, tau)
            minus = step(negative, -tau)
            self.assertAlmostEqual(plus, -minus, places=14)
        self.assertAlmostEqual(positive.theta_tilt, -negative.theta_tilt, places=14)

    def test_normal_force_error_cannot_suppress_real_moment(self):
        nominal, excess, deficient = TorqueTilt(), TorqueTilt(), TorqueTilt()
        for _ in range(100):
            full = step(nominal, 0.12)
            self.assertEqual(full, step(excess, 0.12, fz=12.0))
            self.assertEqual(full, step(deficient, 0.12, fz=-1.0))
        self.assertLess(full, -0.1)

    def test_no_accumulated_angle_lock_or_reset_on_contact_flicker(self):
        law = TorqueTilt()
        for _ in range(1600):
            output = step(law, 0.12)
        self.assertLess(law.theta_tilt, -1.5)
        self.assertLess(output, -0.19)
        angle = law.theta_tilt
        step(law, 0.12, contact=False)
        output = step(law, 0.12, contact=True)
        self.assertLess(law.theta_tilt, angle)
        self.assertLess(output, -0.15)

    def test_rotational_contact_spring_reaches_balance_past_old_angle_limit(self):
        law = TorqueTilt()
        theta, surface_angle, stiffness = 0.0, 1.0, 0.25
        for _ in range(3000):
            tau = stiffness * (theta - surface_angle)
            theta += DT * step(law, tau)
        self.assertGreater(theta, 0.89)
        self.assertLessEqual(theta, surface_angle)
        self.assertLessEqual(abs(stiffness * (theta - surface_angle)), 0.025 + 1e-5)

    def test_reversal_brakes_immediately_and_changes_direction(self):
        law = TorqueTilt()
        # Reverse while accelerating, the old jerk state could accelerate away.
        for _ in range(3):
            step(law, 0.5)
        before = law.omega_y
        self.assertGreater(step(law, -0.04), before)
        for _ in range(100):
            output = step(law, -0.04)
        self.assertGreater(output, 0.02)

    def test_velocity_and_acceleration_limits_with_large_moments(self):
        cfg = TorqueTiltConfig(a_max=0.8, vmax_rad_s=0.12)
        law = TorqueTilt(cfg)
        previous = 0.0
        for tau in np.repeat([4.0, -4.0, 0.0, 4.0], 100):
            output = step(law, tau)
            self.assertLessEqual(abs(output), cfg.vmax_rad_s + 1e-12)
            self.assertLessEqual(abs(output - previous), cfg.a_max * DT + 1e-12)
            previous = output

    def test_contact_loss_stops_and_air_guidance_can_be_explicitly_enabled(self):
        law = TorqueTilt()
        self.assertEqual(step(law, 0.12, contact=False), 0.0)
        for _ in range(60):
            step(law, 0.12)
        for _ in range(20):
            output = step(law, 0.12, contact=False)
        self.assertEqual(output, 0.0)
        self.assertFalse(law.engaged)
        air = TorqueTilt(replace(law.cfg, contact_only=False))
        self.assertLess(step(air, 0.12, fz=0.0, contact=False), 0.0)

    def test_contact_inference_uses_desired_normal_sign(self):
        law = TorqueTilt()
        self.assertEqual(step(law, 0.12, fz=-2.0, contact=None), 0.0)
        self.assertLess(step(law, 0.12, fz=2.0, contact=None), 0.0)
        law.reset()
        self.assertLess(step(law, 0.12, fz=-2.0, contact=None, desired=-DESIRED), 0.0)

    def test_desired_moment_is_the_balance_point_without_bias_learning(self):
        law = TorqueTilt()
        desired = DESIRED.copy()
        desired[4] = 0.10
        for _ in range(500):
            self.assertEqual(step(law, 0.10, desired=desired), 0.0)
        self.assertLess(step(law, 0.14, desired=desired), 0.0)

    def test_default_selection_and_explicit_axis_ownership(self):
        np.testing.assert_array_equal(apply_tilt_selection(None, TorqueTiltConfig()), [1, 1, 0, 1, 0, 1])
        explicit = np.array([1, 1, 0, 1, 1, 1])
        np.testing.assert_array_equal(apply_tilt_selection(explicit, TorqueTiltConfig()), explicit)
        np.testing.assert_array_equal(apply_tilt_selection(None, TorqueTiltConfig(enabled=False)), explicit)

    def test_bad_parameters_and_samples_are_rejected(self):
        for config in ({"mass": 0.0}, {"a_max": 0.0}, {"damping": float("nan")}, {"axis": 2}):
            with self.assertRaises(ValueError):
                TorqueTiltConfig(**config)
        for dt in (0.0, -1.0, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                step(TorqueTilt(), 0.12, dt=dt)
        with self.assertRaises(ValueError):
            step(TorqueTilt(), float("nan"))

    def test_wrapper_preserves_normal_command_and_contact_source(self):
        class ZLaw:
            def update(self, **kwargs):
                return ForceOutput(np.array([0, 0, -0.017, 0, 0, 0]), -0.017, True, 2.0)

        law = LegacyForceWithTilt(ZLaw(), TorqueTilt())
        out = law.update(
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


if __name__ == "__main__":
    unittest.main()
