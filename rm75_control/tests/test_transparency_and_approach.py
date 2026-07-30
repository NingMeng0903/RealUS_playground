"""Transparency (hand feel) and free-space approach-speed decoupling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from rm75_control.control.admittance_common.controller import (
    AdmittanceConfig,
    AdmittanceController,
)


DT = 0.005
POSE = np.zeros(6)


def _production_config() -> AdmittanceConfig:
    raw = yaml.safe_load(
        Path("configs/joint_admittance_8dof.yaml").read_text()
    )
    return AdmittanceConfig.from_dict(raw)


def _hand_push_apparent_damping(f_push_n: float, settle_ticks: int = 600) -> dict:
    """desired_z=0 hand push: return steady |F|/|v| when unsaturated."""
    ctrl = AdmittanceController(DT, _production_config())
    # Warm contact latch with a brief load, then zero desired force.
    for _ in range(50):
        force = np.zeros(6)
        force[2] = 1.0
        ctrl.compute_velocity_command(
            POSE,
            POSE,
            np.zeros(6),
            force,
            np.zeros(6),
            f_ext_raw=force,
            dt_actual=DT,
            in_contact=True,
        )
    for _ in range(settle_ticks):
        force = np.zeros(6)
        force[2] = f_push_n
        ctrl.compute_velocity_command(
            POSE,
            POSE,
            np.zeros(6),
            force,
            np.zeros(6),
            f_ext_raw=force,
            dt_actual=DT,
            in_contact=True,
        )
    vz = float(ctrl.v_force_z)
    vz_cap = float(ctrl._v_z_cap())
    saturated = abs(vz) >= 0.95 * vz_cap - 1e-9
    d_app = abs(f_push_n / vz) if abs(vz) > 1e-6 else float("inf")
    return {
        "d_app": d_app,
        "vz": vz,
        "bd": float(ctrl.damping_z_eff),
        "ke": float(ctrl.ke_est),
        "saturated": saturated,
    }


def test_hand_push_apparent_damping_stays_light_and_not_heavier_with_force():
    """desired=0 hand feel: unsaturated |F/v| ≤ 20 and non-increasing in push.

    Velocity-saturated samples are excluded from the monotonicity check —
    at the vz cap |F|/|v| grows with F even when the controller damping is
    still the light trend baseline.
    """
    pushes = (0.5, 1.0, 1.5, 2.0)
    samples = [_hand_push_apparent_damping(f) for f in pushes]
    unsaturated = [
        (f, s) for f, s in zip(pushes, samples) if not s["saturated"]
    ]
    assert unsaturated, "expected at least one unsaturated hand-push sample"
    for f, sample in zip(pushes, samples):
        if sample["saturated"]:
            # Still require the internal trend damping stay light.
            assert sample["bd"] <= 25.0, (
                f"push {f}N saturated but damping_z_eff={sample['bd']:.1f}"
            )
            continue
        assert sample["d_app"] <= 20.0 + 1e-6, (
            f"push {f}N apparent damping {sample['d_app']:.1f} > 20"
        )
    # Monotone non-increasing among unsaturated samples.
    for i in range(1, len(unsaturated)):
        prev_f, prev = unsaturated[i - 1]
        cur_f, cur = unsaturated[i]
        assert cur["d_app"] <= prev["d_app"] + 1.0, (
            f"apparent damping rose from {prev['d_app']:.1f}@{prev_f}N "
            f"to {cur['d_app']:.1f}@{cur_f}N"
        )


def test_approach_speed_decoupled_from_desired_force():
    """Free-space seek speed is fixed; f_des 1/2/5 N must agree within 10%."""
    cfg = _production_config()
    speeds: dict[float, float] = {}
    for desired in (1.0, 2.0, 5.0):
        ctrl = AdmittanceController(DT, cfg)
        samples = []
        for _ in range(200):
            force = np.zeros(6)
            target = np.zeros(6)
            target[2] = desired
            velocity = ctrl.compute_velocity_command(
                POSE,
                POSE,
                np.zeros(6),
                force,
                target,
                f_ext_raw=force,
                dt_actual=DT,
            )[2]
            samples.append(velocity)
        speeds[desired] = float(np.mean(samples[-50:]))
    values = np.asarray(list(speeds.values()))
    assert np.all(values > 0.0)
    spread = (values.max() - values.min()) / max(abs(values.mean()), 1e-9)
    assert spread <= 0.10, (
        f"approach speeds not decoupled: { {k: 1000*v for k,v in speeds.items()} }"
    )
    # Absolute seek target from yaml.
    assert abs(values.mean() - cfg.seek_vz_m_s) / cfg.seek_vz_m_s <= 0.10


def test_first_contact_peak_near_desired_plus_budget():
    """Seek→impact on a stiff spring should not grossly overshoot f_des+budget."""
    cfg = _production_config()
    ke_n_m = 2500.0
    budget_n = max(
        float(cfg.force_barrier.budget_min_n),
        float(cfg.force_barrier.budget_frac) * 2.0,
    )
    for desired in (1.0, 2.0, 5.0):
        ctrl = AdmittanceController(DT, cfg)
        tcp_z = -0.010  # 10 mm above surface at z=0
        peak = 0.0
        for _ in range(800):
            force_z = max(0.0, ke_n_m * tcp_z)
            force = np.zeros(6)
            force[2] = force_z
            target = np.zeros(6)
            target[2] = desired
            pose = np.zeros(6)
            pose[2] = tcp_z
            velocity = ctrl.compute_velocity_command(
                pose,
                pose,
                np.zeros(6),
                force,
                target,
                f_ext_raw=force,
                dt_actual=DT,
            )[2]
            tcp_z += velocity * DT
            peak = max(peak, force_z)
        budget = max(
            float(cfg.force_barrier.budget_min_n),
            float(cfg.force_barrier.budget_frac) * desired,
        )
        assert peak <= desired + budget + 1.5, (
            f"des={desired}N first-contact peak {peak:.2f}N > "
            f"f_des+budget+1.5 ({desired + budget + 1.5:.2f}N)"
        )
