"""First-impact ΔF/Δz replaces the ke_impact_initial=1500 jump."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.admittance_common.adaptive_ke import (
    AdaptiveKeConfig,
    EnvironmentStiffnessEstimator,
)


def test_first_impact_deltaf_deltaz_without_jump() -> None:
    cfg = AdaptiveKeConfig(
        enabled=True,
        ke_initial=80.0,
        ke_impact_initial=0.0,
        ke_cap_ub_n_m=2000.0,
        ke_min=40.0,
        ke_max=2500.0,
        dx_threshold_m=8e-5,
        settle_ticks=10,
        gate_lateral_velocity=False,
        gate_df_spike=False,
        ke_idle_decay_s=0.0,
        bd_slew_max=1e6,
        ke_slew_max=1e6,
    )
    est = EnvironmentStiffnessEstimator(cfg, dt=0.005, mass_z=1.0)
    pose = np.zeros(6)
    est.update(0.0, pose, in_contact=False, mass_z=1.0, v_force_z=0.0)
    assert est.ke_confident is False
    assert est.ke_for_cap == pytest.approx(2000.0)

    ke = 80.0
    for k in range(20):
        ke, _bd = est.update(
            0.2 + 400.0 * (k + 1) * 0.005 * 0.02,
            pose,
            in_contact=True,
            mass_z=1.0,
            v_force_z=0.02,
            f_err_z=0.0,
            f_des_z=2.0,
        )
        if est.ke_confident:
            break
    assert est.ke_confident
    assert ke > 80.0
    assert est.ke_for_cap == pytest.approx(2000.0)

    while est._contact_ticks <= cfg.settle_ticks:
        ke, _bd = est.update(
            ke * est._x_adm if est._x_adm > 0.0 else 2.0,
            pose,
            in_contact=True,
            mass_z=1.0,
            v_force_z=0.02,
            f_err_z=0.0,
            f_des_z=2.0,
        )
    assert est.ke_for_cap == pytest.approx(ke)


def test_delayed_press_first_impact_does_not_mark_soft_ke_confident() -> None:
    cfg = AdaptiveKeConfig(
        enabled=True,
        ke_initial=80.0,
        ke_impact_initial=0.0,
        ke_cap_ub_n_m=2000.0,
        ke_min=40.0,
        ke_max=2500.0,
        dx_threshold_m=8e-5,
        settle_ticks=10,
        gate_lateral_velocity=False,
        gate_df_spike=False,
        ke_idle_decay_s=0.0,
        bd_slew_max=1e6,
        ke_slew_max=1e6,
    )
    est = EnvironmentStiffnessEstimator(cfg, dt=0.005, mass_z=1.0)
    pose = np.zeros(6)
    est.update(0.0, pose, in_contact=False, mass_z=1.0, v_force_z=0.0)
    for k in range(8):
        est.update(
            0.5 + 50.0 * (k + 1) * 0.005 * 0.05,
            pose,
            in_contact=True,
            mass_z=1.0,
            v_force_z=0.05,
            f_err_z=8.0,
            f_des_z=2.0,
        )
    assert est.ke_confident is False
    assert est.ke_for_cap == pytest.approx(2000.0)
