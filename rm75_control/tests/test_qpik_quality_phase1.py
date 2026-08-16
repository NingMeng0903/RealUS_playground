"""Phase-1 QPIK quality: ψ retarget, rail soft-limit fade, capped reach, uniform scale."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.loop import scale_qdot_into_box
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import QpConfig
from rm75_control.control.joint_admittance_8dof.tasks.psi_retarget import (
    PostureRetarget,
    PsiRetargetConfig,
    d_from_q,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_extension import (
    RailExtensionConfig,
    RailExtensionTask,
)


def test_limit_saturation_uses_soft_band_not_urdf() -> None:
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(
            enabled=True,
            k_ext=0.0,
            k_esc=0.0,
            v_lpf_tau_s=0.0,
            limit_margin_m=0.10,
            soft_min_m=0.10,
            soft_max_m=0.70,
        ),
    )
    # 5 cm past the *soft* max but still inside URDF 0.8 → must already be faded.
    scale = task._limit_saturation(0.70, v=0.05)
    assert scale == 0.0
    assert task.last_limit_saturated
    scale_mid = task._limit_saturation(0.40, v=0.05)
    assert scale_mid == 1.0


def test_ff_owns_does_not_zero_reach_but_caps_it() -> None:
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(
            enabled=True,
            k_ext=5.0,
            k_esc=0.0,
            k_ff=1.0,
            e0_m=0.0,
            e1_m=0.01,
            v_ff_thr_m_s=0.005,
            v_reach_cap_m_s=0.02,
            v_max_m_s=0.08,
            v_lpf_tau_s=0.0,
            d_star_err0_m=1.0,
        ),
    )
    task.set_mode("reach")
    q = 0.5 * (kin.q_lower + kin.q_upper)
    task.capture_reference(q)
    task.d_pref_m = task.extension(q) + 0.30
    j_rail = kin.jacobian(q)[:3, 0]
    n = float(np.linalg.norm(j_rail))
    vel_ff = np.zeros(6)
    vel_ff[:3] = 0.05 * (j_rail / n)
    task(q, sigma_scale=1.0, vel_ff=vel_ff, dt_s=0.005)
    assert abs(task.last_v_reach) > 1e-6
    assert abs(task.last_v_reach) <= 0.02 + 1e-9
    assert task.last_v_reach * task.last_v_ff < 0.0


def test_scale_qdot_into_box_preserves_direction() -> None:
    qdot = np.array([0.2, 0.4, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    lo = np.full(8, -1.0)
    hi = np.full(8, 1.0)
    hi[1] = 0.1
    out = scale_qdot_into_box(qdot, lo, hi)
    assert out[1] <= 0.1 + 1e-12
    assert np.sign(out[0]) == np.sign(qdot[0])
    assert np.sign(out[2]) == np.sign(qdot[2])
    ratio = out[0] / qdot[0]
    assert np.isclose(out[2] / qdot[2], ratio, atol=1e-9)
    clipped = np.clip(qdot, lo, hi)
    assert abs(out[0] - clipped[0]) > 1e-6


def test_psi_rate_limit_caps_step() -> None:
    kin = RobotKinematics()
    rt = PostureRetarget(
        kin,
        PsiRetargetConfig(
            enabled=True,
            psi_rate_rad_s=np.deg2rad(20.0),
        ),
    )
    q = 0.5 * (kin.q_lower + kin.q_upper)
    rt.reset(q)
    start = 0.0
    rt._psi_cmd = start
    rt._psi_star = np.deg2rad(90.0)
    out = rt._rate_limit_psi(0.005)
    assert abs(out - start) <= np.deg2rad(20.0) * 0.005 + 1e-9
    assert abs(out - start) > 0.0


def test_psi_hold_does_not_climb_d_star() -> None:
    kin = RobotKinematics()
    rt = PostureRetarget(kin, PsiRetargetConfig(enabled=True))
    q = np.array([0.375, 0.194, -0.503, -0.069, 1.979, -0.776, 0.547, -4.370])
    rt.reset(q)
    d0 = float(rt._d_star)
    for _ in range(20):
        _psi, d = rt.step(q, 0.005, rail_lo=0.05, rail_hi=0.75)
    assert d == pytest.approx(d0)


def test_unplanned_step_freezes_d_star_from_q_nominal() -> None:
    kin = RobotKinematics()
    rt = PostureRetarget(kin, PsiRetargetConfig(enabled=True))
    q = np.array([0.375, 0.194, -0.503, -0.069, 1.979, -0.776, 0.547, -4.370])
    q_star = np.array([0.0, 0.0, -0.785, 0.0, 1.571, 0.698, 0.785, 0.0])
    rt.reset(q)
    d_live = d_from_q(kin, q)
    d_star = d_from_q(kin, q_star)
    assert abs(d_live - d_star) > 0.01
    last_d = float("nan")
    for _ in range(8):
        _psi, last_d = rt.step(
            q, 0.005, rail_lo=0.005, rail_hi=0.78, q_nominal=q_star
        )
    assert last_d == pytest.approx(d_star, abs=1e-9)
    assert rt.d_star_m == pytest.approx(d_star, abs=1e-9)
    assert last_d != pytest.approx(d_live, abs=1e-3)
    assert not rt.planned


def test_qp_smoothness_weight_is_wired() -> None:
    cfg = QpConfig(smoothness_weight=0.15)
    assert cfg.smoothness_weight == 0.15


def test_governor_floor_and_physical_gate() -> None:
    from rm75_control.control.joint_admittance_8dof.loop import (
        Phase,
        _reference_governor_scale,
    )

    class _Outer:
        pass

    phase = Phase(
        outer=_Outer(),
        governor_err_ok_mm=5.0,
        governor_err_max_mm=25.0,
        governor_scale_min=0.25,
        governor_joint_err_max_deg=0.0,
    )
    raw_free = _reference_governor_scale(
        phase, outer_err_mm=80.0, joint_err_deg=None, physical_saturated=False
    )
    assert raw_free == 1.0
    raw_sat = _reference_governor_scale(
        phase, outer_err_mm=80.0, joint_err_deg=None, physical_saturated=True
    )
    assert raw_sat == 0.25
