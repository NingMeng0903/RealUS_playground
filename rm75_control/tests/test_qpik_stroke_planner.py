"""One-shot (d*, ψ*) stroke planner + rail feedforward.

The force-side phase-2 work was reverted to 55e261d after hardware showed a
contact-loss / impact regression; only the rail-side planner survived, so this
file covers just that.
"""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import (
    QpConfig,
    QpIkController,
)
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits
from rm75_control.control.joint_admittance_8dof.tasks.psi_retarget import (
    PostureRetarget,
    PsiRetargetConfig,
    StrokeInfeasibleError,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_extension import (
    RailExtensionConfig,
    RailExtensionTask,
)


DT = 0.005
_SEED_Q = np.array(
    [0.375, 0.194, -0.503, -0.069, 1.979, -0.776, 0.547, -4.370]
)


def test_minmax_rejects_empty_stroke() -> None:
    kin = RobotKinematics()
    rt = PostureRetarget(
        kin,
        PsiRetargetConfig(enabled=True, n_y=3, n_d=3, n_psi=3, rail_margin_m=0.02),
    )
    rt.reset(_SEED_Q)
    with pytest.raises(StrokeInfeasibleError, match="reduce amplitude"):
        rt.plan_stroke(
            _SEED_Q,
            y_center_m=0.4,
            amplitude_m=2.0,
            rail_lo=0.005,
            rail_hi=0.78,
        )


def test_minmax_does_not_park_rail_at_a_stop() -> None:
    kin = RobotKinematics()
    rt = PostureRetarget(
        kin,
        PsiRetargetConfig(enabled=True, n_y=5, n_d=6, n_psi=5, w_sigma=0.5),
    )
    rt.reset(_SEED_Q)
    y_c = float(kin.fk_placement(_SEED_Q).translation[1])
    d_star, psi_star = rt.plan_stroke(
        _SEED_Q,
        y_center_m=y_c,
        amplitude_m=0.05,
        rail_lo=0.005,
        rail_hi=0.78,
    )
    y_lo, y_hi = y_c - 0.05, y_c + 0.05
    assert y_lo - d_star >= 0.02 - 1e-9
    assert y_hi - d_star <= 0.76 + 1e-9
    assert np.isfinite(psi_star)


def test_psi_star_is_wrapped_for_telemetry() -> None:
    """psi0 ± π grid must not leak an unwrapped ψ* into the CSV."""
    kin = RobotKinematics()
    rt = PostureRetarget(
        kin,
        PsiRetargetConfig(enabled=True, n_y=3, n_d=4, n_psi=8, w_sigma=0.5),
    )
    rt.reset(_SEED_Q)
    y_c = float(kin.fk_placement(_SEED_Q).translation[1])
    _d, psi_star = rt.plan_stroke(
        _SEED_Q,
        y_center_m=y_c,
        amplitude_m=0.05,
        rail_lo=0.005,
        rail_hi=0.78,
    )
    assert -np.pi - 1e-9 <= psi_star <= np.pi + 1e-9
    assert -np.pi - 1e-9 <= rt.psi_star_rad <= np.pi + 1e-9


def test_psi_rate_limit_no_lpf() -> None:
    kin = RobotKinematics()
    rt = PostureRetarget(
        kin, PsiRetargetConfig(enabled=True, psi_rate_rad_s=np.deg2rad(20.0))
    )
    rt.reset(_SEED_Q)
    rt._psi_cmd = 0.0
    rt._psi_star = np.deg2rad(90.0)
    out = rt._rate_limit_psi(0.005)
    assert abs(out) <= np.deg2rad(20.0) * 0.005 + 1e-9
    assert abs(out) > 0.0


def test_rail_ff_tracks_desired_y_minus_d_star() -> None:
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
            v_reach_cap_m_s=0.08,
            v_max_m_s=0.08,
            v_lpf_tau_s=0.0,
        ),
    )
    task.set_mode("reach")
    q = _SEED_Q.copy()
    task.capture_reference(q)
    d_star = float(task.d_pref_m)
    y_tcp_d = float(kin.fk_placement(q).translation[1]) + 0.04
    vel_ff = np.array([0.0, 0.03, 0.0, 0.0, 0.0, 0.0])
    v, _w = task(q, sigma_scale=1.0, vel_ff=vel_ff, dt_s=DT, y_tcp_d=y_tcp_d)
    assert np.isfinite(task.last_rail_ff_m)
    assert task.last_rail_ff_m == pytest.approx(y_tcp_d - d_star, abs=1e-9)
    assert v * 0.03 > 0.0


def test_anti_cancel_term_is_gone() -> None:
    """The λ (J_y,arm q̇)² term traded tracking for arm-Y suppression."""
    assert not hasattr(QpConfig(), "anti_cancel_weight")


def _wln_core() -> tuple[QpIkController, SafetyLimits]:
    kin = RobotKinematics()
    a_max = np.full(kin.nv, 3.0)
    a_max[0] = 0.30
    lim = SafetyLimits.from_kinematics(
        kin, v_scale=0.8, a_max=a_max, position_margin=0.0
    )
    lim.q_lower[0], lim.q_upper[0] = 0.005, 0.78
    cfg = QpConfig()
    cfg.collision.enabled = False
    return QpIkController(kin, lim, cfg), lim


def test_wln_prices_the_rail_only_toward_its_stop() -> None:
    core, lim = _wln_core()
    nv = core.kin.nv
    q_mid = 0.5 * (lim.q_lower + lim.q_upper)

    q_near = q_mid.copy()
    q_near[0] = lim.q_lower[0] + 0.04
    toward = np.zeros(nv)
    toward[0] = -0.05
    scale_toward = core._wln_reg_scale(q_near, toward)
    # reg[0]=1e-3 against the arm's 1e-2: the rail must end up the dearer
    # joint so the QP shifts the stroke instead of grinding into the stop.
    assert scale_toward[0] > 10.0
    assert scale_toward[0] <= core.cfg.wln.max_scale + 1e-9

    away = np.zeros(nv)
    away[0] = +0.05
    assert core._wln_reg_scale(q_near, away)[0] == pytest.approx(1.0)

    # Mid travel is the tuned reg, untouched.
    assert core._wln_reg_scale(q_mid, toward)[0] == pytest.approx(1.0)


def test_wln_never_touches_arm_joints() -> None:
    """J4 sits inside any useful band most of a scan; pricing it buys slack."""
    core, lim = _wln_core()
    nv = core.kin.nv
    for joint in range(1, nv):
        q = 0.5 * (lim.q_lower + lim.q_upper)
        q[joint] = lim.q_upper[joint] - 0.002  # 0.1 deg off the stop
        qdot = np.zeros(nv)
        qdot[joint] = +0.5  # driving straight at it
        assert core._wln_reg_scale(q, qdot)[joint] == pytest.approx(1.0)


def test_barrier_press_cap_has_a_floor_in_contact() -> None:
    from rm75_control.control.admittance_common.force_barrier import (
        ForceBarrierConfig,
        ForceSpaceVelocityDamper,
    )

    cfg = ForceBarrierConfig(enabled=True, v_min_press_m_s=0.003, v_ref_m_s=0.08)
    damper = ForceSpaceVelocityDamper(cfg)
    # Massively over force: prediction and stiffness terms both collapse.
    cap_press, _ = damper.caps(
        f_z=40.0,
        f_des_z=3.0,
        in_contact=True,
        v_z_cap=0.08,
        seek_vz_m_s=0.08,
        contact_enter_n=0.5,
        ke_est_n_m=20000.0,
        mass_eq_kg=1.0,
    )
    assert cap_press >= 0.003 - 1e-12

    # A tighter v_z_cap (recontact sleeve) still wins over the floor.
    cap_small, _ = damper.caps(
        f_z=40.0,
        f_des_z=3.0,
        in_contact=True,
        v_z_cap=0.001,
        seek_vz_m_s=0.001,
        contact_enter_n=0.5,
    )
    assert cap_small <= 0.001 + 1e-12


def test_barrier_caps_the_free_space_approach() -> None:
    """Impact ~ Ke*v*T_delay, so the gap-closing speed sets the first peak."""
    from rm75_control.control.admittance_common.force_barrier import (
        ForceBarrierConfig,
        ForceSpaceVelocityDamper,
    )

    damper = ForceSpaceVelocityDamper(
        ForceBarrierConfig(enabled=True, v_seek_free_m_s=0.030)
    )
    cap_free, _ = damper.caps(
        f_z=0.0,
        f_des_z=3.0,
        in_contact=False,
        v_z_cap=0.08,
        seek_vz_m_s=0.08,
        contact_enter_n=0.5,
    )
    assert cap_free == pytest.approx(0.030)

    # A tighter external sleeve still wins.
    cap_sleeve, _ = damper.caps(
        f_z=0.0,
        f_des_z=3.0,
        in_contact=False,
        v_z_cap=0.08,
        seek_vz_m_s=0.008,
        contact_enter_n=0.5,
    )
    assert cap_sleeve == pytest.approx(0.008)

    # In contact the cap is the admittance's business, not this one.
    cap_contact, _ = damper.caps(
        f_z=0.0,
        f_des_z=3.0,
        in_contact=True,
        v_z_cap=0.08,
        seek_vz_m_s=0.08,
        contact_enter_n=0.5,
    )
    assert cap_contact > 0.030


def test_admittance_error_is_not_low_passed() -> None:
    """The force-axis slew limiter already bounds command jitter to ~4.9 mm/s
    per tick (measured v_force_z step: 2.8 mm/s p95), so a low-pass on the
    admittance error bought nothing and cost twice — stiff-surface impact
    8 N -> 12.2 N, and proactive v_r 6.97 -> 5.89 mm/s on a receding surface.
    """
    from rm75_control.control.admittance_common.controller import AdmittanceConfig

    cfg = AdmittanceConfig()
    assert not hasattr(cfg, "force_lpf_tau_s")
    assert not hasattr(cfg, "force_lpf_snap_n")


def test_qpik_yaml_keeps_rail_planner_and_baseline_force() -> None:
    from pathlib import Path

    import yaml

    from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config

    raw = yaml.safe_load(
        Path(__file__).resolve().parents[1].joinpath(
            "configs", "joint_admittance_8dof.yaml"
        ).read_text(encoding="utf-8")
    )
    cfg = build_joint_ik_config(raw)
    assert cfg.qp.twist_scale_lpf_tau_s == pytest.approx(0.08)
    assert cfg.psi_retarget.enabled
    assert cfg.psi_retarget.n_y >= 3
    assert cfg.rail_extension.enabled
    assert cfg.qp.twist_sigma_floor == pytest.approx(0.02)
    assert cfg.qp.task_weight_min_frac == pytest.approx(0.05)
    assert "anti_cancel_weight" not in raw["inner"]["qp"]
    # Rail/jerk work: limit handoff on the rail only, third-order box.
    assert cfg.qp.wln.enabled
    assert cfg.qp.wln.band_rail_m > 0.0
    # The arm has no spare joint to hand the stroke to; weighting it only
    # bought slack, so its band must stay disabled.
    assert cfg.qp.wln.band_rad == 0.0
    assert cfg.qp.j_max_arm_rad_s3 > 0.0
    assert cfg.qp.j_max_rail_m_s3 > 0.0
    # Rail command shaping must not come back: it fed core.qdot_prev and so
    # multiplied every acceleration limit by its own alpha.
    assert not hasattr(cfg.rail, "cmd_lpf_tau_s")
    # e85c9ab press speed restored; the barrier stays as the brake.
    hm = raw["hybrid_motion"]
    assert hm["proactive_retract_only"] is False
    assert hm["force_dob"]["enabled"] is True
    assert float(hm["force_axis_slew_press_m_s2"]) >= 0.8
    assert hm["force_barrier"]["enabled"] is True
    # The blunt phase-2 instruments stay out.
    assert "vel_dob" not in hm
    assert "v_air_m_s" not in hm
    assert "admittance_damping_retract_z" not in hm
    assert "tafac_scan_ff" not in hm
    # BEFM stays fail-closed until befm_calibrate.py passes on hardware.
    assert hm["bidirectional_flow"]["mode"] == "observe"
    assert hm["bidirectional_flow"]["sign_verified"] is False
    assert hm["bidirectional_flow"]["feedback_delay_verified"] is False


def test_analyzer_rejects_empty_scan(tmp_path) -> None:
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "apps"
        / "joint_admittance_8dof"
        / "analyze_qpik_quality.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_qpik_quality", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("phase,t_wall_s\nmove,0\n", encoding="utf-8")
    assert mod.analyze(csv_path) == 2
    assert "waste_ratio" in mod.GATES
    assert "accel_reversals_per_s" in mod.GATES
    assert "dt_on_time_frac" in mod.GATES


def test_analyzer_jerk_metric_ignores_loop_period_jitter(tmp_path) -> None:
    """A perfectly smooth command must not read as jerk just because the
    scheduler jittered.  Dividing by wall dt inflated the reversal rate 3-4x
    and sent three rounds of tuning after a metric artefact."""
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "apps"
        / "joint_admittance_8dof"
        / "analyze_qpik_quality.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_qpik_quality", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    rng = np.random.default_rng(1)
    n = 3000
    cols = ["phase", "t_wall_s"] + [f"q_cmd_{i}" for i in range(8)]
    lines = [",".join(cols)]
    t = 0.0
    for k in range(n):
        # Smooth sinusoid in sample index; wall clock jitters +-20% like the
        # real loop (measured 5.6-6.9 ms against a 5.0 ms budget).
        t += 0.0061 * float(rng.uniform(0.8, 1.2))
        q = 0.3 * np.sin(2.0 * np.pi * 0.2 * k * 0.0061)
        lines.append(
            ",".join(["scan", f"{t:.6f}"] + [f"{q:.9f}"] * 8)
        )
    csv_path = tmp_path / "jitter.csv"
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        mod.analyze(csv_path)
    out = buf.getvalue()
    line = next(
        ln for ln in out.splitlines() if "accel sign reversals" in ln
    )
    assert "[PASS]" in line, line
