"""One physical admittance configuration across force setpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def _load_scale_fn():
    from rm75_control.control.joint_admittance.api import scale_admittance_for_desired_z

    return scale_admittance_for_desired_z


def test_yaml_adaptive_ke_enabled():
    raw = yaml.safe_load(Path("configs/joint_admittance.yaml").read_text())
    hm = raw["hybrid_motion"]
    assert hm["adaptive_ke"]["enabled"] is True
    assert hm["admittance_mass_z"] <= 1.5


def test_physical_parameters_do_not_scale_with_desired_z():
    raw = yaml.safe_load(Path("configs/joint_admittance.yaml").read_text())
    scale = _load_scale_fn()
    cfg1 = scale(raw, 1.0)
    cfg12 = scale(raw, 12.0)
    assert cfg12.var_damping_f_max_n == pytest.approx(cfg1.var_damping_f_max_n)
    assert cfg12.adaptive_ke.bd_max == pytest.approx(cfg1.adaptive_ke.bd_max)
    assert cfg12.admittance_mass_z == pytest.approx(cfg1.admittance_mass_z)


def test_deadband_fixed_across_desired_z():
    """The force deadband is a sensor-noise quantity and must NOT scale with
    the setpoint: scaling it made a 5 N hold need >0.5 N over-force before any
    retract authority (the "over-force / heavy damping" hand feel)."""
    raw = yaml.safe_load(Path("configs/joint_admittance.yaml").read_text())
    scale = _load_scale_fn()
    cfg1 = scale(raw, 1.0)
    cfg3 = scale(raw, 3.0)
    assert cfg3.deadband_n == pytest.approx(cfg1.deadband_n)
    assert cfg3.deadband_width_n == pytest.approx(cfg1.deadband_width_n)
    assert cfg3.adaptive_ke.contact_force_n == pytest.approx(
        cfg1.adaptive_ke.contact_force_n
    )


def test_f_err_gate_relative_to_setpoint():
    """K̂_e learning / idle-decay gate = max(f_err_gate_n, frac·f_des_z):
    a fixed absolute gate froze K̂_e at ke_impact_initial for large setpoints
    (hand-interaction ripple always exceeded it) and pinned b_d high."""
    raw = yaml.safe_load(Path("configs/joint_admittance.yaml").read_text())
    ak = raw["hybrid_motion"]["adaptive_ke"]
    assert ak["f_err_gate_frac"] > 0.0

    from rm75_control.control.admittance_common.adaptive_ke import (
        AdaptiveKeConfig,
        EnvironmentStiffnessEstimator,
    )

    cfg = AdaptiveKeConfig(
        f_err_gate_n=1.2,
        f_err_gate_frac=0.35,
        f_err_gate_floor_n=3.0,
    )
    est = EnvironmentStiffnessEstimator(cfg, dt=0.005, mass_z=1.0)
    # Low/zero setpoint: the floor eliminates the old 1.2 N hand-push freeze.
    assert est._f_err_gate_eff_n(0.0) == pytest.approx(3.0)
    assert est._f_err_gate_eff_n(1.0) == pytest.approx(3.0)
    # Large setpoint: the relative term takes over (0.35 * 12 = 4.2 > 3.0).
    assert est._f_err_gate_eff_n(12.0) == pytest.approx(4.2)


def test_proactive_feedforward_in_yaml():
    raw = yaml.safe_load(Path("configs/joint_admittance_8dof.yaml").read_text())
    hm = raw["hybrid_motion"]
    assert hm["proactive_feedforward"] is True
    assert hm["proactive_retract_only"] is False
    assert hm["proactive_gain_mode"] == "ke_normalized"
    assert hm["tau_ff_s"] > 0.0
    assert hm["ke_floor_ff"] > 0.0
    assert hm["proactive_leak_s"] > 0.0
    assert hm["proactive_reset_on_reversal"] is True
    assert hm["v_r_max_m_s"] < hm["max_vz_tool_m_s"]


def test_contact_release_and_force_barrier_enabled_in_8dof_yaml():
    raw = yaml.safe_load(Path("configs/joint_admittance_8dof.yaml").read_text())
    hm = raw["hybrid_motion"]
    assert hm["contact_delta_n"] > 0.0
    assert hm["contact_release_n"] < hm["contact_delta_n"]
    assert hm["contact_release_ticks"] >= 10
    barrier = hm["force_barrier"]
    assert barrier["enabled"] is True
    assert "t_react_s" not in barrier
    assert "fdot_lpf_s" not in barrier
    assert barrier["budget_min_n"] > 0.0
    assert hm["damping_law"] == "trend"
    assert hm["damping_alpha_e"] > 0.0
    assert hm["var_damping_d_u"] == 0.0
    assert hm["seek_vz_m_s"] > 0.0
    assert raw["force"]["causal_fc_hz"] == pytest.approx(10.0)
    assert raw["force"]["causal_order"] == 2


def test_yaml_unified_vz_cap():
    """scan-jitter-fix §2a: the send-path cap max_velocity[2] must equal the
    admittance state cap max_vz_tool_m_s, so the state can never wind up past
    what physics receives.

    Also: there must be NO free-space vs in-contact press cap split
    (approach_vz_tool_m_s is removed) — the 5× press-speed jump at every
    contact latch was the source of the dual press-speed-tier jitter observed
    on /tmp/scan_v5.csv. Bounce control is done by stiff-first K̂_e +
    Dimeas inertia, NOT by capping the cap."""
    raw = yaml.safe_load(Path("configs/joint_admittance.yaml").read_text())
    hm = raw["hybrid_motion"]
    assert "approach_vz_tool_m_s" not in hm, (
        "approach_vz_tool_m_s must be removed — a second vz cap that differs "
        "from max_vz_tool_m_s creates a press-speed switch at contact latch"
    )
    assert hm["max_vz_tool_m_s"] == hm["max_velocity"][2], (
        "state cap and send-path cap must match"
    )


def test_yaml_trajectory_agnostic_pbac():
    """Tangential PBAC gains and vel/accel limits must be equal on tool-X and
    tool-Y so an arbitrary spatial path (arc, spline, teleop) is not biased
    along one tool axis."""
    raw = yaml.safe_load(Path("configs/joint_admittance.yaml").read_text())
    hm = raw["hybrid_motion"]
    kp = hm["kp_pos"]
    mv = hm["max_velocity"]
    ma = hm["max_acceleration"]
    assert kp[0] == kp[1], f"kp_pos tangent axes must be symmetric: {kp[0]} vs {kp[1]}"
    assert mv[0] == mv[1], f"max_velocity tangent axes must be symmetric: {mv[0]} vs {mv[1]}"
    assert ma[0] == ma[1], f"max_acceleration tangent axes must be symmetric: {ma[0]} vs {ma[1]}"


def test_yaml_dead_knobs_removed():
    """Regression guard: knobs deleted by scan-jitter-fix / restore-adaptive-
    bounce-fix must stay deleted (takeover state machine, press-delay Smith
    predictor, chase-vs-approach split, impact chop, detach coast,
    ``approach_vz_tool_m_s`` free-space press cap). The Dimeas variable-
    inertia knobs and the 27c1689 adaptive_ke schedule are ACTIVE and must
    NOT be in the dead list.
    """
    raw = yaml.safe_load(Path("configs/joint_admittance.yaml").read_text())
    hm = raw["hybrid_motion"]
    dead_top = {
        # Takeover state machine (never fires in scripted scans)
        "takeover_enabled", "takeover_eff_n", "takeover_ticks",
        "takeover_release_tau_s", "takeover_tension_n",
        "takeover_release_fz_n", "takeover_overforce_margin_n",
        "takeover_unload_rate_n_s",
        # Press-delay Smith predictor
        "press_delay_comp",
        # Chase-vs-approach vz split, detach coast, impact chop, contact
        # flicker gate — replaced by the single unified vz cap + soft
        # detach decay in adaptive_ke.
        "chase_vz_tool_m_s", "chase_window_s", "detach_coast_s",
        "impact_velocity_retain", "contact_flicker_s",
        # Proactive Iₛ-attenuation and over-force effective-error hack —
        # anti-wind-up is the correct fix (Åström & Rundqwist 1989).
        "proactive_is_atten", "proactive_overforce_eff_n",
        # Free-space vs in-contact press cap split — one cap only.
        "approach_vz_tool_m_s",
    }
    for k in dead_top:
        assert k not in hm, f"removed knob {k!r} came back in hybrid_motion"
