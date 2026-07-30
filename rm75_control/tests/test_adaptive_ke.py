"""Unit tests for online environment stiffness + critical damping.

Duan et al. 2018 eq. 14 asymmetric-λ EWMA of |ΔF/Δx|; Keemink et al. 2018
§III.C critical damping b_d = 2ζ√(m_d K̂_e); stiff-first impact + soft
idle/detach decays (restore-adaptive-bounce-fix). Trajectory-agnostic gates.
"""

import math

import numpy as np

from rm75_control.control.hybrid_motion.adaptive_ke import (
    AdaptiveKeConfig,
    EnvironmentStiffnessEstimator,
)


def test_critical_damping_formula():
    cfg = AdaptiveKeConfig(
        enabled=True,
        zeta=1.0,
        ke_initial=400.0,
        ke_impact_initial=0.0,
        bd_slew_max=1e6,
        bd_max=2000.0,
    )
    est = EnvironmentStiffnessEstimator(cfg, dt=0.01, mass_z=2.0)
    m, ke = 2.0, 400.0
    expected = 2.0 * math.sqrt(m * ke)
    assert abs(est.bd - expected) < 1.0


def test_ewma_stiffness_converges():
    cfg = AdaptiveKeConfig(
        enabled=True,
        zeta=1.0,
        ke_initial=100.0,
        ke_forgetting=0.9,
        ke_forgetting_inc=0.9,   # symmetric here so we can measure convergence rate
        ke_min=0.0,
        ke_max=10000.0,
        dx_threshold_m=1e-5,
        contact_force_n=0.1,
        bd_slew_max=1e6,
        ke_slew_max=1e6,
        gate_lateral_velocity=False,
        gate_df_spike=False,
        f_err_gate_n=1e6,
        settle_ticks=0,
        ke_impact_initial=0.0,   # disable jump so we watch pure learning
        ke_idle_decay_s=0.0,     # disable soft decay so plateau is the EWMA fixed point
    )
    est = EnvironmentStiffnessEstimator(cfg, dt=0.01, mass_z=2.0)
    pose = np.zeros(6)
    true_ke = 800.0
    f = 0.0
    v_force_z = 0.00005 / 0.01
    for _ in range(300):
        f += true_ke * 0.00005
        est.update(f, pose, in_contact=True, mass_z=2.0, v_force_z=v_force_z, v_lateral_m_s=0.0)
    assert 400.0 < est.ke_est < 1200.0
    zeta = est.zeta_eff
    assert 0.85 < zeta <= 1.05


def test_gate_lateral_velocity_is_direction_agnostic():
    """A pure geometric-coupling Fz ripple driven by tangential motion must
    be gated identically regardless of which tool-XY direction the motion is
    in — tool-Y, tool-X, and a diagonal all carry the same tangential speed
    magnitude, and gate_lateral_velocity=True must reject all three. This is
    the trajectory-agnostic invariant that lets an arbitrary spatial path
    reuse the same estimator config.
    """

    def run(gate_enabled: bool, lateral_dir_xy: tuple[float, float]) -> float:
        cfg = AdaptiveKeConfig(
            enabled=True,
            zeta=1.0,
            ke_initial=300.0,
            ke_forgetting=0.99,
            ke_forgetting_inc=0.99,
            ke_min=0.0,
            ke_max=1e6,
            dx_threshold_m=1e-6,
            contact_force_n=0.1,
            bd_slew_max=1e6,
            ke_slew_max=1e6,
            gate_lateral_velocity=gate_enabled,
            lateral_vel_gate_m_s=0.005,
            gate_df_spike=False,
            f_err_gate_n=1e6,
            settle_ticks=0,
            ke_impact_initial=0.0,
            ke_idle_decay_s=0.0,
        )
        est = EnvironmentStiffnessEstimator(cfg, dt=0.01, mass_z=2.0)
        pose = np.zeros(6)
        v_lateral_m_s = float(np.linalg.norm(np.asarray(lateral_dir_xy, dtype=float)))
        v_force_z = 3e-4
        for i in range(200):
            f_ripple = 5.0 * math.sin(2 * math.pi * 9.0 * (i * 0.01))
            est.update(
                f_ripple,
                pose,
                in_contact=True,
                mass_z=2.0,
                v_force_z=v_force_z,
                v_lateral_m_s=v_lateral_m_s,
            )
        return est.ke_est

    tool_y = (0.0, 0.01)
    tool_x = (0.01, 0.0)
    diagonal = (0.01 / math.sqrt(2), 0.01 / math.sqrt(2))

    for lateral in (tool_y, tool_x, diagonal):
        ke = run(True, lateral)
        assert abs(ke - 300.0) < 5.0, f"gated K_e should stay near ke_initial, got {ke}"

    for lateral in (tool_y, tool_x, diagonal):
        ke = run(False, lateral)
        assert abs(ke - 300.0) > 20.0, f"ungated K_e should drift from ke_initial, got {ke}"


def test_stiff_first_impact_jump():
    """On a contact rising edge K̂_e must JUMP UP to ``ke_impact_initial``
    (b_d follows immediately, no slew). Underdamped first-impact ticks on
    a hard surface are what start the bounce cascade — the jump is the
    27c1689 fix that hardware confirmed keeps hard surfaces stable.
    """
    cfg = AdaptiveKeConfig(
        enabled=True,
        zeta=1.0,
        ke_initial=80.0,
        ke_impact_initial=1500.0,
        ke_min=40.0,
        ke_max=2500.0,
        bd_slew_max=1e6,
        gate_lateral_velocity=False,
        gate_df_spike=False,
        settle_ticks=10,      # match production; idle-decay gated by settle window
        ke_idle_decay_s=0.0,  # disable idle decay so the jump is observable in isolation
    )
    est = EnvironmentStiffnessEstimator(cfg, dt=0.005, mass_z=1.0)
    pose = np.zeros(6)
    assert est.ke_est == cfg.ke_initial

    est.update(
        0.0, pose, in_contact=False, mass_z=1.0, v_force_z=0.0, v_lateral_m_s=0.0
    )
    assert est.ke_est == cfg.ke_initial

    est.update(
        1.0, pose, in_contact=True, mass_z=1.0, v_force_z=0.0,
        v_lateral_m_s=0.0, allow_impact_init=True,
    )
    assert est.ke_est == cfg.ke_impact_initial, (
        f"stiff-first jump must set K̂_e = ke_impact_initial on rising edge, "
        f"got {est.ke_est}"
    )
    expected_bd = 2.0 * cfg.zeta * math.sqrt(1.0 * cfg.ke_impact_initial)
    assert abs(est.bd - expected_bd) < 1.0, (
        f"b_d must follow the jump with no slew, got {est.bd} vs {expected_bd}"
    )


def test_stiff_first_gated_by_allow_impact_init():
    """The caller sets ``allow_impact_init=False`` on rising edges that
    follow only a brief flicker (turnaround dip), so the stiff-first jump
    fires on genuine impacts only.
    """
    cfg = AdaptiveKeConfig(
        enabled=True,
        zeta=1.0,
        ke_initial=80.0,
        ke_impact_initial=1500.0,
        bd_slew_max=1e6,
        gate_lateral_velocity=False,
    )
    est = EnvironmentStiffnessEstimator(cfg, dt=0.005, mass_z=1.0)
    pose = np.zeros(6)
    est.update(
        1.0, pose, in_contact=True, mass_z=1.0,
        v_force_z=0.0, v_lateral_m_s=0.0,
        allow_impact_init=False,
    )
    assert est.ke_est == cfg.ke_initial, (
        "allow_impact_init=False must suppress the jump — flicker re-contacts "
        f"must not re-jump; got {est.ke_est}"
    )


def test_detach_decay_preserves_short_bounce_flight():
    """Detach behaviour: K̂_e decays toward ``ke_initial`` with time constant
    ``ke_detach_decay_s`` (soft), NOT a hard reset. A 50 ms bounce flight
    must keep almost all of the just-learned surface stiffness — hard-reset
    on every re-impact under-damped the bounce cycle and produced the
    scan_v5.csv cascade.
    """
    cfg = AdaptiveKeConfig(
        enabled=True,
        zeta=1.0,
        ke_initial=80.0,
        ke_impact_initial=1500.0,
        ke_detach_decay_s=1.0,
        bd_slew_max=1e6,
        gate_lateral_velocity=False,
    )
    est = EnvironmentStiffnessEstimator(cfg, dt=0.005, mass_z=1.0)
    pose = np.zeros(6)
    est.update(
        2.0, pose, in_contact=True, mass_z=1.0,
        v_force_z=0.0, v_lateral_m_s=0.0, allow_impact_init=True,
    )
    ke_after_impact = est.ke_est
    assert ke_after_impact >= 1000.0

    for _ in range(10):  # 50 ms of detach
        est.update(0.0, pose, in_contact=False, mass_z=1.0)
    ke_after_short_bounce = est.ke_est
    # tau=1s, dt=5ms, 10 ticks → drop of ~5 % toward ke_initial.
    drop_frac = (ke_after_impact - ke_after_short_bounce) / (ke_after_impact - cfg.ke_initial)
    assert drop_frac < 0.10, (
        f"a 50 ms bounce flight must not erase learned stiffness: "
        f"dropped {drop_frac*100:.1f}% toward ke_initial"
    )
    assert est.ke_est > 1000.0, (
        "K̂_e must still be well above ke_initial after a 50 ms flight; "
        f"got {est.ke_est}"
    )

    for _ in range(2000):  # 10 s of detach ≈ 10·τ — should fully relax
        est.update(0.0, pose, in_contact=False, mass_z=1.0)
    assert abs(est.ke_est - cfg.ke_initial) < 5.0, (
        f"after 10·τ of detach K̂_e must converge to ke_initial, got {est.ke_est}"
    )


def test_idle_decay_in_steady_contact():
    """When no ΔF/Δx learning update fires this tick AND |f_err|_env is
    inside the gate, K̂_e must relax toward ke_initial (idle decay). The
    press regains bandwidth to chase a receding surface instead of staying
    pinned at ke_impact_initial forever.
    """
    cfg = AdaptiveKeConfig(
        enabled=True,
        zeta=1.0,
        ke_initial=80.0,
        ke_impact_initial=1500.0,
        ke_idle_decay_s=2.0,
        f_err_gate_n=1.2,
        bd_slew_max=1e6,
        gate_lateral_velocity=False,
        gate_df_spike=False,
        settle_ticks=0,
        dx_threshold_m=1.0,   # sabotage the learning branch → force idle-decay path
    )
    est = EnvironmentStiffnessEstimator(cfg, dt=0.005, mass_z=1.0)
    pose = np.zeros(6)
    est.update(
        3.0, pose, in_contact=True, mass_z=1.0,
        v_force_z=0.0, v_lateral_m_s=0.0, allow_impact_init=True,
    )
    ke_after_impact = est.ke_est
    assert ke_after_impact >= 1000.0
    for _ in range(2000):  # 10 s of steady, small-error tracking
        est.update(
            3.0, pose, in_contact=True, mass_z=1.0,
            v_force_z=0.0, v_lateral_m_s=0.0, f_err_z=0.0,
            allow_impact_init=False,
        )
    assert est.ke_est < 0.5 * ke_after_impact, (
        f"idle decay must relax K̂_e toward ke_initial when no learning fires: "
        f"K̂_e stayed at {est.ke_est} after 10 s of small-error tracking"
    )


def test_allow_idle_decay_false_freezes_stiff_first_estimate():
    """A low-load/suspect episode must not look like quiet soft tissue.

    At a 1 N setpoint the free-space residual can leave |f_err| below the
    legacy 1.2 N envelope.  The physical-contact caller therefore needs an
    explicit idle-decay veto until reliable load-bearing contact returns.
    """
    cfg = AdaptiveKeConfig(
        enabled=True,
        zeta=1.0,
        ke_initial=80.0,
        ke_impact_initial=1500.0,
        ke_idle_decay_s=0.20,
        f_err_gate_n=1.2,
        bd_slew_max=1e6,
        gate_lateral_velocity=False,
        gate_df_spike=False,
        settle_ticks=0,
        dx_threshold_m=1.0,  # no ΔF/Δx learning: isolate idle decay
    )
    est = EnvironmentStiffnessEstimator(cfg, dt=0.005, mass_z=1.0)
    pose = np.zeros(6)
    est.update(
        0.2,
        pose,
        in_contact=True,
        mass_z=1.0,
        v_force_z=0.0,
        v_lateral_m_s=0.0,
        f_err_z=0.8,
        f_des_z=1.0,
        allow_impact_init=True,
        allow_idle_decay=False,
    )
    ke_stiff_first = est.ke_est
    assert ke_stiff_first == cfg.ke_impact_initial

    for _ in range(400):  # 2 s = 10 idle-decay time constants
        est.update(
            0.2,
            pose,
            in_contact=True,
            mass_z=1.0,
            v_force_z=0.0,
            v_lateral_m_s=0.0,
            f_err_z=0.8,
            f_des_z=1.0,
            allow_impact_init=False,
            allow_idle_decay=False,
        )
    assert est.ke_est == ke_stiff_first, (
        "allow_idle_decay=False must preserve stiff-first K̂_e even when "
        "the 1 N error envelope would otherwise permit idle decay"
    )

    # Prove the test exercises the idle-decay path: opening the gate must now
    # make the same estimator relax rapidly toward the soft floor.
    for _ in range(400):
        est.update(
            0.2,
            pose,
            in_contact=True,
            mass_z=1.0,
            v_force_z=0.0,
            v_lateral_m_s=0.0,
            f_err_z=0.8,
            f_des_z=1.0,
            allow_impact_init=False,
            allow_idle_decay=True,
        )
    assert est.ke_est < 0.5 * ke_stiff_first


def test_idle_decay_frozen_by_f_err_envelope():
    """During an over-force transient (|f_err|_env > gate) the idle decay
    must FREEZE — dropping K̂_e mid-transient would drop b_d and worsen the
    overshoot. Peak-hold envelope on |f_err| (~0.3 s release) means a single
    zero-crossing during an oscillation doesn't unlock the decay.
    """
    cfg = AdaptiveKeConfig(
        enabled=True,
        zeta=1.0,
        ke_initial=80.0,
        ke_impact_initial=1500.0,
        ke_idle_decay_s=2.0,
        f_err_gate_n=1.0,
        bd_slew_max=1e6,
        gate_lateral_velocity=False,
        gate_df_spike=False,
        settle_ticks=0,
        dx_threshold_m=1.0,
    )
    est = EnvironmentStiffnessEstimator(cfg, dt=0.005, mass_z=1.0)
    pose = np.zeros(6)
    est.update(
        3.0, pose, in_contact=True, mass_z=1.0,
        v_force_z=0.0, v_lateral_m_s=0.0, allow_impact_init=True,
    )
    ke_after_impact = est.ke_est
    for _ in range(400):  # 2 s with |f_err|=2 (well above 1 N gate)
        est.update(
            5.0, pose, in_contact=True, mass_z=1.0,
            v_force_z=0.0, v_lateral_m_s=0.0, f_err_z=2.0,
            allow_impact_init=False,
        )
    assert est.ke_est > 0.9 * ke_after_impact, (
        "over-force gate must freeze idle decay; "
        f"K̂_e dropped from {ke_after_impact} to {est.ke_est} during an over-force transient"
    )


def test_asymmetric_forgetting_biases_toward_stiffer_estimate():
    """With ``ke_forgetting_inc`` < ``ke_forgetting`` the estimator tracks a
    rising true stiffness fast (impact-safe) and forgets a soft reading
    slow (avoids over-reacting to a quiet tick). Reference from the 27c1689
    schedule; verified against a step-up + step-down of true K_e.
    """
    def run(true_ke_seq: list[float], cfg: AdaptiveKeConfig) -> float:
        est = EnvironmentStiffnessEstimator(cfg, dt=0.01, mass_z=1.0)
        pose = np.zeros(6)
        f = 0.0
        v_force_z = 5e-3
        for true_ke in true_ke_seq:
            f += true_ke * v_force_z * 0.01
            est.update(
                f, pose, in_contact=True, mass_z=1.0,
                v_force_z=v_force_z, v_lateral_m_s=0.0,
                allow_impact_init=False,
            )
        return est.ke_est

    cfg_kwargs = dict(
        enabled=True,
        zeta=1.0,
        ke_initial=500.0,
        ke_min=0.0,
        ke_max=10000.0,
        dx_threshold_m=1e-6,
        contact_force_n=0.1,
        bd_slew_max=1e6,
        ke_slew_max=1e6,
        gate_lateral_velocity=False,
        gate_df_spike=False,
        f_err_gate_n=1e6,
        settle_ticks=0,
        ke_impact_initial=0.0,
        ke_idle_decay_s=0.0,
    )
    # Fast-up config: ke_forgetting_inc=0.88, ke_forgetting=0.995
    fast_up = AdaptiveKeConfig(ke_forgetting=0.995, ke_forgetting_inc=0.88, **cfg_kwargs)
    # Symmetric baseline: both = 0.995 (slow both ways)
    symm = AdaptiveKeConfig(ke_forgetting=0.995, ke_forgetting_inc=0.995, **cfg_kwargs)

    step_up = [2000.0] * 300  # true K_e rises above initial
    assert run(step_up, fast_up) > run(step_up, symm), (
        "asymmetric ke_forgetting_inc < ke_forgetting must track a stiffness "
        "increase faster than a symmetric-slow lambda"
    )


def test_reset_seeds_ke_initial():
    """``reset()`` restores new-session semantics: K̂_e = ke_initial. This is
    separate from contact-loss behaviour (which does a soft decay only).
    """
    cfg = AdaptiveKeConfig(
        enabled=True,
        ke_initial=80.0,
        ke_impact_initial=1500.0,
        bd_slew_max=1e6,
    )
    est = EnvironmentStiffnessEstimator(cfg, dt=0.005, mass_z=1.0)
    pose = np.zeros(6)
    est.update(
        2.0, pose, in_contact=True, mass_z=1.0,
        v_force_z=0.0, v_lateral_m_s=0.0, allow_impact_init=True,
    )
    assert est.ke_est >= 1000.0
    est.reset()
    assert est.ke_est == cfg.ke_initial
