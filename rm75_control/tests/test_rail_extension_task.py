"""Preferred-extension rail task: scheduling, dead zone, QP integration."""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.tasks.rail_extension import (
    RailExtensionConfig,
    RailExtensionTask,
)

Q_D = np.array(
    [0.0, -0.949552, 0.095255, 0.646858, 1.469911, 0.502701, 0.666503, -0.338137]
)


def _task(**kw) -> tuple[RailExtensionTask, RobotKinematics]:
    kin = RobotKinematics()
    cfg = RailExtensionConfig(**kw)
    task = RailExtensionTask(kin, cfg)
    task.reset(Q_D)
    return task, kin


def _force_err(task: RailExtensionTask, err: float) -> None:
    """Impose an extension error by offsetting the preferred extension.

    Extension e = y_tcp - y_rail is invariant to a pure rail shift (the TCP
    translates with the base), so unit tests create error via d_pref: the same
    error the closed loop sees when the arm reaches while the rail lags.
    """
    task.capture_reference(Q_D)
    task.d_pref_m = float(task.d_pref_m) - float(err)


def test_dead_zone_zero_weight():
    """Inside the dead zone the weight is exactly 0: the rail does not wander."""
    task, _ = _task(e0_m=0.05, e1_m=0.15)
    # At the capture posture the error is zero by construction.
    v, w = task(Q_D)
    assert w == 0.0
    assert abs(task.last_err_m) < 1e-12
    # Small extension error +0.03 < e0 -> still zero weight.
    _force_err(task, 0.03)
    v, w = task(Q_D)
    assert abs(task.last_err_m - 0.03) < 1e-9
    assert w == 0.0


def test_direction_and_saturation():
    """err > 0 (TCP ahead of the rail) -> rail chases +Y; velocity capped."""
    task, _ = _task(k_ext=1.0, v_max_m_s=0.08, e0_m=0.05, e1_m=0.15)
    _force_err(task, 0.20)  # beyond e1
    v, w = task(Q_D)
    assert v > 0.0
    assert abs(v - 0.08) < 1e-12  # capped at v_max
    assert abs(w - task.cfg.w_max) < 1e-12  # full authority
    # Symmetric for negative error.
    _force_err(task, -0.20)
    v, w = task(Q_D)
    assert v < 0.0
    assert abs(w - task.cfg.w_max) < 1e-12


def test_weight_schedule_continuous_and_monotone():
    """w(err) is C1 (smoothstep): continuous, monotone, zero-slope at both
    ends of the band — no hard switching anywhere."""
    task, _ = _task(e0_m=0.05, e1_m=0.15, w_max=5.0)

    def w_of(err: float) -> float:
        _force_err(task, err)
        _, w = task(Q_D)
        return w

    errs = np.linspace(0.0, 0.25, 501)
    ws = np.array([w_of(e) for e in errs])
    # Monotone non-decreasing.
    assert np.all(np.diff(ws) >= -1e-9)
    # Continuous: no jump larger than the local slope allows.
    assert np.max(np.abs(np.diff(ws))) < 5.0 * 2.0 * (errs[1] - errs[0]) / 0.1 + 1e-6
    # C1 at the band edges: numerical derivative ~0 at e0 and e1.
    h = 1e-5
    for edge in (0.05, 0.15):
        d = (w_of(edge + h) - w_of(edge - h)) / (2 * h)
        assert abs(d) < 0.05, (edge, d)
    # Zero inside dead zone, w_max above e1.
    assert w_of(0.049) == 0.0
    assert abs(w_of(0.151) - 5.0) < 1e-9


def test_extension_invariant_to_coupled_translation():
    """Moving rail and TCP together (pure base translation) keeps extension
    constant — the task only reacts to ARM reach, not to absolute position."""
    task, kin = _task()
    e0 = task.extension(Q_D)
    q = Q_D.copy()
    q[0] += 0.12  # rail +0.12 -> TCP translates +0.12 with it
    e1 = task.extension(q)
    assert abs(e1 - e0) < 1e-9


def test_rail_at_travel_limit_zeros_task():
    """When rail is pinned at +limit and task wants +v, weight must be 0."""
    task, kin = _task(e0_m=0.05, e1_m=0.15, w_max=1.5)
    _force_err(task, 0.20)
    q = Q_D.copy()
    q[0] = float(kin.q_upper[0])  # at upper travel limit
    v, w = task(q)
    assert w == 0.0
    assert v == 0.0
    assert task.last_limit_saturated


def test_limit_fade_smoothstep_continuous():
    """Approaching a limit: _limit_saturation is C¹ smoothstep, no linear cliff."""
    task, kin = _task(limit_margin_m=0.08)
    hi = float(kin.q_upper[0])
    margin = 0.08
    qs = np.linspace(hi - margin, hi - 1e-9, 401)
    scales = np.array([task._limit_saturation(q, 0.05) for q in qs])
    assert scales[0] > scales[-1]
    assert np.max(np.abs(np.diff(scales))) < 0.01
    assert abs(scales[0] - 1.0) < 1e-9
    assert scales[-1] < 1e-6
    h = 1e-5
    d_entry = (
        task._limit_saturation(hi - margin + h, 0.05)
        - task._limit_saturation(hi - margin - h, 0.05)
    ) / (2 * h)
    d_exit = (
        task._limit_saturation(hi - h, 0.05)
        - task._limit_saturation(hi - 2 * h, 0.05)
    ) / h
    assert abs(d_entry) < 0.05, d_entry
    assert abs(d_exit) < 0.05, d_exit


def test_sigma_scale_boosts_weight():
    """Bug 2: σ dropping BOOSTS rail authority (was previously inverted).

    The rail is our primary singularity-avoidance mechanism, so σ dips must
    make the QP take it MORE seriously, not less.  The invariant kept:
    ``w_max*(1+k_sigma_boost) ≪ W_task`` so the QP order  slack > rail  is
    preserved even at the peak boost."""
    task, _ = _task(e0_m=0.05, e1_m=0.15, w_max=1.5)
    _force_err(task, 0.20)
    _, w_full = task(Q_D, sigma_scale=1.0)
    _, w_half = task(Q_D, sigma_scale=0.5)
    _, w_zero = task(Q_D, sigma_scale=0.0)
    assert w_half > w_full  # low σ → boost, not cut
    assert w_zero > w_half
    # invariant: total boost bounded by (1 + k_sigma_boost) = 3 at σ=0
    assert w_zero <= (1.0 + task.cfg.k_sigma_boost) * (
        w_full + task.cfg.w_sigma_floor
    ) + 1e-9


def test_feedforward_weight_in_dead_zone_during_scan():
    """Reference vel_ff keeps rail authority during motion even when |err| < e0."""
    task, kin = _task(e0_m=0.01, e1_m=0.06, w_max=1.5, k_ff=1.0)
    task.capture_reference(Q_D)
    vel0 = np.zeros(6)
    vel_ff = np.zeros(6)
    vel_ff[1] = 0.04
    v0, w0 = task(Q_D, sigma_scale=1.0, vel_ff=vel0)
    v_ff, w_ff = task(Q_D, sigma_scale=1.0, vel_ff=vel_ff)
    assert abs(v0) < 1e-9
    assert w0 == 0.0
    assert v_ff > 0.03
    assert w_ff > 0.5


def test_sigma_escape_anti_oppose_only_when_healthy():
    """Opposing σ-escape is blocked only when σ is healthy (≥ sigma_guard_enter).

    Below enter, escape must be allowed to fight reach/FF so the rail can
    pull the arm out of a bad region.
    """
    task, _ = _task(e0_m=0.01, e1_m=0.06, k_ff=1.0, k_esc=0.5)
    q = Q_D.copy()
    q[0] = 0.40  # mid-travel so -Y escape is not limit-saturated
    task.reset(q)
    task.capture_reference(q)
    vel_ff = np.zeros(6)
    vel_ff[1] = 0.04
    # sigma_scale=0.5 ≥ enter(0.45): anti-oppose → escape zeroed, v follows +FF
    v_blocked, _ = task(q, sigma_scale=0.5, sigma_grad_rail=-1.0, vel_ff=vel_ff)
    assert v_blocked > 0.0
    # sigma_scale=0.2 < enter: escape wins against FF → net rail velocity flips
    v_free, _ = task(q, sigma_scale=0.2, sigma_grad_rail=-1.0, vel_ff=vel_ff)
    assert v_free < 0.0
    assert v_free < v_blocked


def test_sigma_grad_activates_escape_velocity_in_dead_zone():
    """σ-escape term must produce a rail velocity even inside |err| < e0
    when σ drops.  Zero grad and no reach error → v ≈ 0."""
    task, _ = _task(e0_m=0.05, e1_m=0.15, w_max=1.5)
    # Zero reach error: capture reference at current pose so err ≡ 0.
    task.capture_reference(Q_D)
    v0, w0 = task(Q_D, sigma_scale=1.0, sigma_grad_rail=0.0)
    # σ low + non-zero gradient → escape velocity fires.
    v_esc, w_esc = task(Q_D, sigma_scale=0.0, sigma_grad_rail=1.0)
    assert abs(v0) < 1e-9  # dead zone + healthy σ = silence
    assert v_esc > 0.0  # positive gradient → +Y rail move
    assert w_esc > w0  # w_sigma_floor kicks in even inside dead zone


def test_qp_consumes_rail_task():
    """The WBC QP moves the rail toward the task velocity when weighted, and
    keeps it parked when the weight is zero (twist = 0 -> no primary demand)."""
    from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
    from rm75_control.control.joint_admittance_8dof.solver.qp_builder import (
        QpConfig,
        QpIkController,
    )
    from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits

    kin = RobotKinematics()
    limits = SafetyLimits.from_kinematics(kin, v_scale=0.5, a_max=20.0)
    # Production-like weighting: the Cartesian equality must dominate the
    # rail task (w_ext=5) so the arm compensates the rail and the TCP holds.
    ctrl = QpIkController(
        kin,
        limits,
        QpConfig(
            task_weight=np.array([100.0, 100.0, 100.0, 50.0, 50.0, 50.0]),
            collision=CollisionConfig(enabled=False),
        ),
    )
    ctrl.reset(Q_D)
    # Weighted task: rail follows v_des (TCP held by the primary equality).
    r = ctrl.step(
        Q_D, np.zeros(6), 0.005, rail_task_vel_m_s=0.05, rail_task_weight=1.5
    )
    assert r.qdot[0] > 0.025, r.qdot[0]
    # TCP stays: the Cartesian rows dominate, arm compensates the rail.
    v_tcp = kin.jacobian(Q_D) @ r.qdot
    assert np.linalg.norm(v_tcp[:3]) < 0.005, v_tcp
    # Zero weight: rail does not move.
    ctrl.reset(Q_D)
    r = ctrl.step(Q_D, np.zeros(6), 0.005, rail_task_vel_m_s=0.05, rail_task_weight=0.0)
    assert abs(r.qdot[0]) < 1e-4, r.qdot[0]
