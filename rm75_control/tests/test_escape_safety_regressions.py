"""Deterministic regressions for the latched rail singularity escape.

These tests deliberately exercise the safety contracts at the QP/task
boundary rather than replaying a hardware log.  The rail escape is a
one-episode, one-way request; a soft objective alone is not enough because a
Cartesian task can otherwise make the solved rail velocity point the wrong
way near a singular posture.
"""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.collision_model import (
    CollisionConfig,
)
from rm75_control.control.joint_admittance_8dof.loop import (
    JointIkConfig,
    JointIkController,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import (
    QpConfig,
    QpIkController,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_extension import (
    RailExtensionConfig,
    RailExtensionTask,
)
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimiter


DT = 0.005
RAIL_ESCAPE_V_MAX = 0.020


def _safe_q() -> np.ndarray:
    """Bent, mid-rail posture used by all deterministic fixtures."""
    return np.array(
        [
            0.40,
            -0.905938,
            1.117987,
            0.459109,
            1.775407,
            -0.342094,
            1.06775,
            0.749873,
        ],
        dtype=float,
    )


def _make_qp(kin: RobotKinematics) -> QpIkController:
    # A small, explicit rail acceleration makes the wrong-direction recovery
    # observable in a short deterministic sequence.  The hard rail envelope
    # must remain authoritative even when the soft rail objective asks for
    # +80 mm/s (opposite to a - escape episode).
    limits = SafetyLimits.from_kinematics(
        kin,
        v_scale=0.5,
        a_max=np.full(kin.nv, 4.0, dtype=float),
        position_margin=np.zeros(kin.nv, dtype=float),
    )
    return QpIkController(
        kin,
        limits,
        QpConfig(
            backend="osqp",
            task_weight=np.array([100.0, 100.0, 100.0, 50.0, 50.0, 50.0]),
            collision=CollisionConfig(enabled=False),
            rail_task_weight_hard_max=2.0,
            rail_task_weight_max_frac=0.2,
        ),
    )


def test_qp_escape_wrong_direction_is_accel_limited_and_converges_to_one_way_zone():
    """A stale +rail velocity must decay toward zero, never cross the - latch."""
    kin = RobotKinematics()
    ctrl = _make_qp(kin)
    q = _safe_q()
    ctrl.reset(q)
    # Simulate the exact hardware failure mode: the escape latch is - but the
    # previous solved command is still +80 mm/s.  The first ticks may remain
    # positive because of the acceleration envelope, but each must approach
    # zero rather than accelerate farther in the wrong direction.
    ctrl.qdot_prev[0] = 0.080
    wrong_direction: list[float] = []
    for _ in range(70):
        r = ctrl.step(
            q,
            np.zeros(6),
            DT,
            rail_task_vel_m_s=0.080,
            rail_task_weight=2.0,
            rail_escape_sign=-1.0,
            rail_escape_v_max_m_s=RAIL_ESCAPE_V_MAX,
        )
        v = float(r.qdot[0])
        # The latched - direction is the interval [-20, 0] mm/s.  If a stale
        # positive command survives an acceleration-limited tick, it is
        # allowed only while monotonically returning toward zero.
        if v > 1.0e-6:
            if wrong_direction:
                assert v <= wrong_direction[-1] + 1.0e-7
            wrong_direction.append(v)
        else:
            assert v >= -RAIL_ESCAPE_V_MAX - 1.0e-6
        q = r.q_next

    # A stricter implementation may project directly into the hard interval,
    # in which case this list is empty.  If stale opposite-sign velocity is
    # retained for an accel-limited tick, every such tick must approach zero.
    if wrong_direction:
        assert max(wrong_direction) <= 0.080 + 1.0e-7
    assert abs(float(ctrl.qdot_prev[0])) <= RAIL_ESCAPE_V_MAX + 1.0e-6


def test_qp_escape_stop_decelerates_to_zero_without_reversing():
    """A travel-stop command closes the rail interval at zero."""
    kin = RobotKinematics()
    ctrl = _make_qp(kin)
    q = _safe_q()
    ctrl.reset(q)
    ctrl.qdot_prev[0] = -RAIL_ESCAPE_V_MAX
    values: list[float] = []
    for _ in range(40):
        r = ctrl.step(
            q,
            np.zeros(6),
            DT,
            rail_task_vel_m_s=-0.080,
            rail_task_weight=2.0,
            rail_escape_sign=-1.0,
            rail_escape_stop=True,
            rail_escape_v_max_m_s=RAIL_ESCAPE_V_MAX,
        )
        v = float(r.qdot[0])
        values.append(v)
        assert v <= 1.0e-6, f"stop command still drives rail: {v:.6f} m/s"
        assert v >= -RAIL_ESCAPE_V_MAX - 1.0e-6
        q = r.q_next

    assert values[0] <= 0.0
    assert abs(float(ctrl.qdot_prev[0])) <= 1.0e-6


def test_escape_slew_override_brakes_stale_rail_at_or_below_point8_in_qp_and_host():
    """The signed episode uses the bridge's 0.8 m/s² slew in both clamps.

    A stale +80 mm/s rail command is intentionally opposed by a -escape.  It
    must monotonically decelerate, never exceed 0.8 m/s², and enter the signed
    [-20, -10] mm/s interval once physically reachable.  The same sequence is
    replayed through the downstream position SafetyLimiter to catch an
    acceleration-history mismatch between the QP and host command path.
    """
    kin = RobotKinematics()
    limits = SafetyLimits.from_kinematics(
        kin,
        v_scale=0.5,
        a_max=np.array([0.30, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]),
        position_margin=np.zeros(kin.nv, dtype=float),
    )
    ctrl = QpIkController(
        kin,
        limits,
        QpConfig(
            backend="osqp",
            task_weight=np.array([100.0, 100.0, 100.0, 50.0, 50.0, 50.0]),
            collision=CollisionConfig(enabled=False),
            rail_task_weight_hard_max=2.0,
            rail_task_weight_max_frac=0.2,
            rail_escape_accel_m_s2=0.80,
        ),
    )
    q = _safe_q()
    ctrl.reset(q)
    ctrl.qdot_prev[0] = 0.080
    qp_values: list[float] = []
    for _ in range(30):
        r = ctrl.step(
            q,
            np.zeros(6),
            DT,
            rail_task_vel_m_s=-0.020,
            rail_task_weight=2.0,
            rail_escape_sign=-1.0,
            rail_escape_active=True,
            rail_escape_v_max_m_s=RAIL_ESCAPE_V_MAX,
            rail_escape_accel_m_s2=0.80,
        )
        v = float(r.qdot[0])
        qp_values.append(v)
        if len(qp_values) > 1:
            assert v <= qp_values[-2] + 1.0e-7
        assert abs((v - (0.080 if len(qp_values) == 1 else qp_values[-2])) / DT) <= 0.80 + 2.0e-5
        q = r.q_next
    assert any(-RAIL_ESCAPE_V_MAX - 1.0e-6 <= v <= -0.010 + 1.0e-6 for v in qp_values)

    # Re-run the stale-history transition through the host position clamp.
    limiter = SafetyLimiter(limits)
    q_host = _safe_q()
    stale_dq = np.zeros(kin.nv, dtype=float)
    stale_dq[0] = 0.080 * DT
    limiter.sync_applied_delta(stale_dq, DT)
    host_values: list[float] = []
    for _ in range(30):
        q_target = q_host.copy()
        q_target[0] += -RAIL_ESCAPE_V_MAX * DT
        rep = limiter.clamp(
            q_host,
            q_target,
            DT,
            rail_escape_active=True,
            rail_escape_sign=-1.0,
            rail_escape_accel_m_s2=0.80,
        )
        v = float(rep.dq[0] / DT)
        host_values.append(v)
        if len(host_values) > 1:
            assert v <= host_values[-2] + 1.0e-7
        assert abs((v - (0.080 if len(host_values) == 1 else host_values[-2])) / DT) <= 0.80 + 2.0e-5
        q_host = rep.q_safe
    assert any(-RAIL_ESCAPE_V_MAX - 1.0e-6 <= v <= -0.010 + 1.0e-6 for v in host_values)


def test_rail_escape_episode_stops_after_80mm_and_mode_reset_clears_latch():
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(
            escape_max_travel_m=0.080,
            escape_v_max_m_s=RAIL_ESCAPE_V_MAX,
        ),
    )
    q = _safe_q()
    task.reset(q)

    # Zero extension error means the only requested motion is singularity
    # escape.  Move the measured rail in the latched direction until the
    # episode budget is exhausted.
    v, _ = task(
        q,
        sigma_scale=0.0,
        sigma_escape_scale=0.0,
        sigma_grad_rail=1.0,
        sigma_min=0.05,
    )
    assert task.escape_active
    assert task.escape_sign > 0.0
    assert v > 0.0

    for k in range(1, 8):
        q_k = q.copy()
        q_k[0] += 0.020 * k
        v, _ = task(
            q_k,
            sigma_scale=0.0,
            sigma_escape_scale=0.0,
            sigma_grad_rail=1.0,
            sigma_min=0.05,
        )
        if task.escape_stopped:
            assert task.escape_travel_m >= 0.080 - 1.0e-9
            assert abs(v) <= 1.0e-12
            break
    else:
        pytest.fail("escape episode never reached its 80 mm travel budget")

    task.set_mode("pose_attract")
    assert not task.escape_active
    assert task.escape_sign == pytest.approx(0.0)
    assert task.escape_travel_m == pytest.approx(0.0)
    assert not task.escape_stopped


def test_outward_escape_at_soft_band_does_not_teleport_target():
    """An outward latch outside soft_min stops at the measured host point."""
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(soft_min_m=0.01, soft_max_m=0.78),
    )
    q = _safe_q()
    q[0] = 0.0
    task.reset(q)
    v, _ = task(
        q,
        sigma_scale=0.0,
        sigma_escape_scale=0.0,
        sigma_grad_rail=-1.0,
        sigma_min=0.05,
    )
    assert task.escape_active
    assert task.escape_sign < 0.0
    assert task.escape_stopped
    assert v == pytest.approx(0.0)
    assert task.escape_position_limit_m == pytest.approx(q[0])

    # A reset must clear a stopped episode even if the mode does not change.
    task.set_mode("reach")
    task.reset(q)
    assert not task.escape_active
    assert task.escape_sign == pytest.approx(0.0)
    assert task.escape_travel_m == pytest.approx(0.0)
    assert not task.escape_stopped


def test_limit_activation_pretriggers_escape_but_healthy_does_not():
    """Near-limit posture starts escape at σ<0.12; healthy D stays out."""
    kin = RobotKinematics()
    ctrl = JointIkController(
        kin,
        JointIkConfig(
            qp=QpConfig(
                backend="osqp",
                collision=CollisionConfig(enabled=False),
            ),
            rail_extension=RailExtensionConfig(enabled=False),
        ),
    )
    ctrl.reset(_safe_q())

    # D-like sigma=0.115 is above the normal .10 enter threshold.  Low limit
    # activation must therefore remain healthy, while the same sigma with a
    # nearly-saturated joint is an early escape trigger.
    ctrl._in_escape_zone = False
    ctrl._update_escape_zone(
        0.115,
        0.10,
        0.12,
        limit_activation=0.10,
        limit_escape_activation=0.80,
        sigma_limit_escape_enter=0.12,
    )
    assert not ctrl.escape_active

    ctrl._update_escape_zone(
        0.115,
        0.10,
        0.12,
        limit_activation=0.85,
        limit_escape_activation=0.80,
        sigma_limit_escape_enter=0.12,
    )
    assert ctrl.escape_active

    # Limit activation alone must not fire above the early-sigma threshold.
    ctrl._in_escape_zone = False
    ctrl._update_escape_zone(
        0.130,
        0.10,
        0.12,
        limit_activation=0.95,
        limit_escape_activation=0.80,
        sigma_limit_escape_enter=0.12,
    )
    assert not ctrl.escape_active


def test_qp_rail_weight_is_capped_by_absolute_and_fractional_hierarchy(
    monkeypatch,
):
    """Rail soft cost stays <=2 and <=20% of effective translation weight."""
    kin = RobotKinematics()
    # Keep this test about hierarchy, not which fixture posture happens to be
    # close to singular.  A healthy singular-value report makes the effective
    # translation weight equal to the configured [10, 12, 20] minimum.
    monkeypatch.setattr(
        kin,
        "singular_values",
        lambda _J: np.ones(6, dtype=float),
    )
    limits = SafetyLimits.from_kinematics(
        kin,
        v_scale=0.5,
        a_max=20.0,
        position_margin=np.zeros(kin.nv, dtype=float),
    )
    ctrl = QpIkController(
        kin,
        limits,
        QpConfig(
            backend="osqp",
            task_weight=np.array([10.0, 12.0, 20.0, 5.0, 5.0, 5.0]),
            collision=CollisionConfig(enabled=False),
            rail_task_weight_hard_max=2.0,
            rail_task_weight_max_frac=0.2,
        ),
    )
    q = _safe_q()
    ctrl.reset(q)
    r = ctrl.step(
        q,
        np.zeros(6),
        DT,
        rail_task_vel_m_s=0.08,
        rail_task_weight=100.0,
    )
    assert r.rail_task_weight_effective <= 2.0 + 1.0e-9
    assert r.rail_task_weight_effective <= (
        0.2 * r.cart_translation_weight_effective + 1.0e-9
    )


def test_coupled_escape_overrides_opposing_plan_rail_pin(monkeypatch):
    """A joint-plan pin cannot bypass the signed rail safety envelope."""
    kin = RobotKinematics()
    cfg = JointIkConfig(
        dt=DT,
        qp=QpConfig(
            backend="osqp",
            collision=CollisionConfig(enabled=False),
            task_weight=np.array([100.0, 100.0, 100.0, 50.0, 50.0, 50.0]),
        ),
        rail_extension=RailExtensionConfig(
            enabled=True,
            w_sigma_floor=1000.0,
            escape_v_min_m_s=0.010,
            escape_v_max_m_s=RAIL_ESCAPE_V_MAX,
        ),
    )
    ctrl = JointIkController(kin, cfg)
    q = np.array(
        [
            0.413360,
            0.071803,
            -0.344304,
            -0.043772,
            1.180648,
            -0.025826,
            1.278589,
            1.599848,
        ],
        dtype=float,
    )
    monkeypatch.setattr(
        ctrl._rail_goodness, "refresh", lambda _q, force=False: (0.0, 20.0)
    )
    ctrl.set_coupled()
    ctrl.set_plan_drives_rail(True)
    ctrl.reset(q)

    qdot_ff = np.zeros(8, dtype=float)
    qdot_ff[0] = -0.080  # deliberately opposite the fixed + escape gradient
    step = ctrl.update(np.zeros(6), dt=DT, qdot_ff=qdot_ff)

    assert step.escape_active
    assert step.rail_escape_active
    assert step.rail_escape_sign > 0.0
    assert not step.plan_drives_rail
    assert np.isnan(step.rail_vel_pin)
    assert step.qdot[0] >= -1.0e-9
    assert step.qdot[0] <= RAIL_ESCAPE_V_MAX + 1.0e-6
    assert step.q_send[0] >= q[0] - 1.0e-12
