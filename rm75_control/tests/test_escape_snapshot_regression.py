"""Offline regression for the 2026-08-10 singularity escape snapshot.

The hardware log is intentionally not read by this test.  The first E4 tick
from ``run_20260810_162413.csv`` is kept as a small fixture instead.  It is a
useful guard against bringing back the old failure mode in which a soft rail
hint was overcome by the Cartesian QP and the rail swept roughly 224 mm.
"""

from __future__ import annotations

import csv
import warnings

import numpy as np

from rm75_control.control.joint_admittance_8dof.collision_model import (
    CollisionConfig,
)
from rm75_control.control.joint_admittance_8dof.loop import (
    JointIkConfig,
    JointIkController,
    JointIkStep,
    _TickLogger,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import QpConfig
from rm75_control.control.joint_admittance_8dof.tasks.rail_extension import (
    RailExtensionConfig,
)


DT_S = 0.005
ESCAPE_V_MAX_M_S = 0.020
ESCAPE_TRAVEL_MAX_M = 0.080

# E4 entry (t_wall_s=73.3827) from run_20260810_162413.csv.  Keeping this
# constant makes the regression deterministic and independent of mounted media
# or a stale log directory.
E4_Q_ENTRY = np.array(
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


def _e4_controller() -> JointIkController:
    """Build the smallest coupled WBC configuration that can enter escape."""
    kin = RobotKinematics()
    cfg = JointIkConfig(
        dt=DT_S,
        control_frame="base",
        a_max_rail_m_s2=0.30,
        qp=QpConfig(
            # Production joint_admittance_8dof.yaml uses ProxQP; this snapshot
            # guards the deployed solver path rather than an easier surrogate.
            backend="proxqp",
            collision=CollisionConfig(enabled=False),
            task_weight=np.array([100.0, 100.0, 100.0, 50.0, 50.0, 50.0]),
            use_dyn_nullspace=False,
            rail_task_weight_hard_max=2.0,
            rail_task_weight_max_frac=0.20,
            # Match production: a latched episode must produce real side
            # motion instead of remaining a near-zero soft preference.
            rail_escape_v_min_m_s=0.010,
            rail_escape_v_max_m_s=ESCAPE_V_MAX_M_S,
        ),
        rail_extension=RailExtensionConfig(
            enabled=True,
            # E4 enters just under σ=0.10, so the nominal σ-floor is tiny at
            # that boundary.  Raise only this offline hint weight to make the
            # bounded-stop path observable in a few seconds; the QP envelope
            # and 80 mm budget remain the production invariants under test.
            w_sigma_floor=1000.0,
            escape_v_min_m_s=0.0,
            escape_v_max_m_s=ESCAPE_V_MAX_M_S,
            escape_max_travel_m=ESCAPE_TRAVEL_MAX_M,
        ),
    )
    ctrl = JointIkController(kin, cfg)
    # The real finite-difference gradient is deliberately replaced with a
    # fixed, valid direction.  This isolates the episode/velocity/travel
    # contract from numerical gradient noise while still exercising the real
    # RobotKinematics, QP, safety clamp, and RailExtensionTask.
    # Use a deliberately strong but finite gradient so the rail reaches the
    # configured 20 mm/s envelope in this short replay; direction is the only
    # geometry quantity under test here.
    ctrl._rail_goodness.refresh = lambda q, force=False: (0.0, 20.0)
    ctrl.set_coupled()
    ctrl.reset(E4_Q_ENTRY)
    return ctrl


def test_e4_escape_is_one_way_slow_and_bounded():
    """Replay the E4 posture without reproducing the +224 mm rail sweep."""
    ctrl = _e4_controller()
    rail_target = []
    rail_qdot = []
    stopped = False

    # 11.5 s at 200 Hz is enough to reach the 80 mm episode budget with the
    # conservative QP/safety limits.  No external log is
    # needed; only the E4 posture and a deterministic sigma-gradient are used.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(2300):
            step = ctrl.update(np.zeros(6, dtype=float), dt=DT_S)
            rail_target.append(float(step.q_send[0]))
            rail_qdot.append(float(step.qdot[0]))
            if step.rail_escape_stopped:
                stopped = True
                # The stop is a hard episode boundary.  A few extra ticks ensure
                # the QP does not restart or reverse the rail after the stop.
                for _ in range(20):
                    after = ctrl.update(np.zeros(6, dtype=float), dt=DT_S)
                    rail_target.append(float(after.q_send[0]))
                    rail_qdot.append(float(after.qdot[0]))
                break

    assert stopped, "E4 escape did not reach its bounded travel stop"
    target = np.asarray(rail_target)
    velocity = np.asarray(rail_qdot)
    delta = np.diff(target)

    # Fixed +gradient fixture means this episode may only move toward +rail.
    assert np.all(delta >= -1.0e-8)
    # qdot is the actual WBC command, not the soft task request.
    assert np.max(np.abs(velocity)) <= ESCAPE_V_MAX_M_S + 1.0e-6
    assert np.max(velocity) >= 0.009
    # The largest allowed episode displacement is 80 mm, rather than the old
    # E4 sweep of about 224 mm.
    # The final host target is clamped at the episode boundary after all
    # velocity/lead processing, so command travel itself cannot coast past
    # 80 mm (the hardware bridge still owns physical motor deceleration).
    assert target[-1] - E4_Q_ENTRY[0] <= ESCAPE_TRAVEL_MAX_M + 1.0e-9
    assert target[-1] - E4_Q_ENTRY[0] >= ESCAPE_TRAVEL_MAX_M - 2.0e-3
    assert np.max(target) - np.min(target) < 0.10

    # Once stopped, the target and actual rail command stay still and do not
    # start a second/opposite-direction episode.
    assert np.max(np.abs(np.diff(target[-20:]))) <= 1.0e-7
    assert np.max(np.abs(velocity[-20:])) <= 1.0e-6

    # Telemetry is the final integrated host command, not a stale pre-safety
    # QP iterate.  This also keeps SafetyLimiter history synchronized after a
    # position/lead clamp.
    assert np.allclose(
        velocity[1:], np.diff(target) / DT_S, atol=1.0e-10, rtol=0.0
    )


def test_tick_logger_header_and_row_lengths(tmp_path):
    """Append-only telemetry schema emits rows matching its header."""
    path = tmp_path / "escape_snapshot.csv"
    logger = _TickLogger(str(path))
    try:
        step = JointIkStep(
            q_send=np.zeros(8),
            qdot=np.zeros(8),
            twist_base=np.zeros(6),
            sigma_min=0.1,
            manip=0.0,
            slack_norm=0.0,
            n_cbf_active=0,
            follow_err_rad=0.0,
            rail_escape_active=True,
            rail_escape_sign=1.0,
            rail_escape_stopped=True,
            rail_escape_travel_m=ESCAPE_TRAVEL_MAX_M,
            rail_escape_v_des_m_s=ESCAPE_V_MAX_M_S,
        )
        logger.write(
            0.0,
            "snapshot",
            0.0,
            step,
            np.zeros(8),
            np.zeros(6),
            np.zeros(6),
            v_max=np.ones(8),
            feedback_fresh_tick=True,
        )
    finally:
        logger.close()

    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 2
    assert len(rows[0]) == len(_TickLogger._HEADER)
    assert len(rows[1]) == len(rows[0])
    assert len(rows[0]) == len(set(rows[0]))
    assert rows[0][-5:] == [
        "rail_escape_stopped",
        "rail_escape_travel_m",
        "rail_escape_v_des_m_s",
        "rail_escape_qdot_cmd_m_s",
        "force_barrier_contact_active",
    ]
