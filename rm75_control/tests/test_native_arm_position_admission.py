"""Native regression tests for hard arm-position admission.

These tests use the isolated position-admission binary when it is available.
They intentionally exercise the C++ worker through ``JointIkController``;
there are no source-string-only assertions here.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.loop import JointIkConfig, JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import QpConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NATIVE = ROOT / "native" / "wbc_rt" / "build_position_admission_20261110" / "wbc_rt"
NATIVE = Path(os.environ.get("WBC_POSITION_ADMISSION_BIN", str(DEFAULT_NATIVE)))

# Frozen from audit/physical_box_failure_checks.json.  At this measured q,
# joint 6 is 3.103944 rad while the physical upper limit is 3.106 rad.  The
# positive history makes the one-tick acceleration lower bound 1.029998 rad/s
# while the margin-backed position upper bound is -0.636055 rad/s.
Q_NEAR_J6_UPPER = np.array(
    [
        0.27560823257411077,
        -2.981836854918179,
        -1.3533719544152885,
        0.5598250925717257,
        0.4247467227749,
        3.103944287751988,
        1.3956579827221434,
        0.9462721418428935,
    ],
    dtype=float,
)
QDOT_FAILURE_HISTORY = np.array(
    [
        -0.005120917080836396,
        -0.16926559057113266,
        -0.11632547732589593,
        -1.075978412448264,
        -0.007708749799395509,
        1.0449986603438433,
        0.007682663840791193,
        0.2797930405010751,
    ],
    dtype=float,
)


def _native_controller() -> JointIkController:
    if not NATIVE.is_file():
        pytest.skip(f"isolated native binary is not available: {NATIVE}")
    qp = QpConfig(backend="proxqp", collision=CollisionConfig(enabled=False))
    cfg = JointIkConfig(
        control_frame="base",
        qp=qp,
        collision=CollisionConfig(enabled=False),
        backend="native",
        native_bin=str(NATIVE),
        native_shm_prefix=f"position_admission_{os.getpid()}_{uuid.uuid4().hex}",
    )
    return JointIkController(RobotKinematics(), cfg)


def _close(controller: JointIkController) -> None:
    try:
        controller.close()
    except Exception:
        native = getattr(controller, "_native", None)
        if native is not None:
            native.shutdown()


def test_native_arm_position_conflict_fails_closed_without_qsend() -> None:
    """The frozen position/acceleration conflict is rejected by native C++."""

    controller = _native_controller()
    try:
        controller.reset(Q_NEAR_J6_UPPER)
        controller.enable()
        controller.begin_hybrid_episode(Q_NEAR_J6_UPPER, QDOT_FAILURE_HISTORY)
        step = controller.update(
            np.zeros(6),
            0.005,
            q_meas=Q_NEAR_J6_UPPER,
            rail_exec_vel_m_s=float(QDOT_FAILURE_HISTORY[0]),
            rail_refresh_dt_s=0.020,
        )
    finally:
        _close(controller)

    # This is the actual native worker's P0 conflict, not a Python/source
    # pre-check.  The command remains the previous in-limit command.
    assert step.qp1_status == "p0_conflict"
    assert step.task_paused
    assert step.box_infeasible
    np.testing.assert_allclose(step.q_send, Q_NEAR_J6_UPPER, atol=1.0e-12, rtol=0.0)
    assert float(step.q_send[5]) <= 3.106 + 1.0e-12
    np.testing.assert_allclose(step.qdot, np.zeros(8), atol=1.0e-12, rtol=0.0)


def test_native_final_qsend_guard_rejects_direct_arm_overshoot() -> None:
    """A direct integrated arm command cannot bypass final qsend admission."""

    controller = _native_controller()
    qdot_ff = np.zeros(8)
    qdot_ff[5] = 1.0  # q5 + 0.005 would be 3.108944 > URDF upper 3.106.
    try:
        controller.reset(Q_NEAR_J6_UPPER)
        controller.enable()
        controller.set_direct_joint_ptp(True)
        controller.begin_hybrid_episode(Q_NEAR_J6_UPPER, np.zeros(8))
        step = controller.update(
            np.zeros(6),
            0.005,
            q_meas=Q_NEAR_J6_UPPER,
            qdot_ff=qdot_ff,
        )
    finally:
        _close(controller)

    assert step.controller_mode == "direct_joint_ptp"
    assert step.task_paused
    assert step.qp1_status == "failed"
    np.testing.assert_allclose(step.q_send, Q_NEAR_J6_UPPER, atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(step.qdot, np.zeros(8), atol=1.0e-12, rtol=0.0)


def test_native_direct_in_limit_arm_command_still_publishes() -> None:
    """The final guard does not reject an in-limit direct command."""

    controller = _native_controller()
    qdot_ff = np.zeros(8)
    qdot_ff[5] = 0.05  # q5 + 0.00025 remains below 3.106.
    try:
        controller.reset(Q_NEAR_J6_UPPER)
        controller.enable()
        controller.set_direct_joint_ptp(True)
        controller.begin_hybrid_episode(Q_NEAR_J6_UPPER, np.zeros(8))
        step = controller.update(
            np.zeros(6),
            0.005,
            q_meas=Q_NEAR_J6_UPPER,
            qdot_ff=qdot_ff,
        )
    finally:
        _close(controller)

    assert not step.task_paused
    assert step.qp1_status == "solved"
    np.testing.assert_allclose(
        step.q_send,
        Q_NEAR_J6_UPPER + 0.005 * qdot_ff,
        atol=1.0e-12,
        rtol=0.0,
    )
    assert float(step.q_send[5]) < 3.106
