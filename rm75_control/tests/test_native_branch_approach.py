"""Regression for the November 26 branch damper / jerk-box conflict."""
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
import yaml

from peirastic.realman8dof.daemon import ControllerService
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import (
    JointIkController, _guard_qpik_step_before_send, pad_inner_should_hold_without_send,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.wbc_rt.client import find_wbc_rt_binary

DT = .005
# Frozen from the first J3 fault, in controller order (rail, J1, ..., J7).
Q = np.array([.037804180726670494, -.9995648059477483, -.9630536479816966,
              .38421580368422853, 2.1672252355136292, .8745286739574006,
              1.1990376041441144, .8591198627674869])


@pytest.fixture
def controller():
    binary = find_wbc_rt_binary()
    if binary is None:
        pytest.skip("native worker has not been built")
    repo = Path(__file__).resolve().parents[2]
    cfg = build_joint_ik_config(yaml.safe_load((repo / "peirastic/configs/controller.yaml").read_text()))
    cfg.native_bin = str(binary)
    cfg.native_shm_prefix = "j3_approach_" + uuid4().hex
    kin = RobotKinematics()
    kin.apply_link7_to_tcp_offset([0., -.015229, .12135, .018, .871, -1.548])
    ctrl = JointIkController(kin, cfg)
    try:
        yield ctrl
    finally:
        ctrl.close()


def test_recorded_j3_history_preserves_both_hard_bounds(controller):
    prev, prev2 = np.zeros(8), np.zeros(8)
    prev[3] = -.19454973824311117
    prev2[3] = -.19455001233595698
    prior = Q - DT * prev
    controller.reset(prior)
    controller.enable()
    controller.update(np.zeros(6), DT, q_meas=prior, auto_commit=False)
    controller.abort_publication()
    controller.begin_hybrid_episode(prior, prev2)
    controller.set_direct_joint_ptp(True)
    seeded = controller.update(np.zeros(6), DT, q_meas=prior, qdot_ff=prev)
    assert not seeded.task_paused
    controller.set_direct_joint_ptp(False)

    measured = Q.copy()
    measured[3] = .38798668472885617
    step = controller.update(np.zeros(6), DT, q_meas=measured, rail_exec_vel_m_s=0.)
    assert step.qp1_status == "p0_conflict" and step.qp2_status == "not_run"
    np.testing.assert_allclose([step.box_lo[3], step.box_hi[3]],
                               [-.18350490776708994, -.18704946415026535], atol=1.e-11, rtol=0)
    np.testing.assert_allclose(step.q_send[1:], Q[1:], atol=1.e-12, rtol=0)
    assert step.n_cbf_active == 0 and step.qp_solver_iterations == 0
    events = []
    assert not pad_inner_should_hold_without_send(step)
    send, reason = _guard_qpik_step_before_send(step, events.append)
    assert not send and events == [reason] and ControllerService._is_latched_fault(reason)


@pytest.mark.parametrize("delay", [0, 11])
def test_demand_toward_j3_wall_with_sample_hold(controller, delay):
    q = Q.copy()
    q[0], q[3] = .28, .58
    controller.reset(q)
    controller.enable()
    controller.begin_hybrid_episode(q, np.zeros(8))
    history = deque([q.copy()] * (delay + 1), maxlen=delay + 1)
    rail_speed = 0.
    closest = q[3]
    for tick in range(350):
        if tick % 3 == 0:
            q = history[0]
        R = controller.kin.fk_placement(q).rotation.copy()
        base = -.45 * controller.kin.jacobian(q)[:, 3]
        twist = np.r_[R.T @ base[:3], R.T @ base[3:]]
        step = controller.update(twist, DT, q_meas=q, rail_exec_vel_m_s=rail_speed,
                                 rail_refresh_dt_s=.020)
        assert not step.task_paused and step.qp1_status in ("solved", "max_iter")
        assert step.qpik_hard_residual_max < 2.e-5
        assert step.q_send[3] >= .35 - 1.e-7
        closest = min(closest, step.q_send[3])
        history.append(step.q_send.copy())
        rail_speed = step.qdot[0]
    assert closest < .39


def test_direct_position_rejection_and_transient_pad_miss_are_distinct():
    step = SimpleNamespace(controller_mode="direct_joint_ptp", qp1_status="failed",
                           task_pause_reason="qsend_position", fallback_reason="qsend_position")
    events = []
    send, reason = _guard_qpik_step_before_send(step, events.append)
    assert not send and events == [reason] and ControllerService._is_latched_fault(reason)
    assert pad_inner_should_hold_without_send(SimpleNamespace(fallback_reason="publication_infeasible"))
