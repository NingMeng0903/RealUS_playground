"""Native mesh broadphase must retain the exact CBF pairs at recorded poses."""
from pathlib import Path
import uuid

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig, CollisionModel
from rm75_control.control.joint_admittance_8dof.loop import JointIkConfig, JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.wbc_rt.client import find_wbc_rt_binary


def test_native_mesh_keeps_all_exact_active_pairs():
    binary = find_wbc_rt_binary()
    if binary is None:
        pytest.skip("native binary not built")
    kin = RobotKinematics()
    mesh = kin.urdf_path.parent / "RM75-6F-8dof.collision.urdf"
    collision = CollisionConfig(enabled=True, collision_urdf=mesh, d_safe=.01, d_activate=.04)
    cfg = JointIkConfig(backend="native", native_bin=str(binary),
                        native_shm_prefix="mesh_contract_" + uuid.uuid4().hex,
                        collision=collision)
    exact = CollisionModel(kin.model, collision_urdf=mesh)
    controller = JointIkController(kin, cfg)
    poses = (
        [0.182348, -.582329, -.870867, 1.205656, 1.627659, .261363, 1.133958, 2.31942],
        [.239506, -1.433736, -1.462377, .940418, 1.838251, 1.544128, 1.30055, 1.252588],
        [.361917, -2.107101, -1.474908, .823027, .994349, 1.939899, 1.548945, 1.5461],
    )
    try:
        for pose in poses:
            q = np.array(pose)
            exact.update(q)
            expected = len(exact.active_pairs(.05))
            assert 0 < expected < collision.max_pairs
            controller.reset(q)
            step = controller.update(np.zeros(6), q_meas=q, rail_exec_vel_m_s=0., auto_commit=False)
            assert step.n_cbf_active == expected
            assert not step.solver_fault_latched, step.fallback_reason
            assert step.qpik_hard_residual_max <= 1.e-5
            assert step.qp_collision_ms > 0.
            assert step.qp_collision_ms <= step.qp_assembly_ms <= step.qpik_total_ms
    finally:
        controller.close()


def test_exhausted_soft_budget_still_solves_and_certifies_qp1():
    binary = find_wbc_rt_binary()
    if binary is None:
        pytest.skip("native binary not built")
    kin = RobotKinematics()
    cfg = JointIkConfig(backend="native", native_bin=str(binary),
                        native_shm_prefix="budget_contract_" + uuid.uuid4().hex)
    # Collision assembly necessarily consumes this budget. It must skip the
    # optional posture QP, while the mandatory command still gets certified.
    cfg.qp.max_solve_ms = .000001
    controller = JointIkController(kin, cfg)
    q = np.array([.239506, -1.433736, -1.462377, .940418,
                  1.838251, 1.544128, 1.30055, 1.252588])
    try:
        controller.reset(q)
        step = controller.update(np.array([0., .1, 0., 0., 0., 0.]),
                                 q_meas=q, rail_exec_vel_m_s=0., auto_commit=False)
        assert step.qp_assembly_ms > cfg.qp.max_solve_ms
        assert step.qp2_status == "not_run"
        assert step.qp_solver_iterations > 1
        assert not step.solver_fault_latched, step.fallback_reason
        assert step.qpik_hard_residual_max <= 1.e-5
    finally:
        controller.close()
