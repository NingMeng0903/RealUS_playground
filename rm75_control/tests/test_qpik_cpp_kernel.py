"""Equivalence: C++ kernel vs the Python / NumPy reference."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.ik_types import project_onto_task_nullspace
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics, full_q_from_arm
from rm75_control.control.joint_admittance_8dof.solver import cpp_kernel
from rm75_control.control.joint_admittance_8dof.solver.cbf_constraints import CbfRows
from rm75_control.control.joint_admittance_8dof.solver.constraint_mgr import (
    build_wbc_inequalities,
)
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import (
    MAX_PREF_ROWS,
    N_PREF_SLACK,
    N_TASK_SLACK,
    QpConfig,
    QpIkController,
)
from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits

Q_SAFE = full_q_from_arm(
    np.deg2rad([5.0, -30.0, 10.0, 60.0, -5.0, 45.0, 0.0]), 0.40
)


def test_cpp_kernel_module_import_is_optional() -> None:
    assert hasattr(cpp_kernel, "available")
    assert isinstance(cpp_kernel.available(), bool)


def test_kinematics_snapshot_matches_python() -> None:
    kin = RobotKinematics()
    J_py = kin.jacobian(Q_SAFE)
    sig_py = kin.singular_values(J_py)
    M_py = kin.mass_matrix(Q_SAFE)
    J, sig, M = cpp_kernel.kinematics_snapshot(kin, Q_SAFE, need_mass=True)
    np.testing.assert_allclose(J, J_py, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(sig, sig_py, atol=1e-8, rtol=1e-8)
    if M is not None:
        np.testing.assert_allclose(M, M_py, atol=1e-8, rtol=1e-8)


def test_inequality_assembly_matches_python() -> None:
    nv = 8
    lo = -0.2 * np.ones(nv)
    hi = 0.2 * np.ones(nv)
    cbf = CbfRows(
        jacobian=np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        lower=np.array([-0.05]),
        slot_index=np.array([0]),
    )
    pref_j = np.zeros((2, nv))
    pref_j[0, 3] = 1.0
    pref_j[1, 4] = -1.0
    pref_s = np.array([2, 3])
    pref_l = np.array([-0.1, -0.2])
    C_py, l_py, u_py = build_wbc_inequalities(
        nv,
        N_TASK_SLACK,
        lo,
        hi,
        cbf,
        4,
        n_pref_slack=N_PREF_SLACK,
        max_pref_rows=MAX_PREF_ROWS,
        pref_jacobian=pref_j,
        pref_slack_col=pref_s,
        pref_lower=pref_l,
    )
    C, l, u = cpp_kernel.build_wbc_inequalities(
        nv,
        N_TASK_SLACK,
        lo,
        hi,
        cbf,
        4,
        n_pref_slack=N_PREF_SLACK,
        max_pref_rows=MAX_PREF_ROWS,
        pref_jacobian=pref_j,
        pref_slack_col=pref_s,
        pref_lower=pref_l,
    )
    np.testing.assert_allclose(C, C_py, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(l, l_py, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(u, u_py, atol=0.0, rtol=0.0)


def test_nullspace_projection_matches_python() -> None:
    kin = RobotKinematics()
    J = kin.jacobian(Q_SAFE)
    qdot0 = np.linspace(-0.05, 0.04, kin.nv)
    damping = 1.0e-3
    py = project_onto_task_nullspace(J, qdot0, damping=damping)
    cxx = cpp_kernel.project_nullspace(J, qdot0, damping=damping, M=None, use_dyn=False)
    np.testing.assert_allclose(cxx, py, atol=1e-10, rtol=1e-9)


def test_dyn_nullspace_matches_python() -> None:
    kin = RobotKinematics()
    J = kin.jacobian(Q_SAFE)
    M = kin.mass_matrix(Q_SAFE)
    qdot0 = np.linspace(-0.05, 0.04, kin.nv)
    py = project_onto_task_nullspace(
        J, qdot0, damping=1.0e-3, M=M, use_dyn=True
    )
    cxx = cpp_kernel.project_nullspace(
        J, qdot0, damping=1.0e-3, M=M, use_dyn=True
    )
    np.testing.assert_allclose(cxx, py, atol=1e-9, rtol=1e-8)


def test_setup_qp1_matches_python() -> None:
    nv, n_task, n_pref = 8, 6, 9
    w = np.diag(np.array([100.0, 100.0, 100.0, 50.0, 50.0, 50.0]))
    J = np.arange(n_task * nv, dtype=float).reshape(n_task, nv) * 0.01
    H_py, g_py, A_py = cpp_kernel.setup_qp1(
        nv, n_task, n_pref, w, J, use_native=False
    )
    H, g, A = cpp_kernel.setup_qp1(nv, n_task, n_pref, w, J, use_native=True)
    np.testing.assert_allclose(H, H_py, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(g, g_py, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(A, A_py, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not cpp_kernel.available(), reason="C++ kernel not built")
def test_cpp_and_python_qdot_match_on_one_tick() -> None:
    kin = RobotKinematics()
    limits = SafetyLimits.from_kinematics(
        kin, v_scale=0.8, a_max=np.concatenate(([0.6], np.full(7, 3.0)))
    )
    twist = np.array([0.0, 0.03, 0.0, 0.0, 0.0, 0.0])
    py = QpIkController(
        kin,
        limits,
        QpConfig(
            backend="proxqp",
            collision=CollisionConfig(enabled=False),
            use_cpp_kernel=False,
        ),
    )
    cxx = QpIkController(
        kin,
        limits,
        QpConfig(
            backend="proxqp",
            collision=CollisionConfig(enabled=False),
            use_cpp_kernel=True,
        ),
    )
    py.reset()
    cxx.reset()
    r_py = py.step(Q_SAFE, twist, 0.007, q_meas=Q_SAFE)
    r_cxx = cxx.step(Q_SAFE, twist, 0.007, q_meas=Q_SAFE)
    np.testing.assert_allclose(r_cxx.qdot, r_py.qdot, atol=5e-6, rtol=5e-5)
