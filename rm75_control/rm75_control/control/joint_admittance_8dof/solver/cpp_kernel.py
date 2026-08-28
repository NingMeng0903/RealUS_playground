"""Optional C++ QPIK kernel (inequality assembly, SVD, nullspace, QP setup).

FK / Jacobian / CRBA stay on Pinocchio's Python bindings: a second C++
``pinocchio::Model`` in-process next to ``pinocchio_pywrap`` segfaulted.
ProxQP stay on the Python ``proxsuite`` bindings for the same ABI reason.

Eigen is used only inside the extension (never via ``pybind11/eigen.h``).
"""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance_8dof.solver.constraint_mgr import (
    build_wbc_inequalities as _py_build_wbc,
    collapse_interval as _py_collapse_interval,
)
from rm75_control.control.joint_admittance_8dof.ik_types import (
    project_onto_task_nullspace as _py_project,
)

_NATIVE = None
try:
    from rm75_control.control.joint_admittance_8dof.solver import _qpik_kernel as _NATIVE
except ImportError:
    try:
        import _qpik_kernel as _NATIVE  # type: ignore
    except ImportError:
        _NATIVE = None


def available() -> bool:
    """True when the native extension imported successfully."""
    return _NATIVE is not None


def singular_values(J: np.ndarray) -> np.ndarray:
    """Native SVD when the kernel is loaded; NumPy otherwise."""
    arr = np.ascontiguousarray(J, dtype=float)
    if available() and hasattr(_NATIVE, "singular_values"):
        return np.asarray(_NATIVE.singular_values(arr), dtype=float)
    return np.linalg.svd(arr, compute_uv=False)


def kinematics_snapshot(
    kin,
    q: np.ndarray,
    *,
    need_mass: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """FK/J/CRBA on Pinocchio Python; SVD on C++ when the kernel is loaded."""
    q = np.asarray(q, dtype=float).reshape(-1)
    J = np.asarray(kin.jacobian(q), dtype=float)
    sigma = singular_values(J)
    M = np.asarray(kin.mass_matrix(q), dtype=float) if need_mass else None
    return J, sigma, M


def build_wbc_inequalities(
    nv: int,
    n_task_slack: int,
    lo_box: np.ndarray,
    hi_box: np.ndarray,
    cbf,
    max_cbf_rows: int,
    *,
    n_pref_slack: int = 0,
    max_pref_rows: int = 0,
    pref_jacobian: np.ndarray | None = None,
    pref_slack_col: np.ndarray | None = None,
    pref_lower: np.ndarray | None = None,
):
    if not available() or not hasattr(_NATIVE, "build_wbc_inequalities"):
        return _py_build_wbc(
            nv,
            n_task_slack,
            lo_box,
            hi_box,
            cbf,
            max_cbf_rows,
            n_pref_slack=n_pref_slack,
            max_pref_rows=max_pref_rows,
            pref_jacobian=pref_jacobian,
            pref_slack_col=pref_slack_col,
            pref_lower=pref_lower,
        )
    jac = np.asarray(getattr(cbf, "jacobian", np.zeros((0, nv))), dtype=float)
    if jac.size == 0:
        jac = np.zeros((0, nv), dtype=float)
    lower = np.asarray(getattr(cbf, "lower", np.zeros(0)), dtype=float)
    slots = getattr(cbf, "slot_index", None)
    if slots is None:
        slot_i = np.zeros(0, dtype=np.int64)
    else:
        slot_i = np.asarray(slots, dtype=np.int64).reshape(-1)
    if pref_jacobian is None:
        pref_j = np.zeros((0, nv), dtype=float)
        pref_s = np.zeros(0, dtype=np.int64)
        pref_l = np.zeros(0, dtype=float)
    else:
        pref_j = np.asarray(pref_jacobian, dtype=float)
        pref_s = np.asarray(pref_slack_col, dtype=np.int64).reshape(-1)
        pref_l = np.asarray(pref_lower, dtype=float).reshape(-1)
    C, lo, hi = _NATIVE.build_wbc_inequalities(
        int(nv),
        int(n_task_slack),
        np.ascontiguousarray(lo_box, dtype=float),
        np.ascontiguousarray(hi_box, dtype=float),
        np.ascontiguousarray(jac, dtype=float),
        np.ascontiguousarray(lower, dtype=float),
        slot_i,
        int(max_cbf_rows),
        int(n_pref_slack),
        int(max_pref_rows),
        np.ascontiguousarray(pref_j, dtype=float),
        pref_s,
        np.ascontiguousarray(pref_l, dtype=float),
    )
    return np.asarray(C, dtype=float), np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)


def project_nullspace(J, qdot0, *, damping, M=None, use_dyn=False, m_floor=0.05):
    if not available() or not hasattr(_NATIVE, "project_nullspace"):
        return _py_project(
            J, qdot0, damping=damping, M=M, use_dyn=use_dyn, m_floor=m_floor
        )
    M_arr = (
        np.ascontiguousarray(M, dtype=float)
        if M is not None
        else np.zeros((0, 0), dtype=float)
    )
    return np.asarray(
        _NATIVE.project_nullspace(
            np.ascontiguousarray(J, dtype=float),
            np.ascontiguousarray(qdot0, dtype=float),
            float(damping),
            M_arr,
            bool(use_dyn),
            float(m_floor),
        ),
        dtype=float,
    )


def setup_qp1(
    nv: int, n_task: int, n_pref: int, w_task, J_task, *, use_native: bool | None = None
):
    w = np.ascontiguousarray(w_task, dtype=float)
    J = np.ascontiguousarray(J_task, dtype=float)
    if use_native is None:
        use_native = available()
    if use_native and available() and hasattr(_NATIVE, "setup_qp1"):
        H, g, A = _NATIVE.setup_qp1(int(nv), int(n_task), int(n_pref), w, J)
        return (
            np.asarray(H, dtype=float),
            np.asarray(g, dtype=float),
            np.asarray(A, dtype=float),
        )
    n_var = int(nv) + int(n_task) + int(n_pref)
    H = np.zeros((n_var, n_var), dtype=float)
    H[nv : nv + n_task, nv : nv + n_task] = w
    g = np.zeros(n_var, dtype=float)
    A = np.zeros((n_task, n_var), dtype=float)
    A[:, :nv] = J
    A[:, nv : nv + n_task] = -np.eye(n_task)
    return H, g, A


def setup_qp2_costs(
    nv: int,
    n_task: int,
    n_pref: int,
    h_reg,
    qdot_nom,
    slack_w,
    *,
    rail_w: float = 0.0,
    rail_vel: float = 0.0,
    smooth=None,
    qdot_prev=None,
    use_native: bool | None = None,
):
    hr = np.ascontiguousarray(h_reg, dtype=float).reshape(-1)
    qn = np.ascontiguousarray(qdot_nom, dtype=float).reshape(-1)
    sw = np.ascontiguousarray(slack_w, dtype=float).reshape(-1)
    sm = (
        np.ascontiguousarray(smooth, dtype=float).reshape(-1)
        if smooth is not None
        else np.zeros(nv, dtype=float)
    )
    qp = (
        np.ascontiguousarray(qdot_prev, dtype=float).reshape(-1)
        if qdot_prev is not None
        else np.zeros(nv, dtype=float)
    )
    if use_native is None:
        use_native = available()
    if use_native and available() and hasattr(_NATIVE, "setup_qp2_costs"):
        H, g = _NATIVE.setup_qp2_costs(
            int(nv),
            int(n_task),
            int(n_pref),
            hr,
            qn,
            sw,
            float(rail_w),
            float(rail_vel),
            sm,
            qp,
        )
        return np.asarray(H, dtype=float), np.asarray(g, dtype=float)
    n_var = int(nv) + int(n_task) + int(n_pref)
    H = np.zeros((n_var, n_var), dtype=float)
    H[:nv, :nv] = np.diag(hr)
    H[nv : nv + n_task, nv : nv + n_task] = np.eye(n_task) * 1.0e-10
    for k in range(n_pref):
        H[nv + n_task + k, nv + n_task + k] = sw[k]
    g = np.zeros(n_var, dtype=float)
    g[:nv] = -hr * qn
    if rail_w > 0.0:
        H[0, 0] += float(rail_w)
        g[0] -= float(rail_w) * float(rail_vel)
    if np.any(sm > 0.0):
        H[:nv, :nv] += np.diag(sm)
        g[:nv] -= sm * qp
    return H, g


def collapse_interval(lo, hi, qdot_prev=None, a_max=None, dt=None):
    """Native ``collapse_interval`` when the kernel is loaded; Python otherwise."""
    lo_a = np.ascontiguousarray(lo, dtype=float).reshape(-1)
    hi_a = np.ascontiguousarray(hi, dtype=float).reshape(-1)
    if not available() or not hasattr(_NATIVE, "collapse_interval"):
        return _py_collapse_interval(
            lo_a, hi_a, qdot_prev=qdot_prev, a_max=a_max, dt=dt
        )
    prev = (
        None
        if qdot_prev is None
        else np.ascontiguousarray(qdot_prev, dtype=float).reshape(-1)
    )
    amax = (
        None if a_max is None else np.ascontiguousarray(a_max, dtype=float).reshape(-1)
    )
    dt_v = float(dt) if dt is not None else 0.0
    lo_o, hi_o = _NATIVE.collapse_interval(lo_a, hi_a, prev, amax, dt_v)
    return np.asarray(lo_o, dtype=float), np.asarray(hi_o, dtype=float)


def solve_dense_qp(H, g, A, b, C, lo, hi, *, warm_x=None, max_iter=400, eps_abs=1e-6):
    """Native ProxQP is not linked (ABI).  Always return None → Python backend."""
    del H, g, A, b, C, lo, hi, warm_x, max_iter, eps_abs
    return None
