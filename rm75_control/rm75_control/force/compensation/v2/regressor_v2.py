"""V2 regressor: explicit dynamic / inertia flags. Does not wrap legacy OLS."""

from __future__ import annotations

import numpy as np

from rm75_control.force.compensation.regressor import inertia_op, skew


def static_design(g_L: np.ndarray, *, include_drift: bool = False, t_s: float = 0.0) -> np.ndarray:
    """One 6×10 (or 6×16) static row-block for ``θ = [m, h, b0, (ḃ)]``."""
    g = np.asarray(g_L, dtype=float).reshape(3)
    cols = 16 if include_drift else 10
    A = np.zeros((6, cols), dtype=float)
    A[0:3, 0] = -g
    A[3:6, 1:4] = skew(g)
    A[0:3, 4:7] = np.eye(3)
    A[3:6, 7:10] = np.eye(3)
    if include_drift:
        A[0:3, 10:13] = float(t_s) * np.eye(3)
        A[3:6, 13:16] = float(t_s) * np.eye(3)
    return A


def payload_wrench_mhb(
    *,
    mass_kg: float,
    h_L: np.ndarray,
    a_L: np.ndarray,
    g_L: np.ndarray,
    omega_L: np.ndarray,
    alpha_L: np.ndarray,
    bias: np.ndarray | None = None,
    inertia_voigt: np.ndarray | None = None,
) -> np.ndarray:
    """Newton–Euler payload wrench in L. ``a_L`` is classical, gravity separate."""
    m = float(mass_kg)
    h = np.asarray(h_L, dtype=float).reshape(3)
    a = np.asarray(a_L, dtype=float).reshape(3)
    g = np.asarray(g_L, dtype=float).reshape(3)
    w = np.asarray(omega_L, dtype=float).reshape(3)
    al = np.asarray(alpha_L, dtype=float).reshape(3)
    aeq = a - g
    f = m * aeq + np.cross(al, h) + np.cross(w, np.cross(w, h))
    tau = -np.cross(aeq, h)
    if inertia_voigt is not None:
        Icol = np.asarray(inertia_voigt, dtype=float).reshape(6)
        tau = tau + (inertia_op(al) + skew(w) @ inertia_op(w)) @ Icol
    out = np.concatenate([f, tau])
    if bias is not None:
        out = out + np.asarray(bias, dtype=float).reshape(6)
    return out


def regressor_row_v2(
    a_L: np.ndarray,
    g_L: np.ndarray,
    omega_L: np.ndarray,
    alpha_L: np.ndarray,
    *,
    use_dynamic_kinematics: bool,
    use_rotational_inertia: bool,
) -> np.ndarray:
    if use_rotational_inertia and not use_dynamic_kinematics:
        raise ValueError("use_rotational_inertia requires use_dynamic_kinematics")
    a = np.asarray(a_L, dtype=float).reshape(3)
    g = np.asarray(g_L, dtype=float).reshape(3)
    w = np.asarray(omega_L, dtype=float).reshape(3)
    al = np.asarray(alpha_L, dtype=float).reshape(3)
    if not use_dynamic_kinematics:
        a = np.zeros(3)
        w = np.zeros(3)
        al = np.zeros(3)
    aeq = a - g
    sw, sa = skew(w), skew(al)
    W = np.zeros((6, 16), dtype=float)
    W[0:3, 0] = aeq
    W[0:3, 1:4] = sa + sw @ sw
    W[3:6, 1:4] = -skew(aeq)
    if use_rotational_inertia:
        W[3:6, 4:10] = inertia_op(al) + sw @ inertia_op(w)
    W[:, 10:16] = np.eye(6)
    return W


def unmodeled_inertia_torque_bound(
    mass_kg: float,
    r_max_m: float,
    alpha_L: np.ndarray,
    omega_L: np.ndarray,
) -> float:
    return float(mass_kg) * float(r_max_m) ** 2 * (
        float(np.linalg.norm(alpha_L)) + float(np.linalg.norm(omega_L)) ** 2
    )


def inertia_triangle_ok(Ixx: float, Iyy: float, Izz: float, *, eps: float = 0.0) -> bool:
    return (
        Ixx + Iyy >= Izz - eps
        and Ixx + Izz >= Iyy - eps
        and Iyy + Izz >= Ixx - eps
        and Ixx > 0.0
        and Iyy > 0.0
        and Izz > 0.0
    )
