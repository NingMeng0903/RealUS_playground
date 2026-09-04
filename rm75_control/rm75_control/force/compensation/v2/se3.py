"""SO(3) geodesic interpolation and SE(3) integration for Payload ID V2."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rsc


def quintic_s(tau: float) -> float:
    t = float(np.clip(tau, 0.0, 1.0))
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def quintic_sdot(tau: float) -> float:
    t = float(np.clip(tau, 0.0, 1.0))
    return 30.0 * t**2 - 60.0 * t**3 + 30.0 * t**4


def geodesic_R(R0: np.ndarray, R1: np.ndarray, tau: float) -> np.ndarray:
    r0 = Rsc.from_matrix(np.asarray(R0, dtype=float))
    r1 = Rsc.from_matrix(np.asarray(R1, dtype=float))
    delta = r0.inv() * r1
    return (r0 * Rsc.from_rotvec(quintic_s(tau) * delta.as_rotvec())).as_matrix()


def so3_log(R: np.ndarray) -> np.ndarray:
    return Rsc.from_matrix(np.asarray(R, dtype=float)).as_rotvec()


def integrate_se3(
    p0: np.ndarray,
    R0: np.ndarray,
    v_body: np.ndarray,
    w_body: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate body-frame twist with ``R_{k+1} = R_k exp([ω]_× dt)``."""
    R = np.asarray(R0, dtype=float).reshape(3, 3)
    p = np.asarray(p0, dtype=float).reshape(3)
    v = np.asarray(v_body, dtype=float).reshape(3)
    w = np.asarray(w_body, dtype=float).reshape(3)
    p_next = p + (R @ v) * float(dt)
    R_next = (Rsc.from_matrix(R) * Rsc.from_rotvec(w * float(dt))).as_matrix()
    return p_next, R_next


def se3_closure_error(
    p0: np.ndarray,
    R0: np.ndarray,
    p1: np.ndarray,
    R1: np.ndarray,
) -> tuple[float, float]:
    dp = float(np.linalg.norm(np.asarray(p1) - np.asarray(p0)))
    dR = float(np.linalg.norm(so3_log(np.asarray(R0).T @ np.asarray(R1))))
    return dp, dR
