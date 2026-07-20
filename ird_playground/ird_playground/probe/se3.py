"""SE(3) helpers: ΔT → natural (p,u) 5-DoF features, Exp map."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def mat4_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def delta_T_tcp_inv_base(T_base_tcp: np.ndarray) -> np.ndarray:
    """ΔT = T_tcp^{-1} T_base = (T_base_tcp)^{-1} when T_base = I in arm-base frame."""
    return invert_T(T_base_tcp)


def rot6d_from_R(R: np.ndarray) -> np.ndarray:
    """Zhou et al. continuous 6D rotation: first two columns of R."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    return np.concatenate([R[:, 0], R[:, 1]], axis=0)


def features_from_delta_T(delta_T: np.ndarray) -> np.ndarray:
    """(6,) = natural 5-DoF [p_base,tcp, u_base] recovered from ΔT.

    ΔT = T_tcp^{-1} T_base. With T_base=I:
      R_base,tcp = R_Δᵀ
      p_base,tcp = −R_Δᵀ t_Δ
      u_base = R_base,tcp @ e_z = R_Δᵀ[:,2] = R_Δ[2,:]ᵀ wait: (R_Δᵀ)[:,2] = R_Δ[2,:].T
    """
    T = np.asarray(delta_T, dtype=np.float64).reshape(4, 4)
    R_delta = T[:3, :3]
    t_delta = T[:3, 3]
    R_base_tcp = R_delta.T
    p = -(R_base_tcp @ t_delta)
    u = R_base_tcp[:, 2].copy()
    u = u / (np.linalg.norm(u) + 1e-12)
    return np.concatenate([p, u], axis=0).astype(np.float64)


def batch_features_from_delta_T(delta_Ts: np.ndarray) -> np.ndarray:
    """(N,6) from (N,4,4)."""
    Ts = np.asarray(delta_Ts, dtype=np.float64)
    if Ts.ndim == 2:
        return features_from_delta_T(Ts)[None, :]
    out = np.empty((Ts.shape[0], 6), dtype=np.float64)
    for i, T in enumerate(Ts):
        out[i] = features_from_delta_T(T)
    return out


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """ξ = [δp(3), δω(3)] → SE(3) via scipy Rotation (axis-angle)."""
    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    dp, dw = xi[:3], xi[3:]
    R = Rotation.from_rotvec(dw).as_matrix()
    return mat4_from_Rt(R, dp)


def se3_mul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.asarray(A, dtype=np.float64) @ np.asarray(B, dtype=np.float64)


def complete_frame_from_tool_axis(tool_axis: np.ndarray) -> np.ndarray:
    """Build a rotation whose +Z is ``tool_axis`` (Zacharias tool axis = TCP +Z)."""
    z = np.asarray(tool_axis, dtype=np.float64).reshape(3)
    z = z / (np.linalg.norm(z) + 1e-12)
    a = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(a, z)
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)
