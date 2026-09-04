"""Canonical frames and wrench maps for Payload ID V2.

L = link_7 (canonical payload origin)
S = force_sensor measurement frame / origin
T = gripper2 / active TCP (command and publish)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rsc

WRENCH_SEMANTICS = "environment_on_tool"
CANONICAL_PAYLOAD_FRAME = "link_7"
RAW_WRENCH_FRAME = "force_sensor"
COMMAND_TCP_FRAME = "gripper2"
PUBLISH_WRENCH_FRAME = "gripper2"


class WrenchSemantics:
    ENVIRONMENT_ON_TOOL = WRENCH_SEMANTICS


def _as3(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(3)


def _as33(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(3, 3)


def _as6(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(6)


@dataclass(frozen=True)
class FrameContract:
    """Single physical origin for payload parameters: link_7.

    ``r_LS_L := p_S^L - p_L^L`` — from link_7 origin to sensor origin, in L.
    ``r_LT_L := p_T^L - p_L^L`` — from link_7 origin to TCP origin, in L.
    """

    canonical_payload_frame: str = CANONICAL_PAYLOAD_FRAME
    raw_wrench_frame: str = RAW_WRENCH_FRAME
    command_tcp_frame: str = COMMAND_TCP_FRAME
    publish_wrench_frame: str = PUBLISH_WRENCH_FRAME
    wrench_semantics: str = WRENCH_SEMANTICS
    gravity_base_m_s2: tuple[float, float, float] = (0.0, 0.0, -9.80665)
    force_sign: tuple[int, ...] = (-1, -1, -1, -1, -1, -1)
    moment_sign: tuple[int, ...] = (-1, -1, -1)
    R_LS: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    r_LS_L: tuple[float, float, float] = (0.0, 0.0, 0.0)
    R_LT: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    r_LT_L: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def R_LS_mat(self) -> np.ndarray:
        return np.asarray(self.R_LS, dtype=float).reshape(3, 3)

    def R_LT_mat(self) -> np.ndarray:
        return np.asarray(self.R_LT, dtype=float).reshape(3, 3)

    def r_LS_L_vec(self) -> np.ndarray:
        return _as3(self.r_LS_L)

    def r_LT_L_vec(self) -> np.ndarray:
        return _as3(self.r_LT_L)

    def gravity_base(self) -> np.ndarray:
        return _as3(self.gravity_base_m_s2)

    def assert_identity_sensor_if_colocated(self) -> None:
        if np.allclose(self.r_LS_L_vec(), 0.0) and np.allclose(self.R_LS_mat(), np.eye(3)):
            return
        # Non-identity T_LS is allowed; colocated case must be explicit I.
        return

    def assert_colocated_sensor(self) -> None:
        if not np.allclose(self.R_LS_mat(), np.eye(3), atol=1e-12):
            raise AssertionError("T_LS rotation is not I while sensor was declared colocated")
        if not np.allclose(self.r_LS_L_vec(), 0.0, atol=1e-12):
            raise AssertionError("r_LS_L is not 0 while sensor was declared colocated")

    @classmethod
    def from_yaml(cls, path: Path) -> FrameContract:
        data = yaml.safe_load(Path(path).read_text()) or {}
        force_sign = tuple(int(x) for x in data.get("force_sign", [-1, -1, -1, -1, -1, -1]))
        if len(force_sign) == 6:
            moment_sign = tuple(int(x) for x in force_sign[3:6])
        else:
            moment_sign = tuple(int(x) for x in data.get("moment_sign", [1, 1, 1]))
        off = tuple(float(x) for x in data.get("sensor_offset_euler_xyz_rad", [0.0, 0.0, 0.0]))
        R_LS = Rsc.from_euler("xyz", off, degrees=False).as_matrix().ravel()
        origin = tuple(float(x) for x in data.get("sensor_origin_in_link7_m", [0.0, 0.0, 0.0]))
        g = tuple(float(x) for x in data.get("gravity_base", data.get("gravity_base_m_s2", [0.0, 0.0, -9.80665])))
        return cls(
            canonical_payload_frame=str(data.get("canonical_payload_frame", CANONICAL_PAYLOAD_FRAME)),
            raw_wrench_frame=str(data.get("raw_wrench_frame", RAW_WRENCH_FRAME)),
            command_tcp_frame=str(data.get("command_tcp_frame", COMMAND_TCP_FRAME)),
            publish_wrench_frame=str(data.get("publish_wrench_frame", PUBLISH_WRENCH_FRAME)),
            wrench_semantics=str(data.get("wrench_semantics", WRENCH_SEMANTICS)),
            gravity_base_m_s2=g,
            force_sign=force_sign if len(force_sign) == 6 else force_sign + moment_sign,
            moment_sign=moment_sign,
            R_LS=tuple(float(x) for x in R_LS),
            r_LS_L=origin,
        )


def apply_raw_sign(wrench_s: np.ndarray, contract: FrameContract) -> np.ndarray:
    w = _as6(wrench_s).copy()
    sign = np.asarray(contract.force_sign, dtype=float).reshape(-1)
    if sign.size < 6:
        sign = np.resize(sign, 6)
    w *= sign[:6]
    return w


def wrench_sensor_to_link7(wrench_s: np.ndarray, contract: FrameContract) -> np.ndarray:
    """``f_L = R_LS f_S``, ``τ_L = R_LS τ_S + r_LS_L × f_L``."""
    w = apply_raw_sign(wrench_s, contract)
    R = contract.R_LS_mat()
    f_L = R @ w[:3]
    tau_L = R @ w[3:6] + np.cross(contract.r_LS_L_vec(), f_L)
    return np.concatenate([f_L, tau_L])


def wrench_link7_to_sensor(wrench_L: np.ndarray, contract: FrameContract) -> np.ndarray:
    w = _as6(wrench_L)
    R = contract.R_LS_mat()
    f_L, tau_L = w[:3], w[3:6]
    f_S = R.T @ f_L
    tau_S = R.T @ (tau_L - np.cross(contract.r_LS_L_vec(), f_L))
    return np.concatenate([f_S, tau_S])


def wrench_link7_to_tcp(
    wrench_L: np.ndarray,
    *,
    R_LT: np.ndarray,
    r_LT_L: np.ndarray,
) -> np.ndarray:
    """``τ_T^L = τ_L − r_LT_L × f_L``, then rotate into TCP axes."""
    w = _as6(wrench_L)
    R = _as33(R_LT)
    r = _as3(r_LT_L)
    f_L, tau_L = w[:3], w[3:6]
    tau_T_L = tau_L - np.cross(r, f_L)
    return np.concatenate([R.T @ f_L, R.T @ tau_T_L])


def wrench_tcp_to_link7(
    wrench_T: np.ndarray,
    *,
    R_LT: np.ndarray,
    r_LT_L: np.ndarray,
) -> np.ndarray:
    w = _as6(wrench_T)
    R = _as33(R_LT)
    r = _as3(r_LT_L)
    f_L = R @ w[:3]
    tau_T_L = R @ w[3:6]
    tau_L = tau_T_L + np.cross(r, f_L)
    return np.concatenate([f_L, tau_L])


def twist_link7_to_tcp(
    twist_L: np.ndarray,
    *,
    R_LT: np.ndarray,
    r_LT_L: np.ndarray,
) -> np.ndarray:
    tw = _as6(twist_L)
    v_L, w_L = tw[:3], tw[3:6]
    v_T_L = v_L + np.cross(w_L, _as3(r_LT_L))
    R = _as33(R_LT)
    return np.concatenate([R.T @ v_T_L, R.T @ w_L])


def twist_about_link7_to_tcp(omega_L: np.ndarray, *, R_LT: np.ndarray, r_LT_L: np.ndarray) -> np.ndarray:
    """Scheme A: ``v_L = 0``, ``ω_L = omega_d`` → gripper2 TCP command."""
    return twist_link7_to_tcp(
        np.concatenate([np.zeros(3), _as3(omega_L)]),
        R_LT=R_LT,
        r_LT_L=r_LT_L,
    )


def tcp_pose_from_link7_pose(
    pose_L: np.ndarray,
    *,
    R_LT: np.ndarray,
    r_LT_L: np.ndarray,
    euler_order: str = "xyz",
    hold_point: str = "tcp",
    p_tcp_hold: np.ndarray | None = None,
) -> np.ndarray:
    """Project a planned link_7 / armtip pose to the live TCP.

    ``R_LT`` maps TCP-frame vectors into link_7: ``v_L = R_LT @ v_T``.
    Attitude is always taken from ``pose_L``. Position: ``hold_point=tcp``
    pins the taught TCP (this cell); ``link_7`` pins the flange and lets
    the current tool swing. Changing the live tool only changes ``R_LT``,
    ``r_LT_L``.
    """
    R_L = Rsc.from_euler(str(euler_order), pose_L[3:6], degrees=False).as_matrix()
    R = _as33(R_LT)
    hold = str(hold_point).strip().lower()
    if hold in {"link_7", "link7", "armtip", "flange"}:
        p_T = _as3(pose_L[:3]) + R_L @ _as3(r_LT_L)
    else:
        if p_tcp_hold is None:
            p_T = _as3(pose_L[:3]) + R_L @ _as3(r_LT_L)
        else:
            p_T = _as3(p_tcp_hold)
    out = np.zeros(6, dtype=float)
    out[:3] = p_T
    out[3:6] = Rsc.from_matrix(R_L @ R).as_euler(str(euler_order), degrees=False)
    return out


def gravity_in_link7(R_BL: np.ndarray, gravity_base: np.ndarray) -> np.ndarray:
    return _as33(R_BL).T @ _as3(gravity_base)


def gravity_force_link7(mass_kg: float, g_L: np.ndarray, bias_f: np.ndarray | None = None) -> np.ndarray:
    """``f_pred = −m g_L + b_F`` (environment-on-tool / weight on sensor)."""
    f = -float(mass_kg) * _as3(g_L)
    if bias_f is not None:
        f = f + _as3(bias_f)
    return f
