"""Closed-form SRS candidate graph and continuous whole-trajectory lift."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class SrsTrajectoryDpConfig:
    coarse_step_deg: float = 5.0
    refine_step_deg: float = 1.0
    refine_half_width_deg: float = 5.0
    psi_lower_deg: float = -180.0
    psi_upper_deg: float = 180.0
    maximum_joint_step_deg: float = 12.0
    maximum_rail_step_m: float = 0.04


@dataclass
class SrsTrajectoryDpResult:
    lift_valid: bool
    q_ref: np.ndarray
    psi_rad: np.ndarray
    branch: int
    candidate_seconds: float
    solve_seconds: float
    candidate_count_min: int
    candidate_count_max: int
    maximum_joint_step_rad: float
    maximum_joint_second_difference_rad: float
    failure: str = ""


def _pose6(T: np.ndarray, order: str) -> np.ndarray:
    pose = np.zeros(6, dtype=np.float64)
    pose[:3] = T[:3, 3]
    pose[3:] = Rotation.from_matrix(T[:3, :3]).as_euler(order, degrees=False)
    return pose


def _wrap_pi(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def _candidate_lattice(
    tcp_world: np.ndarray,
    rail_m: np.ndarray,
    psi_rows: np.ndarray,
    *,
    kin,
    branch: int,
    euler_order: str,
) -> tuple[np.ndarray, np.ndarray]:
    from rm75_control.control.joint_admittance_8dof.model import full_q_from_arm
    from rm75_control.kinematics.srs_ik import (
        flange_tcp_from_kin,
        shoulder_y_from_q_rail,
        srs_ik,
    )

    tcp = np.asarray(tcp_world, dtype=np.float64)
    rail = np.asarray(rail_m, dtype=np.float64)
    psi_rows = np.asarray(psi_rows, dtype=np.float64)
    w, k = psi_rows.shape
    candidates = np.full((w, k, 8), np.nan, dtype=np.float64)
    R_flange_tcp, t_flange_tcp = flange_tcp_from_kin(kin)
    for i in range(w):
        pose = _pose6(tcp[i], euler_order)
        shoulder_y = shoulder_y_from_q_rail(float(rail[i]))
        for j in range(k):
            q_arm = srs_ik(
                pose,
                float(psi_rows[i, j]),
                int(branch),
                y_rail=shoulder_y,
                euler_order=euler_order,
                R_flange_tcp=R_flange_tcp,
                t_flange_tcp=t_flange_tcp,
            )
            if q_arm is not None:
                candidates[i, j] = full_q_from_arm(q_arm, rail_m=float(rail[i]))
    return candidates, np.isfinite(candidates).all(axis=-1)


def _first_order_dp(candidates: np.ndarray, valid: np.ndarray, q_seed: np.ndarray) -> np.ndarray:
    from rm75_control.kinematics.srs_ik import Q_LOWER, Q_UPPER

    w, k, _ = candidates.shape
    scale = np.r_[0.8, np.maximum(Q_UPPER - Q_LOWER, 1.0e-6)]
    cost = np.full((w, k), np.inf, dtype=np.float64)
    parent = np.full((w, k), -1, dtype=np.int32)
    first = np.flatnonzero(valid[0])
    if first.size == 0:
        raise RuntimeError("SRS graph has no candidate at waypoint 0")
    cost[0, first] = np.mean(((candidates[0, first] - q_seed) / scale) ** 2, axis=-1)
    for i in range(1, w):
        previous = np.flatnonzero(np.isfinite(cost[i - 1]))
        current = np.flatnonzero(valid[i])
        if previous.size == 0 or current.size == 0:
            raise RuntimeError(f"SRS graph disconnects at waypoint {i}")
        q_previous = candidates[i - 1, previous]
        q_current = candidates[i, current]
        edge = np.mean(
            ((q_previous[:, None, :] - q_current[None, :, :]) / scale) ** 2,
            axis=-1,
        )
        total = cost[i - 1, previous, None] + edge
        best = np.argmin(total, axis=0)
        cost[i, current] = total[best, np.arange(len(current))]
        parent[i, current] = previous[best]
    index = np.zeros(w, dtype=np.int32)
    index[-1] = int(np.argmin(cost[-1]))
    if not np.isfinite(cost[-1, index[-1]]):
        raise RuntimeError("SRS graph has no complete coarse path")
    for i in range(w - 1, 0, -1):
        index[i - 1] = parent[i, index[i]]
    return index


def _second_order_dp(candidates: np.ndarray, valid: np.ndarray, q_seed: np.ndarray) -> np.ndarray:
    """Refined DP with equal range-normalized velocity and curvature regrets."""
    from rm75_control.kinematics.srs_ik import Q_LOWER, Q_UPPER

    w, k, _ = candidates.shape
    if w < 3:
        return _first_order_dp(candidates, valid, q_seed)
    scale = np.r_[0.8, np.maximum(Q_UPPER - Q_LOWER, 1.0e-6)]
    pair = np.full((k, k), np.inf, dtype=np.float64)
    parent: list[np.ndarray] = []
    for a in np.flatnonzero(valid[0]):
        for b in np.flatnonzero(valid[1]):
            initial = np.mean(((candidates[0, a] - q_seed) / scale) ** 2)
            velocity = np.mean(((candidates[1, b] - candidates[0, a]) / scale) ** 2)
            pair[a, b] = initial + velocity
    for i in range(2, w):
        next_pair = np.full((k, k), np.inf, dtype=np.float64)
        parent_i = np.full((k, k), -1, dtype=np.int32)
        for b in np.flatnonzero(valid[i - 1]):
            previous = np.flatnonzero(np.isfinite(pair[:, b]))
            if previous.size == 0:
                continue
            qa = candidates[i - 2, previous]
            qb = candidates[i - 1, b]
            for c in np.flatnonzero(valid[i]):
                qc = candidates[i, c]
                velocity = np.mean(((qc - qb) / scale) ** 2)
                curvature = np.mean(((qc - 2.0 * qb + qa) / scale) ** 2, axis=-1)
                total = pair[previous, b] + 0.5 * (velocity + curvature)
                best = int(np.argmin(total))
                next_pair[b, c] = total[best]
                parent_i[b, c] = previous[best]
        if not np.isfinite(next_pair).any():
            raise RuntimeError(f"SRS refined graph disconnects at waypoint {i}")
        pair = next_pair
        parent.append(parent_i)
    end = np.unravel_index(int(np.argmin(pair)), pair.shape)
    index = np.zeros(w, dtype=np.int32)
    index[-2], index[-1] = int(end[0]), int(end[1])
    for i in range(w - 1, 1, -1):
        index[i - 2] = parent[i - 2][index[i - 1], index[i]]
    return index


def solve_srs_trajectory_dp(
    tcp_world: np.ndarray,
    rail_m: np.ndarray,
    q_seed: np.ndarray,
    *,
    kin,
    euler_order: str = "xyz",
    config: SrsTrajectoryDpConfig | None = None,
) -> SrsTrajectoryDpResult:
    """Lift a continuous task path without iterative waypoint QP-IK."""
    from rm75_control.kinematics.srs_ik import branch_from_q

    cfg = config or SrsTrajectoryDpConfig()
    tcp = np.asarray(tcp_world, dtype=np.float64)
    rail = np.asarray(rail_m, dtype=np.float64)
    seed = np.asarray(q_seed, dtype=np.float64)
    if tcp.ndim != 3 or tcp.shape[1:] != (4, 4) or rail.shape != (len(tcp),):
        raise ValueError("tcp_world must be (W,4,4) and rail_m must be (W,)")
    if seed.shape != (8,):
        raise ValueError("q_seed must be an 8-vector")
    branch = int(branch_from_q(seed[1:]))
    started = time.perf_counter()
    try:
        coarse_axis = np.arange(
            float(cfg.psi_lower_deg), float(cfg.psi_upper_deg), float(cfg.coarse_step_deg)
        )
        if coarse_axis.size == 0:
            raise RuntimeError("empty psi search interval")
        coarse_psi = np.deg2rad(np.repeat(coarse_axis[None, :], len(tcp), axis=0))
        coarse_q, coarse_valid = _candidate_lattice(
            tcp, rail, coarse_psi, kin=kin, branch=branch, euler_order=euler_order
        )
        candidate_seconds = time.perf_counter() - started
        coarse_index = _first_order_dp(coarse_q, coarse_valid, seed)
        coarse_selected = coarse_psi[np.arange(len(tcp)), coarse_index]

        offsets = np.arange(
            -float(cfg.refine_half_width_deg),
            float(cfg.refine_half_width_deg) + 0.5 * float(cfg.refine_step_deg),
            float(cfg.refine_step_deg),
        )
        refined_psi = _wrap_pi(coarse_selected[:, None] + np.deg2rad(offsets)[None, :])
        refined_q, refined_valid = _candidate_lattice(
            tcp, rail, refined_psi, kin=kin, branch=branch, euler_order=euler_order
        )
        candidate_seconds = time.perf_counter() - started
        refined_index = _second_order_dp(refined_q, refined_valid, seed)
        q = refined_q[np.arange(len(tcp)), refined_index]
        psi = refined_psi[np.arange(len(tcp)), refined_index]
        if not np.isfinite(q).all():
            raise RuntimeError("selected SRS path contains non-finite values")
        delta = np.diff(q[:, 1:], axis=0)
        rail_delta = np.diff(q[:, 0], axis=0)
        second = np.diff(q[:, 1:], n=2, axis=0)
        max_step = float(np.max(np.abs(delta))) if delta.size else 0.0
        max_rail_step = float(np.max(np.abs(rail_delta))) if rail_delta.size else 0.0
        if max_step > np.deg2rad(float(cfg.maximum_joint_step_deg)) + 1.0e-12:
            raise RuntimeError(f"SRS path exceeds joint-step bound: {np.rad2deg(max_step):.3f} deg")
        if max_rail_step > float(cfg.maximum_rail_step_m) + 1.0e-12:
            raise RuntimeError(f"SRS path exceeds rail-step bound: {max_rail_step:.6f} m")
        return SrsTrajectoryDpResult(
            lift_valid=True,
            q_ref=q.astype(np.float32),
            psi_rad=psi.astype(np.float32),
            branch=branch,
            candidate_seconds=float(candidate_seconds),
            solve_seconds=float(time.perf_counter() - started),
            candidate_count_min=int(refined_valid.sum(axis=1).min()),
            candidate_count_max=int(refined_valid.sum(axis=1).max()),
            maximum_joint_step_rad=max_step,
            maximum_joint_second_difference_rad=float(np.max(np.abs(second))) if second.size else 0.0,
        )
    except Exception as exc:
        return SrsTrajectoryDpResult(
            lift_valid=False,
            q_ref=np.full((len(tcp), 8), np.nan, dtype=np.float32),
            psi_rad=np.full(len(tcp), np.nan, dtype=np.float32),
            branch=branch,
            candidate_seconds=float(time.perf_counter() - started),
            solve_seconds=float(time.perf_counter() - started),
            candidate_count_min=0,
            candidate_count_max=0,
            maximum_joint_step_rad=float("inf"),
            maximum_joint_second_difference_rad=float("inf"),
            failure=str(exc),
        )


__all__ = ["SrsTrajectoryDpConfig", "SrsTrajectoryDpResult", "solve_srs_trajectory_dp"]
