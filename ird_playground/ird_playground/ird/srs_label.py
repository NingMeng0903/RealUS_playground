"""Controller-aligned SRS reachability labeling for IRD ground truth.

Mirrors ``resolve_pose_ik_srs`` semantics in
``rm75_control.control.joint_admittance_8dof.pose_ik``:

* ψ grid step 5°, candidates on (-π, π]
* drop candidates with ``|wrap(ψ − psi_home)| > max_psi_swing``
* fixed branch (no mid-query branch switch)
* ``srs_ik`` returning None ⇒ unreachable
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from rm75_control.kinematics.srs_ik import (
    D_WT_FLANGE,
    branch_from_q,
    d_wt_from_tcp_offset,
    psi_from_q,
    srs_ik,
)


def _wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _pose6_from_Rp(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Pack ``[x,y,z,rx,ry,rz]`` with extrinsic xyz Euler (matches srs_ik)."""
    from scipy.spatial.transform import Rotation

    rpy = Rotation.from_matrix(R).as_euler("xyz", degrees=False)
    return np.concatenate([np.asarray(p, dtype=float).reshape(3), rpy]).astype(float)


@dataclass(frozen=True)
class SrsLabelConfig:
    psi_grid_step_rad: float = 5.0 * np.pi / 180.0
    max_psi_swing_rad: float = 150.0 * np.pi / 180.0
    psi_home_rad: float = 0.0
    branch_id: int | None = None  # None → try all 8, OR-reduce
    y_rail_m: float = 0.0
    check_limits: bool = True
    # Default probe45 TCP offset xyz in link_7 (from URDF link_7_to_tcp).
    tcp_offset_xyz: tuple[float, float, float] = (0.0, -0.01523, 0.12135)

    def d_wt(self) -> float:
        return float(d_wt_from_tcp_offset(np.asarray(self.tcp_offset_xyz, dtype=float)))

    def to_manifest(self) -> dict:
        return {
            **asdict(self),
            "d_wt_m": self.d_wt(),
            "labeler": "srs_ik_controller_aligned_v1",
            "D_WT_FLANGE": D_WT_FLANGE,
        }


def srs_reachable_single(
    R: np.ndarray,
    p: np.ndarray,
    cfg: SrsLabelConfig,
    *,
    branch_id: int | None = None,
    psi_home_rad: float | None = None,
) -> tuple[bool, float | None, int | None, np.ndarray | None]:
    """Return (reachable, best_psi, branch, q_arm) for one TCP pose."""
    pose6 = _pose6_from_Rp(R, p)
    psi_home = float(cfg.psi_home_rad if psi_home_rad is None else psi_home_rad)
    branch = cfg.branch_id if branch_id is None else int(branch_id)
    branches = list(range(8)) if branch is None else [int(branch) & 0b111]
    d_wt = cfg.d_wt()
    psi_grid = np.arange(-np.pi, np.pi, float(cfg.psi_grid_step_rad))
    best: tuple[float, float, int, np.ndarray] | None = None  # score, psi, branch, q
    for b in branches:
        for psi in psi_grid:
            if abs(_wrap_pi(float(psi) - psi_home)) > float(cfg.max_psi_swing_rad):
                continue
            q = srs_ik(
                pose6,
                float(psi),
                int(b),
                y_rail=float(cfg.y_rail_m),
                check_limits=cfg.check_limits,
                d_wt=d_wt,
            )
            if q is None:
                continue
            score = -abs(_wrap_pi(float(psi) - psi_home))
            if best is None or score > best[0]:
                best = (score, float(psi), int(b), np.asarray(q, dtype=np.float64))
    if best is None:
        return False, None, None, None
    return True, best[1], best[2], best[3]


def srs_reachable_batch(
    positions: np.ndarray,
    rotations: np.ndarray,
    cfg: SrsLabelConfig,
    *,
    branch_ids: np.ndarray | None = None,
    psi_homes: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Label a batch of TCP poses with controller-aligned SRS semantics."""
    p = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    R = np.asarray(rotations, dtype=np.float64).reshape(-1, 3, 3)
    n = p.shape[0]
    reachable = np.zeros(n, dtype=bool)
    best_psi = np.full(n, np.nan, dtype=np.float64)
    best_branch = np.full(n, -1, dtype=np.int32)
    q_best = np.full((n, 7), np.nan, dtype=np.float64)
    for i in range(n):
        b = None if branch_ids is None else int(branch_ids[i])
        ph = None if psi_homes is None else float(psi_homes[i])
        ok, psi, branch, q = srs_reachable_single(
            R[i], p[i], cfg, branch_id=b, psi_home_rad=ph
        )
        reachable[i] = ok
        if ok and psi is not None and branch is not None and q is not None:
            best_psi[i] = psi
            best_branch[i] = branch
            q_best[i] = q
    return {
        "reachable": reachable,
        "psi": best_psi,
        "branch": best_branch,
        "q_best": q_best,
    }


def branch_and_psi_from_q7(q7: np.ndarray) -> tuple[int, float]:
    q = np.asarray(q7, dtype=np.float64).reshape(7)
    return int(branch_from_q(q)), float(psi_from_q(q))


__all__ = [
    "SrsLabelConfig",
    "branch_and_psi_from_q7",
    "srs_reachable_batch",
    "srs_reachable_single",
]
