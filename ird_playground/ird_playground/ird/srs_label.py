"""Controller-aligned SRS point-reachability labeling for IRD ground truth.

Mirrors the *point* reachability half of ``resolve_pose_ik_srs`` in
``rm75_control.control.joint_admittance_8dof.pose_ik``:

* ψ grid step 5°, candidates on ``[-π, π)`` via ``np.arange(-π, π, step)``
* drop candidates with ``|wrap(ψ − psi_home)| > max_psi_swing`` (inclusive)
* optional hard bounds ``psi_hard_lower_rad`` / ``psi_hard_upper_rad``
* **fixed** branch (required; no silent OR over all 8)
* ``srs_ik`` returning None ⇒ unreachable

This labeler models **point** reachability only.  Path reachability (10 interior
samples) and top-5 ``_goal_score`` filtering stay at the trajectory operator
(Phase 4); do not pretend they live here.

Pose / rail frame contract (must match ``srs_ik``):
  poses are in ``rail_base``; ``y_rail_m`` is shoulder world-Y =
  ``RAIL_ORIGIN_Y + rail_locked_at_m`` (not the prismatic joint value alone).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


_SRS_MOD = None


def _srs_api() -> Any:
    """Load ``srs_ik`` via the offline namespace stub (no Robotic_Arm SDK).

    ``reachability_modules`` installs a path-only ``rm75_control`` package when
    the real ``__init__`` would pull the vendor SDK.  Always go through that
    helper so offline ``import ird_playground.ird.srs_label`` succeeds; do not
    mix with a direct top-level ``rm75_control`` import in the same process.
    """
    global _SRS_MOD
    if _SRS_MOD is not None:
        return _SRS_MOD
    from ird_playground.ird.gt_common import reachability_modules

    reachability_modules()
    from rm75_control.kinematics import srs_ik as mod

    _SRS_MOD = mod
    return mod


def _wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _pose6_from_Rp(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Pack ``[x,y,z,rx,ry,rz]`` with extrinsic xyz Euler (matches srs_ik)."""
    from scipy.spatial.transform import Rotation

    rpy = Rotation.from_matrix(R).as_euler("xyz", degrees=False)
    return np.concatenate([np.asarray(p, dtype=float).reshape(3), rpy]).astype(float)


def _default_y_rail_m() -> float:
    """Shoulder world-Y in ``rail_base`` at the default locked rail joint."""
    srs = _srs_api()
    # RobotModelSpec.default_probe45().rail_locked_at_m == 0.0
    return float(srs.shoulder_y_from_q_rail(0.0))


@dataclass(frozen=True)
class SrsLabelConfig:
    """Controller-aligned point-reachability config.

    ``branch_id`` is required (runtime locks to ``branch_from_q(seed)``).
    ``y_rail_m`` is shoulder world-Y in the pose frame (``rail_base``):
    ``RAIL_ORIGIN_Y + rail_locked_at_m``.  Poses must be expressed in the same
    frame as Pinocchio FK on the 8-DOF URDF (``rail_base``), not ``base_link``.
    """

    branch_id: int
    psi_grid_step_rad: float = 5.0 * np.pi / 180.0
    max_psi_swing_rad: float = 150.0 * np.pi / 180.0
    psi_home_rad: float = 0.0
    psi_hard_lower_rad: float | None = None
    psi_hard_upper_rad: float | None = None
    y_rail_m: float | None = None  # None → RAIL_ORIGIN_Y + 0 (locked rail)
    check_limits: bool = True
    euler_order: str = "xyz"
    # Default probe45 TCP offset xyz in link_7 (from URDF link_7_to_tcp).
    # Flange mode uses the full URDF ``T_flange_tcp``; this is kept for manifests.
    tcp_offset_xyz: tuple[float, float, float] = (0.0, -0.01523, 0.12135)

    def resolved_y_rail_m(self) -> float:
        if self.y_rail_m is not None:
            return float(self.y_rail_m)
        return _default_y_rail_m()

    def to_manifest(self) -> dict:
        srs = _srs_api()
        return {
            **asdict(self),
            "y_rail_m": self.resolved_y_rail_m(),
            "labeler": "srs_ik_controller_aligned_v1",
            "reachability_kind": "point",  # path reachability is Phase 4
            "models_path_reachability": False,
            "models_top5_goal_score": False,
            "D_WT_FLANGE": float(srs.D_WT_FLANGE),
            "D_BS": float(srs.D_BS),
            "D_SE": float(srs.D_SE),
            "D_EW": float(srs.D_EW),
            "Q_LOWER": np.asarray(srs.Q_LOWER, dtype=float).tolist(),
            "Q_UPPER": np.asarray(srs.Q_UPPER, dtype=float).tolist(),
            "RAIL_ORIGIN_Y": float(srs.RAIL_ORIGIN_Y),
            "euler_order": self.euler_order,
            "psi_hard_lower_rad": self.psi_hard_lower_rad,
            "psi_hard_upper_rad": self.psi_hard_upper_rad,
            "per_sample_overrides": ["psi_homes", "branch_ids"],
            "pose_frame": "rail_base",
            "y_rail_meaning": "shoulder_world_y = RAIL_ORIGIN_Y + q_rail",
            "tool_mode": srs.TOOL_MODE_FLANGE,
        }


def srs_reachable_single(
    R: np.ndarray,
    p: np.ndarray,
    cfg: SrsLabelConfig,
    *,
    branch_id: int | None = None,
    psi_home_rad: float | None = None,
) -> tuple[bool, float | None, int | None, np.ndarray | None]:
    """Return (reachable, best_psi, branch, q_arm) for one TCP pose.

    ``branch_id`` must be supplied either on ``cfg`` or as a per-call override.
    Silently OR-ing all 8 branches is forbidden (runtime locks the seed branch).
    """
    srs = _srs_api()
    pose6 = _pose6_from_Rp(R, p)
    psi_home = float(cfg.psi_home_rad if psi_home_rad is None else psi_home_rad)
    if branch_id is not None:
        branch = int(branch_id) & 0b111
    else:
        branch = int(cfg.branch_id) & 0b111
    y_rail = cfg.resolved_y_rail_m()
    psi_grid = np.arange(-np.pi, np.pi, float(cfg.psi_grid_step_rad))
    best: tuple[float, float, int, np.ndarray] | None = None  # score, psi, branch, q
    for psi in psi_grid:
        if abs(_wrap_pi(float(psi) - psi_home)) > float(cfg.max_psi_swing_rad):
            continue
        if cfg.psi_hard_lower_rad is not None and float(psi) < float(cfg.psi_hard_lower_rad):
            continue
        if cfg.psi_hard_upper_rad is not None and float(psi) > float(cfg.psi_hard_upper_rad):
            continue
        q = srs.srs_ik(
            pose6,
            float(psi),
            int(branch),
            y_rail=float(y_rail),
            check_limits=cfg.check_limits,
            euler_order=cfg.euler_order,
            tool_mode=srs.TOOL_MODE_FLANGE,
        )
        if q is None:
            continue
        score = -abs(_wrap_pi(float(psi) - psi_home))
        if best is None or score > best[0]:
            best = (score, float(psi), int(branch), np.asarray(q, dtype=np.float64))
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
    """Label a batch of TCP poses with controller-aligned SRS point semantics.

    Per-sample ``branch_ids`` / ``psi_homes`` overrides are supported and
    recorded in the manifest; path checks are intentionally absent.
    """
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
    srs = _srs_api()
    q = np.asarray(q7, dtype=np.float64).reshape(7)
    return int(srs.branch_from_q(q)), float(srs.psi_from_q(q))


__all__ = [
    "SrsLabelConfig",
    "branch_and_psi_from_q7",
    "srs_reachable_batch",
    "srs_reachable_single",
]
