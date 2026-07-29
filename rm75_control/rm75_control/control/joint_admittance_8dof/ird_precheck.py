"""Pre-execution IRD validation gate for TCP + rail trajectories.

Runtime CBF is an in-motion distance barrier, not a pre-dispatch gate.  This
module is the fail-closed entry that must run before streaming a plan:

* IRD conservative clearance (raw field score − conformal threshold hook)
* SRS ψ / branch for each waypoint (controller-aligned ``srs_ik``)
* Self-collision minimum distance

``ird_playground`` is imported optionally.  If the package / checkpoint is
unavailable, validation fails closed rather than silently passing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionModel
from rm75_control.control.joint_admittance_8dof.model import (
    RobotKinematics,
    full_q_from_arm,
)
from rm75_control.kinematics.srs_ik import (
    branch_from_q,
    flange_tcp_from_kin,
    psi_from_q,
    shoulder_y_from_q_rail,
    srs_ik,
)


class IrdUnavailableError(RuntimeError):
    """Raised when IRD cannot be loaded and fail-closed policy is active."""


@dataclass(frozen=True)
class PrecheckConfig:
    """Thresholds for the pre-execution gate."""

    clearance_margin: float = 0.0
    """Minimum calibrated IRD clearance (field units; typically metres)."""

    conformal_threshold: float = 0.0
    """Split-conformal threshold subtracted from raw scores (hook)."""

    collision_min_distance_m: float = 0.01
    """Minimum self-collision distance (aligns with CBF ``d_safe``)."""

    fail_closed_without_ird: bool = True
    require_srs: bool = True
    psi_step_rad: float = np.deg2rad(5.0)
    max_psi_swing_rad: float = np.deg2rad(150.0)
    branch_id: int | None = None
    """If None, lock to ``branch_from_q`` of the seed / previous solution."""


@dataclass
class WaypointPrecheck:
    index: int
    rail_m: float
    ird_raw: float
    ird_clearance: float
    psi_rad: float
    branch_id: int
    collision_min_distance_m: float
    ok: bool
    reasons: list[str] = field(default_factory=list)
    q_arm: np.ndarray | None = None


@dataclass
class TrajectoryPrecheckResult:
    ok: bool
    waypoints: list[WaypointPrecheck]
    ird_available: bool
    message: str = ""

    @property
    def n_failed(self) -> int:
        return sum(1 for w in self.waypoints if not w.ok)


ClearanceFn = Callable[[np.ndarray, float], float]
"""``(tcp_pose6, rail_m) -> raw IRD score`` (higher = more reachable)."""


def try_import_ird() -> tuple[bool, str]:
    """Return ``(available, detail)`` without raising."""
    try:
        import ird_playground  # noqa: F401

        return True, getattr(ird_playground, "__file__", "ird_playground")
    except Exception as exc:  # pragma: no cover - env dependent
        return False, f"{type(exc).__name__}: {exc}"


def calibrated_ird_clearance(raw_score: float, conformal_threshold: float) -> float:
    """Conservative clearance: ``raw − conformal_threshold`` (Phase-3 hook)."""
    return float(raw_score) - float(conformal_threshold)


def load_conformal_threshold(path: str | None) -> float:
    """Load ``m_safe`` / ``threshold`` from a conformal JSON; 0.0 if missing."""
    if not path:
        return 0.0
    from pathlib import Path

    p = Path(path)
    if not p.is_file():
        return 0.0
    import json

    data = json.loads(p.read_text(encoding="utf-8"))
    return float(data.get("m_safe", data.get("threshold", 0.0)))


def precheck_config_from_conformal(
    path: str | None,
    *,
    base: PrecheckConfig | None = None,
) -> PrecheckConfig:
    """Return a PrecheckConfig with conformal_threshold filled from JSON."""
    cfg = base or PrecheckConfig()
    thr = load_conformal_threshold(path)
    return PrecheckConfig(
        clearance_margin=cfg.clearance_margin,
        conformal_threshold=thr,
        collision_min_distance_m=cfg.collision_min_distance_m,
        fail_closed_without_ird=cfg.fail_closed_without_ird,
        require_srs=cfg.require_srs,
        psi_step_rad=cfg.psi_step_rad,
        max_psi_swing_rad=cfg.max_psi_swing_rad,
        branch_id=cfg.branch_id,
    )


def _pose6_matrix(pose6: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation as Rsc

    pose6 = np.asarray(pose6, dtype=float).reshape(6)
    R = Rsc.from_euler("xyz", pose6[3:6], degrees=False).as_matrix()
    return R, pose6[:3].copy()


def resolve_srs_waypoint(
    pose6: np.ndarray,
    rail_m: float,
    *,
    kin: RobotKinematics,
    q_seed_arm: np.ndarray | None,
    cfg: PrecheckConfig,
) -> tuple[np.ndarray | None, float, int, str | None]:
    """Point-SRS solve for one TCP waypoint. Returns q_arm, ψ, branch, error."""
    pose6 = np.asarray(pose6, dtype=float).reshape(6)
    y_rail = shoulder_y_from_q_rail(float(rail_m))
    R_ft, t_ft = flange_tcp_from_kin(kin)
    if q_seed_arm is None:
        q_seed_arm = np.zeros(7, dtype=float)
    branch = (
        int(cfg.branch_id)
        if cfg.branch_id is not None
        else int(branch_from_q(np.asarray(q_seed_arm, dtype=float)))
    )
    try:
        psi_home = float(psi_from_q(np.asarray(q_seed_arm, dtype=float)))
    except Exception:
        psi_home = 0.0

    best_q = None
    best_psi = float("nan")
    best_score = -np.inf
    step = float(cfg.psi_step_rad)
    for psi in np.arange(-np.pi, np.pi, step):
        if abs(((psi - psi_home + np.pi) % (2.0 * np.pi)) - np.pi) > float(
            cfg.max_psi_swing_rad
        ):
            continue
        q = srs_ik(
            pose6,
            float(psi),
            int(branch),
            y_rail,
            R_flange_tcp=R_ft,
            t_flange_tcp=t_ft,
        )
        if q is None:
            continue
        score = -abs(((float(psi) - psi_home + np.pi) % (2.0 * np.pi)) - np.pi)
        if score > best_score:
            best_score = score
            best_q = np.asarray(q, dtype=float)
            best_psi = float(psi)
    if best_q is None:
        return None, float("nan"), branch, "srs_unreachable"
    return best_q, best_psi, branch, None


def make_field_clearance_fn(
    field: Any,
    *,
    kin: RobotKinematics,
    T_axis_world: np.ndarray | None = None,
    device: str = "cpu",
) -> ClearanceFn:
    """Wrap an ``ird_playground`` field exposing ``score_world`` as clearance fn."""
    import torch

    if T_axis_world is None:
        T_axis = np.eye(4, dtype=np.float64)
    else:
        T_axis = np.asarray(T_axis_world, dtype=np.float64).reshape(4, 4)
    T_axis_t = torch.as_tensor(T_axis, dtype=torch.float32, device=device)

    def _fn(pose6: np.ndarray, rail_m: float) -> float:
        # Rail shifts the shoulder/axis frame along +Y in rail_base.
        T_ax = T_axis.copy()
        T_ax[1, 3] = float(T_ax[1, 3])  # keep; rail handled by pose frame contract
        del rail_m  # poses are assumed already in rail_base for the given rail
        R, p = _pose6_matrix(pose6)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = p
        with torch.no_grad():
            score = field.score_world(
                torch.as_tensor(T, dtype=torch.float32, device=device),
                T_axis_t,
            )
        return float(np.asarray(score.detach().cpu()).reshape(-1)[0])

    return _fn


def validate_tcp_rail_trajectory(
    tcp_poses: Sequence[np.ndarray] | np.ndarray,
    rail_sequence: Sequence[float] | np.ndarray,
    *,
    kin: RobotKinematics | None = None,
    collision: CollisionModel | None = None,
    clearance_fn: ClearanceFn | None = None,
    cfg: PrecheckConfig | None = None,
    q_seed_arm: np.ndarray | None = None,
) -> TrajectoryPrecheckResult:
    """Fail-closed pre-execution check for a TCP trajectory + rail sequence.

    Parameters
    ----------
    tcp_poses:
        ``(N, 6)`` TCP poses ``[x,y,z,rx,ry,rz]`` in ``rail_base``.
    rail_sequence:
        Length ``N`` prismatic rail joint values (metres).
    clearance_fn:
        Optional raw IRD scorer.  When ``None``, attempts a soft import probe
        and fails closed if ``cfg.fail_closed_without_ird``.
    """
    cfg = cfg or PrecheckConfig()
    poses = np.asarray(tcp_poses, dtype=float)
    rails = np.asarray(rail_sequence, dtype=float).reshape(-1)
    if poses.ndim != 2 or poses.shape[1] != 6:
        raise ValueError(f"tcp_poses must be (N,6), got {poses.shape}")
    if rails.shape[0] != poses.shape[0]:
        raise ValueError(
            f"rail_sequence length {rails.shape[0]} != n_waypoints {poses.shape[0]}"
        )

    kin = kin or RobotKinematics()
    if collision is None:
        collision = CollisionModel(kin.model)

    ird_available = clearance_fn is not None
    if clearance_fn is None:
        ok_import, detail = try_import_ird()
        ird_available = bool(ok_import)
        if cfg.fail_closed_without_ird and clearance_fn is None:
            # Import alone is not enough — need an explicit scorer / checkpoint.
            msg = (
                "IRD clearance function unavailable "
                f"(import_ok={ok_import}, detail={detail}); fail-closed"
            )
            return TrajectoryPrecheckResult(
                ok=False,
                waypoints=[],
                ird_available=False,
                message=msg,
            )

    waypoints: list[WaypointPrecheck] = []
    prev_q = None if q_seed_arm is None else np.asarray(q_seed_arm, dtype=float)
    all_ok = True
    for i in range(poses.shape[0]):
        reasons: list[str] = []
        pose = poses[i]
        rail = float(rails[i])

        raw = 0.0
        if clearance_fn is not None:
            raw = float(clearance_fn(pose, rail))
        cal = calibrated_ird_clearance(raw, cfg.conformal_threshold)
        if cal < float(cfg.clearance_margin):
            reasons.append(
                f"ird_clearance={cal:.4g} < margin={cfg.clearance_margin:.4g}"
            )

        q_arm = None
        psi = float("nan")
        branch = -1
        if cfg.require_srs:
            q_arm, psi, branch, err = resolve_srs_waypoint(
                pose,
                rail,
                kin=kin,
                q_seed_arm=prev_q,
                cfg=cfg,
            )
            if err is not None:
                reasons.append(err)
            else:
                prev_q = q_arm
        else:
            branch = 0
            psi = 0.0
            q_arm = prev_q

        d_col = float("inf")
        if q_arm is not None:
            q8 = full_q_from_arm(q_arm, rail_m=rail)
            collision.update(q8)
            d_col = float(collision.min_distance())
            if d_col < float(cfg.collision_min_distance_m):
                reasons.append(
                    f"collision_min_distance={d_col:.4g} < "
                    f"{cfg.collision_min_distance_m:.4g}"
                )
        elif cfg.require_srs:
            reasons.append("collision_skipped_no_q")

        ok = len(reasons) == 0
        all_ok = all_ok and ok
        waypoints.append(
            WaypointPrecheck(
                index=i,
                rail_m=rail,
                ird_raw=raw,
                ird_clearance=cal,
                psi_rad=psi,
                branch_id=int(branch),
                collision_min_distance_m=d_col,
                ok=ok,
                reasons=reasons,
                q_arm=None if q_arm is None else np.asarray(q_arm, dtype=float),
            )
        )

    msg = "ok" if all_ok else f"{sum(1 for w in waypoints if not w.ok)} waypoint(s) failed"
    return TrajectoryPrecheckResult(
        ok=all_ok,
        waypoints=waypoints,
        ird_available=ird_available,
        message=msg,
    )


def assert_trajectory_precheck(result: TrajectoryPrecheckResult) -> None:
    """Raise ``RuntimeError`` if the precheck failed (fail-closed dispatch)."""
    if result.ok:
        return
    details = []
    for w in result.waypoints:
        if not w.ok:
            details.append(f"wp{w.index}:{','.join(w.reasons)}")
    raise RuntimeError(
        f"pre-execution IRD gate failed: {result.message}; " + "; ".join(details[:8])
    )


__all__ = [
    "ClearanceFn",
    "IrdUnavailableError",
    "PrecheckConfig",
    "TrajectoryPrecheckResult",
    "WaypointPrecheck",
    "assert_trajectory_precheck",
    "calibrated_ird_clearance",
    "load_conformal_threshold",
    "make_field_clearance_fn",
    "precheck_config_from_conformal",
    "resolve_srs_waypoint",
    "try_import_ird",
    "validate_tcp_rail_trajectory",
]
