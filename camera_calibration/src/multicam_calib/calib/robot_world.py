"""Robot-world / hand-eye: ``T_ref_railbase`` and ``T_tcp_board``.

Measurement model (one sample)::

    T_ref_board = T_ref_railbase @ T_railbase_tcp @ T_tcp_board

OpenCV ``calibrateRobotWorldHandEye`` is used only for the initial guess.
A Cauchy-loss reprojection bundle adjustment then refines both unknowns.
The rail direction used as world +X is ``R_ref_railbase[:, 1]`` (URDF
``rail_y`` axis). An independent PCA fit of TCP motion is diagnostic only.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from multicam_calib.board.apriltag_board import BoardGeometry
from multicam_calib.calib.pnp import solve_view_pose
from multicam_calib.calib.pose_graph import _average_se3, se3_exp, se3_inv, se3_log
from multicam_calib.calib.urdf_fk import UrdfFK
from multicam_calib.ingress.robot_state import T_to_pose6, pose6_to_T
from multicam_calib.io.config import RobotConfig, load_robot
from multicam_calib.io.results import ExtrinsicsSet, Intrinsics
from multicam_calib.recording.session import Sample


CAMERA_GROUP_A = ("cam1", "cam3")
CAMERA_GROUP_B = ("cam2", "cam4")


def T_translate(xyz: Iterable[float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(list(xyz), dtype=np.float64).reshape(3)
    return T


def T_railbase_baselink(rail_y: float, origin_xyz: Iterable[float] = (0.0, -0.4, 0.0)) -> np.ndarray:
    """Pure translation: slider/base_link in ``rail_base`` at the given ``rail_y``."""
    o = np.asarray(list(origin_xyz), dtype=np.float64).reshape(3)
    return T_translate([o[0], o[1] + float(rail_y), o[2]])


def T_baselink_tcp(
    T_railbase_tcp: np.ndarray,
    rail_y: float,
    origin_xyz: Iterable[float] = (0.0, -0.4, 0.0),
) -> np.ndarray:
    """Arm-only pose: strip the URDF rail translation from SHM ``T_railbase_tcp``.

    ``T_railbase_tcp = T_railbase_baselink(rail_y) @ T_baselink_tcp``, and the
    first factor is a known pure translation, so this does not need Pinocchio.
    """
    return se3_inv(T_railbase_baselink(rail_y, origin_xyz)) @ np.asarray(
        T_railbase_tcp, dtype=np.float64
    ).reshape(4, 4)


def visual_slider_point_ref(
    T_ref_board: np.ndarray,
    T_railbase_tcp: np.ndarray,
    rail_y: float,
    T_tcp_board: np.ndarray,
    origin_xyz: Iterable[float] = (0.0, -0.4, 0.0),
) -> np.ndarray:
    """``base_link`` position in the Stage-1 ref frame, arm motion removed.

    Uses the camera board pose minus SHM ``T_baselink_tcp`` and ``T_tcp_board``.
    The leftover is the slider; it should travel along ``R_ref_railbase[:, 1]``.
    """
    T_bl_tcp = T_baselink_tcp(T_railbase_tcp, rail_y, origin_xyz)
    T_ref_bl = (
        np.asarray(T_ref_board, dtype=np.float64).reshape(4, 4)
        @ se3_inv(T_tcp_board)
        @ se3_inv(T_bl_tcp)
    )
    return T_ref_bl[:3, 3].copy()


def rotation_matrix_to_quat_wxyz(R: np.ndarray) -> list[float]:
    q_xyzw = Rotation.from_matrix(np.asarray(R, dtype=np.float64).reshape(3, 3)).as_quat()
    return [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    h.update(path.read_bytes())
    return h.hexdigest()


def sample_T_railbase_tcp(sample: Sample) -> np.ndarray | None:
    meta = sample.metadata or {}
    raw = meta.get("T_railbase_tcp")
    if raw is not None:
        return np.asarray(raw, dtype=np.float64).reshape(4, 4)
    pose = meta.get("pose")
    if pose is not None:
        return pose6_to_T(np.asarray(pose, dtype=np.float64))
    return None


def sample_rail_m(sample: Sample) -> float | None:
    meta = sample.metadata or {}
    if meta.get("rail_m") is None:
        return None
    return float(meta["rail_m"])


def sample_q_deg(sample: Sample) -> np.ndarray | None:
    raw = (sample.metadata or {}).get("q_deg")
    if raw is None:
        return None
    q = np.asarray(raw, dtype=np.float64).reshape(-1)
    if q.size < 7 or not np.all(np.isfinite(q[:7])):
        return None
    return q[:7].copy()


def sample_capture_group(sample: Sample) -> str:
    return str((sample.metadata or {}).get("capture_group") or "")


def fit_rail_axis(points: np.ndarray) -> np.ndarray:
    """Unit direction of the dominant linear motion (PCA / SVD)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 2:
        raise ValueError("Need at least 2 points to fit a rail axis.")
    centered = pts - pts.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    d = vh[0]
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        raise ValueError("Rail-axis fit is degenerate (points coincide).")
    return d / n


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(3)
    bb = np.asarray(b, dtype=np.float64).reshape(3)
    na = np.linalg.norm(aa)
    nb = np.linalg.norm(bb)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    c = float(np.clip((aa / na) @ (bb / nb), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(c)))


def world_axes_from_railbase(T_ref_railbase: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, y, z) in the Stage-1 ref frame.

    +X is the URDF rail axis ``R[:, 1]``. +Z is ``base_link`` +Z orthogonalized
    against +X so the rail is strictly horizontal.
    """
    R = np.asarray(T_ref_railbase, dtype=np.float64).reshape(4, 4)[:3, :3]
    x = R[:, 1].copy()
    x = x / (np.linalg.norm(x) + 1e-12)
    z_bl = R[:, 2].copy()
    z = z_bl - (z_bl @ x) * x
    zn = np.linalg.norm(z)
    if zn < 1e-9:
        raise RuntimeError("base_link Z is parallel to the rail axis — cannot form world +Z.")
    z = z / zn
    if z @ z_bl < 0:
        z = -z
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-12)
    return x, y, z


def _T_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def calibrate_robot_world_handeye_init(
    T_ref_board: list[np.ndarray],
    T_railbase_tcp: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV Shah init.

    Locked by synthetic round-trip: pass board-in-ref and TCP-in-railbase
    **without** inverting. Then::

        T_gripper2cam = T_ref_railbase
        T_base2world  = inv(T_tcp_board)

    which satisfies ``T_ref_board = T_ref_railbase @ T_railbase_tcp @ T_tcp_board``.
    """
    if len(T_ref_board) != len(T_railbase_tcp) or len(T_ref_board) < 3:
        raise ValueError("Hand-eye init needs >= 3 paired poses.")
    R_world2cam: list[np.ndarray] = []
    t_world2cam: list[np.ndarray] = []
    R_base2gripper: list[np.ndarray] = []
    t_base2gripper: list[np.ndarray] = []
    for A, B in zip(T_ref_board, T_railbase_tcp):
        R_world2cam.append(np.asarray(A[:3, :3], dtype=np.float64))
        t_world2cam.append(np.asarray(A[:3, 3], dtype=np.float64).reshape(3, 1))
        R_base2gripper.append(np.asarray(B[:3, :3], dtype=np.float64))
        t_base2gripper.append(np.asarray(B[:3, 3], dtype=np.float64).reshape(3, 1))
    R_base2world, t_base2world, R_gripper2cam, t_gripper2cam = cv2.calibrateRobotWorldHandEye(
        R_world2cam,
        t_world2cam,
        R_base2gripper,
        t_base2gripper,
        method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,
    )
    T_ref_railbase = _T_from_Rt(R_gripper2cam, t_gripper2cam)
    T_tcp_board = se3_inv(_T_from_Rt(R_base2world, t_base2world))
    return T_ref_railbase, T_tcp_board


@dataclass
class RobotObservation:
    sample_index: int
    alias: str
    T_railbase_tcp: np.ndarray
    rail_m: float
    obj_pts: np.ndarray  # (N, 3) board-frame
    img_pts: np.ndarray  # (N, 2)
    T_ref_cam: np.ndarray
    K: np.ndarray
    dist: np.ndarray
    q_deg: np.ndarray | None = None
    capture_group: str = ""


@dataclass
class RobotWorldSolve:
    T_ref_railbase: np.ndarray
    T_tcp_board: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)
    inlier_sample_indices: list[int] = field(default_factory=list)
    joint_zero_offsets_deg: list[float] = field(default_factory=lambda: [0.0] * 7)


def _collect_observations(
    samples: list[Sample],
    board_geom: BoardGeometry,
    intrinsics: dict[str, Intrinsics],
    stage1: ExtrinsicsSet,
    *,
    min_tags: int,
    aliases: Iterable[str] | None = None,
) -> list[RobotObservation]:
    allow = None if aliases is None else set(aliases)
    out: list[RobotObservation] = []
    for sample in samples:
        T_rt = sample_T_railbase_tcp(sample)
        rail = sample_rail_m(sample)
        if T_rt is None or rail is None:
            continue
        for alias, det in sample.views.items():
            if allow is not None and alias not in allow:
                continue
            if det.num_tags() < min_tags:
                continue
            intr = intrinsics.get(alias)
            T_ref_cam = stage1.poses.get(alias)
            if intr is None or T_ref_cam is None:
                continue
            obj, img, used = board_geom.gather_correspondences(det.tags)
            if len(used) < min_tags:
                continue
            out.append(
                RobotObservation(
                    sample_index=int(sample.index),
                    alias=str(alias),
                    T_railbase_tcp=T_rt,
                    rail_m=float(rail),
                    obj_pts=obj,
                    img_pts=img,
                    T_ref_cam=np.asarray(T_ref_cam, dtype=np.float64),
                    K=np.asarray(intr.K, dtype=np.float64),
                    dist=np.asarray(intr.dist, dtype=np.float64).reshape(-1),
                    q_deg=sample_q_deg(sample),
                    capture_group=sample_capture_group(sample),
                )
            )
    return out


def _fused_T_ref_board(
    sample: Sample,
    board_geom: BoardGeometry,
    intrinsics: dict[str, Intrinsics],
    stage1: ExtrinsicsSet,
    *,
    min_tags: int,
    aliases: Iterable[str] | None = None,
) -> np.ndarray | None:
    allow = None if aliases is None else set(aliases)
    estimates: list[np.ndarray] = []
    for alias, det in sample.views.items():
        if allow is not None and alias not in allow:
            continue
        if det.num_tags() < min_tags:
            continue
        intr = intrinsics.get(alias)
        if intr is None:
            continue
        pose = solve_view_pose(board_geom, det.tags, intr, min_tags=min_tags)
        if pose is None:
            continue
        T_ref_cam = stage1.poses.get(alias)
        if T_ref_cam is None:
            continue
        estimates.append(np.asarray(T_ref_cam, dtype=np.float64) @ pose.T_cam_board)
    if not estimates:
        return None
    return _average_se3(estimates)


def _reproject_residual(
    obs: RobotObservation,
    T_ref_railbase: np.ndarray,
    T_tcp_board: np.ndarray,
    T_railbase_tcp: np.ndarray | None = None,
    *,
    board_scale: float = 1.0,
) -> np.ndarray:
    T_rt = obs.T_railbase_tcp if T_railbase_tcp is None else T_railbase_tcp
    T_ref_board = T_ref_railbase @ T_rt @ T_tcp_board
    T_cam_ref = se3_inv(obs.T_ref_cam)
    T_cam_board = T_cam_ref @ T_ref_board
    rvec, _ = cv2.Rodrigues(T_cam_board[:3, :3])
    tvec = T_cam_board[:3, 3].reshape(3, 1)
    obj = obs.obj_pts if board_scale == 1.0 else (obs.obj_pts * float(board_scale))
    proj, _ = cv2.projectPoints(obj, rvec, tvec, obs.K, obs.dist)
    return (proj.reshape(-1, 2) - obs.img_pts).reshape(-1)


def _fk_T_rt(
    obs: RobotObservation,
    fk: UrdfFK | None,
    offsets_j16: np.ndarray | None,
    cache: dict[int, np.ndarray],
) -> np.ndarray:
    if fk is None or obs.q_deg is None or offsets_j16 is None:
        return obs.T_railbase_tcp
    hit = cache.get(obs.sample_index)
    if hit is not None:
        return hit
    T = fk.fk(obs.rail_m, np.deg2rad(obs.q_deg), offsets_j16)
    cache[obs.sample_index] = T
    return T


def _pack(
    T_ref_railbase: np.ndarray,
    T_tcp_board: np.ndarray,
    offsets_j16: np.ndarray | None = None,
) -> np.ndarray:
    parts = [se3_log(T_ref_railbase), se3_log(T_tcp_board)]
    if offsets_j16 is not None:
        parts.append(np.asarray(offsets_j16, dtype=np.float64).reshape(6))
    return np.concatenate(parts)


def _unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    T_rb = se3_exp(x[:6])
    T_tb = se3_exp(x[6:12])
    offsets = np.asarray(x[12:18], dtype=np.float64) if x.size >= 18 else None
    return T_rb, T_tb, offsets


def _ba_residuals(
    x: np.ndarray,
    observations: list[RobotObservation],
    fk: UrdfFK | None,
) -> np.ndarray:
    T_rb, T_tb, offsets = _unpack(x)
    cache: dict[int, np.ndarray] = {}
    chunks = [
        _reproject_residual(obs, T_rb, T_tb, _fk_T_rt(obs, fk, offsets, cache))
        for obs in observations
    ]
    if not chunks:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(chunks)


def _bundle_adjust(
    T_ref_railbase: np.ndarray,
    T_tcp_board: np.ndarray,
    observations: list[RobotObservation],
    *,
    fk: UrdfFK | None = None,
    lock_joint1: bool = False,
    offsets0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, Any]:
    use_offsets = fk is not None and all(o.q_deg is not None for o in observations)
    if use_offsets:
        x0 = _pack(T_ref_railbase, T_tcp_board, offsets0 if offsets0 is not None else np.zeros(6))
        lo = np.full(18, -np.inf)
        hi = np.full(18, np.inf)
        if lock_joint1:
            lo[12] = 0.0
            hi[12] = 0.0
        res = least_squares(
            _ba_residuals,
            x0,
            args=(observations, fk),
            bounds=(lo, hi),
            loss="cauchy",
            f_scale=1.0,
            max_nfev=200,
            verbose=0,
            method="trf",
        )
    else:
        x0 = _pack(T_ref_railbase, T_tcp_board)
        res = least_squares(
            _ba_residuals,
            x0,
            args=(observations, None),
            loss="cauchy",
            f_scale=1.0,
            max_nfev=200,
            verbose=0,
            method="trf",
        )
    T_rb, T_tb, offsets = _unpack(res.x)
    return T_rb, T_tb, offsets, res


def _x_axis_sigma_deg(res: Any) -> tuple[float, float]:
    """Approximate 1-sigma of world +X (deg) and the 2 m linear equivalent (mm)."""
    if getattr(res, "jac", None) is None or res.jac.size == 0:
        return float("nan"), float("nan")
    J = np.asarray(res.jac, dtype=np.float64)
    n, p = J.shape
    dof = max(1, n - p)
    s2 = float(np.sum(np.asarray(res.fun) ** 2) / dof)
    try:
        cov = np.linalg.pinv(J.T @ J) * s2
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    x = np.asarray(res.x, dtype=np.float64)
    T0 = se3_exp(x[:6])
    axis = T0[:3, 1]
    eps = 1e-6
    Jx = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        dx = x[:6].copy()
        dx[i] += eps
        Jx[:, i] = (se3_exp(dx)[:3, 1] - axis) / eps
    cov_x = Jx @ cov[:3, :3] @ Jx.T
    var_tang = float(np.trace(cov_x) - axis @ cov_x @ axis)
    sigma_rad = float(np.sqrt(max(0.0, var_tang)))
    return float(np.rad2deg(sigma_rad)), float(sigma_rad * 2000.0)


def _per_sample_rmse(
    samples: list[Sample],
    observations: list[RobotObservation],
    T_ref_railbase: np.ndarray,
    T_tcp_board: np.ndarray,
    *,
    fk: UrdfFK | None = None,
    offsets_j16: np.ndarray | None = None,
) -> dict[int, float]:
    del samples
    by_idx: dict[int, list[float]] = {}
    cache: dict[int, np.ndarray] = {}
    for obs in observations:
        T_rt = _fk_T_rt(obs, fk, offsets_j16, cache)
        err = _reproject_residual(obs, T_ref_railbase, T_tcp_board, T_rt)
        rmse = float(np.sqrt(np.mean(err.reshape(-1, 2) ** 2)))
        by_idx.setdefault(obs.sample_index, []).append(rmse)
    return {k: float(np.mean(v)) for k, v in by_idx.items()}


def _single_camera_axes(
    samples: list[Sample],
    board_geom: BoardGeometry,
    intrinsics: dict[str, Intrinsics],
    stage1: ExtrinsicsSet,
    *,
    min_tags: int,
    T_tcp_board: np.ndarray,
    rail_origin: Iterable[float],
) -> dict[str, list[float]]:
    """Per-camera rail direction from that camera's own board poses.

    Arm motion is stripped with SHM ``T_railbase_tcp`` so the PCA is the
    slider track, not the waving TCP. Stage 1 is used only to express the
    axis in the shared ref frame.
    """
    out: dict[str, list[float]] = {}
    aliases = sorted({a for s in samples for a in s.views})
    for alias in aliases:
        pts: list[np.ndarray] = []
        rails: list[float] = []
        T_ref_cam = stage1.poses.get(alias)
        if T_ref_cam is None:
            continue
        for sample in samples:
            det = sample.views.get(alias)
            if det is None or det.num_tags() < min_tags:
                continue
            T_rt = sample_T_railbase_tcp(sample)
            rail = sample_rail_m(sample)
            if T_rt is None or rail is None:
                continue
            intr = intrinsics.get(alias)
            if intr is None:
                continue
            pose = solve_view_pose(board_geom, det.tags, intr, min_tags=min_tags)
            if pose is None:
                continue
            T_ref_board = np.asarray(T_ref_cam, dtype=np.float64) @ pose.T_cam_board
            pts.append(
                visual_slider_point_ref(
                    T_ref_board, T_rt, rail, T_tcp_board, rail_origin
                )
            )
            rails.append(float(rail))
        if len(pts) < 3:
            continue
        # Pose-diversity-only views sit at one rail station; PCA is meaningless.
        if max(rails) - min(rails) < 0.15:
            continue
        try:
            d_ref = fit_rail_axis(np.stack(pts, axis=0))
        except ValueError:
            continue
        out[alias] = [float(v) for v in d_ref]
    return out


def _screw_invariants(T_rel: np.ndarray) -> tuple[float, float]:
    """Relative rotation angle (deg) and axial translation (mm)."""
    rvec = Rotation.from_matrix(np.asarray(T_rel[:3, :3], dtype=np.float64)).as_rotvec()
    ang = float(np.linalg.norm(rvec))
    t = np.asarray(T_rel[:3, 3], dtype=np.float64).reshape(3)
    if ang < 1e-9:
        return 0.0, float(np.linalg.norm(t) * 1000.0)
    axis = rvec / ang
    return float(np.degrees(ang)), float(axis @ t * 1000.0)


def _fk_vs_camera_invariants(
    samples: list[Sample],
    T_board_by_idx: dict[int, np.ndarray],
    *,
    fk: UrdfFK | None,
    offsets_j16: np.ndarray | None,
) -> dict[str, float]:
    T_cam: list[np.ndarray] = []
    T_rob: list[np.ndarray] = []
    for s in samples:
        Tb = T_board_by_idx.get(s.index)
        q = sample_q_deg(s)
        rail = sample_rail_m(s)
        if Tb is None or rail is None:
            continue
        if fk is not None and q is not None and offsets_j16 is not None:
            Tr = fk.fk(rail, np.deg2rad(q), offsets_j16)
        else:
            Tr = sample_T_railbase_tcp(s)
        if Tr is None:
            continue
        T_cam.append(Tb)
        T_rob.append(Tr)
    d_ang: list[float] = []
    d_pitch: list[float] = []
    for i in range(len(T_cam)):
        for j in range(i + 1, len(T_cam)):
            a_cam, p_cam = _screw_invariants(se3_inv(T_cam[i]) @ T_cam[j])
            a_rob, p_rob = _screw_invariants(se3_inv(T_rob[i]) @ T_rob[j])
            if max(a_cam, a_rob) < 2.0:
                continue
            d_ang.append(a_cam - a_rob)
            d_pitch.append(p_cam - p_rob)
    if not d_ang:
        return {
            "rel_angle_rms_deg": float("nan"),
            "screw_pitch_rms_mm": float("nan"),
            "n_pairs": 0,
        }
    return {
        "rel_angle_rms_deg": float(np.sqrt(np.mean(np.square(d_ang)))),
        "screw_pitch_rms_mm": float(np.sqrt(np.mean(np.square(d_pitch)))),
        "n_pairs": len(d_ang),
    }


def _inter_camera_board_disagreement_mm(
    samples: list[Sample],
    board_geom: BoardGeometry,
    intrinsics: dict[str, Intrinsics],
    stage1: ExtrinsicsSet,
    *,
    min_tags: int,
) -> dict[str, float]:
    errs: list[float] = []
    depth: list[float] = []
    for sample in samples:
        poses: list[tuple[np.ndarray, np.ndarray]] = []
        for alias, det in sample.views.items():
            if det.num_tags() < min_tags:
                continue
            intr = intrinsics.get(alias)
            T_ref_cam = stage1.poses.get(alias)
            if intr is None or T_ref_cam is None:
                continue
            pose = solve_view_pose(board_geom, det.tags, intr, min_tags=min_tags)
            if pose is None:
                continue
            T_ref_cam = np.asarray(T_ref_cam, dtype=np.float64)
            T_ref_board = T_ref_cam @ pose.T_cam_board
            z_cam = T_ref_cam[:3, :3] @ np.array([0.0, 0.0, 1.0])
            poses.append((T_ref_board[:3, 3].copy(), z_cam))
        if len(poses) < 2:
            continue
        for i in range(len(poses)):
            for j in range(i + 1, len(poses)):
                d = poses[i][0] - poses[j][0]
                errs.append(float(np.linalg.norm(d) * 1000.0))
                depth.append(float(abs(d @ poses[i][1]) * 1000.0))
    if not errs:
        return {
            "mean_mm": float("nan"),
            "median_mm": float("nan"),
            "max_mm": float("nan"),
            "mean_depth_mm": float("nan"),
            "n_pairs": 0,
        }
    arr = np.asarray(errs, dtype=np.float64)
    return {
        "mean_mm": float(np.mean(arr)),
        "median_mm": float(np.median(arr)),
        "max_mm": float(np.max(arr)),
        "mean_depth_mm": float(np.mean(depth)),
        "n_pairs": int(arr.size),
    }


def _estimate_board_scale(
    observations: list[RobotObservation],
    T_ref_railbase: np.ndarray,
    T_tcp_board: np.ndarray,
    *,
    fk: UrdfFK | None,
    offsets_j16: np.ndarray | None,
) -> float:
    def fun(s: np.ndarray) -> np.ndarray:
        cache: dict[int, np.ndarray] = {}
        chunks = [
            _reproject_residual(
                obs,
                T_ref_railbase,
                T_tcp_board,
                _fk_T_rt(obs, fk, offsets_j16, cache),
                board_scale=float(s[0]),
            )
            for obs in observations
        ]
        return np.concatenate(chunks) if chunks else np.zeros(0)

    res = least_squares(fun, np.array([1.0]), method="trf", max_nfev=40)
    return float(res.x[0])


def _tool_frame_delta_deg(
    samples: list[Sample],
    fk: UrdfFK,
) -> dict[str, float]:
    R_deltas: list[np.ndarray] = []
    for s in samples:
        q = sample_q_deg(s)
        rail = sample_rail_m(s)
        T_shm = sample_T_railbase_tcp(s)
        if q is None or rail is None or T_shm is None:
            continue
        T_fk = fk.fk(rail, np.deg2rad(q), np.zeros(6))
        R_deltas.append(T_fk[:3, :3].T @ T_shm[:3, :3])
    if not R_deltas:
        return {"mean_deg": float("nan"), "spread_deg": float("nan")}
    angs = [float(np.degrees(Rotation.from_matrix(R).magnitude())) for R in R_deltas]
    spread = []
    R0 = R_deltas[0]
    for R in R_deltas[1:]:
        spread.append(float(np.degrees(Rotation.from_matrix(R0.T @ R).magnitude())))
    return {
        "mean_deg": float(np.mean(angs)),
        "spread_deg": float(np.max(spread)) if spread else 0.0,
    }


def _leave_group_rmse(
    observations: list[RobotObservation],
    T_ref_railbase: np.ndarray,
    T_tcp_board: np.ndarray,
    *,
    fk: UrdfFK | None,
    lock_joint1: bool,
) -> dict[str, Any]:
    by_idx: dict[int, list[RobotObservation]] = {}
    for obs in observations:
        by_idx.setdefault(obs.sample_index, []).append(obs)
    keys = sorted(by_idx)
    if len(keys) < 8:
        return {"groups": {}, "note": "not enough samples for leave-out RMSE"}
    groups: dict[str, list[RobotObservation]] = {}
    nfold = 4
    for i, idx in enumerate(keys):
        groups.setdefault(f"fold_{i % nfold}", []).extend(by_idx[idx])
    out: dict[str, Any] = {}
    for name, held in groups.items():
        held_keys = {(o.sample_index, o.alias) for o in held}
        train = [o for o in observations if (o.sample_index, o.alias) not in held_keys]
        if len(train) < 8 or len(held) < 2:
            out[name] = {"error": "too few observations"}
            continue
        try:
            T_rb, T_tb, offsets, _ = _bundle_adjust(
                T_ref_railbase, T_tcp_board, train, fk=fk, lock_joint1=lock_joint1
            )
        except Exception as exc:  # noqa: BLE001
            out[name] = {"error": str(exc)}
            continue
        cache: dict[int, np.ndarray] = {}
        chunks = [
            _reproject_residual(o, T_rb, T_tb, _fk_T_rt(o, fk, offsets, cache)) for o in held
        ]
        err = np.concatenate(chunks) if chunks else np.zeros(0)
        rmse = float(np.sqrt(np.mean(err.reshape(-1, 2) ** 2))) if err.size else float("nan")
        out[name] = {
            "n_train": len({o.sample_index for o in train}),
            "n_held": len({o.sample_index for o in held}),
            "rmse_px": rmse,
        }
    rmses = [v["rmse_px"] for v in out.values() if isinstance(v, dict) and "rmse_px" in v]
    return {
        "groups": out,
        "mean_rmse_px": float(np.mean(rmses)) if rmses else float("nan"),
    }


def solve_robot_world(
    samples: list[Sample],
    board_geom: BoardGeometry,
    intrinsics: dict[str, Intrinsics],
    stage1: ExtrinsicsSet,
    *,
    min_tags: int,
    robot_cfg: RobotConfig | None = None,
) -> RobotWorldSolve:
    robot_cfg = robot_cfg or load_robot()
    usable: list[Sample] = []
    T_boards: list[np.ndarray] = []
    T_tcps: list[np.ndarray] = []
    for sample in samples:
        T_rt = sample_T_railbase_tcp(sample)
        if T_rt is None or sample_rail_m(sample) is None:
            continue
        T_rb = _fused_T_ref_board(sample, board_geom, intrinsics, stage1, min_tags=min_tags)
        if T_rb is None:
            continue
        usable.append(sample)
        T_boards.append(T_rb)
        T_tcps.append(T_rt)
    if len(usable) < 3:
        raise ValueError(f"Need >= 3 robot samples with a fused board pose; got {len(usable)}.")

    T_ref_railbase, T_tcp_board = calibrate_robot_world_handeye_init(T_boards, T_tcps)

    observations = _collect_observations(
        usable, board_geom, intrinsics, stage1, min_tags=min_tags
    )
    if not observations:
        raise RuntimeError("No reprojection observations for robot-world BA.")

    fit_cfg = robot_cfg.kinematic_fit
    fk: UrdfFK | None = None
    lock_joint1 = False
    lock_reasons: list[str] = ["joint_7 locked at 0 (absorbed by T_tcp_board)"]
    want_offsets = bool(fit_cfg.joint_offsets)
    have_q = all(o.q_deg is not None for o in observations)
    if want_offsets and have_q:
        try:
            fk = UrdfFK(robot_cfg.wbc_urdf_path())
        except Exception as exc:  # noqa: BLE001
            fk = None
            lock_reasons.append(f"URDF FK unavailable ({exc}); fell back to 12-parameter model")
    elif want_offsets and not have_q:
        lock_reasons.append("samples missing q_deg; fell back to 12-parameter model")

    rails_all = np.asarray([float(sample_rail_m(s) or 0.0) for s in usable], dtype=np.float64)
    rail_span_all = float(rails_all.max() - rails_all.min()) if rails_all.size else 0.0
    if fk is not None and rail_span_all < float(fit_cfg.rail_span_min_m):
        lock_joint1 = True
        lock_reasons.append(
            f"joint_1 locked at 0 (rail span {rail_span_all:.3f} m "
            f"< {fit_cfg.rail_span_min_m:.3f} m)"
        )

    T_ref_railbase, T_tcp_board, offsets0, res0 = _bundle_adjust(
        T_ref_railbase, T_tcp_board, observations, fk=fk, lock_joint1=lock_joint1
    )
    rmse_by_sample = _per_sample_rmse(
        usable, observations, T_ref_railbase, T_tcp_board, fk=fk, offsets_j16=offsets0
    )
    med = float(np.median(list(rmse_by_sample.values()))) if rmse_by_sample else 0.0
    thresh = max(3.0, 3.0 * med)
    rejected = [idx for idx, r in rmse_by_sample.items() if r > thresh]
    inlier_samples = [s for s in usable if s.index not in rejected]
    if len(inlier_samples) < 3:
        inlier_samples = usable
        rejected = []
    inlier_obs = [o for o in observations if o.sample_index not in set(rejected)]
    T_ref_railbase, T_tcp_board, offsets_j16, res = _bundle_adjust(
        T_ref_railbase,
        T_tcp_board,
        inlier_obs,
        fk=fk,
        lock_joint1=lock_joint1,
        offsets0=offsets0,
    )

    fun = np.asarray(res.fun, dtype=np.float64)
    ba_rmse_px = float(np.sqrt(np.mean(fun.reshape(-1, 2) ** 2))) if fun.size else float("nan")
    fx_vals = [float(o.K[0, 0]) for o in inlier_obs]
    fx_med = float(np.median(fx_vals)) if fx_vals else 1300.0
    ba_rmse_at_2m_mm = float(ba_rmse_px / max(fx_med, 1.0) * 2000.0)
    x_sigma_deg, x_sigma_2m_mm = _x_axis_sigma_deg(res)

    rails = np.asarray([float(sample_rail_m(s) or 0.0) for s in inlier_samples], dtype=np.float64)
    rail_baseline_m = float(rails.max() - rails.min()) if rails.size else 0.0
    n_rail_stations = int(len(np.unique(np.round(rails, 2))))

    rail_origin = robot_cfg.rail_y_origin_in_railbase_m
    T_board_by_idx = {s.index: Tb for s, Tb in zip(usable, T_boards)}
    slider_pts: list[np.ndarray] = []
    for s in inlier_samples:
        T_rt = sample_T_railbase_tcp(s)
        rail = sample_rail_m(s)
        T_board = T_board_by_idx.get(s.index)
        if T_rt is None or rail is None or T_board is None:
            continue
        slider_pts.append(
            visual_slider_point_ref(T_board, T_rt, rail, T_tcp_board, rail_origin)
        )
    rail_axis_residual_deg = float("nan")
    if len(slider_pts) >= 2:
        try:
            d_fit = fit_rail_axis(np.stack(slider_pts, axis=0))
            d_urdf = T_ref_railbase[:3, 1]
            if d_fit @ d_urdf < 0:
                d_fit = -d_fit
            rail_axis_residual_deg = _angle_deg(d_fit, d_urdf)
        except ValueError:
            pass

    x_axis, y_axis, z_axis = world_axes_from_railbase(T_ref_railbase)
    tilt_deg = _angle_deg(T_ref_railbase[:3, 2], z_axis)

    per_cam_axis = _single_camera_axes(
        inlier_samples,
        board_geom,
        intrinsics,
        stage1,
        min_tags=min_tags,
        T_tcp_board=T_tcp_board,
        rail_origin=rail_origin,
    )
    cam_angles: dict[str, float] = {}
    for alias, d in per_cam_axis.items():
        dd = np.asarray(d, dtype=np.float64)
        if dd @ x_axis < 0:
            dd = -dd
        cam_angles[alias] = _angle_deg(dd, x_axis)
    cam_axis_spread_deg = float(np.max(list(cam_angles.values()))) if cam_angles else float("nan")

    leave_one: dict[str, Any] = {}
    for label, group in (("cam13", CAMERA_GROUP_A), ("cam24", CAMERA_GROUP_B)):
        try:
            sub = solve_robot_world_subset(
                inlier_samples, board_geom, intrinsics, stage1, min_tags=min_tags, aliases=group
            )
            d = sub[:3, 1]
            if d @ x_axis < 0:
                d = -d
            leave_one[label] = {
                "rail_axis_ref": [float(v) for v in d],
                "angle_from_full_deg": _angle_deg(d, x_axis),
            }
        except Exception as exc:  # noqa: BLE001
            leave_one[label] = {"error": str(exc)}
    pair_angle = float("nan")
    if "rail_axis_ref" in leave_one.get("cam13", {}) and "rail_axis_ref" in leave_one.get("cam24", {}):
        a = np.asarray(leave_one["cam13"]["rail_axis_ref"], dtype=np.float64)
        b = np.asarray(leave_one["cam24"]["rail_axis_ref"], dtype=np.float64)
        if a @ b < 0:
            b = -b
        pair_angle = _angle_deg(a, b)

    cam_counts: dict[str, int] = {}
    for obs in inlier_obs:
        cam_counts[obs.alias] = cam_counts.get(obs.alias, 0) + 1

    urdf = robot_cfg.wbc_urdf_path()
    offsets_deg7 = [0.0] * 7
    if offsets_j16 is not None:
        offsets_deg7 = [float(np.degrees(v)) for v in offsets_j16] + [0.0]
    board_scale = _estimate_board_scale(
        inlier_obs, T_ref_railbase, T_tcp_board, fk=fk, offsets_j16=offsets_j16
    )
    scale_warn = abs(board_scale - 1.0) > float(fit_cfg.board_scale_warn)
    leave_out = _leave_group_rmse(
        inlier_obs, T_ref_railbase, T_tcp_board, fk=fk, lock_joint1=lock_joint1
    )
    cam_board_mm = _inter_camera_board_disagreement_mm(
        inlier_samples, board_geom, intrinsics, stage1, min_tags=min_tags
    )
    screw = _fk_vs_camera_invariants(
        inlier_samples, T_board_by_idx, fk=fk, offsets_j16=offsets_j16
    )
    tool_delta = _tool_frame_delta_deg(inlier_samples, fk) if fk is not None else {}
    diagnostics: dict[str, Any] = {
        "ba_rmse_px": ba_rmse_px,
        "ba_rmse_at_2m_mm": ba_rmse_at_2m_mm,
        "ba_rmse_px_note": (
            "reprojection after rigid T_ref_railbase + T_tcp_board"
            + (" + joint_1..6 offsets" if offsets_j16 is not None else "")
            + "; includes Stage-1 workspace error and arm FK. "
            "Single-view PnP on the EE board is ~0.3 px — this number is not detection noise."
        ),
        "n_samples": len(inlier_samples),
        "n_samples_rejected": len(rejected),
        "rejected_sample_indices": rejected,
        "n_observations": len(inlier_obs),
        "per_camera_n_obs": cam_counts,
        "rail_baseline_m": rail_baseline_m,
        "n_rail_stations": n_rail_stations,
        "x_axis_sigma_deg": x_sigma_deg,
        "x_axis_sigma_at_2m_mm": x_sigma_2m_mm,
        "x_axis_sigma_note": "approximate 1-sigma from (J^T J)^{-1} s^2 under cauchy loss",
        "rail_axis_residual_deg": rail_axis_residual_deg,
        "rail_axis_residual_note": (
            "visual slider track after removing SHM arm (T_baselink_tcp) vs "
            "URDF R_ref_railbase[:,1]; arm does not need to be frozen"
        ),
        "baselink_z_tilt_from_world_z_deg": tilt_deg,
        "per_camera_rail_axis_ref": per_cam_axis,
        "per_camera_rail_axis_angle_from_full_deg": cam_angles,
        "per_camera_rail_axis_spread_deg": cam_axis_spread_deg,
        "leave_one_group": leave_one,
        "leave_one_group_angle_deg": pair_angle,
        "wbc_urdf": str(urdf),
        "wbc_urdf_sha1": file_sha1(urdf) if urdf.is_file() else "",
        "joint_zero_offsets_deg": offsets_deg7,
        "joint_offset_lock_reasons": lock_reasons,
        "joint_offsets_enabled": offsets_j16 is not None,
        "fk_vs_camera_invariants": screw,
        "inter_camera_board_disagreement_mm": cam_board_mm,
        "leave_out_rmse": leave_out,
        "board_scale_est": board_scale,
        "board_scale_warn": scale_warn,
        "board_scale_note": (
            "estimated EE-board scale vs config (1.0 = config is exact). "
            f"Warn if |s-1| > {float(fit_cfg.board_scale_warn)*100:.2f}%."
        ),
        "tool_frame_delta_from_shm": tool_delta,
    }
    if scale_warn:
        diagnostics["board_scale_alert"] = (
            f"estimated EE board scale {board_scale:.5f} differs from config by "
            f"{abs(board_scale - 1.0) * 100:.2f}% — re-measure tag_size_m / tag_spacing_m"
        )
    return RobotWorldSolve(
        T_ref_railbase=T_ref_railbase,
        T_tcp_board=T_tcp_board,
        diagnostics=diagnostics,
        inlier_sample_indices=[int(s.index) for s in inlier_samples],
        joint_zero_offsets_deg=offsets_deg7,
    )


def solve_robot_world_subset(
    samples: list[Sample],
    board_geom: BoardGeometry,
    intrinsics: dict[str, Intrinsics],
    stage1: ExtrinsicsSet,
    *,
    min_tags: int,
    aliases: Iterable[str],
) -> np.ndarray:
    """Return ``T_ref_railbase`` solved on a camera subset (init + BA, no extras)."""
    allow = set(aliases)
    T_boards: list[np.ndarray] = []
    T_tcps: list[np.ndarray] = []
    kept: list[Sample] = []
    for sample in samples:
        T_rt = sample_T_railbase_tcp(sample)
        if T_rt is None:
            continue
        T_rb = _fused_T_ref_board(
            sample, board_geom, intrinsics, stage1, min_tags=min_tags, aliases=allow
        )
        if T_rb is None:
            continue
        kept.append(sample)
        T_boards.append(T_rb)
        T_tcps.append(T_rt)
    if len(kept) < 3:
        raise ValueError(f"subset {sorted(allow)} has only {len(kept)} usable samples")
    T0, T1 = calibrate_robot_world_handeye_init(T_boards, T_tcps)
    obs = _collect_observations(
        kept, board_geom, intrinsics, stage1, min_tags=min_tags, aliases=allow
    )
    if not obs:
        raise RuntimeError(f"subset {sorted(allow)} has no observations")
    T_ref_railbase, _, _, _ = _bundle_adjust(T0, T1, obs)
    return T_ref_railbase


def build_robot_world_export(
    *,
    T_world_railbase: np.ndarray,
    T_ref_railbase: np.ndarray,
    T_tcp_board: np.ndarray,
    robot_cfg: RobotConfig,
    diagnostics: dict[str, Any],
    rail_direction_world: Iterable[float] | None = None,
) -> dict[str, Any]:
    origin = robot_cfg.rail_y_origin_in_railbase_m
    T_world_baselink = T_world_railbase @ T_railbase_baselink(0.0, origin)
    R = T_world_baselink[:3, :3]
    if rail_direction_world is None:
        rail_direction_world = T_world_railbase[:3, 1]
    return {
        "T_world_railbase": T_world_railbase.tolist(),
        "T_world_baselink_at_rail0": T_world_baselink.tolist(),
        "T_ref_railbase": T_ref_railbase.tolist(),
        "T_tcp_board": T_tcp_board.tolist(),
        "rail_direction_world": [float(v) for v in np.asarray(rail_direction_world, dtype=np.float64).reshape(3)],
        "base_pos_m": [float(v) for v in T_world_baselink[:3, 3]],
        "base_quat_wxyz": rotation_matrix_to_quat_wxyz(R),
        "base_link_height_above_floor_m": float(robot_cfg.base_link_height_above_floor_m),
        "rail_y_origin_in_railbase_m": [float(v) for v in origin],
        "joint_zero_offsets_deg": [
            float(v) for v in (diagnostics.get("joint_zero_offsets_deg") or [0.0] * 7)
        ],
        "diagnostics": diagnostics,
        "wbc_urdf_sha1": diagnostics.get("wbc_urdf_sha1", ""),
    }


__all__ = [
    "CAMERA_GROUP_A",
    "CAMERA_GROUP_B",
    "RobotWorldSolve",
    "T_baselink_tcp",
    "T_railbase_baselink",
    "T_translate",
    "T_to_pose6",
    "build_robot_world_export",
    "calibrate_robot_world_handeye_init",
    "file_sha1",
    "fit_rail_axis",
    "pose6_to_T",
    "rotation_matrix_to_quat_wxyz",
    "sample_T_railbase_tcp",
    "sample_q_deg",
    "solve_robot_world",
    "visual_slider_point_ref",
    "world_axes_from_railbase",
]
