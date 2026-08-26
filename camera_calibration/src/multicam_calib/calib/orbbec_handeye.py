"""Eye-in-hand: Orbbec color optical frame relative to ``link_7``.

The camera rides on the wrist and looks at a **fixed** AprilTag board
(the large board on the table/bed — not the EE board, which moves with the arm).

OpenCV ``calibrateHandEye`` (Park) gives ``T_link7_cam``. A Cauchy-loss
reprojection BA then refines ``T_link7_cam`` and ``T_railbase_board``.

    T_railbase_cam = T_railbase_link7 @ T_link7_cam
    T_cam_board    = inv(T_railbase_cam) @ T_railbase_board

``T_railbase_link7 = T_railbase_tcp @ inv(T_link7_tcp)`` from SHM + URDF.
``T_tcp_cam = inv(T_link7_tcp) @ T_link7_cam`` is stored for tooling that
still talks TCP.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rsc

from multicam_calib.board.apriltag_board import BoardGeometry
from multicam_calib.calib.pnp import solve_view_pose
from multicam_calib.calib.pose_graph import se3_exp, se3_inv, se3_log
from multicam_calib.calib.urdf_fk import UrdfFK
from multicam_calib.io.config import RESULTS_DIR, RobotConfig, load_robot
from multicam_calib.io.results import Intrinsics, load_intrinsics_map, load_joint_zero_offsets_deg

# Gemini / Astra F color factory pinhole at 640x480 (SDK, not Stage 4).
FACTORY_ORBBEC_COLOR_FX = 456.0
_ASPECT_SCALE_TOL = 0.01


@dataclass
class HandeyeObservation:
    T_railbase_link7: np.ndarray
    T_cam_board: np.ndarray
    object_pts: np.ndarray
    image_pts: np.ndarray
    n_tags: int
    pnp_rmse_px: float
    rail_m: float
    q_deg: list[float]


@dataclass
class OrbbecHandeyeResult:
    T_link7_cam: np.ndarray
    T_tcp_cam: np.ndarray
    T_link7_tcp: np.ndarray
    T_railbase_board: np.ndarray
    n_samples: int
    ba_rmse_px: float
    init_rmse_px: float
    pnp_rmse_px: float
    notes: list[str] = field(default_factory=list)
    joint_zero_offsets_deg: list[float] = field(default_factory=lambda: [0.0] * 7)
    per_view_ba_rmse_px: list[float] = field(default_factory=list)
    shm_vs_fk_mm: float | None = None
    shm_vs_fk_deg: float | None = None
    gripper_rot_span_deg: float | None = None
    color_intrinsics: dict[str, Any] = field(default_factory=dict)

    def as_yaml_dict(self) -> dict[str, Any]:
        def _T(T: np.ndarray) -> list[list[float]]:
            return [[float(v) for v in row] for row in np.asarray(T)]

        xyz, rpy = _xyz_rpy_from_T(self.T_link7_cam)
        return {
            "metadata": {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "n_samples": int(self.n_samples),
                "ba_rmse_px": float(self.ba_rmse_px),
                "init_rmse_px": float(self.init_rmse_px),
                "pnp_rmse_px": float(self.pnp_rmse_px),
                "gripper_frame": "link_7",
                "camera_frame": "orbbec_color",
                "urdf_link": "wrist_camera",
                "urdf_joint": "link_7_to_wrist_camera",
                "optical_convention": "opencv_z_forward_y_down",
                "joint_zero_offsets_deg": [float(v) for v in self.joint_zero_offsets_deg],
                "per_view_ba_rmse_px": [float(v) for v in self.per_view_ba_rmse_px],
                "shm_vs_fk_mm": None if self.shm_vs_fk_mm is None else float(self.shm_vs_fk_mm),
                "shm_vs_fk_deg": None if self.shm_vs_fk_deg is None else float(self.shm_vs_fk_deg),
                "gripper_rot_span_deg": None
                if self.gripper_rot_span_deg is None
                else float(self.gripper_rot_span_deg),
            },
            "T_link7_cam": _T(self.T_link7_cam),
            "T_link7_cam_xyz_m": [float(v) for v in xyz],
            "T_link7_cam_rpy_xyz_rad": [float(v) for v in rpy],
            "T_tcp_cam": _T(self.T_tcp_cam),
            "T_link7_tcp": _T(self.T_link7_tcp),
            "T_railbase_board": _T(self.T_railbase_board),
            "color_intrinsics": dict(self.color_intrinsics),
            "notes": list(self.notes),
        }


def orbbec_handeye_path() -> Path:
    return RESULTS_DIR / "orbbec_handeye.yaml"


def save_orbbec_handeye(result: OrbbecHandeyeResult, path: Path | None = None) -> Path:
    p = path or orbbec_handeye_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(result.as_yaml_dict(), fh, sort_keys=False, allow_unicode=True)
    return p


CAPTURES_SCHEMA = "orbbec_handeye_captures_v1"


def orbbec_handeye_captures_path() -> Path:
    return RESULTS_DIR / "orbbec_handeye_captures.yaml"


def orbbec_handeye_captures_last_path() -> Path:
    return RESULTS_DIR / "last" / "orbbec_handeye_captures.yaml"


def _plain_detections(raw: Any) -> dict[int, list[list[float]]]:
    out: dict[int, list[list[float]]] = {}
    if not isinstance(raw, dict):
        return out
    for key, corners in raw.items():
        pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] < 4:
            continue
        out[int(key)] = [[float(x), float(y)] for x, y in pts.tolist()]
    return out


def captures_to_payload(captures: list[dict[str, Any]]) -> dict[str, Any]:
    slim: list[dict[str, Any]] = []
    for cap in captures:
        size = cap.get("image_size") or [0, 0]
        slim.append(
            {
                "n_tags": int(cap.get("n_tags") or 0),
                "rail_m": float(cap.get("rail_m") or 0.0),
                "q_deg": [float(v) for v in (cap.get("q_deg") or [])],
                "image_size": [int(size[0]), int(size[1])] if len(size) >= 2 else [0, 0],
                "T_railbase_tcp": [
                    [float(v) for v in row]
                    for row in np.asarray(cap.get("T_railbase_tcp") or np.eye(4), dtype=np.float64).reshape(4, 4)
                ],
                "detections": _plain_detections(cap.get("detections")),
            }
        )
    return {
        "schema": CAPTURES_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(slim),
        "captures": slim,
    }


def payload_to_captures(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("captures") or []
    else:
        rows = []
    out: list[dict[str, Any]] = []
    for cap in rows:
        if not isinstance(cap, dict):
            continue
        dets = _plain_detections(cap.get("detections"))
        if not dets:
            continue
        size = cap.get("image_size") or [0, 0]
        out.append(
            {
                "n_tags": int(cap.get("n_tags") or len(dets)),
                "rail_m": float(cap.get("rail_m") or 0.0),
                "q_deg": [float(v) for v in (cap.get("q_deg") or [])],
                "image_size": [int(size[0]), int(size[1])] if len(size) >= 2 else [0, 0],
                "T_railbase_tcp": [
                    [float(v) for v in row]
                    for row in np.asarray(cap.get("T_railbase_tcp") or np.eye(4), dtype=np.float64).reshape(4, 4)
                ],
                "detections": dets,
            }
        )
    return out


def save_orbbec_handeye_captures(
    captures: list[dict[str, Any]],
    path: Path | None = None,
    *,
    also_last: bool = False,
) -> Path:
    p = path or orbbec_handeye_captures_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(captures_to_payload(captures), sort_keys=False, allow_unicode=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(p)
    if also_last:
        last = orbbec_handeye_captures_last_path()
        last.parent.mkdir(parents=True, exist_ok=True)
        last.write_text(text, encoding="utf-8")
    return p


def load_orbbec_handeye_captures(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or orbbec_handeye_captures_path()
    if not p.is_file():
        return []
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return payload_to_captures(data)


def _aspect_ratio(size: tuple[int, int]) -> float:
    w, h = int(size[0]), int(size[1])
    if h < 1:
        return float("inf")
    return float(w) / float(h)


def aspect_ratio_change(src: tuple[int, int], dst: tuple[int, int]) -> float:
    a0 = _aspect_ratio(src)
    if not np.isfinite(a0) or a0 <= 0.0:
        return float("inf")
    return abs(_aspect_ratio(dst) - a0) / a0


def _pinhole_guess_intrinsics(width: int, height: int) -> Intrinsics:
    from multicam_calib.devices.orbbec import pinhole_guess_v4l

    return pinhole_guess_v4l(int(width), int(height)).as_intrinsics()


def orbbec_fx_compare_text(*, factory_fx: float | None = None, saved: Intrinsics | None = None) -> str:
    """Status-bar line: saved fx (source) vs factory fx."""
    if saved is None:
        saved = load_intrinsics_map().get("orbbec")
    fac = FACTORY_ORBBEC_COLOR_FX if factory_fx is None else float(factory_fx)
    if saved is None:
        return f"saved fx=none  factory fx={fac:.1f}"
    return (
        f"saved fx={float(saved.K[0, 0]):.1f} ({saved.source})  "
        f"factory fx={fac:.1f}"
    )


def load_orbbec_color_intrinsics(
    *,
    factory: Intrinsics | None = None,
    image_size: tuple[int, int] | None = None,
    saved: Intrinsics | None = None,
) -> Intrinsics:
    """Saved ``orbbec`` color K, else factory from the open device.

    Same-aspect resizes scale fx/fy/cx/cy. A width/height ratio change over
    ~1% is refused (640x480 → 1920x1080 is not a pinhole scale) and falls
    back to ``factory`` or a V4L guess.
    """
    from multicam_calib.calib.orbbec_rgbd import PinholeModel, scale_pinhole_to_image_size

    if saved is None:
        saved = load_intrinsics_map().get("orbbec")
    if saved is not None:
        if image_size is None or (
            int(saved.image_size[0]) == int(image_size[0]) and int(saved.image_size[1]) == int(image_size[1])
        ):
            return saved
        dst = (int(image_size[0]), int(image_size[1]))
        if aspect_ratio_change(saved.image_size, dst) > _ASPECT_SCALE_TOL:
            warnings.warn(
                f"Orbbec K is {saved.image_size[0]}x{saved.image_size[1]}; "
                f"refusing scale to {dst[0]}x{dst[1]} (aspect change). "
                "Use a native 1080p entry or factory/V4L guess.",
                UserWarning,
                stacklevel=2,
            )
            if factory is not None:
                return factory
            return _pinhole_guess_intrinsics(dst[0], dst[1])
        scaled = scale_pinhole_to_image_size(
            PinholeModel(saved.K, saved.dist, saved.image_size, str(saved.source)),
            dst,
        )
        return scaled.as_intrinsics()
    if factory is not None:
        return factory
    raise RuntimeError("No Orbbec color K/d. Run Stage 4 or Open Orbbec for factory intrinsics.")


@dataclass
class ColorIntrinsicsRefit:
    intrinsics: Intrinsics
    rms_px: float
    n_views: int
    n_points: int
    previous_fx: float | None


def refit_color_intrinsics_from_captures(
    captures: list[dict[str, Any]],
    *,
    board_geom: BoardGeometry,
    min_tags: int = 8,
    min_views: int = 10,
    max_fx_rel_change: float = 0.20,
    previous: Intrinsics | None = None,
    source: str = "apriltag26",
) -> ColorIntrinsicsRefit:
    """Chessboard-style ``calibrateCamera`` on Stage 5 AprilTag correspondences."""
    obj_list: list[np.ndarray] = []
    img_list: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    for cap in captures:
        dets_raw = cap.get("detections") or {}
        dets = {int(k): np.asarray(v, dtype=np.float64) for k, v in dets_raw.items()}
        obj, img, used = board_geom.gather_correspondences(dets)
        if len(used) < int(min_tags):
            continue
        obj_list.append(obj.astype(np.float32))
        img_list.append(img.astype(np.float32))
        size = cap.get("image_size") or []
        if len(size) >= 2:
            image_size = (int(size[0]), int(size[1]))
    if len(obj_list) < int(min_views):
        raise RuntimeError(f"need ≥{min_views} views with ≥{min_tags} tags, got {len(obj_list)}")
    if image_size is None:
        raise RuntimeError("captures have no image_size")
    rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(obj_list, img_list, image_size, None, None)
    fx = float(K[0, 0])
    if previous is None:
        previous = load_intrinsics_map().get("orbbec")
    prev_fx = None if previous is None else float(previous.K[0, 0])
    ref_fx = prev_fx if prev_fx is not None else FACTORY_ORBBEC_COLOR_FX
    if ref_fx > 1.0 and abs(fx - ref_fx) / ref_fx > float(max_fx_rel_change):
        raise RuntimeError(
            f"refit fx={fx:.2f} differs from {ref_fx:.2f} by more than "
            f"{100.0 * max_fx_rel_change:.0f}%; refusing to write"
        )
    n_pts = int(sum(p.shape[0] for p in obj_list))
    return ColorIntrinsicsRefit(
        intrinsics=Intrinsics(
            K=np.asarray(K, dtype=np.float64),
            dist=np.asarray(dist, dtype=np.float64).reshape(-1),
            image_size=image_size,
            source=str(source),
        ),
        rms_px=float(rms),
        n_views=len(obj_list),
        n_points=n_pts,
        previous_fx=prev_fx,
    )


def _xyz_rpy_from_T(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    xyz = T[:3, 3].copy()
    rpy = Rsc.from_matrix(T[:3, :3]).as_euler("xyz")
    return xyz, rpy


def _color_intrinsics_payload(intr: Intrinsics, *, rms_px: float | None = None) -> dict[str, Any]:
    K = np.asarray(intr.K, dtype=np.float64).reshape(3, 3)
    return {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "dist": [float(v) for v in np.asarray(intr.dist, dtype=np.float64).reshape(-1)],
        "image_size": [int(intr.image_size[0]), int(intr.image_size[1])],
        "source": str(intr.source),
        "factory_fx": float(FACTORY_ORBBEC_COLOR_FX),
        "rms_px": None if rms_px is None else float(rms_px),
    }


def _Rt(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return T[:3, :3].copy(), T[:3, 3].copy()


def _from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def calibrate_handeye_init(
    T_railbase_link7: list[np.ndarray],
    T_cam_board: list[np.ndarray],
) -> np.ndarray:
    if len(T_railbase_link7) != len(T_cam_board) or len(T_cam_board) < 3:
        raise ValueError(f"need ≥3 paired poses, got {len(T_cam_board)}")
    R_g, t_g, R_t, t_t = [], [], [], []
    for Tg, Tc in zip(T_railbase_link7, T_cam_board):
        Rg, tg = _Rt(Tg)
        Rt, tt = _Rt(Tc)
        R_g.append(Rg)
        t_g.append(tg.reshape(3, 1))
        R_t.append(Rt)
        t_t.append(tt.reshape(3, 1))
    R_cg, t_cg = cv2.calibrateHandEye(R_g, t_g, R_t, t_t, method=cv2.CALIB_HAND_EYE_PARK)
    return _from_Rt(R_cg, t_cg.reshape(3))


def _mean_T_railbase_board(
    T_link7_cam: np.ndarray,
    T_railbase_link7: list[np.ndarray],
    T_cam_board: list[np.ndarray],
) -> np.ndarray:
    acc = []
    for Tg, Tc in zip(T_railbase_link7, T_cam_board):
        acc.append(Tg @ T_link7_cam @ Tc)
    R = sum(T[:3, :3] for T in acc) / len(acc)
    u, _, vt = np.linalg.svd(R)
    R = u @ vt
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = u @ vt
    t = sum(T[:3, 3] for T in acc) / len(acc)
    return _from_Rt(R, t)


def _reproj_rmse(
    obs: list[HandeyeObservation],
    T_link7_cam: np.ndarray,
    T_railbase_board: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> float:
    errs: list[float] = []
    for o in obs:
        T_cam_board = se3_inv(o.T_railbase_link7 @ T_link7_cam) @ T_railbase_board
        rvec, _ = cv2.Rodrigues(T_cam_board[:3, :3])
        tvec = T_cam_board[:3, 3].reshape(3, 1)
        proj, _ = cv2.projectPoints(o.object_pts, rvec, tvec, K, dist)
        err = np.linalg.norm(proj.reshape(-1, 2) - o.image_pts, axis=1)
        errs.extend(err.tolist())
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")


def _per_view_rmse(
    obs: list[HandeyeObservation],
    T_link7_cam: np.ndarray,
    T_railbase_board: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> list[float]:
    out: list[float] = []
    for o in obs:
        out.append(_reproj_rmse([o], T_link7_cam, T_railbase_board, K, dist))
    return out


def _rotation_span_deg(Ts: list[np.ndarray]) -> float:
    best = 0.0
    for i, a in enumerate(Ts):
        for b in Ts[i + 1 :]:
            r = a[:3, :3].T @ b[:3, :3]
            ang = float(np.degrees(np.arccos(np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0))))
            if ang > best:
                best = ang
    return best


def _shm_vs_fk(
    captures: list[dict[str, Any]],
    fk: UrdfFK,
    offsets_j16_rad: np.ndarray | None,
) -> tuple[float | None, float | None]:
    dts: list[float] = []
    angs: list[float] = []
    for cap in captures:
        q = cap.get("q_deg")
        raw = cap.get("T_railbase_tcp")
        if q is None or raw is None or len(q) < 7:
            continue
        t_shm = np.asarray(raw, dtype=np.float64).reshape(4, 4)
        t_fk = fk.fk(float(cap.get("rail_m", 0.0)), np.deg2rad(np.asarray(q, dtype=np.float64)), offsets_j16_rad)
        dts.append(float(np.linalg.norm(t_fk[:3, 3] - t_shm[:3, 3]) * 1000.0))
        r = t_shm[:3, :3].T @ t_fk[:3, :3]
        angs.append(float(np.degrees(np.arccos(np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0)))))
    if not dts:
        return None, None
    return float(np.median(dts)), float(np.median(angs))


def _bundle_adjust(
    obs: list[HandeyeObservation],
    T_link7_cam: np.ndarray,
    T_railbase_board: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    x0 = np.concatenate([se3_log(T_link7_cam), se3_log(T_railbase_board)])

    def residual(x: np.ndarray) -> np.ndarray:
        T_lc = se3_exp(x[:6])
        T_rb = se3_exp(x[6:])
        chunks: list[np.ndarray] = []
        for o in obs:
            T_cam_board = se3_inv(o.T_railbase_link7 @ T_lc) @ T_rb
            rvec, _ = cv2.Rodrigues(T_cam_board[:3, :3])
            tvec = T_cam_board[:3, 3].reshape(3, 1)
            proj, _ = cv2.projectPoints(o.object_pts, rvec, tvec, K, dist)
            chunks.append((proj.reshape(-1, 2) - o.image_pts).reshape(-1))
        return np.concatenate(chunks)

    res = least_squares(residual, x0, loss="cauchy", f_scale=1.0, max_nfev=80)
    T_lc = se3_exp(res.x[:6])
    T_rb = se3_exp(res.x[6:])
    rmse = _reproj_rmse(obs, T_lc, T_rb, K, dist)
    return T_lc, T_rb, rmse


def observations_from_captures(
    captures: list[dict[str, Any]],
    *,
    board_geom: BoardGeometry,
    intrinsics: Intrinsics,
    fk: UrdfFK,
    min_tags: int = 8,
    offsets_j16_rad: np.ndarray | None = None,
) -> list[HandeyeObservation]:
    out: list[HandeyeObservation] = []
    for cap in captures:
        dets = cap["detections"]
        pose = solve_view_pose(board_geom, dets, intrinsics, min_tags=min_tags)
        if pose is None:
            continue
        q = cap.get("q_deg")
        if q is not None and len(q) >= 7:
            T_rt = fk.fk(float(cap.get("rail_m", 0.0)), np.deg2rad(np.asarray(q, dtype=np.float64)), offsets_j16_rad)
        else:
            T_rt = np.asarray(cap["T_railbase_tcp"], dtype=np.float64).reshape(4, 4)
        T_l7 = fk.T_railbase_link7(T_rt)
        obj, img, used = board_geom.gather_correspondences(dets)
        out.append(
            HandeyeObservation(
                T_railbase_link7=T_l7,
                T_cam_board=pose.T_cam_board,
                object_pts=obj.astype(np.float64),
                image_pts=img.astype(np.float64),
                n_tags=len(used),
                pnp_rmse_px=float(pose.reprojection_rmse_px),
                rail_m=float(cap.get("rail_m", 0.0)),
                q_deg=[float(v) for v in cap.get("q_deg", [])],
            )
        )
    return out


def solve_orbbec_handeye(
    captures: list[dict[str, Any]],
    *,
    board_geom: BoardGeometry,
    intrinsics: Intrinsics,
    robot_cfg: RobotConfig | None = None,
    min_tags: int = 8,
    min_samples: int = 6,
    apply_joint_offsets: bool = False,
    color_rms_px: float | None = None,
) -> OrbbecHandeyeResult:
    robot_cfg = robot_cfg or load_robot()
    fk = UrdfFK(robot_cfg.wbc_urdf_path())
    if apply_joint_offsets:
        offsets_deg = load_joint_zero_offsets_deg(urdf_sha1=fk.sha1)
        offsets_j16 = np.deg2rad(offsets_deg[:6])
    else:
        offsets_deg = np.zeros(7, dtype=np.float64)
        offsets_j16 = None
    obs = observations_from_captures(
        captures,
        board_geom=board_geom,
        intrinsics=intrinsics,
        fk=fk,
        min_tags=min_tags,
        offsets_j16_rad=offsets_j16,
    )
    if len(obs) < min_samples:
        raise RuntimeError(f"need ≥{min_samples} valid views, got {len(obs)}")
    T_g = [o.T_railbase_link7 for o in obs]
    T_c = [o.T_cam_board for o in obs]
    T_link7_cam = calibrate_handeye_init(T_g, T_c)
    T_rb = _mean_T_railbase_board(T_link7_cam, T_g, T_c)
    K = intrinsics.K
    dist = intrinsics.dist
    init_rmse = _reproj_rmse(obs, T_link7_cam, T_rb, K, dist)
    T_link7_cam, T_rb, ba_rmse = _bundle_adjust(obs, T_link7_cam, T_rb, K, dist)
    T_l7_tcp = fk.T_link7_tcp()
    T_tcp_cam = se3_inv(T_l7_tcp) @ T_link7_cam
    pnp_rmse = float(np.mean([o.pnp_rmse_px for o in obs]))
    per_view = _per_view_rmse(obs, T_link7_cam, T_rb, K, dist)
    shm_mm, shm_deg = _shm_vs_fk(captures, fk, offsets_j16)
    rot_span = _rotation_span_deg(T_g)
    notes = [
        "Gripper frame is URDF link_7 (flange), not tcp.",
        "Target is the fixed large AprilTag board. Do not use the EE board.",
        f"PnP uses color K/d source={intrinsics.source} "
        f"fx={float(intrinsics.K[0, 0]):.2f} (factory fx={FACTORY_ORBBEC_COLOR_FX:.1f}).",
        "Runtime depth: T_link7_depth = T_link7_cam @ T_color_depth (factory D2C).",
        (
            "FK offsets ON: " + ", ".join(f"j{i + 1}={v:.3f}°" for i, v in enumerate(offsets_deg))
            if apply_joint_offsets
            else "FK offsets OFF (raw q_deg / SHM pose, no joint_zero_offsets.yaml)."
        ),
    ]
    return OrbbecHandeyeResult(
        T_link7_cam=T_link7_cam,
        T_tcp_cam=T_tcp_cam,
        T_link7_tcp=T_l7_tcp,
        T_railbase_board=T_rb,
        n_samples=len(obs),
        ba_rmse_px=ba_rmse,
        init_rmse_px=init_rmse,
        pnp_rmse_px=pnp_rmse,
        notes=notes,
        joint_zero_offsets_deg=[float(v) for v in offsets_deg.tolist()],
        per_view_ba_rmse_px=per_view,
        shm_vs_fk_mm=shm_mm,
        shm_vs_fk_deg=shm_deg,
        gripper_rot_span_deg=rot_span,
        color_intrinsics=_color_intrinsics_payload(intrinsics, rms_px=color_rms_px),
    )
