"""Stage 2: world alignment via robot geometry, bed height, and bed-corner envelope.

Pipeline (requires Stage 1 ``extrinsics_rel.yaml``):

1. **Robot** — EE-board hand-eye; world +X = rail joint axis, +Z = base_link Z
   orthogonalized against the rail; floor through base_link minus 274 mm.
2. **Bed** — multi-position captures; parallel plane height ``z_bed`` above floor.
3. **Corners** — four captures, one board placement per physical bed corner
   (any rotation allowed); fuse four board corner tags (151/1/162/12) per
   sample, pool every physical tag-corner point across all captures, and fit
   the minimum-area bounding rectangle (any orientation, not axis-aligned) →
   bed size; origin at bed-center projected to floor. World XY stay rail-aligned
   unless ``align_xy_to_bed`` is on; the bed may keep a nonzero ``bed_rotation_deg``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from multicam_calib.board.apriltag_board import BoardGeometry, build_board_geometry
from multicam_calib.calib.plane_fit import (
    AxisAlignedRect,
    PlaneFitResult,
    RotatedRect,
    WorldFrameBasis,
    axis_aligned_rect_from_xy,
    min_area_rect_from_xy,
    rect_corners_xy,
    rotate_basis_about_z,
    signed_heights_along_normal,
    transform_xy_between_bases,
)
from multicam_calib.calib.pnp import solve_view_pose
from multicam_calib.calib.pose_graph import _average_se3, se3_inv
from multicam_calib.calib.robot_world import (
    T_railbase_baselink,
    build_robot_world_export,
    sample_T_railbase_tcp,
    sample_rail_m,
    solve_robot_world,
    world_axes_from_railbase,
)
from multicam_calib.io.config import AppConfig, RobotConfig, WorldConfig, load_board_ee, load_robot, load_world
from multicam_calib.io.genesis_export import build_genesis_bundle, save_genesis_bundle
from multicam_calib.io.results import (
    ExtrinsicsSet,
    Intrinsics,
    WorldMeta,
    extrinsics_world_path,
    save_extrinsics,
    save_robot_world,
    save_world_meta,
)
from multicam_calib.recording.session import RecordingSession, Sample, ViewDetections
from multicam_calib.recording.stage2_session import Stage2AlignedState, Stage2Phase, Stage2SessionBundle


@dataclass
class CornerCapturePreview:
    """Result of validating a corners-phase capture before persisting."""

    ok: bool
    message: str
    n_qualifying_cameras: int = 0
    missing_tag_ids: list[int] = field(default_factory=list)
    fusion_std_mm: list[float] = field(default_factory=list)
    rect_xy: dict[str, float] | None = None


@dataclass
class PhaseCapturePreview:
    """Result of validating a floor/bed capture before persisting."""

    ok: bool
    message: str
    n_qualifying_cameras: int = 0
    height_above_floor_mm: float | None = None
    rail_m: float | None = None


@dataclass
class PhaseSampleIssue:
    """One floor/bed sample that must be removed before running that phase."""

    index: int
    reason: str


@dataclass
class Stage2Report:
    reference: str
    phase: Stage2Phase
    T_ref_world: np.ndarray | None
    world_poses: dict[str, np.ndarray] | None
    world_meta: WorldMeta | None
    floor_residual_mm: float
    bed_height_m: float | None
    bed_residual_mm: float | None
    fusion_residual_m: float
    robot_diagnostics: dict | None = None


def _tag_center_board(board_geom: BoardGeometry, tag_id: int) -> np.ndarray:
    corners = board_geom.corners_by_tag[int(tag_id)]
    return corners.mean(axis=0)


def _transform_board_points(T: np.ndarray, board_geom: BoardGeometry, tag_ids: list[int]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for tid in tag_ids:
        p = _tag_center_board(board_geom, tid)
        ph = np.ones(4, dtype=np.float64)
        ph[:3] = p
        out[int(tid)] = (T @ ph)[:3]
    return out


def _collect_ref_points(
    session: RecordingSession,
    board_geom: BoardGeometry,
    intrinsics: dict[str, Intrinsics],
    stage1: ExtrinsicsSet,
    *,
    min_tags: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return (Nx3 ref points, list of board +X axes in ref) from all samples.

    Uses one fused ``T_ref_board`` per sample (multi-cam average) and board-frame
    **model** corners in metres — never pixel coordinates.
    """
    pts: list[np.ndarray] = []
    x_axes: list[np.ndarray] = []
    for sample in session.samples:
        T_ref_board, _ = _estimate_T_ref_board_per_sample(
            sample, board_geom, intrinsics, stage1, min_tags=min_tags
        )
        if T_ref_board is None:
            continue
        x_axes.append(T_ref_board[:3, 0].copy())
        tag_ids: set[int] = set()
        for det in sample.views.values():
            tag_ids.update(int(tid) for tid in det.tags.keys())
        for tag_id in sorted(tag_ids):
            model = board_geom.corners_by_tag.get(tag_id)
            if model is None:
                continue
            for k in range(4):
                ph = np.ones(4, dtype=np.float64)
                ph[:3] = model[k]
                pts.append((T_ref_board @ ph)[:3])
    if not pts:
        return np.empty((0, 3)), []
    return np.stack(pts, axis=0), x_axes


def _estimate_T_ref_board_per_sample(
    sample: Sample,
    board_geom: BoardGeometry,
    intrinsics: dict[str, Intrinsics],
    stage1: ExtrinsicsSet,
    *,
    min_tags: int,
) -> tuple[np.ndarray | None, int]:
    estimates: list[np.ndarray] = []
    for alias, det in sample.views.items():
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
        estimates.append(T_ref_cam @ pose.T_cam_board)
    if len(estimates) < 1:
        return None, 0
    return _average_se3(estimates), len(estimates)


def _height_above_floor_plane_mm(origin_ref: np.ndarray, floor_normal: np.ndarray, floor_d: float) -> float:
    return float((origin_ref @ floor_normal - floor_d) * 1000.0)


def _sample_height_above_floor_mm(
    sample: Sample,
    board_geom: BoardGeometry,
    intrinsics: dict[str, Intrinsics],
    stage1: ExtrinsicsSet,
    state: Stage2AlignedState,
    *,
    min_tags: int,
) -> float | None:
    if not state.floor_aligned:
        return None
    T, _ = _estimate_T_ref_board_per_sample(
        sample, board_geom, intrinsics, stage1, min_tags=min_tags
    )
    if T is None:
        return None
    n = np.asarray(state.floor_normal, dtype=np.float64)
    d = float(state.floor_d)
    return _height_above_floor_plane_mm(T[:3, 3], n, d)


def _first_bed_capture_timestamp_ns(bed_samples: list[Sample]) -> int | None:
    if not bed_samples:
        return None
    return int(min(s.host_timestamp_ns for s in bed_samples))


def _n_qualifying_cameras(views: dict[str, ViewDetections], min_tags: int) -> int:
    return sum(1 for det in views.values() if det.num_tags() >= int(min_tags))


def robot_capture_coverage(samples: list[Sample], *, min_tags: int) -> dict[str, Any]:
    """Display-only stats: rail span, per-camera counts, multi-cam samples."""
    rails: list[float] = []
    cam_counts: dict[str, int] = {}
    n_bridge = 0
    n_scan = 0
    n_pose = 0
    for sample in samples:
        r = sample_rail_m(sample)
        if r is not None:
            rails.append(float(r))
        n_ok = 0
        for alias, det in sample.views.items():
            if det.num_tags() >= min_tags:
                cam_counts[alias] = cam_counts.get(alias, 0) + 1
                n_ok += 1
        if n_ok >= 3:
            n_bridge += 1
        grp = str((sample.metadata or {}).get("capture_group") or "")
        if grp == "rail_scan":
            n_scan += 1
        elif grp == "pose_diversity":
            n_pose += 1
    baseline = (max(rails) - min(rails)) if rails else 0.0
    return {
        "rail_baseline_m": baseline,
        "n_rail_stations": int(len(set(round(x, 2) for x in rails))),
        "per_camera": cam_counts,
        "n_multicam": n_bridge,
        "n_rail_scan": n_scan,
        "n_pose_diversity": n_pose,
    }


def validate_robot_capture(
    *,
    views: dict[str, ViewDetections],
    world_cfg: WorldConfig,
    shm_ok: bool,
    shm_age_s: float | None,
    still_ok: bool,
    still_message: str,
    rail_m: float | None,
    max_age_s: float,
) -> PhaseCapturePreview:
    """Per-sample robot-phase gates: tags, SHM freshness, stillness."""
    min_tags = int(world_cfg.min_tags_robot_view)
    min_cams = int(world_cfg.min_cameras_robot)
    n_cams = _n_qualifying_cameras(views, min_tags)
    if n_cams < min_cams:
        return PhaseCapturePreview(
            ok=False,
            message=(
                f"Too few qualifying cameras ({n_cams}) for robot capture "
                f"(need >= {min_cams} with >= {min_tags} EE tags)."
            ),
            n_qualifying_cameras=n_cams,
            rail_m=rail_m,
        )
    if not shm_ok:
        return PhaseCapturePreview(
            ok=False,
            message=(
                "Robot SHM is missing or stale — cannot attach /dev/shm/rm75_state "
                "(restart the 8-DOF controller if the name was unlinked)."
            ),
            n_qualifying_cameras=n_cams,
            rail_m=rail_m,
        )
    if shm_age_s is not None and shm_age_s > float(max_age_s):
        return PhaseCapturePreview(
            ok=False,
            message=f"Robot SHM stale ({shm_age_s * 1000:.0f} ms > {max_age_s * 1000:.0f} ms).",
            n_qualifying_cameras=n_cams,
            rail_m=rail_m,
        )
    if not still_ok:
        return PhaseCapturePreview(
            ok=False,
            message=still_message,
            n_qualifying_cameras=n_cams,
            rail_m=rail_m,
        )
    rail_txt = f", rail={rail_m:.3f} m" if rail_m is not None else ""
    age_txt = f", shm age {shm_age_s * 1000:.0f} ms" if shm_age_s is not None else ""
    return PhaseCapturePreview(
        ok=True,
        message=f"OK — {n_cams} camera(s){rail_txt}{age_txt}; {still_message}",
        n_qualifying_cameras=n_cams,
        rail_m=rail_m,
    )


def validate_bed_capture(
    *,
    views: dict[str, ViewDetections],
    board_geom: BoardGeometry,
    intrinsics: dict[str, Intrinsics],
    stage1: ExtrinsicsSet,
    app_cfg: AppConfig,
    world_cfg: WorldConfig,
    state: Stage2AlignedState,
) -> PhaseCapturePreview:
    """Reject bed captures before floor align or when board is still on the ground."""
    if not state.floor_aligned:
        return PhaseCapturePreview(
            ok=False,
            message="Bed plane capture rejected: run robot hand-eye first.",
        )

    min_tags = int(app_cfg.calibration.min_tags_per_view)
    tmp = Sample(index=-1, host_timestamp_ns=0, views=views, metadata={"phase": "bed"})
    T, n_cams = _estimate_T_ref_board_per_sample(
        tmp, board_geom, intrinsics, stage1, min_tags=min_tags
    )
    if T is None or n_cams < 2:
        return PhaseCapturePreview(
            ok=False,
            message=f"Too few qualifying cameras ({n_cams}) for bed plane capture.",
            n_qualifying_cameras=n_cams,
        )

    h_mm = _height_above_floor_plane_mm(
        T[:3, 3],
        np.asarray(state.floor_normal, dtype=np.float64),
        float(state.floor_d),
    )
    if h_mm < float(world_cfg.bed_min_height_above_floor_mm):
        return PhaseCapturePreview(
            ok=False,
            message=(
                f"Bed plane capture rejected: board is only {h_mm:.0f} mm above the floor "
                f"(need >= {world_cfg.bed_min_height_above_floor_mm:.0f} mm). "
                "Place the board on the bed and use Bed plane mode."
            ),
            n_qualifying_cameras=n_cams,
            height_above_floor_mm=h_mm,
        )

    return PhaseCapturePreview(
        ok=True,
        message=f"OK — {n_cams} camera(s), {h_mm:.0f} mm above floor",
        n_qualifying_cameras=n_cams,
        height_above_floor_mm=h_mm,
    )


def audit_robot_samples_for_run(
    bundle: Stage2SessionBundle,
    world_cfg: WorldConfig,
) -> list[PhaseSampleIssue]:
    """List robot-folder samples that cannot be used for hand-eye."""
    issues: list[PhaseSampleIssue] = []
    min_tags = int(world_cfg.min_tags_robot_view)
    min_cams = int(world_cfg.min_cameras_robot)
    for sample in bundle.robot.samples:
        phase_meta = sample.metadata.get("phase")
        if phase_meta not in (None, "robot"):
            issues.append(
                PhaseSampleIssue(sample.index, f"wrong phase metadata {phase_meta!r} (expected robot)")
            )
        if sample_T_railbase_tcp(sample) is None or sample_rail_m(sample) is None:
            issues.append(PhaseSampleIssue(sample.index, "missing T_railbase_tcp / rail_m in metadata"))
        n_cams = _n_qualifying_cameras(sample.views, min_tags)
        if n_cams < min_cams:
            issues.append(
                PhaseSampleIssue(
                    sample.index,
                    f"only {n_cams} camera(s) with >= {min_tags} EE tags (need >= {min_cams})",
                )
            )
    return issues


def audit_bed_samples_for_run(
    bundle: Stage2SessionBundle,
    board_geom: BoardGeometry,
    intrinsics: dict[str, Intrinsics],
    stage1: ExtrinsicsSet,
    app_cfg: AppConfig,
    world_cfg: WorldConfig,
    state: Stage2AlignedState,
) -> list[PhaseSampleIssue]:
    issues: list[PhaseSampleIssue] = []
    min_tags = int(app_cfg.calibration.min_tags_per_view)

    for sample in bundle.bed.samples:
        phase_meta = sample.metadata.get("phase")
        if phase_meta not in (None, "bed"):
            issues.append(
                PhaseSampleIssue(sample.index, f"wrong phase metadata {phase_meta!r} (expected bed)")
            )
        if not state.floor_aligned:
            issues.append(PhaseSampleIssue(sample.index, "floor not aligned yet"))
            continue
        h_mm = _sample_height_above_floor_mm(
            sample, board_geom, intrinsics, stage1, state, min_tags=min_tags
        )
        if h_mm is None:
            issues.append(PhaseSampleIssue(sample.index, "no qualifying camera views"))
        elif h_mm < float(world_cfg.bed_min_height_above_floor_mm):
            issues.append(
                PhaseSampleIssue(
                    sample.index,
                    f"only {h_mm:.0f} mm above floor (need >= {world_cfg.bed_min_height_above_floor_mm:.0f} mm)",
                )
            )
    return issues


def _raise_if_phase_sample_issues(phase: Stage2Phase, issues: list[PhaseSampleIssue]) -> None:
    if not issues:
        return
    label = {"robot": "robot hand-eye", "bed": "bed plane", "corners": "corners"}[phase]
    lines = "\n".join(f"  #{i.index:03d}: {i.reason}" for i in issues)
    raise ValueError(
        f"{label.capitalize()} run rejected — remove invalid samples (Delete selected):\n{lines}"
    )


def _camera_board_poses_for_sample(
    sample: Sample,
    board_geom: BoardGeometry,
    intrinsics: dict[str, Intrinsics],
    stage1: ExtrinsicsSet,
    *,
    min_tags: int,
) -> list[tuple[str, np.ndarray]]:
    """Per-camera T_ref_board estimates for one multi-cam sample."""
    out: list[tuple[str, np.ndarray]] = []
    for alias, det in sample.views.items():
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
        out.append((alias, T_ref_cam @ pose.T_cam_board))
    return out


def _fused_board_pose(cam_poses: list[tuple[str, np.ndarray]]) -> tuple[np.ndarray | None, int]:
    if not cam_poses:
        return None, 0
    estimates = [T for _, T in cam_poses]
    return _average_se3(estimates), len(estimates)


def _corner_pose_deviation_mm(
    cam_poses: list[tuple[str, np.ndarray]],
    T_fused: np.ndarray,
    board_geom: BoardGeometry,
    corner_ids: list[int],
) -> tuple[float, list[float]]:
    """Max 3D deviation between per-camera and fused board pose at corner tags."""
    fused_corners = _transform_board_points(T_fused, board_geom, corner_ids)
    devs_mm: list[float] = []
    for _, T_i in cam_poses:
        corners_i = _transform_board_points(T_i, board_geom, corner_ids)
        for tid in corner_ids:
            devs_mm.append(float(np.linalg.norm(corners_i[int(tid)] - fused_corners[int(tid)]) * 1000.0))
    max_dev = max(devs_mm) if devs_mm else 0.0
    return max_dev, devs_mm


def basis_from_aligned_state(state: Stage2AlignedState) -> WorldFrameBasis:
    return WorldFrameBasis(
        origin_ref=np.asarray(state.origin_tmp_ref, dtype=np.float64),
        x_axis=np.asarray(state.x_axis, dtype=np.float64),
        y_axis=np.asarray(state.y_axis, dtype=np.float64),
        z_axis=np.asarray(state.z_axis, dtype=np.float64),
    )


def _solve_robot_geometry(
    robot_sess: RecordingSession,
    board_geom_ee: BoardGeometry,
    intrinsics: dict[str, Intrinsics],
    stage1: ExtrinsicsSet,
    *,
    min_tags: int,
    robot_cfg: RobotConfig,
) -> tuple[PlaneFitResult, np.ndarray, np.ndarray, np.ndarray, np.ndarray, WorldFrameBasis]:
    """Replace the old floor-board SVD. Return the same 6-tuple as ``_fit_floor``."""
    solve = solve_robot_world(
        robot_sess.samples,
        board_geom_ee,
        intrinsics,
        stage1,
        min_tags=min_tags,
        robot_cfg=robot_cfg,
    )
    x_axis, y_axis, z_axis = world_axes_from_railbase(solve.T_ref_railbase)
    T_ref_bl0 = solve.T_ref_railbase @ T_railbase_baselink(
        0.0, robot_cfg.rail_y_origin_in_railbase_m
    )
    h = float(robot_cfg.base_link_height_above_floor_m)
    origin_tmp = T_ref_bl0[:3, 3] - h * z_axis
    floor_d = float(origin_tmp @ z_axis)
    floor_fit = PlaneFitResult(
        normal=z_axis.copy(),
        d=floor_d,
        residual_mm=0.0,
        n_points=0,
    )
    basis_tmp = WorldFrameBasis(origin_ref=origin_tmp, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis)
    # Stash the solve on the fit object via a private attr the caller reads.
    floor_fit._robot_solve = solve  # type: ignore[attr-defined]
    return floor_fit, x_axis, y_axis, z_axis, origin_tmp, basis_tmp


def _persist_robot_world(
    *,
    T_world_ref: np.ndarray,
    state: Stage2AlignedState,
    robot_cfg: RobotConfig,
) -> None:
    if not state.T_ref_railbase:
        return
    T_ref_railbase = np.asarray(state.T_ref_railbase, dtype=np.float64).reshape(4, 4)
    T_tcp_board = (
        np.asarray(state.T_tcp_board, dtype=np.float64).reshape(4, 4)
        if state.T_tcp_board
        else np.eye(4)
    )
    T_world_railbase = T_world_ref @ T_ref_railbase
    diag = dict(state.robot_diagnostics or {})
    save_robot_world(
        build_robot_world_export(
            T_world_railbase=T_world_railbase,
            T_ref_railbase=T_ref_railbase,
            T_tcp_board=T_tcp_board,
            robot_cfg=robot_cfg,
            diagnostics=diag,
        )
    )
    if diag.get("joint_zero_offsets_deg") is not None:
        from multicam_calib.io.results import (
            build_joint_zero_offsets_payload,
            save_joint_zero_offsets,
        )

        save_joint_zero_offsets(build_joint_zero_offsets_payload(diag))


def _fit_bed_height(
    bed_sess: RecordingSession,
    board_geom: BoardGeometry,
    intrinsics: dict[str, Intrinsics],
    stage1: ExtrinsicsSet,
    floor_fit: PlaneFitResult,
    *,
    min_tags: int,
    board_thickness_m: float = 0.0,
) -> tuple[float, float]:
    bed_pts, _ = _collect_ref_points(bed_sess, board_geom, intrinsics, stage1, min_tags=min_tags)
    if bed_pts.shape[0] < 12:
        raise RuntimeError("Too few bed points.")
    bed_heights = signed_heights_along_normal(bed_pts, floor_fit.normal, floor_fit.d)
    z_tag = float(np.median(bed_heights))
    z_bed = z_tag - float(board_thickness_m)
    bed_residual_mm = float(np.sqrt(np.mean((bed_heights - z_tag) ** 2)) * 1000.0)
    return z_bed, bed_residual_mm


def _bed_height_extra(world_cfg: WorldConfig, z_bed: float) -> dict[str, float]:
    t = float(world_cfg.board_thickness_m)
    return {
        "board_thickness_m": t,
        "bed_height_tag_plane_m": float(z_bed) + t,
    }


def _union_corner_tags_across_views(sample: Sample, corner_ids: list[int]) -> set[int]:
    found: set[int] = set()
    for det in sample.views.values():
        for tid in corner_ids:
            if int(tid) in det.tags:
                found.add(int(tid))
    return found


def validate_corner_capture(
    *,
    views: dict[str, ViewDetections],
    board_geom: BoardGeometry,
    intrinsics: dict[str, Intrinsics],
    stage1: ExtrinsicsSet,
    world_cfg: WorldConfig,
    app_cfg: AppConfig,
    basis: WorldFrameBasis | None = None,
) -> CornerCapturePreview:
    """Gate corners capture: all corner IDs + fused board pose + pose consistency."""
    corner_ids = world_cfg.corner_tags.all_ids()
    min_tags = int(world_cfg.min_tags_corner_view)

    tmp = Sample(index=0, host_timestamp_ns=0, views=views)

    found = _union_corner_tags_across_views(tmp, corner_ids)
    missing = [tid for tid in corner_ids if tid not in found]
    if world_cfg.require_all_corner_tags and missing:
        return CornerCapturePreview(
            ok=False,
            message=f"Missing corner tag IDs: {missing}",
            missing_tag_ids=missing,
        )

    cam_poses = _camera_board_poses_for_sample(
        tmp, board_geom, intrinsics, stage1, min_tags=min_tags
    )
    T_fused, n_cams = _fused_board_pose(cam_poses)
    if T_fused is None or n_cams < int(world_cfg.min_cameras_corner_fusion):
        return CornerCapturePreview(
            ok=False,
            message=(
                f"Only {n_cams} camera(s) qualify for board pose "
                f"(need >= {world_cfg.min_cameras_corner_fusion}, "
                f"each with >= {min_tags} tags)."
            ),
            n_qualifying_cameras=n_cams,
            missing_tag_ids=missing,
        )

    max_dev, devs_mm = _corner_pose_deviation_mm(cam_poses, T_fused, board_geom, corner_ids)
    if max_dev > float(world_cfg.corner_fusion_max_std_mm):
        return CornerCapturePreview(
            ok=False,
            message=(
                f"Corner pose deviation {max_dev:.1f} mm > "
                f"{world_cfg.corner_fusion_max_std_mm} mm "
                f"({n_cams} cameras; per-camera board pose disagrees with fused pose)"
            ),
            n_qualifying_cameras=n_cams,
            fusion_std_mm=devs_mm,
        )

    rect_xy = None
    if basis is not None:
        fused_corners = _transform_board_points(T_fused, board_geom, corner_ids)
        xy = np.stack([basis.ref_to_world(p)[:2] for p in fused_corners.values()], axis=0)
        r = axis_aligned_rect_from_xy(xy)
        rect_xy = {"x_min": r.x_min, "x_max": r.x_max, "y_min": r.y_min, "y_max": r.y_max}

    return CornerCapturePreview(
        ok=True,
        message=f"OK — {n_cams} cameras, fused board pose (max dev {max_dev:.1f} mm)",
        n_qualifying_cameras=n_cams,
        fusion_std_mm=devs_mm,
        rect_xy=rect_xy,
    )


def _validated_corner_pose(
    sample: Sample,
    board_geom: BoardGeometry,
    intrinsics: dict[str, Intrinsics],
    stage1: ExtrinsicsSet,
    world_cfg: WorldConfig,
    app_cfg: AppConfig,
    basis: WorldFrameBasis,
) -> tuple[np.ndarray, list[float]]:
    """Validate one corners-phase sample and return its fused board pose in ref frame."""
    preview = validate_corner_capture(
        views=sample.views,
        board_geom=board_geom,
        intrinsics=intrinsics,
        stage1=stage1,
        world_cfg=world_cfg,
        app_cfg=app_cfg,
        basis=basis,
    )
    if not preview.ok:
        raise RuntimeError(preview.message)

    min_tags = int(world_cfg.min_tags_corner_view)
    cam_poses = _camera_board_poses_for_sample(
        sample, board_geom, intrinsics, stage1, min_tags=min_tags
    )
    T_fused, _ = _fused_board_pose(cam_poses)
    if T_fused is None:
        raise RuntimeError("No fused board pose for corner sample.")
    return T_fused, list(preview.fusion_std_mm or [])


def _corner_tag_points_world(
    T_fused: np.ndarray,
    board_geom: BoardGeometry,
    corner_ids: list[int],
    basis: WorldFrameBasis,
) -> np.ndarray:
    """All physical corner points (4 tags x 4 corners) of one board capture, world XY.

    Uses each tag's actual 4 physical corners (not just its center), so the
    board's outward-facing edge — the true bed-corner contact point — is
    included in the point cloud used for the bed's minimum-area rectangle.
    """
    pts: list[np.ndarray] = []
    for tid in corner_ids:
        model_corners = board_geom.corners_by_tag[int(tid)]
        for k in range(4):
            ph = np.ones(4, dtype=np.float64)
            ph[:3] = model_corners[k]
            p_ref = (T_fused @ ph)[:3]
            pts.append(basis.ref_to_world(p_ref)[:2])
    return np.asarray(pts, dtype=np.float64)


def _fuse_corner_rect_world(
    sample: Sample,
    board_geom: BoardGeometry,
    intrinsics: dict[str, Intrinsics],
    stage1: ExtrinsicsSet,
    world_cfg: WorldConfig,
    app_cfg: AppConfig,
    basis: WorldFrameBasis,
) -> tuple[AxisAlignedRect, list[float]]:
    """Per-capture board footprint — diagnostic only, NOT used for bed size."""
    T_fused, stds = _validated_corner_pose(
        sample, board_geom, intrinsics, stage1, world_cfg, app_cfg, basis
    )
    corner_ids = world_cfg.corner_tags.all_ids()
    fused_corners = _transform_board_points(T_fused, board_geom, corner_ids)
    xy = np.stack([basis.ref_to_world(p)[:2] for p in fused_corners.values()], axis=0)
    return axis_aligned_rect_from_xy(xy), stds


def run_stage2_phase(
    *,
    bundle: Stage2SessionBundle,
    phase: Stage2Phase,
    board_geom: BoardGeometry,
    intrinsics: dict[str, Intrinsics],
    stage1: ExtrinsicsSet,
    app_cfg: AppConfig,
    world_cfg: WorldConfig | None = None,
    save_path: Path | None = None,
    board_geom_ee: BoardGeometry | None = None,
    robot_cfg: RobotConfig | None = None,
) -> Stage2Report:
    """Run one Stage 2 phase: robot, bed, or corners (full world export)."""
    world_cfg = world_cfg or load_world()
    robot_cfg = robot_cfg or load_robot()
    min_tags = int(app_cfg.calibration.min_tags_per_view)
    ref = stage1.reference
    state = bundle.load_aligned_state()
    if phase in ("bed", "corners"):
        bundle.inherit_prereq_alignment_from_last(phase)
        state = bundle.load_aligned_state()

    if phase == "robot":
        robot_sess = bundle.as_legacy_session("robot")
        if len(robot_sess.samples) < int(world_cfg.min_robot_samples):
            raise ValueError(f"Need >= {world_cfg.min_robot_samples} robot samples.")
        robot_issues = audit_robot_samples_for_run(bundle, world_cfg)
        _raise_if_phase_sample_issues("robot", robot_issues)
        ee_geom = board_geom_ee or build_board_geometry(load_board_ee())
        floor_fit, x_axis, y_axis, z_axis, origin_tmp, basis_tmp = _solve_robot_geometry(
            robot_sess,
            ee_geom,
            intrinsics,
            stage1,
            min_tags=int(world_cfg.min_tags_robot_view),
            robot_cfg=robot_cfg,
        )
        solve = getattr(floor_fit, "_robot_solve", None)
        state.floor_aligned = True
        state.bed_aligned = False
        state.corners_aligned = False
        state.floor_plane_residual_mm = floor_fit.residual_mm
        state.floor_normal = floor_fit.normal.tolist()
        state.floor_d = float(floor_fit.d)
        state.x_axis = x_axis.tolist()
        state.y_axis = y_axis.tolist()
        state.z_axis = z_axis.tolist()
        state.origin_tmp_ref = origin_tmp.tolist()
        state.bed_height_m = None
        state.bed_plane_residual_mm = None
        if solve is not None:
            state.T_ref_railbase = solve.T_ref_railbase.tolist()
            state.T_tcp_board = solve.T_tcp_board.tolist()
            state.rail_direction_ref = solve.T_ref_railbase[:3, 1].tolist()
            state.baselink_z_tilt_from_world_z_deg = float(
                (solve.diagnostics or {}).get("baselink_z_tilt_from_world_z_deg") or 0.0
            )
            state.robot_diagnostics = dict(solve.diagnostics or {})
        bundle.save_aligned_state(state)
        _persist_robot_world(
            T_world_ref=basis_tmp.T_world_ref(),
            state=state,
            robot_cfg=robot_cfg,
        )

        meta = WorldMeta(
            origin_mode=world_cfg.origin_mode,
            floor_plane_residual_mm=floor_fit.residual_mm,
            bed_height_m=0.0,
            bed_plane_residual_mm=0.0,
            bed_size_m=(0.0, 0.0),
            bed_center_world=[0.0, 0.0, 0.0],
            bed_center_on_floor=[0.0, 0.0, 0.0],
            corner_rects_xy=[],
            bed_outer_rect_xy=[],
            bed_rotation_deg=0.0,
            corner_fusion_std_mm=[],
            phases_completed=["robot"],
            xy_reference=str(world_cfg.xy_reference),
        )
        save_world_meta(meta)

        return Stage2Report(
            reference=ref,
            phase="robot",
            T_ref_world=None,
            world_poses=None,
            world_meta=meta,
            floor_residual_mm=floor_fit.residual_mm,
            bed_height_m=None,
            bed_residual_mm=None,
            fusion_residual_m=0.0,
            robot_diagnostics=state.robot_diagnostics,
        )

    if phase == "bed":
        if not state.floor_aligned:
            raise ValueError("Run robot alignment first.")
        bed_sess = bundle.as_legacy_session("bed")
        if len(bed_sess.samples) < int(world_cfg.min_bed_samples):
            raise ValueError(f"Need >= {world_cfg.min_bed_samples} bed samples.")
        bed_issues = audit_bed_samples_for_run(
            bundle, board_geom, intrinsics, stage1, app_cfg, world_cfg, state
        )
        _raise_if_phase_sample_issues("bed", bed_issues)
        floor_fit = PlaneFitResult(
            normal=np.asarray(state.floor_normal, dtype=np.float64),
            d=float(state.floor_d),
            residual_mm=float(state.floor_plane_residual_mm),
            n_points=0,
        )
        z_bed, bed_residual_mm = _fit_bed_height(
            bed_sess,
            board_geom,
            intrinsics,
            stage1,
            floor_fit,
            min_tags=min_tags,
            board_thickness_m=float(world_cfg.board_thickness_m),
        )
        state.bed_aligned = True
        state.corners_aligned = False
        state.bed_height_m = z_bed
        state.bed_plane_residual_mm = bed_residual_mm
        bundle.save_aligned_state(state)

        meta = WorldMeta(
            origin_mode=world_cfg.origin_mode,
            floor_plane_residual_mm=state.floor_plane_residual_mm,
            bed_height_m=z_bed,
            bed_plane_residual_mm=bed_residual_mm,
            bed_size_m=(0.0, 0.0),
            bed_center_world=[0.0, 0.0, z_bed],
            bed_center_on_floor=[0.0, 0.0, 0.0],
            corner_rects_xy=[],
            bed_outer_rect_xy=[],
            bed_rotation_deg=0.0,
            corner_fusion_std_mm=[],
            phases_completed=["robot", "bed"],
            xy_reference=str(world_cfg.xy_reference),
            extra=_bed_height_extra(world_cfg, z_bed),
        )
        save_world_meta(meta)

        return Stage2Report(
            reference=ref,
            phase="bed",
            T_ref_world=None,
            world_poses=None,
            world_meta=meta,
            floor_residual_mm=state.floor_plane_residual_mm,
            bed_height_m=z_bed,
            bed_residual_mm=bed_residual_mm,
            fusion_residual_m=0.0,
        )

    if phase == "corners":
        if not state.floor_aligned:
            raise ValueError("Run robot alignment first.")
        if not state.bed_aligned:
            raise ValueError("Run bed height alignment first.")
        return _run_corners_export(
            bundle=bundle,
            board_geom=board_geom,
            intrinsics=intrinsics,
            stage1=stage1,
            app_cfg=app_cfg,
            world_cfg=world_cfg,
            state=state,
            save_path=save_path,
        )

    raise ValueError(f"Unknown phase: {phase}")


def _run_corners_export(
    *,
    bundle: Stage2SessionBundle,
    board_geom: BoardGeometry,
    intrinsics: dict[str, Intrinsics],
    stage1: ExtrinsicsSet,
    app_cfg: AppConfig,
    world_cfg: WorldConfig,
    state: Stage2AlignedState,
    save_path: Path | None,
) -> Stage2Report:
    min_tags = int(app_cfg.calibration.min_tags_per_view)
    corners_sess = bundle.as_legacy_session("corners")
    if len(corners_sess.samples) < int(world_cfg.min_corner_samples):
        raise ValueError(f"Need >= {world_cfg.min_corner_samples} corner samples.")

    basis_tmp = basis_from_aligned_state(state)
    z_bed = float(state.bed_height_m or 0.0)
    bed_residual_mm = float(state.bed_plane_residual_mm or 0.0)
    floor_fit = PlaneFitResult(
        normal=np.asarray(state.floor_normal, dtype=np.float64),
        d=float(state.floor_d),
        residual_mm=float(state.floor_plane_residual_mm),
        n_points=0,
    )

    corner_ids = world_cfg.corner_tags.all_ids()
    all_std: list[float] = []
    corner_rect_dicts: list[dict[str, float]] = []
    all_corner_pts: list[np.ndarray] = []
    for sample in corners_sess.samples:
        T_fused, stds = _validated_corner_pose(
            sample, board_geom, intrinsics, stage1, world_cfg, app_cfg, basis_tmp
        )
        all_std.extend(stds)
        all_corner_pts.append(
            _corner_tag_points_world(T_fused, board_geom, corner_ids, basis_tmp)
        )
        fused_corners = _transform_board_points(T_fused, board_geom, corner_ids)
        xy_centers = np.stack(
            [basis_tmp.ref_to_world(p)[:2] for p in fused_corners.values()], axis=0
        )
        r = axis_aligned_rect_from_xy(xy_centers)
        corner_rect_dicts.append(
            {"x_min": r.x_min, "x_max": r.x_max, "y_min": r.y_min, "y_max": r.y_max}
        )

    # Minimum-area rectangle over every physical tag-corner point from every
    # capture — allows the bed to be at any rotation relative to world X/Y
    # (an axis-aligned union would overestimate the size of a rotated bed).
    all_corner_xy = np.concatenate(all_corner_pts, axis=0)
    bed_rect: RotatedRect = min_area_rect_from_xy(all_corner_xy)
    cx, cy = bed_rect.center_xy
    bed_center_on_floor_tmp = np.array([cx, cy, 0.0], dtype=np.float64)
    bed_center_world_tmp = np.array([cx, cy, z_bed], dtype=np.float64)
    bed_skew_deg_pre_align = float(bed_rect.angle_deg)

    if world_cfg.origin_mode == "bed_center_projected_to_floor":
        origin_ref = basis_tmp.world_to_ref(bed_center_on_floor_tmp)
    else:
        origin_ref = basis_tmp.world_to_ref(bed_center_world_tmp)

    basis = WorldFrameBasis(
        origin_ref=origin_ref,
        x_axis=np.asarray(basis_tmp.x_axis, dtype=np.float64),
        y_axis=np.asarray(basis_tmp.y_axis, dtype=np.float64),
        z_axis=np.asarray(basis_tmp.z_axis, dtype=np.float64),
    )

    if world_cfg.align_xy_to_bed:
        basis = rotate_basis_about_z(basis, bed_skew_deg_pre_align)
        corner_xy_final = transform_xy_between_bases(all_corner_xy, basis_tmp, basis)
        aabb = axis_aligned_rect_from_xy(corner_xy_final)
        bed_size_m = (aabb.width, aabb.height)
        bed_rotation_deg = 0.0
        bed_center_on_floor = [0.0, 0.0, 0.0]
        bed_center_world = [0.0, 0.0, z_bed]
        bed_outer_rect_xy = [{"x": float(p[0]), "y": float(p[1])} for p in rect_corners_xy(aabb)]
        xy_aligned_to_bed = True
    else:
        # Origin already moved to the bed-center floor projection, so the
        # center in the *final* world frame is the origin (not the pre-shift
        # tmp-frame coordinates).
        bed_size_m = (float(bed_rect.size[0]), float(bed_rect.size[1]))
        bed_rotation_deg = bed_skew_deg_pre_align
        bed_center_on_floor = [0.0, 0.0, 0.0]
        bed_center_world = [0.0, 0.0, z_bed]
        corners_final = np.asarray(bed_rect.corners_xy, dtype=np.float64) - np.array(
            [cx, cy], dtype=np.float64
        )
        bed_outer_rect_xy = [
            {"x": float(p[0]), "y": float(p[1])} for p in corners_final
        ]
        xy_aligned_to_bed = False

    T_ref_world = basis.T_ref_world()
    T_world_ref = basis.T_world_ref()

    ref = stage1.reference
    world_poses: dict[str, np.ndarray] = {}
    for alias, T_ref_cam in stage1.poses.items():
        world_poses[alias] = T_world_ref @ T_ref_cam

    fusion_residual_m = 0.0

    meta = WorldMeta(
        origin_mode=world_cfg.origin_mode,
        floor_plane_residual_mm=floor_fit.residual_mm,
        bed_height_m=z_bed,
        bed_plane_residual_mm=bed_residual_mm,
        bed_size_m=bed_size_m,
        bed_center_world=bed_center_world,
        bed_center_on_floor=bed_center_on_floor,
        corner_rects_xy=corner_rect_dicts,
        bed_outer_rect_xy=bed_outer_rect_xy,
        bed_rotation_deg=bed_rotation_deg,
        bed_xy_skew_deg_pre_align=bed_skew_deg_pre_align,
        xy_aligned_to_bed=xy_aligned_to_bed,
        xy_reference="bed" if xy_aligned_to_bed else str(world_cfg.xy_reference or "rail"),
        corner_fusion_std_mm=all_std,
        phases_completed=["robot", "bed", "corners"],
        extra=_bed_height_extra(world_cfg, z_bed),
    )

    ext = ExtrinsicsSet(
        reference="world",
        poses=world_poses,
        metadata={
            "stage": "world",
            "based_on_stage1_reference": ref,
            "floor_plane_residual_mm": floor_fit.residual_mm,
            "bed_height_m": z_bed,
            "fusion_residual_m": fusion_residual_m,
        },
    )
    out = save_path or extrinsics_world_path()
    save_extrinsics(ext, out)
    save_world_meta(meta)
    save_genesis_bundle(
        bundle=build_genesis_bundle(
            intrinsics=intrinsics,
            extrinsics_rel=stage1,
            extrinsics_world=ext,
            world_meta=meta,
        )
    )
    _persist_robot_world(
        T_world_ref=T_world_ref,
        state=state,
        robot_cfg=load_robot(),
    )

    state.corners_aligned = True
    bundle.save_aligned_state(state)
    bundle.write_manifest()

    return Stage2Report(
        reference=ref,
        phase="corners",
        T_ref_world=T_ref_world,
        world_poses=world_poses,
        world_meta=meta,
        floor_residual_mm=floor_fit.residual_mm,
        bed_height_m=z_bed,
        bed_residual_mm=bed_residual_mm,
        fusion_residual_m=fusion_residual_m,
    )


def run_stage2(
    *,
    bundle: Stage2SessionBundle,
    board_geom: BoardGeometry,
    intrinsics: dict[str, Intrinsics],
    stage1: ExtrinsicsSet,
    app_cfg: AppConfig,
    world_cfg: WorldConfig | None = None,
    save_path: Path | None = None,
) -> Stage2Report:
    """Run full robot → bed → corners pipeline (legacy all-at-once entry)."""
    world_cfg = world_cfg or load_world()
    robot_cfg = load_robot()
    min_tags = int(app_cfg.calibration.min_tags_per_view)

    robot_sess = bundle.as_legacy_session("robot")
    bed_sess = bundle.as_legacy_session("bed")
    corners_sess = bundle.as_legacy_session("corners")

    if len(robot_sess.samples) < int(world_cfg.min_robot_samples):
        raise ValueError(f"Need >= {world_cfg.min_robot_samples} robot samples.")
    if len(bed_sess.samples) < int(world_cfg.min_bed_samples):
        raise ValueError(f"Need >= {world_cfg.min_bed_samples} bed samples.")
    if len(corners_sess.samples) < int(world_cfg.min_corner_samples):
        raise ValueError(f"Need >= {world_cfg.min_corner_samples} corner samples.")

    ee_geom = build_board_geometry(load_board_ee())
    floor_fit, x_axis, y_axis, z_axis, origin_tmp, _ = _solve_robot_geometry(
        robot_sess,
        ee_geom,
        intrinsics,
        stage1,
        min_tags=int(world_cfg.min_tags_robot_view),
        robot_cfg=robot_cfg,
    )
    solve = getattr(floor_fit, "_robot_solve", None)
    z_bed, bed_residual_mm = _fit_bed_height(
        bed_sess,
        board_geom,
        intrinsics,
        stage1,
        floor_fit,
        min_tags=min_tags,
        board_thickness_m=float(world_cfg.board_thickness_m),
    )

    state = Stage2AlignedState(
        floor_aligned=True,
        bed_aligned=True,
        floor_plane_residual_mm=floor_fit.residual_mm,
        floor_normal=floor_fit.normal.tolist(),
        floor_d=float(floor_fit.d),
        x_axis=x_axis.tolist(),
        y_axis=y_axis.tolist(),
        z_axis=z_axis.tolist(),
        origin_tmp_ref=origin_tmp.tolist(),
        bed_height_m=z_bed,
        bed_plane_residual_mm=bed_residual_mm,
        T_ref_railbase=None if solve is None else solve.T_ref_railbase.tolist(),
        T_tcp_board=None if solve is None else solve.T_tcp_board.tolist(),
        rail_direction_ref=None if solve is None else solve.T_ref_railbase[:3, 1].tolist(),
        robot_diagnostics={} if solve is None else dict(solve.diagnostics or {}),
    )
    bundle.save_aligned_state(state)

    return _run_corners_export(
        bundle=bundle,
        board_geom=board_geom,
        intrinsics=intrinsics,
        stage1=stage1,
        app_cfg=app_cfg,
        world_cfg=world_cfg,
        state=state,
        save_path=save_path,
    )
