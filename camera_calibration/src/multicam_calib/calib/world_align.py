"""Stage 2: world alignment via floor plane, bed height, and bed-corner envelope.

Pipeline (requires Stage 1 ``extrinsics_rel.yaml``):

1. **Floor** — multi-position captures; SVD plane fit → world +Z and XY axes.
2. **Bed** — multi-position captures; parallel plane height ``z_bed`` above floor.
3. **Corners** — four captures, one board placement per physical bed corner
   (any rotation allowed); fuse four board corner tags (151/1/162/12) per
   sample, pool every physical tag-corner point across all captures, and fit
   the minimum-area bounding rectangle (any orientation, not axis-aligned) →
   bed size; origin at bed-center projected to floor (configurable).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from multicam_calib.board.apriltag_board import BoardGeometry
from multicam_calib.calib.plane_fit import (
    AxisAlignedRect,
    PlaneFitResult,
    RotatedRect,
    WorldFrameBasis,
    axis_aligned_rect_from_xy,
    build_world_basis_from_floor,
    fit_plane_svd,
    min_area_rect_from_xy,
    signed_heights_along_normal,
)
from multicam_calib.calib.pnp import solve_view_pose
from multicam_calib.calib.pose_graph import _average_se3, se3_inv
from multicam_calib.io.config import AppConfig, WorldConfig, load_world
from multicam_calib.io.genesis_export import build_genesis_bundle, save_genesis_bundle
from multicam_calib.io.results import (
    ExtrinsicsSet,
    Intrinsics,
    WorldMeta,
    extrinsics_world_path,
    save_extrinsics,
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


def validate_floor_capture(
    *,
    views: dict[str, ViewDetections],
    host_timestamp_ns: int,
    board_geom: BoardGeometry,
    intrinsics: dict[str, Intrinsics],
    stage1: ExtrinsicsSet,
    app_cfg: AppConfig,
    world_cfg: WorldConfig,
    state: Stage2AlignedState,
    bed_samples: list[Sample],
) -> PhaseCapturePreview:
    """Reject ground captures taken on the bed or after bed phase has started."""
    min_tags = int(app_cfg.calibration.min_tags_per_view)
    tmp = Sample(
        index=-1,
        host_timestamp_ns=int(host_timestamp_ns),
        views=views,
        metadata={"phase": "floor"},
    )
    first_bed_ts = _first_bed_capture_timestamp_ns(bed_samples)
    if first_bed_ts is not None and int(host_timestamp_ns) >= first_bed_ts:
        return PhaseCapturePreview(
            ok=False,
            message=(
                "Ground plane capture rejected: bed phase already has samples. "
                "Do not capture ground data after starting bed — board must be on the floor, "
                "or clear bed samples / start a new floor calibration."
            ),
        )

    T, n_cams = _estimate_T_ref_board_per_sample(
        tmp, board_geom, intrinsics, stage1, min_tags=min_tags
    )
    if T is None or n_cams < 2:
        return PhaseCapturePreview(
            ok=False,
            message=f"Too few qualifying cameras ({n_cams}) for ground plane capture.",
            n_qualifying_cameras=n_cams,
        )

    h_mm: float | None = None
    if state.floor_aligned:
        h_mm = _height_above_floor_plane_mm(
            T[:3, 3],
            np.asarray(state.floor_normal, dtype=np.float64),
            float(state.floor_d),
        )
        if h_mm > float(world_cfg.floor_max_height_above_plane_mm):
            return PhaseCapturePreview(
                ok=False,
                message=(
                    f"Ground plane capture rejected: board is {h_mm:.0f} mm above the floor plane "
                    f"(limit {world_cfg.floor_max_height_above_plane_mm:.0f} mm). "
                    "Board appears to be on the bed — switch Capture mode to Bed plane."
                ),
                n_qualifying_cameras=n_cams,
                height_above_floor_mm=h_mm,
            )
        if state.bed_aligned and state.bed_height_m is not None:
            bed_mm = float(state.bed_height_m) * 1000.0
            if abs(h_mm - bed_mm) < float(world_cfg.bed_height_match_tolerance_mm):
                return PhaseCapturePreview(
                    ok=False,
                    message=(
                        f"Ground plane capture rejected: board height {h_mm:.0f} mm matches "
                        f"bed height {bed_mm:.0f} mm. Use Bed plane mode."
                    ),
                    n_qualifying_cameras=n_cams,
                    height_above_floor_mm=h_mm,
                )

    msg = f"OK — {n_cams} camera(s)"
    if h_mm is not None:
        msg += f", {h_mm:.0f} mm above floor"
    return PhaseCapturePreview(ok=True, message=msg, n_qualifying_cameras=n_cams, height_above_floor_mm=h_mm)


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
            message="Bed plane capture rejected: run ground plane alignment first.",
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


def audit_floor_samples_for_run(
    bundle: Stage2SessionBundle,
    board_geom: BoardGeometry,
    intrinsics: dict[str, Intrinsics],
    stage1: ExtrinsicsSet,
    app_cfg: AppConfig,
    world_cfg: WorldConfig,
    state: Stage2AlignedState,
) -> list[PhaseSampleIssue]:
    """List floor-folder samples that must not be used for ground plane alignment."""
    issues: list[PhaseSampleIssue] = []
    min_tags = int(app_cfg.calibration.min_tags_per_view)
    first_bed_ts = _first_bed_capture_timestamp_ns(bundle.bed.samples)

    for sample in bundle.floor.samples:
        phase_meta = sample.metadata.get("phase")
        if phase_meta not in (None, "floor"):
            issues.append(
                PhaseSampleIssue(
                    sample.index,
                    f"wrong phase metadata {phase_meta!r} (expected floor)",
                )
            )
        if first_bed_ts is not None and sample.host_timestamp_ns >= first_bed_ts:
            issues.append(
                PhaseSampleIssue(
                    sample.index,
                    "captured after bed phase started — likely board on bed, not ground",
                )
            )
        if state.floor_aligned:
            h_mm = _sample_height_above_floor_mm(
                sample, board_geom, intrinsics, stage1, state, min_tags=min_tags
            )
            if h_mm is None:
                issues.append(PhaseSampleIssue(sample.index, "no qualifying camera views"))
                continue
            if h_mm > float(world_cfg.floor_max_height_above_plane_mm):
                issues.append(
                    PhaseSampleIssue(
                        sample.index,
                        f"{h_mm:.0f} mm above floor (limit {world_cfg.floor_max_height_above_plane_mm:.0f} mm)",
                    )
                )
            elif state.bed_aligned and state.bed_height_m is not None:
                bed_mm = float(state.bed_height_m) * 1000.0
                if abs(h_mm - bed_mm) < float(world_cfg.bed_height_match_tolerance_mm):
                    issues.append(
                        PhaseSampleIssue(
                            sample.index,
                            f"height {h_mm:.0f} mm matches bed ({bed_mm:.0f} mm) — not ground",
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
    label = {"floor": "ground plane", "bed": "bed plane", "corners": "corners"}[phase]
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


def _fit_floor(
    floor_sess: RecordingSession,
    board_geom: BoardGeometry,
    intrinsics: dict[str, Intrinsics],
    stage1: ExtrinsicsSet,
    *,
    min_tags: int,
) -> tuple[PlaneFitResult, np.ndarray, np.ndarray, np.ndarray, np.ndarray, WorldFrameBasis]:
    floor_pts, x_axes = _collect_ref_points(
        floor_sess, board_geom, intrinsics, stage1, min_tags=min_tags
    )
    if floor_pts.shape[0] < 12:
        raise RuntimeError("Too few floor points for plane fit.")
    floor_fit = fit_plane_svd(floor_pts)
    x_axis, y_axis, z_axis = build_world_basis_from_floor(floor_fit.normal, x_axes)
    floor_centroid = floor_pts.mean(axis=0)
    h0 = float(floor_centroid @ floor_fit.normal - floor_fit.d)
    origin_tmp = floor_centroid - h0 * floor_fit.normal
    basis_tmp = WorldFrameBasis(origin_ref=origin_tmp, x_axis=x_axis, y_axis=y_axis, z_axis=z_axis)
    return floor_fit, x_axis, y_axis, z_axis, origin_tmp, basis_tmp


def _fit_bed_height(
    bed_sess: RecordingSession,
    board_geom: BoardGeometry,
    intrinsics: dict[str, Intrinsics],
    stage1: ExtrinsicsSet,
    floor_fit: PlaneFitResult,
    *,
    min_tags: int,
) -> tuple[float, float]:
    bed_pts, _ = _collect_ref_points(bed_sess, board_geom, intrinsics, stage1, min_tags=min_tags)
    if bed_pts.shape[0] < 12:
        raise RuntimeError("Too few bed points.")
    bed_heights = signed_heights_along_normal(bed_pts, floor_fit.normal, floor_fit.d)
    z_bed = float(np.median(bed_heights))
    bed_residual_mm = float(np.sqrt(np.mean((bed_heights - z_bed) ** 2)) * 1000.0)
    return z_bed, bed_residual_mm


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
) -> Stage2Report:
    """Run one Stage 2 phase: floor, bed, or corners (full world export)."""
    world_cfg = world_cfg or load_world()
    min_tags = int(app_cfg.calibration.min_tags_per_view)
    ref = stage1.reference
    state = bundle.load_aligned_state()

    if phase == "floor":
        floor_sess = bundle.as_legacy_session("floor")
        if len(floor_sess.samples) < int(world_cfg.min_floor_samples):
            raise ValueError(f"Need >= {world_cfg.min_floor_samples} floor samples.")
        floor_issues = audit_floor_samples_for_run(
            bundle, board_geom, intrinsics, stage1, app_cfg, world_cfg, state
        )
        _raise_if_phase_sample_issues("floor", floor_issues)
        floor_fit, x_axis, y_axis, z_axis, origin_tmp, _ = _fit_floor(
            floor_sess, board_geom, intrinsics, stage1, min_tags=min_tags
        )
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
        bundle.save_aligned_state(state)

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
            phases_completed=["floor"],
        )
        save_world_meta(meta)

        return Stage2Report(
            reference=ref,
            phase="floor",
            T_ref_world=None,
            world_poses=None,
            world_meta=meta,
            floor_residual_mm=floor_fit.residual_mm,
            bed_height_m=None,
            bed_residual_mm=None,
            fusion_residual_m=0.0,
        )

    if phase == "bed":
        if not state.floor_aligned:
            raise ValueError("Run floor alignment first.")
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
            bed_sess, board_geom, intrinsics, stage1, floor_fit, min_tags=min_tags
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
            phases_completed=["floor", "bed"],
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
            raise ValueError("Run floor alignment first.")
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
    bed_rect: RotatedRect = min_area_rect_from_xy(np.concatenate(all_corner_pts, axis=0))
    cx, cy = bed_rect.center_xy
    bed_center_world = np.array([cx, cy, z_bed], dtype=np.float64)
    bed_center_on_floor = np.array([cx, cy, 0.0], dtype=np.float64)

    if world_cfg.origin_mode == "bed_center_projected_to_floor":
        origin_ref = basis_tmp.world_to_ref(bed_center_on_floor)
    else:
        origin_ref = basis_tmp.world_to_ref(bed_center_world)

    basis = WorldFrameBasis(
        origin_ref=origin_ref,
        x_axis=basis_tmp.x_axis,
        y_axis=basis_tmp.y_axis,
        z_axis=basis_tmp.z_axis,
    )
    T_ref_world = basis.T_ref_world()
    T_world_ref = basis.T_world_ref()

    ref = stage1.reference
    world_poses: dict[str, np.ndarray] = {}
    for alias, T_ref_cam in stage1.poses.items():
        world_poses[alias] = T_world_ref @ T_ref_cam

    floor_sess = bundle.as_legacy_session("floor")
    contributions: list[np.ndarray] = []
    for sample in floor_sess.samples:
        T_rb, _ = _estimate_T_ref_board_per_sample(sample, board_geom, intrinsics, stage1, min_tags=min_tags)
        if T_rb is not None:
            contributions.append(T_rb[:3, 3])
    fusion_residual_m = 0.0
    if contributions:
        origins = np.stack(contributions, axis=0)
        fusion_residual_m = float(np.mean(np.linalg.norm(origins - origins.mean(axis=0), axis=1)))

    meta = WorldMeta(
        origin_mode=world_cfg.origin_mode,
        floor_plane_residual_mm=floor_fit.residual_mm,
        bed_height_m=z_bed,
        bed_plane_residual_mm=bed_residual_mm,
        bed_size_m=(bed_rect.size[0], bed_rect.size[1]),
        bed_center_world=bed_center_world.tolist(),
        bed_center_on_floor=bed_center_on_floor.tolist(),
        corner_rects_xy=corner_rect_dicts,
        bed_outer_rect_xy=[
            {"x": float(p[0]), "y": float(p[1])} for p in bed_rect.corners_xy
        ],
        bed_rotation_deg=bed_rect.angle_deg,
        corner_fusion_std_mm=all_std,
        phases_completed=["floor", "bed", "corners"],
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
    """Run full floor → bed → corners pipeline (legacy all-at-once entry)."""
    world_cfg = world_cfg or load_world()
    min_tags = int(app_cfg.calibration.min_tags_per_view)

    floor_sess = bundle.as_legacy_session("floor")
    bed_sess = bundle.as_legacy_session("bed")
    corners_sess = bundle.as_legacy_session("corners")

    if len(floor_sess.samples) < int(world_cfg.min_floor_samples):
        raise ValueError(f"Need >= {world_cfg.min_floor_samples} floor samples.")
    if len(bed_sess.samples) < int(world_cfg.min_bed_samples):
        raise ValueError(f"Need >= {world_cfg.min_bed_samples} bed samples.")
    if len(corners_sess.samples) < int(world_cfg.min_corner_samples):
        raise ValueError(f"Need >= {world_cfg.min_corner_samples} corner samples.")

    floor_fit, x_axis, y_axis, z_axis, origin_tmp, basis_tmp = _fit_floor(
        floor_sess, board_geom, intrinsics, stage1, min_tags=min_tags
    )
    z_bed, bed_residual_mm = _fit_bed_height(
        bed_sess, board_geom, intrinsics, stage1, floor_fit, min_tags=min_tags
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
