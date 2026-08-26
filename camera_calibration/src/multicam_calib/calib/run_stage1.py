"""Stage 1 pipeline: relative extrinsics between multiple cameras.

Given a `RecordingSession` with N samples (each containing tag detections from
every camera looking at the same physically-static board), compute:

- Per-view T_cam_board via full-board PnP (calib.pnp.solve_view_pose)
- Initial T_ref_cam for every non-reference camera (calib.pose_graph)
- Joint refinement of all camera & board poses (calib.bundle_adjust)
- Persist result to calibration_results/extrinsics_rel.yaml
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from multicam_calib.board.apriltag_board import BoardGeometry
from multicam_calib.board.detector import corners_to_board_frame
from multicam_calib.calib.bundle_adjust import BAObservation, BAProblem, BAResult, solve_bundle_adjustment
from multicam_calib.calib.pnp import solve_view_pose
from multicam_calib.calib.pose_graph import initialize_camera_poses
from multicam_calib.io.config import AppConfig
from multicam_calib.io.results import ExtrinsicsSet, Intrinsics, extrinsics_rel_path, save_extrinsics
from multicam_calib.recording.session import RecordingSession


@dataclass
class Stage1Report:
    reference: str
    aliases: list[str]
    per_view_pnp_rmse: dict[int, dict[str, float]]  # frame_id -> alias -> RMSE (px)
    initial_cam_poses: dict[str, np.ndarray]
    ba: BAResult
    rejected_views: list[tuple[int, str, str]] = field(default_factory=list)  # (frame, alias, reason)
    board_disagreement_mm: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Stage 1 calibration report",
            f"  reference camera: {self.reference}",
            f"  cameras:          {', '.join(self.aliases)}",
            f"  observations:     {self.ba.n_observations}",
            f"  total RMSE:       {self.ba.total_rmse:.4f} px",
        ]
        mm = getattr(self, "board_disagreement_mm", None)
        if isinstance(mm, dict) and mm.get("mean_mm") is not None:
            lines.append(
                f"  board pose disagreement: {float(mm['mean_mm']):.2f} mm mean "
                f"(depth {float(mm.get('mean_depth_mm', float('nan'))):.2f} mm, "
                f"max {float(mm.get('max_mm', float('nan'))):.2f} mm)"
            )
        lines.append("  per-camera RMSE:")
        for a, r in self.ba.per_camera_rmse.items():
            lines.append(f"    {a:<8} {r:.4f} px")
        lines.append(f"  per-frame RMSE:")
        for f, r in sorted(self.ba.per_frame_rmse.items()):
            lines.append(f"    frame {f:03d}: {r:.4f} px")
        if self.rejected_views:
            lines.append(f"  rejected views ({len(self.rejected_views)}):")
            for fr, al, why in self.rejected_views[:10]:
                lines.append(f"    frame {fr} {al}: {why}")
        return "\n".join(lines)


def _stage1_board_disagreement_mm(
    per_view_poses: dict[int, dict[str, np.ndarray]],
    cam_poses: dict[str, np.ndarray],
) -> dict[str, float]:
    """Same-sample inter-camera board translation disagreement (PnP in ref)."""
    errs: list[float] = []
    depth: list[float] = []
    for views in per_view_poses.values():
        poses: list[tuple[np.ndarray, np.ndarray]] = []
        for alias, T_cam_board in views.items():
            T_ref_cam = cam_poses.get(alias)
            if T_ref_cam is None:
                continue
            T_ref_cam = np.asarray(T_ref_cam, dtype=np.float64)
            T_ref_board = T_ref_cam @ np.asarray(T_cam_board, dtype=np.float64)
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


def run_stage1(
    *,
    session: RecordingSession,
    board_geom: BoardGeometry,
    intrinsics: dict[str, Intrinsics],
    reference: str | None,
    app_cfg: AppConfig,
    save_path: Path | None = None,
) -> Stage1Report:
    """Compute relative extrinsics from a completed recording session."""
    aliases_in_session = list(session.aliases)
    if not aliases_in_session:
        raise ValueError("Session has no cameras registered.")
    ref = reference or aliases_in_session[0]
    if ref not in aliases_in_session:
        raise ValueError(f"Reference {ref!r} not in session aliases {aliases_in_session}")

    aliases = [ref] + [a for a in aliases_in_session if a != ref]

    min_tags = int(app_cfg.calibration.min_tags_per_view)
    min_frames = int(app_cfg.calibration.min_frames)

    if len(session.samples) < min_frames:
        raise ValueError(
            f"Need at least {min_frames} samples (have {len(session.samples)}). "
            "Capture more before running."
        )

    per_view_poses: dict[int, dict[str, np.ndarray]] = {}
    per_view_rmse: dict[int, dict[str, float]] = {}
    rejected: list[tuple[int, str, str]] = []

    observations: list[BAObservation] = []
    for sample in session.samples:
        frame_id = sample.index
        views: dict[str, np.ndarray] = {}
        rmse_map: dict[str, float] = {}
        for alias in aliases:
            det = sample.views.get(alias)
            if det is None or det.num_tags() < min_tags:
                rejected.append((frame_id, alias, f"too few tags ({0 if det is None else det.num_tags()} < {min_tags})"))
                continue
            intr = intrinsics.get(alias)
            if intr is None:
                rejected.append((frame_id, alias, "no intrinsics"))
                continue
            pose = solve_view_pose(board_geom, det.tags, intr, min_tags=min_tags)
            if pose is None:
                rejected.append((frame_id, alias, "solvePnP failed"))
                continue
            views[alias] = pose.T_cam_board
            rmse_map[alias] = pose.reprojection_rmse_px
            for tag_id, corners_px in det.tags.items():
                if tag_id not in board_geom.corners_by_tag:
                    continue
                corners_board = corners_to_board_frame(corners_px)
                for k in range(4):
                    observations.append(
                        BAObservation(
                            frame_id=frame_id,
                            alias=alias,
                            tag_id=int(tag_id),
                            corner_idx=int(k),
                            u=float(corners_board[k, 0]),
                            v=float(corners_board[k, 1]),
                        )
                    )
        if views:
            per_view_poses[frame_id] = views
            per_view_rmse[frame_id] = rmse_map

    if len(per_view_poses) < min_frames:
        raise RuntimeError(
            f"Only {len(per_view_poses)} frames survived filtering (need >= {min_frames}). "
            "Take more samples or lower min_tags_per_view."
        )
    # Convert per-view T_cam_board into T_ref_camera space for the initializer:
    # calib.pose_graph.initialize_camera_poses wants {frame: {alias: T_cam_board}}.
    init = initialize_camera_poses(per_view_poses, reference=ref)

    # Board pose initial guess (in ref frame) is `T_ref_cam · T_cam_board` for any cam that
    # saw the board in that frame — pick the one with the largest number of tags.
    initial_board_poses: dict[int, np.ndarray] = {}
    for frame_id, views in per_view_poses.items():
        # Prefer the reference camera when it saw the board; otherwise any alias.
        if ref in views:
            initial_board_poses[frame_id] = init.poses[ref] @ views[ref]
            continue
        alias = max(views.keys(), key=lambda a: 0)  # any alias
        initial_board_poses[frame_id] = init.poses[alias] @ views[alias]

    problem = BAProblem(
        aliases=aliases,
        intrinsics={a: intrinsics[a] for a in aliases},
        initial_cam_poses=init.poses,
        initial_board_poses=initial_board_poses,
        corners_by_tag=board_geom.corners_by_tag,
        observations=observations,
    )

    ba = solve_bundle_adjustment(
        problem,
        loss=app_cfg.calibration.ba.loss,
        f_scale=app_cfg.calibration.ba.f_scale,
        max_nfev=app_cfg.calibration.ba.max_nfev,
        verbose=app_cfg.calibration.ba.verbose,
    )
    board_mm = _stage1_board_disagreement_mm(per_view_poses, ba.cam_poses)

    ext = ExtrinsicsSet(
        reference=ref,
        poses=ba.cam_poses,
        metadata={
            "stage": "relative",
            "n_frames": len(per_view_poses),
            "n_observations": ba.n_observations,
            "total_rmse_px": ba.total_rmse,
            "per_camera_rmse_px": ba.per_camera_rmse,
            "board_pose_disagreement_mean_mm": board_mm["mean_mm"],
            "board_pose_disagreement_median_mm": board_mm["median_mm"],
            "board_pose_disagreement_max_mm": board_mm["max_mm"],
            "board_pose_disagreement_mean_depth_mm": board_mm["mean_depth_mm"],
            "board_pose_disagreement_n_pairs": board_mm["n_pairs"],
        },
    )
    save_extrinsics(ext, save_path or extrinsics_rel_path())

    return Stage1Report(
        reference=ref,
        aliases=aliases,
        per_view_pnp_rmse=per_view_rmse,
        initial_cam_poses=init.poses,
        ba=ba,
        rejected_views=rejected,
        board_disagreement_mm=board_mm,
    )
