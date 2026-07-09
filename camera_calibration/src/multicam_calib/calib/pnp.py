"""Per-view board pose estimation via full-corner solvePnP.

We do NOT use the per-tag pose returned by AprilTag. Instead every visible tag
on the board contributes its 4 corners into one solvePnP call, so the pose is
constrained by all detected corners over the full board area (large lever arm)
rather than by a single 4-cm tag. This is what makes multi-camera BA later
converge to sub-pixel reprojection error.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from multicam_calib.board.apriltag_board import BoardGeometry
from multicam_calib.io.results import Intrinsics


@dataclass
class ViewPose:
    """SE(3) transform T_cam_board plus the corners that produced it."""

    T_cam_board: np.ndarray  # (4, 4)
    used_tag_ids: list[int]
    num_corners: int
    reprojection_rmse_px: float


def solve_view_pose(
    board: BoardGeometry,
    detections: dict[int, np.ndarray],
    intrinsics: Intrinsics,
    *,
    min_tags: int = 4,
) -> ViewPose | None:
    """Estimate board pose in camera frame from all visible tag corners.

    Returns ``None`` when the view has too few tags to be reliable.
    """
    obj, img, used = board.gather_correspondences(detections)
    if len(used) < min_tags:
        return None
    K = intrinsics.K.astype(np.float64)
    dist = intrinsics.dist.astype(np.float64).reshape(-1, 1)

    # SQPNP (Terzakis & Lourakis, ECCV 2020) is the current best-in-class solver
    # for planar and near-planar targets; it does not suffer from the two-fold
    # ambiguity that trips up SOLVEPNP_ITERATIVE with RANSAC. We follow up with
    # simple reprojection-based outlier filtering and an LM refinement.
    ok, rvec, tvec = cv2.solvePnP(
        objectPoints=obj.reshape(-1, 1, 3),
        imagePoints=img.reshape(-1, 1, 2),
        cameraMatrix=K,
        distCoeffs=dist,
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not ok or rvec is None or tvec is None:
        return None

    proj0, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err_per_point = np.linalg.norm(proj0.reshape(-1, 2) - img, axis=1)
    threshold = max(3.0, 3.0 * float(np.median(err_per_point) + 1e-6))
    inlier_mask = err_per_point < threshold
    if inlier_mask.sum() < max(6, min_tags * 4 // 2):
        # If aggressive filtering rejected too many corners just refine on all.
        inlier_mask[:] = True

    rvec, tvec = cv2.solvePnPRefineLM(
        objectPoints=obj[inlier_mask].reshape(-1, 1, 3),
        imagePoints=img[inlier_mask].reshape(-1, 1, 2),
        cameraMatrix=K,
        distCoeffs=dist,
        rvec=rvec,
        tvec=tvec,
    )

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)

    # Compute inlier reprojection RMSE for reporting.
    proj, _ = cv2.projectPoints(obj[inlier_mask], rvec, tvec, K, dist)
    err = (proj.reshape(-1, 2) - img[inlier_mask]).reshape(-1)
    rmse = float(np.sqrt(np.mean(err * err))) if err.size else float("nan")

    return ViewPose(
        T_cam_board=T,
        used_tag_ids=list(sorted(set(used))),
        num_corners=int(inlier_mask.sum()),
        reprojection_rmse_px=rmse,
    )
