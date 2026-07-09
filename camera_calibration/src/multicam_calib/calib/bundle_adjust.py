"""Joint multi-camera bundle adjustment.

Model:
- Cameras :math:`c \in {c_ref, c_1, ..., c_{K-1}}`. ``T_ref_c`` are variables
  (with T_ref_ref = I frozen).
- Frames :math:`f = 0..F-1`. Each has an unknown board pose ``T_ref_board_f``.
- Observations: for every (frame f, cam c, tag t, corner k) with a valid detection,
  the 2D pixel measurement :math:`u_{fctk}` is compared against the projection
  of the board-frame 3D point :math:`X_{tk}` through:

    :math:`p_{fctk} = \pi(K_c \cdot \Pi_{\text{dist}}(\, T_c^{ref} \cdot T_{ref}^{board_f} \cdot X_{tk}\,))`

  where :math:`T_c^{ref} = (T_{ref}^{c})^{-1}` and :math:`\pi` is the perspective
  projection.

We parameterize every SE(3) as (axis-angle, translation) — a 6-vector per
variable — and let ``scipy.optimize.least_squares`` handle the non-linear
optimization with a Cauchy loss for outlier robustness.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares

from multicam_calib.calib.pose_graph import se3_exp, se3_inv, se3_log
from multicam_calib.io.results import Intrinsics


@dataclass
class BAObservation:
    """One 2D corner measurement bound to a (frame, cam, tag, corner index)."""

    frame_id: int
    alias: str
    tag_id: int
    corner_idx: int  # 0..3 within the tag
    u: float
    v: float


@dataclass
class BAProblem:
    """All the data BA needs, precomputed for fast residual evaluation."""

    aliases: list[str]                       # ordered; aliases[0] is the reference
    intrinsics: dict[str, Intrinsics]        # K, dist by alias
    initial_cam_poses: dict[str, np.ndarray]  # T_ref_cam per alias, 4x4
    initial_board_poses: dict[int, np.ndarray]  # T_ref_board per frame_id, 4x4
    corners_by_tag: dict[int, np.ndarray]   # (4, 3) in board frame
    observations: list[BAObservation]


@dataclass
class _BAGroup:
    """Pre-batched observations for one (camera, frame) pair."""

    alias: str
    frame_id: int
    obs_indices: np.ndarray  # (M,) flat indices into the residual vector / 2
    X_board: np.ndarray      # (M, 3)
    uv: np.ndarray           # (M, 2)


@dataclass
class _BAPrepared:
    """Cached grouping and camera intrinsics — built once per solve."""

    groups: list[_BAGroup]
    n_obs: int
    K_by_alias: dict[str, np.ndarray]
    dist_by_alias: dict[str, np.ndarray]
    cam_order: list[str]
    frame_order: list[int]
    ref: str


@dataclass
class BAResult:
    reference: str
    cam_poses: dict[str, np.ndarray]        # T_ref_cam per alias
    board_poses: dict[int, np.ndarray]      # T_ref_board per frame_id
    per_camera_rmse: dict[str, float]
    per_frame_rmse: dict[int, float]
    total_rmse: float
    n_observations: int
    metadata: dict = field(default_factory=dict)


def _pack(problem: BAProblem) -> tuple[np.ndarray, list[str], list[int]]:
    """Pack all variables into one flat parameter vector.

    Layout: [cam(1), cam(2), ..., cam(K-1), board(f0), board(f1), ...]
    Reference camera is not part of the vector (kept fixed at identity).
    """
    ref = problem.aliases[0]
    cam_order = [a for a in problem.aliases if a != ref]
    frame_order = sorted(problem.initial_board_poses.keys())
    parts: list[np.ndarray] = []
    for a in cam_order:
        parts.append(se3_log(problem.initial_cam_poses[a]))
    for f in frame_order:
        parts.append(se3_log(problem.initial_board_poses[f]))
    x0 = np.concatenate(parts)
    return x0, cam_order, frame_order


def _unpack(x: np.ndarray, cam_order: list[str], frame_order: list[int]) -> tuple[dict[str, np.ndarray], dict[int, np.ndarray]]:
    idx = 0
    cams: dict[str, np.ndarray] = {}
    for a in cam_order:
        cams[a] = se3_exp(x[idx : idx + 6])
        idx += 6
    boards: dict[int, np.ndarray] = {}
    for f in frame_order:
        boards[f] = se3_exp(x[idx : idx + 6])
        idx += 6
    return cams, boards


def _prepare(problem: BAProblem, cam_order: list[str], frame_order: list[int]) -> _BAPrepared:
    """Pre-group observations and cache intrinsics for fast residual evaluation."""
    ref = problem.aliases[0]
    corners_by_tag = problem.corners_by_tag
    obs = problem.observations

    by_pair: dict[tuple[str, int], list[tuple[int, np.ndarray, np.ndarray]]] = defaultdict(list)
    for i, o in enumerate(obs):
        model = corners_by_tag.get(o.tag_id)
        if model is None:
            continue
        by_pair[(o.alias, o.frame_id)].append(
            (i, model[o.corner_idx], np.array([o.u, o.v], dtype=np.float64))
        )

    groups: list[_BAGroup] = []
    for (alias, frame_id), items in by_pair.items():
        idxs = np.array([t[0] for t in items], dtype=np.int64)
        X_board = np.stack([t[1] for t in items], axis=0)
        uv = np.stack([t[2] for t in items], axis=0)
        groups.append(_BAGroup(alias=alias, frame_id=frame_id, obs_indices=idxs, X_board=X_board, uv=uv))

    K_by_alias = {a: problem.intrinsics[a].K.astype(np.float64) for a in problem.aliases}
    dist_by_alias = {a: problem.intrinsics[a].dist.astype(np.float64).reshape(-1) for a in problem.aliases}

    return _BAPrepared(
        groups=groups,
        n_obs=len(obs),
        K_by_alias=K_by_alias,
        dist_by_alias=dist_by_alias,
        cam_order=cam_order,
        frame_order=frame_order,
        ref=ref,
    )


def _residuals(x: np.ndarray, prepared: _BAPrepared, cams: dict[str, np.ndarray], boards: dict[int, np.ndarray]) -> np.ndarray:
    """Vectorised residual evaluation using pre-batched (camera, frame) groups."""
    resid = np.zeros(2 * prepared.n_obs, dtype=np.float64)
    rvec0 = np.zeros(3, dtype=np.float64)
    tvec0 = np.zeros(3, dtype=np.float64)

    for grp in prepared.groups:
        T_ref_cam = cams[grp.alias]
        T_cam_board = se3_inv(T_ref_cam) @ boards[grp.frame_id]
        R = T_cam_board[:3, :3]
        t = T_cam_board[:3, 3]
        Xcam = (R @ grp.X_board.T).T + t

        K = prepared.K_by_alias[grp.alias]
        dist = prepared.dist_by_alias[grp.alias]
        proj, _ = cv2.projectPoints(
            Xcam.reshape(-1, 1, 3).astype(np.float64),
            rvec0,
            tvec0,
            K,
            dist,
        )
        err = proj.reshape(-1, 2) - grp.uv
        resid[2 * grp.obs_indices] = err[:, 0]
        resid[2 * grp.obs_indices + 1] = err[:, 1]
    return resid


def _residuals_wrapper(x: np.ndarray, prepared: _BAPrepared) -> np.ndarray:
    cams, boards = _unpack(x, prepared.cam_order, prepared.frame_order)
    cams[prepared.ref] = np.eye(4)
    return _residuals(x, prepared, cams, boards)


def solve_bundle_adjustment(
    problem: BAProblem,
    *,
    loss: str = "cauchy",
    f_scale: float = 1.0,
    max_nfev: int = 200,
    verbose: int = 1,
) -> BAResult:
    x0, cam_order, frame_order = _pack(problem)
    prepared = _prepare(problem, cam_order, frame_order)
    result = least_squares(
        fun=_residuals_wrapper,
        x0=x0,
        args=(prepared,),
        method="trf",
        loss=loss,
        f_scale=float(f_scale),
        max_nfev=int(max_nfev),
        verbose=int(verbose),
    )
    cams, boards = _unpack(result.x, cam_order, frame_order)
    cams[problem.aliases[0]] = np.eye(4)

    r = result.fun.reshape(-1, 2)
    per_cam_sq: dict[str, list[float]] = {a: [] for a in problem.aliases}
    per_frame_sq: dict[int, list[float]] = {f: [] for f in problem.initial_board_poses.keys()}
    for i, o in enumerate(problem.observations):
        e2 = float(r[i, 0] ** 2 + r[i, 1] ** 2)
        per_cam_sq[o.alias].append(e2)
        per_frame_sq[o.frame_id].append(e2)
    per_cam_rmse = {a: (float(np.sqrt(np.mean(v))) if v else float("nan")) for a, v in per_cam_sq.items()}
    per_frame_rmse = {f: (float(np.sqrt(np.mean(v))) if v else float("nan")) for f, v in per_frame_sq.items()}
    total_rmse = float(np.sqrt(np.mean(r.reshape(-1) ** 2)))

    return BAResult(
        reference=problem.aliases[0],
        cam_poses=cams,
        board_poses=boards,
        per_camera_rmse=per_cam_rmse,
        per_frame_rmse=per_frame_rmse,
        total_rmse=total_rmse,
        n_observations=len(problem.observations),
        metadata={"status": int(result.status), "nfev": int(result.nfev), "cost": float(result.cost)},
    )
