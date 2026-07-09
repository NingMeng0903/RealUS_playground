"""SE(3) utilities and multi-camera initial pose graph.

Rotation initialization uses Chatterjee & Govindu (PAMI 2018) style chordal L2
averaging: start from a spanning tree in the co-visibility graph, then perform
a few sweeps of geodesic mean per node. For our tabletop scenario the graph is
tiny (N=4 cameras, dense co-visibility) so a simple weighted mean already
converges cleanly.

Translations are recovered per co-visible pair from ``T_ci_cj = T_ci_b · T_cj_b^{-1}``
and averaged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


def se3_inv(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=T.dtype)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def se3_compose(*Ts: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    for T in Ts:
        out = out @ T
    return out


def se3_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    import cv2

    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    """Return the 6-vector (rotvec, trans) representation of an SE(3) element.

    This is not the geometric SE(3) log (which would translate via the V^{-1}
    matrix); we use the simple (axis-angle, translation) parameterization that
    is stable and easy to differentiate for BA.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    rot = Rotation.from_matrix(R).as_rotvec()
    return np.concatenate([rot, t])


def se3_exp(v: np.ndarray) -> np.ndarray:
    """Inverse of `se3_log`: (rotvec, trans) -> 4x4 SE(3)."""
    rvec = v[:3]
    t = v[3:]
    R = Rotation.from_rotvec(rvec).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


@dataclass
class InitialCameraPoses:
    reference: str
    poses: dict[str, np.ndarray]  # alias -> T_ref_cam (4x4)
    n_edges: dict[tuple[str, str], int]  # covisibility edge counts (for logging)


def initialize_camera_poses(
    per_view_poses: dict[int, dict[str, np.ndarray]],
    *,
    reference: str,
) -> InitialCameraPoses:
    """Build initial T_ref_cam for every camera from per-frame board poses.

    Parameters
    ----------
    per_view_poses
        ``{frame_id: {alias: T_cam_board (4,4)}}`` — output of `pnp.solve_view_pose`
        across all frames and cameras.
    reference
        Alias to use as the identity frame. Must appear in at least one frame.
    """
    aliases: set[str] = set()
    for views in per_view_poses.values():
        aliases.update(views.keys())
    if reference not in aliases:
        raise ValueError(f"Reference alias {reference!r} never appears; got {sorted(aliases)}")

    # For every co-visible (ref, other) pair in a frame we get an estimate of
    # T_ref_other:
    #     T_ref_board = T_other_board                       (both boards are the same!)
    # =>  T_ref_other = T_ref_board · (T_other_board)^{-1}
    # If a frame doesn't include the reference camera, we can still chain via
    # any other camera whose ref-relative pose is already known — but with 4
    # cameras and a shared board target the reference is usually present in
    # every frame, so we prefer direct edges to the reference for stability
    # and only fall back to two-hop chaining if needed.

    edges: dict[str, list[np.ndarray]] = {a: [] for a in aliases if a != reference}
    edge_counts: dict[tuple[str, str], int] = {}
    for views in per_view_poses.values():
        if reference not in views:
            continue
        T_ref_b = views[reference]
        for alias, T_other_b in views.items():
            if alias == reference:
                continue
            T_ref_other = T_ref_b @ np.linalg.inv(T_other_b)
            edges[alias].append(T_ref_other)
            edge_counts[(reference, alias)] = edge_counts.get((reference, alias), 0) + 1

    # For any camera with zero direct-to-reference observations, chain through
    # its most-covisible neighbour whose reference-relative pose is already known.
    missing = [a for a, lst in edges.items() if not lst]
    if missing:
        # Compute pairwise edges between non-reference cameras as fallback.
        pair_edges: dict[tuple[str, str], list[np.ndarray]] = {}
        for views in per_view_poses.values():
            names = list(views.keys())
            for i in range(len(names)):
                for j in range(len(names)):
                    if i == j:
                        continue
                    a, b = names[i], names[j]
                    T_ab = views[a] @ np.linalg.inv(views[b])  # T_a_b via shared board
                    pair_edges.setdefault((a, b), []).append(T_ab)
                    edge_counts[(a, b)] = edge_counts.get((a, b), 0) + 1
        # Iterate until nothing changes.
        known = {reference: np.eye(4)}
        for a, lst in edges.items():
            if lst:
                known[a] = _average_se3(lst)
        changed = True
        while changed:
            changed = False
            for a in list(missing):
                if a in known:
                    continue
                candidates: list[np.ndarray] = []
                for b in known.keys():
                    lst = pair_edges.get((b, a))
                    if lst:
                        # T_ref_a = T_ref_b · T_b_a
                        T_b_a = _average_se3(lst)
                        candidates.append(known[b] @ T_b_a)
                if candidates:
                    known[a] = _average_se3(candidates)
                    changed = True
        for a in missing:
            if a in known:
                edges[a] = [known[a]]

    poses = {reference: np.eye(4, dtype=np.float64)}
    for a, lst in edges.items():
        if not lst:
            raise RuntimeError(
                f"Camera {a!r} has no co-visibility path to the reference camera. "
                "Take more samples where the board is visible to at least one "
                "camera whose pose is known."
            )
        poses[a] = _average_se3(lst)

    return InitialCameraPoses(reference=reference, poses=poses, n_edges=edge_counts)


def _average_se3(Ts: list[np.ndarray]) -> np.ndarray:
    """Average a list of SE(3) transforms: chordal mean rotation + arithmetic mean translation."""
    rots = np.stack([T[:3, :3] for T in Ts], axis=0)
    ts = np.stack([T[:3, 3] for T in Ts], axis=0)
    # Chordal L2 mean via SVD projection of the arithmetic mean rotation matrix.
    Rmean = rots.mean(axis=0)
    U, _, Vt = np.linalg.svd(Rmean)
    D = np.eye(3)
    D[2, 2] = np.linalg.det(U @ Vt)
    R = U @ D @ Vt
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = ts.mean(axis=0)
    return T
