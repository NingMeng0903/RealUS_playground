"""Faithful port of EasyMocap ``easymocap/mytools/triangulator.py`` (iterative path).

Source reference: ref_code_library/EasyMocap/easymocap/mytools/triangulator.py
Used by mv1p ``SimpleTriangulate`` with ``mode: iterative`` (dist_max=25 px).
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

MAX_VIEWS = 30


def _make_cnk(n: int, k: int) -> dict[tuple[int, int], list[list[int]]]:
    res: dict[tuple[int, int], list[list[int]]] = {}
    for n_ in range(3, n + 1):
        n_0 = [i for i in range(n_)]
        for k_ in range(2, k + 1):
            res[(n_, k_)] = [list(c) for c in itertools.combinations(n_0, k_)]
    return res


_CNK = _make_cnk(MAX_VIEWS, 3)


def batch_triangulate(keypoints_: np.ndarray, pall: np.ndarray, min_view: int = 2) -> np.ndarray:
    """Weighted DLT triangulation (EasyMocap batch_triangulate)."""
    keypoints_ = np.asarray(keypoints_, dtype=np.float64)
    pall = np.asarray(pall, dtype=np.float64)
    v = (keypoints_[:, :, -1] > 0).sum(axis=0)
    valid_joint = np.where(v >= min_view)[0]
    if valid_joint.size == 0:
        return np.zeros((keypoints_.shape[1], 4), dtype=np.float64)
    keypoints = keypoints_[:, valid_joint]
    conf3d = keypoints[:, :, -1].sum(axis=0) / v[valid_joint]
    if len(pall.shape) == 3:
        p0 = pall[None, :, 0, :]
        p1 = pall[None, :, 1, :]
        p2 = pall[None, :, 2, :]
    else:
        p0 = pall[:, :, 0, :].swapaxes(0, 1)
        p1 = pall[:, :, 1, :].swapaxes(0, 1)
        p2 = pall[:, :, 2, :].swapaxes(0, 1)
    up2 = keypoints[:, :, 0].T[:, :, None] * p2
    vp2 = keypoints[:, :, 1].T[:, :, None] * p2
    conf = keypoints[:, :, 2].T[:, :, None]
    au = conf * (up2 - p0)
    av = conf * (vp2 - p1)
    a = np.hstack([au, av])
    _, _, vh = np.linalg.svd(a)
    x = vh[:, -1, :]
    x = x / x[:, 3:]
    result = np.zeros((keypoints_.shape[1], 4), dtype=np.float64)
    result[valid_joint, :3] = x[:, :3]
    result[valid_joint, 3] = conf3d
    return result


def project_points(keypoints: np.ndarray, rt: np.ndarray, einsum: str | None = None) -> np.ndarray:
    homo = np.concatenate([keypoints[..., :3], np.ones_like(keypoints[..., :1])], axis=-1)
    if einsum is None:
        if len(homo.shape) == 2 and len(rt.shape) == 3:
            kpts2d = np.einsum("vab,kb->vka", rt, homo)
        elif len(homo.shape) == 2 and len(rt.shape) == 4:
            kpts2d = np.einsum("vkab,kb->vka", rt, homo)
        else:
            raise ValueError(f"Unsupported shapes homo={homo.shape} rt={rt.shape}")
    else:
        kpts2d = np.einsum(einsum, rt, homo)
    kpts2d[..., :2] /= kpts2d[..., 2:]
    return kpts2d


def robust_triangulate_point(
    kpts2d: np.ndarray,
    pall: np.ndarray,
    dist_max: float,
    min_v: int = 3,
) -> tuple[list[int], np.ndarray | None]:
    n_v = int(kpts2d.shape[0])
    if n_v < min_v:
        return [], None
    key = (len(kpts2d), min(min_v, len(kpts2d)))
    index_ = _CNK.get(key)
    if not index_:
        return [], None
    proposals = np.zeros((len(index_), 4), dtype=np.float64)
    weight_self = np.zeros((n_v, len(index_)), dtype=np.float64)
    for i, index in enumerate(index_):
        weight_self[index, i] = 100.0
        point = batch_triangulate(kpts2d[index, :], pall[index], min_view=min_v)
        proposals[i] = point
    kpts_repro = project_points(proposals, pall)
    conf_mask = (proposals[None, :, -1] > 0) * (kpts2d[..., -1] > 0)
    err = np.linalg.norm(kpts_repro[..., :2] - kpts2d[..., :2], axis=-1) * conf_mask
    valid = 1.0 - err / float(dist_max)
    valid[valid < 0] = 0.0
    conf = kpts2d[..., -1]
    weight = conf
    weight_sum = (weight * valid).sum(axis=0) + ((valid > 0) * weight_self).sum(axis=0) - min_v * 100.0
    if weight_sum.max() < 0:
        return [], None
    best = int(weight_sum.argmax())
    if (err[index_[best], best] > dist_max).any():
        return [], None
    point = proposals[best]
    best_add = np.where(valid[:, best])[0].tolist()
    index = list(index_[best])
    best_add.sort(key=lambda x: -float(weight[x]))
    for add in best_add:
        if add in index:
            continue
        index.append(add)
        point = batch_triangulate(kpts2d[index, :], pall[index], min_view=min_v)
        kpts_repro = project_points(point, pall[index])
        err_add = np.linalg.norm(kpts_repro[..., :2] - kpts2d[index, ..., :2], axis=-1)
        if (err_add > dist_max).any():
            index.remove(add)
            break
    return index, point


def _remove_outview(kpts2d: np.ndarray, out_view: np.ndarray | list[int]) -> bool:
    if len(out_view) == 0:
        return False
    kpts2d[int(out_view[0])] = 0.0
    return True


def _remove_outjoint(
    kpts2d: np.ndarray,
    pall: np.ndarray,
    out_joint: np.ndarray | list[int],
    dist_max: float,
    min_view: int = 3,
) -> bool:
    if len(out_joint) == 0:
        return False
    for nj in out_joint:
        nj = int(nj)
        valid = np.where(kpts2d[:, nj, -1] > 0)[0]
        if len(valid) < min_view:
            kpts2d[:, nj, -1] = 0.0
            continue
        if len(valid) > MAX_VIEWS:
            conf = -kpts2d[:, nj, -1]
            valid = conf.argsort()[:MAX_VIEWS]
        index_j, _point = robust_triangulate_point(
            kpts2d[valid, nj : nj + 1],
            pall[valid],
            dist_max=dist_max,
            min_v=min_view,
        )
        if not index_j:
            kpts2d[:, nj, -1] = 0.0
            continue
        index_j = valid[index_j]
        set0 = np.zeros(kpts2d.shape[0], dtype=np.float64)
        set0[index_j] = 1.0
        kpts2d[:, nj, -1] *= set0
    return True


def project_and_distance(kpts3d: np.ndarray, rt: np.ndarray, kpts2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kpts_proj = project_points(kpts3d, rt)
    conf = (kpts3d[None, :, -1] > 0) * (kpts2d[:, :, -1] > 0)
    dist = np.linalg.norm(kpts_proj[..., :2] - kpts2d[..., :2], axis=-1) * conf
    return dist, conf


def iterative_triangulate(
    kpts2d: np.ndarray,
    rt: np.ndarray,
    previous: np.ndarray | None = None,
    *,
    min_conf: float = 0.1,
    min_view: int = 3,
    min_joints: int = 3,
    dist_max: float = 25.0,
    dist_vel: float = 0.05,
    thres_outlier_view: float = 0.4,
    thres_outlier_joint: float = 0.4,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """EasyMocap iterative_triangulate (mv1p SimpleTriangulate iterative default)."""
    kpts2d = np.asarray(kpts2d, dtype=np.float64).copy()
    rt = np.asarray(rt, dtype=np.float64)
    conf = kpts2d[..., -1]
    kpts2d[conf < min_conf] = 0.0

    if previous is not None:
        previous = np.asarray(previous, dtype=np.float64)
        dist, conf_prev = project_and_distance(previous, rt, kpts2d)
        nottrack = (dist > dist_vel) & conf_prev
        if nottrack.sum() > 0:
            kpts2d[nottrack] = 0.0

    while True:
        kpts3d = batch_triangulate(kpts2d, rt, min_view=min_view)
        dist, conf_mask = project_and_distance(kpts3d, rt, kpts2d)
        vv, jj = np.where(dist > dist_max)
        if vv.shape[0] < 1:
            break
        ratio_outlier_view = (dist > dist_max).sum(axis=1) / (1e-5 + conf_mask.sum(axis=1))
        ratio_outlier_joint = (dist > dist_max).sum(axis=0) / (1e-5 + conf_mask.sum(axis=0))
        out_view = np.where(ratio_outlier_view > thres_outlier_view)[0]
        out_joint = np.where(ratio_outlier_joint > thres_outlier_joint)[0]
        if len(out_view) > 1:
            dist_view = dist.sum(axis=1) / (1e-5 + conf_mask.sum(axis=1))
            out_view = out_view.tolist()
            out_view.sort(key=lambda x: -float(dist_view[x]))
        if _remove_outview(kpts2d, out_view):
            continue
        if _remove_outjoint(kpts2d, rt, out_joint, dist_max, min_view=min_view):
            continue
        kpts2d[vv, jj, -1] = 0.0

    if (kpts3d[..., -1] > 0).sum() < min_joints:
        kpts3d[..., -1] = 0.0
        kpts2d[..., -1] = 0.0
    return kpts3d, kpts2d


def build_diagnostics(
    kpts2d_masked: np.ndarray,
    kpts3d: np.ndarray,
    rt: np.ndarray,
    *,
    dist_max: float,
    n_views: int,
    n_joints: int,
) -> dict[str, Any]:
    """Overlay-compatible diagnostics from EM iterative output."""
    used_views: list[list[int]] = []
    for j in range(n_joints):
        views = np.where(kpts2d_masked[:, j, -1] > 0)[0].tolist()
        used_views.append([int(v) for v in views])

    reproj_err = np.full((n_views, n_joints), np.nan, dtype=np.float32)
    dist, conf = project_and_distance(kpts3d, rt, kpts2d_masked)
    reproj_err = np.where(conf > 0, dist.astype(np.float32), np.nan)

    dropped_view_indices = [v for v in range(n_views) if not np.any(kpts2d_masked[v, :, -1] > 0)]

    valid_count = int(np.sum(kpts3d[:, 3] > 0.0))
    return {
        "mode": "easymocap_iterative",
        "valid_joints": valid_count,
        "n_views": int(n_views),
        "n_joints": int(n_joints),
        "reproj_err": reproj_err.tolist(),
        "used_views": used_views,
        "dropped_view_indices": dropped_view_indices,
    }
