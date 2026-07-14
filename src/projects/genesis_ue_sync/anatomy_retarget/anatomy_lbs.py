"""Linear blend skinning for retargeted anatomy assets."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .pose_adapter import pose_to_smplx55_axis_angle
from .rigged_asset import AnatomyRiggedAsset


_CUDA_ASSET_CACHE: dict[int, tuple[Any, Any, Any]] = {}


def _dense_asset_weights(asset: AnatomyRiggedAsset) -> np.ndarray:
    if asset.lbs_weights is not None:
        return np.asarray(asset.lbs_weights, dtype=np.float32)
    if asset.driver_indices is None or asset.driver_weights is None or asset.source_bone_names is None:
        raise ValueError("asset has no usable skinning weights")
    dense = np.zeros((asset.vertices_rest.shape[0], len(asset.source_bone_names)), dtype=np.float32)
    rows = np.arange(dense.shape[0])
    for slot in range(asset.driver_indices.shape[1]):
        np.add.at(dense, (rows, asset.driver_indices[:, slot]), asset.driver_weights[:, slot])
    return dense


def _cuda_requested() -> bool:
    value = str(os.environ.get("AMONGUS_ANATOMY_LBS_DEVICE", "auto")).strip().lower()
    if value in {"cpu", "off", "false", "0"}:
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _skin_vertices_cuda(
    asset: AnatomyRiggedAsset,
    transforms: np.ndarray,
    transl: Any | None,
) -> np.ndarray:
    import torch

    key = id(asset)
    cached = _CUDA_ASSET_CACHE.get(key)
    if cached is None:
        if asset.driver_indices is None or asset.driver_weights is None:
            from .rigged_asset import sparse_driver_weights

            indices, weights = sparse_driver_weights(asset.lbs_weights)
        else:
            indices, weights = asset.driver_indices, asset.driver_weights
        vertices_t = torch.as_tensor(asset.vertices_rest, dtype=torch.float32, device="cuda")
        indices_t = torch.as_tensor(indices, dtype=torch.long, device="cuda")
        weights_t = torch.as_tensor(weights, dtype=torch.float32, device="cuda")
        cached = (vertices_t, indices_t, weights_t)
        _CUDA_ASSET_CACHE[key] = cached
    vertices_t, indices_t, weights_t = cached
    tf = torch.as_tensor(transforms, dtype=torch.float32, device="cuda")
    selected = tf[indices_t]
    blended = torch.sum(selected * weights_t[..., None, None], dim=1)
    ones = torch.ones((vertices_t.shape[0], 1), dtype=torch.float32, device="cuda")
    homo = torch.cat((vertices_t, ones), dim=1)
    posed = torch.bmm(blended, homo.unsqueeze(-1))[:, :3, 0]
    if transl is not None:
        posed = posed + torch.as_tensor(transl, dtype=torch.float32, device="cuda").reshape(1, 3)
    return posed.detach().cpu().numpy().astype(np.float32, copy=False)


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    rows = np.asarray(axis_angle, dtype=np.float32).reshape(-1, 3)
    out = np.tile(np.eye(3, dtype=np.float32), (rows.shape[0], 1, 1))
    angles = np.linalg.norm(rows, axis=1)
    for idx, angle in enumerate(angles.tolist()):
        if float(angle) < 1.0e-8:
            continue
        axis = rows[idx] / float(angle)
        x, y, z = [float(v) for v in axis.tolist()]
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        one_c = 1.0 - c
        out[idx] = np.asarray(
            [
                [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
                [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
                [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
            ],
            dtype=np.float32,
        )
    return out


def joint_global_transforms(
    *,
    pose_axis_angle: Any,
    rest_joints: np.ndarray,
    parents: np.ndarray,
) -> np.ndarray:
    pose = pose_to_smplx55_axis_angle(pose_axis_angle)
    joints = np.asarray(rest_joints, dtype=np.float32).reshape(-1, 3)
    pa = np.asarray(parents, dtype=np.int32).reshape(-1)
    n = min(int(joints.shape[0]), int(pa.shape[0]), int(pose.shape[0]))
    rot = axis_angle_to_matrix(pose[:n])
    out = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    for idx in range(n):
        local = np.eye(4, dtype=np.float32)
        local[:3, :3] = rot[idx]
        if idx == 0 or int(pa[idx]) < 0:
            local[:3, 3] = joints[idx]
            out[idx] = local
        else:
            parent = int(pa[idx])
            local[:3, 3] = joints[idx] - joints[parent]
            out[idx] = out[parent] @ local
    return out


def _rigid_frame(origin: np.ndarray, primary: np.ndarray, plane: np.ndarray) -> np.ndarray:
    x = np.asarray(primary - origin, dtype=np.float64)
    x /= max(float(np.linalg.norm(x)), 1.0e-10)
    z = np.cross(x, np.asarray(plane - origin, dtype=np.float64))
    z /= max(float(np.linalg.norm(z)), 1.0e-10)
    y = np.cross(z, x)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.stack((x, y, z), axis=1)
    out[:3, 3] = origin
    return out


def _interpolate_rigid(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    t = float(np.clip(alpha, 0.0, 1.0))
    Ra = np.asarray(a[:3, :3], dtype=np.float64)
    Rb = np.asarray(b[:3, :3], dtype=np.float64)
    delta = Rotation.from_matrix(Ra.T @ Rb).as_rotvec()
    R = Ra @ Rotation.from_rotvec(t * delta).as_matrix()
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = R.astype(np.float32)
    out[:3, 3] = ((1.0 - t) * a[:3, 3] + t * b[:3, 3]).astype(np.float32)
    return out


def source_bone_skinning_transforms(
    asset: AnatomyRiggedAsset,
    pose_axis_angle: Any,
) -> np.ndarray:
    """Solve all Blender source-bone transforms from a full SMPL-X pose."""
    if asset.source_bone_names is None:
        raise ValueError("source bone transforms requested for a legacy asset")
    pose_global = joint_global_transforms(
        pose_axis_angle=pose_axis_angle,
        rest_joints=asset.rest_joints,
        parents=asset.parents,
    ).astype(np.float64)
    rest_global = joint_global_transforms(
        pose_axis_angle=np.zeros((55, 3), dtype=np.float32),
        rest_joints=asset.rest_joints,
        parents=asset.parents,
    ).astype(np.float64)
    joint_delta = pose_global @ np.linalg.inv(rest_global)
    rest_points = np.asarray(asset.rest_joints, dtype=np.float64)
    pose_points = pose_global[:, :3, 3]
    joint_index = {name: idx for idx, name in enumerate(asset.joint_names)}
    source_delta = np.tile(np.eye(4, dtype=np.float32), (len(asset.source_bone_names), 1, 1))
    types = asset.source_bone_driver_types or ["direct_joint"] * len(asset.source_bone_names)
    for bi, driver_type in enumerate(types):
        a = int(asset.source_bone_smplx_a[bi])
        b = int(asset.source_bone_smplx_b[bi])
        alpha = float(asset.source_bone_blend[bi])
        if driver_type.startswith("scapula_"):
            side = "left" if driver_type.endswith("left") else "right"
            s, c, h = (joint_index["spine3"], joint_index[f"{side}_collar"], joint_index[f"{side}_shoulder"])
            F0 = _rigid_frame(rest_points[h], rest_points[c], rest_points[s])
            F1 = _rigid_frame(pose_points[h], pose_points[c], pose_points[s])
            source_delta[bi] = (F1 @ np.linalg.inv(F0)).astype(np.float32)
        elif driver_type.startswith("patella_"):
            side = "left" if driver_type.endswith("left") else "right"
            hip, knee, ankle = (joint_index[f"{side}_hip"], joint_index[f"{side}_knee"], joint_index[f"{side}_ankle"])
            F0 = _rigid_frame(rest_points[knee], rest_points[ankle], rest_points[hip])
            F1 = _rigid_frame(pose_points[knee], pose_points[ankle], pose_points[hip])
            source_delta[bi] = (F1 @ np.linalg.inv(F0)).astype(np.float32)
        elif a != b:
            source_delta[bi] = _interpolate_rigid(joint_delta[a], joint_delta[b], alpha)
        else:
            source_delta[bi] = joint_delta[a].astype(np.float32)
    source_pose_global = source_delta @ np.asarray(asset.source_rest_global, dtype=np.float32)
    return source_pose_global @ np.asarray(asset.source_inverse_bind, dtype=np.float32)


def skin_vertices(
    asset: AnatomyRiggedAsset,
    pose_axis_angle: Any,
    *,
    transl: Any | None = None,
) -> np.ndarray:
    asset.validate()
    vertices = np.asarray(asset.vertices_rest, dtype=np.float32).reshape(-1, 3)
    if asset.source_bone_names is not None:
        transforms = source_bone_skinning_transforms(asset, pose_axis_angle)
    else:
        weights = _dense_asset_weights(asset)
        inverse_bind = np.asarray(asset.inverse_bind, dtype=np.float32)
        global_tf = joint_global_transforms(
            pose_axis_angle=pose_axis_angle,
            rest_joints=asset.rest_joints,
            parents=asset.parents,
        )
        joint_count = min(global_tf.shape[0], inverse_bind.shape[0], weights.shape[1])
        transforms = np.matmul(global_tf[:joint_count], inverse_bind[:joint_count])
    if _cuda_requested():
        return _skin_vertices_cuda(asset, transforms, transl)
    if asset.driver_indices is not None and asset.driver_weights is not None:
        selected = transforms[np.asarray(asset.driver_indices, dtype=np.int64)]
        blended = np.sum(selected * np.asarray(asset.driver_weights, dtype=np.float32)[..., None, None], axis=1)
    else:
        weights = _dense_asset_weights(asset)
        joint_count = min(transforms.shape[0], weights.shape[1])
        blended = np.matmul(weights[:, :joint_count], transforms[:joint_count].reshape(joint_count, 16)).reshape(-1, 4, 4)
    homo = np.concatenate([vertices, np.ones((vertices.shape[0], 1), dtype=np.float32)], axis=1)
    posed = np.matmul(blended, homo[:, :, None])[:, :3, 0].astype(np.float32)
    if transl is not None:
        posed += np.asarray(transl, dtype=np.float32).reshape(1, 3)
    return posed


def skin_points(
    asset: AnatomyRiggedAsset,
    rest_points: Any,
    *,
    pose_axis_angle: Any,
    transl: Any | None = None,
    anchor_vertices: Any | None = None,
    anchor_weights: Any | None = None,
    neighbor_k: int = 4,
) -> np.ndarray:
    """Skin arbitrary rest points by interpolating LBS weights from nearby mesh vertices."""
    asset.validate()
    pts = np.asarray(rest_points, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        return pts.copy()
    verts = np.asarray(asset.vertices_rest if anchor_vertices is None else anchor_vertices, dtype=np.float32).reshape(-1, 3)
    weights = np.asarray(_dense_asset_weights(asset) if anchor_weights is None else anchor_weights, dtype=np.float32)
    k = max(1, min(int(neighbor_k), int(verts.shape[0])))
    try:
        from scipy.spatial import cKDTree

        dist, idx = cKDTree(verts).query(pts, k=k)
    except Exception:
        dist = np.linalg.norm(verts[:, None, :] - pts[None, :, :], axis=2).T
        idx = np.argsort(dist, axis=1)[:, :k]
        dist = np.take_along_axis(dist, idx, axis=1)
    if k == 1:
        dist = np.asarray(dist, dtype=np.float32).reshape(-1, 1)
        idx = np.asarray(idx, dtype=np.int64).reshape(-1, 1)
    w_dist = 1.0 / (np.square(dist) + 1.0e-8)
    w_dist /= np.maximum(w_dist.sum(axis=1, keepdims=True), 1.0e-8)
    point_weights = np.zeros((pts.shape[0], weights.shape[1]), dtype=np.float32)
    for ki in range(k):
        point_weights += w_dist[:, ki : ki + 1] * weights[idx[:, ki]]
    point_weights /= np.maximum(point_weights.sum(axis=1, keepdims=True), 1.0e-8)

    if asset.source_bone_names is not None:
        transforms = source_bone_skinning_transforms(asset, pose_axis_angle)
    else:
        inverse_bind = np.asarray(asset.inverse_bind, dtype=np.float32)
        global_tf = joint_global_transforms(
            pose_axis_angle=pose_axis_angle,
            rest_joints=asset.rest_joints,
            parents=asset.parents,
        )
        joint_count = min(global_tf.shape[0], inverse_bind.shape[0], weights.shape[1])
        transforms = np.matmul(global_tf[:joint_count], inverse_bind[:joint_count])
    joint_count = min(transforms.shape[0], weights.shape[1])
    blended = np.matmul(point_weights[:, :joint_count], transforms.reshape(joint_count, 16)).reshape(-1, 4, 4)
    homo = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    posed = np.matmul(blended, homo[:, :, None])[:, :3, 0].astype(np.float32)
    if transl is not None:
        posed += np.asarray(transl, dtype=np.float32).reshape(1, 3)
    return posed


def compute_point_lbs_weights(
    asset: AnatomyRiggedAsset,
    rest_points: Any,
    *,
    anchor_vertices: Any | None = None,
    anchor_weights: Any | None = None,
    neighbor_k: int = 6,
) -> np.ndarray:
    """Interpolate LBS weights for arbitrary rest points from nearby mesh vertices."""
    asset.validate()
    pts = np.asarray(rest_points, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        width = len(asset.source_bone_names) if asset.source_bone_names is not None else len(asset.joint_names)
        return np.zeros((0, width), dtype=np.float32)
    verts = np.asarray(asset.vertices_rest if anchor_vertices is None else anchor_vertices, dtype=np.float32).reshape(-1, 3)
    weights = np.asarray(_dense_asset_weights(asset) if anchor_weights is None else anchor_weights, dtype=np.float32)
    k = max(1, min(int(neighbor_k), int(verts.shape[0])))
    try:
        from scipy.spatial import cKDTree

        dist, idx = cKDTree(verts).query(pts, k=k)
    except Exception:
        dist = np.linalg.norm(verts[:, None, :] - pts[None, :, :], axis=2).T
        idx = np.argsort(dist, axis=1)[:, :k]
        dist = np.take_along_axis(dist, idx, axis=1)
    if k == 1:
        dist = np.asarray(dist, dtype=np.float32).reshape(-1, 1)
        idx = np.asarray(idx, dtype=np.int64).reshape(-1, 1)
    w_dist = 1.0 / (np.square(dist) + 1.0e-8)
    w_dist /= np.maximum(w_dist.sum(axis=1, keepdims=True), 1.0e-8)
    point_weights = np.zeros((pts.shape[0], weights.shape[1]), dtype=np.float32)
    for ki in range(k):
        point_weights += w_dist[:, ki : ki + 1] * weights[idx[:, ki]]
    point_weights /= np.maximum(point_weights.sum(axis=1, keepdims=True), 1.0e-8)
    return point_weights


def skin_points_with_weights(
    asset: AnatomyRiggedAsset,
    rest_points: Any,
    point_weights: np.ndarray,
    *,
    pose_axis_angle: Any,
    transl: Any | None = None,
) -> np.ndarray:
    asset.validate()
    pts = np.asarray(rest_points, dtype=np.float32).reshape(-1, 3)
    weights = np.asarray(point_weights, dtype=np.float32)
    if asset.source_bone_names is not None:
        transforms = source_bone_skinning_transforms(asset, pose_axis_angle)
    else:
        inverse_bind = np.asarray(asset.inverse_bind, dtype=np.float32)
        global_tf = joint_global_transforms(
            pose_axis_angle=pose_axis_angle,
            rest_joints=asset.rest_joints,
            parents=asset.parents,
        )
        joint_count = min(global_tf.shape[0], inverse_bind.shape[0], weights.shape[1])
        transforms = np.matmul(global_tf[:joint_count], inverse_bind[:joint_count])
    joint_count = min(transforms.shape[0], weights.shape[1])
    blended = np.matmul(weights[:, :joint_count], transforms.reshape(joint_count, 16)).reshape(-1, 4, 4)
    homo = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    posed = np.matmul(blended, homo[:, :, None])[:, :3, 0].astype(np.float32)
    if transl is not None:
        posed += np.asarray(transl, dtype=np.float32).reshape(1, 3)
    return posed
