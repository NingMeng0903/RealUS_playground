"""Skin anatomy with matrix LBS and isolated opt-in DQS for soft tissue."""

from __future__ import annotations

import os
import weakref
from typing import Any

import numpy as np

from .pose_adapter import pose_to_smplx55_axis_angle
from .rigged_asset import SOURCE_DRIVER_MODES, AnatomyRiggedAsset


_CUDA_ASSET_CACHE: dict[
    int,
    tuple[weakref.ReferenceType[AnatomyRiggedAsset], Any, Any, Any, Any],
] = {}


def _soft_tissue_vertex_mask(asset: AnatomyRiggedAsset) -> np.ndarray:
    """Return the vertices that may use dual-quaternion blending.

    Bone meshes intentionally retain matrix LBS: most of them have a single
    controlling bone and this preserves their authored rigid vertices exactly.
    The per-mesh tissue labels are optional in older in-memory assets, so a
    missing or malformed labelling deliberately falls back to LBS everywhere.
    """
    ranges = asset.source_vertex_ranges
    tissues = asset.source_tissues
    count = int(np.asarray(asset.vertices_rest).shape[0])
    if ranges is None or tissues is None:
        return np.zeros(count, dtype=bool)
    ranges = np.asarray(ranges, dtype=np.int64).reshape(-1, 2)
    if ranges.shape[0] != len(tissues):
        return np.zeros(count, dtype=bool)
    mask = np.zeros(count, dtype=bool)
    for (start, end), tissue in zip(ranges.tolist(), tissues):
        # Only labels exported by the extraction pipeline are trusted.  This
        # keeps unknown legacy assets on their historical LBS result.
        # Vessels and nerves deliberately stay on the authored LBS path.  The
        # former DQS-only path did not fix their real issue (rest-frame drift)
        # and visibly twisted long hand/foot branches relative to the Blender
        # reference.  DQS remains useful for compact organs.
        if str(tissue).strip().lower() not in {"organ", "connective_tissue"}:
            continue
        lo = max(0, int(start))
        hi = min(count, int(end))
        if hi > lo:
            mask[lo:hi] = True
    return mask


def _dqs_requested() -> bool:
    """Return whether the isolated soft-tissue DQS path is explicitly enabled."""
    return str(os.environ.get("AMONGUS_ANATOMY_DQS", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _matrix_quaternions_numpy(transforms: np.ndarray) -> np.ndarray:
    """Convert proper rotation matrices to scalar-first unit quaternions."""
    matrices = np.asarray(transforms, dtype=np.float64).reshape(-1, 4, 4)[:, :3, :3]
    out = np.empty((matrices.shape[0], 4), dtype=np.float64)
    for index, matrix in enumerate(matrices):
        trace = float(np.trace(matrix))
        if trace > 0.0:
            scale = 2.0 * np.sqrt(trace + 1.0)
            out[index] = (0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale,
                          (matrix[0, 2] - matrix[2, 0]) / scale, (matrix[1, 0] - matrix[0, 1]) / scale)
            continue
        diagonal = np.diag(matrix)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = 2.0 * np.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 1.0e-16))
            out[index] = ((matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                          (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale)
        elif axis == 1:
            scale = 2.0 * np.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 1.0e-16))
            out[index] = ((matrix[0, 2] - matrix[2, 0]) / scale, (matrix[0, 1] + matrix[1, 0]) / scale,
                          0.25 * scale, (matrix[1, 2] + matrix[2, 1]) / scale)
        else:
            scale = 2.0 * np.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 1.0e-16))
            out[index] = ((matrix[1, 0] - matrix[0, 1]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale,
                          (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale)
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1.0e-12)
    return out


def _dual_quaternion_skin_numpy(
    vertices: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    transforms: np.ndarray,
) -> np.ndarray:
    """Skin points with sign-consistent dual-quaternion blending."""
    qr = _matrix_quaternions_numpy(transforms)
    translation = np.asarray(transforms, dtype=np.float64)[:, :3, 3]
    qd = 0.5 * np.concatenate(
        (-np.sum(translation * qr[:, 1:], axis=1, keepdims=True),
         qr[:, :1] * translation + np.cross(translation, qr[:, 1:])), axis=1
    )
    count = int(np.asarray(vertices).reshape(-1, 3).shape[0])
    selected_indices = np.asarray(indices, dtype=np.int64).reshape(count, -1)
    selected_r = qr[selected_indices]
    selected_d = qd[selected_indices]
    selected_w = np.asarray(weights, dtype=np.float64).reshape(count, -1)
    reference = selected_r[np.arange(selected_r.shape[0]), np.argmax(selected_w, axis=1)]
    signs = np.where(np.sum(selected_r * reference[:, None], axis=2, keepdims=True) < 0.0, -1.0, 1.0)
    real = np.sum(selected_r * selected_w[..., None] * signs, axis=1)
    dual = np.sum(selected_d * selected_w[..., None] * signs, axis=1)
    norm = np.maximum(np.linalg.norm(real, axis=1, keepdims=True), 1.0e-12)
    real /= norm
    dual /= norm
    vector = real[:, 1:]
    points = np.asarray(vertices, dtype=np.float64)
    twice_cross = 2.0 * np.cross(vector, points)
    rotated = points + real[:, :1] * twice_cross + np.cross(vector, twice_cross)
    offset = 2.0 * (
        real[:, :1] * dual[:, 1:]
        - dual[:, :1] * vector
        + np.cross(vector, dual[:, 1:])
    )
    return (rotated + offset).astype(np.float32)


def _matrix_quaternions_torch(transforms: Any) -> Any:
    """Torch equivalent of :func:`_matrix_quaternions_numpy`."""
    import torch

    matrix = transforms[:, :3, :3]
    count = matrix.shape[0]
    out = torch.empty((count, 4), dtype=matrix.dtype, device=matrix.device)
    trace = matrix[:, 0, 0] + matrix[:, 1, 1] + matrix[:, 2, 2]
    positive = trace > 0.0
    if torch.any(positive):
        scale = 2.0 * torch.sqrt(torch.clamp(trace[positive] + 1.0, min=1.0e-16))
        m = matrix[positive]
        out[positive] = torch.stack((0.25 * scale, (m[:, 2, 1] - m[:, 1, 2]) / scale,
                                     (m[:, 0, 2] - m[:, 2, 0]) / scale,
                                     (m[:, 1, 0] - m[:, 0, 1]) / scale), dim=1)
    diagonal = torch.stack((matrix[:, 0, 0], matrix[:, 1, 1], matrix[:, 2, 2]), dim=1)
    axis = torch.argmax(diagonal, dim=1)
    for choice in range(3):
        mask = (~positive) & (axis == choice)
        if not torch.any(mask):
            continue
        m = matrix[mask]
        if choice == 0:
            scale = 2.0 * torch.sqrt(torch.clamp(1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], min=1.0e-16))
            values = ( (m[:, 2, 1] - m[:, 1, 2]) / scale, 0.25 * scale,
                       (m[:, 0, 1] + m[:, 1, 0]) / scale, (m[:, 0, 2] + m[:, 2, 0]) / scale )
        elif choice == 1:
            scale = 2.0 * torch.sqrt(torch.clamp(1.0 + m[:, 1, 1] - m[:, 0, 0] - m[:, 2, 2], min=1.0e-16))
            values = ( (m[:, 0, 2] - m[:, 2, 0]) / scale, (m[:, 0, 1] + m[:, 1, 0]) / scale,
                       0.25 * scale, (m[:, 1, 2] + m[:, 2, 1]) / scale )
        else:
            scale = 2.0 * torch.sqrt(torch.clamp(1.0 + m[:, 2, 2] - m[:, 0, 0] - m[:, 1, 1], min=1.0e-16))
            values = ( (m[:, 1, 0] - m[:, 0, 1]) / scale, (m[:, 0, 2] + m[:, 2, 0]) / scale,
                       (m[:, 1, 2] + m[:, 2, 1]) / scale, 0.25 * scale )
        out[mask] = torch.stack(values, dim=1)
    return out / torch.clamp(torch.linalg.vector_norm(out, dim=1, keepdim=True), min=1.0e-12)


def _dual_quaternion_skin_torch(
    vertices: Any,
    indices: Any,
    weights: Any,
    transforms: Any,
) -> Any:
    import torch

    real_bones = _matrix_quaternions_torch(transforms)
    translation = transforms[:, :3, 3]
    dual_bones = 0.5 * torch.cat(
        (-torch.sum(translation * real_bones[:, 1:], dim=1, keepdim=True),
         real_bones[:, :1] * translation + torch.linalg.cross(translation, real_bones[:, 1:])), dim=1
    )
    real = real_bones[indices]
    dual = dual_bones[indices]
    reference = real[torch.arange(real.shape[0], device=real.device), torch.argmax(weights, dim=1)]
    signs = torch.where(torch.sum(real * reference[:, None], dim=2, keepdim=True) < 0.0, -1.0, 1.0)
    blended_real = torch.sum(real * weights[..., None] * signs, dim=1)
    blended_dual = torch.sum(dual * weights[..., None] * signs, dim=1)
    norm = torch.clamp(torch.linalg.vector_norm(blended_real, dim=1, keepdim=True), min=1.0e-12)
    blended_real = blended_real / norm
    blended_dual = blended_dual / norm
    vector = blended_real[:, 1:]
    twice_cross = 2.0 * torch.linalg.cross(vector, vertices)
    rotated = vertices + blended_real[:, :1] * twice_cross + torch.linalg.cross(vector, twice_cross)
    offset = 2.0 * (
        blended_real[:, :1] * blended_dual[:, 1:]
        - blended_dual[:, :1] * vector
        + torch.linalg.cross(vector, blended_dual[:, 1:])
    )
    return rotated + offset


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
    if cached is None or cached[0]() is not asset:
        if asset.driver_indices is None or asset.driver_weights is None:
            from .rigged_asset import sparse_driver_weights

            indices, weights = sparse_driver_weights(asset.lbs_weights)
        else:
            indices, weights = asset.driver_indices, asset.driver_weights
        vertices_t = torch.as_tensor(asset.vertices_rest, dtype=torch.float32, device="cuda")
        indices_t = torch.as_tensor(indices, dtype=torch.long, device="cuda")
        weights_t = torch.as_tensor(weights, dtype=torch.float32, device="cuda")
        soft_mask_t = torch.as_tensor(_soft_tissue_vertex_mask(asset), dtype=torch.bool, device="cuda")
        cached = (
            weakref.ref(asset),
            vertices_t,
            indices_t,
            weights_t,
            soft_mask_t,
        )
        _CUDA_ASSET_CACHE[key] = cached
    _asset_ref, vertices_t, indices_t, weights_t, soft_mask_t = cached
    tf = torch.as_tensor(transforms, dtype=torch.float32, device="cuda")
    selected = tf[indices_t]
    blended = torch.sum(selected * weights_t[..., None, None], dim=1)
    ones = torch.ones((vertices_t.shape[0], 1), dtype=torch.float32, device="cuda")
    homo = torch.cat((vertices_t, ones), dim=1)
    posed = torch.bmm(blended, homo.unsqueeze(-1))[:, :3, 0]
    # Matrix LBS is the parity baseline.  DQS is isolated behind an explicit
    # opt-in because even a tissue-only default diverges from authored Blender
    # matrix skinning.
    if _dqs_requested() and bool(torch.any(soft_mask_t)):
        dqs = _dual_quaternion_skin_torch(vertices_t, indices_t, weights_t, tf)
        posed = torch.where(soft_mask_t[:, None], dqs, posed)
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


def _rigid_frame(origin: np.ndarray, primary: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """Compatibility helper for offline segment-coupling diagnostics."""
    x = np.asarray(primary - origin, dtype=np.float64)
    x /= max(float(np.linalg.norm(x)), 1.0e-10)
    z = np.cross(x, np.asarray(plane - origin, dtype=np.float64))
    z /= max(float(np.linalg.norm(z)), 1.0e-10)
    y = np.cross(z, x)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.stack((x, y, z), axis=1)
    out[:3, 3] = origin
    return out


def _segment_frame(origin: np.ndarray, endpoint: np.ndarray, reference_x: np.ndarray) -> np.ndarray:
    """Stable limb/head frame with its Y axis fixed by anatomical endpoints."""
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    endpoint = np.asarray(endpoint, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(origin)) or not np.all(np.isfinite(endpoint)):
        raise ValueError("segment driver endpoints must be finite")
    y = np.array(endpoint - origin, dtype=np.float64, copy=True)
    length = float(np.linalg.norm(y))
    if length < 1.0e-8:
        raise ValueError("segment driver endpoints are degenerate")
    y /= length
    # ``reference_x`` is commonly a view into source_bind_global.  In-place
    # orthogonalisation must never corrupt the persisted bind matrix.
    x = np.asarray(reference_x, dtype=np.float64).reshape(3).copy()
    if not np.all(np.isfinite(x)):
        raise ValueError("segment driver transverse axis must be finite")
    x -= float(x @ y) * y
    if float(np.linalg.norm(x)) < 1.0e-8:
        # A clavicle can be almost parallel to world X.  Choose the least
        # aligned canonical axis instead of producing a singular frame.
        x = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(y)))]
        x -= float(x @ y) * y
    x /= max(float(np.linalg.norm(x)), 1.0e-10)
    z = np.cross(x, y)
    z /= max(float(np.linalg.norm(z)), 1.0e-10)
    x = np.cross(y, z)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.stack((x, y, z), axis=1)
    out[:3, 3] = origin
    return out


def _three_joint_frame(points: np.ndarray, joints: np.ndarray, reference_x: np.ndarray) -> np.ndarray:
    """Frame from the three explicit V5 driver joints.

    This is used for shoulder girdles, pelvis and head drivers.  It avoids the
    old implicit ``first child`` rule, which made a pelvis point at one hip and
    made a scapula inherit a humerus rotation.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    ids = np.asarray(joints, dtype=np.int64).reshape(-1)
    if (
        ids.shape != (3,)
        or np.any(ids < 0)
        or np.any(ids >= len(points))
        or len(np.unique(ids)) != 3
    ):
        raise ValueError("three-joint driver requires three distinct valid joints")
    if not np.all(np.isfinite(points[ids])):
        raise ValueError("three-joint driver points must be finite")
    origin = np.asarray(points[int(ids[0])], dtype=np.float64)
    primary = np.asarray(points[int(ids[1])] - origin, dtype=np.float64)
    plane = np.asarray(points[int(ids[2])] - origin, dtype=np.float64)
    primary_length = float(np.linalg.norm(primary))
    if primary_length < 1.0e-8:
        raise ValueError("three-joint driver primary segment is degenerate")
    primary /= primary_length
    normal = np.cross(primary, plane)
    if float(np.linalg.norm(normal)) < 1.0e-8:
        return _segment_frame(origin, origin + primary, reference_x)
    normal /= float(np.linalg.norm(normal))
    transverse = np.cross(normal, primary)
    transverse /= max(float(np.linalg.norm(transverse)), 1.0e-10)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.stack((primary, transverse, normal), axis=1)
    out[:3, 3] = origin
    return out


def _endpoint_segment_delta(
    *,
    rest_a: np.ndarray,
    rest_b: np.ndarray,
    pose_a: np.ndarray,
    pose_b: np.ndarray,
    rest_reference_x: np.ndarray,
    proximal_delta: np.ndarray,
    distal_delta: np.ndarray | None = None,
    twist_alpha: float = 0.0,
) -> np.ndarray:
    """Rigid transform for a limb segment; no blended global translations."""
    if float(np.linalg.norm(rest_b - rest_a)) < 1.0e-8:
        raise ValueError("segment driver rest endpoints are degenerate")
    F0 = _segment_frame(rest_a, rest_b, rest_reference_x)
    reference_delta = np.asarray(proximal_delta, dtype=np.float64)
    if distal_delta is not None and float(twist_alpha) > 0.0:
        reference_delta = _interpolate_rigid(
            np.asarray(proximal_delta, dtype=np.float64),
            np.asarray(distal_delta, dtype=np.float64),
            float(twist_alpha),
        ).astype(np.float64)
    # Transport the *actual* transverse axis selected in the rest frame.  This
    # matters when the authored bind X axis is parallel to the segment and
    # _segment_frame had to choose a stable fallback.  Transporting the raw
    # authored axis would discard pure axial rotation in that case.
    F1 = _segment_frame(pose_a, pose_b, reference_delta[:3, :3] @ F0[:3, 0])
    return F1 @ np.linalg.inv(F0)


def _source_rest_local(asset: AnatomyRiggedAsset) -> np.ndarray:
    """Return the schema-v6 fitted target bind-local matrices."""
    stored = asset.target_bind_local
    if stored is not None and np.asarray(stored).shape == np.asarray(asset.target_bind_global).shape:
        return np.asarray(stored, dtype=np.float64)
    raise ValueError("schema-v6 source rig is missing target_bind_local")


def source_bone_driver_frames(
    asset: AnatomyRiggedAsset,
    pose_axis_angle: Any,
) -> np.ndarray:
    """Build the one authoritative SMPL-X controller frame per source bone."""
    if asset.source_bone_names is None:
        raise ValueError("source driver frames require a schema-v6 source rig")
    bone_count = len(asset.source_bone_names)
    modes = list(asset.source_bone_driver_types or [])
    if len(modes) != bone_count:
        raise ValueError("schema-v6 source rig is missing explicit driver modes")
    unknown_modes = sorted(set(modes) - set(SOURCE_DRIVER_MODES))
    if unknown_modes:
        raise ValueError(f"unknown source driver mode(s): {unknown_modes}")
    source_parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    bind = np.asarray(asset.target_bind_global, dtype=np.float64)
    a_ids = np.asarray(asset.source_bone_smplx_a, dtype=np.int64)
    b_ids = np.asarray(asset.source_bone_smplx_b, dtype=np.int64)
    blends = np.asarray(asset.source_bone_blend, dtype=np.float64)
    frame_joints = np.asarray(asset.source_bone_frame_joints, dtype=np.int64)
    expected_vector = (bone_count,)
    if (
        source_parents.shape != expected_vector
        or a_ids.shape != expected_vector
        or b_ids.shape != expected_vector
        or blends.shape != expected_vector
        or bind.shape != (bone_count, 4, 4)
        or frame_joints.shape != (bone_count, 3)
    ):
        raise ValueError("schema-v6 source driver metadata has invalid shapes")
    if (
        not np.all(np.isfinite(bind))
        or not np.allclose(
            bind[:, 3, :],
            np.asarray((0.0, 0.0, 0.0, 1.0)),
            atol=1.0e-6,
            rtol=0.0,
        )
        or np.any(np.abs(np.linalg.det(bind[:, :3, :3])) <= 1.0e-10)
    ):
        raise ValueError("source driver authored bind is invalid")
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
    joint_count = len(pose_global)
    if (
        np.any(a_ids < 0)
        or np.any(a_ids >= joint_count)
        or np.any(b_ids < 0)
        or np.any(b_ids >= joint_count)
        or not np.all(np.isfinite(blends))
        or np.any(blends < 0.0)
        or np.any(blends > 1.0)
    ):
        raise ValueError("source driver contains an unmapped or invalid SMPL-X joint")
    if (
        np.any(frame_joints < -1)
        or np.any(frame_joints >= joint_count)
        or not np.array_equal(frame_joints[:, 0], a_ids)
    ):
        raise ValueError("source driver contains invalid explicit frame joints")
    for bone, parent in enumerate(source_parents.tolist()):
        if int(parent) < -1 or int(parent) >= bone:
            raise ValueError(f"source bone parent {parent} for bone {bone} is not topological")
    if not np.all(np.isfinite(pose_global)) or not np.all(np.isfinite(rest_global)):
        raise ValueError("source driver joint frames must be finite")
    joint_delta = pose_global @ np.linalg.inv(rest_global)
    rest_points = np.asarray(asset.rest_joints, dtype=np.float64)
    pose_points = pose_global[:, :3, 3]
    frames = np.tile(np.eye(4, dtype=np.float64), (bone_count, 1, 1))
    for bone, mode in enumerate(modes):
        if mode == "bind_follow" and int(source_parents[bone]) >= 0:
            continue
        a = int(a_ids[bone])
        b = int(b_ids[bone])
        explicit = (
            np.all(frame_joints[bone] >= 0)
            and len(np.unique(frame_joints[bone])) == 3
        )
        if explicit:
            rest_frame = _three_joint_frame(
                rest_points,
                frame_joints[bone],
                bind[bone, :3, 0],
            )
            frames[bone] = _three_joint_frame(
                pose_points,
                frame_joints[bone],
                joint_delta[a, :3, :3] @ rest_frame[:3, 0],
            )
        elif mode in {"segment_root", "rigid_group", "twist"} and a != b:
            rest_frame = _segment_frame(
                rest_points[a],
                rest_points[b],
                bind[bone, :3, 0],
            )
            delta = _endpoint_segment_delta(
                rest_a=rest_points[a],
                rest_b=rest_points[b],
                pose_a=pose_points[a],
                pose_b=pose_points[b],
                rest_reference_x=bind[bone, :3, 0],
                proximal_delta=joint_delta[a],
                distal_delta=joint_delta[b],
                twist_alpha=float(blends[bone]) if mode == "twist" else 0.0,
            )
            frames[bone] = delta @ rest_frame
        elif mode in {"segment_root", "twist"}:
            raise ValueError(f"{mode} source driver {bone} has a degenerate joint mapping")
        else:
            frames[bone] = pose_global[a]
        if not np.all(np.isfinite(frames[bone])):
            raise ValueError(f"source driver frame {bone} is non-finite")
        rotation = frames[bone, :3, :3]
        if (
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6, rtol=0.0)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6, rtol=0.0)
        ):
            raise ValueError(f"source driver frame {bone} is not a proper rigid frame")
    return frames


def build_source_driver_coupling(asset: AnatomyRiggedAsset) -> np.ndarray:
    """Return C=inv(F_rest)@B_target for each independently driven bone."""
    frames = source_bone_driver_frames(asset, np.zeros((55, 3), dtype=np.float32))
    bind = np.asarray(asset.target_bind_global, dtype=np.float64)
    coupling = np.tile(np.eye(4, dtype=np.float64), (len(bind), 1, 1))
    for bone, mode in enumerate(asset.source_bone_driver_types or []):
        if mode == "bind_follow" and int(asset.source_bone_parents[bone]) >= 0:
            continue
        coupling[bone] = np.linalg.inv(frames[bone]) @ bind[bone]
    return coupling.astype(np.float32)


def with_source_driver_coupling(asset: AnatomyRiggedAsset) -> AnatomyRiggedAsset:
    """Return an asset whose persisted controller coupling matches its bind."""
    coupling = build_source_driver_coupling(asset)
    return type(asset)(**{**asset.__dict__, "source_driver_coupling": coupling})


def source_bone_skinning_transforms(
    asset: AnatomyRiggedAsset,
    pose_axis_angle: Any,
) -> np.ndarray:
    """Solve the source rig once, in parent-before-child local FK order.

    Schema-v6 stores one controller-to-bind coupling for every independently
    driven source bone.  Authored bind_follow children retain their exact
    Blender local bind.  No mesh-derived rebind or translation restoration is
    performed at runtime.
    """
    if asset.source_bone_names is None:
        raise ValueError("source bone transforms require an anatomy schema-v6 rig")
    driver_frames = source_bone_driver_frames(asset, pose_axis_angle)
    modes = list(asset.source_bone_driver_types or [])
    if len(modes) != len(asset.source_bone_names):
        raise ValueError("schema-v6 source rig is missing explicit driver modes")
    rest_global_bones = np.asarray(asset.target_bind_global, dtype=np.float64)
    rest_local_bones = _source_rest_local(asset)
    posed_global = np.empty_like(rest_global_bones)
    source_parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    coupling = np.asarray(
        asset.source_driver_coupling
        if asset.source_driver_coupling is not None
        else build_source_driver_coupling(asset),
        dtype=np.float64,
    )
    if coupling.shape != rest_global_bones.shape or not np.all(np.isfinite(coupling)):
        raise ValueError("schema-v6 source driver coupling is invalid")
    if (
        not np.allclose(
            coupling[:, 3, :],
            np.asarray((0.0, 0.0, 0.0, 1.0)),
            atol=1.0e-6,
            rtol=0.0,
        )
        or np.any(np.abs(np.linalg.det(coupling[:, :3, :3])) <= 1.0e-10)
    ):
        raise ValueError("schema-v6 source driver coupling is not invertible affine")
    for bi, mode in enumerate(modes):
        parent = int(source_parents[bi])
        if parent >= bi or parent < -1:
            raise ValueError(f"source bone parent {parent} for bone {bi} is not topological")
        if mode == "bind_follow" and parent >= 0:
            posed_global[bi] = posed_global[parent] @ rest_local_bones[bi]
            continue
        posed_global[bi] = driver_frames[bi] @ coupling[bi]
    inverse_bind = np.asarray(asset.runtime_inverse_bind, dtype=np.float64)
    if inverse_bind.shape != rest_global_bones.shape or not np.all(np.isfinite(inverse_bind)):
        raise ValueError("schema-v6 target inverse bind is invalid")
    if not np.allclose(
        rest_global_bones @ inverse_bind,
        np.eye(4, dtype=np.float64),
        atol=2.0e-6,
        rtol=0.0,
    ):
        raise ValueError("target inverse bind does not match the fitted bind")
    transforms = posed_global @ inverse_bind
    neutral_pose = pose_to_smplx55_axis_angle(pose_axis_angle)
    if not np.any(neutral_pose):
        if not np.allclose(posed_global, rest_global_bones, atol=2.0e-6, rtol=0.0):
            raise ValueError("neutral source driver coupling does not recover the fitted bind")
        transforms = np.tile(np.eye(4, dtype=np.float64), (len(rest_global_bones), 1, 1))
    return transforms.astype(np.float32)


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
        # Keep material constraints backend-independent.  The former early
        # return silently skipped tube ARAP on CUDA, so production validation
        # measured a different runtime from CPU tests.
        posed = _skin_vertices_cuda(asset, transforms, None)
    else:
        if asset.driver_indices is not None and asset.driver_weights is not None:
            selected = transforms[np.asarray(asset.driver_indices, dtype=np.int64)]
            blended = np.sum(
                selected
                * np.asarray(asset.driver_weights, dtype=np.float32)[
                    ..., None, None
                ],
                axis=1,
            )
            dqs_indices = np.asarray(asset.driver_indices, dtype=np.int64)
            dqs_weights = np.asarray(asset.driver_weights, dtype=np.float32)
        else:
            weights = _dense_asset_weights(asset)
            joint_count = min(transforms.shape[0], weights.shape[1])
            blended = np.matmul(
                weights[:, :joint_count],
                transforms[:joint_count].reshape(joint_count, 16),
            ).reshape(-1, 4, 4)
            from .rigged_asset import sparse_driver_weights

            dqs_indices, dqs_weights = sparse_driver_weights(
                weights[:, :joint_count]
            )
        homo = np.concatenate(
            [vertices, np.ones((vertices.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        posed = np.matmul(blended, homo[:, :, None])[:, :3, 0].astype(
            np.float32
        )
        soft_mask = _soft_tissue_vertex_mask(asset)
        if _dqs_requested() and np.any(soft_mask):
            posed[soft_mask] = _dual_quaternion_skin_numpy(
                vertices[soft_mask],
                dqs_indices[soft_mask],
                dqs_weights[soft_mask],
                transforms,
            )
    # Thin anatomy uses a pre-baked station translation field.  This is a
    # direct evaluation (no online SDF, graph solve or collision iteration),
    # and therefore cannot rotate/collapse a vessel cross-section at a weight
    # boundary.  Organs receive one polar-rigid transform per component.
    if (
        asset.source_bone_names is not None
        and not bool((asset.metadata or {}).get("disable_soft_follow", False))
    ):
        from .soft_follow import apply_regional_organ_follow, apply_station_pose_follow

        posed = apply_station_pose_follow(asset, transforms, posed)
        posed = apply_regional_organ_follow(asset, posed)
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
