"""Skin anatomy with matrix LBS and isolated opt-in DQS for soft tissue."""

from __future__ import annotations

import os
import weakref
from typing import Any, Mapping

import numpy as np

from .coupled_joint_v8 import evaluate_coupled_rbf_response_v8
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
    real, dual = _dual_quaternion_blend_numpy(indices, weights, transforms)
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


def _dual_quaternion_blend_numpy(
    indices: np.ndarray,
    weights: np.ndarray,
    transforms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized real/dual quaternions for fixed material frames."""
    qr = _matrix_quaternions_numpy(transforms)
    translation = np.asarray(transforms, dtype=np.float64)[:, :3, 3]
    qd = 0.5 * np.concatenate(
        (-np.sum(translation * qr[:, 1:], axis=1, keepdims=True),
         qr[:, :1] * translation + np.cross(translation, qr[:, 1:])), axis=1
    )
    count = int(np.asarray(indices).shape[0])
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
    return real, dual


def dual_quaternion_material_transforms_numpy(
    indices: np.ndarray,
    weights: np.ndarray,
    transforms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one proper SE(3) frame per pre-baked material group."""
    real, dual = _dual_quaternion_blend_numpy(indices, weights, transforms)
    w, x, y, z = (real[:, column] for column in range(4))
    rotation = np.empty((len(real), 3, 3), dtype=np.float64)
    rotation[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotation[:, 0, 1] = 2.0 * (x * y - z * w)
    rotation[:, 0, 2] = 2.0 * (x * z + y * w)
    rotation[:, 1, 0] = 2.0 * (x * y + z * w)
    rotation[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotation[:, 1, 2] = 2.0 * (y * z - x * w)
    rotation[:, 2, 0] = 2.0 * (x * z - y * w)
    rotation[:, 2, 1] = 2.0 * (y * z + x * w)
    rotation[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    translation = 2.0 * (
        real[:, :1] * dual[:, 1:]
        - dual[:, :1] * real[:, 1:]
        + np.cross(real[:, 1:], dual[:, 1:])
    )
    return rotation.astype(np.float32), translation.astype(np.float32)


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
    # The trigonometry below already runs in double; only the storage was float32,
    # which cost ~1e-7 of orthogonality per joint before it was ever accumulated.
    # Honour a float64 input so callers that chain these matrices can stay exact.
    dtype = (
        np.float64
        if np.asarray(axis_angle).dtype == np.float64
        else np.float32
    )
    rows = np.asarray(axis_angle, dtype=dtype).reshape(-1, 3)
    out = np.tile(np.eye(3, dtype=dtype), (rows.shape[0], 1, 1))
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
            dtype=dtype,
        )
    return out


def joint_global_transforms(
    *,
    pose_axis_angle: Any,
    rest_joints: np.ndarray,
    parents: np.ndarray,
) -> np.ndarray:
    pose = pose_to_smplx55_axis_angle(pose_axis_angle)
    joints = np.asarray(rest_joints, dtype=np.float64).reshape(-1, 3)
    pa = np.asarray(parents, dtype=np.int32).reshape(-1)
    n = min(int(joints.shape[0]), int(pa.shape[0]), int(pose.shape[0]))
    # Accumulate in float64. A finger sits about ten products deep in this chain,
    # and in float32 the orthogonality drift reached 1.04e-6 on a captured pose --
    # just past the 1e-6 rigid-frame guard downstream, so the runtime hard-failed
    # on a legitimate pose depending on where the drift happened to land.
    rot = axis_angle_to_matrix(np.asarray(pose[:n], dtype=np.float64))
    out = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    for idx in range(n):
        local = np.eye(4, dtype=np.float64)
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
    reference_x = np.asarray(proximal_delta, dtype=np.float64)[:3, :3] @ F0[:3, 0]
    if distal_delta is not None and float(twist_alpha) > 0.0:
        # A twist follower may interpolate roll about the limb axis, but must
        # not inherit distal flexion.  Slerping the complete knee/ankle or
        # elbow/wrist rotations turns foot/hand swing into a bend of the long
        # bone and opens authored surface contacts at the distal epiphysis.
        axis = np.asarray(pose_b - pose_a, dtype=np.float64)
        axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
        distal_x = np.asarray(distal_delta, dtype=np.float64)[:3, :3] @ F0[:3, 0]
        proximal_projected = reference_x - float(reference_x @ axis) * axis
        distal_projected = distal_x - float(distal_x @ axis) * axis
        proximal_norm = float(np.linalg.norm(proximal_projected))
        distal_norm = float(np.linalg.norm(distal_projected))
        if proximal_norm > 1.0e-8 and distal_norm > 1.0e-8:
            proximal_projected /= proximal_norm
            distal_projected /= distal_norm
            angle = float(
                np.arctan2(
                    axis @ np.cross(proximal_projected, distal_projected),
                    np.clip(proximal_projected @ distal_projected, -1.0, 1.0),
                )
            )
            partial = float(np.clip(twist_alpha, 0.0, 1.0)) * angle
            # Rodrigues rotation around the posed anatomical segment axis.
            reference_x = (
                np.cos(partial) * proximal_projected
                + np.sin(partial) * np.cross(axis, proximal_projected)
                + (1.0 - np.cos(partial))
                * float(axis @ proximal_projected)
                * axis
            )
    # Transport the *actual* transverse axis selected in the rest frame.  This
    # matters when the authored bind X axis is parallel to the segment and
    # _segment_frame had to choose a stable fallback.  Transporting the raw
    # authored axis would discard pure axial rotation in that case.
    F1 = _segment_frame(pose_a, pose_b, reference_x)
    return F1 @ np.linalg.inv(F0)


def _source_rest_local(asset: AnatomyRiggedAsset) -> np.ndarray:
    """Return the schema-v6 fitted target bind-local matrices."""
    stored = asset.target_bind_local
    if stored is not None and np.asarray(stored).shape == np.asarray(asset.target_bind_global).shape:
        return np.asarray(stored, dtype=np.float64)
    raise ValueError("schema-v6 source rig is missing target_bind_local")


def _smoothstep01(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _assert_proper_rotation(
    matrix: np.ndarray,
    label: str,
    *,
    atol: float = 1.0e-9,
) -> np.ndarray:
    """Fail closed unless ``matrix`` is a proper orthonormal rotation."""
    rotation = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(rotation)):
        raise ValueError(f"{label} is non-finite")
    determinant = float(np.linalg.det(rotation))
    if determinant <= 0.0:
        raise ValueError(f"{label} is not a proper rotation (det <= 0)")
    if not np.isclose(determinant, 1.0, atol=max(atol, 1.0e-6), rtol=0.0):
        raise ValueError(f"{label} determinant {determinant} is not near 1")
    if not np.allclose(
        rotation.T @ rotation, np.eye(3, dtype=np.float64), atol=atol, rtol=0.0
    ):
        raise ValueError(f"{label} is not orthonormal to {atol}")
    return rotation


def _as_proper_rotation(matrix: np.ndarray, label: str) -> np.ndarray:
    """Nearest proper rotation for authored/driver inputs (SVD polar factor)."""
    rotation = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(rotation)):
        raise ValueError(f"{label} is non-finite")
    try:
        return _assert_proper_rotation(rotation, label, atol=1.0e-6)
    except ValueError:
        u, _singular, vt = np.linalg.svd(rotation)
        projected = u @ vt
        if float(np.linalg.det(projected)) < 0.0:
            u = u.copy()
            u[:, -1] *= -1.0
            projected = u @ vt
        return _assert_proper_rotation(projected, label, atol=1.0e-9)


def _normalize_vector(vector: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{label} is non-finite")
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-12:
        raise ValueError(f"{label} is degenerate")
    return value / norm


def _gram_schmidt_perp(axis: np.ndarray, primary: np.ndarray, label: str) -> np.ndarray:
    """Return the unit component of ``axis`` orthogonal to unit ``primary``."""
    residual = np.asarray(axis, dtype=np.float64).reshape(3) - (
        float(np.dot(axis, primary)) * primary
    )
    return _normalize_vector(residual, label)


def _orthonormal_frame(primary: np.ndarray, secondary: np.ndarray, label: str) -> np.ndarray:
    """Build columns ``[primary, secondary_perp, primary x secondary_perp]``."""
    first = _normalize_vector(primary, f"{label} primary")
    second = _gram_schmidt_perp(secondary, first, f"{label} secondary")
    third = np.cross(first, second)
    third = _normalize_vector(third, f"{label} third")
    second = np.cross(third, first)
    frame = np.stack((first, second, third), axis=1)
    return _assert_proper_rotation(frame, f"{label} frame")


def _slerp_rotation(ra: np.ndarray, rb: np.ndarray, alpha: float) -> np.ndarray:
    """Geodesic interpolation from ``ra`` to ``rb`` (quaternion slerp)."""
    from scipy.spatial.transform import Rotation

    t = float(np.clip(alpha, 0.0, 1.0))
    ra = _as_proper_rotation(ra, "slerp ra")
    rb = _as_proper_rotation(rb, "slerp rb")
    delta = Rotation.from_matrix(ra.T @ rb).as_rotvec()
    out = ra @ Rotation.from_rotvec(t * delta).as_matrix()
    return _as_proper_rotation(out, "slerp result")


def _validate_leg_hinge_solve_entry_v1(
    entry: Mapping[str, Any],
    *,
    side: str,
    bone_count: int,
    joint_count: int,
) -> dict[str, Any]:
    """Fail-closed parse of one ``source_leg_hinge_solve_v1`` side entry."""
    if not isinstance(entry, Mapping):
        raise ValueError(f"source_leg_hinge_solve_v1[{side!r}] must be a mapping")
    required = (
        "femur_bone",
        "knee_bone",
        "ankle_bone",
        "smplx_hip",
        "smplx_knee",
        "smplx_ankle",
        "hinge_axis_femur_local",
        "femoral_head_femur_local",
        "femoral_head_vertex_indices",
        "hinge_axis_sign",
        "blend_lo_deg",
        "blend_hi_deg",
    )
    missing = [name for name in required if name not in entry]
    if missing:
        raise ValueError(
            f"source_leg_hinge_solve_v1[{side!r}] missing fields: {missing}"
        )
    femur_bone = int(entry["femur_bone"])
    knee_bone = int(entry["knee_bone"])
    ankle_bone = int(entry["ankle_bone"])
    hip_j = int(entry["smplx_hip"])
    knee_j = int(entry["smplx_knee"])
    ankle_j = int(entry["smplx_ankle"])
    axis = np.asarray(entry["hinge_axis_femur_local"], dtype=np.float64).reshape(-1)
    head_local = np.asarray(
        entry["femoral_head_femur_local"], dtype=np.float64
    ).reshape(-1)
    hinge_sign = int(entry["hinge_axis_sign"])
    blend_lo = float(entry["blend_lo_deg"])
    blend_hi = float(entry["blend_hi_deg"])
    if (
        femur_bone < 0
        or femur_bone >= bone_count
        or knee_bone < 0
        or knee_bone >= bone_count
        or ankle_bone < 0
        or ankle_bone >= bone_count
        or len({femur_bone, knee_bone, ankle_bone}) != 3
    ):
        raise ValueError(
            f"source_leg_hinge_solve_v1[{side!r}] has invalid bone indices"
        )
    if (
        hip_j < 0
        or knee_j < 0
        or ankle_j < 0
        or hip_j >= joint_count
        or knee_j >= joint_count
        or ankle_j >= joint_count
        or len({hip_j, knee_j, ankle_j}) != 3
    ):
        raise ValueError(
            f"source_leg_hinge_solve_v1[{side!r}] has invalid SMPL-X joints"
        )
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise ValueError(
            f"source_leg_hinge_solve_v1[{side!r}] hinge axis is invalid"
        )
    if head_local.shape != (3,) or not np.all(np.isfinite(head_local)):
        raise ValueError(
            f"source_leg_hinge_solve_v1[{side!r}] femoral head local is invalid"
        )
    head_indices = np.asarray(
        entry.get("femoral_head_vertex_indices", []), dtype=np.int64
    ).reshape(-1)
    if head_indices.size < 4 or np.any(head_indices < 0):
        raise ValueError(
            f"source_leg_hinge_solve_v1[{side!r}] femoral head indices are invalid"
        )
    if hinge_sign not in (-1, 1):
        raise ValueError(
            f"source_leg_hinge_solve_v1[{side!r}] hinge_axis_sign must be ±1"
        )
    if not np.isfinite(blend_lo) or not np.isfinite(blend_hi) or blend_hi <= blend_lo:
        raise ValueError(
            f"source_leg_hinge_solve_v1[{side!r}] blend thresholds are invalid"
        )
    axis = _normalize_vector(axis, f"source_leg_hinge_solve_v1[{side!r}] hinge axis")
    return {
        "femur_bone": femur_bone,
        "knee_bone": knee_bone,
        "ankle_bone": ankle_bone,
        "smplx_hip": hip_j,
        "smplx_knee": knee_j,
        "smplx_ankle": ankle_j,
        "hinge_axis_femur_local": axis,
        "femoral_head_femur_local": np.asarray(head_local, dtype=np.float64),
        "femoral_head_vertex_indices": head_indices.astype(np.int64),
        "hinge_axis_sign": hinge_sign,
        "blend_lo_deg": blend_lo,
        "blend_hi_deg": blend_hi,
    }


def solve_leg_hinge_v1(
    *,
    hip: np.ndarray,
    knee: np.ndarray,
    ankle: np.ndarray,
    bind_hip: np.ndarray,
    bind_knee: np.ndarray,
    bind_ankle: np.ndarray | None = None,
    bind_femur_rotation: np.ndarray,
    hinge_axis_femur_local: np.ndarray,
    driver_femur_rotation: np.ndarray,
    blend_lo_deg: float = 5.0,
    blend_hi_deg: float = 15.0,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Two-segment leg IK with a single authored knee hinge.

    Returns ``(femur_posed_rotation, theta_applied_rad, theta_raw_rad,
    hinge_axis_world)``.

    The leg is exactly determined: the femur is a 3-DoF ball joint pinned at the
    acetabulum and the knee is a 1-DoF hinge about the *authored* axis, giving
    four unknowns for the four constraints imposed by the SMPL-X knee and ankle
    directions.  So both drive targets are reachable without ever bending the
    knee off its anatomical axis:

    1. Two femur DoF put the femur long axis on the posed hip→knee ray.
    2. The remaining femur DoF — twist ``phi`` about that ray — swings the
       authored hinge until the SMPL-X shank lies on the cone the hinge can
       reach, i.e. until ``dot(shank, hinge) `` equals its bind value.  Solved in
       closed form and taken on the branch nearest the driver, so ``phi`` is the
       least twist that makes the ankle reachable.
    3. The knee hinge angle ``theta`` then lands the shank on the SMPL-X ankle
       exactly.

    ``theta`` is signed so positive flexes the tibia posteriorly (the authored
    ``hinge_axis_femur_local`` already carries that orientation).  Straight is 0;
    hyperextension (``theta_raw < 0``) is clamped to 0 for ``theta_applied`` and
    callers should record the clamp.

    ``phi`` tracks the azimuth of the drive shank about the femur axis, so its
    sensitivity scales as ``1 / sin(flexion)``: near the bind pose a degree of
    drive demands several degrees of twist, and twisting the femur that far drags
    the trochlea off the patella.  ``phi`` is therefore faded in with a smoothstep
    over how far the leg has flexed *away from its bind angle*
    (``[blend_lo_deg, blend_hi_deg]``), keeping the driver attitude near bind and
    absorbing the axis mismatch only in deep flexion where the sensitivity is
    bounded.  Measuring that excursion from a straight leg instead would leave the
    fade permanently open, because the bind femur axis and shank are already
    16.4 deg apart on the left and 10.3 deg on the right.

    If the drive asks for a shank the hinge cannot reach at any twist, ``phi``
    takes the closest approach rather than failing, and the unreachable remainder
    shows up as ankle error downstream.
    """
    from scipy.spatial.transform import Rotation

    H = np.asarray(hip, dtype=np.float64).reshape(3)
    K = np.asarray(knee, dtype=np.float64).reshape(3)
    A = np.asarray(ankle, dtype=np.float64).reshape(3)
    H0 = np.asarray(bind_hip, dtype=np.float64).reshape(3)
    K0 = np.asarray(bind_knee, dtype=np.float64).reshape(3)
    R_bind = _as_proper_rotation(bind_femur_rotation, "bind femur rotation")
    R_driver = _as_proper_rotation(driver_femur_rotation, "driver femur rotation")
    h0_local = _normalize_vector(
        hinge_axis_femur_local, "hinge axis femur-local"
    )
    d0 = _normalize_vector(K0 - H0, "bind femur direction")
    h0 = _normalize_vector(R_bind @ h0_local, "bind hinge world")
    if bind_ankle is None:
        # No authored shank: assume the ideal knee whose shank is perpendicular
        # to its hinge.
        s0 = _normalize_vector(
            np.cross(h0, np.cross(d0, h0)), "bind shank fallback"
        )
    else:
        s0 = _normalize_vector(
            np.asarray(bind_ankle, dtype=np.float64).reshape(3) - K0,
            "bind shank direction",
        )
    # The hinge preserves this cone half-angle between shank and axis.
    cone = float(np.dot(s0, h0))

    femur_vec = K - H
    thigh_len = float(np.linalg.norm(femur_vec))
    if thigh_len <= 1.0e-12:
        raise ValueError("leg hinge solve has a degenerate posed femur")
    d = femur_vec / thigh_len
    bone_len = float(np.linalg.norm(K0 - H0))
    if bone_len <= 1.0e-12:
        raise ValueError("leg hinge solve has a degenerate bind femur")
    # Rigid bone knee lands on the SMPL-X femur ray at the authored length.
    K_bone = H + bone_len * d
    shank_from_bone = A - K_bone
    shank_from_bone_len = float(np.linalg.norm(shank_from_bone))
    if shank_from_bone_len <= 1.0e-12:
        raise ValueError("leg hinge solve has a degenerate bone→ankle segment")
    shank_dir = shank_from_bone / shank_from_bone_len

    flex_deg = float(
        np.degrees(np.arccos(float(np.clip(np.dot(d, shank_dir), -1.0, 1.0))))
    )
    # Flexion has to be measured against the bind leg, not against a straight
    # one. This anatomy's bind femur axis and shank already sit 16.4 deg apart on
    # the left and 10.3 deg on the right, so an absolute flex_deg starts above the
    # fade band and the fade is wide open at the bind pose: the twist then arrives
    # at full strength on the first degree of drive.
    bind_flex_deg = float(
        np.degrees(np.arccos(float(np.clip(np.dot(d0, s0), -1.0, 1.0))))
    )
    flex_excursion_deg = abs(flex_deg - bind_flex_deg)

    # Driver attitude with the long axis locked onto the posed femur ray; this
    # is the phi = 0 reference, so direction_error stays ~0 at every flexion.
    d_driver = _normalize_vector(
        R_driver @ R_bind.T @ d0, "driver femur direction"
    )
    align_axis = np.cross(d_driver, d)
    align_norm = float(np.linalg.norm(align_axis))
    if align_norm <= 1.0e-12:
        R_driver_aligned = R_driver
    else:
        align = Rotation.from_rotvec(
            align_axis
            * (
                float(np.arctan2(align_norm, float(np.dot(d_driver, d))))
                / align_norm
            )
        ).as_matrix()
        R_driver_aligned = _as_proper_rotation(
            align @ R_driver, "driver direction aligned"
        )

    # Twist phi about d that brings the authored hinge onto the shank cone:
    #   dot(shank, Rot(d, phi) @ h_driver) = cone
    # Splitting the hinge into its axial and radial parts about d turns this
    # into P*cos(phi) + Q*sin(phi) = r.
    h_driver = _normalize_vector(
        R_driver_aligned @ h0_local, "driver hinge world"
    )
    axial = float(np.dot(h_driver, d))
    radial = h_driver - axial * d
    shank_axial = float(np.dot(shank_dir, d))
    shank_radial = shank_dir - shank_axial * d
    P = float(np.dot(shank_radial, radial))
    Q = float(np.dot(shank_radial, np.cross(d, radial)))
    r = cone - axial * shank_axial
    amplitude = float(np.hypot(P, Q))
    if amplitude <= 1.0e-12:
        phi = 0.0
    else:
        psi = float(np.arctan2(Q, P))
        ratio = float(np.clip(r / amplitude, -1.0, 1.0))
        offset = float(np.arccos(ratio))
        # Two branches; take the one that twists the femur least.
        candidates = [
            float(np.remainder(psi + sign * offset + np.pi, 2.0 * np.pi) - np.pi)
            for sign in (1.0, -1.0)
        ]
        phi = min(candidates, key=abs)
    if flex_excursion_deg < float(blend_hi_deg):
        phi *= _smoothstep01(
            (flex_excursion_deg - float(blend_lo_deg))
            / (float(blend_hi_deg) - float(blend_lo_deg))
        )

    R_femur = _as_proper_rotation(
        Rotation.from_rotvec(d * phi).as_matrix() @ R_driver_aligned,
        "hinge femur rotation",
    )
    hinge_world = _normalize_vector(R_femur @ h0_local, "posed hinge world")

    # Knee angle taking the femur-carried bind shank onto the posed shank.
    carried = _normalize_vector(
        R_femur @ R_bind.T @ s0, "carried shank direction"
    )
    u = carried - float(np.dot(carried, hinge_world)) * hinge_world
    v = shank_dir - float(np.dot(shank_dir, hinge_world)) * hinge_world
    if float(np.linalg.norm(u)) <= 1.0e-12 or float(np.linalg.norm(v)) <= 1.0e-12:
        theta_raw = 0.0
    else:
        theta_raw = float(
            np.arctan2(
                float(np.dot(hinge_world, np.cross(u, v))),
                float(np.dot(u, v)),
            )
        )
    theta_applied = float(max(theta_raw, 0.0))
    return R_femur, theta_applied, theta_raw, hinge_world


def _femur_pose_about_head(
    *,
    parent_global: np.ndarray,
    bind_local: np.ndarray,
    femur_rotation_world: np.ndarray,
    head_femur_local: np.ndarray,
    head_target_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build posed local/global femur transforms rotating about the head centre.

    The fitted femoral-head centre stays at ``head_target_world``; the bone
    origin is placed at ``head_target - R @ head_local``.
    """
    R = _as_proper_rotation(femur_rotation_world, "femur world rotation")
    head_local = np.asarray(head_femur_local, dtype=np.float64).reshape(3)
    head_target = np.asarray(head_target_world, dtype=np.float64).reshape(3)
    origin = head_target - R @ head_local
    posed_global = np.eye(4, dtype=np.float64)
    posed_global[:3, :3] = R
    posed_global[:3, 3] = origin
    posed_local = np.linalg.inv(parent_global) @ posed_global
    # Preserve the bind local scale/homogeneous row; only rotation+origin change.
    del bind_local  # rotation about head fully replaces the bind local pose
    return posed_local, posed_global



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
    # Official SMPL-X joints remain the spatial authority for long-bone
    # segments.  Subject-fitted contact points are only rotation centres for
    # terminal rigid/joint-local controllers; using them as segment endpoints
    # changes effective femur/tibia length and breaks cross-beta reuse.
    rest_points = np.asarray(asset.rest_joints, dtype=np.float64)
    pose_points = pose_global[:, :3, 3]
    contact_rest_points = np.asarray(
        asset.source_driver_rest_joints
        if asset.source_driver_rest_joints is not None
        else rest_points,
        dtype=np.float64,
    )
    use_anatomical_guide_fk_v810 = bool(
        (asset.metadata or {}).get("source_anatomical_guide_fk_v810", False)
    )
    guide_pose_points: np.ndarray | None = None
    if use_anatomical_guide_fk_v810:
        if asset.source_driver_rest_joints is None:
            raise ValueError(
                "source_anatomical_guide_fk_v810 requires driver rest joints"
            )
        guide_pose_global = joint_global_transforms(
            pose_axis_angle=pose_axis_angle,
            rest_joints=contact_rest_points,
            parents=asset.parents,
        ).astype(np.float64)
        guide_pose_points = guide_pose_global[:, :3, 3]
    contact_point_delta = joint_delta.copy()
    for joint, parent in enumerate(
        np.asarray(asset.parents, dtype=np.int64).tolist()
    ):
        if int(parent) >= 0:
            contact_point_delta[joint] = joint_delta[int(parent)]
    contact_pose_points = (
        np.einsum(
            "bij,bj->bi",
            contact_point_delta[:, :3, :3],
            contact_rest_points,
        )
        + contact_point_delta[:, :3, 3]
    )
    if guide_pose_points is not None:
        contact_pose_points = guide_pose_points
    use_anatomical_pivots_v7 = bool(
        (asset.metadata or {}).get("source_anatomical_pivots_v7", False)
        or use_anatomical_guide_fk_v810
    )
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
            segment_rest_a = rest_points[a]
            segment_rest_b = rest_points[b]
            segment_pose_a = pose_points[a]
            segment_pose_b = pose_points[b]
            if use_anatomical_pivots_v7 and mode in {
                "segment_root",
                "rigid_group",
            }:
                # The SMPL-X joint pair remains the orientation/length driver,
                # while the beta-specific socket is the physical rotation
                # centre.  The old path rotated a femur around the statistical
                # SMPL-X hip several centimetres away from the acetabulum.
                segment_rest_a = contact_rest_points[a]
                segment_pose_a = contact_pose_points[a]
                if use_anatomical_guide_fk_v810:
                    segment_rest_b = contact_rest_points[b]
                    segment_pose_b = contact_pose_points[b]
                else:
                    segment_rest_b = segment_rest_a + (
                        rest_points[b] - rest_points[a]
                    )
                    segment_pose_b = segment_pose_a + (
                        pose_points[b] - pose_points[a]
                    )
            rest_frame = _segment_frame(
                segment_rest_a,
                segment_rest_b,
                bind[bone, :3, 0],
            )
            delta = _endpoint_segment_delta(
                rest_a=segment_rest_a,
                rest_b=segment_rest_b,
                pose_a=segment_pose_a,
                pose_b=segment_pose_b,
                rest_reference_x=bind[bone, :3, 0],
                proximal_delta=joint_delta[a],
                distal_delta=joint_delta[b],
                twist_alpha=float(blends[bone]) if mode == "twist" else 0.0,
            )
            if mode == "rigid_group":
                delta[:3, 3] = (
                    contact_pose_points[a]
                    - delta[:3, :3] @ contact_rest_points[a]
                )
            frames[bone] = delta @ rest_frame
        elif mode in {"segment_root", "twist"}:
            raise ValueError(f"{mode} source driver {bone} has a degenerate joint mapping")
        else:
            frames[bone] = pose_global[a]
            frames[bone, :3, 3] = contact_pose_points[a]
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


def source_bone_posed_global(
    asset: AnatomyRiggedAsset,
    pose_axis_angle: Any,
) -> np.ndarray:
    """Solve posed global source-bone matrices once, parent-before-child.

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
    use_source_local_fk = bool(
        (asset.metadata or {}).get("source_joint_local_fk_v1", False)
    )
    use_full_source_local_fk = bool(
        (asset.metadata or {}).get("source_full_local_fk_v2", False)
    )
    use_connected_source_local_fk = bool(
        (asset.metadata or {}).get("source_connected_local_fk_v3", False)
    )
    selected_source_local_fk = {
        int(value)
        for value in (asset.metadata or {}).get("source_local_fk_bones_v3", [])
    }
    if any(
        value < 0 or value >= len(rest_global_bones)
        for value in selected_source_local_fk
    ):
        raise ValueError("source_local_fk_bones_v3 contains an invalid bone")
    source_use_connect = np.asarray(
        asset.source_bone_use_connect
        if asset.source_bone_use_connect is not None
        else np.zeros(len(rest_global_bones), dtype=np.uint8),
        dtype=bool,
    )
    direct_driver_bones = {
        int(value)
        for value in (asset.metadata or {}).get("source_direct_driver_bones_v1", [])
    }
    if any(value < 0 or value >= len(rest_global_bones) for value in direct_driver_bones):
        raise ValueError("source_direct_driver_bones_v1 contains an invalid bone")
    corrective_driver = np.asarray(
        asset.source_bone_corrective_driver
        if asset.source_bone_corrective_driver is not None
        else np.full(len(rest_global_bones), -1, dtype=np.int32),
        dtype=np.int64,
    )
    corrective_gain = np.asarray(
        asset.source_bone_corrective_gain
        if asset.source_bone_corrective_gain is not None
        else np.zeros(len(rest_global_bones), dtype=np.float32),
        dtype=np.float64,
    )
    corrective_axis = np.asarray(
        asset.source_bone_corrective_axis
        if asset.source_bone_corrective_axis is not None
        else np.zeros((len(rest_global_bones), 3), dtype=np.float32),
        dtype=np.float64,
    )
    corrective_input_axis_raw = (asset.metadata or {}).get(
        "source_corrective_input_axes_v1"
    )
    corrective_input_axis = None
    if corrective_input_axis_raw is not None:
        corrective_input_axis = np.asarray(
            corrective_input_axis_raw, dtype=np.float64
        )
        if corrective_input_axis.shape != (len(rest_global_bones), 3):
            raise ValueError(
                "source_corrective_input_axes_v1 has an invalid shape"
            )
        if not np.all(np.isfinite(corrective_input_axis)):
            raise ValueError(
                "source_corrective_input_axes_v1 contains non-finite values"
            )
    use_corrective_rigid_blend = bool(
        (asset.metadata or {}).get("source_corrective_rigid_blend_v2", False)
    )
    knee_hinge_splines = dict(
        (asset.metadata or {}).get("source_knee_hinge_splines_v7", {})
    )
    tibia_glide_splines = dict(
        (asset.metadata or {}).get("source_tibia_glide_splines_v7", {})
    )
    patella_splines = dict(
        (asset.metadata or {}).get("source_patella_splines_v7", {})
    )
    # The v7 spline drove the patella relative to the femur with the V71
    # tibia-local gain, so the patella under-rotated by roughly a factor of
    # four and its tibia-local translation moved by tens of millimetres in
    # deep flexion.  The v8 response restores the authored chain: the patella
    # is an ordinary child of Tibia_Bone with a rotation-only local driver.
    patella_responses = dict(
        (asset.metadata or {}).get("source_patella_v71_response_v8", {})
    )
    legacy_ankle_roll_glide = dict(
        (asset.metadata or {}).get("source_ankle_roll_glide_v8", {})
    )
    if legacy_ankle_roll_glide:
        raise ValueError(
            "source_ankle_roll_glide_v8 is obsolete: independent-axis "
            "translation sums are not valid for composite rotations"
        )
    coupled_joint_responses = dict(
        (asset.metadata or {}).get("source_coupled_joint_response_v8", {})
    )
    input_pose = pose_to_smplx55_axis_angle(pose_axis_angle).astype(np.float64)
    target_pose_global = joint_global_transforms(
        pose_axis_angle=pose_axis_angle,
        rest_joints=asset.rest_joints,
        parents=asset.parents,
    ).astype(np.float64)
    target_rest_global = joint_global_transforms(
        pose_axis_angle=np.zeros((55, 3), dtype=np.float32),
        rest_joints=asset.rest_joints,
        parents=asset.parents,
    ).astype(np.float64)
    target_joint_delta = target_pose_global @ np.linalg.inv(target_rest_global)
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
    # Per-side hinge-constrained leg solve.  Provenance lives in
    # ``source_leg_hinge_solve_v1``; malformed entries fail closed.
    leg_solve_raw = (asset.metadata or {}).get("source_leg_hinge_solve_v1", {})
    if leg_solve_raw is None:
        leg_solve_raw = {}
    if not isinstance(leg_solve_raw, dict):
        raise ValueError("source_leg_hinge_solve_v1 must be a mapping")
    leg_femur_rotation: dict[int, np.ndarray] = {}
    leg_femur_head_local: dict[int, np.ndarray] = {}
    leg_femur_head_target: dict[int, np.ndarray] = {}
    leg_knee_theta: dict[int, float] = {}
    leg_knee_theta_raw: dict[int, float] = {}
    leg_knee_axis_world: dict[int, np.ndarray] = {}
    leg_flexion_by_joint: dict[int, float] = {}
    pose_points = target_pose_global[:, :3, 3]
    joint_count = int(pose_points.shape[0])
    # Anatomical contact pivots (socket / knee pivot) posed by the same parent
    # deltas as ``source_bone_driver_frames``.  The femur origin lives on these
    # points, not on the raw SMPL-X joint centres (which can sit several cm off).
    contact_rest_points = np.asarray(
        asset.source_driver_rest_joints
        if asset.source_driver_rest_joints is not None
        else asset.rest_joints,
        dtype=np.float64,
    )
    use_anatomical_guide_fk_v810 = bool(
        (asset.metadata or {}).get("source_anatomical_guide_fk_v810", False)
    )
    guide_pose_points: np.ndarray | None = None
    if use_anatomical_guide_fk_v810:
        if asset.source_driver_rest_joints is None:
            raise ValueError(
                "source_anatomical_guide_fk_v810 requires driver rest joints"
            )
        guide_pose_global = joint_global_transforms(
            pose_axis_angle=pose_axis_angle,
            rest_joints=contact_rest_points,
            parents=asset.parents,
        ).astype(np.float64)
        guide_pose_points = guide_pose_global[:, :3, 3]
    contact_point_delta = target_joint_delta.copy()
    for joint, parent_j in enumerate(
        np.asarray(asset.parents, dtype=np.int64).tolist()
    ):
        if int(parent_j) >= 0:
            contact_point_delta[joint] = target_joint_delta[int(parent_j)]
    contact_pose_points = (
        np.einsum(
            "bij,bj->bi",
            contact_point_delta[:, :3, :3],
            contact_rest_points,
        )
        + contact_point_delta[:, :3, 3]
    )
    if guide_pose_points is not None:
        contact_pose_points = guide_pose_points
    # Zero pose must recover the fitted bind exactly.  Rest-joint H/K/A are not
    # perfectly colinear, so a geometric hinge solve would invent a nonzero
    # femur twist on the neutral sample; skip it when the SMPL-X pose is zero.
    run_leg_solve = bool(np.any(input_pose))
    if run_leg_solve:
        for side, raw_entry in leg_solve_raw.items():
            entry = _validate_leg_hinge_solve_entry_v1(
                raw_entry,
                side=str(side),
                bone_count=len(rest_global_bones),
                joint_count=joint_count,
            )
            femur_bone = int(entry["femur_bone"])
            knee_bone = int(entry["knee_bone"])
            ankle_bone = int(entry["ankle_bone"])
            hip_j = int(entry["smplx_hip"])
            knee_j = int(entry["smplx_knee"])
            ankle_j = int(entry["smplx_ankle"])
            if guide_pose_points is not None:
                hip = guide_pose_points[hip_j]
                knee = guide_pose_points[knee_j]
                ankle = guide_pose_points[ankle_j]
            else:
                # Legacy anatomical-pivot mode keeps the socket origin but
                # inherits raw SMPL-X segment vectors.
                hip = contact_pose_points[hip_j]
                knee = hip + (pose_points[knee_j] - pose_points[hip_j])
                ankle = knee + (pose_points[ankle_j] - pose_points[knee_j])
            driver_desired = driver_frames[femur_bone] @ coupling[femur_bone]
            R_femur, theta, theta_raw, hinge_axis_world = solve_leg_hinge_v1(
                hip=hip,
                knee=knee,
                ankle=ankle,
                bind_hip=rest_global_bones[femur_bone, :3, 3],
                bind_knee=rest_global_bones[knee_bone, :3, 3],
                bind_ankle=rest_global_bones[ankle_bone, :3, 3],
                bind_femur_rotation=rest_global_bones[femur_bone, :3, :3],
                hinge_axis_femur_local=entry["hinge_axis_femur_local"],
                driver_femur_rotation=driver_desired[:3, :3],
                blend_lo_deg=float(entry["blend_lo_deg"]),
                blend_hi_deg=float(entry["blend_hi_deg"]),
            )
            if femur_bone in leg_femur_rotation or knee_bone in leg_knee_theta:
                raise ValueError(
                    f"source_leg_hinge_solve_v1[{side!r}] reuses a femur/knee bone"
                )
            if knee_j in leg_flexion_by_joint:
                raise ValueError(
                    f"source_leg_hinge_solve_v1[{side!r}] reuses SMPL-X knee {knee_j}"
                )
            leg_femur_rotation[femur_bone] = R_femur
            leg_femur_head_local[femur_bone] = np.asarray(
                entry["femoral_head_femur_local"], dtype=np.float64
            )
            # Head target is the posed socket / contact hip (bind femur origin
            # was placed on the acetabulum centre at reconstruction).
            leg_femur_head_target[femur_bone] = np.asarray(hip, dtype=np.float64)
            leg_knee_theta[knee_bone] = float(theta)
            leg_knee_theta_raw[knee_bone] = float(theta_raw)
            leg_knee_axis_world[knee_bone] = np.asarray(
                hinge_axis_world, dtype=np.float64
            )
            # Signed anatomical flexion (clamped); patella/tibia use this with
            # the same below-zero policy as PatellaOracleLawV7.response_rad.
            leg_flexion_by_joint[knee_j] = float(theta)
    else:
        # Still fail-closed on malformed provenance even for the neutral sample.
        for side, raw_entry in leg_solve_raw.items():
            _validate_leg_hinge_solve_entry_v1(
                raw_entry,
                side=str(side),
                bone_count=len(rest_global_bones),
                joint_count=joint_count,
            )
    for bi, mode in enumerate(modes):
        parent = int(source_parents[bi])
        if parent >= bi or parent < -1:
            raise ValueError(f"source bone parent {parent} for bone {bi} is not topological")
        if bi in leg_femur_rotation:
            if parent < 0:
                raise ValueError(f"leg hinge femur {bi} has no source parent")
            _posed_local, posed_global[bi] = _femur_pose_about_head(
                parent_global=posed_global[parent],
                bind_local=rest_local_bones[bi],
                femur_rotation_world=leg_femur_rotation[bi],
                head_femur_local=leg_femur_head_local[bi],
                head_target_world=leg_femur_head_target[bi],
            )
            continue
        response = patella_responses.get(str(bi))
        if response is not None:
            # Placed ahead of bind_follow on purpose: the authored patella is a
            # driven child, and silently treating it as a rigid follower is the
            # failure that let the femur sweep through the patella.
            if parent < 0:
                raise ValueError(f"V7 patella response {bi} has no source parent")
            joint = int(response.get("smplx_joint", -1))
            axis = np.asarray(response.get("axis_local", []), dtype=np.float64)
            knots = np.radians(
                np.asarray(response.get("knots_deg", []), dtype=np.float64)
            )
            angles = np.radians(
                np.asarray(response.get("response_deg", []), dtype=np.float64)
            )
            translations = np.asarray(
                response.get("translation_parent_local_m", []), dtype=np.float64
            )
            if (
                joint < 0
                or joint >= len(input_pose)
                or axis.shape != (3,)
                or not np.all(np.isfinite(axis))
                or knots.ndim != 1
                or len(knots) < 2
                or angles.shape != knots.shape
                or translations.shape != (len(knots), 3)
                or np.any(np.diff(knots) <= 0.0)
            ):
                raise ValueError(f"V7 patella response {bi} has invalid coefficients")
            axis = axis / max(float(np.linalg.norm(axis)), 1.0e-12)
            if joint in leg_flexion_by_joint:
                flexion = float(leg_flexion_by_joint[joint])
            else:
                flexion = float(np.linalg.norm(input_pose[joint]))
            # Match PatellaOracleLawV7.response_rad: clamp hyperextension to 0,
            # then look up the baked signed response (no extra re-signing).
            flexion_lookup = float(max(flexion, 0.0))
            angle = float(np.interp(flexion_lookup, knots, angles))
            translation = np.asarray(
                [
                    np.interp(flexion_lookup, knots, translations[:, axis_index])
                    for axis_index in range(3)
                ],
                dtype=np.float64,
            )
            maximum = float(response.get("maximum_translation_m", 0.005))
            if float(np.linalg.norm(translation)) > maximum + 1.0e-7:
                raise ValueError(f"V7 patella response {bi} exceeds its baked bound")
            posed_local = np.asarray(rest_local_bones[bi], dtype=np.float64).copy()
            posed_local[:3, 3] += translation
            posed_local[:3, :3] = (
                posed_local[:3, :3] @ axis_angle_to_matrix(axis * angle)[0]
            )
            posed_global[bi] = posed_global[parent] @ posed_local
            continue
        tibia_glide = tibia_glide_splines.get(str(bi))
        if tibia_glide is not None:
            if parent < 0:
                raise ValueError(f"V7 tibia glide {bi} has no source parent")
            joint = int(tibia_glide.get("smplx_joint", -1))
            knots = np.radians(
                np.asarray(tibia_glide.get("knots_deg", []), dtype=np.float64)
            )
            translations = np.asarray(
                tibia_glide.get("translation_parent_local_m", []),
                dtype=np.float64,
            )
            if (
                joint < 0
                or joint >= len(input_pose)
                or knots.ndim != 1
                or len(knots) < 2
                or translations.shape != (len(knots), 3)
                or np.any(np.diff(knots) <= 0.0)
            ):
                raise ValueError(f"V7 tibia glide {bi} has invalid coefficients")
            if joint in leg_flexion_by_joint:
                flexion = float(max(float(leg_flexion_by_joint[joint]), 0.0))
            else:
                flexion = float(np.linalg.norm(input_pose[joint]))
            translation = np.asarray(
                [
                    np.interp(flexion, knots, translations[:, axis_index])
                    for axis_index in range(3)
                ],
                dtype=np.float64,
            )
            maximum = float(tibia_glide.get("maximum_translation_m", 0.0005))
            if float(np.linalg.norm(translation)) > maximum + 1.0e-7:
                raise ValueError(f"V7 tibia glide {bi} exceeds its baked bound")
            posed_local = np.asarray(rest_local_bones[bi], dtype=np.float64).copy()
            posed_local[:3, 3] += translation
            posed_global[bi] = posed_global[parent] @ posed_local
            continue
        if mode == "bind_follow" and parent >= 0:
            posed_global[bi] = posed_global[parent] @ rest_local_bones[bi]
            continue
        patella = patella_splines.get(str(bi))
        if patella is not None:
            reference_bone = int(patella.get("reference_bone", -1))
            joint = int(patella.get("smplx_joint", -1))
            if (
                reference_bone < 0
                or reference_bone >= bi
                or joint < 0
                or joint >= len(input_pose)
            ):
                raise ValueError(f"V7 patella spline {bi} has invalid drivers")
            axis = np.asarray(
                patella.get("axis_reference_local", []), dtype=np.float64
            )
            pivot = np.asarray(
                patella.get("pivot_reference_local_m", []), dtype=np.float64
            )
            if (
                axis.shape != (3,)
                or pivot.shape != (3,)
                or not np.all(np.isfinite(axis))
                or not np.all(np.isfinite(pivot))
            ):
                raise ValueError(f"V7 patella spline {bi} has invalid frame")
            axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
            knots = np.radians(
                np.asarray(patella.get("knots_deg", []), dtype=np.float64)
            )
            response = np.radians(
                np.asarray(patella.get("response_deg", []), dtype=np.float64)
            )
            translations = np.asarray(
                patella.get("translation_reference_local_m", []),
                dtype=np.float64,
            )
            if (
                knots.ndim != 1
                or response.shape != knots.shape
                or translations.shape != (len(knots), 3)
                or len(knots) < 2
                or np.any(np.diff(knots) <= 0.0)
            ):
                raise ValueError(f"V7 patella spline {bi} has invalid knots")
            if joint in leg_flexion_by_joint:
                flexion = float(max(float(leg_flexion_by_joint[joint]), 0.0))
            else:
                flexion = float(np.linalg.norm(input_pose[joint]))
            angle = float(np.interp(flexion, knots, response))
            translation = np.asarray(
                [
                    np.interp(flexion, knots, translations[:, axis_index])
                    for axis_index in range(3)
                ]
            )
            rotation = axis_angle_to_matrix(axis * angle)[0]
            delta = np.eye(4, dtype=np.float64)
            delta[:3, :3] = rotation
            delta[:3, 3] = pivot - rotation @ pivot + translation
            relative_bind = (
                np.linalg.inv(rest_global_bones[reference_bone])
                @ rest_global_bones[bi]
            )
            posed_global[bi] = (
                posed_global[reference_bone] @ delta @ relative_bind
            )
            continue
        corrective_source = int(corrective_driver[bi])
        if corrective_source >= 0:
            if parent < 0 or corrective_source >= len(rest_global_bones):
                raise ValueError(f"source corrective {bi} has an invalid hierarchy")
            joint = int(asset.source_bone_smplx_a[corrective_source])
            if joint < 0 or joint >= len(input_pose):
                raise ValueError(f"source corrective {bi} has an invalid SMPL-X driver")
            if use_corrective_rigid_blend:
                upper = int(source_parents[corrective_source])
                if upper < 0:
                    raise ValueError(
                        f"source corrective {bi} has no proximal rigid driver"
                    )
                upper_delta = (
                    posed_global[upper] @ np.linalg.inv(rest_global_bones[upper])
                )
                lower_delta = (
                    posed_global[parent] @ np.linalg.inv(rest_global_bones[parent])
                )
                lower_fraction = float(
                    np.clip(1.0 - float(corrective_gain[bi]), 0.0, 1.0)
                )
                blended_delta = _interpolate_rigid(
                    upper_delta, lower_delta, lower_fraction
                )
                corrective_origin = rest_global_bones[bi, :3, 3]
                distal_origin = (
                    lower_delta[:3, :3] @ corrective_origin
                    + lower_delta[:3, 3]
                )
                blended_delta[:3, 3] = (
                    distal_origin
                    - blended_delta[:3, :3] @ corrective_origin
                )
                posed_global[bi] = blended_delta @ rest_global_bones[bi]
                continue
            axis = corrective_axis[bi].copy()
            axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
            if corrective_input_axis is None:
                if joint in leg_flexion_by_joint:
                    flexion = float(max(float(leg_flexion_by_joint[joint]), 0.0))
                else:
                    flexion = float(np.linalg.norm(input_pose[joint]))
            else:
                input_axis = corrective_input_axis[bi].copy()
                input_axis_norm = float(np.linalg.norm(input_axis))
                if input_axis_norm < 1.0e-8:
                    raise ValueError(
                        f"source corrective {bi} has a degenerate input axis"
                    )
                input_axis /= input_axis_norm
                driver_parent = int(source_parents[corrective_source])
                if driver_parent >= 0:
                    driver_local = (
                        np.linalg.inv(posed_global[driver_parent])
                        @ posed_global[corrective_source]
                    )
                else:
                    driver_local = posed_global[corrective_source]
                driver_basis = (
                    np.linalg.inv(
                        rest_local_bones[corrective_source, :3, :3]
                    )
                    @ driver_local[:3, :3]
                )
                from scipy.spatial.transform import Rotation

                driver_rotvec = Rotation.from_matrix(
                    driver_basis
                ).as_rotvec()
                flexion = float(driver_rotvec @ input_axis)
            correction = np.eye(4, dtype=np.float64)
            correction[:3, :3] = axis_angle_to_matrix(
                axis * (float(corrective_gain[bi]) * flexion)
            )[0]
            posed_global[bi] = (
                posed_global[parent] @ rest_local_bones[bi] @ correction
            )
            continue
        hinge = knee_hinge_splines.get(str(bi))
        if hinge is not None:
            if parent < 0:
                raise ValueError(f"V7 knee hinge {bi} has no source parent")
            joint = int(hinge.get("smplx_joint", -1))
            if joint < 0 or joint >= len(input_pose):
                raise ValueError(f"V7 knee hinge {bi} has an invalid SMPL-X joint")
            axis = np.asarray(hinge.get("axis_local", []), dtype=np.float64)
            if axis.shape != (3,) or not np.all(np.isfinite(axis)):
                raise ValueError(f"V7 knee hinge {bi} has an invalid local axis")
            axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
            knots = np.radians(
                np.asarray(hinge.get("knots_deg", []), dtype=np.float64)
            )
            response = np.radians(
                np.asarray(hinge.get("response_deg", []), dtype=np.float64)
            )
            if (
                knots.ndim != 1
                or response.shape != knots.shape
                or len(knots) < 2
                or np.any(np.diff(knots) <= 0.0)
            ):
                raise ValueError(f"V7 knee hinge {bi} has an invalid spline")
            if bi in leg_knee_theta:
                # Anatomical flexion already clamped to >= 0 by the solve.
                theta = float(leg_knee_theta[bi])
                flexion = float(max(theta, 0.0))
                # Identity response spline: applied angle equals anatomical flexion.
                angle = float(np.interp(flexion, knots, response))
                # The IK solved theta about the authored hinge carried by the
                # posed femur; rotate about exactly that axis so the shank lands
                # on the drive.  It must agree with the baked spline axis —
                # a mismatch means the two bakes disagree on the hinge.
                solved = _normalize_vector(
                    (
                        posed_global[parent][:3, :3]
                        @ rest_local_bones[bi, :3, :3]
                    ).T
                    @ leg_knee_axis_world[bi],
                    f"V7 knee hinge {bi} solved axis",
                )
                if float(np.dot(solved, axis)) < np.cos(np.radians(1.0)):
                    raise ValueError(
                        f"V7 knee hinge {bi} spline axis disagrees with the "
                        "leg solve hinge axis"
                    )
                axis = solved
            elif joint in leg_flexion_by_joint:
                flexion = float(max(float(leg_flexion_by_joint[joint]), 0.0))
                angle = float(np.interp(flexion, knots, response))
            else:
                flexion = float(np.linalg.norm(input_pose[joint]))
                angle = float(np.interp(flexion, knots, response))
            translation_knots = np.asarray(
                hinge.get("translation_local_m", []), dtype=np.float64
            )
            if translation_knots.shape != (len(knots), 3):
                raise ValueError(
                    f"V7 knee hinge {bi} has invalid translation coefficients"
                )
            translation = np.asarray(
                [
                    np.interp(flexion, knots, translation_knots[:, axis_index])
                    for axis_index in range(3)
                ],
                dtype=np.float64,
            )
            correction = np.eye(4, dtype=np.float64)
            correction[:3, :3] = axis_angle_to_matrix(axis * angle)[0]
            correction[:3, 3] = translation
            posed_global[bi] = (
                posed_global[parent] @ rest_local_bones[bi] @ correction
            )
            continue
        coupled_response = coupled_joint_responses.get(str(bi))
        if coupled_response is not None:
            if parent < 0:
                raise ValueError(f"V8 coupled response {bi} has no source parent")
            joint = int(coupled_response.get("smplx_joint", -1))
            pivot_bind = np.asarray(
                coupled_response.get("anatomical_pivot_target_bind_m", []),
                dtype=np.float64,
            )
            pivot_local = np.asarray(
                coupled_response.get("anatomical_pivot_parent_local_m", []),
                dtype=np.float64,
            )
            if joint < 0 or joint >= len(input_pose):
                raise ValueError(
                    f"V8 coupled response {bi} has an invalid SMPL-X joint"
                )
            if (
                coupled_response.get("pivot_mapping")
                != "smplx_axis_angle_state_to_frozen_anatomical_parent_local"
                or pivot_bind.shape != (3,)
                or pivot_local.shape != (3,)
                or not np.all(np.isfinite(pivot_bind))
                or not np.all(np.isfinite(pivot_local))
                or not np.allclose(
                    pivot_bind,
                    rest_global_bones[bi, :3, 3],
                    atol=2.0e-6,
                    rtol=0.0,
                )
                or not np.allclose(
                    pivot_local,
                    rest_local_bones[bi, :3, 3],
                    atol=2.0e-6,
                    rtol=0.0,
                )
            ):
                raise ValueError(
                    f"V8 coupled response {bi} has an invalid anatomical pivot mapping"
                )
            rotvec = np.asarray(input_pose[joint], dtype=np.float64)
            translation = evaluate_coupled_rbf_response_v8(
                coupled_response,
                rotvec,
            )
            desired = driver_frames[bi] @ coupling[bi]
            posed_local = np.asarray(rest_local_bones[bi], dtype=np.float64).copy()
            posed_local[:3, :3] = (
                np.linalg.inv(posed_global[parent, :3, :3])
                @ desired[:3, :3]
            )
            posed_local[:3, 3] += translation
            posed_global[bi] = posed_global[parent] @ posed_local
            continue
        if bi in direct_driver_bones:
            posed_global[bi] = driver_frames[bi] @ coupling[bi]
            continue
        if (
            parent >= 0
            and (
                use_full_source_local_fk
                or (
                    use_connected_source_local_fk
                    and bool(source_use_connect[bi])
                )
                or bi in selected_source_local_fk
            )
        ):
            # Driver coupling supplies world rotation; Blender's fitted local
            # bind remains authoritative for pivots and local translations.
            desired = driver_frames[bi] @ coupling[bi]
            posed_local = np.asarray(rest_local_bones[bi], dtype=np.float64).copy()
            posed_local[:3, :3] = (
                np.linalg.inv(posed_global[parent, :3, :3])
                @ desired[:3, :3]
            )
            posed_global[bi] = posed_global[parent] @ posed_local
            continue
        if mode == "joint_local" and parent >= 0 and use_source_local_fk:
            joint = int(asset.source_bone_smplx_a[bi])
            if joint < 0 or joint >= len(target_joint_delta):
                raise ValueError(f"joint-local source bone {bi} has an invalid SMPL-X joint")
            # Preserve the already validated target-global orientation
            # retarget, but recover the bone origin through source FK.  This
            # keeps the source local translation and mapped pivot without
            # assuming that SMPL-X and Blender use the same local bone axes.
            desired_global_rotation = (
                target_joint_delta[joint, :3, :3]
                @ rest_global_bones[bi, :3, :3]
            )
            posed_local = np.asarray(rest_local_bones[bi], dtype=np.float64).copy()
            posed_local[:3, :3] = (
                np.linalg.inv(posed_global[parent, :3, :3])
                @ desired_global_rotation
            )
            posed_global[bi] = posed_global[parent] @ posed_local
            continue
        posed_global[bi] = driver_frames[bi] @ coupling[bi]
    return posed_global.astype(np.float64)


def source_bone_skinning_transforms(
    asset: AnatomyRiggedAsset,
    pose_axis_angle: Any,
) -> np.ndarray:
    """Solve the source rig once, in parent-before-child local FK order.

    Schema-v6 stores one controller-to-bind coupling for every independently
    driven source bone.  Authored bind_follow children retain their exact
    Blender local bind.  No mesh-derived rebind or translation restoration is
    performed at runtime.

    The femur is skinned by the same transform that carries its children.  An
    earlier revision skinned it with the SMPL-X driver rotation while the knee
    pivot followed the hinge solve; the condyles then sat up to 54 mm away from
    the tibial plateau they articulate with.
    """
    posed_global = source_bone_posed_global(asset, pose_axis_angle)
    rest_global_bones = np.asarray(asset.target_bind_global, dtype=np.float64)
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
    input_pose = pose_to_smplx55_axis_angle(pose_axis_angle).astype(np.float64)
    transforms = np.asarray(posed_global, dtype=np.float64) @ inverse_bind
    if not np.any(input_pose):
        if not np.allclose(posed_global, rest_global_bones, atol=2.0e-6, rtol=0.0):
            raise ValueError("neutral source driver coupling does not recover the fitted bind")
        transforms = np.tile(np.eye(4, dtype=np.float64), (len(rest_global_bones), 1, 1))
    return transforms.astype(np.float32)


def skin_vertices(
    asset: AnatomyRiggedAsset,
    pose_axis_angle: Any,
    *,
    transl: Any | None = None,
    runtime_coefficients: dict[str, np.ndarray] | None = None,
    runtime_tube_pack: Any | None = None,
    runtime_tube_pack_validated: bool = False,
    runtime_tube_pose_corrective_pack: Any | None = None,
    runtime_tube_pose_corrective_pack_validated: bool = False,
    validate: bool = True,
) -> np.ndarray:
    if validate:
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
        homo = np.concatenate(
            [vertices, np.ones((vertices.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        if asset.driver_indices is not None and asset.driver_weights is not None:
            dqs_indices = np.asarray(asset.driver_indices, dtype=np.int64)
            dqs_weights = np.asarray(asset.driver_weights, dtype=np.float32)
            # Authored anatomy is overwhelmingly rigid: on the V71 asset about
            # 81% of vertices have exactly one non-zero Armature influence.
            # Gathering [N,14,4,4] for those rows wasted hundreds of MB and
            # dominated resident pose latency.  Keep the exact matrix-LBS
            # result while evaluating the single-weight rows directly.
            single = (
                np.abs(dqs_weights[:, 0] - 1.0) <= 1.0e-7
            ) & np.all(np.abs(dqs_weights[:, 1:]) <= 1.0e-12, axis=1)
            posed = np.empty((len(vertices), 3), dtype=np.float32)
            if np.any(single):
                direct = transforms[dqs_indices[single, 0], :3, :]
                posed[single] = np.matmul(
                    direct, homo[single, :, None]
                )[:, :, 0]
            multiple = ~single
            if np.any(multiple):
                selected = transforms[dqs_indices[multiple]]
                blended_multiple = np.sum(
                    selected
                    * dqs_weights[multiple, ..., None, None],
                    axis=1,
                )
                posed[multiple] = np.matmul(
                    blended_multiple[:, :3, :],
                    homo[multiple, :, None],
                )[:, :, 0]
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
    tube_coupling_applied = False
    if runtime_tube_pack is not None:
        from .tube_frames_v8 import apply_tube_coupling_v8

        posed = apply_tube_coupling_v8(
            asset,
            transforms,
            posed,
            runtime_tube_pack,
            runtime_fields=runtime_coefficients,
            validate_live=not bool(runtime_tube_pack_validated),
        )
        tube_coupling_applied = True
    elif runtime_coefficients and any(
        str(name).startswith("tube_coupling_v8.")
        for name in runtime_coefficients
    ):
        from .tube_frames_v8 import (
            apply_tube_coupling_v8,
            tube_coupling_pack_from_runtime_fields_v8,
        )

        pack = tube_coupling_pack_from_runtime_fields_v8(runtime_coefficients)
        posed = apply_tube_coupling_v8(
            asset,
            transforms,
            posed,
            pack,
            runtime_fields=runtime_coefficients,
        )
        tube_coupling_applied = True
    elif runtime_coefficients:
        has_corrective = any(
            str(name).startswith("tube_pose_corrective_v1.")
            for name in runtime_coefficients
        )
        if has_corrective:
            raise ValueError(
                "tube_pose_corrective_v1 requires authoritative tube_coupling_v8"
            )
        from .tube_frames_v7 import apply_tube_material_frames_v7

        posed = apply_tube_material_frames_v7(
            asset,
            transforms,
            posed,
            runtime_coefficients,
        )
    if runtime_tube_pose_corrective_pack is not None:
        if not tube_coupling_applied:
            raise ValueError(
                "tube_pose_corrective_v1 requires authoritative tube_coupling_v8"
            )
        if asset.driver_indices is None or asset.driver_weights is None:
            raise ValueError(
                "tube_pose_corrective_v1 requires original 14-slot Armature weights"
            )
        from .tube_pose_corrective_v8 import apply_tube_pose_corrective_v1

        posed = apply_tube_pose_corrective_v1(
            posed,
            runtime_tube_pose_corrective_pack,
            pose_axis_angle=pose_axis_angle,
            source_transforms=transforms,
            driver_indices=asset.driver_indices,
            driver_weights=asset.driver_weights,
            validate_pack=not bool(runtime_tube_pose_corrective_pack_validated),
        )
    elif runtime_coefficients and any(
        str(name).startswith("tube_pose_corrective_v1.")
        for name in runtime_coefficients
    ):
        if not tube_coupling_applied:
            raise ValueError(
                "tube_pose_corrective_v1 requires authoritative tube_coupling_v8"
            )
        if asset.driver_indices is None or asset.driver_weights is None:
            raise ValueError(
                "tube_pose_corrective_v1 requires original 14-slot Armature weights"
            )
        from .tube_pose_corrective_v8 import (
            apply_tube_pose_corrective_v1,
            tube_pose_corrective_pack_from_runtime_fields_v1,
        )

        corrective = tube_pose_corrective_pack_from_runtime_fields_v1(
            runtime_coefficients
        )
        posed = apply_tube_pose_corrective_v1(
            posed,
            corrective,
            pose_axis_angle=pose_axis_angle,
            source_transforms=transforms,
            driver_indices=asset.driver_indices,
            driver_weights=asset.driver_weights,
        )
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
