from __future__ import annotations

import numpy as np

from projects.genesis_ue_sync.sim_platform.human_motion.contracts import ActionBlock
from projects.genesis_ue_sync.sim_platform.human_motion.refit.refitters import HamiltonianLossWeights


HAND_JOINT_SLICES: tuple[slice, ...] = (slice(3 + 20 * 3, 3 + 22 * 3),)
SPINE_JOINT_SLICES: tuple[slice, ...] = (slice(3 + 3 * 3, 3 + 6 * 3), slice(3 + 9 * 3, 3 + 13 * 3))


def tracking_weight_vector(pose_dim: int, weights: HamiltonianLossWeights) -> np.ndarray:
    """Build per-axis tracking weights for SMPL axis-angle rows."""

    out = np.full((int(pose_dim),), float(weights.tracking_limbs), dtype=np.float32)
    if pose_dim >= 3:
        out[:3] = float(weights.tracking_global_orient)
    for sl in SPINE_JOINT_SLICES:
        out[sl] = float(weights.tracking_spine)
    for sl in HAND_JOINT_SLICES:
        out[sl] = float(weights.tracking_hands)
    return out


def contact_damping_scale(action_blocks: tuple[ActionBlock, ...], frame_count: int, fps: float) -> np.ndarray:
    """Convert semantic contact hints into a per-frame damping multiplier."""

    out = np.ones((int(frame_count),), dtype=np.float32)
    if frame_count <= 0 or fps <= 1e-6:
        return out
    for block in action_blocks:
        mask = block.contact_mask.normalized()
        push = max(mask.get("left_elbow_push", 0.0), mask.get("right_elbow_push", 0.0), mask.get("left_palm_support", 0.0), mask.get("right_palm_support", 0.0))
        start = max(int(round(float(block.start_time_s) * fps)), 0)
        end = min(int(round((float(block.start_time_s) + float(block.duration_s)) * fps)), frame_count)
        if end > start:
            out[start:end] *= float(1.0 + 0.5 * push)
    return out


def temporal_smooth_energy(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape[0] < 3:
        return 0.0
    accel = arr[2:] - 2.0 * arr[1:-1] + arr[:-2]
    return float(np.mean(accel * accel))


def symmetry_energy(poses: np.ndarray, left_indices: tuple[int, ...], right_indices: tuple[int, ...]) -> float:
    arr = np.asarray(poses, dtype=np.float32)
    if len(left_indices) != len(right_indices) or not left_indices:
        return 0.0
    vals = []
    for li, ri in zip(left_indices, right_indices):
        ls = slice(3 + int(li) * 3, 3 + int(li + 1) * 3)
        rs = slice(3 + int(ri) * 3, 3 + int(ri + 1) * 3)
        if ls.stop <= arr.shape[1] and rs.stop <= arr.shape[1]:
            vals.append(arr[:, ls] - arr[:, rs])
    if not vals:
        return 0.0
    diff = np.concatenate(vals, axis=1)
    return float(np.mean(diff * diff))


def vposer_prior_energy(poses: np.ndarray, adapter: object | None) -> float:
    """Return the VPoser latent energy for diagnostics; unavailable adapters return zero."""

    if adapter is None or not bool(getattr(adapter, "available", False)):
        return 0.0
    try:
        import torch

        with torch.no_grad():
            pose_tensor = torch.as_tensor(np.asarray(poses, dtype=np.float32))
            loss = adapter.prior_loss(pose_tensor)
            return float(loss.detach().cpu())
    except Exception:
        return 0.0
