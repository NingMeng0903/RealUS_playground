"""Conservative world-orientation continuity for poorly observed leg chains."""
from __future__ import annotations

import numpy as np


def leg_occlusion_weights(points: np.ndarray, min_confidence: float = 0.05) -> np.ndarray:
    """Return left ankle/foot, right ankle/foot weights from measured anchors.

    A missing knee or ankle leaves chain rotation underconstrained. This is
    an observation-availability flag, not a calibrated uncertainty estimate.
    """
    points = np.asarray(points)
    if points.shape != (67, 4):
        raise ValueError("Expected points [67,4]")
    valid = np.isfinite(points).all(axis=1) & (points[:, 3] > min_confidence)
    missing = [1.0 - float(valid[[13, 14]].mean()),
               1.0 - float(valid[[10, 11]].mean())]
    return np.repeat(np.asarray(missing, dtype=np.float32), 2)


def leg_world_rotations(rotations):
    """Compose native SMPL-X world rotations, retaining autograd to ancestors."""
    import torch

    if tuple(rotations.shape[-3:]) != (55, 3, 3):
        raise ValueError("Expected native SMPL-X rotations [...,55,3,3]")
    output = []
    for hip, knee, ankle, foot in ((1, 4, 7, 10), (2, 5, 8, 11)):
        value = rotations[..., 0, :, :] @ rotations[..., hip, :, :]
        value = value @ rotations[..., knee, :, :]
        value = value @ rotations[..., ankle, :, :]
        output.extend((value, value @ rotations[..., foot, :, :]))
    return torch.stack(output, dim=-3)


def occlusion_rotation_loss(current, previous_world, weights, *, dt_s: float):
    """Weak-chain prior only; complete observations receive exactly zero loss.

    The half-second horizon avoids vanishing regularization at the intended
    low-rate input. Longer source gaps weaken it; this never claims a held
    orientation is measured, or guarantees correct motion under occlusion.
    """
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be finite and positive")
    error = (leg_world_rotations(current) - previous_world).square().mean(dim=(-1, -2))
    scale = min(1.0, 0.5 / float(dt_s))
    return scale ** 2 * (error * weights).sum() / (weights > 0).sum().clamp_min(1)
