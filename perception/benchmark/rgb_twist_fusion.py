"""Circular consensus of camera-independent native-joint twist predictions.

Cross-view agreement is a consistency check, not a calibrated probability of
correctness. In the official analytical IK, multi-child nodes (spine3, head,
wrists) use child-point orientation instead of their corresponding phi.
"""
from __future__ import annotations

import numpy as np


def fuse_twist_predictions(phis, fallback, *, min_resultant: float = 0.75):
    values = np.asarray(phis, dtype=np.float64)
    prior = np.asarray(fallback, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (54, 2) or values.shape[0] < 1:
        raise ValueError('Expected per-view twist [V,54,2]')
    if prior.shape != (54, 2) or not np.isfinite(prior).all():
        raise ValueError('Expected finite fallback twist [54,2]')
    if not 0 <= min_resultant <= 1:
        raise ValueError('min_resultant must be in [0,1]')
    prior_norm = np.linalg.norm(prior, axis=-1)
    if np.any(prior_norm < 1e-8):
        raise ValueError('Fallback twists must have nonzero norm')
    norm = np.linalg.norm(values, axis=-1)
    valid = np.isfinite(values).all(axis=-1) & (norm > 1e-8)
    clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    unit = np.where(valid[..., None], clean / np.maximum(norm[..., None], 1e-8), 0.0)
    count = valid.sum(axis=0)
    mean = unit.sum(axis=0) / np.maximum(count[:, None], 1)
    resultant = np.linalg.norm(mean, axis=-1)
    # At least two cameras must independently supply finite predictions.
    accepted = (count >= 2) & (resultant >= min_resultant) & (resultant > 1e-8)
    consensus = mean / np.maximum(resultant[:, None], 1e-8)
    fused = np.where(accepted[:, None], consensus, prior / prior_norm[:, None])
    return fused.astype(np.float32), accepted, resultant.astype(np.float32)
