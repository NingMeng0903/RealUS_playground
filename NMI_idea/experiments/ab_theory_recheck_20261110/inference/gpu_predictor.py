"""Optional CUDA float64 predictor matching observation.environment.

The implementation is intentionally a predictor-only adapter.  It does not
touch the production environment, renderer, prior, or proposal.  Geometry is
computed in float64 on CUDA and all public fields are copied back to NumPy so
the existing likelihood function is unchanged.  Callers must compare this
adapter with ``env.predict_batch`` before enabling it for an experiment.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

import numpy as np


LADDER_ROOT = Path(
    "/media/camp/EXT_DRIVE/RealUS_playground/NMI_idea/experiments/"
    "ab_module_ladder_20261110"
)
if str(LADDER_ROOT) not in sys.path:
    sys.path.insert(0, str(LADDER_ROOT))
from observation import environment as env  # noqa: E402


def available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _geometry_batch_torch(points: np.ndarray, poses: np.ndarray, torch):
    """Torch float64 translation of environment._geometry_batch."""
    device = torch.device("cuda")
    pts = torch.as_tensor(points, dtype=torch.float64, device=device)
    pose = torch.as_tensor(np.atleast_2d(poses), dtype=torch.float64, device=device)
    dx = pts[:, None, :, :, 0] - pose[None, :, None, None, 0]
    dy = pts[:, None, :, :, 1] - pose[None, :, None, None, 1]
    c = torch.cos(pose[:, 2])[None, :, None, None]
    s = torch.sin(pose[:, 2])[None, :, None, None]
    local = torch.stack(
        (
            c * dx + s * dy,
            -s * dx + c * dy,
            torch.broadcast_to(-pts[:, None, :, :, 2], dx.shape),
        ),
        dim=-1,
    )
    a, b = local[..., :-1, :], local[..., 1:, :]
    delta = b - a
    low = torch.tensor([-0.025, -0.002, 0.015], dtype=torch.float64, device=device)
    high = torch.tensor([0.025, 0.002, 0.070], dtype=torch.float64, device=device)
    safe = torch.where(delta.abs() < 1e-15, torch.ones_like(delta), delta)
    t0, t1 = (low - a) / safe, (high - a) / safe
    parallel = delta.abs() < 1e-15
    outside = torch.any(parallel & ((a < low) | (a > high)), dim=-1)
    lower = torch.where(parallel, torch.full_like(t0, -torch.inf), torch.minimum(t0, t1))
    upper = torch.where(parallel, torch.full_like(t0, torch.inf), torch.maximum(t0, t1))
    lo = torch.maximum(torch.zeros_like(lower[..., 0]), lower.amax(dim=-1))
    hi = torch.minimum(torch.ones_like(upper[..., 0]), upper.amin(dim=-1))
    valid = (hi > lo + 1e-12) & ~outside
    lo = lo.clamp(0.0, 1.0)
    hi = hi.clamp(0.0, 1.0)
    ca = a + lo[..., None] * delta
    cb = a + hi[..., None] * delta
    l3 = torch.linalg.vector_norm(cb - ca, dim=-1) * valid
    connected_prev = torch.zeros_like(valid)
    connected_prev[..., 1:] = (
        valid[..., 1:]
        & valid[..., :-1]
        & (lo[..., 1:] < 1e-9)
        & (hi[..., :-1] > 1.0 - 1e-9)
    )
    breaks = valid & ~connected_prev
    labels = torch.cumsum(breaks.to(torch.int64), dim=-1)
    total = torch.cumsum(l3, dim=-1)
    before = total - l3
    reset = torch.cummax(torch.where(breaks, before, torch.zeros_like(before)), dim=-1).values
    run = total - reset
    run = torch.where(valid, run, torch.zeros_like(run))
    best_idx = run.argmax(dim=-1)
    best_label = torch.gather(labels, -1, best_idx[..., None]).squeeze(-1)
    keep = valid & (labels == best_label[..., None])
    weight = l3 * keep
    length = weight.sum(dim=-1)
    denom = torch.clamp(length, min=1e-15)
    mid = (ca + cb) * 0.5
    centroid = (weight[..., None] * mid).sum(dim=-2) / denom[..., None]
    pa = torch.stack((ca[..., 0], ca[..., 2]), dim=-1)
    pb = torch.stack((cb[..., 0], cb[..., 2]), dim=-1)
    lp = torch.linalg.vector_norm(pb - pa, dim=-1) * keep
    plen = lp.sum(dim=-1)
    pden = torch.clamp(plen, min=1e-15)
    pm = (pa + pb) * 0.5
    e2 = (
        (pa[..., :, None] * pa[..., None, :] + pb[..., :, None] * pb[..., None, :])
        / 3.0
        + (pa[..., :, None] * pb[..., None, :] + pb[..., :, None] * pa[..., None, :])
        / 6.0
    )
    pc = (lp[..., None] * pm).sum(dim=-2) / pden[..., None]
    cov = (lp[..., None, None] * e2).sum(dim=-3) / pden[..., None, None] - pc[..., :, None] * pc[..., None, :]
    axis = 0.5 * torch.atan2(2.0 * cov[..., 0, 1], cov[..., 0, 0] - cov[..., 1, 1])
    segnorm = torch.clamp(torch.linalg.vector_norm(delta, dim=-1), min=1e-15)
    sine = (delta[..., 1].abs() / segnorm).clamp(0.0, 1.0)
    angle = torch.sqrt((weight * torch.arcsin(sine) ** 2).sum(dim=-1) / denom)
    exists = length > 0.0005
    return {
        "u": pc[..., 0],
        "depth": pc[..., 1],
        "length": plen,
        "geometric_length": length,
        "axis": axis,
        "angle_audit": angle,
        "exists": exists,
    }


def _cpu_array(value, torch):
    return value.detach().cpu().numpy()


def predict_batch(particles: Sequence, poses: np.ndarray, *, device: str = "cuda") -> dict:
    """Return all fields of ``env.predict_batch`` using CUDA float64."""
    import torch

    if device != "cuda":
        raise ValueError("gpu_predictor only supports device='cuda'")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not particles:
        raise ValueError("particles must be nonempty")
    pose = np.atleast_2d(np.asarray(poses, dtype=float))
    points = np.asarray([p.points for p in particles], dtype=np.float64)
    g = _geometry_batch_torch(points, pose, torch)
    detect_z = torch.as_tensor(
        np.asarray([p.detect_z for p in particles]), dtype=torch.float64, device="cuda"
    )
    identity_z = torch.as_tensor(
        np.asarray([p.identity_z for p in particles]), dtype=torch.float64, device="cuda"
    )
    detect = torch.sigmoid(2.8 + detect_z)[:, None, :]
    source_q = detect * g["exists"] * (g["length"] / 0.004).clamp(0.0, 1.0)
    pany = 1.0 - torch.prod(1.0 - source_q, dim=-1)
    strength = source_q * torch.maximum(g["length"], torch.full_like(g["length"], 0.001))
    probs = pany[..., None] * strength / torch.clamp(
        strength.sum(dim=-1, keepdim=True), min=1e-30
    )
    intercept = torch.tensor([1.75, 0.75], dtype=torch.float64, device="cuda")
    identity = torch.sigmoid(intercept[None, :] + 0.35 * identity_z)[:, None, :]
    identity = torch.broadcast_to(identity, g["u"].shape)
    means = torch.stack((g["u"], g["depth"], g["length"], g["axis"], identity), dim=-1)
    pose_t = torch.as_tensor(pose, dtype=torch.float64, device="cuda")
    contact = (torch.amax(pose_t[:, :2].abs(), dim=-1) <= 0.04)
    contact = torch.broadcast_to(contact[None, :], pany.shape)
    out = {
        "means": means,
        "p_source": probs,
        "p_visible": pany,
        "contact": contact,
        "geometric_length": g["geometric_length"],
        "angle_audit": g["angle_audit"],
    }
    torch.cuda.synchronize()
    return {key: _cpu_array(value, torch) for key, value in out.items()}


__all__ = ["available", "predict_batch"]
