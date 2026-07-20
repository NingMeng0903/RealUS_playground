"""Sprint 0A Region A: local ellipsoid + tool-axis cone around G(λ).

Outer NLP: only (λ, r).
Inner fixed joint Sobol (K=32): [ε_t, ε_b, ε_n, z, v] →
  p = p0 + R_local @ (a ⊙ ε)
  u = Exp([δω]_×) u0   with area-uniform cone (ρ, φ)

No G(y+δy), no perception stack, no rail/plan neighborhood product.
Optional Sprint 0B: shared global SE(3) bias + tiny shared δr (off by default).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore

from ird_playground.neural.cost import optimization_cost


@dataclass
class LocalExtent:
    """Local ellipsoid half-axes (m) + cone half-angle (deg)."""

    a_t_m: float = 0.003
    a_b_m: float = 0.004
    a_n_m: float = 0.002
    beta_max_deg: float = 3.0
    # Sprint 0B (default off)
    global_trans_m: float = 0.0
    global_rot_deg: float = 0.0
    rail_bias_m: float = 0.0


@dataclass
class RegionAggConfig:
    logit_safe: float = 1.0
    margin_safe: float = 0.20
    tau_logit: float = 0.5
    tau_margin: float = 0.1
    w_cls: float = 1.0
    w_margin: float = 0.5
    w_q: float = 0.2
    w_mean: float = 0.3
    w_worst: float = 0.7
    w_cov: float = 1.0
    p_min: float = 0.90
    tau_p: float = 0.1
    tau_worst: float = 0.5


def make_joint_sobol_ellipsoid_cone(
    num_samples: int = 32,
    *,
    extent: LocalExtent | None = None,
    seed: int = 0,
    antithetic: bool = True,
    device=None,
    dtype=None,
) -> "torch.Tensor":
    """Fixed joint Sobol → physical (δp_t, δp_b, δp_n, ρ, φ).

    Unit cube u∈[0,1]^5 mapped by:
      ε_* = 2u-1 ∈ [-1,1] for ellipsoid axes
      cosρ = 1 - z (1 - cos β_max)   (area-uniform spherical cap)
      φ = 2π v
    Returns (K,5) tensor. Always include an explicit center row at index 0.
    """
    if torch is None:
        raise ImportError("torch required")
    from scipy.stats import qmc

    ext = extent or LocalExtent()
    beta = float(np.deg2rad(ext.beta_max_deg))
    cos_b = float(np.cos(beta))

    n_draw = max(num_samples - 1, 1)
    if antithetic:
        n_pair = (n_draw + 1) // 2
        eng = qmc.Sobol(d=5, scramble=True, seed=seed)
        m = int(np.ceil(np.log2(max(n_pair, 2))))
        u = eng.random_base2(m)[:n_pair]
        # antithetic in unit cube about 0.5
        paired = np.empty((u.shape[0] * 2, 5), dtype=np.float64)
        paired[0::2] = u
        paired[1::2] = 1.0 - u
        u_all = paired[:n_draw]
    else:
        eng = qmc.Sobol(d=5, scramble=True, seed=seed)
        m = int(np.ceil(np.log2(max(n_draw, 2))))
        u_all = eng.random_base2(m)[:n_draw]

    eps = 2.0 * u_all[:, :3] - 1.0
    dpt = eps[:, 0] * ext.a_t_m
    dpb = eps[:, 1] * ext.a_b_m
    dpn = eps[:, 2] * ext.a_n_m
    z = u_all[:, 3]
    v = u_all[:, 4]
    cos_rho = 1.0 - z * (1.0 - cos_b)
    rho = np.arccos(np.clip(cos_rho, -1.0, 1.0))
    phi = 2.0 * np.pi * v
    samples = np.stack([dpt, dpb, dpn, rho, phi], axis=1)
    center = np.zeros((1, 5), dtype=np.float64)
    arr = np.concatenate([center, samples], axis=0)[:num_samples]
    return torch.as_tensor(arr, device=device, dtype=dtype or torch.float32)


def _rodrigues(u: "torch.Tensor", dw: "torch.Tensor") -> "torch.Tensor":
    """Rotate unit vectors u by small/finite rotvecs dw (same broadcast shape)."""
    theta = dw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = dw / theta
    # Rodrigues: u' = u cosθ + (k×u) sinθ + k (k·u) (1-cosθ)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    kxu = torch.linalg.cross(k, u)
    kdu = (k * u).sum(dim=-1, keepdim=True)
    return u * cos_t + kxu * sin_t + k * kdu * (1.0 - cos_t)


def _frames_from_u(u: "torch.Tensor", t_ref: "torch.Tensor") -> "torch.Tensor":
    u = F.normalize(u, dim=-1, eps=1e-8)
    t = t_ref - u * (t_ref * u).sum(dim=-1, keepdim=True)
    t_n = t.norm(dim=-1, keepdim=True)
    fb = torch.zeros_like(t)
    fb[..., 0] = 1.0
    t = torch.where(t_n > 1e-5, t / t_n.clamp_min(1e-6), fb)
    t = t - u * (t * u).sum(dim=-1, keepdim=True)
    t = F.normalize(t, dim=-1, eps=1e-8)
    b = F.normalize(torch.linalg.cross(t, u), dim=-1, eps=1e-8)
    t = torch.linalg.cross(u, b)
    return torch.stack([b, t, u], dim=-1)


def _se3_exp_batch(xi: "torch.Tensor") -> "torch.Tensor":
    """ξ (...,6)=[δp,δω] → (...,4,4)."""
    dp, dw = xi[..., :3], xi[..., 3:]
    theta = dw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = dw / theta
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    z = torch.zeros_like(kx)
    K = torch.stack(
        [
            torch.stack([z, -kz, ky], dim=-1),
            torch.stack([kz, z, -kx], dim=-1),
            torch.stack([-ky, kx, z], dim=-1),
        ],
        dim=-2,
    )
    eye = torch.eye(3, dtype=xi.dtype, device=xi.device).expand_as(K)
    th = theta.unsqueeze(-1)
    R = eye + torch.sin(th) * K + (1.0 - torch.cos(th)) * (K @ K)
    small = (dw.norm(dim=-1) < 1e-6)[..., None, None]
    R = torch.where(small, eye + K * th, R)
    T = torch.eye(4, dtype=xi.dtype, device=xi.device).expand(*xi.shape[:-1], 4, 4).clone()
    T[..., :3, :3] = R
    T[..., :3, 3] = dp
    return T


def local_region_cost(
    point_ird,
    lambda_center: "torch.Tensor",
    rail_center: "torch.Tensor",
    surface_model,
    *,
    local_eps: "torch.Tensor" | None = None,
    extent: LocalExtent | None = None,
    agg: RegionAggConfig | None = None,
    num_samples: int = 32,
    sobol_seed: int = 0,
    T_world_rail: np.ndarray | "torch.Tensor" | None = None,
    T_rail_base0: np.ndarray | "torch.Tensor" | None = None,
) -> dict[str, "torch.Tensor"]:
    """Ellipsoid+cone Region A; returns cost / coverage / per-knot tensors."""
    if torch is None:
        raise ImportError("torch required")
    from ird_playground.ird.query_base import (
        features_from_delta_T_torch,
        invert_T_torch,
        T_base_from_rail_y_torch,
    )

    ext = extent or LocalExtent()
    agg = agg or RegionAggConfig()
    device = point_ird.device
    dtype = torch.float32

    lam = lambda_center.to(device=device, dtype=dtype).reshape(-1)
    rail = rail_center.to(device=device, dtype=dtype).reshape(-1)
    if lam.shape != rail.shape:
        raise ValueError("lambda_center and rail_center must match")
    n = int(lam.shape[0])

    if local_eps is None:
        local_eps = make_joint_sobol_ellipsoid_cone(
            num_samples, extent=ext, seed=sobol_seed, device=device, dtype=dtype
        )
    else:
        local_eps = local_eps.to(device=device, dtype=dtype)
    k = int(local_eps.shape[0])

    if not hasattr(surface_model, "sample_torch"):
        raise TypeError("surface_model must implement sample_torch")
    T0 = surface_model.sample_torch(lam, dtype=dtype, device=device)
    if T0.ndim == 2:
        T0 = T0.unsqueeze(0)
    p0 = T0[..., :3, 3]
    # R_local columns in SyntheticVesselSkinManifold: [b, t, n]
    b = T0[..., :3, 0]
    t = T0[..., :3, 1]
    nn = T0[..., :3, 2]
    u0 = nn

    dpt, dpb, dpn = local_eps[:, 0], local_eps[:, 1], local_eps[:, 2]
    rho, phi = local_eps[:, 3], local_eps[:, 4]

    p = (
        p0[:, None, :]
        + dpt[None, :, None] * t[:, None, :]
        + dpb[None, :, None] * b[:, None, :]
        + dpn[None, :, None] * nn[:, None, :]
    )
    # cone: δω = ρ (cosφ t + sinφ b)
    dw = (
        rho[None, :, None]
        * (
            torch.cos(phi)[None, :, None] * t[:, None, :]
            + torch.sin(phi)[None, :, None] * b[:, None, :]
        )
    )
    u = F.normalize(_rodrigues(u0[:, None, :].expand(-1, k, -1), dw), dim=-1, eps=1e-8)

    R = _frames_from_u(u, t_ref=t[:, None, :].expand(-1, k, -1))
    T_local = torch.eye(4, dtype=dtype, device=device).expand(n, k, 4, 4).clone()
    T_local[..., :3, :3] = R
    T_local[..., :3, 3] = p

    # Optional Sprint 0B: one global SE(3) bias per scenario, shared across knots
    if ext.global_trans_m > 0.0 or ext.global_rot_deg > 0.0:
        # derive global twists from first 3 pos + angles of each sample (shared over N)
        g_scale_t = float(ext.global_trans_m)
        g_scale_r = float(np.deg2rad(ext.global_rot_deg))
        # use sample index hash from (dpt,phi) so scenarios differ but are fixed
        g_dp = torch.stack(
            [
                torch.tanh(dpt / (ext.a_t_m + 1e-9)) * g_scale_t,
                torch.tanh(dpb / (ext.a_b_m + 1e-9)) * g_scale_t,
                torch.tanh(dpn / (ext.a_n_m + 1e-9)) * g_scale_t,
            ],
            dim=-1,
        )  # (K,3)
        g_dw = torch.stack(
            [
                torch.cos(phi) * (rho / (np.deg2rad(ext.beta_max_deg) + 1e-9)) * g_scale_r,
                torch.sin(phi) * (rho / (np.deg2rad(ext.beta_max_deg) + 1e-9)) * g_scale_r,
                torch.zeros_like(phi),
            ],
            dim=-1,
        )
        xi_g = torch.cat([g_dp, g_dw], dim=-1)  # (K,6)
        T_glob = _se3_exp_batch(xi_g)  # (K,4,4)
        T_tcp = T_glob[None, :, :, :] @ T_local
    else:
        T_tcp = T_local

    if T_world_rail is None:
        Twr = torch.eye(4, dtype=dtype, device=device)
    elif torch.is_tensor(T_world_rail):
        Twr = T_world_rail.to(device=device, dtype=dtype)
    else:
        Twr = torch.as_tensor(np.asarray(T_world_rail), dtype=dtype, device=device)
    if T_rail_base0 is None:
        Trb = torch.eye(4, dtype=dtype, device=device)
    elif torch.is_tensor(T_rail_base0):
        Trb = T_rail_base0.to(device=device, dtype=dtype)
    else:
        Trb = torch.as_tensor(np.asarray(T_rail_base0), dtype=dtype, device=device)

    # Shared rail per knot; optional tiny shared δr per scenario (0B)
    if ext.rail_bias_m > 0.0:
        dr = torch.tanh(dpt / (ext.a_t_m + 1e-9)) * float(ext.rail_bias_m)  # (K,)
        rail_s = rail[:, None] + dr[None, :]  # (N,K)
        rail_flat = rail_s.reshape(-1)
        T_base = T_base_from_rail_y_torch(rail_flat, T_world_rail=Twr, T_rail_base0=Trb)
        T_base = T_base.reshape(n, k, 4, 4)
    else:
        T_base = T_base_from_rail_y_torch(rail, T_world_rail=Twr, T_rail_base0=Trb)
        T_base = T_base[:, None, :, :].expand(-1, k, -1, -1)

    dT = invert_T_torch(T_tcp) @ T_base
    feat = features_from_delta_T_torch(dT).reshape(n * k, 6)

    point_ird.model.eval()
    out = point_ird.model.score_features(feat)
    logit = out["reach_logit"].reshape(n, k)
    margin = out["m"].reshape(n, k)
    quality = out["q"].reshape(n, k)

    c_k = optimization_cost(
        logit,
        margin,
        quality,
        logit_safe=agg.logit_safe,
        margin_safe=agg.margin_safe,
        tau_logit=agg.tau_logit,
        tau_margin=agg.tau_margin,
        w_cls=agg.w_cls,
        w_margin=agg.w_margin,
        w_q=agg.w_q,
    )

    p_cov = torch.sigmoid(logit).mean(dim=1)
    cov_pen = F.softplus((agg.p_min - p_cov) / max(agg.tau_p, 1e-6))

    c_mean = c_k.mean(dim=1)
    tau = max(float(agg.tau_worst), 1e-6)
    c_shift = c_k - c_k.max(dim=1, keepdim=True).values
    c_worst = tau * torch.log(torch.exp(c_shift / tau).mean(dim=1) + 1e-12) + c_k.max(dim=1).values

    cost_per_knot = (
        agg.w_mean * c_mean + agg.w_worst * c_worst + agg.w_cov * cov_pen
    )
    return {
        "cost": cost_per_knot.mean(),
        "cost_per_knot": cost_per_knot,
        "c_mean": c_mean,
        "c_worst": c_worst,
        "p_cov": p_cov,
        "point_cost": c_k,
        "reach_logit": logit,
        "margin": margin,
        "quality": quality,
        "local_eps": local_eps,
        "num_samples": torch.tensor(float(k), device=device),
    }


# Back-compat name
robust_region_cost = local_region_cost
