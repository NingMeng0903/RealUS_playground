"""Differentiable relative-pose and rail queries for Neural IRD."""

from __future__ import annotations

import numpy as np

from ird_playground.neural.cost import legacy_margin_score, optimization_cost
from ird_playground.probe.se3 import features_from_delta_T, invert_T, se3_mul

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def trans_y(r: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[1, 3] = float(r)
    return T


def T_base_from_rail_y(
    rail_y: float,
    *,
    T_world_rail: np.ndarray | None = None,
    T_rail_base0: np.ndarray | None = None,
) -> np.ndarray:
    Twr = np.eye(4, dtype=np.float64) if T_world_rail is None else np.asarray(T_world_rail, dtype=np.float64)
    Trb = np.eye(4, dtype=np.float64) if T_rail_base0 is None else np.asarray(T_rail_base0, dtype=np.float64)
    return se3_mul(se3_mul(Twr, trans_y(rail_y)), Trb)


def delta_T_from_tcp_and_rail(
    T_tcp: np.ndarray,
    rail_y: float,
    *,
    T_world_rail: np.ndarray | None = None,
    T_rail_base0: np.ndarray | None = None,
) -> np.ndarray:
    T_base = T_base_from_rail_y(
        rail_y, T_world_rail=T_world_rail, T_rail_base0=T_rail_base0
    )
    return invert_T(np.asarray(T_tcp, dtype=np.float64)) @ T_base


def score_vs_rail_y(
    neural_ird,
    T_tcp: np.ndarray,
    rail_y: float,
    *,
    T_world_rail: np.ndarray | None = None,
    T_rail_base0: np.ndarray | None = None,
) -> dict[str, float]:
    dT = delta_T_from_tcp_and_rail(
        T_tcp, rail_y, T_world_rail=T_world_rail, T_rail_base0=T_rail_base0
    )
    ps = neural_ird.score(dT)
    return {
        "m": ps.m,
        "q": ps.q,
        "score": ps.score,
        "reach_logit": ps.reach_logit,
        "p_reach": ps.p_reach,
    }


def invert_T_torch(T: "torch.Tensor") -> "torch.Tensor":
    """Batch-capable inverse: (...,4,4) → (...,4,4)."""
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = R.transpose(-1, -2)
    Ti = torch.zeros_like(T)
    # identity on last  row/col then fill
    eye = torch.eye(4, dtype=T.dtype, device=T.device)
    if T.ndim == 2:
        Ti = eye.clone()
        Ti[:3, :3] = Rt
        Ti[:3, 3] = -(Rt @ t)
        return Ti
    Ti = eye.expand(T.shape).clone()
    Ti[..., :3, :3] = Rt
    Ti[..., :3, 3] = -(Rt @ t.unsqueeze(-1)).squeeze(-1)
    return Ti


def features_from_delta_T_torch(dT: "torch.Tensor") -> "torch.Tensor":
    """(…,4,4) → (…,6) natural [p,u]."""
    R_delta = dT[..., :3, :3]
    t_delta = dT[..., :3, 3]
    R_base_tcp = R_delta.transpose(-1, -2)
    p = -(R_base_tcp @ t_delta.unsqueeze(-1)).squeeze(-1)
    u = R_base_tcp[..., :, 2]
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.cat([p, u], dim=-1)


def rot6d_features_from_delta_T_torch(dT: "torch.Tensor") -> "torch.Tensor":
    """(...,4,4) -> (...,9) full-pose ``[p,R[:,0],R[:,1]]`` features."""
    R_delta = dT[..., :3, :3]
    t_delta = dT[..., :3, 3]
    R_base_tcp = R_delta.transpose(-1, -2)
    p = -(R_base_tcp @ t_delta.unsqueeze(-1)).squeeze(-1)
    return torch.cat([p, R_base_tcp[..., :, 0], R_base_tcp[..., :, 1]], dim=-1)


def T_base_from_rail_y_torch(
    rail_y: "torch.Tensor",
    *,
    T_world_rail: "torch.Tensor",
    T_rail_base0: "torch.Tensor",
) -> "torch.Tensor":
    """rail_y: () or (N,) → T_base (4,4) or (N,4,4)."""
    device, dtype = rail_y.device, rail_y.dtype
    eye = torch.eye(4, dtype=dtype, device=device)
    if rail_y.ndim == 0:
        Ty = eye.clone()
        Ty[1, 3] = rail_y
        return T_world_rail @ Ty @ T_rail_base0
    n = int(rail_y.shape[0])
    Ty = eye.expand(n, 4, 4).clone()
    Ty[:, 1, 3] = rail_y
    return T_world_rail @ Ty @ T_rail_base0


def cost_from_tcp_and_rail_torch(
    neural_ird,
    T_tcp: "torch.Tensor",
    rail_y: "torch.Tensor",
    *,
    T_world_rail: np.ndarray | "torch.Tensor" | None = None,
    T_rail_base0: np.ndarray | "torch.Tensor" | None = None,
    use_optimization_cost: bool = True,
    cost_kwargs: dict | None = None,
) -> dict[str, "torch.Tensor"]:
    """Full AD w.r.t. T_tcp and rail_y.

    Returns reach_logit, m, q, cost (and legacy score).
    """
    if torch is None:
        raise ImportError("torch required")
    device = neural_ird.device
    dtype = torch.float32
    T_tcp = T_tcp.to(device=device, dtype=dtype)
    rail_y = rail_y.to(device=device, dtype=dtype)

    if T_world_rail is None:
        Twr_t = torch.eye(4, dtype=dtype, device=device)
    elif torch.is_tensor(T_world_rail):
        Twr_t = T_world_rail.to(device=device, dtype=dtype)
    else:
        Twr_t = torch.as_tensor(np.asarray(T_world_rail), dtype=dtype, device=device)

    if T_rail_base0 is None:
        Trb_t = torch.eye(4, dtype=dtype, device=device)
    elif torch.is_tensor(T_rail_base0):
        Trb_t = T_rail_base0.to(device=device, dtype=dtype)
    else:
        Trb_t = torch.as_tensor(np.asarray(T_rail_base0), dtype=dtype, device=device)

    T_base = T_base_from_rail_y_torch(rail_y, T_world_rail=Twr_t, T_rail_base0=Trb_t)
    # broadcast T_tcp if scalar rail / batch mismatch
    if T_tcp.ndim == 2 and T_base.ndim == 3:
        T_tcp = T_tcp.expand(T_base.shape[0], 4, 4)
    elif T_tcp.ndim == 3 and T_base.ndim == 2:
        T_base = T_base.expand(T_tcp.shape[0], 4, 4)

    dT = invert_T_torch(T_tcp) @ T_base
    feat = (
        rot6d_features_from_delta_T_torch(dT)
        if neural_ird.model.feature_spec.use_rot6d
        else features_from_delta_T_torch(dT)
    )
    if feat.ndim == 1:
        feat = feat.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    out = neural_ird.model.score_features(feat)
    logit = out["reach_logit"].squeeze(-1)
    m = out["m"].squeeze(-1)
    q = out["q"].squeeze(-1)
    legacy = legacy_margin_score(m, q, tau_m=neural_ird.model.tau_m, lambda_q=neural_ird.model.lambda_q)
    if use_optimization_cost:
        cost = optimization_cost(logit, m, q, **(cost_kwargs or {}))
    else:
        cost = -legacy

    def _sq(x):
        return x.squeeze(0) if squeeze and x.ndim > 0 and x.shape[0] == 1 else x

    return {
        "reach_logit": _sq(logit),
        "m": _sq(m),
        "q": _sq(q),
        "cost": _sq(cost),
        "score": _sq(legacy),  # deprecated margin-only score
        "p_reach": _sq(torch.sigmoid(logit)),
        "features": feat if not squeeze else feat[0],
        "delta_T": dT if not (squeeze and dT.ndim == 3) else dT[0],
    }


def score_vs_rail_y_torch(
    neural_ird,
    T_tcp,
    rail_y: "torch.Tensor",
    *,
    T_world_rail: np.ndarray | None = None,
    T_rail_base0: np.ndarray | None = None,
) -> "torch.Tensor":
    """Backward-compat: returns legacy score. Prefer cost_from_tcp_and_rail_torch.

    If ``T_tcp`` is a numpy array it is converted once (∂/∂T_tcp disabled);
    pass a Tensor to keep the full graph.
    """
    if torch is None:
        raise ImportError("torch required")
    if not torch.is_tensor(T_tcp):
        T_tcp = torch.as_tensor(np.asarray(T_tcp, dtype=np.float64), dtype=torch.float32, device=neural_ird.device)
    out = cost_from_tcp_and_rail_torch(
        neural_ird,
        T_tcp,
        rail_y,
        T_world_rail=T_world_rail,
        T_rail_base0=T_rail_base0,
        use_optimization_cost=False,
    )
    return out["score"]


def rail_y_grad_ad_fd(
    neural_ird,
    *,
    n: int = 32,
    rail_y: float = 0.0,
    eps: float = 1e-3,
    seed: int = 0,
    T_world_rail: np.ndarray | None = None,
    T_rail_base0: np.ndarray | None = None,
) -> dict[str, float]:
    """Compare AD ∂cost/∂rail_y to central finite differences."""
    if torch is None:
        raise ImportError("torch required")
    from ird_playground.probe.se3 import complete_frame_from_tool_axis, mat4_from_Rt

    rng = np.random.default_rng(seed)
    rels = []
    signs = []
    neural_ird.model.eval()
    for _ in range(n):
        p = rng.uniform(-0.5, 0.5, size=3)
        u = rng.normal(size=3)
        u = u / (np.linalg.norm(u) + 1e-12)
        T_tcp_np = mat4_from_Rt(complete_frame_from_tool_axis(u), p)
        T_tcp = torch.as_tensor(T_tcp_np, dtype=torch.float32, device=neural_ird.device)

        r = torch.tensor(float(rail_y), dtype=torch.float32, device=neural_ird.device, requires_grad=True)
        out = cost_from_tcp_and_rail_torch(
            neural_ird, T_tcp, r, T_world_rail=T_world_rail, T_rail_base0=T_rail_base0
        )
        out["cost"].backward()
        g_ad = float(r.grad.item())

        with torch.no_grad():
            sp = cost_from_tcp_and_rail_torch(
                neural_ird,
                T_tcp,
                torch.tensor(rail_y + eps, dtype=torch.float32, device=neural_ird.device),
                T_world_rail=T_world_rail,
                T_rail_base0=T_rail_base0,
            )["cost"].item()
            sm = cost_from_tcp_and_rail_torch(
                neural_ird,
                T_tcp,
                torch.tensor(rail_y - eps, dtype=torch.float32, device=neural_ird.device),
                T_world_rail=T_world_rail,
                T_rail_base0=T_rail_base0,
            )["cost"].item()
        g_fd = (sp - sm) / (2.0 * eps)
        denom = max(abs(g_fd), abs(g_ad), 1e-6)
        rels.append(abs(g_ad - g_fd) / denom)
        signs.append(1.0 if np.sign(g_ad) == np.sign(g_fd) or abs(g_fd) < 1e-8 else 0.0)

    return {
        "rail_ad_fd_rel": float(np.median(rels)),
        "rail_sign_agree": float(np.mean(signs)),
        "rail_n": float(n),
    }
