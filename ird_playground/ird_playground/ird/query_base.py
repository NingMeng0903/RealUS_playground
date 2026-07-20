"""Query-time base pose from rail_y via full SE(3) composition + AD helpers."""

from __future__ import annotations

import numpy as np

from ird_playground.probe.se3 import features_from_delta_T, invert_T, se3_mul

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def trans_y(r: float) -> np.ndarray:
    """Homogeneous translation along +Y (rail axis)."""
    T = np.eye(4, dtype=np.float64)
    T[1, 3] = float(r)
    return T


def T_base_from_rail_y(
    rail_y: float,
    *,
    T_world_rail: np.ndarray | None = None,
    T_rail_base0: np.ndarray | None = None,
) -> np.ndarray:
    """T_base(r) = T_world_rail · Trans_y(r) · T_rail_base0."""
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
    """ΔT(r) = T_tcp^{-1} T_base(r)."""
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
    """Query network at ΔT(rail_y); returns scalar m,q,score."""
    dT = delta_T_from_tcp_and_rail(
        T_tcp, rail_y, T_world_rail=T_world_rail, T_rail_base0=T_rail_base0
    )
    ps = neural_ird.score(dT)
    return {"m": ps.m, "q": ps.q, "score": ps.score}


def _features_torch_from_delta_T(dT: "torch.Tensor") -> "torch.Tensor":
    """dT (4,4) → features (6,) = t + tool_axis."""
    t = dT[:3, 3]
    R = dT[:3, :3]
    u = R[2, :]
    u = u / (u.norm().clamp_min(1e-6))
    return torch.cat([t, u], dim=0)


def score_vs_rail_y_torch(
    neural_ird,
    T_tcp: np.ndarray,
    rail_y: "torch.Tensor",
    *,
    T_world_rail: np.ndarray | None = None,
    T_rail_base0: np.ndarray | None = None,
) -> "torch.Tensor":
    """Differentiable score w.r.t. rail_y (scalar tensor)."""
    if torch is None:
        raise ImportError("torch required")
    Twr = np.eye(4) if T_world_rail is None else np.asarray(T_world_rail, dtype=np.float64)
    Trb = np.eye(4) if T_rail_base0 is None else np.asarray(T_rail_base0, dtype=np.float64)
    T_tcp = np.asarray(T_tcp, dtype=np.float64)
    device = neural_ird.device

    Twr_t = torch.as_tensor(Twr, dtype=torch.float32, device=device)
    Trb_t = torch.as_tensor(Trb, dtype=torch.float32, device=device)
    Ttcp_t = torch.as_tensor(T_tcp, dtype=torch.float32, device=device)

    # Trans_y(r)
    Ty = torch.eye(4, dtype=torch.float32, device=device)
    Ty = Ty.clone()
    Ty[1, 3] = rail_y
    T_base = Twr_t @ Ty @ Trb_t
    # invert T_tcp
    R = Ttcp_t[:3, :3]
    t = Ttcp_t[:3, 3]
    Ti = torch.eye(4, dtype=torch.float32, device=device)
    Ti = Ti.clone()
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    dT = Ti @ T_base
    feat = _features_torch_from_delta_T(dT).unsqueeze(0)
    _, _, _, score = neural_ird.model(feat)
    return score.squeeze()


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
    """Compare AD ∂score/∂rail_y to central finite differences."""
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
        T_tcp = mat4_from_Rt(complete_frame_from_tool_axis(u), p)

        r = torch.tensor(float(rail_y), dtype=torch.float32, device=neural_ird.device, requires_grad=True)
        s = score_vs_rail_y_torch(
            neural_ird, T_tcp, r, T_world_rail=T_world_rail, T_rail_base0=T_rail_base0
        )
        s.backward()
        g_ad = float(r.grad.item())

        sp = score_vs_rail_y(
            neural_ird, T_tcp, rail_y + eps, T_world_rail=T_world_rail, T_rail_base0=T_rail_base0
        )["score"]
        sm = score_vs_rail_y(
            neural_ird, T_tcp, rail_y - eps, T_world_rail=T_world_rail, T_rail_base0=T_rail_base0
        )["score"]
        g_fd = (sp - sm) / (2.0 * eps)
        denom = max(abs(g_fd), abs(g_ad), 1e-6)
        rels.append(abs(g_ad - g_fd) / denom)
        signs.append(1.0 if np.sign(g_ad) == np.sign(g_fd) or abs(g_fd) < 1e-8 else 0.0)

    return {
        "rail_ad_fd_rel": float(np.median(rels)),
        "rail_sign_agree": float(np.mean(signs)),
        "rail_n": float(n),
    }
