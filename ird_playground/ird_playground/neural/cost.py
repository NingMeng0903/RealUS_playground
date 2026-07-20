"""Trajectory-optimization cost from Neural IRD heads.

Do **not** use margin alone as a global reachability potential.
``reach_logit`` provides the global field; ``margin`` is a local safety
cushion near the supervised boundary; ``q`` is an interior comfort bonus.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore


def optimization_cost(
    reach_logit: "torch.Tensor",
    margin: "torch.Tensor",
    quality: "torch.Tensor",
    *,
    logit_safe: float = 1.0,
    margin_safe: float = 0.20,
    tau_logit: float = 0.5,
    tau_margin: float = 0.1,
    w_cls: float = 1.0,
    w_margin: float = 0.5,
    w_q: float = 0.2,
) -> "torch.Tensor":
    """Scalar / batched IRD cost (lower is better).

    C = w_cls * softplus((ℓ_safe − ℓ)/τ_ℓ)
      + w_margin * softplus((m_safe − m)/τ_m)
      − w_q * q
    """
    if torch is None:
        raise ImportError("torch required")
    c_cls = F.softplus((float(logit_safe) - reach_logit) / max(float(tau_logit), 1e-6))
    c_margin = F.softplus((float(margin_safe) - margin) / max(float(tau_margin), 1e-6))
    return float(w_cls) * c_cls + float(w_margin) * c_margin - float(w_q) * quality


def legacy_margin_score(
    margin: "torch.Tensor",
    quality: "torch.Tensor",
    *,
    tau_m: float = 1.0,
    lambda_q: float = 0.5,
) -> "torch.Tensor":
    """Deprecated v6 score: −softplus(−m/τ) + λ_q q. Kept for checkpoint compat."""
    if torch is None:
        raise ImportError("torch required")
    return -F.softplus(-margin / max(float(tau_m), 1e-6)) + float(lambda_q) * quality
