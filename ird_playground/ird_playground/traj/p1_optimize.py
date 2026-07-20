"""P1 offline optimizer: Bernstein (c_λ, c_r) + local ellipsoid/cone Region A."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


def bernstein_basis(s: "torch.Tensor", n_ctrl: int) -> "torch.Tensor":
    if torch is None:
        raise ImportError("torch required")
    s = s.reshape(-1).clamp(0.0, 1.0)
    deg = int(n_ctrl) - 1
    return torch.stack(
        [float(comb(deg, i)) * (s**i) * ((1.0 - s) ** (deg - i)) for i in range(deg + 1)],
        dim=1,
    )


@dataclass
class P1Config:
    n_ctrl: int = 8
    n_knots_eval: int = 48
    region_k: int = 32
    sobol_seed: int = 0
    w_ird: float = 1.0
    w_track: float = 0.5
    w_smooth_lam: float = 0.1
    w_smooth_rail: float = 0.1
    w_d2_lam: float = 0.05
    w_d2_rail: float = 0.05
    steps: int = 60
    lr: float = 5e-3


def optimize_p1_lambda_rail(
    neural_ird,
    manifold,
    *,
    lambda_ref: np.ndarray | None = None,
    rail_ref: np.ndarray | None = None,
    cfg: P1Config | None = None,
    T_world_rail: np.ndarray | None = None,
    T_rail_base0: np.ndarray | None = None,
    extent=None,
    agg=None,
) -> dict:
    if torch is None:
        raise ImportError("torch required")
    from ird_playground.region.local_region import (
        local_region_cost,
        make_joint_sobol_ellipsoid_cone,
    )

    cfg = cfg or P1Config()
    device = neural_ird.device
    n_ctrl = int(cfg.n_ctrl)
    m = int(cfg.n_knots_eval)
    length = float(getattr(manifold, "length_m", 0.40))

    if lambda_ref is None:
        lambda_ref = np.linspace(0.05 * length, 0.95 * length, m)
    else:
        lambda_ref = np.asarray(lambda_ref, dtype=np.float64).reshape(-1)
        m = int(lambda_ref.size)
    if rail_ref is None:
        rail_ref = np.zeros(m, dtype=np.float64)
    else:
        rail_ref = np.asarray(rail_ref, dtype=np.float64).reshape(-1)

    s = torch.linspace(0.0, 1.0, m, device=device, dtype=torch.float32)
    B = bernstein_basis(s, n_ctrl)
    Br = B.detach().cpu().numpy()
    c_lam = torch.tensor(
        np.linalg.lstsq(Br, lambda_ref, rcond=None)[0],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    c_rail = torch.tensor(
        np.linalg.lstsq(Br, rail_ref, rcond=None)[0],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    lam_ref_t = torch.as_tensor(lambda_ref, dtype=torch.float32, device=device)

    # Fixed joint Sobol — never resample inside the loop
    local_eps = make_joint_sobol_ellipsoid_cone(
        cfg.region_k, extent=extent, seed=cfg.sobol_seed, device=device
    )

    opt = torch.optim.Adam([c_lam, c_rail], lr=cfg.lr)
    history: list[float] = []
    neural_ird.model.eval()
    for _ in range(int(cfg.steps)):
        opt.zero_grad()
        lam = B @ c_lam
        rail = B @ c_rail
        reg = local_region_cost(
            neural_ird,
            lam,
            rail,
            manifold,
            local_eps=local_eps,
            extent=extent,
            agg=agg,
            T_world_rail=T_world_rail,
            T_rail_base0=T_rail_base0,
        )
        d1_l, d1_r = lam[1:] - lam[:-1], rail[1:] - rail[:-1]
        d2_l, d2_r = d1_l[1:] - d1_l[:-1], d1_r[1:] - d1_r[:-1]
        loss = (
            cfg.w_ird * reg["cost"]
            + cfg.w_track * ((lam - lam_ref_t) ** 2).mean()
            + cfg.w_smooth_lam * (d1_l**2).mean()
            + cfg.w_smooth_rail * (d1_r**2).mean()
            + cfg.w_d2_lam * (d2_l**2).mean()
            + cfg.w_d2_rail * (d2_r**2).mean()
        )
        loss.backward()
        opt.step()
        history.append(float(loss.detach().cpu()))

    with torch.no_grad():
        lam_f = (B @ c_lam).detach().cpu().numpy()
        rail_f = (B @ c_rail).detach().cpu().numpy()
        cov = float(
            local_region_cost(
                neural_ird,
                torch.as_tensor(lam_f, device=device),
                torch.as_tensor(rail_f, device=device),
                manifold,
                local_eps=local_eps,
                extent=extent,
                agg=agg,
                T_world_rail=T_world_rail,
                T_rail_base0=T_rail_base0,
            )["p_cov"]
            .mean()
            .cpu()
        )
    return {
        "lambda": lam_f,
        "rail": rail_f,
        "c_lambda": c_lam.detach().cpu().numpy(),
        "c_rail": c_rail.detach().cpu().numpy(),
        "history": history,
        "final_loss": history[-1] if history else float("nan"),
        "final_coverage": cov,
    }
