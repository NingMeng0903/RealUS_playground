"""Neural IRD v6: f_θ(p,u) → (reach_logit, margin, q).

Physical-wavelength Fourier PE on position (independent of AABB span);
Fourier PE on tool axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


# Physical wavelengths (meters): coarse workspace → single-voxel boundary
DEFAULT_P_WAVELENGTHS_M = (0.48, 0.24, 0.12, 0.06, 0.03, 0.015)


def positional_encoding(x: "torch.Tensor", num_freqs: int) -> "torch.Tensor":
    """Normalized-space Fourier (used for direction u)."""
    freqs = (2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype)) * np.pi
    xb = x.unsqueeze(-1) * freqs
    return torch.cat([x, torch.sin(xb).flatten(-2), torch.cos(xb).flatten(-2)], dim=-1)


def physical_position_encoding(
    p_m: "torch.Tensor",
    wavelengths_m: "torch.Tensor",
    *,
    p_scale_m: float = 1.0,
) -> "torch.Tensor":
    """Fourier features with fixed physical wavelengths (meters).

    Returns [p/p_scale, sin(2π p/λ), cos(2π p/λ)] for each λ.
    """
    p_raw = p_m / max(float(p_scale_m), 1e-6)
    phase = 2.0 * np.pi * p_m.unsqueeze(-1) / wavelengths_m
    return torch.cat(
        [p_raw, torch.sin(phase).flatten(-2), torch.cos(phase).flatten(-2)],
        dim=-1,
    )


# backward-compat alias
positional_encoding_xyz = positional_encoding


class ResidualSiLUBlock(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        h = F.silu(self.fc1(x))
        h = self.fc2(h)
        return F.silu(x + h)


class NeuralIRDPoint(nn.Module if nn is not None else object):  # type: ignore[misc]
    """6-D natural [p(3), u(3)] → reach_logit, margin, q.

    Position: physical-wavelength Fourier (default 48…1.5 cm).
    Direction: raw u + num_freqs_u Fourier bands.
    """

    def __init__(
        self,
        *,
        in_dim: int = 6,
        num_freqs: int = 6,
        num_freqs_u: int = 5,
        hidden: int = 256,
        depth: int = 5,
        tau_m: float = 1.0,
        lambda_q: float = 0.5,
        p_wavelengths_m: tuple[float, ...] | list[float] | None = None,
        p_scale_m: float = 1.0,
        use_physical_pe: bool = True,
    ) -> None:
        if torch is None:
            raise ImportError("torch is required for NeuralIRDPoint")
        super().__init__()
        if in_dim != 6:
            raise ValueError("expected 6-D features (p + tool axis)")
        self.in_dim = 6
        self.num_freqs = int(num_freqs)
        self.num_freqs_u = int(num_freqs_u)
        self.hidden = int(hidden)
        self.depth = int(depth)
        self.tau_m = float(tau_m)
        self.lambda_q = float(lambda_q)
        self.p_scale_m = float(p_scale_m)
        self.use_physical_pe = bool(use_physical_pe)
        waves = tuple(p_wavelengths_m) if p_wavelengths_m is not None else DEFAULT_P_WAVELENGTHS_M
        self.register_buffer(
            "p_wavelengths_m",
            torch.tensor(waves, dtype=torch.float32),
        )
        n_wave = int(self.p_wavelengths_m.numel())
        if self.use_physical_pe:
            pe_p = 3 + 3 * 2 * n_wave  # p_raw + sin/cos per λ per axis
        else:
            pe_p = 3 + 3 * 2 * self.num_freqs
        pe_u = 3 + 3 * 2 * self.num_freqs_u
        self.stem = nn.Linear(pe_p + pe_u, hidden)
        self.blocks = nn.ModuleList([ResidualSiLUBlock(hidden) for _ in range(max(1, depth - 1))])
        self.head_cls = nn.Linear(hidden, 1)
        self.head_margin = nn.Linear(hidden, 1)
        self.head_q = nn.Linear(hidden, 1)
        self.register_buffer("aabb_lo", torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32))
        self.register_buffer("aabb_hi", torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32))

    def set_aabb(self, lo: np.ndarray | "torch.Tensor", hi: np.ndarray | "torch.Tensor") -> None:
        self.aabb_lo.copy_(torch.as_tensor(lo, dtype=torch.float32).reshape(3))
        self.aabb_hi.copy_(torch.as_tensor(hi, dtype=torch.float32).reshape(3))

    def encode(self, features: "torch.Tensor") -> "torch.Tensor":
        p = features[..., :3]
        u = features[..., 3:6]
        u = u / (u.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        if self.use_physical_pe:
            p_enc = physical_position_encoding(
                p, self.p_wavelengths_m, p_scale_m=self.p_scale_m
            )
        else:
            span = (self.aabb_hi - self.aabb_lo).clamp_min(1e-6)
            p_n = 2.0 * (p - self.aabb_lo) / span - 1.0
            p_enc = positional_encoding(p_n, self.num_freqs)
        u_enc = positional_encoding(u, self.num_freqs_u)
        return torch.cat([p_enc, u_enc], dim=-1)

    def forward(
        self, features: "torch.Tensor"
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        h = F.silu(self.stem(self.encode(features)))
        for block in self.blocks:
            h = block(h)
        reach_logit = self.head_cls(h)
        margin = self.head_margin(h)
        q = torch.sigmoid(self.head_q(h))
        # Legacy score kept for checkpoint / wandb compat — prefer optimization_cost.
        from ird_playground.neural.cost import legacy_margin_score

        score = legacy_margin_score(margin, q, tau_m=self.tau_m, lambda_q=self.lambda_q)
        return reach_logit, margin, q, score

    def score_features(self, features: "torch.Tensor") -> dict[str, "torch.Tensor"]:
        from ird_playground.neural.cost import optimization_cost

        reach_logit, margin, q, score = self.forward(features)
        p_reach = torch.sigmoid(reach_logit)
        cost = optimization_cost(reach_logit, margin, q)
        return {
            "reach_logit": reach_logit,
            "m": margin,
            "margin": margin,
            "q": q,
            "q_comfort": q,
            "score": score,  # deprecated: margin-only
            "cost": cost,  # preferred for P1 trajectory optimization
            "p_reach": p_reach,
            "d": score,
        }


@dataclass
class PointScore:
    m: float
    q: float
    score: float
    p_reach: float = 0.0
    q_comfort: float = 0.0
    d: float = 0.0
    reach_logit: float = 0.0


class NeuralIRD:
    def __init__(self, model: NeuralIRDPoint, device: str | None = None) -> None:
        if torch is None:
            raise ImportError("torch is required")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.model.eval()

    @classmethod
    def load(cls, checkpoint: str | Path, device: str | None = None) -> "NeuralIRD":
        ckpt = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        cfg = dict(ckpt.get("model_cfg", {}))
        model = NeuralIRDPoint(
            in_dim=int(cfg.get("in_dim", 6)),
            num_freqs=int(cfg.get("num_freqs", 6)),
            num_freqs_u=int(cfg.get("num_freqs_u", 5)),
            hidden=int(cfg.get("hidden", 256)),
            depth=int(cfg.get("depth", 5)),
            tau_m=float(cfg.get("tau_m", 1.0)),
            lambda_q=float(cfg.get("lambda_q", 0.5)),
            p_wavelengths_m=cfg.get("p_wavelengths_m"),
            p_scale_m=float(cfg.get("p_scale_m", 1.0)),
            use_physical_pe=bool(cfg.get("use_physical_pe", True)),
        )
        model.load_state_dict(ckpt["state_dict"], strict=False)
        aabb = cfg.get("aabb")
        if aabb is not None:
            model.set_aabb(np.asarray(aabb["lo"]), np.asarray(aabb["hi"]))
        meta = ckpt.get("meta") or {}
        if "aabb_lo" in meta and "aabb_hi" in meta:
            model.set_aabb(meta["aabb_lo"], meta["aabb_hi"])
        return cls(model, device=device)

    def save(self, path: str | Path, *, model_cfg: dict | None = None, meta: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        waves = self.model.p_wavelengths_m.detach().cpu().numpy().tolist()
        cfg = model_cfg or {
            "in_dim": 6,
            "num_freqs": self.model.num_freqs,
            "num_freqs_u": self.model.num_freqs_u,
            "hidden": self.model.hidden,
            "depth": self.model.depth,
            "tau_m": self.model.tau_m,
            "lambda_q": self.model.lambda_q,
            "use_physical_pe": self.model.use_physical_pe,
            "p_wavelengths_m": waves,
            "p_scale_m": self.model.p_scale_m,
            "aabb": {
                "lo": self.model.aabb_lo.detach().cpu().numpy().tolist(),
                "hi": self.model.aabb_hi.detach().cpu().numpy().tolist(),
            },
        }
        torch.save({"state_dict": self.model.state_dict(), "model_cfg": cfg, "meta": meta or {}}, path)

    @torch.no_grad()
    def score_features_np(self, features: np.ndarray) -> dict[str, np.ndarray]:
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        if x.ndim == 1:
            x = x[None, :]
        out = self.model.score_features(x)
        return {k: v.detach().cpu().numpy().reshape(-1) for k, v in out.items()}

    def score(self, delta_T: np.ndarray) -> PointScore:
        from ird_playground.probe.se3 import features_from_delta_T

        feat = features_from_delta_T(delta_T)
        out = self.score_features_np(feat)
        return PointScore(
            m=float(out["m"][0]),
            q=float(out["q"][0]),
            score=float(out["score"][0]),
            p_reach=float(out["p_reach"][0]),
            q_comfort=float(out["q"][0]),
            d=float(out["score"][0]),
            reach_logit=float(out["reach_logit"][0]),
        )

    def score_batch_delta_T(self, delta_Ts: np.ndarray) -> dict[str, np.ndarray]:
        from ird_playground.probe.se3 import batch_features_from_delta_T

        return self.score_features_np(batch_features_from_delta_T(delta_Ts))

    def region_score(self, **kwargs):
        from ird_playground.region.aggregate import region_score_a

        return region_score_a(self, **kwargs)
