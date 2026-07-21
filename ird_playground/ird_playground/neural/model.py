"""Neural IRD relative-pose field: f_θ(p,R6D) → (clearance, q).

Physical-wavelength Fourier PE on position (independent of AABB span) and
Fourier features on the continuous rotation representation.
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


class MultiResolutionHashEncoding(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Local, continuous 3-D hash-grid encoding with trilinear interpolation."""

    _CORNERS = (
        (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
        (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1),
    )

    def __init__(
        self,
        *,
        levels: int = 12,
        features_per_level: int = 2,
        log2_hash_size: int = 18,
        base_resolution: int = 8,
        max_resolution: int = 256,
    ) -> None:
        super().__init__()
        self.levels = int(levels)
        self.features_per_level = int(features_per_level)
        self.hash_size = 1 << int(log2_hash_size)
        if self.levels <= 0 or self.features_per_level <= 0:
            raise ValueError("hash levels and features_per_level must be positive")
        if self.levels == 1:
            resolutions = [int(base_resolution)]
        else:
            growth = np.exp(
                np.log(float(max_resolution) / float(base_resolution))
                / float(self.levels - 1)
            )
            resolutions = [
                int(np.floor(float(base_resolution) * growth**i))
                for i in range(self.levels)
            ]
        self.resolution_values = tuple(resolutions)
        self.register_buffer(
            "resolutions", torch.tensor(resolutions, dtype=torch.int64), persistent=True
        )
        self.register_buffer(
            "corners",
            torch.tensor(self._CORNERS, dtype=torch.int64),
            persistent=False,
        )
        self.tables = nn.Parameter(
            torch.empty(self.levels, self.hash_size, self.features_per_level)
        )
        nn.init.uniform_(self.tables, -1.0e-4, 1.0e-4)
        self.output_dim = self.levels * self.features_per_level + 3

    def _hash(self, ijk: "torch.Tensor") -> "torch.Tensor":
        x, y, z = ijk.unbind(dim=-1)
        return ((x * 1) ^ (y * 2654435761) ^ (z * 805459861)) % self.hash_size

    def forward(self, x01: "torch.Tensor") -> "torch.Tensor":
        shape = x01.shape[:-1]
        x = x01.reshape(-1, 3).clamp(0.0, 1.0)
        encoded = [2.0 * x - 1.0]
        for level in range(self.levels):
            resolution = self.resolution_values[level]
            grid = x * float(max(resolution - 1, 1))
            base = torch.floor(grid).to(torch.int64)
            frac = grid - base.to(grid.dtype)
            idx = base[:, None, :] + self.corners[None, :, :]
            choose = torch.where(
                self.corners[None, :, :].bool(),
                frac[:, None, :],
                1.0 - frac[:, None, :],
            )
            weight = choose.prod(dim=-1, keepdim=True)
            value = (weight * self.tables[level, self._hash(idx)]).sum(dim=1)
            encoded.append(value)
        return torch.cat(encoded, dim=-1).reshape(*shape, self.output_dim)


class NeuralIRDPoint(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Point IRD over legacy 5-DoF or full 6-DoF relative-pose features.

    Position: physical-wavelength Fourier (default 48…1.5 cm).
    Orientation: legacy tool axis or full rotation 6D plus Fourier bands.
    Optional roll: cos/sin α plus harmonic Fourier bands.
    """

    def __init__(
        self,
        *,
        in_dim: int | None = None,
        feature_kind: str = "pu6",
        num_freqs: int = 6,
        num_freqs_u: int = 5,
        hidden: int = 256,
        depth: int = 5,
        tau_m: float = 1.0,
        lambda_q: float = 0.5,
        p_wavelengths_m: tuple[float, ...] | list[float] | None = None,
        p_scale_m: float = 1.0,
        use_physical_pe: bool = True,
        position_encoder: str = "fourier",
        hash_levels: int = 12,
        hash_features_per_level: int = 2,
        hash_log2_size: int = 18,
        hash_base_resolution: int = 8,
        hash_max_resolution: int = 256,
        couple_reach_to_margin: bool = False,
        clearance_logit_scale: float = 3.0,
    ) -> None:
        if torch is None:
            raise ImportError("torch is required for NeuralIRDPoint")
        from ird_playground.neural.feature_spec import make_feature_spec

        super().__init__()
        # Prefer feature_kind; allow legacy in_dim as a checkpoint hint.
        kind = feature_kind
        if in_dim is not None and feature_kind in ("pu6", "pu"):
            if int(in_dim) == 8:
                kind = "pu_roll8"
            elif int(in_dim) == 9:
                kind = "se3_9d"
            elif int(in_dim) not in (6, 8, 9):
                raise ValueError(f"unsupported in_dim={in_dim}")
        self.feature_spec = make_feature_spec(kind)
        self.in_dim = int(self.feature_spec.dim)
        self.feature_kind = self.feature_spec.kind
        self.num_freqs = int(num_freqs)
        self.num_freqs_u = int(num_freqs_u)
        self.hidden = int(hidden)
        self.depth = int(depth)
        self.tau_m = float(tau_m)
        self.lambda_q = float(lambda_q)
        self.p_scale_m = float(p_scale_m)
        self.use_physical_pe = bool(use_physical_pe)
        self.position_encoder = str(position_encoder).lower().strip()
        self.couple_reach_to_margin = bool(couple_reach_to_margin)
        self.clearance_logit_scale = float(clearance_logit_scale)
        waves = tuple(p_wavelengths_m) if p_wavelengths_m is not None else DEFAULT_P_WAVELENGTHS_M
        self.register_buffer(
            "p_wavelengths_m",
            torch.tensor(waves, dtype=torch.float32),
        )
        n_wave = int(self.p_wavelengths_m.numel())
        self.hash_encoder = None
        if self.position_encoder == "hash_grid":
            self.hash_encoder = MultiResolutionHashEncoding(
                levels=hash_levels,
                features_per_level=hash_features_per_level,
                log2_hash_size=hash_log2_size,
                base_resolution=hash_base_resolution,
                max_resolution=hash_max_resolution,
            )
            pe_p = self.hash_encoder.output_dim
        elif self.position_encoder not in {"fourier", "pe"}:
            raise ValueError(f"unsupported position_encoder={position_encoder!r}")
        elif self.use_physical_pe:
            pe_p = 3 + 3 * 2 * n_wave  # p_raw + sin/cos per λ per axis
        else:
            pe_p = 3 + 3 * 2 * self.num_freqs
        orientation_dim = 6 if self.feature_spec.use_rot6d else 3
        pe_u = orientation_dim + orientation_dim * 2 * self.num_freqs_u
        pe_roll = 0
        if self.feature_spec.use_roll:
            # cosα, sinα + (sin(kα), cos(kα)) per harmonic
            pe_roll = 2 + 2 * len(self.feature_spec.roll_harmonics)
        self.stem = nn.Linear(pe_p + pe_u + pe_roll, hidden)
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
        if features.shape[-1] != self.in_dim:
            raise ValueError(
                f"feature dim mismatch: got {features.shape[-1]}, expected {self.in_dim} "
                f"(feature_kind={self.feature_kind})"
            )
        p = features[..., :3]
        orientation = features[..., 3:9] if self.feature_spec.use_rot6d else features[..., 3:6]
        if not self.feature_spec.use_rot6d:
            orientation = orientation / orientation.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        if self.hash_encoder is not None:
            span = (self.aabb_hi - self.aabb_lo).clamp_min(1e-6)
            p_01 = (p - self.aabb_lo) / span
            p_enc = self.hash_encoder(p_01)
        elif self.use_physical_pe:
            p_enc = physical_position_encoding(
                p, self.p_wavelengths_m, p_scale_m=self.p_scale_m
            )
        else:
            span = (self.aabb_hi - self.aabb_lo).clamp_min(1e-6)
            p_n = 2.0 * (p - self.aabb_lo) / span - 1.0
            p_enc = positional_encoding(p_n, self.num_freqs)
        u_enc = positional_encoding(orientation, self.num_freqs_u)
        parts = [p_enc, u_enc]
        if self.feature_spec.use_roll:
            cos_a = features[..., 6:7]
            sin_a = features[..., 7:8]
            alpha = torch.atan2(sin_a, cos_a)
            roll_parts = [cos_a, sin_a]
            for k in self.feature_spec.roll_harmonics:
                roll_parts.extend([torch.sin(k * alpha), torch.cos(k * alpha)])
            parts.append(torch.cat(roll_parts, dim=-1))
        return torch.cat(parts, dim=-1)

    def forward(
        self, features: "torch.Tensor"
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        h = F.silu(self.stem(self.encode(features)))
        for block in self.blocks:
            h = block(h)
        margin = self.head_margin(h)
        reach_logit = (
            margin * self.clearance_logit_scale
            if self.couple_reach_to_margin
            else self.head_cls(h)
        )
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
            "cost": cost,
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
        feature_kind = str(cfg.get("feature_kind", "pu6"))
        if feature_kind in {"natural_pu", "pu"}:
            feature_kind = "pu6"
        model = NeuralIRDPoint(
            in_dim=int(cfg.get("in_dim", 6)),
            feature_kind=feature_kind,
            num_freqs=int(cfg.get("num_freqs", 6)),
            num_freqs_u=int(cfg.get("num_freqs_u", 5)),
            hidden=int(cfg.get("hidden", 256)),
            depth=int(cfg.get("depth", 5)),
            tau_m=float(cfg.get("tau_m", 1.0)),
            lambda_q=float(cfg.get("lambda_q", 0.5)),
            p_wavelengths_m=cfg.get("p_wavelengths_m"),
            p_scale_m=float(cfg.get("p_scale_m", 1.0)),
            use_physical_pe=bool(cfg.get("use_physical_pe", True)),
            position_encoder=str(cfg.get("position_encoder", "fourier")),
            hash_levels=int(cfg.get("hash_levels", 12)),
            hash_features_per_level=int(cfg.get("hash_features_per_level", 2)),
            hash_log2_size=int(cfg.get("hash_log2_size", 18)),
            hash_base_resolution=int(cfg.get("hash_base_resolution", 8)),
            hash_max_resolution=int(cfg.get("hash_max_resolution", 256)),
            couple_reach_to_margin=bool(cfg.get("couple_reach_to_margin", False)),
            clearance_logit_scale=float(cfg.get("clearance_logit_scale", 3.0)),
        )
        model.load_state_dict(ckpt["state_dict"], strict=True)
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
            "in_dim": int(self.model.in_dim),
            "feature_kind": getattr(self.model, "feature_kind", "pu6"),
            "num_freqs": self.model.num_freqs,
            "num_freqs_u": self.model.num_freqs_u,
            "hidden": self.model.hidden,
            "depth": self.model.depth,
            "tau_m": self.model.tau_m,
            "lambda_q": self.model.lambda_q,
            "use_physical_pe": self.model.use_physical_pe,
            "position_encoder": self.model.position_encoder,
            "hash_levels": getattr(self.model.hash_encoder, "levels", 12),
            "hash_features_per_level": getattr(self.model.hash_encoder, "features_per_level", 2),
            "hash_log2_size": int(np.log2(getattr(self.model.hash_encoder, "hash_size", 1 << 18))),
            "hash_base_resolution": int(self.model.hash_encoder.resolution_values[0]) if self.model.hash_encoder else 8,
            "hash_max_resolution": int(self.model.hash_encoder.resolution_values[-1]) if self.model.hash_encoder else 256,
            "couple_reach_to_margin": self.model.couple_reach_to_margin,
            "clearance_logit_scale": self.model.clearance_logit_scale,
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
        from ird_playground.probe.se3 import features_from_delta_T, rot6d_features_from_delta_T

        feat = (
            rot6d_features_from_delta_T(delta_T)
            if self.model.feature_spec.use_rot6d
            else features_from_delta_T(delta_T)
        )
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
        from ird_playground.probe.se3 import (
            batch_features_from_delta_T,
            batch_rot6d_features_from_delta_T,
        )

        features = (
            batch_rot6d_features_from_delta_T(delta_Ts)
            if self.model.feature_spec.use_rot6d
            else batch_features_from_delta_T(delta_Ts)
        )
        return self.score_features_np(features)

    def region_score(self, **kwargs):
        from ird_playground.region.aggregate import region_score_a

        return region_score_a(self, **kwargs)
