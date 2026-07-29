"""Tolerance-conditioned clearance field distilled from a base signed field."""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

from ird_playground.ird.canonical import FLANGE_CANONICAL_DIM
from ird_playground.neural.signed_field import SmoothResidualBlock

RHO_DIM = 9


class ToleranceConditionedField(nn.Module if nn is not None else object):  # type: ignore[misc]
    """MLP over flange 9-D chart + tolerance descriptor ``rho`` (9-D)."""

    def __init__(
        self,
        *,
        width: int = 192,
        depth: int = 5,
        fourier_bands: int = 3,
        softplus_beta: float = 20.0,
        input_center: np.ndarray | None = None,
        input_scale: np.ndarray | None = None,
        rho_center: np.ndarray | None = None,
        rho_scale: np.ndarray | None = None,
    ) -> None:
        if torch is None:
            raise ImportError("torch required")
        super().__init__()
        self.width = int(width)
        self.depth = int(depth)
        self.fourier_bands = int(fourier_bands)
        self.softplus_beta = float(softplus_beta)
        center = (
            np.zeros(FLANGE_CANONICAL_DIM, dtype=np.float32)
            if input_center is None
            else np.asarray(input_center, dtype=np.float32)
        )
        scale = (
            np.ones(FLANGE_CANONICAL_DIM, dtype=np.float32)
            if input_scale is None
            else np.asarray(input_scale, dtype=np.float32)
        )
        rho_c = np.zeros(RHO_DIM, dtype=np.float32) if rho_center is None else np.asarray(rho_center, dtype=np.float32)
        rho_s = np.ones(RHO_DIM, dtype=np.float32) if rho_scale is None else np.asarray(rho_scale, dtype=np.float32)
        self.register_buffer("input_center", torch.as_tensor(center).reshape(FLANGE_CANONICAL_DIM))
        self.register_buffer("input_scale", torch.as_tensor(scale).reshape(FLANGE_CANONICAL_DIM).clamp_min(1.0e-6))
        self.register_buffer("rho_center", torch.as_tensor(rho_c).reshape(RHO_DIM))
        self.register_buffer("rho_scale", torch.as_tensor(rho_s).reshape(RHO_DIM).clamp_min(1.0e-6))
        encoded_dim = FLANGE_CANONICAL_DIM * (1 + 2 * self.fourier_bands) + RHO_DIM
        self.stem = nn.Linear(encoded_dim, self.width)
        self.blocks = nn.ModuleList(
            [SmoothResidualBlock(self.width, self.softplus_beta) for _ in range(max(1, self.depth - 1))]
        )
        self.head = nn.Linear(self.width, 1)
        nn.init.zeros_(self.head.bias)

    def normalize_canonical(self, canonical: "torch.Tensor") -> "torch.Tensor":
        return (canonical - self.input_center) / self.input_scale

    def normalize_rho(self, rho: "torch.Tensor") -> "torch.Tensor":
        return (rho - self.rho_center) / self.rho_scale

    def encode_canonical(self, x: "torch.Tensor") -> "torch.Tensor":
        parts = [x]
        for k in range(self.fourier_bands):
            phase = np.pi * (2.0**k) * x
            parts.extend((torch.sin(phase), torch.cos(phase)))
        return torch.cat(parts, dim=-1)

    def forward(self, canonical: "torch.Tensor", rho: "torch.Tensor") -> "torch.Tensor":
        if canonical.shape[-1] != FLANGE_CANONICAL_DIM:
            raise ValueError(f"expected {FLANGE_CANONICAL_DIM}-D canonical chart")
        if rho.shape[-1] != RHO_DIM:
            raise ValueError(f"expected {RHO_DIM}-D rho descriptor")
        x = self.encode_canonical(self.normalize_canonical(canonical))
        r = self.normalize_rho(rho)
        h = F.softplus(self.stem(torch.cat((x, r), dim=-1)), beta=self.softplus_beta)
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(-1)


def build_rho_descriptor(
    *,
    box_btn: tuple[float, float, float] = (0.0, 0.0, 0.0),
    cone_half_angle_rad: float = 0.0,
    roll_lo_rad: float = 0.0,
    roll_hi_rad: float = 0.0,
    free_flag: float = 0.0,
    uncertain_flag: float = 0.0,
    cvar_level: float = 0.0,
) -> np.ndarray:
    """Pack the documented ``rho`` descriptor (9-D)."""
    return np.asarray(
        [
            *box_btn,
            cone_half_angle_rad,
            roll_lo_rad,
            roll_hi_rad,
            free_flag,
            uncertain_flag,
            cvar_level,
        ],
        dtype=np.float32,
    )


__all__ = ["RHO_DIM", "ToleranceConditionedField", "build_rho_descriptor"]
