"""Single-output signed reachability field on the RM4D quotient space."""

from __future__ import annotations

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

from ird_playground.ird.canonical import canonical_from_world_torch


class SmoothResidualBlock(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(self, width: int, beta: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.beta = float(beta)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        h = F.softplus(self.fc1(x), beta=self.beta)
        return F.softplus(x + self.fc2(h), beta=self.beta)


class SignedReachabilityField(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Smooth scalar clearance; positive means reachable."""

    def __init__(
        self,
        *,
        width: int = 192,
        depth: int = 5,
        fourier_bands: int = 3,
        softplus_beta: float = 20.0,
        input_center: np.ndarray | None = None,
        input_scale: np.ndarray | None = None,
    ) -> None:
        if torch is None:
            raise ImportError("torch required")
        super().__init__()
        self.width = int(width)
        self.depth = int(depth)
        self.fourier_bands = int(fourier_bands)
        self.softplus_beta = float(softplus_beta)
        center = np.zeros(5, dtype=np.float32) if input_center is None else np.asarray(input_center, dtype=np.float32)
        scale = np.ones(5, dtype=np.float32) if input_scale is None else np.asarray(input_scale, dtype=np.float32)
        self.register_buffer("input_center", torch.as_tensor(center).reshape(5))
        self.register_buffer("input_scale", torch.as_tensor(scale).reshape(5).clamp_min(1.0e-6))
        encoded_dim = 5 * (1 + 2 * self.fourier_bands)
        self.stem = nn.Linear(encoded_dim, self.width)
        self.blocks = nn.ModuleList(
            [SmoothResidualBlock(self.width, self.softplus_beta) for _ in range(max(1, self.depth - 1))]
        )
        self.head = nn.Linear(self.width, 1)
        nn.init.zeros_(self.head.bias)

    def normalize(self, canonical: "torch.Tensor") -> "torch.Tensor":
        return (canonical - self.input_center) / self.input_scale

    def encode_normalized(self, x: "torch.Tensor") -> "torch.Tensor":
        parts = [x]
        for k in range(self.fourier_bands):
            phase = np.pi * (2.0**k) * x
            parts.extend((torch.sin(phase), torch.cos(phase)))
        return torch.cat(parts, dim=-1)

    def forward_normalized(self, x: "torch.Tensor") -> "torch.Tensor":
        h = F.softplus(self.stem(self.encode_normalized(x)), beta=self.softplus_beta)
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(-1)

    def forward(self, canonical: "torch.Tensor") -> "torch.Tensor":
        return self.forward_normalized(self.normalize(canonical))

    def score_world(self, T_tcp_world: "torch.Tensor", T_base_world: "torch.Tensor") -> "torch.Tensor":
        return self(canonical_from_world_torch(T_tcp_world, T_base_world))


class ReachabilitySDF:
    def __init__(self, model: SignedReachabilityField, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "ReachabilitySDF":
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        cfg = blob["model_config"]
        model = SignedReachabilityField(
            width=int(cfg["width"]),
            depth=int(cfg["depth"]),
            fourier_bands=int(cfg["fourier_bands"]),
            softplus_beta=float(cfg["softplus_beta"]),
            input_center=np.asarray(cfg["input_center"], dtype=np.float32),
            input_scale=np.asarray(cfg["input_scale"], dtype=np.float32),
        )
        model.load_state_dict(blob["state_dict"])
        return cls(model, device=device)

    def save(self, path: str | Path, *, meta: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cfg = {
            "width": self.model.width,
            "depth": self.model.depth,
            "fourier_bands": self.model.fourier_bands,
            "softplus_beta": self.model.softplus_beta,
            "input_center": self.model.input_center.detach().cpu().tolist(),
            "input_scale": self.model.input_scale.detach().cpu().tolist(),
        }
        torch.save({"state_dict": self.model.state_dict(), "model_config": cfg, "meta": meta or {}}, path)

    @torch.no_grad()
    def score_np(self, canonical: np.ndarray, *, batch_size: int = 131_072) -> np.ndarray:
        x = np.asarray(canonical, dtype=np.float32)
        out = []
        for start in range(0, len(x), batch_size):
            t = torch.as_tensor(x[start : start + batch_size], device=self.device)
            out.append(self.model(t).cpu().numpy())
        return np.concatenate(out)


__all__ = ["ReachabilitySDF", "SignedReachabilityField"]
