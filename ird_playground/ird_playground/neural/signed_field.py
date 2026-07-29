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

from ird_playground.ird.canonical import (
    FLANGE_CANONICAL_DIM,
    canonical_flange_from_world_torch,
)
from ird_playground.ird.robot_model import (
    RobotModelSpec,
    assert_robot_contract_compatible,
)


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
    """Smooth scalar clearance on the 9-D flange chart; positive means reachable."""

    def __init__(
        self,
        *,
        width: int = 192,
        depth: int = 5,
        fourier_bands: int = 3,
        softplus_beta: float = 20.0,
        input_center: np.ndarray | None = None,
        input_scale: np.ndarray | None = None,
        T_flange_tcp: np.ndarray | None = None,
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
        if center.size != FLANGE_CANONICAL_DIM or scale.size != FLANGE_CANONICAL_DIM:
            raise ValueError(
                f"signed field expects {FLANGE_CANONICAL_DIM}-D flange chart "
                f"center/scale, got {center.size}/{scale.size}"
            )
        self.register_buffer(
            "input_center", torch.as_tensor(center).reshape(FLANGE_CANONICAL_DIM)
        )
        self.register_buffer(
            "input_scale",
            torch.as_tensor(scale).reshape(FLANGE_CANONICAL_DIM).clamp_min(1.0e-6),
        )
        tool = np.eye(4, dtype=np.float32) if T_flange_tcp is None else np.asarray(T_flange_tcp, dtype=np.float32)
        self.register_buffer("T_flange_tcp", torch.as_tensor(tool).reshape(4, 4))
        encoded_dim = FLANGE_CANONICAL_DIM * (1 + 2 * self.fourier_bands)
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
        if canonical.shape[-1] != FLANGE_CANONICAL_DIM:
            raise ValueError(
                f"expected {FLANGE_CANONICAL_DIM}-D flange chart, got {canonical.shape[-1]}"
            )
        return self.forward_normalized(self.normalize(canonical))

    def score_world(self, T_tcp_world: "torch.Tensor", T_axis_world: "torch.Tensor") -> "torch.Tensor":
        """Score a TCP pose via the 9-D flange chart in the J1-axis frame."""
        return self(
            canonical_flange_from_world_torch(T_tcp_world, T_axis_world, self.T_flange_tcp)
        )

class ReachabilitySDF:
    def __init__(
        self,
        model: SignedReachabilityField,
        device: str | None = None,
        *,
        meta: dict | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()
        self.meta = dict(meta or {})

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str | None = None,
        *,
        expected_robot: RobotModelSpec | None = None,
        allow_stale: bool = False,
    ) -> "ReachabilitySDF":
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        cfg = blob["model_config"]
        model = SignedReachabilityField(
            width=int(cfg["width"]),
            depth=int(cfg["depth"]),
            fourier_bands=int(cfg["fourier_bands"]),
            softplus_beta=float(cfg["softplus_beta"]),
            input_center=np.asarray(cfg["input_center"], dtype=np.float32),
            input_scale=np.asarray(cfg["input_scale"], dtype=np.float32),
            T_flange_tcp=(
                np.asarray(cfg["T_flange_tcp"], dtype=np.float32)
                if cfg.get("T_flange_tcp") is not None
                else None
            ),
        )
        model.load_state_dict(blob["state_dict"])
        meta = dict(blob.get("meta") or {})
        recorded = meta.get("robot_contract")
        if recorded is None:
            recorded = dict(meta.get("dataset_meta") or {}).get("robot_contract")
        assert_robot_contract_compatible(
            recorded,
            expected_robot,
            allow_stale=allow_stale,
        )
        return cls(model, device=device, meta=meta)

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
            "T_flange_tcp": self.model.T_flange_tcp.detach().cpu().tolist(),
            "canonical_dim": int(self.model.input_center.numel()),
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
