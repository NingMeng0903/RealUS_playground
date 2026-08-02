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

NEAR_AXIS_R_M = 0.05


def compute_input_stats(
    canonical: np.ndarray,
    *,
    quantile_lo: float = 0.005,
    quantile_hi: float = 0.995,
    min_scale: float = 1.0e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit Fourier-normalization center/scale from training chart rows.

    Defaults of ``center=0``, ``scale=1`` scramble metre and dimensionless
    channels under ``sin(π·2^k·x)`` and are forbidden for training.
    """
    x = np.asarray(canonical, dtype=np.float64)
    if x.ndim != 2 or x.shape[-1] != FLANGE_CANONICAL_DIM:
        raise ValueError(
            f"expected (*, {FLANGE_CANONICAL_DIM}) flange chart, got {x.shape}"
        )
    if x.shape[0] == 0:
        raise ValueError("cannot fit input stats on an empty chart array")
    lo = np.quantile(x, quantile_lo, axis=0)
    hi = np.quantile(x, quantile_hi, axis=0)
    center = (0.5 * (lo + hi)).astype(np.float32)
    scale = np.maximum(0.5 * (hi - lo), min_scale).astype(np.float32)
    assert_fitted_normalization(center, scale)
    return center, scale


def compute_support_box(canonical: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned min/max of training chart rows (query support)."""
    x = np.asarray(canonical, dtype=np.float64)
    if x.ndim != 2 or x.shape[-1] != FLANGE_CANONICAL_DIM:
        raise ValueError(
            f"expected (*, {FLANGE_CANONICAL_DIM}) flange chart, got {x.shape}"
        )
    if x.shape[0] == 0:
        raise ValueError("cannot fit a support box on an empty chart array")
    return x.min(axis=0).astype(np.float32), x.max(axis=0).astype(np.float32)


def assert_fitted_normalization(
    input_center: np.ndarray,
    input_scale: np.ndarray,
    *,
    atol: float = 1.0e-8,
) -> None:
    """Raise if center/scale look like the forbidden identity defaults."""
    center = np.asarray(input_center, dtype=np.float64).reshape(-1)
    scale = np.asarray(input_scale, dtype=np.float64).reshape(-1)
    if center.size != FLANGE_CANONICAL_DIM or scale.size != FLANGE_CANONICAL_DIM:
        raise ValueError(
            f"expected {FLANGE_CANONICAL_DIM}-D center/scale, "
            f"got {center.size}/{scale.size}"
        )
    if np.allclose(center, 0.0, atol=atol) and np.allclose(scale, 1.0, atol=atol):
        raise ValueError(
            "input_center/input_scale must be computed from training data stats; "
            "defaults of 0/1 are forbidden"
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
        support_lo: np.ndarray | None = None,
        support_hi: np.ndarray | None = None,
        guard_weight: float = 8.0,
    ) -> None:
        if torch is None:
            raise ImportError("torch required")
        super().__init__()
        self.width = int(width)
        self.depth = int(depth)
        self.fourier_bands = int(fourier_bands)
        self.guard_weight = float(guard_weight)
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
        for name, value in (("support_lo", support_lo), ("support_hi", support_hi)):
            box = (
                np.full(FLANGE_CANONICAL_DIM, np.nan, dtype=np.float32)
                if value is None
                else np.array(value, dtype=np.float32)
            )
            if box.size != FLANGE_CANONICAL_DIM:
                raise ValueError(
                    f"{name} must have {FLANGE_CANONICAL_DIM} entries, got {box.size}"
                )
            self.register_buffer(
                name, torch.as_tensor(box).reshape(FLANGE_CANONICAL_DIM).clone()
            )
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

    @property
    def has_support_box(self) -> bool:
        return bool(
            torch.isfinite(self.support_lo).all() and torch.isfinite(self.support_hi).all()
        )

    def support_distance(
        self, canonical: "torch.Tensor", *, eps: float = 1.0e-3
    ) -> "torch.Tensor":
        """Smooth normalized distance from the training support box (0 inside)."""
        if not self.has_support_box:
            return canonical.new_zeros(canonical.shape[:-1])
        x = self.normalize(canonical)
        lo = self.normalize(self.support_lo)
        hi = self.normalize(self.support_hi)
        outside = F.relu(lo - x) + F.relu(x - hi)
        return torch.sqrt((outside * outside).sum(dim=-1) + eps * eps) - eps

    def guarded_forward(self, canonical: "torch.Tensor") -> "torch.Tensor":
        """Clearance with a conservative exterior penalty outside the support."""
        clearance = self(canonical)
        if self.guard_weight == 0.0 or not self.has_support_box:
            return clearance
        return clearance - self.guard_weight * self.support_distance(canonical)

    def score_world(
        self, T_tcp_world: "torch.Tensor", T_axis_world: "torch.Tensor"
    ) -> "torch.Tensor":
        """Score a TCP pose on the flange chart in the J1-axis frame."""
        return self.guarded_forward(
            canonical_flange_from_world_torch(
                T_tcp_world, T_axis_world, self.T_flange_tcp
            )
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
        tool_cfg = cfg.get("T_flange_tcp")
        tool = (
            np.asarray(tool_cfg, dtype=np.float32)
            if tool_cfg is not None
            else None
        )
        if expected_robot is not None:
            tool_urdf = np.asarray(expected_robot.tool_frame().T_flange_tcp, dtype=np.float32)
            if tool is None or np.allclose(tool, np.eye(4, dtype=np.float32), atol=1.0e-8):
                tool = tool_urdf
        support = {
            name: (
                np.asarray(cfg[name], dtype=np.float32)
                if cfg.get(name) is not None
                else None
            )
            for name in ("support_lo", "support_hi")
        }
        model = SignedReachabilityField(
            width=int(cfg["width"]),
            depth=int(cfg["depth"]),
            fourier_bands=int(cfg["fourier_bands"]),
            softplus_beta=float(cfg["softplus_beta"]),
            input_center=np.asarray(cfg["input_center"], dtype=np.float32),
            input_scale=np.asarray(cfg["input_scale"], dtype=np.float32),
            T_flange_tcp=tool,
            guard_weight=float(cfg.get("guard_weight", 8.0)),
            **support,
        )
        state = dict(blob["state_dict"])
        if tool is not None:
            state.pop("T_flange_tcp", None)
        for name, value in support.items():
            if value is not None:
                state.pop(name, None)
        model.load_state_dict(state, strict=False)
        meta = dict(blob.get("meta") or {})
        if not allow_stale:
            required = {
                "artifact_schema": "ird_signed_field_v2",
                "calibration_schema": "ird_clearance_calibration_v2",
            }
            for key, expected in required.items():
                if meta.get(key) != expected:
                    raise ValueError(
                        f"checkpoint {path} has incompatible {key}={meta.get(key)!r}; "
                        f"expected {expected!r}. Rebuild it or pass allow_stale=True "
                        "for an explicit legacy audit."
                    )
            for key in ("dataset_sha256", "split_fingerprint", "sampler", "metric_schema", "output_scale"):
                if key not in meta:
                    raise ValueError(f"checkpoint {path} missing required metadata {key!r}")
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
            "guard_weight": float(self.model.guard_weight),
        }
        if self.model.has_support_box:
            cfg["support_lo"] = self.model.support_lo.detach().cpu().tolist()
            cfg["support_hi"] = self.model.support_hi.detach().cpu().tolist()
        torch.save({"state_dict": self.model.state_dict(), "model_config": cfg, "meta": meta or {}}, path)

    def score_world(
        self, T_tcp_world: "torch.Tensor", T_axis_world: "torch.Tensor"
    ) -> "torch.Tensor":
        return self.model.score_world(T_tcp_world, T_axis_world)

    @torch.no_grad()
    def score_np(self, canonical: np.ndarray, *, batch_size: int = 131_072) -> np.ndarray:
        x = np.asarray(canonical, dtype=np.float32)
        out = []
        for start in range(0, len(x), batch_size):
            t = torch.as_tensor(x[start : start + batch_size], device=self.device)
            out.append(self.model(t).cpu().numpy())
        return np.concatenate(out)


__all__ = [
    "NEAR_AXIS_R_M",
    "ReachabilitySDF",
    "SignedReachabilityField",
    "assert_fitted_normalization",
    "compute_input_stats",
    "compute_support_box",
]
