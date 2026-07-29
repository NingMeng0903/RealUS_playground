"""Dump chart-frame Jacobian generators as capacity-head supervision GT.

Reads a pose NPZ (typically ``data/ird/gpu_pose_production.npz``), keeps a
collision-free / reachable subset with finite ``q_best``, runs batched FK, and
writes yaw-canonical TCP linear-Jacobian columns for later capacity-head
training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

from ird_playground.ird.canonical import (
    FLANGE_CANONICAL_DIM,
    canonical_flange_invariants_torch,
)
from ird_playground.ird.gt_common import reachability_modules
from ird_playground.ird.robot_model import RobotModelSpec
from ird_playground.ird.torch_kinematics import TorchRM75Kinematics
from ird_playground.neural.capacity_head import (
    DEFAULT_QDOT_MAX,
    DEFAULT_TAU_MAX,
    tcp_linear_jacobian_chart,
)


@dataclass(frozen=True)
class JacobianGtConfig:
    source_npz: str
    output_npz: str
    robot_spec: str = "configs/robot_probe45.yaml"
    max_samples: int = 50_000
    seed: int = 47
    batch_size: int = 2048
    device: str = "cpu"
    require_reachable: bool = True
    require_collision_free: bool = True
    reachable_key: str = "reachable"
    collision_key: str = "q_selfcol"
    q_key: str = "q_best"
    chart_key: str = "flange_canonical"


def _as_bool_mask(arr: np.ndarray, n: int) -> np.ndarray:
    a = np.asarray(arr).reshape(-1)
    if a.shape[0] != n:
        raise ValueError(f"mask length {a.shape[0]} != {n}")
    if a.dtype == np.bool_:
        return a
    return a > 0.5


def select_jacobian_gt_indices(
    arrays: dict[str, np.ndarray],
    cfg: JacobianGtConfig,
) -> np.ndarray:
    """Indices of reachable / collision-free rows with finite ``q_best``."""
    q = np.asarray(arrays[cfg.q_key], dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 7:
        raise ValueError(f"expected {cfg.q_key} shape (N,7), got {q.shape}")
    n = int(q.shape[0])
    mask = np.isfinite(q).all(axis=1)
    if cfg.require_reachable:
        if cfg.reachable_key not in arrays:
            raise KeyError(f"missing {cfg.reachable_key!r} in source NPZ")
        mask &= _as_bool_mask(arrays[cfg.reachable_key], n)
    if cfg.require_collision_free:
        if cfg.collision_key not in arrays:
            raise KeyError(f"missing {cfg.collision_key!r} in source NPZ")
        mask &= _as_bool_mask(arrays[cfg.collision_key], n)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise RuntimeError("no collision-free / reachable samples with finite q")
    rng = np.random.default_rng(int(cfg.seed))
    if idx.size > int(cfg.max_samples):
        pick = rng.choice(idx.size, size=int(cfg.max_samples), replace=False)
        idx = np.sort(idx[pick])
    return idx.astype(np.int64)


def build_jacobian_gt(
    cfg: JacobianGtConfig,
    *,
    root: Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Compute chart-frame Jacobian generators for a pose-GT subset."""
    if torch is None:
        raise ImportError("torch required")
    root = Path(root) if root is not None else Path.cwd()
    source = Path(cfg.source_npz)
    if not source.is_absolute():
        source = root / source
    spec_path = Path(cfg.robot_spec)
    if not spec_path.is_absolute():
        spec_path = root / spec_path
    spec = RobotModelSpec.from_yaml(spec_path)
    *_, _SelfCollisionFilter, build_locked_rail_model = reachability_modules()
    locked = build_locked_rail_model(
        urdf_path=str(spec.kinematics_urdf),
        rail_locked_at_m=float(spec.rail_locked_at_m),
    )
    kin = TorchRM75Kinematics.from_locked_model(locked, device=cfg.device)
    tool = torch.as_tensor(
        np.asarray(spec.tool_frame().T_flange_tcp, dtype=np.float32),
        device=kin.device,
        dtype=kin.dtype,
    )

    raw = dict(np.load(source, allow_pickle=True))
    idx = select_jacobian_gt_indices(raw, cfg)
    q = np.asarray(raw[cfg.q_key], dtype=np.float32)[idx]
    if cfg.chart_key in raw:
        chart = np.asarray(raw[cfg.chart_key], dtype=np.float32)[idx]
        if chart.shape[-1] != FLANGE_CANONICAL_DIM:
            raise ValueError(
                f"{cfg.chart_key} dim {chart.shape[-1]} != {FLANGE_CANONICAL_DIM}"
            )
    else:
        chart = None

    generators = np.empty((idx.size, 3, 7), dtype=np.float32)
    charts_out = np.empty((idx.size, FLANGE_CANONICAL_DIM), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, idx.size, int(cfg.batch_size)):
            stop = min(idx.size, start + int(cfg.batch_size))
            q_t = torch.as_tensor(q[start:stop], device=kin.device, dtype=kin.dtype)
            gens, p_fl, R_fl = tcp_linear_jacobian_chart(kin, q_t, T_flange_tcp=tool)
            generators[start:stop] = gens.detach().cpu().numpy().astype(np.float32)
            if chart is None:
                charts_out[start:stop] = (
                    canonical_flange_invariants_torch(p_fl, R_fl)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            else:
                charts_out[start:stop] = chart[start:stop]

    out: dict[str, np.ndarray] = {
        "flange_canonical": charts_out,
        "jacobian_generators": generators,
        "q": q,
        "source_index": idx,
        "qdot_max": DEFAULT_QDOT_MAX.astype(np.float32),
        "tau_max": DEFAULT_TAU_MAX.astype(np.float32),
        "FLANGE_CANONICAL_DIM": np.asarray([FLANGE_CANONICAL_DIM], dtype=np.int32),
    }
    for key in (
        "branch_ids",
        "psi",
        "psi_homes",
        "source_pose_id",
        "block_id",
        "reachable",
        "q_selfcol",
    ):
        if key in raw:
            out[key] = np.asarray(raw[key])[idx]

    meta: dict[str, Any] = {
        "kind": "capacity_jacobian_gt_v1",
        "n": int(idx.size),
        "source_npz": str(source),
        "frame": "yaw_canonical_chart_tcp_linear",
        "n_joints": 7,
        "config": asdict(cfg),
        "robot_contract": spec.to_manifest(),
        "qdot_max": DEFAULT_QDOT_MAX.tolist(),
        "tau_max": DEFAULT_TAU_MAX.tolist(),
        "note": (
            "Generators are TCP linear Jacobian columns expressed in the "
            "yaw-canonical chart frame (R_chart^T @ J_v). Capacity training "
            "is deferred; this artifact is the supervision dump."
        ),
    }
    return out, meta


def save_jacobian_gt(
    path: str | Path,
    arrays: dict[str, np.ndarray],
    meta: dict[str, Any],
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(arrays)
    payload["meta_json"] = np.asarray([__import__("json").dumps(meta)])
    np.savez_compressed(out, **payload)
    return out


__all__ = [
    "JacobianGtConfig",
    "build_jacobian_gt",
    "save_jacobian_gt",
    "select_jacobian_gt_indices",
]
