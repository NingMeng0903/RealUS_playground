"""Build a 5-D flange-chart occupancy grid from collision-free FK samples.

Axes: ``(p_z, r, tilt, azimuth, gamma)`` at default 3 cm / 12°.
Phase-0 measured shape with those bounds is ``58×41×16×31×31`` (36.56M cells).

Voxel labels (uint8):
  * ``OCC_UNKNOWN = 0`` — never hit by a collision-free FK draw within budget
    (not proven unreachable)
  * ``OCC_OCCUPIED = 1`` — hit by ≥1 collision-free FK sample

Azimuth is relative: ``wrap_pi(atan2(uz_y, uz_x) - atan2(p_y, p_x))``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import json
import numpy as np

from ird_playground.ird.gt_common import reachability_modules
from ird_playground.ird.metric import metric_manifest
from ird_playground.ird.robot_model import load_robot_model_spec
from ird_playground.ird.torch_kinematics import TorchRM75Kinematics, collision_free_mask

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

CHART_NAMES = ("p_z", "r", "tilt", "azimuth", "gamma")

# Ternary-capable occupancy encoding (FK fill only uses unknown / occupied).
OCC_UNKNOWN: int = 0
OCC_OCCUPIED: int = 1

UNKNOWN_POLICY = (
    "Empty cells within the FK sample budget are unknown, not unreachable. "
    "Only voxels hit by ≥1 collision-free FK sample are marked occupied."
)


def _rotation_from_tilt_azimuth(
    tilt: np.ndarray,
    azimuth: np.ndarray,
) -> np.ndarray:
    """Reference flange rotation with ``gamma=0`` (columns ``[ux, uy, uz]``)."""
    st = np.sin(tilt)
    ct = np.cos(tilt)
    ca = np.cos(azimuth)
    sa = np.sin(azimuth)
    uz = np.stack((st * ca, st * sa, ct), axis=-1)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    ux = np.cross(up, uz)
    small = st < 1.0e-8
    ux[small] = np.array([1.0, 0.0, 0.0])
    ux = ux / np.maximum(np.linalg.norm(ux, axis=-1, keepdims=True), 1.0e-12)
    uy = np.cross(uz, ux)
    return np.stack((ux, uy, uz), axis=-1)


def _wrap_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def flange_pose_to_chart(
    position_axis: np.ndarray,
    rotation_axis: np.ndarray,
) -> np.ndarray:
    """Map flange pose in the J1-axis frame to ``(p_z, r, tilt, azimuth, gamma)``.

    ``azimuth`` is the *relative* bearing
    ``wrap_pi(atan2(uz_y, uz_x) - atan2(p_y, p_x))`` so a base-yaw orbit
    stays in one chart voxel.  At ``r → 0`` the relative azimuth is undefined
    and is set to 0.  ``gamma`` still uses the absolute ``uz`` azimuth for the
    reference frame.
    """
    p = np.asarray(position_axis, dtype=np.float64)
    R = np.asarray(rotation_axis, dtype=np.float64)
    if p.ndim == 1:
        p = p[None, :]
        R = R[None, :, :]
        squeeze = True
    else:
        squeeze = False
    pz = p[..., 2]
    r = np.hypot(p[..., 0], p[..., 1])
    uz = R[..., :, 2]
    tilt = np.arccos(np.clip(uz[..., 2], -1.0, 1.0))
    azimuth_uz = np.arctan2(uz[..., 1], uz[..., 0])
    azimuth_p = np.arctan2(p[..., 1], p[..., 0])
    azimuth = _wrap_pi(azimuth_uz - azimuth_p)
    azimuth = np.where(r < 1.0e-6, 0.0, azimuth)
    R_ref = _rotation_from_tilt_azimuth(tilt, azimuth_uz)
    R_delta = np.matmul(np.swapaxes(R_ref, -1, -2), R)
    gamma = np.arctan2(R_delta[..., 1, 0], R_delta[..., 0, 0])
    out = np.stack((pz, r, tilt, azimuth, gamma), axis=-1).astype(np.float32)
    return out[0] if squeeze else out


def chart_coords_to_indices(
    chart: np.ndarray,
    axes: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, ...]:
    """Return integer voxel indices for each chart coordinate (clipped in-bounds)."""
    chart = np.asarray(chart, dtype=np.float64)
    indices = []
    for dim, axis in enumerate(axes):
        idx = np.searchsorted(axis, chart[..., dim], side="right") - 1
        indices.append(np.clip(idx, 0, len(axis) - 1).astype(np.int32))
    return tuple(indices)


@dataclass(frozen=True)
class FlangeOccupancyConfig:
    output_npz: str = "data/maps/flange_occupancy.npz"
    n_samples: int = 10_000_000
    batch_size: int = 65_536
    step_m: float = 0.03
    step_deg: float = 12.0
    p_z_bounds_m: tuple[float, float] = (-0.35, 1.35)
    r_bounds_m: tuple[float, float] = (0.0, 1.20)
    tilt_bounds_rad: tuple[float, float] = (0.0, np.pi)
    azimuth_bounds_rad: tuple[float, float] = (-np.pi, np.pi)
    gamma_bounds_rad: tuple[float, float] = (-np.pi, np.pi)
    seed: int = 41
    device: str = "cpu"
    robot_spec: str | None = None
    require_collision_free: bool = True

    def axis_arrays(self) -> tuple[np.ndarray, ...]:
        step_rad = np.deg2rad(self.step_deg)
        return (
            np.arange(self.p_z_bounds_m[0], self.p_z_bounds_m[1] + 0.5 * self.step_m, self.step_m),
            np.arange(self.r_bounds_m[0], self.r_bounds_m[1] + 0.5 * self.step_m, self.step_m),
            np.arange(self.tilt_bounds_rad[0], self.tilt_bounds_rad[1] + 0.5 * step_rad, step_rad),
            np.arange(self.azimuth_bounds_rad[0], self.azimuth_bounds_rad[1] + 0.5 * step_rad, step_rad),
            np.arange(self.gamma_bounds_rad[0], self.gamma_bounds_rad[1] + 0.5 * step_rad, step_rad),
        )


def _axis_meta(axes: tuple[np.ndarray, ...], cfg: FlangeOccupancyConfig) -> dict:
    return {
        "names": list(CHART_NAMES),
        "shape": [int(len(a)) for a in axes],
        "n_cells": int(np.prod([len(a) for a in axes])),
        "step_m": float(cfg.step_m),
        "step_deg": float(cfg.step_deg),
        "bounds": {
            "p_z_m": list(cfg.p_z_bounds_m),
            "r_m": list(cfg.r_bounds_m),
            "tilt_rad": list(cfg.tilt_bounds_rad),
            "azimuth_rad": list(cfg.azimuth_bounds_rad),
            "gamma_rad": list(cfg.gamma_bounds_rad),
        },
        "azimuth_definition": (
            "relative: wrap_pi(atan2(uz_y, uz_x) - atan2(p_y, p_x)); "
            "0 at r < 1e-6"
        ),
        "metric": metric_manifest(),
        "note": "chart-coordinate weighted L2 used for EDT; not an SE(3) geodesic",
    }


def _pose_axis_from_tcp(
    p_tcp: "torch.Tensor",
    R_tcp: "torch.Tensor",
    T_root_axis: "torch.Tensor",
    T_flange_tcp: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor"]:
    from ird_playground.ird.canonical import pose_in_axis_frame_torch
    from ird_playground.ird.tool_frame import pose_tcp_to_flange

    p_axis, R_axis = pose_in_axis_frame_torch(p_tcp, R_tcp, T_root_axis)
    return pose_tcp_to_flange(p_axis, R_axis, T_flange_tcp)


def build_flange_occupancy(
    cfg: FlangeOccupancyConfig | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Sample collision-free configurations and mark occupied chart voxels."""
    if torch is None:
        raise ImportError("torch required")
    cfg = cfg or FlangeOccupancyConfig()
    spec = load_robot_model_spec(cfg.robot_spec)
    axes = cfg.axis_arrays()
    shape = tuple(len(a) for a in axes)
    # Default 0 = UNKNOWN (not unreachable).
    occupancy = np.zeros(shape, dtype=np.uint8)
    *_, SelfCollisionFilter, build_locked_rail_model = reachability_modules()
    locked = build_locked_rail_model(
        spec.kinematics_urdf,
        rail_locked_at_m=spec.rail_locked_at_m,
        tcp_frame=spec.tcp_frame,
    )
    collision_filter = None
    margin_m = float(spec.collision_security_margin_m)
    if cfg.require_collision_free:
        # SelfCollisionFilter is keyword-only; security_margin (not *_m).
        collision_filter = SelfCollisionFilter(
            kin_urdf=spec.kinematics_urdf,
            collision_urdf=spec.collision_urdf,
            pair_config=spec.collision_pairs,
            rail_locked_at_m=spec.rail_locked_at_m,
            security_margin=margin_m,
        )
    kin = TorchRM75Kinematics.from_locked_model(locked, device=cfg.device)
    T_root_axis = torch.as_tensor(
        spec.root_to_j1_axis(),
        dtype=torch.float32,
        device=kin.device,
    )
    tool = spec.tool_frame()
    T_flange_tcp = torch.as_tensor(tool.T_flange_tcp, dtype=torch.float32, device=kin.device)
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    q_lo = kin.q_lower.cpu().numpy()
    q_hi = kin.q_upper.cpu().numpy()
    filled = 0
    n_collision_rejected = 0
    for start in range(0, cfg.n_samples, cfg.batch_size):
        stop = min(cfg.n_samples, start + cfg.batch_size)
        q = rng.uniform(q_lo, q_hi, size=(stop - start, 7)).astype(np.float32)
        q_t = torch.as_tensor(q, device=kin.device)
        if collision_filter is not None:
            free = collision_free_mask(q_t, collision_filter, device=kin.device)
            n_collision_rejected += int((~free).sum().item())
            q_t = q_t[free]
            if q_t.numel() == 0:
                continue
        p_tcp, R_tcp = kin.fk(q_t)
        p_fl, R_fl = _pose_axis_from_tcp(p_tcp, R_tcp, T_root_axis, T_flange_tcp)
        chart = flange_pose_to_chart(
            p_fl.detach().cpu().numpy(),
            R_fl.detach().cpu().numpy(),
        )
        idx = chart_coords_to_indices(chart, axes)
        occupancy[idx] = OCC_OCCUPIED
        filled += int(len(chart))
    positive = int((occupancy == OCC_OCCUPIED).sum())
    unknown = int((occupancy == OCC_UNKNOWN).sum())
    arrays = {
        "occupancy": occupancy,
        **{name: axis.astype(np.float32) for name, axis in zip(CHART_NAMES, axes)},
    }
    meta = {
        "schema": "flange_occupancy_v1",
        "config": asdict(cfg),
        "axes": _axis_meta(axes, cfg),
        "label_encoding": {
            "OCC_UNKNOWN": OCC_UNKNOWN,
            "OCC_OCCUPIED": OCC_OCCUPIED,
        },
        "unknown_policy": UNKNOWN_POLICY,
        "n_samples_requested": int(cfg.n_samples),
        "n_fk_positives": int(filled),
        "n_collision_rejected": int(n_collision_rejected),
        "n_occupied_voxels": positive,
        "n_unknown_voxels": unknown,
        "occupancy_fraction": float(positive / max(occupancy.size, 1)),
        "collision": {
            "model": "arm-body SelfCollisionFilter",
            "security_margin_m": margin_m,
        },
        "robot_contract": spec.to_manifest(),
        "tool_frame": tool.to_manifest(),
        "metric": metric_manifest(),
    }
    out_path = Path(cfg.output_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Compressed uint8 occupancy stays small when sparse (smoke / early FK).
    np.savez_compressed(out_path, **arrays)
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return arrays, meta


__all__ = [
    "CHART_NAMES",
    "OCC_OCCUPIED",
    "OCC_UNKNOWN",
    "UNKNOWN_POLICY",
    "FlangeOccupancyConfig",
    "build_flange_occupancy",
    "chart_coords_to_indices",
    "flange_pose_to_chart",
]
