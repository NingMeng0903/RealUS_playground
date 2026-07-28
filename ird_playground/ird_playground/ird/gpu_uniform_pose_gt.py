"""Uniform full-workspace pose labels for suppressing neural far-field islands."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import json
import numpy as np
from scipy.spatial.transform import Rotation

from ird_playground.ird.gpu_pose_gt import GpuPoseGtConfig, _features, _probe_collision_filter
from ird_playground.ird.gt_common import reachability_modules
from ird_playground.ird.robot_model import load_robot_model_spec
from ird_playground.ird.torch_kinematics import TorchRM75Kinematics, select_collision_free_ik, so3_exp
from ird_playground.probe.transform import default_ultrasound_probe


@dataclass(frozen=True)
class UniformPoseGtConfig:
    seed_gt_npz: str = "data/ird/gpu_pose_production.npz"
    output_npz: str = "data/ird/gpu_pose_uniform_production.npz"
    n_queries: int = 500_000
    n_ik_seeds: int = 16
    batch_size: int = 2048
    radius_m: float = 1.2
    z_min_m: float = -0.35
    z_max_m: float = 1.35
    horizontal_probe_fraction: float = 0.20
    horizontal_tilt_max_deg: float = 15.0
    ik_max_iter: int = 100
    seed: int = 83
    device: str = "cuda"
    log_every_batches: int = 10
    robot_spec: str | None = None


def _unit_vectors(rng: np.random.Generator, n: int) -> np.ndarray:
    x = rng.normal(size=(n, 3))
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1.0e-12)


def build_uniform_pose_gt(cfg: UniformPoseGtConfig) -> tuple[dict[str, np.ndarray], dict]:
    import torch

    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    seed_data = np.load(cfg.seed_gt_npz, allow_pickle=False)
    q_pool_np = seed_data["q_best"][seed_data["reachable"] > 0.5]
    q_pool_np = q_pool_np[np.any(q_pool_np != 0.0, axis=1)].astype(np.float32)
    spec = load_robot_model_spec(cfg.robot_spec)
    *_, SelfCollisionFilter, build_locked_rail_model = reachability_modules()
    locked = build_locked_rail_model(
        spec.kinematics_urdf,
        rail_locked_at_m=spec.rail_locked_at_m,
        tcp_frame=spec.tcp_frame,
    )
    collision_filter, collision_urdf, collision_pairs = _probe_collision_filter(
        GpuPoseGtConfig(robot_spec=cfg.robot_spec),
        locked,
        SelfCollisionFilter,
        spec=spec,
    )
    kin = TorchRM75Kinematics.from_locked_model(locked, device=cfg.device)
    q_pool = torch.as_tensor(q_pool_np, device=kin.device)
    horizontal_R = torch.as_tensor(
        default_ultrasound_probe().rotation_matrix(), dtype=torch.float32, device=kin.device
    )
    out_features, out_labels = [], []
    reachable_count = 0
    for batch_index, start in enumerate(range(0, cfg.n_queries, cfg.batch_size)):
        stop = min(cfg.n_queries, start + cfg.batch_size)
        n = stop - start
        radius = np.sqrt(rng.random(n)) * cfg.radius_m
        phi = rng.uniform(-np.pi, np.pi, size=n)
        p_np = np.stack(
            (radius * np.cos(phi), radius * np.sin(phi), rng.uniform(cfg.z_min_m, cfg.z_max_m, size=n)),
            axis=-1,
        ).astype(np.float32)
        R_np = Rotation.random(n, random_state=rng).as_matrix().astype(np.float32)
        task_like = rng.random(n) < cfg.horizontal_probe_fraction
        p = torch.as_tensor(p_np, device=kin.device)
        R = torch.as_tensor(R_np, device=kin.device)
        if task_like.any():
            count = int(task_like.sum())
            axis = _unit_vectors(rng, count)
            angle = rng.uniform(0.0, np.deg2rad(cfg.horizontal_tilt_max_deg), size=count)
            dw = torch.as_tensor(axis * angle[:, None], dtype=torch.float32, device=kin.device)
            R[torch.as_tensor(task_like, device=kin.device)] = so3_exp(dw) @ horizontal_R
        seed_idx = rng.integers(0, len(q_pool_np), size=(n, cfg.n_ik_seeds))
        q0 = q_pool[torch.as_tensor(seed_idx, device=kin.device)]
        result = kin.ik_dls(
            p,
            R,
            q0,
            max_iter=cfg.ik_max_iter,
            tol_pos_m=2.0e-4,
            tol_rot_rad=1.0e-3,
        )
        checked = select_collision_free_ik(
            result,
            collision_filter,
            tol_pos_m=2.0e-4,
            tol_rot_rad=1.0e-3,
        )
        label = checked.reachable.cpu().numpy().astype(np.float32)
        reachable_count += int(label.sum())
        out_features.append(_features(p, R).cpu().numpy().astype(np.float32))
        out_labels.append(label)
        if cfg.log_every_batches and (batch_index + 1) % cfg.log_every_batches == 0:
            print(
                f"[uniform-gt] {stop}/{cfg.n_queries} reachable={reachable_count} "
                f"unreachable={stop-reachable_count}",
                flush=True,
            )
    arrays = {
        "features": np.concatenate(out_features),
        "reachable": np.concatenate(out_labels),
    }
    meta = {
        "label_kind": "uniform_full_workspace_gpu_multiseed_collision_v1",
        "n": cfg.n_queries,
        "n_reachable": reachable_count,
        "n_unreachable": cfg.n_queries - reachable_count,
        "collision_urdf": str(collision_urdf),
        "collision_pairs": str(collision_pairs),
        "robot_contract": spec.to_manifest(),
        "config": asdict(cfg),
    }
    return arrays, meta


def save_uniform_pose_gt(path: str | Path, arrays: dict[str, np.ndarray], meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays, meta_json=np.asarray(json.dumps(meta, sort_keys=True)))


__all__ = ["UniformPoseGtConfig", "build_uniform_pose_gt", "save_uniform_pose_gt"]
