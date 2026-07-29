"""Uniform full-workspace pose labels for suppressing neural far-field islands.

Same flange / SRS / collision contract as ``gpu_pose_gt``: features are flange
se3_rot6d9, IK targets are TCP, reachability is controller-aligned SRS with a
fixed default branch (no generating seed).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import json
import numpy as np
from scipy.spatial.transform import Rotation

from ird_playground.ird.canonical import FLANGE_CANONICAL_DIM
from ird_playground.ird.gpu_pose_gt import (
    DEFAULT_QUERY_BRANCH_ID,
    GpuPoseGtConfig,
    _flange_canonical_np,
    _flange_se3_features,
    _make_srs_cfg,
    _probe_collision_filter,
    _srs_label_tcp,
)
from ird_playground.ird.gt_common import reachability_modules
from ird_playground.ird.metric import metric_manifest
from ird_playground.ird.robot_model import load_robot_model_spec
from ird_playground.ird.torch_kinematics import TorchRM75Kinematics, so3_exp
from ird_playground.probe.transform import default_ultrasound_probe


@dataclass(frozen=True)
class UniformPoseGtConfig:
    seed_gt_npz: str = "data/ird/gpu_pose_production.npz"
    output_npz: str = "data/ird/gpu_pose_uniform_production.npz"
    n_queries: int = 500_000
    n_ik_seeds: int = 16  # unused with SRS; YAML compat
    batch_size: int = 2048
    radius_m: float = 1.2
    z_min_m: float = -0.35
    z_max_m: float = 1.35
    horizontal_probe_fraction: float = 0.20
    horizontal_tilt_max_deg: float = 15.0
    ik_max_iter: int = 100
    default_branch_id: int = DEFAULT_QUERY_BRANCH_ID
    seed: int = 83
    device: str = "cuda"
    log_every_batches: int = 10
    robot_spec: str | None = None
    collision_security_margin_m: float | None = None


def _unit_vectors(rng: np.random.Generator, n: int) -> np.ndarray:
    x = rng.normal(size=(n, 3))
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1.0e-12)


def build_uniform_pose_gt(cfg: UniformPoseGtConfig) -> tuple[dict[str, np.ndarray], dict]:
    import torch

    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    # seed_gt is only used to confirm the robot contract / existence; SRS needs
    # no q pool.  Keep the path for YAML compat and provenance.
    seed_path = Path(cfg.seed_gt_npz)
    if seed_path.is_file():
        _ = np.load(seed_path, allow_pickle=False)
    spec = load_robot_model_spec(cfg.robot_spec)
    *_, SelfCollisionFilter, build_locked_rail_model = reachability_modules()
    locked = build_locked_rail_model(
        spec.kinematics_urdf,
        rail_locked_at_m=spec.rail_locked_at_m,
        tcp_frame=spec.tcp_frame,
    )
    collision_filter, collision_urdf, collision_pairs = _probe_collision_filter(
        GpuPoseGtConfig(
            robot_spec=cfg.robot_spec,
            collision_security_margin_m=cfg.collision_security_margin_m,
            default_branch_id=cfg.default_branch_id,
        ),
        locked,
        SelfCollisionFilter,
        spec=spec,
    )
    kin = TorchRM75Kinematics.from_locked_model(locked, device=cfg.device)
    tool = spec.tool_frame()
    T_flange_tcp = torch.as_tensor(
        tool.T_flange_tcp, dtype=torch.float32, device=kin.device
    )
    srs_cfg = _make_srs_cfg(spec, cfg.default_branch_id)
    horizontal_R = torch.as_tensor(
        default_ultrasound_probe().rotation_matrix(),
        dtype=torch.float32,
        device=kin.device,
    )
    out_features, out_labels = [], []
    out_branch, out_psi, out_qbest = [], [], []
    reachable_count = 0
    for batch_index, start in enumerate(range(0, cfg.n_queries, cfg.batch_size)):
        stop = min(cfg.n_queries, start + cfg.batch_size)
        n = stop - start
        radius = np.sqrt(rng.random(n)) * cfg.radius_m
        phi = rng.uniform(-np.pi, np.pi, size=n)
        p_np = np.stack(
            (
                radius * np.cos(phi),
                radius * np.sin(phi),
                rng.uniform(cfg.z_min_m, cfg.z_max_m, size=n),
            ),
            axis=-1,
        ).astype(np.float32)
        R_np = Rotation.random(n, random_state=rng).as_matrix().astype(np.float32)
        task_like = rng.random(n) < cfg.horizontal_probe_fraction
        p = torch.as_tensor(p_np, device=kin.device)
        R = torch.as_tensor(R_np, device=kin.device)
        if task_like.any():
            count = int(task_like.sum())
            axis = _unit_vectors(rng, count)
            angle = rng.uniform(
                0.0, np.deg2rad(cfg.horizontal_tilt_max_deg), size=count
            )
            dw = torch.as_tensor(
                axis * angle[:, None], dtype=torch.float32, device=kin.device
            )
            R[torch.as_tensor(task_like, device=kin.device)] = so3_exp(dw) @ horizontal_R
        branch_ids = np.full(n, cfg.default_branch_id, dtype=np.int32)
        psi_homes = np.zeros(n, dtype=np.float32)
        labeled = _srs_label_tcp(
            p.detach().cpu().numpy(),
            R.detach().cpu().numpy(),
            srs_cfg=srs_cfg,
            branch_ids=branch_ids,
            psi_homes=psi_homes,
            collision_filter=collision_filter,
        )
        label = labeled["reachable"].astype(np.float32)
        reachable_count += int(label.sum())
        out_features.append(
            _flange_se3_features(p, R, T_flange_tcp).cpu().numpy().astype(np.float32)
        )
        out_labels.append(label)
        out_branch.append(labeled["branch"])
        out_psi.append(labeled["psi"])
        out_qbest.append(labeled["q_best"])
        if cfg.log_every_batches and (batch_index + 1) % cfg.log_every_batches == 0:
            print(
                f"[uniform-gt] {stop}/{cfg.n_queries} reachable={reachable_count} "
                f"unreachable={stop - reachable_count}",
                flush=True,
            )
    features = np.concatenate(out_features)
    flange_canonical = _flange_canonical_np(
        features, np.eye(4, dtype=np.float64), spec.root_to_j1_axis()
    )
    arrays = {
        "features": features,
        "flange_canonical": flange_canonical,
        "reachable": np.concatenate(out_labels),
        "branch_ids": np.concatenate(out_branch),
        "psi": np.concatenate(out_psi),
        "psi_homes": np.zeros(cfg.n_queries, dtype=np.float32),
        "q_best": np.concatenate(out_qbest),
        "FLANGE_CANONICAL_DIM": np.asarray([FLANGE_CANONICAL_DIM], dtype=np.int32),
    }
    meta = {
        "label_kind": "uniform_full_workspace_srs_collision_v1",
        "feature_kind": "flange_se3_rot6d9",
        "feature_frame": "flange",
        "ik_target_frame": "tcp",
        "FLANGE_CANONICAL_DIM": FLANGE_CANONICAL_DIM,
        "n": cfg.n_queries,
        "n_reachable": reachable_count,
        "n_unreachable": cfg.n_queries - reachable_count,
        "collision_urdf": str(collision_urdf),
        "collision_pairs": str(collision_pairs),
        "collision_security_margin_m": float(
            cfg.collision_security_margin_m
            if cfg.collision_security_margin_m is not None
            else spec.collision_security_margin_m
        ),
        "T_flange_tcp": tool.T_flange_tcp.tolist(),
        "metric": metric_manifest(lambda_m_per_rad=spec.metric_lambda_m_per_rad),
        "robot_contract": spec.to_manifest(),
        "srs_conditioning": {
            **srs_cfg.to_manifest(),
            "branch_id_policy": (
                f"fixed default_branch_id={cfg.default_branch_id} for pure TCP queries "
                "(no generating seed)"
            ),
            "psi_home_policy": "psi_home_rad=0.0 for pure TCP queries",
            "default_branch_id": int(cfg.default_branch_id),
            "y_rail_m": srs_cfg.resolved_y_rail_m(),
            "point_only_reachability": True,
        },
        "seed_gt_npz": str(cfg.seed_gt_npz),
        "config": asdict(cfg),
    }
    return arrays, meta


def save_uniform_pose_gt(path: str | Path, arrays: dict[str, np.ndarray], meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(
        partial, **arrays, meta_json=np.asarray(json.dumps(meta, sort_keys=True))
    )
    partial.replace(path)
    # Sidecar YAML for robot_contract / SRS conditioning (mirrors other GT writers).
    import yaml

    yaml_path = path.with_suffix(".yaml")
    yaml_partial = yaml_path.with_name(yaml_path.stem + ".partial.yaml")
    yaml_partial.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    yaml_partial.replace(yaml_path)


__all__ = ["UniformPoseGtConfig", "build_uniform_pose_gt", "save_uniform_pose_gt"]
