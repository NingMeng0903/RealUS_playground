"""Collision-aware GPU pose sampling for the first-stage Neural IRD field."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from ird_playground.ird.gt_common import block_ids, reachability_modules
from ird_playground.ird.robot_model import RobotModelSpec, load_robot_model_spec
from ird_playground.ird.export_gt import LAYER_EXTERIOR, LAYER_INTERIOR
from ird_playground.ird.torch_kinematics import (
    TorchRM75Kinematics,
    select_collision_free_ik,
    so3_exp,
)


@dataclass(frozen=True)
class GpuPoseGtConfig:
    n_fk_positive: int = 100_000
    n_pose_queries: int = 100_000
    n_ik_seeds: int = 12
    batch_size: int = 4096
    joint_limit_inset_fraction: float = 0.02
    near_query_fraction: float = 0.50
    near_pos_max_m: float = 0.04
    near_rot_max_deg: float = 12.0
    far_pos_min_m: float = 0.04
    far_pos_max_m: float = 0.35
    far_rot_min_deg: float = 10.0
    far_rot_max_deg: float = 70.0
    ik_max_iter: int = 100
    ik_damping: float = 0.002
    ik_step_size: float = 0.8
    ik_max_step_rad: float = 0.25
    ik_tol_pos_m: float = 2.0e-4
    ik_tol_rot_rad: float = 1.0e-3
    collision_security_margin_m: float = 0.0
    collision_urdf: str | None = None
    collision_pairs: str | None = None
    robot_spec: str | None = None
    holdout_block_m: float = 0.04
    m_eps: float = 1.0e-3
    seed: int = 42
    device: str = "cuda"
    log_every_batches: int = 10

    def validate(self) -> None:
        if self.n_fk_positive <= 0 or self.n_pose_queries <= 0:
            raise ValueError("sample counts must be positive")
        if self.n_ik_seeds < 2:
            raise ValueError("n_ik_seeds must be at least 2")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= self.joint_limit_inset_fraction < 0.5:
            raise ValueError("joint_limit_inset_fraction must lie in [0,0.5)")
        if not 0.0 <= self.near_query_fraction <= 1.0:
            raise ValueError("near_query_fraction must lie in [0,1]")


def _robot_spec_for_cfg(cfg: GpuPoseGtConfig) -> RobotModelSpec:
    spec = load_robot_model_spec(cfg.robot_spec)
    if cfg.collision_urdf:
        spec = replace(spec, collision_urdf=Path(cfg.collision_urdf).resolve())
    if cfg.collision_pairs:
        spec = replace(spec, collision_pairs=Path(cfg.collision_pairs).resolve())
    spec.validate()
    return spec


def _probe_collision_filter(
    cfg: GpuPoseGtConfig,
    lm,
    SelfCollisionFilter,
    *,
    spec: RobotModelSpec | None = None,
):
    spec = spec or _robot_spec_for_cfg(cfg)
    collision_urdf = spec.collision_urdf
    collision_pairs = spec.collision_pairs
    return SelfCollisionFilter(
        kin_urdf=lm.urdf_path,
        collision_urdf=collision_urdf,
        pair_config=collision_pairs,
        rail_locked_at_m=lm.rail_locked_at_m,
        security_margin=cfg.collision_security_margin_m,
    ), collision_urdf, collision_pairs


def _sample_collision_free_q(cfg, lm, collision_filter, rng) -> np.ndarray:
    span = lm.q_upper - lm.q_lower
    inset = cfg.joint_limit_inset_fraction * span
    lo, hi = lm.q_lower + inset, lm.q_upper - inset
    chunks: list[np.ndarray] = []
    count = 0
    draw = max(cfg.batch_size, min(50_000, cfg.n_fk_positive))
    while count < cfg.n_fk_positive:
        Q = rng.uniform(lo, hi, size=(draw, 7))
        Q = Q[collision_filter.free_mask(Q)]
        chunks.append(Q)
        count += len(Q)
    return np.concatenate(chunks, axis=0)[: cfg.n_fk_positive]


def _random_unit(rng: np.random.Generator, n: int) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)


def _quality(kin: TorchRM75Kinematics, q):
    import torch

    _p, _R, J = kin.fk(q, return_jacobian=True)
    sigma = torch.linalg.svdvals(J)[..., -1]
    span = (kin.q_upper - kin.q_lower).clamp_min(1e-6)
    clearance = torch.minimum(q - kin.q_lower, kin.q_upper - q) / span
    joint = clearance.amin(dim=-1)
    return ((sigma / 0.08).clamp(0.0, 1.0) * (joint / 0.08).clamp(0.0, 1.0))


def _features(p, R):
    import torch

    return torch.cat([p, R[..., :, 0], R[..., :, 1]], dim=-1)


def build_gpu_pose_gt(
    cfg: GpuPoseGtConfig | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Generate independent full-pose labels with mandatory self-collision checks."""
    import torch

    cfg = cfg or GpuPoseGtConfig()
    cfg.validate()
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    (
        _ik_dls, _ik_multi, _SeedPoolConfig, _build_seed_pool,
        _halton, SelfCollisionFilter, build_locked_rail_model,
    ) = reachability_modules()
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    spec = _robot_spec_for_cfg(cfg)
    lm = build_locked_rail_model(
        spec.kinematics_urdf,
        rail_locked_at_m=spec.rail_locked_at_m,
        tcp_frame=spec.tcp_frame,
    )
    collision_filter, collision_urdf, collision_pairs = _probe_collision_filter(
        cfg, lm, SelfCollisionFilter, spec=spec
    )
    kin = TorchRM75Kinematics.from_locked_model(lm, device=cfg.device)

    Q_np = _sample_collision_free_q(cfg, lm, collision_filter, rng)
    q_fk = torch.as_tensor(Q_np, dtype=torch.float32, device=kin.device)
    with torch.no_grad():
        p_fk, R_fk = kin.fk(q_fk)
        f_fk = _features(p_fk, R_fk).cpu().numpy().astype(np.float32)
        quality_fk = _quality(kin, q_fk).cpu().numpy().astype(np.float32)

    out_f: list[np.ndarray] = [f_fk]
    out_y: list[np.ndarray] = [np.ones(cfg.n_fk_positive, dtype=np.float32)]
    out_q: list[np.ndarray] = [quality_fk]
    out_qbest: list[np.ndarray] = [Q_np.astype(np.float32)]
    out_layer: list[np.ndarray] = [
        np.full(cfg.n_fk_positive, LAYER_INTERIOR, dtype=np.int32)
    ]
    out_source: list[np.ndarray] = [np.arange(cfg.n_fk_positive, dtype=np.int64)]
    n_reachable_query = 0
    source_schedule = np.resize(
        np.arange(cfg.n_fk_positive, dtype=np.int64), cfg.n_pose_queries
    )
    rng.shuffle(source_schedule)
    query_pos_delta = np.full((cfg.n_pose_queries, 3), np.nan, dtype=np.float32)
    query_rot_delta = np.full((cfg.n_pose_queries, 3), np.nan, dtype=np.float32)

    for batch_index, start in enumerate(range(0, cfg.n_pose_queries, cfg.batch_size)):
        stop = min(cfg.n_pose_queries, start + cfg.batch_size)
        n = stop - start
        source_idx = source_schedule[start:stop]
        source_q = q_fk[source_idx]
        source_p = p_fk[source_idx]
        source_R = R_fk[source_idx]
        near = rng.random(n) < cfg.near_query_fraction
        pos_mag = np.where(
            near,
            rng.uniform(0.0, cfg.near_pos_max_m, size=n),
            rng.uniform(cfg.far_pos_min_m, cfg.far_pos_max_m, size=n),
        )
        rot_mag = np.where(
            near,
            rng.uniform(0.0, cfg.near_rot_max_deg, size=n),
            rng.uniform(cfg.far_rot_min_deg, cfg.far_rot_max_deg, size=n),
        )
        mode = rng.random(n)
        pos_mag[mode > 0.70] = 0.0
        rot_mag[mode < 0.35] = 0.0
        dp = _random_unit(rng, n) * pos_mag[:, None]
        dw = _random_unit(rng, n) * np.deg2rad(rot_mag)[:, None]
        query_pos_delta[start:stop] = dp.astype(np.float32)
        query_rot_delta[start:stop] = dw.astype(np.float32)
        target_p = source_p + torch.as_tensor(dp, dtype=torch.float32, device=kin.device)
        dR = so3_exp(torch.as_tensor(dw, dtype=torch.float32, device=kin.device))
        target_R = dR @ source_R

        q0 = torch.empty(n, cfg.n_ik_seeds, 7, dtype=torch.float32, device=kin.device)
        q0[:, 0] = source_q
        seed_idx = rng.integers(
            0, cfg.n_fk_positive, size=(n, cfg.n_ik_seeds - 1)
        )
        q0[:, 1:] = q_fk[torch.as_tensor(seed_idx, device=kin.device)]
        result = kin.ik_dls(
            target_p,
            target_R,
            q0,
            max_iter=cfg.ik_max_iter,
            damping=cfg.ik_damping,
            step_size=cfg.ik_step_size,
            max_step_rad=cfg.ik_max_step_rad,
            tol_pos_m=cfg.ik_tol_pos_m,
            tol_rot_rad=cfg.ik_tol_rot_rad,
        )
        checked = select_collision_free_ik(
            result,
            collision_filter,
            tol_pos_m=cfg.ik_tol_pos_m,
            tol_rot_rad=cfg.ik_tol_rot_rad,
        )
        reachable = checked.reachable
        n_reachable_query += int(reachable.sum().item())
        q_best = torch.zeros(n, 7, dtype=torch.float32, device=kin.device)
        q_best[reachable] = checked.q[reachable]
        q_value = torch.zeros(n, dtype=torch.float32, device=kin.device)
        if bool(reachable.any()):
            q_value[reachable] = _quality(kin, checked.q[reachable])
        out_f.append(_features(target_p, target_R).cpu().numpy().astype(np.float32))
        y_np = reachable.cpu().numpy().astype(np.float32)
        out_y.append(y_np)
        out_q.append(q_value.cpu().numpy().astype(np.float32))
        out_qbest.append(q_best.cpu().numpy().astype(np.float32))
        out_layer.append(
            np.where(y_np > 0.5, LAYER_INTERIOR, LAYER_EXTERIOR).astype(np.int32)
        )
        out_source.append(source_idx.astype(np.int64))
        if cfg.log_every_batches > 0 and (batch_index + 1) % cfg.log_every_batches == 0:
            done = stop
            print(
                f"[gpu-pose-gt] queries={done}/{cfg.n_pose_queries} "
                f"reachable={n_reachable_query} unreachable={done - n_reachable_query}",
                flush=True,
            )

    features = np.concatenate(out_f, axis=0)
    y = np.concatenate(out_y)
    quality = np.concatenate(out_q)
    qbest = np.concatenate(out_qbest)
    layer = np.concatenate(out_layer)
    source_id = np.concatenate(out_source)
    pose_delta_translation = np.concatenate(
        [np.full((cfg.n_fk_positive, 3), np.nan, dtype=np.float32), query_pos_delta],
        axis=0,
    )
    pose_delta_rotvec = np.concatenate(
        [np.full((cfg.n_fk_positive, 3), np.nan, dtype=np.float32), query_rot_delta],
        axis=0,
    )
    n_total = len(y)
    m_gt = np.where(y > 0.5, cfg.m_eps, -cfg.m_eps).astype(np.float32)
    scale = np.maximum(np.abs(features[:, :3]).max(axis=0) * 1.05, 0.1).astype(np.float32)
    arrays = {
        "features": features,
        "reachable": y,
        "p_reach": y,
        "y_soft": y,
        "cls_weight": np.ones(n_total, dtype=np.float32),
        "m_gt": m_gt,
        "margin_weight": np.zeros(n_total, dtype=np.float32),
        "layer_id": layer,
        "voxel_id": np.arange(n_total, dtype=np.int32),
        "orient_id": np.zeros(n_total, dtype=np.int32),
        "block_id": block_ids(features, cfg.holdout_block_m),
        "source_pose_id": source_id,
        "pose_delta_translation_m": pose_delta_translation,
        "pose_delta_rotvec_rad": pose_delta_rotvec,
        "q": quality,
        "q_comfort": quality,
        "q_capability": quality,
        "q_manip": quality,
        "q_joint": quality,
        "q_selfcol": y.copy(),
        "q_nullspace": quality,
        "q_best": qbest,
        "q_candidates": qbest[:, None, :],
        "d": y * quality,
        "aabb_lo": -scale,
        "aabb_hi": scale,
        "sigma_p_m": np.asarray([cfg.near_pos_max_m], dtype=np.float32),
        "sigma_r_deg": np.asarray([cfg.near_rot_max_deg], dtype=np.float32),
        "feature_dim": np.asarray([9], dtype=np.int32),
        "feature_kind": np.asarray([3], dtype=np.int32),
        "label_kind": np.asarray([6], dtype=np.int32),
    }
    meta = {
        "label_kind": "gpu_full_pose_collision_checked_v1",
        "feature_kind": "se3_rot6d9",
        "n_total": n_total,
        "n_fk_positive": cfg.n_fk_positive,
        "n_pose_queries": cfg.n_pose_queries,
        "n_query_reachable": n_reachable_query,
        "n_query_unreachable": cfg.n_pose_queries - n_reachable_query,
        "tcp_frame": lm.tcp_frame,
        "urdf_path": str(lm.urdf_path),
        "rail_locked_at_m": lm.rail_locked_at_m,
        "collision_urdf": str(collision_urdf),
        "collision_pairs": str(collision_pairs),
        "robot_contract": spec.to_manifest(),
        "collision_contract": (
            "FK positives filtered; IK positive iff at least one converged "
            "candidate is collision-free under the robot+probe model"
        ),
        "negative_contract": (
            f"all {cfg.n_ik_seeds} GPU DLS candidates failed tolerance or collided"
        ),
        "config": asdict(cfg),
    }
    return arrays, meta
