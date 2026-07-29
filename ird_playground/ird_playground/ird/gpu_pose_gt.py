"""Collision-aware GPU pose sampling for the flange-chart Neural IRD field.

IK / SRS targets stay at the TCP.  Stored ``features`` are flange SE(3)
(rot6d9) in ``rail_base``; ``flange_canonical`` is the 9-D yaw-invariant chart
in the J1-axis frame.  Reachability uses controller-aligned ``srs_ik`` point
labeling (fixed per-sample branch), plus arm-body collision filtering.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from ird_playground.ird.canonical import (
    FLANGE_CANONICAL_DIM,
    canonical_flange_from_se3_features_torch,
)
from ird_playground.ird.gt_common import block_ids, reachability_modules
from ird_playground.ird.metric import LAMBDA_M_PER_RAD, metric_manifest
from ird_playground.ird.robot_model import RobotModelSpec, load_robot_model_spec
from ird_playground.ird.export_gt import LAYER_EXTERIOR, LAYER_INTERIOR
from ird_playground.ird.srs_label import (
    SrsLabelConfig,
    branch_and_psi_from_q7,
    srs_reachable_batch,
)
from ird_playground.ird.tool_frame import pose_tcp_to_flange
from ird_playground.ird.torch_kinematics import TorchRM75Kinematics, so3_exp


# Branch for pure TCP queries with no generating seed (uniform / free queries).
DEFAULT_QUERY_BRANCH_ID = 0
FEATURE_KIND = "flange_se3_rot6d9"
LABEL_KIND = "gpu_full_pose_srs_collision_checked_v1"


@dataclass(frozen=True)
class GpuPoseGtConfig:
    n_fk_positive: int = 100_000
    n_pose_queries: int = 100_000
    n_ik_seeds: int = 12  # retained for YAML compat; unused (SRS has no seed pool)
    batch_size: int = 4096
    joint_limit_inset_fraction: float = 0.02
    near_query_fraction: float = 0.50
    near_pos_max_m: float = 0.04
    near_rot_max_deg: float = 12.0
    far_pos_min_m: float = 0.04
    far_pos_max_m: float = 0.35
    far_rot_min_deg: float = 10.0
    far_rot_max_deg: float = 70.0
    ik_max_iter: int = 100  # unused with SRS; kept for YAML compat
    ik_damping: float = 0.002
    ik_step_size: float = 0.8
    ik_max_step_rad: float = 0.25
    ik_tol_pos_m: float = 2.0e-4
    ik_tol_rot_rad: float = 1.0e-3
    collision_security_margin_m: float | None = None  # None → robot_contract default
    collision_urdf: str | None = None
    collision_pairs: str | None = None
    robot_spec: str | None = None
    holdout_block_m: float = 0.04
    m_eps: float = 1.0e-3
    default_branch_id: int = DEFAULT_QUERY_BRANCH_ID
    seed: int = 42
    device: str = "cuda"
    log_every_batches: int = 10

    def validate(self) -> None:
        if self.n_fk_positive <= 0 or self.n_pose_queries <= 0:
            raise ValueError("sample counts must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= self.joint_limit_inset_fraction < 0.5:
            raise ValueError("joint_limit_inset_fraction must lie in [0,0.5)")
        if not 0.0 <= self.near_query_fraction <= 1.0:
            raise ValueError("near_query_fraction must lie in [0,1]")
        if not (0 <= int(self.default_branch_id) <= 7):
            raise ValueError("default_branch_id must be in 0..7")


def _robot_spec_for_cfg(cfg: GpuPoseGtConfig) -> RobotModelSpec:
    spec = load_robot_model_spec(cfg.robot_spec)
    if cfg.collision_urdf:
        spec = replace(spec, collision_urdf=Path(cfg.collision_urdf).resolve())
    if cfg.collision_pairs:
        spec = replace(spec, collision_pairs=Path(cfg.collision_pairs).resolve())
    if cfg.collision_security_margin_m is not None:
        spec = replace(
            spec, collision_security_margin_m=float(cfg.collision_security_margin_m)
        )
    spec.validate()
    return spec


def _resolved_margin_m(cfg: GpuPoseGtConfig, spec: RobotModelSpec) -> float:
    if cfg.collision_security_margin_m is not None:
        return float(cfg.collision_security_margin_m)
    return float(spec.collision_security_margin_m)


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
    return (
        SelfCollisionFilter(
            kin_urdf=lm.urdf_path,
            collision_urdf=collision_urdf,
            pair_config=collision_pairs,
            rail_locked_at_m=lm.rail_locked_at_m,
            security_margin=_resolved_margin_m(cfg, spec),
        ),
        collision_urdf,
        collision_pairs,
    )


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
    return (sigma / 0.08).clamp(0.0, 1.0) * (joint / 0.08).clamp(0.0, 1.0)


def _se3_features(p, R):
    """Pack ``[p, R[:,0], R[:,1]]`` (rot6d9)."""
    import torch

    return torch.cat([p, R[..., :, 0], R[..., :, 1]], dim=-1)


def _flange_se3_features(p_tcp, R_tcp, T_flange_tcp):
    """Flange SE(3) features from a TCP pose (IK / SRS target stays TCP)."""
    p_fl, R_fl = pose_tcp_to_flange(p_tcp, R_tcp, T_flange_tcp)
    return _se3_features(p_fl, R_fl)


def _flange_canonical_np(features_flange_se3, T_flange_tcp_eye, T_root_axis):
    """9-D chart from flange se3 features already expressed in rail_base.

    ``T_flange_tcp`` is identity here because ``features`` are already flange;
    ``T_root_axis`` maps rail_base → J1-axis frame before the chart.
    """
    import torch

    with torch.no_grad():
        t = torch.as_tensor(features_flange_se3, dtype=torch.float32)
        tool = torch.as_tensor(T_flange_tcp_eye, dtype=torch.float32)
        axis = torch.as_tensor(T_root_axis, dtype=torch.float32)
        return canonical_flange_from_se3_features_torch(t, tool, axis).numpy().astype(
            np.float32
        )


def _srs_label_tcp(
    p_tcp: np.ndarray,
    R_tcp: np.ndarray,
    *,
    srs_cfg: SrsLabelConfig,
    branch_ids: np.ndarray,
    psi_homes: np.ndarray,
    collision_filter,
) -> dict[str, np.ndarray]:
    """Point-reach SRS labels with arm-body collision gate."""
    labeled = srs_reachable_batch(
        p_tcp,
        R_tcp,
        srs_cfg,
        branch_ids=branch_ids,
        psi_homes=psi_homes,
    )
    reachable = np.asarray(labeled["reachable"], dtype=bool).copy()
    q_best = np.asarray(labeled["q_best"], dtype=np.float64).copy()
    psi = np.asarray(labeled["psi"], dtype=np.float64).copy()
    branch = np.asarray(labeled["branch"], dtype=np.int32).copy()
    if reachable.any():
        free = collision_filter.free_mask(q_best[reachable])
        drop = np.flatnonzero(reachable)[~free]
        reachable[drop] = False
        q_best[drop] = np.nan
        psi[drop] = np.nan
        branch[drop] = -1
    return {
        "reachable": reachable,
        "q_best": q_best.astype(np.float32),
        "psi": psi.astype(np.float32),
        "branch": branch,
    }


def _make_srs_cfg(spec: RobotModelSpec, default_branch_id: int) -> SrsLabelConfig:
    from ird_playground.ird.srs_label import _srs_api

    y_rail = float(_srs_api().shoulder_y_from_q_rail(spec.rail_locked_at_m))
    return SrsLabelConfig(branch_id=int(default_branch_id) & 0b111, y_rail_m=y_rail)


def build_gpu_pose_gt(
    cfg: GpuPoseGtConfig | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Generate flange-feature pose labels with SRS point reachability + collision."""
    import torch

    cfg = cfg or GpuPoseGtConfig()
    cfg.validate()
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    (
        _ik_dls,
        _ik_multi,
        _SeedPoolConfig,
        _build_seed_pool,
        _halton,
        SelfCollisionFilter,
        build_locked_rail_model,
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
    tool = spec.tool_frame()
    T_flange_tcp = torch.as_tensor(
        tool.T_flange_tcp, dtype=torch.float32, device=kin.device
    )
    T_root_axis = spec.root_to_j1_axis()
    # Features are already flange: identity tool when mapping se3 → chart.
    T_eye = np.eye(4, dtype=np.float64)
    srs_cfg = _make_srs_cfg(spec, cfg.default_branch_id)
    margin_m = _resolved_margin_m(cfg, spec)

    Q_np = _sample_collision_free_q(cfg, lm, collision_filter, rng)
    q_fk = torch.as_tensor(Q_np, dtype=torch.float32, device=kin.device)
    with torch.no_grad():
        p_fk, R_fk = kin.fk(q_fk)
        f_fk = _flange_se3_features(p_fk, R_fk, T_flange_tcp).cpu().numpy().astype(
            np.float32
        )
        quality_fk = _quality(kin, q_fk).cpu().numpy().astype(np.float32)

    branch_fk = np.empty(cfg.n_fk_positive, dtype=np.int32)
    psi_fk = np.empty(cfg.n_fk_positive, dtype=np.float32)
    for i, q in enumerate(Q_np):
        b, psi = branch_and_psi_from_q7(q)
        branch_fk[i] = b
        psi_fk[i] = psi

    out_f: list[np.ndarray] = [f_fk]
    out_y: list[np.ndarray] = [np.ones(cfg.n_fk_positive, dtype=np.float32)]
    out_q: list[np.ndarray] = [quality_fk]
    out_qbest: list[np.ndarray] = [Q_np.astype(np.float32)]
    out_layer: list[np.ndarray] = [
        np.full(cfg.n_fk_positive, LAYER_INTERIOR, dtype=np.int32)
    ]
    out_source: list[np.ndarray] = [np.arange(cfg.n_fk_positive, dtype=np.int64)]
    out_branch: list[np.ndarray] = [branch_fk]
    out_psi: list[np.ndarray] = [psi_fk]
    out_psi_home: list[np.ndarray] = [psi_fk.copy()]
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

        # Branch / ψ home locked to the generating FK seed (runtime semantics).
        branch_ids = branch_fk[source_idx]
        psi_homes = psi_fk[source_idx]
        labeled = _srs_label_tcp(
            target_p.detach().cpu().numpy(),
            target_R.detach().cpu().numpy(),
            srs_cfg=srs_cfg,
            branch_ids=branch_ids,
            psi_homes=psi_homes,
            collision_filter=collision_filter,
        )
        reachable = labeled["reachable"]
        n_reachable_query += int(reachable.sum())
        q_best = labeled["q_best"]
        q_value = np.zeros(n, dtype=np.float32)
        if reachable.any():
            q_t = torch.as_tensor(
                q_best[reachable], dtype=torch.float32, device=kin.device
            )
            q_value[reachable] = _quality(kin, q_t).cpu().numpy().astype(np.float32)

        out_f.append(
            _flange_se3_features(target_p, target_R, T_flange_tcp)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        y_np = reachable.astype(np.float32)
        out_y.append(y_np)
        out_q.append(q_value)
        out_qbest.append(q_best)
        out_layer.append(
            np.where(y_np > 0.5, LAYER_INTERIOR, LAYER_EXTERIOR).astype(np.int32)
        )
        out_source.append(source_idx.astype(np.int64))
        out_branch.append(labeled["branch"])
        out_psi.append(labeled["psi"])
        out_psi_home.append(psi_homes.astype(np.float32))
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
    branch_ids_all = np.concatenate(out_branch)
    psi_all = np.concatenate(out_psi)
    psi_homes_all = np.concatenate(out_psi_home)
    flange_canonical = _flange_canonical_np(features, T_eye, T_root_axis)
    if flange_canonical.shape[-1] != FLANGE_CANONICAL_DIM:
        raise RuntimeError(
            f"expected flange_canonical dim {FLANGE_CANONICAL_DIM}, "
            f"got {flange_canonical.shape[-1]}"
        )
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
        "flange_canonical": flange_canonical,
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
        "branch_ids": branch_ids_all,
        "psi": psi_all,
        "psi_homes": psi_homes_all,
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
        "FLANGE_CANONICAL_DIM": np.asarray([FLANGE_CANONICAL_DIM], dtype=np.int32),
    }
    meta = {
        "label_kind": LABEL_KIND,
        "feature_kind": FEATURE_KIND,
        "feature_frame": "flange",
        "ik_target_frame": "tcp",
        "FLANGE_CANONICAL_DIM": FLANGE_CANONICAL_DIM,
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
        "collision_security_margin_m": margin_m,
        "T_flange_tcp": tool.T_flange_tcp.tolist(),
        "metric": metric_manifest(lambda_m_per_rad=spec.metric_lambda_m_per_rad),
        "robot_contract": spec.to_manifest(),
        "srs_conditioning": {
            **srs_cfg.to_manifest(),
            "branch_id_policy": (
                "per_sample branch_from_q(generating_fk_seed) for FK positives "
                "and seed-conditioned pose queries; "
                f"default_branch_id={cfg.default_branch_id} for pure TCP queries"
            ),
            "psi_home_policy": (
                "psi_from_q(generating_fk_seed) for seed-conditioned samples; "
                "SrsLabelConfig.psi_home_rad for pure TCP queries"
            ),
            "default_branch_id": int(cfg.default_branch_id),
            "y_rail_m": srs_cfg.resolved_y_rail_m(),
            "point_only_reachability": True,
        },
        "collision_contract": (
            "FK positives filtered; query positive iff SRS point-reachable on the "
            "locked seed branch and the returned arm configuration is collision-free "
            f"under the arm-body model (security_margin={margin_m} m)"
        ),
        "negative_contract": (
            "SRS returned None on the locked branch (within ψ swing / hard bounds) "
            "or every surviving solution collided"
        ),
        "config": asdict(cfg),
    }
    return arrays, meta


__all__ = [
    "DEFAULT_QUERY_BRANCH_ID",
    "FEATURE_KIND",
    "LABEL_KIND",
    "GpuPoseGtConfig",
    "_flange_canonical_np",
    "_flange_se3_features",
    "_make_srs_cfg",
    "_probe_collision_filter",
    "_quality",
    "_se3_features",
    "_srs_label_tcp",
    "build_gpu_pose_gt",
]
