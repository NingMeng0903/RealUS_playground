"""Build collision-checked local SE(3) boundary stencils from GPU pose GT."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from ird_playground.ird.gt_common import (
    CLEARANCE_POSITION,
    CLEARANCE_ROTATION,
    block_ids,
    reachability_modules,
)
from ird_playground.ird.robot_model import (
    assert_robot_contract_compatible,
    load_robot_model_spec,
)
from ird_playground.ird.export_gt import (
    LAYER_BND_NEG,
    LAYER_BND_POS,
    LAYER_JITTER_NEG,
    LAYER_JITTER_POS,
    load_ird_gt,
)
from ird_playground.ird.gpu_pose_gt import (
    GpuPoseGtConfig,
    _features,
    _probe_collision_filter,
    _quality,
)
from ird_playground.ird.torch_kinematics import (
    TorchRM75Kinematics,
    select_collision_free_ik,
    so3_exp,
    so3_log,
)


@dataclass(frozen=True)
class GpuBoundaryStencilConfig:
    base_gt_npz: str = "data/ird/gpu_pose_production.npz"
    n_position_boundaries: int = 15_000
    n_rotation_boundaries: int = 15_000
    position_offsets_mm: tuple[float, ...] = (1.0, 3.0, 6.0, 10.0)
    rotation_offsets_deg: tuple[float, ...] = (1.0, 3.0, 5.0)
    margin_sigma_p_m: float = 0.006
    margin_sigma_rot_deg: float = 3.0
    n_ik_seeds: int = 12
    batch_size: int = 512
    bisection_iterations: int = 12
    ik_max_iter: int = 100
    ik_tol_pos_m: float = 2.0e-4
    ik_tol_rot_rad: float = 1.0e-3
    collision_urdf: str | None = None
    collision_pairs: str | None = None
    collision_security_margin_m: float = 0.0
    robot_spec: str | None = None
    seed: int = 43
    device: str = "cuda"
    log_every_batches: int = 5

    def validate(self) -> None:
        if self.n_position_boundaries <= 0 or self.n_rotation_boundaries <= 0:
            raise ValueError("boundary counts must be positive")
        if self.n_ik_seeds < 2 or self.batch_size <= 0:
            raise ValueError("invalid seed or batch count")
        if self.bisection_iterations < 2:
            raise ValueError("bisection_iterations must be at least 2")


def _rotation_from_features(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features[:, 3:6], dtype=np.float64)
    x /= np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    y = np.asarray(features[:, 6:9], dtype=np.float64)
    y = y - x * np.sum(x * y, axis=1, keepdims=True)
    y /= np.clip(np.linalg.norm(y, axis=1, keepdims=True), 1e-12, None)
    return np.stack([x, y, np.cross(x, y)], axis=2)


def _solve(
    kin,
    collision_filter,
    target_p,
    target_R,
    q_hint,
    q_extra,
    cfg,
):
    import torch

    q0 = torch.cat([q_hint[:, None, :], q_extra], dim=1)
    result = kin.ik_dls(
        target_p,
        target_R,
        q0,
        max_iter=cfg.ik_max_iter,
        tol_pos_m=cfg.ik_tol_pos_m,
        tol_rot_rad=cfg.ik_tol_rot_rad,
    )
    return select_collision_free_ik(
        result,
        collision_filter,
        tol_pos_m=cfg.ik_tol_pos_m,
        tol_rot_rad=cfg.ik_tol_rot_rad,
    )


def _select_pairs(base: dict[str, np.ndarray], cfg, rng):
    import torch

    negative = np.flatnonzero(
        (base["reachable"] < 0.5) & (base["source_pose_id"] >= 0)
    )
    target = base["features"][negative]
    source = base["features"][base["source_pose_id"][negative]]
    dp = np.linalg.norm(target[:, :3] - source[:, :3], axis=1)
    R_t = torch.as_tensor(_rotation_from_features(target), dtype=torch.float64)
    R_s = torch.as_tensor(_rotation_from_features(source), dtype=torch.float64)
    angle = torch.linalg.vector_norm(so3_log(R_t @ R_s.transpose(-1, -2)), dim=-1).numpy()
    pos = negative[angle < 1.0e-6]
    rot = negative[dp < 1.0e-7]
    if len(pos) < cfg.n_position_boundaries or len(rot) < cfg.n_rotation_boundaries:
        raise RuntimeError(
            f"insufficient pure pairs: position={len(pos)} rotation={len(rot)}"
        )
    rng.shuffle(pos)
    rng.shuffle(rot)
    return pos[: cfg.n_position_boundaries], rot[: cfg.n_rotation_boundaries]


def _empty_stencil_fields() -> dict[str, list[np.ndarray]]:
    return {k: [] for k in (
        "features", "reachable", "m_gt", "margin_weight", "layer_id", "q",
        "q_best", "source_pose_id", "boundary_id", "signed_m", "signed_deg", "kind",
    )}


def _append_stencil_batch(fields, features, reachable, q_value, q_best, source_id,
                          boundary_id, signed, kind, consistent):
    y = reachable.astype(np.float32)
    if kind == CLEARANCE_POSITION:
        signed_m = signed * 1.0e-3
        margin = signed_m / 0.006
        signed_deg = np.full_like(signed, np.nan)
    else:
        margin = signed / 3.0
        signed_m = np.full_like(signed, np.nan)
        signed_deg = signed
    layer = np.where(
        y > 0.5,
        np.where(np.abs(signed) <= 1.0 + 1e-7, LAYER_BND_POS, LAYER_JITTER_POS),
        np.where(np.abs(signed) <= 1.0 + 1e-7, LAYER_BND_NEG, LAYER_JITTER_NEG),
    ).astype(np.int32)
    fields["features"].append(features.astype(np.float32))
    fields["reachable"].append(y)
    fields["m_gt"].append(margin.astype(np.float32))
    fields["margin_weight"].append(consistent.astype(np.float32))
    fields["layer_id"].append(layer)
    fields["q"].append(q_value.astype(np.float32))
    fields["q_best"].append(q_best.astype(np.float32))
    fields["source_pose_id"].append(source_id.astype(np.int64))
    fields["boundary_id"].append(boundary_id.astype(np.int64))
    fields["signed_m"].append(signed_m.astype(np.float32))
    fields["signed_deg"].append(signed_deg.astype(np.float32))
    fields["kind"].append(np.full(len(y), kind, dtype=np.int8))


def build_gpu_boundary_stencils(
    cfg: GpuBoundaryStencilConfig | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    import torch

    cfg = cfg or GpuBoundaryStencilConfig()
    cfg.validate()
    base_path = Path(cfg.base_gt_npz)
    if not base_path.is_absolute():
        base_path = Path(__file__).resolve().parents[2] / base_path
    base = load_ird_gt(base_path)
    spec = load_robot_model_spec(cfg.robot_spec)
    base_manifest_path = base_path.with_suffix(".yaml")
    base_manifest = (
        yaml.safe_load(base_manifest_path.read_text(encoding="utf-8")) or {}
        if base_manifest_path.is_file()
        else {}
    )
    assert_robot_contract_compatible(base_manifest.get("robot_contract"), spec)
    rng = np.random.default_rng(cfg.seed)
    (
        _ik, _ikm, _SeedPoolConfig, _seed_pool, _halton,
        SelfCollisionFilter, build_locked_rail_model,
    ) = reachability_modules()
    lm = build_locked_rail_model(
        spec.kinematics_urdf,
        rail_locked_at_m=spec.rail_locked_at_m,
        tcp_frame=spec.tcp_frame,
    )
    collision_cfg = GpuPoseGtConfig(
        collision_urdf=cfg.collision_urdf,
        collision_pairs=cfg.collision_pairs,
        collision_security_margin_m=cfg.collision_security_margin_m,
        robot_spec=cfg.robot_spec,
    )
    collision_filter, collision_urdf, collision_pairs = _probe_collision_filter(
        collision_cfg, lm, SelfCollisionFilter, spec=spec
    )
    kin = TorchRM75Kinematics.from_locked_model(lm, device=cfg.device)
    q_pool_np = base["q_best"][base["reachable"] > 0.5]
    q_pool = torch.as_tensor(q_pool_np, dtype=torch.float32, device=kin.device)
    pos_pairs, rot_pairs = _select_pairs(base, cfg, rng)
    fields = _empty_stencil_fields()
    accepted = {CLEARANCE_POSITION: 0, CLEARANCE_ROTATION: 0}
    next_boundary_id = 0

    for kind, pair_ids, offsets in (
        (CLEARANCE_POSITION, pos_pairs, cfg.position_offsets_mm),
        (CLEARANCE_ROTATION, rot_pairs, cfg.rotation_offsets_deg),
    ):
        for batch_index, start in enumerate(range(0, len(pair_ids), cfg.batch_size)):
            ids = pair_ids[start : start + cfg.batch_size]
            n = len(ids)
            source_ids = base["source_pose_id"][ids]
            source_f = base["features"][source_ids]
            target_f = base["features"][ids]
            source_p = torch.as_tensor(source_f[:, :3], dtype=torch.float32, device=kin.device)
            target_p = torch.as_tensor(target_f[:, :3], dtype=torch.float32, device=kin.device)
            source_R = torch.as_tensor(
                _rotation_from_features(source_f), dtype=torch.float32, device=kin.device
            )
            target_R = torch.as_tensor(
                _rotation_from_features(target_f), dtype=torch.float32, device=kin.device
            )
            delta_p = target_p - source_p
            delta_w = so3_log(target_R @ source_R.transpose(-1, -2))
            q_lo = torch.as_tensor(
                base["q_best"][source_ids], dtype=torch.float32, device=kin.device
            )
            extra_idx = rng.integers(0, len(q_pool), size=(n, cfg.n_ik_seeds - 1))
            q_extra = q_pool[torch.as_tensor(extra_idx, device=kin.device)]
            lo = torch.zeros(n, dtype=torch.float32, device=kin.device)
            hi = torch.ones(n, dtype=torch.float32, device=kin.device)
            for _ in range(cfg.bisection_iterations):
                mid = 0.5 * (lo + hi)
                p_mid = source_p + mid[:, None] * delta_p
                R_mid = so3_exp(mid[:, None] * delta_w) @ source_R
                checked = _solve(
                    kin, collision_filter, p_mid, R_mid, q_lo, q_extra, cfg
                )
                reach = checked.reachable
                lo = torch.where(reach, mid, lo)
                hi = torch.where(reach, hi, mid)
                q_lo[reach] = checked.q[reach]
            t_boundary = 0.5 * (lo + hi)
            p_boundary = source_p + t_boundary[:, None] * delta_p
            R_boundary = so3_exp(t_boundary[:, None] * delta_w) @ source_R
            if kind == CLEARANCE_POSITION:
                direction = delta_p / torch.linalg.vector_norm(
                    delta_p, dim=-1, keepdim=True
                ).clamp_min(1e-9)
            else:
                direction = delta_w / torch.linalg.vector_norm(
                    delta_w, dim=-1, keepdim=True
                ).clamp_min(1e-9)

            signed_values = np.asarray(
                list(reversed(offsets)) + [-x for x in offsets], dtype=np.float32
            )
            all_y, batch_rows = [], []
            for signed_value in signed_values:
                if kind == CLEARANCE_POSITION:
                    p_stencil = p_boundary - float(signed_value) * 1e-3 * direction
                    R_stencil = R_boundary
                else:
                    p_stencil = p_boundary
                    R_stencil = so3_exp(
                        -np.deg2rad(float(signed_value)) * direction
                    ) @ R_boundary
                checked = _solve(
                    kin, collision_filter, p_stencil, R_stencil, q_lo, q_extra, cfg
                )
                y = checked.reachable
                q_best = torch.zeros(n, 7, dtype=torch.float32, device=kin.device)
                q_value = torch.zeros(n, dtype=torch.float32, device=kin.device)
                q_best[y] = checked.q[y]
                if bool(y.any()):
                    q_value[y] = _quality(kin, checked.q[y])
                all_y.append(y.cpu().numpy())
                batch_rows.append((
                    _features(p_stencil, R_stencil).cpu().numpy(),
                    q_value.cpu().numpy(), q_best.cpu().numpy(),
                ))
            labels = np.stack(all_y, axis=1)
            closest_pos = int(np.flatnonzero(signed_values > 0)[-1])
            closest_neg = int(np.flatnonzero(signed_values < 0)[0])
            straddles = labels[:, closest_pos] & (~labels[:, closest_neg])
            expected = signed_values[None, :] > 0
            monotonic = np.all(labels == expected, axis=1)
            keep = straddles
            kept_ids = np.arange(next_boundary_id, next_boundary_id + int(keep.sum()))
            next_boundary_id += int(keep.sum())
            for col, signed_value in enumerate(signed_values):
                features, q_value, q_best = batch_rows[col]
                _append_stencil_batch(
                    fields,
                    features[keep],
                    labels[keep, col],
                    q_value[keep],
                    q_best[keep],
                    source_ids[keep],
                    kept_ids,
                    np.full(int(keep.sum()), signed_value, dtype=np.float32),
                    kind,
                    monotonic[keep],
                )
            accepted[kind] += int(keep.sum())
            if cfg.log_every_batches > 0 and (batch_index + 1) % cfg.log_every_batches == 0:
                print(
                    f"[gpu-stencil] kind={kind} pairs={min(start+n,len(pair_ids))}/{len(pair_ids)} "
                    f"accepted={accepted[kind]}", flush=True,
                )

    stencil = {k: np.concatenate(v, axis=0) for k, v in fields.items()}
    n_base = len(base["reachable"])
    n_stencil = len(stencil["reachable"])
    combined = dict(base)
    defaults = {
        "boundary_id": np.full(n_base, -1, dtype=np.int64),
        "boundary_signed_m": np.full(n_base, np.nan, dtype=np.float32),
        "boundary_signed_rot_deg": np.full(n_base, np.nan, dtype=np.float32),
        "clearance_kind": np.full(n_base, -1, dtype=np.int8),
    }
    for key, value in defaults.items():
        if key not in combined:
            combined[key] = value
    append_map = {
        "features": stencil["features"],
        "reachable": stencil["reachable"],
        "p_reach": stencil["reachable"],
        "y_soft": stencil["reachable"],
        "cls_weight": np.ones(n_stencil, dtype=np.float32),
        "m_gt": stencil["m_gt"],
        "margin_weight": stencil["margin_weight"],
        "layer_id": stencil["layer_id"],
        "voxel_id": np.arange(n_base, n_base + n_stencil, dtype=np.int32),
        "orient_id": np.zeros(n_stencil, dtype=np.int32),
        "block_id": block_ids(stencil["features"], 0.04),
        "source_pose_id": stencil["source_pose_id"],
        "q": stencil["q"],
        "q_comfort": stencil["q"],
        "q_capability": stencil["q"],
        "q_manip": stencil["q"],
        "q_joint": stencil["q"],
        "q_selfcol": stencil["reachable"],
        "q_nullspace": stencil["q"],
        "q_best": stencil["q_best"],
        "q_candidates": stencil["q_best"][:, None, :],
        "d": stencil["reachable"] * stencil["q"],
        "boundary_id": stencil["boundary_id"],
        "boundary_signed_m": stencil["signed_m"],
        "boundary_signed_rot_deg": stencil["signed_deg"],
        "clearance_kind": stencil["kind"],
    }
    for key, values in append_map.items():
        combined[key] = np.concatenate([combined[key], values], axis=0)
    combined["label_kind"] = np.asarray([7], dtype=np.int32)
    meta = {
        "label_kind": "gpu_pose_plus_collision_checked_boundary_stencils_v1",
        "base_gt_npz": str(base_path),
        "n_base": n_base,
        "n_stencil": n_stencil,
        "n_position_boundaries": accepted[CLEARANCE_POSITION],
        "n_rotation_boundaries": accepted[CLEARANCE_ROTATION],
        "n_monotonic_stencil_rows": int(stencil["margin_weight"].sum()),
        "collision_urdf": str(collision_urdf),
        "collision_pairs": str(collision_pairs),
        "collision_contract": "every positive stencil pose has a collision-free IK solution",
        "robot_contract": spec.to_manifest(),
        "config": asdict(cfg),
    }
    return combined, meta
