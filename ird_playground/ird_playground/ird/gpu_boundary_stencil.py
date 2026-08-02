"""Build collision-checked SE(3) boundary stencils from flange pose GT.

Bisection and stencil offsets use the declared SE(3) metric from
``ird.metric`` (``λ`` so 1 cm ≡ 1 deg).  Stored ``m_gt`` / signed distances
are in **metres** of that metric — never a dimensionless mix of mm and deg.
Reachability checks use the same SRS + collision contract as ``gpu_pose_gt``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from ird_playground.ird.canonical import FLANGE_CANONICAL_DIM
from ird_playground.ird.gt_common import (
    CLEARANCE_POSITION,
    CLEARANCE_ROTATION,
    block_ids,
    reachability_modules,
)
from ird_playground.ird.metric import LAMBDA_M_PER_RAD, metric_manifest, se3_distance_m_torch
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
    DEFAULT_QUERY_BRANCH_ID,
    GpuPoseGtConfig,
    _flange_se3_features,
    _make_srs_cfg,
    _probe_collision_filter,
    _quality,
    _srs_label_tcp,
)
from ird_playground.ird.tool_frame import pose_flange_to_tcp
from ird_playground.ird.torch_kinematics import (
    TorchRM75Kinematics,
    so3_exp,
    so3_log,
)


@dataclass(frozen=True)
class GpuBoundaryStencilConfig:
    base_gt_npz: str = "data/ird/gpu_pose_production.npz"
    n_position_boundaries: int = 15_000
    n_rotation_boundaries: int = 15_000
    # Declared-metric offsets (metres).  1 cm ≡ 1 deg under λ.
    metric_offsets_m: tuple[float, ...] = (0.001, 0.003, 0.006, 0.010)
    # Legacy fields kept for YAML compat; ignored when metric_offsets_m is set.
    position_offsets_mm: tuple[float, ...] | None = None
    rotation_offsets_deg: tuple[float, ...] | None = None
    margin_sigma_p_m: float = 0.006  # unused; retained for YAML compat
    margin_sigma_rot_deg: float = 3.0  # unused; retained for YAML compat
    n_ik_seeds: int = 12  # unused with SRS
    batch_size: int = 512
    bisection_iterations: int = 12
    ik_max_iter: int = 100
    ik_tol_pos_m: float = 2.0e-4
    ik_tol_rot_rad: float = 1.0e-3
    collision_urdf: str | None = None
    collision_pairs: str | None = None
    collision_security_margin_m: float | None = None
    robot_spec: str | None = None
    default_branch_id: int = DEFAULT_QUERY_BRANCH_ID
    seed: int = 43
    device: str = "cuda"
    log_every_batches: int = 5

    def validate(self) -> None:
        if self.n_position_boundaries <= 0 or self.n_rotation_boundaries <= 0:
            raise ValueError("boundary counts must be positive")
        if self.batch_size <= 0:
            raise ValueError("invalid batch count")
        if self.bisection_iterations < 2:
            raise ValueError("bisection_iterations must be at least 2")
        if len(self.resolved_metric_offsets_m()) < 1:
            raise ValueError("metric_offsets_m must be non-empty")

    def resolved_metric_offsets_m(self) -> tuple[float, ...]:
        if self.metric_offsets_m:
            return tuple(float(x) for x in self.metric_offsets_m)
        # Fallback: convert legacy separate mm/deg lists into metric metres.
        offs: list[float] = []
        if self.position_offsets_mm:
            offs.extend(float(x) * 1.0e-3 for x in self.position_offsets_mm)
        if self.rotation_offsets_deg:
            offs.extend(
                float(np.deg2rad(x)) * LAMBDA_M_PER_RAD
                for x in self.rotation_offsets_deg
            )
        return tuple(sorted(set(offs)))


def _rotation_from_features(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features[:, 3:6], dtype=np.float64)
    x /= np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    y = np.asarray(features[:, 6:9], dtype=np.float64)
    y = y - x * np.sum(x * y, axis=1, keepdims=True)
    y /= np.clip(np.linalg.norm(y, axis=1, keepdims=True), 1e-12, None)
    return np.stack([x, y, np.cross(x, y)], axis=2)


def _tcp_from_flange_features(features: np.ndarray, T_flange_tcp: np.ndarray):
    """Recover TCP ``(p, R)`` from flange se3_rot6d9 features."""
    import torch

    p_fl = torch.as_tensor(features[:, :3], dtype=torch.float32)
    R_fl = torch.as_tensor(_rotation_from_features(features), dtype=torch.float32)
    tool = torch.as_tensor(T_flange_tcp, dtype=torch.float32)
    return pose_flange_to_tcp(p_fl, R_fl, tool)


def _select_pairs(base: dict[str, np.ndarray], cfg, rng):
    """Select pure-position / pure-rotation unreachable queries via TCP deltas.

    Pair purity is defined on the TCP IK target (where queries were generated).
    Flange SE(3) features move under pure TCP rotation because of the tool
    offset, so do not use flange ``features`` norms for this gate.
    """
    negative = np.flatnonzero(
        (base["reachable"] < 0.5) & (base["source_pose_id"] >= 0)
    )
    if "pose_delta_translation_m" in base and "pose_delta_rotvec_rad" in base:
        dp = np.linalg.norm(
            np.asarray(base["pose_delta_translation_m"][negative], dtype=np.float64),
            axis=1,
        )
        angle = np.linalg.norm(
            np.asarray(base["pose_delta_rotvec_rad"][negative], dtype=np.float64),
            axis=1,
        )
    else:
        # Legacy fallback: approximate from flange features (imperfect).
        target = base["features"][negative]
        source = base["features"][base["source_pose_id"][negative]]
        dp = np.linalg.norm(target[:, :3] - source[:, :3], axis=1)
        import torch

        R_t = torch.as_tensor(_rotation_from_features(target), dtype=torch.float64)
        R_s = torch.as_tensor(_rotation_from_features(source), dtype=torch.float64)
        angle = torch.linalg.vector_norm(
            so3_log(R_t @ R_s.transpose(-1, -2)), dim=-1
        ).numpy()
    pos = negative[angle < 1.0e-9]
    rot = negative[dp < 1.0e-9]
    if len(pos) < cfg.n_position_boundaries or len(rot) < cfg.n_rotation_boundaries:
        raise RuntimeError(
            f"insufficient pure pairs: position={len(pos)} rotation={len(rot)}"
        )
    rng.shuffle(pos)
    rng.shuffle(rot)
    return pos[: cfg.n_position_boundaries], rot[: cfg.n_rotation_boundaries]


def _empty_stencil_fields() -> dict[str, list[np.ndarray]]:
    return {
        k: []
        for k in (
            "features",
            "reachable",
            "m_gt",
            "margin_weight",
            "layer_id",
            "q",
            "q_best",
            "source_pose_id",
            "boundary_id",
            "signed_m",
            "signed_deg",
            "kind",
            "branch_ids",
            "psi",
            "psi_homes",
            "boundary_features",
        )
    }


def _append_stencil_batch(
    fields,
    features,
    reachable,
    q_value,
    q_best,
    source_id,
    boundary_id,
    signed_m,
    kind,
    consistent,
    branch_ids,
    psi,
    psi_homes,
    boundary_features,
    *,
    near_band_m: float,
):
    """Append one offset column. ``signed_m`` is declared-metric metres."""
    y = reachable.astype(np.float32)
    signed = np.asarray(signed_m, dtype=np.float32)
    # Layer: near-boundary band vs jitter farther out (band = smallest offset).
    layer = np.where(
        y > 0.5,
        np.where(np.abs(signed) <= near_band_m + 1e-12, LAYER_BND_POS, LAYER_JITTER_POS),
        np.where(np.abs(signed) <= near_band_m + 1e-12, LAYER_BND_NEG, LAYER_JITTER_NEG),
    ).astype(np.int32)
    fields["features"].append(features.astype(np.float32))
    fields["reachable"].append(y)
    fields["m_gt"].append(signed.astype(np.float32))
    fields["margin_weight"].append(consistent.astype(np.float32))
    fields["layer_id"].append(layer)
    fields["q"].append(q_value.astype(np.float32))
    fields["q_best"].append(q_best.astype(np.float32))
    fields["source_pose_id"].append(source_id.astype(np.int64))
    fields["boundary_id"].append(boundary_id.astype(np.int64))
    fields["signed_m"].append(signed.astype(np.float32))
    # Rotation-equivalent view for diagnostics only (not used as m_gt).
    signed_deg = np.rad2deg(signed / LAMBDA_M_PER_RAD).astype(np.float32)
    fields["signed_deg"].append(signed_deg)
    fields["kind"].append(np.full(len(y), kind, dtype=np.int8))
    fields["branch_ids"].append(branch_ids.astype(np.int32))
    fields["psi"].append(psi.astype(np.float32))
    fields["psi_homes"].append(psi_homes.astype(np.float32))
    fields["boundary_features"].append(boundary_features.astype(np.float32))


def _label_tcp_batch(
    p_tcp,
    R_tcp,
    *,
    srs_cfg,
    branch_ids,
    psi_homes,
    collision_filter,
    kin,
):
    import torch

    labeled = _srs_label_tcp(
        p_tcp.detach().cpu().numpy(),
        R_tcp.detach().cpu().numpy(),
        srs_cfg=srs_cfg,
        branch_ids=branch_ids,
        psi_homes=psi_homes,
        collision_filter=collision_filter,
    )
    reachable = labeled["reachable"]
    q_best = labeled["q_best"]
    q_value = np.zeros(len(reachable), dtype=np.float32)
    if reachable.any():
        q_t = torch.as_tensor(q_best[reachable], dtype=torch.float32, device=kin.device)
        q_value[reachable] = _quality(kin, q_t).cpu().numpy().astype(np.float32)
    return labeled, q_value


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
    if cfg.collision_security_margin_m is not None:
        from dataclasses import replace

        spec = replace(
            spec, collision_security_margin_m=float(cfg.collision_security_margin_m)
        )
    base_manifest_path = base_path.with_suffix(".yaml")
    base_manifest = (
        yaml.safe_load(base_manifest_path.read_text(encoding="utf-8")) or {}
        if base_manifest_path.is_file()
        else {}
    )
    assert_robot_contract_compatible(base_manifest.get("robot_contract"), spec)
    rng = np.random.default_rng(cfg.seed)
    (
        _ik,
        _ikm,
        _SeedPoolConfig,
        _seed_pool,
        _halton,
        SelfCollisionFilter,
        build_locked_rail_model,
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
        default_branch_id=cfg.default_branch_id,
    )
    collision_filter, collision_urdf, collision_pairs = _probe_collision_filter(
        collision_cfg, lm, SelfCollisionFilter, spec=spec
    )
    kin = TorchRM75Kinematics.from_locked_model(lm, device=cfg.device)
    tool = spec.tool_frame()
    T_flange_tcp = tool.T_flange_tcp
    T_flange_tcp_t = torch.as_tensor(
        T_flange_tcp, dtype=torch.float32, device=kin.device
    )
    srs_cfg = _make_srs_cfg(spec, cfg.default_branch_id)
    metric_offsets = cfg.resolved_metric_offsets_m()
    near_band_m = float(min(metric_offsets))

    pos_pairs, rot_pairs = _select_pairs(base, cfg, rng)
    fields = _empty_stencil_fields()
    accepted = {CLEARANCE_POSITION: 0, CLEARANCE_ROTATION: 0}
    next_boundary_id = 0

    # Per-sample SRS conditioning from the base GT when present.
    has_branch = "branch_ids" in base and "psi_homes" in base

    for kind, pair_ids in (
        (CLEARANCE_POSITION, pos_pairs),
        (CLEARANCE_ROTATION, rot_pairs),
    ):
        for batch_index, start in enumerate(range(0, len(pair_ids), cfg.batch_size)):
            ids = pair_ids[start : start + cfg.batch_size]
            n = len(ids)
            source_ids = base["source_pose_id"][ids]
            source_f = base["features"][source_ids]
            target_f = base["features"][ids]
            # Features are flange se3; IK/SRS targets are TCP.
            source_p_fl = torch.as_tensor(
                source_f[:, :3], dtype=torch.float32, device=kin.device
            )
            target_p_fl = torch.as_tensor(
                target_f[:, :3], dtype=torch.float32, device=kin.device
            )
            source_R_fl = torch.as_tensor(
                _rotation_from_features(source_f), dtype=torch.float32, device=kin.device
            )
            target_R_fl = torch.as_tensor(
                _rotation_from_features(target_f), dtype=torch.float32, device=kin.device
            )
            source_p, source_R = pose_flange_to_tcp(
                source_p_fl, source_R_fl, T_flange_tcp_t
            )
            target_p, target_R = pose_flange_to_tcp(
                target_p_fl, target_R_fl, T_flange_tcp_t
            )
            delta_p = target_p - source_p
            delta_w = so3_log(target_R @ source_R.transpose(-1, -2))
            # Declared SE(3) ray length (metres).
            ray_m = se3_distance_m_torch(
                delta_p, delta_w, lambda_m_per_rad=spec.metric_lambda_m_per_rad
            ).clamp_min(1.0e-12)

            if has_branch:
                branch_ids = np.asarray(base["branch_ids"][source_ids], dtype=np.int32)
                psi_homes = np.asarray(base["psi_homes"][source_ids], dtype=np.float32)
            else:
                branch_ids = np.full(n, cfg.default_branch_id, dtype=np.int32)
                psi_homes = np.zeros(n, dtype=np.float32)

            lo = torch.zeros(n, dtype=torch.float32, device=kin.device)
            hi = torch.ones(n, dtype=torch.float32, device=kin.device)
            for _ in range(cfg.bisection_iterations):
                mid = 0.5 * (lo + hi)
                p_mid = source_p + mid[:, None] * delta_p
                R_mid = so3_exp(mid[:, None] * delta_w) @ source_R
                labeled, _qv = _label_tcp_batch(
                    p_mid,
                    R_mid,
                    srs_cfg=srs_cfg,
                    branch_ids=branch_ids,
                    psi_homes=psi_homes,
                    collision_filter=collision_filter,
                    kin=kin,
                )
                reach = torch.as_tensor(
                    labeled["reachable"], dtype=torch.bool, device=kin.device
                )
                lo = torch.where(reach, mid, lo)
                hi = torch.where(reach, hi, mid)
            t_boundary = 0.5 * (lo + hi)
            p_boundary = source_p + t_boundary[:, None] * delta_p
            R_boundary = so3_exp(t_boundary[:, None] * delta_w) @ source_R
            boundary_features = (
                _flange_se3_features(p_boundary, R_boundary, T_flange_tcp_t)
                .cpu()
                .numpy()
            )

            # Arc-length parameter: Δt = δ_m / ray_m along the SE(3) geodesic.
            signed_values = np.asarray(
                list(reversed(metric_offsets)) + [-x for x in metric_offsets],
                dtype=np.float32,
            )
            all_y, batch_rows = [], []
            for signed_m in signed_values:
                # Positive signed_m → toward reachable source (decreasing t).
                dt = float(signed_m) / ray_m
                p_stencil = p_boundary - dt[:, None] * delta_p
                R_stencil = so3_exp(-dt[:, None] * delta_w) @ R_boundary
                labeled, q_value = _label_tcp_batch(
                    p_stencil,
                    R_stencil,
                    srs_cfg=srs_cfg,
                    branch_ids=branch_ids,
                    psi_homes=psi_homes,
                    collision_filter=collision_filter,
                    kin=kin,
                )
                y = labeled["reachable"]
                all_y.append(y)
                batch_rows.append(
                    (
                        _flange_se3_features(p_stencil, R_stencil, T_flange_tcp_t)
                        .cpu()
                        .numpy(),
                        q_value,
                        labeled["q_best"],
                        labeled["branch"],
                        labeled["psi"],
                    )
                )
            labels = np.stack(all_y, axis=1)
            closest_pos = int(np.flatnonzero(signed_values > 0)[-1])
            closest_neg = int(np.flatnonzero(signed_values < 0)[0])
            straddles = labels[:, closest_pos] & (~labels[:, closest_neg])
            expected = signed_values[None, :] > 0
            monotonic = np.all(labels == expected, axis=1)
            keep = straddles
            kept_ids = np.arange(next_boundary_id, next_boundary_id + int(keep.sum()))
            next_boundary_id += int(keep.sum())
            for col, signed_m in enumerate(signed_values):
                features, q_value, q_best, branch, psi = batch_rows[col]
                _append_stencil_batch(
                    fields,
                    features[keep],
                    labels[keep, col],
                    q_value[keep],
                    q_best[keep],
                    source_ids[keep],
                    kept_ids,
                    np.full(int(keep.sum()), signed_m, dtype=np.float32),
                    kind,
                    monotonic[keep],
                    branch_ids[keep],
                    psi[keep],
                    psi_homes[keep],
                    boundary_features[keep],
                    near_band_m=near_band_m,
                )
            accepted[kind] += int(keep.sum())
            if cfg.log_every_batches > 0 and (batch_index + 1) % cfg.log_every_batches == 0:
                print(
                    f"[gpu-stencil] kind={kind} pairs={min(start+n,len(pair_ids))}/{len(pair_ids)} "
                    f"accepted={accepted[kind]}",
                    flush=True,
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
    for key in ("branch_ids", "psi", "psi_homes", "flange_canonical"):
        if key not in combined and key in base:
            combined[key] = base[key]
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
        "boundary_features": stencil["boundary_features"],
        "branch_ids": stencil["branch_ids"],
        "psi": stencil["psi"],
        "psi_homes": stencil["psi_homes"],
    }
    for key, values in append_map.items():
        if key not in combined:
            # Pad missing base columns (e.g. older arrays) before concat.
            if key in ("branch_ids",):
                combined[key] = np.full(n_base, -1, dtype=values.dtype)
            elif key in ("psi", "psi_homes"):
                combined[key] = np.full(n_base, np.nan, dtype=np.float32)
            elif key == "boundary_features":
                combined[key] = np.full(
                    (n_base, values.shape[1]), np.nan, dtype=np.float32
                )
            else:
                continue
        combined[key] = np.concatenate([combined[key], values], axis=0)
    # Recompute flange_canonical for the combined set if base had it.
    if "flange_canonical" in base:
        from ird_playground.ird.gpu_pose_gt import _flange_canonical_np

        combined["flange_canonical"] = _flange_canonical_np(
            combined["features"],
            np.eye(4, dtype=np.float64),
            spec.root_to_j1_axis(),
        )
    combined["label_kind"] = np.asarray([7], dtype=np.int32)
    combined["FLANGE_CANONICAL_DIM"] = np.asarray(
        [FLANGE_CANONICAL_DIM], dtype=np.int32
    )
    meta = {
        "label_kind": "gpu_pose_plus_srs_boundary_stencils_metric_m_v1",
        "feature_kind": "flange_se3_rot6d9",
        "feature_frame": "flange",
        "ik_target_frame": "tcp",
        "FLANGE_CANONICAL_DIM": FLANGE_CANONICAL_DIM,
        "base_gt_npz": str(base_path),
        "n_base": n_base,
        "n_stencil": n_stencil,
        "n_position_boundaries": accepted[CLEARANCE_POSITION],
        "n_rotation_boundaries": accepted[CLEARANCE_ROTATION],
        "n_monotonic_stencil_rows": int(stencil["margin_weight"].sum()),
        "collision_urdf": str(collision_urdf),
        "collision_pairs": str(collision_pairs),
        "collision_security_margin_m": float(spec.collision_security_margin_m),
        "T_flange_tcp": tool.T_flange_tcp.tolist(),
        "metric": metric_manifest(lambda_m_per_rad=spec.metric_lambda_m_per_rad),
        "metric_offsets_m": list(metric_offsets),
        "boundary_pose_schema": "on_manifold_flange_se3_bisection_v1",
        "m_gt_units": "metres_declared_se3_metric",
        "collision_contract": (
            "every positive stencil pose has an SRS+collision-free arm solution "
            "on the locked seed branch"
        ),
        "robot_contract": spec.to_manifest(),
        "srs_conditioning": {
            **srs_cfg.to_manifest(),
            "branch_id_policy": (
                "inherit branch_ids/psi_homes from base GT source seed when present; "
                f"else default_branch_id={cfg.default_branch_id}"
            ),
            "point_only_reachability": True,
            "y_rail_m": srs_cfg.resolved_y_rail_m(),
        },
        "config": asdict(cfg),
    }
    return combined, meta


__all__ = [
    "GpuBoundaryStencilConfig",
    "build_gpu_boundary_stencils",
]
