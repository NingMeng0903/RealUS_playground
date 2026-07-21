"""Efficient continuous FK/IK clearance supervision for Neural IRD.

The expensive operation is finding a trusted reachability boundary, not
evaluating a dense voxel grid. Each accepted 1-D ray is bisected once and then
expanded into a short signed-clearance curve:

* position: +/- 1, 3, 6, 10 mm;
* orientation: +/- 1, 3, 5 deg along a random SO(3) tangent direction.

Reachable-side samples use continuation IK from the boundary solution. On the
unreachable side, the nearest and farthest offsets are exhaustively checked
with the full seed pool; intermediate labels carry lower confidence under an
explicit local no-reentry assumption. This keeps label generation tractable
without pretending that a failed single-seed IK call proves unreachability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import sys
import types

import numpy as np

from ird_playground.ird.export_gt import (
    LAYER_BND_NEG,
    LAYER_BND_POS,
    LAYER_EXTERIOR,
    LAYER_INTERIOR,
    LAYER_JITTER_NEG,
    LAYER_JITTER_POS,
    _enforce_sign,
)

CLEARANCE_INTERIOR = -1
CLEARANCE_POSITION = 0
CLEARANCE_ROTATION = 1


@dataclass(frozen=True)
class ContinuousGtConfig:
    n_fk_interior: int = 4_000
    n_boundary_rays: int = 4_000
    rotation_ray_fraction: float = 0.35
    joint_limit_inset_fraction: float = 0.05
    n_random_seeds: int = 4
    ray_max_pos_m: float = 0.16
    ray_max_rot_deg: float = 25.0
    boundary_tol_m: float = 0.001
    boundary_tol_rot_deg: float = 0.25
    ik_tol_pos_m: float = 2.0e-4
    ik_tol_rot_rad: float = 1.0e-3
    ik_max_iter: int = 70
    margin_sigma_p_m: float = 0.006
    margin_sigma_rot_deg: float = 3.0
    position_offsets_mm: tuple[float, ...] = (1.0, 3.0, 6.0, 10.0)
    rotation_offsets_deg: tuple[float, ...] = (1.0, 3.0, 5.0)
    inferred_negative_weight: float = 0.5
    self_collision: bool = True
    collision_security_margin_m: float = 0.0
    collision_urdf: str | None = None
    collision_pairs: str | None = None
    m_clip: float = 3.0
    seed: int = 42
    holdout_block_m: float = 0.04
    log_every_rays: int = 100

    def validate(self) -> None:
        if self.n_fk_interior <= 0 or self.n_boundary_rays <= 0:
            raise ValueError("n_fk_interior and n_boundary_rays must be positive")
        if not 0.0 <= self.rotation_ray_fraction <= 1.0:
            raise ValueError("rotation_ray_fraction must lie in [0,1]")
        if not 0.0 <= self.joint_limit_inset_fraction < 0.5:
            raise ValueError("joint_limit_inset_fraction must lie in [0,0.5)")
        if not (0.0 < self.boundary_tol_m < self.ray_max_pos_m):
            raise ValueError("invalid position boundary tolerance")
        if not (0.0 < self.boundary_tol_rot_deg < self.ray_max_rot_deg):
            raise ValueError("invalid rotation boundary tolerance")
        if self.ik_tol_pos_m > self.boundary_tol_m:
            raise ValueError("IK position tolerance must not exceed boundary tolerance")
        if not self.position_offsets_mm or tuple(sorted(self.position_offsets_mm)) != tuple(self.position_offsets_mm):
            raise ValueError("position_offsets_mm must be non-empty and sorted")
        if not self.rotation_offsets_deg or tuple(sorted(self.rotation_offsets_deg)) != tuple(self.rotation_offsets_deg):
            raise ValueError("rotation_offsets_deg must be non-empty and sorted")
        if max(self.position_offsets_mm) * 1.0e-3 >= self.ray_max_pos_m:
            raise ValueError("position offsets must be smaller than ray_max_pos_m")
        if max(self.rotation_offsets_deg) >= self.ray_max_rot_deg:
            raise ValueError("rotation offsets must be smaller than ray_max_rot_deg")


def _quality_from_q(lm, q: np.ndarray) -> float:
    import pinocchio as pin

    pin.computeJointJacobians(lm.model, lm.data, q)
    pin.updateFramePlacements(lm.model, lm.data)
    J = pin.getFrameJacobian(
        lm.model, lm.data, lm.tcp_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )
    sigma = float(np.linalg.svd(J, compute_uv=False).min())
    span = np.maximum(lm.q_upper - lm.q_lower, 1.0e-6)
    clearance = float(np.min(np.minimum(q - lm.q_lower, lm.q_upper - q) / span))
    return float(np.clip((sigma / 0.08) * np.clip(clearance / 0.08, 0.0, 1.0), 0.0, 1.0))


def _feature_from_pose(M) -> np.ndarray:
    R = np.asarray(M.rotation)
    return np.concatenate([np.asarray(M.translation), R[:, 0], R[:, 1]])


def _pose_on_ray(M0, axis: np.ndarray, coordinate: float, kind: int):
    import pinocchio as pin

    if kind == CLEARANCE_POSITION:
        return pin.SE3(M0.rotation.copy(), M0.translation + axis * float(coordinate))
    if kind == CLEARANCE_ROTATION:
        return pin.SE3(pin.exp3(axis * np.deg2rad(float(coordinate))) @ M0.rotation, M0.translation.copy())
    raise ValueError(f"unsupported clearance kind {kind}")


def _block_ids(features: np.ndarray, width_m: float) -> np.ndarray:
    ijk = np.floor(features[:, :3] / float(width_m)).astype(np.int64)
    return ijk[:, 0] * 1_000_000_000_000 + ijk[:, 1] * 1_000_000 + ijk[:, 2]


def _reachability_modules():
    try:
        from rm75_control.tools.reachability.kinematics.ik_dls import ik_dls, ik_dls_multiseed
        from rm75_control.tools.reachability.kinematics.ik_seeds import SeedPoolConfig, build_seed_pool, halton_matrix
        from rm75_control.tools.reachability.build.self_collision import SelfCollisionFilter
        from rm75_control.tools.reachability.kinematics.model_locked_rail import build_locked_rail_model
        return ik_dls, ik_dls_multiseed, SeedPoolConfig, build_seed_pool, halton_matrix, SelfCollisionFilter, build_locked_rail_model
    except ModuleNotFoundError as exc:
        if exc.name not in {"Robotic_Arm", "Robotic_Arm.rm_ctypes_wrap"}:
            raise
    for name in list(sys.modules):
        if name == "rm75_control" or name.startswith("rm75_control."):
            sys.modules.pop(name, None)
    package_root = Path(__file__).resolve().parents[3] / "rm75_control" / "rm75_control"
    package = types.ModuleType("rm75_control")
    package.__path__ = [str(package_root)]
    sys.modules["rm75_control"] = package
    from rm75_control.tools.reachability.kinematics.ik_dls import ik_dls, ik_dls_multiseed
    from rm75_control.tools.reachability.kinematics.ik_seeds import SeedPoolConfig, build_seed_pool, halton_matrix
    from rm75_control.tools.reachability.build.self_collision import SelfCollisionFilter
    from rm75_control.tools.reachability.kinematics.model_locked_rail import build_locked_rail_model
    return ik_dls, ik_dls_multiseed, SeedPoolConfig, build_seed_pool, halton_matrix, SelfCollisionFilter, build_locked_rail_model


def build_continuous_ird_gt(cfg: ContinuousGtConfig | None = None) -> tuple[dict[str, np.ndarray], dict]:
    cfg = cfg or ContinuousGtConfig()
    cfg.validate()
    import pinocchio as pin

    (
        ik_dls, ik_dls_multiseed, SeedPoolConfig, build_seed_pool,
        halton_matrix, SelfCollisionFilter, build_locked_rail_model,
    ) = _reachability_modules()
    rng = np.random.default_rng(cfg.seed)
    lm = build_locked_rail_model()
    n_candidates = max(cfg.n_fk_interior * 2, cfg.n_fk_interior + 128)
    u_q = halton_matrix(n_candidates, lm.model.nq)
    span_q = lm.q_upper - lm.q_lower
    inset_q = cfg.joint_limit_inset_fraction * span_q
    Q = (lm.q_lower + inset_q)[None, :] + u_q * (span_q - 2.0 * inset_q)[None, :]
    collision_filter = None
    if cfg.self_collision:
        rm_root = Path(__file__).resolve().parents[3] / "rm75_control"
        collision_urdf = Path(cfg.collision_urdf) if cfg.collision_urdf else (
            rm_root / "data/urdf_patched/RM75-horizontal_probe.collision.urdf"
        )
        collision_pairs = Path(cfg.collision_pairs) if cfg.collision_pairs else (
            rm_root / "data/urdf_patched/collision_pairs_probe.yaml"
        )
        collision_filter = SelfCollisionFilter(
            kin_urdf=lm.urdf_path,
            collision_urdf=collision_urdf,
            pair_config=collision_pairs,
            rail_locked_at_m=lm.rail_locked_at_m,
            security_margin=cfg.collision_security_margin_m,
        )
        Q = Q[collision_filter.free_mask(Q)]
        if len(Q) < cfg.n_fk_interior:
            raise RuntimeError(
                f"only {len(Q)} collision-free FK samples from {n_candidates} candidates"
            )
    Q = Q[: cfg.n_fk_interior]
    seeds_global = build_seed_pool(
        lm.q_lower,
        lm.q_upper,
        SeedPoolConfig(n_random=cfg.n_random_seeds, random_seed=cfg.seed),
    )
    ik_kw = dict(
        max_iter=cfg.ik_max_iter,
        tol_pos_m=cfg.ik_tol_pos_m,
        tol_rot_rad=cfg.ik_tol_rot_rad,
    )

    fields: dict[str, list] = {
        k: [] for k in (
            "feature", "y", "m", "q", "mw", "cw", "layer", "qbest",
            "boundary_id", "signed_m", "signed_deg", "kind",
        )
    }
    poses: list[tuple[object, np.ndarray]] = []
    accepted_by_kind = {CLEARANCE_POSITION: 0, CLEARANCE_ROTATION: 0}
    rejected = 0

    def append(
        M, *, y: float, margin: float, quality: float, margin_weight: float,
        cls_weight: float, layer: int, qbest: np.ndarray, boundary_id: int = -1,
        signed_m: float = np.nan, signed_deg: float = np.nan, kind: int = CLEARANCE_INTERIOR,
    ) -> None:
        fields["feature"].append(_feature_from_pose(M))
        fields["y"].append(y)
        fields["m"].append(float(np.clip(margin, -cfg.m_clip, cfg.m_clip)))
        fields["q"].append(quality if y >= 0.5 else 0.0)
        fields["mw"].append(margin_weight)
        fields["cw"].append(cls_weight)
        fields["layer"].append(layer)
        fields["qbest"].append(np.asarray(qbest, dtype=np.float32))
        fields["boundary_id"].append(boundary_id)
        fields["signed_m"].append(signed_m)
        fields["signed_deg"].append(signed_deg)
        fields["kind"].append(kind)

    for q in Q:
        pin.forwardKinematics(lm.model, lm.data, q)
        pin.updateFramePlacement(lm.model, lm.data, lm.tcp_id)
        M = lm.data.oMf[lm.tcp_id].copy()
        poses.append((M, q.copy()))
        append(
            M, y=1.0, margin=cfg.m_clip, quality=_quality_from_q(lm, q),
            margin_weight=0.0, cls_weight=1.0, layer=LAYER_INTERIOR, qbest=q,
        )

    def exhaustive(target, q_hint):
        seeds = np.vstack([np.asarray(q_hint)[None, :], seeds_global])
        result = ik_dls_multiseed(lm, target, seeds, keep_best=False, **ik_kw)
        if collision_filter is None or not result.report.ok:
            return result
        if not collision_filter.in_collision(result.q):
            return result
        # The first converged branch collided. Search every branch and accept
        # only a converged, probe-aware collision-free configuration.
        last = result
        for seed in seeds:
            candidate = ik_dls(lm, target, seed, **ik_kw)
            last = candidate
            if candidate.report.ok and not collision_filter.in_collision(candidate.q):
                return candidate
        return replace(last, report=replace(last.report, ok=False))

    def local_then_exhaustive(target, q_hint):
        local = ik_dls(lm, target, np.asarray(q_hint), **ik_kw)
        local_free = local.report.ok and (
            collision_filter is None or not collision_filter.in_collision(local.q)
        )
        return local if local_free else exhaustive(target, q_hint)

    for ray_idx in range(cfg.n_boundary_rays):
        if ray_idx and ray_idx % max(1, cfg.log_every_rays) == 0:
            print(
                f"[clearance-gt] rays={ray_idx}/{cfg.n_boundary_rays} "
                f"pos={accepted_by_kind[CLEARANCE_POSITION]} "
                f"rot={accepted_by_kind[CLEARANCE_ROTATION]} rejected={rejected}",
                flush=True,
            )
        M0, q0 = poses[ray_idx % len(poses)]
        kind = CLEARANCE_ROTATION if rng.random() < cfg.rotation_ray_fraction else CLEARANCE_POSITION
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis) + 1.0e-12
        ray_max = cfg.ray_max_pos_m if kind == CLEARANCE_POSITION else cfg.ray_max_rot_deg
        tolerance = cfg.boundary_tol_m if kind == CLEARANCE_POSITION else cfg.boundary_tol_rot_deg

        M_far = _pose_on_ray(M0, axis, ray_max, kind)
        far = exhaustive(M_far, q0)
        if far.report.ok:
            rejected += 1
            continue
        lo, hi, q_lo = 0.0, ray_max, q0.copy()
        while hi - lo > tolerance:
            mid = 0.5 * (lo + hi)
            M_mid = _pose_on_ray(M0, axis, mid, kind)
            result = local_then_exhaustive(M_mid, q_lo)
            if result.report.ok:
                lo, q_lo = mid, result.q.copy()
            else:
                hi = mid

        offsets = (
            tuple(x * 1.0e-3 for x in cfg.position_offsets_mm)
            if kind == CLEARANCE_POSITION
            else cfg.rotation_offsets_deg
        )
        # A curve is only valid when every requested offset exists on both
        # sides. Clamping would assign several clearances to the same pose and
        # create contradictory regression targets.
        if lo < offsets[-1] or hi + offsets[-1] > ray_max:
            rejected += 1
            continue
        # Verify the closest and farthest outside points exhaustively. If either
        # is reachable, this ray does not define a trustworthy local curve.
        neg_results = {}
        curve_valid = True
        for offset in (offsets[0], offsets[-1]):
            coord = hi + offset
            M_neg = _pose_on_ray(M0, axis, coord, kind)
            res = exhaustive(M_neg, q_lo)
            neg_results[offset] = (M_neg, res)
            if res.report.ok:
                curve_valid = False
                break
        if not curve_valid:
            rejected += 1
            continue

        positive_rows = []
        q_hint = q_lo.copy()
        for offset in offsets:
            coord = lo - offset
            M_pos = _pose_on_ray(M0, axis, coord, kind)
            res = local_then_exhaustive(M_pos, q_hint)
            if not res.report.ok:
                curve_valid = False
                break
            q_hint = res.q.copy()
            positive_rows.append((offset, M_pos, res))
        if not curve_valid:
            rejected += 1
            continue

        for j, (offset, M_pos, res) in enumerate(positive_rows):
            normalized = offset / (cfg.margin_sigma_p_m if kind == CLEARANCE_POSITION else cfg.margin_sigma_rot_deg)
            append(
                M_pos, y=1.0, margin=normalized, quality=_quality_from_q(lm, res.q),
                margin_weight=1.0, cls_weight=1.0,
                layer=LAYER_BND_POS if j == 0 else LAYER_JITTER_POS,
                qbest=res.q, boundary_id=ray_idx,
                signed_m=offset if kind == CLEARANCE_POSITION else np.nan,
                signed_deg=offset if kind == CLEARANCE_ROTATION else np.nan, kind=kind,
            )
        for j, offset in enumerate(offsets):
            if offset in neg_results:
                M_neg, res = neg_results[offset]
                confidence = 1.0
                q_failed = res.q
            else:
                M_neg = _pose_on_ray(M0, axis, hi + offset, kind)
                confidence = cfg.inferred_negative_weight
                q_failed = far.q
            normalized = -offset / (cfg.margin_sigma_p_m if kind == CLEARANCE_POSITION else cfg.margin_sigma_rot_deg)
            append(
                M_neg, y=0.0, margin=normalized, quality=0.0,
                margin_weight=confidence, cls_weight=confidence,
                layer=(LAYER_BND_NEG if j == 0 else (LAYER_EXTERIOR if j == len(offsets) - 1 else LAYER_JITTER_NEG)),
                qbest=q_failed, boundary_id=ray_idx,
                signed_m=-offset if kind == CLEARANCE_POSITION else np.nan,
                signed_deg=-offset if kind == CLEARANCE_ROTATION else np.nan, kind=kind,
            )
        accepted_by_kind[kind] += 1

    accepted = sum(accepted_by_kind.values())
    if accepted == 0:
        raise RuntimeError("no trusted clearance curves; enlarge ray ranges or seed pool")

    features = np.asarray(fields["feature"], dtype=np.float32)
    y = np.asarray(fields["y"], dtype=np.float32)
    margin = _enforce_sign(np.asarray(fields["m"]), y, 1.0e-5, cfg.m_clip)
    quality = np.asarray(fields["q"], dtype=np.float32)
    qbest = np.asarray(fields["qbest"], dtype=np.float32)
    n = len(y)
    scale = np.maximum(np.abs(features[:, :3]).max(axis=0) * 1.05, 0.1).astype(np.float32)
    arrays = {
        "features": features,
        "reachable": y,
        "p_reach": y,
        "y_soft": y,
        "cls_weight": np.asarray(fields["cw"], dtype=np.float32),
        "m_gt": margin,
        "margin_weight": np.asarray(fields["mw"], dtype=np.float32),
        "layer_id": np.asarray(fields["layer"], dtype=np.int32),
        "voxel_id": np.arange(n, dtype=np.int32),
        "orient_id": np.zeros(n, dtype=np.int32),
        "block_id": _block_ids(features, cfg.holdout_block_m),
        "boundary_id": np.asarray(fields["boundary_id"], dtype=np.int32),
        "boundary_signed_m": np.asarray(fields["signed_m"], dtype=np.float32),
        "boundary_signed_rot_deg": np.asarray(fields["signed_deg"], dtype=np.float32),
        "clearance_kind": np.asarray(fields["kind"], dtype=np.int8),
        "q": quality,
        "q_comfort": quality,
        "q_capability": quality,
        "q_manip": quality,
        "q_joint": quality,
        "q_selfcol": quality,
        "q_nullspace": quality,
        "q_best": qbest,
        "q_candidates": qbest[:, None, :],
        "d": y * quality,
        "aabb_lo": -scale,
        "aabb_hi": scale,
        "sigma_p_m": np.asarray([cfg.margin_sigma_p_m], dtype=np.float32),
        "sigma_r_deg": np.asarray([cfg.margin_sigma_rot_deg], dtype=np.float32),
        "feature_dim": np.asarray([9], dtype=np.int32),
        "feature_kind": np.asarray([3], dtype=np.int32),
        "label_kind": np.asarray([5], dtype=np.int32),
    }
    meta = {
        "label_kind": "continuous_fk_ik_clearance_curves_v2",
        "feature_kind": "se3_rot6d9",
        "rotation_representation": "first_two_columns_of_R_base_tcp",
        "tcp_frame": lm.tcp_frame,
        "urdf_path": str(lm.urdf_path),
        "rail_locked_at_m": lm.rail_locked_at_m,
        "n_total": n,
        "n_boundary_rays_requested": cfg.n_boundary_rays,
        "n_position_curves": accepted_by_kind[CLEARANCE_POSITION],
        "n_rotation_curves": accepted_by_kind[CLEARANCE_ROTATION],
        "n_boundary_rays_rejected": rejected,
        "physical_boundary_tolerance_m": cfg.boundary_tol_m,
        "angular_boundary_tolerance_deg": cfg.boundary_tol_rot_deg,
        "ik_position_tolerance_m": cfg.ik_tol_pos_m,
        "ik_rotation_tolerance_rad": cfg.ik_tol_rot_rad,
        "negative_contract": "nearest/farthest exhaustive multi-seed; intermediate local no-reentry weighted labels",
        "quality_contract": "conditioning and joint clearance of continuation IK branch",
        "collision_contract": (
            "reachable iff at least one IK branch is collision-free under the robot+probe model"
            if cfg.self_collision else "self-collision disabled"
        ),
        "config": asdict(cfg),
    }
    return arrays, meta
