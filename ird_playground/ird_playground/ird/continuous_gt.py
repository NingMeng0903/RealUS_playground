"""Continuous FK/IK-supervised GT for the Neural IRD point field.

This deliberately does not upsample a voxel map.  Positives come from the
locked-rail RM75 Pinocchio FK, and each near-boundary pair is bracketed by a
multi-seed IK query then refined with SE(3) bisection.  The generator is
offline and conservative: an IK failure is labelled negative only after the
configured seed pool has been exhausted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
    features_from_p_u,
)


@dataclass(frozen=True)
class ContinuousGtConfig:
    """Controls for continuous 5-DoF [p, tool +Z] GT generation."""

    n_fk_interior: int = 2_000
    n_boundary_rays: int = 2_000
    n_random_seeds: int = 6
    ray_max_pos_m: float = 0.12
    ray_max_rot_deg: float = 18.0
    boundary_tol_m: float = 2.0e-4
    ik_tol_pos_m: float = 5.0e-5
    ik_tol_rot_rad: float = 1.0e-3
    ik_max_iter: int = 80
    margin_sigma_p_m: float = 0.003
    boundary_offset_m: float = 0.001
    exterior_offset_m: float = 0.010
    m_clip: float = 3.0
    seed: int = 42
    holdout_block_m: float = 0.04
    log_every_rays: int = 100

    def validate(self) -> None:
        if self.n_fk_interior <= 0 or self.n_boundary_rays <= 0:
            raise ValueError("n_fk_interior and n_boundary_rays must be positive")
        if not (0.0 < self.boundary_tol_m <= self.boundary_offset_m):
            raise ValueError("boundary_tol_m must be positive and <= boundary_offset_m")
        if self.ik_tol_pos_m > self.boundary_tol_m:
            raise ValueError("IK position tolerance must not exceed boundary tolerance")
        if self.ray_max_pos_m <= self.exterior_offset_m:
            raise ValueError("ray_max_pos_m must exceed exterior_offset_m")


def _quality_from_q(lm, q: np.ndarray) -> float:
    """Unitless comfort proxy from conditioning and joint-limit clearance."""
    import pinocchio as pin

    pin.computeJointJacobians(lm.model, lm.data, q)
    pin.updateFramePlacements(lm.model, lm.data)
    J = pin.getFrameJacobian(
        lm.model, lm.data, lm.tcp_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )
    sigma = float(np.linalg.svd(J, compute_uv=False).min())
    span = np.maximum(lm.q_upper - lm.q_lower, 1.0e-6)
    joint_clearance = float(np.min(np.minimum(q - lm.q_lower, lm.q_upper - q) / span))
    return float(np.clip((sigma / 0.08) * np.clip(joint_clearance / 0.08, 0.0, 1.0), 0.0, 1.0))


def _feature_from_pose(M) -> np.ndarray:
    return features_from_p_u(np.asarray(M.translation), np.asarray(M.rotation)[:, 2])


def _ray_pose(M0, pos_dir: np.ndarray, rot_axis: np.ndarray, fraction: float, cfg: ContinuousGtConfig):
    import pinocchio as pin

    f = float(fraction)
    R = pin.exp3(rot_axis * (f * np.deg2rad(cfg.ray_max_rot_deg))) @ M0.rotation
    p = M0.translation + pos_dir * (f * cfg.ray_max_pos_m)
    return pin.SE3(R, p)


def _block_ids(features: np.ndarray, width_m: float) -> np.ndarray:
    ijk = np.floor(features[:, :3] / float(width_m)).astype(np.int64)
    return ijk[:, 0] * 1_000_000_000_000 + ijk[:, 1] * 1_000_000 + ijk[:, 2]


def _reachability_modules():
    """Import offline reachability code without initializing the hardware SDK.

    ``rm75_control.__init__`` intentionally imports live-controller helpers,
    which require the vendor ``Robotic_Arm`` package.  The capability-map
    kinematics are pure Pinocchio and must also remain usable on CI/offline
    training hosts.
    """
    try:
        from rm75_control.tools.reachability.kinematics.ik_dls import ik_dls_multiseed
        from rm75_control.tools.reachability.kinematics.ik_seeds import SeedPoolConfig, build_seed_pool, halton_matrix
        from rm75_control.tools.reachability.kinematics.model_locked_rail import build_locked_rail_model
        return ik_dls_multiseed, SeedPoolConfig, build_seed_pool, halton_matrix, build_locked_rail_model
    except ModuleNotFoundError as exc:
        if exc.name not in {"Robotic_Arm", "Robotic_Arm.rm_ctypes_wrap"}:
            raise
    # Bypass only the package initializer; never shadow submodule source files.
    for name in list(sys.modules):
        if name == "rm75_control" or name.startswith("rm75_control."):
            sys.modules.pop(name, None)
    package_root = Path(__file__).resolve().parents[3] / "rm75_control" / "rm75_control"
    offline_pkg = types.ModuleType("rm75_control")
    offline_pkg.__path__ = [str(package_root)]
    sys.modules["rm75_control"] = offline_pkg
    from rm75_control.tools.reachability.kinematics.ik_dls import ik_dls_multiseed
    from rm75_control.tools.reachability.kinematics.ik_seeds import SeedPoolConfig, build_seed_pool, halton_matrix
    from rm75_control.tools.reachability.kinematics.model_locked_rail import build_locked_rail_model
    return ik_dls_multiseed, SeedPoolConfig, build_seed_pool, halton_matrix, build_locked_rail_model


def build_continuous_ird_gt(cfg: ContinuousGtConfig | None = None) -> tuple[dict[str, np.ndarray], dict]:
    """Generate continuous FK/IK labels in the arm-base frame.

    Returns the standard IRD NPZ contract and a provenance dictionary suitable
    for ``save_ird_gt``.  Generation is intentionally single-process because
    Pinocchio ``Data`` is mutable; callers may shard runs externally if needed.
    """
    cfg = cfg or ContinuousGtConfig()
    cfg.validate()
    import pinocchio as pin
    (
        ik_dls_multiseed,
        SeedPoolConfig,
        build_seed_pool,
        halton_matrix,
        build_locked_rail_model,
    ) = _reachability_modules()

    rng = np.random.default_rng(cfg.seed)
    lm = build_locked_rail_model()
    # Low-discrepancy FK samples span the actual joint-limited manifold.
    u_q = halton_matrix(cfg.n_fk_interior, lm.model.nq)
    Q = lm.q_lower[None, :] + u_q * (lm.q_upper - lm.q_lower)[None, :]
    seeds_global = build_seed_pool(
        lm.q_lower,
        lm.q_upper,
        SeedPoolConfig(n_random=cfg.n_random_seeds, random_seed=cfg.seed),
    )

    f_list: list[np.ndarray] = []
    y_list: list[float] = []
    m_list: list[float] = []
    q_list: list[float] = []
    mw_list: list[float] = []
    layer_list: list[int] = []
    qbest_list: list[np.ndarray] = []
    boundary_id_list: list[int] = []
    boundary_signed_m_list: list[float] = []
    accepted = 0
    failed_rays = 0

    def append(
        M, y: float, margin_m: float, quality: float, mw: float, layer: int, qbest: np.ndarray,
        *, boundary_id: int = -1, boundary_signed_m: float = np.nan,
    ) -> None:
        f_list.append(_feature_from_pose(M))
        y_list.append(float(y))
        sign_margin = (margin_m / cfg.margin_sigma_p_m)
        m_list.append(float(np.clip(sign_margin, -cfg.m_clip, cfg.m_clip)))
        q_list.append(float(quality if y >= 0.5 else 0.0))
        mw_list.append(float(mw))
        layer_list.append(int(layer))
        qbest_list.append(np.asarray(qbest, dtype=np.float32))
        boundary_id_list.append(int(boundary_id))
        boundary_signed_m_list.append(float(boundary_signed_m))

    # Exact continuous FK positives, including quality from actual Jacobian.
    poses = []
    for q in Q:
        pin.forwardKinematics(lm.model, lm.data, q)
        pin.updateFramePlacement(lm.model, lm.data, lm.tcp_id)
        M = lm.data.oMf[lm.tcp_id].copy()
        poses.append((M, q.copy()))
        append(M, 1.0, cfg.margin_sigma_p_m, _quality_from_q(lm, q), 0.0, LAYER_INTERIOR, q)

    # Every accepted ray supplies IK-verified +/- boundary and wider jitter pairs.
    for ray_idx in range(cfg.n_boundary_rays):
        if ray_idx and ray_idx % max(1, int(cfg.log_every_rays)) == 0:
            print(
                f"[continuous-gt] rays={ray_idx}/{cfg.n_boundary_rays} "
                f"accepted={accepted} rejected={failed_rays}",
                flush=True,
            )
        M0, q0 = poses[ray_idx % len(poses)]
        pos_dir = rng.normal(size=3)
        pos_dir /= np.linalg.norm(pos_dir) + 1.0e-12
        rot_axis = rng.normal(size=3)
        rot_axis /= np.linalg.norm(rot_axis) + 1.0e-12

        def solve(frac: float):
            target = _ray_pose(M0, pos_dir, rot_axis, frac, cfg)
            # Include the source configuration first; it makes the local branch
            # deterministic while global seeds still search alternate branches.
            seeds = np.vstack([q0[None, :], seeds_global])
            return target, ik_dls_multiseed(
                lm,
                target,
                seeds,
                # Reachability labels need an existential IK answer.  Stopping
                # at the first solution preserves multi-seed coverage for
                # failures while avoiding an unnecessary full seed sweep for
                # every successful bisection point.
                keep_best=False,
                max_iter=cfg.ik_max_iter,
                tol_pos_m=cfg.ik_tol_pos_m,
                tol_rot_rad=cfg.ik_tol_rot_rad,
            )

        M_hi, hi = solve(1.0)
        if hi.report.ok:
            failed_rays += 1
            continue
        lo_f, hi_f = 0.0, 1.0
        lo_M, lo = M0, None
        for _ in range(32):
            if (hi_f - lo_f) * cfg.ray_max_pos_m <= cfg.boundary_tol_m:
                break
            mid = 0.5 * (lo_f + hi_f)
            M_mid, res = solve(mid)
            if res.report.ok:
                lo_f, lo_M, lo = mid, M_mid, res
            else:
                hi_f, M_hi, hi = mid, M_mid, res
        if lo is None:
            # The ray may cross a narrow local failure basin immediately.  It
            # is not trusted as a continuous boundary sample.
            failed_rays += 1
            continue

        offset_f = cfg.boundary_offset_m / cfg.ray_max_pos_m
        inside_f = max(0.0, lo_f - offset_f)
        outside_f = min(1.0, hi_f + offset_f)
        M_in, res_in = solve(inside_f)
        M_out, res_out = solve(outside_f)
        if not res_in.report.ok or res_out.report.ok:
            failed_rays += 1
            continue
        quality = _quality_from_q(lm, res_in.q)
        append(
            M_in, 1.0, cfg.boundary_offset_m, quality, 1.0, LAYER_BND_POS, res_in.q,
            boundary_id=ray_idx, boundary_signed_m=cfg.boundary_offset_m,
        )
        append(
            M_out, 0.0, -cfg.boundary_offset_m, 0.0, 1.0, LAYER_BND_NEG, res_out.q,
            boundary_id=ray_idx, boundary_signed_m=-cfg.boundary_offset_m,
        )

        jitter_f = min(1.0, 2.0 * offset_f)
        M_jp, res_jp = solve(max(0.0, lo_f - jitter_f))
        M_jn, res_jn = solve(min(1.0, hi_f + jitter_f))
        if res_jp.report.ok:
            append(
                M_jp, 1.0, 2.0 * cfg.boundary_offset_m, _quality_from_q(lm, res_jp.q), 1.0,
                LAYER_JITTER_POS, res_jp.q, boundary_id=ray_idx,
                boundary_signed_m=2.0 * cfg.boundary_offset_m,
            )
        if not res_jn.report.ok:
            append(
                M_jn, 0.0, -2.0 * cfg.boundary_offset_m, 0.0, 1.0, LAYER_JITTER_NEG,
                res_jn.q, boundary_id=ray_idx, boundary_signed_m=-2.0 * cfg.boundary_offset_m,
            )
        # A farther verified negative stabilizes the global classifier.
        far_f = min(1.0, hi_f + cfg.exterior_offset_m / cfg.ray_max_pos_m)
        M_far, res_far = solve(far_f)
        if not res_far.report.ok:
            append(
                M_far, 0.0, -cfg.exterior_offset_m, 0.0, 0.0, LAYER_EXTERIOR, res_far.q,
                boundary_id=ray_idx, boundary_signed_m=-cfg.exterior_offset_m,
            )
        accepted += 1

    if accepted == 0:
        raise RuntimeError("no trusted FK/IK boundary rays; enlarge ray range or seed pool")

    features = np.asarray(f_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    m = _enforce_sign(np.asarray(m_list), y, 1.0e-5, cfg.m_clip)
    qv = np.asarray(q_list, dtype=np.float32)
    mw = np.asarray(mw_list, dtype=np.float32)
    layers = np.asarray(layer_list, dtype=np.int32)
    qbest = np.asarray(qbest_list, dtype=np.float32)
    scale = np.maximum(np.abs(features[:, :3]).max(axis=0) * 1.05, 0.1).astype(np.float32)
    arrays = {
        "features": features,
        "reachable": y,
        "p_reach": y,
        "y_soft": y,
        "cls_weight": np.ones_like(y),
        "m_gt": m,
        "margin_weight": mw,
        "layer_id": layers,
        "voxel_id": np.arange(len(y), dtype=np.int32),
        "orient_id": np.zeros(len(y), dtype=np.int32),
        "block_id": _block_ids(features, cfg.holdout_block_m),
        "boundary_id": np.asarray(boundary_id_list, dtype=np.int32),
        "boundary_signed_m": np.asarray(boundary_signed_m_list, dtype=np.float32),
        "q": qv,
        "q_comfort": qv,
        "q_capability": qv,
        "q_manip": qv,
        "q_joint": qv,
        "q_selfcol": qv,
        "q_nullspace": qv,
        "q_best": qbest,
        "q_candidates": qbest[:, None, :],
        "d": y * qv,
        "aabb_lo": -scale,
        "aabb_hi": scale,
        "sigma_p_m": np.asarray([cfg.margin_sigma_p_m], dtype=np.float32),
        "sigma_r_deg": np.asarray([cfg.ray_max_rot_deg], dtype=np.float32),
        "feature_dim": np.asarray([6], dtype=np.int32),
        "feature_kind": np.asarray([1], dtype=np.int32),
        "label_kind": np.asarray([4], dtype=np.int32),  # 4 = continuous FK/IK boundary
    }
    meta = {
        "label_kind": "continuous_fk_multiseed_ik_se3_bisection_v1",
        "feature_kind": "natural_pu",
        "tool_axis": "tcp_plus_z",
        "tcp_frame": lm.tcp_frame,
        "urdf_path": str(lm.urdf_path),
        "rail_locked_at_m": lm.rail_locked_at_m,
        "n_total": int(len(y)),
        "n_boundary_rays_requested": cfg.n_boundary_rays,
        "n_boundary_rays_accepted": accepted,
        "n_boundary_rays_rejected": failed_rays,
        "config": asdict(cfg),
        "physical_boundary_tolerance_m": cfg.boundary_tol_m,
        "ik_position_tolerance_m": cfg.ik_tol_pos_m,
        "ik_rotation_tolerance_rad": cfg.ik_tol_rot_rad,
        "negative_contract": "multi-seed DLS IK exhausted at specified tolerances",
        "quality_contract": "conditioning/joint-clearance of first successful multi-seed IK branch",
    }
    return arrays, meta
