"""Acceptance evaluation for the RM4D signed IRD and Region A query."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from ird_playground.ird.canonical import rotation_from_6d_torch
from ird_playground.ird.gpu_pose_gt import GpuPoseGtConfig, _probe_collision_filter
from ird_playground.ird.gt_common import reachability_modules
from ird_playground.ird.robot_model import load_robot_model_spec
from ird_playground.ird.torch_kinematics import TorchRM75Kinematics, so3_log
from ird_playground.neural.signed_field import ReachabilitySDF
from ird_playground.neural.train_signed import (
    _side_indices,
    _split_indices,
    evaluate_signed_field,
    load_signed_train_config,
)
from ird_playground.region.operator import RegionA, RegionAConfig


def _pose_matrices(features: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    R = rotation_from_6d_torch(x[:, 3:9])
    bottom = torch.zeros(len(x), 1, 4, dtype=x.dtype, device=device)
    bottom[:, 0, 3] = 1.0
    return torch.cat((torch.cat((R, x[:, :3, None]), dim=-1), bottom), dim=-2)


@torch.no_grad()
def _region_pair_direction(
    field: ReachabilitySDF,
    region: RegionA,
    source: dict[str, np.ndarray],
    val_groups: np.ndarray,
    kind: int,
    signed_key: str,
    n: int,
    seed: int,
    T_axis_root: torch.Tensor,
) -> float:
    bid = source["boundary_id"]
    signed = source[signed_key]
    valid = (source["clearance_kind"] == kind) & np.isin(bid, val_groups)
    local_bid = np.where(valid, bid, -1)
    gp, ip = _side_indices(local_bid, signed, True, True)
    gn, inn = _side_indices(local_bid, signed, False, True)
    _, pa, na = np.intersect1d(gp, gn, assume_unique=True, return_indices=True)
    ip, inn = ip[pa], inn[na]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(ip), size=min(n, len(ip)), replace=False)
    Tp = _pose_matrices(source["features"][ip[pick]], field.device)
    Tn = _pose_matrices(source["features"][inn[pick]], field.device)
    sp = region(field.model, Tp, T_axis_root).robust_clearance
    sn = region(field.model, Tn, T_axis_root).robust_clearance
    return float((sp > sn).float().mean().item())


def _ad_fd_audit(
    field: ReachabilitySDF,
    region: RegionA,
    features: np.ndarray,
    *,
    T_axis_root: torch.Tensor,
    n: int = 24,
    eps: float = 1.0e-3,
    seed: int = 3,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(features), size=min(n, len(features)), replace=False)
    rail_rel, tcp_rel, finite = [], [], []
    axis = T_axis_root
    for i in ids:
        base_pose = _pose_matrices(features[i : i + 1], field.device)[0]
        xyz = base_pose[:3, 3].detach().clone().requires_grad_(True)
        T = torch.cat((torch.cat((base_pose[:3, :3], xyz[:, None]), dim=1), base_pose[3:4]), dim=0)
        rail = torch.tensor(0.0, device=field.device, requires_grad=True)
        value = region.query_tcp_rail(
            field.model, T, rail, T_world_rail=torch.eye(4, device=field.device), T_rail_base0=axis
        ).robust_clearance
        g_xyz, g_rail = torch.autograd.grad(value, (xyz, rail))
        with torch.no_grad():
            rp = region.query_tcp_rail(field.model, T, rail + eps, T_world_rail=torch.eye(4, device=field.device), T_rail_base0=axis).robust_clearance
            rm = region.query_tcp_rail(field.model, T, rail - eps, T_world_rail=torch.eye(4, device=field.device), T_rail_base0=axis).robust_clearance
            rail_fd = (rp - rm) / (2.0 * eps)
            Tplus, Tminus = T.clone(), T.clone()
            Tplus[0, 3] += eps
            Tminus[0, 3] -= eps
            xp = region(field.model, Tplus, axis).robust_clearance
            xm = region(field.model, Tminus, axis).robust_clearance
            x_fd = (xp - xm) / (2.0 * eps)
        rail_rel.append(float(torch.abs(g_rail - rail_fd) / torch.maximum(torch.maximum(torch.abs(g_rail), torch.abs(rail_fd)), torch.tensor(1.0e-5, device=field.device))))
        tcp_rel.append(float(torch.abs(g_xyz[0] - x_fd) / torch.maximum(torch.maximum(torch.abs(g_xyz[0]), torch.abs(x_fd)), torch.tensor(1.0e-5, device=field.device))))
        finite.append(float(torch.isfinite(g_xyz).all() and torch.isfinite(g_rail)))
    return {
        "rail_ad_fd_median_relative_error": float(np.median(rail_rel)),
        "rail_ad_fd_p95_relative_error": float(np.percentile(rail_rel, 95)),
        "tcp_x_ad_fd_median_relative_error": float(np.median(tcp_rel)),
        "tcp_x_ad_fd_p95_relative_error": float(np.percentile(tcp_rel, 95)),
        "region_gradient_finite_rate": float(np.mean(finite)),
        "ad_fd_n": int(len(ids)),
    }


def _geometry_audit(source: dict[str, np.ndarray], *, seed: int = 29) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    *_, SelfCollisionFilter, build_locked_rail_model = reachability_modules()
    locked = build_locked_rail_model()
    collision_filter, _, _ = _probe_collision_filter(
        GpuPoseGtConfig(), locked, SelfCollisionFilter
    )
    positive = np.flatnonzero((source["reachable"] > 0.5) & np.any(source["q_best"] != 0.0, axis=1))
    ids = rng.choice(positive, size=min(10_000, len(positive)), replace=False)
    q_np = source["q_best"][ids].astype(np.float64)
    free = collision_filter.free_mask(q_np)
    kin = TorchRM75Kinematics.from_locked_model(locked, device="cuda")
    q = torch.as_tensor(source["q_best"][ids], device=kin.device)
    p, R = kin.fk(q)
    target = torch.as_tensor(source["features"][ids], device=kin.device)
    Rt = rotation_from_6d_torch(target[:, 3:9])
    pos_error = torch.linalg.vector_norm(p - target[:, :3], dim=-1)
    rot_error = torch.linalg.vector_norm(so3_log(Rt @ R.transpose(-1, -2)), dim=-1)

    # Empirical check of the RM4D last-axis symmetry under the probe collision model.
    n_roll, n_angle = 512, 12
    span = locked.q_upper - locked.q_lower
    Q = locked.q_lower + rng.random((n_roll, 7)) * span
    angles = np.linspace(locked.q_lower[-1], locked.q_upper[-1], n_angle, endpoint=False)
    sweep = np.repeat(Q[:, None, :], n_angle, axis=1)
    sweep[:, :, -1] = angles
    roll_free = collision_filter.free_mask(sweep.reshape(-1, 7)).reshape(n_roll, n_angle)
    varied = np.any(roll_free != roll_free[:, :1], axis=1)
    return {
        "positive_collision_audit_n": int(len(ids)),
        "positive_collision_failures": int((~free).sum()),
        "positive_pose_tolerance_failures": int(((pos_error > 2.0e-4) | (rot_error > 1.0e-3)).sum().item()),
        "positive_fk_position_max_m": float(pos_error.max().item()),
        "positive_fk_rotation_max_rad": float(rot_error.max().item()),
        "roll_collision_variation_rate": float(varied.mean()),
        "roll_collision_audit_n": int(n_roll),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/rm4d_signed_production.yaml"))
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--region-pairs", type=int, default=512)
    ap.add_argument("--allow-stale-checkpoint", action="store_true")
    ap.add_argument(
        "--report-near-axis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Report accuracy restricted to flange chart r < 5 cm (default: on).",
    )
    args = ap.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    cfg_path = args.config if args.config.is_absolute() else root / args.config
    cfg = load_signed_train_config(cfg_path, root=root)
    raw_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    robot_spec_path = raw_cfg.get("build", {}).get("robot_spec")
    if robot_spec_path is not None and not Path(robot_spec_path).is_absolute():
        robot_spec_path = root / robot_spec_path
    robot_spec = load_robot_model_spec(robot_spec_path)
    source_path = Path(raw_cfg["build"]["source_npz"])
    if not source_path.is_absolute():
        source_path = root / source_path
    ckpt = args.checkpoint or Path(cfg.checkpoint)
    if not ckpt.is_absolute():
        ckpt = root / ckpt
    field = ReachabilitySDF.load(
        ckpt,
        expected_robot=robot_spec,
        allow_stale=args.allow_stale_checkpoint,
    )
    T_axis_root = torch.as_tensor(
        robot_spec.root_to_j1_axis(), dtype=torch.float32, device=field.device
    )
    gt_npz = np.load(cfg.gt_npz, allow_pickle=False)
    arrays = {k: gt_npz[k] for k in gt_npz.files if k != "meta_json"}
    _, val_idx = _split_indices(
        arrays["boundary_id"],
        cfg.val_fraction,
        cfg.seed,
        arrays["source_pose_id"],
    )
    metrics = evaluate_signed_field(
        field,
        arrays,
        val_idx,
        report_near_axis=bool(args.report_near_axis),
        near_axis_r_m=cfg.near_axis_r_m,
    )
    source_npz = np.load(source_path, allow_pickle=False)
    source = {k: source_npz[k] for k in source_npz.files}
    val_groups = np.unique(arrays["boundary_id"][val_idx][arrays["boundary_id"][val_idx] >= 0])
    region = RegionA(RegionAConfig(samples=64, seed=17)).to(field.device)
    metrics["region_direction_agreement_m"] = _region_pair_direction(
        field, region, source, val_groups, 0, "boundary_signed_m", args.region_pairs, 10, T_axis_root
    )
    metrics["region_direction_agreement_deg"] = _region_pair_direction(
        field, region, source, val_groups, 1, "boundary_signed_rot_deg", args.region_pairs, 11, T_axis_root
    )
    source_interior = source["features"][(source["boundary_id"] < 0) & (source["reachable"] > 0.5)]
    metrics.update(_ad_fd_audit(field, region, source_interior, T_axis_root=T_axis_root))
    metrics.update(_geometry_audit(source))
    metrics["pass"] = bool(
        metrics["balanced_accuracy"] >= 0.90
        and metrics["direction_agreement_m"] >= 0.95
        and metrics["direction_agreement_deg"] >= 0.95
        and metrics["crossing_p95_m"] <= 0.001
        and metrics["crossing_p95_deg"] <= 1.0
        and metrics["region_direction_agreement_m"] >= 0.90
        and metrics["region_direction_agreement_deg"] >= 0.90
        and metrics["rail_ad_fd_median_relative_error"] <= 0.05
        and metrics["tcp_x_ad_fd_median_relative_error"] <= 0.05
        and metrics["region_gradient_finite_rate"] == 1.0
        and metrics["positive_collision_failures"] == 0
        and metrics["positive_pose_tolerance_failures"] == 0
        and metrics["roll_collision_variation_rate"] <= 0.01
    )
    report = root / "data/reports/eval_rm4d_signed.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"report -> {report}")
    return 0 if metrics["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
