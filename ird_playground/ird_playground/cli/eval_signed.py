"""Acceptance evaluation for the RM4D signed IRD and Region A query."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from ird_playground.ird.canonical import rotation_from_6d_torch
from ird_playground.ird.gpu_pose_gt import GpuPoseGtConfig, _probe_collision_filter
from ird_playground.ird.gt_common import reachability_modules
from ird_playground.ird.robot_model import RobotModelSpec, load_robot_model_spec
from ird_playground.ird.tool_frame import pose_flange_to_tcp, pose_tcp_to_flange
from ird_playground.ird.torch_kinematics import TorchRM75Kinematics, so3_log
from ird_playground.calib import false_acceptance_report, load_conformal_json
from ird_playground.ird.splits import FiveWaySplitConfig, five_way_split_indices
from ird_playground.neural.signed_field import ReachabilitySDF
from ird_playground.neural.train_signed import (
    _side_indices,
    evaluate_signed_field,
    load_signed_train_config,
)
from ird_playground.region.operator import RegionA, RegionAConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pose_matrices(features: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    R = rotation_from_6d_torch(x[:, 3:9])
    bottom = torch.zeros(len(x), 1, 4, dtype=x.dtype, device=device)
    bottom[:, 0, 3] = 1.0
    return torch.cat((torch.cat((R, x[:, :3, None]), dim=-1), bottom), dim=-2)


def _tcp_pose_matrices(
    features: np.ndarray, device: torch.device, T_flange_tcp: torch.Tensor
) -> torch.Tensor:
    """Map flange SE(3) features to TCP poses for world-path scoring."""
    T_flange = _pose_matrices(features, device)
    p, R = pose_flange_to_tcp(
        T_flange[..., :3, 3], T_flange[..., :3, :3], T_flange_tcp
    )
    bottom = torch.zeros(len(T_flange), 1, 4, dtype=T_flange.dtype, device=device)
    bottom[:, 0, 3] = 1.0
    return torch.cat((torch.cat((R, p[..., None]), dim=-1), bottom), dim=-2)


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
    tool = field.model.T_flange_tcp
    Tp = _tcp_pose_matrices(source["features"][ip[pick]], field.device, tool)
    Tn = _tcp_pose_matrices(source["features"][inn[pick]], field.device, tool)
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
    tool = field.model.T_flange_tcp
    for i in ids:
        base_pose = _tcp_pose_matrices(features[i : i + 1], field.device, tool)[0]
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


def _geometry_audit(
    source: dict[str, np.ndarray],
    robot_spec: RobotModelSpec,
    *,
    seed: int = 29,
) -> dict[str, float]:
    """FK round-trip against stored ``features`` (flange SE3 in rail_base)."""
    rng = np.random.default_rng(seed)
    *_, SelfCollisionFilter, build_locked_rail_model = reachability_modules()
    locked = build_locked_rail_model(
        robot_spec.kinematics_urdf,
        rail_locked_at_m=robot_spec.rail_locked_at_m,
        tcp_frame=robot_spec.tcp_frame,
    )
    collision_filter, _, _ = _probe_collision_filter(
        GpuPoseGtConfig(), locked, SelfCollisionFilter
    )
    positive = np.flatnonzero((source["reachable"] > 0.5) & np.any(source["q_best"] != 0.0, axis=1))
    ids = rng.choice(positive, size=min(10_000, len(positive)), replace=False)
    q_np = source["q_best"][ids].astype(np.float64)
    free = collision_filter.free_mask(q_np)
    kin = TorchRM75Kinematics.from_locked_model(locked, device="cuda")
    q = torch.as_tensor(source["q_best"][ids], device=kin.device)
    p_tcp, R_tcp = kin.fk(q)
    tool = torch.as_tensor(
        robot_spec.tool_frame().T_flange_tcp, dtype=p_tcp.dtype, device=p_tcp.device
    )
    p_fl, R_fl = pose_tcp_to_flange(p_tcp, R_tcp, tool)
    target = torch.as_tensor(source["features"][ids], device=kin.device)
    Rt = rotation_from_6d_torch(target[:, 3:9])
    pos_error = torch.linalg.vector_norm(p_fl - target[:, :3], dim=-1)
    rot_error = torch.linalg.vector_norm(so3_log(Rt @ R_fl.transpose(-1, -2)), dim=-1)

    # Flange-roll / q7 collision dependence (gamma is a real chart DOF).
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
        "feature_frame": "flange",
        "roll_collision_variation_rate": float(varied.mean()),
        "roll_collision_audit_n": int(n_roll),
    }


def _exterior_grad_audit(
    field: ReachabilitySDF,
    canonical: np.ndarray,
    reachable: np.ndarray,
    *,
    n: int = 2048,
    seed: int = 19,
) -> dict[str, float]:
    """Murooka-style check: unreachable points should keep a usable ∇f."""
    rng = np.random.default_rng(seed)
    neg = np.flatnonzero(reachable < 0.5)
    if len(neg) == 0:
        return {
            "exterior_grad_n": 0,
            "exterior_grad_median_norm": float("nan"),
            "exterior_grad_p10_norm": float("nan"),
            "exterior_score_median": float("nan"),
            "exterior_grad_finite_rate": 1.0,
        }
    ids = rng.choice(neg, size=min(n, len(neg)), replace=False)
    x = torch.as_tensor(canonical[ids], dtype=torch.float32, device=field.device)
    xn = field.model.normalize(x).detach().requires_grad_(True)
    pred = field.model.forward_normalized(xn)
    grad = torch.autograd.grad(pred.sum(), xn, create_graph=False)[0]
    norms = torch.linalg.vector_norm(grad, dim=-1)
    finite = torch.isfinite(norms) & torch.isfinite(pred)
    norms_np = norms.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    return {
        "exterior_grad_n": int(len(ids)),
        "exterior_grad_median_norm": float(np.median(norms_np)),
        "exterior_grad_p10_norm": float(np.percentile(norms_np, 10)),
        "exterior_score_median": float(np.median(pred_np)),
        "exterior_grad_finite_rate": float(finite.float().mean().item()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/rm4d_signed_production.yaml"))
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--region-pairs", type=int, default=512)
    ap.add_argument("--allow-stale-checkpoint", action="store_true")
    ap.add_argument(
        "--conformal",
        type=Path,
        default=None,
        help="Optional conformal JSON; if missing, fit on a train subset and write default path.",
    )
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
    splits = five_way_split_indices(
        arrays["boundary_id"],
        arrays["source_pose_id"],
        seed=cfg.seed,
        config=FiveWaySplitConfig(
            cfg.val_fraction,
            cfg.zero_calibration_fraction,
            cfg.safety_calibration_fraction,
            cfg.test_fraction,
        ),
    )
    val_idx = splits["test"]
    conformal_path = args.conformal
    if conformal_path is None:
        conformal_path = root / "data/calib/conformal_rm4d_signed.json"
    elif not conformal_path.is_absolute():
        conformal_path = root / conformal_path
    if not conformal_path.is_file():
        raise FileNotFoundError(
            f"independent calibration is required before final evaluation: {conformal_path}"
        )
    conf = load_conformal_json(conformal_path)
    if conf.get("checkpoint_sha256") != _sha256(ckpt):
        raise ValueError("calibration/checkpoint fingerprint mismatch")
    if conf.get("dataset_sha256") != _sha256(Path(cfg.gt_npz)):
        raise ValueError("calibration/dataset fingerprint mismatch")
    zero_bias = float(conf["zero_bias"])
    safety_threshold = float(conf["safety_threshold"])
    metrics = evaluate_signed_field(
        field,
        arrays,
        val_idx,
        report_near_axis=bool(args.report_near_axis),
        near_axis_r_m=cfg.near_axis_r_m,
        zero_bias=zero_bias,
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
    metrics.update(_geometry_audit(source, robot_spec))
    cls_w = arrays.get(
        "classification_weight", np.ones(len(arrays["reachable"]), dtype=np.float32)
    )
    metrics.update(
        _exterior_grad_audit(
            field,
            arrays["canonical"][val_idx],
            arrays["reachable"][val_idx],
        )
    )
    va = val_idx[cls_w[val_idx] > 0]
    safety_report = false_acceptance_report(
        field.score_np(arrays["canonical"][va]) - zero_bias,
        arrays["reachable"][va] > 0.5,
        safety_threshold,
    )
    metrics["zero_bias"] = zero_bias
    metrics["safety_threshold"] = safety_threshold
    metrics["m_safe"] = zero_bias + safety_threshold
    metrics["calibration_path"] = str(conformal_path)
    metrics.update({f"safety_{key}": value for key, value in safety_report.items()})
    metrics["pass"] = bool(
        metrics["balanced_accuracy"] >= 0.8988
        and metrics["direction_agreement_m"] >= 0.99
        and metrics["direction_agreement_deg"] >= 0.99
        and metrics["crossing_p95_m"] <= 0.001
        and metrics["crossing_p95_deg"] <= 0.2
        and metrics["strict_straddle_rate_m"] >= 0.339
        and metrics["strict_straddle_rate_deg"] >= 0.203
        and metrics["wide_straddle_rate_m"] >= 0.887
        and metrics["wide_straddle_rate_deg"] >= 0.684
        and metrics["region_direction_agreement_m"] >= 0.90
        and metrics["region_direction_agreement_deg"] >= 0.90
        and metrics["rail_ad_fd_median_relative_error"] <= 0.05
        and metrics["tcp_x_ad_fd_median_relative_error"] <= 0.05
        and metrics["region_gradient_finite_rate"] == 1.0
        and metrics["exterior_grad_finite_rate"] == 1.0
        and metrics["exterior_grad_p10_norm"] > 1.0e-4
        and metrics["positive_collision_failures"] == 0
        and metrics["positive_pose_tolerance_failures"] == 0
        and metrics["roll_collision_variation_rate"] <= 0.05
        and metrics["safety_false_accept_rate_upper"] <= float(conf["alpha"])
    )
    report = root / "data/reports/eval_rm4d_signed.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"report -> {report}")
    return 0 if metrics["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
