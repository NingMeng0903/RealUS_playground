#!/usr/bin/env python3
"""Phase 0 read-only audit: quantify RM4D TCP-chart defects vs flange chart.

Writes ONLY:
  ird_playground/data/audit/phase0_report.json
  ird_playground/data/audit/phase0_report.md

Does not modify any existing module or regenerate GT under data/ird/.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Environment bootstrap (must precede heavy rm75_control imports)
# ---------------------------------------------------------------------------
IRD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = IRD_ROOT.parent
if str(IRD_ROOT) not in sys.path:
    sys.path.insert(0, str(IRD_ROOT))


def _bootstrap_rm75() -> None:
    """Stub Robotic_Arm-gated package imports the same way GT builders do."""
    from ird_playground.ird.gt_common import reachability_modules

    reachability_modules()


_bootstrap_rm75()

import torch
from scipy.spatial.transform import Rotation

from ird_playground.ird.canonical import (
    canonical_flange_from_se3_features_torch,
    canonical_flange_invariants_torch,
    canonical_from_se3_features_torch,
    canonical_invariants_torch,
    pose_in_axis_frame_torch,
    rotation_from_6d_torch,
)
from ird_playground.ird.gt_common import reachability_modules
from ird_playground.map.build_flange_tensor import (
    FlangeOccupancyConfig,
    chart_coords_to_indices,
    flange_pose_to_chart,
)
from ird_playground.ird.robot_model import load_robot_model_spec
from ird_playground.ird.srs_label import (
    SrsLabelConfig,
    branch_and_psi_from_q7,
    srs_reachable_batch,
)
from ird_playground.ird.tool_frame import pose_tcp_to_flange
from ird_playground.ird.torch_kinematics import (
    TorchRM75Kinematics,
    collision_free_mask,
    select_collision_free_ik,
    so3_exp,
)

# Rail_base → SRS analytic frame: arm mount sits at y = -0.4 m.
SRS_XY_SHIFT = np.array([0.0, -0.4, 0.0], dtype=np.float64)

TCP_CHART_NAMES = ("p_z", "u_z", "r", "p_dot_u", "p_cross_u")
FLANGE_CHART_NAMES = (
    "p_z",
    "r",
    "ux_z",
    "p_dot_ux",
    "p_cross_ux",
    "uz_z",
    "p_dot_uz",
    "p_cross_uz",
)


@dataclass
class AuditConfig:
    seed: int = 20260729
    device: str = "cuda"
    # Item 1
    conflict_npz: tuple[str, ...] = (
        "data/ird/gpu_pose_production.npz",
        "data/ird/gpu_pose_stencils_production.npz",
    )
    conflict_max_samples: int = 200_000
    conflict_bin_fine: tuple[float, ...] = (0.03, 0.05, 0.03, 0.03, 0.03)
    conflict_bin_coarse: tuple[float, ...] = (0.05, 0.10, 0.05, 0.05, 0.05)
    flange_bin_fine: tuple[float, ...] = (
        0.03,
        0.03,
        0.05,
        0.03,
        0.03,
        0.05,
        0.03,
        0.03,
    )
    flange_bin_coarse: tuple[float, ...] = (
        0.05,
        0.05,
        0.10,
        0.05,
        0.05,
        0.10,
        0.05,
        0.05,
    )
    # Item 2
    q7_n_configs: int = 64
    q7_n_steps: int = 180
    # Item 3
    q1_n_samples: int = 8_000
    # Cylinder demo scanning workspace (TCP, rail_base frame)
    scan_center_x_m: float = 0.30
    scan_center_z_m: float = 0.10
    scan_radius_m: float = 0.09
    scan_y_pad_m: float = 0.05
    scan_path_y_min_m: float = -0.22
    scan_path_y_max_m: float = 0.22
    scan_radial_pad_m: float = 0.04
    # Item 4
    gamma_n_configs: int = 128
    gamma_n_steps: int = 360
    # Item 5
    fk_curve_counts: tuple[int, ...] = (1_000_000, 10_000_000, 100_000_000)
    fk_batch_size: int = 65_536
    fk_max_wall_s: float = 900.0
    # Item 6
    dls_srs_n_poses: int = 1_500
    dls_n_seeds: int = 12
    # Item 7
    base_n_configs: int = 256
    base_n_phi: int = 72


def _seed_everything(seed: int) -> np.random.Generator:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return rng


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if torch.is_tensor(obj):
        return _jsonable(obj.detach().cpu().tolist())
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        if math.isnan(x) or math.isinf(x):
            return str(x)
        return x
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    return obj


def _load_robot(device: str):
    spec = load_robot_model_spec()
    (
        _a,
        _b,
        _c,
        _d,
        _e,
        SelfCollisionFilter,
        build_locked_rail_model,
    ) = reachability_modules()
    lm = build_locked_rail_model(
        spec.kinematics_urdf,
        rail_locked_at_m=spec.rail_locked_at_m,
        tcp_frame=spec.tcp_frame,
    )
    collision = SelfCollisionFilter(
        kin_urdf=lm.urdf_path,
        collision_urdf=spec.collision_urdf,
        pair_config=spec.collision_pairs,
        rail_locked_at_m=lm.rail_locked_at_m,
        security_margin=spec.collision_security_margin_m,
    )
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    kin = TorchRM75Kinematics.from_locked_model(lm, device=device)
    return spec, lm, collision, kin


def _sample_collision_free(
    lm,
    collision,
    rng: np.random.Generator,
    n: int,
    *,
    inset: float = 0.02,
    draw: int = 8192,
) -> np.ndarray:
    span = lm.q_upper - lm.q_lower
    lo = lm.q_lower + inset * span
    hi = lm.q_upper - inset * span
    chunks: list[np.ndarray] = []
    while sum(len(c) for c in chunks) < n:
        Q = rng.uniform(lo, hi, size=(draw, 7))
        free = collision.free_mask(Q)
        if free.any():
            chunks.append(Q[free])
    return np.concatenate(chunks, axis=0)[:n].astype(np.float64)


def _quantize_conflict(
    chart: np.ndarray,
    labels: np.ndarray,
    bins: np.ndarray,
) -> dict[str, Any]:
    keys = np.floor(chart / bins[None, :]).astype(np.int64)
    # Pack into a single int64 hash (bins are coarse enough for no overflow).
    # Use a polynomial rolling hash.
    h = np.zeros(len(keys), dtype=np.int64)
    for d in range(keys.shape[1]):
        h = h * 1_000_003 + keys[:, d]
    bucket_reach: dict[int, set[int]] = defaultdict(set)
    bucket_count: dict[int, int] = defaultdict(int)
    for i, key in enumerate(h.tolist()):
        bucket_reach[key].add(int(labels[i] > 0.5))
        bucket_count[key] += 1
    non_singleton = {k: v for k, v in bucket_count.items() if v >= 2}
    conflicted = {k for k, labs in bucket_reach.items() if k in non_singleton and labs == {0, 1}}
    n = int(len(labels))
    n_ns = int(len(non_singleton))
    n_conf_buckets = int(len(conflicted))
    n_in_conflict = int(sum(bucket_count[k] for k in conflicted))
    return {
        "n_samples": n,
        "n_buckets": int(len(bucket_count)),
        "n_non_singleton_buckets": n_ns,
        "n_conflicted_buckets": n_conf_buckets,
        "fraction_non_singleton_conflicted": (
            float(n_conf_buckets / n_ns) if n_ns else float("nan")
        ),
        "fraction_samples_in_conflicted_buckets": float(n_in_conflict / max(n, 1)),
        "bin_widths": bins.tolist(),
    }


def audit_label_conflicts(cfg: AuditConfig, spec, device: str) -> dict[str, Any]:
    T_axis = torch.as_tensor(spec.root_to_j1_axis(), dtype=torch.float32, device=device)
    T_ft = torch.as_tensor(spec.tool_frame().T_flange_tcp, dtype=torch.float32, device=device)
    results: dict[str, Any] = {"files": [], "by_file": {}}
    for rel in cfg.conflict_npz:
        path = IRD_ROOT / rel
        if not path.is_file():
            results["by_file"][rel] = {"error": f"missing file {path}"}
            continue
        blob = np.load(path, allow_pickle=True)
        keys = list(blob.keys())
        has_robot_contract = "robot_contract" in keys
        manifest: dict[str, Any] = {}
        yaml_side = path.with_suffix(".yaml")
        if yaml_side.is_file():
            import yaml

            manifest = yaml.safe_load(yaml_side.read_text()) or {}
        meta_json = None
        if "meta_json" in keys:
            raw = blob["meta_json"]
            meta_json = json.loads(str(raw.item() if raw.shape == () else raw))
            has_robot_contract = has_robot_contract or ("robot_contract" in meta_json)
        collision_urdf = None
        if manifest:
            collision_urdf = manifest.get("collision_urdf") or (manifest.get("config") or {}).get(
                "collision_urdf"
            )
        if meta_json:
            collision_urdf = collision_urdf or meta_json.get("collision_urdf")
        if "features" not in keys or "reachable" not in keys:
            results["by_file"][rel] = {
                "error": "missing features/reachable",
                "keys": keys,
                "has_robot_contract": has_robot_contract,
                "collision_urdf": collision_urdf,
            }
            continue
        feat = np.asarray(blob["features"], dtype=np.float32)
        lab = np.asarray(blob["reachable"], dtype=np.float32)
        n_take = min(cfg.conflict_max_samples, len(feat))
        # Deterministic subsample
        rng = np.random.default_rng(cfg.seed + 11)
        if n_take < len(feat):
            idx = rng.choice(len(feat), size=n_take, replace=False)
            idx.sort()
            feat = feat[idx]
            lab = lab[idx]
        with torch.no_grad():
            t = torch.as_tensor(feat, device=device)
            tcp_chart = canonical_from_se3_features_torch(t, T_axis).cpu().numpy()
            fl_chart = (
                canonical_flange_from_se3_features_torch(t, T_ft, T_axis).cpu().numpy()
            )
        file_out: dict[str, Any] = {
            "path": str(path),
            "n_file": int(len(blob["features"])),
            "n_used": int(n_take),
            "seed": int(cfg.seed + 11),
            "has_robot_contract": bool(has_robot_contract),
            "collision_urdf": collision_urdf,
            "stale_horizontal_probe": bool(
                collision_urdf is not None and "horizontal_probe" in str(collision_urdf)
            ),
            "tcp_chart": {},
            "flange_chart": {},
        }
        for name, bins in (
            ("fine", np.asarray(cfg.conflict_bin_fine, dtype=np.float64)),
            ("coarse", np.asarray(cfg.conflict_bin_coarse, dtype=np.float64)),
        ):
            file_out["tcp_chart"][name] = _quantize_conflict(tcp_chart, lab, bins)
        for name, bins in (
            ("fine", np.asarray(cfg.flange_bin_fine, dtype=np.float64)),
            ("coarse", np.asarray(cfg.flange_bin_coarse, dtype=np.float64)),
        ):
            file_out["flange_chart"][name] = _quantize_conflict(fl_chart, lab, bins)
        # Flag if flange is not dramatically lower
        tcp_f = file_out["tcp_chart"]["fine"]["fraction_samples_in_conflicted_buckets"]
        fl_f = file_out["flange_chart"]["fine"]["fraction_samples_in_conflicted_buckets"]
        file_out["flag_flange_not_much_lower"] = bool(fl_f > 0.5 * tcp_f and fl_f > 0.01)
        results["files"].append(rel)
        results["by_file"][rel] = file_out
    return results


def audit_q7_invariance(cfg: AuditConfig, spec, lm, collision, kin, rng) -> dict[str, Any]:
    Q = _sample_collision_free(lm, collision, rng, cfg.q7_n_configs)
    T_axis = torch.as_tensor(spec.root_to_j1_axis(), dtype=torch.float32, device=kin.device)
    T_ft = torch.as_tensor(spec.tool_frame().T_flange_tcp, dtype=torch.float32, device=kin.device)
    q7_grid = torch.linspace(
        float(lm.q_lower[6]),
        float(lm.q_upper[6]),
        cfg.q7_n_steps,
        device=kin.device,
        dtype=torch.float32,
    )
    tcp_ranges = []
    fl_ranges = []
    with torch.no_grad():
        for q in Q:
            base = torch.as_tensor(q, device=kin.device, dtype=torch.float32)
            qq = base.expand(cfg.q7_n_steps, 7).clone()
            qq[:, 6] = q7_grid
            p, R = kin.fk(qq)
            p_a, R_a = pose_in_axis_frame_torch(p, R, T_axis)
            tcp = canonical_invariants_torch(p_a, R_a)
            p_f, R_f = pose_tcp_to_flange(p_a, R_a, T_ft)
            fl = canonical_flange_invariants_torch(p_f, R_f)
            tcp_ranges.append((tcp.max(0).values - tcp.min(0).values).cpu().numpy())
            fl_ranges.append((fl.max(0).values - fl.min(0).values).cpu().numpy())
    tcp_ranges = np.asarray(tcp_ranges, dtype=np.float64)
    fl_ranges = np.asarray(fl_ranges, dtype=np.float64)
    tcp_max = tcp_ranges.max(axis=0)
    tcp_mean = tcp_ranges.mean(axis=0)
    fl_max = fl_ranges.max(axis=0)
    fl_mean = fl_ranges.mean(axis=0)
    # Prior expectation references
    prior = {"tcp_u_z": 1.2303, "tcp_p_dot_u": 0.5169, "flange_corresponding": 0.0}
    return {
        "n_configs": int(cfg.q7_n_configs),
        "n_q7_steps": int(cfg.q7_n_steps),
        "seed": int(cfg.seed),
        "tcp_chart_names": list(TCP_CHART_NAMES),
        "flange_chart_names": list(FLANGE_CHART_NAMES),
        "tcp_max_abs_variation": tcp_max.tolist(),
        "tcp_mean_abs_variation": tcp_mean.tolist(),
        "flange_max_abs_variation": fl_max.tolist(),
        "flange_mean_abs_variation": fl_mean.tolist(),
        "tcp_u_z_max": float(tcp_max[1]),
        "tcp_p_dot_u_max": float(tcp_max[3]),
        "flange_uz_related_max": {
            "uz_z": float(fl_max[5]),
            "p_dot_uz": float(fl_max[6]),
            "p_cross_uz": float(fl_max[7]),
            "p_z": float(fl_max[0]),
            "r": float(fl_max[1]),
        },
        "prior_expectation": prior,
        "confirms_prior_tcp_u_z_within_20pct": bool(
            abs(float(tcp_max[1]) - prior["tcp_u_z"]) / prior["tcp_u_z"] < 0.20
            or abs(float(tcp_mean[1]) - prior["tcp_u_z"]) / prior["tcp_u_z"] < 0.20
        ),
        "note": (
            "Flange ux_* components encode gamma and are expected to vary under q7; "
            "q7-invariance is uz-related + (p_z, r) only."
        ),
    }


def audit_q1_realizability(cfg: AuditConfig, spec, lm, collision, kin, rng) -> dict[str, Any]:
    """Fraction of scanning-workspace poses whose yaw representative needs illegal q1."""
    import xml.etree.ElementTree as ET

    joint = None
    for node in ET.parse(spec.kinematics_urdf).getroot().findall("joint"):
        if node.get("name") == spec.j1_joint:
            joint = node
            break
    lim = joint.find("limit")
    q1_lo = float(lim.get("lower"))
    q1_hi = float(lim.get("upper"))
    gap_rad = (2.0 * math.pi) - (q1_hi - q1_lo)
    gap_deg = math.degrees(gap_rad)

    # Rejection-sample FK configs whose TCP lies in the cylinder scanning tube.
    y0 = cfg.scan_path_y_min_m - cfg.scan_y_pad_m
    y1 = cfg.scan_path_y_max_m + cfg.scan_y_pad_m
    cx = cfg.scan_center_x_m
    cz = cfg.scan_center_z_m
    radius_m = cfg.scan_radius_m + cfg.scan_radial_pad_m
    collected: list[np.ndarray] = []
    attempts = 0
    max_attempts = cfg.q1_n_samples * 80
    while len(collected) < cfg.q1_n_samples and attempts < max_attempts:
        batch = min(8192, max_attempts - attempts)
        Q = _sample_collision_free(lm, collision, rng, batch)
        attempts += batch
        q_t = torch.as_tensor(Q, device=kin.device, dtype=torch.float32)
        with torch.no_grad():
            p_tcp, _R_tcp = kin.fk(q_t)
            p_np = p_tcp.cpu().numpy()
        radial = np.hypot(p_np[:, 0] - cx, p_np[:, 2] - cz)
        mask = (p_np[:, 1] >= y0) & (p_np[:, 1] <= y1) & (radial <= radius_m)
        if mask.any():
            collected.append(Q[mask])
    if not collected:
        return {
            "error": "no collision-free FK samples landed in scanning workspace",
            "workspace": {
                "cylinder_center_xz_m": [cx, cz],
                "radius_m": float(radius_m),
                "y_range_m": [y0, y1],
                "source": "cylinder_region_ird_demo.DemoConfig + pad",
            },
            "q1_limits_rad": [q1_lo, q1_hi],
            "q1_gap_deg": gap_deg,
            "n_samples": 0,
            "seed": int(cfg.seed),
        }
    Q = np.concatenate(collected, axis=0)[: cfg.q1_n_samples]
    q_t = torch.as_tensor(Q, device=kin.device, dtype=torch.float32)
    T_axis = torch.as_tensor(spec.root_to_j1_axis(), dtype=torch.float32, device=kin.device)
    with torch.no_grad():
        p_tcp, R_tcp = kin.fk(q_t)
        p_a, _R_a = pose_in_axis_frame_torch(p_tcp, R_tcp, T_axis)
        p_a = p_a.cpu().numpy()
    alpha = np.arctan2(p_a[:, 1], p_a[:, 0])
    q1 = Q[:, 0]
    # Canonical representative: rotate about J1 so p_xy aligns with +X ⇒ q1_rep = q1 - alpha
    q1_rep = q1 - alpha
    # Wrap into (-pi, pi]
    q1_rep = (q1_rep + math.pi) % (2.0 * math.pi) - math.pi
    unrealizable = (q1_rep < q1_lo) | (q1_rep > q1_hi)
    frac = float(np.mean(unrealizable))
    geometric_prior = float(gap_rad / (2.0 * math.pi))
    return {
        "n_samples": int(len(Q)),
        "n_attempts_drawn": int(attempts),
        "seed": int(cfg.seed),
        "q1_limits_rad": [q1_lo, q1_hi],
        "q1_limits_deg": [math.degrees(q1_lo), math.degrees(q1_hi)],
        "q1_gap_deg": float(gap_deg),
        "geometric_unrealizable_prior": geometric_prior,
        "fraction_representatives_unrealizable": frac,
        "false_positive_contribution_estimate": frac,
        "workspace": {
            "cylinder_center_xz_m": [float(cx), float(cz)],
            "radius_m_with_pad": float(radius_m),
            "y_range_m": [float(y0), float(y1)],
            "source": "cylinder_region_ird_demo.DemoConfig + pad",
        },
        "recommend_q1_aux_head": bool(frac >= 0.02),
        "decision": (
            "add auxiliary q1-reachable-interval head"
            if frac >= 0.02
            else "document only; fraction < 2%"
        ),
    }


def _free_interval_widths_deg(free: np.ndarray, step_deg: float) -> list[float]:
    """Angular widths of contiguous free runs on a circular gamma grid."""
    n = len(free)
    if n == 0:
        return []
    # Unwrap circular: if both ends free, merge later
    widths: list[float] = []
    i = 0
    while i < n:
        if not free[i]:
            i += 1
            continue
        j = i
        while j < n and free[j]:
            j += 1
        widths.append((j - i) * step_deg)
        i = j
    if free[0] and free[-1] and len(widths) >= 2:
        # merge first and last
        merged = widths[0] + widths[-1]
        widths = [merged] + widths[1:-1]
    return widths


def _gamma_sweep_stats(
    Q: np.ndarray,
    lm,
    collision,
    n_steps: int,
) -> dict[str, Any]:
    step_deg = 360.0 / n_steps
    flip_flags: list[bool] = []
    all_widths: list[float] = []
    n_flip_configs = 0
    for q in Q:
        q7_local = q[6] + np.linspace(-math.pi, math.pi, n_steps, endpoint=False)
        q7_local = np.clip(q7_local, lm.q_lower[6], lm.q_upper[6])
        if np.ptp(q7_local) < math.radians(90.0):
            continue
        Qs = np.tile(q, (n_steps, 1))
        Qs[:, 6] = q7_local
        free = collision.free_mask(Qs)
        flips = int(np.count_nonzero(free[1:] != free[:-1]))
        if free[0] != free[-1]:
            flips += 1
        flip_flags.append(flips > 0)
        if flips > 0:
            n_flip_configs += 1
        all_widths.extend(_free_interval_widths_deg(free, step_deg))
    flip_rate = float(np.mean(flip_flags)) if flip_flags else float("nan")
    flip_widths: list[float] = []
    for q, did_flip in zip(Q[: len(flip_flags)], flip_flags):
        if not did_flip:
            continue
        q7_local = q[6] + np.linspace(-math.pi, math.pi, n_steps, endpoint=False)
        q7_local = np.clip(q7_local, lm.q_lower[6], lm.q_upper[6])
        Qs = np.tile(q, (n_steps, 1))
        Qs[:, 6] = q7_local
        free = collision.free_mask(Qs)
        flip_widths.extend(_free_interval_widths_deg(free, step_deg))
    width_src = np.asarray(flip_widths if flip_widths else all_widths, dtype=np.float64)
    stats: dict[str, Any] = {}
    recommend_deg = 12.0
    if width_src.size:
        stats = {
            "n_intervals": int(width_src.size),
            "min_deg": float(width_src.min()),
            "p10_deg": float(np.percentile(width_src, 10)),
            "p50_deg": float(np.percentile(width_src, 50)),
            "p90_deg": float(np.percentile(width_src, 90)),
            "mean_deg": float(width_src.mean()),
            "conditioned_on_flip_configs": bool(bool(flip_widths)),
        }
        recommend_deg = float(
            np.clip(max(1.0, math.floor(stats["p10_deg"] / 2.0)), 1.0, 12.0)
        )
    return {
        "n_configs": int(len(flip_flags)),
        "flip_rate": flip_rate,
        "n_configs_with_flip": int(n_flip_configs),
        "free_interval_width_deg": stats,
        "recommended_gamma_resolution_deg": recommend_deg,
        "gamma_is_load_bearing": bool(flip_rate == flip_rate and flip_rate >= 0.05),
    }


def audit_gamma_collision(cfg: AuditConfig, lm, collision, rng) -> dict[str, Any]:
    """Sweep q7 (= flange roll about uz) with q1..q6 fixed; measure collision flips.

    Uniform free configs rarely graze the probe, so we ALSO report a near-contact
    stratum: keep draws with ``min_distance < 5 cm`` (margin temporarily 0).
    """
    Q_uniform = _sample_collision_free(lm, collision, rng, cfg.gamma_n_configs)
    stats_uniform = _gamma_sweep_stats(Q_uniform, lm, collision, cfg.gamma_n_steps)

    saved_margin = float(collision.security_margin)
    collision.set_security_margin(0.0)
    near: list[np.ndarray] = []
    draws = 0
    while len(near) < cfg.gamma_n_configs and draws < cfg.gamma_n_configs * 40:
        batch = _sample_collision_free(lm, collision, rng, 512, inset=0.0)
        draws += len(batch)
        for q in batch:
            d = collision.min_distance(q)
            if 0.0 <= d < 0.05:
                near.append(q.copy())
                if len(near) >= cfg.gamma_n_configs:
                    break
    Q_near = np.asarray(near[: cfg.gamma_n_configs], dtype=np.float64) if near else np.zeros((0, 7))
    stats_near: dict[str, Any]
    if len(Q_near):
        stats_near = _gamma_sweep_stats(Q_near, lm, collision, cfg.gamma_n_steps)
    else:
        stats_near = {"error": "no near-contact samples", "n_configs": 0}
    collision.set_security_margin(saved_margin)

    # Prefer uniform stratum for the headline decision: near-contact clearance is
    # often q7-invariant (arm–arm), so it understates probe-roll dependence.
    primary = stats_uniform
    return {
        "n_gamma_steps": int(cfg.gamma_n_steps),
        "seed": int(cfg.seed),
        "method": (
            "proxy: sweep q7 with q1..q6 fixed (exact flange p + uz hold for serial wrist)"
        ),
        "security_margin_m_uniform_stratum": saved_margin,
        "security_margin_m_near_stratum": 0.0,
        "uniform_collision_free": stats_uniform,
        "near_contact_clearance_lt_5cm": stats_near,
        "flip_rate": primary.get("flip_rate"),
        "n_configs": primary.get("n_configs"),
        "free_interval_width_deg": primary.get("free_interval_width_deg"),
        "recommended_gamma_resolution_deg": primary.get(
            "recommended_gamma_resolution_deg", 12.0
        ),
        "gamma_is_load_bearing": bool(
            (primary.get("flip_rate") or 0.0) >= 0.01
        ),
        "decision_note": (
            "Headline flip_rate uses the uniform collision-free stratum. "
            "Near-contact (clearance<5cm) is reported separately; it is often dominated "
            "by q7-invariant arm–arm pairs and can understate probe-roll dependence. "
            "Gamma remains a required chart axis (not a symmetry)."
        ),
    }


def audit_fk_coverage(cfg: AuditConfig, spec, lm, collision, kin, rng) -> dict[str, Any]:
    occ_cfg = FlangeOccupancyConfig(
        step_m=0.03,
        step_deg=12.0,
        seed=cfg.seed,
        device=str(kin.device),
    )
    axes = occ_cfg.axis_arrays()
    shape = tuple(len(a) for a in axes)
    n_cells = int(np.prod(shape))
    occupancy = np.zeros(shape, dtype=np.uint8)
    T_axis = torch.as_tensor(spec.root_to_j1_axis(), dtype=torch.float32, device=kin.device)
    T_ft = torch.as_tensor(spec.tool_frame().T_flange_tcp, dtype=torch.float32, device=kin.device)
    q_lo = lm.q_lower
    q_hi = lm.q_upper
    targets = list(cfg.fk_curve_counts)
    curve = []
    n_accepted = 0
    n_drawn = 0
    t0 = time.time()
    extrapolated = False
    next_target_i = 0
    # Draw until last target or wall clock
    last_target = targets[-1]
    while n_drawn < last_target and (time.time() - t0) < cfg.fk_max_wall_s:
        batch = min(cfg.fk_batch_size, last_target - n_drawn)
        Q = rng.uniform(q_lo, q_hi, size=(batch, 7)).astype(np.float32)
        n_drawn += batch
        free = collision.free_mask(Q)
        Qf = Q[free]
        if len(Qf) == 0:
            continue
        q_t = torch.as_tensor(Qf, device=kin.device)
        with torch.no_grad():
            p, R = kin.fk(q_t)
            p_a, R_a = pose_in_axis_frame_torch(p, R, T_axis)
            p_f, R_f = pose_tcp_to_flange(p_a, R_a, T_ft)
            chart = flange_pose_to_chart(
                p_f.detach().cpu().numpy(),
                R_f.detach().cpu().numpy(),
            )
        idx = chart_coords_to_indices(chart, axes)
        occupancy[idx] = 1
        n_accepted += int(len(chart))
        while next_target_i < len(targets) and n_drawn >= targets[next_target_i]:
            pos = int(occupancy.sum())
            curve.append(
                {
                    "n_fk_draws": int(targets[next_target_i]),
                    "n_collision_free": int(n_accepted),
                    "n_occupied_cells": pos,
                    "occupied_fraction": float(pos / n_cells),
                    "measured": True,
                    "wall_s": float(time.time() - t0),
                }
            )
            next_target_i += 1
    # Extrapolate missing targets via unique-coupon heuristic:
    # occupied ≈ N_inf * (1 - exp(-n_free / N_eff))
    if next_target_i < len(targets) and len(curve) >= 2:
        extrapolated = True
        f1 = curve[0]["occupied_fraction"]
        n1 = max(curve[0]["n_collision_free"], 1)
        f2 = curve[-1]["occupied_fraction"]
        n2 = max(curve[-1]["n_collision_free"], 1)
        # Solve for N_inf, N_eff from two points (crude)
        # f = a (1 - exp(-n/b))
        # Use ratio: (1-f1/a)/(1-f2/a) = exp(-(n1-n2)/b) — iterate a
        best = None
        for a in np.linspace(max(f2 * 1.01, f2 + 1e-6), min(1.0, f2 * 5 + 0.01), 200):
            if f1 >= a or f2 >= a:
                continue
            # b = -n / log(1 - f/a)
            b1 = -n1 / math.log(1.0 - f1 / a)
            b2 = -n2 / math.log(1.0 - f2 / a)
            err = abs(b1 - b2) / max(b1, b2)
            if best is None or err < best[0]:
                best = (err, float(a), float(0.5 * (b1 + b2)))
        while next_target_i < len(targets):
            tgt = targets[next_target_i]
            # assume free rate from measured
            free_rate = n_accepted / max(n_drawn, 1)
            n_free_est = free_rate * tgt
            if best is not None:
                a, b = best[1], best[2]
                frac = a * (1.0 - math.exp(-n_free_est / b))
            else:
                # linear in log space fallback
                frac = float("nan")
            curve.append(
                {
                    "n_fk_draws": int(tgt),
                    "n_collision_free_estimated": float(n_free_est),
                    "occupied_fraction": float(frac) if frac == frac else None,
                    "measured": False,
                    "extrapolated": True,
                    "fit_err": None if best is None else float(best[0]),
                }
            )
            next_target_i += 1

    # Recommend sample count: first n where incremental occupied gain < 1% relative
    recommend = None
    for i in range(1, len(curve)):
        if not curve[i].get("measured"):
            break
        prev = curve[i - 1]["occupied_fraction"]
        cur = curve[i]["occupied_fraction"]
        if prev > 0 and (cur - prev) / prev < 0.01:
            recommend = curve[i]["n_fk_draws"]
            break
    if recommend is None:
        recommend = int(targets[min(1, len(targets) - 1)])

    return {
        "seed": int(cfg.seed),
        "grid_shape": list(shape),
        "n_cells": n_cells,
        "step_m": 0.03,
        "step_deg": 12.0,
        "n_drawn": int(n_drawn),
        "n_collision_free": int(n_accepted),
        "free_rate": float(n_accepted / max(n_drawn, 1)),
        "wall_s": float(time.time() - t0),
        "extrapolated": bool(extrapolated),
        "saturation_curve": curve,
        "recommended_fk_samples": int(recommend),
        "unknown_vs_unreachable_criterion": (
            "Mark a cell 'unknown' until it has been hit by ≥1 collision-free FK "
            "sample; after the recommended sample budget, remaining empty cells "
            "are treated as unreachable for occupancy, with a hold-out FK batch "
            "used to estimate the residual false-negative (missed reachable) rate. "
            "Do not equate empty with unreachable before the saturation knee."
        ),
        "bug_note_build_flange_tensor": (
            "ird_playground/map/build_flange_tensor.py SelfCollisionFilter call "
            "uses positional args / security_margin_m kwarg that do not match the "
            "keyword-only SelfCollisionFilter API — occupancy builder is currently broken."
        ),
    }


def audit_dls_vs_srs(cfg: AuditConfig, spec, lm, collision, kin, rng) -> dict[str, Any]:
    """Compare 12-seed DLS labels vs controller-aligned SRS on the same poses."""
    # Build near/far pose queries around collision-free FK seeds (same idea as gpu_pose_gt).
    n = cfg.dls_srs_n_poses
    Q_seed = _sample_collision_free(lm, collision, rng, max(n, 512))
    q_fk = torch.as_tensor(Q_seed, device=kin.device, dtype=torch.float32)
    with torch.no_grad():
        p_fk, R_fk = kin.fk(q_fk)
    src_idx = rng.integers(0, len(Q_seed), size=n)
    source_q = q_fk[torch.as_tensor(src_idx, device=kin.device)]
    source_p = p_fk[torch.as_tensor(src_idx, device=kin.device)]
    source_R = R_fk[torch.as_tensor(src_idx, device=kin.device)]
    near = rng.random(n) < 0.5
    pos_mag = np.where(
        near,
        rng.uniform(0.0, 0.04, size=n),
        rng.uniform(0.04, 0.25, size=n),
    )
    rot_mag = np.where(
        near,
        rng.uniform(0.0, 12.0, size=n),
        rng.uniform(10.0, 45.0, size=n),
    )
    mode = rng.random(n)
    pos_mag[mode > 0.70] = 0.0
    rot_mag[mode < 0.35] = 0.0
    dp = rng.normal(size=(n, 3))
    dp /= np.clip(np.linalg.norm(dp, axis=1, keepdims=True), 1e-12, None)
    dp *= pos_mag[:, None]
    dw = rng.normal(size=(n, 3))
    dw /= np.clip(np.linalg.norm(dw, axis=1, keepdims=True), 1e-12, None)
    dw *= np.deg2rad(rot_mag)[:, None]
    target_p = source_p + torch.as_tensor(dp, device=kin.device, dtype=torch.float32)
    dR = so3_exp(torch.as_tensor(dw, device=kin.device, dtype=torch.float32))
    target_R = dR @ source_R

    # DLS 12-seed
    q0 = torch.empty(n, cfg.dls_n_seeds, 7, device=kin.device, dtype=torch.float32)
    q0[:, 0] = source_q
    extra = rng.integers(0, len(Q_seed), size=(n, cfg.dls_n_seeds - 1))
    q0[:, 1:] = q_fk[torch.as_tensor(extra, device=kin.device)]
    t0 = time.time()
    dls = kin.ik_dls(
        target_p,
        target_R,
        q0,
        max_iter=100,
        tol_pos_m=2.0e-4,
        tol_rot_rad=1.0e-3,
    )
    dls_chk = select_collision_free_ik(
        dls, collision, tol_pos_m=2.0e-4, tol_rot_rad=1.0e-3
    )
    if kin.device.type == "cuda":
        torch.cuda.synchronize()
    dls_s = time.time() - t0
    dls_reach = dls_chk.reachable.detach().cpu().numpy().astype(bool)

    # SRS on flange (controller-correct for probe45) via srs_label API
    T_ft = torch.as_tensor(spec.tool_frame().T_flange_tcp, dtype=torch.float32, device=kin.device)
    with torch.no_grad():
        p_fl, R_fl = pose_tcp_to_flange(target_p, target_R, T_ft)
    p_srs = p_fl.detach().cpu().numpy() - SRS_XY_SHIFT
    R_srs = R_fl.detach().cpu().numpy()
    branch_ids = np.array(
        [branch_and_psi_from_q7(Q_seed[i])[0] for i in src_idx], dtype=np.int32
    )
    psi_homes = np.array(
        [branch_and_psi_from_q7(Q_seed[i])[1] for i in src_idx], dtype=np.float64
    )
    # tcp_offset (0,0,0) kept for manifest; flange tool_mode uses URDF T_flange_tcp.
    # branch_id on the config is a required placeholder; per-sample branch_ids override.
    srs_cfg = SrsLabelConfig(
        branch_id=int(branch_ids[0]),
        tcp_offset_xyz=(0.0, 0.0, 0.0),
        psi_home_rad=0.0,
    )
    t0 = time.time()
    srs_out = srs_reachable_batch(
        p_srs, R_srs, srs_cfg, branch_ids=branch_ids, psi_homes=psi_homes
    )
    srs_s = time.time() - t0
    srs_reach = srs_out["reachable"].astype(bool)

    # Optional: also measure literal srs_label on TCP (known broken for probe45
    # under coaxial assumptions; flange mode + TCP pose is still the wrong frame).
    p_tcp = target_p.detach().cpu().numpy() - SRS_XY_SHIFT
    R_tcp = target_R.detach().cpu().numpy()
    naive = srs_reachable_batch(
        p_tcp,
        R_tcp,
        SrsLabelConfig(branch_id=int(branch_ids[0])),
        branch_ids=branch_ids,
        psi_homes=psi_homes,
    )
    naive_reach = naive["reachable"].astype(bool)

    def _confusion(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
        # a = DLS, b = SRS
        tp = int(np.sum(a & b))
        tn = int(np.sum(~a & ~b))
        fp = int(np.sum(a & ~b))  # DLS yes, SRS no
        fn = int(np.sum(~a & b))
        return {
            "dls_yes_srs_yes": tp,
            "dls_no_srs_no": tn,
            "dls_yes_srs_no": fp,
            "dls_no_srs_yes": fn,
            "dls_reachable_rate": float(np.mean(a)),
            "srs_reachable_rate": float(np.mean(b)),
            "dls_yes_srs_no_rate": float(fp / max(len(a), 1)),
            "dls_yes_srs_no_rate_among_dls_yes": float(fp / max(int(a.sum()), 1)),
        }

    conf = _confusion(dls_reach, srs_reach)
    conf_naive = _confusion(dls_reach, naive_reach)

    # Region breakdown: near vs far by SE(3) perturbation magnitude
    near_mask = near
    far_mask = ~near
    by_region = {
        "near_perturbation": _confusion(dls_reach[near_mask], srs_reach[near_mask]),
        "far_perturbation": _confusion(dls_reach[far_mask], srs_reach[far_mask]),
        "n_near": int(near_mask.sum()),
        "n_far": int(far_mask.sum()),
    }
    fp_rate = conf["dls_yes_srs_no_rate"]
    if fp_rate < 0.03:
        priority = "low (~nicety relative to canonicalization)"
    elif fp_rate < 0.10:
        priority = "moderate"
    else:
        priority = "HIGH — larger than typical canonicalization conflict; prioritize SRS-aligned labels"
    return {
        "n_poses": int(n),
        "seed": int(cfg.seed),
        "dls": {
            "n_seeds": int(cfg.dls_n_seeds),
            "tol_pos_m": 2.0e-4,
            "tol_rot_rad": 1.0e-3,
            "wall_s": float(dls_s),
        },
        "srs": {
            "method": (
                "srs_label.srs_reachable_batch on FLANGE pose in SRS frame "
                "(rail_base minus [0,-0.4,0]), d_wt=D_WT_FLANGE, fixed branch from seed, "
                "psi_home from seed, 5° grid, 150° max swing"
            ),
            "d_wt_m": float(srs_cfg.d_wt()),
            "wall_s": float(srs_s),
            "note_probe45": (
                "Literal srs_label on angled TCP is invalid: srs_ik assumes "
                "W = p - L * R[:,2]. Audit uses flange pose + D_WT_FLANGE instead."
            ),
        },
        "confusion_flange_srs": conf,
        "confusion_naive_tcp_srs": conf_naive,
        "by_region": by_region,
        "priority": priority,
        "bugs": [
            {
                "file": "ird_playground/ird_playground/ird/srs_label.py",
                "lines": "48-51, 62-97",
                "issue": (
                    "Default tcp_offset_xyz is probe45 (y,z) offset and d_wt_from_tcp_offset "
                    "feeds srs_ik which still models W=p-L*R_z; angled TCP breaks round-trip."
                ),
            },
            {
                "file": "rm75_control/rm75_control/kinematics/srs_ik.py",
                "lines": "328-330",
                "issue": (
                    "Comment claims TCP lies on joint_7/flange Z; probe45 URDF violates this."
                ),
            },
        ],
    }


def audit_base_yaw_symmetry(cfg: AuditConfig, lm, collision, rng) -> dict[str, Any]:
    Q = _sample_collision_free(lm, collision, rng, cfg.base_n_configs)
    phis = np.linspace(-math.pi, math.pi, cfg.base_n_phi, endpoint=False)
    flip_config = []
    n_trials = 0
    n_flips = 0
    for q in Q:
        # baseline must be free
        base_free = not collision.in_collision(q)
        if not base_free:
            continue
        flipped = False
        for phi in phis:
            q2 = q.copy()
            q2[0] = q[0] + phi
            # skip if outside joint limits (not a symmetry violation)
            if q2[0] < lm.q_lower[0] or q2[0] > lm.q_upper[0]:
                continue
            n_trials += 1
            free = not collision.in_collision(q2)
            if free != base_free:
                n_flips += 1
                flipped = True
        flip_config.append(flipped)
    # Inspect collision URDF for world-fixed geometry beyond base_link
    from ird_playground.ird.robot_model import RobotModelSpec

    spec = load_robot_model_spec()
    env_bug = None
    try:
        spec.assert_base_yaw_invariant_collision_model()
    except ValueError as exc:
        env_bug = str(exc)
    flip_rate_trials = float(n_flips / max(n_trials, 1))
    flip_rate_configs = float(np.mean(flip_config)) if flip_config else float("nan")
    return {
        "n_configs": int(len(flip_config)),
        "n_phi": int(cfg.base_n_phi),
        "n_trials_within_q1_limits": int(n_trials),
        "seed": int(cfg.seed),
        "flip_rate_per_trial": flip_rate_trials,
        "fraction_configs_with_any_flip": flip_rate_configs,
        "security_margin_m": float(collision.security_margin),
        "recommend_base_link_cylinder_envelope": bool(flip_rate_trials >= 1e-3 or flip_rate_configs >= 0.01),
        "env_geometry_bug": env_bug,
        "decision": (
            "replace base_link collision with J1 swept cylinder envelope"
            if (flip_rate_trials >= 1e-3 or flip_rate_configs >= 0.01)
            else "residual small; document only"
        ),
    }


def _print_summary(report: dict[str, Any]) -> None:
    rows = []
    c = report.get("1_label_conflicts", {})
    for f, v in (c.get("by_file") or {}).items():
        if "tcp_chart" not in v:
            continue
        tcp = v["tcp_chart"]["fine"]["fraction_samples_in_conflicted_buckets"]
        fl = v["flange_chart"]["fine"]["fraction_samples_in_conflicted_buckets"]
        rows.append(
            (
                "1 conflict samples (fine)",
                f"TCP={tcp:.4f} flange={fl:.4f}",
                v["n_used"],
                v["seed"],
            )
        )
    q7 = report.get("2_q7_invariance", {})
    if q7:
        rows.append(
            (
                "2 TCP u_z / p·u max Δ",
                f"{q7.get('tcp_u_z_max')}/{q7.get('tcp_p_dot_u_max')}",
                q7.get("n_configs"),
                q7.get("seed"),
            )
        )
        uz = q7.get("flange_uz_related_max", {})
        rows.append(
            (
                "2 flange uz-related max Δ",
                f"uz_z={uz.get('uz_z')} p·uz={uz.get('p_dot_uz')}",
                q7.get("n_configs"),
                q7.get("seed"),
            )
        )
    q1 = report.get("3_q1_realizability", {})
    if q1 and "fraction_representatives_unrealizable" in q1:
        rows.append(
            (
                "3 q1 unrealizable frac",
                f"{q1['fraction_representatives_unrealizable']:.4f}",
                q1.get("n_samples"),
                q1.get("seed"),
            )
        )
    g = report.get("4_gamma_collision", {})
    if g:
        rows.append(
            (
                "4 gamma flip rate / res°",
                f"{g.get('flip_rate')}/{g.get('recommended_gamma_resolution_deg')}°",
                g.get("n_configs"),
                g.get("seed"),
            )
        )
        uni = g.get("uniform_collision_free") or {}
        if uni:
            rows.append(
                (
                    "4 gamma flip (uniform)",
                    f"{uni.get('flip_rate')}",
                    uni.get("n_configs"),
                    g.get("seed"),
                )
            )
    fk = report.get("5_fk_coverage", {})
    if fk:
        pts = ", ".join(
            f"{p['n_fk_draws']}:{(p.get('occupied_fraction') or float('nan')):.4f}"
            + ("*" if p.get("extrapolated") else "")
            for p in fk.get("saturation_curve", [])
        )
        rows.append(
            ("5 occ frac @ N", pts, fk.get("n_collision_free"), fk.get("seed"))
        )
    d = report.get("6_dls_vs_srs", {})
    if d:
        conf = d.get("confusion_flange_srs", {})
        rows.append(
            (
                "6 DLS✓ SRS✗ rate",
                f"{conf.get('dls_yes_srs_no_rate')}",
                d.get("n_poses"),
                d.get("seed"),
            )
        )
    b = report.get("7_base_yaw_symmetry", {})
    if b:
        rows.append(
            (
                "7 base_link flip/trial",
                f"{b.get('flip_rate_per_trial')}",
                b.get("n_configs"),
                b.get("seed"),
            )
        )
    print("\n=== Phase 0 audit summary ===")
    print(f"{'metric':<28} {'value':<48} {'n':>10} {'seed':>10}")
    print("-" * 100)
    for m, v, n, s in rows:
        print(f"{m:<28} {str(v):<48} {str(n):>10} {str(s):>10}")
    print()


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Phase 0 canonicalization audit",
        "",
        f"Seed (master): `{report.get('seed')}`",
        f"Device: `{report.get('device')}`",
        f"Generated: `{report.get('timestamp')}`",
        "",
        "## 1. Beta label conflict rate",
        "",
    ]
    c = report.get("1_label_conflicts", {})
    for f, v in (c.get("by_file") or {}).items():
        lines.append(f"### `{f}`")
        if "error" in v:
            lines.append(f"- ERROR: {v['error']}")
            continue
        lines.append(
            f"- n_used={v['n_used']} / n_file={v['n_file']}, seed={v['seed']}, "
            f"robot_contract={v['has_robot_contract']}, stale_horizontal_probe={v['stale_horizontal_probe']}"
        )
        lines.append(f"- collision_urdf: `{v.get('collision_urdf')}`")
        for chart in ("tcp_chart", "flange_chart"):
            lines.append(f"- **{chart}**")
            for res, stats in v[chart].items():
                lines.append(
                    f"  - {res} bins={stats['bin_widths']}: "
                    f"non-singleton conflicted bucket frac="
                    f"{stats['fraction_non_singleton_conflicted']:.6f}, "
                    f"sample frac in conflicted buckets="
                    f"{stats['fraction_samples_in_conflicted_buckets']:.6f} "
                    f"(n_ns={stats['n_non_singleton_buckets']}, "
                    f"n_conf={stats['n_conflicted_buckets']})"
                )
        if v.get("flag_flange_not_much_lower"):
            lines.append("- **FLAG: flange conflict rate not dramatically lower than TCP**")
        lines.append("")
    q7 = report.get("2_q7_invariance", {})
    lines += [
        "## 2. q7 invariance",
        "",
        f"- n_configs={q7.get('n_configs')}, n_steps={q7.get('n_q7_steps')}, seed={q7.get('seed')}",
        f"- TCP max variation: {q7.get('tcp_max_abs_variation')}",
        f"- TCP u_z max={q7.get('tcp_u_z_max')}, p·u max={q7.get('tcp_p_dot_u_max')} "
        f"(prior 1.2303 / 0.5169)",
        f"- Flange max variation: {q7.get('flange_max_abs_variation')}",
        f"- Flange uz-related max: {q7.get('flange_uz_related_max')}",
        f"- Note: {q7.get('note')}",
        "",
        "## 3. J1 representative realizability",
        "",
    ]
    q1 = report.get("3_q1_realizability", {})
    lines += [
        f"- {json.dumps(_jsonable(q1), indent=2)}",
        "",
        "## 4. Gamma collision dependence",
        "",
        f"- {json.dumps(_jsonable(report.get('4_gamma_collision', {})), indent=2)}",
        "",
        "## 5. FK coverage saturation",
        "",
        f"- {json.dumps(_jsonable(report.get('5_fk_coverage', {})), indent=2)}",
        "",
        "## 6. DLS vs SRS",
        "",
        f"- {json.dumps(_jsonable(report.get('6_dls_vs_srs', {})), indent=2)}",
        "",
        "## 7. base_link axisymmetry residual",
        "",
        f"- {json.dumps(_jsonable(report.get('7_base_yaw_symmetry', {})), indent=2)}",
        "",
        "## Bugs found",
        "",
    ]
    bugs = []
    bugs.append(
        report.get("5_fk_coverage", {}).get("bug_note_build_flange_tensor")
    )
    for b in report.get("6_dls_vs_srs", {}).get("bugs") or []:
        bugs.append(f"{b['file']}:{b['lines']}: {b['issue']}")
    if report.get("7_base_yaw_symmetry", {}).get("env_geometry_bug"):
        bugs.append(report["7_base_yaw_symmetry"]["env_geometry_bug"])
    for b in bugs:
        if b:
            lines.append(f"- {b}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 0 RM4D canonicalization audit (read-only)")
    p.add_argument("--seed", type=int, default=20260729)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--mode",
        choices=("smoke", "thorough"),
        default="smoke",
        help="smoke: fast subsampled; thorough: larger N including FK curve toward 1e8",
    )
    p.add_argument("--out-dir", type=Path, default=IRD_ROOT / "data" / "audit")
    p.add_argument("--skip", nargs="*", default=[], help="skip item numbers 1..7")
    p.add_argument(
        "--merge-existing",
        action="store_true",
        help="merge into existing phase0_report.json for skipped items",
    )
    p.add_argument("--fk-max-wall-s", type=float, default=None)
    p.add_argument("--conflict-max-samples", type=int, default=None)
    p.add_argument("--dls-srs-n", type=int, default=None)
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> AuditConfig:
    cfg = AuditConfig(seed=args.seed, device=args.device)
    if args.mode == "smoke":
        cfg.conflict_max_samples = 80_000
        cfg.q7_n_configs = 32
        cfg.q7_n_steps = 120
        cfg.q1_n_samples = 2_000
        cfg.gamma_n_configs = 48
        cfg.gamma_n_steps = 180
        cfg.fk_curve_counts = (100_000, 1_000_000, 10_000_000)
        cfg.fk_max_wall_s = 180.0
        cfg.dls_srs_n_poses = 400
        cfg.base_n_configs = 64
        cfg.base_n_phi = 36
    else:
        cfg.conflict_max_samples = 300_000
        cfg.q7_n_configs = 128
        cfg.q1_n_samples = 12_000
        cfg.gamma_n_configs = 256
        cfg.fk_curve_counts = (1_000_000, 10_000_000, 100_000_000)
        cfg.fk_max_wall_s = 1200.0
        cfg.dls_srs_n_poses = 2_000
        cfg.base_n_configs = 512
    if args.fk_max_wall_s is not None:
        cfg.fk_max_wall_s = args.fk_max_wall_s
    if args.conflict_max_samples is not None:
        cfg.conflict_max_samples = args.conflict_max_samples
    if args.dls_srs_n is not None:
        cfg.dls_srs_n_poses = args.dls_srs_n
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    skip = {str(x) for x in args.skip}
    rng = _seed_everything(cfg.seed)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading robot on {cfg.device} ...")
    spec, lm, collision, kin = _load_robot(cfg.device)
    report: dict[str, Any] = {}
    json_path = out_dir / "phase0_report.json"
    if args.merge_existing and json_path.is_file():
        report = json.loads(json_path.read_text(encoding="utf-8"))
        report["merged_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        report["merge_note"] = f"re-ran items except skip={sorted(skip)}"
    report.update(
        {
            "schema": "phase0_canonicalization_audit_v1",
            "seed": cfg.seed,
            "device": str(kin.device),
            "mode": args.mode,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": asdict(cfg),
            "robot_contract_present_in_live_spec": True,
            "T_root_j1_axis": spec.root_to_j1_axis().tolist(),
        }
    )

    if "1" not in skip:
        print("Audit 1: label conflicts ...")
        report["1_label_conflicts"] = audit_label_conflicts(cfg, spec, str(kin.device))
    if "2" not in skip:
        print("Audit 2: q7 invariance ...")
        report["2_q7_invariance"] = audit_q7_invariance(
            cfg, spec, lm, collision, kin, rng
        )
    if "3" not in skip:
        print("Audit 3: q1 realizability ...")
        report["3_q1_realizability"] = audit_q1_realizability(
            cfg, spec, lm, collision, kin, rng
        )
    if "4" not in skip:
        print("Audit 4: gamma collision ...")
        report["4_gamma_collision"] = audit_gamma_collision(cfg, lm, collision, rng)
    if "5" not in skip:
        print("Audit 5: FK coverage ...")
        report["5_fk_coverage"] = audit_fk_coverage(cfg, spec, lm, collision, kin, rng)
    if "6" not in skip:
        print("Audit 6: DLS vs SRS ...")
        report["6_dls_vs_srs"] = audit_dls_vs_srs(cfg, spec, lm, collision, kin, rng)
    if "7" not in skip:
        print("Audit 7: base yaw symmetry ...")
        report["7_base_yaw_symmetry"] = audit_base_yaw_symmetry(cfg, lm, collision, rng)

    json_path = out_dir / "phase0_report.json"
    md_path = out_dir / "phase0_report.md"
    json_path.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    _write_markdown(report, md_path)
    _print_summary(report)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
