"""Project collision-checked pose GT into the RM4D invariant embedding."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import json
import numpy as np
import yaml

from ird_playground.ird.canonical import canonical_from_se3_features
from ird_playground.ird.export_gt import load_ird_gt
from ird_playground.ird.robot_model import (
    assert_robot_contract_compatible,
    load_robot_model_spec,
)


@dataclass(frozen=True)
class CanonicalGtConfig:
    source_npz: str = "data/ird/gpu_pose_stencils_production.npz"
    auxiliary_npz: tuple[str, ...] = ()
    output_npz: str = "data/ird/rm4d_signed_production.npz"
    quantile_lo: float = 0.005
    quantile_hi: float = 0.995
    min_pair_separation_normalized: float = 1.0e-5
    robot_spec: str | None = None


def _nearest_side_indices(boundary_id: np.ndarray, signed: np.ndarray, positive: bool) -> tuple[np.ndarray, np.ndarray]:
    side = signed > 0 if positive else signed < 0
    idx = np.flatnonzero((boundary_id >= 0) & side & np.isfinite(signed))
    order = np.lexsort((np.abs(signed[idx]), boundary_id[idx]))
    idx = idx[order]
    groups, first = np.unique(boundary_id[idx], return_index=True)
    return groups.astype(np.int64), idx[first].astype(np.int64)


def build_canonical_gt(cfg: CanonicalGtConfig) -> tuple[dict[str, np.ndarray], dict]:
    source = Path(cfg.source_npz)
    arrays = load_ird_gt(source)
    spec = load_robot_model_spec(cfg.robot_spec)
    T_root_axis = spec.root_to_j1_axis()
    canonical = canonical_from_se3_features(
        arrays["features"], T_root_axis=T_root_axis
    )
    source_manifest_path = source.with_suffix(".yaml")
    source_manifest = (
        yaml.safe_load(source_manifest_path.read_text(encoding="utf-8")) or {}
        if source_manifest_path.is_file()
        else {}
    )
    assert_robot_contract_compatible(source_manifest.get("robot_contract"), spec)
    auxiliary = []
    for raw_path in cfg.auxiliary_npz:
        path = Path(raw_path)
        data = np.load(path, allow_pickle=False)
        auxiliary_meta = (
            json.loads(str(data["meta_json"].item()))
            if "meta_json" in data.files
            else {}
        )
        assert_robot_contract_compatible(
            auxiliary_meta.get("robot_contract"), spec
        )
        auxiliary.append(
            (
                path,
                canonical_from_se3_features(
                    data["features"], T_root_axis=T_root_axis
                ),
                np.asarray(data["reachable"], dtype=np.float32),
            )
        )
    scale_data = np.concatenate([canonical, *(item[1] for item in auxiliary)], axis=0)
    lo = np.quantile(scale_data, cfg.quantile_lo, axis=0).astype(np.float32)
    hi = np.quantile(scale_data, cfg.quantile_hi, axis=0).astype(np.float32)
    center = 0.5 * (lo + hi)
    scale = np.maximum(0.5 * (hi - lo), 1.0e-4).astype(np.float32)

    bid = np.asarray(arrays.get("boundary_id", np.full(len(canonical), -1)), dtype=np.int64)
    kind = np.asarray(arrays.get("clearance_kind", np.full(len(canonical), -1)), dtype=np.int8)
    signed_m = np.asarray(arrays.get("boundary_signed_m", np.full(len(canonical), np.nan)), dtype=np.float32)
    signed_deg = np.asarray(arrays.get("boundary_signed_rot_deg", np.full(len(canonical), np.nan)), dtype=np.float32)
    signed = np.where(np.isfinite(signed_m), signed_m * 1000.0, signed_deg)
    gpos, ipos = _nearest_side_indices(bid, signed, True)
    gneg, ineg = _nearest_side_indices(bid, signed, False)
    common, p_at, n_at = np.intersect1d(gpos, gneg, assume_unique=True, return_indices=True)
    ip, inn = ipos[p_at], ineg[n_at]

    max_gid = int(max(bid.max(initial=-1), -1))
    group_normal = np.zeros((max_gid + 1, 5), dtype=np.float32)
    group_slope = np.zeros(max_gid + 1, dtype=np.float32)
    dx = (canonical[ip] - canonical[inn]) / scale
    norm = np.linalg.norm(dx, axis=1)
    target = np.asarray(arrays["m_gt"], dtype=np.float32)
    slope = (target[ip] - target[inn]) / np.maximum(norm, 1.0e-12)
    monotonic = (
        (np.asarray(arrays["margin_weight"])[ip] > 0)
        & (np.asarray(arrays["margin_weight"])[inn] > 0)
        & (norm >= cfg.min_pair_separation_normalized)
        & np.isfinite(slope)
    )
    valid_groups = common[monotonic]
    group_normal[valid_groups] = dx[monotonic] / norm[monotonic, None]
    group_slope[valid_groups] = np.clip(slope[monotonic], 0.05, 50.0)

    normal = np.zeros_like(canonical)
    slope_row = np.zeros(len(canonical), dtype=np.float32)
    normal_weight = np.zeros(len(canonical), dtype=np.float32)
    st = bid >= 0
    normal[st] = group_normal[bid[st]]
    slope_row[st] = group_slope[bid[st]]
    normal_weight[st] = (group_slope[bid[st]] > 0).astype(np.float32)

    center_canonical = 0.5 * (canonical[ip[monotonic]] + canonical[inn[monotonic]])
    center_kind = kind[ip[monotonic]]
    center_count = len(valid_groups)
    center_normal = group_normal[valid_groups]
    center_slope = group_slope[valid_groups]
    center_signed_m = np.where(center_kind == 0, 0.0, np.nan).astype(np.float32)
    center_signed_deg = np.where(center_kind == 1, 0.0, np.nan).astype(np.float32)

    aux_n = sum(len(item[1]) for item in auxiliary)
    aux_canonical = [item[1].astype(np.float32) for item in auxiliary]
    aux_y = [item[2] for item in auxiliary]
    source_pose_id = np.asarray(
        arrays.get("source_pose_id", np.full(len(canonical), -1)), dtype=np.int64
    )
    out = {
        "canonical": np.concatenate((canonical, center_canonical.astype(np.float32), *aux_canonical)),
        "reachable": np.concatenate((np.asarray(arrays["reachable"], dtype=np.float32), np.full(center_count, 0.5, dtype=np.float32), *aux_y)),
        "classification_weight": np.concatenate((np.ones(len(canonical), dtype=np.float32), np.zeros(center_count, dtype=np.float32), np.ones(aux_n, dtype=np.float32))),
        "sdf_target": np.concatenate((target, np.zeros(center_count + aux_n, dtype=np.float32))),
        "sdf_weight": np.concatenate((np.asarray(arrays["margin_weight"], dtype=np.float32), np.ones(center_count, dtype=np.float32), np.zeros(aux_n, dtype=np.float32))),
        "normal": np.concatenate((normal, center_normal.astype(np.float32), np.zeros((aux_n, 5), dtype=np.float32))),
        "normal_slope": np.concatenate((slope_row, center_slope.astype(np.float32), np.zeros(aux_n, dtype=np.float32))),
        "normal_weight": np.concatenate((normal_weight, np.ones(center_count, dtype=np.float32), np.zeros(aux_n, dtype=np.float32))),
        "boundary_id": np.concatenate((bid, valid_groups.astype(np.int64), np.full(aux_n, -1, dtype=np.int64))),
        "clearance_kind": np.concatenate((kind, center_kind.astype(np.int8), np.full(aux_n, -1, dtype=np.int8))),
        "boundary_signed_m": np.concatenate((signed_m, center_signed_m, np.full(aux_n, np.nan, dtype=np.float32))),
        "boundary_signed_rot_deg": np.concatenate((signed_deg, center_signed_deg, np.full(aux_n, np.nan, dtype=np.float32))),
        "source_pose_id": np.concatenate(
            (
                source_pose_id,
                source_pose_id[ip[monotonic]],
                np.full(aux_n, -1, dtype=np.int64),
            )
        ),
        "input_center": center,
        "input_scale": scale,
    }
    meta = {
        "representation": "rm4d_j1_axis_continuous_embedding_v2",
        "intrinsic_dimension": 4,
        "embedding_dimension": 5,
        "features": ["p_z_m", "approach_z", "p_xy_radius_m", "p_xy_dot_approach_xy_m", "p_xy_cross_approach_xy_m"],
        "source_npz": str(source.resolve()),
        "n": len(canonical) + center_count + aux_n,
        "n_source": len(canonical),
        "n_auxiliary": aux_n,
        "auxiliary_npz": [str(item[0].resolve()) for item in auxiliary],
        "n_zero_boundary": center_count,
        "n_boundary_groups": int(np.unique(bid[st]).size),
        "n_oriented_groups": int(len(valid_groups)),
        "collision_contract": source_manifest.get("collision_contract"),
        "canonical_frame": "physical_joint_1_axis",
        "robot_contract": spec.to_manifest(),
        "config": asdict(cfg),
    }
    return out, meta


def save_canonical_gt(path: str | Path, arrays: dict[str, np.ndarray], meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(arrays)
    payload["meta_json"] = np.asarray(json.dumps(meta, sort_keys=True))
    np.savez_compressed(path, **payload)


__all__ = ["CanonicalGtConfig", "build_canonical_gt", "save_canonical_gt"]
