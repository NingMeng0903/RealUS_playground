"""Project collision-checked pose / stencil GT into the 9-D flange chart.

Supervision policy (Phase 3):
  * near-field signed distance from stencil metres when ``margin_weight > 0``
  * far-field prior from anisotropic chart EDT when provided and stencil is
    absent on that row
  * classification sign from FK+ / IK− ``reachable`` labels
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import json
import numpy as np
import yaml

from ird_playground.ird.canonical import (
    FLANGE_CANONICAL_DIM,
    canonical_flange_from_se3_features,
    rotation_from_6d_torch,
)
from ird_playground.ird.export_gt import load_ird_gt
from ird_playground.ird.metric import metric_manifest
from ird_playground.ird.robot_model import (
    assert_robot_contract_compatible,
    load_robot_model_spec,
)
from ird_playground.ird.torch_kinematics import so3_exp, so3_log
from ird_playground.map.build_flange_tensor import CHART_NAMES, flange_pose_to_chart
from ird_playground.neural.signed_field import compute_input_stats

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


@dataclass(frozen=True)
class CanonicalGtConfig:
    source_npz: str = "data/ird/gpu_pose_stencils_production.npz"
    auxiliary_npz: tuple[str, ...] = ()
    output_npz: str = "data/ird/rm4d_signed_production.npz"
    edt_npz: str | None = None
    far_field_sdf_weight: float = 0.25
    quantile_lo: float = 0.005
    quantile_hi: float = 0.995
    min_pair_separation_normalized: float = 1.0e-5
    robot_spec: str | None = None


def _nearest_side_indices(
    boundary_id: np.ndarray, signed: np.ndarray, positive: bool
) -> tuple[np.ndarray, np.ndarray]:
    side = signed > 0 if positive else signed < 0
    idx = np.flatnonzero((boundary_id >= 0) & side & np.isfinite(signed))
    order = np.lexsort((np.abs(signed[idx]), boundary_id[idx]))
    idx = idx[order]
    groups, first = np.unique(boundary_id[idx], return_index=True)
    return groups.astype(np.int64), idx[first].astype(np.int64)


def _require_source_pose_id(arrays: dict[str, np.ndarray], n: int) -> np.ndarray:
    if "source_pose_id" not in arrays:
        raise KeyError(
            "source_pose_id is required for grouped train/val splits; "
            "silent fallback is forbidden"
        )
    source = np.asarray(arrays["source_pose_id"], dtype=np.int64).reshape(-1)
    if source.shape[0] != n:
        raise ValueError(
            f"source_pose_id length {source.shape[0]} != canonical length {n}"
        )
    return source


def _flange_canonical_from_arrays(
    arrays: dict[str, np.ndarray],
    *,
    T_root_axis: np.ndarray,
) -> np.ndarray:
    if "flange_canonical" in arrays:
        canonical = np.asarray(arrays["flange_canonical"], dtype=np.float32)
        if canonical.ndim != 2 or canonical.shape[-1] != FLANGE_CANONICAL_DIM:
            raise ValueError(
                f"flange_canonical must be (*, {FLANGE_CANONICAL_DIM}), "
                f"got {canonical.shape}"
            )
        return canonical
    features = np.asarray(arrays["features"], dtype=np.float32)
    # Production pose GT stores flange SE(3) features; identity tool transform.
    eye = np.eye(4, dtype=np.float32)
    return canonical_flange_from_se3_features(
        features, eye, T_root_axis=T_root_axis
    )


def interpolate_boundary_features(
    negative_features: np.ndarray,
    positive_features: np.ndarray,
    negative_signed: np.ndarray,
    positive_signed: np.ndarray,
) -> np.ndarray:
    """Interpolate an SE(3) stencil pair to its signed zero crossing."""
    if torch is None:
        raise ImportError("torch required")
    neg = np.asarray(negative_features, dtype=np.float32)
    pos = np.asarray(positive_features, dtype=np.float32)
    sn = np.asarray(negative_signed, dtype=np.float64).reshape(-1)
    sp = np.asarray(positive_signed, dtype=np.float64).reshape(-1)
    if neg.shape != pos.shape or neg.ndim != 2 or neg.shape[1] != 9:
        raise ValueError("boundary features must both have shape (N, 9)")
    if np.any(sn >= 0.0) or np.any(sp <= 0.0):
        raise ValueError("boundary interpolation requires negative/positive sides")
    alpha = (-sn / np.maximum(sp - sn, 1.0e-12)).astype(np.float32)
    with torch.no_grad():
        nf = torch.from_numpy(neg)
        pf = torch.from_numpy(pos)
        a = torch.from_numpy(alpha)
        position = nf[:, :3] + a[:, None] * (pf[:, :3] - nf[:, :3])
        r_neg = rotation_from_6d_torch(nf[:, 3:9])
        r_pos = rotation_from_6d_torch(pf[:, 3:9])
        rotation = so3_exp(a[:, None] * so3_log(r_pos @ r_neg.transpose(-1, -2))) @ r_neg
        return torch.cat(
            (position, rotation[:, :, 0], rotation[:, :, 1]), dim=-1
        ).numpy()


def _features_to_chart5(
    features: np.ndarray,
    *,
    T_root_axis: np.ndarray,
) -> np.ndarray:
    """Map flange SE(3) features in rail_base to 5-D EDT chart coordinates."""
    if torch is None:
        raise ImportError("torch required")
    x = np.asarray(features, dtype=np.float32)
    axis = torch.as_tensor(np.asarray(T_root_axis, dtype=np.float32))
    Ra = axis[:3, :3]
    pa = axis[:3, 3]
    with torch.no_grad():
        t = torch.from_numpy(x)
        p = t[:, :3]
        R = rotation_from_6d_torch(t[:, 3:9])
        R_axis = Ra.transpose(-1, -2) @ R
        p_axis = (Ra.transpose(-1, -2) @ (p - pa).unsqueeze(-1)).squeeze(-1)
        return flange_pose_to_chart(p_axis.numpy(), R_axis.numpy())


def _sample_edt_nearest(
    chart5: np.ndarray,
    edt_blob: dict[str, np.ndarray],
) -> np.ndarray:
    from ird_playground.map.build_flange_tensor import chart_coords_to_indices

    axes = tuple(np.asarray(edt_blob[name]) for name in CHART_NAMES)
    sdf = np.asarray(edt_blob["sdf"], dtype=np.float32)
    idx = chart_coords_to_indices(chart5, axes)
    return sdf[idx].astype(np.float32)


def _apply_far_field_edt(
    *,
    features: np.ndarray,
    reachable: np.ndarray,
    sdf_target: np.ndarray,
    sdf_weight: np.ndarray,
    T_root_axis: np.ndarray,
    edt_npz: str | Path,
    far_weight: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    blob = np.load(Path(edt_npz), allow_pickle=False)
    edt = {k: blob[k] for k in blob.files}
    chart5 = _features_to_chart5(features, T_root_axis=T_root_axis)
    edt_vals = _sample_edt_nearest(chart5, edt)
    far = sdf_weight <= 0.0
    # Keep sign consistent with FK+/IK− labels when both are available.
    signed = edt_vals.copy()
    pos = reachable > 0.5
    neg = reachable < 0.5
    signed[far & pos] = np.maximum(np.abs(signed[far & pos]), 1.0e-4)
    signed[far & neg] = -np.maximum(np.abs(signed[far & neg]), 1.0e-4)
    sdf_target = sdf_target.copy()
    sdf_weight = sdf_weight.copy()
    sdf_target[far] = signed[far]
    sdf_weight[far] = float(far_weight)
    meta = {
        "edt_npz": str(Path(edt_npz).resolve()),
        "far_field_rows": int(far.sum()),
        "far_field_sdf_weight": float(far_weight),
        "distance_definition": (
            "chart-coordinate weighted L2 metres; far-field prior only"
        ),
        "metric": metric_manifest(),
    }
    return sdf_target, sdf_weight, meta


def build_canonical_gt(cfg: CanonicalGtConfig) -> tuple[dict[str, np.ndarray], dict]:
    source = Path(cfg.source_npz)
    arrays = load_ird_gt(source)
    spec = load_robot_model_spec(cfg.robot_spec)
    T_root_axis = spec.root_to_j1_axis()
    canonical = _flange_canonical_from_arrays(arrays, T_root_axis=T_root_axis)
    source_pose_id = _require_source_pose_id(arrays, len(canonical))

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
        aux_arrays = {k: data[k] for k in data.files if k != "meta_json"}
        aux_canonical = _flange_canonical_from_arrays(
            aux_arrays, T_root_axis=T_root_axis
        )
        auxiliary.append(
            (
                path,
                aux_canonical,
                np.asarray(data["reachable"], dtype=np.float32),
                _require_source_pose_id(aux_arrays, len(aux_canonical))
                if "source_pose_id" in aux_arrays
                else np.full(len(aux_canonical), -1, dtype=np.int64),
            )
        )

    scale_data = np.concatenate([canonical, *(item[1] for item in auxiliary)], axis=0)
    center, scale = compute_input_stats(
        scale_data,
        quantile_lo=cfg.quantile_lo,
        quantile_hi=cfg.quantile_hi,
    )

    bid = np.asarray(
        arrays.get("boundary_id", np.full(len(canonical), -1)), dtype=np.int64
    )
    kind = np.asarray(
        arrays.get("clearance_kind", np.full(len(canonical), -1)), dtype=np.int8
    )
    signed_m = np.asarray(
        arrays.get("boundary_signed_m", np.full(len(canonical), np.nan)),
        dtype=np.float32,
    )
    signed_deg = np.asarray(
        arrays.get("boundary_signed_rot_deg", np.full(len(canonical), np.nan)),
        dtype=np.float32,
    )
    # Prefer declared-metric metres; fall back to deg column only for pairing.
    signed = np.where(np.isfinite(signed_m), signed_m, signed_deg)
    gpos, ipos = _nearest_side_indices(bid, signed, True)
    gneg, ineg = _nearest_side_indices(bid, signed, False)
    common, p_at, n_at = np.intersect1d(gpos, gneg, assume_unique=True, return_indices=True)
    ip, inn = ipos[p_at], ineg[n_at]

    max_gid = int(max(bid.max(initial=-1), -1))
    group_normal = np.zeros((max_gid + 1, FLANGE_CANONICAL_DIM), dtype=np.float32)
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

    center_count = len(valid_groups)
    boundary_features = arrays.get("boundary_features")
    center_features = None
    if center_count and boundary_features is not None:
        candidate = np.asarray(boundary_features)[ip[monotonic]]
        if np.isfinite(candidate).all():
            center_features = candidate
    if center_count and center_features is None:
        if "features" not in arrays:
            raise KeyError(
                "features or boundary_features is required for on-manifold zero supervision"
            )
        center_features = interpolate_boundary_features(
            np.asarray(arrays["features"])[inn[monotonic]],
            np.asarray(arrays["features"])[ip[monotonic]],
            signed[inn[monotonic]],
            signed[ip[monotonic]],
        )
    center_canonical = (
        _flange_canonical_from_arrays(
            {"features": center_features},
            T_root_axis=T_root_axis,
        )
        if center_count
        else np.zeros((0, FLANGE_CANONICAL_DIM), dtype=np.float32)
    )
    center_kind = kind[ip[monotonic]]
    center_normal = group_normal[valid_groups]
    center_slope = group_slope[valid_groups]
    center_signed_m = np.where(center_kind == 0, 0.0, np.nan).astype(np.float32)
    center_signed_deg = np.where(center_kind == 1, 0.0, np.nan).astype(np.float32)

    sdf_target = target.copy()
    sdf_weight = np.asarray(arrays["margin_weight"], dtype=np.float32).copy()
    edt_meta: dict = {}
    if cfg.edt_npz is not None and Path(cfg.edt_npz).is_file():
        sdf_target, sdf_weight, edt_meta = _apply_far_field_edt(
            features=np.asarray(arrays["features"], dtype=np.float32),
            reachable=np.asarray(arrays["reachable"], dtype=np.float32),
            sdf_target=sdf_target,
            sdf_weight=sdf_weight,
            T_root_axis=T_root_axis,
            edt_npz=cfg.edt_npz,
            far_weight=cfg.far_field_sdf_weight,
        )

    aux_n = sum(len(item[1]) for item in auxiliary)
    aux_canonical = [item[1].astype(np.float32) for item in auxiliary]
    aux_y = [item[2] for item in auxiliary]
    next_source_id = int(source_pose_id[source_pose_id >= 0].max(initial=-1)) + 1
    aux_source = []
    for _, aux_chart, _, raw_source in auxiliary:
        raw_source = np.asarray(raw_source, dtype=np.int64)
        mapped = np.empty(len(aux_chart), dtype=np.int64)
        for value in np.unique(raw_source[raw_source >= 0]).tolist():
            mapped[raw_source == value] = next_source_id
            next_source_id += 1
        missing = np.flatnonzero(raw_source < 0)
        mapped[missing] = np.arange(next_source_id, next_source_id + len(missing))
        next_source_id += len(missing)
        aux_source.append(mapped)
    block_id = np.asarray(
        arrays.get("block_id", np.full(len(canonical), -1)), dtype=np.int64
    )
    q_best = (
        np.asarray(arrays["q_best"], dtype=np.float32)
        if "q_best" in arrays
        else None
    )

    out = {
        "canonical": np.concatenate(
            (canonical, center_canonical.astype(np.float32), *aux_canonical)
        ),
        "reachable": np.concatenate(
            (
                np.asarray(arrays["reachable"], dtype=np.float32),
                np.full(center_count, 0.5, dtype=np.float32),
                *aux_y,
            )
        ),
        "classification_weight": np.concatenate(
            (
                np.ones(len(canonical), dtype=np.float32),
                np.zeros(center_count, dtype=np.float32),
                np.ones(aux_n, dtype=np.float32),
            )
        ),
        "sdf_target": np.concatenate(
            (sdf_target, np.zeros(center_count + aux_n, dtype=np.float32))
        ),
        "sdf_weight": np.concatenate(
            (
                sdf_weight,
                np.ones(center_count, dtype=np.float32),
                np.zeros(aux_n, dtype=np.float32),
            )
        ),
        "normal": np.concatenate(
            (
                normal,
                center_normal.astype(np.float32),
                np.zeros((aux_n, FLANGE_CANONICAL_DIM), dtype=np.float32),
            )
        ),
        "normal_slope": np.concatenate(
            (slope_row, center_slope.astype(np.float32), np.zeros(aux_n, dtype=np.float32))
        ),
        "normal_weight": np.concatenate(
            (
                normal_weight,
                np.ones(center_count, dtype=np.float32),
                np.zeros(aux_n, dtype=np.float32),
            )
        ),
        "boundary_id": np.concatenate(
            (bid, valid_groups.astype(np.int64), np.full(aux_n, -1, dtype=np.int64))
        ),
        "clearance_kind": np.concatenate(
            (kind, center_kind.astype(np.int8), np.full(aux_n, -1, dtype=np.int8))
        ),
        "boundary_signed_m": np.concatenate(
            (signed_m, center_signed_m, np.full(aux_n, np.nan, dtype=np.float32))
        ),
        "boundary_signed_rot_deg": np.concatenate(
            (signed_deg, center_signed_deg, np.full(aux_n, np.nan, dtype=np.float32))
        ),
        "source_pose_id": np.concatenate(
            (
                source_pose_id,
                source_pose_id[ip[monotonic]] if center_count else np.zeros(0, dtype=np.int64),
                *(aux_source if aux_source else [np.zeros(0, dtype=np.int64)]),
            )
        ),
        "block_id": np.concatenate(
            (
                block_id,
                block_id[ip[monotonic]] if center_count else np.zeros(0, dtype=np.int64),
                np.full(aux_n, -1, dtype=np.int64),
            )
        ),
        "sample_origin": np.concatenate(
            (
                np.zeros(len(canonical), dtype=np.int8),
                np.ones(center_count, dtype=np.int8),
                *(np.full(len(item), 2, dtype=np.int8) for item in aux_canonical),
            )
        ),
        "input_center": center,
        "input_scale": scale,
    }
    if q_best is not None:
        # Propagate for external orbit holdout (or explicit orbit_id if added later).
        dof = q_best.shape[-1]
        center_q = (
            q_best[ip[monotonic]]
            if center_count
            else np.zeros((0, dof), dtype=np.float32)
        )
        aux_q = [
            np.zeros((len(item[1]), dof), dtype=np.float32) for item in auxiliary
        ]
        out["q_best"] = np.concatenate((q_best, center_q, *aux_q), axis=0)
    meta = {
        "representation": "flange_j1_axis_9d_embedding_v1",
        "intrinsic_dimension": 5,
        "embedding_dimension": FLANGE_CANONICAL_DIM,
        "features": [
            "p_z_m",
            "r_m",
            "u_x_dot_z",
            "p_xy_dot_u_x_m",
            "p_xy_cross_u_x_m",
            "u_z_dot_z",
            "p_xy_dot_u_z_m",
            "p_xy_cross_u_z_m",
            "u_y_dot_z",
        ],
        "source_npz": str(source.resolve()),
        "n": len(canonical) + center_count + aux_n,
        "n_source": len(canonical),
        "n_auxiliary": aux_n,
        "auxiliary_npz": [str(item[0].resolve()) for item in auxiliary],
        "n_zero_boundary": center_count,
        "zero_boundary_schema": "on_manifold_se3_interpolation_v1",
        "source_id_schema": "namespaced_source_or_unique_aux_v1",
        "n_boundary_groups": int(np.unique(bid[st]).size) if st.any() else 0,
        "n_oriented_groups": int(len(valid_groups)),
        "collision_contract": source_manifest.get("collision_contract"),
        "canonical_frame": "physical_joint_1_axis",
        "robot_contract": spec.to_manifest(),
        "metric": metric_manifest(),
        "supervision": {
            "near_field": "stencil metres (margin_weight > 0)",
            "far_field": "EDT metres when edt_npz provided",
            "sign": "FK positive / IK negative reachable labels",
            "q1_aux_head": False,
        },
        "far_field_edt": edt_meta,
        "config": asdict(cfg),
    }
    return out, meta


def save_canonical_gt(path: str | Path, arrays: dict[str, np.ndarray], meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(arrays)
    payload["meta_json"] = np.asarray(json.dumps(meta, sort_keys=True))
    np.savez_compressed(path, **payload)


__all__ = [
    "CanonicalGtConfig",
    "build_canonical_gt",
    "interpolate_boundary_features",
    "save_canonical_gt",
]
