"""Build a compact in-place transverse tilt geometry/IK bank.

The bank is deliberately geometry-first.  It decodes the frozen dynamic
table, uses :class:`dynamic_scan.DynamicScan` to select common, single-trunk
station rows, interpolates the *actual* center-to-endpoint frame rotations in
SO(3), and asks the public ``CapabilityEvaluator`` for finite endpoint IK at
the eight intermediate tilt choices.  It does not claim a continuous contact
path, force feasibility, anatomy collision safety, or hardware execution.

Run with the public genesis environment, for example::

    /media/camp/EXT_DRIVE/envs/genesis/bin/python tilt_geometry.py

The primary NPZ keys are the exact compact contract requested by the study::

    p_idx, site_idx, angles_deg, contact_xyz_m, frames, vessel_xyz_m,
    reference_frame, valid, q, edge_mask

Additional diagnostic arrays (dIRD margin, selected rail, and robot targets)
are included so a consumer does not have to recover the public capability
scores from hidden state.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
DYNAMIC_DIR = HERE.parent / "ab_leg_dynamic_20261112"
FROZEN_PATH = DYNAMIC_DIR / "DYNAMIC_GEOMETRY_FROZEN.npz"
DYNAMIC_SCAN_PATH = DYNAMIC_DIR / "dynamic_scan.py"
CAPABILITY_DIR = HERE.parent / "ab_module_ladder_20261110" / "capability"
CAPABILITY_PATH = CAPABILITY_DIR / "capability.py"

ANGLES_DEG = np.asarray(
    [-12.0, -10.5, -9.0, -7.5, -6.0, 0.0, 6.0, 7.5, 9.0, 10.5, 12.0],
    dtype=float,
)
SOURCE_ANGLES_DEG = np.asarray([-12.0, 0.0, 12.0], dtype=float)
SOURCE_VIEW_INDEX = np.asarray([7, 3, 8], dtype=np.int64)
ANGLE_TOL_DEG = 1.0e-8
MIN_DEPTH_MM = 8.0
MAX_DEPTH_MM = 65.0
EDGE_SAMPLE_T = np.asarray([0.25, 0.50, 0.75], dtype=float)


@dataclass(frozen=True)
class Source:
    data: dict[str, np.ndarray]
    metadata: dict[str, Any]
    scan: Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _load_dynamic_scan_class() -> Any:
    """Load the existing DynamicScan module without copying or editing it."""

    if not DYNAMIC_SCAN_PATH.is_file():
        raise FileNotFoundError(DYNAMIC_SCAN_PATH)
    name = "ab_fisher_dynamic_scan_for_tilt_bank"
    spec = importlib.util.spec_from_file_location(name, DYNAMIC_SCAN_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {DYNAMIC_SCAN_PATH}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules during import.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.DynamicScan


def _load_source(path: Path = FROZEN_PATH) -> Source:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        data = {key: np.asarray(payload[key]) for key in payload.files if key != "metadata_json"}
        metadata_json = payload["metadata_json"].item() if "metadata_json" in payload else "{}"
    metadata = json.loads(str(metadata_json))
    DynamicScan = _load_dynamic_scan_class()
    scan = DynamicScan(data)
    return Source(data=data, metadata=metadata, scan=scan)


def _central_view_indices(data: dict[str, np.ndarray]) -> tuple[int, int, int]:
    offsets = np.asarray(data["view_offsets_mm"], dtype=float)
    tilts = np.asarray(data["view_tilts_deg"], dtype=float)

    def one(tilt: float) -> int:
        candidates = np.flatnonzero((np.abs(offsets) <= 1.0e-9) & (np.abs(tilts - tilt) <= ANGLE_TOL_DEG))
        if len(candidates) != 1:
            raise ValueError(f"expected one central view at tilt {tilt:g}, got {candidates.tolist()}")
        return int(candidates[0])

    return one(-12.0), one(0.0), one(12.0)


def _nominal_depth_mm(source: Source, central: int) -> np.ndarray:
    data = source.data
    # Dot with the center-frame inward axis, matching DynamicScan's nominal
    # depth convention.  This is a signed geometry check, not a tissue claim.
    delta = data["posed_vessel_m"][:, :, None, :] - data["posed_views_m"][:, :, central : central + 1, :]
    inward = data["posed_frames"][:, :, central : central + 1, :, 2]
    return np.asarray(np.einsum("pskc,pskc->psk", delta, inward)[:, :, 0] * 1000.0, dtype=float)


def _select_rows(source: Source, central: int, minus: int, plus: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = source.data
    scan = source.scan
    pose_count, station_count = data["posed_views_m"].shape[:2]
    labels = np.asarray(data["labels"]).astype(str)
    branch_indices = set(int(i) for i in np.asarray(scan.branch_indices).reshape(-1))
    depth = _nominal_depth_mm(source, central)
    source_cols = np.asarray([minus, central, plus], dtype=np.int64)
    common = np.all(np.asarray(scan.eligible[:, :, source_cols], dtype=bool), axis=2)
    common &= np.all(np.isfinite(depth), axis=0)
    common &= np.all((depth >= MIN_DEPTH_MM) & (depth <= MAX_DEPTH_MM), axis=0)
    # Keep only the public main-trunk label and remove fork/shared stations
    # reported by DynamicScan's branch reconstruction.
    single_trunk = np.asarray([label == "R_POST_TIB_V" and i not in branch_indices
                               for i, label in enumerate(labels)], dtype=bool)
    common &= single_trunk[None, :]
    site_mask = np.all(common, axis=0)
    site_idx = np.flatnonzero(site_mask).astype(np.int64)
    if len(site_idx) < 10:
        raise ValueError(f"only {len(site_idx)} common single-trunk sites passed the depth/eligibility gate")
    p_idx = np.repeat(np.arange(pose_count, dtype=np.int64), len(site_idx))
    row_site_idx = np.tile(site_idx, pose_count)
    return p_idx, row_site_idx, depth


def _make_robot_target(robot_from_atlas: np.ndarray, frame: np.ndarray, point: np.ndarray) -> np.ndarray:
    target = np.eye(4, dtype=float)
    target[:3, :3] = np.asarray(frame, dtype=float)
    target[:3, 3] = np.asarray(point, dtype=float)
    return np.asarray(robot_from_atlas, dtype=float) @ target


def _interpolate_frame(ref: np.ndarray, minus: np.ndarray, plus: np.ndarray, angle: float) -> np.ndarray:
    """Interpolate the measured center/end rotations in local rotvec space."""

    ref = np.asarray(ref, dtype=float)
    rv_minus = Rotation.from_matrix(ref.T @ np.asarray(minus, dtype=float)).as_rotvec()
    rv_plus = Rotation.from_matrix(ref.T @ np.asarray(plus, dtype=float)).as_rotvec()
    if angle < -ANGLE_TOL_DEG:
        rv = rv_minus * (float(angle) / -12.0)
    elif angle > ANGLE_TOL_DEG:
        rv = rv_plus * (float(angle) / 12.0)
    else:
        return ref.copy()
    return ref @ Rotation.from_rotvec(rv).as_matrix()


def _frame_bank(source: Source, p_idx: np.ndarray, site_idx: np.ndarray,
                central: int, minus: int, plus: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    data = source.data
    n = len(p_idx)
    k = len(ANGLES_DEG)
    contact = np.empty((n, k, 3), dtype=float)
    frames = np.empty((n, k, 3, 3), dtype=float)
    reference = np.empty((n, 3, 3), dtype=float)
    endpoint_error = 0.0
    center_error = 0.0
    for row, (p, site) in enumerate(zip(p_idx, site_idx, strict=True)):
        xyz = np.asarray(data["posed_views_m"][p, site, central], dtype=float)
        ref = np.asarray(data["posed_frames"][p, site, central], dtype=float)
        end_minus = np.asarray(data["posed_frames"][p, site, minus], dtype=float)
        end_plus = np.asarray(data["posed_frames"][p, site, plus], dtype=float)
        reference[row] = ref
        contact[row] = xyz
        for j, angle in enumerate(ANGLES_DEG):
            frames[row, j] = _interpolate_frame(ref, end_minus, end_plus, float(angle))
        endpoint_error = max(endpoint_error,
                             float(np.max(np.abs(frames[row, 0] - end_minus))),
                             float(np.max(np.abs(frames[row, -1] - end_plus))))
        center_error = max(center_error, float(np.max(np.abs(frames[row, 5] - ref))))
    checks = {
        "minus12_frame_max_abs_error": float(endpoint_error),
        "zero_frame_max_abs_error": float(center_error),
    }
    if endpoint_error > 2.0e-7 or center_error > 2.0e-12:
        raise ValueError(f"rotvec frame endpoint/center check failed: {checks}")
    return contact, frames, reference, checks


def _source_order(angle: float) -> np.ndarray:
    # Prefer the measured source angle closest in magnitude.  At the exact
    # +/-6 tie, prefer the neutral source to keep the seed close to neutral.
    order = sorted(range(3), key=lambda i: (abs(float(angle) - SOURCE_ANGLES_DEG[i]),
                                              abs(float(SOURCE_ANGLES_DEG[i])), i))
    return np.asarray(order, dtype=np.int64)


def _build_robot_bank(source: Source, p_idx: np.ndarray, site_idx: np.ndarray,
                      contact: np.ndarray, frames: np.ndarray,
                      central: int, minus: int, plus: int,
                      static: bool = True, verbose: bool = True) -> dict[str, np.ndarray | Any]:
    """Score all choices and finite-check intermediate endpoints."""

    if str(CAPABILITY_DIR) not in sys.path:
        sys.path.insert(0, str(CAPABILITY_DIR))
    from capability import CapabilityEvaluator  # type: ignore

    data = source.data
    n, k = contact.shape[:2]
    robot_from_atlas = np.asarray(data["robot_from_atlas"], dtype=float)
    targets = np.empty((n, k, 4, 4), dtype=float)
    for row in range(n):
        for j in range(k):
            targets[row, j] = _make_robot_target(robot_from_atlas, frames[row, j], contact[row, j])

    # Three measured endpoint choices are the only seed provenance.  Intermediate
    # rows start from the nearest source q/rail, then fall back to the other
    # measured source choices if the finite search misses.
    source_q = np.asarray(data["q_solution"][p_idx, site_idx][:, SOURCE_VIEW_INDEX], dtype=float)
    source_rail = np.asarray(data["chosen_rail_m"][p_idx, site_idx][:, SOURCE_VIEW_INDEX], dtype=float)
    source_found = np.asarray(data["static_found"][p_idx, site_idx][:, SOURCE_VIEW_INDEX], dtype=bool)
    if not np.all(source_found) or not np.isfinite(source_q).all():
        raise ValueError("selected rows do not have finite frozen endpoint IK witnesses")

    # Score the three public rails for every requested target.  Static search
    # uses these same public rails; no new planning/optimization state is made.
    evaluator = CapabilityEvaluator(device="cpu")
    score_targets = np.repeat(targets[:, :, None], 3, axis=2).reshape(-1, 4, 4)
    score_rails = np.repeat(source_rail[:, None, :], k, axis=1).reshape(-1)
    score = evaluator.score(score_targets, score_rails)
    margins_by_source = np.asarray(score["raw_calibrated_margin"], dtype=float).reshape(n, k, 3)
    proxy_by_source = np.asarray(score["proxy_pass"], dtype=bool).reshape(n, k, 3)

    q = np.full((n, k, 8), np.nan, dtype=float)
    valid = np.zeros((n, k), dtype=bool)
    static_found = np.zeros((n, k), dtype=bool)
    dird_margin = np.full((n, k), np.nan, dtype=float)
    proxy_pass = np.zeros((n, k), dtype=bool)
    rail_m = np.full((n, k), np.nan, dtype=float)
    static_seconds = np.zeros((n, k), dtype=float)
    static_attempts = np.zeros((n, k), dtype=np.int64)
    selected_source = np.full((n, k), -1, dtype=np.int64)
    endpoint_source_for_angle = {int(j): int(np.flatnonzero(np.isclose(SOURCE_ANGLES_DEG, angle))[0])
                                 for j, angle in enumerate(ANGLES_DEG)
                                 if np.any(np.isclose(SOURCE_ANGLES_DEG, angle, atol=ANGLE_TOL_DEG))}

    for row, (p, site) in enumerate(zip(p_idx, site_idx, strict=True)):
        for j, angle in enumerate(ANGLES_DEG):
            source_idx = endpoint_source_for_angle.get(j)
            if source_idx is not None:
                # Exact frozen endpoint: do not re-search it.  Verify its
                # source target is the same measured target to sub-micron scale.
                q[row, j] = source_q[row, source_idx]
                valid[row, j] = True
                static_found[row, j] = True
                selected_source[row, j] = source_idx
                rail_m[row, j] = source_rail[row, source_idx]
                dird_margin[row, j] = margins_by_source[row, j, source_idx]
                proxy_pass[row, j] = proxy_by_source[row, j, source_idx]
                continue

            order = _source_order(float(angle))
            started = time.perf_counter()
            if static:
                for source_idx in order.tolist():
                    static_attempts[row, j] += 1
                    witness = evaluator._static_pose(
                        targets[row, j],
                        float(source_rail[row, source_idx]),
                        source_q[row, source_idx],
                    )
                    if bool(witness.get("found", False)):
                        q[row, j] = np.asarray(witness["q_solution"], dtype=float)
                        valid[row, j] = True
                        static_found[row, j] = True
                        selected_source[row, j] = int(source_idx)
                        rail_m[row, j] = source_rail[row, source_idx]
                        dird_margin[row, j] = margins_by_source[row, j, source_idx]
                        proxy_pass[row, j] = proxy_by_source[row, j, source_idx]
                        break
            else:
                source_idx = int(order[0])
                q[row, j] = source_q[row, source_idx]
                valid[row, j] = False
                selected_source[row, j] = source_idx
                rail_m[row, j] = source_rail[row, source_idx]
                dird_margin[row, j] = margins_by_source[row, j, source_idx]
                proxy_pass[row, j] = proxy_by_source[row, j, source_idx]
            static_seconds[row, j] = time.perf_counter() - started
        if verbose and ((row + 1) % max(1, n // 8) == 0 or row + 1 == n):
            print(f"IK rows {row + 1}/{n}", flush=True)

    # Reconstruct target poses from output geometry and verify the copied
    # center/endpoints are exactly the frozen public robot targets.
    for row, (p, site) in enumerate(zip(p_idx, site_idx, strict=True)):
        for j, source_idx in endpoint_source_for_angle.items():
            source_target = np.asarray(data["target_robot"][p, site, SOURCE_VIEW_INDEX[source_idx]], dtype=float)
            err = float(np.max(np.abs(targets[row, j] - source_target)))
            if err > 3.0e-6:
                raise ValueError(f"target reconstruction mismatch at row={row} angle={ANGLES_DEG[j]}: {err:g}")

    return {
        "targets": targets,
        "q": q,
        "valid": valid,
        "static_found": static_found,
        "dird_margin": dird_margin,
        "proxy_pass": proxy_pass,
        "rail_m": rail_m,
        "static_seconds": static_seconds,
        "static_attempts": static_attempts,
        "selected_source": selected_source,
        "evaluator": evaluator,
        "score_threshold": float(score["raw_threshold"]),
    }


def _edge_pairs() -> list[tuple[int, int, str]]:
    pairs: list[tuple[int, int, str]] = []
    for i in range(len(ANGLES_DEG) - 1):
        pairs.append((i, i + 1, "adjacent_angle"))
    center = int(np.flatnonzero(np.isclose(ANGLES_DEG, 0.0, atol=ANGLE_TOL_DEG))[0])
    for i, angle in enumerate(ANGLES_DEG):
        if i != center and angle != 0.0:
            pairs.append((center, i, "center_to_tilt"))
    # Keep every negative/positive cross-zero edge.  The exact -a,+b,
    # a+b=18 subset is tagged separately, while the complete 5x5 set lets
    # downstream studies compare wider 36/48-degree caps without rebuilding
    # endpoint IK.  (The q checks are still only three interior samples.)
    negative = [i for i, angle in enumerate(ANGLES_DEG) if angle < 0.0]
    positive = [i for i, angle in enumerate(ANGLES_DEG) if angle > 0.0]
    for i in negative:
        for j in positive:
            kind = "signed_sum_18" if abs(float(ANGLES_DEG[j] + abs(ANGLES_DEG[i])) - 18.0) <= ANGLE_TOL_DEG else "cross_zero"
            pairs.append((i, j, kind))
    # Center-to-+/-6 and a few adjacent cases occur in more than one policy;
    # preserve the first semantic tag and audit each unique pair once.
    unique: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for i, j, kind in pairs:
        key = (min(i, j), max(i, j))
        if key not in seen:
            seen.add(key)
            unique.append((i, j, kind))
    return unique


def _build_edge_mask(q: np.ndarray, valid: np.ndarray, evaluator: Any,
                     verbose: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
    n, k = valid.shape
    mask = np.zeros((n, k, k), dtype=bool)
    for row in range(n):
        for i in range(k):
            mask[row, i, i] = bool(valid[row, i])
    pairs = _edge_pairs()
    pair_records = []
    for i, j, kind in pairs:
        pass_count = 0
        for row in range(n):
            if not (valid[row, i] and valid[row, j]):
                continue
            passed = True
            for t in EDGE_SAMPLE_T:
                q_interp = (1.0 - float(t)) * q[row, i] + float(t) * q[row, j]
                if not np.isfinite(q_interp).all() or not bool(evaluator.check_q(q_interp).get("valid", False)):
                    passed = False
                    break
            if passed:
                mask[row, i, j] = True
                mask[row, j, i] = True
                pass_count += 1
        pair_records.append({
            "from_angle_deg": float(ANGLES_DEG[i]),
            "to_angle_deg": float(ANGLES_DEG[j]),
            "kind": kind,
            "interior_samples": EDGE_SAMPLE_T.tolist(),
            "passed_rows": int(pass_count),
            "total_rows": int(n),
        })
        if verbose:
            print(f"edge {ANGLES_DEG[i]:g}->{ANGLES_DEG[j]:g}: {pass_count}/{n}", flush=True)
    return mask, {
        "definition": "valid diagonal plus adjacent-angle, direct center-to-every-tilt, and all negative-by-positive cross-zero q edges",
        "interior_q_samples": EDGE_SAMPLE_T.tolist(),
        "endpoint_samples": "q endpoints are already finite static witnesses",
        "direct_center_to_tilt_angles_deg": [float(x) for x in ANGLES_DEG if x != 0.0],
        "pairs": pair_records,
        "no_continuous_path_claim": True,
    }


def build_bank(*, frozen_path: Path = FROZEN_PATH,
               output_path: Path = HERE / "TILT_BANK.npz",
               audit_path: Path = HERE / "TILT_BANK.json",
               static: bool = True, edges: bool = True,
               verbose: bool = True) -> dict[str, Any]:
    source = _load_source(frozen_path)
    data = source.data
    minus, central, plus = _central_view_indices(data)
    p_idx, site_idx, depth = _select_rows(source, central, minus, plus)
    contact, frames, reference, frame_checks = _frame_bank(source, p_idx, site_idx, central, minus, plus)
    vessel = np.asarray(data["posed_vessel_m"][p_idx, site_idx], dtype=float)
    bank = _build_robot_bank(source, p_idx, site_idx, contact, frames, central, minus, plus,
                             static=static, verbose=verbose)
    if edges:
        edge_mask, edge_checks = _build_edge_mask(bank["q"], bank["valid"], bank["evaluator"], verbose=verbose)
    else:
        edge_mask = np.zeros((len(p_idx), len(ANGLES_DEG), len(ANGLES_DEG)), dtype=bool)
        edge_mask[np.arange(len(p_idx))[:, None], np.arange(len(ANGLES_DEG)), np.arange(len(ANGLES_DEG))] = bank["valid"]
        edge_checks = {"definition": "valid diagonal only; edge sampling disabled", "pairs": []}

    evaluator_manifest = _jsonable(bank["evaluator"].manifest())
    pose_angles = np.asarray(data["pose_knee_deg"], dtype=float)
    selected_depth = depth[p_idx, site_idx]
    n = len(p_idx)
    metadata: dict[str, Any] = {
        "schema": "ab_tilt_bank_v1",
        "status": "geometry_only_verified_endpoint_ik_bank",
        "frame_convention": {
            "geometry_frame": "frozen_dynamic_atlas_frame",
            "ik_target_frame": "robot_from_atlas applied to T_atlas_contact",
            "reference_frame": "zero-degree central frame at each pose/station",
            "contact_policy": "same central contact xyz repeated at every tilt",
            "tilt_axis": "actual local rotvec from central frame to measured +/-12-degree endpoint",
            "sign_convention": "negative uses center->frozen -12 endpoint rotvec; positive uses center->frozen +12 endpoint rotvec",
        },
        "schema_contract": {
            "p_idx": "int64[N], zero-based frozen pose index",
            "site_idx": "int64[N], zero-based frozen station index",
            "angles_deg": "float64[K], K=11",
            "contact_xyz_m": "float64[N,K,3], fixed in-place center contact",
            "frames": "float64[N,K,3,3], SO(3) contact frames",
            "vessel_xyz_m": "float64[N,3], posed vessel anchor in geometry frame",
            "reference_frame": "float64[N,3,3], center reference frame",
            "valid": "bool[N,K], finite endpoint IK witness found",
            "q": "float64[N,K,8], public q witness; NaN when invalid",
            "edge_mask": "bool[N,K,K], sampled q-transition availability",
        },
        "angles_deg": ANGLES_DEG.tolist(),
        "pose_index_to_knee_deg": pose_angles.tolist(),
        "selected_pose_count": int(len(pose_angles)),
        "selected_row_count": int(n),
        "selected_site_count_per_pose": int(len(np.unique(site_idx))),
        "selected_site_idx": np.unique(site_idx).astype(int).tolist(),
        "selected_station_row_ids": np.asarray(data["row_ids"])[np.unique(site_idx)].astype(int).tolist(),
        "selected_labels": np.asarray(data["labels"])[np.unique(site_idx)].astype(str).tolist(),
        "nominal_center_depth_mm": {
            "min": float(np.min(selected_depth)),
            "max": float(np.max(selected_depth)),
            "requested_range": [MIN_DEPTH_MM, MAX_DEPTH_MM],
            "per_row_mm": selected_depth.tolist(),
        },
        "source_view_indices": {"minus12": minus, "center0": central, "plus12": plus},
        "frame_checks": frame_checks,
        "robot_target_checks": {
            "canonical_registration_path": str(frozen_path.resolve()),
            "robot_from_atlas": np.asarray(data["robot_from_atlas"]).tolist(),
            "dird_raw_threshold": float(bank["score_threshold"]),
            "dird_margin_min": float(np.nanmin(bank["dird_margin"])),
            "dird_margin_max": float(np.nanmax(bank["dird_margin"])),
            "static_found_count": int(np.sum(bank["static_found"])),
            "valid_count": int(np.sum(bank["valid"])),
            "total_count": int(bank["valid"].size),
        },
        "edge_checks": edge_checks,
        "public_capability": evaluator_manifest,
        "source_assets": {
            "frozen_dynamic_geometry": {"path": str(frozen_path.resolve()), "sha256": _sha256(frozen_path)},
            "dynamic_scan": {"path": str(DYNAMIC_SCAN_PATH.resolve()), "sha256": _sha256(DYNAMIC_SCAN_PATH)},
            "capability_evaluator": {"path": str(CAPABILITY_PATH.resolve()), "sha256": _sha256(CAPABILITY_PATH)},
        },
        "scope_notes": [
            "Geometry-only table using frozen surface/vessel anchors and one fixed registration.",
            "Finite public dIRD score and endpoint CapabilityEvaluator._static_pose witnesses only.",
            "edge_mask checks only three interior q interpolants plus finite endpoints for listed edges.",
            "No continuous Cartesian path/contact proof, force validation, anatomy/environment collision, dynamics, or hardware execution claim.",
            "The selected stations are common single-trunk rows with nominal center depth in 8--65 mm at every requested knee pose.",
        ],
    }
    metadata["output"] = str(output_path.resolve())
    metadata["audit_output"] = str(audit_path.resolve())
    arrays: dict[str, Any] = {
        "p_idx": np.asarray(p_idx, dtype=np.int64),
        "site_idx": np.asarray(site_idx, dtype=np.int64),
        "angles_deg": np.asarray(ANGLES_DEG, dtype=np.float64),
        "contact_xyz_m": np.asarray(contact, dtype=np.float64),
        "frames": np.asarray(frames, dtype=np.float64),
        "vessel_xyz_m": np.asarray(vessel, dtype=np.float64),
        "reference_frame": np.asarray(reference, dtype=np.float64),
        "valid": np.asarray(bank["valid"], dtype=bool),
        "q": np.asarray(bank["q"], dtype=np.float64),
        "edge_mask": np.asarray(edge_mask, dtype=bool),
        # Explicit diagnostics requested by the public capability contract.
        "dird_margin": np.asarray(bank["dird_margin"], dtype=np.float64),
        "proxy_pass": np.asarray(bank["proxy_pass"], dtype=bool),
        "rail_m": np.asarray(bank["rail_m"], dtype=np.float64),
        "static_found": np.asarray(bank["static_found"], dtype=bool),
        "nominal_depth_mm": np.asarray(selected_depth, dtype=np.float64),
        "target_robot": np.asarray(bank["targets"], dtype=np.float64),
        "metadata_json": np.asarray(json.dumps(_jsonable(metadata), sort_keys=True, separators=(",", ":"))),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(_jsonable(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, default=FROZEN_PATH)
    parser.add_argument("--output", type=Path, default=HERE / "TILT_BANK.npz")
    parser.add_argument("--audit", type=Path, default=HERE / "TILT_BANK.json")
    parser.add_argument("--no-static", action="store_true", help="geometry/proxy draft only; not final finite IK")
    parser.add_argument("--no-edges", action="store_true", help="skip sampled q transition checks")
    args = parser.parse_args(argv)
    metadata = build_bank(frozen_path=args.frozen, output_path=args.output, audit_path=args.audit,
                          static=not args.no_static, edges=not args.no_edges)
    print(json.dumps({
        "status": "ok",
        "output": metadata["output"],
        "audit_output": metadata["audit_output"],
        "rows": metadata["selected_row_count"],
        "sites_per_pose": metadata["selected_site_count_per_pose"],
        "angles_deg": metadata["angles_deg"],
        "valid": metadata["robot_target_checks"]["valid_count"],
        "total": metadata["robot_target_checks"]["total_count"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
