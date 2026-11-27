"""Build bounded long-leg sensor views and geometry-only public robot checks.

This experiment deliberately keeps two things separate:

* the complete ordered path in ``LONG_LEG_ASSETS.npz`` (all 234 original
  physical vessel/surface rows), and
* a finite view table sampled at approximately ``station_step_mm`` for which
  the public dIRD score and finite endpoint IK are evaluated.

The view table uses only the sealed full retained contact surface for skin
queries.  Lateral candidates are projected explicitly through
``SurfaceMaterialAtlas.query(..., project=True)`` and decoded back through the
same chart; a projection residual is always retained.  The strict repaired
tetra inverse in ``LONG_LEG_ASSETS`` is copied as provenance only: invalid
rows remain invalid and are never projected into the deep field.

Robot checks are public geometry-only checks.  The configured public anchor,
kinematics, dIRD field and finite ``CapabilityEvaluator._static_pose`` endpoint
search are used; no anatomy collision, contact, dynamics, path or native
control claim is made.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ASSET_PATH = HERE / "LONG_LEG_ASSETS.npz"
SURFACE_PATH = (
    HERE.parent / "ab_validation_20261110" / "repair_20261110"
    / "harmonic_full_domain_followup" / "contact_multichart_right.npz"
)
CAPABILITY_DIR = HERE.parent / "ab_module_ladder_20261110" / "capability"
PLACEMENT_PATH = (
    CAPABILITY_DIR.parent / "planning" / "native_regression_20261110"
    / "public_prior_layout" / "SELECTED.json"
)
MM_PER_M = 1000.0
POSE_DEG = 0
LATERAL_OFFSETS_MM = np.asarray([-16.0, -8.0, 0.0, 8.0, 16.0], dtype=float)
STATION_STEP_MM = 15.0
PUBLIC_RAIL_GRID_M = np.linspace(0.0, 0.8, 5)


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


def _load_surface_atlas(path: Path):
    """Load the actual contact ``SurfaceMaterialAtlas`` without modifying it."""

    # The production package is intentionally read-only here.  The source
    # checkout used by the validated contact asset is Among_US, not the
    # lightweight RealUS_playground checkout.
    source_root = Path("/media/camp/EXT_DRIVE/Among_US/src")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.shaft_rebuild import (  # type: ignore
        SurfaceMaterialAtlas,
    )

    return SurfaceMaterialAtlas.load(path)


def _station_rows(path_s_m: np.ndarray, step_mm: float) -> np.ndarray:
    """Select original rows nearest a regular arc-length grid, including ends."""

    if len(path_s_m) < 2 or np.any(np.diff(path_s_m) < -1.0e-12):
        raise ValueError("path arc length must be finite and nondecreasing")
    if not np.isfinite(path_s_m).all() or step_mm <= 0.0:
        raise ValueError("invalid station arc length or spacing")
    total_mm = float(path_s_m[-1] * MM_PER_M)
    targets = np.r_[np.arange(0.0, total_mm, float(step_mm)), total_mm]
    rows = np.asarray([int(np.argmin(np.abs(path_s_m * MM_PER_M - target))) for target in targets], dtype=int)
    rows = np.unique(np.r_[0, rows, len(path_s_m) - 1])
    return rows.astype(np.int64)


def _nonzero_tangent(points: np.ndarray, arc_s_m: np.ndarray) -> np.ndarray:
    """Finite arc-length tangent with duplicate-row protection."""

    p = np.asarray(points, dtype=float)
    s = np.asarray(arc_s_m, dtype=float)
    result = np.empty_like(p)
    for i in range(len(p)):
        left = i - 1
        while left >= 0 and s[i] - s[left] <= 1.0e-12:
            left -= 1
        right = i + 1
        while right < len(p) and s[right] - s[i] <= 1.0e-12:
            right += 1
        if left >= 0 and right < len(p):
            direction = p[right] - p[left]
        elif right < len(p):
            direction = p[right] - p[i]
        elif left >= 0:
            direction = p[i] - p[left]
        else:
            raise ValueError("path contains no nonzero segment")
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-12:
            raise ValueError(f"zero tangent at path row {i}")
        result[i] = direction / norm
    return result


def _frames(skin_m: np.ndarray, vessel_m: np.ndarray, arc_s_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return orthonormal (long, lateral, inward) frames and physical depths."""

    long_axis = _nonzero_tangent(skin_m, arc_s_m)
    raw_inward = np.asarray(vessel_m, dtype=float) - np.asarray(skin_m, dtype=float)
    raw_norm = np.linalg.norm(raw_inward, axis=1)
    if np.any(raw_norm <= 1.0e-10):
        bad = int(np.flatnonzero(raw_norm <= 1.0e-10)[0])
        raise ValueError(f"surface-to-vessel vector is degenerate at path row {bad}")
    raw_inward /= raw_norm[:, None]
    out = np.empty((len(skin_m), 3, 3), dtype=float)
    for i, tangent in enumerate(long_axis):
        # Start with the actual vessel direction, then make a true frame.  A
        # possible longitudinal component is removed, but orientation is
        # retained by the positive dot test below.
        inward = raw_inward[i] - tangent * float(raw_inward[i] @ tangent)
        inward_norm = float(np.linalg.norm(inward))
        if inward_norm <= 1.0e-8:
            # This is only a geometric fallback for a nearly tangent vessel;
            # it is still required to point toward the actual vessel.
            inward = raw_inward[i].copy()
            inward_norm = float(np.linalg.norm(inward))
        inward /= inward_norm
        lateral = np.cross(inward, tangent)
        lateral_norm = float(np.linalg.norm(lateral))
        if lateral_norm <= 1.0e-8:
            raise ValueError(f"no lateral direction at path row {i}")
        lateral /= lateral_norm
        # Keep signs continuous while enforcing positive inward orientation.
        if i and float(lateral @ out[i - 1, :, 1]) < 0.0:
            lateral *= -1.0
        inward = np.cross(tangent, lateral)
        inward /= float(np.linalg.norm(inward))
        if float(inward @ raw_inward[i]) < 0.0:
            lateral *= -1.0
            inward = np.cross(tangent, lateral)
            inward /= float(np.linalg.norm(inward))
        out[i] = np.stack((tangent, lateral, inward), axis=1)
    # Exact frame and physical-depth checks are useful to downstream callers.
    gram = np.einsum("nij,njk->nik", out.transpose(0, 2, 1), out)
    if not np.allclose(gram, np.eye(3), atol=1.0e-8):
        raise ValueError("station frames are not orthonormal")
    depth_mm = np.einsum("ni,ni->n", raw_inward * raw_norm[:, None], out[:, :, 2]) * MM_PER_M
    if np.any(depth_mm <= 0.0):
        bad = int(np.flatnonzero(depth_mm <= 0.0)[0])
        raise ValueError(f"inward-positive physical depth failed at row {bad}")
    return out, depth_mm


@dataclass(frozen=True)
class SurfaceViews:
    points_m: np.ndarray
    actual_lat_mm: np.ndarray
    projection_residual_m: np.ndarray
    projection_feasible: np.ndarray
    chart_ids: np.ndarray
    coordinates_uv_m: np.ndarray
    face_ids: np.ndarray
    barycentric: np.ndarray


def _surface_views(
    surface: Any,
    skin_m: np.ndarray,
    frames: np.ndarray,
    *,
    offsets_mm: np.ndarray = LATERAL_OFFSETS_MM,
    projection_tolerance_mm: float = 5.0,
) -> SurfaceViews:
    """Explicitly project and decode lateral candidates with the full atlas."""

    n, k = len(skin_m), len(offsets_mm)
    lateral = frames[:, :, 1]
    desired = skin_m[:, None, :] + (offsets_mm[None, :, None] / MM_PER_M) * lateral[:, None, :]
    query = surface.query(desired.reshape(-1, 3), project=True)
    if len(query) != n * k:
        raise RuntimeError("SurfaceMaterialAtlas returned the wrong query count")
    points = np.empty((n * k, 3), dtype=float)
    chart_ids = np.empty(n * k, dtype=np.int64)
    coordinates = np.empty((n * k, 2), dtype=float)
    face_ids = np.empty(n * k, dtype=np.int64)
    bary = np.empty((n * k, 3), dtype=float)
    residual = np.empty(n * k, dtype=float)
    for i, item in enumerate(query):
        chart_ids[i] = int(item["chart_id"])
        coordinates[i] = np.asarray(item["coordinate_uv_m"], dtype=float)
        face_ids[i] = int(item["source_face_id"])
        bary[i] = np.asarray(item["barycentric"], dtype=float)
        residual[i] = float(item["projection_distance_m"])
        points[i] = np.asarray(surface.decode(int(chart_ids[i]), coordinates[i])[0], dtype=float)
    points = points.reshape(n, k, 3)
    chart_ids = chart_ids.reshape(n, k)
    coordinates = coordinates.reshape(n, k, 2)
    face_ids = face_ids.reshape(n, k)
    bary = bary.reshape(n, k, 3)
    residual = residual.reshape(n, k)
    actual_lat = np.einsum("nki,ni->nk", points - skin_m[:, None, :], lateral) * MM_PER_M
    feasible = np.isfinite(residual) & (residual <= float(projection_tolerance_mm) / MM_PER_M)
    return SurfaceViews(points, actual_lat, residual, feasible, chart_ids, coordinates, face_ids, bary)


def _make_transform(frame: np.ndarray, point_m: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = frame
    transform[:3, 3] = point_m
    return transform


def _robot_views(
    points_m: np.ndarray,
    frames: np.ndarray,
    *,
    anchor: np.ndarray,
    q_seed: np.ndarray,
    capability: Any,
    static: bool = True,
) -> dict[str, np.ndarray | dict[str, Any]]:
    """Score and exactly check finite robot candidates at each sensor view."""

    n, k = points_m.shape[:2]
    if anchor.shape != (4, 4) or q_seed.shape != (8,):
        raise ValueError("public anchor or q seed has an invalid shape")
    # Align the path midpoint/view-zero with the previously audited public
    # TCP anchor.  The leg's full long axis is therefore a rigid public-body
    # placement; no anatomy is used by the robot evaluator.
    mid = n // 2
    T_atlas_mid = _make_transform(frames[mid], points_m[mid, 2])
    robot_from_atlas = anchor @ np.linalg.inv(T_atlas_mid)
    T_atlas = np.empty((n, k, 4, 4), dtype=float)
    for i in range(n):
        for j in range(k):
            T_atlas[i, j] = _make_transform(frames[i], points_m[i, j])
    target_robot = np.einsum("ab,nkbc->nkac", robot_from_atlas, T_atlas)
    # q[0] is the public physical rail coordinate.  The grid covers the
    # requested [0, .8] m range; each view additionally gets three rails
    # suggested from its body-y displacement relative to the anchor.
    displacement_y = target_robot[:, :, 1, 3] - float(anchor[1, 3])
    rails = np.empty((n, k, len(PUBLIC_RAIL_GRID_M) + 3), dtype=float)
    for i in range(n):
        for j in range(k):
            suggested = float(q_seed[0]) + float(displacement_y[i, j]) + np.asarray([-0.1, 0.0, 0.1])
            rails[i, j] = np.clip(np.r_[PUBLIC_RAIL_GRID_M, suggested], 0.0, 0.8)
    c = rails.shape[-1]
    flat_targets = np.broadcast_to(target_robot[:, :, None], (n, k, c, 4, 4)).reshape(-1, 4, 4)
    flat_rails = rails.reshape(-1)
    scores = capability.score(flat_targets, flat_rails)
    margin = np.asarray(scores["raw_calibrated_margin"], dtype=float).reshape(n, k, c)
    proxy_pass = np.asarray(scores["proxy_pass"], dtype=bool).reshape(n, k, c)
    static_found = np.zeros((n, k, c), dtype=bool)
    q_candidate = np.full((n, k, c, 8), np.nan, dtype=float)
    static_seconds = np.zeros((n, k, c), dtype=float)
    static_checks: list[dict[str, Any]] = []
    if static:
        for i in range(n):
            for j in range(k):
                for ci in range(c):
                    import time

                    started = time.perf_counter()
                    witness = capability._static_pose(target_robot[i, j], float(rails[i, j, ci]), q_seed)
                    static_seconds[i, j, ci] = time.perf_counter() - started
                    static_found[i, j, ci] = bool(witness.get("found", False))
                    if static_found[i, j, ci]:
                        q_candidate[i, j, ci] = np.asarray(witness["q_solution"], dtype=float)
                    static_checks.append(
                        {
                            "row": i,
                            "view": j,
                            "candidate": ci,
                            "rail_m": float(rails[i, j, ci]),
                            "found": bool(static_found[i, j, ci]),
                            "status": str(witness.get("status", "unknown")),
                            "seconds": float(static_seconds[i, j, ci]),
                        }
                    )
    else:
        static_checks.append({"status": "not_run", "scope": "score_only"})
    # Prefer a finite exact endpoint; within that set, retain the highest dIRD
    # margin.  If no finite endpoint is found, retain the highest score but
    # leave q_solution NaN and static_found false.
    rank_margin = np.where(static_found, margin, -np.inf) if static else margin
    chosen = np.argmax(rank_margin, axis=2)
    any_found = np.any(static_found, axis=2)
    if static:
        fallback = np.argmax(margin, axis=2)
        chosen = np.where(any_found, chosen, fallback)
    ii = np.arange(n)[:, None]
    jj = np.arange(k)[None, :]
    chosen_margin = margin[ii, jj, chosen]
    chosen_found = static_found[ii, jj, chosen]
    chosen_q = q_candidate[ii, jj, chosen]
    return {
        "robot_from_atlas": robot_from_atlas,
        "target_robot": target_robot,
        "candidate_rail_m": rails,
        "candidate_dird_margin": margin,
        "candidate_proxy_pass": proxy_pass,
        "candidate_static_found": static_found,
        "candidate_q_solution": q_candidate,
        "candidate_static_seconds": static_seconds,
        "chosen_candidate_index": chosen.astype(np.int64),
        "chosen_rail_m": rails[ii, jj, chosen],
        "chosen_dird_margin": chosen_margin,
        "chosen_static_found": chosen_found,
        "chosen_q_solution": chosen_q,
        "candidate_static_checks": static_checks,
        "dird_raw_threshold": float(scores["raw_threshold"]),
    }


def _load_assets(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _path_metadata(asset_meta: dict[str, Any], input_path: Path, *, row_slice: tuple[int, int] | None) -> dict[str, Any]:
    return {
        "schema": "long_leg_views_v1",
        "status": "geometry_and_public_robot_static_ik",
        "pose_deg": POSE_DEG,
        "pose_scope": "canonical fixed pose; no simulated tissue/probe dynamics",
        "station_step_requested_mm": STATION_STEP_MM,
        "station_selection": "nearest original LONG_LEG_ASSETS row to arc-length grid, always including run ends",
        "lateral_offsets_nominal_mm": LATERAL_OFFSETS_MM.tolist(),
        "frame_columns": ["longitudinal_tangent", "lateral", "inward"],
        "lateral_sign": "cross(inward,longitudinal), sign continuity; inward dot(actual vessel-skin) > 0",
        "surface_projection": {
            "asset": str(SURFACE_PATH.resolve()),
            "contract": "full retained right-leg contact multichart d=0 surface",
            "query": "SurfaceMaterialAtlas.query(project=True) followed by exact decode(chart_id, coordinate_uv_m)",
            "projection_tolerance_mm": 5.0,
            "residual_is_retained": True,
        },
        "internal_scope": "internal_valid/internal_xi copied from strict repaired atlas inverse; invalid rows remain NaN",
        "robot_scope": "public CapabilityEvaluator score + finite _static_pose endpoint; no anatomy collision/path/native claim",
        "input_asset": {"path": str(input_path.resolve()), "sha256": _sha256(input_path)},
        "input_asset_metadata": asset_meta,
        "row_slice": None if row_slice is None else [int(row_slice[0]), int(row_slice[1])],
    }


def build_views(
    *,
    assets_path: Path = ASSET_PATH,
    surface_path: Path = SURFACE_PATH,
    output_path: Path,
    audit_path: Path,
    row_slice: tuple[int, int] | None = None,
    station_step_mm: float = STATION_STEP_MM,
    static: bool = True,
) -> dict[str, Any]:
    """Build one bounded output (full path or the repaired 106--220 run)."""

    if not assets_path.is_file():
        raise FileNotFoundError(assets_path)
    if not surface_path.is_file():
        raise FileNotFoundError(surface_path)
    assets = _load_assets(assets_path)
    n_all = len(assets["vessel_m"])
    if row_slice is None:
        lo, hi = 0, n_all - 1
    else:
        lo, hi = map(int, row_slice)
        if not (0 <= lo <= hi < n_all):
            raise ValueError(f"invalid row slice {row_slice} for {n_all} rows")
    rows_all = np.arange(lo, hi + 1, dtype=np.int64)
    vessel_all = np.asarray(assets["vessel_m"], dtype=float)[rows_all]
    skin_all = np.asarray(assets["skin_m"], dtype=float)[rows_all]
    if not np.isfinite(vessel_all).all() or not np.isfinite(skin_all).all():
        raise ValueError("LONG_LEG_ASSETS path contains nonfinite physical coordinates")
    arc_s_all = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(vessel_all, axis=0), axis=1))]
    station_local = _station_rows(arc_s_all, station_step_mm)
    rows = rows_all[station_local]
    vessel = vessel_all[station_local]
    skin = skin_all[station_local]
    arc_s = arc_s_all[station_local]
    # Frames are first computed for all source rows, then selected, so station
    # tangents at the ends are not biased by the 15-mm row decimation.
    frames_all, depth_all_mm = _frames(skin_all, vessel_all, arc_s_all)
    frames = frames_all[station_local]
    vessel_depth_mm = depth_all_mm[station_local]
    surface = _load_surface_atlas(surface_path)
    views = _surface_views(surface, skin, frames)
    with np.load(assets_path, allow_pickle=False) as source:
        asset_meta_raw = json.loads(str(source["metadata_json"].item())) if "metadata_json" in source.files else {}
    # Capability imports torch and the public robot dependencies.  Fail early
    # with a clear message rather than emitting an unqualified geometry-only
    # file when the requested dIRD/static audit cannot be run.
    sys.path.insert(0, str(CAPABILITY_DIR))
    from capability import CapabilityEvaluator  # type: ignore

    if not PLACEMENT_PATH.is_file():
        raise FileNotFoundError(PLACEMENT_PATH)
    placement = json.loads(PLACEMENT_PATH.read_text(encoding="utf-8"))
    q_seed = np.asarray(placement["q_seed"], dtype=float)
    anchor = np.asarray(placement["T_world_body"], dtype=float)
    capability = CapabilityEvaluator()
    robot = _robot_views(views.points_m, frames, anchor=anchor, q_seed=q_seed, capability=capability, static=static)

    # The public minimum schema uses the chosen endpoint fields directly.
    metadata = _path_metadata(asset_meta_raw, assets_path, row_slice=row_slice)
    metadata.update(
        {
            "source_row_count": int(len(rows_all)),
            "station_count": int(len(rows)),
            "station_row_ids": rows.tolist(),
            "station_s_mm": (arc_s * MM_PER_M).tolist(),
            "source_path_length_m": float(arc_s_all[-1]),
            "source_path_length_mm": float(arc_s_all[-1] * MM_PER_M),
            "source_label_span": [str(assets["labels"][lo]), str(assets["labels"][hi])],
            "view_projection_residual_max_mm": float(np.max(views.projection_residual_m) * MM_PER_M),
            "view_projection_residual_median_mm": float(np.median(views.projection_residual_m) * MM_PER_M),
            "view_projection_within_tolerance_count": int(np.sum(views.projection_feasible)),
            "view_projection_total_count": int(views.projection_feasible.size),
            "vessel_depth_mm": {
                "min": float(np.min(vessel_depth_mm)),
                "median": float(np.median(vessel_depth_mm)),
                "max": float(np.max(vessel_depth_mm)),
            },
            "public_robot": {
                "placement": str(PLACEMENT_PATH.resolve()),
                "placement_sha256": _sha256(PLACEMENT_PATH),
                "q_seed": q_seed.tolist(),
                "anchor": anchor.tolist(),
                "evaluator_manifest": _jsonable(capability.manifest()),
                "candidate_count_per_view": int(robot["candidate_rail_m"].shape[-1]),
                "static_requested": bool(static),
                "chosen_static_found_count": int(np.sum(robot["chosen_static_found"])),
                "chosen_view_count": int(robot["chosen_static_found"].size),
                "candidate_static_found_count": int(np.sum(robot["candidate_static_found"])),
                "candidate_total_count": int(robot["candidate_static_found"].size),
                "dird_margin_chosen_min": float(np.min(robot["chosen_dird_margin"])),
                "dird_margin_chosen_max": float(np.max(robot["chosen_dird_margin"])),
                "scope_note": "score_is_safety_certificate=False; endpoint IK is a finite search, not an infeasibility proof",
            },
        }
    )
    # Keep complete original path rows attached for FULL_LEG_VIEWS consumers,
    # while the main view arrays remain at the requested ~15-mm station rate.
    arrays: dict[str, Any] = {
        "row_ids": rows,
        "s_mm": arc_s * MM_PER_M,
        "skin_m": skin,
        "vessel_m": vessel,
        "frames": frames,
        "vessel_depth_mm": vessel_depth_mm,
        "views_m": views.points_m,
        "view_lat_mm": views.actual_lat_mm,
        "view_projection_residual_m": views.projection_residual_m,
        "view_projection_feasible": views.projection_feasible,
        "view_chart_ids": views.chart_ids,
        "view_coordinates_uv_m": views.coordinates_uv_m,
        "view_face_ids": views.face_ids,
        "view_face": views.face_ids,
        "view_barycentric": views.barycentric,
        "view_bary": views.barycentric,
        "internal_valid": np.asarray(assets["internal_valid"], dtype=bool)[rows],
        "internal_xi": np.asarray(assets["internal_xi"], dtype=float)[rows],
        "internal_residual_m": np.asarray(assets["internal_residual_m"], dtype=float)[rows],
        "internal_tet_index": np.asarray(assets["internal_tet_index"], dtype=np.int64)[rows],
        "labels": np.asarray(assets["labels"])[rows].astype(str),
        "chart_ids": np.asarray(assets["chart_ids"], dtype=np.int64)[rows],
        "coordinates_uv_m": np.asarray(assets["coordinates_uv_m"], dtype=float)[rows],
        "dird_margin": np.asarray(robot["chosen_dird_margin"], dtype=float),
        "static_found": np.asarray(robot["chosen_static_found"], dtype=bool),
        "q_solution": np.asarray(robot["chosen_q_solution"], dtype=float),
        "chosen_candidate_index": np.asarray(robot["chosen_candidate_index"], dtype=np.int64),
        "chosen_rail_m": np.asarray(robot["chosen_rail_m"], dtype=float),
        "candidate_rail_m": np.asarray(robot["candidate_rail_m"], dtype=float),
        "candidate_q0": np.asarray(robot["candidate_rail_m"], dtype=float),
        "candidate_dird_margin": np.asarray(robot["candidate_dird_margin"], dtype=float),
        "candidate_proxy_pass": np.asarray(robot["candidate_proxy_pass"], dtype=bool),
        "candidate_static_found": np.asarray(robot["candidate_static_found"], dtype=bool),
        "candidate_q_solution": np.asarray(robot["candidate_q_solution"], dtype=float),
        "candidate_static_seconds": np.asarray(robot["candidate_static_seconds"], dtype=float),
        "robot_from_atlas": np.asarray(robot["robot_from_atlas"], dtype=float),
        "target_robot": np.asarray(robot["target_robot"], dtype=float),
        "source_row_ids": rows_all,
        "source_s_mm": arc_s_all * MM_PER_M,
        "source_skin_m": skin_all,
        "source_vessel_m": vessel_all,
        "source_internal_valid": np.asarray(assets["internal_valid"], dtype=bool)[rows_all],
        "source_internal_xi": np.asarray(assets["internal_xi"], dtype=float)[rows_all],
        "source_labels": np.asarray(assets["labels"])[rows_all].astype(str),
        "source_chart_ids": np.asarray(assets["chart_ids"], dtype=np.int64)[rows_all],
        "source_coordinates_uv_m": np.asarray(assets["coordinates_uv_m"], dtype=float)[rows_all],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, metadata_json=np.asarray(json.dumps(_jsonable(metadata), sort_keys=True)), **arrays)
    metadata["output"] = str(output_path.resolve())
    metadata["audit_output"] = str(audit_path.resolve())
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(_jsonable(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def build_checked_views(
    *,
    source_path: Path = HERE / "LONG_LEG_VIEWS.npz",
    output_path: Path = HERE / "LONG_LEG_VIEWS_CHECKED.npz",
    audit_path: Path = HERE / "LONG_LEG_VIEWS_CHECKED.json",
    surface_path: Path = SURFACE_PATH,
    static: bool = True,
) -> dict[str, Any]:
    """Recheck the long table at a public inward-axis yaw-π placement.

    The yaw is a probe-frame half-turn about the local inward axis.  It keeps
    the contact-facing direction unchanged while reversing the long/lateral
    directions, which is an allowed public TCP orientation option.  The
    original K=5 table is retained under ``original_*`` arrays and the new
    table adds the requested ±24-mm lateral candidates (K=7).
    """

    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    old = _load_assets(source_path)
    with np.load(source_path, allow_pickle=False) as source:
        old_meta = json.loads(str(source["metadata_json"].item()))
    n = len(old["row_ids"])
    if old["frames"].shape != (n, 3, 3) or old["skin_m"].shape != (n, 3):
        raise ValueError("source long view table has an unexpected geometry shape")
    checked_offsets = np.asarray([-24.0, -16.0, -8.0, 0.0, 8.0, 16.0, 24.0], dtype=float)
    surface = _load_surface_atlas(surface_path)
    checked_surface = _surface_views(surface, old["skin_m"], old["frames"], offsets_mm=checked_offsets)
    sys.path.insert(0, str(CAPABILITY_DIR))
    from capability import CapabilityEvaluator  # type: ignore

    placement = json.loads(PLACEMENT_PATH.read_text(encoding="utf-8"))
    q_seed = np.asarray(placement["q_seed"], dtype=float)
    anchor = np.asarray(placement["T_world_body"], dtype=float)
    # Public short-axis yaw π = rotation by 180 degrees around the local
    # inward/TCP-z axis.  The physical inward direction is preserved.
    yaw_pi = np.diag([-1.0, -1.0, 1.0])
    checked_anchor = anchor.copy()
    checked_anchor[:3, :3] = anchor[:3, :3] @ yaw_pi
    capability = CapabilityEvaluator()
    # Keep both public orientation options.  A station-wise choice based only
    # on endpoint geometry can use yaw-π at the proximal end while retaining
    # the original anchor elsewhere; no trial/anatomy observation enters this
    # selection.
    robot_base = _robot_views(
        checked_surface.points_m, old["frames"], anchor=anchor, q_seed=q_seed,
        capability=capability, static=static,
    )
    robot_yaw = _robot_views(
        checked_surface.points_m, old["frames"], anchor=checked_anchor, q_seed=q_seed,
        capability=capability, static=static,
    )
    robot: dict[str, Any] = {}
    for key in (
        "candidate_rail_m", "candidate_dird_margin", "candidate_proxy_pass",
        "candidate_static_found", "candidate_q_solution", "candidate_static_seconds",
    ):
        robot[key] = np.concatenate((np.asarray(robot_base[key]), np.asarray(robot_yaw[key])), axis=2)
    combined_margin = np.asarray(robot["candidate_dird_margin"], dtype=float)
    combined_static = np.asarray(robot["candidate_static_found"], dtype=bool)
    rank_margin = np.where(combined_static, combined_margin, -np.inf) if static else combined_margin
    chosen = np.argmax(rank_margin, axis=2)
    any_found = np.any(combined_static, axis=2)
    if static:
        chosen = np.where(any_found, chosen, np.argmax(combined_margin, axis=2))
    ii = np.arange(n)[:, None]
    jj = np.arange(checked_surface.points_m.shape[1])[None, :]
    robot["chosen_candidate_index"] = chosen.astype(np.int64)
    robot["chosen_rail_m"] = np.asarray(robot["candidate_rail_m"])[ii, jj, chosen]
    robot["chosen_dird_margin"] = combined_margin[ii, jj, chosen]
    robot["chosen_static_found"] = combined_static[ii, jj, chosen]
    robot["chosen_q_solution"] = np.asarray(robot["candidate_q_solution"])[ii, jj, chosen]
    robot["candidate_placement_id"] = np.concatenate(
        (np.zeros_like(np.asarray(robot_base["candidate_rail_m"]), dtype=np.int64),
         np.ones_like(np.asarray(robot_yaw["candidate_rail_m"]), dtype=np.int64)), axis=2
    )
    robot["placement_id"] = np.asarray(robot["candidate_placement_id"])[ii, jj, chosen]
    robot["rotation_deg"] = np.where(robot["placement_id"] == 0, 0.0, 180.0)
    robot["target_robot_options"] = np.stack(
        (np.asarray(robot_base["target_robot"]), np.asarray(robot_yaw["target_robot"])), axis=2
    )
    robot["target_robot_chosen"] = np.where(
        robot["placement_id"][..., None, None] == 0,
        np.asarray(robot_base["target_robot"]), np.asarray(robot_yaw["target_robot"])
    )
    robot["robot_from_atlas_options"] = np.stack(
        (np.asarray(robot_base["robot_from_atlas"]), np.asarray(robot_yaw["robot_from_atlas"])), axis=0
    )
    robot["robot_from_atlas_chosen"] = np.where(
        robot["placement_id"][..., None, None] == 0,
        np.broadcast_to(np.asarray(robot_base["robot_from_atlas"]), (n, checked_surface.points_m.shape[1], 4, 4)),
        np.broadcast_to(np.asarray(robot_yaw["robot_from_atlas"]), (n, checked_surface.points_m.shape[1], 4, 4)),
    )
    metadata = dict(old_meta)
    metadata.update(
        {
            "schema": "long_leg_views_checked_v1",
            "status": "geometry_and_public_robot_static_ik_checked_public_inward_yaw_pi",
            "source_original_table": {"path": str(source_path.resolve()), "sha256": _sha256(source_path)},
            "checked_lateral_offsets_nominal_mm": checked_offsets.tolist(),
            "checked_public_placement": {
                "base_placement": str(PLACEMENT_PATH.resolve()),
                "base_anchor": anchor.tolist(),
                "rotation_local_axis": "inward/TCP-z",
                "rotation_deg": 180.0,
                "checked_anchor": checked_anchor.tolist(),
                "interpretation": "contact-facing inward axis unchanged; long/lateral axes reversed",
            },
            "station_count": int(n),
            "view_projection_residual_max_mm": float(np.max(checked_surface.projection_residual_m) * MM_PER_M),
            "view_projection_residual_median_mm": float(np.median(checked_surface.projection_residual_m) * MM_PER_M),
            "view_projection_within_tolerance_count": int(np.sum(checked_surface.projection_feasible)),
            "view_projection_total_count": int(checked_surface.projection_feasible.size),
            "public_robot": {
                "placement_rotation": "public base anchor plus optional local inward/TCP-z yaw π",
                "static_requested": bool(static),
                "candidate_count_per_view": int(robot["candidate_rail_m"].shape[-1]),
                "chosen_static_found_count": int(np.sum(robot["chosen_static_found"])),
                "chosen_view_count": int(robot["chosen_static_found"].size),
                "candidate_static_found_count": int(np.sum(robot["candidate_static_found"])),
                "candidate_total_count": int(robot["candidate_static_found"].size),
                "dird_margin_chosen_min": float(np.min(robot["chosen_dird_margin"])),
                "dird_margin_chosen_max": float(np.max(robot["chosen_dird_margin"])),
                "chosen_base_anchor_count": int(np.sum(np.asarray(robot["placement_id"]) == 0)),
                "chosen_inward_yaw_pi_count": int(np.sum(np.asarray(robot["placement_id"]) == 1)),
                "evaluator_manifest": _jsonable(capability.manifest()),
            },
        }
    )
    arrays = dict(old)
    arrays.pop("metadata_json", None)
    # Explicitly retain the original geometry/robot table rather than relying
    # on a hidden file lookup when reviewers compare the two placements.
    for key in (
        "views_m", "view_lat_mm", "view_projection_residual_m", "view_projection_feasible",
        "view_chart_ids", "view_coordinates_uv_m", "view_face_ids", "view_face",
        "view_barycentric", "view_bary", "dird_margin", "static_found", "q_solution",
        "chosen_candidate_index", "chosen_rail_m", "candidate_rail_m", "candidate_q0",
        "candidate_dird_margin", "candidate_proxy_pass", "candidate_static_found",
        "candidate_q_solution", "candidate_static_seconds", "robot_from_atlas", "target_robot",
    ):
        if key in arrays:
            arrays[f"original_{key}"] = np.asarray(arrays[key]).copy()
    arrays.update(
        {
            "views_m": checked_surface.points_m,
            "view_lat_mm": checked_surface.actual_lat_mm,
            "view_projection_residual_m": checked_surface.projection_residual_m,
            "view_projection_feasible": checked_surface.projection_feasible,
            "view_chart_ids": checked_surface.chart_ids,
            "view_coordinates_uv_m": checked_surface.coordinates_uv_m,
            "view_face_ids": checked_surface.face_ids,
            "view_face": checked_surface.face_ids,
            "view_barycentric": checked_surface.barycentric,
            "view_bary": checked_surface.barycentric,
            "dird_margin": np.asarray(robot["chosen_dird_margin"], dtype=float),
            "static_found": np.asarray(robot["chosen_static_found"], dtype=bool),
            "q_solution": np.asarray(robot["chosen_q_solution"], dtype=float),
            "chosen_candidate_index": np.asarray(robot["chosen_candidate_index"], dtype=np.int64),
            "chosen_rail_m": np.asarray(robot["chosen_rail_m"], dtype=float),
            "candidate_rail_m": np.asarray(robot["candidate_rail_m"], dtype=float),
            "candidate_q0": np.asarray(robot["candidate_rail_m"], dtype=float),
            "candidate_dird_margin": np.asarray(robot["candidate_dird_margin"], dtype=float),
            "candidate_proxy_pass": np.asarray(robot["candidate_proxy_pass"], dtype=bool),
            "candidate_static_found": np.asarray(robot["candidate_static_found"], dtype=bool),
            "candidate_q_solution": np.asarray(robot["candidate_q_solution"], dtype=float),
            "candidate_static_seconds": np.asarray(robot["candidate_static_seconds"], dtype=float),
            "candidate_placement_id": np.asarray(robot["candidate_placement_id"], dtype=np.int64),
            "chosen_placement_id": np.asarray(robot["placement_id"], dtype=np.int64),
            "chosen_rotation_deg": np.asarray(robot["rotation_deg"], dtype=float),
            "robot_from_atlas_options": np.asarray(robot["robot_from_atlas_options"], dtype=float),
            "robot_from_atlas": np.asarray(robot["robot_from_atlas_chosen"], dtype=float),
            "candidate_target_robot": np.asarray(robot["target_robot_options"], dtype=float),
            "target_robot": np.asarray(robot["target_robot_chosen"], dtype=float),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, metadata_json=np.asarray(json.dumps(_jsonable(metadata), sort_keys=True)), **arrays)
    metadata["output"] = str(output_path.resolve())
    metadata["audit_output"] = str(audit_path.resolve())
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(_jsonable(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ASSET_PATH)
    parser.add_argument("--surface", type=Path, default=SURFACE_PATH)
    parser.add_argument("--station-step-mm", type=float, default=STATION_STEP_MM)
    parser.add_argument("--no-static", action="store_true", help="score only; never use for the requested final audit")
    parser.add_argument("--only", choices=("full", "long", "both", "checked"), default="both")
    args = parser.parse_args(argv)
    outputs = []
    if args.only in ("long", "both"):
        metadata = build_views(
            assets_path=args.assets,
            surface_path=args.surface,
            output_path=HERE / "LONG_LEG_VIEWS.npz",
            audit_path=HERE / "LONG_LEG_VIEWS.json",
            row_slice=(106, 220),
            station_step_mm=args.station_step_mm,
            static=not args.no_static,
        )
        outputs.append({"name": "long", "output": metadata["output"], "stations": metadata["station_count"]})
    if args.only in ("full", "both"):
        metadata = build_views(
            assets_path=args.assets,
            surface_path=args.surface,
            output_path=HERE / "FULL_LEG_VIEWS.npz",
            audit_path=HERE / "FULL_LEG_VIEWS.json",
            row_slice=None,
            station_step_mm=args.station_step_mm,
            static=not args.no_static,
        )
        outputs.append({"name": "full", "output": metadata["output"], "stations": metadata["station_count"]})
    if args.only == "checked":
        metadata = build_checked_views(static=not args.no_static)
        outputs.append({"name": "checked", "output": metadata["output"], "stations": metadata["station_count"]})
    print(json.dumps({"status": "ok", "outputs": outputs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

