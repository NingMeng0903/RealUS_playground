"""Bounded dynamic right-leg geometry and public robot candidate table.

This module is intentionally geometry-first.  It reuses the frozen public
long-leg table, keeps one atlas-to-robot registration for every pose, and
transports fixed real-skin triangle/barycentric and vessel-tetra/barycentric
anchors through the original SMPL-X LBS bridge.  The finite robot checks are
the same public dIRD score and endpoint ``CapabilityEvaluator._static_pose``
search used by :mod:`long_leg_views`; no anatomy collision, continuous path,
force, or hardware claim is made.

The output is a compact candidate table for four bounded pose states:

``0, 11, 15, 19`` degrees at the right SMPL-X knee (canonical, a modest
15-degree bend, and two +/-4-degree dynamic knots around that bend).  The
contact window retains the seven original 48-mm lateral offsets and two
central +/-12-degree tilt alternatives.  The 50-mm value in metadata is the
modeled total FOV; rail/contact reach is recorded separately.

The script needs the existing genesis environment for torch and the public
robot evaluator.  Run it with e.g.::

    /media/camp/EXT_DRIVE/envs/genesis/bin/python dynamic_geometry.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "ab_leg_bifurcation_20261112"
FROZEN_PATH = BASE / "LONG_LEG_VIEWS_FINAL_FROZEN.npz"
SURFACE_PATH = (
    HERE.parent / "ab_validation_20261110" / "repair_20261110"
    / "harmonic_full_domain_followup" / "contact_multichart_right.npz"
)
KNEE_DIR = BASE / "knee_rebake"
ATLAS_PATH = KNEE_DIR / "atlas_right.npz"
ATTACHMENTS_PATH = KNEE_DIR / "attachments_right.npz"
VESSEL_REBIND_PATH = KNEE_DIR / "VESSEL_REBIND.npz"
BODY_DIR = Path("/media/camp/EXT_DRIVE/Among_US/outputs/anatomy_retarget/recovered_smplx_20261108")
BODY_WEIGHTS_PATH = BODY_DIR / "smpl_canonical_weights.npz"
SKELETON_PATH = BODY_DIR / "smpl_canonical_skeleton.json"
CAPABILITY_DIR = BASE.parent / "ab_module_ladder_20261110" / "capability"
PLACEMENT_PATH = (
    CAPABILITY_DIR.parent / "planning" / "native_regression_20261110"
    / "public_prior_layout" / "SELECTED.json"
)

OUTPUT_PATH = HERE / "DYNAMIC_GEOMETRY.npz"
AUDIT_PATH = HERE / "DYNAMIC_GEOMETRY.json"

MM_PER_M = 1000.0
POSE_KNEE_DEG = np.asarray([0.0, 11.0, 15.0, 19.0], dtype=float)
POSE_STATE_NAMES = np.asarray(
    ["canonical_0deg", "dynamic_knot_minus4deg", "modest_bend_15deg", "dynamic_knot_plus4deg"],
    dtype=str,
)
# First seven contacts are exactly the frozen table's inward-yaw-pi common
# placement window.  The last two are central-contact orientation alternatives.
CONTACT_OFFSETS_MM = np.asarray([-24.0, -16.0, -8.0, 0.0, 8.0, 16.0, 24.0, 0.0, 0.0], dtype=float)
CONTACT_TILTS_DEG = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -12.0, 12.0], dtype=float)
CONTACT_KIND = np.asarray(
    ["base_offset" for _ in range(7)] + ["central_tilt_minus12", "central_tilt_plus12"],
    dtype=str,
)
PUBLIC_RAIL_GRID_M = np.linspace(0.0, 0.8, 5)
RAIL_SUGGESTIONS_M = np.asarray([-0.1, 0.0, 0.1], dtype=float)
JOINT_COUNT = 55
RIGHT_KNEE_INDEX = 5


@dataclass(frozen=True)
class SourceBundle:
    frozen: dict[str, np.ndarray]
    frozen_meta: dict[str, Any]
    surface: dict[str, np.ndarray]
    atlas: dict[str, np.ndarray]
    attachments: dict[str, np.ndarray]
    vessel_rebind: dict[str, np.ndarray]
    skeleton: dict[str, Any]
    body: dict[str, np.ndarray]


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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _load_sources() -> SourceBundle:
    frozen = _load_npz(FROZEN_PATH)
    if "metadata_json" not in frozen:
        raise KeyError(f"{FROZEN_PATH} has no metadata_json")
    frozen_meta = json.loads(str(frozen["metadata_json"].item()))
    surface = _load_npz(SURFACE_PATH)
    atlas = _load_npz(ATLAS_PATH)
    attachments = _load_npz(ATTACHMENTS_PATH)
    vessel_rebind = _load_npz(VESSEL_REBIND_PATH)
    with SKELETON_PATH.open("r", encoding="utf-8") as handle:
        skeleton = json.load(handle)
    body = _load_npz(BODY_WEIGHTS_PATH)
    return SourceBundle(frozen, frozen_meta, surface, atlas, attachments, vessel_rebind, skeleton, body)


def _rotation_from_axis_angle(axis_angle: np.ndarray) -> np.ndarray:
    """Use the original anatomy pose API, keeping this fallback-free."""

    source_root = Path("/media/camp/EXT_DRIVE/Among_US/src")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import axis_angle_to_matrix  # type: ignore

    return np.asarray(axis_angle_to_matrix(axis_angle), dtype=np.float32)


def _joint_transforms(bundle: SourceBundle, knee_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the original SMPL-X global pose and LBS joint transforms."""

    source_root = Path("/media/camp/EXT_DRIVE/Among_US/src")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import joint_global_transforms  # type: ignore

    pose = np.zeros((JOINT_COUNT, 3), dtype=np.float32)
    pose[RIGHT_KNEE_INDEX, 0] = np.deg2rad(float(knee_deg))
    rest_joints = np.asarray(bundle.body["rest_joints"], dtype=np.float32).reshape(JOINT_COUNT, 3)
    parents = np.asarray(bundle.body["parents"], dtype=np.int32).reshape(JOINT_COUNT)
    inverse_bind = np.asarray(bundle.body["inverse_bind"], dtype=np.float32).reshape(JOINT_COUNT, 4, 4)
    global_tf = np.asarray(
        joint_global_transforms(pose_axis_angle=pose, rest_joints=rest_joints, parents=parents),
        dtype=np.float32,
    )
    lbs_tf = np.einsum("nij,njk->nik", global_tf, inverse_bind).astype(np.float32)
    return pose, lbs_tf


def _apply_lbs(
    points: np.ndarray,
    rotations: np.ndarray,
    weights: np.ndarray,
    joint_transforms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the original ``lbs_bridge.apply_lbs_pose`` API."""

    source_root = Path("/media/camp/EXT_DRIVE/Among_US/src")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.lbs_bridge import (  # type: ignore
        LbsKinematicState,
        apply_lbs_pose,
    )

    p = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    r = np.asarray(rotations, dtype=np.float32).reshape(-1, 3, 3)
    w = np.asarray(weights, dtype=np.float32).reshape(-1, JOINT_COUNT)
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1.0e-12)
    state = LbsKinematicState(joint_transforms=np.asarray(joint_transforms, dtype=np.float32))
    out = apply_lbs_pose(p, r, w, state)
    return np.asarray(out.points, dtype=np.float64), np.asarray(out.rotations, dtype=np.float64)


def _tilt_frame(frame: np.ndarray, tilt_deg: float) -> np.ndarray:
    """Rotate a frame about its local longitudinal/contact-window axis."""

    if abs(float(tilt_deg)) <= 1.0e-12:
        return np.asarray(frame, dtype=float).copy()
    axis = np.asarray(frame[:, 0], dtype=float)
    rv = axis * np.deg2rad(float(tilt_deg))
    rot = np.asarray(_rotation_from_axis_angle(rv.reshape(1, 3))[0], dtype=float)
    return rot @ np.asarray(frame, dtype=float)


def _surface_anchor_geometry(bundle: SourceBundle) -> dict[str, np.ndarray]:
    """Decode the frozen source triangle/barycentric contact anchors exactly."""

    f = bundle.frozen
    s = bundle.surface
    rows = np.asarray(f["row_ids"], dtype=np.int64)
    n = len(rows)
    k_base = len(CONTACT_OFFSETS_MM)
    if np.asarray(f["views_m"]).shape != (n, 7, 3):
        raise ValueError("frozen table must contain 7 lateral contacts")
    face_base = np.asarray(f["view_face_ids"], dtype=np.int64)
    bary_base = np.asarray(f["view_barycentric"], dtype=float)
    if face_base.shape != (n, 7) or bary_base.shape != (n, 7, 3):
        raise ValueError("frozen surface anchor shapes are not the expected (N,7,...) contract")
    surface_faces = np.asarray(s["faces"], dtype=np.int64)
    surface_vertices = np.asarray(s["vertices"], dtype=float)
    surface_weights = np.asarray(s["weights"], dtype=float)
    face_triangles = surface_faces[face_base]
    base_points = np.einsum("nki,nkij->nkj", bary_base, surface_vertices[face_triangles])
    base_weights = np.einsum("nki,nkij->nkj", bary_base, surface_weights[face_triangles])
    if not np.allclose(base_points, f["views_m"], atol=2.0e-8, rtol=0.0):
        raise ValueError("decoded frozen surface anchors do not reproduce views_m")
    # Extra tilt rows reuse the exact central offset's triangle/barycentric
    # anchor; only their contact orientation is varied.
    face_ids = np.concatenate((face_base, face_base[:, 3:4], face_base[:, 3:4]), axis=1)
    bary = np.concatenate((bary_base, bary_base[:, 3:4], bary_base[:, 3:4]), axis=1)
    points = np.concatenate((base_points, base_points[:, 3:4], base_points[:, 3:4]), axis=1)
    weights = np.concatenate((base_weights, base_weights[:, 3:4], base_weights[:, 3:4]), axis=1)
    frame_base = np.asarray(f["frames"], dtype=float)
    frames = np.empty((n, k_base, 3, 3), dtype=float)
    for i in range(n):
        for j, tilt in enumerate(CONTACT_TILTS_DEG):
            frames[i, j] = _tilt_frame(frame_base[i], float(tilt))
    return {
        "row_ids": rows,
        "points": points,
        "weights": weights,
        "face_ids": face_ids,
        "barycentric": bary,
        "frames": frames,
        "base_points": base_points,
        "base_weights": base_weights,
    }


def _vessel_anchor_geometry(bundle: SourceBundle) -> dict[str, np.ndarray]:
    """Decode knee-rebake tetrahedral anchors and their fixed volume weights."""

    f = bundle.frozen
    rb = bundle.vessel_rebind
    atlas = bundle.atlas
    att = bundle.attachments
    rows = np.asarray(f["row_ids"], dtype=np.int64)
    vessel = np.asarray(rb["vessel_m"], dtype=float)[rows]
    valid = np.asarray(rb["internal_valid"], dtype=bool)[rows]
    tet_indices = np.asarray(rb["internal_tet_index"], dtype=np.int64)[rows]
    if not np.all(valid):
        raise ValueError("dynamic table includes an invalid rebake vessel row")
    tets = np.asarray(atlas["harmonic_tets"], dtype=np.int64)[tet_indices]
    vertices = np.asarray(atlas["harmonic_vertices"], dtype=float)
    bary = np.empty((len(rows), 4), dtype=float)
    residual = np.empty(len(rows), dtype=float)
    for i, (point, tet) in enumerate(zip(vessel, tets, strict=True)):
        corners = vertices[tet]
        matrix = (corners[:3] - corners[3]).T
        first = np.linalg.solve(matrix, point - corners[3])
        bary[i, :3] = first
        bary[i, 3] = 1.0 - float(np.sum(first))
        residual[i] = float(np.linalg.norm(np.einsum("i,ij->j", bary[i], corners) - point))
    if np.min(bary) < -1.0e-6 or np.max(bary) > 1.0 + 1.0e-6:
        raise ValueError("vessel tetra barycentric coordinates escaped their rebake tetrahedra")
    volume_weights = np.asarray(att["volume_lbs_weights"], dtype=float)
    point_weights = np.einsum("ni,nij->nj", bary, volume_weights[tets])
    point_weights /= np.maximum(point_weights.sum(axis=1, keepdims=True), 1.0e-12)
    return {
        "rows": rows,
        "vessel": vessel,
        "internal_valid": valid,
        "internal_xi": np.asarray(rb["internal_xi"], dtype=float)[rows],
        "legacy_internal_xi": np.asarray(f["internal_xi"], dtype=float),
        "tet_indices": tet_indices,
        "tet_vertex_indices": tets,
        "barycentric": bary,
        "reconstruction_residual_m": residual,
        "weights": point_weights,
    }


def _rotation_metrics(rotations: np.ndarray) -> dict[str, float]:
    r = np.asarray(rotations, dtype=float).reshape(-1, 3, 3)
    gram = np.einsum("nji,njk->nik", r, r)
    return {
        "orthonormal_max_abs_error": float(np.max(np.abs(gram - np.eye(3)))) if len(r) else 0.0,
        "det_min": float(np.min(np.linalg.det(r))) if len(r) else 1.0,
        "det_max": float(np.max(np.linalg.det(r))) if len(r) else 1.0,
    }


def _build_pose_geometry(bundle: SourceBundle, surface: dict[str, np.ndarray], vessel: dict[str, np.ndarray]):
    n, k = surface["points"].shape[:2]
    states = len(POSE_KNEE_DEG)
    pose_axis_angle = np.zeros((states, JOINT_COUNT, 3), dtype=float)
    joint_transforms = np.empty((states, JOINT_COUNT, 4, 4), dtype=float)
    posed_views = np.empty((states, n, k, 3), dtype=float)
    posed_frames = np.empty((states, n, k, 3, 3), dtype=float)
    posed_vessel = np.empty((states, n, 3), dtype=float)
    surface_points_flat = surface["points"].reshape(-1, 3)
    surface_weights_flat = surface["weights"].reshape(-1, JOINT_COUNT)
    surface_frames_flat = surface["frames"].reshape(-1, 3, 3)
    vessel_points = vessel["vessel"]
    vessel_weights = vessel["weights"]
    # Identity pose is decoded by LBS too, making the canonical row a direct
    # check of the original body/pose API rather than a special-case copy.
    for state_i, angle in enumerate(POSE_KNEE_DEG):
        pose, transforms = _joint_transforms(bundle, float(angle))
        pose_axis_angle[state_i] = pose
        joint_transforms[state_i] = transforms
        p_surface, r_surface = _apply_lbs(
            surface_points_flat,
            surface_frames_flat,
            surface_weights_flat,
            transforms,
        )
        p_vessel, _r_vessel = _apply_lbs(
            vessel_points,
            np.repeat(np.eye(3, dtype=float)[None], n, axis=0),
            vessel_weights,
            transforms,
        )
        posed_views[state_i] = p_surface.reshape(n, k, 3)
        posed_frames[state_i] = r_surface.reshape(n, k, 3, 3)
        posed_vessel[state_i] = p_vessel
    return pose_axis_angle, joint_transforms, posed_views, posed_frames, posed_vessel


def _make_transform(frame: np.ndarray, point: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=float)
    out[:3, :3] = np.asarray(frame, dtype=float)
    out[:3, 3] = np.asarray(point, dtype=float)
    return out


def _fixed_robot_targets(
    robot_from_atlas: np.ndarray,
    posed_views: np.ndarray,
    posed_frames: np.ndarray,
) -> np.ndarray:
    states, n, k = posed_views.shape[:3]
    out = np.empty((states, n, k, 4, 4), dtype=float)
    for si in range(states):
        for i in range(n):
            for j in range(k):
                out[si, i, j] = robot_from_atlas @ _make_transform(
                    posed_frames[si, i, j], posed_views[si, i, j]
                )
    return out


def _seed_for_candidate(
    frozen: dict[str, np.ndarray],
    station: int,
    candidate: int,
    fallback: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Use a saved public witness where finite, otherwise the public q seed."""

    source_candidate = min(int(candidate), 6)
    q = np.asarray(frozen["q_solution"], dtype=float)[station, source_candidate]
    found = bool(np.asarray(frozen["static_found"], dtype=bool)[station, source_candidate])
    if found and q.shape == (8,) and np.isfinite(q).all():
        return q.copy(), "frozen_chosen_witness"
    return np.asarray(fallback, dtype=float).copy(), "placement_q_seed"


def _run_robot_table(
    bundle: SourceBundle,
    targets: np.ndarray,
    *,
    static: bool,
    verbose: bool = True,
) -> dict[str, np.ndarray | dict[str, Any]]:
    """Score every rail option and finite-check one retained rail per contact."""

    if str(CAPABILITY_DIR) not in sys.path:
        sys.path.insert(0, str(CAPABILITY_DIR))
    from capability import CapabilityEvaluator  # type: ignore

    placement = json.loads(PLACEMENT_PATH.read_text(encoding="utf-8"))
    q_seed = np.asarray(placement["q_seed"], dtype=float).reshape(8)
    anchor = np.asarray(placement["T_world_body"], dtype=float).reshape(4, 4)
    capability = CapabilityEvaluator()
    states, n, k = targets.shape[:3]
    c_rails = len(PUBLIC_RAIL_GRID_M) + len(RAIL_SUGGESTIONS_M)
    rail_options = np.empty((states, n, k, c_rails), dtype=float)
    for si in range(states):
        for i in range(n):
            for j in range(k):
                displacement_y = float(targets[si, i, j, 1, 3] - anchor[1, 3])
                suggested = np.clip(q_seed[0] + displacement_y + RAIL_SUGGESTIONS_M, 0.0, 0.8)
                rail_options[si, i, j] = np.r_[PUBLIC_RAIL_GRID_M, suggested]
    flat_targets = targets[:, :, :, None].repeat(c_rails, axis=3).reshape(-1, 4, 4)
    flat_rails = rail_options.reshape(-1)
    scores = capability.score(flat_targets, flat_rails)
    candidate_margin = np.asarray(scores["raw_calibrated_margin"], dtype=float).reshape(states, n, k, c_rails)
    candidate_proxy = np.asarray(scores["proxy_pass"], dtype=bool).reshape(states, n, k, c_rails)
    chosen_rail = np.full((states, n, k), np.nan, dtype=float)
    chosen_margin = np.full((states, n, k), np.nan, dtype=float)
    chosen_proxy = np.zeros((states, n, k), dtype=bool)
    static_found = np.zeros((states, n, k), dtype=bool)
    q_solution = np.full((states, n, k, 8), np.nan, dtype=float)
    static_seconds = np.zeros((states, n, k), dtype=float)
    static_attempts = np.zeros((states, n, k), dtype=np.int64)
    selected_rank = np.full((states, n, k), -1, dtype=np.int64)
    seed_used = np.full((states, n, k, 8), np.nan, dtype=float)
    seed_source = np.empty((states, n, k), dtype="<U32")
    candidate_checks: list[dict[str, Any]] = []
    order = np.argsort(-candidate_margin, axis=-1, kind="stable")
    for si in range(states):
        if verbose:
            print(f"robot state {si + 1}/{states}: knee={POSE_KNEE_DEG[si]:g} deg", flush=True)
        for i in range(n):
            for j in range(k):
                seed, source = _seed_for_candidate(bundle.frozen, i, j, q_seed)
                seed_used[si, i, j] = seed
                seed_source[si, i, j] = source
                ranked = order[si, i, j]
                found = False
                start = time.perf_counter()
                for rank, rail_index in enumerate(ranked.tolist()):
                    rail = float(rail_options[si, i, j, rail_index])
                    margin = float(candidate_margin[si, i, j, rail_index])
                    proxy = bool(candidate_proxy[si, i, j, rail_index])
                    static_attempts[si, i, j] += 1
                    witness: dict[str, Any] = {"found": False, "status": "not_run"}
                    if static:
                        witness = capability._static_pose(targets[si, i, j], rail, seed)
                    if bool(witness.get("found", False)) or not static:
                        chosen_rail[si, i, j] = rail
                        chosen_margin[si, i, j] = margin
                        chosen_proxy[si, i, j] = proxy
                        selected_rank[si, i, j] = int(rank)
                        if static and bool(witness.get("found", False)):
                            static_found[si, i, j] = True
                            q_solution[si, i, j] = np.asarray(witness["q_solution"], dtype=float)
                        found = bool(witness.get("found", False)) if static else True
                        candidate_checks.append(
                            {
                                "state": int(si), "station": int(i), "candidate": int(j),
                                "rail_index": int(rail_index), "rail_m": rail,
                                "margin": margin, "proxy_pass": proxy,
                                "found": bool(found), "status": str(witness.get("status", "score_only")),
                                "attempts": int(static_attempts[si, i, j]),
                            }
                        )
                        if found or not static:
                            break
                static_seconds[si, i, j] = time.perf_counter() - start
                # If every ranked rail missed finite endpoint IK, retain the
                # highest-score rail only as a rejected geometry candidate.
                if not np.isfinite(chosen_rail[si, i, j]):
                    rail_index = int(ranked[0])
                    chosen_rail[si, i, j] = rail_options[si, i, j, rail_index]
                    chosen_margin[si, i, j] = candidate_margin[si, i, j, rail_index]
                    chosen_proxy[si, i, j] = candidate_proxy[si, i, j, rail_index]
                    selected_rank[si, i, j] = 0
                    candidate_checks.append(
                        {
                            "state": int(si), "station": int(i), "candidate": int(j),
                            "rail_index": rail_index, "rail_m": float(chosen_rail[si, i, j]),
                            "margin": float(chosen_margin[si, i, j]),
                            "proxy_pass": bool(chosen_proxy[si, i, j]), "found": False,
                            "status": "search_not_found", "attempts": int(static_attempts[si, i, j]),
                        }
                    )
    return {
        "q_seed": q_seed,
        "anchor": anchor,
        "evaluator_manifest": _jsonable(capability.manifest()),
        "rail_options_m": rail_options,
        "candidate_dird_margin": candidate_margin,
        "candidate_proxy_pass": candidate_proxy,
        "candidate_static_found": np.broadcast_to(static_found[..., None], candidate_margin.shape).copy()
        if static else np.zeros_like(candidate_proxy),
        "candidate_q_solution": np.broadcast_to(q_solution[..., None, :], (states, n, k, c_rails, 8)).copy()
        if static else np.full((states, n, k, c_rails, 8), np.nan),
        "chosen_rail_m": chosen_rail,
        "chosen_dird_margin": chosen_margin,
        "chosen_proxy_pass": chosen_proxy,
        "chosen_static_found": static_found,
        "chosen_q_solution": q_solution,
        "candidate_static_seconds": static_seconds,
        "candidate_static_attempts": static_attempts,
        "candidate_selected_rail_rank": selected_rank,
        "candidate_seed_q": seed_used,
        "candidate_seed_source": seed_source,
        "candidate_static_checks": candidate_checks,
        "dird_raw_threshold": float(scores["raw_threshold"]),
    }


def _pose_intrinsic_metrics(vessel: dict[str, np.ndarray]) -> dict[str, Any]:
    old = np.asarray(vessel["legacy_internal_xi"], dtype=float)
    new = np.asarray(vessel["internal_xi"], dtype=float)
    theta_delta = np.angle(np.exp(1j * (new[:, 0] - old[:, 0])))
    delta = np.column_stack((theta_delta, new[:, 1:] - old[:, 1:]))
    return {
        "rebake_vs_frozen_delta_max": float(np.max(np.abs(delta))),
        "rebake_vs_frozen_delta_median": float(np.median(np.abs(delta))),
        "dynamic_pose_delta_max": 0.0,
        "dynamic_pose_delta_note": "tetra anchors retain fixed rebake theta/h/d across all LBS states",
    }


def build_dynamic(
    *,
    output_path: Path = OUTPUT_PATH,
    audit_path: Path = AUDIT_PATH,
    static: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Build the bounded dynamic geometry/candidate table."""

    bundle = _load_sources()
    surface = _surface_anchor_geometry(bundle)
    vessel = _vessel_anchor_geometry(bundle)
    pose_axis_angle, joint_transforms, posed_views, posed_frames, posed_vessel = _build_pose_geometry(
        bundle, surface, vessel
    )
    frozen = bundle.frozen
    robot_from_atlas = np.asarray(frozen["robot_from_atlas"], dtype=float)
    targets = _fixed_robot_targets(robot_from_atlas, posed_views, posed_frames)
    robot = _run_robot_table(bundle, targets, static=static, verbose=verbose)

    # Repeated initial central view check: this is the strict registration
    # contract for the canonical state, not a new pose/station placement.
    base_initial = np.asarray(frozen["target_robot"], dtype=float)[0, 3]
    dynamic_initial = targets[0, 0, 3]
    initial_translation_error = float(np.linalg.norm(dynamic_initial[:3, 3] - base_initial[:3, 3]))
    initial_rotation_error = float(np.max(np.abs(dynamic_initial[:3, :3] - base_initial[:3, :3])))
    # The original LBS bridge serializes the identity pose through float32;
    # retain the measured sub-micron mismatch rather than silently copying
    # canonical coordinates around that check.
    if initial_translation_error > 5.0e-7 or initial_rotation_error > 5.0e-7:
        raise ValueError(
            "canonical fixed registration failed to reproduce the frozen initial central world transform: "
            f"translation={initial_translation_error:.3g} m rotation={initial_rotation_error:.3g}"
        )
    if not np.all(np.isfinite(targets)):
        raise ValueError("robot target table contains nonfinite values")
    robot_rotation_metrics = _rotation_metrics(targets[..., :3, :3])
    if robot_rotation_metrics["orthonormal_max_abs_error"] > 2.0e-6 or robot_rotation_metrics["det_min"] < 0.99999:
        raise ValueError(f"target robot orientation check failed: {robot_rotation_metrics}")

    # Geometric stability and gross-motion witnesses.
    canonical = posed_views[0]
    central_skin = posed_views[:, :, 3]
    displacement_surface = np.linalg.norm(central_skin - central_skin[0:1], axis=-1)
    displacement_vessel = np.linalg.norm(posed_vessel - posed_vessel[0:1], axis=-1)
    central_inward = posed_frames[:, :, 3, :, 2]
    central_depth_vec = posed_vessel - central_skin
    central_depth_mm = np.linalg.norm(central_depth_vec, axis=-1) * MM_PER_M
    inward_dot = np.einsum("sni,sni->sn", central_depth_vec, central_inward)
    if np.any(central_depth_mm <= 0.0) or np.any(inward_dot <= 0.0):
        raise ValueError("posed central contact-to-vessel orientation/depth check failed")
    base_projection_feasible = np.asarray(frozen["view_projection_feasible"], dtype=bool)
    coverage_count = int(np.sum(base_projection_feasible))
    coverage_total = int(base_projection_feasible.size)
    retained = np.asarray(robot["chosen_static_found"], dtype=bool)
    metadata: dict[str, Any] = {
        "schema": "dynamic_geometry_v1",
        "status": "bounded_smplex_lbs_geometry_and_public_robot_endpoint_checks",
        "pose_schedule": {
            "knee_joint_index": RIGHT_KNEE_INDEX,
            "knee_joint_name": "right_knee",
            "angles_deg": POSE_KNEE_DEG.tolist(),
            "state_names": POSE_STATE_NAMES.tolist(),
            "axis_angle_convention": "SMPL-X 55-joint local axis-angle; right_knee x component",
            "dynamic_knots": [11.0, 19.0],
            "modest_bend": 15.0,
        },
        "station_table": {
            "source": str(FROZEN_PATH.resolve()),
            "row_ids": np.asarray(frozen["row_ids"], dtype=np.int64).tolist(),
            "station_count": int(len(frozen["row_ids"])),
            "nominal_path_length_mm": float(np.asarray(frozen["s_mm"])[-1]),
            "labels": np.asarray(frozen["labels"]).astype(str).tolist(),
            "branch_policy": "preserve frozen R_POPV side-branch rows and R_POST_TIB_V main rows",
        },
        "contact_window": {
            "candidate_count": int(len(CONTACT_OFFSETS_MM)),
            "offsets_mm": CONTACT_OFFSETS_MM.tolist(),
            "tilts_deg": CONTACT_TILTS_DEG.tolist(),
            "kinds": CONTACT_KIND.tolist(),
            "modeled_total_fov_mm": 50.0,
            "base_offset_span_mm": float(np.max(CONTACT_OFFSETS_MM[:7]) - np.min(CONTACT_OFFSETS_MM[:7])),
            "fov_vs_reach_note": "50 mm is modeled total FOV; rail/contact reach is a separate q[0] candidate variable",
            "projection_tolerance_mm": 5.0,
            "outward_search": "base +/-24 mm offsets plus central +/-12 deg orientation alternatives",
        },
        "atlas_motion_policy": {
            "registration": "one fixed robot_from_atlas copied from LONG_LEG_VIEWS_FINAL_FROZEN; no per-pose or per-station scene placement",
            "skin_patch": "moves with each fixed contact-surface source triangle/barycentric anchor and fixed SMPL-X surface weights",
            "vessel_branch": "moves with each fixed knee_rebake harmonic tetrahedron/barycentric anchor and fixed volume LBS weights",
            "branch_and_patch_move_with_atlas": True,
            "intrinsic_coordinates": "theta/h/d are held fixed; no pose-wise re-projection",
        },
        "source_assets": {
            "frozen_table": {"path": str(FROZEN_PATH.resolve()), "sha256": _sha256(FROZEN_PATH)},
            "contact_surface": {"path": str(SURFACE_PATH.resolve()), "sha256": _sha256(SURFACE_PATH)},
            "rebake_atlas": {"path": str(ATLAS_PATH.resolve()), "sha256": _sha256(ATLAS_PATH)},
            "rebake_attachments": {"path": str(ATTACHMENTS_PATH.resolve()), "sha256": _sha256(ATTACHMENTS_PATH)},
            "rebake_vessel_rebind": {"path": str(VESSEL_REBIND_PATH.resolve()), "sha256": _sha256(VESSEL_REBIND_PATH)},
            "smplx_body_weights": {"path": str(BODY_WEIGHTS_PATH.resolve()), "sha256": _sha256(BODY_WEIGHTS_PATH)},
            "smplx_skeleton": {"path": str(SKELETON_PATH.resolve()), "sha256": _sha256(SKELETON_PATH)},
            "public_placement": {"path": str(PLACEMENT_PATH.resolve()), "sha256": _sha256(PLACEMENT_PATH)},
        },
        "surface_anchor_checks": {
            "frozen_decode_max_error_m": float(np.max(np.linalg.norm(surface["base_points"] - frozen["views_m"], axis=-1))),
            "face_barycentric_min": float(np.min(surface["barycentric"])),
            "face_barycentric_max": float(np.max(surface["barycentric"])),
            "surface_weight_sum_max_abs_error": float(np.max(np.abs(surface["weights"].sum(axis=-1) - 1.0))),
        },
        "vessel_anchor_checks": {
            "tet_count": int(len(vessel["tet_indices"])),
            "tet_barycentric_min": float(np.min(vessel["barycentric"])),
            "tet_barycentric_max": float(np.max(vessel["barycentric"])),
            "tet_barycentric_sum_max_abs_error": float(np.max(np.abs(vessel["barycentric"].sum(axis=-1) - 1.0))),
            "tet_reconstruction_residual_max_m": float(np.max(vessel["reconstruction_residual_m"])),
            "intrinsic_stability": _pose_intrinsic_metrics(vessel),
        },
        "registration_checks": {
            "fixed_registration_count": 1,
            "robot_from_atlas": robot_from_atlas.tolist(),
            "initial_central_translation_error_m": initial_translation_error,
            "initial_central_rotation_max_abs_error": initial_rotation_error,
            "initial_central_margin_frozen": float(np.asarray(frozen["dird_margin"])[0, 3]),
            "initial_central_margin_rechecked": float(np.asarray(robot["chosen_dird_margin"])[0, 0, 3]),
            "target_robot_rotation": robot_rotation_metrics,
        },
        "gross_motion_checks": {
            "central_skin_displacement_min_mm": float(np.min(displacement_surface) * MM_PER_M),
            "central_skin_displacement_max_mm": float(np.max(displacement_surface) * MM_PER_M),
            "central_vessel_displacement_min_mm": float(np.min(displacement_vessel) * MM_PER_M),
            "central_vessel_displacement_max_mm": float(np.max(displacement_vessel) * MM_PER_M),
            "central_contact_vessel_depth_min_mm": float(np.min(central_depth_mm)),
            "central_contact_vessel_depth_max_mm": float(np.max(central_depth_mm)),
            "central_inward_dot_min_m": float(np.min(inward_dot)),
            "canonical_pose_reproduces_input": bool(np.max(np.linalg.norm(canonical - surface["points"], axis=-1)) <= 2.0e-7),
        },
        "coverage_checks": {
            "frozen_surface_projection_feasible_count": coverage_count,
            "frozen_surface_projection_total_count": coverage_total,
            "frozen_surface_projection_fraction": float(coverage_count / max(coverage_total, 1)),
            "nominal_path_length_mm": float(np.asarray(frozen["s_mm"])[-1]),
            "retained_static_found_count": int(np.sum(retained)),
            "retained_candidate_total": int(retained.size),
            "retained_static_fraction": float(np.mean(retained)),
        },
        "public_robot": {
            "placement": str(PLACEMENT_PATH.resolve()),
            "q_seed": np.asarray(robot["q_seed"]).tolist(),
            "anchor": np.asarray(robot["anchor"]).tolist(),
            "static_requested": bool(static),
            "candidate_count_per_view": int(len(CONTACT_OFFSETS_MM)),
            "rail_option_count_per_candidate": int(robot["rail_options_m"].shape[-1]),
            "dird_margin_chosen_min": float(np.nanmin(robot["chosen_dird_margin"])),
            "dird_margin_chosen_max": float(np.nanmax(robot["chosen_dird_margin"])),
            "finite_endpoint_check": "CapabilityEvaluator._static_pose on each retained contact candidate; failures are rejected masks, not infeasibility proofs",
            "evaluator_manifest": robot["evaluator_manifest"],
        },
        "scope_notes": [
            "Geometry-only table; finite public dIRD score and endpoint IK witness only.",
            "No continuous collision/path validation, force validation, tissue mechanics, or hardware execution claim.",
            "The frozen public side branch and nominal 392.672-mm station path are retained; no branch relabeling or per-pose re-placement.",
        ],
    }

    arrays: dict[str, Any] = {
        "pose_knee_deg": POSE_KNEE_DEG,
        "pose_state_names": POSE_STATE_NAMES,
        "pose_axis_angle": pose_axis_angle,
        "joint_transforms": joint_transforms,
        "row_ids": np.asarray(frozen["row_ids"], dtype=np.int64),
        "s_mm": np.asarray(frozen["s_mm"], dtype=float),
        "labels": np.asarray(frozen["labels"]).astype(str),
        "skin_m": np.asarray(frozen["skin_m"], dtype=float),
        "vessel_m": vessel["vessel"],
        "frames_station": np.asarray(frozen["frames"], dtype=float),
        "vessel_depth_mm": np.asarray(frozen["vessel_depth_mm"], dtype=float),
        "views_m": surface["points"],
        "view_offsets_mm": CONTACT_OFFSETS_MM,
        "view_tilts_deg": CONTACT_TILTS_DEG,
        "view_kinds": CONTACT_KIND,
        "view_face_ids": surface["face_ids"],
        "view_barycentric": surface["barycentric"],
        "view_lbs_weights": surface["weights"],
        "view_lat_mm": np.concatenate((np.asarray(frozen["view_lat_mm"], dtype=float), np.zeros((len(frozen["row_ids"]), 2))), axis=1),
        "view_projection_residual_m": np.concatenate((np.asarray(frozen["view_projection_residual_m"], dtype=float), np.repeat(np.asarray(frozen["view_projection_residual_m"], dtype=float)[:, 3:4], 2, axis=1)), axis=1),
        "view_projection_feasible": np.concatenate((np.asarray(frozen["view_projection_feasible"], dtype=bool), np.repeat(np.asarray(frozen["view_projection_feasible"], dtype=bool)[:, 3:4], 2, axis=1)), axis=1),
        "posed_views_m": posed_views,
        "frames": surface["frames"],
        "posed_frames": posed_frames,
        "internal_valid": vessel["internal_valid"],
        "internal_xi": vessel["internal_xi"],
        "legacy_internal_xi": vessel["legacy_internal_xi"],
        "vessel_tet_indices": vessel["tet_indices"],
        "vessel_tet_vertex_indices": vessel["tet_vertex_indices"],
        "vessel_tet_barycentric": vessel["barycentric"],
        "vessel_tet_reconstruction_residual_m": vessel["reconstruction_residual_m"],
        "posed_vessel_m": posed_vessel,
        "robot_from_atlas": robot_from_atlas,
        "target_robot": targets,
        "rail_options_m": robot["rail_options_m"],
        "candidate_rail_m": robot["rail_options_m"],
        "candidate_dird_margin": robot["candidate_dird_margin"],
        "candidate_proxy_pass": robot["candidate_proxy_pass"],
        "candidate_static_found": robot["candidate_static_found"],
        "candidate_q_solution": robot["candidate_q_solution"],
        "candidate_seed_q": robot["candidate_seed_q"],
        "candidate_static_seconds": robot["candidate_static_seconds"],
        "candidate_static_attempts": robot["candidate_static_attempts"],
        "candidate_selected_rail_rank": robot["candidate_selected_rail_rank"],
        "chosen_rail_m": robot["chosen_rail_m"],
        "dird_margin": robot["chosen_dird_margin"],
        "proxy_pass": robot["chosen_proxy_pass"],
        "static_found": robot["chosen_static_found"],
        "q_solution": robot["chosen_q_solution"],
        "retained_candidate_mask": robot["chosen_static_found"],
        # Preserve the original public table under explicit base_* names for
        # downstream comparison without requiring a hidden source lookup.
        "base_views_m": np.asarray(frozen["views_m"], dtype=float),
        "base_frames": np.asarray(frozen["frames"], dtype=float),
        "base_target_robot": np.asarray(frozen["target_robot"], dtype=float),
        "base_dird_margin": np.asarray(frozen["dird_margin"], dtype=float),
        "base_static_found": np.asarray(frozen["static_found"], dtype=bool),
        "base_q_solution": np.asarray(frozen["q_solution"], dtype=float),
        "base_chosen_rail_m": np.asarray(frozen["chosen_rail_m"], dtype=float),
        "base_internal_xi": np.asarray(frozen["internal_xi"], dtype=float),
        "base_internal_tet_index": np.asarray(frozen["internal_tet_index"], dtype=np.int64),
    }
    metadata["output"] = str(output_path.resolve())
    metadata["audit_output"] = str(audit_path.resolve())
    metadata_json = json.dumps(_jsonable(metadata), sort_keys=True, separators=(",", ":"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, metadata_json=np.asarray(metadata_json), **arrays)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(_jsonable(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    parser.add_argument("--no-static", action="store_true", help="score only; not suitable as the final finite IK audit")
    args = parser.parse_args(argv)
    metadata = build_dynamic(output_path=args.output, audit_path=args.audit, static=not args.no_static)
    print(
        json.dumps(
            {
                "status": "ok",
                "output": metadata["output"],
                "audit_output": metadata["audit_output"],
                "states": metadata["pose_schedule"]["angles_deg"],
                "stations": metadata["station_table"]["station_count"],
                "candidates_per_view": metadata["contact_window"]["candidate_count"],
                "retained_static_found_count": metadata["coverage_checks"]["retained_static_found_count"],
                "retained_candidate_total": metadata["coverage_checks"]["retained_candidate_total"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

