"""Build the mechanically screened full-B motion bank.

This module is deliberately a data builder, not a controller.  It consumes the
archived moving-atlas geometry and the original dIRD/URDF evaluator from the
pinned review clone, then writes a small graph of public, replayable motion
traces.  No hidden anatomy labels or benchmark outcomes are used here.

The bank is generated at the nominal 15 degree atlas state.  Each contact has
two dIRD-ranked rail/SRS realizations at the same TCP pose.  Every directed
edge is sampled at a common 20 ms period for a fixed 1.5 s rest-to-rest slot;
the edge is retained only if all sampled configurations pass the original
joint-limit and self-collision checks and the measured joint rates stay below
the original velocity limits.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


HERE = Path(__file__).resolve()
WORK = HERE.parents[1]
PINNED = WORK / "RealUS_playground"
MOUNTED = Path("/media/camp/EXT_DRIVE/RealUS_playground")
if not (PINNED / "NMI_idea").exists():
    PINNED = MOUNTED
sys.path[:0] = [str(PINNED), str(MOUNTED)]

from NMI_idea.experiments.ab_module_ladder_20261110.capability.capability import (  # noqa: E402
    CapabilityEvaluator,
)


GEOMETRY_REL = Path(
    "NMI_idea/experiments/ab_leg_dynamic_20261112/DYNAMIC_GEOMETRY_FROZEN.npz"
)
GEOMETRY = PINNED / GEOMETRY_REL
if not GEOMETRY.exists():
    GEOMETRY = MOUNTED / GEOMETRY_REL
OUT = HERE.parent / "motion"
OUT.mkdir(parents=True, exist_ok=True)

POSE_INDEX = 2
POSE_DEG = 15.0
SITES = (3, 4, 5, 6, 7)
VIEWS = (3, 7, 8)
REALIZATIONS = ("dird_top", "dird_diverse")
EDGE_DURATION_S = 1.5
EDGE_DURATION_OPTIONS_S = (1.5, 2.0, 2.5)
SAMPLE_DT_S = 0.020
TRACE_MODE = os.environ.get("FULL_B_MOTION_TRACE", "joint").strip().lower()
if TRACE_MODE not in {"joint", "cartesian"}:
    raise ValueError("FULL_B_MOTION_TRACE must be 'joint' or 'cartesian'")
_slot_override = os.environ.get("FULL_B_MOTION_SLOT_S")
if _slot_override:
    EDGE_DURATION_S = float(_slot_override)
    EDGE_DURATION_OPTIONS_S = (EDGE_DURATION_S,)
OUTPUT_BASENAME = os.environ.get("FULL_B_MOTION_OUTPUT", "BANK").strip() or "BANK"
MIN_RAIL_SEPARATION_M = 0.15
MIN_JOINT_SEPARATION_RAD = 0.25
ARCHIVE_MARGIN_TOL = 1.5
VELOCITY_FRACTION_LIMIT = 1.0
# The archived sweep is intentionally coarse (200 mm).  The near bank probes
# the *same calibrated dIRD field* locally around its selected rail and solves
# each probe with the selected witness as the continuous IK seed.  Selection is
# by live dIRD margin only, before any edge is built or screened.
NEAR_LOCAL_DELTAS_M = (0.020, -0.020, 0.025, -0.025, 0.010, -0.010)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def f(value):
    """Convert numpy scalars/arrays recursively to JSON-safe values."""
    if isinstance(value, np.ndarray):
        return [f(x) for x in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): f(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [f(x) for x in value]
    return value


def _reason_from_check(check: dict) -> str:
    failed = []
    if not check.get("limits_pass", False):
        failed.append("joint limits")
    if not check.get("self_collision_pass", False):
        failed.append("self collision margin")
    if not check.get("fk_pass", True):
        failed.append("endpoint FK")
    return ", ".join(failed) if failed else "passed"


def _rest_to_rest_progress(count: int):
    """Return a zero-slope, monotone profile on the fixed sample grid.

    ``sin(pi*u)**0.1`` has a peak/mean ratio of about 1.068, much smaller
    than cubic smoothstep's 1.5.  The cumulative trapezoid is normalized so
    the first and last samples are exactly rest states.
    """
    u = np.linspace(0.0, 1.0, count)
    raw_velocity = np.sin(np.pi * u) ** 0.1
    cumulative = np.concatenate(
        ([0.0], np.cumsum(0.5 * (raw_velocity[1:] + raw_velocity[:-1]) * np.diff(u)))
    )
    return cumulative / cumulative[-1]


def _cartesian_ik(cap: CapabilityEvaluator, target: np.ndarray, rail: float, seed: np.ndarray):
    """Damped fixed-rail Cartesian IK, seeded by the preceding sample."""
    q = np.asarray(seed, dtype=float).copy()
    q[0] = float(rail)
    for _ in range(36):
        actual = cap.fk(q)
        translation = target[:3, 3] - actual[:3, 3]
        rotation = Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()
        if np.linalg.norm(translation) < 4e-6 and np.linalg.norm(rotation) < 4e-5:
            return q, True
        jac = cap.kin.jacobian(q)[:, 1:]
        delta = np.linalg.solve(jac @ jac.T + 2e-7 * np.eye(6), np.r_[translation, rotation])
        q[1:] += np.clip(jac.T @ delta, -0.13, 0.13)
        q[0] = float(rail)
    actual = cap.fk(q)
    p_err = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
    r_err = float(Rotation.from_matrix(actual[:3, :3].T @ target[:3, :3]).magnitude())
    return q, bool(p_err < 2.5e-4 and r_err < np.deg2rad(0.15))


def _edge_trace_cartesian(cap: CapabilityEvaluator, qa: np.ndarray, qb: np.ndarray, duration_s: float):
    """Sample a rest-to-rest Cartesian SE(3) path with a blended rail."""
    sample_count = int(round(duration_s / SAMPLE_DT_S)) + 1
    t = np.linspace(0.0, duration_s, sample_count)
    progress = _rest_to_rest_progress(sample_count)
    start_pose, end_pose = cap.fk(qa), cap.fk(qb)
    slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack((start_pose[:3, :3], end_pose[:3, :3]))))
    rotations = slerp(progress).as_matrix()
    targets = np.repeat(np.eye(4)[None], sample_count, axis=0)
    targets[:, :3, 3] = start_pose[:3, 3] + progress[:, None] * (end_pose[:3, 3] - start_pose[:3, 3])
    targets[:, :3, :3] = rotations
    q_rows = []
    checks = []
    fk_errors = []
    q = np.asarray(qa, dtype=float).copy()
    solved = True
    for index, (target, p) in enumerate(zip(targets, progress)):
        rail = float(qa[0] + p * (qb[0] - qa[0]))
        if index == 0:
            q = np.asarray(qa, dtype=float).copy()
        elif index == len(targets) - 1:
            q = np.asarray(qb, dtype=float).copy()
        else:
            q, solved_here = _cartesian_ik(cap, target, rail, q)
            solved = solved and solved_here
        q_rows.append(q.copy())
        checks.append(cap.check_q(q))
        actual = cap.fk(q)
        fk_errors.append(
            (
                float(np.linalg.norm(actual[:3, 3] - target[:3, 3])),
                float(Rotation.from_matrix(actual[:3, :3].T @ target[:3, :3]).magnitude()),
            )
        )
    q = np.asarray(q_rows)
    limits_pass = all(x["limits_pass"] for x in checks)
    collision_pass = all(x["self_collision_pass"] for x in checks)
    fk_pass = solved and max(x[0] for x in fk_errors) <= 2.5e-4 and max(x[1] for x in fk_errors) <= np.deg2rad(0.15)
    dt = float(t[1] - t[0])
    rates = np.abs(np.diff(q, axis=0)) / dt
    rate_fraction = rates / np.asarray(cap.kin.v_max, dtype=float)[None]
    max_rate_fraction = float(np.max(rate_fraction))
    rate_pass = max_rate_fraction <= VELOCITY_FRACTION_LIMIT + 1.0e-10
    fk = np.asarray([cap.fk(row) for row in q])
    tip_xyz_mm = fk[:, :3, 3] * 1000.0
    travel_mm = float(np.linalg.norm(np.diff(tip_xyz_mm, axis=0), axis=1).sum())
    rot = Rotation.from_matrix(fk[:, :3, :3])
    turn_deg = float(np.rad2deg((rot[:-1].inv() * rot[1:]).magnitude()).sum())
    minimum_self_collision_m = float(min(x["min_self_collision_distance_m"] for x in checks))
    min_joint_margin = float(min(min(x["joint_limit_margin_by_coordinate"]) for x in checks))
    legal = bool(limits_pass and collision_pass and fk_pass and rate_pass)
    reasons = []
    if not limits_pass:
        reasons.append("sampled joint limits")
    if not collision_pass:
        reasons.append("sampled self collision")
    if not fk_pass:
        reasons.append("Cartesian IK/FK tolerance")
    if not rate_pass:
        reasons.append("sampled velocity limit")
    trace = {
        "t_s": t.tolist(),
        "q": q.tolist(),
        "tip_xyz_mm": tip_xyz_mm.tolist(),
        "joint_rate_fraction_max": max_rate_fraction,
        "min_self_collision_distance_m": minimum_self_collision_m,
        "min_joint_limit_margin_rad_or_m": min_joint_margin,
        "max_cartesian_position_error_mm": 1000.0 * max(x[0] for x in fk_errors),
        "max_cartesian_rotation_error_deg": float(np.rad2deg(max(x[1] for x in fk_errors))),
        "interpolation": "linear_translation_slerp_rotation_with_linear_rail_blend_and_continuous_fixed_rail_ik",
    }
    metrics = {
        "legal": legal,
        "reason": "passed" if legal else "; ".join(reasons),
        "duration_s": duration_s,
        "travel_mm": travel_mm,
        "turn_deg": turn_deg,
        "rail_mm": float(abs(qb[0] - qa[0]) * 1000.0),
        "max_joint_rate_fraction": max_rate_fraction,
        "min_self_collision_distance_m": minimum_self_collision_m,
        "min_joint_limit_margin": min_joint_margin,
        "max_cartesian_position_error_mm": 1000.0 * max(x[0] for x in fk_errors),
        "max_cartesian_rotation_error_deg": float(np.rad2deg(max(x[1] for x in fk_errors))),
    }
    return legal, metrics, trace


def _edge_trace_joint(cap: CapabilityEvaluator, qa: np.ndarray, qb: np.ndarray, duration_s: float):
    """Rest-to-rest q interpolation used as the inexpensive replay fallback."""
    sample_count = int(round(duration_s / SAMPLE_DT_S)) + 1
    t = np.linspace(0.0, duration_s, sample_count)
    progress = _rest_to_rest_progress(sample_count)
    q = qa[None] + progress[:, None] * (qb - qa)[None]
    checks = [cap.check_q(row) for row in q]
    limits_pass = all(x["limits_pass"] for x in checks)
    collision_pass = all(x["self_collision_pass"] for x in checks)
    dt = float(t[1] - t[0])
    rate_fraction = np.abs(np.diff(q, axis=0)) / dt / np.asarray(cap.kin.v_max)[None]
    max_rate_fraction = float(np.max(rate_fraction))
    rate_pass = max_rate_fraction <= VELOCITY_FRACTION_LIMIT + 1.0e-10
    fk = np.asarray([cap.fk(row) for row in q])
    tip_xyz_mm = fk[:, :3, 3] * 1000.0
    travel_mm = float(np.linalg.norm(np.diff(tip_xyz_mm, axis=0), axis=1).sum())
    rot = Rotation.from_matrix(fk[:, :3, :3])
    turn_deg = float(np.rad2deg((rot[:-1].inv() * rot[1:]).magnitude()).sum())
    minimum_self_collision_m = float(min(x["min_self_collision_distance_m"] for x in checks))
    min_joint_margin = float(min(min(x["joint_limit_margin_by_coordinate"]) for x in checks))
    legal = bool(limits_pass and collision_pass and rate_pass)
    reasons = []
    if not limits_pass:
        reasons.append("sampled joint limits")
    if not collision_pass:
        reasons.append("sampled self collision")
    if not rate_pass:
        reasons.append("sampled velocity limit")
    trace = {
        "t_s": t.tolist(),
        "q": q.tolist(),
        "tip_xyz_mm": tip_xyz_mm.tolist(),
        "joint_rate_fraction_max": max_rate_fraction,
        "min_self_collision_distance_m": minimum_self_collision_m,
        "min_joint_limit_margin_rad_or_m": min_joint_margin,
        "interpolation": "rest_to_rest_joint_profile_with_fk_replay",
    }
    metrics = {
        "legal": legal,
        "reason": "passed" if legal else "; ".join(reasons),
        "duration_s": duration_s,
        "travel_mm": travel_mm,
        "turn_deg": turn_deg,
        "rail_mm": float(abs(qb[0] - qa[0]) * 1000.0),
        "max_joint_rate_fraction": max_rate_fraction,
        "min_self_collision_distance_m": minimum_self_collision_m,
        "min_joint_limit_margin": min_joint_margin,
    }
    return legal, metrics, trace


def _edge_trace(cap: CapabilityEvaluator, qa: np.ndarray, qb: np.ndarray, duration_s: float):
    return (
        _edge_trace_cartesian(cap, qa, qb, duration_s)
        if TRACE_MODE == "cartesian"
        else _edge_trace_joint(cap, qa, qb, duration_s)
    )


def _candidate_rows(cap: CapabilityEvaluator, z, site: int, view: int, rejections):
    target = np.asarray(z["target_robot"][POSE_INDEX, site, view], dtype=float)
    # The archived q witness is only a seed.  Each candidate rail is solved
    # again with the current pinned evaluator, so an archived q is never copied
    # into a node without a current FK/limit/collision check.
    seed = np.asarray(z["q_solution"][POSE_INDEX, site, view], dtype=float)
    rows = []
    for candidate_index, rail in enumerate(
        np.asarray(z["rail_options_m"][POSE_INDEX, site, view], dtype=float)
    ):
        archived_margin = float(
            z["candidate_dird_margin"][POSE_INDEX, site, view, candidate_index]
        )
        archived_proxy = bool(
            z["candidate_proxy_pass"][POSE_INDEX, site, view, candidate_index]
        )
        archived_found = bool(
            z["candidate_static_found"][POSE_INDEX, site, view, candidate_index]
        )
        record = {
            "site": site,
            "view": view,
            "candidate_index": candidate_index,
            "rail_m": float(rail),
            "archive_dird_margin": archived_margin,
            "archive_proxy_pass": archived_proxy,
            "archive_static_found": archived_found,
        }
        if not archived_proxy:
            record["status"] = "rejected"
            record["reason"] = "archived dIRD proxy below calibrated threshold"
            rejections.append(record)
            continue
        if not archived_found:
            record["status"] = "rejected"
            record["reason"] = "archived static witness unavailable"
            rejections.append(record)
            continue
        try:
            # _static_pose is the same finite SRS+collision search exposed by
            # CapabilityEvaluator.evaluate_pose, with rail fixed to this dIRD
            # candidate.  Calling it directly avoids a duplicate score pass.
            static = cap._static_pose(target, float(rail), seed)
        except Exception as exc:  # pragma: no cover - defensive audit record
            record["status"] = "rejected"
            record["reason"] = f"live static search exception: {type(exc).__name__}: {exc}"
            rejections.append(record)
            continue
        if not static.get("found", False):
            record["status"] = "rejected"
            record["reason"] = "live static dIRD/SRS search found no valid witness"
            record["live_search"] = static.get("search")
            rejections.append(record)
            continue
        q = np.asarray(static["q_solution"], dtype=float)
        check = cap.check_q(q, target)
        if not check["valid"]:
            record["status"] = "rejected"
            record["reason"] = f"live endpoint check failed: {_reason_from_check(check)}"
            record["live_check"] = check
            rejections.append(record)
            continue
        live_score = cap.score(target[None], [float(rail)])
        live_margin = float(live_score["raw_calibrated_margin"][0])
        record.update(
            {
                "status": "valid",
                "q": q,
                "live_check": check,
                "live_dird_raw_score": float(live_score["raw_score"][0]),
                "live_dird_margin": live_margin,
                "live_dird_proxy_pass": bool(live_score["proxy_pass"][0]),
                "live_search": {
                    "branch": static.get("branch"),
                    "psi_rad": static.get("psi_rad"),
                    "robot_configurations_checked": static.get("search", {}).get(
                        "robot_configurations_checked"
                    ),
                },
            }
        )
        rows.append(record)
    if not rows:
        raise RuntimeError(f"No mechanically valid dIRD candidate at site={site}, view={view}")
    rows.sort(key=lambda x: (-x["archive_dird_margin"], -x["live_dird_margin"], x["candidate_index"]))
    primary = rows[0]
    # The coarse archive provides the reproducible dIRD-ranked anchor.  Build
    # the second realization from a local +/-10--25 mm rail proposal, using
    # the anchor's full q as the IK seed.  This preserves the same TCP while
    # avoiding the 200--400 mm rail jumps in the fallback/far bank.
    local_rows = []
    for delta in NEAR_LOCAL_DELTAS_M:
        rail = float(primary["rail_m"] + delta)
        if not (0.0 <= rail <= 0.8):
            continue
        try:
            static = cap._static_pose(target, rail, np.asarray(primary["q"], dtype=float))
        except Exception as exc:  # pragma: no cover - defensive audit record
            rejections.append(
                {
                    "site": site,
                    "view": view,
                    "kind": "near_candidate",
                    "rail_m": rail,
                    "delta_m": delta,
                    "reason": f"local dIRD proposal exception: {type(exc).__name__}: {exc}",
                }
            )
            continue
        if not static.get("found", False):
            rejections.append(
                {
                    "site": site,
                    "view": view,
                    "kind": "near_candidate",
                    "rail_m": rail,
                    "delta_m": delta,
                    "reason": "local dIRD proposal had no live SRS witness",
                }
            )
            continue
        q = np.asarray(static["q_solution"], dtype=float)
        check = cap.check_q(q, target)
        if not check["valid"]:
            rejections.append(
                {
                    "site": site,
                    "view": view,
                    "kind": "near_candidate",
                    "rail_m": rail,
                    "delta_m": delta,
                    "reason": f"local dIRD proposal endpoint check failed: {_reason_from_check(check)}",
                    "live_check": check,
                }
            )
            continue
        live_score = cap.score(target[None], [rail])
        live_margin = float(live_score["raw_calibrated_margin"][0])
        local_rows.append(
            {
                "site": site,
                "view": view,
                "candidate_index": -1,
                "proposal": "live_dird_local_probe",
                "delta_m": float(delta),
                "rail_m": rail,
                "archive_dird_margin": live_margin,
                "archive_proxy_pass": bool(live_score["proxy_pass"][0]),
                "archive_static_found": True,
                "q": q,
                "live_check": check,
                "live_dird_raw_score": float(live_score["raw_score"][0]),
                "live_dird_margin": live_margin,
                "live_dird_proxy_pass": bool(live_score["proxy_pass"][0]),
                "live_search": {
                    "branch": static.get("branch"),
                    "psi_rad": static.get("psi_rad"),
                    "robot_configurations_checked": static.get("search", {}).get(
                        "robot_configurations_checked"
                    ),
                },
            }
        )
    if local_rows:
        # Mechanical tie break: dIRD margin, then smallest absolute rail delta,
        # then positive sign.  No task outcome or edge metric enters here.
        secondary = max(
            local_rows,
            key=lambda row: (
                row["live_dird_margin"],
                -abs(row["delta_m"]),
                row["delta_m"],
            ),
        )
        primary["selection"] = "dird_top"
        secondary["selection"] = "dird_diverse"
        for row in rows:
            if row is not primary:
                row["selection"] = None
                row["status"] = "valid_not_selected"
        rows.extend(local_rows)
        return target, primary, secondary, rows
    near = [
        row
        for row in rows[1:]
        if row["archive_dird_margin"] >= primary["archive_dird_margin"] - ARCHIVE_MARGIN_TOL
        and abs(row["rail_m"] - primary["rail_m"]) >= MIN_RAIL_SEPARATION_M
        and float(np.linalg.norm(np.asarray(row["q"]) - np.asarray(primary["q"])))
        >= MIN_JOINT_SEPARATION_RAD
    ]
    if not near:
        # This is only a defensive fallback.  The canonical graph currently
        # has a qualifying pair at every requested contact/view.
        near = [
            row
            for row in rows[1:]
            if abs(row["rail_m"] - primary["rail_m"]) >= MIN_RAIL_SEPARATION_M
        ]
    if not near:
        raise RuntimeError(f"No distinct dIRD realization at site={site}, view={view}")
    secondary = min(
        near,
        key=lambda row: float(np.linalg.norm(np.asarray(row["q"]) - np.asarray(primary["q"]))),
    )
    for row in rows:
        row["selection"] = (
            "dird_top" if row is primary else "dird_diverse" if row is secondary else None
        )
        if row["selection"] is None:
            row["status"] = "valid_not_selected"
    return target, primary, secondary, rows


def _repair_continuous_node_branches(cap, z, nodes, node_by_key, selected, rejections):
    """Use the center witness as a public seed along each graph chain.

    The archived dIRD table ranks each pose independently.  Independently
    ranked SRS witnesses can therefore jump between valid elbows even when
    their TCPs are adjacent.  This pass keeps the selected site's top/near
    rails, but re-solves every view and station from the preceding public q.
    It is still a finite dIRD/SRS search with endpoint FK and collision checks;
    no hidden anatomy or benchmark outcome enters the seed.
    """
    for realization in REALIZATIONS:
        center_start = np.asarray(
            nodes[node_by_key[(3, 3, realization)]]["q"], dtype=float
        )
        for view in VIEWS:
            # Opposite tilt starts from the already solved minus-tilt witness
            # when available, which keeps the direct 7<->8 edge on the same
            # SRS branch instead of introducing a hidden elbow jump.
            seed = (
                np.asarray(nodes[node_by_key[(3, 7, realization)]]["q"], dtype=float).copy()
                if view == 8
                else center_start.copy()
            )
            if view != 3:
                site = 3
                target = np.asarray(z["target_robot"][POSE_INDEX, site, view], dtype=float)
                rail = float(selected[(site, 3)][realization]["rail_m"])
                static = cap._static_pose(target, rail, seed)
                if static.get("found", False):
                    q = np.asarray(static["q_solution"], dtype=float)
                    check = cap.check_q(q, target)
                    if check["valid"]:
                        node = nodes[node_by_key[(site, view, realization)]]
                        score = cap.score(target[None], [rail])
                        node.update(
                            {
                                "q": q.tolist(),
                                "tip_xyz_mm": (cap.fk(q)[:3, 3] * 1000.0).tolist(),
                                "rail_m": rail,
                                "endpoint_check": check,
                                "min_self_collision_distance_m": check[
                                    "min_self_collision_distance_m"
                                ],
                                "min_joint_limit_margin": min(
                                    check["joint_limit_margin_by_coordinate"]
                                ),
                                "dird_score": {
                                    **node["dird_score"],
                                    "raw_score": float(score["raw_score"][0]),
                                    "calibrated_margin": float(score["raw_calibrated_margin"][0]),
                                    "proxy_pass": bool(score["proxy_pass"][0]),
                                    "rail_override": "site_center_rail",
                                },
                                "continuity_seed": "site3_view3_same_realization",
                            }
                        )
                        seed = q
                    else:
                        rejections.append(
                            {
                                "kind": "continuity_seed",
                                "site": site,
                                "view": view,
                                "realization": realization,
                                "reason": f"center-to-view endpoint check failed: {_reason_from_check(check)}",
                            }
                        )
                else:
                    rejections.append(
                        {
                            "kind": "continuity_seed",
                            "site": site,
                            "view": view,
                            "realization": realization,
                            "reason": "center-to-view continuous SRS seed not found",
                        }
                    )
            for site in (4, 5, 6, 7):
                target = np.asarray(z["target_robot"][POSE_INDEX, site, view], dtype=float)
                rail = float(selected[(site, 3)][realization]["rail_m"])
                static = cap._static_pose(target, rail, seed)
                if not static.get("found", False):
                    rejections.append(
                        {
                            "kind": "continuity_seed",
                            "site": site,
                            "view": view,
                            "realization": realization,
                            "reason": "station-to-station continuous SRS seed not found",
                        }
                    )
                    continue
                q = np.asarray(static["q_solution"], dtype=float)
                check = cap.check_q(q, target)
                if not check["valid"]:
                    rejections.append(
                        {
                            "kind": "continuity_seed",
                            "site": site,
                            "view": view,
                            "realization": realization,
                            "reason": f"station-to-station endpoint check failed: {_reason_from_check(check)}",
                        }
                    )
                    continue
                node = nodes[node_by_key[(site, view, realization)]]
                score = cap.score(target[None], [rail])
                node.update(
                    {
                        "q": q.tolist(),
                        "tip_xyz_mm": (cap.fk(q)[:3, 3] * 1000.0).tolist(),
                        "rail_m": rail,
                        "endpoint_check": check,
                        "min_self_collision_distance_m": check[
                            "min_self_collision_distance_m"
                        ],
                        "min_joint_limit_margin": min(
                            check["joint_limit_margin_by_coordinate"]
                        ),
                        "dird_score": {
                            **node["dird_score"],
                            "raw_score": float(score["raw_score"][0]),
                            "calibrated_margin": float(score["raw_calibrated_margin"][0]),
                            "proxy_pass": bool(score["proxy_pass"][0]),
                            "rail_override": "site_center_rail",
                        },
                        "continuity_seed": f"site{site - 1}_view{view}_same_realization",
                    }
                )
                seed = q


def build():
    if not GEOMETRY.exists():
        raise FileNotFoundError(f"missing pinned geometry: {GEOMETRY}")
    z = np.load(GEOMETRY, allow_pickle=False)
    cap = CapabilityEvaluator(device="cpu")
    rejections = []
    nodes = []
    node_by_key = {}
    node_audit = []
    node_id = 0
    selected = {}
    for site in SITES:
        for view in VIEWS:
            target, primary, secondary, rows = _candidate_rows(cap, z, site, view, rejections)
            selected[(site, view)] = {"dird_top": primary, "dird_diverse": secondary}
            for realization, row in (("dird_top", primary), ("dird_diverse", secondary)):
                q = np.asarray(row["q"], dtype=float)
                check = row["live_check"]
                fk = cap.fk(q)
                node = {
                    "id": node_id,
                    "site": site,
                    "view": view,
                    "realization": realization,
                    "q": q.tolist(),
                    "tip_xyz_mm": (fk[:3, 3] * 1000.0).tolist(),
                    "dird_score": {
                        "raw_score": row["live_dird_raw_score"],
                        "calibrated_margin": row["live_dird_margin"],
                        "proxy_pass": row["live_dird_proxy_pass"],
                        "archive_calibrated_margin": row["archive_dird_margin"],
                        "archive_candidate_index": row["candidate_index"],
                        "archive_candidate_rank": int(
                            1
                            + sum(
                                other["archive_dird_margin"] > row["archive_dird_margin"]
                                for other in rows
                            )
                        ),
                    },
                    "rail_m": row["rail_m"],
                    "endpoint_check": check,
                    "min_self_collision_distance_m": check["min_self_collision_distance_m"],
                    "min_joint_limit_margin": min(
                        check["joint_limit_margin_by_coordinate"]
                    ),
                    "dird_search": row["live_search"],
                }
                nodes.append(node)
                node_by_key[(site, view, realization)] = node_id
                node_audit.append(
                    {
                        "id": node_id,
                        "site": site,
                        "view": view,
                        "realization": realization,
                        "candidate_rows": rows,
                    }
                )
                node_id += 1

    # Keep each public graph chain on a continuous SRS branch before building
    # any edge or measuring any edge cost.  This is a mechanics-only repair;
    # it does not inspect benchmark outcomes.
    _repair_continuous_node_branches(cap, z, nodes, node_by_key, selected, rejections)

    # The requested graph is fixed before any benchmark outcome is seen:
    # adjacent daughter/trunk chains, two opposite tilt branches, and a return
    # edge for every primitive.  The realization swap is a same-TCP regrasp.
    undirected = set()

    def add_pair(a, b):
        if a == b:
            return
        undirected.add(tuple(sorted((a, b))))

    for realization in REALIZATIONS:
        for view in VIEWS:
            for a, b in ((5, 4), (4, 3), (5, 6), (6, 7)):
                add_pair(
                    node_by_key[(a, view, realization)],
                    node_by_key[(b, view, realization)],
                )
        for site in SITES:
            add_pair(
                node_by_key[(site, 3, realization)],
                node_by_key[(site, 7, realization)],
            )
            add_pair(
                node_by_key[(site, 3, realization)],
                node_by_key[(site, 8, realization)],
            )
            # Direct opposite-tilt motion is retained in addition to the
            # center bridge so a short policy can explicitly test the two
            # complementary views and return/recover without hidden edges.
            add_pair(
                node_by_key[(site, 7, realization)],
                node_by_key[(site, 8, realization)],
            )
    for site in SITES:
        for view in VIEWS:
            add_pair(
                node_by_key[(site, view, "dird_top")],
                node_by_key[(site, view, "dird_diverse")],
            )

    edges = []
    edge_audit = []
    for left, right in sorted(undirected):
        for start, goal in ((left, right), (right, left)):
            qa = np.asarray(nodes[start]["q"], dtype=float)
            qb = np.asarray(nodes[goal]["q"], dtype=float)
            attempts = []
            legal = False
            metrics = trace = None
            for duration_s in EDGE_DURATION_OPTIONS_S:
                legal, metrics, trace = _edge_trace(cap, qa, qb, duration_s)
                attempts.append(metrics)
                if legal:
                    break
            audit = {
                "start": start,
                "goal": goal,
                **metrics,
                "duration_attempts": attempts,
            }
            edge_audit.append(audit)
            if not legal:
                rejections.append(
                    {
                        "kind": "edge",
                        "start": start,
                        "goal": goal,
                        "reason": metrics["reason"],
                        "metrics": metrics,
                    }
                )
                continue
            edges.append(
                {
                    "start": start,
                    "goal": goal,
                    "duration_s": metrics["duration_s"],
                    "travel_mm": metrics["travel_mm"],
                    "turn_deg": metrics["turn_deg"],
                    "rail_mm": metrics["rail_mm"],
                    "max_joint_rate_fraction": metrics["max_joint_rate_fraction"],
                    "min_self_collision_distance_m": metrics[
                        "min_self_collision_distance_m"
                    ],
                    "min_joint_limit_margin": metrics["min_joint_limit_margin"],
                    "samples": trace,
                }
            )

    # A public hold/retry primitive is part of the graph, rather than an
    # implicit controller special case.  It has the same slot and sampled FK
    # trace as every other edge, with zero travel/turn/rail motion.
    for node in nodes:
        node_index = int(node["id"])
        qa = np.asarray(node["q"], dtype=float)
        legal, metrics, trace = _edge_trace(cap, qa, qa, EDGE_DURATION_S)
        audit = {"start": node_index, "goal": node_index, "kind": "hold", **metrics}
        edge_audit.append(audit)
        if not legal:
            rejections.append(
                {
                    "kind": "hold_edge",
                    "start": node_index,
                    "goal": node_index,
                    "reason": metrics["reason"],
                    "metrics": metrics,
                }
            )
            continue
        edges.append(
            {
                "start": node_index,
                "goal": node_index,
                "kind": "hold",
                "duration_s": EDGE_DURATION_S,
                "travel_mm": metrics["travel_mm"],
                "turn_deg": metrics["turn_deg"],
                "rail_mm": metrics["rail_mm"],
                "max_joint_rate_fraction": metrics["max_joint_rate_fraction"],
                "min_self_collision_distance_m": metrics["min_self_collision_distance_m"],
                "min_joint_limit_margin": metrics["min_joint_limit_margin"],
                "samples": trace,
            }
        )

    def ids(site, view, realization="dird_top"):
        return node_by_key[(site, view, realization)]

    task_starts = {"daughter": ids(5, 3), "trunk": ids(5, 3)}
    task_start_options = {
        "daughter": [ids(5, 3, r) for r in REALIZATIONS],
        "trunk": [ids(5, 3, r) for r in REALIZATIONS],
    }
    task_targets = {
        "daughter": {
            "reference": ids(5, 3),
            "adjacent": [ids(4, v) for v in VIEWS],
            "final": [ids(3, v) for v in VIEWS],
        },
        "trunk": {
            "reference": ids(5, 3),
            "adjacent": [ids(6, v) for v in VIEWS],
            "final": [ids(7, v) for v in VIEWS],
        },
    }
    paths = {
        "daughter_center_to_final": [ids(5, 3), ids(4, 3), ids(3, 3)],
        "daughter_center_to_final_minus_tilt": [ids(5, 3), ids(4, 7), ids(3, 7)],
        "daughter_center_to_final_plus_tilt": [ids(5, 3), ids(4, 8), ids(3, 8)],
        "trunk_center_to_final": [ids(5, 3), ids(6, 3), ids(7, 3)],
        "trunk_center_to_final_minus_tilt": [ids(5, 3), ids(6, 7), ids(7, 7)],
        "trunk_center_to_final_plus_tilt": [ids(5, 3), ids(6, 8), ids(7, 8)],
    }
    source_commit = git_head(PINNED)
    evaluator_manifest = cap.manifest()
    selection_policy = {
        "nominal_pose_index": POSE_INDEX,
        "nominal_knee_deg": POSE_DEG,
        "primary": "highest archived calibrated dIRD margin among proxy-pass candidates with live endpoint check",
        "diverse": f"nearest q witness among candidates within {ARCHIVE_MARGIN_TOL} dIRD-margin units of primary and rail separation >= {MIN_RAIL_SEPARATION_M} m; fallback only if needed",
        "outcome_independent": True,
        "joint_separation_floor_rad": MIN_JOINT_SEPARATION_RAD,
    }
    provenance = {
        "source_commit": source_commit,
        "source_clone": str(PINNED),
        "geometry_path": str(GEOMETRY),
        "geometry_sha256": sha256_file(GEOMETRY),
        "geometry_schema": "dynamic_geometry_v1",
        "atlas_pose_state": str(z["pose_state_names"][POSE_INDEX]),
        "atlas_pose_knee_deg": float(z["pose_knee_deg"][POSE_INDEX]),
        "robot_from_atlas": z["robot_from_atlas"],
        "evaluator_manifest": evaluator_manifest,
        "scope": [
            "nominal calibrated atlas pose only",
            "FK, configured joint limits, original arm/probe self-collision pairs, and sampled velocity limits",
            "no body/environment/contact/dynamics or hardware claim",
            "dIRD finite field is a proxy margin, not a safety certificate",
        ],
    }
    summary = {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "requested_undirected_edge_count": len(undirected),
        "hold_edge_count": sum(e.get("kind") == "hold" for e in edges),
        "rejected_count": len(rejections),
        "edge_duration_s": EDGE_DURATION_S,
        "edge_duration_options_s": list(EDGE_DURATION_OPTIONS_S),
        "edge_sample_dt_s": SAMPLE_DT_S,
        "trace_mode": TRACE_MODE,
        "all_retained_edges_uniform_duration": bool(
            edges and all(abs(e["duration_s"] - EDGE_DURATION_S) < 1e-12 for e in edges)
        ),
        "all_retained_edges_legal": bool(edges and all(e["max_joint_rate_fraction"] <= 1.0 for e in edges)),
    }
    bank = {
        "schema": "full_b_motion_bank_v1",
        "status": "screened_nominal_atlas_replay_bank",
        "selection_policy": selection_policy,
        "provenance": provenance,
        "summary": summary,
        "nodes": nodes,
        "edges": edges,
        "task_starts": task_starts,
        "task_start_options": task_start_options,
        "task_targets": task_targets,
        "paths": paths,
        "rejections": rejections,
    }
    preflight = {
        "schema": "full_b_motion_preflight_v1",
        "status": "pass" if summary["edge_count"] and not any(not x["legal"] for x in edge_audit) else "review",
        "selection_policy": selection_policy,
        "provenance": provenance,
        "summary": summary,
        "node_audit": node_audit,
        "edge_audit": edge_audit,
        "rejections": rejections,
    }
    (OUT / f"{OUTPUT_BASENAME}.json").write_text(json.dumps(f(bank), indent=2, sort_keys=True) + "\n")
    preflight_name = "PREFLIGHT.json" if OUTPUT_BASENAME == "BANK" else f"PREFLIGHT_{OUTPUT_BASENAME}.json"
    rejection_name = "REJECTIONS.json" if OUTPUT_BASENAME == "BANK" else f"REJECTIONS_{OUTPUT_BASENAME}.json"
    (OUT / preflight_name).write_text(json.dumps(f(preflight), indent=2, sort_keys=True) + "\n")
    (OUT / rejection_name).write_text(json.dumps(f(rejections), indent=2, sort_keys=True) + "\n")
    return bank, preflight


if __name__ == "__main__":
    bank, _ = build()
    print(json.dumps({"summary": bank["summary"], "task_starts": bank["task_starts"]}, indent=2))
