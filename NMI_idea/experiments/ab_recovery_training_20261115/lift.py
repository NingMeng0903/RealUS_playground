"""Sampled RM75 kinematic witness for the recovery benchmark.

This is an offline secondary replay.  The recovery benchmark is generated in
the material chart, while this module lifts the saved chart paths to the
registered reference robot pose and checks them with the public RM75
capability evaluator.  It does not execute the robot, model body contact, or
feed FK results back into the saved observations.

The current witness deliberately accepts only a fixed-angle schedule.  The
new records contain the knee angle in each motion sample, but the frozen
reference-pose axes/origins are not a posed Among_US map.  A moving-angle
record therefore fails loudly instead of silently treating a changing body
pose as a rigid reference path.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LEGACY = REPO / "NMI_idea" / "experiments" / "ab_streaming_prodmp_two_vessel_20261113"
DEFAULT_SECTIONS = LEGACY / "SECTIONS.json"
DEFAULT_ATLAS = REPO / "NMI_idea" / "experiments" / "ab_leg_dynamic_20261112" / "DYNAMIC_GEOMETRY_FROZEN.npz"
DEFAULT_BANK = REPO / "NMI_idea" / "experiments" / "ab_full_b_rollout_20261113" / "motion" / "BANK_MAIN.json.xz"
DEFAULT_CASES = 6
DEFAULT_MAX_SAMPLES = 401
ANGLE_TOLERANCE_DEG = 1.0e-7


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_legacy_helpers():
    """Return the legacy center witness loader and damped Cartesian IK."""
    # Keep the implementation and witness convention identical to the old
    # streaming lift.  Importing this module has no side effects beyond its
    # public imports; its command-line check is guarded by __main__.
    sys.path.insert(0, str(LEGACY))
    from check_kinematics import ik, load_witness  # pylint: disable=import-outside-toplevel
    return ik, load_witness


def _json_rows(path: Path):
    if path.suffix != ".gz":
        raise ValueError(f"expected a gzipped JSONL record file, got {path}")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}") from exc


def _motion_block(value, *, row_index, arm):
    """Extract a new benchmark motion block as a finite N x 6 array."""
    if isinstance(value, dict):
        value = value.get("motion_t_s_x_vs_vx_q")
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 6 or len(arr) < 2:
        raise ValueError(f"case {row_index} arm {arm}: expected N x 6 motion_t_s_x_vs_vx_q")
    if not np.isfinite(arr).all():
        raise ValueError(f"case {row_index} arm {arm}: nonfinite saved motion")
    dt = np.diff(arr[:, 0])
    if np.any(dt <= 0.):
        raise ValueError(f"case {row_index} arm {arm}: motion time must increase strictly")
    return arr


def _paths_for_row(row, row_index, include_a):
    """Return the saved paths and the benchmark outcome attached to each arm."""
    suffixes = row.get("suffixes") or {}
    arms = {}
    outcomes = {}
    for arm in ("mpc", "fixed", "learned"):
        trace = suffixes.get(arm)
        source = "shared_prefix_and_suffix"
        # Cases without a trigger use the common continuation for all three
        # policy arms.  run.py stores that continuation as a_new_trace.
        if not trace:
            trace = row.get("a_new_trace")
            source = "common_a_new"
        arr = _motion_block(trace, row_index=row_index, arm=arm)
        if arr is None:
            raise ValueError(f"case {row_index}: no saved trace for required arm {arm}")
        if source == "shared_prefix_and_suffix":
            prefix = _motion_block(row["prefix"], row_index=row_index, arm=arm)
            if not np.array_equal(prefix[-1], arr[0]):
                raise ValueError(f"case {row_index} arm {arm}: suffix has a different initial state")
            arr = np.concatenate((prefix[:-1], arr))
        arms[arm] = (arr, source)
        if arm in row:
            outcomes[arm] = row[arm]
    if include_a:
        for arm, key in (("a_old", "a_old_trace"), ("a_new", "a_new_trace")):
            arr = _motion_block(row.get(key), row_index=row_index, arm=arm)
            if arr is None:
                raise ValueError(f"case {row_index}: --include-a requested but {key} is absent")
            arms[arm] = (arr, key)
            if arm in row:
                outcomes[arm] = row[arm]
    return arms, outcomes


def _subsample(trace, max_samples):
    if len(trace) <= max_samples:
        return trace
    index = np.linspace(0, len(trace) - 1, max_samples).round().astype(int)
    index = np.unique(index)
    return trace[index]


def _chart_lift(trace, sections):
    """Map chart station/lateral samples to reference atlas SE(3) poses."""
    stations = np.asarray(sections["s_mm"], dtype=float)
    origins = np.asarray(sections["origin_atlas_mm"], dtype=float)
    axes = np.asarray(sections["axes_atlas"], dtype=float)
    if stations.ndim != 1 or origins.shape != (len(stations), 3) or axes.shape != (len(stations), 3, 3):
        raise ValueError("invalid registered section axes/origins")
    s, x = trace[:, 1], trace[:, 2]
    if s.min() < stations[0] - 1.0e-8 or s.max() > stations[-1] + 1.0e-8:
        raise ValueError("saved chart station leaves the registered section grid")
    s = np.clip(s, stations[0], stations[-1])
    origin = np.column_stack([np.interp(s, stations, origins[:, axis]) for axis in range(3)])
    rotation = Slerp(stations, Rotation.from_matrix(axes))(s).as_matrix()
    target = np.repeat(np.eye(4)[None], len(trace), axis=0)
    target[:, :3, :3] = rotation
    # x is in chart millimetres; capability poses are in metres.
    target[:, :3, 3] = (origin + x[:, None] * rotation[:, :, 1]) / 1000.
    return target, origin, rotation


def _turn_deg(rotations):
    if len(rotations) < 2:
        return 0.
    relative = np.swapaxes(rotations[:-1], -2, -1) @ rotations[1:]
    return float(np.rad2deg(Rotation.from_matrix(relative).magnitude()).sum())


def _path_cost(points, rotations):
    travel = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.
    return travel, _turn_deg(rotations)


def summarize_path(cap, target, time_s, witness, ik):
    """Track each target with legacy fixed-rail IK and check the actual FK."""
    q = np.asarray(witness, dtype=float).copy()
    joints, checks = [], []
    for desired in target:
        q = ik(cap, desired, q, float(witness[0]))
        joints.append(q.copy())
        checks.append(cap.check_q(q, desired))
    joints = np.asarray(joints)
    checks = list(checks)
    dt = np.diff(time_s)
    if len(joints) > 1:
        rate = np.abs(np.diff(joints, axis=0)) / dt[:, None] / np.asarray(cap.kin.v_max)
        violating = np.flatnonzero((rate > 1. + 1.e-10).any(axis=1)).tolist()
        peak_rate = float(rate.max())
    else:
        rate, violating, peak_rate = np.zeros((0, 8)), [], 0.
    invalid = [i for i, check in enumerate(checks) if not check["valid"]]
    clearance = [check["min_self_collision_distance_m"] for check in checks]
    actual = np.asarray([cap.fk(q_i) for q_i in joints])
    preposition = cap.fk(witness)
    all_poses = np.concatenate((preposition[None], actual), axis=0)
    suffix_travel, suffix_turn = _path_cost(actual[:, :3, 3], actual[:, :3, :3])
    full_travel, full_turn = _path_cost(all_poses[:, :3, 3], all_poses[:, :3, :3])
    return {
        "passes": not (invalid or violating),
        "sample_count": len(checks),
        "duration_s": float(time_s[-1] - time_s[0]),
        "invalid_kinematic_samples": invalid,
        "joint_limit_violations": [i for i, check in enumerate(checks) if not check["limits_pass"]],
        "fk_violations": [i for i, check in enumerate(checks) if not check["fk_pass"]],
        "self_collision_violations": [i for i, check in enumerate(checks) if not check["self_collision_pass"]],
        "velocity_violation_intervals": violating,
        "minimum_self_clearance_mm": (1000. * min(clearance)
                                       if all(value is not None for value in clearance) else None),
        "minimum_joint_margin_rad_or_m": min(min(check["joint_limit_margin_by_coordinate"])
                                              for check in checks),
        "maximum_tracking_position_error_mm": 1000. * max(check["fk_position_error_m"] for check in checks),
        "maximum_tracking_rotation_error_deg": float(np.rad2deg(max(check["fk_rotation_error_rad"] for check in checks))),
        "peak_joint_rate_fraction": peak_rate,
        "tcp_travel_mm": 1000. * suffix_travel,
        "tcp_turn_deg": suffix_turn,
        "tcp_travel_including_preposition_mm": 1000. * full_travel,
        "tcp_turn_including_preposition_deg": full_turn,
        "preposition_tcp_mm": 1000. * float(np.linalg.norm(actual[0, :3, 3] - preposition[:3, 3])),
        "preposition_turn_deg": float(np.rad2deg(Rotation.from_matrix(
            preposition[:3, :3].T @ actual[0, :3, :3]).magnitude())),
        "initial_q": joints[0].tolist(),
        "final_q": joints[-1].tolist(),
        "preposition_max_arm_change_rad": float(np.abs(joints[0, 1:] - witness[1:]).max()),
    }


def _check_angle_schedule(trace, row_index, arm):
    angles = trace[:, 5]
    if np.any(abs(angles - 15.) > ANGLE_TOLERANCE_DEG):
        raise ValueError(f"case {row_index} arm {arm}: the registered reference-pose lift requires 15°")
    return float(np.mean(angles))


def check(repo, records, sections_path=DEFAULT_SECTIONS, atlas_path=DEFAULT_ATLAS,
          bank_path=DEFAULT_BANK, case_count=DEFAULT_CASES, include_a=False,
          max_samples=DEFAULT_MAX_SAMPLES):
    if case_count not in (4, 6):
        raise ValueError("case_count must be the predeclared first 4 or first 6 cases")
    records = Path(records).resolve()
    sections_path, atlas_path, bank_path = (Path(p).resolve() for p in (sections_path, atlas_path, bank_path))
    if not records.exists():
        raise FileNotFoundError(records)
    for path in (sections_path, atlas_path, bank_path):
        if not path.exists():
            raise FileNotFoundError(path)
    ik, load_witness = load_legacy_helpers()
    sections = json.loads(sections_path.read_text(encoding="utf-8"))
    with np.load(atlas_path, allow_pickle=False) as data:
        robot_from_atlas = np.asarray(data["robot_from_atlas"], dtype=float)
    if robot_from_atlas.shape != (4, 4) or not np.isfinite(robot_from_atlas).all():
        raise ValueError("invalid robot_from_atlas transform")
    expected = sections.get("source_sha256", {}).get("dynamic_geometry_frozen_npz")
    if expected and expected != sha256(atlas_path):
        raise ValueError("SECTIONS.json and robot transform must come from the same frozen atlas")
    node_id, witness = load_witness(bank_path)
    witness = np.asarray(witness, dtype=float)
    if witness.shape != (8,) or not np.isfinite(witness).all():
        raise ValueError("legacy witness q must have shape (8,)")
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "rm75_control"))
    from NMI_idea.experiments.ab_module_ladder_20261110.capability.capability import (  # noqa: PLC0415
        CapabilityEvaluator,
    )
    cap = CapabilityEvaluator(device="cpu")

    rows = []
    for row_index, row in enumerate(_json_rows(records)):
        if row_index >= case_count:
            break
        paths, outcomes = _paths_for_row(row, row_index, include_a)
        for arm, (trace, source) in paths.items():
            trace = _subsample(trace, max_samples)
            angle_deg = _check_angle_schedule(trace, row_index, arm)
            chart_target, origin, rotation = _chart_lift(trace, sections)
            target_robot = np.einsum("ij,njk->nik", robot_from_atlas, chart_target)
            result = summarize_path(cap, target_robot, trace[:, 0], witness, ik)
            chart_travel_m, chart_turn_deg = _path_cost(
                chart_target[:, :3, 3], chart_target[:, :3, :3])
            result |= {
                "record_index": row_index,
                "seed": row.get("seed"),
                "family": row.get("family"),
                "angle_mode": row.get("angle_mode"),
                "arm": arm,
                "trace_source": source,
                "angle_deg": angle_deg,
                "chart_travel_mm": 1000. * chart_travel_m,
                "chart_turn_deg": chart_turn_deg,
                "maximum_logged_lateral_rate_mm_s": float(np.max(np.abs(trace[:, 4])))
                if len(trace) else 0.,
                "benchmark_outcome": outcomes.get(arm),
            }
            rows.append(result)
    if len(rows) < case_count * 3:
        raise ValueError(f"expected all MPC/fixed/learned paths for {case_count} cases; got {len(rows)}")
    scope = ("Secondary sampled RM75 kinematic replay of saved reference-pose chart paths. "
             "IK/FK, configured joint limits and configured robot/probe self-collision are checked at "
             "sampled targets; the first state is prepositioned from the legacy center witness. "
             "This is not real execution, not a contact/dynamics test, not an exhaustive IK proof, "
             "and FK is not fed back into observations or controller decisions.")
    by_arm = {}
    for arm in sorted({row["arm"] for row in rows}):
        group = [row for row in rows if row["arm"] == arm]
        by_arm[arm] = {
            "paths": len(group), "passed": sum(row["passes"] for row in group),
            "mean_tcp_travel_mm": float(np.mean([row["tcp_travel_mm"] for row in group])),
            "mean_tcp_turn_deg": float(np.mean([row["tcp_turn_deg"] for row in group])),
            "max_tracking_position_error_mm": max(row["maximum_tracking_position_error_mm"] for row in group),
            "max_tracking_rotation_error_deg": max(row["maximum_tracking_rotation_error_deg"] for row in group),
            "max_peak_joint_rate_fraction": max(row["peak_joint_rate_fraction"] for row in group),
        }
    return {
        "schema": "ab_recovery_rm75_lift_v1",
        "scope": scope,
        "case_indices": list(range(case_count)),
        "case_count": case_count,
        "arms": sorted(by_arm),
        "include_a_controls": bool(include_a),
        "sample_cap": int(max_samples),
        "witness_node_id": node_id,
        "witness_q": witness.tolist(),
        "rail_m": float(witness[0]),
        "position_tolerance_mm": 1000. * cap.search_config.fk_position_tolerance_m,
        "rotation_tolerance_deg": float(np.rad2deg(cap.search_config.fk_rotation_tolerance_rad)),
        "clearance_required_mm": 1000. * cap.search_config.collision_margin_m,
        "paths_checked": len(rows),
        "paths_passed": sum(row["passes"] for row in rows),
        "samples_checked": sum(row["sample_count"] for row in rows),
        "by_arm": by_arm,
        "source_sha256": {
            "records": sha256(records), "sections": sha256(sections_path),
            "atlas": sha256(atlas_path), "bank": sha256(bank_path),
        },
        "robot_source_sha256": {str(key): value for key, value in cap.manifest()["source_sha256"].items()},
        "paths": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", "--traces", dest="records", type=Path, required=True,
                        help="gzipped JSONL produced by recovery run.py")
    parser.add_argument("--out", type=Path, required=True, help="JSON report path")
    parser.add_argument("--cases", "--case-count", dest="cases", type=int, choices=(4, 6), default=DEFAULT_CASES,
                        help="predeclared first test cases to replay")
    parser.add_argument("--include-a", action="store_true", help="also replay a_old and a_new controls")
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--sections", type=Path, default=DEFAULT_SECTIONS)
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    args = parser.parse_args(argv)
    if args.max_samples < 2:
        parser.error("--max-samples must be at least 2")
    report = check(args.repo.resolve(), args.records.resolve(), args.sections.resolve(), args.atlas.resolve(),
                   args.bank.resolve(), args.cases, args.include_a, args.max_samples)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Passed {report['paths_passed']}/{report['paths_checked']} paths over "
          f"{report['samples_checked']} sampled states.")


if __name__ == "__main__":
    main()
