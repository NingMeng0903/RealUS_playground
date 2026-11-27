#!/usr/bin/env python3
"""Explicit truth-leaked A upper-bound diagnostic; excluded from legal A/B."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CONTINUOUS = HERE.parent
LADDER = CONTINUOUS.parent
PLANNING = LADDER / "planning"
PROTOCOL_PATH = HERE / "PROTOCOL.txt"
LEGAL_OUTPUT = CONTINUOUS / "formal_run_20261110"
LEGAL_RESULTS = LEGAL_OUTPUT / "RESULTS.json"
PUBLIC_PRIOR_HASH = "92432d67a7251adb93035f11570e9d1669f85be46307a8820c9b5f0f77188dbf"
PRIVATE_GT_SEED = 720380
PRIVATE_GT_N = 8
MAX_ACTIONS = 6
DEADLINE_S = 60.0
CONDITIONS = ("iid", "shifted")
METHOD = "A_continuous_servo"
ORACLE_MARKER = "ORACLE_LEAKED_TRUTH"
SUPPORT_THRESHOLD = 1e-12

sys.path.insert(0, str(PLANNING))
from system import CapabilityEvaluator, Robot, run_episode  # noqa: E402
from run_experiment import contexts  # noqa: E402
from observation import environment as env  # noqa: E402


WORLD_FIELDS = (
    "registration", "modes", "y_controls", "z_controls", "depths",
    "identity_z", "detect_z", "points",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def world_digest(worlds) -> str:
    digest = hashlib.sha256()
    for index, world in enumerate(worlds):
        digest.update(f"world:{index}\n".encode())
        for name in WORLD_FIELDS:
            value = np.ascontiguousarray(np.asarray(getattr(world, name)))
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def shift_private_world(world):
    registration = np.asarray(world.registration, dtype=float).copy()
    registration[1] += 0.008
    modes = np.asarray(world.modes, dtype=float).copy()
    modes[:, 1] += 0.4
    return env._world(
        registration, modes,
        np.asarray(world.y_controls, dtype=float).copy(),
        np.asarray(world.z_controls, dtype=float).copy(),
        np.asarray(world.depths, dtype=float).copy(),
        np.asarray(world.identity_z, dtype=float).copy(),
        np.asarray(world.detect_z, dtype=float).copy(),
    )


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def nonfinite_paths(value, path="root"):
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            found.extend(nonfinite_paths(item, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(nonfinite_paths(item, f"{path}[{index}]"))
    elif isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
            found.append(path)
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        found.append(path)
    return found


def belief_diag(weights):
    w = np.asarray(weights, dtype=float)
    if w.size == 0 or not np.all(np.isfinite(w)):
        return dict(raw_ess=float("nan"), posterior_max_weight=float("nan"),
                    unique_effective_support=0, mismatch_warning=True)
    ess = float(1.0 / np.sum(w**2))
    maximum = float(np.max(w))
    unique = int(np.sum(w > SUPPORT_THRESHOLD))
    return dict(
        raw_ess=ess,
        posterior_max_weight=maximum,
        unique_effective_support=unique,
        support_threshold=SUPPORT_THRESHOLD,
        mismatch_warning=bool(ess < 4.0 or maximum > 0.5 or unique < 4),
    )


def annotate_events(row):
    diagnostics = []
    for event in row.get("events", []):
        event["belief_diagnostic"] = belief_diag(event.get("posterior", []))
        event["pre_update_belief_diagnostic"] = belief_diag(event.get("prior_before", []))
        diagnostics.append(event["belief_diagnostic"])
    row["belief_diagnostics"] = diagnostics
    row["diagnostic_summary"] = dict(
        min_raw_ess=min((d["raw_ess"] for d in diagnostics), default=float("nan")),
        max_posterior_weight=max((d["posterior_max_weight"] for d in diagnostics), default=float("nan")),
        min_unique_effective_support=min((d["unique_effective_support"] for d in diagnostics), default=0),
        any_mismatch_warning=any(d["mismatch_warning"] for d in diagnostics),
    )
    return row


def write_state(path, state):
    path.write_text(json.dumps(jsonable(state), indent=2, allow_nan=True) + "\n", encoding="utf-8")


def failure_row(metadata, error):
    return annotate_events(dict(
        **metadata,
        status="exception",
        failure_type=type(error).__name__,
        failure_message=str(error),
        failure_traceback=traceback.format_exc(),
        correct_public_stop=False,
        false_public_stop=False,
        public_stop=False,
        ever_audit_quality=False,
        rmt_s=DEADLINE_S,
        elapsed_s=DEADLINE_S,
        actions=0,
        invalid_endpoint_attempts=0,
        endpoint_queries=0,
        model_geometry_queries=0,
        events=[],
        posterior=[],
        scope="ORACLE_LEAKED_TRUTH diagnostic only",
    ))


def summarize(rows):
    out = {}
    for condition in CONDITIONS:
        selected = [r for r in rows if r.get("condition") == condition]
        out[condition] = dict(
            n=len(selected),
            completed=sum(r.get("status") == "completed" for r in selected),
            failures=sum(r.get("status") != "completed" for r in selected),
            correct_public_stops=sum(bool(r.get("correct_public_stop")) for r in selected),
            false_public_stops=sum(bool(r.get("false_public_stop")) for r in selected),
            public_stops=sum(bool(r.get("public_stop")) for r in selected),
            ever_audit_quality=sum(bool(r.get("ever_audit_quality")) for r in selected),
            mean_RMT_s=float(np.mean([r.get("rmt_s", DEADLINE_S) for r in selected])) if selected else float("nan"),
            actions=sum(int(r.get("actions", 0)) for r in selected),
            min_raw_ess=min((r.get("diagnostic_summary", {}).get("min_raw_ess", float("nan")) for r in selected), default=float("nan")),
            max_posterior_weight=max((r.get("diagnostic_summary", {}).get("max_posterior_weight", float("nan")) for r in selected), default=float("nan")),
        )
    return out


def run(output: Path):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)

    legal_results_sha256 = sha256_file(LEGAL_RESULTS)
    legal_state = json.loads(LEGAL_RESULTS.read_text())
    expected_gt_hashes = legal_state["audit_private"]["private_gt_sha256"]
    protocol_sha256 = sha256_file(PROTOCOL_PATH)
    environment_path = LADDER / "observation" / "environment.py"
    planning_system_path = PLANNING / "system.py"
    planning_protocol_path = PLANNING / "PROTOCOL.json"

    private_iid = env.sample_worlds(seed=PRIVATE_GT_SEED, n=PRIVATE_GT_N, split="test")
    private_shifted = [shift_private_world(world) for world in private_iid]
    private_by_condition = {"iid": private_iid, "shifted": private_shifted}
    actual_gt_hashes = {
        condition: world_digest(worlds)
        for condition, worlds in private_by_condition.items()
    }
    if actual_gt_hashes != expected_gt_hashes:
        raise AssertionError("oracle worlds do not match sealed legal experiment GT hashes")

    evaluator = CapabilityEvaluator()
    scene = next(row for row in contexts(evaluator) if row["id"] == "a0_identity")
    state = dict(
        ORACLE_LEAKED_TRUTH=True,
        schema="oracle_leaked_truth_upper_bound_v1",
        status="running",
        exclusion="not legal A/B, not prescan, never merge into formal summary",
        legal_results_sha256=legal_results_sha256,
        legal_public_prior_sha256=legal_state.get("public_prior_sha256"),
        source_hashes=dict(
            oracle_protocol_sha256=protocol_sha256,
            oracle_runner_sha256=sha256_file(Path(__file__).resolve()),
            environment_sha256=sha256_file(environment_path),
            planning_system_sha256=sha256_file(planning_system_path),
            planning_protocol_sha256=sha256_file(planning_protocol_path),
        ),
        population=dict(
            private_gt_seed=PRIVATE_GT_SEED,
            private_gt_n=PRIVATE_GT_N,
            private_gt_split="test",
            conditions=list(CONDITIONS),
            leaked_particle_rule="particles=[the same private GT World]",
        ),
        episode_plan=dict(
            expected=PRIVATE_GT_N * len(CONDITIONS),
            method=METHOD,
            context="a0_identity",
            max_actions=MAX_ACTIONS,
            deadline_s=DEADLINE_S,
            native_replay_run=False,
            renderer_key="('continuous_mismatch', condition, object_index, 'a0_identity')",
        ),
        AUDIT_ONLY=dict(
            private_gt_sha256=actual_gt_hashes,
            source_legal_private_gt_sha256=expected_gt_hashes,
            truth_leak_description="sole planner particle is the private GT itself",
        ),
        rows=[],
        summary={},
        host_wall_s=0.0,
    )

    started = time.perf_counter()
    case_index = 0
    for condition in CONDITIONS:
        for object_index, world in enumerate(private_by_condition[condition]):
            metadata = dict(
                ORACLE_LEAKED_TRUTH=True,
                case_index=case_index,
                condition=condition,
                object_index_for_audit_only=object_index,
                method=METHOD,
                context="a0_identity",
                particles_are_private_gt=True,
                renderer_key=("continuous_mismatch", condition, object_index, "a0_identity"),
            )
            robot = Robot(
                evaluator,
                np.asarray(scene["T_world_body"], dtype=float),
                np.asarray(scene["q_seed"], dtype=float),
            )
            episode_started = time.perf_counter()
            try:
                row = run_episode(
                    world,
                    [world],
                    METHOD,
                    robot,
                    ("continuous_mismatch", condition, object_index, "a0_identity"),
                    max_actions=MAX_ACTIONS,
                )
                if row.get("method") != METHOD:
                    raise AssertionError(f"unexpected returned method {row.get('method')!r}")
                row.update(metadata)
                row["status"] = "completed"
                row["host_episode_wall_s"] = time.perf_counter() - episode_started
                annotate_events(row)
                anomalies = nonfinite_paths(row)
                if anomalies:
                    row["status"] = "numeric_anomaly"
                    row["numeric_anomaly_paths"] = anomalies
                    row["rmt_s_before_anomaly_override"] = row.get("rmt_s")
                    row["rmt_s"] = DEADLINE_S
                    row["elapsed_s"] = DEADLINE_S
            except Exception as error:
                row = failure_row(metadata, error)
                row["host_episode_wall_s"] = time.perf_counter() - episode_started
            state["rows"].append(row)
            state["summary"] = summarize(state["rows"])
            state["host_wall_s"] = time.perf_counter() - started
            write_state(output / "RESULTS.json", state)
            case_index += 1

    state["status"] = "complete"
    state["summary"] = summarize(state["rows"])
    state["host_wall_s"] = time.perf_counter() - started
    write_state(output / "RESULTS.json", state)
    results_sha256 = sha256_file(output / "RESULTS.json")
    (output / "RESULTS.sha256").write_text(
        f"RESULTS.json  {results_sha256}\n"
        f"PROTOCOL.txt  {protocol_sha256}\n"
        f"run_oracle_diagnostic.py  {state['source_hashes']['oracle_runner_sha256']}\n"
        f"legal_RESULTS.json  {legal_results_sha256}\n"
        f"environment.py  {state['source_hashes']['environment_sha256']}\n"
        f"planning/system.py  {state['source_hashes']['planning_system_sha256']}\n",
        encoding="utf-8",
    )
    legal_a = {
        c: next((g for k, g in legal_state.get("summary", {}).get("by_condition_method", {}).items()
                 if k == f"{c}/{METHOD}"), None)
        for c in CONDITIONS
    }
    report = [
        "ORACLE_LEAKED_TRUTH upper-bound diagnostic",
        f"status={state['status']} episodes={len(state['rows'])}/16",
        "This is invalid-policy truth leakage, excluded from legal A/B and prescan claims.",
        f"GT hashes matched sealed legal run: {actual_gt_hashes == expected_gt_hashes}",
        f"host wall seconds={state['host_wall_s']:.6f}; native replay run=False",
        "",
        "Oracle by condition:",
    ]
    for condition, group in state["summary"].items():
        report.append(f"  {condition}: {group}")
    report.extend([
        "",
        "Legal finite-prior A context (copied descriptively; not merged):",
        f"  iid: {legal_a['iid']}",
        f"  shifted: {legal_a['shifted']}",
        "",
        "ORACLE_LEAKED_TRUTH is intentionally uppercase and must remain visible",
        "in every downstream interpretation.",
        f"RESULTS.json sha256={results_sha256}",
    ])
    (output / "REPORT.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.release:
        raise SystemExit("oracle diagnostic requires explicit --release")
    state = run(args.output)
    print(json.dumps({
        "ORACLE_LEAKED_TRUTH": state["ORACLE_LEAKED_TRUTH"],
        "status": state["status"],
        "episodes": len(state["rows"]),
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
