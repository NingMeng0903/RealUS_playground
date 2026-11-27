#!/usr/bin/env python3
"""Finite public-prior versus independent continuous-GT mismatch screen.

The formal workload is intentionally guarded by ``--release``.  Importing
this module never instantiates a strategy episode or writes an output.
"""
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
LADDER = HERE.parent
PLANNING = LADDER / "planning"
PROTOCOL_PATH = HERE / "PROTOCOL.txt"
MAX_ACTIONS = 6
DEADLINE_S = 60.0
PUBLIC_PRIOR_SEED = 720502
PUBLIC_PRIOR_N = 32
PRIVATE_GT_SEED = 720380
PRIVATE_GT_N = 8
METHOD_ORDER_SEED = 720531
SUPPORT_THRESHOLD = 1e-12
METHODS = ("A_continuous_servo", "B_two_step")
CONDITIONS = ("iid", "shifted")

# Importing the existing planning implementation is deliberate.  It is
# read-only for this experiment; no local controller copy is maintained.
sys.path.insert(0, str(PLANNING))
from system import CapabilityEvaluator, Robot, run_episode  # noqa: E402
from run_experiment import contexts  # noqa: E402
from observation import environment as env  # noqa: E402


WORLD_FIELDS = (
    "registration",
    "modes",
    "y_controls",
    "z_controls",
    "depths",
    "identity_z",
    "detect_z",
    "points",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def world_digest(worlds) -> str:
    """Stable digest of private/public World fields without exposing them."""
    digest = hashlib.sha256()
    for index, world in enumerate(worlds):
        digest.update(f"world:{index}\n".encode("utf-8"))
        for name in WORLD_FIELDS:
            value = np.ascontiguousarray(np.asarray(getattr(world, name)))
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def shift_private_world(world):
    """Apply the frozen +8 mm common registration/+0.10 slope perturbation."""
    registration = np.asarray(world.registration, dtype=float).copy()
    registration[1] += 0.008
    modes = np.asarray(world.modes, dtype=float).copy()
    # environment._points multiplies modes[:,1] by .25, hence +.4 gives
    # the declared +.10 slope term for each of the two sources.
    modes[:, 1] += 0.4
    return env._world(
        registration,
        modes,
        np.asarray(world.y_controls, dtype=float).copy(),
        np.asarray(world.z_controls, dtype=float).copy(),
        np.asarray(world.depths, dtype=float).copy(),
        np.asarray(world.identity_z, dtype=float).copy(),
        np.asarray(world.detect_z, dtype=float).copy(),
    )


def nearest_prior_geometry(gt_world, prior_worlds):
    distances = np.asarray([
        np.sqrt(np.mean((gt_world.points - prior.points) ** 2))
        for prior in prior_worlds
    ], dtype=float)
    nearest = int(np.argmin(distances))
    return nearest, float(distances[nearest])


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
    if isinstance(value, Path):
        return str(value)
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


def belief_diagnostic(weights):
    weights = np.asarray(weights, dtype=float)
    if weights.size == 0 or not np.all(np.isfinite(weights)):
        return dict(
            raw_ess=float("nan"),
            posterior_max_weight=float("nan"),
            unique_effective_support=0,
            support_threshold=SUPPORT_THRESHOLD,
            mismatch_warning=True,
            warning_reason="nonfinite_or_empty_weights",
        )
    ess = float(1.0 / np.sum(weights**2))
    max_weight = float(np.max(weights))
    unique = int(np.sum(weights > SUPPORT_THRESHOLD))
    warning = bool(
        ess < max(4.0, weights.size * 0.08)
        or max_weight > 0.5
        or unique < max(4, int(np.ceil(weights.size * 0.08)))
    )
    reasons = []
    if ess < max(4.0, weights.size * 0.08):
        reasons.append("low_raw_ess")
    if max_weight > 0.5:
        reasons.append("high_posterior_max")
    if unique < max(4, int(np.ceil(weights.size * 0.08))):
        reasons.append("small_effective_support")
    return dict(
        raw_ess=ess,
        posterior_max_weight=max_weight,
        unique_effective_support=unique,
        support_threshold=SUPPORT_THRESHOLD,
        mismatch_warning=warning,
        warning_reason=reasons,
    )


def add_belief_diagnostics(row):
    event_diags = []
    for index, event in enumerate(row.get("events", [])):
        posterior = event.get("posterior", [])
        prior_before = event.get("prior_before", [])
        post_diag = belief_diagnostic(posterior)
        pre_diag = belief_diagnostic(prior_before)
        event["belief_diagnostic"] = post_diag
        event["pre_update_belief_diagnostic"] = pre_diag
        event["belief_diagnostic"]["event_index"] = index
        event_diags.append(post_diag)
    row["belief_diagnostics"] = event_diags
    if event_diags:
        row["diagnostic_summary"] = dict(
            min_raw_ess=float(np.nanmin([d["raw_ess"] for d in event_diags])),
            max_posterior_weight=float(
                np.nanmax([d["posterior_max_weight"] for d in event_diags])
            ),
            min_unique_effective_support=int(
                min(d["unique_effective_support"] for d in event_diags)
            ),
            any_mismatch_warning=bool(any(d["mismatch_warning"] for d in event_diags)),
        )
    else:
        row["diagnostic_summary"] = dict(
            min_raw_ess=float("nan"),
            max_posterior_weight=float("nan"),
            min_unique_effective_support=0,
            any_mismatch_warning=True,
        )
    return row


def summarize(rows):
    by_group = {}
    for condition in CONDITIONS:
        for method in METHODS:
            selected = [
                row for row in rows
                if row.get("condition") == condition and row.get("method") == method
            ]
            rmt = np.asarray([row.get("rmt_s", DEADLINE_S) for row in selected], dtype=float)
            actions = np.asarray([row.get("actions", 0) for row in selected], dtype=float)
            diag = [row.get("diagnostic_summary", {}) for row in selected]
            by_group[f"{condition}/{method}"] = dict(
                n=len(selected),
                completed=sum(row.get("status") == "completed" for row in selected),
                failures=sum(row.get("status") != "completed" for row in selected),
                correct_public_stops=sum(bool(row.get("correct_public_stop", False)) for row in selected),
                false_public_stops=sum(bool(row.get("false_public_stop", False)) for row in selected),
                mean_RMT_s=float(np.mean(rmt)) if rmt.size else float("nan"),
                median_RMT_s=float(np.median(rmt)) if rmt.size else float("nan"),
                mean_actions=float(np.mean(actions)) if actions.size else float("nan"),
                any_mismatch_warning=bool(any(d.get("any_mismatch_warning", True) for d in diag)),
                min_raw_ess=float(np.nanmin([d.get("min_raw_ess", float("nan")) for d in diag])) if diag else float("nan"),
                max_posterior_weight=float(np.nanmax([d.get("max_posterior_weight", float("nan")) for d in diag])) if diag else float("nan"),
                min_unique_effective_support=int(min([d.get("min_unique_effective_support", 0) for d in diag], default=0)),
            )

    paired_rows = []
    for method in METHODS:
        for object_index in range(PRIVATE_GT_N):
            iid = next((
                row for row in rows
                if row.get("method") == method and row.get("condition") == "iid"
                and row.get("object_index_for_audit_only") == object_index
            ), None)
            shifted = next((
                row for row in rows
                if row.get("method") == method and row.get("condition") == "shifted"
                and row.get("object_index_for_audit_only") == object_index
            ), None)
            if iid is None or shifted is None:
                continue
            paired_rows.append(dict(
                method=method,
                object_index_for_audit_only=object_index,
                iid_status=iid.get("status"),
                shifted_status=shifted.get("status"),
                delta_shifted_minus_iid=dict(
                    RMT_s=float(shifted.get("rmt_s", DEADLINE_S) - iid.get("rmt_s", DEADLINE_S)),
                    correct_public_stop=int(bool(shifted.get("correct_public_stop", False)))
                    - int(bool(iid.get("correct_public_stop", False))),
                    false_public_stop=int(bool(shifted.get("false_public_stop", False)))
                    - int(bool(iid.get("false_public_stop", False))),
                    actions=int(shifted.get("actions", 0)) - int(iid.get("actions", 0)),
                    nearest_prior_geometry_rms_m=float(
                        shifted["audit_only"]["nearest_prior_geometry_rms_m"]
                        - iid["audit_only"]["nearest_prior_geometry_rms_m"]
                    ),
                ),
            ))

    paired_summary = {}
    for method in METHODS:
        selected = [row for row in paired_rows if row["method"] == method]
        deltas = [row["delta_shifted_minus_iid"] for row in selected]
        paired_summary[method] = dict(
            n_pairs=len(selected),
            mean_delta_shifted_minus_iid=dict(
                RMT_s=float(np.mean([x["RMT_s"] for x in deltas])) if deltas else float("nan"),
                correct_public_stop=float(np.mean([x["correct_public_stop"] for x in deltas])) if deltas else float("nan"),
                false_public_stop=float(np.mean([x["false_public_stop"] for x in deltas])) if deltas else float("nan"),
                actions=float(np.mean([x["actions"] for x in deltas])) if deltas else float("nan"),
                nearest_prior_geometry_rms_m=float(np.mean([x["nearest_prior_geometry_rms_m"] for x in deltas])) if deltas else float("nan"),
            ),
        )
    return dict(
        by_condition_method=by_group,
        paired_clusters=paired_rows,
        paired_cluster_summary=paired_summary,
    )


def write_state(path: Path, state):
    path.write_text(
        json.dumps(jsonable(state), indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def make_failure_row(metadata, error, stack):
    row = dict(
        **metadata,
        status="exception",
        failure_type=type(error).__name__,
        failure_message=str(error),
        failure_traceback=stack,
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
        scope="L0 geometric sensing plus real static robot endpoint search",
    )
    return add_belief_diagnostics(row)


def invoke_episode(run_episode_fn, private_world, particles, method, robot, world_key):
    """Call the existing actor without any method aliasing."""
    row = run_episode_fn(
        private_world,
        particles,
        method,
        robot,
        world_key,
        max_actions=MAX_ACTIONS,
    )
    if row.get("method") != method:
        raise AssertionError(
            f"episode method contract violated: requested {method!r}, "
            f"returned {row.get('method')!r}"
        )
    return row


def method_passthrough_contract():
    """Lightweight no-GT contract check; does not run a planning episode."""
    captured = []

    def stub_run_episode(private_world, particles, method, robot, world_key, *, max_actions):
        captured.append((method, max_actions))
        return {"method": method}

    for method in METHODS:
        row = invoke_episode(
            stub_run_episode,
            object(),
            object(),
            method,
            object(),
            ("method-contract", method),
        )
        assert row["method"] == method
    assert [item[0] for item in captured] == list(METHODS)
    assert all(item[1] == MAX_ACTIONS for item in captured)
    return True


def run(release_output: Path):
    method_passthrough_contract()
    if release_output.exists():
        raise FileExistsError(f"output directory already exists: {release_output}")
    release_output.mkdir(parents=True, exist_ok=False)

    protocol_sha256 = sha256_file(PROTOCOL_PATH)
    system_path = PLANNING / "system.py"
    planning_protocol_path = PLANNING / "PROTOCOL.json"
    environment_path = LADDER / "observation" / "environment.py"

    public_prior = env.sample_prior(seed=PUBLIC_PRIOR_SEED, n=PUBLIC_PRIOR_N)
    private_iid = env.sample_worlds(seed=PRIVATE_GT_SEED, n=PRIVATE_GT_N, split="test")
    private_shifted = [shift_private_world(world) for world in private_iid]
    private_by_condition = {"iid": private_iid, "shifted": private_shifted}

    public_prior_sha256 = world_digest(public_prior)
    private_gt_sha256 = {
        condition: world_digest(worlds)
        for condition, worlds in private_by_condition.items()
    }

    evaluator = CapabilityEvaluator()
    scene_candidates = contexts(evaluator)
    scene = next(row for row in scene_candidates if row["id"] == "a0_identity")

    nearest_geometry = []
    for condition, worlds in private_by_condition.items():
        for object_index, world in enumerate(worlds):
            nearest_index, rms = nearest_prior_geometry(world, public_prior)
            nearest_geometry.append(dict(
                condition=condition,
                object_index_for_audit_only=object_index,
                nearest_public_prior_index_for_audit_only=nearest_index,
                nearest_prior_geometry_rms_m=rms,
            ))
    nearest_lookup = {
        (row["condition"], row["object_index_for_audit_only"]): row
        for row in nearest_geometry
    }

    # The hashes for private worlds and nearest geometry are deliberately
    # nested under audit_private; no policy-facing object receives them.
    state = dict(
        schema="continuous_gt_finite_prior_mismatch_v1",
        status="running",
        protocol_sha256=protocol_sha256,
        public_prior_sha256=public_prior_sha256,
        source_hashes=dict(
            environment_sha256=sha256_file(environment_path),
            planning_system_sha256=sha256_file(system_path),
            planning_protocol_sha256=sha256_file(planning_protocol_path),
            runner_sha256=sha256_file(Path(__file__).resolve()),
        ),
        population=dict(
            public_prior_seed=PUBLIC_PRIOR_SEED,
            public_prior_n=PUBLIC_PRIOR_N,
            private_gt_seed=PRIVATE_GT_SEED,
            private_gt_n=PRIVATE_GT_N,
            private_gt_split="test",
            independent_objects=True,
            conditions=list(CONDITIONS),
            shifted_registration_y_m=0.008,
            shifted_modes_slope_delta=0.4,
            shifted_physical_slope_term=0.10,
        ),
        episode_plan=dict(
            total_expected=PRIVATE_GT_N * len(CONDITIONS) * len(METHODS),
            context="a0_identity",
            methods=list(METHODS),
            planner_method_passthrough=True,
            max_actions=MAX_ACTIONS,
            deadline_s=DEADLINE_S,
            method_order_seed=METHOD_ORDER_SEED,
            method_order_rule="one master RNG permutation per object/condition case",
            fresh_robot_cache_per_episode=True,
            run_episode_reused=True,
        ),
        audit_private=dict(
            private_gt_sha256=private_gt_sha256,
            nearest_prior_geometry_definition="sqrt(mean((gt.points-prior.points)**2))",
            nearest_prior_geometry=nearest_geometry,
            gt_hash_policy="private GT digests appear only in this audit section",
        ),
        mismatch_diagnostic=dict(
            support_threshold=SUPPORT_THRESHOLD,
            warning_ess_threshold=f"max(4, 0.08*{PUBLIC_PRIOR_N})",
            warning_max_posterior_weight=0.5,
            warning_min_unique_support=f"max(4, ceil(0.08*{PUBLIC_PRIOR_N}))",
            interpretation="finite-prior warning signal, not continuous-SMC calibration",
        ),
        rows=[],
        summary={},
        host_wall_s=0.0,
    )

    order_rng = np.random.default_rng(METHOD_ORDER_SEED)
    started = time.perf_counter()
    case_index = 0
    for object_index in range(PRIVATE_GT_N):
        for condition in CONDITIONS:
            method_order = [METHODS[index] for index in order_rng.permutation(len(METHODS))]
            world = private_by_condition[condition][object_index]
            audit = nearest_lookup[(condition, object_index)]
            for method_order_index, method in enumerate(method_order):
                metadata = dict(
                    case_index=case_index,
                    object_index_for_audit_only=object_index,
                    condition=condition,
                    method=method,
                    planner_method=method,
                    method_order=list(method_order),
                    method_order_index=method_order_index,
                    context="a0_identity",
                    public_prior_sha256=public_prior_sha256,
                    audit_only=dict(
                        nearest_public_prior_index_for_audit_only=audit[
                            "nearest_public_prior_index_for_audit_only"
                        ],
                        nearest_prior_geometry_rms_m=audit["nearest_prior_geometry_rms_m"],
                    ),
                )
                robot = Robot(
                    evaluator,
                    np.asarray(scene["T_world_body"], dtype=float),
                    np.asarray(scene["q_seed"], dtype=float),
                )
                episode_started = time.perf_counter()
                try:
                    row = invoke_episode(
                        run_episode,
                        world,
                        public_prior,
                        method,
                        robot,
                        ("continuous_mismatch", condition, object_index, "a0_identity"),
                    )
                    row.update(metadata)
                    row["status"] = "completed"
                    row["host_episode_wall_s"] = time.perf_counter() - episode_started
                    add_belief_diagnostics(row)
                    anomalies = nonfinite_paths(row)
                    if anomalies:
                        row["status"] = "numeric_anomaly"
                        row["numeric_anomaly_paths"] = anomalies
                        row["rmt_s_before_anomaly_override"] = row.get("rmt_s")
                        row["rmt_s"] = DEADLINE_S
                        row["elapsed_s"] = DEADLINE_S
                except Exception as error:  # preserve every scheduled sample
                    row = make_failure_row(metadata, error, traceback.format_exc())
                    row["host_episode_wall_s"] = time.perf_counter() - episode_started
                state["rows"].append(row)
                state["summary"] = summarize(state["rows"])
                state["host_wall_s"] = time.perf_counter() - started
                write_state(release_output / "RESULTS.json", state)
                case_index += 1

    state["status"] = "complete"
    state["summary"] = summarize(state["rows"])
    state["host_wall_s"] = time.perf_counter() - started
    write_state(release_output / "RESULTS.json", state)
    result_sha256 = sha256_file(release_output / "RESULTS.json")
    (release_output / "RESULTS.sha256").write_text(
        f"RESULTS.json  {result_sha256}\n"
        f"PROTOCOL.txt  {protocol_sha256}\n"
        f"environment.py  {state['source_hashes']['environment_sha256']}\n"
        f"planning/system.py  {state['source_hashes']['planning_system_sha256']}\n"
        f"planning/PROTOCOL.json  {state['source_hashes']['planning_protocol_sha256']}\n"
        f"run_continuous_mismatch.py  {state['source_hashes']['runner_sha256']}\n"
        f"public_prior_sha256  {public_prior_sha256}\n",
        encoding="utf-8",
    )

    summary = state["summary"]
    report_lines = [
        "Continuous-GT / finite-public-prior mismatch screen",
        f"Status: {state['status']}",
        "Scope: finite 32-point public belief versus independent continuous GT;",
        "not calibrated continuous SMC, not a robot/clinical generalization claim.",
        "",
        f"Episodes: {len(state['rows'])}/{state['episode_plan']['total_expected']}; "
        f"public prior sha256={public_prior_sha256}",
        f"Host wall seconds: {state['host_wall_s']:.6f}",
        "",
        "By condition/method:",
    ]
    for key, group in summary["by_condition_method"].items():
        report_lines.append(
            f"  {key}: n={group['n']} completed={group['completed']} failures={group['failures']} "
            f"correct={group['correct_public_stops']} false={group['false_public_stops']} "
            f"mean_RMT={group['mean_RMT_s']:.6g} min_raw_ESS={group['min_raw_ess']:.6g} "
            f"max_post={group['max_posterior_weight']:.6g} "
            f"min_support={group['min_unique_effective_support']}"
        )
    report_lines.extend([
        "",
        "Paired shifted-minus-iid cluster means:",
    ])
    for method, group in summary["paired_cluster_summary"].items():
        report_lines.append(f"  {method}: {group}")
    report_lines.extend([
        "",
        "Private GT hashes and nearest-prior geometry RMS values are retained only",
        "under RESULTS.json.audit_private; they are never policy inputs.",
        f"RESULTS.json sha256={result_sha256}",
    ])
    (release_output / "REPORT.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", action="store_true", help="explicitly release the formal workload")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.release:
        raise SystemExit("formal workload is guarded; pass --release after independent review")
    state = run(args.output)
    print(json.dumps({
        "status": state["status"],
        "episodes": len(state["rows"]),
        "output": str(args.output),
        "public_prior_sha256": state["public_prior_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
