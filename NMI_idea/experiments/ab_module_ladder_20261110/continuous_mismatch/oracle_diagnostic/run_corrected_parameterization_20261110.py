#!/usr/bin/env python3
"""Run the posthoc canonical-mode oracle diagnostic in a separate output."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CONTINUOUS = HERE.parent
LADDER = CONTINUOUS.parent
PLANNING = LADDER / "planning"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PLANNING))

# Reuse only the frozen actor/diagnostic helpers.  The original oracle runner
# is imported read-only; this script never calls its run() and never writes its
# run_20261110 output directory.
from run_oracle_diagnostic import (  # noqa: E402
    CONDITIONS,
    DEADLINE_S,
    LEGAL_OUTPUT,
    LEGAL_RESULTS,
    MAX_ACTIONS,
    METHOD,
    PRIVATE_GT_N,
    PRIVATE_GT_SEED,
    annotate_events,
    failure_row,
    jsonable,
    nonfinite_paths,
    sha256_file,
    summarize,
    world_digest,
    write_state,
)
from system import CapabilityEvaluator, Robot, run_episode  # noqa: E402
from run_experiment import contexts  # noqa: E402
from observation import environment as env  # noqa: E402
from observation.servo import template_candidates  # noqa: E402


PROTOCOL_PATH = HERE / "CORRECTION.txt"
ORIGINAL_ORACLE_OUTPUT = HERE / "run_20261110"
ORIGINAL_ORACLE_RESULTS = ORIGINAL_ORACLE_OUTPUT / "RESULTS.json"
ORIGINAL_ORACLE_RESULTS_SHA256 = (
    "7f3b71a8629c2dd599b7851eb85c3ff5c288fb584a9bc7ff493fd9d585e3b9af"
)
PUBLIC_GRID = np.asarray(
    [[x, y, yaw]
     for x in np.linspace(-.04, .04, 9)
     for y in np.linspace(-.04, .04, 9)
     for yaw in np.linspace(-.8, .8, 7)],
    dtype=float,
)
POINTS_ATOL = 1e-14
MEANS_ATOL = 1e-8
MEANS_RTOL = 1e-10
PSOURCE_ATOL = 1e-12
PSOURCE_RTOL = 0.0


def _shift_private_world(world):
    """Build the exact private shifted world used by the sealed experiments."""
    registration = np.asarray(world.registration, dtype=float).copy()
    registration[1] += 0.008
    modes = np.asarray(world.modes, dtype=float).copy()
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


def canonical_public_hypothesis(world, condition):
    """Return the physically equivalent public particle and audit metadata."""
    if condition == "iid":
        modes = np.asarray(world.modes, dtype=float)
        labels = modes[:, 1].copy()
        if not np.all(np.isin(labels, (-1.0, 1.0))):
            raise AssertionError(f"iid mode labels are not canonical: {labels!r}")
        return world, dict(
            kind="identity_iid_world_already_canonical",
            original_slope_labels=labels.tolist(),
            canonical_slope_labels=labels.tolist(),
            slope_deltas=np.zeros_like(labels).tolist(),
            y_control_offset_max_m=0.0,
        )

    if condition != "shifted":
        raise ValueError(condition)

    registration = np.asarray(world.registration, dtype=float).copy()
    modes_shifted = np.asarray(world.modes, dtype=float).copy()
    shifted_labels = modes_shifted[:, 1].copy()
    shifted_abs = np.abs(shifted_labels)
    valid_abs = np.isclose(shifted_abs, 0.6, atol=1e-14, rtol=0) | np.isclose(
        shifted_abs, 1.4, atol=1e-14, rtol=0,
    )
    if not np.all(valid_abs):
        raise AssertionError(f"unexpected shifted mode labels: {shifted_labels!r}")
    modes_canonical = modes_shifted.copy()
    modes_canonical[:, 1] = np.where(shifted_labels >= 0.0, 1.0, -1.0)
    deltas = shifted_labels - modes_canonical[:, 1]
    y_controls = np.asarray(world.y_controls, dtype=float).copy()
    y_controls += deltas[:, None] * 0.25 * env.CONTROL_X[None, :]
    hypothesis = env._world(
        registration,
        modes_canonical,
        y_controls,
        np.asarray(world.z_controls, dtype=float).copy(),
        np.asarray(world.depths, dtype=float).copy(),
        np.asarray(world.identity_z, dtype=float).copy(),
        np.asarray(world.detect_z, dtype=float).copy(),
    )
    labels = np.asarray(hypothesis.modes[:, 1], dtype=float)
    if not np.all(np.isin(labels, (-1.0, 1.0))):
        raise AssertionError(f"canonical mode labels are not +/-1: {labels!r}")
    return hypothesis, dict(
        kind="shifted_slope_label_canonicalization_with_linear_y_control_offset",
        original_slope_labels=shifted_labels.tolist(),
        canonical_slope_labels=labels.tolist(),
        slope_deltas=deltas.tolist(),
        y_control_offset_m=(deltas[:, None] * 0.25 * env.CONTROL_X[None, :]).tolist(),
        y_control_offset_max_m=float(np.max(np.abs(deltas[:, None] * 0.25 * env.CONTROL_X[None, :]))),
    )


def _max_abs(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        return float(np.max(np.abs(a.astype(float) - b.astype(float))))
    return 0.0 if np.array_equal(a, b) else float("inf")


def verify_physical_equivalence(private_world, hypothesis, condition, object_index, scene_pose):
    """Verify the correction before any actor episode is allowed to start."""
    points_max_abs = _max_abs(private_world.points, hypothesis.points)
    if not np.allclose(private_world.points, hypothesis.points, rtol=0.0, atol=POINTS_ATOL):
        raise AssertionError(
            f"source point mismatch condition={condition} object={object_index}: "
            f"max={points_max_abs:.17g}"
        )

    private_pred = env.predict_batch([private_world], PUBLIC_GRID)
    hypothesis_pred = env.predict_batch([hypothesis], PUBLIC_GRID)
    # Means and p_source are the required public prediction equivalence checks.
    np.testing.assert_allclose(
        private_pred["means"], hypothesis_pred["means"],
        rtol=MEANS_RTOL, atol=MEANS_ATOL,
        err_msg=f"means mismatch condition={condition} object={object_index}",
    )
    np.testing.assert_allclose(
        private_pred["p_source"], hypothesis_pred["p_source"],
        rtol=PSOURCE_RTOL, atol=PSOURCE_ATOL,
        err_msg=f"p_source mismatch condition={condition} object={object_index}",
    )
    for key in ("p_visible", "geometric_length", "angle_audit"):
        np.testing.assert_allclose(
            private_pred[key], hypothesis_pred[key],
            rtol=1e-10, atol=1e-8,
            err_msg=f"{key} mismatch condition={condition} object={object_index}",
        )
    if not np.array_equal(private_pred["contact"], hypothesis_pred["contact"]):
        raise AssertionError(f"contact mismatch condition={condition} object={object_index}")

    weights = np.ones(1, dtype=float)
    candidates = template_candidates([hypothesis], weights, np.asarray(scene_pose, dtype=float))
    labels = np.asarray(hypothesis.modes[:, 1], dtype=float)
    if len(candidates) == 0:
        raise AssertionError(f"empty template_candidates condition={condition} object={object_index}")
    if not np.all(np.isin(labels, (-1.0, 1.0))):
        raise AssertionError(f"noncanonical labels condition={condition} object={object_index}")
    return dict(
        condition=condition,
        object_index_for_audit_only=object_index,
        source_points_max_abs_m=points_max_abs,
        prediction_grid_shape=list(PUBLIC_GRID.shape),
        prediction_means_max_abs=float(_max_abs(private_pred["means"], hypothesis_pred["means"])),
        prediction_p_source_max_abs=float(_max_abs(private_pred["p_source"], hypothesis_pred["p_source"])),
        prediction_p_visible_max_abs=float(_max_abs(private_pred["p_visible"], hypothesis_pred["p_visible"])),
        prediction_geometric_length_max_abs=float(_max_abs(private_pred["geometric_length"], hypothesis_pred["geometric_length"])),
        prediction_angle_audit_max_abs=float(_max_abs(private_pred["angle_audit"], hypothesis_pred["angle_audit"])),
        contact_equal=True,
        canonical_mode_labels=labels.tolist(),
        template_candidate_count_at_initial_pose=len(candidates),
        passed=True,
    )


def run(output: Path):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)

    original_oracle_results_sha256 = sha256_file(ORIGINAL_ORACLE_RESULTS)
    if original_oracle_results_sha256 != ORIGINAL_ORACLE_RESULTS_SHA256:
        raise AssertionError(
            "sealed original oracle RESULTS hash changed: "
            f"{original_oracle_results_sha256} != {ORIGINAL_ORACLE_RESULTS_SHA256}"
        )
    original_oracle_state = json.loads(ORIGINAL_ORACLE_RESULTS.read_text(encoding="utf-8"))
    legal_results_sha256 = sha256_file(LEGAL_RESULTS)
    legal_state = json.loads(LEGAL_RESULTS.read_text(encoding="utf-8"))
    expected_gt_hashes = legal_state["audit_private"]["private_gt_sha256"]
    protocol_sha256 = sha256_file(PROTOCOL_PATH)
    environment_path = LADDER / "observation" / "environment.py"
    planning_system_path = PLANNING / "system.py"
    planning_protocol_path = PLANNING / "PROTOCOL.json"
    original_runner_path = HERE / "run_oracle_diagnostic.py"

    private_iid = env.sample_worlds(seed=PRIVATE_GT_SEED, n=PRIVATE_GT_N, split="test")
    private_shifted = [_shift_private_world(world) for world in private_iid]
    private_by_condition = {"iid": private_iid, "shifted": private_shifted}
    actual_gt_hashes = {
        condition: world_digest(worlds)
        for condition, worlds in private_by_condition.items()
    }
    if actual_gt_hashes != expected_gt_hashes:
        raise AssertionError("corrected oracle worlds do not match sealed legal experiment GT hashes")
    if actual_gt_hashes != original_oracle_state["AUDIT_ONLY"]["private_gt_sha256"]:
        raise AssertionError("corrected oracle worlds do not match original oracle GT hashes")

    evaluator = CapabilityEvaluator()
    scene = next(row for row in contexts(evaluator) if row["id"] == "a0_identity")
    initial_pose = np.zeros(3, dtype=float)

    # Build and verify all particles before the first episode.  The private
    # rendering worlds remain the original objects in private_by_condition.
    hypotheses = {}
    parameterization_audit = []
    for condition in CONDITIONS:
        for object_index, private_world in enumerate(private_by_condition[condition]):
            hypothesis, parameterization = canonical_public_hypothesis(private_world, condition)
            equivalence = verify_physical_equivalence(
                private_world, hypothesis, condition, object_index, initial_pose,
            )
            parameterization_audit.append(dict(
                condition=condition,
                object_index_for_audit_only=object_index,
                hypothesis_world_sha256=world_digest([hypothesis]),
                parameterization=parameterization,
                equivalence=equivalence,
            ))
            hypotheses[(condition, object_index)] = hypothesis

    state = dict(
        ORACLE_LEAKED_TRUTH=True,
        schema="oracle_leaked_truth_canonical_parameterization_v1",
        status="running",
        exclusion="posthoc oracle input-contract correction; not legal A/B, not prescan, never merge",
        original_oracle_results_sha256=original_oracle_results_sha256,
        legal_results_sha256=legal_results_sha256,
        legal_public_prior_sha256=legal_state.get("public_prior_sha256"),
        source_hashes=dict(
            correction_protocol_sha256=protocol_sha256,
            correction_runner_sha256=sha256_file(Path(__file__).resolve()),
            original_oracle_runner_sha256=sha256_file(original_runner_path),
            original_oracle_protocol_sha256=sha256_file(HERE / "PROTOCOL.txt"),
            environment_sha256=sha256_file(environment_path),
            planning_system_sha256=sha256_file(planning_system_path),
            planning_protocol_sha256=sha256_file(planning_protocol_path),
        ),
        population=dict(
            private_gt_seed=PRIVATE_GT_SEED,
            private_gt_n=PRIVATE_GT_N,
            private_gt_split="test",
            conditions=list(CONDITIONS),
            leaked_particle_rule="particles=[canonical physically equivalent representation of the same private GT]",
        ),
        episode_plan=dict(
            expected=PRIVATE_GT_N * len(CONDITIONS),
            method=METHOD,
            context="a0_identity",
            max_actions=MAX_ACTIONS,
            deadline_s=DEADLINE_S,
            native_replay_run=False,
            renderer_key="('continuous_mismatch', condition, object_index, 'a0_identity')",
            candidate_family_unchanged=True,
            parameterization_only_change=True,
        ),
        AUDIT_ONLY=dict(
            private_gt_sha256=actual_gt_hashes,
            source_legal_private_gt_sha256=expected_gt_hashes,
            source_original_oracle_results_sha256=original_oracle_results_sha256,
            truth_leak_description="sole planner particle is the same private GT geometry, represented with canonical +/-1 slope labels",
            private_gt_rendering_unchanged=True,
            parameterization_audit=parameterization_audit,
            equivalence_tolerances=dict(
                source_points_atol=POINTS_ATOL,
                prediction_means_atol=MEANS_ATOL,
                prediction_means_rtol=MEANS_RTOL,
                prediction_p_source_atol=PSOURCE_ATOL,
                prediction_p_source_rtol=PSOURCE_RTOL,
            ),
        ),
        rows=[],
        summary={},
        host_wall_s=0.0,
    )

    started = time.perf_counter()
    case_index = 0
    for condition in CONDITIONS:
        for object_index, private_world in enumerate(private_by_condition[condition]):
            hypothesis = hypotheses[(condition, object_index)]
            metadata = dict(
                ORACLE_LEAKED_TRUTH=True,
                case_index=case_index,
                condition=condition,
                object_index_for_audit_only=object_index,
                method=METHOD,
                context="a0_identity",
                particles_are_private_gt=True,
                hypothesis_parameterization=(
                    "identity_iid" if condition == "iid" else "canonical_shifted_modes_plus_y_controls"
                ),
                private_gt_rendering_unchanged=True,
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
                    private_world,
                    [hypothesis],
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
        f"CORRECTION.txt  {protocol_sha256}\n"
        f"run_corrected_parameterization_20261110.py  {state['source_hashes']['correction_runner_sha256']}\n"
        f"run_oracle_diagnostic.py  {state['source_hashes']['original_oracle_runner_sha256']}\n"
        f"original_oracle_RESULTS.json  {original_oracle_results_sha256}\n"
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
        "ORACLE_LEAKED_TRUTH corrected parameterization diagnostic",
        f"status={state['status']} episodes={len(state['rows'])}/16",
        "Posthoc input-contract correction only; original oracle and legal A/B outputs are excluded and unchanged.",
        f"original oracle RESULTS hash unchanged={original_oracle_results_sha256 == ORIGINAL_ORACLE_RESULTS_SHA256}",
        f"GT hashes match sealed legal/original oracle={actual_gt_hashes == expected_gt_hashes}",
        f"all parameterization equivalence checks passed={all(x['equivalence']['passed'] for x in parameterization_audit)}",
        f"host wall seconds={state['host_wall_s']:.6f}; native replay run=False",
        "",
        "Corrected oracle by condition:",
    ]
    for condition, group in state["summary"].items():
        report.append(f"  {condition}: {group}")
    report.extend([
        "",
        "Original oracle context (copied descriptively; not merged):",
        f"  iid: {original_oracle_state['summary'].get('iid')}",
        f"  shifted: {original_oracle_state['summary'].get('shifted')}",
        "",
        "Legal finite-prior A context (copied descriptively; not merged):",
        f"  iid: {legal_a['iid']}",
        f"  shifted: {legal_a['shifted']}",
        "",
        "Interpretation: known-anatomy comparison, not a mathematical performance upper bound.",
        "A_continuous_servo retains its public-quality heuristic and finite candidate family;",
        "it is not guaranteed to choose a globally optimal imaging action.  Known-anatomy",
        "failure alone does not prove geometry infeasibility or six-action-budget causality.",
        "",
        "ORACLE_LEAKED_TRUTH is intentionally uppercase and must remain visible downstream.",
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
        raise SystemExit("corrected oracle diagnostic requires explicit --release")
    state = run(args.output)
    print(json.dumps({
        "ORACLE_LEAKED_TRUTH": state["ORACLE_LEAKED_TRUTH"],
        "status": state["status"],
        "episodes": len(state["rows"]),
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
