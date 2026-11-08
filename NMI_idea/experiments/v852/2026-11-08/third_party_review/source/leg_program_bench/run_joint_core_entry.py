"""Bounded entry-conditioned joint-kernel experiment, independent of v2 tests.

All program durations include their actual joint-space transfer. A rollout
retains one observed training object and propagates its sampled measured
exit q/qdot to the next kernel query. No Cartesian transition timing is used.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from . import core_execution as core
from .core_execution import CoreWorld, SPANS, TREES, old_ledger, visible_near, sign
from .dird_probe import DirdProbe
from .joint_execution import JointServoConfig
from .run_joint_core import (PublicGoalPlanner, RecordingBackend, PlannedProgramExecutor,
                             image_pose, default_specs, write_json, content_hash)
from .run_joint_execution import DEFAULT_CHECKPOINT, save_trajectory
from .simulation import Spec
from .smplx_scene import SmplxAtlasLeg


class EntryProgramExecutor(PlannedProgramExecutor):
    def call(self, *args, **kwargs):
        before = self.backend.state()
        result, packets = super().call(*args, **kwargs)
        after = self.backend.state()
        result.update(entry_q=before["q_meas"], entry_qdot=before["qdot"],
                      exit_q=after["q_meas"], exit_qdot=after["qdot"],
                      safe_exit=bool(self.safe), entry_tcp=before["T_tcp_world"],
                      exit_tcp=after["T_tcp_world"])
        return result, packets


def entry_distance(q, qdot, record):
    """Predeclared interpolation support: 6cm rail / .35rad joints / velocity."""
    q_scale = np.r_[.06, np.full(7, .35)]
    v_scale = np.r_[.03, np.full(7, .12)]
    return float(max(np.max(abs(np.asarray(q) - record["entry_q"]) / q_scale),
                     np.max(abs(np.asarray(qdot) - record["entry_qdot"]) / v_scale)))


class JointKernelRollout:
    def __init__(self, bundle, common_q):
        self.bundle = bundle
        self.q = np.asarray(common_q).copy()
        self.qdot = np.zeros(8)
        self.time = 0.
        self.safe = True
        self.supported = True
        self.calls = []

    def call(self, action):
        if not self.safe:
            result = {"action": action, "public_success": False, "independent_success": False,
                      "success": False, "safe_exit": False, "duration": 0.,
                      "status": "known_safety_stop_refusal", "entry_distance": 0., "frame_count": 0}
            self.calls.append(result)
            return result
        candidates = [r for r in self.bundle["kernels"] if r["action"] == action]
        ranked = sorted(candidates, key=lambda r: entry_distance(self.q, self.qdot, r))
        if not ranked or entry_distance(self.q, self.qdot, ranked[0]) > 1.:
            self.supported = False
            result = {"action": action, "public_success": False, "independent_success": False,
                      "success": False, "safe_exit": True, "duration": 0., "frame_count": 0,
                      "status": "entry_outside_observed_support", "entry_distance": entry_distance(self.q, self.qdot, ranked[0]) if ranked else None}
            self.calls.append(result)
            return result
        result = dict(ranked[0])
        result["entry_distance"] = entry_distance(self.q, self.qdot, result)
        self.time += result["duration"]
        self.q = np.asarray(result["exit_q"]).copy()
        self.qdot = np.asarray(result["exit_qdot"]).copy()
        self.safe = bool(result["safe_exit"])
        self.calls.append(result)
        return result


def predict_one(bundle, common_q, tree, *, release=0., hidden=False):
    e = JointKernelRollout(bundle, common_q)
    ledger, _ = old_ledger("prediction_object", bundle["near"], release, hidden)
    mode = tree.default_mode
    if tree.kind == "wait" and not hidden:
        e.time = float(release)
    if tree.kind == "probe":
        result = e.call("probe:0")
        if not e.supported:
            return {"supported": False, "success": False, "duration": e.time, "calls": e.calls}
        mode = tree.probe_mapping * sign(result.get("feature", 0.))
    completed = True
    independent = True
    for name in ("a", "f"):
        near = visible_near(ledger, e.time, "prediction_object")
        if tree.kind in ("react", "wait") and near is not None:
            mode = tree.near_mapping * sign(near)
        action = f"a:{mode}" if name == "a" else "probe:0"
        result = e.call(action)
        if not e.supported:
            break
        if not result["public_success"] and tree.recover and name != "f" and e.safe:
            mode = -mode
            result = e.call(f"a:{mode}")
            if not e.supported:
                break
        independent &= bool(result["independent_success"])
        if not result["public_success"]:
            completed = False
            break
    return {"supported": e.supported, "success": bool(e.supported and completed and independent),
            "duration": e.time, "calls": e.calls, "exit_q": e.q.tolist(), "exit_qdot": e.qdot.tolist()}


def posterior_weights(bundles, near, *, visible):
    if not visible:
        return np.full(len(bundles), 1 / len(bundles))
    values = np.asarray([r["near"] for r in bundles])
    bandwidth = max(.02, float(np.std(values)) * .5)
    weights = np.exp(-.5 * ((values - near) / bandwidth) ** 2) + 1e-12
    return weights / weights.sum()


def evaluate_tree(bundles, weights, common_q, tree, deadlines, **condition):
    samples = [predict_one(bundle, common_q, tree, **condition) for bundle in bundles]
    unknown = float(sum(w for w, r in zip(weights, samples) if not r["supported"]))
    lower = [float(sum(w for w, r in zip(weights, samples) if r["supported"] and r["success"] and r["duration"] <= d)) for d in deadlines]
    upper = [min(1., x + unknown) for x in lower]
    cost = float(sum(w * (r["duration"] + (90. if not r["supported"] else 0.)) for w, r in zip(weights, samples)))
    return {"tree": tree.name, "deadlines": list(deadlines), "completion_lower": lower,
            "completion_upper": upper, "unsupported_mass": unknown,
            "support_interval_note": "entry-outside-support paths contribute [0,1]; not a finite-sample confidence interval",
            "expected_duration_with_unsupported_penalty": cost, "samples": samples}


def build_entry_grid(world, planner):
    common = planner.candidates(image_pose(world, .60))[0]["q"]
    probe_start = planner.candidates(image_pose(world, .56), common)[0]["q"]
    probe_end = planner.candidates(image_pose(world, .60), probe_start)
    alternatives = [r for r in probe_end if np.max(abs(np.asarray(r["q"]) - common)) > .08]
    probe = (alternatives or probe_end)[0]["q"]
    a_end = planner.candidates(image_pose(world, .76), common)
    if not a_end:
        raise RuntimeError("predeclared public A endpoint has no candidate in finite library")
    return {"common": common, "public_probe_end": probe, "public_a_end": a_end[0]["q"]}


def run(output, *, seed=20261110, backend_name="genesis_pd"):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    SPANS["near"] = SPANS["probe"] = SPANS["f"] = (.56, .60)
    atlas = SmplxAtlasLeg.load()
    registration_source = Path("outputs/v852_joint_core_20261108/local_continuation_v2/frozen_protocol.json")
    registration = json.loads(registration_source.read_text())["selected_registration"]
    backend = RecordingBackend(JointServoConfig(backend=backend_name, dt=.02),
                               base_world=np.asarray(registration["base_world"]),
                               T_image_tcp=np.asarray(registration["T_image_tcp"]))
    backend.public_nominal_q = np.asarray(registration["public_nominal_q"])
    probe = DirdProbe(checkpoint=DEFAULT_CHECKPOINT, allow_stale_audit=False)
    planner = PublicGoalPlanner(probe, backend)
    EntryProgramExecutor.planner = planner
    public = CoreWorld(atlas, Spec("public_entry_grid", 61108, 1, -1, knee_deg=60.), "basic")
    grid = build_entry_grid(public, planner)
    deadlines = [20., 40., 60., 90.]
    conditions = {"early": {"release": 0., "hidden": False},
                  "late": {"release": 5., "hidden": False},
                  "hide": {"release": 0., "hidden": True}}
    protocol = {"schema": "v852_joint_entry_kernel_v3", "backend": backend_name,
                "scope": "local continuation at pose60, same known shape; no contact, B-mode, registration or full-leg claim",
                "seed": seed, "spans": {k: list(v) for k, v in SPANS.items()},
                "registration_source": str(registration_source), "registration_sha256": hashlib.sha256(registration_source.read_bytes()).hexdigest(),
                "entry_grid": grid, "entry_grid_qdot": [0.] * 8,
                "entry_support_q_scale": [.06] + [.35] * 7, "entry_support_qdot_scale": [.03] + [.12] * 7,
                "training_objects": 2, "local_program_calls": 18, "test_objects": 2,
                "complete_chains_for_training": 0, "deadlines": deadlines, "selection_deadline": 40.,
                "conditions": conditions, "trees": [asdict(t) for t in TREES],
                "selection_rule": "maximize conservative completion lower at40, then supported mass, then minimize full-duration cost; unknown paths penalized90s",
                "joint_kernel": "nearest observed actual q/qdot entry, whole outcome+duration+exit joint record; one persistent training object per rollout",
                "transition_time_approximation": None, "formal_guarantee": False,
                "old_evidence": "actual independent test-time slowProbe preparation; same derived packet across all arms",
                "physical_execution": False, "plant": backend.manifest(), "checkpoint": probe.metadata()}
    write_json(output / "frozen_protocol.json", protocol)
    source_hashes = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                     for name in ("core_execution.py", "joint_execution.py", "run_joint_core.py", "run_joint_core_entry.py")}
    write_json(output / "source_freeze.json", source_hashes)
    bundles = []
    audits = []
    for i, spec in enumerate(default_specs(2, "entry_train", seed * 10 + 1000, 60.)):
        world = CoreWorld(atlas, spec, "basic")
        bundle = {"object_id": spec.object_id, "near": 0., "kernels": []}
        for entry_index, (entry_name, q) in enumerate(grid.items()):
            for action_index, (name, mode) in enumerate((("a", -1), ("a", 1), ("probe", 0))):
                backend.reset(np.asarray(q))
                engine = EntryProgramExecutor(world, backend=backend).set_rng(seed + i * 1000 + entry_index * 10 + action_index)
                result, packets = engine.call(name, mode)
                result["action"] = f"{name}:{mode}"
                result["entry_id"] = entry_name
                result["record_id"] = f"{spec.object_id}:{entry_name}:{name}:{mode}"
                result["feature"] = float(world.evidence(packets)[0][0])
                bundle["kernels"].append(result)
                if entry_name == "common" and name == "probe":
                    bundle["near"] = result["feature"]
                save_trajectory(output / f"local_{i}_{entry_index}_{action_index}.npz", backend.history)
                write_json(output / f"local_{i}_{entry_index}_{action_index}.json", result)
                print(json.dumps({"train": spec.object_id, "entry": entry_name, "action": result["action"],
                                  "status": result["status"], "duration": result["duration"]}), flush=True)
        bundles.append(bundle)
        audits.append({"spec": asdict(spec), "world_hash": world.digest})
    write_json(output / "local_training.json", bundles)
    write_json(output / "local_training_audit.json", audits)
    prepared = []
    for i, spec in enumerate(default_specs(2, "entry_test", seed * 10 + 5000, 60.)):
        world = CoreWorld(atlas, spec, "basic")
        backend.reset(np.asarray(grid["common"]))
        engine = EntryProgramExecutor(world, backend=backend).set_rng(seed + 5000 + i)
        result, packets = engine.call("near")
        near = float(world.evidence(packets)[0][0])
        save_trajectory(output / f"prepare_{i}.npz", backend.history)
        prepared.append({"spec": asdict(spec), "near": near, "preparation": result})
    write_json(output / "test_preparation.json", prepared)
    predictions = []
    for prep in prepared:
        for condition, parameters in conditions.items():
            visible = not parameters["hidden"] and parameters["release"] <= 0
            weights = posterior_weights(bundles, prep["near"], visible=visible)
            evaluations = [evaluate_tree(bundles, weights, grid["common"], t, deadlines, **parameters) for t in TREES]
            def score(row):
                return (row["completion_lower"][1], -row["unsupported_mass"], -row["expected_duration_with_unsupported_penalty"])
            chosen = max(evaluations, key=score)
            predictions.append({"object_id": prep["spec"]["object_id"], "condition": condition,
                                "posterior_weights": weights.tolist(), "old_near_visible_at_selection": visible,
                                "evaluations": evaluations, "selected_tree": chosen["tree"],
                                "selected_prediction": chosen})
    write_json(output / "prospective_predictions.json", predictions)
    freeze = {"prediction_sha256": hashlib.sha256((output / "prospective_predictions.json").read_bytes()).hexdigest(),
              "local_data_sha256": hashlib.sha256((output / "local_training.json").read_bytes()).hexdigest(),
              "before_complete_test": True, "time_monotonic": time.perf_counter()}
    write_json(output / "prediction_freeze.json", freeze)
    outcomes = []
    for pred in predictions:
        prep = next(r for r in prepared if r["spec"]["object_id"] == pred["object_id"])
        tree = next(t for t in TREES if t.name == pred["selected_tree"])
        world = CoreWorld(atlas, Spec(**prep["spec"]), "basic")
        backend.reset(np.asarray(grid["common"]))
        with patch("leg_program_bench.core_execution.ProgramExecutor", EntryProgramExecutor):
            result = core.execute_tree(world, prep["near"], tree, chain=("a", "f"),
                                       seed=seed + 7000 + int(pred["object_id"][-3:]),
                                       backend=backend, **conditions[pred["condition"]])
        result.update(object_id=pred["object_id"], condition=pred["condition"],
                      selected_tree=tree.name, selection_deadline=40.)
        save_trajectory(output / f"chain_{len(outcomes):03d}.npz", backend.history)
        write_json(output / f"chain_{len(outcomes):03d}.json", result)
        outcomes.append(result)
        print(json.dumps({"test": pred["object_id"], "condition": pred["condition"], "tree": tree.name,
                          "success": result["success"], "duration": result["duration"],
                          "statuses": [r["status"] for r in result["calls"]]}), flush=True)
    write_json(output / "outcomes.json", outcomes)
    errors = []
    unsupported = []
    for pred, result in zip(predictions, outcomes):
        target = np.asarray([result["success"] and result["duration"] <= t for t in deadlines], float)
        point = np.asarray(pred["selected_prediction"]["completion_lower"])
        errors.append(float(np.mean(abs(target - point))))
        unsupported.append(pred["selected_prediction"]["unsupported_mass"])
    summary = {"schema": protocol["schema"], "n_train_objects": 2, "n_local_programs": 18,
               "n_test_objects": 2, "n_full_chains": len(outcomes), "n_completed": sum(r["success"] for r in outcomes),
               "mean_conservative_curve_absolute_error": float(np.mean(errors)),
               "mean_selected_unsupported_mass": float(np.mean(unsupported)),
               "by_condition": {c: {"n": sum(r["condition"] == c for r in outcomes),
                                     "completed": sum(r["condition"] == c and r["success"] for r in outcomes),
                                     "mean_duration": float(np.mean([r["duration"] for r in outcomes if r["condition"] == c])),
                                     "selected_trees": [r["selected_tree"] for r in outcomes if r["condition"] == c]}
                                for c in conditions},
               "no_cartesian_transition_model": True, "all_failures_retained": True,
               "physical_execution": False, "contact_or_Bmode_validation": False,
               "formal_guarantee": False, "wall_seconds": time.perf_counter() - began}
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--backend", choices=("genesis_pd", "pinocchio_lag"), default="genesis_pd")
    args = parser.parse_args()
    print(json.dumps(run(args.output, seed=args.seed, backend_name=args.backend)))


if __name__ == "__main__":
    main()
