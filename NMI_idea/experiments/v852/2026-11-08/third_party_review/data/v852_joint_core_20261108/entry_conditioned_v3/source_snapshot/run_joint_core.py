"""Independent hard-tier A/B/C runner driven by measured Genesis joint FK.

This does not connect to hardware. It preserves the core tree/release/fresh
logic while optionally replacing global DLS moves with a bounded public SRS
goal search and densely checked joint-space transfer. All failures count.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from . import core_execution as core
from .core_execution import CoreWorld, ProgramExecutor, SPANS, TREES, predict_tree
from .dird_probe import DirdProbe
from .joint_execution import JointServoBackend, JointServoConfig
from .run_joint_execution import DEFAULT_CHECKPOINT, save_trajectory
from .simulation import Spec
from .smplx_scene import SmplxAtlasLeg


Q_NOMINAL = np.r_[.4, np.deg2rad([-89.5, -94.5, 65.2, 96., 89.3, 61., 94.6])]
RAILS = (.15, .3, .45, .6, .75)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def image_pose(world, progress, mode=0):
    T = np.eye(4)
    T[:3, 3], T[:3, :3] = world.reference_at(progress, mode)
    return T


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class RecordingBackend(JointServoBackend):
    def __init__(self, *args, **kwargs):
        self.history = []
        super().__init__(*args, **kwargs)

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        self.history = []
        return result

    def step(self, *args, **kwargs):
        row = super().step(*args, **kwargs)
        row.setdefault("phase", "image_servo")
        row.setdefault("waypoint_index", -1)
        self.history.append(row)
        return row

    def execute_path(self, *args, **kwargs):
        result = super().execute_path(*args, **kwargs)
        # Reinitialize command history from the measured settled exit, without
        # assigning the physical state or advancing/resetting simulation time.
        self._last_pose_command = self.q.copy()
        self.last_command_velocity = self.qdot.copy()
        result["pose_command_history_synced_to_measured_exit"] = True
        return result


class PublicGoalPlanner:
    """Finite SRS/rail proposals from public image poses, never hidden anatomy.

    The search visits all eight existing SRS branches on a fixed 10-degree
    swivel grid. dIRD orders rails, exact FK/limits/self-collision audit each
    returned pose. It is a finite search, not a global reachability proof.
    """
    def __init__(self, probe, backend, *, psi_step_deg=10.):
        self.probe = probe
        self.backend = backend
        self.psi_step_deg = psi_step_deg
        self.cache = {}
        self.events = []

    def candidates(self, T_image_world, q_seed=None):
        from rm75_control.kinematics.srs_ik import flange_tcp_from_kin, srs_ik, shoulder_y_from_q_rail
        target = np.linalg.inv(self.backend.base_world) @ T_image_world @ self.backend.T_image_tcp
        key = tuple(np.round(target.reshape(-1), 7))
        start = time.perf_counter()
        if key not in self.cache:
            R_tool, t_tool = flange_tcp_from_kin(self.backend.kin)
            pose6 = np.r_[target[:3, 3], Rotation.from_matrix(target[:3, :3]).as_euler("xyz")]
            rankings = self.probe.rank_modes(np.repeat((T_image_world @ self.backend.T_image_tcp)[None], 3, axis=0),
                                             self.backend.base_world, RAILS)
            rail_rank = {float(r["rail_m"][0]): r["rank"] for r in rankings["modes"]}
            solutions = []
            for rail in RAILS:
                for branch in range(8):
                    for psi in np.deg2rad(np.arange(-180., 180., self.psi_step_deg)):
                        arm = srs_ik(pose6, psi, branch, y_rail=shoulder_y_from_q_rail(rail),
                                     R_flange_tcp=R_tool, t_flange_tcp=t_tool)
                        if arm is None:
                            continue
                        q = np.r_[rail, arm]
                        if not self.backend._inside_limits(q):
                            continue
                        clearance = self.backend.clearance(q)
                        if clearance < self.backend.config.collision_margin_m:
                            continue
                        actual = self.backend.fk(q)
                        pe = np.linalg.norm(actual[:3, 3] - target[:3, 3])
                        re = Rotation.from_matrix(actual[:3, :3] @ target[:3, :3].T).magnitude()
                        if pe > .0005 or re > np.deg2rad(1):
                            continue
                        solutions.append({"q": q.tolist(), "rail": rail, "branch": branch,
                                          "psi": float(psi), "clearance": clearance,
                                          "rail_rank": rail_rank.get(rail, 99)})
            self.cache[key] = solutions
        q_seed = getattr(self.backend, "public_nominal_q", Q_NOMINAL) if q_seed is None else np.asarray(q_seed)
        scale = np.r_[.8, np.full(7, 3.)]
        solutions = sorted(self.cache[key], key=lambda r: (
            float(np.mean(((np.asarray(r["q"]) - q_seed) / scale) ** 2)) + .001 * r["rail_rank"]))
        self.events.append({"target_hash": content_hash(list(key)), "solutions": len(solutions),
                            "wall_seconds": time.perf_counter() - start})
        return solutions


class PlannedProgramExecutor(ProgramExecutor):
    """Same sensor/servo, with a finite checked free-space SetMode transfer."""
    planner = None

    def move(self, start, mode, max_time=20.):
        if not self.safe:
            return False
        if self.backend is None:
            return super().move(start, mode, max_time=max_time)
        before = time.perf_counter()
        target = image_pose(self.world, start, mode)
        current = np.asarray(self.backend.state()["q_meas"])
        candidates = self.planner.candidates(target, current)
        event = {"start": start, "mode": mode, "candidate_count": len(candidates), "attempts": []}
        for candidate in candidates[:24]:
            qgoal = np.asarray(candidate["q"])
            minimum, _ = self.backend._segment_clearance(current, qgoal)
            if minimum < self.backend.config.collision_margin_m:
                event["attempts"].append({"status": "joint_transfer_collision_rejection", "clearance": minimum})
                continue
            result = self.backend.execute_path(np.array([qgoal]), max_duration_s=max_time)
            self._sync_joint()
            self.trace.extend(result["trajectory"])
            event["attempts"].append({"status": result["status"], "duration": result["duration_s"]})
            event["planning_wall_seconds"] = time.perf_counter() - before - result["wall_seconds"]
            self.transfer_events.append(event)
            self.safe = result["status"] not in ("measured_state_collision_or_limit_stop",)
            return bool(result["success"])
        event["planning_wall_seconds"] = time.perf_counter() - before
        self.transfer_events.append(event)
        return False

    def __init__(self, *args, **kwargs):
        self.transfer_events = []
        super().__init__(*args, **kwargs)

    def call(self, *args, **kwargs):
        name = args[0] if args else kwargs.get("name")
        actual_args = args
        if name == "f" and SPANS["f"] == SPANS["probe"]:
            # Exact physical program reuse; freshness is an acceptance time,
            # not a different speed or an extra learned transition kernel.
            actual_args = ("probe",) + tuple(args[1:])
        result, packets = super().call(*actual_args, **kwargs)
        result["program"] = name
        result["independent_success"] = bool(result.get("independent_success", result["success"]))
        result["success"] = bool(result["public_success"] and result["independent_success"])
        result["scan_speed_m_s"] = .010 if actual_args[0] in ("near", "probe") else .028
        result["transfer_events"] = self.transfer_events[-1:]
        return result, packets


def fixed_registration(world, backend, anchor, q_nominal=Q_NOMINAL):
    bridge = np.eye(4)
    bridge[:3, :3] = Rotation.from_euler("x", np.pi).as_matrix()
    base = image_pose(world, anchor) @ bridge @ np.linalg.inv(backend.fk(q_nominal))
    return base, bridge


def public_screen(atlas, backend, probe, anchors=(.4, .45, .5, .55, .6)):
    reference = CoreWorld(atlas, Spec("public_registration_reference", 61108, 1, -1), "basic")
    screens = []
    selected = None
    designs = [(anchor, wrist) for wrist in (0., -np.pi/2, np.pi/2, np.pi) for anchor in anchors]
    for anchor, wrist in designs:
        nominal = Q_NOMINAL.copy(); nominal[-1] += wrist
        if not backend._inside_limits(nominal) or backend.clearance(nominal) < backend.config.collision_margin_m:
            screens.append({"anchor": anchor, "wrist_offset_rad": wrist,
                            "all_start_poses_supported": False, "nominal_state_rejected": True})
            continue
        backend.public_nominal_q = nominal
        backend.base_world, backend.T_image_tcp = fixed_registration(reference, backend, anchor, nominal)
        planner = PublicGoalPlanner(probe, backend)
        row = {"anchor": anchor, "wrist_offset_rad": wrist, "public_nominal_q": nominal.tolist(),
               "base_world": backend.base_world.tolist(),
               "T_image_tcp": backend.T_image_tcp.tolist(), "public_poses": {}}
        usable = True
        for pose in (0., 60.):
            world = CoreWorld(atlas, Spec("public_registration_reference", 61108, 1, -1, knee_deg=pose), "basic")
            states = {}
            for name, progress in (("near", .10), ("common", .60), ("probe", .56), ("a", .62), ("f", .14)):
                candidates = planner.candidates(image_pose(world, progress))
                states[name] = {"n_candidates": len(candidates), "q": candidates[0]["q"] if candidates else None}
                usable &= bool(candidates)
            row["public_poses"][str(pose)] = states
        row["all_start_poses_supported"] = bool(usable)
        screens.append(row)
        print(json.dumps({"public_screen_anchor": anchor, "wrist_offset_rad": wrist,
                          "all_start_poses_supported": bool(usable)}), flush=True)
        if usable and selected is None:
            selected = row
            break
    # Even if full coverage is absent, execute and record the declared most
    # distal-supported configuration; unsupported Return remains a failure.
    selected = selected or next(r for r in reversed(screens) if "base_world" in r)
    backend.base_world = np.asarray(selected["base_world"])
    backend.T_image_tcp = np.asarray(selected["T_image_tcp"])
    backend.public_nominal_q = np.asarray(selected["public_nominal_q"])
    return selected, screens, PublicGoalPlanner(probe, backend)


def default_specs(n, prefix, seed, pose):
    return [Spec(f"{prefix}_{i:03d}", seed + i, 1 if i % 2 == 0 else -1,
                 (1 if i % 2 == 0 else -1) if i % 4 < 2 else (-1 if i % 2 == 0 else 1),
                 knee_deg=pose) for i in range(n)]


def _start_q(planner, world, name, mode=0):
    progress = .60 if name == "common" else SPANS[name][0]
    choices = planner.candidates(image_pose(world, progress, mode))
    return np.asarray(choices[0]["q"]) if choices else None


def local_joint_bundle(atlas, spec, backend, planner, seed, output, executor_class):
    world = CoreWorld(atlas, spec, "basic")
    row = {"object_id": spec.object_id, "near": 0., "probe": 0.,
           "stratum": f"{spec.knee_deg:g}:1:1", "programs": {}}
    details = []
    for index, (name, mode) in enumerate((("near", 0), ("probe", 0), ("a", -1), ("a", 1), ("f", 0))):
        q = _start_q(planner, world, name, mode)
        if q is None:
            record = {"program": name, "mode": mode, "success": False, "status": "public_entry_unsupported",
                      "duration": 0., "entry_duration": 0., "service_duration": 0., "nominal_duration": 1.,
                      "public_success": False, "independent_success": False, "scan_speed_m_s": .010 if name in ("near", "probe") else .028,
                      "exit_position_local": [0., 0., 0.], "exit_rotation_local": [0., 0., 0.],
                      "exit_velocity_local": [0., 0., 0.], "frame_count": 0, "fresh_frames": 0,
                      "coverage": 0., "backend": backend.config.backend}
            packets = []
        else:
            backend.reset(q)
            engine = executor_class(world, backend=backend).set_rng(seed + index)
            record, packets = engine.call(name, mode, fresh_after=0.)
            if name in ("near", "probe"):
                row[name] = float(world.evidence(packets)[0][0])
            save_trajectory(output / f"local_{spec.object_id}_{name}_{mode:+d}.npz", backend.history)
        if name != "near":
            row["programs"][f"{name}:{mode}"] = record
        details.append(record)
        print(json.dumps({"local_object": spec.object_id, "program": name, "mode": mode,
                          "status": record["status"], "duration": record["duration"]}), flush=True)
    row["acquisition_seconds"] = sum(r["duration"] for r in details)
    return row, {"spec": asdict(spec), "world_hash": world.digest, "programs": details}


def run(output, *, train_objects=2, test_objects=2, backend_name="genesis_pd", controller="srs", screen_cache=None,
        task="full_leg", seed=20261109):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    if train_objects < 1 or test_objects < 1:
        raise ValueError("positive independent training and test object counts are required")
    if task not in ("full_leg", "local_continuation"):
        raise ValueError("unknown joint-core task")
    if task == "local_continuation":
        # This runner is an independent process: the primary suite's module
        # state and source are untouched. Explicitly record this separate task.
        SPANS["near"] = SPANS["probe"] = SPANS["f"] = (.56, .60)
    atlas = SmplxAtlasLeg.load()
    probe = DirdProbe(checkpoint=DEFAULT_CHECKPOINT, allow_stale_audit=False)
    backend = RecordingBackend(JointServoConfig(backend=backend_name, dt=.02))
    if screen_cache is None:
        selected, screens, planner = public_screen(atlas, backend, probe)
    else:
        screens = json.loads(Path(screen_cache).read_text())
        if task == "local_continuation":
            selected = next(r for r in screens if r.get("anchor") == .6 and r.get("wrist_offset_rad", 0.) == 0.)
        else:
            def coverage(row):
                points = [point for pose in row.get("public_poses", {}).values() for point in pose.values()]
                return sum(point["n_candidates"] > 0 for point in points)
            selected = next((r for r in screens if r.get("all_start_poses_supported")),
                            max((r for r in screens if "base_world" in r), key=coverage))
        backend.base_world = np.asarray(selected["base_world"])
        backend.T_image_tcp = np.asarray(selected["T_image_tcp"])
        backend.public_nominal_q = np.asarray(selected.get("public_nominal_q", Q_NOMINAL))
        reference = CoreWorld(atlas, Spec("public_registration_reference", 61108, 1, -1), "basic")
        expected_base, expected_bridge = fixed_registration(reference, backend, selected["anchor"], backend.public_nominal_q)
        if not np.allclose(backend.base_world, expected_base, atol=1e-9):
            raise ValueError("cached registration does not match the current public reference")
        planner = PublicGoalPlanner(probe, backend)
    write_json(output / "public_registration_screen.json", screens)
    PlannedProgramExecutor.planner = planner
    executor_class = PlannedProgramExecutor if controller == "srs" else ProgramExecutor
    context = patch("leg_program_bench.core_execution.ProgramExecutor", executor_class)
    protocol = {"schema": "v852_joint_core_development_v1", "backend": backend_name,
                "executor_version": "v2_public_policy_feedback_independent_terminal_audit",
                "task": task, "spans": {k: list(v) for k, v in SPANS.items()},
                "controller": controller, "executor_class": executor_class.__name__,
                "train_objects": train_objects, "test_objects": test_objects,
                "training": "independent joint-only local programs; no full-chain terminal labels",
                "selected_registration": selected, "plant": backend.manifest(),
                "public_screen_cache": str(screen_cache) if screen_cache is not None else None,
                "checkpoint": probe.metadata(), "chain": ["a", "f"],
                "trees": ["react_-1_+1", "probe_+1"],
                "conditions": {"early": {"release": 0., "hidden": False},
                               "late": {"release": 5., "hidden": False},
                               "hide": {"release": 0., "hidden": True}},
                "prediction": "persistent measured local object bundles; public Cartesian transition approximation, explicitly not a fully learned joint transfer kernel",
                "formal_guarantee": False, "physical_execution": False,
                "contact_and_Bmode": False, "fresh_rule": "f captures after f request, immutable old packet never accepted",
                "registration_rule": "at most 20 predeclared public anchor/wrist placements; all SRS branches, 10-degree swivel grid; no hidden vessel-dependent placement",
                "recheck_implementation": "same slow Probe kernel with fresh acceptance time" if task == "local_continuation" else "original f program",
                "seed": seed}
    write_json(output / "frozen_protocol.json", protocol)
    training = []
    local_audits = []
    with context:
        for i, spec in enumerate(default_specs(train_objects, "joint_train", seed * 10 + 1000, 0.)):
            row, audit = local_joint_bundle(atlas, spec, backend, planner, seed + 100 + i * 100, output, executor_class)
            training.append(row); local_audits.append(audit)
    write_json(output / "local_training.json", training)
    write_json(output / "local_training_audit.json", local_audits)
    # Test-time preparation is allowed; its outcomes cannot fit the model.
    prepared = []
    for i, spec in enumerate(default_specs(test_objects, "joint_test", seed * 10 + 5000, 60.)):
        world = CoreWorld(atlas, spec, "basic")
        q = _start_q(planner, world, "near")
        prep = {"status": "public_entry_unsupported", "duration": 0.}
        near = 0.
        if q is not None:
            backend.reset(q)
            engine = executor_class(world, backend=backend).set_rng(seed + 2000 + i)
            prep, packets = engine.call("near")
            near = float(world.evidence(packets)[0][0])
            save_trajectory(output / f"prepare_{spec.object_id}.npz", backend.history)
        common_q = _start_q(planner, world, "common")
        prepared.append({"spec": asdict(spec), "near": near, "preparation": prep,
                         "common_q": common_q.tolist() if common_q is not None else None,
                         "world_hash": world.digest})
    write_json(output / "test_preparation.json", prepared)
    chosen_trees = [t for t in TREES if t.name in protocol["trees"]]
    deadlines = np.asarray([20., 40., 60., 90.])
    predictions = []
    for prep in prepared:
        world = CoreWorld(atlas, Spec(**prep["spec"]), "basic")
        for condition, kwargs in protocol["conditions"].items():
            for tree in chosen_trees:
                # Whole-object sampling preserves all measured local nuisance.
                weights = np.array([np.exp(-.5 * ((r["near"] - prep["near"]) / .08) ** 2)
                                    if not kwargs["hidden"] and kwargs["release"] == 0 else 1. for r in training])
                weights /= weights.sum()
                samples = [predict_tree(world, row, tree, chain=("a", "f"), **kwargs) for row in training]
                curve = [float(sum(w * (r["success"] and r["duration"] <= d) for w, r in zip(weights, samples))) for d in deadlines]
                predictions.append({"object_id": prep["spec"]["object_id"], "condition": condition,
                                    "tree": tree.name, "deadlines": deadlines.tolist(), "completion": curve,
                                    "local_bundle_rollouts": samples,
                                    "interval_kind": "unidentified transfer kernel: conservative [0,1]",
                                    "lower": [0.] * len(deadlines), "upper": [1.] * len(deadlines)})
    write_json(output / "prospective_predictions.json", predictions)
    freeze = {"prediction_sha256": hashlib.sha256((output / "prospective_predictions.json").read_bytes()).hexdigest(),
              "local_data_sha256": hashlib.sha256((output / "local_training.json").read_bytes()).hexdigest(),
              "frozen_before_complete_test": True, "time_monotonic": time.perf_counter()}
    write_json(output / "prediction_freeze.json", freeze)
    outcomes = []
    with patch("leg_program_bench.core_execution.ProgramExecutor", executor_class):
        for prep in prepared:
            spec = Spec(**prep["spec"])
            for condition, kwargs in protocol["conditions"].items():
                for tree in chosen_trees:
                    if prep["common_q"] is None:
                        outcome = {"tree": tree.name, "success": False, "status": "public_common_entry_unsupported",
                                   "duration": 0., "calls": [], "decisions": [], "program_calls": 0,
                                   "fresh_frames": 0, "new_frames": 0, "recoveries": 0,
                                   "packet_hash": content_hash({"near": prep["near"]})}
                    else:
                        backend.reset(np.asarray(prep["common_q"]))
                        world = CoreWorld(atlas, spec, "basic")
                        outcome = core.execute_tree(world, prep["near"], tree, chain=("a", "f"),
                                                    seed=seed + 3000 + int(spec.object_id[-3:]), backend=backend, **kwargs)
                        save_trajectory(output / f"chain_{len(outcomes):03d}.npz", backend.history)
                    outcome.update(object_id=spec.object_id, condition=condition, preparation_status=prep["preparation"]["status"])
                    outcomes.append(outcome)
                    write_json(output / f"chain_{len(outcomes) - 1:03d}.json", outcome)
                    print(json.dumps({"object_id": spec.object_id, "condition": condition, "tree": tree.name,
                                      "success": outcome["success"], "duration": outcome["duration"],
                                      "statuses": [r["status"] for r in outcome["calls"]]}), flush=True)
    write_json(output / "outcomes.json", outcomes)
    errors = []
    for pred, actual in zip(predictions, outcomes):
        observed = np.asarray([actual["success"] and actual["duration"] <= d for d in pred["deadlines"]], float)
        errors.append(float(np.mean(abs(np.asarray(pred["completion"]) - observed))))
    summary = {"schema": protocol["schema"], "backend": backend_name, "controller": controller,
               "task": task, "spans": protocol["spans"],
               "n_train_objects": train_objects, "n_test_objects": test_objects,
               "n_full_chains": len(outcomes), "n_completed": sum(r["success"] for r in outcomes),
               "mean_curve_absolute_error": float(np.mean(errors)),
               "all_failures_retained": True, "formal_guarantee": False,
               "prediction_transfer_model_is_approximate": True,
               "physical_execution": False, "contact_or_Bmode_validation": False,
               "by_condition": {c: {"attempted": sum(r["condition"] == c for r in outcomes),
                                     "completed": sum(r["condition"] == c and r["success"] for r in outcomes),
                                     "mean_duration": float(np.mean([r["duration"] for r in outcomes if r["condition"] == c]))}
                                for c in protocol["conditions"]},
               "wall_seconds": time.perf_counter() - began}
    write_json(output / "summary.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-objects", type=int, default=2)
    p.add_argument("--test-objects", type=int, default=2)
    p.add_argument("--backend", choices=("genesis_pd", "pinocchio_lag"), default="genesis_pd")
    p.add_argument("--controller", choices=("dls", "srs"), default="srs")
    p.add_argument("--screen-cache", type=Path)
    p.add_argument("--task", choices=("full_leg", "local_continuation"), default="full_leg")
    p.add_argument("--seed", type=int, default=20261109)
    args = p.parse_args()
    print(json.dumps(run(args.output, train_objects=args.train_objects, test_objects=args.test_objects,
                         backend_name=args.backend, controller=args.controller, screen_cache=args.screen_cache,
                         task=args.task, seed=args.seed)))


if __name__ == "__main__":
    main()
