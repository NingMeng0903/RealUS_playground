"""Paired closed-loop experiment with a public policy and a private renderer."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
from threadpoolctl import threadpool_limits

import motion
from planner import METHODS, Planner, initial_belief
import sensor
from sensor import INITIAL_MODE, MODE_NAMES, OUTCOMES, PUBLIC_GATE, SOURCE_NAMES, TRANSITIONS, env


HERE = Path(__file__).resolve().parent
METRICS = (
    "success", "false_stop", "restricted_time_s", "recovery_exposed",
    "recovered", "loss_events", "recovery_events",
)
PAIRS = (
    ("B_two", "A_task"),
    ("B_two", "B_fixed"),
    ("B_two", "B_no_future_cap"),
    ("A_task", "A_geometry"),
    ("A_task", "A_iid"),
    ("A_task", "A_hard"),
)


@dataclass(frozen=True)
class Context:
    identifier: str
    index: int
    initial_index: int
    available: np.ndarray
    joint_time: np.ndarray
    cost: np.ndarray
    start_found: bool


@dataclass(frozen=True)
class PrivateGeometry:
    visible: np.ndarray
    qualified: np.ndarray
    center_m: np.ndarray
    length_m: np.ndarray
    angle_rad: np.ndarray


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _nullable(values):
    values = np.asarray(values)
    return [float(value) if np.isfinite(value) else None for value in values]


def _shifted_worlds(seed, count, geometry_spec):
    y_shift = float(geometry_spec["signed_registration_y_shift_m"])
    slope_shift = float(geometry_spec["signed_curve_slope_shift"])
    shifted = []
    for index, world in enumerate(env.sample_worlds(seed, count)):
        sign = 1 if index % 2 == 0 else -1
        registration = world.registration + np.array([0.0, sign * y_shift])
        controls = world.y_controls + sign * slope_shift * env.CONTROL_X[None, :]
        points = env._points(registration, world.modes, controls, world.z_controls, world.depths)
        shifted.append(replace(world, registration=registration, y_controls=controls, points=points))
    return shifted


def _private_geometry(worlds, poses):
    prediction = env.predict_batch(worlds, poses)
    length = prediction["geometric_length"][..., 0]
    center = prediction["means"][..., 0, 0]
    angle = prediction["angle_audit"][..., 0]
    qualified = (
        prediction["contact"] & (length >= sensor.LENGTH_THRESHOLD)
        & (np.abs(center) <= 0.002) & (angle <= np.deg2rad(10.0))
    )
    return PrivateGeometry(length > 0.0005, qualified, center, length, angle)


def _label_swap_control(worlds, poses):
    swapped = [
        replace(
            world, modes=world.modes[::-1], y_controls=world.y_controls[::-1],
            z_controls=world.z_controls[::-1], depths=world.depths[::-1],
            identity_z=world.identity_z[::-1], detect_z=world.detect_z[::-1],
            points=world.points[::-1],
        )
        for world in worlds
    ]
    blind = sensor.channel(worlds, poses, identity_blind=True)
    blind_swapped = sensor.channel(swapped, poses, identity_blind=True)
    return {
        "worlds": len(worlds),
        "max_abs_public_likelihood_change": float(
            np.max(np.abs(blind.likelihood - blind_swapped.likelihood))
        ),
        "max_abs_target_joint_vs_swapped_decoy": float(
            np.max(np.abs(blind.joint[..., 0, :] - blind_swapped.joint[..., 1, :]))
        ),
        "max_abs_private_correct_mass_change": float(
            np.max(np.abs(blind.correct_joint - blind_swapped.correct_joint))
        ),
    }


def _uniforms(seed, object_count, context_count, regimes, steps):
    """Pair all methods through a separate renderer stream for each group."""
    result = np.empty((object_count, context_count, len(regimes), steps + 1, 2))
    for obj in range(object_count):
        for ctx in range(context_count):
            for regime, name in enumerate(regimes):
                regime_id = tuple(TRANSITIONS).index(name)
                key = np.random.SeedSequence([seed, obj, ctx, regime_id, 20261111])
                result[obj, ctx, regime] = np.random.default_rng(key).random((steps + 1, 2))
    return result


def _mode_schedule(regime, uniforms, steps):
    modes = np.empty(steps, dtype=int)
    modes[0] = int(uniforms[0, 0] >= INITIAL_MODE[regime][0])
    for step in range(1, steps):
        modes[step] = int(uniforms[step, 0] >= TRANSITIONS[regime][modes[step - 1], 0])
    return modes


def _draw(cdf, object_index, action, mode, uniform):
    event = int(np.searchsorted(cdf[object_index, mode, action], uniform, side="right"))
    return divmod(event, len(OUTCOMES))


def _model_regime(method, regime):
    return "iid" if method == "A_iid" and regime == "bursty" else regime


def _rollout(
    method, planner, regime, context, poses, free_cost, motion_time, geometry,
    private_cdf, uniforms, object_index, seed, cohort, motion_name, task,
):
    deadline = float(task["deadline_s"])
    steps = int(task["max_actions"])
    observation = float(task["observation_s"])
    current = context.initial_index
    model_regime = _model_regime(method, regime)
    belief = initial_belief(planner.world_count, INITIAL_MODE[model_regime])
    modes = _mode_schedule(regime, uniforms, steps)
    pending = None
    followup_due = False
    elapsed = 0.0
    ledger = []
    first_public_gate = None
    ever_target = False
    loss_open = False
    losses = 0
    recoveries = 0
    stop = None
    failure = None
    terminal_decision = None
    reachable = (
        context.available & np.isfinite(context.cost[current])
        & (context.cost[current] <= deadline)
    )
    oracle = {
        "feasible_candidate_count": int(reachable.sum()),
        "geometrically_visible_count": int((reachable & geometry.visible[object_index]).sum()),
        "qualified_candidate_count": int((reachable & geometry.qualified[object_index]).sum()),
    }

    if not context.start_found:
        failure = "initial_endpoint_unavailable"
    for ordinal in range(steps if failure is None else 0):
        remaining = deadline - elapsed
        planning_started = time.perf_counter()
        planning_budget = remaining
        executed_policy = method
        fallback_reason = None
        if followup_due:
            if pending is None:
                fallback_reason = "no_forecast_continuation"
            elif not context.available[pending] or not np.isfinite(context.cost[current, pending]):
                fallback_reason = "forecast_endpoint_unavailable"
            elif context.cost[current, pending] > remaining:
                fallback_reason = "forecast_action_unaffordable"
            if fallback_reason is not None:
                executed_policy = "A_task"
        plan = planner.plan(
            executed_policy, belief, current, remaining, steps - ordinal,
            context.available, context.cost, free_cost,
            pending=pending if followup_due and fallback_reason is None else None,
        )
        planning_wall = time.perf_counter() - planning_started
        elapsed += planning_wall
        action = plan.action
        if (
            followup_due and fallback_reason is None and action is not None
            and elapsed + context.cost[current, action] > deadline
        ):
            fallback_reason = "forecast_action_unaffordable_after_planning"
            executed_policy = "A_task"
            fallback_started = time.perf_counter()
            planning_budget = deadline - elapsed
            plan = planner.plan(
                executed_policy, belief, current, planning_budget, steps - ordinal,
                context.available, context.cost, free_cost,
            )
            fallback_wall = time.perf_counter() - fallback_started
            planning_wall += fallback_wall
            elapsed += fallback_wall
            action = plan.action
        decision = {
            "ordinal": ordinal,
            "from_index": current,
            "action_index": action,
            "pending_from_previous": pending,
            "followup_due": followup_due,
            "executed_policy": executed_policy,
            "forecast_invalid_fallback": fallback_reason,
            "remaining_before_planning_s": remaining,
            "planning_budget_s": planning_budget,
            "remaining_actions": steps - ordinal,
            "planning_wall_s": planning_wall,
            "scores": _nullable(plan.scores),
            "immediate_scores": _nullable(plan.immediate_scores),
            "predicted_continuation": (
                None if plan.continuations is None else plan.continuations.tolist()
            ),
            "predicted_continuation_saved_s": (
                None if plan.continuation_scores is None else plan.continuation_scores.tolist()
            ),
            "fixed_action": plan.fixed_action,
            "predictive_mode_probability": belief.sum(axis=0).tolist(),
        }
        if action is None:
            terminal_decision = decision
            failure = "no_affordable_action"
            break

        total_action = float(context.cost[current, action])
        if elapsed + total_action > deadline:
            terminal_decision = decision
            failure = "deadline_after_planning"
            break

        transit = total_action - observation
        elapsed += total_action
        time_observed = elapsed
        mode = int(modes[ordinal])
        uniform = float(uniforms[ordinal + 1, 1])
        source, outcome = _draw(private_cdf, object_index, action, mode, uniform)
        public_gate = bool(PUBLIC_GATE[outcome])
        private_qualified = bool(geometry.qualified[object_index, action])
        correct = source == 0 and private_qualified and public_gate

        reopened_target = False
        new_loss = False
        if source == 0:
            reopened_target = loss_open
            recoveries += int(reopened_target)
            loss_open = False
            ever_target = True
        elif ever_target and not loss_open:
            new_loss = True
            losses += 1
            loss_open = True

        update_started = time.perf_counter()
        assessment = planner.assess(belief, action)
        accepted = bool(assessment.accepted[outcome])
        likelihood = float(assessment.probability[outcome])
        correct_mass = float(assessment.correct_mass[outcome])
        if not accepted:
            try:
                belief = planner.advance(belief, action, outcome, hard=method == "A_hard")
            except ValueError:
                failure = "zero_model_support"
        update_wall = time.perf_counter() - update_started
        elapsed += update_wall

        if method == "A_task" and public_gate and first_public_gate is None:
            first_public_gate = {
                "ordinal": ordinal,
                "elapsed_s": time_observed,
                "correct": correct,
            }
        decision.update({
            "pose": poses[action].tolist(),
            "reference_motion_s": float(motion_time[current, action]),
            "joint_displacement_s": float(context.joint_time[current, action]),
            "charged_motion_s": transit,
            "observation_s": observation,
            "time_observed_s": time_observed,
            "update_wall_s": update_wall,
            "elapsed_after_update_s": elapsed,
            "outcome_index": outcome,
            "public_gate": public_gate,
            "observation_probability": likelihood,
            "correct_joint_mass": correct_mass,
            "posterior_correct_probability": correct_mass / likelihood if likelihood > 0 else None,
            "accepted_by_posterior_gate": accepted,
            "next_mode_probability": None if accepted or failure else belief.sum(axis=0).tolist(),
        })
        ledger.append({
            "public": decision,
            "audit": {
                "mode": MODE_NAMES[mode],
                "source": SOURCE_NAMES[source],
                "target_in_fov": bool(geometry.visible[object_index, action]),
                "endpoint_qualified": private_qualified,
                "target_center_m": float(geometry.center_m[object_index, action]),
                "target_geometric_length_m": float(geometry.length_m[object_index, action]),
                "target_out_of_plane_angle_rad": float(geometry.angle_rad[object_index, action]),
                "correct_if_stopped": correct,
                "loss_started": new_loss,
                "loss_open": loss_open,
                "reacquired": reopened_target,
            },
        })
        if elapsed > deadline:
            failure = "deadline_after_update"
        if failure is not None:
            break
        if accepted:
            stop = {"ordinal": ordinal, "elapsed_s": elapsed, "correct": correct}
            failure = None if correct else "wrong_stop"
            break
        current = action
        if followup_due or plan.continuations is None:
            followup_due = False
            pending = None
        else:
            followup_due = True
            forecast = int(plan.continuations[outcome])
            pending = forecast if forecast >= 0 else None

    if failure is None and stop is None:
        failure = "action_limit"
    success = bool(stop and stop["correct"])
    return {
        "cohort": cohort,
        "object_id": object_index,
        "private_world_seed": seed,
        "context": context.identifier,
        "context_index": context.index,
        "regime": regime,
        "method": method,
        "motion": motion_name,
        "initial_pose_index": context.initial_index,
        "deadline_s": deadline,
        "max_actions": steps,
        "model_regime": model_regime,
        "elapsed_s": elapsed,
        "restricted_time_s": elapsed if success else deadline,
        "success": success,
        "false_stop": bool(stop and not stop["correct"]),
        "failure": failure,
        "stop": stop,
        "first_public_gate_counterfactual": first_public_gate,
        "recovery_exposed": losses > 0,
        "recovered": recoveries > 0,
        "loss_events": losses,
        "recovery_events": recoveries,
        "oracle": oracle,
        "terminal_decision": terminal_decision,
        "blocks": ledger,
    }


def _group_summary(rows):
    count = len(rows)
    exposed = sum(row["recovery_exposed"] for row in rows)
    recovered = sum(row["recovered"] for row in rows)
    loss_events = sum(row["loss_events"] for row in rows)
    recovery_events = sum(row["recovery_events"] for row in rows)
    outcomes = [
        block["audit"] for row in rows for block in row["blocks"]
    ]
    failures = {}
    fallbacks = {}
    for row in rows:
        if row["failure"] is not None:
            failures[row["failure"]] = failures.get(row["failure"], 0) + 1
        decisions = [block["public"] for block in row["blocks"]]
        if row["terminal_decision"] is not None:
            decisions.append(row["terminal_decision"])
        for decision in decisions:
            reason = decision["forecast_invalid_fallback"]
            if reason is not None:
                fallbacks[reason] = fallbacks.get(reason, 0) + 1
    result = {
        "trials": count,
        "success_count": sum(row["success"] for row in rows),
        "success_rate": sum(row["success"] for row in rows) / count,
        "false_stop_count": sum(row["false_stop"] for row in rows),
        "false_stop_rate": sum(row["false_stop"] for row in rows) / count,
        "mean_restricted_time_s": sum(row["restricted_time_s"] for row in rows) / count,
        "recovery_exposed_trials": exposed,
        "recovered_trials": recovered,
        "recovery_rate_given_exposure": recovered / exposed if exposed else None,
        "loss_events": loss_events,
        "recovery_events": recovery_events,
        "recovery_rate_per_loss": recovery_events / loss_events if loss_events else None,
        "acquisitions": len(outcomes),
        "source_counts": {
            name: sum(event["source"] == name for event in outcomes) for name in SOURCE_NAMES
        },
        "target_in_fov_acquisitions": sum(event["target_in_fov"] for event in outcomes),
        "in_fov_without_target_return": sum(
            event["target_in_fov"] and event["source"] != "target" for event in outcomes
        ),
        "qualified_endpoint_acquisitions": sum(event["endpoint_qualified"] for event in outcomes),
        "oracle_any_geometrically_visible": sum(
            row["oracle"]["geometrically_visible_count"] > 0 for row in rows
        ),
        "oracle_any_qualified_candidate": sum(
            row["oracle"]["qualified_candidate_count"] > 0 for row in rows
        ),
        "failures": failures,
        "forecast_invalid_fallbacks": fallbacks,
    }
    if rows[0]["method"] == "A_task":
        first = [row["first_public_gate_counterfactual"] for row in rows]
        result["same_path_first_public_gate"] = {
            "success_count": sum(event is not None and event["correct"] for event in first),
            "false_stop_count": sum(event is not None and not event["correct"] for event in first),
            "mean_restricted_time_s": sum(
                event["elapsed_s"] if event is not None and event["correct"] else row["deadline_s"]
                for event, row in zip(first, rows)
            ) / count,
        }
    return result


def _paired_deltas(rows, methods, bootstrap_draws=4000):
    """Resample independent objects, keeping every placement paired."""
    estimates = []
    by_method = {}
    for row in rows:
        by_method.setdefault(row["method"], {})[(row["object_id"], row["context"])] = row
    for left, right in PAIRS:
        if left not in methods or right not in methods:
            continue
        lrows, rrows = by_method[left], by_method[right]
        if lrows.keys() != rrows.keys():
            raise ValueError(f"unpaired rows for {left} vs {right}")
        by_object = {}
        for key in sorted(lrows):
            delta = [
                float(lrows[key][metric]) - float(rrows[key][metric])
                for metric in METRICS
            ]
            by_object.setdefault(key[0], []).append(delta)
        contrasts = np.asarray([np.mean(value, axis=0) for value in by_object.values()])
        rng = np.random.default_rng(721120)
        bootstrap_index = rng.integers(len(contrasts), size=(bootstrap_draws, len(contrasts)))
        sampled = contrasts[bootstrap_index].mean(axis=1)
        interval = np.quantile(sampled, [0.025, 0.975], axis=0).T
        estimates.append({
            "left": left,
            "right": right,
            "independent_objects": len(contrasts),
            "delta_left_minus_right": {
                metric: {"mean": float(mean), "object_bootstrap_95": bounds.tolist()}
                for metric, mean, bounds in zip(METRICS, contrasts.mean(axis=0), interval)
            },
        })
    return estimates


def _summarize(rows, regimes, contexts, methods):
    groups = []
    pairs = []
    for regime in regimes:
        for context in ["pooled", *(context.identifier for context in contexts)]:
            subset = [
                row for row in rows
                if row["regime"] == regime and (context == "pooled" or row["context"] == context)
            ]
            for method in methods:
                selected = [row for row in subset if row["method"] == method]
                groups.append({
                    "regime": regime, "context": context, "method": method,
                    **_group_summary(selected),
                })
            pairs.append({
                "regime": regime, "context": context,
                "comparisons": _paired_deltas(subset, methods),
            })
    return {"groups": groups, "paired_differences": pairs}


def _run(cohort, output, motion_name, capability_path):
    protocol_path = HERE / "PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text())
    capability = json.loads(capability_path.read_text())
    source_files = [
        protocol_path, capability_path, HERE / "planner.py", HERE / "run_stress.py",
        Path(sensor.__file__), Path(motion.__file__), Path(env.__file__),
    ]
    source_sha256 = {str(path): _sha256(path) for path in source_files}
    geometry_spec = protocol["geometry"]
    task = protocol["task"]
    prefix = "development" if cohort == "dev" else "test"
    seed = int(geometry_spec[f"{prefix}_seed"])
    count = int(geometry_spec[f"{prefix}_objects"])
    regimes = protocol["observation"]["regimes"] if motion_name == "prodmp" else ["bursty"]
    methods = protocol["methods"] if motion_name == "prodmp" else ["A_task", "B_two"]
    if not set(methods).issubset(METHODS):
        raise ValueError("protocol contains an unknown planner")
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "rows": output / "rows.jsonl",
        "summary": output / "summary.json",
        "model_inputs": output / "model_inputs.npz",
        "renderer_audit": output / "renderer_audit.npz",
    }

    poses = np.asarray(capability["poses"], dtype=float)
    motion_time = motion.duration_matrix(poses, method=motion_name)
    observation = float(task["observation_s"])
    free_cost = motion_time + observation
    contexts = []
    for index, context in enumerate(capability["contexts"]):
        joint_time = np.asarray(context["joint_time_s"], dtype=float)
        available = np.asarray(context["available"], dtype=bool)
        initial_index = int(context["initial_index"])
        contexts.append(Context(
            context["id"], index, initial_index, available, joint_time,
            np.maximum(motion_time, joint_time) + observation,
            bool(context.get("start_found", available[initial_index])),
        ))

    world_bank = env.sample_worlds(geometry_spec["public_prior_seed"], geometry_spec["public_prior_count"])
    worlds = _shifted_worlds(seed, count, geometry_spec)
    model = sensor.channel(world_bank, poses)
    private = sensor.channel(worlds, poses)
    geometry = _private_geometry(worlds, poses)
    if not np.array_equal(geometry.visible, private.geometric_target_visible):
        raise ValueError("private channel and geometry disagree on target visibility")
    expected_correct = (
        private.joint[..., 0, :] * geometry.qualified[:, None, :, None]
        * PUBLIC_GATE[None, None, None, :]
    )
    if not np.array_equal(private.correct_joint, expected_correct):
        raise ValueError("private channel and geometry disagree on correct stop mass")
    control = _label_swap_control(world_bank[:3] + worlds[:3], poses)
    if control["max_abs_public_likelihood_change"] > 1e-12:
        raise ValueError("identity-blind label swap changed public likelihood")

    cdf = np.cumsum(private.joint.reshape(count, 2, len(poses), -1), axis=-1)
    cdf /= cdf[..., -1:]
    uniforms = _uniforms(seed, count, len(contexts), regimes, int(task["max_actions"]))
    np.savez_compressed(
        paths["model_inputs"], poses=poses, likelihood=model.likelihood,
        correct_joint=model.correct_joint, joint=model.joint, public_gate=PUBLIC_GATE,
        motion_time_s=motion_time, motion_and_observation_s=free_cost,
        context_ids=np.asarray([context.identifier for context in contexts]),
        initial_index=np.asarray([context.initial_index for context in contexts]),
        available=np.stack([context.available for context in contexts]),
        joint_time_s=np.stack([context.joint_time for context in contexts]),
        action_cost_s=np.stack([context.cost for context in contexts]),
        mode_model_ids=np.asarray(tuple(TRANSITIONS)),
        mode_transitions=np.stack(list(TRANSITIONS.values())),
        stationary_modes=np.stack(list(INITIAL_MODE.values())),
    )
    np.savez_compressed(
        paths["renderer_audit"], joint=private.joint, uniforms=uniforms,
        target_visible=geometry.visible, target_qualified=geometry.qualified,
        target_center_m=geometry.center_m, target_geometric_length_m=geometry.length_m,
        target_out_of_plane_angle_rad=geometry.angle_rad,
        world_points=np.stack([world.points for world in worlds]), regime_ids=np.asarray(regimes),
    )
    planners = {
        name: Planner(
            model.likelihood, model.correct_joint, model.joint, PUBLIC_GATE, transition,
            threshold=task["stop_probability"],
        )
        for name, transition in TRANSITIONS.items()
    }
    rows = []
    print(f"{cohort}: {count} independent objects × {len(contexts)} contexts × "
          f"{len(regimes)} regimes × {len(methods)} methods", flush=True)
    with paths["rows"].open("w") as file:
        for object_index in range(count):
            for context in contexts:
                for condition_index, regime in enumerate(regimes):
                    for method in methods:
                        planner = planners[_model_regime(method, regime)]
                        row = _rollout(
                            method, planner, regime, context, poses, free_cost, motion_time, geometry,
                            cdf, uniforms[object_index, context.index, condition_index],
                            object_index, seed, cohort, motion_name, task,
                        )
                        file.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
                        rows.append(row)
            file.flush()
            print(f"{cohort}: {object_index + 1}/{count} objects complete", flush=True)

    summary = _summarize(rows, regimes, contexts, methods)
    summary["identity_blind_label_swap"] = control
    summary["static_capability"] = {
        context.identifier: {
            "available_poses": int(context.available.sum()),
            "pose_count": len(poses),
            "joint_time_exceeds_reference_edges": int(
                (context.joint_time > motion_time).sum()
            ),
        }
        for context in contexts
    }
    _write_json(paths["summary"], summary)
    _write_json(output / "manifest.json", {
        "cohort": cohort,
        "motion": motion_name,
        "world_seed": seed,
        "objects": count,
        "contexts": [context.identifier for context in contexts],
        "regimes": regimes,
        "methods": methods,
        "rows": len(rows),
        "public_outcomes": OUTCOMES,
        "source_order": SOURCE_NAMES,
        "mode_order": MODE_NAMES,
        "protocol": protocol,
        "source_sha256": source_sha256,
        "artifacts": {
            name: {"file": path.name, "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "blas_threads": 1,
        },
    })
    print(output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=["dev", "test"], required=True)
    parser.add_argument("--motion", choices=["prodmp", "dmp"], default="prodmp")
    parser.add_argument("--output", type=Path, required=True, help="New results directory")
    parser.add_argument(
        "--capability", type=Path, default=HERE / "CAPABILITY_STRESS.json",
        help="Static endpoint and joint-time fixture",
    )
    args = parser.parse_args()
    with threadpool_limits(limits=1, user_api="blas"):
        _run(args.cohort, args.output.resolve(), args.motion, args.capability.resolve())


if __name__ == "__main__":
    main()
