"""Run a full paired Y scan with a persistent internal posterior.

This study uses synthetic, unlabelled cross-section features, ideal contact,
the repaired posed leg chart, and robot-checked static skin viewpoints.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import time

import numpy as np

from bifurcation_model import (
    Belief, FORK, JUNCTION, LOWER, UPPER, SectionGeometry,
    draw_truth, graph_nodes, predictive_bins, render, target_points,
)
from continuous_map import Event, JOINT_IDENTITY_GATE, MIN_ESS, refine, review_coverage
from gps_projection import (
    FORK_SECTION, JUNCTION_SECTIONS, LOWER_SECTIONS, SECTIONS, TRUNK_SECTIONS, UPPER_SECTIONS,
    build_robot_table, make_banks, projected_gps,
)
from learned_prodmp import LearnedProDMP
from leg_chart import LegChart


HERE = Path(__file__).resolve().parent
EPISODE_SEED = 718201
IDENTITY_GATE = 0.88
MAP_RADIUS_MM = 3.0
MAX_SEGMENTS = 62
TRAVEL_COST = 0.00125
FUTURE_VALUE_WEIGHT = 0.65
SKILLS = ("advance", "sweep_left", "sweep_right")
METHODS = ("A_joint", "A_independent", "B_joint", "B_open_loop", "B_endpoint")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_row(handle, row):
    handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MotionLibrary:
    def __init__(self, primitive):
        self.primitive = primitive

    @lru_cache(maxsize=24000)
    def get(self, start, goal, skill, learned):
        return self.primitive.path(start, goal, skill=skill, samples=21, learned=learned)


class ScanContext:
    """Cache public model projections by pose and dependence assumption."""

    def __init__(self, geometry, robot_pose_row, shared):
        self.geometry = geometry
        belief = Belief.initialize(shared)
        self.theta = belief.theta
        self.log_prior = belief.log_weights
        self.banks = make_banks(geometry, self.theta, robot_pose_row)
        self.graph = geometry.body_world(graph_nodes(self.theta))
        self.graph_flat = self.graph.reshape(len(self.graph), -1)
        self.graph_sq = self.graph_flat**2
        self.targets = {s: geometry.body_world(target_points(self.theta, s.x, s.label)) for s in SECTIONS}

    def start_belief(self):
        return Belief(self.theta, self.log_prior.copy(), set())

    def map_readout(self, weights):
        center = (weights @ self.graph_flat).reshape(-1, 3)
        variance = np.maximum(0.0, (weights @ self.graph_sq).reshape(-1, 3) - center**2)
        deviation = np.max(np.linalg.norm(self.graph - center, axis=-1), axis=1)
        order = np.argsort(deviation, kind="stable")
        rank = min(len(order) - 1, int(np.searchsorted(np.cumsum(weights[order]), .9)))
        return center, variance, float(deviation[order[rank]])

    def most_uncertain_section(self, weights):
        def trace_var(section):
            points = self.targets[section]
            return float(weights @ np.sum((points - weights @ points)**2, axis=1))

        return max(SECTIONS, key=trace_var)


def map_metrics(readout, truth_world):
    mean, variance, radius = readout
    error = np.linalg.norm(mean - truth_world, axis=1)
    return {"rmse_mm": float(np.sqrt(np.mean(error**2))), "max_error_mm": float(error.max()),
            "node_rms_sd_mm": float(np.sqrt(variance.sum(axis=1).mean())),
            "joint_90_radius_mm": radius, "joint_90_covered": bool(error.max() <= radius + 1.e-9),
            "trunk_rmse_mm": float(np.sqrt(np.mean(error[:9]**2))),
            "lower_rmse_mm": float(np.sqrt(np.mean(error[9:17]**2))),
            "upper_rmse_mm": float(np.sqrt(np.mean(error[17:]**2)))}


def chosen_indices(identities, label):
    """Public association check for one fresh end-frame."""
    if identities is None or not identities.size:
        return ()
    if label == FORK:
        left, right = (int(np.argmax(identities[row])) for row in (LOWER, UPPER))
        if left != right and min(identities[LOWER, left], identities[UPPER, right]) >= IDENTITY_GATE:
            return left, right
    elif label == JUNCTION:
        real_lumen = identities.sum(axis=0)
        i = int(np.argmax(real_lumen))
        if real_lumen[i] >= IDENTITY_GATE:
            return (i,)
    else:
        i = int(np.argmax(identities[label]))
        if identities[label, i] >= IDENTITY_GATE:
            return (i,)
    return ()


def is_correct(indices, label, truth_labels):
    if label == FORK:
        return list(truth_labels[list(indices)]) == [LOWER, UPPER]
    if label == JUNCTION:
        return truth_labels[indices[0]] >= 0
    return truth_labels[indices[0]] == label


def certify(context, belief, events, evidence, seed, shared, continuous_refinement):
    updated = refine(context.geometry, belief, events, seed, shared) if continuous_refinement else None
    if updated is None:
        updated = {"theta": belief.theta, "log_weights": belief.log_weights.copy(),
                   "readout": context.map_readout(belief.weights), "ess": None,
                   "support_samples": len(belief.theta)}
    joint, reviewed = review_coverage(context.geometry, updated["theta"], updated["log_weights"], evidence)
    return updated, joint, reviewed


def select_action(context, belief, current, target, following, failures, method, motion):
    """A and B share views and recovery. B forecasts midpoint observations."""
    weights = belief.weights
    bank = context.banks[target]
    base, visible, distance = bank.scores(weights)
    for i, xy in enumerate(bank.xy):
        base[i] -= .055 * min(4, failures.get(tuple(xy), 0))
    base -= TRAVEL_COST * np.linalg.norm(bank.xy - current, axis=1)
    selected = None
    # Widen automatically after misses. The same candidates are available to A and B.
    count = 4 if sum(failures.values()) < 2 else len(bank.xy)
    options = np.argsort(-base, kind="stable")[:count]
    learned = method != "B_endpoint"
    for i in options:
        goal = bank.xy[i]
        for skill in SKILLS if learned else ("advance",):
            path = motion.get(tuple(current), tuple(goal), skill, learned)
            if np.any(np.abs(path) > np.array([42., 36.]) + 1.e-9):
                continue
            length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            score = base[i] - TRAVEL_COST * (length - np.linalg.norm(goal - current))
            future_score = 0.0
            if method.startswith("B_") and following is not None:
                future = context.banks[following]
                hit_now = weights * bank.capture[i]
                travel = TRAVEL_COST * np.linalg.norm(future.xy - goal, axis=1)
                if method == "B_open_loop":
                    bin_prob = np.ones((len(weights), 1))
                else:
                    mu, probability, _ = context.geometry.predict(context.theta, path[len(path)//2])
                    bin_prob = predictive_bins(mu, probability)
                joint = hit_now[:, None] * bin_prob
                value = future.capture @ joint - travel[:, None] * joint.sum(axis=0)
                future_score = float(value.max(axis=0).sum())
                score += FUTURE_VALUE_WEIGHT * future_score
            if selected is None or score > selected[0] + 1.e-12:
                selected = (float(score), goal, skill, path, {
                    "capture_probability": float(visible[i]), "dird_margin": float(bank.margin[i]),
                    "expected_distance_mm": float(distance[i]), "lookahead": future_score,
                })
    return selected


def run_episode(context, motion, method, seed, pose_deg, continuous, traces, truth_rows,
                *, permanent_occlusion=False, temporary_dropout=False, continuous_refinement=True):
    episode = f"{pose_deg}_{seed}"
    geometry = context.geometry
    truth = draw_truth(seed, continuous)
    truth_world = geometry.body_world(graph_nodes(truth[None]))[0]
    belief = context.start_belief()
    initial = context.map_readout(belief.weights)
    order = (LOWER_SECTIONS, UPPER_SECTIONS) if seed % 2 == 0 else (UPPER_SECTIONS, LOWER_SECTIONS)
    required = (*TRUNK_SECTIONS, *JUNCTION_SECTIONS, FORK_SECTION, *order[0], *order[1])
    entry = np.array([-36., 0.])
    current = entry.copy()
    last_good = None
    completed = []
    events, evidence, evaluated = [], {}, {}
    misses, retreats, frames, segments = 0, 0, 0, 0
    distance_mm, duration_s = 0., 0.
    failures = {}
    retreated = False
    status = "right_censored"
    duplicate_ok = None
    before_readout = initial
    certification = None
    last_certified_event_count = -1
    previous_target = None
    for step in range(MAX_SEGMENTS):
        if len(completed) == len(required):
            if len(events) != last_certified_event_count:
                certification = certify(context, belief, events, evidence, seed + 50019 + step,
                                        method != "A_independent", continuous_refinement)
                last_certified_event_count = len(events)
            posterior, joint, reviewed = certification
            sufficient_samples = not continuous_refinement or (posterior["ess"] or 0.) >= MIN_ESS
            if joint >= JOINT_IDENTITY_GATE and posterior["readout"][2] <= MAP_RADIUS_MM and sufficient_samples:
                status = "declared_complete"
                break
            if joint < JOINT_IDENTITY_GATE:
                target = min(reviewed, key=lambda s: reviewed[s][0])
            elif continuous_refinement:
                p = np.exp(posterior["log_weights"])
                def remaining_variance(s):
                    xyz = geometry.body_world(target_points(posterior["theta"], s.x, s.label))
                    return float(p @ np.sum((xyz - p @ xyz)**2, axis=1))
                target = max(SECTIONS, key=remaining_variance)
            else:
                target = context.most_uncertain_section(belief.weights)
        else:
            target = required[len(completed)]
        if target != previous_target:
            failures.clear()
            retreated = False
            previous_target = target
        next_i = len(completed) + 1
        following = required[next_i] if next_i < len(required) else None
        fallback = entry if last_good is None else last_good
        retreat = (sum(failures.values()) >= 3 and not retreated and np.linalg.norm(current - fallback) > 1.)
        if retreat:
            goal, skill = fallback, "advance"
            path = motion.get(tuple(current), tuple(goal), skill, method != "B_endpoint")
            if np.any(np.abs(path) > np.array([42., 36.]) + 1.e-9):
                retreat = False
            else:
                choice = {}
                retreated = True
                retreats += 1
        if not retreat:
            chosen = select_action(context, belief, current, target, following, failures, method, motion)
            if chosen is None:
                status = "no_chart_program"
                break
            _, goal, skill, path, choice = chosen
        world_path = geometry.skin_world(path)
        segments += 1
        length = float(np.linalg.norm(np.diff(world_path, axis=0), axis=1).sum())
        distance_mm += length
        segment_s = max(.30, length / 18.)
        duration_s += segment_s
        gps_before = projected_gps(context.banks, belief.weights, (target,))[0]
        # No acquisition on the approach before first skin contact.
        view_samples = (len(path)-1,) if step == 0 else (len(path)//2, len(path)-1)
        frame_rows = []
        accepted = ()
        accepted_id = None
        for j in view_samples:
            xy = path[j]
            rng = np.random.default_rng(np.random.SeedSequence([seed, 351, step, j]))
            obs, truth_labels = render(geometry, truth, xy, rng,
                                       suppressed_label=UPPER if permanent_occlusion else None,
                                       dropout=temporary_dropout and (step < 3 or 13 <= step < 16))
            event_id = f"{step}:{j}"
            try:
                identity = belief.update(event_id, geometry, xy, obs)
            except ValueError as exc:
                status = "model_support_failure"
                frame_rows.append({"id": event_id, "xy_mm": xy.tolist(), "uv_mm": obs.tolist(),
                                   "update_error": str(exc)})
                write_row(truth_rows, {"episode": episode, "method": method, "id": event_id,
                                      "detection_labels": truth_labels.tolist()})
                break
            frames += 1
            public_event = Event(xy.copy(), obs, identity)
            events.append(public_event)
            if duplicate_ok is None:
                saved = belief.log_weights.copy()
                duplicate_ok = (belief.update(event_id, geometry, xy, obs) is None
                                and np.array_equal(saved, belief.log_weights))
            frame_rows.append({"id": event_id, "xy_mm": xy.tolist(), "uv_mm": obs.tolist(),
                               "identity": identity.tolist()})
            write_row(truth_rows, {"episode": episode, "method": method, "id": event_id,
                                  "detection_labels": truth_labels.tolist()})
            if j == len(path)-1 and not retreat:
                accepted = chosen_indices(identity, target.label)
                if accepted:
                    accepted_id = event_id
                    evidence.setdefault(target, []).append((event_id, public_event, accepted))
                    evaluated[event_id] = bool(is_correct(accepted, target.label, truth_labels))
        if accepted and status != "model_support_failure":
            last_good = goal.copy()
            if target.name not in completed:
                completed.append(target.name)
            failures.clear()
            retreated = False
        elif not retreat:
            misses += 1
            failures[tuple(goal)] = failures.get(tuple(goal), 0) + 1
        current = goal.copy()
        before_readout = context.map_readout(belief.weights)
        write_row(traces, {"episode": episode, "method": method, "step": step,
                           "section": target.name, "action": "retreat" if retreat else skill,
                           "learned": method != "B_endpoint", "approach": step == 0,
                           "path_xy_mm": path.tolist(), "length_mm": length, "duration_s": segment_s,
                           "gps": gps_before, "choice": choice, "frames": frame_rows,
                           "accepted_indices": list(accepted), "accepted_event": accepted_id,
                           "provisional_sections": list(completed),
                           "map_radius_mm": before_readout[2]})
        if status == "model_support_failure":
            break
    if certification is None or len(events) != last_certified_event_count:
        certification = certify(context, belief, events, evidence, seed + 80019,
                                method != "A_independent", continuous_refinement)
    posterior, joint, reviewed = certification
    valid = {s.name: {"probability": probability, "event": event_id}
             for s, (probability, event_id) in reviewed.items() if probability >= IDENTITY_GATE}
    correct = [evaluated[row["event"]] for row in valid.values()]
    sufficient_samples = not continuous_refinement or (posterior["ess"] or 0.) >= MIN_ESS
    if (status == "right_censored" and len(valid) == len(required) and joint >= JOINT_IDENTITY_GATE
            and posterior["readout"][2] <= MAP_RADIUS_MM and sufficient_samples):
        status = "declared_complete"
    final = map_metrics(posterior["readout"], truth_world)
    return ({"episode": episode, "method": method, "seed": seed, "pose_deg": pose_deg,
             "regime": ("permanent_occlusion" if permanent_occlusion else "temporary_dropout" if temporary_dropout
                        else "continuous" if continuous else "matched"),
             "status": status, "sections": len(required), "completed_sections": list(valid),
             "correct_sections": int(sum(correct)), "false_sections": int(len(correct) - sum(correct)),
             "success": bool(status == "declared_complete" and len(correct) == len(required)
                             and all(correct) and final["max_error_mm"] <= MAP_RADIUS_MM),
             "joint_identity_probability": joint, "certified_evidence": valid,
             "importance_ess": posterior["ess"], "support_samples": posterior["support_samples"],
             "frames": frames, "segments": segments,
             "failed_attempts": misses, "retreats": retreats, "path_mm": distance_mm,
             "simulated_duration_s": duration_s, "duplicate_invariant": duplicate_ok,
             "initial_map": map_metrics(initial, truth_world), "final_map": final},
            posterior["theta"], posterior["log_weights"], posterior["readout"][0], posterior["readout"][1], truth, truth_world)


def aggregate(rows):
    grouped = {}
    for regime in sorted({r["regime"] for r in rows}):
        for method in sorted({r["method"] for r in rows if r["regime"] == regime}):
            data = [r for r in rows if r["regime"] == regime and r["method"] == method]
            grouped[f"{regime}/{method}"] = {
                "n": len(data), "declared_complete": sum(r["status"] == "declared_complete" for r in data),
                "all_sections_correct": sum(r["correct_sections"] == r["sections"] for r in data),
                "accurate_complete": sum(r["success"] for r in data),
                "false_sections": sum(r["false_sections"] for r in data),
                "coverage": float(np.mean([r["correct_sections"] / r["sections"] for r in data])),
                "initial_rmse_mm": float(np.mean([r["initial_map"]["rmse_mm"] for r in data])),
                "final_rmse_mm": float(np.mean([r["final_map"]["rmse_mm"] for r in data])),
                "initial_joint_radius_mm": float(np.mean([r["initial_map"]["joint_90_radius_mm"] for r in data])),
                "final_joint_radius_mm": float(np.mean([r["final_map"]["joint_90_radius_mm"] for r in data])),
                "empirical_joint_coverage": float(np.mean([r["final_map"]["joint_90_covered"] for r in data])),
                "mean_frames": float(np.mean([r["frames"] for r in data])),
                "mean_path_mm": float(np.mean([r["path_mm"] for r in data])),
                "mean_failed_attempts": float(np.mean([r["failed_attempts"] for r in data])),
                "total_retreats": sum(r["retreats"] for r in data),
                "mean_simulated_duration_s": float(np.mean([r["simulated_duration_s"] for r in data])),
                "model_support_failures": sum(r["status"] == "model_support_failure" for r in data),
                "median_importance_ess": (float(np.median([r["importance_ess"] or 0. for r in data]))
                                          if any(r["importance_ess"] is not None for r in data) else None),
            }
    return grouped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--robot-table", type=Path)
    parser.add_argument("--discrete-only", action="store_true")
    parser.add_argument("--poses", type=int, nargs="+", default=[0, 45, 75])
    parser.add_argument("--n-per-regime", type=int, default=16)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    all_regimes = ["matched", "continuous", "permanent_occlusion", "temporary_dropout"]
    parser.add_argument("--regimes", nargs="+", default=all_regimes, choices=all_regimes)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    charts = {pose: LegChart(pose) for pose in args.poses}
    table_path = args.robot_table or args.out / "ROBOT_VIEWS.json"
    table = json.loads(table_path.read_text()) if args.robot_table else build_robot_table(charts, table_path)
    table_rows = {r["pose_deg"]: r for r in table["poses"]}
    primitive = LearnedProDMP.fit_default(seed=20261112)
    motion = MotionLibrary(primitive)
    methods, rows, maps, samples, positions, variances, true_parameters, truth_worlds = [], [], [], [], [], [], [], []
    protocol = {"poses_deg": args.poses, "per_pose_per_regime": args.n_per_regime,
                "methods": args.methods, "regimes": args.regimes, "episode_seed": EPISODE_SEED,
                "start_index": args.start_index,
                "identity_gate": IDENTITY_GATE, "max_segments_censoring_guard": MAX_SEGMENTS,
                "joint_identity_gate": JOINT_IDENTITY_GATE, "continuous_refinement": not args.discrete_only,
                "importance_min_ess": MIN_ESS,
                "map_radius_gate_mm": MAP_RADIUS_MM, "targets": [s.name for s in SECTIONS],
                "robot_table": str(table_path.resolve()), "robot_table_sha256": sha256(table_path),
                "source_sha256": {name: sha256(HERE / name) for name in (
                    "scan_bifurcation.py", "bifurcation_model.py", "continuous_map.py", "gps_projection.py", "leg_chart.py", "learned_prodmp.py")},
                "atlas": {str(pose): chart.metadata for pose, chart in charts.items()}}
    write_json(args.out / "PROTOCOL.json", protocol)
    start = time.perf_counter()
    with (args.out / "PUBLIC_SCAN.jsonl").open("w") as traces, (args.out / "EVAL_ONLY_LABELS.jsonl").open("w") as eval_rows, (args.out / "EPISODES.jsonl").open("w") as episode_rows:
        for pose in args.poses:
            geometry = SectionGeometry(charts[pose])
            contexts = {True: ScanContext(geometry, table_rows[pose], True)}
            if "A_independent" in args.methods:
                contexts[False] = ScanContext(geometry, table_rows[pose], False)
            write_json(args.out / f"INITIAL_GPS_{pose:02d}.json",
                       projected_gps(contexts[True].banks, np.exp(contexts[True].log_prior)))
            for ri, regime in enumerate(all_regimes):
                if regime not in args.regimes:
                    continue
                count = min(args.n_per_regime, 4) if ri >= 2 else args.n_per_regime
                for i in range(args.start_index, args.start_index + count):
                    seed = int(np.random.SeedSequence([EPISODE_SEED, pose, ri, i]).generate_state(1)[0])
                    for method in args.methods:
                        if ri >= 2 and method not in ("A_joint", "B_joint"):
                            continue
                        context = contexts[method != "A_independent"]
                        row, theta, weights, pos, var, truth, truth_world = run_episode(
                            context, motion, method, seed, pose, ri in (1, 3), traces, eval_rows,
                            permanent_occlusion=ri == 2, temporary_dropout=ri == 3,
                            continuous_refinement=not args.discrete_only,
                        )
                        write_row(episode_rows, row)
                        rows.append(row)
                        methods.append(method)
                        maps.append(weights)
                        samples.append(theta)
                        positions.append(pos)
                        variances.append(var)
                        true_parameters.append(truth)
                        truth_worlds.append(truth_world)
                episode_rows.flush()
                traces.flush()
                eval_rows.flush()
                print(f"pose {pose}, {regime}: {count} paired episodes, {time.perf_counter()-start:.1f}s", flush=True)
    width = max(map(len, maps))
    padded = np.full((len(maps), width), -np.inf)
    padded_samples = np.zeros((len(samples), width, 6))
    for i, log_weight in enumerate(maps):
        padded[i, :len(log_weight)] = log_weight
        padded_samples[i, :len(log_weight)] = samples[i]
    np.savez_compressed(args.out / "FINAL_MAPS.npz", episode=np.array([r["episode"] for r in rows]),
                        method=np.array(methods), log_weights=padded, posterior_coefficients=padded_samples,
                        posterior_count=np.array(list(map(len, maps))), mean_world_mm=positions,
                        variance_world_mm2=variances, evaluation_truth_coefficients=true_parameters,
                        evaluation_truth_world_mm=truth_worlds)
    summary = {"scope": __doc__, "wall_seconds": time.perf_counter()-start,
               "groups": aggregate(rows), "motion_cache": motion.get.cache_info()._asdict()}
    write_json(args.out / "SUMMARY.json", summary)
    print(json.dumps(summary["groups"], indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
