"""Paired frozen versus adaptive material maps with matched recheck budgets."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binomtest


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "ab_knee_latent_rl_20261112"
sys.path[:0] = [str(BASE), str(HERE.parent / "ab_learned_acquisition_20261112")]
from acquisition import turn_angle
from belief import Reading
from experiment import State, current_identity, identity_outcome, prefix_hash, seed_at
from knee_world import CAPTURE_S
from policy import imagined_frames, leaf_value, terminal_score, travel_cost

from map_model import MapFilter, MapWorld
import offline_policies as offline


N_CAPTURES = 3
MAX_TRAVEL_MM = 95.
MAX_ROTATION_DEG = 180.
MAX_TIME_S = 3.30
TURN_SPEED_DEG_S = 120.
FISHER_WEIGHT = .10
TRAIN_START = 6016
TEST_START = 6032
MODES = ("nominal", "stress")
ARMS = ("frozen", "online", "online_on_frozen", "frozen_on_online",
        "online_fisher", "cql_7", "cql_29", "fqi_7")


@dataclass(frozen=True)
class Endpoint:
    site: int
    view: int
    t: float

    @classmethod
    def from_frame(cls, frame):
        return cls(frame.site, frame.view, frame.t)


@dataclass(frozen=True)
class Budget:
    frames: int = N_CAPTURES
    travel_mm: float = MAX_TRAVEL_MM
    turn_deg: float = MAX_ROTATION_DEG
    seconds: float = MAX_TIME_S

    def spend(self, motion):
        return Budget(self.frames - 1, self.travel_mm - motion.action.distance_mm,
                      self.turn_deg - motion.turn_deg, self.seconds - motion.action.duration)

    def fractions(self):
        return (self.travel_mm / MAX_TRAVEL_MM, self.seconds / MAX_TIME_S,
                self.turn_deg / MAX_ROTATION_DEG)


@dataclass(frozen=True)
class Motion:
    action: object
    turn_deg: float


def motions(world, reading, previous, budget):
    # Fixed ordered action slots are shared by planning and both offline learners.
    choices = []
    start = (previous.site, previous.view)
    for action in world.candidates(reading, previous):
        if action is None:
            choices.append(None)
            continue
        turn = turn_angle(world, reading.q, start, (action.site, action.view))
        duration = action.duration + turn / TURN_SPEED_DEG_S
        remaining_capture_time = (budget.frames - 1) * CAPTURE_S
        feasible = (action.distance_mm <= budget.travel_mm + 1.e-8
                    and turn <= budget.turn_deg + 1.e-8
                    and duration + remaining_capture_time <= budget.seconds + 1.e-8)
        choices.append(Motion(replace(action, duration=duration, t=previous.t + duration), turn)
                       if feasible else None)
    if not any(m is not None for m in choices):
        raise RuntimeError("no repeat/acquisition within the common budget")
    return tuple(choices)


def leaf(world, belief, end_t):
    # Time prediction is from the available pose; evaluation uses the common
    # noisy pose packet at end_t, which is not available to either policy here.
    predicted = belief.copy()
    predicted.advance(belief.predicted_reading(end_t))
    return leaf_value(world, predicted)


def plan(world, belief, previous, choices, budget, end_t, *, depth=2, fisher=False):
    scores = np.full(len(choices), -np.inf)
    for slot, motion in enumerate(choices):
        if motion is None:
            continue
        action = motion.action
        replan = depth == 2 and budget.frames > 1
        if replan:
            tail_budget = budget.spend(motion)
            tail = motions(world, belief.predicted_reading(action.t), action, tail_budget)
        outcomes = []
        for reading, obs in imagined_frames(world, belief, action, 4 if depth == 2 else 2,
                                           depth=depth):
            child = belief.copy()
            child.observe(world, reading, action.site, action.view, obs,
                          event=-budget.frames, retain=world.identity_site(action.site))
            if replan:
                next_scores = plan(world, child, action, tail, tail_budget, end_t,
                                   depth=1, fisher=fisher)
                outcomes.append(float(next_scores.max()))
            else:
                outcomes.append(leaf(world, child, end_t))
        scores[slot] = float(np.mean(outcomes)) - travel_cost(action)
        if fisher:
            scores[slot] += FISHER_WEIGHT * offline.fisher_bonus(world, belief, action)
    return scores


def features(world, belief, choices, budget):
    return offline.features(world, belief, tuple(m.action if m else None for m in choices),
                            budget.frames, budget.fractions())


def prepare(world, case):
    state = State(MapFilter(), {}, None, (), 0)
    for frame in world.initial(case):
        state = state.updated(world, frame, retain=world.identity_site(frame.site))
    if len(state.frames) != 28:
        raise RuntimeError("expected the full shared 28-frame first sweep")
    return state


def observe(world, case, motion):
    frame = world.observe(case, motion.action)
    if abs(frame.t - motion.action.t) > 1.e-10:
        raise RuntimeError("capture did not use its scheduled physical time")
    return frame


def read_pose(case, t):
    q, velocity = case.true_angle_rate(t)
    noise = np.random.default_rng(np.random.SeedSequence([case.seed, 147, round(t * 1000)]))
    return Reading(t, q + case.pose_bias + noise.normal(0., .10), velocity + noise.normal(0., .15))


def map_errors(world, case, belief, t):
    mean, _ = belief.forecast(world, t, world.query_sites)
    truth = world.truth_world(case, t, world.query_sites)
    error = np.linalg.norm(mean - truth, axis=-1)
    return float(np.sqrt(np.mean(error**2))), float(error.max())


def at_time(state, reading):
    belief = state.belief.copy()
    belief.advance(reading)
    return replace(state, belief=belief)


def eval_step(world, case, before, frame, after, motion):
    shadow = at_time(before, Reading(frame.t, frame.q, frame.velocity))
    _, _, _, prior_ok, prior_wrong = current_identity(shadow)
    _, _, _, now_ok, now_wrong = current_identity(after)
    prior_count, now_count = sum(prior_ok), sum(now_ok)
    identity_gain = now_count > prior_count or (now_count == prior_count
                                                and sum(now_wrong) < sum(prior_wrong))
    prior_rmse, _ = map_errors(world, case, shadow.belief, frame.t)
    now_rmse, _ = map_errors(world, case, after.belief, frame.t)
    return {
        "site": frame.site, "view": frame.view, "kind": motion.action.kind, "t": frame.t,
        "q": frame.q, "centroids": frame.centroids.tolist(), "labels_for_evaluation": list(frame.labels),
        "probe_xyz_mm": world.probe_world(frame.q, frame.site, frame.view).tolist(),
        "travel_mm": motion.action.distance_mm, "turn_deg": motion.turn_deg,
        "duration_s": motion.action.duration, "empty": len(frame.centroids) == 0,
        "identity_improved": identity_gain, "identity_no_improvement": not identity_gain,
        "immediate_non_improving": not identity_gain and prior_rmse - now_rmse < .10,
        "shadow_rmse_mm": prior_rmse, "updated_rmse_mm": now_rmse,
    }


def final_outcome(world, case, prefix, state, end_t):
    terminal = at_time(state, read_pose(case, end_t))
    ids = identity_outcome(prefix, terminal)
    current, maximum = map_errors(world, case, terminal.belief, end_t)
    future, _ = map_errors(world, case, terminal.belief, end_t + .5)
    return {
        **ids, "strict": bool(ids["both_correct"] and maximum <= 3.),
        "unresolved": int(ids["confirmed_labels"]) < 2,
        "rmse_mm": current, "max_mm": maximum, "future_rmse_mm": future,
    }


def rollout(world, case, prefix, name, *, replay=None):
    frozen = name.startswith("frozen")
    state = replace(prefix, belief=prefix.belief.copy())
    if frozen:
        state.belief.freeze_map()
    budget = Budget()
    end_t = prefix.previous.t + budget.seconds
    steps, executions = [], []
    initial_marginal = prefix.belief.weights.sum(axis=1)
    for stage in range(N_CAPTURES):
        previous = Endpoint.from_frame(state.previous)
        choices = motions(world, state.belief.reading, previous, budget)
        if replay is None:
            if name in MODELS:
                x, mask = features(world, state.belief, choices, budget)
                scores = offline.values(MODELS[name], x, mask)
            else:
                scores = plan(world, state.belief, previous, choices, budget, end_t,
                              depth=2 if budget.frames > 1 else 1, fisher=name == "online_fisher")
            slot = int(np.argmax(scores))
            motion = choices[slot]
            if motion is None:
                raise RuntimeError(f"{name} chose a masked action")
            frame = observe(world, case, motion)
        else:
            slot, motion, frame = replay[stage]
            if choices[slot] != motion:
                raise RuntimeError("cross replay did not preserve the same executable command")
        next_state = state.updated(world, frame, retain=world.identity_site(frame.site))
        if frozen:
            belief = next_state.belief
            if abs(float(belief.weights.sum()) - 1.) > 1.e-10 or any(
                np.max(np.abs(w.joint.sum(axis=0) - belief.weights)) > 1.e-10
                for w in belief.witnesses
            ):
                raise RuntimeError("frozen posterior and identity marginals disagree")
        steps.append(eval_step(world, case, state, frame, next_state, motion))
        executions.append((slot, motion, frame))
        state = next_state
        budget = budget.spend(motion)
    if min(budget.fractions()) < -1.e-8 or budget.frames != 0 or state.previous.t > end_t + 1.e-8:
        raise RuntimeError("ran outside a shared physical budget")
    map_change = float(np.abs(initial_marginal - state.belief.weights.sum(axis=1)).sum())
    if frozen and map_change > 1.e-10:
        raise RuntimeError("frozen map marginal changed")
    result = final_outcome(world, case, prefix, state, end_t)
    result.update({
        "captures": len(steps), "prefix_sha": prefix_hash(prefix),
        "common_end_t": end_t, "last_capture_t": state.previous.t,
        "elapsed_s": state.previous.t - prefix.previous.t,
        "travel_mm": MAX_TRAVEL_MM - budget.travel_mm, "turn_deg": MAX_ROTATION_DEG - budget.turn_deg,
        "non_improving": sum(s["immediate_non_improving"] for s in steps),
        "identity_no_improvement": sum(s["identity_no_improvement"] for s in steps),
        "empty_rechecks": sum(s["empty"] for s in steps),
        "map_marginal_change_l1": map_change,
        "steps": steps, "actions": [slot for slot, _, _ in executions],
    })
    return result, executions


def train_case(item):
    mode, index = item
    world = WORLDS[mode]
    seed = seed_at("train", TRAIN_START + index)
    case = world.case(seed)
    prefix = prepare(world, case)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1250]))
    prefix_choices = motions(world, prefix.belief.reading, Endpoint.from_frame(prefix.previous), Budget())
    initial = [slot for slot, value in enumerate(prefix_choices) if value is not None]
    rows = []
    for first_slot in initial:
        state, budget = prefix, Budget()
        end_t = prefix.previous.t + budget.seconds
        for stage in range(N_CAPTURES):
            choices = motions(world, state.belief.reading, Endpoint.from_frame(state.previous), budget)
            x, mask = features(world, state.belief, choices, budget)
            slot = first_slot if stage == 0 else int(rng.choice(np.flatnonzero(mask)))
            motion = choices[slot]
            frame = observe(world, case, motion)
            next_state = state.updated(world, frame, retain=world.identity_site(frame.site))
            next_budget = budget.spend(motion)
            done = next_budget.frames == 0
            reward = -travel_cost(motion.action)
            if done:
                row = final_outcome(world, case, prefix, next_state, end_t)
                reward += terminal_score(row["both_correct"], row["strict"], sum(row["correct"]),
                                         sum(row["wrong"]), row["future_rmse_mm"])
                next_x, next_mask = np.zeros_like(x), np.zeros_like(mask)
            else:
                tail = motions(world, next_state.belief.reading,
                               Endpoint.from_frame(next_state.previous), next_budget)
                next_x, next_mask = features(world, next_state.belief, tail, next_budget)
            rows.append((x, mask, slot, reward, next_x, next_mask, done))
            state, budget = next_state, next_budget
    return seed, rows


def test_case(item):
    mode, index = item
    world = WORLDS[mode]
    seed = seed_at("test", TEST_START + index)
    case = world.case(seed)
    prefix = prepare(world, case)
    arms, paths = {}, {}
    for name in ARMS:
        replay = paths[name.split("_on_")[-1]] if "_on_" in name else None
        arms[name], paths[name] = rollout(world, case, prefix, name, replay=replay)
    assert len({(v["prefix_sha"], v["captures"], v["common_end_t"]) for v in arms.values()}) == 1
    for a, b in (("frozen", "online_on_frozen"), ("online", "frozen_on_online")):
        assert all((r["t"], r["site"], r["view"], r["centroids"]) ==
                   (s["t"], s["site"], s["view"], s["centroids"])
                   for r, s in zip(arms[a]["steps"], arms[b]["steps"]))
    return {
        "mode": mode, "index": index, "seed": seed,
        "prefix_t": prefix.previous.t, "prefix_sites": [f.site for f in prefix.frames],
        "prefix_poses_xyz_mm": [world.probe_world(f.q, f.site, f.view).tolist() for f in prefix.frames],
        "arms": arms,
    }


def summarize(rows):
    rng = np.random.default_rng(2026111234)
    summary = {}
    count_keys = ("strict", "both_correct", "wrong_accept", "wrong_branch", "wrong_clutter",
                  "unresolved", "prefix_confident_wrong_case", "prefix_confident_wrong_replaced")
    mean_keys = ("non_improving", "identity_no_improvement", "empty_rechecks", "elapsed_s",
                 "travel_mm", "turn_deg", "map_marginal_change_l1")
    contrasts = (("online", "frozen"), ("online_on_frozen", "frozen"),
                 ("online", "online_on_frozen"), ("frozen_on_online", "frozen"),
                 ("online_fisher", "online"), ("cql_7", "online"), ("cql_29", "online"),
                 ("fqi_7", "online"))
    for mode in MODES:
        batch = [r["arms"] for r in rows if r["mode"] == mode]
        resample = rng.integers(0, len(batch), size=(4000, len(batch)))
        data, pairs = {}, {}
        for arm in ARMS:
            values = [row[arm] for row in batch]
            row = {key: int(sum(v[key] for v in values)) for key in count_keys}
            row.update({key: float(np.mean([v[key] for v in values])) for key in mean_keys})
            row["pooled_rmse_mm"] = float(np.sqrt(np.mean([v["rmse_mm"]**2 for v in values])))
            row["changed_commands_vs_frozen"] = sum(
                tuple((s["site"], s["view"], s["t"]) for s in r[arm]["steps"]) !=
                tuple((s["site"], s["view"], s["t"]) for s in r["frozen"]["steps"]) for r in batch)
            data[arm] = row
        for arm, base in contrasts:
            stats = {}
            for key in (*count_keys[:6], *mean_keys):
                delta = np.array([float(r[arm][key]) - float(r[base][key]) for r in batch])
                wins, losses = int((delta > 0).sum()), int((delta < 0).sum())
                stats[key] = {"difference": float(delta.mean()),
                              "ci95": np.quantile(delta[resample].mean(axis=1), [.025, .975]).tolist(),
                              "increased": wins, "decreased": losses}
                if key in count_keys[:6]:
                    stats[key]["paired_p"] = float(binomtest(wins, wins + losses).pvalue) if wins + losses else 1.
            pairs[f"{arm}_minus_{base}"] = stats
        summary[mode] = {"cases": len(batch), "arms": data, "paired": pairs}
    # Secondary solver comparisons across both conditions and training seeds.
    secondary = [summary[mode]["paired"][f"{name}_minus_online"]["strict"]
                 for mode in MODES for name in ("online_fisher", "cql_7", "cql_29", "fqi_7")]
    adjusted = 0.
    for i, item in enumerate(sorted(secondary, key=lambda x: x["paired_p"])):
        adjusted = max(adjusted, (len(secondary) - i) * item["paired_p"])
        item["holm_secondary_p"] = min(adjusted, 1.)
    return summary


def plot(output, rows, summary):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    stress = [row for row in rows if row["mode"] == "stress"]
    example = next((r for r in stress if r["arms"]["online"]["strict"] != r["arms"]["frozen"]["strict"]),
                   stress[0])
    world = WORLDS["stress"]
    case = world.case(example["seed"])
    final_t = example["arms"]["online"]["common_end_t"]
    trunk = world.truth_world(case, final_t, [(site, 0) for site in range(world.n)])
    branch = world.truth_world(case, final_t, [(site, 1) for site in sorted(world.branch_sites)])
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}]],
                        subplot_titles=("Acquisition poses in one paired case", "Strict completions"))
    for name, coords in (("Simulated main vessel", trunk), ("Simulated branch", branch)):
        fig.add_trace(go.Scatter3d(x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                                  mode="lines", name=name, line=dict(width=7, color="#a3adb8")), row=1, col=1)
    xyz = np.asarray(example["prefix_poses_xyz_mm"])
    fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
                              name="Shared first sweep", marker=dict(size=2.5, color="#8493a2")), row=1, col=1)
    names = {
        "frozen": ("Frozen map", "#a46b3c"), "online": ("Online map", "#21788b"),
        "online_fisher": ("Online + Fisher", "#6b6098"), "cql_7": ("CQL", "#9c4660"),
        "fqi_7": ("FQI", "#557944"),
    }
    for arm, (label, color) in names.items():
        item = example["arms"][arm]
        points = np.asarray([step["probe_xyz_mm"] for step in item["steps"]])
        fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
                                  name=label, legendgroup=arm, marker=dict(size=5, color=color),
                                  text=[f"Capture {j+1} · station {s['site']} · view {s['view']}"
                                        for j, s in enumerate(item["steps"])],
                                  hovertemplate="%{text}<extra>%{fullData.name}</extra>"), row=1, col=1)
        fig.add_trace(go.Bar(x=["Nominal", "Confusion + missed frames"],
                             y=[summary[m]["arms"][arm]["strict"] for m in MODES],
                             name=label, legendgroup=arm, showlegend=False, marker_color=color), row=1, col=2)
    fig.update_layout(title="Material map update during rechecks", template="plotly_white",
                      height=620, margin=dict(l=30, r=30, t=90, b=100),
                      legend=dict(orientation="h", y=-.15))
    fig.update_scenes(xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)", aspectmode="data")
    fig.update_yaxes(title_text=f"Cases / {summary['stress']['cases']}", row=1, col=2)
    fig.write_html(output / "MAP_RECHECK_3D.html", include_plotlyjs=True)
    (output / "PLOT_CASE.json").write_text(json.dumps({
        "mode": example["mode"], "index": example["index"], "seed": example["seed"],
        "selection": "first strict-outcome discordance in stress, otherwise first stress case",
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--train", type=int, default=256, help="combined nominal and stress training cases")
    parser.add_argument("--test", type=int, default=128, help="paired cases per test condition")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    global WORLDS, MODELS
    WORLDS = {name: MapWorld(mode=name) for name in MODES}
    ctx = get_context("fork")
    train_tasks = [(MODES[i % len(MODES)], i) for i in range(args.train)]
    records, train_seeds = [], []
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as pool:
        for seed, items in pool.map(train_case, train_tasks, chunksize=4):
            train_seeds.append(seed)
            records.extend(items)
    keys = ("features", "mask", "action", "reward", "next_features", "next_mask", "done")
    dataset = {key: np.asarray(vals) for key, vals in zip(keys, zip(*records))}
    np.savez_compressed(args.out / "TRAIN_TRANSITIONS.npz", **dataset)
    print(f"Collected {len(records)} transitions from {len(train_seeds)} cases", flush=True)
    MODELS = offline.fit_models(dataset, args.out, device="cpu", steps=args.steps)
    assert set(MODELS) == {"cql_7", "cql_29", "fqi_7"}
    test_tasks = [(name, i) for i in range(args.test) for name in MODES]
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as pool:
        with (args.out / "TEST.jsonl").open("w") as stream:
            for i, row in enumerate(pool.map(test_case, test_tasks, chunksize=1), 1):
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
                if i % 16 == 0:
                    print(f"Evaluated {i}/{len(test_tasks)}", flush=True)
    assert not set(train_seeds) & {row["seed"] for row in rows}
    summary = summarize(rows)
    (args.out / "SUMMARY.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (args.out / "MANIFEST.json").write_text(json.dumps({
        "budget": asdict(Budget()), "turn_speed_deg_s": TURN_SPEED_DEG_S, "fisher_weight": FISHER_WEIGHT,
        "trains": train_seeds, "transitions": len(records), "test_start": TEST_START,
        "test_cases_per_mode": args.test, "fit_steps": args.steps,
        "geometry_sha256": hashlib.sha256(WORLDS["nominal"].geometry_path.read_bytes()).hexdigest(),
        "frozen_sha256": {f: hashlib.sha256((BASE / f).read_bytes()).hexdigest() for f in
                          ("knee_world.py", "belief.py", "policy.py", "experiment.py", "learners.py")},
        "sha256": {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in
                   ("closed_loop.py", "map_model.py", "offline_policies.py", "PROTOCOL.txt")},
    }, indent=2) + "\n")
    plot(args.out, rows, summary)


if __name__ == "__main__":
    main()
