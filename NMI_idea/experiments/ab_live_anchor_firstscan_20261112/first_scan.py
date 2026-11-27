"""Small first-scan Y-window selection trial on the frozen knee simulator.

This experiment changes only the view/window at stations 4 -> 3 -> 2.  The
28-station order, one capture per station, all non-fork views, edge validity,
and the shared estimator remain fixed.  It is not a full scan reorder.

The frozen world/filter/policy/learner modules are imported read-only from the
previous knee experiment.  A new CQL model is trained from scratch on this
first-scan action space; no recheck checkpoint is loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "ab_knee_latent_rl_20261112"
sys.path.insert(0, str(PREVIOUS))

from belief import Filter, Reading  # noqa: E402
from experiment import (  # noqa: E402
    BUDGET, BASES, QUERY_DELAY, State, apply, current_identity,
    evaluate_outcome, forecast_outcome, identity_outcome, prefix_hash, seed_at,
)
from knee_world import CAPTURE_S, TRAVEL_SPEED_MM_S, Candidate, GEOMETRY, World  # noqa: E402
from policy import (  # noqa: E402
    RULE_DEPTH, fast_behavior_scores, leaf_value, proposal_features, rule_scores,
    terminal_score,
)
import learners  # noqa: E402


FORK_SITES = (4, 3, 2)
DEFAULT_TRAIN_COUNT = 128
DEFAULT_TEST_START = 512
DEFAULT_TEST_COUNT = 128
TRAIN_SEEDS = (7, 29)
WINDOW_VIEWS = (3, 7, 8, 2, 4, 1, 5, 0, 6)
PRIMARY_MODEL = "cql_firstscan_7.pt"
BOOTSTRAPS = 4000
SALT_TRAIN = 10000
SALT_TEST = 20000


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, tuple):
        return [safe(x) for x in value]
    if isinstance(value, list):
        return [safe(x) for x in value]
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(safe(value), indent=2, allow_nan=False) + "\n")


def new_seed(split: str, index: int) -> int:
    # Offsets leave all prior train/test indices untouched.
    return seed_at(split, int(index) + (1024 if split == "train" else 512))


def state_from_prefix(world: World, case: Any) -> State:
    """Execute the frozen default route only through site 5."""
    state = State(Filter(latent=True), {}, None, (), 0)
    for site, view in world.initial_route:
        if site < 5:
            break
        if state.previous is None:
            action = Candidate(site, view, CAPTURE_S, 0., CAPTURE_S, "initial fixed")
        else:
            action = world.make_action(state.belief.reading.q, state.previous,
                                       (site, view), "initial fixed")
        frame = world.observe(case, action)
        reading = Reading(frame.t, frame.q, frame.velocity)
        belief = state.belief.copy()
        belief.observe(world, reading, frame.site, frame.view, frame.centroids,
                       event=state.event, retain=world.identity_site(frame.site))
        labels = state.labels.copy()
        if world.identity_site(frame.site):
            for j, label in enumerate(frame.labels):
                labels[(state.event, j)] = int(label)
        state = State(belief, labels, frame, (*state.frames, frame), state.event + 1)
    if state.previous is None or state.previous.site != 5:
        raise RuntimeError("prefix did not stop at fixed site 5")
    return state


def fake_previous(site: int, view: int, t: float) -> Candidate:
    return Candidate(int(site), int(view), 0., 0., float(t), "schedule")


def base_action(world: World, q: float, previous: Any, site: int, view: int) -> Candidate:
    return world.make_action(float(q), previous, (int(site), int(view)), "first-scan window")


def pad_action(action: Candidate, duration: float) -> Candidate:
    duration = float(duration)
    return Candidate(action.site, action.view, duration, action.distance_mm,
                     action.t - action.duration + duration, action.kind)


def candidate_views(world: World, site: int, previous: Any) -> tuple[int, ...]:
    next_goal = (site - 1, 3)
    result = []
    for view in WINDOW_VIEWS:
        if view not in world.view_bank[site]:
            continue
        if world.edge_valid((previous.site, previous.view), (site, view)) \
                and world.edge_valid((site, view), next_goal):
            result.append(int(view))
    if 3 not in result:
        raise RuntimeError(f"original view is not legal at fork site {site}")
    return tuple(result)


def max_duration(world: World, q: float, previous_options: list[tuple[int, int, float]],
                 site: int, views: tuple[int, ...]) -> float:
    durations = []
    for previous_site, previous_view, previous_t in previous_options:
        previous = fake_previous(previous_site, previous_view, previous_t)
        for view in views:
            action = base_action(world, q, previous, site, view)
            durations.append(float(action.duration))
    if not durations:
        raise RuntimeError(f"empty duration envelope at site {site}")
    return float(max(durations))


def envelope_duration(world: World, previous_options: list[tuple[int, int, float]],
                      site: int, views: tuple[int, ...]) -> float:
    """Upper envelope over all frozen pose knots, with no future observations."""
    return float(max(
        max_duration(world, float(q), previous_options, site, views)
        for q in np.asarray(world.knot_angles, dtype=float)
    ))


def build_schedule(world: World, case: Any, prefix: State) -> dict[str, Any]:
    """Preplan padded durations from public pose knots only.

    No future frame is observed or filtered while constructing this schedule;
    all durations are a fixed upper envelope over baked pose knots and legal
    view pairs.  This avoids leaking future q/centroid information.
    """
    durations: dict[int, float] = {}
    # All legal fork windows from the default route are included.  The frozen
    # edge validator is checked again when actions are materialized.
    views_by_site = {site: tuple(v for v in WINDOW_VIEWS if v in world.view_bank[site])
                     for site in FORK_SITES}
    t0 = float(prefix.previous.t)
    durations[4] = envelope_duration(world, [(5, 3, t0)], 4, views_by_site[4])
    durations[3] = envelope_duration(
        world, [(4, view, t0 + durations[4]) for view in views_by_site[4]],
        3, views_by_site[3],
    )
    durations[2] = envelope_duration(
        world, [(3, view, t0 + durations[4] + durations[3]) for view in views_by_site[3]],
        2, views_by_site[2],
    )
    durations[1] = envelope_duration(
        world, [(2, view, t0 + durations[4] + durations[3] + durations[2])
                for view in views_by_site[2]],
        1, (3,),
    )
    durations[0] = envelope_duration(
        world, [(1, 3, t0 + durations[4] + durations[3] + durations[2] + durations[1])],
        0, (3,),
    )
    return {
        "durations_s": {str(site): float(value) for site, value in durations.items()},
        "fork_sites": list(FORK_SITES),
        "view_pool": {str(site): list(views_by_site[site]) for site in FORK_SITES},
        "pose_knots_used": np.asarray(world.knot_angles, dtype=float).tolist(),
        "schedule_uses_future_observations": False,
    }


def scheduled_action(world: World, state: State, site: int, view: int,
                     schedule: Mapping[str, Any]) -> Candidate:
    duration = float(schedule["durations_s"][str(site)])
    action = base_action(world, state.belief.reading.q, state.previous, site, view)
    if duration + 1.e-9 < action.duration:
        raise RuntimeError(f"schedule under-padded site {site} view {view}")
    return pad_action(action, duration)


def fork_actions(world: World, state: State, site: int,
                 schedule: Mapping[str, Any]) -> tuple[Any, ...]:
    views = set(schedule["view_pool"][str(site)])
    actions = []
    for view in WINDOW_VIEWS:
        legal = (view in views
                 and world.edge_valid((state.previous.site, state.previous.view), (site, view))
                 and world.edge_valid((site, view), (site - 1, 3)))
        if not legal:
            actions.append(None)
        else:
            actions.append(scheduled_action(world, state, site, view, schedule))
    return tuple(actions)


def first_features(world: World, state: State, site: int,
                   schedule: Mapping[str, Any], stage: int):
    """Compact public Filter/candidate features for this first-scan task."""
    actions = fork_actions(world, state, site, schedule)
    mean, sd = state.belief.stats()
    _, confidence = state.belief.decisions()
    state_vec = np.r_[stage / 3., (state.belief.reading.q - 15.) / 4.,
                       state.belief.reading.velocity / 8., mean / 5., sd / 5.,
                       confidence, len(state.belief.witnesses) / 30.]
    details = np.zeros((len(actions), 11), dtype=np.float32)
    mask = np.asarray([action is not None for action in actions], dtype=bool)
    for slot in np.flatnonzero(mask):
        action = actions[slot]
        offset = float(world.scan.offsets[action.view])
        details[slot] = np.array([
            site / max(1, world.n - 1), action.view / max(1, world.n_views - 1),
            action.duration / 2., action.distance_mm / 100., abs(offset) / 24.,
            float(np.mean(confidence)), 1. / (1. + float(np.mean(sd))),
            1. / (1. + abs(offset)), world.clutter_probability(site, action.view),
            float(world.identity_site(site)), float(slot == WINDOW_VIEWS.index(3)),
        ], dtype=np.float32)
    features = np.zeros((len(actions), len(state_vec) + 11 + len(actions)), dtype=np.float32)
    for slot in np.flatnonzero(mask):
        features[slot] = np.r_[state_vec, details[slot], np.eye(len(actions))[slot]]
    return actions, features, mask, details


def apply_frame(world: World, case: Any, state: State, action: Candidate,
                frame_cache: dict[tuple[int, int, int], Any] | None = None) -> State:
    key = (round(action.t * 1000), int(action.site), int(action.view))
    if frame_cache is not None and key in frame_cache:
        frame = frame_cache[key]
    else:
        frame = world.observe(case, action)
        if frame_cache is not None:
            frame_cache[key] = frame
    if state.previous is not None and frame.t <= state.previous.t:
        raise RuntimeError("non-increasing first-scan timestamp")
    belief = state.belief.copy()
    reading = Reading(frame.t, frame.q, frame.velocity)
    belief.observe(world, reading, frame.site, frame.view, frame.centroids,
                   event=state.event, retain=world.identity_site(frame.site))
    labels = state.labels.copy()
    if world.identity_site(frame.site):
        for j, label in enumerate(frame.labels):
            labels[(state.event, j)] = int(label)
    return State(belief, labels, frame, (*state.frames, frame), state.event + 1)


def complete_scan(world: World, case: Any, prefix: State,
                  schedule: Mapping[str, Any], choices: Mapping[int, int]) -> tuple[State, list[Candidate]]:
    state = prefix
    actions = []
    for site in FORK_SITES:
        view = int(choices[site])
        available = fork_actions(world, state, site, schedule)
        slot = WINDOW_VIEWS.index(view)
        action = available[slot]
        if action is None:
            raise RuntimeError(f"choice {site}/{view} is illegal")
        state = apply_frame(world, case, state, action)
        actions.append(action)
    for site in (1, 0):
        action = scheduled_action(world, state, site, 3, schedule)
        if not world.edge_valid((state.previous.site, state.previous.view), (site, 3)):
            raise RuntimeError(f"fixed downstream edge invalid at site {site}")
        state = apply_frame(world, case, state, action)
        actions.append(action)
    return state, actions


def terminal_reward(world: World, case: Any, state: State) -> float:
    _, _, _, correct, wrong = current_identity(state)
    current = forecast_outcome(world, case, state, state.previous.t)
    # The +0.5 s forecast is reported as an auxiliary outcome, not used to
    # redefine the terminal strict event or leak future value into training.
    return float(terminal_score(all(correct), all(correct) and current["max_mm"] <= 3.,
                                sum(correct), sum(wrong), current["rmse_mm"]))


def fork_features(world: World, state: State, site: int,
                  schedule: Mapping[str, Any], stage: int):
    actions, features, mask, _ = first_features(world, state, site, schedule, stage)
    return actions, features, mask


def belief_rule_slot(world: World, state: State, site: int,
                     schedule: Mapping[str, Any], stage: int) -> int:
    # Frozen shared-estimator candidate-information rule.  Unlike compact CQL
    # features, this evaluates candidate-specific predicted detections and
    # separation under the current Filter posterior, with no learned weights.
    actions = fork_actions(world, state, site, schedule)
    features, mask = proposal_features(
        world, state.belief, actions, len(FORK_SITES) - stage, len(FORK_SITES),
    )
    values = fast_behavior_scores(features, mask)
    values = np.where(mask, values, -np.inf)
    return int(np.argmax(values))


def fixed_choices() -> dict[int, int]:
    return {site: 3 for site in FORK_SITES}


def run_policy(world: World, case: Any, prefix: State, schedule: Mapping[str, Any],
               *, model: Any = None, policy: str = "fixed",
               frame_cache: dict[tuple[int, int, int], Any] | None = None) -> dict[str, Any]:
    state = prefix
    actions = []
    selected_views = {}
    legal_view_counts = {}
    legal_view_options = {}
    for stage, site in enumerate(FORK_SITES):
        candidates, features, mask = fork_features(world, state, site, schedule, stage)
        legal_view_counts[site] = int(mask.sum())
        legal_view_options[site] = [int(action.view) for action, valid in zip(candidates, mask)
                                    if valid and action is not None]
        if policy == "fixed":
            slot = WINDOW_VIEWS.index(3)
        elif policy == "belief_rule":
            slot = belief_rule_slot(world, state, site, schedule, stage)
        elif policy == "cql":
            if model is None:
                raise ValueError("CQL model required")
            values = learners.scores(model, features[None], mask[None])[0]
            slot = int(np.argmax(values))
        else:
            raise ValueError(policy)
        if not mask[slot] or candidates[slot] is None:
            raise RuntimeError(f"{policy} selected illegal slot {slot} at site {site}")
        action = candidates[slot]
        state = apply_frame(world, case, state, action, frame_cache)
        actions.append(action)
        selected_views[site] = int(action.view)
    for site in (1, 0):
        action = scheduled_action(world, state, site, 3, schedule)
        state = apply_frame(world, case, state, action, frame_cache)
        actions.append(action)
    return outcome(world, case, prefix, state, actions, selected_views,
                   legal_view_counts, legal_view_options, schedule, policy)


def outcome(world: World, case: Any, prefix: State, state: State,
            actions: list[Candidate], selected_views: Mapping[int, int],
            legal_view_counts: Mapping[int, int], legal_view_options: Mapping[int, list[int]],
            schedule: Mapping[str, Any], policy: str) -> dict[str, Any]:
    identity = identity_outcome(prefix, state)
    now = float(state.previous.t)
    future = [forecast_outcome(world, case, state, now + h) for h in (.5, 1.)]
    anchor = float(prefix.previous.t + QUERY_DELAY)
    if anchor <= now:
        anchor = now + QUERY_DELAY
    fixed = forecast_outcome(world, case, state, anchor)
    fork_times = [float(frame.t) for frame in state.frames[-5:-2]]
    # The last five frames are fork 4,3,2 and fixed 1,0.
    if len(fork_times) != 3:
        raise RuntimeError("first-scan trace does not contain exactly three fork frames")
    prefix_travel = 0.0
    for previous, frame in zip(prefix.frames[:-1], prefix.frames[1:]):
        prefix_travel += float(world.make_action(
            previous.q, previous, (frame.site, frame.view), "prefix").distance_mm)
    supplementary_travel = float(sum(action.distance_mm for action in actions))
    trace = [{
        "t": float(frame.t), "site": int(frame.site), "view": int(frame.view),
        "q": float(frame.q), "velocity": float(frame.velocity),
        "probe_xyz_mm": world.probe_world(frame.q, frame.site, frame.view).tolist(),
        "centroids_mm": np.asarray(frame.centroids).tolist(),
    } for frame in state.frames]
    row = {
        **identity,
        "strict": bool(identity["both_correct"] and now is not None
                        and forecast_outcome(world, case, state, now)["max_mm"] <= 3.),
        "current": forecast_outcome(world, case, state, now),
        "future": future, "future_half": future[0], "future_one": future[1],
        "common_clock": fixed,
        "material_within_3mm": bool(forecast_outcome(world, case, state, now)["max_mm"] <= 3.),
        "elapsed_s": float(now - prefix.previous.t),
        "total_elapsed_s": now,
        "travel_mm": supplementary_travel,
        "full_scan_travel_mm": float(prefix_travel + supplementary_travel),
        "prefix_travel_mm": float(prefix_travel),
        "fork_travel_mm": float(sum(action.distance_mm for action in actions[:3])),
        "captures": int(len(actions)), "policy": policy,
        "selected_views": {str(site): int(view) for site, view in selected_views.items()},
        "legal_view_counts": {str(site): int(value) for site, value in legal_view_counts.items()},
        "legal_view_options": {str(site): value for site, value in legal_view_options.items()},
        "fork_times_s": fork_times,
        "common_clock_t": float(fixed["t"]),
        "schedule_durations_s": dict(schedule["durations_s"]),
        "fork_timestamps_equal": True,
        "trace": trace,
        "prefix_sha256": prefix_hash(prefix),
    }
    return row


def zero_next(features: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros_like(features), np.zeros_like(mask)


def collect_training(world: World, *, count: int, out: Path) -> dict[str, np.ndarray]:
    rows = []
    reports = []
    for index in range(count):
        seed = new_seed("train", index)
        case = world.case(seed)
        prefix = state_from_prefix(world, case)
        schedule = build_schedule(world, case, prefix)
        case_reports = []
        for episode, epsilon in enumerate((0.65, 0.30)):
            state = prefix
            trajectory = []
            rng = np.random.default_rng(np.random.SeedSequence([seed, SALT_TRAIN, episode]))
            for stage, site in enumerate(FORK_SITES):
                actions, features, mask = fork_features(world, state, site, schedule, stage)
                legal = np.flatnonzero(mask)
                if not len(legal):
                    raise RuntimeError("empty training candidate mask")
                preferred = belief_rule_slot(world, state, site, schedule, stage)
                if rng.random() < epsilon:
                    slot = int(rng.choice(legal))
                else:
                    slot = int(preferred)
                action = actions[slot]
                state = apply_frame(world, case, state, action)
                trajectory.append((features, mask, slot, action))
            # Fixed downstream route is part of the first-scan terminal state.
            for site in (1, 0):
                state = apply_frame(world, case, state,
                                    scheduled_action(world, state, site, 3, schedule))
            terminal = terminal_reward(world, case, state)
            for stage, (features, mask, slot, action) in enumerate(trajectory):
                if stage == len(trajectory) - 1:
                    next_features, next_mask = zero_next(features, mask)
                else:
                    # Reconstruct the actual intermediate state rather than use
                    # the terminal state's belief for a valid offline transition.
                    replay = prefix
                    for prior_stage, (_, _, prior_slot, prior_action) in enumerate(trajectory[:stage + 1]):
                        replay = apply_frame(world, case, replay, prior_action)
                    _, next_features, next_mask = fork_features(
                        world, replay, FORK_SITES[stage + 1], schedule, stage + 1,
                    )
                # Shared task cost: the frozen rule's public time/travel cost,
                # retained here only as an action cost (not a new selection).
                reward = -.055 * float(action.duration) - .00045 * float(action.distance_mm)
                if stage == len(trajectory) - 1:
                    reward += terminal
                rows.append({
                    "features": features, "mask": mask, "action": slot,
                    "reward": reward, "next_features": next_features,
                    "next_mask": next_mask, "done": stage == len(trajectory) - 1,
                })
            case_reports.append({"episode": episode, "epsilon": epsilon,
                                 "views": [int(item[3].view) for item in trajectory],
                                 "terminal_reward": float(terminal)})
        reports.append({"index": index, "seed": seed, "prefix_sha256": prefix_hash(prefix),
                        "episodes": case_reports})
        if (index + 1) % 16 == 0:
            print(f"train cases {index + 1}/{count}", flush=True)
    fields = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    np.savez_compressed(out / "TRAIN_TRANSITIONS.npz", **fields)
    write_json(out / "TRAIN_CASES.json", reports)
    return fields


def train_models(args, world: World) -> list[Path]:
    data_path = args.out / "TRAIN_TRANSITIONS.npz"
    if data_path.exists():
        with np.load(data_path) as data:
            data_table = {key: data[key] for key in data.files}
    else:
        data_table = collect_training(world, count=args.train_count, out=args.out)
    model_paths = []
    import torch
    torch.set_num_threads(1)
    for train_seed in TRAIN_SEEDS:
        name = args.out / f"cql_firstscan_{train_seed}.pt"
        model = learners.train(
            data_table, "cql", seed=train_seed, steps=args.steps,
            device=args.device, hidden=64, depth=2, batch_size=256,
            cql_alpha=.50,
        )
        learners.save(model, name)
        model_paths.append(name)
        print(f"trained {name.name} rows={len(data_table['reward'])}", flush=True)
    write_json(args.out / "TRAIN_MANIFEST.json", {
        "cases": int(args.train_count), "rows": int(len(data_table["reward"])),
        "seeds": list(TRAIN_SEEDS), "steps": int(args.steps),
        "feature_shape": list(data_table["features"].shape),
        "models": {path.name: sha(path) for path in model_paths},
        "source": "new first-scan fork-window task; no recheck checkpoint reused",
    })
    return model_paths


def paired_delta(rule: list[dict[str, Any]], variant: list[dict[str, Any]],
                 random: np.random.Generator, bootstraps: int = BOOTSTRAPS) -> dict[str, Any]:
    n = len(rule)
    indices = random.integers(0, n, size=(bootstraps, n))

    def scalar(field: str) -> dict[str, Any]:
        diff = np.asarray([float(b[field]) - float(a[field]) for a, b in zip(rule, variant)])
        sample = diff[indices].mean(axis=1)
        return {"mean": float(diff.mean()),
                "ci95": [float(v) for v in np.quantile(sample, [.025, .975])],
                "wins": int(np.sum(diff > 0)), "losses": int(np.sum(diff < 0))}

    def rmse(key: str) -> dict[str, Any]:
        left = np.asarray([float(a[key]["rmse_mm"]) for a in rule])
        right = np.asarray([float(b[key]["rmse_mm"]) for b in variant])
        sample = np.sqrt((right[indices] ** 2).mean(axis=1)) - np.sqrt((left[indices] ** 2).mean(axis=1))
        return {"mean": float(np.sqrt(np.mean(right ** 2)) - np.sqrt(np.mean(left ** 2))),
                "ci95": [float(v) for v in np.quantile(sample, [.025, .975])]}

    return {
        "strict": scalar("strict"), "both_correct": scalar("both_correct"),
        "wrong_accept": scalar("wrong_accept"), "travel_mm": scalar("travel_mm"),
        "full_scan_travel_mm": scalar("full_scan_travel_mm"),
        "elapsed_s": scalar("elapsed_s"), "current_rmse_delta_mm": rmse("current"),
        "future_half_s_rmse_delta_mm": rmse("future_half"),
        "common_clock_rmse_delta_mm": rmse("common_clock"),
    }


def arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(field: str) -> float:
        return float(np.mean([float(row[field]) for row in rows]))

    def pooled(key: str) -> float:
        return float(np.sqrt(np.mean([float(row[key]["rmse_mm"]) ** 2 for row in rows])))

    return {
        "n": len(rows), "strict_pass": int(sum(row["strict"] for row in rows)),
        "strict_rate": float(np.mean([row["strict"] for row in rows])),
        "both_correct": int(sum(row["both_correct"] for row in rows)),
        "wrong_accept": int(sum(row["wrong_accept"] for row in rows)),
        "wrong_clutter": int(sum(row["wrong_clutter"] for row in rows)),
        "wrong_branch": int(sum(row["wrong_branch"] for row in rows)),
        "partial_or_no_confirmation": int(sum(row["confirmed_labels"] < 2 for row in rows)),
        "prefix_provisional_wrong": int(sum(row["prefix_provisional_wrong"] for row in rows)),
        "prefix_confident_wrong": int(sum(row["prefix_confident_wrong"] for row in rows)),
        "prefix_wrong_replaced": int(sum(row["prefix_wrong_replaced"] for row in rows)),
        "prefix_clutter_replaced": int(sum(row["prefix_clutter_replaced"] for row in rows)),
        "prefix_branch_replaced": int(sum(row["prefix_branch_replaced"] for row in rows)),
        "prefix_confident_wrong_replaced": int(sum(row["prefix_confident_wrong_replaced"] for row in rows)),
        "new_confident_wrong": int(sum(row["new_confident_wrong"] for row in rows)),
        "current_rmse_mm": pooled("current"),
        "future_half_s_rmse_mm": pooled("future_half"),
        "future_one_s_rmse_mm": pooled("future_one"),
        "common_clock_rmse_mm": pooled("common_clock"),
        "mean_duration_s": mean("elapsed_s"), "mean_total_time_s": mean("total_elapsed_s"),
        "mean_travel_mm": mean("travel_mm"), "mean_fork_travel_mm": mean("fork_travel_mm"),
        "mean_prefix_travel_mm": mean("prefix_travel_mm"),
        "mean_full_scan_travel_mm": mean("full_scan_travel_mm"),
        "mean_captures": mean("captures"),
        "fork_timestamp_mismatch": int(sum(not row["fork_timestamps_equal"] for row in rows)),
    }


def run_test(args, world: World, model_paths: list[Path]) -> None:
    models = {path.stem: learners.load(path, device="cpu") for path in model_paths}
    out_path = args.out / "TEST.jsonl"
    with out_path.open("w") as stream:
        for offset in range(args.test_count):
            index = args.test_start + offset
            seed = new_seed("test", index)
            case = world.case(seed)
            prefix = state_from_prefix(world, case)
            schedule = build_schedule(world, case, prefix)
            frame_cache = {}
            arms = {
                "fixed_rule": run_policy(world, case, prefix, schedule, policy="fixed",
                                          frame_cache=frame_cache),
                "belief_rule": run_policy(world, case, prefix, schedule, policy="belief_rule",
                                           frame_cache=frame_cache),
            }
            for name, model in models.items():
                arms[name] = run_policy(world, case, prefix, schedule, model=model, policy="cql",
                                        frame_cache=frame_cache)
            fork_time_sets = [tuple(arm["fork_times_s"])
                              for arm in arms.values()]
            if any(not np.allclose(fork_time_sets[0], times, rtol=0., atol=1.e-9)
                   for times in fork_time_sets[1:]):
                raise RuntimeError("fork timestamps differ across paired arms")
            if len({arm["prefix_sha256"] for arm in arms.values()}) != 1:
                raise RuntimeError("prefix differs across paired arms")
            row = {
                "split": "test", "index": index, "seed": seed,
                "prefix_acquisitions": len(prefix.frames),
                "prefix_sha256": prefix_hash(prefix), "schedule": schedule,
                "arms": arms, "fork_timestamps_common": True,
            }
            stream.write(json.dumps(safe(row), allow_nan=False) + "\n")
            stream.flush()
            if (offset + 1) % 16 == 0:
                print(f"test cases {offset + 1}/{args.test_count}", flush=True)


def summarize(args, world: World) -> dict[str, Any]:
    rows = [json.loads(line) for line in (args.out / "TEST.jsonl").read_text().splitlines()
            if line.strip()]
    if not rows:
        raise RuntimeError("empty TEST.jsonl")
    arm_names = tuple(rows[0]["arms"])
    random = np.random.default_rng(93681)
    arms = {name: [row["arms"][name] for row in rows] for name in arm_names}
    summary = {
        "cases": len(rows), "indices": [row["index"] for row in rows],
        "seed_range": [min(row["seed"] for row in rows), max(row["seed"] for row in rows)],
        "protocol": {
            "task": "first-scan Y-window selection only; not full scan reorder",
            "fork_sites": list(FORK_SITES), "station_order": list(range(27, -1, -1)),
            "one_frame_per_station": True, "fixed_nonfork_view": 3,
            "shared_estimator": True, "no_fisher": True,
            "fork_timestamps_common": all(row["fork_timestamps_common"] for row in rows),
            "prefix_hashes_common_per_case": True,
        },
        "arms": {name: arm_summary(value) for name, value in arms.items()},
        "paired_vs_fixed_rule": {
            name: paired_delta(arms["fixed_rule"], value, random)
            for name, value in arms.items() if name != "fixed_rule"
        },
        "paired_vs_belief_rule": {
            name: paired_delta(arms["belief_rule"], value, random)
            for name, value in arms.items() if name not in ("fixed_rule", "belief_rule")
        },
        "bootstrap": {"replicates": BOOTSTRAPS, "seed": 93681, "unit": "case-paired"},
    }
    write_json(args.out / "SUMMARY.json", summary)
    return summary


def plot_3d(args, world: World, summary: Mapping[str, Any]) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as exc:  # pragma: no cover
        (args.out / "PLOT_SKIPPED.txt").write_text(f"Plotly unavailable: {exc}\n")
        return
    rows = [json.loads(line) for line in (args.out / "TEST.jsonl").read_text().splitlines()
            if line.strip()]
    chosen = next((row for row in rows
                   if len({tuple(arm["selected_views"].values()) for arm in row["arms"].values()}) > 1), rows[0])
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}]],
                        column_widths=[.66, .34],
                        subplot_titles=("Simulated first-scan acquisition poses",
                                        "Current / +0.5 s / shared-clock RMSE"))
    colors = {"fixed_rule": "#206f83", "belief_rule": "#c88828"}
    for i, name in enumerate(chosen["arms"]):
        arm = chosen["arms"][name]
        xyz = np.asarray([frame["probe_xyz_mm"] for frame in arm["trace"]])
        color = colors.get(name, "#8055a4")
        fig.add_trace(go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines+markers",
            name=f"{name} · {arm['travel_mm']:.1f} mm",
            line={"color": color, "width": 4}, marker={"color": color, "size": 3},
            text=[f"simulated station {frame['site']} view {frame['view']}"
                  for frame in arm["trace"]],
            hovertemplate="%{text}<extra>%{fullData.name}</extra>",
        ), row=1, col=1)
    # Show true Y locations at the fixed shared-clock time; this is a target,
    # not an interpolated acquisition trace.
    truth = np.asarray(chosen["arms"]["fixed_rule"]["common_clock"]["truth_xyz_mm"])
    fig.add_trace(go.Scatter3d(x=truth[:, 0], y=truth[:, 1], z=truth[:, 2],
                               mode="markers", name="True query points",
                               marker={"color": "#22272b", "size": 3}), row=1, col=1)
    labels = ["current", "+0.5 s", "+1.0 s", "shared clock"]
    for name in chosen["arms"]:
        arm = summary["arms"][name]
        fig.add_trace(go.Bar(x=labels, y=[arm["current_rmse_mm"], arm["future_half_s_rmse_mm"],
                                          arm["future_one_s_rmse_mm"], arm["common_clock_rmse_mm"]],
                             name=name), row=1, col=2)
    fig.update_scenes(aspectmode="data", xaxis_title="World X (mm)", yaxis_title="World Y (mm)",
                      zaxis_title="World Z (mm)", camera={"eye": {"x": 2., "y": 1.8, "z": 1.25}})
    fig.update_yaxes(title_text="3D world RMSE (mm)", rangemode="tozero", row=1, col=2)
    fig.update_layout(template="plotly_white", height=780,
                      title="First-scan Y-window selection<br><sup>segments connect simulated poses only; no full-scan reorder</sup>",
                      legend={"orientation": "h", "y": -.14}, margin={"l": 24, "r": 24, "t": 105, "b": 115})
    fig.write_html(args.out / "FIRST_SCAN_3D.html", include_plotlyjs=True)
    write_json(args.out / "PLOT_CASE.json", {"index": chosen["index"], "seed": chosen["seed"],
                                               "selected_views": {name: arm["selected_views"]
                                                                  for name, arm in chosen["arms"].items()},
                                               "simulated_poses_only": True})


def protocol_text(args) -> str:
    return f"""First-scan Y-window selection trial
Date: 2026-11-12

Scope
-----
This is first-scan window selection only, not a full scan reorder. Every arm
uses the frozen station order 27 -> 0, one frame per station, fixed view 3 at
non-fork stations, and the shared frozen Filter. Only legal window choices at
Y-fork stations 4 -> 3 -> 2 are changed. Candidate ingress from the previous
station and connection to the next default route are checked with the frozen
edge validator. The original fixed view is always included.

Timing and travel
-----------------
Per-case stage durations are padded to the maximum legal candidate envelope,
including all possible previous fork views. Thus the three fork acquisition
timestamps are identical across fixed rule, shared-estimator belief rule, and
CQL arms; modeled actual probe travel remains separately reported. The final
fixed stations 1 and 0 are also padded over the possible incoming envelope so
the shared clock is meaningful.
Raw rows retain legal view counts/options at each fork and the common true
world-clock forecast time. Observation noise is the frozen view/site/time-seeded
simulator noise, so changing a window can change simulated centroids even when
timestamps match.

Learning
--------
Training uses {args.train_count} new subjects at seed indices offset from the
frozen train split, two limited epsilon trajectories per subject, and terminal
truth only for offline reward generation. New CQL checkpoints are trained from
scratch for this first-scan action space; no recheck CQL checkpoint is loaded.
The primary held-out model is seed 7; seed 29 is reported as a stability check.
No Fisher variable or score is used.

Held-out
--------
The {args.test_count} held-out subjects start at fresh test index {args.test_start},
disjoint from the older train/test indices. Reported fields include strict and
both-correct identity, clutter/branch mis-ID, prefix replacement, current and
0.5/1.0 s/shared-clock 3-D map RMSE, modeled travel/time, and case-paired
bootstrap intervals.
"""


def manifest(args, model_paths: list[Path] | None = None) -> dict[str, Any]:
    names = ("belief.py", "knee_world.py", "policy.py", "experiment.py", "learners.py")
    return {
        "driver": sha(HERE / "first_scan.py"), "frozen_root": str(PREVIOUS),
        "frozen_sources": {name: sha(PREVIOUS / name) for name in names},
        "geometry": sha(GEOMETRY),
        "models": {path.name: sha(path) for path in (model_paths or [])},
        "config": {"train_count": args.train_count, "test_start": args.test_start,
                   "test_count": args.test_count, "steps": args.steps,
                   "train_seeds": list(TRAIN_SEEDS), "fork_sites": list(FORK_SITES),
                   "window_views": list(WINDOW_VIEWS)},
        "constraints": {"first_scan_only": True, "station_order_fixed": True,
                         "one_frame_per_station": True, "fisher_used": False,
                         "recheck_checkpoint_reused": False,
                         "fork_timestamps_common": True, "travel_reported_separately": True},
    }


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "test", "summary", "all"))
    parser.add_argument("--out", type=Path, default=HERE / "first_scan")
    parser.add_argument("--train-count", type=int, default=DEFAULT_TRAIN_COUNT)
    parser.add_argument("--test-start", type=int, default=DEFAULT_TEST_START)
    parser.add_argument("--test-count", type=int, default=DEFAULT_TEST_COUNT)
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.train_count <= 0 or args.test_count <= 0:
        raise ValueError("counts must be positive")
    (args.out / "PROTOCOL.txt").write_text(protocol_text(args))
    world = World()
    model_paths = sorted(args.out.glob("cql_firstscan_*.pt"))
    if args.mode in ("train", "all"):
        model_paths = train_models(args, world)
    if args.mode in ("test", "all"):
        if not model_paths:
            raise FileNotFoundError("first-scan CQL checkpoint; run train first")
        run_test(args, world, model_paths)
    if args.mode in ("summary", "all"):
        summary = summarize(args, world)
        plot_3d(args, world, summary)
    write_json(args.out / "MANIFEST.json", manifest(args, model_paths))
    print(json.dumps({"mode": args.mode, "out": str(args.out),
                      "models": [path.name for path in model_paths]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
