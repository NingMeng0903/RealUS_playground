"""Learn a clutter likelihood and test its incremental effect on two-step rechecks."""

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
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "ab_knee_latent_rl_20261112"
sys.path.insert(0, str(FROZEN))
from belief import (DEPTH, HALF_WIDTH, IMAGE_SD, NEAR_CLUTTER_RATE, NEAR_CLUTTER_SD,
                    OUTLIER_RATE, OUTLIER_SD, Filter, Reading, assignments,
                    in_view_probability, norm_pdf)
from experiment import State, current_identity, forecast_outcome, prefix_hash, seed_at
from knee_world import CAPTURE_S, Candidate, World
from policy import leaf_value


LOW = np.array([-HALF_WIDTH, DEPTH[0]])
HIGH = np.array([HALF_WIDTH, DEPTH[1]])
UNIFORM_DENSITY = 1. / np.prod(HIGH - LOW)
ANGULAR_SPEED_DEG_S = 120.
STRESS_CASES = ("original", "coherent_clutter")
ARMS = ("nominal", "pooled", "view", "view_replay")
SPLIT_TRAIN = 4096
SPLIT_TEST = 4096
BOOTSTRAPS = 4000


@dataclass(frozen=True)
class Clutter:
    chance: float
    weights: tuple[float, ...]
    sd: tuple[float, float]
    branch_centers: bool

    def centers(self, base):
        return base if self.branch_centers else base.mean(axis=0, keepdims=True)

    def density(self, obs, base):
        return component_densities(obs, self.centers(base), self.sd) @ self.weights

    def sample(self, base, rng):
        part = rng.choice(len(self.weights), p=self.weights)
        if not part:
            return rng.uniform(LOW, HIGH)
        center = self.centers(base)[part - 1]
        for _ in range(500):
            point = rng.normal(center, self.sd)
            if np.all((LOW <= point) & (point <= HIGH)):
                return point
        raise RuntimeError("learned clutter center was effectively outside the aperture")


def component_densities(obs, centers, sd):
    nearby = np.prod(norm_pdf(obs[:, None, :] - centers, sd), axis=-1)
    nearby /= np.maximum(in_view_probability(centers, sd), 1.e-9)
    return np.column_stack((np.full(len(obs), UNIFORM_DENSITY), nearby))


class AcousticFilter(Filter):
    def __init__(self, calibration):
        super().__init__(latent=True)
        self.calibration = calibration

    def clutter(self, world, site, view):
        if self.calibration is not None and view in self.calibration and world.identity_site(site):
            return self.calibration[view]
        return Clutter(world.clutter_probability(site, view),
                       (1. - NEAR_CLUTTER_RATE, NEAR_CLUTTER_RATE),
                       tuple(NEAR_CLUTTER_SD), False)

    def emission(self, world, reading, site, view, obs):
        if self.calibration is None or not world.identity_site(site):
            return super().emission(world, reading, site, view, obs)
        centers, hit = self.imaging(world, reading, site, view)
        n_obs, n_lumens = len(obs), centers.shape[-2]
        errors = obs[None, None, None, :, :] - centers[..., None, :]
        ordinary = np.prod(norm_pdf(errors, IMAGE_SD), axis=-1)
        broad = np.prod(norm_pdf(errors, OUTLIER_SD), axis=-1)
        detection = world.detection_probability(reading.q, site, view)
        observed = detection * ((1. - OUTLIER_RATE) * ordinary + OUTLIER_RATE * broad)
        base, _ = world.projection(reading.q, site, view)
        model = self.clutter(world, site, view)
        false_density = model.density(obs, base)
        likelihood = np.zeros_like(self.weights)
        responsibilities = np.zeros((n_obs, 3, *self.weights.shape))
        for matches, unused in assignments(n_lumens, n_obs):
            if len(unused) > 1:
                continue
            term = np.ones_like(self.weights)
            for label, j in enumerate(matches):
                term *= observed[..., label, j] if j >= 0 else 1. - hit[..., label]
            if unused:
                term *= model.chance * false_density[unused[0]]
            else:
                term *= 1. - model.chance
            likelihood += term
            for label, j in enumerate(matches):
                if j >= 0:
                    responsibilities[j, label] += term
            for j in unused:
                responsibilities[j, 2] += term
        return likelihood, responsibilities


@dataclass(frozen=True)
class Schedule:
    sites: tuple[int, ...]
    views: tuple[int, ...]
    durations: tuple[float, ...]
    return_goal: tuple[int, int]

    def options(self, world, reading, previous, stage):
        site = self.sites[stage]
        prior = (previous.site, previous.view)
        next_goal = ((self.sites[stage + 1], world.scan.central)
                     if stage + 1 < len(self.sites) else self.return_goal)
        actions = []
        for view in self.views:
            goal = (site, view)
            if (world.allowed[goal] and world.edge_valid(prior, goal)
                    and world.edge_valid(goal, next_goal)):
                move = world.make_action(reading.q, previous, goal, "recheck")
                turn = turn_angle(world, reading.q, prior, goal)
                if move.duration + turn / ANGULAR_SPEED_DEG_S > self.durations[stage] + 1.e-9:
                    raise RuntimeError("declared clock cannot fit a legal candidate")
                actions.append(replace(move, duration=self.durations[stage],
                                       t=previous.t + self.durations[stage]))
        if not actions:
            raise RuntimeError("no feasible recheck continuation")
        return actions


def turn_angle(world, q, start, end):
    frames = np.stack([world._blend(world.scan.view_frames[:, goal[0], goal[1]], q)
                       for goal in (start, end)])
    # Blend of pose knots needs projection back to a rotation before measuring geodesic angle.
    rotation = Rotation.from_matrix(frames)
    return float(np.rad2deg((rotation[0].inv() * rotation[1]).magnitude()))


def build_schedule(world):
    ordered = tuple(sorted(world.branch_sites, reverse=True))
    sites = ordered[:2]
    central = world.scan.central
    views = tuple(map(int, np.flatnonzero(world.scan.offsets == 0.)))
    if len(views) < 3 or len(ordered) < 3:
        raise RuntimeError("expected three legal centered tilt choices and three branch stations")
    previous = (world.initial_route[-1],)
    durations = []
    for stage, site in enumerate(sites):
        next_goal = (ordered[stage + 1], central)
        viable = []
        costs = []
        for start in previous:
            for view in views:
                goal = (site, view)
                if (not world.allowed[goal] or not world.edge_valid(start, goal)
                        or not world.edge_valid(goal, next_goal)):
                    continue
                viable.append(goal)
                candidate = Candidate(*start, 0., 0., 0., "clock")
                translation = max(world.make_action(q, candidate, goal, "clock").duration
                                  for q in world.knot_angles)
                # Max separately over the full interpolated pose interval, then pad.
                turn = max(turn_angle(world, q, start, goal) for q in np.linspace(11., 19., 33))
                costs.append(translation + turn / ANGULAR_SPEED_DEG_S)
        if not viable:
            raise RuntimeError("no recheck schedule")
        durations.append(float(max(costs) + .02))
        previous = tuple(set(viable))
    return Schedule(sites, views, tuple(durations), (ordered[2], central))


def fit_mixture(densities, prior):
    weights = prior / prior.sum()
    for _ in range(32):
        weighted = densities * weights
        posterior = weighted / np.maximum(weighted.sum(axis=1, keepdims=True), 1.e-300)
        weights = (posterior.sum(axis=0) + prior) / (len(densities) + np.sum(prior))
    return weights


def fit_models(world, schedule, count):
    observations = []
    frame_count = {view: 0 for view in schedule.views}
    train_seeds = []
    for i in range(count):
        seed = seed_at("train", SPLIT_TRAIN + i)
        train_seeds.append(seed)
        case = world.case(seed)
        start = world.initial(case)[-1].t
        for repeat in range(4):
            for j, site in enumerate((*schedule.sites, schedule.return_goal[0])):
                for view in schedule.views:
                    if not world.allowed[site, view]:
                        continue
                    t = start + .42 + .45 * j + .76 * repeat
                    frame = world.observe(case, t, site, view)
                    frame_count[view] += 1
                    base, _ = world.projection(frame.q, site, view)
                    for label, point in zip(frame.labels, frame.centroids):
                        if label == -1:
                            observations.append((view, point, base))
    if not observations:
        raise RuntimeError("training records did not contain any clutter")
    views = np.array([row[0] for row in observations])
    points = np.array([row[1] for row in observations])
    bases = np.array([row[2] for row in observations])

    # Fit a single residual scale and three mixture weights on training false detections.
    # Nearby components are tied to the public projected branch centers.
    best = None
    for sd in ((2., 1.), (3., 1.5), (4., 2.), (5., 2.5), (7., 3.5)):
        densities = np.stack([component_densities(p[None], b, sd)[0]
                              for p, b in zip(points, bases)])
        weights = fit_mixture(densities, np.array([1., 1., 1.]))
        score = float(np.log(densities @ weights).sum())
        if best is None or score > best[0]:
            best = score, sd, weights, densities
    _, sd, weights, densities = best
    chance = (len(points) + 2.) / (sum(frame_count.values()) + 10.)
    pooled = {view: Clutter(chance, tuple(weights), sd, True) for view in schedule.views}
    fitted = {}
    for view in schedule.views:
        keep = views == view
        # Shared population pseudocounts reduce unstable window-specific estimates.
        p = (int(keep.sum()) + 64. * chance) / (frame_count[view] + 64.)
        mixture = fit_mixture(densities[keep], 48. * weights)
        fitted[view] = Clutter(p, tuple(mixture), sd, True)
    return {"nominal": None, "pooled": pooled, "view": fitted}, {
        "cases": count, "seeds": train_seeds, "frames_by_view": frame_count,
        "false_points": len(points),
        "fits": {name: {view: asdict(model) for view, model in items.items()}
                 for name, items in (("pooled", pooled), ("view", fitted))},
    }


def acquire(world, case, action, mode):
    frame = world.observe(case, action)
    if mode == "original" or not world.identity_site(frame.site) or -1 not in frame.labels:
        return frame
    # Preserve the simulator's true masks, false-mask counts, geometry and timing.
    # Only the false structure is rendered with a smooth subject-specific offset.
    local = np.random.default_rng(np.random.SeedSequence([case.seed, 3217, round(frame.t * 1000), frame.site]))
    memory = np.random.default_rng(np.random.SeedSequence([case.seed, 4917]))
    harmonics = memory.standard_normal((2, 2))
    q, _ = case.true_angle_rate(frame.t)
    base, _ = world.projection(q, frame.site, frame.view)
    phase = frame.t * 2. * np.pi / 8.
    drift = np.array([np.cos(phase), np.sin(phase)]) @ harmonics
    angle = np.deg2rad(VIEW_TILTS[frame.view])
    tilt = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    offset = np.array([2.9, 1.35]) * (tilt @ drift)
    near = np.clip(base[case.ghost_label] + offset, LOW, HIGH)
    ghost = near if local.random() < .78 else local.uniform(LOW, HIGH)
    points = frame.centroids.copy()
    points[frame.labels.index(-1)] = ghost
    order = np.argsort(points[:, 0], kind="stable")
    return replace(frame, centroids=points[order], labels=tuple(frame.labels[j] for j in order))


def update(state, world, frame):
    belief = state.belief.copy()
    normalizer = belief.observe(world, Reading(frame.t, frame.q, frame.velocity),
                                frame.site, frame.view, frame.centroids, event=state.event,
                                retain=world.identity_site(frame.site))
    labels = state.labels.copy()
    if world.identity_site(frame.site):
        labels.update({(state.event, j): label for j, label in enumerate(frame.labels)})
    return State(belief, labels, frame, (*state.frames, frame), state.event + 1), -float(np.log(normalizer))


def imagined(world, belief, action, draws, stage):
    reading = belief.predicted_reading(action.t)
    weights, _ = belief.prior_at(reading)
    base, rows = world.projection(reading.q, action.site, action.view)
    centers = belief.means(base, rows, reading).reshape(-1, len(base), 2)
    p_detection = world.detection_probability(reading.q, action.site, action.view)
    clutter = belief.clutter(world, action.site, action.view)
    rng = np.random.default_rng(2026111200 + stage)
    indexes = np.searchsorted(np.cumsum(weights.ravel()), (np.arange(draws) + rng.uniform(.15, .85)) / draws)
    u = rng.random((draws, len(base)))
    sd = np.where(rng.random((draws, len(base), 1)) < OUTLIER_RATE, OUTLIER_SD, IMAGE_SD)
    shifted = centers[indexes] + sd * rng.standard_normal((draws, len(base), 2))
    false = rng.random(draws)
    for k, points in enumerate(shifted):
        keep = (u[k] < p_detection) & np.all((points >= LOW) & (points <= HIGH), axis=-1)
        obs = points[keep]
        if false[k] < clutter.chance:
            obs = np.vstack((obs, clutter.sample(base, rng)))
        yield reading, obs


def action_cost(world, belief, previous, action):
    degrees = turn_angle(world, belief.reading.q, (previous.site, previous.view), (action.site, action.view))
    return .055 * action.duration + .00045 * action.distance_mm + .055 * degrees / ANGULAR_SPEED_DEG_S


def plan(world, schedule, belief, previous, stage):
    actions = schedule.options(world, belief.reading, previous, stage)
    values = []
    for action in actions:
        outcomes = []
        for reading, obs in imagined(world, belief, action, 5 if stage == 0 else 3, stage):
            next_belief = belief.copy()
            next_belief.observe(world, reading, action.site, action.view, obs,
                                event=-2 + stage, retain=True)
            if stage + 1 < len(schedule.sites):
                _, scores = plan(world, schedule, next_belief, action, stage + 1)
                outcomes.append(max(scores))
            else:
                outcomes.append(leaf_value(world, next_belief))
        values.append(np.mean(outcomes) - action_cost(world, belief, previous, action))
    return actions, values


def first_sweep(world, case, mode):
    # Original fixed 28-site route and acquisition clock. Apply the same
    # declared acoustic stress to any fork frames from every policy.
    result = []
    for site, view in world.initial_route:
        if result:
            action = world.make_action(result[-1].q, result[-1], (site, view), "first sweep")
        else:
            action = Candidate(site, view, CAPTURE_S, 0., CAPTURE_S, "first sweep")
        result.append(acquire(world, case, action, mode))
    return tuple(result)


def run_arm(world, schedule, case, mode, prefix_frames, calibration, replay=None):
    prefix = State(AcousticFilter(calibration), {}, None, (), 0)
    for frame in prefix_frames:
        prefix, _ = update(prefix, world, frame)
    state = prefix
    suffix = []
    travel = turn = 0.
    nll = []
    for stage in range(len(schedule.sites)):
        if replay is None:
            actions, scores = plan(world, schedule, state.belief, state.previous, stage)
            action = actions[int(np.argmax(scores))]
            frame = acquire(world, case, action, mode)
        else:
            action, frame = replay[stage]
        turn += turn_angle(world, state.previous.q, (state.previous.site, state.previous.view),
                           (action.site, action.view))
        travel += action.distance_mm
        state, score = update(state, world, frame)
        suffix.append((action, frame))
        nll.append(score)
    _, _, labels, correct, wrong = current_identity(state)
    current = forecast_outcome(world, case, state, state.previous.t)
    future = forecast_outcome(world, case, state, state.previous.t + .5)
    result = {
        "strict": bool(all(correct) and current["max_mm"] <= 3.),
        "both_correct": bool(all(correct)), "wrong": bool(any(wrong)),
        "clutter_wrong": bool(any(flag and label == -1 for flag, label in zip(wrong, labels))),
        "rejected": bool(sum(correct) + sum(wrong) < 2),
        "current_rmse_mm": current["rmse_mm"], "future_rmse_mm": future["rmse_mm"],
        "future_90_coverage": future["coordinate_90_coverage"],
        "nll": float(np.mean(nll)), "travel_mm": travel, "turn_deg": turn,
        "end_t": state.previous.t, "captures": len(suffix), "prefix_sha": prefix_hash(prefix),
        "views": [frame.view for _, frame in suffix],
        "times": [frame.t for _, frame in suffix],
        "probe_xyz_mm": [world.probe_world(frame.q, frame.site, frame.view).tolist()
                         for _, frame in suffix],
        "frame_centroids": [frame.centroids.tolist() for _, frame in suffix],
    }
    return result, suffix


def test_one(item):
    index, mode = item
    seed = seed_at("test", SPLIT_TEST + index)
    case = WORLD.case(seed)
    prefix = first_sweep(WORLD, case, mode)
    arms = {}
    for name in ARMS:
        replay = nominal_suffix if name == "view_replay" else None
        result, suffix = run_arm(WORLD, SCHEDULE, case, mode, prefix,
                                 MODELS["view" if name == "view_replay" else name], replay)
        if name == "nominal":
            nominal_suffix = suffix
        arms[name] = result
    assert len({r["prefix_sha"] for r in arms.values()}) == 1
    assert len({tuple(r["times"]) for r in arms.values()}) == 1
    assert arms["view_replay"]["views"] == arms["nominal"]["views"]
    assert arms["view_replay"]["frame_centroids"] == arms["nominal"]["frame_centroids"]
    return {"index": index, "seed": seed, "mode": mode, "arms": arms}


def summarize(records):
    rng = np.random.default_rng(2026111214)
    result = {}
    for mode in STRESS_CASES:
        rows = [r["arms"] for r in records if r["mode"] == mode]
        samples = rng.integers(0, len(rows), size=(BOOTSTRAPS, len(rows)))
        block = {"cases": len(rows), "arms": {}, "paired": {}}
        for name in ARMS:
            series = [r[name] for r in rows]
            block["arms"][name] = {
                key: int(sum(v[key] for v in series)) for key in
                ("strict", "both_correct", "wrong", "clutter_wrong", "rejected")
            }
            for key in ("nll", "travel_mm", "turn_deg", "future_90_coverage"):
                block["arms"][name][key] = float(np.mean([v[key] for v in series]))
            for key in ("current_rmse_mm", "future_rmse_mm"):
                block["arms"][name][key] = float(np.sqrt(np.mean([v[key]**2 for v in series])))
            block["arms"][name]["changed_view_cases"] = sum(r[name]["views"] != r["nominal"]["views"] for r in rows)
        pairs = (("view", "nominal"), ("pooled", "nominal"), ("view", "pooled"),
                 ("view_replay", "nominal"), ("view", "view_replay"))
        for name, reference in pairs:
            contrast = {}
            for key in ("strict", "both_correct", "wrong", "nll", "travel_mm", "turn_deg"):
                delta = np.array([float(r[name][key]) - float(r[reference][key]) for r in rows])
                contrast[key] = {"difference": float(delta.mean()),
                                 "ci95": np.quantile(delta[samples].mean(axis=1), [.025, .975]).tolist()}
            left = np.array([r[name]["future_rmse_mm"]**2 for r in rows])
            right = np.array([r[reference]["future_rmse_mm"]**2 for r in rows])
            delta = np.sqrt(left[samples].mean(axis=1)) - np.sqrt(right[samples].mean(axis=1))
            contrast["future_rmse_mm"] = {"difference": float(np.sqrt(left.mean()) - np.sqrt(right.mean())),
                                         "ci95": np.quantile(delta, [.025, .975]).tolist()}
            block["paired"][f"{name}_minus_{reference}"] = contrast
        result[mode] = block
    return result


def plot(output, records, summary):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    example = next((r for r in records if r["mode"] == "coherent_clutter"
                    and r["arms"]["nominal"]["views"] != r["arms"]["view"]["views"]),
                   records[0])
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}]],
                        subplot_titles=("Baked leg and recheck poses", "Strict completions"))
    pose = int(np.argmin(np.abs(WORLD.angles - 15.)))
    trunk = WORLD.scan.trunk[pose] * 1000.
    branch = WORLD.scan.dense_branch[pose] * 1000.
    for label, line in (("Atlas main vessel", trunk), ("Atlas branch", branch)):
        fig.add_trace(go.Scatter3d(x=line[:, 0], y=line[:, 1], z=line[:, 2],
                                  mode="lines", name=label, line=dict(width=6, color="#a8b3bf")), row=1, col=1)
    colors = {"nominal": "#1e617c", "pooled": "#b68036", "view": "#9a4970", "view_replay": "#555e6c"}
    labels = {"nominal": "Nominal likelihood", "pooled": "Pooled fit", "view": "View-conditioned fit",
              "view_replay": "Fit on nominal captures"}
    for name in ARMS:
        row = example["arms"][name]
        if name != "view_replay":
            points = np.array(row["probe_xyz_mm"])
            fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                                      mode="markers", name=f"{labels[name]} · views {row['views']}",
                                      marker=dict(color=colors[name], size=6),
                                      hovertemplate="Acquisition pose<extra>%{fullData.name}</extra>"), row=1, col=1)
        fig.add_trace(go.Bar(x=["Original", "Coherent clutter"],
                             y=[summary[mode]["arms"][name]["strict"] for mode in STRESS_CASES],
                             name=labels[name], marker_color=colors[name]), row=1, col=2)
    fig.update_layout(title="Learned acoustic likelihood for two-step recheck", template="plotly_white",
                      legend=dict(orientation="h", y=-.13), margin=dict(l=35, r=25, t=95, b=115), height=620)
    fig.update_scenes(xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)", aspectmode="data")
    fig.update_yaxes(title_text=f"Cases / {summary['original']['cases']}", row=1, col=2)
    fig.write_html(output / "ACQUISITION_3D.html", include_plotlyjs=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "acquisition")
    parser.add_argument("--train", type=int, default=256)
    parser.add_argument("--test", type=int, default=128)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    global WORLD, SCHEDULE, MODELS, VIEW_TILTS
    WORLD = World()
    with np.load(WORLD.geometry_path, allow_pickle=False) as asset:
        VIEW_TILTS = asset["view_tilts_deg"]
    SCHEDULE = build_schedule(WORLD)
    MODELS, metadata = fit_models(WORLD, SCHEDULE, args.train)
    metadata["schedule"] = asdict(SCHEDULE)
    metadata["geometry_sha256"] = hashlib.sha256(WORLD.geometry_path.read_bytes()).hexdigest()
    metadata["split"] = {"train_start": SPLIT_TRAIN, "test_start": SPLIT_TEST,
                         "test_count": args.test}
    (args.out / "TRAIN.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Trained on {args.train} cases; testing {args.test} per condition", flush=True)
    tasks = [(i, mode) for i in range(args.test) for mode in STRESS_CASES]
    records = []
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=get_context("fork")) as pool:
        with (args.out / "TEST.jsonl").open("w") as stream:
            for i, row in enumerate(pool.map(test_one, tasks, chunksize=2), 1):
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                records.append(row)
                if i % 32 == 0:
                    print(f"Evaluated {i}/{len(tasks)}", flush=True)
    assert not set(metadata["seeds"]) & {r["seed"] for r in records}
    summary = summarize(records)
    (args.out / "SUMMARY.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plot(args.out, records, summary)


if __name__ == "__main__":
    main()
