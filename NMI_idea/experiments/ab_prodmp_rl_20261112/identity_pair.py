"""Train a one-switch, rule-relative continuation on paired simulated actions."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import rl_rollouts as rollout
from compare_data_quality import _arm_metrics, _paired, _slim_run
from summarize_rl import _load_geometry, _path_points


HERE = Path(__file__).resolve().parent
OUT = HERE / "identity_pair_20261112"
TRAIN_SEED = 30476123
TEST_SEED = 70314297
RIDGE = 12.
MARGIN = .025
ARMS = ("rule", "posterior", "paired")


def seed_at(base, index):
    return int(np.random.SeedSequence([base, index]).generate_state(1)[0])


def planner_without_fisher():
    for name in ("locked_seen", "B_seen"):
        rollout.study.ARMS[name] = replace(rollout.study.ARMS[name], fisher=0.)
    return rollout._make_model_planner(n_predictive=8, n_first=3, switch_margin=.006)


def eligible(shortlist, observations):
    return (bool(observations) and shortlist.mask.sum() > 1
            and (len(observations) > 1 or not shortlist.anchor_choice.usable))


def features(shortlist, observations, slot):
    anchor, proposal = np.asarray(shortlist.features[[0, slot]], dtype=float)
    values = np.asarray(observations, dtype=float) / 25.
    # Counts and variances are already normalized by the original evidence limits.
    counts, variance = anchor[6:38], anchor[38:70]
    site, view, distance, score = proposal[-6:-2]
    site0, view0, distance0, score0 = anchor[-6:-2]
    return np.asarray([
        anchor[0], anchor[3], anchor[4], anchor[5],
        min(len(values), 3), np.min(np.abs(values)), np.ptp(values),
        np.mean(counts < 1.), np.mean(variance > 1.),
        site, view, distance, score,
        site - site0, view - view0, distance - distance0, score - score0,
        float(shortlist.indices[slot] == shortlist.b_index),
    ], dtype=float)


def run(planner, scan, support, model, seed, prepared, picker=None):
    return rollout.run_episode(
        planner, scan, support, model, episode_seed=seed, mode="evaluate", policy="rule",
        q_model=None, cap=80, prepared_episode=prepared, tail_override=picker,
    )


def outcome(row):
    return {
        "strict_complete": bool(row["complete"]),
        "online_stop": bool(row["online_stop"]),
        "captures": len(row["steps"]),
        "path_m": float(row["recheck_path_mm"]) / 1000.,
        "incorrect_witnesses": row["incorrect_witnesses"],
        "map_max_error_mm": row["map_max_error_mm"],
    }


def value(row):
    return float(row["strict_complete"]) - .002 * row["captures"] - .05 * row["path_m"]


def _common_head(row, stop):
    return [(x["pose"], x["action"], x["centroids_mm"], x["n_ready"]) for x in row["steps"][:stop]]


def collect_one(stack, seed):
    scan, support, model, planner = stack
    prepared = rollout.prepare_episode(scan, support, seed)
    rule, decisions = run(planner, scan, support, model, seed, prepared)
    base = outcome(rule)
    for item in decisions:
        k, decision = item["step_index"], item["decision"]
        observations = tuple(rule["steps"][k - 1]["centroids_mm"])
        shortlist = decision.shortlist
        if eligible(shortlist, observations):
            break
    else:
        return {"seed": seed, "anchor": base, "options": []}

    number = decision.decision_number
    options = []
    for slot in np.flatnonzero(shortlist.mask)[1:]:
        slot = int(slot)

        def force(current, values, decision_number, default_slot):
            if decision_number == number:
                assert np.array_equal(current.indices, shortlist.indices) and values == observations
                return slot
            return None

        alternative, _ = run(planner, scan, support, model, seed, prepared, force)
        assert alternative["prefix_steps"] == rule["prefix_steps"]
        assert _common_head(alternative, k) == _common_head(rule, k)
        assert alternative["steps"][k]["tail_choice"]["selected_slot"] == slot
        comparison = outcome(alternative)
        options.append({
            "slot": slot, "action": alternative["steps"][k]["action"],
            "features": features(shortlist, observations, slot).tolist(),
            "alternative": comparison, "advantage": value(comparison) - value(base),
        })
    return {
        "seed": seed, "anchor": base, "decision_number": number,
        "public_observations_mm": list(observations), "options": options,
    }


def read_rows(directory, prefix):
    rows = []
    for path in sorted(directory.glob(f"{prefix}_*.jsonl")):
        with path.open() as stream:
            rows.extend(json.loads(line) for line in stream)
    return rows


def fit(directory):
    rows = read_rows(directory, "train")
    expected = {seed_at(TRAIN_SEED, i) for i in range(64)}
    if len(rows) != 64 or {r["seed"] for r in rows} != expected:
        raise ValueError("incomplete paired training cases")
    actions = [option for row in rows for option in row["options"]]
    x = np.asarray([a["features"] for a in actions], dtype=float)
    y = np.asarray([a["advantage"] for a in actions], dtype=float)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1.e-8] = 1.
    xn = (x - mean) / scale
    weight = np.linalg.solve(xn.T @ xn + RIDGE * np.eye(x.shape[1]), xn.T @ y)
    np.savez(directory / "PAIR_MODEL.npz", mean=mean, scale=scale, weight=weight, intercept=y.mean())
    result = {
        "training_cases": len(rows), "cases_with_a_pair": sum(bool(r["options"]) for r in rows),
        "paired_actions": len(y), "mean_advantage": float(y.mean()),
        "advantages_above_policy_margin": int((y > MARGIN).sum()),
        "rule_successes": sum(r["anchor"]["strict_complete"] for r in rows),
        "rule_identity_failures": sum(bool(r["anchor"]["incorrect_witnesses"]) and not r["anchor"]["strict_complete"] for r in rows),
        "rule_failures_with_a_passing_alternative": sum(
            not r["anchor"]["strict_complete"] and any(a["alternative"]["strict_complete"] for a in r["options"])
            for r in rows
        ),
        "discordant_action_pairs": sum(
            r["anchor"]["strict_complete"] != a["alternative"]["strict_complete"]
            for r in rows for a in r["options"]
        ),
    }
    (directory / "TRAIN_SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def test_one(stack, seed, checkpoint):
    scan, support, model, planner = stack
    prepared = rollout.prepare_episode(scan, support, seed)
    arms = {}
    prefix = None
    for arm in ARMS:
        used = False

        def choose(shortlist, observations, decision_number, default_slot):
            nonlocal used
            if used or not eligible(shortlist, observations):
                return None
            used = True
            if arm == "posterior":
                return int(np.flatnonzero(shortlist.indices == shortlist.b_index)[0])
            if arm == "paired":
                valid = np.flatnonzero(shortlist.mask)[1:]
                x = np.stack([features(shortlist, observations, slot) for slot in valid])
                values = (x - checkpoint["mean"]) / checkpoint["scale"] @ checkpoint["weight"] + checkpoint["intercept"]
                best = int(np.argmax(values))
                return int(valid[best]) if values[best] > MARGIN else None
            return None

        row, _ = run(planner, scan, support, model, seed, prepared, None if arm == "rule" else choose)
        if prefix is None:
            prefix = row["prefix_steps"]
        elif row["prefix_steps"] != prefix:
            raise ValueError("initial sweeps differ")
        arms[arm] = _slim_run(row)
    return {"seed": seed, "arms": arms}


def summarize(directory):
    records = read_rows(directory, "test")
    seeds = [seed_at(TEST_SEED, i) for i in range(48)]
    train_seeds = {r["seed"] for r in read_rows(directory, "train")}
    if set(seeds) & train_seeds or len(records) != len(seeds) or {r["seed"] for r in records} != set(seeds):
        raise ValueError("holdout is incomplete or overlaps training")
    cases = {r["seed"]: r["arms"] for r in records}
    result = {
        "cases": len(cases),
        "arms": {arm: _arm_metrics([r[arm] for r in cases.values()]) for arm in ARMS},
        "paired": {arm: _paired(cases, arm, "rule") for arm in ARMS[1:]},
    }
    for arm in ARMS:
        all_runs = [r[arm] for r in cases.values()]
        wrong = [r for r in all_runs if not r["strict_complete"]]
        result["arms"][arm].update({
            "wrong_identity_failures": sum(bool(r["incorrect_witnesses"]) for r in wrong),
            "map_error_failures": sum(r["map_max_error_mm"] > 3. for r in wrong),
            "changed_cases": sum(bool(r["actual_switch_count"]) for r in all_runs),
        })
    (directory / "SUMMARY.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    example = next((i for i in seeds if cases[i]["paired"]["strict_complete"] != cases[i]["rule"]["strict_complete"]),
                   seeds[0])
    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}]], column_widths=[.55, .45],
        subplot_titles=("Recheck paths in one discordant case", "Strict completion across 48 cases"),
    )
    geometry = _load_geometry()
    for key, title in (("trunk", "Trunk template"), ("branch", "Branch template")):
        xyz = np.asarray(geometry[key])
        fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines", name=title,
                                   line={"color": "#adb8bb", "width": 4}), row=1, col=1)
    for arm, label, color in zip(ARMS, ("Rule", "Posterior reselection", "Learned reselection"), ("#277083", "#9476a9", "#c27636")):
        xyz = np.asarray(_path_points(cases[example][arm]))
        if xyz.size:
            fig.add_trace(go.Scatter3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], name=label, mode="lines",
                line={"color": color, "width": 5}, legendgroup=arm,
            ), row=1, col=1)
        caps = np.arange(81)
        y = [sum(r[arm]["strict_complete"] and r[arm]["captures"] <= c for r in cases.values()) / len(cases) for c in caps]
        fig.add_trace(go.Scatter(x=caps, y=y, name=label, showlegend=False, legendgroup=arm,
                                 line={"color": color, "shape": "hv", "width": 3}), row=1, col=2)
    fig.update_scenes(aspectmode="data", camera={"eye": {"x": 1.65, "y": 1.65, "z": 1.65}},
                      xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)")
    fig.update_xaxes(title_text="Maximum rechecks", range=[20, 80], row=1, col=2)
    fig.update_yaxes(title_text="Strict completion rate", tickformat=".0%", range=[0., 1.02], row=1, col=2)
    fig.update_layout(template="plotly_white", height=740,
                      title="Paired recheck policies<br><sup>Shared initial sweep · Simulated observations · Reference paths</sup>",
                      margin={"l": 25, "r": 30, "t": 116, "b": 105},
                      legend={"orientation": "h", "y": -.12})
    fig.write_html(directory / "IDENTITY_PAIR_3D.html", include_plotlyjs=True)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("collect", "fit", "test", "summarize"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "fit":
        return fit(args.out)
    if args.mode == "summarize":
        return summarize(args.out)
    stack = planner_without_fisher()
    checkpoint = None
    if args.mode == "test":
        with np.load(args.out / "PAIR_MODEL.npz", allow_pickle=False) as model:
            checkpoint = dict(model)
    base = TRAIN_SEED if args.mode == "collect" else TEST_SEED
    prefix = "train" if args.mode == "collect" else "test"
    with (args.out / f"{prefix}_{args.start}.jsonl").open("w") as stream:
        for index in range(args.start, args.start + args.n):
            seed = seed_at(base, index)
            row = collect_one(stack, seed) if args.mode == "collect" else test_one(stack, seed, checkpoint)
            stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
            stream.flush()


if __name__ == "__main__":
    main()
