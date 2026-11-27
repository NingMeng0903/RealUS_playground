"""Paired pilot for causal search data and a chart-position embedding.

The script imports the frozen recovery benchmark but does not edit it. Each
controller arm receives the same hidden Scene and starts from the same public
state. Truth is used only by the evaluator after an episode, or to hide the
target in the synthetic visibility tape.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import gzip
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
RECOVERY = HERE.parent / "ab_recovery_training_20261115"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RECOVERY))

from controller import (  # noqa: E402
    EDGES, END, FRAME_COUNT, FRAME_STRIDE, FRAMES_PER_BLOCK, NAMES,
    Record, State, Controller, fork,
)
from embedding import position_embedding  # noqa: E402
from search_actions import add_search_sweeps  # noqa: E402
from world import TARGET, World  # noqa: E402

SEED_BASE = {"train": 3_100_000, "dev": 3_200_000, "test": 3_300_000}
WORLD_SPLIT = {"train": "train", "dev": "dev", "test": "heldout"}
CATEGORIES = ("nominal", "offset", "loss")
LOSS_FRAMES = (35, 36, 37, 38)
OUTCOME_KEYS = ("success", "wrong", "wrong_certification", "coverage", "arrived",
                "tcp_mm", "turn_deg", "seconds", "frames", "mean_abs_lateral_mm")


class ConstantModel:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, x):
        return np.full(len(np.asarray(x)), self.value, dtype=float)


def _json(stream, value):
    stream.write(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")


def _checkpoint(state: State) -> dict:
    value = dict(vars(state))
    value["belief"] = np.asarray(state.belief, dtype=float).tolist()
    return value


def _archive(record: Record, *, prefix_frames=0, prefix_samples=0) -> dict:
    return {
        "packet_offset": int(prefix_frames), "motion_offset": int(prefix_samples),
        "packets": record.packets[prefix_frames:],
        "motion_t_s_x_vs_vx_q": record.motion[prefix_samples:],
        "decisions": record.decisions, "accepts": record.accepts,
        "blocks": record.blocks,
    }


def _load_sections() -> dict:
    path = RECOVERY.parent / "ab_streaming_prodmp_two_vessel_20261113" / "SECTIONS.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_coefficients() -> dict:
    path = RECOVERY / "results" / "prodmp_weights.npz"
    if not path.exists():
        raise FileNotFoundError(f"frozen recovery ProDMP weights not found: {path}")
    with np.load(path) as data:
        return {key: np.asarray(data[key], dtype=float) for key in data.files}


def _category(index: int) -> str:
    return CATEGORIES[int(index) % len(CATEGORIES)]


def _case(world: World, split: str, index: int, category: str):
    scene = world.scene(SEED_BASE[split] + int(index), WORLD_SPLIT[split],
                       FRAME_COUNT, False, alias=False)
    if category == "loss":
        clear = np.asarray(scene.clear, dtype=bool).copy()
        clear[list(LOSS_FRAMES), TARGET] = False
        scene = replace(scene, clear=clear)
    return scene


def _initial(ctrl: Controller, scene, lateral_offset: float):
    x = float(np.clip(ctrl.world.nominal(2.0)[0] + lateral_offset, -30., 0.))
    state = State(ctrl.world.prior.copy(), 2.0, x, angle=float(scene.angles_deg[0]))
    record = Record(motion=[[0., 2., x, 0., 0., float(scene.angles_deg[0])]])
    ctrl.observe(scene, state, record, False)
    return state, record


def _initial_offset(category: str, index: int) -> float:
    if category != "offset":
        return 0.0
    return 2.5 if int(index) % 2 == 0 else -2.5


def _checkpoint_kind(state: State) -> str | None:
    """Public condition, shared by collection and every evaluation arm."""

    if state.missed >= 2:
        return "lost"
    if state.trigger():
        return "unresolved"
    if state.slot >= 0 and abs(state.residual) >= 1.5:
        return "near_edge"
    if state.s >= END - .30 and state.uncovered():
        return "uncovered_end"
    return None


def _active(state: State) -> bool:
    return _checkpoint_kind(state) is not None


def _menu(ctrl: Controller, state: State):
    choices = ctrl.options(state, False, full=_active(state))
    return add_search_sweeps(ctrl, state, False, choices) if _active(state) else choices


def _mpc_tail(ctrl: Controller, scene, state: State, record: Record):
    """Complete observation-updating MPC shared by branch labels and test."""

    while state.frame + FRAMES_PER_BLOCK < FRAME_COUNT:
        if state.s >= END - .065 and not state.uncovered():
            break
        choices = _menu(ctrl, state)
        if not choices:
            break
        candidate = ctrl.mpc(state, choices, False, active=_active(state), informative=True)
        ctrl.execute(scene, state, record, candidate, False)
    return state, record


def _checkpoints(ctrl: Controller, scene, category: str, index: int):
    """At most two different public recovery contexts from one MPC prefix."""

    state, record = _initial(ctrl, scene, _initial_offset(category, index))
    selected = []
    seen = set()
    while state.frame + FRAMES_PER_BLOCK < FRAME_COUNT:
        if state.s >= END - .065 and not state.uncovered():
            break
        kind = _checkpoint_kind(state)
        if kind is not None and kind not in seen and len(selected) < 2:
            before, prefix = fork(state, record)
            selected.append((before, prefix, kind))
            seen.add(kind)
        choices = _menu(ctrl, state)
        if not choices:
            break
        candidate = ctrl.mpc(state, choices, False, active=_active(state), informative=True)
        ctrl.execute(scene, state, record, candidate, False)
    return selected


def _packet_tokens(record: Record, state: State) -> np.ndarray:
    """Public anonymous packet tokens; no roles or target truth are read."""

    tokens = np.zeros((3, 6), dtype=float)
    if not record.packets:
        return tokens.reshape(-1)
    packet = record.packets[-1]
    ys = np.asarray(packet.get("y", []), dtype=float)
    assoc = np.asarray(packet.get("association", []), dtype=float)
    n = min(len(ys), 3)
    for i in range(n):
        yx, yz = ys[i]
        responsibility = float(assoc[i]) if i < len(assoc) else 0.0
        tokens[i] = [
            np.clip(yx / 30.0, -2.0, 2.0),
            np.clip((yz - 27.5) / 30.0, -2.0, 2.0),
            np.clip((float(packet.get("s", state.s)) - state.s) / 20.0, -2.0, 2.0),
            np.clip((state.x + yx) / 30.0, -2.0, 2.0),
            np.clip(responsibility, 0.0, 1.0),
            1.0,
        ]
    return tokens.reshape(-1)


def _embedding(ctrl: Controller, state: State, candidate, record: Record) -> np.ndarray:
    """Embedding plus candidate-path acoustic coverage summaries.

    The base module receives endpoint target predictions.  These additional
    features use the same public ``predict`` call at four points along the
    candidate, so the footprint is represented as a path rather than one
    endpoint.  The summaries are not multiplied into the Bayesian likelihood.
    """

    s_values = np.asarray(candidate.s, dtype=float)
    x_values = np.asarray(candidate.lateral.x, dtype=float)
    sample_indices = np.linspace(0, len(s_values) - 1, 4).round().astype(int)
    coverage = []
    top_coverage = []
    posterior = np.asarray(state.belief, dtype=float)
    order = np.argsort(posterior)[::-1][:3]
    for index in sample_indices:
        _, visibility = ctrl.world.predict(float(s_values[index]), float(x_values[index]),
                                           state.angle, alias=False)
        target_visibility = np.asarray(visibility[:, 0], dtype=float)
        coverage.append(float(np.clip(posterior @ target_visibility, 0.0, 1.0)))
        top_coverage.append(float(np.max(target_visibility[order])) if len(order) else 0.0)
    _, visibility = ctrl.world.predict(float(s_values[-1]), float(x_values[-1]),
                                       state.angle, alias=False)
    # ``predict`` returns probe-relative means.  The embedding's target token
    # expects the public chart prediction, so use target_grid directly and
    # subtract the candidate window exactly once inside ``position_embedding``.
    target_positions = ctrl.world.target_grid(float(s_values[-1]), state.angle, alias=False)
    base = position_embedding(
        current_s=state.s,
        current_x=state.x,
        candidate_s=s_values,
        candidate_x=x_values,
        query_interval=(float(EDGES[state.query_bin()]),
                        float(EDGES[state.query_bin() + 1])),
        last_trusted_s=state.last_trusted,
        belief=posterior,
        target_positions=np.asarray(target_positions, dtype=float),
        target_visibility=np.asarray(visibility[:, 0], dtype=float),
        packet_responsibilities=(
            np.asarray(record.packets[-1].get("association", []), dtype=float)
            if record.packets else None),
        angle_deg=state.angle,
    )
    return np.r_[base, np.asarray(coverage), np.asarray(top_coverage), _packet_tokens(record, state)]


def _feature_pair(ctrl: Controller, state: State, candidate, record: Record):
    base = np.asarray(ctrl.policy_features(state, candidate, False), dtype=float)
    return base, np.r_[base, _embedding(ctrl, state, candidate, record)]


def _measures(ctrl: Controller, record: Record, state: State) -> dict:
    values = record.details(state)
    values["success"] = bool(values.pop("complete"))
    values["false_complete"] = bool(values["arrived"] and not values["success"]
                                    and not state.uncovered())
    values["mean_abs_lateral_mm"] = float(np.mean([
        abs(float(packet["target_truth"][0]) - float(packet["x"]))
        for packet in record.packets
    ])) if record.packets else 0.0
    trace = np.asarray(record.motion, dtype=float)
    values["tcp_mm"], values["turn_deg"] = ctrl.geometry.cost(trace[:, 1], trace[:, 2])
    values["tcp_cost_error"] = abs(float(values["tcp_mm"]) - float(state.travel))
    return values


def _branch_result(ctrl: Controller, scene, before: State, prefix: Record, candidate):
    state, record = fork(before, prefix)
    ctrl.execute(scene, state, record, candidate, False)
    _mpc_tail(ctrl, scene, state, record)
    return _measures(ctrl, record, state), record


def _collect_branches(ctrl: Controller, world: World, split: str, count: int, out: Path):
    rows = []
    with gzip.open(out / f"{split}_branches.jsonl.gz", "wt", encoding="utf-8",
                   compresslevel=2) as stream:
        for index in range(int(count)):
            category = _category(index)
            scene = _case(world, split, index, category)
            for checkpoint_number, (state, prefix, checkpoint_kind) in enumerate(
                    _checkpoints(ctrl, scene, category, index)):
                choices = _menu(ctrl, state)
                detail = {
                    "split": split, "index": int(index), "seed": SEED_BASE[split] + int(index),
                    "category": category, "checkpoint_number": checkpoint_number,
                    "checkpoint_kind": checkpoint_kind,
                    "checkpoint": _checkpoint(state), "prefix": _archive(prefix),
                    "branches": {},
                }
                for name, candidate in choices.items():
                    result, suffix = _branch_result(ctrl, scene, state, prefix, candidate)
                    base, extended = _feature_pair(ctrl, state, candidate, prefix)
                    detail["branches"][name] = {
                        "action": name,
                        "result": {key: result[key] for key in OUTCOME_KEYS},
                        "features": base.tolist(),
                        "features_embedding": extended.tolist(),
                        "candidate": {"s": np.asarray(candidate.s).tolist(),
                                      "x": np.asarray(candidate.lateral.x).tolist(),
                                      "distance": float(candidate.distance),
                                      "turn": float(candidate.turn)},
                        "suffix": _archive(suffix, prefix_frames=len(prefix.packets),
                                            prefix_samples=len(prefix.motion) - 1),
                    }
                    rows.append({
                        "features": base, "features_embedding": extended,
                        "success": float(result["success"]), "wrong": float(result["wrong"]),
                        "action": name, "category": category,
                        "checkpoint_kind": checkpoint_kind,
                    })
                _json(stream, detail)
    return rows


def _fit_models(rows, key: str):
    x = np.asarray([row[key] for row in rows], dtype=float)
    y = np.asarray([[row["success"], row["wrong"]] for row in rows], dtype=float)
    models = []
    for target in y.T:
        if len(target) < 4 or np.allclose(target, target[0]):
            models.append(ConstantModel(float(np.mean(target))))
        else:
            model = HistGradientBoostingRegressor(
                max_iter=80, max_leaf_nodes=9, learning_rate=.07,
                min_samples_leaf=max(2, min(16, len(target) // 5)),
                l2_regularization=4.0, random_state=902)
            model.fit(x, target)
            models.append(model)
    return models


def _fixed_action(rows):
    values = {}
    for action in NAMES:
        subset = [row for row in rows if row["action"] == action]
        if subset:
            values[action] = float(np.mean([
                row["success"] - 1.5 * row["wrong"] for row in subset]))
    return (max(values, key=values.get) if values else "advance"), values


def _score(ctrl: Controller, state: State, record: Record, choices: dict, models, use_embedding: bool):
    rows, names = [], []
    for name, candidate in choices.items():
        base, extended = _feature_pair(ctrl, state, candidate, record)
        rows.append(extended if use_embedding else base)
        names.append(name)
    x = np.asarray(rows, dtype=float)
    pred = np.column_stack([np.clip(model.predict(x), 0., 1.) for model in models])
    result = {}
    for name, value in zip(names, pred, strict=True):
        # Correct completion is primary; wrong acceptance and first-block
        # travel are secondary. No evaluator truth enters this calculation.
        score = float(value[0] - 1.5 * value[1] - 0.003 * choices[name].distance)
        result[name] = {"predicted_success": float(value[0]),
                        "predicted_wrong": float(value[1]), "score": score}
    return result


def _episode(ctrl: Controller, scene, category: str, index: int, arm: str, models=None):
    state, record = _initial(ctrl, scene, _initial_offset(category, index))
    decisions = []
    while state.frame + FRAMES_PER_BLOCK < FRAME_COUNT:
        if state.s >= END - .065 and not state.uncovered():
            break
        active = _active(state)
        choices = _menu(ctrl, state)
        if not choices:
            break
        if active:
            if arm == "mpc":
                candidate = ctrl.mpc(state, choices, False, active=True, informative=True)
                predictions = {}
            else:
                predictions = _score(ctrl, state, record, choices, models,
                                     use_embedding=(arm == "embedding"))
                name = max(predictions, key=lambda key: (predictions[key]["score"],
                                                          key == "advance"))
                candidate = choices[name]
        else:
            candidate = choices.get("advance")
            predictions = {}
        if candidate is None:
            break
        decisions.append({
            "frame": int(state.frame), "active_checkpoint": active,
            "state": _checkpoint(state), "chosen": candidate.name,
            "predictions": predictions,
            "candidates": {
                name: {"s": np.asarray(option.s).tolist(),
                       "x": np.asarray(option.lateral.x).tolist(),
                       "distance": float(option.distance), "turn": float(option.turn)}
                for name, option in choices.items()
            },
        })
        ctrl.execute(scene, state, record, candidate, False)
    return _measures(ctrl, record, state), record, decisions


def _summary(episodes, arms):
    result = {"n": len(episodes), "arms": {}}
    for arm in arms:
        result["arms"][arm] = {
            key: float(np.mean([row["arms"][arm][key] for row in episodes]))
            for key in OUTCOME_KEYS
        }
    result["paired_to_mpc"] = {}
    for arm in arms:
        if arm == "mpc":
            continue
        result["paired_to_mpc"][arm] = {}
        for key in OUTCOME_KEYS:
            delta = np.asarray([row["arms"][arm][key] - row["arms"]["mpc"][key]
                                for row in episodes], dtype=float)
            result["paired_to_mpc"][arm][key] = {
                "delta": float(np.mean(delta)),
                "wins": int(np.sum(delta > 0)) if key in ("success", "coverage") else None,
                "losses": int(np.sum(delta < 0)) if key in ("success", "coverage") else None,
            }
    return result


def run(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"output directory must be empty: {out}")
    started = perf_counter()
    world = World()
    ctrl = Controller(world, _load_sections(), _load_coefficients())
    train_rows = _collect_branches(ctrl, world, "train", args.train_count, out)
    dev_rows = _collect_branches(ctrl, world, "dev", args.dev_count, out)
    base_models = _fit_models(train_rows, "features")
    embedding_models = _fit_models(train_rows, "features_embedding")
    fixed_action, fixed_scores = _fixed_action(dev_rows)

    episodes = []
    with gzip.open(out / "test_episodes.jsonl.gz", "wt", encoding="utf-8",
                   compresslevel=2) as stream:
        for index in range(int(args.test_count)):
            category = _category(index)
            scene = _case(world, "test", index, category)
            arms, traces = {}, {}
            for arm, models in (("mpc", None), ("no_embedding", base_models),
                                ("embedding", embedding_models)):
                result, record, decisions = _episode(
                    ctrl, scene, category, index, arm, models)
                arms[arm] = {key: result[key] for key in OUTCOME_KEYS}
                traces[arm] = decisions
            episode = {"index": int(index), "seed": SEED_BASE["test"] + int(index),
                       "category": category, "arms": arms}
            episodes.append(episode)
            _json(stream, {**episode, "decisions": traces})

    summary = {
        "schema": "ab_search_embedding_a_pilot_v1",
        "scope": "Synthetic registered chart world; frozen ProDMP and feedback MPC; paired Scene tape per arm.",
        "seed_bases": SEED_BASE,
        "category_counts": {
            split: dict(Counter(_category(i) for i in range(count)))
            for split, count in (("train", args.train_count), ("dev", args.dev_count),
                                 ("test", args.test_count))},
        "interventions": {
            "nominal": "no added initial offset or visibility intervention",
            "offset": "initial public chart lateral position shifted by +/-2.5 mm",
            "loss": {"target_source_clear_frames": list(LOSS_FRAMES),
                     "truth_used_only_to_modify_synthetic_clear_tape": True},
        },
        "branch_rows": {"train": len(train_rows), "dev": len(dev_rows)},
        "checkpoint_scope": "first public trigger per scene; late near-edge states are logged without a one-second branch",
        "fixed_dev_action": fixed_action, "fixed_dev_scores": fixed_scores,
        "test": _summary(episodes, ("mpc", "no_embedding", "embedding")),
        "embedding": "causal_chart_position_embedding_v1 + path visibility summaries + anonymous packet tokens",
        "posterior_calibration_warning": (
            "Inherited recovery posterior intervals have previously shown poor empirical coverage; "
            "this pilot does not interpret interval width as calibrated uncertainty."),
        "jobs_requested": int(args.jobs), "jobs_used": 1,
        "runtime_s": perf_counter() - started,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    with (out / "models.pkl").open("wb") as stream:
        pickle.dump({"no_embedding": base_models, "embedding": embedding_models}, stream, protocol=4)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--dev-count", type=int, default=32)
    parser.add_argument("--test-count", type=int, default=64)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.train_count, args.dev_count, args.test_count, args.jobs) < 1:
        parser.error("counts and jobs must be positive")
    summary = run(args)
    print(json.dumps({"out": str(Path(args.out).resolve()),
                      "train_rows": summary["branch_rows"]["train"],
                      "dev_rows": summary["branch_rows"]["dev"],
                      "test": summary["test"]}, indent=2))


if __name__ == "__main__":
    main()

