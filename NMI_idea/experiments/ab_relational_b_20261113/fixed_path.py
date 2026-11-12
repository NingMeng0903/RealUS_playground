"""Replay the archived legacy-nearest paths through every public sensor view.

This is an identity-evidence diagnostic.  It deliberately holds the legacy
node and clock sequence fixed, so it does not claim a completion improvement
or compare a new planner.  Each interface receives its own posterior update
on the same packet tape while target-station eligibility is taken from the
archived legacy trace.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import lzma
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ARMS = ("legacy_nearest_7x7", "matched_nearest", "all_returns")
OLD, NEAREST, SET = 0, 1, 2


def _load_rows(folder: Path) -> list[dict]:
    path = folder / "cases.jsonl"
    if path.exists():
        with path.open() as stream:
            return [json.loads(line) for line in stream]
    compressed = folder / "cases.jsonl.xz"
    if not compressed.exists():
        raise FileNotFoundError(path)
    with lzma.open(compressed, "rt") as stream:
        return [json.loads(line) for line in stream]


def _fixed_trace(data, row: dict, detect: np.ndarray, cdf: np.ndarray, tape: np.ndarray,
                 bench, return_sets, control) -> tuple[dict, list[dict]]:
    scene, tables = data.scene, data.tables
    archived = row["arms"][ARMS[OLD]]["trace"]
    # All three interfaces get their own public initialization at the common
    # anchor.  The legacy trace remains the only source of motion and target
    # station eligibility.
    states = {}
    for mode, arm in enumerate(ARMS):
        states[arm], _ = bench.begin(data, detect, cdf, tape, mode)
    records: list[dict] = []
    summary = {arm: Counter() for arm in ARMS}
    previous_at = scene.start
    target_progress = 0
    for step, raw in enumerate(archived):
        start, goal, archived_slot, archived_clock = map(int, raw[:4])
        if start != previous_at or archived_slot != step:
            raise ValueError(f"legacy path is not contiguous at step {step}")
        edge = scene.edge_costs[start, goal]
        if not np.all(np.isfinite(edge)):
            raise ValueError(f"legacy path uses an unavailable edge {start}->{goal}")
        clock = int(states[ARMS[OLD]][5] + edge[3])
        if clock != archived_clock:
            raise ValueError(f"clock mismatch at step {step}: {clock} != {archived_clock}")
        address = int(scene.to_address[goal])
        packet, kinds = return_sets.draw(detect, cdf, clock, address, 0, tape,
                                         tables.mapping, tables.lookup)
        nearest_slot = int(tables.nearest_slot[packet])
        nearest_packet = int(tables.nearest_packet[packet])
        nearest_kind = int(kinds[nearest_slot]) if nearest_slot >= 0 else -1
        legacy_obs, legacy_kind = control.capture_draw(detect, cdf, clock, address, 0, tape)
        target_site = int(scene.targets[target_progress])
        target_frame = scene.site[goal] == target_site
        if int(legacy_obs) != int(raw[4]) or int(packet) != int(raw[5]):
            raise AssertionError(f"archived packet mismatch at {row['task']} index {row['index']} step {step}")
        if int(legacy_kind) != int(raw[7]) or int(nearest_kind) != int(raw[8]):
            raise AssertionError(f"archived identity-kind mismatch at {row['task']} index {row['index']} step {step}")
        if int(np.any(kinds == 0)) != int(raw[9]):
            raise AssertionError(f"archived target-presence mismatch at {row['task']} index {row['index']} step {step}")
        record = dict(index=int(row["index"]), case_seed=int(row["case_seed"]), task=row["task"],
                      step=step, start=start, goal=goal, clock=clock, address=address,
                      target_frame=bool(target_frame), target_site=target_site,
                      target_progress=target_progress, packet=int(packet),
                      bins=tables.bins[packet].tolist(), kinds=kinds.tolist(),
                      nearest_packet=nearest_packet, nearest_slot=nearest_slot,
                      nearest_kind=nearest_kind, legacy_observation=int(legacy_obs),
                      legacy_kind=int(legacy_kind), arms={})
        for mode, arm in enumerate(ARMS):
            state = states[arm]
            prior, good, reserve, at, slot, previous_clock, progress, wrong = state
            if at != start or slot != step or previous_clock + int(edge[3]) != clock:
                raise AssertionError(f"state mismatch for {arm} at {row['index']} step {step}")
            if mode == OLD:
                obs, kind, selected = int(legacy_obs), int(legacy_kind), -1
            elif mode == NEAREST:
                obs, kind, selected = nearest_packet, nearest_kind, nearest_slot
            else:
                obs, selected = int(packet), -1
                # The all-return posterior chooses a public slot after mixing
                # hypotheses.  The private kind is used only for evaluation.
                kind = -1
            compressed = scene.public() if mode == OLD else data.compressed
            target_bool = bool(target_frame)
            post, new_good, confidence, chosen, claim = bench.update(
                mode, compressed, data.all_returns, tables.bins, clock, address, obs,
                prior, good, target_bool)
            if mode == SET:
                selected = int(chosen)
                kind = int(kinds[selected]) if selected >= 0 else -1
            if mode == OLD:
                if abs(float(confidence) - float(raw[11])) > 1.e-12 or bool(claim) != bool(raw[10]):
                    raise AssertionError(f"legacy confidence/claim mismatch at {row['index']} step {step}")
            identity_correct = bool(claim and kind == 0)
            identity_wrong = bool(claim and kind != 0)
            if target_frame:
                summary[arm]["eligible_frames"] += 1
                summary[arm]["target_present_frames"] += int(np.any(kinds == 0))
                summary[arm]["correct_accepts"] += int(identity_correct)
                summary[arm]["wrong_accepts"] += int(identity_wrong)
                summary[arm]["missed_target_frames"] += int(np.any(kinds == 0) and not claim)
            record["arms"][arm] = dict(observation=int(obs), selected_slot=int(selected),
                                       selected_kind=int(kind), confidence=float(confidence),
                                       claim=bool(claim), correct_accept=identity_correct,
                                       wrong_accept=identity_wrong,
                                       target_present=bool(np.any(kinds == 0)))
            reserve = reserve - edge
            # Progress here is an interface-local state variable for the
            # posterior digest only.  Target eligibility remains fixed above.
            progress += int(claim)
            states[arm] = (post, new_good, reserve, goal, slot + 1, clock, progress,
                           bool(wrong or identity_wrong))
        all_return = record["arms"][ARMS[SET]]
        all_selected = all_return["selected_kind"]
        nearest_selected = record["arms"][ARMS[NEAREST]]["selected_kind"]
        if target_frame and all_selected == 0 and nearest_selected != 0:
            summary[ARMS[SET]]["correct_selection_where_coarse_nearest_wrong"] += 1
            summary[ARMS[SET]]["correct_accept_where_coarse_nearest_wrong"] += int(all_return["claim"])
        if target_frame and all_return["correct_accept"] and legacy_kind != 0:
            summary[ARMS[SET]]["correct_accept_where_legacy_nearest_wrong"] += 1
        records.append(record)
        previous_at = goal
        target_progress += int(raw[10])
    return {arm: dict(summary[arm]) for arm in ARMS}, records


def _bootstrap_difference(rows: list[dict], arm: str, seed: int = 57216) -> dict:
    """Descriptive case-clustered interval for correct/wrong accepted counts."""
    grouped: dict[tuple[str, int], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = row["task"], int(row["index"])
        for arm in ("legacy_nearest_7x7", arm):
            counts = row["summary"][arm]
            grouped[key][arm].append(int(counts.get("correct_accepts", 0)))
    keys = sorted(grouped)
    if not keys:
        return {}
    values = np.asarray([[grouped[key][arm][0] for arm in ("legacy_nearest_7x7", arm)] for key in keys], dtype=float)
    delta = values[:, 1] - values[:, 0]
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(delta), size=(20000, len(delta)))
    interval = np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist()
    return dict(case_n=len(keys), mean_delta=float(delta.mean()), ci95=interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--sensor", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    settings_path = args.sensor / "settings.json"
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    grid = int(settings.get("grid", 3))
    sys.path.insert(0, str(HERE))
    import bench
    import return_sets
    bank_hash, sources = bench.initialize(args.repo, args.motion, args.out,
                                          int(settings.get("model_size", 256)), grid)
    rows = _load_rows(args.sensor)
    output_rows: list[dict] = []
    for row in rows:
        task = row["task"]
        data = bench.DATA[task]
        seed = int(row["case_seed"])
        detect, cdf = bench.case(task, seed)
        tape = bench.acoustics.tape(seed, len(data.clocks))
        summary, records = _fixed_trace(data, row, detect, cdf, tape, bench, return_sets, bench.control)
        output_rows.append(dict(index=row["index"], case_seed=seed, task=task,
                                summary=summary, frames=records))
    report = {}
    for task in sorted(bench.DATA):
        task_rows = [row for row in output_rows if row["task"] == task]
        report[task] = {arm: dict(Counter({key: sum(row["summary"][arm].get(key, 0) for row in task_rows)
                                          for key in set().union(*(row["summary"][arm] for row in task_rows))}))
                        for arm in ARMS}
        for arm in (ARMS[NEAREST], ARMS[SET]):
            report[task][arm]["bootstrap_correct_vs_legacy"] = _bootstrap_difference(task_rows, arm)
    (args.out / "settings.json").write_text(json.dumps(dict(sensor=str(args.sensor), grid=grid,
        n_cases=len(rows), bank_sha256=bank_hash, sources=sources, arms=ARMS), indent=2) + "\n")
    (args.out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (args.out / "frames.jsonl").open("w") as stream:
        for row in output_rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    for task, values in report.items():
        print(task, json.dumps(values, sort_keys=True))


if __name__ == "__main__":
    main()
