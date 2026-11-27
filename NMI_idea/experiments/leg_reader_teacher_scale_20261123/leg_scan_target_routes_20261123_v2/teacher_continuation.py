"""Equal-time review and topology-anchor tests with one frozen public continuation."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import multiprocessing as mp
from pathlib import Path

import numpy as np

from run_target_pilot import (ROOT, BASE, FEATURES, shape_public_reader, geometric_opportunity,
                              extract, motion, spec_for)
from route_rollouts import initialize
from environment import Arm
from route_metrics import paired, outcomes, totals
from storage import atomic_json, digest, read_json


HOLDOUT_SEED_OFFSET = 971_207_812_811
SEARCH_TAPES = 2
PROOF_TAPES = 2
FIXED_ARM = Arm("teacher_public_continuation", dird=False, stop=False)


def load_frozen_reader(path, expected):
    import torch
    from reader_fit import Readout

    if digest(path) != expected:
        raise ValueError("Public reader weight hash differs from its freeze")
    weights = torch.load(path, map_location="cpu", weights_only=False)
    if tuple(weights["feature_order"]) != FEATURES:
        raise ValueError("The frozen public feature contract has changed")
    model = Readout(len(FEATURES)).eval()
    model.load_state_dict(weights["model"])
    return dict(weights, model=model)


def complete_public(env, node, spec, tape, limits):
    while node.station < spec[2]:
        env.route.advance(node, spec, tape, limits)
    if node.station != spec[2] or node.tick > limits.ticks or node.distance > limits.mm + 1.e-6:
        raise RuntimeError("The common continuation did not respect its terminal envelope")
    return node


def start_case(pilot, case):
    env = pilot._environment(case.shape)
    spec = spec_for(env.scanner, case)
    node, trial, limits = env.start(case, spec)
    target_neutral = -750. if case.task == "femoral" else -910.
    index = int(np.argmin(abs(env.scanner.stations - target_neutral)))
    at = min(spec[2] - 1, index + 1)
    while node.station < at:
        env.route.advance(node, spec, trial, limits)
    return env, spec, node, trial, limits, index


def menu(env, spec, start, limits, anchor):
    plans = {"continue": None, "hold": None}
    caps = limits.probe(start, spec)
    delta = (start.station - anchor) / motion.AXIAL_RANGE
    if delta > 0:
        for turn in (-.80, -.40, 0., .40, .80):
            controls = np.array([turn, np.clip(delta, 0., 1.), .25 * turn])
            path = env.plan(start, spec, 2, controls, caps, FIXED_ARM)
            if path is not None and abs(path.goal[0] - anchor) <= .35:
                plans[f"anchor_{turn:+.2f}"] = path
    slot = max((p.ticks for p in plans.values() if p is not None), default=min(caps[0], 8))
    return plans, slot, caps


def review(env, spec, prefix, tape, limits, name, plan, slot, caps):
    node = prefix.fork()
    if plan is not None:
        result = env.material.run_probe(node, spec, tape, plan, *caps, feedback=True,
                                        use_dird=False, copy_node=False)
        if not result.legal:
            raise RuntimeError("A physically admitted anchor program failed in execution")
        node = result.node
    if name != "continue":
        until = prefix.tick + slot
        if node.tick > until:
            raise RuntimeError("Active and fixed review time differ")
        while node.tick < until:
            node.tick += 1
            env.scanner._observe(node, spec, tape, node.station, node.sector, node.tick)
        if node.tick != until or node.station != prefix.station or node.sector != prefix.sector:
            raise RuntimeError("Review did not return to the same legal continuation boundary")
    return node


def read(env, spec, n, model, threshold, *, locked_count=None, locked_region=None):
    from reader_fit import score

    public = shape_public_reader(env, spec, n.reader.frames, n.reader.mixed_posterior)
    e = extract(public, spec, env.material)
    values, mask = e["extended"], e["extended_mask"]
    vector = np.c_[values[:, :BASE], mask[:, :BASE], values[:, BASE:], mask[:, BASE:]].astype(np.float32)
    confidence = score(vector, model)
    frames, slots = e["frame_index"], e["slot"]
    labels = np.asarray([n.reader.frames[i][3][j] for i, j in zip(frames, slots)])
    region = e["region"]
    r = outcomes({"label": labels, "region": region, "complete": n.station == spec[2]},
                 confidence, threshold)
    if locked_count is not None:
        ix = np.flatnonzero((frames < locked_count) & (region == locked_region))
        chosen = ix[np.argmax(confidence[ix])] if len(ix) else -1
        r["old_image"] = (int(labels[chosen]) if chosen >= 0 and confidence[chosen] >= threshold else -99)
    best = {}
    for i, value in enumerate(confidence):
        if value < threshold:
            continue
        station = int(round(n.reader.frames[frames[i]][0]))
        if station not in best or value > best[station][0]:
            best[station] = (float(value), int(labels[i]))
    axis = env.scanner.surface.y[spec[1]:spec[2] + 1]
    width = abs(np.diff(np.r_[axis[0], (axis[:-1] + axis[1:]) / 2., axis[-1]]))
    r["correct_mm"] = float(sum(w for k, w in enumerate(width, spec[1]) if best.get(k, (0., -99))[1] == 0))
    return r


def utility(final):
    return (3. * final["strict"] + .38 * final["correct_regions"] + .002 * final["correct_mm"]
            - .66 * final["wrong_region_count"] - .22 * final["wrong_branch_regions"]
            + .26 * (final["old_image"] == 0) - .32 * (final["old_image"] not in (-99, 0)))


_PILOT = None
_READER = None
_THRESHOLDS = None


def initialize_worker(path, sha, thresholds):
    global _PILOT, _READER, _THRESHOLDS
    _PILOT = initialize()
    _READER = load_frozen_reader(path, sha)
    _THRESHOLDS = thresholds


def run_one(case):
    env, spec, node, trial, limits, anchor = start_case(_PILOT, case)
    plans, slot, caps = menu(env, spec, node, limits, anchor)
    threshold = _THRESHOLDS[case.task]["full_public_embedding"]
    count = len(node.reader.frames)
    old_region = motion.region_of(spec, node.station)
    old = read(env, spec, node, _READER, threshold, locked_count=count, locked_region=old_region)["old_image"]
    public_state = env.features(node, spec, limits, caps, FIXED_ARM)
    layout_id = trial.layout.identifier

    def branch(name, tape):
        if tape.layout.identifier != layout_id:
            raise RuntimeError("Independent future changed the latent anatomy")
        current = review(env, spec, node, tape, limits, name, plans[name], slot, caps)
        local = read(env, spec, current, _READER, threshold, locked_count=count, locked_region=old_region)
        walked = complete_public(env, current, spec, tape, limits)
        end = read(env, spec, walked, _READER, threshold, locked_count=count, locked_region=old_region)
        return end | {"old_before": old, "old_after_review": local["old_image"], "clock_ticks": walked.tick,
                      "distance_mm": walked.distance, "utility": utility(end)}

    common = [trial.branch(node.tick, case.tape_seed + 21_497_687 + i) for i in range(SEARCH_TAPES)]
    comparison = {name: [branch(name, future) for future in common] for name in plans}
    # Continue is measured separately. The review action is compared at equal duration.
    selection = max((name for name in plans if name != "continue"),
                    key=lambda name: (np.mean([q["utility"] for q in comparison[name]]), name == "hold"))
    tested = {name: [branch(name, trial)] for name in plans}
    for name in plans:
        for i in range(PROOF_TAPES):
            future = trial.branch(node.tick, case.tape_seed + 78_264_971 + i)
            tested[name].append(branch(name, future))
    return {"id": case.identifier, "task": case.task, "profile": case.profile, "source": case.source,
            "layout_id": layout_id, "public_prefix": public_state, "selected": selection,
            "anchor_station": anchor, "prefix_station": int(node.station), "matched_review_clock": slot,
            "selection_tapes": comparison, "holdout_tapes": tested}


def holdout(case):
    return replace(case, identifier=case.identifier.replace("evaluation", "teacher_proof"),
                   split="teacher_proof", world_seed=case.world_seed + HOLDOUT_SEED_OFFSET,
                   tape_seed=case.tape_seed + HOLDOUT_SEED_OFFSET, action_seed=case.action_seed + HOLDOUT_SEED_OFFSET)


def nearest_pairs(rows):
    features = np.stack([r["public_prefix"] for r in rows])
    scale = np.maximum(features.std(axis=0), .08)
    found = []
    for i, row in enumerate(rows):
        other = [j for j, r in enumerate(rows) if r["layout_id"] != row["layout_id"]
                 and r["source"] == row["source"] and r["profile"] == row["profile"]]
        if not other:
            continue
        j = min(other, key=lambda j: np.linalg.norm((features[i] - features[j]) / scale))
        r = rows[j]
        shared = row["holdout_tapes"].keys() & r["holdout_tapes"].keys()
        signs = {}
        for action in sorted(shared - {"hold", "continue"}):
            delta = []
            for world in (row, r):
                condition = world["holdout_tapes"]
                effect = [a["utility"] - b["utility"] for a, b in zip(condition[action], condition["hold"], strict=True)]
                delta.append(float(np.mean(effect)))
            signs[action] = delta
        found.append({"ids": [row["id"], r["id"]], "different_layouts": True,
                      "normalized_distance": float(np.linalg.norm((features[i] - features[j]) / scale) / np.sqrt(features.shape[1])),
                      "anchor_differences_vs_hold": signs})
    return found


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=ROOT / "fixed_pilot")
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()
    file = args.panel / "reader_freeze.json"
    freeze = read_json(file)
    routes = tuple(freeze["approved_routes_for_teacher"])
    if not routes:
        print("Same-track entry condition was not met on either route.", flush=True)
        return
    reader = freeze["readers"]["full_public_embedding"]
    pilot = initialize()
    cases = [holdout(c) for task in routes for c in pilot.case_plan("evaluation", task)]
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=initialize_worker,
                             initargs=(reader["weights"], reader["sha256"], freeze["thresholds_development"])) as pool:
        rows = list(pool.map(run_one, cases))
    report = {}
    for task in routes:
        data = [r for r in rows if r["task"] == task]
        selected = [r["holdout_tapes"][r["selected"]][0] for r in data]
        baseline = [r["holdout_tapes"]["hold"][0] for r in data]
        reference = [r["holdout_tapes"]["continue"][0] for r in data]
        cluster = [r["layout_id"] for r in data]
        corrected = lambda a, b: int(a == 0 and b != 0)
        report[task] = {
            "counts": {"teacher": totals(selected), "hold": totals(baseline), "continue": totals(reference)},
            "paired_vs_equal_time_hold": paired(selected, baseline, cluster, 112798),
            "paired_vs_continue": paired(selected, reference, cluster, 112470),
            "anchor_executions": sum(r["selected"].startswith("anchor_") for r in data),
            "old_image_after_review": {
                "rescued": sum(corrected(a["old_after_review"], b["old_after_review"])
                               for a, b in zip(selected, baseline, strict=True)),
                "lost": sum(corrected(b["old_after_review"], a["old_after_review"])
                            for a, b in zip(selected, baseline, strict=True)),
            },
            "proof_rescues_on_both_tapes": sum(
                all(a["strict"] and not b["strict"] for a, b in zip(
                    r["holdout_tapes"][r["selected"]][1:], r["holdout_tapes"]["hold"][1:], strict=True))
                for r in data),
            "proof_losses_on_both_tapes": sum(
                all(b["strict"] and not a["strict"] for a, b in zip(
                    r["holdout_tapes"][r["selected"]][1:], r["holdout_tapes"]["hold"][1:], strict=True))
                for r in data),
            "mean_time_s": {name: float(np.mean([v["clock_ticks"] for v in panel]) * motion.CLOCK_S)
                            for name, panel in (("teacher", selected), ("hold", baseline), ("continue", reference))},
            "mean_travel_mm": {name: float(np.mean([v["distance_mm"] for v in panel]))
                               for name, panel in (("teacher", selected), ("hold", baseline), ("continue", reference))},
            "public_history_nearest_pairs": nearest_pairs(data), "rows": data,
        }
    result = {"frozen_readout": digest(file), "stage": "privileged action review",
              "same_continuation": "environment.Route.advance on public state for every planned and executed suffix",
              "selection_draws": SEARCH_TAPES, "independent_proof_draws": PROOF_TAPES, "routes": report}
    atomic_json(args.panel / "teacher_probe_audit.json", result)
    for task, r in report.items():
        p = r["paired_vs_equal_time_hold"]["strict"]
        print(task, r["counts"]["teacher"]["strict"], r["counts"]["hold"]["strict"],
              p["rescued_or_added"], p["lost_or_removed"], flush=True)


if __name__ == "__main__":
    main()
