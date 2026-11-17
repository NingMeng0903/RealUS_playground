"""Replay the realized first motion while changing the evidence available to the controller."""

import argparse
from collections import Counter
from copy import copy
from dataclasses import replace
import gzip
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from adaptive_controller import FRAME_STRIDE, FRAMES_PER_BLOCK
from run_adaptive_recovery import (WORKERS, init_worker, paired_statistics, score_outcome,
                                   shared_prefix, tape_for)


ARMS = ("mpc", "b")
MODES = ("full", "no_packets", "lock_structure")
FAMILIES = ("connection", "parallel")


def observer_with_fixed_structure(controller, state):
    """Permit within-hypothesis updates while keeping the mass of each structure fixed."""
    observer = copy(controller)
    observer.world = copy(controller.world)
    structure = observer.world.hypotheses[:, 0]
    groups = [structure == i for i in np.unique(structure)]
    base_likelihood = observer.world.likelihood

    def likelihood(*args):
        values, responses = base_likelihood(*args)
        for mask in groups:
            weights = state.belief[mask]
            mass = weights.sum()
            if mass > 0.:
                scale = (weights / mass) @ values[mask]
                if scale > 0.:
                    values[mask] /= scale
        return values, responses

    observer.world.likelihood = likelihood
    return observer, groups


def evidence_replay(controller, root, prefix, tape, walked, block_end, mode):
    state = replace(root, belief=root.belief.copy())
    record = prefix.copy()
    record.detailed = False
    path = walked.motion[len(prefix.motion) - 1:]
    record.motion.extend(path[1:])
    record.decisions.append(dict(walked.decisions[len(prefix.decisions)]))
    if mode == "lock_structure":
        observer, groups = observer_with_fixed_structure(controller, state)
    for exposure, pose in enumerate(path[FRAME_STRIDE::FRAME_STRIDE], 1):
        state.s, state.x, state.vs, state.vx, state.angle = pose[1:]
        state.frame = root.frame + exposure
        if mode == "lock_structure":
            observer.observe(tape, state, record)
        else:
            # The exposure consumes time. Its packet is withheld entirely,
            # so the filter does not treat the blackout as a negative image.
            state.slot = -1
            state.residual = 0.
            state.seen = state.ambiguous = 0
            state.missed += 1
            record.packets.append({"frame": state.frame, "s": state.s, "x": state.x, "q": state.angle,
                                   "y": [], "association": [], "accepted_slot": -1, "roles_truth": [],
                                   "target_truth": [0., 0.]})
    assert state.frame == root.frame + FRAMES_PER_BLOCK == block_end.frame
    assert (state.s, state.x, state.vs, state.vx) == (block_end.s, block_end.x, block_end.vs, block_end.vx)
    if mode == "lock_structure":
        for mask in groups:
            np.testing.assert_allclose(state.belief[mask].sum(), root.belief[mask].sum(), atol=2e-12)
    else:
        np.testing.assert_array_equal(state.belief, root.belief)
    state.travel = block_end.travel
    controller.tail(tape, state, record, open_blocks=1)
    return float(score_outcome(state, record, root)["corrected_complete"])


def run_case(job):
    study, tape_id, public_arm, row = job
    family, seed = row["family"], row["seed"]
    controller = WORKERS[family]
    scene = controller.world.draw(seed)
    root, prefix = shared_prefix(controller, scene)
    assert root.frame == row["checkpoint"]["frame"]
    np.testing.assert_allclose(root.belief, row["public_root"]["belief"], atol=1e-12, rtol=0.)
    if study == "primary":
        actions = (row["first_decision"]["mpc"], row["first_decision"]["picked"])
    else:
        actions = (row["selected"]["mpc_shared_menu"], row["selected"][public_arm])
    original_arms = ("mpc_shared_menu", public_arm)
    menu = controller.options_shared(root)
    structure = controller.world.hypotheses[:, 0]
    correct = structure == structure[scene.latent]
    trials = []
    for trial in row["verification"]:
        tape = tape_for(scene.latent, seed, trial["replica"], tape_id)
        scores = {}
        for arm, action in zip(original_arms, actions):
            if action in scores:
                continue
            state = replace(root, belief=root.belief.copy())
            record = prefix.copy()
            record.detailed = False
            controller.execute(tape, state, record, menu[action])
            p_correct = float(state.belief[correct].sum())
            values = {mode: evidence_replay(controller, root, prefix, tape, record, state, mode)
                      for mode in ("no_packets", "lock_structure")}
            controller.tail(tape, state, record, open_blocks=1)
            full = score_outcome(state, record, root)
            fields = ("corrected_complete", "tcp_mm", "seconds", "final_wrong")
            archived = trial["outcomes"][arm]
            np.testing.assert_allclose([full[k] for k in fields], [archived[k] for k in fields], atol=1e-12, rtol=0.)
            scores[action] = dict(values, full=float(full["corrected_complete"]), p_correct=p_correct)
        result = dict(zip(ARMS, (scores[actions[0]], scores[actions[1]])))
        for arm, original in zip(ARMS, original_arms):
            assert result[arm]["full"] == trial["outcomes"][original]["corrected_complete"]
        trials.append(result)
    return {"study": study, "family": family, "seed": seed, "actions": dict(zip(ARMS, actions)),
            "checkpoint": row["checkpoint"]["category"], "root_p_correct_structure": float(root.belief[correct].sum()),
            "trials": trials}


def summarize(rows):
    stats = paired_statistics(rows, ARMS, MODES, lambda row: row["trials"])
    replay_count = sum(len(row["trials"]) for row in rows)
    stats["counts"] = {mode: {arm: sum(trial[arm][mode] for row in rows for trial in row["trials"])
                              for arm in ARMS} for mode in MODES}
    stats["trial_count"] = replay_count
    stats["checkpoint_counts"] = dict(Counter(row["checkpoint"] for row in rows))
    ambiguous = [row for row in rows if .05 < row["root_p_correct_structure"] < .95]
    stats["ambiguous_at_root"] = {"worlds": len(ambiguous), "trials": sum(len(row["trials"]) for row in ambiguous),
                                 "above_095_after_first": {
                                     arm: sum(trial[arm]["p_correct"] > .95 for row in ambiguous for trial in row["trials"])
                                     for arm in ARMS}}
    interaction = np.asarray([np.mean([(trial["b"]["full"] - trial["mpc"]["full"])
                                       - (trial["b"]["no_packets"] - trial["mpc"]["no_packets"])
                                       for trial in row["trials"]]) for row in rows])
    indices = np.random.default_rng(4710).integers(len(rows), size=(10_000, len(rows)))
    stats["loss_of_B_gain_without_packets"] = {"mean": float(interaction.mean()),
                                               "ci95": np.quantile(interaction[indices].mean(axis=1), [.025, .975]).tolist()}
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--limited", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--jobs", default=8, type=int)
    args = parser.parse_args()
    jobs = []
    for study, filename, tape_id, arm in (("primary", args.primary, 1101, "public_first_only"),
                                          ("limited", args.limited, 1117, "public_budgeted")):
        with gzip.open(filename, "rt") as stream:
            jobs.extend((study, tape_id, arm, row) for row in map(json.loads, stream) if row["family"] in FAMILIES)
    init_worker(None)
    with get_context("fork").Pool(args.jobs) as pool:
        rows = list(pool.imap(run_case, jobs))
    args.out.mkdir(parents=True, exist_ok=True)
    report = {study: {family: summarize([r for r in rows if r["study"] == study
                                        and (r["family"] == family or family == "pooled")])
                      for family in (*FAMILIES, "pooled")} for study in ("primary", "limited")}
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with gzip.open(args.out / "cases.jsonl.gz", "wt", compresslevel=3) as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
