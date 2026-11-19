"""Rebuild model state, replay full MPC commands and check original parity."""

import argparse
from collections import Counter
import json
import lzma
from pathlib import Path
import sys

import numpy as np

from audit_paths import TASKS, load_bank, noise


def proposal_score(control, prior, likelihood, correct):
    probability = np.einsum("oh,h->o", likelihood, prior)
    correct_probability = np.einsum("oh,h->o", correct, prior)
    accept = (probability > 1.e-260) & (correct_probability / np.maximum(probability, 1.e-260) >= control.CONFIDENCE)
    accept[0] = False
    values = 2. * correct[accept, :].sum(axis=0) - likelihood[accept, :].sum(axis=0)
    return float(values @ prior), values


def replay(args):
    study = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(study), str(args.repo / "NMI_idea/experiments/ab_full_b_rollout_20261113")]
    import acoustics
    import control
    import run as legacy
    from world import Geometry

    settings = json.loads((args.cohort / "settings.json").read_text())
    legacy.initialize(args.repo, args.base, settings["model_size"], 6, "joint")
    bank, old = json.loads(load_bank(args.motion)), json.loads(load_bank(args.base))
    nodes = sorted(bank["nodes"], key=lambda n: n["id"])
    old_ids = {n["id"] for n in old["nodes"]}
    old_edges = {(e["start"], e["goal"]) for e in old["edges"]}
    sites = np.array([n["site"] for n in nodes])
    clocks = np.arange(7) * 1.5
    reserve = np.array([65., 90., 250., 6.])
    world = acoustics.setup_world(args.repo)
    rebuilt = {}
    for task in legacy.TASKS:
        target_sites, label = TASKS[task]
        geometry = Geometry(acoustics, world, nodes, label)
        p, c, _, _ = acoustics.model_tables(geometry, clocks, settings["model_size"], "joint")
        rebuilt[task] = geometry, p, c, np.array(target_sites)
    cases = args.cohort / "cases.jsonl"
    stream = cases.open() if cases.exists() else lzma.open(cases.with_suffix(".jsonl.xz"), "rt")
    counts = Counter()
    with stream:
        for line in stream:
            row = json.loads(line)
            if row["index"] >= args.first_cases and row["index"] not in args.extra_cases:
                continue
            task = row["task"]
            geometry, like, correct, targets = rebuilt[task]
            addresses = geometry.node_to_address
            start = bank["task_starts"][task]
            initial = np.full(settings["model_size"], 1. / settings["model_size"])
            prior, _ = control.posterior(initial, like, correct, 0, addresses[start], row["anchor"])
            assert row["state_sha256"] == legacy.encode_state(prior, prior, start, 0, 0, 0, reserve)
            counts["public_anchor_states"] += 1
            for site, records in row["proposals"]["scores"].items():
                position = list(targets).index(int(site))
                ref_site = targets[min(position + 1, 1)]
                for rec in records:
                    address = geometry.index[int(site), rec["view"]]
                    p, c = like[position + 1, address], correct[position + 1, address]
                    claim, _ = proposal_score(control, prior, p, c)
                    utilities = []
                    for view in rec["reference_views"]:
                        ref = geometry.index[int(ref_site), view]
                        _, value = proposal_score(control, prior, like[position + 2, ref], correct[position + 2, ref])
                        utilities.append(value)
                    gap, information = 0., 0.
                    if utilities:
                        values = np.asarray(utilities)
                        baseline = max(values @ prior)
                        information = sum(np.max(values @ (prior * likelihood)) for likelihood in p) - baseline
                        gap = prior @ np.max(values, axis=0) - baseline
                    np.testing.assert_allclose([claim, information, gap],
                                               [rec["immediate_claim_utility"], rec["decision_information"],
                                                rec["perfect_information_claim_gap"]], rtol=0, atol=2.e-12)
                    counts["public_proposal_scores"] += 1
            controls = {"MPC": row["proposals"]["selected"], **row.get("menu_controls", {})}
            models = {}
            for name in (*controls, "coarse", "original"):
                possible = old_ids.copy()
                if name in controls:
                    selected = controls[name]
                    possible.update(n["id"] for n in nodes if n["view"] in selected.get(str(n["site"]), []))
                costs = np.full((len(nodes), len(nodes), 4), np.inf)
                for edge in bank["edges"]:
                    a, b = edge["start"], edge["goal"]
                    if a not in possible or b not in possible or not set(sites[[a, b]]) <= {5, *targets}:
                        continue
                    if name == "original" and (a, b) not in old_edges:
                        continue
                    costs[a, b] = (edge["travel_mm"], edge["turn_deg"], edge["rail_mm"], edge["duration_s"] / 1.5)
                value, lower = control.nominal_potential(sites, costs, targets, 6, .45 if task == "daughter" else .68)
                dummy = np.zeros(len(nodes), dtype=np.int64)
                models[name] = like, correct, addresses, sites, dummy, dummy, costs, targets, value, lower, dummy, 5
            archived_data = legacy.DATA[task]
            archived_scene = archived_data["scene"]
            scenario = acoustics.make_scenario(world, row["case_seed"])
            _, _, d, c = acoustics.probabilities(archived_data["geometry"], scenario, clocks, "joint")
            d, c = np.ascontiguousarray(d[:, :, None]), np.ascontiguousarray(c[:, :, None])
            for repeat, tape_seed in enumerate(row["repeat_seeds"][:args.first_repeats]):
                tape = noise(tape_seed)
                tape[0] = noise(row["case_seed"])[0]
                old_prior, old_anchor, _ = legacy.initial_prior(archived_data, d, c, tape, "anchor")
                assert old_anchor == row["anchor"]
                np.testing.assert_array_equal(old_prior, prior)
                archived_run = control.simulate(archived_scene.public(), d, c, 0, tape, old_prior,
                                               archived_scene.start, archived_scene.reserve, control.empty_option())
                steps, old_trace = archived_run[2], archived_run[-1]
                stored = np.array(row["paths"]["original"][repeat]["trace"]).reshape(-1, 12)
                assert steps == len(stored)
                np.testing.assert_allclose(old_trace[:steps, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]], stored[:, :10],
                                           rtol=0, atol=1.e-12)
                counts["archive_original_complete_traces"] += 1
                for arm, paths in row["paths"].items():
                    selected = paths[repeat]
                    model = models.get(arm, models["MPC"])
                    belief, good, remaining = prior.copy(), prior.copy(), reserve.copy()
                    at, clock, progress = start, 0, 0
                    for slot, step in enumerate(selected["trace"]):
                        goal, obs, claim = int(step[1]), int(step[2]), bool(step[4])
                        if not step[11]:
                            assert control.mpc_action(model, belief, good, at, slot, clock, progress, remaining) == goal
                            counts["independent_mpc_commands"] += 1
                        expense = model[6][at, goal]
                        assert np.all(expense <= remaining + 1.e-8)
                        remaining -= expense
                        clock += int(expense[3])
                        address = addresses[goal]
                        posterior, confidence = control.posterior(belief, like, correct, clock, address, obs)
                        assert np.isclose(confidence, step[3], atol=1.e-12, rtol=0)
                        good, _ = control.advance_joint(belief, good, like, correct, clock, address, obs, claim)
                        belief, at, progress = posterior, goal, progress + claim
                    slot = len(selected["trace"])
                    assert control.mpc_action(model, belief, good, at, slot, clock, progress, remaining) == -1
                    assert selected["final_public_state_sha256"] == legacy.encode_state(belief, good, at, slot, clock,
                                                                                      progress, remaining)
                    counts["terminal_state_hashes"] += 1
    return dict(counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("repo", "motion", "base", "cohort"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    parser.add_argument("--first-cases", type=int, default=4)
    parser.add_argument("--first-repeats", type=int, default=4)
    parser.add_argument("--extra-cases", nargs="*", type=int, default=[])
    print(json.dumps(replay(parser.parse_args()), indent=2))
