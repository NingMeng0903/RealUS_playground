"""Recompute the timing preflight's public histories and surrogate scores."""

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
import lzma
from pathlib import Path
import sys

import numpy as np

from audit_paths import digest, noise, packet, seed


def witness(scene, at, progress, slots, resources, path):
    if path is None:
        return None
    left = resources.copy()
    for goal in path:
        assert progress < len(scene.targets) and slots > 0
        expense = scene.edge_costs[at, goal]
        assert np.all(expense <= left + 1.e-8)
        left -= expense
        progress += scene.site[goal] == scene.targets[progress]
        at = goal
        slots -= 1
    assert progress == len(scene.targets)
    return True


def possible(scene, at, progress, slots, left):
    if progress == len(scene.targets):
        return True
    if slots < len(scene.targets) - progress or np.any(scene.needs[slots, at, progress] > left + 1.e-8):
        return False
    for goal, cost in enumerate(scene.edge_costs[at]):
        advances = scene.site[goal] == scene.targets[progress]
        if goal == at and not advances:
            continue
        if np.all(cost <= left + 1.e-8) and possible(scene, goal, progress + advances, slots - 1, left - cost):
            return True
    return False


def audit(args):
    study = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(args.repo / "NMI_idea/experiments/ab_full_b_rollout_20261113"))
    import acoustics
    import control
    import run as legacy

    summary = json.loads((args.data / "SUMMARY.json").read_text())
    assert summary["source_sha256"] == digest((study / "temporal_preflight.py").read_bytes())
    assert summary["bank_sha256"] == digest(args.bank.read_bytes())
    count, size = summary["n"], summary["model_atoms"]
    legacy.initialize(args.repo, args.bank, size, 6, "joint")
    models = {}
    clocks = np.arange(19) * .5
    for task, source in legacy.DATA.items():
        base = source["scene"]
        costs, reserve = base.edge_costs.copy(), base.reserve.copy()
        costs[:, :, -1] *= 3
        reserve[-1] *= 3
        like, correct, detect, cdf = acoustics.model_tables(source["geometry"], clocks, size, "joint")
        np.testing.assert_array_equal(like[::3], base.likelihood)
        np.testing.assert_array_equal(correct[::3], base.correct)
        potential, needs = control.nominal_potential(base.site, costs, base.targets, 6, .45 if task == "daughter" else .68)
        models[task] = replace(base, likelihood=like, correct=correct, detect=detect, cdf=cdf,
                               edge_costs=costs, reserve=reserve, potential=potential, needs=needs, clock_unit=.5)
    path = args.data / "records.jsonl"
    stream = path.open() if path.exists() else lzma.open(path.with_suffix(".jsonl.xz"), "rt")
    verified, grouped, prefixes = Counter(), defaultdict(list), {}
    with stream:
        for line in stream:
            row = json.loads(line)
            task, index, depth = row["task"], row["index"], row["depth"]
            scene = models[task]
            public = scene.public()
            case_seed = seed(summary["namespace"], index)
            atom = np.random.default_rng(np.random.SeedSequence([case_seed, 29102])).integers(size)
            assert row["model_atom"] == atom and row["seed"] == case_seed
            tape = noise(case_seed)
            at, clock, progress = scene.start, 0, 0
            left = scene.reserve.copy()
            address = scene.to_address[at]
            obs, _ = packet(scene.detect[0, address, atom], scene.cdf[0, address, atom], tape[0], acoustics.DISTANCE)
            assert obs == row["anchor"]
            b, _ = control.posterior(np.full(size, 1. / size), scene.likelihood, scene.correct, 0, address, obs)
            good = b.copy()
            for slot, (goal, obs) in enumerate(row["history"]):
                assert control.mpc_action(public, b, good, at, slot, clock, progress, left) == goal
                cost = scene.edge_costs[at, goal]
                clock += int(cost[-1])
                assert clock % 3 == 0
                left -= cost
                address = scene.to_address[goal]
                packet_obs, _ = packet(scene.detect[clock, address, atom], scene.cdf[clock, address, atom],
                                       tape[clock // 3], acoustics.DISTANCE)
                assert packet_obs == obs
                posterior, confidence = control.posterior(b, scene.likelihood, scene.correct, clock, address, obs)
                claim = obs > 0 and confidence >= .87 and scene.site[goal] == scene.targets[progress]
                good, _ = control.advance_joint(b, good, scene.likelihood, scene.correct, clock, address, obs, claim)
                b, at, progress = posterior, goal, progress + claim
                verified["causal_packet_commands"] += 1
            used = len(row["history"])
            assert row["state_sha256"] == legacy.encode_state(b, good, at, used, clock, progress, left)
            assert row["clock_s"] == .5 * clock and row["progress"] == progress and row["slots_used"] == used
            np.testing.assert_allclose(row["reserve"], left, atol=1.e-12, rtol=0)
            if depth == 1:
                prefixes[task, index] = row["history"]
            else:
                assert row["history"][:1] == prefixes[task, index]
            values, claims = [0.] * 3, [0.] * 3
            eligible = [False] * 3
            if used == depth and progress < len(scene.targets):
                goal = control.mpc_action(public, b, good, at, used, clock, progress, left)
                assert goal == row["command"]
                if goal >= 0 and scene.site[goal] == scene.targets[progress]:
                    expense = scene.edge_costs[at, goal]
                    for hold in range(3):
                        arrival = clock + int(expense[-1]) + hold
                        remaining = left - expense - [0., 0., 0., hold]
                        advanced_possible = np.all(remaining >= -1.e-8) and possible(scene, goal, progress + 1, 5 - used, remaining)
                        if not advanced_possible:
                            assert not row["eligible"][hold]
                            values[hold], claims[hold] = values[0], claims[0]
                            continue
                        record = row["measurements"][hold]
                        assert record["witness"][0] == goal and record["arrival_s"] == .5 * arrival and arrival <= 18
                        witness(scene, at, progress, 6 - used, left - [0., 0., 0., hold], record["witness"])
                        address = scene.to_address[goal]
                        like, correct = scene.likelihood[arrival, address], scene.correct[arrival, address]
                        mass = like @ b
                        accept = (mass > 1.e-260) & (correct @ b >= .87 * mass)
                        accept[0] = False
                        c = (correct @ good)[accept].sum()
                        w = np.maximum(0., (like @ good) - (correct @ good))[accept].sum()
                        old = 0.
                        unresolved = possible(scene, goal, progress, 5 - used, remaining)
                        assert unresolved == (record["unresolved_witness"] is not None)
                        if unresolved:
                            witness(scene, goal, progress, 5 - used, remaining, record["unresolved_witness"])
                            old = scene.potential[5 - used, goal, progress]
                        witness(scene, goal, progress + 1, 5 - used, remaining, record["advanced_witness"])
                        advanced = scene.potential[5 - used, goal, progress + 1]
                        value = c * advanced + max(0., good.sum() - c - w) * old - w
                        np.testing.assert_allclose([value, c, w], [record["value"], record["correct_mass"], record["wrong_mass"]],
                                                   atol=1.e-12, rtol=0)
                        eligible[hold] = True
                        values[hold], claims[hold] = value, c - w
                        verified["checked_hold_values_and_witnesses"] += 1
            assert eligible == row["eligible"] and row["active"] == eligible[0]
            np.testing.assert_allclose(values, row["values"], atol=1.e-12, rtol=0)
            np.testing.assert_allclose(claims, row["signed_claim"], atol=1.e-12, rtol=0)
            # Compare ranks from the logged scores to preserve exact tie breaking.
            chosen = int(np.argmax(row["values"]))
            assert row["selected_hold_s"] == .5 * chosen
            grouped[task, depth].append((index, eligible[0], values, claims, chosen))
            verified["model_public_states"] += 1
    cells = {}
    for (task, depth), rows in grouped.items():
        assert sorted(r[0] for r in rows) == list(range(count))
        values = np.array([r[2] for r in rows])
        signed = np.array([r[3] for r in rows])
        archived = summary["summary"][task][str(depth)]
        gain = values.max(axis=1).mean() - values.mean(axis=0).max()
        extra = np.mean([.5 * r[4] for r in rows])
        assert sum(r[1] for r in rows) == archived["active"]
        np.testing.assert_allclose(values.mean(axis=0), archived["mean_nominal_values"], atol=1.e-12, rtol=0)
        np.testing.assert_allclose(signed.mean(axis=0), archived["mean_signed_claim_mass"], atol=1.e-12, rtol=0)
        np.testing.assert_allclose([gain, extra], [archived["contextual_nominal_minus_best_fixed"], archived["mean_extra_hold_s"]],
                                   atol=1.e-12, rtol=0)
        cells[f"{task}/{depth}"] = {"contextual_nominal_minus_best_fixed": float(gain), "mean_added_hold_s": float(extra)}
    assert len(grouped) == 4
    return {"verification": dict(verified), "cells": cells}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "bank", "data"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    print(json.dumps(audit(parser.parse_args()), indent=2))
