"""Cheap diagnostic: can a public two-block revisit improve the live-MPC tail?

This is a mechanism screen, not CEM, a learned student, or a confirmatory test.
Only the inherited two-vessel synthetic world is used. Offline truth scores
outcomes; the executed controller and stop condition use public observations.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RECOVERY = HERE.parent / "ab_recovery_training_20261115"
sys.path.insert(0, str(RECOVERY))
from controller import Controller, FRAME_COUNT, FRAMES_PER_BLOCK, KINDS, fork  # noqa: E402
from world import World  # noqa: E402

spec = importlib.util.spec_from_file_location("recovery_run_for_revisit_gate", RECOVERY / "run.py")
assert spec and spec.loader
recovery_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery_run)
KINDS["search"] = "revisit"


def run_arm(ctrl, scene, initial_state, initial_record, pattern):
    state, record = fork(initial_state, initial_record)
    if pattern != "mpc":
        n, lateral_offset = pattern
        for _ in range(n):
            if state.frame + FRAMES_PER_BLOCK >= FRAME_COUNT:
                break
            goal_s = max(2.0, state.s - 6.0)
            goal_x = float(np.clip(state.x + lateral_offset, -30.0, 0.0))
            candidate = ctrl.candidate(state, "search", goal_s, 0.0, goal_x, alias=False)
            if candidate is None:
                break
            ctrl.execute(scene, state, record, candidate, alias=False)
    ctrl.finish(scene, state, record, alias=False, allow_recover=True, informative=True)
    return recovery_run.measures(ctrl, record, state), record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=24)
    parser.add_argument("--search-replicas", type=int, default=2)
    parser.add_argument("--verify-replicas", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=800000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.cases, args.search_replicas, args.verify_replicas) <= 0:
        parser.error("all counts must be positive")
    world = World()
    with np.load(RECOVERY / "results" / "prodmp_weights.npz") as weights:
        coefficients = {key: weights[key] for key in weights.files}
    ctrl = Controller(world, recovery_run.load_sections(), coefficients)
    patterns = ("mpc", (1, 0.0), (2, 0.0), (2, -5.0), (2, 5.0))
    rows = []
    for index in range(args.cases):
        scene = world.scene(args.seed_base + index, "train", FRAME_COUNT, False)
        before, prefix = recovery_run.context(ctrl, scene)
        prefix_wrong = any(not accept["identity_ok"] for accept in prefix.accepts)
        row = {"seed": scene.seed, "checkpoint_frame": before.frame,
               "checkpoint_s": before.s, "distance_to_public_transition": before.s - world.transition_s,
               "trigger": before.trigger(), "prefix_wrong": prefix_wrong}
        if not before.trigger() or prefix_wrong:
            rows.append(row)
            continue
        search = {}
        for pattern in patterns:
            outcomes = []
            for replica in range(args.search_replicas):
                replay = world.resample_tail(scene, before.frame, 50000 + replica)
                result, _ = run_arm(ctrl, replay, before, prefix, pattern)
                outcomes.append(result)
            search[str(pattern)] = {"success": np.mean([x["success"] for x in outcomes]),
                                    "wrong": np.mean([x["wrong"] for x in outcomes]),
                                    "tcp_mm": np.mean([x["tcp_mm"] for x in outcomes])}
        # Selection is privileged: the simulator truth scores the full episode.
        # A selected pattern is not a deployable policy label.
        chosen = max(patterns, key=lambda p: (search[str(p)]["success"],
                                               -search[str(p)]["wrong"],
                                               -search[str(p)]["tcp_mm"]))
        verify = []
        for replica in range(args.verify_replicas):
            replay = world.resample_tail(scene, before.frame, 60000 + replica)
            base, _ = run_arm(ctrl, replay, before, prefix, "mpc")
            selected, record = run_arm(ctrl, replay, before, prefix, chosen)
            verify.append({"mpc": base, "revisit": selected,
                           "revisit_actions": [d["action"] for d in record.decisions
                                               if d["frame"] >= before.frame]})
        row.update({"chosen": chosen, "search": search, "verify": verify})
        rows.append(row)
    eligible = [r for r in rows if "verify" in r]
    def average(arm, key):
        return float(np.mean([v[arm][key] for row in eligible for v in row["verify"]])) if eligible else None
    summary = {"cases": args.cases, "eligible": len(eligible),
               "search_replicas": args.search_replicas, "verify_replicas": args.verify_replicas,
               "mpc_success": average("mpc", "success"),
               "revisit_success": average("revisit", "success"),
               "mpc_wrong": average("mpc", "wrong"),
               "revisit_wrong": average("revisit", "wrong"),
               "mpc_tcp_mm": average("mpc", "tcp_mm"),
               "revisit_tcp_mm": average("revisit", "tcp_mm"),
               "case_revisit_wins": sum(np.mean([v["revisit"]["success"] - v["mpc"]["success"]
                                                   for v in row["verify"]]) > 0 for row in eligible),
               "case_revisit_losses": sum(np.mean([v["revisit"]["success"] - v["mpc"]["success"]
                                                     for v in row["verify"]]) < 0 for row in eligible)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    def scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"not JSON serializable: {type(value).__name__}")
    args.out.write_text(json.dumps({"summary": summary, "cases": rows}, indent=2, default=scalar), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=scalar))


if __name__ == "__main__":
    main()
