"""Preregister and independently confirm the original v6 dynamic target program."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import multiprocessing as mp
from pathlib import Path

from study_v7 import BASE, COHORTS, DEFAULT_PACK, ROOT, ROUTES, SEEDS, V6_RUN, atomic_json, digest, freeze, init_pinned, pinned, shards
from live_programs import SCORE_STREAMS, prefix_setup, run_programs
from run_closed import contrast, metric_section
from scale_worlds import FUTURE_SEEDS
import numpy as np


PROGRAMS = ("mpc", "equal_time_hold", "objects_no_stop")
GATE = {"strict_independent_layout_bootstrap_lower95": ">0", "wrong_branch_independent_upper95": 0.03,
        "actual_strict_net": ">=0", "contrast": "objects_no_stop versus mpc"}


def rollout(case):
    ctx = prefix_setup(case); node, tape = ctx["node"], ctx["tape"]
    outcomes = {}
    for stream, offset in zip(SCORE_STREAMS, (None, *FUTURE_SEEDS["proof"]), strict=True):
        future = tape if offset is None else tape.branch(node.tick, case.tape_seed+offset)
        if future.layout is not tape.layout:
            raise RuntimeError("Locked v6 comparison changed anatomy")
        outcomes[stream] = run_programs(ctx, future, PROGRAMS)
    return {"strict": np.array([[arms[m]["strict"] for m in PROGRAMS] for arms in outcomes.values()])}, {
        "case": asdict(case), "layout_id": tape.layout.identifier, "complete": True, "trigger": ctx["trigger"], "outcomes": outcomes}


def pack(rows):
    return {"strict": np.stack([a["strict"] for a, _ in rows])}


def summary(rows):
    routes = {}
    for route in ROUTES:
        roster = [r for r in rows if r["case"]["task"] == route]
        by_arm = {name: [{band: r["outcomes"][band][name] for band in SCORE_STREAMS} for r in roster] for name in PROGRAMS}
        arms = {name: {"scores": metric_section(entries),
                       "vs_mpc": contrast(entries, by_arm["mpc"]),
                       "vs_hold": contrast(entries, by_arm["equal_time_hold"])} for name,entries in by_arm.items()}
        c = arms["objects_no_stop"]["vs_mpc"]
        passed = (c["strict"]["independent_tape_expected"]["lower95"] > 0. and
                  c["strict"]["actual"]["delta"] >= 0. and c["wrong_branch"]["independent_tape_expected"]["upper95"] <= .03)
        routes[route] = {"layouts": len(roster), "triggered": sum(r["trigger"] for r in roster), "arms": arms,
                         "predeclared_confirmation_pass": passed}
    return routes


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "run")
    p.add_argument("--atlas-pack", type=Path, default=DEFAULT_PACK)
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args(); out, atlas = args.out.resolve(), args.atlas_pack.resolve()
    refs = pinned(atlas)
    protocol = {"experiment": ROOT.name, "base_commit": BASE, "primary": "groin_to_distal_calf/objects_no_stop",
                "comparator": "pinned_v6_shared_readout_MPC", "v6_reader_freeze": digest(V6_RUN / "reader_freeze.json"),
                "v6_closed_freeze": digest(V6_RUN / "closed_freeze.json"), "data_sha256": digest(atlas), "case_seed": SEEDS["confirmation"],
                "layout_counts_by_route": dict(zip(ROUTES, COHORTS["confirmation"], strict=True)),
                "policy_and_reader_changed": False, "programs": PROGRAMS, "evaluation_tapes": SCORE_STREAMS,
                "proof_offsets": FUTURE_SEEDS["proof"], "gate": GATE, "dird": False,
                "source": {name: digest(ROOT / name) for name in ("study_v7.py", "confirm_v6.py")}}
    sha = freeze(out, "confirmation_protocol.json", protocol)
    print("Locked v6 dynamic no-stop program is the preregistered whole-route candidate", flush=True)
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=init_pinned, initargs=(atlas, refs)) as pool:
        rows = shards(pool, out, "confirmation", rollout, pack)
    report = summary(rows)
    atomic_json(out / "confirmation_results.json", {"frozen": sha, "routes": report, "actual_scoring_executions":
        sum(len({a["program"] for a in band.values()}) for r in rows for band in r["outcomes"].values())})
    for name, result in report.items():
        print(name, {m: [e["scores"]["strict"], e["scores"]["wrong_branch"]] for m,e in result["arms"].items()},
              result["predeclared_confirmation_pass"], flush=True)


if __name__ == "__main__":
    main()
