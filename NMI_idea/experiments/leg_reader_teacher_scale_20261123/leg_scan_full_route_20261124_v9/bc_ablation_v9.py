"""Replay the pre-PPO sampled student as a supplementary, frozen control."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from reader_v9 import load_bundle
from run_v9 import chunks, eval_job, pack_eval, summary
from scan_v9 import initialize
from study_v9 import (DEFAULT_PACK, PROOF_TAPES, ROOT, assert_hashes, atomic_json, digest, freeze, pinned,
                      read_json, roster, verify_shard)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "run")
    parser.add_argument("--atlas-pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    out = args.out.resolve()
    protocol = read_json(out / "full_route_protocol.json")
    protocol_hash = digest(out / "full_route_protocol.json")
    primary = digest(out / "independent_freeze.json")
    weights = out / "bc_checkpoint" / "actor.json"
    assert_hashes(protocol["sources"])
    if digest(weights) != read_json(weights.parent / "manifest.json")["actor_sha256"]:
        raise ValueError("Behavior-cloning checkpoint changed")
    reader = out / "reader_bundle.json"
    if digest(reader) != protocol["reader_sha256"] or digest(args.atlas_pack) != protocol["atlas_sha256"]:
        raise ValueError("Inputs differ from the primary protocol")
    run = out / "bc_ablation"
    run.mkdir(exist_ok=True)
    locked = freeze(run, "freeze.json", {"protocol_sha256": protocol_hash, "primary_freeze_sha256": primary,
                                       "bc_actor_sha256": digest(weights), "reader_sha256": digest(reader),
                                       "source_sha256": digest(Path(__file__).resolve()), "action_rule": "sample",
                                       "arm": "BC_new", "devices": "frozen CPU rollout",
                                       "purpose": "supplementary pre-PPO policy; no primary selection change"})
    actors = (("BC_new", "new", weights, False),)
    test = list(roster("test", protocol["independent_test_layouts"]))
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"), initializer=initialize,
                             initargs=(args.atlas_pack, pinned(args.atlas_pack), load_bundle(reader))) as pool:
        _, bc_rows = chunks(pool, run / "independent", test, lambda c: (c, actors, PROOF_TAPES), eval_job, pack_eval, size=16)
    atomic_json(run / "standalone_results.json", {"freeze_sha256": locked, "scores": summary(bc_rows)})
    primary_shards = sorted((out / "independent").glob("*/commit.json"))
    if len(primary_shards) != (len(test) + 15) // 16 or not (out / "full_route_results.json").exists():
        print("BC control complete. Rerun after the primary panel to merge the paired scores.", flush=True)
        return
    main_rows = [row for p in primary_shards for row in verify_shard(p.parent)[1]]
    for row, bc_row in zip(main_rows, bc_rows, strict=True):
        if row["case"] != bc_row["case"] or row["layout_id"] != bc_row["layout_id"]:
            raise ValueError("Supplementary replay does not match the paired layout")
        for tape, arms in bc_row["outcomes"].items():
            if not arms["BC_new"]["completed_legal"] or arms["BC_new"]["layout_id"] != row["layout_id"]:
                raise ValueError("Invalid BC replay")
            row["outcomes"][tape].update(arms)
    comparisons = (("BC_new", "MPC_new"), ("student_new", "BC_new"), ("PPO_only_new", "BC_new"))
    table = summary(main_rows, comparisons)
    atomic_json(run / "paired_results.json", {"freeze_sha256": locked, "primary_freeze_sha256": primary, "scores": {
        route: {arm: scores[arm] for arm in ("BC_new", "student_new", "PPO_only_new")} for route, scores in table.items()
    }})
    print("Supplementary frozen BC control verified", len(bc_rows), flush=True)


if __name__ == "__main__":
    main()
