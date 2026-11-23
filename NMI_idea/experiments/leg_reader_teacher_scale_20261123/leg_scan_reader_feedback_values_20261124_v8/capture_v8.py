"""Acquire fixed MPC tracks and matched replay prefixes before reader selection."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

from study_v8 import BASE, COHORTS, DEFAULT_PACK, ROOT, SEEDS, check_hashes, digest, freeze, hashes, shards
from study_data import current_scan, init_worlds, pack_acquired


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"run")
    parser.add_argument("--atlas-pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    source = hashes(("study_v8.py", "capture_v8.py"))
    rule = {"base_git": BASE, "sources": source, "atlas_sha256": digest(args.atlas_pack),
            "cohort": COHORTS["reader_test"], "case_seed": SEEDS["reader_test"],
            "fixed_actor": "common MPC; auxiliary before/after_hold/after_probe histories", "dird": False}
    freeze(args.out, "capture_protocol.json", rule)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"), initializer=init_worlds,
                             initargs=(args.atlas_pack,)) as pool:
        shards(pool, args.out, "reader_test", current_scan, pack_acquired)
    check_hashes(source)


if __name__ == "__main__":
    main()
