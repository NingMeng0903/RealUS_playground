"""Describe observation-conditioned choices at equal measured first-probe exits."""

from itertools import combinations
from pathlib import Path

import numpy as np

import evaluate as e
import probe as p
from run_learned_recovery import write_json


EXIT = ("s_after_first", "x_after_first", "vs_after_first", "vx_after_first",
        "remaining_mm_after_first", "first_exit_frame")


def compare_exits(rows):
    comparable, switches = 0, 0
    matched_roots, switching_roots = set(), set()
    examples = {}
    for row in rows:
        trials = row["outcomes"]["program"]
        for (i, a), (j, b) in combinations(enumerate(trials), 2):
            if a["first"] == p.HANDOFF or a["first"] != b["first"]:
                continue
            left, right = ([trial[k] for k in EXIT] for trial in (a, b))
            if not np.allclose(left, right, rtol=0., atol=1.e-7):
                continue
            matched_roots.add(row["seed"])
            comparable += 1
            if a["requested_second"] == b["requested_second"]:
                continue
            switching_roots.add(row["seed"])
            switches += 1
            examples.setdefault(f'{row["family"]}/{row["context"]}', {
                "seed": row["seed"], "tapes": [i, j], "first": a["first"], "exit": left,
                "second": [t["requested_second"] for t in (a, b)],
                "empty_frames": [t["empty_after_first"] for t in (a, b)],
                "posterior_tv": [t["posterior_tv_after_first"] for t in (a, b)],
                "completion": [int(t[e.PRIMARY]) for t in (a, b)],
            })
    return {"tolerance": 1.e-7, "matched_roots": len(matched_roots),
            "roots_with_switch": len(switching_roots), "matched_tape_pairs": comparable,
            "pairs_with_switch": switches, "examples": examples}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=p.HERE / "results/main")
    args = parser.parse_args()
    write_json(args.out / "branch_diagnostics.json",
               {split: compare_exits(p.base.read_rows(args.out / f"{split}.jsonl.gz"))
                for split in ("test", "drift")})
