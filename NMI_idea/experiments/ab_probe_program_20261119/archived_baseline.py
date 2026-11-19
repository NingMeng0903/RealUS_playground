"""Replicate the previously released evidence B on the current paired case streams."""

from functools import partial
import json
from pathlib import Path
from runpy import run_path

import joblib

import evaluate as e
import probe as p
from diagnose_b_selection import run_parallel, save_archive
from run_adaptive_recovery import paired_statistics
from run_learned_recovery import write_json
from summarize import METRICS


ARCHIVED = p.PREVIOUS / "results/teacher_b_training_v1/models.joblib"
MODEL_SHA = "519e08f76b583c6b0c423ecb7dbac27387df572f447c9a6388bc2e5a26758b53"
PAIRED_ARMS = ("archived_evidence", "mpc", "fixed", "old", "random", "evidence", "contrast", "program")


def evaluate_reference(row):
    c, truth, root, prefix, menu, _, anchor, _ = e.checkpoint(row)
    first = p.choose_first(c, root, prefix, menu, anchor, p.MODEL_BANK["archived_evidence"])
    results = []
    for world in p.base.scans.paired_futures(truth, row, 8, 18432):
        state, record = p.run_motion(c, root, prefix, world, menu.get(first))
        result = p.actual_outcome(c, state, record, world, root, p.deadline(root))
        results.append(result)
    return row | {"action": first, "outcomes": results}


def summarize_reference(out, split):
    truth = {row["seed"]: row["outcomes"] for row in p.base.read_rows(out / f"{split}.jsonl.gz")}
    archived = p.base.read_rows(out / f"{split}_archived.jsonl.gz")
    assert set(truth) == {row["seed"] for row in archived}
    return {scope: paired_statistics([row for row in archived if scope == "all" or row["family"] == scope],
                                     PAIRED_ARMS, METRICS, partial(paired_tapes, main=truth))
            for scope in ("all", "connection", "parallel")}


def paired_tapes(row, main):
    return [{name: (reference if name == PAIRED_ARMS[0] else main[row["seed"]][name][i])
             for name in PAIRED_ARMS} for i, reference in enumerate(row["outcomes"])]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("record", "summary"))
    parser.add_argument("--out", type=Path, default=p.HERE / "results/main")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.stage == "record":
        p.base.configure()
        if p.base.fingerprint(ARCHIVED) != MODEL_SHA:
            raise ValueError("the archived evidence model does not match its released configuration")
        p.MODEL_BANK["archived_evidence"] = joblib.load(ARCHIVED)["evidence"]
        scenarios = run_path(str(p.HERE / "run.py"))["scenarios"]
        splits = (("test", scenarios(74_700_000, 32)), ("drift", scenarios(74_800_000, 12, setting="drift")))
        manifest = json.loads((args.out / "frozen.json").read_text())
        for split, rows in splits:
            expected = manifest["seeds"]["heldout" if split == "test" else "drift"]
            assert sorted(row["seed"] for row in rows) == expected
            results = run_parallel(evaluate_reference, rows, args.workers, f"{split} released evidence B")
            save_archive(args.out / f"{split}_archived.jsonl.gz", results)
    else:
        write_json(args.out / "archived_summary.json",
                   {split: summarize_reference(args.out, split) for split in ("test", "drift")})
