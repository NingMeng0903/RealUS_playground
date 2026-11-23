"""Separate winner changes from submission decisions on the frozen reader panel."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from study_data import DEFAULT_PACK, ROOT, atomic_json, digest, read_json
from fit_public import load_public
from reader_panel import histories, public_variants
from readout_state import candidates, confidence, evaluate
from route_metrics import totals


def analyze(out, atlas):
    freeze_path = out / "reader_freeze.json"
    freeze = read_json(freeze_path)
    previous = read_json(out / "reader_results.json")
    if digest(freeze_path) != previous["freeze_sha256"]:
        raise RuntimeError("Reader test freeze does not match the diagnostic inputs")
    if digest(freeze["fit"]["weights"]) != freeze["fit"]["sha256"]:
        raise RuntimeError("Reader fit weights differ from the frozen reader")
    model = load_public(freeze["fit"]["weights"], "cuda:0")
    with np.load(atlas, allow_pickle=False) as z:
        links = z["links"]
    panels = defaultdict(lambda: defaultdict(list))
    flips = defaultdict(Counter)
    paths = sorted((out / "reader_test").glob("*/data.npz"))
    for meta, _, public, posterior, embedding, labels, identity in histories(paths, model, ("terminal",)):
        task = meta["case"]["task"]
        variants = public_variants(public, posterior, embedding, links, freeze["edge_models"], identity)
        raw_winners, _ = candidates(public, variants["unary"])
        count = evaluate(public, labels, confidence(public, variants["unary"], freeze["risk"][task]["unary"]),
                         freeze["thresholds"][task]["methods"]["unary"]["threshold"], complete=True)["submitted"]
        for name, state in variants.items():
            raw = confidence(public, state, None)
            risk = confidence(public, state, freeze["risk"][task][name])
            fused_winners, _ = candidates(public, state)
            harmed = (labels[raw_winners] == 0) & (labels[fused_winners] != 0)
            previous_winner, new_winner = raw_winners[harmed], fused_winners[harmed]
            unchanged_rival = np.isclose(posterior[new_winner, 0], state["probabilities"][new_winner, 0], atol=1.e-6)
            target_decreased = state["probabilities"][previous_winner, 0] < posterior[previous_winner, 0] - 1.e-6
            artifact = labels[new_winner] < 0
            counts = flips[task, name]
            counts.update({"harmful_flips": int(harmed.sum()), "target_score_decreased": int(target_decreased.sum()),
                           "new_winner_is_artifact": int(artifact.sum()),
                           "target_decreased_and_new_artifact_unchanged": int((target_decreased & artifact & unchanged_rival).sum())})
            panels[task][name].append({"all_winners": evaluate(public, labels, raw, 0., complete=True),
                                       "raw_matched": evaluate(public, labels, raw, 0., complete=True, submit_count=count),
                                       "risk_matched": evaluate(public, labels, risk, 0., complete=True, submit_count=count)})
    results = {}
    for task, methods in panels.items():
        results[task] = {}
        unary = methods["unary"]
        for name, items in methods.items():
            metrics = {k: totals(row[k] for row in items) for k in ("all_winners", "raw_matched", "risk_matched")}
            fused = np.asarray([r["all_winners"]["region_outcomes"] for r in items])
            baseline = np.asarray([r["all_winners"]["region_outcomes"] for r in unary])
            valid = np.asarray([r["all_winners"]["capture"] for r in items])
            correct_to_wrong = valid & (baseline == 1) & (fused == 0)
            wrong_to_correct = valid & (baseline == 0) & (fused == 1)
            metrics["winner_flips_in_regions_with_true_echo"] = {
                "correct_to_wrong_by_region": correct_to_wrong.sum(axis=0).tolist(),
                "wrong_to_correct_by_region": wrong_to_correct.sum(axis=0).tolist(),
            } | dict(flips[task, name])
            prior = previous["routes"][task]["terminal"]["matched"][name]
            if any(metrics["risk_matched"][key] != prior[key] for key in metrics["risk_matched"]):
                raise RuntimeError("Frozen same-track risk result did not reproduce")
            results[task][name] = metrics
    return {"reader_freeze_sha256": digest(freeze_path), "diagnostic_only": True, "routes": results}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "run")
    parser.add_argument("--atlas-pack", type=Path, default=DEFAULT_PACK)
    args = parser.parse_args()
    results = analyze(args.out.resolve(), args.atlas_pack.resolve())
    atomic_json(args.out / "fusion_diagnostic.json", results)
    for task, methods in results["routes"].items():
        print(task, {method: [row[k]["wrong_region_count"] for k in ("all_winners", "raw_matched", "risk_matched")]
                     for method, row in methods.items()}, flush=True)


if __name__ == "__main__":
    main()
