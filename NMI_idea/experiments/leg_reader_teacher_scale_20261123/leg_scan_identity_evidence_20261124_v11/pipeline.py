"""Freeze models, teacher data, controls and the independent identity panel."""

from collections import Counter
from pathlib import Path

import numpy as np

import run_v11 as run
from evidence import ROOT, V10
from evidence_model import fit as fit_evidence
from mixed_reader import fit as fit_reader
from recovery_task import ROUTES
from study_v9 import atomic_json, digest, freeze, read_json


def errors(records):
    return {name: int(sum(r["selected_ids"].count(identity) for r in records))
            for name, identity in (("artifact", -1), ("mixed", -2), ("reject", -99))}


def paired_delta(rows, arm, other, key):
    values = np.array([[float(t["arms"][arm][key]) - float(t["arms"][other][key]) for t in row["tapes"].values()]
                       for row in rows])
    rng = np.random.default_rng(701_542)

    def interval(x):
        sampled = x[rng.integers(0, len(x), (2500, len(x)))].mean(axis=1)
        return [float(x.mean()), *np.quantile(sampled, (.025, .975)).tolist()]

    estimate = {"actual": interval(values[:, 0]), "rescued": int((values[:, 0] > 0).sum()),
                "lost": int((values[:, 0] < 0).sum())}
    if values.shape[1] > 1:
        estimate["independent"] = interval(values[:, 1:].mean(axis=1))
    return estimate


def measure(rows):
    summary = {}
    for route in ROUTES:
        relevant = [r for r in rows if r["case"]["task"] == route]
        actual = [r["tapes"]["actual"] for r in relevant]
        arms = actual[0]["arms"]
        counts = Counter()
        for row in actual:
            executed = row["executed"]
            counts.update([k for k in ("repeat", "geometry", "information", "task") if k in executed])
            active = [executed[k] for k in ("repeat", "geometry", "information", "task") if k in executed]
            if active:
                if len(active) != 4 or len({a["ticks"] for a in active}) != 1:
                    raise RuntimeError("The active controls did not share elapsed time")
                counts["target_region_" + str(executed["target_region"])] += 1
                counts["different_information_and_geometry"] += executed["information"]["view"] != executed["geometry"]["view"]
                counts["task_revisited_other_window"] += executed["task"]["view"] != "repeat"
            if "memory" in executed:
                change = executed["memory"]
                counts["old_image_promoted"] += change["changed"]
                if change["changed"]:
                    counts["a_wrong_b_correct"] += change["anchors"]["anchor_labels"][0] != 0 and change["anchors"]["anchor_labels"][1] == 0
        tab = {}
        for arm in arms:
            s = [r["arms"][arm] for r in actual]
            ids = np.stack([r["selected_ids"] for r in s])
            tab[arm] = {"strict": sum(r["strict"] for r in s), "wrong_branch": sum(r["wrong_branch"] for r in s),
                        "named_wrong_regions": int((ids > 0).sum()), **errors(s),
                        "correct_by_region": (ids == 0).sum(0).tolist(),
                        "visible_by_region": np.sum([r["capture"] for r in s], axis=0).tolist(),
                        "strict_per_tape": [sum(r["tapes"][t]["arms"][arm]["strict"] for r in relevant) for t in relevant[0]["tapes"]],
                        "mean_s": float(np.mean([r["time_s"] for r in s])), "mean_mm": float(np.mean([r["travel_mm"] for r in s]))}
            tab[arm]["paired"] = {ref: {m: paired_delta(relevant, arm, ref, m) for m in ("strict", "wrong_branch")}
                                  for ref in ("v10", "base", "repeat", "geometry", "task") if arm != ref}
        summary[route] = {"layouts": len(relevant), "triggered": dict(counts), "arms": tab}
    return summary


def teacher_panel(rows):
    counts, regions = Counter(), {}
    for row in rows:
        for q in row["queries"]:
            win, rep = q["proof_teacher"], q["proof_repeat"]
            old = q["before"]
            counts.update(prefixes=1, high_confidence=bool(q["closed_high"] and q["anchor_labels"][0] != 0),
                          true_a=q["hypothesis"] == 0, true_b=q["hypothesis"] == 1, neither=q["hypothesis"] == 2,
                          alias=q["same_true_target"], proof_nonrepeat=q["chosen"] != "repeat")
            for program in q["search"]:
                for branch in program:
                    gain = run.teacher_utility(branch["delta"])
                    counts["branches"] += 1
                    counts["positive" if gain > 0 else "negative" if gain < 0 else "neutral"] += 1
            if q["proof_event"]["ticks"] != q["repeat_event"]["ticks"]:
                raise RuntimeError("Teacher proof did not use equal-time controls")
            r = regions.setdefault(q["region"], Counter())
            j = q["region"] - 1
            r.update(prefixes=1, chosen_strict=win["strict"], repeat_strict=rep["strict"],
                     initial_strict=old["strict"], chosen_wrong=win["wrong_branch"], repeat_wrong=rep["wrong_branch"],
                     rescue_vs_repeat=win["strict"] and not rep["strict"], loss_vs_repeat=rep["strict"] and not win["strict"],
                     correct_target_vs_repeat=int(win["selected_ids"][j] == 0) - int(rep["selected_ids"][j] == 0))
    return {"totals": counts, "regions": regions, "layouts": len({r["layout_id"] for r in rows if r["queries"]})}


def finish(args, out, actor, mined):
    sources = {name: digest(ROOT / name) for name in ("run_v11.py", "evidence.py", "mixed_reader.py",
                                                      "evidence_model.py", "pipeline.py")}
    protocol_hash = freeze(out, "protocol.json", {"base_git": "4c3859b1024f2a9a890fb83f98df86c77c65314a",
        "source_sha256": sources, "atlas_sha256": digest(args.atlas), "frozen_v10_actor_sha256": digest(actor),
        "frozen_v10_reader_sha256": digest(run.READER), "cohorts": {k:getattr(args,k) for k in run.SEEDS},
        "layout_seeds": run.SEEDS, "teacher_selection_tapes": run.REVEALS, "teacher_proof_tape": run.PROOF,
        "holdout_tapes": run.TEST_TAPES, "reader_replay_layouts_per_batch": [8,7,5],
        "femoral": "v10 frozen actor and readout; mixed readout remains an on-trajectory diagnostic",
        "action": "same material station, alternate sectors, actual contact acquisition, equal elapsed time and same exit pose"})
    blend = out / "mixed_reader" / "reader_bundle.json"
    if not (blend.parent / "results.json").exists():
        fit_reader(out / "acquisition", run.READER, blend.parent)
    rr = read_json(blend.parent / "results.json")
    if digest(blend) != rr["bundle_sha256"]:
        raise RuntimeError("The reader candidate does not match its training record")
    v10_dev = rr["baseline_v10"]["groin_to_distal_calf"]
    best = next(x for x in rr["epochs"] if x["epoch"] == rr["selected"]["epoch"])["development"]["groin_to_distal_calf"]
    new_score = (best["strict"], -best["wrong_branch_cases"])
    old_score = (v10_dev["strict"], -v10_dev["wrong_branch_cases"])
    use_blend = new_score > old_score and best["wrong_branch_cases"] <= v10_dev["wrong_branch_cases"]
    reader = blend if use_blend else run.READER
    freeze(out, "reader_freeze.json", {"protocol_sha256": protocol_hash, "reader_sha256": digest(reader),
          "mixed_reader_sha256": digest(blend), "selected": "mixed" if use_blend else "v10", "development_mixed": new_score,
          "development_v10": old_score})
    with run.pool(args, reader, actor) as work:
        shards, teacher_rows = run.collect(work, out / "teach", list(run.cases("teach", args.teach)), run.teacher_job)
    atomic_json(out / "teacher_results.json", teacher_panel(teacher_rows))
    model = out / "model" / "model.json"
    if not (model.parent / "training.json").exists():
        fit_evidence(shards, model.parent)
    if digest(model) != read_json(model.parent / "training.json")["model_sha256"]:
        raise RuntimeError("The evidence model does not match its optimizer record")
    with run.pool(args, reader, actor, model, blend) as work:
        results = []
        for i, cutoff in enumerate((.20, .34, .49)):
            settings = {"trigger": cutoff, "advantage": 0., "trust": .66}
            _, dev = run.collect(work, out / "dev_policy" / f"{i:02d}", list(run.cases("select", args.select)),
                                 run.control_job, lambda c: (c, (None,), settings))
            scored = measure(dev)
            record = scored["groin_to_distal_calf"]
            key = (record["arms"]["task_old_image"]["wrong_branch"] <= record["arms"]["base"]["wrong_branch"],
                   record["arms"]["task_old_image"]["strict"], -record["arms"]["task_old_image"]["wrong_branch"])
            results.append({"index": i, "settings": settings, "key": key, "scores": scored})
        atomic_json(out / "development_results.json", results)
        candidates = [r for r in results if r["scores"]["groin_to_distal_calf"]["triggered"].get("task", 0)]
        chosen = max(candidates or results, key=lambda row: row["key"])
        deploy = freeze(out, "test_freeze.json", {"protocol_sha256": protocol_hash, "reader_sha256": digest(reader),
                        "model_sha256": digest(model), "selected": chosen["index"], "rule": chosen["settings"]})
        _, test = run.collect(work, out / "independent", list(run.cases("test", args.test)), run.control_job,
                              lambda c: (c, run.TEST_TAPES, chosen["settings"]))
    cohorts = [mined, teacher_rows, dev, test]
    all_ids = [r["layout_id"] for group in cohorts for r in group]
    if len(set(all_ids)) != len(all_ids) or any(digest(ROOT/name) != sha for name, sha in sources.items()):
        raise RuntimeError("Overlapping layouts or frozen code changed")
    atomic_json(out / "paired_outcomes.json", test)
    result = {"scores": measure(test), "layout_cohorts_disjoint": len(all_ids), "layouts": len(test),
              "freeze_sha256": deploy, "protocol_sha256": protocol_hash, "paired_sha256": digest(out / "paired_outcomes.json")}
    atomic_json(out / "results.json", result)
    for route, panel in result["scores"].items():
        print(route, {arm: (m["strict"], m["wrong_branch"]) for arm, m in panel["arms"].items()}, flush=True)
