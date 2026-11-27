"""Verify committed layout, reader, and physically executed anchor artifacts."""

from argparse import ArgumentParser
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from anchor_experiment import DEV_STREAMS, PROGRAMS, TEST_STREAMS
from study_data import DEFAULT_PACK, ROOT, STAGE_COUNTS, atomic_json, digest, load_shard, read_json, roster


READER_SOURCE = ("fit_and_evaluate.py", "reader_panel.py", "study_data.py", "train_links.py", "object_tracks.py")


def verify_phase(out, name):
    cases = []
    paths = sorted((out / name).glob("*/commit.json"))
    for path in paths:
        index = read_json(path)
        data, rows = load_shard(path.parent)
        if index["data_sha256"] != digest(data) or index["metadata_sha256"] != digest(path.parent / "cases.jsonl.gz"):
            raise RuntimeError(f"Changed shard content {path}")
        if len(rows) != index["cases"] or not all(row["complete"] for row in rows):
            raise RuntimeError(f"Incomplete shard {path}")
        if name.startswith("anchor_"):
            streams = TEST_STREAMS if name.endswith("_test") else DEV_STREAMS
            expected = [[[m["programs"][stream][p]["strict"] for p in PROGRAMS] for stream in streams] for m in rows]
            with np.load(data, allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays["strict"], expected)
        cases.extend(rows)
    if len(cases) != STAGE_COUNTS[name] or [r["case"] for r in cases] != [asdict(c) for c in roster(name)]:
        raise RuntimeError(f"Stage roster differs from protocol for {name}")
    return cases, len(paths)


def physical_checks(cases, stage):
    traces = Counter()
    streams = TEST_STREAMS if stage.endswith("_test") else DEV_STREAMS
    for row in cases:
        first, target = row["anchor"]["prefix_station"], row["anchor"]["station"]
        expected = list(range(first + 1, target + 1))
        if not 3 <= len(expected) <= 6:
            raise RuntimeError("Anchor span deviated from the protocol")
        if list(row["programs"]) != list(streams):
            raise RuntimeError("Case lacks an independent noise tape")
        for techniques in row["programs"].values():
            if list(techniques) != list(PROGRAMS):
                raise RuntimeError("Case lacks a paired action program")
            for name, arm in techniques.items():
                if (arm["layout_id"] != row["layout_id"] or not arm["complete_budget"]
                        or [p["station"] for p in arm["forward_path"]] != expected):
                    raise RuntimeError("Continuation did not preserve the physical route or anatomy")
                b = arm["boundary"]
                if b["station"] != target or (name != "mpc" and b["tick"] != row["elapsed_comparison_boundary_tick"]):
                    raise RuntimeError("Action arm violated the registered comparison clock")
                required = list(range(row["origin"]["region"] + 1, 6))
                if arm["suffix_regions"] != required:
                    raise RuntimeError("Missing required suffix regions")
                criterion = all(arm["suffix_outcomes"][r] == 1 for r in required)
                final_delivery = all(arm["region_outcomes"][r] == 1 for r in required)
                if arm["suffix_valid"] != criterion or arm["joint"] != (arm["old_correct"] and criterion):
                    raise RuntimeError("Joint outcome did not meet its per-region definition")
                traces["budget_legal_complete_continuations"] += 1
                traces["post_prefix_and_final_suffix_all_correct_differ"] += criterion != final_delivery
    return dict(traces)


def useful_signal(cases):
    by_route = {}
    for route in sorted({r["case"]["task"] for r in cases}):
        scope = [r for r in cases if r["case"]["task"] == route]
        active = PROGRAMS[2:]
        proof = ("proof_0", "proof_1")
        result = {}
        for ref in PROGRAMS[:2]:
            differences = np.asarray([[[int(m["programs"][t][p]["strict"]) - int(m["programs"][t][ref]["strict"])
                                        for t in proof] for p in active] for m in scope])
            one_helps = (differences > 0).any(axis=2)
            one_harms = (differences < 0).any(axis=2)
            result[ref] = {"layout_with_any_active_program_rescue": int(one_helps.any(axis=1).sum()),
                           "layout_with_any_active_program_harm": int(one_harms.any(axis=1).sum()),
                           "layout_with_an_active_program_rescuing_both": int((differences > 0).all(axis=2).any(axis=1).sum()),
                           "layout_with_an_active_program_harming_both": int((differences < 0).all(axis=2).any(axis=1).sum())}
        ranks = [r for m in scope for r in m["programs"]["actual"]["anchor_information"]["ranking"]
                 if r["hypotheses"] == 2]
        variations, clears, changes = [], 0, 0
        for r in ranks:
            available = np.asarray(r["legal"], dtype=bool)
            if available.sum() < 2:
                continue
            a = np.array(r["identity_scores"], dtype=float)[available]
            b = np.array(r["unary_scores_same_geometry"], dtype=float)[available]
            amplitude = float(np.max(abs(a - b)))
            variations.append(amplitude)
            unary_margin = float(np.diff(np.sort(b)[-2:])[0])
            clears += 2 * amplitude < unary_margin
            changes += int(a.argmax() != b.argmax())
        result["same_geometry_rank_sensitivity"] = {"decisions_with_at_least_two_legal_windows": len(variations),
              "new_graph_score_change_median": float(np.median(variations)) if variations else None,
              "new_graph_score_change_maximum": max(variations, default=None),
              "decisions_unary_lead_exceeds_twice_the_change": int(clears),
              "discrete_changes_vs_unary": changes}
        by_route[route] = result
    return by_route


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "run")
    parser.add_argument("--atlas-pack", type=Path, default=DEFAULT_PACK)
    args = parser.parse_args()
    out = args.out.resolve()
    acquisition = read_json(out / "acquisition.json")
    if acquisition["atlas_sha256"] != digest(args.atlas_pack):
        raise RuntimeError("Atlas pack differs from the acquisition record")
    read_path, anchor_path = (out / name for name in ("reader_freeze.json", "integrated_freeze.json"))
    read, anchor = read_json(read_path), read_json(anchor_path)
    reader_result, anchor_result = (read_json(out / name) for name in ("reader_results.json", "anchor_results.json"))
    if reader_result["freeze_sha256"] != digest(read_path) or anchor_result["protocol_sha256"] != digest(anchor_path):
        raise RuntimeError("Study results disagree with the corresponding freeze")
    if digest(read_path) != anchor["reader_freeze_sha256"] or digest(read["fit"]["weights"]) != read["fit"]["sha256"]:
        raise RuntimeError("Deployed weights or reader protocol changed")
    for item, module_names in ((read, READER_SOURCE), (anchor, anchor["source_sha256"])):
        if any(item["source_sha256"][name] != digest(ROOT / name) for name in module_names):
            raise RuntimeError("Source used in a frozen experiment changed")
    layouts, seeds, stages, signal = set(), set(), {}, {}
    for stage in STAGE_COUNTS:
        cases, shards = verify_phase(out, stage)
        ids = {r["layout_id"] for r in cases}
        nums = {r["case"]["world_seed"] for r in cases}
        if len(ids) != len(cases) or layouts & ids or seeds & nums:
            raise RuntimeError("Study stages reused layouts or case seeds")
        layouts.update(ids)
        seeds.update(nums)
        stats = physical_checks(cases, stage) if stage.startswith("anchor_") else {}
        sources = {route: sorted({r["case"]["source"] for r in cases if r["case"]["task"] == route})
                   for route in sorted({r["case"]["task"] for r in cases})}
        stages[stage] = {"cases": len(ids), "sha256_verified_shards": shards, "source_indices_by_route": sources} | stats
        if stage == "anchor_test":
            signal = useful_signal(cases)
    realized = sum(s.get("budget_legal_complete_continuations", 0) for s in stages.values())
    if realized != anchor_result["all_candidates_explicitly_executed_continuations"]:
        raise RuntimeError("Reported branch count differs from the committed simulations")
    result = {"new_distinct_layouts": len(layouts), "stages": stages, "complete_anchor_continuations": realized,
              "independent_tape_active_menu_descriptive_signal": signal,
              "reader_freeze_sha256": digest(read_path), "anchor_freeze_sha256": digest(anchor_path),
              "association_weights_sha256": read["fit"]["sha256"],
              "reader_freeze_reference_only_source_changes": [name for name, hash_value in read["source_sha256"].items()
                  if name not in READER_SOURCE and hash_value != digest(ROOT / name)]}
    atomic_json(out / "verification.json", result)
    print(f"Verified {len(layouts)} disjoint layouts and {realized} complete anchor continuations", flush=True)


if __name__ == "__main__":
    main()
