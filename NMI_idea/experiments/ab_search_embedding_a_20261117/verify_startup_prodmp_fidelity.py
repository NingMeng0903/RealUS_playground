"""Change the first-motion representation while holding the startup choice fixed."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from adaptive_controller import FRAME_STRIDE, FreeCandidate
from diagnose_b_selection import run_parallel, save_archive
from evaluate_startup_without_gates import FRESH_PURPOSE, recorded_menu
from run_adaptive_recovery import paired_statistics
from run_learned_recovery import fingerprint, read_rows, write_json
from train_startup_b import TEST_TAPES, futures, load_controllers, menu_metadata, start_case
from verify_startup_ood import OUTCOMES, execute_trial


SELECTORS = ("fixed", "completion_first")
ARMS = tuple(f"{selector}_{kind}" for selector in SELECTORS for kind in ("deployed", "analytic"))
FIELDS = ("corrected_complete", "first_interval_recaptured", "final_wrong", "new_identity_mistake",
          "tcp_mm", "turn_degs", "seconds")


def compare_reference(controller, root, deployed):
    info = {"name": deployed.name, "renderer": getattr(deployed, "renderer", "legacy_prodmp")}
    if info["renderer"] != "fit":
        return deployed, info
    if not isinstance(deployed, FreeCandidate):
        raise ValueError("a fitted motion has no free-motion request")
    analytic = controller.free_candidate(root, deployed.name, *deployed.request, renderer="waypoint")
    info["analytic_admissible"] = analytic is not None
    if analytic is None:
        return deployed, info
    np.testing.assert_array_equal(deployed.s, analytic.s)
    np.testing.assert_array_equal(deployed.vs, analytic.vs)
    for field in ("x", "v"):
        np.testing.assert_allclose(getattr(deployed.lateral, field)[[0, -1]],
                                   getattr(analytic.lateral, field)[[0, -1]], atol=1e-12, rtol=0.)
    difference = np.abs(deployed.lateral.x - analytic.lateral.x)
    info["max_nominal_position_difference_mm"] = float(difference.max())
    info["max_nominal_difference_at_acquisition_mm"] = float(difference[FRAME_STRIDE::FRAME_STRIDE].max())
    return analytic, info


def replay_case(row):
    controller, scene, root, prefix = start_case(row)
    menu = controller.options_shared(root, renderer="fit")
    if menu_metadata(menu) != recorded_menu(row):
        raise ValueError("the original admissible candidate menu changed")
    plans, shapes = {}, {}
    for selector in SELECTORS:
        deployed = menu[row["selection"][selector]]
        analytic, shapes[selector] = compare_reference(controller, root, deployed)
        plans[f"{selector}_deployed"] = deployed
        plans[f"{selector}_analytic"] = analytic

    verification = []
    for tape, future in enumerate(futures(scene, row["seed"], TEST_TAPES, FRESH_PURPOSE)):
        outcomes, cache = {}, {}
        for arm, action in plans.items():
            key = (action.name, getattr(action, "renderer", "legacy_prodmp"))
            if key not in cache:
                cache[key] = execute_trial(controller, root, prefix, future, action)
            outcomes[arm] = cache[key]
        for selector in SELECTORS:
            archived = row["verification"][tape][selector]
            observed = outcomes[f"{selector}_deployed"]
            if any(observed[k] != archived[k] for k in OUTCOMES):
                raise ValueError(f"deployed replay changed for {selector} at {row['seed']}, tape {tape}")
        verification.append(outcomes)
    return {k: row[k] for k in ("seed", "family", "group")} | {"first_shapes": shapes, "verification": verification}


def summarize(rows):
    summary = {}
    for selector in SELECTORS:
        pair = (f"{selector}_deployed", f"{selector}_analytic")
        matched = [r for r in rows if r["first_shapes"][selector].get("analytic_admissible")]
        comparisons = {}
        for kind, cohort in (("full_policy", rows), ("both_renderers_admissible", matched)):
            if not cohort:
                continue
            stats = paired_statistics(cohort, pair, FIELDS, lambda r: r["verification"])
            tapes = [t for r in cohort for t in r["verification"]]
            stats["correct_completion"] = {a: int(sum(t[a]["corrected_complete"] for t in tapes)) for a in pair}
            stats["discordant_runs"] = {k: sum(t[pair[0]][k] != t[pair[1]][k] for t in tapes) for k in FIELDS[:4]}
            comparisons[kind] = stats
        summary[selector] = {
            "scenes_per_deployed_renderer": dict(Counter(r["first_shapes"][selector]["renderer"] for r in rows)),
            "fitted_scenes_with_analytic_infeasible": sum(
                r["first_shapes"][selector].get("analytic_admissible") is False for r in rows),
            "max_nominal_position_difference_mm": max((r["first_shapes"][selector][
                "max_nominal_position_difference_mm"] for r in matched), default=None),
            "max_nominal_difference_at_acquisition_mm": max((r["first_shapes"][selector][
                "max_nominal_difference_at_acquisition_mm"] for r in matched), default=None),
            "comparisons": comparisons,
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/startup_b_without_gates_v1"))
    parser.add_argument("--controller", type=Path, default=Path("results/startup_b_new_library_v1"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=24)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    library = load_controllers(args.controller)
    archive = args.source / "fresh.jsonl.gz"
    prior = json.loads((args.source / "fresh_summary.json").read_text())
    if fingerprint(archive) != prior["archive_sha256"]:
        raise ValueError("source results changed since the paired comparison")
    rows = run_parallel(replay_case, read_rows(archive), args.jobs, "startup ProDMP fidelity")
    args.out.mkdir(parents=True, exist_ok=True)
    replay = args.out / "replay.jsonl.gz"
    save_archive(replay, rows)
    write_json(args.out / "summary.json", {
        "source_archive_sha256": prior["archive_sha256"], "replay_sha256": fingerprint(replay),
        "prodmp_sha256": library["prodmp_sha256"], "selectors": summarize(rows),
    })


if __name__ == "__main__":
    main()
