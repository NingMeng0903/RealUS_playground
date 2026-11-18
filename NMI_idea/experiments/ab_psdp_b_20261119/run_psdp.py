"""Train and freeze a PSDP-style finite-horizon B before paired holdout tests."""

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np

import benchmark as b
from collect import collect_pairs, collect_successors, scenario_rows
from policy import HERE, LEGACY_RUN, MODELS, load_references, p, scores, select_direct, select_modes
from diagnose_b_selection import run_parallel, save_archive
from learned_recovery import fit_targets
from run_learned_recovery import write_json


TRAIN = {"successors": (79_010_000, 24), "pairs": (79_110_000, 32)}
HOLDOUT = {"test": (79_210_000, 32), "drift": (79_310_000, 12)}
STAGES = ("successors", "child_fit", "pairs", "root_fit", "freeze", "test", "drift", "summary")
MIX = 640 / (8122 + 640)
BASE = HERE.parent
BASE_COMMIT = "b66151b3c0271be1aae1268a83c6a0b3b988740b"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return p.base.fingerprint(path)


def frozen_files(folder, files):
    return {name: digest(folder / name) for name in files}


def assert_files(folder, hashes):
    for name, sha in hashes.items():
        if digest(folder / name) != sha:
            raise ValueError(f"frozen input changed: {folder / name}")


def collect(out, key, job, rows, workers):
    results = run_parallel(job, rows, workers, key)
    save_archive(out / f"{key}.jsonl.gz", results)
    return results


def fit_job(features, values, name):
    print(f"fit {name}: {len(values)} labeled decisions, {features.shape[1]} features", flush=True)
    return fit_targets(features, values)


def freeze_child(out):
    rows = p.base.read_rows(out / "successors.jsonl.gz")
    reached = [child for row in rows for child in row["children"]]
    x = np.concatenate([child["features"] for child in reached])
    y = np.concatenate([child["targets"] for child in reached])
    child = fit_job(x, y, "one remaining probe")
    joblib.dump(child, out / "child.joblib", compress=3)
    manifest = {
        "seeds": [row["seed"] for row in rows],
        "public_successors": len(reached), "labeled_rows": len(y),
        "files": frozen_files(out, ("child.joblib", "successors.jsonl.gz")),
        "code": frozen_files(HERE, ("collect.py", "policy.py")),
    }
    write_json(out / "child_frozen.json", manifest)


def load_child(out):
    manifest = read(out / "child_frozen.json")
    assert_files(out, manifest["files"])
    assert_files(HERE, manifest["code"])
    MODELS["child"] = joblib.load(out / "child.joblib")
    return manifest


def pair_supervision(out):
    child_manifest = load_child(out)
    paired = read(out / "pairs_frozen.json")
    assert_files(out, paired["files"])
    if paired["child_sha256"] != child_manifest["files"]["child.joblib"]:
        raise ValueError("root replays use a different frozen successor policy")
    return p.base.read_rows(out / "pairs.jsonl.gz")


def fit_root(out):
    rows = pair_supervision(out)
    features = np.concatenate([row["features"] for row in rows])
    baseline = np.concatenate([np.repeat(np.mean(row["baseline"], axis=0)[None], len(row["names"]), axis=0)
                               for row in rows])
    direct = np.concatenate([np.mean(row["direct"], axis=1) for row in rows]) - baseline
    adaptive = np.concatenate([np.mean(row["adaptive"], axis=1) for row in rows]) - baseline
    residual = np.concatenate([np.mean(np.asarray(row["adaptive"]) - np.asarray(row["direct"]), axis=1)
                               for row in rows])
    np.testing.assert_allclose(adaptive, direct + residual, rtol=0., atol=1.e-10)
    # Control: same X, same number of examples, same frozen successor policy.
    # Give most examples a direct label under an adaptive head, reproducing
    # the old ratio without introducing unrelated historical states.
    selected = np.random.default_rng(24121).choice(
        len(features), size=round(MIX * len(features)), replace=False)
    mask = np.zeros(len(features), dtype=bool)
    mask[selected] = True
    mixed = np.where(mask[:, None], adaptive, direct)
    targets = {"q1": direct, "q2": adaptive, "mixed_q2": mixed, "residual": residual,
               "mixed_residual": np.where(mask[:, None], residual, 0.)}
    models = {name: fit_job(features, y, name) for name, y in targets.items()}
    joblib.dump(models, out / "heads.joblib", compress=3)
    write_json(out / "target_design.json", {
        "rows_each_head": len(features), "mixed_adaptive_rows": sorted(selected.tolist()),
        "mixed_adaptive_share": float(mask.mean()),
        "positive_completion_effect": int((residual[:, 0] > 1.e-10).sum()),
        "negative_completion_effect": int((residual[:, 0] < -1.e-10).sum()),
        "mean_paired_effect": residual.mean(axis=0).tolist(),
    })


def audit(out, rows):
    load_child(out)
    MODELS.update(joblib.load(out / "heads.joblib"))
    copied, anchor_checks, same_on_handoff = 0, 0, 0
    adaptive_mpc_nomination = 0
    for row in rows:
        one, two = map(np.asarray, (row["direct"], row["adaptive"]))
        np.testing.assert_allclose(one[row["names"].index(row["anchor"])], row["baseline"], atol=1.e-10, rtol=0.)
        anchor_checks += 1
        for i, name in enumerate(row["names"]):
            handoff = np.asarray(row["traces"][name]["branch"]) == p.HANDOFF
            np.testing.assert_array_equal(one[i, handoff], two[i, handoff])
            same_on_handoff += int(handoff.sum())
        adaptive_mpc_nomination += not np.allclose(two[row["names"].index(row["anchor"])], row["baseline"])
    for row in rows[:8]:
        c, truth, root, prefix, menu, _, anchor, _ = p.base.at_handoff(row)
        gate = p.deadline(root)
        state, log = p.run_motion(c, root, prefix, truth, menu[anchor])
        assert p.actual_outcome(c, root, prefix, truth, root, gate) == p.actual_outcome(
            c, state, log, truth, root, gate)
        info = p.base.policy_inputs(c, root, prefix, menu, anchor)
        clean = scores(info, "contrast")
        clean[info["names"].index(anchor)] = 0.
        assert select_modes(info, clean, clean) == select_direct(info, clean)
        false_log = log.copy()
        false_log.packets = [packet | {
            "roles_truth": [992] * len(packet["y"]), "target_truth": [9999., 9999.],
        } for packet in log.packets]
        false_log.certificates = [evt | {"identity_ok": False} for evt in log.certificates]
        real = p.second_inputs(c, state, log, gate)
        fake = p.second_inputs(c, state, false_log, gate)
        if real[2] is not None:
            np.testing.assert_array_equal(real[2]["features"], fake[2]["features"])
            assert p.choose_second(*real, MODELS["child"])[0] == p.choose_second(*fake, MODELS["child"])[0]
            copied += 1
    return {"old_mpc_equivalence_checks": 8, "zero_increment_reproduces_frozen_single": 8,
            "sampled_root_anchor_checks": anchor_checks, "unchanged_tails_on_child_handoff": same_on_handoff,
            "roots_with_distinct_adaptive_incumbent_label": adaptive_mpc_nomination,
            "oracle_metadata_poison_checks": copied}


def freeze(out, references, sources):
    rows = pair_supervision(out)
    child = read(out / "child_frozen.json")
    previous = read(LEGACY_RUN / "frozen.json")
    old_seeds = {seed for group in previous["seeds"].values() for seed in group}
    fresh = {"child_training": child["seeds"], "root_training": [r["seed"] for r in rows]}
    fresh.update({split: [r["seed"] for r in scenario_rows(*param)] for split, param in HOLDOUT.items()})
    used = set(old_seeds)
    for label, seeds in fresh.items():
        if len(set(seeds)) != len(seeds) or used & set(seeds):
            raise ValueError(f"overlapping ancestor scenes: {label}")
        used.update(seeds)
    executed_code = ("probe", "evaluate", "experiment", "adaptive_controller", "learned_recovery",
                     "adaptive_teacher", "benchmark_arbitrary_start", "run_learned_recovery",
                     "verify_startup_ood", "recovery_world", "run_adaptive_recovery", "benchmark_window_encoding",
                     "window_embedding")
    dependencies = {str(Path(sys.modules[module].__file__).resolve().relative_to(BASE)):
                    digest(Path(sys.modules[module].__file__)) for module in executed_code}
    manifest = {
        "upstream_commit": BASE_COMMIT, "seeds": {k: sorted(v) for k, v in fresh.items()},
        "code": frozen_files(HERE, ("policy.py", "collect.py", "benchmark.py", "run_psdp.py", "summarize.py", "PROTOCOL.md")),
        "sources": sources | dependencies | {str(k.relative_to(BASE)): v for k, v in references.items()},
        "files": frozen_files(out, ("successors.jsonl.gz", "child.joblib", "child_frozen.json",
                                    "pairs.jsonl.gz", "pairs_frozen.json", "heads.joblib", "target_design.json")),
        "target_fields": list(p.TARGETS), "audit": audit(out, rows),
    }
    write_json(out / "frozen.json", manifest)
    print(manifest["audit"], flush=True)


def drive(stage, out, workers, references, sources):
    if stage == "successors":
        collect(out, stage, collect_successors, scenario_rows(*TRAIN[stage]), workers)
    elif stage == "child_fit":
        freeze_child(out)
    elif stage == "pairs":
        child = load_child(out)
        rows = scenario_rows(*TRAIN[stage])
        if {row["seed"] for row in rows} & set(child["seeds"]):
            raise ValueError("successor and root training use the same scenarios")
        collect(out, stage, collect_pairs, rows, workers)
        write_json(out / "pairs_frozen.json", {
            "child_sha256": child["files"]["child.joblib"],
            "files": frozen_files(out, ("pairs.jsonl.gz",)),
        })
    elif stage == "root_fit":
        fit_root(out)
    elif stage == "freeze":
        freeze(out, references, sources)
    elif stage in HOLDOUT:
        f = read(out / "frozen.json")
        for path, tag in ((BASE, "sources"), (HERE, "code"), (out, "files")):
            assert_files(path, f[tag])
        load_child(out)
        MODELS.update(joblib.load(out / "heads.joblib"))
        fixed = read(LEGACY_RUN / "fixed.json")
        rows = scenario_rows(*HOLDOUT[stage], setting=stage)
        if sorted(row["seed"] for row in rows) != f["seeds"][stage]:
            raise ValueError("holdout seeds changed")
        collect(out, stage, b.evaluate_case, [(r, fixed[r["family"]]["single"][0]) for r in rows], workers)
    else:
        from runpy import run_path
        run_path(str(HERE / "summarize.py"))["make_summary"](out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", nargs="+", choices=STAGES)
    parser.add_argument("--out", type=Path, default=HERE / "results/main")
    parser.add_argument("--workers", type=int, default=27)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sources = p.base.configure()
    references = load_references()
    for stage in args.stages:
        print(stage, flush=True)
        drive(stage, args.out, args.workers, references, sources)


if __name__ == "__main__":
    main()
