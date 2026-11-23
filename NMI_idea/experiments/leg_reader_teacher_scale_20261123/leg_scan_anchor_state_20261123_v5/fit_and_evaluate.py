"""One bounded association update and frozen same-track evaluation."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

from study_data import (DEFAULT_PACK, EXPERIMENT, ROOT, V4_RUN, atomic_json, collect, current_scan, digest,
                        fit_inventory, init_worlds, pack_acquired, read_json)
import numpy as np
from fit_public import load_public
from reader_panel import calibrate_edges, choose_risk, evaluate_panel
from train_links import train


def fit_phase(out, atlas, workers):
    if not read_json(out / "gpu_preflight.json")["cuda_optimizer_step_verified"]:
        raise RuntimeError("CUDA preflight failed")
    v4 = read_json(V4_RUN / "reader_freeze.json")
    prior = v4["updated"]["weights"]
    if digest(prior) != v4["updated"]["sha256"]:
        raise RuntimeError("V4 frozen warm start differs from its manifest")
    paths, fit_meta = fit_inventory(out)
    archived = sorted((V4_RUN / "fit").glob("*/data.npz"))[:12]
    fit_paths = [*archived, *paths]
    location = out / "reader_weights" / "fit.json"
    manifest = read_json(location) if location.exists() else train(fit_paths, prior, out / "reader_weights")
    if manifest["shards"] != [str(p) for p in fit_paths] or digest(manifest["weights"]) != manifest["sha256"]:
        raise RuntimeError("Reader fit weights do not match their frozen inputs")
    model = load_public(manifest["weights"], "cuda:0")
    with np.load(atlas, allow_pickle=False) as f:
        links = f["links"]
    stage = out / "reader_freeze.json"
    if stage.exists():
        freeze = read_json(stage)
        if freeze["fit"]["sha256"] != manifest["sha256"]:
            raise RuntimeError("Reader deployment parameters changed")
    else:
        edges = calibrate_edges(sorted((V4_RUN / "calibrate").glob("*/data.npz")), model, links)
        print("Matched edge probabilities before one-to-one assignment", flush=True)
        risk, thresholds = choose_risk(sorted((V4_RUN / "select_reader").glob("*/data.npz")), model, links, edges,
                                       {route: v4["position_threshold"] for route in ("femoral", "groin_to_distal_calf")})
        freeze = {"experiment": EXPERIMENT, "fit": manifest, "edge_models": edges, "risk": risk, "thresholds": thresholds,
                  "new_layouts_fit": len(fit_meta), "archived_fit_scans": len(archived) * 96,
                  "source_sha256": {p.name: digest(p) for p in sorted(ROOT.glob("*.py"))},
                  "independent_fit_edge_risk_threshold_sets": True, "long_temporal_gap_training": True,
                  "intent": "fixed graph and autonomous delivery thresholds for the downstream pilot"}
        atomic_json(stage, freeze)
    print("Autonomous submission thresholds frozen before acquiring new tests", flush=True)
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=init_worlds, initargs=(atlas,)) as pool:
        test, meta = collect(pool, out, "reader_test", current_scan, pack_acquired)
    fit_ids = {m["layout_id"] for m in fit_meta}
    if any(m["layout_id"] in fit_ids for m in meta):
        raise RuntimeError("Reader test overlaps new training layouts")
    compared = evaluate_panel(test, model, links, freeze)
    atomic_json(out / "reader_results.json", {"freeze_sha256": digest(stage), "routes": compared})
    for task, result in compared.items():
        print(task, {name: [x["strict"], x["wrong_region_count"], x["wrong_branch"]]
                     for name, x in result["terminal"]["matched"].items()}, flush=True)


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "run")
    p.add_argument("--atlas-pack", type=Path, default=DEFAULT_PACK)
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()
    fit_phase(args.out.resolve(), args.atlas_pack.resolve(), args.workers)


if __name__ == "__main__":
    main()
