"""Symmetric public/oracle delivery-score factorial on frozen v6 trajectories.

The experiment keeps the anonymous reader and the oracle association state as
separate graph inputs. Candidate delivery and regional winner calibration are
fit independently on disjoint development layouts. The terminal v6 reader
fresh panel is used only for the final comparison. This module is also an
importable diagnostic library. ``fit_decoder`` returns a candidate and winner
calibration, while ``decode`` returns one score per anonymous candidate.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import multiprocessing as mp
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
V7 = ROOT.parent / "leg_scan_public_evidence_20261124_v7"
sys.path.insert(0, str(V7))

from study_v7 import DEFAULT_PACK, ROUTES, V4_RUN, V6_RUN, atomic_json, digest, read_json  # noqa: E402
from reader_study import init_panel, load_panel, warm_bundle  # noqa: E402
from causal_reader import scores as frozen_scores  # noqa: E402
from reader_v7 import candidate_delivery, direct_features  # noqa: E402
from readout_state import evaluate, fit_risk  # noqa: E402
from route_metrics import paired  # noqa: E402


GRAPH_STATES = {"public": "objects", "oracle": "object_oracle"}
JOINT_MODES = ("frozen", "refit")
WINNER_MODES = ("frozen", "refit")
VARIANTS = tuple(
    f"{graph}_{joint}_{winner}"
    for graph in GRAPH_STATES
    for joint in JOINT_MODES
    for winner in WINNER_MODES
)

_BASE = None


def _state(row, graph: str):
    try:
        return row["states"][GRAPH_STATES[str(graph)]]
    except KeyError as exc:
        raise ValueError(f"row has no {graph} graph state") from exc


def _join_features(public, state, candidate):
    """One calibrated winner row per region, matching v6 semantics."""

    role = state["object_role"]
    tracks = state["track_ids"]
    rows = []
    indices = []
    for region in range(6):
        ix = np.flatnonzero(public["regions"] == region)
        if not len(ix):
            continue
        best = int(ix[np.argmax(candidate[ix])])
        others = ix[tracks[ix] != tracks[best]]
        runner = float(np.max(candidate[others], initial=0.))
        rows.append([
            float(candidate[best]),
            float(candidate[best] - runner),
            float(role[best, 0] - np.max(role[best, 1:])),
            float(public["quality"][best]),
            min(1., float(state["view_counts"][best]) / 5.),
            1. - float(state["usable"][best]),
            min(1., float(len(ix)) / 120.),
            min(1., float(len(np.unique(tracks[ix]))) / 40.),
            *np.eye(6, dtype=float)[region],
        ])
        indices.append(best)
    return np.asarray(indices, dtype=np.int32), np.asarray(rows, dtype=np.float64).reshape(-1, 14)


def _fit_candidate(rows, graph: str, joint: dict | None):
    samples = []
    for row in rows:
        state = _state(row, graph)
        features = direct_features(row["public"], state)
        labels = np.asarray(row["labels"]) == 0
        if len(features):
            samples.append((features, labels))
    if not samples:
        raise RuntimeError(f"No candidate samples for {graph}")
    return fit_risk(samples) | {"level": "candidate", "graph": graph, "fit_layouts": sorted({r["layout"] for r in rows})}


def _fit_winner(rows, graph: str, candidate_model: dict):
    samples = []
    for row in rows:
        state = _state(row, graph)
        candidate = candidate_delivery(row["public"], state, candidate_model)
        indices, features = _join_features(row["public"], state, candidate)
        if len(indices):
            samples.append((features, np.asarray(row["labels"])[indices] == 0))
    if not samples:
        raise RuntimeError(f"No winner samples for {graph}")
    return fit_risk(samples) | {"level": "winner", "graph": graph, "fit_layouts": sorted({r["layout"] for r in rows})}


def fit_decoder(fit_rows, risk_rows, threshold_rows, old, mode: str, *, joint_mode="refit", winner_mode="refit"):
    """Fit a decoder with layout-disjoint candidate, winner and threshold data.

    ``mode`` is ``public`` or ``oracle``. The candidate joint model is fit on
    ``fit_rows`` when ``joint_mode='refit'`` and is copied from the frozen v6
    calibration otherwise. The regional winner model is fit on ``risk_rows``
    when requested. ``threshold_rows`` selects a delivery threshold only after
    candidate and winner fitting. Rows must contain ``public``, ``states``,
    ``labels``, ``layout``, ``route``, and ``old_calibration``.
    """

    if mode not in GRAPH_STATES or joint_mode not in JOINT_MODES or winner_mode not in WINNER_MODES:
        raise ValueError("invalid graph or fit mode")
    frozen = old[GRAPH_STATES[mode]]
    candidate_model = frozen["joint"] if joint_mode == "frozen" else _fit_candidate(fit_rows, mode, None)
    if joint_mode == "frozen":
        candidate_model = dict(candidate_model) | {"fit_layouts": [], "source": "v6_frozen"}
    winner_model = frozen["winner"] if winner_mode == "frozen" else _fit_winner(risk_rows, mode, candidate_model)
    if winner_mode == "frozen":
        winner_model = dict(winner_model) | {"fit_layouts": [], "source": "v6_frozen"}
    threshold, threshold_report = _select_threshold(threshold_rows, mode, candidate_model, winner_model, old)
    if isinstance(threshold, dict):
        if len(threshold) != 1:
            raise RuntimeError("A route-local decoder must have one threshold")
        threshold = next(iter(threshold.values()))
    return {
        "graph": mode,
        "candidate_mode": joint_mode,
        "winner_mode": winner_mode,
        "candidate": candidate_model,
        "winner": winner_model,
        "threshold": float(threshold),
        "threshold_selection": threshold_report,
    }


def decode(public, state, config, *, mask=None):
    """Return candidate scores for deployment without labels or truth access."""

    candidate = candidate_delivery(public, state, config["candidate"])
    indices, features = _join_features(public, state, candidate)
    result = np.full(len(public["frames"]), -1., dtype=np.float32)
    if len(indices):
        normalized = (features - np.asarray(config["winner"]["center"])) / np.asarray(config["winner"]["scale"])
        weights = np.asarray(config["winner"]["weight"])
        result[indices] = 1. / (1. + np.exp(-(normalized @ weights[:-1] + weights[-1])))
    if mask is not None:
        result[~np.asarray(mask, dtype=bool)] = -1.
    return result


def _baseline_count(row, old):
    route = row["route"]
    calibration = old["calibration"][route]["unary"] if "calibration" in old else old["unary"]
    state = row["states"]["unary"]
    values = frozen_scores(row["public"], {"unary": state}, "unary", calibration)
    return evaluate(row["public"], row["labels"], values, calibration["threshold"], complete=True)["submitted"]


def _select_threshold(rows, graph, candidate_model, winner_model, old):
    """Choose one threshold per route on threshold-only development layouts."""

    route_reports = {}
    by_route = {route: [row for row in rows if row["route"] == route] for route in ROUTES}
    for route, deck in by_route.items():
        if not deck:
            continue
        records = []
        baseline_wrong = 0
        for row in deck:
            state = _state(row, graph)
            values = decode(row["public"], state, {"candidate": candidate_model, "winner": winner_model})
            baseline = _baseline_count(row, old)
            baseline_wrong += int(_baseline_result(row, old)["wrong_branch"])
            records.append((row, values, baseline))
        candidates = np.unique(np.r_[0., np.concatenate([v[v >= 0.] for _, v, _ in records]), 1.])
        options = []
        for threshold in candidates:
            outcomes = [evaluate(row["public"], row["labels"], values, float(threshold), complete=True)
                        for row, values, _ in records]
            submitted = sum(x["submitted"] for x in outcomes)
            strict = sum(x["strict"] for x in outcomes)
            wrong = sum(x["wrong_branch"] for x in outcomes)
            baseline_submitted = sum(base for _, _, base in records)
            admissible = abs(submitted - baseline_submitted) <= max(4, int(np.ceil(.035 * baseline_submitted))) and wrong <= baseline_wrong
            options.append({"threshold": float(threshold), "admissible": bool(admissible), "submitted": int(submitted),
                            "strict": int(strict), "wrong_branch": int(wrong), "baseline_submitted": int(baseline_submitted),
                            "baseline_wrong_branch": int(baseline_wrong)})
        eligible = [x for x in options if x["admissible"]]
        selected = max(eligible, key=lambda x: (x["strict"], -x["wrong_branch"], -abs(x["submitted"] - x["baseline_submitted"]))) if eligible else min(
            options, key=lambda x: (abs(x["submitted"] - x["baseline_submitted"]), x["wrong_branch"], -x["strict"]))
        route_reports[route] = {"selected": selected, "options": options, "layouts": len(deck),
                                "fit_source": "threshold_development_only"}
    # A decoder is used per route. The caller records the route-specific value.
    # Returning the femoral threshold here is only for API completeness. Main
    # evaluation stores and uses each route report separately.
    return {route: report["selected"]["threshold"] for route, report in route_reports.items()}, route_reports


def _baseline_result(row, old):
    route = row["route"]
    calibration = old["calibration"][route]["unary"] if "calibration" in old else old["unary"]
    values = frozen_scores(row["public"], {"unary": row["states"]["unary"]}, "unary", calibration)
    return evaluate(row["public"], row["labels"], values, calibration["threshold"], complete=True)


def _decode_route(row, config, variant):
    route_config = config["route_configs"][row["route"]]
    return decode(row["public"], _state(row, variant["graph"]), route_config)


def _metrics(outcomes):
    fields = ("strict", "acquired_ceiling", "wrong_delivery", "wrong_branch", "wrong_region_count",
              "wrong_branch_regions", "artifact_regions", "correct_regions", "submitted")
    report = {key: int(sum(row[key] for row in outcomes)) for key in fields}
    report["cases"] = len(outcomes)
    report["captured_regions"] = int(sum(sum(row["capture"]) for row in outcomes))
    report["regions"] = []
    for region in range(6):
        report["regions"].append({
            "region": region + 1,
            "captured": int(sum(bool(row["capture"][region]) for row in outcomes)),
            "correct": int(sum(row["region_outcomes"][region] == 1 for row in outcomes)),
            "wrong_branch": int(sum(row["selected_ids"][region] > 0 for row in outcomes)),
            "abstained": int(sum(row["selected_ids"][region] == -99 for row in outcomes)),
        })
    report["late_regions"] = {"regions": [4, 5, 6], "correct": sum(item["correct"] for item in report["regions"][3:]),
                              "wrong_branch": sum(item["wrong_branch"] for item in report["regions"][3:])}
    return report


def _evaluate_variant(rows, variant, configs, old, *, matched=True):
    outcomes, references = [], []
    for row in rows:
        base = _baseline_result(row, old)
        config = configs[variant][row["route"]]
        values = decode(row["public"], _state(row, variant.split("_")[0]), config)
        outcome = evaluate(row["public"], row["labels"], values, config["threshold"], complete=True,
                           submit_count=base["submitted"] if matched else None)
        outcomes.append(outcome); references.append(base)
    result = _metrics(outcomes)
    result["paired_vs_unary"] = paired(outcomes, references, [r["layout"] for r in rows], 205_341_470)
    return result, outcomes, references


def _load_stage(paths, bundle):
    init_panel(bundle)
    result = []
    for path in paths:
        result.extend(r for r in load_panel((path, "evaluate")) if r["desc"]["scope"] == "terminal")
    return result


def _fit_all(dev, old):
    configs = {}
    fit_rows = [r for r in dev if r["fold"] == 0]
    risk_rows = [r for r in dev if r["fold"] == 1]
    threshold_rows = [r for r in dev if r["fold"] == 2]
    split = {"candidate_fit_layouts": sorted({r["layout"] for r in fit_rows}),
             "winner_fit_layouts": sorted({r["layout"] for r in risk_rows}),
             "threshold_layouts": sorted({r["layout"] for r in threshold_rows})}
    for graph in GRAPH_STATES:
        configs[graph] = {}
        for joint_mode in JOINT_MODES:
            for winner_mode in WINNER_MODES:
                # fit_decoder needs route-local old calibration. Refit each route
                # with exactly the same disjoint rows and save route thresholds.
                route_configs = {}
                reports = {}
                for route in ROUTES:
                    local_old = old["calibration"][route]
                    local = fit_decoder([r for r in fit_rows if r["route"] == route],
                                        [r for r in risk_rows if r["route"] == route],
                                        [r for r in threshold_rows if r["route"] == route],
                                        local_old, graph, joint_mode=joint_mode, winner_mode=winner_mode)
                    route_configs[route] = local
                    reports[route] = local["threshold_selection"]
                configs[graph][f"{joint_mode}_{winner_mode}"] = {"route_configs": route_configs,
                                                                   "threshold_reports": reports,
                                                                   "split": split}
    return configs, split


def _prepare_rows(rows, old):
    return [dict(row) | {"old_calibration": old["calibration"]} for row in rows]


def run(*, out=ROOT / "run", atlas=DEFAULT_PACK, workers=6):
    out = Path(out).resolve(); atlas = Path(atlas).resolve()
    base = warm_bundle(V6_RUN, atlas)
    old = read_json(V6_RUN / "reader_freeze.json")
    dev_paths = sorted((V4_RUN / "select_reader").glob("*/data.npz"))
    test_paths = sorted((V6_RUN / "reader_fresh").glob("*/data.npz"))
    if not dev_paths or not test_paths:
        raise FileNotFoundError("v6 development or reader_fresh shards are missing")
    protocol = {"experiment": ROOT.name, "base_v7": read_json(V7 / "run/mechanism_protocol.json").get("base_commit"),
                "atlas_sha256": digest(atlas), "v6_reader_freeze_sha256": digest(V6_RUN / "reader_freeze.json"),
                "development_shards": {str(p.parent.name): digest(p) for p in dev_paths},
                "test_shards": {str(p.parent.name): digest(p) for p in test_paths},
                "graphs": list(GRAPH_STATES), "joint_modes": list(JOINT_MODES), "winner_modes": list(WINNER_MODES),
                "split_rule": "v6 select_reader ordinal//2 modulo 3, candidate fold0, winner fold1, threshold fold2",
                "same_case_budget": "test submit_count equals frozen unary reader per layout", "dird": False}
    atomic_json(out / "readout_protocol.json", protocol)
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(workers, len(dev_paths)), mp_context=ctx) as pool:
        dev = [r for batch in pool.map(_load_stage, ([p] for p in dev_paths), [base] * len(dev_paths)) for r in batch]
    dev = _prepare_rows(dev, old)
    configs, split = _fit_all(dev, old)
    atomic_json(out / "readout_fit_manifest.json", {"protocol_sha256": digest(out / "readout_protocol.json"), "split": split,
                                                     "graphs": list(GRAPH_STATES), "variants": VARIANTS})
    with ProcessPoolExecutor(max_workers=min(workers, len(test_paths)), mp_context=ctx) as pool:
        test = [r for batch in pool.map(_load_stage, ([p] for p in test_paths), [base] * len(test_paths)) for r in batch]
    test = _prepare_rows(test, old)
    results = {}
    for graph in GRAPH_STATES:
        for joint_mode in JOINT_MODES:
            for winner_mode in WINNER_MODES:
                key = f"{graph}_{joint_mode}_{winner_mode}"
                cfg = configs[graph][f"{joint_mode}_{winner_mode}"]
                route_configs = cfg["route_configs"]
                outcomes = {}; raw_outcomes = {}
                for route in ROUTES:
                    rows = [r for r in test if r["route"] == route]
                    variant = f"{graph}_{joint_mode}_{winner_mode}"
                    result, matched, refs = _evaluate_variant(rows, variant, {variant: route_configs}, old, matched=True)
                    raw, _, _ = _evaluate_variant(rows, variant, {variant: route_configs}, old, matched=False)
                    outcomes[route] = {"matched": result, "calibrated": raw}
                results[key] = {"graph": graph, "joint_mode": joint_mode, "winner_mode": winner_mode,
                                "route_results": outcomes, "threshold_reports": cfg["threshold_reports"], "split": split}
    atomic_json(out / "readout_symmetry_results.json", {"protocol_sha256": digest(out / "readout_protocol.json"),
                                                         "variants": results, "test_layouts": len(test),
                                                         "test_routes": {route: sum(r["route"] == route for r in test) for route in ROUTES}})
    return results


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "run")
    parser.add_argument("--atlas-pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    results = run(out=args.out, atlas=args.atlas_pack, workers=args.workers)
    for key, value in results.items():
        print(key, {route: value["route_results"][route]["matched"]["strict"] for route in ROUTES}, flush=True)


if __name__ == "__main__":
    main()
