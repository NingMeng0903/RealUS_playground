"""Train and independently calibrate the public association score.

The v5 retrieval and forest are kept byte-for-byte as the deployed mechanism.
This experiment changes only the public edge confidence.  The identity-
probability disagreement field is deliberately omitted from the main score so
that a useful new observation is not rejected merely because it disagrees with
the old candidate.  An independent calibration layout split selects the
probability map, and a third operation-point split selects the forest cutoff.
The v6 reader-fresh panel is held out for testing only.
"""

from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
V5 = ROOT.parent / "leg_scan_anchor_state_20261123_v5"
V6 = ROOT.parent / "leg_scan_identity_state_20261123_v6"
V4 = ROOT.parent / "leg_scan_reader_feedback_20261123_v4"
V5_RUN = V5 / "run"
V6_RUN = V6 / "run" / "confirmed"
sys.path.insert(1, str(V6))
sys.path.insert(1, str(V5))
sys.path.insert(1, str(V4))

from causal_reader import PUBLIC_FIELDS, heads, model_state  # noqa: E402
from object_tracks import EDGE_FIELDS, edge_inputs, edge_probabilities, infer, retrieve  # noqa: E402
from reader_study import warm_bundle  # noqa: E402
from v6_data import DEFAULT_PACK, digest, load_shard, read_json  # noqa: E402


EXPERIMENT = "leg_scan_reader_feedback_values_20261124_v8"
BASE = "2f2c704eb0085f78e638d41cd2467c436713b695"
SEED = 2026112408
# The primary public score may use the learned association embedding and
# physical/quality features, but it must not depend on either hand-crafted
# identity agreement field.  ``role_overlap`` is an identity-distribution
# overlap proxy and ``target_probability_disagreement`` directly compares the
# target unary; both are excluded from the deployed score.  The latter two
# columns remain in the feature table for controlled diagnostics and for exact
# compatibility with the v5 forest API.
USED_FEATURES = tuple(range(8))
POSITIVE_MIN = 16
OP_TARGET_PRECISION = 0.90
NATURAL_PANEL_CAP = 1_400_000

_BASE_BUNDLE: dict[str, Any] | None = None


def graph(public: dict[str, Any], unary: Any, embedding: Any, links: Any,
          bundle: dict[str, Any]) -> dict[str, Any]:
    """Run the exact v5 retrieval/forest with the frozen v8 public edge model.

    The return schema is the v5 ``infer`` schema, including ``edges``,
    ``scored_edges``, ``track_ids``, ``tracks``, ``hypotheses`` and the
    ``stats.edge_threshold`` operation point used by downstream readers.
    """

    return infer(public, unary, embedding, links, edge_model=bundle)


def _json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _layout_split(layout: str) -> str:
    value = int(hashlib.sha256(str(layout).encode()).hexdigest()[:8], 16) % 4
    return ("train", "train", "calibration", "operation")[value]


def init_worker(atlas: str, v6_run: str) -> None:
    global _BASE_BUNDLE
    _BASE_BUNDLE = warm_bundle(Path(v6_run), Path(atlas))


def _panel_worker(path: str, *, v6: bool = False) -> dict[str, Any]:
    """Extract edge features and exact instance labels from one archive shard."""

    if _BASE_BUNDLE is None:
        raise RuntimeError("worker was not initialized")
    path_obj = Path(path)
    _, cases = load_shard(path_obj.parent)
    arrays_by_key: dict[str, np.ndarray]
    records: list[dict[str, Any]] = []
    with np.load(path_obj, allow_pickle=False) as archive:
        arrays_by_key = {key: archive[key] for key in (*PUBLIC_FIELDS, "labels", "instance", "offsets")}
        descriptions = [(case, history) for case in cases for history in case["histories"]]
        offsets = arrays_by_key["offsets"]
        for number, ((lo, hi), (case, desc)) in enumerate(zip(zip(offsets[:-1], offsets[1:]), descriptions, strict=True)):
            public = {key: arrays_by_key[key][lo:hi] for key in PUBLIC_FIELDS}
            labels = arrays_by_key["labels"][lo:hi]
            instance = arrays_by_key["instance"][lo:hi]
            p, embedding = heads(public["features"], _BASE_BUNDLE["model"])
            pairs = retrieve(public)
            features = edge_inputs(public, p, embedding, pairs)
            if len(pairs):
                left, right = pairs[:, 0], pairs[:, 1]
                positive = (instance[left] >= 0) & (instance[left] == instance[right])
                identity_disagreement = np.abs(p[left, 0] - p[right, 0])
                physical = np.minimum(p[left, :6].sum(axis=1), p[right, :6].sum(axis=1))
                hard_positive = positive & (identity_disagreement >= .20)
                hard_negative = (~positive) & (features[:, 0] >= .45) & (features[:, 1] >= .45) & (identity_disagreement <= .10)
            else:
                positive = np.empty(0, dtype=bool)
                hard_positive = positive.copy()
                hard_negative = positive.copy()
                physical = np.empty(0, dtype=float)
            split = _layout_split(case["layout_id"] if "layout_id" in case else case.get("identifier", number))
            records.append({
                "layout": case.get("layout_id", case.get("identifier", str(number))),
                "case": case,
                "scope": desc.get("scope", "unknown"),
                "route": case["case"]["task"] if "case" in case else case["task"],
                "split": split,
                "features": features.astype(np.float32),
                "labels": positive.astype(np.float32),
                "hard_positive": hard_positive.astype(bool),
                "hard_negative": hard_negative.astype(bool),
                "pairs": int(len(pairs)),
                "positive_count": int(positive.sum()),
                "physical_mean": float(physical.mean()) if len(physical) else 0.,
                "v6": bool(v6),
            })
    return {"records": records, "path": str(path_obj), "v6": bool(v6)}


def _paths(stage: Path) -> list[Path]:
    return sorted(stage.glob("*/data.npz"))


def _merge_records(batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = [record for batch in batches for record in batch["records"]]
    # A case contributes before, probe, and terminal prefixes.  These records
    # intentionally share one layout and are assigned to the same split.
    seen: dict[str, str] = {}
    for record in result:
        prior = seen.setdefault(record["layout"], str(record["case"]["case"]["identifier"]))
        if prior != str(record["case"]["case"]["identifier"]):
            raise RuntimeError("a layout was reused by different cases")
    return result


def _matrix(
    records: list[dict[str, Any]],
    split: str,
    *,
    natural_mix: bool = False,
    cap: int = NATURAL_PANEL_CAP,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    chunks = [record for record in records if record["split"] == split]
    if not chunks:
        raise RuntimeError(f"empty {split} edge panel")
    x = np.concatenate([record["features"] for record in chunks if len(record["features"])], axis=0)
    y = np.concatenate([record["labels"] for record in chunks if len(record["labels"])], axis=0)
    if len(x) == 0 or np.unique(y).size < 2:
        raise RuntimeError(f"{split} edge panel lacks both classes")
    unique_layouts = sorted({str(record["layout"]) for record in chunks})
    unique_cases = sorted({str(record["case"]["case"]["identifier"]) for record in chunks})
    natural_rows = int(len(y))
    natural_positive = int(np.count_nonzero(y >= .5))
    natural_negative = int(natural_rows - natural_positive)
    # The fitted train panel is deliberately class-balanced for optimization.
    # Calibration and operation-point panels instead use a uniform sample of
    # the complete retrieved-pair population, preserving its natural class
    # mixture.  This prevents a 1:6 train cap from being misreported as an
    # operating precision estimate.
    split_seed = int(hashlib.sha256(split.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(SEED + split_seed)
    if natural_mix:
        selected = min(natural_rows, int(cap))
        keep = np.sort(rng.choice(natural_rows, selected, replace=False)) if selected < natural_rows else np.arange(natural_rows)
        sampling = "uniform_over_all_retrieved_pairs"
    else:
        pos = np.flatnonzero(y >= .5)
        neg = np.flatnonzero(y < .5)
        if len(pos) > 200_000:
            pos = np.sort(rng.choice(pos, 200_000, replace=False))
        keep_neg = neg if len(neg) <= max(64, 6 * len(pos)) else rng.choice(neg, max(64, 6 * len(pos)), replace=False)
        keep = np.r_[pos, np.sort(keep_neg)]
        sampling = "all_positive_at_most_six_negatives_per_positive"
    selected_y = y[keep]
    return x[keep].astype(np.float32), y[keep].astype(np.float32), {
        "layouts": len(unique_layouts), "prefix_records": len(chunks), "unique_layouts": len(unique_layouts), "unique_cases": len(unique_cases),
        "rows": int(len(keep)), "positive": int(np.count_nonzero(selected_y >= .5)),
        "negative": int(np.count_nonzero(selected_y < .5)), "natural_rows": natural_rows,
        "natural_positive": natural_positive, "natural_negative": natural_negative,
        "sampling": sampling, "cap": int(cap) if natural_mix else None,
        "hard_positive": int(sum(record["hard_positive"].sum() for record in chunks)),
        "hard_negative": int(sum(record["hard_negative"].sum() for record in chunks)),
    }


def _fit_cuda(x: np.ndarray, y: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    import torch.nn.functional as F
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for edge training")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    selected = np.asarray(USED_FEATURES, dtype=int)
    mean = x[:, selected].mean(axis=0).astype(np.float32)
    scale = np.maximum(x[:, selected].std(axis=0), .035).astype(np.float32)
    xx = (x[:, selected] - mean) / scale
    positive = max(float((y >= .5).sum()), 1.)
    negative = max(float((y < .5).sum()), 1.)
    pos_weight = torch.tensor([negative / positive], dtype=torch.float32, device="cuda")
    features = torch.as_tensor(xx, dtype=torch.float32, device="cuda")
    targets = torch.as_tensor(y[:, None], dtype=torch.float32, device="cuda")
    model = torch.nn.Linear(len(selected), 1, bias=True).cuda()
    initial = [v.detach().clone() for v in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.e-3, weight_decay=2.e-4)
    best_loss = float("inf")
    best_state = None
    losses: list[float] = []
    for epoch in range(80):
        model.train()
        logits = model(features)
        loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        losses.append(value)
        if value < best_loss:
            best_loss = value
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("edge training did not produce a checkpoint")
    model.load_state_dict(best_state)
    torch.cuda.synchronize()
    changed = any(not torch.equal(parameter.detach(), before) for parameter, before in zip(model.parameters(), initial))
    if not changed:
        raise RuntimeError("CUDA edge optimizer did not change parameters")
    weight = np.zeros(len(EDGE_FIELDS), dtype=np.float64)
    weight[selected] = np.asarray(best_state["weight"], dtype=np.float64).reshape(-1)
    bias = float(np.asarray(best_state["bias"], dtype=np.float64).reshape(-1)[0])
    bundle = {"mean": np.pad(mean.astype(float), (0, len(EDGE_FIELDS) - len(selected))).tolist(),
              "scale": np.pad(scale.astype(float), (0, len(EDGE_FIELDS) - len(selected)), constant_values=1.).tolist(),
              "weight": np.r_[weight, bias].tolist(), "features": list(EDGE_FIELDS),
              "used_features": [EDGE_FIELDS[i] for i in selected], "threshold": .90,
              "calibration": [1., 0.], "cal_rows": 0, "cal_positive": 0,
              "fit_rows": int(len(y)), "fit_positive": int((y >= .5).sum())}
    return bundle, {"device": "cuda:0", "optimizer": "AdamW", "epochs": 80,
                    "parameter_count": len(selected) + 1, "optimizer_steps": 80,
                    "cuda_optimizer_step_verified": True, "best_loss": best_loss,
                    "loss_tail": losses[-5:]}


def _calibrate(bundle: dict[str, Any], x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    from scipy.optimize import minimize
    from scipy.special import expit
    raw = edge_probabilities(x, bundle)
    logits = np.log(np.clip(raw, 1e-7, 1 - 1e-7) / np.clip(1 - raw, 1e-7, 1))
    design = np.c_[logits, np.ones(len(logits))]
    target = y.astype(float)
    def objective(ab):
        z = design @ ab
        loss = np.mean(np.logaddexp(0., z) - target * z) + .0005 * np.sum(ab * ab)
        grad = design.T @ (expit(z) - target) / len(target) + .0005 * ab
        return float(loss), grad
    fit = minimize(objective, np.array([.5, 0.]), jac=True, method="L-BFGS-B",
                   bounds=[(-8., 8.), (-12., 12.)], options={"maxiter": 500})
    if fit.success and np.isfinite(fit.x).all():
        calibration = fit.x
        fallback = None
    else:
        # Perfectly separable panels can make the unbounded Platt objective
        # terminate abnormally. A bounded constant calibrator remains a
        # genuine independent calibration result and is recorded explicitly.
        mean = float(np.clip(target.mean(), 1.e-4, 1. - 1.e-4))
        calibration = np.array([0., np.log(mean / (1. - mean))])
        fallback = "bounded_constant_after_platt_failure"
    bundle["calibration"] = calibration.tolist()
    bundle["cal_rows"] = int(len(y)); bundle["cal_positive"] = int((y >= .5).sum())
    bundle["calibration_fallback"] = fallback
    return bundle


def _operation_threshold(bundle: dict[str, Any], x: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, Any]]:
    scores = edge_probabilities(x, bundle)
    thresholds = np.unique(np.r_[0., np.quantile(scores, np.linspace(.01, .999, 300)), 1.])
    choices = []
    for threshold in thresholds:
        selected = scores >= threshold
        count = int(selected.sum()); true = int((selected & (y >= .5)).sum())
        precision = true / max(count, 1); recall = true / max(int((y >= .5).sum()), 1)
        choices.append((precision >= OP_TARGET_PRECISION, recall, precision, count, float(threshold)))
    valid = [row for row in choices if row[0]]
    chosen = max(valid, key=lambda row: (row[1], row[2], -row[4])) if valid else max(choices, key=lambda row: (row[2], row[1]))
    return chosen[-1], {"target_precision": OP_TARGET_PRECISION, "selected_threshold": chosen[-1],
                        "precision": chosen[2], "recall": chosen[1], "selected": chosen[3],
                        "positive": int((y >= .5).sum()), "certified": bool(chosen[0]),
                        "candidates": len(choices)}


def _metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, Any]:
    selected = scores >= threshold; positive = labels >= .5
    tp = int(np.sum(selected & positive)); fp = int(np.sum(selected & ~positive)); fn = int(np.sum(~selected & positive))
    return {"rows": int(len(labels)), "positive": int(positive.sum()), "selected": int(selected.sum()),
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "threshold": float(threshold)}


def _truth_forced(bundle: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    normal, forced, hard_normal, hard_forced = [], [], [], []
    for record in records:
        x = record["features"].astype(float); y = record["labels"] >= .5
        if not len(x):
            continue
        p = edge_probabilities(x, bundle)
        xf = x.copy(); xf[y, 1] = 1.
        pf = edge_probabilities(xf, bundle)
        normal.append(p); forced.append(pf)
        hard_normal.append(p[record["hard_positive"]]); hard_forced.append(pf[record["hard_positive"]])
    result = {"rows": int(sum(len(x) for x in normal)), "positive": int(sum(len(x) for x in hard_normal)),
              "normal_positive_median": float(np.median(np.concatenate(hard_normal))) if hard_normal and any(len(x) for x in hard_normal) else None,
              "truth_forced_cosine_positive_median": float(np.median(np.concatenate(hard_forced))) if hard_forced and any(len(x) for x in hard_forced) else None,
              "normal_positive_recall_at_.90": float(np.mean(np.concatenate(hard_normal) >= .90)) if hard_normal and any(len(x) for x in hard_normal) else None,
              "truth_forced_positive_recall_at_.90": float(np.mean(np.concatenate(hard_forced) >= .90)) if hard_forced and any(len(x) for x in hard_forced) else None}
    return result


def _no_identity_control(bundle: dict[str, Any]) -> dict[str, Any]:
    """Verify that the primary score is invariant to the two identity fields.

    The public score still receives the physical mass, geometry and embedding.
    This controlled pair keeps those quantities fixed while changing the
    identity-overlap and target-disagreement columns.  It is a mechanism
    check, not a task metric; the archived hard-positive audit below remains
    the evidence about real retrieval pairs.
    """
    agreement = np.asarray([.72, .82, .31, .20, .25, .91, .88, .08, .99, .00], dtype=np.float64)
    conflict = agreement.copy()
    conflict[8] = .40
    conflict[9] = .59
    scores = edge_probabilities(np.vstack((agreement, conflict)), bundle)
    return {
        "description": "same geometry/embedding/physical mass; role_overlap and target_probability_disagreement changed",
        "agreement_score": float(scores[0]), "conflict_score": float(scores[1]),
        "absolute_score_delta": float(abs(scores[0] - scores[1])),
        "invariant": bool(np.allclose(scores[0], scores[1], rtol=0., atol=1.e-12)),
        "zeroed_features": ["role_overlap", "target_probability_disagreement"],
    }


def _v6_edge_audit(bundle: dict[str, Any], paths: list[Path]) -> dict[str, Any]:
    """Compare old v5 and v8 edge operating points on v6 test-only archives.

    Hard-positive recall means a same-instance retrieved pair with target unary
    disagreement >= .20 that passes the edge cutoff.  For hard negatives the
    meaningful analogue is rejection (specificity); both rejection and false
    acceptance are emitted explicitly.  Forest counts are included to separate
    a score threshold failure from later one-to-one/topology pruning.
    """
    old_bundle = read_json(V5_RUN / "reader_freeze.json")["edge_models"]["prematch"]
    old_threshold = float(old_bundle.get("threshold", .90))
    new_threshold = float(bundle["threshold"])
    counters: dict[str, int] = {
        "retrieved_rows": 0, "retrieved_positive": 0, "hard_positive": 0,
        "hard_negative": 0, "old_hard_positive_pass": 0, "new_hard_positive_pass": 0,
        "old_hard_negative_reject": 0, "new_hard_negative_reject": 0,
        "old_hard_negative_accept": 0, "new_hard_negative_accept": 0,
        "old_forest_kept_edges": 0, "new_forest_kept_edges": 0,
        "old_forest_true_kept_edges": 0, "new_forest_true_kept_edges": 0,
        "old_hard_positive_forest_kept": 0, "new_hard_positive_forest_kept": 0,
    }
    old_tp = old_fp = new_tp = new_fp = 0
    old_selected = new_selected = 0
    old_positive_missed = new_positive_missed = 0
    layouts: set[str] = set(); cases_seen: set[str] = set()
    for path in paths:
        _, cases = load_shard(path.parent)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in (*PUBLIC_FIELDS, "instance", "offsets")}
            descriptions = [(case, history) for case in cases for history in case["histories"]]
            for (lo, hi), (case, _desc) in zip(zip(arrays["offsets"][:-1], arrays["offsets"][1:]), descriptions, strict=True):
                layout = str(case.get("layout_id", case.get("identifier", "unknown")))
                case_obj = case.get("case", case)
                layouts.add(layout); cases_seen.add(str(case_obj.get("identifier", layout)))
                public = {key: arrays[key][lo:hi] for key in PUBLIC_FIELDS}
                instance = arrays["instance"][lo:hi]
                p, embedding = heads(public["features"], _BASE_BUNDLE["model"])
                pairs = retrieve(public)
                if not len(pairs):
                    continue
                features = edge_inputs(public, p, embedding, pairs)
                old_scores = edge_probabilities(features, old_bundle)
                new_scores = edge_probabilities(features, bundle)
                left, right = pairs[:, 0], pairs[:, 1]
                positive = (instance[left] >= 0) & (instance[left] == instance[right])
                disagreement = np.abs(p[left, 0] - p[right, 0])
                hard_positive = positive & (disagreement >= .20)
                hard_negative = (~positive) & (features[:, 0] >= .45) & (features[:, 1] >= .45) & (disagreement <= .10)
                old_pass = old_scores >= old_threshold; new_pass = new_scores >= new_threshold
                counters["retrieved_rows"] += len(pairs); counters["retrieved_positive"] += int(positive.sum())
                counters["hard_positive"] += int(hard_positive.sum()); counters["hard_negative"] += int(hard_negative.sum())
                counters["old_hard_positive_pass"] += int(np.sum(hard_positive & old_pass)); counters["new_hard_positive_pass"] += int(np.sum(hard_positive & new_pass))
                counters["old_hard_negative_reject"] += int(np.sum(hard_negative & ~old_pass)); counters["new_hard_negative_reject"] += int(np.sum(hard_negative & ~new_pass))
                counters["old_hard_negative_accept"] += int(np.sum(hard_negative & old_pass)); counters["new_hard_negative_accept"] += int(np.sum(hard_negative & new_pass))
                old_selected += int(old_pass.sum()); new_selected += int(new_pass.sum())
                old_tp += int(np.sum(old_pass & positive)); old_fp += int(np.sum(old_pass & ~positive))
                new_tp += int(np.sum(new_pass & positive)); new_fp += int(np.sum(new_pass & ~positive))
                old_positive_missed += int(np.sum(positive & ~old_pass)); new_positive_missed += int(np.sum(positive & ~new_pass))
                old_state = infer(public, p, embedding, _BASE_BUNDLE["links"], edge_model=old_bundle)
                new_state = infer(public, p, embedding, _BASE_BUNDLE["links"], edge_model=bundle)
                for key, state in (("old", old_state), ("new", new_state)):
                    edges = state["edges"][:, :2].astype(int) if len(state["edges"]) else np.empty((0, 2), dtype=int)
                    counters[f"{key}_forest_kept_edges"] += int(len(edges))
                    if not len(edges):
                        continue
                    edge_positive = (instance[edges[:, 0]] >= 0) & (instance[edges[:, 0]] == instance[edges[:, 1]])
                    counters[f"{key}_forest_true_kept_edges"] += int(edge_positive.sum())
                    hard_pairs = {tuple(map(int, row)) for row in pairs[hard_positive]}
                    counters[f"{key}_hard_positive_forest_kept"] += sum(tuple(map(int, row)) in hard_pairs for row in edges)
    hp = max(counters["hard_positive"], 1); hn = max(counters["hard_negative"], 1)
    return {
        "unique_layouts": len(layouts), "unique_cases": len(cases_seen), "thresholds": {"old_v5": old_threshold, "new_v8": new_threshold},
        "counts": counters,
        "old": {"retrieved_precision": old_tp / max(old_selected, 1), "retrieved_recall": old_tp / max(counters["retrieved_positive"], 1),
                "hard_positive_recall": counters["old_hard_positive_pass"] / hp,
                "hard_negative_rejection": counters["old_hard_negative_reject"] / hn,
                "hard_negative_false_acceptance": counters["old_hard_negative_accept"] / hn,
                "positive_missed": old_positive_missed},
        "new": {"retrieved_precision": new_tp / max(new_selected, 1), "retrieved_recall": new_tp / max(counters["retrieved_positive"], 1),
                "hard_positive_recall": counters["new_hard_positive_pass"] / hp,
                "hard_negative_rejection": counters["new_hard_negative_reject"] / hn,
                "hard_negative_false_acceptance": counters["new_hard_negative_accept"] / hn,
                "positive_missed": new_positive_missed},
    }


def _test_graph(bundle: dict[str, Any], paths: list[Path]) -> dict[str, Any]:
    """Run the exact public v5 forest on v6 snapshots, test-only."""
    if _BASE_BUNDLE is None:
        raise RuntimeError("test graph needs initialized bundle")
    rows = 0; positives = 0; kept = 0; true_kept = 0; scored = []
    for path in paths:
        _, cases = load_shard(path.parent)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in (*PUBLIC_FIELDS, "instance", "offsets")}
            descriptions = [(case, history) for case in cases for history in case["histories"]]
            for (lo, hi), (case, desc) in zip(zip(arrays["offsets"][:-1], arrays["offsets"][1:]), descriptions, strict=True):
                public = {key: arrays[key][lo:hi] for key in PUBLIC_FIELDS}; instance = arrays["instance"][lo:hi]
                p, e = heads(public["features"], _BASE_BUNDLE["model"]); pairs = retrieve(public)
                x = edge_inputs(public, p, e, pairs); probs = edge_probabilities(x, bundle)
                if len(pairs):
                    pos = (instance[pairs[:, 0]] >= 0) & (instance[pairs[:, 0]] == instance[pairs[:, 1]])
                    rows += len(pairs); positives += int(pos.sum()); scored.extend(probs.tolist())
                    state = infer(public, p, e, _BASE_BUNDLE["links"], edge_model=bundle)
                    edges = state["edges"][:, :2].astype(int) if len(state["edges"]) else np.empty((0, 2), dtype=int)
                    kept += len(edges)
                    if len(edges):
                        truth = (instance[edges[:, 0]] >= 0) & (instance[edges[:, 0]] == instance[edges[:, 1]])
                        true_kept += int(truth.sum())
    return {"rows": rows, "positive": positives, "forest_kept_edges": kept, "forest_true_kept_edges": true_kept,
            "forest_precision": true_kept / max(kept, 1), "all_scored_mean": float(np.mean(scored)) if scored else 0.}


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "run")
    parser.add_argument("--atlas-pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    protocol = {"experiment": EXPERIMENT, "base": BASE, "seed": SEED, "used_features": [EDGE_FIELDS[i] for i in USED_FEATURES],
                "identity_disagreement_used": False, "v5_train": str(V5_RUN / "fit_revisit"),
                "v6_test": str(V6_RUN / "reader_fresh"), "v6_is_test_only": True,
                "calibration_operation_sampling": "uniform_over_all_retrieved_pairs",
                "calibration_operation_cap": NATURAL_PANEL_CAP,
                "primary_no_identity_fields": ["role_overlap", "target_probability_disagreement"],
                "atlas_sha256": digest(args.atlas_pack), "v5_source_sha256": digest(V5 / "object_tracks.py"),
                "source_sha256": digest(Path(__file__))}
    protocol_path = out / "edge_protocol.json"
    if protocol_path.exists() and read_json(protocol_path) != protocol:
        raise RuntimeError("Frozen edge protocol differs")
    atomic_json(protocol_path, protocol)
    v5_paths = _paths(V5_RUN / "fit_revisit"); v6_paths = _paths(V6_RUN / "reader_fresh")
    if not v5_paths or not v6_paths:
        raise FileNotFoundError("v5 fit_revisit or v6 reader_fresh shards are missing")
    init_worker(str(args.atlas_pack), str(V6_RUN))
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context, initializer=init_worker,
                             initargs=(str(args.atlas_pack), str(V6_RUN))) as pool:
        batches = list(pool.map(_panel_worker, [str(path) for path in v5_paths], chunksize=1))
    records = _merge_records(batches)
    train_x, train_y, train_stats = _matrix(records, "train")
    cal_x, cal_y, cal_stats = _matrix(records, "calibration", natural_mix=True)
    op_x, op_y, op_stats = _matrix(records, "operation", natural_mix=True)
    bundle, fit_stats = _fit_cuda(train_x, train_y)
    bundle = _calibrate(bundle, cal_x, cal_y)
    threshold, operation = _operation_threshold(bundle, op_x, op_y); bundle["threshold"] = threshold
    bundle.update({"experiment": EXPERIMENT, "protocol_sha256": digest(protocol_path), "calibration_split": cal_stats,
                   "operation_split": op_stats, "train_split": train_stats})
    atomic_json(out / "edge_model_v8.json", bundle)
    scores_cal = edge_probabilities(cal_x, bundle); scores_op = edge_probabilities(op_x, bundle)
    metrics = {"calibration": _metrics(scores_cal, cal_y, threshold), "operation": _metrics(scores_op, op_y, threshold),
               "truth_forced_train": _truth_forced(bundle, records), "fit": fit_stats,
               "operation_point": operation, "hard_cases": {"train": train_stats, "calibration": cal_stats, "operation": op_stats},
               "no_identity_control": _no_identity_control(bundle)}
    atomic_json(out / "edge_training_results.json", metrics)
    # v6 reader-fresh is test-only and is loaded after every learned choice is frozen.
    test = _test_graph(bundle, v6_paths)
    test["v6_edge_audit"] = _v6_edge_audit(bundle, v6_paths)
    atomic_json(out / "edge_test_results.json", test)
    print({"train": train_stats, "calibration": metrics["calibration"], "operation": operation, "test": test}, flush=True)


if __name__ == "__main__":
    main()
