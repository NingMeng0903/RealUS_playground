"""Compare the frozen coarse and dense follow-up arms on paired held-out objects."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_stress import source_and_stop
from audit_query_support import oracle_support


METRICS = ("success", "false_stop", "restricted_time_s")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_arm(path):
    manifest = json.loads((path / "manifest.json").read_text())
    for file, expected in manifest["source_sha256"].items():
        if sha(file) != expected:
            raise ValueError(f"unexpected source content in {file}")
    for entry in manifest["artifacts"].values():
        if sha(path / entry["file"]) != entry["sha256"]:
            raise ValueError(f"unexpected artifact content in {entry['file']}")
    rows = [json.loads(line) for line in (path / "rows.jsonl").read_text().splitlines()]
    if len(rows) != manifest["rows"]:
        raise ValueError("episode count differs from manifest")
    with np.load(path / "model_inputs.npz") as data:
        model = {
            key: data[key] for key in (
                "poses", "correct_joint", "likelihood", "available",
                "context_ids", "action_cost_s", "initial_index",
            )
        }
    with np.load(path / "renderer_audit.npz") as data:
        private = {key: data[key] for key in ("world_points", "uniforms", "joint", "target_qualified")}
    return manifest, rows, model, private


def confirm_pairing(coarse, dense):
    c_man, _, c_model, c_private = coarse
    d_man, _, d_model, d_private = dense
    for name in ("objects", "contexts", "regimes", "methods", "motion", "world_seed"):
        if c_man[name] != d_man[name]:
            raise ValueError(f"unpaired cohort field {name}")
    for name in ("uniforms", "world_points"):
        if not np.array_equal(c_private[name], d_private[name]):
            raise ValueError(f"unpaired renderer field {name}")
    pose_lookup = {tuple(p): i for i, p in enumerate(d_model["poses"])}
    old = np.array([pose_lookup[tuple(p)] for p in c_model["poses"]])
    checks = [
        np.array_equal(c_model["available"], d_model["available"][:, old]),
        np.array_equal(c_model["action_cost_s"], d_model["action_cost_s"][:, old][:, :, old]),
        np.array_equal(c_model["initial_index"], np.searchsorted(old, d_model["initial_index"])),
        np.array_equal(c_model["likelihood"], d_model["likelihood"][:, :, old]),
        np.array_equal(c_model["correct_joint"], d_model["correct_joint"][:, :, old]),
        np.array_equal(c_private["joint"], d_private["joint"][:, :, old]),
        np.array_equal(c_private["target_qualified"], d_private["target_qualified"][:, old]),
    ]
    if not all(checks):
        raise ValueError(f"common pose dynamics/emission mismatch: {checks}")
    return {"independent_objects": c_man["objects"], "same_worlds_and_random_numbers": True,
            "common_endpoint_mask_cost_and_likelihood_exact": True}


def paired_gains(coarse, dense):
    left = {(row["object_id"], row["context"], row["method"]): row for row in coarse}
    right = {(row["object_id"], row["context"], row["method"]): row for row in dense}
    if left.keys() != right.keys():
        raise ValueError("episode keys do not pair")
    objects = sorted({obj for obj, _, _ in left})
    contexts = sorted({ctx for _, ctx, _ in left})
    methods = sorted({method for _, _, method in left})
    indices = np.random.default_rng(721130).integers(len(objects), size=(4000, len(objects)))
    result = []
    for method in methods:
        for name, group in [(name, [name]) for name in contexts] + [("pooled", contexts)]:
            matrix = np.asarray([
                [
                    [
                        float(right[obj, ctx, method][metric]) - float(left[obj, ctx, method][metric])
                        for metric in METRICS
                    ]
                    for ctx in group
                ]
                for obj in objects
            ])
            paired = matrix.mean(axis=1)
            bounds = np.quantile(paired[indices].mean(axis=1), [.025, .975], axis=0).T
            result.append({
                "method": method, "context": name,
                "independent_objects": len(objects),
                "dense_only_correct_count": int((matrix[..., 0] > 0).sum()),
                "coarse_only_correct_count": int((matrix[..., 0] < 0).sum()),
                "dense_minus_coarse": {
                    k: {"mean": float(value), "object_bootstrap_95": interval.tolist()}
                    for k, value, interval in zip(METRICS, paired.mean(axis=0), bounds)
                },
            })
    return result


def acquisition_stages(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    result = {}
    for method, group in grouped.items():
        result[method] = {
            "episodes": len(group),
            "with_qualified_endpoint": sum(r["oracle"]["qualified_candidate_count"] > 0 for r in group),
            "visited_qualified_endpoint": sum(
                any(b["audit"]["endpoint_qualified"] for b in r["blocks"]) for r in group
            ),
            "obtained_correct_public_gate": sum(
                any(b["audit"]["correct_if_stopped"] for b in r["blocks"]) for r in group
            ),
            "obtained_but_no_correct_stop": sum(
                not r["success"] and any(b["audit"]["correct_if_stopped"] for b in r["blocks"]) for r in group
            ),
            "correct_stops": sum(r["success"] for r in group),
        }
    return result


def exact_paired_success_interval(rows, left, right):
    """Convolve the empirical object-level integer contrast distribution."""
    by_object = defaultdict(int)
    contexts = set()
    for row in rows:
        if row["method"] not in (left, right):
            continue
        contexts.add(row["context"])
        sign = 1 if row["method"] == left else -1
        by_object[row["object_id"]] += sign * int(row["success"])
    totals = np.array(list(by_object.values()))
    size = len(totals)
    minimum = int(totals.min())
    empirical = np.bincount(totals - minimum) / size
    distribution = np.array([1.0])
    for _ in range(size):
        distribution = np.convolve(distribution, empirical)
    indices = np.searchsorted(distribution.cumsum(), [.025, .975])
    return {
        "left": left, "right": right, "objects": size, "placements": len(contexts),
        "delta_success": float(totals.mean() / len(contexts)),
        "object_bootstrap_exact_95": ((indices + size * minimum) / (size * len(contexts))).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coarse", type=Path)
    parser.add_argument("dense", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    c = load_arm(args.coarse)
    d = load_arm(args.dense)
    result = {
        "pairing": confirm_pairing(c, d),
        "scope": "Independent candidate-library follow-up on the declared synthetic observation model. Supports positive correct-stop query is an upper bound, and poses have static robot witnesses.",
        "coarse": {"query_support": oracle_support(c[2], c[3], c[2]["correct_joint"]),
                   "source_and_stop": source_and_stop(c[1]), "acquisition_stages": acquisition_stages(c[1])},
        "dense": {"query_support": oracle_support(d[2], d[3], d[2]["correct_joint"]),
                  "source_and_stop": source_and_stop(d[1]), "acquisition_stages": acquisition_stages(d[1])},
        "paired_gains": paired_gains(c[1], d[1]),
        "dense_paired_success": exact_paired_success_interval(d[1], "B_two", "A_task"),
    }
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
