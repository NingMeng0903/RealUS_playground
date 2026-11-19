"""Paired scene-level summaries, with the task and loss strata preserved."""

from collections import Counter

import numpy as np

from benchmark import ARMS, PRIMARY
from policy import p
from run_learned_recovery import write_json


FIELDS = (PRIMARY, "corrected_complete", "final_wrong", "new_identity_mistake", "tcp_mm",
          "turn_degs", "seconds", "frames", "new_query_image_retained", "centered_complete")
BINARY = {PRIMARY, "corrected_complete", "final_wrong", "new_identity_mistake",
          "new_query_image_retained", "centered_complete"}
COMPARISONS = (
    *(("psdp", tag) for tag in ("mpc", "fixed_single", "random_data", "evidence_data", "published_evidence",
                               "contrast_data", "legacy_program", "legacy_precommit", "contrast_child",
                               "psdp_mixed", "psdp_direct", "psdp_precommit")),
    ("contrast_data", "mpc"), ("contrast_data", "random_data"), ("contrast_data", "evidence_data"),
    ("evidence_data", "random_data"), ("contrast_data", "published_evidence"),
    ("legacy_program", "legacy_precommit"), ("contrast_child", "contrast_data"),
    ("matched_q1", "contrast_data"), ("matched_q1_child", "matched_q1"),
    ("matched_q2_child", "matched_q1_child"), ("matched_q2_child", "matched_mixed_child"),
)


def paired_bootstrap(rows):
    rng = np.random.default_rng(24126)
    ids = {}
    for i, row in enumerate(rows):
        ids.setdefault((row["family"], row["context"]), []).append(i)
    return np.concatenate([rng.choice(group, (8000, len(group)), replace=True) for group in ids.values()], axis=1)


def summarize(rows):
    values = {arm: np.asarray([[[t[k] for k in FIELDS] for t in row["outcomes"][arm]] for row in rows])
              for arm in ARMS}
    boot = paired_bootstrap(rows)
    arms = {}
    for arm, array in values.items():
        trials = [t for row in rows for t in row["outcomes"][arm]]
        cpu = [t["decision_cpu_seconds"] for t in trials if t["decision_cpu_seconds"] is not None]
        arms[arm] = {
            "counts": {k: int(array[:, :, i].sum()) for i, k in enumerate(FIELDS) if k in BINARY},
            "means": dict(zip(FIELDS, array.mean(axis=(0, 1)).tolist(), strict=True)),
            "decision_cpu_ms_p50": None if not cpu else 1000 * np.median(cpu),
            "decision_cpu_ms_p95": None if not cpu else 1000 * np.quantile(cpu, .95),
            "planned_adaptive": sum(t["observe_mode"] for t in trials),
            "child_handoff_with_available_moves": sum(
                bool(t["child_alternatives"]) and t["second"] == p.HANDOFF for t in trials),
            "unavailable_second": sum(t["second_unavailable"] for t in trials),
            "actual_probes": dict(Counter(t["probes"] for t in trials)),
            "roots_where_second_changes": sum(
                len({t["second"] for t in row["outcomes"][arm]}) > 1 for row in rows),
            "max_tcp_mm": max(t["tcp_mm"] for t in trials),
            "max_seconds": max(t["seconds"] for t in trials),
            "max_continuity_error": max(t["max_handoff_error"] for t in trials),
        }
    pairs = {}
    for model, reference in COMPARISONS:
        d = values[model] - values[reference]
        clustered = d.mean(axis=1)
        ci = np.quantile(clustered[boot].mean(axis=1), (.025, .975), axis=0)
        pairs[f"{model}_minus_{reference}"] = {
            k: {"mean": float(clustered[:, i].mean()), "ci95": ci[:, i].tolist(),
                "positive_tapes": int((d[:, :, i] > 0).sum()), "negative_tapes": int((d[:, :, i] < 0).sum())}
            for i, k in enumerate(FIELDS)
        }
    return {"cases": len(rows), "tapes_per_case": len(rows[0]["outcomes"]["mpc"]),
            "arms": arms, "pairs": pairs}


def make_summary(out):
    result = {"frozen_sha256": p.base.fingerprint(out / "frozen.json"), "test_hashes": {}}
    for split in ("test", "drift"):
        path = out / f"{split}.jsonl.gz"
        rows = p.base.read_rows(path)
        result["test_hashes"][split] = p.base.fingerprint(path)
        groups = {"all": rows}
        for key in ("family", "context"):
            for value in sorted({row[key] for row in rows}):
                groups[value] = [r for r in rows if r[key] == value]
        result[split] = {name: summarize(cases) for name, cases in groups.items()}
        print(split)
        for arm in ARMS:
            v = result[split]["all"]["arms"][arm]
            print(arm, v["counts"][PRIMARY], v["counts"]["final_wrong"],
                  v["counts"]["new_identity_mistake"], round(v["means"]["tcp_mm"], 2),
                  v["actual_probes"], flush=True)
    write_json(out / "summary.json", result)
