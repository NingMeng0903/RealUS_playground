#!/usr/bin/env python3
"""Summarize paired executable-skill episodes.

The input is one JSON object per line.  The script is deliberately
descriptive: it reports arm/stratum counts, exact paired outcome tables, and
metric differences.  It does not select a preferred arm or turn a censored
episode into a failure.

Required record fields are ``arm``, ``case_id``, ``replicate``, ``stratum``,
and ``outcome``.  The remaining fields are optional numeric measurements.  A
pair is keyed by ``(case_id, replicate, stratum)``; duplicate records and
incomplete required pair sets are errors rather than silently dropped data.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


OUTCOMES = ("correct", "wrong_stop", "unresolved")
OUTCOME_RANK = {"wrong_stop": 0, "unresolved": 1, "correct": 2}
NUMERIC_FIELDS = (
    "frames",
    "returns",
    "retreats",
    "coverage_invalidations",
    "reference_reacquisitions",
    "false_hits",
    "missing",
    "completed_bins",
)
# ``rework`` is a transparent derived quantity.  It sums the rework fields
# available on both sides of a paired record.  The common runner currently
# emits returns/retreats; coverage/reference terms are included when a future
# runner supplies them.
REWORK_FIELDS = (
    "returns",
    "retreats",
    "coverage_invalidations",
    "reference_reacquisitions",
)
PAIRINGS = (
    ("B_actual", "A_task"),
    ("B_actual", "A_fixed_goal"),
    ("B_actual", "A_legacy"),
    ("B_actual", "B_nominal"),
    ("B_actual", "B_no_lookahead"),
    ("B_actual", "B_MAP"),
)
KEY_FIELDS = ("case_id", "replicate", "stratum")
BOOTSTRAP_SEED = 721421
DEFAULT_BOOTSTRAP_SAMPLES = 4000


def _finite_number(value: Any, *, field: str, line_number: int) -> float:
    """Parse a finite scalar numeric field with a useful input error."""
    if isinstance(value, bool):
        raise ValueError(f"line {line_number}: {field} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"line {line_number}: {field} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"line {line_number}: {field} must be finite")
    return number


def _identity_value(value: Any, *, field: str, line_number: int) -> Any:
    """Validate an identity value while preserving its JSON representation."""
    if value is None or isinstance(value, (dict, list, tuple, set)):
        raise ValueError(f"line {line_number}: {field} must be a non-null scalar")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"line {line_number}: {field} must not be empty")
    # JSON scalars are hashable except for the containers rejected above.
    try:
        hash(value)
    except TypeError as exc:
        raise ValueError(f"line {line_number}: {field} must be hashable") from exc
    return value


def _freeze(value: Any) -> tuple[str, Any]:
    """Make exact JSON scalar equality explicit (1 is not the string '1')."""
    return (type(value).__name__, value)


def _sort_token(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _record_key(record: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(_freeze(record[field]) for field in KEY_FIELDS)


def _parse_record(payload: Mapping[str, Any], line_number: int) -> dict[str, Any]:
    required = ("arm", "case_id", "replicate", "stratum", "outcome")
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"line {line_number}: missing required field(s): {', '.join(missing)}")

    arm = payload["arm"]
    if not isinstance(arm, str) or not arm.strip():
        raise ValueError(f"line {line_number}: arm must be a non-empty string")
    outcome = payload["outcome"]
    if outcome not in OUTCOMES:
        raise ValueError(
            f"line {line_number}: outcome must be one of {OUTCOMES}, got {outcome!r}"
        )

    record: dict[str, Any] = {
        "arm": arm,
        "case_id": _identity_value(payload["case_id"], field="case_id", line_number=line_number),
        "replicate": _identity_value(
            payload["replicate"], field="replicate", line_number=line_number
        ),
        "stratum": _identity_value(payload["stratum"], field="stratum", line_number=line_number),
        "outcome": outcome,
    }
    for field in NUMERIC_FIELDS:
        if field not in payload or payload[field] is None:
            continue
        record[field] = _finite_number(payload[field], field=field, line_number=line_number)
    return record


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read and validate an ``EPISODES.jsonl`` file."""
    source = Path(path)
    records: list[dict[str, Any]] = []
    seen: dict[tuple[str, tuple[tuple[str, Any], ...]], int] = {}
    try:
        handle = source.open("r", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read {source}: {exc}") from exc
    with handle:
        for line_number, text in enumerate(handle, start=1):
            if not text.strip():
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"line {line_number}: each JSONL row must be an object")
            record = _parse_record(payload, line_number)
            key = (record["arm"], _record_key(record))
            if key in seen:
                first_line = seen[key]
                raise ValueError(
                    "duplicate record for "
                    f"arm={record['arm']!r}, case_id={record['case_id']!r}, "
                    f"replicate={record['replicate']!r}, stratum={record['stratum']!r} "
                    f"(first seen on line {first_line}, repeated on line {line_number})"
                )
            seen[key] = line_number
            records.append(record)
    return records


def _index_records(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[tuple[tuple[str, Any], ...], Mapping[str, Any]]]:
    indexed: dict[str, dict[tuple[tuple[str, Any], ...], Mapping[str, Any]]] = defaultdict(dict)
    for record in records:
        indexed[str(record["arm"])][_record_key(record)] = record
    return indexed


def _validate_complete_pairs(
    indexed: Mapping[str, Mapping[tuple[tuple[str, Any], ...], Mapping[str, Any]]]
) -> None:
    """Require the exact same identity keys for every requested comparison."""
    for left, right in PAIRINGS:
        if left not in indexed or right not in indexed:
            missing_arms = [arm for arm in (left, right) if arm not in indexed]
            raise ValueError(
                f"incomplete paired set for {left} vs {right}: "
                f"missing arm(s): {', '.join(missing_arms)}"
            )
        left_keys = set(indexed.get(left, {}))
        right_keys = set(indexed.get(right, {}))
        missing_left = sorted(right_keys - left_keys, key=_sort_token)
        missing_right = sorted(left_keys - right_keys, key=_sort_token)
        if missing_left or missing_right:
            details: list[str] = []
            if missing_left:
                details.append(f"missing {left}: {len(missing_left)}")
            if missing_right:
                details.append(f"missing {right}: {len(missing_right)}")
            preview = (missing_left + missing_right)[:3]
            raise ValueError(
                f"incomplete paired set for {left} vs {right}: "
                + ", ".join(details)
                + (f"; example keys={preview!r}" if preview else "")
            )


def _metric_summary(values: Iterable[float]) -> dict[str, Any]:
    values_array = np.asarray(list(values), dtype=float)
    if values_array.size == 0:
        return {"n": 0, "mean": None, "median": None, "sd": None}
    return {
        "n": int(values_array.size),
        "mean": float(np.mean(values_array)),
        "median": float(np.median(values_array)),
        "sd": float(np.std(values_array, ddof=1)) if values_array.size > 1 else 0.0,
        "min": float(np.min(values_array)),
        "max": float(np.max(values_array)),
    }


def _arm_stratum_aggregate(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, tuple[str, Any]], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(str(record["arm"]), _freeze(record["stratum"]))].append(record)

    result: list[dict[str, Any]] = []
    for (arm, _stratum_key), members in sorted(
        groups.items(), key=lambda item: (item[0][0], _sort_token(item[1][0]["stratum"]))
    ):
        n = len(members)
        counts = {outcome: sum(row["outcome"] == outcome for row in members) for outcome in OUTCOMES}
        item: dict[str, Any] = {
            "arm": arm,
            "stratum": members[0]["stratum"],
            "cohort_size": n,
            "n": n,
            "correct_count": counts["correct"],
            "wrong_stop_count": counts["wrong_stop"],
            "unresolved_count": counts["unresolved"],
            "censor_count": counts["unresolved"],
            "outcome_counts": counts,
            "correct_interval": [
                counts["correct"] / n,
                (counts["correct"] + counts["unresolved"]) / n,
            ],
            "correct_rate": counts["correct"] / n,
            "metrics": {},
        }
        for field in NUMERIC_FIELDS:
            values = [float(row[field]) for row in members if field in row]
            if values:
                item["metrics"][field] = _metric_summary(values)
        result.append(item)
    return result


def _outcome_matrix(rows: Sequence[Mapping[str, Any]], left: str, right: str) -> dict[str, dict[str, int]]:
    matrix = {left_outcome: {right_outcome: 0 for right_outcome in OUTCOMES} for left_outcome in OUTCOMES}
    for row in rows:
        matrix[row["left"]["outcome"]][row["right"]["outcome"]] += 1
    return matrix


def _winner_summary(rows: Sequence[Mapping[str, Any]], left: str, right: str) -> dict[str, Any]:
    outcome_left = []
    outcome_right = []
    correctness_left = []
    correctness_right = []
    for row in rows:
        left_outcome = row["left"]["outcome"]
        right_outcome = row["right"]["outcome"]
        outcome_left.append(OUTCOME_RANK[left_outcome])
        outcome_right.append(OUTCOME_RANK[right_outcome])
        correctness_left.append(int(left_outcome == "correct"))
        correctness_right.append(int(right_outcome == "correct"))

    def counts(left_values: Sequence[int], right_values: Sequence[int]) -> dict[str, Any]:
        left_wins = sum(a > b for a, b in zip(left_values, right_values))
        right_wins = sum(a < b for a, b in zip(left_values, right_values))
        ties = sum(a == b for a, b in zip(left_values, right_values))
        # Labels are included alongside left/right so a JSON consumer need not
        # infer which side a count refers to.
        return {
            "left": left_wins,
            "right": right_wins,
            left: left_wins,
            right: right_wins,
            "ties": ties,
        }

    outcome_wins = counts(outcome_left, outcome_right)
    correctness_wins = counts(correctness_left, correctness_right)
    return {
        "wins": outcome_wins,
        "losses": {
            "left": outcome_wins["right"],
            "right": outcome_wins["left"],
            left: outcome_wins["right"],
            right: outcome_wins["left"],
        },
        "ties": outcome_wins["ties"],
        "outcome_order": "correct > unresolved > wrong_stop",
        "correctness_wins": correctness_wins,
        "correctness_losses": {
            "left": correctness_wins["right"],
            "right": correctness_wins["left"],
            left: correctness_wins["right"],
            right: correctness_wins["left"],
        },
        "correctness_ties": correctness_wins["ties"],
    }


def _row_metric_value(record: Mapping[str, Any], metric: str) -> float | None:
    if metric == "rework":
        if not any(field in record for field in REWORK_FIELDS):
            return None
        return float(sum(float(record[field]) for field in REWORK_FIELDS if field in record))
    value = record.get(metric)
    return None if value is None else float(value)


def _delta_values(rows: Sequence[Mapping[str, Any]], metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        if metric == "rework":
            # Optional fields are only compared where both arms recorded the
            # same component; this avoids treating an omitted value as zero.
            common_fields = [
                field
                for field in REWORK_FIELDS
                if field in row["left"] and field in row["right"]
            ]
            if not common_fields:
                continue
            values.append(
                float(
                    sum(float(row["left"][field]) - float(row["right"][field]) for field in common_fields)
                )
            )
            continue
        left_value = _row_metric_value(row["left"], metric)
        right_value = _row_metric_value(row["right"], metric)
        if left_value is not None and right_value is not None:
            values.append(left_value - right_value)
    return values


def _delta_summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for metric in (*NUMERIC_FIELDS, "rework"):
        values = _delta_values(rows, metric)
        if values:
            output[metric] = _metric_summary(values)
    return output


def _bootstrap_means(values: np.ndarray, *, seed: int, samples: int) -> list[float] | None:
    if values.size == 0:
        return None
    rng = np.random.default_rng(seed)
    # Chunk draws to keep memory bounded when a real study has many cases.
    chunks: list[np.ndarray] = []
    chunk_size = 512
    for start in range(0, samples, chunk_size):
        size = min(chunk_size, samples - start)
        indices = rng.integers(0, values.size, size=(size, values.size))
        chunks.append(values[indices].mean(axis=1))
    return np.concatenate(chunks).tolist()


def _case_metric_values(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[tuple[str, Any], list[float]]:
    grouped: dict[tuple[str, Any], list[float]] = defaultdict(list)
    for row in rows:
        value: float | None
        if metric == "correct_indicator":
            value = float(row["left"]["outcome"] == "correct") - float(
                row["right"]["outcome"] == "correct"
            )
        elif metric == "outcome_order":
            value = float(OUTCOME_RANK[row["left"]["outcome"]] - OUTCOME_RANK[row["right"]["outcome"]])
        else:
            deltas = _delta_values([row], metric)
            value = deltas[0] if deltas else None
        if value is not None:
            grouped[_freeze(row["left"]["case_id"])].append(float(value))
    return grouped


def _noise_summary(values: Sequence[float], *, n_repeated_cases: int | None = None) -> dict[str, Any]:
    summary = _metric_summary(values)
    if n_repeated_cases is not None:
        summary["repeated_case_count"] = int(n_repeated_cases)
    if values:
        summary["mean_abs"] = float(np.mean(np.abs(np.asarray(values, dtype=float))))
    return summary


def _uncertainty(
    rows: Sequence[Mapping[str, Any]], *, seed: int, samples: int
) -> dict[str, Any]:
    metrics = ("correct_indicator", "outcome_order", "frames", "rework")
    output: dict[str, Any] = {
        "bootstrap_unit": "case_id",
        "replicate_and_stratum_rows_are_not_independent_bootstrap_units": True,
        "bootstrap_seed": int(seed),
        "bootstrap_samples": int(samples),
        "n_paired_records": len(rows),
        "n_independent_cases": len({_freeze(row["left"]["case_id"]) for row in rows}),
        "estimates": {},
        "structural_case_noise": {},
        "same_fixture_paired_noise": {},
    }
    for metric in metrics:
        grouped = _case_metric_values(rows, metric)
        case_means = np.asarray([np.mean(values) for values in grouped.values()], dtype=float)
        if case_means.size:
            draws = _bootstrap_means(case_means, seed=seed + metrics.index(metric), samples=samples)
            assert draws is not None
            output["estimates"][metric] = {
                "estimate": float(np.mean(case_means)),
                "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
                "n_cases": int(case_means.size),
            }
            output["structural_case_noise"][metric] = _noise_summary(case_means.tolist())
        else:
            output["estimates"][metric] = {"estimate": None, "ci95": None, "n_cases": 0}
            output["structural_case_noise"][metric] = _noise_summary([])

        # A same-fixture/same-case replicate contributes residual noise only
        # after subtracting that case's mean; it is reported separately from
        # the case-level bootstrap uncertainty.
        residuals: list[float] = []
        repeated_case_count = 0
        for values in grouped.values():
            if len(values) < 2:
                continue
            repeated_case_count += 1
            mean = float(np.mean(values))
            residuals.extend(float(value - mean) for value in values)
        output["same_fixture_paired_noise"][metric] = _noise_summary(
            residuals, n_repeated_cases=repeated_case_count
        )
    return output


def _paired_rows(
    left_records: Mapping[tuple[tuple[str, Any], ...], Mapping[str, Any]],
    right_records: Mapping[tuple[tuple[str, Any], ...], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(left_records, key=_sort_token):
        rows.append({"key": key, "left": left_records[key], "right": right_records[key]})
    return rows


def _subset_summary(rows: Sequence[Mapping[str, Any]], left: str, right: str) -> dict[str, Any]:
    counts = _winner_summary(rows, left, right)
    summary: dict[str, Any] = {
        "cohort_size": len(rows),
        "n_pairs": len(rows),
        "n_cases": len({_freeze(row["left"]["case_id"]) for row in rows}),
        "outcomes": _outcome_matrix(rows, left, right),
        "outcome_pairs": _outcome_matrix(rows, left, right),
        "wins": counts["wins"],
        "losses": counts["losses"],
        "ties": counts["ties"],
        "correctness_wins": counts["correctness_wins"],
        "correctness_losses": counts["correctness_losses"],
        "correctness_ties": counts["correctness_ties"],
        "metric_deltas": _delta_summaries(rows),
    }
    if "frames" in summary["metric_deltas"]:
        summary["frame_delta"] = summary["metric_deltas"]["frames"]
    if "rework" in summary["metric_deltas"]:
        summary["rework_delta"] = summary["metric_deltas"]["rework"]
    return summary


def _pair_summary(
    rows: Sequence[Mapping[str, Any]],
    left: str,
    right: str,
    *,
    seed: int,
    samples: int,
    include_by_stratum: bool = True,
) -> dict[str, Any]:
    counts = _winner_summary(rows, left, right)
    all_records = _subset_summary(rows, left, right)
    common_correct_rows = [
        row
        for row in rows
        if row["left"]["outcome"] == "correct" and row["right"]["outcome"] == "correct"
    ]
    common_correct = _subset_summary(common_correct_rows, left, right)
    output: dict[str, Any] = {
        "left_arm": left,
        "right_arm": right,
        "key_fields": list(KEY_FIELDS),
        "n_pairs": len(rows),
        "n_cases": len({_freeze(row["left"]["case_id"]) for row in rows}),
        "strata": sorted(
            {row["left"]["stratum"] for row in rows}, key=_sort_token
        ),
        "outcomes": _outcome_matrix(rows, left, right),
        "wins": counts["wins"],
        "losses": counts["losses"],
        "ties": counts["ties"],
        "correctness_wins": counts["correctness_wins"],
        "correctness_losses": counts["correctness_losses"],
        "correctness_ties": counts["correctness_ties"],
        "outcome_order": counts["outcome_order"],
        "all_records": all_records,
        "common_correct_subset": common_correct,
        "uncertainty": _uncertainty(rows, seed=seed, samples=samples),
    }
    # These aliases make the requested frame/rework summaries easy to find
    # without duplicating the full metric table.
    if "frames" in all_records["metric_deltas"]:
        output["frame_delta"] = all_records["metric_deltas"]["frames"]
    if "rework" in all_records["metric_deltas"]:
        output["rework_delta"] = all_records["metric_deltas"]["rework"]
    if "frames" in common_correct["metric_deltas"]:
        output["common_correct_frame_delta"] = common_correct["metric_deltas"]["frames"]
    if "rework" in common_correct["metric_deltas"]:
        output["common_correct_rework_delta"] = common_correct["metric_deltas"]["rework"]

    if include_by_stratum:
        by_stratum: list[dict[str, Any]] = []
        grouped: dict[tuple[str, Any], list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[_freeze(row["left"]["stratum"])].append(row)
        for _stratum_key, members in sorted(grouped.items(), key=lambda item: _sort_token(item[1][0]["left"]["stratum"])):
            by_stratum.append(
                _pair_summary(
                    members,
                    left,
                    right,
                    seed=seed,
                    samples=samples,
                    include_by_stratum=False,
                )
            )
        output["by_stratum"] = by_stratum
    return output


def summarize(
    episodes_path: str | Path,
    *,
    seed: int = BOOTSTRAP_SEED,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    """Return a descriptive paired aggregate for an ``EPISODES.jsonl`` file."""
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    source = Path(episodes_path).resolve()
    records = read_records(source)
    if not records:
        raise ValueError("EPISODES.jsonl contains no records")
    indexed = _index_records(records)
    _validate_complete_pairs(indexed)

    paired: dict[str, Any] = {}
    for left, right in PAIRINGS:
        rows = _paired_rows(indexed[left], indexed[right])
        paired[f"{left}_vs_{right}"] = _pair_summary(
            rows, left, right, seed=seed, samples=bootstrap_samples
        )

    return {
        "schema": "ab_executable_skills_pair_summary_v1",
        "descriptive_only": True,
        "input": str(source),
        "record_count": len(records),
        "arms": sorted({str(record["arm"]) for record in records}),
        "strata": sorted({record["stratum"] for record in records}, key=_sort_token),
        "cohorts": _arm_stratum_aggregate(records),
        "paired": paired,
        "notes": {
            "unresolved_retained": True,
            "correct_interval": "[correct/n, (correct+unresolved)/n]",
            "pair_key": "(case_id, replicate, stratum)",
            "bootstrap_unit": "case_id; replicate rows are not independent fixtures",
            "outcome_wins": "descriptive ordinal order correct > unresolved > wrong_stop",
            "rework_definition": "sum of shared available returns, retreats, coverage_invalidations, and reference_reacquisitions; omitted when no rework component is present on both arms",
            "no_subjective_verdict": True,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", type=Path, help="EPISODES.jsonl")
    parser.add_argument("--output", "-o", type=Path, required=True, help="output SUMMARY.json")
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    args = parser.parse_args(argv)
    try:
        result = summarize(
            args.episodes,
            seed=args.seed,
            bootstrap_samples=args.bootstrap_samples,
        )
        text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"wrote {args.output} ({result['record_count']} records, "
        f"{len(result['paired'])} paired comparisons)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
