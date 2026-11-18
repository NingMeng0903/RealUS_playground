"""Train and evaluate the three-source second-stage B selector.

This runner deliberately keeps the experiment policy in ``experiment.py``.  The
adapter owns the ultrasound world, the frozen ProDMP action library, the public
457-dimensional feature construction and the closed-loop executor.  This file
only owns split construction, label assembly, fitting, fixed-action selection,
and statistical reporting.

The stages are explicit:

``collect``
    Generate public training states and paired outcomes for all nominated
    source actions.  Every source sees the same state archive and retains all
    successful, failed and mistaken trials.
``train``
    Append exactly four nominated actions per state to the frozen historical
    B training rows and fit one model per source with the original
    ``learned_recovery.fit_targets`` implementation.
``qualify``
    Evaluate fixed requests on a distinct development split and freeze one
    request per family.  An unavailable request is scored using that row's
    measured MPC fallback, and availability is reported rather than silently
    imputed.
``test``
    Evaluate a fresh paired split with the same initial states and disturbances
    across all arms.
``all``
    Run the four stages in order.

The adapter API is intentionally strict.  A missing field or a changed schema
raises instead of silently falling back to a weaker experiment.
"""

from __future__ import annotations

import argparse
import functools
import gzip
import hashlib
import importlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    EXP = importlib.import_module("experiment")
except Exception as exc:  # pragma: no cover - a clear error is more useful below
    raise RuntimeError("experiment.py must be present beside run.py") from exc

# ``experiment`` imports the generator adapter as a module, but the original
# archive/diagnostic helpers live one level further down in its source
# experiment.  Add those exact frozen paths before importing the helpers; do
# not fall back to an arbitrary same-named module elsewhere on PYTHONPATH.
for _source_path in (
    getattr(EXP, "GENERATOR", None),
    getattr(getattr(EXP, "generator", None), "EXPERIMENT", None),
):
    if _source_path is not None and str(_source_path) not in sys.path:
        sys.path.insert(0, str(_source_path))

# These are the repository's original training/archive utilities.  They are
# imported after ``experiment`` has put the frozen generator source on
# sys.path, so the run cannot accidentally use a duplicate helper.
from diagnose_b_selection import run_parallel, save_archive
from learned_recovery import fit_targets
from run_learned_recovery import fingerprint, read_rows, write_json


SOURCES = tuple(getattr(EXP, "SOURCES", ("random", "direct", "evidence")))
if SOURCES != ("random", "direct", "evidence"):
    raise RuntimeError(f"unexpected source order {SOURCES!r}; expected random/direct/evidence")

CELLS = tuple(getattr(EXP, "CELLS", ()))
FIELDS = tuple(getattr(EXP, "FIELDS", ()))
VISIBLE_TARGETS = tuple(getattr(EXP, "VISIBLE_TARGETS", ()))
PROPOSALS = int(getattr(EXP, "PROPOSALS", 4))
LABEL_TAPES = int(getattr(EXP, "LABEL_TAPES", 8))
DEV_TAPES = int(getattr(EXP, "DEV_TAPES", 4))
TEST_TAPES = int(getattr(EXP, "TEST_TAPES", 8))

if len(CELLS) == 0:
    raise RuntimeError("experiment.CELLS is empty")
if not FIELDS:
    raise RuntimeError("experiment.FIELDS is empty")
if VISIBLE_TARGETS != (
    "corrected_complete", "final_wrong", "new_identity_mistake", "tcp_mm", "turn_degs", "seconds"
):
    raise RuntimeError(
        "experiment.VISIBLE_TARGETS must equal learned_recovery.VISIBLE_TARGETS "
        "(corrected_complete, final_wrong, new_identity_mistake, tcp_mm, turn_degs, seconds)"
    )
if PROPOSALS != 4 or LABEL_TAPES != 8 or DEV_TAPES != 4 or TEST_TAPES != 8:
    raise RuntimeError("the protocol requires 4 proposals, 8 label tapes, 4 dev tapes and 8 test tapes")


FIT_FEATURES = 457
N_TARGETS = len(VISIBLE_TARGETS)
FULL_TRAIN_ROOTS = 128
FULL_DEV_ROOTS = 32
FULL_TEST_ROOTS = 128
SMOKE_TRAIN_ROOTS = 4
SMOKE_DEV_ROOTS = 4
SMOKE_TEST_ROOTS = 4

# Distinct ranges are part of the frozen protocol; smoke uses a different
# million-range so it can never be confused with a full run.
SEEDS = {
    "train": 74_100_000,
    "dev": 74_200_000,
    "test": 74_300_000,
}
SMOKE_SEEDS = {
    "train": 75_100_000,
    "dev": 75_200_000,
    "test": 75_300_000,
}

ARMS = ("mpc", "best_fixed", "old_b", "random_b", "direct_b", "evidence_b")
SOURCES_TO_ARMS = {source: f"{source}_b" for source in SOURCES}
REFERENCE_ARMS = ("random_b", "old_b", "mpc", "best_fixed")
REPORT_METRICS = tuple(dict.fromkeys((*VISIBLE_TARGETS, *FIELDS)))


def _require(name: str):
    value = getattr(EXP, name, None)
    if value is None or not callable(value):
        raise RuntimeError(f"experiment.py must export callable {name}()")
    return value


configure = _require("configure")
scenario = _require("scenario")
collect_one = _require("collect")
base_training = _require("base_training")
configure_models = _require("configure_models")
qualify_one = _require("qualify")
test_one = _require("test")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return [_jsonable(item) for item in sorted(value, key=repr)]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_hash(path: Path) -> str:
    return fingerprint(path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use the repository's canonical writer, which has the same JSON handling
    # as the original recovery experiment.
    write_json(path, _jsonable(value))


def _write_jsonl_gz(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_archive(path, list(rows))


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return list(read_rows(path))


def _require_out(out: Path, smoke: bool) -> None:
    out = out.resolve()
    marker = out / "RUN_CONFIG.json"
    if marker.exists():
        old = json.loads(marker.read_text())
        if bool(old.get("smoke")) != bool(smoke):
            raise RuntimeError(f"refusing to mix smoke and full outputs in {out}")


def _root_count(split: str, smoke: bool) -> int:
    full = {"train": FULL_TRAIN_ROOTS, "dev": FULL_DEV_ROOTS, "test": FULL_TEST_ROOTS}
    tiny = {"train": SMOKE_TRAIN_ROOTS, "dev": SMOKE_DEV_ROOTS, "test": SMOKE_TEST_ROOTS}
    return (tiny if smoke else full)[split]


def _split_seed(split: str, cell_index: int, index: int, smoke: bool) -> int:
    bases = SMOKE_SEEDS if smoke else SEEDS
    # The cell stride is deliberately larger than every split root count.
    return int(bases[split] + cell_index * 10_000 + index)


def _cell_label(cell: Any) -> str:
    return "/".join(str(x) for x in cell) if isinstance(cell, (tuple, list)) else str(cell)


def _jobs(split: str, smoke: bool) -> list[tuple[int, Any, int, bool]]:
    count = _root_count(split, smoke)
    jobs: list[tuple[int, Any, int, bool]] = []
    for cell_index, cell in enumerate(CELLS):
        for index in range(count // len(CELLS)):
            seed = _split_seed(split, cell_index, index, smoke)
            jobs.append((cell_index, cell, seed, smoke))
    if len(jobs) != count:
        raise RuntimeError(f"{split} requires {count} roots but CELLS={len(CELLS)} cannot divide it")
    return jobs


def _scenario_job(job: tuple[int, Any, int, bool]) -> dict[str, Any]:
    cell_index, cell, seed, smoke = job
    row = scenario(cell, seed)
    if not isinstance(row, Mapping):
        raise TypeError("experiment.scenario() must return a mapping")
    output = dict(row)
    output["_split_cell_index"] = cell_index
    output["_split_cell"] = _jsonable(cell)
    output["_split_seed"] = seed
    output["_smoke"] = bool(smoke)
    return output


def _collect_job(job: tuple[int, Any, int, bool]) -> dict[str, Any]:
    return dict(collect_one(_scenario_job(job)))


def _qualify_job(job: tuple[int, Any, int, bool]) -> dict[str, Any]:
    return dict(qualify_one(_scenario_job(job)))


def _test_job(fixed: Mapping[str, str], job: tuple[int, Any, int, bool]) -> dict[str, Any]:
    row = _scenario_job(job)
    family = _family_of_row(row)
    if family not in fixed:
        raise RuntimeError(f"no frozen best-fixed request for family {family!r}")
    return dict(test_one(row, fixed[family]))


def _family_of_row(row: Mapping[str, Any]) -> str:
    if "family" in row:
        return str(row["family"])
    cell = row.get("_split_cell")
    if isinstance(cell, (tuple, list)) and cell:
        return str(cell[0])
    raise RuntimeError("row has no public family field")


def _context_of_row(row: Mapping[str, Any]) -> str:
    if "context" in row:
        return str(row["context"])
    cell = row.get("_split_cell")
    if isinstance(cell, (tuple, list)) and len(cell) > 1:
        return str(cell[1])
    return "unknown"


def _validate_collection_row(row: Mapping[str, Any]) -> None:
    required = ("names", "incumbent", "features", "nominations", "outcomes")
    missing = [key for key in required if key not in row]
    if missing:
        raise RuntimeError(f"collect row {row.get('_split_seed')} missing {missing}")
    names = list(row["names"])
    if len(names) < PROPOSALS + 1 or len(set(names)) != len(names):
        raise RuntimeError("collect row names must be unique and include four candidates plus incumbent")
    incumbent = str(row["incumbent"])
    if incumbent not in names:
        raise RuntimeError(f"incumbent {incumbent!r} is absent from names")
    features = np.asarray(row["features"], dtype=float)
    if features.shape != (len(names), FIT_FEATURES):
        raise RuntimeError(
            f"collect row {row.get('_split_seed')} features shape {features.shape}; "
            f"expected {(len(names), FIT_FEATURES)}"
        )
    nominations = row["nominations"]
    if not isinstance(nominations, Mapping):
        raise RuntimeError("nominations must be a mapping source -> four request ids")
    outcomes = row["outcomes"]
    if not isinstance(outcomes, Mapping):
        raise RuntimeError("outcomes must be a mapping request id -> trial dictionaries")
    for source in SOURCES:
        chosen = list(nominations.get(source, ()))
        if len(chosen) != PROPOSALS or len(set(chosen)) != PROPOSALS:
            raise RuntimeError(f"{source} nomination count is not exactly {PROPOSALS}")
        if incumbent in chosen or any(name not in names for name in chosen):
            raise RuntimeError(f"invalid {source} nominations in row {row.get('_split_seed')}")
    for name in names:
        if name not in outcomes:
            raise RuntimeError(f"missing outcomes for {name!r}")


def _trial_dicts(value: Any, *, request_id: str, source: str | None = None) -> list[dict[str, float]]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"outcomes[{request_id!r}] must be a nonempty list of trial records")
    result: list[dict[str, float]] = []
    for index, trial in enumerate(value):
        if isinstance(trial, Mapping):
            missing = [field for field in VISIBLE_TARGETS if field not in trial]
            if missing:
                raise RuntimeError(f"{request_id!r} trial {index} missing {missing}")
            record = {field: float(trial[field]) for field in VISIBLE_TARGETS}
        elif isinstance(trial, (list, tuple)) and len(trial) == len(FIELDS):
            mapping = dict(zip(FIELDS, trial))
            missing = [field for field in VISIBLE_TARGETS if field not in mapping]
            if missing:
                raise RuntimeError(f"{request_id!r} positional trial missing {missing}")
            record = {field: float(mapping[field]) for field in VISIBLE_TARGETS}
        else:
            raise RuntimeError(
                f"{request_id!r} trial {index} must be a metric dictionary or a full FIELDS-length vector"
            )
        if not all(math.isfinite(v) for v in record.values()):
            raise RuntimeError(f"non-finite outcome in {request_id!r} trial {index}")
        result.append(record)
    return result


def _mean_trial(trials: Sequence[Mapping[str, float]]) -> np.ndarray:
    return np.asarray([np.mean([float(trial[field]) for trial in trials]) for field in VISIBLE_TARGETS], dtype=float)


def _report_trial_dicts(value: Any, *, request_id: str) -> list[dict[str, float]]:
    """Parse the complete archived trial schema used by the final report."""
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"outcomes[{request_id!r}] must be a nonempty list of trial records")
    result: list[dict[str, float]] = []
    for index, trial in enumerate(value):
        if isinstance(trial, Mapping):
            missing = [field for field in REPORT_METRICS if field not in trial]
            if missing:
                raise RuntimeError(f"{request_id!r} trial {index} missing report fields {missing}")
            record = {field: float(trial[field]) for field in REPORT_METRICS}
        elif isinstance(trial, (list, tuple)) and len(trial) == len(FIELDS):
            mapping = dict(zip(FIELDS, trial))
            missing = [field for field in REPORT_METRICS if field not in mapping]
            if missing:
                raise RuntimeError(f"{request_id!r} positional trial missing report fields {missing}")
            record = {field: float(mapping[field]) for field in REPORT_METRICS}
        else:
            raise RuntimeError(f"{request_id!r} trial {index} is not a full metric dictionary/vector")
        if not all(math.isfinite(v) for v in record.values()):
            raise RuntimeError(f"non-finite report outcome in {request_id!r} trial {index}")
        result.append(record)
    return result


def _training_extension(rows: Sequence[Mapping[str, Any]], source: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    audit: list[dict[str, Any]] = []
    expected_tapes = LABEL_TAPES
    # run_parallel is intentionally allowed to return in completion order;
    # fitting must nevertheless have a canonical state/action order.  Within a
    # state, retain the incumbent first and sort the four nominated IDs.  The
    # incumbent row is explicit (zero paired delta) so every source contributes
    # exactly (4 candidates + 1 MPC) rows per root.
    ordered_rows = sorted(rows, key=lambda row: int(row["_split_seed"]))
    for row in ordered_rows:
        _validate_collection_row(row)
        names = list(row["names"])
        name_to_index = {name: i for i, name in enumerate(names)}
        x = np.asarray(row["features"], dtype=float)
        incumbent = str(row["incumbent"])
        base_trials = _trial_dicts(row["outcomes"][incumbent], request_id=incumbent)
        if len(base_trials) != expected_tapes:
            raise RuntimeError(
                f"row {row.get('_split_seed')} incumbent has {len(base_trials)} tapes; expected {expected_tapes}"
            )
        incumbent_mean = _mean_trial(base_trials)
        canonical_ids = [incumbent, *sorted(str(request_id) for request_id in row["nominations"][source])]
        for request_id in canonical_ids:
            if request_id == incumbent:
                xs.append(x[name_to_index[incumbent]])
                ys.append(np.zeros(N_TARGETS, dtype=float))
                audit.append({
                    "split_seed": int(row["_split_seed"]),
                    "family": _family_of_row(row),
                    "context": _context_of_row(row),
                    "source": source,
                    "request_id": incumbent,
                    "incumbent": incumbent,
                    "tapes": len(base_trials),
                    "target_mean_delta": [0.] * N_TARGETS,
                })
                continue
            trials = _trial_dicts(row["outcomes"][request_id], request_id=request_id, source=source)
            if len(trials) != expected_tapes:
                raise RuntimeError(
                    f"row {row.get('_split_seed')} {source}/{request_id} has {len(trials)} tapes; "
                    f"expected {expected_tapes}"
                )
            index = name_to_index[request_id]
            xs.append(x[index])
            ys.append(_mean_trial(trials) - incumbent_mean)
            audit.append({
                "split_seed": int(row["_split_seed"]),
                "family": _family_of_row(row),
                "context": _context_of_row(row),
                "source": source,
                "request_id": request_id,
                "incumbent": incumbent,
                "tapes": len(trials),
                "target_mean_delta": (ys[-1]).tolist(),
            })
    if not xs:
        raise RuntimeError(f"no new training rows for source {source}")
    return np.stack(xs), np.stack(ys), audit


def _load_manifest(out: Path) -> dict[str, Any]:
    path = out / "RUN_CONFIG.json"
    if not path.exists():
        raise RuntimeError(f"missing run manifest {path}; run collect first")
    return json.loads(path.read_text())


def _save_manifest(out: Path, smoke: bool, adapter_hashes: Mapping[str, str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "protocol": "ab_teacher_b_training_20261118",
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": _file_hash(Path(__file__).resolve()),
        "adapter_hashes": dict(adapter_hashes),
        "sources": list(SOURCES),
        "cells": [_jsonable(cell) for cell in CELLS],
        "fields": list(FIELDS),
        "visible_targets": list(VISIBLE_TARGETS),
        "proposals": PROPOSALS,
        "label_tapes": LABEL_TAPES,
        "dev_tapes": DEV_TAPES,
        "test_tapes": TEST_TAPES,
        "feature_dim": FIT_FEATURES,
        "train_roots": _root_count("train", smoke),
        "dev_roots": _root_count("dev", smoke),
        "test_roots": _root_count("test", smoke),
        "seeds": dict(SMOKE_SEEDS if smoke else SEEDS),
        "smoke": bool(smoke),
    }
    _write_json(out / "RUN_CONFIG.json", config)


def _bind_precomputed_rows(path: Path, split: str, smoke: bool) -> list[dict[str, Any]]:
    """Attach split metadata to a verified raw adapter archive.

    The raw collector intentionally need not carry runner-only ``_split_*``
    keys.  Binding is accepted only when its exact seed set equals this
    protocol's train roots; no row is silently re-seeded or dropped.
    """
    raw = _read_jsonl_gz(path)
    expected_jobs = _jobs(split, smoke)
    expected = {
        int(seed): (int(cell_index), cell)
        for cell_index, cell, seed, _ in expected_jobs
    }
    observed = [row.get("seed") for row in raw]
    if any(seed is None for seed in observed):
        raise RuntimeError("precomputed collect archive has a row without seed")
    observed_int = [int(seed) for seed in observed]
    if len(observed_int) != len(set(observed_int)):
        raise RuntimeError("precomputed collect archive contains duplicate seeds")
    if set(observed_int) != set(expected):
        raise RuntimeError("precomputed collect archive seed set differs from frozen train split")
    bound: list[dict[str, Any]] = []
    for row in raw:
        seed = int(row["seed"])
        cell_index, cell = expected[seed]
        item = dict(row)
        item.update({"_split_cell_index": cell_index, "_split_cell": _jsonable(cell),
                     "_split_seed": seed, "_smoke": bool(smoke)})
        bound.append(item)
    return sorted(bound, key=lambda row: int(row["_split_seed"]))


def record_collection(out: Path, rows: Sequence[Mapping[str, Any]], smoke: bool,
                      *, precomputed_source: Path | None = None) -> None:
    """Validate and persist the frozen training archive.

    This is public for binding an independently produced, seed-matched raw
    archive without running the expensive closed-loop collector again.
    """
    rows = [dict(row) for row in rows]
    for row in rows:
        _validate_collection_row(row)
        if row.get("_smoke") != bool(smoke):
            raise RuntimeError("collect row smoke marker mismatch")
    expected = _root_count("train", smoke)
    if len(rows) != expected:
        raise RuntimeError(f"collect returned {len(rows)} rows; expected {expected}")
    rows = sorted(rows, key=lambda row: int(row["_split_seed"]))
    _write_jsonl_gz(out / "train_cases.jsonl.gz", rows)
    _write_json(out / "collect_summary.json", {
        "rows": len(rows),
        "source_nomination_counts": {source: sum(len(row["nominations"][source]) for row in rows) for source in SOURCES},
        "archive_sha256": _file_hash(out / "train_cases.jsonl.gz"),
        "precomputed_source": str(precomputed_source.resolve()) if precomputed_source is not None else None,
        "precomputed_sha256": _file_hash(precomputed_source) if precomputed_source is not None else None,
    })


def stage_collect(out: Path, workers: int, smoke: bool, precomputed: Path | None = None) -> None:
    _require_out(out, smoke)
    adapter_hashes = configure()
    if not isinstance(adapter_hashes, Mapping) or not adapter_hashes:
        raise RuntimeError("experiment.configure() must return nonempty path -> SHA256 manifest")
    _save_manifest(out, smoke, adapter_hashes)
    jobs = _jobs("train", smoke)
    if precomputed is not None:
        rows = _bind_precomputed_rows(precomputed, "train", smoke)
    else:
        rows = run_parallel(_collect_job, jobs, workers, "collect") if workers > 1 else [_collect_job(job) for job in jobs]
    record_collection(out, rows, smoke, precomputed_source=precomputed)


def stage_train(out: Path, smoke: bool) -> None:
    manifest = _load_manifest(out)
    if bool(manifest.get("smoke")) != bool(smoke):
        raise RuntimeError("train smoke flag differs from collect manifest")
    rows = _read_jsonl_gz(out / "train_cases.jsonl.gz")
    if len(rows) != _root_count("train", smoke):
        raise RuntimeError("training archive root count mismatch")
    adapter_hashes = configure()
    if dict(adapter_hashes) != dict(manifest["adapter_hashes"]):
        raise RuntimeError("adapter source hashes changed since collect")
    base = base_training()
    if not isinstance(base, (tuple, list)) or len(base) != 3:
        raise RuntimeError("base_training() must return exactly (X, Y, seeds_or_counts)")
    base_x = np.asarray(base[0], dtype=float)
    base_y = np.asarray(base[1], dtype=float)
    if base_x.ndim != 2 or base_x.shape[1] != FIT_FEATURES:
        raise RuntimeError(f"historical X must have shape (n,{FIT_FEATURES}), got {base_x.shape}")
    if base_y.ndim != 2 or base_y.shape[1] != N_TARGETS:
        raise RuntimeError(f"historical Y must have shape (n,{N_TARGETS}), got {base_y.shape}")
    source_models: dict[str, Any] = {}
    source_counts: dict[str, int] = {}
    audits: dict[str, list[dict[str, Any]]] = {}
    for source in SOURCES:
        extension_x, extension_y, audit = _training_extension(rows, source)
        source_counts[source] = int(len(extension_x))
        audits[source] = audit
        if len(extension_x) != len(rows) * (PROPOSALS + 1):
            raise RuntimeError(f"{source} has unequal extension size {len(extension_x)}")
        x = np.vstack((base_x, extension_x))
        y = np.vstack((base_y, extension_y))
        source_models[source] = fit_targets(x, y)
    if len(set(source_counts.values())) != 1:
        raise RuntimeError(f"source training counts differ: {source_counts}")
    configure_models(source_models)
    joblib.dump(source_models, out / "models.joblib", compress=3)
    _write_json(out / "training_summary.json", {
        "base_rows": int(len(base_x)),
        "base_seed_metadata": _jsonable(base[2]),
        "new_rows_per_source": source_counts,
        "total_rows_per_source": {source: int(len(base_x) + count) for source, count in source_counts.items()},
        "model_sha256": _file_hash(out / "models.joblib"),
        "labels": audits,
    })


def _metric_from_trials(value: Any, field: str, *, request_id: str) -> float:
    trials = _trial_dicts(value, request_id=request_id)
    return float(np.mean([trial[field] for trial in trials]))


def _qualify_outcomes(row: Mapping[str, Any]) -> Mapping[str, Any]:
    outcomes = row.get("outcomes")
    if not isinstance(outcomes, Mapping):
        raise RuntimeError("qualify row must contain outcomes mapping")
    if "mpc" not in outcomes:
        raise RuntimeError("qualify row must include measured MPC fallback outcomes")
    _trial_dicts(outcomes["mpc"], request_id="mpc")
    return outcomes


def _choose_fixed(dev_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    families = sorted({_family_of_row(row) for row in dev_rows})
    result: dict[str, Any] = {}
    for family in families:
        family_rows = [row for row in dev_rows if _family_of_row(row) == family]
        candidate_ids = sorted({str(name) for row in family_rows for name in row.get("available", ()) if str(name) != "mpc"})
        choices: list[dict[str, Any]] = []
        for request_id in candidate_ids:
            present = 0
            records: list[dict[str, float]] = []
            for row in family_rows:
                outcomes = _qualify_outcomes(row)
                available = set(str(name) for name in row.get("available", ()))
                if request_id in available and request_id in outcomes:
                    present += 1
                    records.extend(_trial_dicts(outcomes[request_id], request_id=request_id))
                else:
                    records.extend(_trial_dicts(outcomes["mpc"], request_id="mpc"))
            availability = present / len(family_rows)
            if availability < .8:
                continue
            means = {metric: float(np.mean([record[metric] for record in records])) for metric in VISIBLE_TARGETS}
            choices.append({"request_id": request_id, "availability": availability, "means": means})
        if not choices:
            # This is an explicit protocol failure, not a silent switch to an
            # adaptive MPC arm.  The adapter's measured MPC remains the only
            # fallback for test rows.
            raise RuntimeError(f"no fixed request reaches 80% availability for family {family}")
        choices.sort(key=lambda choice: (
            -choice["means"]["corrected_complete"],
            choice["means"]["final_wrong"],
            choice["means"]["new_identity_mistake"],
            choice["means"]["tcp_mm"],
            choice["means"]["turn_degs"],
            choice["means"]["seconds"],
            choice["request_id"],
        ))
        result[family] = {"id": choices[0]["request_id"], "candidates": choices}
    return result


def record_qualification(out: Path, rows: Sequence[Mapping[str, Any]], smoke: bool,
                         *, precomputed_source: Path | None = None) -> dict[str, Any]:
    """Validate and persist a frozen development archive and fixed requests."""
    rows = [dict(row) for row in rows]
    expected = _root_count("dev", smoke)
    if len(rows) != expected:
        raise RuntimeError(f"qualify returned {len(rows)} rows; expected {expected}")
    for row in rows:
        if row.get("_smoke") != bool(smoke):
            raise RuntimeError("qualify row smoke marker mismatch")
        _qualify_outcomes(row)
    rows = sorted(rows, key=lambda row: int(row["_split_seed"]))
    fixed = _choose_fixed(rows)
    _write_jsonl_gz(out / "dev_cases.jsonl.gz", rows)
    _write_json(out / "fixed.json", fixed)
    _write_json(out / "qualify_summary.json", {
        "rows": len(rows),
        "fixed": fixed,
        "archive_sha256": _file_hash(out / "dev_cases.jsonl.gz"),
        "fixed_sha256": _file_hash(out / "fixed.json"),
        "precomputed_source": str(precomputed_source.resolve()) if precomputed_source is not None else None,
        "precomputed_sha256": _file_hash(precomputed_source) if precomputed_source is not None else None,
    })
    return fixed


def stage_qualify(out: Path, workers: int, smoke: bool, precomputed: Path | None = None) -> None:
    manifest = _load_manifest(out)
    if bool(manifest.get("smoke")) != bool(smoke):
        raise RuntimeError("qualify smoke flag differs from collect manifest")
    configure()
    jobs = _jobs("dev", smoke)
    rows = (_bind_precomputed_rows(precomputed, "dev", smoke) if precomputed is not None else
            (run_parallel(_qualify_job, jobs, workers, "qualify") if workers > 1 else [_qualify_job(job) for job in jobs]))
    record_qualification(out, rows, smoke, precomputed_source=precomputed)


def _load_models(out: Path, smoke: bool) -> dict[str, Any]:
    manifest = _load_manifest(out)
    if bool(manifest.get("smoke")) != bool(smoke):
        raise RuntimeError("test smoke flag differs from collect manifest")
    current_hashes = configure()
    if dict(current_hashes) != dict(manifest["adapter_hashes"]):
        raise RuntimeError("adapter source hashes changed since collect")
    path = out / "models.joblib"
    if not path.exists():
        raise RuntimeError(f"missing fitted models {path}; run train first")
    models = joblib.load(path)
    if set(models) != set(SOURCES):
        raise RuntimeError(f"models.joblib has keys {sorted(models)}; expected {list(SOURCES)}")
    configure_models(models)
    return models


def _validate_test_row(row: Mapping[str, Any]) -> None:
    if "arms" not in row or not isinstance(row["arms"], Mapping):
        raise RuntimeError("test row missing arms")
    missing = [arm for arm in ARMS if arm not in row["arms"]]
    if missing:
        raise RuntimeError(f"test row missing arms {missing}")
    for arm in ARMS:
        record = row["arms"][arm]
        if not isinstance(record, Mapping):
            raise RuntimeError(f"arm {arm} is not a mapping")
        if "outcomes" not in record:
            raise RuntimeError(f"arm {arm} missing outcomes")
        _report_trial_dicts(record["outcomes"], request_id=f"{arm}@{row.get('_split_seed')}")


def stage_test(out: Path, workers: int, smoke: bool) -> None:
    _load_models(out, smoke)
    fixed = json.loads((out / "fixed.json").read_text())
    fixed_ids = {str(family): str(value["id"]) for family, value in fixed.items()}
    jobs = _jobs("test", smoke)
    # The adapter's test function is a public closed-loop call.  It receives
    # the frozen fixed request and runs every arm on the same tape roots.
    test_function = functools.partial(_test_job, fixed_ids)
    rows = run_parallel(test_function, jobs, workers, "test") if workers > 1 else [test_function(job) for job in jobs]
    if len(rows) != _root_count("test", smoke):
        raise RuntimeError(f"test returned {len(rows)} rows; expected {_root_count('test', smoke)}")
    for row in rows:
        _validate_test_row(row)
    _write_jsonl_gz(out / "test_cases.jsonl.gz", rows)
    summary = _summarize(rows)
    summary["archive_sha256"] = _file_hash(out / "test_cases.jsonl.gz")
    _write_json(out / "test_summary.json", summary)


def _scene_means(rows: Sequence[Mapping[str, Any]], arm: str, metric: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        trials = _report_trial_dicts(row["arms"][arm]["outcomes"], request_id=f"{arm}@{row.get('_split_seed')}")
        values.append(float(np.mean([trial[metric] for trial in trials])))
    return np.asarray(values, dtype=float)


def _bootstrap_difference(rows: Sequence[Mapping[str, Any]], arm: str, reference: str, metric: str,
                          *, seed: int, repeats: int = 4000) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("cannot bootstrap an empty group")
    left = _scene_means(rows, arm, metric)
    right = _scene_means(rows, reference, metric)
    observed = float(np.mean(left - right))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(rows), size=(repeats, len(rows)))
    samples = np.mean((left - right)[indices], axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"arm": arm, "reference": reference, "metric": metric, "n_scenes": len(rows),
            "difference": observed, "ci95": [float(low), float(high)]}


def _group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"scenes": len(rows), "tapes": 0, "arms": {}}
    if not rows:
        return output
    for arm in ARMS:
        outcome_lists = [_report_trial_dicts(row["arms"][arm]["outcomes"], request_id=f"{arm}@{row.get('_split_seed')}") for row in rows]
        output["tapes"] = max(output["tapes"], max(len(trials) for trials in outcome_lists))
        metrics = {metric: float(np.mean([trial[metric] for trials in outcome_lists for trial in trials]))
                   for metric in REPORT_METRICS}
        cpu_values = [row["arms"][arm].get("cpu_seconds") for row in rows
                      if row["arms"][arm].get("cpu_seconds") is not None]
        if cpu_values:
            metrics["cpu_seconds"] = float(np.mean([float(value) for value in cpu_values]))
            metrics["decision_ms"] = 1000. * metrics["cpu_seconds"]
        output["arms"][arm] = metrics
    return output


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    families = sorted({_family_of_row(row) for row in rows})
    contexts = sorted({_context_of_row(row) for row in rows})
    groups: dict[str, Sequence[Mapping[str, Any]]] = {"pooled": rows}
    groups.update({f"family:{family}": [row for row in rows if _family_of_row(row) == family] for family in families})
    groups.update({f"context:{context}": [row for row in rows if _context_of_row(row) == context] for context in contexts})
    summaries = {name: _group_summary(group) for name, group in groups.items()}
    bootstrap: dict[str, list[dict[str, Any]]] = {}
    for group_index, (name, group) in enumerate(groups.items()):
        bootstrap[name] = []
        for source_index, arm in enumerate(("random_b", "direct_b", "evidence_b")):
            for reference_index, reference in enumerate(REFERENCE_ARMS):
                if arm == reference:
                    continue
                for metric_index, metric in enumerate(REPORT_METRICS):
                    bootstrap[name].append(_bootstrap_difference(
                        group, arm, reference, metric,
                        seed=74_390_000 + group_index * 100_000 + source_index * 10_000 + reference_index * 100 + metric_index,
                    ))
    return {
        "protocol": "ab_teacher_b_training_20261118",
        "arms": list(ARMS),
        "metrics": list(REPORT_METRICS),
        "groups": summaries,
        "paired_scene_bootstrap": bootstrap,
        "rows": len(rows),
        "archive_sha256": None,
    }


def _run_stage(stage: str, args: argparse.Namespace) -> None:
    out = args.out.resolve()
    if stage == "collect":
        stage_collect(out, args.workers, args.smoke, args.precomputed_collect)
    elif stage == "train":
        stage_train(out, args.smoke)
    elif stage == "qualify":
        stage_qualify(out, args.workers, args.smoke, args.precomputed_qualify)
    elif stage == "test":
        stage_test(out, args.workers, args.smoke)
    elif stage == "all":
        stage_collect(out, args.workers, args.smoke, args.precomputed_collect)
        stage_train(out, args.smoke)
        stage_qualify(out, args.workers, args.smoke, args.precomputed_qualify)
        stage_test(out, args.workers, args.smoke)
    else:  # pragma: no cover
        raise ValueError(stage)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("all", "collect", "train", "qualify", "test"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True,
                        help="explicit worker count; use 1 for a deterministic single-process run")
    parser.add_argument("--smoke", action="store_true",
                        help="small run on the separate 75_100_000 seed range and a separate output path")
    parser.add_argument("--precomputed-collect", type=Path, default=None,
                        help="verified raw train archive for binding; avoids rerunning collect")
    parser.add_argument("--precomputed-qualify", type=Path, default=None,
                        help="verified raw development archive for binding; avoids rerunning qualify")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    try:
        _run_stage(args.stage, args)
    except Exception as exc:
        raise SystemExit(f"{args.stage} failed: {exc}") from exc
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
