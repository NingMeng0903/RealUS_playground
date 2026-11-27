#!/usr/bin/env python3
"""Reproducible development-stage orchestration for adaptive stopping.

This module is an execution coordinator only.  It does not select a policy
from test outcomes, alter the planner, or run a controller itself.  Each job
is a subprocess invocation of ``run_adaptive.py`` with a disjoint output
sub-root.  The default matrix is

    (A_task, A_info, B_H1, B_H2) x fixed runner pair-dwell 0.30
    pair_fixed x (0.15, 0.30, 0.60)
    x (fixture index 0, fixture index 1; both constrained)

The genesis Python and single-thread numerical environment are recorded in
each job manifest.  Source/model hashes are checked before and after jobs;
source changes invalidate the run and never silently permit a resume or a
freeze.

No job is started merely by importing this module.  The CLI supports
``--dry-run`` for an auditable command plan and ``--select-freeze`` for a
read-only completeness/selection pass over an already completed full dev
matrix.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_adaptive.py"
DEFAULT_MODEL = HERE / "models" / "v3" / "MODEL.json"
GENESIS_PYTHON = Path("/media/camp/EXT_DRIVE/envs/genesis/bin/python")
ADAPTIVE_ARMS = ("A_task", "A_info", "B_H1", "B_H2")
ARMS = ADAPTIVE_ARMS + ("pair_fixed",)
DWELLS = (0.15, 0.30, 0.60)
DEFAULT_ADAPTIVE_DWELL = 0.30
FIXTURE_INDEXES = (0, 1)
MIRROR_MODE = "constrained"
DEFAULT_COUNT = 8
FULL_COUNT = 128
MAX_PROBE_FRAMES = 8
RISK_GATE = 0.90
SOURCE_FILES = (
    "run_adaptive.py", "adaptive_planning.py", "acoustics.py", "belief_model.py",
    "fixture.py", "native_adapter.py", "robot_motion.py", "train_calibrate.py",
    "study_driver.py",
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "1", "pass", "passed", "correct"}:
        return True
    if text in {"false", "no", "0", "fail", "failed", "incorrect"}:
        return False
    return None


def _mean(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values]
    return statistics.fmean(values) if values else None


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    """Descriptive 95% Wilson interval; never used as a risk gate."""
    if total <= 0:
        return None
    p = float(successes) / float(total)
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    half = z * ((p * (1.0 - p) / total + z * z / (4.0 * total * total)) ** 0.5) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _source_paths(model_artifact: Path) -> list[Path]:
    paths = [HERE / name for name in SOURCE_FILES] + [model_artifact.resolve()]
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def source_manifest(model_artifact: str | Path = DEFAULT_MODEL) -> dict[str, Any]:
    """Hash every source/input path that can change a dev result."""
    model = Path(model_artifact).resolve()
    files = []
    missing = []
    for path in _source_paths(model):
        if not path.is_file():
            missing.append(str(path))
            continue
        files.append({"path": str(path), "sha256": _sha256(path), "size": path.stat().st_size})
    return {
        "schema": "adaptive_study_source_manifest_v1",
        "created_utc": _utc(), "model_artifact": str(model),
        "files": files, "missing": missing,
        "all_inputs": [entry["path"] for entry in files],
    }


def source_hash_map(manifest: Mapping[str, Any]) -> dict[str, str]:
    return {str(item["path"]): str(item["sha256"])
            for item in manifest.get("files", []) if isinstance(item, Mapping)}


def source_manifests_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return source_hash_map(left) == source_hash_map(right) and not left.get("missing") and not right.get("missing")


@dataclass(frozen=True)
class JobSpec:
    arm: str
    dwell: float
    fixture_index: int
    mode: str
    count: int
    output_root: str

    @property
    def config_id(self) -> str:
        return f"{self.arm}_dwell_{int(round(self.dwell * 100)):02d}"

    @property
    def job_id(self) -> str:
        return f"{self.config_id}/fixture_{self.fixture_index:02d}_{self.mode}"

    @property
    def job_root(self) -> Path:
        return Path(self.output_root).resolve() / self.config_id / self.job_id.split("/", 1)[1]


def build_job_specs(output_root: str | Path, *, count: int = DEFAULT_COUNT,
                    arms: Sequence[str] | None = None, dwells: Sequence[float] | None = None,
                    fixture_indexes: Sequence[int] = FIXTURE_INDEXES,
                    mode: str = MIRROR_MODE) -> list[JobSpec]:
    """Build the fixed 14-job matrix without touching the filesystem."""
    if count <= 0:
        raise ValueError("count must be positive")
    if mode != MIRROR_MODE:
        raise ValueError("development matrix requires constrained mirrors")
    if set(fixture_indexes) != set(FIXTURE_INDEXES):
        raise ValueError("development matrix requires fixture indexes 0 and 1")
    root = str(Path(output_root).resolve())
    if arms is None and dwells is None:
        config_pairs = [(arm, DEFAULT_ADAPTIVE_DWELL) for arm in ADAPTIVE_ARMS]
        config_pairs.extend(("pair_fixed", dwell) for dwell in DWELLS)
    else:
        chosen_arms = tuple(arms if arms is not None else ARMS)
        chosen_dwells = tuple(dwells if dwells is not None else DWELLS)
        unknown = set(chosen_arms) - set(ARMS)
        if unknown:
            raise ValueError(f"unsupported development arms: {sorted(unknown)}")
        config_pairs = [(str(arm), float(dwell)) for arm in chosen_arms for dwell in chosen_dwells]
    specs = [JobSpec(str(arm), float(dwell), int(index), mode, int(count), root)
             for arm, dwell in config_pairs for index in fixture_indexes]
    if len({spec.job_root for spec in specs}) != len(specs):
        raise AssertionError("job output roots are not disjoint")
    return specs


def build_command(spec: JobSpec, *, python_executable: str | Path = GENESIS_PYTHON,
                  model_artifact: str | Path = DEFAULT_MODEL) -> list[str]:
    """Build the exact runner command for one matrix cell."""
    return [
        str(Path(python_executable)), str(RUNNER),
        "--stage", "dev", "--count", str(spec.count), "--case-start", "0",
        "--index", str(spec.fixture_index), "--mode", spec.mode,
        "--arm", spec.arm, "--model-artifact", str(Path(model_artifact).resolve()),
        "--output", str(spec.job_root), "--resume",
        "--pair-dwell", f"{spec.dwell:g}", "--max-probe-frames", str(MAX_PROBE_FRAMES),
        "--backend", "native",
    ]


def deterministic_environment() -> dict[str, str]:
    """Return the reproducibility environment; never inherit thread fan-out."""
    return {
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1", "OMP_DYNAMIC": "FALSE",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def _existing_start_manifest(output_root: Path) -> dict[str, Any] | None:
    for name in ("DRIVER_SOURCE_START.json", "SOURCE_HASH_START.json"):
        path = output_root / name
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raise RuntimeError(f"invalid existing source manifest: {path}")
    return None


def _assert_resume_source(output_root: Path, current: Mapping[str, Any], *, allow_new: bool) -> None:
    prior = _existing_start_manifest(output_root)
    if prior is not None:
        if not source_manifests_equal(prior, current):
            raise RuntimeError("source/model hash changed since the existing driver run; resume is invalid")
        return
    if not allow_new:
        existing = [path for path in output_root.iterdir()] if output_root.exists() else []
        if existing:
            raise RuntimeError("output root has data but no DRIVER_SOURCE_START.json; refusing silent reuse")


def _job_log_dir(spec: JobSpec) -> Path:
    return Path(spec.output_root).resolve() / "driver_logs" / spec.job_id


def _run_job(spec: JobSpec, *, python_executable: str, model_artifact: str,
             baseline: Mapping[str, Any]) -> dict[str, Any]:
    """Worker entry point: launch one runner subprocess and capture both logs."""
    output_root = Path(spec.output_root).resolve()
    log_dir = _job_log_dir(spec)
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "stdout.log"
    stderr_path = log_dir / "stderr.log"
    result_path = log_dir / "JOB_RESULT.json"
    command = build_command(spec, python_executable=python_executable, model_artifact=model_artifact)
    env = os.environ.copy()
    env.update(deterministic_environment())
    start_hash = source_manifest(model_artifact)
    record: dict[str, Any] = {
        "schema": "adaptive_study_job_result_v1", "job_id": spec.job_id,
        "spec": asdict(spec), "command": command, "cwd": str(HERE),
        "environment": deterministic_environment(), "started_utc": _utc(),
        "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
        "source_hash_start": start_hash,
    }
    if not source_manifests_equal(start_hash, baseline):
        record.update({"status": "invalid_source_changed", "return_code": None,
                       "finished_utc": _utc(), "source_hash_end": start_hash})
        _write_json(result_path, record)
        return record
    started = time.monotonic()
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(command, cwd=str(HERE), env=env,
                                       stdout=stdout, stderr=stderr, check=False)
        return_code = int(completed.returncode)
        error = None
    except Exception as exc:  # preserve a job-level failure rather than aborting siblings
        return_code = None
        error = repr(exc)
    end_hash = source_manifest(model_artifact)
    changed = not source_manifests_equal(start_hash, end_hash) or not source_manifests_equal(baseline, end_hash)
    status = "invalid_source_changed" if changed else "ok" if return_code == 0 else "runner_failed"
    record.update({"status": status, "return_code": return_code, "error": error,
                   "elapsed_s": time.monotonic() - started, "finished_utc": _utc(),
                   "source_hash_end": end_hash, "source_changed": changed})
    _write_json(result_path, record)
    return record


def run_jobs(output_root: str | Path, *, count: int = DEFAULT_COUNT,
             max_workers: int = 4, dry_run: bool = False,
             python_executable: str | Path = GENESIS_PYTHON,
             model_artifact: str | Path = DEFAULT_MODEL,
             specs: Sequence[JobSpec] | None = None) -> dict[str, Any]:
    """Run or plan the fixed dev matrix, with hash-guarded resume semantics."""
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    root = Path(output_root).resolve()
    model = Path(model_artifact).resolve()
    executable = Path(python_executable).resolve()
    if not RUNNER.is_file():
        raise FileNotFoundError(RUNNER)
    if not model.is_file():
        raise FileNotFoundError(model)
    if not executable.is_file():
        raise FileNotFoundError(executable)
    matrix = list(specs) if specs is not None else build_job_specs(root, count=count)
    if any(Path(spec.output_root).resolve() != root for spec in matrix):
        raise ValueError("all job specs must use the requested output root")
    baseline = source_manifest(model)
    root.mkdir(parents=True, exist_ok=True)
    root_was_new = not root.exists() or not any(root.iterdir())
    _assert_resume_source(root, baseline, allow_new=root_was_new)
    prior_plan: dict[str, Any] = {}
    prior_plan_path = root / "DRIVER_MANIFEST.json"
    if prior_plan_path.is_file():
        try:
            prior_plan = json.loads(prior_plan_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise RuntimeError(f"invalid existing driver manifest: {prior_plan_path}")
    prior_count = _finite(prior_plan.get("count"))
    if prior_count is not None and int(prior_count) not in (int(count), DEFAULT_COUNT, FULL_COUNT):
        raise RuntimeError("existing output root was created with a different count; refusing silent reuse")
    start_path = root / "DRIVER_SOURCE_START.json"
    if not start_path.exists():
        _write_json(start_path, baseline)
    plan = {
        "schema": "adaptive_study_driver_manifest_v1", "created_utc": _utc(),
        "stage": "dev", "count": count, "max_workers": max_workers,
        "python": str(executable), "model_artifact": str(model),
        "max_probe_frames": MAX_PROBE_FRAMES, "mode": MIRROR_MODE,
        "environment": deterministic_environment(), "source_manifest": baseline,
        "resume_same_root": bool(prior_count is not None),
        "prior_count": None if prior_count is None else int(prior_count),
        "pilot_to_full_continuation": bool(prior_count == DEFAULT_COUNT and count == FULL_COUNT),
        "jobs": [asdict(spec) | {"job_id": spec.job_id, "job_root": str(spec.job_root),
                                  "command": build_command(spec, python_executable=executable, model_artifact=model)}
                  for spec in matrix],
    }
    _write_json(root / "DRIVER_MANIFEST.json", plan)
    if dry_run:
        results = [{"job_id": spec.job_id, "status": "planned", "command": build_command(
            spec, python_executable=executable, model_artifact=model), "job_root": str(spec.job_root)}
                   for spec in matrix]
    else:
        results: list[dict[str, Any]] = []
        worker_count = min(int(max_workers), len(matrix) or 1)
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = [pool.submit(_run_job, spec, python_executable=str(executable),
                                   model_artifact=str(model), baseline=baseline)
                       for spec in matrix]
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda item: str(item.get("job_id", "")))
    end = source_manifest(model)
    changed = not source_manifests_equal(baseline, end) or any(
        item.get("status") == "invalid_source_changed" for item in results)
    _write_json(root / "DRIVER_SOURCE_END.json", end)
    summary = {"schema": "adaptive_study_driver_run_v1", "finished_utc": _utc(),
               "dry_run": dry_run, "source_changed": changed,
               "source_manifest_start": baseline, "source_manifest_end": end,
               "jobs": results, "all_jobs_ok": bool(results) and all(item.get("status") == "ok" for item in results)}
    _write_json(root / "DRIVER_RUN_SUMMARY.json", summary)
    return summary


def _result_paths(spec: JobSpec, *, fixture_index: int | None = None) -> list[Path]:
    index = spec.fixture_index if fixture_index is None else int(fixture_index)
    # Each fixture job has its own output sub-root.  This lookup deliberately
    # reconstructs the sibling root from the immutable config ID so a config
    # summary can pair index 0 and index 1 without merging result files.
    job_root = Path(spec.output_root).resolve() / spec.config_id / f"fixture_{index:02d}_{spec.mode}"
    base = job_root / "dev" / f"fixture_{index:02d}_{spec.mode}"
    if not base.exists():
        return []
    return sorted(base.rglob("RESULT.json"))


def _gate_protocol_pass(row: Mapping[str, Any]) -> bool:
    """Check only the prespecified .90 commit protocol, not empirical risk."""
    protocol = row.get("protocol") if isinstance(row.get("protocol"), Mapping) else {}
    gate = _finite(protocol.get("risk_gate", RISK_GATE))
    if gate is None or abs(gate - RISK_GATE) > 1.0e-9:
        return False
    decisions = row.get("decisions")
    if not isinstance(decisions, list):
        return False
    for decision in decisions:
        if not isinstance(decision, Mapping) or str(decision.get("decision", "")) != "commit":
            continue
        score = _finite(decision.get("score"))
        if score is None or score < RISK_GATE - 1.0e-9:
            return False
    return True


def _risk_eligible(row: Mapping[str, Any]) -> bool:
    """Backward-compatible alias; output names this as gate protocol only."""
    return _gate_protocol_pass(row)


def _committed(row: Mapping[str, Any]) -> bool:
    value = _bool(row.get("committed"))
    if value is not None:
        return value
    terminal = _finite(row.get("terminal_count"))
    if terminal is not None and terminal > 0:
        return True
    decisions = row.get("decisions")
    return isinstance(decisions, list) and any(
        isinstance(item, Mapping) and str(item.get("decision", "")) == "commit" for item in decisions)


def _load_result_rows(spec: JobSpec, *, fixture_index: int | None = None) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    rows: list[dict[str, Any]] = []
    paths: list[str] = []
    errors: list[str] = []
    for path in _result_paths(spec, fixture_index=fixture_index):
        paths.append(str(path))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
            continue
        if isinstance(payload, Mapping):
            row = dict(payload)
            row["_result_path"] = str(path)
            rows.append(row)
    return rows, paths, errors


def aggregate_config(spec: JobSpec, *, expected_count: int | None = None,
                     baseline: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Aggregate one arm/dwell config and verify both exact mirror ID sets."""
    expected_count = int(spec.count if expected_count is None else expected_count)
    expected_ids = set(range(expected_count))
    rows: list[dict[str, Any]] = []
    paths: list[str] = []
    errors: list[str] = []
    for fixture_index in FIXTURE_INDEXES:
        mirror_rows, mirror_paths, mirror_errors = _load_result_rows(spec, fixture_index=fixture_index)
        rows.extend(mirror_rows)
        paths.extend(mirror_paths)
        errors.extend(mirror_errors)
    by_mirror: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for row in rows:
        index = _finite(row.get("fixture_index"))
        if index is None or int(index) not in by_mirror:
            continue
        by_mirror[int(index)].append(row)
    mirror_audit: dict[str, Any] = {}
    valid_rows: list[dict[str, Any]] = []
    duplicate = False
    missing = False
    for index in FIXTURE_INDEXES:
        mirror_rows = by_mirror[index]
        ids = [int(_finite(row.get("case_id"))) for row in mirror_rows if _finite(row.get("case_id")) is not None]
        counts: dict[int, int] = {}
        for case_id in ids:
            counts[case_id] = counts.get(case_id, 0) + 1
        duplicate_ids = sorted(case_id for case_id, n in counts.items() if n > 1)
        observed = set(ids)
        missing_ids = sorted(expected_ids - observed)
        extra_ids = sorted(observed - expected_ids)
        duplicate = duplicate or bool(duplicate_ids)
        missing = missing or bool(missing_ids or extra_ids)
        selected_rows = [row for row in mirror_rows if _finite(row.get("case_id")) is not None and
                         int(float(row["case_id"])) in expected_ids]
        valid_rows.extend(selected_rows)
        mirror_audit[str(index)] = {
            "records": len(mirror_rows), "expected_records": expected_count,
            "observed_ids": sorted(observed), "missing_ids": missing_ids,
            "extra_ids": extra_ids, "duplicate_ids": duplicate_ids,
            "complete": not missing_ids and not extra_ids and not duplicate_ids and len(mirror_rows) == expected_count,
        }
    source_manifests = [row.get("source_manifest") for row in valid_rows if row.get("source_manifest") is not None]
    source_consistent = bool(source_manifests) and all(item == source_manifests[0] for item in source_manifests)
    if baseline is not None:
        expected_hashes = source_hash_map(baseline)
        # run_adaptive's per-result manifest intentionally contains its import
        # modules only; the driver manifest additionally hashes the runner,
        # driver and model.  Require every runner entry to match the driver's
        # overlapping paths, rather than demanding identical key sets.
        source_consistent = source_consistent and all(
            all(expected_hashes.get(path) == digest for path, digest in source_hash_map(item).items())
            for item in source_manifests if isinstance(item, Mapping))
    complete = all(item["complete"] for item in mirror_audit.values()) and source_consistent and not errors
    correct = [bool(row.get("correct_complete", False)) for row in valid_rows]
    false = [bool(row.get("false_complete", False)) for row in valid_rows]
    no_complete = [bool(row.get("no_complete", (
        not ok and not bad and "withdraw" not in str(row.get("reason", "")).lower()
        and "watchdog" not in str(row.get("reason", "")).lower())))
                   for row, ok, bad in zip(valid_rows, correct, false)]
    withdraw = [bool(row.get("withdraw", row.get("abstain", False))) for row in valid_rows]
    watchdog = [bool(row.get("watchdog", False)) for row in valid_rows]
    committed = [_committed(row) for row in valid_rows]
    committed_fail = [commit and not ok for commit, ok in zip(committed, correct)]
    probes = [_finite(row.get("probe_count")) for row in valid_rows]
    probes = [value for value in probes if value is not None]
    gate_flags = [_gate_protocol_pass(row) for row in valid_rows]
    wrong_declaration = [bool(row.get("false_complete", False)) for row in valid_rows]
    committed_n = int(sum(committed))
    committed_failure_n = int(sum(committed_fail))
    committed_fail_rate = committed_failure_n / committed_n if committed_n else None
    if not gate_flags or not all(gate_flags):
        risk_status = "gate_protocol_failed"
    elif committed_n == 0:
        risk_status = "risk_unassessed"
    elif committed_fail_rate is not None and committed_fail_rate <= 0.10:
        risk_status = "risk_qualified"
    else:
        risk_status = "risk_failed"
    config_root = Path(spec.output_root).resolve() / spec.config_id
    job_roots = [config_root / f"fixture_{index:02d}_{spec.mode}" for index in FIXTURE_INDEXES]
    manifest_paths = [path for job_root in job_roots for path in job_root.rglob("MANIFEST*.json")]
    summary = {
        "schema": "adaptive_study_config_summary_v1", "config_id": spec.config_id,
        "arm": spec.arm, "dwell": spec.dwell, "count_requested": spec.count,
        "expected_count": expected_count, "fixture_indexes": list(FIXTURE_INDEXES),
        "mode": spec.mode, "job_roots": [str(path) for path in job_roots], "result_paths": paths,
        "parse_errors": errors, "mirror_audit": mirror_audit,
        "source_manifest_consistent": source_consistent,
        "complete": complete, "valid_record_count": len(valid_rows),
        "correct_rate": _mean([float(value) for value in correct]),
        "false_rate": _mean([float(value) for value in false]),
        "correct_complete_count": int(sum(correct)),
        "false_complete_count": int(sum(false)),
        "no_complete_count": int(sum(no_complete)),
        "withdraw_count": int(sum(withdraw)),
        "watchdog_count": int(sum(watchdog)),
        "no_complete_rate": _mean([float(value) for value in no_complete]),
        "withdraw_rate": _mean([float(value) for value in withdraw]),
        "watchdog_rate": _mean([float(value) for value in watchdog]),
        "mean_probe_count": _mean(probes),
        "commitcoverage": _mean([float(value) for value in committed]),
        "committed_fail_rate": committed_fail_rate,
        "committed_n": committed_n,
        "committed_count": committed_n,
        "committed_failure_n": committed_failure_n,
        "committed_failure_count": committed_failure_n,
        "committed_fail_rate_ci95_descriptive": _wilson_interval(committed_failure_n, committed_n),
        "risk_status": risk_status,
        "risk_qualified": risk_status == "risk_qualified",
        "risk_qualification_rule": "gate_protocol_pass_all and committed_n>0 and committed_failure_n/committed_n<=0.10; two terminal rejections and mechanical failures count as failures",
        "risk_ci_is_descriptive_only": True,
        "gate_protocol_pass_rate": _mean([float(value) for value in gate_flags]),
        "gate_protocol_pass_all": bool(gate_flags) and all(gate_flags),
        "gate_protocol_missing_or_violation_count": int(sum(not value for value in gate_flags)),
        "wrong_declaration_count": int(sum(wrong_declaration)),
        "wrong_declaration_rate": _mean([float(value) for value in wrong_declaration]),
        "risk_eligibility": "deferred_to_Astra; gate_protocol_pass is not empirical committed-risk qualification",
        "risk_eligible_rate": None,
        "risk_eligible_all": None,
        "source_input_paths": paths + [str(path) for path in job_roots] + [str(path) for path in manifest_paths],
    }
    return summary


def _selection_key(summary: Mapping[str, Any], *, family: str) -> tuple[Any, ...]:
    rate = summary.get("correct_rate")
    probes = summary.get("mean_probe_count")
    rate = -float(rate) if rate is not None else float("inf")
    probes = float(probes) if probes is not None else float("inf")
    arm_order = {"A_task": 0, "A_info": 1, "B_H1": 0, "B_H2": 1, "pair_fixed": 0}
    tie_arm = arm_order.get(str(summary.get("arm")), 99)
    dwell = float(summary.get("dwell", float("inf")))
    return rate, probes, tie_arm, dwell


def select_global_freeze(output_root: str | Path, *, count: int = FULL_COUNT,
                         model_artifact: str | Path = DEFAULT_MODEL) -> dict[str, Any]:
    """Check the full matrix and select frozen A/B/pair without test data."""
    root = Path(output_root).resolve()
    model = Path(model_artifact).resolve()
    start = _existing_start_manifest(root)
    baseline = start or source_manifest(model)
    current_source = source_manifest(model)
    specs = build_job_specs(root, count=count)
    summaries = [aggregate_config(spec, expected_count=count, baseline=baseline) for spec in specs]
    source_changed = not source_manifests_equal(baseline, current_source)
    all_complete = bool(summaries) and all(item.get("complete") for item in summaries) and not source_changed
    all_gate_protocol = bool(summaries) and all(item.get("gate_protocol_pass_all") for item in summaries)
    by_family = {
        "A": [item for item in summaries if item.get("arm") in {"A_task", "A_info"}],
        "B": [item for item in summaries if item.get("arm") in {"B_H1", "B_H2"}],
        "pair": [item for item in summaries if item.get("arm") == "pair_fixed"],
    }
    selected: dict[str, Any] = {}
    selected_unfiltered: dict[str, Any] = {}
    for family, candidates in by_family.items():
        complete_candidates = [item for item in candidates if item.get("complete")]
        selected_unfiltered[family] = (
            min(complete_candidates, key=lambda item: _selection_key(item, family=family))
            if complete_candidates else None
        )
        eligible = [item for item in complete_candidates if item.get("risk_qualified")]
        selected[family] = min(eligible, key=lambda item: _selection_key(item, family=family)) if eligible else None
    risk_winners_available = all(value is not None for value in selected.values())
    ready = count == FULL_COUNT and all_complete and all_gate_protocol and risk_winners_available
    if ready:
        status = "ready"
    elif count != FULL_COUNT or not all_complete or not all_gate_protocol:
        status = "incomplete"
    else:
        status = "negative"
    paths = sorted({path for item in summaries for path in item.get("source_input_paths", [])})
    payload: dict[str, Any] = {
        "schema": "adaptive_study_global_freeze_v1", "stage": "dev", "created_utc": _utc(),
        "requested_count": count, "required_full_count": FULL_COUNT,
        "mirror_indexes": list(FIXTURE_INDEXES), "mode": MIRROR_MODE,
        "complete_all_configs": all_complete, "gate_protocol_pass_all_configs": all_gate_protocol,
        "risk_eligibility_status": "deferred_to_Astra",
        "risk_qualified_by_family": {family: bool(item is not None) for family, item in selected.items()},
        "source_changed_since_start": source_changed,
        "freeze_ready": ready, "status": status,
        "reason": None if ready else (
            "pilot_count_is_not_freeze_eligible" if count != FULL_COUNT else
            "one_or_more_config_mirrors_missing_or_source_inconsistent" if not all_complete else
            "gate_protocol_missing_or_violated" if not all_gate_protocol else
            "risk_qualification_not_met_for_A_B_or_pair"),
        "selection_rule": "correct_rate descending; mean_probe_count ascending; deterministic H1/A_task and dwell ties",
        "selected_a": None if selected["A"] is None else selected["A"]["arm"],
        "selected_a_dwell": None if selected["A"] is None else selected["A"]["dwell"],
        "selected_b": None if selected["B"] is None else selected["B"]["arm"],
        "selected_b_dwell": None if selected["B"] is None else selected["B"]["dwell"],
        "selected_pair": "pair_fixed" if selected["pair"] is not None else None,
        "selected_pair_dwell": None if selected["pair"] is None else selected["pair"]["dwell"],
        "selected_config_summaries": selected,
        "selected_unfiltered_config_summaries": selected_unfiltered,
        "config_summaries": summaries,
        "model_artifact": str(model), "model_artifact_sha256": _sha256(model) if model.is_file() else None,
        "source_manifest_start": baseline, "source_manifest_end": current_source,
        "all_input_paths": sorted(set(paths + [str(model), str(root / "DRIVER_MANIFEST.json"),
                                                str(root / "DRIVER_SOURCE_START.json")])),
        "test_freeze_contract": {
            "arms": [selected["B"]["arm"], selected["A"]["arm"], "pair_fixed"] if ready else [],
            "mirror_indexes": list(FIXTURE_INDEXES), "mode": MIRROR_MODE,
            "same_global_selection_for_both_mirrors": True,
        },
    }
    return payload


def write_global_freeze(output_root: str | Path, payload: Mapping[str, Any]) -> list[Path]:
    """Write one global freeze and two runner-compatible per-fixture adapters."""
    root = Path(output_root).resolve()
    if not payload.get("freeze_ready"):
        raise RuntimeError("refusing to write freeze: development matrix is incomplete or gate protocol is not auditable")
    global_path = root / "FREEZE_GLOBAL.json"
    _write_json(global_path, payload)
    paths = [global_path]
    for index in FIXTURE_INDEXES:
        adapter = {
            "schema": "adaptive_stopping_freeze_v1", "stage": "dev",
            "fixture_index": index, "mode": MIRROR_MODE,
            "selected_a": payload["selected_a"], "selected_b": payload["selected_b"],
            "selected_pair": payload["selected_pair"],
            "selected_pair_dwell": payload["selected_pair_dwell"],
            "risk_gate": RISK_GATE, "watchdog_max_probe_frames": MAX_PROBE_FRAMES,
            "model_artifact": payload["model_artifact"],
            "model_artifact_sha256": payload["model_artifact_sha256"],
            "coverage_case_ids": list(range(FULL_COUNT)), "coverage_complete": True,
            "source_manifest": payload["source_manifest_start"],
            "global_freeze": str(global_path), "global_selection_for_both_mirrors": True,
            "test_arms": payload["test_freeze_contract"]["arms"],
            "all_input_paths": payload["all_input_paths"],
        }
        path = root / f"FREEZE_{index:02d}_{MIRROR_MODE}.json"
        _write_json(path, adapter)
        paths.append(path)
    return paths


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True,
                        help="independent root for the dev matrix")
    parser.add_argument("--count", type=int, choices=(8, 128), default=DEFAULT_COUNT,
                        help="8 for pilot or 128 for the full development matrix")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--python", type=Path, default=GENESIS_PYTHON,
                        help=f"runner Python (default {GENESIS_PYTHON})")
    parser.add_argument("--model-artifact", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dry-run", action="store_true", help="write the command plan without starting jobs")
    parser.add_argument("--select-freeze", action="store_true",
                        help="inspect existing full dev outputs and write freeze adapters if complete")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    root = args.output_root.resolve()
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be positive")
    if args.select_freeze:
        payload = select_global_freeze(root, count=args.count, model_artifact=args.model_artifact)
        if payload.get("freeze_ready") and not args.dry_run:
            paths = write_global_freeze(root, payload)
            payload = dict(payload)
            payload["freeze_paths"] = [str(path) for path in paths]
            _write_json(root / "FREEZE_SELECTION.json", payload)
        elif args.dry_run:
            _write_json(root / "FREEZE_SELECTION_DRY_RUN.json", payload)
        print(json.dumps({"status": payload["status"], "freeze_ready": payload["freeze_ready"],
                          "selected_a": payload.get("selected_a"), "selected_b": payload.get("selected_b"),
                          "selected_pair_dwell": payload.get("selected_pair_dwell")}, ensure_ascii=False))
        return 0 if payload.get("freeze_ready") else 2
    summary = run_jobs(root, count=args.count, max_workers=args.max_workers, dry_run=args.dry_run,
                       python_executable=args.python, model_artifact=args.model_artifact)
    print(json.dumps({"status": "planned" if args.dry_run else "finished",
                      "dry_run": summary["dry_run"], "source_changed": summary["source_changed"],
                      "all_jobs_ok": summary["all_jobs_ok"], "jobs": len(summary["jobs"])}, ensure_ascii=False))
    return 0 if args.dry_run or summary["all_jobs_ok"] and not summary["source_changed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

