#!/usr/bin/env python3
"""Measure offline ``wbc_rt`` scheduling contention.

This entry point is deliberately a small, hardware-free experiment.  It
loads one measured pose from a CSV, then reuses
``profile_controller_runtime.run_case`` twice:

* ``native_same_core`` pins four bounded standard-library workers to the
  requested native CPU;
* ``background_cpu`` puts the same four workers on CPUs 10/12 (or the CPUs
  supplied with ``--background-cpus``).

The workers are ordinary child processes.  They do not import the robot SDK,
open a socket, or attach to the controller's SHM.  ``run_case`` assigns a
fresh UUID SHM prefix for every case, so this tool has no option that can
reuse a production ``rm75_wbc`` segment.  It is therefore suitable for
comparing native scheduling changes offline; it is not a hardware test.

Run from ``rm75_control``::

    python apps/joint_admittance_8dof/profile_native_contention.py \
        --seed-csv apps/logs/peirastic/run_YYYYMMDD_HHMMSS.csv \
        --ticks 600 --control-cpu 6 --native-cpu 8 \
        --output /tmp/native-contention.json

The process intentionally uses only a few workers and never changes another
process's affinity.  A worker changes its own mask once, reports the result,
and exits in the ``finally`` cleanup path.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any, Iterable

try:  # Script path: ``python apps/.../profile_native_contention.py``.
    from profile_controller_runtime import run_case
except ImportError:  # Module path: ``python -m ...profile_native_contention``.
    from .profile_controller_runtime import run_case


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "joint_admittance_8dof.yaml"
DEFAULT_CONTROL_CPU = 6
DEFAULT_NATIVE_CPU = 8
DEFAULT_BACKGROUND_CPUS = (10, 12)
DEFAULT_WORKERS = 4
MAX_WORKERS = 16
_PRODUCTION_CONTROL_CPUS = (2, 4)
_WORK_CHUNK = 16_384


def _cpu_id(value: str) -> int:
    """Argparse converter for a non-negative Linux CPU id."""

    try:
        cpu = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("CPU must be a non-negative integer") from exc
    if cpu < 0:
        raise argparse.ArgumentTypeError("CPU must be a non-negative integer")
    return cpu


def _cpu_list(value: str) -> tuple[int, ...]:
    """Parse ``10,12`` or ``10-12`` without depending on project modules."""

    result: list[int] = []
    for part in str(value).split(","):
        text = part.strip()
        if not text:
            continue
        pieces = text.split("-")
        try:
            if len(pieces) == 1:
                lo = hi = int(pieces[0])
            elif len(pieces) == 2:
                lo, hi = (int(piece) for piece in pieces)
            else:
                raise ValueError
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid CPU list {value!r}; use e.g. 10,12"
            ) from exc
        if lo < 0 or hi < lo:
            raise argparse.ArgumentTypeError(
                f"invalid CPU list {value!r}; use non-negative ascending ranges"
            )
        result.extend(range(lo, hi + 1))
    if not result:
        raise argparse.ArgumentTypeError("CPU list must not be empty")
    # Repeated CPUs make the placement claim ambiguous and do not add useful
    # contention coverage.  Keep the CLI deterministic instead.
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("CPU list must not contain duplicates")
    return tuple(result)


def _worker_count(value: str) -> int:
    """Argparse converter keeping this experiment a bounded load test."""

    try:
        workers = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("workers must be an integer from 1 to 16") from exc
    if not 1 <= workers <= MAX_WORKERS:
        raise argparse.ArgumentTypeError("workers must be an integer from 1 to 16")
    return workers


def build_parser() -> argparse.ArgumentParser:
    """Return the public command-line parser for tests and shell users."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-csv", type=Path, required=True,
                        help="CSV containing q_meas_0 through q_meas_7")
    parser.add_argument("--ticks", type=int, default=600,
                        help="offline native ticks per placement (default: 600)")
    parser.add_argument("--control-cpu", type=_cpu_id, default=DEFAULT_CONTROL_CPU,
                        help=f"benchmark calling thread CPU (default: {DEFAULT_CONTROL_CPU})")
    parser.add_argument("--native-cpu", type=_cpu_id, default=DEFAULT_NATIVE_CPU,
                        help=f"wbc_rt child CPU (default: {DEFAULT_NATIVE_CPU})")
    parser.add_argument("--background-cpus", type=_cpu_list,
                        default=DEFAULT_BACKGROUND_CPUS,
                        help="background CPUs, comma/range list (default: 10,12)")
    parser.add_argument("--workers", type=_worker_count, default=DEFAULT_WORKERS,
                        help=f"bounded CPU workers (default: {DEFAULT_WORKERS}, max: {MAX_WORKERS})")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help=f"offline joint-admittance YAML (default: {DEFAULT_CONFIG})")
    parser.add_argument("--output", type=Path, required=True,
                        help="JSON result path")
    return parser


def _physical_siblings(cpu: int) -> set[int]:
    """Read Linux SMT siblings, falling back to the CPU itself."""

    try:
        from rm75_control.control.admittance_common.cpu_resources import physical_siblings

        return set(physical_siblings(int(cpu)))
    except (ImportError, OSError, ValueError, TypeError):
        return {int(cpu)}


def _validate_cpu_layout(
    control_cpu: int,
    native_cpu: int,
    background_cpus: Iterable[int],
    *,
    allowed_cpus: Iterable[int] | None = None,
) -> tuple[int, ...]:
    """Reject unsafe/meaningless placements before any worker is started.

    The real service reserves logical CPUs 2 and 4.  Include their SMT
    siblings when sysfs exposes them, so a background worker cannot silently
    share either physical control core.  This function is pure with respect
    to affinity: it reads the current process mask and never changes it.
    """

    control = int(control_cpu)
    native = int(native_cpu)
    background = tuple(int(cpu) for cpu in background_cpus)
    if control < 0 or native < 0 or any(cpu < 0 for cpu in background):
        raise ValueError("benchmark CPU ids must be non-negative")
    if control == native:
        raise ValueError("control_cpu and native_cpu must be different")
    if not background:
        raise ValueError("background_cpus must not be empty")

    reserved = set(_PRODUCTION_CONTROL_CPUS)
    for cpu in _PRODUCTION_CONTROL_CPUS:
        reserved.update(_physical_siblings(cpu))
    requested = {control, native, *background}
    overlap = sorted(requested & reserved)
    if overlap:
        raise ValueError(
            "refusing production control CPU placement; choose CPUs away from "
            f"2/4 and their SMT siblings (requested={overlap})"
        )

    control_core = _physical_siblings(control)
    native_core = _physical_siblings(native)
    if control_core & native_core:
        raise ValueError(
            "control_cpu and native_cpu must be on different physical cores "
            f"(control={sorted(control_core)}, native={sorted(native_core)})"
        )
    background_core_overlap = (set().union(*(_physical_siblings(cpu) for cpu in background))
                               & (control_core | native_core))
    if background_core_overlap:
        raise ValueError(
            "background CPUs must not share a physical core with control/native "
            f"(SMT overlap={sorted(background_core_overlap)})"
        )

    if allowed_cpus is None:
        try:
            allowed = set(os.sched_getaffinity(0))
        except (AttributeError, OSError, ValueError) as exc:
            raise ValueError("Linux CPU affinity is required for this benchmark") from exc
    else:
        allowed = {int(cpu) for cpu in allowed_cpus}
    missing = sorted(requested - allowed)
    if missing:
        raise ValueError(
            f"requested benchmark CPUs are outside this process cpuset: {missing}"
        )
    return background


def _read_seed(seed_csv: Path) -> list[float]:
    """Read and validate the first measured pose without importing NumPy."""

    path = Path(seed_csv).expanduser().resolve()
    try:
        with path.open(newline="") as handle:
            row = next(csv.DictReader(handle))
    except (OSError, StopIteration) as exc:
        raise ValueError(f"seed CSV has no readable data row: {path}") from exc
    values: list[float] = []
    for index in range(8):
        field = f"q_meas_{index}"
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"seed CSV first row lacks finite {field}") from exc
        if not (value == value and abs(value) != float("inf")):
            raise ValueError(f"seed CSV first row lacks finite {field}")
        values.append(value)
    return values


def _assert_offline_config(config: Path) -> Path:
    """Reject an explicit shared-SHM request before native can start.

    ``run_case`` always replaces its config value with a UUID prefix.  Still,
    rejecting a configured prefix here makes the safety contract visible and
    prevents a future runtime change from turning this tool into a production
    SHM client accidentally.
    """

    path = Path(config).expanduser().resolve()
    try:
        import yaml

        raw = yaml.safe_load(path.read_text()) or {}
    except (OSError, UnicodeError, ImportError, ValueError, TypeError) as exc:
        raise ValueError(f"cannot read offline config {path}: {exc}") from exc
    inner = raw.get("inner") if isinstance(raw, dict) else None
    if isinstance(inner, dict) and "native_shm_prefix" in inner:
        raise ValueError(
            "offline contention benchmark refuses configured native_shm_prefix; "
            "remove the shared production SHM setting"
        )
    return path


def _cpu_burn(
    stop: Any,
    ready: Any,
    worker_index: int,
    cpu: int,
    seed: int,
) -> None:
    """Bounded standard-library CPU load running only in this child process."""

    report: dict[str, Any] = {
        "worker": int(worker_index),
        "pid": os.getpid(),
        "requested_cpu": int(cpu),
        "applied": False,
        "mask_before": None,
        "mask_during": None,
        "error": None,
    }
    try:
        report["mask_before"] = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {int(cpu)})
        report["mask_during"] = sorted(os.sched_getaffinity(0))
        report["applied"] = report["mask_during"] == [int(cpu)]
        if not report["applied"]:
            report["error"] = "affinity mask was not exactly the requested CPU"
    except (AttributeError, OSError, ValueError, TypeError) as exc:
        report["error"] = f"affinity:{type(exc).__name__}: {exc}"
    try:
        ready.put(report)
    except (BrokenPipeError, EOFError, OSError):
        return
    if report["error"]:
        return

    # Integer arithmetic keeps the load deterministic and independent of
    # NumPy/OpenMP.  Check the stop event once per modest chunk so teardown is
    # prompt even if a controller run exits on its first fault.
    state = (0x9E3779B9 ^ (int(seed) + worker_index * 0x45D9F3B)) & 0xFFFFFFFF
    while not stop.is_set():
        for _ in range(_WORK_CHUNK):
            state ^= (state << 13) & 0xFFFFFFFF
            state ^= state >> 17
            state ^= (state << 5) & 0xFFFFFFFF
            state &= 0xFFFFFFFF


def _start_burners(
    cpus: tuple[int, ...],
    *,
    worker_count: int = DEFAULT_WORKERS,
    seed: int = 0,
) -> tuple[Any, list[Any], Any, list[dict[str, Any]]]:
    """Start finite-process CPU load and verify each child affinity."""

    if not 1 <= int(worker_count) <= MAX_WORKERS:
        raise ValueError(f"worker_count must be in [1, {MAX_WORKERS}]")
    if not cpus:
        raise ValueError("contention CPU list must not be empty")
    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    ready = ctx.Queue()
    workers: list[Any] = []
    assigned = tuple(cpus[index % len(cpus)] for index in range(int(worker_count)))
    try:
        for index, cpu in enumerate(assigned):
            worker = ctx.Process(
                target=_cpu_burn,
                args=(stop, ready, index, int(cpu), int(seed)),
                name=f"native-contention-{index}",
            )
            worker.daemon = True
            worker.start()
            workers.append(worker)

        reports: list[dict[str, Any]] = []
        deadline = time.monotonic() + 5.0
        while len(reports) < len(workers) and time.monotonic() < deadline:
            try:
                report = ready.get(timeout=min(0.1, max(0.001, deadline - time.monotonic())))
            except Exception:
                continue
            reports.append(dict(report))
        if len(reports) != len(workers):
            raise RuntimeError("CPU contention worker did not report startup")
        reports.sort(key=lambda report: int(report.get("worker", -1)))
        errors = [report for report in reports if report.get("error") or not report.get("applied")]
        if errors:
            raise RuntimeError(f"CPU contention worker affinity failed: {errors}")
        return stop, workers, ready, reports
    except BaseException:
        stop.set()
        for worker in workers:
            worker.join(timeout=1.0)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=1.0)
        ready.close()
        ready.join_thread()
        raise


def _stop_burners(stop: Any, workers: Iterable[Any], ready: Any | None = None) -> None:
    """Stop and reap only the workers created by this benchmark."""

    stop.set()
    for worker in workers:
        worker.join(timeout=1.0)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=1.0)
    if ready is not None:
        ready.close()
        ready.join_thread()


def _timeout_count(case: dict[str, Any]) -> int:
    """Count native timeout statuses reported by ``run_case``."""

    statuses = case.get("statuses") or {}
    return sum(int(count) for name, count in statuses.items()
               if "timeout" in str(name).lower())


def _case_summary(case: dict[str, Any], *, mode: str, workers: list[dict[str, Any]]) -> dict[str, Any]:
    """Expose the timing comparison while retaining the full run-case data."""

    ticks = int(case.get("ticks", 0))
    timeouts = _timeout_count(case)
    timing = case.get("timing") or {}
    roundtrip = timing.get("native_roundtrip_ms") or {}
    return {
        "mode": mode,
        "worker_affinity": workers,
        "ticks": ticks,
        "timeouts": timeouts,
        "timeout_rate": (timeouts / ticks if ticks else 0.0),
        "native_roundtrip_ms": roundtrip,
        "statuses": case.get("statuses", {}),
        "first_failure": case.get("first_failure"),
        "run_case": case,
    }


def _comparison(cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build a compact same-core minus background comparison."""

    same = cases.get("native_same_core", {})
    background = cases.get("background_cpu", {})
    same_rt = same.get("native_roundtrip_ms") or {}
    bg_rt = background.get("native_roundtrip_ms") or {}

    def delta(field: str) -> float | None:
        if field not in same_rt or field not in bg_rt:
            return None
        return float(same_rt[field]) - float(bg_rt[field])

    return {
        "roundtrip_delta_same_minus_background_ms": {
            field: delta(field) for field in ("p50_ms", "p99_ms", "max_ms")
        },
        "timeouts": {
            "native_same_core": int(same.get("timeouts", 0)),
            "background_cpu": int(background.get("timeouts", 0)),
        },
    }


def run_contention_benchmark(
    *,
    seed_csv: Path,
    ticks: int = 600,
    control_cpu: int = DEFAULT_CONTROL_CPU,
    native_cpu: int = DEFAULT_NATIVE_CPU,
    background_cpus: Iterable[int] = DEFAULT_BACKGROUND_CPUS,
    config: Path = DEFAULT_CONFIG,
    worker_count: int = DEFAULT_WORKERS,
    seed: int = 0,
) -> dict[str, Any]:
    """Run both offline contention placements and return JSON-safe data."""

    if int(ticks) < 1:
        raise ValueError("ticks must be positive")
    config_path = _assert_offline_config(config)
    seed_values = _read_seed(seed_csv)
    background = _validate_cpu_layout(control_cpu, native_cpu, background_cpus)
    if not 1 <= int(worker_count) <= MAX_WORKERS:
        raise ValueError(f"worker_count must be in [1, {MAX_WORKERS}]")

    cases: dict[str, dict[str, Any]] = {}
    layouts = (
        ("native_same_core", (int(native_cpu),)),
        ("background_cpu", background),
    )
    for mode, load_cpus in layouts:
        stop = workers = ready = None
        reports: list[dict[str, Any]] = []
        try:
            stop, workers, ready, reports = _start_burners(
                tuple(load_cpus), worker_count=worker_count, seed=seed
            )
            # ``run_case`` is the only native client entry point here.  It
            # creates ``runtime_profile_<uuid>`` SHM and never opens hardware.
            case = run_case(
                seed_values,
                ticks=int(ticks),
                config=config_path,
                control_cpu=int(control_cpu),
                native_cpu=int(native_cpu),
            )
            cases[mode] = _case_summary(case, mode=mode, workers=reports)
        finally:
            if stop is not None and workers is not None:
                _stop_burners(stop, workers, ready)

    return {
        "offline": True,
        "hardware_connected": False,
        "native_only": True,
        "native_shm_policy": "unique UUID per run_case; configured/shared production SHM refused",
        "seed_csv": str(Path(seed_csv).expanduser().resolve()),
        "config": str(config_path),
        "ticks_requested": int(ticks),
        "control_cpu": int(control_cpu),
        "native_cpu": int(native_cpu),
        "background_cpus": list(background),
        "worker_count": int(worker_count),
        "cases": cases,
        "comparison": _comparison(cases),
    }


def main(argv: list[str] | None = None) -> int:
    # Set numeric pools before run_case imports NumPy/SciPy.  This benchmark's
    # workers are standard-library-only and remain intentionally small.
    from rm75_control.control.admittance_common.observer_runtime import limit_numeric_threads

    limit_numeric_threads(threads=1)
    args = build_parser().parse_args(argv)
    result = run_contention_benchmark(
        seed_csv=args.seed_csv,
        ticks=args.ticks,
        control_cpu=args.control_cpu,
        native_cpu=args.native_cpu,
        background_cpus=args.background_cpus,
        config=args.config,
        worker_count=args.workers,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name, case in result["cases"].items():
        roundtrip = case.get("native_roundtrip_ms") or {}
        print(
            f"[{name}] native_roundtrip p99={roundtrip.get('p99_ms', 'n/a')} ms "
            f"max={roundtrip.get('max_ms', 'n/a')} ms "
            f"timeouts={case.get('timeouts', 0)}",
            flush=True,
        )
    print(f"[OK] offline contention result {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
