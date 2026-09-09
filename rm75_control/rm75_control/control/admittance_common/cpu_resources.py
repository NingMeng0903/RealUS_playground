"""Physical-core separation for the controller and its standalone observers.

Only changes the calling process's threads. Never changes another application's
affinity, scheduler policy or priority. All operations are best effort on Linux.
"""

from __future__ import annotations

from collections.abc import Iterable
import os
from pathlib import Path


def parse_cpu_list(value: str) -> set[int]:
    result: set[int] = set()
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        ends = part.split("-")
        if len(ends) == 1:
            lo = hi = int(ends[0])
        elif len(ends) == 2:
            lo, hi = map(int, ends)
        else:
            raise ValueError(f"invalid CPU list: {value!r}")
        if lo < 0 or hi < lo:
            raise ValueError(f"invalid CPU list: {value!r}")
        result.update(range(lo, hi + 1))
    return result


def physical_siblings(cpu: int, *, sysfs: Path = Path("/sys/devices/system/cpu")) -> set[int]:
    try:
        return parse_cpu_list((sysfs / f"cpu{int(cpu)}/topology/thread_siblings_list").read_text())
    except (OSError, ValueError):
        return {int(cpu)}


def configured_control_cpus(config_path: str | Path | None = None) -> set[int]:
    """Read reserved controller CPUs, honoring an explicit config path.

    ``RM75_OBSERVER_AVOID_CPUS`` remains the explicit environment override for
    existing observer entry points when no config path is supplied.  A
    caller-supplied ``config_path`` takes precedence so
    ``run_controller --config`` always follows its actual timing section.
    """
    if config_path is None:
        override = os.environ.get("RM75_OBSERVER_AVOID_CPUS")
        if override is not None:
            return parse_cpu_list(override)
    config = (
        Path(config_path)
        if config_path is not None
        else Path(__file__).resolve().parents[4] / "peirastic/configs/controller.yaml"
    )
    try:
        import yaml
        timing = (yaml.safe_load(config.read_text()) or {}).get("timing", {})
        return {int(timing[key]) for key in ("control_cpu", "native_cpu")
                if timing.get(key) is not None}
    except (ImportError, OSError, ValueError, TypeError):
        return set()


def prepare_background_cpus(
    *,
    avoid_cpus: Iterable[int] | None = None,
    config_path: str | Path | None = None,
) -> list[int]:
    """Keep observers off configured control cores, including SMT siblings.

    Existing task-specific masks are narrowed, never widened. An empty result
    leaves a constrained task alone rather than assigning an invalid mask.
    ``avoid_cpus`` takes precedence over the config/environment lookup.  Call
    before starting camera, BLAS, Torch or Genesis worker threads.
    """
    try:
        configured = (
            {int(cpu) for cpu in avoid_cpus}
            if avoid_cpus is not None
            else (
                configured_control_cpus(config_path)
                if config_path is not None
                else configured_control_cpus()
            )
        )
        avoid = set().union(*(physical_siblings(cpu) for cpu in configured))
        allowed = set(os.sched_getaffinity(0))
        available = allowed - avoid
        if not avoid or not available:
            return sorted(allowed)
        tasks = list(Path("/proc/self/task").iterdir())
        for task in tasks:
            try:
                tid = int(task.name)
                mask = set(os.sched_getaffinity(tid)) - avoid
                if mask:
                    os.sched_setaffinity(tid, mask)
            except (OSError, ValueError):
                continue
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError, ValueError):
        return []


def scheduling_note(control_cpu: int | None, native_cpu: int | None) -> str:
    """Expose explicit placement and accidental SMT sharing at startup."""
    shared = (control_cpu is not None and native_cpu is not None
              and native_cpu in physical_siblings(control_cpu))
    note = f"control_cpu={control_cpu} native_cpu={native_cpu}"
    if shared:
        note += " (shared physical core: choose separate core IDs)"
    return note
