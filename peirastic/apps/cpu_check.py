"""Read-only CPU placement and scheduling observer.

This entry point intentionally has no robot, shared-memory, HDF5, camera, or
numerical-library imports.  It inspects Linux ``/proc`` and changes only the
observer process itself: its niceness and affinity are prepared before the
short sampling interval.  It never changes a controller or any other task.

The report separates *eligibility* (a task's allowed mask overlaps the
reserved controller/SMT CPUs) from scheduling evidence.  ``schedstat``
run-queue time is useful evidence that a task waited to run, but by itself it
does not prove that the robot controller was delayed by that task.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


CONTROL_CPU = 2
NATIVE_CPU = 4
DEFAULT_RESERVED_CPUS = frozenset({2, 3, 4, 5})
DEFAULT_SECONDS = 3.0
MAX_SECONDS = 60.0
TOP_BACKGROUND = 20

_PROC_ROOT = Path("/proc")
_SYS_CPU_ROOT = Path("/sys/devices/system/cpu")
_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "controller.yaml"


def parse_cpu_list(value: str | bytes | None) -> set[int]:
    """Parse a Linux ``Cpus_allowed_list`` value."""

    if value is None:
        return set()
    if isinstance(value, bytes):
        value = value.decode("ascii", "replace")
    result: set[int] = set()
    for part in str(value).strip().split(","):
        part = part.strip()
        if not part:
            continue
        ends = part.split("-")
        try:
            if len(ends) == 1:
                lo = hi = int(ends[0], 10)
            elif len(ends) == 2:
                lo, hi = (int(item, 10) for item in ends)
            else:
                raise ValueError
        except ValueError as exc:
            raise ValueError(f"invalid CPU list: {value!r}") from exc
        if lo < 0 or hi < lo:
            raise ValueError(f"invalid CPU list: {value!r}")
        result.update(range(lo, hi + 1))
    return result


def parse_cpu_mask(value: str | bytes | None) -> set[int]:
    """Parse the hexadecimal ``Cpus_allowed`` form used by proc status."""

    if value is None:
        return set()
    if isinstance(value, bytes):
        value = value.decode("ascii", "replace")
    compact = str(value).strip().replace(",", "")
    if not compact:
        return set()
    try:
        bits = int(compact, 16)
    except ValueError as exc:
        raise ValueError(f"invalid CPU mask: {value!r}") from exc
    return {cpu for cpu in range(bits.bit_length()) if bits & (1 << cpu)}


def _safe_read_text(path: Path) -> str | None:
    """Read a proc file while tolerating a task exiting mid-snapshot."""

    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None


def _status_fields(text: str | None) -> dict[str, str]:
    if not text:
        return {}
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip()] = value.strip()
    return fields


def _read_status(path: Path) -> dict[str, str]:
    return _status_fields(_safe_read_text(path))


def affinity_from_status(fields: Mapping[str, str]) -> set[int]:
    """Return a task mask, accepting both proc status encodings."""

    value = fields.get("Cpus_allowed_list")
    if value:
        try:
            return parse_cpu_list(value)
        except ValueError:
            pass
    value = fields.get("Cpus_allowed")
    if value:
        try:
            return parse_cpu_mask(value)
        except ValueError:
            pass
    return set()


def _parse_stat(text: str | None) -> dict[str, int | str]:
    """Parse the few fields needed from ``/proc/<tid>/stat``.

    The comm field may contain spaces and closing parentheses, so split at the
    final ``) `` before interpreting the remaining fields.
    """

    if not text:
        return {}
    close = text.rfind(") ")
    if close < 0:
        return {}
    comm = text[:close]
    open_paren = comm.find("(")
    if open_paren >= 0:
        comm = comm[open_paren + 1 :]
    rest = text[close + 2 :].split()
    if not rest:
        return {}
    out: dict[str, int | str] = {"state": rest[0], "comm": comm}

    def integer(index: int, key: str) -> None:
        if index >= len(rest):
            return
        try:
            out[key] = int(rest[index], 10)
        except ValueError:
            return

    # rest[0] is field 3 (state); field N therefore has index N - 3.
    integer(1, "ppid")
    integer(11, "utime_ticks")
    integer(12, "stime_ticks")
    integer(15, "priority")
    integer(16, "nice")
    integer(36, "processor")
    return out


def _read_stat(path: Path) -> dict[str, int | str]:
    return _parse_stat(_safe_read_text(path))


def _parse_schedstat(text: str | None) -> tuple[int | None, int | None]:
    if not text:
        return None, None
    fields = text.split()
    if len(fields) < 2:
        return None, None
    try:
        return int(fields[0], 10), int(fields[1], 10)
    except ValueError:
        return None, None


def _read_schedstat(path: Path) -> tuple[int | None, int | None]:
    return _parse_schedstat(_safe_read_text(path))


def _read_cmdline(path: Path) -> list[str]:
    """Read command arguments for matching only; never return them in reports."""

    try:
        raw = path.read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return []
    return [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]


def _basename(value: str) -> str:
    # Path() does not treat a Windows-style separator as a separator on Linux.
    return value.replace("\\", "/").rsplit("/", 1)[-1]


def _module_basename(tokens: Sequence[str]) -> str:
    """Return a safe executable/script/module basename, never ``-c`` text."""

    if not tokens:
        return "unknown"
    first = _basename(tokens[0])
    interpreters = {
        "python",
        "python2",
        "python3",
        "python3.10",
        "python3.11",
        "python3.12",
        "pypy",
        "pypy3",
    }
    if first in interpreters:
        if "-c" in tokens:
            return first
        try:
            module_index = tokens.index("-m")
        except ValueError:
            module_index = -1
        if module_index >= 0 and module_index + 1 < len(tokens):
            module = tokens[module_index + 1].strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", module):
                return module.rsplit(".", 1)[-1]
        # Python options that take no argument may precede the script.
        for token in tokens[1:]:
            if token in {"-B", "-E", "-I", "-O", "-s", "-S", "-u", "-v", "-W"}:
                continue
            if token.startswith("-"):
                continue
            return _basename(token)
        return first
    if first in {"bash", "sh", "dash", "zsh", "fish"} and "-c" in tokens:
        return first
    if first:
        return first
    return "unknown"


def classify_command(tokens: Sequence[str]) -> str | None:
    """Classify only an executable, Python module, or script token.

    In particular, ``python -c '...wbc_rt...'`` and arbitrary command
    arguments are never inspected as process identity.
    """

    if not tokens:
        return None
    lowered = [token.lower().replace("\\", "/") for token in tokens]
    first = lowered[0].rsplit("/", 1)[-1]
    if first in {"wbc_rt", "wbc_rt.exe", "wbc_rt_native", "wbc_rt_native.exe"}:
        return "native"
    interpreters = {
        "python",
        "python2",
        "python3",
        "python3.10",
        "python3.11",
        "python3.12",
        "pypy",
        "pypy3",
    }
    if first in interpreters:
        if "-c" in lowered:
            return None
        try:
            index = lowered.index("-m")
        except ValueError:
            index = -1
        if index >= 0 and index + 1 < len(lowered):
            module = lowered[index + 1]
            if module in {"peirastic.apps.run_controller", "peirastic.realman8dof.daemon"}:
                return "controller"
        for token in lowered[1:]:
            if token.startswith("-"):
                continue
            script = token.rsplit("/", 1)[-1]
            if script in {"run_controller.py", "run_controller.pyc", "run_controller"}:
                return "controller"
            break
        return None
    if first in {"run_controller.py", "run_controller.pyc", "run_controller"}:
        return "controller"
    return None


def _read_siblings(cpu: int, sys_root: Path) -> set[int]:
    path = sys_root / f"cpu{int(cpu)}" / "topology" / "thread_siblings_list"
    text = _safe_read_text(path)
    if text:
        try:
            parsed = parse_cpu_list(text)
            if parsed:
                return parsed
        except ValueError:
            pass
    return {int(cpu)}


def configured_cpus(config_path: Path = _CONFIG_PATH) -> tuple[int, int]:
    """Read timing CPU IDs with a tiny parser, avoiding a YAML dependency."""

    control, native = CONTROL_CPU, NATIVE_CPU
    text = _safe_read_text(Path(config_path))
    if not text:
        return control, native
    in_timing = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line[:1].isspace():
            in_timing = stripped == "timing:"
            continue
        if not in_timing:
            continue
        match = re.match(r"\s*(control_cpu|native_cpu)\s*:\s*([^#\s]+)", line)
        if not match:
            continue
        key, value = match.groups()
        if value.lower() in {"null", "none"}:
            continue
        try:
            parsed = int(value, 10)
        except ValueError:
            continue
        if key == "control_cpu":
            control = parsed
        else:
            native = parsed
    return control, native


def reserved_cpus(
    control_cpu: int,
    native_cpu: int,
    *,
    sys_root: Path = _SYS_CPU_ROOT,
) -> set[int]:
    """Resolve control/native SMT siblings, retaining the documented fallback."""

    control_siblings = _read_siblings(control_cpu, sys_root)
    native_siblings = _read_siblings(native_cpu, sys_root)
    resolved = control_siblings | native_siblings
    if (
        (control_cpu, native_cpu) == (CONTROL_CPU, NATIVE_CPU)
        and (control_siblings <= {CONTROL_CPU} or native_siblings <= {NATIVE_CPU})
    ):
        # A restricted container may hide topology files.  The machine config
        # still reserves both known SMT pairs 2/3 and 4/5.
        return set(DEFAULT_RESERVED_CPUS)
    return resolved


def _numeric_proc_dirs(proc_root: Path) -> list[int]:
    try:
        entries = proc_root.iterdir()
    except (FileNotFoundError, PermissionError, OSError):
        return []
    pids: list[int] = []
    for entry in entries:
        if entry.name.isdigit():
            try:
                pids.append(int(entry.name))
            except ValueError:
                continue
    return sorted(set(pids))


def _parent_pid(pid: int, proc_root: Path) -> int | None:
    fields = _read_status(proc_root / str(pid) / "status")
    try:
        return int(fields["PPid"], 10)
    except (KeyError, TypeError, ValueError):
        stat = _read_stat(proc_root / str(pid) / "stat")
        value = stat.get("ppid")
        return int(value) if isinstance(value, int) else None


def ancestor_pids(proc_root: Path = _PROC_ROOT, *, self_pid: int | None = None) -> set[int]:
    """Return self plus its parent chain, tolerating proc races."""

    current = int(os.getpid() if self_pid is None else self_pid)
    ignored: set[int] = set()
    for _ in range(128):
        if current <= 0 or current in ignored:
            break
        ignored.add(current)
        parent = _parent_pid(current, proc_root)
        if parent is None or parent == current:
            break
        current = parent
    return ignored


def _thread_ids(pid: int, proc_root: Path) -> list[int]:
    task_root = proc_root / str(pid) / "task"
    try:
        entries = task_root.iterdir()
    except (FileNotFoundError, PermissionError, OSError):
        return [pid]
    tids: list[int] = []
    for entry in entries:
        if entry.name.isdigit():
            try:
                tids.append(int(entry.name))
            except ValueError:
                continue
    return sorted(set(tids)) or [pid]


def _thread_record(pid: int, tid: int, proc_root: Path) -> dict[str, Any]:
    task = proc_root / str(pid) / "task" / str(tid)
    status = _read_status(task / "status")
    stat = _read_stat(task / "stat")
    # Some fake proc trees provide only process-level stat/status.  Linux's
    # task files normally exist, but the fallback makes races harmless.
    if not status:
        status = _read_status(proc_root / str(pid) / "status")
    if not stat:
        stat = _read_stat(proc_root / str(pid) / "stat")
    cpu_ns, wait_ns = _read_schedstat(task / "schedstat")
    if cpu_ns is None and tid == pid:
        cpu_ns, wait_ns = _read_schedstat(proc_root / str(pid) / "schedstat")
    affinity = affinity_from_status(status)
    comm = str(status.get("Name") or stat.get("comm") or _safe_read_text(task / "comm") or "unknown").strip()
    return {
        "tid": tid,
        "comm": comm,
        "cpus_allowed": sorted(affinity),
        "cpu_ns": cpu_ns,
        "wait_ns": wait_ns,
        "nice": stat.get("nice"),
        "priority": stat.get("priority"),
        "processor": stat.get("processor"),
        "state": stat.get("state"),
    }


def _process_record(pid: int, proc_root: Path, reserved: set[int]) -> dict[str, Any] | None:
    process_root = proc_root / str(pid)
    status = _read_status(process_root / "status")
    stat = _read_stat(process_root / "stat")
    comm_text = _safe_read_text(process_root / "comm")
    comm = str(status.get("Name") or stat.get("comm") or comm_text or "unknown").strip()
    tokens = _read_cmdline(process_root / "cmdline")
    kind = classify_command(tokens)
    module = _module_basename(tokens)
    threads = [_thread_record(pid, tid, proc_root) for tid in _thread_ids(pid, proc_root)]
    if not threads:
        return None
    overlap = [thread for thread in threads if set(thread["cpus_allowed"]) & reserved]
    parent = status.get("PPid")
    try:
        parent_int: int | None = int(parent, 10) if parent is not None else None
    except ValueError:
        parent_int = None
    # No cmdline or argument text is retained.  module is a basename only.
    return {
        "pid": pid,
        "comm": comm,
        "module": module,
        "kind": kind,
        "ppid": parent_int,
        "thread_count": len(threads),
        "overlap_tid_count": len(overlap),
        "threads": threads,
    }


def collect_processes(
    *,
    proc_root: Path = _PROC_ROOT,
    reserved: Iterable[int] = DEFAULT_RESERVED_CPUS,
    ignored_pids: Iterable[int] = (),
) -> list[dict[str, Any]]:
    """Collect a best-effort process/thread snapshot from proc."""

    reserved_set = set(reserved)
    ignored = set(int(pid) for pid in ignored_pids)
    processes: list[dict[str, Any]] = []
    for pid in _numeric_proc_dirs(proc_root):
        if pid in ignored:
            continue
        try:
            record = _process_record(pid, proc_root, reserved_set)
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, ValueError):
            # A process can disappear between any two proc reads.
            continue
        if record is not None:
            processes.append(record)
    return processes


def _safe_process_view(
    process: Mapping[str, Any],
    *,
    thread_deltas: Mapping[int, tuple[float | None, float | None]] | None = None,
) -> dict[str, Any]:
    """Drop internal-only fields before returning a process in a report."""

    actual_affinity = sorted(
        {
            int(cpu)
            for thread in process.get("threads", [])
            for cpu in thread.get("cpus_allowed", [])
        }
    )
    safe_threads: list[dict[str, Any]] = []
    total_cpu = 0.0
    total_wait = 0.0
    have_cpu = False
    have_wait = False
    for thread in process.get("threads", []):
        safe_thread = {
            key: thread.get(key)
            for key in (
                "tid",
                "comm",
                "cpus_allowed",
                "nice",
                "priority",
                "processor",
                "state",
            )
        }
        tid = thread.get("tid")
        cpu_ms, wait_ms = (thread_deltas or {}).get(tid, (None, None))
        safe_thread["cpu_ms"] = cpu_ms
        safe_thread["runqueue_wait_ms"] = wait_ms
        if cpu_ms is not None:
            total_cpu += cpu_ms
            have_cpu = True
        if wait_ms is not None:
            total_wait += wait_ms
            have_wait = True
        safe_threads.append(safe_thread)
    return {
        "pid": process.get("pid"),
        "comm": process.get("comm"),
        "module": process.get("module"),
        "kind": process.get("kind"),
        "ppid": process.get("ppid"),
        "thread_count": process.get("thread_count", 0),
        "overlap_tid_count": process.get("overlap_tid_count", 0),
        "actual_affinity": actual_affinity,
        "sample_cpu_ms": total_cpu if have_cpu else None,
        "sample_runqueue_wait_ms": total_wait if have_wait else None,
        "threads": safe_threads,
    }


def _schedstat_deltas(
    first: Mapping[str, Any] | None,
    last: Mapping[str, Any] | None,
) -> dict[int, tuple[float | None, float | None]]:
    """Return per-TID schedstat deltas in milliseconds for one process."""

    first_threads = {t.get("tid"): t for t in (first or {}).get("threads", [])}
    last_threads = {t.get("tid"): t for t in (last or {}).get("threads", [])}
    result: dict[int, tuple[float | None, float | None]] = {}
    for tid in set(first_threads) | set(last_threads):
        if not isinstance(tid, int):
            continue
        a, b = first_threads.get(tid), last_threads.get(tid)
        if a is None or b is None:
            result[tid] = (None, None)
            continue
        cpu_ns = _delta(a.get("cpu_ns"), b.get("cpu_ns"))
        wait_ns = _delta(a.get("wait_ns"), b.get("wait_ns"))
        result[tid] = (
            cpu_ns / 1_000_000.0 if cpu_ns is not None else None,
            wait_ns / 1_000_000.0 if wait_ns is not None else None,
        )
    return result


def _kind_process_views(
    kind: str,
    before: Iterable[Mapping[str, Any]],
    after: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    before_by_pid = {
        p.get("pid"): p
        for p in before
        if p.get("kind") == kind and isinstance(p.get("pid"), int)
    }
    after_by_pid = {
        p.get("pid"): p
        for p in after
        if p.get("kind") == kind and isinstance(p.get("pid"), int)
    }
    views: list[dict[str, Any]] = []
    for pid in sorted(set(before_by_pid) | set(after_by_pid)):
        first, last = before_by_pid.get(pid), after_by_pid.get(pid)
        source = last or first
        if source is not None:
            views.append(_safe_process_view(source, thread_deltas=_schedstat_deltas(first, last)))
    return views


def _delta(value_a: Any, value_b: Any) -> int | None:
    if not isinstance(value_a, int) or not isinstance(value_b, int):
        return None
    return max(0, value_b - value_a)


def _background_deltas(
    before: Iterable[Mapping[str, Any]],
    after: Iterable[Mapping[str, Any]],
    reserved: set[int],
    *,
    excluded_pids: Iterable[int] = (),
) -> list[dict[str, Any]]:
    excluded = set(int(pid) for pid in excluded_pids)
    before_by_pid = {
        p.get("pid"): p
        for p in before
        if isinstance(p.get("pid"), int) and p.get("pid") not in excluded
    }
    after_by_pid = {
        p.get("pid"): p
        for p in after
        if isinstance(p.get("pid"), int) and p.get("pid") not in excluded
    }
    rows: list[dict[str, Any]] = []
    for pid in sorted(set(before_by_pid) | set(after_by_pid)):
        first = before_by_pid.get(pid)
        last = after_by_pid.get(pid)
        if first is None and last is None:
            continue
        first_threads = {t.get("tid"): t for t in (first or {}).get("threads", [])}
        last_threads = {t.get("tid"): t for t in (last or {}).get("threads", [])}
        tids = sorted(set(first_threads) | set(last_threads))
        eligible_start = [
            tid for tid, thread in first_threads.items()
            if isinstance(tid, int) and set(thread.get("cpus_allowed", [])) & reserved
        ]
        eligible_end = [
            tid for tid, thread in last_threads.items()
            if isinstance(tid, int) and set(thread.get("cpus_allowed", [])) & reserved
        ]
        overlap_tids = sorted(set(eligible_start) | set(eligible_end))
        if not overlap_tids:
            continue
        cpu_ns = 0
        wait_ns = 0
        have_cpu = False
        have_wait = False
        sampled = 0
        missing = 0
        for tid in overlap_tids:
            a, b = first_threads.get(tid), last_threads.get(tid)
            if a is None or b is None:
                missing += 1
                continue
            cpu = _delta(a.get("cpu_ns"), b.get("cpu_ns"))
            wait = _delta(a.get("wait_ns"), b.get("wait_ns"))
            sampled += 1
            if cpu is not None:
                cpu_ns += cpu
                have_cpu = True
            if wait is not None:
                wait_ns += wait
                have_wait = True
        source = last or first or {}
        rows.append(
            {
                "pid": pid,
                "comm": source.get("comm"),
                "module": source.get("module"),
                "thread_count": source.get("thread_count", len(tids)),
                "overlap_tid_count": len(overlap_tids),
                "overlap_tid_count_start": len(set(eligible_start)),
                "overlap_tid_count_end": len(set(eligible_end)),
                "overlap_tids": overlap_tids,
                "sampled_tid_count": sampled,
                "missing_tid_count": missing,
                "cpu_ms": cpu_ns / 1_000_000.0 if have_cpu else None,
                # schedstat's second counter is cumulative run-queue wait for
                # all overlapping TIDs during this sample, not a max delay.
                "runqueue_wait_ms": wait_ns / 1_000_000.0 if have_wait else None,
                "wait_ms": wait_ns / 1_000_000.0 if have_wait else None,
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["cpu_ms"] or 0.0) + float(row["wait_ms"] or 0.0),
            float(row["cpu_ms"] or 0.0),
        ),
        reverse=True,
    )
    return rows


def _prepare_affinity(reserved: set[int]) -> dict[str, Any]:
    """Set only this process/thread group to a low-priority background mask."""

    info: dict[str, Any] = {
        "affinity_before": [],
        "affinity_after": [],
        "nice_before": None,
        "nice_after": None,
        "set_affinity": False,
        "set_nice": False,
        "errors": [],
    }
    try:
        before = set(os.sched_getaffinity(0))
        info["affinity_before"] = sorted(before)
        available = before - reserved
        if available:
            # sched_setaffinity(0, ...) applies to this thread.  Narrow every
            # currently visible self-thread as well; no other PID is touched.
            tids: list[int] = []
            try:
                tids = [int(path.name) for path in (Path("/proc/self/task")).iterdir() if path.name.isdigit()]
            except (FileNotFoundError, PermissionError, OSError, ValueError):
                tids = [0]
            for tid in tids or [0]:
                try:
                    os.sched_setaffinity(tid, available)
                    info["set_affinity"] = True
                except (AttributeError, PermissionError, OSError) as exc:
                    info["errors"].append(f"affinity:{type(exc).__name__}")
        info["affinity_after"] = sorted(os.sched_getaffinity(0))
    except (AttributeError, PermissionError, OSError) as exc:
        info["errors"].append(f"affinity:{type(exc).__name__}")
    try:
        current = int(os.getpriority(os.PRIO_PROCESS, 0))
        info["nice_before"] = current
        target = max(current, 10)
        if target != current:
            os.setpriority(os.PRIO_PROCESS, 0, target)
            info["set_nice"] = True
        info["nice_after"] = int(os.getpriority(os.PRIO_PROCESS, 0))
    except (AttributeError, PermissionError, OSError) as exc:
        info["errors"].append(f"nice:{type(exc).__name__}")
    return info


def bootstrap_observer(reserved: Iterable[int] = DEFAULT_RESERVED_CPUS) -> dict[str, Any]:
    """Prepare only this observer before sampling; safe on non-Linux hosts."""

    return _prepare_affinity(set(reserved))


def _timed_snapshot(
    *,
    seconds: float,
    proc_root: Path,
    reserved: set[int],
    ignored_pids: set[int],
    monotonic: Any | None = None,
    sleep: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float, int, int]:
    monotonic = time.monotonic if monotonic is None else monotonic
    sleep = time.sleep if sleep is None else sleep
    start_ns = time.time_ns()
    t0 = monotonic()
    before = collect_processes(proc_root=proc_root, reserved=reserved, ignored_pids=ignored_pids)
    if seconds > 0:
        sleep(seconds)
    after = collect_processes(proc_root=proc_root, reserved=reserved, ignored_pids=ignored_pids)
    t1 = monotonic()
    return before, after, max(0.0, float(t1 - t0)), start_ns, time.time_ns()


def build_report(
    *,
    requested_seconds: float,
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    elapsed_seconds: float,
    wall_start_ns: int,
    wall_end_ns: int,
    placement: Mapping[str, Any],
    control_cpu: int,
    native_cpu: int,
    reserved: set[int],
) -> dict[str, Any]:
    controllers = _kind_process_views("controller", before, after)
    natives = _kind_process_views("native", before, after)
    intended_pids = {
        int(process["pid"])
        for process in list(before) + list(after)
        if process.get("kind") in {"controller", "native"}
        and isinstance(process.get("pid"), int)
    }
    background = _background_deltas(before, after, reserved, excluded_pids=intended_pids)
    wait_evidence = [
        row
        for row in background
        if row.get("runqueue_wait_ms") is not None and row["runqueue_wait_ms"] > 0.0
    ]
    before_threads = sum(int(p.get("thread_count", 0)) for p in before)
    after_threads = sum(int(p.get("thread_count", 0)) for p in after)
    report = {
        "schema": "peirastic.cpu_check.v1",
        "read_only": True,
        "sampling": {
            "requested_seconds": float(requested_seconds),
            "elapsed_seconds": float(elapsed_seconds),
            "wall_start_ns": int(wall_start_ns),
            "wall_end_ns": int(wall_end_ns),
            "clock": "CLOCK_MONOTONIC for duration; CLOCK_REALTIME ns for report time",
        },
        "placement": {
            "control_cpu": int(control_cpu),
            "native_cpu": int(native_cpu),
            "smt_reserved_cpus": sorted(int(cpu) for cpu in reserved),
            "observer": dict(placement),
        },
        "counts": {
            "processes_start": len(before),
            "processes_end": len(after),
            "threads_start": before_threads,
            "threads_end": after_threads,
            "controller_processes": len(controllers),
            "native_processes": len(natives),
            "background_overlap_processes": len(background),
            "background_overlap_threads": sum(int(row["overlap_tid_count"]) for row in background),
            "background_schedstat_wait_evidence_processes": len(wait_evidence),
        },
        "controller": controllers,
        "native": natives,
        "controller_running": bool(controllers),
        "native_running": bool(natives),
        "scope": {
            "snapshot_only": True,
            "historical_attribution": False,
            "note": "Only tasks visible during this sample are reported; this is not a historical process census.",
        },
        "background_candidates": background[:TOP_BACKGROUND],
        "contention": {
            "eligibility": bool(background),
            "eligible_processes": len(background),
            "eligible_threads": sum(int(row["overlap_tid_count"]) for row in background),
            "schedstat_wait_evidence": bool(wait_evidence),
            "schedstat_wait_evidence_processes": len(wait_evidence),
            "actual_controller_contention_proven": False,
            "note": (
                "Affinity overlap is eligibility. Positive schedstat wait is scheduling evidence; "
                "the value is aggregate run-queue wait across overlapping TIDs during this sample; "
                "it does not prove controller/native interference or historical causation."
            ),
        },
    }
    return report


def collect_report(
    seconds: float = DEFAULT_SECONDS,
    *,
    proc_root: Path = _PROC_ROOT,
    sys_root: Path = _SYS_CPU_ROOT,
    config_path: Path = _CONFIG_PATH,
    self_pid: int | None = None,
    monotonic: Any | None = None,
    sleep: Any | None = None,
) -> dict[str, Any]:
    """Bootstrap, sample, and build a JSON-serializable observer report."""

    seconds = validate_seconds(seconds)
    control_cpu, native_cpu = configured_cpus(config_path)
    reserved = reserved_cpus(control_cpu, native_cpu, sys_root=sys_root)
    placement = bootstrap_observer(reserved)
    ignored = ancestor_pids(proc_root, self_pid=self_pid)
    before, after, elapsed, wall_start, wall_end = _timed_snapshot(
        seconds=seconds,
        proc_root=proc_root,
        reserved=reserved,
        ignored_pids=ignored,
        monotonic=monotonic,
        sleep=sleep,
    )
    return build_report(
        requested_seconds=seconds,
        before=before,
        after=after,
        elapsed_seconds=elapsed,
        wall_start_ns=wall_start,
        wall_end_ns=wall_end,
        placement=placement,
        control_cpu=control_cpu,
        native_cpu=native_cpu,
        reserved=reserved,
    )


def validate_seconds(value: float | str) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("seconds must be a positive number") from exc
    if not (seconds > 0.0) or seconds > MAX_SECONDS:
        raise ValueError(f"seconds must be > 0 and <= {MAX_SECONDS:g}")
    return seconds


def _print_thread(thread: Mapping[str, Any]) -> None:
    mask = _format_cpus(thread.get("cpus_allowed", []))
    print(
        "    "
        f"TID {thread.get('tid')} mask={mask} nice={thread.get('nice', '?')} "
        f"priority={thread.get('priority', '?')}"
    )


def _format_cpus(cpus: Iterable[int]) -> str:
    values = sorted({int(cpu) for cpu in cpus})
    if not values:
        return "?"
    ranges: list[str] = []
    start = previous = values[0]
    for cpu in values[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def print_human(report: Mapping[str, Any]) -> None:
    sampling = report["sampling"]
    placement = report["placement"]
    counts = report["counts"]
    print("CPU observer (read-only)")
    print(
        f"Sampled {sampling['elapsed_seconds']:.3f}s "
        f"(requested {sampling['requested_seconds']:.3f}s); "
        f"control={placement['control_cpu']} native={placement['native_cpu']} "
        f"reserved={_format_cpus(placement['smt_reserved_cpus'])}"
    )
    observer = placement.get("observer", {})
    print(
        f"Observer affinity={','.join(map(str, observer.get('affinity_after', []))) or '?'} "
        f"nice={observer.get('nice_after', '?')}"
    )
    for key, title in (("controller", "Controller"), ("native", "Native")):
        rows = report.get(key, [])
        print(f"{title}: {len(rows)}")
        for process in rows:
            print(
                f"  PID {process.get('pid')} {process.get('comm')} "
                f"[{process.get('module')}] threads={process.get('thread_count')} "
                f"affinity={_format_cpus(process.get('actual_affinity', []))} "
                f"overlapTIDs={process.get('overlap_tid_count')} "
                f"CPUms(sum)={_format_number(process.get('sample_cpu_ms'))} "
                f"runqueue_wait_ms(sum)={_format_number(process.get('sample_runqueue_wait_ms'))}"
            )
            for thread in process.get("threads", []):
                _print_thread(thread)
    print(
        f"Eligible background (affinity overlaps reserved): {counts['background_overlap_processes']} processes, "
        f"{counts['background_overlap_threads']} TIDs"
    )
    for row in report.get("background_candidates", [])[:8]:
        print(
            f"  PID {row.get('pid')} {row.get('comm')} [{row.get('module')}] "
            f"overlapTIDs={row.get('overlap_tid_count')} "
            f"CPUms(sum)={_format_number(row.get('cpu_ms'))} "
            f"runqueue_wait_ms(sum)={_format_number(row.get('runqueue_wait_ms'))}"
        )
    contention = report["contention"]
    evidence = "yes" if contention["schedstat_wait_evidence"] else "no"
    print(f"schedstat runqueue-wait evidence (aggregate sample time): {evidence}")
    print("Running detection is point-in-time; controller contention or historical cause is not proven.")


def _format_number(value: Any) -> str:
    return "?" if value is None else f"{float(value):.3f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only /proc CPU affinity and schedstat observer; never attaches robot IPC."
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=DEFAULT_SECONDS,
        help=f"sampling duration in seconds, >0 and <= {MAX_SECONDS:g} (default: {DEFAULT_SECONDS:g})",
    )
    parser.add_argument("--json", action="store_true", help="print a JSON report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        seconds = validate_seconds(args.seconds)
    except ValueError as exc:
        parser.error(str(exc))
    report = collect_report(seconds)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_human(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
