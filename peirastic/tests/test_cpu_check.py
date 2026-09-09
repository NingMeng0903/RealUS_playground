"""Fake-proc tests for the dependency-free CPU observer."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from peirastic.apps import cpu_check as C


def _stat(pid: int, comm: str, *, ppid: int = 1, priority: int = 20, nice: int = 0) -> str:
    # /proc stat fields 3..39 after ``pid (comm)``.  Only the fields consumed
    # by the observer need meaningful values.
    fields = ["S"] + ["0"] * 36
    fields[1] = str(ppid)  # field 4
    fields[15] = str(priority)  # field 18
    fields[16] = str(nice)  # field 19
    fields[36] = "2"  # field 39, last CPU
    return f"{pid} ({comm}) " + " ".join(fields) + "\n"


def _status(name: str, cpus: str, *, ppid: int = 1) -> str:
    return (
        f"Name:\t{name}\n"
        "State:\tS (sleeping)\n"
        f"PPid:\t{ppid}\n"
        f"Cpus_allowed_list:\t{cpus}\n"
    )


def _proc_task(
    root: Path,
    pid: int,
    *,
    comm: str,
    cpus: str,
    cpu_ns: int,
    wait_ns: int,
    cmdline: list[str],
    ppid: int = 1,
    tid: int | None = None,
    nice: int = 0,
    priority: int = 20,
) -> Path:
    tid = pid if tid is None else tid
    process = root / str(pid)
    task = process / "task" / str(tid)
    task.mkdir(parents=True, exist_ok=True)
    (process / "status").write_text(_status(comm, cpus, ppid=ppid))
    (process / "stat").write_text(_stat(pid, comm, ppid=ppid, priority=priority, nice=nice))
    (process / "comm").write_text(comm + "\n")
    (process / "cmdline").write_bytes(b"\0".join(item.encode() for item in cmdline) + b"\0")
    (task / "status").write_text(_status(comm, cpus, ppid=ppid))
    (task / "stat").write_text(_stat(tid, comm, ppid=ppid, priority=priority, nice=nice))
    (task / "schedstat").write_text(f"{cpu_ns} {wait_ns} 1\n")
    return task / "schedstat"


def test_parsers_and_safe_command_classification() -> None:
    assert C.parse_cpu_list("0-2,4,7-8") == {0, 1, 2, 4, 7, 8}
    assert C.parse_cpu_mask("00000000,00000029") == {0, 3, 5}
    assert C.classify_command(["python3", "-m", "peirastic.apps.run_controller", "--token=secret"]) == "controller"
    assert C.classify_command(["/tmp/wbc_rt", "--secret=secret"]) == "native"
    assert C.classify_command(["python3", "-c", "import wbc_rt"]) is None
    assert C.classify_command(["python3", "worker.py"]) is None
    assert C._module_basename(["python3", "-m", "peirastic.apps.run_controller", "--secret=do-not-print"]) == "run_controller"
    assert C._module_basename(["python3", "-c", "print('secret')"]) == "python3"


def test_reserved_cpu_topology_and_default_fallback(tmp_path: Path) -> None:
    sys_root = tmp_path / "sys"
    for cpu, siblings in ((2, "2-3\n"), (4, "4-5\n")):
        path = sys_root / f"cpu{cpu}" / "topology"
        path.mkdir(parents=True)
        (path / "thread_siblings_list").write_text(siblings)
    assert C.reserved_cpus(2, 4, sys_root=sys_root) == {2, 3, 4, 5}
    assert C.reserved_cpus(2, 4, sys_root=tmp_path / "missing") == {2, 3, 4, 5}


def test_fake_proc_report_detects_tasks_and_hides_arguments(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    controller_stat = _proc_task(
        proc,
        100,
        comm="python3",
        cpus="2-3",
        cpu_ns=1_000_000_000,
        wait_ns=200_000_000,
        cmdline=["python3", "-m", "peirastic.apps.run_controller", "--secret=controller-pass"],
        nice=5,
        priority=15,
    )
    native_stat = _proc_task(
        proc,
        200,
        comm="wbc_rt",
        cpus="4-5",
        cpu_ns=2_000_000,
        wait_ns=1_000_000,
        cmdline=["/opt/wbc_rt", "--secret=native-pass"],
    )
    background_stat = _proc_task(
        proc,
        300,
        comm="worker",
        cpus="2",
        cpu_ns=0,
        wait_ns=0,
        cmdline=["python3", "worker.py", "--password=background-pass"],
    )
    _proc_task(
        proc,
        400,
        comm="safe-worker",
        cpus="6-7",
        cpu_ns=1_000_000,
        wait_ns=1_000_000,
        cmdline=["python3", "safe.py"],
    )
    # PID 998 is an ancestor of the fake observer PID 999 and must not be
    # reported even though its mask overlaps the reserved CPUs.
    _proc_task(proc, 999, comm="fake-observer", cpus="6-7", cpu_ns=0, wait_ns=0, cmdline=["pytest"])
    _proc_task(proc, 998, comm="parent", cpus="2", cpu_ns=5_000_000, wait_ns=5_000_000, cmdline=["shell", "--secret=ancestor"])
    (proc / "999" / "status").write_text(_status("fake-observer", "6-7", ppid=998))
    (proc / "999" / "task" / "999" / "status").write_text(_status("fake-observer", "6-7", ppid=998))

    def fake_bootstrap(reserved):
        return {
            "affinity_before": [0, 2, 3, 4, 5, 6],
            "affinity_after": [0, 6],
            "nice_before": 0,
            "nice_after": 10,
            "set_affinity": True,
            "set_nice": True,
            "errors": [],
        }

    monkeypatch.setattr(C, "bootstrap_observer", fake_bootstrap)
    times = iter((10.0, 13.0))

    def fake_sleep(seconds: float) -> None:
        assert seconds == pytest.approx(3.0)
        controller_stat.write_text("4000000000 600000000 2\n")
        native_stat.write_text("5000000000 300000000 2\n")
        background_stat.write_text("3000000000 700000000 2\n")

    report = C.collect_report(
        3.0,
        proc_root=proc,
        sys_root=tmp_path / "missing-sys",
        config_path=tmp_path / "missing-controller.yaml",
        self_pid=999,
        monotonic=lambda: next(times),
        sleep=fake_sleep,
    )
    assert report["placement"]["control_cpu"] == 2
    assert report["placement"]["native_cpu"] == 4
    assert report["placement"]["smt_reserved_cpus"] == [2, 3, 4, 5]
    assert [row["pid"] for row in report["controller"]] == [100]
    assert [row["pid"] for row in report["native"]] == [200]
    assert report["controller_running"] is True
    assert report["native_running"] is True
    assert report["controller"][0]["threads"][0]["nice"] == 5
    assert report["controller"][0]["threads"][0]["priority"] == 15
    assert report["controller"][0]["actual_affinity"] == [2, 3]
    assert report["controller"][0]["threads"][0]["cpu_ms"] == pytest.approx(3_000.0)
    assert report["controller"][0]["threads"][0]["runqueue_wait_ms"] == pytest.approx(400.0)
    assert report["native"][0]["sample_runqueue_wait_ms"] == pytest.approx(299.0)
    background = {row["pid"]: row for row in report["background_candidates"]}
    assert 300 in background
    assert 400 not in background
    assert 998 not in background
    assert background[300]["overlap_tid_count"] == 1
    assert background[300]["cpu_ms"] == pytest.approx(3_000.0)
    assert background[300]["wait_ms"] == pytest.approx(700.0)
    assert background[300]["runqueue_wait_ms"] == pytest.approx(700.0)
    assert report["contention"]["eligibility"] is True
    assert report["contention"]["schedstat_wait_evidence"] is True
    assert report["contention"]["actual_controller_contention_proven"] is False
    serialized = json.dumps(report)
    assert "secret" not in serialized
    assert report["sampling"]["elapsed_seconds"] == pytest.approx(3.0)


def test_missing_tasks_are_tolerated(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    _proc_task(proc, 123, comm="worker", cpus="2", cpu_ns=0, wait_ns=0, cmdline=["worker"])
    (proc / "123" / "task" / "123" / "schedstat").unlink()
    rows = C.collect_processes(proc_root=proc, reserved={2}, ignored_pids={999})
    assert rows[0]["pid"] == 123
    assert rows[0]["threads"][0]["cpu_ns"] is None


def test_duration_validation_and_help_without_hardware() -> None:
    with pytest.raises(ValueError):
        C.validate_seconds(0)
    with pytest.raises(ValueError):
        C.validate_seconds(60.001)
    completed = subprocess.run(
        [sys.executable, "-m", "peirastic.apps.cpu_check", "--help"],
        cwd=Path(__file__).resolve().parents[2],
        env={"PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1", "PYTHONPATH": ""},
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert "--seconds" in completed.stdout
    assert "--json" in completed.stdout
    assert "robot IPC" in completed.stdout
