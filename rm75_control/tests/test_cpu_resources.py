"""Core reservation is local, topology-aware and compatible with cpusets."""

import json
import os
from pathlib import Path
import subprocess
import sys

from rm75_control.control.admittance_common import cpu_resources as C


def test_cpu_list_and_smt_topology(tmp_path):
    assert C.parse_cpu_list("0-1,4,6-7") == {0, 1, 4, 6, 7}
    path = tmp_path / "cpu2/topology"
    path.mkdir(parents=True)
    (path / "thread_siblings_list").write_text("2-3\n")
    assert C.physical_siblings(2, sysfs=tmp_path) == {2, 3}
    assert C.physical_siblings(8, sysfs=tmp_path) == {8}


def test_background_excludes_siblings_without_widening_existing_masks(monkeypatch):
    monkeypatch.setattr(C, "configured_control_cpus", lambda: {2, 4})
    monkeypatch.setattr(C, "physical_siblings", lambda c: {c, c + 1})
    monkeypatch.setattr(C.Path, "iterdir", lambda self: [Path("/proc/self/task/10"), Path("/proc/self/task/11")])
    masks = {0: set(range(8)), 10: set(range(8)), 11: {1, 3, 6}}
    monkeypatch.setattr(C.os, "sched_getaffinity", lambda tid: masks[tid])
    changes = []
    monkeypatch.setattr(C.os, "sched_setaffinity", lambda tid, mask: changes.append((tid, mask)))
    C.prepare_background_cpus()
    assert changes == [(10, {0, 1, 6, 7}), (11, {1, 6})]


def test_background_does_not_empty_a_constrained_cpuset(monkeypatch):
    monkeypatch.setattr(C, "configured_control_cpus", lambda: {2})
    monkeypatch.setattr(C, "physical_siblings", lambda c: {2, 3})
    monkeypatch.setattr(C.os, "sched_getaffinity", lambda tid: {2, 3})
    monkeypatch.setattr(C.os, "sched_setaffinity", lambda *a: (_ for _ in ()).throw(AssertionError("empty mask")))
    assert C.prepare_background_cpus() == [2, 3]


def test_scheduling_note_flags_sibling_cores(monkeypatch):
    monkeypatch.setattr(C, "physical_siblings", lambda c: {2, 3})
    assert "shared physical core" in C.scheduling_note(2, 3)
    assert "shared" not in C.scheduling_note(2, 4)


def test_module_entry_packages_do_not_preload_numeric_libraries():
    root = Path(__file__).resolve().parents[2]
    program = """
import sys, json
import peirastic
import rm75_control.control.joint_admittance_8dof.viewer
print(json.dumps([name for name in ('numpy', 'scipy', 'torch', 'pinocchio') if name in sys.modules]))
"""
    completed = subprocess.run([sys.executable, "-c", program], cwd=root,
                               capture_output=True, text=True, check=True, timeout=15)
    assert json.loads(completed.stdout) == []


def test_background_default_uses_machine_config(monkeypatch):
    monkeypatch.delenv("RM75_OBSERVER_AVOID_CPUS", raising=False)
    assert C.configured_control_cpus() == {2, 4}
    monkeypatch.setenv("RM75_OBSERVER_AVOID_CPUS", "6,10")
    assert C.configured_control_cpus() == {6, 10}
