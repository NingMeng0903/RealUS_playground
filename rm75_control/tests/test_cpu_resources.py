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


def test_background_custom_config_path(tmp_path, monkeypatch):
    config = tmp_path / "custom-controller.yaml"
    config.write_text("timing:\n  control_cpu: 6\n  native_cpu: 10\n")
    monkeypatch.setenv("RM75_OBSERVER_AVOID_CPUS", "0,1")
    # An explicit controller config must win over the observer-only override.
    assert C.configured_control_cpus(config) == {6, 10}


def test_controller_bootstrap_is_config_aware_before_numeric_import(tmp_path):
    """Exercise startup placement and numeric limits without daemon/hardware."""

    root = Path(__file__).resolve().parents[2]
    config = tmp_path / "controller.yaml"
    program = f"""
import json, os, sys
from pathlib import Path

allowed_before = set(os.sched_getaffinity(0))
if not allowed_before:
    raise SystemExit("empty cpuset")
reserved = sorted(allowed_before)
control_cpu = reserved[0]
native_cpu = reserved[1] if len(reserved) > 1 else reserved[0]
config = Path({str(config)!r})
config.write_text(
    "timing:\\n"
    f"  control_cpu: {{control_cpu}}\\n"
    f"  native_cpu: {{native_cpu}}\\n"
)
assert not any(name in sys.modules for name in ("numpy", "scipy", "torch", "pinocchio"))
from rm75_control.control.admittance_common.cpu_resources import (
    physical_siblings,
    prepare_background_cpus,
)
from rm75_control.control.admittance_common.observer_runtime import limit_numeric_threads

nice_before = os.getpriority(os.PRIO_PROCESS, 0)
background = set(prepare_background_cpus(config_path=config))
avoid = physical_siblings(control_cpu) | physical_siblings(native_cpu)
expected = allowed_before - avoid
if not expected:
    expected = allowed_before
assert background == expected, (background, expected)
assert not (background & avoid) or not (allowed_before - avoid)
limit_numeric_threads(threads=1)
assert not any(name in sys.modules for name in ("numpy", "scipy", "torch", "pinocchio"))
import numpy as np
np.ones((64, 3), dtype=np.float64).T @ np.ones((64, 3), dtype=np.float64)
pool_threads = []
try:
    from threadpoolctl import threadpool_info
    pool_threads = [int(pool["num_threads"]) for pool in threadpool_info()]
except Exception:
    pass
thread_count = int(next(line.split()[1] for line in open("/proc/self/status")
                         if line.startswith("Threads:")))
assert thread_count <= 1, thread_count
assert all(count <= 1 for count in pool_threads), pool_threads
assert os.getpriority(os.PRIO_PROCESS, 0) == nice_before
# A later control-loop bind may widen the calling thread back to a reserved
# CPU as long as that CPU remains in the enclosing cpuset.
os.sched_setaffinity(0, {{control_cpu}})
assert set(os.sched_getaffinity(0)) == {{control_cpu}}
print(json.dumps({{"background": sorted(background), "avoid": sorted(avoid),
                  "threads": thread_count, "pools": pool_threads,
                  "nice": nice_before}}))
"""
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env.pop("RM75_OBSERVER_AVOID_CPUS", None)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(root / "rm75_control"), str(root), env.get("PYTHONPATH")) if part
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["threads"] <= 1
    assert all(count <= 1 for count in result["pools"])


def test_run_controller_main_bootstraps_before_daemon_import(tmp_path):
    """Run the real entrypoint with daemon/clock stubs, never with hardware."""

    root = Path(__file__).resolve().parents[2]
    config = tmp_path / "custom-controller.yaml"
    program = f"""
import json, os, sys, types
from pathlib import Path

allowed_before = set(os.sched_getaffinity(0))
if not allowed_before:
    raise SystemExit("empty cpuset")
reserved = sorted(allowed_before)
control_cpu = reserved[0]
native_cpu = reserved[1] if len(reserved) > 1 else reserved[0]
config = Path({str(config)!r})
config.write_text(
    "timing:\\n"
    f"  control_cpu: {{control_cpu}}\\n"
    f"  native_cpu: {{native_cpu}}\\n"
)
original_nice = os.getpriority(os.PRIO_PROCESS, 0)
state = {{}}

def snapshot():
    return {{
        "affinity": sorted(os.sched_getaffinity(0)),
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "numeric": [name for name in ("numpy", "scipy", "torch", "pinocchio")
                    if name in sys.modules],
    }}

def run_service(config_path, **kwargs):
    state["service"] = {{"config": str(config_path), "kwargs": kwargs}}
    return 37

class DaemonStub(types.ModuleType):
    def __getattribute__(self, name):
        if name == "run_service":
            state["daemon_import"] = snapshot()
        return super().__getattribute__(name)

realman = types.ModuleType("peirastic.realman8dof")
realman.__path__ = []
daemon = DaemonStub("peirastic.realman8dof.daemon")
daemon.run_service = run_service
sys.modules["peirastic.realman8dof"] = realman
sys.modules["peirastic.realman8dof.daemon"] = daemon

clock = types.ModuleType("realus_clock")
clock.get_clock = lambda: state.setdefault("clock", snapshot())
sys.modules["realus_clock"] = clock

sys.argv = [
    "run_controller.py", "--config", str(config),
    "--shm-prefix", "bootstrap-test", "--dry-run", "--no-panel",
]
assert not any(name in sys.modules for name in ("numpy", "scipy", "torch", "pinocchio"))
from peirastic.apps import run_controller

result = run_controller.main()
assert result == 37
from rm75_control.control.admittance_common.cpu_resources import physical_siblings
avoid = physical_siblings(control_cpu) | physical_siblings(native_cpu)
expected = allowed_before - avoid
if not expected:
    expected = allowed_before
assert state["daemon_import"]["affinity"] == sorted(expected), state
assert state["daemon_import"]["nice"] == original_nice, state
assert state["daemon_import"]["numeric"] == [], state
assert state["clock"]["nice"] == original_nice, state
assert state["service"] == {{
    "config": str(config),
    "kwargs": {{
        "shm_prefix": "bootstrap-test",
        "log_csv": None,
        "dry_run": True,
        "panel": False,
    }},
}}, state
print(json.dumps(state, sort_keys=True))
"""
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env.pop("RM75_OBSERVER_AVOID_CPUS", None)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(root / "rm75_control"), str(root), env.get("PYTHONPATH")) if part
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    state = json.loads(completed.stdout.strip().splitlines()[-1])
    assert state["daemon_import"]["numeric"] == []
    assert state["service"]["kwargs"]["dry_run"] is True
