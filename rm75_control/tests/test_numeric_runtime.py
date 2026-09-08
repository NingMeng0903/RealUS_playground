"""Process-isolated checks for standalone numerical observer bootstrap."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_MODULE = "rm75_control.control.admittance_common.observer_runtime"
CAMERA_PYTHON = Path("/media/camp/EXT_DRIVE/envs/camera_calib/bin/python")
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "TBB_NUM_THREADS",
    "OPENCV_FOR_THREADS_NUM",
    "QD_NUM_THREADS",
)


def _child(code: str, *, python: Path | None = None, env_updates: dict[str, str | None] | None = None) -> dict:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    # The workstation has an editable rm75-control install from another
    # checkout; force children to exercise this workspace's source tree.
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT / "rm75_control"), str(ROOT), env.get("PYTHONPATH")) if part
    )
    for name, value in (env_updates or {}).items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    result = subprocess.run(
        [str(python or Path(sys.executable)), "-c", code],
        cwd=str(ROOT),
        env=env,
        check=True,
        text=True,
        capture_output=True,
        timeout=20,
    )
    # NumPy may print optional runtime diagnostics to stderr; keep the child
    # protocol on stdout so this test remains independent of that output.
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_limits_apply_before_numpy_import_and_bound_blas_workers() -> None:
    result = _child(
        f"""
import json, os, sys
from {RUNTIME_MODULE} import limit_numeric_threads
assert 'numpy' not in sys.modules
limit_numeric_threads(threads=1)
import numpy as np
# Force the BLAS path to initialize, while keeping this child lightweight.
np.ones((64, 3), dtype=np.float64).T @ np.ones((64, 3), dtype=np.float64)
threads = 1
try:
    threads = int(next(line.split()[1] for line in open('/proc/self/status')
                       if line.startswith('Threads:')))
except (OSError, StopIteration, ValueError):
    pass
pools = []
try:
    from threadpoolctl import threadpool_info
    pools = threadpool_info()
except Exception:
    pass
print(json.dumps({{'env': {{name: os.environ.get(name) for name in {THREAD_ENV!r}}},
                    'wait': {{name: os.environ.get(name) for name in
                              ('OMP_WAIT_POLICY', 'OMP_DYNAMIC', 'MKL_DYNAMIC', 'KMP_BLOCKTIME')}},
                    'threads': threads, 'pools': pools}}))
"""
    )
    assert all(value == "1" for value in result["env"].values())
    assert result["wait"] == {
        "OMP_WAIT_POLICY": "PASSIVE",
        "OMP_DYNAMIC": "FALSE",
        "MKL_DYNAMIC": "FALSE",
        "KMP_BLOCKTIME": "0",
    }
    # A NumPy BLAS call must not fan this otherwise single-threaded child out.
    assert result["threads"] <= 1
    assert all(int(pool["num_threads"]) <= 1 for pool in result["pools"])


def test_limits_apply_to_already_loaded_numpy_when_threadpoolctl_is_present() -> None:
    result = _child(
        f"""
import json, os
import numpy as np
before = []
try:
    from threadpoolctl import threadpool_info
    before = threadpool_info()
except ImportError:
    pass
from {RUNTIME_MODULE} import limit_numeric_threads
limit_numeric_threads(threads=1)
np.ones((64, 3), dtype=np.float64).T @ np.ones((64, 3), dtype=np.float64)
pthreads = 1
try:
    pthreads = int(next(line.split()[1] for line in open('/proc/self/status')
                        if line.startswith('Threads:')))
except (OSError, StopIteration, ValueError):
    pass
pools = []
try:
    from threadpoolctl import threadpool_info
    pools = threadpool_info()
except Exception:
    pass
print(json.dumps({{'before': before, 'pools': pools, 'threads': pthreads}}))
""",
        python=CAMERA_PYTHON if CAMERA_PYTHON.is_file() else None,
        env_updates={"OPENBLAS_NUM_THREADS": "4"},
    )
    # The minimal rm75 environment intentionally has no threadpoolctl.  In
    # environments that do provide it, verify the dynamic post-import clamp.
    if not result["pools"]:
        pytest.skip("threadpoolctl is optional in the controller environment")
    assert any(int(pool["num_threads"]) > 1 for pool in result["before"])
    assert all(int(pool["num_threads"]) <= 1 for pool in result["pools"])
    # Existing workers can remain asleep; the active BLAS limit is what matters.


@pytest.mark.skipif(not CAMERA_PYTHON.is_file(), reason="camera_calib environment is unavailable")
@pytest.mark.parametrize("load_cv2_before_helper", (False, True))
def test_opencv_threads_are_bounded_before_and_after_cv2_import(load_cv2_before_helper: bool) -> None:
    if load_cv2_before_helper:
        load = """
import cv2
before = cv2.getNumThreads()
from rm75_control.control.admittance_common.observer_runtime import limit_numeric_threads
assert 'cv2' in __import__('sys').modules
limit_numeric_threads(threads=1)
after = cv2.getNumThreads()
"""
    else:
        load = """
import sys
assert 'cv2' not in sys.modules
from rm75_control.control.admittance_common.observer_runtime import limit_numeric_threads
limit_numeric_threads(threads=1)
assert 'cv2' not in sys.modules
import cv2
before = None
after = cv2.getNumThreads()
"""
    result = _child(
        load
        + "\nimport json\nprint(json.dumps({'before': before, 'after': after}))\n",
        python=CAMERA_PYTHON,
        # Ensure the late-import case starts from OpenCV's observed default,
        # rather than inheriting a caller's pre-existing override.
        env_updates={"OPENCV_FOR_THREADS_NUM": None},
    )
    assert result["after"] == 1
    if load_cv2_before_helper:
        assert result["before"] >= 1


def _guard_and_first_numeric_import(path: Path) -> tuple[int, int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    first_numeric = None
    guard_call = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".", 1)[0] in {"numpy", "scipy", "torch"} for alias in node.names):
                first_numeric = min(first_numeric or node.lineno, node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".", 1)[0] in {"numpy", "scipy", "torch"}:
                first_numeric = min(first_numeric or node.lineno, node.lineno)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "prepare_observer_process":
                guard_call = min(guard_call or node.lineno, node.lineno)
    assert first_numeric is not None
    assert guard_call is not None
    return guard_call, first_numeric


@pytest.mark.parametrize(
    "relative",
    (
        "perception/apps/run_orbbec_cloud_publisher.py",
        "rm75_control/rm75_control/control/joint_admittance_8dof/viewer/demo.py",
        "peirastic/DEMO/phathom_scanning/s_scan.py",
    ),
)
def test_standalone_scripts_prepare_before_numeric_imports_without_hardware(relative: str) -> None:
    path = ROOT / relative
    guard_line, numeric_line = _guard_and_first_numeric_import(path)
    assert guard_line < numeric_line
    source = path.read_text(encoding="utf-8")
    assert 'if __name__ == "__main__":' in source
    # Constructors only occur inside main(); importing these scripts must not
    # claim a camera, robot, or Genesis viewer.
    tree = ast.parse(source, filename=str(path))
    top_level_calls = [
        node
        for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ]
    assert not top_level_calls
