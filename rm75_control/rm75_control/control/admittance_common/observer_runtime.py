"""Resource defaults for standalone telemetry viewers (never the controller).

The viewer and camera publisher import large numerical stacks (NumPy, SciPy,
PyTorch, and sometimes Genesis).  Those stacks otherwise inherit the host's
OpenMP/BLAS defaults, which can turn a small ``N x 3`` operation into hundreds
of runnable threads.  Keep this module deliberately dependency-free so it can
be imported before any of those stacks.
"""

from __future__ import annotations

import os
import sys


_NUMERIC_THREAD_ENV = (
    # BLAS/OpenMP implementations used by NumPy, SciPy and PyTorch.
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    # NumExpr has both a configured limit and a runtime worker count.
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    # oneTBB is used by some optional numerical/image dependencies.
    "TBB_NUM_THREADS",
    # OpenCV reads this before importing its worker pool.
    "OPENCV_FOR_THREADS_NUM",
    # Genesis reads this for both Quadrants CPU workers and JIT compilation.
    "QD_NUM_THREADS",
)

# OpenMP runtimes otherwise keep workers spinning briefly after a small call.
# These settings do not select CPUs or otherwise alter process affinity.
_PASSIVE_WAIT_ENV = {
    "OMP_WAIT_POLICY": "PASSIVE",
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
    "KMP_BLOCKTIME": "0",
}

# ``threadpool_limits`` changes libraries already loaded in this process.  It
# must remain reachable for the life of the process: keeping the limiter alive
# also makes repeated setup calls deterministic and avoids accidental context
# manager restoration by callers.
_THREADPOOL_LIMITER = None


def limit_numeric_threads(threads: int = 1) -> None:
    """Limit numerical worker pools without changing CPU affinity.

    Environment variables are set before NumPy/SciPy/PyTorch are imported.
    When :mod:`threadpoolctl` is available, its process-wide limiter also
    clamps pools that were loaded before this function was called.  The
    optional dependency is deliberately best-effort: camera/viewer startup
    must still work in the minimal controller environment.
    """

    try:
        value = int(threads)
    except (TypeError, ValueError) as exc:
        raise ValueError("threads must be a positive integer") from exc
    if value < 1 or value != threads:
        raise ValueError("threads must be a positive integer")

    value_text = str(value)
    for name in _NUMERIC_THREAD_ENV:
        os.environ[name] = value_text
    for name, setting in _PASSIVE_WAIT_ENV.items():
        os.environ[name] = setting

    # Do not import cv2 here: standalone entry points call this before their
    # optional camera stack loads.  If a caller already imported it, the
    # runtime setter is the only reliable way to change its worker count.
    cv2 = sys.modules.get("cv2")
    if cv2 is not None:
        try:
            cv2.setNumThreads(value)
        except Exception:
            # Older/minimal OpenCV builds may not expose this setter.
            pass

    global _THREADPOOL_LIMITER
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        # threadpoolctl is optional and is absent in the lean controller env.
        return
    try:
        # Construction applies the limit immediately.  Keep this object alive
        # instead of using ``with`` so limits are not restored on return.
        _THREADPOOL_LIMITER = threadpool_limits(limits=value)
    except Exception:
        # A partially installed/incompatible threadpoolctl must not prevent a
        # viewer or publisher from starting; environment limits still apply to
        # libraries imported after this call.
        _THREADPOOL_LIMITER = None


def prepare_observer_process() -> None:
    """Prepare a low-priority standalone observer process.

    Call this only from a standalone process entry point, before numerical or
    GUI imports.  CPU placement is delegated to ``cpu_resources`` so the
    controller and observer use the same configured physical-core rules.
    """

    limit_numeric_threads(threads=1)
    try:
        from .cpu_resources import prepare_background_cpus

        prepare_background_cpus()
    except (ImportError, OSError, ValueError):
        # Resource placement is best effort on non-Linux/minimal installs;
        # numerical limits and observer priority remain useful on their own.
        pass
    try:
        # Absolute priority: repeated setup must not keep increasing niceness.
        current = os.getpriority(os.PRIO_PROCESS, 0)
        os.setpriority(os.PRIO_PROCESS, 0, max(current, 10))
    except (AttributeError, OSError):
        pass
