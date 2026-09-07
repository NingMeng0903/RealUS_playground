"""Resource defaults for standalone telemetry viewers (never the controller)."""

from __future__ import annotations

import os


def prepare_observer_process() -> None:
    """Run before numerical/GUI imports, without assigning any CPU affinity."""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    try:
        # Absolute priority: repeated setup must not keep increasing niceness.
        current = os.getpriority(os.PRIO_PROCESS, 0)
        os.setpriority(os.PRIO_PROCESS, 0, max(current, 10))
    except (AttributeError, OSError):
        pass
