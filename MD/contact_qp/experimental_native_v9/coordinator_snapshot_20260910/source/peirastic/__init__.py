"""Peirastic: generic outer-loop modes on top of the RM75 8-DOF velocity IK."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO, _REPO / "rm75_control", _REPO / "src"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

__all__ = ["Mode", "ModeRequest", "PeirasticArm"]


def __getattr__(name: str):
    # `python -m peirastic.apps...` imports this package before the entrypoint.
    # Keep NumPy/BLAS unloaded until that entrypoint has set its thread budget.
    modules = {"PeirasticArm": "peirastic.api", "Mode": "peirastic.core.modes",
               "ModeRequest": "peirastic.core.modes"}
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name]), name)
    globals()[name] = value
    return value
