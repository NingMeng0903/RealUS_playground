"""Peirastic: generic outer-loop modes on top of the RM75 8-DOF velocity IK."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO, _REPO / "rm75_control", _REPO / "src"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from peirastic.api import PeirasticArm
from peirastic.core.modes import Mode, ModeRequest

__all__ = ["Mode", "ModeRequest", "PeirasticArm"]
