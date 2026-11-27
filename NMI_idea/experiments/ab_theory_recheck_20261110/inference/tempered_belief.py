"""Stable integration name for the continuous tempered posterior.

The recheck harness imports ``TemperedBelief``; keeping this tiny adapter
avoids coupling it to the implementation module name while preserving the
same public ``particles``, ``weights``, ``diagnostics`` and ``update_block``
API.
"""
from __future__ import annotations

try:  # package import used by the harness
    from .continuous_smc import ContinuousBelief
except ImportError:  # direct module import used by small standalone scripts
    from continuous_smc import ContinuousBelief


class TemperedBelief(ContinuousBelief):
    """Adaptive likelihood-tempered resample--move continuous belief."""

    pass


__all__ = ["TemperedBelief"]
