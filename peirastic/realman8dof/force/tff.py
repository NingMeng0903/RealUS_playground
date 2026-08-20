"""Task-function force split: position axes vs force axes, then compose."""

from __future__ import annotations

import numpy as np

# 1 = position / velocity track, 0 = force. Tool Z is the force axis.
SELECTION_TOOL_Z_FORCE = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 1.0])


def compose_tff(
    v_pos: np.ndarray,
    v_force: np.ndarray,
    selection: np.ndarray | None = None,
) -> np.ndarray:
    """v* = S v_pos + (I-S) v_force, all in the same frame."""

    s = np.asarray(
        SELECTION_TOOL_Z_FORCE if selection is None else selection, dtype=float
    ).reshape(6)
    s = np.clip(s, 0.0, 1.0)
    vp = np.asarray(v_pos, dtype=float).reshape(6)
    vf = np.asarray(v_force, dtype=float).reshape(6)
    return s * vp + (1.0 - s) * vf
