"""Rail prismatic DOF top-level mode + locked-substyle enums.

Two-layer hierarchy replaces the flat LOCKED / REPOSITION / RELIEF triplet:

    RailMode
      COUPLED             rail is a regular QP joint (reg / v_max / a_max / resync)
      LOCKED              rail is not decided by QP; how it moves is a LockedStyle
        LockedStyle.HOLD      hold q_ref (scan default)
        LockedStyle.RAIL_ONLY external plan drives rail, arm frozen
        LockedStyle.TCP_FIXED external plan drives rail, arm QP compensates TCP
"""

from __future__ import annotations

from enum import Enum


class RailMode(str, Enum):
    """Top-level rail control mode."""

    COUPLED = "coupled"  # rail is a normal QP joint (respects reg / v_max / a_max)
    LOCKED = "locked"    # rail motion is externally imposed (see LockedStyle)


class LockedStyle(str, Enum):
    """How the rail is externally driven while in RailMode.LOCKED."""

    HOLD = "hold"              # q_cmd[0] pinned to q_ref (hold position)
    RAIL_ONLY = "rail_only"    # external qdot_ff[0] drives rail; arm 1..7 frozen
    TCP_FIXED = "tcp_fixed"    # external qdot_ff[0] drives rail; arm QP holds TCP
