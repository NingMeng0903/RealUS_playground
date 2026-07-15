"""link_7 flange pose for φ regressor (tool-independent)."""

from __future__ import annotations

import numpy as np

_KIN = None


def link7_pose_from_q_deg(q_deg: np.ndarray) -> np.ndarray:
    global _KIN
    if _KIN is None:
        from rm75_control.control.joint_admittance.model import RobotKinematics

        _KIN = RobotKinematics()
    from rm75_control.control.joint_admittance.model import deg2rad

    return _KIN.frame_pose(deg2rad(np.asarray(q_deg, dtype=float)), "link_7")
