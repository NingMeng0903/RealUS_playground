"""Map ROS2 ``sensor_msgs/Joy`` or plain axis arrays to PEIRASTIC OSC deltas."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from projects.genesis_ue_sync.sim_platform.control.teleop.xbox_gamepad import XboxAxisMap


def axes_with_deadzone(axes: Sequence[float], indices: XboxAxisMap, *, deadzone: float) -> np.ndarray:
    raw = [float(a) for a in axes]
    n = len(raw)

    def pick(idx: int) -> float:
        if idx < 0 or idx >= n:
            return 0.0
        v = float(raw[idx])
        return 0.0 if abs(v) < deadzone else v

    m = indices
    return np.array(
        [pick(m.trans_x), pick(m.trans_y), pick(m.trans_z), pick(m.rot_x), pick(m.rot_y), pick(m.rot_z)],
        dtype=np.float32,
    )


def _normalize_trigger_rest_neg_one(raw: float) -> float:
    return float(max(0.0, min(1.0, (float(raw) + 1.0) * 0.5)))


def augment_linux_xbox_hybrid_inplace(
    vec: np.ndarray,
    raw_axes: Sequence[float],
    indices: XboxAxisMap,
    *,
    lt_axis_idx: int = 2,
    rt_axis_idx: int = 5,
    hat_axis_pair: tuple[int, int] | None = None,
    trigger_rot_y_scale: float = 3.25,
    hat_rot_scale_y: float = 1.08,
    hat_rot_scale_z: float = 1.08,
) -> None:
    """Optional LT/RT and analog hat axes; aligns with hybrid ``XboxGamepad.read_action_vector``."""

    raw = [float(a) for a in raw_axes]
    n = len(raw)
    m = indices

    if trigger_rot_y_scale > 1e-6 and m.rot_y < 0:
        lt_raw = raw[lt_axis_idx] if 0 <= lt_axis_idx < n else -1.0
        rt_raw = raw[rt_axis_idx] if 0 <= rt_axis_idx < n else -1.0
        lt_n = _normalize_trigger_rest_neg_one(lt_raw)
        rt_n = _normalize_trigger_rest_neg_one(rt_raw)
        vec[4] = float(vec[4]) + float(trigger_rot_y_scale) * float(lt_n - rt_n)

    if hat_axis_pair is not None and (hat_rot_scale_y > 1e-6 or hat_rot_scale_z > 1e-6):
        ia, ib = int(hat_axis_pair[0]), int(hat_axis_pair[1])
        if 0 <= ia < n and 0 <= ib < n:
            hx = raw[ia]
            hy = raw[ib]
            if m.rot_z < 0:
                vec[5] = float(vec[5]) + float(hat_rot_scale_z) * float(hx)
            if m.rot_y < 0:
                vec[4] = float(vec[4]) + float(hat_rot_scale_y) * float(hy)


__all__ = ["axes_with_deadzone", "augment_linux_xbox_hybrid_inplace"]
