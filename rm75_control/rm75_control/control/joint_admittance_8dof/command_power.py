"""Frozen six-axis command-model power constraint (not measured robot work)."""
import numpy as np


def validate_command_power(wrench_base, minimum_w):
    if wrench_base is None and minimum_w is None:
        return None, None
    if wrench_base is None or minimum_w is None or isinstance(minimum_w, (bool, np.bool_)):
        raise ValueError("command power requires a wrench and finite minimum")
    wrench = np.asarray(wrench_base, dtype=float)
    minimum = float(minimum_w)
    if wrench.shape != (6,) or not np.isfinite(wrench).all() or not np.isfinite(minimum):
        raise ValueError("invalid command power constraint")
    return wrench.copy(), minimum


def final_command_qdot(qdot, *, proposed_rail_m, published_rail_m, dt_s):
    """Apply only the final rail payload mutation to the admitted joint rate.

    Rail observer rebasing is a position estimate, not command displacement.
    Both positions here belong to THIS proposal, after that same rebase.
    """
    result = np.asarray(qdot, dtype=float).copy()
    values = np.asarray([proposed_rail_m, published_rail_m, dt_s], dtype=float)
    if result.shape != (8,) or not np.isfinite(result).all() or not np.isfinite(values).all() or dt_s <= 0:
        raise ValueError("invalid final command model inputs")
    result[0] += (published_rail_m - proposed_rail_m) / dt_s
    return result
