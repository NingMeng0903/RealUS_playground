"""Hard linear intervals on the final commanded TCP twist in base axes."""
import numpy as np

MAX_COMMAND_TWIST_ROWS = 16

def validate_command_twist(rows, lower, upper):
    if rows is None and lower is None and upper is None:
        return np.zeros((0, 6)), np.zeros(0), np.zeros(0)
    if rows is None or lower is None or upper is None:
        raise ValueError('command twist rows and bounds must be supplied together')
    a = np.asarray(rows, dtype=float)
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (a.ndim != 2 or a.shape[1] != 6 or len(a) > MAX_COMMAND_TWIST_ROWS
            or lo.shape != (len(a),) or hi.shape != (len(a),)
            or not np.isfinite(a).all() or not np.isfinite(lo).all()
            or not np.isfinite(hi).all() or np.any(lo > hi)):
        raise ValueError('invalid command twist intervals')
    return a.copy(), lo.copy(), hi.copy()
