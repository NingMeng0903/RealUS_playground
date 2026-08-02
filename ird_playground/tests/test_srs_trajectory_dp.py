from __future__ import annotations

import numpy as np

from ird_playground.optimization import srs_trajectory_dp as module


def test_first_order_dp_selects_continuous_candidate_corridor():
    candidates = np.zeros((4, 3, 8), dtype=np.float64)
    candidates[:, 0, 1] = [0.0, 0.1, 0.2, 0.3]
    candidates[:, 1, 1] = [1.0, -1.0, 1.0, -1.0]
    candidates[:, 2, 1] = [2.0, 2.1, 2.2, 2.3]
    valid = np.ones((4, 3), dtype=bool)
    index = module._first_order_dp(candidates, valid, np.zeros(8))
    assert np.array_equal(index, np.zeros(4, dtype=np.int32))


def test_first_order_dp_fails_closed_on_disconnected_waypoint():
    candidates = np.zeros((3, 2, 8), dtype=np.float64)
    valid = np.ones((3, 2), dtype=bool)
    valid[1] = False
    try:
        module._first_order_dp(candidates, valid, np.zeros(8))
    except RuntimeError as exc:
        assert "disconnects" in str(exc)
    else:
        raise AssertionError("disconnected graph was accepted")


def test_wrap_pi_is_stable_at_psi_boundary():
    wrapped = module._wrap_pi(np.deg2rad(np.array([179.0, 181.0, -181.0])))
    assert np.allclose(np.rad2deg(wrapped), [179.0, -179.0, 179.0])
