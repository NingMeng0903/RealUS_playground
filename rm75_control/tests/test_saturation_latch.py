"""Hold-aware rail prediction. Scale fade lives in test_pad_slew_and_sat_scale."""

from __future__ import annotations

import pytest

from rm75_control.control.joint_admittance_8dof.saturation_latch import (
    predict_rail_position_m,
)


def test_predict_rail_position_coasts_without_filter() -> None:
    pred = predict_rail_position_m(0.401, 0.024, 0.0187)
    assert pred == pytest.approx(0.401 + 0.024 * 0.0187)
    clipped = predict_rail_position_m(0.401, 0.50, 1.00, lo_m=0.005, hi_m=0.78)
    assert clipped == pytest.approx(0.78)
