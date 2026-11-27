import numpy as np
import pytest

from perception.benchmark.adaptive_observation import AdaptiveObservationFilter


def points(x=0.0):
    result = np.ones((67, 4), np.float32)
    result[:, :3] = x
    return result


def test_noise_reduction_preserves_confidence_and_uses_source_rate():
    dense, sparse = AdaptiveObservationFilter(), AdaptiveObservationFilter()
    dense.apply(points(), 0)
    sparse.apply(points(), 0)
    measured = points(.01)
    a = dense.apply(measured, 33_333_333)
    b = sparse.apply(measured, 625_000_000)
    assert 0 < a[0, 0] < b[0, 0] < measured[0, 0]
    np.testing.assert_array_equal(a[:, 3], measured[:, 3])
    # At realtime 1.6 Hz the source-clock filter follows most of a step.
    assert b[0, 0] > .008


def test_visibility_reset_nan_isolation_and_snapshot_retry():
    filter_ = AdaptiveObservationFilter()
    filter_.apply(points(), 0)
    hidden = points(.2)
    hidden[0, :3] = np.nan
    hidden[0, 3] = 0
    filtered = filter_.apply(hidden, 33_333_333)
    assert np.isfinite(filtered[1:]).all()
    saved = filter_.snapshot()
    measured = points(.5)
    filtered = filter_.apply(measured, 66_666_667)
    np.testing.assert_array_equal(filtered[0], measured[0])
    filter_.restore(saved)
    repeated = filter_.apply(measured, 66_666_667)
    np.testing.assert_array_equal(repeated, filtered)
    assert saved.timestamp_ns == 33_333_333
    with pytest.raises(ValueError, match="increase strictly"):
        filter_.apply(measured, 66_666_667)


def test_untimed_warmup_does_not_pollute_first_actual_sample():
    filter_ = AdaptiveObservationFilter()
    filter_.apply(points(10), None)
    assert filter_.state is None
    np.testing.assert_array_equal(filter_.apply(points(.1), 100), points(.1))
