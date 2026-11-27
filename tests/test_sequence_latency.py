import pytest
from perception.apps.evaluate_mocap_benchmark import sequence_latency


def test_fresh_wall_time_includes_rgb_instead_of_old_frontend_backend_sum():
    row = dict(frontend_elapsed_s=.2, backend_elapsed_s=.1, latency_s=.5,
               hybrik_rgb={'wall_elapsed_s':.15})
    assert sequence_latency(row) == (.5, 'latency_s')
    del row['latency_s']
    value, source = sequence_latency(row)
    assert value == pytest.approx(.45)
    assert source == 'component_sum'


def test_incomplete_rgb_timing_is_not_reported_as_complete_pipeline():
    row = dict(frontend_elapsed_s=.2, backend_elapsed_s=.1, hybrik_rgb={'camera_ids':['cam1','cam2']})
    assert sequence_latency(row) == (None, 'rgb_timing_missing')


def test_overlapped_rgb_is_not_counted_twice_in_component_fallback():
    row = dict(frontend_elapsed_s=.3, backend_elapsed_s=.2,
               frontend_includes_rgb=True, hybrik_rgb={'wall_elapsed_s': .15})
    assert sequence_latency(row) == (.5, 'overlapped_component_sum')
