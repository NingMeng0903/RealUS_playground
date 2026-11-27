"""Actual physical SRS/FK regressions for the wrapper's departure gate."""
import json
from pathlib import Path
import numpy as np
import pytest
from adapter import CapabilityBridge

HERE = Path(__file__).resolve().parent
CASES = json.loads((HERE / 'before_departure_fix/affected_queries.json').read_text())['cases']


@pytest.fixture(scope='module')
def bridge():
    return CapabilityBridge()


@pytest.mark.parametrize('case', CASES, ids=[c['context'] + ':' + c['edge'] for c in CASES])
def test_real_previously_accepted_arm_departure_is_rejected(bridge, case):
    result = bridge.query_path(case['T_tcp_railbase'], case['rail_m'], case['q_seed'])
    ex = result['exact']
    assert case['before_exact']['validated'] is True
    assert ex['lift_valid'] and ex['limits_pass'] and ex['collision_pass'] and ex['fk_pass']
    assert np.degrees(ex['q_seed_to_first_max_joint_delta_rad']) > 12.
    assert ex['continuity_pass'] is False
    assert ex['validated'] is False
    assert 'departure arm step 12.549042 deg exceeds 12.000000 deg' in ex['reason']
    assert np.allclose(ex['q_ref'], case['before_exact']['q_ref'], rtol=0, atol=0)
    assert result['min_raw_score'] == pytest.approx(case['before_proxy'], abs=1e-12)


def test_real_nominal_departure_remains_valid(bridge):
    nominal = json.loads((HERE / 'nominal_context.json').read_text())
    q = np.asarray(nominal['q_seed'])
    T = np.repeat(np.asarray(nominal['T_tcp_railbase'])[None], 3, axis=0)
    ex = bridge.query_path(T, np.full(3, q[0]), q)['exact']
    assert ex['validated'] and ex['continuity_pass']
    assert ex['continuity_failure_reasons'] == []
    assert ex['q_seed_to_first_rail_delta_m'] == pytest.approx(0.)


def test_real_rail_departure_beyond_same_bound_is_rejected(bridge):
    nominal = json.loads((HERE / 'nominal_context.json').read_text())
    q = np.asarray(nominal['q_seed'])
    rail = q[0]
    q[0] += .05
    T = np.repeat(np.asarray(nominal['T_tcp_railbase'])[None], 3, axis=0)
    ex = bridge.query_path(T, np.full(3, rail), q)['exact']
    assert ex['lift_valid'] and ex['limits_pass'] and ex['collision_pass'] and ex['fk_pass']
    assert ex['continuity_pass'] is False and ex['validated'] is False
    assert ex['q_seed_to_first_rail_delta_m'] == pytest.approx(.05)
    assert 'departure rail step 0.050000 m exceeds 0.040000 m' in ex['reason']
