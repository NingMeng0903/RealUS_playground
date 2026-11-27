"""Non-wiring witnesses for the Step 3R confirmation rules."""
from types import SimpleNamespace

import numpy as np

from reader import dev_rank_comparison as rank
from reader.eval_reader import _tape_truth


def test_paired_gains_and_losses_remain_visible_and_missing_geometry_blocks_continuation():
    equal = rank._paired_flags([True, False, True, False], [True, True, False, False])
    improvement = rank._paired_flags([False, False], [True, False])
    regression = rank._paired_flags([True, False], [False, False])
    assert equal == {'fixed_denominator': 4, 'v6': 2, 'evidence_ranked': 2, 'both': 1, 'neither': 1,
                     'gains': 1, 'losses': 1, 'net': 0}
    station = {'paired_joint_counts': dict.fromkeys((*rank.visible._FUNNEL_STAGES[2:-1], rank._OLD_VERIFIER_STAGE), equal),
               'paired_required_geometry_measured_on_first_complete_frame': equal}
    dynamic = {'paired_whole_tape_criteria': {
        rank._DYNAMIC_JOINT: equal, 'lumen_and_all_masks_representable_and_correct': equal,
        'all_rendered_role_centers_strictly_below_6_mm': equal,
    }, 'paired_geometry': {'geometry_measured': equal}}
    assert not rank._decision(station, dynamic)['eligible_for_further_validation']
    dynamic['paired_whole_tape_criteria'][rank._DYNAMIC_JOINT] = improvement
    decision = rank._decision(station, dynamic)
    assert decision['eligible_for_further_validation']
    assert not decision['formal_gate_3_opened']
    dynamic['paired_geometry']['geometry_measured'] = regression
    decision = rank._decision(station, dynamic)
    assert decision['recommendation'] == 'retain_v6_only'
    assert decision['regressed_guard_counts'] == ['dynamic.geometry_measured']


def test_dynamic_joint_requires_lumen_representable_exact_masks_and_strict_world_tolerance():
    def criteria(targets, masks, supported, residual):
        frame = SimpleNamespace(echo_targets=targets, centers_mm=np.zeros((6, 3)))
        truth = _tape_truth(SimpleNamespace(frames=[frame]))
        residuals = np.full((1, 6, 3), np.nan)
        residuals[truth.observed] = residual
        cell = SimpleNamespace(residuals=residuals, identity=[np.array(masks, dtype=int)],
                               trace=[{'supported_exact_slots': supported}])
        return rank._dynamic_criteria({'truth': truth, 'cells': {'v6': cell}}, 'v6')

    for targets, masks, supported in [((), (), ()), (((),), (0,), (True,))]:
        empty = criteria(targets, masks, supported, [0., 0., 0.])
        assert not empty['has_rendered_lumen']
        assert not empty['lumen_and_all_masks_representable_and_correct']
        assert not empty[rank._DYNAMIC_JOINT]
    assert not criteria(((0,),), (1,), (True,), [6., 0., 0.])[rank._DYNAMIC_JOINT]
    assert not criteria(((0,),), (1,), (True,), [np.nan, 0., 0.])[rank._DYNAMIC_JOINT]
    assert not criteria(((0,),), (1,), (False,), [1., 0., 0.])[rank._DYNAMIC_JOINT]
    assert not criteria(((0,),), (2,), (True,), [1., 0., 0.])[rank._DYNAMIC_JOINT]
    assert criteria(((0,),), (1,), (True,), [1., 0., 0.])[rank._DYNAMIC_JOINT]
