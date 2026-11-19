"""Check controller causality using actual FK and a perturbed future tape."""

import argparse
from pathlib import Path
import sys

import numpy as np


ROOT = Path.cwd()
EXPERIMENT = ROOT / 'NMI_idea/experiments/ab_streaming_prodmp_two_vessel_20261113'
sys.path.insert(0, str(EXPERIMENT))

from atlas_geometry import ScanTemplate
from atlas_scan import FRAME_COUNT, FRAME_DT_S, FRAME_STRIDE, PAIRED_ARMS, Scan
from check_kinematics import load_witness
from NMI_idea.experiments.ab_module_ladder_20261110.capability.capability import CapabilityEvaluator
from primitive import TOTAL_STEPS
from residual import motion_tape
from sensor import Scene


def check(scan):
    rng = np.random.default_rng(113)
    scene = scan.sensor.scene(rng, FRAME_COUNT)
    motion = motion_tape(rng, FRAME_COUNT, FRAME_DT_S, 'ou')

    boundary = 14
    future_motion = motion.copy()
    future_motion[boundary:] += 4.
    future_clear = scene.clear.copy()
    future_clear[boundary:] = False
    future_noise = scene.noise.copy()
    future_noise[boundary:] += 10.
    future = Scene(scene.offset, scene.bias, future_clear, future_noise)

    observation = scan.observation

    def corrupted_evaluation(case, frame, pose, z_mm):
        packet, roles, physical = observation(case, frame, pose, z_mm)
        return packet, 1 - roles, physical + 18.

    first_q = first_packet = None
    for arm in PAIRED_ARMS:
        record, traces = scan.execute(scene, arm, motion)
        altered_record, altered_traces = scan.execute(future, arm, future_motion)
        assert record['failure'] is None and altered_record['failure'] is None
        assert len(record['frames']) == len(altered_record['frames']) == FRAME_COUNT
        assert record['frames'][:boundary] == altered_record['frames'][:boundary]
        assert record['frames'][boundary]['packet_mm'] != altered_record['frames'][boundary]['packet_mm']
        prefix_tick = boundary * FRAME_STRIDE
        old_decisions = [item for item in record['decisions'] if item['step'] < prefix_tick]
        new_decisions = [item for item in altered_record['decisions'] if item['step'] < prefix_tick]
        assert old_decisions == new_decisions
        for original, altered in zip(traces, altered_traces, strict=True):
            assert len(original) == len(altered) == TOTAL_STEPS + 1
            np.testing.assert_array_equal(original[:prefix_tick + 1], altered[:prefix_tick + 1])
        if first_q is None:
            first_q, first_packet = traces[0][0], record['frames'][0]['packet_mm']
        np.testing.assert_array_equal(traces[0][0], first_q)
        assert record['frames'][0]['packet_mm'] == first_packet

        scan.observation = corrupted_evaluation
        try:
            edited, edited_traces = scan.execute(scene, arm, motion)
        finally:
            scan.observation = observation
        assert not edited['success'] and edited['wrong']
        assert record['decisions'] == edited['decisions']
        assert record['revisions'] == edited['revisions']
        for original, altered in zip(traces, edited_traces, strict=True):
            np.testing.assert_array_equal(original, altered)
        for actual, alternate in zip(record['frames'], edited['frames'], strict=True):
            for key in ('packet_mm', 'artery_probability', 'selected_slot', 'accepted', 'offset_mean_mm'):
                assert actual[key] == alternate[key]
        print(arm, 'passed future-tape and evaluator-truth perturbation', flush=True)

    legacy, old_trace = scan.execute(scene, 'mpc_stream')
    explicit_zero, new_trace = scan.execute(scene, 'mpc_stream', np.zeros(FRAME_COUNT))
    for old, new in zip(old_trace, new_trace, strict=True):
        np.testing.assert_array_equal(old, new)
    assert legacy['decisions'] == explicit_zero['decisions']
    for old, new in zip(legacy['frames'], explicit_zero['frames'], strict=True):
        assert {k: new[k] for k in old} == old
    print('Explicit zero motion matched all legacy fields and paths bitwise', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--among-us', type=Path, required=True)
    parser.add_argument('--repair', type=Path, required=True)
    args = parser.parse_args()
    bank = EXPERIMENT.parent / 'ab_full_b_rollout_20261113/motion/BANK_MAIN.json.xz'
    _, witness = load_witness(bank)
    shape = ScanTemplate(args.among_us, args.repair).instantiate((0., 0.), 15.)
    with np.load(EXPERIMENT / 'verification/DEMONSTRATIONS.npz', allow_pickle=False) as data:
        scan = Scan(shape, CapabilityEvaluator(device='cpu'), witness, data['shape_regression'])
    check(scan)


if __name__ == '__main__':
    main()
