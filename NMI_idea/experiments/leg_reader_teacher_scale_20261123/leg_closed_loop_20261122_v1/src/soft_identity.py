"""Evaluator-side continuous identity potential for labeled development scans."""

import numpy as np


def region_potential(reader, region):
    frame_ids, slots, evidence = reader.ranked(region)
    if not len(frame_ids):
        return 0.
    is_target = np.array([reader.frames[i][3][j] == 0 for i, j in zip(frame_ids, slots)])
    confidence = np.clip(evidence, 0., 1.)
    target = np.max(confidence[is_target], initial=0.)
    false_peak = np.max(confidence[~is_target], initial=0.)
    false_commitment = max(0., (false_peak - .36) / .64)
    return .65 * target - .50 * false_commitment ** 2


def update_identity(reader, spec, first, last, potentials):
    regions = set()
    for frame in reader.frames[first:last]:
        nearby = int(np.floor(frame[0] + .5))
        start = max(0, nearby - spec[1] - 3)
        end = min(len(spec[4]), nearby - spec[1] + 4)
        regions.update(spec[4][start:end])
    change = 0.
    for region in regions:
        after = region_potential(reader, int(region))
        change += after - potentials[region]
        potentials[region] = after
    return change
