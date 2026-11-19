"""Relative imaging features and marginal target-detection probabilities."""

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.special import ndtr

from adaptive_controller import FRAME_STRIDE, station_bin
from recovery_world import FOV_MAX, FOV_MIN, NOISE_SD, PACKET_OUTAGE


QUADRATURE_NODES = (-np.sqrt(3.), 0., np.sqrt(3.))
QUADRATURE_WEIGHTS = (1. / 6., 2. / 3., 1. / 6.)


@dataclass(frozen=True)
class ViewCalibration:
    mean_mm: tuple = (0., 0.)
    sd_mm: tuple = (0., 0.)
    width: float = 1.

    def samples(self):
        if not any(self.sd_mm):
            return ((self.mean_mm[0], self.mean_mm[1], 1.),)
        mx, md = self.mean_mm
        sx, sd = self.sd_mm
        return tuple((mx + nx * sx, md + nd * sd, wx * wd)
                     for (nx, wx), (nd, wd) in product(zip(QUADRATURE_NODES, QUADRATURE_WEIGHTS), repeat=2))


def path_views(candidate):
    """Use the same five acquisition instants as the incumbent features."""
    ticks = np.arange(2 * FRAME_STRIDE, len(candidate.s), 2 * FRAME_STRIDE)
    return tuple(zip(candidate.s[ticks], candidate.lateral.x[ticks], strict=True))


def signed_margins(world, state, views):
    outside_scale = (FOV_MAX - FOV_MIN) * .5
    windows = []
    for s, x in views:
        y = world.target_grid(s, state.angle) - [x, 0.]
        margin = np.minimum(y - FOV_MIN, FOV_MAX - y) / outside_scale
        lateral = (y[:, 0] - (FOV_MAX[0] + FOV_MIN[0]) * .5) / outside_scale[0]
        windows.append(state.belief @ np.column_stack((lateral, margin)))
    return np.asarray(windows).ravel()


def target_detection_probabilities(world, state, views, calibration):
    """Average conditional sensor probabilities over the public belief and offset model."""
    samples = calibration.samples()
    probabilities = []
    for s, x in views:
        probability = 0.
        for dx, dd, weight in samples:
            means, visible = world.imaging(s, x - dx, state.angle, width=calibration.width, depth_shift=dd)
            target = means[:, 0]
            detected_in_window = np.prod(ndtr((FOV_MAX - target) / NOISE_SD)
                                         - ndtr((FOV_MIN - target) / NOISE_SD), axis=1)
            probability += weight * (state.belief @ (visible[:, 0] * detected_in_window))
        probabilities.append((1. - PACKET_OUTAGE) * probability)
    return np.asarray(probabilities)


def detection_features(probabilities, views, query):
    expected_query_packets = sum(p for p, (s, _) in zip(probabilities, views, strict=True) if station_bin(s) == query)
    return np.r_[probabilities, expected_query_packets]
