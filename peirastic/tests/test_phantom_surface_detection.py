"""Detection must preserve real curved/white points above the old height cap."""

import numpy as np

from peirastic.DEMO.phathom_scanning.detect import select_top_points, _upper_plateau


def test_high_phantom_with_white_texture_keeps_the_complete_measured_top():
    u, v = np.meshgrid(np.linspace(-0.10, 0.10, 51), np.linspace(-0.08, 0.08, 41))
    height = 0.175 + 0.012 * np.cos(u * np.pi / 0.2)
    top = np.column_stack([u.ravel(), v.ravel(), height.ravel()])
    table = np.column_stack([u.ravel(), v.ravel(), np.zeros(u.size)])
    colors = np.tile([0.55, 0.35, 0.18], (len(top), 1))
    white = (np.abs(top[:, 0]) < 0.015) & (np.abs(top[:, 1]) < 0.025)
    colors[white] = [0.96, 0.95, 0.92]
    picked, _ = select_top_points(np.vstack([table, top]),
                                  np.vstack([np.tile([0.3, 0.55, 0.85], (len(table), 1)), colors]))
    assert len(picked) >= 0.97 * len(top)
    assert np.ptp(picked[:, 0]) > 0.195
    assert np.ptp(picked[:, 2]) > 0.010
    assert picked[:, 2].min() > 0.17
    assert np.any((np.abs(picked[:, 0]) < 0.005) & (np.abs(picked[:, 1]) < 0.005))


def test_curved_top_is_not_cut_to_a_planar_eight_mm_slab():
    u, v = np.meshgrid(np.linspace(-0.10, 0.10, 51), np.linspace(-0.08, 0.08, 41))
    top = np.column_stack([u.ravel(), v.ravel(), (0.20 - 2.0 * u**2).ravel()])
    side = np.array([[0.10, y, z] for y in np.linspace(-0.08, 0.08, 41)
                     for z in np.linspace(0.08, 0.176, 35)])
    selected = _upper_plateau(np.vstack([top, side]))
    assert selected[:, 0].min() < -0.095
    assert selected[:, 0].max() > 0.095
    assert np.ptp(selected[:, 2]) >= 0.019
    assert selected[:, 2].min() > 0.16  # do not retain the vertical side
