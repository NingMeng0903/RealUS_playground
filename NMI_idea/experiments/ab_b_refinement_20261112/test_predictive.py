"""Sensor-only checks for :mod:`predictive` (no geometry or truth labels)."""

from __future__ import annotations

import numpy as np

from predictive import predictive_sets


MIXTURE = ((0.85, 0.85), (0.15, 4.0))


def test_miss_and_clutter_masses_are_explicit() -> None:
    result = predictive_sets(
        (np.zeros(1), np.eye(1)),
        [(np.ones(1), 0.0)],
        half_width=5.0,
        p_detect=0.0,
        noise_mixture=MIXTURE,
        clutter_rate=0.2,
        count=8,
    )
    assert len(result) <= 8
    assert abs(sum(weight for _, weight in result) - 1.0) < 1.0e-12
    assert abs(next(weight for values, weight in result if values == ()) - 0.8) < 1.0e-12
    assert abs(sum(weight for values, weight in result if values) - 0.2) < 1.0e-12
    assert all(abs(value) <= 5.0 + 1.0e-12 for values, _ in result for value in values)


def test_correlated_pair_is_joint_and_sorted() -> None:
    result = predictive_sets(
        (np.zeros(2), np.array([[1.0, 0.9], [0.9, 1.0]])),
        [(np.array([1.0, 0.0]), 0.0), (np.array([0.0, 1.0]), 0.0)],
        half_width=100.0,
        p_detect=1.0,
        noise_mixture=((1.0, 1.0e-6),),
        clutter_rate=0.0,
        count=8,
    )
    assert len(result) == 8
    assert all(len(values) == 2 and tuple(values) == tuple(sorted(values)) for values, _ in result)
    assert abs(sum(weight for _, weight in result) - 1.0) < 1.0e-12


def test_repeated_call_is_seed_free_and_bounded() -> None:
    kwargs = dict(
        half_width=5.0,
        p_detect=0.91,
        noise_mixture=MIXTURE,
        clutter_rate=0.06,
        count=4,
    )
    belief = (np.array([0.25]), np.array([[0.7]]))
    models = [(np.array([1.0]), -0.3)]
    assert predictive_sets(belief, models, **kwargs) == predictive_sets(belief, models, **kwargs)
    assert len(predictive_sets(belief, models, **kwargs)) <= 4

