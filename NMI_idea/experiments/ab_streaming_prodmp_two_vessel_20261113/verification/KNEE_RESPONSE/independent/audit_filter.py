"""Independent contractions and emissions for the knee-response integration."""

import argparse
import json
from math import tau
from pathlib import Path
import sys

import numpy as np
from scipy.special import ndtr

EXPERIMENT_RELATIVE = Path("NMI_idea/experiments/ab_streaming_prodmp_two_vessel_20261113")
FRAME_STRIDE = 5
SAMPLE_DT_S = .02
CENTER_TOL_MM = 3.
IDENTITY_MIN = .95
FOV_MIN_MM = np.array([-25., 3.])
FOV_MAX_MM = np.array([25., 52.])
NOISE_SD_MM = np.array([1., .68])
P_CLEAR = np.array([[.72, .28], [.06, .94]])
STATE_INDEX = ((0, 0), (0, 1), (1, 0), (1, 1))


def configure_experiment(requested):
    """Load production modules from a specified or nearby checkout."""
    if requested is None:
        possible = (item for parent in Path(__file__).resolve().parents
                    for item in (parent, parent / EXPERIMENT_RELATIVE,
                                 parent / "RealUS_playground" / EXPERIMENT_RELATIVE))
        requested = next((item for item in possible if (item / "pose_response.py").is_file()), None)
    if requested is None or not (requested / "pose_response.py").is_file():
        raise ValueError("pass --experiment with the path to the streaming experiment directory")
    experiment = requested.resolve()
    sys.path.insert(0, str(experiment))
    return experiment


def independent_predict(model, belief, frames):
    clear_matrix = np.array([
        [P_CLEAR[source[0], dest[0]] * P_CLEAR[source[1], dest[1]] for dest in STATE_INDEX]
        for source in STATE_INDEX
    ])
    dz = np.linalg.matrix_power(model.residual.transition, frames)
    dc = np.linalg.matrix_power(clear_matrix, frames)
    return np.einsum("wzka,zy,ab->wykb", belief, dz, dc, optimize=True)


def independent_emit(packet, means):
    visible = np.isfinite(means).all(axis=-1)
    finite_means = np.nan_to_num(means, nan=0.0)
    in_bounds_prob = (
        ndtr((FOV_MAX_MM - finite_means) / NOISE_SD_MM)
        - ndtr((FOV_MIN_MM - finite_means) / NOISE_SD_MM)
    ).prod(axis=-1)
    ratios = (packet - finite_means[..., None, :]) / NOISE_SD_MM
    density = np.exp(-0.5 * np.sum(ratios ** 2, axis=-1)) / (tau * NOISE_SD_MM.prod())
    mass = np.empty((len(means), len(STATE_INDEX)))
    artery_mass = np.zeros(mass.shape + (len(packet),))
    for j, state in enumerate(STATE_INDEX):
        clear = visible & np.asarray(state, dtype=bool)
        miss = 1.0 - clear * in_bounds_prob
        present = clear[..., None] * density
        if len(packet) == 0:
            mass[:, j] = miss.prod(axis=-1)
        elif len(packet) == 1:
            artery_mass[:, j, 0] = present[:, 0, 0] * miss[:, 1]
            mass[:, j] = artery_mass[:, j, 0] + present[:, 1, 0] * miss[:, 0]
        else:
            artery_mass[:, j, 0] = present[:, 0, 0] * present[:, 1, 1]
            artery_mass[:, j, 1] = present[:, 0, 1] * present[:, 1, 0]
            mass[:, j] = artery_mass[:, j].sum(axis=-1)
    return mass, artery_mass


def geometry_indices(model):
    static = model.sensor.offsets_mm
    residual = model.residual.z_mm
    key_to_row = {
        (*offset, slope): i
        for i, (offset, slope) in enumerate(zip(model.offsets_mm, model.slopes_mm_per_deg, strict=True))
    }
    return np.array([
        [[key_to_row[(u + z, v, slope)] for slope in model.slope_grid] for z in residual]
        for u, v in static
    ])


def independent_condition(model, prior, packet, means, index):
    likelihood, artery_mass = independent_emit(packet, means)
    pred = independent_predict(model, prior, 1)
    joint = pred * likelihood[index]
    norm = joint.sum()
    assoc = (pred[..., None] * artery_mass[index]).sum(axis=(0, 1, 2, 3)) / norm
    return joint / norm, assoc


def pointwise_project(geometry, pose, offsets, response, envelope):
    curves = (geometry.vessels_m + .001 * (offsets @ geometry.offset_axes.T)[:, None, None, :]
              + .001 * (response[:, None] * envelope)[:, :, None, None]
              * geometry.offset_axes[:, 0])
    relative = curves - pose[:3, 3]
    normal_distance = relative @ pose[:3, 0]
    projects = np.full((len(curves), 2, 2), np.nan)
    for role in range(2):
        d0, d1 = normal_distance[:, :-1, role], normal_distance[:, 1:, role]
        crosses = ((d0 <= 0) & (d1 > 0)) | ((d0 >= 0) & (d1 < 0))
        weight = np.divide(-d0, d1 - d0, where=crosses, out=np.zeros_like(d0))
        hit_points = ((1. - weight[..., None]) * relative[:, :-1, role]
                      + weight[..., None] * relative[:, 1:, role])
        lateral_depth = 1000. * (hit_points @ pose[:3, 1:3])
        available_depth = np.where(crosses & (lateral_depth[..., 1] > 0.), lateral_depth[..., 1], np.inf)
        candidates = available_depth.argmin(axis=-1)
        finite = np.isfinite(available_depth[np.arange(len(curves)), candidates])
        projects[finite, role] = lateral_depth[np.flatnonzero(finite), candidates[finite]]
    return projects


def check_filter():
    from atlas_scan import FRAME_DT_S
    from pose_response import PoseResponseFilter
    from residual import ResidualFilter
    from sensor import Sensor

    rng = np.random.default_rng(43943)
    sensor = Sensor()
    model = PoseResponseFilter(sensor, FRAME_DT_S)
    prior = model.prior()
    independent_index = geometry_indices(model)
    np.testing.assert_array_equal(independent_index, model.offset_indices)
    correlated = rng.uniform(size=prior.shape) ** 5 * prior
    correlated /= correlated.sum()
    draws = np.column_stack((rng.uniform(-31., 31., (len(model.offsets_mm), 2, 1)).reshape(-1),
                             rng.uniform(1., 54., (len(model.offsets_mm), 2, 1)).reshape(-1)))
    means = draws.reshape(-1, 2, 2)
    means[::11, 0] = np.nan
    means[::31, 1] = np.nan
    errs = {}
    for steps in (0, 1, 5, 10, 20):
        actual = model.predict(correlated, steps)
        expected = independent_predict(model, correlated, steps)
        errs[f"predict_{steps}"] = float(np.max(np.abs(actual - expected)))
        np.testing.assert_allclose(actual, expected, rtol=3.e-14, atol=3.e-17)
        np.testing.assert_allclose(actual.sum(axis=(1, 3)), correlated.sum(axis=(1, 3)), atol=2.e-16, rtol=0.)
        hit_bounds = np.array([[-CENTER_TOL_MM, FOV_MIN_MM[1]], [CENTER_TOL_MM, FOV_MAX_MM[1]]])
        artery_means = means[independent_index, 0, :]
        event = np.prod(ndtr((hit_bounds[1] - artery_means) / NOISE_SD_MM)
                        - ndtr((hit_bounds[0] - artery_means) / NOISE_SD_MM), axis=-1)
        clear_artery = [j for j, state in enumerate(STATE_INDEX) if state[0]]
        expected_hit = np.sum(expected[..., clear_artery] * np.nan_to_num(event)[..., None])
        hit = model.predicted_hit(correlated, means, steps)
        np.testing.assert_allclose(hit, expected_hit, atol=2.e-16, rtol=0.)
        errs[f"hit_{steps}"] = abs(hit - expected_hit)
    for packet in (np.empty((0, 2)), np.array([[3., 28.]]), np.array([[-1., 20.], [8., 25.]])):
        expected, expected_assoc = independent_condition(model, correlated, packet, means, independent_index)
        actual, assoc = model.update_projected(correlated, packet, means)
        np.testing.assert_allclose(actual, expected, rtol=5.e-14, atol=2.e-16)
        np.testing.assert_allclose(assoc, expected_assoc, rtol=0., atol=3.e-15)
        errs[f"posterior_{len(packet)}"] = float(np.max(np.abs(actual - expected)))
        errs[f"association_{len(packet)}"] = float(np.max(np.abs(assoc - expected_assoc), initial=0.))
    reset = model.start_scan(correlated)
    np.testing.assert_allclose(reset.sum(axis=(1, 3)), correlated.sum(axis=(1, 3)), atol=2.e-16, rtol=0.)
    np.testing.assert_allclose(reset.sum(axis=(0, 2, 3)), model.residual.z_prior, atol=2.e-16, rtol=0.)
    visibility = np.array([P_CLEAR[1, 0], P_CLEAR[0, 1]]) / (P_CLEAR[1, 0] + P_CLEAR[0, 1])
    np.testing.assert_allclose(reset.sum(axis=(0, 1, 2)), np.outer(visibility, visibility).ravel(),
                               atol=1.e-14, rtol=0.)

    baseline = ResidualFilter(sensor, FRAME_DT_S, "ou")
    collapsed = PoseResponseFilter(sensor, FRAME_DT_S, slopes_mm_per_deg=[0.])
    means0 = np.broadcast_to([[-4., 25.], [3., 31.]], (len(baseline.offsets_mm), 2, 2)).copy()
    means0[:, :, 0] += baseline.offsets_mm[:, 0, None]
    means0[:, :, 1] += baseline.offsets_mm[:, 1, None]
    ordinary, zero, full = baseline.prior(), collapsed.prior(), model.prior()
    for packet in (np.array([[-3., 25.], [4., 31.]]), np.array([[4., 31.]]), np.empty((0, 2))) * 9:
        ordinary, identity = baseline.update_projected(ordinary, packet, means0)
        zero, zero_identity = collapsed.update_projected(zero, packet, means0)
        full, full_identity = model.update_projected(full, packet, np.repeat(means0, len(model.slope_grid), axis=0))
        np.testing.assert_allclose(zero[:, :, 0, :], ordinary, rtol=0., atol=3.e-16)
        np.testing.assert_allclose(full.sum(axis=2), ordinary, rtol=0., atol=3.e-16)
        np.testing.assert_allclose(zero_identity, identity, rtol=0., atol=3.e-15)
        np.testing.assert_allclose(full_identity, identity, rtol=0., atol=3.e-14)
        np.testing.assert_allclose(model.slope_distribution(full), model.slope_prior, rtol=0., atol=2.e-15)
    return {key: float(value) for key, value in errs.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path)
    args = parser.parse_args()
    configure_experiment(args.experiment)
    print(json.dumps(check_filter(), indent=2, allow_nan=False))
