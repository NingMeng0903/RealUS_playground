"""Joint estimation of a constant registration offset and local tissue motion."""

from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr, ndtri

from sensor import Sensor


TAU_S = .8
STATIONARY_SD_MM = 1.5
SUPPORT_MM = 6.
GRID_MM = .5
MOTION_MODES = ("static", "ou", "transient")


def motion_tape(rng, frames, dt_s, mode):
    """Draw an evaluator-only lateral shift in physical mm at exposure times."""
    if mode == "static":
        return np.zeros(frames)
    retention = np.exp(-dt_s / TAU_S)
    z = rng.normal(0., STATIONARY_SD_MM, size=frames)
    for k in range(1, frames):
        z[k] = retention * z[k - 1] + np.sqrt(1. - retention**2) * z[k]
    if mode == "ou":
        return z
    if mode != "transient":
        raise ValueError(f"unknown motion mode: {mode!r}")
    time = dt_s * np.arange(frames) - rng.uniform(.65, 2.1)
    amplitude = rng.choice((-1., 1.)) * rng.uniform(3., 5.)
    ramp_s, recovery_s = .3, rng.uniform(.45, .85)
    ramp = .5 - .5 * np.cos(np.pi * np.clip(time / ramp_s, 0., 1.))
    return .30 * z + amplitude * ramp * np.exp(-np.maximum(time - ramp_s, 0.) / recovery_s)


def ou_lattice(grid, dt_s, tau_s, sigma_mm):
    """Discretize a stationary Gaussian pair using both source and destination cells."""
    rho = np.exp(-dt_s / tau_s)
    bounds = np.r_[-np.inf, .5 * (grid[1:] + grid[:-1]), np.inf]
    cdf = ndtr(bounds / sigma_mm)
    node, weight = leggauss(48)
    source = sigma_mm * ndtri(cdf[:-1, None] + np.diff(cdf)[:, None] * (.5 + .5 * node))
    cdf_destination = ndtr((bounds - rho * source[..., None]) / (sigma_mm * np.sqrt(1. - rho**2)))
    joint = np.einsum("i,j,ijk->ik", np.diff(cdf), .5 * weight, np.diff(cdf_destination, axis=-1))
    # Stationary OU is reversible. Enforce symmetry at quadrature precision
    # and share the resulting marginal exactly with the IID filter.
    joint = .5 * (joint + joint.T)
    mass = joint.sum(axis=1)
    return joint / mass[:, None], mass / mass.sum()


class ResidualFilter:
    """A joint probability map over (persistent offset, lateral residual, visibility)."""

    def __init__(self, sensor: Sensor, dt_s: float, model: str, grid_mm: float = GRID_MM):
        if model not in ("iid", "ou"):
            raise ValueError("residual model must be 'iid' or 'ou'")
        self.sensor = sensor
        self.grid_mm = grid_mm
        self.z_mm = np.linspace(-SUPPORT_MM, SUPPORT_MM, 1 + round(2. * SUPPORT_MM / grid_mm))
        transition, self.z_prior = ou_lattice(self.z_mm, dt_s, TAU_S, STATIONARY_SD_MM)
        self.transition = transition if model == "ou" else np.tile(self.z_prior, (len(self.z_mm), 1))
        nominal = sensor.offsets_mm[:, None, :] + np.column_stack((self.z_mm, np.zeros_like(self.z_mm)))
        self.offsets_mm, indices = np.unique(nominal.reshape(-1, 2), axis=0, return_inverse=True)
        self.offset_indices = indices.reshape(nominal.shape[:2])

    def prior(self):
        return self.sensor.prior()[:, None, :] * self.z_prior[None, :, None]

    @lru_cache(maxsize=128)
    def _transition_power(self, frames):
        return np.linalg.matrix_power(self.transition, frames)

    def predict(self, belief, frames):
        if frames == 0:
            return belief
        propagated = np.swapaxes(np.swapaxes(belief, 1, 2) @ self._transition_power(frames), 1, 2)
        return self.sensor.predict(propagated, frames)

    def update_projected(self, belief, packet, means_mm):
        likelihood, association = self.sensor.projected_likelihood(packet, means_mm)
        indices = self.offset_indices
        return self.sensor.condition(self.predict(belief, 1), likelihood[indices], association[indices])

    def _total_weights(self, weights):
        return np.bincount(self.offset_indices.ravel(), weights.ravel(), minlength=len(self.offsets_mm))

    def offset_distribution(self, belief, frames=0):
        return self._total_weights(self.predict(belief, frames).sum(axis=-1))

    def offset_mean(self, belief, frames=0):
        return self.offset_distribution(belief, frames) @ self.offsets_mm

    def residual_mean(self, belief):
        return float(belief.sum(axis=(0, 2)) @ self.z_mm)

    def predicted_hit(self, belief, means_mm, frames):
        pred = self.predict(belief, frames)
        # Group equal total offsets so the same physical projection contributes
        # once, preserving the association between residual and visibility.
        grouped = np.stack([self._total_weights(pred[..., i]) for i in range(pred.shape[-1])], axis=-1)
        return self.sensor.hit_probability(grouped, means_mm)


def offset_stats(model, belief, frames=0):
    """Mean and central 90% interval of the effective lateral displacement."""
    weights = model.offset_distribution(belief, frames)
    weights = weights / weights.sum()
    lateral = model.offsets_mm[:, 0]
    bounds = lateral[np.searchsorted(np.cumsum(weights), [.05, .95])]
    return weights @ model.offsets_mm, bounds
