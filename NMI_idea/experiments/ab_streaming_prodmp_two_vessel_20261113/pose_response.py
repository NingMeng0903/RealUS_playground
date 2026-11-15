"""Joint finite-grid filter for a fixed, signed knee-response slope."""

import numpy as np

from residual import GRID_MM, ResidualFilter
from sensor import _normal_cell_prior


DEFAULT_SLOPE_GRID_MM_PER_DEG = np.linspace(-0.32, 0.32, 17)


class PoseResponseFilter:
    """Joint (persistent offset, OU residual, slope, visibility) posterior."""

    def __init__(self, sensor, dt_s, slopes_mm_per_deg=None, prior_sd=.15, grid_mm=GRID_MM):
        self.sensor = sensor
        self.residual = ResidualFilter(sensor, dt_s, "ou", grid_mm)
        slopes = DEFAULT_SLOPE_GRID_MM_PER_DEG if slopes_mm_per_deg is None else slopes_mm_per_deg
        self.slope_grid = np.array(slopes, dtype=float, copy=True)
        if not np.isfinite(self.slope_grid).all():
            raise ValueError("slope grid must be finite")
        self.slope_prior = _normal_cell_prior(self.slope_grid, prior_sd)
        count = len(self.slope_grid)

        # Collapse equal (w + z, k) rows for projection, retaining all
        # correlations in the filtering state.
        base_offsets = self.residual.offsets_mm
        self.offsets_mm = np.repeat(base_offsets, count, axis=0)
        self.slopes_mm_per_deg = np.tile(self.slope_grid, len(base_offsets))
        self.offset_indices = (self.residual.offset_indices[..., None] * count
                               + np.arange(count))
        self._flat_indices = self.offset_indices.ravel()
        self.visibility_prior = sensor.prior().sum(axis=0)

    def prior(self):
        return self.residual.prior()[:, :, None, :] * self.slope_prior[None, None, :, None]

    def start_scan(self, belief):
        """Carry the joint (w, k) map into a new visit with independent tapes."""
        persistent = belief.sum(axis=(1, 3))
        return (persistent[:, None, :, None] * self.residual.z_prior[None, :, None, None]
                * self.visibility_prior[None, None, None, :])

    def predict(self, belief, frames):
        """Propagate the dynamic axes, keeping the slope fixed."""
        if frames == 0:
            return belief
        transition = self.residual._transition_power(frames)
        propagated = np.swapaxes(np.swapaxes(belief, 1, -1) @ transition, 1, -1)
        return self.sensor.predict(propagated, frames)

    def update_projected(self, belief, packet, means_mm):
        likelihood, association = self.sensor.projected_likelihood(packet, means_mm)
        indices = self.offset_indices
        return self.sensor.condition(self.predict(belief, 1), likelihood[indices], association[indices])

    def grouped_prediction(self, belief, frames):
        prediction = self.predict(belief, frames)
        return np.column_stack([
            np.bincount(self._flat_indices, weights=prediction[..., visibility].ravel(),
                        minlength=len(self.offsets_mm))
            for visibility in range(prediction.shape[-1])
        ])

    def predicted_hit(self, belief, means_mm, frames):
        return self.sensor.hit_probability(self.grouped_prediction(belief, frames), means_mm)

    def offset_distribution(self, belief, frames=0):
        return self.grouped_prediction(belief, frames).sum(axis=-1)

    def offset_mean(self, belief, frames=0):
        """Mean w + z. The vertex-dependent slope term is projected separately."""
        return self.offset_distribution(belief, frames) @ self.offsets_mm

    def slope_distribution(self, belief):
        return belief.sum(axis=(0, 1, 3))
