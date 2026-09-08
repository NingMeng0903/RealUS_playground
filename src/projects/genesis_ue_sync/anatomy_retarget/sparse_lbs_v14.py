"""Precompute positive Blender weight slots without changing summation order."""
from __future__ import annotations

import numpy as np


class SparseLBSV14:
    """Evaluate the authored LBS while avoiding transforms for zero-weight slots.

    Contributions are restored to the original slot positions before NumPy's
    reduction. A CSR reduction could change the float64 summation order and
    break exact playback of the already saved float32 vertices.
    """

    def __init__(self, points, indices, weights):
        points = np.asarray(points, dtype=np.float64)
        indices = np.asarray(indices, dtype=np.int64)
        weights = np.asarray(weights, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or indices.shape != weights.shape or indices.shape[0] != len(points):
            raise ValueError('inconsistent sparse LBS dimensions')
        if not np.isfinite(points).all() or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError('invalid sparse LBS values')
        if np.any(indices < 0) or not np.allclose(weights.sum(1), 1., atol=2e-6, rtol=0):
            raise ValueError('invalid sparse LBS indices or normalization')
        self.shape = (*weights.shape, 3)
        self.rows, self.slots = np.nonzero(weights > 0)
        self.controllers = indices[self.rows, self.slots].copy()
        self.points = points[self.rows].copy()
        self.mass = weights[self.rows, self.slots].copy()
        self.maximum_controller = int(indices.max()) if indices.size else -1
        for name in ('rows', 'slots', 'controllers', 'points', 'mass'):
            getattr(self, name).setflags(write=False)

    def __call__(self, transforms):
        transforms = np.asarray(transforms, dtype=np.float64)
        if transforms.ndim != 3 or transforms.shape[1:] != (4, 4) or len(transforms) <= self.maximum_controller:
            raise ValueError('invalid sparse LBS transform dimensions')
        if not np.isfinite(transforms).all():
            raise ValueError('non-finite sparse LBS transforms')
        selected = transforms[self.controllers]
        mapped = np.einsum('nij,nj->ni', selected[:, :3, :3], self.points) + selected[:, :3, 3]
        contributions = np.zeros(self.shape, dtype=np.float64)
        contributions[self.rows, self.slots] = mapped * self.mass[:, None]
        return np.sum(contributions, axis=1)
