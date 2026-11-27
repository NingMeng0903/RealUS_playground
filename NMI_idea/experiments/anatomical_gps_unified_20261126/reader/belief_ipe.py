"""Pose-free relative material integrated positional encoding (Proposition 3)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh, splu

from sim.metric_graph_gp import MetricGraphPrior


ROLE_COUNT = 6
JUNCTIONS = ('SFJ', 'CFA_bif', 'CFV_bif')
PAIRS = np.array(list(combinations(range(ROLE_COUNT), 2)), dtype=np.intp)
SPECTRAL_WIDTH = 8
FREQUENCIES = np.eye(3)[:, None, :] * (2. ** np.arange(6))[None, :, None]
FREQUENCIES = (FREQUENCIES / np.array([1., 20., 1.])).reshape(-1, 3)


def wrap_angle(theta):
    return (theta + np.pi) % (2. * np.pi) - np.pi


def gaussian_fourier(mean, covariance, frequencies=FREQUENCIES):
    """Exact real and imaginary moments of the Gaussian characteristic function."""
    phase = mean @ frequencies.T
    variance = np.einsum('fi,...ij,fj->...f', frequencies, covariance, frequencies)
    attenuation = np.exp(-.5 * np.maximum(variance, 0.))
    return np.stack((attenuation * np.sin(phase), attenuation * np.cos(phase)), axis=-1)


def kirchhoff_eigenfunctions(prior: MetricGraphPrior, k: int = SPECTRAL_WIDTH):
    """Lowest modes, including component constants, M-orthonormal on FE knots."""
    stiffness = prior.kirchhoff_laplacian()
    mass = prior.graph.lumped_mass_mm
    k = min(k, len(mass))
    if len(mass) <= k + 1:
        eigenvalues, eigenvectors = eigh(stiffness.toarray(), np.diag(mass), subset_by_index=(0, k - 1))
    else:
        m = sparse.diags(mass, format='csc')
        # Shift below the zero modes, which may be repeated on disconnected traces.
        eigenvalues, eigenvectors = eigsh(stiffness, k=k, M=m, sigma=-1.e-5,
                                          v0=np.random.default_rng(123).standard_normal(len(mass)), tol=1.e-12)
        order = np.argsort(eigenvalues)
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    pivots = np.argmax(np.abs(eigenvectors), axis=0)
    eigenvectors *= np.sign(eigenvectors[pivots, np.arange(k)])
    return eigenvalues, eigenvectors


@dataclass(frozen=True)
class _StationFeatures:
    structural: np.ndarray
    pair_ids: np.ndarray
    pair_members: np.ndarray
    nominal: np.ndarray
    contrasts: sparse.csr_matrix


class BeliefIPE:
    """Fixed-width role and pair tokens; only relative material moments enter.

    At the same nominal height, every role's affine map has identical shared
    columns. Subtracting the sparse maps *before* applying the belief removes
    arbitrary shared drift and propagates the full covariance, including all
    cross terms. Pose is absent from the material feature computation.
    """

    role_width = SPECTRAL_WIDTH + 1 + len(JUNCTIONS) + 1
    pair_width = 2 * len(FREQUENCIES) + 1
    feature_dim = ROLE_COUNT * role_width + len(PAIRS) * pair_width

    def __init__(self, prior: MetricGraphPrior):
        if prior.coordinate_mode != 'material':
            raise ValueError('Belief-IPE needs a material prior')
        self.prior = prior
        self.eigenvalues, self.eigenfunctions = kirchhoff_eigenfunctions(prior)
        self._station = lru_cache(maxsize=256)(self._make_station)

    def _make_station(self, h_mm: float) -> _StationFeatures:
        structural = np.zeros((ROLE_COUNT, self.role_width))
        graph = self.prior.graph
        junction_heights = [graph.vertices[graph.junctions[j].vertex_id].h_mm if j in graph.junctions
                            else np.nan for j in JUNCTIONS]
        signed_mm = np.nan_to_num((h_mm - np.array(junction_heights)) / 20.)
        queries = [self.prior.center_query(h_mm, r) for r in range(ROLE_COUNT)]
        for role, query in enumerate(queries):
            if query is None:
                continue
            a, b, t = graph._station_query(h_mm, role)
            spectral = self.eigenfunctions
            structural[role, :spectral.shape[1]] = (1. - t) * spectral[a] + t * spectral[b]
            edge = self.prior.station_edge(h_mm, role)
            structural[role, SPECTRAL_WIDTH] = 0. if edge is None else edge[1]
            structural[role, SPECTRAL_WIDTH + 1:-1] = signed_mm
            structural[role, -1] = 1.
        roles = np.asarray([r for r, q in enumerate(queries) if q is not None], dtype=np.intp)
        lookup = np.full(ROLE_COUNT, -1, dtype=np.intp)
        lookup[roles] = np.arange(roles.size)
        members = lookup[PAIRS]
        ids = np.flatnonzero((members >= 0).all(axis=1))
        ref = queries[roles[0]] if roles.size else None
        others = [queries[r] for r in roles[1:]]
        design = (sparse.vstack([q.operator - ref.operator for q in others], format='csr') if others
                  else sparse.csr_matrix((0, self.prior.layout.size)))
        design.eliminate_zeros()
        nominal = np.asarray([q.nominal - ref.nominal for q in others]).reshape(-1, 3)
        return _StationFeatures(structural, ids, members[ids], nominal, design)

    def contrasts(self, h_mm, mean, precision_csc=None, *, covariance=None, factor=None):
        """Return valid pair IDs, wrapped means, and exact 3×3 covariance per pair."""
        station = self._station(float(h_mm))
        h = station.contrasts
        if not h.shape[0]:
            return station.pair_ids, np.empty((0, 3)), np.empty((0, 3, 3))
        # All pair contrasts are differences in an independent reference
        # basis. Shared coordinates cancel before the solve, and only 3(n-1)
        # right-hand sides are needed instead of 3n(n-1)/2.
        basis = np.zeros((len(station.nominal) + 1, 3))
        basis[1:] = station.nominal + (h @ mean).reshape(-1, 3)
        i, j = station.pair_members.T
        mu = basis[i] - basis[j]
        mu[:, 0] = wrap_angle(mu[:, 0])
        if covariance is None:
            if factor is None:
                factor = splu(self.prior.precision if precision_csc is None else precision_csc.tocsc())
            sol = factor.solve(h.T.toarray(order='F'))
        else:
            sol = np.asarray(covariance) @ h.T.toarray()
        size = len(basis)
        joint = np.zeros((size, 3, size, 3))
        joint[1:, :, 1:, :] = (h @ sol).reshape(size - 1, 3, size - 1, 3)
        cov = joint[i, :, i, :] + joint[j, :, j, :] - joint[i, :, j, :] - joint[j, :, i, :]
        return station.pair_ids, mu, .5 * (cov + np.swapaxes(cov, -1, -2))

    def features(self, h_mm, mean, precision_csc=None, *, covariance=None, factor=None):
        station = self._station(float(h_mm))
        ids, mu, cov = self.contrasts(h_mm, mean, precision_csc, covariance=covariance, factor=factor)
        pairs = np.zeros((len(PAIRS), self.pair_width))
        pairs[ids, :-1] = gaussian_fourier(mu, cov).reshape(len(ids), self.pair_width - 1)
        pairs[ids, -1] = 1.
        return np.concatenate((station.structural.ravel(), pairs.ravel()))
