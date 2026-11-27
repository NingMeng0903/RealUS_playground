"""Training-only vascular graph prior, linear geometry, and validity audit.

The continuous reference is (kappa² - Δ_G)^-2 with Kirchhoff/Neumann
boundary conditions. The implemented Gaussian uses a mass-lumped P1 weak
form: junction values are exactly shared, but the first-element slopes of
an individual FE draw need not sum to zero. Shared drift knot states follow
exact Matérn-3/2 transitions. Geometry uses the piecewise-linear embedding
at *nominal* world-Y stations (a frozen-chart linearization, not evaluation
of a continuous GP at the random displaced height).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from io import BytesIO
from pathlib import Path

import numpy as np
from numba import njit
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.special import gammainc

from .anatomy_assets import (
    FULL_LEG_Y_BOUNDS_MM, NEUTRAL_SMPLX_SURFACE_PATH, THIGH_NYSORA_Y_BOUNDS_MM,
    verify_neutral_smplx_surface_sha256,
)
from .material_chart import RightLegMaterialChart


_CONTACT_ATLAS = (
    Path(__file__).resolve().parents[2] / 'ab_validation_20261110'
    / 'repair_20261110/harmonic_full_domain_followup/contact_multichart_right.npz'
)
_DEFAULT_KAPPA = 1. / 45.
_DOMAINS = {'thigh_nysora': THIGH_NYSORA_Y_BOUNDS_MM, 'full_leg': FULL_LEG_Y_BOUNDS_MM}
_COORDINATE_MODES = ('material', 'cartesian_xyz')
# Native MedSeg voxel-contact identities. Main artery (1) and main vein (0)
# are separate components; the role label is not evidence of a connection.
_BRANCHES = (
    ('SFJ', 0, 2, (23, 27), None),
    ('CFA_bif', 1, 5, (21, 36), (21, 35)),
    ('CFV_bif', 0, 3, (23, 29), (23, 39)),
)


@dataclass(frozen=True)
class MetricGraphVertex:
    id: int
    knot_id: int
    h_mm: float
    landmarks: tuple[str, ...]


@dataclass(frozen=True)
class MetricGraphEdge:
    id: int
    role: int
    vertex_u: int
    vertex_v: int
    knot_ids: np.ndarray
    s_mm: np.ndarray
    connector_mm: float

    @property
    def length_mm(self) -> float:
        return float(self.s_mm[-1])


@dataclass(frozen=True)
class JunctionEvidence:
    native_nodes: tuple[int, int]
    native_labels: tuple[int, int]
    shared_voxel_faces: int
    vertex_id: int
    branch_root_h_mm: float
    connector_mm: float


@dataclass(frozen=True)
class VascularMetricGraph:
    source_id: str
    h_mm: np.ndarray
    supported: np.ndarray
    links: np.ndarray
    sample_knots: np.ndarray
    knot_xyz_mm: np.ndarray
    vertices: tuple[MetricGraphVertex, ...]
    edges: tuple[MetricGraphEdge, ...]
    junctions: dict[str, JunctionEvidence]
    unlocated_junctions: dict[str, str]
    _station_segments: dict[tuple[int, int], tuple[int, int]]
    domain: str | None
    _source: object = field(repr=False, compare=False)
    _native_graph: Mapping = field(repr=False, compare=False)

    @classmethod
    def from_source(cls, source, native_graph: Mapping, *, domain: str | None = None) -> VascularMetricGraph:
        """Embed verified native topology in a source's available traces.

        Native RAS coordinates are not neutral-world coordinates. A verified
        contact authorizes a *modeled* attachment of a supported proximal
        branch root to a nearby trunk knot. The short connector is part of
        the metric graph and is audited but never supplies station support.
        Source gaps are preserved, and dubious/missing registrations are
        recorded rather than creating a branch or connecting across a gap.
        Hunter and popliteal splits are approximate named landmarks only.
        """
        h = np.asarray(source.h_mm, dtype=float)
        xyz = np.asarray(source.centers_mm, dtype=float)
        supported = np.asarray(source.supported, dtype=bool)
        if h.ndim != 1 or not h.size or xyz.shape != (h.size, 6, 3) or supported.shape != xyz.shape[:2]:
            raise ValueError('source requires h_mm (N,), centers_mm (N,6,3), supported (N,6)')
        dh = np.diff(h)
        if not np.isfinite(h).all() or not (np.all(dh > 0) or np.all(dh < 0)):
            raise ValueError('source station heights must be finite and strictly monotone')
        adjacent_support = supported[1:] & supported[:-1]
        links = np.asarray(getattr(source, 'links', adjacent_support), dtype=bool)
        if links.shape != adjacent_support.shape:
            raise ValueError('source links must have shape (N-1,6)')
        if domain is not None:
            lo, hi = _DOMAINS[domain]
            indices = np.flatnonzero((h >= lo) & (h <= hi))
            if not len(indices):
                raise ValueError(f'source has no stations in {domain}')
            a, b = indices[0], indices[-1]
            h, xyz, supported, links = h[a:b + 1], xyz[a:b + 1], supported[a:b + 1], links[a:b]
            dh = np.diff(h)
        if not supported.any() or not np.isfinite(xyz[supported]).all():
            raise ValueError('source must have at least one supported finite center and no invalid supported centers')
        rows, roles = np.nonzero(supported)
        knots = xyz[rows, roles]
        if not np.allclose(knots[:, 1], h[rows], rtol=0., atol=1e-6):
            raise ValueError('source center Y must equal its nominal station h_mm')
        knots[:, 1] = h[rows]
        links = links & supported[1:] & supported[:-1]
        sample_knots = np.full(supported.shape, -1, dtype=np.intp)
        sample_knots[rows, roles] = np.arange(len(knots))
        proximal_first = h[0] >= h[-1]
        runs: list[tuple[int, np.ndarray]] = []
        for role in range(6):
            stations = np.flatnonzero(supported[:, role])
            if not proximal_first:
                stations = stations[::-1]
            boundaries = np.flatnonzero(
                (np.abs(np.diff(stations)) != 1)
                | ~links[np.minimum(stations[1:], stations[:-1]), role]
            ) + 1
            runs.extend((role, run) for run in np.split(stations, boundaries) if run.size)

        contacts: dict[tuple[int, int], Mapping] = {}
        for entry in native_graph.get('verified_junction_edges', ()):
            key = tuple(sorted(entry['labels']))
            if key not in contacts or entry['shared_voxel_faces'] > contacts[key]['shared_voxel_faces']:
                contacts[key] = entry
        junction_knots = {}
        unlocated = {}
        attachments: dict[int, int] = {}
        welds: dict[int, int] = {}
        for name, trunk, branch, types, native_trunk in _BRANCHES:
            if types not in contacts or (native_trunk is not None and native_trunk not in contacts):
                unlocated[name] = 'native_contact_unverified'
                continue
            branch_run = next((run for role, run in runs if role == branch), None)
            if branch_run is None:
                unlocated[name] = 'source_branch_unsupported'
                continue
            root = branch_run[0]
            # Register to a nearby trunk knot with source links on both
            # sides: never infer a bifurcation at a gap or chart boundary.
            through = np.zeros(h.size, dtype=bool)
            through[1:-1] = links[:-1, trunk] & links[1:, trunk]
            local = np.flatnonzero(through & (np.abs(h - h[root]) <= 15.))
            if not local.size:
                unlocated[name] = 'no_connected_local_trunk'
                continue
            root_knot = int(sample_knots[root, branch])
            candidates = sample_knots[local, trunk]
            candidates = candidates[~np.isin(candidates, [item[0] for item in junction_knots.values()])]
            if not candidates.size:
                unlocated[name] = 'local_trunk_knots_already_assigned'
                continue
            distance = np.linalg.norm(knots[candidates] - knots[root_knot], axis=1)
            nearest = int(np.argmin(distance))
            node = int(candidates[nearest])
            connector = float(distance[nearest])
            if connector > 30.:
                unlocated[name] = 'registration_connector_exceeds_30_mm'
                continue
            if connector == 0.:
                welds[root_knot] = node
            else:
                attachments[root_knot] = node
            junction_knots[name] = (node, contacts[types], float(h[root]), connector)

        if welds:
            # Remove the redundant DOF; no zero-length element or duplicate
            # vertex remains, even for coincident native traces.
            remap = np.cumsum(~np.isin(np.arange(len(knots)), list(welds))) - 1
            for dropped, kept in welds.items():
                remap[dropped] = remap[kept]
            sample_knots[supported] = remap[sample_knots[supported]]
            knots = np.delete(knots, list(welds), axis=0)
            attachments = {int(remap[k]): int(remap[v]) for k, v in attachments.items()}
            junction_knots = {k: (int(remap[v[0]]), *v[1:]) for k, v in junction_knots.items()}

        named: dict[int, list[str]] = {}
        for name, (knot, _, _, _) in junction_knots.items():
            named.setdefault(knot, []).append(name)
        tolerance = float(np.median(np.abs(dh))) / 2. if dh.size else 0.
        for name, station in (('Hunter', -780.), ('popliteal', -850.)):
            i = int(np.argmin(np.abs(h - station)))
            if abs(h[i] - station) <= tolerance:
                for role, suffix in ((0, 'vein'), (1, 'artery')):
                    knot = int(sample_knots[i, role])
                    if knot >= 0:
                        named.setdefault(knot, []).append(f'{name}_{suffix}')

        edge_inputs = []
        vertex_knots = set(named)
        for role, stations in runs:
            trace = sample_knots[stations, role]
            attached = attachments.get(int(trace[0]))
            if attached is not None:
                trace = np.r_[attached, trace]
            split = np.r_[0, [i for i in range(1, len(trace) - 1) if trace[i] in named], len(trace) - 1].astype(int)
            vertex_knots.update(trace[split])
            for start, end in zip(split[:-1], split[1:], strict=True):
                if end > start:
                    edge_inputs.append((role, trace[start:end + 1], attached is not None and start == 0))
        vertices = tuple(
            MetricGraphVertex(i, int(k), float(knots[k, 1]), tuple(named.get(k, ())))
            for i, k in enumerate(sorted(vertex_knots))
        )
        vertex_id = {v.knot_id: v.id for v in vertices}
        edges = []
        station_segments = {}
        for i, (role, trace, attached) in enumerate(edge_inputs):
            ds = np.linalg.norm(np.diff(knots[trace], axis=0), axis=1)
            edge = MetricGraphEdge(i, role, vertex_id[trace[0]], vertex_id[trace[-1]], trace,
                                   np.r_[0., np.cumsum(ds)], float(ds[0]) if attached else 0.)
            edges.append(edge)
            for j in range(int(attached), len(trace) - 1):
                station_segments[int(trace[j]), int(trace[j + 1])] = (i, j)
        junctions = {
            k: JunctionEvidence(tuple(entry['nodes']), tuple(entry['labels']),
                                int(entry['shared_voxel_faces']), vertex_id[knot], root_h, connector)
            for k, (knot, entry, root_h, connector) in junction_knots.items()
        }
        return cls(getattr(source, 'source_id', ''), h.copy(), supported.copy(), links,
                   sample_knots, knots, vertices, tuple(edges), junctions, unlocated, station_segments,
                   domain, source, native_graph)

    @cached_property
    def element_knots(self) -> np.ndarray:
        """Endpoint knot pairs (K,2) for all FE elements, including connectors."""
        if not self.edges:
            return np.empty((0, 2), dtype=np.intp)
        return np.concatenate([np.column_stack((e.knot_ids[:-1], e.knot_ids[1:])) for e in self.edges])

    @cached_property
    def stiffness(self) -> sparse.csc_matrix:
        i, j = self.element_knots.T
        weight = 1. / np.linalg.norm(self.knot_xyz_mm[i] - self.knot_xyz_mm[j], axis=1)
        n = len(self.knot_xyz_mm)
        return sparse.csc_matrix((np.r_[weight, -weight, -weight, weight],
                                  (np.r_[i, i, j, j], np.r_[i, j, i, j])), shape=(n, n))

    @cached_property
    def lumped_mass_mm(self) -> np.ndarray:
        i, j = self.element_knots.T
        half = 0.5 * np.linalg.norm(self.knot_xyz_mm[i] - self.knot_xyz_mm[j], axis=1)
        n = len(self.knot_xyz_mm)
        mass = np.bincount(i, half, minlength=n) + np.bincount(j, half, minlength=n)
        # A supported singleton is an independent point, not an invented
        # edge. Give it a reference mass of one station interval.
        mass[mass == 0.] = np.median(np.abs(np.diff(self.h_mm))) if self.h_mm.size > 1 else 5.
        return mass

    @cached_property
    def _default_relative_fe(self) -> _RelativeFE:
        return _RelativeFE.build(self, _DEFAULT_KAPPA)

    @cached_property
    def _ordered_stations(self) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(self.h_mm)
        return self.h_mm[order], order

    def _station_query(self, h_mm: float, role: int) -> tuple[int, int, float] | None:
        """Return bracketing knot ids and distal weight, or None at gaps."""
        h, order = self._ordered_stations
        if not 0 <= role < self.supported.shape[1] or not np.isfinite(h_mm):
            return None
        i = int(np.searchsorted(h, h_mm))
        if i < len(h) and h[i] == h_mm:
            knot = int(self.sample_knots[order[i], role])
            return (knot, knot, 0.) if knot >= 0 else None
        if i == 0 or i == len(h) or not self.links[min(order[i], order[i - 1]), role]:
            return None
        a, b = self.sample_knots[order[[i, i - 1]], role]
        return int(a), int(b), float((h[i] - h_mm) / (h[i] - h[i - 1]))


@dataclass(frozen=True)
class DifficultyProfile:
    name: str
    sigma_shared_mm: float | tuple[float, float, float]
    sigma_relative_mm: float | tuple[float, float, float]
    sigma_landmark_mm: float
    fixed_shift_mm: float | None = None
    shape_std: float = 0.


ORDINARY = DifficultyProfile('ordinary', 20., 4.5, 18.)
STRESS = DifficultyProfile('stress', 38., 7., 30.)
ORDINARY_MATERIAL = DifficultyProfile('ordinary', (20., 14., .45), (4.5, 4., .25), 18.)
STRESS_MATERIAL = DifficultyProfile('stress', (36., 24., .70), (7., 6., .35), 30.)


def shift_sweep(offset_mm: float) -> DifficultyProfile:
    """A point-mass intervention: fixed +X shift with signed axial shear.

    Uses the stress precision only as the reference inference prior; a
    deterministic sweep is not a matched draw from that Gaussian.
    """
    if not np.isfinite(offset_mm) or offset_mm < 0.:
        raise ValueError('sweep offset must be finite and nonnegative')
    return DifficultyProfile('shift_sweep', STRESS.sigma_shared_mm, STRESS.sigma_relative_mm,
                             STRESS.sigma_landmark_mm, float(offset_mm))


def _profile(spec: str | DifficultyProfile, coordinate_mode: str) -> DifficultyProfile:
    if isinstance(spec, DifficultyProfile):
        return spec
    profiles = (ORDINARY_MATERIAL, STRESS_MATERIAL) if coordinate_mode == 'material' else (ORDINARY, STRESS)
    return {p.name: p for p in profiles}[spec]


@dataclass(frozen=True)
class StateLayout:
    shared: slice
    relative: slice
    landmarks: slice
    shared_shape: tuple[int, int, int]  # knot, (value, derivative/lambda), XYZ
    relative_shape: tuple[int, int]  # FE knot, XYZ
    size: int


@dataclass(frozen=True)
class LinearQuery:
    """Affine query in the prior's coordinates: nominal + operator @ latent."""

    nominal: np.ndarray
    operator: sparse.csr_matrix


def _tree_variances(operator: sparse.csc_matrix, mass: np.ndarray) -> np.ndarray:
    """Diagonal of A^-1 D A^-1 by two sweeps of Gaussian tree elimination.

    In a cavity message, p is the Schur pivot and b is the variance of the
    effective random right-hand side before division by p. Contributions
    from distinct subtrees are independent. The isolated case has p=A_ii.
    """
    n = len(mass)
    diag = operator.diagonal()
    parent = np.full(n, -2, dtype=np.intp)
    coupling_sq = np.zeros(n)
    schur = diag.copy()
    rhs_var = mass.copy()
    traversal = []
    for root in range(n):
        if parent[root] != -2:
            continue
        parent[root] = -1
        stack = [root]
        while stack:
            j = stack.pop()
            traversal.append(j)
            for pos in range(operator.indptr[j], operator.indptr[j + 1]):
                i = operator.indices[pos]
                if i == j or i == parent[j]:
                    continue
                if parent[i] != -2:
                    raise ValueError('expected an acyclic arterial/venous FE graph')
                parent[i] = j
                coupling_sq[i] = operator.data[pos] ** 2
                stack.append(i)
    down_p = np.zeros(n)
    down_b = np.zeros(n)
    for i in reversed(traversal):
        j = parent[i]
        if j >= 0:
            down_p[i] = coupling_sq[i] / schur[i]
            down_b[i] = down_p[i] * rhs_var[i] / schur[i]
            schur[j] -= down_p[i]
            rhs_var[j] += down_b[i]
    for i in traversal:
        j = parent[i]
        if j >= 0:
            cavity_p = schur[j] + down_p[i]
            cavity_b = rhs_var[j] - down_b[i]
            correction = coupling_sq[i] / cavity_p
            schur[i] -= correction
            rhs_var[i] += correction * cavity_b / cavity_p
    return rhs_var / schur**2


@dataclass(frozen=True)
class _RelativeFE:
    operator: sparse.csc_matrix
    mass_sqrt: np.ndarray
    mean_variance: float
    solve: Callable

    @classmethod
    def build(cls, graph: VascularMetricGraph, kappa_per_mm: float) -> _RelativeFE:
        mass = graph.lumped_mass_mm
        operator = (graph.stiffness + sparse.diags(kappa_per_mm**2 * mass, format='csc')).tocsc()
        mean_variance = float(np.mean(_tree_variances(operator, mass)))
        return cls(operator, np.sqrt(mass), mean_variance, splu(operator).solve)


class MetricGraphPrior:
    """Sparse CSC precision and draws for the *unconditioned* latent prior.

    Field channels are either XYZ millimetres or arc mm, axial mm, and logit-depth.
    ``mean`` is the inference prior mean (zero); ``sample_latent`` samples
    that Gaussian for every profile. Use ``generation_latent`` to apply the
    deterministic sweep intervention when configured.
    """

    def __init__(self, graph: VascularMetricGraph, difficulty: str | DifficultyProfile = 'ordinary',
                 *, coordinate_mode: str = 'material', domain: str = 'thigh_nysora',
                 chart: RightLegMaterialChart | None = None, shared_length_mm: float = 180.,
                 kappa_per_mm: float = _DEFAULT_KAPPA):
        if coordinate_mode not in _COORDINATE_MODES or domain not in _DOMAINS:
            raise ValueError('unknown coordinate mode or graph domain')
        if graph.domain != domain:
            graph = VascularMetricGraph.from_source(graph._source, graph._native_graph, domain=domain)
        self.graph = graph
        self.domain, self.coordinate_mode = domain, coordinate_mode
        self.chart = (RightLegMaterialChart() if chart is None else chart) if coordinate_mode == 'material' else None
        self.profile = p = _profile(difficulty, coordinate_mode)
        self._shared_sigma = np.broadcast_to(np.asarray(p.sigma_shared_mm, dtype=float), (3,))
        self._relative_sigma = np.broadcast_to(np.asarray(p.sigma_relative_mm, dtype=float), (3,))
        scales = (*self._shared_sigma, *self._relative_sigma, p.sigma_landmark_mm, shared_length_mm, kappa_per_mm)
        if not all(np.isfinite(s) and s > 0. for s in scales):
            raise ValueError('GP scales must be finite and positive')
        self.shared_h_mm = np.sort(graph.h_mm)
        self.lambda_shared_per_mm = np.sqrt(3.) / shared_length_mm
        t = self.lambda_shared_per_mm * np.diff(self.shared_h_mm)
        ones = np.ones_like(t)
        self._transition = np.exp(-t)[:, None, None] * np.stack(
            (np.column_stack((ones + t, t)), np.column_stack((-t, ones - t))), axis=1)
        # In normalized derivative units P_inf is the identity before per-channel scaling.
        exp2 = np.exp(-2. * t)
        q00 = gammainc(3., 2. * t)
        q01 = 2. * t**2 * exp2
        covariance = np.stack((np.column_stack((q00, q01)), np.column_stack((q01, q00 + 4. * t * exp2))), axis=1)
        self._innovation_factor = np.linalg.cholesky(covariance)
        self._fe = graph._default_relative_fe if kappa_per_mm == _DEFAULT_KAPPA else _RelativeFE.build(graph, kappa_per_mm)
        self._relative_unit_scale = 1. / np.sqrt(self._fe.mean_variance)
        m, n, v = len(self.shared_h_mm), len(graph.knot_xyz_mm), len(graph.vertices)
        shared_end = 6 * m
        relative_end = shared_end + 3 * n
        self.layout = StateLayout(slice(0, shared_end), slice(shared_end, relative_end),
                                  slice(relative_end, relative_end + v), (m, 2, 3), (n, 3), relative_end + v)

    @cached_property
    def mean(self) -> np.ndarray:
        return np.zeros(self.layout.size)

    @cached_property
    def precision(self) -> sparse.csc_matrix:
        """The exact block sparse precision of ``sample_latent``."""
        p = self.profile
        m = len(self.shared_h_mm)
        blocks = [np.eye(2)]
        for f, chol in zip(self._transition, self._innovation_factor, strict=True):
            factor = np.linalg.solve(chol, np.column_stack((-f, np.eye(2))))
            blocks.append(factor.T @ factor)
        pair = np.arange(2 * m - 2).reshape(-1, 2)
        pair = np.column_stack((pair, pair + 2))
        i = np.r_[0, 1, 0, 1, np.repeat(pair, 4, axis=1).ravel()]
        j = np.r_[0, 0, 1, 1, np.tile(pair, (1, 4)).ravel()]
        shared = sparse.csc_matrix((np.r_[blocks[0].ravel(), np.asarray(blocks[1:]).ravel()], (i, j)),
                                   shape=(2 * m, 2 * m))
        a = self._fe.operator
        relative = (a @ sparse.diags(1. / (self._relative_unit_scale * self._fe.mass_sqrt)**2) @ a).tocsc()
        return sparse.block_diag((sparse.kron(shared, sparse.diags(self._shared_sigma**-2), format='csc'),
                                  sparse.kron(relative, sparse.diags(self._relative_sigma**-2), format='csc'),
                                  sparse.eye(len(self.graph.vertices)) / p.sigma_landmark_mm**2), format='csc')

    def sample_latent(self, rng: np.random.Generator, size: int | None = None) -> np.ndarray:
        """Generate (D,) or (B,D) raw draws via exact transitions and sparse solves."""
        b = 1 if size is None else size
        if b < 0:
            raise ValueError('draw count must be nonnegative')
        layout = self.layout
        if b == 0:
            return np.empty((0, layout.size))
        latent = np.empty((b, layout.size))
        state = latent[:, layout.shared].reshape(b, *layout.shared_shape)
        state[:, 0] = self._shared_sigma * rng.standard_normal((b, 2, 3))
        innovations = rng.standard_normal((len(self.shared_h_mm) - 1, b, 2, 3)) * self._shared_sigma
        for i, (f, chol) in enumerate(zip(self._transition, self._innovation_factor, strict=True)):
            state[:, i + 1] = f @ state[:, i] + chol @ innovations[i]
        n = layout.relative_shape[0]
        noise = rng.standard_normal((b * 3, n)).T
        noise *= (self._relative_unit_scale * self._fe.mass_sqrt)[:, None]
        field = self._fe.solve(noise).reshape(n, b, 3).transpose(1, 0, 2)
        latent[:, layout.relative].reshape(b, n, 3)[...] = field * self._relative_sigma
        latent[:, layout.landmarks] = self.profile.sigma_landmark_mm * rng.standard_normal((b, len(self.graph.vertices)))
        return latent[0] if size is None else latent

    def generation_latent(self, rng: np.random.Generator, size: int | None = None) -> np.ndarray:
        """Raw matched draw, or the explicitly deterministic sweep intervention."""
        offset = self.profile.fixed_shift_mm
        if offset is None:
            return self.sample_latent(rng, size)
        latent = np.zeros(self.layout.size)
        latent[self.layout.shared][::6] = offset
        h = self.shared_h_mm
        if self.coordinate_mode == 'cartesian_xyz' and h[-1] > h[0]:
            landmark_h = np.array([v.h_mm for v in self.graph.vertices])
            latent[self.layout.landmarks] = .2 * offset * (2. * (landmark_h - h[0]) / (h[-1] - h[0]) - 1.)
        return latent if size is None else np.broadcast_to(latent, (size, latent.size)).copy()

    @cached_property
    def knot_operator(self) -> sparse.csr_matrix:
        """Stacked displacement maps, in ``graph.knot_xyz_mm`` order."""
        graph, layout = self.graph, self.layout
        n = len(graph.knot_xyz_mm)
        base = np.arange(n * 3)
        station = np.searchsorted(self.shared_h_mm, graph.knot_xyz_mm[:, 1])
        shared = (6 * station[:, None] + np.arange(3)).ravel()
        rows = [base, base]
        cols = [shared, layout.relative.start + base]
        data = [np.ones_like(base, dtype=float)] * 2
        for edge in graph.edges:
            interior = edge.knot_ids[1:-1]
            if interior.size:
                tau = edge.s_mm[1:-1] / edge.length_mm
                rows.extend((3 * interior + 1, 3 * interior + 1))
                cols.extend((np.full(interior.size, layout.landmarks.start + edge.vertex_u),
                             np.full(interior.size, layout.landmarks.start + edge.vertex_v)))
                data.extend((1. - tau, tau))
        rows.append(np.array([3 * v.knot_id + 1 for v in graph.vertices], dtype=np.intp))
        cols.append(layout.landmarks.start + np.arange(len(graph.vertices)))
        data.append(np.ones(len(graph.vertices)))
        design = sparse.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                                   shape=(3 * n, layout.size))
        if self.chart is not None:
            scale = np.ones((n, 3))
            scale[:, 0] = 1. / self.chart.nominal_reference_radius(graph.knot_xyz_mm[:, 1])
            design.data *= np.repeat(scale.ravel(), np.diff(design.indptr))
        return design

    @cached_property
    def _material_reference(self):
        return None if self.chart is None else self.chart.encode_with_audit(self.graph.knot_xyz_mm)

    @property
    def nominal_projection_audit(self) -> dict | None:
        return None if self._material_reference is None else self._material_reference[2]

    @property
    def nominal_knots(self) -> np.ndarray:
        """Per-knot nominal coordinates (XYZ mm or theta/h/eta)."""
        return self.graph.knot_xyz_mm if self._material_reference is None else self._material_reference[0]

    def prior_precision_csc(self) -> sparse.csc_matrix:
        return self.precision

    def knot_design_matrix(self) -> sparse.csr_matrix:
        """Map the flat latent to stacked per-knot coordinate displacements."""
        return self.knot_operator

    def kirchhoff_laplacian(self) -> sparse.csc_matrix:
        """Generalized Kirchhoff eigenproblem: K ψ = λ diag(graph.lumped_mass_mm) ψ."""
        return self.graph.stiffness

    def _nominal_delta(self, a: int, b: int) -> np.ndarray:
        delta = self.nominal_knots[b] - self.nominal_knots[a]
        if self.chart is not None:
            delta[0] = (delta[0] + np.pi) % (2. * np.pi) - np.pi
        return delta

    def _interpolate_knots(self, a: int, b: int, t: float) -> LinearQuery:
        xyz = self.nominal_knots
        op = self.knot_operator
        if t == 0. or a == b:
            return LinearQuery(xyz[a].copy(), op[3 * a:3 * a + 3])
        if t == 1.:
            return LinearQuery(xyz[b].copy(), op[3 * b:3 * b + 3])
        return LinearQuery(xyz[a] + t * self._nominal_delta(a, b),
                           (1. - t) * op[3 * a:3 * a + 3] + t * op[3 * b:3 * b + 3])

    def center_query(self, h_mm: float, role: int) -> LinearQuery | None:
        """Nominal-station affine center map; no extrapolation or gap filling."""
        bracket = self.graph._station_query(h_mm, role)
        return None if bracket is None else self._interpolate_knots(*bracket)

    def _edge_segment(self, edge_id: int, tau: float) -> tuple[int, int, float, float]:
        if not 0. <= tau <= 1.:
            raise ValueError('intrinsic edge progress must be in [0,1]')
        edge = self.graph.edges[edge_id]
        s = tau * edge.length_mm
        i = min(int(np.searchsorted(edge.s_mm, s, side='right')) - 1, len(edge.s_mm) - 2)
        ds = edge.s_mm[i + 1] - edge.s_mm[i]
        a, b = edge.knot_ids[i:i + 2]
        return int(a), int(b), float((s - edge.s_mm[i]) / ds), float(ds)

    def edge_query(self, edge_id: int, tau: float) -> LinearQuery:
        """Affine center map at intrinsic arc progress, including connectors."""
        a, b, t, _ = self._edge_segment(edge_id, tau)
        return self._interpolate_knots(a, b, t)

    def edge_tangent_query(self, edge_id: int, tau: float) -> LinearQuery:
        """Unnormalized tangent dc/ds; right derivative at interior knots."""
        a, b, _, ds = self._edge_segment(edge_id, tau)
        op = self.knot_operator
        return LinearQuery(self._nominal_delta(a, b) / ds, (op[3 * b:3 * b + 3] - op[3 * a:3 * a + 3]) / ds)

    def station_edge(self, h_mm: float, role: int) -> tuple[int, float] | None:
        """Return (edge id, intrinsic progress) for supported non-singletons."""
        bracket = self.graph._station_query(h_mm, role)
        if bracket is None:
            return None
        a, b, t = bracket
        if a != b:
            edge_id, i = self.graph._station_segments[a, b]
        else:
            # Prefer the distal side of a split, otherwise use its proximal
            # side. Incidence can include a synthetic connector on another role.
            h, order = self.graph._ordered_stations
            k = int(np.searchsorted(h, h_mm))
            for neighbor, ta in ((k - 1, 0.), (k + 1, 1.)):
                if 0 <= neighbor < len(h):
                    other = int(self.graph.sample_knots[order[neighbor], role])
                    pair = (a, other) if ta == 0. else (other, a)
                    segment = self.graph._station_segments.get(pair)
                    if segment is not None:
                        edge_id, i = segment
                        t = ta
                        break
            else:
                return None
        edge = self.graph.edges[edge_id]
        return edge_id, float(((1. - t) * edge.s_mm[i] + t * edge.s_mm[i + 1]) / edge.length_mm)

    def _linear_knots(self, x: np.ndarray) -> np.ndarray:
        state = np.asarray(x)
        if state.ndim not in (1, 2) or state.shape[-1] != self.layout.size:
            raise ValueError('latent must have shape (D,) or (B,D) for this prior')
        nominal = self.nominal_knots
        return (self.knot_operator @ state.T).T.reshape(*state.shape[:-1], *nominal.shape) + nominal

    @staticmethod
    def _transport_height(coordinates, chart):
        """C2 height-tail map with the right chart's proximal pelvic exclusion."""
        theta, h, _ = np.moveaxis(coordinates, -1, 0)
        low, proximal_onset = FULL_LEG_Y_BOUNDS_MM
        below = h < low
        span = chart.h_bounds_mm[0] - low
        extension = np.tanh((h[below] - low) / span)
        h[below] = low + span * extension
        dh = np.ones_like(h)
        dh[below] = 1. - extension**2
        limit, d_limit = chart.proximal_limit_mm(theta)
        above = h > limit
        span = chart.h_bounds_mm[1] - proximal_onset
        extension = np.tanh((h[above] - limit[above]) / span)
        h[above] = limit[above] + span * extension
        dh[above] = 1. - extension**2
        dtheta = np.zeros_like(h)
        dtheta[above] = extension**2 * d_limit[above]
        return dh, dtheta

    def material_knots(self, x: np.ndarray, chart: RightLegMaterialChart | None = None) -> np.ndarray:
        if self.chart is None:
            raise ValueError('material addresses require coordinate_mode=material')
        coordinates = self._linear_knots(x)
        self._transport_height(coordinates, self.chart if chart is None else chart)
        return coordinates

    def decode_knots_to_world(self, x: np.ndarray, chart: RightLegMaterialChart | None = None) -> np.ndarray:
        if self.chart is None:
            return self._linear_knots(x)
        chart = self.chart if chart is None else chart
        return chart.tilde_xi_to_world(self.material_knots(x, chart))

    def dworld_dx(self, x: np.ndarray, chart: RightLegMaterialChart | None = None) -> sparse.csr_matrix:
        """Sparse stacked Cartesian derivative at one latent vector."""
        if np.ndim(x) != 1:
            raise ValueError('expected one latent vector')
        if self.chart is None:
            return self.knot_operator
        chart = self.chart if chart is None else chart
        coordinates = self._linear_knots(x)
        dh, dtheta = self._transport_height(coordinates, chart)
        jac = chart.dworld_dtilde_xi(coordinates)
        jac[:, :, 0] += jac[:, :, 1] * dtheta[:, None]
        jac[:, :, 1] *= dh[:, None]
        n = len(jac)
        transform = sparse.bsr_matrix((jac, np.arange(n), np.arange(n + 1)), shape=(3 * n, 3 * n))
        return (transform @ self.knot_operator).tocsr()

    def element_interiors(self, x, knots_mm, elements, step_mm):
        """Sample the actual material interpolant; respect nominal angular lifts."""
        start, stop = elements.T
        coords = self._linear_knots(x)
        left = coords[start]
        delta = coords[stop] - left
        gap = self.nominal_knots[stop, 0] - self.nominal_knots[start, 0]
        delta[:, 0] -= (2. * np.pi) * np.floor((gap + np.pi) / (2. * np.pi))
        # Sample the annular arc, then densify the actual world polyline,
        # including any chords across mesh-visibility switches.
        chords = np.linalg.norm(knots_mm[stop] - knots_mm[start], axis=1)
        arc = abs(delta[:, 0]) * self.chart.nominal_reference_radius(self.nominal_knots[start, 1])
        subdivisions = np.maximum(2, np.ceil(np.maximum(chords, 1.5 * arc) / step_mm).astype(int))
        if not len(subdivisions):
            return np.empty((0, 3)), np.zeros(1, dtype=np.intp)
        offsets = np.r_[0, np.cumsum(subdivisions - 1)]
        ids = np.repeat(np.arange(len(elements)), subdivisions - 1)
        u = (np.arange(len(ids)) - offsets[ids] + 1.) / subdivisions[ids]
        addresses = left[ids] + u[:, None] * delta[ids]
        self._transport_height(addresses, self.chart)
        inner = self.chart.tilde_xi_to_world(addresses)
        full_offsets = offsets + 2 * np.arange(len(elements) + 1)
        vertices = np.empty((len(inner) + 2 * len(elements), 3))
        mask = np.ones(len(vertices), dtype=bool)
        mask[full_offsets[:-1]] = mask[full_offsets[1:] - 1] = False
        vertices[full_offsets[:-1]], vertices[full_offsets[1:] - 1] = knots_mm[start], knots_mm[stop]
        vertices[mask] = inner
        gaps = np.diff(vertices, axis=0)
        counts = np.maximum(1, np.ceil(np.linalg.norm(gaps, axis=1) / step_mm).astype(int))
        counts[full_offsets[1:-1] - 1] = 0
        dense_offsets = np.r_[0, np.cumsum(counts)]
        edge_offsets = dense_offsets[full_offsets[:-1]]
        rows = np.repeat(np.arange(len(gaps)), counts)
        fraction = (np.arange(len(rows)) - dense_offsets[rows]) / counts[rows]
        interior = np.ones(len(rows), dtype=bool)
        interior[edge_offsets] = False
        rows, fraction = rows[interior], fraction[interior]
        return vertices[rows] + fraction[:, None] * gaps[rows], np.r_[edge_offsets - np.arange(len(elements)), len(rows)]


@njit
def _slice_parity(points, triangles, starts, indices, bin_base, bin_width):
    """Cross parity on horizontal triangle slices, with half-open crossings."""
    inside = np.zeros(len(points), dtype=np.bool_)
    for i in range(len(points)):
        x, y, z = points[i]
        bucket = int(np.floor((y - bin_base) / bin_width))
        if bucket < 0 or bucket + 1 >= len(starts):
            continue
        odd = False
        for q in range(starts[bucket], starts[bucket + 1]):
            tri = triangles[indices[q]]
            count = 0
            x0, z0, x1, z1 = 0., 0., 0., 0.
            for edge in range(3):
                a = tri[edge]
                b = tri[(edge + 1) % 3]
                if (a[1] > y) != (b[1] > y):
                    t = (y - a[1]) / (b[1] - a[1])
                    xx = a[0] + t * (b[0] - a[0])
                    zz = a[2] + t * (b[2] - a[2])
                    if count == 0:
                        x0, z0 = xx, zz
                    else:
                        x1, z1 = xx, zz
                    count += 1
            if count == 2 and (z0 > z) != (z1 > z):
                crossing = x0 + (z - z0) / (z1 - z0) * (x1 - x0)
                if crossing > x:
                    odd = not odd
        inside[i] = odd
    return inside


class _TrainingValidity:
    """Closed horizontal sections checked against raw skin triangles;
    nearest-voxel exclusion by a population bone raster, not subject bone.
    """

    def __init__(self, occupancy, skin_path: Path = NEUTRAL_SMPLX_SURFACE_PATH, bone_min_fraction: float = .55):
        if not 0. < bone_min_fraction <= 1.:
            raise ValueError('bone exclusion fraction must be in (0,1]')
        self.bone_count = None
        if occupancy is not None:
            get = occupancy.__getitem__ if isinstance(occupancy, Mapping) else lambda key: getattr(occupancy, key)
            self.bone_count = np.asarray(get('count'))
            self.threshold = float(get('cohort_size')) * bone_min_fraction
            self.axes = tuple(np.asarray(get(f'{axis}_mm'), dtype=float) for axis in 'xyz')
            if self.bone_count.shape != tuple(v.size for v in self.axes) or any(
                v.ndim != 1 or v.size < 2 or not np.isfinite(v).all() or not (np.diff(v) > 0.).all()
                for v in self.axes
            ) or not np.isfinite(self.threshold) or self.threshold <= 0.:
                raise ValueError('invalid bone occupancy axes, shape or cohort')
            self.cell_edges = tuple(np.r_[v[0] - .5 * (v[1] - v[0]), (v[1:] + v[:-1]) / 2.,
                                          v[-1] + .5 * (v[-1] - v[-2])] for v in self.axes)
        path = Path(skin_path)
        skin_bytes = path.read_bytes()
        canonical_path = path.resolve() == NEUTRAL_SMPLX_SURFACE_PATH.resolve()
        if canonical_path:
            verify_neutral_smplx_surface_sha256(skin_bytes)
        with np.load(BytesIO(skin_bytes), allow_pickle=False) as asset:
            canonical = 'triangles' in asset
            if canonical and not canonical_path:
                verify_neutral_smplx_surface_sha256(skin_bytes)
            vertices = asset['vertices'].astype(float) * (1. if canonical else 1000.)
            faces = asset['triangles' if canonical else 'faces']
        self.triangles = vertices[faces]
        y = vertices[:, 1]
        self.min_y, self.max_y = float(y.min()), float(y.max())
        mesh_edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
        edges, multiplicity = np.unique(mesh_edges, return_counts=True, axis=0)
        # A cut/non-manifold boundary at this Y makes the entire section
        # unknown; do not close holes or treat an open surface as a volume.
        boundary = np.sort(y[edges[multiplicity != 2]], axis=1)
        self.lower_body_boundary_edges = int(np.count_nonzero(boundary[:, 0] < 0.))
        if canonical and (self.lower_body_boundary_edges or self.min_y >= -1350. or self.max_y <= 0.):
            raise ValueError('canonical neutral skin must have closed horizontal sections across [-1350, 0] mm')
        intervals: list[list[float]] = []
        for lo, hi in boundary[np.argsort(boundary[:, 0])]:
            if intervals and lo <= intervals[-1][1]:
                intervals[-1][1] = max(intervals[-1][1], hi)
            else:
                intervals.append([lo, hi])
        self.boundary = np.asarray(intervals).reshape(-1, 2)
        self.bin_mm = 5.
        buckets = [[] for _ in range(1 + int((self.max_y - self.min_y) / self.bin_mm))]
        tri_y = self.triangles[:, :, 1]
        lo = ((tri_y.min(axis=1) - self.min_y) / self.bin_mm).astype(int)
        hi = ((tri_y.max(axis=1) - self.min_y) / self.bin_mm).astype(int)
        for i, (a, b) in enumerate(zip(lo, hi, strict=True)):
            for j in range(a, b + 1):
                buckets[j].append(i)
        self.starts = np.r_[0, np.cumsum([len(b) for b in buckets])]
        self.indices = np.fromiter((i for b in buckets for i in b), dtype=np.intp)

    def __call__(self, points_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        y = points_mm[:, 1]
        skin_known = (y > self.min_y) & (y < self.max_y)
        if self.boundary.size:
            cut = np.searchsorted(self.boundary[:, 0], y, side='right') - 1
            skin_known &= (cut < 0) | (y > self.boundary[cut, 1])
        inside = np.zeros(len(y), dtype=bool)
        inside[skin_known] = _slice_parity(points_mm[skin_known], self.triangles, self.starts, self.indices,
                                           self.min_y, self.bin_mm)
        bone_known = np.zeros(len(y), dtype=bool)
        bone_hit = np.zeros(len(y), dtype=bool)
        if self.bone_count is not None:
            cell = np.column_stack([np.searchsorted(edge, points_mm[:, dim], side='right') - 1
                                    for dim, edge in enumerate(self.cell_edges)])
            bone_known = ((cell >= 0) & (cell < self.bone_count.shape)).all(axis=1)
            bone_hit[bone_known] = self.bone_count[tuple(cell[bone_known].T)] >= self.threshold
        return inside, skin_known, bone_hit, bone_known


@dataclass(frozen=True)
class GeometryAudit:
    checked_points: int
    checked_segments: int
    max_step_mm: float
    skin_unknown: int
    outside_skin: int
    bone_unknown: int
    in_excluded_bone: int
    axial_reversals: int
    unsupported_material_points: int = 0

    @property
    def passes_known_checks(self) -> bool:
        return self.outside_skin == self.in_excluded_bone == self.unsupported_material_points == 0

    @property
    def all_locations_known(self) -> bool:
        return self.skin_unknown == 0 and self.bone_unknown == 0


@dataclass(frozen=True)
class RejectionCounts:
    """Reasons can overlap; ``draws`` counts each rejected draw once."""

    draws: int
    outside_skin: int
    excluded_bone: int
    unknown_required: int
    unsupported_material: int


@dataclass(frozen=True)
class SampledAnatomy:
    source_id: str
    shape_id: object
    prior: MetricGraphPrior
    latent: np.ndarray
    knot_centers_mm: np.ndarray
    audit: GeometryAudit
    attempts: int
    rejections: RejectionCounts
    sampling_law: str
    shape_beta: np.ndarray
    achieved_euclidean_rms_shift_mm: float
    achieved_euclidean_shift_mm_median_p90: tuple[float, float]

    @property
    def graph(self) -> VascularMetricGraph:
        return self.prior.graph

    @property
    def h_mm(self) -> np.ndarray:
        """Nominal station labels, not the displaced center Y coordinates."""
        return self.graph.h_mm

    @property
    def supported(self) -> np.ndarray:
        return self.graph.supported

    @cached_property
    def centers_mm(self) -> np.ndarray:
        centers = np.full((*self.supported.shape, 3), np.nan)
        centers[self.supported] = self.knot_centers_mm[self.graph.sample_knots[self.supported]]
        return centers

    def edge_centers_mm(self, edge_id: int) -> np.ndarray:
        """Graph knot centers, with bit-identical endpoints on incident edges."""
        return self.knot_centers_mm[self.graph.edges[edge_id].knot_ids]

    def edge_polyline_mm(self, edge_id: int, *, step_mm: float = 1.5) -> np.ndarray:
        """Sample the material centerline that joins this edge's graph knots."""
        knots = self.graph.edges[edge_id].knot_ids
        if self.prior.chart is None:
            return self.knot_centers_mm[knots]
        elements = np.column_stack((knots[:-1], knots[1:]))
        points, offsets = self.prior.element_interiors(self.latent, self.knot_centers_mm, elements, step_mm)
        result = np.empty((len(points) + len(knots), 3))
        junctions = offsets + np.arange(len(knots))
        use = np.ones(len(result), dtype=bool)
        use[junctions] = False
        result[junctions] = self.knot_centers_mm[knots]
        result[use] = points
        return result


@dataclass(frozen=True)
class ExhaustedDraw:
    shape_id: object
    rejections: RejectionCounts
    last_audit: GeometryAudit


class GeometryRejectionError(RuntimeError):
    def __init__(self, source_id: str, accepted: tuple[SampledAnatomy, ...], exhausted: tuple[ExhaustedDraw, ...]):
        self.accepted = accepted
        self.exhausted = exhausted
        self.attempts = sum(a.attempts for a in accepted) + sum(e.rejections.draws for e in exhausted)
        first = exhausted[0]
        super().__init__(f'{source_id}: {len(exhausted)} shapes exhausted rejection, {len(accepted)} accepted '
                         f'in {self.attempts} total draws; first exhausted shape={first.shape_id}, {first.last_audit}')


class MetricGraphAnatomyGenerator:
    """Generate source-specific anatomies; ``shape_id`` identifies the draw.

    By default stochastic profiles reject known exclusions, while deterministic
    sweeps report violations without rejecting the intervention. ``raw`` keeps
    the matched latent Gaussian; ``reject`` conditions it and cannot claim
    zero KL to the raw precision. Material priors log the initial nominal
    registration projection; no proposal is projected or silently resampled.

    The audit checks the deformed centerline polylines at knots and at
    subdivisions no farther apart than the configured spacing. It may miss
    intersections between tested points and does not test vessel walls.
    Bone exclusion uses a thresholded population raster, not subject bone.
    """

    def __init__(self, bank, *, coordinate_mode: str = 'material', domain: str = 'thigh_nysora',
                 skin_path: Path = NEUTRAL_SMPLX_SURFACE_PATH, bone_min_fraction: float = .55,
                 audit_step_mm: float = 1.5, shape_std: float | None = None):
        if coordinate_mode not in _COORDINATE_MODES or domain not in _DOMAINS:
            raise ValueError('unknown coordinate mode or graph domain')
        if shape_std is not None and (not np.isfinite(shape_std) or shape_std < 0.):
            raise ValueError('shape_std must be finite and nonnegative')
        if not np.isfinite(audit_step_mm) or audit_step_mm <= 0.:
            raise ValueError('audit spacing must be finite and positive')
        self.bank = bank
        self.coordinate_mode, self.domain = coordinate_mode, domain
        self.shape_std = shape_std
        self.skin_path = skin_path
        self.bone_min_fraction = bone_min_fraction
        self.audit_step_mm = audit_step_mm
        self._graphs: dict[str, VascularMetricGraph] = {}
        self._priors: dict[tuple[str, DifficultyProfile], MetricGraphPrior] = {}

    def graph(self, source_id: str) -> VascularMetricGraph:
        if source_id not in self._graphs:
            self._graphs[source_id] = VascularMetricGraph.from_source(
                self.bank.source(source_id), self.bank.native_graph, domain=self.domain)
        return self._graphs[source_id]

    def prior(self, source_id: str, difficulty: str | DifficultyProfile = 'ordinary') -> MetricGraphPrior:
        p = _profile(difficulty, self.coordinate_mode)
        # FE factors are reused across profiles with different variance scales.
        key = (source_id, p)
        if key not in self._priors:
            self._priors[key] = MetricGraphPrior(self.graph(source_id), p, coordinate_mode=self.coordinate_mode,
                                                 domain=self.domain)
        return self._priors[key]

    @cached_property
    def _validity(self) -> _TrainingValidity:
        return _TrainingValidity(getattr(self.bank, 'bone_occupancy', None), self.skin_path, self.bone_min_fraction)

    def audit_points(self, graph: VascularMetricGraph, knots_mm: np.ndarray, *,
                     prior: MetricGraphPrior | None = None, latent: np.ndarray | None = None) -> np.ndarray:
        """Export the exact checked points, starting with the unique graph knots.

        Supply both prior and latent to sample a material centerline. Without
        them, the geometry is interpreted as Cartesian straight chords, as for
        raw source centerlines. A Cartesian prior also uses straight chords.
        """
        knots_mm = np.asarray(knots_mm)
        if knots_mm.shape != graph.knot_xyz_mm.shape or not np.isfinite(knots_mm).all():
            raise ValueError('one finite world XYZ center is required per unique graph knot')
        if (prior is None) != (latent is None) or (prior is not None and prior.graph is not graph):
            raise ValueError('prior and latent must be provided together for this graph')
        if prior is not None and prior.chart is not None:
            interiors, _ = prior.element_interiors(latent, knots_mm, graph.element_knots, self.audit_step_mm)
            return np.concatenate((knots_mm, interiors))
        a, b = graph.element_knots.T
        direction = knots_mm[b] - knots_mm[a]
        subdivisions = np.maximum(1, np.ceil(np.linalg.norm(direction, axis=1) / self.audit_step_mm).astype(int))
        counts = subdivisions - 1
        ids = np.repeat(np.arange(len(a)), counts)
        if not ids.size:
            return knots_mm
        offset = np.repeat(np.cumsum(counts) - counts, counts)
        u = (np.arange(ids.size) - offset + 1.) / subdivisions[ids]
        return np.concatenate((knots_mm, knots_mm[a[ids]] + u[:, None] * direction[ids]))

    def classify_points(self, points_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return masks (inside skin, skin known, bone hit, bone known)."""
        points = np.asarray(points_mm)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('expected finite world XYZ points with shape (N,3)')
        result = tuple(np.asarray(m, dtype=bool) for m in self._validity(points))
        if len(result) != 4 or any(mask.shape != (len(points),) for mask in result):
            raise ValueError('validity checker must return four boolean masks with one value per point')
        return result

    def audit_centers(self, graph: VascularMetricGraph, knots_mm: np.ndarray, *,
                      prior: MetricGraphPrior | None = None, latent: np.ndarray | None = None) -> GeometryAudit:
        """Audit point samples; report (but do not reject) axial reversals."""
        a, b = graph.element_knots.T
        points = self.audit_points(graph, knots_mm, prior=prior, latent=latent)
        direction = knots_mm[b] - knots_mm[a]
        unsupported = 0
        if prior is not None and prior.chart is not None:
            unsupported = prior.chart.count_unsupported_world_rays(points)
        inside, skin_known, bone_hit, bone_known = self.classify_points(points)
        return GeometryAudit(len(points), len(a), self.audit_step_mm, int(np.count_nonzero(~skin_known)),
                             int(np.count_nonzero(skin_known & ~inside)), int(np.count_nonzero(~bone_known)),
                             int(np.count_nonzero(bone_known & bone_hit)),
                             int(np.count_nonzero((graph.knot_xyz_mm[b, 1] < graph.knot_xyz_mm[a, 1])
                                                  & (direction[:, 1] >= 0.))), unsupported)

    def sample_batch(self, source_id: str, shape_ids: Iterable[object], rng: np.random.Generator,
                     difficulty: str | DifficultyProfile = 'ordinary', *, geometry_policy: str | None = None,
                     max_attempts: int = 128, require_known: bool = False) -> list[SampledAnatomy]:
        """Draw a batch from one prior, batching sparse solves across pending draws."""
        ids = tuple(shape_ids)
        prior = self.prior(source_id, difficulty)
        deterministic = prior.profile.fixed_shift_mm is not None
        policy = ('raw' if deterministic else 'reject') if geometry_policy is None else geometry_policy
        if policy not in ('raw', 'reject') or max_attempts < 1:
            raise ValueError('geometry_policy must be raw or reject and max_attempts must be positive')
        if require_known and policy != 'reject':
            raise ValueError('require_known requires rejection')
        law = 'deterministic_intervention' if deterministic else (
            'raw_gaussian' if policy == 'raw' else 'geometry_conditioned_gaussian')
        shape_std = prior.profile.shape_std if self.shape_std is None else self.shape_std
        if not np.isfinite(shape_std) or shape_std < 0.:
            raise ValueError('shape_std must be finite and nonnegative')
        # A shape is an episode property, not a geometry-rejection outcome.
        betas = np.zeros((len(ids), 300))
        if shape_std:
            betas[:, :10] = rng.normal(0., shape_std, size=(len(ids), 10))
        nominal = prior.decode_knots_to_world(prior.mean)
        pending = list(range(len(ids)))
        outcomes = {}
        rejected = np.zeros((len(ids), 5), dtype=np.intp)
        exhausted = []
        limit = max_attempts if policy == 'reject' and not deterministic else 1
        for attempt in range(1, limit + 1):
            if not pending:
                break
            states = prior.generation_latent(rng, len(pending))
            geometries = prior.decode_knots_to_world(states)
            retry = []
            accepted = []
            for j, (k, geometry) in enumerate(zip(pending, geometries, strict=True)):
                report = self.audit_centers(prior.graph, geometry, prior=prior, latent=states[j])
                if policy == 'reject':
                    reasons = (report.outside_skin > 0, report.in_excluded_bone > 0,
                               require_known and not report.all_locations_known, report.unsupported_material_points > 0)
                    if any(reasons):
                        rejected[k] += (1, *reasons)
                        if attempt == limit:
                            exhausted.append(ExhaustedDraw(ids[k], RejectionCounts(*map(int, rejected[k])), report))
                        retry.append(k)
                        continue
                accepted.append((j, k, report))
            for j, k, report in accepted:
                state, geometry = states[j], geometries[j]
                if retry:
                    # Do not pin an entire batch of rejected proposals.
                    state, geometry = state.copy(), geometry.copy()
                shifts_sq = np.sum((geometry - nominal)**2, axis=1)
                quantiles = np.percentile(np.sqrt(shifts_sq), (50, 90))
                outcomes[k] = SampledAnatomy(source_id, ids[k], prior, state, geometry, report, attempt,
                                             RejectionCounts(*map(int, rejected[k])), law, betas[k],
                                             float(np.sqrt(shifts_sq.mean())), tuple(map(float, quantiles)))
            pending = retry
        if exhausted:
            raise GeometryRejectionError(source_id, tuple(outcomes[i] for i in sorted(outcomes)), tuple(exhausted))
        return [outcomes[i] for i in range(len(ids))]

    def sample_anatomy(self, source_id: str, shape_id: object, rng: np.random.Generator,
                       difficulty: str | DifficultyProfile = 'ordinary', *, geometry_policy: str | None = None,
                       max_attempts: int = 128, require_known: bool = False) -> SampledAnatomy:
        return self.sample_batch(source_id, (shape_id,), rng, difficulty, geometry_policy=geometry_policy,
                                 max_attempts=max_attempts, require_known=require_known)[0]

    def shift_sweep_material(self, source_id: str, offset_mm: float, *, shape_id=None) -> SampledAnatomy:
        """Fixed shared circumferential arc shift, with no shear or resampling."""
        if self.coordinate_mode != 'material':
            raise ValueError('material sweep requires coordinate_mode=material')
        if not np.isfinite(offset_mm):
            raise ValueError('material sweep offset must be finite')
        p = STRESS_MATERIAL
        profile = DifficultyProfile('shift_sweep_material', p.sigma_shared_mm, p.sigma_relative_mm,
                                    p.sigma_landmark_mm, float(offset_mm))
        return self.sample_anatomy(source_id, shape_id, np.random.default_rng(0), profile, geometry_policy='raw')
