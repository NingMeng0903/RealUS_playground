"""Train-only canonical priors and separate, supervised physics scan tapes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cached_property, lru_cache
import operator

import numpy as np
from scipy import sparse

from sim.anatomy_assets import (
    HOLDOUT_SOURCE_ID,
    MEDSEG_RIGHT,
    NYSORA_STATIONS,
    THIGH_NYSORA_Y_BOUNDS_MM,
    AnatomySource,
    AnatomySplit,
    TrainingAnatomyBank,
    VesselRole,
)
from sim.audit_training_support import station_support_masks
from sim.material_chart import RightLegMaterialChart
from sim.metric_graph_gp import (
    ORDINARY_MATERIAL,
    STRESS_MATERIAL,
    DifficultyProfile,
    MetricGraphAnatomyGenerator,
    MetricGraphPrior,
    VascularMetricGraph,
)
from sim.physics_ultrasound_kernel import (
    PhysicsUltrasoundKernel,
    ProbePose,
    SpecularReflector,
    UltrasoundObservation,
    UltrasoundTruth,
    VesselTruth,
    _draw_key,
)
from sim.smplx_dynamic_atlas import DynamicMotionProfile, SMPLXLegKinematics


_DOMAIN = 'thigh_nysora'
_LO, _HI = THIGH_NYSORA_Y_BOUNDS_MM
_TRUNK = (VesselRole.MAIN_VEIN, VesselRole.MAIN_ARTERY)
_ELEVATION = np.array([0., 1., 0.])
_BENCHMARKS = frozenset(('offset', 'dynamic', 'retroactive', 'station', 'conformal'))
_DFA_CAP_MM = 120.
_DFA_TREND_DECAY_MM = 30.
_TAPE_PROFILE = DifficultyProfile('scan_tape', (10., 3.5, .18), (1.8, 1.2, .10), 2.5)
_VIEW_ANGLES = np.linspace(-np.pi, np.pi, 72, endpoint=False)
_VIEW_RAYS = np.column_stack((np.cos(_VIEW_ANGLES), np.sin(_VIEW_ANGLES)))
_REWINDOW_STEPS = len(_VIEW_ANGLES) // 8  # 45 degrees around the nominal training trunk
_SWEEP_FORCE_N = (2.4, 3.2, 3.7, 4.1, 4.35, 4.45, 2.6, 2.4)


def _read_only(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _wrap(angle):
    return (angle + np.pi) % (2. * np.pi) - np.pi


def _training_sources(bank: TrainingAnatomyBank, split: AnatomySplit) -> tuple[AnatomySource, ...]:
    train = split.train_source_ids
    if (not train or len(set(train)) != len(train) or set(train).intersection(split.test_source_ids)
            or HOLDOUT_SOURCE_ID in train):
        raise ValueError('split needs unique non-holdout training IDs disjoint from test IDs')
    return tuple(bank.source(source_id) for source_id in train)


def station_windows(graph: VascularMetricGraph) -> dict[str, np.ndarray]:
    """Overlapping Gate 2 windows on raw co-present source traces."""
    heights, order = graph._ordered_stations
    _, masks = station_support_masks(heights, graph.supported[order], graph)
    return {name: _read_only(heights[rows]) for name, rows in masks.items()}


def _material_evidence(graph: VascularMetricGraph, heights: np.ndarray, chart: RightLegMaterialChart):
    """Resample observed knots/links on a union grid without crossing source gaps."""
    h, order = graph._ordered_stations
    values = np.full((len(heights), 6, 3), np.nan)
    links = np.zeros((len(heights) - 1, 6), dtype=bool)
    addresses, _, audit = chart.encode_with_audit(graph.knot_xyz_mm)
    valid = chart.corridor_supported(addresses[:, 0], addresses[:, 1])
    audit['unsupported_material_rays'] = int(np.count_nonzero(~valid))
    addresses[~valid] = np.nan
    if len(h) > 1:
        segment = np.searchsorted(h, (heights[:-1] + heights[1:]) / 2.) - 1
        inside = (segment >= 0) & (segment < len(h) - 1)
        left, right = order[segment[inside]], order[segment[inside] + 1]
        links[inside] = (graph.links[np.minimum(left, right)] & valid[graph.sample_knots[left]]
                         & valid[graph.sample_knots[right]])
    for row, height in enumerate(heights):
        for role in range(6):
            bracket = graph._station_query(float(height), role)
            if bracket is not None:
                a, b, t = bracket
                if valid[a] and valid[b]:
                    delta = addresses[b] - addresses[a]
                    delta[0] = _wrap(delta[0])
                    values[row, role] = addresses[a] + t * delta
    return values, links, audit


def _extend_dfa(heights, material, links, source_values, source_links, chart):
    """Continue the observed distal DFA end relative to contiguous training artery."""
    artery, dfa = VesselRole.MAIN_ARTERY, VesselRole.DFA
    observed_dfa = np.flatnonzero(np.isfinite(material[:, dfa, 0]))
    growth = np.zeros(len(heights))
    if not observed_dfa.size:
        return growth, ()
    boundary = int(observed_dfa[0])
    if boundary == 0 or not np.isfinite(material[boundary, artery, 0]):
        return growth, ()
    trends, contributors = [], []
    for i in range(source_values.shape[2]):
        branch_pair = source_values[:, [artery, dfa], i]
        if not np.isfinite(branch_pair[boundary]).all():
            continue
        contributors.append(i)
        end = boundary
        while (end + 1 < len(heights) and heights[end + 1] - heights[boundary] <= 20.
               and source_links[end, [artery, dfa], i].all()):
            end += 1
        if end > boundary:
            relative = (branch_pair[boundary:end + 1, 1] - branch_pair[boundary:end + 1, 0])[:, [0, 2]]
            relative[:, 0] = np.unwrap(_wrap(relative[:, 0]))
            distance = heights[boundary] - heights[boundary:end + 1]
            slope = np.linalg.lstsq(np.column_stack((np.ones(len(distance)), distance)), relative, rcond=None)[0][1]
            trends.append(np.clip(slope, [-.010, -.080], [.010, .080]))
    if not contributors:
        return growth, ()
    trend = np.mean(trends, axis=0) if trends else np.zeros(2)
    anchor = material[boundary, dfa, [0, 2]] - material[boundary, artery, [0, 2]]
    anchor[0] = _wrap(anchor[0])
    for row in range(boundary - 1, -1, -1):
        distance = heights[boundary] - heights[row]
        if distance > _DFA_CAP_MM or not links[row, artery]:
            break
        address = material[row, artery].copy()
        address[[0, 2]] += anchor + trend * (-_DFA_TREND_DECAY_MM * np.expm1(-distance / _DFA_TREND_DECAY_MM))
        delta = material[row + 1, dfa] - address
        delta[0] = _wrap(delta[0])
        segment = address + np.linspace(0., 1., 5)[:, None] * delta
        if not chart.corridor_supported(segment[:, 0], segment[:, 1]).all():
            break
        material[row, dfa] = address
        links[row, dfa] = True
        growth[row] = distance
    return growth, tuple(contributors) if growth.any() else ()


def _relative_variance(heights, source_values, present, chart):
    """Local train-pair variance with the common station translation removed."""
    ss, count = np.zeros((*present.shape[:2], 3)), np.zeros((*present.shape[:2], 3))
    radius = chart.nominal_reference_radius(heights)[:, None]
    clip_lo, clip_hi = chart.eta_from_d(np.array([chart.d_min + 1e-4, chart.d_max - 1e-4]))
    for i in range(present.shape[2]):
        for j in range(i):
            common = present[:, :, i] & present[:, :, j]
            a, b = source_values[:, :, i], source_values[:, :, j]
            delta = a - b
            angles = _wrap(delta[..., 0])
            trunk = common[:, :2]
            origin = np.arctan2(np.where(trunk, np.sin(angles[:, :2]), 0.).sum(axis=1),
                                np.where(trunk, np.cos(angles[:, :2]), 0.).sum(axis=1))
            delta[..., 0] = radius * _wrap(angles - origin[:, None])
            # Axial coordinates were registered to fixed station heights.
            # Saturated depth logits are projection bounds, not depth samples.
            interior = (clip_lo + 1e-8 < a[..., 2]) & (a[..., 2] < clip_hi - 1e-8)
            interior &= (clip_lo + 1e-8 < b[..., 2]) & (b[..., 2] < clip_hi - 1e-8)
            usable = np.stack((common, np.zeros_like(common), common & interior), axis=-1)
            anchors = usable[:, :2]
            n = anchors.sum(axis=1, keepdims=True)
            mean = np.where(anchors, delta[:, :2], 0.).sum(axis=1, keepdims=True) / np.maximum(n, 1)
            usable &= n > 0
            usable[:, :2] &= n > 1
            squared = np.where(usable, delta - mean, 0.)**2
            # A trunk contrast co-centers two samples; peripheral roles keep
            # the deviation from their shared trunk anchor.
            squared[:, :2] *= n / np.maximum(n - 1, 1)
            ss += .5 * squared
            count += usable
    weights = np.maximum(0., 1. - abs(heights[:, None] - heights[None, :]) / 25.)
    smooth_ss, smooth_n = (np.einsum('ij,jrc->irc', weights, x) for x in (ss, count))
    local = np.divide(smooth_ss, smooth_n, out=np.zeros_like(smooth_ss), where=smooth_n > 0.)
    # Transfer peripheral angular spread to roles without local pair evidence;
    # logit-depth spread stays local to the observed role, with no pooled floor.
    known = local[:, 2:, 0][smooth_n[:, 2:, 0] > 0.]
    if known.size:
        local[:, 2:, 0] = np.maximum(local[:, 2:, 0], .65 * np.quantile(known, .75))
    return local


class _SplitPrior(MetricGraphPrior):
    """Diagonal relative-coordinate covariance dilation, with matching precisions and draws."""

    def __init__(self, graph, chart, empirical_var, dfa_growth_mm, windows, *, multi_source):
        super().__init__(graph, domain=_DOMAIN, chart=chart)
        self.provenance = graph._source.provenance
        self.station_windows = windows
        scale = np.ones(self.layout.relative_shape)
        if multi_source:
            rows = np.flatnonzero(dfa_growth_mm)
            if rows.size:
                boundary = rows[-1] + 1  # pooled graph heights are ascending
                empirical_var[rows, VesselRole.DFA] = empirical_var[boundary, VesselRole.DFA]
            supported = graph.supported
            np.maximum.at(scale, graph.sample_knots[supported],
                          np.sqrt(1. + empirical_var[supported] / self._relative_sigma**2))
            # Tail distance adds uncertainty in relative separation only,
            # not in the registered axial coordinate or the common trunk field.
            tail = graph.sample_knots[rows, VesselRole.DFA]
            scale[tail[:, None], [0, 2]] *= 1. + 2. * (-np.expm1(-dfa_growth_mm[rows, None] / 40.))
        self.relative_std_scale = scale
        _read_only(self.relative_std_scale)

    @cached_property
    def precision(self) -> sparse.csc_matrix:
        base = MetricGraphPrior.precision.func(self)
        if np.any(self.relative_std_scale != 1.):
            inverse = np.ones(self.layout.size)
            inverse[self.layout.relative] = 1. / self.relative_std_scale.ravel()
            # Scale in place on the new CSC matrix, without retaining another precision.
            base.data *= inverse[base.indices] * np.repeat(inverse, np.diff(base.indptr))
        return base

    def sample_latent(self, rng: np.random.Generator, size: int | None = None) -> np.ndarray:
        latent = super().sample_latent(rng, size)
        relative = latent[..., self.layout.relative].reshape(*latent.shape[:-1], *self.layout.relative_shape)
        relative *= self.relative_std_scale
        return latent


def _contributors(mask: np.ndarray, source_ids: Sequence[str]) -> tuple:
    return tuple(tuple(tuple(source_ids[i] for i in np.flatnonzero(role)) for role in row) for row in mask)


def build_split_prior(bank: TrainingAnatomyBank, split: AnatomySplit, *,
                      chart: RightLegMaterialChart | None = None) -> MetricGraphPrior:
    """Pool training addresses and source-supported links on a common nominal grid.

    Verified contacts in the native MedSeg graph can authorize connectors, not
    station support. If MedSeg is not a training source, no such contacts are
    read. A CT-only split stops at the actual proximal coverage boundary.
    """
    sources = _training_sources(bank, split)
    native = bank.native_graph if MEDSEG_RIGHT in split.train_source_ids else {}
    chart = RightLegMaterialChart() if chart is None else chart
    graphs = tuple(VascularMetricGraph.from_source(s, native, domain=_DOMAIN) for s in sources)
    observed_h = np.concatenate([graph.h_mm[graph.supported.any(axis=1)] for graph in graphs])
    heights = np.unique(np.concatenate([graph.h_mm for graph in graphs]))
    heights = heights[(observed_h.min() <= heights) & (heights <= observed_h.max())]
    evidence = [_material_evidence(graph, heights, chart) for graph in graphs]
    source_values = np.stack([value for value, _, _ in evidence], axis=2)
    source_links = np.stack([links for _, links, _ in evidence], axis=2)
    present = np.isfinite(source_values[..., 0])
    observed = present.any(axis=2)
    material = np.full((*observed.shape, 3), np.nan)
    observations = source_values[observed]
    # Circular mean for theta; in a degenerate antipodal mean choose the first
    # training contribution rather than fabricate an arbitrary chart direction.
    sine = np.nansum(np.sin(observations[..., 0]), axis=1)
    cosine = np.nansum(np.cos(observations[..., 0]), axis=1)
    first = observations[np.arange(len(observations)), present[observed].argmax(axis=1), 0]
    theta = np.where(np.hypot(sine, cosine) > 1e-12, np.arctan2(sine, cosine), first)
    material[observed] = np.column_stack((theta, heights[np.nonzero(observed)[0]],
                                         np.nanmean(observations[..., 2], axis=1)))
    valid = chart.corridor_supported(material[..., 0], material[..., 1])
    present &= valid[..., None]
    material[~valid] = np.nan
    source_links &= present[:-1] & present[1:]
    links = source_links.any(axis=2)
    empirical_var = _relative_variance(heights, source_values, present, chart) if len(sources) > 1 else None
    growth, dfa_sources = _extend_dfa(heights, material, links, source_values, source_links, chart)
    supported = np.isfinite(material[..., 0])
    centers = np.full_like(material, np.nan)
    centers[supported] = chart.tilde_xi_to_world(material[supported])
    provenance = {
        'train_source_ids': split.train_source_ids,
        'native_topology_source': MEDSEG_RIGHT if MEDSEG_RIGHT in split.train_source_ids else None,
        'canonical_pool': 'equal training-source mean of periodic azimuth and logit depth at each nominal height',
        'link_policy': 'straight-through role traces only along contributed source links, except marked distal DFA',
        'clinical_station_labels_available': False,
        'direct_source_ids_by_row_role': _contributors(present, split.train_source_ids),
        'link_source_ids_by_row_role': _contributors(source_links, split.train_source_ids),
        'source_projection_audits': {g.source_id: e[2] for g, e in zip(graphs, evidence, strict=True)},
        'supported_height_range_mm': (float(heights[supported.any(axis=1)].min()),
                                      float(heights[supported.any(axis=1)].max())),
        'extrapolated_dfa_heights_mm': tuple(map(float, heights[growth > 0.])),
        'dfa_extrapolation_source_ids': tuple(split.train_source_ids[i] for i in dfa_sources),
        'dfa_extrapolation': ('distal training boundary separation and tapered local trend relative to the '
                              'connected training artery') if growth.any() else None,
        'relative_covariance_policy': ('trunk-centered train-pair contrasts in local 25 mm neighborhoods; '
                                       'peripheral azimuth transfer, observed per-role logit depth only, '
                                       'no empirical axial dilation'
                                       if len(sources) > 1 else 'unchanged single-source GP scale'),
        'station_window_policy': 'union of overlapping Gate 2 windows computed on individual raw training sources',
    }
    names = np.full(supported.shape, '', dtype=object)
    names[supported] = 'pooled_train'
    names[growth > 0., VesselRole.DFA] = 'extrapolated_relative_to_train_artery'
    source = AnatomySource('split_train_pool', _read_only(heights), _read_only(centers),
                           _read_only(supported), _read_only(links), _read_only(names), provenance)
    graph = VascularMetricGraph.from_source(source, native, domain=_DOMAIN)
    source_windows = [station_windows(source_graph) for source_graph in graphs]
    windows = {name: _read_only(np.unique(np.concatenate([w[name] for w in source_windows])))
               for name in NYSORA_STATIONS}
    return _SplitPrior(graph, chart, empirical_var, growth, windows, multi_source=len(sources) > 1)


@dataclass(frozen=True)
class ScanSupervision:
    """Offline-only supervision captured alongside the rendered public frame."""

    echo_targets: tuple[tuple[int, ...], ...]
    centers_mm: np.ndarray  # (6, 3); NaN where the local slice has no sample
    true_q_rad: np.ndarray  # (4,), immutable actual pose at acquisition
    acquisition_h_mm: float  # canonical physical steering height
    echo_ghosts: tuple[bool, ...]  # measured echo contains a reflector contributor


@dataclass(frozen=True)
class ScanFrame:
    obs: UltrasoundObservation
    probe_pose: ProbePose
    f_contact: float
    q_hat: np.ndarray
    Sigma_q: np.ndarray
    dt: float
    supervision: ScanSupervision

    @property
    def echo_targets(self) -> tuple[tuple[int, ...], ...]:
        return self.supervision.echo_targets

    @property
    def centers_mm(self) -> np.ndarray:
        return self.supervision.centers_mm


@dataclass(frozen=True)
class ScanEpisode:
    source_id: str
    station_id: str | None
    frames: tuple[ScanFrame, ...]
    h_start_mm: float
    provenance: Mapping[str, object]


class _SupervisedKernel(PhysicsUltrasoundKernel):
    """Collect provenance at the actual measurement boundary, before the same stable sort.

    Candidate construction, shadowing, grouping, group noise and rejection all
    remain in the unmodified physics kernel. No matching to noisy centroids and
    no second acoustic render is needed. The public observation is unchanged.
    """

    def _measure(self, group, tape_key, caliber, compression, pulse):
        row = super()._measure(group, tape_key, caliber, compression, pulse)
        if row is not None:
            targets = tuple(sorted({self._roles[c.draw_key] for c in group if c.draw_key in self._roles}))
            ghost = any(c.key.startswith(b'R') for c in group)
            self._targets.append((row[0], targets, ghost))
        return row

    def render_supervised(self, truth, pose, force, step, roles):
        self._roles = roles
        self._targets = []
        obs = super().render(truth, pose, force, step)
        self._targets.sort(key=lambda row: row[0])
        return obs, tuple(target for _, target, _ in self._targets), tuple(ghost for _, _, ghost in self._targets)


def _beam_frame(depth, elevation):
    """Transverse frame with depth along a physically aimed beam."""
    depth = depth / np.linalg.norm(depth)
    lateral = np.cross(elevation, depth)
    lateral /= np.linalg.norm(lateral)
    return np.column_stack((lateral, np.cross(depth, lateral), depth))


def _cross_xz(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _skin_apertures(chart, target):
    """Intersect transverse rays with actual neutral skin, not the chart's medial cut."""
    triangles, starts, ids = chart._mesh  # verified neutral skin and height buckets
    lo, hi = chart.h_bounds_mm
    h = target[1]
    n = len(starts) - 1
    bucket = min(n - 1, int((h - lo) * n / (hi - lo)))
    triangles = triangles[ids[starts[bucket]:starts[bucket + 1]]]
    edge_end = np.roll(triangles, -1, axis=1)
    face, edge = np.nonzero((triangles[..., 1] > h) != (edge_end[..., 1] > h))
    a, b = triangles[face, edge], edge_end[face, edge]
    crossing = a + ((h - a[:, 1]) / (b[:, 1] - a[:, 1]))[:, None] * (b - a)
    cut = crossing[:, [0, 2]].reshape(-1, 2, 2)
    if not len(cut):
        raise ValueError(f'neutral skin has no transverse section at {h:g} mm')
    segment = cut[:, 1] - cut[:, 0]
    start = cut[:, 0] - target[[0, 2]]
    ray = _VIEW_RAYS[:, None]
    den = _cross_xz(ray, segment)
    denom_ok = abs(den) > 1e-9
    travel = np.divide(_cross_xz(start, segment), den, out=np.full_like(den, np.inf), where=denom_ok)
    progress = np.divide(_cross_xz(start, ray), den, out=np.full_like(den, np.inf), where=denom_ok)
    travel[(travel <= 0.) | (progress < 0.) | (progress >= 1.)] = np.inf
    edge_id = travel.argmin(axis=1)
    distance = travel[np.arange(len(ray)), edge_id]
    aperture = np.broadcast_to(target, (len(ray), 3)).copy()
    aperture[:, [0, 2]] += _VIEW_RAYS * np.where(np.isfinite(distance), distance, 0.)[:, None]
    normal_alignment = np.divide(abs(den[np.arange(len(ray)), edge_id]), np.linalg.norm(segment[edge_id], axis=1),
                                 out=np.zeros(len(ray)), where=np.isfinite(distance))
    # Stay on the right exposed skin (do not point from the contralateral thigh).
    distance[aperture[:, 0] >= -4.] = np.inf
    return aperture, distance, normal_alignment


def _slice_centers(graph, rows, posed, pose: ProbePose, h: float, slice_half_width: float) -> np.ndarray:
    centers = np.full((6, 3), np.nan)
    elevation = (posed - pose.origin_mm) @ pose.rotation[:, 1]
    connected = graph.links[rows[:-1]] & ((rows[1:] - rows[:-1]) == 1)[:, None]
    for role in range(6):
        intersects = np.flatnonzero(connected[:, role] & (elevation[:-1, role] * elevation[1:, role] <= 0.))
        if intersects.size:
            a, b = elevation[intersects, role], elevation[intersects + 1, role]
            t = np.divide(a, a - b, out=np.zeros_like(a), where=a != b)
            bracket_h = graph.h_mm[rows[intersects]] * (1. - t) + graph.h_mm[rows[intersects + 1]] * t
            best = int(np.argmin(abs(bracket_h - h)))
            i, w = intersects[best], t[best]
            centers[role] = (1. - w) * posed[i, role] + w * posed[i + 1, role]
        else:
            # A real endpoint or singleton may be in the finite slice; never
            # connect through missing source data to make an intersection.
            in_slice = np.flatnonzero(abs(elevation[:, role]) <= slice_half_width)
            if in_slice.size:
                nearest = in_slice[np.lexsort((abs(graph.h_mm[rows[in_slice]] - h),
                                              abs(elevation[in_slice, role])))[0]]
                centers[role] = posed[nearest, role]
    return centers


def _rock(basis: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    u, v = np.cos(pitch), np.sin(pitch)
    return basis @ np.array([[c, -s * v, s * u], [0., u, v], [-s, -c * v, c * u]])


class _ScanMotion(DynamicMotionProfile):
    def __init__(self, seed, *, moving):
        super().__init__(seed, moving=moving, jitter_deg=(2., 2.5, 3., 4.) if moving else .15,
                         latency_s=.06 if moving else 0., ou_tau_s=.85,
                         ou_stationary_std_mm=(.85, .65, .95) if moving else (.45, .35, .50))
        if moving:
            self._amp = np.deg2rad([12.5, 10., 8., 17.5])
            self._offset = np.deg2rad([12.5, 0., 0., 17.5])


def _bone_mesh(occupancy, h):
    """Local marching-cubes surface of the registered population bone field."""
    from skimage.measure import marching_cubes

    keep = np.flatnonzero(np.abs(occupancy.y_mm - h) <= 18.)
    if len(keep) < 2:
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)
    field = occupancy.count[:, keep[0]:keep[-1] + 1, :] / occupancy.cohort_size
    if not field.min() < .55 < field.max():
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)
    vertices, faces, _, _ = marching_cubes(field, level=.55, allow_degenerate=False)
    grid = (occupancy.x_mm, occupancy.y_mm[keep], occupancy.z_mm)
    world = np.column_stack([np.interp(v, np.arange(len(axis)), axis) for v, axis in zip(vertices.T, grid, strict=True)])
    return world, faces

class DynamicScanSimulator:
    """Generate reproducible train or evaluation tapes without mixing their anatomy.

    Only start-height sampling uses station labels. After that, the probe
    visits neutral skin contacts steered from the nominal training trunk,
    independent of the label, source anatomy intervention, or realized echoes.
    Acoustic cues are synthetic mechanical populations, not clinical measurements.
    """

    def __init__(self, bank: TrainingAnatomyBank, split: AnatomySplit, *,
                 kinematics: SMPLXLegKinematics | None = None):
        self._bank, self._split = bank, split
        self._train = _training_sources(bank, split)
        self._native = bank.native_graph if MEDSEG_RIGHT in split.train_source_ids else {}
        self.kinematics = SMPLXLegKinematics() if kinematics is None else kinematics
        self._graphs: dict[str, VascularMetricGraph] = {}
        self._generator = MetricGraphAnatomyGenerator(replace(bank, native_graph=self._native),
                                                     coordinate_mode='material', domain=_DOMAIN)
        self._offset_geometry = {}
        self._bone_meshes = {}
        self._evaluation_sources: dict[str, AnatomySource] = {}
        self._station_overrides: dict[str, dict[str, np.ndarray]] = {}
        self._source_windows: dict[str, dict[str, np.ndarray]] = {}

    @cached_property
    def _chart(self) -> RightLegMaterialChart:
        return RightLegMaterialChart()

    def register_evaluation_sources(self, sources: Iterable[AnatomySource], *,
                                    station_windows_by_source: Mapping[str, Mapping[str, Sequence[float]]] | None = None):
        """Accept already-authorized external test geometry; performs no loader I/O."""
        pending = {}
        for source in sources:
            sid = source.source_id
            if (sid not in self._split.test_source_ids or sid in self._bank.source_ids
                    or sid in self._evaluation_sources or sid in pending):
                raise ValueError(f'external evaluation source {sid!r} must be a new split test ID')
            pending[sid] = source
        windows = {}
        for sid, per_station in (station_windows_by_source or {}).items():
            if sid not in pending or set(per_station).difference(NYSORA_STATIONS):
                raise ValueError('external station windows must name newly registered sources and known stations')
            windows[sid] = {}
            for name, values in per_station.items():
                h = np.asarray(values, dtype=float)
                if h.ndim != 1 or not np.isfinite(h).all():
                    raise ValueError('external station heights must be finite vectors')
                windows[sid][name] = _read_only(np.unique(h))
        self._evaluation_sources.update(pending)
        self._station_overrides.update(windows)

    def _graph(self, source_id: str) -> VascularMetricGraph:
        if source_id not in self._split.train_source_ids and source_id not in self._split.test_source_ids:
            raise ValueError(f'source {source_id!r} is not in this split')
        if source_id not in self._graphs:
            source = self._evaluation_sources.get(source_id)
            if source is None:
                self._graphs[source_id] = self._generator.graph(source_id)
            else:
                self._graphs[source_id] = VascularMetricGraph.from_source(source, self._native, domain=_DOMAIN)
        return self._graphs[source_id]

    @cached_property
    def _train_scope(self):
        graphs = [self._graph(s.source_id) for s in self._train]
        coverage = _read_only(np.unique(np.concatenate([g.h_mm[g.supported[:, _TRUNK].all(axis=1)] for g in graphs])))
        return coverage, frozenset(
            station for graph in graphs for station, h in station_windows(graph).items() if h.size
        )

    def _windows(self, source_id: str) -> dict[str, np.ndarray]:
        if source_id not in self._source_windows:
            graph = self._graph(source_id)
            windows = station_windows(graph)
            _, allowed = self._train_scope
            override = self._station_overrides.get(source_id, {})
            for name, roles in NYSORA_STATIONS.items():
                if name not in allowed:
                    windows[name] = _read_only(windows[name][:0])
                elif name in override:
                    # External candidates have their own archived landmark
                    # evidence, but still require raw same-source role support.
                    rows = np.isin(graph.h_mm, override[name]) & graph.supported[:, sorted(roles)].all(axis=1)
                    windows[name] = _read_only(np.sort(graph.h_mm[rows]))
            self._source_windows[source_id] = windows
        return self._source_windows[source_id]

    def station_ids(self, source_id: str) -> tuple[str, ...]:
        return tuple(name for name, windows in self._windows(source_id).items() if windows.size)

    @lru_cache(maxsize=2048)
    def _aim(self, h):
        """Fixed neutral-skin steering from the nominal training trunk at h."""
        training, role_index, weights = [], [], []
        for source in self._train:
            graph = self._graph(source.source_id)
            for role in _TRUNK:
                bracket = graph._station_query(h, role)
                if bracket is not None:
                    a, b, w = bracket
                    point = (1. - w) * graph.knot_xyz_mm[a] + w * graph.knot_xyz_mm[b]
                    distance = 0.
                else:
                    rows = np.flatnonzero(graph.supported[:, role])
                    if not rows.size:
                        continue
                    row = rows[np.argmin(abs(graph.h_mm[rows] - h))]
                    distance = abs(graph.h_mm[row] - h)
                    if distance > 20.:
                        continue
                    point = graph.knot_xyz_mm[graph.sample_knots[row, role]].copy()
                point[1] = h
                training.append(point)
                role_index.append(role)
                weights.append(np.exp(-.5 * (distance / 10.)**2))
        if not training:
            raise ValueError(f'no training anatomy available to aim at {h:g} mm')
        training, role_index, weights = np.asarray(training), np.asarray(role_index), np.asarray(weights)
        means = [np.average(training[role_index == role], axis=0, weights=weights[role_index == role])
                 for role in np.unique(role_index)]
        center = np.mean(means, axis=0)
        aperture, depth, alignment = _skin_apertures(self._chart, center)
        relative = training[:, [0, 2]] - center[[0, 2]]
        lateral = abs(_cross_xz(relative[None], _VIEW_RAYS[:, None]))
        spread = (lateral**2 @ weights) / weights.sum()
        projected_depth = depth[:, None] - _VIEW_RAYS @ relative.T
        # Keep margin for placement and anatomy variability without changing
        # the acoustic field of view. This envelope is nominal training only.
        outside = np.maximum(lateral - 19., 0.)**2
        outside += np.maximum(projected_depth - 68., 0.)**2 + np.maximum(8. - projected_depth, 0.)**2
        costs = depth / 35. + 2.5 * (1. - alignment) + spread / 16.**2 + outside.max(axis=1) / 10.**2
        return center, aperture, costs

    def _rest_geometry(self, graph, benchmark, offset_mm, rng, profile):
        source_id = graph.source_id
        if source_id in self._evaluation_sources:
            if not offset_mm:
                return graph.knot_xyz_mm, 'raw registered external coordinates; source support unchanged'
            xi = self._chart.world_to_xi(graph.knot_xyz_mm)
            xi[:, 0] += offset_mm / self._chart.nominal_reference_radius(xi[:, 1])
            return self._chart.xi_to_world(xi), 'arc-shifted raw external geometry; source support unchanged'
        if benchmark == 'offset':
            key = source_id, offset_mm
            if key not in self._offset_geometry:
                draw = self._generator.shift_sweep_material(source_id, offset_mm)
                self._offset_geometry[key] = draw.knot_centers_mm, 'fixed source material arc intervention'
            return self._offset_geometry[key]
        prior = self._generator.prior(source_id, profile)
        latent, policy = prior.sample_latent(rng), 'source material GP draw'
        material = prior.material_knots(latent)
        valid = self._chart.corridor_supported(material[:, 0], material[:, 1])
        knots = np.full_like(material, np.nan)
        knots[valid] = self._chart.tilde_xi_to_world(material[valid])
        return knots, policy

    def episode(self, source_id: str, seed: int, *, benchmark: str = 'dynamic', offset_mm: float = 0.,
                station_id: str | None = None, frames: int = 8,
                augment: DifficultyProfile | None = None) -> ScanEpisode:
        """Render one tape; only train-source station tapes may augment the material prior."""
        frames = operator.index(frames)
        if benchmark not in _BENCHMARKS or frames < 1 or not np.isfinite(offset_mm):
            raise ValueError('invalid tape benchmark, frame count, or offset')
        if offset_mm and benchmark != 'offset':
            raise ValueError('nonzero offset is an intervention for the offset benchmark only')
        if augment is not None and (
            source_id not in self._split.train_source_ids or benchmark != 'station'
            or augment not in (ORDINARY_MATERIAL, STRESS_MATERIAL)
        ):
            raise ValueError('ordinary/stress material augmentation requires a train-source station tape')
        graph = self._graph(source_id)
        rng_seed, shape_seed, motion_seed, acoustic_seed, station_seed = np.random.SeedSequence(seed).spawn(5)
        rng, shape_rng = np.random.default_rng(rng_seed), np.random.default_rng(shape_seed)
        coverage, _ = self._train_scope
        lo, hi = coverage[0], coverage[-1]
        # Retain the unstratified tape draw and acquisition RNG for all
        # benchmarks. Stratified site/label draws use a separate stream so
        # different station-window cardinalities cannot affect later controls.
        h_start = float(rng.choice(coverage))
        if station_id is not None or benchmark in ('station', 'conformal'):
            windows = self._windows(source_id)
            station_rng = np.random.default_rng(station_seed)
            if station_id is None:
                choices = self.station_ids(source_id)
                if not choices:
                    raise ValueError(f'no supported station windows for {source_id!r}')
                station_id = str(station_rng.choice(choices))
            if station_id not in windows or not windows[station_id].size:
                raise ValueError(f'station {station_id!r} is unsupported for {source_id!r} in this split')
            h_start = float(station_rng.choice(windows[station_id]))
        profile = augment if augment is not None else _TAPE_PROFILE
        rest_knots, geometry = self._rest_geometry(graph, benchmark, offset_mm, shape_rng, profile)
        direction = int(rng.choice((-1, 1)))
        moving = benchmark in ('dynamic', 'retroactive')
        motion = _ScanMotion(int(motion_seed.generate_state(1)[0]), moving=moving)
        kernel = _SupervisedKernel(int(acoustic_seed.generate_state(1)[0]), lateral_half_width_mm=28.)
        target0, contacts0, costs0 = self._aim(h_start)
        view_indices = np.arange(len(_VIEW_ANGLES))
        placement = int(rng.integers(-1, 2))
        if benchmark == 'retroactive':
            # A roll around the trunk needs two distinct feasible skin
            # contacts. Select the pair from the same training-only section.
            partners = (view_indices + direction * _REWINDOW_STEPS) % len(view_indices)
            pair_cost = costs0 + costs0[partners]
            if not np.isfinite(pair_cost).any():
                raise ValueError(f'no pair of right skin contacts 45 degrees apart at {h_start:g} mm')
            first_view = int(np.argmin(pair_cost + .25 * costs0))
            shifted_view = (first_view + placement) % len(view_indices)
            if np.isfinite(pair_cost[shifted_view]):
                first_view = shifted_view
            recheck_view = int(partners[first_view])
        else:
            best = int(np.argmin(costs0))
            first_view = (best + placement) % len(view_indices)
            if not np.isfinite(costs0[first_view]):
                first_view = best
            if not np.isfinite(costs0[first_view]):
                raise ValueError(f'no right skin contact at {h_start:g} mm')
        initial_aperture = contacts0[first_view]
        initial_basis = _beam_frame(target0 - initial_aperture, _ELEVATION)
        station_scan = benchmark in ('station', 'conformal')
        sweep_scan = benchmark in ('dynamic', 'offset')
        dt = .18 if station_scan or sweep_scan else .16

        # Fix the virtual image source and reflector in world space at the
        # opening view. The later window changes the broken-ray incidence.
        use_reflector = benchmark == 'retroactive' or (benchmark == 'dynamic' and rng.random() < .30)
        reflectors = ()
        if use_reflector:
            depth = np.linalg.norm(target0 - initial_aperture)
            image = initial_aperture + initial_basis @ np.array([
                direction * min(7., .14 * depth), 0., np.clip(depth + 9., 19., 74.)
            ])
            points = np.stack((image - 8. * initial_basis[:, 2], image - 16. * initial_basis[:, 2]))
            world, orientations, _ = self.kinematics.forward_pose_canonical(motion.truth_at(0.).q, points)
            normal = orientations[0] @ -initial_basis[:, 2]
            reflectors = (SpecularReflector(world[0], normal, world[1], radius_mm=16., half_angle_deg=17.,
                                            diam_mm=6.3 + rng.uniform(-1., 1.)),)
        use_shadow = benchmark == 'dynamic' and rng.random() < .35
        patch = initial_aperture + np.array([[-13., -6., 11.], [-8., -6., 11.],
                                            [-8., 6., 11.], [-13., 6., 11.]]) @ initial_basis.T
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        if benchmark == 'dynamic' and h_start < -790.:
            if h_start not in self._bone_meshes:
                self._bone_meshes[h_start] = _bone_mesh(self._bank.bone_occupancy, h_start)
            bone, bone_faces = self._bone_meshes[h_start]
        else:
            bone, bone_faces = np.empty((0, 3)), np.empty((0, 3), dtype=int)

        tokens = tuple(map(int, rng.choice(2**61, size=6, replace=False)))
        roles = {_draw_key(b'V', token): role for role, token in enumerate(tokens)}
        arterial = np.isin(np.arange(6), [VesselRole.MAIN_ARTERY, VesselRole.DFA])
        # One broad overlapping diameter population; no role-specific table.
        diameter = np.maximum(1.5, 7.2 * rng.lognormal(0., .3, size=6))
        compliance = rng.lognormal(np.log(np.where(arterial, .035, .24)), .18)
        pulse = rng.lognormal(np.log(np.where(arterial, .12, .012)), .20)
        pulse_displacement = rng.normal(size=(6, 3)) * np.where(arterial, .55, .20)[:, None]
        compression_displacement = np.where(arterial, .12, .27)[:, None] * initial_basis[:, 2]
        sliding = rng.normal(0., .65, size=(6, 3))
        ou = motion.ou_initial((6,))
        tape = []
        dwell, previous_aperture, dwell_view = 0., None, None
        # Broad GP draws can transport distant knots into the local view.
        # In every case the acquisition, including the anchor, is chosen
        # without looking at the resulting geometry or rendered echoes.
        rows = (np.arange(len(graph.h_mm)) if augment is not None
                else np.flatnonzero(abs(graph.h_mm - h_start) <= 45.))
        local_knots = graph.sample_knots[rows]
        ii, role_index = np.nonzero(local_knots >= 0)
        indices = local_knots[ii, role_index]
        finite = np.isfinite(rest_knots[indices, 0])
        ii, role_index, indices = ii[finite], role_index[finite], indices[finite]
        rest = rest_knots[indices]
        fixed = np.concatenate((bone, patch if use_shadow else patch[:0]))
        cutoff = max(1, frames // 2)
        retro_echo_counts = None
        if benchmark == 'retroactive':
            retro_echo_counts = {
                phase: dict(ghost_echoes=0, lumen_echoes=0, mixed_ghost_lumen_echoes=0)
                for phase in ('pre_rewindow', 'post_rewindow')
            }
        for step in range(frames):
            t = step * dt
            truth, tracker = motion.sample(t)
            desired_view = first_view
            if benchmark == 'retroactive':
                recheck = step >= cutoff
                # Brief low-force holds only, moving by one overlapping
                # slice before enough dwell has accrued for a pulse summary.
                h = h_start if recheck else h_start - 2.5 * ((step // 2) % 2)
                desired_view = recheck_view if recheck else first_view
                pitch = 0.
                force = 4.5 if recheck else 2.4
            elif sweep_scan:
                cycle, phase = divmod(step, len(_SWEEP_FORCE_N))
                travel = min(phase, 3)
                if cycle % 2:
                    travel = 3 - travel
                if benchmark == 'dynamic':
                    sweep_t = travel * dt
                    h = h_start + direction * 11. * np.sin(1.4 * sweep_t)
                    desired_view += int(np.round(4. * np.sin(2.7 * sweep_t)))
                    pitch = .055 * np.sin(1.6 * sweep_t)
                else:
                    h = h_start + direction * (5. / 3.) * travel
                    pitch = 0.
                # Sweep, settle, compress twice, then release for the pulse
                # views. The delayed tracker and tissue keep moving in dynamic tapes.
                force = _SWEEP_FORCE_N[phase]
            else:
                pitch = 0.
                h = h_start
                # One overlapping longitudinal/transverse scout, then
                # three views at the supported anchor. The last view is
                # .36 s after arrival; longer tapes can scout both sides.
                if step % 4 == 0 and step + 1 < frames:
                    sweep = direction if (step // 4) % 2 == 0 else -direction
                    h += 3. * sweep
                    desired_view += 2 * sweep
                force = 2.4 if step == 0 else 3.8 + .7 * np.sin(2.7 * t)
            # Do not clamp the sweep to the source trace: when the finite
            # slice passes a true endpoint, the lumen can disappear.
            h = float(np.clip(h, _LO, _HI + 4.))
            target, contacts, costs = (target0, contacts0, costs0) if h == h_start else self._aim(h)
            angle_error = _wrap(_VIEW_ANGLES - _VIEW_ANGLES[desired_view % len(view_indices)])
            view = int(np.argmin(costs + 2. * (angle_error / .18)**2))
            if benchmark == 'retroactive' and h == h_start:
                view = desired_view  # preserve the prescribed full 45-degree rewindow
            rest_aperture = contacts[view]
            moved_window = (dwell_view is None or abs(h - dwell_view[0]) > 2.
                            or abs(_wrap(_VIEW_ANGLES[view] - dwell_view[1])) > .10
                            or abs(pitch - dwell_view[2]) > .06)
            if moved_window:
                dwell, dwell_view = (0. if station_scan or sweep_scan else dt), (h, _VIEW_ANGLES[view], pitch)
            else:
                dwell += dt
                if benchmark == 'retroactive' and not recheck:
                    dwell = min(dwell, .32)
            # Transport the neutral probe with the prescribed motion only.
            # Static tapes use the exact nominal contact, bit-independent of
            # which source knots survive the offset geometry intervention.
            probe_points = np.stack((rest_aperture, target)) if moving else fixed[:0]
            points = np.concatenate((rest, probe_points, fixed))
            world, orientations, _ = self.kinematics.forward_pose_canonical(truth.q, points)
            aperture, aimed = world[len(rest):len(rest) + 2] if moving else (rest_aperture, target)
            elevation = orientations[len(rest), :, 1] if moving else _ELEVATION
            rotation = _rock(_beam_frame(aimed - aperture, elevation), 0., pitch)
            # Blur depends on aperture motion relative to the moving tissue,
            # not their common rigid transport. The rewindow transit itself
            # is between acquisitions; the first acquired frame has settled.
            speed = (0. if previous_aperture is None or (benchmark == 'retroactive' and step == cutoff)
                     else float(np.linalg.norm(rest_aperture - previous_aperture) / dt))
            previous_aperture = rest_aperture
            pose = ProbePose(_read_only(aperture.copy()), _read_only(rotation), speed, dwell)
            # Lumen micromotion lives in neutral-world axes transported with
            # the tissue. It does not include the acoustic compression cue.
            response = motion.vessel_response(truth, force, pulse_displacement_mm=pulse_displacement,
                                              compression_compliance_mm_per_N=compression_displacement,
                                              flexion_sliding_mm_per_rad=sliding)
            if step:
                ou = motion.ou_step(ou, dt)
            shifted = world[:len(rest)] + np.einsum('nij,nj->ni', orientations[:len(rest)], (response + ou)[role_index])
            posed = np.full((*local_knots.shape, 3), np.nan)
            posed[ii, role_index] = shifted
            centers = _slice_centers(graph, rows, posed, pose, h, kernel.slice_half_width_mm)
            vessels = tuple(VesselTruth(centers[r], diameter[r], compliance[r], pulse[r], noise_key=tokens[r])
                            for r in np.flatnonzero(np.isfinite(centers[:, 0])))
            start_bone = len(rest) + len(probe_points)
            end_bone = start_bone + len(bone)
            fascia = world[end_bone:end_bone + len(patch)][faces] if use_shadow else None
            bone_triangles = world[start_bone:end_bone][bone_faces] if len(bone_faces) else None
            observation, echo_targets, echo_ghosts = kernel.render_supervised(
                UltrasoundTruth(vessels, bone_triangles_mm=bone_triangles,
                                fascia_triangles_mm=fascia, reflectors=reflectors),
                pose, force, step, roles,
            )
            if retro_echo_counts is not None:
                counts = retro_echo_counts['post_rewindow' if recheck else 'pre_rewindow']
                for targets, ghost in zip(echo_targets, echo_ghosts, strict=True):
                    lumen = bool(targets)
                    counts['ghost_echoes'] += ghost
                    counts['lumen_echoes'] += lumen
                    counts['mixed_ghost_lumen_echoes'] += ghost and lumen
            supervision = ScanSupervision(
                echo_targets, _read_only(centers), np.frombuffer(truth.q.tobytes(), dtype=truth.q.dtype),
                h, echo_ghosts,
            )
            tape.append(ScanFrame(observation, pose, float(force), _read_only(tracker.q_hat),
                                  _read_only(tracker.Sigma_q), dt, supervision))
        return ScanEpisode(source_id, station_id, tuple(tape), h_start, {
            'benchmark': benchmark,
            'source_role': 'train' if source_id in self._split.train_source_ids else 'evaluation',
            'geometry': geometry,
            'material_profile': (profile.name if benchmark != 'offset' and source_id not in self._evaluation_sources
                                 else None),
            'material_support_policy': ('archived source support' if source_id in self._evaluation_sources
                                        else 'unavailable chart corridors are not rendered as lumens'),
            'clinical_station_labels_available': False,
            'station_label_policy': 'requested tag on an overlapping Gate 2 source window, not a disjoint clinical class',
            'training_trunk_height_scope_mm': (float(lo), float(hi)),
            'acquisition': 'nominal training-trunk steering from neutral skin, blind given start height and seed',
            'sweep_cues': 'settled compression and low-force pulse dwell after a transverse sweep' if sweep_scan else None,
            'rewindow_deg': 45. if benchmark == 'retroactive' and frames > 1 else None,
            'rewindow_step': cutoff if retro_echo_counts is not None and frames > 1 else None,
            'echo_counts_by_rewindow': retro_echo_counts,
            'physics': 'synthetic echo kernel; not measured B-mode',
            'shadow': ('posed population bone mesh where available and optional synthetic fascia patch; '
                       'not patient-specific occlusion'),
            'reflector': 'fixed world-space synthetic specular proxy' if use_reflector else None,
            'offset_mm': float(offset_mm),
            'offset_policy': ('fixed material arc displacement of source anatomy only; nominal probe and tracker '
                              'inputs and acoustic noise keys paired across offsets') if benchmark == 'offset' else None,
        })
