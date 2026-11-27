"""Label-blind, sparse multi-hypothesis information filter for anonymous echoes."""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property, lru_cache
from itertools import combinations
from math import log, pi

import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit, logsumexp, softmax
from scipy.sparse.linalg import splu

from sim.anatomy_assets import VesselRole
from sim.physics_ultrasound_kernel import ProbePose, UltrasoundObservation
from sim.smplx_dynamic_atlas import (DynamicMotionProfile, _HIP, _KNEE, _ROT_VEC_FROM_Q, _local_rotation,
                                     _skin_stencil)

from .belief_ipe import BeliefIPE, ROLE_COUNT, wrap_angle
from .joint_association import joint_beam
from .station_classifier import topological_station_probabilities


NULL_ROLE = ROLE_COUNT
N_OUTPUT = ROLE_COUNT + 1
ARTERY = frozenset((int(VesselRole.MAIN_ARTERY), int(VesselRole.DFA)))
_SENSOR_VARIANCE = np.array([.8**2, 1.**2])
_OU_STD = np.array([.55, .85, 1.])
_PAIR = np.array(list(combinations(range(ROLE_COUNT), 2)), dtype=np.intp)
_LOG_2PI = log(2. * pi)
_RANK_ITERS = 3
_STEP_LIMIT = np.array([.30, 10., 1.])
_STOP_STEP = np.array([.13, 4., .40])
_BRANCH_LOG_WINDOW = 4.5
_ROLE_BIT = 1 << np.arange(ROLE_COUNT)
_MASK_MEMBERS = (np.arange(1 << ROLE_COUNT)[:, None] & _ROLE_BIT) != 0
_IDENTITY_MARGINAL = np.zeros((1 << ROLE_COUNT, N_OUTPUT))
_IDENTITY_MARGINAL[0, NULL_ROLE] = 1.
_IDENTITY_MARGINAL[1:, :ROLE_COUNT] = _MASK_MEMBERS[1:] / _MASK_MEMBERS[1:].sum(axis=1, keepdims=True)
_COMPRESSIVE = np.array([.24, .035])
_PULSE = np.array([.012, .12])


def joint_pullback_covariance(R_us, J_img_q_stack, Sigma_q):
    """Joint Theorem 2 covariance with a *shared* frame pose error.

    R_us is either a stacked square covariance or an array of per-echo 2×2
    blocks; J may be stacked as (2M,Q) or (M,2,Q).
    """
    j = np.asarray(J_img_q_stack)
    j = j.reshape(-1, j.shape[-1])
    r = np.asarray(R_us)
    if r.ndim == 3:
        blocks = np.zeros((len(r), 2, len(r), 2))
        blocks[np.arange(len(r)), :, np.arange(len(r)), :] = r
        r = blocks.reshape(2 * len(r), 2 * len(r))
    pullback = j @ np.asarray(Sigma_q) @ j.T
    return r + .5 * (pullback + pullback.T)


def _physical_scores(echoes, groups, excess_force, dwell_s, caliber_mean, caliber_std):
    """Broad shared caliber prior and morphology-class, not role-caliber, cues."""
    if not groups:
        return np.empty((len(echoes), 0))
    size = np.array([max(1, len(group)) for group in groups])
    arterial = np.array([sum(r in ARTERY for r in group) for group in groups])
    null = np.array([not group for group in groups])
    counts = np.column_stack((size - arterial, arterial))
    remaining = (1. - np.minimum(.95, excess_force * _COMPRESSIVE))**2
    area = counts @ remaining
    scale = np.where(null, 1., np.sqrt(area / size))
    ratio = scale - 1.
    calib_mu = caliber_mean * np.sqrt(size) * scale
    calib_sigma = np.sqrt(caliber_std**2 * size + .35**2)
    pulse = np.where(null, 0., (counts * remaining) @ _PULSE / area)
    cue_sd = np.stack((calib_sigma, .052 + .045 * excess_force + .065 * null,
                       np.where(null, .02, .022 + .025 * arterial / size)))
    cue_mu = np.stack((calib_mu, ratio, pulse))
    values = echoes[:, 2:, None]
    observed = np.isfinite(values) & np.array([True, excess_force > 0., dwell_s >= .35])[None, :, None]
    z = np.where(observed, (values - cue_mu) / cue_sd, 0.)
    return np.sum(-.5 * z**2 - np.where(observed, np.log(cue_sd), 0.), axis=1)


@dataclass(frozen=True)
class _Association:
    score: float
    groups: tuple[tuple[int, ...], ...]
    echoes: np.ndarray
    operator: np.ndarray  # matched group centroid projection of flat (role, image coord)
    residual: np.ndarray
    acoustic_score: float = 0.


def _score_associations(echoes, roles, image, covariance, visible, merge_distance, force, dwell,
                        caliber_mean, caliber_std, topk=8, *, allow_merge=True):
    """Bounded assignment beam then joint re-ranking (including pose cross terms)."""
    r = len(roles)
    if not len(echoes):
        return (_Association(float(np.log1p(-visible).sum()), (), np.empty(0, int),
                             np.zeros((0, 2 * r)), np.zeros(0)),)
    # Conditions on a resolved echo or two acoustic lumens. The centroid uses
    # exchangeable unknown baseline areas and expected force-dependent area.
    groups = [()] + [(int(role),) for role in roles]
    role_bits = [0] + [1 << i for i in range(r)]
    local = [()] + [(i,) for i in range(r)]
    mix = [()] + [(1.,) for _ in roles]
    excess = max(0., force - 2.5)
    merge_logp = [0.] * len(groups)
    if allow_merge and r >= 2:
        i, j = _PAIR[_PAIR[:, 1] < r].T
        delta = image[i] - image[j]
        dist = np.linalg.norm(delta, axis=1)
        blocks = covariance.reshape(r, 2, r, 2).transpose(0, 2, 1, 3)
        diff = blocks[i, i] + blocks[j, j] - blocks[i, j] - blocks[j, i]
        axis = delta / np.maximum(dist, 1.e-9)[:, None]
        spread = np.sqrt(np.maximum(np.einsum('ni,nij,nj->n', axis, diff, axis), .5))
        # A frame-wide pose error is not evidence of two lumens touching.
        contact = expit(1.65 * (merge_distance - dist) / spread)
        valid = contact >= .035
        edge = [(int(a), int(b), float(np.log(p))) for a, b, p in zip(i[valid], j[valid], contact[valid], strict=True)]
        trees = {}
        for a, b, score in edge:
            trees[(1 << a) | (1 << b)] = score
        # The kernel merges connected components. A maximum-product spanning
        # tree conservatively approximates the probability of connection.
        level = trees.copy()
        for _ in range(3, r + 1):
            grown = {}
            for mask, weight in level.items():
                for a, b, score in edge:
                    if bool(mask & (1 << a)) == bool(mask & (1 << b)):
                        continue
                    child = mask | (1 << a) | (1 << b)
                    if weight + score >= -4.5:
                        grown[child] = max(grown.get(child, -np.inf), weight + score)
            if not grown:
                break
            trees.update(grown)
            level = grown
        role_compliance = _COMPRESSIVE[[int(role in ARTERY) for role in roles]]
        for mask, score in trees.items():
            members = tuple(k for k in range(r) if mask & (1 << k))
            area = (1. - np.minimum(.95, excess * role_compliance[list(members)]))**2
            groups.append(tuple(int(roles[k]) for k in members))
            role_bits.append(mask)
            local.append(members)
            mix.append(area / area.sum())
            merge_logp.append(score)
    g = len(groups)
    transform = np.zeros((g, 2, 2 * r))
    odds = np.log(visible / (1. - visible))
    assignment_prior = np.empty(g)
    assignment_prior[0] = log(.42 / (50. * 77.))
    for k, members in enumerate(local[1:], 1):
        for i, weight in zip(members, mix[k], strict=True):
            transform[k, 0, 2 * i] = transform[k, 1, 2 * i + 1] = weight
        assignment_prior[k] = odds[list(members)].sum() + merge_logp[k]
    cue = _physical_scores(echoes, groups, excess, dwell, caliber_mean, caliber_std)
    acoustic = cue + np.r_[assignment_prior[0], merge_logp[1:]]
    base = cue + assignment_prior
    residual = echoes[:, None, :2] - transform[None, 1:] @ image.reshape(-1)
    full = transform.reshape(2 * g, 2 * r)
    joint = full @ covariance @ full.T
    paths, scores = joint_beam(base, residual, joint, np.asarray(role_bits, dtype=np.intp), topk)
    missed = np.log1p(-visible).sum()
    candidates = []
    for path, score in zip(paths, scores, strict=True):
        slots = np.flatnonzero(path)
        groups_used = path[slots]
        coordinates = (2 * groups_used[:, None] + np.arange(2)).ravel()
        candidates.append(_Association(float(score + missed), tuple(groups[j] for j in path), slots,
                                       full[coordinates], residual[slots, groups_used - 1].ravel(),
                                       float(acoustic[np.arange(len(path)), path].sum())))
    return tuple(candidates)


def _contributor_masks(groups):
    return tuple(sum(1 << role for role in group) for group in groups)


def _mask_marginals(weighted_masks, count):
    out = np.zeros((count, 1 << ROLE_COUNT))
    if count:
        slots = np.arange(count)
        for weight, masks in weighted_masks:
            out[slots, masks] += weight
    return out


def _identity_marginals(weighted_associations, count):
    weights = softmax([weight for weight, _ in weighted_associations])
    return _mask_marginals(((weight, _contributor_masks(association.groups))
                            for weight, (_, association) in zip(weights, weighted_associations, strict=True)), count)


def _mixture_moments(weights, means, covariances):
    points = np.stack(means)
    mean = np.einsum('b,b...i->...i', weights, points)
    delta = points - mean
    covariance = np.einsum('b,b...ij->...ij', weights, np.stack(covariances))
    covariance += np.einsum('b,b...i,b...j->...ij', weights, delta, delta)
    return mean, covariance


@dataclass(frozen=True)
class FrameEstimate:
    role_probabilities: np.ndarray
    centers_mm: np.ndarray
    image_centers_mm: np.ndarray
    image_covariance_mm2: np.ndarray
    station_probabilities: np.ndarray
    h_mm: float
    ipe: np.ndarray
    world_covariance_mm2: np.ndarray
    identity_probabilities: np.ndarray | None = None
    station_role_presence: np.ndarray | None = None
    station_landmark_offsets_mm: np.ndarray | None = None
    pose_mean_rad: np.ndarray | None = None
    pose_covariance_rad2: np.ndarray | None = None


@dataclass(frozen=True)
class _Frame:
    echoes: np.ndarray
    probe: ProbePose
    force: float
    q: np.ndarray
    pose_cov: np.ndarray
    h: float
    transforms: tuple | None = None


@dataclass(frozen=True)
class _Design:
    roles: np.ndarray
    nominal: np.ndarray
    slope: np.ndarray
    cols: np.ndarray
    operator: np.ndarray
    derivative: np.ndarray
    csc_indices: np.ndarray
    csc_indptr: np.ndarray


@dataclass
class _Branch:
    precision: sparse.csc_matrix
    factor: object
    information: np.ndarray
    mean: np.ndarray
    log_weight: float
    ou_mean: np.ndarray
    ou_cov: np.ndarray
    path: _PathFrame | None = None


@dataclass(frozen=True, slots=True)
class _PathFrame:
    """A retained assignment and its local filtered readout, never a global Gaussian."""

    masks: tuple[int, ...]
    log_evidence: float
    centers: np.ndarray
    world_cov: np.ndarray
    q: np.ndarray
    q_cov: np.ndarray
    ipe: np.ndarray
    presence: np.ndarray
    previous: _PathFrame | None


class _RankSolve:
    """Woodbury solve for the exact updated sparse precision."""

    def __init__(self, base, cols, h, u, innovation):
        self.base, self.cols, self.h, self.u, self.innovation = base, cols, h, u, innovation
        self.shape = base.shape
        self.depth = getattr(base, 'depth', 0) + 1

    def solve(self, rhs):
        result = self.base.solve(rhs)
        result -= self.u @ cho_solve(self.innovation, self.h @ result[self.cols], check_finite=False)
        return result


@dataclass(frozen=True)
class _Prediction:
    world: np.ndarray
    image: np.ndarray
    H_world: np.ndarray
    H: np.ndarray
    J_world_q: np.ndarray
    J_q: np.ndarray
    J_world_ou: np.ndarray
    J_ou: np.ndarray
    variance: np.ndarray
    visible: np.ndarray


@dataclass(frozen=True, slots=True)
class _Proposal:
    parent: int
    association: _Association
    masks: tuple[int, ...]
    observation: np.ndarray
    linearization: np.ndarray
    prediction: _Prediction
    mean: np.ndarray
    factor_data: tuple | None
    log_evidence: float
    objective: float


def _world_and_chart_jacobian(chart, mat):
    """Evaluate guarded radii once for both XYZ and dXYZ/d(theta,h,eta)."""
    xi = chart.xi_from_tilde_xi(mat).reshape(-1, 3)
    axis, slope, skin, bone, radii = chart._geometry(xi)
    u = np.column_stack((np.cos(xi[:, 0]), np.sin(xi[:, 0])))
    world = np.empty_like(xi)
    world[:, [0, 2]] = axis + radii[:, :1] * u
    world[:, 1] = xi[:, 1]
    jac = np.zeros((len(xi), 3, 3))
    jac[:, [0, 2], 0] = radii[:, :1] * np.column_stack((-u[:, 1], u[:, 0])) + radii[:, 1:2] * u
    jac[:, [0, 2], 1] = slope + radii[:, 2:3] * u
    jac[:, 1, 1] = 1.
    d_range = chart.d_max - chart.d_min
    s = (xi[:, 2] - chart.d_min) / d_range
    jac[:, [0, 2], 2] = (bone[:, :1] - skin[:, :1]) * u * (d_range * s * (1. - s))[:, None]
    return world, jac


class _NeutralSkin:
    """Exact neutral SMPL-X skinning reduced to its three distinct transforms."""

    def __init__(self, kinematics):
        self.rest = kinematics._neutral
        group = np.zeros(len(kinematics._parents), dtype=np.intp)
        active = {_HIP: 1, _KNEE: 2}
        for joint, parent in enumerate(kinematics._parents[1:], 1):
            group[joint] = active.get(joint, group[parent])
        self.weights = kinematics._weights @ np.eye(3)[group]

    def transforms(self, q):
        hip, knee = self.rest.joints_m[[_HIP, _KNEE]]
        rh, dh = _local_rotation(_ROT_VEC_FROM_Q[0], q)
        rk, dk = _local_rotation(_ROT_VEC_FROM_Q[1], q)
        r = rh @ rk
        d = dh @ rk + rh @ dk
        z = np.zeros_like(dh)
        return (np.stack((np.eye(3), rh, r)),
                np.stack((np.zeros(3), hip - rh @ hip, hip + rh @ (knee - hip) - r @ knee)),
                np.stack((z, dh, d)),
                np.stack((z[..., 0], -dh @ hip, dh @ (knee - hip) - d @ knee)))

    def deform(self, points_mm, transforms, *, pose_jacobian=True):
        points = points_mm * .001
        rot, trans, dr, dt = transforms
        ids, stencil, ds = _skin_stencil(self.rest.skin_tree, points, gradient=True)
        w_vertices = self.weights[ids]
        w = np.einsum('nk,nkj->nj', stencil, w_vertices)
        total = w.sum(axis=1, keepdims=True)
        w /= total
        dw = np.einsum('nkc,nkj->njc', ds, w_vertices)
        dw = (dw - w[..., None] * dw.sum(axis=1, keepdims=True)) / total[..., None]
        blend = np.einsum('nj,jrc->nrc', w, rot)
        offsets = np.einsum('jrc,nc->njr', rot - np.eye(3), points) + trans
        posed = (points + np.einsum('nj,njr->nr', w, offsets)) * 1000.
        gradient = blend + np.einsum('njr,njc->nrc', offsets, dw)
        jq = None
        if pose_jacobian:
            dq = (dr.transpose(0, 2, 1, 3).reshape(36, 3) @ points.T).T.reshape(-1, 3, 3, 4)
            jq = (np.einsum('nj,njrq->nrq', w, dq) + np.einsum('nj,jqr->nrq', w, dt)) * 1000.
        return posed, gradient, jq

    def invert_point(self, target, transforms):
        rot, trans, _, _ = transforms
        seeds = ((target - 1000. * trans)[:, None, :] @ rot)[:, 0, :]
        posed, gradient, _ = self.deform(seeds, transforms, pose_jacobian=False)
        norm = np.linalg.norm(posed - target, axis=1)
        # In overlapping projections, prefer a seed consistent with its rest
        # segment over the identity solution on the torso or contralateral leg.
        hip_h, knee_h = 1000. * self.rest.joints_m[[_HIP, _KNEE], 1]
        norm[0] += max(0., hip_h - seeds[0, 1] + 10.)
        norm[1] += max(0., knee_h - seeds[1, 1] - 20.)
        norm[2] += max(0., seeds[2, 1] - knee_h - 25.)
        best_rest, best_error = seeds[0], np.inf
        for best in np.argsort(norm):
            rest, residual, jac = seeds[best], target - posed[best], gradient[best]
            for _ in range(8):
                if residual @ residual < .02**2:
                    return rest
                dx = np.linalg.solve(jac, residual)
                rest = rest + dx * min(1., 20. / max(np.linalg.norm(dx), 1.e-9))
                estimate, gradient_at_rest, _ = self.deform(rest[None], transforms, pose_jacobian=False)
                residual, jac = target - estimate[0], gradient_at_rest[0]
            error = residual @ residual
            if error < best_error:
                best_rest, best_error = rest, error
            if error < .02**2:
                break
        return best_rest

    def invert_height(self, target, transforms):
        return float(self.invert_point(target, transforms)[1])

class DynamicFactorGraphReader:
    """Sparse iterated EIF with bounded discrete associations and local OU lag.

    Each discrete branch retains a static-anatomy Gaussian. The per-frame tissue
    block is propagated as a separate OU Gaussian and marginalized jointly with
    tracker error for the anatomical information factor. The carried OU block is
    moment-matched separately; cross-time anatomy–OU covariances are not retained.

    Discrete smoothing reweights retained assignment paths, not assignments
    recomputed against the terminal anatomy. For geometry, each path stores
    filtered frame-local moments; only their mixture weights are smoothed.
    """

    def __init__(self, prior, kinematics, *, disable_pullback=False, disable_discount=False, scorer=None,
                 rho_corr=.25, rho_novel=.85, caliber_mean_mm=6., caliber_std_mm=3., compute_ipe=False,
                 max_branches=5, topk=8, max_iterations=_RANK_ITERS):
        if prior.coordinate_mode != 'material':
            raise ValueError('dynamic factor graph requires material coordinates')
        if not (0 < rho_corr <= rho_novel <= 1 and 1 <= max_branches <= 6 and 1 <= topk <= 8
                and 1 <= max_iterations <= _RANK_ITERS and caliber_std_mm > 0.):
            raise ValueError('invalid reader hyperparameters')
        self.prior = prior
        self.kinematics = kinematics
        self._skin = _NeutralSkin(kinematics)
        self.disable_pullback, self.disable_discount = disable_pullback, disable_discount
        self.scorer = scorer
        self.rho_corr, self.rho_novel = rho_corr, rho_novel
        self.caliber_mean_mm, self.caliber_std_mm = caliber_mean_mm, caliber_std_mm
        self.compute_ipe = compute_ipe or scorer is not None
        self.max_branches, self.topk, self.max_iterations = max_branches, topk, max_iterations
        self._initial_precision = prior.prior_precision_csc()
        self._initial_factor = splu(self._initial_precision)
        self._initial_information = self._initial_precision @ prior.mean
        self._ou = DynamicMotionProfile(moving=False, ou_stationary_std_mm=_OU_STD)
        self._ou_initial_cov = np.diag(np.tile(_OU_STD**2, ROLE_COUNT))
        self._h_bounds = (float(prior.graph.h_mm.min()), float(prior.graph.h_mm.max()))
        op = prior.knot_design_matrix()
        self._knot_rows = tuple((op.indices[a:b], op.data[a:b]) for a, b in zip(op.indptr[:-1], op.indptr[1:]))
        self._design_basis = lru_cache(maxsize=128)(self._make_design_basis)
        landmarks = [prior.graph.vertices[prior.graph.junctions[name].vertex_id].h_mm
                     if name in prior.graph.junctions else np.nan for name in ('SFJ', 'CFA_bif', 'CFV_bif')]
        popliteal = getattr(prior, 'station_windows', {}).get('S5_POPLITEAL', ())
        self._landmarks = np.array([*landmarks, np.mean(popliteal) if len(popliteal) else np.nan])
        self._ou_cache = None
        self._nominal_knots = prior.nominal_knots
        self.reset()

    @cached_property
    def belief_ipe(self):
        return BeliefIPE(self.prior)

    def reset(self):
        self._branches = [_Branch(self._initial_precision, self._initial_factor, self._initial_information,
                                  self.prior.mean, 0., np.zeros(3 * ROLE_COUNT), self._ou_initial_cov)]
        self._history = []
        self._latest = None
        self._previous = None
        self._seeded = False
        self._station_window = None

    def _height(self, probe, q, transforms=None):
        # Probe origin is the public contact point. An oblique ray endpoint
        # selects a different height and may be on the wrong side of a joint.
        if not np.any(q):
            return float(np.clip(probe.origin_mm[1], *self._h_bounds))
        if transforms is None:
            transforms = self._skin.transforms(q)
        h = self._skin.invert_height(probe.origin_mm, transforms)
        return float(np.clip(h, *self._h_bounds))

    def _design(self, h):
        roles, terms, weights = [], [], []
        for role in range(ROLE_COUNT):
            bracket = self.prior.graph._station_query(h, role)
            if bracket is not None:
                a, b, t = bracket
                roles.append(role)
                terms.append((a, b))
                weights.append(t)
        role_ids, nominal, delta, slope, cols, op, diff, grad, indices, indptr = self._design_basis(tuple(roles), tuple(terms))
        t = np.asarray(weights)
        return _Design(role_ids, nominal + t[:, None] * delta, slope, cols, op + t[:, None, None] * diff, grad,
                       indices, indptr)

    def _make_design_basis(self, roles, terms):
        a, b = np.asarray(terms, dtype=np.intp).reshape(-1, 2).T
        nominal = self._nominal_knots[a]
        delta = self._nominal_knots[b] - nominal
        delta[:, 0] = wrap_angle(delta[:, 0])
        derivative = np.divide(1., delta[:, 1], out=np.zeros(len(a)), where=a != b)
        slope = delta * derivative[:, None]
        slope[a == b, 1] = 1.
        rows = [self._knot_rows[3*k+c][0] for x, y in terms for k in (x, y) for c in range(3)]
        cols = np.unique(np.concatenate(rows)) if rows else np.empty(0, dtype=np.intp)
        op = np.zeros((len(roles), 3, len(cols)))
        diff = np.zeros_like(op)
        for i, (left, right) in enumerate(terms):
            for j in range(3):
                ids, vals = self._knot_rows[3*left + j]
                ids = np.searchsorted(cols, ids)
                op[i, j, ids] = vals
                if left != right:
                    diff[i, j, ids] = -vals
                    ids, vals = self._knot_rows[3*right + j]
                    diff[i, j, np.searchsorted(cols, ids)] += vals
        grad = diff * derivative[:, None, None]
        dtype = self._initial_precision.indices.dtype
        indices = np.tile(cols.astype(dtype, copy=False), len(cols))
        indptr = (len(cols) * np.searchsorted(cols, np.arange(self._initial_precision.shape[1] + 1))).astype(dtype)
        return np.asarray(roles, dtype=np.intp), nominal, delta, slope, cols, op, diff, grad, indices, indptr

    def _predict_many(self, frame, design, branches):
        n = len(design.roles)
        if not n:
            p = _Prediction(np.zeros((0, 3)), np.zeros((0, 2)), np.zeros((0, len(design.cols))),
                            np.zeros((0, len(design.cols))), np.zeros((0, 4)), np.zeros((0, 4)),
                            np.zeros((0, 3*ROLE_COUNT)), np.zeros((0, 3*ROLE_COUNT)), np.zeros((0, 0)), np.zeros(0))
            return [p] * len(branches)
        means = np.stack([branch.mean[design.cols] for branch in branches])
        mat = design.nominal + np.einsum('ril,bl->bri', design.operator, means)
        dh, dtheta = self.prior._transport_height(mat, self.prior.chart)
        rest, j_rest = _world_and_chart_jacobian(self.prior.chart, mat)
        pose, dpose, jq = self._skin.deform(rest, frame.transforms, pose_jacobian=np.any(frame.pose_cov))
        world = pose.reshape(mat.shape)
        j_q = np.zeros(mat.shape[:-1] + (3, 4)) if jq is None else jq.reshape(mat.shape[:-1] + (3, 4))
        j_xi = (dpose @ j_rest).reshape(mat.shape[:-1] + (3, 3))
        long = j_xi[..., :, 1]
        long = long / np.linalg.norm(long, axis=-1, keepdims=True)
        angular = j_xi[..., :, 0] - (j_xi[..., :, 0] * long).sum(axis=-1, keepdims=True) * long
        angular /= np.linalg.norm(angular, axis=-1, keepdims=True)
        local = np.stack((long, angular, np.cross(long, angular)), axis=-1)
        j_xi[..., 0] += j_xi[..., 1] * dtheta[..., None]
        j_xi[..., 1] *= dh[..., None]
        tangent = (j_xi @ (design.slope + np.einsum('ril,bl->bri', design.derivative, means))[..., None])[..., 0]
        h_world = j_xi @ design.operator
        ou_index = (3 * design.roles[:, None] + np.arange(3)).ravel()
        mean_ou = np.stack([branch.ou_mean[ou_index].reshape(n, 3) for branch in branches])
        world += (local @ mean_ou[..., None])[..., 0]
        elevation = frame.probe.rotation[:, 1]
        plane_offset = (world - frame.probe.origin_mm) @ elevation
        plane_slope = tangent @ elevation
        intersect = np.abs(plane_slope) > .45
        shift = np.divide(-plane_offset, plane_slope, out=np.zeros_like(plane_offset), where=intersect)
        intersect &= np.abs(shift) < 40.
        # First-order slice intersection. Its implicit derivative removes
        # longitudinal motion along the local centerline, not cross-plane drift.
        world += tangent * np.clip(shift, -40., 40.)[..., None]
        pull = np.divide(tangent, plane_slope[..., None], out=np.zeros_like(tangent), where=intersect[..., None])
        projection = np.eye(3) - pull[..., :, None] * elevation
        shot = (world - frame.probe.origin_mm) @ frame.probe.rotation
        p_img = frame.probe.rotation[:, [0, 2]].T
        visible = (.82 * expit((25. - abs(shot[..., 0])) / 5.) * expit((shot[..., 2] - 3.) / 5.)
                   * expit((80. - shot[..., 2]) / 6.) * np.exp(-.5 * (shot[..., 1] / 8.)**2))
        visible = np.clip(visible, .015, .82)
        h_world = projection @ h_world
        j_q = projection @ j_q
        j_local_ou = projection @ local
        j_ou = np.zeros((len(branches), n, 3, 3 * ROLE_COUNT))
        for i, role in enumerate(design.roles):
            j_ou[:, i, :, 3*role:3*role+3] = j_local_ou[:, i]
        sensor = np.diag(np.tile(_SENSOR_VARIANCE + np.array([(frame.probe.speed_mm_s * .06)**2 / 12., 0.]), n))
        count = len(branches)
        h_image = (p_img @ h_world).reshape(count, 2*n, -1)
        pose = (p_img @ j_q).reshape(count, 2*n, 4)
        ou = (p_img @ j_ou).reshape(count, 2*n, -1)
        noise = sensor + ou @ np.stack([b.ou_cov for b in branches]) @ ou.swapaxes(-1, -2)
        if jq is not None:
            pullback = pose @ frame.pose_cov @ pose.swapaxes(-1, -2)
            noise += .5 * (pullback + pullback.swapaxes(-1, -2))
        image = shot[..., [0, 2]]
        return [_Prediction(world[i], image[i], h_world[i].reshape(3*n, -1), h_image[i],
                            j_q[i].reshape(3*n, 4), pose[i], j_ou[i].reshape(3*n, -1), ou[i], noise[i], visible[i])
                for i in range(count)]

    @staticmethod
    def _belief_rhs(factor, h, cols):
        rhs = np.zeros((factor.shape[0], len(h)), order='F')
        rhs[cols] = h.T
        return factor.solve(rhs) if len(h) else rhs

    @classmethod
    def _belief_rhs_many(cls, branches, predictions, cols):
        # Numerical seeds and trial linearizations can share one exact sparse
        # factor. Solve their right-hand sides together, then use column views.
        by_factor = {}
        for i, branch in enumerate(branches):
            by_factor.setdefault(id(branch.factor), []).append(i)
        solved = [None] * len(predictions)
        for group in by_factor.values():
            first = group[0]
            factor = branches[first].factor
            if len(group) == 1:
                solved[first] = cls._belief_rhs(factor, predictions[first].H, cols)
            else:
                block = cls._belief_rhs(factor, np.concatenate([predictions[i].H for i in group]), cols)
                for i, view in zip(group, np.split(block, len(group), axis=1), strict=True):
                    solved[i] = view
        return solved

    @classmethod
    def _project_belief(cls, factor, h, cols):
        covariance = h @ cls._belief_rhs(factor, h, cols)[cols]
        return .5 * (covariance + covariance.T)

    def _associate(self, frame, design, branch, pred, *, solved=None):
        if solved is None:
            solved = self._belief_rhs(branch.factor, pred.H, design.cols)
        projected = pred.H @ solved[design.cols]
        covariance = pred.variance + .5 * (projected + projected.T)
        return _score_associations(frame.echoes, design.roles, pred.image, covariance, pred.visible,
                                   4.7 + frame.probe.speed_mm_s * .06, frame.force, frame.probe.dwell_s,
                                   self.caliber_mean_mm, self.caliber_std_mm, self.topk)

    def _discount(self, frame):
        if self.disable_discount:
            return 1.
        prev = self._previous
        if prev is None:
            return self.rho_novel
        delta_tilt = frame.probe.rotation[:, 2] @ prev.probe.rotation[:, 2]
        novel = (delta_tilt <= np.cos(np.deg2rad(25.)) or abs(frame.h - prev.h) >= 15.
                 or (frame.force > 2.5 and abs(frame.force - prev.force) >= .5)
                 or (frame.probe.dwell_s >= .35 > prev.probe.dwell_s))
        return self.rho_novel if novel else self.rho_corr

    def _local_factor(self, parent, linearization, pred, design, association, observation, rho, solved):
        transform = association.operator
        present = np.any(transform[:, ::2], axis=0)
        # A new view provides independent non-detection evidence. A repeated
        # view can miss the same lumen for the same acoustic reason.
        missed_weight = 1. if self.disable_discount or rho == self.rho_novel else rho
        acoustic = (rho * (association.acoustic_score + np.log(pred.visible[present]).sum())
                    + missed_weight * np.log1p(-pred.visible[~present]).sum())
        if not len(observation):
            return parent.mean, None, acoustic, 0.
        h = transform @ pred.H
        r = transform @ pred.variance @ transform.T / rho
        cols = design.cols
        residual = observation - transform @ pred.image.ravel()
        u = solved @ transform.T
        projected = h @ u[cols]
        innovation = cho_factor(r + .5 * (projected + projected.T), lower=True, check_finite=False)
        chol = cho_factor(r, lower=True, check_finite=False)
        delta = linearization - parent.mean
        adjusted = residual + h @ delta[cols]
        shift = cho_solve(innovation, adjusted, check_finite=False)
        mean = parent.mean + u @ shift
        log_det_r = 2. * np.log(np.diag(chol[0])).sum()
        # Exact integrated power likelihood for this final tangent, including
        # the normalization of p(y|x)^rho (matters across match counts).
        log_evidence = (acoustic - .5 * (adjusted @ shift
                        + 2. * np.log(np.diag(innovation[0])).sum()
                        + (rho - 1.) * log_det_r + len(residual) * (rho * _LOG_2PI + rho * np.log(rho))))
        # Nonlinear MAP objective at the actual iterate, for a backtracking
        # trust step. The final retained precision is always undamped.
        objective = (delta @ (parent.precision @ delta)
                     + residual @ cho_solve(chol, residual, check_finite=False) + log_det_r)
        return mean, (h, r, u, innovation, residual, chol, solved), log_evidence, objective

    @staticmethod
    def _update(parent, linearization, pred, design, transform, mean, factor_data, log_weight, rho):
        if factor_data is None:
            return _Branch(parent.precision, parent.factor, parent.information, parent.mean,
                           log_weight, parent.ou_mean, parent.ou_cov)
        h, r, u, innovation, residual, chol, _ = factor_data
        ht_rinv = cho_solve(chol, h, check_finite=False).T
        cols = design.cols
        # CSC structure depends only on the interpolating knot interval.
        # Materialize the block directly in column order without a COO sort.
        block = (ht_rinv @ h).ravel(order='F')
        local = sparse.csc_matrix((block, design.csc_indices, design.csc_indptr), shape=parent.precision.shape)
        precision = parent.precision + local
        information = parent.information.copy()
        target = residual + h @ linearization[cols]
        information[cols] += ht_rinv @ target
        # Bound the length of the rank-update chain; compact to the exact CSC
        # factor periodically instead of refactoring every retained branch.
        factor = (splu(precision) if getattr(parent.factor, 'depth', 0) >= 7
                  else _RankSolve(parent.factor, cols, h, u, innovation))
        z, c = parent.ou_mean, parent.ou_cov
        j_ou = transform @ pred.J_ou
        cross_z = c @ j_ou.T
        k = rho * cho_solve(chol, cross_z.T, check_finite=False).T
        z = z + k @ (residual - h @ (mean[cols] - linearization[cols]))
        observed_cov = h @ u[cols]
        post_observed = observed_cov - observed_cov @ cho_solve(innovation, observed_cov, check_finite=False)
        c = c - k @ cross_z.T + k @ post_observed @ k.T
        c = .5 * (c + c.T)
        return _Branch(precision, factor, information, mean, log_weight, z, c)

    def _record_frame(self, frame, design, parent, branch, proposal, rho, presence_rate):
        a, pred = proposal.association, proposal.prediction
        conditioned = self._condition_world(frame, design, parent, pred, a, proposal.linearization,
                                            proposal.factor_data, rho)
        if conditioned is None:
            # No match: this proposal was evaluated at its unshifted parent.
            world = pred.world
            covariance = (self._project_belief(parent.factor, pred.H_world, design.cols)
                          + pred.J_world_q @ frame.pose_cov @ pred.J_world_q.T
                          + pred.J_world_ou @ parent.ou_cov @ pred.J_world_ou.T)
            q, q_cov = frame.q, frame.pose_cov
        else:
            world, covariance, q, q_cov = conditioned
        ids = 3 * np.arange(len(design.roles))[:, None] + np.arange(3)
        ipe = self.belief_ipe.features(frame.h, branch.mean, factor=branch.factor) if self.compute_ipe else np.empty(0)
        presence = self._accumulate_presence(proposal.masks, parent.path, presence_rate)
        return _PathFrame(proposal.masks, proposal.log_evidence, world, covariance[ids[..., None], ids[:, None, :]],
                          q, q_cov, ipe, presence, parent.path)

    def _frame_estimate(self, frame, roles, paths, weights):
        identities = _mask_marginals(((w, path.masks) for w, path in zip(weights, paths, strict=True)),
                                    len(frame.echoes))
        role_probs = identities @ _IDENTITY_MARGINAL
        centers = np.full((ROLE_COUNT, 3), np.nan)
        image = np.full((ROLE_COUNT, 2), np.nan)
        imcov = np.full((ROLE_COUNT, 2, 2), np.nan)
        wcov = np.full((ROLE_COUNT, 3, 3), np.nan)
        if len(roles):
            world, world_cov = _mixture_moments(weights, [p.centers for p in paths], [p.world_cov for p in paths])
            centers[roles], wcov[roles] = world, world_cov
            p = frame.probe.rotation[:, [0, 2]]
            image[roles] = (world - frame.probe.origin_mm) @ p
            imcov[roles] = p.T @ world_cov @ p
        q_post, q_cov = _mixture_moments(weights, [p.q for p in paths], [p.q_cov for p in paths])
        ipe = weights @ np.stack([p.ipe for p in paths])
        presence = weights @ np.stack([p.presence for p in paths])
        station = topological_station_probabilities(
            self.prior, frame.h, role_probs, identity_probabilities=identities, accumulated_presence=presence[:N_OUTPUT])
        relative_h = frame.h - self._landmarks
        if self.scorer is not None:
            identities, station = self.scorer.score_joint(
                frame.echoes, identities, ipe, frame.h, station, frame.force, frame.probe.dwell_s,
                station_role_presence=presence, station_landmark_offsets_mm=relative_h)
            role_probs = identities @ _IDENTITY_MARGINAL
        return FrameEstimate(role_probs, centers, image, imcov, station, frame.h, ipe, wcov,
                             identities, presence, relative_h, q_post, q_cov)

    @staticmethod
    def _condition_world(frame, design, prior, prediction, association, linearization, data=None, rho=1.):
        """Exact conditional pose and world moments at this association tangent."""
        t = association.operator
        if not len(t):
            return None
        if data is None:
            h = t @ prediction.H
            r = t @ prediction.variance @ t.T
            solved_img = DynamicFactorGraphReader._belief_rhs(prior.factor, prediction.H, design.cols)
            u = solved_img @ t.T
            residual = association.residual
        else:
            h, r, u, _, residual, _, solved_img = data
            r = r * rho  # rho discounts static anatomy learning, not this frame's localization.
        projected = h @ u[design.cols]
        innovation = cho_factor(r + .5 * (projected + projected.T), lower=True, check_finite=False)
        pose = frame.pose_cov @ (t @ prediction.J_q).T
        ou = prior.ou_cov @ (t @ prediction.J_ou).T
        h_world, q_world, z_world = prediction.H_world, prediction.J_world_q, prediction.J_world_ou
        # Reuse the image-plane covariance solve. Only non-intersecting
        # centerlines can carry a nonzero elevation component.
        p = frame.probe.rotation[:, [0, 2]]
        e = frame.probe.rotation[:, 1]
        local = solved_img[design.cols].reshape(len(design.cols), -1, 2) @ p.T
        elevation = e @ h_world.reshape(-1, 3, len(design.cols))
        off_plane = np.max(np.abs(elevation), axis=1, initial=0.) > 1.e-10
        if off_plane.any():
            lift = DynamicFactorGraphReader._belief_rhs(prior.factor, elevation[off_plane], design.cols)
            local[:, off_plane] += lift[design.cols, :, None] * e
        world_cov = (h_world @ local.reshape(len(design.cols), -1) + q_world @ frame.pose_cov @ q_world.T
                     + z_world @ prior.ou_cov @ z_world.T)
        cross = h_world @ u[design.cols] + q_world @ pose + z_world @ ou
        delta = linearization[design.cols] - prior.mean[design.cols]
        rhs = np.column_stack((residual + h @ delta, cross.T, pose.T))
        solved = cho_solve(innovation, rhs, check_finite=False)
        world = prediction.world + (cross @ solved[:, 0] - h_world @ delta).reshape(-1, 3)
        covariance = world_cov - cross @ solved[:, 1:-4]
        pose_cov = frame.pose_cov - pose @ solved[:, -4:]
        return world, .5 * (covariance + covariance.T), frame.q + pose @ solved[:, 0], .5 * (pose_cov + pose_cov.T)

    @staticmethod
    def _accumulate_presence(masks, parent_path, rate):
        # Accumulate along each path before taking its posterior expectation:
        # online and replay then use the same prefix-based station features.
        seen, null = 0, False
        for mask in masks:
            seen |= mask
            null |= mask == 0
        present = np.r_[_MASK_MEMBERS[seen], null].astype(float)
        if rate is None:
            return np.tile(present, 2)
        maximum, average = parent_path.presence.reshape(2, N_OUTPUT)
        return np.r_[np.maximum(maximum, present), average + rate * (present - average)]

    def _initial_modes(self, frame, design, parent):
        seeds = [parent]  # Keep the unshifted prior as well as the probe-centered search.
        if self.max_branches == 1:
            return seeds
        # Public acoustic aperture and beam depth give a neutral azimuth
        # seed without assuming that the requested scan site is a vessel.
        target = frame.probe.origin_mm + np.median(frame.echoes[:, 1]) * frame.probe.rotation[:, 2]
        rest = self._skin.invert_point(target, frame.transforms) if np.any(frame.q) else target
        axis = self.prior.chart._axis(frame.h)
        offset = rest[[0, 2]] - axis
        theta_probe = np.arctan2(offset[1], offset[0])
        trunk = design.roles < 2
        angles = design.nominal[trunk, 0] if trunk.any() else design.nominal[:, 0]
        theta_prior = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
        arc = float(wrap_angle(theta_probe - theta_prior) * self.prior.chart.nominal_reference_radius(frame.h))
        for offset in (0., -15., 15., -30., 30.):
            shift = arc + offset
            if shift != 0.:
                mean = parent.mean.copy()
                mean[self.prior.layout.shared][::6] += shift
                seeds.append(replace(parent, mean=mean))
        return seeds

    def _propose(self, frame, design, parents, seeds, rho):
        states = [seed for _, seed in seeds]
        predictions = self._predict_many(frame, design, states)
        solved = self._belief_rhs_many(states, predictions, design.cols)
        prelim = []
        for j, ((i, seed), pred, rhs) in enumerate(zip(seeds, predictions, solved, strict=True)):
            for a in self._associate(frame, design, seed, pred, solved=rhs):
                if seed is not parents[i] and not len(a.echoes):
                    # An all-missed/NULL assignment has no geometry to support
                    # a search tangent other than the parent predictive mean.
                    continue
                prelim.append((seed.log_weight + rho * a.score, i, _contributor_masks(a.groups), j, a))
        prelim.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))
        proposals, seen = [], set()
        for weight, i, masks, j, a in prelim:
            key = (i, masks)
            if key in seen:
                continue
            # Seed points are numerical starts, not independent path mass.
            # Identical masks from a different parent remain distinct histories.
            seen.add(key)
            lin = states[j].mean
            observation = frame.echoes[a.echoes, :2].ravel()
            pred = predictions[j]
            mean, data, evidence, objective = self._local_factor(
                parents[i], lin, pred, design, a, observation, rho, solved[j])
            proposals.append(_Proposal(i, a, masks, observation, lin, pred, mean, data, evidence, objective))
            if len(proposals) >= self.max_branches + 2 or (len(proposals) >= 3 and weight < prelim[0][0] - 6.):
                break
        self._refine(frame, design, parents, proposals, rho)
        return proposals

    def _refine(self, frame, design, parents, proposals, rho):
        # Every iterate is linearized against the same parent Gaussian.
        cols = design.cols
        for _ in range(self.max_iterations - 1):
            queue, states = [], []
            for j, p in enumerate(proposals):
                if p.factor_data is None:
                    continue
                dx = p.mean - p.linearization
                local = design.operator @ dx[cols]
                if np.all(np.abs(local) <= _STOP_STEP):
                    continue
                fraction = min(1., 1. / np.max(np.abs(local) / _STEP_LIMIT))
                queue.append((j, fraction))
                states.append(replace(parents[p.parent], mean=p.linearization + fraction * dx))
            if not queue:
                break
            for backtrack in range(2):
                preds = self._predict_many(frame, design, states)
                solved = self._belief_rhs_many(states, preds, cols)
                retry, retry_states = [], []
                for (j, fraction), state, pred, rhs in zip(queue, states, preds, solved, strict=True):
                    p = proposals[j]
                    parent = parents[p.parent]
                    mean, data, evidence, objective = self._local_factor(
                        parent, state.mean, pred, design, p.association, p.observation, rho, rhs)
                    if objective < p.objective:
                        proposals[j] = replace(p, linearization=state.mean, prediction=pred, mean=mean,
                                               factor_data=data, log_evidence=evidence, objective=objective)
                    elif backtrack == 0 and fraction > .15:
                        fraction *= .5
                        retry.append((j, fraction))
                        retry_states.append(replace(parent, mean=p.linearization + fraction * (p.mean - p.linearization)))
                if not retry:
                    break
                queue, states = retry, retry_states

    def step(self, obs: UltrasoundObservation, probe_pose: ProbePose, f_contact, q_hat, Sigma_q, dt):
        q = np.array(q_hat, dtype=float, copy=True)
        pose_cov = np.zeros((4, 4)) if self.disable_pullback else np.array(Sigma_q, dtype=float, copy=True)
        transforms = self._skin.transforms(q)
        frame = _Frame(np.array(obs.echoes, dtype=float, copy=True), probe_pose, float(f_contact), q, pose_cov,
                       self._height(probe_pose, q, transforms), transforms)
        if self._ou_cache is None or self._ou_cache[0] != dt:
            decay, process = self._ou.ou_transition(dt)
            self._ou_cache = (dt, decay, np.kron(np.eye(ROLE_COUNT), process))
        _, decay, process = self._ou_cache
        parents = [replace(b, ou_mean=decay * b.ou_mean, ou_cov=decay**2 * b.ou_cov + process) for b in self._branches]
        design = self._design(frame.h)
        rho = self._discount(frame)
        first_usable = not self._seeded and len(frame.echoes) > 0 and len(design.roles) > 0
        seeds = ([(i, s) for i, parent in enumerate(parents) for s in self._initial_modes(frame, design, parent)]
                 if first_usable else list(enumerate(parents)))
        proposals = self._propose(frame, design, parents, seeds, rho)
        # Cancel a shared log increment before adding parent weights. If the
        # suffix has one equally likely extension per path, it adds no evidence
        # about their relative ancestry probabilities.
        offset = max(p.log_evidence for p in proposals)
        candidates = [(parents[p.parent].log_weight + (p.log_evidence - offset), p) for p in proposals]
        candidates.sort(key=lambda row: (-row[0], row[1].parent, row[1].masks))
        retained = [(w, p) for w, p in candidates[:self.max_branches]
                    if w >= candidates[0][0] - _BRANCH_LOG_WINDOW]
        norm = logsumexp([w for w, _ in retained])
        if self._station_window is None or abs(frame.h - self._station_window[0]) > 12.:
            station_window, presence_rate = (frame.h, 1.), None
        else:
            anchor, weight = self._station_window
            station_window, presence_rate = (anchor, weight + rho), rho / (weight + rho)
        branches = []
        for weight, p in retained:
            parent = parents[p.parent]
            branch = self._update(parent, p.linearization, p.prediction, design, p.association.operator,
                                  p.mean, p.factor_data, weight - norm, rho)
            branch.path = self._record_frame(frame, design, parent, branch, p, rho, presence_rate)
            branches.append(branch)
        weights = softmax([b.log_weight for b in branches])
        estimate = self._frame_estimate(frame, design.roles, [b.path for b in branches], weights)
        self._branches = branches
        self._seeded |= first_usable
        self._station_window = station_window
        self._latest = estimate
        self._previous = frame
        self._history.append((frame, design.roles))
        return estimate

    def relabel_history(self):
        """Read past identities from the terminal posterior over retained paths."""
        weights = softmax([b.log_weight for b in self._branches])
        paths = [b.path for b in self._branches]
        estimates = []
        for frame, roles in reversed(self._history):
            estimates.append(self._frame_estimate(frame, roles, paths, weights))
            paths = [path.previous for path in paths]
        estimates.reverse()
        return tuple(estimates)

    def get_belief_summary(self):
        if self._latest is None:
            return None
        frame, roles = self._history[-1]
        latest = self._latest
        weights = softmax([b.log_weight for b in self._branches])
        ordered = np.sort(weights)[::-1]
        residual = np.stack([b.path.centers for b in self._branches]) - latest.centers_mm[roles]
        disagreement = np.full((ROLE_COUNT, 3, 3), np.nan)
        disagreement[roles] = np.einsum('b,bri,brj->rij', weights, residual, residual)
        ipe = latest.ipe
        if not ipe.size:
            ipe = weights @ np.stack([self.belief_ipe.features(frame.h, b.mean, factor=b.factor) for b in self._branches])
        return dict(posterior_entropy=float(-weights @ np.log(np.maximum(weights, np.finfo(float).tiny))),
                    top2_branch_margin=float(ordered[0] - (ordered[1] if len(ordered) > 1 else 0.)),
                    branch_probabilities=weights, centers_mm=latest.centers_mm,
                    centers_covariance_mm2=latest.world_covariance_mm2, belief_ipe=ipe,
                    spatial_disagreement_mm2=disagreement, station_probabilities=latest.station_probabilities)

    def expected_info_gain(self, probe_pose, q_hat, Sigma_q, f_contact=2.5):
        """Visibility-weighted local linear Gaussian information in nats."""
        q = np.asarray(q_hat, dtype=float)
        c = np.zeros((4, 4)) if self.disable_pullback else np.asarray(Sigma_q, dtype=float)
        transforms = self._skin.transforms(q)
        frame = _Frame(np.empty((0, 5)), probe_pose, float(f_contact), q, c,
                       self._height(probe_pose, q, transforms), transforms)
        design = self._design(frame.h)
        if not len(design.roles):
            return 0.
        weights = softmax([b.log_weight for b in self._branches])
        predictions = self._predict_many(frame, design, self._branches)
        gain = 0.
        for branch, pred, w in zip(self._branches, predictions, weights, strict=True):
            factor = np.repeat(np.sqrt(pred.visible), 2)
            signal = self._project_belief(branch.factor, factor[:, None] * pred.H, design.cols)
            gain += .5 * w * (np.linalg.slogdet(pred.variance + self._discount(frame) * signal)[1]
                              - np.linalg.slogdet(pred.variance)[1])
        # Mixture spread contributes to the anticipated geometry gain; the
        # conditional branch variance above excludes between-mode information.
        common = sum(w * p.image for w, p in zip(weights, predictions, strict=True))
        for w, p in zip(weights, predictions, strict=True):
            error = (np.sqrt(p.visible[:, None]) * (p.image - common)).ravel()
            gain += .5 * w * np.log1p(error @ np.linalg.solve(p.variance, error))
        return float(max(gain, 0.))
