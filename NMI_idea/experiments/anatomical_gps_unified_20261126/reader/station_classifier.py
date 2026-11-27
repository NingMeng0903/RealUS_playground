"""Train-only posterior calibration and joint certificates for sampled scan segments."""

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
from numbers import Integral
from pathlib import Path
from typing import Hashable

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from scipy.stats import beta
import torch
from torch import nn
from torch.nn import functional as F

from sim.anatomy_assets import HOLDOUT_SOURCE_ID, NYSORA_STATIONS, AnatomySplit


STATION_IDS = tuple(NYSORA_STATIONS)
_ROLE_COUNT = 6
_IDENTITY_WIDTH = 1 << _ROLE_COUNT
_ROLE_WIDTH = _ROLE_COUNT + 1
_STATION_WIDTH = len(STATION_IDS)
_STATION_PRESENCE_WIDTH = 2 * _ROLE_WIDTH
_STATION_LANDMARK_WIDTH = 4
_STATION_CONTEXT_WIDTH = _STATION_PRESENCE_WIDTH + _STATION_LANDMARK_WIDTH + _STATION_WIDTH
_ECHO_WIDTH = 10
_CKPT_VERSION = 5
_CUE_SCALE = np.array([.04, .04, .125, 1., 4.], dtype=np.float32)
_BRANCH_ROLES = (2, 3, 5)
_BRANCH_DETECTION = np.array([[.65 if role in NYSORA_STATIONS[name] else .10 for role in _BRANCH_ROLES]
                              for name in STATION_IDS])
_BRANCH_MEMBERSHIP = (np.arange(1 << _ROLE_COUNT)[:, None] & (1 << np.array(_BRANCH_ROLES))) != 0
_HEIGHT_STD_MM = 10.

# Predeclared observability group: at least 4:1 posterior odds on every echo
# and a two-standard-deviation major axis within the physical error budget.
_IDENTITY_CONFIDENCE = .8
_OBSERVABILITY_RADIUS = 2.
_STRATUM_NAMES = ('other', 'well_observed')


def topological_station_probabilities(prior, h_mm, role_probabilities, *, identity_probabilities=None,
                                      accumulated_presence=None) -> np.ndarray:
    """Score train-supported station windows and distinguishing branch detections."""
    windows = getattr(prior, 'station_windows', None)
    if windows is None:
        from .split_prior import station_windows

        windows = station_windows(prior.graph)
    logits = np.full(_STATION_WIDTH, -np.inf)
    for idx, station in enumerate(STATION_IDS):
        heights = np.asarray(windows.get(station, ()))
        if heights.size:
            # Marginalize a uniform supported landmark height blurred by
            # tracker/slice error. Do not favor a wider window merely because
            # it has more knots or take one closest knot as conclusive.
            logits[idx] = (logsumexp(-.5 * ((heights - h_mm) / _HEIGHT_STD_MM)**2) - math.log(heights.size)
                           if np.isfinite(h_mm) else 0.)
    supported = np.isfinite(logits)
    if not supported.any():
        return np.zeros(_STATION_WIDTH)

    roles = np.asarray(role_probabilities)
    if len(roles) or accumulated_presence is not None:
        if accumulated_presence is not None:
            presence = np.asarray(accumulated_presence)[list(_BRANCH_ROLES)]
        elif identity_probabilities is None:
            # Legacy readers lack a joint assignment posterior. Do not
            # multiply uncertain or duplicated marginal detections.
            presence = roles[:, _BRANCH_ROLES].max(axis=0)
        else:
            identities = _normalized_identity_probabilities(identity_probabilities, len(roles))
            # The joint association uses each role at most once; summing its
            # mask probabilities recovers the chance it was seen, even in a merge.
            presence = np.clip(identities.sum(axis=0) @ _BRANCH_MEMBERSHIP, 0., 1.)
        # Missing branches are common; seeing a distinctive branch is more
        # diagnostic than failing to see it. Nearby windows can share branches.
        logits += np.log(_BRANCH_DETECTION * presence + (1. - _BRANCH_DETECTION) * (1. - presence)).sum(axis=1)
    logits[supported] = np.maximum(logits[supported] - logits[supported].max(), -80.)
    return softmax(logits)


def _authority(split: AnatomySplit) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    train, test = tuple(sorted(split.train_source_ids)), tuple(sorted(split.test_source_ids))
    if (not train or HOLDOUT_SOURCE_ID in train or set(train) & set(test)
            or len(set(train)) != len(train) or len(set(test)) != len(test)):
        raise ValueError('invalid training/evaluation source split')
    return train, test, split.relationship


def _episode_key(source_id: str, episode_id: Hashable) -> str:
    # Portable, type-preserving identifiers in the checkpoint, including int64 seeds.
    def canonical(value):
        if isinstance(value, str):
            return value
        if isinstance(value, Integral):
            return int(value)
        if isinstance(value, tuple):
            return tuple(canonical(part) for part in value)
        raise TypeError('episode IDs must be strings, integers or tuples thereof')

    return hashlib.sha256(repr((source_id, canonical(episode_id))).encode()).hexdigest()


def _features(echoes, ipe, station_probabilities, *, station_role_presence=None, station_landmark_offsets_mm=None):
    x = np.asarray(getattr(echoes, 'echoes', echoes), dtype=np.float32)
    stations = np.asarray(station_probabilities, dtype=np.float32)
    ipe = np.asarray(ipe, dtype=np.float32).reshape(-1)
    if x.ndim != 2 or x.shape[1] != 5 or stations.shape != (_STATION_WIDTH,):
        raise ValueError('expected anonymous (M,5) echoes and (5,) station probabilities')
    # Station-only context is separate from the relative IPE shared by both heads.
    presence_end = len(ipe) + _STATION_PRESENCE_WIDTH
    context = np.zeros(len(ipe) + _STATION_CONTEXT_WIDTH, dtype=np.float32)
    context[:len(ipe)] = np.where(np.isfinite(ipe), ipe, 0.)
    if station_role_presence is not None:
        presence = np.asarray(station_role_presence, dtype=np.float32)
        if presence.shape != (_STATION_PRESENCE_WIDTH,):
            raise ValueError('expected seven local role-presence maxima followed by seven means')
        context[len(ipe):presence_end] = presence
    if station_landmark_offsets_mm is not None:
        offsets = np.asarray(station_landmark_offsets_mm, dtype=np.float32)
        if offsets.shape != (_STATION_LANDMARK_WIDTH,):
            raise ValueError('expected axial offsets from SFJ, CFA_bif, CFV_bif and popliteal')
        context[presence_end:-_STATION_WIDTH] = np.where(np.isfinite(offsets), .01 * offsets, 0.)
    context[-_STATION_WIDTH:] = stations
    present = np.isfinite(x)
    cues = np.where(present, x * _CUE_SCALE, 0.)
    features = np.empty((len(x), _ECHO_WIDTH), dtype=np.float32)
    features[:, :5], features[:, 5:] = cues, present
    return context, features


def _echo_label(roles) -> int:
    label = 0
    for role in roles:
        if not isinstance(role, Integral) or not 0 <= role < _ROLE_COUNT:
            raise ValueError('echo supervision requires role IDs from 0 through 5')
        label |= 1 << int(role)
    return label


# Canonical exact contributor sets: ID is a bit mask; ID 0 is pure NULL.
ECHO_IDENTITIES = tuple(tuple(role for role in range(_ROLE_COUNT) if label & (1 << role))
                        for label in range(_IDENTITY_WIDTH))


def _normalized_identity_probabilities(values, echo_count):
    mass = np.asarray(values, dtype=float)
    if mass.shape != (echo_count, _IDENTITY_WIDTH) or not np.isfinite(mass).all() or (mass < 0.).any():
        raise ValueError('expected nonnegative exact contributor probabilities with shape (M,64)')
    totals = mass.sum(axis=-1, keepdims=True)
    if not np.isfinite(totals).all() or (totals <= 0.).any():
        raise ValueError('every echo needs positive total exact contributor probability')
    return mass / totals


@dataclass(frozen=True)
class TrainingRecord:
    source_id: str
    episode_id: Hashable
    context: np.ndarray
    echoes: np.ndarray
    graph_identities: np.ndarray
    identity_targets: np.ndarray
    station_target: int


def make_training_record(source_id, episode_id, frame, estimate, station_id=None) -> TrainingRecord:
    """Pair graph-only public features with simulation labels outside inference."""
    context, echoes = _features(frame.obs, estimate.ipe, estimate.station_probabilities,
                                station_role_presence=estimate.station_role_presence,
                                station_landmark_offsets_mm=estimate.station_landmark_offsets_mm)
    if len(frame.echo_targets) != len(echoes):
        raise ValueError('each echo must have a contributor label')
    labels = np.fromiter((_echo_label(roles) for roles in frame.echo_targets), dtype=np.intp, count=len(echoes))
    station_target = -1 if station_id is None else STATION_IDS.index(station_id)
    graph = _normalized_identity_probabilities(estimate.identity_probabilities, len(echoes))
    return TrainingRecord(source_id, episode_id, context, echoes, graph, labels, station_target)


class _ResidualMLP(nn.Module):
    def __init__(self, context_width: int, identity_prior, beam_backoff):
        super().__init__()
        self.register_buffer('identity_prior', torch.as_tensor(identity_prior, dtype=torch.float32))
        self.register_buffer('beam_backoff', torch.as_tensor(beam_backoff, dtype=torch.float32))
        self.context = nn.Sequential(nn.Linear(context_width - _STATION_CONTEXT_WIDTH, 64), nn.SiLU(),
                                     nn.Linear(64, 64), nn.SiLU(), nn.Dropout(.1))
        self.echo = nn.Sequential(nn.Linear(64 + _ECHO_WIDTH + _IDENTITY_WIDTH, 128), nn.SiLU(),
                                  nn.Linear(128, _IDENTITY_WIDTH))
        self.station = nn.Sequential(nn.Linear(64 + _STATION_CONTEXT_WIDTH, 64), nn.SiLU(),
                                     nn.Linear(64, _STATION_WIDTH))
        for head in (self.echo[-1], self.station[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, context, echoes, graph):
        hidden = self.context(context[:, :-_STATION_CONTEXT_WIDTH])
        per_echo = hidden[:, None, :].expand(-1, echoes.shape[1], -1)
        # Beam truncation is not evidence that an omitted set is impossible.
        # The fit-only mixture is a reference for residual logits, not an
        # output probability floor.
        reference = (1. - self.beam_backoff) * graph + self.beam_backoff * self.identity_prior
        identity_logits = reference.log() + self.echo(torch.cat((per_echo, echoes, graph), dim=-1))
        station_base = context[:, -_STATION_WIDTH:].clamp_min(1.e-8).log()
        station_context = torch.cat((hidden, context[:, -_STATION_CONTEXT_WIDTH:]), dim=-1)
        station_logits = station_base + self.station(station_context)
        return identity_logits, station_logits


class NeuralStationScorer:
    """A frozen CPU scorer; calibration temperatures come from independent train dev episodes."""

    def __init__(self, model, context_width, identity_temperature, station_temperature, authority, fitted_episodes):
        self._model = model.to('cpu').eval().requires_grad_(False)
        self._context_width = context_width
        self.identity_temperature = float(identity_temperature)
        self.station_temperature = float(station_temperature)
        self._authority = authority
        self._fitted_episodes = frozenset(fitted_episodes)

    @torch.inference_mode()
    def score_joint(self, echoes, exact_identity_probabilities, ipe, h_mm, station_probabilities, force, dwell,
                    *, station_role_presence=None, station_landmark_offsets_mm=None):
        """Return normalized mask and station probabilities from public observations and beliefs."""
        context, x = _features(echoes, ipe, station_probabilities, station_role_presence=station_role_presence,
                               station_landmark_offsets_mm=station_landmark_offsets_mm)
        if len(context) != self._context_width:
            raise ValueError('IPE feature width does not match this checkpoint')
        graph = _normalized_identity_probabilities(exact_identity_probabilities, len(x)).astype(np.float32)
        identity, station = self._model(torch.from_numpy(context)[None], torch.from_numpy(x)[None],
                                        torch.from_numpy(graph)[None])
        identity = torch.softmax(identity[0].double() / self.identity_temperature, dim=-1).numpy()
        supported = context[-_STATION_WIDTH:] > 0.
        if supported.any():
            logits = station[0].numpy() / self.station_temperature
            station = softmax(np.where(supported, logits, -np.inf))
        else:
            station = np.zeros(_STATION_WIDTH, dtype=np.float32)
        return identity, station

    def save(self, path, *, certifier=None, metadata=None) -> None:
        if certifier is not None and (certifier._authority != self._authority
                                      or self._fitted_episodes & certifier._calibration_episodes):
            raise ValueError('certifier authority or calibration episodes conflict with the scorer')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'version': _CKPT_VERSION, 'authority': self._authority,
                    'context_width': self._context_width, 'model': self._model.state_dict(),
                    'identity_temperature': self.identity_temperature, 'station_temperature': self.station_temperature,
                    'fitted_episodes': sorted(self._fitted_episodes),
                    'certifier': None if certifier is None else certifier.state_dict(),
                    'metadata': {} if metadata is None else dict(metadata)}, path)

    @classmethod
    def load(cls, path, split) -> 'NeuralStationScorer':
        return load_station_checkpoint(path, split)[0]


def _training_episodes(records, allowed_sources):
    if any(record.source_id not in allowed_sources for record in records):
        raise ValueError('calibration accepts only split.train_source_ids')
    return {_episode_key(record.source_id, record.episode_id) for record in records}


def _training_batch(records, device):
    context = np.stack([record.context for record in records])
    max_echoes = max(len(record.echoes) for record in records)
    echoes = np.zeros((len(records), max_echoes, _ECHO_WIDTH), dtype=np.float32)
    graph = np.zeros((len(records), max_echoes, _IDENTITY_WIDTH), dtype=np.float32)
    identity_target = np.full((len(records), max_echoes), -1, dtype=np.int64)
    station_target = np.full(len(records), -1, dtype=np.int64)
    for idx, record in enumerate(records):
        count = len(record.echoes)
        labels = np.asarray(record.identity_targets)
        if (labels.shape != (count,) or not np.issubdtype(labels.dtype, np.integer)
                or (labels < 0).any() or (labels >= _IDENTITY_WIDTH).any()):
            raise ValueError('each supervised echo needs one legal exact contributor mask')
        echoes[idx, :count] = record.echoes
        graph[idx, :count] = _normalized_identity_probabilities(record.graph_identities, count)
        identity_target[idx, :count] = labels
        if record.station_target != -1:
            if (not 0 <= record.station_target < _STATION_WIDTH
                    or record.context[-_STATION_WIDTH + record.station_target] <= 0.):
                raise ValueError('supervised station has no verified window in the training prior')
            station_target[idx] = record.station_target
    if not ((identity_target >= 0).any() and (station_target >= 0).any()):
        raise ValueError('fit and temperature partitions each need exact echo and station supervision')
    return tuple(torch.as_tensor(value, device=device) for value in (
        context, echoes, graph, identity_target, station_target))


def _fit_identity_reference(records):
    counts = np.ones(_IDENTITY_WIDTH)
    omitted = n = 0
    for record in records:
        labels = record.identity_targets
        n += len(labels)
        counts += np.bincount(labels, minlength=_IDENTITY_WIDTH)
        omitted += np.count_nonzero(record.graph_identities[np.arange(len(labels)), labels] == 0.)
    # Symmetric Dirichlet prior over legal masks and Beta(1,1) prior over
    # beam omission keep the finite training estimate strictly full-support.
    return counts / counts.sum(), (omitted + 1) / (n + 2)


def _temperature(logits, targets) -> float:
    valid = targets >= 0
    logits, targets = logits[valid].astype(float), targets[valid]
    observed = logits[np.arange(len(targets)), targets]
    # Optimize inverse temperature: cross-entropy is convex in this scalar.
    def nll(scale):
        return float((logsumexp(scale * logits, axis=-1) - scale * observed).mean())

    scale = minimize_scalar(nll, method='bounded', bounds=(.05, 20.), options={'xatol': 1.e-5}).x
    # The bounded optimizer need not evaluate the endpoints or the no-op.
    return 1. / min((.05, scale, 1., 20.), key=nll)


@torch.inference_mode()
def _fit_temperatures(model, dev, batch_size):
    context, x, graph, identity_target, station_target = dev
    identity_logits, station_logits = [], []
    for start in range(0, len(context), batch_size):
        batch = slice(start, start + batch_size)
        identity, station = model(context[batch], x[batch], graph[batch])
        identity_logits.append(identity.cpu().numpy())
        station_logits.append(station.masked_fill(context[batch, -_STATION_WIDTH:] <= 0., -torch.inf).cpu().numpy())
    return (_temperature(np.concatenate(identity_logits), identity_target.cpu().numpy()),
            _temperature(np.concatenate(station_logits), station_target.cpu().numpy()))


def fit_station_scorer(fit_records, temperature_records, split, *, device='cpu', seed=0,
                       epochs=24, batch_size=256) -> NeuralStationScorer:
    """Fit both residual heads, then tune one temperature per head on disjoint train dev episodes."""
    authority = _authority(split)
    fit_records, temperature_records = tuple(fit_records), tuple(temperature_records)
    fit_ids = _training_episodes(fit_records, authority[0])
    dev_ids = _training_episodes(temperature_records, authority[0])
    if not fit_records or not temperature_records or fit_ids & dev_ids:
        raise ValueError('fit and temperature data require nonempty, disjoint episodes')
    if epochs < 1 or batch_size < 1:
        raise ValueError('epochs and batch size must be positive')
    train = _training_batch(fit_records, device)
    identity_prior, beam_backoff = _fit_identity_reference(fit_records)
    dev = _training_batch(temperature_records, device)
    width = train[0].shape[1]
    if dev[0].shape[1] != width:
        raise ValueError('fit and temperature feature widths differ')
    fit_device = train[0].device
    rng_devices = [] if fit_device.type == 'cpu' else [fit_device]
    with torch.random.fork_rng(devices=rng_devices, device_type=fit_device.type if rng_devices else 'cuda'):
        torch.random.default_generator.manual_seed(seed)
        if rng_devices:
            device_rng = torch.Generator(device=fit_device).manual_seed(seed)
            torch.get_device_module(fit_device).set_rng_state(device_rng.get_state(), fit_device)
        model = _ResidualMLP(width, identity_prior, beam_backoff).to(fit_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2.e-3, weight_decay=.05)
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            order = torch.as_tensor(rng.permutation(len(fit_records)), device=fit_device)
            for start in range(0, len(order), batch_size):
                rows = order[start:start + batch_size]
                cx, ex, gx, it, st = (part[rows] for part in train)
                identity, station = model(cx, ex, gx)
                station = station.masked_fill(cx[:, -_STATION_WIDTH:] <= 0., -1.e6)
                loss = F.cross_entropy(identity.flatten(0, 1), it.flatten(), ignore_index=-1, reduction='sum')
                loss = loss / (it >= 0).sum().clamp_min(1)
                loss = loss + (F.cross_entropy(station, st, ignore_index=-1, reduction='sum')
                               / (st >= 0).sum().clamp_min(1))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
    identity_t, station_t = _fit_temperatures(model.eval(), dev, batch_size)
    return NeuralStationScorer(model, width, identity_t, station_t, authority, fit_ids | dev_ids)


@dataclass(frozen=True)
class _FramePrediction:
    identity_scores: np.ndarray
    identity_confidence: np.ndarray
    centers_mm: np.ndarray
    axes: np.ndarray
    scales_mm: np.ndarray
    valid_geometry: np.ndarray


@dataclass(frozen=True)
class SegmentPrediction:
    frames: tuple[_FramePrediction, ...]


@dataclass(frozen=True)
class _FrameTruth:
    identities: np.ndarray
    centers_mm: np.ndarray


@dataclass(frozen=True)
class SegmentRecord:
    source_id: str
    episode_id: Hashable
    prediction: SegmentPrediction
    truth: tuple[_FrameTruth, ...]


def _frame_prediction(estimate) -> _FramePrediction:
    probabilities = _normalized_identity_probabilities(
        estimate.identity_probabilities, len(estimate.role_probabilities))
    identity_confidence = probabilities.max(axis=-1)
    # Closed log odds against MAP: ties score zero, and a true zero scores
    # +inf. One nat and one standardized geometry radius each cost one unit
    # in the joint maximum. No finite cutoff admits a zero-probability set.
    log_probabilities = np.full_like(probabilities, -np.inf)
    np.log(probabilities, out=log_probabilities, where=probabilities > 0.)
    identity_scores = np.log(identity_confidence)[:, None] - log_probabilities
    centers = np.array(estimate.image_centers_mm, dtype=float, copy=True)
    cov = np.asarray(estimate.image_covariance_mm2, dtype=float)
    if centers.shape != (_ROLE_COUNT, 2) or cov.shape != (_ROLE_COUNT, 2, 2):
        raise ValueError('certificate needs six image centers and 2D covariances')
    valid = np.isfinite(centers).all(axis=-1) & np.isfinite(cov).all(axis=(1, 2))
    axes = np.zeros_like(cov)
    scales = np.full((_ROLE_COUNT, 2), np.inf)
    ids = np.flatnonzero(valid)
    if len(ids):
        sub = cov[ids]
        values, directions = np.linalg.eigh(.5 * (sub + sub.transpose(0, 2, 1)))
        psd = values[:, 0] >= -1.e-8
        valid[ids[~psd]] = False
        axes[ids[psd]] = directions[psd]
        # A 1 micron minimum scale prevents division by zero for degenerate
        # tracked points. Membership and verification use the same ellipse.
        scales[ids[psd]] = np.sqrt(np.maximum(values[psd], 1.e-6))
    return _FramePrediction(identity_scores, identity_confidence, centers, axes, scales, valid)


def make_segment_prediction(estimates) -> SegmentPrediction:
    """Prepare the truth-free inputs to the certificate from reader outputs alone."""
    frames = tuple(_frame_prediction(estimate) for estimate in estimates)
    if not frames:
        raise ValueError('segment must contain at least one frame')
    return SegmentPrediction(frames)


def make_segment_record(source_id, episode_id, frames, estimates) -> SegmentRecord:
    """Attach simulator geometry and exact contributor IDs for calibration or audit only."""
    prediction = make_segment_prediction(estimates)
    truth = []
    for frame, predicted in zip(frames, prediction.frames, strict=True):
        identities = np.fromiter((_echo_label(roles) for roles in frame.echo_targets), dtype=np.intp)
        if len(identities) != len(predicted.identity_scores):
            raise ValueError('predictive echo rows and truth labels differ')
        pose = frame.probe_pose
        centers = (np.asarray(frame.centers_mm) - pose.origin_mm) @ pose.rotation[:, (0, 2)]
        truth.append(_FrameTruth(identities, centers))
    return SegmentRecord(source_id, episode_id, prediction, tuple(truth))


def _visible_roles(identities) -> np.ndarray:
    mask = int(np.bitwise_or.reduce(identities, initial=0))
    return np.fromiter(ECHO_IDENTITIES[mask], dtype=np.intp)


def _well_observed(prediction: SegmentPrediction, tolerance_mm: float) -> bool:
    has_lumen = False
    for frame in prediction.frames:
        if (frame.identity_confidence < _IDENTITY_CONFIDENCE).any():
            return False
        roles = _visible_roles(frame.identity_scores.argmin(axis=-1))
        if len(roles):
            has_lumen = True
            if (not frame.valid_geometry[roles].all()
                    or np.any(frame.scales_mm[roles, -1] > tolerance_mm / _OBSERVABILITY_RADIUS)):
                return False
    return has_lumen


def _identity_nonconformity(frame: _FramePrediction, nominal: bool) -> np.ndarray:
    if nominal:
        # Preserve the uncalibrated comparator: p >= alpha at its default
        # Gaussian radius. Recover -log(p) from the same exact posterior.
        return np.sqrt(2. * (frame.identity_scores - np.log(frame.identity_confidence)[:, None]))
    return frame.identity_scores


def segment_nonconformity(record: SegmentRecord, *, nominal=False) -> float:
    """Maximum standardized geometry and log MAP odds over the whole tape.

    Every contributor to every rendered echo is covered, including each
    member of a merge. Missing estimated geometry is an unrestricted plane,
    not a successful geometric estimate; the verifier rejects it.
    """
    score = 0.
    for frame, truth in zip(record.prediction.frames, record.truth, strict=True):
        ids = truth.identities
        identity_score = _identity_nonconformity(frame, nominal)[np.arange(len(ids)), ids]
        score = max(score, identity_score.max(initial=0.))
        roles = _visible_roles(ids)
        if not np.isfinite(truth.centers_mm[roles]).all():
            raise ValueError('observed lumen has no finite supervised center')
        roles = roles[frame.valid_geometry[roles]]
        if len(roles):
            delta = truth.centers_mm[roles] - frame.centers_mm[roles]
            standardized = np.einsum('ni,nij->nj', delta, frame.axes[roles]) / frame.scales_mm[roles]
            radius = np.hypot(standardized[:, 0], standardized[:, 1]).max()
            score = max(score, radius)
    return float(score)


@dataclass(frozen=True)
class FramePredictiveSet:
    echo_candidates: np.ndarray  # (M,64) boolean; columns ECHO_IDENTITIES
    centers_mm: np.ndarray
    ellipse_axes: np.ndarray
    ellipse_radii_mm: np.ndarray
    valid_geometry: np.ndarray


@dataclass(frozen=True)
class SegmentPredictiveSet:
    frames: tuple[FramePredictiveSet, ...]
    unbounded: bool


def _risk(failures, trials, confidence):
    # One-sided Clopper–Pearson upper limit; no estimate for an empty sample.
    upper = None if not trials else (
        1. if failures == trials else float(beta.ppf(confidence, failures + 1, trials - failures)))
    return {'failures': failures, 'trials': trials, 'rate': failures / trials if trials else None,
            'upper_bound': upper, 'confidence': confidence}


def _alpha(value):
    value = float(value)
    if not 0. < value < 1.:
        raise ValueError('alpha must be strictly between zero and one')
    return value


@dataclass(frozen=True)
class SegmentCertifier:
    cutoff: float
    tolerance_mm: float
    alpha: float
    calibration_segments: int
    rank: int | None
    _authority: tuple | None = None
    _calibration_episodes: frozenset[str] = frozenset()
    _strata: tuple[tuple[int, float, int], ...] = ()  # order statistic rank, cutoff, calibration count
    _nominal: bool = False

    def __post_init__(self):
        _alpha(self.alpha)
        if not (self.cutoff >= 0. and np.isfinite(self.tolerance_mm) and self.tolerance_mm > 0.):
            raise ValueError('certificate cutoff must be nonnegative and tolerance finite and positive')

    def state_dict(self) -> dict:
        return {'cutoff': self.cutoff, 'tolerance_mm': self.tolerance_mm, 'alpha': self.alpha,
                'calibration_segments': self.calibration_segments, 'rank': self.rank,
                '_authority': self._authority, '_calibration_episodes': sorted(self._calibration_episodes),
                '_strata': self._strata, '_nominal': self._nominal}

    def _stratum(self, prediction: SegmentPrediction) -> int:
        return int(_well_observed(prediction, self.tolerance_mm))

    def _prediction_cutoff(self, prediction: SegmentPrediction) -> float:
        return self._strata[self._stratum(prediction)][1] if self._strata else self.cutoff

    def predict_set(self, prediction: SegmentPrediction) -> SegmentPredictiveSet:
        """Closed set of admissible exact identities and contributor ellipses.

        An assignment imposes no center constraint on roles it does not use.
        Missing geometry is the full plane for that role; a maximal cutoff
        makes the entire segment set unrestricted.
        """
        cutoff = self._prediction_cutoff(prediction)
        unbounded = math.isinf(cutoff)
        frames = []
        for frame in prediction.frames:
            radii = np.full_like(frame.scales_mm, np.inf)
            valid = frame.valid_geometry
            if not unbounded:
                radii[valid] = cutoff * frame.scales_mm[valid]
            candidates = _identity_nonconformity(frame, self._nominal) <= cutoff
            frames.append(FramePredictiveSet(candidates, frame.centers_mm, frame.axes, radii, valid))
        return SegmentPredictiveSet(tuple(frames), unbounded)

    def verify(self, predicted_set: SegmentPredictiveSet) -> bool:
        """Universal check of identity and center error, without access to simulator truth."""
        if predicted_set.unbounded:
            return False
        has_lumen = False
        for frame in predicted_set.frames:
            possible = frame.echo_candidates
            if np.any(possible.sum(axis=1) != 1):
                return False  # abstain on an empty or ambiguous identity factor
            roles = _visible_roles(possible.argmax(axis=1))
            if len(roles):
                has_lumen = True
                # The largest semi-axis is the farthest point of this entire
                # ellipse from the committed center, not a sampled boundary.
                if (not frame.valid_geometry[roles].all()
                        or np.any(frame.ellipse_radii_mm[roles] > self.tolerance_mm)):
                    return False
        return has_lumen

    def outcomes(self, records) -> np.ndarray:
        """Rows are independent audit units; columns: miscovered, falsely certified, certified."""
        records = tuple(records)
        keys = {_episode_key(record.source_id, record.episode_id) for record in records}
        if len(keys) != len(records) or keys & self._calibration_episodes:
            raise ValueError('audit units must be unique and independent of conformal calibration units')
        outcomes = np.zeros((len(records), 3), dtype=bool)
        for idx, record in enumerate(records):
            score = segment_nonconformity(record, nominal=self._nominal)
            miscovered = score > self._prediction_cutoff(record.prediction)
            predicted = self.predict_set(record.prediction)
            certified = self.verify(predicted)
            false_certification = False
            if certified:
                for frame, truth in zip(predicted.frames, record.truth, strict=True):
                    ids = truth.identities
                    roles = _visible_roles(ids)
                    errors = np.linalg.norm(truth.centers_mm[roles] - frame.centers_mm[roles], axis=-1)
                    if (not frame.echo_candidates[np.arange(len(ids)), ids].all()
                            or not np.all(errors <= self.tolerance_mm)):
                        false_certification = True
                        break
            outcomes[idx] = miscovered, false_certification, certified
        return outcomes

    def audit(self, records=None, *, outcomes=None, confidence=.95) -> dict:
        """Independent tape audit, including risk among accepted tapes as a diagnostic."""
        if not 0. < confidence < 1.:
            raise ValueError('confidence must be strictly between zero and one')
        if (records is None) == (outcomes is None):
            raise ValueError('pass either segment records or precomputed audit outcomes')
        if records is not None:
            records = tuple(records)
        result = self.outcomes(records) if outcomes is None else np.asarray(outcomes, dtype=bool)
        if result.ndim != 2 or result.shape[1] != 3:
            raise ValueError('audit outcomes must have shape (N,3)')

        unbounded = math.isinf(self.cutoff)
        rank_bound = None if self.rank is None else _rank_violation_bound(self.calibration_segments, self.rank)
        strata = {}
        if self._strata:
            groups = None if records is None else np.fromiter((self._stratum(r.prediction) for r in records), int)
            for idx, (name, (rank, cutoff, count)) in enumerate(zip(_STRATUM_NAMES, self._strata, strict=True)):
                strata[name] = {'cutoff': None if math.isinf(cutoff) else cutoff, 'cutoff_unbounded': math.isinf(cutoff),
                                'calibration_segments': count, 'rank': rank,
                                'rank_violation_bound': _rank_violation_bound(count, rank)}
                if groups is not None:
                    strata[name].update(_audit_counts(result[groups == idx], confidence))
            rank_bound = max(stratum['rank_violation_bound'] for stratum in strata.values())
        return {'cutoff': None if unbounded else self.cutoff, 'cutoff_unbounded': unbounded,
                'alpha': self.alpha, 'calibration_segments': self.calibration_segments, 'rank': self.rank,
                'tolerance_mm': self.tolerance_mm, 'rank_violation_bound': rank_bound, 'strata': strata,
                **_audit_counts(result, confidence),
                'audit_unit': 'independent scan segment, not independent anatomy',
                'certification_scope': 'exact contributor identities and geometry of rendered echoes, not target capture',
                'joint_score': ('max standardized geometry and sqrt(-2 log p(identity))'
                                if self._nominal else 'max standardized geometry and log(p(MAP)/p(identity))'),
                'rank_coverage_scope': ('exchangeable segments within calibration strata, not conditional on acceptance'
                                        if rank_bound is not None else None)}


def _rank_violation_bound(count, rank):
    return (count + 1 - rank) / (count + 1)


def _audit_counts(result, confidence):
    miscovered, false_certified, accepted = (int(count) for count in result.sum(axis=0))
    n = len(result)
    return {'audit_segments': n, 'certified_segments': accepted, 'acceptance_rate': accepted / n if n else None,
            'miscoverage': _risk(miscovered, n, confidence),
            'false_certification': _risk(false_certified, n, confidence),
            'conditional_failure': _risk(false_certified, accepted, confidence)}


def conformal_rank_cutoff(scores, *, alpha=.2) -> tuple[int, float]:
    """Return the closed-set order statistic, or +inf when the rank exceeds n."""
    alpha = _alpha(alpha)
    values = np.fromiter(scores, dtype=float)
    if not (values >= 0.).all():
        raise ValueError('nonconformity scores must be nonnegative')
    n = len(values)
    rank = math.ceil((n + 1) * (1 - Fraction(str(alpha))))
    if rank > n:
        return rank, math.inf
    return rank, float(np.partition(values, rank - 1)[rank - 1])


def fit_segment_certifier(records, split, *, alpha=.2, tolerance_mm=8., scorer=None) -> SegmentCertifier:
    """Fit each predeclared observability stratum on disjoint training-source tapes.

    Rank coverage holds in each stratum under segment exchangeability, hence
    marginally as well. It is not conditional on passing the verifier.
    """
    records = tuple(records)
    if not records:
        raise ValueError('conformal calibration requires at least one supervised segment')
    authority, alpha = _authority(split), _alpha(alpha)
    episodes = _training_episodes(records, authority[0])
    if len(episodes) != len(records):
        raise ValueError('each calibration record must represent a distinct segment')
    if scorer is not None and (authority != scorer._authority or episodes & scorer._fitted_episodes):
        raise ValueError('conformal calibration must be independent of scorer fit and temperature calibration')
    groups = ([], [])
    for record in records:
        groups[_well_observed(record.prediction, tolerance_mm)].append(segment_nonconformity(record))
    strata = tuple((*conformal_rank_cutoff(group, alpha=alpha), len(group)) for group in groups)
    # The top-level cutoff summarizes bounded strata only; actual cutoffs
    # (including unrestricted underpowered strata) are in state and the audit.
    cutoff = max((q for _, q, _ in strata if np.isfinite(q)), default=math.inf)
    return SegmentCertifier(cutoff, float(tolerance_mm), alpha, len(records), None, authority, frozenset(episodes), strata)


def uncalibrated_segment_certifier(*, tolerance_mm=8., alpha=.2, cutoff=None) -> SegmentCertifier:
    """Nominal Gaussian radial cutoff and per-echo probability cutoff, without a rank claim."""
    alpha = _alpha(alpha)
    cutoff = math.sqrt(-2. * math.log(alpha)) if cutoff is None else float(cutoff)
    return SegmentCertifier(cutoff, float(tolerance_mm), alpha, 0, None, _nominal=True)


def load_station_checkpoint(path, split) -> tuple[NeuralStationScorer, SegmentCertifier | None, dict]:
    """Load a frozen CPU scorer and optional certifier, rejecting mismatched source authority."""
    authority = _authority(split)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['version'] != _CKPT_VERSION or payload['authority'] != authority:
        raise ValueError('checkpoint version or source split does not match the requested reader')
    width = payload['context_width']
    state = payload['model']
    model = _ResidualMLP(width, state['identity_prior'], state['beam_backoff'])
    model.load_state_dict(state)
    scorer = NeuralStationScorer(model, width, payload['identity_temperature'], payload['station_temperature'],
                                  authority, payload['fitted_episodes'])
    state = payload['certifier']
    certifier = None
    if state is not None:
        state['_calibration_episodes'] = frozenset(state['_calibration_episodes'])
        certifier = SegmentCertifier(**state)
        if certifier._authority != authority or certifier._calibration_episodes & scorer._fitted_episodes:
            raise ValueError('checkpoint certifier has incompatible authority or overlaps scorer training')
    return scorer, certifier, payload['metadata']
