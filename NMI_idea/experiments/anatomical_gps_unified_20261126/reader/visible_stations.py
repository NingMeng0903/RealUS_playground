"""Public multi-label readout for visible topological vessel combinations.

No scan requests, source IDs, support masks or simulator labels enter inference.
An observation of a role in one frame cannot complete a combination in another.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize
from scipy.special import expit

from sim.anatomy_assets import NYSORA_STATIONS, THIGH_NYSORA_Y_BOUNDS_MM, VesselRole


STATIONS = tuple(NYSORA_STATIONS)
DECISION_THRESHOLD = .5
_ROLE_SETS = tuple(tuple(sorted(NYSORA_STATIONS[name])) for name in STATIONS)
_UNIQUE_ROLE_SETS = tuple(dict.fromkeys(_ROLE_SETS))
_SINGLETON_MASKS = 1 << np.arange(len(VesselRole))
_COMBO_COLUMNS = tuple((_SINGLETON_MASKS[list(roles)], tuple(i for i, group in enumerate(_ROLE_SETS) if group == roles))
                       for roles in _UNIQUE_ROLE_SETS)
_POLICY = 'frame-injective-exact-singletons-v1'
_HEIGHT_STEP_MM = 10.
_HEIGHT_BANDWIDTH_MM = 10.
_LO, _HI = THIGH_NYSORA_Y_BOUNDS_MM
_HEIGHT_CENTERS_MM = np.arange(_LO, _HI + _HEIGHT_STEP_MM, _HEIGHT_STEP_MM)
_L2 = 1.
_INTERCEPT_PSEUDOCOUNT_PER_OUTCOME = .5
_POSE_WIDTH = len(_HEIGHT_CENTERS_MM)
_WIDTHS = {'pose_only': _POSE_WIDTH, 'visible': 2 * _POSE_WIDTH + 2}


def frame_combo_scores(identity_probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best injective singleton match (geometric mean) and exact-MAP match per combination.

    The input columns are *exact* contributor masks, never role marginals.
    The soft score is a matching feature, not a joint posterior probability:
    echo identities need not be conditionally independent.
    """
    probs = np.asarray(identity_probabilities, dtype=float)
    if (probs.ndim != 2 or probs.shape[1] != 1 << len(VesselRole) or not np.isfinite(probs).all()
            or (probs < 0.).any() or not np.allclose(probs.sum(axis=1), 1.)):
        raise ValueError('expected one exact contributor-mask distribution per echo')
    scores = np.zeros(len(STATIONS))
    map_matches = np.zeros(len(STATIONS), dtype=bool)
    ids = probs.argmax(axis=1)
    for masks, stations in _COMBO_COLUMNS:
        if len(probs) < len(masks):
            continue
        singleton = probs[:, masks]
        # The finite sentinel is more costly than any complete assignment of
        # positive float probabilities and also permits infeasible matrices.
        weights = np.full_like(singleton, -1.e6)
        np.log(singleton, out=weights, where=singleton > 0.)
        slots, roles = linear_sum_assignment(weights, maximize=True)
        if (singleton[slots, roles] > 0.).all():
            scores[list(stations)] = np.exp(weights[slots, roles].mean())
        map_matches[list(stations)] = np.isin(masks, ids).all()
    return scores, map_matches


def _height_basis(heights_mm) -> np.ndarray:
    h = np.asarray(heights_mm, dtype=float)
    if h.ndim != 1 or not len(h) or not np.isfinite(h).all():
        raise ValueError('expected a nonempty vector of finite public pose-estimated heights')
    return np.exp(-.5 * ((h[:, None] - _HEIGHT_CENTERS_MM) / _HEIGHT_BANDWIDTH_MM)**2)


def pose_features(heights_mm) -> np.ndarray:
    """Comparator features: only heights estimated from public probe and tracker pose."""
    return np.broadcast_to(_height_basis(heights_mm).max(axis=0), (len(STATIONS), _POSE_WIDTH))


def visible_features(heights_mm, frame_identity_probabilities) -> np.ndarray:
    """Pool complete frame-local witnesses by max, including height-conditioned matches."""
    basis = _height_basis(heights_mm)
    scores, map_matches = zip(*(frame_combo_scores(probs) for _, probs in
                                zip(basis, frame_identity_probabilities, strict=True)), strict=True)
    scores = np.stack(scores)
    features = np.empty((len(STATIONS), _WIDTHS['visible']))
    features[:, :_POSE_WIDTH] = basis.max(axis=0)
    features[:, _POSE_WIDTH] = scores.max(axis=0)
    features[:, _POSE_WIDTH + 1] = np.any(map_matches, axis=0)
    features[:, _POSE_WIDTH + 2:] = (scores[..., None] * basis[:, None, :]).max(axis=0)
    return features


def _config() -> dict:
    return {'feature_protocol': _POLICY, 'station_ids': list(STATIONS), 'threshold': DECISION_THRESHOLD,
            'height_centers_mm': _HEIGHT_CENTERS_MM.tolist(), 'height_bandwidth_mm': _HEIGHT_BANDWIDTH_MM,
            'slope_l2': _L2, 'intercept_pseudocount_per_outcome': _INTERCEPT_PSEUDOCOUNT_PER_OUTCOME}


def _fit_logistic(features, labels) -> np.ndarray:
    """Identical convex Bernoulli fit for both arms, including all-zero classes."""
    def objective(params):
        bias, slopes = params[0], params[1:]
        logits = bias + features @ slopes
        residual = expit(logits) - labels
        prior = _INTERCEPT_PSEUDOCOUNT_PER_OUTCOME * (np.logaddexp(0., bias) + np.logaddexp(0., -bias))
        loss = np.logaddexp(0., np.where(labels, -logits, logits)).sum() + .5 * _L2 * (slopes @ slopes) + prior
        gradient = np.r_[residual.sum() + _INTERCEPT_PSEUDOCOUNT_PER_OUTCOME * (2. * expit(bias) - 1.),
                         features.T @ residual + _L2 * slopes]
        return loss, gradient

    start = np.zeros(features.shape[1] + 1)
    prior = _INTERCEPT_PSEUDOCOUNT_PER_OUTCOME
    start[0] = np.log((labels.sum() + prior) / ((~labels).sum() + prior))
    result = minimize(objective, start, jac=True, method='L-BFGS-B')
    if not result.success:
        raise RuntimeError(f'visible station logistic fit failed: {result.message}')
    return result.x


@dataclass(frozen=True)
class VisibleStationReadout:
    """Frozen per-combination Bernoulli readout from public segment features."""

    mode: str
    parameters: np.ndarray  # (station, intercept + feature coefficients)

    def __post_init__(self):
        params = np.array(self.parameters, dtype=float, copy=True)
        if (self.mode not in _WIDTHS or params.shape != (len(STATIONS), _WIDTHS[self.mode] + 1)
                or not np.isfinite(params).all()):
            raise ValueError('invalid fitted visible-station readout parameters')
        params.setflags(write=False)
        object.__setattr__(self, 'parameters', params)

    @classmethod
    def fit(cls, mode, features, labels) -> VisibleStationReadout:
        x, y = np.asarray(features, dtype=float), np.asarray(labels)
        if (mode not in _WIDTHS or x.ndim != 3 or not len(x) or x.shape[1:] != (len(STATIONS), _WIDTHS[mode])
                or y.shape != x.shape[:2] or not np.isin(y, (0, 1)).all() or not np.isfinite(x).all()):
            raise ValueError('expected aligned training segments and binary combination labels')
        y = y.astype(bool, copy=False)
        return cls(mode, np.stack([_fit_logistic(x[:, idx], y[:, idx]) for idx in range(len(STATIONS))]))

    def predict_proba(self, features) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        if x.shape[-2:] != (len(STATIONS), _WIDTHS[self.mode]) or not np.isfinite(x).all():
            raise ValueError('features do not match fitted visible-station readout')
        probability = expit(self.parameters[:, 0] + np.einsum('...sf,sf->...s', x, self.parameters[:, 1:]))
        if self.mode == 'visible':
            # No possible complete assignment (including the all-empty case)
            # is not positive evidence, irrespective of pose and fitted bias.
            probability = np.where(x[..., _POSE_WIDTH] > 0., probability, 0.)
        return probability

    def predict(self, features) -> np.ndarray:
        return self.predict_proba(features) >= DECISION_THRESHOLD

    def state_dict(self) -> dict:
        return {'config': _config(), 'mode': self.mode, 'parameters': self.parameters.tolist()}

    @classmethod
    def from_state_dict(cls, state) -> VisibleStationReadout:
        if state['config'] != _config():
            raise ValueError('incompatible visible-station readout feature configuration')
        return cls(state['mode'], state['parameters'])
