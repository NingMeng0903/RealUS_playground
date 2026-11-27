"""CEM-discovered motion proposals and a public, learned recovery selector."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor

from adaptive_controller import (BINS, CENTER_MM, EDGES, QUALIFIED_EVIDENCE, AdaptiveController, FreeCandidate,
                                 station_bin)
from adaptive_teacher import replace_shape
from budgeted_recovery import CAPS
from embedding import position_embedding
from recovery_world import FOV_MAX, FOV_MIN


TARGETS = ("corrected_complete", "final_wrong", "new_centered_mistake", "tcp_mm", "turn_degs", "seconds")
VISIBLE_TARGETS = (*TARGETS[:2], "new_identity_mistake", *TARGETS[3:])
OBJECTIVE = np.array([1., -1.25, 0., -.002, 0., -.02])


def outcome_vector(metrics, targets=TARGETS):
    return [float(metrics[field]) for field in targets]


def resource_limits(targets):
    return {targets[TARGETS.index(field)]: value for field, value in CAPS.items()}


def cem_requests(discovery_rows, count):
    """Compress verified successful training proposals into shared relative requests."""
    requests = np.unique([row["example"]["request"] for row in discovery_rows if row["example"]], axis=0)
    if not len(requests):
        raise ValueError("CEM discovery yielded no verified demonstrations")
    if len(requests) <= count:
        return requests
    scaled = requests / [4., 4., 2.5]
    clusters = KMeans(n_clusters=count, random_state=629, n_init=10).fit(scaled)
    medoids = []
    for index, center in enumerate(clusters.cluster_centers_):
        members = np.flatnonzero(clusters.labels_ == index)
        nearest = np.argmin(np.linalg.norm(scaled[members] - center, axis=1))
        medoids.append(members[nearest])
    return requests[sorted(medoids)]


class CEMMenuController(AdaptiveController):
    def __init__(self, world, sections, coefficients, fitted, requests, *, evidence_mode=QUALIFIED_EVIDENCE,
                 material_surface=None):
        super().__init__(world, sections, coefficients, fitted, evidence_mode=evidence_mode)
        self.requests = requests
        self.material_surface = material_surface

    def options_shared(self, state, *, expanded=True, renderer="fit"):
        menu = super().options_shared(state, expanded=expanded, renderer="waypoint")
        if not expanded:
            return menu
        if renderer == "fit":
            menu = {name: replace_shape(self, state, action)[0] for name, action in menu.items()}
        for index, request in enumerate(self.requests):
            name = f"cem_{index}"
            action = self.free_candidate(state, name, *request, renderer=renderer)
            if action is None and renderer == "fit":
                action = self.free_candidate(state, name, *request)
            if action is not None and all(
                np.max(abs(action.s - other.s)) >= .05 or np.max(abs(action.lateral.x - other.lateral.x)) >= .05
                for other in menu.values()
            ):
                menu[name] = action
        return menu


def _features(controller, state, candidate, responsibilities):
    indices = np.linspace(0, len(candidate.s) - 1, 6).round().astype(int)[1:]
    predictions = [controller.world.predict(candidate.s[i], candidate.lateral.x[i], state.angle) for i in indices]
    means, visibility = (np.stack(value) for value in zip(*predictions, strict=True))
    visible = state.capture_support is not None
    imaging = visibility[:, :, 0] if visible else visibility[:, :, 0] * np.exp(-.5 * (means[:, :, 0, 0] / CENTER_MM)**2)
    bins = np.asarray([station_bin(candidate.s[i]) for i in indices])
    coverage = np.stack([1. - np.prod(1. - imaging[bins == j], axis=0) for j in range(BINS)])
    if visible:
        coverage *= 1. - np.stack(state.capture_support)
    b = state.belief
    query = state.query_bin()
    embedding = position_embedding(
        current_s=state.s, current_x=state.x, candidate_s=candidate.s, candidate_x=candidate.lateral.x,
        query_interval=(EDGES[query], EDGES[query + 1]), last_trusted_s=state.last_trusted,
        belief=b, target_positions=means[-1, :, 0] + [candidate.lateral.goal, 0.],
        target_visibility=visibility[-1, :, 0], packet_responsibilities=responsibilities, angle_deg=state.angle,
        material_surface=controller.material_surface if visible else None,
        field_of_view=(FOV_MIN, FOV_MAX) if visible else None,
    )
    # Keep location and observable separability distinct. Seeing both tracks
    # with separated centroids is often informative even without a centered hit.
    separation = np.linalg.norm(means[:, :, 0] - means[:, :, 1], axis=-1)
    disambiguation = visibility[:, :, 0] * visibility[:, :, 1] * np.minimum(separation / 6., 1.)
    witness_risk = [1. - b @ w.identity if w is not None else 1. for w in state.ledger]
    return np.r_[
        controller.policy_features(state, candidate, False), embedding, b,
        state.travel / 67., state.vx / 18., state.residual / (FOV_MAX[0] if visible else CENTER_MM), witness_risk,
        (coverage @ b), np.sqrt(np.maximum((coverage**2) @ b - (coverage @ b)**2, 0.)),
        imaging @ b, disambiguation @ b, (candidate.lateral.v[indices] / 18.),
        float(isinstance(candidate, FreeCandidate) and candidate.renderer == "fit"),
    ]


def features_for_menu(controller, state, menu, incumbent, responsibilities):
    """All inputs describe the causal history, public model and feasible references."""
    features = np.stack([_features(controller, state, action, responsibilities) for action in menu.values()])
    reference = features[list(menu).index(incumbent)]
    # Preserve absolute geometry and expose differences from the actual
    # incumbent, so the learned model can predict the paired effect directly.
    return np.column_stack((features, features - reference))


def choose_from_deltas(names, incumbent, deltas, *, targets=TARGETS, caps=None):
    """Apply the shared objective and predicted resource limits."""
    deltas = np.asarray(deltas, dtype=float).copy()
    incumbent_index = names.index(incumbent)
    deltas[incumbent_index] = 0.
    allowed = np.ones(len(names), dtype=bool)
    limits = resource_limits(targets) if caps is None else caps
    for metric, cap in limits.items():
        allowed &= deltas[:, targets.index(metric)] <= cap + 1e-8
    scores = deltas @ OBJECTIVE
    index = max(np.flatnonzero(allowed), key=lambda i: (scores[i], i == incumbent_index))
    return names[index], deltas, allowed


def fit_targets(features, targets):
    models = []
    for values in targets.T:
        model = HistGradientBoostingRegressor(
            max_iter=125, max_leaf_nodes=15, min_samples_leaf=24, learning_rate=.055,
            l2_regularization=3., early_stopping=False, random_state=629,
        )
        models.append(model.fit(features, values))
    return models


def predict_deltas(models, features):
    deltas = np.column_stack([model.predict(features) for model in models])
    deltas[:, :3] = np.clip(deltas[:, :3], -1., 1.)
    return deltas
