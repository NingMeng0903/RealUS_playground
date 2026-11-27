"""Local synthetic acoustic residual test using the existing geometric renderer."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import ndtr, roots_hermitenorm
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parents[1]
LADDER = EXPERIMENTS / 'ab_module_ladder_20261110'
STRESS = EXPERIMENTS / 'ab_observation_stress_20261111'
sys.path[:0] = [str(LADDER / 'observation'), str(STRESS)]
from environment import _geometry_batch
import motion

START = 0
PROBES = (1, 2)
TERMINALS = (3, 4)
POSES = np.column_stack((np.zeros((5, 2)), [0., -.025, .025, -.10, .10]))
FEATURES = np.column_stack((np.ones(5), POSES[:, 2] / .10))
GATE = dict(u_m=.002, geometric_length_m=.015, angle_rad=float(np.deg2rad(10)))
BUDGET = 5.4
DWELL = np.array([0., .15, .15, 3., 3.])
TIME_PRICE = .04
NODES = 25
TRAIN_N, DEV_N, TEST_N = 384, 128, 128
TRAIN_SEED, DEV_SEED, TEST_SEED = 721210, 721211, 721219
TRAIN_REPEATS = 4
MAX_PROBES = 3
TRUE_SD = np.array([.065, .23])
SENSOR_SD = .035
# y intercept, y slope, y curvature, depth, z slope, z curvature.
NOMINAL = np.array([.00002, .008, .22, .034, .006, .40])
HALFWIDTH = np.array([.00017, .015, .35, .002, .015, .22])
CAPABILITY_SOURCE = STRESS / 'CAPABILITY.json'


def dump(path, obj):
    with path.open('x') as out:
        json.dump(obj, out, indent=2, allow_nan=False)
        out.write('\n')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry(parameters):
    params = np.atleast_2d(parameters)
    x = np.linspace(-.05, .05, 101)
    y0, slope, curve, depth, z_slope, z_curve = params[:, :, None].transpose(1, 0, 2)
    y = y0 + slope * x + .5 * curve * x**2
    z = -depth + z_slope * x + .5 * z_curve * x**2
    points = np.stack((np.broadcast_to(x, y.shape), y, z), axis=-1)[:, None]
    fields = {key: values[..., 0] for key, values in _geometry_batch(points, POSES).items()}
    fields['gate'] = (fields['exists'] & (abs(fields['u']) <= GATE['u_m'])
                      & (fields['geometric_length'] >= GATE['geometric_length_m'])
                      & (fields['angle_audit'] <= GATE['angle_rad']))
    fields['base'] = -.20 + .09 * np.tanh((fields['length'] - .018) / .015)
    return fields


def sample_objects(seed, n):
    geo_rng, latent_rng, reset_rng, noise_rng = [np.random.default_rng(stream)
                                                for stream in np.random.SeedSequence(seed).spawn(4)]
    params = geo_rng.uniform(NOMINAL - HALFWIDTH, NOMINAL + HALFWIDTH, (n, len(NOMINAL)))
    latent = latent_rng.normal(size=(n, 2)) * TRUE_SD
    theta_reset = reset_rng.normal(size=n) * TRUE_SD[1]
    return geometry(params), latent, theta_reset, noise_rng


def training_quality(seed, n, repeats):
    geo, latent, _, noise_rng = sample_objects(seed, n)
    public_means = geo['base'] + latent @ FEATURES.T
    return public_means[..., None] + noise_rng.normal(size=(n, len(POSES), repeats)) * SENSOR_SD


@dataclass(frozen=True)
class Model:
    mean: np.ndarray
    covariance: np.ndarray
    noise_variance: float

    def serialized(self):
        return dict(mean=self.mean.tolist(), covariance=self.covariance.tolist(),
                    noise_variance=self.noise_variance, noise_sd=float(np.sqrt(self.noise_variance)))

    @classmethod
    def from_file(cls, path):
        values = read(path)
        return cls(np.array(values['mean']), np.array(values['covariance']), values['noise_variance'])


def fit_public_quality(y, nominal_base):
    """Group-level moments using only repeated public Y and the nominal offset."""
    n, angles, repeats = y.shape
    cell_mean = y.mean(axis=-1)
    noise = np.sum((y - cell_mean[..., None])**2) / (n * angles * (repeats - 1))
    precision = FEATURES.T @ FEATURES
    residual = cell_mean - nominal_base
    coeffs = np.linalg.solve(precision, FEATURES.T @ residual.T).T
    # Remove the repeated measurement noise contribution to the between-object
    # covariance. Geometry mismatch in nominal offsets stays in the fitted model.
    covariance = np.cov(coeffs.T) - (noise / repeats) * np.linalg.inv(precision)
    eig = np.linalg.eigvalsh(covariance)
    if eig.min() <= 0:
        raise ValueError(f'nonpositive noise-corrected training covariance: {eig}')
    projected = residual - coeffs @ FEATURES.T
    diagnostics = dict(n_objects=n, observations_per_object=angles * repeats,
                       angle_cell_mean_residual_rmse=float(np.sqrt(np.mean(projected**2))),
                       covariance_eigenvalues=eig.tolist())
    return Model(coeffs.mean(axis=0), covariance, float(noise)), diagnostics


class PublicTree:
    """Conditional linear-Gaussian planner; no simulator state in its API."""
    def __init__(self, model, nominal_geometry, duration, *, reset=False, order=NODES):
        self.model = model
        self.base = nominal_geometry['base'][0]
        self.gate = nominal_geometry['gate'][0]
        self.duration = duration
        self.cost = duration + DWELL[None, :]
        self.precision = np.linalg.inv(model.covariance)
        self.natural = self.precision @ model.mean
        self.reset = reset
        self.nodes, weights = roots_hermitenorm(order)
        self.weights = weights / np.sqrt(2 * np.pi)

    @lru_cache(maxsize=None)
    def covariance(self, counts):
        design = FEATURES[list(PROBES)]
        precision = self.precision + design.T @ (np.asarray(counts)[:, None] * design) / self.model.noise_variance
        return np.linalg.inv(precision)

    def update(self, natural, counts, action, y):
        counts = list(counts)
        counts[PROBES.index(action)] += 1
        update = (np.asarray(y) - self.base[action])[..., None] * FEATURES[action] / self.model.noise_variance
        return natural + update, tuple(counts)

    def observations(self, natural, counts, action):
        x = FEATURES[action]
        cov = self.covariance(counts)
        mean = self.base[action] + (natural @ cov) @ x
        variance = x @ cov @ x + self.model.noise_variance
        return mean[..., None] + np.sqrt(variance) * self.nodes

    def predict(self, natural, counts):
        cov = self.covariance(counts)
        mean = natural @ cov
        features = FEATURES[list(TERMINALS)]
        quality = mean @ features.T
        variance = np.einsum('ai,ij,aj->a', features, cov, features)
        if self.reset:
            quality = mean[..., :1] + self.model.mean[1] * features[:, 1]
            variance = cov[0, 0] + self.model.covariance[1, 1] * features[:, 1]**2
        margin = quality + self.base[list(TERMINALS)]
        acoustic = ndtr(margin / np.sqrt(variance + self.model.noise_variance))
        return acoustic, acoustic * self.gate[list(TERMINALS)], margin

    def options(self, natural, counts, current, remaining, depth):
        _, p, _ = self.predict(natural, counts)
        terminal_cost = self.cost[current, list(TERMINALS)]
        values = np.where(terminal_cost <= remaining + 1e-10,
                          p - TIME_PRICE * terminal_cost, -np.inf)
        if depth == 1:
            return values
        return np.concatenate((values, np.stack([
            self.probe_option(natural, counts, current, remaining, action, depth)
            for action in PROBES], axis=-1)), axis=-1)

    def probe_option(self, natural, counts, current, remaining, action, depth, fixed=()):
        cost = self.cost[current, action]
        if cost + self.cost[action, list(TERMINALS)].min() > remaining + 1e-10:
            return np.full(natural.shape[:-1], -np.inf)
        y = self.observations(natural, counts, action)
        next_natural, next_counts = self.update(natural[..., None, :], counts, action, y)
        if fixed:
            val = self.probe_option(next_natural, next_counts, action, remaining - cost,
                                    fixed[0], depth - 1, fixed[1:])
        else:
            val = np.max(self.options(next_natural, next_counts, action, remaining - cost, depth - 1), axis=-1)
        return val @ self.weights - TIME_PRICE * cost

    def terminal_choice(self, natural, counts, current, remaining):
        values = self.options(natural, counts, current, remaining, 1)
        return TERMINALS[int(np.argmax(values))]

    def best_fixed_order(self):
        orders = [PROBES, PROBES[::-1]]
        values = [float(self.probe_option(self.natural, (0, 0), START, BUDGET, order[0], 3, order[1:]))
                  for order in orders]
        return orders[int(np.argmax(values))], values


def robot_and_motion(path):
    sys.path.insert(0, str(LADDER / 'capability'))
    from capability import CapabilityEvaluator

    ctx = next(c for c in read(CAPABILITY_SOURCE)['contexts'] if c['id'] == 'prior_only_layout')
    anchor = np.array(ctx['T_world_body'])
    q = np.array(ctx['q_kin'][ctx['initial_index']])
    rail = float(q[0])
    local = np.broadcast_to(np.eye(4), (len(POSES), 4, 4)).copy()
    local[:, :3, :3] = Rotation.from_euler('z', POSES[:, 2]).as_matrix()
    local[:, :2, 3] = POSES[:, :2]
    targets = anchor @ local
    evaluator = CapabilityEvaluator()
    witness = [evaluator._static_pose(target, rail, q) for target in targets]
    duration = motion.duration_matrix(POSES, method='prodmp')
    # Check that the same duration yields the expected TCP endpoint through the
    # original integral-basis ProDMP world pose decoder, including zero twist.
    error = []
    for start in (START, *PROBES):
        for end in (*PROBES, *TERMINALS):
            ref = motion.reference(POSES[start], POSES[end], anchor, duration_s=duration[start, end])
            endpoint, twist = ref.sample_world(ref.duration_s())
            error.append([float(np.linalg.norm(endpoint[:3] - targets[end, :3, 3])),
                          float((Rotation.from_euler('xyz', endpoint[3:]).inv() *
                                 Rotation.from_matrix(targets[end, :3, :3])).magnitude()),
                          float(np.linalg.norm(twist))])
    max_error = np.max(error, axis=0)
    if max_error.max() > 1e-10:
        raise AssertionError(f'ProDMP world endpoint mismatch {max_error}')
    dump(path, dict(layout=ctx['id'], input_capability_sha256=digest(CAPABILITY_SOURCE),
                    anchor=anchor.tolist(), fixed_rail_m=rail, q_seed=q.tolist(),
                    targets_world=targets.tolist(), endpoint_witnesses=witness,
                    prodmp_motion_s=duration.tolist(), prodmp_max_endpoint_errors=max_error.tolist(),
                    scope='Finite static endpoint IK/FK, limits and configured self-collision; offline ProDMP reference duration.'))
    if not all(entry['found'] for entry in witness):
        raise RuntimeError('one or more shared endpoint witnesses were not found; see ROBOT.json')
    return duration


@dataclass(frozen=True)
class PublicPolicy:
    name: str
    kind: str
    tree: PublicTree
    depth: int = 1
    fixed: tuple = ()

    def difference_margins(self, trace):
        residual = {a: y - self.tree.base[a] for a, y in trace}
        low, high = (residual[a] for a in PROBES)
        b = .5 * (low + high)
        theta = (high - low) / (FEATURES[PROBES[1], 1] - FEATURES[PROBES[0], 1])
        return self.tree.base[list(TERMINALS)] + FEATURES[list(TERMINALS)] @ [b, theta]

    def choose(self, natural, counts, current, elapsed, trace):
        remaining = BUDGET - elapsed
        if len(trace) < len(self.fixed):
            return self.fixed[len(trace)]
        if self.kind == 'geometry':
            scores = np.where(self.tree.gate[list(TERMINALS)], self.tree.base[list(TERMINALS)], -np.inf)
            return TERMINALS[int(np.argmax(scores))]
        if self.kind == 'difference':
            margins = np.where(self.tree.gate[list(TERMINALS)], self.difference_margins(trace), -np.inf)
            return TERMINALS[int(np.argmax(margins))]
        values = self.tree.options(natural, counts, current, remaining, self.depth - len(trace))
        return (TERMINALS + PROBES)[int(np.argmax(values))]

    def forecast(self, natural, counts, trace, action):
        if self.kind == 'geometry':
            return None, None, None
        if self.kind == 'difference':
            margin = self.difference_margins(trace)[TERMINALS.index(action)]
            return None, None, bool(self.tree.gate[action] and margin >= 0.)
        acoustic, p, _ = self.tree.predict(natural, counts)
        idx = TERMINALS.index(action)
        return float(acoustic[idx]), float(p[idx]), bool(p[idx] >= .5)


@dataclass(frozen=True)
class PrivateCase:
    base: np.ndarray
    gate: np.ndarray
    latent: np.ndarray
    reset_theta: float
    probe_noise: np.ndarray
    terminal_noise: np.ndarray

    def probe(self, action, visit):
        return float(self.base[action] + FEATURES[action] @ self.latent
                     + self.probe_noise[PROBES.index(action), visit])

    def terminal(self, action, reset):
        b, theta = self.latent
        if reset:
            theta = self.reset_theta
        y = self.base[action] + FEATURES[action] @ [b, theta] + self.terminal_noise[TERMINALS.index(action)]
        physical = bool(self.gate[action])
        acoustic = bool(y >= 0.)
        return physical and acoustic, physical, acoustic


def evaluate_case(policy, case, reset):
    natural, counts, current = policy.tree.natural, (0, 0), START
    trace = []
    elapsed = 0.
    while True:
        action = policy.choose(natural, counts, current, elapsed, tuple(trace))
        elapsed += policy.tree.cost[current, action]
        if elapsed > BUDGET + 1e-10:
            raise AssertionError(f'{policy.name} over time budget: {elapsed}')
        if action in TERMINALS:
            break
        if action not in PROBES or len(trace) >= MAX_PROBES:
            raise AssertionError(f'{policy.name} exceeded the legal probe count')
        y = case.probe(action, counts[PROBES.index(action)])
        trace.append((action, y))
        if policy.kind == 'learned':
            natural, counts = policy.tree.update(natural, counts, action, y)
        else:
            counts = tuple(c + (probe == action) for probe, c in zip(PROBES, counts))
        current = action
    acoustic_p, joint_p, predicted = policy.forecast(natural, counts, tuple(trace), action)
    passed, physical, acoustic = case.terminal(action, reset)
    return dict(actions=[a for a, _ in trace] + [action], public_probe_y=[y for _, y in trace],
                time_s=float(elapsed), acoustic_probability=acoustic_p, joint_probability=joint_p,
                predicted_pass=predicted, physical_pass=physical, acoustic_pass=acoustic, pass_=passed)


def policies(matched, reset_model):
    fixed, _ = matched.best_fixed_order()
    return [PublicPolicy('geometry', 'geometry', matched),
            *(PublicPolicy(f'h{depth}', 'learned', matched, depth) for depth in range(1, 5)),
            PublicPolicy('fixed_pair_learned', 'learned', matched, 3, fixed),
            PublicPolicy('fixed_pair_same_order', 'learned', matched, 3, PROBES),
            PublicPolicy('paired_difference', 'difference', matched, 3, PROBES),
            PublicPolicy('reset_h4', 'learned', reset_model, 4)]


def summarize(rows):
    hit = np.array([row['pass_'] for row in rows])
    acoust = np.array([row['acoustic_pass'] for row in rows])
    elapsed = np.array([row['time_s'] for row in rows])
    counts = np.bincount([len(row['actions']) - 1 for row in rows], minlength=MAX_PROBES + 1)
    out = dict(n=len(rows), passes=int(hit.sum()), success=float(hit.mean()),
               physical_rate=float(np.mean([row['physical_pass'] for row in rows])),
               acoustic_rate=float(acoust.mean()), mean_time_s=float(elapsed.mean()),
               min_time_s=float(elapsed.min()), max_time_s=float(elapsed.max()),
               mean_utility=float((hit - TIME_PRICE * elapsed).mean()), probe_counts=counts.tolist(),
               positive_yaw_terminals=sum(row['actions'][-1] == TERMINALS[-1] for row in rows),
               prediction_errors=None, false_pass_predictions=None, false_fail_predictions=None,
               brier_joint=None, brier_acoustic=None, mean_joint_forecast=None)
    if rows[0]['predicted_pass'] is not None:
        guesses = np.array([row['predicted_pass'] for row in rows])
        out.update(prediction_errors=int(np.sum(guesses != hit)),
                   false_pass_predictions=int(np.sum(guesses & ~hit)),
                   false_fail_predictions=int(np.sum(~guesses & hit)))
    if rows[0]['joint_probability'] is not None:
        joint = np.array([row['joint_probability'] for row in rows])
        acoustic = np.array([row['acoustic_probability'] for row in rows])
        out.update(brier_joint=float(np.mean((joint - hit)**2)), brier_acoustic=float(np.mean((acoustic - acoust)**2)),
                   mean_joint_forecast=float(joint.mean()))
    return out


def bootstrap_delta(a, b, indices):
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    ci = np.quantile(delta[indices].mean(axis=1), [.025, .975])
    return dict(delta=float(delta.mean()), ci95=ci.tolist())


def contrasts(arms, seed, n):
    indices = np.random.default_rng(seed + 97531).integers(n, size=(10000, n))
    pos, neg = arms['persistent'], arms['resampled']

    def metric(rows, name):
        return np.asarray([float(row['pass_']) - (TIME_PRICE * row['time_s'] if name == 'utility' else 0.)
                           for row in rows])

    def compare(data, a, b):
        return {name: bootstrap_delta(metric(data[a], name), metric(data[b], name), indices)
                for name in ('success', 'utility')}

    return dict(persistent={f'h4_minus_{b}': compare(pos, 'h4', b) for b in
                            ('geometry', 'h1', 'h2', 'h3', 'fixed_pair_learned', 'paired_difference')}
                           | {'same_order_learned_minus_difference':
                              compare(pos, 'fixed_pair_same_order', 'paired_difference')},
                resampled={f'{a}_minus_{b}': compare(neg, a, b) for a, b in
                           (('h4', 'h1'), ('reset_h4', 'h1'), ('reset_h4', 'h4'))},
                persistence_interaction={name: bootstrap_delta(metric(pos['h4'], name) - metric(pos['h1'], name),
                                                                metric(neg['h4'], name) - metric(neg['h1'], name),
                                                                indices)
                                         for name in ('success', 'utility')})


def evaluate_split(seed, n, model, duration):
    geo, latent, reset_theta, rng = sample_objects(seed, n)
    probe_noise = rng.normal(size=(n, len(PROBES), MAX_PROBES)) * SENSOR_SD
    terminal_noise = rng.normal(size=(n, len(TERMINALS))) * SENSOR_SD
    nominal = geometry(NOMINAL)
    matched = PublicTree(model, nominal, duration)
    reset_model = PublicTree(model, nominal, duration, reset=True)
    all_policies = policies(matched, reset_model)
    arms = {}
    for reset, label in ((False, 'persistent'), (True, 'resampled')):
        active = [p for p in all_policies if reset or p.name != 'reset_h4']
        arms[label] = {p.name: [] for p in active}
        for i in range(n):
            case = PrivateCase(geo['base'][i], geo['gate'][i], latent[i], float(reset_theta[i]),
                               probe_noise[i], terminal_noise[i])
            for policy in active:
                arms[label][policy.name].append(evaluate_case(policy, case, reset))
    # Only realized audit outputs are retained. Latents and true geometry stay in the simulator.
    return dict(seed=seed, n=n, arms={arm: {name: summarize(rows) for name, rows in data.items()}
                                     for arm, data in arms.items()},
                paired_contrasts=contrasts(arms, seed, n),
                private_simulation_audit=dict(per_action_physical_pass=geo['gate'].mean(axis=0).tolist()),
                rows=arms)


def public_predictions(model, duration):
    nominal = geometry(NOMINAL)
    predictions = {}
    for order in (NODES, 41):
        match = PublicTree(model, nominal, duration, order=order)
        reset = PublicTree(model, nominal, duration, reset=True, order=order)
        group = {}
        for label, tree in (('persistent_model', match), ('reset_aware_model', reset)):
            p, _, _ = tree.predict(tree.natural, (0, 0))
            root = {}
            for depth in range(1, 5):
                values = tree.options(tree.natural, (0, 0), START, BUDGET, depth)
                actions = TERMINALS if depth == 1 else TERMINALS + PROBES
                root[f'h{depth}'] = dict(actions=list(actions), utilities=values.tolist(),
                                         chosen_action=actions[int(np.argmax(values))])
            fixed_order, values = tree.best_fixed_order()
            cov_zero = tree.covariance((0, 0))
            cov_pair = tree.covariance((1, 1))
            group[label] = dict(terminal_prior_acoustic_p=p.tolist(), root=root,
                                fixed_pair_orders=[list(PROBES), list(PROBES[::-1])],
                                fixed_pair_utilities=values, best_fixed_pair=list(fixed_order),
                                covariance_before=cov_zero.tolist(), covariance_after_pair=cov_pair.tolist())
        predictions[str(order)] = group
    return predictions


def protocol(duration, out):
    nominal = geometry(NOMINAL)
    sources = [Path(__file__), LADDER / 'observation/environment.py', STRESS / 'motion.py',
               LADDER / 'capability/capability.py', CAPABILITY_SOURCE]
    cost = duration + DWELL[None, :]
    reach = {START: 0.}
    for _ in range(MAX_PROBES):
        reach = {a: max(elapsed + cost[current, a] for current, elapsed in reach.items()) for a in PROBES}
    worst_cost = max(elapsed + cost[current, a] for current, elapsed in reach.items() for a in TERMINALS)
    return dict(scope='Local coarse-ROI scene, one continuous vessel, synthetic acoustic quality, offline static robot endpoint witnesses.',
                labels=['start', 'probe_minus', 'probe_plus', 'terminal_minus', 'terminal_plus'],
                poses_xy_yaw=POSES.tolist(), geometric_thresholds=GATE,
                geometry_fields=['y0_m', 'y_slope', 'y_curvature_per_m', 'depth_m', 'z_slope', 'z_curvature_per_m'],
                nominal_centerline=NOMINAL.tolist(), public_uniform_shape_prior_halfwidth=HALFWIDTH.tolist(),
                planner_geometry_input='Shared nominal clipping only. The public shape prior describes the simulator support; there is no runtime particle filter.',
                simulator=dict(b_sd=TRUE_SD[0], theta_sd=TRUE_SD[1], sensor_sd=SENSOR_SD,
                               mechanism='Y = -0.20 + 0.09*tanh((clipped image length in m - .018)/.015) + b + theta*(yaw/.10) + eps',
                               success='Realized terminal Y >= 0 and independent evaluation of the physical gate on the rendered true centerline.',
                               arms='Persistent theta or an independent theta redraw at terminal only. b and geometry are unchanged.',
                               paired_noise='Common Gaussian noise keyed by object and action, plus visit number for probes. The arms share probe outcomes and same-action terminal noise.'),
                split=dict(train=[TRAIN_SEED, TRAIN_N], dev=[DEV_SEED, DEV_N], test=[TEST_SEED, TEST_N],
                           independent=True, repeated_train_y_per_pose=TRAIN_REPEATS),
                action_price=dict(budget_s=BUDGET, dwell_s=DWELL.tolist(), max_probes=MAX_PROBES,
                                  utility='Pr(physical and acoustic success) - .04 * elapsed seconds',
                                  time_price_per_s=TIME_PRICE, longest_allowed_three_probe_sequence_s=float(worst_cost),
                                  termination='Exactly one terminal. Two terminal dwells exceed budget. Each probe endpoint can be revisited.'),
                planner=dict(quad_order=NODES, pretest_quad_check_order=41, horizons=[1, 2, 3, 4],
                             learned='Fit random intercept and slope moments to grouped repeated public Y minus nominal geometry. Carry Gaussian precision and precision-weighted mean.',
                             fixed='Select one of two balanced probe orders by public expected utility before observations, then use same learned terminal selection; also compare the canonical minus then plus order.',
                             difference='Canonical minus then plus; subtract nominal geometry at each probe; intercept is mean, slope is difference divided by .50; choose highest nominal terminal plus extrapolated residual.',
                             reset_aware='Retain learned posterior on b. Marginalize a fresh independent theta using the learned prior marginal.'),
                source_sha256={str(p): digest(p) for p in sources},
                pretest_artifact_sha256={p.name: digest(p) for p in (out/'ROBOT.json', out/'MODEL.json',
                                        out/'TRAIN_PUBLIC.npz', out/'DEV.json', out/'PUBLIC_PREDICTIONS.json')})


def prepare(out):
    if (out/'TEST.json').exists() or (out/'PROTOCOL.json').exists():
        raise FileExistsError(f'existing frozen or test artifact at {out}')
    out.mkdir(parents=True, exist_ok=True)
    duration = robot_and_motion(out/'ROBOT.json')
    y = training_quality(TRAIN_SEED, TRAIN_N, TRAIN_REPEATS)
    with (out/'TRAIN_PUBLIC.npz').open('xb') as stream:
        np.savez_compressed(stream, public_y=y, poses=POSES)
    model, diagnostics = fit_public_quality(y, geometry(NOMINAL)['base'][0])
    dump(out/'MODEL.json', model.serialized() | {'fit': diagnostics})
    dump(out/'PUBLIC_PREDICTIONS.json', public_predictions(model, duration))
    dump(out/'DEV.json', evaluate_split(DEV_SEED, DEV_N, model, duration))
    dump(out/'PROTOCOL.json', protocol(duration, out))


def test(out):
    frozen = read(out/'PROTOCOL.json')
    for name, sha in frozen['source_sha256'].items():
        if digest(Path(name)) != sha:
            raise ValueError(f'frozen source changed: {name}')
    for name, sha in frozen['pretest_artifact_sha256'].items():
        if digest(out/name) != sha:
            raise ValueError(f'frozen artifact changed: {name}')
    if (out/'TEST.json').exists():
        raise FileExistsError(out/'TEST.json')
    duration = np.asarray(read(out/'ROBOT.json')['prodmp_motion_s'])
    model = Model.from_file(out/'MODEL.json')
    results = evaluate_split(TEST_SEED, TEST_N, model, duration)
    dump(out/'TEST.json', results | {'frozen_protocol_sha256': digest(out/'PROTOCOL.json')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'test'])
    parser.add_argument('--output', type=Path, default=HERE)
    args = parser.parse_args()
    started = time.perf_counter()
    {'prepare': prepare, 'test': test}[args.mode](args.output)
    print(f'{args.mode} finished in {time.perf_counter() - started:.2f} s at {args.output}')


if __name__ == '__main__':
    main()
