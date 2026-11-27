"""No-patient-prescan A/B synthetic image acquisition benchmark.

The public controller receives only a generic template, its broad prior, its own
chosen probe poses, and observations. World/true vessel data are private to
sampling and independent evaluation. This is not real B-mode or a robot QP test.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import sys
import time
from typing import NamedTuple

import numpy as np
from scipy.special import betaln, logsumexp
from scipy.stats import beta as beta_dist

HERE = Path(__file__).resolve().parent
REPO = Path("/media/camp/EXT_DRIVE/Among_US")
sys.path.insert(0, str(REPO))
from DMP_playground.dmp_honest.dmp import fit_discrete_dmp

CFG = json.loads((HERE / "PROTOCOL.json").read_text())
ACTIONS = CFG["actions"]
HYP = tuple((lane, tangent) for lane in (-1, 1) for tangent in (-1, 1))
PRIOR = np.full(len(HYP), 1.0 / len(HYP))
FOV = float(CFG["sensor"]["fov_half_width_m"])
U_SD = float(CFG["sensor"]["u_noise_sd_m"])
ANG_SD = float(CFG["sensor"]["tangent_noise_sd_rad"])
DEPTH_SD = 0.002
P_DETECT = float(CFG["sensor"]["detect_given_vessel"])
P_FALSE = float(CFG["sensor"]["false_detection_without_vessel"])
FALSE_PENALTY = float(CFG["planning"]["false_accept_penalty_s"])
TIMEOUT_PENALTY = float(CFG["planning"]["timeout_penalty_s"])
DEADLINE = float(CFG["planning"]["deadline_s"])
BETA_TRUE = tuple(CFG["sensor"]["identity_score_true_beta"])
BETA_DECOY = tuple(CFG["sensor"]["identity_score_decoy_beta"])


@dataclass(frozen=True)
class Pose:
    s: float
    y: float
    yaw: float

    @property
    def x(self) -> float:
        return self.s * float(CFG["template"]["s_scale_m"])

    def vector(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw], float)


@dataclass(frozen=True)
class Observation:
    visible: bool
    u: float = math.nan
    tangent: float = math.nan
    score: float = math.nan
    depth: float = math.nan

    def public_dict(self) -> dict:
        f = lambda x: float(x) if math.isfinite(x) else None
        return dict(visible=bool(self.visible), image_u_m=f(self.u),
                    tangent_rad=f(self.tangent), target_identity_score=f(self.score),
                    depth_m=f(self.depth))


@dataclass(frozen=True)
class World:
    mode: int
    c0: float
    c1: float
    c2: float
    depth_delta: float
    decoy_c0: float


@dataclass(frozen=True)
class Event:
    public: Observation
    source: str


class VesselState(NamedTuple):
    y: float
    angle: float
    depth: float
    is_target: bool


def pose_for_action(index: int) -> Pose:
    a = ACTIONS[index]
    return Pose(float(a["s"]), float(a["y_m"]), float(a["yaw_rad"]))


START = Pose(float(CFG["template"]["s_start"]), 0.0, 0.0)


def template_curve(s: float) -> float:
    # Generic surface-chart template, deliberately independent of the patient.
    return 0.0


def curve_states(mode: int, pose: Pose, world: World | None = None) -> tuple[VesselState, VesselState]:
    lane_sign, tangent_sign = HYP[mode]
    s = pose.s
    lane = lane_sign * 0.022
    nominal_slope = 0.2 * math.tan(tangent_sign * 0.17)
    c0 = c1 = c2 = dep = decoy_c0 = 0.0
    if world is not None:
        c0, c1, c2, dep, decoy_c0 = world.c0, world.c1, world.c2, world.depth_delta, world.decoy_c0
    phase = math.pi * (s - 0.4) / 0.4
    target_y = template_curve(s) + lane + nominal_slope * (s - 0.5) + c0 + c1 * (s - 0.5) + c2 * math.sin(phase)
    target_slope = nominal_slope + c1 + c2 * (math.pi / 0.4) * math.cos(phase)
    target = VesselState(target_y, math.atan2(target_slope, 0.2), 0.040 + dep, True)
    decoy_y = template_curve(s) - lane - nominal_slope * (s - 0.5) + decoy_c0
    decoy = VesselState(decoy_y, math.atan2(-nominal_slope, 0.2), 0.052, False)
    return target, decoy


def selected_vessel(mode: int, pose: Pose, world: World | None = None) -> VesselState | None:
    vessels = curve_states(mode, pose, world)
    ordered = sorted(vessels, key=lambda v: (abs(v.y - pose.y), v.y))
    return ordered[0] if abs(ordered[0].y - pose.y) <= FOV else None


def normal_logpdf(x: float, mean: float, sd: float) -> float:
    z = (x - mean) / sd
    return -0.5 * z * z - math.log(sd) - 0.5 * math.log(2.0 * math.pi)


def beta_logpdf(x: float, aa: float, bb: float) -> float:
    x = min(max(float(x), 1e-12), 1.0 - 1e-12)
    return (aa - 1.0) * math.log(x) + (bb - 1.0) * math.log1p(-x) - float(betaln(aa, bb))


def detection_probability(vessel: VesselState | None) -> float:
    return P_DETECT if vessel is not None else P_FALSE


def log_likelihood(mode: int, pose: Pose, obs: Observation) -> float:
    vessel = selected_vessel(mode, pose)
    pv = detection_probability(vessel)
    if not obs.visible:
        return math.log1p(-pv)
    if vessel is None:
        if abs(obs.u) > FOV or abs(obs.tangent) > 0.35 or not (0.02 <= obs.depth <= 0.07):
            return -math.inf
        return math.log(pv) - math.log(2 * FOV) - math.log(0.7) - math.log(.05) + beta_logpdf(obs.score, *BETA_DECOY)
    ab = BETA_TRUE if vessel.is_target else BETA_DECOY
    return (math.log(pv)
            + normal_logpdf(obs.u, vessel.y - pose.y, U_SD)
            + normal_logpdf(obs.tangent, vessel.angle, ANG_SD)
            + normal_logpdf(obs.depth, vessel.depth, DEPTH_SD)
            + beta_logpdf(obs.score, *ab))


def posterior(prior: np.ndarray, pose: Pose, obs: Observation) -> np.ndarray:
    lp = np.log(np.maximum(prior, 1e-300)) + np.array([log_likelihood(i, pose, obs) for i in range(len(HYP))])
    if not np.isfinite(lp).any():
        # No model hypothesis supports an outlier: preserve prior and record no invented evidence.
        return prior.copy()
    lp -= logsumexp(lp)
    return np.exp(lp)


def sensor(world: World, pose: Pose, world_id: int, step: int, action: int, seed: int) -> Event:
    rng = np.random.default_rng(np.random.SeedSequence([seed, world_id, step, action]))
    vessel = selected_vessel(world.mode, pose, world)
    pv = detection_probability(vessel)
    if rng.random() >= pv:
        return Event(Observation(False), "missing")
    if vessel is None:
        obs = Observation(True, float(rng.uniform(-FOV, FOV)), float(rng.uniform(-0.35, 0.35)),
                          float(rng.beta(*BETA_DECOY)), float(rng.uniform(.02, .07)))
        return Event(obs, "false_detection")
    ab = BETA_TRUE if vessel.is_target else BETA_DECOY
    obs = Observation(True,
        float(vessel.y - pose.y + rng.normal(0, U_SD)),
        float(vessel.angle + rng.normal(0, ANG_SD)),
        float(rng.beta(*ab)),
        float(vessel.depth + rng.normal(0, DEPTH_SD)))
    return Event(obs, "target" if vessel.is_target else "decoy")


def public_stop(obs: Observation, pose: Pose) -> bool:
    rule = CFG["quality"]["public_stop"]
    return bool(obs.visible and abs(obs.u) <= rule["abs_u_max_m"]
                and abs(obs.tangent - pose.yaw) <= rule["abs_angle_error_max_rad"]
                and obs.score >= rule["target_score_min"]
                and rule["depth_min_m"] <= obs.depth <= rule["depth_max_m"])


def independent_quality(event: Event, world: World, pose: Pose) -> bool:
    rule = CFG["quality"]["independent_truth"]
    target, _ = curve_states(world.mode, pose, world)
    return bool(event.public.visible and event.source == "target"
                and abs(target.y - pose.y) <= rule["abs_u_max_m"]
                and abs(target.angle - pose.yaw) <= rule["abs_angle_error_max_rad"]
                and rule["depth_min_m"] <= target.depth <= rule["depth_max_m"]
                and rule["contact"])


def model_quality(mode: int, pose: Pose, source: str) -> bool:
    rule = CFG["quality"]["independent_truth"]
    target, _ = curve_states(mode, pose)
    return bool(source == "target"
                and abs(target.y - pose.y) <= rule["abs_u_max_m"]
                and abs(target.angle - pose.yaw) <= rule["abs_angle_error_max_rad"]
                and rule["depth_min_m"] <= target.depth <= rule["depth_max_m"])


def model_nodes(mode: int, pose: Pose) -> tuple[tuple[float, Observation, str], ...]:
    """Fixed, result-independent public-model quadrature (continuous feature values)."""
    vessel = selected_vessel(mode, pose)
    pv = detection_probability(vessel)
    nodes: list[tuple[float, Observation, str]] = [(1.0 - pv, Observation(False), "missing")]
    z_u = [-1.2, -0.5, 0.5, 1.2, -0.8, 0.8, 0.0, 0.0]
    z_a = [0.5, -1.2, 1.2, -0.5, 0.8, -0.8, 0.0, 0.0]
    q_score = [.15, .35, .65, .85, .25, .75, .45, .55]
    for i in range(8):
        if vessel is None:
            u = FOV * (2 * q_score[i] - 1)
            angle = .35 * (2 * q_score[7 - i] - 1)
            score = float(beta_dist.ppf(q_score[i], *BETA_DECOY))
            depth = .02 + .05 * q_score[(i + 3) % 8]
            source = "false_detection"
        else:
            u = vessel.y - pose.y + U_SD * z_u[i]
            angle = vessel.angle + ANG_SD * z_a[i]
            ab = BETA_TRUE if vessel.is_target else BETA_DECOY
            score = float(beta_dist.ppf(q_score[i], *ab))
            depth = vessel.depth + DEPTH_SD * z_u[7 - i]
            source = "target" if vessel.is_target else "decoy"
        nodes.append((pv / 8.0, Observation(True, float(u), float(angle), float(score), float(depth)), source))
    return tuple(nodes)


MODEL_NODES = {(h, a): model_nodes(h, pose_for_action(a)) for h in range(len(HYP)) for a in range(len(ACTIONS))}


class PathBank:
    def __init__(self) -> None:
        started = time.perf_counter()
        self.paths = {}
        self.cost = np.zeros((len(ACTIONS) + 1, len(ACTIONS)), float)
        self.max_endpoint_error = 0.0
        self.max_nominal_speed = 0.0
        for source in range(-1, len(ACTIONS)):
            src = START if source < 0 else pose_for_action(source)
            for dest in range(len(ACTIONS)):
                dst = pose_for_action(dest)
                ss = np.linspace(0, 1, 41)
                smooth = 3 * ss**2 - 2 * ss**3
                demo = src.vector()[None] + smooth[:, None] * (dst.vector() - src.vector())[None]
                dmp = fit_discrete_dmp(ss, demo, n_basis=30)
                dense, _ = dmp.rollout(np.linspace(0, 1, 161))
                path = dense[np.linspace(0, 160, 21).astype(int)]
                endpoint_error = float(np.linalg.norm(path[-1] - dst.vector()))
                self.max_endpoint_error = max(self.max_endpoint_error, endpoint_error)
                xyz_length = float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum())
                yaw_length = float(np.abs(np.diff(path[:, 2])).sum())
                translation_steps = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
                yaw_steps = np.abs(np.diff(path[:, 2]))
                duration = max(xyz_length / .02, yaw_length / .5, float(translation_steps.max()) * 20 / .02,
                               float(yaw_steps.max()) * 20 / .5, .3)
                nominal_speed = float(translation_steps.max() / (duration / 20))
                self.max_nominal_speed = max(self.max_nominal_speed, nominal_speed)
                self.paths[source, dest] = dict(path=path, duration=duration, xyz_length=xyz_length,
                                                yaw_length=yaw_length, endpoint_error=endpoint_error)
                self.cost[source + 1, dest] = duration + float(CFG["planning"]["observation_time_s"])
        if self.max_endpoint_error > 0.002:
            raise ValueError(f"DMP action bank endpoint error too large: {self.max_endpoint_error}")
        self.setup_wall_s = time.perf_counter() - started

    def action_cost(self, source: int, dest: int) -> float:
        return float(self.cost[source + 1, dest])

    def duration(self, source: int, dest: int) -> float:
        return float(self.paths[source, dest]["duration"])


def terminal_risk() -> np.ndarray:
    risk = np.zeros((len(HYP), len(ACTIONS)), float)
    for h in range(len(HYP)):
        for a in range(len(ACTIONS)):
            pose = pose_for_action(a)
            for weight, obs, source in MODEL_NODES[h, a]:
                if public_stop(obs, pose):
                    risk[h, a] += weight * (0.0 if model_quality(h, pose, source) else FALSE_PENALTY)
                else:
                    risk[h, a] += weight * TIMEOUT_PENALTY
    return risk


TERMINAL_RISK = terminal_risk()


def terminal_values(bank: PathBank, belief: np.ndarray, source: int) -> np.ndarray:
    return bank.cost[source + 1] + belief @ TERMINAL_RISK


def choose_terminal(bank: PathBank, belief: np.ndarray, source: int) -> tuple[int, float]:
    values = terminal_values(bank, belief, source)
    index = int(np.argmin(values))
    if values[index] >= TIMEOUT_PENALTY:
        return -1, TIMEOUT_PENALTY
    return index, float(values[index])


def choose_b(bank: PathBank, belief: np.ndarray, source: int = -1) -> tuple[int, float]:
    best_action, best_value = -1, TIMEOUT_PENALTY
    for a in range(len(ACTIONS)):
        pose = pose_for_action(a)
        value = bank.action_cost(source, a)
        for h in range(len(HYP)):
            for weight, obs, private_source in MODEL_NODES[h, a]:
                if weight <= 0:
                    continue
                mass = belief[h] * weight
                if public_stop(obs, pose):
                    value += mass * (0.0 if model_quality(h, pose, private_source) else FALSE_PENALTY)
                else:
                    bnew = posterior(belief, pose, obs)
                    _, second = choose_terminal(bank, bnew, a)
                    value += mass * second
        if value < best_value - 1e-12:
            best_action, best_value = a, value
    return best_action, float(best_value)


def choose_fixed(bank: PathBank, belief: np.ndarray, source: int = -1) -> tuple[int, int, float]:
    best = (-1, -1, TIMEOUT_PENALTY)
    for first in range(len(ACTIONS)):
        pose = pose_for_action(first)
        for second in range(-1, len(ACTIONS)):
            value = bank.action_cost(source, first)
            for h in range(len(HYP)):
                for weight, obs, private_source in MODEL_NODES[h, first]:
                    mass = belief[h] * weight
                    if public_stop(obs, pose):
                        value += mass * (0.0 if model_quality(h, pose, private_source) else FALSE_PENALTY)
                    elif second < 0:
                        value += mass * TIMEOUT_PENALTY
                    else:
                        # Same fixed second action for all possible public images,
                        # but its risk is averaged under the posterior given that image.
                        bnew = posterior(belief, pose, obs)
                        value += mass * (bank.action_cost(first, second) + bnew @ TERMINAL_RISK[:, second])
            if value < best[2] - 1e-12:
                best = (first, second, float(value))
    return best


def choose_entropy(bank: PathBank, belief: np.ndarray, source: int = -1) -> tuple[int, float]:
    def entropy(b: np.ndarray) -> float:
        return float(-np.sum(b[b > 0] * np.log(b[b > 0])))
    prior_h = entropy(belief)
    best = (-1, -math.inf)
    for a in range(len(ACTIONS)):
        pose = pose_for_action(a)
        gain = 0.0
        for h in range(len(HYP)):
            for weight, obs, _ in MODEL_NODES[h, a]:
                gain += belief[h] * weight * (prior_h - entropy(posterior(belief, pose, obs)))
        utility = gain - 0.05 * bank.action_cost(source, a)
        if utility > best[1] + 1e-12:
            best = (a, utility)
    return best


def exhaustive_reference(bank: PathBank, belief: np.ndarray) -> tuple[int, float]:
    """Independent explicit root/second enumeration of the same fixed quadrature."""
    candidates = []
    for root in range(len(ACTIONS)):
        pose = pose_for_action(root)
        future = 0.0
        for h, prior_mass in enumerate(belief):
            for weight, obs, private_source in MODEL_NODES[h, root]:
                branch_mass = prior_mass * weight
                if public_stop(obs, pose):
                    future += branch_mass * (0.0 if model_quality(h, pose, private_source) else FALSE_PENALTY)
                    continue
                joint = np.array([belief[k] * math.exp(log_likelihood(k, pose, obs)) for k in range(len(HYP))])
                if not np.isfinite(joint).all() or joint.sum() == 0:
                    joint = belief.copy()
                else:
                    joint /= joint.sum()
                second_values = np.array([
                    bank.action_cost(root, a) + sum(joint[k] * TERMINAL_RISK[k, a] for k in range(len(HYP)))
                    for a in range(len(ACTIONS))])
                future += branch_mass * min(TIMEOUT_PENALTY, float(second_values.min()))
        candidates.append(bank.action_cost(-1, root) + future)
    value = min(candidates)
    return (int(np.argmin(candidates)), float(value)) if value < TIMEOUT_PENALTY else (-1, TIMEOUT_PENALTY)


def draw_worlds(n: int, seed: int, *, shifted: bool = False) -> list[World]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        c0 = float(rng.normal(.006 if shifted else 0.0, .003))
        out.append(World(int(rng.integers(0, len(HYP))), c0, float(rng.normal(0, .006)),
                         float(rng.normal(0, .002)), float(rng.normal(0, .003)),
                         float(rng.normal(0, .002))))
    return out


def run_episode(method: str, bank: PathBank, world: World, world_id: int, seed: int) -> dict:
    belief = PRIOR.copy()
    prev = -1
    clock = 0.0
    motion = 0.0
    compute = 0.0
    first_quality_time = math.nan
    false_accept = False
    public_stopped = False
    trace = []
    fixed_second = None
    for step in range(2):
        t0 = time.perf_counter()
        if method == "A_one_step_greedy":
            action, _ = choose_terminal(bank, belief, prev)
        elif method == "B_two_step_contingent":
            action, _ = (choose_b(bank, belief, prev) if step == 0 else choose_terminal(bank, belief, prev))
        elif method == "two_step_belief_MPC_reference":
            action, _ = (exhaustive_reference(bank, belief) if step == 0 else choose_terminal(bank, belief, prev))
        elif method == "same_budget_fixed_sequence" or method == "B_no_branch":
            if step == 0:
                action, fixed_second, _ = choose_fixed(bank, belief, prev)
            else:
                action = fixed_second if fixed_second is not None else -1
        elif method == "entropy_greedy":
            action, _ = choose_entropy(bank, belief, prev) if step == 0 else choose_terminal(bank, belief, prev)
        elif method == "B_no_capability_proxy":
            action, _ = (choose_b(bank, belief, prev) if step == 0 else choose_terminal(bank, belief, prev))
        else:
            raise ValueError(method)
        planning_s = time.perf_counter() - t0
        compute += planning_s
        clock += planning_s
        if action < 0 or clock >= DEADLINE:
            break
        pose = pose_for_action(action)
        duration = bank.duration(prev, action)
        motion += duration
        clock += duration + float(CFG["planning"]["observation_time_s"])
        event = sensor(world, pose, world_id, step, action, seed)
        good = independent_quality(event, world, pose)
        stop = public_stop(event.public, pose)
        if good and not math.isfinite(first_quality_time):
            first_quality_time = clock
        if stop:
            public_stopped = True
            false_accept = not good
        trace.append(dict(step=step, action=ACTIONS[action]["id"], action_id=action,
                          prior_belief=belief.tolist(), observation=event.public.public_dict(),
                          public_stop=stop, evaluator_quality=good, private_source_for_audit=event.source,
                          clock_s=clock, planning_s=planning_s))
        if stop or clock >= DEADLINE:
            break
        belief = posterior(belief, pose, event.public)
        prev = action
    return dict(method=method, world_id=world_id, latent_mode_for_audit=world.mode,
                success=bool(math.isfinite(first_quality_time) and first_quality_time <= DEADLINE),
                first_pilot_quality_time_s=float(first_quality_time) if math.isfinite(first_quality_time) else None,
                censored_time_s=float(first_quality_time) if math.isfinite(first_quality_time) and first_quality_time <= DEADLINE else DEADLINE,
                false_accept=bool(false_accept), public_stopped=public_stopped,
                no_find=not (math.isfinite(first_quality_time) and first_quality_time <= DEADLINE), n_observations=len(trace),
                compute_s=compute, motion_s=motion, total_s=min(clock, DEADLINE), trace=trace)


def check_causal_noninterference(bank: PathBank) -> dict:
    """Counterfactual different hidden worlds, forcibly same public history."""
    a = draw_worlds(2, 9191)
    a[1] = World(3 if a[0].mode != 3 else 0, .01, -.01, .003, -.003, .002)
    obs_prefix = [Observation(False), Observation(True, .004, .11, .61, .04)]
    results = {}
    for method in ("A_one_step_greedy", "B_two_step_contingent", "same_budget_fixed_sequence", "entropy_greedy"):
        histories = []
        for _world in a:
            b, prev, actions, beliefs = PRIOR.copy(), -1, [], []
            fixed_second = None
            for step, obs in enumerate(obs_prefix):
                if method == "A_one_step_greedy": act, _ = choose_terminal(bank, b, prev)
                elif method == "B_two_step_contingent": act, _ = choose_b(bank, b, prev) if step == 0 else choose_terminal(bank, b, prev)
                elif method == "same_budget_fixed_sequence":
                    if step == 0: act, fixed_second, _ = choose_fixed(bank, b, prev)
                    else: act = fixed_second
                else: act, _ = choose_entropy(bank, b, prev) if step == 0 else choose_terminal(bank, b, prev)
                actions.append(act)
                if act < 0: break
                b = posterior(b, pose_for_action(act), obs)
                beliefs.append(b.tolist())
                prev = act
            histories.append(dict(actions=actions, beliefs=beliefs))
        results[method] = dict(pass_equal=histories[0] == histories[1], histories=histories)
    return results


def check_prior_sensitivity(bank: PathBank) -> dict:
    """Change only a legal public mode prior, never the patient truth."""
    priors = [np.array([.7, .1, .1, .1]), np.array([.1, .1, .1, .7])]
    rows = []
    for prior in priors:
        a_action, a_value = choose_terminal(bank, prior, -1)
        b_action, b_value = choose_b(bank, prior)
        rows.append(dict(prior=prior.tolist(), A_action=a_action, A_value=a_value,
                         B_action=b_action, B_value=b_value,
                         terminal_action_values=terminal_values(bank, prior, -1).tolist()))
    score_difference = float(np.max(np.abs(np.array(rows[0]["terminal_action_values"])
                                            - np.array(rows[1]["terminal_action_values"]))))
    return dict(rows=rows, score_difference=score_difference,
                actions_change=rows[0]["A_action"] != rows[1]["A_action"] or rows[0]["B_action"] != rows[1]["B_action"])


def summarize(rows: list[dict]) -> dict:
    methods = sorted(set(r["method"] for r in rows))
    result = {}
    for method in methods:
        arr = [r for r in rows if r["method"] == method]
        result[method] = dict(n=len(arr), pilot_single_frame_quality_rate=float(np.mean([r["success"] for r in arr])),
                              mean_censored_time_s=float(np.mean([r["censored_time_s"] for r in arr])),
                              false_accept_rate=float(np.mean([r["false_accept"] for r in arr])),
                              no_find_rate=float(np.mean([r["no_find"] for r in arr])),
                              mean_observations=float(np.mean([r["n_observations"] for r in arr])),
                              mean_motion_s=float(np.mean([r["motion_s"] for r in arr])),
                              mean_compute_s=float(np.mean([r["compute_s"] for r in arr])))
    return result


def main() -> None:
    protocol_sha = hashlib.sha256((HERE / "PROTOCOL.json").read_bytes()).hexdigest()
    amendment_sha = hashlib.sha256((HERE / "AMENDMENT_TEMPLATE_PRIOR.txt").read_bytes()).hexdigest()
    started = time.perf_counter()
    bank = PathBank()
    methods = CFG["planning"]["methods"]
    worlds = draw_worlds(int(CFG["evaluation"]["n_test_worlds"]), int(CFG["evaluation"]["test_seed"]))
    shifted = draw_worlds(int(CFG["evaluation"]["n_shift_worlds"]), int(CFG["evaluation"]["shift_seed"]), shifted=True)
    ref_action, ref_value = exhaustive_reference(bank, PRIOR)
    b_action, b_value = choose_b(bank, PRIOR)
    causal = check_causal_noninterference(bank)
    prior_sensitivity = check_prior_sensitivity(bank)
    all_rows = []
    for cohort, group, seed in (("test", worlds, int(CFG["evaluation"]["test_seed"])),
                                ("shift", shifted, int(CFG["evaluation"]["shift_seed"]))):
        for world_id, world in enumerate(group):
            for method in methods:
                row = run_episode(method, bank, world, world_id, seed)
                row["cohort"] = cohort
                all_rows.append(row)
    report = dict(scope=CFG["scope"], protocol_sha256=protocol_sha, amendment_sha256=amendment_sha,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  generic_template="synthetic chart line, not current patient or SMPL-X production atlas",
                  not_la_contract="Only one image per short action; no 15mm/.5s/5-frame LA. All success numbers are pilot single-view quality, not first qualified LA.",
                  dmp_source=str(Path(fit_discrete_dmp.__code__.co_filename)),
                  shared_bank_setup_wall_s=bank.setup_wall_s,
                  max_dmp_endpoint_error=bank.max_endpoint_error,
                  max_dmp_nominal_translation_speed=bank.max_nominal_speed,
                  policy_reference=dict(B_root_action=b_action, exhaustive_root_action=ref_action,
                                        B_value=b_value, exhaustive_value=ref_value,
                                        absolute_value_error=abs(b_value-ref_value)),
                  causal_counterfactual=causal, prior_sensitivity=prior_sensitivity,
                  summary={cohort:summarize([r for r in all_rows if r["cohort"] == cohort]) for cohort in ("test", "shift")},
                  rows=all_rows,
                  limitations=["Synthetic continuous-feature likelihood, no real B-mode.", "Generic synthetic chart template; not full production SMPL-X atlas.", "Primitive DMP action bank, no closed-loop image servo within each action.", "No actual bottom-QP or contact implementation in this isolated benchmark.", "Four discrete nominal hypotheses plus smooth held-out deviations; not calibrated patient population probabilities.", "B equals known two-step belief MPC reference by construction; no novel POMDP superiority."])
    out = HERE / "results.json"
    out.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(dict(summary=report["summary"], policy_reference=report["policy_reference"],
                          causal={k:v["pass_equal"] for k,v in causal.items()},
                          elapsed_s=time.perf_counter()-started), indent=2))
    assert abs(b_value-ref_value) < 1e-9
    assert all(v["pass_equal"] for v in causal.values())
    assert prior_sensitivity["score_difference"] > 1e-6


if __name__ == "__main__":
    main()
