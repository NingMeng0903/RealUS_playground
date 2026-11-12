"""Replanning MPC, short public-history options, and complete paired rollouts."""
from __future__ import annotations

from dataclasses import dataclass
import math

from numba import njit
import numpy as np
from scipy.stats import binomtest

from acoustics import DISTANCE, PIXELS


# Suffix MPC uses a finite stochastic horizon. All arms use exactly this
# implementation whenever control belongs to MPC, including imagined tails.
CONFIDENCE = .87
MPC_SCENARIOS = 10
MPC_TRAVEL_PRICE = .00035
MPC_TURN_PRICE = .00015
MPC_RAIL_PRICE = .00015
RISK_PENALTY = 1.
COST_MARGINS = np.array([2., 2.])
HANDOFF, COMPLEMENT, RECOVER, FIXED, OPEN = 0, 1, 2, 3, 4
RULE_ONE, RULE_FIXED, RULE_STATUS, RULE_GEOMETRY, RULE_LOCKED = 0, 1, 2, 3, 4


@dataclass(frozen=True)
class Scene:
    """Public model arrays, with a separate outcome generator for each case."""

    likelihood: np.ndarray
    correct: np.ndarray
    detect: np.ndarray
    cdf: np.ndarray
    to_address: np.ndarray
    site: np.ndarray
    tilt: np.ndarray
    rho: np.ndarray
    edge_costs: np.ndarray
    targets: np.ndarray
    start: int
    reserve: np.ndarray
    potential: np.ndarray
    needs: np.ndarray
    material_coordinate: np.ndarray
    anchor: int
    clock_unit: float

    def public(self):
        return (self.likelihood, self.correct, self.to_address, self.site, self.tilt, self.rho,
                self.edge_costs, self.targets, self.potential, self.needs,
                self.material_coordinate, self.anchor)


@njit(cache=True)
def capture_draw(detect, cdf, slot, address, state, noise):
    # Exogenous detection and image-bin innovations are separate by return.
    best_kinds = np.empty(3, dtype=np.int64)
    best_pixels = np.empty(3, dtype=np.int64)
    best_distance = math.inf
    ties = 0
    count = 0
    for kind in range(3):
        if noise[slot, kind] < detect[slot, address, state, kind]:
            count += 1
            pixel = min(np.searchsorted(cdf[slot, address, state, kind], noise[slot, kind + 3]), PIXELS - 1)
            distance = DISTANCE[pixel]
            if distance < best_distance - 1.e-12:
                best_distance = distance
                ties = 0
            if abs(distance - best_distance) <= 1.e-12:
                best_pixels[ties] = pixel
                best_kinds[ties] = kind
                ties += 1
    if count == 0:
        return 0, -1
    index = int(noise[slot, 6] * ties)
    return 1 + best_pixels[index] + PIXELS * (count > 1), best_kinds[index]


@njit(cache=True)
def allowed(costs, at, goal, reserve):
    for k in range(len(reserve)):
        if not costs[at, goal, k] <= reserve[k] + 1.e-8:
            return False
    return True


@njit(cache=True)
def dot_model(prior, weight):
    total = 0.
    for h in range(len(prior)):
        total += prior[h] * weight[h]
    return total


@njit(cache=True)
def posterior(prior, likelihood, correct, slot, address, obs):
    p = likelihood[slot, address, obs] * prior
    denominator = p.sum()
    if denominator <= 1.e-260:
        # A broad contamination floor handles a live out-of-support packet.
        # There is no confidence for declaring its identity.
        return prior.copy(), 0.
    numerator = dot_model(prior, correct[slot, address, obs])
    return p / denominator, numerator / denominator


@njit(cache=True)
def expected_claim(prior, likelihood, correct, slot, address):
    yes, wrong = 0., 0.
    for obs in range(1, likelihood.shape[2]):
        probability = dot_model(prior, likelihood[slot, address, obs])
        joint = dot_model(prior, correct[slot, address, obs])
        if probability > 1.e-260 and joint >= CONFIDENCE * probability:
            yes += joint
            wrong += probability - joint
    return yes, wrong


@njit(cache=True)
def claim_under_mass(belief, good, likelihood, correct, slot, address):
    """Joint success/error mass, with the claim decision made under belief."""
    yes, wrong = 0., 0.
    for obs in range(1, likelihood.shape[2]):
        ordinary = likelihood[slot, address, obs]
        associated = correct[slot, address, obs]
        p = dot_model(belief, ordinary)
        if p > 1.e-260 and dot_model(belief, associated) >= CONFIDENCE * p:
            joint = dot_model(good, associated)
            yes += joint
            wrong += max(0., dot_model(good, ordinary) - joint)
    return yes, wrong


@njit(cache=True)
def advance_joint(prior, good, likelihood, correct, slot, address, obs, claim):
    ordinary = likelihood[slot, address, obs]
    p = dot_model(prior, ordinary)
    if p <= 1.e-260:
        return good.copy(), 0.
    inherited = dot_model(good, ordinary) / p
    new = good * (correct[slot, address, obs] if claim else ordinary) / p
    return new, max(0., inherited - new.sum())


@njit(cache=True)
def motion_price(costs, start, end):
    return (MPC_TRAVEL_PRICE * costs[start, end, 0]
            + MPC_TURN_PRICE * costs[start, end, 1]
            + MPC_RAIL_PRICE * costs[start, end, 2])


@njit(cache=True)
def remaining_value(potential, needs, slots, at, progress, reserve):
    for k in range(len(reserve)):
        if needs[slots, at, progress, k] > reserve[k] + 1.e-8:
            return 0.
    return potential[slots, at, progress]


@njit(cache=True)
def mpc_action(model, prior, good, at, slot, clock, progress, reserve):
    """Optimize a fixed two-capture plan, predicting actual mask updates.

    The second command is fixed across first-mask draws during optimization.
    Execute only the first command and replan after a real observation.
    """
    like, correct, address, sites, _, _, costs, targets, potential, needs, _, _ = model
    cap = len(potential) - 1
    if slot >= cap or progress >= len(targets):
        return -1
    best, selected = -math.inf, -1
    for first in range(len(sites)):
        if not allowed(costs, at, first, reserve):
            continue
        first_price = motion_price(costs, at, first)
        first_clock = clock + int(costs[at, first, 3])
        if slot + 1 == cap:
            if sites[first] != targets[progress] or progress + 1 != len(targets):
                continue
            c, w = claim_under_mass(prior, good, like, correct, first_clock, address[first])
            value = c - RISK_PENALTY * w - first_price
            if value > best + 1.e-12:
                best, selected = value, first
            continue
        remaining = reserve - costs[at, first]
        masses = np.empty(like.shape[2])
        for obs in range(len(masses)):
            masses[obs] = dot_model(prior, like[first_clock, address[first], obs])
        cum = np.cumsum(masses)
        cum[-1] = 1.
        next_priors = np.empty((MPC_SCENARIOS, len(prior)))
        next_progress = np.empty(MPC_SCENARIOS, dtype=np.int64)
        next_good = np.empty_like(next_priors)
        good_mass = np.empty(MPC_SCENARIOS)
        loss = np.empty(MPC_SCENARIOS)
        for i in range(MPC_SCENARIOS):
            obs = min(np.searchsorted(cum, (i + .5) / MPC_SCENARIOS), len(cum) - 1)
            post, conf = posterior(prior, like, correct, first_clock, address[first], obs)
            next_priors[i] = post
            next_progress[i] = progress
            claim = sites[first] == targets[progress] and obs > 0 and conf >= CONFIDENCE
            if claim:
                next_progress[i] += 1
            g, new_loss = advance_joint(prior, good, like, correct, first_clock, address[first], obs, claim)
            next_good[i] = g
            good_mass[i] = g.sum()
            loss[i] = new_loss
        first_value = 0.
        for i in range(MPC_SCENARIOS):
            after = next_progress[i]
            future = remaining_value(potential, needs, cap - slot - 1, first, after, remaining)
            first_value += good_mass[i] * future - RISK_PENALTY * loss[i]
        value = first_value / MPC_SCENARIOS - first_price
        if value > best + 1.e-12:
            best, selected = value, first
        for second in range(len(sites)):
            if not allowed(costs, first, second, remaining):
                continue
            second_price = motion_price(costs, first, second)
            after_second = remaining - costs[first, second]
            second_clock = first_clock + int(costs[first, second, 3])
            value = 0.
            for i in range(MPC_SCENARIOS):
                after = next_progress[i]
                if after == len(targets):
                    value += good_mass[i] - RISK_PENALTY * loss[i]
                    continue
                old = remaining_value(potential, needs, cap - slot - 2, second, after, after_second)
                increment, wrong = 0., 0.
                if sites[second] == targets[after]:
                    c, wrong = claim_under_mass(next_priors[i], next_good[i], like, correct,
                                                second_clock, address[second])
                    advanced = remaining_value(potential, needs, cap - slot - 2,
                                               second, after + 1, after_second)
                    increment = c * advanced - (c + wrong) * old
                value += (good_mass[i] * old + increment - RISK_PENALTY * (wrong + loss[i])
                          - second_price)
            value = value / MPC_SCENARIOS - first_price
            if value > best + 1.e-12:
                best, selected = value, first
    return selected if best > 0. else -1


@njit(cache=True)
def residual_sign_error(prior, likelihood, material, slot, address):
    """Expected Bayes error of the task-local displacement sign after capture."""
    residual = 0.
    for obs in range(likelihood.shape[2]):
        negative, positive = 0., 0.
        for h in range(len(prior)):
            joint = prior[h] * likelihood[slot, address, obs, h]
            if material[h] >= 0.:
                positive += joint
            else:
                negative += joint
        residual += min(negative, positive)
    return residual


@njit(cache=True)
def best_followup(model, at, slot, clock, progress, reserve, prior, first, action_kind, public_only,
                  required_rho=-1):
    likelihood, correct, addresses, sites, tilts, rhos, costs, targets, potential, needs, material, anchor = model
    best, selected = -math.inf, -1
    if slot >= len(potential) - 1 or progress >= len(targets):
        return -1
    for goal in range(len(sites)):
        if not allowed(costs, at, goal, reserve):
            continue
        if required_rho >= 0 and rhos[goal] != required_rho:
            continue
        if action_kind == RECOVER:
            if sites[goal] != anchor or tilts[goal] != 0:
                continue
            priority = 0.
        else:
            if tilts[first] != 0 and tilts[goal] != -tilts[first]:
                continue
            if tilts[first] == 0 and tilts[goal] == 0:
                continue
            # Opposite viewing direction and a registered address preserve a
            # paired measurement of shared displacement and calibration.
            priority = .09 * (sites[goal] == sites[first]) + .05 * (sites[goal] == targets[progress])
            if not public_only:
                arrival = clock + int(costs[at, goal, 3])
                priority -= .6 * residual_sign_error(prior, likelihood, material, arrival, addresses[goal])
                if sites[goal] == targets[progress]:
                    yes, wrong = expected_claim(prior, likelihood, correct, arrival, addresses[goal])
                    priority += .5 * (yes - wrong)
        remaining = reserve - costs[at, goal]
        successor = 0
        for nxt in range(len(sites)):
            if sites[nxt] == targets[progress] and allowed(costs, goal, nxt, remaining):
                successor += 1
        future = remaining_value(potential, needs, len(potential) - slot - 2, goal, progress, remaining)
        score = (priority + future
                 + .025 * successor - motion_price(costs, at, goal)
                 + .005 * (rhos[goal] == rhos[first]))
        if score > best + 1.e-12:
            best, selected = score, goal
    return selected


@njit(cache=True)
def option_action(model, prior, option, at, slot, clock, progress, reserve, last_obs, owned):
    """Select an option action from public measurements and checked exits."""
    first, limit, rule, fixed_second, fixed_third = option
    _, _, _, _, _, rhos, costs, targets, potential, _, material, _ = model
    if first < 0 or owned >= limit or slot >= len(potential) - 1 or progress >= len(targets):
        return -1, HANDOFF
    if owned == 0:
        return (first, OPEN) if allowed(costs, at, first, reserve) else (-1, HANDOFF)
    if rule == RULE_FIXED:
        goal = fixed_second if owned == 1 else fixed_third
        if goal >= 0 and allowed(costs, at, goal, reserve):
            return goal, FIXED
        return -1, HANDOFF
    if progress > 0:
        return -1, HANDOFF
    if rule in (RULE_GEOMETRY, RULE_LOCKED) and last_obs > 0:
        positive = 0.
        for h in range(len(prior)):
            if material[h] >= 0.:
                positive += prior[h]
        if max(positive, 1. - positive) >= .88:
            return -1, HANDOFF
    kind = RECOVER if last_obs == 0 else COMPLEMENT
    required = rhos[first] if rule == RULE_LOCKED else -1
    reference = first if owned == 1 else at
    goal = best_followup(model, at, slot, clock, progress, reserve, prior, reference,
                         kind, rule == RULE_STATUS, required)
    if goal < 0:
        return -1, HANDOFF
    return goal, kind


@njit(cache=True)
def simulate(model, true_detect, true_cdf, true_state, noise, prior, start, reserve, option):
    """One entire episode. Controller calls never see truth or the noise tape."""
    like, correct, addresses, sites, _, _, costs, targets, potential, _, _, _ = model
    belief = prior.copy()
    good = prior.copy()
    remaining = reserve.copy()
    at, progress, obs, owned, handoff = start, 0, 0, 0, -1
    wrong = False
    cap = len(potential) - 1
    trace = np.full((cap, 11), -1., dtype=np.float64)
    total = np.zeros(len(reserve))
    min_ess = 1. / dot_model(belief, belief)
    steps = 0
    clock = 0
    for slot in range(cap):
        goal, stage = -1, HANDOFF
        if handoff < 0:
            goal, stage = option_action(model, belief, option, at, slot, clock, progress, remaining, obs, owned)
            if goal < 0:
                handoff = slot
        if handoff >= 0:
            goal = mpc_action(model, belief, good, at, slot, clock, progress, remaining)
        else:
            owned += 1
        if goal < 0:
            break
        if not allowed(costs, at, goal, remaining):
            raise ValueError("selected edge violates the shared resource check")
        expense = costs[at, goal]
        clock += int(expense[3])
        obs, kind = capture_draw(true_detect, true_cdf, clock, addresses[goal], true_state, noise)
        previous = belief
        belief, conf = posterior(previous, like, correct, clock, addresses[goal], obs)
        min_ess = min(min_ess, 1. / dot_model(belief, belief))
        claim = obs > 0 and sites[goal] == targets[progress] and conf >= CONFIDENCE
        good, _ = advance_joint(previous, good, like, correct, clock, addresses[goal], obs, claim)
        if claim:
            progress += 1
            wrong = wrong or kind != 0
        trace[slot, 0], trace[slot, 1], trace[slot, 2] = at, goal, stage
        trace[slot, 3], trace[slot, 4], trace[slot, 5] = obs, conf, int(claim)
        trace[slot, 6], trace[slot, 7], trace[slot, 8], trace[slot, 9] = kind, expense[0], expense[1], expense[2]
        trace[slot, 10] = clock
        remaining -= expense
        total += expense
        at = goal
        steps += 1
        if progress == len(targets):
            break
    return (int(progress == len(targets) and not wrong), int(wrong), steps, owned,
            handoff, progress, min_ess, total, trace)


@njit(cache=True)
def rollout(model, true_detect, true_cdf, indices, tapes, prior, start, reserve, option):
    records = np.empty((len(indices), 7))
    for i in range(len(indices)):
        success, wrong, steps, _, _, _, _, cost, _ = simulate(
            model, true_detect, true_cdf, indices[i], tapes[i], prior, start, reserve, option)
        records[i, 0], records[i, 1], records[i, 2] = success, wrong, cost[3]
        records[i, 3], records[i, 4], records[i, 5] = cost[0], cost[1], cost[2]
        records[i, 6] = steps
    return records


def nominal_potential(sites, costs, targets, cap, hit):
    """Shared approximate terminal value for the two-slot MPC optimizer."""
    count = len(targets)
    out = np.zeros((cap + 1, len(sites), count + 1))
    out[:, :, count] = 1.
    needed = np.full((*out.shape, costs.shape[-1]), np.inf)
    needed[:, :, count] = 0.
    # Geometry-limited opportunity over later slots. Each recognized image
    # advances the publicly specified next station.
    for n in range(1, cap + 1):
        for at in range(len(sites)):
            for progress in range(count):
                values = []
                for goal in np.flatnonzero(np.isfinite(costs[at, :, 0])):
                    here = out[n - 1, goal, progress]
                    if sites[goal] == targets[progress]:
                        here = hit * out[n - 1, goal, progress + 1] + (1. - hit) * here
                    next_progress = progress + int(sites[goal] == targets[progress])
                    needed[n, at, progress] = np.minimum(needed[n, at, progress],
                        costs[at, goal] + needed[n - 1, goal, next_progress])
                    values.append(here)
                out[n, at, progress] = max(values, default=0.)
    return out, needed


def library(scene, prior):
    """Freeze the same public proposal pool for every policy in a case."""
    model = scene.public()
    anchor, reserve, at = scene.anchor, scene.reserve, scene.start
    root = mpc_action(model, prior, prior, at, 0, 0, 0, reserve)
    candidates = [j for j in range(len(scene.site)) if allowed(scene.edge_costs, at, j, reserve)]
    # Each signed view and admissible task-equivalent realization are
    # represented. Tie-breaking uses checked physical costs and node ID.
    candidates.sort(key=lambda j: (scene.site[j] not in (anchor, scene.targets[0]),
                                   scene.tilt[j] == 0, scene.edge_costs[at, j, 0], j))
    roots = candidates[:8]
    if root >= 0 and root not in roots:
        roots = [*roots[:7], root]
    result = {}
    for j in roots:
        remaining = reserve - scene.edge_costs[at, j]
        clock = int(scene.edge_costs[at, j, 3])
        second = best_followup(model, j, 1, clock, 0, remaining, prior, j, COMPLEMENT, False)
        third = -1
        if second >= 0:
            remaining_second = remaining - scene.edge_costs[j, second]
            clock += int(scene.edge_costs[j, second, 3])
            third = best_followup(model, second, 2, clock, 0, remaining_second, prior, second,
                                  COMPLEMENT, False)
        result[j] = (second, third)
    return result


def empty_option():
    return np.array([-1, 0, RULE_ONE, -1, -1], dtype=np.int64)


def programs(roots, arm, scene, prior, cap=16):
    candidates = []
    for root, (second, third) in roots.items():
        if arm == "one":
            candidates.append((root, 1, RULE_ONE, -1, -1))
        elif arm == "fixed":
            if second >= 0:
                candidates.append((root, 2, RULE_FIXED, second, -1))
                if third >= 0:
                    candidates.append((root, 3, RULE_FIXED, second, third))
        else:
            if arm == "single_realization":
                equivalent = [j for j in roots if scene.site[j] == scene.site[root]
                              and scene.tilt[j] == scene.tilt[root]]
                representative = min(equivalent, key=lambda j: (
                    scene.rho[j] != scene.rho[scene.start],
                    scene.edge_costs[scene.start, j, 2], j))
                if root != representative:
                    continue
            rule = RULE_STATUS if arm == "status" else RULE_LOCKED if arm == "single_realization" else RULE_GEOMETRY
            if second >= 0:
                candidates.extend(((root, 2, rule, -1, -1), (root, 3, rule, -1, -1)))
    return [np.asarray(c, dtype=np.int64) for c in candidates[:cap]]


def plan_random(seed, prior, cap, n):
    # Distinct from evaluation case seeds and the physical acquisition tape.
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 597141]))
    # Stratification gives every candidate the same collection of plausible
    # cases. Independent screen and validation seeds still matter.
    u = (np.arange(n) + rng.random(n)) / n
    cdf = np.cumsum(prior)
    cdf[-1] = 1.
    idx = rng.permutation(np.searchsorted(cdf, u))
    return idx, rng.random((n, cap, 7))


def advantage(candidate, incumbent):
    delta = candidate - incumbent
    wins = int((delta[:, 0] == 1).sum())
    losses = int((delta[:, 0] == -1).sum())
    mean = delta.mean(axis=0)
    return {"delta": mean.tolist(), "wins_losses": [wins, losses],
            "success_sign_p": float(binomtest(wins, wins + losses, .5, alternative="greater").pvalue)
            if wins + losses else 1.}


def estimate(scene, prior, options, plans, base, compute_cache=None):
    model = scene.public()
    states, noise = plans
    scored = []
    for option in options:
        key = tuple(map(int, option))
        if compute_cache is not None and key in compute_cache:
            data = compute_cache[key]
        else:
            data = rollout(model, scene.detect, scene.cdf, states, noise, prior,
                           scene.start, scene.reserve, option)
            if compute_cache is not None:
                compute_cache[key] = data
        score = advantage(data, base)
        scored.append(dict(score, option=key, predicted_mean=data.mean(axis=0).tolist()))
    return scored


def choose_option(scene, prior, arm, roots, screen, check, base_screen, base_check,
                  screen_cache=None, validation_cache=None, unguarded=False):
    options = programs(roots, arm, scene, prior)
    screen_scores = estimate(scene, prior, options, screen, base_screen, screen_cache)
    # Risk and cost are separate guards on predictions. These empirical
    # model-screening quantities are not real outcome confidence bounds.
    def eligible(row):
        success, wrong, elapsed_ticks, travel, turn, _, _ = row["delta"]
        return (success > .0 and wrong <= .02 and elapsed_ticks * scene.clock_unit <= .10
                and np.all(np.array([travel, turn]) <= COST_MARGINS))
    feasible = [r for r in screen_scores if eligible(r) or unguarded]
    if not feasible:
        return empty_option(), {"screen": screen_scores, "validation": None, "reason": "screen"}
    selected = max(feasible, key=lambda r: (r["delta"][0], -r["delta"][1], -r["delta"][2]))
    choice = np.array(selected["option"], dtype=np.int64)
    validation, = estimate(scene, prior, [choice], check, base_check, validation_cache)
    pass_gate = (eligible(validation) and validation["success_sign_p"] <= .20) or unguarded
    return (choice if pass_gate else empty_option(),
            {"screen": screen_scores, "validation": validation, "reason": "execute" if pass_gate else "validation"})
