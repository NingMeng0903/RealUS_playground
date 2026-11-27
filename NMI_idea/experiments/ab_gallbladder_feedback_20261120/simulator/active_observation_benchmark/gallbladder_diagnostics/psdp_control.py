"""Streaming ProDMP MPC and executable short probes for the PSDP experiment."""

from dataclasses import replace
from itertools import product

import numpy as np
from scipy.spatial.distance import cdist

from psdp_scene import HORIZON, review, rotation_cost
from gallbladder_mpc import CONTACT, FRAME_DT, intercostal_contacts
from servo import FRAME_STEPS, Motion


ACTIONS = ("axis", "fan_minus", "fan_plus", "subcostal_medial", "subcostal_lateral", "intercostal")
HANDOFF = "mpc"
FIXED = (("axis",), ("fan_minus",), ("fan_plus",), ("subcostal_medial",),
         ("subcostal_lateral",), ("intercostal",), ("axis", "fan_minus"), ("axis", "fan_plus"))
CONTACT_SCALE = np.array([1., 2.3, .25, .25])


class Controller:
    def __init__(self, scene):
        self.scene = scene
        body = scene.body
        entries = np.array(list(intercostal_contacts(body)), dtype=float).reshape(-1, 2)
        self.intercostal = entries
        subcostal = [(float(scene.envelope(theta)) - offset, theta)
                     for theta, offset in product((-59., -50., -41., -32., -23., -14.), (15., 27., 40.))]
        other = list(product((-225., -198., -178., -154.), (-75., -64., -49., -37., -25.)))
        contacts = [*subcostal, *entries[::2], *other]
        poses = [self.aim(xy, scene.nominal.center, scene.nominal.axis) for xy in contacts]
        self.bank = np.unique(np.clip(
            np.vstack([np.asarray(poses) + [0., 0., roll, rock]
                       for roll, rock in ((0., 0.), (-16., -5.), (16., 5.), (0., -13.), (0., 13.))]),
            *CONTACT), axis=0)
        self.positions = body.surface(self.bank)
        self.coverage, self.visible = self.forecasts(self.bank)
        self.transfer = np.exp(-cdist(self.positions, self.positions) / 92.
                               - cdist(self.bank[:, 2:], self.bank[:, 2:]) / 120.)
        self.join_position = 0.
        self.join_velocity = 0.
        self.decisions = 0
        self._endpoints = {}

    def aim(self, xy, center, axis):
        xy = np.clip(xy, CONTACT[0, :2], CONTACT[1, :2])
        return np.clip(self.scene.body.aim(xy, center, axis), *CONTACT)

    def forecasts(self, goals):
        predictions = [self.scene.project(q) for q in goals]
        return (np.stack([p[2] for p in predictions]),
                np.stack([p[1].mean(axis=1) for p in predictions]))

    def propose(self, state, *, source=None):
        """Same request dictionary for every selector, including an explicit MPC handoff."""
        source = state if source is None else source
        center, axis, _ = self.scene.estimate(source)
        aim = self.aim(state.q[:2], center, axis)
        fan = np.array([0., 0., 14., 7.])
        actions = {"axis": aim, "fan_minus": np.clip(aim - fan, *CONTACT),
                   "fan_plus": np.clip(aim + fan, *CONTACT)}
        for name, delta in (("subcostal_medial", 7.), ("subcostal_lateral", -7.)):
            theta = np.clip(state.q[1] + delta, -59., 7.5)
            offset = self.scene.envelope(np.clip(state.q[1], -59., 7.5)) - state.q[0]
            y = self.scene.envelope(theta) - np.clip(offset, 16., 39.)
            actions[name] = self.aim([y, theta], center, axis)
        if len(self.intercostal):
            ranked = np.argsort(np.linalg.norm((self.intercostal - state.q[:2]) * [1., 2.3], axis=1))[:9]
            alternatives = np.asarray([self.aim(self.intercostal[i], center, axis) for i in ranked])
            coverage, visible = self.forecasts(alternatives)
            weights = self.scene.gall_weights(source)
            displace = np.linalg.norm(self.scene.body.surface(alternatives) - self.scene.body.surface(state.q), axis=1)
            value = (coverage @ weights + .09 * (visible @ weights)) * np.exp(-displace / 78.)
            actions["intercostal"] = alternatives[np.argmax(value)]
        return {HANDOFF: None, **actions}

    def feature_rows(self, state, actions, deadline):
        scene = self.scene
        weights = scene.gall_weights(state)
        liver, liver_width = scene.liver_prior.moment(state.liver)
        center, axis, length = scene.estimate(state)
        q = state.q
        xyz = scene.body.surface(q)
        view = scene.body.view(q)
        rel = center - xyz
        data = np.full((7, 9), np.nan)
        for j, packet in enumerate(state.recent, start=7 - len(state.recent)):
            data[j, :3] = packet.reading[0, [0, 1, 3]]
            data[j, 3:] = packet.marks.ravel()
        # Missing readouts are represented by flags as well as the filled values.
        available = np.isfinite(data)
        data = np.where(available, data, 0.)
        denom = np.maximum(available.sum(axis=0), 1)
        h, sin_theta, cos_theta = view.address
        common = np.r_[h, sin_theta, cos_theta, q, state.v, (deadline - state.tick) * FRAME_DT,
                       [rel @ view.right, rel @ view.forward, rel @ view.normal], liver, liver_width,
                       axis, length, 1. / (weights @ weights), weights,
                       available[-1].astype(float), data[-1], available.mean(axis=0), data.sum(axis=0) / denom]
        matrix = []
        for name, target in actions.items():
            target = q if target is None else target
            pts, support, quality = scene.project(target)
            means = weights @ support
            xds = np.einsum("i,ijk->jk", weights, pts)
            d = np.linalg.norm(scene.body.surface(target) - xyz)
            p = weights @ quality
            j = (HANDOFF, *ACTIONS).index(name)
            matrix.append(np.r_[common, np.eye(1 + len(ACTIONS))[j], target, target - q, d,
                                p, np.sqrt(max(weights @ quality ** 2 - p ** 2, 0.)), means, xds.ravel()])
        return np.asarray(matrix)

    def checked_plan(self, state, target):
        plan = self.scene.motor.replan(state.q, state.v, target)
        if plan is not None:
            self.join_position = max(self.join_position, float(np.max(np.abs(plan.x[0] - state.q))))
            self.join_velocity = max(self.join_velocity, float(np.max(np.abs(plan.v[0] - state.v))))
            if self.join_position > 1.e-7 or self.join_velocity > 1.e-7:
                raise RuntimeError("non-continuous ProDMP splice")
        return plan

    def advance(self, state, motion, tape, *, trace=None, phase):
        q, v = motion.x[-1], motion.v[-1]
        body = self.scene.body
        travel = np.linalg.norm(np.diff(body.surface(motion.x), axis=0), axis=1).sum()
        turn = rotation_cost(body.view(state.q), body.view(q))
        next_state = replace(state, q=q, v=v, travel=state.travel + travel, turn=state.turn + turn)
        packet = self.scene.observe(q, state.tick + 1, tape)
        next_state = self.scene.assimilate(next_state, packet)
        if trace is not None:
            trace.append(self.scene.trace_row(next_state, packet, phase))
        return next_state

    def segment(self, state, target, tape, deadline, *, trace=None, phase="probe", stop=True):
        """Execute a finite ProDMP request while assimilating each acquired frame."""
        if target is None or (stop and review(state) is not None):
            return state
        motion = self.checked_plan(state, target)
        if motion is None:
            return state
        hold = np.broadcast_to(motion.x[-1], (3 * FRAME_STEPS, 4))
        x = np.vstack((motion.x, hold))
        v = np.vstack((motion.v, np.zeros_like(hold)))
        for k in range(FRAME_STEPS, len(x), FRAME_STEPS):
            if state.tick >= deadline or (stop and review(state) is not None):
                break
            state = self.advance(state, Motion(x[k - FRAME_STEPS:k + 1], v[k - FRAME_STEPS:k + 1]),
                                 tape, trace=trace, phase=phase)
        return state

    def rank(self, state, visits, local):
        scene = self.scene
        w = scene.gall_weights(state)
        position = scene.body.surface(state.q)
        proximity = np.linalg.norm(self.positions - position, axis=1)
        roll = np.linalg.norm(self.bank[:, 2:] - state.q[2:], axis=1)
        recovery = np.exp(-.14 * visits)
        discount = np.exp(-proximity / 83. - roll / 106.)
        immediate = self.coverage @ w + .095 * (self.visible @ w)
        base = immediate * recovery * discount
        shortlist = np.argsort(-base, kind="stable")[:14]
        tails = np.argsort(-immediate * recovery, kind="stable")[:20]
        partial = np.maximum(self.visible[shortlist] - self.coverage[shortlist], 0.)
        missed = 1. - self.visible[shortlist]
        follow = np.stack((partial, missed)) * w
        after = ((follow @ self.coverage[tails].T) * self.transfer[shortlist[:, None], tails]
                 * recovery[tails])
        after[:, shortlist[:, None] == tails] = 0.
        scores = base[shortlist] + .47 * discount[shortlist] * after.max(axis=-1).sum(axis=0)
        chart_scores = {int(i): float(value) for i, value in zip(shortlist, scores)}
        # Also allow pose corrections generated at the actual contact.
        candidate = np.asarray(list(local.values()))
        coverage, visible = self.forecasts(candidate)
        d = np.linalg.norm(scene.body.surface(candidate) - position, axis=1)
        turn = np.linalg.norm(candidate[:, 2:] - state.q[2:], axis=1)
        prior_visits = np.exp(-cdist(candidate * CONTACT_SCALE, self.bank * CONTACT_SCALE) ** 2 / (2. * 9. ** 2)) @ visits
        local_discount = np.exp(-d / 83. - turn / 106.)
        follow = np.stack((np.maximum(visible - coverage, 0.), 1. - visible)) * w
        transit = np.exp(-cdist(scene.body.surface(candidate), self.positions[tails]) / 92.
                         - cdist(candidate[:, 2:], self.bank[tails, 2:]) / 120.)
        after = (follow @ self.coverage[tails].T) * transit * recovery[tails]
        local_score = local_discount * ((coverage @ w + .095 * (visible @ w)) * np.exp(-.29 * prior_visits)
                                        + .47 * after.max(axis=-1).sum(axis=0))
        goals = [(score, self.bank[i]) for i, score in chart_scores.items()]
        goals.extend((float(score), q) for score, q in zip(local_score, candidate))
        return sorted(goals, key=lambda pair: pair[0], reverse=True)

    def mpc(self, state, tape, deadline, *, trace=None):
        """Consider local correction, continuation, another window or revisiting at every frame."""
        visits = np.zeros(len(self.bank))
        for packet in state.recent:
            visits += self.visit(packet.q)
        plan, goal, local = None, None, None
        while state.tick < deadline and review(state) is None:
            self.decisions += 1
            # The full menu is rescored every frame. Regenerate continuous refs
            # when physical progress or the posterior shifts enough to move them.
            estimate = self.scene.estimate(state)[0]
            if local is None or np.linalg.norm(estimate - location) > 2.5 or np.linalg.norm(state.q[:2] - contact) > 4.:
                local = self.propose(state)
                del local[HANDOFF]
                location, contact = estimate, state.q[:2]
            ordered = self.rank(state, visits, local)
            if state.captures and state.tick - state.captures[-1].tick <= 2:
                ordered.insert(0, (float("inf"), state.q))
            running = plan is not None and len(plan.x) > FRAME_STEPS
            if running:
                current = next((score for score, target in ordered if np.array_equal(target, goal)), None)
                if current is not None and ordered[0][0] <= current + .031:
                    ordered.insert(0, (current, goal))
            for _, target in ordered[:7]:
                if running and np.linalg.norm((goal - target) * CONTACT_SCALE) < 2.4:
                    break
                ref = self.checked_plan(state, target)
                if ref is not None:
                    plan, goal = ref, target
                    break
            if plan is None or len(plan.x) <= FRAME_STEPS:
                idle = np.broadcast_to(state.q, (FRAME_STEPS + 1, 4))
                block = Motion(idle, np.zeros_like(idle))
                if np.max(np.abs(state.v)) > 1.e-7:
                    raise RuntimeError("stopped a moving probe without a feasible splice")
            else:
                block = Motion(plan.x[:FRAME_STEPS + 1], plan.v[:FRAME_STEPS + 1])
                plan = Motion(plan.x[FRAME_STEPS:], plan.v[FRAME_STEPS:])
            state = self.advance(state, block, tape, trace=trace, phase="mpc")
            visits += self.visit(state.q)
        return state

    def visit(self, q):
        squared = (((self.bank - q) * CONTACT_SCALE) ** 2).sum(axis=1)
        return .36 * np.exp(-squared / (2. * 6.5 ** 2))

    def outcome(self, state):
        frame = review(state)
        return {"correct": bool(frame is not None and frame.qualified and frame.genuine),
                "wrong": bool(frame is not None and not (frame.qualified and frame.genuine)),
                "acquired": state.acquired is not None, "travel_mm": float(state.travel),
                "turn_deg": float(state.turn), "time_s": state.tick * FRAME_DT,
                "accepted_tick": None if frame is None else frame.tick, "first_quality_tick": state.acquired}


def decision_deadline(prefix):
    return prefix.tick + HORIZON


def public_without_new_images(before, after):
    """Keep the measured exit, time and cost. Carry in only pre-probe images for selection."""
    return replace(after, liver=before.liver, evidence=before.evidence,
                   recent=before.recent, captures=before.captures)
