"""Continuous chart-valued goal optimization and 10 Hz visual ProDMP correction."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from model import Belief, CONTACT, Sensors, TARGET_PHASE, likelihood_margin, observe, passes
from pilot import LateralPrimitive
from NMI_idea.experiments.ab_leg_bifurcation_20261112.learned_prodmp import _FallbackIntegralBasis


DT = .02
FRAME_STEPS = 5
BLOCK_STEPS = 50
CINE_FRAMES = 12
MAX_BLOCKS = 3
SPEED = np.array([48., 21., 49., 40.])
ACCEL = np.array([190., 93., 195., 172.])
CHART_MM = np.array([1., 2.24, .23, .23])
PRIOR_PATCH = np.array([-153., -40., 0., 0.])
SCOUTS = np.array([[-147., -36.4, 2., .4], [-157.5, -43.15, -.7, 1.75]])
ANCHORS = np.array([[y, theta] for y in (-170., -161., -152., -143.) for theta in (-47., -42., -37., -32.)])


@lru_cache(maxsize=120)
def duration_grid(steps):
    phase = np.linspace(0., 1., steps + 1)
    return phase, *_FallbackIntegralBasis().evaluate(phase)


@dataclass(frozen=True)
class Cine:
    chart: np.ndarray
    usable: bool


@dataclass(frozen=True)
class State:
    chart: np.ndarray
    belief: Belief
    tick: int = 0
    clips: tuple = ()
    travel: float = 0.
    turn: float = 0.
    revisions: int = 0
    pos_error: float = 0.
    vel_error: float = 0.
    trace: tuple = ()
    motion_samples: tuple = ()


@dataclass(frozen=True)
class Motion:
    x: np.ndarray
    v: np.ndarray


class Executor:
    def __init__(self, body, case, replica, *, learn_ribs=True, contact=CONTACT, sensor_frames=240):
        self.body, self.case = body, case
        self.sensor = Sensors(body, case, replica, frames=sensor_frames)
        self.learn_ribs = learn_ribs
        self.contact = contact
        self.primitive = LateralPrimitive()
        phase = np.linspace(0., 1., 51)
        x = phase ** 3 * (10. - 15. * phase + 6. * phase ** 2)
        self.shape = self.primitive.fit(x, 30. * phase ** 2 * (1. - phase) ** 2)

    def sense(self, pose, tick, belief):
        view = self.body.view(pose)
        frame = self.sensor.sample(view, tick)
        return self.assimilate(view, belief, frame), frame

    def assimilate(self, view, belief, frame):
        return observe(self.body, view, belief, frame, learn_rib=self.learn_ribs)

    def cine(self, state, *, frames=CINE_FRAMES):
        b, seen, phase = state.belief, [], []
        for tick in range(state.tick + 1, state.tick + 1 + frames):
            b, packet = self.sense(state.chart, tick, b)
            seen.append(packet.seen)
            phase.append(packet.phase)
        phase, seen = np.asarray(phase), np.asarray(seen)
        usable = all(seen[np.abs((phase - p + .5) % 1. - .5) <= .16].any(axis=0).all() for p in TARGET_PHASE)
        return State(state.chart, b, state.tick + frames, (*state.clips, Cine(state.chart, usable)),
                     state.travel, state.turn, state.revisions, state.pos_error, state.vel_error, state.trace,
                     state.motion_samples)

    def raw_plan(self, q, v, target, steps, *, reference_steps=BLOCK_STEPS):
        if steps not in self.primitive.grids:
            self.primitive.grids[steps] = duration_grid(steps)
        shape = self.shape * (BLOCK_STEPS / reference_steps) ** 2
        plan = [self.primitive.condition(0, steps, x, speed, g, 0., (g - x) * shape)
                for x, speed, g in zip(q, v, target)]
        return Motion(np.column_stack([t.x for t in plan]), np.column_stack([t.v for t in plan]))

    def feasible(self, plan):
        x, v = plan.x, plan.v
        if (np.any(x < self.contact[0] - 1.e-7) or np.any(x > self.contact[1] + 1.e-7) or np.any(np.abs(v) > SPEED)
                or np.any(np.abs(np.diff(v, axis=0)) > DT * ACCEL)):
            return False
        real = self.body.surface(x)
        return np.max(np.linalg.norm(np.diff(real, axis=0), axis=1)) / DT <= 49.

    def plan(self, start, v, target, left, safe, *, reference_steps=BLOCK_STEPS):
        for portion in (1., .65, .40, .20):
            g = np.clip(safe + portion * (target - safe), *self.contact)
            ref = self.raw_plan(start, v, g, left, reference_steps=reference_steps)
            if self.feasible(ref):
                return ref
        if left == reference_steps and np.max(np.abs(v)) < 1.e-8:
            # At a hard chart boundary, a fitted basis can overshoot at
            # takeoff. An endpoint-matched quintic keeps this recovery feasible.
            t = np.linspace(0., 1., left + 1)[:, None]
            for portion in (1., .65, .40, .20):
                delta = np.clip(safe + portion * (target - safe), *self.contact) - start
                x = start + (t ** 3 * (10. - 15. * t + 6. * t ** 2)) * delta
                v_ref = (30. * t ** 2 * (1. - t) ** 2) * delta / (left * DT)
                ref = Motion(x, v_ref)
                if self.feasible(ref):
                    return ref
        return None

    def inverse_angle(self, q, b):
        level = self.body.view([q[0], q[1], 0., 0.])
        centers, axes, _ = self.body.trajectories(b.mean, TARGET_PHASE)
        center = centers[0].mean(axis=0)
        # Through-contact plane from both atrioventricular branches and LV apex.
        long_axis = self.body.rotation[:, :, 1]
        tips = center[2:] - .89 * axes[0, 0, 2:, 1, None] * long_axis[2:]
        targets = np.vstack((center, tips)) - level.origin
        weights = np.array([1., 1., 1., 1., .62, 1.1])
        system = (targets.T * weights) @ targets
        _, vectors = np.linalg.eigh(system)
        normal = vectors[:, 0]
        if normal @ level.normal < 0.:
            normal = -normal
        rho = np.arctan2(-normal @ level.right, normal @ level.normal)
        after_roll = -np.sin(rho) * level.right + np.cos(rho) * level.normal
        rock = np.arctan2(-normal @ level.forward, normal @ after_roll)
        return np.clip([q[0], q[1], np.rad2deg(rho), np.rad2deg(rock)], *self.contact)

    def margin(self, chart, belief):
        view = self.body.view(chart, belief.bone_mean())
        gm = self.body.goal_metrics(view, belief.mean)
        return float(likelihood_margin(gm)[0])

    def refine(self, original, b, axis, *, extent=1.):
        q = original.copy()
        direct = self.inverse_angle(q, b)
        q[2:] = .35 * q[2:] + .65 * direct[2:]
        # A quadratic local line search over one real-valued coordinate.
        # The three evaluations estimate slope/curvature, not allowed commands.
        step = extent * [1.3, .70, .75, .65][axis]
        shift = np.zeros(4)
        shift[axis] = step
        left, right = np.clip(q - shift, *self.contact), np.clip(q + shift, *self.contact)
        f0, fm, fp = self.margin(q, b), self.margin(left, b), self.margin(right, b)
        curvature = fm + fp - 2. * f0
        fraction = np.clip(.5 * (fm - fp) / curvature, -1., 1.) if curvature < -1.e-5 else np.clip((fp - fm) * 3., -1., 1.)
        result = np.clip(q + fraction * shift, *self.contact)
        choices = (original, q, left, right, result)
        return max(choices, key=lambda p: self.margin(p, b) - .011 * np.linalg.norm((p - original) * CHART_MM))

    def move(self, state, goal, *, streamed, details=False, frames=10, revise=None):
        steps = frames * FRAME_STEPS
        revise = self.refine if revise is None else revise
        q, b = state.chart, state.belief
        zero = np.zeros(4)
        path = self.plan(q, zero, goal, steps, q, reference_steps=steps)
        if path is None:
            path = self.raw_plan(q, zero, q, steps, reference_steps=steps)
        start = 0
        trace = list(state.trace)
        motion_samples = list(state.motion_samples)
        covered, angles = state.travel, state.turn
        revisions, pe, ve = state.revisions, state.pos_error, state.vel_error
        before = self.body.view(q)
        tick = state.tick
        for done in range(FRAME_STEPS, steps + 1, FRAME_STEPS):
            i = done - start
            q, velocity = path.x[i], path.v[i]
            if details:
                for step, sample in enumerate(path.x[i - FRAME_STEPS + 1:i + 1], 1):
                    motion_samples.append([tick / 10. + step * DT, *sample])
            world = self.body.surface(path.x[i - FRAME_STEPS:i + 1])
            covered += np.linalg.norm(np.diff(world, axis=0), axis=1).sum()
            view = self.body.view(q)
            rotation = before.forward @ view.forward + before.right @ view.right + before.normal @ view.normal
            angles += np.rad2deg(np.arccos(np.clip((rotation - 1.) * .5, -1., 1.)))
            before = view
            tick += 1
            frame = self.sensor.sample(view, tick)
            b = self.assimilate(view, b, frame)
            updated = False
            remaining = steps - done
            if streamed and remaining >= 15 and (frame.seen.any() or np.isfinite(frame.echo).any()):
                old_goal = path.x[-1]
                proposed = revise(old_goal, b, (done // FRAME_STEPS - 1) % 4, extent=remaining / 50. + .25)
                if np.linalg.norm((proposed - old_goal) * CHART_MM) > .025:
                    suffix = self.plan(q, velocity, proposed, remaining, old_goal, reference_steps=steps)
                    if suffix is not None and np.linalg.norm((suffix.x[-1] - old_goal) * CHART_MM) > .02:
                        pe = max(pe, float(np.max(np.abs(suffix.x[0] - q))))
                        ve = max(ve, float(np.max(np.abs(suffix.v[0] - velocity))))
                        path, start = suffix, done
                        revisions += 1
                        updated = True
            if details:
                trace.append([tick / 10., *q.tolist(), *velocity.tolist(), *path.x[-1].tolist(), int(updated), covered])
        return State(q, b, tick, state.clips, float(covered), float(angles), revisions, pe, ve, tuple(trace),
                     tuple(motion_samples))

    def warmup(self, *, details=False):
        state = self.cine(State(self.case.start, Belief.prior()))
        for goal in SCOUTS:
            state = self.cine(self.move(state, goal, streamed=False, details=details))
        return state

    def goals(self, state, *, anchors=None):
        q, b = state.chart, state.belief
        if anchors is None:
            anchors = np.vstack((ANCHORS, q[:2], PRIOR_PATCH[:2] + b.mean[[1, 0]] * [1., .30]))
        # Each seed becomes an inverse-geometry proposal with a continuously
        # valued orientation, and the top proposals are refined in contact coordinates.
        options = np.stack([self.inverse_angle(np.r_[xy, q[2:]], b) for xy in anchors])
        distance = np.linalg.norm((options - q) * CHART_MM, axis=1)
        scores = np.array([self.margin(goal, b) for goal in options])
        choose = scores - .011 * distance
        ranked = np.argsort(choose)[::-1]
        first = int(ranked[0])
        different = np.linalg.norm((options - options[first]) * CHART_MM, axis=1)
        viable = ranked[(different[ranked] > 5.) & (scores[ranked] > scores[first] - 1.4)]
        second = int(viable[0]) if len(viable) else int(ranked[1])

        # Value of another view is estimated from predicted variation in a set
        # of observable quantities under the current heart posterior.
        points = b.points()
        informativeness = []
        select = ranked[:7]
        for k in select:
            view = self.body.view(options[k], b.bone_mean())
            predicted = self.body.moments(view, points, TARGET_PHASE)[0]
            area = predicted[..., 0]
            p = np.log1p(area / 75.)
            uncertainty = np.mean(np.log1p(p.var(axis=0) / .18 ** 2) * np.minimum(area.mean(axis=0) / 450., 1.))
            informativeness.append(float(.61 * choose[k] + .31 * uncertainty + .013 * min(different[k], 18.)))
        third = int(select[np.argmax(informativeness)])
        if third in (first, second):
            third = next((int(k) for k in select if k not in (first, second)), third)
        goals = []
        for k in (first, second, third):
            candidate = self.refine(options[k], b, 0)
            candidate = self.refine(candidate, b, 1)
            goals.append(candidate)
        return np.array(goals)

    def confirmation(self, state):
        b = state.belief
        points = b.points()
        best, value = None, 0.
        best_rank = -np.inf
        for clip in state.clips:
            if not clip.usable:
                continue
            view = self.body.view(clip.chart, b.bone_mean())
            gm = self.body.goal_metrics(view, points)
            probability = float(passes(gm).mean())
            rank = probability + .08 * likelihood_margin(gm).mean()
            if rank > best_rank:
                best, value, best_rank = clip, probability, float(rank)
        return (best if value >= .58 else None), value

    def mpc(self, state, steps, streamed, *, details=False):
        for _ in range(steps):
            done, confidence = self.confirmation(state)
            if done is not None and confidence >= .92:
                break
            goal = self.goals(state)[0]
            state = self.cine(self.move(state, goal, streamed=streamed, details=details))
        return state

    def outcome(self, state):
        clip, confidence = self.confirmation(state)
        gt = [bool(passes(self.body.goal_metrics(self.body.view(c.chart, self.case.rib), self.case.cardiac))[0]) for c in state.clips]
        success = clip is not None and gt[next(i for i, c in enumerate(state.clips) if c is clip)]
        valid = any(c.usable and v for c, v in zip(state.clips, gt))
        return {"success": bool(success), "wrong": bool(clip is not None and not success),
                "visited_valid_view": any(gt), "recorded_valid_cine": valid, "confidence": confidence,
                "travel_mm": state.travel, "turn_deg": state.turn, "time_s": state.tick / 10.,
                "revisions": state.revisions, "join_position_error": state.pos_error, "join_velocity_error": state.vel_error,
                "accepted_chart": None if clip is None else clip.chart.tolist()}

    def features(self, state, goals, steps):
        b = state.belief
        now = self.body.view(state.chart)
        base = np.r_[now.address, state.chart[2:] / 10., b.mean / [9., 6., 7., .060, .23, .24],
                     np.sqrt(np.diag(b.covariance)), b.bone_mean() / 6., b.bone_std() / 6., steps]
        data = []
        for i, goal in enumerate(goals):
            view = self.body.view(goal, b.bone_mean())
            gm = self.body.goal_metrics(view, b.points())
            sector = [np.mean(x < 85.) for x in np.array_split(view.bone_distance, 3)]
            data.append(np.r_[base, view.address, goal[2:] / 10., (goal - state.chart) * CHART_MM / 15.,
                              gm.mean(axis=0), passes(gm).mean(), sector, np.eye(3)[i]])
        return np.array(data)
