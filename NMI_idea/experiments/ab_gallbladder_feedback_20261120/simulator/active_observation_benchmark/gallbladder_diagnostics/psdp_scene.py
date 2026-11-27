"""Registered anatomy, public belief and image observations for the gallbladder replay."""

from dataclasses import dataclass, replace

import numpy as np
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from scipy.stats import qmc
from shapely.geometry import Point

from liver_guided_collection import LiverPrior, Noise, SagittalBody, SWEEPS, costal_envelope, sweep_motion
from gallbladder_belief import ORGAN_LIMIT
from gallbladder_geometry import BEAM_ANGLE, DEPTH_MM
from gallbladder_mpc import FRAME_DT, Motor
from gallbladder_prescreen import generate, initialize
from servo import FRAME_STEPS


REGIONS = ("neck", "body", "fundus")
HORIZON = 260
TAPE_FRAMES = 640


@dataclass(frozen=True)
class Packet:
    tick: int
    q: np.ndarray
    reading: np.ndarray
    marks: np.ndarray
    qualified: bool
    genuine: bool


@dataclass(frozen=True)
class State:
    q: np.ndarray
    v: np.ndarray
    tick: int
    travel: float
    turn: float
    liver: np.ndarray
    evidence: np.ndarray
    recent: tuple = ()
    captures: tuple = ()
    acquired: int | None = None


class Tape:
    def __init__(self, seed, condition, stream, *, archived=False):
        self.noise = Noise(seed, stream, condition, 300 if archived else TAPE_FRAMES)
        rng = np.random.default_rng([seed, stream, 6431])
        self.uniform = rng.uniform(size=(TAPE_FRAMES, 3))
        scale = 2.8 if condition == "wide" else 4.6
        self.jitter = scale * rng.normal(size=(TAPE_FRAMES, 3, 2))
        self.decoys = rng.normal(0., 11., size=(TAPE_FRAMES, 3, 2))


def largest(polygon):
    if polygon is None or polygon.is_empty:
        return None
    if hasattr(polygon, "geoms"):
        return max(polygon.geoms, key=lambda p: p.area)
    return polygon


class Scene:
    def __init__(self, seed, condition, *, geometry=None):
        self.case, original, self.truth = generate(seed) if geometry is None else geometry
        self.condition = condition
        self.body = SagittalBody(original.reference, self.case.beta)
        self.motor = Motor(self.body, self.case)
        self.envelope = costal_envelope(self.body)
        self.nominal = self.body.anatomy(np.zeros(7))
        self.liver_prior = LiverPrior(self.body, self.nominal)
        self.start = self.case.start.copy()
        self.start[0] = self.envelope(self.start[1]) - 32.
        self.start[2:] *= [.07, .30]
        self.fan = self.body._fan_polygon()
        modes = 2. * qmc.Sobol(7, scramble=True, seed=6102).random_base2(6) - 1.
        self.parameters = modes * ORGAN_LIMIT
        organ = [self.body.anatomy(p) for p in self.parameters]
        self.centers = np.stack([x.center for x in organ])
        self.axes = np.stack([x.axis for x in organ])
        self.landmarks = np.asarray([[x.landmarks[name] for name in REGIONS] for x in organ])
        self.lengths = np.linalg.norm(self.landmarks[:, 0] - self.landmarks[:, -1], axis=1)
        distances = cdist(self.parameters[:, :3], self.liver_prior.shifts, metric="sqeuclidean")
        self.liver_to_gall = np.exp(-distances / (2. * 5. ** 2))
        self._truth_cache = {}
        self._pose_cache = {}

    def tape(self, stream, *, archived=False):
        return Tape(self.case.id, self.condition, stream, archived=archived)

    def at_start(self):
        return State(self.start.copy(), np.zeros(4), 0, 0., 0., self.liver_prior.initial.copy(),
                     np.zeros(len(self.centers)))

    def gall_weights(self, state):
        score = state.evidence + np.log(self.liver_to_gall @ state.liver + 1.e-9)
        return np.exp(score - logsumexp(score))

    def estimate(self, state):
        weights = self.gall_weights(state)
        axis = weights @ self.axes
        return weights @ self.centers, axis / np.linalg.norm(axis), float(weights @ self.lengths)

    def project(self, q):
        key = tuple(q)
        cached = self._pose_cache.get(key)
        if cached is not None:
            return cached
        view = self.body.view(q, np.zeros(2))
        rel = self.landmarks - view.origin
        x, depth, normal = rel @ view.right, rel @ view.forward, rel @ view.normal
        inside = ((depth >= 4.) & (np.hypot(x, depth) <= DEPTH_MM)
                  & (np.abs(x) <= depth * np.tan(BEAM_ANGLE)))
        clear = inside & ~view.beam_blocked(x, depth)
        support = clear * np.exp(-.5 * (np.maximum(np.abs(normal) - 4., 0.) / 8.5) ** 2)
        projected = np.stack((x, depth), axis=-1)
        quality = support.min(axis=-1)
        # Keep a broad, smooth range around the model threshold for planning.
        quality *= np.clip(np.linalg.norm(projected[:, 0] - projected[:, 2], axis=1)
                           / np.maximum(self.lengths, 1.), 0., 1.)
        result = (projected, support, quality)
        if len(self._pose_cache) > 12000:
            self._pose_cache.clear()
        self._pose_cache[key] = result
        return result

    def truth_packet(self, q):
        key = tuple(q)
        cached = self._truth_cache.get(key)
        if cached is not None:
            return cached
        view = self.body.view(q, self.case.rib)
        section = self.body.measure(view, self.truth)
        points = np.asarray([self.truth.landmarks[name] for name in REGIONS]) - view.origin
        xy = np.column_stack((points @ view.right, points @ view.forward))
        n = points @ view.normal
        x, y = xy.T
        shape = self.body._section(self.truth.gallbladder, self.body.gallbladder_faces, view)
        if shape is not None:
            shape = shape.intersection(self.fan)
            shadow = self.body._shadow_polygon(view)
            if shadow is not None:
                shape = shape.difference(shadow)
        lumen = largest(shape)
        visibility = np.zeros(3)
        if lumen is not None:
            distance = np.array([lumen.distance(Point(p)) for p in xy])
            in_image = ((y >= 4.) & (np.hypot(x, y) <= DEPTH_MM)
                        & (np.abs(x) <= y * np.tan(BEAM_ANGLE)) & ~view.beam_blocked(x, y))
            plane = np.exp(-.5 * (np.maximum(np.abs(n) - 5., 0.) / 7.2) ** 2)
            local = np.exp(-.5 * (np.maximum(distance - 2., 0.) / 4.5) ** 2)
            visibility = in_image * plane * local
        out = (section, xy, visibility)
        if len(self._truth_cache) > 12000:
            self._truth_cache.clear()
        self._truth_cache[key] = out
        return out

    def observe(self, q, tick, tape):
        section, projected, visibility = self.truth_packet(q)
        reading, detected = tape.noise.sample(section, tick)
        marks = np.full((3, 2), np.nan)
        if np.isfinite(reading[0, 0]):
            amplitude = .96 if self.condition == "wide" else .88
            correct = tape.uniform[tick] < amplitude * visibility if detected[0] else np.zeros(3, dtype=bool)
            fake_rate = .024 if self.condition == "wide" else .052
            false = ~correct & (tape.uniform[tick] > 1. - fake_rate)
            marks[correct] = projected[correct] + tape.jitter[tick, correct]
            marks[false] = reading[0, 1:3] + tape.decoys[tick, false]
        return Packet(tick, q.copy(), reading, marks, section.qualified, bool(detected[0]))

    def assimilate(self, state, packet, *, liver_override=None, liver_only=False):
        if liver_override is None:
            liver = self.liver_prior.update(packet.q, state.liver, packet.reading, self.condition)
        else:
            liver = liver_override
        evidence = state.evidence
        marks_seen = np.isfinite(packet.marks[:, 0])
        if not liver_only and marks_seen.any():
            coords, support, _ = self.project(packet.q)
            residual = (coords[:, marks_seen] - packet.marks[marks_seen]) / [7., 8.]
            likelihood = (.06 + .94 * support[:, marks_seen]
                          * np.exp(-.5 * (residual ** 2).sum(axis=-1)))
            evidence = .982 * evidence + .44 * np.log(likelihood).sum(axis=1)
            evidence -= evidence.max()
        elif not liver_only and np.isfinite(packet.reading[1, 0]) and not np.isfinite(packet.reading[0, 0]):
            _, support, _ = self.project(packet.q)
            evidence = .995 * evidence + .16 * np.log(1. - .55 * support.max(axis=1))
            evidence -= evidence.max()
        recent = (*state.recent, packet)[-7:]
        captures = state.captures
        if review_candidate(packet):
            captures = (*captures, packet)[-28:]
        acquired = state.acquired
        if acquired is None and packet.qualified and packet.genuine:
            acquired = packet.tick
        return replace(state, q=packet.q, tick=packet.tick, liver=liver, evidence=evidence,
                       recent=recent, captures=captures, acquired=acquired)

    def start_prefix(self, index, tape, *, trace=None):
        final, offset = SWEEPS[index]
        motion = sweep_motion(self.motor, self.envelope, self.start, final, offset)
        coords = self.body.surface(motion.x)
        state = self.at_start()
        for k in range(FRAME_STEPS, len(motion.x), FRAME_STEPS):
            delta = np.linalg.norm(np.diff(coords[k - FRAME_STEPS:k + 1], axis=0), axis=-1).sum()
            q = motion.x[k]
            turn = rotation_cost(self.body.view(state.q), self.body.view(q))
            state = replace(state, travel=state.travel + delta, turn=state.turn + turn, v=motion.v[k])
            packet = self.observe(q, state.tick + 1, tape)
            state = self.assimilate(state, packet, liver_only=True)
            if trace is not None:
                trace.append(self.trace_row(state, packet, "liver_sweep"))
        return state

    def archive_exit(self, record, tape, rotation_index):
        """Re-sense the archived scan at its executed poses with the new noisy regional readout."""
        state = self.at_start()
        sweep, turn = record["sweep_trace"], record["rotation_attempts"][rotation_index]["trace"]
        for phase, rows in enumerate((sweep, turn)):
            for row in rows:
                q = np.asarray(row["q"])
                angle = rotation_cost(self.body.view(state.q), self.body.view(q))
                state = replace(state, travel=row["travel_mm"], turn=state.turn + angle,
                                v=np.asarray(row["velocity"]))
                packet = self.observe(q, row["tick"], tape)
                old = np.asarray(row["readout"], dtype=float).reshape(2, 5)
                if not np.allclose(packet.reading, old, rtol=1.e-8, atol=1.e-6, equal_nan=True):
                    raise RuntimeError("archived arrival did not reproduce the original detector tape")
                state = self.assimilate(state, packet, liver_only=(phase == 0))
            if phase == 0:
                state = replace(state, liver=np.asarray(record["sweep_posterior"]))
                before = state
        return before, state

    def trace_row(self, state, packet, phase):
        liver, spread = self.liver_prior.moment(state.liver)
        c, _, _ = self.estimate(state)
        return [state.tick * FRAME_DT, *state.q.tolist(), *self.body.surface(state.q).tolist(), state.travel,
                bool(np.isfinite(packet.reading[0, 0])), packet.qualified, packet.genuine, 0.,
                *c.tolist(), spread, *(float(v) if np.isfinite(v) else None for v in packet.reading.ravel()),
                [float(v) if np.isfinite(v) else None for v in packet.marks.ravel()], liver.tolist(), phase]


def rotation_cost(before, after):
    relative = before.forward @ after.forward + before.right @ after.right + before.normal @ after.normal
    return float(np.rad2deg(np.arccos(np.clip((relative - 1.) / 2., -1., 1.))))


def review_candidate(packet):
    area, _, _, span, _ = packet.reading[0]
    mark = packet.marks
    if not np.isfinite(mark).all() or not area >= 75. or not span >= 24.:
        return False
    axis = mark[0] - mark[2]
    span_marks = np.linalg.norm(axis)
    if span_marks < 25. or span_marks > 95.:
        return False
    fraction = (mark[1] - mark[2]) @ axis / span_marks ** 2
    transverse = np.linalg.norm(mark[1] - mark[2] - fraction * axis)
    return bool(-.09 <= fraction <= 1.09 and transverse <= 21. and span >= .66 * span_marks)


def review(state):
    """Submit a single three-region image with support from an adjacent acquisition."""
    for full in reversed(state.captures):
        if state.tick - full.tick > 6:
            break
        for adjacent in state.recent:
            if full.tick == adjacent.tick or np.max(np.abs(full.q - adjacent.q)) > 4.5:
                continue
            present = np.isfinite(adjacent.marks[:, 0])
            if present.sum() < 2 or not adjacent.reading[0, 3] >= 18.:
                continue
            if np.max(np.linalg.norm(full.marks[present] - adjacent.marks[present], axis=1)) <= 16.:
                return full
    return None
