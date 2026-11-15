"""Streaming ProDMP execution, public state and shared MPC for the recovery study."""

from dataclasses import dataclass, field, replace
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent / "ab_streaming_prodmp_two_vessel_20261113"
sys.path[:0] = [str(HERE.parents[2]), str(LEGACY)]
from primitive import (  # noqa: E402
    BLOCK_STEPS, FRAME_STRIDE, LateralPrimitive, SAMPLE_DT_S, admissible,
)

FRAME_COUNT = 81
FRAMES_PER_BLOCK = BLOCK_STEPS // FRAME_STRIDE
START = 2.0
END = 31.4
EDGES = np.array([START, 7., 12., 17., 22., 27., END])
BINS = len(EDGES) - 1
CENTER_MM = 3.
IDENTITY_ACCEPT = .95
IDENTITY_STATION = 28.0
NAMES = ("advance", "wait", "left", "right", "short", "revisit")
KINDS = {"advance": "advance", "wait": "wait", "left": "side", "right": "side",
         "short": "short", "revisit": "revisit"}
MAX_TRAVEL = 67.
PHASE = np.linspace(0., 1., BLOCK_STEPS + 1)
LOOK_SAMPLES = (BLOCK_STEPS // 5, 3 * BLOCK_STEPS // 5, BLOCK_STEPS)
PROGRESS_SPEED = 7.8


class ChartGeometry:
    """Reference-pose contact map. Costs are sampled Cartesian commands."""

    def __init__(self, sections):
        self.stations = np.array(sections["s_mm"])
        self.origins = np.array(sections["origin_atlas_mm"])
        self.rotations = Slerp(self.stations, Rotation.from_matrix(sections["axes_atlas"]))

    def poses(self, s, x):
        s = np.clip(s, self.stations[0], self.stations[-1])
        origins = np.column_stack([np.interp(s, self.stations, col) for col in self.origins.T])
        rot = self.rotations(s).as_matrix()
        return origins + np.asarray(x)[:, None] * rot[:, :, 1], rot

    def cost(self, s, x):
        xyz, rot = self.poses(s, x)
        distance = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
        relative = np.swapaxes(rot[:-1], -2, -1) @ rot[1:]
        turns = float(np.rad2deg(Rotation.from_matrix(relative).magnitude()).sum())
        return distance, turns


def station_bin(s):
    if s < START - 1e-8 or s > END + 1e-8:
        return -1
    return min(int(np.searchsorted(EDGES, s, side="right") - 1), BINS - 1)


@dataclass
class State:
    belief: np.ndarray
    s: float
    x: float
    vs: float = 0.
    vx: float = 0.
    angle: float = 15.
    frame: int = 0
    travel: float = 0.
    covered: int = 0
    missed: int = 0
    ambiguous: int = 0
    seen: int = 0
    last_trusted: float | None = None
    slot: int = -1
    residual: float = 0.
    query_identity: float = .5

    def uncovered(self):
        return [j for j in range(BINS) if not (self.covered & (1 << j))]

    def query_bin(self):
        missing = self.uncovered()
        if not missing:
            return BINS - 1
        current = station_bin(self.s)
        past = [j for j in missing if j <= current]
        return past[0] if past else missing[0]

    def trigger(self):
        if self.frame < FRAMES_PER_BLOCK or self.frame > FRAME_COUNT - 1 - 2 * FRAMES_PER_BLOCK:
            return False
        current = station_bin(self.s)
        pending = self.uncovered()
        if not pending:
            return False
        early_loss = pending[0] < current or (pending[0] == current and self.missed >= 4)
        query_uncertain = current <= pending[0] <= current + 1 and .12 < self.query_identity < .88
        return early_loss or query_uncertain


@dataclass(frozen=True)
class Candidate:
    name: str
    s: np.ndarray
    vs: np.ndarray
    lateral: object
    distance: float
    turn: float
    goal_dx: float


@dataclass
class Record:
    packets: list = field(default_factory=list)
    motion: list = field(default_factory=list)
    decisions: list = field(default_factory=list)
    accepts: list = field(default_factory=list)
    blocks: list = field(default_factory=list)
    handoff: float = 0.
    revisions: int = 0

    def copy(self):
        return Record(self.packets.copy(), self.motion.copy(), self.decisions.copy(),
                      self.accepts.copy(), self.blocks.copy(), self.handoff, self.revisions)

    def details(self, state):
        wrong_certification = any(not entry["identity_ok"] for entry in self.accepts)
        mistaken = [row for row in self.packets if row["accepted_slot"] >= 0
                    and row["roles_truth"][row["accepted_slot"]] != 0]
        wrong = any(abs(row["y"][row["accepted_slot"]][0]) <= CENTER_MM for row in mistaken)
        verified = {entry["bin"] for entry in self.accepts if entry["identity_ok"] and entry["center_ok"]}
        complete = state.s >= END - .28 and len(verified) == BINS and not wrong
        return {
            "complete": complete,
            "wrong": wrong,
            "wrong_certification": wrong_certification,
            "wrong_confident_frames": len(mistaken),
            "coverage": len(verified),
            "public_coverage": int(state.covered).bit_count(),
            "arrived": state.s >= END - .28,
            "frames": state.frame + 1,
            "seconds": state.frame * FRAME_STRIDE * SAMPLE_DT_S,
            "mpc_revisions": self.revisions,
            "max_handoff_error": self.handoff,
        }


def longitudinal(s, vs, goal, goal_v):
    # One-second quintic with sampled position/speed/acceleration gates.
    d = goal - s - vs
    end_delta = goal_v - vs
    a, b, c = 10. * d - 4. * end_delta, -15. * d + 7. * end_delta, 6. * d - 3. * end_delta
    x = s + PHASE * (vs + PHASE**2 * (a + PHASE * (b + PHASE * c)))
    v = vs + PHASE**2 * (3. * a + PHASE * (4. * b + 5. * c * PHASE))
    return x, v


def skill_features(x, vx, goal):
    return np.array([1., (goal - x) / 5., vx / 6.])


class Controller:
    def __init__(self, world, sections, coefficients=None):
        self.world = world
        self.geometry = ChartGeometry(sections)
        self.primitive = LateralPrimitive()
        self.coefficients = coefficients or {}
        self.zero = np.zeros(self.primitive.null.shape[0])

    def shape(self, name, x, vx, goal):
        mapping = self.coefficients.get(KINDS[name])
        return self.zero if mapping is None else skill_features(x, vx, goal) @ mapping

    def mean(self, state, s, alias):
        return state.belief @ self.world.target_grid(s, state.angle, alias=alias)

    def candidate(self, state, name, goal_s, goal_v, goal_x, alias):
        s, vs = longitudinal(state.s, state.vs, goal_s, goal_v)
        if s.min() < START - 1e-7 or s.max() > END + 1e-7 or np.max(abs(vs)) > 16.:
            return None
        if np.max(abs(np.diff(vs))) / SAMPLE_DT_S > 75.:
            return None
        goal_x = float(np.clip(goal_x, -30., 0.))
        center = self.mean(state, goal_s, alias)[0]
        dx = float(np.clip(goal_x - state.x, -8.5, 8.5))
        for factor in (1., .5, .25, 0.):
            goal = state.x + factor * dx
            shape = self.shape(name, state.x, state.vx, goal)
            for weights in (shape, self.zero):
                plan = self.primitive.condition(0, BLOCK_STEPS, state.x, state.vx, goal, 0., weights)
                if admissible(plan):
                    travel, turn = self.geometry.cost(s, plan.x)
                    if state.travel + travel <= MAX_TRAVEL:
                        return Candidate(name, s, vs, plan, travel, turn, float(goal - center))
        return None

    def forecast(self, state, candidate, alias):
        q = state.query_bin()
        remaining = state.uncovered()
        values = {j: np.zeros_like(state.belief) for j in remaining}
        near = np.zeros_like(state.belief)
        for i in LOOK_SAMPLES:
            means, vis = self.world.predict(candidate.s[i], candidate.lateral.x[i], state.angle, alias=alias)
            center = np.exp(-.5 * (means[:, 0, 0] / 2.1)**2)
            probability = vis[:, 0] * center
            j = station_bin(candidate.s[i])
            if j in values:
                values[j] = np.maximum(values[j], probability)
            near = np.maximum(near, probability)
        seen_q = float(state.belief @ values.get(q, np.zeros_like(near)))
        total = sum(float(state.belief @ value) for value in values.values())
        end_s = candidate.s[-1]
        expected_hit = float(state.belief @ near)
        return np.array([seen_q, total, expected_hit, end_s - state.s, candidate.distance, candidate.turn])

    def options(self, state, alias, *, full=True, perturb=0.):
        go = min(state.s + PROGRESS_SPEED, END)
        vel = PROGRESS_SPEED if go < END - PROGRESS_SPEED * .5 else 0.
        center = self.mean(state, go, alias)[0]
        base = []
        for offset in (0., -3.8, 3.8):
            candidate = self.candidate(state, "advance", go, vel, center + offset + perturb, alias)
            if candidate is not None:
                base.append(candidate)
        head = {}
        if base:
            head["advance"] = max(base, key=lambda c: self.adv_score(state, c, alias))
        if not full and head:
            return head
        stop = float(np.clip(state.s + .26 * state.vs, START, END))
        mid_center = self.mean(state, stop, alias)[0]
        left = min(mid_center - 4.5, state.x - 3.8)
        right = max(mid_center + 4.5, state.x + 3.8)
        q = state.query_bin()
        q_s = float(np.mean(EDGES[q:q + 2]))
        last = state.last_trusted if state.last_trusted is not None and state.last_trusted < state.s - 1.0 else q_s
        target_s = q_s if q_s < state.s - 1. else last
        target_s = float(np.clip(np.clip(target_s, state.s - 6., state.s + 3.), START, END))
        short_s = float(np.clip(max(stop, state.s + 3.7), START, END))
        for name, s, x in (("wait", stop, state.x), ("left", stop, left), ("right", stop, right),
                           ("short", short_s, self.mean(state, short_s, alias)[0]),
                           ("revisit", target_s, self.mean(state, target_s, alias)[0])):
            c = self.candidate(state, name, s, 0., x, alias)
            if c is not None and not any(
                np.max(abs(c.s - other.s)) < .02
                and np.max(abs(c.lateral.x - other.lateral.x)) < .02
                for other in head.values()
            ):
                head[name] = c
        return head

    def adv_score(self, state, candidate, alias):
        f = self.forecast(state, candidate, alias)
        end_s, end_x = candidate.s[-1], candidate.lateral.goal
        tail_s = min(end_s + PROGRESS_SPEED, END)
        tail_target = self.mean(state, tail_s, alias)[0]
        tail_x = end_x + np.clip(tail_target - end_x, -8.5, 8.5)
        means, vis = self.world.predict(tail_s, tail_x, state.angle, alias=alias)
        tail = state.belief @ (vis[:, 0] * np.exp(-.5 * (means[:, 0, 0] / 2.1)**2))
        return float(.65 * f[1] + .65 * f[2] + .45 * tail - .018 * f[4])

    def info(self, state, candidate, alias):
        """Common-seed two-frame conditional preview using only public hypotheses."""
        query_s = IDENTITY_STATION
        prior_role = self.world.identity_prob(state.belief, query_s, state.angle, alias=alias)
        prior_conf = max(prior_role, 1. - prior_role)
        if prior_conf >= .995:
            return 0.
        terminal = []
        for sim in range(4):
            rng = np.random.default_rng([9426, state.frame, sim])
            posterior = state.belief
            for sample in (3 * BLOCK_STEPS // 5, BLOCK_STEPS):
                s, x = candidate.s[sample], candidate.lateral.x[sample]
                y = self.world.sample(posterior, s, x, state.angle, rng, alias=alias)
                posterior, _ = self.world.update(posterior, y, s, x, state.angle, alias=alias)
            prob = self.world.identity_prob(posterior, query_s, state.angle, alias=alias)
            terminal.append(max(prob, 1. - prob))
        return max(0., float(np.mean(terminal)) - prior_conf)

    def mpc(self, state, choices, alias, *, active=True, informative=True):
        if not active and "advance" in choices:
            return choices["advance"]
        if not informative:
            return max(choices.values(), key=lambda c: self.greedy_recovery(state, c, alias))
        scores = {}
        for name, c in choices.items():
            scores[name] = self.greedy_recovery(state, c, alias) + 1.1 * self.info(state, c, alias)
        return choices[max(scores, key=scores.get)]

    def greedy_recovery(self, state, c, alias):
        f = self.forecast(state, c, alias)
        urgency = float(state.s < END - 1. and state.frame >= 50)
        return float(1.35 * f[0] + .36 * f[1] + .30 * f[2] + (.018 + .025 * urgency) * f[3] - .016 * f[4])

    def observe(self, scene, state, record, alias):
        packet, roles, truth = self.world.observe(scene, state.frame, state.s, state.x)
        state.angle = float(scene.angles_deg[state.frame])
        state.belief, assoc = self.world.update(
            state.belief, packet, state.s, state.x, state.angle, alias=alias)
        slot = int(np.argmax(assoc)) if len(assoc) and float(np.max(assoc)) >= IDENTITY_ACCEPT else -1
        state.slot = slot
        state.seen = len(packet)
        state.residual = float(packet[slot, 0]) if slot >= 0 else 0.
        state.missed = 0 if slot >= 0 else state.missed + 1
        state.ambiguous = (state.ambiguous + 1) if slot < 0 and len(packet) > 0 else 0
        j = station_bin(state.s)
        if slot >= 0 and abs(state.residual) <= CENTER_MM and j >= 0:
            state.last_trusted = state.s
            if not (state.covered & (1 << j)):
                state.covered |= 1 << j
                record.accepts.append({
                    "frame": state.frame, "bin": j, "slot": slot,
                    "identity_ok": bool(roles[slot] == 0),
                    "center_ok": bool(abs(truth[0] - state.x) <= CENTER_MM),
                })
        query = state.query_bin()
        point = float(np.mean(EDGES[query:query + 2]))
        state.query_identity = self.world.identity_prob(state.belief, point, state.angle, alias=alias)
        record.packets.append({
            "frame": state.frame, "s": state.s, "x": state.x, "q": state.angle,
            "y": packet.tolist(), "association": np.asarray(assoc).tolist(),
            "accepted_slot": slot, "roles_truth": roles.tolist(), "target_truth": truth.tolist(),
        })

    def initial(self, scene, alias):
        x = float(np.clip(self.world.nominal(START)[0], -30., 0.))
        state = State(self.world.prior.copy(), START, x)
        record = Record(motion=[[0., START, x, 0., 0., float(scene.angles_deg[0])]])
        self.observe(scene, state, record, alias)
        return state, record

    def execute(self, scene, state, record, candidate, alias):
        """Install a feasible 1 s primitive; revise the suffix after each new packet."""
        start_frame = state.frame
        start_s, start_v = state.s, state.vs
        plan = candidate.lateral
        handoff = max(abs(plan.x[0] - state.x), abs(plan.v[0] - state.vx),
                      abs(candidate.s[0] - start_s), abs(candidate.vs[0] - start_v))
        record.handoff = max(record.handoff, float(handoff))
        from_row = len(record.motion) - 1
        record.decisions.append({"frame": start_frame, "action": candidate.name, "planned_s": float(candidate.s[-1]),
                                 "planned_x": plan.goal, "planned_travel": candidate.distance})
        for tick in range(FRAME_STRIDE, BLOCK_STEPS + 1, FRAME_STRIDE):
            end_frame = state.frame + 1
            angle_begin, angle_end = scene.angles_deg[end_frame - 1:end_frame + 1]
            for i in range(tick - FRAME_STRIDE + 1, tick + 1):
                angle = angle_begin + (angle_end - angle_begin) * (i - tick + FRAME_STRIDE) / FRAME_STRIDE
                record.motion.append([start_frame * FRAME_STRIDE * SAMPLE_DT_S + i * SAMPLE_DT_S,
                                      float(candidate.s[i]), float(plan.x[i - plan.step]),
                                      float(candidate.vs[i]), float(plan.v[i - plan.step]), float(angle)])
            state.s, state.x, state.vs, state.vx = record.motion[-1][1:5]
            state.frame = end_frame
            self.observe(scene, state, record, alias)
            if tick >= BLOCK_STEPS or state.slot < 0 or candidate.name == "wait":
                continue
            goal = self.mean(state, candidate.s[-1], alias)[0]
            goal += candidate.goal_dx + .22 * state.residual
            if KINDS[candidate.name] == "side":
                # Visual correction stays in the selected acoustic window.
                goal = float(np.clip(goal, candidate.lateral.goal - 1.4, candidate.lateral.goal + 1.4))
            goal = float(np.clip(goal, -30., 0.))
            for factor in (.45, .18):
                new_goal = plan.goal + factor * np.clip(goal - plan.goal, -3.0, 3.0)
                shape = self.shape(candidate.name, state.x, state.vx, new_goal)
                if abs(new_goal - plan.goal) <= .025:
                    break
                suffix = self.primitive.condition(tick, BLOCK_STEPS - tick, state.x, state.vx,
                                                  new_goal, 0., shape)
                if not admissible(suffix):
                    continue
                past = np.array(record.motion[from_row:])
                preview = np.r_[past[:, 2], suffix.x[1:]]
                if state.travel + self.geometry.cost(candidate.s, preview)[0] > MAX_TRAVEL:
                    continue
                record.handoff = max(record.handoff, float(max(abs(suffix.x[0] - state.x),
                                                              abs(suffix.v[0] - state.vx))))
                record.revisions += 1
                plan = suffix
                break
        block = np.array(record.motion[from_row:])
        state.travel += self.geometry.cost(block[:, 1], block[:, 2])[0]
        record.blocks.append((candidate.name, from_row, len(record.motion)))

    def finish(self, scene, state, record, alias, *, allow_recover, informative=False, stop_at_trigger=False,
               rng=None):
        """Roll the same feedback-enabled MPC. Stop only at legal decision times."""
        while state.frame + FRAMES_PER_BLOCK < FRAME_COUNT:
            if state.s >= END - .065 and (not state.uncovered() or not allow_recover):
                break
            active = allow_recover and (state.trigger() or state.s >= END - .30)
            if stop_at_trigger and state.trigger():
                break
            perturb = float(rng.uniform(-3.5, 3.5)) if rng is not None and rng.random() < .35 else 0.
            choices = self.options(state, alias, full=active, perturb=perturb)
            if not choices:
                break
            candidate = self.mpc(state, choices, alias, active=active, informative=informative)
            self.execute(scene, state, record, candidate, alias)
        return state, record

    def uncertainty(self, state, q, alias, *, angle=None, intervals=False):
        s = float(np.mean(EDGES[q:q + 2]))
        angle = state.angle if angle is None else angle
        target = self.world.target_grid(s, angle, alias=alias)
        p = state.belief
        mean = p @ target
        centered = target - mean
        covariance = centered.T @ (p[:, None] * centered)
        var = np.diag(covariance)
        side = float(self.world.identity_prob(p, s, angle, alias=alias))
        out = {"mean": mean.tolist(), "var": var.tolist(), "cov": covariance.tolist(), "side": side}
        if intervals:
            bounds = []
            for axis in target.T:
                order = np.argsort(axis, kind="stable")
                where = np.searchsorted(np.cumsum(p[order]), [.05, .95])
                bounds.append(axis[order[np.minimum(where, len(p) - 1)]].tolist())
            out["interval90"] = bounds
        return out

    def policy_features(self, state, candidate, alias):
        s = state.s
        query = state.query_bin()
        u = self.uncertainty(state, query, alias)
        f = self.forecast(state, candidate, alias)
        code = [float(candidate.name == name) for name in NAMES]
        return np.r_[s / 10., state.x / 10., state.vs / 10., state.frame / 50.,
                     state.missed / 10., state.ambiguous / 10., state.seen / 3., state.covered.bit_count() / BINS,
                     (state.last_trusted if state.last_trusted is not None else s - 10.) - s,
                     (u["mean"][0] - state.x) / 5., u["mean"][1] / 20.,
                     np.sqrt(u["var"][0]), np.sqrt(u["var"][1]), u["side"],
                     self.world.identity_prob(state.belief, IDENTITY_STATION, state.angle, alias=alias),
                     (state.angle - 15.) / 10., query / BINS, f / [1., 3., 1., 8., 10., 4.],
                     (candidate.lateral.goal - state.x) / 8.,
                     code]


def fork(state, record):
    return replace(state, belief=state.belief.copy()), record.copy()
