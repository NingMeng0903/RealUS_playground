"""Common streaming ProDMP controller for three gallbladder window requests."""

from dataclasses import dataclass
from itertools import groupby, product

import numpy as np
from scipy.spatial.distance import cdist

from gallbladder_geometry import DEPTH_MM
from gallbladder_belief import Prior
from servo import ACCEL, DT, Executor, FRAME_STEPS, SPEED


FRAME_DT = DT * FRAME_STEPS
CONTACT = np.array([[-270., -83., -90., -35.], [-132., 12., 90., 35.]])
ARMS = ("direct", "subcostal", "intercostal")
ROOT_FRAMES = 5
FRAME_LIMIT = 340


def intercostal_contacts(body):
    """Project neighboring native rib arcs to the material skin, then sample their gaps."""
    heights = np.arange(-260., -115., 2.)
    names = body.reference.bone_names
    for theta in (-80., -75., -70., -65., -60., -55.):
        theta_samples = np.full(len(heights), theta)
        origins = body.surface(np.column_stack((heights, theta_samples)))
        h = (heights - body.atlas.y0) / (body.atlas.y1 - body.atlas.y0)
        centerline = body.atlas.to_world(h, np.deg2rad(theta_samples), np.zeros(len(h))) + body.core(h)
        vectors = centerline - origins
        length = np.linalg.norm(vectors, axis=-1)
        labels = []
        for origin, ray, limit in zip(origins, vectors / length[:, None], length):
            distance, rib = body.bones.hits(origin, ray[None])
            name = names[rib[0]] if rib[0] >= 0 and distance[0] < limit else ""
            labels.append(int(name[4:-1]) if name.startswith("Rib_") and name.endswith("R") else 0)
        runs = [(index, np.array(list(group))) for index, group in groupby(range(len(labels)),
                                                                         key=labels.__getitem__)]
        for (_, below), (gap_id, gap), (_, above) in zip(runs, runs[1:], runs[2:]):
            if gap_id or not labels[below[-1]] or not labels[above[0]]:
                continue
            if abs(labels[below[-1]] - labels[above[0]]) != 1 or not 4 <= len(gap) <= 22:
                continue
            for height in np.quantile(heights[gap], [.23, .50, .77]):
                if CONTACT[0, 0] <= height <= CONTACT[1, 0]:
                    yield float(height), theta


def chart_bank(body, start, center, axis):
    """Both regional entries and ordinary midline/oblique alternatives."""
    subcostal = tuple(product(range(-263, -206, 11), (-46., -37., -28., -19.)))
    intercostal = tuple(intercostal_contacts(body))
    other = tuple(product(range(-233, -131, 13), (-76., -62., -49., -38., -26., -14., -2., 4.)))
    poses, families = [start], [0]
    for family, contacts in enumerate((other, subcostal, intercostal)):
        for xy in contacts:
            aimed = body.aim(xy, center, axis)
            for roll in (0., -12., 12.) if family else (0.,):
                chart = np.clip(aimed + [0., 0., roll, 0.], *CONTACT)
                poses.append(chart)
                families.append(family)
    return np.asarray(poses), np.asarray(families)


@dataclass(frozen=True)
class Sample:
    chart: np.ndarray
    posterior: np.ndarray
    readout: np.ndarray
    public_quality: float
    geometry_pass: bool
    true_target: bool
    tick: int
    travel: float


class Motor(Executor):
    """Reuse the existing fitted torso primitive and its continuous constraints."""

    def __init__(self, body, case):
        super().__init__(body, case, 0, contact=CONTACT, sensor_frames=3)

    def replan(self, q, velocity, target):
        delta = np.abs(target - q)
        distance = np.linalg.norm(self.body.surface(target) - self.body.surface(q))
        seconds = max(.4, 2.03 * distance / 49., max(2.03 * delta / SPEED),
                      max(np.sqrt(6.0 * delta / ACCEL)))
        frames = int(np.ceil(seconds / FRAME_DT))
        for extra in (0, max(2, frames // 4), max(4, frames // 2)):
            steps = (frames + extra) * FRAME_STEPS
            plan = self.raw_plan(q, velocity, target, steps, reference_steps=steps)
            if self.feasible(plan):
                return plan
        return None


class Planner:
    def __init__(self, body, start, case, condition):
        self.body, self.condition = body, condition
        self.prior = Prior(body)
        self.motor = Motor(body, case)
        center, axis, _, _ = self.prior.information(self.prior.initial)
        self.charts, self.family = chart_bank(body, start, center, axis)
        self.surface = body.surface(self.charts)
        self.cache = {}
        forecast = [self.forecast(chart) for chart in self.charts]
        self.detect = np.asarray([row.detect[:, 0] for row in forecast])
        self.quality = np.asarray([row.quality for row in forecast])
        self.confirm = self.detect * self.quality
        # Discounts compare two executable segments on the same contact chart.
        # They affect proposal ranking, while physical time and distances are
        # computed from every executed surface sample.
        displacement = cdist(self.surface, self.surface)
        swivel = cdist(self.charts[:, 2:], self.charts[:, 2:])
        self.transfer = np.exp(-displacement / 174. - swivel / 200.)

    def forecast(self, chart):
        key = np.asarray(chart).tobytes()
        entry = self.cache.get(key)
        if entry is None:
            view = self.body.view(chart, np.zeros(2))
            entry = self.prior.predict(view, self.condition)
            self.cache[key] = entry
        return entry

    def scores(self, q, weights, visits, *, allowed=None):
        """Two-step conditional score using long view, partial detection or miss."""
        surface = self.body.surface(q)
        travel = np.linalg.norm(self.surface - surface, axis=-1)
        swivel = np.linalg.norm(self.charts[:, 2:] - q[2:], axis=-1)
        discount = np.exp(-travel / 174. - swivel / 200.)
        repetition = np.exp(-.095 * visits)
        score = self.confirm @ weights
        detection = self.detect @ weights
        detection_var = self.detect ** 2 @ weights - detection ** 2
        immediate = (score + .10 * detection + .08 * np.maximum(detection_var, 0.)) * repetition
        rank = immediate * discount
        valid = np.arange(len(rank)) if allowed is None else np.flatnonzero(self.family == allowed)
        subset = valid[np.argsort(-rank[valid], kind="stable")[:18]]
        tails = np.argsort(-immediate, kind="stable")[:28]
        conditional = np.stack(((self.detect - self.confirm)[subset], 1. - self.detect[subset]), axis=0)
        tail_values = (conditional * weights) @ self.confirm[tails].T
        tail_values *= self.transfer[subset[:, None], tails] * repetition[tails]
        total = rank[subset] + .63 * discount[subset] * tail_values.max(axis=-1).sum(axis=0)
        order = np.argsort(-total, kind="stable")
        return subset[order], total[order], score

    def refine(self, goal, posterior):
        nominal = self.charts[goal]
        center, axis, _, _ = self.prior.information(posterior)
        aim = np.clip(self.body.aim(nominal[:2], center, axis), *CONTACT)
        aimed = self.forecast(aim)
        central = (aimed.quality * aimed.detect[:, 0]) @ posterior
        fixed = self.confirm[goal] @ posterior
        return aim if central > fixed + .019 else nominal

    def sense(self, anatomy, chart, posterior, noise, tick, travel, rib):
        actual = self.body.view(chart, rib)
        true = self.body.measure(actual, anatomy)
        readings, genuine = noise.sample(true, tick)
        forecast = self.forecast(chart)
        belief = self.prior.update(posterior, forecast, readings, self.condition)
        eps = np.isfinite(readings[0, 0])
        public_quality = float(forecast.quality @ belief) if eps else 0.
        return Sample(chart.copy(), belief, readings, public_quality, bool(true.qualified),
                      bool(genuine[0]), tick, travel)

    def prescan(self, anatomy, start, noise, rib):
        samples = []
        belief = self.prior.initial
        for tick in range(ROOT_FRAMES):
            sample = self.sense(anatomy, start, belief, noise, tick, 0., rib)
            samples.append(sample)
            belief = sample.posterior
        return samples

    def public_diagnostic(self, start, root):
        b = root[-1].posterior
        none = np.zeros(len(self.charts))
        direct, value, raw_success = self.scores(start, b, none)
        category = []
        for which in (1, 2):
            order, q, _ = self.scores(start, b, none, allowed=which)
            category.append((order[0], q[0]) if len(order) else (None, -1.))
        _, _, width, _ = self.prior.information(b)
        return {"direct_forecast": float(raw_success[direct[0]]),
                "direct_target_family": int(self.family[direct[0]]),
                "subcostal_approach_score": float(category[0][1]),
                "intercostal_approach_score": float(category[1][1]),
                "root_posterior_spread_mm": width,
                "root_any_detection": any(np.isfinite(row.readout[0, 0]) for row in root)}

    def choose_confirmation(self, ledger, recent, posterior):
        """Use the recorded image and current public belief to submit a view."""
        if len(recent) < 2 or not ledger:
            return None
        _, _, _, length = self.prior.information(posterior)
        qualified = [frame for frame in recent if np.isfinite(frame.readout[0, 0])
                     and frame.readout[0, 0] >= 165.
                     and frame.readout[0, 3] >= .61 * length]
        if len(qualified) < 2:
            return None
        def score(frame):
            predicted = self.forecast(frame.chart).quality @ posterior
            area, x, _, span, _ = frame.readout[0]
            return predicted + .18 * min(span / length, 1.1) + .06 * min(area / 530., 1.) - .0015 * abs(x)
        candidates = [frame for frame in ledger if frame.readout[0, 0] >= 165.
                      and frame.readout[0, 3] >= .61 * length]
        if not candidates:
            return None
        options = sorted(candidates, key=lambda frame: frame.public_quality, reverse=True)[:12]
        best = max(options, key=score)
        return best if score(best) >= .81 else None

    def run(self, anatomy, noise, rib, root, arm, *, frames=FRAME_LIMIT, trace=False):
        q = root[-1].chart
        posterior = root[-1].posterior
        velocity = np.zeros(4)
        family = ARMS.index(arm)
        requested_available = family == 0 or bool(np.any(self.family == family))
        if not requested_available:
            family = 0
        locked = None
        goal = None
        plan = None
        visits = np.zeros(len(self.charts))
        travel = turns = 0.
        previous_view = self.body.view(q)
        ledger = [sample for sample in root if sample.readout[0, 0] >= 150.]
        recent = list(root)
        first = next((row for row in root if row.geometry_pass and row.true_target), None)
        accepted = self.choose_confirmation(ledger, recent, posterior)
        recorded = []
        changes = 0
        release = None
        region_visits = 0
        join_pos, join_vel = 0., 0.
        projections = self.surface
        tick = root[-1].tick

        while accepted is None and tick + 1 < frames:
            tick += 1
            order, score, _ = self.scores(q, posterior, visits, allowed=family or None)
            current_ok = plan is not None and len(plan.x) > FRAME_STEPS
            # For the intervention, hold only the entry family. The chosen
            # within-family view can adjust from the live belief at each frame.
            if family and locked is None:
                locked = int(order[0])
            best = int(order[0])
            if current_ok and goal in order:
                old_score = score[np.where(order == goal)[0][0]]
                if score[0] - old_score < .034:
                    best = goal
            candidates = [best] + [int(g) for g in order[:6] if g != best]
            for selected in candidates:
                target = self.refine(selected, posterior)
                if current_ok and selected == goal and np.linalg.norm(target - plan.x[-1]) <= 1.1:
                    break
                motion = self.motor.replan(q, velocity, target)
                if motion is None:
                    continue
                join_pos = max(join_pos, float(np.max(np.abs(motion.x[0] - q))))
                join_vel = max(join_vel, float(np.max(np.abs(motion.v[0] - velocity))))
                changes += int(goal is not None and selected != goal)
                plan, goal = motion, selected
                break
            if plan is not None and len(plan.x) > FRAME_STEPS:
                motion = plan.x[:FRAME_STEPS + 1]
                world = self.body.surface(motion)
                travel += float(np.linalg.norm(np.diff(world, axis=0), axis=1).sum())
                q = plan.x[FRAME_STEPS]
                velocity = plan.v[FRAME_STEPS]
                plan = type(plan)(plan.x[FRAME_STEPS:], plan.v[FRAME_STEPS:])
            view = self.body.view(q)
            relative = (view.forward @ previous_view.forward + view.right @ previous_view.right
                        + view.normal @ previous_view.normal)
            turns += float(np.rad2deg(np.arccos(np.clip((relative - 1.) / 2., -1., 1.))))
            previous_view = view
            sample = self.sense(anatomy, q, posterior, noise, tick, travel, rib)
            posterior = sample.posterior
            if first is None and sample.geometry_pass and sample.true_target:
                first = sample
            recent.append(sample)
            recent = recent[-7:]
            if np.isfinite(sample.readout[0, 0]) and sample.readout[0, 0] >= 150.:
                ledger.append(sample)
            # Include orientation in the revisit count so a new view at the
            # same skin address remains a candidate after a partial image.
            contact_distance = np.linalg.norm(projections - view.origin, axis=1)
            nearby = ((contact_distance < 9.)
                      & (np.linalg.norm(self.charts[:, 2:] - q[2:], axis=1) < 11.))
            visits += nearby.astype(float) / 2.
            if family and goal is not None:
                near_request = contact_distance[goal] < 9.
                region_visits = region_visits + 1 if near_request else 0
                region_observed = near_request and (np.isfinite(sample.readout[0, 0]) or region_visits >= 3)
                enough_evidence = sample.public_quality >= .79 and sample.readout[0, 0] >= 210.
                if region_observed or enough_evidence:
                    family = 0
                    release = tick
            if trace:
                mean, _, smear, _ = self.prior.information(posterior)
                recorded.append([tick * FRAME_DT, *q, *view.origin, travel,
                                 bool(np.isfinite(sample.readout[0, 0])), sample.geometry_pass,
                                 sample.true_target, sample.public_quality, *mean, smear,
                                 *(float(x) if np.isfinite(x) else None for x in sample.readout.ravel())])
            accepted = self.choose_confirmation(ledger, recent, posterior)
        center, _, posterior_width, _ = self.prior.information(posterior)
        success = accepted is not None and accepted.geometry_pass and accepted.true_target
        return {"success": bool(success), "wrong": accepted is not None and not success,
                "stopped_without_confirmation": accepted is None, "acquired": first is not None,
                "travel_mm": travel, "turn_deg": turns, "time_s": tick * FRAME_DT,
                "first_quality_time_s": None if first is None else first.tick * FRAME_DT,
                "first_quality_travel_mm": None if first is None else first.travel,
                "first_quality_chart": None if first is None else first.chart.tolist(),
                "first_quality_contact_mm": None if first is None else self.body.surface(first.chart).tolist(),
                "accepted_tick": None if accepted is None else accepted.tick,
                "released_tick": release, "target_revisions": changes,
                "requested_approach_available": requested_available,
                "init_family_goal": locked, "join_position_error": join_pos, "join_velocity_error": join_vel,
                "posterior_center_mm": center.tolist(), "posterior_spread_mm": posterior_width,
                "trace": recorded}


def probe_geometry(body, planner, anatomy, rib):
    """Audit true window availability on the public contact lattice."""
    center = anatomy.center
    rows = []
    for family in (0, 1, 2):
        indices = np.flatnonzero(planner.family == family)
        valid_all, valid_clean = [], []
        for n in indices:
            chart = planner.charts[n]
            # A true-geometry check is restricted to the pose bank and kept
            # outside the planner, including during reference construction.
            view = body.view(chart)
            clean = body.measure(view, anatomy)
            if clean.qualified:
                valid_clean.append(int(n))
                if body.measure(body.view(chart, rib), anatomy).qualified:
                    valid_all.append(int(n))
        if valid_all:
            xy = planner.charts[valid_all, :2].round(4)
            distances = np.linalg.norm(planner.surface[valid_all] - center, axis=-1)
            max_clearance = float(DEPTH_MM - distances.min())
        else:
            xy, max_clearance = np.empty((0, 2)), None
        reachable = set()
        for i, coordinates in zip(valid_all, xy):
            key = tuple(coordinates)
            if key in reachable:
                continue
            if planner.motor.replan(planner.charts[0], np.zeros(4), planner.charts[i]) is not None:
                reachable.add(key)
        rows.append({"clean_qualified": len(valid_clean), "visible_qualified": len(valid_all),
                     "clear_contacts": len(np.unique(xy, axis=0)), "reachable_contacts": len(reachable),
                     "depth_clearance_mm": max_clearance})
    return rows
