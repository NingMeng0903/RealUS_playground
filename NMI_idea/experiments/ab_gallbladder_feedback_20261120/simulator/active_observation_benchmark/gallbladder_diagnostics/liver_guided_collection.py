"""Generate contact-chart sweeps and local long-axis attempts from a liver posterior."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import map_coordinates
from scipy.special import logsumexp
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gallbladder_belief import Noise, ORGAN_LIMIT, SENSOR
from gallbladder_geometry import BEAM_ANGLE, DEPTH_MM, Body
from gallbladder_mpc import CONTACT, FRAME_DT, Motor
from gallbladder_prescreen import DEFAULT, finite, generate, initialize
from model import View, READOUT
from servo import DT, FRAME_STEPS, Motion


OUT = DEFAULT.parent / "gallbladder_liver_guided"
SWEEPS = ((-24., 24.), (-39., 32.), (-54., 40.))
FRAMES = 300
GRID_STEP = 5.5
RIB_OFFSET = (15., 47.)


class SagittalBody(Body):
    """A skin-normal contact frame with cranial direction in the image."""

    def view(self, chart, rib=None):
        q = np.asarray(chart, dtype=float)
        origin, above, below, around, opposite = self.surface(
            q + np.array([[0., 0., 0., 0.], [.5, 0., 0., 0.], [-.5, 0., 0., 0.],
                          [0., .4, 0., 0.], [0., -.4, 0., 0.]]))
        right = above - below
        right /= np.linalg.norm(right)
        forward = np.cross(right, around - opposite)
        forward /= np.linalg.norm(forward)
        roll, rock = np.deg2rad(q[2:])
        right = np.cos(roll) * right + np.sin(roll) * np.cross(forward, right)
        forward = np.cos(rock) * forward + np.sin(rock) * np.cross(forward, right)
        normal = np.cross(forward, right)
        h = (q[0] - self.atlas.y0) / (self.atlas.y1 - self.atlas.y0)
        angle = np.deg2rad(q[1])
        frame = View(q.copy(), np.array([h, np.sin(angle), np.cos(angle)]), origin,
                     forward, right, normal, np.full(len(READOUT), np.inf))
        return frame if rib is None else self.with_ribs(frame, rib)


def costal_envelope(body):
    """Infer a guide from projected right rib surfaces and the medial template anchor."""
    angles = np.array([-60., -55., -50., -45., -40.])
    heights = np.arange(-246., -115., 2.)
    observed = []
    for angle in angles:
        angles_at_height = np.full(len(heights), angle)
        origins = body.surface(np.column_stack((heights, angles_at_height)))
        h = (heights - body.atlas.y0) / (body.atlas.y1 - body.atlas.y0)
        core = body.atlas.to_world(h, np.deg2rad(angles_at_height), np.zeros(len(h))) + body.core(h)
        rays = core - origins
        distance = np.linalg.norm(rays, axis=1)
        edge = None
        for y, origin, direction, length in zip(heights, origins, rays / distance[:, None], distance):
            hit, ident = body.bones.hits(origin, direction[None])
            if (hit[0] < length and body.reference.bone_names[ident[0]].startswith("Rib_")
                    and body.reference.bone_names[ident[0]].endswith("R")):
                edge = y
                break
        if edge is not None:
            observed.append((angle, edge))
    if len(observed) < 2:
        raise RuntimeError("the clinical template needs at least two lateral costal controls")
    # The source has no modeled anterior costal cartilage. This conservative
    # medial guide is a procedural anchor, interpolated to the measured ribs.
    knots = np.asarray([*observed, (0., -132.), (8., -132.)])
    knots[:, 1] = np.minimum(-125., np.maximum.accumulate(knots[:, 1]))
    return PchipInterpolator(knots[:, 0], knots[:, 1], extrapolate=False)


class LiverPrior:
    """Likelihoods of segmented liver sections under explicit translation hypotheses."""

    def __init__(self, body, nominal):
        self.body = body
        self.shifts = np.asarray(list(product(np.linspace(-1., 1., 5), repeat=3))) * ORGAN_LIMIT[:3]
        self.initial = np.full(len(self.shifts), 1. / len(self.shifts))
        mesh = trimesh.Trimesh(nominal.liver, body.liver_faces, process=False)
        voxels = mesh.voxelized(2.4).fill()
        self.grid = voxels.matrix.astype(np.float32)
        self.origin = voxels.transform[:3, 3]
        self.scale = np.diag(voxels.transform[:3, :3])
        self.lower = nominal.liver.min(axis=0) - ORGAN_LIMIT[:3] - 3.
        self.upper = nominal.liver.max(axis=0) + ORGAN_LIMIT[:3] + 3.
        x, d = np.meshgrid(np.arange(-122., 124., GRID_STEP), np.arange(6., DEPTH_MM, GRID_STEP))
        fan = (np.abs(x) <= d * np.tan(BEAM_ANGLE)) & (np.hypot(x, d) <= DEPTH_MM)
        self.x, self.depth = x[fan], d[fan]

    def update(self, chart, weights, readout, condition):
        area, x_mean, depth_mean = readout[1, :3]
        if not np.isfinite([area, x_mean, depth_mean]).all() or area < 190.:
            return weights
        frame = self.body.view(chart, np.zeros(2))
        points = frame.origin + self.x[:, None] * frame.right + self.depth[:, None] * frame.forward
        keep = (~frame.beam_blocked(self.x, self.depth) & (points >= self.lower).all(axis=1)
                & (points <= self.upper).all(axis=1))
        points = points[keep]
        if not len(points):
            return weights
        coords = (points[None] - self.shifts[:, None] - self.origin) / self.scale
        occupancy = map_coordinates(self.grid, coords.reshape(-1, 3).T, order=1, prefilter=False).reshape(
            len(self.shifts), len(points))
        count = occupancy.sum(axis=1)
        predicted_area = count * GRID_STEP ** 2
        predicted_x = occupancy @ self.x[keep] / np.maximum(count, 1.e-6)
        predicted_depth = occupancy @ self.depth[keep] / np.maximum(count, 1.e-6)
        innovation = np.column_stack((np.log1p(predicted_area / 500.) - np.log1p(area / 500.),
                                      predicted_x - x_mean, predicted_depth - depth_mean))
        feature_sigma = SENSOR[condition][-1]
        sigma = np.hypot([.16, 2.6, 3.2], feature_sigma[:3] * [1., 1.45, 1.45])
        loss = .5 * np.sum((innovation / sigma) ** 2, axis=1)
        likelihood = .98 * np.exp(-loss) + .02
        log_p = np.log(np.maximum(weights, 1.e-45)) + .52 * np.log(likelihood)
        posterior = np.exp(log_p - logsumexp(log_p))
        return .999 * posterior + .001 * self.initial

    def moment(self, weights):
        mean = weights @ self.shifts
        radius = np.sqrt(weights @ np.sum((self.shifts - mean) ** 2, axis=1))
        return mean, float(radius)


def sweep_motion(motor, envelope, start, final_angle, offset):
    start_offset = float(envelope(start[1]) - start[0])
    travel = abs(final_angle - start[1])
    frames = max(34, int(np.ceil(1.8 * travel / 21. / FRAME_DT)))
    for stretch in (1., 1.45, 1.95, 2.5, 3.2, 3.9):
        steps = int(np.ceil(stretch * frames)) * FRAME_STEPS
        scalar = motor.raw_plan(np.zeros(4), np.zeros(4), np.array([1., 0., 0., 0.]), steps, reference_steps=steps)
        progress, rate = scalar.x[:, 0], scalar.v[:, 0]
        theta = start[1] + progress * (final_angle - start[1])
        y = envelope(theta) - start_offset - progress * (offset - start_offset)
        q = np.column_stack((y, theta, start[2] * (1. - progress), start[3] * (1. - progress)))
        dq = np.column_stack(((envelope.derivative()(theta) * (final_angle - start[1]) - (offset - start_offset)) * rate,
                              rate * (final_angle - start[1]), -rate * start[2], -rate * start[3]))
        motion = Motion(q, dq)
        if motor.feasible(motion) and np.max(np.abs(q[0] - start)) < 1.e-8:
            return motion
    raise RuntimeError("costal guide did not admit a feasible continuous movement")


def acquire(body, anatomy, rib, motion, noise, prior, weights, tick, travel):
    data, partition = [], body.surface(motion.x)
    cost = np.r_[0., np.cumsum(np.linalg.norm(np.diff(partition, axis=0), axis=1))]
    for i in range(FRAME_STEPS, len(motion.x), FRAME_STEPS):
        tick += 1
        q = motion.x[i]
        truth = body.measure(body.view(q, rib), anatomy)
        readout, genuine = noise.sample(truth, tick)
        weights = prior.update(q, weights, readout, noise.condition)
        liver, spread = prior.moment(weights)
        data.append({"tick": tick, "q": q.tolist(), "velocity": motion.v[i].tolist(),
                     "readout": finite(readout), "organ_shift_mean_mm": liver.tolist(), "liver_spread_mm": spread,
                     "quality_image": bool(truth.qualified and genuine[0]), "travel_mm": float(travel + cost[i])})
    return weights, tick, float(travel + cost[-1]), data


def rotate_local(body, motor, envelope, q, liver_mean):
    """Yield distinct local rotation and fan trajectories from the same public state."""
    guess = body.anatomy(np.r_[liver_mean, np.zeros(4)])
    choices = []
    for dy, dtheta, roll in product((0., -5., 5.), (0., -2., 2.), (0., -8., 8.)):
        xy = q[:2] + [dy, dtheta]
        offset = envelope(xy[1]) - xy[0]
        if not (RIB_OFFSET[0] <= offset <= RIB_OFFSET[1]):
            continue
        aim = np.clip(body.aim(xy, guess.center, guess.axis) + [0., 0., roll, 0.], *CONTACT)
        predicted = body.measure(body.view(aim, np.zeros(2)), guess)
        score = (1.4 * predicted.qualified + .55 * predicted.coverage + .013 * predicted.gall[3]
                 - .014 * np.linalg.norm(body.surface(aim) - body.surface(q))
                 - .002 * np.linalg.norm(aim[2:] - q[2:]))
        choices.append((score, aim))
    previous = []
    for _, aim in sorted(choices, key=lambda item: item[0], reverse=True):
        if any(np.linalg.norm((aim - old) * [1., 2., .65, .65]) < 5. for old in previous):
            continue
        path = motor.replan(q, np.zeros(4), aim)
        if path is None:
            continue
        lateral_offset = envelope(path.x[:, 1]) - path.x[:, 0]
        if not np.isfinite(lateral_offset).all() or np.any(
                (lateral_offset < RIB_OFFSET[0] - 1.e-4) | (lateral_offset > RIB_OFFSET[1] + 1.e-4)):
            continue
        previous.append(aim)
        yield path
        if len(previous) == 3:
            break


def append_hold(motion):
    stationary = np.broadcast_to(motion.x[-1], (3 * FRAME_STEPS, 4))
    return Motion(np.vstack((motion.x, stationary)), np.vstack((motion.v, np.zeros_like(stationary))))


def run_case(seed):
    case, ordinary, true_anatomy = generate(seed)
    body = SagittalBody(ordinary.reference, case.beta)
    motor = Motor(body, case)
    prior = LiverPrior(body, body.anatomy(np.zeros(7)))
    envelope = costal_envelope(body)
    start = case.start.copy()
    start[0] = envelope(start[1]) - 32.
    start[2:] *= [.07, .30]
    plans = [sweep_motion(motor, envelope, start, final, offset) for final, offset in SWEEPS]
    out = {"seed": seed, "beta": case.beta.tolist(), "rib_shift_mm": case.rib.tolist(),
           "true_organ_params": case.organ.tolist(), "start": start.tolist(), "runs": {}}
    for copy, condition in enumerate(SENSOR):
        noise = Noise(seed, copy, condition, FRAMES)
        records = []
        for name, motion in enumerate(plans):
            weights, tick, travel, prefix = acquire(body, true_anatomy, case.rib, motion, noise, prior,
                                                    prior.initial, 0, 0.)
            pred_shift, width = prior.moment(weights)
            attempts = []
            for selected in rotate_local(body, motor, envelope, motion.x[-1], pred_shift):
                local = append_hold(selected)
                _, _, final_travel, tail = acquire(body, true_anatomy, case.rib, local, noise, prior, weights,
                                                  tick, travel)
                attempts.append({"target_chart": local.x[-1].tolist(), "total_travel_mm": final_travel,
                                 "qualified_view": any(p["quality_image"] for p in tail),
                                 "position_join_error": float(np.max(np.abs(motion.x[-1] - local.x[0]))),
                                 "velocity_join_error": float(np.max(np.abs(motion.v[-1] - local.v[0]))),
                                 "trace": tail})
            offsets = envelope(motion.x[:, 1]) - motion.x[:, 0]
            records.append({"sweep": name, "sweep_travel_mm": travel,
                            "clinical_guide_offsets_mm": [float(offsets.min()), float(offsets.max())],
                            "liver_frames_observed": sum(p["readout"][5] is not None for p in prefix),
                            "initial_shift_error_mm": float(np.linalg.norm(case.organ[:3])),
                            "sweep_shift_error_mm": float(np.linalg.norm(pred_shift - case.organ[:3])),
                            "initial_spread_mm": prior.moment(prior.initial)[1], "post_sweep_spread_mm": width,
                            "sweep_qualified": any(p["quality_image"] for p in prefix),
                            "sweep_trace": prefix, "rotation_attempts": attempts, "liver_shift_grid_mm": prior.shifts.tolist(),
                            "sweep_posterior": weights.tolist()})
        out["runs"][condition] = records
    return out


def summarize(rows):
    counts = {}
    for condition in SENSOR:
        tracks = [track for row in rows for track in row["runs"][condition]]
        attempts = [attempt for t in tracks for attempt in t["rotation_attempts"]]
        med = lambda field: float(np.median([track[field] for track in tracks]))
        counts[condition] = {"sweeps": len(tracks), "rotation_attempts": len(attempts),
                             "median_sweep_mm": med("sweep_travel_mm"),
                             "median_prior_liver_error_mm": med("initial_shift_error_mm"),
                             "median_sweep_liver_error_mm": med("sweep_shift_error_mm"),
                             "median_prior_spread_mm": med("initial_spread_mm"),
                             "median_sweep_spread_mm": med("post_sweep_spread_mm"),
                             "seen_liver_tracks": sum(t["liver_frames_observed"] > 0 for t in tracks),
                             "acquired_in_sweep": sum(t["sweep_qualified"] for t in tracks),
                             "qualified_rotation_attempts": sum(a["qualified_view"] for a in attempts),
                             "sweeps_with_qualified_rotation": sum(any(a["qualified_view"] for a in t["rotation_attempts"])
                                                                   for t in tracks),
                             "rotation_adds_first_qualifying_view": sum(not t["sweep_qualified"] and any(
                                 a["qualified_view"] for a in t["rotation_attempts"]) for t in tracks),
                             "maximum_position_join_error": max((a["position_join_error"] for a in attempts), default=0.),
                             "maximum_velocity_join_error": max((a["velocity_join_error"] for a in attempts), default=0.)}
    return {"roots": len(rows), "conditions": counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=int, default=48)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = range(58_645_000, 58_645_000 + args.roots)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as pool, gzip.open(
            args.out / "trajectory_candidates.jsonl.gz.part", "wt") as stream:
        for i, row in enumerate(pool.map(run_case, seeds), 1):
            stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
            for condition in row["runs"].values():
                for track in condition:
                    track.pop("sweep_trace")
                    for attempt in track["rotation_attempts"]:
                        attempt.pop("trace")
            rows.append(row)
            if i == 1 or i % 8 == 0 or i == args.roots:
                print(f"liver-guided collection: {i}/{args.roots} anatomy roots", flush=True)
    (args.out / "trajectory_candidates.jsonl.gz.part").replace(args.out / "trajectory_candidates.jsonl.gz")
    (args.out / "summary.json").write_text(json.dumps(summarize(rows), ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
