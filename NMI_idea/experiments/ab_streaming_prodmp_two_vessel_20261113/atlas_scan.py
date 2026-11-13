"""Run a material-address scan with MPC, in-segment ProDMP and realized FK."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(REPO / "rm75_control")]

from atlas_geometry import ScanTemplate  # noqa: E402
from check_kinematics import ik, load_witness  # noqa: E402
from primitive import (  # noqa: E402
    BLOCK_STEPS,
    FRAME_STRIDE,
    LATERAL_BOUNDS_MM,
    SAMPLE_DT_S,
    TOTAL_STEPS,
    LateralPrimitive,
    Plan,
    admissible,
)
from residual import (  # noqa: E402
    GRID_MM,
    MOTION_MODES,
    STATIONARY_SD_MM,
    TAU_S,
    ResidualFilter,
    motion_tape,
    offset_stats,
)
from sensor import CENTER_TOL_MM, IDENTITY_MIN, Sensor, UnsupportedObservation  # noqa: E402


SEED = 2026111390
DYNAMIC_SEED = 2026111403
VIEWS_MM = (-2., 0., 2.)
CENTRAL = VIEWS_MM.index(0.)
FRAMES_PER_BLOCK = BLOCK_STEPS // FRAME_STRIDE
FRAME_COUNT = TOTAL_STEPS // FRAME_STRIDE + 1
FRAME_DT_S = FRAME_STRIDE * SAMPLE_DT_S
RESIDUAL_ARMS = {"mpc_iid": "iid", "mpc_ou": "ou"}
PAIRED_ARMS = ("mpc_stream", *RESIDUAL_ARMS)
ARMS = (*PAIRED_ARMS, "fixed_first_stream", "mpc_boundary")


class KinematicRejection(RuntimeError):
    """A sampled path or its initial state failed robot kinematic checks."""


@dataclass(frozen=True)
class Trajectory:
    plan: Plan
    joints: np.ndarray
    poses: np.ndarray


def sha(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project(shape, pose, shifts_mm):
    """Intersect the two transported centrelines with a physical image plane.

    The fixed offset axes live in robot space. Each hypothesis shifts the same
    two 3-D curves, including when the probe tilts or loses a view.
    """
    shifts = np.atleast_2d(shifts_mm) @ shape.offset_axes.T * .001
    axes = pose[:3, :3]
    curves = (shape.vessels_m - pose[:3, 3]) @ axes
    curves = curves[None] + (shifts @ axes)[:, None, None, :]
    left, right = curves[:, :-1], curves[:, 1:]
    cross = ((left[..., 0] <= 0.) & (right[..., 0] > 0.)) | (
        (left[..., 0] >= 0.) & (right[..., 0] < 0.))
    fraction = np.divide(-left[..., 0], right[..., 0] - left[..., 0],
                         out=np.zeros_like(left[..., 0]), where=cross)
    points = left[..., 1:] + fraction[..., None] * (right - left)[..., 1:]
    # If the scan plane crosses one curve twice, prefer the nearer in-depth
    # intersection. A plane that misses yields a missing role.
    depth = np.where(cross & (points[..., 1] > 0.), points[..., 1], np.inf)
    which = np.argmin(depth, axis=1, keepdims=True)
    selected = np.take_along_axis(points, which[..., None], axis=1)[:, 0]
    return np.where(np.isfinite(depth.min(axis=1))[..., None], 1000. * selected, np.nan)


class Scan:
    def __init__(self, shape, capability, witness, regression):
        self.shape = shape
        self.capability = capability
        self.witness = witness
        self.regression = regression
        self.primitive = LateralPrimitive()
        length = float(shape.s_mm[-1])
        self.path = np.linspace(.10 * length, .90 * length, TOTAL_STEPS + 1)
        self.nominal_x, self.nominal_v = self.nominal_path()
        self.sensor = Sensor()
        self.models = {arm: ResidualFilter(self.sensor, FRAME_DT_S, model)
                       for arm, model in RESIDUAL_ARMS.items()}
        self.initial_x = float(self.nominal_x[0])
        initial_pose = shape.pose(self.path[0], self.initial_x)
        q = ik(capability, initial_pose, witness, witness[0])
        if not capability.check_q(q, initial_pose)["valid"]:
            raise KinematicRejection("no valid start configuration at the material anchor")
        self.initial_q = q
        self.initial_pose = capability.fk(q)

    def nominal_path(self):
        """Centre the public template artery in the transported probe plane."""
        stations = self.path[::BLOCK_STEPS // 2]
        views = np.linspace(*LATERAL_BOUNDS_MM, 61)
        nominal = []
        for station in stations:
            def lateral(view):
                pose = self.shape.pose(station, view)
                return self.project(pose, [0., 0.])[0, 0, 0]

            errors = np.array([lateral(view) for view in views])
            if not np.isfinite(errors).any():
                raise KinematicRejection(f"nominal artery has no plane crossing at station {station:.3f}")
            crossings = np.flatnonzero(np.isfinite(errors[:-1]) & np.isfinite(errors[1:])
                                       & (errors[:-1] * errors[1:] < 0.))
            nearest = int(np.nanargmin(np.abs(errors)))
            if len(crossings):
                i = int(crossings[np.argmin(np.abs(crossings + .5 - nearest))])
                nominal.append(brentq(lateral, views[i], views[i + 1], xtol=.002))
            else:
                nominal.append(views[nearest])
        nominal_spline = CubicSpline(stations, nominal, bc_type="natural")
        x = np.clip(nominal_spline(self.path), *LATERAL_BOUNDS_MM)
        return x, np.gradient(x, SAMPLE_DT_S)

    def project(self, pose, offsets):
        return project(self.shape, pose, offsets)

    def candidate(self, step, end, x, v, goal, q):
        goal = float(np.clip(goal, *LATERAL_BOUNDS_MM))
        weights = np.array([1., goal - x, v]) @ self.regression
        plan = self.primitive.condition(step, end - step, x, v, goal, self.nominal_v[end], weights)
        if not admissible(plan):
            return None
        targets = self.shape.pose(self.path[step:end + 1], plan.x)
        joints, poses = [q], [self.capability.fk(q)]
        for target in targets[1:]:
            previous = joints[-1]
            current = ik(self.capability, target, previous, self.witness[0])
            if (not np.all(np.isfinite(current))
                    or np.any(np.abs(current - previous) > SAMPLE_DT_S * self.capability.kin.v_max + 1.e-10)
                    or not self.capability.check_q(current, target)["valid"]):
                return None
            joints.append(current)
            poses.append(self.capability.fk(current))
        return Trajectory(plan, np.asarray(joints), np.asarray(poses))

    def acquisition_score(self, trajectory, model, belief, future_frames):
        # Expected useful acquisitions at three representative exposures.
        # Score in-segment evidence where the continuation can use it.
        probes = np.unique(np.round(np.linspace(1, FRAMES_PER_BLOCK, 3)).astype(int))
        value = sum(model.predicted_hit(
            belief, self.project(trajectory.poses[frame * FRAME_STRIDE], model.offsets_mm),
            future_frames + frame) for frame in probes)
        return value / len(probes)

    def choices(self, step, x, v, q):
        end = step + BLOCK_STEPS
        nominal = self.nominal_x[end]
        goals = np.clip(nominal + np.array(VIEWS_MM), *LATERAL_BOUNDS_MM).tolist()
        trajectories = {goal: self.candidate(step, end, x, v, goal, q) for goal in dict.fromkeys(goals)}
        if not any(plan is not None for plan in trajectories.values()):
            # A large feedback correction can leave the nominal window outside
            # the current rate-limited reachable set. Try retaining the lateral
            # offset while advancing, or simply holding the current x.
            carry = x + nominal - self.nominal_x[step]
            for goal in np.clip([carry, x], *LATERAL_BOUNDS_MM):
                if goal not in trajectories:
                    goals.append(float(goal))
                    trajectories[goal] = self.candidate(step, end, x, v, goal, q)
        return goals, [trajectories[goal] for goal in goals]

    def select(self, step, x, v, q, model, belief, fixed_first):
        """Compare fixed two-segment nominal plans, then execute only segment one."""
        end = step + BLOCK_STEPS
        goals, choices = self.choices(step, x, v, q)
        values = np.full(len(choices), -np.inf)
        immediate = values.copy()
        for i, first in enumerate(choices):
            if first is None:
                continue
            immediate[i] = self.acquisition_score(first, model, belief, 0) - .008 * travel(first.poses)
            if end == TOTAL_STEPS:
                values[i] = immediate[i]
                continue
            _, continuation = self.choices(end, first.plan.x[-1], first.plan.v[-1], first.joints[-1])
            for second in continuation:
                if second is not None:
                    tail = self.acquisition_score(second, model, belief, FRAMES_PER_BLOCK) - .008 * travel(second.poses)
                    values[i] = max(values[i], immediate[i] + .65 * tail)
        if not np.any(np.isfinite(immediate)):
            raise KinematicRejection("no feasible first segment from the realized joint state")
        if not np.any(np.isfinite(values)):
            values = immediate
        if fixed_first and choices[CENTRAL] is None:
            raise KinematicRejection("fixed central first segment unavailable")
        chosen = CENTRAL if fixed_first else int(np.argmax(values))
        report = {"step": step, "view_offset_ref_mm": float(goals[chosen] - self.nominal_x[end]),
                  "goal_ref_mm": goals[chosen], "candidate_goals_ref_mm": goals,
                  "used_carry_or_hold": chosen >= len(VIEWS_MM),
                  "values": [float(x) if np.isfinite(x) else None for x in values],
                  "fixed_first_center_available": choices[CENTRAL] is not None}
        return choices[chosen], report

    def revise(self, old, step, end, x, v, q, model, belief, pose, packet, slot):
        # Convert image error in physical mm to the template lateral unit,
        # using the end-of-segment image Jacobian for the current shape.
        current_offset = model.offset_mean(belief)
        end_offset = model.offset_mean(belief, (end - step) // FRAME_STRIDE)
        nominal = self.nominal_x[end]
        lateral = np.clip([nominal - .05, nominal + .05], *LATERAL_BOUNDS_MM)
        end_poses = self.shape.pose(self.path[end], [lateral[0], nominal, lateral[1]])
        left, center, right = [self.project(p, end_offset)[0, 0, 0] for p in end_poses]
        derivative = (left - right) / (lateral[1] - lateral[0])
        if not np.isfinite(derivative) or derivative <= .10 or not np.isfinite(center):
            return old
        prediction = self.project(pose, current_offset)[0, 0, 0]
        innovation = packet[slot, 0] - prediction if np.isfinite(prediction) else 0.
        goal = nominal + np.clip((center + .28 * innovation) / derivative, -7., 7.)
        if abs(goal - old.plan.goal) < .08:
            return old
        for fraction in (1., .5):
            adjusted = old.plan.goal + fraction * (goal - old.plan.goal)
            updated = self.candidate(step, end, x, v, adjusted, q)
            if updated is not None:
                return updated
        return old

    def observation(self, scene, frame, pose, z_mm):
        """Generate an observation at the actual FK pose in the shared world."""
        effective = scene.offset + [z_mm, 0.] if z_mm else scene.offset
        measured = self.project(pose, effective)[0]
        physical = self.project(pose, effective - scene.bias)[0]
        return self.sensor.observe_means(scene, frame, measured, physical)

    def execute(self, scene, arm, motion_mm=None):
        x, v = self.initial_x, 0.
        q, pose = self.initial_q, self.initial_pose
        model = self.models.get(arm, self.sensor)
        belief = model.prior()
        joints, poses, lateral = [q], [pose], [[x, v]]
        frames, decisions = [], []
        trajectory, failure = None, None
        revisions = 0
        max_handoff_x, max_handoff_v = 0., 0.
        for frame in range(FRAME_COUNT):
            step = frame * FRAME_STRIDE
            means = self.project(pose, model.offsets_mm)
            packet, roles, true_u = self.observation(scene, frame, pose, 0. if motion_mm is None else motion_mm[frame])
            try:
                belief, association = model.update_projected(belief, packet, means)
            except UnsupportedObservation as error:
                failure = str(error)
                break
            slot = int(np.argmax(association)) if len(association) and association.max() >= IDENTITY_MIN else -1
            accepted = slot >= 0 and abs(packet[slot, 0]) <= CENTER_TOL_MM
            wrong = accepted and roles[slot] != 0
            correct = accepted and not wrong and np.isfinite(true_u[0]) and abs(true_u[0]) <= CENTER_TOL_MM
            frames.append({"t_s": frame * FRAME_STRIDE * SAMPLE_DT_S,
                           "station_ref_mm": float(self.path[step]), "packet_mm": packet.tolist(),
                           "role_truth": roles.tolist(),
                           "artery_probability": association.tolist(), "selected_slot": slot,
                           "accepted": bool(accepted), "correct": bool(correct), "wrong": bool(wrong),
                           "physical_lateral_mm": float(true_u[0]) if np.isfinite(true_u[0]) else None})
            if motion_mm is not None:
                mean, interval = offset_stats(model, belief)
                frames[-1].update(offset_mean_mm=mean.tolist(), offset_interval90_mm=interval.tolist(),
                                  dynamic_lateral_mean_mm=model.residual_mean(belief) if arm in self.models else 0.)
            if step == TOTAL_STEPS:
                break
            end = (step // BLOCK_STEPS + 1) * BLOCK_STEPS
            if step % BLOCK_STEPS == 0:
                try:
                    trajectory, report = self.select(step, x, v, q, model, belief,
                                                     frame == 0 and arm == "fixed_first_stream")
                except KinematicRejection as error:
                    failure = str(error)
                    break
                decisions.append(report)
            elif arm != "mpc_boundary" and slot >= 0:
                updated = self.revise(trajectory, step, end, x, v, q, model, belief, pose, packet, slot)
                if updated is not trajectory:
                    next_old = trajectory.poses[step - trajectory.plan.step + FRAME_STRIDE, :3, 3]
                    next_new = updated.poses[FRAME_STRIDE, :3, 3]
                    change = float(1000. * np.linalg.norm(next_new - next_old))
                    frames[-1]["next_frame_shift_from_revision_mm"] = change
                    trajectory = updated
                    revisions += int(change >= .001)
            i = step - trajectory.plan.step
            max_handoff_x = max(max_handoff_x, abs(trajectory.plan.x[i] - x))
            max_handoff_v = max(max_handoff_v, abs(trajectory.plan.v[i] - v))
            j = i + FRAME_STRIDE
            joints.extend(trajectory.joints[i + 1:j + 1])
            poses.extend(trajectory.poses[i + 1:j + 1])
            lateral.extend(np.column_stack((trajectory.plan.x[i + 1:j + 1],
                                           trajectory.plan.v[i + 1:j + 1])))
            x, v = trajectory.plan.x[j], trajectory.plan.v[j]
            q, pose = joints[-1], poses[-1]
        return score(frames, failure, np.asarray(poses), self.path, decisions, revisions,
                     max_handoff_x, max_handoff_v), (
            np.asarray(joints), np.asarray(poses), np.asarray(lateral))


def travel(poses):
    return float(1000. * np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).sum())


def score(frames, failure, poses, path, decisions, revisions, handoff_x, handoff_v):
    def bins(key):
        return {min(4, int(5 * (frame["station_ref_mm"] - path[0]) / (path[-1] - path[0])))
                for frame in frames if frame[key]}

    wrong = any(frame["wrong"] for frame in frames)
    covered = len(bins("correct"))
    rotations = Rotation.from_matrix(poses[:, :3, :3])
    turn = np.rad2deg((rotations[:-1].inv() * rotations[1:]).magnitude()).sum() if len(poses) > 1 else 0.
    return dict(success=failure is None and covered == 5 and not wrong, wrong=wrong,
                false_complete=len(bins("accepted")) == 5 and (failure is not None or covered != 5 or wrong),
                coverage=covered, failure=failure, decisions=decisions, frames=frames,
                tcp_mm=travel(poses), turn_deg=float(turn), duration_s=(len(poses) - 1) * SAMPLE_DT_S,
                revisions=revisions, max_chart_handoff_x_ref_mm=float(handoff_x),
                max_chart_handoff_v_ref_mm_s=float(handoff_v))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--among-us", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    shape_arg = parser.add_mutually_exclusive_group()
    shape_arg.add_argument("--delta-beta", nargs=2, type=float)
    shape_arg.add_argument("--beta", nargs=10, type=float)
    parser.add_argument("--knee-deg", type=float, default=15.)
    parser.add_argument("--cases", type=int, default=16)
    parser.add_argument("--arm", action="append", choices=ARMS)
    parser.add_argument("--motion", choices=MOTION_MODES, default="static",
                        help="OU and transient worlds default to a paired static-map, IID and OU test")
    args = parser.parse_args(argv)
    if args.cases < 1:
        parser.error("--cases must be positive")
    from NMI_idea.experiments.ab_module_ladder_20261110.capability.capability import CapabilityEvaluator

    template = ScanTemplate(args.among_us, args.repair)
    shape = template.instantiate(args.delta_beta or (0., 0.), args.knee_deg, beta=args.beta)
    bank = HERE.parent / "ab_full_b_rollout_20261113/motion/BANK_MAIN.json.xz"
    demonstration = HERE / "verification/DEMONSTRATIONS.npz"
    capability = CapabilityEvaluator(device="cpu")
    witness_id, witness = load_witness(bank)
    with np.load(demonstration, allow_pickle=False) as frozen:
        scan = Scan(shape, capability, witness, frozen["shape_regression"])
    arms = args.arm or (("mpc_stream",) if args.motion == "static" else PAIRED_ARMS)
    log_motion = args.motion != "static" or any(arm in RESIDUAL_ARMS for arm in arms)
    rng = np.random.default_rng(SEED)
    motion_rng = np.random.default_rng(DYNAMIC_SEED)
    cases = []
    ref_stations = shape.s_mm[::8]
    traces = {"witness_q": witness, "nominal_vessels_m": shape.vessels_m,
              "vessel_material_xi": template.vessel_xi,
              "sample_stations_ref_mm": ref_stations,
              "sample_material_probe_poses": shape.pose(ref_stations, np.zeros_like(ref_stations))}
    if log_motion:
        traces["offset_axes"] = shape.offset_axes
    for i in range(args.cases):
        scene = scan.sensor.scene(rng, FRAME_COUNT)
        z_mm = motion_tape(motion_rng, FRAME_COUNT, FRAME_DT_S, args.motion) if log_motion else None
        if log_motion:
            traces[f"{i}_z_mm"] = z_mm
        traces[f"{i}_clear"] = scene.clear
        traces[f"{i}_noise_mm"] = scene.noise
        row = {"case": i, "hidden_static_offset_mm": scene.offset.tolist(),
               "hidden_calibration_bias_mm": scene.bias.tolist(), "arms": {}}
        for arm in arms:
            result, (q, poses, lateral) = scan.execute(scene, arm, z_mm)
            row["arms"][arm] = result
            key = f"{i}_{arm}"
            traces[f"{key}_q"] = q
            traces[f"{key}_fk_m"] = poses
            traces[f"{key}_x_v"] = lateral
        cases.append(row)
        print(f"case {i + 1}/{args.cases}", flush=True)
    means = {}
    for arm in arms:
        rows = [case["arms"][arm] for case in cases]
        keys = ("success", "wrong", "false_complete", "coverage", "tcp_mm", "turn_deg", "duration_s", "revisions")
        means[arm] = {key: float(np.mean([row[key] for row in rows])) for key in keys}
        means[arm]["run_failures"] = sum(row["failure"] is not None for row in rows)
    report = {
        "scope": "Known static SMPL-X beta and knee pose; two synthetic centreline returns; sampled RM75 kinematics in the feedback loop",
        "geometry": shape.metadata,
        "seed": SEED,
        "case_count": len(cases),
        "frames_per_complete_run": FRAME_COUNT,
        "kinematic_witness_node_id": witness_id,
        "fixed_rail_m": float(witness[0]),
        "sample_dt_s": SAMPLE_DT_S,
        "frame_dt_s": SAMPLE_DT_S * FRAME_STRIDE,
        "nominal_lateral_range_ref_mm": [float(scan.nominal_x.min()), float(scan.nominal_x.max())],
        "model_input_sha256": {bank.name: sha(bank), demonstration.name: sha(demonstration)},
        "robot_input_sha256": {Path(name).name: digest
                                for name, digest in capability.manifest()["source_sha256"].items()},
        "means": means,
        "cases": cases,
    }
    if log_motion:
        report["scope"] = ("Known static SMPL-X beta and knee pose; two synthetic centreline returns; "
                           "shared lateral vessel motion relative to the chart; sampled RM75 kinematics")
        report["motion"] = {"kind": args.motion, "seed": DYNAMIC_SEED, "tau_s": TAU_S,
                            "stationary_sd_mm": STATIONARY_SD_MM, "filter_grid_mm": GRID_MM}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "SUMMARY.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.out / "TRACES.npz", station_ref_mm=scan.path, **traces)
    print(json.dumps(means, indent=2), flush=True)


if __name__ == "__main__":
    main()
