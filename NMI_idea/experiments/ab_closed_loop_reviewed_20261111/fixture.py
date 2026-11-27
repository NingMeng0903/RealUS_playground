"""Public two-window fixture, measured patch and exact native motion rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from robot_motion import MotionState, NativeMotionEnvironment, WindowConstraint


HERE = Path(__file__).resolve().parent
GEOMETRY = HERE.parent / "ab_live_surface_20261111"
_spec = importlib.util.spec_from_file_location("ab_live_patch", GEOMETRY / "run_geometry.py")
_patch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_patch)
ASSETS = _patch.ASSETS
TERMINALS = ("left", "right")
DWELL = {"side": 0.15, "fork": 0.30, "terminal": 1.0}
SETTLE_S = 0.12
DEADLINE_EPS_S = 1e-8


@dataclass(frozen=True)
class Capture:
    name: str
    kind: str
    goal: tuple[float, float, float]
    window: str

    @property
    def dwell_s(self):
        return DWELL[self.kind]


@dataclass(frozen=True)
class EpisodeState:
    key: tuple
    elapsed: float
    pose: np.ndarray
    native: MotionState | None
    valid: bool = True


@dataclass(frozen=True)
class Transfer:
    output: EpisodeState
    reason: str
    duration: float
    route_id: str | None


@dataclass(frozen=True)
class CaptureResult:
    capture: Capture
    transfer: Transfer


def local_pose(frame, pose):
    rotation = frame[:3, :3].T @ pose[:3, :3]
    position = frame[:3, :3].T @ (pose[:3, 3] - frame[:3, 3])
    return np.array([position[0], position[1], np.arctan2(rotation[2, 1], rotation[1, 1])])


def segment_hits_box(start, goal, box):
    """Exact slab intersection for the planar segment between two probe poses."""
    enter, leave = 0.0, 1.0
    for p, delta, low, high in zip(start[:2], np.subtract(goal[:2], start[:2]), box[::2], box[1::2]):
        if abs(delta) < 1e-12:
            if not low <= p <= high:
                return False
            continue
        a, b = sorted(((low - p) / delta, (high - p) / delta))
        enter, leave = max(enter, a), min(leave, b)
        if enter > leave:
            return False
    return True


def segment_duration(start, goal):
    distance = np.linalg.norm(np.subtract(start[:2], goal[:2]))
    turn = abs(float(goal[2] - start[2]))
    return max(0.55, 1.68 * distance / 0.027, 1.68 * turn / 0.30)


def timed_route(start, waypoints):
    times = [segment_duration(a, b) for a, b in zip((start, *waypoints[:-1]), waypoints)]
    return float(sum(times) + SETTLE_S * len(times))


@dataclass(frozen=True)
class Rig:
    name: str
    address: dict
    root_pose: np.ndarray
    measured: np.ndarray
    actual_surface: np.ndarray
    wall: tuple | None
    main_side: str
    width: float
    terminal_y: float
    sensor_points: np.ndarray
    nominal: np.ndarray
    atlas_frame: np.ndarray
    atlas_to_robot: np.ndarray

    @cached_property
    def geometry(self):
        return WindowConstraint(
            name=self.name, xmin=-0.030, xmax=0.030, ymin=-0.024, ymax=0.026,
            blocked=() if self.wall is None else (self.wall,),
        )

    @cached_property
    def captures(self):
        offset = 0.0008 if self.main_side == "left" else -0.0008
        yl, yr = self.terminal_y - offset, self.terminal_y + offset
        return {
            "left": Capture("left", "terminal", (-self.width, yl, 0.10), "left"),
            "right": Capture("right", "terminal", (self.width, yr, -0.10), "right"),
            "side_left": Capture("side_left", "side", (-self.width, yl - 0.005, 0.075), "left"),
            "side_right": Capture("side_right", "side", (self.width, yr - 0.005, -0.075), "right"),
            "fork_minus": Capture("fork_minus", "fork", (0.006, -0.003, -0.075), "fork"),
            "fork_plus": Capture("fork_plus", "fork", (-0.006, -0.003, 0.075), "fork"),
        }

    @property
    def fixed_pair(self):
        if self.main_side == "left":
            return ("fork_minus", "fork_plus")
        return ("fork_plus", "fork_minus")

    @property
    def programs(self):
        # The first programs match the two accessible side apertures and the two orders
        # of the central diagnostic pair. All crossed terminal retreats remain legal.
        return {
            "side_left": ("side_left",), "side_right": ("side_right",),
            "fork_minus": ("fork_minus",), "fork_plus": ("fork_plus",),
            "pair_minus_plus": ("fork_minus", "fork_plus"),
            "pair_plus_minus": ("fork_plus", "fork_minus"),
        }

    @property
    def start(self):
        return EpisodeState((), 0.0, self.root_pose.copy(), None)

    def public_metadata(self):
        return {
            "name": self.name, "material_address": self.address,
            "wall_tcp_envelope_m": self.wall, "main_side": self.main_side,
            "nominal_frame": self.nominal.tolist(), "measured_frame": self.measured.tolist(),
            "production_atlas_frame": self.atlas_frame.tolist(),
            "atlas_to_robot": self.atlas_to_robot.tolist(),
            "surface_measurement_count": len(self.sensor_points),
            "goals": {k: v.goal for k, v in self.captures.items()},
            "fixed_pair": self.fixed_pair,
        }

    def clear(self, start, waypoints):
        if not all(self.geometry.check_local(point[:2])[0] for point in (start, *waypoints)):
            return False
        if self.wall is None:
            return True
        return not any(segment_hits_box(a, b, self.wall) for a, b in zip((start, *waypoints[:-1]), waypoints))

    def route_options(self, start, end):
        end = tuple(end)
        candidates = {"direct": (end,)}
        if self.wall is not None and segment_hits_box(start, end, self.wall):
            xlo, xhi, ymin, _ = self.wall
            sign = 1.0 if end[0] > start[0] else -1.0
            near = (xlo - 0.0025) if sign > 0.0 else (xhi + 0.0025)
            far = (xhi + 0.0025) if sign > 0.0 else (xlo - 0.0025)
            for label, y in (("round_tip", ymin - 0.003), ("retreat_fork", -0.003)):
                angle = float(start[2])
                candidates[label] = ((near, y, angle), (far, y, angle), end)
        return {name: waypoints for name, waypoints in candidates.items() if self.clear(start, waypoints)}


def make_rig(index, mode, native, *, seed=721411):
    """Generate public geometry without generating any acoustic test outcomes."""
    # Mirrors use a common fixture and independent target seeds. Different indices
    # produce independently calibrated sensor patches on the production address set.
    family = index // 2
    rng = np.random.default_rng(np.random.SeedSequence([seed, family]))
    geometry = json.loads((GEOMETRY / "GEOMETRY_RESULTS.json").read_text())
    addresses = [r["address"] for r in geometry["records"] if r["knee_pose_deg"] == 30]
    address = addresses[(family * 11 + 5) % len(addresses)]
    side, index_skin = address["side"], address["skin_vertex_id"]
    with np.load(ASSETS / f"atlas_{side}.npz") as atlas, np.load(ASSETS / f"{side}_pose_30.npz") as pose:
        canonical, faces = atlas["skin_vertices"].astype(float), atlas["skin_faces"]
        distances, ids = cKDTree(atlas["harmonic_vertices"]).query(canonical)
        assert np.max(distances) == 0. and np.all(atlas["harmonic_d"][ids] == 0.)
        verts = pose["vertices"][ids].astype(float)
        face_ids = np.flatnonzero((faces == index_skin).any(axis=1))
        sign = np.sign(_patch.incident_normals(canonical, faces, face_ids) @ atlas["skin_normals"][index_skin])
        normal = sign * _patch.incident_normals(verts, faces, face_ids)
        neighbors = np.unique(faces[face_ids])
        delta = verts[neighbors] - verts[index_skin]
        delta -= (delta @ normal)[:, None] * normal
        direction = np.linalg.lstsq(delta, atlas["skin_h"][neighbors] - atlas["skin_h"][index_skin], rcond=1e-8)[0]
        atlas_frame = np.eye(4)
        atlas_frame[:3, :3] = _patch.frame(normal, direction)
        atlas_frame[:3, 3] = verts[index_skin]
    atlas_to_robot = native.anchor @ np.linalg.inv(atlas_frame)
    nominal = atlas_to_robot @ atlas_frame
    true = nominal.copy()
    normal_bias_m = rng.uniform(-0.0015, 0.0015)
    angles = np.deg2rad(rng.uniform(-1.25, 1.25, size=2))
    true[:3, :3] = nominal[:3, :3] @ Rotation.from_rotvec([*angles, 0.0]).as_matrix()
    true[:3, 3] += normal_bias_m * nominal[:3, 2]
    xy = _patch.PATCH.copy()
    points_mm = np.column_stack([xy, np.zeros(len(xy))]) @ true[:3, :3].T + 1000.0 * true[:3, 3]
    points_mm += rng.normal(scale=0.25, size=points_mm.shape)
    status, estimate = _patch.estimate_contact(points_mm, nominal[:3, 3] * 1000.0, nominal[:3, 2], nominal[:3, 0])
    if status != "accepted":
        raise RuntimeError(f"fixture {index}: surface estimator rejected {status}")
    origin, basis = estimate
    measured = np.eye(4)
    measured[:3, :3], measured[:3, 3] = basis, origin / 1000.0
    width = float(rng.uniform(0.015, 0.017))
    terminal_y = float(rng.uniform(0.017, 0.019))
    wall_halfwidth = float(rng.uniform(0.0036, 0.0044))
    wall_start = float(rng.uniform(0.009, 0.010))
    wall_center = 0.0011 if index % 2 == 0 else -0.0011
    wall = None if mode == "open" else (wall_center - wall_halfwidth, wall_center + wall_halfwidth, wall_start, 0.032)
    return Rig(
        f"fixture_{index:02}_{mode}", address, nominal, measured, true, wall,
        "left" if index % 2 == 0 else "right", width, terminal_y, points_mm / 1000.0, nominal,
        atlas_frame, atlas_to_robot,
    )


def checked_trace(trace, rig):
    poses = np.asarray([sample["pose_actual"] for sample in trace])
    world_points = poses[:, :3, 3]
    public_xyz = (world_points - rig.measured[:3, 3]) @ rig.measured[:3, :3]
    true_xyz = (world_points - rig.actual_surface[:3, 3]) @ rig.actual_surface[:3, :3]
    in_bounds = all(rig.geometry.check_local(point[:2])[0] for point in public_xyz)
    swept_clear = rig.wall is None or not any(
        segment_hits_box(a, b, rig.wall) for a, b in zip(public_xyz[:-1], public_xyz[1:])
    )
    gap = float(np.max(np.abs(true_xyz[:, 2])))
    return bool(in_bounds and swept_clear and gap <= 0.003), gap, public_xyz


class MotionRollouts:
    """Finite checked native motion catalog. Cache keys contain the complete event prefix."""

    def __init__(self, rig: Rig, native: NativeMotionEnvironment, output: Path | None, *, model=None):
        self.rig = rig
        self.native = native
        self.output = output
        self.model = model
        if output:
            output.mkdir(parents=True, exist_ok=True)
        self.cache: dict[tuple, CaptureResult] = {}
        self.entries = {}
        self.cache_hits = 0
        self.rows = []
        self.dird_queries = 0

    def candidates(self, state, capture):
        start = local_pose(self.rig.measured, state.pose)
        options = self.rig.route_options(start, capture.goal)
        if not options:
            return []
        q = self.native.q_seed if state.native is None else state.native.q
        poses = []
        segments = []
        for name, waypoints in options.items():
            route = [start]
            for a, b in zip((start, *waypoints[:-1]), waypoints):
                route.extend([(np.asarray(a) + b) / 2.0, b])
            offset = len(poses)
            poses.extend(self.native.measured_goal_pose(self.rig.measured, point) for point in route)
            segments.append((name, waypoints, offset, len(poses)))
        report = self.native.capability.score(np.asarray(poses), q[0])
        self.dird_queries += 1
        margins = np.asarray(report["raw_calibrated_margin"])
        ranked = []
        for name, points, low, high in segments:
            duration = timed_route(start, points)
            rank = float(np.mean(margins[low:high]))
            ranked.append((name, points, duration, rank))
        # Near-equal time routes prefer the one with greater predicted dIRD margin.
        # All routes remain eligible if a higher priority candidate fails execution.
        ranked.sort(key=lambda row: (round(row[2] / 0.10), -row[3], row[2], row[0]))
        return ranked

    def _run_route(self, state, capture, name, waypoints, estimate_s, margin):
        current = state
        route_rows = []
        route_started = time.perf_counter()
        label = "__".join((*state.key, capture.name, name))
        for step, goal in enumerate(waypoints):
            start_local = local_pose(self.rig.measured, current.pose)
            planned = segment_duration(start_local, goal)
            if capture.kind == "fork":
                planned = max(planned, 1.10 if abs(goal[2] - start_local[2]) > 0.12 else 0.80)
            final = step == len(waypoints) - 1
            result = self.native.execute_local(
                start_local, goal, self.rig.measured, planned, current.native,
                settle_s=SETTLE_S + capture.dwell_s * final,
                label=f"{self.rig.name}/{label}/{step}",
            )
            trace = result.get("trace", [])
            valid, gap, xy = checked_trace(trace, self.rig) if trace else (False, None, None)
            native_gate = bool(result.get("motion_tracking_pass_2mm_2deg", False))
            aligned_position, aligned_rotation = None, None
            if trace:
                after = np.array([row["pose_actual"] for row in trace])
                before = np.concatenate((current.pose[None], after[:-1]), axis=0)
                desired = np.array([row["pose_desired"] for row in trace])
                aligned_position = float(np.max(np.linalg.norm(before[:, :3, 3] - desired[:, :3, 3], axis=1)))
                mismatch = np.swapaxes(before[:, :3, :3], 1, 2) @ desired[:, :3, :3]
                aligned_rotation = float(np.max(Rotation.from_matrix(mismatch).magnitude()))
            aligned_gate = trace and aligned_position <= .002 and aligned_rotation <= np.deg2rad(2.)
            motion_ok = bool(native_gate and aligned_gate)
            paused = int(result.get("task_paused_count", 0))
            duration = float(result.get("simulated_duration_s") or result.get("integrated_duration_s") or 0.0)
            valid = valid and motion_ok and not paused and result.get("exit_state_valid", False)
            result_id = f"{label}_{step}"
            if trace and self.output:
                np.savez_compressed(
                    self.output / f"{result_id}.npz",
                    time_s=np.array([row["t_s"] for row in trace]),
                    q=np.array([row["q"] for row in trace]),
                    qdot=np.array([row["qdot"] for row in trace]),
                    desired=np.array([row["pose_desired"] for row in trace]),
                    actual=np.array([row["pose_actual"] for row in trace]),
                    measured_before=before,
                    local_xyz_m=xy,
                    self_clearance_m=np.array([row["min_self_distance_m"] for row in trace]),
                )
            item = {
                "step": step, "record": result_id, "duration_s": duration, "reference_duration_s": planned,
                "requested_dwell_s": capture.dwell_s * final, "reference": result.get("actual_reference_kind"),
                "tracking_position_p95_m": result.get("position_error_p95_m"),
                "tracking_rotation_p95_rad": result.get("rotation_error_p95_rad"),
                "control_ticks": len(trace), "tracking_pass": motion_ok, "native_tracking_gate": native_gate, "valid": valid,
                "same_tick_position_error_max_m": aligned_position, "same_tick_rotation_error_max_rad": aligned_rotation,
                "configured_jerk_margin_min": result.get("jerk_margin_min"),
                "start_q": (self.native.q_seed if current.native is None else current.native.q).tolist(),
                "start_qdot": (np.zeros(8) if current.native is None else current.native.qdot).tolist(),
                "end_q": np.asarray(result["q_exit"]).tolist(), "end_qdot": np.asarray(result["qdot_exit"]).tolist(),
                "position_error_max_m": result.get("position_error_max_m"),
                "rotation_error_endpoint_rad": result.get("endpoint_rotation_error_rad"),
                "synthetic_contact_gap_max_m": gap, "paused": paused,
                "native_failure_count": len(result.get("failures", [])),
                "minimum_self_collision_distance_m": result.get("minimum_self_collision_distance_m"),
                "reconstruction_error": result.get("prefix_reconstruction_max_abs_error"),
                "wall_seconds": result.get("wall_seconds"),
                "restored_prefix_wall_seconds": result.get("prefix_restore_wall_seconds"),
            }
            route_rows.append(item)
            native_state = MotionState.from_result(result) if result.get("exit_state_valid", False) else None
            pose = np.asarray(result.get("pose_exit", current.pose))
            current = EpisodeState(state.key + (capture.name,), current.elapsed + duration, pose, native_state, valid)
            if not valid:
                break
        row = {
            "route": label, "start": state.key, "capture": capture.name, "candidate": name,
            "commanded_duration_prediction_s": estimate_s + capture.dwell_s,
            "dird_mean_calibrated_margin": margin, "valid": current.valid,
            "duration_s": current.elapsed - state.elapsed, "steps": route_rows,
            "calculation_wall_seconds": time.perf_counter() - route_started,
        }
        self.rows.append(row)
        return Transfer(current, "executed" if current.valid else "native_or_workspace_guard", row["duration_s"], label)

    def capture(self, state, name):
        if not state.valid:
            raise ValueError("cannot continue from a rejected actual state")
        key = (*state.key, name)
        if key in self.cache:
            cached = self.cache[key]
            entry = self.entries[key]
            a = None if entry.native is None else entry.native.snapshot
            b = None if state.native is None else state.native.snapshot
            if entry.elapsed != state.elapsed or not np.array_equal(entry.pose, state.pose) or a != b:
                raise ValueError("cached prefix differs from complete deterministic entry state")
            self.cache_hits += 1
            return cached
        capture = self.rig.captures[name]
        candidates = self.candidates(state, capture)
        if self.model is None:
            attempts = [self._run_route(state, capture, *candidate) for candidate in candidates]
            valid = [route for route in attempts if route.output.valid]
            attempt = min(valid, key=lambda route: route.duration) if valid else None
        else:
            prediction = self.model.get(state.key, name)
            if prediction is None:
                prediction = self.model.capture(state, name)
            selected_route = prediction.transfer.route_id
            attempt = None
            for candidate in candidates:
                label = "__".join((*state.key, capture.name, candidate[0]))
                if selected_route == label:
                    attempt = self._run_route(state, capture, *candidate)
                    break
        if attempt is None:
            attempt = Transfer(
                EpisodeState(key, state.elapsed, state.pose.copy(), state.native, False),
                "no_checked_route", 0.0, None,
            )
        result = CaptureResult(capture, attempt)
        self.cache[key] = result
        self.entries[key] = state
        return result

    def program(self, state, sequence):
        events = []
        for name in sequence:
            result = self.capture(state, name)
            events.append(result)
            state = result.transfer.output
            if not state.valid:
                break
        return state, events

    def get(self, prefix, name):
        return self.cache.get((*prefix, name))

    def finish(self):
        if self.output:
            (self.output / "MOTION.json").write_text(
                json.dumps({"fixture": self.rig.public_metadata(), "dird_queries": self.dird_queries, "execution_cache_hits": self.cache_hits, "routes": self.rows},
                           indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else x),
                encoding="utf-8",
            )

