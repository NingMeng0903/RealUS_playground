"""Follow a programmed knee motion and infer relative vessel motion from US."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from atlas_geometry import REFERENCE_KNEE_DEG
from atlas_scan import FRAME_DT_S, Scan, project
from pose_response import PoseResponseFilter
from primitive import FRAME_STRIDE, TOTAL_STEPS
from sensor import Scene


KNEE_ANGLES_DEG = (5., 10., 15., 20., 25.)
RESPONSE_ARM = "mpc_ou_g"


def knee_envelope(station_mm, knee_station_mm):
    """One on the knee side, smoothly tapering to zero at 17 mm."""
    distance = np.abs(np.asarray(station_mm) - knee_station_mm)
    phase = np.clip((distance - 5.) / 12., 0., 1.)
    return .5 * (1. + np.cos(np.pi * phase))


def rotation_between(first, last, weight):
    first = Rotation.from_matrix(first)
    last = Rotation.from_matrix(last)
    relative = (first.inv() * last).as_rotvec()
    return (first * Rotation.from_rotvec(np.asarray(weight)[..., None] * relative)).as_matrix()


def frame_from_offset_axes(axes):
    return np.column_stack((np.cross(axes[:, 0], axes[:, 1]), axes))


@dataclass(frozen=True)
class VesselGeometry:
    vessels_m: np.ndarray
    offset_axes: np.ndarray


class KneeAtlas:
    """Interpolate published pose samples at the corresponding material points."""

    def __init__(self, template, *, check_midpoints=True):
        self.template = template
        self.angles = np.asarray(KNEE_ANGLES_DEG)
        self.worlds = [template.instantiate((0., 0.), angle) for angle in self.angles]
        self.shapes = [template.with_spline(world) for world in self.worlds]
        self.reference = self.shapes[KNEE_ANGLES_DEG.index(REFERENCE_KNEE_DEG)]
        _, _, _, joints = template._transport.deform("right", (0., 0.), REFERENCE_KNEE_DEG)
        transform = template.robot_from_atlas
        knee_m = transform[:3, :3] @ joints[5] + transform[:3, 3]
        distances = np.linalg.norm(self.reference.vessels_m[:, 0] - knee_m, axis=1)
        index = int(np.argmin(distances))
        self.knee_station_mm = float(template.s_mm[index])
        self.knee_distance_mm = float(1000. * distances[index])
        self.envelope = knee_envelope(template.s_mm, self.knee_station_mm)
        self.interpolation_checks = self.check_midpoints() if check_midpoints else []

    def bracket(self, angle):
        q = np.asarray(angle, dtype=float)
        if not np.all(np.isfinite(q) & (q >= self.angles[0]) & (q <= self.angles[-1])):
            raise ValueError("body angle outside the evaluated atlas range")
        lower = np.clip(np.searchsorted(self.angles, q, side="right") - 1, 0, len(self.angles) - 2)
        weight = (q - self.angles[lower]) / (self.angles[lower + 1] - self.angles[lower])
        return lower, weight

    def vessels(self, angle, *, physical=False):
        lower, weight = self.bracket(angle)
        sequence = self.worlds if physical else self.shapes
        a, b = sequence[lower:lower + 2]
        if weight == 0.:
            return VesselGeometry(a.vessels_m, a.offset_axes)
        if weight == 1.:
            return VesselGeometry(b.vessels_m, b.offset_axes)
        axes = rotation_between(frame_from_offset_axes(a.offset_axes),
                                frame_from_offset_axes(b.offset_axes), weight)
        return VesselGeometry(a.vessels_m + weight * (b.vessels_m - a.vessels_m), axes[:, 1:])

    def poses(self, stations, offsets, angles):
        stations, offsets, angles = np.broadcast_arrays(stations, offsets, angles)
        lower, weight = self.bracket(angles.ravel())
        station, offset = stations.ravel(), offsets.ravel()
        result = np.empty((len(station), 4, 4))
        for index in np.unique(lower):
            group = np.flatnonzero(lower == index)
            f = weight[group]
            if np.all(f == 1.):
                result[group] = self.shapes[index + 1].pose(station[group], offset[group])
                continue
            a = self.shapes[index].pose(station[group], offset[group])
            if np.any(f > 0.):
                b = self.shapes[index + 1].pose(station[group], offset[group])
                a[:, :3, 3] += f[:, None] * (b[:, :3, 3] - a[:, :3, 3])
                a[:, :3, :3] = rotation_between(a[:, :3, :3], b[:, :3, :3], f)
            result[group] = a
        return result.reshape(stations.shape + (4, 4))

    def check_midpoints(self):
        station, offset = np.meshgrid(self.template.s_mm[::4], [-22., -12., -2.])
        reports = []
        for angle in .5 * (self.angles[:-1] + self.angles[1:]):
            world = self.template.instantiate((0., 0.), angle)
            shape = self.template.with_spline(world)
            exact = shape.pose(station, offset)
            approx = self.poses(station, offset, angle)
            difference = Rotation.from_matrix(exact[..., :3, :3].reshape(-1, 3, 3)).inv() * (
                Rotation.from_matrix(approx[..., :3, :3].reshape(-1, 3, 3)))
            report = dict(
                knee_deg=float(angle),
                contact_mm=float(1000. * np.linalg.norm(exact[..., :3, 3] - approx[..., :3, 3], axis=-1).max()),
                rotation_deg=float(np.rad2deg(difference.magnitude()).max()),
                world_vessel_mm=float(1000. * np.linalg.norm(
                    world.vessels_m - self.vessels(angle, physical=True).vessels_m, axis=-1).max()),
                planning_vessel_mm=float(1000. * np.linalg.norm(
                    shape.vessels_m - self.vessels(angle).vessels_m, axis=-1).max()),
            )
            if max(report["contact_mm"], report["world_vessel_mm"], report["planning_vessel_mm"]) > .25:
                raise ValueError(f"interpolated anatomy exceeds the 0.25 mm midpoint tolerance: {report}")
            if report["rotation_deg"] > .25:
                raise ValueError(f"interpolated pose exceeds the 0.25 degree midpoint tolerance: {report}")
            reports.append(report)
        return reports


@dataclass(frozen=True)
class KneeScene:
    sensor: Scene
    slope_mm_per_deg: float


def central_interval(values, weights):
    order = np.argsort(values, kind="stable")
    cutoffs = np.searchsorted(np.cumsum(weights[order]), [.05, .95])
    return np.asarray(values)[order[cutoffs]]


class KneeScan(Scan):
    """Use the same sampled IK, ProDMP and sensor with time-indexed anatomy."""

    def __init__(self, atlas, capability, witness, regression, body_angles):
        self.atlas = atlas
        self.body_angles = np.array(body_angles, dtype=float, copy=True)
        if self.body_angles.shape != (TOTAL_STEPS + 1,):
            raise ValueError("an angle is required at every robot sample")
        self.body_angles.setflags(write=False)
        self.delta_angles = self.body_angles - REFERENCE_KNEE_DEG
        # Geometry at each exposure is an entire atlas at a single knee angle.
        planning = {q: atlas.vessels(q) for q in np.unique(self.body_angles)}
        physical = {q: atlas.vessels(q, physical=True) for q in planning}
        self.planning_geometry = [planning[q] for q in self.body_angles]
        self.physical_geometry = [physical[q] for q in self.body_angles]
        super().__init__(atlas.reference, capability, witness, regression)
        self.models[RESPONSE_ARM] = PoseResponseFilter(self.sensor, FRAME_DT_S)

    def poses_at(self, steps, offsets):
        return self.atlas.poses(self.path[steps], offsets, self.body_angles[steps])

    def project_at(self, step, pose, offsets, *, slopes=None):
        response = None if slopes is None else self.delta_angles[step] * np.asarray(slopes)
        return project(self.planning_geometry[step], pose, offsets,
                       response_mm=response, envelope=self.atlas.envelope)

    def project_hypotheses(self, step, pose, model):
        slopes = model.slopes_mm_per_deg if isinstance(model, PoseResponseFilter) else None
        return self.project_at(step, pose, model.offsets_mm, slopes=slopes)

    def project_estimate(self, step, pose, model, belief, frames=0):
        slopes = (model.slope_distribution(belief) @ model.slope_grid
                  if isinstance(model, PoseResponseFilter) else None)
        return self.project_at(step, pose, model.offset_mean(belief, frames), slopes=slopes)

    def observation(self, scene, frame, pose, z_mm):
        step = frame * FRAME_STRIDE
        tape = scene.sensor
        effective = tape.offset + [z_mm, 0.]
        response = scene.slope_mm_per_deg * self.delta_angles[step]
        geometry = self.physical_geometry[step]
        measured = project(geometry, pose, effective, response_mm=response, envelope=self.atlas.envelope)[0]
        physical = project(geometry, pose, effective - tape.bias,
                           response_mm=response, envelope=self.atlas.envelope)[0]
        return self.sensor.observe_means(tape, frame, measured, physical)

    def feature_at(self, step):
        return self.delta_angles[step] * knee_envelope(self.path[step], self.atlas.knee_station_mm)

    def offset_forecast(self, step, model, belief, frames):
        values = model.offsets_mm[:, 0]
        if isinstance(model, PoseResponseFilter):
            values = values + model.slopes_mm_per_deg * self.feature_at(step)
        weights = model.offset_distribution(belief, frames)
        return float(weights @ values), central_interval(values, weights)

    def frame_estimate(self, step, model, belief, arm):
        mean, interval = self.offset_forecast(step, model, belief, 0)
        result = dict(material_station_effective_lateral_mean_mm=mean,
                      material_station_effective_lateral_interval90_mm=interval.tolist())
        if isinstance(model, PoseResponseFilter):
            weights = model.slope_distribution(belief)
            result["slope_mean_mm_per_deg"] = float(weights @ model.slope_grid)
        return result
