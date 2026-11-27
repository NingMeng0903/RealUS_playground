"""Local oblique centerline renderer and simulator-only grading for v16."""

import numpy as np
from scipy.spatial.transform import Rotation

from segment_v16 import ANGULAR_RATE_DEG_S, MAX_ATTITUDE_DEG, PULSES, base_frame, unit, validate_segment


class ObliqueScene:
    """Same fixed v15 anatomy and noise tape, with an actual 3-D slice plane."""

    def __init__(self, surface, layout, station):
        self.surface, self.layout, self.station = surface, layout, station
        left, right = max(0, station - 14), min(len(surface.y) - 1, station + 14)
        self.stations = np.arange(4 * left, 4 * right + 1) * .25
        self.line = np.asarray([self.centers(s) for s in self.stations])

    def centers(self, station):
        points = self.layout.at(station)
        y = np.interp(station, np.arange(len(self.surface.y)), self.surface.y)
        return np.column_stack((points[:, 0], np.full(len(points), y), points[:, 1]))

    def intersect(self, origin, frame):
        normal = frame[:, 1]
        plane = (self.line - origin) @ normal
        hit_rows = (plane[:-1] * plane[1:] < 0.) | (np.abs(plane[:-1]) <= 1.e-8)
        raw = np.full((self.line.shape[1], 3), np.nan)
        for role in range(len(raw)):
            cuts = np.flatnonzero(hit_rows[:, role])
            if not len(cuts):
                continue
            k = int(cuts[np.argmin(np.abs(self.stations[cuts] - self.station))])
            lo, hi = self.stations[k:k + 2]
            val = plane[k, role]
            if abs(val) <= 1.e-8:
                raw[role] = self.line[k, role]
                continue
            for _ in range(16):
                middle = (lo + hi) / 2
                point = self.centers(middle)[role]
                f = (point - origin) @ normal
                if f * val > 0:
                    lo = middle
                else:
                    hi = middle
            raw[role] = self.centers((lo + hi) / 2)[role]
        return (raw - origin) @ frame[:, (0, 2)]

    def rays(self, origin, frame, means):
        surface = self.surface
        valid = np.isfinite(means).all(axis=1)
        gate = np.ones(len(means))
        offsets = np.clip(means[valid, 0], -25., 25.)
        depths = np.clip(means[valid, 1], 3., 80.)
        points = (origin + offsets[:, None, None] * frame[:, 0]
                  + depths[:, None, None] * np.linspace(0., 1., 22)[None, :, None] * frame[:, 2])
        flat = points.reshape(-1, 3)
        s = np.interp(flat[:, 1], surface.y[::-1], np.arange(len(surface.y))[::-1])
        bins = np.rint(s * 4).astype(int)
        attenuation = np.empty(len(flat))
        for k in np.unique(bins):
            mask = bins == k
            attenuation[mask] = surface.occupancy(k / 4., flat[mask][:, (0, 2)])
        gate[valid] = attenuation.reshape(len(offsets), 22).max(axis=1)
        return gate

    def true_rule(self, origin, sweep):
        centers = self.centers(self.station)
        target = centers[0]
        tangents = np.asarray([target - self.centers(self.station + .125)[0],
                               self.centers(self.station - .125)[0] - target])
        tangents = tangents[np.isfinite(tangents).all(axis=1)]
        if not np.isfinite(target).all() or not len(tangents):
            return {"pass": False, "truth_supported": False}
        return {"truth_supported": True} | validate_segment(sweep, origin, target, unit(tangents))


def settle_rotations(sweep, clock):
    start = base_frame(sweep.sector)
    displacement = Rotation.from_matrix(sweep.frame(0.) @ start.T).as_rotvec()
    ticks = int(np.ceil(np.rad2deg(np.linalg.norm(displacement)) / (ANGULAR_RATE_DEG_S * clock)))
    if not ticks:
        return np.empty((0, 3, 3))
    return Rotation.from_rotvec(np.arange(1, ticks + 1)[:, None] * displacement / ticks).as_matrix() @ start


def public_delivery(frames):
    """Frozen public centerline-echo readout. Identity labels enter only below."""
    rows = [(i, j, float(x), float(depth), q, point)
            for i, (xy, _, q, world) in enumerate(frames) if q >= .35
            for j, ((x, depth), point) in enumerate(zip(xy, world, strict=True)) if abs(x) <= 9.]
    if not rows:
        return None
    world = np.asarray([row[-1] for row in rows])
    ticks = np.asarray([row[0] for row in rows])
    weights = np.exp(-.5 * (np.linalg.norm(world[:, None] - world[None], axis=2) / 3.5) ** 2)
    weights *= ticks[:, None] != ticks[None]
    scores = np.asarray([.45 * row[4] - .50 * abs(row[2]) / 9. for row in rows])
    scores += .55 * np.sum(weights, axis=1) / (len(frames) - 1)
    selected = int(np.argmax(scores))
    return rows[selected][:2] if weights[selected].max() >= .35 else None


def acquire(scene, scan, sweep, start_tick, approach, travel, clock):
    target = (scene.station, sweep.sector)
    origin = scene.surface.world(*target)
    if (scan.node.station, scan.node.sector) != target or not scene.surface.legal(*target):
        raise RuntimeError("Oblique action did not reach its checked contact")
    settle = settle_rotations(sweep, clock)
    if start_tick <= scan.node.tick + len(settle) or start_tick + PULSES >= len(scan.tape.tape.normal):
        raise RuntimeError("Acoustic replay clock does not cover the settled trajectory")
    frames = []
    previous = base_frame(sweep.sector)
    for frame in settle:
        step = Rotation.from_matrix(frame @ previous.T).magnitude()
        if np.rad2deg(step) > ANGULAR_RATE_DEG_S * clock + 1.e-6:
            raise RuntimeError("Probe lean exceeded the kinematic rotation rate")
        previous = frame
    neutral = base_frame(sweep.sector)
    for tick, frame in enumerate(sweep.frame(np.linspace(0., 1., PULSES)), start_tick):
        angle = Rotation.from_matrix(frame @ neutral.T).magnitude()
        if np.rad2deg(angle) > MAX_ATTITUDE_DEG + 1.e-6:
            raise RuntimeError("Oblique probe attitude departed from the chart envelope")
        step = Rotation.from_matrix(frame @ previous.T).magnitude()
        if np.rad2deg(step) > ANGULAR_RATE_DEG_S * clock + 1.e-6:
            raise RuntimeError("Swept acquisition exceeded the kinematic rotation rate")
        previous = frame
        mean = scene.intersect(origin, frame)
        gate = scene.rays(origin, frame, mean)
        coupling = (1. - .12 * abs(np.sin(sweep.sector * np.pi / 12.))) * np.cos(angle)
        xy, labels, q = scan.tape.observe(mean, gate, tick, station=scene.station, sector=sweep.sector,
                                         contact_quality=coupling)
        world = origin + xy @ frame[:, (0, 2)].T
        frames.append((xy, labels, q, world))
    selection = public_delivery(frames)
    label = None if selection is None else int(frames[selection[0]][1][selection[1]])
    truth = scene.true_rule(origin, sweep)
    truth.update({"delivered_role": label, "target_delivered": label == 0,
                  "target_seen": any(np.any(frame[1] == 0) for frame in frames),
                  "fresh_target_pulses": int(sum(np.any(frame[1] == 0) for frame in frames)),
                  "joint": bool(label == 0 and truth["pass"]),
                  "selected_pulse": None if selection is None else [start_tick + selection[0], selection[1]],
                  "origin_xyz": origin.tolist(), "approach": travel,
                  "leaving_pose": [*target, int(scan.node.tick)], "settle_ticks": len(settle),
                  "sync_wait_ticks": start_tick - int(scan.node.tick) - len(settle) - 1,
                  "pulse_ticks": [start_tick, start_tick + PULSES - 1],
                  "echo_labels": [frame[1].tolist() for frame in frames],
                  "program_ticks": approach.ticks + len(settle) + PULSES})
    return truth
