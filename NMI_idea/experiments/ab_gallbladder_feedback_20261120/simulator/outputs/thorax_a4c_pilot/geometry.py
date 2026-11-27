"""Phase-matched ellipsoid slices with bone shadows on the registered chest."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


PREVIEW = Path(__file__).resolve().parents[1] / "thorax_preview"
sys.path.insert(0, str(PREVIEW))
from generate import BEAM_ANGLE, Bones, CHAMBERS, SOURCE, ThoraxAtlas, heart_cases
from gallbladder_runtime import ATLAS_ARCHIVE


N_PHASE = 24
PHASE = np.arange(N_PHASE) / N_PHASE
REFERENCE = (5, 12)  # Near filling and near least ventricular volume.
DEPTH_MM = 194.
BEAMS = np.linspace(-BEAM_ANGLE, BEAM_ANGLE, 181)
TAKE = 40
POINT_NUMBER = np.arange(TAKE)
DISK_ANGLE = POINT_NUMBER * (np.pi * (3. - np.sqrt(5.)))
DISK_RADIUS = np.sqrt((POINT_NUMBER + .5) / TAKE)
DISK_X = DISK_RADIUS * np.cos(DISK_ANGLE)
DISK_Y = DISK_RADIUS * np.sin(DISK_ANGLE)


@dataclass(frozen=True)
class View:
    chart: np.ndarray  # superior-inferior mm, theta degrees, roll and rock degrees
    address: np.ndarray  # h, sin(theta), cos(theta)
    origin: np.ndarray
    forward: np.ndarray
    right: np.ndarray
    normal: np.ndarray
    bone_distance: np.ndarray

    def beam_blocked(self, x, y):
        bearing = np.arctan2(x, y)
        index = np.rint((bearing / BEAM_ANGLE + 1.) * (len(BEAMS) - 1) / 2.).astype(int)
        return np.hypot(x, y) >= self.bone_distance[index.clip(0, len(BEAMS) - 1)]


class Anatomy:
    def __init__(self):
        with np.load(SOURCE, allow_pickle=False) as data:
            self.atlas = ThoraxAtlas(data["skin_vertices"] * 1000., data["skin_faces"])
        with np.load(ATLAS_ARCHIVE, allow_pickle=False) as data:
            self.bones = Bones(data["bone_vertices_mm"], data["bone_faces"], data["bone_face_id"])
            heart_points = data["heart_reference_vertices_mm"]
        t, bpm, _, phase, volume, xyz, _, rotation, rest, rest_axes = heart_cases(heart_points)
        cycle = t * bpm[0] / 60. < 1.
        self.phase = np.r_[phase[0, cycle], 1.]
        self.volume = np.vstack((volume[0, cycle], volume[0, 0]))
        self.motion = np.vstack((xyz[0, cycle, 0], xyz[0, 0, 0])) - rest[0]
        self.rest, self.rest_axes, self.rotation = rest, rest_axes, rotation
        self.shortening = np.array([[.38, .24, .38], [.38, .24, .38], [.39, .22, .39], [.39, .22, .39]])
        self.midpoint = np.mean(rest, axis=0)
        self.plane_normal = np.linalg.svd(rest - self.midpoint, full_matrices=False)[2][-1]

    def view(self, chart):
        y, angle, roll, rock = chart
        theta, roll, rock = np.deg2rad([angle, roll, rock])
        h = (y - self.atlas.y0) / (self.atlas.y1 - self.atlas.y0)
        origin = self.atlas.to_world(h, theta, 1.)
        forward = self.midpoint - origin
        forward /= np.linalg.norm(forward)
        right = np.cross(self.plane_normal, forward)
        right /= np.linalg.norm(right)
        if right[0] < 0.:
            right = -right
        right = np.cos(roll) * right + np.sin(roll) * np.cross(forward, right)
        forward = np.cos(rock) * forward + np.sin(rock) * np.cross(forward, right)
        normal = np.cross(forward, right)
        beam = np.cos(BEAMS[:, None]) * forward + np.sin(BEAMS[:, None]) * right
        distance, _ = self.bones.hits(origin, beam)
        return View(np.asarray(chart), np.array([h, np.sin(theta), np.cos(theta)]),
                    origin, forward, right, normal, distance)

    def trajectories(self, parameters, phase=PHASE):
        # Per-subject parameters are lateral/depth displacement, contraction
        # gain, overall size, superior offset, then four small phase offsets.
        parameters = np.asarray(parameters)
        ph = (np.asarray(phase)[None, :, None] + parameters[:, None, 5:9]) % 1.
        volume = np.stack([np.interp(ph[..., i], self.phase, self.volume[:, i]) for i in range(4)], axis=-1)
        volume = volume ** parameters[:, None, 2, None]
        semi = self.rest_axes * parameters[:, None, 3, None, None] * volume[..., None] ** self.shortening
        offset = parameters[:, None, None, [0, 4, 1]]
        motion = np.stack([np.interp(phase, self.phase, self.motion[:, i]) for i in range(3)], axis=-1)
        centers = self.rest[None, None] + offset + parameters[:, None, 2, None, None] * motion[None, :, None, :]
        return centers, semi, volume

    def predict(self, view, centers, axes):
        """Analytic planar ellipse plus quadrature for partial visibility."""
        basis = np.column_stack((view.right, view.forward))
        r_to_local = self.rotation.swapaxes(-1, -2)
        projection = (r_to_local @ basis) / axes[..., None]
        delta = np.einsum("...ci,cij->...cj", view.origin - centers, self.rotation) / axes
        m00, m11 = np.sum(projection ** 2, axis=-2).transpose(3, 0, 1, 2)
        m01 = np.sum(projection[..., 0] * projection[..., 1], axis=-1)
        q0, q1 = np.sum(projection * delta[..., None], axis=-2).transpose(3, 0, 1, 2)
        det = m00 * m11 - m01 ** 2
        cx, cy = (m01 * q1 - m11 * q0) / det, (m01 * q0 - m00 * q1) / det
        w = 1. - np.sum(delta ** 2, axis=-1) - cx * q0 - cy * q1
        w_pos = np.maximum(w, 0.)
        root_x = np.sqrt(np.maximum(w_pos * m11 / det, 1.e-12))
        root_y = np.sqrt(w_pos / m11)
        root_xy = -(m01 / m11) * root_x
        x = cx[..., None] + root_x[..., None] * DISK_X
        y = cy[..., None] + root_xy[..., None] * DISK_X + root_y[..., None] * DISK_Y
        inside = (w[..., None] > 0.) & (y >= 4.) & (np.hypot(x, y) <= DEPTH_MM) & (np.abs(x) <= y * np.tan(BEAM_ANGLE))
        seen = inside & ~view.beam_blocked(x, y)
        n = seen.sum(axis=-1)
        safe = np.maximum(n, 1)
        visible_center_x = (x * seen).sum(axis=-1) / safe
        visible_center_y = (y * seen).sum(axis=-1) / safe
        ax = (x - visible_center_x[..., None]) * seen
        ay = (y - visible_center_y[..., None]) * seen
        xvar = (ax ** 2).sum(axis=-1) / safe
        yvar = (ay ** 2).sum(axis=-1) / safe
        covar = (ax * ay).sum(axis=-1) / safe
        shape = np.sqrt(np.maximum(.5 * (xvar + yvar + np.hypot(xvar - yvar, 2. * covar)), 0.)) * 4.
        area = np.pi * w_pos / np.sqrt(det) * n / TAKE
        features = np.stack((area, visible_center_x, visible_center_y, shape), axis=-1)

        axis = self.rotation[:, :, 1]
        t0, t1 = (axis @ basis).T
        variance = w_pos * (m11 * t0 ** 2 - 2. * m01 * t0 * t1 + m00 * t1 ** 2) / det
        long_fraction = np.sqrt(np.maximum(variance, 0.)) / axes[..., 1]
        apex = centers[..., 3, :] - axes[..., 3, 1, None] * axis[3]
        apex_distance = np.abs((apex - view.origin) @ view.normal)
        is_apical = ((view.forward @ axis[3]) > .43) & (view.origin[1] < apex[..., 1] + 3.)
        intact = n / TAKE
        return features, w_pos * intact, long_fraction, apex_distance, is_apical, intact

    def quality(self, view, centers, axes):
        features, fraction, long, apex, apical, intact = self.predict(view, centers, axes)
        select = np.asarray(REFERENCE)
        coverage = fraction[:, select].min(axis=(1, 2))
        ventricular = long[:, select, 2:].min(axis=(1, 2))
        apex = apex[:, select].max(axis=1)
        approach = apical[:, select].all(axis=1)
        preserved = intact[:, select].min(axis=(1, 2))
        correct = (coverage >= .40) & (ventricular >= .84) & (apex <= 9.) & approach & (preserved >= .80)
        quality = (.56 * coverage + .27 * ventricular + .17 * np.exp(-apex / 8.)) * np.where(approach, 1., .4)
        return features, correct, np.stack((coverage, ventricular, apex, approach.astype(float), quality, preserved), axis=-1)

    def raster(self, view, centers, axes, size=220):
        x, y = np.meshgrid(np.linspace(-98., 98., size), np.linspace(3., DEPTH_MM, size))
        sector = (np.abs(x) <= y * np.tan(BEAM_ANGLE)) & (np.hypot(x, y) <= DEPTH_MM)
        shadow = sector & view.beam_blocked(x, y)
        pixels = view.origin + x[..., None] * view.right + y[..., None] * view.forward
        p = np.einsum("...ci,cij->...cj", pixels[..., None, :] - centers, self.rotation)
        radius = np.linalg.norm(p / axes, axis=-1)
        identity = radius.argmin(axis=-1) + 1
        clean = np.where(sector & (radius.min(axis=-1) < 1.), identity, 0).astype(np.uint8)
        return x, y, sector, shadow, clean, np.where(shadow, 0, clean).astype(np.uint8)
