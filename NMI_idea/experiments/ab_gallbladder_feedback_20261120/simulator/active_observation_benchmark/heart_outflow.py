"""Five cardiac structures, posed SMPL-X attachments and respiratory ray shadows."""

from __future__ import annotations

from copy import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from scipy.spatial.transform import Rotation


CONTINUOUS = Path(__file__).resolve().parents[1] / "outputs" / "thorax_continuous"
sys.path.insert(0, str(CONTINUOUS))
from model import Body, Reference, READOUT, SMPLX, SOURCE, TARGET_PHASE
from geometry import Anatomy, BEAM_ANGLE, DEPTH_MM
from generate import Bones, mesh_arrays


CLASSES = ("RA", "LA", "RV", "LV", "AoV")
SPINE_JOINTS = (3, 6, 9)
VALVE_AXES = np.array([9., 8.5, 2.3])
OUTLET_FRACTIONS = np.linspace(0., 1.22, 11)


class PosedReference(Reference):
    """Evaluate native SMPL-X spine correctives and skinning at fixed face IDs."""

    def __init__(self):
        super().__init__()
        with np.load(SOURCE, allow_pickle=False) as source:
            self.skin = source["skin_vertices"]
            self.skin_faces = source["skin_faces"]
            self.lungs = tuple(mesh_arrays(source, name) for name in ("Lung_L", "Lung_R"))
        with np.load(SMPLX, allow_pickle=False) as model:
            self.skin_weights = model["weights"].astype(float)
            self.joint_regressor = model["J_regressor"].astype(float)
            self.parents = model["kintree_table"][0].astype(int)
            self.parents[0] = -1
            columns = np.concatenate([np.arange(9 * (j - 1), 9 * j) for j in SPINE_JOINTS])
            self.pose_basis = model["posedirs"][..., columns].astype(float)

    def pose_rim(self, beta, spine_degrees):
        """Use the source triangle and barycentric weights for every skin knot."""
        if np.array_equal(spine_degrees, np.zeros(2)):
            return self.rim @ beta
        rest = self.skin + self.shape @ beta / 1000.
        joints = self.joint_regressor @ rest
        rotations = np.tile(np.eye(3), (len(joints), 1, 1))
        rotations[list(SPINE_JOINTS)] = Rotation.from_euler(
            "xz", np.asarray(spine_degrees) * np.array([[.2], [.35], [.45]]), degrees=True,
        ).as_matrix()
        corrective = (rotations[list(SPINE_JOINTS)] - np.eye(3)).ravel()
        local_skin = rest + self.pose_basis @ corrective
        world_rotation = np.empty_like(rotations)
        world_joint = np.empty_like(joints)
        for j, parent in enumerate(self.parents):
            if parent < 0:
                world_rotation[j], world_joint[j] = rotations[j], joints[j]
            else:
                world_rotation[j] = world_rotation[parent] @ rotations[j]
                world_joint[j] = world_joint[parent] + world_rotation[parent] @ (joints[j] - joints[parent])
        offset = world_joint - np.einsum("jkl,jl->jk", world_rotation, joints)
        blended = np.einsum("vj,jkl->vkl", self.skin_weights, world_rotation)
        skin = np.einsum("vkl,vl->vk", blended, local_skin) + self.skin_weights @ offset
        faces = self.skin_faces[np.maximum(self.atlas.face, 0)]
        rim = np.einsum("hmv,hmvc->hmc", np.nan_to_num(self.atlas.bary), (skin - self.skin)[faces]) * 1000.
        for i, ring in enumerate(rim):
            real = self.atlas.face[i] >= 0
            for axis in range(3):
                ring[~real, axis] = np.interp(self.atlas.theta[~real], self.atlas.theta[real],
                                             ring[real, axis], period=2. * np.pi)
        return rim


class OutflowBody(Body):
    def __init__(self, reference, beta, spine_degrees):
        posed = copy(reference)
        posed.rim = reference.pose_rim(beta, spine_degrees)[..., None]
        super().__init__(posed, np.ones(1))
        self.beta_delta = np.asarray(beta)
        self.spine_degrees = np.asarray(spine_degrees)
        self.pose_change_rms_mm = float(np.sqrt(np.mean(np.sum((posed.rim[..., 0] - reference.rim @ beta) ** 2, axis=-1))))

        # The registered source has an outer heart surface. Attach a synthetic
        # valve region anterior and medial to the LV basal outflow, keeping
        # the two sampling endpoints in the same transported material atlas.
        anatomy = reference.anatomy
        long = anatomy.rotation[3, :, 1]
        base = anatomy.rest[3] + .60 * anatomy.rest_axes[3, 1] * long + [-2., 0., 5.]
        valve = anatomy.rest[3] + [-20., 39., 13.]
        base, valve = self.warp(np.vstack((base, valve)))
        self.outlet_base = base - self.rest[3]
        self.valve_offset = valve - self.rest[3]
        flow = valve - base
        flow /= np.linalg.norm(flow)
        right = np.cross([0., 0., 1.], flow)
        right /= np.linalg.norm(right)
        valve_frame = np.column_stack((right, np.cross(flow, right), flow))
        self.five = SimpleNamespace(rotation=np.concatenate((self.rotation, valve_frame[None])))
        self.lung_geometry = []
        for pts, faces in reference.lungs:
            points = self.warp(pts)
            self.lung_geometry.append((Bones(points, faces, np.zeros(len(faces), dtype=np.int16)), points.mean(axis=0)))
        self.lung_cache = {}

    def lung_distances(self, view, inflation, *, width=None):
        amount = float(inflation)
        key = (*view.chart, amount, width)
        if key in self.lung_cache:
            return self.lung_cache[key]
        ray = np.cos(READOUT[:, None]) * view.forward + np.sin(READOUT[:, None]) * view.right
        distance = np.full(len(ray), np.inf)
        scale = 1. + amount * np.array([.095, .070, .055])
        shift = np.array([0., -12. * amount, 3. * amount])
        for side, (mesh, center) in enumerate(self.lung_geometry):
            if width is not None:
                # The source's aerated medial boundary overcovers the synthetic
                # four-chamber anatomy. Recess that surface from a fixed lateral
                # anchor. This is a simulation calibration, not respiratory travel.
                center = center.copy()
                center[0] = mesh.mesh.bounds[1 - side]
                scale = (1. + amount * np.array([.12, .06, .05])) * [width, 1., 1.]
                shift = np.array([0., -9. * amount, 2.5 * amount])
            origin = center + (view.origin - center - shift) / scale
            directions = ray / scale
            depth, _ = mesh.hits(origin, directions)
            distance = np.minimum(distance, depth)
        if len(self.lung_cache) >= 8192:
            self.lung_cache.clear()
        self.lung_cache[key] = distance
        return distance

    def acoustic_view(self, view, rib, inflation, *, lung_width=None):
        bone = self.with_ribs(view, rib)
        if inflation is None:
            return bone
        lung = self.lung_distances(view, inflation, width=lung_width)
        return replace(bone, bone_distance=np.minimum(bone.bone_distance, lung))

    def five_trajectories(self, parameters, phase):
        centers, axes, _ = self.trajectories(parameters, phase)
        size = np.exp(np.atleast_2d(parameters)[:, None, 3, None])
        valve = centers[..., 3, :] + size * self.valve_offset
        valve_axes = np.broadcast_to(size * VALVE_AXES, valve.shape)
        return np.concatenate((centers, valve[..., None, :]), axis=2), np.concatenate((axes, valve_axes[..., None, :]), axis=2)

    def outflow(self, view, parameters, phase):
        """Analytic ellipsoid sections and visible LV-to-valve corridor samples."""
        centers, axes = self.five_trajectories(parameters, phase)
        features, central, length, apex, apical, visible = Anatomy.predict(self.five, view, centers, axes)
        base, valve = self.outlet_endpoints(centers, parameters)
        corridor = base[..., None, :] + OUTLET_FRACTIONS[:, None] * (valve - base)[..., None, :]
        relative = corridor - view.origin
        x, depth, normal = relative @ view.right, relative @ view.forward, relative @ view.normal
        in_sector = (depth >= 4.) & (np.hypot(x, depth) <= DEPTH_MM) & (np.abs(x) <= depth * np.tan(BEAM_ANGLE))
        intact_corridor = (np.abs(normal) <= 6.) & in_sector & ~view.beam_blocked(x, depth)
        return features, central, length, apex, apical, visible, intact_corridor

    def outlet_endpoints(self, centers, parameters):
        size = np.exp(np.atleast_2d(parameters)[:, None, 3, None])
        return centers[..., 3, :] + size * self.outlet_base, centers[..., 4, :]

    def masks(self, view, parameters, phase):
        """Return the clean and acoustically occluded five-class mask of one frame."""
        centers, axes = self.five_trajectories(parameters, phase)
        return Anatomy.raster(self.five, view, centers[0, 0], axes[0, 0])

    def mask_channels(self, view, parameters, phase, *, size=128):
        """Four chamber channels and the aortic annulus region."""
        centers, axes = self.five_trajectories(parameters, phase)
        x, y, sector, shadow, _, chambers = Anatomy.raster(
            self, view, centers[0, 0, :4], axes[0, 0, :4], size=size,
        )
        pixels = view.origin + x[..., None] * view.right + y[..., None] * view.forward
        annulus = (pixels - centers[0, 0, 4]) @ self.five.rotation[4] / axes[0, 0, 4]
        available = sector & ~shadow
        rings = np.sum(annulus**2, axis=-1) <= 1.
        return np.concatenate((chambers[None] == np.arange(1, 5)[:, None, None],
                               (rings & available)[None]))


def clip_geometry(summary):
    """Per-frame eligibility. The thresholds define the simulated task."""
    features, central, length, apex, apical, visible, corridor = summary
    a4c = ((central[..., :4] >= .40).all(axis=-1) & (length[..., 2:4] >= .84).all(axis=-1)
           & (apex <= 9.) & apical & (visible[..., :4] >= .80).all(axis=-1)
           & (central[..., 4] <= .16))
    a5c = ((central[..., 3] >= .35) & (length[..., 3] >= .82) & (apex <= 10.) & apical
           & (central[..., 1] >= .23) & (visible[..., 3] >= .72) & (visible[..., 4] >= .70)
           & (central[..., 4] >= .32) & (features[..., 4, 0] >= 30.) & corridor.all(axis=-1))
    return np.stack((a4c, a5c), axis=-1)


def aim(body, xy, parameters, view_kind):
    """Public initial tilt for a contact. Final search can move and refine it."""
    level = body.view(np.r_[xy, 0., 0.])
    centers, axes = body.five_trajectories(parameters, TARGET_PHASE)
    center = centers[0].mean(axis=0)
    apex = center[3] - .9 * axes[0, 0, 3, 1] * body.rotation[3, :, 1]
    if view_kind == 0:
        targets, weights = np.vstack((center[:4], apex)), np.array([1., 1., 1., 1., 1.35])
    else:
        targets, weights = np.vstack((center[[1, 3, 4]], apex)), np.array([.45, 1., 2., 1.55])
    rays = targets - level.origin
    _, vectors = np.linalg.eigh((rays.T * weights) @ rays)
    normal = vectors[:, 0]
    if normal @ level.normal < 0.:
        normal = -normal
    roll = np.arctan2(-normal @ level.right, normal @ level.normal)
    after_roll = -np.sin(roll) * level.right + np.cos(roll) * level.normal
    rock = np.arctan2(-normal @ level.forward, normal @ after_roll)
    return np.r_[xy, np.rad2deg([roll, rock])]
