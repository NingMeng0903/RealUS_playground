"""Build a torso material atlas and three synthetic motion and ultrasound previews."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import cg
from scipy.spatial.transform import Rotation
import trimesh
from vtkmodules.vtkCommonCore import reference
from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator
import pyvista as pv


OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parents[1]))
from gallbladder_runtime import ANATOMY_ARCHIVE, REPO_ROOT

ATLAS = REPO_ROOT / "NMI_idea/experiments/ab_streaming_prodmp_two_vessel_20261113"
sys.path.insert(0, str(ATLAS))
from spline_atlas import PeriodicCubic, RadialDepth

SOURCE = ANATOMY_ARCHIVE
FPS = 30
SAMPLES = 132
CHAMBERS = ("RA", "LA", "RV", "LV")
BEAM_ANGLE = np.deg2rad(39.)
BEAM_RANGE = 195.
X_HALF = 83.
DEPTH = 166.


def cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def ray_intersections(origin, directions, segments):
    a = segments[:, 0][:, [0, 2]] - origin
    edge = np.diff(segments[:, :, [0, 2]], axis=1)[:, 0]
    denominator = cross2(directions[:, None, :], edge)
    parallel = np.abs(denominator) < 1.e-10
    denominator = np.where(parallel, 1., denominator)
    distance = cross2(a, edge) / denominator
    fraction = cross2(a, directions[:, None, :]) / denominator
    distance[parallel | (distance <= 0) | (fraction < -1.e-9) | (fraction > 1 + 1.e-9)] = np.inf
    segment_index = np.argmin(distance, axis=1)
    return distance[np.arange(len(directions)), segment_index], segment_index


def graph_depth(points):
    index = np.arange(np.prod(points.shape[:-1])).reshape(points.shape[:-1])
    src, dst = [], []
    for axis in (0, 1):
        lower = [slice(None)] * 3
        upper = lower.copy()
        lower[axis], upper[axis] = slice(None, -1), slice(1, None)
        src.append(index[tuple(lower)].ravel())
        dst.append(index[tuple(upper)].ravel())
    src.append(index.ravel())
    dst.append(np.roll(index, -1, axis=2).ravel())
    i, j = np.concatenate(src), np.concatenate(dst)
    xyz = points.reshape(-1, 3)
    weights = 1. / np.sum((xyz[i] - xyz[j]) ** 2, axis=1)
    adjacency = coo_matrix((np.tile(weights, 2), (np.r_[i, j], np.r_[j, i])),
                           shape=(len(xyz), len(xyz))).tocsr()
    diagonal = np.asarray(adjacency.sum(axis=1)).ravel()
    laplacian = diags(diagonal, format="csr") - adjacency
    inner, outer = index[:, 0].ravel(), index[:, -1].ravel()
    free = index[:, 1:-1].ravel()
    rhs = -np.asarray(laplacian[free][:, inner].sum(axis=1)).ravel()
    depth = np.zeros(len(xyz))
    depth[inner] = 1.
    depth[free], info = cg(laplacian[free][:, free], rhs, rtol=1.e-11,
                          M=diags(1. / diagonal[free]), maxiter=2500)
    if info:
        raise RuntimeError(f"harmonic solve failed ({info=})")
    residual = np.max(np.abs((laplacian @ depth)[free]) / diagonal[free])
    return depth.reshape(index.shape), float(residual)


class ThoraxAtlas:
    def __init__(self, vertices, faces):
        # Source coordinates are millimeters, with +x left, +y up, +z anterior.
        self.y0, self.y1 = -280., 94.
        self.h = np.linspace(0., 1., 67)
        self.theta = np.linspace(0., 2. * np.pi, 144, endpoint=False)
        self.rho = np.linspace(.42, 1., 11)
        y_levels = self.y0 + (self.y1 - self.y0) * self.h
        # The T-pose arms remain in the source mesh. Cap the two axillary cuts
        # outside the anterior scan patch and flag every resulting artificial knot.
        blend = np.clip((y_levels + 46.) / 140., 0., 1.)
        cap_width = 184. - 33. * (6 * blend ** 5 - 15 * blend ** 4 + 10 * blend ** 3)
        directions = np.column_stack((np.sin(self.theta), np.cos(self.theta)))
        shape = len(self.h), len(self.theta)
        self.face = np.full(shape, -1, dtype=np.int32)
        self.bary = np.full(shape + (3,), np.nan)
        self.skin = np.empty(shape + (3,))
        centers = np.zeros((len(self.h), 3))
        radii = np.empty(shape)
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        for i, (y, cap) in enumerate(zip(y_levels, cap_width)):
            segments, incident = trimesh.intersections.mesh_plane(
                mesh, [0., 1., 0.], [0., y, 0.], return_faces=True)
            front_back, _ = ray_intersections([0., 20.], np.array([[0., 1.], [0., -1.]]), segments)
            center_z = 20. + (front_back[0] - front_back[1]) / 2.
            centers[i] = [0., y, center_z]
            distance, edge = ray_intersections([0., center_z], directions, segments)
            max_distance = cap / np.maximum(np.abs(directions[:, 0]), 1.e-12)
            actual = distance < max_distance
            radii[i] = np.minimum(distance, max_distance)
            self.skin[i] = centers[i] + radii[i, :, None] * np.column_stack(
                (directions[:, 0], np.zeros(len(directions)), directions[:, 1]))
            self.face[i, actual] = incident[edge[actual]]
            source_triangles = vertices[faces[self.face[i, actual]]]
            self.bary[i, actual] = trimesh.triangles.points_to_barycentric(source_triangles, self.skin[i, actual])
        if not np.isfinite(radii).all() or np.min(radii) <= 0.:
            raise ValueError("invalid torso ray boundary")
        self.centers = centers
        self.core = CubicSpline(self.h, centers, axis=0, extrapolate=False)
        self.core_scale = float(.65 * radii.min())
        self.radius = PeriodicCubic(self.h, self.theta, np.log(radii - self.core_scale)[..., None])
        self.grid = self.to_world(self.h[:, None, None], self.theta[None, None, :], self.rho[None, :, None])
        self.depth_grid, self.graph_residual = graph_depth(self.grid)
        self.depth = RadialDepth(self.h, self.theta, self.rho, self.depth_grid)

    def to_world(self, h, theta, rho):
        h, theta, rho = np.broadcast_arrays(h, theta, rho)
        radius = self.core_scale * rho + np.exp(self.radius.evaluate(h, theta)[..., 0]) * rho ** 3
        xyz = self.core(h)
        xyz[..., 0] += radius * np.sin(theta)
        xyz[..., 1] = self.y0 + (self.y1 - self.y0) * h
        xyz[..., 2] += radius * np.cos(theta)
        return xyz

    def to_disk(self, xyz):
        h = (xyz[..., 1] - self.y0) / (self.y1 - self.y0)
        offset = xyz - self.core(h)
        theta = np.mod(np.arctan2(offset[..., 0], offset[..., 2]), 2. * np.pi)
        r = np.linalg.norm(offset[..., [0, 2]], axis=-1)
        growth = np.exp(self.radius.evaluate(h, theta)[..., 0])
        lower = np.zeros_like(h)
        upper = r / self.core_scale
        for _ in range(40):
            middle = (lower + upper) / 2.
            inside = self.core_scale * middle + growth * middle ** 3 < r
            lower, upper = np.where(inside, middle, lower), np.where(inside, upper, middle)
        rho = (lower + upper) / 2.
        return h, theta, rho

    def addresses(self, xyz):
        h, theta, rho = self.to_disk(xyz)
        d = np.full(h.shape, np.nan)
        shell = (rho >= self.rho[0]) & (rho <= 1.)
        d[shell] = self.depth.evaluate(h[shell], theta[shell], rho[shell])
        return np.stack((h, rho * np.sin(theta), rho * np.cos(theta)), axis=-1), np.stack((theta, h, d), axis=-1)


def mesh_arrays(source, name):
    i, = np.where(source["mesh_names"] == name)[0]
    start, end = source["mesh_ranges"][i]
    faces = source["faces"]
    matches = (faces[:, 0] >= start) & (faces[:, 0] < end)
    selected = faces[matches] - start
    if not selected.size or selected.min() < 0 or selected.max() >= end - start:
        raise ValueError(f"invalid topology for {name}")
    return source["candidate_vertices"][start:end].astype(float) * 1000., selected


def polydata(points, faces):
    return pv.PolyData(points, np.column_stack((np.full(len(faces), 3), faces)).ravel())


def superior_heights(mesh, columns):
    locator = vtkStaticCellLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()
    lower, upper = mesh.bounds[2:4]
    height = np.full(len(columns), np.nan)
    t = reference(0.)
    sub_id, cell_id = reference(0), reference(0)
    point, coordinates = [0.] * 3, [0.] * 3
    for i, (x, z) in enumerate(columns):
        if locator.IntersectWithLine((x, upper + 1., z), (x, lower - 1., z), 1.e-6,
                                     t, point, coordinates, sub_id, cell_id):
            height[i] = point[1]
    return height


def place_liver_and_gallbladder(liver, liver_faces, gall, gall_faces, diaphragm, diaphragm_faces, bones):
    # The liver and gallbladder are translated together from their registered
    # pose, preserving the gallbladder's position relative to the liver wall.
    shift = np.array([0., 5., 0.])
    original = polydata(liver, liver_faces)
    moved = polydata(liver + shift, liver_faces)
    dome = polydata(diaphragm, diaphragm_faces)
    x, z = np.meshgrid(np.arange(np.floor(liver[:, 0].min()), np.ceil(liver[:, 0].max()) + .5, .5),
                       np.arange(np.floor(liver[:, 2].min()), np.ceil(liver[:, 2].max()) + .5, .5))
    columns = np.column_stack((x.ravel(), z.ravel()))
    gap = superior_heights(dome, columns) - superior_heights(moved, columns)
    minimum = float(np.nanmin(gap))
    liver_contact = moved.collision(bones)[1]
    gall_contact = polydata(gall + shift, gall_faces).collision(bones)[1]
    if minimum <= 0. or liver_contact or gall_contact:
        raise RuntimeError("the revised liver and gallbladder must fit below the sampled dome and inside the ribs")
    placement = {
        "liver_gallbladder_shift_mm": shift.tolist(),
        "diaphragm_superior_envelope_columns": int(np.isfinite(gap).sum()),
        "liver_min_gap_to_diaphragm_superior_envelope_mm": minimum,
        "liver_bone_mesh_contacts_before": original.collision(bones)[1],
        "liver_bone_mesh_contacts_after": liver_contact,
        "gallbladder_bone_mesh_contacts_after": gall_contact,
        "liver_thick_diaphragm_mesh_contacts_before": original.collision(dome)[1],
        "liver_thick_diaphragm_mesh_contacts_after": moved.collision(dome)[1],
    }
    return liver + shift, gall + shift, shift, placement


def bone_mesh(source):
    names = [f"Rib_{i}{side}" for i in range(1, 13) for side in ("L", "R")] + ["Sternum"]
    vertex_groups, face_groups, bone_id = [], [], []
    count = 0
    for i, name in enumerate(names):
        points, faces = mesh_arrays(source, name)
        vertex_groups.append(points)
        face_groups.append(faces + count)
        bone_id.append(np.full(len(faces), i, dtype=np.int16))
        count += len(points)
    return np.vstack(vertex_groups), np.vstack(face_groups), np.concatenate(bone_id), np.array(names)


def heart_cases(heart_vertices):
    # Chamber positions use the registered heart box. The orientation columns
    # and three semiaxes define each ellipsoid in its own local frame.
    lower, upper = heart_vertices.min(axis=0), heart_vertices.max(axis=0)
    fraction = np.array([[.222, .530, .551], [.562, .480, .345],
                         [.256, .254, .623], [.709, .186, .546]])
    centers = lower + fraction * (upper - lower)
    max_axes = np.array([[16.5, 21.5, 14.0], [16.8, 17.8, 14.5],
                         [19.5, 27.5, 18.0], [20.5, 32.0, 19.0]]) * ((upper[0] - lower[0]) / 101.5)
    rotation = Rotation.from_euler("xz", [[-7., -4.], [-10., 6.], [-15., 13.], [-14., 20.]],
                                   degrees=True).as_matrix()
    t = np.arange(SAMPLES) / FPS
    bpm = np.array([62., 73., 84.])
    delay = np.array([[0., .023, .159, .176], [0., .032, .166, .154], [0., .020, .156, .172]])
    atrial = CubicHermiteSpline([0., .103, .466, .680, .855, 1.],
                               [.79, .51, 1., .78, .79, .79], np.zeros(6))
    phase = np.mod(t[None, :] * bpm[:, None] / 60., 1.)
    local = np.mod(phase[..., None] - delay[:, None, :], 1.)
    # Late ventricular filling starts just after that side's atrial squeeze.
    # The two sides share one clock with small, sample-dependent phase delays.
    volume = np.empty_like(local)
    volume[..., :2] = atrial(local[..., :2])
    refill_delay = .014
    for case in range(len(bpm)):
        for side in (0, 1):
            vent_start = delay[case, side + 2]
            kick = 1. + delay[case, side] + refill_delay - vent_start
            vent = CubicHermiteSpline([0., .062, .323, .376, .637, kick, 1.],
                                      [1., 1., .45, .45, .83, .85, 1.], np.zeros(7))
            volume[case, :, side + 2] = vent(local[case, :, side + 2])
    contraction = np.array([[1., 1., 1., 1.], [1.07, .94, 1.06, 1.04], [.95, 1.09, .93, 1.07]])
    volume = 1. - contraction[:, None, :] * (1. - volume)
    # The exponents sum to one, so the product of the semiaxis ratios
    # reproduces the chamber volume curve with less longitudinal shortening.
    shortening = np.array([[.38, .24, .38], [.38, .24, .38], [.39, .22, .39], [.39, .22, .39]])
    axes = max_axes * volume[..., None] ** shortening
    motion = (1.-volume[..., 2:].mean(axis=-1)) / .55
    xyz = centers[None, None] + motion[..., None, None] * np.array([.8, -3.4, -1.1])
    return t, bpm, delay, phase, volume, xyz, axes, rotation, centers, max_axes


def chamber_clearance(centers, axes, rotation):
    """Lower bound on the gap via separation along each line of centers."""
    minimum = np.inf
    for i, j in zip(*np.triu_indices(len(rotation), 1)):
        offset = centers[..., j, :] - centers[..., i, :]
        length = np.linalg.norm(offset, axis=-1)
        direction = offset / length[..., None]
        extent_i = np.linalg.norm((direction @ rotation[i]) * axes[..., i, :], axis=-1)
        extent_j = np.linalg.norm((direction @ rotation[j]) * axes[..., j, :], axis=-1)
        minimum = min(minimum, np.min(length - extent_i - extent_j))
    return float(minimum)


class Bones:
    def __init__(self, points, faces, ids):
        self.mesh = polydata(points, faces)
        self.locator = vtkStaticCellLocator()
        self.locator.SetDataSet(self.mesh)
        self.locator.BuildLocator()
        self.ids = ids

    def hits(self, origin, directions):
        distance = np.full(len(directions), np.inf)
        bone = np.full(len(directions), -1, dtype=np.int16)
        t = reference(0.)
        sub_id = reference(0)
        cell_id = reference(0)
        point, coordinates = [0.] * 3, [0.] * 3
        for i, v in enumerate(directions):
            if self.locator.IntersectWithLine(origin, origin + BEAM_RANGE * v, 1.e-6,
                                              t, point, coordinates, sub_id, cell_id):
                distance[i] = float(t) * BEAM_RANGE
                bone[i] = self.ids[int(cell_id)]
        return distance, bone

    def clearance(self, points, radius):
        distance2 = reference(0.)
        cell_id, sub_id = reference(0), reference(0)
        closest = [0.] * 3
        gap = []
        for center, r in zip(points.reshape(-1, 3), radius.ravel()):
            self.locator.FindClosestPoint(center, closest, cell_id, sub_id, distance2)
            gap.append(np.sqrt(float(distance2)) - r)
        return float(min(gap))


@dataclass
class Fan:
    h: float
    theta: float
    origin: np.ndarray
    forward: np.ndarray
    right: np.ndarray

    def sample(self, bones, width, height):
        x, y = np.meshgrid(np.linspace(-X_HALF, X_HALF, width), np.linspace(2., DEPTH, height))
        angle = np.arctan2(x, y)
        distance = np.hypot(x, y)
        beam = np.linspace(-BEAM_ANGLE, BEAM_ANGLE, 721)
        ray = np.cos(beam[:, None]) * self.forward + np.sin(beam[:, None]) * self.right
        hits, bone_ids = bones.hits(self.origin, ray)
        nearest = np.rint((angle / BEAM_ANGLE + 1.) * (len(beam) - 1) / 2.).astype(int).clip(0, len(beam) - 1)
        entry = hits[nearest]
        fan = (np.abs(angle) <= BEAM_ANGLE) & (distance <= DEPTH)
        behind = fan & (distance >= entry)
        pixels = self.origin + y[..., None] * self.forward + x[..., None] * self.right
        return pixels, fan, behind, distance - entry, bone_ids[nearest], y


def masks(pixels, centers, axes, rotation, sector):
    local = np.einsum("...ci,cij->...cj", pixels[..., None, :] - centers, rotation)
    distance = np.linalg.norm(local / axes, axis=-1)
    closest = np.argmin(distance, axis=-1)
    nearest = np.min(distance, axis=-1)
    return np.where(sector & (nearest < 1.), closest + 1, 0).astype(np.uint8), nearest


def windows(atlas, bones, centers, axes, rotation):
    # Search once to select illustrative fixed views using the simplified heart.
    # All three motion cases use exactly those same two probe poses.
    midpoint = centers.mean(axis=0)
    normal = np.linalg.svd(centers - midpoint, full_matrices=False)[2][-1]
    candidates = []
    for y in np.arange(-132., -27., 5.):
        h = (y - atlas.y0) / (atlas.y1 - atlas.y0)
        for angle in np.arange(-42., 52., 4.):
            theta = np.deg2rad(angle)
            origin = atlas.to_world(h, theta, 1.)
            forward = midpoint - origin
            forward /= np.linalg.norm(forward)
            right = np.cross(normal, forward)
            right /= np.linalg.norm(right)
            if right[0] < 0.:
                right = -right
            fan = Fan(h, theta, origin, forward, right)
            pixel, sector, shadow, *_ = fan.sample(bones, 76, 76)
            clean, _ = masks(pixel, centers, axes, rotation, sector)
            counts = np.bincount(clean.ravel(), minlength=5)[1:]
            seen = np.bincount(clean[~shadow].ravel(), minlength=5)[1:]
            fractions = seen / np.maximum(1, counts)
            if np.min(counts) >= 26:
                candidates.append((fan, counts, fractions))
    if not candidates:
        raise RuntimeError("no four-chamber slice found in the preview grid")
    open_fan, _, _ = max(candidates, key=lambda v: (v[2].min(), v[2].mean(), v[1].min()))
    alternatives = [v for v in candidates if .18 <= v[2].mean() <= .81 and v[2].max() > .6]
    if not alternatives:
        raise RuntimeError("no suitable partially shadowed slice found in the preview grid")
    blocked_fan, _, _ = min(alternatives, key=lambda v: np.linalg.norm(v[0].origin - open_fan.origin)
                            + 17. * abs(v[2].mean() - .53))
    return open_fan, blocked_fan


def simulate_observations(fans, bones, centers, axes, rotation):
    height, width = 208, 210
    shape = (*axes.shape[:2], len(fans), height, width)
    images, truth, observed = (np.zeros(shape, dtype=np.uint8) for _ in range(3))
    saved_sectors, saved_shadows, saved_plane, saved_ids = [], [], [], []
    rng = np.random.default_rng(2259)
    for window, fan in enumerate(fans):
        pixels, sector, shadow, from_bone, bone_id, y = fan.sample(bones, width, height)
        saved_sectors.append(sector)
        saved_shadows.append(shadow)
        saved_plane.append(pixels)
        saved_ids.append(bone_id)
        rib_echo = np.exp(-.5 * (np.clip(from_bone, -250., 250.) / 1.6) ** 2)
        texture = gaussian_filter(rng.standard_normal(sector.shape), (1.1, .8))
        gain = .92 - .43 * y / DEPTH
        for case in range(len(axes)):
            jitter = np.zeros((4, 3))
            size_jitter = np.zeros(4)
            lost = np.zeros(4, dtype=bool)
            for i, (c, a) in enumerate(zip(centers[case], axes[case])):
                gt, distance = masks(pixels, c, a, rotation, sector)
                jitter = .82 * jitter + .61 * rng.normal(size=(4, 3))
                size_jitter = .75 * size_jitter + .45 * rng.normal(size=4)
                kept, _ = masks(pixels, c + jitter, a + size_jitter[:, None], rotation, sector & ~shadow)
                lost = np.where(lost, rng.random(4) < .63, rng.random(4) < .010)
                kept[np.isin(kept, 1 + np.flatnonzero(lost))] = 0
                speckle = texture + .48 * gaussian_filter(rng.standard_normal(sector.shape), (.65, .55))
                echoes = gain * np.where(gt != 0, 25. + 21. * speckle, 103. + 82. * speckle)
                echoes += 110. * np.exp(-.5 * ((distance - 1.) / .048) ** 2)
                echoes = np.where(shadow, 6. + 13. * speckle, echoes) + 190. * rib_echo
                images[case, i, window] = np.where(sector, echoes.clip(0, 255), 0).astype(np.uint8)
                truth[case, i, window], observed[case, i, window] = gt, kept
    return (images, truth, observed, np.array(saved_sectors), np.array(saved_shadows),
            np.asarray(saved_plane, dtype=np.float32), np.array(saved_ids))


def main(source_path, out):
    out.mkdir(parents=True, exist_ok=True)
    with np.load(source_path, allow_pickle=False) as source:
        data = {name: source[name] for name in ("skin_vertices", "skin_faces", "mesh_names", "mesh_ranges", "faces", "candidate_vertices")}
    skin_vertices = data["skin_vertices"].astype(float) * 1000.
    atlas = ThoraxAtlas(skin_vertices, data["skin_faces"])
    bone_points, bone_faces, bone_ids, bone_names = bone_mesh(data)
    liver_points, liver_faces = mesh_arrays(data, "Liver")
    gall_points, gall_faces = mesh_arrays(data, "Gallbladder")
    diaphragm_points, diaphragm_faces = mesh_arrays(data, "Diaphragm")
    heart_points, heart_faces = mesh_arrays(data, "Heart")
    bones = Bones(bone_points, bone_faces, bone_ids)
    liver_points, gall_points, organ_shift, placement = place_liver_and_gallbladder(
        liver_points, liver_faces, gall_points, gall_faces, diaphragm_points, diaphragm_faces, bones.mesh)
    gall_center = trimesh.Trimesh(gall_points, gall_faces, process=False).center_mass
    t, bpm, delay, phase, volume, xyz, axes, rotation, rest_centers, rest_axes = heart_cases(heart_points)
    fans = windows(atlas, bones, rest_centers, rest_axes, rotation)
    print("atlas and views built", flush=True)
    images, truth, observed, sector, shadow, plane, shadow_bone = simulate_observations(fans, bones, xyz, axes, rotation)
    heart_disk, heart_shell = atlas.addresses(xyz)
    liver_disk, _ = atlas.addresses(liver_points)
    gall_disk, _ = atlas.addresses(gall_points)
    gall_center_disk, gall_center_shell = atlas.addresses(gall_center[None])

    valid = atlas.face >= 0
    anchor = np.einsum("ij,ijk->ik", atlas.bary[valid], skin_vertices[data["skin_faces"][atlas.face[valid]]])
    rng = np.random.default_rng(14)
    rr_h, rr_theta, rr_rho = rng.random(4000), rng.random(4000) * 2. * np.pi, rng.random(4000)
    points = atlas.to_world(rr_h, rr_theta, rr_rho)
    check_error = np.linalg.norm(points - atlas.to_world(*atlas.to_disk(points)), axis=1).max()
    unit_surface = pv.Sphere(theta_resolution=24, phi_resolution=18, radius=1.).points
    local_surface = axes[..., None, :] * unit_surface
    ellipsoid_points = xyz[..., None, :] + np.einsum("cij,...cnj->...cni", rotation, local_surface)
    shell_rho_max = atlas.to_disk(ellipsoid_points)[2].max()
    heart_lowest = np.min(xyz[..., 1] - np.linalg.norm(axes * rotation[:, 1, :], axis=-1))
    intervals = volume.min(axis=1), volume.max(axis=1)
    missing = [[float(np.count_nonzero((truth[:, :, i] == j) & shadow[i]) /
                      np.count_nonzero(truth[:, :, i] == j)) for j in range(1, 5)] for i in range(len(fans))]
    report = {
        "source": str(source_path), "length_unit": "mm", "axis": ["subject left", "superior", "anterior"],
        "axial_range_mm": [atlas.y0, atlas.y1], "atlas_knot_shape": list(atlas.depth_grid.shape),
        "surface_anchors": int(valid.sum()), "virtual_cut_knots": int((~valid).sum()),
        "surface_anchor_max_error_mm": float(np.linalg.norm(anchor - atlas.skin[valid], axis=-1).max()),
        "minimum_outward_depth_gap": float((-np.diff(atlas.depth_grid, axis=1)).min()),
        "graph_residual_normalized": atlas.graph_residual,
        "inverse_roundtrip_max_error_mm": float(check_error),
        "bone_objects": len(bone_names), "max_chamber_disk_radius_within_torso": float(shell_rho_max),
        "minimum_chamber_to_chamber_gap_mm": chamber_clearance(xyz, axes, rotation),
        "minimum_chamber_to_bone_gap_mm": bones.clearance(xyz, axes.max(axis=-1)),
        "liver_to_heart_superior_inferior_gap_mm": float(heart_lowest - liver_points[:, 1].max()),
        "max_liver_disk_radius_within_torso": float(np.linalg.norm(liver_disk[..., 1:], axis=-1).max()),
        "max_gallbladder_disk_radius_within_torso": float(np.linalg.norm(gall_disk[..., 1:], axis=-1).max()),
        "gallbladder_center_mm": gall_center.tolist(), "gallbladder_center_disk": gall_center_disk[0].tolist(),
        **placement,
        "case_bpm": bpm.tolist(), "chamber_order": CHAMBERS, "cycle_phase_offsets": delay.tolist(),
        "chamber_rest_semiaxes_mm": rest_axes.tolist(),
        "case_volume_ranges": [v.tolist() for v in intervals],
        "fraction_of_true_chamber_pixels_in_bone_shadow": [
            float(np.sum((truth[:, :, i] > 0) & shadow[i]) / np.count_nonzero(truth[:, :, i])) for i in range(len(fans))],
        "shadow_fraction_by_chamber": missing,
        "frames_per_case": len(t), "frame_rate_hz": FPS,
    }
    if (check_error > 1.e-4 or report["surface_anchor_max_error_mm"] > 1.e-5 or
            shell_rho_max >= 1. or report["minimum_chamber_to_bone_gap_mm"] <= 0. or
            report["minimum_chamber_to_chamber_gap_mm"] <= 0. or
            report["liver_to_heart_superior_inferior_gap_mm"] <= 0. or
            report["max_liver_disk_radius_within_torso"] >= 1. or report["max_gallbladder_disk_radius_within_torso"] >= 1. or
            np.any(observed[:, :, 0, shadow[0]]) or
            np.any(observed[:, :, 1, shadow[1]])):
        raise RuntimeError(f"preview geometry or simulated label validation failed: {report}")

    np.savez_compressed(out / "thorax_atlas.npz", h=atlas.h, theta=atlas.theta, rho=atlas.rho,
                        depth=atlas.depth_grid, world_knots_mm=atlas.grid, core_mm=atlas.centers,
                        core_scale_mm=atlas.core_scale, skin_mm=atlas.skin, skin_face_id=atlas.face,
                        barycentric=atlas.bary, source_skin_faces=data["skin_faces"],
                        bone_vertices_mm=bone_points, bone_faces=bone_faces, bone_face_id=bone_ids, bone_names=bone_names,
                        liver_vertices_mm=liver_points, liver_faces=liver_faces, liver_disk=liver_disk,
                        gallbladder_vertices_mm=gall_points, gallbladder_faces=gall_faces, gallbladder_disk=gall_disk,
                        gallbladder_center_mm=gall_center, gallbladder_center_disk=gall_center_disk[0],
                        gallbladder_center_shell_address=gall_center_shell[0],
                        diaphragm_vertices_mm=diaphragm_points, diaphragm_faces=diaphragm_faces,
                        liver_gallbladder_shift_mm=organ_shift,
                        heart_reference_vertices_mm=heart_points, heart_reference_faces=heart_faces)
    np.savez_compressed(out / "heartbeat_examples.npz", time_seconds=t, bpm=bpm, phase=phase,
                        chamber_order=np.array(CHAMBERS), chamber_phase_offsets=delay, relative_chamber_volume=volume,
                        chamber_centers_mm=xyz, chamber_semi_axes_mm=axes, chamber_rest_semi_axes_mm=rest_axes,
                        chamber_rotation_world_from_local=rotation, chamber_disk=heart_disk,
                        chamber_shell_coordinates=heart_shell, us_proxy=images, clean_mask=truth, observed_mask=observed,
                        sector=sector, bone_shadow=shadow, shadow_bone_id=shadow_bone, image_plane_mm=plane,
                        probe_origin_mm=np.array([f.origin for f in fans]), probe_forward=np.array([f.forward for f in fans]),
                        probe_right=np.array([f.right for f in fans]),
                        probe_skin_address=np.array([[f.theta % (2. * np.pi), f.h, 0.] for f in fans]))
    (out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    main(args.source, args.out)
