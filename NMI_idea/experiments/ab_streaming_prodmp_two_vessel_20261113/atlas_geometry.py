"""Fixed material coordinates for the right-leg streaming scan."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

HERE = Path(__file__).resolve().parent
TRANSFER = HERE.parent / "ab_bodyshape_atlas_transfer_20261113"
REFERENCE_KNEE_DEG = 15.
TAU = 2. * np.pi


def readonly(value):
    value = np.asarray(value)
    value.setflags(write=False)
    return value


def unit(vector):
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    if np.any(norm < 1.e-12):
        raise ValueError("degenerate material surface direction")
    return vector / norm


def frame(tangent, inward):
    inward = unit(inward)
    forward = unit(tangent - np.dot(tangent, inward) * inward)
    return np.column_stack((forward, np.cross(inward, forward), inward))


def surface_directions(outer, faces, h_levels):
    """Physical vectors sampled at the same material vertices in every shape."""
    xyz = outer.reshape(-1, 3)
    tris = xyz[faces]
    normal = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normals = np.zeros_like(xyz)
    for column in faces.T:
        np.add.at(normals, column, normal)
    tangents = np.gradient(outer, h_levels, axis=0, edge_order=2).reshape(-1, 3)
    return unit(tangents), unit(normals)


def array_digest(*arrays):
    sha = hashlib.sha256()
    for array in arrays:
        data = np.asarray(array)
        sha.update(str(data.shape).encode())
        sha.update(data.dtype.str.encode())
        sha.update(data.tobytes())
    return sha.hexdigest()


class SurfaceChart:
    """Periodic surface lookup in the fixed material labels, evaluated in 3-D later."""

    def __init__(self, asset):
        self.faces = asset.skin_faces
        self.h = asset.h.reshape(asset.grid_shape)[:, -1]
        self.theta = np.mod(asset.theta.reshape(asset.grid_shape)[:, -1], TAU)
        self.levels = self.h[:, 0]
        self.bearings = self.theta[0]
        if not (np.all(self.h == self.levels[:, None]) and np.all(self.theta == self.bearings)
                and np.all(np.diff(self.levels) > 0.) and np.all(np.diff(self.bearings) > 0.)):
            raise ValueError("shaft surface is not a periodic longitudinal grid")
        self.cells = np.full((len(self.levels) - 1, len(self.bearings), 2), -1, dtype=int)
        for face_id, face in enumerate(self.faces):
            i, j = np.unravel_index(face, self.h.shape)
            j = int(j.max() if j.max() - j.min() > 1 else j.min())
            slots = self.cells[i.min(), j]
            if slots[1] >= 0:
                raise ValueError("more than two triangles at one chart grid cell")
            slots[int(slots[0] >= 0)] = face_id
        if np.any(self.cells < 0):
            raise ValueError("incomplete periodic material surface")

    def address(self, h, theta):
        if not self.levels[0] <= h <= self.levels[-1]:
            raise ValueError("station outside the material skin chart")
        theta = float(theta % TAU)
        i = min(int(np.searchsorted(self.levels, h, side="right")) - 1, len(self.levels) - 2)
        j = (int(np.searchsorted(self.bearings, theta, side="right")) - 1) % len(self.bearings)
        for face_id in self.cells[i, j]:
            ids = self.faces[face_id]
            angle_delta = self.theta.ravel()[ids] - theta
            equations = np.stack((np.ones(3), self.h.ravel()[ids], np.sin(angle_delta)))
            weights = np.linalg.solve(equations, [1., h, 0.])
            if np.min(weights) >= -1.e-9 and weights @ np.cos(angle_delta) > 0.:
                return ids, weights
        raise ValueError("address outside both triangles of the material cell")

    def labels(self, faces, weights):
        ids = self.faces[faces]
        h = np.sum(self.h.ravel()[ids] * weights, axis=-1)
        bearing = np.angle(np.sum(np.exp(1j * self.theta.ravel()[ids]) * weights, axis=-1))
        return h, bearing


@dataclass(frozen=True)
class ShapeGeometry:
    """The fixed chart evaluated at one known body shape and knee angle."""

    template: ScanTemplate
    vessels_m: np.ndarray
    volume_refiner: object
    skin_xyz: np.ndarray
    skin_tangent: np.ndarray
    skin_inward: np.ndarray
    offset_axes: np.ndarray
    metadata: dict

    @property
    def s_mm(self):
        return self.template.s_mm

    def _robot_points(self, points):
        transform = self.template.robot_from_atlas
        return points @ transform[:3, :3].T + transform[:3, 3]

    def xyz_to_xi(self, points_m):
        """Convert robot-frame metres to strict baked (theta, h, d) addresses."""
        points_m = np.asarray(points_m, dtype=float)
        if points_m.shape[-1:] != (3,) or not np.isfinite(points_m).all():
            raise ValueError("xyz must be finite with a final dimension of three")
        transform = self.template.robot_from_atlas
        local = (points_m - transform[:3, 3]) @ transform[:3, :3]
        return self.volume_refiner.p_to_xi(local).reshape(points_m.shape)

    def xi_to_xyz(self, addresses):
        """Evaluate a baked (theta, h, d) address in robot-frame metres."""
        addresses = np.asarray(addresses, dtype=float)
        if addresses.shape[-1:] != (3,) or not np.isfinite(addresses).all():
            raise ValueError("xi must be finite with a final dimension of three")
        return self._robot_points(self.volume_refiner.xi_to_p(addresses)).reshape(addresses.shape)

    def surface_address(self, s_mm, x_ref_mm):
        """Return the same triangle node IDs and barycentrics for every beta."""
        return self.template.surface_address(s_mm, x_ref_mm)

    def pose(self, s_mm, x_ref_mm):
        """Map a scalar or broadcast material scan query to a physical SE(3) pose."""
        station, lateral = np.broadcast_arrays(np.asarray(s_mm, dtype=float),
                                               np.asarray(x_ref_mm, dtype=float))
        if (not np.isfinite(station).all() or not np.isfinite(lateral).all()
                or np.any((station < self.s_mm[0]) | (station > self.s_mm[-1]))):
            raise ValueError("finite scan query must lie within the registered station range")
        result = np.broadcast_to(np.eye(4), station.shape + (4, 4)).copy()
        if station.size == 0:
            return result
        reference = self.template.reference_probe_frame(station.reshape(-1)).as_matrix()
        transform = self.template.robot_from_atlas
        for i, (s, x) in enumerate(zip(station.flat, lateral.flat, strict=True)):
            ids, bary = self.surface_address(s, x)
            current_frame = frame(bary @ self.skin_tangent[ids], bary @ self.skin_inward[ids])
            template_frame = frame(bary @ self.template.tangent[ids], bary @ self.template.inward[ids])
            matrix = result.reshape(-1, 4, 4)[i]
            relative_rotation = template_frame.T @ reference[i]
            matrix[:3, :3] = transform[:3, :3] @ current_frame @ relative_rotation
            matrix[:3, 3] = self._robot_points(bary @ self.skin_xyz[ids])
        return result


class ScanTemplate:
    """Reference chart with fixed source weights, labels, and vessel addresses."""

    def __init__(self, among: Path, repair: Path):
        import trimesh

        among, repair = Path(among).resolve(), Path(repair).resolve()
        sys.path[:0] = [str(among / "src"), str(TRANSFER)]
        from run_geometry import boundary, positive_overlap, sha
        from transport import ShapeTransport
        from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import load_leg_volume_atlas
        from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.harmonic import locate_baked_tetrahedra
        from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.volume_refine import VolumeTetRefiner

        self._transport = ShapeTransport(among, repair)
        self.reference_beta = self._transport.beta
        self.asset = self._transport.sides["right"]
        self._atlas = load_leg_volume_atlas(self._transport.source_paths["atlas_right"])
        self._refiner = VolumeTetRefiner
        self._overlap = positive_overlap
        self._kernel = repair / "harmonic_independent" / "tet_overlap"
        if not self._kernel.is_file():
            raise FileNotFoundError(f"missing volume overlap checker: {self._kernel}")
        self._boundary_faces = boundary(self.asset.tets).astype(np.int64)
        self._reference_det = self.determinants(self.asset.vertices)
        if np.any(self._reference_det <= 0.):
            raise ValueError("invalid orientation in the original material template")
        source_volume, skin, faces, _ = self._transport.deform("right", (0., 0.), REFERENCE_KNEE_DEG)
        self._reference_volume = readonly(source_volume.astype(np.float32))
        self._reference_metrics = self.check_geometry(self._reference_volume, skin, faces, self.reference_beta,
                                                      REFERENCE_KNEE_DEG)

        sections_path = HERE / "SECTIONS.json"
        dynamic_path = HERE.parent / "ab_leg_dynamic_20261112/DYNAMIC_GEOMETRY_FROZEN.npz"
        sections = json.loads(sections_path.read_text())
        self.s_mm = readonly(np.asarray(sections["s_mm"], dtype=float))
        origins = np.asarray(sections["origin_atlas_mm"]) / 1000.
        axes = np.asarray(sections["axes_atlas"])
        centers = np.asarray(sections["centers_mm"]) / 1000.
        with np.load(dynamic_path, allow_pickle=False) as data:
            self.robot_from_atlas = readonly(data["robot_from_atlas"])
        if sha(dynamic_path) != sections["source_sha256"]["dynamic_geometry_frozen_npz"]:
            raise ValueError("section table and robot transform refer to different reference geometry")
        self.reference_probe_frame = Slerp(self.s_mm, Rotation.from_matrix(axes))

        self.chart = SurfaceChart(self.asset)
        outer = self.outer(self._reference_volume)
        outer_mesh = trimesh.Trimesh(vertices=outer.reshape(-1, 3), faces=self.chart.faces, process=False)
        nearest, distance, faces = trimesh.proximity.closest_point(outer_mesh, origins)
        bary = trimesh.triangles.points_to_barycentric(outer_mesh.triangles[faces], nearest)
        h, theta = self.chart.labels(faces, bary)
        self._station = CubicSpline(self.s_mm, h, bc_type="natural")
        self._angle = CubicSpline(self.s_mm, np.unwrap(theta), bc_type="natural")

        tangents, normals = surface_directions(outer, self.chart.faces, self.chart.levels)
        mid = len(faces) // 2
        mid_ids, mid_weights = self.chart.faces[faces[mid]], bary[mid]
        self._sign_t = np.copysign(1., (mid_weights @ tangents[mid_ids]) @ axes[mid, :, 0])
        self._sign_n = np.copysign(1., (mid_weights @ normals[mid_ids]) @ axes[mid, :, 2])
        self.tangent, self.inward = readonly(self._sign_t * tangents), readonly(self._sign_n * normals)
        reference_xyz = outer.reshape(-1, 3)
        lateral = []
        for hi, angle, axis in zip(h, theta, axes[:, :, 1], strict=True):
            left, lw = self.chart.address(hi, angle - 1.e-4)
            right, rw = self.chart.address(hi, angle + 1.e-4)
            derivative = (rw @ reference_xyz[right] - lw @ reference_xyz[left]) / 2.e-4
            lateral.append(float(derivative @ axis))
        self._lateral_per_radian = CubicSpline(self.s_mm, lateral, bc_type="natural")
        if np.any(np.asarray(lateral) * lateral[mid] <= 0.) or abs(lateral[mid]) < 1.e-4:
            raise ValueError("invalid lateral scale on the reference material chart")

        vessels = origins[:, None, :] + np.einsum("nij,nvj->nvi", axes[:, :, 1:], centers)
        refiner = self.make_refiner(self._reference_volume)
        self.vessel_tets, self.vessel_bary = map(readonly, locate_baked_tetrahedra(
            refiner.fields, vessels.reshape(-1, 3)))
        self.vessel_xi = readonly(refiner.p_to_xi(vessels).reshape(vessels.shape))
        reconstructed = np.einsum("ni,nij->nj", self.vessel_bary,
                                  self._reference_volume[self.asset.tets[self.vessel_tets]])
        residual = float(np.max(np.linalg.norm(reconstructed - vessels.reshape(-1, 3), axis=1)))
        if residual > 3.e-6:
            raise ValueError("fixed tetrahedral vessel attachment exceeds 3 micrometres")
        self._binding = {
            "vessel_count": len(self.vessel_tets), "surface_anchor_count": len(faces),
            "vessel_reconstruction_max_m": residual,
            "surface_reference_migration_max_mm": 1000. * float(distance.max()),
            "sha256": array_digest(self.asset.tets, self.asset.h, self.asset.theta, self.asset.depth,
                                   self.asset.source_surface_face_indices, self.asset.source_surface_barycentric,
                                   self.asset.stencil_source_ids, self.asset.stencil.indices,
                                   self.asset.stencil.indptr, self.asset.stencil.data, self.chart.faces,
                                   faces, bary, self.vessel_tets, self.vessel_bary),
        }
        paths = {**self._transport.source_paths, "sections": sections_path,
                 "robot_transform": dynamic_path, "volume_overlap_checker": self._kernel}
        self._source_hashes = {name: sha(path) for name, path in paths.items() if path.is_file()}

    def outer(self, volume):
        return volume.reshape(self.asset.grid_shape + (3,))[:, -1].astype(float)

    def determinants(self, volume):
        cells = np.asarray(volume, dtype=float)[self.asset.tets]
        return np.linalg.det(cells[:, 1:] - cells[:, :1])

    def make_refiner(self, volume):
        atlas = replace(self._atlas, harmonic_vertices=volume, skin_vertices=self.outer(volume).reshape(-1, 3))
        return self._refiner.from_atlas(atlas)

    def surface_address(self, s_mm, x_ref_mm):
        s, x = float(s_mm), float(x_ref_mm)
        if not np.isfinite(s + x) or not self.s_mm[0] <= s <= self.s_mm[-1]:
            raise ValueError("finite station and lateral offset required within the registered scan")
        scale = float(self._lateral_per_radian(s))
        return self.chart.address(float(self._station(s)), float(self._angle(s)) + .001 * x / scale)

    def check_geometry(self, volume, skin, faces, beta, knee):
        from igl.copyleft.cgal import remesh_self_intersections

        def intersections(vertices, triangles):
            return len(remesh_self_intersections(np.asarray(vertices, dtype=float),
                                                 np.asarray(triangles, dtype=np.int64), detect_only=True,
                                                 first_only=False, cutoff=10_000_000)[2])

        det = self.determinants(volume)
        metrics = {
            "tetrahedra": len(det), "nonpositive_tets": int(np.count_nonzero(det <= 0.)),
            "min_det_ratio": float(np.min(det / self._reference_det)),
            "positive_overlap_pairs": self._overlap(self._kernel, volume, self.asset.tets),
            "volume_boundary_intersection_pairs": intersections(volume, self._boundary_faces),
            "retained_skin_intersection_pairs": intersections(np.asarray(skin, dtype=np.float32), faces),
        }
        metrics["geometry_pass"] = (metrics["nonpositive_tets"] == 0 and metrics["min_det_ratio"] > 1.e-3
                                    and metrics["positive_overlap_pairs"] == 0
                                    and metrics["volume_boundary_intersection_pairs"] == 0
                                    and metrics["retained_skin_intersection_pairs"] == 0)
        if not metrics["geometry_pass"]:
            raise ValueError(f"unsupported geometry at beta={beta.tolist()}, knee={knee:g}°: {metrics}")
        return metrics

    def instantiate(self, delta_beta, knee_deg, *, beta=None):
        volume, skin, faces, _ = self._transport.deform("right", delta_beta, knee_deg, beta=beta)
        requested_beta = self.reference_beta + np.r_[delta_beta, np.zeros(8)] if beta is None else np.asarray(beta, float)
        volume = readonly(volume.astype(np.float32))
        reference = np.array_equal(requested_beta, self.reference_beta) and knee_deg == REFERENCE_KNEE_DEG
        metrics = self._reference_metrics if reference else self.check_geometry(volume, skin, faces, requested_beta, knee_deg)
        outer = self.outer(volume)
        tangent, inward = surface_directions(outer, self.chart.faces, self.chart.levels)
        vessels = np.einsum("ni,nij->nj", self.vessel_bary, volume[self.asset.tets[self.vessel_tets]]).reshape(-1, 2, 3)
        r, p = self.robot_from_atlas[:3, :3], self.robot_from_atlas[:3, 3]
        mid = float(self.s_mm[len(self.s_mm) // 2])
        ids, bary = self.surface_address(mid, 0.)
        frame_now = frame(self._sign_t * (bary @ tangent[ids]), self._sign_n * (bary @ inward[ids]))
        frame_ref = frame(bary @ self.tangent[ids], bary @ self.inward[ids])
        offset_axes = (r @ frame_now @ frame_ref.T @ self.reference_probe_frame(mid).as_matrix())[:, 1:]
        metadata = {
            "side": "right", "reference_beta": self.reference_beta.tolist(), "beta": requested_beta.tolist(),
            "beta_delta": (requested_beta - self.reference_beta).tolist(), "knee_deg": float(knee_deg),
            "reference_knee_deg": REFERENCE_KNEE_DEG, "material_binding": self._binding.copy(),
            "geometry_gate": metrics.copy(), "input_sha256": self._source_hashes.copy(),
        }
        return ShapeGeometry(self, readonly(vessels @ r.T + p), self.make_refiner(volume),
                             readonly(outer.reshape(-1, 3)), readonly(self._sign_t * tangent),
                             readonly(self._sign_n * inward), readonly(offset_axes), metadata)
