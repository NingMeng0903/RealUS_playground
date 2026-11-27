"""Registered right-leg SMPL-X kinematics, not a tissue mechanics solver.

XYZ uses the SMPL-X world axes (x left, y superior, z anterior) in mm;
there is no axis swap, recentering, or CT-specific fit. The sealed shaft atlas
uses this same coordinate frame but was baked at its recovered subject beta,
not at zero beta. Its native material labels are (theta in rad, h, d), with
h and d unitless. It excludes the centerline and the proximal/distal ends.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
from scipy.spatial import cKDTree

from NMI_idea.experiments.ab_leg_bifurcation_20261112.leg_chart import (
    ChartDomainError,
    LegChart,
)
from src.projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import (
    LegVolumeAtlas,
    load_leg_volume_atlas,
)

_EXPERIMENTS = Path(__file__).resolve().parents[2]
_AMONG_US = Path("/media/camp/EXT_DRIVE/Among_US")
DEFAULT_MODEL_PATH = _AMONG_US / "ref_code_library/InteractVLM/data/body_models/smplx/SMPLX_MALE.npz"
DEFAULT_REFERENCE_MANIFEST_PATH = (
    _AMONG_US / "outputs/anatomy_retarget/recovered_smplx_20261108/source_manifest.json"
)
DEFAULT_ATLAS_PATH = (
    _EXPERIMENTS / "ab_validation_20261110/repair_20261110/harmonic/shaft_roi_v2/atlas_right.npz"
)
# The seals pair the original atlas geometry with its specific nodal attachments.
_ATLAS_SHA = "f1a42becc403ebf7687025d4ab8b7d3480e52d4713df8c6357449b23d4ebba1c"
_ATTACHMENTS_SHA = "bfc7c76b083be69604ec18c963d4110870d2437afdb19aba8d95c5e6fb4f1b3f"
_MM_PER_M = 1000.0
NEUTRAL_BETA = np.zeros(300, dtype=np.float64)
NEUTRAL_BETA.setflags(write=False)
_TAU = 2.0 * np.pi
_LABEL_TOL = 2.0e-8
_HIP, _KNEE, _ANKLE = 2, 5, 8
_POSE_JOINTS = np.array([_HIP, _KNEE])
# q = (hip flexion, hip abduction, hip axial rotation, knee flexion).
# Hip is one local rotation vector, not three sequential Euler rotations.
_ROT_VEC_FROM_Q = np.array([
    [[-1., 0., 0., 0.], [0., 0., 1., 0.], [0., -1., 0., 0.]],
    [[0., 0., 0., 1.], [0., 0., 0., 0.], [0., 0., 0., 0.]],
])


def _require_sha(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"asset does not match its registered SHA-256: {path}")


def _xyz_rows(values, name: str, *, finite: bool = True) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (..., 3)")
    if finite and not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _local_rotation(q_map: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rodrigues and its derivative via the SO(3) right Jacobian."""
    rotvec = q_map @ q
    angle = np.linalg.norm(rotvec)
    a = np.sinc(angle / np.pi)
    b = 0.5 * np.sinc(angle / _TAU) ** 2
    c = (1.0 - a) / angle**2 if angle > 1.0e-3 else 1.0 / 6.0 - angle**2 / 120.0
    skew = _skew(rotvec)
    skew2 = skew @ skew
    rotation = np.eye(3) + a * skew + b * skew2
    right_jac = np.eye(3) - b * skew + c * skew2
    derivative = np.stack([rotation @ _skew(direction) for direction in (right_jac @ q_map).T])
    return rotation, derivative


def _skin_stencil(tree: cKDTree, points_m: np.ndarray, *, gradient=False):
    """Local Shepard extension of actual surface samples, exact at a vertex.

    The eighth neighbor defines a compact support radius; its zero weight
    makes an exchange at the stencil boundary continuous. This is not a
    containment test or a correspondence calibrated on internal anatomy.
    """
    distance, indices = tree.query(points_m, k=8)
    exact = distance[:, 0] <= 1.0e-12
    radius = distance[:, -1:]
    distance = np.maximum(distance, 1.0e-12)
    ratio = np.maximum(radius - distance, 0.0) / distance
    weights = ratio**2
    weights[exact] = 0.0
    weights[exact, 0] = 1.0
    total = weights.sum(axis=1, keepdims=True)
    weights /= total
    if not gradient:
        return indices, weights
    unit = (points_m[:, None, :] - tree.data[indices]) / distance[..., None]
    d_weights = 2.0 * (ratio / distance)[..., None] * (
        unit[:, -1:, :] - (radius / distance)[..., None] * unit)
    d_weights[exact] = 0.0
    d_weights -= weights[..., None] * d_weights.sum(axis=1, keepdims=True)
    return indices, weights, d_weights / total[..., None]


@dataclass
class _RestShape:
    beta: np.ndarray
    vertices_m: np.ndarray
    joints_m: np.ndarray

    @cached_property
    def skin_tree(self) -> cKDTree:
        return cKDTree(self.vertices_m)


class SMPLXLegKinematics:
    """Four-DOF right-leg motion with the actual 55-joint SMPL-X hierarchy.

    ``beta=None`` uses the sealed atlas's recovered reference shape. Supply
    an array of zeros for the unshaped template. The model's first 300 shape
    modes are supported; the expression modes are not part of ``beta``.
    Pose units are radians in the order ``(hip flexion, hip abduction,
    hip axial rotation, knee flexion)``. Local rotation vectors are
    ``(-flexion, axial, -abduction)`` at the right hip and ``(knee,0,0)``
    at the right knee, matching the bake's positive knee-flexion convention.
    The ankle has its real inherited transform and no independent drive.

    An independent pose-Jacobian check can central-difference the first
    result of ``forward_pose`` at ``q ± 1e-5 * eye(4)[i]``. For a known
    regular sample from the existing chart audit use nominal coordinates
    ``[[-24,0,12], [0,12,18], [24,-12,18]]`` with ``self.chart.xi``. Check
    ``self.material_support(xi)`` first, then compare neutral ``Phi`` at
    ``beta=None, q=zeros(4)`` against ``self.chart.world_from_xi(xi)`` and
    ``self.chart.world(chart_points_mm)``. The saved neutral pose uses float32.
    """

    def __init__(
        self,
        *,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        atlas_path: Path | str = DEFAULT_ATLAS_PATH,
        attachments_path: Path | str | None = None,
        reference_manifest_path: Path | str = DEFAULT_REFERENCE_MANIFEST_PATH,
    ) -> None:
        model_path = Path(model_path)
        self.atlas_path = Path(atlas_path)
        attachments_path = (self.atlas_path.with_name("attachments_right.npz")
                            if attachments_path is None else Path(attachments_path))
        manifest = json.loads(Path(reference_manifest_path).read_text())
        _require_sha(model_path, manifest["model_sha256"])
        _require_sha(self.atlas_path, _ATLAS_SHA)
        _require_sha(attachments_path, _ATTACHMENTS_SHA)
        self.atlas: LegVolumeAtlas = load_leg_volume_atlas(self.atlas_path)

        with np.load(model_path, allow_pickle=False) as model:
            self._template_m = model["v_template"].astype(np.float64)
            self._shape_dirs = model["shapedirs"][..., :300]
            self._joint_regressor = model["J_regressor"].astype(np.float64)
            self._weights = model["weights"].astype(np.float64)
            hierarchy = model["kintree_table"]
            pose_columns = ((9 * (_POSE_JOINTS - 1))[:, None] + np.arange(9)).ravel()
            self._pose_dirs = model["posedirs"][..., pose_columns]
        parent_ids, joint_ids = hierarchy
        joint_columns = {int(joint): column for column, joint in enumerate(joint_ids)}
        self._parents = np.array([joint_columns[int(parent)] for parent in parent_ids[1:]])
        self._parents = np.r_[-1, self._parents]
        if (joint_ids.size != 55 or parent_ids[0] in joint_columns or
                np.any(self._parents >= np.arange(55)) or
                not np.array_equal(self._parents[[_HIP, _KNEE, _ANKLE]], [0, _HIP, _KNEE])):
            raise ValueError("unexpected SMPL-X right-leg hierarchy")
        self._reference = self._make_shape(self._validate_beta(manifest["betas"]))
        self._last_shape = self._reference
        baked_joints = np.stack([self.atlas.pelvis, self.atlas.hip, self.atlas.knee, self.atlas.ankle])
        if not np.allclose(self._reference.joints_m[[0, _HIP, _KNEE, _ANKLE]], baked_joints,
                           rtol=0, atol=2.0e-7):
            raise ValueError("SMPL-X reference joints do not register to the sealed shaft atlas")

        with np.load(attachments_path, allow_pickle=False) as attached:
            nh, nr, ntheta = attached["grid_shape"]
            self._atlas_weights = attached["volume_lbs_weights"].astype(np.float64)
        self._atlas_weights /= self._atlas_weights.sum(axis=1, keepdims=True)
        self._atlas_points_m = self.atlas.harmonic_vertices.astype(np.float64)
        self._labels = np.stack((self.atlas.harmonic_theta, self.atlas.harmonic_h,
                                 self.atlas.harmonic_d), axis=1).astype(np.float64)
        grid = self._labels.reshape(nh, nr, ntheta, 3)
        self._h_grid = grid[:, 0, 0, 1]
        self._theta_grid = grid[0, 0, :, 0]
        # The sealed atlas is ordered by (h slab, radial layer, angular sector,
        # 6 tetrahedra). The entire radial column is searched for label d.
        self._tet_bins = self.atlas.harmonic_tets.reshape(nh - 1, nr - 1, ntheta, 6, 4)
        self._tet_bins = self._tet_bins.transpose(0, 2, 1, 3, 4).reshape(nh - 1, ntheta, -1, 4)

    @property
    def reference_beta(self) -> np.ndarray:
        return self._reference.beta.copy()

    @cached_property
    def chart(self) -> LegChart:
        """The existing checked nominal (longitudinal, lateral, depth) patch."""
        chart = LegChart(0, atlas_path=self.atlas_path)
        if not np.allclose(chart.pose_vertices_m, self._atlas_points_m, rtol=0, atol=2.0e-7):
            raise ValueError("LegChart neutral pose does not match the canonical atlas")
        return chart

    def _validate_beta(self, beta) -> np.ndarray:
        values = np.asarray(beta, dtype=np.float64)
        if values.ndim != 1 or values.size > self._shape_dirs.shape[-1] or not np.isfinite(values).all():
            raise ValueError("beta must be a finite vector of at most 300 shape coefficients")
        return np.trim_zeros(values, trim="b").copy()

    def _make_shape(self, beta: np.ndarray) -> _RestShape:
        vertices = self._template_m
        if beta.size:
            vertices = vertices + np.einsum("vcs,s->vc", self._shape_dirs[..., :beta.size], beta)
        return _RestShape(beta, vertices, self._joint_regressor @ vertices)

    def _shape(self, beta) -> _RestShape:
        if beta is None:
            return self._reference
        values = self._validate_beta(beta)
        if np.array_equal(values, self._reference.beta):
            return self._reference
        if not np.array_equal(values, self._last_shape.beta):
            self._last_shape = self._make_shape(values)
        return self._last_shape

    @cached_property
    def _neutral(self) -> _RestShape:
        return self._make_shape(NEUTRAL_BETA[:0])

    def _joint_transforms(self, q_t, joints: np.ndarray):
        q = np.asarray(q_t, dtype=np.float64)
        if q.shape != (4,) or not np.isfinite(q).all():
            raise ValueError("q_t must contain four finite pose coordinates in radians")
        rotation = np.empty((55, 3, 3))
        translation = np.empty((55, 3))
        d_rotation = np.empty((55, 4, 3, 3))
        d_translation = np.empty((55, 4, 3))
        local_rot = np.empty((2, 3, 3))
        d_local_rot = np.empty((2, 4, 3, 3))
        for i, q_map in enumerate(_ROT_VEC_FROM_Q):
            local_rot[i], d_local_rot[i] = _local_rotation(q_map, q)
        rotation[0], translation[0] = np.eye(3), joints[0]
        d_rotation[0], d_translation[0] = 0.0, 0.0
        for i, parent in enumerate(self._parents[1:], start=1):
            offset = joints[i] - joints[parent]
            translation[i] = rotation[parent] @ offset + translation[parent]
            d_translation[i] = d_rotation[parent] @ offset + d_translation[parent]
            if i == _HIP or i == _KNEE:
                local = 0 if i == _HIP else 1
                rotation[i] = rotation[parent] @ local_rot[local]
                d_rotation[i] = (d_rotation[parent] @ local_rot[local]
                                 + rotation[parent] @ d_local_rot[local])
            else:
                rotation[i], d_rotation[i] = rotation[parent], d_rotation[parent]
        # Remove the shaped rest joint transform (inverse bind).
        translation -= np.einsum("jrc,jc->jr", rotation, joints)
        d_translation -= np.einsum("jqrc,jc->jqr", d_rotation, joints)
        return (rotation, translation, d_rotation, d_translation), (local_rot, d_local_rot)

    @staticmethod
    def _skin(points, weights, transforms, d_points=None):
        rotation, translation, d_rotation, d_translation = transforms
        blended = np.einsum("nj,jrc->nrc", weights, rotation)
        phi = np.einsum("nrc,nc->nr", blended, points) + weights @ translation
        jac = (np.einsum("nj,jqrc,nc->nrq", weights, d_rotation, points, optimize=True)
               + np.einsum("nj,jqr->nrq", weights, d_translation))
        if d_points is not None:
            jac += blended @ d_points
        return phi, blended, jac

    def _locate_material(self, labels: np.ndarray):
        theta, h, d = labels.T
        valid = np.isfinite(labels).all(axis=1)
        valid &= (h >= self._h_grid[0] - _LABEL_TOL) & (h <= self._h_grid[-1] + _LABEL_TOL)
        valid &= (d >= -_LABEL_TOL) & (d <= 1.0 + _LABEL_TOL)
        rows = np.flatnonzero(valid)
        query = np.column_stack((theta[rows] % _TAU,
                                 np.clip(h[rows], self._h_grid[0], self._h_grid[-1]),
                                 np.clip(d[rows], 0.0, 1.0)))
        hi = np.searchsorted(self._h_grid, query[:, 1], side="right") - 1
        hi = np.minimum(hi, len(self._h_grid) - 2)
        ti = np.searchsorted(self._theta_grid, query[:, 0], side="right") - 1
        candidates = self._tet_bins[hi, ti]
        corners = self._labels[candidates]
        theta0 = query[:, None, None, 0]
        corners[..., 0] = theta0 + (corners[..., 0] - theta0 + np.pi) % _TAU - np.pi
        mat = np.swapaxes(corners[..., 1:, :] - corners[..., :1, :], -1, -2)
        local = np.linalg.solve(mat, (query[:, None, :] - corners[..., 0, :])[..., None])[..., 0]
        bary = np.concatenate((1.0 - local.sum(axis=-1, keepdims=True), local), axis=-1)
        score = bary.min(axis=-1)
        column = score.argmax(axis=1)
        selected = np.arange(rows.size)
        found = score[selected, column] >= -_LABEL_TOL
        selected, column = selected[found], column[found]
        weights = np.maximum(bary[selected, column], 0.0)
        weights /= weights.sum(axis=1, keepdims=True)
        return rows[found], candidates[selected, column], weights, mat[selected, column]

    def material_support(self, material_coordinates) -> np.ndarray:
        """Return a mask of containing tetrahedra in the sealed annular shaft.

        Nonfinite addresses and unsupported h/d return False. Theta is
        periodic. This mask is for the reference atlas, not a certificate of
        containment or injectivity after an arbitrary shape/pose change.
        """
        labels = _xyz_rows(material_coordinates, "material_coordinates", finite=False)
        rows, _, _, _ = self._locate_material(labels.reshape(-1, 3))
        mask = np.zeros(labels.size // 3, dtype=bool)
        mask[rows] = True
        return mask.reshape(labels.shape[:-1])

    def forward_pose(self, beta, q_t, points_mm=None, *, material_coordinates=None, chart_points_mm=None):
        """Return ``(Phi_mm, G_frame, J_q_mm_per_rad)`` for one beta/pose and batched points.

        Exactly one of these coordinate inputs is required (each ``(...,3)``):

        * ``points_mm``: Cartesian rest positions **already expressed in the
          requested beta's rest shape**. SMPL-X skin weights are extended by
          local interpolation; this allows the groin/calf XYZ route but is
          neither the harmonic map nor a claim that the points are inside skin.
          ``G_frame`` is the proper polar rotation of the blended joint
          rotations, in rest XYZ basis directions.
        * ``material_coordinates``: sealed (theta,h,d) atlas addresses. Native
          containing tetrahedra transport by interpolating their posed nodal
          positions; nodes use the baked attachments, not nearest-surface
          weights. Nonreference beta shifts the nodes with interpolated
          SMPL-X shape displacements, an unverified interior shape extension.
          The proper frame is built from posed (increasing h, increasing
          theta, toward increasing d) tangents in the selected tetrahedron.
          Out-of-domain queries raise ``ChartDomainError`` rather than snap.
        * ``chart_points_mm``: nominal bounded LegChart points; first map to
          native material coordinates with the existing neutral LegChart.

        Neither interior path applies surface pose correctives. Use
        ``forward_vertices`` for the exact SMPL-X surface model. A local
        probe orientation ``rho`` becomes ``G_frame @ rho``. Pose Jacobians
        are analytic derivatives of Phi only, holding beta and input fixed;
        a moving probe-frame residual additionally needs its own derivative.
        """
        if sum(value is not None for value in (points_mm, material_coordinates, chart_points_mm)) != 1:
            raise ValueError("provide exactly one of points_mm, material_coordinates, chart_points_mm")
        shape = self._shape(beta)
        transforms, _ = self._joint_transforms(q_t, shape.joints_m)
        if points_mm is not None:
            array = _xyz_rows(points_mm, "points_mm")
            points = array.reshape(-1, 3) / _MM_PER_M
            neighbors, stencil = _skin_stencil(shape.skin_tree, points)
            weights = np.einsum("nk,nkj->nj", stencil, self._weights[neighbors])
            weights /= weights.sum(axis=1, keepdims=True)
            phi, blended, jac = self._skin(points, weights, transforms)
            u, _, vh = np.linalg.svd(blended)
            u[:, :, -1] *= np.where(np.linalg.det(u @ vh) < 0.0, -1.0, 1.0)[:, None]
            frame = u @ vh
        else:
            if chart_points_mm is not None:
                material_coordinates = self.chart.xi(_xyz_rows(chart_points_mm, "chart_points_mm"))
            array = _xyz_rows(material_coordinates, "material_coordinates", finite=False)
            rows, corners, weights, label_edges = self._locate_material(array.reshape(-1, 3))
            if len(rows) != array.size // 3:
                missing = np.setdiff1d(np.arange(array.size // 3), rows)
                raise ChartDomainError(f"no containing sealed shaft tetrahedron for material rows {missing.tolist()}")
            ids, inverse = np.unique(corners.ravel(), return_inverse=True)
            points = self._atlas_points_m[ids]
            if shape is not self._reference:
                neighbors, stencil = _skin_stencil(self._reference.skin_tree, points)
                displacement = shape.vertices_m[neighbors] - self._reference.vertices_m[neighbors]
                points += np.einsum("nk,nkc->nc", stencil, displacement)
            nodes, _, node_jac = self._skin(points, self._atlas_weights[ids], transforms)
            indices = inverse.reshape(-1, 4)
            posed = nodes[indices]
            phi = np.einsum("nk,nkr->nr", weights, posed)
            jac = np.einsum("nk,nkrq->nrq", weights, node_jac[indices])
            posed_edges = np.swapaxes(posed[:, 1:] - posed[:, :1], -1, -2)
            gradient = posed_edges @ np.linalg.inv(label_edges)
            if np.any(np.linalg.det(gradient[..., [1, 0, 2]]) <= 0.0):
                raise ChartDomainError("posed material tetrahedron is folded or singular")
            longitudinal = gradient[:, :, 1].copy()
            longitudinal /= np.linalg.norm(longitudinal, axis=1, keepdims=True)
            angular = gradient[:, :, 0]
            angular = angular - np.sum(angular * longitudinal, axis=1, keepdims=True) * longitudinal
            angular /= np.linalg.norm(angular, axis=1, keepdims=True)
            frame = np.stack((longitudinal, angular, np.cross(longitudinal, angular)), axis=-1)
        batch = array.shape[:-1]
        return ((phi * _MM_PER_M).reshape(batch + (3,)), frame.reshape(batch + (3, 3)),
                (jac * _MM_PER_M).reshape(batch + (3, 4)))

    def _deform_canonical(self, q_t, points_mm, beta, *, gradient=False):
        """Neutral rest -> shaped rest -> LBS, carrying neutral stencil weights."""
        points = points_mm.reshape(-1, 3) / _MM_PER_M
        shape = self._neutral if beta is None or beta is NEUTRAL_BETA else self._shape(beta)
        neighbors, stencil, *derivative = _skin_stencil(self._neutral.skin_tree, points, gradient=gradient)
        vertex_weights = self._weights[neighbors]
        weights = np.einsum("nk,nkj->nj", stencil, vertex_weights)
        total = weights.sum(axis=1, keepdims=True)
        weights /= total
        if shape.beta.size:
            displacement = shape.vertices_m[neighbors] - self._neutral.vertices_m[neighbors]
            points += np.einsum("nk,nkc->nc", stencil, displacement)
        transforms, _ = self._joint_transforms(q_t, shape.joints_m)
        phi, blended, jac = self._skin(points, weights, transforms)
        if gradient:
            d_stencil = derivative[0]
            d_weights = np.einsum("nkj,nkc->njc", vertex_weights, d_stencil)
            d_weights -= weights[..., None] * d_weights.sum(axis=1, keepdims=True)
            d_weights /= total[..., None]
            gradient_mm = blended
            if shape.beta.size:
                gradient_mm = blended @ (np.eye(3) + np.einsum("nkr,nkc->nrc", displacement, d_stencil))
            rotation, translation, _, _ = transforms
            # Subtract the patient point: the stencil-gradient sum is zero.
            offsets = np.einsum("jrc,nc->njr", rotation - np.eye(3), points) + translation
            gradient_mm += np.einsum("njr,njc->nrc", offsets, d_weights)
            blended = gradient_mm
        return phi * _MM_PER_M, blended, jac * _MM_PER_M

    def forward_pose_canonical(self, q_t, points_mm, *, beta=None):
        """Transport beta=0 rest-world points; optional beta first shapes them."""
        array = _xyz_rows(points_mm, "points_mm")
        phi, blended, jac = self._deform_canonical(q_t, array, beta)
        u, _, vh = np.linalg.svd(blended)
        u[:, :, -1] *= np.where(np.linalg.det(u @ vh) < 0.0, -1.0, 1.0)[:, None]
        batch = array.shape[:-1]
        return phi.reshape(array.shape), (u @ vh).reshape(batch + (3, 3)), jac.reshape(batch + (3, 4))

    def pose_points_and_material_frame(self, beta, q_t, chart, tilde_xi):
        """Return posed XYZ/frame and exact local pose and material Jacobians."""
        tilde_xi = _xyz_rows(tilde_xi, "tilde_xi")
        xi = chart.xi_from_tilde_xi(tilde_xi)
        phi, dphi_dx, jq = self._deform_canonical(q_t, chart.xi_to_world(xi), beta, gradient=True)
        tangents = dphi_dx @ chart.dworld_dxi(xi).reshape(-1, 3, 3)
        longitudinal = tangents[:, :, 1].copy()
        longitudinal /= np.linalg.norm(longitudinal, axis=1, keepdims=True)
        angular = tangents[:, :, 0].copy()
        angular -= np.sum(angular * longitudinal, axis=1, keepdims=True) * longitudinal
        angular /= np.linalg.norm(angular, axis=1, keepdims=True)
        # The third Gram-Schmidt direction is chosen with positive orientation.
        normal = np.cross(longitudinal, angular)
        if np.any(~np.isfinite(normal)) or np.any(abs(np.sum(normal * tangents[:, :, 2], axis=1)) < 1.e-9):
            raise ChartDomainError("posed canonical chart has a singular material frame")
        batch = tilde_xi.shape[:-1]
        frame = np.stack((longitudinal, angular, normal), axis=-1)
        jxi = dphi_dx @ chart.dworld_dtilde_xi(tilde_xi).reshape(-1, 3, 3)
        return phi.reshape(batch + (3,)), frame.reshape(batch + (3, 3)), jq.reshape(batch + (3, 4)), jxi.reshape(batch + (3, 3))

    def forward_vertices(self, beta, q_t, vertex_indices) -> tuple[np.ndarray, np.ndarray]:
        """Exact model vertex ``(Phi_mm, J_q_mm_per_rad)`` with pose correctives.

        Other joints, global translation and expression are fixed to zero.
        This path uses the model's own shaped vertices and fixed LBS weights,
        not either of the interior extensions.
        """
        indices = np.asarray(vertex_indices)
        if (indices.dtype.kind not in "iu" or np.any(indices < 0)
                or np.any(indices >= len(self._template_m))):
            raise ValueError("vertex_indices must be valid integer SMPL-X vertex indices")
        ids = indices.ravel()
        shape = self._shape(beta)
        transforms, (local, d_local) = self._joint_transforms(q_t, shape.joints_m)
        dirs = self._pose_dirs[ids]
        correctives = dirs @ (local - np.eye(3)).ravel()
        d_correctives = dirs @ d_local.transpose(0, 2, 3, 1).reshape(18, 4)
        phi, _, jac = self._skin(shape.vertices_m[ids] + correctives, self._weights[ids],
                                 transforms, d_correctives)
        return ((phi * _MM_PER_M).reshape(indices.shape + (3,)),
                (jac * _MM_PER_M).reshape(indices.shape + (3, 4)))


@dataclass(frozen=True)
class MotionTruth:
    """Simulator-only current pose/velocity (rad, rad/s) and cardiac phase (rad)."""

    q: np.ndarray
    qdot: np.ndarray
    cardiac_phase_rad: float


@dataclass(frozen=True)
class TrackerEstimate:
    """Public pose estimate, conditional jitter covariance, and measurement time."""

    q_hat: np.ndarray
    Sigma_q: np.ndarray
    measurement_time_s: float


def _nonnegative(value, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return scalar


class DynamicMotionProfile:
    """Seeded external motion, lagged tracker, and a separate tissue-lag stream.

    Hip excursions are at most ±10 degrees, knees span 0–30 degrees, motion
    frequencies are 0.15–0.35 Hz, and the heart rate is 65–85 bpm. Setting
    ``moving=False`` freezes pose but does not disable cardiac motion.
    Tracker jitter accepts a scalar or four standard deviations in degrees,
    with an optional 4×4 correlation matrix. ``Sigma_q`` is the actual
    conditional covariance in rad²; latency is a deterministic mean error
    relative to current truth, not zero-mean noise or part of this covariance.
    The same seed/time gives the same tracker sample independent of query
    order. OU draws are an independent stream, reproducible for the same
    sequence of transitions and batch shapes.
    """

    def __init__(self, seed: int = 0, *, moving: bool = True, jitter_deg=2.0,
                 latency_s: float = 0.0, tracker_correlation=None,
                 ou_tau_s: float = 1.5, ou_stationary_std_mm=1.0) -> None:
        dynamics_seed, tracker_seed, ou_seed = np.random.SeedSequence(seed).spawn(3)
        rng = np.random.default_rng(dynamics_seed)
        self._tracker_key = tracker_seed.generate_state(4).tolist()
        self._ou_rng = np.random.default_rng(ou_seed)
        self.latency_s = _nonnegative(latency_s, "latency_s")
        self.ou_tau_s = _nonnegative(ou_tau_s, "ou_tau_s")
        if self.ou_tau_s == 0.0:
            raise ValueError("ou_tau_s must be positive")
        self._ou_std_mm = np.broadcast_to(np.asarray(ou_stationary_std_mm, dtype=float), (3,)).copy()
        if not np.isfinite(self._ou_std_mm).all() or np.any(self._ou_std_mm < 0.0):
            raise ValueError("ou_stationary_std_mm must be finite and nonnegative")

        self._amp = np.deg2rad([10.0, 10.0, 10.0, 15.0])
        self._offset = np.deg2rad([0.0, 0.0, 0.0, 15.0])
        if not moving:
            self._amp.fill(0.0)
            self._offset.fill(0.0)
        self._omega = _TAU * rng.uniform(0.15, 0.35, size=4)
        self._phase = rng.uniform(0.0, _TAU, size=4)
        self._heart_bpm = float(rng.uniform(65.0, 85.0))
        self._heart_phase = float(rng.uniform(0.0, _TAU))

        jitter = np.broadcast_to(np.asarray(jitter_deg, dtype=float), (4,))
        if not np.isfinite(jitter).all() or np.any(jitter < 0.0):
            raise ValueError("jitter_deg must be finite and nonnegative")
        correlation = np.eye(4) if tracker_correlation is None else np.asarray(tracker_correlation, dtype=float)
        if (correlation.shape != (4, 4) or not np.isfinite(correlation).all()
                or not np.allclose(correlation, correlation.T, rtol=0, atol=1.0e-12)
                or not np.allclose(np.diag(correlation), 1.0, rtol=0, atol=1.0e-12)):
            raise ValueError("tracker_correlation must be a symmetric 4x4 correlation matrix")
        eigenvalues, eigenvectors = np.linalg.eigh(correlation)
        if eigenvalues[0] < -1.0e-12:
            raise ValueError("tracker_correlation must be positive semidefinite")
        self._tracker_noise = (np.deg2rad(jitter)[:, None] * eigenvectors
                               * np.sqrt(np.maximum(eigenvalues, 0.0)))
        self._tracker_cov = self._tracker_noise @ self._tracker_noise.T

    @property
    def heart_rate_bpm(self) -> float:
        return self._heart_bpm

    def _pose_at(self, time_s: float) -> np.ndarray:
        return self._offset + self._amp * np.sin(self._omega * time_s + self._phase)

    def truth_at(self, time_s: float) -> MotionTruth:
        t = _nonnegative(time_s, "time_s")
        qdot = self._amp * self._omega * np.cos(self._omega * t + self._phase)
        phase = (self._heart_phase + _TAU * self._heart_bpm * t / 60.0) % _TAU
        return MotionTruth(self._pose_at(t), qdot, float(phase))

    def tracker_at(self, time_s: float) -> TrackerEstimate:
        t = _nonnegative(time_s, "time_s")
        # The synthetic trajectory extends smoothly into the latency pre-roll.
        measured_at = t - self.latency_s
        # Use exact binary timestamps rather than rounded frame numbers.
        time_key = struct.unpack("!II", struct.pack("!d", t + 0.0))
        rng = np.random.default_rng(np.random.SeedSequence([*self._tracker_key, *time_key]))
        q_hat = self._pose_at(measured_at) + self._tracker_noise @ rng.standard_normal(4)
        return TrackerEstimate(q_hat, self._tracker_cov.copy(), measured_at)

    def sample(self, time_s: float) -> tuple[MotionTruth, TrackerEstimate]:
        return self.truth_at(time_s), self.tracker_at(time_s)

    def ou_initial(self, batch_shape: tuple[int, ...] = ()) -> np.ndarray:
        """Draw a stationary OU initial state in local mm, one vector per item."""
        return self._ou_rng.standard_normal((*batch_shape, 3)) * self._ou_std_mm

    def ou_transition(self, dt_s: float) -> tuple[float, np.ndarray]:
        """Return scalar decay and exact three-dimensional innovation covariance."""
        rate = _nonnegative(dt_s, "dt_s") / self.ou_tau_s
        cov = np.diag(-np.expm1(-2.0 * rate) * self._ou_std_mm**2)
        return float(np.exp(-rate)), cov

    def ou_step(self, z_mm, dt_s: float) -> np.ndarray:
        z = _xyz_rows(z_mm, "z_mm")
        rate = _nonnegative(dt_s, "dt_s") / self.ou_tau_s
        if rate == 0.0:
            return z.copy()
        amplitude = np.sqrt(-np.expm1(-2.0 * rate)) * self._ou_std_mm
        return np.exp(-rate) * z + amplitude * self._ou_rng.standard_normal(z.shape)

    @staticmethod
    def vessel_response(truth: MotionTruth, f_contact_N: float, *, pulse_displacement_mm,
                        compression_compliance_mm_per_N, flexion_sliding_mm_per_rad) -> np.ndarray:
        """Deterministic local center displacement g (mm), excluding OU z.

        Supply signed 3-vectors or broadcastable batches of physical response
        coefficients in a common local frame. Positive compliance along an
        axis gives a negative shift under positive force. These coefficients
        are independent of task roles and are not ultrasound caliber/diameter
        responses. Add an independently evolved ``ou_step`` state for g+z.
        """
        force = _nonnegative(f_contact_N, "f_contact_N")
        pulse = _xyz_rows(pulse_displacement_mm, "pulse_displacement_mm")
        compliance = _xyz_rows(compression_compliance_mm_per_N, "compression_compliance_mm_per_N")
        sliding = _xyz_rows(flexion_sliding_mm_per_rad, "flexion_sliding_mm_per_rad")
        return np.cos(truth.cardiac_phase_rad) * pulse - force * compliance + truth.q[3] * sliding


__all__ = ["NEUTRAL_BETA", "SMPLXLegKinematics", "DynamicMotionProfile", "MotionTruth", "TrackerEstimate"]
