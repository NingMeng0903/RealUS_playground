"""Native SMPL-X surface and fixed material attachments for the existing atlas.

This adapter recovers the original male subject from preserved vertex IDs.
Skin geometry uses the actual SMPL-X shape/pose bases and 55-joint LBS, followed
by the production subdivision stencil. Internal anatomy uses a declared fixed
canonical surface attachment; it is not a validated tissue deformation model.
No raw licensed model is copied into this package.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from .atlas_scene import AtlasLeg, _points, _unit


class SmplxAtlasLeg(AtlasLeg):
    """Actual original-male SMPL-X atlas; a shape tuple controls two beta deltas.

    ``shape=(a,b)`` means beta0 += 5*(a-1), beta1 += 5*(b-1).
    It does *not* mean literal width and length scaling. ``knee_deg`` controls
    native right-knee joint 5 about its local X axis; all other local poses are
    zero. Canonical material coordinates and source IDs remain fixed.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        root = self.atlas_path.parents[6]
        self.model_path = root / "ref_code_library/InteractVLM/data/body_models/smplx/SMPLX_MALE.npz"
        production = self.atlas_path.parent.parent
        self.stencil_path = production / "material/surface_stencil_right.npz"
        self.base_atlas_path = production.parent / "bake/base_atlas/atlas_right.npz"
        for path in (self.model_path, self.stencil_path, self.base_atlas_path):
            if not path.is_file():
                raise FileNotFoundError(f"Native SMPL-X atlas dependency is missing: {path}")
        with np.load(self.model_path, allow_pickle=False) as model:
            self._template = np.asarray(model["v_template"], dtype=float)
            self._shapedirs = np.asarray(model["shapedirs"][..., :10], dtype=float)
            # Only joint 5 moves in this public interface. Retain its exact nine
            # native pose-corrective columns, rather than loading a whole model
            # into every pose cache. Flattening is native (R_1-I,...,R_54-I).
            self._knee_posedirs = np.asarray(model["posedirs"][..., 36:45], dtype=float)
            self._regressor = np.asarray(model["J_regressor"], dtype=float)
            self._weights = np.asarray(model["weights"], dtype=float)
            self._parents = np.asarray(model["kintree_table"][0], dtype=np.int64)
            self._parents[0] = -1
            self._model_faces = np.asarray(model["f"], dtype=np.int32)
        if self._template.shape != (10475, 3) or self._weights.shape != (10475, 55):
            raise ValueError("Expected the original 10475-vertex, 55-joint SMPL-X model")
        with np.load(self.base_atlas_path, allow_pickle=False) as base:
            source_ids = np.asarray(base["full_vertex_indices"], dtype=int)
            source_points = np.asarray(base["skin_vertices"], dtype=float)
            source_joints = np.stack([base[k] for k in ("pelvis", "hip", "knee", "ankle")])
        fit_matrix = self._shapedirs[source_ids].reshape(-1, 10)
        self.betas = np.linalg.lstsq(
            fit_matrix, (source_points - self._template[source_ids]).reshape(-1), rcond=None)[0]
        self._native_canonical = self._template + np.einsum("vij,j->vi", self._shapedirs, self.betas)
        self._canonical_joints = self._regressor @ self._native_canonical
        self._subject_fit_errors = np.linalg.norm(self._native_canonical[source_ids] - source_points, axis=1)
        self._joint_fit_errors = np.linalg.norm(self._canonical_joints[[0, 2, 5, 8]] - source_joints, axis=1)
        if max(self._subject_fit_errors.max(), self._joint_fit_errors.max()) > 1e-5:
            raise ValueError("Raw male SMPL-X does not reproduce the atlas's preserved subject IDs")
        with np.load(self.stencil_path, allow_pickle=False) as stencil:
            self._source_ids = np.asarray(stencil["source_full_vertex_indices"], dtype=int)
            if not np.array_equal(self._source_ids, source_ids):
                raise ValueError("Production stencil IDs differ from the base atlas IDs")
            self._stencil = csr_matrix((stencil["stencil_weights"], stencil["stencil_indices"],
                                       stencil["stencil_indptr"]), shape=(len(self.skin_vertices), len(source_ids)))
            if not np.array_equal(stencil["surface_faces"], self.skin_faces):
                raise ValueError("Production surface topology differs from the stored stencil")
        self._stencil_fit_errors = np.linalg.norm(
            self._stencil @ self._native_canonical[self._source_ids] - self.skin_vertices, axis=1)
        if np.max(self._stencil_fit_errors) > 1e-5:
            raise ValueError("Native SMPL-X + production stencil does not reproduce atlas skin")
        # A fixed canonical right-leg surface supplies interior attachments.
        # No nearest search is performed in the posed body during forward warp.
        self._anchor_ids = self._source_ids
        self._anchor_rest = self._native_canonical[self._anchor_ids]
        self._anchor_tree = cKDTree(self._anchor_rest)
        self._pose_cache: dict[tuple[float, float, float], dict[str, Any]] = {}
        self._last_inverse_max_m = 0.0
        self.deformation_model = "native_SMPLX55_surface_with_fixed_canonical_interior_attachment"
        self.recovered_assets_dir = Path(__file__).parent / "smplx_recovered_assets"
        self._export_canonical_assets()

    def _shape_betas(self, shape: tuple[float, float]) -> np.ndarray:
        if len(shape) != 2 or not np.all(np.isfinite(shape)):
            raise ValueError("shape must contain two finite beta-control values")
        beta = self.betas.copy()
        beta[:2] += 5 * (np.asarray(shape, dtype=float) - 1)
        return beta

    @staticmethod
    def _knee_rotation(knee_deg: float) -> np.ndarray:
        if not np.isfinite(knee_deg) or not 0 <= knee_deg <= 90:
            raise ValueError("Native benchmark knee flexion must be within [0,90] degrees")
        angle = np.deg2rad(knee_deg)
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])

    def _state(self, knee_deg: float, shape: tuple[float, float]) -> dict[str, Any]:
        key = (float(knee_deg), float(shape[0]), float(shape[1]))
        cached = self._pose_cache.get(key)
        if cached is not None:
            return cached
        beta = self._shape_betas(shape)
        knee_rotation = self._knee_rotation(knee_deg)
        rest = self._template + np.einsum("vij,j->vi", self._shapedirs, beta)
        joints = self._regressor @ rest
        global_mats = np.zeros((55, 4, 4))
        transforms = np.zeros_like(global_mats)
        for j in range(55):
            parent = self._parents[j]
            local = np.eye(4)
            local[:3, :3] = knee_rotation if j == 5 else np.eye(3)
            local[:3, 3] = joints[j] if parent < 0 else joints[j] - joints[parent]
            global_mats[j] = local if parent < 0 else global_mats[parent] @ local
            inverse_rest = np.eye(4)
            inverse_rest[:3, 3] = -joints[j]
            transforms[j] = global_mats[j] @ inverse_rest
        corrective = np.einsum("vij,j->vi", self._knee_posedirs, (knee_rotation - np.eye(3)).reshape(9))
        vertex_transforms = np.einsum("vj,jab->vab", self._weights, transforms)
        posed = np.einsum("vab,vb->va", vertex_transforms[:, :3, :3], rest + corrective) + vertex_transforms[:, :3, 3]
        skin = self._stencil @ posed[self._source_ids]
        normals = self._mesh_normals(skin, self.skin_faces)
        anchor_rotation = vertex_transforms[self._anchor_ids, :3, :3]
        anchor_posed = posed[self._anchor_ids]
        # F(p)=sum_i alpha_i(p) [R_i (p - anchor_i) + posed_anchor_i].
        # alpha is a function of canonical material position only.
        anchor_offset = anchor_posed - np.einsum("vab,vb->va", anchor_rotation, self._anchor_rest)
        cached = {"betas": beta, "rest_vertices": rest, "rest_joints": joints,
                  "joint_global": global_mats, "joint_transforms": transforms,
                  "native_vertices": posed, "surface_vertices": skin, "surface_normals": normals,
                  "anchor_rotation": anchor_rotation, "anchor_offset": anchor_offset,
                  "anchor_posed": anchor_posed, "posed_anchor_tree": cKDTree(anchor_posed),
                  "posed_skin_tree": cKDTree(skin)}
        self._pose_cache[key] = cached
        return cached

    @staticmethod
    def _mesh_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        tri = vertices[faces]
        face_normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normals = np.zeros_like(vertices)
        for j in range(3):
            np.add.at(normals, faces[:, j], face_normal)
        return _unit(normals)

    def _affine(self, points: np.ndarray, state: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Smooth fixed canonical attachment, without nearest-set jumps.

        All 523 preserved right-leg anchors participate through a 35 mm Gaussian
        and regularized inverse-square distance. Exponent shifting prevents
        underflow for off-surface points. Chunking bounds working memory.
        """
        rotations = np.empty((len(points), 3, 3))
        offsets = np.empty((len(points), 3))
        for begin in range(0, len(points), 2048):
            end = min(begin + 2048, len(points))
            distance2 = np.sum((points[begin:end, None] - self._anchor_rest[None]) ** 2, axis=-1)
            exponent = -distance2 / (2 * .035 ** 2)
            exponent -= exponent.max(axis=1, keepdims=True)
            weights = np.exp(exponent) / (distance2 + .002 ** 2)
            weights /= weights.sum(axis=1, keepdims=True)
            rotations[begin:end] = (weights @ state["anchor_rotation"].reshape(-1, 9)).reshape(-1, 3, 3)
            offsets[begin:end] = weights @ state["anchor_offset"]
        return rotations, offsets

    def _interior_forward(self, points: np.ndarray, state: dict[str, Any]) -> np.ndarray:
        rotation, offset = self._affine(points, state)
        return np.einsum("nab,nb->na", rotation, points) + offset

    def deform_points(self, points: np.ndarray, knee_deg: float = 0,
                      shape: tuple[float, float] = (1., 1.)) -> np.ndarray:
        p, out_shape = _points(points)
        state = self._state(knee_deg, shape)
        if knee_deg == 0 and tuple(shape) == (1., 1.):
            return p.copy().reshape(out_shape)
        # Surface vertices have exact production lineage, so preserve it rather
        # than replacing it with the approximate interior attachment.
        distance, index = self._skin_tree.query(p)
        on_stored_skin = distance < 1e-9
        out = np.empty_like(p)
        out[~on_stored_skin] = self._interior_forward(p[~on_stored_skin], state)
        out[on_stored_skin] = state["surface_vertices"][index[on_stored_skin]]
        return out.reshape(out_shape)

    def undeform_points(self, points: np.ndarray, knee_deg: float = 0,
                        shape: tuple[float, float] = (1., 1.)) -> np.ndarray:
        target, out_shape = _points(points)
        state = self._state(knee_deg, shape)
        if knee_deg == 0 and tuple(shape) == (1., 1.):
            self._last_inverse_max_m = 0.
            return target.copy().reshape(out_shape)
        _, nearest = state["posed_anchor_tree"].query(target)
        rotation = state["anchor_rotation"][nearest]
        p = self._anchor_rest[nearest] + np.linalg.solve(
            rotation, (target - state["anchor_posed"][nearest])[..., None])[..., 0]
        # Re-evaluate only canonical attachments while solving the inverse.
        # Each iteration is an inverse of the currently blended LBS affine map.
        for _ in range(40):
            rotation, offset = self._affine(p, state)
            next_p = np.linalg.solve(rotation, (target - offset)[..., None])[..., 0]
            if np.max(np.linalg.norm(next_p - p, axis=1), initial=0) < 1e-9:
                p = next_p
                break
            p = .75 * next_p + .25 * p
        error = np.linalg.norm(self._interior_forward(p, state) - target, axis=1)
        # Native surface nodes have an exact inverse via stored lineage.
        surface_distance, surface_index = state["posed_skin_tree"].query(target)
        on_stored_skin = surface_distance < 1e-9
        p[on_stored_skin] = self.skin_vertices[surface_index[on_stored_skin]]
        error[on_stored_skin] = 0
        self._last_inverse_max_m = float(np.max(error, initial=0))
        if self._last_inverse_max_m > 5e-4:
            raise ValueError(f"SMPL-X internal attachment inverse exceeds 0.5 mm: {self._last_inverse_max_m:.6g} m")
        return p.reshape(out_shape)

    def posed_skin(self, knee_deg: float = 0, shape: tuple[float, float] = (1., 1.)) -> tuple[np.ndarray, np.ndarray]:
        """Exact production stencil applied after native SMPL-X surface LBS."""
        state = self._state(knee_deg, shape)
        return state["surface_vertices"].copy(), self.skin_faces.copy()

    def native_smplx_vertices(self, knee_deg: float = 0,
                              shape: tuple[float, float] = (1., 1.)) -> np.ndarray:
        return self._state(knee_deg, shape)["native_vertices"].copy()

    def skin_point_and_normal(self, material_progress: np.ndarray | float,
                              knee_deg: float = 0, shape: tuple[float, float] = (1., 1.)):
        canonical = self._curve(self.skin_reference_points, material_progress)
        p, out_shape = _points(canonical)
        distance, index = self._skin_tree.query(p, k=8)
        weights = 1 / np.maximum(distance, 1e-10) ** 2
        weights /= weights.sum(axis=1, keepdims=True)
        state = self._state(knee_deg, shape)
        # Fixed canonical skin-route attachment to the exact posed mesh.
        displacement = state["surface_vertices"][index] - self.skin_vertices[index]
        point = p + np.einsum("nk,nka->na", weights, displacement)
        normal = _unit(np.einsum("nk,nka->na", weights, state["surface_normals"][index]))
        return point.reshape(out_shape), normal.reshape(out_shape)

    def skin_reference_curve(self, progress: np.ndarray | float, knee_deg: float = 0,
                             shape: tuple[float, float] = (1., 1.)) -> np.ndarray:
        return self.skin_point_and_normal(progress, knee_deg, shape)[0]

    def contact_query(self, world_point: np.ndarray, knee_deg: float = 0,
                      shape: tuple[float, float] = (1., 1.)) -> dict[str, np.ndarray]:
        target, out_shape = _points(world_point)
        state = self._state(knee_deg, shape)
        _, index = state["posed_skin_tree"].query(target)
        point = state["surface_vertices"][index]
        normal = state["surface_normals"][index]
        signed = np.sum((target - point) * normal, axis=1)
        return {"point": point.reshape(out_shape), "normal": normal.reshape(out_shape),
                "signed_distance_m": signed.reshape(out_shape[:-1]), "skin_vertex_index": index.reshape(out_shape[:-1])}

    def _export_canonical_assets(self) -> None:
        """Write derived rest assets and weights; do not copy the raw model."""
        out = self.recovered_assets_dir
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "recovered_atlas_betas.npy", self.betas)
        for filename, vertices in (("smpl_canonical_tpose.obj", self._native_canonical),
                                   ("smpl_canonical_tpose_neutral.obj", self._template)):
            # 'neutral' in this legacy filename means zero betas in the *same
            # male model*, not the different gender-neutral model file.
            with (out / filename).open("w") as stream:
                stream.write("# Derived original-male SMPL-X rest mesh; source model is not copied.\n")
                np.savetxt(stream, vertices, fmt="v %.10f %.10f %.10f")
                np.savetxt(stream, self._model_faces + 1, fmt="f %d %d %d")
        inverse_bind = np.broadcast_to(np.eye(4), (55, 4, 4)).copy()
        inverse_bind[:, :3, 3] = -self._canonical_joints
        joint_names = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
                       "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
                       "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
                       "left_elbow", "right_elbow", "left_wrist", "right_wrist", "jaw", "left_eye_smplhf", "right_eye_smplhf"]
        for side in ("left", "right"):
            joint_names.extend(f"{side}_{finger}{joint}" for finger in ("index", "middle", "pinky", "ring", "thumb")
                               for joint in (1, 2, 3))
        np.savez_compressed(out / "smpl_canonical_weights.npz", lbs_weights=self._weights.astype(np.float32),
                            faces=self._model_faces, parents=self._parents, rest_joints=self._canonical_joints,
                            inverse_bind=inverse_bind, joint_names=np.asarray(joint_names))
        skeleton = {"model_type": "smplx", "gender": "male", "joint_names": joint_names,
                    "parents": self._parents.tolist(), "rest_joints_subject": self._canonical_joints.tolist(),
                    "rest_joints_neutral": (self._regressor @ self._template).tolist(), "inverse_bind": inverse_bind.tolist()}
        (out / "smpl_canonical_skeleton.json").write_text(json.dumps(skeleton, indent=2))
        digest = hashlib.sha256()
        with self.model_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        self.model_sha256 = digest.hexdigest()
        manifest = {"model_type": "smplx", "gender": "male", "model_path": str(self.model_path),
                    "model_sha256": self.model_sha256, "betas": self.betas.tolist(),
                    "beta_source": "least_squares_recovery_from_preserved_base_atlas_vertex_IDs",
                    "vertex_fit_max_m": float(self._subject_fit_errors.max()),
                    "subject_obj": str(out / "smpl_canonical_tpose.obj"),
                    "neutral_obj": str(out / "smpl_canonical_tpose_neutral.obj"),
                    "weights_npz": str(out / "smpl_canonical_weights.npz"),
                    "skeleton_json": str(out / "smpl_canonical_skeleton.json"),
                    "neutral_filename_semantics": "zero_betas_same_male_model",
                    "raw_model_copied": False, "internal_deformation": "fixed_canonical_surface_attachment_not_tissue_physics"}
        (out / "source_manifest.json").write_text(json.dumps(manifest, indent=2))

    def audit(self) -> dict[str, Any]:
        result = super().audit()
        result.update({
            "deformation_model": self.deformation_model,
            "model_path": str(self.model_path), "model_sha256": self.model_sha256,
            "model_gender": "male", "native_vertex_count": 10475, "native_joint_count": 55,
            "recovered_betas": self.betas.tolist(),
            "subject_vertex_recovery_max_m": float(self._subject_fit_errors.max()),
            "subject_vertex_recovery_rms_m": float(np.sqrt(np.mean(self._subject_fit_errors ** 2))),
            "subject_joint_recovery_max_m": float(self._joint_fit_errors.max()),
            "production_stencil_surface_max_m": float(self._stencil_fit_errors.max()),
            "surface_pose_model": "actual_SMPLX_shape_pose_correctives_FK_LBS_then_exact_production_CSR_stencil",
            "shape_parameters": "beta0_plus_5_times_shape0_minus1_and_beta1_plus_5_times_shape1_minus1_not_literal_scales",
            "internal_pose_model": "fixed_canonical_smooth_523_surface_anchor_LBS_affine_interpolation",
            "internal_attachment_kernel": "35mm_Gaussian_times_inverse_square_with_2mm_regularizer",
            "internal_pose_is_tissue_physics": False,
            "inverse_model": "posed_anchor_seed_then_iterated_canonical_blended_affine_inverse",
            "last_inverse_forward_residual_max_m": self._last_inverse_max_m,
            "recovered_assets_dir": str(self.recovered_assets_dir),
        })
        return result
