"""Bounded, offline left-arm pose correction fitting for CompiledAnatomyV14.

This module owns the small pose-dependent part of the V14 arm experiment.  A
compiled subject supplies the frozen 235-controller rig and sparse weights;
this code caches one source global frame and one target parent-local frame per
pose, evaluates only a requested vertex subset during Powell trials, and
finally hands the fitted local twists to :mod:`pose_corrector_v14`.

The optimizer is intentionally outside ``PoseCorrectorV14``.  The corrector
only stores/evaluates a fixed-radius RBF at run time.  This separation keeps
the runtime deterministic and makes it impossible for an AMASS pose or a
geometry query to become a hidden training sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import igl
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .chain_rest_fit_v1 import _global_to_local
from .pose_map_v1 import _fk
from .pose_corrector_v14 import (
    DEFAULT_MAX_ROTATION_NORM,
    DEFAULT_MAX_TRANSLATION_NORM,
    PoseCorrectionAmplitudeError,
    PoseCorrectionSupportError,
    PoseCorrectorV14,
)


ARM_CONTROLLER_NAMES: tuple[str, ...] = (
    "Shoulder_Rotate_L",
    "Elbow_Rot_L",
    "Wrist_Rotate_L",
)
ARM_CONTROLLER_IDS = np.asarray((131, 132, 135), dtype=np.int64)
ARM_SELECTED_JOINT_IDS = np.asarray((13, 16, 18, 20), dtype=np.int64)
ARM_RBF_RADIUS = 3.0
ARM_MAX_ROTATION_NORM = DEFAULT_MAX_ROTATION_NORM
ARM_MAX_TRANSLATION_NORM = DEFAULT_MAX_TRANSLATION_NORM
DEFAULT_MAXFEV = 300
DEFAULT_SLSQP_MAXITER = 60

SKIN_EXCESS_LIMIT_M = 1.0e-3
JOINT_PENETRATION_LIMIT_M = 5.0e-4
JOINT_GAP_LIMIT_M = 3.0e-3


def _finite(value: Any, *, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result.copy()


def _pose(value: Any, *, name: str = "pose") -> np.ndarray:
    result = _finite(value, name=name, shape=(55, 3))
    return result


def _ids(value: Any, *, name: str, count: int, allow_empty: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if not allow_empty and len(raw) == 0:
        raise ValueError(f"{name} must not be empty")
    result = np.asarray(raw, dtype=np.int64).copy()
    if len(np.unique(result)) != len(result):
        raise ValueError(f"{name} contains duplicate IDs")
    if np.any(result < 0) or np.any(result >= count):
        raise ValueError(f"{name} contains an out-of-range ID for [0, {count})")
    return result


def _as_json(value: Any) -> Any:
    """Convert NumPy/scalar values to strict JSON values for diagnostics."""
    if isinstance(value, Mapping):
        return {str(key): _as_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_as_json(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        result = float(value)
        return result if np.isfinite(result) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _twist_array(value: Any, *, name: str = "local_twists") -> np.ndarray:
    result = _finite(value, name=name, shape=(len(ARM_CONTROLLER_IDS), 6))
    return result


def _check_twist_amplitude(
    twists: np.ndarray,
    *,
    max_rotation_norm: float = ARM_MAX_ROTATION_NORM,
    max_translation_norm: float = ARM_MAX_TRANSLATION_NORM,
    name: str = "local_twists",
) -> None:
    values = _twist_array(twists, name=name)
    rotation_norm = np.linalg.norm(values[:, :3], axis=1)
    translation_norm = np.linalg.norm(values[:, 3:], axis=1)
    rot_limit = float(max_rotation_norm) + max(1.0e-12, abs(float(max_rotation_norm)) * 1.0e-10)
    trans_limit = float(max_translation_norm) + max(1.0e-12, abs(float(max_translation_norm)) * 1.0e-10)
    if np.any(rotation_norm > rot_limit):
        index = int(np.flatnonzero(rotation_norm > rot_limit)[0])
        raise PoseCorrectionAmplitudeError(
            f"{name} rotation norm exceeds {float(max_rotation_norm):.9g} rad "
            f"at controller {int(ARM_CONTROLLER_IDS[index])}: {rotation_norm[index]:.9g}"
        )
    if np.any(translation_norm > trans_limit):
        index = int(np.flatnonzero(translation_norm > trans_limit)[0])
        raise PoseCorrectionAmplitudeError(
            f"{name} translation norm exceeds {float(max_translation_norm):.9g} m "
            f"at controller {int(ARM_CONTROLLER_IDS[index])}: {translation_norm[index]:.9g}"
        )


def _twist_matrix(twist: np.ndarray) -> np.ndarray:
    values = _twist_array(twist, name="local_twists")
    result = np.tile(np.eye(4, dtype=np.float64), (len(values), 1, 1))
    result[:, :3, :3] = Rotation.from_rotvec(values[:, :3]).as_matrix()
    result[:, :3, 3] = values[:, 3:]
    return result


def _subset_lbs(
    rest_vertices: np.ndarray,
    vertex_ids: np.ndarray,
    driver_indices: np.ndarray,
    driver_weights: np.ndarray,
    posed_globals: np.ndarray,
    inverse_bind: np.ndarray,
) -> np.ndarray:
    """Apply the unchanged authored sparse weights only to ``vertex_ids``."""
    ids = np.asarray(vertex_ids, dtype=np.int64)
    points = np.asarray(rest_vertices, dtype=np.float64)[ids]
    indices = np.asarray(driver_indices, dtype=np.int64)[ids]
    weights = np.asarray(driver_weights, dtype=np.float64)[ids]
    if indices.shape != weights.shape or indices.shape[0] != len(points):
        raise ValueError("subset sparse drivers do not match the requested vertices")
    transforms = np.asarray(posed_globals, dtype=np.float64) @ np.asarray(
        inverse_bind, dtype=np.float64
    )
    selected = transforms[indices]
    mapped = (
        np.einsum("nsij,nj->nsi", selected[:, :, :3, :3], points)
        + selected[:, :, :3, 3]
    )
    return np.sum(mapped * weights[:, :, None], axis=1)


def _apply_local_postdelta(
    base_local: np.ndarray,
    twists: np.ndarray,
    parents: np.ndarray,
) -> np.ndarray:
    """Apply the three local post-deltas and reconstruct one complete FK."""
    _check_twist_amplitude(twists)
    local = np.asarray(base_local, dtype=np.float64).copy()
    deltas = _twist_matrix(twists)
    local[ARM_CONTROLLER_IDS] = local[ARM_CONTROLLER_IDS] @ deltas
    return _fk(local, np.asarray(parents, dtype=np.int64))


def _is_identity_selected_pose(pose: np.ndarray) -> bool:
    selected = np.asarray(pose, dtype=np.float64)[ARM_SELECTED_JOINT_IDS]
    rotations = Rotation.from_rotvec(selected).as_matrix()
    return bool(np.allclose(rotations, np.eye(3)[None], atol=1.0e-6, rtol=0.0))


@dataclass
class ArmPoseGeometryV14:
    """SMPL-X geometry associated with one explicit fit or regression pose."""

    name: str
    pose: np.ndarray
    skin_vertices: np.ndarray
    skin_faces: np.ndarray
    smplx_joints: np.ndarray | None = None
    source_vertices: np.ndarray | None = None
    vertex_tissue: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = str(self.name)
        if not self.name:
            raise ValueError("pose geometry name must be non-empty")
        self.pose = _pose(self.pose)
        self.skin_vertices = _finite(self.skin_vertices, name="skin_vertices")
        if self.skin_vertices.ndim != 2 or self.skin_vertices.shape[1] != 3:
            raise ValueError("skin_vertices must have shape [N, 3]")
        self.skin_faces = np.asarray(self.skin_faces)
        if self.skin_faces.ndim != 2 or self.skin_faces.shape[1] != 3:
            raise ValueError("skin_faces must have shape [F, 3]")
        if self.skin_faces.dtype.kind not in "iu":
            raise ValueError("skin_faces must contain integer indices")
        self.skin_faces = np.asarray(self.skin_faces, dtype=np.int64).copy()
        if np.any(self.skin_faces < 0) or np.any(self.skin_faces >= len(self.skin_vertices)):
            raise ValueError("skin_faces contain out-of-range indices")
        if len(self.skin_faces) == 0:
            raise ValueError("skin_faces must be non-empty")
        if self.smplx_joints is not None:
            self.smplx_joints = _finite(self.smplx_joints, name="smplx_joints", shape=(55, 3))
        if self.source_vertices is not None:
            self.source_vertices = _finite(self.source_vertices, name="source_vertices")
            if self.source_vertices.ndim != 2 or self.source_vertices.shape[1] != 3:
                raise ValueError("source_vertices must have shape [N, 3]")
        if self.vertex_tissue is not None:
            self.vertex_tissue = np.asarray(self.vertex_tissue).copy()
            if self.vertex_tissue.ndim != 1:
                raise ValueError("vertex_tissue must be one-dimensional")


@dataclass
class ArmPoseFrameV14:
    """Cached source motion and target parent-local motion for one pose."""

    name: str
    pose: np.ndarray
    source_global: np.ndarray
    base_target_local: np.ndarray
    geometry: ArmPoseGeometryV14


def cache_arm_pose_frames_v14(
    compiled: Any,
    poses: Mapping[str, Any],
    geometries: Mapping[str, ArmPoseGeometryV14] | None = None,
) -> dict[str, ArmPoseFrameV14]:
    """Cache sourceG and base target local poses once per caller-supplied pose."""
    if not isinstance(poses, Mapping) or not poses:
        raise ValueError("poses must be a non-empty name-to-[55,3] mapping")
    asset = compiled.source_asset
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    if len(parents) != 235:
        raise ValueError("compiled source rig must contain 235 controllers")
    result: dict[str, ArmPoseFrameV14] = {}
    for raw_name, raw_pose in poses.items():
        name = str(raw_name)
        if name in result:
            raise ValueError(f"duplicate pose name: {name}")
        pose = _pose(raw_pose, name=f"pose[{name}]")
        geometry = None if geometries is None else geometries.get(name)
        if geometry is None:
            geometry = ArmPoseGeometryV14(
                name=name,
                pose=pose,
                skin_vertices=np.zeros((3, 3), dtype=np.float64),
                skin_faces=np.asarray([[0, 1, 2]], dtype=np.int64),
            )
        if not np.allclose(geometry.pose, pose, atol=2.0e-7, rtol=0.0):
            raise ValueError(f"geometry pose differs from requested pose: {name}")
        # Use the same compiled source-motion authority as runtime replay.  In
        # particular, a compiled subject may carry an authenticated effective
        # motion asset/driver response that is not identical to the original
        # ``source_asset`` compatibility view.
        source_global = np.asarray(compiled.source_globals(pose), dtype=np.float64)
        base_target_global = np.asarray(compiled.globals_from_source(source_global), dtype=np.float64)
        base_target_local = _global_to_local(base_target_global, parents)
        if base_target_local.shape != (235, 4, 4) or not np.all(np.isfinite(base_target_local)):
            raise ValueError(f"invalid cached target local pose: {name}")
        result[name] = ArmPoseFrameV14(
            name=name,
            pose=pose,
            source_global=source_global,
            base_target_local=base_target_local,
            geometry=geometry,
        )
    return result


def evaluate_arm_twists_v14(
    compiled: Any,
    frame: ArmPoseFrameV14,
    twists: Any,
    vertex_ids: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one bounded local correction using float64 FK + subset LBS."""
    values = _twist_array(twists)
    ids = _ids(vertex_ids, name="vertex_ids", count=len(compiled.target_rest), allow_empty=False)
    globals_ = _apply_local_postdelta(
        frame.base_target_local, values, np.asarray(compiled.parents, dtype=np.int64)
    )
    positions = _subset_lbs(
        np.asarray(compiled.target_rest, dtype=np.float64),
        ids,
        np.asarray(compiled.indices, dtype=np.int64),
        np.asarray(compiled.weights, dtype=np.float64),
        globals_,
        np.asarray(compiled.target_inverse, dtype=np.float64),
    )
    if not np.all(np.isfinite(positions)):
        raise ValueError("subset LBS produced non-finite positions")
    return positions, globals_


@dataclass
class _MeshV14:
    ids: np.ndarray
    faces: np.ndarray


class ArmScoreContextV14:
    """Frozen mesh/domain queries used by every objective evaluation."""

    def __init__(self, compiled: Any, calibration: Any):
        self.compiled = compiled
        self.asset = compiled.source_asset
        self.calibration = calibration
        self.names = list(self.asset.source_bone_names)
        self.meshes: dict[str, _MeshV14] = {}
        ranges = np.asarray(self.asset.source_vertex_ranges, dtype=np.int64)
        tissues = list(self.asset.source_tissues or ())
        faces_global = np.asarray(self.asset.faces, dtype=np.int64)
        for mesh_name, (start, stop) in zip(self.asset.source_mesh_names, ranges.tolist()):
            ids = np.arange(int(start), int(stop), dtype=np.int64)
            face_mask = np.all((faces_global >= int(start)) & (faces_global < int(stop)), axis=1)
            local_faces = faces_global[face_mask] - int(start)
            if len(ids) and len(local_faces):
                self.meshes[str(mesh_name)] = _MeshV14(ids=ids, faces=local_faces)
        required_meshes = ("Humerus_L", "Radius_L", "Ulna_L", "Scaphoid_L")
        missing = [name for name in required_meshes if name not in self.meshes]
        if missing:
            raise ValueError(f"left-arm scoring meshes are missing: {missing}")
        self.arm_mesh_names = ("Humerus_L", "Radius_L", "Ulna_L")
        self.arm_bone_ids = np.unique(np.concatenate([self.meshes[name].ids for name in self.arm_mesh_names]))

        wrist_root = self.names.index("Wrist_Rotate_L")
        hand_controllers: set[int] = set()
        parents = np.asarray(self.asset.source_bone_parents, dtype=np.int64)
        for controller in range(len(parents)):
            probe = controller
            while probe >= 0 and probe != wrist_root:
                probe = int(parents[probe])
            if probe == wrist_root:
                hand_controllers.add(controller)
        hand_meshes: list[str] = []
        for mesh_name, tissue, owner in zip(
            self.asset.source_mesh_names,
            tissues,
            np.asarray(self.asset.source_mesh_controller_bones, dtype=np.int64).tolist(),
        ):
            if str(tissue).lower() == "bone" and int(owner) in hand_controllers:
                if str(mesh_name) not in self.arm_mesh_names and str(mesh_name) in self.meshes:
                    hand_meshes.append(str(mesh_name))
        self.hand_mesh_names = tuple(hand_meshes)
        # The first arm bake used 30 samples per hand mesh.  Rest5 showed that
        # this can miss the worst distal phalanx/skin violation, so the
        # objective now uses every hand-bone vertex (about 9k points in the
        # frozen subject).  Mesh ownership still provides the strata for the
        # diagnostic count; there is no random sampling or hidden resampling.
        hand_vertices = [self.meshes[name].ids for name in self.hand_mesh_names]
        self.hand_sample_ids = (
            np.unique(np.concatenate(hand_vertices)) if hand_vertices else np.empty(0, dtype=np.int64)
        )

        domains = getattr(calibration, "domains", None)
        if not isinstance(domains, Mapping):
            raise ValueError("calibration must expose frozen domains")
        self.domain_ids: dict[str, np.ndarray] = {}
        for key in (
            "elbow/left/humerus.fit",
            "elbow/left/radius.fit",
            "elbow/left/ulna.fit",
            "calibration/left/wrist/radius.fit",
            "calibration/left/wrist/ulna.fit",
            "calibration/left/wrist/hand.fit",
        ):
            if key not in domains:
                raise ValueError(f"frozen calibration domain is missing: {key}")
            self.domain_ids[key] = _ids(
                domains[key], name=f"calibration domain {key}",
                count=len(self.asset.vertices_rest), allow_empty=False,
            )

        target_mesh_ids = np.unique(
            np.concatenate(
                [
                    self.meshes[name].ids
                    for name in ("Radius_L", "Ulna_L", "Scaphoid_L")
                ]
            )
        )
        domain_query_ids = np.unique(
            np.concatenate(
                list(self.domain_ids.values())
                + ([self.hand_sample_ids] if len(self.hand_sample_ids) else [])
            )
        )
        self.score_ids = np.unique(
            np.concatenate((self.arm_bone_ids, target_mesh_ids, domain_query_ids))
        )
        if not len(self.score_ids):
            raise ValueError("left-arm score subset is empty")
        self._score_lookup = np.full(len(self.asset.vertices_rest), -1, dtype=np.int64)
        self._score_lookup[self.score_ids] = np.arange(len(self.score_ids), dtype=np.int64)

    def _positions(self, vertices: np.ndarray, ids: np.ndarray) -> np.ndarray:
        rows = self._score_lookup[np.asarray(ids, dtype=np.int64)]
        if np.any(rows < 0):
            raise ValueError("score query references a vertex outside the frozen subset")
        return np.asarray(vertices, dtype=np.float64)[rows]

    def _mesh_positions(self, vertices: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
        mesh = self.meshes[name]
        return self._positions(vertices, mesh.ids), mesh.faces

    @staticmethod
    def _signed(points: np.ndarray, target_vertices: np.ndarray, target_faces: np.ndarray) -> np.ndarray:
        values = igl.signed_distance(
            np.ascontiguousarray(points, dtype=np.float64),
            np.ascontiguousarray(target_vertices, dtype=np.float64),
            np.ascontiguousarray(target_faces, dtype=np.int64),
        )[0]
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (len(points),) or not np.all(np.isfinite(values)):
            raise ValueError("signed-distance query returned invalid values")
        return values

    @staticmethod
    def _skin_metrics(values: np.ndarray) -> dict[str, Any]:
        excess = np.maximum(np.asarray(values, dtype=np.float64) - SKIN_EXCESS_LIMIT_M, 0.0)
        outside = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
        return {
            "sample_count": int(len(values)),
            "max_signed_m": float(np.max(values)),
            "max_outside_m": float(np.max(outside)),
            "max_excess_over_1mm_m": float(np.max(excess)),
            "mean_excess_over_1mm_m": float(np.mean(excess)),
            "outside_count": int(np.count_nonzero(values > SKIN_EXCESS_LIMIT_M)),
            "passed": bool(float(np.max(excess)) <= 0.0),
        }

    @staticmethod
    def _joint_metrics(values: np.ndarray, *, require_contact: bool = True) -> dict[str, Any]:
        signed = np.asarray(values, dtype=np.float64)
        penetration = np.maximum(-signed - JOINT_PENETRATION_LIMIT_M, 0.0)
        gap = float(np.min(np.abs(signed)))
        return {
            "sample_count": int(len(signed)),
            "min_signed_m": float(np.min(signed)),
            "max_penetration_over_0_5mm_m": float(np.max(penetration)),
            "nearest_gap_m": gap,
            "contact_required": bool(require_contact),
            "passed": bool(
                float(np.max(penetration)) <= 0.0
                and (not require_contact or gap <= JOINT_GAP_LIMIT_M)
            ),
        }

    def score_pose(
        self,
        vertices: np.ndarray,
        geometry: ArmPoseGeometryV14,
        *,
        twists: np.ndarray,
    ) -> dict[str, Any]:
        positions = np.asarray(vertices, dtype=np.float64)
        if positions.shape != (len(self.score_ids), 3) or not np.all(np.isfinite(positions)):
            raise ValueError("score vertices must be finite [score_vertex_count, 3]")
        values_arm = self._signed(
            self._positions(positions, self.arm_bone_ids),
            geometry.skin_vertices,
            geometry.skin_faces,
        )
        skin_arm = self._skin_metrics(values_arm)
        if len(self.hand_sample_ids):
            values_hand = self._signed(
                self._positions(positions, self.hand_sample_ids),
                geometry.skin_vertices,
                geometry.skin_faces,
            )
            skin_hand = self._skin_metrics(values_hand)
            skin_hand["sampling"] = "full_hand_bone_vertices"
        else:
            skin_hand = {
                "sample_count": 0,
                "available": False,
                "passed": False,
                "reason": "no hand bone meshes available for stratified sampling",
            }

        joint: dict[str, Any] = {}
        humerus_query = self.domain_ids["elbow/left/humerus.fit"]
        for target_name, label in (("Radius_L", "humerus_to_radius"), ("Ulna_L", "humerus_to_ulna")):
            target_vertices, target_faces = self._mesh_positions(positions, target_name)
            values = self._signed(self._positions(positions, humerus_query), target_vertices, target_faces)
            joint[label] = self._joint_metrics(values)

        scaphoid_vertices, scaphoid_faces = self._mesh_positions(positions, "Scaphoid_L")
        wrist_query_rows = (
            # The radius scaphoid fossa is a direct contact constraint.  The
            # ulnar side is separated from the carpus by the TFCC; retain its
            # penetration diagnostic without falsely attracting it to the
            # scaphoid.  ``wrist/hand.fit`` is a domain on Scaphoid itself and
            # therefore is not a cross-surface query.
            ("radius_to_scaphoid", self.domain_ids["calibration/left/wrist/radius.fit"], True),
            ("ulna_to_scaphoid", self.domain_ids["calibration/left/wrist/ulna.fit"], False),
        )
        for label, query_ids, require_contact in wrist_query_rows:
            values = self._signed(self._positions(positions, query_ids), scaphoid_vertices, scaphoid_faces)
            joint[label] = self._joint_metrics(values, require_contact=require_contact)

        values = np.asarray(twists, dtype=np.float64)
        rotation_norms = np.linalg.norm(values[:, :3], axis=1)
        translation_norms = np.linalg.norm(values[:, 3:], axis=1)
        amplitude_prior = float(
            np.sum((rotation_norms / ARM_MAX_ROTATION_NORM) ** 2)
            + np.sum((translation_norms / ARM_MAX_TRANSLATION_NORM) ** 2)
        )
        return {
            "skin_arm": skin_arm,
            "skin_hand_full": skin_hand,
            # Keep the old key as an explicit alias for downstream report
            # readers; both entries refer to the same full-vertex query.
            "skin_hand_stratified": skin_hand,
            "joint_surfaces": joint,
            "deformation_amplitude_prior": amplitude_prior,
            "passed": bool(
                skin_arm.get("passed", False)
                and skin_hand.get("passed", False)
                and all(row.get("passed", False) for row in joint.values())
            ),
        }


def _objective_score(metrics: Mapping[str, Any]) -> float:
    """Dimensionless objective used by Powell; thresholds remain explicit in diagnostics."""
    total = 0.0
    for key in ("skin_arm", "skin_hand_full"):
        row = metrics[key]
        if row.get("available", True) is False:
            total += 100.0
            continue
        total += (float(row["max_excess_over_1mm_m"]) / SKIN_EXCESS_LIMIT_M) ** 2
        total += (float(row["mean_excess_over_1mm_m"]) / SKIN_EXCESS_LIMIT_M) ** 2
    for row in metrics["joint_surfaces"].values():
        total += (float(row["max_penetration_over_0_5mm_m"]) / JOINT_PENETRATION_LIMIT_M) ** 2
        if row.get("contact_required", True):
            total += (max(float(row["nearest_gap_m"]) - JOINT_GAP_LIMIT_M, 0.0) / JOINT_GAP_LIMIT_M) ** 2
    # Keep corrections small without making the prior strong enough to hide
    # an actual surface failure.
    total += 0.02 * float(metrics["deformation_amplitude_prior"])
    return float(total)


@dataclass
class ArmPoseFitResultV14:
    corrector: PoseCorrectorV14
    local_twists: np.ndarray
    fit_pose_names: tuple[str, ...]
    reports: dict[str, dict[str, Any]]
    frames: dict[str, ArmPoseFrameV14]
    score_ids: np.ndarray


def _fit_one_pose(
    compiled: Any,
    frame: ArmPoseFrameV14,
    scorer: ArmScoreContextV14,
    *,
    maxfev: int,
    optimizer: str = "powell_slsqp",
    slsqp_maxiter: int = DEFAULT_SLSQP_MAXITER,
) -> tuple[np.ndarray, dict[str, Any]]:
    zero = np.zeros((len(ARM_CONTROLLER_IDS), 6), dtype=np.float64)
    if _is_identity_selected_pose(frame.pose):
        try:
            baseline_vertices = np.asarray(compiled.target_rest, dtype=np.float64)[scorer.score_ids]
            baseline_metrics = scorer.score_pose(baseline_vertices, frame.geometry, twists=zero)
            baseline_score = _objective_score(baseline_metrics)
            return zero, {
                "status": "neutral_zero",
                "optimizer": "none",
                "evaluations": 0,
                "zero_score": baseline_score,
                "best_score": baseline_score,
                "chosen_twists": zero.tolist(),
                "metrics_before": baseline_metrics,
                "metrics_after": baseline_metrics,
                "optimizer_success": True,
                "optimizer_message": "neutral correction is fixed exactly zero",
            }
        except Exception as exc:
            return zero, {
                "status": "rejected",
                "optimizer": "none",
                "evaluations": 0,
                "zero_score": None,
                "best_score": None,
                "chosen_twists": zero.tolist(),
                "metrics_before": None,
                "metrics_after": None,
                "optimizer_success": False,
                "optimizer_message": f"neutral scoring failed: {type(exc).__name__}: {exc}",
            }

    optimizer_name = str(optimizer).strip().lower()
    if optimizer_name not in {"powell", "slsqp", "powell_slsqp"}:
        raise ValueError("optimizer must be one of 'powell', 'slsqp', or 'powell_slsqp'")
    if int(slsqp_maxiter) <= 0:
        raise ValueError("slsqp_maxiter must be positive")
    evaluations = 0
    invalid_amplitude = 0
    evaluation_errors: list[str] = []
    best_score = float("inf")
    best_twists = zero.copy()
    best_metrics: dict[str, Any] | None = None

    def evaluate_vector(vector: np.ndarray, *, record: bool = True) -> float:
        nonlocal evaluations, invalid_amplitude, best_score, best_twists, best_metrics
        values = np.asarray(vector, dtype=np.float64).reshape(len(ARM_CONTROLLER_IDS), 6)
        if record:
            evaluations += 1
        try:
            _check_twist_amplitude(values)
            vertices, _globals = evaluate_arm_twists_v14(
                compiled, frame, values, scorer.score_ids
            )
            metrics = scorer.score_pose(vertices, frame.geometry, twists=values)
            score = _objective_score(metrics)
        except PoseCorrectionAmplitudeError:
            invalid_amplitude += 1
            return 1.0e9 + float(np.sum(values * values))
        except Exception as exc:
            if record and len(evaluation_errors) < 8:
                evaluation_errors.append(f"{type(exc).__name__}: {exc}")
            return 5.0e8 + float(np.sum(values * values))
        if record and score < best_score:
            best_score = float(score)
            best_twists = values.copy()
            best_metrics = metrics
        return float(score)

    zero_score = evaluate_vector(zero.reshape(-1), record=True)
    zero_metrics = best_metrics
    # Powell's component bounds prevent unbounded search while the norm check
    # above rejects a vector that violates the SE(3) amplitude contract.  No
    # value is clipped or silently replaced.
    component_bounds = []
    for _ in ARM_CONTROLLER_IDS:
        component_bounds.extend([(-ARM_MAX_ROTATION_NORM, ARM_MAX_ROTATION_NORM)] * 3)
        component_bounds.extend([(-ARM_MAX_TRANSLATION_NORM, ARM_MAX_TRANSLATION_NORM)] * 3)
    powell_report: dict[str, Any] = {
        "requested": optimizer_name in {"powell", "powell_slsqp"},
        "success": None,
        "message": "not requested",
        "iterations": None,
        "evaluations_before": int(evaluations),
    }
    slsqp_report: dict[str, Any] = {
        "requested": optimizer_name in {"slsqp", "powell_slsqp"},
        "success": None,
        "message": "not requested",
        "iterations": None,
        "evaluations_before": None,
    }
    if optimizer_name in {"powell", "powell_slsqp"}:
        try:
            optimization = minimize(
                evaluate_vector,
                zero.reshape(-1),
                method="Powell",
                bounds=component_bounds,
                options={"maxfev": int(maxfev), "xtol": 1.0e-4, "ftol": 1.0e-4},
            )
            powell_report.update(
                success=bool(optimization.success),
                message=str(optimization.message),
                iterations=int(getattr(optimization, "nit", 0)),
            )
        except Exception as exc:
            powell_report.update(
                success=False,
                message=f"optimizer failed: {type(exc).__name__}: {exc}",
                iterations=None,
            )

    if optimizer_name in {"slsqp", "powell_slsqp"}:
        scale = np.tile(
            np.asarray(
                [
                    ARM_MAX_ROTATION_NORM,
                    ARM_MAX_ROTATION_NORM,
                    ARM_MAX_ROTATION_NORM,
                    ARM_MAX_TRANSLATION_NORM,
                    ARM_MAX_TRANSLATION_NORM,
                    ARM_MAX_TRANSLATION_NORM,
                ],
                dtype=np.float64,
            ),
            len(ARM_CONTROLLER_IDS),
        )

        def normalized_values(vector: np.ndarray) -> np.ndarray:
            return np.asarray(vector, dtype=np.float64).reshape(-1) * scale

        normalized_start = best_twists.reshape(-1) / scale
        # SLSQP's inequality constraints must start feasible.  A Powell trial
        # can be infinitesimally outside a norm sphere because the objective
        # uses a small validation tolerance; in that case restart SLSQP from
        # zero rather than projecting/clamping the trial.
        feasible_start = True
        for controller in range(len(ARM_CONTROLLER_IDS)):
            feasible_start &= bool(
                np.dot(normalized_start[6 * controller : 6 * controller + 3],
                       normalized_start[6 * controller : 6 * controller + 3]) <= 1.0
                and np.dot(normalized_start[6 * controller + 3 : 6 * controller + 6],
                           normalized_start[6 * controller + 3 : 6 * controller + 6]) <= 1.0
            )
        if not feasible_start:
            normalized_start = np.zeros_like(normalized_start)

        constraints = []
        for controller in range(len(ARM_CONTROLLER_IDS)):
            rot_slice = slice(6 * controller, 6 * controller + 3)
            trans_slice = slice(6 * controller + 3, 6 * controller + 6)
            constraints.extend(
                [
                    {
                        "type": "ineq",
                        "fun": lambda vector, sl=rot_slice: 1.0 - float(np.dot(vector[sl], vector[sl])),
                    },
                    {
                        "type": "ineq",
                        "fun": lambda vector, sl=trans_slice: 1.0 - float(np.dot(vector[sl], vector[sl])),
                    },
                ]
            )

        def evaluate_normalized(vector: np.ndarray) -> float:
            return evaluate_vector(normalized_values(vector), record=True)

        slsqp_report["evaluations_before"] = int(evaluations)
        try:
            optimization = minimize(
                evaluate_normalized,
                normalized_start,
                method="SLSQP",
                bounds=[(-1.0, 1.0)] * len(normalized_start),
                constraints=constraints,
                options={"maxiter": int(slsqp_maxiter), "ftol": 1.0e-8, "eps": 2.0e-4},
            )
            slsqp_report.update(
                success=bool(optimization.success),
                message=str(optimization.message),
                iterations=int(getattr(optimization, "nit", 0)),
            )
        except Exception as exc:
            slsqp_report.update(
                success=False,
                message=f"optimizer failed: {type(exc).__name__}: {exc}",
                iterations=None,
            )

    # The zero candidate remains authoritative whenever Powell fails to find a
    # strictly better valid point.  If even zero could not be scored, keep it
    # and report rejection; never return a hard-clamped trial.
    if best_metrics is None or not np.isfinite(best_score) or best_score >= zero_score:
        chosen = zero
        chosen_score = zero_score if np.isfinite(zero_score) else None
        chosen_metrics = zero_metrics if np.isfinite(zero_score) else None
        status = "zero_preserved" if chosen_metrics is not None else "rejected"
    else:
        chosen = best_twists
        chosen_score = float(best_score)
        chosen_metrics = best_metrics
        status = "optimized"
    _check_twist_amplitude(chosen)
    return chosen, {
        "status": status,
        "optimizer": optimizer_name,
        "maxfev": int(maxfev),
        "slsqp_maxiter": int(slsqp_maxiter),
        "evaluations": int(evaluations),
        "invalid_amplitude_evaluations": int(invalid_amplitude),
        "zero_score": float(zero_score) if np.isfinite(zero_score) else None,
        "best_score": float(chosen_score) if chosen_score is not None else None,
        "chosen_twists": chosen.tolist(),
        "metrics_before": zero_metrics,
        "metrics_after": chosen_metrics,
        "optimizer_success": bool(
            all(row.get("success") is not False for row in (powell_report, slsqp_report))
            and any(row.get("success") is True for row in (powell_report, slsqp_report))
        ),
        "optimizer_message": "; ".join(
            f"{name}: {row.get('message')}"
            for name, row in (("Powell", powell_report), ("SLSQP", slsqp_report))
            if row.get("requested")
        ),
        "powell": powell_report,
        "slsqp": slsqp_report,
        "evaluation_errors": evaluation_errors,
        "zero_candidate_preserved": bool(status in {"zero_preserved", "rejected"}),
    }


def corrector_twists_for_pose_v14(corrector: PoseCorrectorV14, pose: Any) -> np.ndarray:
    """Expose the saved corrector's local twist for NPZ diagnostics."""
    values = _pose(pose)
    if _is_identity_selected_pose(values):
        return np.zeros((len(ARM_CONTROLLER_IDS), 6), dtype=np.float64)
    # PoseCorrectorV14 deliberately keeps support policy in its runtime
    # evaluator.  This helper only obtains the twist for a supported export;
    # unsupported queries propagate the explicit support exception.
    feature = corrector._query_feature(values)
    twists = corrector._interpolated_twists(feature)
    return np.asarray(twists, dtype=np.float64)


def fit_arm_pose_corrector_v14(
    compiled: Any,
    fit_poses: Mapping[str, Any],
    geometries: Mapping[str, ArmPoseGeometryV14],
    calibration: Any,
    *,
    maxfev: int = DEFAULT_MAXFEV,
    optimizer: str = "powell_slsqp",
    slsqp_maxiter: int = DEFAULT_SLSQP_MAXITER,
) -> ArmPoseFitResultV14:
    """Fit only explicit arm pose names and bake them into a fixed RBF.

    ``fit_poses`` is the complete training set supplied by the caller.  No
    held-out pose loader, AMASS file, or geometry discovery is performed here.
    The caller decides which names are training versus regression.
    """
    if int(maxfev) <= 0:
        raise ValueError("maxfev must be positive")
    optimizer_name = str(optimizer).strip().lower()
    if optimizer_name not in {"powell", "slsqp", "powell_slsqp"}:
        raise ValueError("optimizer must be one of 'powell', 'slsqp', or 'powell_slsqp'")
    if int(slsqp_maxiter) <= 0:
        raise ValueError("slsqp_maxiter must be positive")
    if not isinstance(fit_poses, Mapping) or not fit_poses:
        raise ValueError("fit_poses must be a non-empty mapping")
    fit_names = tuple(str(name) for name in fit_poses)
    if len(set(fit_names)) != len(fit_names):
        raise ValueError("fit pose names contain duplicates")
    if "tpose" not in fit_names:
        raise ValueError("fit_poses must include the explicit tpose neutral sample")
    missing = [name for name in fit_names if name not in geometries]
    if missing:
        raise ValueError(f"fit geometry is missing explicit pose names: {missing}")
    frames = cache_arm_pose_frames_v14(compiled, fit_poses, geometries)
    scorer = ArmScoreContextV14(compiled, calibration)
    twists = np.zeros((len(fit_names), len(ARM_CONTROLLER_IDS), 6), dtype=np.float64)
    reports: dict[str, dict[str, Any]] = {}
    for row, name in enumerate(fit_names):
        chosen, report = _fit_one_pose(
            compiled,
            frames[name],
            scorer,
            maxfev=int(maxfev),
            optimizer=optimizer_name,
            slsqp_maxiter=int(slsqp_maxiter),
        )
        twists[row] = chosen
        reports[name] = report
    if not np.allclose(twists[fit_names.index("tpose")], 0.0, atol=0.0, rtol=0.0):
        raise ValueError("neutral tpose correction must be exactly zero")
    corrector = PoseCorrectorV14.fit(
        np.asarray([frames[name].pose for name in fit_names], dtype=np.float64),
        ARM_SELECTED_JOINT_IDS,
        ARM_CONTROLLER_IDS,
        twists,
        radius=ARM_RBF_RADIUS,
        max_rotation_norm=ARM_MAX_ROTATION_NORM,
        max_translation_norm=ARM_MAX_TRANSLATION_NORM,
    )
    return ArmPoseFitResultV14(
        corrector=corrector,
        local_twists=twists,
        fit_pose_names=fit_names,
        reports=reports,
        frames=frames,
        score_ids=scorer.score_ids.copy(),
    )


def evaluate_arm_regression_v14(
    fit_result: ArmPoseFitResultV14,
    poses: Mapping[str, Any],
    geometries: Mapping[str, ArmPoseGeometryV14],
    compiled: Any,
    calibration: Any,
) -> dict[str, dict[str, Any]]:
    """Score explicit regression poses without fitting them."""
    if not isinstance(poses, Mapping):
        raise ValueError("regression poses must be a mapping")
    missing = [str(name) for name in poses if str(name) not in geometries]
    if missing:
        raise ValueError(f"regression geometry is missing explicit pose names: {missing}")
    frames = cache_arm_pose_frames_v14(compiled, poses, geometries)
    scorer = ArmScoreContextV14(compiled, calibration)
    result: dict[str, dict[str, Any]] = {}
    zero = np.zeros((len(ARM_CONTROLLER_IDS), 6), dtype=np.float64)
    for name, frame in frames.items():
        before = None
        after = None
        try:
            before_vertices, _ = evaluate_arm_twists_v14(compiled, frame, zero, scorer.score_ids)
            before = scorer.score_pose(before_vertices, frame.geometry, twists=zero)
        except Exception as exc:
            before = {"status": "rejected", "error": f"{type(exc).__name__}: {exc}"}
        try:
            twist = corrector_twists_for_pose_v14(fit_result.corrector, frame.pose)
            after_vertices, _ = evaluate_arm_twists_v14(compiled, frame, twist, scorer.score_ids)
            after = scorer.score_pose(after_vertices, frame.geometry, twists=twist)
            result[name] = {
                "status": "supported",
                "correction_twists": twist.tolist(),
                "metrics_before": before,
                "metrics_after": after,
            }
        except PoseCorrectionSupportError as exc:
            result[name] = {
                "status": "unsupported",
                "error": str(exc),
                "correction_twists": None,
                "metrics_before": before,
                "metrics_after": None,
            }
        except Exception as exc:
            result[name] = {
                "status": "rejected",
                "error": f"{type(exc).__name__}: {exc}",
                "correction_twists": None,
                "metrics_before": before,
                "metrics_after": None,
            }
    return result


def load_pose_geometry_directory(
    root: str | Path,
    *,
    subject: str | None = None,
) -> dict[str, ArmPoseGeometryV14]:
    """Load only caller-provided geometry NPZs; never discover AMASS poses."""
    directory = Path(root)
    if not directory.is_dir():
        raise ValueError(f"geometry directory does not exist: {directory}")
    found: dict[str, ArmPoseGeometryV14] = {}
    duplicate_paths: dict[str, list[str]] = {}
    for path in sorted(directory.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            required = {"pose", "skin_vertices", "skin_faces"}
            if not required <= set(data.files):
                continue
            file_subject = str(np.asarray(data["subject"]).item()) if "subject" in data.files and np.asarray(data["subject"]).ndim == 0 else None
            if subject is not None and file_subject not in {None, str(subject)}:
                continue
            if "pose_name" in data.files and np.asarray(data["pose_name"]).ndim == 0:
                name = str(np.asarray(data["pose_name"]).item())
            else:
                stem = path.stem
                parts = stem.split("_", 2)
                name = parts[-1] if len(parts) >= 3 and parts[0] == "subject" else stem
            geometry = ArmPoseGeometryV14(
                name=name,
                pose=data["pose"].copy(),
                skin_vertices=data["skin_vertices"].copy(),
                skin_faces=data["skin_faces"].copy(),
                smplx_joints=data["smplx_joints"].copy() if "smplx_joints" in data.files else None,
                source_vertices=data["source_vertices"].copy() if "source_vertices" in data.files else None,
                vertex_tissue=data["vertex_tissue"].copy() if "vertex_tissue" in data.files else None,
                metadata={
                    "path": str(path),
                    "subject": file_subject,
                    "metadata_json": str(np.asarray(data["metadata_json"]).item())
                    if "metadata_json" in data.files and np.asarray(data["metadata_json"]).ndim == 0
                    else None,
                },
            )
        if name in found:
            duplicate_paths.setdefault(name, []).extend(
                [str(found[name].metadata.get("path", "")), str(path)]
            )
            continue
        found[name] = geometry
    if duplicate_paths:
        raise ValueError(f"duplicate pose names in geometry directory: {duplicate_paths}")
    if not found:
        raise ValueError(f"no pose/skin/joints NPZs found under geometry directory: {directory}")
    return found


def write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(json.dumps(_as_json(value), indent=2, allow_nan=False) + "\n")


__all__ = [
    "ARM_CONTROLLER_IDS",
    "ARM_CONTROLLER_NAMES",
    "ARM_MAX_ROTATION_NORM",
    "ARM_MAX_TRANSLATION_NORM",
    "ARM_RBF_RADIUS",
    "ARM_SELECTED_JOINT_IDS",
    "ArmPoseFitResultV14",
    "ArmPoseFrameV14",
    "ArmPoseGeometryV14",
    "ArmScoreContextV14",
    "PoseCorrectionSupportError",
    "cache_arm_pose_frames_v14",
    "corrector_twists_for_pose_v14",
    "evaluate_arm_regression_v14",
    "evaluate_arm_twists_v14",
    "fit_arm_pose_corrector_v14",
    "load_pose_geometry_directory",
    "write_json",
]
