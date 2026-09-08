"""One rest-map authority for all 235 controllers and every material point.

The frozen materialized 142 rig is a motion oracle. Its *target* bind is
the reference of source_bone_posed_global. We never mutate that oracle or
enable its preview metadata branches. A compiled target uses one complete
parent-local FK, including wrist/ankle roots and their descendants.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import hashlib
import json

import numpy as np

from .anatomy_lbs import source_bone_posed_global
from .chain_rest_fit_v1 import _global_to_local
from .sparse_lbs_v14 import SparseLBSV14
from .pose_map_v1 import _fk
from .motion_response_v14 import frozen_asset_copy_v14
from .v8_artifacts import (
    SourceOperatorV8,
    SubjectRuntimePackV8,
    load_subject_runtime,
    materialize_subject,
    save_subject_runtime,
)


SOURCE_PACK_KIND_V14 = "SubjectRuntimePackV8"
COMPILED_KIND_V14 = "CompiledAnatomyV14"
COMPILED_SCHEMA_V14 = 14
_HEX_DIGITS = frozenset("0123456789abcdef")

__all__ = [
    "COMPILED_KIND_V14",
    "COMPILED_SCHEMA_V14",
    "CompileConfigV14",
    "CompiledAnatomyV14",
    "SOURCE_PACK_KIND_V14",
    "compile_subject",
    "load_compiled_subject",
    "pose",
]


def _finite(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def _rigid(matrices: Any, count: int, name: str) -> np.ndarray:
    result = _finite(matrices, (count, 4, 4), name)
    if not np.allclose(result[:, 3], [0., 0., 0., 1.], atol=1e-8, rtol=0):
        raise ValueError(f"{name} has invalid affine rows")
    r = result[:, :3, :3]
    if (not np.allclose(r.swapaxes(1, 2) @ r, np.eye(3), atol=3e-6, rtol=0)
            or np.any(np.linalg.det(r) <= 0)):
        raise ValueError(f"{name} must contain proper rigid frames, no scale")
    return result


def _digest(array: Any) -> str:
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(a.dtype.str.encode() + str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def driver_rotation_maps_v14(reference_bind, target_bind):
    """Express source angular responses in the new anatomical local axes.

    Root registration is factored out, so a single global rigid map remains
    equivariant. Only rotations are conjugated: transporting a full SE(3)
    bind change would cancel the new anatomical pivots during skinning.
    """
    reference = np.asarray(reference_bind, dtype=float)[:, :3, :3]
    target = np.asarray(target_bind, dtype=float)[:, :3, :3]
    def proper(rotations):
        u, _, vt = np.linalg.svd(rotations)
        result = u @ vt
        if np.any(np.linalg.det(result) <= 0):
            raise ValueError('driver rotation frame contains reflection')
        return result
    reference, target = proper(reference), proper(target)
    root_map = target[0] @ reference[0].T
    q = proper(reference.swapaxes(1, 2) @ root_map.T @ target)
    q[np.max(np.abs(q-np.eye(3)),axis=(1,2)) < 2e-12] = np.eye(3)
    return q


def _is_digest(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in _HEX_DIGITS for char in text)


def _metadata_truthy(value: Any) -> bool:
    """Return a scalar truth value without NumPy's ambiguous-array error."""

    if isinstance(value, np.ndarray):
        return bool(np.any(value))
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, (list, tuple, set)):
        return bool(value)
    return bool(value)


def _call_pack_digest(pack: Any, name: str) -> str:
    method = getattr(pack, name, None)
    if not callable(method):
        raise ValueError(f"source pack must expose {name}()")
    try:
        value = method(validate=False)
    except TypeError:
        # The actual V8 pack accepts ``validate=False``.  Keeping this tiny
        # fallback makes the strict structural mock used by unit tests API
        # compatible without creating a production bare-asset path.
        value = method()
    text = str(value)
    if not _is_digest(text):
        raise ValueError(f"source pack {name} must return a SHA-256 digest")
    return text


def _validate_source_asset_contract(asset: Any) -> None:
    """Validate the source fields consumed by the V14 motion evaluator.

    ``SubjectRuntimePackV8.validate`` performs the full schema-v8 check for a
    real pack.  This second, small check also covers a strict structural mock
    used by tests and keeps the runtime from accepting an arbitrary object that
    merely happens to have a ``rigged_asset`` attribute.
    """

    required = (
        "source_bone_names",
        "source_bone_parents",
        "target_bind_global",
        "vertices_rest",
        "faces",
        "driver_indices",
        "driver_weights",
    )
    missing = [name for name in required if not hasattr(asset, name)]
    if missing:
        raise ValueError(f"source rig is missing required fields: {missing}")
    names = list(getattr(asset, "source_bone_names") or ())
    n = len(names)
    if n != 235:
        raise ValueError("V14 requires the frozen 235-controller rig")
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64).reshape(-1)
    if parents.shape != (n,) or any(
        int(parent) >= index or int(parent) < -1
        for index, parent in enumerate(parents.tolist())
    ):
        raise ValueError("source controller hierarchy is invalid")
    vertices = np.asarray(asset.vertices_rest, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("source vertices_rest must be non-empty [N, 3]")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("source vertices_rest contains non-finite values")
    faces = np.asarray(asset.faces)
    if (faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0
            or faces.dtype.kind not in {"b", "i", "u"}):
        raise ValueError("source faces must be a non-empty integer [M, 3] array")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("source faces contain an out-of-range vertex ID")
    bind = np.asarray(asset.target_bind_global, dtype=np.float64)
    if bind.shape != (n, 4, 4) or not np.all(np.isfinite(bind)):
        raise ValueError("source target_bind_global must be finite [235, 4, 4]")
    _rigid(bind, n, "source target bind")
    indices = np.asarray(asset.driver_indices, dtype=np.int64)
    weights = np.asarray(asset.driver_weights, dtype=np.float64)
    if indices.ndim != 2 or weights.shape != indices.shape or indices.shape[0] != len(vertices):
        raise ValueError("source sparse drivers must both be [N, K]")
    if np.any(indices < 0) or np.any(indices >= n):
        raise ValueError("source driver indices are out of range")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("source driver weights must be finite and non-negative")
    if not np.allclose(weights.sum(axis=1), 1.0, atol=2.0e-6, rtol=0.0):
        raise ValueError("source driver weight rows must sum to one")
    metadata = getattr(asset, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise ValueError("source rig metadata must be a mapping")
    if metadata.get("source_full_local_fk_v2") is not True:
        raise ValueError("source_full_local_fk_v2 must be explicitly true")
    if metadata.get("pose_cache_forbidden") is not True:
        raise ValueError("source pose-cache policy must be frozen")
    if metadata.get("disable_soft_follow") is not True:
        raise ValueError("source soft-follow policy must be frozen")
    if metadata.get("requires_blender_at_runtime") is not False or metadata.get(
        "requires_blend_file_at_runtime"
    ) is not False:
        raise ValueError("source metadata must disable Blender runtime dependencies")
    pose_cache = getattr(asset, "pose_cache_vertices", None)
    pose_cache_hash = getattr(asset, "pose_cache_hash", "") or ""
    if pose_cache is not None or str(pose_cache_hash):
        raise ValueError("source pack contains a pose-specific cache")
    if metadata.get("whole_chain_source_bind_global") is not None:
        raise ValueError("preview source bind override is not an allowed motion oracle")
    # A true preview branch must never be selected by an accidental metadata
    # flag.  False/empty markers are harmless provenance; truthy ones are not.
    for key, value in metadata.items():
        if "preview" in str(key).lower() and _metadata_truthy(value):
            raise ValueError(f"source metadata enables preview path: {key}")


def _validate_source_pack(pack: Any, betas: np.ndarray) -> dict[str, Any]:
    """Validate one complete V8 source pack and return immutable identity."""

    if not isinstance(pack, SubjectRuntimePackV8):
        # Structural mocks are accepted only when they implement the complete
        # pack identity/validation surface.  A bare rigged asset never reaches
        # this branch.
        required = ("rigged_asset", "betas", "validate", "runtime_digest", "audit_digest")
        if any(not hasattr(pack, name) for name in required):
            raise ValueError(
                "compile_subject accepts SourceOperatorV8 or SubjectRuntimePackV8; "
                "a bare materialized asset is rejected"
            )
    validator = getattr(pack, "validate", None)
    if not callable(validator):
        raise ValueError("source pack must expose validate()")
    validator()
    pack_betas = np.asarray(getattr(pack, "betas", None), dtype=np.float32).reshape(-1)
    if pack_betas.shape != (10,) or not np.all(np.isfinite(pack_betas)):
        raise ValueError("source pack betas must contain exactly 10 finite values")
    if not np.array_equal(pack_betas, np.asarray(betas, dtype=np.float32)):
        raise ValueError("source pack beta mismatch")
    asset = getattr(pack, "rigged_asset", None)
    if asset is None:
        raise ValueError("source pack is missing rigged_asset")
    _validate_source_asset_contract(asset)
    runtime_digest = _call_pack_digest(pack, "runtime_digest")
    audit_digest = _call_pack_digest(pack, "audit_digest")
    content_method = getattr(pack, "content_digest", None)
    content_digest = runtime_digest
    if callable(content_method):
        content_digest = str(content_method())
        if not _is_digest(content_digest):
            raise ValueError("source pack content_digest must be a SHA-256 digest")
    return {
        "kind": SOURCE_PACK_KIND_V14,
        "runtime_digest": runtime_digest,
        "audit_digest": audit_digest,
        "content_digest": content_digest,
        "betas_float32": pack_betas.copy(),
        "rigged_asset_vertices_digest": _digest(asset.vertices_rest),
        "rigged_asset_faces_digest": _digest(asset.faces),
        "rigged_asset_target_bind_digest": _digest(asset.target_bind_global),
        "rigged_asset_driver_indices_digest": _digest(asset.driver_indices),
        "rigged_asset_driver_weights_digest": _digest(asset.driver_weights),
    }


def _validate_compiled_provenance(
    provenance: Any,
    identity: Mapping[str, Any],
    asset: Any,
) -> dict[str, Any]:
    """Check that the compiled manifest still names the embedded source.

    The source pack's V8 digest authenticates its complete numerical payload;
    these explicit fields make a V14 bundle auditable without opening the
    large embedded arrays.  They are required on newly compiled artifacts and
    are checked again after loading.
    """

    if not isinstance(provenance, Mapping):
        raise ValueError("compiled provenance must be a mapping")
    result = dict(provenance)
    required = {
        "source_pack_kind": identity["kind"],
        "source_pack_runtime_digest": identity["runtime_digest"],
        "source_pack_audit_digest": identity["audit_digest"],
        "source_pack_content_digest": identity["content_digest"],
        "source_vertices_digest": identity["rigged_asset_vertices_digest"],
        "faces_digest": _digest(asset.faces),
        "driver_indices_digest": identity["rigged_asset_driver_indices_digest"],
        "driver_weights_digest": identity["rigged_asset_driver_weights_digest"],
        "complete_terminal_rebind": True,
        "authored_weights_modified": False,
    }
    for key, expected in required.items():
        if key not in result or result[key] != expected:
            raise ValueError(f"compiled provenance mismatch: {key}")
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"compiled provenance is not JSON-safe: {exc}") from exc
    return result


@dataclass
class CompileConfigV14:
    # Each point and controller is sampled from these same rest maps.
    # A None entry in axial_fields means identity before its rigid transform.
    rigid_maps: np.ndarray | None = None
    axial_fields: tuple[Any | None, ...] | None = None
    provenance: dict | None = None
    gender: str = "male"
    # Kept as a parsing guard for callers that still pass the old option.  A
    # public bare-materialized-asset path is intentionally not supported.
    materialized_betas: np.ndarray | None = None
    # Optional authenticated authored shape. The materialized subject remains
    # the motion oracle, while its linear beta vertex basis is not inherited
    # by the rest geometry. All tissues and binds share this shape reference.
    shape_reference_operator: SourceOperatorV8 | None = None
    # Explicit offline-calibrated motion response, persisted separately from
    # the immutable source pack. None reproduces the historical response.
    motion_response: Any = None
    # Optional anatomical reference pivots in the shape reference coordinates,
    # before the configured rest map. Geometry and weights remain unchanged.
    shape_reference_bind: np.ndarray | None = None
    rotation_transport: str = 'driver_axes'


@dataclass
class CompiledAnatomyV14:
    source_pack: Any
    betas: np.ndarray
    target_rest: np.ndarray
    reference_bind: np.ndarray
    target_bind: np.ndarray
    translation_maps: np.ndarray
    provenance: dict
    corrector: Any = None
    motion_response: Any = None
    rotation_maps: np.ndarray | None = None

    @property
    def source_asset(self) -> Any:
        """Read-only snapshot of the complete source pack's rigged asset."""

        return getattr(self, '_source_snapshot', self.source_pack.rigged_asset)

    def __post_init__(self) -> None:
        identity = _validate_source_pack(self.source_pack, self.betas)
        self._source_snapshot = frozen_asset_copy_v14(self.source_pack.rigged_asset)
        a = self.source_asset
        self.motion_asset = (a if self.motion_response is None
                             else self.motion_response.apply(a, source_authenticated=True))
        self.provenance = _validate_compiled_provenance(
            self.provenance, identity, a
        )
        n = len(a.source_bone_names)
        if n != 235:
            raise ValueError("V14 requires the frozen 235-controller rig")
        self.parents = np.array(a.source_bone_parents, dtype=np.int64, copy=True)
        if self.parents.shape != (n,) or any(p >= i or p < -1 for i, p in enumerate(self.parents)):
            raise ValueError("invalid controller hierarchy")
        self.reference_bind = _rigid(self.reference_bind, n, "reference bind").copy()
        self.target_bind = _rigid(self.target_bind, n, "target bind").copy()
        if not np.array_equal(self.reference_bind, np.asarray(self.motion_asset.target_bind_global, dtype=np.float64)):
            raise ValueError("motion reference differs from the persisted source response bind")
        self.target_rest = _finite(self.target_rest, np.shape(a.vertices_rest), "target rest").copy()
        self.translation_maps = _finite(self.translation_maps, (n, 3, 3), "residual translation maps").copy()
        self.rotation_maps = (np.tile(np.eye(3), (n, 1, 1)) if self.rotation_maps is None
                              else _finite(self.rotation_maps, (n, 3, 3), 'angular response frame maps').copy())
        frames = np.tile(np.eye(4), (n, 1, 1)); frames[:, :3, :3] = self.rotation_maps
        _rigid(frames, n, 'angular response frame maps')
        if np.any(np.linalg.det(self.translation_maps) <= 0):
            raise ValueError("residual translation maps invert or collapse")
        self.betas = _finite(self.betas, (10,), "betas").copy()
        self.reference_local = _global_to_local(self.reference_bind, self.parents)
        self.target_local = _global_to_local(self.target_bind, self.parents)
        self.reference_local_inverse = np.linalg.inv(self.reference_local)
        self.target_inverse = np.linalg.inv(self.target_bind)
        self.indices = np.array(a.driver_indices, dtype=np.int64, copy=True)
        self.weights = np.array(a.driver_weights, dtype=np.float64, copy=True)
        if (self.indices.shape != self.weights.shape or self.indices.shape[0] != len(self.target_rest)
                or np.any(self.indices < 0) or np.any(self.indices >= n)
                or not np.isfinite(self.weights).all() or np.any(self.weights < 0)
                or not np.allclose(self.weights.sum(1), 1, atol=2e-6, rtol=0)):
            raise ValueError("invalid original sparse weights")
        if (a.metadata or {}).get("whole_chain_source_bind_global") is not None:
            raise ValueError("preview source bind override is not an allowed motion oracle")
        for name in ('parents','betas','indices','weights','reference_bind','target_bind','target_rest',
                     'translation_maps','rotation_maps','reference_local','target_local',
                     'reference_local_inverse','target_inverse'):
            getattr(self,name).setflags(write=False)
        self._lbs = SparseLBSV14(self.target_rest, self.indices, self.weights)

    def globals_from_source(self, source_global: np.ndarray, correction: np.ndarray | None = None) -> np.ndarray:
        """Transfer all local deltas; no terminal world-space overrides."""
        source_local = _global_to_local(source_global, self.parents)
        delta = self.reference_local_inverse @ source_local
        delta[:, :3, :3] = self.rotation_maps.swapaxes(1, 2) @ delta[:, :3, :3] @ self.rotation_maps
        delta[:, :3, 3] = np.einsum("bij,bj->bi", self.translation_maps, delta[:, :3, 3])
        local = self.target_local @ delta
        if correction is not None:
            local = local @ _rigid(correction, len(local), "pose correction")
        return _fk(local, self.parents)

    def source_globals(self, pose55: np.ndarray) -> np.ndarray:
        """The single motion evaluator used by both offline scoring and replay."""
        p = _finite(pose55, (55, 3), "SMPL-X pose")
        return source_bone_posed_global(self.motion_asset, p)

    def apply_pose(self, pose55: np.ndarray, transl: np.ndarray | None = None,
                   *, return_globals: bool = False) -> Any:
        p = _finite(pose55, (55, 3), "SMPL-X pose")
        source_global = self.source_globals(p)
        correction = None if self.corrector is None else self.corrector.evaluate(p)
        global_ = self.globals_from_source(source_global, correction)
        # Preserve the compiled rest exactly, rather than injecting inverse-
        # multiplication roundoff. The source oracle is still evaluated above
        # so unsupported/malformed runtime metadata cannot bypass validation.
        if not np.any(p) and (correction is None or np.allclose(correction, np.eye(4), atol=1e-12, rtol=0)):
            vertices = self.target_rest.copy()
            global_ = self.target_bind.copy()
        else:
            vertices = self._lbs(global_ @ self.target_inverse)
        if transl is not None:
            t = _finite(transl, (3,), "translation")
            vertices = vertices + t
            global_ = global_.copy()
            global_[:, :3, 3] += t
        if not np.isfinite(vertices).all():
            raise ValueError("non-finite posed anatomy")
        result = np.asarray(vertices, dtype=np.float32)
        return (result, global_) if return_globals else result

    def save(self, directory: str | Path) -> None:
        root = Path(directory)
        identity = _validate_source_pack(self.source_pack, self.betas)
        if not isinstance(self.source_pack, SubjectRuntimePackV8):
            raise TypeError(
                "CompiledAnatomyV14.save requires a concrete SubjectRuntimePackV8; "
                "structural mock packs are compile-only"
            )
        provenance = _validate_compiled_provenance(
            self.provenance, identity, self.source_asset
        )
        root.mkdir(parents=True, exist_ok=False)
        # save_subject_runtime persists the complete rig, sparse weights,
        # runtime coefficients, cache identity and audit digest.  It loads
        # those exact arrays back; it does not rebuild global transforms.
        save_subject_runtime(root / "source_pack", self.source_pack)
        arrays = dict(
            betas=np.asarray(self.betas, dtype=np.float64),
            target_rest=np.asarray(self.target_rest, dtype=np.float64),
            reference_bind=np.asarray(self.reference_bind, dtype=np.float64),
            target_bind=np.asarray(self.target_bind, dtype=np.float64),
            translation_maps=np.asarray(self.translation_maps, dtype=np.float64),
            rotation_maps=np.asarray(self.rotation_maps, dtype=np.float64),
        )
        np.savez_compressed(root / "compiled.npz", **arrays)
        names = ["compiled.npz"]
        if self.corrector is not None:
            self.corrector.save(root / "pose_corrector.npz")
            names.append("pose_corrector.npz")
        if self.motion_response is not None:
            self.motion_response.save(root / "motion_response.npz")
            names.append("motion_response.npz")
        source_manifest_path = root / "source_pack" / "manifest.json"
        manifest = dict(
            artifact_kind=COMPILED_KIND_V14,
            schema_version=COMPILED_SCHEMA_V14,
            composition="complete_parent_local_v14",
            publishable=False,
            anatomical_passed=False,
            requires_runtime_optimization=False,
            requires_runtime_blender=False,
            support=self.support_report(),
            provenance=provenance,
            source_pack={
                **{key: value for key, value in identity.items() if key != "betas_float32"},
                "betas_float32": identity["betas_float32"].tolist(),
                "manifest_sha256": hashlib.sha256(source_manifest_path.read_bytes()).hexdigest(),
            },
            files={
                n: hashlib.sha256((root / n).read_bytes()).hexdigest()
                for n in names
            },
        )
        embedded = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if (embedded.get("artifact_kind") != SOURCE_PACK_KIND_V14
                or embedded.get("runtime_digest") != identity["runtime_digest"]
                or embedded.get("audit_digest") != identity["audit_digest"]):
            raise ValueError("embedded source pack identity changed during save")
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")

    def support_report(self) -> dict:
        responses = (self.source_asset.metadata or {}).get('source_coupled_joint_response_v8',{})
        result = dict(anatomically_validated=False,universal_pose_claim=False,
                      rotation_transport=self.provenance.get('rotation_transport', 'legacy_material_axes'),
                      translation_transport=self.provenance.get('translation_transport', 'legacy_shape_reference_axes'),
                      source_weights_modified=False,
                      out_of_domain_policy='raise; no angle clipping or zero correction fallback',
                      computational_source_response_bounds={
                          str(b):dict(smplx_joint=int(r['smplx_joint']),joint_kind=str(r['joint_kind']),
                                      rotvec_norm_radius_rad=float(r['support_radius_rad']),
                                      translation_bound_m=float(r['maximum_translation_m']))
                          for b,r in responses.items()})
        if self.corrector is not None:
            result['pose_corrector']=dict(selected_joint_ids=self.corrector.selected_joint_ids.tolist(),
                controller_ids=self.corrector.controller_ids.tolist(),
                feature_space='concatenated selected local rotation matrices',
                kernel_support_radius=float(self.corrector.radius),
                rotation_norm_bound_rad=float(self.corrector.max_rotation_norm),
                translation_norm_bound_m=float(self.corrector.max_translation_norm),
                sample_count=int(len(self.corrector.sample_features)))
        if self.motion_response is not None:
            result['calibrated_motion_response'] = self.motion_response.provenance
        return result


def _coerce_source_pack(
    beta: np.ndarray,
    source: Any,
    config: CompileConfigV14,
) -> Any:
    """Resolve an operator or complete subject pack to one validated L1 pack."""

    if isinstance(source, SourceOperatorV8):
        source.validate()
        gender = str(config.gender).strip().lower()
        if not gender:
            raise ValueError("CompileConfigV14.gender must be non-empty")
        pack = materialize_subject(source, betas=beta, gender=gender)
    elif isinstance(source, SubjectRuntimePackV8):
        pack = source
    elif hasattr(source, "template_asset"):
        raise TypeError(
            "compile_subject requires a concrete SourceOperatorV8 for operator input"
        )
    elif hasattr(source, "rigged_asset"):
        # This is reserved for strict structural mocks used in tests.  The
        # identity/validation surface is checked by _validate_source_pack;
        # arbitrary objects cannot use a bare-asset fallback.
        pack = source
    elif hasattr(source, "vertices_rest"):
        raise ValueError(
            "bare materialized asset is rejected; compile_subject requires "
            "a complete SubjectRuntimePackV8 (materialized_betas is not a "
            "public bypass)"
        )
    else:
        raise TypeError(
            "compile_subject source must be SourceOperatorV8 or SubjectRuntimePackV8"
        )
    _validate_source_pack(pack, beta)
    return pack


def original_shape_reference_v14(operator: SourceOperatorV8, motion_asset: Any):
    """Rigidly register the complete authored shape to the motion root.

    This preserves every original distance while ensuring SMPL-X root motion
    acts about the same root as its materialized motion oracle. No beta shape
    basis is applied to bones, vessels, nerves, or other materials.
    """
    shape = operator.template_asset
    original_bind = _rigid(shape.target_bind_global, 235, 'original shape bind')
    motion_bind = _rigid(motion_asset.target_bind_global, 235, 'motion reference bind')
    if int(shape.source_bone_parents[0]) != -1:
        raise ValueError('shape reference controller zero must be a root')
    alignment = motion_bind[0] @ np.linalg.inv(original_bind[0])
    vertices = np.asarray(shape.vertices_rest,dtype=np.float64) @ alignment[:3,:3].T + alignment[:3,3]
    bind = alignment[None] @ original_bind
    return vertices, bind, alignment


def compile_subject(betas: np.ndarray, source: Any,
                    config: CompileConfigV14 | None = None) -> CompiledAnatomyV14:
    """Compile one complete V8 source pack, or materialize one V8 operator.

    This is the kinematic/rest compiler. Anatomical fitting and the acceptance
    gate must run before any caller describes its artifact as usable anatomy.
    """
    config = config or CompileConfigV14()
    if config.rotation_transport not in ('driver_axes', 'material_axes'):
        raise ValueError('rotation_transport must be driver_axes or material_axes')
    beta = _finite(betas, (10,), "betas")
    if config.materialized_betas is not None:
        raise ValueError(
            "materialized_betas is only for a removed internal factory; "
            "public compile_subject accepts source packs"
        )
    source_pack = _coerce_source_pack(beta, source, config)
    source_asset = source_pack.rigged_asset
    n = len(source_asset.source_bone_names)
    maps = (np.tile(np.eye(4), (n, 1, 1)) if config.rigid_maps is None
            else _rigid(config.rigid_maps, n, "rest rigid maps"))
    axial = config.axial_fields or (None,) * n
    if len(axial) != n:
        raise ValueError("one optional axial field is required per controller")
    motion_asset = (source_asset if config.motion_response is None
                    else config.motion_response.apply(source_asset, source_authenticated=True))
    base = _rigid(motion_asset.target_bind_global, n, "calibrated source motion bind")
    if (not np.array_equal(base, np.asarray(source_asset.target_bind_global, dtype=float))
            and config.shape_reference_bind is None):
        raise ValueError('a calibrated motion pivot also requires an explicit shape reference bind')
    shape_asset = source_asset
    shape_identity = dict(shape_reference_kind="materialized_subject",
                          inherits_beta_vertex_basis=True)
    if config.shape_reference_operator is not None:
        operator = config.shape_reference_operator
        if not isinstance(operator, SourceOperatorV8):
            raise TypeError('shape reference requires an authenticated SourceOperatorV8')
        operator.validate()
        operator_digest = operator.runtime_digest(validate=False)
        if operator_digest != source_pack.operator_runtime_digest:
            raise ValueError('shape reference and motion source use different source operators')
        shape_asset = operator.template_asset
        for field_name in ('faces','driver_indices','driver_weights','source_bone_parents'):
            if not np.array_equal(getattr(shape_asset,field_name),getattr(source_asset,field_name)):
                raise ValueError(f'shape reference changed original {field_name}')
        if list(shape_asset.source_bone_names) != list(source_asset.source_bone_names):
            raise ValueError('shape reference controller order differs from motion oracle')
        shape_identity = dict(shape_reference_kind="frozen_operator_template",
                              inherits_beta_vertex_basis=False,
                              shape_operator_runtime_digest=operator_digest)
    shape_bind = _rigid(shape_asset.target_bind_global, n, "shape reference bind")
    points = _finite(shape_asset.vertices_rest,np.shape(source_asset.vertices_rest),"shape reference vertices")
    if config.shape_reference_operator is not None:
        points, shape_bind, alignment = original_shape_reference_v14(config.shape_reference_operator,source_asset)
        shape_identity['shape_root_alignment'] = alignment.tolist()
    if config.shape_reference_bind is not None:
        updated = _rigid(config.shape_reference_bind, n, "anatomical shape reference bind")
        shape_identity['reference_pivot_changed_controllers'] = np.flatnonzero(
            np.any(updated != shape_bind, axis=(1, 2))).tolist()
        shape_identity['anatomical_reference_bind_digest'] = _digest(updated)
        shape_bind = updated
    bind = shape_bind.copy()
    translation_maps = np.tile(np.eye(3), (n, 1, 1))
    indices = np.asarray(source_asset.driver_indices)
    weights = np.asarray(source_asset.driver_weights, dtype=np.float64)
    target = points.copy()
    displacement = np.zeros_like(points)
    rows, slots = np.nonzero(weights > 0)
    controllers = indices[rows, slots]
    order = np.argsort(controllers, kind="stable")
    boundaries = np.searchsorted(controllers[order], np.arange(n + 1))
    for bone in range(n):
        selected_slots = order[boundaries[bone]:boundaries[bone + 1]]
        ids, inverse = np.unique(rows[selected_slots], return_inverse=True)
        mass = np.bincount(inverse, weights=weights[rows[selected_slots], slots[selected_slots]], minlength=len(ids))
        field = axial[bone]
        selected = points[ids]
        origin = shape_bind[bone, :3, 3][None]
        j = np.eye(3)
        if field is not None:
            selected = field.map_points(selected)
            mapped_origin = field.map_points(origin)
            j = field.jacobian(origin)[0]
        else:
            mapped_origin = origin
        rigid = maps[bone]
        moved = selected @ rigid[:3, :3].T + rigid[:3, 3]
        displacement[ids] += mass[:, None] * (moved - points[ids])
        jac = rigid[:3, :3] @ j
        u, _, vt = np.linalg.svd(jac)
        rotation = u @ vt
        if np.linalg.det(rotation) <= 0:
            raise ValueError("rest field inverts controller frame")
        bind[bone, :3, :3] = rotation @ shape_bind[bone, :3, :3]
        bind[bone, :3, 3] = (mapped_origin @ rigid[:3, :3].T + rigid[:3, 3])[0]
        # Delta translations are measured in the motion reference's bind
        # axes, which may differ from the root-registered template axes.
        # Transport the spatial vector through the same rest field Jacobian
        # and express the result in the target bind axes.
        translation_maps[bone] = bind[bone, :3, :3].T @ jac @ base[bone, :3, :3]
    target += displacement
    identity = _validate_source_pack(source_pack, beta)
    provenance = dict(config.provenance or {})
    provenance.update(shape_identity,
                      shape_reference_vertices_digest=_digest(shape_asset.vertices_rest),
                      shape_reference_bind_digest=_digest(shape_asset.target_bind_global))
    provenance['rotation_transport'] = config.rotation_transport
    provenance['translation_transport'] = 'motion_reference_axes'
    rotation_maps = (driver_rotation_maps_v14(base, bind) if config.rotation_transport == 'driver_axes'
                     else np.tile(np.eye(3), (n, 1, 1)))
    provenance.update(source_pack_kind=SOURCE_PACK_KIND_V14,
                      source_pack_runtime_digest=identity["runtime_digest"],
                      source_pack_audit_digest=identity["audit_digest"],
                      source_pack_content_digest=identity["content_digest"],
                      source_vertices_digest=_digest(source_asset.vertices_rest),
                      faces_digest=_digest(source_asset.faces),
                      driver_indices_digest=_digest(source_asset.driver_indices),
                      driver_weights_digest=_digest(source_asset.driver_weights),
                      complete_terminal_rebind=True, authored_weights_modified=False,
                      rest_field_application_count=1,
                      active_axial_controller_count=sum(f is not None for f in axial))
    return CompiledAnatomyV14(source_pack, beta, target, base, bind, translation_maps, provenance,
                              motion_response=config.motion_response, rotation_maps=rotation_maps)


def pose(compiled: CompiledAnatomyV14, pose55: np.ndarray,
         transl: np.ndarray | None = None) -> np.ndarray:
    return compiled.apply_pose(pose55, transl)


def load_compiled_subject(directory: str | Path) -> CompiledAnatomyV14:
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("artifact_kind") != COMPILED_KIND_V14
            or manifest.get("schema_version") != COMPILED_SCHEMA_V14
            or manifest.get("composition") != "complete_parent_local_v14"):
        raise ValueError("not a supported V14 compiled anatomy")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("compiled anatomy is missing a file inventory")
    names = set(files)
    if "compiled.npz" not in names or names - {"compiled.npz", "pose_corrector.npz", "motion_response.npz"}:
        raise ValueError("invalid compiled file inventory")
    for name, expected in files.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"compiled file digest mismatch: {name}")
    source_entry = manifest.get("source_pack")
    if not isinstance(source_entry, dict):
        raise ValueError("compiled anatomy is missing source_pack identity")
    source_manifest_path = root / "source_pack" / "manifest.json"
    if not source_manifest_path.is_file():
        raise ValueError("compiled anatomy is missing embedded source pack")
    expected_manifest_digest = str(source_entry.get("manifest_sha256", ""))
    if hashlib.sha256(source_manifest_path.read_bytes()).hexdigest() != expected_manifest_digest:
        raise ValueError("embedded source pack manifest digest mismatch")
    source_pack = load_subject_runtime(root / "source_pack", validate=True, mmap=False)
    with np.load(root / "compiled.npz", allow_pickle=False) as data:
        values = {k: data[k].copy() for k in ["betas", "target_rest", "reference_bind", "target_bind", "translation_maps"]}
        # Older V14 artifacts transported material-local angular matrices
        # without a change of basis. Keep their exact saved replay semantics.
        values['rotation_maps'] = data['rotation_maps'].copy() if 'rotation_maps' in data else None
    identity = _validate_source_pack(source_pack, values["betas"])
    for key in ("kind", "runtime_digest", "audit_digest", "content_digest",
                "rigged_asset_vertices_digest", "rigged_asset_driver_indices_digest",
                "rigged_asset_driver_weights_digest"):
        if str(source_entry.get(key, "")) != str(identity[key]):
            raise ValueError(f"embedded source pack identity mismatch: {key}")
    # The direct face/bind digests were added after the first V14 fit output.
    # Runtime digest authentication still covers those arrays, so old bundles
    # remain readable; new bundles carry the explicit audit fields.
    for key in ("rigged_asset_faces_digest", "rigged_asset_target_bind_digest"):
        if key in source_entry and str(source_entry[key]) != str(identity[key]):
            raise ValueError(f"embedded source pack identity mismatch: {key}")
    stored_betas = np.asarray(source_entry.get("betas_float32", []), dtype=np.float32)
    if not np.array_equal(stored_betas, np.asarray(values["betas"], dtype=np.float32)):
        raise ValueError("embedded source pack beta identity mismatch")
    corrector = None
    if "pose_corrector.npz" in names:
        from .pose_corrector_v14 import PoseCorrectorV14
        corrector = PoseCorrectorV14.load(root / "pose_corrector.npz")
    motion_response = None
    if "motion_response.npz" in names:
        from .motion_response_v14 import BakedMotionResponseV14
        motion_response = BakedMotionResponseV14.load(root / "motion_response.npz")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("compiled anatomy is missing provenance")
    return CompiledAnatomyV14(
        source_pack, **values, provenance=provenance, corrector=corrector,
        motion_response=motion_response
    )
