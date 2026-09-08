"""Bounded static feasibility probe for the left Radius--Ulna rest pair.

This command is deliberately an ablation, not a retarget candidate.  It starts
from the T-pose candidate geometry in the V14 restfit export and applies two
independent rigid transforms to ``Radius_L`` and ``Ulna_L``.  Each transform is
limited to a 5 mm translation norm and a 3 degree rotation norm about that
bone's proximal frozen ``elbow/left/<bone>.fit`` centroid.  All other anatomy,
weights, bind data, vessels, and runtime code are left untouched.

Only frozen ``fit`` domains are used by the SLSQP objective: both directions
of Radius--Ulna, both directions of Radius/Ulna--Humerus, and every vertex of
the two forearm meshes against the supplied SMPL-X skin.  The fit target is
controlled by ``--fit-clearance-mm`` (zero by default) and every direction has
a 3 mm nearest-gap target.  The result is then measured independently on the
original ``validation`` domains and complete triangle surfaces using the
original 0.5 mm acceptance threshold.  Wrist Radius/Ulna-to-Scaphoid values
are diagnostics only; the probe has no wrist term and never forces an
Ulna--Scaphoid gap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import igl
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    _calibration_content_digest,
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.surface_validation_v14 import audit_bone_pair
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import load_source_operator


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_INPUT = ROOT / "outputs/anatomy_retarget/v14_arm_restfit_20260908_005/subject_213328_tpose.npz"
DEFAULT_OPERATOR = ROOT / "outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8"
DEFAULT_CALIBRATION = ROOT / (
    "outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node1_006/"
    "anatomical_calibration_v1"
)
DEFAULT_OUTPUT = ROOT / "outputs/anatomy_retarget/v14_forearm_pair_probe_20260908_001"

SUBJECT = "213328"
PARTITION_FIT = "fit"
PARTITION_VALIDATION = "validation"
FOREARM_BONES = ("Radius_L", "Ulna_L")
PRIMARY_BONES = ("Humerus_L", "Radius_L", "Ulna_L")
TRIANGLE_PAIRS = (
    ("Humerus_L", "Radius_L"),
    ("Humerus_L", "Ulna_L"),
    ("Radius_L", "Ulna_L"),
)
ROTATION_LIMIT_RAD = float(np.deg2rad(3.0))
TRANSLATION_LIMIT_M = 5.0e-3
REPORT_PENETRATION_TOLERANCE_M = 5.0e-4
SKIN_REPORT_THRESHOLD_M = 1.0e-3
FIT_DEFAULT_CLEARANCE_M = 0.0
FIT_CLEARANCE_SCALE_M = 1.0e-4
FIT_GAP_LIMIT_M = 3.0e-3
SLSQP_MAXITER = 40


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        result = float(value)
        return result if np.isfinite(result) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_geometry(path: Path) -> dict[str, Any]:
    required = {
        "source_vertices",
        "candidate_vertices",
        "faces",
        "skin_vertices",
        "skin_faces",
        "pose",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{path}: missing required fields {missing}")
        source = np.asarray(data["source_vertices"], dtype=np.float64)
        candidate = np.asarray(data["candidate_vertices"], dtype=np.float64)
        faces = np.asarray(data["faces"], dtype=np.int32)
        skin = np.asarray(data["skin_vertices"], dtype=np.float64)
        skin_faces = np.asarray(data["skin_faces"], dtype=np.int32)
        pose = np.asarray(data["pose"], dtype=np.float32).reshape(-1)
        tissue = (
            np.asarray(data["vertex_tissue"], dtype=np.int8)
            if "vertex_tissue" in data.files
            else None
        )
        smplx_joints = (
            np.asarray(data["smplx_joints"], dtype=np.float64)
            if "smplx_joints" in data.files
            else None
        )
    if source.ndim != 2 or source.shape[1] != 3 or candidate.shape != source.shape:
        raise ValueError(f"{path}: source/candidate vertices must have matching [N,3] shapes")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError(f"{path}: faces must be non-empty [F,3]")
    if np.any(faces < 0) or np.any(faces >= len(source)):
        raise ValueError(f"{path}: anatomy faces contain out-of-range indices")
    if skin.ndim != 2 or skin.shape[1] != 3 or len(skin) == 0:
        raise ValueError(f"{path}: skin_vertices must be non-empty [N,3]")
    if skin_faces.ndim != 2 or skin_faces.shape[1] != 3 or len(skin_faces) == 0:
        raise ValueError(f"{path}: skin_faces must be non-empty [F,3]")
    if np.any(skin_faces < 0) or np.any(skin_faces >= len(skin)):
        raise ValueError(f"{path}: skin faces contain out-of-range indices")
    if pose.size != 165:
        raise ValueError(f"{path}: pose must contain 55*3 values")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(candidate)):
        raise ValueError(f"{path}: source/candidate vertices contain non-finite values")
    if not np.all(np.isfinite(skin)):
        raise ValueError(f"{path}: skin vertices contain non-finite values")
    if not np.allclose(pose.reshape(55, 3), 0.0, atol=1.0e-7, rtol=0.0):
        raise ValueError(f"{path}: the forearm rest probe requires a T-pose input")
    if tissue is not None and tissue.shape != (len(source),):
        raise ValueError(f"{path}: vertex_tissue does not match anatomy vertex count")
    if smplx_joints is not None and smplx_joints.shape != (55, 3):
        raise ValueError(f"{path}: smplx_joints must have shape [55,3]")
    return {
        "source_vertices": source,
        "candidate_vertices": candidate,
        "faces": faces,
        "skin_vertices": skin,
        "skin_faces": skin_faces,
        "pose": pose.reshape(55, 3),
        "vertex_tissue": tissue,
        "smplx_joints": smplx_joints,
        "path": path.resolve(),
        "sha256": _sha256(path),
        "array_hashes": {
            "source_vertices": _array_sha256(source),
            "candidate_vertices": _array_sha256(candidate),
            "faces": _array_sha256(faces),
            "skin_vertices": _array_sha256(skin),
            "skin_faces": _array_sha256(skin_faces),
            "pose": _array_sha256(pose),
            "vertex_tissue": _array_sha256(tissue) if tissue is not None else None,
            "smplx_joints": _array_sha256(smplx_joints) if smplx_joints is not None else None,
        },
    }


def _mesh_layout(asset: Any, faces: np.ndarray) -> dict[str, dict[str, Any]]:
    names = [str(name) for name in asset.source_mesh_names]
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
    global_faces = np.asarray(faces, dtype=np.int64)
    layout: dict[str, dict[str, Any]] = {}
    for name in (*PRIMARY_BONES, "Scaphoid_L"):
        if name not in names:
            raise ValueError(f"frozen source operator is missing {name}")
        index = names.index(name)
        start, stop = map(int, ranges[index])
        ids = np.arange(start, stop, dtype=np.int64)
        mask = np.all((global_faces >= start) & (global_faces < stop), axis=1)
        local_faces = global_faces[mask] - start
        if len(ids) == 0 or len(local_faces) == 0:
            raise ValueError(f"mesh {name} has no vertices or complete local faces")
        layout[name] = {
            "mesh_name": name,
            "mesh_index": int(index),
            "vertex_start": start,
            "vertex_stop": stop,
            "vertex_ids": ids,
            "faces_local": np.asarray(local_faces, dtype=np.int64),
            "face_global_ids": np.flatnonzero(mask).astype(np.int64),
        }
    return layout


def _domain_ids(calibration: Any, key: str, vertex_count: int) -> np.ndarray:
    if key not in calibration.domains:
        raise KeyError(f"missing frozen calibration domain {key}")
    ids = np.asarray(calibration.domains[key], dtype=np.int64).reshape(-1)
    if len(ids) < 3 or np.any(ids < 0) or np.any(ids >= vertex_count):
        raise ValueError(f"invalid calibration domain {key}")
    return ids


def _signed(points: np.ndarray, target_vertices: np.ndarray, target_faces: np.ndarray) -> np.ndarray:
    values = np.asarray(
        igl.signed_distance(
            np.ascontiguousarray(np.asarray(points, dtype=np.float64)),
            np.ascontiguousarray(np.asarray(target_vertices, dtype=np.float64)),
            np.ascontiguousarray(np.asarray(target_faces, dtype=np.int64)),
        )[0],
        dtype=np.float64,
    ).reshape(-1)
    if len(values) != len(np.asarray(points)) or not np.all(np.isfinite(values)):
        raise ValueError("signed-distance query returned invalid values")
    return values


def _distance_summary(values: np.ndarray, *, penetration_tolerance_m: float) -> dict[str, Any]:
    signed = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(signed) == 0 or not np.all(np.isfinite(signed)):
        raise ValueError("distance summary received an empty or non-finite sample")
    negative = signed < 0.0
    return {
        "sample_count": int(len(signed)),
        "min_signed_m": float(np.min(signed)),
        "p05_signed_m": float(np.quantile(signed, 0.05)),
        "median_signed_m": float(np.median(signed)),
        "p95_signed_m": float(np.quantile(signed, 0.95)),
        "max_signed_m": float(np.max(signed)),
        "minimum_absolute_m": float(np.min(np.abs(signed))),
        "negative_sample_count": int(np.count_nonzero(negative)),
        "penetration_count_below_tolerance": int(
            np.count_nonzero(signed < -float(penetration_tolerance_m))
        ),
        "penetration_tolerance_m": float(penetration_tolerance_m),
    }


def _skin_summary(values: np.ndarray, *, per_bone: Mapping[str, np.ndarray] | None = None) -> dict[str, Any]:
    signed = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(signed) == 0 or not np.all(np.isfinite(signed)):
        raise ValueError("skin summary received an empty or non-finite sample")
    outside = np.maximum(signed, 0.0)
    result: dict[str, Any] = {
        "sample_count": int(len(signed)),
        "outside_count_gt_0": int(np.count_nonzero(signed > 0.0)),
        "outside_count_gt_1mm": int(np.count_nonzero(signed > SKIN_REPORT_THRESHOLD_M)),
        "max_outside_m": float(np.max(outside)),
        "outside_p95_m": float(np.quantile(outside, 0.95)),
        "mean_outside_m": float(np.mean(outside)),
        "min_signed_m": float(np.min(signed)),
        "median_signed_m": float(np.median(signed)),
        "p95_signed_m": float(np.quantile(signed, 0.95)),
        "signed_convention": "negative means inside the SMPL-X skin; positive means outside",
        "uses_every_forearm_vertex": True,
    }
    if per_bone is not None:
        result["per_bone"] = {}
        for name, bone_values in per_bone.items():
            result["per_bone"][name] = _skin_summary(np.asarray(bone_values), per_bone=None)
    return result


def _rigid_apply(points: np.ndarray, *, pivot: np.ndarray, rotvec: np.ndarray, translation: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_matrix()
    center = np.asarray(pivot, dtype=np.float64).reshape(1, 3)
    return (np.asarray(points, dtype=np.float64) - center) @ rotation.T + center + np.asarray(
        translation, dtype=np.float64
    ).reshape(1, 3)


def _physical_parameters(
    normalized: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    x = np.asarray(normalized, dtype=np.float64).reshape(12)
    return {
        "Radius_L": {
            "rotvec": x[0:3] * ROTATION_LIMIT_RAD,
            "translation": x[3:6] * TRANSLATION_LIMIT_M,
        },
        "Ulna_L": {
            "rotvec": x[6:9] * ROTATION_LIMIT_RAD,
            "translation": x[9:12] * TRANSLATION_LIMIT_M,
        },
    }


def _parameter_bounds_ok(normalized: np.ndarray, *, tolerance: float = 1.0e-9) -> bool:
    x = np.asarray(normalized, dtype=np.float64).reshape(12)
    if not np.all(np.isfinite(x)):
        return False
    for start in (0, 6):
        if np.linalg.norm(x[start : start + 3]) > 1.0 + tolerance:
            return False
        if np.linalg.norm(x[start + 3 : start + 6]) > 1.0 + tolerance:
            return False
    return True


class _ProbeContext:
    """Immutable arrays and frozen fit IDs used by every objective call."""

    def __init__(
        self,
        geometry: Mapping[str, Any],
        calibration: Any,
        layout: Mapping[str, Mapping[str, Any]],
        *,
        fit_clearance_m: float = FIT_DEFAULT_CLEARANCE_M,
    ):
        self.base = np.asarray(geometry["candidate_vertices"], dtype=np.float64)
        self.skin = np.asarray(geometry["skin_vertices"], dtype=np.float64)
        self.skin_faces = np.asarray(geometry["skin_faces"], dtype=np.int64)
        self.calibration = calibration
        self.layout = layout
        self.fit_clearance_m = float(fit_clearance_m)
        if not np.isfinite(self.fit_clearance_m) or self.fit_clearance_m < 0.0:
            raise ValueError("fit_clearance_m must be finite and non-negative")
        count = len(self.base)
        self.domains_fit = {
            bone[:-2].lower(): _domain_ids(
                calibration, f"elbow/left/{bone[:-2].lower()}.fit", count
            )
            for bone in PRIMARY_BONES
        }
        self.forearm_ids = np.unique(
            np.concatenate([layout[name]["vertex_ids"] for name in FOREARM_BONES])
        )
        self.forearm_ids_by_bone = {
            name: np.asarray(layout[name]["vertex_ids"], dtype=np.int64)
            for name in FOREARM_BONES
        }
        self.humerus_ids = np.asarray(layout["Humerus_L"]["vertex_ids"], dtype=np.int64)
        self.pivots = {
            name: np.mean(self.base[self.domains_fit[name[:-2].lower()]], axis=0)
            for name in FOREARM_BONES
        }
        self.fit_skin_ids = self.forearm_ids.copy()
        self.fit_query_pairs = (
            ("humerus_to_radius", "humerus", "Radius_L"),
            ("radius_to_humerus", "radius", "Humerus_L"),
            ("humerus_to_ulna", "humerus", "Ulna_L"),
            ("ulna_to_humerus", "ulna", "Humerus_L"),
            ("radius_to_ulna", "radius", "Ulna_L"),
            ("ulna_to_radius", "ulna", "Radius_L"),
        )

    def transformed_parts(self, normalized: np.ndarray) -> dict[str, np.ndarray]:
        physical = _physical_parameters(normalized)
        result = {
            "Humerus_L": self.base[self.layout["Humerus_L"]["vertex_ids"]],
            "Radius_L": self.base[self.layout["Radius_L"]["vertex_ids"]],
            "Ulna_L": self.base[self.layout["Ulna_L"]["vertex_ids"]],
            "Scaphoid_L": self.base[self.layout["Scaphoid_L"]["vertex_ids"]],
        }
        for name in FOREARM_BONES:
            result[name] = _rigid_apply(
                result[name],
                pivot=self.pivots[name],
                rotvec=physical[name]["rotvec"],
                translation=physical[name]["translation"],
            )
        return result

    def objective(self, normalized: np.ndarray) -> float:
        x = np.asarray(normalized, dtype=np.float64).reshape(12)
        if not _parameter_bounds_ok(x):
            return 1.0e12 + float(np.dot(x, x))
        parts = self.transformed_parts(x)
        target_faces = {
            name: self.layout[name]["faces_local"] for name in (*PRIMARY_BONES, "Scaphoid_L")
        }
        query_ids = self.domains_fit
        objective = 0.0
        pair_count = 0
        for _label, query_name, target_name in self.fit_query_pairs:
            query_points = self.base[query_ids[query_name]] if query_name == "humerus" else parts[
                {"radius": "Radius_L", "ulna": "Ulna_L"}[query_name]
            ][
                self._local_rows(query_ids[query_name], {"radius": "Radius_L", "ulna": "Ulna_L"}[query_name])
            ]
            signed = _signed(query_points, parts[target_name], target_faces[target_name])
            # Fit clearance is explicit: a zero target requires the query
            # samples to be on or outside the target surface; a positive
            # target asks for that much signed separation.  The 0.5 mm value
            # used in the final report is intentionally not used here.
            excess = np.maximum(self.fit_clearance_m - signed, 0.0)
            clearance_scale = max(self.fit_clearance_m, FIT_CLEARANCE_SCALE_M)
            objective += float(np.mean((excess / clearance_scale) ** 2))
            objective += float(np.max((excess / clearance_scale) ** 2))
            nearest_gap = float(np.min(np.abs(signed)))
            gap_excess = max(nearest_gap - FIT_GAP_LIMIT_M, 0.0)
            objective += float((gap_excess / FIT_GAP_LIMIT_M) ** 2)
            pair_count += len(signed)
        skin_points = np.concatenate([parts[name] for name in FOREARM_BONES], axis=0)
        skin_signed = _signed(skin_points, self.skin, self.skin_faces)
        skin_excess = np.maximum(skin_signed, 0.0)
        # Skin is a full forearm population term.  A 1 mm scale makes the
        # objective numerically comparable to the 0.1 mm bone penetration term,
        # while the final report retains every signed sample and exact counts.
        skin_scale = SKIN_REPORT_THRESHOLD_M
        objective += float(np.mean((skin_excess / skin_scale) ** 2))
        objective += float(np.max((skin_excess / skin_scale) ** 2))
        # A very small regularizer resolves flat finite-difference directions
        # without making a feasible contact improvement unattractive.
        objective += 1.0e-6 * float(np.dot(x, x))
        del pair_count
        return float(objective)

    def _local_rows(self, global_ids: np.ndarray, mesh_name: str) -> np.ndarray:
        start = int(self.layout[mesh_name]["vertex_start"])
        stop = int(self.layout[mesh_name]["vertex_stop"])
        ids = np.asarray(global_ids, dtype=np.int64)
        if np.any(ids < start) or np.any(ids >= stop):
            raise ValueError(f"fit query IDs for {mesh_name} are not in that mesh")
        return ids - start


def _apply_probe_to_vertices(
    base_vertices: np.ndarray,
    normalized: np.ndarray,
    *,
    context: _ProbeContext,
) -> np.ndarray:
    output = np.asarray(base_vertices, dtype=np.float64).copy()
    physical = _physical_parameters(normalized)
    for name in FOREARM_BONES:
        info = context.layout[name]
        ids = info["vertex_ids"]
        output[ids] = _rigid_apply(
            output[ids],
            pivot=context.pivots[name],
            rotvec=physical[name]["rotvec"],
            translation=physical[name]["translation"],
        )
    return output


def _directional_metrics(
    vertices: np.ndarray,
    *,
    calibration: Any,
    layout: Mapping[str, Mapping[str, Any]],
    partition: str,
    fit_clearance_m: float | None = None,
) -> dict[str, Any]:
    count = len(vertices)
    domain_ids = {
        "humerus": _domain_ids(calibration, f"elbow/left/humerus.{partition}", count),
        "radius": _domain_ids(calibration, f"elbow/left/radius.{partition}", count),
        "ulna": _domain_ids(calibration, f"elbow/left/ulna.{partition}", count),
    }
    result: dict[str, Any] = {}
    pairs = (
        ("humerus_to_radius", "humerus", "Radius_L"),
        ("radius_to_humerus", "radius", "Humerus_L"),
        ("humerus_to_ulna", "humerus", "Ulna_L"),
        ("ulna_to_humerus", "ulna", "Humerus_L"),
        ("radius_to_ulna", "radius", "Ulna_L"),
        ("ulna_to_radius", "ulna", "Radius_L"),
    )
    for label, query_name, target_name in pairs:
        target_info = layout[target_name]
        target_ids = target_info["vertex_ids"]
        query_key = f"elbow/left/{query_name}.{partition}"
        values = _signed(
            vertices[domain_ids[query_name]],
            vertices[target_ids],
            target_info["faces_local"],
        )
        row = {
            "query_domain": query_key,
            "query_vertex_ids_global": domain_ids[query_name].tolist(),
            "target_mesh": target_name,
            "target_vertex_range": [
                int(target_info["vertex_start"]),
                int(target_info["vertex_stop"]),
            ],
            **_distance_summary(values, penetration_tolerance_m=REPORT_PENETRATION_TOLERANCE_M),
        }
        if fit_clearance_m is not None:
            target = float(fit_clearance_m)
            nearest_gap = float(np.min(np.abs(values)))
            row.update(
                {
                    "fit_clearance_target_m": target,
                    "fit_clearance_violation_count": int(np.count_nonzero(values < target)),
                    "fit_clearance_violation_max_m": float(np.max(np.maximum(target - values, 0.0))),
                    "fit_gap_limit_m": FIT_GAP_LIMIT_M,
                    "fit_nearest_absolute_gap_m": nearest_gap,
                    "fit_gap_violation": bool(nearest_gap > FIT_GAP_LIMIT_M),
                    "fit_target_passed": bool(
                        np.all(values >= target) and nearest_gap <= FIT_GAP_LIMIT_M
                    ),
                }
            )
        result[label] = row
    return {
        "partition": partition,
        "query_policy": "frozen elbow/left/<bone> partition IDs to complete opposing mesh triangles",
        "pairs": result,
    }


def _skin_metrics(vertices: np.ndarray, *, geometry: Mapping[str, Any], layout: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    values_by_bone: dict[str, np.ndarray] = {}
    for name in FOREARM_BONES:
        info = layout[name]
        values_by_bone[name] = _signed(
            vertices[info["vertex_ids"]],
            geometry["skin_vertices"],
            geometry["skin_faces"],
        )
    return {
        "population": "all_vertices_of_Radius_L_and_Ulna_L",
        "total": _skin_summary(np.concatenate(list(values_by_bone.values())), per_bone=values_by_bone),
        "per_bone": {
            name: _skin_summary(values, per_bone=None) for name, values in values_by_bone.items()
        },
    }


def _wrist_diagnostics(
    vertices: np.ndarray,
    *,
    calibration: Any,
    layout: Mapping[str, Mapping[str, Any]],
    partition: str,
) -> dict[str, Any]:
    target = layout["Scaphoid_L"]
    result: dict[str, Any] = {
        "partition": partition,
        "target_mesh": "Scaphoid_L",
        "objective_term": False,
        "ulna_scaphoid_gap_forced": False,
        "pairs": {},
    }
    for name, interpretation in (
        ("radius", "radius-to-scaphoid direct-contact diagnostic"),
        ("ulna", "ulna-to-scaphoid TFCC-side diagnostic; no direct-contact requirement"),
    ):
        key = f"calibration/left/wrist/{name}.{partition}"
        query_ids = _domain_ids(calibration, key, len(vertices))
        values = _signed(vertices[query_ids], vertices[target["vertex_ids"]], target["faces_local"])
        result["pairs"][f"{name}_to_scaphoid"] = {
            "query_domain": key,
            "query_vertex_ids_global": query_ids.tolist(),
            "interpretation": interpretation,
            "contact_required_by_probe": False,
            "gap_gate_applied": False,
            **_distance_summary(values, penetration_tolerance_m=REPORT_PENETRATION_TOLERANCE_M),
        }
    hand_key = f"calibration/left/wrist/hand.{partition}"
    hand_ids = _domain_ids(calibration, hand_key, len(vertices))
    result["original_hand_domain"] = {
        "domain": hand_key,
        "vertex_ids_global": hand_ids.tolist(),
        "role": "frozen Scaphoid-side contact reference; not used as a query-to-bone objective",
    }
    return result


def _triangle_pair_metrics(
    vertices: np.ndarray,
    *,
    layout: Mapping[str, Mapping[str, Any]],
    calibration: Any,
) -> dict[str, Any]:
    def forearm_axis() -> tuple[np.ndarray, np.ndarray, float]:
        elbow_ids = np.unique(
            np.concatenate(
                [
                    _domain_ids(calibration, f"elbow/left/{name}.fit", len(vertices))
                    for name in ("humerus", "radius", "ulna")
                ]
            )
        )
        wrist_ids = np.unique(
            np.concatenate(
                [
                    _domain_ids(calibration, f"calibration/left/wrist/{name}.fit", len(vertices))
                    for name in ("radius", "ulna")
                ]
            )
        )
        proximal = np.mean(vertices[elbow_ids], axis=0)
        distal = np.mean(vertices[wrist_ids], axis=0)
        delta = distal - proximal
        length = float(np.linalg.norm(delta))
        if not np.isfinite(length) or length <= 1.0e-8:
            raise ValueError("forearm proximal-distal reference axis is degenerate")
        return proximal, delta / length, length

    def crossing_location(
        first: str,
        second: str,
        triangle_pairs: np.ndarray,
    ) -> dict[str, Any]:
        if len(triangle_pairs) == 0:
            return {
                "triangle_pair_count": 0,
                "reference_axis_defined": False,
                "fit_geometry_queries_cover_region": None,
            }
        origin, axis, length = forearm_axis()
        first_info = layout[first]
        second_info = layout[second]
        first_triangles = vertices[first_info["vertex_ids"]][first_info["faces_local"]]
        second_triangles = vertices[second_info["vertex_ids"]][second_info["faces_local"]]
        first_centroids = np.mean(first_triangles[triangle_pairs[:, 0]], axis=1)
        second_centroids = np.mean(second_triangles[triangle_pairs[:, 1]], axis=1)
        midpoints = 0.5 * (first_centroids + second_centroids)
        axial_m = (midpoints - origin) @ axis
        axial_normalized = axial_m / length

        query_ranges: dict[str, list[float]] = {}
        query_points: list[np.ndarray] = []
        for mesh_name in (first, second):
            short_name = mesh_name[:-2].lower()
            key = f"elbow/left/{short_name}.fit"
            ids = _domain_ids(calibration, key, len(vertices))
            query_ranges[mesh_name] = [
                float(np.min((vertices[ids] - origin) @ axis) / length),
                float(np.max((vertices[ids] - origin) @ axis) / length),
            ]
            query_points.append(vertices[ids])
        all_query_points = np.concatenate(query_points, axis=0)
        query_low = min(row[0] for row in query_ranges.values())
        query_high = max(row[1] for row in query_ranges.values())
        axial_covered = (axial_normalized >= query_low) & (axial_normalized <= query_high)
        nearest = np.min(
            np.linalg.norm(midpoints[:, None, :] - all_query_points[None, :, :], axis=2), axis=1
        )

        def region(value: float) -> str:
            if value <= 1.0 / 3.0:
                return "proximal"
            if value <= 2.0 / 3.0:
                return "midshaft"
            return "distal"

        region_counts = {name: 0 for name in ("proximal", "midshaft", "distal")}
        for value in axial_normalized.tolist():
            region_counts[region(float(value))] += 1
        details = []
        for index, (pair, a_centroid, b_centroid, midpoint, axial, normalized, distance, covered) in enumerate(
            zip(
                triangle_pairs[:20],
                first_centroids[:20],
                second_centroids[:20],
                midpoints[:20],
                axial_m[:20],
                axial_normalized[:20],
                nearest[:20],
                axial_covered[:20],
            )
        ):
            details.append(
                {
                    "sample_index": int(index),
                    "first_triangle_local": int(pair[0]),
                    "second_triangle_local": int(pair[1]),
                    "first_centroid_m": a_centroid.tolist(),
                    "second_centroid_m": b_centroid.tolist(),
                    "midpoint_m": midpoint.tolist(),
                    "axial_from_proximal_m": float(axial),
                    "axial_normalized": float(normalized),
                    "proximal_distal_region": region(float(normalized)),
                    "nearest_fit_query_distance_m": float(distance),
                    "within_fit_query_axial_range": bool(covered),
                }
            )
        return {
            "triangle_pair_count": int(len(triangle_pairs)),
            "reference_axis_defined": True,
            "reference_axis_origin_m": origin.tolist(),
            "reference_axis_unit_proximal_to_distal": axis.tolist(),
            "reference_axis_length_m": length,
            "region_counts": region_counts,
            "axial_normalized_range": [
                float(np.min(axial_normalized)),
                float(np.max(axial_normalized)),
            ],
            "axial_fit_query_ranges_normalized": query_ranges,
            "union_fit_query_axial_range_normalized": [float(query_low), float(query_high)],
            "fit_geometry_queries_cover_region": bool(np.any(axial_covered)),
            "fit_geometry_query_axial_coverage_fraction": float(np.mean(axial_covered)),
            "fit_geometry_query_nearest_distance_range_m": [
                float(np.min(nearest)),
                float(np.max(nearest)),
            ],
            "crossing_location_samples": details,
            "coverage_interpretation": (
                "axial overlap with sparse frozen fit query points is a coverage diagnostic, "
                "not a proof that every crossing triangle is represented by the optimizer"
            ),
        }

    result: dict[str, Any] = {}
    for first, second in TRIANGLE_PAIRS:
        a = layout[first]
        b = layout[second]
        audit = audit_bone_pair(
            vertices[a["vertex_ids"]],
            a["faces_local"],
            vertices[b["vertex_ids"]],
            b["faces_local"],
            depth_tolerance_m=REPORT_PENETRATION_TOLERANCE_M,
        )
        pairs = np.asarray(audit.get("triangle_pairs", []), dtype=np.int64).reshape(-1, 2)
        result[f"{first}--{second}"] = {
            "first_mesh": first,
            "second_mesh": second,
            "triangle_pair_count": int(audit.get("triangle_pair_count", len(pairs))),
            "triangle_pairs_sample": pairs[:20].tolist(),
            "mesh_quality": audit.get("mesh_quality"),
            "signed_samples": audit.get("signed_samples"),
            "depth_tolerance_m": float(
                audit.get("depth_tolerance_m", REPORT_PENETRATION_TOLERANCE_M)
            ),
            "passed": bool(audit.get("passed", False)),
            "reason": str(audit.get("reason", "unknown")),
            "method": str(audit.get("method", "unknown")),
            "vtk_version": str(audit.get("vtk_version", "unknown")),
            "complete_surface_check": True,
            "proximal_distal_location": crossing_location(first, second, pairs),
        }
    return result


def _rigid_shape_metrics(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    x = np.asarray(before, dtype=np.float64).reshape(-1, 3)
    y = np.asarray(after, dtype=np.float64).reshape(-1, 3)
    if x.shape != y.shape or len(x) < 3:
        raise ValueError("rigid shape arrays must have matching [N,3] shape with N>=3")
    cx = np.mean(x, axis=0)
    cy = np.mean(y, axis=0)
    xx = x - cx
    yy = y - cy
    u, _s, vh = np.linalg.svd(xx.T @ yy)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vh[-1, :] *= -1.0
        rotation = vh.T @ u.T
    aligned = xx @ rotation.T + cy
    residual = np.linalg.norm(aligned - y, axis=1)
    pair_before = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    pair_after = np.linalg.norm(y[:, None, :] - y[None, :, :], axis=2)
    upper = np.triu_indices(len(x), k=1)
    pair_delta = np.abs(pair_before[upper] - pair_after[upper])
    return {
        "vertex_count": int(len(x)),
        "rigid_procrustes_rms_m": float(np.sqrt(np.mean(residual**2))),
        "rigid_procrustes_max_m": float(np.max(residual)),
        "pairwise_absolute_max_m": float(np.max(pair_delta)),
        "rotation_determinant": float(np.linalg.det(rotation)),
        "shape_unchanged_within_float_tolerance": bool(float(np.max(pair_delta)) <= 1.0e-10),
    }


def _variant_metrics(
    vertices: np.ndarray,
    *,
    geometry: Mapping[str, Any],
    calibration: Any,
    layout: Mapping[str, Mapping[str, Any]],
    fit_clearance_m: float = FIT_DEFAULT_CLEARANCE_M,
) -> dict[str, Any]:
    return {
        "skin_all_forearm_vertices": _skin_metrics(vertices, geometry=geometry, layout=layout),
        "fit_domain_signed_penetration": _directional_metrics(
            vertices,
            calibration=calibration,
            layout=layout,
            partition=PARTITION_FIT,
            fit_clearance_m=fit_clearance_m,
        ),
        "validation_domain_signed_penetration": _directional_metrics(
            vertices, calibration=calibration, layout=layout, partition=PARTITION_VALIDATION
        ),
        "wrist_contact_diagnostics_fit": _wrist_diagnostics(
            vertices, calibration=calibration, layout=layout, partition=PARTITION_FIT
        ),
        "wrist_contact_diagnostics_validation": _wrist_diagnostics(
            vertices, calibration=calibration, layout=layout, partition=PARTITION_VALIDATION
        ),
        "complete_triangle_pair_checks": _triangle_pair_metrics(
            vertices, layout=layout, calibration=calibration
        ),
    }


def _save_probe_npz(
    path: Path,
    *,
    before: np.ndarray,
    after: np.ndarray,
    geometry: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    tissue = geometry.get("vertex_tissue")
    joints = geometry.get("smplx_joints")
    kwargs: dict[str, Any] = {
        "faces": np.asarray(geometry["faces"], dtype=np.int32),
        "skin_faces": np.asarray(geometry["skin_faces"], dtype=np.int32),
        "source_vertices": np.asarray(before, dtype=np.float32),
        "candidate_vertices": np.asarray(after, dtype=np.float32),
        "skin_vertices": np.asarray(geometry["skin_vertices"], dtype=np.float32),
        "pose": np.asarray(geometry["pose"], dtype=np.float32),
        "subject": np.asarray(SUBJECT),
        "pose_name": np.asarray("tpose"),
        "metadata_json": np.asarray(json.dumps(_json_ready(metadata), sort_keys=True, allow_nan=False)),
    }
    if tissue is not None:
        kwargs["vertex_tissue"] = np.asarray(tissue, dtype=np.int8)
    if joints is not None:
        kwargs["smplx_joints"] = np.asarray(joints, dtype=np.float32)
    np.savez_compressed(path, **kwargs)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--operator", type=Path, default=DEFAULT_OPERATOR)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--fit-clearance-mm",
        type=float,
        default=FIT_DEFAULT_CLEARANCE_M * 1000.0,
        help="minimum signed fit-domain bone clearance target; default 0 mm",
    )
    parser.add_argument("--slsqp-maxiter", type=int, default=SLSQP_MAXITER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    input_path = Path(args.input).expanduser().resolve()
    operator_path = Path(args.operator).expanduser().resolve()
    calibration_path = Path(args.calibration).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite forearm pair probe output: {output}")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if int(args.slsqp_maxiter) <= 0:
        raise ValueError("--slsqp-maxiter must be positive")
    if not np.isfinite(float(args.fit_clearance_mm)) or float(args.fit_clearance_mm) < 0.0:
        raise ValueError("--fit-clearance-mm must be finite and non-negative")
    fit_clearance_m = float(args.fit_clearance_mm) * 1.0e-3
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": 14,
        "artifact_kind": "ForearmPairRestFeasibilityProbeV14",
        "status": "running",
        "probe_only": True,
        "production_candidate": False,
        "anatomical_passed": False,
        "publishable": False,
        "operational_evaluation_completed": False,
        "fit_or_tuning_performed": True,
        "objective_domain_policy": "fit domains only; validation IDs are never queried by the optimizer",
        "fit_target_clearance_mm": float(args.fit_clearance_mm),
        "acceptance_reporting": {
            "penetration_tolerance_mm": REPORT_PENETRATION_TOLERANCE_M * 1000.0,
            "unchanged_from_original_probe": True,
            "fit_target_is_separate_from_acceptance_threshold": True,
        },
        "objective_terms": {
            "bone_surface_pairs": [
                "elbow/left/humerus.fit -> Radius_L",
                "elbow/left/radius.fit -> Humerus_L",
                "elbow/left/humerus.fit -> Ulna_L",
                "elbow/left/ulna.fit -> Humerus_L",
                "elbow/left/radius.fit -> Ulna_L",
                "elbow/left/ulna.fit -> Radius_L",
            ],
            "skin": "all Radius_L and Ulna_L vertices -> supplied T-pose SMPL-X skin",
            "wrist_terms": [],
            "fit_clearance_target_m": fit_clearance_m,
            "fit_clearance_scale_m": FIT_CLEARANCE_SCALE_M,
            "fit_gap_limit_m": FIT_GAP_LIMIT_M,
            "skin_objective_scale_m": SKIN_REPORT_THRESHOLD_M,
        },
        "constraints": {
            "bones": list(FOREARM_BONES),
            "transform": "independent rigid transform per bone",
            "rotation_pivot": "centroid of candidate input vertices in elbow/left/<bone>.fit",
            "rotation_norm_limit_deg": 3.0,
            "rotation_norm_limit_rad": ROTATION_LIMIT_RAD,
            "translation_norm_limit_m": TRANSLATION_LIMIT_M,
            "translation_norm_limit_mm": 5.0,
            "axial_scale_allowed": False,
            "shape_change_allowed": False,
            "wrist_geometry_fixed": True,
            "vessels_modified": False,
            "weights_or_bind_modified": False,
            "runtime_modified": False,
        },
        "optimizer": {
            "method": "SLSQP",
            "parameterization": "12 normalized parameters: [Radius rot3, trans3, Ulna rot3, trans3]",
            "initial_candidate": "exact zero transform applied to rest5 candidate geometry",
            "maxiter": int(args.slsqp_maxiter),
            "bounds": "per-bone rotation and translation Euclidean norm spheres, plus component bounds [-1,1]",
        },
        "input": str(input_path),
        "operator": str(operator_path),
        "calibration": str(calibration_path),
        "failures": [],
    }
    try:
        geometry = _load_geometry(input_path)
        operator = load_source_operator(operator_path, validate=True, mmap=True)
        operator_digest = operator.runtime_digest(validate=False)
        calibration = load_anatomical_calibration_v1(
            calibration_path,
            operator=operator,
            required_scope="full_main_chain",
        )
        calibration_digest = _calibration_content_digest(calibration)
        if len(geometry["candidate_vertices"]) != len(operator.template_asset.vertices_rest):
            raise ValueError("probe input vertex count differs from frozen operator topology")
        if not np.array_equal(geometry["faces"], np.asarray(operator.template_asset.faces, dtype=np.int32)):
            raise ValueError("probe input faces differ from frozen operator topology")
        layout = _mesh_layout(operator.template_asset, geometry["faces"])
        context = _ProbeContext(
            geometry,
            calibration,
            layout,
            fit_clearance_m=fit_clearance_m,
        )
        zero = np.zeros(12, dtype=np.float64)
        evaluations = 0
        evaluation_errors: list[str] = []
        best_score = float("inf")
        best_x = zero.copy()

        def objective(value: np.ndarray) -> float:
            nonlocal evaluations, best_score, best_x
            evaluations += 1
            try:
                score = float(context.objective(value))
            except Exception as exc:
                if len(evaluation_errors) < 8:
                    evaluation_errors.append(f"{type(exc).__name__}: {exc}")
                return 1.0e12
            if np.isfinite(score) and score < best_score and _parameter_bounds_ok(value):
                best_score = score
                best_x = np.asarray(value, dtype=np.float64).copy()
            return score

        zero_score = objective(zero)
        constraints = []
        for start_index in (0, 6):
            constraints.extend(
                (
                    {
                        "type": "ineq",
                        "fun": lambda value, start=start_index: 1.0
                        - float(np.dot(value[start : start + 3], value[start : start + 3])),
                    },
                    {
                        "type": "ineq",
                        "fun": lambda value, start=start_index: 1.0
                        - float(np.dot(value[start + 3 : start + 6], value[start + 3 : start + 6])),
                    },
                )
            )
        optimization = minimize(
            objective,
            zero,
            method="SLSQP",
            bounds=[(-1.0, 1.0)] * 12,
            constraints=constraints,
            options={"maxiter": int(args.slsqp_maxiter), "ftol": 1.0e-9, "eps": 2.0e-4},
        )
        trial_x = np.asarray(optimization.x, dtype=np.float64).reshape(12)
        trial_score = float(objective(trial_x))
        if np.isfinite(best_score) and best_score < zero_score:
            chosen_x = best_x.copy()
            chosen_status = "best_feasible_trial"
        else:
            chosen_x = zero.copy()
            chosen_status = "zero_preserved_no_improvement"
        before_vertices = np.asarray(geometry["candidate_vertices"], dtype=np.float64).copy()
        after_vertices = _apply_probe_to_vertices(before_vertices, chosen_x, context=context)
        physical = _physical_parameters(chosen_x)
        transform_report: dict[str, Any] = {}
        for name in FOREARM_BONES:
            rotvec = physical[name]["rotvec"]
            translation = physical[name]["translation"]
            transform_report[name] = {
                "pivot_fit_centroid_m": context.pivots[name].tolist(),
                "pivot_domain": f"elbow/left/{name[:-2].lower()}.fit",
                "rotation_vector_rad": rotvec.tolist(),
                "rotation_angle_deg": float(np.degrees(np.linalg.norm(rotvec))),
                "translation_m": translation.tolist(),
                "translation_norm_mm": float(np.linalg.norm(translation) * 1000.0),
                "rotation_bound_passed": bool(np.linalg.norm(rotvec) <= ROTATION_LIMIT_RAD + 1.0e-12),
                "translation_bound_passed": bool(np.linalg.norm(translation) <= TRANSLATION_LIMIT_M + 1.0e-12),
                "vertex_count": int(len(layout[name]["vertex_ids"])),
                "shape_transform": "rigid; no scale or bend",
            }
        before_metrics = _variant_metrics(
            before_vertices,
            geometry=geometry,
            calibration=calibration,
            layout=layout,
            fit_clearance_m=fit_clearance_m,
        )
        after_metrics = _variant_metrics(
            after_vertices,
            geometry=geometry,
            calibration=calibration,
            layout=layout,
            fit_clearance_m=fit_clearance_m,
        )
        shape_report = {
            name: _rigid_shape_metrics(
                before_vertices[layout[name]["vertex_ids"]],
                after_vertices[layout[name]["vertex_ids"]],
            )
            for name in FOREARM_BONES
        }
        modified_ids = np.unique(
            np.concatenate([layout[name]["vertex_ids"] for name in FOREARM_BONES])
        )
        unchanged = np.ones(len(before_vertices), dtype=bool)
        unchanged[modified_ids] = False
        non_forearm_delta = np.linalg.norm(after_vertices[unchanged] - before_vertices[unchanged], axis=1)
        tissue_max_displacement: dict[str, float] = {}
        ranges = np.asarray(operator.template_asset.source_vertex_ranges, dtype=np.int64).reshape(-1, 2)
        for tissue, (start, stop) in zip(operator.template_asset.source_tissues, ranges.tolist()):
            values = np.linalg.norm(
                after_vertices[int(start) : int(stop)] - before_vertices[int(start) : int(stop)],
                axis=1,
            )
            label = str(tissue).strip().lower()
            tissue_max_displacement[label] = max(
                tissue_max_displacement.get(label, 0.0), float(np.max(values)) if len(values) else 0.0
            )
        npz_path = output / "subject_213328_tpose_forearm_pair_probe.npz"
        metadata = {
            "schema_version": 14,
            "artifact_kind": "ForearmPairRestFeasibilityProbeV14Geometry",
            "probe_only": True,
            "source_geometry": "rest5 candidate T-pose before independent rigid offsets",
            "candidate_geometry": "after bounded independent Radius_L/Ulna_L rigid offsets",
            "chosen_normalized_parameters": chosen_x.tolist(),
            "chosen_status": chosen_status,
            "operator_runtime_digest": operator_digest,
            "calibration_content_digest": calibration_digest,
        }
        _save_probe_npz(
            npz_path,
            before=before_vertices,
            after=after_vertices,
            geometry=geometry,
            metadata=metadata,
        )
        report.update(
            {
                "status": "complete",
                "operational_evaluation_completed": True,
                "subject": SUBJECT,
                "input_sha256": geometry["sha256"],
                "input_array_hashes": geometry["array_hashes"],
                "operator_runtime_digest": operator_digest,
                "operator_template_vertices_digest": _array_sha256(operator.template_asset.vertices_rest),
                "calibration_content_digest": calibration_digest,
                "calibration_fixed_domain_digest": str(getattr(calibration, "fixed_domain_digest", "")),
                "script_sha256": _sha256(Path(__file__).resolve()),
                "fit_domains": {
                    f"elbow/left/{bone}.fit": int(len(context.domains_fit[bone]))
                    for bone in ("humerus", "radius", "ulna")
                },
                "skin_fit_population": {
                    "ids": context.fit_skin_ids.tolist(),
                    "count": int(len(context.fit_skin_ids)),
                    "policy": "every Radius_L and Ulna_L vertex",
                },
                "pivot_domains": {
                    name: {
                        "domain": f"elbow/left/{name[:-2].lower()}.fit",
                        "centroid_m": context.pivots[name].tolist(),
                    }
                    for name in FOREARM_BONES
                },
                "optimizer_result": {
                    "success": bool(optimization.success),
                    "status_code": int(optimization.status),
                    "message": str(optimization.message),
                    "iterations": int(getattr(optimization, "nit", 0)),
                    "function_evaluations": int(evaluations),
                    "zero_score": float(zero_score),
                    "trial_parameters_normalized": trial_x.tolist(),
                    "trial_score": trial_score,
                    "best_feasible_parameters_normalized": best_x.tolist(),
                    "best_feasible_score": float(best_score),
                    "chosen_parameters_normalized": chosen_x.tolist(),
                    "chosen_status": chosen_status,
                    "score_improvement": float(zero_score - context.objective(chosen_x)),
                    "evaluation_errors": evaluation_errors,
                },
                "chosen_transforms": transform_report,
                "shape_preservation": shape_report,
                "non_forearm_max_displacement_m": float(np.max(non_forearm_delta))
                if len(non_forearm_delta)
                else 0.0,
                "unchanged_mesh_scope": {
                    "modified_meshes": list(FOREARM_BONES),
                    "all_other_vertices_max_displacement_m": float(np.max(non_forearm_delta))
                    if len(non_forearm_delta)
                    else 0.0,
                    "tissue_max_displacement_m": tissue_max_displacement,
                    "vessel_vertices_modified": bool(tissue_max_displacement.get("vessel", 0.0) > 0.0),
                },
                "before_zero_candidate": before_metrics,
                "after_chosen_probe": after_metrics,
                "geometry_npz": str(npz_path),
                "geometry_npz_sha256": _sha256(npz_path),
                "feasibility_conclusion": {
                    "independent_offsets_within_bounds": bool(
                        _parameter_bounds_ok(chosen_x)
                        and all(
                            row["rotation_bound_passed"] and row["translation_bound_passed"]
                            for row in transform_report.values()
                        )
                    ),
                    "fit_objective_improved": bool(context.objective(chosen_x) < zero_score),
                    "validation_radius_ulna_penetration_removed": bool(
                        after_metrics["validation_domain_signed_penetration"]["pairs"]["radius_to_ulna"][
                            "penetration_count_below_tolerance"
                        ]
                        == 0
                        and after_metrics["validation_domain_signed_penetration"]["pairs"]["ulna_to_radius"][
                            "penetration_count_below_tolerance"
                        ]
                        == 0
                    ),
                    "full_triangle_radius_ulna_contacts_removed": bool(
                        after_metrics["complete_triangle_pair_checks"]["Radius_L--Ulna_L"][
                            "triangle_pair_count"
                        ]
                        == 0
                    ),
                    "sufficient_for_full_bone_nonoverlap": bool(
                        after_metrics["complete_triangle_pair_checks"]["Radius_L--Ulna_L"][
                            "triangle_pair_count"
                        ]
                        == 0
                        and after_metrics["validation_domain_signed_penetration"]["pairs"]["radius_to_ulna"][
                            "penetration_count_below_tolerance"
                        ]
                        == 0
                        and after_metrics["validation_domain_signed_penetration"]["pairs"]["ulna_to_radius"][
                            "penetration_count_below_tolerance"
                        ]
                        == 0
                    ),
                    "necessity_claim": "not established by one static ablation; this probe tests bounded sufficiency only",
                    "wrist_contact_interpretation": "radius and ulna wrist diagnostics are reported; ulna-to-scaphoid gap is not forced",
                },
            }
        )
    except Exception as exc:
        report["status"] = "evaluation_error"
        report["failures"].append({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        report["elapsed_seconds"] = float(time.perf_counter() - started)
        report["anatomical_passed"] = False
        report["publishable"] = False
        report["operational_evaluation_completed"] = bool(report.get("status") == "complete")
        _write_json(output / "report.json", report)
    if report.get("status") != "complete":
        print(f"forearm_pair_rest_probe status={report.get('status')} output={output}", flush=True)
        return 1
    print(
        "forearm_pair_rest_probe "
        f"zero_score={report['optimizer_result']['zero_score']:.6g} "
        f"chosen_score={report['optimizer_result']['best_feasible_score']:.6g} "
        f"ru_contacts_after={report['feasibility_conclusion']['full_triangle_radius_ulna_contacts_removed']} "
        f"output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
