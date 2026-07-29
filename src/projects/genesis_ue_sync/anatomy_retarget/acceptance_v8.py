"""Fail-closed, topology-bound geometry gates for Anatomy Retarget V8.

The helpers in this module deliberately consume final arrays instead of a
candidate's self-reported metrics.  Domains are selected once in material
space, bound to a topology digest, and split into fit and validation subsets.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


V8_ACCEPTANCE_SPEC = "anatomy_acceptance_v8"


def topology_digest(vertex_count: int, faces: np.ndarray) -> str:
    triangles = np.ascontiguousarray(np.asarray(faces, dtype=np.int32).reshape(-1, 3))
    digest = hashlib.sha256(b"anatomy-topology-v8\0")
    digest.update(np.asarray([int(vertex_count)], dtype=np.int64).tobytes())
    digest.update(np.asarray(triangles.shape, dtype=np.int64).tobytes())
    digest.update(triangles.tobytes())
    return digest.hexdigest()


def _ids(value: Any, *, name: str, vertex_count: int) -> np.ndarray:
    result = np.unique(np.asarray(value, dtype=np.int64).reshape(-1))
    if not len(result):
        raise ValueError(f"domain {name!r} is empty")
    if np.any(result < 0) or np.any(result >= int(vertex_count)):
        raise ValueError(f"domain {name!r} references an invalid vertex")
    return result.astype(np.int32)


@dataclass(frozen=True)
class FrozenValidationDomainsV8:
    topology_digest: str
    vertex_count: int
    domains: Mapping[str, np.ndarray]
    fit_validation_pairs: tuple[tuple[str, str], ...]
    provenance: Mapping[str, Any]

    def validate(self, faces: np.ndarray | None = None) -> None:
        if len(self.topology_digest) != 64:
            raise ValueError("domains require a SHA-256 topology digest")
        if int(self.vertex_count) <= 0:
            raise ValueError("domains require a positive vertex_count")
        normalized = {
            str(name): _ids(value, name=str(name), vertex_count=self.vertex_count)
            for name, value in self.domains.items()
        }
        if not normalized:
            raise ValueError("at least one frozen domain is required")
        for fit_name, validation_name in self.fit_validation_pairs:
            if fit_name not in normalized or validation_name not in normalized:
                raise ValueError(
                    f"unknown fit/validation pair {fit_name!r}, {validation_name!r}"
                )
            overlap = np.intersect1d(
                normalized[fit_name], normalized[validation_name], assume_unique=True
            )
            if len(overlap):
                raise ValueError(
                    f"fit and validation domains overlap: {fit_name}/{validation_name}"
                )
        if faces is not None:
            actual = topology_digest(self.vertex_count, faces)
            if actual != self.topology_digest:
                raise ValueError("frozen-domain topology digest mismatch")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": 8,
            "artifact_kind": "FrozenValidationDomainsV8",
            "spec": V8_ACCEPTANCE_SPEC,
            "topology_digest": self.topology_digest,
            "vertex_count": int(self.vertex_count),
            "domains": {
                str(name): np.asarray(ids, dtype=np.int32).reshape(-1).tolist()
                for name, ids in sorted(self.domains.items())
            },
            "fit_validation_pairs": [
                [str(fit_name), str(validation_name)]
                for fit_name, validation_name in self.fit_validation_pairs
            ],
            "provenance": dict(self.provenance),
        }

    def save_json(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenValidationDomainsV8":
        if int(value.get("schema_version", -1)) != 8:
            raise ValueError("frozen domains must use schema_version 8")
        if value.get("artifact_kind") != "FrozenValidationDomainsV8":
            raise ValueError("invalid frozen-domain artifact kind")
        vertex_count = int(value["vertex_count"])
        result = cls(
            topology_digest=str(value["topology_digest"]),
            vertex_count=vertex_count,
            domains={
                str(name): _ids(ids, name=str(name), vertex_count=vertex_count)
                for name, ids in dict(value["domains"]).items()
            },
            fit_validation_pairs=tuple(
                (str(pair[0]), str(pair[1]))
                for pair in value.get("fit_validation_pairs", ())
            ),
            provenance=dict(value.get("provenance", {})),
        )
        result.validate()
        return result

    @classmethod
    def load_json(cls, path: Path | str) -> "FrozenValidationDomainsV8":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def fit_sphere(points: np.ndarray) -> dict[str, Any]:
    xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(xyz) < 4 or not np.all(np.isfinite(xyz)):
        return {"available": False, "reason": "sphere fit needs four finite points"}
    matrix = np.column_stack((2.0 * xyz, np.ones(len(xyz))))
    target = np.einsum("ij,ij->i", xyz, xyz)
    solution, _residuals, rank, singular = np.linalg.lstsq(
        matrix, target, rcond=None
    )
    if int(rank) < 4 or singular[-1] <= 1.0e-12 * singular[0]:
        return {"available": False, "reason": "sphere fit is rank deficient"}
    center = solution[:3]
    radius_squared = float(solution[3] + np.dot(center, center))
    if radius_squared <= 0.0:
        return {"available": False, "reason": "sphere fit has invalid radius"}
    radius = float(np.sqrt(radius_squared))
    residual = np.abs(np.linalg.norm(xyz - center, axis=1) - radius)
    return {
        "available": True,
        "center": center.tolist(),
        "radius_m": radius,
        "rms_residual_m": float(np.sqrt(np.mean(residual * residual))),
        "max_residual_m": float(np.max(residual)),
        "condition_number": float(singular[0] / singular[-1]),
    }


def fit_sphere_center_fixed_radius(
    points: np.ndarray,
    *,
    radius_m: float,
    initial_center: Any,
    multistart: bool = True,
) -> dict[str, Any]:
    """Fit only a socket centre when the opposing head supplies the radius."""
    from scipy.optimize import least_squares

    xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    radius = float(radius_m)
    initial = np.asarray(initial_center, dtype=np.float64).reshape(3)
    if (
        len(xyz) < 4
        or not np.all(np.isfinite(xyz))
        or not np.all(np.isfinite(initial))
        or not np.isfinite(radius)
        or radius <= 0.0
    ):
        return {"available": False, "reason": "invalid fixed-radius sphere inputs"}
    # A socket is only a partial sphere.  Starting solely at the opposing head
    # or at an unconstrained algebraic sphere can converge to the wrong side of
    # that surface while still reporting optimizer success.  Deterministic
    # starts around the material-domain centroid remove that ambiguity without
    # consulting any pose-dependent nearest point.
    centroid = np.mean(xyz, axis=0)
    starts = [initial]
    if multistart:
        starts.append(centroid)
        unconstrained = fit_sphere(xyz)
        if unconstrained.get("available", False):
            starts.append(np.asarray(unconstrained["center"], dtype=np.float64))
        for axis in np.eye(3, dtype=np.float64):
            starts.append(centroid + radius * axis)
            starts.append(centroid - radius * axis)
    solutions = [
        least_squares(
            lambda center: np.linalg.norm(xyz - center[None, :], axis=1) - radius,
            start,
            method="trf",
            loss="linear",
            max_nfev=1000,
        )
        for start in starts
    ]
    solved = min(
        solutions,
        key=lambda item: float(
            np.sqrt(
                np.mean(
                    (
                        np.linalg.norm(xyz - item.x[None, :], axis=1)
                        - radius
                    )
                    ** 2
                )
            )
        ),
    )
    if not solved.success or not np.all(np.isfinite(solved.x)):
        return {"available": False, "reason": "fixed-radius sphere fit failed"}
    residual = np.abs(np.linalg.norm(xyz - solved.x, axis=1) - radius)
    return {
        "available": True,
        "center": solved.x.tolist(),
        "radius_m": radius,
        "radius_constrained": True,
        "rms_residual_m": float(np.sqrt(np.mean(residual * residual))),
        "max_residual_m": float(np.max(residual)),
        "optimizer_cost": float(solved.cost),
    }


def independent_joint_center_gate(
    vertices: np.ndarray,
    domains: FrozenValidationDomainsV8,
    *,
    first_fit: str,
    second_fit: str,
    first_validation: str,
    second_validation: str,
    maximum_center_error_m: float = 0.002,
) -> dict[str, Any]:
    """Fit on one material subset and measure again on disjoint vertices."""
    points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    if len(points) != domains.vertex_count:
        return {"available": False, "pass": False, "reason": "vertex count mismatch"}
    names = (first_fit, second_fit, first_validation, second_validation)
    if any(name not in domains.domains for name in names):
        return {"available": False, "pass": False, "reason": "domain missing"}
    first_fit_result = fit_sphere(points[domains.domains[first_fit]])
    first_validation_result = fit_sphere(
        points[domains.domains[first_validation]]
    )
    if not (
        first_fit_result.get("available", False)
        and first_validation_result.get("available", False)
    ):
        return {
            "available": False,
            "pass": False,
            "reason": "one or more head sphere fits unavailable",
            "fits": {
                "first_fit": first_fit_result,
                "first_validation": first_validation_result,
            },
        }
    second_fit_result = fit_sphere_center_fixed_radius(
        points[domains.domains[second_fit]],
        radius_m=first_fit_result["radius_m"],
        initial_center=first_fit_result["center"],
    )
    second_validation_result = fit_sphere_center_fixed_radius(
        points[domains.domains[second_validation]],
        radius_m=first_validation_result["radius_m"],
        initial_center=first_validation_result["center"],
    )
    fits = {
        "first_fit": first_fit_result,
        "second_fit": second_fit_result,
        "first_validation": first_validation_result,
        "second_validation": second_validation_result,
    }
    if not (
        second_fit_result.get("available", False)
        and second_validation_result.get("available", False)
    ):
        return {
            "available": False,
            "pass": False,
            "reason": "one or more socket fits unavailable",
            "fits": fits,
        }
    fit_error = float(
        np.linalg.norm(
            np.asarray(fits["first_fit"]["center"])
            - np.asarray(fits["second_fit"]["center"])
        )
    )
    validation_error = float(
        np.linalg.norm(
            np.asarray(fits["first_validation"]["center"])
            - np.asarray(fits["second_validation"]["center"])
        )
    )
    passed = validation_error <= float(maximum_center_error_m)
    return {
        "available": True,
        "pass": bool(passed),
        "fit_center_error_m": fit_error,
        "validation_center_error_m": validation_error,
        "maximum_center_error_m": float(maximum_center_error_m),
        "fits": fits,
    }


def rigid_compound_gate(
    reference_points: np.ndarray,
    posed_points: np.ndarray,
    *,
    maximum_rms_m: float = 0.0005,
    maximum_error_m: float = 0.001,
) -> dict[str, Any]:
    reference = np.asarray(reference_points, dtype=np.float64).reshape(-1, 3)
    posed = np.asarray(posed_points, dtype=np.float64).reshape(-1, 3)
    if reference.shape != posed.shape or len(reference) < 3:
        return {"available": False, "pass": False, "reason": "compound shape mismatch"}
    source_center = np.mean(reference, axis=0)
    target_center = np.mean(posed, axis=0)
    covariance = (reference - source_center).T @ (posed - target_center)
    left, _singular, right = np.linalg.svd(covariance)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right[-1] *= -1.0
        rotation = right.T @ left.T
    aligned = (reference - source_center) @ rotation.T + target_center
    errors = np.linalg.norm(aligned - posed, axis=1)
    rms = float(np.sqrt(np.mean(errors * errors)))
    maximum = float(np.max(errors))
    return {
        "available": True,
        "pass": bool(rms <= maximum_rms_m and maximum <= maximum_error_m),
        "rms_error_m": rms,
        "max_error_m": maximum,
        "maximum_rms_m": float(maximum_rms_m),
        "maximum_error_m": float(maximum_error_m),
        "rotation": rotation.tolist(),
        "translation": (target_center - rotation @ source_center).tolist(),
    }


def bone_station_profile(
    points: np.ndarray,
    *,
    station_count: int = 7,
) -> dict[str, Any]:
    """Measure robust radial thickness at fixed fractions of a bone axis."""
    xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(xyz) < max(20, 3 * station_count) or station_count < 5:
        return {"available": False, "reason": "not enough points or stations"}
    center = np.mean(xyz, axis=0)
    _u, singular, axes = np.linalg.svd(xyz - center, full_matrices=False)
    if singular[0] <= 1.0e-12:
        return {"available": False, "reason": "bone axis is degenerate"}
    axis = axes[0]
    axial = (xyz - center) @ axis
    low, high = np.quantile(axial, (0.02, 0.98))
    length = float(high - low)
    if length <= 1.0e-8:
        return {"available": False, "reason": "bone length is degenerate"}
    fractions = np.linspace(0.1, 0.9, station_count)
    half_width = 0.45 * length / max(station_count - 1, 1)
    radii: list[float] = []
    counts: list[int] = []
    for fraction in fractions:
        location = low + fraction * length
        selected = np.abs(axial - location) <= half_width
        counts.append(int(np.count_nonzero(selected)))
        if counts[-1] < 3:
            return {"available": False, "reason": "a station has fewer than 3 points"}
        local = xyz[selected] - center
        radial = local - np.outer(local @ axis, axis)
        radial_center = np.median(radial, axis=0)
        radii.append(float(np.median(np.linalg.norm(radial - radial_center, axis=1))))
    return {
        "available": True,
        "axis": axis.tolist(),
        "length_m": length,
        "fractions": fractions.tolist(),
        "radii_m": radii,
        "sample_counts": counts,
    }


def compare_bone_station_profiles(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    maximum_rms_relative_error: float = 0.05,
    maximum_relative_error: float = 0.10,
    maximum_extra_taper: float = 0.05,
) -> dict[str, Any]:
    if not reference.get("available", False) or not candidate.get("available", False):
        return {"available": False, "pass": False, "reason": "profile unavailable"}
    first = np.asarray(reference["radii_m"], dtype=np.float64)
    second = np.asarray(candidate["radii_m"], dtype=np.float64)
    if first.shape != second.shape or np.any(first <= 1.0e-8):
        return {"available": False, "pass": False, "reason": "profile shape mismatch"}
    relative = second / first - 1.0
    rms = float(np.sqrt(np.mean(relative * relative)))
    maximum = float(np.max(np.abs(relative)))
    first_step = np.diff(first) / np.maximum(first[:-1], 1.0e-8)
    second_step = np.diff(second) / np.maximum(second[:-1], 1.0e-8)
    extra_taper = float(np.max(np.abs(second_step - first_step))) if len(first_step) else 0.0
    return {
        "available": True,
        "pass": bool(
            rms <= maximum_rms_relative_error
            and maximum <= maximum_relative_error
            and extra_taper <= maximum_extra_taper
        ),
        "rms_relative_error": rms,
        "max_relative_error": maximum,
        "max_extra_taper": extra_taper,
        "relative_errors": relative.tolist(),
    }


def require_available_gates(gates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    for name, gate in gates.items():
        if not bool(gate.get("available", False)):
            failures.append(f"{name}:unavailable")
        elif not bool(gate.get("pass", False)):
            failures.append(f"{name}:failed")
    return {
        "schema_version": 8,
        "spec": V8_ACCEPTANCE_SPEC,
        "available": not any(item.endswith(":unavailable") for item in failures),
        "passed": not failures,
        "failures": failures,
        "gates": dict(gates),
    }
