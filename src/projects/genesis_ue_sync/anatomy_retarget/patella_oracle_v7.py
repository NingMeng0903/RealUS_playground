"""Immutable V71 patella response law for AnatomyPatellaOracleV7.

Earlier operators treated the authored patella surface trajectory as a pose
target.  That couples a subject's bind geometry to one Action bake and lets
keyed/unkeyed Action frames silently poison the gain.  This module freezes a
parent-local angular response measured only from bone matrices, excludes
unkeyed frames, and uses evaluated meshes solely for a penetration envelope
and later contact-corridor translations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .joint_contact_v7 import FrozenJointMaterialDomainsV7


PATELLA_ORACLE_SCHEMA_VERSION = 7
PATELLA_ORACLE_KIND = "AnatomyPatellaOracleV7"
SIDES = ("left", "right")

_SIDE_SUFFIX = {"left": "L", "right": "R"}
_AXIS_LOCAL = np.asarray((0.0, 0.0, -1.0), dtype=np.float64)
_LOCAL_WINDOW_DEG = 7.5
_MIN_LOCAL_SAMPLES = 3
_MAX_TRANSLATION_DRIFT = 1.0e-4
_MAX_OFF_AXIS_DEG = 0.5
_MIN_KEYED_FRAMES = 12
_MESH_SAMPLE_STRIDE = 3
_DEADBAND_LOOSE_M = 0.0002
_DEADBAND_TIGHT_M = 0.001


def _as_float64_1d(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite")
    return array


def _as_unit3(value: np.ndarray, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{label} must be a non-zero finite vector")
    return vector / norm


def _array_digest(digest: Any, label: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(label.encode("utf-8"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.tobytes())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bone_index(names: Sequence[str], name: str) -> int:
    try:
        return list(names).index(name)
    except ValueError as exc:
        raise ValueError(f"action is missing required bone {name!r}") from exc


def _side_bone_names(side: str) -> dict[str, str]:
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    suffix = _SIDE_SUFFIX[side]
    return {
        "femur": f"Femur_Rot_{suffix}",
        "knee": f"Knee_Rotate_{suffix}",
        "tibia": f"Tibia_Bone_{suffix}",
        "patella": f"Patella_Rotate_{suffix}",
    }


def _signed_axis_angle(
    rest_local: np.ndarray,
    action_local: np.ndarray,
    axis_local: np.ndarray,
) -> tuple[float, float]:
    """Return (signed_angle_rad, off_axis_residual_rad) about ``axis_local``."""
    rest_r = np.asarray(rest_local, dtype=np.float64)[:3, :3]
    action_r = np.asarray(action_local, dtype=np.float64)[:3, :3]
    relative = np.linalg.inv(rest_r) @ action_r
    rotvec = Rotation.from_matrix(relative).as_rotvec()
    axis = _as_unit3(axis_local, "axis_local")
    parallel = float(np.dot(rotvec, axis))
    residual = float(np.linalg.norm(rotvec - parallel * axis))
    return parallel, residual


def _rotation4(axis_local: np.ndarray, angle_rad: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    axis = _as_unit3(axis_local, "axis_local")
    matrix[:3, :3] = Rotation.from_rotvec(axis * float(angle_rad)).as_matrix()
    return matrix


def _nearest_distances(points: np.ndarray, target: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if not len(points) or not len(target):
        raise ValueError("nearest-distance queries require non-empty point sets")
    try:
        from scipy.spatial import cKDTree

        return np.asarray(cKDTree(target).query(points, k=1)[0], dtype=np.float64)
    except Exception:
        squared = np.sum(
            (points[:, None, :] - target[None, :, :]) ** 2, axis=2
        )
        return np.sqrt(np.min(squared, axis=1))


def _interp_response_deg(
    knots_deg: np.ndarray,
    response_deg: np.ndarray,
    flexion_deg_abs: np.ndarray,
) -> np.ndarray:
    knots = np.asarray(knots_deg, dtype=np.float64).reshape(-1)
    response = np.asarray(response_deg, dtype=np.float64).reshape(-1)
    query = np.asarray(flexion_deg_abs, dtype=np.float64)
    out = np.empty(query.shape, dtype=np.float64)
    if len(knots) < 2:
        raise ValueError("response interpolation requires at least two knots")
    final_slope = (response[-1] - response[-2]) / (knots[-1] - knots[-2])
    inside = query <= knots[-1]
    out[inside] = np.interp(query[inside], knots, response)
    out[~inside] = response[-1] + final_slope * (query[~inside] - knots[-1])
    return out


def _fit_response_deg(
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    knots_deg: np.ndarray,
) -> tuple[np.ndarray, float]:
    theta = np.asarray(theta_deg, dtype=np.float64).reshape(-1)
    phi = np.asarray(phi_deg, dtype=np.float64).reshape(-1)
    knots = np.asarray(knots_deg, dtype=np.float64).reshape(-1)
    denom = float(np.sum(theta * theta))
    if denom <= 0.0:
        raise ValueError("keyed flexion samples are degenerate for through-origin fit")
    slope = float(np.sum(theta * phi) / denom)
    response = np.zeros(len(knots), dtype=np.float64)
    for index, knot in enumerate(knots):
        if float(knot) == 0.0:
            response[index] = 0.0
            continue
        local = np.abs(theta - float(knot)) <= _LOCAL_WINDOW_DEG
        if int(np.count_nonzero(local)) >= _MIN_LOCAL_SAMPLES:
            local_theta = theta[local]
            # Exclude exact-zero flexion from the local ratio (should not appear
            # among keyed frames, but keep the fit fail-closed on division).
            usable = np.abs(local_theta) > 1.0e-12
            if int(np.count_nonzero(usable)) < _MIN_LOCAL_SAMPLES:
                response[index] = slope * float(knot)
            else:
                ratios = phi[local][usable] / local_theta[usable]
                response[index] = float(np.mean(ratios)) * float(knot)
        else:
            response[index] = slope * float(knot)
    # Response is negative under flexion; cumulative minimum keeps it
    # non-increasing so later knots cannot undo earlier contact rotation.
    response = np.minimum.accumulate(response)
    response[0] = 0.0
    return response, slope


def _penetration_envelope_m(
    action: Mapping[str, np.ndarray],
    *,
    side: str,
) -> float:
    try:
        import igl
    except Exception as exc:
        raise ValueError(
            "igl is required for patella penetration envelopes; import failed"
        ) from exc
    suffix = _SIDE_SUFFIX[side]
    patella_key = f"mesh__Patella_{suffix}__vertices"
    femur_v_key = f"mesh__Femur_{suffix}__vertices"
    femur_f_key = f"mesh__Femur_{suffix}__faces"
    for key in (patella_key, femur_v_key, femur_f_key):
        if key not in action:
            raise ValueError(f"action is missing mesh array {key!r}")
    patella = np.asarray(action[patella_key], dtype=np.float64)
    femur_v = np.asarray(action[femur_v_key], dtype=np.float64)
    femur_f = np.asarray(action[femur_f_key], dtype=np.int32)
    if patella.ndim != 3 or femur_v.ndim != 3:
        raise ValueError(f"mesh vertex arrays for {side} must be [F,V,3]")
    if patella.shape[0] != femur_v.shape[0]:
        raise ValueError(f"patella/femur frame counts disagree for {side}")
    minima: list[float] = []
    for frame in range(0, int(patella.shape[0]), _MESH_SAMPLE_STRIDE):
        signed = igl.signed_distance(
            np.ascontiguousarray(patella[frame]),
            np.ascontiguousarray(femur_v[frame]),
            np.ascontiguousarray(femur_f),
        )[0]
        minima.append(float(np.min(np.asarray(signed, dtype=np.float64))))
    if not minima:
        raise ValueError(f"no frames available for {side} penetration envelope")
    return float(abs(min(0.0, min(minima))))


@dataclass(frozen=True)
class PatellaOracleLawV7:
    knots_deg: np.ndarray
    response_deg: Mapping[str, np.ndarray]
    axis_patella_local: Mapping[str, np.ndarray]
    axis_knee_local: Mapping[str, np.ndarray]
    response_slope: Mapping[str, float]
    response_max_residual_deg: Mapping[str, float]
    keyed_frame_count: Mapping[str, int]
    observed_max_flexion_deg: Mapping[str, float]
    penetration_envelope_m: Mapping[str, float]
    corridor_min_m: float
    corridor_max_m: float
    corridor_target_m: float
    max_contact_translation_m: float
    action_source_digest: str
    action_frame_count: int
    provenance: Mapping[str, Any]
    schema_version: int = PATELLA_ORACLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "knots_deg", np.asarray(self.knots_deg, dtype=np.float64).reshape(-1)
        )
        object.__setattr__(
            self,
            "response_deg",
            MappingProxyType(
                {
                    side: np.asarray(self.response_deg[side], dtype=np.float64).reshape(-1)
                    for side in SIDES
                }
            ),
        )
        object.__setattr__(
            self,
            "axis_patella_local",
            MappingProxyType(
                {
                    side: _as_unit3(self.axis_patella_local[side], f"axis_patella_local[{side}]")
                    for side in SIDES
                }
            ),
        )
        object.__setattr__(
            self,
            "axis_knee_local",
            MappingProxyType(
                {
                    side: _as_unit3(self.axis_knee_local[side], f"axis_knee_local[{side}]")
                    for side in SIDES
                }
            ),
        )
        object.__setattr__(
            self,
            "response_slope",
            MappingProxyType({side: float(self.response_slope[side]) for side in SIDES}),
        )
        object.__setattr__(
            self,
            "response_max_residual_deg",
            MappingProxyType(
                {side: float(self.response_max_residual_deg[side]) for side in SIDES}
            ),
        )
        object.__setattr__(
            self,
            "keyed_frame_count",
            MappingProxyType(
                {side: int(self.keyed_frame_count[side]) for side in SIDES}
            ),
        )
        object.__setattr__(
            self,
            "observed_max_flexion_deg",
            MappingProxyType(
                {side: float(self.observed_max_flexion_deg[side]) for side in SIDES}
            ),
        )
        object.__setattr__(
            self,
            "penetration_envelope_m",
            MappingProxyType(
                {side: float(self.penetration_envelope_m[side]) for side in SIDES}
            ),
        )
        object.__setattr__(
            self,
            "provenance",
            MappingProxyType(dict(self.provenance)),
        )

    def validate(self) -> None:
        if int(self.schema_version) != PATELLA_ORACLE_SCHEMA_VERSION:
            raise ValueError(
                f"patella oracle schema must be {PATELLA_ORACLE_SCHEMA_VERSION}, "
                f"got {self.schema_version}"
            )
        knots = np.asarray(self.knots_deg, dtype=np.float64).reshape(-1)
        if len(knots) < 2:
            raise ValueError("knots_deg must contain at least two knots")
        if float(knots[0]) != 0.0:
            raise ValueError("knots_deg[0] must be 0.0")
        if not np.all(np.diff(knots) > 0.0):
            raise ValueError("knots_deg must be strictly ascending")
        if not np.all(np.isfinite(knots)):
            raise ValueError("knots_deg must be finite")
        if not str(self.action_source_digest) or len(str(self.action_source_digest)) != 64:
            raise ValueError("action_source_digest must be a full SHA-256 hex digest")
        if int(self.action_frame_count) < 1:
            raise ValueError("action_frame_count must be positive")
        if not (
            float(self.corridor_min_m)
            <= float(self.corridor_target_m)
            <= float(self.corridor_max_m)
        ):
            raise ValueError("corridor bounds are inconsistent")
        if float(self.max_contact_translation_m) <= 0.0:
            raise ValueError("max_contact_translation_m must be positive")
        for side in SIDES:
            response = np.asarray(self.response_deg[side], dtype=np.float64).reshape(-1)
            if response.shape != knots.shape:
                raise ValueError(f"response_deg[{side}] must match knots_deg")
            if float(response[0]) != 0.0:
                raise ValueError(f"response_deg[{side}][0] must be 0.0")
            if not np.all(np.isfinite(response)):
                raise ValueError(f"response_deg[{side}] must be finite")
            if np.any(np.diff(response) > 1.0e-12):
                raise ValueError(f"response_deg[{side}] must be non-increasing")
            if int(self.keyed_frame_count[side]) < _MIN_KEYED_FRAMES:
                raise ValueError(
                    f"{side} keyed_frame_count must be >= {_MIN_KEYED_FRAMES}"
                )
            if float(self.penetration_envelope_m[side]) < 0.0:
                raise ValueError(f"penetration_envelope_m[{side}] must be non-negative")
            _as_unit3(self.axis_patella_local[side], f"axis_patella_local[{side}]")
            _as_unit3(self.axis_knee_local[side], f"axis_knee_local[{side}]")
        provenance = dict(self.provenance)
        for key in (
            "method",
            "action_path",
            "bone_names",
            "unit_note",
            "keyed_frame_selection",
            "forbidden_oracle_note",
        ):
            if key not in provenance:
                raise ValueError(f"provenance.{key} is required")

    def response_rad(
        self, side: str, flexion_rad: float | np.ndarray
    ) -> np.ndarray | float:
        """Patella rotation (rad) as a function of anatomical knee flexion.

        Knots start at 0 deg (straight leg).  Hyperextension (``flexion_rad < 0``)
        is clamped to 0 so the response stays at the rest value rather than
        inventing an odd extension through unsigned interpolation.  Positive
        flexion looks up ``response_deg`` directly (those samples already carry
        the authored sign for flexion, typically negative).
        """
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        self.validate()
        values = np.asarray(flexion_rad, dtype=np.float64)
        scalar = values.ndim == 0
        flat = np.atleast_1d(values)
        if not np.all(np.isfinite(flat)):
            raise ValueError("flexion_rad must be finite")
        # Explicit below-zero policy: clamp to the first knot (straight / rest).
        flex_deg = np.degrees(np.maximum(flat, 0.0))
        response_deg = _interp_response_deg(
            self.knots_deg, self.response_deg[side], flex_deg
        )
        out = np.radians(response_deg)
        if scalar:
            return float(out[0])
        return out

    def content_digest(self) -> str:
        self.validate()
        digest = hashlib.sha256()
        digest.update(PATELLA_ORACLE_KIND.encode("utf-8"))
        digest.update(f"schema:{int(self.schema_version)}".encode("ascii"))
        _array_digest(digest, "knots_deg", np.asarray(self.knots_deg, dtype=np.float64))
        for side in SIDES:
            _array_digest(
                digest,
                f"response_deg.{side}",
                np.asarray(self.response_deg[side], dtype=np.float64),
            )
            _array_digest(
                digest,
                f"axis_patella_local.{side}",
                np.asarray(self.axis_patella_local[side], dtype=np.float64),
            )
            _array_digest(
                digest,
                f"axis_knee_local.{side}",
                np.asarray(self.axis_knee_local[side], dtype=np.float64),
            )
            digest.update(
                f"response_slope.{side}:{float(self.response_slope[side]):.17g}".encode(
                    "ascii"
                )
            )
            digest.update(
                f"response_max_residual_deg.{side}:"
                f"{float(self.response_max_residual_deg[side]):.17g}".encode("ascii")
            )
            digest.update(
                f"keyed_frame_count.{side}:{int(self.keyed_frame_count[side])}".encode(
                    "ascii"
                )
            )
            digest.update(
                f"observed_max_flexion_deg.{side}:"
                f"{float(self.observed_max_flexion_deg[side]):.17g}".encode("ascii")
            )
            digest.update(
                f"penetration_envelope_m.{side}:"
                f"{float(self.penetration_envelope_m[side]):.17g}".encode("ascii")
            )
        for name, value in (
            ("corridor_min_m", self.corridor_min_m),
            ("corridor_max_m", self.corridor_max_m),
            ("corridor_target_m", self.corridor_target_m),
            ("max_contact_translation_m", self.max_contact_translation_m),
        ):
            digest.update(f"{name}:{float(value):.17g}".encode("ascii"))
        digest.update(f"action_source_digest:{self.action_source_digest}".encode("ascii"))
        digest.update(
            f"action_frame_count:{int(self.action_frame_count)}".encode("ascii")
        )
        digest.update(
            json.dumps(dict(self.provenance), sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        return digest.hexdigest()


def extract_patella_law_v7(
    action_path: Path | str,
    *,
    knots_deg: np.ndarray | None = None,
    minimum_flexion_deg: float = 5.0,
    minimum_response_deg: float = 0.5,
    corridor_min_m: float = 0.0,
    corridor_max_m: float = 0.004,
    corridor_target_m: float = 0.0015,
    max_contact_translation_m: float = 0.005,
) -> PatellaOracleLawV7:
    path = Path(action_path)
    if not path.is_file():
        raise ValueError(f"action file does not exist: {path}")
    source_digest = _sha256_file(path)
    action = np.load(path, allow_pickle=False)
    try:
        bone_names = [str(name) for name in action["bone_names"]]
        rest_local = np.asarray(action["bone_rest_local"], dtype=np.float64)
        action_local = np.asarray(action["bone_action_local"], dtype=np.float64)
    except Exception as exc:
        raise ValueError("action file is missing required bone arrays") from exc
    if action_local.ndim != 4 or action_local.shape[-2:] != (4, 4):
        raise ValueError("bone_action_local must be [F,B,4,4]")
    if rest_local.shape != (action_local.shape[1], 4, 4):
        raise ValueError("bone_rest_local must be [B,4,4] matching action bones")
    frame_count = int(action_local.shape[0])
    if knots_deg is None:
        knots = np.linspace(0.0, 120.0, 25, dtype=np.float64)
    else:
        knots = _as_float64_1d(knots_deg, "knots_deg")
        if len(knots) < 2 or float(knots[0]) != 0.0 or not np.all(np.diff(knots) > 0.0):
            raise ValueError("knots_deg must start at 0 and be strictly ascending")

    response_deg: dict[str, np.ndarray] = {}
    response_slope: dict[str, float] = {}
    response_max_residual_deg: dict[str, float] = {}
    keyed_frame_count: dict[str, int] = {}
    observed_max_flexion_deg: dict[str, float] = {}
    penetration_envelope_m: dict[str, float] = {}
    axis_patella_local: dict[str, np.ndarray] = {}
    axis_knee_local: dict[str, np.ndarray] = {}
    bone_names_used: dict[str, dict[str, str]] = {}

    for side in SIDES:
        names = _side_bone_names(side)
        bone_names_used[side] = dict(names)
        knee_index = _bone_index(bone_names, names["knee"])
        patella_index = _bone_index(bone_names, names["patella"])
        # Fail closed on any local translation drift: the V71 law is pure rotation.
        rest_translation = rest_local[patella_index, :3, 3]
        drifts = np.linalg.norm(
            action_local[:, patella_index, :3, 3] - rest_translation[None, :],
            axis=1,
        )
        if float(np.max(drifts)) > _MAX_TRANSLATION_DRIFT:
            raise ValueError(
                f"{side} patella local translation drift exceeds {_MAX_TRANSLATION_DRIFT}"
            )

        theta = np.empty(frame_count, dtype=np.float64)
        phi = np.empty(frame_count, dtype=np.float64)
        off_axis = np.empty(frame_count, dtype=np.float64)
        for frame in range(frame_count):
            theta[frame], _ = _signed_axis_angle(
                rest_local[knee_index],
                action_local[frame, knee_index],
                _AXIS_LOCAL,
            )
            phi[frame], off_axis[frame] = _signed_axis_angle(
                rest_local[patella_index],
                action_local[frame, patella_index],
                _AXIS_LOCAL,
            )
        theta_deg = np.degrees(theta)
        phi_deg = np.degrees(phi)
        off_axis_deg = np.degrees(off_axis)
        keyed = (np.abs(theta_deg) > float(minimum_flexion_deg)) & (
            np.abs(phi_deg) > float(minimum_response_deg)
        )
        keyed_count = int(np.count_nonzero(keyed))
        if keyed_count < _MIN_KEYED_FRAMES:
            raise ValueError(
                f"{side} has only {keyed_count} keyed frames; need >= {_MIN_KEYED_FRAMES}"
            )
        if float(np.max(off_axis_deg[keyed])) > _MAX_OFF_AXIS_DEG:
            raise ValueError(
                f"{side} patella rotation has off-axis residual above {_MAX_OFF_AXIS_DEG} deg"
            )

        fitted, slope = _fit_response_deg(theta_deg[keyed], phi_deg[keyed], knots)
        predicted = _interp_response_deg(knots, fitted, np.abs(theta_deg[keyed]))
        # Preserve the signed response: keyed theta is positive under flexion, so
        # sign(theta) recovers the odd extension used by response_rad.
        predicted *= np.sign(theta_deg[keyed])
        residual = float(np.max(np.abs(phi_deg[keyed] - predicted)))

        response_deg[side] = fitted
        response_slope[side] = float(slope)
        response_max_residual_deg[side] = residual
        keyed_frame_count[side] = keyed_count
        observed_max_flexion_deg[side] = float(np.max(np.abs(theta_deg)))
        penetration_envelope_m[side] = _penetration_envelope_m(action, side=side)
        axis_patella_local[side] = _AXIS_LOCAL.copy()
        axis_knee_local[side] = _AXIS_LOCAL.copy()

    provenance = {
        "method": "v71_action_local_ls_monotone_response",
        # Only the file name enters provenance: the digest must identify the law,
        # not the directory a reviewer happened to run from.  action_source_digest
        # already pins the exact bytes.
        "action_path": str(path.name),
        "bone_names": bone_names_used,
        "unit_note": (
            "bone matrices are in Blender/armature centimetres; exported mesh "
            "vertex arrays are in metres; only bone matrices are used for the "
            "response law"
        ),
        "keyed_frame_selection": {
            "minimum_flexion_deg": float(minimum_flexion_deg),
            "minimum_response_deg": float(minimum_response_deg),
        },
        "forbidden_oracle_note": (
            "the evaluated patella surface trajectory is NOT used as a target "
            "and is only used for the penetration envelope"
        ),
    }
    law = PatellaOracleLawV7(
        knots_deg=knots,
        response_deg=response_deg,
        axis_patella_local=axis_patella_local,
        axis_knee_local=axis_knee_local,
        response_slope=response_slope,
        response_max_residual_deg=response_max_residual_deg,
        keyed_frame_count=keyed_frame_count,
        observed_max_flexion_deg=observed_max_flexion_deg,
        penetration_envelope_m=penetration_envelope_m,
        corridor_min_m=float(corridor_min_m),
        corridor_max_m=float(corridor_max_m),
        corridor_target_m=float(corridor_target_m),
        max_contact_translation_m=float(max_contact_translation_m),
        action_source_digest=source_digest,
        action_frame_count=frame_count,
        provenance=provenance,
    )
    law.validate()
    return law


def save_patella_oracle_v7(path: Path | str, law: PatellaOracleLawV7) -> Path:
    law.validate()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": np.asarray(int(law.schema_version), dtype=np.int32),
        "kind": np.asarray(PATELLA_ORACLE_KIND),
        "knots_deg": np.asarray(law.knots_deg, dtype=np.float64),
        "corridor_min_m": np.asarray(float(law.corridor_min_m), dtype=np.float64),
        "corridor_max_m": np.asarray(float(law.corridor_max_m), dtype=np.float64),
        "corridor_target_m": np.asarray(float(law.corridor_target_m), dtype=np.float64),
        "max_contact_translation_m": np.asarray(
            float(law.max_contact_translation_m), dtype=np.float64
        ),
        "action_source_digest": np.asarray(str(law.action_source_digest)),
        "action_frame_count": np.asarray(int(law.action_frame_count), dtype=np.int32),
        "content_digest": np.asarray(law.content_digest()),
        "provenance_json": np.asarray(
            json.dumps(dict(law.provenance), sort_keys=True, separators=(",", ":"))
        ),
    }
    for side in SIDES:
        payload[f"response_deg__{side}"] = np.asarray(
            law.response_deg[side], dtype=np.float64
        )
        payload[f"axis_patella_local__{side}"] = np.asarray(
            law.axis_patella_local[side], dtype=np.float64
        )
        payload[f"axis_knee_local__{side}"] = np.asarray(
            law.axis_knee_local[side], dtype=np.float64
        )
        payload[f"response_slope__{side}"] = np.asarray(
            float(law.response_slope[side]), dtype=np.float64
        )
        payload[f"response_max_residual_deg__{side}"] = np.asarray(
            float(law.response_max_residual_deg[side]), dtype=np.float64
        )
        payload[f"keyed_frame_count__{side}"] = np.asarray(
            int(law.keyed_frame_count[side]), dtype=np.int32
        )
        payload[f"observed_max_flexion_deg__{side}"] = np.asarray(
            float(law.observed_max_flexion_deg[side]), dtype=np.float64
        )
        payload[f"penetration_envelope_m__{side}"] = np.asarray(
            float(law.penetration_envelope_m[side]), dtype=np.float64
        )
    np.savez(output, **payload)
    return output


def load_patella_oracle_v7(path: Path | str) -> PatellaOracleLawV7:
    data = np.load(Path(path), allow_pickle=False)
    if "schema_version" not in data.files:
        raise ValueError("patella oracle is missing schema_version")
    schema = int(np.asarray(data["schema_version"]).reshape(-1)[0])
    if schema != PATELLA_ORACLE_SCHEMA_VERSION:
        raise ValueError(
            f"patella oracle schema must be {PATELLA_ORACLE_SCHEMA_VERSION}, got {schema}"
        )
    kind = str(np.asarray(data["kind"]).reshape(-1)[0])
    if kind != PATELLA_ORACLE_KIND:
        raise ValueError(f"unexpected patella oracle kind {kind!r}")
    expected = str(np.asarray(data["content_digest"]).reshape(-1)[0])
    provenance = json.loads(str(np.asarray(data["provenance_json"]).reshape(-1)[0]))
    law = PatellaOracleLawV7(
        knots_deg=np.asarray(data["knots_deg"], dtype=np.float64),
        response_deg={
            side: np.asarray(data[f"response_deg__{side}"], dtype=np.float64)
            for side in SIDES
        },
        axis_patella_local={
            side: np.asarray(data[f"axis_patella_local__{side}"], dtype=np.float64)
            for side in SIDES
        },
        axis_knee_local={
            side: np.asarray(data[f"axis_knee_local__{side}"], dtype=np.float64)
            for side in SIDES
        },
        response_slope={
            side: float(np.asarray(data[f"response_slope__{side}"]).reshape(-1)[0])
            for side in SIDES
        },
        response_max_residual_deg={
            side: float(
                np.asarray(data[f"response_max_residual_deg__{side}"]).reshape(-1)[0]
            )
            for side in SIDES
        },
        keyed_frame_count={
            side: int(np.asarray(data[f"keyed_frame_count__{side}"]).reshape(-1)[0])
            for side in SIDES
        },
        observed_max_flexion_deg={
            side: float(
                np.asarray(data[f"observed_max_flexion_deg__{side}"]).reshape(-1)[0]
            )
            for side in SIDES
        },
        penetration_envelope_m={
            side: float(
                np.asarray(data[f"penetration_envelope_m__{side}"]).reshape(-1)[0]
            )
            for side in SIDES
        },
        corridor_min_m=float(np.asarray(data["corridor_min_m"]).reshape(-1)[0]),
        corridor_max_m=float(np.asarray(data["corridor_max_m"]).reshape(-1)[0]),
        corridor_target_m=float(np.asarray(data["corridor_target_m"]).reshape(-1)[0]),
        max_contact_translation_m=float(
            np.asarray(data["max_contact_translation_m"]).reshape(-1)[0]
        ),
        action_source_digest=str(np.asarray(data["action_source_digest"]).reshape(-1)[0]),
        action_frame_count=int(np.asarray(data["action_frame_count"]).reshape(-1)[0]),
        provenance=provenance,
        schema_version=schema,
    )
    digest = law.content_digest()
    if digest != expected:
        raise ValueError(
            "stored patella oracle content_digest does not match recomputed digest"
        )
    return law


def patella_bind_frames_v7(asset: Any, *, side: str) -> dict[str, Any]:
    names = _side_bone_names(side)
    bone_names = list(asset.source_bone_names)
    indices = {key: _bone_index(bone_names, name) for key, name in names.items()}
    bind_global = np.asarray(asset.target_bind_global, dtype=np.float64)
    if bind_global.ndim != 3 or bind_global.shape[-2:] != (4, 4):
        raise ValueError("asset.target_bind_global must be [B,4,4]")
    frames: dict[str, Any] = {
        "femur": int(indices["femur"]),
        "knee": int(indices["knee"]),
        "tibia": int(indices["tibia"]),
        "patella": int(indices["patella"]),
    }
    for key in ("femur", "knee", "tibia", "patella"):
        frames[f"{key}_bind"] = np.asarray(
            bind_global[indices[key]], dtype=np.float64
        ).copy()
    bg_femur = frames["femur_bind"]
    bg_knee = frames["knee_bind"]
    bg_tibia = frames["tibia_bind"]
    bg_patella = frames["patella_bind"]
    frames["knee_local"] = np.linalg.inv(bg_femur) @ bg_knee
    frames["tibia_local"] = np.linalg.inv(bg_knee) @ bg_tibia
    frames["patella_local"] = np.linalg.inv(bg_tibia) @ bg_patella
    return frames


def patella_world_transform_v7(
    law: PatellaOracleLawV7,
    *,
    frames: Mapping[str, Any],
    side: str,
    flexion_rad: float,
    knee_axis_local: np.ndarray | None = None,
    contact_translation_parent_local_m: np.ndarray | None = None,
) -> np.ndarray:
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    law.validate()
    knee_axis = (
        _as_unit3(knee_axis_local, "knee_axis_local")
        if knee_axis_local is not None
        else np.asarray(law.axis_knee_local[side], dtype=np.float64)
    )
    patella_axis = np.asarray(law.axis_patella_local[side], dtype=np.float64)
    c_knee = _rotation4(knee_axis, float(flexion_rad))
    c_patella = _rotation4(
        patella_axis, float(law.response_rad(side, float(flexion_rad)))
    )
    l_p = np.asarray(frames["patella_local"], dtype=np.float64).copy()
    if contact_translation_parent_local_m is not None:
        delta = np.asarray(contact_translation_parent_local_m, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(delta)):
            raise ValueError("contact_translation_parent_local_m must be finite")
        l_p[:3, 3] = l_p[:3, 3] + delta
    bg_femur = np.asarray(frames["femur_bind"], dtype=np.float64)
    knee_local = np.asarray(frames["knee_local"], dtype=np.float64)
    tibia_local = np.asarray(frames["tibia_local"], dtype=np.float64)
    bg_patella = np.asarray(frames["patella_bind"], dtype=np.float64)
    return (
        bg_femur
        @ knee_local
        @ c_knee
        @ tibia_local
        @ l_p
        @ c_patella
        @ np.linalg.inv(bg_patella)
    )


def _smooth_translation_table(
    table: np.ndarray, *, max_norm_m: float
) -> np.ndarray:
    glide = np.asarray(table, dtype=np.float64).copy()
    if len(glide) > 2:
        padded = np.pad(glide, ((1, 1), (0, 0)), mode="edge")
        glide = (padded[:-2] + 2.0 * padded[1:-1] + padded[2:]) / 4.0
    glide[0] = 0.0
    norms = np.linalg.norm(glide, axis=1)
    active = norms > float(max_norm_m)
    if np.any(active):
        glide[active] *= (float(max_norm_m) / norms[active])[:, None]
    return glide


def _hard_min_distance(points: np.ndarray, target: np.ndarray) -> float:
    return float(np.min(_nearest_distances(points, target)))


def _contact_translation_for_gap(
    *,
    posed_patella: np.ndarray,
    femur_points: np.ndarray,
    target_gap_m: float,
    max_translation_m: float,
) -> np.ndarray:
    def measurements(translation: np.ndarray) -> float:
        moved = posed_patella + np.asarray(translation, dtype=np.float64).reshape(1, 3)
        distances = _nearest_distances(moved, femur_points)
        return float(np.quantile(distances, 0.05))

    bound = float(max_translation_m)
    try:
        from scipy.optimize import least_squares

        solved = least_squares(
            lambda translation: np.concatenate(
                (
                    np.asarray(
                        [(measurements(translation) - float(target_gap_m)) / 0.001],
                        dtype=np.float64,
                    ),
                    np.asarray(translation, dtype=np.float64) / max(bound, 1.0e-9) * 0.04,
                )
            ),
            np.zeros(3, dtype=np.float64),
            bounds=(-bound, bound),
            max_nfev=256,
            xtol=1.0e-11,
            ftol=1.0e-11,
            gtol=1.0e-11,
        )
        translation = np.asarray(solved.x, dtype=np.float64)
    except Exception:
        # Deterministic fallback: move the patella centroid toward the femur
        # centroid by the unsigned gap error, clipped to the translation budget.
        delta = np.mean(femur_points, axis=0) - np.mean(posed_patella, axis=0)
        current = measurements(np.zeros(3))
        direction = delta / max(float(np.linalg.norm(delta)), 1.0e-12)
        translation = direction * float(current - target_gap_m)
        norm = float(np.linalg.norm(translation))
        if norm > bound:
            translation *= bound / norm
    norm = float(np.linalg.norm(translation))
    if norm > bound:
        translation = translation * (bound / norm)
    return translation


def solve_patella_contact_corrections_v7(
    law: PatellaOracleLawV7,
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    domains: FrozenJointMaterialDomainsV7,
    asset: Any,
    side: str,
    knots_deg: np.ndarray | None = None,
    knee_axis_local: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    del faces  # topology is frozen in domains; contact uses point clouds only
    law.validate()
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("vertices must be [N,3]")
    patella_ids = np.asarray(domains.require(f"{side}/patella"), dtype=np.int64)
    femur_ids = np.asarray(domains.require(f"{side}/femur"), dtype=np.int64)
    patella_bind = points[patella_ids]
    femur_points = points[femur_ids]
    frames = patella_bind_frames_v7(asset, side=side)
    knots = (
        np.asarray(law.knots_deg, dtype=np.float64).reshape(-1)
        if knots_deg is None
        else _as_float64_1d(knots_deg, "knots_deg")
    )
    translations = np.zeros((len(knots), 3), dtype=np.float64)
    per_knot: list[dict[str, Any]] = []
    # Contact translation is expressed in the patella PARENT-LOCAL frame.  The
    # world residual is therefore mapped back through the posed tibia rotation.
    for index, knot in enumerate(knots):
        flexion = float(np.radians(knot))
        world = patella_world_transform_v7(
            law,
            frames=frames,
            side=side,
            flexion_rad=flexion,
            knee_axis_local=knee_axis_local,
        )
        posed = (world[:3, :3] @ patella_bind.T).T + world[:3, 3]
        uncorrected = _hard_min_distance(posed, femur_points)
        deadband_lo = float(law.corridor_min_m) + _DEADBAND_LOOSE_M
        deadband_hi = float(law.corridor_max_m) - _DEADBAND_TIGHT_M
        if deadband_lo <= uncorrected <= deadband_hi:
            parent_local = np.zeros(3, dtype=np.float64)
        else:
            # Approximate parent-local axes from the posed patella parent
            # (tibia) frame embedded in the FK chain before L_p.
            bg_femur = frames["femur_bind"]
            knee_local = frames["knee_local"]
            tibia_local = frames["tibia_local"]
            c_knee = _rotation4(
                (
                    _as_unit3(knee_axis_local, "knee_axis_local")
                    if knee_axis_local is not None
                    else law.axis_knee_local[side]
                ),
                flexion,
            )
            parent_world = bg_femur @ knee_local @ c_knee @ tibia_local
            parent_rotation = parent_world[:3, :3]
            world_translation = _contact_translation_for_gap(
                posed_patella=posed,
                femur_points=femur_points,
                target_gap_m=float(law.corridor_target_m),
                max_translation_m=float(law.max_contact_translation_m),
            )
            parent_local = parent_rotation.T @ world_translation
            norm = float(np.linalg.norm(parent_local))
            if norm > float(law.max_contact_translation_m):
                parent_local *= float(law.max_contact_translation_m) / norm
        translations[index] = parent_local
        corrected_world = patella_world_transform_v7(
            law,
            frames=frames,
            side=side,
            flexion_rad=flexion,
            knee_axis_local=knee_axis_local,
            contact_translation_parent_local_m=parent_local,
        )
        corrected_points = (
            (corrected_world[:3, :3] @ patella_bind.T).T + corrected_world[:3, 3]
        )
        corrected = _hard_min_distance(corrected_points, femur_points)
        per_knot.append(
            {
                "flexion_deg": float(knot),
                "uncorrected_hard_min_m": float(uncorrected),
                "corrected_hard_min_m": float(corrected),
                "translation_norm_m": float(np.linalg.norm(parent_local)),
                "corridor_satisfied": bool(
                    float(law.corridor_min_m)
                    <= corrected
                    <= float(law.corridor_max_m)
                ),
            }
        )

    translations = _smooth_translation_table(
        translations, max_norm_m=float(law.max_contact_translation_m)
    )
    # Re-evaluate corridor after smoothing so the report matches the returned table.
    per_knot_final: list[dict[str, Any]] = []
    for index, knot in enumerate(knots):
        flexion = float(np.radians(knot))
        world = patella_world_transform_v7(
            law,
            frames=frames,
            side=side,
            flexion_rad=flexion,
            knee_axis_local=knee_axis_local,
        )
        posed = (world[:3, :3] @ patella_bind.T).T + world[:3, 3]
        uncorrected = _hard_min_distance(posed, femur_points)
        corrected_world = patella_world_transform_v7(
            law,
            frames=frames,
            side=side,
            flexion_rad=flexion,
            knee_axis_local=knee_axis_local,
            contact_translation_parent_local_m=translations[index],
        )
        corrected_points = (
            (corrected_world[:3, :3] @ patella_bind.T).T + corrected_world[:3, 3]
        )
        corrected = _hard_min_distance(corrected_points, femur_points)
        per_knot_final.append(
            {
                "flexion_deg": float(knot),
                "uncorrected_hard_min_m": float(uncorrected),
                "corrected_hard_min_m": float(corrected),
                "translation_norm_m": float(np.linalg.norm(translations[index])),
                "corridor_satisfied": bool(
                    float(law.corridor_min_m)
                    <= corrected
                    <= float(law.corridor_max_m)
                ),
            }
        )

    corrected_gaps = np.asarray(
        [row["corrected_hard_min_m"] for row in per_knot_final], dtype=np.float64
    )
    violations = np.maximum(
        float(law.corridor_min_m) - corrected_gaps,
        corrected_gaps - float(law.corridor_max_m),
    )
    violations = np.maximum(violations, 0.0)
    report = {
        "per_knot": per_knot_final,
        "max_translation_m": float(np.max(np.linalg.norm(translations, axis=1))),
        "corridor_satisfied": bool(all(row["corridor_satisfied"] for row in per_knot_final)),
        "worst_gap_m": float(np.max(violations)),
    }
    return translations, report


def _contact_translation_at_flexion(
    table: np.ndarray,
    knots_deg: np.ndarray,
    flexion_rad: float,
) -> np.ndarray:
    knots = np.asarray(knots_deg, dtype=np.float64).reshape(-1)
    values = np.asarray(table, dtype=np.float64)
    if values.shape != (len(knots), 3):
        raise ValueError("side contact translations must be [K,3] matching knots")
    angle = float(np.degrees(abs(float(flexion_rad))))
    if angle <= knots[-1]:
        return np.asarray(
            [np.interp(angle, knots, values[:, axis]) for axis in range(3)],
            dtype=np.float64,
        )
    slope = (values[-1] - values[-2]) / (knots[-1] - knots[-2])
    return values[-1] + slope * (angle - knots[-1])


def patella_oracle_sweep_v7(
    law: PatellaOracleLawV7,
    *,
    asset: Any,
    domains: FrozenJointMaterialDomainsV7,
    flexion_rad: np.ndarray,
    side_contact_translations: Mapping[str, np.ndarray] | None = None,
    knee_axis_local: Mapping[str, np.ndarray] | None = None,
    base_vertices: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    law.validate()
    flexion = np.asarray(flexion_rad, dtype=np.float64).reshape(-1)
    if not len(flexion):
        raise ValueError("flexion_rad must be non-empty")
    if float(flexion[0]) != 0.0:
        raise ValueError("flexion_rad[0] must be 0.0 for relative motion metrics")
    if not np.all(np.isfinite(flexion)):
        raise ValueError("flexion_rad must be finite")
    base = np.asarray(
        asset.vertices_rest if base_vertices is None else base_vertices,
        dtype=np.float64,
    )
    if base.ndim != 2 or base.shape[1] != 3:
        raise ValueError("base_vertices must be [N,3]")
    sweep = np.repeat(base[None, :, :], len(flexion), axis=0).astype(np.float32)
    per_side_min: dict[str, list[float]] = {side: [] for side in SIDES}
    frames_by_side = {side: patella_bind_frames_v7(asset, side=side) for side in SIDES}
    for pose_index, angle in enumerate(flexion):
        for side in SIDES:
            patella_ids = np.asarray(domains.require(f"{side}/patella"), dtype=np.int64)
            femur_ids = np.asarray(domains.require(f"{side}/femur"), dtype=np.int64)
            contact = None
            if side_contact_translations is not None and side in side_contact_translations:
                table = np.asarray(side_contact_translations[side], dtype=np.float64)
                if table.shape == (len(flexion), 3):
                    contact = table[pose_index]
                else:
                    contact = _contact_translation_at_flexion(
                        table, law.knots_deg, float(angle)
                    )
            knee_axis = None
            if knee_axis_local is not None and side in knee_axis_local:
                knee_axis = np.asarray(knee_axis_local[side], dtype=np.float64)
            world = patella_world_transform_v7(
                law,
                frames=frames_by_side[side],
                side=side,
                flexion_rad=float(angle),
                knee_axis_local=knee_axis,
                contact_translation_parent_local_m=contact,
            )
            bind = base[patella_ids]
            posed = (world[:3, :3] @ bind.T).T + world[:3, 3]
            sweep[pose_index, patella_ids] = posed.astype(np.float32)
            per_side_min[side].append(
                _hard_min_distance(posed, base[femur_ids])
            )
    report = {
        "law_digest": law.content_digest(),
        "min_patella_femur_distance_m": {
            side: np.asarray(values, dtype=np.float64) for side, values in per_side_min.items()
        },
    }
    return sweep, report


__all__ = [
    "PATELLA_ORACLE_SCHEMA_VERSION",
    "PATELLA_ORACLE_KIND",
    "SIDES",
    "PatellaOracleLawV7",
    "extract_patella_law_v7",
    "save_patella_oracle_v7",
    "load_patella_oracle_v7",
    "patella_bind_frames_v7",
    "patella_world_transform_v7",
    "solve_patella_contact_corrections_v7",
    "patella_oracle_sweep_v7",
]
