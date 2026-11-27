"""Label-blind short-axis echo simulator (world XYZ and probe-local XYZ in mm).

This is a geometric sensing model, not a wave or B-mode image simulator. The
caller supplies in-station lumen samples, acoustic surfaces in the *current*
world frame, and same-window dwell time. The pose uses a point-aperture ray
model and a finite elevational slice; shadows are opaque, not partial-volume
attenuation or diffraction. A planar finite reflector models one reciprocal
specular path via its mirror image, not arbitrary multi-bounce reverberation.
"""

from dataclasses import dataclass, field
import hashlib
import math
import operator
import struct
from collections.abc import Sequence
from typing import Literal

import numpy as np


ObservationMode = Literal["centroid_only", "centroid_caliber", "full_physical"]


def _finite_vector(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite XYZ vector")
    return result


def _nonnegative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _positive(value: float, name: str) -> None:
    _nonnegative(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class ProbePose:
    """Right-handed rotation with world-space lateral, elevation, depth columns."""

    origin_mm: np.ndarray
    rotation: np.ndarray
    speed_mm_s: float = 0.0
    dwell_s: float = 0.0

    def __post_init__(self) -> None:
        origin = _finite_vector(self.origin_mm, "origin_mm")
        r = np.asarray(self.rotation, dtype=np.float64)
        if (r.shape != (3, 3) or not np.all(np.isfinite(r))
                or not np.allclose(r.T @ r, np.eye(3), rtol=0, atol=1e-6)
                or np.linalg.det(r) <= 0):
            raise ValueError("rotation must be a right-handed orthonormal 3x3 matrix")
        _nonnegative(self.speed_mm_s, "speed_mm_s")
        _nonnegative(self.dwell_s, "dwell_s")
        object.__setattr__(self, "origin_mm", origin)
        object.__setattr__(self, "rotation", r)


@dataclass(frozen=True)
class VesselTruth:
    """One lumen sample with independent acoustic/mechanical properties.

    Diameter is the area-equivalent baseline caliber; compliance is fractional
    diameter decrease per N of *excess* force. Pulse amplitude is fractional
    diameter modulation. The optional task role is metadata, not acoustic input.
    An optional noise_key is an independent episode-stable random token, never
    a role or list index. Keep it fixed when transporting this physical lumen.
    """

    center_mm: np.ndarray
    diam_mm: float
    compliance_per_n: float = 0.0
    pulse_amplitude: float = 0.0
    comp_ratio_std: float = 0.05
    pulse_amp_std: float = 0.02
    role: int | None = field(default=None, compare=False, repr=False)
    noise_key: int | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "center_mm", _finite_vector(self.center_mm, "center_mm"))
        _positive(self.diam_mm, "diam_mm")
        _nonnegative(self.compliance_per_n, "compliance_per_n")
        _nonnegative(self.pulse_amplitude, "pulse_amplitude")
        _nonnegative(self.comp_ratio_std, "comp_ratio_std")
        _nonnegative(self.pulse_amp_std, "pulse_amp_std")
        if self.noise_key is not None:
            object.__setattr__(self, "noise_key", operator.index(self.noise_key))


@dataclass(frozen=True)
class SpecularReflector:
    """World-space circular reflector and independent point scatterer.

    Normal faces the insonifying aperture. Both scatterer and aperture must be
    on that side; the beam and broken ray must be inside the specular lobe.
    The optional noise_key has the same contract as for VesselTruth.
    """

    point_mm: np.ndarray
    normal: np.ndarray
    source_mm: np.ndarray
    radius_mm: float = 30.0
    half_angle_deg: float = 18.0
    diam_mm: float = 4.0
    noise_key: int | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "point_mm", _finite_vector(self.point_mm, "point_mm"))
        object.__setattr__(self, "source_mm", _finite_vector(self.source_mm, "source_mm"))
        normal = _finite_vector(self.normal, "normal")
        scale = np.linalg.norm(normal)
        _positive(scale, "normal length")
        object.__setattr__(self, "normal", normal / scale)
        _positive(self.radius_mm, "radius_mm")
        _positive(self.diam_mm, "diam_mm")
        if not 0 < self.half_angle_deg < 90:
            raise ValueError("half_angle_deg must be in (0, 90)")
        if self.noise_key is not None:
            object.__setattr__(self, "noise_key", operator.index(self.noise_key))


@dataclass(frozen=True)
class UltrasoundTruth:
    """Caller-selected in-station lumens and optional opaque triangle meshes.

    Each mesh is shaped (T, 3, 3): triangle, vertex, world XYZ. Missing lumens
    are omitted, not represented by NaN centers. Use one sample per local lumen;
    this renderer does not interpolate or intersect entire vessel centerlines.
    """

    vessels: Sequence[VesselTruth]
    bone_triangles_mm: np.ndarray | None = None
    fascia_triangles_mm: np.ndarray | None = None
    reflectors: Sequence[SpecularReflector] = ()


@dataclass(frozen=True)
class UltrasoundObservation:
    """Public N×5 array: lateral, depth, diameter, compression ratio, pulse.

    Rows are sorted solely on measured lateral position (exact ties can occur).
    Unobserved channels are NaN. No identities, merge flags or shadow flags are
    exposed; an observation can be empty and still has shape (0, 5).
    """

    echoes: np.ndarray

    @property
    def x_mm(self) -> np.ndarray:
        return self.echoes[:, 0]

    @property
    def z_mm(self) -> np.ndarray:
        return self.echoes[:, 1]

    @property
    def diam_mm(self) -> np.ndarray:
        return self.echoes[:, 2]

    @property
    def comp_ratio(self) -> np.ndarray:
        return self.echoes[:, 3]

    @property
    def pulse_amp(self) -> np.ndarray:
        return self.echoes[:, 4]


class _TriangleMesh:
    def __init__(self, triangles: np.ndarray):
        vertices = np.asarray(triangles, dtype=np.float64)
        if vertices.ndim != 3 or vertices.shape[1:] != (3, 3) or not np.all(np.isfinite(vertices)):
            raise ValueError("acoustic mesh must contain finite world triangles of shape (T, 3, 3)")
        self.v0 = vertices[:, 0]
        self.e1 = vertices[:, 1] - self.v0
        self.e2 = vertices[:, 2] - self.v0
        self.det_tol = 32 * np.finfo(float).eps * np.linalg.norm(self.e1, axis=1) * np.linalg.norm(self.e2, axis=1)

    def blocks(self, start: np.ndarray, end: np.ndarray) -> bool:
        """Double-sided signed Möller–Trumbore over the open segment.

        Parallel/coplanar and degenerate triangles do not return an isolated
        intersection. The endpoint tolerance avoids self-occluding reflectors.
        """
        ray = end - start
        length = np.linalg.norm(ray)
        endpoint_tol = 1e-7
        if length <= 2 * endpoint_tol:
            return False
        direction = ray / length
        p = np.cross(direction, self.e2)
        det = np.einsum("ij,ij->i", self.e1, p)
        indices = np.flatnonzero(np.abs(det) > self.det_tol)
        if not indices.size:
            return False
        relative = start - self.v0[indices]
        inverse_det = 1.0 / det[indices]
        u = np.einsum("ij,ij->i", relative, p[indices]) * inverse_det
        q = np.cross(relative, self.e1[indices])
        v = (q @ direction) * inverse_det
        distance = np.einsum("ij,ij->i", self.e2[indices], q) * inverse_det
        slack = 32 * np.finfo(float).eps
        return bool(np.any((u >= -slack) & (v >= -slack) & (u + v <= 1 + slack)
                           & (distance > endpoint_tol) & (distance < length - endpoint_tol)))


def _occluded(meshes: Sequence[_TriangleMesh], start: np.ndarray, end: np.ndarray) -> bool:
    return any(mesh.blocks(start, end) for mesh in meshes)


def _physical_key(prefix: bytes, *values: float) -> bytes:
    # Canonical little-endian doubles; signed zero has no physical meaning.
    return prefix + struct.pack(f"<{len(values)}d", *(value if value != 0 else 0.0 for value in values))


def _draw_key(physical_key: bytes, token: int | None) -> bytes:
    if token is None:
        return physical_key
    # Distinct from a fixed-length physical fingerprint and self-delimiting.
    return b"S" + physical_key[:1] + str(token).encode("ascii") + b"\0"


@dataclass(frozen=True)
class _Candidate:
    key: bytes
    draw_key: bytes
    xz: np.ndarray
    diam_mm: float
    comp_ratio: float
    pulse_amp: float
    comp_std: float
    pulse_std: float


def _merged_groups(candidates: list[_Candidate], distance_mm: float) -> list[list[_Candidate]]:
    """Connected components under finite two-dimensional acoustic resolution."""
    roots = list(range(len(candidates)))

    def find(index: int) -> int:
        while roots[index] != index:
            roots[index] = roots[roots[index]]
            index = roots[index]
        return index

    limit = distance_mm * distance_mm
    for j, candidate in enumerate(candidates):
        for i in range(j):
            separation = candidate.xz - candidates[i].xz
            if separation @ separation < limit:
                roots[find(j)] = find(i)
    groups: dict[int, list[_Candidate]] = {}
    for i, candidate in enumerate(candidates):
        groups.setdefault(find(i), []).append(candidate)
    return list(groups.values())


class PhysicsUltrasoundKernel:
    """Stateless sensing with common random numbers for matched echoes.

    A draw depends on tape seed, frame index, and geometric fingerprints or
    opt-in stable noise tokens, never on appearance, role, list index, mode,
    or view angle. Tokens preserve noise across moving counterfactuals if the
    echo has the same contributors. For example, with the default 2.5 N
    nominal force, compliance coefficients 0.24 and 0.04 /N generate mean
    ratios -0.48 and -0.08 at 4.5 N. The caller sets the independently
    measured/simulated lumen properties, not a task-role table.
    """

    def __init__(
        self, seed: int = 0, *, nominal_force_n: float = 2.5,
        lateral_half_width_mm: float = 25.0, slice_half_width_mm: float = 3.0,
        depth_min_mm: float = 3.0, depth_max_mm: float = 80.0,
        merge_distance_mm: float = 4.7, blur_exposure_s: float = 0.06,
        centroid_noise_mm: tuple[float, float] = (0.8, 1.0), diameter_noise_mm: float = 0.35,
    ) -> None:
        self.seed = operator.index(seed)
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        for name, value in (("nominal_force_n", nominal_force_n), ("depth_min_mm", depth_min_mm),
                            ("blur_exposure_s", blur_exposure_s), ("diameter_noise_mm", diameter_noise_mm)):
            _nonnegative(value, name)
        for name, value in (("lateral_half_width_mm", lateral_half_width_mm),
                            ("slice_half_width_mm", slice_half_width_mm), ("merge_distance_mm", merge_distance_mm),
                            ("depth_max_mm", depth_max_mm)):
            _positive(value, name)
        if depth_max_mm <= depth_min_mm:
            raise ValueError("depth_max_mm must exceed depth_min_mm")
        self.centroid_noise_mm = np.asarray(centroid_noise_mm, dtype=np.float64)
        if (self.centroid_noise_mm.shape != (2,) or not np.all(np.isfinite(self.centroid_noise_mm))
                or np.any(self.centroid_noise_mm < 0)):
            raise ValueError("centroid_noise_mm must contain nonnegative finite lateral and depth stds")
        self.nominal_force_n = nominal_force_n
        self.lateral_half_width_mm = lateral_half_width_mm
        self.slice_half_width_mm = slice_half_width_mm
        self.depth_min_mm = depth_min_mm
        self.depth_max_mm = depth_max_mm
        self.merge_distance_mm = merge_distance_mm
        self.blur_exposure_s = blur_exposure_s
        self.diameter_noise_mm = diameter_noise_mm

    def _in_frame(self, x: float, z: float) -> bool:
        return abs(x) <= self.lateral_half_width_mm and self.depth_min_mm <= z <= self.depth_max_mm

    def _project(self, point: np.ndarray, pose: ProbePose) -> np.ndarray | None:
        x, elevation, z = (point - pose.origin_mm) @ pose.rotation
        if abs(elevation) > self.slice_half_width_mm or not self._in_frame(x, z):
            return None
        return np.array([x, z])

    def _ghost(
        self, reflector: SpecularReflector, pose: ProbePose, meshes: Sequence[_TriangleMesh],
    ) -> _Candidate | None:
        normal = reflector.normal
        cosine_limit = math.cos(math.radians(reflector.half_angle_deg))
        if -pose.rotation[:, 2] @ normal < cosine_limit:
            return None
        aperture_height = (pose.origin_mm - reflector.point_mm) @ normal
        source_height = (reflector.source_mm - reflector.point_mm) @ normal
        if aperture_height <= 0 or source_height <= 0:
            return None
        image = reflector.source_mm - 2 * source_height * normal
        xz = self._project(image, pose)
        if xz is None:
            return None
        ray = image - pose.origin_mm
        if aperture_height + source_height < cosine_limit * np.linalg.norm(ray):
            return None
        hit = pose.origin_mm + ray * (aperture_height / (aperture_height + source_height))
        offset = hit - reflector.point_mm
        if offset @ offset > reflector.radius_mm ** 2:
            return None
        if _occluded(meshes, pose.origin_mm, hit) or _occluded(meshes, hit, reflector.source_mm):
            return None
        key = _physical_key(b"R", *reflector.point_mm, *normal, *reflector.source_mm,
                            reflector.radius_mm, reflector.half_angle_deg, reflector.diam_mm)
        geometry_key = _physical_key(b"R", *reflector.point_mm, *normal, *reflector.source_mm)
        return _Candidate(key, _draw_key(geometry_key, reflector.noise_key), xz,
                          reflector.diam_mm, 0.0, 0.0, 0.12, 0.02)

    def _measure(
        self, group: list[_Candidate], tape_key: bytes, caliber: bool, compression: bool, pulse: bool,
    ) -> tuple[float, float, float, float, float] | None:
        digest = hashlib.blake2b(tape_key, digest_size=16)
        for key in sorted(candidate.draw_key for candidate in group):
            digest.update(key)
        rng = np.random.default_rng(int.from_bytes(digest.digest(), "little"))
        noise = rng.standard_normal(5)
        # The equivalent diameter sums lumen areas. Moment-style compression
        # and pulse summaries do not resolve individual members or pulse phase.
        baseline_area = sum(c.diam_mm ** 2 for c in group)
        compressed_area = sum((c.diam_mm * (1 + c.comp_ratio)) ** 2 for c in group)
        weights = [(c.diam_mm * (1 + c.comp_ratio)) ** 2 / compressed_area for c in group]
        x = sum(w * c.xz[0] for w, c in zip(weights, group)) + self.centroid_noise_mm[0] * noise[0]
        z = sum(w * c.xz[1] for w, c in zip(weights, group)) + self.centroid_noise_mm[1] * noise[1]
        if not self._in_frame(x, z):
            return None
        diam = ratio = pulse_amp = np.nan
        if caliber:
            diam = max(0.0, math.sqrt(compressed_area) + self.diameter_noise_mm * noise[2])
        if compression:
            sigma = math.sqrt(sum((w * c.comp_std) ** 2 for w, c in zip(weights, group)))
            ratio = math.sqrt(compressed_area / baseline_area) - 1 + sigma * noise[3]
        if pulse:
            amplitude = sum(w * c.pulse_amp for w, c in zip(weights, group))
            sigma = math.sqrt(sum((w * c.pulse_std) ** 2 for w, c in zip(weights, group)))
            pulse_amp = max(0.0, amplitude + sigma * noise[4])
        return x, z, diam, ratio, pulse_amp

    def render(
        self, truth_world: UltrasoundTruth, probe_pose: ProbePose, f_contact: float,
        tape_step: int, mode: ObservationMode = "full_physical",
    ) -> UltrasoundObservation:
        """Observe one frame; `tape_step` is an integer index and force is in N.

        Dwell is uninterrupted time in the current acoustic window. The pulse
        channel is a noisy amplitude summary after dwell, not a waveform fit.
        """
        if mode not in ("centroid_only", "centroid_caliber", "full_physical"):
            raise ValueError(f"unknown observation mode: {mode!r}")
        _nonnegative(f_contact, "f_contact")
        step = operator.index(tape_step)
        if step < 0:
            raise ValueError("tape_step must be nonnegative")
        tape_key = f"{self.seed}:{step}:".encode("ascii")
        meshes = tuple(_TriangleMesh(mesh) for mesh in (truth_world.bone_triangles_mm,
                                                        truth_world.fascia_triangles_mm) if mesh is not None)
        excess_force = max(0.0, f_contact - self.nominal_force_n)
        candidates = []
        for vessel in truth_world.vessels:
            xz = self._project(vessel.center_mm, probe_pose)
            if xz is None or _occluded(meshes, probe_pose.origin_mm, vessel.center_mm):
                continue
            properties = (vessel.diam_mm, vessel.compliance_per_n, vessel.pulse_amplitude,
                          vessel.comp_ratio_std, vessel.pulse_amp_std)
            key = _physical_key(b"V", *vessel.center_mm, *properties)
            ratio = -min(0.95, excess_force * vessel.compliance_per_n)
            geometry_key = _physical_key(b"V", *vessel.center_mm)
            candidates.append(_Candidate(key, _draw_key(geometry_key, vessel.noise_key), xz,
                                         vessel.diam_mm, ratio, vessel.pulse_amplitude,
                                         vessel.comp_ratio_std, vessel.pulse_amp_std))
        for reflector in truth_world.reflectors:
            ghost = self._ghost(reflector, probe_pose, meshes)
            if ghost is not None:
                candidates.append(ghost)

        candidates.sort(key=lambda c: c.key)
        resolution = self.merge_distance_mm + probe_pose.speed_mm_s * self.blur_exposure_s
        caliber = mode != "centroid_only"
        compression = mode == "full_physical" and excess_force > 0
        pulse = mode == "full_physical" and probe_pose.dwell_s >= 0.35
        rows = []
        for group in _merged_groups(candidates, resolution):
            row = self._measure(group, tape_key, caliber, compression, pulse)
            if row is not None:
                rows.append(row)
        rows.sort(key=lambda row: row[0])
        echoes = np.asarray(rows, dtype=np.float64).reshape(-1, 5)
        echoes.setflags(write=False)
        return UltrasoundObservation(echoes)
