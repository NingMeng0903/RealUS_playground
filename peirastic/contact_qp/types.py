"""Versioned SI interfaces. Control compression and physical wrench are distinct."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math

import numpy as np

SCHEMA_VERSION = 1
WINDOW_NAMES = ("left", "center", "right")
REQUIRED_WINDOWS = (0, 2)


def vector(value, shape, *, name="array", finite=True):
    a = np.array(value, dtype=float, copy=True)
    if a.shape != shape or (finite and not np.isfinite(a).all()):
        raise ValueError(f"{name} must have shape {shape} and valid values")
    a.setflags(write=False)
    return a


def positive(value, name, *, zero=False):
    x = float(value)
    if not math.isfinite(x) or (x < 0 if zero else x <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return x


@dataclass(frozen=True)
class ProbeGeometry:
    """T_tcp_face maps FACE points to the existing TCP; no pivot redefinition."""
    half_length_m: float
    T_tcp_face: np.ndarray = field(default_factory=lambda: np.eye(4))
    image_x_sign: int = 1
    calibration_version: str = "unverified"
    verified: bool = False

    def __post_init__(self):
        object.__setattr__(self, "half_length_m", positive(self.half_length_m, "half_length_m"))
        t = vector(self.T_tcp_face, (4, 4), name="T_tcp_face")
        if (not np.allclose(t[3], [0, 0, 0, 1], atol=1e-10)
                or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-9)
                or not np.isclose(np.linalg.det(t[:3, :3]), 1.0, atol=1e-9)):
            raise ValueError("T_tcp_face must be a proper rigid transform")
        if self.image_x_sign not in (-1, 1) or not self.calibration_version:
            raise ValueError("explicit image-x sign and calibration version required")
        object.__setattr__(self, "T_tcp_face", t)

    @classmethod
    def synthetic(cls, half_length_m=0.025, **kwargs):
        return cls(half_length_m, calibration_version="synthetic_v1", verified=False, **kwargs)


@dataclass(frozen=True)
class ContactObservation:
    frame_seq: int
    source_id: str
    effective_time_s: float
    received_time_s: float
    quality: np.ndarray
    valid: np.ndarray
    registration_version: str
    window_version: str
    schema_version: int = SCHEMA_VERSION
    calibration_version: str = "unverified"

    def __post_init__(self):
        if self.schema_version != SCHEMA_VERSION or self.frame_seq < 0:
            raise ValueError("unsupported observation schema/sequence")
        if not self.source_id or not self.registration_version or not self.window_version or not self.calibration_version:
            raise ValueError("observation source and configuration versions are required")
        for name in ("effective_time_s", "received_time_s"):
            object.__setattr__(self, name, positive(getattr(self, name), name, zero=True))
        if self.effective_time_s > self.received_time_s + 1e-6:
            raise ValueError("effective image time is in the future or in another clock domain")
        q = vector(self.quality, (3,), name="quality")
        if np.any((q < 0) | (q > 1)):
            raise ValueError("quality must be in [0,1]")
        valid = np.array(self.valid, dtype=bool, copy=True)
        if valid.shape != (3,):
            raise ValueError("valid must contain left/center/right flags")
        valid.setflags(write=False)
        object.__setattr__(self, "quality", q)
        object.__setattr__(self, "valid", valid)

    @property
    def version(self):
        return self.registration_version, self.window_version, self.calibration_version

    def fresh(self, now_s, max_age_s):
        age = float(now_s) - self.effective_time_s
        return bool(now_s >= self.received_time_s and 0 <= age <= max_age_s
                    and self.valid[list(REQUIRED_WINDOWS)].all())

    def to_dict(self):
        return {"schema_version": self.schema_version, "frame_seq": self.frame_seq,
                "source_id": self.source_id, "effective_time_s": self.effective_time_s,
                "received_time_s": self.received_time_s, "quality": self.quality.tolist(),
                "valid": self.valid.tolist(), "registration_version": self.registration_version,
                "window_version": self.window_version, "calibration_version": self.calibration_version}

    @classmethod
    def from_dict(cls, raw):
        return cls(**{k: raw[k] for k in cls.__dataclass_fields__ if k in raw})


@dataclass(frozen=True)
class TwistConstraints:
    """Generic lower <= A V <= upper. Frame is part of the contract."""
    A: np.ndarray = field(default_factory=lambda: np.empty((0, 6)))
    lower: np.ndarray = field(default_factory=lambda: np.empty(0))
    upper: np.ndarray = field(default_factory=lambda: np.empty(0))
    frame: str = "tcp_tool"
    valid_until_s: float = math.inf
    sequence: int = 0
    stop_epoch: int = 0
    labels: tuple[str, ...] = ()

    def __post_init__(self):
        raw = np.asarray(self.A)
        if raw.ndim != 2 or raw.shape[1] != 6:
            raise ValueError("A must have six columns")
        a = vector(raw, raw.shape, name="A")
        lo = vector(self.lower, (len(a),), name="lower", finite=False)
        hi = vector(self.upper, (len(a),), name="upper", finite=False)
        if np.isnan(lo).any() or np.isnan(hi).any() or np.any(lo > hi):
            raise ValueError("invalid inequality bounds")
        if np.isposinf(lo).any() or np.isneginf(hi).any():
            raise ValueError("infinite bound has wrong sign")
        if self.frame not in ("tcp_tool", "tcp_base") or math.isnan(self.valid_until_s):
            raise ValueError("invalid constraint frame/expiry")
        if self.sequence < 0 or self.stop_epoch < 0:
            raise ValueError("negative command sequence/epoch")
        if self.labels and len(self.labels) != len(a):
            raise ValueError("constraint label count mismatch")
        object.__setattr__(self, "A", a)
        object.__setattr__(self, "lower", lo)
        object.__setattr__(self, "upper", hi)
        object.__setattr__(self, "labels", tuple(self.labels))

    def violation(self, twist):
        v = vector(twist, (6,), name="twist")
        if not len(self.A):
            return 0.0
        with np.errstate(over="ignore", invalid="ignore"):
            value = self.A @ v
        if not np.isfinite(value).all():
            return math.inf
        return float(max(0.0, np.max(self.lower-value), np.max(value-self.upper)))


class ContactStatus(str, Enum):
    NOMINAL = "nominal"
    REPAIR = "repair"
    IMAGE_UNAVAILABLE = "image_unavailable"
    TASK_INFEASIBLE = "task_infeasible"
    MECHANICAL_INFEASIBLE = "mechanical_infeasible"
    SOLVER_FAILED = "solver_failed"
    DEFERRED = "deferred"
    CERTIFICATE_INVALID = "certificate_invalid"
