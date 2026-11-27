"""Pure static-observation gate for a stateful SMPL-X tracker.

The gate keeps an immutable copy of the last observation which the tracker
accepted as a new anchor.  ``matches`` never advances that anchor: callers can
therefore hold the last fitted pose while repeated RGB frames arrive.  The
gate uses observations only; it has no GT, filtering, smoothing, or model
dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


POINT_COUNT = 67
BODY25_COUNT = 25
CONFIDENCE_TOLERANCE = 0.05
ROTATION_JOINT_COUNT = 55
IGNORED_RGB_JOINTS = frozenset((0, 22, 23, 24))  # pelvis, jaw, and SMPL-X eyes


def _readonly_copy(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class StationaryObservationSnapshot:
    """Serializable-in-memory gate state with read-only array fields."""

    points: np.ndarray
    rgb_rotations: np.ndarray | None
    rgb_agreement: np.ndarray | None
    rgb_pelvis_world_m: np.ndarray | None


def _validate_tolerance(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _validate_points(points: np.ndarray, min_confidence: float) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(points, dtype=np.float64)
    if value.shape != (POINT_COUNT, 4):
        raise ValueError(f"points67 must have shape (67, 4), got {value.shape}")
    valid = np.isfinite(value).all(axis=1) & (value[:, 3] > min_confidence)
    if int(valid[:BODY25_COUNT].sum()) < 12:
        # A transiently weak frame must not become a stationary anchor.  It is
        # a normal gate miss for matches/commit, handled by the caller.
        return np.array(value, copy=True), valid
    return np.array(value, copy=True), valid


def _validate_rotations(value: np.ndarray) -> np.ndarray:
    rotations = np.asarray(value, dtype=np.float64)
    if rotations.shape != (ROTATION_JOINT_COUNT, 3, 3):
        raise ValueError(f"rgb_rotations must have shape (55, 3, 3), got {rotations.shape}")
    if not np.isfinite(rotations).all():
        raise ValueError("rgb_rotations must be finite")
    identity = np.eye(3, dtype=np.float64)
    orthogonal = np.matmul(rotations.transpose(0, 2, 1), rotations)
    determinants = np.linalg.det(rotations)
    if (not np.allclose(orthogonal, identity, atol=1e-3)
            or not np.allclose(determinants, 1.0, atol=1e-3)):
        raise ValueError("rgb_rotations must contain proper SO(3) matrices")
    return np.array(rotations, copy=True)


def _validate_agreement(value: np.ndarray) -> np.ndarray:
    agreement = np.asarray(value, dtype=np.float64)
    if agreement.shape != (ROTATION_JOINT_COUNT,):
        raise ValueError(f"rgb_agreement must have shape (55,), got {agreement.shape}")
    if not np.isfinite(agreement).all() or np.any((agreement < 0.0) | (agreement > 1.0)):
        raise ValueError("rgb_agreement must be finite and in [0, 1]")
    return np.array(agreement, copy=True)


def _validate_pelvis(value: np.ndarray) -> np.ndarray:
    pelvis = np.asarray(value, dtype=np.float64)
    if pelvis.shape != (3,):
        raise ValueError(f"rgb_pelvis_world_m must have shape (3,), got {pelvis.shape}")
    if not np.isfinite(pelvis).all():
        raise ValueError("rgb_pelvis_world_m must be finite")
    return np.array(pelvis, copy=True)


class StationaryObservationGate:
    """Compare accepted 67-point observations against an immutable anchor.

    A call to :meth:`commit` replaces the anchor because the caller has
    accepted a new tracking result.  A call to :meth:`matches` is pure and
    never changes it.  Before the first commit, ``matches`` returns ``False``.
    """

    def __init__(self, position_tolerance_m: float = 0.002,
                 rotation_tolerance_deg: float = 2.0,
                 min_confidence: float = 0.05) -> None:
        self.position_tolerance_m = _validate_tolerance("position_tolerance_m", position_tolerance_m)
        self.rotation_tolerance_deg = _validate_tolerance("rotation_tolerance_deg", rotation_tolerance_deg)
        self.min_confidence = _validate_tolerance("min_confidence", min_confidence)
        self._anchor: StationaryObservationSnapshot | None = None

    @property
    def has_anchor(self) -> bool:
        return self._anchor is not None

    def _prepare(self, points67: np.ndarray, rgb_rotations: np.ndarray | None,
                 rgb_agreement: np.ndarray | None,
                 rgb_pelvis_world_m: np.ndarray | None) -> tuple[StationaryObservationSnapshot, np.ndarray]:
        points, valid = _validate_points(points67, self.min_confidence)
        rotations = _validate_rotations(rgb_rotations) if rgb_rotations is not None else None
        agreement = _validate_agreement(rgb_agreement) if rgb_agreement is not None else None
        pelvis = _validate_pelvis(rgb_pelvis_world_m) if rgb_pelvis_world_m is not None else None
        snapshot = StationaryObservationSnapshot(
            points=_readonly_copy(points),
            rgb_rotations=_readonly_copy(rotations),
            rgb_agreement=_readonly_copy(agreement),
            rgb_pelvis_world_m=_readonly_copy(pelvis),
        )
        return snapshot, valid

    def _rgb_matches(self, current: StationaryObservationSnapshot,
                     anchor: StationaryObservationSnapshot) -> bool:
        if (current.rgb_rotations is None) != (anchor.rgb_rotations is None):
            return False
        if (current.rgb_agreement is None) != (anchor.rgb_agreement is None):
            return False
        if current.rgb_rotations is not None:
            # A provided rotation tensor is required to retain its presence;
            # angles are gated only where both agreement vectors mark a local
            # pose as positive.  Root and fixed face joints are deliberately
            # excluded even if a caller supplies nonzero agreement there.
            if current.rgb_agreement is not None:
                active = np.ones(ROTATION_JOINT_COUNT, dtype=bool)
                active[list(IGNORED_RGB_JOINTS)] = False
                current_mask = (current.rgb_agreement > 0.0) & active
                anchor_mask = (anchor.rgb_agreement > 0.0) & active
                if not np.array_equal(current_mask, anchor_mask):
                    return False
                if np.any(np.abs(current.rgb_agreement[active] - anchor.rgb_agreement[active]) > CONFIDENCE_TOLERANCE):
                    return False
                if np.any(current_mask):
                    relative = np.matmul(
                        anchor.rgb_rotations[current_mask].transpose(0, 2, 1),
                        current.rgb_rotations[current_mask],
                    )
                    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
                    angles = np.arccos(cosine)
                    if np.any(angles > np.deg2rad(self.rotation_tolerance_deg)):
                        return False
        elif current.rgb_agreement is not None:
            active = np.ones(ROTATION_JOINT_COUNT, dtype=bool)
            active[list(IGNORED_RGB_JOINTS)] = False
            current_mask = (current.rgb_agreement > 0.0) & active
            anchor_mask = (anchor.rgb_agreement > 0.0) & active
            if not np.array_equal(current_mask, anchor_mask):
                return False
            if np.any(np.abs(current.rgb_agreement[active] - anchor.rgb_agreement[active]) > CONFIDENCE_TOLERANCE):
                return False
        return True

    def matches(self, points67: np.ndarray, rgb_rotations: np.ndarray | None = None,
                rgb_agreement: np.ndarray | None = None,
                rgb_pelvis_world_m: np.ndarray | None = None) -> bool:
        """Return whether the current observation can reuse the held result."""
        current, valid = self._prepare(points67, rgb_rotations, rgb_agreement, rgb_pelvis_world_m)
        anchor = self._anchor
        if anchor is None or int(valid[:BODY25_COUNT].sum()) < 12:
            return False
        anchor_valid = np.isfinite(anchor.points).all(axis=1) & (anchor.points[:, 3] > self.min_confidence)
        if not np.array_equal(valid, anchor_valid):
            return False
        if np.any(np.linalg.norm(current.points[valid, :3] - anchor.points[valid, :3], axis=1) > self.position_tolerance_m):
            return False
        if np.any(np.abs(current.points[valid, 3] - anchor.points[valid, 3]) > CONFIDENCE_TOLERANCE):
            return False
        if not self._rgb_matches(current, anchor):
            return False
        if (current.rgb_pelvis_world_m is None) != (anchor.rgb_pelvis_world_m is None):
            return False
        if current.rgb_pelvis_world_m is not None and float(np.linalg.norm(
                current.rgb_pelvis_world_m - anchor.rgb_pelvis_world_m)) > self.position_tolerance_m:
            return False
        return True

    def commit(self, points67: np.ndarray, rgb_rotations: np.ndarray | None = None,
               rgb_agreement: np.ndarray | None = None,
               rgb_pelvis_world_m: np.ndarray | None = None) -> bool:
        """Accept and copy a new anchor; return ``False`` for weak Body25 input."""
        snapshot, valid = self._prepare(points67, rgb_rotations, rgb_agreement, rgb_pelvis_world_m)
        if int(valid[:BODY25_COUNT].sum()) < 12:
            return False
        self._anchor = snapshot
        return True

    def snapshot(self) -> StationaryObservationSnapshot | None:
        """Return an independent read-only copy of the current anchor."""
        if self._anchor is None:
            return None
        anchor = self._anchor
        return StationaryObservationSnapshot(
            points=_readonly_copy(anchor.points),
            rgb_rotations=_readonly_copy(anchor.rgb_rotations),
            rgb_agreement=_readonly_copy(anchor.rgb_agreement),
            rgb_pelvis_world_m=_readonly_copy(anchor.rgb_pelvis_world_m),
        )

    def restore(self, snapshot: StationaryObservationSnapshot | None) -> None:
        """Restore a snapshot without retaining references to its arrays."""
        if snapshot is None:
            self._anchor = None
            return
        points, valid = _validate_points(snapshot.points, self.min_confidence)
        if int(valid[:BODY25_COUNT].sum()) < 12:
            raise ValueError("snapshot has fewer than 12 valid Body25 points")
        rotations = _validate_rotations(snapshot.rgb_rotations) if snapshot.rgb_rotations is not None else None
        agreement = _validate_agreement(snapshot.rgb_agreement) if snapshot.rgb_agreement is not None else None
        pelvis = _validate_pelvis(snapshot.rgb_pelvis_world_m) if snapshot.rgb_pelvis_world_m is not None else None
        self._anchor = StationaryObservationSnapshot(
            points=_readonly_copy(points),
            rgb_rotations=_readonly_copy(rotations),
            rgb_agreement=_readonly_copy(agreement),
            rgb_pelvis_world_m=_readonly_copy(pelvis),
        )


__all__ = ["StationaryObservationGate", "StationaryObservationSnapshot"]
