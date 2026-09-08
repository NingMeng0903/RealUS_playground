"""Contact-gated motion references for force-position tracking episodes."""

from __future__ import annotations

import inspect
import time
from typing import Callable

import numpy as np

from rm75_control.control.admittance_common.reference import (
    MotionReference,
    MotionReferenceSource,
)


class ContactGatedReference:
    """Hold a reference at its first sample until contact is confirmed.

    The gate owns only trajectory time.  The force controller remains the
    same object for the whole phase and continues to update its own contact
    tracker.  Once contact has been observed, the gate latches and never
    rewinds on a later contact loss.

    ``contact_present`` is intentionally bound after phase compilation: the
    compiled outer owns the controller that produces this signal.
    """

    def __init__(
        self,
        reference: MotionReferenceSource,
        contact_present: Callable[[], bool] | None = None,
        *,
        start_force_n: float | None = None,
        start_force_s: float = 0.1,
    ) -> None:
        self.reference = reference
        self.start_force_n = start_force_n
        self.start_force_s = float(start_force_s)
        if start_force_n is not None and (
            not np.isfinite(start_force_n) or start_force_n <= 0
            or not np.isfinite(start_force_s) or start_force_s < 0
        ):
            raise ValueError("invalid independent scan force gate")
        self._force_ready = start_force_n is None
        self._force_since = None
        self._force_last_time = None
        self._seen_air = False
        self._seek_wall_start = None
        self._seek_normal = None
        self._contact_present: Callable[[], bool] | None = None
        self._started = False
        self._elapsed_s = 0.0
        self._last_input_t_s: float | None = None
        self._first_sample: MotionReference | None = None
        if contact_present is not None:
            self.bind_contact_present(contact_present)

    @property
    def started(self) -> bool:
        """Whether the first confirmed contact has started the trajectory."""

        return bool(self._started)

    @property
    def elapsed_s(self) -> float:
        """Reference time accumulated from the post-contact input clock."""

        return float(self._elapsed_s)

    def bind_contact_present(self, contact_present: Callable[[], bool]) -> None:
        """Attach the compiled controller's contact signal."""

        if not callable(contact_present):
            raise TypeError("contact_present must be a callable returning bool")
        self._contact_present = contact_present

    def _clear(self) -> None:
        self._started = False
        self._elapsed_s = 0.0
        self._last_input_t_s = None
        self._first_sample = None
        self._force_ready = self.start_force_n is None
        self._force_since = self._force_last_time = None
        self._seen_air = False
        self._seek_wall_start = None
        self._seek_normal = None

    def guard_approach(self, pose) -> None:
        """Bound an ICRA seek even if the external acquisition process disappears."""
        if self.start_force_n is None or self._started:
            return
        now = time.monotonic()
        if self._seek_wall_start is None:
            self._seek_wall_start = now
        if now - self._seek_wall_start > 25.0:
            raise RuntimeError("ICRA contact seek timed out before sustained 4 N")
        if getattr(self.reference, "spec", {}).get("schema") == "icra_path_v1":
            anchor = self._anchor().pose_d
            if self._seek_normal is None:
                from scipy.spatial.transform import Rotation

                self._seek_normal = -Rotation.from_euler("xyz", anchor[3:]).as_matrix()[:, 2]
            if float((np.asarray(pose)[:3] - anchor[:3]) @ self._seek_normal) < -0.010:
                raise RuntimeError("ICRA contact seek exceeded taught surface by 10 mm")

    def observe_force(self, fz: float, sample_time_s: float, *, valid: bool = True) -> None:
        """Use fresh compensated samples, independently of physical contact tuning."""
        if self.start_force_n is None or self._started:
            return
        if not valid or not np.isfinite(fz) or not np.isfinite(sample_time_s):
            self._force_since = None
            self._force_ready = False
            return
        if self._force_last_time is not None:
            if sample_time_s <= self._force_last_time:
                return
            if sample_time_s - self._force_last_time > 0.1:
                self._force_since = None
                self._force_ready = False
        self._force_last_time = sample_time_s
        if fz < self.start_force_n:
            self._seen_air = True
            self._force_since = None
            self._force_ready = False
        elif self._seen_air:
            if self._force_since is None:
                self._force_since = sample_time_s
            self._force_ready = sample_time_s - self._force_since >= self.start_force_s - 1e-9

    @staticmethod
    def _call_set_origin(reference, pose0: np.ndarray) -> None:
        """Call a child origin hook, with a zero time anchor when supported."""

        fn = getattr(reference, "set_origin", None)
        if not callable(fn):
            return
        signature_known = True
        try:
            params = inspect.signature(fn).parameters.values()
            accepts_time = any(
                p.name == "t_s" or p.kind == inspect.Parameter.VAR_KEYWORD
                for p in params
            )
        except (TypeError, ValueError):
            signature_known = False
            accepts_time = False
        if signature_known:
            # A known signature gets exactly one call.  In particular, do
            # not catch a TypeError raised inside the child and invoke it a
            # second time with a different argument list.
            if accepts_time:
                fn(pose0, t_s=0.0)
            else:
                fn(pose0)
            return
        # Only opaque callables need the compatibility fallback.
        try:
            fn(pose0, t_s=0.0)
        except TypeError:
            fn(pose0)

    def set_origin(self, pose0: np.ndarray, *, t_s: float | None = None) -> None:
        """Reset the contact latch and re-anchor the child at time zero."""

        del t_s
        self._clear()
        self._call_set_origin(self.reference, np.asarray(pose0, dtype=float))

    def _anchor(self) -> MotionReference:
        if self._first_sample is None:
            sample = self.reference.sample(0.0)
            self._first_sample = MotionReference(
                pose_d=np.asarray(sample.pose_d, dtype=float).reshape(6).copy(),
                vel_ff=np.asarray(sample.vel_ff, dtype=float).reshape(6).copy(),
                t_ref=0.0,
                valid=bool(getattr(sample, "valid", True)),
            )
        return self._first_sample

    @staticmethod
    def _hold(sample: MotionReference) -> MotionReference:
        return MotionReference(
            pose_d=np.asarray(sample.pose_d, dtype=float).reshape(6).copy(),
            vel_ff=np.zeros(6, dtype=float),
            t_ref=0.0,
            valid=bool(getattr(sample, "valid", True)),
        )

    def sample(self, t_s: float) -> MotionReference:
        """Return the held first pose or the latched child trajectory sample."""

        if self._contact_present is None:
            raise RuntimeError(
                "ContactGatedReference requires bind_contact_present() "
                "before sampling"
            )
        anchor = self._anchor()
        try:
            t_input = float(t_s)
        except (TypeError, ValueError, OverflowError):
            t_input = float("nan")

        if not self._started:
            if not self._force_ready or not bool(self._contact_present()):
                # Do not pass the runner's absolute/governor time through to
                # an unstarted child: this is both the hold and the initial
                # time anchor for sources without set_origin().
                return self._hold(anchor)
            self._started = True
            self._elapsed_s = 0.0
            self._last_input_t_s = t_input if np.isfinite(t_input) else None
            return anchor

        if np.isfinite(t_input):
            if self._last_input_t_s is not None:
                delta = t_input - float(self._last_input_t_s)
                if np.isfinite(delta) and delta > 0.0:
                    self._elapsed_s += float(delta)
            # Preserve a monotonic input anchor if a caller briefly rewinds
            # its governor clock; a later tick must not create a time burst.
            if self._last_input_t_s is None or t_input > self._last_input_t_s:
                self._last_input_t_s = t_input
        return self.reference.sample(float(self._elapsed_s))


__all__ = ["ContactGatedReference"]
