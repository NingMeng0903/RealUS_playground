"""Contact-gated motion references for force-position tracking episodes."""

from __future__ import annotations

import inspect
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
    ) -> None:
        self.reference = reference
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
            if not bool(self._contact_present()):
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
