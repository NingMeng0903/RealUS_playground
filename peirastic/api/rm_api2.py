"""RM_API2-shaped aliases. SI methods stay canonical; these convert units."""

from __future__ import annotations

from typing import Any

import numpy as np

from peirastic.api.codes import OK


def rm_speed_scale(v: float) -> float:
    """RM ``v`` is 1–100 percent. Values in (0, 1] stay peirastic fractions."""

    x = float(v)
    if x > 1.0:
        if not 1.0 <= x <= 100.0:
            raise ValueError(f"RM v must be 1–100 or (0, 1], got {v}")
        return x / 100.0
    if not 0.0 < x <= 1.0:
        raise ValueError(f"RM v must be 1–100 or (0, 1], got {v}")
    return x


def rm_joint_to_si(joint, *, rail_m: float | None = None) -> list[float]:
    """``[rail_mm, j1..j7 °]`` or 7 arm degrees (rail 400 mm)."""

    j = np.asarray(joint, dtype=float).reshape(-1)
    if j.size == 8:
        return [float(j[0]) * 0.001, *np.deg2rad(j[1:]).tolist()]
    if j.size == 7:
        rail = 0.4 if rail_m is None else float(rail_m)
        if not np.isfinite(rail):
            raise ValueError("7-arm RM joint target needs a finite live rail position")
        return [rail, *np.deg2rad(j).tolist()]
    raise ValueError(f"RM joint must be 7 or 8 numbers, got {j.size}")


def si_joint_to_rm(q) -> list[float]:
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.size != 8:
        raise ValueError(f"SI q must be 8-vec, got {q.size}")
    return [float(q[0]) * 1000.0, *np.rad2deg(q[1:]).tolist()]


class _RmApi2Mixin:
    """``rm_*`` names and percent/degree units. Internals stay SI."""

    _movev_session: bool

    def rm_movej(self, joint, v: int | float, r: int = 0, connect: int = 0, block: int = 1) -> int:
        arr = np.asarray(joint, dtype=float).reshape(-1)
        rail = None
        if arr.size == 7 and hasattr(self, "_current_rail_m"):
            rail = self._current_rail_m()
        return self.movej(
            rm_joint_to_si(joint, rail_m=rail),
            v=rm_speed_scale(v),
            r=r,
            connect=connect,
            block=block,
        )

    def rm_movej_p(self, pose, v: int | float, r: int = 0, connect: int = 0, block: int = 1) -> int:
        return self.movej_p(pose, v=rm_speed_scale(v), r=r, connect=connect, block=block)

    def rm_movel(self, pose, v: int | float, r: int = 0, connect: int = 0, block: int = 1) -> int:
        """Pose-to-pose PTP (our ``cartesian``), not a vendor TCP line."""

        return self.cartesian(pose, v=rm_speed_scale(v), r=r, connect=connect, block=block)

    def rm_moves(self, pose, v: int | float, r: int = 0, connect: int = 0, block: int = 1) -> int:
        return self.moves(pose, v=rm_speed_scale(v), r=r, connect=connect, block=block)

    def rm_movep_canfd(self, pose, follow: bool = False, trajectory_mode: int = 0, radio: int = 0) -> int:
        del follow, trajectory_mode, radio
        return self.movep_canfd(pose, follow=False)

    def rm_set_movev_canfd_init(
        self,
        avoid_singularity_flag: int = 1,
        frame_type: int = 0,
        dt: int = 5,
        follow: bool = False,
    ) -> int:
        del avoid_singularity_flag
        frame = "world" if int(frame_type) == 1 else "tool"
        self.set_movev_canfd_init(frame_type=frame, dt_ms=float(dt))
        self._movev_follow = bool(follow)
        return self.cartesian_velocity(
            duration_s=None, block=0, label="cartesian_velocity", follow=bool(follow)
        )

    def rm_movev_canfd(
        self,
        cartesian_velocity,
        follow: bool = False,
        trajectory_mode: int = 0,
        radio: int = 0,
    ) -> int:
        del trajectory_mode, radio
        want = bool(follow)
        if getattr(self, "twist", None) is None:
            return self.cartesian_velocity(
                cartesian_velocity, block=0, label="movev_canfd", follow=want
            )
        if not getattr(self, "_movev_session", False) or want != bool(
            getattr(self, "_movev_follow", want)
        ):
            ret = self.cartesian_velocity(None, block=0, label="movev_canfd", follow=want)
            if ret != OK:
                return ret
            self._movev_follow = want
        return self.set_cartesian_velocity(cartesian_velocity)

    def rm_set_force_control(
        self,
        *,
        force_axes=None,
        track_axes=None,
        desired_force=None,
        **kwargs: Any,
    ) -> int:
        """Axis force mask and targets. Only hybrid modes consume these."""

        return self.set_force_control(
            force_axes=force_axes,
            track_axes=track_axes,
            desired_force=desired_force,
            **kwargs,
        )

    def rm_set_arm_stop(self) -> int:
        return self.set_arm_stop()

    def rm_get_joint(self) -> tuple[int, list[float]]:
        ret, q = self.get_joint_radian()
        if ret != OK:
            return ret, []
        return OK, si_joint_to_rm(q)
