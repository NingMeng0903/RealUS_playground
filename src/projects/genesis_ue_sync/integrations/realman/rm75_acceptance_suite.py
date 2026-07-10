"""Automated RM75 Genesis acceptance checks for force sensor + hybrid control modes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from projects.genesis_ue_sync.integrations.realman.hybrid_presets import HYBRID_PRESETS, HybridPreset
from projects.genesis_ue_sync.integrations.realman.hybrid_streaming_teleop import (
    HybridZPhase,
    streaming_hybrid_move_param,
    update_hybrid_z_phase,
)
from projects.genesis_ue_sync.integrations.realman.sim_robot_interface import rm_force_position_move_t
from projects.genesis_ue_sync.sim_platform.control.controllers.base import CartesianControlTarget
from projects.genesis_ue_sync.sim_platform.control.controllers.common import apply_pose_delta_wxyz
from projects.genesis_ue_sync.sim_platform.control.controllers.force_position_hybrid import (
    ForcePositionHybridParams,
)
from projects.genesis_ue_sync.sim_platform.control.controllers.realman_control_modes import (
    RM_CTRL_ADAPTIVE,
    RM_CTRL_FLOAT,
    RM_CTRL_FORCE_MOTION,
    RM_CTRL_FORCE_TRACK,
    RM_CTRL_MOTION,
    RM_CTRL_SPRING,
)
from projects.genesis_ue_sync.sim_platform.control.teleop.gamepad_cartesian import teleop_hybrid_limit_vel


@dataclass
class AcceptanceCaseResult:
    name: str
    passed: bool
    message: str
    metrics: dict[str, float | int | str | bool] = field(default_factory=dict)


@dataclass
class AcceptanceReport:
    results: list[AcceptanceCaseResult]

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    def print_summary(self) -> None:
        print("\n=== RM75 Acceptance Report (strict) ===", flush=True)
        for item in self.results:
            tag = "PASS" if item.passed else "FAIL"
            metric_bits = " ".join(f"{k}={v}" for k, v in item.metrics.items())
            suffix = f"  {metric_bits}" if metric_bits else ""
            print(f"[{tag}] {item.name}: {item.message}{suffix}", flush=True)
        print(
            f"Summary: {self.passed_count}/{len(self.results)} passed",
            flush=True,
        )


@dataclass
class RM75AcceptanceConfig:
    contact_threshold_n: float = 0.12
    min_contact_fz_n: float = 0.05
    desired_fz: float = 5.0
    settle_steps: int = 80
    cartesian_steps: int = 120
    xy_delta_m: float = 0.03
    approach_step_m: float = 0.00035
    max_approach_steps: int = 220
    hybrid_hold_steps: int = 90
    trans_scale: float = 0.25
    rot_scale: float = 0.35
    quiet_force_n: float = 2.0
    max_tcp_step_m: float = 0.055
    max_tcp_step_cartesian_slack: float = 1.05
    max_tcp_step_hybrid_m: float = 0.036
    max_tcp_step_rad: float = 0.38
    show_viewer: bool = False
    step_pause_s: float = 0.0
    fast: bool = False

    def scaled(self) -> RM75AcceptanceConfig:
        if not self.fast:
            return self
        return RM75AcceptanceConfig(
            contact_threshold_n=self.contact_threshold_n,
            min_contact_fz_n=self.min_contact_fz_n,
            desired_fz=self.desired_fz,
            settle_steps=max(30, self.settle_steps // 2),
            cartesian_steps=max(60, self.cartesian_steps // 2),
            xy_delta_m=self.xy_delta_m,
            approach_step_m=self.approach_step_m,
            max_approach_steps=max(120, self.max_approach_steps // 2),
            hybrid_hold_steps=max(45, self.hybrid_hold_steps // 2),
            trans_scale=self.trans_scale,
            rot_scale=self.rot_scale,
            quiet_force_n=self.quiet_force_n,
            max_tcp_step_m=self.max_tcp_step_m,
            max_tcp_step_cartesian_slack=self.max_tcp_step_cartesian_slack,
            max_tcp_step_hybrid_m=self.max_tcp_step_hybrid_m,
            max_tcp_step_rad=self.max_tcp_step_rad,
            show_viewer=self.show_viewer,
            step_pause_s=self.step_pause_s,
            fast=True,
        )


@dataclass
class RM75AcceptanceContext:
    runtime: Any
    bot: Any
    motion: Any
    cart: Any
    home_q: np.ndarray
    config: RM75AcceptanceConfig = field(default_factory=RM75AcceptanceConfig)


def _tcp_pose(ctx: RM75AcceptanceContext) -> np.ndarray:
    return np.asarray(ctx.motion.get_tcp_pose(), dtype=np.float32).reshape(7)


def _sensor_zero_force(ctx: RM75AcceptanceContext) -> np.ndarray:
    _, payload = ctx.bot.rm_get_force_data()
    return np.asarray(payload.get("zero_force_data") or payload.get("force_data") or [0.0] * 6, dtype=np.float32)


def _runtime_step(ctx: RM75AcceptanceContext) -> None:
    ctx.runtime.step()
    pause = float(ctx.config.step_pause_s)
    if pause > 0.0:
        import time

        time.sleep(pause)


def _settle(ctx: RM75AcceptanceContext, steps: int | None = None) -> None:
    n = int(ctx.config.settle_steps if steps is None else steps)
    for _ in range(max(n, 1)):
        _runtime_step(ctx)


def _stop_all(ctx: RM75AcceptanceContext) -> None:
    ctx.bot.stop_all()
    ctx.cart.reset()


def _go_home(ctx: RM75AcceptanceContext) -> None:
    _stop_all(ctx)
    ctx.bot.move_joints(np.asarray(ctx.home_q, dtype=np.float32).reshape(-1).tolist())
    _settle(ctx, steps=int(ctx.config.settle_steps) * 2)


def _contact_threshold(ctx: RM75AcceptanceContext) -> float:
    return float(ctx.config.contact_threshold_n)


def _min_contact_fz(ctx: RM75AcceptanceContext) -> float:
    return float(ctx.config.min_contact_fz_n)


def _limit_vel(ctx: RM75AcceptanceContext) -> list[float]:
    return teleop_hybrid_limit_vel(ctx.config.trans_scale, ctx.config.rot_scale)


def _assert_tcp_step_stable(
    ctx: RM75AcceptanceContext,
    prev: np.ndarray,
    *,
    label: str,
    max_lin_m: float | None = None,
) -> np.ndarray:
    cur = _tcp_pose(ctx)
    base_lin = float(ctx.config.max_tcp_step_m if max_lin_m is None else max_lin_m)
    slack = 1.0 if max_lin_m is not None else float(ctx.config.max_tcp_step_cartesian_slack)
    max_lin = base_lin * slack
    dp = float(np.linalg.norm(cur[:3] - prev[:3]))
    if dp > max_lin:
        raise AssertionError(f"{label}: TCP jumped {dp:.4f} m in one step (max {max_lin:.4f})")
    q_prev = prev[3:7] / max(float(np.linalg.norm(prev[3:7])), 1e-8)
    q_cur = cur[3:7] / max(float(np.linalg.norm(cur[3:7])), 1e-8)
    dot = float(np.clip(abs(float(np.dot(q_prev, q_cur))), -1.0, 1.0))
    dang = 2.0 * float(np.arccos(dot))
    if dang > float(ctx.config.max_tcp_step_rad):
        raise AssertionError(f"{label}: TCP rotated {dang:.3f} rad in one step (max {ctx.config.max_tcp_step_rad:.3f})")
    return cur


def _manual_hybrid_param(
    ctx: RM75AcceptanceContext,
    target_pose: np.ndarray,
    preset: HybridPreset,
) -> rm_force_position_move_t:
    param = ctx.bot.default_force_position_move_param(pose=target_pose)
    param.control_mode = [RM_CTRL_MOTION, RM_CTRL_MOTION, int(preset.z_mode), 0, 0, 0]
    fz = float(
        ctx.config.desired_fz
        if int(preset.z_mode) in {RM_CTRL_FORCE_TRACK, RM_CTRL_ADAPTIVE}
        else float(preset.desired_fz)
    )
    param.desired_force = [0.0, 0.0, fz, 0.0, 0.0, 0.0]
    param.limit_vel = _limit_vel(ctx)
    return param


def _step_cartesian_target(
    ctx: RM75AcceptanceContext,
    target_pose: np.ndarray,
    *,
    delta_xyz: tuple[float, float, float] | None = None,
) -> np.ndarray:
    prev = _tcp_pose(ctx)
    next_pose = np.asarray(target_pose, dtype=np.float32).reshape(7).copy()
    if delta_xyz is not None:
        delta = np.array([*delta_xyz, 0.0, 0.0, 0.0], dtype=np.float32)
        next_pose = apply_pose_delta_wxyz(next_pose, delta)
    ctx.cart.step(
        CartesianControlTarget(
            pose=next_pose,
            nullspace_target=np.asarray(ctx.home_q, dtype=np.float32).reshape(-1),
        )
    )
    _runtime_step(ctx)
    _assert_tcp_step_stable(ctx, prev, label="cartesian")
    return next_pose


def _step_cartesian_to(ctx: RM75AcceptanceContext, target_pose: np.ndarray) -> None:
    prev = _tcp_pose(ctx)
    ctx.cart.step(
        CartesianControlTarget(
            pose=np.asarray(target_pose, dtype=np.float32).reshape(7),
            nullspace_target=np.asarray(ctx.home_q, dtype=np.float32).reshape(-1),
        )
    )
    _runtime_step(ctx)
    _assert_tcp_step_stable(ctx, prev, label="cartesian_to")


def _approach_bed_xy(ctx: RM75AcceptanceContext, *, bed_xy: tuple[float, float] = (0.05, 0.0), steps: int = 70) -> np.ndarray:
    start = _tcp_pose(ctx)
    target = start.copy()
    goal = np.array([float(bed_xy[0]), float(bed_xy[1]), float(start[2])], dtype=np.float32)
    for i in range(max(int(steps), 1)):
        alpha = float(i + 1) / float(max(int(steps), 1))
        target = start.copy()
        target[:3] = (1.0 - alpha) * start[:3] + alpha * goal
        _step_cartesian_to(ctx, target)
    return np.asarray(target, dtype=np.float32).reshape(7)


def _make_contact(
    ctx: RM75AcceptanceContext,
    *,
    start_pose: np.ndarray | None = None,
) -> tuple[bool, float, np.ndarray, dict[str, float]]:
    if start_pose is None:
        target = _approach_bed_xy(ctx)
    else:
        target = np.asarray(start_pose, dtype=np.float32).reshape(7).copy()
    z_start = float(_tcp_pose(ctx)[2])
    max_fz = 0.0
    z_min = z_start
    metrics: dict[str, float] = {"z_start": z_start}
    for _ in range(int(ctx.config.max_approach_steps)):
        target = _step_cartesian_target(ctx, target, delta_xyz=(0.0, 0.0, -float(ctx.config.approach_step_m)))
        fz = abs(float(_sensor_zero_force(ctx)[2]))
        max_fz = max(max_fz, fz)
        z_min = min(z_min, float(_tcp_pose(ctx)[2]))
    metrics["max_fz"] = max_fz
    metrics["z_drop"] = z_start - z_min
    metrics["z_min"] = z_min
    force_ok = max_fz >= _min_contact_fz(ctx)
    geom_ok = (z_start - z_min) >= 0.02
    ok = bool(force_ok and geom_ok)
    return ok, max_fz, target, metrics


def _start_hybrid_streaming(ctx: RM75AcceptanceContext) -> None:
    if ctx.bot.rm_start_force_position_move() != 0:
        raise RuntimeError("rm_start_force_position_move failed")


def _hybrid_move_one(
    ctx: RM75AcceptanceContext,
    preset: HybridPreset,
    target: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    prev = _tcp_pose(ctx)
    param = _manual_hybrid_param(ctx, target, preset)
    if ctx.bot.rm_force_position_move(param) != 0:
        raise RuntimeError(f"rm_force_position_move failed for {preset.label}")
    return _assert_tcp_step_stable(ctx, prev, label=label, max_lin_m=float(ctx.config.max_tcp_step_hybrid_m))


def _run_hybrid_script(
    ctx: RM75AcceptanceContext,
    preset: HybridPreset,
    *,
    descend: bool = True,
    hold_steps: int | None = None,
) -> dict[str, float]:
    _stop_all(ctx)
    ctx.bot.rm_set_force_sensor(True)
    _settle(ctx, steps=max(20, ctx.config.settle_steps // 2))
    if ctx.bot.rm_start_force_position_move() != 0:
        raise RuntimeError("rm_start_force_position_move failed")

    target = _tcp_pose(ctx)
    z0 = float(target[2])
    z_cmd_end = z0
    z_meas_min = z0
    max_fz = 0.0
    steps = int(hold_steps if hold_steps is not None else ctx.config.hybrid_hold_steps)

    for i in range(steps):
        if descend:
            target = apply_pose_delta_wxyz(
                target,
                np.array([0.0, 0.0, -float(ctx.config.approach_step_m), 0.0, 0.0, 0.0], dtype=np.float32),
            )
        z_cmd_end = float(target[2])
        _hybrid_move_one(ctx, preset, target, label=f"hybrid/{preset.label}/{i}")
        z_meas_min = min(z_meas_min, float(_tcp_pose(ctx)[2]))
        max_fz = max(max_fz, abs(float(_sensor_zero_force(ctx)[2])))

    ctx.bot.rm_stop_force_position_move()
    return {
        "z0": z0,
        "z_cmd_end": z_cmd_end,
        "z_meas_min": z_meas_min,
        "z_travel": z0 - z_meas_min,
        "z_cmd_travel": z0 - z_cmd_end,
        "max_fz": max_fz,
    }


def _case(name: str, fn: Callable[[], dict[str, float | int | str | bool] | None]) -> AcceptanceCaseResult:
    try:
        metrics = fn() or {}
        return AcceptanceCaseResult(name=name, passed=True, message="ok", metrics=dict(metrics))
    except AssertionError as exc:
        return AcceptanceCaseResult(name=name, passed=False, message=str(exc))
    except Exception as exc:
        return AcceptanceCaseResult(name=name, passed=False, message=f"{type(exc).__name__}: {exc}")


def test_force_api_fields(ctx: RM75AcceptanceContext) -> dict[str, float | int | str | bool]:
    tag, data = ctx.bot.rm_get_force_data()
    assert tag == 0, f"rm_get_force_data tag={tag}"
    expected = {"force_data", "zero_force_data", "work_zero_force_data", "tool_zero_force_data"}
    assert expected.issubset(set(data)), f"missing keys: {expected - set(data)}"
    assert len(data["force_data"]) == 6
    return {"fields": len(data)}


def test_force_quiet_at_rest(ctx: RM75AcceptanceContext) -> dict[str, float | int | str | bool]:
    _stop_all(ctx)
    ctx.bot.rm_set_force_sensor(True)
    _settle(ctx)
    f0 = _sensor_zero_force(ctx)
    peak = float(np.max(np.abs(f0[:3])))
    assert peak <= float(ctx.config.quiet_force_n), f"|F|={peak:.3f}N exceeds quiet threshold"
    return {"peak_f_n": round(peak, 4)}


def test_cartesian_xy_motion(ctx: RM75AcceptanceContext) -> dict[str, float | int | str | bool]:
    _stop_all(ctx)
    ctx.bot.rm_set_force_sensor(True)
    _settle(ctx)
    start = _tcp_pose(ctx)
    _approach_bed_xy(ctx, bed_xy=(0.05, 0.0), steps=int(ctx.config.cartesian_steps))
    end = _tcp_pose(ctx)
    dxy = float(np.linalg.norm(end[:2] - start[:2]))
    min_planar = max(0.012, float(ctx.config.xy_delta_m) * 0.4)
    assert dxy >= min_planar, f"TCP planar move only {dxy:.4f} m (need >={min_planar:.4f})"
    _go_home(ctx)
    return {"dxy_m": round(dxy, 4)}


def test_contact_on_support(ctx: RM75AcceptanceContext) -> dict[str, float | int | str | bool]:
    _go_home(ctx)
    ctx.bot.rm_set_force_sensor(True)
    _settle(ctx)
    ok, max_fz, _, metrics = _make_contact(ctx)
    assert ok, (
        f"strict contact failed: need |Fz|>={_min_contact_fz(ctx):.3f}N and bed geometry, "
        f"got max_fz={max_fz:.4f} z_drop={metrics.get('z_drop', 0):.4f}"
    )
    return {k: round(float(v), 4) for k, v in metrics.items()}


def test_hybrid_state_machine(ctx: RM75AcceptanceContext) -> dict[str, float | int | str | bool]:
    _stop_all(ctx)
    assert ctx.bot.rm_start_force_position_move() == 0
    assert ctx.bot.rm_movej([0.0] * 7, v=20, r=0, connect=0, block=1) == 1
    assert ctx.bot.rm_stop_force_position_move() == 0
    assert ctx.bot.rm_movej([0.0] * 7, v=20, r=0, connect=0, block=1) == 0
    return {}


def test_streaming_contact_phase_switch(ctx: RM75AcceptanceContext) -> dict[str, float | int | str | bool]:
    _go_home(ctx)
    ctx.bot.rm_set_force_sensor(True)
    _settle(ctx)
    ok, peak_fz, target, cmetrics = _make_contact(ctx)
    assert ok, "need strict bed contact before phase test"
    fz_at_contact = float(_sensor_zero_force(ctx)[2])
    phase_fz = fz_at_contact if abs(fz_at_contact) >= _min_contact_fz(ctx) else float(peak_fz)
    _start_hybrid_streaming(ctx)
    phase_th = max(_min_contact_fz(ctx), min(_contact_threshold(ctx), abs(phase_fz) * 0.85))
    engaged, phase = update_hybrid_z_phase(
        engaged=False,
        fz_sensor=phase_fz,
        threshold_n=phase_th,
    )
    assert engaged and phase == HybridZPhase.FORCE_TRACK, (
        f"must enter FORCE_TRACK at contact: fz={phase_fz:.4f}N phase_th={phase_th:.4f} peak={peak_fz:.4f}"
    )
    track = streaming_hybrid_move_param(
        ctx.bot,
        target,
        phase=HybridZPhase.FORCE_TRACK,
        desired_fz=float(ctx.config.desired_fz),
        limit_vel=_limit_vel(ctx),
    )
    assert int(track.control_mode[2]) == RM_CTRL_FORCE_TRACK
    assert float(track.desired_force[2]) == float(ctx.config.desired_fz)
    assert ctx.bot.rm_force_position_move(track) == 0
    ctx.bot.rm_stop_force_position_move()
    return {"peak_fz": round(peak_fz, 4), "phase_th": round(phase_th, 4), "fz": round(phase_fz, 4)}


def test_mode4_resists_pose_while_contact(ctx: RM75AcceptanceContext) -> dict[str, float | int | str | bool]:
    _go_home(ctx)
    ctx.bot.rm_set_force_sensor(True)
    ok, peak_fz, _, _ = _make_contact(ctx)
    assert ok, "need strict contact for mode4 semantics"
    current = _tcp_pose(ctx)
    target = current.copy()
    target[2] -= 0.03
    limits = _limit_vel(ctx)
    motion_params = ForcePositionHybridParams(
        mode=1,
        control_mode=[RM_CTRL_MOTION, RM_CTRL_MOTION, RM_CTRL_MOTION, 0, 0, 0],
        desired_force=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        limit_vel=limits,
    )
    force_params = ForcePositionHybridParams(
        mode=1,
        control_mode=[RM_CTRL_MOTION, RM_CTRL_MOTION, RM_CTRL_FORCE_TRACK, 0, 0, 0],
        desired_force=[0.0, 0.0, float(ctx.config.desired_fz), 0.0, 0.0, 0.0],
        limit_vel=limits,
    )
    ctx.bot.hybrid.reset()
    adj3, meta3 = ctx.bot.hybrid.adjusted_pose(target, motion_params)
    ctx.bot.hybrid.reset()
    adj4, meta4 = ctx.bot.hybrid.adjusted_pose(target, force_params)
    err3 = abs(float(adj3[2]) - float(target[2]))
    err4 = abs(float(adj4[2]) - float(target[2]))
    assert err3 < err4 + 0.002, (
        f"mode3 should track Z target closer than mode4: err3={err3:.4f} err4={err4:.4f}"
    )
    assert meta3.get("sim_model") == meta4.get("sim_model") == "realman_mbk_outer_loop_v2"
    return {
        "err3": round(err3, 4),
        "err4": round(err4, 4),
        "peak_fz": round(peak_fz, 4),
    }


def test_float_mode_under_contact(ctx: RM75AcceptanceContext) -> dict[str, float | int | str | bool]:
    """Mode 1 (float) at contact: admittance outer loop runs without TCP spikes (not zero drift)."""

    _go_home(ctx)
    ctx.bot.rm_set_force_sensor(True)
    ok, peak_fz, _, _ = _make_contact(ctx)
    assert ok, "need strict contact for float"
    preset = next(p for p in HYBRID_PRESETS if p.label == "z_float_1")
    hold_pose = _tcp_pose(ctx).copy()
    limits = _limit_vel(ctx)
    float_params = ForcePositionHybridParams(
        mode=1,
        control_mode=[RM_CTRL_MOTION, RM_CTRL_MOTION, RM_CTRL_FLOAT, 0, 0, 0],
        desired_force=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        limit_vel=limits,
    )
    ctx.bot.hybrid.reset()
    adj, meta = ctx.bot.hybrid.adjusted_pose(hold_pose, float_params)
    assert bool(np.all(np.isfinite(adj))), "float adjusted_pose non-finite"
    assert meta.get("sim_model") == "realman_mbk_outer_loop_v2"
    _start_hybrid_streaming(ctx)
    max_step_dz = 0.0
    prev_z = float(hold_pose[2])
    steps = max(25, int(ctx.config.hybrid_hold_steps) // 3)
    for i in range(steps):
        _hybrid_move_one(ctx, preset, hold_pose, label=f"float_hold/{i}")
        z_now = float(_tcp_pose(ctx)[2])
        max_step_dz = max(max_step_dz, abs(z_now - prev_z))
        prev_z = z_now
    ctx.bot.rm_stop_force_position_move()
    assert max_step_dz <= float(ctx.config.max_tcp_step_hybrid_m) * 1.05, (
        f"float hold TCP step too large: {max_step_dz:.4f}m"
    )
    return {
        "max_step_dz": round(max_step_dz, 4),
        "peak_fz": round(peak_fz, 4),
        "adj_z": round(float(adj[2]), 4),
        "hold_z0": round(float(hold_pose[2]), 4),
    }


def test_spring_stiffer_than_float(ctx: RM75AcceptanceContext) -> dict[str, float | int | str | bool]:
    """Spring (mode 2) should resist a deeper Z command more than float (mode 1) at the same contact."""

    _go_home(ctx)
    ctx.bot.rm_set_force_sensor(True)
    ok, peak_fz, _, _ = _make_contact(ctx)
    assert ok, "need strict contact for spring/float compare"
    current = _tcp_pose(ctx)
    target = current.copy()
    target[2] -= 0.02
    limits = _limit_vel(ctx)
    float_params = ForcePositionHybridParams(
        mode=1,
        control_mode=[RM_CTRL_MOTION, RM_CTRL_MOTION, RM_CTRL_FLOAT, 0, 0, 0],
        desired_force=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        limit_vel=limits,
    )
    spring_params = ForcePositionHybridParams(
        mode=1,
        control_mode=[RM_CTRL_MOTION, RM_CTRL_MOTION, RM_CTRL_SPRING, 0, 0, 0],
        desired_force=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        limit_vel=limits,
    )
    ctx.bot.hybrid.reset()
    adj_float, _ = ctx.bot.hybrid.adjusted_pose(target, float_params)
    ctx.bot.hybrid.reset()
    adj_spring, _ = ctx.bot.hybrid.adjusted_pose(target, spring_params)
    err_float = abs(float(adj_float[2]) - float(target[2]))
    err_spring = abs(float(adj_spring[2]) - float(target[2]))
    assert err_spring >= err_float - 0.002, (
        f"spring should resist deeper Z cmd more than float: err_spring={err_spring:.4f} "
        f"err_float={err_float:.4f} adj_spring_z={float(adj_spring[2]):.4f} adj_float_z={float(adj_float[2]):.4f}"
    )
    return {
        "err_float": round(err_float, 4),
        "err_spring": round(err_spring, 4),
        "peak_fz": round(peak_fz, 4),
    }


def test_mode7_follows_pose_without_contact(ctx: RM75AcceptanceContext) -> dict[str, float | int | str | bool]:
    preset = next(p for p in HYBRID_PRESETS if p.label == "z_force_motion_7")
    stats = _run_hybrid_script(ctx, preset, descend=True, hold_steps=50)
    assert stats["z_cmd_travel"] > 0.004, "mode7 command did not move Z target"
    if stats["max_fz"] < _min_contact_fz(ctx):
        assert abs(stats["z_travel"] - stats["z_cmd_travel"]) < 0.02, "mode7 Z should track command before contact"
    return {k: round(float(v), 4) for k, v in stats.items()}


def test_hybrid_contact_phased_preset(ctx: RM75AcceptanceContext) -> dict[str, float | int | str | bool]:
    preset = next(p for p in HYBRID_PRESETS if p.label == "hybrid_contact_7to4")
    _go_home(ctx)
    ctx.bot.rm_set_force_sensor(True)
    ok, peak_fz, target, _ = _make_contact(ctx)
    assert ok, "need contact for phased preset"
    fz_at_contact = float(_sensor_zero_force(ctx)[2])
    phase_fz = fz_at_contact if abs(fz_at_contact) >= _min_contact_fz(ctx) else float(peak_fz)
    _start_hybrid_streaming(ctx)
    phase_th = max(_min_contact_fz(ctx), min(_contact_threshold(ctx), abs(phase_fz) * 0.85))
    _, phase = update_hybrid_z_phase(engaged=False, fz_sensor=phase_fz, threshold_n=phase_th)
    assert phase == HybridZPhase.FORCE_TRACK, (
        f"phased preset needs FORCE_TRACK: fz={phase_fz:.4f} th={phase_th:.4f} peak={peak_fz:.4f}"
    )
    param = streaming_hybrid_move_param(
        ctx.bot,
        target,
        phase=phase,
        desired_fz=float(ctx.config.desired_fz),
        limit_vel=_limit_vel(ctx),
    )
    assert int(param.control_mode[2]) == RM_CTRL_FORCE_TRACK
    assert ctx.bot.rm_force_position_move(param) == 0
    ctx.bot.rm_stop_force_position_move()
    return {"peak_fz": round(peak_fz, 4)}


def test_hybrid_preset_runs(ctx: RM75AcceptanceContext, preset: HybridPreset) -> dict[str, float | int | str | bool]:
    stats = _run_hybrid_script(ctx, preset, descend=True, hold_steps=max(35, ctx.config.hybrid_hold_steps // 2))
    assert stats["max_fz"] >= 0.0
    if int(preset.z_mode) == RM_CTRL_MOTION:
        assert stats["z_cmd_travel"] > 0.002, f"{preset.label} did not command Z motion"
    if int(preset.z_mode) == RM_CTRL_ADAPTIVE:
        ctx.bot.hybrid.reset()
        limits = _limit_vel(ctx)
        params = ForcePositionHybridParams(
            mode=1,
            control_mode=[RM_CTRL_MOTION, RM_CTRL_MOTION, RM_CTRL_ADAPTIVE, 0, 0, 0],
            desired_force=[0.0, 0.0, float(ctx.config.desired_fz), 0.0, 0.0, 0.0],
            limit_vel=limits,
        )
        _, meta = ctx.bot.hybrid.adjusted_pose(_tcp_pose(ctx), params)
        assert meta.get("sim_model") == "realman_mbk_outer_loop_v2"
    return {k: round(float(v), 4) for k, v in stats.items()}


def run_rm75_acceptance_suite(ctx: RM75AcceptanceContext) -> AcceptanceReport:
    cfg = ctx.config.scaled()
    pause = float(cfg.step_pause_s)
    if cfg.show_viewer and pause <= 0.0:
        pause = 0.01
    cfg = RM75AcceptanceConfig(
        **{**cfg.__dict__, "step_pause_s": pause},
    )
    ctx = RM75AcceptanceContext(
        runtime=ctx.runtime,
        bot=ctx.bot,
        motion=ctx.motion,
        cart=ctx.cart,
        home_q=ctx.home_q,
        config=cfg,
    )
    results: list[AcceptanceCaseResult] = []

    def run(name: str, fn: Callable[[], dict[str, float | int | str | bool] | None]) -> None:
        item = _case(name, fn)
        print(f"[{'PASS' if item.passed else 'FAIL'}] {item.name}: {item.message}", flush=True)
        results.append(item)

    banner = "=== RM75 acceptance self-test (strict"
    if cfg.show_viewer:
        banner += ", viewer ON"
    banner += ") ==="
    print(banner, flush=True)

    run("force_api_fields", lambda: test_force_api_fields(ctx))
    run("force_quiet_at_rest", lambda: test_force_quiet_at_rest(ctx))
    run("cartesian_xy_motion", lambda: test_cartesian_xy_motion(ctx))
    run("contact_on_support", lambda: test_contact_on_support(ctx))
    run("streaming_contact_phase_switch", lambda: test_streaming_contact_phase_switch(ctx))
    run("mode4_resists_pose_while_contact", lambda: test_mode4_resists_pose_while_contact(ctx))
    run("float_mode_under_contact", lambda: test_float_mode_under_contact(ctx))
    run("spring_stiffer_than_float", lambda: test_spring_stiffer_than_float(ctx))
    run("hybrid_state_machine", lambda: test_hybrid_state_machine(ctx))
    run("mode7_follows_pose_without_contact", lambda: test_mode7_follows_pose_without_contact(ctx))
    run("hybrid_contact_phased_preset", lambda: test_hybrid_contact_phased_preset(ctx))
    for preset in HYBRID_PRESETS:
        if preset.contact_phased:
            continue
        label = preset.label
        run(
            f"hybrid_preset_{label}",
            lambda p=preset: test_hybrid_preset_runs(ctx, p),
        )

    report = AcceptanceReport(results=results)
    report.print_summary()
    if cfg.show_viewer:
        print("Viewer stays open 3s after report…", flush=True)
        import time

        time.sleep(3.0)
    return report
