"""Host-side dual-speed limit homing for LW100 (FA24 + Modbus DI).

Calibration contact is expected: stop → backoff → slow multi-touch.
Never treat a limit press during homing as a fault / e-stop.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rm75_control.hw.lw100.drive import LW100Drive
from rm75_control.hw.lw100.rail_calibration import (
    ENCODER_CPR,
    VERSION,
    RailCalibration,
    load_calibration,
    save_calibration,
)


@dataclass
class RailHomeConfig:
    sign: float = -1.0
    lead_mm: float = 10.0
    home_di: str = "di4"  # −Y home (HW confirmed)
    plus_di: str = "di3"
    di_nc: bool = True
    di_debounce_n: int = 3
    home_search_m_s: float = 0.020
    home_creep_m_s: float = 0.003
    home_backoff_mm: float = 5.0
    home_touch_count: int = 3
    home_search_timeout_s: float = 60.0
    home_to_post_m_s: float = 0.030
    post_home_m: float = 0.01
    soft_min_m: float = 0.01
    soft_max_m: float = 0.78
    max_speed_rpm: int = 900
    accel_ms: int = 200
    decel_ms: int = 200
    scurve_ms: int = 30
    host: str = "192.168.0.7"
    # After FA-60 adopt, |raw_counts0| must be within this (5 mm @ 10 mm/rev).
    max_origin_raw_counts: int = 65_536


def _di_index(name: str) -> int:
    n = str(name).strip().lower()
    if n in ("di3", "3", "a", "home", "neg", "-y"):
        return 0
    if n in ("di4", "4", "b", "plus", "pos", "+y"):
        return 1
    raise ValueError(f"unknown DI name {name!r} (use di3/di4)")


def _di_name(index: int) -> str:
    return "di3" if int(index) == 0 else "di4"


def _mps_to_rpm(v_m_s: float, lead_mm: float) -> float:
    lead = max(float(lead_mm), 1e-6)
    return float(v_m_s) * 1000.0 / lead * 60.0


def _host_m(drive: LW100Drive, sign: float) -> float:
    return float(sign) * float(drive.read_rail_m_fast())


def _set_host_velocity(
    drive: LW100Drive, v_host_m_s: float, *, sign: float, lead_mm: float
) -> int:
    """Positive v_host increases host rail_y (+Y)."""
    rpm = float(sign) * _mps_to_rpm(v_host_m_s, lead_mm)
    return drive.set_velocity_rpm(rpm, force=True)


def _pressed_pair(drive: LW100Drive, cfg: RailHomeConfig) -> tuple[bool, bool]:
    return drive.read_limit_pressed(nc=bool(cfg.di_nc), debounce_n=int(cfg.di_debounce_n))


def _slot_pressed(pair: tuple[bool, bool], slot: int) -> bool:
    return bool(pair[int(slot)])


def _any_limit(pair: tuple[bool, bool]) -> bool:
    return bool(pair[0] or pair[1])


def _first_pressed_slot(pair: tuple[bool, bool], prefer: int) -> int | None:
    """Return pressed DI slot; prefer ``prefer`` if both pressed."""
    if pair[prefer]:
        return int(prefer)
    if pair[0]:
        return 0
    if pair[1]:
        return 1
    return None


def _stop(drive: LW100Drive) -> None:
    try:
        drive.set_velocity_rpm(0, force=True)
    except Exception:
        try:
            drive.kill_velocity_hard(attempts=2, disable_on_fail=False)
        except Exception:
            pass


def _settle_after_stop(cfg: RailHomeConfig, *, pad_s: float = 0.08) -> None:
    """Wait for FA41 coast to finish before reversing (avoids false wrong-way)."""
    time.sleep(max(0.05, float(cfg.decel_ms) * 1e-3 + float(pad_s)))


def _guarded_move(
    drive: LW100Drive,
    cfg: RailHomeConfig,
    v_host: float,
    *,
    what: str,
    timeout_s: float,
    max_travel_m: float,
    stop_when: Callable[[float, float], bool] | None = None,
    probe_s: float = 0.35,
    min_progress_m: float = 0.0002,
    wrong_way_m: float = 0.001,
    log: Callable[[str], None] | None = None,
) -> str:
    """Command ``v_host`` with direction / liveness / travel / timeout guards.

    ``stop_when(meas, start)`` → True to stop successfully.
    Direction check: only hard-fail on clear reverse motion
    (``progress < -wrong_way_m``). Merely slow/stuck is left to the
    encoder-liveness and timeout guards — important when reversing off a
    hard limit where FA41 coast still dribbles into the switch.
    """

    def _log(msg: str) -> None:
        if log is not None:
            log(msg)
        else:
            print(msg, flush=True)

    if abs(float(v_host)) < 1e-9:
        _stop(drive)
        return "zero_cmd"

    start = _host_m(drive, cfg.sign)
    start_raw = int(drive._read_encoder_counts_raw(retries=2))
    _set_host_velocity(drive, float(v_host), sign=cfg.sign, lead_mm=cfg.lead_mm)
    t0 = time.monotonic()
    deadline = t0 + max(0.5, float(timeout_s))
    dir_checked = False
    last_raw = start_raw
    last_raw_t = t0
    sgn = 1.0 if float(v_host) > 0.0 else -1.0
    # Stuck check is later / looser than wrong-way (reversals need time).
    stuck_s = max(float(probe_s) + 0.35, 0.70)

    while time.monotonic() < deadline:
        now = time.monotonic()
        meas = _host_m(drive, cfg.sign)
        travel = abs(meas - start)
        if travel > float(max_travel_m):
            _stop(drive)
            raise RuntimeError(
                f"{what}: travel {travel * 1000:.1f} mm exceeded "
                f"max {float(max_travel_m) * 1000:.1f} mm"
            )

        if stop_when is not None:
            try:
                done = bool(stop_when(meas, start))
            except Exception:
                _stop(drive)
                raise
            if done:
                _stop(drive)
                return "stop_when"

        progress = (meas - start) * sgn

        # Clear reverse only (not "not enough progress yet").
        if (now - t0) >= float(probe_s) and progress < -abs(float(wrong_way_m)):
            _stop(drive)
            raise RuntimeError(
                f"{what}: axis moved the wrong way "
                f"(Δ={(meas - start) * 1000:+.2f} mm, "
                f"expected sign={'+' if sgn > 0 else '-'})"
            )

        # Slow / stuck after a longer window (reverse-off-limit needs time).
        if not dir_checked and (now - t0) >= stuck_s:
            dir_checked = True
            if progress < float(min_progress_m):
                _stop(drive)
                raise RuntimeError(
                    f"{what}: axis stuck / no progress "
                    f"(Δ={(meas - start) * 1000:+.2f} mm after {stuck_s:.2f}s, "
                    f"expected sign={'+' if sgn > 0 else '-'})"
                )

        # Encoder liveness while commanding non-zero speed.
        try:
            raw_now = int(drive._read_encoder_counts_raw(retries=1))
        except Exception:
            raw_now = last_raw
        if raw_now != last_raw:
            last_raw = raw_now
            last_raw_t = now
        elif (now - last_raw_t) > 0.70 and (now - t0) > stuck_s:
            _stop(drive)
            raise RuntimeError(
                f"{what}: no encoder motion for 0.7 s "
                f"(SON dropped after FA-60? axis jammed?)"
            )

        time.sleep(0.02)

    _stop(drive)
    raise TimeoutError(f"{what}: timed out after {timeout_s:.1f}s")


def home_to_limit(
    drive: LW100Drive,
    cfg: RailHomeConfig,
    *,
    log: Callable[[str], None] | None = None,
) -> RailCalibration:
    """Search toward −Y, on limit contact: stop → backoff → creep×N → zero → post_home.

    Hitting a limit during this sequence is normal (not a fault).
    """

    def _log(msg: str) -> None:
        if log is not None:
            log(msg)
        else:
            print(msg, flush=True)

    drive.ensure_fa20_ignore()
    _stop(drive)

    # Temporary zero for relative backoff distances during search.
    drive.set_rail_zero()
    prefer = _di_index(cfg.home_di)
    search_v = -abs(float(cfg.home_search_m_s))  # host −Y
    _log(
        f"[home] search -Y @ {abs(search_v) * 1000:.0f} mm/s "
        f"(home={cfg.home_di}, sign={cfg.sign})"
    )

    # --- Search: first limit contact wins (expected) ---
    home_slot: int | None = None
    pair = _pressed_pair(drive, cfg)
    if _any_limit(pair):
        home_slot = _first_pressed_slot(pair, prefer)
        _stop(drive)
        _log(f"[home] already on {_di_name(home_slot)} — backoff/creep")
    else:
        def _hit(_meas: float, _start: float) -> bool:
            hit = _first_pressed_slot(_pressed_pair(drive, cfg), prefer)
            if hit is not None:
                nonlocal home_slot
                home_slot = hit
                return True
            return False

        _guarded_move(
            drive,
            cfg,
            search_v,
            what="limit search",
            timeout_s=float(cfg.home_search_timeout_s),
            max_travel_m=0.85,
            stop_when=_hit,
            probe_s=0.40,
            min_progress_m=0.0005,
            log=_log,
        )
        if home_slot is not None:
            _log(f"[home] search hit {_di_name(home_slot)}")

    assert home_slot is not None
    active_home = _di_name(home_slot)
    # Search was 20 mm/s into the switch — wait out FA41 before reversing.
    _stop(drive)
    _settle_after_stop(cfg)
    _log(f"[home] settled after {active_home} contact (FA41={cfg.decel_ms} ms)")

    # --- Multi-creep on the DI we actually hit ---
    samples: list[int] = []
    n_touch = max(1, int(cfg.home_touch_count))
    backoff_m = max(0.001, float(cfg.home_backoff_mm) * 1e-3)
    creep = abs(float(cfg.home_creep_m_s))
    leave_v = -search_v  # +Y if search was −Y
    approach_v = search_v

    for i in range(n_touch):
        _log(
            f"[home] backoff {cfg.home_backoff_mm:.1f} mm "
            f"(touch {i + 1}/{n_touch}, home={active_home})"
        )
        leave_speed = creep if leave_v > 0 else -creep

        def _backed(meas: float, start: float) -> bool:
            delta = meas - start
            if leave_v > 0:
                return delta >= backoff_m
            return delta <= -backoff_m

        # Reverse off a pressed switch: allow brief unload; only hard-fail
        # on clear reverse (>1.5 mm wrong way).
        _guarded_move(
            drive,
            cfg,
            leave_speed,
            what=f"backoff touch {i + 1}",
            timeout_s=15.0,
            max_travel_m=backoff_m + 0.002,
            stop_when=_backed,
            probe_s=0.45,
            min_progress_m=0.00015,
            wrong_way_m=0.0015,
            log=_log,
        )
        _stop(drive)
        _settle_after_stop(cfg, pad_s=0.05)

        _log(f"[home] creep → {active_home} @ {creep * 1000:.1f} mm/s")
        approach_speed = -creep if approach_v < 0 else creep
        got_touch = False

        def _on_switch(_meas: float, _start: float) -> bool:
            nonlocal got_touch
            if _slot_pressed(_pressed_pair(drive, cfg), home_slot):
                got_touch = True
                return True
            return False

        _guarded_move(
            drive,
            cfg,
            approach_speed,
            what=f"creep touch {i + 1}",
            timeout_s=15.0,
            max_travel_m=backoff_m + 0.003,
            stop_when=_on_switch,
            probe_s=0.30,
            min_progress_m=0.00015,
            log=_log,
        )
        _stop(drive)
        _settle_after_stop(cfg, pad_s=0.05)
        if not got_touch:
            raise TimeoutError(f"creep did not reach {active_home}")
        raw = int(drive._read_encoder_counts_raw(retries=3))
        samples.append(raw)
        _log(f"[home] touch {i + 1}: raw_counts={raw}")
        time.sleep(0.15)

    if not samples:
        raise RuntimeError("no touch samples")
    med = int(statistics.median(samples))
    if len(samples) >= 3:
        mad = statistics.median([abs(s - med) for s in samples])
        kept = [s for s in samples if abs(s - med) <= max(50, 3 * mad)]
        if kept:
            med = int(statistics.median(kept))
    _log(f"[home] median pre-adopt raw={med} from {samples}")

    # Pin encoder frame origin at mechanical home via FA-60.
    # Carriage is stationary (FA24=0) → origin from count delta, no DI re-creep.
    raw_pre = int(drive._read_encoder_counts_raw(retries=3))
    _log("[home] FA-60 adopt encoder frame at home switch (bias cleared)")
    post_adopt = int(drive.adopt_encoder_frame())
    drive.rewire_velocity_after_adopt(
        accel_ms=int(cfg.accel_ms),
        decel_ms=int(cfg.decel_ms),
        scurve_ms=int(cfg.scurve_ms),
        max_speed_rpm=int(cfg.max_speed_rpm),
    )
    try:
        drive.ensure_fa20_ignore()
    except Exception as exc:  # noqa: BLE001
        _log(f"[home] WARN FA-20 after adopt: {exc}")
    if not drive.frame_trusted:
        raise RuntimeError("encoder frame untrusted after FA-60 adopt")

    raw_post = int(drive._read_encoder_counts_raw(retries=3))
    # Same physical point: origin_raw = raw_post - (raw_pre - med)
    origin_raw = int(raw_post) - (int(raw_pre) - int(med))
    _log(
        f"[home] FA-60 adopt: raw_pre={raw_pre} post_raw={post_adopt} "
        f"raw_post={raw_post} origin_raw={origin_raw} "
        f"(|origin| must be < {cfg.max_origin_raw_counts})"
    )

    max_origin = int(cfg.max_origin_raw_counts)
    if abs(int(origin_raw)) > max_origin:
        raise RuntimeError(
            f"FA-60 did not clear monitor to near-zero "
            f"(raw_counts0={origin_raw}, max={max_origin} ≈ "
            f"{max_origin / ENCODER_CPR * float(cfg.lead_mm):.1f} mm) — "
            f"refusing to save calibration"
        )

    drive.set_rail_zero_raw(int(origin_raw))
    host_now = _host_m(drive, cfg.sign)
    _log(
        f"[home] software zero set (frame origin at home); "
        f"host_m={host_now * 1000:.2f} mm raw_counts0={origin_raw}"
    )

    # Park off the switch (same leave direction as backoff).
    if leave_v < 0:
        target = -abs(float(cfg.post_home_m))
    else:
        target = abs(float(cfg.post_home_m))

    v_post = abs(float(cfg.home_to_post_m_s))
    settle_m = 0.0003  # 0.3 mm accept (FA41 coast)
    approach_band = 0.0015  # switch to creep 1.5 mm early
    _log(f"[home] move to post_home={target * 1000:.1f} mm @ {v_post * 1000:.0f} mm/s")

    # Phase 1: coarse approach to target - approach_band.
    coarse_goal = target - (approach_band if target > 0 else -approach_band)

    def _near_coarse(meas: float, _start: float) -> bool:
        if meas < -0.001:
            raise RuntimeError(
                f"post_home: crossed home (meas={meas * 1000:.2f} mm < -1 mm)"
            )
        if target > 0:
            return meas >= coarse_goal
        return meas <= coarse_goal

    if abs(_host_m(drive, cfg.sign) - target) > approach_band:
        _guarded_move(
            drive,
            cfg,
            v_post if target > 0 else -v_post,
            what="post_home coarse",
            timeout_s=15.0,
            max_travel_m=abs(target) + 0.005,
            stop_when=_near_coarse,
            probe_s=0.35,
            min_progress_m=0.0003,
            log=_log,
        )
        _stop(drive)
        time.sleep(0.15)

    # Phase 2: creep into settle band.
    creep_post = min(0.003, v_post)

    def _in_settle(meas: float, _start: float) -> bool:
        if meas < -0.001:
            raise RuntimeError(
                f"post_home: crossed home (meas={meas * 1000:.2f} mm < -1 mm)"
            )
        return abs(target - meas) <= settle_m

    meas = _host_m(drive, cfg.sign)
    if abs(target - meas) > settle_m:
        err = target - meas
        _guarded_move(
            drive,
            cfg,
            creep_post if err > 0 else -creep_post,
            what="post_home creep",
            timeout_s=10.0,
            max_travel_m=abs(err) + 0.003,
            stop_when=_in_settle,
            probe_s=0.25,
            min_progress_m=0.0001,
            log=_log,
        )
    _stop(drive)
    time.sleep(0.30)  # FA41 finish

    meas = _host_m(drive, cfg.sign)
    err = target - meas
    if abs(err) > settle_m:
        # One gentle correction only.
        nudge = min(0.003, max(0.001, abs(err)))
        _log(f"[home] post_home nudge {err * 1000:+.2f} mm @ {nudge * 1000:.1f} mm/s")
        _set_host_velocity(
            drive,
            nudge if err > 0 else -nudge,
            sign=cfg.sign,
            lead_mm=cfg.lead_mm,
        )
        time.sleep(min(0.4, abs(err) / max(nudge, 1e-6)))
        _stop(drive)
        time.sleep(0.25)
        meas = _host_m(drive, cfg.sign)
        err = target - meas
        if abs(err) > 0.001:  # 1 mm hard fail
            raise RuntimeError(
                f"post_home settle failed: meas={meas * 1000:.2f} mm "
                f"target={target * 1000:.1f} mm err={err * 1000:+.2f} mm"
            )

    _stop(drive)
    time.sleep(0.2)
    final_m = _host_m(drive, cfg.sign)
    if final_m < -0.001:
        raise RuntimeError(
            f"post_home ended past home: {final_m * 1000:.2f} mm"
        )
    raw_now = int(drive._read_encoder_counts_raw(retries=3))
    cal = RailCalibration(
        raw_counts0=int(origin_raw),
        counts0_host=int(drive._counts0),
        last_raw_counts=raw_now,
        frame_origin_at_home=True,
        sign=float(cfg.sign),
        lead_mm=float(cfg.lead_mm),
        soft_min_m=float(cfg.soft_min_m),
        soft_max_m=float(cfg.soft_max_m),
        post_home_m=float(abs(cfg.post_home_m)),
        rail_m_at_cal=float(final_m),
        host=str(cfg.host),
        calibrated_unix=time.time(),
        valid=True,
    )
    _log(
        f"[home] done @ {final_m * 1000:.2f} mm  raw_counts0={origin_raw}  "
        f"last_raw={raw_now}  home_di={active_home}  frame_origin_at_home=True"
    )
    return cal


def home_and_save(
    drive: LW100Drive,
    cfg: RailHomeConfig,
    path: str | Path,
    *,
    log: Callable[[str], None] | None = None,
) -> RailCalibration:
    cal = home_to_limit(drive, cfg, log=log)
    out = save_calibration(Path(path), cal)
    verified = load_calibration(out)
    if (
        verified is None
        or int(verified.version) != int(VERSION)
        or not bool(verified.frame_origin_at_home)
    ):
        raise RuntimeError(
            f"calibration save verify failed at {out} "
            f"(got version={getattr(verified, 'version', None)}, "
            f"frame_origin_at_home={getattr(verified, 'frame_origin_at_home', None)})"
        )
    msg = (
        f"[home] saved + verified → {out} "
        f"(version={verified.version}, frame_origin_at_home=True)"
    )
    if log is not None:
        log(msg)
    else:
        print(msg, flush=True)
    return cal
