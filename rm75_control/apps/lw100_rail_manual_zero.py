#!/usr/bin/env python3
"""Manual-zero rail calibration (both limit switches broken).

You hand-crank the carriage to mechanical / agreed zero, then this script:
  1. FA-60 adopts the encoder frame at the current pose (same as limit-home),
  2. writes ``var/lw100_rail_zero.json`` in the same schema as
     ``lw100_rail_home_limit.py``,
  3. creeps to a soft park (default **5 mm**) and stops (FA24=0).

No DI / limit search. Travel is capped in software only — keep a hand on
e-stop / power.

  cd rm75_control && source env.sh
  # 1) manually twist rail to 0
  # 2) motor reconnected, drive powered, Modbus up
  python apps/lw100_rail_manual_zero.py --force --i-am-at-zero
  python apps/lw100_rail_manual_zero.py --force --i-am-at-zero --post-mm 5 --speed-mm-s 15
  python apps/lw100_rail_manual_zero.py --force --i-am-at-zero --release   # drop SON at end
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

from rm75_control.hw.lw100.drive import LW100Drive, LW100DriveConfig
from rm75_control.hw.lw100.rail_calibration import (
    VERSION,
    RailCalibration,
    default_calibration_path,
    load_calibration,
    save_calibration,
    sync_calibration_frame,
)
from rm75_control.hw.lw100.rail_home_limit import (
    RailHomeConfig,
    _guarded_move,
    _host_m,
    _stop,
)
from rm75_control.hw.lw100.registers import (
    P_FA24_INT_SPEED1,
    P_FA25_INT_SPEED2,
    P_FA26_INT_SPEED3,
    P_FA27_INT_SPEED4,
    P_FA53_FORCE_ENABLE,
    P_FC15_DI_FORCE1,
    P_FC16_DI_FORCE2,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--config",
        type=Path,
        default=_repo_root() / "configs/joint_admittance_8dof.yaml",
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--force", action="store_true", help="overwrite existing calibration")
    p.add_argument(
        "--i-am-at-zero",
        action="store_true",
        help="required: confirm carriage is already at the agreed zero",
    )
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument(
        "--post-mm",
        type=float,
        default=5.0,
        help="soft park after zero (host +Y mm). Default 5 mm.",
    )
    p.add_argument(
        "--speed-mm-s",
        type=float,
        default=15.0,
        help="coarse move speed toward post (mm/s). Keep low — no limit DI.",
    )
    p.add_argument(
        "--soft-min-mm",
        type=float,
        default=None,
        help="soft_min written into JSON (default: same as --post-mm)",
    )
    p.add_argument(
        "--release",
        action="store_true",
        help="release SON at end (default: keep SON for controller handoff)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _cfg_from_yaml(raw: dict, args: argparse.Namespace) -> tuple[RailHomeConfig, dict]:
    hw = raw.get("hw", {}).get("lw100", {}) or {}
    host = str(args.host or hw.get("host", "192.168.0.7"))
    port = int(args.port or hw.get("port", 8234))
    slave = int(hw.get("slave", hw.get("slave_id", 1)))
    post_m = max(float(args.post_mm), 0.5) * 1e-3
    soft_min_m = (
        max(float(args.soft_min_mm), 0.5) * 1e-3
        if args.soft_min_mm is not None
        else post_m
    )
    soft_max_m = float(hw.get("soft_max_m", 0.78))
    if not (0.0 < soft_min_m < soft_max_m):
        raise ValueError(
            f"invalid soft band soft_min={soft_min_m} soft_max={soft_max_m}"
        )
    cfg = RailHomeConfig(
        sign=float(hw.get("sign", -1.0)),
        lead_mm=float(hw.get("lead_mm", 10.0)),
        home_di=str(hw.get("home_di", "di4")),
        plus_di=str(hw.get("plus_di", "di3")),
        di_nc=bool(hw.get("di_nc", True)),
        home_to_post_m_s=max(float(args.speed_mm_s), 1.0) * 1e-3,
        post_home_m=post_m,
        soft_min_m=soft_min_m,
        soft_max_m=soft_max_m,
        max_speed_rpm=int(hw.get("max_speed_rpm", 900)),
        accel_ms=int(hw.get("accel_ms", 200)),
        decel_ms=int(hw.get("decel_ms", 200)),
        scurve_ms=int(hw.get("scurve_ms", 30)),
        host=host,
        max_origin_raw_counts=65_536,
    )
    drive_kw = {
        "host": host,
        "port": port,
        "slave_id": slave,
        "lead_mm": cfg.lead_mm,
        "timeout_s": float(hw.get("timeout_s", 0.35)),
        "retries": int(hw.get("retries", 2)),
        "inter_frame_delay_s": float(hw.get("inter_frame_delay_s", 0.002)),
        "enable_settle_s": float(hw.get("enable_settle_s", 0.3)),
        "verbose": bool(args.verbose or hw.get("verbose", False)),
    }
    return cfg, drive_kw


def _sanitize_speed_slots(
    drive: LW100Drive, log, *, drop_enable: bool = False
) -> None:
    """Zero unused speed slots (and optionally drop SON force).

    After ``start_velocity_session`` the drive needs FA53/FC15 SON held —
    never clear those then, or FA24 commands produce zero motion.
    """
    writes = [
        (P_FA24_INT_SPEED1, 0, "FA24=0"),
        (P_FA25_INT_SPEED2, 0, "FA25=0"),
        (P_FA26_INT_SPEED3, 0, "FA26=0"),
        (P_FA27_INT_SPEED4, 0, "FA27=0"),
        (P_FC16_DI_FORCE2, 0, "FC16=0"),
    ]
    if drop_enable:
        writes.extend(
            (
                (P_FC15_DI_FORCE1, 0, "FC15=0"),
                (P_FA53_FORCE_ENABLE, 0, "FA53=0"),
            )
        )
    for p, val, note in writes:
        try:
            drive.write_param(p, val)
            log(f"[manual-zero] sanitize {note}")
        except Exception as exc:  # noqa: BLE001
            log(f"[manual-zero] WARN sanitize {note}: {exc}")


def _assert_son_live(drive: LW100Drive, log) -> None:
    """Ensure software enable is on before commanding FA24."""
    try:
        fa53 = int(drive.read_param(P_FA53_FORCE_ENABLE))
        fc15 = int(drive.read_param(P_FC15_DI_FORCE1))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"cannot read enable state: {exc}") from exc
    log(f"[manual-zero] enable check FA53={fa53} FC15={fc15}")
    if fa53 == 1 and (fc15 & 1):
        return
    log("[manual-zero] SON not latched — re-enable (no FA61)")
    drive.enable()
    time.sleep(max(0.15, float(drive.config.enable_settle_s)))
    fa53 = int(drive.read_param(P_FA53_FORCE_ENABLE))
    fc15 = int(drive.read_param(P_FC15_DI_FORCE1))
    log(f"[manual-zero] after enable FA53={fa53} FC15={fc15}")
    if not (fa53 == 1 and (fc15 & 1)):
        raise RuntimeError(
            f"SON still off after enable (FA53={fa53} FC15={fc15}) — check drive alarm"
        )


def _adopt_zero_here(drive: LW100Drive, cfg: RailHomeConfig, log) -> int:
    """FA-60 adopt at current pose; return origin_raw (must be near 0)."""
    raw_pre = int(drive._read_encoder_counts_raw(retries=3))
    log(f"[manual-zero] FA-60 adopt at manual zero (raw_pre={raw_pre})")
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
        log(f"[manual-zero] WARN FA-20 after adopt: {exc}")
    if not drive.frame_trusted:
        raise RuntimeError("encoder frame untrusted after FA-60 adopt")
    raw_post = int(drive._read_encoder_counts_raw(retries=3))
    # Stationary adopt: origin is the post-wipe reading (same physical point).
    origin_raw = int(raw_post)
    log(
        f"[manual-zero] adopt post={post_adopt} raw_post={raw_post} "
        f"origin_raw={origin_raw}"
    )
    max_origin = int(cfg.max_origin_raw_counts)
    if abs(int(origin_raw)) > max_origin:
        raise RuntimeError(
            f"FA-60 did not clear monitor near zero "
            f"(raw_counts0={origin_raw}, max={max_origin}) — abort"
        )
    drive.set_rail_zero_raw(int(origin_raw))
    host_now = _host_m(drive, cfg.sign)
    log(
        f"[manual-zero] software zero latched; "
        f"host_m={host_now * 1000:.2f} mm raw_counts0={origin_raw}"
    )
    return int(origin_raw)


def _move_to_post(drive: LW100Drive, cfg: RailHomeConfig, log) -> float:
    """Creep host +Y to post_home_m. No DI — travel hard-capped."""
    _assert_son_live(drive, log)
    target = abs(float(cfg.post_home_m))
    v_post = abs(float(cfg.home_to_post_m_s))
    settle_m = 0.0004
    approach_band = 0.0015
    log(
        f"[manual-zero] move to post={target * 1000:.1f} mm "
        f"@ {v_post * 1000:.0f} mm/s (NO limit DI — software cap only)"
    )
    coarse_goal = max(0.0, target - approach_band)

    def _near_coarse(meas: float, _start: float) -> bool:
        if meas < -0.002:
            raise RuntimeError(
                f"post move crossed below zero (meas={meas * 1000:.2f} mm)"
            )
        if meas > target + 0.003:
            raise RuntimeError(
                f"post move overshot (meas={meas * 1000:.2f} mm > "
                f"{(target + 0.003) * 1000:.1f} mm)"
            )
        return meas >= coarse_goal

    if abs(_host_m(drive, cfg.sign) - target) > approach_band:
        _guarded_move(
            drive,
            cfg,
            v_post,
            what="manual post coarse",
            timeout_s=30.0,
            max_travel_m=target + 0.008,
            stop_when=_near_coarse,
            probe_s=0.35,
            min_progress_m=0.0002,
            log=log,
        )
        _stop(drive)
        time.sleep(0.15)

    creep = min(0.003, v_post)

    def _in_settle(meas: float, _start: float) -> bool:
        if meas < -0.002:
            raise RuntimeError(
                f"post creep crossed below zero (meas={meas * 1000:.2f} mm)"
            )
        return abs(target - meas) <= settle_m

    meas = _host_m(drive, cfg.sign)
    if abs(target - meas) > settle_m:
        err = target - meas
        _guarded_move(
            drive,
            cfg,
            creep if err > 0 else -creep,
            what="manual post creep",
            timeout_s=15.0,
            max_travel_m=abs(err) + 0.004,
            stop_when=_in_settle,
            probe_s=0.25,
            min_progress_m=0.0001,
            log=log,
        )
    _stop(drive)
    time.sleep(0.25)
    final_m = _host_m(drive, cfg.sign)
    err = target - final_m
    if abs(err) > 0.0015:
        raise RuntimeError(
            f"post settle failed: meas={final_m * 1000:.2f} mm "
            f"target={target * 1000:.1f} mm err={err * 1000:+.2f} mm"
        )
    log(f"[manual-zero] parked @ {final_m * 1000:.2f} mm")
    return float(final_m)


def main() -> int:
    args = parse_args()
    if not args.i_am_at_zero:
        print(
            "[manual-zero] refuse: hand-crank to zero first, then re-run with "
            "--i-am-at-zero --force",
            file=sys.stderr,
            flush=True,
        )
        return 2
    if not args.force:
        print(
            "[manual-zero] refuse: pass --force to overwrite calibration",
            file=sys.stderr,
            flush=True,
        )
        return 2

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        print(f"config not found: {cfg_path}", file=sys.stderr)
        return 2
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    try:
        home_cfg, drive_kw = _cfg_from_yaml(raw, args)
    except ValueError as exc:
        print(f"[manual-zero] BAD ARGS: {exc}", file=sys.stderr)
        return 2

    hw = raw.get("hw", {}).get("lw100", {}) or {}
    if args.out is not None:
        out = Path(args.out)
    elif hw.get("calibration_path"):
        out = Path(str(hw["calibration_path"]))
        if not out.is_absolute():
            out = (_repo_root() / out).resolve()
    else:
        out = default_calibration_path(_repo_root())

    existing = load_calibration(out)
    if existing is not None:
        print(
            f"[manual-zero] overwriting {out} "
            f"(counts0={existing.raw_counts0}, last_raw={existing.last_raw_counts})",
            flush=True,
        )

    def log(msg: str) -> None:
        print(msg, flush=True)

    keep_son = not bool(args.release)
    print(
        f"[manual-zero] {drive_kw['host']}:{drive_kw['port']} → "
        f"adopt@here → park {home_cfg.post_home_m * 1000:.1f} mm → {out}",
        flush=True,
    )
    print(
        "[manual-zero] LIMITS BROKEN — software travel cap only. "
        "Keep e-stop ready.",
        flush=True,
    )

    drive = LW100Drive(LW100DriveConfig(**drive_kw))
    drive._disable_on_exit = not keep_son  # noqa: SLF001
    try:
        drive.connect()
        # Before SON: wipe stale FA24=180 and force-enable leftovers.
        _sanitize_speed_slots(drive, log, drop_enable=True)
        drive.start_velocity_session(
            accel_ms=int(home_cfg.accel_ms),
            decel_ms=int(home_cfg.decel_ms),
            scurve_ms=int(home_cfg.scurve_ms),
            max_speed_rpm=min(int(home_cfg.max_speed_rpm), 600),
        )
        # Keep SON; only re-zero speed slots.
        _sanitize_speed_slots(drive, log, drop_enable=False)
        _stop(drive)
        if not drive.frame_trusted:
            print(
                "[manual-zero] FAILED: encoder frame untrusted at session start",
                file=sys.stderr,
                flush=True,
            )
            return 1

        try:
            origin_raw = _adopt_zero_here(drive, home_cfg, log)
            # adopt clears velocity session — restart safely
            drive.start_velocity_session(
                accel_ms=int(home_cfg.accel_ms),
                decel_ms=int(home_cfg.decel_ms),
                scurve_ms=int(home_cfg.scurve_ms),
                max_speed_rpm=min(int(home_cfg.max_speed_rpm), 600),
            )
            _sanitize_speed_slots(drive, log, drop_enable=False)
            _stop(drive)
            _assert_son_live(drive, log)

            final_m = _move_to_post(drive, home_cfg, log)
            raw_now = int(drive._read_encoder_counts_raw(retries=3))
            cal = RailCalibration(
                version=VERSION,
                raw_counts0=int(origin_raw),
                counts0_host=int(drive._counts0),
                last_raw_counts=raw_now,
                frame_origin_at_home=True,
                sign=float(home_cfg.sign),
                lead_mm=float(home_cfg.lead_mm),
                soft_min_m=float(home_cfg.soft_min_m),
                soft_max_m=float(home_cfg.soft_max_m),
                post_home_m=float(abs(home_cfg.post_home_m)),
                rail_m_at_cal=float(final_m),
                host=str(home_cfg.host),
                calibrated_unix=time.time(),
                valid=True,
            )
            save_calibration(out, cal)
            sync_calibration_frame(out, drive, require_continuity=False)
            verified = load_calibration(out)
            if (
                verified is None
                or int(verified.version) != int(VERSION)
                or not bool(verified.frame_origin_at_home)
            ):
                raise RuntimeError(f"calibration save verify failed at {out}")
            log(
                f"[manual-zero] saved + verified → {out} "
                f"(raw_counts0={verified.raw_counts0}, "
                f"last_raw={verified.last_raw_counts}, "
                f"soft_min={verified.soft_min_m * 1000:.1f} mm, "
                f"rail_m_at_cal={verified.rail_m_at_cal * 1000:.2f} mm)"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[manual-zero] FAILED: {exc}", file=sys.stderr, flush=True)
            try:
                _stop(drive)
            except Exception:
                pass
            if keep_son:
                try:
                    drive.disable()
                except Exception:
                    pass
            return 1
        finally:
            try:
                _stop(drive)
            except Exception:
                pass
            if not keep_son:
                try:
                    drive.disable()
                except Exception:
                    pass
    finally:
        try:
            drive.close()
        except Exception:
            pass

    if keep_son:
        sm = home_cfg.soft_min_m * 1000.0
        print(
            "[manual-zero] OK — SON held (FA24=0). Start Window A without "
            "power-cycling the drive.\n"
            f"  IMPORTANT: JSON soft_min={sm:.1f} mm. Match YAML before Window A:\n"
            f"    hw.lw100.soft_min_m / post_home_m  → {home_cfg.soft_min_m}\n"
            f"    qpik.hard_limits.rail.soft_min_m   → {home_cfg.soft_min_m}\n"
            "  otherwise rail_servo rejects soft-limit mismatch.",
            flush=True,
        )
    else:
        print("[manual-zero] OK — SON released.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
