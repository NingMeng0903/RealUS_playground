#!/usr/bin/env python3
"""Slow crawl to soft_max (default 780 mm) after an existing rail zero.

Assumes ``apps/lw100_rail_home_limit.py`` already wrote ``var/lw100_rail_zero.json``.
Does NOT re-home. Moves at a constant host speed (default 20 mm/s) toward
``soft_max_m`` while polling both end-limit DIs; any press → FA24=0 stop.

  cd rm75_control && source env.sh
  python apps/lw100_rail_crawl_soft_max.py            # dry-run: print plan
  python apps/lw100_rail_crawl_soft_max.py --run      # hardware crawl
  python apps/lw100_rail_crawl_soft_max.py --run --target-mm 780 --speed-mm-s 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

from rm75_control.hw.lw100.drive import LW100Drive, LW100DriveConfig
from rm75_control.hw.lw100.rail_calibration import (
    default_calibration_path,
    load_calibration,
    sync_calibration_frame,
    validate_on_drive,
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
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--target-mm", type=float, default=None, help="default: soft_max_m×1000")
    p.add_argument("--speed-mm-s", type=float, default=20.0, help="host crawl speed")
    p.add_argument("--poll-hz", type=float, default=40.0)
    p.add_argument("--tol-mm", type=float, default=1.0, help="arrive when |err| ≤ this")
    p.add_argument(
        "--release",
        action="store_true",
        help="release SON at end (default: keep SON on)",
    )
    p.add_argument("--run", action="store_true", help="execute on hardware")
    return p.parse_args()


def _mps_to_rpm(v_m_s: float, lead_mm: float) -> float:
    return float(v_m_s) * 1000.0 / max(float(lead_mm), 1e-6) * 60.0


def _host_m(drive: LW100Drive, sign: float) -> float:
    return float(sign) * float(drive.read_rail_m_fast())


def _set_host_velocity(
    drive: LW100Drive, v_host_m_s: float, *, sign: float, lead_mm: float
) -> int:
    """Positive v_host increases host rail_y (+Y)."""
    rpm = float(sign) * _mps_to_rpm(v_host_m_s, lead_mm)
    return drive.set_velocity_rpm(rpm, force=True)


def _di_label(home_di: str, plus_di: str, di3_p: bool, di4_p: bool) -> str:
    parts: list[str] = []
    for name, pressed in (("di3", di3_p), ("di4", di4_p)):
        if not pressed:
            continue
        role = "home(-Y)" if name == home_di.lower() else (
            "plus(+Y)" if name == plus_di.lower() else "?"
        )
        parts.append(f"{name}/{role}")
    return ",".join(parts) if parts else "none"


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        print(f"config not found: {cfg_path}", file=sys.stderr)
        return 2

    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    hw = raw.get("hw", {}).get("lw100", {}) or {}
    host = str(args.host or hw.get("host", "192.168.0.7"))
    port = int(args.port or hw.get("port", 8234))
    slave = int(hw.get("slave", hw.get("slave_id", 1)))
    sign = float(hw.get("sign", -1.0))
    lead_mm = float(hw.get("lead_mm", 10.0))
    home_di = str(hw.get("home_di", "di4")).lower()
    plus_di = str(hw.get("plus_di", "di3")).lower()
    di_nc = bool(hw.get("di_nc", True))
    di_debounce_n = int(hw.get("di_debounce_n", 3))
    soft_min_m = float(hw.get("soft_min_m", 0.01))
    soft_max_m = float(hw.get("soft_max_m", 0.78))
    target_m = (
        float(args.target_mm) * 1e-3
        if args.target_mm is not None
        else soft_max_m
    )
    speed_m_s = float(args.speed_mm_s) * 1e-3
    if speed_m_s <= 0.0:
        print("speed-mm-s must be > 0", file=sys.stderr)
        return 2
    if not (soft_min_m - 1e-6 <= target_m <= soft_max_m + 1e-6):
        print(
            f"target {target_m * 1000:.1f} mm outside soft band "
            f"[{soft_min_m * 1000:.0f}, {soft_max_m * 1000:.0f}] mm",
            file=sys.stderr,
        )
        return 2

    if hw.get("calibration_path"):
        cal_path = Path(str(hw["calibration_path"]))
        if not cal_path.is_absolute():
            cal_path = (_repo_root() / cal_path).resolve()
    else:
        cal_path = default_calibration_path(_repo_root())

    cal = load_calibration(cal_path)
    if cal is None:
        print(
            f"no valid calibration at {cal_path} — "
            f"run apps/lw100_rail_home_limit.py --force first",
            file=sys.stderr,
        )
        return 2

    rpm_cmd = abs(_mps_to_rpm(speed_m_s, lead_mm))
    print("=== LW100 crawl to soft_max (limit-DI e-stop) ===", flush=True)
    print(f"  drive: {host}:{port}  slave={slave}", flush=True)
    print(f"  cal:   {cal_path}  raw_counts0={cal.raw_counts0}", flush=True)
    print(
        f"  plan:  → {target_m * 1000:.1f} mm @ {speed_m_s * 1000:.1f} mm/s "
        f"(~{rpm_cmd:.0f} r/min)  soft=[{soft_min_m * 1000:.0f},{soft_max_m * 1000:.0f}] mm",
        flush=True,
    )
    print(
        f"  stop:  any of {home_di}(home/-Y) or {plus_di}(plus/+Y) pressed "
        f"(nc={di_nc}) → FA24=0",
        flush=True,
    )
    if not args.run:
        print("  dry-run only — re-run with --run to execute", flush=True)
        return 0

    drive_cfg = LW100DriveConfig(
        host=host,
        port=port,
        slave_id=slave,
        lead_mm=lead_mm,
        timeout_s=float(hw.get("timeout_s", 0.35)),
        retries=int(hw.get("retries", 2)),
        inter_frame_delay_s=float(hw.get("inter_frame_delay_s", 0.002)),
        enable_settle_s=float(hw.get("enable_settle_s", 0.3)),
        verbose=bool(hw.get("verbose", False)),
    )
    keep_son = not bool(args.release)
    drive = LW100Drive(drive_cfg)
    drive._disable_on_exit = not keep_son  # noqa: SLF001
    poll_dt = 1.0 / max(float(args.poll_hz), 1.0)
    tol_m = max(float(args.tol_mm), 0.1) * 1e-3
    reason = "unknown"
    host_now = float("nan")

    try:
        drive.connect()
        drive.start_velocity_session(
            accel_ms=int(hw.get("accel_ms", 200)),
            decel_ms=int(hw.get("decel_ms", 200)),
            scurve_ms=int(hw.get("scurve_ms", 30)),
            max_speed_rpm=int(hw.get("max_speed_rpm", 900)),
        )
        if not drive.frame_trusted:
            print(
                "FAILED: encoder frame untrusted during session setup",
                file=sys.stderr,
                flush=True,
            )
            return 1

        ok, why, host_now, power_cycle, _comms = validate_on_drive(
            drive,
            cal,
            sign=sign,
            check_limit_di=True,
            di_nc=di_nc,
            home_di=home_di,
            plus_di=plus_di,
        )
        if not ok:
            print(f"FAILED: calibration invalid — {why}", file=sys.stderr, flush=True)
            if power_cycle:
                print(
                    "  re-run: python apps/lw100_rail_home_limit.py --force",
                    file=sys.stderr,
                    flush=True,
                )
            return 1

        di3_p, di4_p = drive.read_limit_pressed(nc=di_nc, debounce_n=di_debounce_n)
        if di3_p or di4_p:
            print(
                f"FAILED: limit already pressed at start "
                f"({_di_label(home_di, plus_di, di3_p, di4_p)}) — clear and retry",
                file=sys.stderr,
                flush=True,
            )
            return 1

        err0 = target_m - host_now
        if abs(err0) <= tol_m:
            print(
                f"already at target: host={host_now * 1000:.2f} mm "
                f"(tol={tol_m * 1000:.1f} mm)",
                flush=True,
            )
            reason = "already_there"
            return 0

        v_cmd = speed_m_s if err0 > 0.0 else -speed_m_s
        eta_s = abs(err0) / speed_m_s
        print(
            f"[crawl] start host={host_now * 1000:.2f} mm → "
            f"{target_m * 1000:.1f} mm  v={v_cmd * 1000:+.1f} mm/s  "
            f"eta≈{eta_s:.0f}s  Ctrl+C or limit → stop",
            flush=True,
        )

        _set_host_velocity(drive, v_cmd, sign=sign, lead_mm=lead_mm)
        t0 = time.monotonic()
        last_print = 0.0
        try:
            while True:
                t = time.monotonic()
                host_now = _host_m(drive, sign)
                di3_p, di4_p = drive.read_limit_pressed(
                    nc=di_nc, debounce_n=di_debounce_n
                )
                if di3_p or di4_p:
                    drive.set_velocity_rpm(0, force=True)
                    reason = (
                        f"limit_stop:{_di_label(home_di, plus_di, di3_p, di4_p)}"
                    )
                    print(
                        f"\n[crawl] STOP — {reason}  host={host_now * 1000:.2f} mm",
                        flush=True,
                    )
                    break

                err = target_m - host_now
                if abs(err) <= tol_m or (err0 > 0.0 and err <= 0.0) or (
                    err0 < 0.0 and err >= 0.0
                ):
                    drive.set_velocity_rpm(0, force=True)
                    reason = "arrived"
                    print(
                        f"\n[crawl] arrived host={host_now * 1000:.2f} mm "
                        f"(target={target_m * 1000:.1f})",
                        flush=True,
                    )
                    break

                # Soft-band hard stop (should not need if target==soft_max).
                if host_now < soft_min_m - 0.002 or host_now > soft_max_m + 0.002:
                    drive.set_velocity_rpm(0, force=True)
                    reason = "soft_band"
                    print(
                        f"\n[crawl] STOP — soft band  host={host_now * 1000:.2f} mm",
                        flush=True,
                    )
                    break

                if t - last_print >= 0.5:
                    print(
                        f"  t={t - t0:6.1f}s  host={host_now * 1000:7.2f} mm  "
                        f"err={err * 1000:+7.2f} mm",
                        flush=True,
                    )
                    last_print = t

                # Keep commanding (Modbus drop recovery).
                _set_host_velocity(drive, v_cmd, sign=sign, lead_mm=lead_mm)
                elapsed = time.monotonic() - t
                time.sleep(max(0.0, poll_dt - elapsed))
        except KeyboardInterrupt:
            drive.set_velocity_rpm(0, force=True)
            reason = "keyboard_interrupt"
            host_now = _host_m(drive, sign)
            print(
                f"\n[crawl] STOP — Ctrl+C  host={host_now * 1000:.2f} mm",
                flush=True,
            )

        try:
            sync_calibration_frame(cal_path, drive, require_continuity=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[crawl] WARN: cal sync failed: {exc}", flush=True)

        print(
            f"[crawl] done reason={reason}  host={host_now * 1000:.2f} mm  "
            f"elapsed={time.monotonic() - t0:.1f}s",
            flush=True,
        )
        if reason.startswith("limit_stop"):
            return 3
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {exc}", file=sys.stderr, flush=True)
        try:
            drive.set_velocity_rpm(0, force=True)
        except Exception:
            pass
        return 1
    finally:
        try:
            drive.set_velocity_rpm(0, force=True)
        except Exception:
            pass
        if not keep_son:
            try:
                drive.disable()
            except Exception:
                pass
        try:
            drive.close()
        except Exception:
            pass
        if keep_son:
            print("[crawl] SON held (FA24=0).", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
