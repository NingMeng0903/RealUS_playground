#!/usr/bin/env python3
"""Independent LW100 rail limit homing (software zero). Run after drive power-on.

Does NOT start the 8-DOF controller. Writes ``var/lw100_rail_zero.json`` so the
controller can reuse the zero until the drive is power-cycled.

By default leaves SON enabled (FA24=0 hold) so the controller inherits the same
encoder frame without a disable→enable edge that can wipe multi-turn counts.

  cd rm75_control && source env.sh
  python apps/lw100_rail_home_limit.py --force
  # then start the controller while SON stays on

  python apps/lw100_rail_home_limit.py --force --release   # release SON at end
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from rm75_control.hw.lw100.drive import LW100Drive, LW100DriveConfig
from rm75_control.hw.lw100.rail_calibration import (
    default_calibration_path,
    load_calibration,
    sync_calibration_frame,
)
from rm75_control.hw.lw100.rail_home_limit import RailHomeConfig, home_and_save


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
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument(
        "--release",
        action="store_true",
        help="release SON at end (default: keep SON on for controller handoff)",
    )
    return p.parse_args()


def _home_cfg_from_yaml(raw: dict, args: argparse.Namespace) -> tuple[RailHomeConfig, dict]:
    hw = raw.get("hw", {}).get("lw100", {}) or {}
    host = str(args.host or hw.get("host", "192.168.0.7"))
    port = int(args.port or hw.get("port", 8234))
    slave = int(hw.get("slave", hw.get("slave_id", 1)))
    cfg = RailHomeConfig(
        sign=float(hw.get("sign", -1.0)),
        lead_mm=float(hw.get("lead_mm", 10.0)),
        home_di=str(hw.get("home_di", "di4")),
        plus_di=str(hw.get("plus_di", "di3")),
        di_nc=bool(hw.get("di_nc", True)),
        di_debounce_n=int(hw.get("di_debounce_n", 3)),
        home_search_m_s=float(hw.get("home_search_m_s", 0.020)),
        home_creep_m_s=float(hw.get("home_creep_m_s", 0.003)),
        home_backoff_mm=float(hw.get("home_backoff_mm", 5.0)),
        home_touch_count=int(hw.get("home_touch_count", 3)),
        home_search_timeout_s=float(hw.get("home_search_timeout_s", 60.0)),
        home_to_post_m_s=float(hw.get("home_to_post_m_s", 0.030)),
        post_home_m=float(hw.get("post_home_m", 0.01)),
        soft_min_m=float(hw.get("soft_min_m", 0.01)),
        soft_max_m=float(hw.get("soft_max_m", 0.78)),
        max_speed_rpm=int(hw.get("max_speed_rpm", 900)),
        accel_ms=int(hw.get("accel_ms", 200)),
        decel_ms=int(hw.get("decel_ms", 200)),
        scurve_ms=int(hw.get("scurve_ms", 30)),
        host=host,
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
        "verbose": bool(hw.get("verbose", False)),
    }
    return cfg, drive_kw


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        print(f"config not found: {cfg_path}", file=sys.stderr)
        return 2
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    home_cfg, drive_kw = _home_cfg_from_yaml(raw, args)
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
    if existing is not None and not args.force:
        print(
            f"[home] calibration exists: {out} "
            f"(counts0={existing.raw_counts0}, last_raw={existing.last_raw_counts}). "
            f"Use --force to re-home.",
            flush=True,
        )
        return 0

    keep_son = not bool(args.release)
    print(f"[home] {drive_kw['host']}:{drive_kw['port']} → limit home → {out}", flush=True)
    drive_cfg = LW100DriveConfig(**drive_kw)
    drive = LW100Drive(drive_cfg)
    drive._disable_on_exit = not keep_son  # noqa: SLF001 — keep SON → skip disable
    try:
        drive.connect()
        drive.start_velocity_session(
            accel_ms=int(hw.get("accel_ms", 200)),
            decel_ms=int(hw.get("decel_ms", 200)),
            scurve_ms=int(hw.get("scurve_ms", 30)),
            max_speed_rpm=int(home_cfg.max_speed_rpm),
        )
        if not drive.frame_trusted:
            print(
                "[home] FAILED: encoder frame untrusted during session setup — "
                "check Modbus, then retry",
                file=sys.stderr,
                flush=True,
            )
            return 1
        try:
            home_and_save(drive, home_cfg, out)
            # Pair counts0 + last_raw from live drive (same raw frame).
            # Fresh home is continuous by construction — still require pairing.
            sync_calibration_frame(out, drive, require_continuity=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[home] FAILED: {exc}", file=sys.stderr, flush=True)
            try:
                drive.set_velocity_rpm(0, force=True)
            except Exception:
                pass
            if keep_son:
                # Failure: still release so the axis is not left energized silently.
                try:
                    drive.disable()
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
    finally:
        try:
            drive.close()
        except Exception:
            pass

    if keep_son:
        print(
            "[home] OK — SON held (FA24=0). Start the controller now without "
            "power-cycling the drive.\n"
            "  (Use --release next time if you need the motor de-energized.)",
            flush=True,
        )
    else:
        print("[home] OK — SON released. Start the controller when ready.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
