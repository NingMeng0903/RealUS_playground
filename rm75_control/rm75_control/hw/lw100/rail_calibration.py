"""Persist / validate LW100 rail software zero across controller restarts.

Power-cycle detection (frame origin pinned at mechanical home by FA-60):
  - Home script calls ``adopt_encoder_frame()`` at the home switch so
    ``raw_counts0 ≈ 0`` and soft band starts at ~10 mm (``raw ≈ 131k``).
  - After a drive power-cycle the monitor returns near 0 → reported host_m ≈ 0
    which is *below* soft_min → refuse start (no blind zone).
  - Do NOT reject just because |Δraw| is large while the drive stayed powered
    (carriage may have been pushed — counts0 still valid if host_m in band).

Encoder-frame bookkeeping:
  - Host writes (FA-60 / FA61 / SON) may wipe 0x1001/0x1002; ``LW100Drive`` bias
    keeps the *live* process continuous, but the JSON must always store
    ``raw_counts0`` and ``last_raw_counts`` in the *same* raw frame
    (``sync_calibration_frame``). Never update last_raw alone after a wipe.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rm75_control.hw.lw100.drive import LW100Drive
from rm75_control.hw.lw100.modbus_rtu_tcp import ModbusRtuError

DEFAULT_REL_PATH = Path("var/lw100_rail_zero.json")
VERSION = 4
ENCODER_CPR = 131_072

# Post-power-on / post-FA-60 monitor readings cluster near 0 (seen: -3, 1, …).
# ~1 mm @ 10 mm/rev — soft band starts at 10 mm so raw≈0 is never a valid pose.
BOOT_RAW_ABS = 13_107
# Minimum jump before we consider a reboot signature (noise / settle).
MIN_REBOOT_JUMP = 13_107


@dataclass
class RailCalibration:
    version: int = VERSION
    raw_counts0: int = 0
    counts0_host: int = 0
    last_raw_counts: int | None = None
    # True when home script pinned the encoder frame origin at mechanical home
    # via FA-60 adopt. Required for power-cycle detection via soft-band gate.
    frame_origin_at_home: bool = False
    sign: float = 1.0
    lead_mm: float = 10.0
    soft_min_m: float = 0.01
    soft_max_m: float = 0.78
    post_home_m: float = 0.01
    rail_m_at_cal: float = 0.01
    host: str = ""
    calibrated_unix: float = 0.0
    valid: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RailCalibration:
        last = d.get("last_raw_counts", None)
        return cls(
            version=int(d.get("version", VERSION)),
            raw_counts0=int(d.get("raw_counts0", d.get("counts0", 0))),
            counts0_host=int(d.get("counts0_host", d.get("counts0", 0))),
            last_raw_counts=(int(last) if last is not None else None),
            frame_origin_at_home=bool(d.get("frame_origin_at_home", False)),
            sign=float(d.get("sign", 1.0)),
            lead_mm=float(d.get("lead_mm", 10.0)),
            soft_min_m=float(d.get("soft_min_m", 0.01)),
            soft_max_m=float(d.get("soft_max_m", 0.78)),
            post_home_m=float(d.get("post_home_m", 0.01)),
            rail_m_at_cal=float(d.get("rail_m_at_cal", 0.01)),
            host=str(d.get("host", "")),
            calibrated_unix=float(d.get("calibrated_unix", 0.0)),
            valid=bool(d.get("valid", True)),
        )


def default_calibration_path(repo_root: Path | None = None) -> Path:
    if repo_root is not None:
        return (repo_root / DEFAULT_REL_PATH).resolve()
    here = Path(__file__).resolve()
    return (here.parents[3] / DEFAULT_REL_PATH).resolve()


def load_calibration(path: Path | str) -> RailCalibration | None:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    cal = RailCalibration.from_dict(data)
    if not cal.valid:
        return None
    return cal


def save_calibration(path: Path | str, cal: RailCalibration) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cal.valid = True
    cal.version = VERSION
    if cal.calibrated_unix <= 0.0:
        cal.calibrated_unix = time.time()
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(cal.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(p)
    return p


def sync_calibration_frame(
    path: Path | str,
    drive: LW100Drive,
    *,
    require_continuity: bool = True,
) -> RailCalibration | None:
    """Write ``raw_counts0`` and ``last_raw_counts`` in the *same* raw monitor frame.

    ``raw_counts0 = drive._counts0 - drive._counts_bias`` so the JSON stays in the
    drive's current raw frame even after host-side bias bookkeeping for FA-60/SON.

    When ``require_continuity`` is True, refuse to write (and invalidate the file)
    if the live raw looks like an encoder reboot vs the file's ``last_raw_counts``.
    That prevents laundering a power-cycle wipe into a "continuous" calibration.
    """
    cal = load_calibration(path)
    if cal is None:
        return None
    try:
        raw_now = int(drive._read_encoder_counts_raw(retries=3))
    except Exception:
        return None

    if require_continuity and cal.last_raw_counts is not None:
        last_raw = int(cal.last_raw_counts)
        if looks_like_encoder_reboot(raw_now, last_raw, lead_mm=float(cal.lead_mm)):
            invalidate_calibration(path)
            return None
        # Also refuse huge unexplained jumps even if not classic boot-cluster.
        jump = abs(raw_now - last_raw)
        if jump > travel_counts(float(cal.lead_mm), travel_m=0.85):
            invalidate_calibration(path)
            return None

    cal.raw_counts0 = int(drive._counts0) - int(drive._counts_bias)
    cal.counts0_host = int(drive._counts0)
    cal.last_raw_counts = raw_now
    try:
        save_calibration(path, cal)
    except OSError:
        return None
    return cal


def invalidate_calibration(path: Path | str) -> None:
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    except (OSError, json.JSONDecodeError):
        data = None
    if isinstance(data, dict):
        data["valid"] = False
        try:
            p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return
        except OSError:
            pass
    cal = load_calibration(p)
    if cal is None:
        # Already invalid / missing — best-effort stamp a stub.
        try:
            if p.is_file():
                stub = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(stub, dict):
                    stub["valid"] = False
                    p.write_text(
                        json.dumps(stub, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                    )
        except Exception:
            pass
        return
    cal.valid = False
    try:
        p.write_text(json.dumps(cal.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError:
        pass


def apply_calibration(drive: LW100Drive, cal: RailCalibration) -> float:
    drive.set_rail_zero_raw(int(cal.raw_counts0))
    return float(drive.read_rail_m())


def shift_calibration_raw_frame(cal: RailCalibration, delta: int) -> None:
    """Remap file counts after an FA-60 wipe (``delta = pre_raw - post_raw``).

    Soft-reset clears the monitor toward 0; host ``_counts_bias`` keeps the live
    process continuous, but the JSON must move into the new raw frame or the next
    controller start will see a false power-cycle (last≫0, now≈0).
    """
    d = int(delta)
    if d == 0:
        return
    cal.raw_counts0 = int(cal.raw_counts0) - d
    cal.counts0_host = int(cal.counts0_host) - d
    if cal.last_raw_counts is not None:
        cal.last_raw_counts = int(cal.last_raw_counts) - d


def counts_to_m(counts: int | float, lead_mm: float, cpr: int = ENCODER_CPR) -> float:
    return float(counts) / float(max(cpr, 1)) * float(lead_mm) * 1e-3


def travel_counts(lead_mm: float, travel_m: float = 0.80, cpr: int = ENCODER_CPR) -> int:
    return max(1, int(round(float(travel_m) / max(counts_to_m(1, lead_mm, cpr), 1e-12))))


def looks_like_encoder_reboot(
    raw_now: int,
    last_raw: int,
    *,
    lead_mm: float,
    boot_abs: int = BOOT_RAW_ABS,
    min_jump: int = MIN_REBOOT_JUMP,
) -> bool:
    """True only for encoder-frame reset, not for carriage motion on a live drive.

    Accept (not reboot):
      - small Δraw (controller restart, same pose)
      - large Δraw in either direction while the drive stayed powered
        (carriage pushed with controller off — ``counts0`` still valid)
      - raw≈0 after a legitimate home near the encoder origin
    Reject (reboot):
      - previous raw was far from the post-power-on cluster, current raw is back
        inside that cluster, with a non-trivial jump (classic PSU cycle)
      - |Δraw| exceeds mechanical travel (impossible without reset/wrap)
    """
    jump = abs(int(raw_now) - int(last_raw))
    if jump < int(min_jump):
        return False

    # Impossible without multi-turn loss / wrap.
    if jump > travel_counts(lead_mm, travel_m=0.85):
        return True

    now_boot = abs(int(raw_now)) < int(boot_abs)
    last_away = abs(int(last_raw)) >= int(boot_abs) * 2

    # Classic PSU cycle: large previous count → near-zero monitor again.
    # Do NOT treat "was near 0, now far" as reboot — that is a normal push after home.
    return bool(now_boot and last_away)


def _limit_pressed_by_name(name: str, di3_p: bool, di4_p: bool) -> bool:
    n = str(name).strip().lower()
    if n.startswith("di3"):
        return bool(di3_p)
    if n.startswith("di4"):
        return bool(di4_p)
    return False


class CalValidationError(Exception):
    """Calibration missing or invalidated (e.g. drive power-cycle)."""

    def __init__(
        self,
        reason: str,
        *,
        power_cycle: bool = False,
        comms: bool = False,
        frame_unknown: bool = False,
    ) -> None:
        super().__init__(reason)
        self.reason = str(reason)
        self.power_cycle = bool(power_cycle)
        self.comms = bool(comms)
        self.frame_unknown = bool(frame_unknown)


def _read_raw_with_retry(drive: LW100Drive, *, attempts: int = 3) -> int:
    """Read raw encoder with recover/reconnect; raise ModbusRtuError on total failure."""
    last_exc: Exception | None = None
    for i in range(max(1, int(attempts))):
        try:
            return int(drive._read_encoder_counts_raw(retries=3))
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            try:
                drive._client.recover()
            except Exception:
                try:
                    drive._client.reconnect()
                except Exception:
                    pass
            time.sleep(0.05 * (i + 1))
    raise ModbusRtuError(f"encoder read failed after {attempts} attempts: {last_exc}")


def validate_on_drive(
    drive: LW100Drive,
    cal: RailCalibration,
    *,
    mech_min_m: float = -0.02,
    mech_max_m: float = 0.82,
    sign: float | None = None,
    check_limit_di: bool = True,
    di_nc: bool = True,
    home_di: str = "di4",
    plus_di: str = "di3",
) -> tuple[bool, str, float, bool, bool]:
    """Validate cal on live drive.

    Returns ``(ok, reason, host_rail_m, power_cycle_suspected, comms_fail)``.
    On success, updates ``cal.last_raw_counts`` to the current raw reading.
    """
    if cal is None or not cal.valid:
        return False, "no valid calibration file", float("nan"), False, False

    # frame_origin_at_home is informational (home script still pins origin).
    # Primary power-cycle gate is |raw| ≤ BOOT_RAW_ABS below — works even after
    # a trusted mid-session wipe remapped the frame away from mechanical home.

    try:
        raw_now = _read_raw_with_retry(drive, attempts=3)
    except Exception as exc:  # noqa: BLE001
        return False, f"encoder read failed: {exc}", float("nan"), False, True

    if cal.last_raw_counts is None:
        return (
            False,
            "calibration file has no last_raw_counts — re-run home script once",
            float("nan"),
            False,
            False,
        )

    # Boot-raw gate only while frame origin is still at mechanical home
    # (|raw_counts0| small). After a trusted mid-session wipe+resync the origin
    # sits at the wipe pose — raw≈0 then means "still there", not power-cycle.
    # Soft-band below remains the general net.
    origin_at_home = abs(int(cal.raw_counts0)) <= 65_536  # 5 mm @ 10 mm/rev
    if origin_at_home and abs(int(raw_now)) <= BOOT_RAW_ABS:
        return (
            False,
            (
                f"monitor raw={raw_now} within ±{BOOT_RAW_ABS} counts (~1 mm) of "
                f"home-pinned frame origin — drive power-cycled or encoder wiped; "
                f"absolute position invalid. "
                f"Re-run apps/lw100_rail_home_limit.py --force"
            ),
            float("nan"),
            True,
            False,
        )

    last_raw = int(cal.last_raw_counts)
    if looks_like_encoder_reboot(raw_now, last_raw, lead_mm=float(cal.lead_mm)):
        jump = abs(raw_now - last_raw)
        mm = counts_to_m(jump, cal.lead_mm) * 1000.0
        return (
            False,
            (
                f"encoder reboot signature (now={raw_now}, last={last_raw}, "
                f"Δ={jump} counts ~{mm:.0f} mm) — drive power-cycled; "
                f"absolute position lost"
            ),
            float("nan"),
            True,
            False,
        )

    try:
        drive_m = apply_calibration(drive, cal)
    except Exception as exc:  # noqa: BLE001
        return False, f"failed to apply counts0: {exc}", float("nan"), False, True

    s = float(cal.sign if sign is None else sign)
    host_m = s * float(drive_m)
    if not (mech_min_m <= host_m <= mech_max_m):
        return (
            False,
            (
                f"rail_m={host_m * 1000:.1f} mm outside "
                f"[{mech_min_m * 1000:.0f}, {mech_max_m * 1000:.0f}] mm"
            ),
            host_m,
            False,
            False,
        )

    # Second gate: soft travel band (catches bad zeros even when raw ≠ 0).
    soft_lo = float(cal.soft_min_m)
    soft_hi = float(cal.soft_max_m)
    band_margin = 0.002  # 2 mm
    if host_m < soft_lo - band_margin or host_m > soft_hi + band_margin:
        return (
            False,
            (
                f"rail_m={host_m * 1000:.1f} mm outside soft band "
                f"[{soft_lo * 1000:.0f}, {soft_hi * 1000:.0f}] mm "
                f"(±{band_margin * 1000:.0f} mm) — absolute position invalid "
                f"(power-cycle / wipe / bad zero)"
            ),
            host_m,
            True,
            False,
        )

    # Limit DI vs pose consistency (catches false soft-band OK after bad zero).
    if check_limit_di:
        try:
            di3_p, di4_p = drive.read_limit_pressed(nc=bool(di_nc), debounce_n=2, settle_s=0.01)
        except Exception:
            di3_p, di4_p = False, False
        home_hit = _limit_pressed_by_name(home_di, di3_p, di4_p)
        plus_hit = _limit_pressed_by_name(plus_di, di3_p, di4_p)
        if home_hit and host_m > soft_lo + 0.025:
            return (
                False,
                (
                    f"home limit pressed but rail_m={host_m * 1000:.1f} mm "
                    f"(inconsistent zero — re-home)"
                ),
                host_m,
                True,
                False,
            )
        if plus_hit and host_m < soft_hi - 0.025:
            return (
                False,
                (
                    f"plus limit pressed but rail_m={host_m * 1000:.1f} mm "
                    f"(inconsistent zero — re-home)"
                ),
                host_m,
                True,
                False,
            )

    cal.last_raw_counts = raw_now
    return True, "ok", host_m, False, False


MISSING_CAL_MSG = (
    "[RAIL] No valid rail zero calibration.\n"
    "Run: python apps/lw100_rail_home_limit.py\n"
    "Then start the controller again."
)

POWER_CYCLE_CAL_MSG = (
    "[RAIL] Drive power-cycle detected — software zero is invalid. Refusing to start.\n"
    "Run: python apps/lw100_rail_home_limit.py --force\n"
    "The home script pins the encoder frame origin at the mechanical home switch;\n"
    "do not power-cycle the drive between homing and controller start.\n"
    "Then start the controller again."
)

COMMS_FAIL_MSG = (
    "[RAIL] Modbus read failed — check link / USR-TCP232, not calibration.\n"
    "Retry after the drive Modbus path is healthy."
)

FRAME_UNKNOWN_MSG = (
    "[RAIL] Encoder frame unknown after a host write wiped the monitor without a "
    "usable pre-read. Refusing to start.\n"
    "Run: python apps/lw100_rail_home_limit.py --force\n"
    "Then start the controller again (keep SON handoff; do not power-cycle)."
)
