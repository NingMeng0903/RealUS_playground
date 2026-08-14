"""Offline verdict for the isolated rail sine (no hardware)."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

_PATH = (
    Path(__file__).resolve().parents[1] / "apps" / "lw100_isolated_sine_track.py"
)
_SPEC = importlib.util.spec_from_file_location("lw100_isolated_sine_track", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
A_CMD_REV_GATE = _MOD.A_CMD_REV_GATE
analyze_rows = _MOD.analyze_rows
cinf_sine_m = _MOD.cinf_sine_m


def test_cinf_sine_starts_at_center_and_stays_in_band() -> None:
    xs = [cinf_sine_m(t, center_m=0.275, amp_m=0.15, omega=0.10 / 0.15) for t in
          [i * 0.02 for i in range(400)]]
    assert xs[0] == 0.275
    assert min(xs) >= 0.12
    assert max(xs) <= 0.43


def test_analyze_quiet_follow_is_quiet() -> None:
    rows = []
    dt = 0.02
    for i in range(200):
        t = i * dt
        x = 0.275 + 0.05 * math.sin(2.0 * math.pi * 0.2 * t)
        rows.append(
            {
                "t_s": str(t),
                "follow": "1",
                "a_cmd_m_s2": str(0.04 * math.cos(2.0 * math.pi * 0.2 * t)),
                "v_cmd_m_s": str(0.02 * math.cos(2.0 * math.pi * 0.2 * t)),
                "e_track_mm": "0.3",
            }
        )
    m = analyze_rows(rows)
    assert m["verdict"] == "QUIET"
    assert m["a_cmd_rev_per_s"] < A_CMD_REV_GATE


def test_analyze_chattering_a_cmd_is_fighting() -> None:
    rows = []
    dt = 0.02
    for i in range(250):
        t = i * dt
        sign = 1.0 if (i % 2 == 0) else -1.0
        rows.append(
            {
                "t_wall_s": str(t),
                "follow": "1",
                "a_cmd_m_s2": str(0.4 * sign),
                "v_cmd_m_s": str(0.02 * sign),
                "e_track_mm": "0.4",
            }
        )
    m = analyze_rows(rows)
    assert m["fighting"] is True
    assert m["verdict"] == "FIGHTING"
    assert m["a_cmd_rev_per_s"] >= A_CMD_REV_GATE


def test_analyze_loose_track_without_reversals() -> None:
    rows = []
    dt = 0.02
    for i in range(200):
        t = i * dt
        rows.append(
            {
                "t_s": str(t),
                "follow": "1",
                "a_cmd_m_s2": "0.02",
                "v_cmd_m_s": "0.02",
                "e_track_mm": "5.0",
            }
        )
    m = analyze_rows(rows)
    assert m["verdict"] == "TRACK_LOOSE"
    assert m["fighting"] is False


def test_profiles_stay_inside_isolated_band() -> None:
    for _name, prof in _MOD.PROFILES.items():
        amp = float(prof["amp_m"])
        assert _MOD.CENTER_M - amp >= _MOD.BAND_LO_M - 1e-9
        assert _MOD.CENTER_M + amp <= _MOD.BAND_HI_M + 1e-9
