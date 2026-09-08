"""Offline tests for the asynchronous phantom force trace."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

from peirastic.DEMO.phathom_scanning.force_trace import ForceTraceRecorder


class _FakeForceBus:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.stop_calls = 0

    def read(self):
        index = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[index]

    def stop(self):
        self.stop_calls += 1


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_trace_filters_hybrid_label_and_preserves_invalid_force(tmp_path: Path) -> None:
    nan_wrench = np.full(6, np.nan)
    force_t0 = time.monotonic()
    bus = _FakeForceBus(
        [
            (True, 1, force_t0 + 0.01, [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]),
            # The same sequence must never be treated as a new valid sample.
            (True, 1, force_t0 + 0.02, [9.0, 9.0, 9.0, 9.0, 9.0, 9.0]),
            # A missing force frame may use the controller's Fz only.
            (False, 2, force_t0 + 0.03, nan_wrench),
            (True, 3, force_t0 + 0.04, [4.0, 5.0, 6.0, 0.4, 0.5, 0.6]),
        ]
    )
    calls = [0]

    def read_status():
        calls[0] += 1
        n = calls[0]
        if n <= 2:
            return {"mode": 3, "msg": "other:tracking", "t_mono": time.monotonic()}
        stage = "approach" if n <= 5 else "tracking"
        stale = n == 7
        return {
            "mode": 4,
            "msg": (
                f"phathom_s_scan:{stage} t=0.1 contact={int(stage == 'tracking')} "
                "vz_cmd=+2.5mm/s"
            ),
            "t_mono": time.monotonic(),
            "f_ext_z": 8.0,
            "stale": stale,
        }

    recorder = ForceTraceRecorder(
        tmp_path,
        read_status=read_status,
        desired_force_n=4.0,
        force_bus=bus,
        sample_hz=80.0,
    )
    recorder.start()
    time.sleep(0.16)
    recorder.stop()
    paths = recorder.finish("completed")

    rows = _read_rows(Path(paths["csv"]))
    assert rows
    assert {row["stage"] for row in rows} <= {"approach", "tracking"}
    assert all(float(row["desired_fz_n"]) == pytest.approx(4.0) for row in rows)
    assert all(float(row["vz_cmd_mm_s"]) == pytest.approx(2.5) for row in rows)
    assert any(row["force_source"] == "force_bus_tool_compensated" for row in rows)

    duplicate = [row for row in rows if row["force_seq"] == "1" and row["stale"] == "1"]
    assert duplicate
    assert duplicate[0]["valid"] == "0"
    assert math.isnan(float(duplicate[0]["fx_n"]))
    assert math.isnan(float(duplicate[0]["fz_n"]))

    fallback = [row for row in rows if row["force_source"] == "controller_f_ext_z_fallback"]
    assert fallback
    assert float(fallback[0]["fz_n"]) == pytest.approx(8.0)
    assert math.isnan(float(fallback[0]["fx_n"]))
    assert math.isnan(float(fallback[0]["fy_n"]))
    assert math.isnan(float(fallback[0]["mx_nm"]))

    stale = [row for row in rows if row["force_source"] == "stale_controller"]
    assert stale
    assert stale[0]["valid"] == "0"
    assert all(math.isnan(float(stale[0][key])) for key in ("fx_n", "fy_n", "fz_n"))
    assert bus.stop_calls == 0


def test_aborted_empty_trace_is_plotted_and_finish_is_idempotent(tmp_path: Path) -> None:
    bus = _FakeForceBus([(False, 0, float("nan"), np.full(6, np.nan))])
    recorder = ForceTraceRecorder(
        tmp_path,
        read_status=lambda: {"mode": 3, "msg": "idle"},
        desired_force_n=4.0,
        force_bus=bus,
        sample_hz=100.0,
    )
    recorder.start()
    time.sleep(0.03)
    recorder.stop()
    first = recorder.finish("aborted: contact timeout")
    second = recorder.finish("completed")

    assert first == second
    assert set(first) == {"csv", "png", "summary"}
    assert all(Path(path).is_file() for path in first.values())
    assert Path(first["png"]).stat().st_size > 0
    summary = json.loads(Path(first["summary"]).read_text(encoding="utf-8"))
    assert summary["outcome"] == "aborted: contact timeout"
    assert summary["empty"] is True
    assert summary["active_samples"] == 0
    assert summary["units_frame"]["wrench_frame"] == "tool compensated"
    assert bus.stop_calls == 0


def test_sampling_error_is_exposed_and_artifacts_are_retained(tmp_path: Path) -> None:
    class BrokenBus:
        def read(self):
            raise OSError("fake force relay disappeared")

    recorder = ForceTraceRecorder(
        tmp_path,
        read_status=lambda: {
            "mode": 4,
            "msg": "phathom_s_scan:tracking t=0 contact=1 vz_cmd=0",
            "t_mono": time.monotonic(),
        },
        desired_force_n=4.0,
        force_bus=BrokenBus(),
        sample_hz=100.0,
    )
    recorder.start()
    deadline = time.monotonic() + 1.0
    while recorder.error is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert recorder.error is not None
    with pytest.raises(RuntimeError, match="force trace recorder failed"):
        recorder.check_error()
    recorder.stop()
    with pytest.raises(RuntimeError, match="force trace finalization failed"):
        recorder.finish("aborted: recorder error")
    assert (tmp_path / "force_trace.csv").is_file()
    assert (tmp_path / "force_trace.png").is_file()
    assert (tmp_path / "force_trace_summary.json").is_file()


def test_old_force_timestamp_cannot_become_a_valid_first_sample(tmp_path: Path) -> None:
    bus = _FakeForceBus(
        [(True, 77, time.monotonic() - 10.0, [1.0, 2.0, 3.0, 0.1, 0.2, 0.3])]
    )
    recorder = ForceTraceRecorder(
        tmp_path,
        read_status=lambda: {
            "mode": 4,
            "status": 1,
            "msg": "phathom_s_scan:tracking t=0 contact=1 vz_cmd=0",
            "t_mono": time.monotonic(),
        },
        desired_force_n=4.0,
        force_bus=bus,
        sample_hz=80.0,
    )
    recorder.start()
    time.sleep(0.04)
    recorder.stop()
    paths = recorder.finish("aborted: stale force")
    rows = _read_rows(Path(paths["csv"]))

    assert rows
    assert all(
        row["force_source"] in {"force_bus_stale", "force_bus_invalid", "force_bus_duplicate"}
        for row in rows
    )
    assert all(row["valid"] == "0" for row in rows)
    assert all(math.isnan(float(row["fz_n"])) for row in rows)


def test_frozen_status_timestamp_allows_only_one_fallback_sample(tmp_path: Path) -> None:
    status_t = time.monotonic()
    bus = _FakeForceBus([(False, 0, float("nan"), np.full(6, np.nan))])

    def read_status():
        return {
            "mode": 4,
            "status": 1,
            "msg": "phathom_s_scan:tracking t=0 contact=1 vz_cmd=0",
            "t_mono": status_t,
            "f_ext_z": 4.0,
        }

    recorder = ForceTraceRecorder(
        tmp_path,
        read_status=read_status,
        desired_force_n=4.0,
        force_bus=bus,
        sample_hz=100.0,
    )
    recorder.start()
    time.sleep(0.05)
    recorder.stop()
    paths = recorder.finish("aborted: frozen status")
    rows = _read_rows(Path(paths["csv"]))

    assert len(rows) >= 2
    valid = [row for row in rows if row["valid"] == "1"]
    stale = [row for row in rows if row["stale"] == "1"]
    assert len(valid) == 1
    assert valid[0]["force_source"] == "controller_f_ext_z_fallback"
    assert float(valid[0]["fz_n"]) == pytest.approx(4.0)
    assert stale
    assert all(row["valid"] == "0" for row in stale)


def test_missing_status_timestamp_is_nan_and_invalid(tmp_path: Path) -> None:
    bus = _FakeForceBus(
        [(True, 1, time.monotonic(), [0.0, 0.0, 1.0, 0.0, 0.0, 0.0])]
    )
    recorder = ForceTraceRecorder(
        tmp_path,
        read_status=lambda: {
            "mode": 4,
            "status": 1,
            "msg": "phathom_s_scan:approach t=0 contact=0 vz_cmd=1",
        },
        desired_force_n=4.0,
        force_bus=bus,
        sample_hz=100.0,
    )
    recorder.start()
    time.sleep(0.02)
    recorder.stop()
    paths = recorder.finish("aborted: stale status")
    rows = _read_rows(Path(paths["csv"]))

    assert rows
    assert all(row["status_t_mono_s"] == "nan" for row in rows)
    assert all(row["stale"] == "1" and row["valid"] == "0" for row in rows)
    assert all(math.isnan(float(row["fz_n"])) for row in rows)


def test_terminal_controller_status_stops_active_sampling(tmp_path: Path) -> None:
    force_t0 = time.monotonic()
    bus = _FakeForceBus(
        [(True, 1, force_t0, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0])]
    )
    calls = [0]

    def read_status():
        calls[0] += 1
        status = 1 if calls[0] == 1 else 5  # RUNNING, then ESTOP
        return {
            "mode": 4,
            "status": status,
            "msg": "phathom_s_scan:tracking t=0 contact=1 vz_cmd=0",
            "t_mono": time.monotonic(),
        }

    recorder = ForceTraceRecorder(
        tmp_path,
        read_status=read_status,
        desired_force_n=4.0,
        force_bus=bus,
        sample_hz=100.0,
    )
    recorder.start()
    time.sleep(0.04)
    recorder.stop()
    paths = recorder.finish("aborted: estop")
    rows = _read_rows(Path(paths["csv"]))

    assert rows
    assert len(rows) == 1
