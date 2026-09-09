"""Offline contracts for the native contention profiler."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from apps.joint_admittance_8dof import profile_native_contention as profiler


def _seed_csv(path: Path) -> Path:
    fields = [f"q_meas_{index}" for index in range(8)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: "0.1" for field in fields})
    return path


def test_parser_exposes_offline_contention_interface(tmp_path: Path) -> None:
    args = profiler.build_parser().parse_args([
        "--seed-csv", str(tmp_path / "seed.csv"),
        "--ticks", "37",
        "--control-cpu", "6",
        "--native-cpu", "8",
        "--config", str(tmp_path / "controller.yaml"),
        "--output", str(tmp_path / "result.json"),
    ])

    assert args.ticks == 37
    assert args.control_cpu == 6
    assert args.native_cpu == 8
    assert args.background_cpus == (10, 12)
    assert args.workers == 4
    assert args.seed_csv == tmp_path / "seed.csv"
    assert args.output == tmp_path / "result.json"


def test_cpu_layout_rejects_production_control_cores(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profiler, "_physical_siblings", lambda cpu: {int(cpu)})

    with pytest.raises(ValueError, match="production control CPU"):
        profiler._validate_cpu_layout(6, 8, (2, 10), allowed_cpus=range(32))

    with pytest.raises(ValueError, match="production control CPU"):
        profiler._validate_cpu_layout(4, 8, (10, 12), allowed_cpus=range(32))


def test_cpu_layout_rejects_smt_overlap_between_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    siblings = {
        2: {2, 3}, 3: {2, 3}, 4: {4, 5}, 5: {4, 5},
        6: {6, 7}, 7: {6, 7}, 8: {8, 9}, 9: {8, 9},
        10: {10, 11}, 11: {10, 11}, 12: {12, 13}, 13: {12, 13},
    }
    monkeypatch.setattr(profiler, "_physical_siblings", lambda cpu: siblings[int(cpu)])

    with pytest.raises(ValueError, match="different physical cores"):
        profiler._validate_cpu_layout(6, 7, (10, 12), allowed_cpus=range(32))
    with pytest.raises(ValueError, match="background CPUs"):
        profiler._validate_cpu_layout(6, 8, (7, 12), allowed_cpus=range(32))


def test_case_summary_counts_timeout_and_comparison() -> None:
    same = profiler._case_summary(
        {
            "ticks": 10,
            "statuses": {"solved": 9, "native_timeout": 1},
            "first_failure": {"reason": "native_timeout"},
            "timing": {"native_roundtrip_ms": {
                "p50_ms": 2.0, "p99_ms": 22.0, "max_ms": 30.0,
            }},
        },
        mode="native_same_core",
        workers=[],
    )
    background = profiler._case_summary(
        {
            "ticks": 10,
            "statuses": {"solved": 10},
            "timing": {"native_roundtrip_ms": {
                "p50_ms": 1.0, "p99_ms": 3.0, "max_ms": 4.0,
            }},
        },
        mode="background_cpu",
        workers=[],
    )

    result = profiler._comparison({
        "native_same_core": same,
        "background_cpu": background,
    })
    assert same["timeouts"] == 1
    assert same["timeout_rate"] == pytest.approx(0.1)
    assert result["roundtrip_delta_same_minus_background_ms"] == {
        "p50_ms": 1.0, "p99_ms": 19.0, "max_ms": 26.0,
    }
    assert result["timeouts"] == {
        "native_same_core": 1, "background_cpu": 0,
    }


def test_run_benchmark_runs_both_modes_and_reaps_load(monkeypatch: pytest.MonkeyPatch,
                                                      tmp_path: Path) -> None:
    seed_csv = _seed_csv(tmp_path / "seed.csv")
    config = tmp_path / "controller.yaml"
    config.write_text("inner:\n  backend: native\n")
    calls = []
    stopped = []

    class _Stop:
        def set(self):
            pass

    monkeypatch.setattr(
        profiler,
        "_validate_cpu_layout",
        lambda control, native, background: tuple(background),
    )
    monkeypatch.setattr(
        profiler,
        "_start_burners",
        lambda cpus, **kwargs: (_Stop(), [], object(), [
            {"worker": 0, "requested_cpu": cpus[0], "applied": True},
        ]),
    )
    monkeypatch.setattr(
        profiler,
        "_stop_burners",
        lambda stop, workers, ready: stopped.append((stop, workers, ready)),
    )

    def fake_run_case(seed, **kwargs):
        calls.append((list(seed), kwargs))
        p99 = 7.0 if len(calls) == 1 else 9.0
        return {
            "ticks": kwargs["ticks"],
            "control_cpu": kwargs["control_cpu"],
            "native_cpu": kwargs["native_cpu"],
            "statuses": {"solved": kwargs["ticks"]},
            "first_failure": None,
            "timing": {"native_roundtrip_ms": {
                "p50_ms": 2.0, "p99_ms": p99, "max_ms": p99 + 1.0,
            }},
        }

    monkeypatch.setattr(profiler, "run_case", fake_run_case)
    result = profiler.run_contention_benchmark(
        seed_csv=seed_csv,
        ticks=3,
        control_cpu=6,
        native_cpu=8,
        background_cpus=(10, 12),
        config=config,
    )

    assert result["offline"] is True
    assert result["hardware_connected"] is False
    assert list(result["cases"]) == ["native_same_core", "background_cpu"]
    assert len(calls) == 2
    assert all(call[1]["control_cpu"] == 6 for call in calls)
    assert all(call[1]["native_cpu"] == 8 for call in calls)
    assert len(stopped) == 2
    assert result["comparison"]["roundtrip_delta_same_minus_background_ms"]["p99_ms"] == pytest.approx(-2.0)


def test_offline_config_rejects_shared_native_shm(tmp_path: Path) -> None:
    config = tmp_path / "controller.yaml"
    config.write_text("inner:\n  native_shm_prefix: rm75_wbc\n")

    with pytest.raises(ValueError, match="refuses configured native_shm_prefix"):
        profiler._assert_offline_config(config)
