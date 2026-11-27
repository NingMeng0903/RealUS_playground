import json
from pathlib import Path

import numpy as np
import pytest

from perception.apps.build_ue_motion_benchmark import build_benchmark_log
from perception.benchmark.evaluation import evaluate_predictions


def test_static_warmup_has_unique_capture_clock_and_preserves_source(tmp_path):
    source = tmp_path / "source.jsonl"
    rows = [{"frame_index": i + 1, "sim_time_ns": (i + 1) * 10_000_000,
             "human": {"motion_frame_index": 1000 + i * 4, "motion_time_ns": 8_333_333_333 + i * 33_333_333}}
            for i in range(3)]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "benchmark.jsonl"
    summary = build_benchmark_log(source, output)
    result = [json.loads(line) for line in output.read_text().splitlines()]
    assert summary["warmup_frames"] == 15
    assert [row["frame_index"] for row in result] == list(range(18))
    assert all(row["human"]["motion_frame_index"] == 1000 for row in result[:15])
    assert result[0]["benchmark_time_ns"] == 0
    assert result[15]["benchmark_time_ns"] == 500_000_000
    assert result[15]["human"]["motion_time_ns"] == rows[0]["human"]["motion_time_ns"]
    assert result[17]["sim_time_ns"] == rows[2]["sim_time_ns"]
    assert result[0]["human"]["benchmark_phase"] == "shape_initialization"
    with pytest.raises(FileExistsError):
        build_benchmark_log(source, output)


def test_metrics_keep_global_error_and_exclude_fixed_face(tmp_path):
    names = [f"joint{i}" for i in range(55)]
    target = np.arange(165, dtype=float).reshape(55, 3) * .01
    gt = {"joint_names": names, "missing_joints": [], "bones": [
        {"joint_name": name, "world_location_cm": (point * [100, -100, 100]).tolist()}
        for name, point in zip(names, target)]}
    (tmp_path / "gt.json").write_text(json.dumps(gt))
    (tmp_path / "manifest.jsonl").write_text(json.dumps({"status": "ok", "frame_index": 0, "gt_path": "gt.json"}) + "\n")
    predicted = target + [.1, 0, 0]
    predicted[22:25] += 1000  # Fixed facial joints do not affect body/hand metrics.
    result = evaluate_predictions(tmp_path, [{"frame_index": 0, "joints_world_m": predicted,
                                             "latency_elapsed_s": .4}], method="test")
    assert result["accuracy"]["world_body22_mpjpe_mm"] == pytest.approx(100)
    assert result["accuracy"]["world_hands30_mpjpe_mm"] == pytest.approx(100)
    assert result["accuracy"]["pelvis_relative_body22_mpjpe_mm"] == pytest.approx(0, abs=1e-10)
    assert result["steady_state_latency"]["mean_throughput_hz"] == pytest.approx(2.5)
    assert result["steady_state_latency"]["p95_meets_2hz"] is True
