import json

import numpy as np
import pytest

from perception.benchmark.evaluation import evaluate_predictions


def recording(tmp_path):
    names = [f"joint_{i}" for i in range(55)]
    joints = np.arange(165, dtype=float).reshape(55, 3) / 100
    gt = {"joint_names": names, "missing_joints": [], "bones": [
        {"joint_name": name, "world_location_cm": (p * [100, -100, 100]).tolist()}
        for name, p in zip(names, joints)]}
    (tmp_path / "gt.json").write_text(json.dumps(gt))
    (tmp_path / "manifest.jsonl").write_text("\n".join(json.dumps({
        "frame_index": frame, "status": "ok", "gt_path": "gt.json"}) for frame in (15, 30)))
    return joints


def test_world_translation_does_not_hide_in_relative_metric(tmp_path):
    joints = recording(tmp_path)
    result = evaluate_predictions(tmp_path, [{"frame_index": 15, "joints_world_m": joints + [.1, 0, 0],
                                               "latency_elapsed_s": .2}], method="test")
    assert result["accuracy"]["world_body22_mpjpe_mm"] == pytest.approx(100)
    assert result["accuracy"]["pelvis_relative_body22_mpjpe_mm"] == pytest.approx(0, abs=1e-9)


def test_failed_frames_count_toward_measured_throughput(tmp_path):
    joints = recording(tmp_path)
    result = evaluate_predictions(tmp_path, [
        {"frame_index": 15, "joints_world_m": joints, "latency_elapsed_s": .2},
        {"frame_index": 30, "status": "rejected", "reason": "no person", "latency_elapsed_s": .3}], method="test")
    assert result["frames_rejected"] == 1
    assert result["steady_state_latency"]["samples"] == 2
    assert result["steady_state_latency"]["successful_output_hz"] == pytest.approx(2)
    assert result["steady_state_latency"]["success_fraction"] == .5
    assert result["successful_frame_latency"]["mean_s"] == pytest.approx(.2)


def test_fast_rejections_do_not_hide_slow_successful_reconstruction(tmp_path):
    joints = recording(tmp_path)
    result = evaluate_predictions(tmp_path, [
        {"frame_index": 15, "joints_world_m": joints, "latency_elapsed_s": .6},
        {"frame_index": 30, "status": "rejected", "latency_elapsed_s": .01}], method="test")
    assert result["steady_state_latency"]["mean_s"] < .5
    assert result["successful_frame_latency"]["p95_s"] == pytest.approx(.6)
    assert result["successful_frame_latency"]["p95_meets_2hz"] is False


def test_native55_contract_rejects_proxy_skeleton(tmp_path):
    joints = recording(tmp_path)
    with pytest.raises(ValueError, match="native SMPL-X55"):
        evaluate_predictions(tmp_path, [{"frame_index": 15, "joints_world_m": joints[:25]}], method="test")


def test_missing_rejection_latency_cannot_claim_complete_throughput(tmp_path):
    joints = recording(tmp_path)
    result = evaluate_predictions(tmp_path, [
        {"frame_index": 15, "joints_world_m": joints, "latency_elapsed_s": .2},
        {"frame_index": 30, "status": "rejected", "reason": "missing pelvis"},
    ], method="test")
    latency = result["steady_state_latency"]
    assert latency["samples"] == 1
    assert latency["latency_coverage_fraction"] == .5
    assert latency["latency_complete"] is False
    assert latency["successful_output_hz"] is None
    assert latency["p95_meets_2hz"] is False


def test_exception_rejection_retains_recording_frame_and_reason(tmp_path):
    from perception.apps.evaluate_mocap_benchmark import sequence_predictions
    (tmp_path / "index.json").write_text(json.dumps({
        "index": [], "rejected_index": [{
            "source_frame_index": 137, "error": "missing pelvis", "fit_ok": False,
        }],
    }))
    predictions = sequence_predictions(tmp_path)
    assert predictions == [{
        "frame_index": 137, "status": "rejected", "reason": "missing pelvis",
        "latency_elapsed_s": None,
    }]
