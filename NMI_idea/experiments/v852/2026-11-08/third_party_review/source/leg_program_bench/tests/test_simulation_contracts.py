"""Causal/prospective contracts against real AtlasLeg assets.

Sensor fault injection below only exercises packet replay and delay handling;
it is not experimental data or a replacement anatomical world.
"""
from dataclasses import asdict, fields, replace
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from leg_program_bench.atlas_scene import AtlasLeg
from leg_program_bench.simulation import FeaturePacket, LegWorld, Spec, run_scan
from leg_program_bench import run_experiment as experiment


@pytest.fixture(scope="module")
def atlas():
    return AtlasLeg.load()


def world(atlas, **changes):
    args = dict(object_id="contract_object", geometry_seed=314, u=1, v=-1,
                knee_deg=0., amplitude_scale=.3)
    args.update(changes)
    return LegWorld(atlas, Spec(**args), n=120)


def test_feature_packet_excludes_hidden_geometry_and_class(atlas):
    w = world(atlas)
    packet = w.observe(w.new_state(.4), np.random.default_rng(8))
    assert {f.name for f in fields(FeaturePacket)} == {
        "frame_id", "object_id", "structure_id", "capture_time", "release_time",
        "capture_position", "capture_rotation", "detected", "lateral_m", "depth_m", "confidence",
    }
    assert not ({"u", "v", "geometry_seed", "true_s", "true_u", "truth", "true_xi"} & asdict(packet).keys())
    assert packet.object_id == "contract_object"


def test_miss_never_discloses_nearest_truth_displacement(atlas):
    w = world(atlas)
    state = w.new_state(.4)
    state.position += np.array([10., 10., 10.])
    packet = w.observe(state, np.random.default_rng(9))
    assert not packet.detected
    assert packet.lateral_m is None and packet.depth_m is None
    assert packet.confidence == 0
    assert w.reconstructed_canonical(packet) is None


def test_same_hidden_material_world_is_preserved_across_pose_and_shape(atlas):
    a = world(atlas)
    b = world(atlas, knee_deg=60., width_scale=1.18, length_scale=.88)
    assert a.digest == b.digest
    np.testing.assert_array_equal(a.true_xi, b.true_xi)
    assert np.max(np.linalg.norm(a.true_points - b.true_points, axis=1)) > .05


def test_scan_chaining_keeps_state_clock_pose_queue_and_frame_identity(atlas, monkeypatch):
    w = world(atlas)
    state = w.new_state(.4)
    rng = np.random.default_rng(18)
    first = run_scan(w, state, .4, .415, 0, rng, max_time=2.)
    previous_time, previous_frame = state.time, state.frame_id
    previous_position = state.position.copy()
    previous_used = state.used_frames.copy()
    pending = replace(w.observe(state, rng), frame_id=99999, release_time=100.)
    state.queue.append(pending)
    state.recon_history.append(("preserved_history", previous_time))
    assert previous_time > 0 and previous_frame > 0
    assert np.linalg.norm(previous_position - w.reference_at(.4)[0]) > 1e-6
    monkeypatch.setattr(w, "new_state", lambda *a, **kw: pytest.fail("scan reset its physical state"))
    second = run_scan(w, state, .415, .43, 0, rng, max_time=2.)
    assert state.time > previous_time
    assert state.frame_id > previous_frame
    assert previous_used <= state.used_frames
    assert pending in state.queue
    assert state.recon_history == [("preserved_history", previous_time)]
    assert state.time == pytest.approx(first.duration + second.duration)
    assert not (set(first.unique_frames) & set(second.unique_frames))


def _miss_observer(w, template):
    def observe(state, rng, delay=.04, noise=.00035):
        packet = replace(template, frame_id=state.frame_id, capture_time=state.time,
                         release_time=state.time + delay, detected=False,
                         lateral_m=None, depth_m=None, confidence=0.)
        state.frame_id += 1
        return packet
    return observe


def test_duplicate_frame_is_used_once(atlas, monkeypatch):
    w = world(atlas)
    state = w.new_state(.4)
    packet = w.observe(state, np.random.default_rng(7), delay=0., noise=0.)
    assert packet.detected, "real anatomical fixture must produce a visible target"
    state.queue.extend([packet, packet])
    monkeypatch.setattr(w, "observe", _miss_observer(w, packet))
    result = run_scan(w, state, .4, .42, 0, np.random.default_rng(4), max_time=.6)
    assert sum(p.frame_id == packet.frame_id for p in result.packets) == 1
    assert result.unique_frames.count(packet.frame_id) <= 1
    assert packet.frame_id in state.used_frames


def test_late_old_frame_does_not_count_as_fresh_recheck(atlas, monkeypatch):
    w = world(atlas)
    state = w.new_state(.4)
    packet = w.observe(state, np.random.default_rng(7), delay=0., noise=0.)
    state.time = 10.
    state.queue = [replace(packet, release_time=10.)]
    monkeypatch.setattr(w, "observe", _miss_observer(w, packet))
    result = run_scan(w, state, .4, .42, 0, np.random.default_rng(4), max_time=.6,
                      fresh_after=9.)
    assert packet.frame_id not in result.unique_frames
    assert result.accepted_frames == 0
    assert not result.success
    assert not result.audit["independent_success"]


def test_last_valid_packet_expires_if_sensor_stops_releasing(atlas, monkeypatch):
    w = world(atlas)
    state = w.new_state(.4)
    packet = w.observe(state, np.random.default_rng(7), delay=0., noise=0.)
    assert packet.detected
    state.queue = [packet]
    miss = _miss_observer(w, packet)
    monkeypatch.setattr(w, "observe", lambda state, rng, **kw: miss(state, rng, delay=100.))
    result = run_scan(w, state, .4, .5, 0, np.random.default_rng(4), max_time=3.)
    assert result.status == "lost_lock", "an expired last_valid frame must not keep forward motion alive"


def test_controlled_seed_reproduces_same_world_execution(atlas):
    a, b = world(atlas), world(atlas)
    sa, sb = a.new_state(.4), b.new_state(.4)
    ra = run_scan(a, sa, .4, .42, 0, np.random.default_rng(991), max_time=2.)
    rb = run_scan(b, sb, .4, .42, 0, np.random.default_rng(991), max_time=2.)
    assert a.digest == b.digest
    assert asdict(ra) == asdict(rb)
    np.testing.assert_array_equal(sa.position, sb.position)
    np.testing.assert_array_equal(sa.rotation, sb.rotation)
    assert sa.time == sb.time


class StopAtFirstCompleteExecution(Exception):
    pass


def test_predictions_freeze_before_first_complete_trace(atlas, tmp_path, monkeypatch):
    """Instrument the protocol boundary; still execute real anatomical preparation."""
    monkeypatch.setattr(experiment.AtlasLeg, "load", lambda *a, **kw: atlas)
    monkeypatch.setattr(experiment, "generate_training", lambda *a, **kw: ([], []))
    monkeypatch.setattr(experiment, "build_bundle", lambda *a, **kw: (object(), object()))
    monkeypatch.setattr(experiment, "make_test_specs", lambda *a: [
        ("contract_test", Spec("freeze_object", 919, 1, -1, amplitude_scale=.2))])
    monkeypatch.setattr(experiment, "predict_fixed", lambda model, pairs, world, y, policy, deadlines, rng, **kw:
                        (np.zeros(len(deadlines)), np.zeros(2), np.zeros(2, dtype=bool)))
    original_write = experiment.write_json
    seen = []

    def write(path, data):
        if Path(path).name == "prediction_freeze.json":
            assert not (tmp_path / "complete_test_traces.jsonl").exists()
            assert (tmp_path / "prospective_predictions.jsonl").stat().st_size > 0
            seen.append("freeze")
        original_write(path, data)

    def execute(*a, **kw):
        assert seen == ["freeze"]
        assert (tmp_path / "prediction_freeze.json").is_file()
        raise StopAtFirstCompleteExecution

    monkeypatch.setattr(experiment, "write_json", write)
    monkeypatch.setattr(experiment, "execute_fixed", execute)
    args = SimpleNamespace(output=tmp_path, seed=123, train_objects=1, test_objects=1,
                           rho=.8, mc=2, asset_root="/media/camp/EXT_DRIVE/Among_US")
    with pytest.raises(StopAtFirstCompleteExecution):
        experiment.run(args)


def test_existing_test_outcomes_are_rejected_before_new_prospective_run(tmp_path, monkeypatch):
    previous = tmp_path / "complete_test_traces.jsonl"
    previous.write_text('{"old_test_outcome":true}\n')

    def should_not_load(*args, **kwargs):
        pytest.fail("prospective output guard must run before new data generation")

    monkeypatch.setattr(experiment.AtlasLeg, "load", should_not_load)
    args = SimpleNamespace(output=tmp_path, seed=123, train_objects=1, test_objects=1,
                           rho=.8, mc=2, asset_root="/media/camp/EXT_DRIVE/Among_US")
    with pytest.raises((FileExistsError, ValueError, RuntimeError)):
        experiment.run(args)
    assert previous.read_text() == '{"old_test_outcome":true}\n'
