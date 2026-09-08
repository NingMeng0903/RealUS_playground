import json
from pathlib import Path

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.validation_motion_v15 import (
    CLIPS_V15,
    recover_smplh156_from_smplx55,
    freeze_validation_motion_v15,
    load_frozen_validation_clip_v15,
    load_validation_clip_v15,
)


def _write_source(root: Path, spec, *, frame_count: int | None = None, nan_pose=False) -> Path:
    path = root / spec.source_relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    count = frame_count if frame_count is not None else spec.stop_frame + 4
    poses = (
        np.arange(count * 156, dtype=np.float64).reshape(count, 156) * 1.0e-5
        + spec.sid * 1.0e-4
    )
    trans = np.column_stack(
        (
            np.linspace(0.1, 0.2, count),
            np.linspace(-0.2, 0.3, count),
            np.linspace(0.8, 0.9, count),
        )
    )
    if nan_pose:
        poses[spec.start_frame, 0] = np.nan
    np.savez(
        path,
        poses=poses,
        trans=trans,
        mocap_framerate=np.asarray(spec.source_fps),
        gender=np.asarray("female"),
        # This sentinel proves the freeze path does not transfer source shape.
        betas=np.full(16, 99.0, dtype=np.float64),
    )
    return path


def _babel_record(spec):
    labels = []
    for index, segment in enumerate(spec.babel_segments):
        labels.append(
            {
                "raw_label": segment.raw_label,
                "proc_label": segment.proc_label,
                "seg_id": f"segment-{spec.sid}-{index}",
                "act_cat": [segment.proc_label],
                "start_t": segment.start_t,
                "end_t": segment.end_t,
            }
        )
    return {
        "babel_sid": spec.sid,
        "feat_p": spec.babel_feat_p,
        "frame_ann": {"labels": labels},
    }


def _write_babel(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    payload = {str(spec.sid): _babel_record(spec) for spec in CLIPS_V15}
    path = root / "train.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_v15_specs_are_exact_native_continuous_windows():
    assert len(CLIPS_V15) == 6
    assert [spec.name for spec in CLIPS_V15] == [
        "walk_sid8836",
        "turn_sid11003",
        "sitstand_sid4336",
        "reach_sid12951",
        "drink_sid3307",
        "armup_sid4010",
    ]
    for spec in CLIPS_V15:
        frame_ids = spec.frame_ids
        assert frame_ids[0] == spec.start_frame
        assert frame_ids[-1] == spec.stop_frame - 1
        assert np.array_equal(np.diff(frame_ids), np.ones(len(frame_ids) - 1))
    reach = next(spec for spec in CLIPS_V15 if spec.sid == 12951)
    assert (reach.start_frame, reach.stop_frame) == (206, 291)
    assert "knocking" in reach.selection_note


def test_smplh_adapter_and_recovery_keep_all_body_hands_root():
    source = np.arange(156, dtype=np.float64).reshape(52, 3) * 0.01
    from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import smplh156_to_smplx55

    adapted = smplh156_to_smplx55(source)
    recovered = recover_smplh156_from_smplx55(adapted)
    np.testing.assert_array_equal(recovered, source.astype(np.float32))
    np.testing.assert_array_equal(adapted[:22], source[:22].astype(np.float32))
    np.testing.assert_array_equal(adapted[25:40], source[22:37].astype(np.float32))
    np.testing.assert_array_equal(adapted[40:55], source[37:52].astype(np.float32))
    np.testing.assert_array_equal(adapted[22:25], np.zeros((3, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="face slots"):
        recover_smplh156_from_smplx55(adapted + np.pad(
            np.ones((1, 3), dtype=np.float32), ((22, 32), (0, 0))
        ))


def test_load_clip_preserves_native_frames_trans_and_read_only_arrays(tmp_path):
    source_root = tmp_path / "amass"
    spec = CLIPS_V15[0]
    _write_source(source_root, spec)
    clip = load_validation_clip_v15(spec.name, amass_root=source_root)

    assert clip.poses.shape == (spec.stop_frame - spec.start_frame, 55, 3)
    assert clip.transl.shape == (spec.stop_frame - spec.start_frame, 3)
    np.testing.assert_array_equal(clip.frame_ids, np.arange(169, 429))
    assert clip.source_fps == spec.source_fps
    assert clip.poses.flags.writeable is False
    assert clip.transl.flags.writeable is False
    assert clip.frame_ids.flags.writeable is False
    assert np.all(clip.poses[:, 22:25] == 0)
    assert np.all(np.isfinite(clip.transl))


def test_freeze_manifest_hashes_no_betas_and_write_once(tmp_path):
    source_root = tmp_path / "amass"
    babel_root = tmp_path / "babel"
    for spec in CLIPS_V15:
        _write_source(source_root, spec)
    _write_babel(babel_root)

    output = tmp_path / "frozen_v15"
    manifest = freeze_validation_motion_v15(
        output, amass_root=source_root, babel_root=babel_root
    )
    assert manifest["protocol"] == "v15_babel_amass_continuous_validation_v1"
    assert manifest["used_for_fit"] is False
    assert manifest["amass_betas_transferred"] is False
    assert manifest["native_frame_stride"] == 1
    assert set(manifest["clips"]) == {spec.name for spec in CLIPS_V15}
    for spec in CLIPS_V15:
        record = manifest["clips"][spec.name]
        assert record["sid"] == spec.sid
        assert record["used_for_fit"] is False
        assert record["pose_clipped"] is False
        assert record["amass_betas_transferred"] is False
        assert record["frame_ids"] == list(range(spec.start_frame, spec.stop_frame))
        assert len(record["source_sha256"]) == 64
        assert len(record["babel_label_sha256"]) == 64
        assert len(record["babel_record_sha256"]) == 64
        with np.load(output / record["output_file"], allow_pickle=False) as frozen:
            assert set(frozen.files) == {"poses", "transl", "frame_ids", "source_fps"}
        loaded = load_frozen_validation_clip_v15(output, spec.name)
        assert loaded.frame_count == spec.stop_frame - spec.start_frame

    with pytest.raises(FileExistsError, match="already exists"):
        freeze_validation_motion_v15(
            output, amass_root=source_root, babel_root=babel_root
        )


def test_load_rejects_nonfinite_source_and_out_of_bounds_stop(tmp_path):
    source_root = tmp_path / "amass"
    spec = CLIPS_V15[0]
    _write_source(source_root, spec, nan_pose=True)
    with pytest.raises(ValueError, match="non-finite"):
        load_validation_clip_v15(spec.name, amass_root=source_root)

    short_root = tmp_path / "short_amass"
    _write_source(short_root, spec, frame_count=spec.stop_frame - 1)
    with pytest.raises(ValueError, match="exceeds source length"):
        load_validation_clip_v15(spec.name, amass_root=short_root)
