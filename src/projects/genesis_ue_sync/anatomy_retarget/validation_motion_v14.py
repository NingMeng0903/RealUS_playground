"""Predeclared continuous AMASS validation clips for V14.

Different source recordings from the V13 diagnostic frames. These clips are
validation only, never imported by the rest or pose fitters. Once evaluated,
reusing them for tuning turns them into regression data, not unseen evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

from .pose_adapter import smplh156_to_smplx55
from .validation_poses_v13 import DEFAULT_AMASS_ROOT


CLIPS_V14 = {
    'sitting2': ('BioMotionLab_NTroje/rub086/0015_sitting2_poses.npz', 120, 480, 10),
    'kicking2': ('BioMotionLab_NTroje/rub086/0024_kicking2_poses.npz', 120, 480, 10),
    'walking2': ('BioMotionLab_NTroje/rub086/0006_normal_walk2_poses.npz', 0, 360, 10),
}


def freeze_validation_motion(output: str | Path, amass_root: str | Path = DEFAULT_AMASS_ROOT) -> dict:
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    manifest = dict(protocol='v14_continuous_validation_v1', used_for_fit=False,
                    pose_clipped=False, amass_shape_transferred=False, clips={})
    for name, (relative, start, stop, step) in CLIPS_V14.items():
        path = Path(amass_root) / relative
        with np.load(path, allow_pickle=False) as data:
            raw = np.asarray(data['poses'])
            if raw.ndim != 2 or raw.shape[1] != 156 or stop > len(raw):
                raise ValueError(f'{path}: predeclared clip unavailable; do not silently replace it')
            frame_ids = np.arange(start, stop, step, dtype=np.int64)
            poses = np.stack([smplh156_to_smplx55(row) for row in raw[frame_ids]]).astype(np.float32)
            recovered = np.concatenate((poses[:, :22], poses[:, 25:40], poses[:, 40:55]), axis=1)
            if not np.array_equal(recovered, raw[frame_ids].astype(np.float32).reshape(-1, 52, 3)):
                raise ValueError('AMASS rotations changed during adaptation')
            translation = np.asarray(data['trans'][frame_ids], dtype=np.float64)
            rate = float(np.asarray(data['mocap_framerate']).item())
            if not np.isfinite(poses).all() or not np.isfinite(translation).all() or rate <= 0:
                raise ValueError('invalid motion input')
        target = output / f'{name}.npz'
        np.savez_compressed(target, poses=poses, transl=translation, frame_ids=frame_ids,
                            source_fps=np.array(rate), output_fps=np.array(rate/step))
        manifest['clips'][name] = dict(source_path=str(path.resolve()),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            start=start, stop_exclusive=stop, step=step, source_fps=rate,
            frame_count=len(frame_ids), output_fps=rate/step,
            archive_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest
