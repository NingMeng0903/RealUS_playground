#!/usr/bin/env python3
"""Publish placed GT AMASS poses on the live track ZMQ topic for coordinate-chain tests."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from common.project import project_paths
from projects.genesis_ue_sync.multiview_realtime.track_stream import (
    DEFAULT_TRACK_PUB_BIND,
    TOPIC_MULTIVIEW_TRACK_V1,
    track_keypoints3d_to_dict,
)
from projects.genesis_ue_sync.sim_platform.datasets import HumanMotionSequence, evaluate_smpl_sequence, load_amass_sequence
from projects.genesis_ue_sync.sim_platform.human_refit.placement_resolver import (
    resolve_or_compute_placement_for_amass,
)
from projects.genesis_ue_sync.sim_platform.scenes import load_sync_scene_spec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--amass-npz", type=Path, required=True)
    p.add_argument("--scene-spec", type=Path, default=Path("configs/scenes/amass_lie_sync_scene.yaml"))
    p.add_argument("--pub-bind", type=str, default=DEFAULT_TRACK_PUB_BIND)
    p.add_argument("--frame-start", type=int, default=0)
    p.add_argument("--frame-step", type=int, default=4)
    p.add_argument("--frame-limit", type=int, default=0)
    p.add_argument("--playback-fps", type=float, default=30.0)
    p.add_argument("--loop", action="store_true")
    p.add_argument("--fit-samples", type=int, default=11)
    p.add_argument("--force-placement", action="store_true")
    p.add_argument("--log-every", type=int, default=30)
    return p.parse_args()


def _resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return project_paths(__file__).root / path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    repo = project_paths(__file__).root
    npz_path = _resolve_repo_path(args.amass_npz)
    scene_path = _resolve_repo_path(args.scene_spec)

    scene_spec = load_sync_scene_spec(scene_path)
    seq = load_amass_sequence(npz_path)
    placement, placement_path = resolve_or_compute_placement_for_amass(
        scene_spec,
        seq,
        amass_npz_path=npz_path,
        repo_root=repo,
        placement_sample_frames=int(args.fit_samples),
        device="cpu",
        force_recompute=bool(args.force_placement),
        fit_samples=int(args.fit_samples),
    )
    world_off = np.asarray(placement.world_offset_m, dtype=np.float32).reshape(3)
    trans = np.asarray(seq.trans[:, :3], dtype=np.float32) + world_off.reshape(1, 3)
    poses = np.asarray(seq.poses, dtype=np.float32)
    betas = np.asarray(seq.betas, dtype=np.float32)

    start = max(0, int(args.frame_start))
    step = max(1, int(args.frame_step))
    indices = list(range(start, int(seq.frame_count), step))
    if int(args.frame_limit) > 0:
        indices = indices[: int(args.frame_limit)]
    if not indices:
        raise RuntimeError("No frames selected.")
    local_seq = HumanMotionSequence(
        source_dataset=seq.source_dataset,
        sequence_name=f"{seq.sequence_name}_zero_trans_for_bypass",
        source_path=seq.source_path,
        model_type=seq.model_type,
        fps=seq.fps,
        gender=seq.gender,
        betas=betas,
        poses=poses[np.asarray(indices, dtype=np.int64)].copy(),
        trans=np.zeros((len(indices), 3), dtype=np.float32),
        metadata=dict(seq.metadata),
    )
    _unused_vertices, local_joints = evaluate_smpl_sequence(
        local_seq,
        device="cpu",
        include_vertices=False,
        include_joints=True,
    )
    if local_joints is None:
        raise RuntimeError("Failed to evaluate local SMPL joints for GT bypass.")
    # World SMPL joints = local (zero-trans) joints + placed world translation.
    world_joints_by_frame: dict[int, np.ndarray] = {}
    for pos, frame_idx in enumerate(indices):
        joints_local = np.asarray(local_joints[pos, :, :3], dtype=np.float32)
        joints_world = joints_local + trans[int(frame_idx)].reshape(1, 3)
        kp = np.concatenate([joints_world, np.ones((joints_world.shape[0], 1), dtype=np.float32)], axis=1)
        world_joints_by_frame[int(frame_idx)] = kp

    try:
        import zmq
    except ImportError as exc:
        raise ImportError("pyzmq required for GT bypass publisher.") from exc

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.bind(str(args.pub_bind))
    topic = TOPIC_MULTIVIEW_TRACK_V1.encode("utf-8")
    dt = 1.0 / max(float(args.playback_fps), 1.0e-6)
    logging.info("GT bypass publisher bind=%s topic=%s", args.pub_bind, TOPIC_MULTIVIEW_TRACK_V1)
    logging.info(
        "GT placement %s world_offset_m=%s frames=%s step=%s fps=%.3f",
        placement_path,
        [round(float(v), 4) for v in world_off.tolist()],
        len(indices),
        step,
        float(args.playback_fps),
    )
    time.sleep(0.25)

    sent = 0
    try:
        while True:
            for frame_idx in indices:
                kp = world_joints_by_frame[int(frame_idx)]
                pelvis = kp[0, :3]
                payload = track_keypoints3d_to_dict(
                    frame_index=int(frame_idx),
                    timestamp_ns=time.time_ns(),
                    keypoints3d=kp,
                    schema="smpl",
                    translation_m=pelvis,
                )
                sock.send_multipart([topic, json.dumps(payload, ensure_ascii=True).encode("utf-8")])
                sent += 1
                if int(args.log_every) > 0 and sent % int(args.log_every) == 0:
                    logging.info(
                        "published GT frame=%s pelvis_m=%s",
                        int(frame_idx),
                        [round(float(v), 3) for v in pelvis.tolist()],
                    )
                time.sleep(dt)
            if not bool(args.loop):
                break
    except KeyboardInterrupt:
        logging.info("GT bypass publisher stopped")
    finally:
        sock.close(linger=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
