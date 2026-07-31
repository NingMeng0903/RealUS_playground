#!/usr/bin/env python3
"""Dependency-focused cold-process entry point for one V8 pose evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import smplx_pose_hash
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    ANATOMY_V8_SCHEMA_VERSION,
    POSE_EVALUATION_KIND,
    apply_subject_pose,
    load_subject_runtime,
)


def _pose(path: Path | None, zero: bool) -> tuple[np.ndarray, np.ndarray]:
    if zero:
        return np.zeros((55, 3), dtype=np.float32), np.zeros(3, dtype=np.float32)
    if path is None or not path.is_file():
        raise ValueError(f"pose input does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        key = "pose_axis_angle" if "pose_axis_angle" in data.files else "pose"
        if key not in data.files:
            raise ValueError(f"{path} must contain pose_axis_angle or pose")
        pose = np.asarray(data[key], dtype=np.float32).reshape(55, 3)
        transl = (
            np.asarray(data["transl"], dtype=np.float32).reshape(3)
            if "transl" in data.files
            else np.zeros(3, dtype=np.float32)
        )
    if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(transl)):
        raise ValueError("pose input contains non-finite values")
    return pose, transl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a V8 L1 pack without Blender or a pose cache"
    )
    parser.add_argument("--subject", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pose-file", type=Path)
    source.add_argument("--zero-pose", action="store_true")
    parser.add_argument("--translation", type=float, nargs=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    subject_path = args.subject.expanduser().resolve()
    manifest = json.loads(
        (subject_path / "manifest.json").read_text(encoding="utf-8")
    )
    subject = load_subject_runtime(subject_path, validate=False)
    # Every flat array is authenticated by its manifest SHA while loading.
    # Keep the pose path's invariant check constant-time; the full composite
    # schema/runtime digest is independently recomputed by validation/evidence.
    metadata = dict(subject.rigged_asset.metadata or {})
    if metadata.get("source_full_local_fk_v2") is not True:
        raise ValueError("pose runtime requires source_full_local_fk_v2=true")
    if (
        subject.rigged_asset.pose_cache_vertices is not None
        or str(subject.rigged_asset.pose_cache_hash)
    ):
        raise ValueError("pose runtime forbids a pose-specific vertex cache")
    pose, transl = _pose(
        None if args.pose_file is None else args.pose_file.expanduser().resolve(),
        bool(args.zero_pose),
    )
    if args.translation is not None:
        transl = np.asarray(args.translation, dtype=np.float32)
    vertices = apply_subject_pose(
        subject,
        pose_axis_angle=pose,
        transl=transl,
        validate=False,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pose_digest = smplx_pose_hash(pose, transl)
    subject_digest = str(manifest.get("runtime_digest", ""))
    if len(subject_digest) != 64 or any(
        char not in "0123456789abcdef" for char in subject_digest
    ):
        raise ValueError("subject manifest is missing its runtime SHA-256 digest")
    np.savez(
        output,
        schema_version=np.asarray(ANATOMY_V8_SCHEMA_VERSION, dtype=np.int32),
        artifact_kind=np.asarray(POSE_EVALUATION_KIND),
        subject_runtime_digest=np.asarray(subject_digest),
        pose_digest=np.asarray(pose_digest),
        pose_axis_angle=np.asarray(pose, dtype=np.float32),
        transl=np.asarray(transl, dtype=np.float32),
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(subject.rigged_asset.faces, dtype=np.int32),
    )
    print(
        f"{POSE_EVALUATION_KIND} subject={subject_digest} "
        f"pose={pose_digest} -> {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
