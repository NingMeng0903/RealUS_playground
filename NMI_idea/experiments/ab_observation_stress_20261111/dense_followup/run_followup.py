"""Run one arm of the frozen stress runner with an optional chunked renderer.

The runner itself is imported from the durable arm-local symlink and receives
only an arm-local ``HERE`` path.  Chunking changes allocation strategy along
the action axis; it does not change the categorical channel or planner.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_runner(arm: Path):
    # One CLI invocation runs one arm.  Keeping the arm first on sys.path makes
    # its persistent symlinks the visible source files for planner/sensor/motion.
    sys.path.insert(0, str(arm))
    for name in ("run_stress", "planner", "sensor", "motion"):
        sys.modules.pop(name, None)
    runner = importlib.import_module("run_stress")
    runner.HERE = arm
    return runner


def _chunked_channel(sensor_module, original, chunk_size: int):
    if chunk_size not in (16, 32):
        raise ValueError("renderer action chunks must be 16 or 32")

    def channel(worlds, poses, identity_blind: bool = False):
        values = np.atleast_2d(np.asarray(poses, dtype=float))
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("poses must have shape [A,3]")
        parts = []
        for start in range(0, len(values), chunk_size):
            parts.append(original(worlds, values[start:start + chunk_size], identity_blind=identity_blind))
        if len(parts) == 1:
            return parts[0]
        return sensor_module.ObservationChannel(
            joint=np.concatenate([part.joint for part in parts], axis=2),
            likelihood=np.concatenate([part.likelihood for part in parts], axis=2),
            correct_joint=np.concatenate([part.correct_joint for part in parts], axis=2),
            geometric_target_visible=np.concatenate(
                [part.geometric_target_visible for part in parts], axis=1
            ),
        )

    return channel


def _compare_channels(runner, original, chunked, arm: Path) -> dict:
    protocol = json.loads((arm / "PROTOCOL.json").read_text())
    capability = json.loads((arm / "CAPABILITY.json").read_text())
    poses = np.asarray(capability["poses"], dtype=float)
    seed = int(protocol["geometry"]["public_prior_seed"])
    count = int(protocol["geometry"]["public_prior_count"])
    worlds = runner.sensor.env.sample_worlds(seed, count)
    full = original(worlds, poses)
    chunk = chunked(worlds, poses)
    checks = {}
    for label, left, right in (
        ("joint", full.joint, chunk.joint),
        ("likelihood", full.likelihood, chunk.likelihood),
        ("correct_joint", full.correct_joint, chunk.correct_joint),
        ("geometric_target_visible", full.geometric_target_visible, chunk.geometric_target_visible),
    ):
        if left.dtype == bool or right.dtype == bool:
            difference = np.not_equal(left, right)
        else:
            difference = np.abs(left - right)
        checks[label] = {
            "shape": list(left.shape),
            "array_equal": bool(np.array_equal(left, right)),
            "allclose_1e-14": bool(np.allclose(left, right, atol=1e-14, rtol=1e-14)),
            "max_abs": float(difference.max()) if difference.size else 0.0,
        }
    result = {
        "public_prior_seed": seed,
        "public_prior_count": count,
        "pose_count": len(poses),
        "chunk_size": int(getattr(chunked, "chunk_size", -1)),
        "checks": checks,
        "all_array_equal": all(row["array_equal"] for row in checks.values()),
        "allclose_1e-14": all(row["allclose_1e-14"] for row in checks.values()),
    }
    (arm / "CHANNEL_VERIFY.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run(args: argparse.Namespace) -> None:
    arm = (HERE / args.arm).resolve()
    runner = _load_runner(arm)
    original = runner.sensor.channel
    chunked = _chunked_channel(runner.sensor, original, args.chunk_size)
    # Expose the setting for the post-run manifest without touching frozen
    # sensor.py.  _run uses this module-global sensor object.
    chunked.chunk_size = args.chunk_size
    runner.sensor.channel = chunked
    if args.verify_channel:
        verification = _compare_channels(runner, original, chunked, arm)
        print(json.dumps(verification, sort_keys=True), flush=True)
        if not verification["all_array_equal"]:
            raise RuntimeError("chunked renderer differs from original channel")
    output = Path(args.output).resolve()
    capability = arm / "CAPABILITY.json"
    with threadpool_limits(limits=1, user_api="blas"):
        runner._run(args.cohort, output, "prodmp", capability)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.setdefault("followup", {}).update({
        "arm": args.arm,
        "renderer_action_chunk_size": args.chunk_size,
        "renderer_chunking_preserves_channel": True,
        "wrapper_source": str(Path(__file__)),
        "wrapper_sha256": _sha(Path(__file__)),
        "channel_verification": str(arm / "CHANNEL_VERIFY.json") if args.verify_channel else None,
    })
    manifest.setdefault("source_sha256", {})[str(Path(__file__))] = _sha(Path(__file__))
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(output, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("coarse", "dense"), required=True)
    parser.add_argument("--cohort", choices=("dev", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, choices=(16, 32), default=16)
    parser.add_argument("--verify-channel", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
