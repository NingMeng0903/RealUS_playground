"""Publish or check the independent handheld contact features; no robot IPC."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from peirastic.contact_qp.features import FeatureConfig
from peirastic.realman8dof.modes.pad_scan_config import (
    default_pad_scan_config_path,
    load_pad_scan_config,
)


def feature_publisher_command(config_path=None, repo=None):
    root = Path(repo) if repo is not None else Path(__file__).resolve().parents[2]
    command = ["bash", str(root / "scripts/run_icra_outer.sh"), "pad-confidence"]
    if config_path is not None:
        command.extend(["--pad-contact-config", str(Path(config_path).expanduser().resolve())])
    return command


def ensure_pad_feature_publisher(
    config,
    *,
    config_path=None,
    repo=None,
    check=None,
    spawn=subprocess.Popen,
    ready_timeout_s=0.4,
    start_timeout_s=8.0,
):
    """Attach to a live 20-region publisher, or start one.

    The handheld outer is blind without this stream. Returning the child lets
    the gamepad stop a publisher it started; ``None`` means one was already up.
    """
    from peirastic.apps.contact_qp_check import wait_for_features
    probe = wait_for_features if check is None else check
    try:
        probe(config, timeout_s=ready_timeout_s, frames=1)
        return None, "already_live"
    except (RuntimeError, OSError, ValueError):
        pass
    child = spawn(feature_publisher_command(config_path, repo))
    try:
        probe(config, timeout_s=start_timeout_s, frames=1)
        return child, "started"
    except (RuntimeError, OSError, ValueError):
        return child, "started_no_frames"


def feature_identity(config):
    feature = FeatureConfig(**config["feature"]["config"])
    return dict(
        region_count=feature.region_count,
        region_edges=feature.region_edges.tolist(),
        region_layout_version=feature.region_layout_version,
        feature_endpoint=config["feature_endpoint"],
        echo_columns=feature.width,
        timestamp_semantics="effective_image_time",
    )


def publisher_arguments(config, config_path, *, input_endpoint, max_hz, max_frames):
    """The generic publisher does not read its output endpoint from YAML."""
    return [
        "--feature-config", str(config_path),
        "--output", config["feature_endpoint"],
        "--input", input_endpoint,
        "--max-hz", str(max_hz),
        "--max-frames", str(max_frames),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pad-contact-config", "--config", dest="config",
                        default=str(default_pad_scan_config_path()))
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="check fresh compatible frames, without robot IPC")
    action.add_argument("--validate-only", action="store_true", help="validate configuration without opening sockets")
    parser.add_argument("--input", default="tcp://127.0.0.1:17359")
    parser.add_argument("--max-hz", type=float, default=20.)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=8.)
    parser.add_argument("--frames", type=int, default=3)
    args = parser.parse_args(argv)
    if not 0 < args.timeout_s <= 60 or args.frames < 1 or args.max_frames < 0:
        parser.error("timeout must be in (0,60], frames positive, and max-frames nonnegative")
    try:
        config = load_pad_scan_config(args.config)
        print("[PAD-FEATURE CONFIG] " + json.dumps(feature_identity(config)), flush=True)
        if args.validate_only:
            return 0
        if args.check:
            from peirastic.apps.contact_qp_check import wait_for_features
            result = wait_for_features(config, timeout_s=args.timeout_s, frames=args.frames)
            print("[PAD-FEATURE READY] " + json.dumps(result), flush=True)
            return 0
        from peirastic.apps.contact_qp_features import main as publish
        return publish(publisher_arguments(config, args.config, input_endpoint=args.input,
                                           max_hz=args.max_hz, max_frames=args.max_frames))
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"[PAD-FEATURE FAILED] {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
