#!/usr/bin/env python3
"""Run the offline Blender anatomy retarget step and optionally publish it to Genesis."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from common.project import project_paths
from projects.genesis_ue_sync.anatomy_retarget.asset_align import normalize_rigged_asset_file
from projects.genesis_ue_sync.anatomy_retarget.blender_retarget_runner import run_retarget
from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import load_rigged_asset
from projects.genesis_ue_sync.multiview_realtime.track_stream import (
    DEFAULT_ANATOMY_ASSET_PUB_BIND,
    TOPIC_ANATOMY_ASSET_V1,
    anatomy_asset_control_to_dict,
)


def _load_config(path: Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return dict(json.loads(text))
    try:
        import yaml  # type: ignore

        return dict(yaml.safe_load(text) or {})
    except Exception:
        return dict(json.loads(text))


def parse_args() -> argparse.Namespace:
    paths = project_paths(__file__)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=paths.configs_root / "anatomy" / "anatomy_retarget.yaml")
    p.add_argument("--canonical-dir", type=Path, default=paths.outputs_root / "anatomy_retarget" / "latest_canonical")
    p.add_argument("--output-dir", type=Path, default=paths.outputs_root / "anatomy_retarget" / "latest_asset")
    p.add_argument("--blend", type=Path, default=None)
    p.add_argument("--timeout-s", type=float, default=900.0)
    p.add_argument("--publish-genesis", action="store_true")
    p.add_argument("--publish-bind", type=str, default=DEFAULT_ANATOMY_ASSET_PUB_BIND)
    p.add_argument("--publish-duration-s", type=float, default=2.0)
    p.add_argument("--publish-rate-hz", type=float, default=10.0)
    p.add_argument("--model-id", type=str, default="patient_anatomy")
    p.add_argument("--color-rgba", type=str, default="0.8,0.05,0.05,0.85")
    return p.parse_args()


def _parse_rgba(raw: str) -> tuple[float, float, float, float]:
    vals = [float(v.strip()) for v in str(raw).split(",") if v.strip()]
    if len(vals) != 4:
        raise ValueError(f"Expected color as r,g,b,a, got {raw!r}")
    return tuple(max(0.0, min(1.0, v)) for v in vals)  # type: ignore[return-value]


def _publish_upsert(
    *,
    bind: str,
    model_id: str,
    asset_npz: Path,
    color_rgba: tuple[float, float, float, float],
    duration_s: float,
    rate_hz: float,
) -> int:
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.bind(str(bind))
    payload = anatomy_asset_control_to_dict(
        action="upsert",
        model_id=str(model_id),
        asset_npz=str(asset_npz.resolve()),
        color_rgba=color_rgba,
        timestamp_ns=time.time_ns(),
    )
    topic = TOPIC_ANATOMY_ASSET_V1.encode("utf-8")
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    end = time.time() + max(0.1, float(duration_s))
    interval = 1.0 / max(1.0, float(rate_hz))
    sent = 0
    time.sleep(0.2)
    while time.time() < end:
        sock.send_multipart([topic, body])
        sent += 1
        time.sleep(interval)
    sock.close(0)
    return int(sent)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    cfg = _load_config(args.config)
    blend = args.blend or Path(str(cfg.get("blend_path", "")))
    if not blend:
        raise ValueError("Missing anatomy blend path; pass --blend or set blend_path in config.")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_npz = out_dir / "anatomy_rigged.npz"
    output_glb = out_dir / "anatomy_rigged.glb"
    report_json = out_dir / "retarget_report.json"
    result = run_retarget(
        blend_path=blend,
        canonical_dir=args.canonical_dir,
        mapping_path=args.config,
        output_npz=output_npz,
        output_glb=output_glb,
        report_json=report_json,
        timeout_s=float(args.timeout_s),
    )
    if not result.ok:
        logging.error("Blender retarget failed returncode=%s log=%s", result.returncode, result.log_path)
        return int(result.returncode or 1)
    normalize_rigged_asset_file(output_npz, config=cfg, force=False)
    asset = load_rigged_asset(output_npz, validate=True)
    logging.info(
        "retarget ok vertices=%s faces=%s joints=%s output=%s",
        asset.vertices_rest.shape[0],
        asset.faces.shape[0],
        len(asset.joint_names),
        output_npz,
    )
    if args.publish_genesis:
        sent = _publish_upsert(
            bind=str(args.publish_bind),
            model_id=str(args.model_id),
            asset_npz=output_npz,
            color_rgba=_parse_rgba(str(args.color_rgba)),
            duration_s=float(args.publish_duration_s),
            rate_hz=float(args.publish_rate_hz),
        )
        logging.info("published anatomy upsert sent=%s bind=%s", sent, args.publish_bind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
