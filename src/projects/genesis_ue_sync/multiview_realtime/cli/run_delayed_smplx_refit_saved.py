#!/usr/bin/env python3
"""Re-run EasyMocap SMPL-X fit + UE overlays from saved moment_* directories."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from projects.genesis_ue_sync.multiview_realtime.config import MultiviewRealtimeConfig
from projects.genesis_ue_sync.multiview_realtime.easymocap.moment_pipeline import load_fixed_betas, refit_saved_moment
from projects.genesis_ue_sync.tracking.calibration import load_calibration_bundle


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path, help="Sequence run dir containing moment_XXXX/")
    p.add_argument("--config", type=Path, default=Path("configs/tracking/multiview_realtime_dwpose_triangulation.yaml"))
    p.add_argument("--gender", type=str, default="male", choices=["male", "female", "neutral"])
    p.add_argument("--fit-model", type=str, default="smplx", choices=["smplh", "smplx"])
    p.add_argument("--thres2d", type=float, default=0.15)
    p.add_argument("--max-repro-error", type=float, default=50.0)
    p.add_argument("--mesh-alpha", type=float, default=0.82)
    p.add_argument("--mesh-rgb", type=str, default="255,128,32")
    p.add_argument("--face-stride", type=int, default=1)
    p.add_argument("--max-triangle-px", type=float, default=520.0)
    p.add_argument("--betas-path", type=Path, default=None)
    p.add_argument("--bed-sdf", action="store_true")
    p.add_argument("--scene-spec-path", type=Path, default=None)
    return p.parse_args()


def _parse_rgb(raw: str) -> tuple[int, int, int]:
    parts = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected RGB as r,g,b, got: {raw}")
    return tuple(max(0, min(255, v)) for v in parts)  # type: ignore[return-value]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"run_dir not found: {run_dir}")
    cfg = MultiviewRealtimeConfig.load(args.config)
    calibration = load_calibration_bundle(cfg.calibration_path)
    mesh_rgb = _parse_rgb(args.mesh_rgb)
    body_model_cache: dict = {}
    fixed_betas = load_fixed_betas(args.betas_path) if args.betas_path else None
    scene_spec_path = str(args.scene_spec_path) if args.scene_spec_path else cfg.scene_spec_path
    ok = 0
    for moment_dir in sorted(run_dir.glob("moment_*")):
        if not moment_dir.is_dir():
            continue
        summary = refit_saved_moment(
            moment_dir=moment_dir,
            calibration=calibration,
            camera_ids=list(cfg.camera_ids),
            gender=str(args.gender),
            fit_model=str(args.fit_model),
            thres2d=float(args.thres2d),
            max_repro_error=float(args.max_repro_error),
            mesh_alpha=float(args.mesh_alpha),
            mesh_rgb=mesh_rgb,
            face_stride=int(args.face_stride),
            max_triangle_px=float(args.max_triangle_px),
            body_model_cache=body_model_cache,
            fixed_betas=fixed_betas,
            bed_sdf=bool(args.bed_sdf),
            scene_spec_path=scene_spec_path,
            pose_backend=cfg.pose_backend,
        )
        if summary.get("fit_ok"):
            ok += 1
            logging.info("refit ok %s", moment_dir.name)
        else:
            logging.warning("refit failed %s: %s", moment_dir.name, summary.get("fit_error"))
    logging.info("refit done: %d ok under %s", ok, run_dir.resolve())
    return 0 if ok > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
