#!/usr/bin/env python3
"""Export a subject-beta SMPL-X canonical T-pose bundle for anatomy retargeting."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from common.project import project_paths
from projects.genesis_ue_sync.anatomy_retarget.canonical_export import export_canonical_tpose, load_betas


def parse_args() -> argparse.Namespace:
    paths = project_paths(__file__)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--betas", type=Path, required=True, help="betas.npy, smplx_result.npz, or a terminal-8 run directory.")
    p.add_argument("--output-dir", type=Path, default=paths.outputs_root / "anatomy_retarget" / "canonical")
    p.add_argument("--staging-dir", type=Path, default=paths.outputs_root / "anatomy_retarget" / "latest_canonical")
    p.add_argument("--gender", type=str, default="male", choices=["male", "female", "neutral"])
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    p.add_argument("--no-staging", action="store_true")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    betas = load_betas(args.betas)
    result = export_canonical_tpose(
        betas=betas,
        output_dir=args.output_dir,
        staging_dir=None if args.no_staging else args.staging_dir,
        gender=str(args.gender),
        device=str(args.device),
        source=str(args.betas),
    )
    logging.info("canonical T-pose exported -> %s", result.output_dir)
    if not args.no_staging:
        logging.info("canonical staging updated -> %s", args.staging_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
