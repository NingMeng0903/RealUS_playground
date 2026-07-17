#!/usr/bin/env python3
"""Terminal 9 wrapper: anatomy retarget + optional vessel/bone export.

Active bake profile (bdfd6de, fit code unchanged):
  bones-ok-feet-long-pelvis-deform-vessels-ok
  骨头至少好了，但是脚太长，盆骨变形，血管不穿
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def _smplx_output_root(repo: Path) -> Path:
    return Path(os.environ.get("REALUS_SMPLX_OUTPUT_ROOT", repo / "smplx_outputs"))


def _latest_smplx_npz(repo: Path) -> Path | None:
    fit = _smplx_output_root(repo)
    if not fit.is_dir():
        return None
    runs = sorted([p for p in fit.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    for run in runs:
        cand = run / "moment_0000" / "smplx_result.npz"
        if cand.is_file():
            return cand
    return None


def _run_smplx_npz(repo: Path, run: str) -> Path:
    run_path = Path(run)
    if not run_path.is_absolute():
        run_path = _smplx_output_root(repo) / run_path
    candidate = run_path / "moment_0000" / "smplx_result.npz"
    if not candidate.is_file():
        raise FileNotFoundError(f"smplx_result.npz not found for --run {run}: {candidate}")
    return candidate


def main() -> int:
    repo = Path(os.environ.get("REALUS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    os.chdir(repo)
    sys.path.insert(0, str(repo / "src"))

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=repo / "configs/anatomy/anatomy_retarget.yaml")
    ap.add_argument("--run", type=str, default="", help="Capture run whose gender/betas must match anatomy and track")
    ap.add_argument("--gender", choices=["male", "female", "neutral"], default="male")
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    ap.add_argument("--canonical-dir", type=Path, default=repo / "outputs/anatomy_retarget/latest_canonical")
    ap.add_argument("--output-dir", type=Path, default=repo / "outputs/anatomy_retarget/latest_asset")
    ap.add_argument("--publish-bind", type=str, default="tcp://127.0.0.1:5601")
    ap.add_argument("--publish-duration-s", type=float, default=5.0)
    ap.add_argument("--publish-genesis", action="store_true", default=True)
    ap.add_argument("--export-vessels", action="store_true", help="Also run leg vessel centerline + thigh bone export")
    args, unknown = ap.parse_known_args()

    exact_fit: Path | None = None
    if args.run:
        exact_fit = _run_smplx_npz(repo, args.run)
        from projects.genesis_ue_sync.anatomy_retarget.canonical_export import export_canonical_tpose, load_betas
        from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import smplx_shape_hash

        betas = load_betas(exact_fit)
        shape_hash = smplx_shape_hash(betas, gender=args.gender)
        canonical_cache = repo / "outputs/anatomy_retarget/canonical_cache" / shape_hash
        manifest = canonical_cache / "source_manifest.json"
        if not manifest.is_file():
            export_canonical_tpose(
                betas=betas,
                output_dir=canonical_cache,
                staging_dir=None,
                gender=args.gender,
                device=args.device,
                source=str(exact_fit.parents[1]),
            )
        args.canonical_dir = canonical_cache

    argv = [
        "run_anatomy_retarget",
        "--config",
        str(args.config),
        "--canonical-dir",
        str(args.canonical_dir),
        "--output-dir",
        str(args.output_dir),
        "--publish-bind",
        str(args.publish_bind),
        "--publish-duration-s",
        str(args.publish_duration_s),
    ]
    if exact_fit is not None:
        argv.extend(["--motion-npz", str(exact_fit)])
    if args.publish_genesis:
        argv.append("--publish-genesis")
    argv.extend(unknown)
    sys.argv = argv
    try:
        runpy.run_module(
            "projects.genesis_ue_sync.anatomy_retarget.cli.run_anatomy_retarget",
            run_name="__main__",
        )
    except SystemExit as exc:
        code = int(exc.code or 0)
        if code != 0:
            return code

    if args.export_vessels:
        asset = Path(args.output_dir) / "anatomy_rigged.npz"
        v_argv = [
            "run_export_vessel_segments",
            "--asset-npz",
            str(asset),
            "--output-dir",
            str(repo / "outputs/anatomy_retarget/limb_vessel_planning"),
            "--canonical-dir",
            str(args.canonical_dir),
        ]
        motion_fit = exact_fit or _latest_smplx_npz(repo)
        if motion_fit is not None:
            v_argv.extend(["--motion-npz", str(motion_fit)])
        sys.argv = v_argv
        runpy.run_module(
            "projects.genesis_ue_sync.anatomy_retarget.cli.run_export_vessel_segments",
            run_name="__main__",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
