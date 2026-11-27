#!/usr/bin/env python3
"""Terminal 9 wrapper: anatomy retarget + optional vessel/bone export.

Active bake profile (bdfd6de, fit code unchanged):
  bones-ok-feet-long-pelvis-deform-vessels-ok
  骨头至少好了，但是脚太长，盆骨变形，血管不穿
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

import numpy as np


_CANONICAL_FILES = (
    "source_manifest.json",
    "smpl_canonical_tpose.obj",
    "smpl_canonical_tpose_neutral.obj",
    "smpl_canonical_skeleton.json",
    "smpl_canonical_weights.npz",
)


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
    # Normal capture runs contain moment_0000/smplx_result.npz.  Offline
    # benchmark/initialization exports also use a single directory containing
    # smplx_result.npz directly (for example warmup_init).  Accept both forms
    # so --run can consume an actual fitted RGB initialization without copying
    # or relinking the fit into REALUS_SMPLX_OUTPUT_ROOT.
    candidates = (
        run_path if run_path.name == "smplx_result.npz" else None,
        run_path / "smplx_result.npz",
        run_path / "moment_0000" / "smplx_result.npz",
    )
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    checked = ", ".join(str(path) for path in candidates if path is not None)
    raise FileNotFoundError(f"smplx_result.npz not found for --run {run}: checked {checked}")


def _canonical_is_complete(path: Path) -> bool:
    root = Path(path).expanduser()
    return all((root / name).is_file() for name in _CANONICAL_FILES)


def _ensure_canonical_tpose(
    *,
    repo: Path,
    canonical_dir: Path,
    gender: str,
    device: str,
    betas: np.ndarray | None = None,
    source: str,
) -> bool:
    """Create the first canonical T-pose cache when a fresh setup has none.

    Window 9 normally receives betas from Window 8.  A clean migration checkout
    has no capture run yet, so use an explicit zero-beta T-pose as a bounded
    bootstrap.  The anatomy CLI still owns the quality gate; this helper only
    supplies the required canonical input files.
    """

    canonical = Path(canonical_dir).expanduser()
    if _canonical_is_complete(canonical):
        return False
    canonical.mkdir(parents=True, exist_ok=True)
    from projects.genesis_ue_sync.anatomy_retarget.canonical_export import (
        export_canonical_tpose,
    )

    del repo  # the exporter resolves models relative to the repository package
    export_canonical_tpose(
        betas=np.zeros(10, dtype=np.float32) if betas is None else np.asarray(betas, dtype=np.float32),
        output_dir=canonical,
        staging_dir=None,
        gender=str(gender),
        device=str(device),
        source=str(source),
    )
    if not _canonical_is_complete(canonical):
        missing = [name for name in _CANONICAL_FILES if not (canonical / name).is_file()]
        raise RuntimeError(f"canonical T-pose export incomplete: {missing}")
    return True


def _published_asset_path(output_dir: Path) -> Path:
    """Resolve the accepted run asset written by the anatomy CLI."""

    root = Path(output_dir).expanduser()
    pointer = root / "latest.json"
    if pointer.is_file():
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        relative = payload.get("asset")
        if relative:
            candidate = root / str(relative)
            if candidate.is_file():
                return candidate
    candidates = sorted(root.glob("*/anatomy_rigged.npz"), key=lambda item: item.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"published anatomy asset not found under {root}; latest.json was not updated")


def _write_zero_tpose_motion(*, canonical_dir: Path, output_dir: Path) -> Path:
    """Write a deterministic zero-pose EasyMocap-compatible input for vessel export."""

    from projects.genesis_ue_sync.anatomy_retarget.obj_io import read_obj_vertices

    tpose = read_obj_vertices(Path(canonical_dir) / "smpl_canonical_tpose.obj")
    path = Path(output_dir) / "zero_tpose_motion.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        Rh=np.zeros(3, dtype=np.float32),
        Th=np.zeros(3, dtype=np.float32),
        poses=np.zeros(165, dtype=np.float32),
        vertices=np.asarray(tpose, dtype=np.float32),
        source="run_anatomy_retarget:zero_tpose",
    )
    return path


def main() -> int:
    repo = Path(os.environ.get("REALUS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    os.chdir(repo)
    sys.path.insert(0, str(repo / "src"))

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=repo / "configs/anatomy/anatomy_retarget.yaml")
    ap.add_argument("--run", type=str, default="", help="Capture run whose gender/betas must match anatomy and track")
    ap.add_argument("--gender", choices=["male", "female", "neutral"], default="male")
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    default_canonical_dir = repo / "outputs/anatomy_retarget/latest_canonical"
    ap.add_argument("--canonical-dir", type=Path, default=default_canonical_dir)
    ap.add_argument("--output-dir", type=Path, default=repo / "outputs/anatomy_retarget/latest_asset")
    ap.add_argument("--publish-bind", type=str, default="tcp://127.0.0.1:5601")
    ap.add_argument("--publish-duration-s", type=float, default=5.0)
    ap.add_argument(
        "--publish-genesis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Publish only after the underlying quality gate passes (default: enabled).",
    )
    ap.add_argument("--export-vessels", action="store_true", help="Also run leg vessel centerline + thigh bone export")
    ap.add_argument(
        "--vessels-output-dir",
        type=Path,
        default=repo / "outputs/anatomy_retarget/limb_vessel_planning",
        help="Output root for --export-vessels.",
    )
    args, unknown = ap.parse_known_args()

    exact_fit: Path | None = None
    if args.run:
        exact_fit = _run_smplx_npz(repo, args.run)
        from projects.genesis_ue_sync.anatomy_retarget.canonical_export import load_betas
        from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import smplx_shape_hash

        betas = load_betas(exact_fit)
        shape_hash = smplx_shape_hash(betas, gender=args.gender)
        # Preserve the shape-keyed cache for the default invocation, while
        # allowing isolated benchmark runs to keep all canonical artifacts
        # under their requested output directory.
        canonical_cache = repo / "outputs/anatomy_retarget/canonical_cache" / shape_hash
        explicit_canonical_dir = Path(args.canonical_dir).expanduser()
        if not explicit_canonical_dir.is_absolute():
            explicit_canonical_dir = repo / explicit_canonical_dir
        if explicit_canonical_dir.resolve() != default_canonical_dir.resolve():
            canonical_cache = explicit_canonical_dir
        _ensure_canonical_tpose(
            repo=repo,
            canonical_dir=canonical_cache,
            gender=args.gender,
            device=args.device,
            betas=betas,
            source=str(exact_fit.parents[1]),
        )
        args.canonical_dir = canonical_cache
    else:
        generated = _ensure_canonical_tpose(
            repo=repo,
            canonical_dir=args.canonical_dir,
            gender=args.gender,
            device=args.device,
            source="run_anatomy_retarget:zero_betas_bootstrap",
        )
        if generated:
            print(f"INFO generated canonical T-pose -> {Path(args.canonical_dir).expanduser()}")

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
        asset = _published_asset_path(args.output_dir)
        v_argv = [
            "run_export_vessel_segments",
            "--asset-npz",
            str(asset),
            "--output-dir",
            str(args.vessels_output_dir),
            "--canonical-dir",
            str(args.canonical_dir),
        ]
        motion_fit = exact_fit or _latest_smplx_npz(repo)
        if motion_fit is None:
            motion_fit = _write_zero_tpose_motion(
                canonical_dir=args.canonical_dir,
                output_dir=Path(args.vessels_output_dir).expanduser().parent,
            )
            print(f"INFO generated zero-pose motion for vessel export -> {motion_fit}")
        v_argv.extend(["--motion-npz", str(motion_fit)])
        sys.argv = v_argv
        runpy.run_module(
            "projects.genesis_ue_sync.anatomy_retarget.cli.run_export_vessel_segments",
            run_name="__main__",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
