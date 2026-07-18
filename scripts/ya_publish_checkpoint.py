#!/usr/bin/env python3
"""Publish a two-stage checkpoint asset to Genesis without a full Blender rebake.

Builds stage1_harmonic / post_merge_bones / final from existing donor NPZs,
writes an immutable run under --output-dir, updates latest.json, and optionally
publishes on tcp://127.0.0.1:5601.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    repo = Path(os.environ.get("REALUS_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(repo / "src"))

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        choices=("stage1_harmonic", "post_merge_bones", "final"),
        required=True,
    )
    ap.add_argument("--canonical-dir", type=Path, required=True)
    ap.add_argument(
        "--hand-donor",
        type=Path,
        default=None,
        help="Legacy Stage-2 hand donor; never used as Stage-1 authority.",
    )
    ap.add_argument("--extremity-donor", type=Path, default=None, help="fe99 material-fit limbs")
    ap.add_argument("--axial-donor", type=Path, default=None, help="16fa axial compound")
    ap.add_argument(
        "--base-asset",
        type=Path,
        default=None,
        help="Schema-6 topology shell (defaults to extremity donor or axial donor)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "outputs/anatomy_retarget/checkpoint_runs",
    )
    ap.add_argument("--publish-genesis", action="store_true")
    ap.add_argument("--publish-bind", type=str, default="tcp://127.0.0.1:5601")
    ap.add_argument("--publish-duration-s", type=float, default=5.0)
    ap.add_argument("--model-id", type=str, default="patient_anatomy")
    args = ap.parse_args()

    from projects.genesis_ue_sync.anatomy_retarget.cli.run_anatomy_retarget import (
        _apply_stage1_harmonic_soft_reference,
        _file_digest,
        _merge_fast_extremity_donor,
        _publish_upsert,
    )
    from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import smplx_shape_hash
    from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import (
        load_rigged_asset,
        save_rigged_asset,
    )
    from projects.genesis_ue_sync.anatomy_retarget.shape_volume import (
        apply_material_bounded_soft_volume,
    )

    canonical_dir = args.canonical_dir.expanduser().resolve()
    manifest = json.loads((canonical_dir / "source_manifest.json").read_text(encoding="utf-8"))
    gender = str(manifest.get("gender", "male"))
    betas = [float(v) for v in manifest.get("betas", [])][:10]
    shape_hash = smplx_shape_hash(betas, gender=gender)

    base_path = args.base_asset or args.extremity_donor or args.axial_donor
    if base_path is None:
        raise SystemExit("need --base-asset or --extremity-donor")
    base_path = Path(base_path).expanduser().resolve()
    asset = load_rigged_asset(base_path, validate=True)

    report: dict = {"checkpoint": args.checkpoint, "base_asset": str(base_path)}

    if args.checkpoint == "stage1_harmonic":
        raise SystemExit(
            "Stage-1 must be built by run_anatomy_retarget.py "
            "--stage1-harmonic-only; historical donor assets are not valid "
            "for the current subject."
        )
    else:
        if args.extremity_donor is None:
            raise SystemExit(f"{args.checkpoint} requires --extremity-donor")
        if args.hand_donor is None:
            raise SystemExit(f"{args.checkpoint} requires --hand-donor")
        donor = load_rigged_asset(Path(args.extremity_donor).resolve(), validate=True)
        axial = (
            None
            if args.axial_donor is None
            else load_rigged_asset(Path(args.axial_donor).resolve(), validate=True)
        )
        asset, donor_report = _merge_fast_extremity_donor(
            asset,
            donor,
            expected_shape_hash=shape_hash,
            canonical_dir=canonical_dir,
            hand_donor_path=Path(args.hand_donor).resolve(),
            axial_donor=axial,
        )
        report["merge"] = donor_report
        if args.checkpoint == "final":
            asset, jelly = apply_material_bounded_soft_volume(
                asset, canonical_dir=canonical_dir
            )
            report["material_bounded_soft_volume"] = jelly
        else:
            report["material_bounded_soft_volume"] = {
                "skipped": True,
                "reason": "checkpoint=post_merge_bones",
            }

    meta = dict(asset.metadata or {})
    meta.update(
        {
            "gender": gender,
            "betas": betas,
            "shape_hash": shape_hash,
            "checkpoint_publish": args.checkpoint,
            "ya_checkpoint_script": True,
        }
    )
    # Drop any leftover pose cache from the schema-6 base so Genesis always
    # live-LBS the checkpoint rest mesh (cache hit would redraw the base bake).
    asset = type(asset)(
        **{
            **asset.__dict__,
            "metadata": meta,
            "pose_cache_vertices": None,
            "pose_cache_hash": "",
        }
    )

    out_root = args.output_dir.expanduser().resolve()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = out_root / args.checkpoint / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    npz_path = run_dir / "anatomy_rigged.npz"
    save_rigged_asset(npz_path, asset)
    (run_dir / "checkpoint_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    digest = _file_digest(npz_path)
    latest = {
        "schema_version": 6,
        "content_hash": digest,
        "checkpoint": args.checkpoint,
        "run": str(run_dir.relative_to(out_root)),
        "asset": str(npz_path.relative_to(out_root)),
    }
    (out_root / "latest.json").write_text(json.dumps(latest, indent=2), encoding="utf-8")
    # Also mirror into latest_asset for Genesis viewers that hardcode that path.
    latest_asset = repo / "outputs/anatomy_retarget/latest_asset"
    latest_asset.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy2(npz_path, latest_asset / "anatomy_rigged.npz")
    (latest_asset / "latest.json").write_text(
        json.dumps(
            {
                "schema_version": 6,
                "content_hash": digest,
                "checkpoint": args.checkpoint,
                "asset": "anatomy_rigged.npz",
                "source_run": str(npz_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {npz_path}")
    print(f"mirrored to {latest_asset / 'anatomy_rigged.npz'} (schema 6)")

    if args.publish_genesis:
        sent = _publish_upsert(
            bind=str(args.publish_bind),
            model_id=str(args.model_id),
            asset_npz=npz_path,
            color_rgba=(0.8, 0.05, 0.05, 0.85),
            duration_s=float(args.publish_duration_s),
            rate_hz=10.0,
        )
        print(f"published genesis sent={sent} bind={args.publish_bind}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
