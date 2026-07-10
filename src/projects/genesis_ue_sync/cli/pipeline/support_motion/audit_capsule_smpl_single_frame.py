#!/usr/bin/env python3
"""Single-frame audit: Genesis capsule URDF DoFs vs SMPL FK joint positions (JSON to stdout).

Debug NDJSON (session ``a6d3ad``) to ``.cursor/debug-a6d3ad.log`` when ``AMONGUS_DEBUG_CAPSULE_AUDIT=1``.

Examples::

  cd Among_US && conda activate genesis && export PYTHONPATH=src AMONGUS_DEBUG_CAPSULE_AUDIT=1 && \\
  python -m projects.genesis_ue_sync.cli.pipeline.support_motion.audit_capsule_smpl_single_frame \\
    --amass-npz dataset/raw/humans/amass_hf/raw/CMU/114/114_11_poses.npz \\
    --scene-spec configs/scenes/amass_lie_sync_scene.yaml \\
    --frame-index 0 \\
    --backend cuda

  # No scene: world_offset zero (capsule root uses NPZ trans or joint0 only if --no-smpl-joint0-root)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common.project import project_paths

from projects.genesis_ue_sync.sim_platform.human_motion.validation.capsule_frame_audit import (
    run_capsule_frame_audit,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--amass-npz", type=Path, required=True)
    p.add_argument("--scene-spec", type=Path, default=None, help="Optional; enables bed fit / world_offset.")
    p.add_argument("--frame-index", type=int, default=0)
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Capsule URDF cache root (default: <repo>/outputs/genesis_capsule_urdf_cache).",
    )
    p.add_argument("--backend", type=str, default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--capsule-force-rewrite", action="store_true")
    p.add_argument("--human-name", type=str, default="patient_audit")
    p.add_argument("--no-smpl-joint0-root", action="store_true", help="Drive root translation from trans+offset only.")
    p.add_argument("--human-center-mode", type=str, default="bed_center", choices=("bed_center", "scene_anchor"))
    p.add_argument(
        "--genesis-human-proxy",
        type=str,
        default="urdf",
        choices=("urdf", "mjcf"),
        help="Genesis human asset: URDF Euler chain (default, UE-compatible) or MJCF ball joints.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo = project_paths(__file__).root
    npz = args.amass_npz if args.amass_npz.is_absolute() else repo / args.amass_npz
    if not npz.is_file():
        raise SystemExit(f"missing npz: {npz}")
    scene = None
    if args.scene_spec is not None:
        scene = args.scene_spec if args.scene_spec.is_absolute() else repo / args.scene_spec
        if not scene.is_file():
            raise SystemExit(f"missing scene spec: {scene}")
    cache_dir = args.cache_dir or (repo / "outputs" / "genesis_capsule_urdf_cache")
    report = run_capsule_frame_audit(
        npz_path=npz,
        scene_spec_path=scene,
        frame_idx=int(args.frame_index),
        cache_dir=Path(cache_dir),
        human_name=str(args.human_name),
        backend=str(args.backend),
        capsule_force_rewrite=bool(args.capsule_force_rewrite),
        human_center_mode=str(args.human_center_mode),
        use_smpl_joint0_root=not bool(args.no_smpl_joint0_root),
        genesis_proxy=str(args.genesis_human_proxy),
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))
    fc = report.get("floating_dof_layout_check")
    if isinstance(fc, dict):
        ok = bool(fc.get("packed_len_matches_genesis_n_dofs", True))
        if str(fc.get("genesis_proxy", "urdf")) == "urdf":
            ok = ok and bool(fc.get("packed_len_matches_6_plus_urdf_continuous", True))
        if not ok:
            sys.exit(2)


if __name__ == "__main__":
    main()
