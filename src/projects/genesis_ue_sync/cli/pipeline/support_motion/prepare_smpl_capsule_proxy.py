#!/usr/bin/env python3
"""Prepare a transparent SMPL capsule URDF for Genesis collision + placement preview."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
SRC_ROOT = next(parent for parent in (_THIS_FILE.parent, *_THIS_FILE.parents) if parent.name == "src")
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import load_amass_sequence
from projects.genesis_ue_sync.sim_platform.embodiments.smpl_capsule_runtime import prepare_smpl_capsule_runtime_asset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--amass-npz", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "outputs" / "genesis_capsule_urdf_cache")
    p.add_argument("--torch-device", type=str, default="cpu")
    p.add_argument("--rgba", type=float, nargs=4, default=(0.98, 0.48, 0.12, 0.52), metavar=("R", "G", "B", "A"))
    p.add_argument("--force-rewrite", action="store_true")
    p.add_argument("--output-json", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = args.amass_npz if args.amass_npz.is_absolute() else (REPO_ROOT / args.amass_npz)
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    seq = load_amass_sequence(npz_path)
    asset = prepare_smpl_capsule_runtime_asset(
        seq,
        cache_dir=args.cache_dir,
        device=args.torch_device,
        visual_rgba=tuple(float(x) for x in args.rgba),
        force_rewrite=bool(args.force_rewrite),
    )
    payload = {
        "amass_npz": str(npz_path),
        "runtime_urdf": str(asset.urdf_path),
        "root_link_name": asset.root_link_name,
        "shape_key": asset.proxy_geometry.shape_key,
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
