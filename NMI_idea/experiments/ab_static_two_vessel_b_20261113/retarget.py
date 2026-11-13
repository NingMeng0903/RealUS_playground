"""Recreate the local artery/vein export using the Among_US retarget pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("among-us", "blender", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--blend", type=Path, help="override the source asset in the Among_US configuration")
    args = parser.parse_args()
    root = args.among_us.resolve()
    sys.path.insert(0, str(root / "src"))
    from projects.genesis_ue_sync.anatomy_retarget.asset_align import normalize_rigged_asset_file
    from projects.genesis_ue_sync.anatomy_retarget.blender_retarget_runner import run_retarget

    config_path = root / "configs/anatomy/anatomy_retarget.yaml"
    config = yaml.safe_load(config_path.read_text())
    blend = args.blend or Path(config["blend_path"])
    canonical = root / "outputs/anatomy_retarget/recovered_smplx_20261108"
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ["AMONGUS_BLENDER_BIN"] = str(args.blender.resolve())
    mapping = out / "retarget.yaml"
    mapping.write_text(yaml.safe_dump({**config, "include_collections": [], "include_meshes": ["Artery", "Vein"]}))
    asset = out / "vessels_registered.npz"
    result = run_retarget(blend_path=blend, canonical_dir=canonical, mapping_path=mapping,
                          output_npz=asset, output_glb=out / "vessels.glb",
                          report_json=out / "RETARGET.json", timeout_s=120.)
    if not result.ok:
        raise RuntimeError(f"Among_US retarget exited {result.returncode}. See {result.log_path}")
    data = normalize_rigged_asset_file(asset, config=config)
    assert list(data.source_mesh_names) == ["Artery", "Vein"]
    named = {
        "rigged_source_blend": blend,
        "canonical_skeleton": canonical / "smpl_canonical_skeleton.json",
        "canonical_weights": canonical / "smpl_canonical_weights.npz",
        "mapping": config_path,
        "retarget_code": root / "src/projects/genesis_ue_sync/anatomy_retarget/blender_scripts/blender_retarget_script.py",
        "local_registered_export": asset,
    }
    provenance = {name: {"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                  for name, path in named.items()}
    provenance["vertex_count"] = int(len(data.vertices_rest))
    provenance["face_count"] = int(len(data.faces))
    (out / "ASSET_PROVENANCE.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
