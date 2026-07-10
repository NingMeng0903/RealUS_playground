#!/usr/bin/env python3
"""Bake UE T-pose body FBX from EasyMocap 10-D shapes (same-shape path).

Writes:
  outputs/ue_bake/subject_shape_tpose.npz
  outputs/ue_bake/subject_shape_tpose.fbx  (via Blender + SMPL-X addon when available)

Then point scene ue_avatar / prepare at the imported mesh, or keep body_name and
use this NPZ as motion.sequence_npz_path shape reference for retarget cache.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def _load_shapes(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        beta = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
    elif path.suffix == ".npz":
        data = np.load(path)
        key = "shapes" if "shapes" in data.files else "betas"
        beta = np.asarray(data[key], dtype=np.float32).reshape(-1)
    else:
        raise ValueError(f"Unsupported shapes file: {path}")
    if beta.size < 10:
        beta = np.pad(beta, (0, 10 - beta.size))
    return beta[:10].astype(np.float32)


def _write_tpose_npz(out_npz: Path, shapes10: np.ndarray, *, gender: str = "neutral") -> Path:
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    # BEDLAM / Blender addon: pad to 16 with zeros (EasyMocap only estimates 10).
    betas16 = np.zeros((16,), dtype=np.float32)
    betas16[:10] = shapes10
    np.savez_compressed(
        out_npz,
        betas=betas16,
        poses=np.zeros((1, 165), dtype=np.float32),
        trans=np.zeros((1, 3), dtype=np.float32),
        mocap_frame_rate=np.asarray(30, dtype=np.float32),
        model_type=np.asarray("smplx_locked_head"),
        gender=np.asarray(gender),
        source_shapes_dim=np.asarray(10),
        source="realus_easymocap",
    )
    return out_npz


def _try_blender_fbx(npz_path: Path, fbx_path: Path, blender_bin: str, addon_root: Path) -> bool:
    toolkit = Path("/media/camp/EXT_DRIVE/Among_US/ref_code_library/bedlam2_retargeting/processing/fbx_toolkit.py")
    if not toolkit.is_file():
        toolkit = Path(os.environ.get("REALUS_PROJECT_ROOT", ".")) / "ref_code_library/bedlam2_retargeting/processing/fbx_toolkit.py"
    if not Path(blender_bin).is_file() or not toolkit.is_file():
        return False
    fbx_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        blender_bin,
        "-b",
        "-P",
        str(toolkit),
        "--",
        "--smplx_animation_path",
        str(npz_path),
        "--out_fbx_path",
        str(fbx_path),
        "--tpose",
        "--anim_format",
        "SMPL-X",
    ]
    env = os.environ.copy()
    env["AMONGUS_BLENDER_BIN"] = blender_bin
    print("running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, env=env, check=False)
    return proc.returncode == 0 and fbx_path.is_file()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shapes", type=Path, required=True, help="betas.npy or smplx_result.npz")
    ap.add_argument("--gender", type=str, default="neutral")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/ue_bake"))
    ap.add_argument("--skip-fbx", action="store_true")
    args = ap.parse_args()

    repo = Path(os.environ.get("REALUS_PROJECT_ROOT", ".")).resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir
    shapes = _load_shapes(args.shapes if args.shapes.is_absolute() else repo / args.shapes)
    npz_path = _write_tpose_npz(out_dir / "subject_shape_tpose.npz", shapes, gender=str(args.gender))
    meta = {
        "shapes10": [float(v) for v in shapes.tolist()],
        "npz": str(npz_path),
        "gender": str(args.gender),
        "note": "UE bake must use this NPZ/FBX so GEN_visible_human matches EasyMocap body shape",
    }
    fbx_path = out_dir / "subject_shape_tpose.fbx"
    if not args.skip_fbx:
        blender = os.environ.get("AMONGUS_BLENDER_BIN", "/media/camp/EXT_DRIVE/blender/blender-4.5.8-linux-x64/blender")
        addon = repo / "ref_code_library/smplx_blender_addon"
        ok = _try_blender_fbx(npz_path, fbx_path, blender, addon)
        meta["fbx"] = str(fbx_path) if ok else None
        meta["fbx_ok"] = bool(ok)
        if not ok:
            print("WARNING: Blender FBX bake skipped/failed; NPZ written for shape reference", flush=True)
    (out_dir / "subject_shape_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    # Convenience copy for scene motion.sequence_npz_path
    latest = repo / "outputs/ue_bake/latest_subject_shape_tpose.npz"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_bytes(npz_path.read_bytes())
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
