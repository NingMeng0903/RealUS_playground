"""Add collision-checked boundary stencils to full-pose GPU GT."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ird_playground.ird.export_gt import assert_gt_contract, save_ird_gt
from ird_playground.ird.gpu_boundary_stencil import (
    GpuBoundaryStencilConfig,
    build_gpu_boundary_stencils,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/gpu_boundary_stencil_smoke.yaml"))
    args = ap.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    path = args.config if args.config.is_absolute() else root / args.config
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sampling = dict(raw.get("sampling") or {})
    for key in ("base_gt_npz", "robot_spec", "collision_urdf", "collision_pairs"):
        if key in sampling and not Path(sampling[key]).is_absolute():
            sampling[key] = str(root / sampling[key])
    arrays, meta = build_gpu_boundary_stencils(GpuBoundaryStencilConfig(**sampling))
    assert_gt_contract(arrays)
    out = Path(raw.get("out", "data/ird/gpu_pose_stencils.npz"))
    if not out.is_absolute():
        out = root / out
    meta["config_path"] = str(path)
    save_ird_gt(out, arrays, meta)
    print(f"wrote {out} N={len(arrays['reachable'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
