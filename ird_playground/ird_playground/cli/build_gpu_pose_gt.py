"""Build collision-aware full-pose Neural IRD GT with batched GPU IK."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ird_playground.ird.export_gt import assert_gt_contract, save_ird_gt
from ird_playground.ird.gpu_pose_gt import GpuPoseGtConfig, build_gpu_pose_gt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/gpu_pose_gt_smoke.yaml"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config_path = args.config if args.config.is_absolute() else root / args.config
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    sampling = dict(raw.get("sampling") or {})
    for key in ("robot_spec", "collision_urdf", "collision_pairs"):
        if key in sampling and not Path(sampling[key]).is_absolute():
            sampling[key] = str(root / sampling[key])
    cfg = GpuPoseGtConfig(**sampling)
    arrays, meta = build_gpu_pose_gt(cfg)
    assert_gt_contract(arrays)
    out = args.out or Path(raw.get("out", "data/ird/gpu_pose_gt.npz"))
    if not out.is_absolute():
        out = root / out
    meta["config_path"] = str(config_path)
    save_ird_gt(out, arrays, meta)
    print(f"wrote {out} N={len(arrays['reachable'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
