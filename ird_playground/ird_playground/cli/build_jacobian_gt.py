"""CLI: dump chart-frame Jacobian GT for the Phase-5 capacity head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from ird_playground.ird.jacobian_gt import (
    JacobianGtConfig,
    build_jacobian_gt,
    save_jacobian_gt,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML with a top-level 'build' mapping (JacobianGtConfig fields).",
    )
    ap.add_argument(
        "--source",
        type=Path,
        default=Path("data/ird/gpu_pose_production.npz"),
        help="Pose GT NPZ (reachable / collision-free subset used).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/ird/jacobian_gt_production.npz"),
        help="Output NPZ path for chart-frame Jacobian generators.",
    )
    ap.add_argument("--robot-spec", type=Path, default=Path("configs/robot_probe45.yaml"))
    ap.add_argument("--max-samples", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    fields: dict = {
        "source_npz": str(args.source),
        "output_npz": str(args.out),
        "robot_spec": str(args.robot_spec),
        "max_samples": int(args.max_samples),
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "device": str(args.device),
    }
    if args.config is not None:
        path = args.config if args.config.is_absolute() else root / args.config
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        build = dict(raw.get("build") or raw)
        fields.update(build)

    allowed = {f.name for f in JacobianGtConfig.__dataclass_fields__.values()}
    fields = {k: v for k, v in fields.items() if k in allowed}
    for key in ("source_npz", "output_npz", "robot_spec"):
        if key in fields and fields[key] is not None and not Path(str(fields[key])).is_absolute():
            fields[key] = str(root / fields[key])

    cfg = JacobianGtConfig(**fields)
    arrays, meta = build_jacobian_gt(cfg, root=root)
    out = save_jacobian_gt(cfg.output_npz, arrays, meta)
    print(json.dumps({"wrote": str(out), "n": meta["n"], "frame": meta["frame"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
