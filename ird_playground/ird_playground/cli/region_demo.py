"""Smoke-demo Region A aggregation on a trained point field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ird_playground.neural.model import NeuralIRD
from ird_playground.probe.se3 import mat4_from_Rt
from ird_playground.region.aggregate import region_score_a


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--num-samples", type=int, default=32)
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    ckpt = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    net = NeuralIRD.load(ckpt)

    # Identity-ish center: TCP at (0.4,0,0.3) with Z up
    R = np.eye(3)
    T_mu = mat4_from_Rt(R, np.array([0.4, 0.0, 0.3]))
    rs = region_score_a(
        net,
        T_mu=T_mu,
        T_base=np.eye(4),
        position_extent=(0.02, 0.01, 0.002),
        orientation_extent=(8.0, 5.0, 3.0),
        num_samples=args.num_samples,
    )
    out = {
        "score": rs.score,
        "mean_score": rs.mean_score,
        "softmin_score": rs.softmin_score,
        "coverage": rs.coverage,
        "min_score": rs.min_score,
        "num_samples": rs.num_samples,
    }
    report = root / "data/reports/region_a_demo.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
