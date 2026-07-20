"""Export IRD GT NPZ from a capability map (sampling from YAML)."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ird_playground.ird.export_gt import (
    IrdGtConfig,
    assert_gt_contract,
    export_ird_gt_from_capability_map,
    save_ird_gt,
)
from ird_playground.ird.capability_io import load_capability_map_dir
from ird_playground.ird.map_loader import resolve_map_dir


def load_ird_gt_config(path: Path, *, root: Path) -> tuple[Path, Path, IrdGtConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    samp = dict(raw.get("sampling") or {})
    map_dir = Path(raw.get("map_dir", ""))
    out = Path(raw.get("out", "data/ird/gt_samples.npz"))
    if not map_dir.is_absolute():
        map_dir = (root / map_dir).resolve()
    if not out.is_absolute():
        out = root / out

    n_int = int(samp.get("n_interior", 700_000))
    n_bnd = int(samp.get("n_boundary", 800_000))
    n_ext = int(samp.get("n_exterior", 500_000))

    cfg = IrdGtConfig(
        n_interior=n_int,
        n_boundary=n_bnd,
        n_exterior=n_ext,
        n_jitter=int(samp.get("n_jitter", 400_000)),
        max_orients_per_voxel=int(samp.get("max_orients_per_voxel", 28)),
        hard_negative_frac=float(samp.get("hard_negative_frac", 0.45)),
        hard_negative_radius_m=float(samp.get("hard_negative_radius_m", 0.06)),
        sigma_p_m=float(samp.get("sigma_p_m", 0.03)),
        sigma_r_deg=float(samp.get("sigma_r_deg", 10.0)),
        m_clip=float(samp.get("m_clip", 3.0)),
        m_eps=float(samp.get("m_eps", 0.05)),
        bbox_margin_m=float(samp.get("bbox_margin_m", 0.20)),
        comfort_from=str(samp.get("comfort_from", "auto")),
        k_candidates=int(samp.get("k_candidates", 4)),
        seed=int(samp.get("seed", 0)),
        orient_knn=int(samp.get("orient_knn", 7)),
        soft_tau=float(samp.get("soft_tau", 0.05)),
        unknown_soft_max=float(samp.get("unknown_soft_max", 0.25)),
        trusted_neg_soft_max=float(samp.get("trusted_neg_soft_max", 0.0)),
        min_positive_support=int(samp.get("min_positive_support", 3)),
        min_trusted_face_pairs=int(samp.get("min_trusted_face_pairs", 5000)),
    )
    return map_dir, out, cfg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/ird_gt_config.yaml"))
    ap.add_argument("--map", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    cfg_path = args.config if args.config.is_absolute() else root / args.config
    map_dir, out, cfg = load_ird_gt_config(cfg_path, root=root)
    if args.map is not None:
        map_dir = resolve_map_dir(args.map if args.map.is_absolute() else root / args.map)
    else:
        map_dir = resolve_map_dir(map_dir)
    if args.out is not None:
        out = args.out if args.out.is_absolute() else root / args.out

    cm = load_capability_map_dir(map_dir, mmap=True)
    arrays = export_ird_gt_from_capability_map(cm, cfg)
    assert_gt_contract(arrays)
    save_ird_gt(
        out,
        arrays,
        meta={
            "map_dir": str(map_dir),
            "config": str(cfg_path),
            "n_interior": cfg.n_interior,
            "n_boundary": cfg.n_boundary,
            "n_exterior": cfg.n_exterior,
            "n_jitter": cfg.n_jitter,
            "sigma_p_m": cfg.sigma_p_m,
            "m_clip": cfg.m_clip,
            "feature_dim": 6,
            "seed": cfg.seed,
            "n_total": int(arrays["features"].shape[0]),
            "contract": "MC-hit=pos; C+>=min & C-==0 trusted faces; no soft_tau fallback; natural(p,u)",
            "feature_kind": "natural_pu",
            "label_kind": "stable_support_v6",
        },
    )
    print(f"wrote {out}  N={arrays['features'].shape[0]} dim={arrays['features'].shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
