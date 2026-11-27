"""Acquire fit and development histories for the corrected public reader."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

from study_worlds import (BASE_COMMIT, DEFAULT_PACK, EXPERIMENT, READER_SCANS, ROOT, SEED_BASES, SHARD, V1, V2, V3,
                          acquire, atomic_json, collect, digest, init_worlds, pack_acquired, read_json)
from material_assets import DEFAULT_ATLAS_PATH, DEFAULT_CHART_SOURCE
from scan_kernel import BONE_PRIOR


def collection_protocol(pack, counts):
    dependencies = [ROOT / f for f in ("study_worlds.py", "public_features.py", "collect_reader.py")]
    dependencies.extend([V3 / "scale_worlds.py", V3 / "fast_embedding.py", V2 / "route_rollouts.py",
                         V2 / "connected_reader.py", V2 / "public_embedding.py"])
    dependencies.extend(sorted(V1.rglob("*.py")))
    return {"experiment": EXPERIMENT, "base_commit": BASE_COMMIT, "dird": False,
            "atlas_sha256": digest(pack), "counts": counts, "shard": SHARD, "seeds": SEED_BASES,
            "input_assets": {p.name: digest(p) for p in (BONE_PRIOR, DEFAULT_CHART_SOURCE, DEFAULT_ATLAS_PATH)},
            "source_sha256": {str(p.relative_to(ROOT.parent)): digest(p) for p in dependencies},
            "fit_data": "chronological prefix, actual cross-view or revisit history, full physical scan",
            "evaluation": "matched ordinary MPC terminal tracks and paired probe/hold histories"}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "run")
    parser.add_argument("--atlas-pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--smoke", type=int, default=0)
    args = parser.parse_args()
    counts = {"smoke": args.smoke} if args.smoke else READER_SCANS
    out = args.out.resolve()
    protocol = collection_protocol(args.atlas_pack, counts)
    contract = out / "reader_acquisition.json"
    if contract.exists() and read_json(contract) != protocol:
        raise RuntimeError("Reader data collection contract changed")
    atomic_json(contract, protocol)
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=init_worlds,
                             initargs=(args.atlas_pack.resolve(),)) as pool:
        for stage, count in counts.items():
            if stage != "reader_test":
                collect(pool, out, stage, count, acquire, pack_acquired, SHARD)


if __name__ == "__main__":
    main()
