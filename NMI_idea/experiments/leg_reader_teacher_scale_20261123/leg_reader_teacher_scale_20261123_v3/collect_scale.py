"""Collect complete reader trajectories in committed atlas-only shards."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

from scale_worlds import (DEFAULT_PACK, EXPERIMENT, ROOT, SEED_RANGES, V1, V2, atomic_json, collect,
                          collect_reader_case, digest, init_worlds, pack_tracks, read_json)
from material_assets import DEFAULT_ATLAS_PATH, DEFAULT_CHART_SOURCE
from scan_kernel import BONE_PRIOR


READER_COUNTS = {"reader_fit": 12288, "reader_calibration": 512, "reader_test": 1024}
READER_SHARD = 128


def contract(pack, counts):
    files = [ROOT / name for name in ("collect_scale.py", "scale_worlds.py", "fast_embedding.py")]
    files += [V2 / name for name in ("route_rollouts.py", "connected_reader.py", "public_embedding.py")]
    files += sorted(V1.rglob("*.py"))
    input_files = [pack, BONE_PRIOR, DEFAULT_ATLAS_PATH, DEFAULT_CHART_SOURCE]
    return {"experiment": EXPERIMENT, "atlas_only": True, "dird": False,
            "sources_sha256": {str(path.relative_to(ROOT.parent)): digest(path) for path in files},
            "inputs_sha256": {path.name: digest(path) for path in input_files},
            "counts": counts, "shard_size": READER_SHARD,
            "seeds": {stage: SEED_RANGES[stage] for stage in counts}}


def collect_readers(out, pack, workers, counts):
    expected = contract(pack, counts)
    protocol = out / "reader_collection.json"
    if protocol.exists() and expected != read_json(protocol):
        raise RuntimeError("Reader collection settings changed. Use a new output directory.")
    atomic_json(protocol, expected)
    result = {}
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"),
                             initializer=init_worlds, initargs=(pack,)) as pool:
        for stage, count in counts.items():
            result[stage] = collect(pool, stage, count, out, collect_reader_case, pack_tracks, READER_SHARD)
    return result


def cli():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "run")
    parser.add_argument("--atlas-pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--smoke", type=int, default=0)
    args = parser.parse_args()
    counts = {"smoke": args.smoke} if args.smoke else READER_COUNTS
    pack = args.atlas_pack.resolve()
    results = collect_readers(args.out, pack, args.workers, counts)
    print({stage: len(meta) for stage, (_, meta) in results.items()}, flush=True)


if __name__ == "__main__":
    cli()
