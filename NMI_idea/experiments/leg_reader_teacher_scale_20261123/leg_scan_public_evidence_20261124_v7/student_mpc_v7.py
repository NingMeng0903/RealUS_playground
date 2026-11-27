"""Acquire ordinary MPC on the same student test layouts and proof tapes."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import multiprocessing as mp

from study_v7 import DEFAULT_PACK, ROOT, atomic_json, digest, freeze, init_pinned, pinned, read_json, shards
from live_programs import SCORE_STREAMS, execute, prefix_setup
from cycles_v7 import EVALUATION, branch, card
from verify_v7 import committed
import numpy as np


def ordinary(case):
    ctx = prefix_setup(case)
    values = {band: execute(ctx, branch(ctx, offset), "mpc") for band, offset in zip(SCORE_STREAMS, EVALUATION, strict=True)}
    return {"strict": np.asarray([r["strict"] for r in values.values()])}, {
        "case": asdict(case), "layout_id": ctx["tape"].layout.identifier, "complete": True, "mpc": values}


def pack(rows):
    return {"strict": np.stack([r[0]["strict"] for r in rows])}


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "run")
    p.add_argument("--atlas-pack", type=Path, default=DEFAULT_PACK)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--score-only", action="store_true")
    a = p.parse_args()
    run = read_json(a.out/"mechanism_protocol.json")
    if digest(a.atlas_pack) != run["atlas_sha256"] or tuple(run["scoring_offsets"]) != EVALUATION:
        raise RuntimeError("MPC replay does not use the student protocol input")
    protocol = {"student_protocol_sha256": digest(a.out/"mechanism_protocol.json"), "scoring_offsets": EVALUATION,
                "reference": "ordinary MPC from same public prefix", "source_sha256": digest(Path(__file__))}
    lock = freeze(a.out, "student_mpc_protocol.json", protocol)
    path = a.out / "student_mpc_replay"
    if not a.score_only:
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn"), initializer=init_pinned,
                                 initargs=(a.atlas_pack, pinned(a.atlas_pack))) as pool:
            rows = shards(pool, path, "student_test", ordinary, pack)
        atomic_json(a.out/"student_mpc_complete.json", {"protocol_sha256": lock, "layouts": len(rows)})
        return
    if read_json(a.out/"student_mpc_complete.json")["protocol_sha256"] != lock:
        raise RuntimeError("Paired ordinary MPC uses another protocol")
    mpc = committed(path, "student_test")
    students = committed(a.out, "student_test")
    merged = join(students, mpc)
    routes = card(merged, ("mpc", "reference", "student"), "mpc")
    # The ordinary scan and two-decision policy start their scoring slices at
    # different prefixes. Full-route statistics use the same scoring event.
    for route in routes.values():
        for arm in route["arms"].values():
            for field in ("joint", "old_pool_correct", "suffix_valid"):
                arm["scores"].pop(field)
                arm["vs_reference"].pop(field)
    atomic_json(a.out/"student_mpc_results.json", {"protocol_sha256": lock, "routes": routes})


def join(students, mpc):
    merged = []
    for control, student in zip(mpc, students, strict=True):
        if (control["layout_id"], control["case"]) != (student["layout_id"], student["case"]):
            raise RuntimeError("Nonidentical student and MPC layout or tape")
        scores = {band: student["outcomes"][band] | {"mpc": control["mpc"][band]} for band in SCORE_STREAMS}
        merged.append(student | {"outcomes": scores})
    return merged


if __name__ == "__main__":
    main()
