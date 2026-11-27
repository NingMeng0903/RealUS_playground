"""Freeze the two completed primary-seed policies before their paired acceptance."""

from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ["LEG_ENABLE_DIRD"] = "1"
os.environ["LEG_ENABLE_MATERIAL_ATLAS"] = "1"

from dataclasses import asdict
from run_online import (ARM_NAMES, TRAIN_SEEDS, TEST_SEED, TARGET, gpu_manifest,
                        source_hashes, test_arms, test_cases, today)
from storage import atomic_json, digest, read_json


def main():
    seed = TRAIN_SEEDS[0]
    train = ROOT / "training" / f"seed_{seed}"
    runs = {}
    for arm in ARM_NAMES:
        record = read_json(train / arm / "complete.json")
        counts = record["recount"]
        if counts["complete_unique"] != TARGET or counts["excluded"]:
            raise RuntimeError(f"{arm} requires all complete, unique training scans")
        if digest(record["policy"]) != record["policy_sha256"]:
            raise RuntimeError(f"The completed {arm} policy differs from its recorded hash")
        runs[f"{arm}_{seed}"] = record
    source_protocol = ROOT / "protocol.json"
    protocol = digest(source_protocol)
    if any(row["protocol_sha256"] != protocol for row in runs.values()):
        raise RuntimeError("Primary policies disagree on the declared protocol")
    out = ROOT / "primary_acceptance"
    out.mkdir(exist_ok=True)
    (out / "development").mkdir(exist_ok=True)
    for relative in ("protocol.json", "development/same_track_reader.json"):
        data = (ROOT / relative).read_bytes()
        p = out / relative
        if p.exists() and p.read_bytes() != data:
            raise RuntimeError(f"Primary acceptance input differs at {relative}")
        if not p.exists():
            tmp = p.with_suffix(".pending")
            tmp.write_bytes(data)
            tmp.replace(p)
    primary = {
        "utc": today(), "frozen": True, "primary_seed_only": True, "protocol_sha256": protocol,
        "source_sha256": source_hashes(), "threshold": read_json(out / "development/same_track_reader.json")["chosen"]["threshold"],
        "gpu": read_json(ROOT / "gpu_preflight.json"), "runs": runs,
        "reader_calibration_sha256": digest(out / "development/same_track_reader.json"),
        "selection": "Final complete on-policy checkpoints, as declared in the protocol",
        "test_seed": TEST_SEED, "test_cases": [asdict(case) for case in test_cases()],
        "test_arms": [asdict(arm) for arm in test_arms(runs)[:4]],
        "frozen_budget_scales": [1., .47],
        "source_repair": read_json(ROOT / "preflight/source_sector_wrap_repair.json"),
    }
    if not (out / "freeze.json").exists():
        atomic_json(out / "freeze.json", primary)
    print(out / "freeze.json", flush=True)


if __name__ == "__main__":
    main()
