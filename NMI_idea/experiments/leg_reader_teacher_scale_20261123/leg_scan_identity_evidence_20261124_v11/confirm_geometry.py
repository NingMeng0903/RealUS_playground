"""Confirm the locked geometric window control on a second layout cohort."""

from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path

import run_v11 as run
from evidence import ROOT
from pipeline import measure
from recovery_task import DEFAULT_PACK
from study_v9 import atomic_json, digest, freeze, read_json


LAYOUT_SEED = 8_158_696_000_000_000
TAPES = (None, 104_293_477, 104_311_579)
COUNT = 144


def confirm(args):
    out = args.out
    first = read_json(out / "protocol.json")
    freeze_doc = read_json(out / "test_freeze.json")
    if any(digest(ROOT / name) != expected for name, expected in first["source_sha256"].items()):
        raise RuntimeError("The v11 source differs from the first frozen panel")
    if digest(args.atlas) != first["atlas_sha256"]:
        raise RuntimeError("The atlas differs from the first frozen panel")

    actor_lock = read_json(run.V10_RUN / "independent_freeze.json")
    actor_folder = (run.V10_RUN / "online" / f"{actor_lock['candidate']:02d}" if actor_lock["candidate"]
                    else run.V10_RUN / "skill_adapted")
    actor = actor_folder / "actor.json"
    blend = out / "mixed_reader" / "reader_bundle.json"
    model = out / "model" / "model.json"
    reader = blend if read_json(out / "reader_freeze.json")["selected"] == "mixed" else run.READER
    for path, expected in ((actor, first["frozen_v10_actor_sha256"]), (run.READER, first["frozen_v10_reader_sha256"]),
                           (reader, freeze_doc["reader_sha256"]), (model, freeze_doc["model_sha256"])):
        if digest(path) != expected:
            raise RuntimeError(f"The frozen asset changed: {path}")

    folder = out / "geometry_confirmation"
    first_result = read_json(out / "results.json")
    settings = freeze_doc["rule"]
    lock = {
        "source_sha256": digest(Path(__file__)),
        "first_protocol_sha256": digest(out / "protocol.json"),
        "first_results_sha256": digest(out / "results.json"),
        "first_panel_sha256": first_result["paired_sha256"],
        "frozen_settings": settings,
        "layouts": COUNT,
        "layout_seed": LAYOUT_SEED,
        "tapes": TAPES,
        "primary": "whole-route strict delivery, geometry minus equal-time repeat, mean over the two retapes",
        "secondary": "whole-route strict and wrong-branch delivery versus v10; actual tape and both retapes",
    }
    lock_sha = freeze(folder, "freeze.json", lock)
    roster = []
    for j, case in enumerate(run.cases("test", COUNT)):
        seed = LAYOUT_SEED + j * 29
        roster.append(replace(case, identifier=f"v11_geometry_confirm_{j:05d}", split="v11_geometry_confirm",
                              world_seed=seed, tape_seed=seed + 1, action_seed=seed + 2))
    with run.pool(args, reader, actor, model, blend) as work:
        _, paired = run.collect(work, folder / "shards", roster, run.control_job,
                                lambda case: (case, TAPES, settings))
    previous = [out / "acquisition" / x for x in ("mine", "cal", "dev")] + [
        out / "teach", out / "dev_policy" / f"{freeze_doc['selected']:02d}", out / "independent"]
    prior_ids = set()
    for group in previous:
        for marker in group.glob("*/commit.json"):
            _, rows = run.verify_shard(marker.parent)
            prior_ids.update(row["layout_id"] for row in rows)
    if prior_ids.intersection(row["layout_id"] for row in paired):
        raise RuntimeError("The confirmation cohort reused a previous layout")
    panel = folder / "paired_outcomes.json"
    atomic_json(panel, paired)
    result = {"freeze_sha256": lock_sha, "paired_sha256": digest(panel), "layouts": len(paired),
              "prior_layouts_checked": len(prior_ids), "scores": measure(paired)}
    atomic_json(folder / "results.json", result)
    whole = result["scores"]["groin_to_distal_calf"]["arms"]
    print({arm: whole[arm]["strict_per_tape"] for arm in ("v10", "base", "repeat", "geometry")}, flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "run")
    parser.add_argument("--atlas", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--workers", type=int, default=14)
    confirm(parser.parse_args())
