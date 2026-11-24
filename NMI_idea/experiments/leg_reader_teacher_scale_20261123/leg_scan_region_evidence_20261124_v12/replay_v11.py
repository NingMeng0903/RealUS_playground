"""Read the exact v11 geometric trajectories with both frozen readers."""

from argparse import ArgumentParser
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
V11 = ROOT.parent / "leg_scan_identity_evidence_20261124_v11"
sys.path.insert(1, str(V11))

import run_v11 as original
from environment import Case
from evidence import execute_view, query, views
from pipeline import paired_delta
from recovery_task import DEFAULT_PACK
from study_v9 import atomic_json, digest, read_json

RUN11 = V11 / "run"
MIXED = RUN11 / "mixed_reader" / "reader_bundle.json"


def actor_path():
    lock = read_json(original.V10_RUN / "independent_freeze.json")
    stage = f"online/{lock['candidate']:02d}" if lock["candidate"] else "skill_adapted"
    return original.V10_RUN / stage / "actor.json"


def case_from(record):
    return Case(**record["case"])


def replay(job):
    reference, offsets = job
    case = case_from(reference)
    panel = {}
    for (tag, sample), offset in zip(reference["tapes"].items(), offsets, strict=True):
        live, public = original.acquire(case, offset)
        mixed, base_mixed = original.assessed(live, public, original._WORK_READER)
        base_old = original.assessed(live, public, original._V10_READER)[1]
        if original.outcome(base_old) != sample["arms"]["v10"] or original.outcome(base_mixed) != sample["arms"]["base"]:
            raise RuntimeError("The v11 starting trajectory did not replay exactly")
        geo = sample["executed"].get("geometry")
        branch = live
        if geo:
            state = query(public, mixed, sample["executed"]["target_region"] - 1)
            action = next(v for v in views(live, state) if (v.name, v.station, v.sector) ==
                          (geo["view"], geo["station"], geo["sector"]))
            branch, public, _, grade, detail = execute_view(live, action, state, original._WORK_READER)
            if detail["ticks"] != geo["ticks"] or detail["event"] != geo["event"]:
                raise RuntimeError("The recorded geometric action changed")
        else:
            grade = base_mixed
        if original.outcome(grade) != sample["arms"]["geometry"]:
            raise RuntimeError("The v11 geometry/readout did not reproduce")
        reread = original.assessed(branch, public, original._V10_READER)[1]
        panel[tag] = {"arms": {"base_v10": original.outcome(base_old), "base_mixed": original.outcome(base_mixed),
                               "geometry_v10": original.outcome(reread), "geometry_mixed": original.outcome(grade)}}
    return {"layout_id": reference["layout_id"], "case": reference["case"], "tapes": panel}


def run(args):
    actor = actor_path()
    frozen = read_json(RUN11 / "protocol.json")
    settings = read_json(RUN11 / "test_freeze.json")
    if (digest(args.atlas) != frozen["atlas_sha256"] or digest(actor) != frozen["frozen_v10_actor_sha256"]
            or digest(original.READER) != frozen["frozen_v10_reader_sha256"] or digest(MIXED) != settings["reader_sha256"]):
        raise RuntimeError("A v11 replay asset changed")
    data = ((RUN11, "original", original.TEST_TAPES),
            (RUN11 / "geometry_confirmation", "confirmation", (None, 104_293_477, 104_311_579)))
    summaries = {}
    with original.pool(args, MIXED, actor) as pool:
        for folder, stage, offsets in data:
            old = folder / "paired_outcomes.json"
            if digest(old) != read_json(folder / "results.json")["paired_sha256"]:
                raise RuntimeError("Stored v11 panel changed")
            rows = [r for r in read_json(old) if r["case"]["task"] == "groin_to_distal_calf"]
            done = []
            for at in range(0, len(rows), 12):
                done.extend(pool.map(replay, [(r, offsets) for r in rows[at:at + 12]]))
                print(stage, len(done), "/", len(rows), flush=True)
            table = {}
            comparisons = (("base_mixed", "base_v10"), ("geometry_v10", "base_v10"),
                           ("geometry_mixed", "base_mixed"), ("geometry_mixed", "geometry_v10"))
            for arm in done[0]["tapes"]["actual"]["arms"]:
                table[arm] = {key: [sum(r["tapes"][t]["arms"][arm][key] for r in done)
                                    for t in ("actual", "proof_0", "proof_1")]
                              for key in ("strict", "wrong_branch")}
            out = args.out / f"v11_cross_{stage}.json"
            atomic_json(out, done)
            summaries[stage] = {"layouts": len(done), "source_panel_sha256": digest(old), "outcomes_sha256": digest(out),
                                "arms": table, "paired": {f"{a}_minus_{b}": {key: paired_delta(done, a, b, key)
                                for key in ("strict", "wrong_branch")} for a, b in comparisons}}
    atomic_json(args.out / "v11_cross_summary.json", summaries)
    print({n: r["arms"] for n, r in summaries.items()}, flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "run")
    parser.add_argument("--atlas", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--workers", type=int, default=12)
    run(parser.parse_args())
