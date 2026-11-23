"""Recompute delivered outcomes and check frozen inputs and scan shards."""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import numpy as np

import run_v11 as run
from confirm_geometry import COUNT, LAYOUT_SEED, TAPES
from evidence import ROOT, query, views
from evidence_model import infer, prior_for
from public_v9 import Prefix
from recovery_task import DEFAULT_PACK, ROUTES
from study_v9 import atomic_json, digest, read_json, verify_shard


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_files(directory, hashes):
    for name, expected in hashes.items():
        require(digest(directory / name) == expected, f"Digest mismatch: {directory / name}")


def load_shards(folder):
    return [r for marker in sorted(folder.glob("*/commit.json")) for r in verify_shard(marker.parent)[1]]


def layout_ids(rows):
    ids = {r["layout_id"] for r in rows}
    require(len(ids) == len(rows), "A cohort contains repeat layouts")
    return ids


def interval(values, rng):
    bootstrap = [values[indices].mean() for indices in rng.integers(len(values), size=(2500, len(values)))]
    return np.quantile(bootstrap, [.025, .975])


def verify_panel(panel, result):
    require(len(panel) == result["layouts"], "Panel length differs")
    layout_ids(panel)
    checks = Counter()
    eligible_anchors = Counter()
    for row in panel:
        require(set(row["tapes"]) == {"actual", "proof_0", "proof_1"}, "Unexpected tape list")
        for sample in row["tapes"].values():
            arms, execution = sample["arms"], sample["executed"]
            for grade in arms.values():
                ids = grade["selected_ids"]
                require(len(ids) == len(grade["capture"]) == 6 and grade["completed_legal"],
                        "An admitted trajectory did not finish six legal regions")
                require(grade["strict"] == all(i == 0 for i in ids), "Incorrect strict grade")
                require(grade["wrong_branch"] == any(i > 0 for i in ids), "Incorrect wrong branch grade")
                require(all(i != 0 or visible for i, visible in zip(ids, grade["capture"], strict=True)),
                        "A correct delivery lacked a captured target")
                checks["episode_grades"] += 1
            if row["case"]["task"] == "femoral":
                require(all(m == arms["v10"] for k, m in arms.items() if k != "mixed_readout"),
                        "Frozen femoral execution changed")
            if not execution:
                continue
            checks["matched_acquisition_sets"] += 1
            target_region = execution["target_region"]
            for arm in ("repeat", "geometry", "information", "task"):
                e = execution[arm]
                dt, ds = arms[arm]["time_s"] - arms["base"]["time_s"], arms[arm]["travel_mm"] - arms["base"]["travel_mm"]
                require(e["ticks"] == execution["repeat"]["ticks"]
                        and np.isclose(dt, .1 * e["ticks"]) and np.isclose(ds, e["distance_mm"]),
                        "Matched acquisition disagrees with the physical clock or odometer")
                require(e["station"] == execution["repeat"]["station"], "Control changed axial station")
            memory = execution["memory"]
            require(memory["region"] == target_region and memory["frame_slot_keys"][0] is not None,
                    "Missing fixed old address")
            require(np.isclose(sum(memory["posterior"]), 1.) and min(memory["posterior"]) >= 0.,
                    "Posterior does not define a probability distribution")
            eligible_anchors["true_b"] += memory["anchors"]["hypothesis"] == 1
            eligible_anchors["old_images_promoted"] += memory["changed"]
            if execution["task"]["event"] == 1:
                eligible_anchors["B_only"] += 1
                eligible_anchors["B_only_and_true_b"] += memory["anchors"]["hypothesis"] == 1
            if not memory["changed"]:
                require(arms["task"] == arms["task_old_image"], "A supposedly unchanged readout changed")

    for route in ROUTES:
        rows = [r["tapes"] for r in panel if r["case"]["task"] == route]
        expected = result["scores"][route]
        require(len(rows) == expected["layouts"], "Wrong route denominator")
        for arm, scores in expected["arms"].items():
            original = [r["actual"]["arms"][arm] for r in rows]
            ids = np.array([a["selected_ids"] for a in original])
            for key, computed in {
                "strict": sum(a["strict"] for a in original),
                "wrong_branch": sum(a["wrong_branch"] for a in original),
                "named_wrong_regions": int((ids > 0).sum()),
                "artifact": int((ids == -1).sum()),
                "mixed": int((ids == -2).sum()),
                "reject": int((ids == -99).sum()),
                "correct_by_region": (ids == 0).sum(0).tolist(),
                "visible_by_region": np.array([a["capture"] for a in original]).sum(0).tolist(),
                "strict_per_tape": [sum(r[t]["arms"][arm]["strict"] for r in rows)
                                    for t in ("actual", "proof_0", "proof_1")],
            }.items():
                require(computed == scores[key], f"Different {route}/{arm}/{key}")
            for ref, pair in scores["paired"].items():
                for key, expected_delta in pair.items():
                    tape_deltas = np.array([[int(r[t]["arms"][arm][key]) - int(r[t]["arms"][ref][key])
                                             for t in ("actual", "proof_0", "proof_1")] for r in rows])
                    actual = tape_deltas[:, 0]
                    require(int((actual > 0).sum()) == expected_delta["rescued"]
                            and int((actual < 0).sum()) == expected_delta["lost"]
                            and np.isclose(actual.mean(), expected_delta["actual"][0])
                            and np.isclose(tape_deltas[:, 1:].mean(), expected_delta["independent"][0]),
                            f"Different paired outcome {route}/{arm}/{ref}/{key}")
                    random = np.random.default_rng(701_542)
                    for tag, values in (("actual", actual), ("independent", tape_deltas[:, 1:].mean(axis=1))):
                        require(np.allclose(interval(values, random), expected_delta[tag][1:]),
                                f"Different paired interval {route}/{arm}/{ref}/{key}/{tag}")
                    checks["paired_endpoints"] += 1
    return dict(checks) | dict(eligible_anchors)


def verify_public_replay(atlas, actor, reader, model):
    run.init(atlas, reader, actor, model, reader)
    case = next(c for c in run.cases("select", 72) if c.task == ROUTES[1])
    live, public = run.acquire(case, None)
    reader_node = live.scan.node.reader
    old_frames = reader_node.frames
    # The echo label field is private. Keep echo positions and every public measurement fixed.
    reader_node.frames = [f[:3] + (np.full(len(f[2]), 551),) + f[4:] for f in old_frames]
    try:
        scrubbed = Prefix(live.scan.env, live.scan.spec).update(live.scan.node)
        require(public.keys() == scrubbed.keys() and
                all(np.array_equal(public[k], scrubbed[k], equal_nan=True) for k in public),
                "Changing private labels changed the public prefix")
        for r in (3, 4, 5):
            p1, p2 = [run.predict(x, run._WORK_READER, case.task) for x in (public, scrubbed)]
            a, b = query(public, p1, r), query(scrubbed, p2, r)
            require(a is not None and b is not None and np.array_equal(a.features, b.features),
                    "Private labels changed the selected identity context")
            plans_a, plans_b = views(live, a), views(live, b)
            poses = lambda plans: [(v.name, v.station, v.sector, v.dwell, v.ticks, v.visibility) for v in plans]
            require(len(plans_a) >= 2 and poses(plans_a) == poses(plans_b)
                    and np.array_equal(prior_for(run._MODEL, a), prior_for(run._MODEL, b)), "Action set or prior changed")
            for x, y in zip(infer(run._MODEL, plans_a), infer(run._MODEL, plans_b), strict=True):
                require(np.array_equal(x, y), "Public action values changed")
    finally:
        reader_node.frames = old_frames
    return True


def verify(out, atlas):
    p = read_json(out / "protocol.json")
    r = read_json(out / "reader_freeze.json")
    t = read_json(out / "test_freeze.json")
    result = read_json(out / "results.json")
    check_files(ROOT, p["source_sha256"])
    check_files(out, {"protocol.json": result["protocol_sha256"], "test_freeze.json": result["freeze_sha256"],
                     "paired_outcomes.json": result["paired_sha256"]})
    check_files(out / "model", {"model.json": t["model_sha256"],
                              "checkpoint.pt": read_json(out / "model" / "training.json")["checkpoint_sha256"]})
    check_files(out / "mixed_reader", {"reader_bundle.json": r["mixed_reader_sha256"],
                                     "best.pt": read_json(out / "mixed_reader" / "results.json")["selected"]["checkpoint_sha256"]})
    require(digest(atlas) == p["atlas_sha256"] and digest(run.READER) == p["frozen_v10_reader_sha256"],
            "The atlas or v10 reader changed")
    actor_lock = read_json(run.V10_RUN / "independent_freeze.json")
    actor_path = run.V10_RUN / ("online/%02d" % actor_lock["candidate"] if actor_lock["candidate"] else "skill_adapted") / "actor.json"
    require(digest(actor_path) == p["frozen_v10_actor_sha256"], "The frozen actor changed")
    reader = out / "mixed_reader" / "reader_bundle.json" if r["selected"] == "mixed" else run.READER
    require(digest(reader) == r["reader_sha256"] == t["reader_sha256"], "Frozen reader differs")

    groups = {name: load_shards(out / "acquisition" / name) for name in ("mine", "cal", "dev")}
    groups.update(teach=load_shards(out / "teach"),
                  select=load_shards(out / "dev_policy" / f"{t['selected']:02d}"),
                  test=load_shards(out / "independent"))
    for name, rows in groups.items():
        require(len(rows) == p["cohorts"][name], f"Missing committed scans in {name}")
    ids = [layout_ids(rows) for rows in groups.values()]
    actual = set.union(*ids)
    require(len(actual) == sum(len(x) for x in ids) == result["layout_cohorts_disjoint"], "Cohort overlap")
    primary_panel = read_json(out / "paired_outcomes.json")
    require(primary_panel == groups["test"], "Independent panel differs from committed shards")
    report = {"source_and_checkpoint_digests": True, "disjoint_layouts": len(actual),
              "primary": verify_panel(primary_panel, result)}

    confirmation = out / "geometry_confirmation"
    conf = read_json(confirmation / "results.json")
    lock = read_json(confirmation / "freeze.json")
    check_files(confirmation, {"freeze.json": conf["freeze_sha256"], "paired_outcomes.json": conf["paired_sha256"]})
    require(lock["source_sha256"] == digest(ROOT / "confirm_geometry.py") and lock["layouts"] == COUNT
            and lock["layout_seed"] == LAYOUT_SEED and lock["tapes"] == list(TAPES)
            and lock["first_results_sha256"] == digest(out / "results.json"), "Confirmation lock mismatch")
    confirm_panel = read_json(confirmation / "paired_outcomes.json")
    require(confirm_panel == load_shards(confirmation / "shards") and not layout_ids(confirm_panel) & actual,
            "Confirmation cohort repeated layouts or differs from shards")
    report["confirmation"] = verify_panel(confirm_panel, conf)
    report["public_prefix_only_label_swap"] = verify_public_replay(
        atlas, actor_path, reader, out / "model" / "model.json")
    atomic_json(out / "verification.json", report)
    print(report)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "run")
    parser.add_argument("--atlas", type=Path, default=DEFAULT_PACK)
    args = parser.parse_args()
    verify(args.out, args.atlas)
