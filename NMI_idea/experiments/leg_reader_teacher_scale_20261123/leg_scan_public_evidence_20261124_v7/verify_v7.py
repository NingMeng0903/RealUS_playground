"""Recount verified complete episodes and export paired results by layout."""

from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path

from study_v7 import COHORTS, DEFAULT_PACK, ROOT, ROUTES, V6_RUN, atomic_json, digest, load_shard, pinned, read_json, roster
from live_programs import SCORE_STREAMS
import numpy as np


def committed(out, stage):
    rows = []
    for lock in sorted((out / stage).glob("*/commit.json")):
        data, batch = load_shard(lock.parent)
        index = read_json(lock)
        if digest(data) != index["data_sha256"] or digest(lock.with_name("cases.jsonl.gz")) != index["metadata_sha256"]:
            raise RuntimeError(f"Invalid shard digest {lock}")
        rows.extend(batch)
    if not all(row["complete"] for row in rows) or [r["case"] for r in rows] != [asdict(c) for c in roster(stage)]:
        raise RuntimeError(f"Incorrect completed roster for {stage}")
    return rows


def verify_lock(out, prefix, atlas):
    path = out / f"{prefix}_protocol.json"
    data = read_json(path)
    if data.get("atlas_sha256", data.get("data_sha256")) != digest(atlas):
        raise RuntimeError(f"Changed atlas for {prefix}")
    if any(digest(ROOT / name) != sha for name, sha in data.get("source_sha256", data.get("source", {})).items()):
        raise RuntimeError(f"Changed source for {prefix}")
    return digest(path)


def subject_pairs(rows, *, mpc_prefix_comparable=True):
    fields = ("strict", "wrong_branch", "joint", "old_pool_correct", "suffix_valid", "time_s", "travel_mm",
              "mm", "longest_mm", "capture", "selected_ids")
    card = []
    for row in rows:
        scores = {}
        for band in SCORE_STREAMS:
            for method, result in row["outcomes"][band].items():
                if not result["completed_legal"] or result["layout_id"] != row["layout_id"]:
                    raise RuntimeError("Comparison did not reach a legal endpoint on the common layout")
                selected = fields
                if method == "mpc" and not mpc_prefix_comparable:
                    selected = tuple(k for k in fields if k not in ("joint", "old_pool_correct", "suffix_valid"))
                scores.setdefault(method, {})[band] = {key: result[key] for key in selected}
        card.append({"layout": row["layout_id"], "route": row["case"]["task"], "arms": scores})
    return card


def recompute(card, reference, report, *, confirmation=False, fields=("strict", "wrong_branch", "joint")):
    for route in ROUTES:
        samples = [r["arms"] for r in card if r["route"] == route]
        n = len(samples)
        indices = np.random.default_rng(180_411_381).integers(0, n, (3500, n))
        for action, present in report[route]["arms"].items():
            found = present["vs_mpc" if confirmation else "vs_reference"]
            for field in fields:
                a = np.asarray([[r[action][band][field] for band in SCORE_STREAMS] for r in samples], dtype=float)
                b = np.asarray([[r[reference][band][field] for band in SCORE_STREAMS] for r in samples], dtype=float)
                diff = a - b
                assert a[:, 0].sum() == present["scores"][field]
                assert int((diff[:, 0] > 0).sum()) == found[field]["actual"]["added_or_increased"]
                assert int((diff[:, 0] < 0).sum()) == found[field]["actual"]["removed_or_decreased"]
                proof = diff[:, 1:].mean(axis=1)
                low, high = np.quantile(proof[indices].mean(axis=1), [.025, .975])
                expected = found[field]["independent_tape_expected"]
                np.testing.assert_allclose([expected["delta"], expected["lower95"], expected["upper95"]], [proof.mean(), low, high])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "run")
    parser.add_argument("--atlas-pack", type=Path, default=DEFAULT_PACK)
    args = parser.parse_args()
    out = args.out.resolve()
    pinned(args.atlas_pack)
    locks = {n: verify_lock(out, n, args.atlas_pack) for n in ("confirmation", "observation", "mechanism")}
    confirm, observed, mechanism, student = (read_json(out / f) for f in (
        "confirmation_results.json", "observation_model.json", "mechanism_results.json", "student_results.json"))
    assert [confirm["frozen"], observed["protocol_sha256"], mechanism["protocol_sha256"], student["protocol_sha256"]] == [
        locks["confirmation"], locks["observation"], locks["mechanism"], locks["mechanism"]]
    for name, model_sha in read_json(out / "mechanism_protocol.json")["models"].items():
        assert digest(out / f"{name}_model.json") == model_sha
    if (out / "edge_score_replay.json").exists():
        replay = read_json(out / "edge_score_replay.json")
        assert replay["parent_protocol_sha256"] == locks["mechanism"]
        assert replay["source_sha256"] == digest(ROOT / "edge_score_replay_v7.py")
    sets = {stage: committed(out, stage) for stage in COHORTS}
    ids = [r["layout_id"] for rows in sets.values() for r in rows]
    historical = set()
    for stage in ("closed_dev", "closed_test", "reader_fresh"):
        for path in (V6_RUN / stage).glob("*/commit.json"):
            _, rows = load_shard(path.parent)
            historical.update(r["layout_id"] for r in rows)
    assert len(ids) == len(set(ids)) and not set(ids) & historical
    for key in ("world_seed", "tape_seed", "action_seed"):
        assert len({r["case"][key] for rows in sets.values() for r in rows}) == len(ids)
    visited = sets["student_wave0"] + sets["student_wave1"]
    for row in visited + sets["student_test"]:
        result, start, visit = row["actual_student"], row["parent_origin"], row["visited_origin"]
        assert result["completed_legal"] and result["layout_id"] == row["layout_id"]
        assert visit["count"] > start["count"] and visit["station"] == start["station"] + 2 and len(row["first_actual_poses"]) == 2
    for i, stage in enumerate(("student_wave0", "student_wave1")):
        fit = read_json(out / f"{stage}_fit.json")
        expected = {r["layout_id"] for r in (sets["student_wave0"] if i == 0 else visited)}
        used = {layout for key in ("fit_layouts", "validation_layouts", "calibration_layouts") for layout in fit["split"][key]}
        assert used == expected and fit["split"]["layout_disjoint"] and fit["protocol"] == locks["mechanism"]
        assert fit["cuda_optimizer_step_verified"] and fit["optimizer_steps"] > 0 and fit["best_validation_checkpoint_restored"]
        assert fit["bundle_sha256"] == digest(out / f"{stage}_student.json") and fit["checkpoint_sha256"] == digest(Path(fit["checkpoint"]))
        assert fit["best_epoch"] == int(np.argmin(fit["validation_losses"])) + 1
    assert student["real_student_training_episodes"] == len(visited)
    assert student["student_weights_sha256"] == digest(out / "student_wave1_student.json")
    from student_mpc_v7 import join
    control_protocol = out / "student_mpc_protocol.json"
    control_results = read_json(out / "student_mpc_results.json")
    assert control_results["protocol_sha256"] == digest(control_protocol)
    assert read_json(control_protocol)["student_protocol_sha256"] == locks["mechanism"]
    ordinary = committed(out / "student_mpc_replay", "student_test")
    paired_students = join(sets["student_test"], ordinary)
    pairs = {stage: subject_pairs(sets[stage]) for stage in ("confirmation", "mechanism_test")}
    pairs["student_test"] = subject_pairs(paired_students, mpc_prefix_comparable=False)
    recompute(pairs["confirmation"], "mpc", confirm["routes"], confirmation=True)
    recompute(pairs["mechanism_test"], "mpc", mechanism["routes"])
    recompute(pairs["student_test"], "reference", student["test"])
    recompute(pairs["student_test"], "mpc", control_results["routes"], fields=("strict", "wrong_branch"))
    for row in visited:
        teacher = row["samples"]
        assert {len(samples) for samples in teacher.values()} <= {3, 6}
        assert all(sample == [0., 0, 0] for sample in teacher["reference"])
        assert len(teacher["reference"]) >= max(map(len, teacher.values()))
    counts = {"verified_layouts": len(ids), "verified_v6_layouts_excluded": len(historical),
              "layouts_by_stage": {n: len(rows) for n, rows in sets.items()},
              "student_training_episodes": len(visited), "student_test_layouts": len(sets["student_test"]),
              "teacher_search_complete_continuations": sum(sum(map(len, r["samples"].values())) for r in visited),
              "observation_complete_probe_continuations": sum(len(r["observations"]) for r in sets["evidence_dev"]),
              "confirmation_complete_scoring_executions": confirm["actual_scoring_executions"],
              "mechanism_complete_scoring_executions": sum(len({v["effective"] for v in band.values()})
                  for row in sets["mechanism_test"] for band in row["outcomes"].values()),
              "student_test_complete_scoring_executions": len(sets["student_test"]) * len(SCORE_STREAMS) * 3,
              "independent_layout_bootstrap_recomputed": True, "all_pairs_legal": True, "protocol_sha256": locks}
    atomic_json(out / "paired_compact.json", pairs)
    atomic_json(out / "verification.json", counts | {"paired_compact_sha256": digest(out / "paired_compact.json")})
    print(counts)


if __name__ == "__main__":
    main()
