"""Verify the committed route panels, frozen checkpoints, and observation boundary."""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import numpy as np

import data_v12
from closed_scan import action_vector, programs
from evidence import query as v11_query
from fit_v12 import hydrate, linear, net
from public_v9 import Prefix
from recovery_reader import predict, read_shard
from recovery_task import DEFAULT_PACK, Live
from replay_v11 import ROOT, RUN11, actor_path, case_from
from run_v12 import TAPES
from study_v9 import atomic_json, digest, read_json, verify_shard
import study_worlds
from track_state import request, tracks


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _committed(folder):
    return [row for f in sorted(folder.glob("*/commit.json")) for row in verify_shard(f.parent)[1]]


def _check_panel(rows, expected):
    n = expected["layouts"]
    require(len(rows) == n and len({row["layout_id"] for row in rows}) == n, "Wrong layout count")
    counters = Counter()
    for r in rows:
        for sample in r["tapes"].values():
            arms = sample["arms"]
            for score in arms.values():
                ids = score["selected_ids"]
                require(len(ids) == 6 and score["completed_legal"], "Incomplete route")
                require(score["strict"] == all(i == 0 for i in ids)
                        and score["wrong_branch"] == any(i > 0 for i in ids), "Terminal score differs")
                require(all(i != 0 or v for i, v in zip(ids, score["capture"], strict=True)), "Result invented a target")
                counters["case_arm_scores"] += 1
            if r["case"]["task"] == "femoral":
                require(all(x == arms["v10"] for x in arms.values()), "Frozen femoral trajectory changed")
            for arm in ("joint_old_readout", "joint_no_material_readout"):
                require(all(arms[arm][k] == arms["joint"][k] for k in ("time_s", "travel_mm", "capture")),
                        "A fixed-path reader switch changed acquisition")
            for arm, acts in sample["actions"].items():
                if arm == "v11":
                    continue
                seen = set()
                previous = -1
                for a in acts:
                    key = (a["region"], a["program"])
                    require(a["start_tick"] >= previous and a["end_tick"] > a["start_tick"] and
                            .1*a["end_tick"] <= arms[arm]["time_s"]+1.e-6 and key not in seen, "Invalid action history")
                    previous = a["end_tick"]
                    seen.add(key)
                    require(all(a["start_tick"]-1 <= key[0] < a["end_tick"] for key in a["bridge_keys"]),
                            "A bridge used an unacquired echo")
                    counters["real_evidence_actions"] += 1
                    counters["continuations_after_evidence"] += a["before_end"]
    for k, info in expected["arms"].items():
        for p, metric in (("strict", "strict"), ("wrong", "wrong_branch")):
            counts = [sum(r["tapes"][t]["arms"][k][metric] for r in rows) for t in ("actual", "proof_0", "proof_1")]
            require(counts == info[p], f"{k} denominators differ")
        for ref, endpoints in info["deltas"].items():
            for name, e in endpoints.items():
                d = np.array([[int(row["tapes"][t]["arms"][k][name])-int(row["tapes"][t]["arms"][ref][name])
                               for t in ("actual", "proof_0", "proof_1")] for row in rows])
                require((d[:, 0] > 0).sum() == e["rescued"] and (d[:, 0] < 0).sum() == e["lost"],
                        f"{k} paired counts differ from report")
                rng = np.random.default_rng(701_542)
                for label, v in (("actual", d[:, 0]), ("independent", d[:, 1:].mean(axis=1))):
                    q = np.quantile(v[rng.integers(n, size=(2500, n))].mean(axis=1), [.025, .975])
                    require(np.allclose([v.mean(), *q], e[label]), f"{k} paired interval differs")
    return dict(counters)


def _label_swap(atlas, model_dir, choice):
    edge = hydrate(read_json(model_dir/"association.json"))
    decoder = hydrate(read_json(model_dir/"reader.json"))
    data_v12.init(atlas, actor_path(), edge, decoder, choice["blend"])
    engine = data_v12._ENGINE
    station = next(data_v12.cases("policy_dev", 4))
    live = engine.to_frontier(Live(station), 4)
    raw = engine.read(live)
    raw_moves = programs(live, raw)
    a = np.stack([action_vector(live, raw, k) for k in (None, *raw_moves)])
    require(len(a) > 1, "Label-swap check requires more than continuation")
    actual = live.scan.node.reader.frames
    live.scan.node.reader.frames = [f[:3] + (np.full(len(f[2]), 281),) + f[4:] for f in actual]
    live.scan.prefix = Prefix(live.scan.env, live.scan.spec)
    scrubbed = engine.read(live)
    moves = programs(live, scrubbed)
    b = np.stack([action_vector(live, scrubbed, k) for k in (None, *moves)])
    model = hydrate(read_json(model_dir/f"student_{choice['model'][:2]}.json"))
    require(np.array_equal(a, b) and np.array_equal(linear(model, a), linear(model, b)),
            "Privileged labels changed a public decision")
    require([(k.key, k.station, k.sector, k.anchor) for k in raw_moves] ==
            [(k.key, k.station, k.sector, k.anchor) for k in moves], "Private labels changed legal plans")
    return model, a


def _candidate_choices(out, edge, reader):
    counts = {"v11": Counter(), "v12": Counter(), "identical_a": 0}
    for path in sorted((out/"reader_data"/"reader_dev").glob("*/observer/snapshots.npz")):
        for row in read_shard(path):
            if row["scope"] != "plain":
                continue
            p, ids, y = row["public"], row["instance"], row["labels"]
            base = predict(p, reader, row["route"])
            own = v11_query(p, base, 4)
            new = request(p, base | {"track_id": tracks(p, base, edge)}, 4)
            if own is None or new is None:
                continue
            lookup = {k: a for a, k in enumerate(zip(p["frames"], p["slots"], strict=True))}
            old_ids = tuple(lookup[a.key] if a is not None else -1 for a in own.anchors)
            if old_ids[0] != new.indices[0]:
                continue
            counts["identical_a"] += 1
            for name, (a, b) in (("v11", old_ids), ("v12", new.indices)):
                counts[name].update(have_b=int(b>=0), old_is_wrong=int(y[a]!=0),
                    same_instance=int(b>=0 and ids[a]>=0 and ids[a]==ids[b]),
                    old_wrong_alternative_target=int(b>=0 and y[a]!=0 and y[b]==0))
    return counts


def _augmentation(out):
    changed = 0
    displacements = []
    for row in _committed(out/"reader_data"/"fit"):
        case = case_from(row)
        env = study_worlds._WORLDS.environment(case.shape)
        args = (case.source, case.world_seed, case.profile, case.split)
        base = env.layout_factory.build(*args)
        variant = data_v12.RelativeLayouts(env.layout_factory).build(*args)
        require(variant.identifier == row["layout_id"] and np.array_equal(base.links, variant.links)
                and np.array_equal(np.isfinite(base.centers), np.isfinite(variant.centers)),
                "The synthetic augmentation altered support or generated a different layout")
        if row["relative"]:
            changed += 1
            require(np.array_equal(base.centers[:, 2:], variant.centers[:, 2:], equal_nan=True),
                    "A relative artery/vein perturbation moved unrelated anatomy")
            displacements.append(float(np.nanmax(np.linalg.norm(variant.centers-base.centers, axis=-1))))
    return {"changed_layouts": changed, "maximum_displacement_mm": max(displacements)}


def _old_localization():
    counter = Counter()
    for folder in (RUN11, RUN11/"geometry_confirmation"):
        for row in read_json(folder/"paired_outcomes.json"):
            if row["case"]["task"] != "groin_to_distal_calf":
                continue
            tape = row["tapes"]["actual"]
            missed = [i+1 for i, vessel in enumerate(tape["arms"]["geometry"]["selected_ids"]) if vessel != 0]
            if len(missed) == 1:
                counter[f"sole_failed_region_{missed[0]}"] += 1
            if missed == [5]:
                counter["r5_arteries"] += tape["arms"]["geometry"]["selected_ids"][4] == 1
                counter["r5_not_reviewed"] += tape["executed"].get("target_region") != 5
    return dict(counter)


def _parity(out, choice, model, a):
    import torch

    prefix = choice["model"][:2]
    artifact = out/"models"/f"student_{prefix}.pt"
    calibration = read_json(out/"models"/f"student_{prefix}_training.json")
    require(digest(artifact) == calibration["checkpoint_sha256"], "Student checkpoint changed")
    weights = torch.load(artifact, map_location="cuda", weights_only=False)
    require(len(weights["optimizer"]["state"]) > 0 and all(p.is_cuda for p in weights["model"].values()),
            "Student did not store optimizer state and CUDA parameters")
    fresh = net(a.shape[1], (54, 36, 4))
    fresh.load_state_dict(weights["model"])
    x = torch.as_tensor(((a-model["mean"])/model["scale"]).clip(-6., 6.), device="cuda")
    result = fresh(x).detach().cpu().numpy()
    diff = float(np.max(np.abs(result-linear(model, a))))
    require(diff < .001, "CUDA checkpoint and actor export disagree")
    return diff


def verify(out, atlas):
    config, data = read_json(out/"protocol.json"), read_json(out/"reader_data_freeze.json")
    for listing in (config["source"], data["sources"]):
        require(all(digest(ROOT/k) == h for k, h in listing.items()), "Source code no longer matches freeze")
    choice, scores = read_json(out/"test_freeze.json"), read_json(out/"results.json")
    require(digest(out/"protocol.json") == choice["protocol_sha256"] and digest(atlas) == config["atlas_sha256"] and
            digest(out/"test_freeze.json") == scores["freeze_sha256"] and tuple(config["heldout_tapes"]) == TAPES,
            "Heldout freeze mismatch")
    gpu = read_json(out/"gpu_preflight.json")
    require(gpu["cuda_optimizer_step_verified"], "GPU preflight was incomplete")
    stems = {"association": ("association_training.json", "checkpoint_sha256"),
             "reader": ("reader_training.json", "reader_checkpoint_sha256")}
    model_folder = out/"models"
    for name, (report, field) in stems.items():
        require(digest(model_folder/f"{name}.pt") == read_json(model_folder/report)[field], "Reader checkpoint changed")
        require(digest(model_folder/f"{name}.json") == choice[f"{name}_sha256"], "Reader export changed")
    require(digest(model_folder/f"student_{choice['model'][:2]}.json") == choice["actor_sha256"], "Actor export changed")
    universe = []
    for name in ("fit", "cal", "reader_dev"):
        universe.extend(r["layout_id"] for r in _committed(out/"reader_data"/name))
    for name in ("teach_fit", "teach_cal", "adapt_fit", "adapt_cal"):
        universe.extend(r["layout_id"] for r in _committed(out/"teacher"/name))
    universe.extend(r["layout_id"] for r in _committed(out/"evaluation"/"policy_dev"))
    report = {}
    for name in ("test", "femoral"):
        path = out/f"{name}_paired.json"
        require(digest(path) == scores[name]["paired_sha256"], "Independent results file changed")
        raw = read_json(path)
        require(raw == _committed(out/"evaluation"/name), "Independent results differ from episode logs")
        universe.extend(row["layout_id"] for row in raw)
        report[name] = _check_panel(raw, scores[name])
    require(len(set(universe)) == len(universe) == scores["disjoint_layouts"], "A layout leaked across phases")
    for name in ("original", "confirmation"):
        n = 80 if name == "original" else 96
        f = out/f"v11_cross_{name}.json"
        recorded = read_json(out/"v11_cross_summary.json")[name]
        require(len(read_json(f)) == n and digest(f) == recorded["outcomes_sha256"], "Fixed-motion crossover mismatch")
    model, features = _label_swap(atlas, model_folder, choice)
    report |= {"disjoint_layouts": len(universe), "public_label_swap": True, "v11_matched_layouts": 176,
               "actor_cpu_cuda_max_abs": _parity(out, choice, model, features),
               "v11_failure_localization": _old_localization(), "relative_geometry_checks": _augmentation(out),
               "r5_shared_incumbent_reader_dev": _candidate_choices(out, data_v12._ENGINE.edge, data_v12._ENGINE.old)}
    atomic_json(out/"verification.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"run")
    parser.add_argument("--atlas", type=Path, default=DEFAULT_PACK)
    args = parser.parse_args()
    verify(args.out, args.atlas)
