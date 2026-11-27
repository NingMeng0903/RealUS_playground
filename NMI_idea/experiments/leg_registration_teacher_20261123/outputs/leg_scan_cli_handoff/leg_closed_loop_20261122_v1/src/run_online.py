"""Run, resume, freeze and accept the right-leg GPU online experiment."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
from functools import lru_cache
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

from audits import (integrated_interventions, paired, render_report, same_track_report,
                    summarize, training_summary, verify_split, verify_topology)
from environment import (ACTION_MM, ACTION_TICKS, EXTRA_MM, EXTRA_TICKS, DEV_SEED,
                         DEV_SHAPES, FEATURE_NAMES, SKILLS, SMOKE_SEED, SOURCES, TEST_SEED,
                         TRAIN_SEEDS, TRAIN_SHAPES, Arm, ScanEnvironment,
                         physical_subset, rollout, run_cases, spec_for, test_cases)
import motion
from scan_kernel import BONE_PRIOR, TASKS
from storage import atomic_json, digest, read_json, read_lines, recount, shards, stash

ROOT = Path(__file__).resolve().parent.parent
HANDOFF = ROOT.parent
CT_PACK = HANDOFF / "leg_scan_study/topology/ct_geometry.npz"
BLENDER = Path("/media/camp/PEI_T7/1_team meeting/atlas_ppt_build/assets/blender_topology_holdout/geometry.npz")
BLENDER_LINKS = HANDOFF / "leg_scan_study/topology/blender_holdout_geometry_links.npz"
ARM_NAMES = ("ppo", "teacher_ppo")
BATCH_SIZE = 256
TARGET = 30_720
VALIDATION_SPACING = 5_120
TEACH_PER_UPDATE = 16
DEV_COUNT = 96
SMOKE_COUNT = 64


@dataclass(frozen=True)
class BatchJob:
    case: object
    arms: tuple
    weight_map: dict
    threshold: float
    teach: bool = False
    forced: bool = False
    audit: bool = False
    budget_scale: float = 1.


def today():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stage_progress(out, event, **details):
    row = {"utc": today(), "pid": os.getpid(), "event": event} | details
    with (out / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    atomic_json(out / "progress.json", row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def protect_holdout():
    blocked = (str(BLENDER.parent), str(BLENDER_LINKS),
               "/SMPLX_Vascular_Retarget_Audit/evidence/",
               "/vessel_centerlines_rest.obj", "/blender_holdout_topology",
               "/blender_holdout/")

    def guard(event, args):
        if event != "open" or not args or isinstance(args[0], int):
            return
        filename = os.fsdecode(args[0])
        if any(token in filename for token in blocked):
            raise PermissionError(f"Blender held-out input before evaluation authorization: {filename}")

    sys.addaudithook(guard)


def init_worker(pack, held, links):
    global WORKER_INPUT, WORKER_ACTORS
    if held is None:
        protect_holdout()
    WORKER_INPUT = pack, held, links
    WORKER_ACTORS = {}
    _env.cache_clear()


@lru_cache(maxsize=7)
def _env(shape, is_holdout):
    pack, held, links = WORKER_INPUT
    if is_holdout:
        if held is None or links is None:
            raise PermissionError("A training worker cannot construct Blender geometry")
        scanner = motion.Scanner.holdout(pack, held, topology=links)
        factory = None
    else:
        from synthetic_anatomy import LayoutFactory

        scanner = motion.Scanner.training(pack, shape)
        factory = LayoutFactory(pack, scanner, shape)
    return ScanEnvironment(scanner, factory)


def worker_actor(weight_path):
    if weight_path not in WORKER_ACTORS:
        from policy import NumpyActor

        if len(WORKER_ACTORS) >= 9:
            WORKER_ACTORS.pop(next(iter(WORKER_ACTORS)))
        with np.load(weight_path, allow_pickle=False) as weights:
            WORKER_ACTORS[weight_path] = NumpyActor({name: weights[name] for name in weights.files})
    return WORKER_ACTORS[weight_path]


def work(job):
    case = job.case
    env = _env(case.shape, case.split == "blender")
    rows = []
    for arm in job.arms:
        actor = worker_actor(job.weight_map[arm.policy]) if arm.policy else None
        result = rollout(env, case, arm, job.threshold, actor,
                         teacher=job.teach and arm.sampled, audit=job.audit,
                         forced=job.forced and not arm.sampled, budget_scale=job.budget_scale,
                         diagnostics_ids=case.split == "blender")
        if case.split == "blender":
            result.metrics.update(physical_subset(env, spec_for(env.scanner, case), result.metrics, case.source))
        rows.append((case, arm.name, result))
    return rows


def harvest(executor, jobs, chunksize=2):
    return [item for group in executor.map(work, jobs, chunksize=chunksize) for item in group]


def pool(args, holdout=False):
    return ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=init_worker,
                               initargs=(args.pack, BLENDER if holdout else None,
                                         BLENDER_LINKS if holdout else None))


def rollout_weights(policy, directory, stem):
    directory.mkdir(parents=True, exist_ok=True)
    file = directory / (stem + ".npz")
    temporary = file.with_suffix(".tmp")
    with temporary.open("wb") as out:
        np.savez(out, **{name: tensor.detach().cpu().numpy() for name, tensor in policy.state_dict().items()})
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, file)
    return str(file)


def source_hashes():
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "src").rglob("*.py"))}


def check_protocol(plan, path):
    """Keep the original protocol and checkpoint lineage after a reviewed numerical repair."""
    import copy

    if not path.is_file():
        atomic_json(path, plan)
        return
    current = json.loads(json.dumps(plan))
    original = read_json(path)
    if original == current:
        return
    file = path.parent / "preflight" / "source_sector_wrap_repair.json"
    if not file.is_file():
        raise RuntimeError("Existing protocol disagrees with source, inputs or requested design")
    repair = read_json(file)
    left, right = copy.copy(original), copy.copy(current)
    before, after = left.pop("source_sha256"), right.pop("source_sha256")
    changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
    if (left != right or repair["original_protocol_sha256"] != digest(path)
            or changed != sorted(repair["source_changes"])
            or any(repair["source_changes"][name] != [before.get(name), after.get(name)] for name in changed)):
        raise RuntimeError("The recorded numerical repair differs from this invocation")


def gpu_manifest(out):
    pre = out / "preflight"
    data = read_json(pre / "cuda_step.json")
    driver = (pre / "driver.txt").read_text().strip().splitlines()
    if not data["cuda_optimizer_step_verified"] or not driver:
        raise RuntimeError("CUDA optimizer preflight record is missing")
    data |= {"nvidia_smi": driver, "driver": driver[0].split(",")[1].strip(),
             "preflight_script_sha256": digest(HANDOFF / "gpu_preflight.sh")}
    atomic_json(out / "gpu_preflight.json", data)
    return data


def protocol(args, gpu):
    from material_assets import DEFAULT_ATLAS_PATH, DEFAULT_CHART_SOURCE

    field = Path("/media/camp/EXT_DRIVE/RealUS_playground/ird_playground/data/checkpoints/rm4d_signed/selected.pt")
    material_files = (DEFAULT_ATLAS_PATH, DEFAULT_CHART_SOURCE, field,
                      HANDOFF / "reference_source/legacy_dird/dird_probe.py")
    return {
        "experiment": "right_leg_closed_loop_relative_anatomy_20261122_v1", "production": args.target >= TARGET,
        "source_sha256": source_hashes(), "gpu": gpu,
        "data_sha256": {str(file): digest(file) for file in (
            args.pack, args.pack.with_name("medseg_ct_route.npz"),
            args.pack.with_name("medseg_native_graph.json"),
            args.pack.with_name("ct_reference_topology.json.gz"), BONE_PRIOR, *material_files)},
        "geometry_splits": {"train_shapes": TRAIN_SHAPES, "development_shapes": DEV_SHAPES, "sources": SOURCES,
                            "native_medseg_cases": 1, "test_anatomies": 1,
                            "same_origin": "BodyParts3D mirrored and right with heritage shared by Open3DModel"},
        "seeds": {"training_replicates": list(TRAIN_SEEDS[:args.repeats]), "development": DEV_SEED,
                  "calibration": DEV_SEED + 1, "smoke": SMOKE_SEED, "test": TEST_SEED,
                  "pairing": "environment and measurement seed shared across arms; distinct replicate streams"},
        "count": {"complete_scans_per_arm_per_seed": args.target, "batch": args.batch,
                  "development_interval": VALIDATION_SPACING, "development_cases": args.validation,
                  "smoke_scan_cases": SMOKE_COUNT, "blender_cases_per_arm": 160},
        "world": {"samples_per_second": 1. / motion.CLOCK_S, "motion_speed_mm_s": motion.MAX_SPEED_MM_S,
                  "task_definitions": TASKS, "CT": "all 17 mapped stations, one 31-layer native study",
                  "profiles": ("ordinary", "stress"),
                  "geometry": "one static relative vessel layout per case, smooth source-bounded role deformations",
                  "anatomy_log": "centers, layout hash, fixed source support and synthetic junction metadata per episode",
                  "recoverable_window": "paired resolution checked on an independent design tape at legal adjacent angle",
                  "validation_profiles": "exclusive SMPL-X feature shapes, seeds and increased shape perturbation range",
                  "bone": "population occupancy train and public planner, exact archived bones only for held-out echoes",
                  "frame_quality": "probe contact and measurement pulse model, independent of hidden gates"},
        "budgets": {"nominal": "common reachable continuation reserve plus wide time and travel allowance",
                    "extra_ticks": EXTRA_TICKS, "extra_mm": EXTRA_MM,
                    "probe_action_ticks": ACTION_TICKS, "probe_action_mm": ACTION_MM,
                    "fixed_probe_penalty": 0., "distance_regularizer": 0., "low_budget_extra_scale": .47},
        "actor": {"features": FEATURE_NAMES, "skills": SKILLS, "continuous": ["circumferential", "axial", "ProDMP_shape"],
                  "policy": "GPU SMDP PPO with two decisions, sampled high skill and conditioned Gaussian motor",
                  "optimizer": "fused CUDA Adam lr .00025", "clock_discount": .996, "step_gae_decay": .96,
                  "atlas": "bounded shaft material address and local fibre on supported geometry",
                  "dird": "virtual shaft-frame scores rank jointly feasible, equivalent 2D ProDMP realizations"},
        "teacher": {"prefixes_sampled_per_update": min(args.batch, TEACH_PER_UPDATE),
                    "evaluation": "continue, actual student, public proposals and continuous samples on two common futures",
                    "independent": "resample sensor innovations conditional on acquired prefix",
                    "terminal_followup": ".20 Bernoulli on first-tape best feasible probe",
                    "objective": "conditional signed gains and soft per-skill preferences"},
        "frozen_test_seed_keys": [asdict(case) for case in test_cases()],
        "holdout_data_access": "locked out of smoke, development and training process",
    }


def policy_instance(seed):
    import torch
    from policy import ScanPolicy

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    net = ScanPolicy(len(FEATURE_NAMES)).to("cuda:0")
    optimizer = torch.optim.Adam(net.parameters(), lr=.00025, fused=True)
    return net, optimizer


def make_checkpoint(policy, optimizer, rng, cursor, arm, seed, spec_hash):
    import torch

    return {"policy": policy.state_dict(), "optimizer": optimizer.state_dict(), "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(), "numpy_rng": rng.bit_generator.state,
            "python_rng": random.getstate(), "cursor": cursor, "arm": arm, "seed": seed,
            "protocol_sha256": spec_hash}


def restore(path, policy, optimizer, rng, arm, seed, spec_hash):
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved["arm"], saved["seed"], saved["protocol_sha256"]) != (arm, seed, spec_hash):
        raise RuntimeError("Checkpoint does not match the protocol, seed and arm of the resume request")
    policy.load_state_dict(saved["policy"])
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["cpu_rng"])
    torch.cuda.set_rng_state(saved["cuda_rng"])
    rng.bit_generator.state = saved["numpy_rng"]
    random.setstate(saved["python_rng"])
    return saved["cursor"]


def cuda_state(policy, optimizer):
    import torch

    states = [t for values in optimizer.state.values() for t in values.values() if isinstance(t, torch.Tensor)]
    if not states or not all(p.is_cuda for p in policy.parameters()) or not all(t.is_cuda for t in states):
        raise RuntimeError("Policy and all optimizer state tensors must remain on CUDA")
    return {"device": str(next(policy.parameters()).device), "parameters": sum(p.numel() for p in policy.parameters()),
            "optimizer_tensor_counts_by_device": {"cuda": len(states)}, "optimizer_steps": int(states[0].item())}


def calibrate(args, executor):
    out = args.out / "development"
    out.mkdir(exist_ok=True)
    summary = out / "same_track_reader.json"
    if summary.is_file():
        return read_json(summary)["chosen"]["threshold"]
    cases = list(run_cases(args.validation, "calibration", DEV_SEED + 1))
    rows = harvest(executor, [BatchJob(case, (Arm("mpc"),), {}, .55, audit=True) for case in cases])
    packed = out / "same_track" / "00000"
    if not packed.exists():
        stash(packed, rows, {"partition": "calibration", "frozen_split_seed": DEV_SEED + 1}, extra=True)
    audit = same_track_report([item for _, _, item in rows])
    atomic_json(summary, audit)
    stage_progress(args.out, "same_track_reader_checked", cases=len(rows), threshold=audit["chosen"]["threshold"],
                   old_correct_equal=audit["old_correct_at_equal"], new_correct_equal=audit["new_correct_at_equal_global"])
    return audit["chosen"]["threshold"]


def smoke(args, executor, threshold, spec_hash):
    from policy import NumpyActor, update_ppo
    from teacher import fit_teacher
    from audits import verify_generated_layouts

    out = args.out / "smoke"
    out.mkdir(exist_ok=True)
    report_file = out / "verification.json"
    if report_file.is_file():
        if read_json(report_file)["protocol_sha256"] != spec_hash:
            raise RuntimeError("Smoke result differs from the training source/protocol")
        return
    rng = np.random.default_rng(SMOKE_SEED)
    net, optim = policy_instance(SMOKE_SEED)
    weight = rollout_weights(net, out, "before")
    cases = list(run_cases(SMOKE_COUNT, "smoke", SMOKE_SEED))
    smoke_arms = (Arm("forced_skills"), Arm("sampled_student", "student", sampled=True))
    jobs = [BatchJob(case, smoke_arms, {"student": weight}, threshold, forced=True, teach=(i % 5 == 0))
            for i, case in enumerate(cases)]
    rows = harvest(executor, jobs)
    student = [item for _, arm, item in rows if arm == "sampled_student"]
    teaching = [item.teaching for item in student if item.teaching and item.teaching["eligible_prefix"]]
    update = update_ppo(net, optim, [item.steps for item in student], rng, epochs=2, minibatch=512,
                        discount=.996, gae_decay=.96)
    aux = fit_teacher(net, optim, teaching)
    gpu = cuda_state(net, optim)
    posterior = rollout_weights(net, out, "after")
    with np.load(weight, allow_pickle=False) as first, np.load(posterior, allow_pickle=False) as second:
        actor_before = NumpyActor({k: first[k] for k in first.files})
        actor_after = NumpyActor({k: second[k] for k in second.files})
    predicted = []
    for item in student:
        for row in item.steps:
            if row["stage"] == 0 and row["mask"][1]:
                before = actor_before.choose(row["features"], row["mask"])
                after = actor_after.choose(row["features"], row["mask"])
                predicted.append({"skill_changed": before.skill != after.skill,
                                  "motor_shift": float(np.linalg.norm(before.action - after.action)),
                                  "probability_shift": float(np.abs(before.probabilities - after.probabilities).sum())})
    pairs = [branch for p in teaching for branch in p["branches"]]
    assert pairs and all(len(b["replays"]) == 2 for b in pairs)
    assert all(p["student_tape_seed"] != p["independent_tape_seed"] for p in teaching)
    results = {arm: summarize([item.metrics for _, name, item in rows if name == arm])
               for arm in ("forced_skills", "sampled_student")}
    assert len(rows) == 2 * SMOKE_COUNT and all(item.metrics["complete"] for _, _, item in rows)
    assert min(results["forced_skills"]["executed_by_skill"][1:]) > 0
    assert results["forced_skills"]["totals"]["early_stops"] > 0
    assert results["sampled_student"]["totals"]["probes"] > 0
    assert max(p["probability_shift"] for p in predicted) > .001
    interventions = integrated_interventions(args.pack, actor_before, threshold)
    anatomy = verify_generated_layouts(args.pack, rows, teaching)
    report = {"status": "passed", "protocol_sha256": spec_hash, "threshold": threshold,
              "cases_by_source_and_profile": {str(i): {p: sum(c.source == i and c.profile == p for c in cases)
                                                        for p in ("ordinary", "stress")} for i in range(4)},
              "results": results, "ppo_cuda_update": update, "teacher_gpu_update": aux, "cuda": gpu,
              "teacher_distinct_future_tapes": len(teaching), "counterfactual_branches": len(pairs),
              "deterministic_skill_changes_on_same_real_prefix": sum(p["skill_changed"] for p in predicted),
              "mean_control_shift": np.mean([p["motor_shift"] for p in predicted]),
              "causal_interventions": interventions, "synthetic_anatomy": anatomy}
    directory = out / "rollouts" / "00000"
    if not directory.exists():
        stash(directory, rows, {"partition": "smoke", "protocol_sha256": spec_hash},
              make_checkpoint(net, optim, rng, 1, "smoke", SMOKE_SEED, spec_hash))
    atomic_json(report_file, report)
    stage_progress(args.out, "smoke_passed", complete_scans=len(rows), windows=results["forced_skills"]["executed_by_skill"][1].item(),
                   revisits=results["forced_skills"]["executed_by_skill"][2].item(),
                   stops=results["forced_skills"]["totals"]["early_stops"],
                   changed_skills=report["deterministic_skill_changes_on_same_real_prefix"],
                   distinct_layouts=anatomy["distinct_layouts"], recoverable_windows=anatomy["recoverable_windows"])


def validate(executor, args, arm, weight, threshold, directory, slot, completed):
    target = directory / "validation" / f"{slot:05d}"
    if target.exists():
        rows = list(read_lines(target / "episodes.jsonl.gz"))
        return summarize(rows)
    cases = list(run_cases(args.validation, "development", DEV_SEED))
    jobs = [BatchJob(case, (Arm(arm, arm),), {arm: weight}, threshold) for case in cases]
    rows = harvest(executor, jobs)
    stash(target, rows, {"partition": "development", "training_complete_scans": completed},
          extra=False)
    metrics = summarize(item.metrics for _, _, item in rows)
    atomic_json(directory / "validation" / f"{slot:05d}.json", metrics)
    return metrics


def train_arm(executor, args, spec_hash, arm, seed, threshold):
    import torch
    from policy import update_ppo
    from teacher import fit_teacher

    directory = args.out / "training" / f"seed_{seed}" / arm
    directory.mkdir(parents=True, exist_ok=True)
    latest = directory / "complete.json"
    if latest.is_file():
        data = read_json(latest)
        if data["protocol_sha256"] != spec_hash:
            raise RuntimeError("Completed arm uses a different protocol")
        return data
    paths = list(shards(directory / "batches"))
    net, optim = policy_instance(seed + ARM_NAMES.index(arm))
    rng = np.random.default_rng(np.random.SeedSequence([seed, ARM_NAMES.index(arm), 216]))
    cursor = 0
    if paths:
        cursor = restore(paths[-1] / "checkpoint.pt", net, optim, rng, arm, seed, spec_hash)
        assert cursor == sum(read_json(path / "commit.json")["episodes"] for path in paths)
        if cursor % VALIDATION_SPACING == 0 or cursor == args.target:
            weight = rollout_weights(net, directory / "rollout_weights", f"{cursor:07d}")
            slot = (cursor + VALIDATION_SPACING - 1) // VALIDATION_SPACING
            validate(executor, args, arm, weight, threshold, directory, slot, cursor)
    stage_progress(args.out, "training_arm_started", seed=seed, arm=arm, completed=cursor, target=args.target)
    current = Arm(arm, arm, sampled=True)
    for index in range(len(paths), (args.target + args.batch - 1) // args.batch):
        begun = time.monotonic()
        size = min(args.batch, args.target - cursor)
        weight = rollout_weights(net, directory / "rollout_weights", f"{cursor:07d}")
        cases = list(run_cases(size, "training", seed, cursor))
        mask = set()
        if arm == "teacher_ppo":
            # Independent choice of which actual student episodes are eligible for replays.
            teacher_rng = np.random.default_rng(np.random.SeedSequence([seed, cursor, 1204]))
            mask = set(map(int, teacher_rng.choice(size, min(size, TEACH_PER_UPDATE), replace=False)))
        jobs = [BatchJob(case, (current,), {arm: weight}, threshold, teach=i in mask)
                for i, case in enumerate(cases)]
        examples = harvest(executor, jobs)
        collected = time.monotonic()
        if len(examples) != size or any(not row.metrics["complete"] for _, _, row in examples):
            raise RuntimeError("Incomplete batch. No count will be committed.")
        previous = net.skill_high.weight.detach().clone()
        ppo = update_ppo(net, optim, [item.steps for _, _, item in examples], rng,
                         epochs=3, minibatch=512, discount=.996, gae_decay=.96)
        lessons = [item.teaching for _, _, item in examples
                   if item.teaching and item.teaching["eligible_prefix"]]
        teacher = fit_teacher(net, optim, lessons)
        torch.cuda.synchronize()
        if torch.equal(previous, net.skill_high.weight):
            raise RuntimeError("A complete rollout batch made no CUDA actor update")
        next_cursor = cursor + size
        snapshot = make_checkpoint(net, optim, rng, next_cursor, arm, seed, spec_hash)
        train_time = time.monotonic() - collected
        summary = summarize(item.metrics for _, _, item in examples)
        shard = directory / "batches" / f"{index:05d}"
        commit = stash(shard, examples,
                       {"partition": "training", "arm": arm, "seed": seed, "batch": index,
                        "start_episode": cursor, "stop_episode": next_cursor,
                        "policy_sha256": digest(weight), "protocol_sha256": spec_hash,
                        "motion_source_sha256": digest(ROOT / "src" / "motion.py"),
                        "rollouts_seconds": collected - begun, "gpu_update_seconds": train_time,
                        "ppo": ppo, "teacher": teacher, "outcomes": summary, "cuda": cuda_state(net, optim)},
                       snapshot)
        cursor = next_cursor
        stage_progress(args.out, "training_shard_committed", seed=seed, arm=arm, batch=index + 1,
                       complete_scans=cursor, target=args.target, batch_frames=commit["frames"],
                       batch_seconds=round(time.monotonic() - begun, 3),
                       strict=summary["totals"]["pass"], correct_regions=summary["totals"]["correct_regions"],
                       window=summary["executed_by_skill"][1].item(), revisit=summary["executed_by_skill"][2].item(),
                       early_stop=summary["totals"]["early_stops"], teacher_prefixes=len(lessons))
        if cursor % VALIDATION_SPACING == 0 or cursor == args.target:
            current_weights = rollout_weights(net, directory / "rollout_weights", f"{cursor:07d}")
            # The final cursor can fall between nominal validation milestones.
            slot = (cursor + VALIDATION_SPACING - 1) // VALIDATION_SPACING
            vals = validate(executor, args, arm, current_weights, threshold, directory, slot, cursor)
            stage_progress(args.out, "fixed_development_check", seed=seed, arm=arm, scans=cursor,
                           cases=vals["episodes"], strict=vals["totals"]["pass"],
                           correct_regions=vals["totals"]["correct_regions"], wrong=vals["totals"]["wrong_delivery"])
    verified = recount(directory / "batches", arm)
    if verified["complete_unique"] != args.target or verified["excluded"]:
        raise RuntimeError(f"Counted rollout target mismatch for {arm}: {verified}")
    data = {"arm": arm, "seed": seed, "summary": training_summary(directory / "batches"),
            "recount": verified, "policy": rollout_weights(net, directory / "rollout_weights", f"{cursor:07d}"),
            "last_checkpoint": str((list(shards(directory / "batches"))[-1] / "checkpoint.pt")),
            "protocol_sha256": spec_hash}
    data["policy_sha256"] = digest(data["policy"])
    data["checkpoint_sha256"] = digest(data["last_checkpoint"])
    atomic_json(latest, data)
    stage_progress(args.out, "training_arm_complete", seed=seed, arm=arm, complete_scans=verified["complete_unique"])
    return data


def train(args):
    protect_holdout()
    args.out.mkdir(parents=True, exist_ok=True)
    gpu = gpu_manifest(args.out)
    plan = protocol(args, gpu)
    protocol_file = args.out / "protocol.json"
    check_protocol(plan, protocol_file)
    spec_hash = digest(protocol_file)
    preflight = args.out / "preflight"
    if not (preflight / "topology_and_splits.json").is_file():
        atomic_json(preflight / "topology_and_splits.json",
                    {"topology": verify_topology(args.pack), "split_seed_audit": verify_split(args.target, args.validation)})
    stage_progress(args.out, "protocol_recorded", target_per_arm_per_seed=args.target,
                   replicates=args.repeats, workers=args.workers)
    with pool(args) as executor:
        threshold = calibrate(args, executor)
        smoke(args, executor, threshold, spec_hash)
        if args.prepare_only:
            return
        weights = {}
        for seed in TRAIN_SEEDS[:args.repeats]:
            for arm in reversed(ARM_NAMES):
                name = f"{arm}_{seed}"
                weights[name] = train_arm(executor, args, spec_hash, arm, seed, threshold)
    if args.target < TARGET:
        stage_progress(args.out, "short_rehearsal_complete", scans_per_arm=args.target)
        return
    freeze_file = args.out / "freeze.json"
    records = {
        "utc": today(), "frozen": True, "protocol_sha256": spec_hash, "threshold": threshold,
        "source_sha256": source_hashes(), "gpu": gpu, "runs": weights,
        "reader_calibration_sha256": digest(args.out / "development" / "same_track_reader.json"),
        "selection": "Final complete on-policy checkpoint per arm and seed, specified before training",
        "test_seed": TEST_SEED, "test_cases": [asdict(case) for case in test_cases()],
        "test_arms": [asdict(arm) for arm in test_arms(weights)],
        "frozen_budget_scales": [1., .47],
    }
    repair = preflight / "source_sector_wrap_repair.json"
    if repair.is_file():
        records["source_repair"] = read_json(repair) | {"record_sha256": digest(repair)}
    if not freeze_file.exists():
        atomic_json(freeze_file, records)
    stage_progress(args.out, "training_frozen", arms=len(weights), threshold=threshold,
                   unique_scans_by_arm={name: row["recount"]["complete_unique"] for name, row in weights.items()})


def test_arms(weights):
    main = TRAIN_SEEDS[0]
    student, teacher = (f"{name}_{main}" for name in ARM_NAMES)
    arms = [
        Arm("mpc"), Arm("active_mpc"), Arm("ppo", student), Arm("teacher_ppo", teacher),
        Arm("teacher_fixed_params", teacher, fixed_control=True),
        Arm("teacher_no_frame_feedback", teacher, feedback=False),
        Arm("teacher_late_opportunity", teacher, event_timing=True),
        Arm("teacher_no_early_exit", teacher, stop=False),
        Arm("teacher_no_material_position", teacher, position=False),
        Arm("teacher_no_dird", teacher, dird=False),
    ]
    return arms + [Arm(key, key) for key in weights if key not in (student, teacher)]


def freeze_check(args):
    path = args.out / "freeze.json"
    frozen = read_json(path)
    if not frozen["frozen"] or digest(args.out / "protocol.json") != frozen["protocol_sha256"]:
        raise RuntimeError("Evaluation requires verified frozen training data")
    if frozen["source_sha256"] != source_hashes():
        raise RuntimeError("Evaluators require the recorded frozen source")
    for row in frozen["runs"].values():
        if row["recount"]["complete_unique"] < TARGET or row["recount"]["excluded"]:
            raise RuntimeError("Official held-out evaluation requires the full rollout target")
        if digest(row["policy"]) != row["policy_sha256"] or digest(row["last_checkpoint"]) != row["checkpoint_sha256"]:
            raise RuntimeError("Frozen student weights or checkpoint hash differs")
    plan = read_json(args.out / "protocol.json")
    for path, expected in plan["data_sha256"].items():
        if digest(path) != expected:
            raise RuntimeError(f"Frozen public model input changed: {path}")
    return frozen


def eval_group(executor, args, frozen, arms, cases, budget, directory):
    directory.mkdir(parents=True, exist_ok=True)
    weight = {name: row["policy"] for name, row in frozen["runs"].items()}
    for slot, low in enumerate(range(0, len(cases), 16)):
        path = directory / f"{slot:05d}"
        if path.exists():
            continue
        jobs = [BatchJob(case, tuple(arms), weight, frozen["threshold"], budget_scale=budget)
                for case in cases[low:low + 16]]
        rows = harvest(executor, jobs, chunksize=1)
        if not all(row.metrics["complete"] for _, _, row in rows):
            raise RuntimeError("At least one acceptance rollout did not reach the final station")
        stash(path, rows, {"budget_scale": budget, "frame_clock_s": motion.CLOCK_S, "partition": "blender"})
        stage_progress(args.out, "heldout_group_committed", budget=budget, cases=len(jobs), arm_scans=len(rows),
                       index=slot + 1, total_groups=(len(cases) + 15) // 16)
    return [row for path in shards(directory) for row in read_lines(path / "episodes.jsonl.gz")]


def evaluate(args):
    frozen = freeze_check(args)
    out = args.out / "evaluation"
    out.mkdir(exist_ok=True)
    # This is the first access to held-out experiment assets.
    atomic_json(out / "holdout_inputs.json", {
        str(p): digest(p) for p in (BLENDER, BLENDER_LINKS,
                                    Path("/media/camp/PEI_T7/SMPLX_Vascular_Retarget_Audit/evidence/geometry_stages.npz"))})
    cases = [type(next(test_cases()))(**item) for item in frozen["test_cases"]]
    arms = [Arm(**row) for row in frozen["test_arms"]]
    with pool(args, holdout=True) as executor:
        wide_rows = eval_group(executor, args, frozen, arms, cases, 1., out / "wide")
        narrower_rows = eval_group(executor, args, frozen, arms[:4], cases, .47, out / "reduced")
    base = [(a, "mpc") for a in ("active_mpc", "ppo", "teacher_ppo")]
    base += [("ppo", "active_mpc"), ("teacher_ppo", "active_mpc"), ("teacher_ppo", "ppo")]
    comparisons = base + [(a.name, "teacher_ppo") for a in arms[4:10]]
    comparisons += [(a.name, "active_mpc") for a in arms[10:]]
    acceptance = {
        "freeze_sha256": digest(args.out / "freeze.json"), "test_cases": len(cases), "test_anatomies": 1,
        "wide": paired(wide_rows, [a.name for a in arms], comparisons),
        "reduced_budget": paired(narrower_rows, [a.name for a in arms[:4]], base),
        "same_track_development": read_json(args.out / "development" / "same_track_reader.json"),
        "training_arm_counts": {name: item["recount"] for name, item in frozen["runs"].items()},
        "notes": ["Bootstrap resamples entire random bands across all tasks and both measurement profiles.",
                  "The one Blender anatomy had appeared in historical diagnosis. Test tapes were frozen anew.",
                  "Geometric contact/speed checks are applied on the continuous two-dimensional surface chart.",
                  "The atlas shaft material projection and calibrated physical robot frames are recorded separately."],
    }
    atomic_json(out / "acceptance.json", acceptance)
    render_report(args.out / "REPORT.txt", frozen, acceptance)
    stage_progress(args.out, "independent_acceptance_complete", cases_per_arm=len(cases), arms=len(arms),
                   reduced_cases_per_arm=len(cases), report=str(args.out / "REPORT.txt"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "evaluate"))
    parser.add_argument("--out", type=Path, default=ROOT)
    parser.add_argument("--pack", type=Path, default=CT_PACK)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--target", type=int, default=TARGET)
    parser.add_argument("--repeats", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--validation", type=int, default=DEV_COUNT)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.pack = args.pack.resolve()
    if args.batch <= 0 or args.validation <= 0 or args.target <= 0 or args.workers <= 0:
        parser.error("Batch, target, validation and workers must be positive")
    if args.stage == "train" and VALIDATION_SPACING % args.batch:
        parser.error("Batch size must divide the fixed development validation interval")
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (train if args.stage == "train" else evaluate)(args)


if __name__ == "__main__":
    main()
