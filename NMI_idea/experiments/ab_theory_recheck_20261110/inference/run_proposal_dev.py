"""Reproduce the bounded 28D public-history proposal DEV record.

This driver is intentionally DEV-only.  It evaluates the same first
``sample_worlds(720902, 1)`` object under iid and shifted conditions, at
prefixes 5 and 8, using the public sensing history only for inference.  The
private world is retained only by the retrospective predictive/terminal fork
and is represented in the JSON by an audit-only geometry hash.  The output
directory is created only for a fresh run; an existing directory is never
overwritten.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import shlex
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LADDER = ROOT.parent / "ab_module_ladder_20261110"
AMONG = ROOT.parent.parent.parent.parent / "Among_US"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(LADDER) not in sys.path:
    sys.path.insert(0, str(LADDER))

from inference import informed_proposal as ip  # noqa: E402
from observation import environment as env  # noqa: E402
import shared_history_experiment as study  # noqa: E402


PRIOR_SEED = 720901
WORLD_SEED = 720902
PATTERN_SEED = 720907
IMPORTANCE_SEED = 720908
MH_SEED = 720909
FIXED_SEED = 720906
N = 128
PREFIXES = (5, 8)
CONDITIONS = ("iid", "shifted")
MH_SCALES = (1.0, 0.5, 1.8)
MH_SWEEPS = 8
RESAMPLE_FRACTION = 0.60


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def serial(value: Any):
    """Convert diagnostics and public/audit forks to strict JSON values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): serial(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(item) for item in value]
    return value


def source_hashes() -> dict[str, str]:
    """Hash every source directly used by this driver and its diagnostic fork."""
    paths = {
        "run_proposal_dev.py": Path(__file__).resolve(),
        "informed_proposal.py": Path(ip.__file__).resolve(),
        "observation_environment.py": Path(env.__file__).resolve(),
        "shared_history_experiment.py": Path(study.__file__).resolve(),
        "planning_system.py": LADDER / "planning" / "system.py",
        "observation_servo.py": LADDER / "observation" / "servo.py",
        "among_imaging.py": AMONG / "DMP_playground" / "kmp_honest" / "imaging.py",
        "among_servo.py": AMONG / "DMP_playground" / "kmp_honest" / "servo.py",
        "study_protocol.txt": ROOT / "STUDY_PROTOCOL.txt",
        "proposal_protocol.txt": ROOT / "inference" / "PROPOSAL_PROTOCOL.txt",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"source manifest file(s) missing: {missing}")
    return {name: sha256(path) for name, path in paths.items()}


def history_pairs(shared: list[dict], prefix: int) -> list[tuple[dict, np.ndarray]]:
    """Flatten only the requested public prefix, preserving executed poses."""
    return [
        (frame, np.asarray(block["pose"], dtype=float).copy())
        for block in shared[:prefix]
        for frame in block["public"]
    ]


def public_history_json(shared: list[dict], prefix: int) -> list[dict]:
    return [
        {"frame": frame, "pose": block["pose"]}
        for block in shared[:prefix]
        for frame in block["public"]
    ]


def audit_history_json(shared: list[dict], prefix: int) -> list[dict]:
    return [
        {"audit": audit, "pose": block["pose"]}
        for block in shared[:prefix]
        for audit in block["audit_for_evaluation_only"]
    ]


def shared_history_json(shared: list[dict]) -> list[dict]:
    """Store the sensing fork without exposing private world coordinates."""
    return [
        {
            "index": i,
            "pose": block["pose"],
            "public": block["public"],
            "audit_for_evaluation_only": block["audit_for_evaluation_only"],
            "motion_s": block["motion_s"],
            "elapsed_s": block["elapsed_s"],
            "shared_policy_compute_s": block["shared_policy_compute_s"],
        }
        for i, block in enumerate(shared)
    ]


def particle_arrays(particles, weights, proposal: ip.InformedProposal):
    vectors = np.asarray([ip.vector_from_world(world) for world in particles], dtype=float)
    modes = np.asarray([world.modes for world in particles], dtype=float)
    logq = np.asarray(proposal.log_q(vectors, modes), dtype=float)
    logp = np.asarray(proposal.log_prior(vectors, modes), dtype=float)
    logll = np.asarray(proposal.log_likelihood_full(particles), dtype=float)
    weights = np.asarray(weights, dtype=float).copy()
    log_weights = np.full_like(weights, -np.inf, dtype=float)
    positive = weights > 0.0
    log_weights[positive] = np.log(weights[positive])
    return {
        "vectors": vectors,
        "modes": modes,
        "weights": weights,
        "log_weights": log_weights,
        "logq": logq,
        "logp": logp,
        "logLL": logll,
    }


def raw_store(raw: dict[str, np.ndarray], condition: str, prefix: int, method: str, arrays):
    base = f"{condition}_p{prefix}_{method}"
    keys = {}
    for field, values in arrays.items():
        key = f"{base}_{field}"
        raw[key] = np.asarray(values)
        keys[field] = key
    return keys


def evaluate_terminal(
    private_world,
    shared: list[dict],
    history: list[tuple[dict, np.ndarray]],
    particles,
    weights,
    method: str,
    condition: str,
    prefix: int,
):
    key = (0, condition, "dev")
    selected_pose, decision = study.choose_terminal(particles, weights, history)
    duration = study.MOTION.duration(shared[prefix - 1]["pose"], selected_pose)
    terminal = study.terminal_evaluation(
        private_world,
        selected_pose,
        key,
        prefix,
        shared[prefix - 1]["elapsed_s"] + duration,
    )
    predictive = study.predictive_metrics(
        particles,
        weights,
        private_world,
        (key, prefix),
    )
    return {
        "selected": selected_pose,
        "decision": decision,
        "predictive": predictive,
        "terminal_public": terminal["public"],
        "terminal_audit_for_evaluation_only": terminal["audit_for_evaluation_only"],
        "public_stop": terminal["public_stop"],
        "audit_quality": terminal["audit_quality"],
        "correct_public_stop": terminal["correct_public_stop"],
        "wrong_public_stop": terminal["wrong_public_stop"],
        "diagnostic_fork_only": terminal["diagnostic_fork_only"],
        "method": method,
        "condition": condition,
        "prefix": prefix,
        "movement_s": duration,
    }


def fixed128_case(
    private_world,
    prior,
    proposal: ip.InformedProposal,
    shared: list[dict],
    condition: str,
    prefix: int,
    raw: dict[str, np.ndarray],
):
    history = history_pairs(shared, prefix)
    belief = env.Belief(list(prior[:N]), seed=FIXED_SEED, support="finite")
    update_diagnostics = []
    start = time.perf_counter()
    for block in shared[:prefix]:
        update_diagnostics.extend(belief.update_block(block["public"], block["pose"]))
    inference_s = time.perf_counter() - start
    arrays = particle_arrays(belief.particles, belief.weights, proposal)
    raw_keys = raw_store(raw, condition, prefix, "fixed128", arrays)
    record = evaluate_terminal(
        private_world,
        shared,
        history,
        belief.particles,
        belief.weights,
        "fixed128",
        condition,
        prefix,
    )
    record.update(
        {
            "valid_posterior": True,
            "inference_s": inference_s,
            "raw_keys": raw_keys,
            "raw_logq_role": "diagnostic_only; fixed128 was not sampled from q",
            "diagnostics": {
                "updates": update_diagnostics,
                "posterior_history_frames": len(belief.history),
                "raw_ess": float(1.0 / np.sum(belief.weights**2)),
                "max_weight": float(np.max(belief.weights)),
                "support": "finite",
                "unique_geometries": len({world.points.tobytes() for world in belief.particles}),
            },
            "public_history": public_history_json(shared, prefix),
            "sensing_audit_for_evaluation_only": audit_history_json(shared, prefix),
        }
    )
    return record


def informed_cases(
    private_world,
    shared: list[dict],
    condition: str,
    prefix: int,
    raw: dict[str, np.ndarray],
):
    history = history_pairs(shared, prefix)
    start = time.perf_counter()
    proposal = ip.InformedProposal(history, pattern_seed=PATTERN_SEED)
    result = proposal.importance_sample(N, seed=IMPORTANCE_SEED)
    importance_s = time.perf_counter() - start
    pre_arrays = particle_arrays(result.worlds, result.weights, proposal)
    pre_keys = raw_store(raw, condition, prefix, "informed28", pre_arrays)
    pre_record = evaluate_terminal(
        private_world,
        shared,
        history,
        result.worlds,
        result.weights,
        "informed28",
        condition,
        prefix,
    )
    pre_record.update(
        {
            "valid_posterior": bool(result.diagnostics["posterior_valid"]),
            "inference_s": importance_s,
            "raw_keys": pre_keys,
            "diagnostics": result.diagnostics,
            "public_history": public_history_json(shared, prefix),
            "sensing_audit_for_evaluation_only": audit_history_json(shared, prefix),
            "proposal_parameters": {
                "epsilon": proposal.epsilon,
                "continuous_dim": ip.CONTINUOUS_DIM,
                "mode_count": len(ip.MODE_OPTIONS),
                "pattern_seed": PATTERN_SEED,
                "pattern_count_requested": proposal.pattern_count,
                "pattern_count_realized": len(proposal.patterns),
                "component_count": len(proposal.components),
                "component_hash": proposal.component_hash,
                "history_hash": proposal.history_hash,
            },
        }
    )

    mh_start = time.perf_counter()
    moved = proposal.rejuvenate(
        result,
        seed=MH_SEED,
        sweeps=MH_SWEEPS,
        scales=MH_SCALES,
        resample_fraction=RESAMPLE_FRACTION,
    )
    mh_total_s = time.perf_counter() - start
    mh_arrays = particle_arrays(moved.worlds, moved.weights, proposal)
    mh_keys = raw_store(raw, condition, prefix, "informed28_mh8", mh_arrays)
    mh_record = evaluate_terminal(
        private_world,
        shared,
        history,
        moved.worlds,
        moved.weights,
        "informed28+MH8",
        condition,
        prefix,
    )
    mh_record.update(
        {
            "valid_posterior": bool(moved.diagnostics["posterior_valid"]),
            "inference_s": mh_total_s,
            "importance_s": importance_s,
            "mh_only_s": time.perf_counter() - mh_start,
            "raw_keys": mh_keys,
            "diagnostics": moved.diagnostics,
            "public_history": public_history_json(shared, prefix),
            "sensing_audit_for_evaluation_only": audit_history_json(shared, prefix),
            "proposal_parameters": {
                "epsilon": proposal.epsilon,
                "continuous_dim": ip.CONTINUOUS_DIM,
                "mode_count": len(ip.MODE_OPTIONS),
                "pattern_seed": PATTERN_SEED,
                "pattern_count_requested": proposal.pattern_count,
                "pattern_count_realized": len(proposal.patterns),
                "component_count": len(proposal.components),
                "component_hash": proposal.component_hash,
                "history_hash": proposal.history_hash,
                "mh_seed": MH_SEED,
                "mh_sweeps": MH_SWEEPS,
                "mh_scales": list(MH_SCALES),
                "resample_fraction": RESAMPLE_FRACTION,
            },
        }
    )
    return pre_record, mh_record


def main() -> int:
    output_dir = Path(__file__).resolve().parent / "proposal_dev_28d"
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing DEV output directory: {output_dir}"
        )
    output_dir.mkdir(parents=False, exist_ok=False)

    prior = env.sample_prior(PRIOR_SEED, N)
    private_world = env.sample_worlds(WORLD_SEED, 1)[0]
    raw: dict[str, np.ndarray] = {}
    cases = []
    source_manifest = source_hashes()
    command = "PYTHONPATH=" + shlex.quote(str(ROOT)) + ":" + shlex.quote(str(LADDER))
    command += " " + shlex.join([str(Path(sys.executable).resolve()), str(Path(__file__).resolve())])

    for condition in CONDITIONS:
        condition_world = private_world if condition == "iid" else study.shifted(private_world)
        key = (0, condition, "dev")
        shared = study.generate_history(condition_world, key)
        case = {
            "object_index": 0,
            "condition": condition,
            "history_key": list(key),
            "private_geometry_hash_for_audit_only": study.geometry_digest(condition_world),
            "shared_history": shared_history_json(shared),
            "rows": [],
        }
        for prefix in PREFIXES:
            history = history_pairs(shared, prefix)
            proposal = ip.InformedProposal(history, pattern_seed=PATTERN_SEED)
            fixed = fixed128_case(
                condition_world, prior, proposal, shared, condition, prefix, raw
            )
            informed, moved = informed_cases(
                condition_world, shared, condition, prefix, raw
            )
            # Keep the method order stable for audit and human comparison.
            case["rows"].extend([fixed, informed, moved])
        cases.append(case)

    raw_path = output_dir / "RAW.npz"
    np.savez_compressed(raw_path, **raw)
    raw_manifest = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in sorted(raw.items())
    }
    results = {
        "record": "public-history informed proposal DEV, corrected 28D",
        "cohort": "dev",
        "scope": "exact iid/shifted x prefix5/prefix8; fixed128, informed28, informed28+MH8",
        "command": command,
        "seeds": {
            "prior": PRIOR_SEED,
            "world": WORLD_SEED,
            "pattern": PATTERN_SEED,
            "importance": IMPORTANCE_SEED,
            "mh": MH_SEED,
            "fixed128": FIXED_SEED,
        },
        "parameters": {
            "N": N,
            "continuous_dim": ip.CONTINUOUS_DIM,
            "epsilon": ip.EPSILON,
            "mode_count": len(ip.MODE_OPTIONS),
            "prefixes": list(PREFIXES),
            "conditions": list(CONDITIONS),
            "mh_sweeps": MH_SWEEPS,
            "mh_scales": list(MH_SCALES),
            "resample_fraction": RESAMPLE_FRACTION,
            "no_hidden_truth_in_inference": True,
            "predictive_and_terminal_are_diagnostic_forks": True,
            "wallclock_is_not_an_acceptance_metric": True,
        },
        "source_hashes": source_manifest,
        "private_world_coordinates_not_recorded": True,
        "shared_history_contains_public_and_audit_forks": True,
        "raw_npz": "RAW.npz",
        "raw_npz_sha256": sha256(raw_path),
        "raw_arrays": raw_manifest,
        "cases": cases,
    }
    (output_dir / "RESULTS.json").write_text(
        json.dumps(serial(results), indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "output": str(output_dir),
        "raw_sha256": results["raw_npz_sha256"],
        "rows": sum(len(case["rows"]) for case in cases),
        "source_hashes": source_manifest,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
