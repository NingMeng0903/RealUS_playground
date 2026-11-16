"""Privileged CEM headroom diagnostic for the frozen recovery benchmark.

The CEM evaluator may inspect the hidden world only while searching offline.
The selected two-block ProDMP program is then executed with public 10 Hz
updates and the same informative MPC tail as the comparator.  This is a
world-oracle headroom upper bound, not a deployable policy.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
RECOVERY = HERE.parent / "ab_recovery_training_20261115"
LEGACY = HERE.parent / "ab_streaming_prodmp_two_vessel_20261113"
sys.path[:0] = [str(RECOVERY), str(LEGACY)]

from controller import (  # noqa: E402
    CENTER_MM, END, FRAME_COUNT, FRAMES_PER_BLOCK, IDENTITY_ACCEPT,
    IDENTITY_STATION, Controller, Record, State, fork,
)
from world import World  # noqa: E402


PARAM_LOW = np.array([-6.0, -8.5, -6.0, -8.5], dtype=float)
PARAM_HIGH = np.array([7.8, 8.5, 7.8, 8.5], dtype=float)
MODE_MEANS = {
    "left": np.array([0., -6., 0., 3.]),
    "right": np.array([0., 6., 0., -3.]),
    "revisit": np.array([-4.8, 0., -4.8, 0.]),
    "advance": np.array([4.2, 0., 4.2, 0.]),
}
MODE_STD = np.array([1.8, 2.8, 1.8, 2.8], dtype=float)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def public_state(state: State) -> dict:
    out = dict(vars(state))
    out["belief"] = np.asarray(state.belief, dtype=float).tolist()
    return _jsonable(out)


def public_packet(row: dict) -> dict:
    """Archive only packet fields available to the controller."""
    return {
        "frame": int(row["frame"]), "s": float(row["s"]),
        "x": float(row["x"]), "q": float(row["q"]),
        "y": _jsonable(row["y"]),
        "association": _jsonable(row["association"]),
        "accepted_slot": int(row["accepted_slot"]),
    }


def public_archive(record: Record, *, packet_offset=0, motion_offset=0) -> dict:
    return {
        "packet_offset": int(packet_offset),
        "motion_offset": int(motion_offset),
        "packets": [public_packet(p) for p in record.packets[packet_offset:]],
        "motion_t_s_x_vs_vx_q": _jsonable(record.motion[motion_offset:]),
        "decisions": _jsonable(record.decisions),
        "blocks": _jsonable(record.blocks),
        "mpc_revisions": int(record.revisions),
    }


def first_checkpoint(ctrl: Controller, scene):
    """Stop at the first trigger using public state/belief only."""
    state, record = ctrl.initial(scene, False)
    ctrl.finish(scene, state, record, False, allow_recover=False,
                stop_at_trigger=True, informative=True)
    return state, record


def checkpoint_reason(state: State) -> str:
    if state.missed > 0:
        return "loss"
    if state.ambiguous > 0 or .12 < state.query_identity < .88:
        return "identity_ambiguity"
    return "recovery_trigger"


def find_case(world: World, ctrl: Controller, ordinal: int):
    # Public trigger is the only selection criterion; hidden truth is never
    # used to choose a checkpoint.
    for attempt in range(400):
        seed = 4_100_000 + ordinal * 1000 + attempt
        scene = world.scene(seed, "train", FRAME_COUNT, False, alias=False)
        state, record = first_checkpoint(ctrl, scene)
        if state.trigger():
            prefix_details = record.details(state)
            return scene, state, record, {
                "attempt": attempt,
                "reason": checkpoint_reason(state),
                "prefix_wrong": bool(prefix_details["wrong"]),
                "prefix_wrong_certification": bool(prefix_details["wrong_certification"]),
                "prefix_ever_wrong": bool(prefix_details["wrong"] or
                                           prefix_details["wrong_certification"]),
            }
    raise RuntimeError(f"no public recovery trigger for case {ordinal}")


def candidate(ctrl: Controller, state: State, ds: float, dx: float):
    """Create one legal, relative ProDMP segment from public state."""
    goal_s = float(np.clip(state.s + ds, 2., END))
    goal_x = float(np.clip(state.x + dx, -30., 0.))
    # The side primitive supplies the frozen ProDMP shape; the goal is free.
    return ctrl.candidate(state, "left", goal_s, 0., goal_x, False)


def public_ready(ctrl: Controller, state: State) -> bool:
    if state.slot < 0 or abs(float(state.residual)) > CENTER_MM:
        return False
    if state.missed or state.ambiguous:
        return False
    p = ctrl.world.identity_prob(state.belief, IDENTITY_STATION,
                                 state.angle, alias=False)
    return max(p, 1. - p) >= IDENTITY_ACCEPT


def run_program(ctrl: Controller, scene, before: State, prefix: Record,
                theta: np.ndarray):
    state, record = fork(before, prefix)
    packet_offset, motion_offset = len(prefix.packets), len(prefix.motion) - 1
    segments, invalid = [], None
    for block in range(2):
        if state.frame + FRAMES_PER_BLOCK >= FRAME_COUNT:
            break
        ds, dx = float(theta[2 * block]), float(theta[2 * block + 1])
        c = candidate(ctrl, state, ds, dx)
        if c is None:
            invalid = {"segment": block, "requested": [ds, dx],
                       "reason": "rejected_by_frozen_constraints"}
            break
        old_s, old_x = state.s, state.x
        ctrl.execute(scene, state, record, c, False)
        segments.append({
            "segment": block, "requested": [ds, dx],
            "effective": [float(c.s[-1] - old_s), float(c.lateral.goal - old_x)],
            "goal_s": float(c.s[-1]), "goal_x": float(c.lateral.goal),
            "distance_mm": float(c.distance), "turn_deg": float(c.turn),
            "end_frame": int(state.frame),
            "public_ready": bool(public_ready(ctrl, state)),
        })
        if block == 0 and public_ready(ctrl, state):
            break
    # Fair handoff: same public feedback and informative MPC tail for both.
    ctrl.finish(scene, state, record, False, allow_recover=True,
                informative=True)
    return state, record, {"segments": segments, "invalid": invalid,
                           "packet_offset": packet_offset,
                           "motion_offset": motion_offset}


def measures(ctrl: Controller, record: Record, state: State) -> dict:
    details = record.details(state)
    trace = np.asarray(record.motion, dtype=float)
    tcp, turn = ctrl.geometry.cost(trace[:, 1], trace[:, 2])
    return {
        "strict_success": bool(details["complete"]),
        "ever_wrong": bool(details["wrong"] or details["wrong_certification"]),
        "wrong": bool(details["wrong"]),
        "wrong_certification": bool(details["wrong_certification"]),
        "coverage": int(details["coverage"]),
        "public_coverage": int(details["public_coverage"]),
        "arrived": bool(details["arrived"]),
        "tcp_mm": float(tcp), "turn_deg": float(turn),
        "seconds": float(details["seconds"]), "frames": int(details["frames"]),
        "revisions": int(details["mpc_revisions"]),
    }


def utility(measure: dict) -> float:
    return (4. * float(measure["strict_success"])
            - 3. * float(measure["ever_wrong"])
            - .003 * float(measure["tcp_mm"])
            - .002 * float(measure["seconds"]))


def evaluate(world, ctrl, scene, before, prefix, theta, replica_ids, *, detailed=False):
    rows = []
    for replica in replica_ids:
        future = world.resample_tail(scene, before.frame, int(replica))
        state, record, info = run_program(ctrl, future, before, prefix, theta)
        m = measures(ctrl, record, state)
        valid = info["invalid"] is None
        row = {"replica": int(replica), "measure": m,
                     "valid_program": bool(valid),
                     # Retain invalid/fallback outcomes, but never allow one
                     # to masquerade as a positive CEM demonstration.
                     "utility": float(utility(m)) if valid else -1.0e9,
                     "program": _jsonable(info)}
        if detailed:
            row["end_public_state"] = public_state(state)
            row["actual_public_trace"] = public_archive(
                record, packet_offset=len(prefix.packets),
                motion_offset=len(prefix.motion) - 1)
        rows.append(row)
    # Do not renormalize away a replica in which the same requested program
    # violates an execution constraint.  Such a program is invalid for the
    # complete paired outcome and receives the finite failure sentinel.
    values = [r["utility"] for r in rows]
    return (float(np.mean(values)) if values else -1.0e9), rows


def public_score(ctrl: Controller, before: State, theta: np.ndarray) -> float:
    """Public-history ranking hook; no hidden evaluator fields are read."""
    c = candidate(ctrl, before, float(theta[0]), float(theta[1]))
    if c is None:
        return -1.0e9
    return float(ctrl.greedy_recovery(before, c, False)
                 + 1.1 * ctrl.info(before, c, False))


def covariance(points: np.ndarray, mean: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.diag(MODE_STD ** 2)
    centered = points - mean
    return centered.T @ centered / max(1, len(points) - 1) + np.eye(4) * .25 ** 2


def cem(world, ctrl, scene, before, prefix, mode, seed, population,
        iterations, search_replicas):
    rng = np.random.default_rng(seed)
    mean = np.clip(MODE_MEANS[mode], PARAM_LOW, PARAM_HIGH).astype(float)
    cov = np.diag(MODE_STD ** 2)
    ids = [1000 + i for i in range(search_replicas)]
    history, all_rows = [], []
    for iteration in range(iterations):
        samples = np.clip(rng.multivariate_normal(mean, cov, size=population),
                          PARAM_LOW, PARAM_HIGH)
        current = []
        seen = {}
        for sample, theta in enumerate(samples):
            value, replicas = evaluate(world, ctrl, scene, before, prefix,
                                       theta, ids)
            effective = None
            if replicas:
                segments = replicas[0]["program"]["segments"]
                effective = [v for segment in segments
                             for v in segment["effective"]]
                effective += [None] * (4 - len(effective))
            key = tuple(np.round(theta, 6).tolist())
            row = {"mode": mode, "iteration": iteration, "sample": sample,
                   "theta": theta.tolist(), "public_score": public_score(ctrl, before, theta),
                   "effective_theta_replica0": effective,
                   "theta_key": key,
                   "duplicate_of_sample": seen.get(key),
                   "utility_mean": value, "replica_ids": ids, "replicas": replicas}
            seen.setdefault(key, sample)
            current.append(row)
            all_rows.append(row)
        current.sort(key=lambda r: r["utility_mean"], reverse=True)
        elite_count = max(2, int(np.ceil(.2 * len(current))))
        elite = np.asarray([r["theta"] for r in current[:elite_count]])
        elite_mean = elite.mean(axis=0)
        mean = np.clip(.65 * elite_mean + .35 * mean, PARAM_LOW, PARAM_HIGH)
        cov = .65 * covariance(elite, elite_mean) + .35 * cov
        history.append({"iteration": iteration,
                        "best_utility": current[0]["utility_mean"],
                        "best_theta": current[0]["theta"],
                        "elite_count": elite_count, "mean": mean.tolist(),
                        "covariance": cov.tolist()})
    best = max(all_rows, key=lambda r: r["utility_mean"])
    return best, history, all_rows


def baseline(world, ctrl, scene, before, prefix, replica_ids):
    rows = []
    for replica in replica_ids:
        future = world.resample_tail(scene, before.frame, int(replica))
        state, record = fork(before, prefix)
        ctrl.finish(future, state, record, False, allow_recover=True,
                    informative=True)
        m = measures(ctrl, record, state)
        rows.append({"replica": int(replica), "measure": m,
                     "utility": float(utility(m)),
                     "end_public_state": public_state(state),
                     "actual_public_trace": public_archive(
                         record, packet_offset=len(prefix.packets),
                         motion_offset=len(prefix.motion) - 1)})
    return rows


def run(args) -> dict:
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    sections = json.loads((LEGACY / "SECTIONS.json").read_text(encoding="utf-8"))
    coefficients = dict(np.load(RECOVERY / "results" / "prodmp_weights.npz"))
    world, ctrl = World(), None
    ctrl = Controller(world, sections, coefficients)
    started = perf_counter()
    summaries = []
    archive_path = out / "oracle_cem_cases.jsonl.gz"
    with gzip.open(archive_path, "wt", encoding="utf-8", compresslevel=2) as stream:
        for case_index in range(args.cases):
            scene, before, prefix, meta = find_case(world, ctrl, case_index)
            search_ids = [1000 + i for i in range(args.search_replicas)]
            incumbent_rows = baseline(world, ctrl, scene, before, prefix, search_ids)
            incumbent_value = float(np.mean([r["utility"] for r in incumbent_rows]))
            incumbent = {
                "mode": "full_mpc_incumbent", "iteration": -1, "sample": -1,
                "theta": None, "public_score": None,
                "utility_mean": incumbent_value, "replica_ids": search_ids,
                "replicas": incumbent_rows, "is_incumbent": True,
            }
            results = []
            for mode_index, mode in enumerate(MODE_MEANS):
                best, history, candidates = cem(
                    world, ctrl, scene, before, prefix, mode,
                    73_001 + 101 * case_index + mode_index,
                    args.population, args.iterations, args.search_replicas)
                results.append({"mode": mode, "best": best, "history": history,
                                "candidate_count": len(candidates),
                                "candidates": candidates})
            # The original full MPC is an explicit incumbent.  Invalid CEM
            # programs can fall back to the MPC tail, so they are never
            # allowed to outrank this incumbent as demonstrations.
            results.append({"mode": "full_mpc_incumbent", "best": incumbent,
                            "history": [], "candidate_count": 1,
                            "candidates": [incumbent]})
            chosen = max(results, key=lambda r: r["best"]["utility_mean"])
            theta = (None if chosen["best"].get("is_incumbent") else
                     np.asarray(chosen["best"]["theta"], dtype=float))
            verify_ids = [9000 + 100 * case_index + i
                          for i in range(args.verify_replicas)]
            base_rows = baseline(world, ctrl, scene, before, prefix, verify_ids)
            if theta is None:
                selected_rows = base_rows
                selected_value = float(np.mean([r["utility"] for r in selected_rows]))
            else:
                selected_value, selected_rows = evaluate(
                    world, ctrl, scene, before, prefix, theta, verify_ids,
                    detailed=True)
            all_thetas = [np.asarray(row["theta"], dtype=float)
                          for result in results for row in result["candidates"]
                          if row.get("theta") is not None]
            public_order = [{"index": i, "score": public_score(ctrl, before, t)}
                            for i, t in enumerate(all_thetas)]
            row = {
                "case": case_index, "seed": int(scene.seed),
                "checkpoint_reason": meta["reason"],
                "seed_attempt": int(meta["attempt"]),
                "checkpoint_frame": int(before.frame),
                "prefix_wrong": bool(meta["prefix_wrong"]),
                "prefix_wrong_certification": bool(meta["prefix_wrong_certification"]),
                "headroom_eligible": not bool(meta["prefix_ever_wrong"]),
                "checkpoint_public_state": public_state(before),
                "checkpoint_public_history": public_archive(prefix),
                "chosen_mode": chosen["mode"],
                "chosen_theta": None if theta is None else theta.tolist(),
                "selected_search_utility": float(chosen["best"]["utility_mean"]),
                "selected_verify_utility": float(selected_value),
                "baseline_name": "complete_mpc_informative_true",
                "baseline_verify": base_rows, "selected_verify": selected_rows,
                "public_rerank": public_order,
                "cem": {"population": args.population,
                        "iterations": args.iterations,
                        "search_replica_ids": search_ids,
                        "verify_replica_ids": verify_ids,
                        "parameter_order": ["seg0_delta_s", "seg0_delta_x",
                                             "seg1_delta_s", "seg1_delta_x"],
                        "parameter_low": PARAM_LOW.tolist(),
                        "parameter_high": PARAM_HIGH.tolist(),
                        "results": results},
                "oracle_role": "world_oracle_offline_headroom_only",
                "truth_free_deployable_action": False,
            }
            stream.write(json.dumps(_jsonable(row), allow_nan=False,
                                    separators=(",", ":")) + "\n")
            base_success = sum(r["measure"]["strict_success"] for r in base_rows)
            selected_success = sum(r["measure"]["strict_success"] for r in selected_rows)
            summaries.append({
                "case": case_index, "seed": int(scene.seed),
                "checkpoint_frame": int(before.frame), "reason": meta["reason"],
                "prefix_wrong": bool(meta["prefix_wrong"]),
                "headroom_eligible": not bool(meta["prefix_ever_wrong"]),
                "mode": chosen["mode"],
                "theta": None if theta is None else theta.tolist(),
                "baseline_success": f"{base_success}/{len(base_rows)}",
                "selected_success": f"{selected_success}/{len(selected_rows)}",
                "baseline_wrong": int(sum(r["measure"]["ever_wrong"] for r in base_rows)),
                "selected_wrong": int(sum(r["measure"]["ever_wrong"] for r in selected_rows)),
                "selected_search_utility": float(chosen["best"]["utility_mean"]),
                "selected_verify_utility": float(selected_value),
            })
    summary = {
        "protocol": "oracle_cem_headroom_v1",
        "oracle_role": "privileged_world_oracle_offline_search",
        "public_history_rerank_hook": "public_score_single_step_only",
        "baseline": "complete MPC: public 10Hz observe/recondition, informative=True, allow_recover=True",
        "intervention": "up to two 1s relative s/x ProDMP segments, then identical MPC tail",
        "parameter_order": ["seg0_delta_s_mm", "seg0_delta_x_mm",
                            "seg1_delta_s_mm", "seg1_delta_x_mm"],
        "parameter_low": PARAM_LOW.tolist(), "parameter_high": PARAM_HIGH.tolist(),
        "cases": summaries, "archive": str(archive_path),
        "elapsed_s": float(perf_counter() - started),
        "limitations": [
            "CEM reads hidden evaluator fields only during offline search; it is not a deployable policy.",
            "The comparison is an expanded continuous-action/search upper bound until MPC receives the same action library.",
            "Record.details strict success permanently counts any earlier wrong acceptance as failure; correction after wrong submission needs a new ledger and is not tested.",
            "No hidden transition coordinate, role label, or target packet selects an action.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2,
                                                 allow_nan=False) + "\n",
                                        encoding="utf-8")
    print(json.dumps(_jsonable(summary), indent=2, allow_nan=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--search-replicas", type=int, default=2)
    parser.add_argument("--verify-replicas", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if min(args.cases, args.population, args.iterations,
           args.search_replicas, args.verify_replicas) < 1:
        parser.error("counts must be positive")
    run(args)


if __name__ == "__main__":
    main()
