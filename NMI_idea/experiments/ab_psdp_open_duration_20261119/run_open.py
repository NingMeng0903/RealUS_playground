"""Replay frozen scan policies with an open-ended disturbance stream and no time cap."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ab_psdp_b_20261119"))
import run_psdp as study
import benchmark as b
from policy import MODELS, p, load_references

from adaptive_controller import EDGES, END, FRAME_COUNT
from diagnose_b_selection import run_parallel, save_archive
from recovery_world import noise_tape
from run_learned_recovery import write_json


ARMS = ("mpc", "random_data", "evidence_data", "contrast_data", "legacy_program",
        "contrast_child", "psdp", "psdp_direct")
PRIMARY = b.PRIMARY
FIRST_FRAME_EXTENSION = FRAME_COUNT
LAST_LEGACY_TRIGGER = FRAME_COUNT - 1 - 2 * p.FRAMES_PER_BLOCK
SAFETY_TOL = 1.e-7


class ExtendedColumn:
    def __init__(self, tape, field):
        self.tape = tape
        self.field = field

    def __getitem__(self, frame):
        tape, index = self.tape.chunk(frame)
        return getattr(tape, self.field)[index]


class ExtendedTape:
    """Preserve the entire recorded prefix and extend only as queried."""

    def __init__(self, tape, seed, index):
        self.original = tape
        self.seed = seed
        self.replica = index
        self.chunks = {}
        for field in ("emit", "error", "uniform_clutter", "outage"):
            setattr(self, field, ExtendedColumn(self, field))

    def chunk(self, frame):
        if frame < FIRST_FRAME_EXTENSION:
            return self.original, frame
        n, index = divmod(frame - FIRST_FRAME_EXTENSION, FRAME_COUNT)
        if n not in self.chunks:
            rng = np.random.default_rng([self.seed, self.replica, b.TEST_PURPOSE, 24236, n])
            tape = noise_tape(rng)
            self.chunks[n] = replace(tape, error=tape.error * p.base.scans.NOISE_SCALE)
        return self.chunks[n], index


def task_ready(state):
    return state.s >= END - .065 and not state.uncovered()


def move_toward_gap(c, state):
    """Choose among legal repositions when the ordinary policy repeats a static command."""
    choices = {name: action for name, action in c.options_shared(state, expanded=True, renderer="fit").items()
               if action.distance > SAFETY_TOL}
    if not choices:
        return None
    query = state.query_bin()
    target = float(np.mean(EDGES[query:query + 2])) if state.uncovered() else END
    departure = {name: abs(action.s[-1] - target) for name, action in choices.items()}
    best = min(departure.values())
    nearest = {name: action for name, action in choices.items() if departure[name] <= best + SAFETY_TOL}
    return c.mpc(state, nearest, False, active=True, informative=True)


def finish(c, state, record, scene, gate, mode):
    state, record = p.copy_state(state), record.copy()
    last_stationary = None
    recovery_moves = 0
    while True:
        if task_ready(state):
            return state, record, "public_task_complete", recovery_moves
        if mode == "original" and not c.can_continue(state):
            return state, record, "legacy_time_limit", recovery_moves
        open_decision = state.frame < gate
        if mode == "open" and state.frame > LAST_LEGACY_TRIGGER:
            # Keep the usual evidence condition usable after the old timer.
            open_decision |= replace(state, frame=LAST_LEGACY_TRIGGER).trigger()
        action = c.choose_mpc(state, expanded=True, open_decision=open_decision)
        if action is None:
            return state, record, "no_feasible_command", recovery_moves
        state_key = (round(state.s, 7), round(state.x, 7), state.covered, state.query_bin())
        if state_key == last_stationary and action.distance <= SAFETY_TOL:
            if mode == "extended" and state.frame >= FRAME_COUNT - 1:
                # Retain the legacy trajectory through the old boundary. A
                # repeated stall after it remains an unresolved trial, not a
                # statement that another frame could never change the belief.
                return state, record, "stationary_unresolved", recovery_moves
            if mode == "open":
                action = move_toward_gap(c, state)
                if action is None:
                    return state, record, "no_feasible_reposition", recovery_moves
                recovery_moves += 1
        previous_coverage = state.covered
        c.execute(scene, state, record, action)
        last_stationary = (state_key if record.decisions[-1]["executed_tcp_mm"] <= SAFETY_TOL
                           and state.covered == previous_coverage else None)


def outcome(c, state, record, scene, root, gate, mode):
    revisions_before = record.revisions
    tail_start = len(record.decisions)
    state, log, reason, recovery_moves = finish(c, state, record, scene, gate, mode)
    result = p.score_outcome(state, log, root)
    final = {event["bin"]: event for event in log.certificates}
    revoked = {event["bin"]: event["frame"] for event in log.retractions}
    retained = {j for j, event in final.items() if event["identity_ok"] and event["frame"] > root.frame
                and event["accepted_at"] >= revoked.get(j, -1)}
    result["new_query_image_retained"] = root.query_bin() in retained
    result[PRIMARY] = float(bool(retained) and bool(result["corrected_complete"]))
    if result["max_handoff_error"] > SAFETY_TOL or result["tcp_mm"] > p.MAX_TRAVEL + SAFETY_TOL:
        raise ValueError("Motion safety or continuity check failed")
    result = {k: result[k] for k in p.base.FIELDS}
    tail = log.decisions[tail_start:]
    return result | {
        "reason": reason, "recovery_moves": recovery_moves,
        "terminal_s_mm": state.s, "missing_bins": state.uncovered(),
        "servo_total_after_checkpoint": log.revisions,
        "servo_after_handoff": log.revisions - revisions_before,
        "tail_blocks": len(tail), "tail_wait_blocks": sum(
            block["executed_tcp_mm"] < SAFETY_TOL for block in tail),
        "tail_trace": [[v["frame"], v["action"], v["executed_tcp_mm"], v["executed_goal"]] for v in tail],
    }


def replay_seed(job):
    row, fixed = job
    c, truth, root, prefix, menu, aliases, anchor, cpu = b.legacy_eval.checkpoint(row)
    gate = p.deadline(root)
    full, _ = b.make_plans(c, root, prefix, menu, aliases, anchor, fixed, cpu)
    plans = {arm: full[arm] for arm in ARMS}
    result = {mode: {arm: [] for arm in ARMS} for mode in ("original", "extended", "open")}
    for index, scene in enumerate(p.base.scans.paired_futures(truth, row, b.TEST_TAPES, b.TEST_PURPOSE)):
        scene = replace(scene, tape=ExtendedTape(scene.tape, row["seed"], index))
        reached = {name: p.run_motion(c, root, prefix, scene, menu.get(name)) for name, _ in plans.values()}
        chosen, finished, eval_cache = {}, {}, {}
        for arm, (first, adaptive) in plans.items():
            state, log = reached[first]
            next_stage = ("legacy_child" if arm == "legacy_program" else "child") if adaptive else None
            key = first, next_stage
            if key not in chosen:
                action = None
                if adaptive:
                    _, action = p.choose_second(*p.second_inputs(c, state, log, gate), MODELS[next_stage])
                chosen[key] = action
            second_action = chosen[key]
            second = p.HANDOFF if second_action is None else second_action.name
            if (first, second) not in finished:
                finished[first, second] = p.run_motion(c, state, log, scene, second_action)
            end, end_log = finished[first, second]
            for mode in result:
                key = mode, first, second
                if key not in eval_cache:
                    eval_cache[key] = outcome(c, end, end_log, scene, root, gate, mode)
                trial = eval_cache[key]
                result[mode][arm].append(trial | {
                    "first": first, "second": second, "child_requested": adaptive,
                    "changed_from_mpc_first": first not in (p.HANDOFF, anchor),
                    "probes": int(first != p.HANDOFF) + int(second != p.HANDOFF),
                    "servo_total_after_checkpoint": trial["servo_total_after_checkpoint"] - prefix.revisions,
                    "servo_in_b_probes": end_log.revisions - prefix.revisions,
                })
    return row | {"root_frame": root.frame, "root_triggers_mpc_recovery": root.trigger(),
                  "incumbent": anchor, "outcomes": result}


def validate_reference(rows, split, base):
    original = {r["seed"]: r["outcomes"] for r in p.base.read_rows(base / f"{split}.jsonl.gz")}
    for row in rows:
        for arm in ARMS:
            for actual, expected in zip(row["outcomes"]["original"][arm], original[row["seed"]][arm], strict=True):
                for field in (*p.base.FIELDS, "first", "second", "probes"):
                    if actual[field] != expected[field]:
                        raise ValueError(f"Original replay differs: {row['seed']} {arm} {field}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("test", "drift"))
    parser.add_argument("--workers", type=int, default=26)
    parser.add_argument("--limit-scenes", type=int)
    parser.add_argument("--out", type=Path, default=HERE / "results/main")
    args = parser.parse_args()
    base = study.HERE / "results/main"
    frozen = study.read(base / "frozen.json")
    for folder, key in ((study.BASE, "sources"), (study.HERE, "code"), (base, "files")):
        study.assert_files(folder, frozen[key])
    p.base.configure()
    load_references()
    study.load_child(base)
    MODELS.update(joblib.load(base / "heads.joblib"))
    fixed = study.read(study.LEGACY_RUN / "fixed.json")
    rows = study.scenario_rows(*study.HOLDOUT[args.split], setting=args.split)[:args.limit_scenes]
    cases = run_parallel(replay_seed, [(r, fixed[r["family"]]["single"][0]) for r in rows],
                         args.workers, f"{args.split} replay")
    validate_reference(cases, args.split, base)
    args.out.mkdir(parents=True, exist_ok=True)
    save_archive(args.out / f"{args.split}.jsonl.gz", cases)
    write_json(args.out / f"{args.split}_provenance.json", {
        "frozen_model_manifest": p.base.fingerprint(base / "frozen.json"),
        "source_score_archive": p.base.fingerprint(base / f"{args.split}.jsonl.gz"),
        "source_code": p.base.fingerprint(Path(__file__)),
        "archive_sha256": p.base.fingerprint(args.out / f"{args.split}.jsonl.gz"),
        "archived_capped_trials_matched": len(cases) * b.TEST_TAPES * len(ARMS),
    })


if __name__ == "__main__":
    main()
