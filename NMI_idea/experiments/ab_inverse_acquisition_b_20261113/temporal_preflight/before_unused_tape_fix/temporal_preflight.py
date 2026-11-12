"""Causal prior-only check of stationary holds after one/two MPC captures.

This computes public expected claim mass and the shared nominal tail surrogate.
It is neither a terminal-outcome census nor a comparison to time-aware MPC.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import world as extension

NAMESPACE = 719920
HOLD_TICKS = (0, 1, 2)


def run(args):
    began = time.perf_counter()
    acoustics, control, shared = extension.load(args.repo)
    sys.path.insert(0, str(args.repo / "NMI_idea/experiments/ab_continuation_b_20261113"))
    import continuations as cont
    cont.load_shared(args.repo)
    shared.initialize(args.repo, args.bank, 256, 6, "joint")
    rows, parity = [], {}
    for task, data in shared.DATA.items():
        original = data["scene"]
        clocks = np.arange(19) * .5
        like, correct, detect, cdf = acoustics.model_tables(data["geometry"], clocks, 256, "joint")
        parity[task] = dict(likelihood_max_abs=float(np.max(np.abs(like[::3]-original.likelihood))),
                            correct_max_abs=float(np.max(np.abs(correct[::3]-original.correct))))
        assert max(parity[task].values()) < 1e-12
        costs = original.edge_costs.copy()
        costs[..., 3] *= 3
        reserve = original.reserve.copy()
        reserve[-1] *= 3
        potential, needs = control.nominal_potential(original.site, costs, original.targets, 6,
                                                     .45 if task == "daughter" else .68)
        scene = replace(original, likelihood=like, correct=correct, detect=detect, cdf=cdf,
                        edge_costs=costs, reserve=reserve, potential=potential, needs=needs, clock_unit=.5)
        uniform = np.full(256, 1./256)
        for index in range(args.n):
            seed = acoustics.seed_for(NAMESPACE, index)
            rng = np.random.default_rng(np.random.SeedSequence([seed, 29102]))
            atom = int(rng.integers(256))
            tape = acoustics.tape(seed, 19)
            tape[::3] = acoustics.tape(seed, 7)
            td = np.ascontiguousarray(detect[:, :, atom:atom+1])
            tc = np.ascontiguousarray(cdf[:, :, atom:atom+1])
            obs, _ = control.capture_draw(td, tc, 0, scene.to_address[scene.start], 0, tape)
            prior, _ = control.posterior(uniform, like, correct, 0, scene.to_address[scene.start], obs)
            initial = prior, prior.copy(), reserve.copy(), scene.start, 0, 0, 0, False
            for depth in (1, 2):
                state, trace = cont.execute(scene.public(), td, tc, tape, initial,
                                            np.empty(0, dtype=np.int64), depth)
                b, good, remaining, at, slot, clock, progress, _ = state
                row = dict(task=task, index=index, depth=depth, model_atom=atom, seed=seed,
                           anchor=int(obs), history=cont.trace_public(trace), state_sha256=cont.state_hash(state),
                           clock_s=float(clock*.5), progress=int(progress), slots_used=int(slot),
                           reserve=remaining.tolist(), active=False, values=[0.,0.,0.],
                           signed_claim=[0.,0.,0.], eligible=[False,False,False], selected_hold_s=0.)
                if slot != depth or progress == len(scene.targets):
                    row["reason"] = "already_complete_or_MPC_terminated"
                    rows.append(row)
                    continue
                goal = cont.mpc_command(scene, state)
                row["command"] = int(goal)
                if goal < 0 or scene.site[goal] != scene.targets[progress]:
                    row["reason"] = "no_needed_site_command_at_checkpoint"
                    rows.append(row)
                    continue
                row["active"] = True
                measures = []
                for hold in HOLD_TICKS:
                    held_reserve = remaining.copy()
                    held_reserve[-1] -= hold
                    arrival = clock + hold + int(costs[at, goal, 3])
                    witness = cont.witness_for(scene, at, 6-slot, progress, held_reserve, (goal,))
                    if arrival > 18 or witness is None:
                        measures.append(None)
                        continue
                    after = held_reserve - costs[at, goal]
                    c, w = control.claim_under_mass(b, good, like, correct, arrival, scene.to_address[goal])
                    old_witness = cont.full_witness(scene, goal, 5-slot, progress, after)
                    next_witness = cont.full_witness(scene, goal, 5-slot, progress+1, after)
                    old = control.remaining_value(potential, needs, 5-slot, goal, progress, after) if old_witness is not None else 0.
                    advanced = control.remaining_value(potential, needs, 5-slot, goal, progress+1, after) if next_witness is not None else 0.
                    value = c*advanced + max(0.,good.sum()-c-w)*old - w
                    measures.append(dict(value=float(value),signed_claim=float(c-w),correct_mass=float(c),
                                         wrong_mass=float(w),arrival_s=float(arrival*.5),witness=list(witness),
                                         unresolved_witness=None if old_witness is None else list(old_witness),
                                         advanced_witness=None if next_witness is None else list(next_witness)))
                if measures[0] is None:
                    row.update(active=False,reason="original_needed_site_command_has_no_joint_completion_witness")
                    rows.append(row)
                    continue
                # Fixed-hold policies fall back to no hold when that hold
                # removes every ideal-acceptance completion route.
                row["eligible"] = [m is not None for m in measures]
                row["values"] = [(m or measures[0])["value"] for m in measures]
                row["signed_claim"] = [(m or measures[0])["signed_claim"] for m in measures]
                chosen = int(np.argmax(row["values"]))
                row.update(selected_hold_s=float(HOLD_TICKS[chosen]*.5),measurements=measures,
                           selected_index=chosen,reason="public_nominal_value_comparison")
                rows.append(row)
    summary = {}
    for task in shared.TASKS:
        summary[task] = {}
        for depth in (1,2):
            records = [r for r in rows if r["task"]==task and r["depth"]==depth]
            values = np.array([r["values"] for r in records])
            signed = np.array([r["signed_claim"] for r in records])
            best_fixed = int(np.argmax(values.mean(axis=0)))
            contextual = float(values.max(axis=1).mean())
            summary[task][str(depth)] = dict(
                n_model_histories=len(records),active=sum(r["active"] for r in records),
                mean_nominal_values=values.mean(axis=0).tolist(),
                mean_signed_claim_mass=signed.mean(axis=0).tolist(),
                best_fixed_hold_s=HOLD_TICKS[best_fixed]*.5,
                contextual_nominal_minus_no_hold=contextual-float(values[:,0].mean()),
                contextual_nominal_minus_best_fixed=contextual-float(values[:,best_fixed].mean()),
                contextual_claim_minus_best_fixed=float(signed.max(axis=1).mean()-signed.mean(axis=0).max()),
                selected_hold_counts=[sum(r.get("selected_index",0)==i for r in records) for i in range(3)],
                mean_extra_hold_s=float(np.mean([r["selected_hold_s"] for r in records])),
                infeasible_hold_counts=[sum(r["active"] and not r["eligible"][i] for r in records) for i in range(3)],
                reasons={reason:sum(r["reason"]==reason for r in records) for reason in sorted({r["reason"] for r in records})})
    report = dict(namespace=NAMESPACE,n=args.n,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_atoms=256,bank_sha256=hashlib.sha256(args.bank.read_bytes()).hexdigest(),hold_s=[0.,.5,1.],
        clocks_s=[0.,9.],base_edge_s=1.5,legacy_grid_parity=parity,summary=summary,
        elapsed_wall_s=time.perf_counter()-began,
        scope="Prior-only public-history claim/nominal-tail surrogate, no terminal outcome or B-vs-time-aware-MPC test.",
        latency="CPU does not advance clocks; deployed timing needs explicit planning-latency treatment before validation.",
        dropout_scope="Latest additional dropout ends4.45s: depth1 can encounter it, depth2 earliest4.5s tests lag/identity.")
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"records.jsonl").write_text("".join(json.dumps(row,separators=(",",":"))+"\n" for row in rows))
    (args.out/"SUMMARY.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--repo",type=Path,required=True)
    parser.add_argument("--bank",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--n",type=int,default=128)
    run(parser.parse_args())
