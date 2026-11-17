"""Replay fixed openings with a public online repair of the shared image ledger."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from adaptive_controller import CENTER_MM, Witness, station_bin
from diagnose_b_selection import VERIFY_PURPOSE, run_parallel, save_archive
from learned_recovery import CEMMenuController
from run_adaptive_recovery import OUTCOMES, paired_statistics, real_trial, shared_prefix, tape_for, trace
from run_learned_recovery import CONTROLLERS, FAMILIES, configure, read_rows, write_json


ORIGINAL = {}
ARMS = ("mpc", "teacher", "mpc_with_update", "teacher_with_update")


class CenteredEvidenceController(CEMMenuController):
    """Allow a newly associated image to supersede an inferior retained image."""

    def observe(self, scene, state, record, alias=False):
        super().observe(scene, state, record, alias)
        station, slot = station_bin(state.s), state.slot
        if station < 0 or slot < 0 or abs(state.residual) > CENTER_MM:
            return
        current = state.ledger[station]
        if current is None or current.frame == state.frame:
            return
        last = record.packets[-1]
        _, conditional = self.world.likelihood(np.asarray(last["y"]), state.s, state.x, state.angle)
        lateral = self.world.target_grid(state.s, state.angle)[:, 0]
        associated, centered = conditional[:, slot], abs(lateral - state.x) <= CENTER_MM
        old_quality = float(state.belief @ (current.identity * current.centered))
        new_quality = float(state.belief @ (associated * centered))
        if new_quality <= old_quality + 1e-9:
            return

        ledger = list(state.ledger)
        ledger[station] = Witness(state.frame, slot, state.s, associated.copy(), centered)
        state.ledger = tuple(ledger)
        record.retractions.append({
            "frame": state.frame, "bin": station, "witness": current.frame, "reason": "higher_centered_probability",
            "old_quality": old_quality, "new_quality": new_quality,
        })
        record.certificates.append({
            "frame": state.frame, "bin": station, "slot": slot,
            "identity_ok": last["roles_truth"][slot] == 0, "center_ok": abs(last["target_truth"][0] - state.x) <= CENTER_MM,
        })


def compare_case(job):
    row, update_prefix = job
    seed, family = row["seed"], row["family"]
    old, updated = ORIGINAL[family], CONTROLLERS[family]
    scene = old.world.draw(seed)
    root, prefix = shared_prefix(old, scene)
    revised_root, revised_prefix = shared_prefix(updated, scene) if update_prefix else (root, prefix)
    if update_prefix:
        np.testing.assert_allclose(revised_prefix.motion, prefix.motion, atol=1e-12, rtol=0.)
        np.testing.assert_allclose(revised_root.belief, root.belief, atol=1e-12, rtol=0.)
    menu = updated.options_shared(revised_root, renderer="fit")
    if list(menu) != row["names"]:
        raise ValueError("shared candidate library differs")
    np.testing.assert_allclose(root.belief, row["public_root"]["belief"], atol=1e-12, rtol=0.)
    actions = {arm: row["selected"][arm.removesuffix("_with_update")] for arm in ARMS}
    trials = []
    for saved in row["verification"]:
        replica = saved["replica"]
        disturbance = tape_for(scene.latent, seed, replica, VERIFY_PURPOSE)
        outcomes = {arm: saved["outcomes"][arm] | {"quality_replacements": 0} for arm in ARMS[:2]}
        logs = {arm: saved["traces"][arm] for arm in ARMS[:2]} if replica == 0 else {}
        runs = {}
        for arm in ARMS[2:]:
            name = actions[arm]
            if name not in runs:
                metrics, record = real_trial(updated, revised_root, revised_prefix, disturbance, [menu[name]],
                                             detailed=replica == 0)
                replaced = sum(r.get("reason") == "higher_centered_probability" for r in record.retractions)
                suffix = trace(record, after_frame=root.frame) if replica == 0 else None
                runs[name] = metrics | {"quality_replacements": replaced}, suffix
            outcomes[arm], log = runs[name]
            if replica == 0:
                logs[arm] = log
        trials.append({"replica": replica, "outcomes": outcomes, "traces": logs})
    output = {key: row[key] for key in ("family", "seed", "prefix", "checkpoint", "public_root")}
    if update_prefix:
        updated_prefix = trace(revised_prefix)
        output["prefix_by_arm"] = {arm: updated_prefix for arm in ARMS[2:]}
    return output | {"selected": actions, "verification": trials}


def summarize(rows):
    results = {}
    extract = lambda row: [t["outcomes"] for t in row["verification"]]
    for family in (*FAMILIES, "pooled"):
        cohort = [r for r in rows if r["family"] == family or family == "pooled"]
        data = paired_statistics(cohort, ARMS, (*OUTCOMES, "quality_replacements"), extract)
        data["replays"] = sum(len(r["verification"]) for r in cohort)
        data["counts"] = {arm: {f: int(sum(t[arm][f] for r in cohort for t in extract(r)))
                                for f in ("corrected_complete", "final_wrong", "new_centered_mistake")}
                          for arm in ARMS}
        direct = {}
        for pair in (("mpc", "mpc_with_update"), ("teacher", "teacher_with_update"),
                     ("mpc_with_update", "teacher_with_update")):
            direct[pair[-1] + "_vs_" + pair[0]] = paired_statistics(
                cohort, pair, OUTCOMES, extract)["vs_reference"][pair[-1]]
        data["direct_contrasts"] = direct
        results[family] = data
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=24)
    parser.add_argument("--update-prefix", action="store_true", help="enable replacement from the first scan frame")
    args = parser.parse_args()
    if args.jobs <= 0:
        parser.error("positive worker count required")
    configure(args.source)
    ORIGINAL.update(CONTROLLERS)
    configure(args.source, controller_type=CenteredEvidenceController)
    started = perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = run_parallel(compare_case, [(row, args.update_prefix) for row in read_rows(args.cases)], args.jobs,
                        "centered evidence")
    name = "full_online_update" if args.update_prefix else "online_update"
    save_archive(args.out / f"{name}_cases.jsonl.gz", rows)
    write_json(args.out / f"{name}_summary.json", summarize(rows) | {
        "updated_during_prefix": args.update_prefix, "elapsed_s": perf_counter() - started,
    })


if __name__ == "__main__":
    main()
