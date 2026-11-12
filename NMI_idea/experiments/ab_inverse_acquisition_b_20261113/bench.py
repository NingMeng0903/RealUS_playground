"""Paired opportunity census on task-preserving model-generated views.

The opportunity selector is evaluator-only. The deployed comparator always
updates and replans through the pinned complete MPC implementation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import sys
import time

import numpy as np

import world as extension

DATA, CONFIG, BANK, MENU_CACHE = {}, None, None, {}
CAPTURES = 6
NAMESPACES = {"smoke": 719900, "screen": 719901, "train": 719902,
              "development": 719903, "confirmation": 719904, "sensitivity": 719905}
FIELDS = ("correct_completion", "any_wrong", "tcp_mm", "turn_deg", "motion_s", "rail_mm", "captures", "combined_s")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize(args):
    global acoustics, control, shared, cont, DATA, CONFIG, BANK, MENU_CACHE
    CONFIG = args
    DATA, MENU_CACHE = {}, {}
    acoustics, control, shared = extension.load(args.repo)
    sys.path.insert(0, str(Path(args.repo) / "NMI_idea/experiments/ab_continuation_b_20261113"))
    import continuations as cont
    cont.load_shared(args.repo)
    BANK = json.loads(args.motion.read_text())
    nodes = sorted(BANK["nodes"], key=lambda x: x["id"])
    assert [n["id"] for n in nodes] == list(range(len(nodes)))
    sites = np.array([n["site"] for n in nodes], dtype=np.int64)
    views = np.array([n["view"] for n in nodes], dtype=np.int64)
    angles = np.array([n.get("tilt_deg", extension.ANGLES.get(n["view"], np.nan)) for n in nodes])
    assert np.isfinite(angles).all()
    rho = np.array([int(n["realization"] != "dird_top") for n in nodes], dtype=np.int64)
    base_bank = json.loads(args.base.read_text())
    old_pairs = {(e["start"], e["goal"]) for e in base_bank["edges"]}
    expanded_edges = {(e["start"], e["goal"]): e for e in BANK["edges"]}
    for edge in base_bank["edges"]:
        assert expanded_edges.get((edge["start"], edge["goal"])) == edge, "an archived edge/trace changed"
    old_nodes = {n["id"] for n in base_bank["nodes"]}
    assert nodes[:len(old_nodes)] == [dict(n, tilt_deg=float(extension.ANGLES[n["view"]]))
                                    for n in base_bank["nodes"]] or all(
        all(nodes[n["id"]].get(k) == v for k, v in n.items()) for n in base_bank["nodes"])
    world = acoustics.setup_world(args.repo)
    clocks = np.arange(CAPTURES + 1) * 1.5
    reports = {}
    for task, (targets, label) in shared.TASKS.items():
        geometry = extension.Geometry(acoustics, world, nodes, label)
        reports[task] = extension.parity_report(acoustics, geometry)
        if reports[task]["legacy_kernel_max_abs"] > 1e-12 or reports[task]["view_id_invariance_max_abs"] > 1e-12:
            raise AssertionError("legacy or view-identifier acoustic parity failed")
        like, correct, detect, cdf = acoustics.model_tables(geometry, clocks, args.model_size, "joint")
        costs = np.full((len(nodes), len(nodes), 4), np.inf)
        original = costs.copy()
        for edge in BANK["edges"]:
            a, b = edge["start"], edge["goal"]
            if sites[a] not in (5, *targets) or sites[b] not in (5, *targets):
                continue
            if abs(edge["duration_s"] - 1.5) > 1e-9:
                raise ValueError("the opportunity protocol freezes 1.5 s motion slots")
            costs[a, b] = (edge["travel_mm"], edge["turn_deg"], edge["rail_mm"], 1.)
            if (a, b) in old_pairs:
                original[a, b] = costs[a, b]
        reserve = np.array([65., 90., 250., CAPTURES])
        targets = np.array(targets, dtype=np.int64)
        potential, needs = control.nominal_potential(sites, costs, targets, CAPTURES, .45 if task == "daughter" else .68)
        material = np.array([acoustics.make_scenario(world, acoustics.seed_for(acoustics.PRIOR_SEED, j)).body.w_subject_mm[5]
                             for j in range(args.model_size)])
        scene = control.Scene(like, correct, detect, cdf, geometry.node_to_address, sites,
            np.sign(angles).astype(np.int64), rho, costs, targets, BANK["task_starts"][task],
            reserve, potential, needs, material, 5, 1.5)
        coarse_costs = costs.copy()
        unavailable = np.array([n not in old_nodes for n in range(len(nodes))])
        coarse_costs[unavailable] = np.inf
        coarse_costs[:, unavailable] = np.inf
        DATA[task] = dict(scene=scene, geometry=geometry, clocks=clocks, nodes=nodes, angles=angles,
                          old=old_nodes, original=with_costs(scene, original, task),
                          coarse=with_costs(scene, coarse_costs, task))
    return reports


def with_costs(scene, costs, task):
    potential, needs = control.nominal_potential(scene.site, costs, scene.targets, CAPTURES,
                                                .45 if task == "daughter" else .68)
    return replace(scene, edge_costs=costs, potential=potential, needs=needs)


def identity_utility(scene, prior, address, clock):
    like, correct = scene.likelihood[clock, address], scene.correct[clock, address]
    mass, true_mass = like @ prior, correct @ prior
    accept = (mass > 1e-260) & (true_mass >= control.CONFIDENCE * mass)
    accept[0] = False
    # The rule is fixed using the public belief, not a clairvoyant rule per h.
    particle_values = (2. * correct[accept] - like[accept]).sum(axis=0)
    return float(particle_values @ prior), particle_values


def route_through(scene, goal):
    """A sufficient full joint-cap witness that claims at this target pose.

    A missing witness from this bounded search is not an infeasibility proof.
    """
    if scene.site[goal] == scene.targets[0]:
        return cont.witness_for(scene, scene.start, CAPTURES, 0, scene.reserve, (goal,))
    if scene.site[goal] != scene.targets[1]:
        return None
    for first in np.flatnonzero(scene.site == scene.targets[0]):
        costs = scene.edge_costs[scene.start, first] + scene.edge_costs[first, goal]
        if np.all(costs <= scene.reserve + 1e-8):
            return (int(first), int(goal))
    return None


def inverse_menu(task, prior, anchor):
    key = task, int(anchor)
    if key in MENU_CACHE:
        return MENU_CACHE[key]
    started = time.process_time()
    data = DATA[task]
    scene, nodes = data["scene"], data["nodes"]
    selections, values = {}, {}
    for position, site in enumerate(scene.targets):
        reference_site = int(scene.targets[min(position + 1, len(scene.targets) - 1)])
        rows = []
        # A view is eligible only if at least one of its checked realizations
        # lies on a full-task route. This is geometry-only ideal acceptance.
        for point, address in data["geometry"].index.items():
            s, view = point
            if s != site or view in extension.ANGLES:
                continue
            targets_at_view = [n["id"] for n in nodes if n["site"] == site and n["view"] == view]
            # Certify using only the shared coarse graph plus this view, so
            # the witness cannot depend on another angle dropped by the menu.
            included = np.array([n["id"] in data["old"] or (n["site"] == site and n["view"] == view)
                                 for n in nodes])
            checked_costs = scene.edge_costs.copy()
            checked_costs[~included], checked_costs[:, ~included] = np.inf, np.inf
            checked_scene = replace(scene, edge_costs=checked_costs)
            witnesses = [route_through(checked_scene, n) for n in targets_at_view]
            if not any(witness is not None for witness in witnesses):
                continue
            continuation, references = [], []
            for refview in (3, 7, 8):
                legal = False
                for goal, witness in zip(targets_at_view, witnesses):
                    if witness is None:
                        continue
                    reached = witness[:witness.index(goal)+1]
                    used = np.zeros(4)
                    at = scene.start
                    for next_node in reached:
                        used += checked_costs[at, next_node]
                        at = next_node
                    for ref in nodes:
                        if ref["site"] != reference_site or ref["view"] != refview:
                            continue
                        if len(reached) < CAPTURES and np.all(used + checked_costs[goal, ref["id"]] <= scene.reserve + 1e-8):
                            legal = True
                            break
                    if legal:
                        break
                if legal:
                    refaddress = data["geometry"].index[reference_site, refview]
                    _, value = identity_utility(scene, prior, refaddress, min(position + 2, CAPTURES))
                    continuation.append(value)
                    references.append(refview)
            utilities = np.array(continuation) if continuation else np.zeros((1, len(prior)))
            incumbent = float(np.max(utilities @ prior))
            utility, _ = identity_utility(scene, prior, address, position + 1)
            weighted = scene.likelihood[position + 1, address] * prior[None, :]
            information = float(np.max(weighted @ utilities.T, axis=1).sum() - incumbent)
            rows.append(dict(view=int(view), angle=float(data["geometry"].angles[point]),
                             immediate_claim_utility=utility, decision_information=information,
                             reference_views=references,
                             perfect_information_claim_gap=float(prior @ np.max(utilities, axis=0)-incumbent),
                             ideal_completion_witnesses=witnesses))
        selected = []
        if rows:
            for objective in ("immediate_claim_utility", "decision_information"):
                row = max(rows, key=lambda r: (r[objective], -abs(r["angle"]), -r["angle"]))
                if row["view"] not in selected:
                    selected.append(row["view"])
        selections[str(int(site))] = selected
        values[str(int(site))] = rows
    available = np.array([n["id"] in data["old"] or n["view"] in selections.get(str(n["site"]), [])
                          for n in nodes])
    costs = scene.edge_costs.copy()
    costs[~available], costs[:, ~available] = np.inf, np.inf
    augmented = with_costs(scene, costs, task)
    elapsed = time.process_time() - started
    result = augmented, dict(selected=selections, scores=values, construction_cpu_s=elapsed)
    MENU_CACHE[key] = result
    return result


def menu_control(task, proposals, kind):
    data = DATA[task]
    selections = {}
    for site, rows in proposals["scores"].items():
        count = len(proposals["selected"][site])
        if kind == "claim_only":
            ranked = sorted(rows, key=lambda r: (-r["immediate_claim_utility"], abs(r["angle"]), r["angle"]))
        else:
            # Fixed preference is selected before outcome access. It matches
            # cardinality and falls back by angular distance if +/-6 is absent.
            ranked = sorted(rows, key=lambda r: (min(abs(r["angle"]+6.), abs(r["angle"]-6.)), r["angle"]))
        selections[site] = [row["view"] for row in ranked[:count]]
    available = np.array([n["id"] in data["old"] or n["view"] in selections.get(str(n["site"]), [])
                          for n in data["nodes"]])
    costs = data["scene"].edge_costs.copy()
    costs[~available], costs[:, ~available] = np.inf, np.inf
    return with_costs(data["scene"], costs, task), selections


def initial(task, seed):
    data = DATA[task]
    scene = data["scene"]
    scenario = acoustics.make_scenario(data["geometry"].world, seed)
    _, _, detect, cdf = acoustics.probabilities(data["geometry"], scenario, data["clocks"], "joint")
    detect, cdf = np.ascontiguousarray(detect[:, :, None]), np.ascontiguousarray(cdf[:, :, None])
    tape = acoustics.tape(seed, len(data["clocks"]))
    obs, _ = control.capture_draw(detect, cdf, 0, scene.to_address[scene.start], 0, tape)
    uniform = np.full(len(scene.material_coordinate), 1. / len(scene.material_coordinate))
    prior, _ = control.posterior(uniform, scene.likelihood, scene.correct, 0, scene.to_address[scene.start], obs)
    state = prior, prior.copy(), scene.reserve.copy(), scene.start, 0, 0, 0, False
    return state, int(obs), tape, detect, cdf


def finish(scene, state, detect, cdf, tape, prefix):
    began = time.process_time()
    final, trace = cont.execute(scene.public(), detect, cdf, tape, state,
                                np.asarray(prefix, dtype=np.int64), CAPTURES)
    cpu = time.process_time() - began
    result = cont.metrics(scene, final, cpu)
    return result, trace.tolist(), cont.state_hash(final)


def trial(index):
    seed = acoustics.seed_for(NAMESPACES[CONFIG.split], index)
    output = []
    for task, data in DATA.items():
        state, anchor, prefix_tape, detect, cdf = initial(task, seed)
        scene, proposals = inverse_menu(task, state[0], anchor)
        claim_scene, claim_selection = menu_control(task, proposals, "claim_only")
        fixed_scene, fixed_selection = menu_control(task, proposals, "fixed_angles")
        extra_scenes = {"claim_only": claim_scene, "fixed_angles": fixed_scene}
        baseline, library = cont.candidate_library(scene, state, 0, "single")
        # Task-preserving acquisition: first command must reach the next
        # required site. Direct MPC remains available regardless.
        library = {name: value for name, value in library.items()
                   if name == "MPC" or scene.site[value[0][0]] == scene.targets[0]}
        fixed = -1
        for node in data["nodes"]:
            if node["site"] == scene.targets[0] and node["view"] == 3 and node["realization"] == "dird_top":
                if cont.witness_for(scene, state[3], CAPTURES, 0, state[2], (node["id"],)) is not None:
                    fixed = node["id"]
                break
        paths = {"MPC": [], "coarse": [], "original": [], "fixed": [], "claim_only": [], "fixed_angles": []}
        measurements = {name: [] for name in paths}
        for name in library:
            paths.setdefault(name, [])
            measurements.setdefault(name, [])
        seeds, hashes = [], []
        for repeat in range(CONFIG.repeats):
            suffix_seed = acoustics.seed_for(seed, 930000 + repeat)
            tape = acoustics.tape(suffix_seed, len(data["clocks"]))
            tape[0] = prefix_tape[0]
            seeds.append(suffix_seed)
            hashes.append(hashlib.sha256(tape.tobytes()).hexdigest())
            for arm in measurements:
                used_scene = data[arm] if arm in ("coarse", "original") else extra_scenes.get(arm, scene)
                prefix = (() if fixed < 0 else (fixed,)) if arm == "fixed" else library.get(arm, ((), None))[0]
                result, trace, final_hash = finish(used_scene, state, detect, cdf, tape, prefix)
                # Charge the deterministic proposal computation to augmented
                # MPC and each intervention, even if this worker caches it.
                if arm not in ("coarse", "original"):
                    result[-1] += proposals["construction_cpu_s"]
                measurements[arm].append(result)
                paths[arm].append(dict(trace=trace, final_public_state_sha256=final_hash))
        record = dict(index=index, case_seed=seed, task=task, anchor=anchor,
                      state_sha256=cont.state_hash(state), proposals=proposals,
                      menu_controls=dict(claim_only=claim_selection, fixed_angles=fixed_selection),
                      mpc_command=int(baseline), fixed_command=int(fixed),
                      candidates={name: dict(prefix=list(value[0]), witness=value[1]) for name,value in library.items()},
                      repeat_seeds=seeds, tape_sha256=hashes, measurements=measurements, paths=paths)
        output.append(record)
    return output


def summarize(out):
    from paired_metrics import summarize as paired
    rows = [json.loads(line) for line in (out / "cases.jsonl").read_text().splitlines() if line]
    report = {}
    for task in ("daughter", "trunk"):
        cells = [r for r in rows if r["task"] == task]
        if not cells:
            continue
        repeats = len(cells[0]["repeat_seeds"])
        half = repeats // 2
        if half == 0 or half * 2 != repeats:
            raise ValueError("split-repeat audit needs an even number of at least two repeats")
        selected, base, fixed, coarse, original, raw, selection, claim_only, fixed_angles = [], [], [], [], [], [], [], [], []
        for row in cells:
            values = {k: np.array(v) for k,v in row["measurements"].items()}
            incumbent = values["MPC"]
            chosen = incumbent.copy()
            unconstrained = incumbent.copy()
            halves = []
            for rank, test in ((np.arange(half), np.arange(half, repeats)),
                               (np.arange(half, repeats), np.arange(half))):
                best, best_name, unc, unc_name = 0., "MPC", 0., "MPC"
                for name in row["candidates"]:
                    delta = (values[name][rank] - incumbent[rank]).mean(axis=0)
                    if delta[0] > unc + 1e-12:
                        unc, unc_name = delta[0], name
                    if np.all(delta[2:5] <= np.array([2., 2., .1]) + 1e-12) and delta[0] > best + 1e-12:
                        best, best_name = delta[0], name
                chosen[test], unconstrained[test] = values[best_name][test], values[unc_name][test]
                halves.append(dict(choice=best_name, unconstrained=unc_name,
                                   rank_gain=float(best), test_gain=float((values[best_name][test,0]-incumbent[test,0]).mean())))
            selected.append(chosen)
            raw.append(unconstrained)
            base.append(incumbent)
            fixed.append(values["fixed"])
            coarse.append(values["coarse"])
            original.append(values["original"])
            claim_only.append(values["claim_only"])
            fixed_angles.append(values["fixed_angles"])
            selection.append(dict(index=row["index"], halves=halves))
        comparisons = {"split_cost_screened_vs_augmented": paired(selected, base),
                       "split_hard_caps_vs_augmented": paired(raw, base),
                       "split_cost_screened_vs_fixed": paired(selected, fixed),
                       "augmented_vs_coarse": paired(base, coarse),
                       "coarse_vs_original": paired(coarse, original),
                       "augmented_vs_claim_only": paired(base, claim_only),
                       "augmented_vs_fixed_angles": paired(base, fixed_angles),
                       "fixed_vs_augmented": paired(fixed, base)}
        gate = comparisons["split_cost_screened_vs_augmented"]
        pass_screen = (gate["delta"]["correct_completion"] >= .03 and
                       gate["delta_ci95_case_clustered"]["correct_completion"][0] > 0. and
                       gate["delta"]["any_wrong"] <= 0. and
                       gate["all_costs_noninferior_with_margin"] and
                       comparisons["split_cost_screened_vs_fixed"]["delta"]["correct_completion"] > 0.)
        report[task] = dict(comparisons=comparisons, selections=selection,
            screen_pass=bool(pass_screen), independent_cases=len(cells), tapes_per_case=repeats,
            opportunity_interpretation="Attained label-assisted split-repeat selection; neither a public policy nor a population upper bound.")
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--motion", type=Path)
    p.add_argument("--base", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", choices=NAMESPACES, default="smoke")
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--model-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repo / "NMI_idea/experiments/ab_continuation_b_20261113"))
    if args.report_only:
        summarize(args.out)
        return
    if (args.out / "cases.jsonl").exists():
        raise FileExistsError("do not overwrite an existing cohort")
    began = time.time()
    parity = initialize(args)
    (args.out / "acoustic_parity.json").write_text(json.dumps(parity, indent=2) + "\n")
    frozen = {name: sha(Path(__file__).parent / name) for name in ("bench.py", "world.py", "motion.py")}
    legacy = args.repo / "NMI_idea/experiments/ab_full_b_rollout_20261113"
    frozen.update({"legacy/"+name: sha(legacy/name) for name in ("control.py", "acoustics.py", "run.py")})
    continuation_dir = args.repo / "NMI_idea/experiments/ab_continuation_b_20261113"
    frozen.update({"continuation/"+name:sha(continuation_dir/name) for name in ("continuations.py", "paired_metrics.py")})
    (args.out / "settings.json").write_text(json.dumps(dict(
        split=args.split, namespace=NAMESPACES[args.split], n=args.n, repeats=args.repeats,
        model_size=args.model_size, fields=FIELDS, bank_sha256=sha(args.motion), base_sha256=sha(args.base),
        source_sha256=frozen, checkpoint="one common anchor; zero MPC commands", compute_advances_acoustic_clock=False,
        claim="Fresh correct identities at both ordered target sites and sampled continuation, any wrong claim fails.",
        proposal="at most two offgrid angles per required site; menu frozen from public anchor and shared with MPC",
        ghost_extension="endpoint-preserving smooth interpolation; no new view-index random regimes"), indent=2)+"\n")
    # Compile the exact shared executor before forking. Smoke is not evidence.
    state, anchor, tape, detect, cdf = initial("trunk", acoustics.seed_for(NAMESPACES["smoke"], 9000))
    finish(DATA["trunk"]["original"], state, detect, cdf, tape, ())
    print(json.dumps({"initialized_s":time.time()-began,"nodes":len(BANK["nodes"]),"edges":len(BANK["edges"])}),flush=True)
    with (args.out / "cases.jsonl").open("w") as stream:
        if args.workers == 1:
            iterator = map(trial, range(args.n))
            pool = None
        else:
            pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("fork"))
            iterator = pool.map(trial, range(args.n))
        try:
            for index, rows in enumerate(iterator):
                for row in rows:
                    stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                stream.flush()
                print(json.dumps({"cases_done":index+1,"elapsed_s":time.time()-began}),flush=True)
        finally:
            if pool is not None:
                pool.shutdown()
    report = summarize(args.out)
    print(json.dumps({"screen_pass":{task: cell["screen_pass"] for task,cell in report.items()}}),flush=True)


if __name__ == "__main__":
    main()
