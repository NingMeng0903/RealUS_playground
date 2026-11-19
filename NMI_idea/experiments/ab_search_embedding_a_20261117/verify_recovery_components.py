"""Paired tests of evidence, candidate discovery, rendering and action selection."""

import argparse
from collections import Counter
from copy import copy
from io import BytesIO
import json
from pathlib import Path
from shutil import copyfile
from time import process_time

import joblib
import numpy as np

from adaptive_controller import AdaptiveController, BINS, END, IDENTITY_ACCEPT, station_bin
from adaptive_teacher import cem_discover, evaluate, fixed_worlds, posterior_worlds
from audit_prodmp_weights import legacy_reference
from diagnose_b_selection import run_parallel, save_archive
from learned_recovery import (OBJECTIVE, VISIBLE_TARGETS, choose_from_deltas, features_for_menu, outcome_vector,
                              predict_deltas, resource_limits)
from run_adaptive_recovery import MARGINS, paired_statistics, real_trial, shared_prefix, tape_for
from run_learned_recovery import CONTROLLERS, FAMILIES, configure, fingerprint, read_rows, write_json


TARGETS = VISIBLE_TARGETS
FIELDS = (*TARGETS, "frames", "ever_wrong", "feedback_revisions", "centered_complete")
CAPS = resource_limits(TARGETS)
CAPS_WITHOUT_RELATIVE_MOTION = {key: value for key, value in CAPS.items() if key not in ("tcp_mm", "turn_degs")}
NS = {"discovery": 24_210_000, "development": 25_210_000, "holdout": 26_210_000, "render": 27_210_000,
      "library_test": 28_210_000}
FIXED = ("advance", "wait", "short", "side_left", "side_right", "revisit")
SHADOW_SCALE = 2.
EVIDENCE_TAPE = 1562
SELECTION_TAPE = 4167
B_MODELS = None
STOCK_CONTROLLERS = {}


def bands(rows, names, fields=FIELDS):
    def measures(r):
        return r["verification"]

    output = {}
    for family in ("pooled", *FAMILIES):
        subset = [r for r in rows if family in ("pooled", r["family"])]
        if not subset:
            continue
        stats = paired_statistics(subset, names, fields, measures)
        counts = ("corrected_complete", "final_wrong", "new_identity_mistake", "ever_wrong")
        stats["counts"] = {arm: {key: int(sum(t[arm].get(key, 0) for r in subset for t in measures(r)))
                                 for key in counts if key in fields} for arm in names}
        stats["contrast_checks"] = {}
        for arm, effects in stats["vs_reference"].items():
            stats["contrast_checks"][arm] = {
                "completion": effects["corrected_complete"]["delta"] >= .05
                and effects["corrected_complete"]["ci95"][0] > 0.,
                "mistakes_within_margin": all(effects[k]["ci95"][1] <= .02
                                               for k in ("final_wrong", "new_identity_mistake") if k in effects),
                "execution_within_upper_margins": all(effects[k]["ci95"][1] <= v
                                                       for k, v in MARGINS.items() if k in effects),
                "execution_equivalent": all(effects[k]["ci95"][0] >= -v and effects[k]["ci95"][1] <= v
                                            for k, v in MARGINS.items() if k in effects),
            }
        output[family] = stats
    return output


def case_start(family, seed):
    controller = CONTROLLERS[family]
    scene = controller.world.draw(seed)
    root, prefix = shared_prefix(controller, scene)
    return controller, scene, root, prefix


def complete_paths(controller, scene, root, prefix, seed, actions, replicas, purpose):
    trials = []
    for replica in range(replicas):
        future = tape_for(scene.latent, seed, replica, purpose)
        outcomes = {}
        completed = {}
        for label, action in actions.items():
            # Compare the same motion once when two policies make an identical choice.
            key = id(action)
            if key not in completed:
                run, record = real_trial(controller, root, prefix, future, [action])
                if run["max_handoff_error"] >= 1e-7:
                    raise ValueError("motion handoff was discontinuous")
                completed[key] = {k: run[k] for k in FIELDS} | {
                    "fresh_target_in_first_segment": run["first_block_true_image"],
                    "suffix_frames": run["frames"] - len(prefix.packets),
                }
            outcomes[label] = completed[key]
        trials.append(outcomes)
    return trials


def rollout_scores(controller, root, menu, scenarios):
    return np.stack([np.mean([outcome_vector(out, TARGETS) for out in
                              evaluate(controller, root, act, scenarios, open_blocks=1)[0]], axis=0)
                     for act in menu.values()])


def pick(menu, incumbent, estimates):
    actions = list(menu)
    selected, delta, eligible = choose_from_deltas(actions, incumbent, estimates - estimates[actions.index(incumbent)],
                                                  targets=TARGETS)
    return selected, {"delta": delta, "eligible": eligible}


def replay_evidence(world, record, *, retain_candidates):
    belief = world.prior.copy()
    images = [[] for _ in range(BINS)]
    selected = [None] * BINS
    error_start = [None] * BINS
    delays = []
    confirmations = 0
    times_ms = []
    ever_wrong = False
    for p in record.packets:
        start = process_time()
        y = np.asarray(p["y"], dtype=float).reshape(-1, 2)
        mass, responsibilities = world.likelihood(y, p["s"], p["x"], p["q"])
        belief *= mass
        belief /= belief.sum()
        probabilities = belief @ responsibilities
        j = station_bin(p["s"])
        if j >= 0:
            for slot, at_capture in enumerate(probabilities):
                if retain_candidates or at_capture >= IDENTITY_ACCEPT:
                    images[j].append((p["frame"], slot, responsibilities[:, slot]))
        replacement = []
        for candidates in images:
            scores = np.array([conditional @ belief for _, _, conditional in candidates])
            best = scores.max(initial=0.)
            index = np.flatnonzero(scores >= max(IDENTITY_ACCEPT, best - 1e-9))
            replacement.append(None if not len(index) else candidates[index[0]][:2])
        times_ms.append(1000 * (process_time() - start))

        for j, match in enumerate(replacement):
            if match == selected[j]:
                continue
            if match is not None:
                confirmations += 1
                wrong = record.packets[match[0]]["roles_truth"][match[1]] != 0
                if wrong:
                    ever_wrong = True
                    if error_start[j] is None:
                        error_start[j] = p["frame"]
                elif error_start[j] is not None:
                    delays.append((p["frame"] - error_start[j]) * .1)
                    error_start[j] = None
        selected = replacement
    correct = [record.packets[f]["roles_truth"][slot] == 0 for f, slot in (x for x in selected if x is not None)]
    return {"corrected_complete": len(correct) == BINS and all(correct) and record.packets[-1]["s"] >= END - .28,
            "final_wrong": not all(correct), "ever_wrong": ever_wrong, "verified_intervals": sum(correct),
            "confirmations": confirmations, "ms_per_frame": times_ms, "correction_delay_s": delays}


def evidence_case(row):
    family, seed = row["family"], row["seed"]
    controller, scene, root, prefix = case_start(family, seed)
    action = controller.options_shared(root)[row["selected"]["mpc"]]
    outputs = []
    for prior in row["verification"]:
        future = tape_for(scene.latent, seed, prior["replica"], EVIDENCE_TAPE)
        metrics, record = real_trial(controller, root, prefix, future, [action])
        assert metrics["corrected_complete"] == prior["outcomes"]["mpc"]["corrected_complete"]
        results = {name: replay_evidence(controller.world, record, retain_candidates=retain)
                   for name, retain in (("confirm_at_capture", False), ("retain_candidates", True))}
        assert metrics["corrected_complete"] == results["retain_candidates"]["corrected_complete"]
        assert metrics["final_wrong"] == results["retain_candidates"]["final_wrong"]
        outputs.append(results)
    return {"family": family, "seed": seed, "verification": outputs}


def selection_case(job):
    family, seed, samples, replicas, fixed_action = job
    c, scene, root, prefix = case_start(family, seed)
    start = process_time()
    menu = c.options_shared(root)
    mpc = c.mpc(root, menu, False, active=True, informative=True).name
    mpc_time = process_time() - start
    start = process_time()
    values = rollout_scores(c, root, menu, posterior_worlds(root, samples, 750_423))
    teacher, teacher_log = pick(menu, mpc, values)
    teacher_time = mpc_time + process_time() - start
    if fixed_action is None:
        oracle, oracle_log = pick(menu, mpc, rollout_scores(c, root, menu, fixed_worlds(scene.latent, 6, seed, 4133)))
        selected = {"mpc": mpc, "teacher": teacher, "oracle": oracle}
        selected |= {name: name if name in menu else mpc for name in FIXED}
        times = {"mpc": mpc_time, "teacher": teacher_time}
    else:
        start = process_time()
        features = features_for_menu(c, root, menu, mpc, prefix.packets[-1]["association"])
        learned, learned_log = pick(menu, mpc, predict_deltas(B_MODELS, features))
        selected = {"mpc": mpc, "teacher": teacher, "fixed": fixed_action if fixed_action in menu else mpc,
                    "learned_b_reference_fit": learned}
        times = {"mpc": mpc_time, "teacher": teacher_time, "learned_b_reference_fit": mpc_time + process_time() - start}
        oracle_log = learned_log
    measured = complete_paths(c, scene, root, prefix, seed, {k: menu[v] for k, v in selected.items()}, replicas,
                              SELECTION_TAPE)
    return {"family": family, "seed": seed, "names": list(menu), "selected": selected, "verification": measured,
            "decision_cpu_s": times, "teacher_prediction": teacher_log, "extra_prediction": oracle_log}


def phase_jobs(phase, cases, samples, repeats, extra=None):
    return [(family, NS[phase] + 100_000 * i + j, samples, repeats, extra)
            for i, family in enumerate(FAMILIES) for j in range(cases)]


def run_selection(args, out):
    global B_MODELS
    B_MODELS = joblib.load(args.source / "models.joblib")["learned_b"]
    dev = run_parallel(selection_case, phase_jobs("development", 24, args.samples, 4), args.jobs, "selection dev")
    save_archive(out / "selection_dev.jsonl.gz", dev)
    dev_bands = bands(dev, ("mpc", "teacher", "oracle", *FIXED))
    deltas = dev_bands["pooled"]["vs_reference"]

    def fixed_rank(name):
        row = deltas[name]
        excess = sum(max(0., row[key]["delta"] - bound) / bound for key, bound in CAPS.items())
        return excess, -row["corrected_complete"]["delta"], row["tcp_mm"]["delta"], name

    fixed = min(FIXED, key=fixed_rank)
    test = run_parallel(selection_case, phase_jobs("holdout", args.cases, args.samples, 4, fixed),
                        args.jobs, "selection holdout")
    save_archive(out / "selection_holdout.jsonl.gz", test)
    names = ("mpc", "fixed", "teacher", "learned_b_reference_fit")
    stats = bands(test, names)
    for family, values in stats.items():
        cohort = [r for r in test if family in ("pooled", r["family"])]
        values["choice_cpu_mean_ms"] = {arm: 1000. * float(np.mean([r["decision_cpu_s"][arm] for r in cohort]))
                                       for arm in ("mpc", "teacher", "learned_b_reference_fit")}
        values["differs_from_mpc"] = {arm: sum(r["selected"][arm] != r["selected"]["mpc"] for r in cohort)
                                     for arm in names[1:]}
    output = {"fixed_choice_from_dev": fixed, "shadow_scale": SHADOW_SCALE, "posterior_draws_per_action": args.samples,
              "dev": dev_bands, "holdout": stats, "reference_fit_sha256": fingerprint(args.source / "models.joblib")}
    write_json(out / "selection_summary.json", output)
    mean = stats["pooled"]
    print("selection heldout counts", mean["counts"], "teacher checks", mean["contrast_checks"]["teacher"], flush=True)


def run_evidence(args, out):
    cases = read_rows(args.source / "test.jsonl.gz")
    rows = run_parallel(evidence_case, cases, args.jobs, "causal history replay")
    arms = ("confirm_at_capture", "retain_candidates")
    result = bands(rows, arms, ("corrected_complete", "final_wrong", "ever_wrong", "verified_intervals"))
    for family, report in result.items():
        trials = [t for row in rows if family in ("pooled", row["family"]) for t in row["verification"]]
        report["timing"] = {}
        for arm in arms:
            ms = [x for t in trials for x in t[arm]["ms_per_frame"]]
            timing = [x for t in trials for x in t[arm]["correction_delay_s"]]
            report["timing"][arm] = {"frame_cpu_ms_p50_p95": np.quantile(ms, [.5, .95]),
                                     "measured_corrections": len(timing),
                                     "correction_simulated_s_median": float(np.median(timing)) if timing else None}
    save_archive(out / "evidence_cases.jsonl.gz", rows)
    write_json(out / "evidence_summary.json", result | {"motion_source_sha256": fingerprint(args.source / "test.jsonl.gz")})


def stock_controller(world, sections, coefficients, _cem_fit, _requests, *, evidence_mode, material_surface):
    return AdaptiveController(world, sections, coefficients, evidence_mode=evidence_mode)


def evidence_without_cem_case(job):
    row, tape_namespace = job
    family, seed = row["family"], row["seed"]
    augmented, scene, root, prefix = case_start(family, seed)
    stock = STOCK_CONTROLLERS[family]
    stock_root, stock_prefix = shared_prefix(stock, scene)
    if stock_prefix.packets != prefix.packets or stock_prefix.motion != prefix.motion:
        raise ValueError("paired motion arms have different initial scans")
    stock_action = stock.choose_mpc(stock_root, expanded=True, open_decision=True)
    archived_action = augmented.options_shared(root)[row["selected"]["mpc"]]
    outputs = []
    for index, saved in enumerate(row["verification"]):
        future = tape_for(scene.latent, seed, index, tape_namespace)
        trials = {}
        for label, controller, state, history, action in (
            ("stock", stock, stock_root, stock_prefix, stock_action),
            ("cem", augmented, root, prefix, archived_action),
        ):
            run, record = real_trial(controller, state, history, future, [action])
            if label == "cem":
                expected = saved["outcomes"]["mpc"] if "outcomes" in saved else saved["mpc"]
                if any(run[k] != expected[k] for k in TARGETS):
                    raise ValueError("augmented MPC replay differs from its archived outcomes")
            elif any(step["action"].startswith("cem_") for step in record.decisions):
                raise ValueError("stock controller used a CEM request")
            for rule, retain in (("capture", False), ("retain", True)):
                result = replay_evidence(controller.world, record, retain_candidates=retain)
                if retain and any(run[k] != result[k] for k in ("corrected_complete", "final_wrong")):
                    raise ValueError("retention replay differs from the closed-loop ledger")
                metrics = {k: run[k] for k in ("tcp_mm", "turn_degs", "seconds", "frames")}
                metrics.update({k: result[k] for k in ("corrected_complete", "final_wrong",
                                                        "ever_wrong", "verified_intervals")})
                trials[f"{label}_{rule}"] = metrics
        outputs.append(trials)
    return {"family": family, "seed": seed, "verification": outputs}


def run_evidence_without_cem(args, out):
    settings = (
        ("reference", args.source, args.source / "test.jsonl.gz", EVIDENCE_TAPE),
        ("wider_shadows", out / "shadow_library", out / "selection_holdout.jsonl.gz", SELECTION_TAPE),
    )
    reports = {}
    fields = ("corrected_complete", "final_wrong", "ever_wrong", "verified_intervals",
              "tcp_mm", "turn_degs", "seconds", "frames")
    for name, library, archive, tape in settings:
        configure(library, controller_type=stock_controller)
        STOCK_CONTROLLERS.update(CONTROLLERS)
        configure(library)
        jobs = [(row, tape) for row in read_rows(archive)]
        rows = run_parallel(evidence_without_cem_case, jobs, args.jobs, f"{name} retention without CEM")
        save_archive(out / f"evidence_without_cem_{name}.jsonl.gz", rows)
        reports[name] = {
            "archive_sha256": fingerprint(archive),
            "stock_retention_effect": bands(rows, ("stock_capture", "stock_retain"), fields),
            "augmented_retention_effect": bands(rows, ("cem_capture", "cem_retain"), fields),
            "stock_vs_augmented_with_retention": bands(rows, ("cem_retain", "stock_retain"), fields),
        }
    write_json(out / "evidence_without_cem_summary.json", reports)


def within_caps(candidate, baseline, caps=CAPS):
    return all(candidate[k] <= baseline[k] + bound + 1e-8 for k, bound in caps.items())


def mean_targets(trials, arm):
    return {k: float(np.mean([t[arm][k] for t in trials])) for k in TARGETS}


def discovery_case(job):
    family, seed, _, repeats, _ = job
    shared, scene, root, prefix = case_start(family, seed)
    c = copy(shared)
    c.requests = []
    actions, records = {}, {}
    for label in ("random", "cem"):
        base, baseline, ranked, draws = cem_discover(c, root, scene.latent, seed, population=6, iterations=2,
                                                     samples=3, refit=label == "cem")
        actions["stock"] = base
        allowed = [(a, p) for a, p in ranked if within_caps(p["expected"], baseline)]
        winner = allowed[0][0] if allowed else None
        actions[label] = winner or base
        records[label] = {"winner_request": None if winner is None else winner.request,
                          "proposals": len(draws), "distinct_admissible": len(ranked),
                          "admissible_with_forecast_cost_caps": len(allowed),
                          "forecasts_with_positive_completion": sum(p["expected"]["corrected_complete"] >
                                                                  baseline["corrected_complete"] for _, p in allowed),
                          "draws": [{k: v for k, v in row.items() if k != "replicas"} for row in draws]}
    outcomes = complete_paths(c, scene, root, prefix, seed, actions, repeats, 7973)
    base = mean_targets(outcomes, "stock")
    for label, record in records.items():
        measured = mean_targets(outcomes, label)
        delta = {k: measured[k] - base[k] for k in TARGETS}
        record["verification_delta"] = delta
        record["verified_positive"] = (
            record["winner_request"] is not None and delta["corrected_complete"] >= 2. / repeats
            and within_caps(measured, base) and delta["final_wrong"] <= 0 and delta["new_identity_mistake"] <= 0.)
    return {"family": family, "seed": seed, "discovery": records, "verification": outcomes}


def nominate(rows, source):
    candidates = [r["discovery"][source] for r in rows if r["discovery"][source]["winner_request"] is not None]
    ordered = sorted(candidates, key=lambda r: (
        -int(r["verified_positive"]), -r["verification_delta"]["corrected_complete"],
        -float(OBJECTIVE @ outcome_vector(r["verification_delta"], TARGETS))))
    return list(dict.fromkeys(tuple(r["winner_request"]) for r in ordered))[:6]


def library_case(job):
    family, seed, samples, repeats, requests = job
    shared, scene, root, prefix = case_start(family, seed)
    c = copy(shared)
    c.requests = []
    stock = c.options_shared(root, renderer="waypoint")
    mpc = c.mpc(root, stock, False, active=True, informative=True).name
    scenarios = posterior_worlds(root, samples, 615_479)
    forecast = dict(zip(stock, rollout_scores(c, root, stock, scenarios), strict=True))
    selected = {}
    approved = {}
    actions = {}
    for label, menu_requests in (("stock", []), *requests.items()):
        new = {}
        for slot, request in enumerate(menu_requests):
            name = f"extra_{slot}"
            action = c.free_candidate(root, name, *request)
            if action is not None:
                new[name] = action
        menu = stock | new
        values = list(forecast.values()) + (list(rollout_scores(c, root, new, scenarios)) if new else [])
        name, prediction = pick(menu, mpc, np.stack(values))
        selected[label] = name
        approved[label] = int(np.count_nonzero(prediction["eligible"][len(stock):]))
        actions[label] = menu[name]
    outcomes = complete_paths(c, scene, root, prefix, seed, actions, repeats, 7991)
    return {"family": family, "seed": seed, "selected": selected, "approved_extras": approved, "verification": outcomes}


def run_discovery(args, out):
    names = ("stock", "random", "cem")
    dev = run_parallel(discovery_case, phase_jobs("discovery", 24, 3, 8), args.jobs, "candidate discovery")
    save_archive(out / "discovery_cases.jsonl.gz", dev)
    requests = {source: nominate(dev, source) for source in names[1:]}
    k = min(map(len, requests.values()))
    requests = {source: values[:k] for source, values in requests.items()}
    output = {"development": bands(dev, names), "requests": requests, "new_requests_per_method": k,
              "requested_rollout_draws_per_candidate": 3,
              "verified_development_cases": {name: sum(r["discovery"][name]["verified_positive"] for r in dev)
                                              for name in names[1:]},
              "searched_paths": {name: {
                  key: sum(r["discovery"][name][key] for r in dev)
                  for key in ("proposals", "distinct_admissible", "admissible_with_forecast_cost_caps",
                              "forecasts_with_positive_completion")} for name in names[1:]},
              "sampling": {"population": 6, "rounds": 2, "means": 3, "verification_tapes": 8}}
    if k:
        test = run_parallel(library_case, phase_jobs("library_test", args.cases, args.samples, 4, requests),
                            args.jobs, "candidate source holdout")
        save_archive(out / "library_cases.jsonl.gz", test)
        output["library_holdout"] = bands(test, names)
        output["library_holdout_cem_vs_random"] = bands(test, ("random", "cem"))
        output["online_uses_extra"] = {arm: sum(r["selected"][arm].startswith("extra_") for r in test) for arm in names}
    write_json(out / "discovery_summary.json", output)
    print("verified offline discovery", output["verified_development_cases"], "requests per menu", k, flush=True)


def renderer_case(job):
    family, seed, _, repeats, _ = job
    c, scene, root, prefix = case_start(family, seed)
    # Rotate through the frozen requests on public states. Only renderings of
    # one preselected request are compared within each paired execution.
    start = (seed - NS["render"]) % len(c.requests)
    for j in range(len(c.requests)):
        request = c.requests[(start + j) % len(c.requests)]
        begin = process_time()
        analytic = c.free_candidate(root, "replay_request", *request)
        elapsed = process_time() - begin
        if analytic is not None:
            break
    else:
        return {"family": family, "seed": seed, "reference_feasible": False}
    begin = process_time()
    residual = c.shape("left", root.x, root.vx, analytic.lateral.goal)
    old = legacy_reference(c, root, analytic, residual)
    native = old or legacy_reference(c, root, analytic, c.zero)
    old_cpu = process_time() - begin
    begin = process_time()
    learned = c.free_candidate(root, analytic.name, *request, renderer="fit")
    learned_cpu = process_time() - begin
    for motion in (native, learned):
        if motion is not None:
            np.testing.assert_allclose(motion.s, analytic.s, atol=1e-12, rtol=0.)
            np.testing.assert_allclose(motion.vs, analytic.vs, atol=1e-12, rtol=0.)
            for k in ("x", "v"):
                np.testing.assert_allclose(getattr(motion.lateral, k)[[0, -1]], getattr(analytic.lateral, k)[[0, -1]],
                                           atol=1e-10, rtol=0.)
    actions = {"analytic": analytic, "pre_cem_prodmp": native or analytic, "cem_fit_prodmp": learned or analytic}
    outcomes = complete_paths(c, scene, root, prefix, seed, actions, repeats, 8381)
    return {"family": family, "seed": seed, "request": request, "reference_feasible": True,
            "feasible": {"pre_cem_first_fit": old is not None, "pre_cem_prodmp": native is not None,
                         "cem_fit_prodmp": learned is not None},
            "cpu_s": {"analytic": elapsed, "pre_cem_prodmp": old_cpu, "cem_fit_prodmp": learned_cpu},
            "verification": outcomes}


def run_renderer(args, out):
    cases = run_parallel(renderer_case, phase_jobs("render", args.cases, 1, 4), args.jobs, "same request rendering")
    save_archive(out / "render_cases.jsonl.gz", cases)
    names = ("analytic", "pre_cem_prodmp", "cem_fit_prodmp")
    valid = [r for r in cases if r["reference_feasible"]]
    report = bands(valid, names)
    for family, values in report.items():
        rows = [r for r in valid if family in ("pooled", r["family"])]
        values["feasible_requests"] = {arm: sum(r["feasible"][arm] for r in rows) for arm in rows[0]["feasible"]}
        values["renderer_cpu_mean_ms"] = {name: 1000. * float(np.mean([r["cpu_s"][name] for r in rows]))
                                           for name in names}
    write_json(out / "render_summary.json", report | {"worlds_without_legal_analytic_request": len(cases) - len(valid)})
    if valid:
        print("same request feasibility", report["pooled"]["feasible_requests"], flush=True)


def diagnose_misses(row):
    family, seed = row["family"], row["seed"]
    c, scene, root, prefix = case_start(family, seed)
    menu = c.options_shared(root)
    mpc = row["selected"]["mpc"]
    stats = []
    for index, saved in enumerate(row["verification"]):
        baseline = saved["mpc"]
        if baseline["corrected_complete"]:
            continue
        future = tape_for(scene.latent, seed, index, SELECTION_TAPE)
        actual, record = real_trial(c, root, prefix, future, [menu[mpc]])
        for outcome in TARGETS:
            assert actual[outcome] == baseline[outcome]
        observed = {station_bin(p["s"]) for p in record.packets if 0 in p["roles_truth"]}
        possible = {}
        capped = []
        for name, action in menu.items():
            if name == mpc:
                continue
            result, _ = real_trial(c, root, prefix, future, [action])
            if result["corrected_complete"]:
                possible[name] = {key: float(result[key] - actual[key]) for key in CAPS}
                if within_caps(result, actual):
                    capped.append(name)
        stats.append({"replica": index, "reached_endpoint": record.motion[-1][1] >= END - .28,
                      "bins_without_target_detection": [j for j in range(BINS) if j not in observed],
                      "hard_budget_recovery_requests": possible, "relative_budget_recovery_requests": capped})
    return {"family": family, "seed": seed, "mpc_failures": stats}


def run_ceiling(args, out):
    source = out / "selection_holdout.jsonl.gz"
    rows = run_parallel(diagnose_misses, read_rows(source), args.jobs, "failed scans replay")
    save_archive(out / "misses_cases.jsonl.gz", rows)
    output = {}
    for family in ("pooled", *FAMILIES):
        cases = [r for r in rows if family in ("pooled", r["family"])]
        records = [fail for r in cases for fail in r["mpc_failures"]]
        output[family] = {
            "failed_trials": len(records),
            "missing_raw_target_in_at_least_one_bin": sum(bool(v["bins_without_target_detection"]) for v in records),
            "missed_endpoint": sum(not v["reached_endpoint"] for v in records),
            "has_hard_budget_first_action_to_complete": sum(bool(v["hard_budget_recovery_requests"]) for v in records),
            "has_execution_only_matched_first_action_to_complete": sum(
                any(all(delta[k] <= CAPS[k] + 1e-8 for k in ("tcp_mm", "turn_degs", "seconds"))
                    for delta in v["hard_budget_recovery_requests"].values()) for v in records),
            "has_matched_cost_first_action_to_complete": sum(bool(v["relative_budget_recovery_requests"])
                                                            for v in records),
        }
    write_json(out / "misses_summary.json", output | {"selection_archive_sha256": fingerprint(source)})


def replay_relaxed_choice(job):
    row, fixed = job
    family, seed = row["family"], row["seed"]
    c, scene, root, prefix = case_start(family, seed)
    menu = c.options_shared(root)
    names = row["names"]
    if names != list(menu):
        raise ValueError("current candidate menu differs from the saved forecasts")
    incumbent = row["selected"]["mpc"]
    choices = {"fixed_probe": fixed if fixed in menu else incumbent}
    for arm, prediction in (("teacher", "teacher_prediction"), ("learned_b_reference_fit", "extra_prediction")):
        deltas = row[prediction]["delta"]
        original, _, eligible = choose_from_deltas(names, incumbent, deltas, targets=TARGETS)
        if original != row["selected"][arm] or eligible.tolist() != row[prediction]["eligible"]:
            raise ValueError("saved forecasts do not reproduce the original selection")
        choices[f"{arm}_relaxed"], _, _ = choose_from_deltas(
            names, incumbent, deltas, targets=TARGETS, caps=CAPS_WITHOUT_RELATIVE_MOTION)
    archived_actions = {action: arm for arm, action in row["selected"].items()}
    new_actions = {arm: menu[action] for arm, action in choices.items() if action not in archived_actions}
    replays = complete_paths(c, scene, root, prefix, seed, new_actions, len(row["verification"]), SELECTION_TAPE)
    keep = ("mpc", "teacher", "learned_b_reference_fit")
    verification = []
    for saved, replayed in zip(row["verification"], replays, strict=True):
        trial = {arm: saved[arm] for arm in keep}
        for arm, action in choices.items():
            trial[arm] = replayed[arm] if arm in replayed else saved[archived_actions[action]]
        verification.append(trial)
    return {"family": family, "seed": seed, "selected": {arm: row["selected"][arm] for arm in keep} | choices,
            "verification": verification}


def run_relaxed(args, out):
    development = read_rows(out / "selection_dev.jsonl.gz")
    source = out / "selection_holdout.jsonl.gz"
    fixed = max(FIXED, key=lambda arm: sum(t[arm]["corrected_complete"]
                                         for case in development for t in case["verification"]))
    rows = run_parallel(replay_relaxed_choice, [(row, fixed) for row in read_rows(source)],
                        args.jobs, "relaxed travel and turn caps")
    save_archive(out / "relaxed_choice_cases.jsonl.gz", rows)
    names = ("mpc", "teacher", "learned_b_reference_fit", "fixed_probe", "teacher_relaxed",
             "learned_b_reference_fit_relaxed")
    reports = bands(rows, names)
    for family, report in reports.items():
        cohort = [r for r in rows if family in ("pooled", r["family"])]
        report["decisions_changed_from_mpc"] = {arm: sum(r["selected"][arm] != r["selected"]["mpc"]
                                                        for r in cohort) for arm in names[1:]}
        report["completion_wins_losses"] = {}
        for arm in names[1:]:
            differences = [t[arm]["corrected_complete"] - t["mpc"]["corrected_complete"]
                           for r in cohort for t in r["verification"]]
            report["completion_wins_losses"][arm] = [sum(d > 0 for d in differences), sum(d < 0 for d in differences)]
    misses = [failure for row in read_rows(out / "misses_cases.jsonl.gz") for failure in row["mpc_failures"]]
    zero_delta = dict.fromkeys(CAPS_WITHOUT_RELATIVE_MOTION, 0.)
    surviving = [[metrics for metrics in failure["hard_budget_recovery_requests"].values()
                  if within_caps(metrics, zero_delta, CAPS_WITHOUT_RELATIVE_MOTION)]
                 for failure in misses]
    min_travel = [min(m["tcp_mm"] for m in options) for options in surviving if options]
    write_json(out / "relaxed_choice_summary.json", {
        "evaluation": "retrospective on the previous holdout",
        "remaining_forecast_caps": CAPS_WITHOUT_RELATIVE_MOTION,
        "comparison_margins_from_original_protocol": MARGINS,
        "fixed_probe_chosen_by_development_completion": fixed,
        "selection_archive_sha256": fingerprint(source), "comparison": reports,
        "failure_oracle": {"failed_tapes": len(misses), "tapes_with_eligible_alternative": sum(map(bool, surviving)),
                           "eligible_alternatives": sum(map(len, surviving)),
                           "median_least_additional_travel_mm": float(np.median(min_travel)) if min_travel else None},
    })


def audit_recorded_input(directory, out):
    import h5py
    from PIL import Image

    annotations = ("annot", "identity", "label", "vessel", "segmentation", "mask")
    files = sorted(list(directory.rglob("*.h5")) + list(directory.rglob("*.hdf5")))
    if not files:
        raise ValueError("no aligned HDF5 recordings in the supplied directory")
    sidecars = sum(any(tag in p.name.lower() for tag in annotations)
                   for p in directory.rglob("*") if p.is_file())
    audit = Counter()
    timings, alignment = [], []
    metadata_fields = set()
    for file in files:
        with h5py.File(file) as recording:
            frames = recording["ultrasound/jpeg"]
            if not len(frames):
                continue
            audit["nonempty_ultrasound_files"] += 1
            audit["image_samples"] += len(frames)
            audit["image_tcp_equal_length"] += int(len(recording["tcp/pose_m_rad"]) == len(frames))
            timestamps = recording["timestamp_ns"][:]
            dt = np.diff(timestamps) * 1.e-6
            audit["monotonic_timestamps"] += int(len(timestamps) == len(frames) and np.all(dt > 0))
            timings.extend(dt[dt > 0])
            alignment.extend(np.abs(recording["alignment/tcp/match_error_ns"][:]) * 1.e-6)
            middle = len(frames) // 2
            with Image.open(BytesIO(frames[middle].tobytes())) as image:
                image.load()
                audit["files_with_decoded_jpeg_sample"] += 1
            metadata_fields.update(json.loads(recording["ultrasound/metadata_json"][middle]).keys())
            fields = []
            recording.visititems(lambda key, item: fields.append(key) if isinstance(item, h5py.Dataset) else None)
            audit["files_with_identity_label_or_mask_dataset"] += int(
                any(tag in key.lower() for key in fields for tag in annotations))
    write_json(out / "recorded_input_audit.json", dict(audit) | {
        "aligned_hdf5_files": len(files), "matching_annotation_sidecar_files": sidecars,
        "positive_frame_interval_ms_p50_p95": np.quantile(timings, [.5, .95]),
        "tcp_match_absolute_error_ms_p50_p95": np.quantile(alignment, [.5, .95]),
        "ultrasound_metadata_field_names": sorted(metadata_fields),
        "analysis": "readiness from an existing contact study; no patient images, identifiers, or poses exported",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("evidence", "evidence_without_cem", "selection", "discovery", "render",
                                         "ceiling", "relaxed", "recordings"))
    parser.add_argument("--source", type=Path, default=Path("results/visible_atlas_v1"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=64, help="held-out scenes per family")
    parser.add_argument("--samples", type=int, default=16, help="public posterior draws per feasible action")
    parser.add_argument("--jobs", type=int, default=20)
    parser.add_argument("--recordings", type=Path, help="directory of aligned ultrasound and TCP recordings")
    args = parser.parse_args()
    if min(args.jobs, args.cases, args.samples) < 1:
        parser.error("positive case, worker and sample counts required")
    args.out.mkdir(exist_ok=True, parents=True)
    if args.phase == "recordings":
        if args.recordings is None:
            parser.error("--recordings is required")
        audit_recorded_input(args.recordings, args.out)
        return
    if args.phase == "evidence_without_cem":
        run_evidence_without_cem(args, args.out)
        return
    source = args.source
    if args.phase != "evidence":
        source = args.out / "shadow_library"
        source.mkdir(exist_ok=True)
        data = json.loads((args.source / "library.json").read_text())
        write_json(source / "library.json", data | {"shadow_scale": SHADOW_SCALE})
        copyfile(args.source / "prodmp_fit.npy", source / "prodmp_fit.npy")
    configure(source)
    {"evidence": run_evidence, "selection": run_selection, "discovery": run_discovery,
     "render": run_renderer, "ceiling": run_ceiling, "relaxed": run_relaxed}[args.phase](args, args.out)


if __name__ == "__main__":
    main()
