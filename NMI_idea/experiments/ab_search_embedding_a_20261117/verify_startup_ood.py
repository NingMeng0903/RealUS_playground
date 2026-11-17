"""Closed-loop factorial comparison for startup acquisition failure."""

import argparse
from collections import Counter
from copy import copy
from dataclasses import dataclass, replace
from itertools import product
import json
from pathlib import Path

import numpy as np

from adaptive_controller import AdaptiveController, FreeCandidate, START, station_bin
from adaptive_teacher import cem_discover, fit_nominal_primitive, posterior_worlds
from audit_prodmp_weights import legacy_reference
from diagnose_b_selection import run_parallel, save_archive
from learned_recovery import OBJECTIVE, choose_from_deltas, outcome_vector
from recovery_world import FOV_MAX, FOV_MIN, REFERENCE_KNEE_DEG, RecoveryWorld, Scene, noise_tape
from run_adaptive_recovery import real_trial, tape_for
from run_learned_recovery import CONTROLLERS, FAMILIES, configure, fingerprint, read_rows, write_json
from verify_recovery_components import (CAPS_WITHOUT_RELATIVE_MOTION, FIELDS, TARGETS, bands,
                                        rollout_scores, within_caps)


SEED = {"development": 35_410_000, "holdout": 36_410_000}
SCREEN_COUNTS = {"development": 160, "holdout": 224}
QUOTA = {"development": 12, "holdout": 16}
START_GROUPS = ("zero", "one")
MENUS = ("stock", "random", "cem")
FIT_TAPE = 9811
TEST_TAPE = 9812
FIT_REPEATS = 6
TEST_REPEATS = 8
PUBLIC_ROLLOUTS = 8
BLANK_SEED = 36_810_000
BLANK_SCREEN_COUNT = 1024
BLANK_QUOTA = 16
OUTCOMES = (*FIELDS, "first_interval_recaptured", "both_vessels_observed")
MAIN_ARMS = tuple(f"{menu}_{render}_{memory}" for menu, render, memory in
                  product(MENUS, ("old", "cemfit"), ("capture", "retain")))
AUX_ARMS = ("mpc", "cem_randomfit_retain")


@dataclass(frozen=True)
class PerturbedScene(Scene):
    lateral_jitter: float
    depth_jitter: float
    shadow_width_factor: float
    shadow_shift: float


class StartupWorld(RecoveryWorld):
    """The scout can move laterally out of the earlier acoustic shadow."""

    def __init__(self, family, reference):
        super().__init__(family, reference)
        self.hypotheses = np.asarray(list(product(range(2), (-9., -3., 0., 3., 9.), (-1., 1.), range(2))))
        track, lateral, side, layout = self.hypotheses.T
        self.prior = np.select((lateral == 0., abs(lateral) == 3.), (.40, .20), default=.10) / 8.
        self.shape = self._geometry_grid(track, lateral, side, layout)
        self.shadow_center = np.where(layout == 0., 12.4, 15.4)

    def imaging(self, s, x, angle, *, width=1., lane_shift=0., depth_shift=0.):
        means, visibility = super().predict(s, x, angle)
        means[:, :, 1] += depth_shift
        lateral, side, layout = self.hypotheses[:, 1:].T
        lane = self.nominal(START)[0] + .10 * lateral + .45 * side + lane_shift
        longitudinal = np.exp(-.5 * ((s - (START + 1.)) / (3.6 * width))**4)
        shadow = .998 * longitudinal / (1. + ((x - lane) / (3.8 * width))**6)
        deeper = means[:, :2, 1] >= means[:, [1, 0], 1]
        attenuated = (layout[:, None] == 0) | deeper
        visibility[:, :2] *= 1. - shadow[:, None] * attenuated
        visibility *= np.all((means >= FOV_MIN) & (means <= FOV_MAX), axis=-1)
        return means, visibility

    def predict(self, s, x, angle, *, alias=False):
        return self.imaging(s, x, angle)

    def observe(self, scene, frame, s, x):
        if not isinstance(scene, PerturbedScene):
            return super().observe(scene, frame, s, x)
        means, visibility = self.imaging(
            s, x - scene.lateral_jitter, REFERENCE_KNEE_DEG, width=scene.shadow_width_factor,
            lane_shift=scene.shadow_shift, depth_shift=scene.depth_jitter,
        )
        tape = scene.tape
        return self._emit_packet(means[scene.latent], visibility[scene.latent], x, tape.emit[frame],
                                 tape.error[frame], tape.uniform_clutter[frame], tape.outage[frame])

    def startup_draw(self, seed, split):
        rng = np.random.default_rng(seed)
        lateral = abs(self.hypotheses[:, 1])
        eligible = lateral <= 3. if split == "development" else lateral == 9.
        weights = self.prior * eligible
        latent = int(rng.choice(len(weights), p=weights / weights.sum()))
        tape = noise_tape(rng)
        if split == "development":
            return Scene(latent, tape)
        return PerturbedScene(latent, tape, rng.uniform(-.6, .6), rng.uniform(-.55, .55),
                              rng.uniform(1.25, 1.55), rng.uniform(-.7, .7))


class StartupController(AdaptiveController):
    confirm_at_capture = False

    def retain_visible(self, state, record, conditional):
        super().retain_visible(state, record, conditional, confirm_at_capture=self.confirm_at_capture)


def startup_controller(world, sections, coefficients, _fit, _requests, *, evidence_mode, material_surface):
    return StartupController(StartupWorld(world.family, world.reference), sections, coefficients,
                             evidence_mode=evidence_mode)


def scout(controller, scene):
    root, prefix = controller.initial_scan(scene, detailed=True)
    hold = controller.candidate(root, "wait", root.s, 0., root.x, False)
    if hold is None:
        raise ValueError("common scout hold is infeasible")
    controller.execute(scene, root, prefix, hold)
    return root, prefix


def observed_vessels(packets):
    return {role for p in packets for role in p["roles_truth"] if role < 2}


def screen(job):
    family, seed, split = job
    c = CONTROLLERS[family]
    scene = c.world.startup_draw(seed, split)
    root, prefix = scout(c, scene)
    vessels = observed_vessels(prefix.packets)
    return {"family": family, "seed": seed, "group": ("zero", "one", "two")[len(vessels)],
            "observed_vessels": sorted(vessels), "blank_packets": sum(not p["y"] for p in prefix.packets),
            "scout_frame": root.frame}


def run_screen(args):
    report = {}
    for split, start in SEED.items():
        jobs = [(family, start + 10_000 * i + j, split) for i, family in enumerate(FAMILIES)
                for j in range(SCREEN_COUNTS[split])]
        rows = run_parallel(screen, jobs, args.jobs, f"{split} startup screening")
        selected = []
        for family, group in product(FAMILIES, START_GROUPS):
            eligible = [row for row in rows if row["family"] == family and row["group"] == group]
            selected.extend(eligible[:QUOTA[split]])
        save_archive(args.out / f"{split}_screening.jsonl.gz", rows)
        save_archive(args.out / f"{split}_starts.jsonl.gz", selected)
        report[split] = {"screened": dict(Counter(r["group"] for r in rows)),
                         "selected": {family: dict(Counter(r["group"] for r in selected if r["family"] == family))
                                      for family in FAMILIES}}
    write_json(args.out / "screening_summary.json", report)


def run_blank_screen(args):
    jobs = [(family, BLANK_SEED + 10_000 * i + j, "holdout") for i, family in enumerate(FAMILIES)
            for j in range(BLANK_SCREEN_COUNT)]
    rows = run_parallel(screen, jobs, args.jobs, "entirely empty scout screening")
    empty = [r for r in rows if r["blank_packets"] == r["scout_frame"] + 1]
    chosen = []
    for family in FAMILIES:
        chosen.extend([r for r in empty if r["family"] == family][:BLANK_QUOTA])
    save_archive(args.out / "blank_screening.jsonl.gz", rows)
    save_archive(args.out / "blank_starts.jsonl.gz", chosen)
    write_json(args.out / "blank_screening_summary.json", {
        "screened": dict(Counter(r["family"] for r in rows)),
        "empty_scout": dict(Counter(r["family"] for r in empty)),
        "selected": dict(Counter(r["family"] for r in chosen)),
    })


def execute_trial(c, root, prefix, future, action):
    metrics, record = real_trial(c, root, prefix, future, [action])
    if metrics["max_handoff_error"] >= 1e-7:
        raise ValueError("motion handoff was discontinuous")
    recaptured = any(p["frame"] > root.frame and station_bin(p["s"]) == 0 and 0 in p["roles_truth"]
                     for p in record.packets)
    return {k: metrics[k] for k in FIELDS} | {
        "first_interval_recaptured": recaptured, "both_vessels_observed": len(observed_vessels(record.packets)) == 2,
    }


def discover(row):
    c = CONTROLLERS[row["family"]]
    seed = row["seed"]
    scene = c.world.startup_draw(seed, "development")
    root, prefix = scout(c, scene)
    futures = [tape_for(scene.latent, seed, t, FIT_TAPE) for t in range(FIT_REPEATS)]
    demonstrations = {}
    for label in ("random", "cem"):
        _, baseline, ranked, draws = cem_discover(c, root, scene.latent, seed, population=6, iterations=2,
                                                   samples=3, refit=label == "cem")
        record = {"proposals": len(draws), "admissible": len(ranked), "demo": None}
        for action, forecast in ranked:
            if not within_caps(forecast["expected"], baseline, CAPS_WITHOUT_RELATIVE_MOTION):
                continue
            trials = [execute_trial(c, root, prefix, future, action) for future in futures]
            record["finalist"] = {"request": action.request, "trials": trials}
            if sum(t["corrected_complete"] for t in trials) >= 4 and not any(t["final_wrong"] for t in trials):
                record["demo"] = {"nominal_x_vx": np.column_stack((action.lateral.x, action.lateral.v)),
                                  "request": action.request}
            break
        demonstrations[label] = record
    return row | {"discovery": demonstrations}


def ranked_examples(rows, source):
    candidates = [r["discovery"][source] for r in rows if r["discovery"][source]["demo"] is not None]

    def ranking(record):
        outcomes = record["finalist"]["trials"]
        u = np.mean([outcome_vector(t, TARGETS) for t in outcomes], axis=0)
        return u[0], sum(t["first_interval_recaptured"] for t in outcomes), float(u @ OBJECTIVE)

    return [r["demo"] for r in sorted(candidates, key=ranking, reverse=True)]


def run_discover(args):
    starts = read_rows(args.out / "development_starts.jsonl.gz")
    rows = run_parallel(discover, starts, args.jobs, "startup motion discovery")
    save_archive(args.out / "development_discovery.jsonl.gz", rows)
    samples = {source: ranked_examples(rows, source) for source in ("random", "cem")}
    qualified = {source: len(examples) for source, examples in samples.items()}
    count = min(qualified.values())
    primitive = CONTROLLERS[FAMILIES[0]].primitive
    fitted = {source: fit_nominal_primitive(primitive, examples[:count]) for source, examples in samples.items()}
    requests = {source: list(dict.fromkeys(tuple(d["request"]) for d in examples))[:6]
                for source, examples in samples.items()}
    request_count = min(map(len, requests.values()))
    if any(weights is None for weights in fitted.values()) or request_count == 0:
        raise ValueError(f"insufficient qualified independent demonstration shapes: {qualified}")
    for source, weights in fitted.items():
        np.save(args.out / f"{source}_fit.npy", weights)
    write_json(args.out / "development_bundle.json", {
        "qualified": qualified,
        "matched_fit_examples": count, "requests": {k: v[:request_count] for k, v in requests.items()},
        "requests_per_source": request_count,
        "fit_sha256": {source: fingerprint(args.out / f"{source}_fit.npy") for source in fitted},
        "proposals": {k: sum(r["discovery"][k]["proposals"] for r in rows) for k in samples},
    })


def public_choices(c, root, bundle):
    stock = c.options_shared(root)
    incumbent = c.mpc(root, stock, False, active=True, informative=True).name
    scenarios = posterior_worlds(root, PUBLIC_ROLLOUTS, 571_819)
    values = rollout_scores(c, root, stock, scenarios)
    reference = values[list(stock).index(incumbent)]
    choices = {"mpc": stock[incumbent]}
    for source, requests in (("stock", []), *bundle["requests"].items()):
        extras = {}
        for i, request in enumerate(requests):
            name = f"{source}_extra_{i}"
            action = c.free_candidate(root, name, *request)
            if action is not None:
                extras[name] = action
        menu = stock | extras
        estimates = np.vstack((values, rollout_scores(c, root, extras, scenarios))) if extras else values
        picked, _, _ = choose_from_deltas(list(menu), incumbent, estimates - reference, targets=TARGETS,
                                          caps=CAPS_WITHOUT_RELATIVE_MOTION)
        choices[source] = menu[picked]
    return choices


def render_action(c, root, action, renderer, fits):
    if renderer == "nominal" or not isinstance(action, FreeCandidate):
        return action, "original"
    if renderer == "old":
        weights = c.shape("left", root.x, root.vx, action.lateral.goal)
        candidate = legacy_reference(c, root, action, weights)
        if candidate is None:
            candidate = legacy_reference(c, root, action, c.zero)
    else:
        c.learned = fits[renderer]
        candidate = c.free_candidate(root, action.name, *action.request, renderer="fit")
    if candidate is None:
        return action, "analytic_fallback"
    return candidate, renderer


def feasible_requests(c, root, requests, fits):
    counts = Counter(analytic=0, old=0, cemfit=0, randomfit=0)
    for i, request in enumerate(requests):
        action = c.free_candidate(root, f"feasibility_{i}", *request)
        if action is None:
            continue
        counts["analytic"] += 1
        for renderer in ("old", "cemfit", "randomfit"):
            _, used = render_action(c, root, action, renderer, fits)
            counts[renderer] += int(used == renderer)
    return dict(counts)


def replay(job):
    row, bundle, fits, blank_choice = job
    c = CONTROLLERS[row["family"]]
    seed = row["seed"]
    scene = c.world.startup_draw(seed, "holdout")
    root, prefix = scout(c, scene)
    capture_controller = copy(c)
    capture_controller.confirm_at_capture = True
    capture_root, capture_prefix = scout(capture_controller, scene)
    np.testing.assert_array_equal(capture_root.belief, root.belief)
    # The measured packets agree even when the detailed ledger differs.
    assert [(p["y"], p["s"], p["x"]) for p in capture_prefix.packets] == [
        (p["y"], p["s"], p["x"]) for p in prefix.packets]
    assert capture_prefix.motion == prefix.motion
    if blank_choice is None:
        choices = public_choices(c, root, bundle)
    else:
        assert all(not p["y"] for p in prefix.packets)
        expected_belief, choices = blank_choice
        np.testing.assert_array_equal(root.belief, expected_belief)
    admissible = feasible_requests(c, root, bundle["requests"]["cem"], fits)
    arms = {arm: tuple(arm.split("_")) for arm in MAIN_ARMS}
    arms |= {"mpc": ("mpc", "nominal", "retain"), "cem_randomfit_retain": ("cem", "randomfit", "retain")}
    cache = {}
    planned = {}
    for arm, (menu, renderer, _) in arms.items():
        action = choices[menu]
        key = (id(action), renderer)
        if key not in cache:
            cache[key] = render_action(c, root, action, renderer, fits)
        planned[arm] = cache[key]
    outcomes = []
    for t in range(TEST_REPEATS):
        tape = tape_for(scene.latent, seed, t, TEST_TAPE).tape
        future = replace(scene, tape=tape)
        executed = {}
        trials = {}
        for arm, (_, _, memory) in arms.items():
            action, _ = planned[arm]
            key = (id(action), memory)
            if key not in executed:
                if memory == "retain":
                    executed[key] = execute_trial(c, root, prefix, future, action)
                else:
                    executed[key] = execute_trial(capture_controller, capture_root, capture_prefix, future, action)
            trials[arm] = executed[key]
        outcomes.append(trials)
    return row | {"offset_mm": float(c.world.hypotheses[scene.latent, 1] + scene.lateral_jitter),
                  "depth_jitter_mm": scene.depth_jitter, "shadow_width_factor": scene.shadow_width_factor,
                  "chosen": {k: {"name": v.name, "request": getattr(v, "request", None)} for k, v in choices.items()},
                  "request_feasibility": admissible,
                  "renderer": {arm: used for arm, (_, used) in planned.items()}, "verification": outcomes}


def run_verify(args):
    p = args.out / "development_bundle.json"
    bundle = json.loads(p.read_text())
    weights = {}
    for source, renderer in (("cem", "cemfit"), ("random", "randomfit")):
        f = args.out / f"{source}_fit.npy"
        if fingerprint(f) != bundle["fit_sha256"][source]:
            raise ValueError("fit differs from the frozen development bundle")
        weights[renderer] = np.load(f, allow_pickle=False)
    blank = args.phase == "verify_blank"
    prefix = "blank" if blank else "holdout"
    starts = read_rows(args.out / f"{prefix}_starts.jsonl.gz")
    fixed = {}
    if blank:
        # All candidate-free scouts at the same public pose have the same posterior.
        for family in FAMILIES:
            controller = CONTROLLERS[family]
            seed = next(r["seed"] for r in starts if r["family"] == family)
            root, trace = scout(controller, controller.world.startup_draw(seed, "holdout"))
            assert all(not packet["y"] for packet in trace.packets)
            fixed[family] = (root.belief, public_choices(controller, root, bundle))
    jobs = [(r, bundle, weights, fixed.get(r["family"])) for r in starts]
    rows = run_parallel(replay, jobs, args.jobs, f"{prefix} startup OOD factorial")
    save_archive(args.out / f"{prefix}_factorial.jsonl.gz", rows)
    contrasts = {
        "all_arms": ("mpc", *MAIN_ARMS, "cem_randomfit_retain"),
        "cem_proposal_effect": ("stock_cemfit_retain", "cem_cemfit_retain"),
        "cem_vs_random_proposals": ("random_cemfit_retain", "cem_cemfit_retain"),
        "cem_fit_effect": ("cem_old_retain", "cem_cemfit_retain"),
        "cem_fit_vs_random_fit": ("cem_randomfit_retain", "cem_cemfit_retain"),
        "retention_effect_in_full_stack": ("cem_cemfit_capture", "cem_cemfit_retain"),
        "full_stack_vs_stock_stack": ("stock_old_capture", "cem_cemfit_retain"),
        "full_stack_vs_stock_retention": ("stock_old_retain", "cem_cemfit_retain"),
    }
    report = {}
    groups = ("all",) if blank else ("all", *START_GROUPS)
    for group in groups:
        cohort = [r for r in rows if group in ("all", r["group"])]
        report[group] = {key: bands(cohort, arms, OUTCOMES) for key, arms in contrasts.items()}
        report[group]["first_interval_recaptures"] = {
            arm: sum(t[arm]["first_interval_recaptured"] for r in cohort for t in r["verification"])
            for arm in (*MAIN_ARMS, *AUX_ARMS)}
        report[group]["renderers"] = {arm: dict(Counter(r["renderer"][arm] for r in cohort))
                                        for arm in (*MAIN_ARMS, *AUX_ARMS)}
        report[group]["request_feasibility"] = {m: sum(r["request_feasibility"][m] for r in cohort)
                                                for m in ("analytic", "old", "cemfit", "randomfit")}
        report[group]["tapes"] = sum(len(r["verification"]) for r in cohort)
    write_json(args.out / f"{prefix}_summary.json", {"development_bundle_sha256": fingerprint(p), "groups": report})


def replay_nominal(row):
    c = CONTROLLERS[row["family"]]
    seed = row["seed"]
    scene = c.world.startup_draw(seed, "holdout")
    root, prefix = scout(c, scene)
    choice = row["chosen"]["cem"]
    request = choice["request"]
    action = (c.options_shared(root)[choice["name"]] if request is None
              else c.free_candidate(root, choice["name"], *request))
    if action is None:
        raise ValueError("archived first request is no longer admissible")
    outputs = []
    for t, saved in enumerate(row["verification"]):
        future = replace(scene, tape=tape_for(scene.latent, seed, t, TEST_TAPE).tape)
        result = execute_trial(c, root, prefix, future, action)
        paired = {arm: saved[arm] for arm in ("cem_old_retain", "cem_cemfit_retain")}
        for arm in paired:
            if row["renderer"][arm] in ("analytic_fallback", "original"):
                assert paired[arm] == result
        outputs.append(paired | {"cem_nominal_retain": result})
    return {"family": row["family"], "seed": seed, "group": row["group"], "verification": outputs}


def run_nominal(args):
    report = {}
    for study in ("holdout", "blank"):
        path = args.out / f"{study}_factorial.jsonl.gz"
        rows = run_parallel(replay_nominal, read_rows(path), args.jobs, f"{study} nominal CEM requests")
        save_archive(args.out / f"{study}_nominal_requests.jsonl.gz", rows)
        report[study] = {"factorial_sha256": fingerprint(path)}
        names = ("cem_nominal_retain", "cem_old_retain", "cem_cemfit_retain")
        for group in ("all", *START_GROUPS):
            cohort = [row for row in rows if group in ("all", row["group"])]
            if cohort:
                report[study][group] = bands(cohort, names, OUTCOMES)
    write_json(args.out / "nominal_request_summary.json", report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("screen", "discover", "verify", "screen_blank", "verify_blank",
                                     "verify_nominal"))
    p.add_argument("--source", type=Path, default=Path("results/visible_atlas_v1"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=20)
    args = p.parse_args()
    if args.jobs < 1:
        p.error("--jobs must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    configure(args.source, controller_type=startup_controller)
    {"screen": run_screen, "discover": run_discover, "verify": run_verify,
     "screen_blank": run_blank_screen, "verify_blank": run_verify, "verify_nominal": run_nominal}[args.phase](args)


if __name__ == "__main__":
    main()
