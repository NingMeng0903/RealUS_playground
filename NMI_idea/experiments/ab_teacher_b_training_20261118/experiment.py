"""Shared ProDMP executor and paired labels for the second-stage B comparison."""

from dataclasses import replace
import json
from pathlib import Path
import sys
from time import process_time

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
GENERATOR = HERE.parent / "ab_teacher_generation_20261118"
sys.path.insert(0, str(GENERATOR))
import study as generator

import benchmark_arbitrary_start as scans
from adaptive_controller import VISIBLE_EVIDENCE
from adaptive_teacher import evaluate, posterior_worlds
from benchmark_window_encoding import choice_from_predictions
from learned_recovery import CEMMenuController, VISIBLE_TARGETS, features_for_menu, predict_deltas
from run_learned_recovery import configure as configure_controllers, fingerprint, read_rows
from train_startup_b import actual_values
from verify_startup_ood import PerturbedScene, StartupWorld
from window_embedding import (ViewCalibration, detection_features, path_views, signed_margins,
                              target_detection_probabilities)


SOURCES = ("random", "direct", "evidence")
CELLS = generator.CELLS
FIELDS = generator.FIELDS
PROPOSALS = 4
LABEL_TAPES = 8
DEV_TAPES = 4
TEST_TAPES = 8
STOCK_REQUESTS = tuple(name for name in generator.STOCK_FIXED if name != "recover") + ("short", "revisit")
GENERATOR_FREEZE = GENERATOR / "results" / "teacher_generation_v1" / "frozen.json"
ORIGINAL_RESULTS = generator.EXPERIMENT / "results"
CALIBRATION = ORIGINAL_RESULTS / "window_encoding_v1" / "calibration.json"
OLD_FREEZE = generator.EXISTING_B.with_name("frozen.json")
BASE_FILES = (
    ORIGINAL_RESULTS / "arbitrary_start_v1" / "training.jsonl.gz",
    ORIGINAL_RESULTS / "first_interval_data_v1" / "additional_training.jsonl.gz",
    ORIGINAL_RESULTS / "identity_recovery_data_v1" / "unconditioned_train.jsonl.gz",
)
NEW_REQUESTS = {}
MODELS = {}


class SharedController(CEMMenuController):
    def expanded_menu(self, state, *, renderer="fit"):
        stock = super().options_shared(state, expanded=True, renderer=renderer)
        new, aliases = generator.novel_actions(self, state, NEW_REQUESTS, stock)
        return stock | new, aliases

    def options_shared(self, state, *, expanded=True, renderer="fit"):
        if expanded:
            return self.expanded_menu(state, renderer=renderer)[0]
        return super().options_shared(state, expanded=False, renderer=renderer)

    def choose_mpc(self, state, *, expanded, open_decision=False, renderer="fit"):
        return super().choose_mpc(state, expanded=expanded, open_decision=open_decision, renderer=renderer)


def controller_for(world, sections, coefficients, fit, requests, *, evidence_mode, material_surface):
    if evidence_mode != VISIBLE_EVIDENCE or material_surface is None:
        raise ValueError("the shared expanded controller requires the existing visible-evidence atlas")
    return SharedController(StartupWorld(world.family, world.reference), sections, coefficients, fit, requests,
                            evidence_mode=evidence_mode, material_surface=material_surface)


def configure():
    freeze = json.loads(GENERATOR_FREEZE.read_text())
    names = sorted(set().union(*(freeze["libraries"][source] for source in SOURCES)))
    NEW_REQUESTS.update({name: generator.REQUESTS[name] for name in names})
    configure_controllers(generator.SOURCE, controller_type=controller_for)
    scans.CALIBRATION = ViewCalibration(**json.loads(CALIBRATION.read_text())["parameters"])
    MODELS["old_b"] = joblib.load(generator.EXISTING_B)["unconditioned_b"]
    paths = (GENERATOR_FREEZE, GENERATOR / "study.py", generator.SOURCE / "library.json",
             generator.SOURCE / "prodmp_fit.npy", generator.MATERIAL_SURFACE, CALIBRATION,
             generator.EXISTING_B, OLD_FREEZE, *BASE_FILES)
    digests = {str(path.relative_to(HERE.parent)): fingerprint(path) for path in paths}
    old = json.loads(OLD_FREEZE.read_text())
    checked = {
        generator.SOURCE / "library.json": old["shared_sha256"]["library"],
        generator.SOURCE / "prodmp_fit.npy": old["shared_sha256"]["prodmp"],
        CALIBRATION: old["shared_sha256"]["calibration"],
        generator.EXISTING_B: old["models_sha256"],
    }
    for path, expected in checked.items():
        if digests[str(path.relative_to(HERE.parent))] != expected:
            raise ValueError(f"frozen source {path.name} changed")
    return digests


def configure_models(models):
    MODELS.update({f"{name}_b": fitted for name, fitted in models.items()})


def scenario(cell, seed):
    return generator.scenario(cell, seed)


def base_training():
    rows = [row for path in BASE_FILES for row in read_rows(path)]
    features = np.concatenate([scans.retrained_inputs(row) for row in rows])
    labels = np.concatenate([actual_values(row) for row in rows])
    return features, labels, {int(row["seed"]) for row in rows}


def at_handoff(row):
    controller, scene, state, prefix = generator.checkpoint(row)
    started = process_time()
    menu, aliases = controller.expanded_menu(state)
    incumbent = generator.mpc_action(controller, state, menu).name
    return controller, scene, state, prefix, menu, aliases, incumbent, process_time() - started


def policy_inputs(controller, state, prefix, menu, incumbent):
    view_paths = [path_views(action) for action in menu.values()]
    margins = [signed_margins(controller.world, state, views) for views in view_paths]
    detection = [detection_features(target_detection_probabilities(controller.world, state, views, scans.CALIBRATION),
                                    views, state.query_bin()) for views in view_paths]
    features = features_for_menu(controller, state, menu, incumbent, prefix.packets[-1]["association"])
    info = {"names": list(menu), "incumbent": incumbent, "first_required_bin": state.first_required_bin,
            "features": features, "margins": margins, "calibrated_detection": detection}
    return {"names": list(menu), "incumbent": incumbent, "features": scans.retrained_inputs(info)}


def label_worlds(state, public_packets, seed):
    samples = posterior_worlds(state, LABEL_TAPES, seed + 18301)
    empty_run = len(public_packets) >= 2 and all(not p["y"] for p in public_packets[-2:])
    for i, sample in enumerate(samples):
        rng = np.random.default_rng([seed, 18302, i])
        tape = sample.tape
        if empty_run:
            # Predict an outage continuation from observed empty packets.
            outage = tape.outage.copy()
            end = state.frame + int(rng.integers(2, 5)) + 1
            outage[state.frame + 1:end] = 0.
            tape = replace(tape, outage=outage)
        tape = replace(tape, error=tape.error * scans.NOISE_SCALE)
        yield PerturbedScene(sample.latent, tape, rng.uniform(-3., 3.), rng.uniform(-1.5, 1.5),
                             rng.uniform(1.25, 1.55), rng.uniform(-.7, .7))


def nominate(controller, truth, state, menu, aliases, incumbent, row):
    generated = sorted(set(aliases.values()) - {incumbent})
    pool = generated if len(generated) >= PROPOSALS else sorted(set(menu) - {incumbent})
    if len(pool) < PROPOSALS:
        raise ValueError("fewer than four distinct feasible non-incumbent motions")
    rng = np.random.default_rng([row["seed"], 18303])
    forecast = posterior_worlds(state, generator.PROBE_SAMPLES, row["seed"] + 18304)
    geometric = {name: generator.direct_score(controller, truth, state, menu[name]) for name in pool}
    evidence = {name: generator.evidence_score(controller, state, menu[name], forecast) for name in pool}

    def top(scores):
        return sorted(pool, key=lambda name: (-round(scores[name], 12), menu[name].distance, name))[:PROPOSALS]

    return {
        "random": rng.permutation(pool)[:PROPOSALS].tolist(),
        "direct": top(geometric),
        "evidence": top({name: estimate["score"] for name, estimate in evidence.items()}),
    }, {"pool": pool, "new_menu_pool": len(generated) >= PROPOSALS, "geometry": geometric, "evidence": evidence}


def collect(row):
    controller, truth, state, prefix, menu, aliases, incumbent, _ = at_handoff(row)
    inputs = policy_inputs(controller, state, prefix, menu, incumbent)
    nominations, score_trace = nominate(controller, truth, state, menu, aliases, incumbent, row)
    selected = sorted({incumbent} | set().union(*nominations.values()))
    bank = list(label_worlds(state, prefix.packets, row["seed"]))
    outcomes = {}
    for name in selected:
        replays, _ = evaluate(controller, state, menu[name], bank, open_blocks=1)
        outcomes[name] = [{field: replay[field] for field in VISIBLE_TARGETS} for replay in replays]
    chosen_features = inputs["features"][[inputs["names"].index(name) for name in selected]]
    return row | {
        "names": selected, "incumbent": incumbent, "features": chosen_features,
        "nominations": nominations, "outcomes": outcomes, "forecast": score_trace,
        "aliases": aliases, "sampled_modes": [scene.latent for scene in bank],
    }


def qualify(row):
    controller, truth, state, prefix, menu, aliases, incumbent, _ = at_handoff(row)
    named = {name: aliases.get(name, name) for name in (*STOCK_REQUESTS, *NEW_REQUESTS)}
    executed = {name: resolved if resolved in menu else incumbent for name, resolved in named.items()}
    executed["mpc"] = incumbent
    bank = scans.paired_futures(truth, row, DEV_TAPES, 18305)
    evaluated = {key: [generator.trial(controller, state, prefix, future, menu[key]) for future in bank]
                 for key in sorted(set(executed.values()))}
    return row | {"outcomes": {name: evaluated[key] for name, key in executed.items()},
                  "available": [name for name, key in named.items() if key in menu]}


def test(row, fixed_id):
    controller, truth, state, prefix, menu, aliases, incumbent, baseline_cpu = at_handoff(row)
    started = process_time()
    inputs = policy_inputs(controller, state, prefix, menu, incumbent)
    feature_cpu = process_time() - started
    fixed = aliases.get(fixed_id, fixed_id)
    choices = {"mpc": incumbent, "best_fixed": fixed if fixed in menu else incumbent}
    cpu = {"mpc": baseline_cpu, "best_fixed": None}
    for arm in ("old_b", *(f"{source}_b" for source in SOURCES)):
        started = process_time()
        predictions = predict_deltas(MODELS[arm], inputs["features"])
        choices[arm], _ = choice_from_predictions(inputs, predictions)
        cpu[arm] = baseline_cpu + feature_cpu + process_time() - started
    bank = scans.paired_futures(truth, row, TEST_TAPES, 18306)
    outcomes = {name: [generator.trial(controller, state, prefix, future, menu[name]) for future in bank]
                for name in sorted(set(choices.values()))}
    return row | {"arms": {arm: {"action": action, "outcomes": outcomes[action], "cpu_seconds": cpu[arm],
                                 "renderer": getattr(menu[action], "renderer", "prodmp")}
                            for arm, action in choices.items()},
                  "fixed_fallback": fixed not in menu, "aliases": aliases,
                  "available": list(menu), "handoff_s": state.s, "handoff_x": state.x,
                  "root_frame": state.frame}
