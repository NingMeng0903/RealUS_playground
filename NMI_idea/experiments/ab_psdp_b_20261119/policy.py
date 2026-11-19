"""Continuation-specific action values for a two-decision probe program."""

import json
from pathlib import Path
import sys

import joblib
import numpy as np


HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent / "ab_probe_program_20261119"
LEGACY_RUN = LEGACY / "results/main"
sys.path.insert(0, str(LEGACY))
import probe as p
import evaluate as legacy_eval
from learned_recovery import predict_deltas


MODELS = {}
DIRECT = False
OBSERVE = True
STOP = (p.HANDOFF, DIRECT)
LEGACY_FILES = ("single_models.joblib", "child_models.joblib", "program_models.joblib", "fixed.json")


def load_references():
    frozen = json.loads((LEGACY_RUN / "frozen.json").read_text())
    for name in LEGACY_FILES:
        if p.base.fingerprint(LEGACY_RUN / name) != frozen["files"][name]:
            raise ValueError(f"published reference changed: {name}")
    MODELS.update(joblib.load(LEGACY_RUN / "single_models.joblib"))
    MODELS["legacy_child"] = joblib.load(LEGACY_RUN / "child_models.joblib")["child"]
    MODELS["legacy_program"] = joblib.load(LEGACY_RUN / "program_models.joblib")["program"]
    archive = p.PREVIOUS / "results/teacher_b_training_v1/models.joblib"
    expected = "519e08f76b583c6b0c423ecb7dbac27387df572f447c9a6388bc2e5a26758b53"
    if p.base.fingerprint(archive) != expected:
        raise ValueError("previously released evidence teacher model changed")
    MODELS["published_evidence"] = joblib.load(archive)["evidence"]
    return {LEGACY_RUN / name: frozen["files"][name] for name in LEGACY_FILES} | {archive: expected}


def select_direct(info, scores):
    name, _ = p.base.choice_from_predictions(info, scores.copy())
    return STOP if name == info["incumbent"] else (name, DIRECT)


def select_modes(info, scores_direct, scores_adaptive):
    # The incumbent's direct mode equals immediate MPC by the executor contract.
    # Its adaptive mode may keep control after executing the same first motion.
    names = info["names"]
    incumbent = (info["incumbent"], DIRECT)
    conditional = [(name, OBSERVE) for name in names]
    if scores_direct is None:
        modes = [incumbent, *conditional]
        scores = np.vstack((np.zeros(scores_adaptive.shape[1]), scores_adaptive))
    else:
        modes = [(name, DIRECT) for name in names] + conditional
        scores = np.vstack((scores_direct, scores_adaptive))
    scores[:, :3] = np.clip(scores[:, :3], -1., 1.)
    picked = p.base.choice_from_predictions({"names": modes, "incumbent": incumbent}, scores)[0]
    return STOP if picked == incumbent else picked


def scores(info, model):
    return predict_deltas(MODELS[model], info["features"])


def full_choice(info, *, residual="residual"):
    direct = scores(info, "contrast")
    direct[info["names"].index(info["incumbent"])] = 0.
    return select_modes(info, direct, direct + scores(info, residual))


def gated_reference(info, name):
    choice = select_direct(info, scores(info, name))
    return STOP if choice == STOP else (choice[0], OBSERVE)
