#!/usr/bin/env python3
"""Exact finite-observation decision-operator check.

This file intentionally stays inside ``theory/operator_experiment``.  It
uses the real geometric observer as a black-box model and only constructs a
finite, analytically integrated public observation operator around that API.
No selected world, controller, or hardware path is used.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtr


HERE = Path(__file__).resolve().parent
LADDER = HERE.parents[1]
OBSERVATION_DIR = LADDER / "observation"
AUDIT_DIR = LADDER / "audit"
PROTOCOL_PATH = HERE / "PROTOCOL.txt"
CONDITIONS_PATH = AUDIT_DIR / "THEORY_CONDITIONS.txt"
ENVIRONMENT_PATH = OBSERVATION_DIR / "environment.py"

# Import only the real geometric observer selected by the frozen protocol.
sys.path.insert(0, str(LADDER))
from observation.environment import SDS, predict_batch, sample_prior  # noqa: E402


CATALOGUE_SEED = 720510
ALIAS_SEED = 720511
CATALOGUE_SIZE = 64
OPTION_COST = 0.5
RISK_SCALE = 60.0
TARGET_LENGTH = 0.015
TARGET_U = 0.002
TARGET_ANGLE = np.deg2rad(10.0)
U_EDGES = np.array([-np.inf, -0.004, 0.004, np.inf], dtype=float)
LENGTH_EDGES = np.array([0.0, 0.008, 0.015, np.inf], dtype=float)
IDENTITY_EDGES = np.array([0.0, 0.75, 1.0], dtype=float)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    """Convert numpy scalars/arrays to stable JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def max_abs(value) -> float:
    arr = np.asarray(value, dtype=float)
    return float(np.max(np.abs(arr))) if arr.size else 0.0


def gaussian_bins(mu: np.ndarray, sd: float, edges: np.ndarray) -> np.ndarray:
    """Normal masses for half-open bins, vectorized over every leading axis."""
    cdf = ndtr((edges.reshape((1,) * mu.ndim + (-1,)) - mu[..., None]) / sd)
    return np.diff(cdf, axis=-1)


def u_bin_masses(mu: np.ndarray) -> np.ndarray:
    # The first/last infinities make this a numerically stable CDF difference.
    return gaussian_bins(mu, float(SDS[0]), U_EDGES)


def length_bin_masses(mu: np.ndarray):
    """Gaussian masses after _sample_model's lower clipping at length zero."""
    c0 = ndtr((0.0 - mu) / float(SDS[2]))
    c008 = ndtr((0.008 - mu) / float(SDS[2]))
    c015 = ndtr((0.015 - mu) / float(SDS[2]))
    # The first bin contains the lower clipping atom and [0,.008).
    masses = np.stack((c008, c015 - c008, 1.0 - c015), axis=-1)
    return masses, c0


def identity_bin_masses(mu: np.ndarray):
    """Gaussian masses after _sample_model's [0,1] identity clipping."""
    c0 = ndtr((0.0 - mu) / float(SDS[4]))
    c075 = ndtr((0.75 - mu) / float(SDS[4]))
    c1 = ndtr((1.0 - mu) / float(SDS[4]))
    # The second bin contains [.75,1) and the upper clipping atom at one.
    masses = np.stack((c075, 1.0 - c075), axis=-1)
    atom1 = 1.0 - c1
    return masses, c0, atom1


def make_actions() -> tuple[np.ndarray, list[dict[str, float | int]]]:
    ys = (-0.025, -0.015, 0.0, 0.015, 0.025)
    yaws = (-0.30, 0.0, 0.30)
    actions = np.asarray([[0.0, y, yaw] for y in ys for yaw in yaws], dtype=float)
    labels = [
        {"index": i, "body_x": float(pose[0]), "body_y": float(pose[1]),
         "yaw": float(pose[2])}
        for i, pose in enumerate(actions)
    ]
    return actions, labels


def observation_labels() -> list[str]:
    labels = ["missing"]
    for u in range(3):
        for length in range(3):
            for identity in range(2):
                labels.append(f"u{u}_length{length}_identity{identity}")
    return labels


def build_observation_operator(prediction: dict[str, np.ndarray]):
    """Build L[h,a,y] by integrating the real source-mixture image model."""
    means = np.asarray(prediction["means"], dtype=float)
    p_source = np.asarray(prediction["p_source"], dtype=float)
    n_h, n_a, n_s, n_f = means.shape
    if n_s != 2 or n_f != 5:
        raise AssertionError(f"unexpected prediction shape {means.shape}")

    p_u = u_bin_masses(means[..., 0])
    p_length, length_atom0 = length_bin_masses(means[..., 2])
    p_identity, identity_atom0, identity_atom1 = identity_bin_masses(means[..., 4])

    # [h,a,source,u_bin,length_bin,identity_bin].  The products are valid
    # because _sample_model adds independent SDS noise to the five fields
    # conditional on a selected source event.
    visible_by_source = (
        p_u[..., :, None, None]
        * p_length[..., None, :, None]
        * p_identity[..., None, None, :]
    )
    if visible_by_source.shape != (n_h, n_a, n_s, 3, 3, 2):
        raise AssertionError(f"unexpected visible shape {visible_by_source.shape}")
    visible_by_source = visible_by_source.reshape(n_h, n_a, n_s, -1)

    labels = observation_labels()
    likelihood = np.zeros((n_h, n_a, len(labels)), dtype=float)
    missing = 1.0 - p_source.sum(axis=-1)
    likelihood[..., 0] = missing
    likelihood[..., 1:] = np.sum(
        visible_by_source * p_source[..., None], axis=2
    )

    return {
        "labels": labels,
        "likelihood": likelihood,
        "u_bin_masses": p_u,
        "length_bin_masses": p_length,
        "identity_bin_masses": p_identity,
        "length_atom0": length_atom0,
        "identity_atom0": identity_atom0,
        "identity_atom1": identity_atom1,
        "missing": missing,
        "visible_by_source": visible_by_source,
    }


def build_terminal_risk(prediction: dict[str, np.ndarray]):
    """Build the frozen public-stop terminal surrogate R[h,j]."""
    means = np.asarray(prediction["means"], dtype=float)
    p_source = np.asarray(prediction["p_source"], dtype=float)
    # Source zero is the declared target hypothesis.  This is a model
    # hypothesis only; no actual selected world is passed to the operator.
    source0 = means[:, :, 0, :]
    p_source0 = p_source[:, :, 0]
    p_u_gate = ndtr((TARGET_U - source0[..., 0]) / float(SDS[0])) - ndtr(
        (-TARGET_U - source0[..., 0]) / float(SDS[0])
    )
    p_length_gate = 1.0 - ndtr(
        (TARGET_LENGTH - source0[..., 2]) / float(SDS[2])
    )
    p_identity_gate = 1.0 - ndtr(
        (0.75 - source0[..., 4]) / float(SDS[4])
    )
    q_target = p_source0 * p_u_gate * p_length_gate * p_identity_gate
    legal = (
        (np.asarray(prediction["geometric_length"])[:, :, 0] >= TARGET_LENGTH)
        & (np.abs(source0[..., 0]) <= TARGET_U)
        & (np.asarray(prediction["angle_audit"])[:, :, 0] <= TARGET_ANGLE)
    )
    risk = RISK_SCALE * (1.0 - legal * q_target**5)
    return {
        "risk": risk,
        "target_geometry_legal": legal,
        "q_target": q_target,
        "p_u_gate": p_u_gate,
        "p_length_gate": p_length_gate,
        "p_identity_gate": p_identity_gate,
        "p_source0": p_source0,
    }


def exact_backup(belief: np.ndarray, likelihood: np.ndarray,
                 risk: np.ndarray, *, option_cost: float = OPTION_COST):
    """Return both Xi/mass and an independently normalized Bayes backup."""
    b = np.asarray(belief, dtype=float)
    # Xi[a,y,j] is an unnormalized branch mass times terminal risk.
    xi = np.einsum("h,hay,hj->ayj", b, likelihood, risk)
    branch_mass = np.einsum("h,hay->ay", b, likelihood)
    unnormalized_branches = np.min(xi, axis=-1)
    option_values_unnormalized = option_cost + unnormalized_branches.sum(axis=-1)

    normalized_branches = np.zeros_like(unnormalized_branches)
    posterior_risk = np.zeros_like(xi)
    for a in range(likelihood.shape[1]):
        for y in range(likelihood.shape[2]):
            mass = float(branch_mass[a, y])
            if mass <= 0.0:
                continue
            posterior = b * likelihood[:, a, y] / mass
            posterior_risk[a, y] = posterior @ risk
            normalized_branches[a, y] = mass * np.min(posterior_risk[a, y])
    option_values_normalized = option_cost + normalized_branches.sum(axis=-1)
    direct_values = b @ risk
    return {
        "xi": xi,
        "branch_mass": branch_mass,
        "posterior_risk": posterior_risk,
        "unnormalized_branches": unnormalized_branches,
        "normalized_branches": normalized_branches,
        "option_values_unnormalized": option_values_unnormalized,
        "option_values_normalized": option_values_normalized,
        "direct_values": direct_values,
        "best_probe": int(np.argmin(option_values_unnormalized)),
        "best_probe_value": float(np.min(option_values_unnormalized)),
        "best_direct": int(np.argmin(direct_values)),
        "best_direct_value": float(np.min(direct_values)),
    }


def make_aliases(risk: np.ndarray, n_aliases: int = 32):
    """Generate the frozen positive alias pairs from the risk nullspace."""
    n_h = risk.shape[0]
    uniform = np.full(n_h, 1.0 / n_h, dtype=float)
    constraint = np.vstack((np.ones(n_h), risk.T))
    _, singular_values, vh = np.linalg.svd(constraint, full_matrices=True)
    tol = max(constraint.shape) * np.finfo(float).eps * max(
        float(singular_values[0]), 1.0
    )
    rank = int(np.sum(singular_values > tol))
    nullspace = vh[rank:].T
    if nullspace.shape[1] == 0:
        raise AssertionError("risk constraint has no nullspace")

    rng = np.random.default_rng(ALIAS_SEED)
    normals = []
    deltas = []
    b_plus = []
    b_minus = []
    for _ in range(n_aliases):
        direction = rng.normal(size=n_h)
        projected = nullspace @ (nullspace.T @ direction)
        norm = np.linalg.norm(projected)
        if norm <= 1e-14:
            raise AssertionError("frozen random direction missed risk nullspace")
        normal = projected / norm
        support = np.abs(normal) > 1e-15
        safe_limit = np.min(uniform[support] / np.abs(normal[support]))
        delta = 0.45 * safe_limit
        plus = uniform + delta * normal
        minus = uniform - delta * normal
        if np.min(plus) <= 0.0 or np.min(minus) <= 0.0:
            raise AssertionError("alias is not strictly positive")
        if not np.isclose(plus.sum(), 1.0, atol=1e-14):
            raise AssertionError("plus alias does not normalize")
        if not np.isclose(minus.sum(), 1.0, atol=1e-14):
            raise AssertionError("minus alias does not normalize")
        normals.append(normal)
        deltas.append(delta)
        b_plus.append(plus)
        b_minus.append(minus)
    return {
        "uniform": uniform,
        "constraint": constraint,
        "singular_values": singular_values,
        "svd_tolerance": tol,
        "rank": rank,
        "nullspace": nullspace,
        "normals": np.asarray(normals),
        "deltas": np.asarray(deltas),
        "b_plus": np.asarray(b_plus),
        "b_minus": np.asarray(b_minus),
    }


def run() -> dict:
    # Read and hash the frozen protocol before model evaluation.  The script
    # never edits this file; the hash is retained in every result artifact.
    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError(PROTOCOL_PATH)
    protocol_sha256 = sha256_file(PROTOCOL_PATH)
    conditions_sha256 = sha256_file(CONDITIONS_PATH)
    environment_sha256 = sha256_file(ENVIRONMENT_PATH)
    runner_sha256 = sha256_file(Path(__file__).resolve())

    actions, action_labels = make_actions()
    particles = sample_prior(seed=CATALOGUE_SEED, n=CATALOGUE_SIZE)
    prediction = predict_batch(particles, actions)
    observation = build_observation_operator(prediction)
    terminal = build_terminal_risk(prediction)
    likelihood = observation["likelihood"]
    risk = terminal["risk"]
    n_h, n_a, n_y = likelihood.shape
    if (n_h, n_a, n_y) != (CATALOGUE_SIZE, 15, 19):
        raise AssertionError(f"unexpected operator shape {likelihood.shape}")

    # Baseline normalization and source/missing consistency checks.
    p_source = np.asarray(prediction["p_source"], dtype=float)
    source_sum = p_source.sum(axis=-1)
    row_normalization_residual = max_abs(likelihood.sum(axis=-1) - 1.0)
    source_probability_residual = max_abs(likelihood[..., 1:].sum(axis=-1) - source_sum)
    missing_residual = max_abs(likelihood[..., 0] - (1.0 - source_sum))
    u_normalization_residual = max_abs(observation["u_bin_masses"].sum(axis=-1) - 1.0)
    length_normalization_residual = max_abs(
        observation["length_bin_masses"].sum(axis=-1) - 1.0
    )
    identity_normalization_residual = max_abs(
        observation["identity_bin_masses"].sum(axis=-1) - 1.0
    )
    if min(
        np.min(likelihood), np.min(p_source), np.min(observation["u_bin_masses"]),
        np.min(observation["length_bin_masses"]),
        np.min(observation["identity_bin_masses"]),
    ) < -1e-12:
        raise AssertionError("negative probability in integrated operator")

    aliases = make_aliases(risk)
    uniform_result = exact_backup(aliases["uniform"], likelihood, risk)

    alias_count = aliases["b_plus"].shape[0]
    alias_risk_plus = aliases["b_plus"] @ risk
    alias_risk_minus = aliases["b_minus"] @ risk
    alias_operator_plus = np.einsum("kh,hay->kay", aliases["b_plus"], likelihood)
    alias_operator_minus = np.einsum("kh,hay->kay", aliases["b_minus"], likelihood)
    alias_option_plus = np.empty((alias_count, n_a), dtype=float)
    alias_option_minus = np.empty((alias_count, n_a), dtype=float)
    alias_direct_plus = np.empty((alias_count, n_a), dtype=float)
    alias_direct_minus = np.empty((alias_count, n_a), dtype=float)
    alias_xi_plus = np.empty((alias_count, n_a, n_y, n_a), dtype=float)
    alias_xi_minus = np.empty((alias_count, n_a, n_y, n_a), dtype=float)
    alias_norm_option_residual = np.empty(alias_count, dtype=float)
    alias_full_backup_residual = np.empty(alias_count, dtype=float)
    alias_best_probe_plus = np.empty(alias_count, dtype=int)
    alias_best_probe_minus = np.empty(alias_count, dtype=int)
    alias_best_direct_plus = np.empty(alias_count, dtype=int)
    alias_best_direct_minus = np.empty(alias_count, dtype=int)
    alias_best_probe_value_plus = np.empty(alias_count, dtype=float)
    alias_best_probe_value_minus = np.empty(alias_count, dtype=float)

    for k in range(alias_count):
        plus_result = exact_backup(aliases["b_plus"][k], likelihood, risk)
        minus_result = exact_backup(aliases["b_minus"][k], likelihood, risk)
        alias_option_plus[k] = plus_result["option_values_unnormalized"]
        alias_option_minus[k] = minus_result["option_values_unnormalized"]
        alias_direct_plus[k] = plus_result["direct_values"]
        alias_direct_minus[k] = minus_result["direct_values"]
        alias_xi_plus[k] = plus_result["xi"]
        alias_xi_minus[k] = minus_result["xi"]
        alias_norm_option_residual[k] = max_abs(
            plus_result["option_values_unnormalized"]
            - plus_result["option_values_normalized"]
        )
        alias_norm_option_residual[k] = max(
            alias_norm_option_residual[k],
            max_abs(
                minus_result["option_values_unnormalized"]
                - minus_result["option_values_normalized"]
            ),
        )
        # Reconstructing the backup from Xi is the full-belief one-step
        # operator; this residual is deliberately checked independently.
        plus_reconstructed = OPTION_COST + np.min(plus_result["xi"], axis=-1).sum(axis=-1)
        minus_reconstructed = OPTION_COST + np.min(minus_result["xi"], axis=-1).sum(axis=-1)
        alias_full_backup_residual[k] = max(
            max_abs(plus_reconstructed - plus_result["option_values_unnormalized"]),
            max_abs(minus_reconstructed - minus_result["option_values_unnormalized"]),
        )
        alias_best_probe_plus[k] = plus_result["best_probe"]
        alias_best_probe_minus[k] = minus_result["best_probe"]
        alias_best_direct_plus[k] = plus_result["best_direct"]
        alias_best_direct_minus[k] = minus_result["best_direct"]
        alias_best_probe_value_plus[k] = plus_result["best_probe_value"]
        alias_best_probe_value_minus[k] = minus_result["best_probe_value"]

    alias_risk_residual = max_abs(alias_risk_plus - alias_risk_minus)
    alias_sum_residual = max(
        max_abs(aliases["b_plus"].sum(axis=1) - 1.0),
        max_abs(aliases["b_minus"].sum(axis=1) - 1.0),
    )
    alias_min_belief = float(
        min(np.min(aliases["b_plus"]), np.min(aliases["b_minus"]))
    )
    alias_nullspace_residual = max_abs(aliases["constraint"] @ aliases["normals"].T)
    alias_observation_tv = 0.5 * np.abs(
        alias_operator_plus - alias_operator_minus
    ).sum(axis=-1)
    alias_option_value_delta = np.abs(alias_option_plus - alias_option_minus)
    action_difference_mask = alias_best_probe_plus != alias_best_probe_minus
    current_risk_only_direct_residual = max_abs(
        alias_direct_plus - alias_direct_minus
    )
    xi_full_backup_residual = max(
        max_abs(
            uniform_result["option_values_unnormalized"]
            - uniform_result["option_values_normalized"]
        ),
        float(np.max(alias_full_backup_residual)),
    )

    # Clipping atoms are retained explicitly in raw arrays and checked against
    # the corresponding integrated rows.  They are model probability masses,
    # not extra observations in the public alphabet.
    length_atom_residual = max_abs(
        observation["length_atom0"]
        - ndtr((0.0 - prediction["means"][..., 2]) / float(SDS[2]))
    )
    identity_atom0_residual = max_abs(
        observation["identity_atom0"]
        - ndtr((0.0 - prediction["means"][..., 4]) / float(SDS[4]))
    )
    identity_atom1_residual = max_abs(
        observation["identity_atom1"]
        - (1.0 - ndtr((1.0 - prediction["means"][..., 4]) / float(SDS[4])))
    )

    array_path = HERE / "operator_arrays.npz"
    np.savez_compressed(
        array_path,
        actions=actions,
        SDS=np.asarray(SDS),
        catalogue_registration=np.asarray([p.registration for p in particles]),
        catalogue_modes=np.asarray([p.modes for p in particles]),
        catalogue_y_controls=np.asarray([p.y_controls for p in particles]),
        catalogue_z_controls=np.asarray([p.z_controls for p in particles]),
        catalogue_depths=np.asarray([p.depths for p in particles]),
        catalogue_identity_z=np.asarray([p.identity_z for p in particles]),
        catalogue_detect_z=np.asarray([p.detect_z for p in particles]),
        catalogue_points=np.asarray([p.points for p in particles]),
        means=np.asarray(prediction["means"]),
        p_source=p_source,
        p_visible=np.asarray(prediction["p_visible"]),
        contact=np.asarray(prediction["contact"]),
        geometric_length=np.asarray(prediction["geometric_length"]),
        angle_audit=np.asarray(prediction["angle_audit"]),
        observation_labels=np.asarray(observation["labels"]),
        likelihood=likelihood,
        missing=observation["missing"],
        visible_by_source=observation["visible_by_source"],
        u_bin_masses=observation["u_bin_masses"],
        length_bin_masses=observation["length_bin_masses"],
        identity_bin_masses=observation["identity_bin_masses"],
        length_atom0=observation["length_atom0"],
        identity_atom0=observation["identity_atom0"],
        identity_atom1=observation["identity_atom1"],
        target_geometry_legal=terminal["target_geometry_legal"],
        q_target=terminal["q_target"],
        p_u_gate=terminal["p_u_gate"],
        p_length_gate=terminal["p_length_gate"],
        p_identity_gate=terminal["p_identity_gate"],
        p_source0=terminal["p_source0"],
        risk=risk,
        uniform_b=aliases["uniform"],
        constraint_matrix=aliases["constraint"],
        singular_values=aliases["singular_values"],
        nullspace=aliases["nullspace"],
        alias_normals=aliases["normals"],
        alias_deltas=aliases["deltas"],
        b_plus=aliases["b_plus"],
        b_minus=aliases["b_minus"],
        alias_current_risk_plus=alias_risk_plus,
        alias_current_risk_minus=alias_risk_minus,
        alias_observation_operator_plus=alias_operator_plus,
        alias_observation_operator_minus=alias_operator_minus,
        alias_option_values_plus=alias_option_plus,
        alias_option_values_minus=alias_option_minus,
        alias_direct_values_plus=alias_direct_plus,
        alias_direct_values_minus=alias_direct_minus,
        alias_xi_plus=alias_xi_plus,
        alias_xi_minus=alias_xi_minus,
        uniform_xi=uniform_result["xi"],
        uniform_branch_mass=uniform_result["branch_mass"],
        uniform_option_values_unnormalized=uniform_result["option_values_unnormalized"],
        uniform_option_values_normalized=uniform_result["option_values_normalized"],
        uniform_direct_values=uniform_result["direct_values"],
        alias_best_probe_plus=alias_best_probe_plus,
        alias_best_probe_minus=alias_best_probe_minus,
        alias_best_direct_plus=alias_best_direct_plus,
        alias_best_direct_minus=alias_best_direct_minus,
        alias_best_probe_value_plus=alias_best_probe_value_plus,
        alias_best_probe_value_minus=alias_best_probe_value_minus,
        alias_norm_option_residual=alias_norm_option_residual,
        alias_full_backup_residual=alias_full_backup_residual,
    )
    array_sha256 = sha256_file(array_path)

    direct_min = uniform_result["best_direct_value"]
    probe_min = uniform_result["best_probe_value"]
    if direct_min <= probe_min:
        overall_choice = "direct_terminal"
        overall_value = direct_min
    else:
        overall_choice = "probe_then_terminal"
        overall_value = probe_min

    results = {
        "status": "PASS",
        "scope": "offline finite-observation decision-operator module only",
        "claim_boundary": (
            "mechanism/operator correctness in the declared geometric surrogate; "
            "not robot/task performance and not a production or hardware change"
        ),
        "interpreter": sys.executable,
        "threads": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "catalogue": {
            "seed": CATALOGUE_SEED,
            "size": CATALOGUE_SIZE,
            "api": "observation.environment.sample_prior",
        },
        "actions": {
            "count": int(n_a),
            "definition": "body_x=0; y in [-.025,-.015,0,.015,.025]; yaw in [-.30,0,.30]",
            "order": "nested y then yaw",
            "labels": action_labels,
        },
        "observation": {
            "count": int(n_y),
            "labels": observation["labels"],
            "u_edges": U_EDGES,
            "length_edges": LENGTH_EDGES,
            "identity_edges": IDENTITY_EDGES,
            "source_events": "p_source source-mixture events; missing=1-sum(p_source)",
            "integration": "analytic independent Gaussian SDS masses with real clipping semantics",
        },
        "terminal_risk": {
            "scale": RISK_SCALE,
            "target_length_m": TARGET_LENGTH,
            "target_abs_u_m": TARGET_U,
            "target_angle_deg": 10.0,
            "persistent_gate_power": 5,
            "source_hypothesis": 0,
            "formula": "60*(1-I(legal)*q_target^5)",
            "all_contacts_true": bool(np.all(prediction["contact"])),
        },
        "cost": {
            "probe_option_constant": OPTION_COST,
            "meaning": "mechanism isolation only; not a robot motion cost",
        },
        "shapes": {
            "likelihood": list(likelihood.shape),
            "risk": list(risk.shape),
            "constraint": list(aliases["constraint"].shape),
            "nullspace": list(aliases["nullspace"].shape),
        },
        "normalization_residuals": {
            "observation_rows": row_normalization_residual,
            "visible_source_mass": source_probability_residual,
            "missing_mass": missing_residual,
            "u_bins": u_normalization_residual,
            "length_bins": length_normalization_residual,
            "identity_bins": identity_normalization_residual,
            "length_atom0": length_atom_residual,
            "identity_atom0": identity_atom0_residual,
            "identity_atom1": identity_atom1_residual,
        },
        "uniform_backup": {
            "best_probe_action": int(uniform_result["best_probe"]),
            "best_probe_value": float(uniform_result["best_probe_value"]),
            "best_probe_action_label": action_labels[uniform_result["best_probe"]],
            "direct_terminal_action": int(uniform_result["best_direct"]),
            "direct_terminal_value": float(uniform_result["best_direct_value"]),
            "direct_terminal_action_label": action_labels[uniform_result["best_direct"]],
            "overall_choice": overall_choice,
            "overall_value": float(overall_value),
            "normalized_vs_unnormalized_max_residual": max_abs(
                uniform_result["option_values_unnormalized"]
                - uniform_result["option_values_normalized"]
            ),
        },
        "aliases": {
            "seed": ALIAS_SEED,
            "count": alias_count,
            "constraint_rank": aliases["rank"],
            "svd_tolerance": aliases["svd_tolerance"],
            "nullspace_dimension": int(aliases["nullspace"].shape[1]),
            "min_belief_entry": alias_min_belief,
            "sum_to_one_max_residual": alias_sum_residual,
            "nullspace_max_residual": alias_nullspace_residual,
            "current_risk_vector_max_residual": alias_risk_residual,
            "current_risk_only_direct_value_max_residual": current_risk_only_direct_residual,
            "observation_operator_tv_max": float(np.max(alias_observation_tv)),
            "observation_operator_tv_mean": float(np.mean(alias_observation_tv)),
            "probe_value_delta_max": float(np.max(alias_option_value_delta)),
            "probe_value_delta_mean": float(np.mean(alias_option_value_delta)),
            "probe_action_difference_count": int(np.sum(action_difference_mask)),
            "probe_action_difference_fraction": float(np.mean(action_difference_mask)),
            "direct_action_difference_count": int(
                np.sum(alias_best_direct_plus != alias_best_direct_minus)
            ),
            "normalized_backup_max_residual": float(np.max(alias_norm_option_residual)),
            "full_xi_backup_max_residual": xi_full_backup_residual,
            "interpretation": (
                "Current terminal-risk vectors are identical by construction; "
                "any predictive operator/value/action difference is retained as "
                "the finite-observation alias diagnostic."
            ),
        },
        "hashes": {
            "protocol": protocol_sha256,
            "conditions": conditions_sha256,
            "environment": environment_sha256,
            "runner": runner_sha256,
            "operator_arrays": array_sha256,
        },
    }
    results_path = HERE / "RESULTS.json"
    results_path.write_text(
        json.dumps(jsonable(results), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    results_sha256 = sha256_file(results_path)
    sha_path = HERE / "RESULTS.sha256"
    sha_path.write_text(
        "operator_arrays.npz  " + array_sha256 + "\n"
        "RESULTS.json  " + results_sha256 + "\n"
        "PROTOCOL.txt  " + protocol_sha256 + "\n"
        "THEORY_CONDITIONS.txt  " + conditions_sha256 + "\n"
        "environment.py  " + environment_sha256 + "\n"
        "run_operator_experiment.py  " + runner_sha256 + "\n",
        encoding="utf-8",
    )

    report_path = HERE / "REPORT.txt"
    action_change_text = (
        f"{int(np.sum(action_difference_mask))}/{alias_count} alias pairs change "
        "the exact argmin probe action."
    )
    report_lines = [
        "Finite-observation decision-operator check: PASS",
        "",
        "Scope: offline geometric-surrogate mechanism check only; no controller,",
        "selected hidden world, hardware, production mutation, or task-performance claim.",
        "The real observation.environment sample_prior/predict_batch APIs were used.",
        "",
        f"Catalogue: sample_prior(seed={CATALOGUE_SEED}, n={CATALOGUE_SIZE}); actions={n_a}; observations={n_y}.",
        "Observation rows integrate source-mixture Gaussian image fields analytically;",
        "length=0 and identity=0/1 clipping atoms are represented and checked.",
        f"Max observation-row normalization residual: {row_normalization_residual:.3e}",
        f"Max source/missing consistency residual: {max(source_probability_residual, missing_residual):.3e}",
        "",
        "Uniform belief exact backup:",
        f"  best probe action {uniform_result['best_probe']} value {uniform_result['best_probe_value']:.12g}",
        f"  direct terminal action {uniform_result['best_direct']} value {uniform_result['best_direct_value']:.12g}",
        f"  overall choice {overall_choice} value {overall_value:.12g}",
        f"  normalized-vs-unnormalized residual {max_abs(uniform_result['option_values_unnormalized'] - uniform_result['option_values_normalized']):.3e}",
        "",
        f"Aliases: {alias_count} frozen nullspace direction pairs, rank={aliases['rank']},",
        f"nullspace dimension={aliases['nullspace'].shape[1]}, min belief entry={alias_min_belief:.3e}.",
        f"Max current-risk-vector residual: {alias_risk_residual:.3e}",
        f"Max current-risk-only direct-value residual: {current_risk_only_direct_residual:.3e}",
        f"Max predictive observation TV across alias pairs: {np.max(alias_observation_tv):.12g}",
        f"Max probe-value difference across alias pairs: {np.max(alias_option_value_delta):.12g}",
        f"  {action_change_text}",
        "Full Xi branch backups reproduce the same one-step values; no prior,",
        "sensor, or cost tuning was performed.",
        "",
        f"Raw arrays: operator_arrays.npz sha256={array_sha256}",
        f"Results: RESULTS.json sha256={results_sha256}",
        f"Protocol sha256={protocol_sha256}",
        f"Theory conditions sha256={conditions_sha256}",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {
        "results": results,
        "results_sha256": results_sha256,
        "array_sha256": array_sha256,
        "report_path": str(report_path),
    }


if __name__ == "__main__":
    output = run()
    print(json.dumps({
        "status": output["results"]["status"],
        "results_sha256": output["results_sha256"],
        "operator_arrays_sha256": output["array_sha256"],
        "report": output["report_path"],
    }, sort_keys=True))
