#!/usr/bin/env python3
"""Low-rank audit of the frozen observation-conditioned Xi operator."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
SOURCE_ARRAYS = PARENT / "operator_arrays.npz"
SOURCE_RESULTS = PARENT / "RESULTS.json"
PROTOCOL = HERE / "PROTOCOL.txt"
RANKS = np.asarray([1, 4, 8, 16, 32, 64], dtype=int)
DIRICHLET_SEED = 720512
DIRICHLET_COUNT = 32
DIRICHLET_ALPHA = 1.0
OPTION_COST = 0.5
ACTION_TIE_TOLERANCE = 1e-10  # diagnostic only; raw argmins remain saved
EXPECTED_ARRAY_SHA256 = "69dee7a49016a0b66ff020a4bd6b0f858aecfe7c48ced65867102525661859d4"
EXPECTED_RESULTS_SHA256 = "cb02cf84f6fd93bf684e1039e3f715699eb66e5b3e8418d5904dca066fc3e6bf"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
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


def max_abs(array) -> float:
    value = np.asarray(array, dtype=float)
    return float(np.max(np.abs(value))) if value.size else 0.0


def timed(callable_, repetitions: int = 5):
    # Warm once, then report descriptive wall time only; portable cost is
    # recorded separately from these machine-dependent timings.
    callable_()
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        callable_()
        samples.append(time.perf_counter() - start)
    return {
        "repetitions": repetitions,
        "samples_s": samples,
        "median_s": float(np.median(samples)),
        "mean_s": float(np.mean(samples)),
        "min_s": float(np.min(samples)),
    }


def run() -> dict:
    source_arrays_sha256 = sha256_file(SOURCE_ARRAYS)
    source_results_sha256 = sha256_file(SOURCE_RESULTS)
    protocol_sha256 = sha256_file(PROTOCOL)
    if source_arrays_sha256 != EXPECTED_ARRAY_SHA256:
        raise RuntimeError("sealed operator_arrays.npz hash changed")
    if source_results_sha256 != EXPECTED_RESULTS_SHA256:
        raise RuntimeError("sealed parent RESULTS.json hash changed")

    with np.load(SOURCE_ARRAYS, allow_pickle=False) as source:
        likelihood = np.asarray(source["likelihood"], dtype=float)
        risk = np.asarray(source["risk"], dtype=float)
        actions = np.asarray(source["actions"], dtype=float)
        alias_plus = np.asarray(source["b_plus"], dtype=float)
        alias_minus = np.asarray(source["b_minus"], dtype=float)
        saved_uniform_xi = np.asarray(source["uniform_xi"], dtype=float)
        saved_alias_xi_plus = np.asarray(source["alias_xi_plus"], dtype=float)
        saved_alias_xi_minus = np.asarray(source["alias_xi_minus"], dtype=float)
        saved_uniform_option = np.asarray(
            source["uniform_option_values_unnormalized"], dtype=float
        )
        saved_alias_option_plus = np.asarray(
            source["alias_option_values_plus"], dtype=float
        )
        saved_alias_option_minus = np.asarray(
            source["alias_option_values_minus"], dtype=float
        )

    n_h, n_a, n_y = likelihood.shape
    if risk.shape != (n_h, n_a):
        raise AssertionError(f"risk shape {risk.shape} does not match {likelihood.shape}")
    if actions.shape != (n_a, 3) or n_h != 64 or n_a != 15 or n_y != 19:
        raise AssertionError("unexpected sealed operator dimensions")

    # The row order is frozen by PROTOCOL.txt: ((a*Y)+y)*A+j.
    operator_tensor = np.einsum("hay,hj->ayjh", likelihood, risk)
    matrix = operator_tensor.reshape(n_a * n_y * n_a, n_h)
    rows = matrix.shape[0]

    # Check the exact compiled matvec against the parent module's saved Xi and
    # backup arrays before making any low-rank approximation.
    uniform = np.full(n_h, 1.0 / n_h, dtype=float)
    alias_beliefs = np.vstack((alias_plus, alias_minus))
    alias_labels = [f"alias_{k:02d}_plus" for k in range(len(alias_plus))]
    alias_labels += [f"alias_{k:02d}_minus" for k in range(len(alias_minus))]
    rng = np.random.default_rng(DIRICHLET_SEED)
    dirichlet = rng.dirichlet(
        np.full(n_h, DIRICHLET_ALPHA, dtype=float), size=DIRICHLET_COUNT
    )
    dirichlet_labels = [f"dirichlet_{k:02d}" for k in range(DIRICHLET_COUNT)]
    test_beliefs = np.vstack((alias_beliefs, dirichlet))
    test_labels = alias_labels + dirichlet_labels
    test_count = len(test_beliefs)

    uniform_xi_from_matrix = (matrix @ uniform).reshape(n_a, n_y, n_a)
    alias_xi_from_matrix = (alias_beliefs @ matrix.T).reshape(
        len(alias_beliefs), n_a, n_y, n_a
    )
    exact_xi = test_beliefs @ matrix.T
    exact_xi_tensor = exact_xi.reshape(test_count, n_a, n_y, n_a)
    if max_abs(uniform_xi_from_matrix - saved_uniform_xi) > 1e-10:
        raise AssertionError("compiled uniform Xi disagrees with sealed Xi")
    if max_abs(alias_xi_from_matrix[: len(alias_plus)] - saved_alias_xi_plus) > 1e-10:
        raise AssertionError("compiled plus Xi disagrees with sealed Xi")
    if max_abs(alias_xi_from_matrix[len(alias_plus):] - saved_alias_xi_minus) > 1e-10:
        raise AssertionError("compiled minus Xi disagrees with sealed Xi")

    exact_future = np.min(exact_xi_tensor, axis=-1).sum(axis=-1)
    exact_probe_values = OPTION_COST + exact_future
    exact_probe_actions = np.argmin(exact_future, axis=-1)
    exact_probe_sorted = np.sort(exact_future, axis=-1)
    exact_probe_gap = exact_probe_sorted[:, 1] - exact_probe_sorted[:, 0]
    exact_terminal_values = test_beliefs @ risk
    exact_direct_actions = np.argmin(exact_terminal_values, axis=-1)
    exact_direct_min = np.min(exact_terminal_values, axis=-1)
    exact_global_kind = np.where(
        exact_direct_min <= np.min(exact_probe_values, axis=-1),
        "direct_terminal",
        "probe_then_terminal",
    )
    exact_global_action = np.where(
        exact_global_kind == "direct_terminal",
        exact_direct_actions,
        exact_probe_actions,
    )
    exact_global_direct_count = int(np.sum(exact_global_kind == "direct_terminal"))
    exact_global_probe_count = int(np.sum(exact_global_kind == "probe_then_terminal"))
    exact_global_probe_indices = np.flatnonzero(
        exact_global_kind == "probe_then_terminal"
    )
    # Parent's frozen backups include the .5 option cost.  This is a separate
    # exact check for the 64 alias beliefs and the uniform reference.
    exact_alias_probe = exact_probe_values[: len(alias_beliefs)]
    if max_abs(exact_alias_probe[: len(alias_plus)] - saved_alias_option_plus) > 1e-10:
        raise AssertionError("compiled plus backup disagrees with sealed backup")
    if max_abs(exact_alias_probe[len(alias_plus):] - saved_alias_option_minus) > 1e-10:
        raise AssertionError("compiled minus backup disagrees with sealed backup")
    if max_abs(exact_probe_values[-1:] - exact_probe_values[-1:]) > 0:
        raise AssertionError("unreachable exact backup check")
    uniform_exact_future = np.min(uniform_xi_from_matrix, axis=-1).sum(axis=-1)
    uniform_exact_option = OPTION_COST + uniform_exact_future
    if max_abs(uniform_exact_option - saved_uniform_option) > 1e-10:
        raise AssertionError("compiled uniform backup disagrees with sealed backup")

    # One economy SVD is shared by every requested rank.
    left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    svd_tolerance = max(matrix.shape) * np.finfo(float).eps * max(
        float(singular_values[0]), 1.0
    )
    numerical_rank = int(np.sum(singular_values > svd_tolerance))

    # Arrays saved for independently checking every belief/rank result.
    n_ranks = len(RANKS)
    compressed_xi = np.empty((n_ranks, test_count, rows), dtype=float)
    compressed_future = np.empty((n_ranks, test_count, n_a), dtype=float)
    compressed_probe_values = np.empty_like(compressed_future)
    compressed_probe_actions = np.empty((n_ranks, test_count), dtype=int)
    compressed_terminal_actions = np.empty(
        (n_ranks, test_count, n_a, n_y), dtype=int
    )
    compressed_global_kind = np.empty((n_ranks, test_count), dtype="U20")
    compressed_global_action = np.empty((n_ranks, test_count), dtype=int)
    tensor_entry_sup_error = np.empty(n_ranks, dtype=float)
    tensor_entry_rmse = np.empty(n_ranks, dtype=float)
    tensor_entry_mean_abs_error = np.empty(n_ranks, dtype=float)
    entry_bound_by_action = np.empty((n_ranks, n_a), dtype=float)
    propagation_bound = np.empty(n_ranks, dtype=float)
    propagation_bound_by_action = np.empty((n_ranks, n_a), dtype=float)
    future_error_sup = np.empty(n_ranks, dtype=float)
    future_error_mean_abs = np.empty(n_ranks, dtype=float)
    future_error_rmse = np.empty(n_ranks, dtype=float)
    probe_value_error_sup = np.empty(n_ranks, dtype=float)
    probe_action_difference_count = np.empty(n_ranks, dtype=int)
    probe_action_non_tie_difference_count = np.empty(n_ranks, dtype=int)
    probe_action_tie_ambiguous_difference_count = np.empty(n_ranks, dtype=int)
    exact_probe_tie_count = int(np.sum(exact_probe_gap <= ACTION_TIE_TOLERANCE))
    compressed_probe_gap = np.empty((n_ranks, test_count), dtype=float)
    terminal_action_difference_count = np.empty(n_ranks, dtype=int)
    global_policy_difference_count = np.empty(n_ranks, dtype=int)
    global_policy_kind_difference_count = np.empty(n_ranks, dtype=int)
    negative_matrix_count = np.empty(n_ranks, dtype=int)
    negative_matrix_min = np.empty(n_ranks, dtype=float)
    negative_xi_count = np.empty(n_ranks, dtype=int)
    negative_xi_min = np.empty(n_ranks, dtype=float)
    factor_bytes = np.empty(n_ranks, dtype=int)
    full_operator_matmuls_per_belief = rows * n_h
    lowrank_operator_matmuls_per_belief = np.empty(n_ranks, dtype=int)
    factor_timings = []
    factor_matrices = []
    factor_rights = []
    entry_row_supremum = np.empty((n_ranks, rows), dtype=float)

    full_timing = timed(lambda: test_beliefs @ matrix.T)
    for index, rank in enumerate(RANKS):
        # Absorb singular values into the left factor.  This is the exact
        # un-clipped algebraic truncated reconstruction requested by protocol.
        left_factor = left[:, :rank] * singular_values[:rank][None, :]
        right_factor = right[:rank, :]
        reconstruction = left_factor @ right_factor
        error = reconstruction - matrix
        error_abs = np.abs(error)
        row_sup = error_abs.max(axis=1).reshape(n_a, n_y, n_a)
        entry_row_supremum[index] = row_sup.reshape(-1)
        tensor_entry_sup_error[index] = float(error_abs.max())
        tensor_entry_rmse[index] = float(np.sqrt(np.mean(error**2)))
        tensor_entry_mean_abs_error[index] = float(np.mean(error_abs))
        entry_bound_by_action[index] = row_sup.max(axis=(1, 2))
        propagation_bound_by_action[index] = row_sup.max(axis=2).sum(axis=1)
        propagation_bound[index] = float(propagation_bound_by_action[index].max())

        # Factorized matvec, rather than a dense reconstruction, is the
        # measured compressed-inference path.  All signed entries are kept.
        def factor_matvec():
            return (test_beliefs @ right_factor.T) @ left_factor.T

        approx_xi = factor_matvec()
        factor_timings.append(timed(factor_matvec))
        factor_matrices.append(left_factor)
        factor_rights.append(right_factor)
        compressed_xi[index] = approx_xi
        approx_tensor = approx_xi.reshape(test_count, n_a, n_y, n_a)
        approx_future = np.min(approx_tensor, axis=-1).sum(axis=-1)
        approx_probe = OPTION_COST + approx_future
        approx_probe_action = np.argmin(approx_future, axis=-1)
        approx_probe_sorted = np.sort(approx_future, axis=-1)
        compressed_probe_gap[index] = (
            approx_probe_sorted[:, 1] - approx_probe_sorted[:, 0]
        )
        approx_terminal_action = np.argmin(approx_tensor, axis=-1)
        approx_direct_min = exact_direct_min  # direct risk is exact by design
        approx_kind = np.where(
            approx_direct_min <= np.min(approx_probe, axis=-1),
            "direct_terminal",
            "probe_then_terminal",
        )
        approx_global_action = np.where(
            approx_kind == "direct_terminal",
            exact_direct_actions,
            approx_probe_action,
        )
        compressed_future[index] = approx_future
        compressed_probe_values[index] = approx_probe
        compressed_probe_actions[index] = approx_probe_action
        compressed_terminal_actions[index] = approx_terminal_action
        compressed_global_kind[index] = approx_kind
        compressed_global_action[index] = approx_global_action

        delta_future = approx_future - exact_future
        future_error_sup[index] = float(np.max(np.abs(delta_future)))
        future_error_mean_abs[index] = float(np.mean(np.abs(delta_future)))
        future_error_rmse[index] = float(np.sqrt(np.mean(delta_future**2)))
        probe_value_error_sup[index] = float(
            np.max(np.abs(approx_probe - exact_probe_values))
        )
        probe_action_difference_count[index] = int(
            np.sum(approx_probe_action != exact_probe_actions)
        )
        action_changed = approx_probe_action != exact_probe_actions
        probe_action_tie_ambiguous_difference_count[index] = int(
            np.sum(action_changed & (exact_probe_gap <= ACTION_TIE_TOLERANCE))
        )
        probe_action_non_tie_difference_count[index] = int(
            np.sum(action_changed & (exact_probe_gap > ACTION_TIE_TOLERANCE))
        )
        terminal_action_difference_count[index] = int(
            np.sum(approx_terminal_action != np.argmin(exact_xi_tensor, axis=-1))
        )
        global_policy_kind_difference_count[index] = int(
            np.sum(approx_kind != exact_global_kind)
        )
        global_policy_difference_count[index] = int(
            np.sum(
                (approx_kind != exact_global_kind)
                | (
                    (approx_kind == exact_global_kind)
                    & (approx_global_action != exact_global_action)
                )
            )
        )

        matrix_negative = reconstruction < 0.0
        xi_negative = approx_xi < 0.0
        negative_matrix_count[index] = int(np.sum(matrix_negative))
        negative_matrix_min[index] = float(np.min(reconstruction))
        negative_xi_count[index] = int(np.sum(xi_negative))
        negative_xi_min[index] = float(np.min(approx_xi))
        factor_bytes[index] = int(
            (left_factor.size + right_factor.size) * np.dtype(float).itemsize
        )
        lowrank_operator_matmuls_per_belief[index] = int(
            rank * (n_h + rows)
        )

        # This is the promised uniform-simplex propagation check.  The
        # realized held-out error must not exceed the conservative bound.
        if future_error_sup[index] > propagation_bound[index] + 1e-10:
            raise AssertionError("future error exceeded simplex propagation bound")

    # Exact direct risk/action is retained in every compressed path.  Confirm
    # that this field is never altered by the low-rank operator.
    exact_direct_again = test_beliefs @ risk
    if max_abs(exact_direct_again - exact_terminal_values) > 1e-12:
        raise AssertionError("direct risk reconstruction is not exact")

    full_matrix_bytes = int(matrix.nbytes)
    direct_risk_bytes = int(risk.nbytes)
    full_operator_matmuls_batch = int(test_count * full_operator_matmuls_per_belief)
    lowrank_operator_matmuls_batch = (
        test_count * lowrank_operator_matmuls_per_belief
    )
    direct_risk_matmuls_per_belief = int(n_h * n_a)
    direct_risk_matmuls_batch = int(test_count * direct_risk_matmuls_per_belief)

    # The report deliberately separates conditional probe results from the
    # complete direct-vs-probe policy.  The parent audit's frozen cost makes
    # direct termination globally optimal, but compression may still perturb
    # conditional probe rankings.
    rank_reports = []
    for index, rank in enumerate(RANKS):
        rank_reports.append({
            "rank": int(rank),
            "tensor_entry_sup_error": tensor_entry_sup_error[index],
            "tensor_entry_rmse_descriptive": tensor_entry_rmse[index],
            "tensor_entry_mean_abs_error": tensor_entry_mean_abs_error[index],
            "uniform_simplex_entry_bound": tensor_entry_sup_error[index],
            "uniform_simplex_propagation_bound": propagation_bound[index],
            "max_test_future_risk_error": future_error_sup[index],
            "mean_test_future_risk_abs_error": future_error_mean_abs[index],
            "test_future_risk_rmse_descriptive": future_error_rmse[index],
            "max_test_probe_value_error": probe_value_error_sup[index],
            "conditional_probe_action_difference_count": int(
                probe_action_difference_count[index]
            ),
            "conditional_probe_action_non_tie_difference_count": int(
                probe_action_non_tie_difference_count[index]
            ),
            "conditional_probe_action_tie_ambiguous_difference_count": int(
                probe_action_tie_ambiguous_difference_count[index]
            ),
            "exact_probe_tie_count": exact_probe_tie_count,
            "conditional_terminal_action_difference_count": int(
                terminal_action_difference_count[index]
            ),
            "global_policy_kind_difference_count_at_cost_0_5": int(
                global_policy_kind_difference_count[index]
            ),
            "global_policy_difference_count_at_cost_0_5": int(
                global_policy_difference_count[index]
            ),
            "negative_M_entry_count": int(negative_matrix_count[index]),
            "negative_M_min": negative_matrix_min[index],
            "negative_Xi_entry_count_on_test_beliefs": int(negative_xi_count[index]),
            "negative_Xi_min_on_test_beliefs": negative_xi_min[index],
            "operator_matmuls_per_belief": int(
                lowrank_operator_matmuls_per_belief[index]
            ),
            "operator_matmuls_for_test_batch": int(
                lowrank_operator_matmuls_batch[index]
            ),
            "factor_representation_bytes": int(factor_bytes[index]),
            "factor_timing": factor_timings[index],
        })

    runner_sha256 = sha256_file(Path(__file__).resolve())
    arrays_path = HERE / "compression_arrays.npz"
    # Store the factor matrices in a padded object-free stack so the exact
    # reconstruction can be independently inspected without pickles.
    left_factors_padded = np.zeros((n_ranks, rows, int(RANKS[-1])), dtype=float)
    right_factors_padded = np.zeros((n_ranks, int(RANKS[-1]), n_h), dtype=float)
    for index, rank in enumerate(RANKS):
        left_factors_padded[index, :, :rank] = factor_matrices[index]
        right_factors_padded[index, :rank, :] = factor_rights[index]
    np.savez_compressed(
        arrays_path,
        source_matrix=matrix,
        operator_tensor=operator_tensor,
        likelihood=likelihood,
        risk=risk,
        actions=actions,
        singular_values=singular_values,
        svd_left=left,
        svd_right=right,
        requested_ranks=RANKS,
        alias_beliefs=alias_beliefs,
        dirichlet_beliefs=dirichlet,
        test_beliefs=test_beliefs,
        exact_xi=exact_xi,
        compressed_xi=compressed_xi,
        exact_future=exact_future,
        compressed_future=compressed_future,
        exact_probe_values=exact_probe_values,
        compressed_probe_values=compressed_probe_values,
        exact_probe_actions=exact_probe_actions,
        exact_probe_gap=exact_probe_gap,
        compressed_probe_actions=compressed_probe_actions,
        exact_terminal_values=exact_terminal_values,
        exact_direct_actions=exact_direct_actions,
        exact_global_kind=exact_global_kind,
        exact_global_action=exact_global_action,
        compressed_terminal_actions=compressed_terminal_actions,
        compressed_global_kind=compressed_global_kind,
        compressed_global_action=compressed_global_action,
        compressed_probe_gap=compressed_probe_gap,
        tensor_entry_sup_error=tensor_entry_sup_error,
        tensor_entry_rmse=tensor_entry_rmse,
        tensor_entry_mean_abs_error=tensor_entry_mean_abs_error,
        entry_row_supremum=entry_row_supremum,
        entry_bound_by_action=entry_bound_by_action,
        propagation_bound_by_action=propagation_bound_by_action,
        propagation_bound=propagation_bound,
        future_error_sup=future_error_sup,
        future_error_mean_abs=future_error_mean_abs,
        future_error_rmse=future_error_rmse,
        probe_value_error_sup=probe_value_error_sup,
        probe_action_difference_count=probe_action_difference_count,
        probe_action_non_tie_difference_count=probe_action_non_tie_difference_count,
        probe_action_tie_ambiguous_difference_count=probe_action_tie_ambiguous_difference_count,
        exact_probe_tie_count=np.asarray(exact_probe_tie_count),
        terminal_action_difference_count=terminal_action_difference_count,
        global_policy_kind_difference_count=global_policy_kind_difference_count,
        global_policy_difference_count=global_policy_difference_count,
        negative_matrix_count=negative_matrix_count,
        negative_matrix_min=negative_matrix_min,
        negative_xi_count=negative_xi_count,
        negative_xi_min=negative_xi_min,
        factor_bytes=factor_bytes,
        full_operator_matmuls_per_belief=np.asarray(full_operator_matmuls_per_belief),
        lowrank_operator_matmuls_per_belief=lowrank_operator_matmuls_per_belief,
        left_factors_padded=left_factors_padded,
        right_factors_padded=right_factors_padded,
    )
    arrays_sha256 = sha256_file(arrays_path)

    results = {
        "status": "PASS",
        "scope": "offline SVD audit of sealed observation-conditioned Xi only",
        "claim_boundary": (
            "standard fixed-math operator compression diagnostic; not a new "
            "theory result, robot/task-performance claim, controller change, "
            "or hardware result"
        ),
        "interpreter": os.environ.get("PYTHON", "") or __import__("sys").executable,
        "threads": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "frozen_sources": {
            "operator_arrays_sha256": source_arrays_sha256,
            "parent_results_sha256": source_results_sha256,
            "compression_protocol_sha256": protocol_sha256,
            "runner_sha256": runner_sha256,
            "parent_files_unchanged": True,
        },
        "dimensions": {
            "H": n_h,
            "A": n_a,
            "Y": n_y,
            "rows": rows,
            "matrix_shape": list(matrix.shape),
            "row_order": "row=((a*Y)+y)*A+j",
        },
        "operator_formula": "M[((a*Y)+y)*A+j,h] = L[h,a,y] * R[h,j]",
        "fixed_cost": OPTION_COST,
        "action_tie_diagnostic": {
            "tolerance": ACTION_TIE_TOLERANCE,
            "exact_conditional_probe_tie_count": exact_probe_tie_count,
            "meaning": "raw argmin differences are retained; non-tie counts exclude exact/near ties",
        },
        "exact_global_policy_at_cost_0_5": {
            "direct_terminal_count": exact_global_direct_count,
            "probe_then_terminal_count": exact_global_probe_count,
            "probe_then_terminal_test_indices": exact_global_probe_indices,
            "sealed_aliases_direct_terminal_count": int(
                np.sum(exact_global_kind[: len(alias_beliefs)] == "direct_terminal")
            ),
            "fresh_dirichlet_direct_terminal_count": int(
                np.sum(exact_global_kind[len(alias_beliefs):] == "direct_terminal")
            ),
        },
        "svd": {
            "requested_ranks": RANKS,
            "numerical_rank": numerical_rank,
            "tolerance": svd_tolerance,
            "singular_values": singular_values,
        },
        "belief_tests": {
            "sealed_alias_count": len(alias_beliefs),
            "fresh_dirichlet_count": len(dirichlet),
            "fresh_dirichlet_seed": DIRICHLET_SEED,
            "fresh_dirichlet_alpha": DIRICHLET_ALPHA,
            "total_test_beliefs": test_count,
            "uniform_reference": True,
            "labels_saved_in_order": test_labels,
        },
        "exact_compilation_checks": {
            "uniform_xi_max_residual": max_abs(uniform_xi_from_matrix - saved_uniform_xi),
            "alias_xi_max_residual": max_abs(alias_xi_from_matrix - np.concatenate((saved_alias_xi_plus, saved_alias_xi_minus), axis=0)),
            "uniform_backup_max_residual": max_abs(uniform_exact_option - saved_uniform_option),
            "alias_backup_max_residual": max_abs(exact_alias_probe - np.vstack((saved_alias_option_plus, saved_alias_option_minus))),
            "direct_risk_exact_max_residual": max_abs(exact_direct_again - exact_terminal_values),
        },
        "uniform_reference": {
            "exact_future_risk_by_probe": uniform_exact_future,
            "exact_probe_values_at_cost_0_5": uniform_exact_option,
            "direct_terminal_values": uniform @ risk,
            "conditional_probe_action": int(np.argmin(uniform_exact_future)),
            "direct_terminal_action": int(np.argmin(uniform @ risk)),
        },
        "rank_reports": rank_reports,
        "cost_accounting": {
            "full_operator_matmuls_per_belief": full_operator_matmuls_per_belief,
            "full_operator_matmuls_test_batch": full_operator_matmuls_batch,
            "lowrank_operator_matmuls_per_belief": lowrank_operator_matmuls_per_belief,
            "lowrank_operator_matmuls_test_batch": lowrank_operator_matmuls_batch,
            "direct_risk_matmuls_per_belief_unchanged": direct_risk_matmuls_per_belief,
            "direct_risk_matmuls_test_batch_unchanged": direct_risk_matmuls_batch,
            "full_M_bytes": full_matrix_bytes,
            "lowrank_factor_bytes": factor_bytes,
            "exact_direct_R_bytes_unchanged": direct_risk_bytes,
            "full_M_plus_exact_R_bytes": full_matrix_bytes + direct_risk_bytes,
            "lowrank_factor_plus_exact_R_bytes": factor_bytes + direct_risk_bytes,
            "full_timing": full_timing,
        },
        "artifacts": {
            "arrays_file": arrays_path.name,
            "per_belief_arrays": "compression_arrays.npz contains exact/compressed Xi/F/actions",
            "rmse_is_descriptive_only": True,
            "negative_entries_clipped": False,
            "negative_entries_are_probabilities": False,
        },
    }
    results_path = HERE / "RESULTS.json"
    results_path.write_text(
        json.dumps(jsonable(results), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    results_sha256 = sha256_file(results_path)
    (HERE / "RESULTS.sha256").write_text(
        f"compression_arrays.npz  {arrays_sha256}\n"
        f"RESULTS.json  {results_sha256}\n"
        f"PROTOCOL.txt  {protocol_sha256}\n"
        f"parent_operator_arrays.npz  {source_arrays_sha256}\n"
        f"parent_RESULTS.json  {source_results_sha256}\n"
        f"run_compression.py  {runner_sha256}\n",
        encoding="utf-8",
    )

    report_path = HERE / "REPORT.txt"
    lines = [
        "Low-rank Xi operator audit: PASS",
        "",
        "Scope: sealed offline operator compression only; no new world, sensor,",
        "controller, production, hardware, or task-performance operation.",
        "Direct b@R remains exact; only observation-conditioned Xi is approximated.",
        "",
        f"M shape={matrix.shape}, numerical rank={numerical_rank}, requested ranks={RANKS.tolist()}.",
        f"Exact compiled Xi/backup residual={max(exact_xi_from_matrix_residual := max_abs(uniform_xi_from_matrix - saved_uniform_xi), max_abs(exact_alias_probe - np.vstack((saved_alias_option_plus, saved_alias_option_minus)))):.3e}",
        f"Test beliefs: 64 sealed aliases + {len(dirichlet)} Dirichlet(alpha=ones, seed={DIRICHLET_SEED}); uniform retained.",
        "",
        "Rank summary (RMSE is descriptive only; simplex bounds are uniform):",
    ]
    for report in rank_reports:
        lines.extend([
            f"  r={report['rank']:>2}: entry_sup={report['tensor_entry_sup_error']:.6e}; "
            f"future_sup={report['max_test_future_risk_error']:.6e}; "
            f"bound_entry={report['uniform_simplex_entry_bound']:.6e}; "
            f"bound_backup={report['uniform_simplex_propagation_bound']:.6e};",
            f"       conditional_probe_action_changes(raw)={report['conditional_probe_action_difference_count']}/{test_count}; "
            f"non_tie={report['conditional_probe_action_non_tie_difference_count']}; "
            f"global_policy_changes_at_c=.5={report['global_policy_difference_count_at_cost_0_5']}/{test_count}; "
            f"negative_M={report['negative_M_entry_count']} (min {report['negative_M_min']:.3e}); "
            f"negative_Xi={report['negative_Xi_entry_count_on_test_beliefs']} (min {report['negative_Xi_min_on_test_beliefs']:.3e})",
        ])
    lines.extend([
        "",
        f"At frozen c=.5 the exact test baseline is {exact_global_direct_count} direct and "
        f"{exact_global_probe_count} probe-then-terminal ({int(np.sum(exact_global_kind[:len(alias_beliefs)] == 'direct_terminal'))}/"
        f"{len(alias_beliefs)} sealed aliases direct; {int(np.sum(exact_global_kind[len(alias_beliefs):] == 'direct_terminal'))}/"
        f"{len(dirichlet)} fresh Dirichlet direct).",
        "Conditional probe rankings/values are reported independently and were not tuned.",
        f"Exact conditional probe ties within {ACTION_TIE_TOLERANCE:.1e}: {exact_probe_tie_count}/{test_count}; "
        "raw argmin changes at ties are not substantive action changes.",
        f"Full operator timing median per test batch: {full_timing['median_s']:.6e}s.",
        "",
        f"compression_arrays.npz sha256={arrays_sha256}",
        f"RESULTS.json sha256={results_sha256}",
        f"PROTOCOL.txt sha256={protocol_sha256}",
        f"parent operator_arrays.npz sha256={source_arrays_sha256}",
        f"parent RESULTS.json sha256={source_results_sha256}",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "results_sha256": results_sha256,
        "arrays_sha256": arrays_sha256,
        "report": str(report_path),
        "status": "PASS",
    }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
