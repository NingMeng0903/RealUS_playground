"""Finite V8.5.2 identifiability and option-kernel stress tests.

This module is intentionally separate from the SMPL-X geometry and scan
simulator.  It checks only finite distributions, a finite option kernel, and
the evidence release gate.  Its Monte Carlo times are synthetic option-kernel
durations; they are not robot or ultrasound measurements.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .evidence_ledger import EvidenceLedger, EvidenceRecord
from .learning import common_policy_mixture


BINARY = (-1, 1)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def binary_joint(rho: float) -> dict[tuple[int, int], float]:
    """Return the binary table P(U,V)=(1+rho U V)/4."""

    rho = float(rho)
    if not np.isfinite(rho) or not -1.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [-1, 1]")
    return {(u, v): (1.0 + rho * u * v) / 4.0 for u in BINARY for v in BINARY}


def _marginal(table: Mapping[tuple[int, int], float], axis: int) -> dict[int, float]:
    return {
        value: float(sum(prob for pair, prob in table.items() if pair[axis] == value))
        for value in BINARY
    }


def total_variation(first: Mapping[Any, float], second: Mapping[Any, float]) -> float:
    keys = set(first) | set(second)
    return 0.5 * sum(abs(float(first.get(key, 0.0)) - float(second.get(key, 0.0))) for key in keys)


def binary_pairing_audit(rhos: Sequence[float] = (-0.8, 0.0, 0.8)) -> dict[str, Any]:
    """Show same U/V margins but different task-relevant pairings."""

    tables = {str(float(rho)): binary_joint(float(rho)) for rho in rhos}
    margins = {
        rho: {"U": _marginal(table, 0), "V": _marginal(table, 1)}
        for rho, table in tables.items()
    }
    values = {
        rho: float(sum(prob for (u, v), prob in table.items() if u == v))
        for rho, table in tables.items()
    }
    reference = next(iter(tables.values()))
    margin_tvs = {
        rho: {
            axis: total_variation(margins[next(iter(tables))][axis], margins[rho][axis])
            for axis in ("U", "V")
        }
        for rho in tables
    }
    return {
        "rho_values": [float(rho) for rho in rhos],
        "tables": tables,
        "margins": margins,
        "margin_total_variation_from_first": margin_tvs,
        "task_value_policy_m_equals_U": values,
        "task_value_range": [float(min(values.values())), float(max(values.values()))],
        "all_margins_equal": all(all(value == 0.0 for value in row.values()) for row in margin_tvs.values()),
        "pairing_changes_task_value": len(set(values.values())) > 1,
        "unpaired_margin_only_interval": [0.0, 1.0],
    }


def _triple_distribution(sign: int) -> dict[tuple[int, int, int], float]:
    result: dict[tuple[int, int, int], float] = {}
    for u in BINARY:
        for v in BINARY:
            w = sign * u * v
            result[(u, v, w)] = 0.25
    return result


def _project_distribution(
    table: Mapping[tuple[int, int, int], float], axes: tuple[int, int]
) -> dict[tuple[int, int], float]:
    result: dict[tuple[int, int], float] = {}
    for state, probability in table.items():
        key = (state[axes[0]], state[axes[1]])
        result[key] = result.get(key, 0.0) + float(probability)
    return result


def _pair_only_policies() -> Iterable[dict[tuple[int, int], int]]:
    keys = [(u, v) for u in BINARY for v in BINARY]
    for mask in range(16):
        yield {key: (1 if (mask >> index) & 1 else -1) for index, key in enumerate(keys)}


def triple_probe_audit() -> dict[str, Any]:
    """Verify the parity counterexample: all two-way probes agree, triples do not."""

    positive = _triple_distribution(+1)
    negative = _triple_distribution(-1)
    pairwise_tvs = {
        f"{left}{right}": total_variation(
            _project_distribution(positive, (left, right)),
            _project_distribution(negative, (left, right)),
        )
        for left, right in ((0, 1), (0, 2), (1, 2))
    }
    policy_values = []
    for policy in _pair_only_policies():
        positive_value = sum(
            probability
            for (u, v, w), probability in positive.items()
            if policy[(u, v)] == w
        )
        negative_value = sum(
            probability
            for (u, v, w), probability in negative.items()
            if policy[(u, v)] == w
        )
        policy_values.append({"positive": positive_value, "negative": negative_value})
    common_pair_only_lower = max(min(item.values()) for item in policy_values)
    triple_probe_value = 1.0  # observing W and selecting m=W in either world
    return {
        "worlds": {"positive": positive, "negative": negative},
        "pairwise_total_variation": pairwise_tvs,
        "all_pairwise_probes_identical": all(value == 0.0 for value in pairwise_tvs.values()),
        "full_triple_total_variation": total_variation(positive, negative),
        "pair_only_policy_value_range": {
            "positive": [min(item["positive"] for item in policy_values), max(item["positive"] for item in policy_values)],
            "negative": [min(item["negative"] for item in policy_values), max(item["negative"] for item in policy_values)],
        },
        "pair_only_common_policy_lower": common_pair_only_lower,
        "triple_probe_policy_value": triple_probe_value,
        "additional_pair_only_data_changes_model_set": False,
        "triple_probe_distinguishes_worlds": total_variation(positive, negative) > 0.0,
    }


def common_policy_audit() -> dict[str, Any]:
    """Call the production shared-policy LP on opposite model rows."""

    matrix = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    mixture, lower, upper = common_policy_mixture(matrix)
    return {
        "matrix": matrix,
        "mixture": mixture,
        "lower": lower,
        "upper": upper,
        "shared_policy_not_model_oracle": lower < upper,
        "mixture_sum": float(mixture.sum()),
    }


@dataclass(frozen=True)
class OptionResult:
    success: bool
    duration: float
    status: str
    trace: tuple[str, ...]
    fresh_packet_visible: bool


@dataclass(frozen=True)
class FiniteOptionTree:
    """A finite, deterministic option kernel with an auditable release gate."""

    probe_duration: float = 0.9
    correct_duration: float = 1.4
    mismatch_duration: float = 0.8
    recovery_duration: float = 1.0
    recheck_duration: float = 0.6

    def _ledger(self, hidden_mode: int, release_time: float, retain_old: bool, run_id: int) -> EvidenceLedger:
        ledger = EvidenceLedger()
        ledger.add(
            EvidenceRecord(
                f"old-{run_id}",
                "finite-object",
                "target",
                -1.0,
                0.0,
                release_time,
                {"mode": hidden_mode},
                kind="historical",
            )
        )
        if not retain_old:
            ledger.set_hidden(f"old-{run_id}", True)
        return ledger

    def _probe(self, clock: float, hidden_mode: int, ledger: EvidenceLedger, run_id: int, trace: list[str]) -> float:
        clock += self.probe_duration
        packet_id = f"probe-{run_id}"
        ledger.add(
            EvidenceRecord(
                packet_id,
                "finite-object",
                "target",
                clock,
                clock,
                clock,
                {"mode": hidden_mode},
                kind="fresh",
            )
        )
        trace.append("Probe")
        return clock

    def _execute(self, clock: float, hidden_mode: int, mode: int, trace: list[str]) -> tuple[float, bool]:
        if mode == hidden_mode:
            clock += self.correct_duration
            trace.append("ExecuteCorrect")
            return clock, True
        clock += self.mismatch_duration
        trace.append("ExecuteMismatch")
        clock += self.recovery_duration
        trace.append("Recover")
        return clock, True

    def run(
        self,
        hidden_mode: int,
        policy: str,
        *,
        release_time: float = 0.0,
        retain_old: bool = True,
        run_id: int = 0,
    ) -> OptionResult:
        if hidden_mode not in BINARY:
            raise ValueError("hidden_mode must be -1 or +1")
        if policy not in {"adaptive_old", "blind_plus", "blind_minus", "probe_then_execute", "oracle"}:
            raise ValueError(f"unknown policy {policy!r}")
        clock = 0.0
        trace: list[str] = []
        ledger = self._ledger(hidden_mode, float(release_time), retain_old, run_id)
        mode: int | None = None
        if policy == "oracle":
            mode = hidden_mode
            trace.append("OracleChoice")
        elif policy == "blind_plus":
            mode = 1
            trace.append("BlindPlus")
        elif policy == "blind_minus":
            mode = -1
            trace.append("BlindMinus")
        elif policy == "probe_then_execute":
            clock = self._probe(clock, hidden_mode, ledger, run_id, trace)
            feature = ledger.latest_feature(clock, "finite-object", region_id="target", fresh_after=0.0, include_historical=False)
            mode = int(feature.payload["mode"]) if feature is not None else None
        else:
            feature = ledger.latest_feature(clock, "finite-object", region_id="target")
            if feature is not None:
                mode = int(feature.payload["mode"])
                trace.append("ReadOld")
            elif retain_old and float(release_time) > clock:
                clock = float(release_time)
                trace.append("WaitForRelease")
                feature = ledger.latest_feature(clock, "finite-object", region_id="target")
                if feature is not None:
                    mode = int(feature.payload["mode"])
                    trace.append("ReadOld")
            if mode is None:
                clock = self._probe(clock, hidden_mode, ledger, run_id, trace)
                feature = ledger.latest_feature(clock, "finite-object", region_id="target", fresh_after=0.0, include_historical=False)
                mode = int(feature.payload["mode"]) if feature is not None else None
        if mode not in BINARY:
            return OptionResult(False, float("inf"), "no_visible_mode", tuple(trace), False)
        clock, action_success = self._execute(clock, hidden_mode, mode, trace)
        if not action_success:
            return OptionResult(False, clock, "action_failed", tuple(trace), False)
        clock += self.recheck_duration
        trace.append("FreshRecheck")
        fresh_id = f"recheck-{run_id}"
        ledger.add(
            EvidenceRecord(
                fresh_id,
                "finite-object",
                "target",
                clock,
                clock,
                clock,
                {"accepted": True},
                kind="fresh",
            )
        )
        fresh = ledger.latest_feature(
            clock,
            "finite-object",
            region_id="target",
            fresh_after=0.0,
            include_historical=False,
        )
        return OptionResult(bool(fresh is not None), clock, "complete" if fresh is not None else "missing_fresh", tuple(trace), bool(fresh is not None))

    def kernel_manifest(self) -> dict[str, float]:
        return {
            "probe": self.probe_duration,
            "execute_correct": self.correct_duration,
            "execute_mismatch": self.mismatch_duration,
            "recover_after_mismatch": self.recovery_duration,
            "fresh_recheck": self.recheck_duration,
        }


def exact_completion_cdf(
    tree: FiniteOptionTree,
    policy: str,
    deadlines: Sequence[float],
    *,
    release_time: float = 0.0,
    retain_old: bool = True,
) -> np.ndarray:
    results = [
        tree.run(mode, policy, release_time=release_time, retain_old=retain_old, run_id=index)
        for index, mode in enumerate(BINARY)
    ]
    return np.asarray(
        [np.mean([result.success and result.duration <= float(deadline) for result in results]) for deadline in deadlines],
        dtype=float,
    )


def monte_carlo_completion_cdf(
    tree: FiniteOptionTree,
    policy: str,
    deadlines: Sequence[float],
    *,
    release_time: float = 0.0,
    retain_old: bool = True,
    n: int = 20_000,
    seed: int = 20261108,
) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    results = [
        tree.run(int(rng.choice(BINARY)), policy, release_time=release_time, retain_old=retain_old, run_id=index)
        for index in range(int(n))
    ]
    return np.asarray(
        [np.mean([result.success and result.duration <= float(deadline) for result in results]) for deadline in deadlines],
        dtype=float,
    )


def policy_library_oracle(
    tree: FiniteOptionTree,
    policies: Sequence[str],
    deadlines: Sequence[float],
    *,
    release_time: float = 0.0,
    retain_old: bool = True,
) -> dict[str, Any]:
    curves = {
        policy: exact_completion_cdf(
            tree,
            policy,
            deadlines,
            release_time=release_time,
            retain_old=retain_old,
        )
        for policy in policies
    }
    matrix = np.vstack([curves[policy] for policy in policies])
    best = np.max(matrix, axis=0)
    selected = [policies[int(index)] for index in np.argmax(matrix, axis=0)]
    full_information = exact_completion_cdf(
        tree,
        "oracle",
        deadlines,
        release_time=release_time,
        retain_old=retain_old,
    )
    return {
        "policies": list(policies),
        "curves": curves,
        "best_library_curve": best,
        "selected_policy_by_deadline": selected,
        "full_information_oracle_curve": full_information,
    }


def _case(
    tree: FiniteOptionTree,
    name: str,
    policy: str,
    deadlines: Sequence[float],
    *,
    release_time: float,
    retain_old: bool,
    mc_n: int,
    seed: int,
) -> dict[str, Any]:
    predicted = exact_completion_cdf(
        tree,
        policy,
        deadlines,
        release_time=release_time,
        retain_old=retain_old,
    )
    measured = monte_carlo_completion_cdf(
        tree,
        policy,
        deadlines,
        release_time=release_time,
        retain_old=retain_old,
        n=mc_n,
        seed=seed,
    )
    return {
        "name": name,
        "policy": policy,
        "release_time": release_time,
        "retain_old": retain_old,
        "deadlines": list(deadlines),
        "predicted_completion_cdf": predicted,
        "independent_monte_carlo_cdf": measured,
        "absolute_error": np.abs(predicted - measured),
        "max_absolute_error": float(np.max(np.abs(predicted - measured))),
        "mc_n": mc_n,
    }


def build_report(*, mc_n: int = 20_000, seed: int = 20261108) -> dict[str, Any]:
    deadlines = (2.0, 2.4, 2.8, 3.0, 3.2)
    tree = FiniteOptionTree()
    policies = ("adaptive_old", "blind_plus", "blind_minus", "probe_then_execute")
    a_case = _case(
        tree,
        "A_local_to_untrained_chain",
        "adaptive_old",
        deadlines,
        release_time=0.0,
        retain_old=True,
        mc_n=mc_n,
        seed=seed + 1,
    )
    b_early = _case(
        tree,
        "B_early_release",
        "adaptive_old",
        deadlines,
        release_time=0.0,
        retain_old=True,
        mc_n=mc_n,
        seed=seed + 2,
    )
    b_late = _case(
        tree,
        "B_late_release",
        "adaptive_old",
        deadlines,
        release_time=0.8,
        retain_old=True,
        mc_n=mc_n,
        seed=seed + 3,
    )
    c_retained = _case(
        tree,
        "C_old_knowledge_retained",
        "adaptive_old",
        deadlines,
        release_time=0.0,
        retain_old=True,
        mc_n=mc_n,
        seed=seed + 4,
    )
    c_hidden = _case(
        tree,
        "C_old_knowledge_hidden",
        "adaptive_old",
        deadlines,
        release_time=0.0,
        retain_old=False,
        mc_n=mc_n,
        seed=seed + 5,
    )
    return {
        "schema": "v852_finite_limits_report_v1",
        "scope": "finite mathematical and program-option stress test only; not geometry, B-mode, or real execution",
        "protocol": {
            "hidden_modes": list(BINARY),
            "deadlines": deadlines,
            "old_capture_time": -1.0,
            "request_time": 0.0,
            "fresh_acceptance": "capture_time >= request_time and kind=fresh",
            "late_release_can_wait": True,
            "recovery_allowed": True,
            "same_old_packet_capture_cost": True,
        },
        "binary_pairing": binary_pairing_audit(),
        "triple_parity": triple_probe_audit(),
        "common_policy_lp": common_policy_audit(),
        "option_kernel": tree.kernel_manifest(),
        "experiments": {
            "A": a_case,
            "B": {"early": b_early, "late": b_late},
            "C": {"retained": c_retained, "hidden": c_hidden},
        },
        "tree_library_oracle": policy_library_oracle(
            tree,
            policies,
            deadlines,
            release_time=0.0,
            retain_old=True,
        ),
        "interpretation_limits": [
            "Pair-only observations cannot shrink the parity model set because all pairwise distributions are identical.",
            "The option durations are finite-kernel values, not robot timing measurements.",
            "Monte Carlo checks an independently sampled copy of the declared kernel; it is not an independent physical test.",
            "The release gate controls upper-level visibility; a hidden historical packet is never accepted as fresh evidence.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/v852_finite_limits_20261108"))
    parser.add_argument("--mc-n", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261108)
    args = parser.parse_args(argv)
    report = build_report(mc_n=args.mc_n, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    _write_json(args.output / "finite_limits_report.json", report)
    _write_json(args.output / "binary_pairing.json", report["binary_pairing"])
    _write_json(args.output / "triple_parity.json", report["triple_parity"])
    _write_json(args.output / "common_policy_lp.json", report["common_policy_lp"])
    _write_json(args.output / "option_tree_abc.json", {
        "scope": report["scope"],
        "protocol": report["protocol"],
        "option_kernel": report["option_kernel"],
        "experiments": report["experiments"],
        "tree_library_oracle": report["tree_library_oracle"],
    })
    print(json.dumps({
        "output": str(args.output),
        "scope": report["scope"],
        "pairing_changes_task_value": report["binary_pairing"]["pairing_changes_task_value"],
        "pairwise_parity_identical": report["triple_parity"]["all_pairwise_probes_identical"],
        "common_policy_lower": report["common_policy_lp"]["lower"],
        "max_mc_error_A": report["experiments"]["A"]["max_absolute_error"],
        "max_mc_error_B_early": report["experiments"]["B"]["early"]["max_absolute_error"],
        "max_mc_error_B_late": report["experiments"]["B"]["late"]["max_absolute_error"],
        "max_mc_error_C_retained": report["experiments"]["C"]["retained"]["max_absolute_error"],
        "max_mc_error_C_hidden": report["experiments"]["C"]["hidden"]["max_absolute_error"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

