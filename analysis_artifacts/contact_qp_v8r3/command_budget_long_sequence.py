#!/usr/bin/env python3
"""Offline randomized CommandBudget comparison against a Decimal work oracle.

The oracle integrates only committed, frozen six-axis pairs. It does not call
production power/settlement helpers or inspect the budget's private pair state.
This checks a logical command model, not robot work or hardware safety.
"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, getcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE_NAMES = (
    "analysis_artifacts/contact_qp_v8r3/command_budget_long_sequence.py",
    "peirastic/contact_qp/command_budget.py",
    "peirastic/contact_qp/port_constraint.py",
    "peirastic/contact_qp/types.py",
)
getcontext().prec = 50
D = Decimal.from_float
ZERO = Decimal(0)


def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCE_NAMES}


class WorkOracle:
    def __init__(self, initial, capacity, reserve, horizon):
        self.balance = D(initial)
        self.capacity = D(capacity)
        self.reserve = D(reserve)
        self.horizon = D(horizon)
        self.active = None
        self.pending = None
        self.last_time = None
        self.axis_work = [ZERO] * 6
        self.net_work = ZERO
        self.positive_input_work = ZERO
        self.negative_input_work = ZERO
        self.cap_discarded = ZERO
        self.cap_events = 0
        self.normal_only_balance = D(initial)

    @staticmethod
    def pair(command_id, raw, velocity, reviewed, horizon):
        # Construct a separate immutable numeric copy, using explicit scalar
        # products. No production wrench/power helper is involved.
        wrench = tuple(-D(float(x)) for x in raw)
        twist = tuple(D(float(x)) for x in velocity)
        axis_power = tuple(w * v for w, v in zip(wrench, twist))
        return dict(command_id=command_id, axis_power=axis_power,
                    power=sum(axis_power, ZERO), expires=D(reviewed + horizon))

    def settle(self, now):
        endpoint = D(now)
        if self.last_time is not None:
            assert endpoint >= self.last_time
        if self.active is not None:
            end = min(endpoint, self.active["expires"])
            dt = max(ZERO, end - self.last_time)
            axis = [p * dt for p in self.active["axis_power"]]
            work = sum(axis, ZERO)
            unclipped = self.balance + work
            discarded = max(ZERO, unclipped - self.capacity)
            self.cap_discarded += discarded
            self.cap_events += int(discarded > 0)
            self.balance = min(self.capacity, unclipped)
            self.normal_only_balance = min(self.capacity, self.normal_only_balance + axis[2])
            self.axis_work = [a + b for a, b in zip(self.axis_work, axis)]
            self.net_work += work
            self.positive_input_work += max(ZERO, work)
            self.negative_input_work += min(ZERO, work)
            if endpoint >= self.active["expires"]:
                self.active = None
        self.last_time = endpoint

    def amounts(self):
        old = ZERO
        if self.active is not None:
            old = max(ZERO, -self.active["power"]) * max(
                ZERO, self.active["expires"] - self.last_time)
        new = ZERO if self.pending is None else max(ZERO, -self.pending["power"]) * self.horizon
        reserved = old + new
        raw = self.balance - self.reserve - reserved
        return dict(balance_j=self.balance, reserved_j=reserved,
                    raw_available_j=raw, available_j=max(ZERO, raw))


def run(steps, seed):
    before = hashes()
    from peirastic.contact_qp.command_budget import CommandBudget

    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    config = dict(initial_j=.08, capacity_j=.12, stopping_reserve_j=.002,
                  max_command_interval_s=.05,
                  settlement_port="logical_final_model",
                  wrench_convention="negative_control_raw_tcp_v1")
    tank = CommandBudget(**config)
    oracle = WorkOracle(.08, .12, .002, .05)
    errors = {key: 0. for key in oracle.amounts()}
    minimums = {key: float("inf") for key in oracle.amounts()}
    max_reserved = 0.
    checkpoints = 0
    events = Counter()
    max_events_buffered = 0
    no_send = 0
    positive_pending = 0
    opposite_pending = 0
    scaled_spend = 0
    changed_wrench_snapshots = 0
    now = 1.
    max_epoch_hold = 0.
    active_committed = None
    trace_digest = hashlib.sha256()
    checkpoints_sparse = []

    def check(label):
        nonlocal checkpoints, max_reserved
        expected = oracle.amounts()
        actual = tank.facts
        checkpoints += 1
        for key, value in expected.items():
            actual_value = float(actual[key])
            error = abs(actual_value - float(value))
            errors[key] = max(errors[key], error)
            minimums[key] = min(minimums[key], actual_value)
            assert error <= 2e-11, (label, key, actual_value, str(value), error)
        assert actual["raw_available_j"] >= -2e-11, (label, actual)
        assert actual["balance_j"] >= config["stopping_reserve_j"] - 2e-11
        assert actual["balance_j"] <= config["capacity_j"] + 2e-11
        assert actual["physical_certified"] is False
        assert actual["predicted_recovery_credited_j"] == 0.
        max_reserved = max(max_reserved, actual["reserved_j"])

    def settle_at(value):
        oracle.settle(value)

    def collect_events():
        nonlocal max_events_buffered
        batch = tank.drain_events()
        max_events_buffered = max(max_events_buffered, len(batch))
        events.update(event["event"] for event in batch)
        assert tank.drain_events() == []

    for index in range(steps):
        now += float(rng.uniform(.001, .004))
        settle_at(now)
        assert tank.advance(now)
        check("advance")

        # Distinct six-axis measurements and pair orientations never alter an
        # already committed pair. Only the second snapshot is reviewed.
        raw = rng.normal(size=6) * np.array([2., 2., 4., .2, .2, .2])
        alternate_raw = raw + rng.normal(size=6) * np.array([3., 3., 3., .3, .3, .3])
        angle = float(rng.uniform(-np.pi, np.pi))
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        tank.snapshot(now_s=now, wrench_control_raw=alternate_raw,
                      rotation_base_tcp=rotation)
        check("uncommitted_alternate_snapshot")
        snapshot = tank.snapshot(now_s=now, wrench_control_raw=raw,
                                 rotation_base_tcp=rotation)
        changed_wrench_snapshots += 1
        check("uncommitted_replacement_snapshot")

        velocity = rng.normal(size=6) * np.array([.006, .006, .004, .08, .08, .08])
        model_power = sum(-float(w) * float(v) for w, v in zip(raw, velocity))
        # Long alternating spend/recovery phases exercise near-empty reserve
        # and capacity clipping. All six components remain present.
        spend = index % 10000 < 6500
        if (model_power > 0.) == spend:
            velocity = -velocity
            model_power = -model_power
        if spend:
            # Candidate generation uses the public advertised budget, also
            # bounded by the oracle. At numerical dust request zero spend;
            # ledger comparisons still use the independent Decimal result.
            spendable = min(snapshot.available_j, float(oracle.amounts()["available_j"]))
            allowed = 0. if spendable < 1e-12 else .75 * spendable / .05
            if -model_power > allowed:
                velocity *= allowed / (-model_power)
                scaled_spend += 1

        now += float(rng.uniform(.00005, .0005))
        settle_at(now)
        reviewed = now
        assert tank.reserve(index, snapshot, velocity, now_s=now), (index, tank.facts)
        oracle.pending = oracle.pair(index, raw, velocity, reviewed, .05)
        positive_pending += int(oracle.pending["power"] > 0)
        opposite_pending += int(oracle.active is not None
                                and oracle.pending["power"] * oracle.active["power"] < 0)
        check("reserve_without_precredit")
        trace_digest.update(np.asarray([index, now, *raw, *velocity], dtype="<f8").tobytes())

        if (index + 1) % 17 == 0:
            now += float(rng.uniform(.0001, .0008))
            settle_at(now)
            assert tank.reject_new_only(index, definitely_not_sent=True, now_s=now)
            oracle.pending = None
            no_send += 1
            check("known_no_send_old_pair_continues")
        else:
            tank.publication_started(index)
            check("publication_start_without_precredit")
            now += float(rng.uniform(.0001, .0008))
            settle_at(now)
            tank.commit(index, velocity, now_s=now, rotation_base_tcp=rotation, dual_success=True)
            if active_committed is not None:
                max_epoch_hold = max(max_epoch_hold, now - active_committed)
            active_committed = now
            oracle.active = oracle.pending
            oracle.pending = None
            check("commit_old_pair_settled_once")

        collect_events()
        if (index + 1) % 10000 == 0:
            checkpoints_sparse.append(dict(step=index + 1, time_s=now,
                balance_j=tank.balance_j, available_j=tank.available_j,
                cap_discarded_j=float(oracle.cap_discarded)))
            print(f"{index + 1}/{steps}: balance={tank.balance_j:.9g} J, "
                  f"max_error={max(errors.values()):.3g} J", flush=True)

    now += .002
    settle_at(now)
    tank.stop(now_s=now)
    if active_committed is not None:
        max_epoch_hold = max(max_epoch_hold, now - active_committed)
    oracle.active = None
    check("logical_stop")
    stopped_balance = tank.balance_j
    now += .004
    settle_at(now)
    assert not tank.advance(now)
    assert tank.balance_j == stopped_balance
    check("after_logical_stop_no_further_work")
    collect_events()

    assert no_send == steps // 17
    assert positive_pending > 0 and opposite_pending > 0 and oracle.cap_events > 0
    assert abs(float(oracle.net_work - oracle.axis_work[2])) > .001
    assert abs(float(oracle.balance - (D(.08) + oracle.net_work - oracle.cap_discarded))) < 1e-40
    after = hashes()
    elapsed = time.perf_counter() - start
    result = dict(
        schema="command_budget_long_sequence_decimal_oracle_v1",
        scope="Pure offline logical command-model API check; no hardware, measured work, closed-loop control, or physical passivity claim.",
        source_sha256=before, source_sha256_end=after,
        source_unchanged_during_run=before == after,
        seed=seed, rng="numpy.default_rng / PCG64", steps=steps,
        elapsed_wall_s=elapsed, simulated_time_s=now - 1.,
        command=shlex.join([sys.executable, *sys.argv]),
        relevant_environment={key: os.environ.get(key) for key in (
            "PYTHONPATH", "LD_LIBRARY_PATH", "PYTHONNOUSERSITE",
            "PYTHONDONTWRITEBYTECODE", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        python=platform.python_version(), numpy=np.__version__, config=config,
        oracle="50-digit Decimal arithmetic from exact floating-point inputs; scalar six-axis product; chronological committed-pair intervals, upper capacity clipping, independent old+pending liabilities. Uncommitted snapshots/pairs earn no work.",
        input_trace_sha256=trace_digest.hexdigest(), checkpoints=checkpoints,
        maximum_absolute_error_j=errors, minimum_observed_j=minimums,
        maximum_reserved_j=max_reserved, final=tank.facts,
        work_input_to_logical_tank_j=float(oracle.net_work),
        work_output_from_logical_tank_j=float(-oracle.net_work),
        positive_input_work_j=float(oracle.positive_input_work),
        negative_input_work_j=float(oracle.negative_input_work),
        axis_input_work_j=dict(zip(("Fx_vx", "Fy_vy", "Fz_vz", "Mx_wx", "My_wy", "Mz_wz"),
                                  map(float, oracle.axis_work))),
        normal_only_counterfactual_final_balance_j=float(oracle.normal_only_balance),
        capacity_discarded_j=float(oracle.cap_discarded), cap_clipping_partitions=oracle.cap_events,
        known_no_send_steps=no_send, committed_steps=steps - no_send,
        changed_uncommitted_wrench_snapshots=changed_wrench_snapshots,
        positive_power_pending_steps=positive_pending,
        pending_and_old_power_opposite_sign_steps=opposite_pending,
        scaled_spend_candidates=scaled_spend,
        candidate_generation="Random six-axis directions; spend requests scaled to 75% of min(public snapshot, oracle) available / 50ms, zero spend below 1e-12 J; no change to the balance oracle.",
        max_successful_pair_actual_logical_hold_s=max_epoch_hold,
        event_counts=dict(events), maximum_events_buffered_with_per_step_drain=max_events_buffered,
        sparse_checkpoints=checkpoints_sparse,
        limitations=[
            "This integrates frozen body-frame command-model pairs, not changing measured body frames.",
            "Output ends at logical stop; real actuator stopping/tail energy remains unknown.",
            "All publications in commit branches are assumed dual_success; no device is contacted.",
            "This long normal-path run does not replace separate invalid-input/expiry/partial-publication unit tests.",
        ],
        passed=True,
    )
    assert before == after, "Production or analysis source changed during the run"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    if args.steps < 100000:
        parser.error("This long-run validation requires at least 100000 steps")
    result = run(args.steps, args.seed)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"saved {args.output}; wall={result['elapsed_wall_s']:.3f}s", flush=True)
