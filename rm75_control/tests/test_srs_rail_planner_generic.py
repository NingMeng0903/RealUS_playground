from __future__ import annotations

import math

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.srs_rail_planner import (
    EdgeEvaluation,
    NodeEvaluation,
    SrsCandidate,
    SrsPlanRequest,
    SrsPlannerConfig,
    SrsRailPosturePlanner,
    lexicographic_materially_better,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def _wide_config(**overrides) -> SrsPlannerConfig:
    values = dict(
        edge_samples=8,
        rail_v_max_m_s=100.0,
        rail_a_max_m_s2=1000.0,
        arm_v_max_rad_s=100.0,
        arm_a_max_rad_s2=1000.0,
        psi_v_max_rad_s=100.0,
        psi_a_max_rad_s2=1000.0,
        consistent_plans_to_commit=1,
        min_dwell_s=0.0,
    )
    values.update(overrides)
    return SrsPlannerConfig(**values)


def _node(ctx) -> NodeEvaluation:
    # A deterministic injected stand-in for RM75 SRS mapping.  Edge samples
    # depend on unwrapped psi, so a +/-pi crossing is testable without Pinocchio.
    q = np.zeros(8)
    q[0] = ctx.candidate.rail_m
    q[1] = ctx.candidate.unwrapped_psi_rad
    q[2] = 0.01 * ctx.candidate.branch
    return NodeEvaluation(True, q=q, score=ctx.candidate.score)


def test_lexicographic_score_and_horizon_only_publish_short_corridor() -> None:
    # Objective 0 dominates objective 1; this catches accidental weighted sums.
    assert lexicographic_materially_better((0.0, 1000.0), (1.0, 0.0))
    current = SrsCandidate(0.0, 0.0, 0, q=np.zeros(8))
    layers = (
        (SrsCandidate(0.01, 0.1, 0, score=(0.0, 1000.0), tag="lex"),
         SrsCandidate(0.01, 0.1, 0, score=(1.0, 0.0), tag="weighted")),
        (SrsCandidate(0.02, 0.2, 0, score=(0.0, 0.0), tag="near"),),
        (SrsCandidate(0.03, 0.3, 0, score=(0.0, 0.0), tag="far"),),
    )
    edge_calls = []

    def edge(ctx):
        edge_calls.append(ctx)
        assert len(ctx.samples) == 8
        assert all(0.0 < sample.alpha <= 1.0 for sample in ctx.samples)
        return EdgeEvaluation(True)

    planner = SrsRailPosturePlanner(
        node_evaluator=_node,
        edge_evaluator=edge,
        config=_wide_config(corridor_nodes=2),
    )
    try:
        plan = planner.plan_now(
            SrsPlanRequest(
                current=current,
                horizon=("p1", "p2", "p3"),
                candidate_layers=layers,
                current_waypoint="p0",
            )
        )
        assert plan.corridor[0].tag == "lex"
        assert [c.tag for c in plan.corridor] == ["lex", "near"]
        assert plan.short_target.tag == "near"
        assert plan.layers_evaluated == 3
        assert len(edge_calls) >= 3
    finally:
        planner.shutdown()


def test_every_edge_sample_runs_collision_and_unwraps_psi() -> None:
    current = SrsCandidate(0.0, math.pi - 0.04, 0, 0, q=np.zeros(8))
    bad = SrsCandidate(0.02, -math.pi + 0.04, 0, 1, score=(-10.0,), tag="bad")
    safe = SrsCandidate(0.01, math.pi - 0.02, 0, 0, score=(0.0,), tag="safe")
    seen_bad_alphas = []

    def collision(sample) -> bool:
        candidate = sample.node.candidate
        if candidate.tag == "bad":
            seen_bad_alphas.append(sample.node.edge_alpha)
            # Interior collision only: checking endpoints would miss it.
            return not (0.35 < sample.node.edge_alpha < 0.75)
        return True

    planner = SrsRailPosturePlanner(
        node_evaluator=_node,
        collision_evaluator=collision,
        config=_wide_config(),
    )
    try:
        plan = planner.plan_now(
            SrsPlanRequest(current=current, manifold=(bad, safe), contact=False)
        )
        assert plan.target.tag == "safe"
        assert any(0.35 < alpha < 0.75 for alpha in seen_bad_alphas)
    finally:
        planner.shutdown()


def test_contact_locks_branch_and_winding_noncontact_switch_gets_full_edge() -> None:
    current = SrsCandidate(0.0, 0.0, 2, 0, q=np.zeros(8))
    switched = SrsCandidate(0.0, 0.02, 3, 1, score=(-1.0,), tag="switch")
    locked = SrsCandidate(0.0, 0.01, 2, 0, score=(0.0,), tag="locked")
    switches = []

    def edge(ctx):
        if ctx.discrete_switch:
            switches.append(ctx)
            assert len(ctx.samples) == 8
        return True

    planner = SrsRailPosturePlanner(
        node_evaluator=_node,
        edge_evaluator=edge,
        config=_wide_config(),
    )
    try:
        contact_plan = planner.plan_now(
            SrsPlanRequest(current=current, manifold=(switched, locked), contact=True)
        )
        assert contact_plan.target.discrete_key == current.discrete_key
        planner.reset_commit()
        free_plan = planner.plan_now(
            SrsPlanRequest(current=current, manifold=(switched, locked), contact=False)
        )
        assert free_plan.target.tag == "switch"
        assert switches and switches[-1].discrete_switch
        # A newly asserted contact lock overrides a free-space commit
        # immediately; the dwell gate may not leak the old topology.
        relocked = planner.plan_now(
            SrsPlanRequest(current=current, manifold=(switched, locked), contact=True)
        )
        assert relocked.target.discrete_key == current.discrete_key
    finally:
        planner.shutdown()


def test_velocity_acceleration_are_fail_closed() -> None:
    current = SrsCandidate(0.0, 0.0, 0, q=np.zeros(8))
    too_fast = SrsCandidate(0.20, 0.0, 0, score=(-1.0,), tag="fast")
    hold = SrsCandidate(0.0, 0.0, 0, score=(0.0,), tag="hold")
    config = SrsPlannerConfig(
        edge_samples=10,
        rail_v_max_m_s=0.08,
        rail_a_max_m_s2=0.30,
        arm_v_max_rad_s=10.0,
        arm_a_max_rad_s2=100.0,
        psi_v_max_rad_s=10.0,
        psi_a_max_rad_s2=100.0,
        consistent_plans_to_commit=1,
        min_dwell_s=0.0,
    )
    planner = SrsRailPosturePlanner(node_evaluator=_node, config=config)
    try:
        plan = planner.plan_now(
            SrsPlanRequest(
                current=current,
                manifold=(too_fast, hold),
                layer_dt_s=0.2,
            )
        )
        assert plan.target.tag == "hold"
    finally:
        planner.shutdown()


def test_three_consistent_plans_and_one_second_dwell_before_commit() -> None:
    clock = FakeClock()
    current = SrsCandidate(0.0, 0.0, 0, q=np.zeros(8), tag="current")
    better = SrsCandidate(0.01, 0.02, 0, score=(-1.0,), tag="better")
    planner = SrsRailPosturePlanner(
        node_evaluator=_node,
        clock=clock,
        config=_wide_config(
            consistent_plans_to_commit=3,
            min_dwell_s=1.0,
            material_improvement=0.1,
        ),
    )
    request = SrsPlanRequest(current=current, manifold=(better,))
    try:
        first = planner.plan_now(request)
        assert first.target.tag == "current" and first.proposal_count == 1
        clock.value = 0.5
        second = planner.plan_now(request)
        assert second.target.tag == "current" and second.proposal_count == 2
        clock.value = 1.0
        third = planner.plan_now(request)
        assert third.target.tag == "better"
        assert third.commit_changed
        assert third.proposal_count == 3
    finally:
        planner.shutdown()
