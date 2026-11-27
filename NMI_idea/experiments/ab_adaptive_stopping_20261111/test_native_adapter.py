"""Contract tests for the adaptive native adapter.

The deterministic four-event smoke is opt-in because one native replay can
take several seconds on the shared fixture.  Run it with
``RUN_NATIVE_ADAPTER_SMOKE=1`` using the experiment's native Python binary.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from fixture import Capture, SINGLE_DWELLS, single_dwell_action
from native_adapter import ACTION_IDS, NativeAdapter, PredictedEpisodeState


def test_action_catalog_is_interruptible_and_terminal_dwell_is_fixed():
    probes = [action for action in ACTION_IDS if "single_dwell" in action]
    assert len(probes) == 12
    assert {Capture(action, "side", (0.0, 0.0, 0.0), "left").dwell_s for action in probes if action.startswith("side_")} == set(SINGLE_DWELLS)
    assert {Capture(action, "fork", (0.0, 0.0, 0.0), "fork").dwell_s for action in probes if action.startswith("fork_")} == set(SINGLE_DWELLS)
    assert Capture("left", "terminal", (0.0, 0.0, 0.0), "left").dwell_s == 1.0
    assert single_dwell_action("fork_minus", 0.60) == "fork_minus__single_dwell_0.6"


def test_adapter_imports_without_a_deadline_symbol():
    # This is intentionally a source-level contract: the adaptive API does
    # not receive or consult a task deadline parameter.
    source = Path(__file__).with_name("native_adapter.py").read_text(encoding="utf-8")
    assert "deadline_s" not in source.lower()


def test_public_motion_proposal_is_unknown_stateful_and_cached(tmp_path):
    """Proposal uses route/dIRD only and advances a separate predicted state."""
    with NativeAdapter(0, "open", tmp_path / "proposal", backend="python") as adapter:
        root = adapter.start
        first = adapter.motion.propose(root, "fork_minus__single_dwell_0.15")
        assert first.available
        assert first.status == "predicted_unknown"
        assert first.native_checked is False
        assert first.success_probability is None
        assert first.dird_query_increment == 1
        assert first.routes and first.routes[0].prediction_status == "predicted_unknown"
        assert first.margin is not None and np.isfinite(first.margin)
        assert isinstance(first.predicted_output, PredictedEpisodeState)
        assert first.predicted_output.native is None
        assert first.predicted_output.q_proxy.shape == (8,)
        assert first.predicted_output.elapsed > root.elapsed
        assert adapter.verification_calls == 0
        assert adapter.execution_calls == 0
        assert adapter._planning.rows == []

        cached = adapter.motion.propose(root, "fork_minus__single_dwell_0.15")
        assert cached.cache_hit
        assert cached.dird_query_increment == 0
        assert adapter.proposal_cache_hits == 1
        assert adapter.proposal_dird_queries == 1

        # The predicted exit is not a nominal reset: its pose/clock feeds the
        # next public route query, while qdot/history remain explicitly absent.
        root_left = adapter.motion.propose(root, "left")
        after_left = adapter.motion.propose(first.predicted_output, "left")
        assert after_left.available
        assert after_left.predicted_output.origin_state_fingerprint == first.predicted_output.origin_state_fingerprint
        assert after_left.duration_s != root_left.duration_s
        assert after_left.predicted_output.elapsed > first.predicted_output.elapsed
        with pytest.raises(TypeError):
            adapter.checked(first.predicted_output, "left")



@pytest.mark.skipif(os.environ.get("RUN_NATIVE_ADAPTER_SMOKE") != "1", reason="opt-in native replay smoke")
def test_native_closed_loop_smoke(tmp_path):
    """fork(.15) -> fork(other side,.60) -> terminal -> terminal.

    The assertions use entry q/qdot and the complete replay snapshot rather
    than a nominal reset, and check the same-tick tracking fields from every
    selected native segment.
    """
    with NativeAdapter(0, "open", tmp_path / "smoke", backend="native") as adapter:
        state = adapter.start
        events = []
        for action_id in (
            "fork_minus__single_dwell_0.15",
            "fork_plus__single_dwell_0.6",
            "left",
            "right",
        ):
            proposal = adapter.motion.propose(state, action_id)
            assert proposal.available and proposal.status == "predicted_unknown"
            assert proposal.native_checked is False
            event = adapter.execute(state, action_id)
            assert event.action_id == action_id
            assert event.feasible, event.result
            assert event.actual_output["reference_kind"] == "ProDMPIntegralBasisReference"
            assert event.actual_output["same_tick_position_error_max_m"] <= 0.002
            assert event.actual_output["same_tick_rotation_error_max_rad"] <= np.deg2rad(2.0)
            if events:
                np.testing.assert_allclose(event.actual_output["entry_q"], events[-1].actual_output["q"], atol=1e-8)
                np.testing.assert_allclose(event.actual_output["entry_qdot"], events[-1].actual_output["qdot"], atol=1e-8)
                assert event.actual_output["entry_snapshot"] is not None
            events.append(event)
            state = event.output
        adapter.finish()
        manifest = (tmp_path / "smoke" / "ADAPTER.json").read_text(encoding="utf-8")
        assert '"no_deadline": true' in manifest
        assert (tmp_path / "smoke" / "execution").is_dir()


@pytest.mark.skipif(os.environ.get("RUN_NATIVE_ADAPTER_REUSE_SMOKE") != "1", reason="opt-in checkpoint smoke")
def test_complete_state_checkpoint_reuse(tmp_path):
    with NativeAdapter(0, "open", tmp_path / "reuse", backend="native", reuse_checked=True) as adapter:
        state = adapter.start
        checked = adapter.checked(state, "fork_minus__single_dwell_0.15")
        assert checked.result["full_state_checkpoint"]
        reused = adapter.execute(state, "fork_minus__single_dwell_0.15")
        assert reused.source == "planning_checkpoint"
        assert reused.call_kind == "execution_reuse"
        assert reused.actual_output["snapshot"] is not None
        assert any((tmp_path / "reuse" / "execution").glob("reuse_*.json"))
