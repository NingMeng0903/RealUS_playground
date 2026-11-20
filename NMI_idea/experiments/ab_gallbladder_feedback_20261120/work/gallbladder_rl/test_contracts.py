"""Asset-free synthetic interface contract checks; never simulate physical feasibility."""
import importlib.util
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
RESULT = HERE / "results" / "interface_checks.json"
spec = importlib.util.spec_from_file_location("gallbladder_rl_env_contract", HERE / "env.py")
envmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(envmod)


class FakeMotion:
    def __init__(self, frames):
        self.frames = frames


class FakeAPI:
    MIN_HOLD_FRAMES = 12
    SensorExit = staticmethod(lambda state, history, accepted=None: SimpleNamespace(accepted=accepted))
    endpoint = staticmethod(lambda scene, point, tape, initial, deadline: {"correct": False, "wrong": False})

    @staticmethod
    def frame_count(plan):
        return plan.frames


class ContractTests(unittest.TestCase):
    def test_packet_features_whitelist_and_missing_mask(self):
        packet_a = SimpleNamespace(reading=np.arange(10).reshape(2, 5), marks=np.arange(6).reshape(3, 2),
                                   q=np.array([-200., -30., 0., 0.]), tick=12, qualified=False, genuine=False)
        packet_b = SimpleNamespace(reading=np.arange(10).reshape(2, 5), marks=np.arange(6).reshape(3, 2),
                                   q=np.array([-200., -30., 0., 0.]), tick=12, qualified=True, genuine=True)
        np.testing.assert_array_equal(envmod.packet_features(packet_a), envmod.packet_features(packet_b))
        packet = SimpleNamespace(reading=np.array([[np.nan, 1, np.inf, 3, 4], [5, 6, 7, 8, 9]]),
                                 marks=np.array([[np.nan, 1], [2, np.inf], [4, 5]]),
                                 q=np.array([-200., -30., 0., 0.]), tick=12)
        features = envmod.packet_features(packet)
        self.assertTrue(np.isfinite(features).all())
        # Reading masks occupy the 10 features immediately following normalized readings.
        np.testing.assert_array_equal(features[10:20], np.isfinite(packet.reading).ravel().astype(np.float32))
        np.testing.assert_array_equal(features[26:32], np.isfinite(packet.marks).ravel().astype(np.float32))
        self.assertEqual(features.shape, (envmod.PACKET_DIM,))
        np.testing.assert_allclose(features[32:36], 2.*(packet.q-envmod.CONTACT[0])/(envmod.CONTACT[1]-envmod.CONTACT[0])-1.)
        self.assertAlmostEqual(float(features[36]), packet.tick/640.)

    def test_public_encoder_schema_order_and_posterior_blocks(self):
        state = SimpleNamespace(q=np.array([-200., -30., 0., 0.]), v=np.array([1., 2., 3., 4.]),
                                tick=4, liver=np.ones(125))
        packets = [SimpleNamespace(reading=np.full((2, 5), i, dtype=float), marks=np.full((3, 2), i, dtype=float),
                                   q=np.array([-200.+i, -30., 0., 0.]), tick=i*3) for i in range(10)]
        obs = envmod.encode_public(state, packets, np.ones(64), deadline=300)
        self.assertEqual(obs.shape, (envmod.OBSERVATION_DIM,))
        self.assertTrue(np.isfinite(obs).all())
        # Eight packet feature blocks begin after q(4), v(4), remaining/tick(2).
        packet_block = obs[10:10+8*envmod.PACKET_DIM].reshape(8, envmod.PACKET_DIM)
        expected = np.stack([envmod.packet_features(p) for p in packets[-8:]])
        np.testing.assert_array_equal(packet_block, expected)
        # The public gallbladder and liver posterior blocks are 64 and 125 values.
        gall_start = 10 + envmod.HISTORY_FRAMES*envmod.PACKET_DIM
        liver_start = gall_start + 64
        self.assertEqual(obs[gall_start:liver_start].shape, (64,))
        self.assertEqual(obs[liver_start:liver_start+125].shape, (125,))
        np.testing.assert_allclose(obs[gall_start:liver_start], np.full(64, 1/64))
        np.testing.assert_allclose(obs[liver_start:liver_start+125], np.full(125, 1/125))
        # Each history frame keeps its own q/tick aligned with its readings.
        np.testing.assert_allclose(packet_block[:, 36], np.arange(2, 10)*3/640.)
        np.testing.assert_allclose(packet_block[:, 32], 2.*((-200.+np.arange(2, 10))-envmod.CONTACT[0, 0])/(envmod.CONTACT[1, 0]-envmod.CONTACT[0, 0])-1.)

    def make_env(self, tick, plan_result=None, safe_frames=0):
        obj = envmod.GallbladderRLEnv.__new__(envmod.GallbladderRLEnv)
        obj._done = False
        obj._api = FakeAPI()
        obj.state = SimpleNamespace(q=np.zeros(4), v=np.zeros(4), tick=tick, travel=0.)
        obj.deadline = tick + 13
        obj.decision_frames = 1
        obj.safe_suffix = FakeMotion(safe_frames)
        obj.time_cost = obj.travel_cost = 0.
        obj.wrong_cost = .8
        obj.last_delta = np.zeros(4)
        obj.last_mode = 0
        obj.last_rejected = False
        obj._policy_steps = 0
        obj._accepted = None
        obj._ever = obj._ever_before_reset = obj._strict_at_cine = False
        obj.initial = None
        obj.curriculum = "prefix"
        obj.decisions = []
        obj.scene = obj.ctrl = obj.tape = None
        obj.history = []
        obj._plan_calls = []
        obj._executed = []
        obj._finish_calls = 0
        obj._confirm_calls = 0
        obj._plan_result = plan_result
        obj._plan = lambda target: (obj._plan_calls.append(np.asarray(target).copy()) or obj._plan_result)
        def execute(plan, frames, phase):
            obj._executed.append((plan, frames, phase))
            obj.state.tick += frames
        obj._execute = execute
        obj._hold = lambda q, frames=1: FakeMotion(max(frames, 1))
        obj._finish_safely = lambda: setattr(obj, "_finish_calls", obj._finish_calls + 1)
        def confirm():
            obj._confirm_calls += 1
            obj.state.tick += 1
            obj._accepted = object()
        obj._settle_and_confirm = confirm
        obj._observation = lambda: np.zeros(envmod.OBSERVATION_DIM, dtype=np.float32)
        obj._info = lambda **extra: extra
        return obj

    def test_empty_observation_does_not_block_move_plan_or_execution(self):
        plan = FakeMotion(3)
        obj = self.make_env(tick=0, plan_result=plan)
        obj.safe_suffix = FakeMotion(0)
        obj._plan = lambda target: (obj._plan_calls.append(np.asarray(target).copy()) or plan)
        # No packets/readings are consulted by step before issuing a valid movement action.
        result = obj.step({"mode": 0, "delta": np.array([.25, 0., 0., 0.])})
        self.assertEqual(len(obj._plan_calls), 1)
        self.assertEqual(len(obj._executed), 1)
        self.assertEqual(obj._executed[0][2], "rl_feedback_move")
        self.assertFalse(result[2])

    def test_exactly_twelve_frames_remain_without_early_timeout_and_confirm_runs(self):
        # First land at exactly the confirmation reserve: the strict '>' timeout
        # condition must not drain the budget before a subsequent confirm action.
        obj = self.make_env(tick=0)
        obj.deadline = 13
        obj.safe_suffix = FakeMotion(0)
        obj._plan_result = None
        _, _, done, _ = obj.step({"mode": 0, "delta": np.zeros(4)})
        self.assertFalse(done)
        self.assertEqual(obj.deadline-obj.state.tick, 12)
        self.assertEqual(obj._finish_calls, 0)
        _, _, done, _ = obj.step({"mode": 1, "delta": np.zeros(4)})
        self.assertTrue(done)
        self.assertEqual(obj._confirm_calls, 1)

    def test_infeasible_mode_zero_preserves_saved_suffix(self):
        suffix = FakeMotion(4)
        obj = self.make_env(tick=0, plan_result=None, safe_frames=4)
        obj.safe_suffix = suffix
        obj.decision_frames = 2
        _, _, _, info = obj.step({"mode": 0, "delta": np.array([.5, 0., 0., 0.])})
        self.assertTrue(obj.last_rejected)
        self.assertEqual(obj._executed[0][0], suffix)
        self.assertEqual(obj._executed[0][1], 2)
        self.assertTrue(info["rejected"])

    def test_model_confirms_arrived_window_then_replans_after_failed_cine(self):
        obj = self.make_env(tick=50)
        del obj._settle_and_confirm  # Exercise the actual cache invalidation.
        obj.state.q = np.array([-170., -45., 0., 0.])
        obj.deadline = 100
        obj.freeze_belief = False
        obj.stations = []
        obj._model_target = obj.state.q.copy()
        obj._model_cache_valid = True
        obj._model_tick = 40
        obj._model_fit = None
        fitted = object()
        station = {"kind": "continue"}
        next_target = obj.state.q + np.array([6., 0., 0., 0.])
        obj.scene = SimpleNamespace(estimate=lambda state: (None, None, 20.),
                                    body=SimpleNamespace(anatomy=lambda fit: fit),
                                    assimilate=lambda state, packet: state)
        obj._api.anchors = lambda *args: (None, None)
        obj._api.fit_anatomy = lambda *args: (fitted, None, 0)
        obj._api.SensorExit = lambda state, history: SimpleNamespace(state=state, history=history)

        def fail_cine(scene, controller, tape, point, anatomy, deadline):
            state = SimpleNamespace(**vars(point.state))
            state.tick += obj._api.MIN_HOLD_FRAMES
            return SimpleNamespace(state=state, history=point.history, station=station, accepted=None)

        def propose(point, modes, previous_fit, stations):
            self.assertIs(previous_fit, fitted)
            self.assertEqual(stations, [station])
            return (next_target, None, None), fitted, None, {}

        obj._api.station_cine = fail_cine
        obj.scanner = SimpleNamespace(propose=propose)
        confirm = obj.model_action()
        self.assertEqual(confirm["mode"], 1)
        self.assertFalse(obj.step(confirm)[2])
        move = obj.model_action()
        self.assertEqual(move["mode"], 0)
        np.testing.assert_allclose(move["delta"], (next_target-obj.state.q)/envmod.ACTION_SCALE)

    def test_invalid_action_is_rejected(self):
        obj = self.make_env(tick=0, plan_result=FakeMotion(1))
        with self.assertRaises(ValueError):
            obj.step({"mode": 7, "delta": np.zeros(4)})
        with self.assertRaises(ValueError):
            obj.step({"mode": 0, "delta": np.array([2., 0., 0., 0.])})


def main():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests)
    result = unittest.TestResult()
    suite.run(result)
    report = {"audit_type": "synthetic_contract_check",
              "scope": "Asset-free unittest with SimpleNamespace and stub API; no torch, no asset loading, no simulator execution, no claim of physical motion feasibility.",
              "tests_run": result.testsRun, "failures": [str(t) for t, _ in result.failures],
              "errors": [str(t) for t, _ in result.errors], "passed": result.wasSuccessful()}
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    final_result = runner.run(unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests))
    print(stream.getvalue(), end="")
    if not final_result.wasSuccessful():
        raise SystemExit(1)
    print(f"wrote {RESULT}; audit_type=synthetic_contract_check")


if __name__ == "__main__":
    main()
