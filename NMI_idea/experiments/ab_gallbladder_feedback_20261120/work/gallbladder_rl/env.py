"""Asset-backed, recurrent RL interface to the archived gallbladder simulator.

Actors receive simulated detector readings, not B-mode pixels. This adapter
does not manufacture missing anatomy assets or reuse archived outcome labels
as new rollouts. Every action is available during an empty observation.

Both model_action() and an RL actor use the same bounded chart increments,
motion admissibility checker, confirmation routine and physical deadline.
The default executor uses measured-state quintics, not ProDMP parameters.
The inherited anatomy posterior still assumes nominal ribs: no joint rib or
dropout posterior is claimed by this first implementation.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import os
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CONTACT = np.array([[-270., -83., -90., -35.], [-132., 12., 90., 35.]])
SPEED = np.array([48., 21., 49., 40.])
ACTION_SCALE = np.array([12., 5., 12., 8.])
READOUT_SCALE = np.array([[1000., 100., 200., 100., 1.],
                          [10000., 100., 200., 200., 1.]])
HISTORY_FRAMES = 8
PACKET_DIM = 37
OBSERVATION_DIM = 507
PRIVILEGED_DIM = 17
ASSET_NAMES = {'REALUS_GB_ANATOMY': 'registered_torso.npz',
               'REALUS_GB_SMPLX': 'SMPLX_MALE.npz',
               'REALUS_GB_ATLAS': 'thorax_atlas.npz'}


def asset_status():
    return {name: {'path': str(Path(os.environ.get(name, ROOT/'local_assets'/file)).expanduser().resolve()),
                   'available': Path(os.environ.get(name, ROOT/'local_assets'/file)).expanduser().is_file()}
            for name, file in ASSET_NAMES.items()}


def require_assets():
    missing = {name: item['path'] for name, item in asset_status().items() if not item['available']}
    if missing:
        raise FileNotFoundError('Original anatomy assets are required; no synthetic substitute is used. '
                                + json.dumps(missing))


def asset_fingerprints():
    """Record actual bytes, and whether they match the archived asset manifest."""
    output = asset_status()
    expected_path = ROOT/'reference_results/asset_inputs.json'
    expected = json.loads(expected_path.read_text()) if expected_path.is_file() else {}
    for name, item in output.items():
        item['sha256'] = None
        if item['available']:
            hasher = hashlib.sha256()
            with Path(item['path']).open('rb') as source:
                for chunk in iter(lambda: source.read(8*1024*1024), b''):
                    hasher.update(chunk)
            item['sha256'] = hasher.hexdigest()
        item['archived_sha256'] = expected.get(name, {}).get('sha256')
        item['matches_archive'] = (item['sha256'] == item['archived_sha256']) if item['available'] else None
    return output


def packet_features(packet):
    """Whitelist public fields; qualified, genuine, condition and seed are absent."""
    if packet is None:
        return np.zeros(PACKET_DIM, dtype=np.float32)
    reading = np.asarray(packet.reading, dtype=float).reshape(2, 5)
    marks = np.asarray(packet.marks, dtype=float).reshape(3, 2)
    rvalid, mvalid = np.isfinite(reading), np.isfinite(marks)
    values = np.where(rvalid, reading / READOUT_SCALE, 0.)
    points = np.where(mvalid, marks / [100., 200.], 0.)
    q = 2. * (np.asarray(packet.q) - CONTACT[0]) / (CONTACT[1]-CONTACT[0]) - 1.
    return np.r_[np.clip(values, -5., 5.).ravel(), rvalid.ravel(),
                 np.clip(points, -5., 5.).ravel(), mvalid.ravel(), q,
                 float(packet.tick)/640.].astype(np.float32)


def encode_public(state, history, gall_weights, deadline, previous_delta=None,
                  previous_mode=0, rejected=False, advice=None):
    """Encode only supplied public arrays, with explicit missing-value masks."""
    gall = np.asarray(gall_weights, dtype=float).reshape(-1)
    liver = np.asarray(state.liver, dtype=float).reshape(-1)
    if gall.shape != (64,) or liver.shape != (125,):
        raise ValueError('This schema expects the archived 64-organ / 125-liver public prior.')
    if not np.isfinite(gall).all() or not np.isfinite(liver).all() or gall.sum() <= 0 or liver.sum() <= 0:
        raise ValueError('Invalid public posterior.')
    packets = list(history)[-HISTORY_FRAMES:]
    image = [packet_features(None)] * (HISTORY_FRAMES - len(packets))
    image += [packet_features(p) for p in packets]
    q = 2. * (np.asarray(state.q) - CONTACT[0]) / (CONTACT[1]-CONTACT[0]) - 1.
    last = np.zeros(4) if previous_delta is None else np.asarray(previous_delta, dtype=float)
    advise = np.zeros(5) if advice is None else np.r_[np.asarray(advice, dtype=float), 1.]
    out = np.r_[q, np.asarray(state.v)/SPEED,
                max(0, deadline-state.tick)/260., state.tick/640.,
                np.concatenate(image), gall/gall.sum(), liver/liver.sum(),
                last, [previous_mode == 0, previous_mode == 1], float(rejected), advise]
    if out.shape != (OBSERVATION_DIM,) or not np.isfinite(out).all():
        raise ValueError(f'Invalid observation shape/values: {out.shape}')
    return out.astype(np.float32)


class GallbladderRLEnv:
    """Finite-budget online episodes starting at the SAME archived costal exit.

    reset_mode='near_success' is a training curriculum only. It physically
    executes a prefix and a witness-neighborhood approach; the witness is never
    passed to the actor. A missing witness falls back to the original prefix
    with an explicit reset diagnostic, rather than silently deleting the case.
    """
    observation_dim = OBSERVATION_DIM
    privileged_dim = PRIVILEGED_DIM

    def __init__(self, planner_features=False, freeze_belief=False, decision_frames=3,
                 reset_mode='prefix', wrong_cost=.8, time_cost=0., travel_cost=0.):
        if reset_mode not in ('prefix', 'near_success'):
            raise ValueError('reset_mode must be prefix or near_success')
        if decision_frames < 1:
            raise ValueError('decision_frames must be positive')
        self.planner_features = bool(planner_features)
        self.freeze_belief = bool(freeze_belief)
        self.decision_frames = int(decision_frames)
        self.reset_mode = reset_mode
        self.wrong_cost, self.time_cost, self.travel_cost = wrong_cost, time_cost, travel_cost
        self._api = None
        self._witness_cache = {}

    def _load(self):
        if self._api is not None:
            return
        require_assets()
        sys.path.insert(0, str(ROOT/'work/gallbladder_feedback'))
        from engine import Scanner, SensorExit, endpoint, strict_frame
        from acquire import HOLD_FRAMES, MIN_HOLD_FRAMES, fit_anatomy
        from motion_feedback import _quintic_from_state, advance_block, assert_admissible
        from view_geometry import FRAME_STEPS, Motion, frame_count, prior_run
        from engine import station_cine
        from policy import anchors, image_rank, center_preserving_target
        prior_run.initialize()
        self._api = SimpleNamespace(**locals())

    def reset(self, seed, condition='wide', stream=20000):
        self._load()
        if condition not in ('wide', 'stress'):
            raise ValueError('Only the archived wide/stress conditions are supported.')
        api = self._api
        self.scanner = scanner = api.Scanner(int(seed), condition, int(stream))
        self.scene, self.ctrl, self.tape = scanner.scene, scanner.ctrl, scanner.tape
        self.state = scanner.initial.state
        self.history = list(scanner.initial.history)
        self.deadline = scanner.deadline
        self.initial = scanner.prefix
        self.safe_suffix = self._hold(self.state.q, 0)
        self.curriculum = 'prefix'
        if self.reset_mode == 'near_success':
            self._near_success_reset(int(seed), int(stream))
        self.frozen_liver = np.asarray(self.state.liver).copy()
        self.frozen_evidence = np.asarray(self.state.evidence).copy()
        self.last_delta = np.zeros(4)
        self.last_mode = 0
        self.last_rejected = False
        self.stations, self.decisions = [], []
        self._model_target = None
        self._model_tick = -10000
        self._model_fit = None
        self._model_cache_valid = False
        self._done = False
        self._ever = False
        self._ever_before_reset = any(api.strict_frame(self.scene, p, self.tape) for p in self.history)
        self._strict_at_cine = False
        self._accepted = None
        # Replay actual public prefix with actual measured velocities, not the
        # terminal posterior repeated at earlier timestamps.
        warmup = self._warmup()
        self._reset_tick = int(self.state.tick)
        self._policy_steps = 0
        return self._observation(), self._info(warmup=warmup, reset_mode=self.curriculum)

    def _near_success_reset(self, seed, stream):
        from witnesses import find_witness
        from curriculum import reverse_checkpoints
        if seed not in self._witness_cache:
            _, target, _, _, _ = find_witness(self.scene, self.initial)
            self._witness_cache[seed] = target
        target = self._witness_cache[seed]
        if target is None:
            self.curriculum = 'prefix_no_witness_found'
            return
        desired = stream % 3
        chosen = None
        for k, item in enumerate(reverse_checkpoints(self.scene, self.ctrl, self.tape,
                self.initial, self.scanner.initial.history, target, variant=stream)):
            chosen = item
            if k == desired:
                break
        if chosen is None:
            self.curriculum = 'prefix_no_feasible_checkpoint'
            return
        self.curriculum, self.state, history, self.safe_suffix = chosen
        self.history = list(history)

    def _warmup(self):
        state = self.scene.at_start()
        speeds = {(int(row['end_tick']), tuple(np.round(row['q_end'], 9))): np.asarray(row['v_end'])
                  for row in self.ctrl._motion_trace}
        history, observations = [], []
        for packet in self.history:
            key = (int(packet.tick), tuple(np.round(packet.q, 9)))
            if key not in speeds:
                raise RuntimeError('Checkpoint history lacks its actual motion velocity.')
            state = replace(state, v=speeds[key])
            state = self.scene.assimilate(state, packet)
            history.append(packet)
            observations.append(encode_public(state, history, self.scene.gall_weights(state), self.deadline))
        if history and (not np.allclose(state.liver, self.state.liver)
                        or not np.allclose(state.evidence, self.state.evidence)):
            raise RuntimeError('Replayed prefix does not reproduce the public posterior.')
        # The current observation is processed when the first decision is made.
        return np.asarray(observations[:-1], dtype=np.float32).reshape(-1, self.observation_dim)

    def _hold(self, q, frames=1):
        api = self._api
        x = np.broadcast_to(q, (frames * api.FRAME_STEPS + 1, 4)).copy()
        return api.Motion(x, np.zeros_like(x))

    def _plan(self, target, reserve=True):
        api = self._api
        target = np.clip(np.asarray(target, dtype=float), *CONTACT)
        left = self.deadline - self.state.tick - (api.HOLD_FRAMES if reserve else 0)
        if target.shape != (4,) or not np.isfinite(target).all():
            raise ValueError('Invalid local target.')
        for frames in (4, 5, 7, 10, 15, 22, 33, 48, 72, 105):
            if frames > left:
                break
            plan = api._quintic_from_state(self.state.q, self.state.v, target, frames)
            if self.scene.motor.feasible(plan):
                return api.assert_admissible(self.scene, self.state, plan,
                       self.deadline-(api.HOLD_FRAMES if reserve else 0))
        return None

    def _execute(self, plan, frames, phase):
        api = self._api
        for frame in range(1, frames+1):
            self.state = api.advance_block(self.ctrl, self.tape, self.state, plan, frame,
                                           None, self.history, phase)
            if self.freeze_belief:
                self.state = replace(self.state, liver=self.frozen_liver.copy(),
                                     evidence=self.frozen_evidence.copy())
            # Diagnostic truth never alters the selected action or stopping.
            self._ever |= bool(api.strict_frame(self.scene, self.history[-1], self.tape))
        index = frames * api.FRAME_STEPS
        self.safe_suffix = api.Motion(plan.x[index:].copy(), plan.v[index:].copy())

    def _finish_safely(self):
        api = self._api
        count = api.frame_count(self.safe_suffix)
        if count:
            api.assert_admissible(self.scene, self.state, self.safe_suffix, self.deadline)
            self._execute(self.safe_suffix, count, 'rl_budget_brake')
        if np.max(np.abs(self.state.v)) >= 1.e-7:
            raise RuntimeError('Saved feasible suffix did not stop.')
        # Consume the remaining real budget; no truth-driven auto-confirmation.
        while self.state.tick < self.deadline:
            self._execute(self._hold(self.state.q), 1, 'rl_timeout_hold')

    def _settle_and_confirm(self):
        api = self._api
        if np.max(np.abs(self.state.v)) >= 1.e-7:
            accel = np.asarray(api.prior_run.ACCEL)
            stopping = self.state.v * np.abs(self.state.v)/(2.*accel)
            brake = None
            for factor in (1., 1.5, 2.):
                brake = self._plan(self.state.q + factor*stopping)
                if brake is not None:
                    break
            if brake is None:
                brake = self.safe_suffix
            count = api.frame_count(brake)
            if count:
                self._execute(brake, count, 'rl_requested_brake')
        if np.max(np.abs(self.state.v)) >= 1.e-7:
            raise RuntimeError('Confirmation must start from actual rest.')
        if self.deadline-self.state.tick < api.MIN_HOLD_FRAMES:
            self._finish_safely()
            return
        fitted, _, _ = api.fit_anatomy(self.scene, self.state, self.history, self._model_fit)
        self._model_fit = fitted
        anatomy = self.scene.body.anatomy(fitted)
        before = self.state.tick
        # Frozen-belief ablation must freeze every assimilation in the cine too.
        original_assimilate = self.scene.assimilate
        if self.freeze_belief:
            def assimilate_frozen(state, packet, **kwargs):
                out = original_assimilate(state, packet, **kwargs)
                return replace(out, liver=self.frozen_liver.copy(), evidence=self.frozen_evidence.copy())
            self.scene.assimilate = assimilate_frozen
        try:
            point = api.station_cine(self.scene, self.ctrl, self.tape,
                    api.SensorExit(self.state, tuple(self.history)), anatomy, self.deadline)
        finally:
            self.scene.assimilate = original_assimilate
        self.state, self.history = point.state, list(point.history)
        self.stations.append(point.station)
        for packet in self.history:
            if packet.tick > before:
                valid = bool(api.strict_frame(self.scene, packet, self.tape))
                self._strict_at_cine |= valid
                self._ever |= valid
        self._accepted = point.accepted
        self.safe_suffix = self._hold(self.state.q, 0)
        self._model_cache_valid = False
        self._model_target = None

    def _advice_target(self):
        if not self._model_cache_valid or self.state.tick-self._model_tick >= 12:
            point = self._api.SensorExit(self.state, tuple(self.history))
            proposed, self._model_fit, _, _ = self.scanner.propose(
                point, False, self._model_fit, self.stations)
            self._model_target = None if proposed is None else np.asarray(proposed[0]).copy()
            self._model_tick = self.state.tick
            self._model_cache_valid = True
        return self._model_target

    def _observation(self):
        advice = None
        if self.planner_features and not self._done:
            target = self._advice_target()
            if target is not None:
                advice = np.clip((target-self.state.q)/ACTION_SCALE, -1., 1.)
        return encode_public(self.state, self.history, self.scene.gall_weights(self.state), self.deadline,
                             self.last_delta, self.last_mode, self.last_rejected, advice)

    def _privileged(self):
        view = self.scene.body.view(self.state.q)
        truth = self.scene.truth
        marks = np.asarray([truth.landmarks[k] for k in ('neck', 'body', 'fundus')])
        value = np.r_[(truth.center-view.origin)/100., truth.axis,
                      ((marks-view.origin)/100.).ravel(), np.asarray(self.scene.case.rib)/20.]
        if value.shape != (self.privileged_dim,) or not np.isfinite(value).all():
            raise RuntimeError('Invalid privileged critic vector.')
        return value.astype(np.float32)

    def _info(self, **extra):
        # Caller must explicitly route privileged only to the training critic.
        return dict(privileged=self._privileged(), tick=int(self.state.tick),
                    remaining_frames=int(self.deadline-self.state.tick), **extra)

    def model_action(self):
        """A0: continuous-fit model direction plus public keep/center feedback.

        This is a new common-executor comparator, not the unchanged archived A.
        No claim of joint rib belief or exact belief-space planning is made.
        """
        _, _, length = self.scene.estimate(self.state)
        primary, _ = self._api.anchors(self.state, self.history, length)
        if primary is not None and self._api.image_rank(primary, length) >= 3.4:
            target = self._api.center_preserving_target(self.scene, primary)
            if np.linalg.norm((target-self.state.q)/ACTION_SCALE) < .15:
                return {'mode': 1, 'delta': np.zeros(4)}
        else:
            # Reaching a model-proposed station must trigger a physical attempt
            # at confirmation even if a strong landmark has not appeared yet.
            if (self._model_target is not None and
                    np.linalg.norm((self._model_target-self.state.q)/ACTION_SCALE) < .15):
                return {'mode': 1, 'delta': np.zeros(4)}
            target = self._advice_target()
            if target is None:
                # Evaluate public local alternatives even when readings are empty.
                offsets = np.vstack((np.eye(4), -np.eye(4))) * ACTION_SCALE
                poses = np.clip(self.state.q + offsets, *CONTACT)
                weights = self.scene.gall_weights(self.state)
                scores = []
                for q in poses:
                    _, support, quality = self.scene.project(q)
                    scores.append(float(weights @ (quality + .1*support.max(axis=1))))
                target = poses[int(np.argmax(scores))]
        delta = np.clip((target-self.state.q)/ACTION_SCALE, -1., 1.)
        if np.linalg.norm(delta) < .1:
            return {'mode': 1, 'delta': np.zeros(4)}
        return {'mode': 0, 'delta': delta}

    def step(self, action):
        if self._done:
            raise RuntimeError('reset is required after termination')
        api = self._api
        mode = int(action['mode'])
        delta = np.asarray(action.get('delta', np.zeros(4)), dtype=float)
        if mode not in (0, 1) or delta.shape != (4,) or not np.isfinite(delta).all():
            raise ValueError('Expected mode 0/1 and a finite four-vector delta.')
        if np.any(np.abs(delta) > 1.+1.e-6):
            raise ValueError('delta must already be normalized to [-1,1].')
        before_tick, before_travel = self.state.tick, self.state.travel
        self.last_delta, self.last_mode, self.last_rejected = delta.copy(), mode, False
        self._policy_steps += 1
        if mode == 1:
            self._settle_and_confirm()
        else:
            plan = self._plan(self.state.q + ACTION_SCALE*delta)
            if plan is None:
                self.last_rejected = True
                plan = self.safe_suffix
            count = api.frame_count(plan)
            if count:
                self._execute(plan, min(count, self.decision_frames), 'rl_feedback_move')
            elif np.max(np.abs(self.state.v)) < 1.e-7:
                self._execute(self._hold(self.state.q), 1, 'rl_rejected_hold')
            else:
                raise RuntimeError('No feasible fallback suffix from a moving state.')
        if self.state.tick > self.deadline-api.MIN_HOLD_FRAMES and self._accepted is None:
            self._finish_safely()
        self._done = self._accepted is not None or self.state.tick >= self.deadline
        seconds = (self.state.tick-before_tick)/10.
        travel = self.state.travel-before_travel
        reward = -self.time_cost*seconds - self.travel_cost*travel
        self.decisions.append({'tick': int(before_tick), 'mode': mode, 'delta': delta.tolist(),
                               'rejected': self.last_rejected, 'executed_frames': int(self.state.tick-before_tick)})
        extra = {'executed_frames': int(self.state.tick-before_tick), 'rejected': self.last_rejected}
        if self._done:
            point = api.SensorExit(self.state, tuple(self.history), self._accepted)
            outcome = api.endpoint(self.scene, point, self.tape, self.initial, self.deadline)
            outcome.update(timeout=self._accepted is None, ever_after_reset=self._ever,
                           ever=self._ever_before_reset or self._ever,
                           strict_at_cine=self._strict_at_cine, reset_kind=self.curriculum,
                           policy_steps=self._policy_steps, decisions=self.decisions)
            reward += float(outcome['correct']) - self.wrong_cost*float(outcome['wrong'])
            extra['outcome'] = outcome
        if self.state.tick <= before_tick or self.state.tick > self.deadline:
            raise RuntimeError('Action must consume positive time within the original deadline.')
        return self._observation(), float(reward), self._done, self._info(**extra)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true', help='Check original assets without importing the simulator.')
    args = parser.parse_args()
    print(json.dumps({'assets': asset_status(), 'ready': all(v['available'] for v in asset_status().values()),
                      'observation_dim': OBSERVATION_DIM, 'privileged_dim': PRIVILEGED_DIM,
                      'observation_kind': 'simulated_detector_readings_not_bmode',
                      'start': 'after_shared_costal_sweep', 'original_asset_replay_performed': False}, indent=2))
