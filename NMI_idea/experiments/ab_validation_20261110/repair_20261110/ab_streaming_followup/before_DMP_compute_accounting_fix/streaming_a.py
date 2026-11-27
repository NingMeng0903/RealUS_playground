"""L0 synthetic streaming A study. No patient-specific prescan or QP claim.

World truth is only passed to frame generation and independent audit. Policy and
response fitting consume public body context, commands, and observed frames.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import betaln, logsumexp, ndtr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/media/camp/EXT_DRIVE/Among_US")
from DMP_playground.dmp_honest.dmp import fit_discrete_dmp

CFG = json.loads((HERE / "PROTOCOL.json").read_text())
PROTO_SHA = hashlib.sha256((HERE / "PROTOCOL.json").read_bytes()).hexdigest()
WEIGHTS = np.asarray(CFG["template_prior"]["mode_weights"], float)
FOV = float(CFG["image_stream"]["fov_halfwidth_m"])
FRAME_OFFSETS = CFG["image_stream"]["hold_frame_offsets_s"]
ANGLE_LA = math.radians(10.0)
MODES = [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, 0)]
Y_GRID = (-0.010, 0.0, 0.010)
A_GRID = (-0.10, 0.0, 0.10)
POSES = [(0.0, 0.0)] + list(dict.fromkeys((lane * .022 + dy, direction * .17 + da)
    for lane, direction in MODES[:4] for dy in Y_GRID for da in A_GRID))
assert len(POSES) == 37
START = (0.0, 0.0)


def command_q(pose):
    return np.asarray([pose[0] / .020, pose[1] / .17], float)


def output_z(u, angle):
    return np.asarray([u / .020, angle / .17], float)


def source_noise_key(seed, world_id, ordinal, action_id, frame_id):
    return np.random.default_rng(np.random.SeedSequence(
        [seed, world_id, ordinal, action_id, frame_id]))


def curve_cov():
    ind = np.arange(5)
    return .003**2 * (.65 ** np.abs(ind[:, None] - ind[None, :]))


CURVE_CHOL = np.linalg.cholesky(curve_cov())


@dataclass(frozen=True)
class World:
    public_c: np.ndarray
    mode: int
    broad_lane: float
    broad_angle: float
    controls: np.ndarray
    registration: float
    response_E: np.ndarray
    length_scale: float
    depth: float


def sample_worlds(seed, n, shift=False, exact=False):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        mode = int(rng.choice(5, p=WEIGHTS))
        c = rng.uniform(-1.3 if shift else -1.0, 1.3 if shift else 1.0, 3)
        out.append(World(c, mode, float(rng.normal(0, .035)),
            float(rng.normal(0, .25)), CURVE_CHOL @ rng.normal(size=5),
            float(rng.normal(0, .002)),
            np.zeros((2, 2)) if exact else rng.normal(0, .08 if shift else .04, (2, 2)),
            float(rng.uniform(.7, 1.1)), float(.040 + rng.normal(0, .003))))
    return out


def hidden_z0(world):
    lane, angle = MODES[world.mode]
    base_y = (lane * .022) if world.mode < 4 else world.broad_lane
    base_a = (angle * .17) if world.mode < 4 else world.broad_angle
    y = base_y + world.registration + world.controls[2]
    slope = .2 * math.tan(base_a) + (world.controls[3] - world.controls[1]) / .2
    tangent = math.atan2(slope, .2)
    return output_z(y, tangent)


def public_prior_particles():
    rng = np.random.default_rng(931100)
    z, scale, depth, weights, response_e = [], [], [], [], []
    for mode in range(5):
        for _ in range(16):
            # Draws are generic public-prior samples, never held-out patients.
            controls = CURVE_CHOL @ rng.normal(size=5)
            lane, ang = MODES[mode]
            y = (lane * .022 if mode < 4 else rng.normal(0, .035))
            a = (ang * .17 if mode < 4 else rng.normal(0, .25))
            y += rng.normal(0, .002) + controls[2]
            a = math.atan2(.2 * math.tan(a) + (controls[3] - controls[1]) / .2, .2)
            z.append(output_z(y, a))
            scale.append(rng.uniform(.7, 1.1))
            depth.append(.040 + rng.normal(0, .003))
            weights.append(WEIGHTS[mode] / 16.0)
            response_e.append(rng.normal(0, .04, (2, 2)))
    return (np.asarray(z), np.asarray(scale), np.asarray(depth),
            np.asarray(weights), np.asarray(response_e))


PRIOR_Z, PRIOR_SCALE, PRIOR_DEPTH, PRIOR_W, PRIOR_E = public_prior_particles()


def response_j(c, E=None, exact=False):
    if exact:
        return np.eye(2)
    c = np.asarray(c)
    shift = .45 * np.array([[c[0], .5*c[1]], [-.35*c[1], c[2]]])
    return np.eye(2) + shift + (np.zeros((2, 2)) if E is None else E)


def feature(c, dq):
    dq = np.asarray(dq)
    return np.concatenate([dq] + [float(ci) * dq for ci in c])


def learned_j(c, W):
    basis = np.eye(2)
    return np.stack([basis[j] + feature(c, basis[j]) @ W for j in range(2)], axis=1)


def constant_j(W):
    basis = np.eye(2)
    return np.stack([basis[j] + basis[j] @ W for j in range(2)], axis=1)


def segment_length(u, angle, scale):
    return .024 * scale * np.exp(-np.square(u / .012) - np.square(angle / .32))


def truth_geometry(world, pose, exact=False):
    z = hidden_z0(world) - response_j(world.public_c, world.response_E, exact) @ command_q(pose)
    u, a = z[0] * .020, z[1] * .17
    length = float(segment_length(u, a, world.length_scale))
    return dict(u=float(u), angle=float(a), length=length, depth=world.depth,
                contact=abs(pose[0]) <= .030)


def normal_logpdf(x, mu, sd):
    return -.5 * np.square((x - mu) / sd) - math.log(sd * math.sqrt(2 * math.pi))


def beta_logpdf(x, aa, bb):
    x = min(max(float(x), 1e-12), 1-1e-12)
    return (aa-1)*math.log(x) + (bb-1)*math.log1p(-x) - float(betaln(aa, bb))


def render_frame(world, pose, seed, world_id, ordinal, action_id, frame_id, exact=False):
    rng = source_noise_key(seed, world_id, ordinal, action_id, frame_id)
    true = truth_geometry(world, pose, exact)
    target_available = bool(abs(true["u"]) <= FOV and true["length"] >= .008 and true["contact"])
    target = bool(rng.random() < (.98 if target_available else 0.0))
    false_detection = bool(not target_available and rng.random() < .01)
    visible = target or false_detection
    public_contact = bool(true["contact"] ^ (rng.random() < .01))
    public = dict(visible=visible, u=None, angle=None, length=None, depth=None,
                  identity=None, contact=public_contact)
    if target:
        public.update(u=float(true["u"] + rng.normal(0, .0006)),
            angle=float(true["angle"] + rng.normal(0, .025)),
            length=float(true["length"] + rng.normal(0, .001)),
            depth=float(true["depth"] + rng.normal(0, .001)),
            identity=float(rng.beta(20, 2)))
    elif false_detection:
        public.update(u=float(rng.uniform(-FOV, FOV)),
            angle=float(rng.uniform(-.4, .4)), length=float(rng.uniform(0, .010)),
            depth=float(rng.uniform(.02, .07)), identity=float(rng.beta(2, 20)))
    return public, dict(source="target" if target else "false" if false_detection else "missing", **true)


def public_frame_good(frame):
    return bool(frame["visible"] and frame["contact"] and frame["identity"] >= .75
        and frame["length"] >= .015 and abs(frame["u"]) <= .002
        and abs(frame["angle"]) <= ANGLE_LA)


def public_verifier(frames):
    return bool(len(frames) == 5 and len({f["evidence_id"] for f in frames}) == 5
        and frames[-1]["time_s"] - frames[0]["time_s"] >= .5 - 1e-10
        and all(public_frame_good(f) for f in frames))


def independent_audit(events):
    return bool(len(events) == 5 and all(
        priv["source"] == "target" and priv["contact"] and priv["length"] >= .015
        and abs(priv["u"]) <= .002 and abs(priv["angle"]) <= ANGLE_LA
        for _, priv in events))


class MotionBank:
    def __init__(self):
        self.cache = {}
        self.max_translation_endpoint_error = 0.0
        self.max_yaw_endpoint_error = 0.0
        self.max_translation_speed = 0.0
        self.max_yaw_speed = 0.0

    def reach(self, source_pose, action_id):
        dest = POSES[action_id]
        key = (tuple(source_pose), action_id)
        if key not in self.cache:
            ss = np.linspace(0, 1, 41)
            smooth = 3*ss**2 - 2*ss**3
            start = np.array([.1, source_pose[0], source_pose[1]])
            goal = np.array([.1, dest[0], dest[1]])
            demo = start[None] + smooth[:, None] * (goal-start)[None]
            dmp = fit_discrete_dmp(ss, demo, n_basis=30)
            dense, _ = dmp.rollout(np.linspace(0, 1, 161))
            path = dense[np.linspace(0, 160, 21).astype(int)]
            steps = np.diff(path, axis=0)
            trans = np.linalg.norm(steps[:, :2], axis=1)
            yaw = np.abs(steps[:, 2])
            duration = max(float(trans.max())*20/.02, float(yaw.max())*20/.5, .3)
            actual = (float(path[-1, 1]), float(path[-1, 2]))
            self.max_translation_endpoint_error = max(self.max_translation_endpoint_error,
                abs(actual[0]-dest[0]))
            self.max_yaw_endpoint_error = max(self.max_yaw_endpoint_error,
                abs(actual[1]-dest[1]))
            self.max_translation_speed = max(self.max_translation_speed, float(trans.max())*20/duration)
            self.max_yaw_speed = max(self.max_yaw_speed, float(yaw.max())*20/duration)
            self.cache[key] = (actual, duration)
        return self.cache[key]

    def prewarm(self):
        start = time.perf_counter()
        endpoints = [self.reach(START, a)[0] for a in range(len(POSES))]
        for source in endpoints:
            for action in range(len(POSES)):
                self.reach(source, action)
        return time.perf_counter() - start


def train_public_response(seed, n_worlds, exact=False):
    worlds = sample_worlds(seed, n_worlds, exact=exact)
    rng = np.random.default_rng(seed+700)
    motion_bank = MotionBank()
    x, y = [], []
    x_constant = []
    attempted = 0
    for wi, world in enumerate(worlds):
        for pair in range(16):
            branch = int(rng.integers(4))
            a0 = 1 + branch*9 + int(rng.integers(9))
            a1 = 1 + branch*9 + int(rng.integers(9))
            if a0 == a1:
                a1 = 1 + branch*9 + ((a1-1-branch*9+1) % 9)
            attempted += 1
            p0, _ = motion_bank.reach(START, a0)
            p1, _ = motion_bank.reach(p0, a1)
            f0, _ = render_frame(world, p0, seed+800, wi, pair, a0, 0, exact)
            f1, _ = render_frame(world, p1, seed+801, wi, pair, a1, 1, exact)
            if not (f0["visible"] and f1["visible"] and
                    f0["identity"] >= .75 and f1["identity"] >= .75):
                continue
            dq = command_q(p1) - command_q(p0)
            dz = output_z(f0["u"]-f1["u"], f0["angle"]-f1["angle"])
            x.append(feature(world.public_c, dq))
            x_constant.append(dq)
            y.append(dz-dq)
    X, Y = np.asarray(x), np.asarray(y)
    if len(X) < 50:
        raise RuntimeError(f"insufficient public response pairs: {len(X)}")
    W = np.linalg.solve(X.T @ X + .001*np.eye(X.shape[1]), X.T @ Y)
    Xc = np.asarray(x_constant)
    Wc = np.linalg.solve(Xc.T @ Xc + .001*np.eye(2), Xc.T @ Y)
    return dict(W=W.tolist(), W_constant=Wc.tolist(), public_pairs_used=len(X),
                public_pairs_attempted=attempted, train_seed=seed,
                train_worlds=n_worlds, exact=exact,
                max_DMP_translation_endpoint_error_m=motion_bank.max_translation_endpoint_error,
                max_DMP_yaw_endpoint_error_rad=motion_bank.max_yaw_endpoint_error)


def particle_geometry(pose, Jbase):
    q = command_q(pose)
    effective = Jbase[None, :, :] + PRIOR_E
    z = PRIOR_Z - np.einsum("nij,j->ni", effective, q)
    u, angle = z[:, 0] * .020, z[:, 1] * .17
    length = segment_length(u, angle, PRIOR_SCALE)
    contact = abs(pose[0]) <= .030
    valid = (np.abs(u) <= FOV) & (length >= .008) & contact
    return u, angle, length, valid


class Belief:
    def __init__(self):
        self.weights = PRIOR_W.copy()
        self.evidence_ids = set()
        self.mismatch_count = 0

    def update(self, frame, pose, Jbase):
        evidence_id = frame["evidence_id"]
        if evidence_id in self.evidence_ids:
            raise ValueError(f"duplicate evidence: {evidence_id}")
        self.evidence_ids.add(evidence_id)
        u, angle, length, valid = particle_geometry(pose, Jbase)
        if not frame["visible"]:
            ll = np.log(np.where(valid, .02, .99))
        else:
            log_target = (math.log(.98)
                + normal_logpdf(frame["u"], u, .0006)
                + normal_logpdf(frame["angle"], angle, .025)
                + normal_logpdf(frame["length"], length, .001)
                + normal_logpdf(frame["depth"], PRIOR_DEPTH, .001)
                + beta_logpdf(frame["identity"], 20, 2))
            false_support = (abs(frame["u"]) <= FOV and abs(frame["angle"]) <= .4
                and 0 <= frame["length"] <= .010 and .02 <= frame["depth"] <= .07)
            log_false = (math.log(.01) - math.log(2*FOV) - math.log(.8)
                - math.log(.010) - math.log(.05)
                + beta_logpdf(frame["identity"], 2, 20)) if false_support else -1e8
            ll = np.where(valid, log_target, log_false)
        logw = np.log(np.maximum(self.weights, 1e-300)) + ll
        if not np.isfinite(logw).any() or np.max(logw) <= -1e7:
            self.mismatch_count += 1
            return
        self.weights = np.exp(logw - logsumexp(logw))

    def effective_sample_size(self):
        return float(1.0 / np.sum(np.square(self.weights)))


def response_for_method(method, c, model, online_j):
    if method == "analytic_geometry_A":
        return np.eye(2)
    if method == "constant_jacobian_A":
        return constant_j(np.asarray(model["W_constant"]))
    if method == "trained_residual_A":
        return learned_j(c, np.asarray(model["W"]))
    if method == "online_broyden_A":
        return online_j
    raise ValueError(method)


def rank_actions(belief, c, current_pose, visited, Jbase, seen):
    scores = np.full(len(POSES), -math.inf)
    for i, pose in enumerate(POSES):
        if i in visited:
            continue
        u, a, length, valid = particle_geometry(pose, Jbase)
        geo = np.exp(-.5*np.square(u/.003) - .5*np.square(a/.15))
        long_segment = ndtr((length-.015)/.0015)
        p_la = float(np.sum(belief.weights * geo * long_segment * valid))
        p_detect = float(np.sum(belief.weights * valid))
        nominal_motion = max(abs(pose[0]-current_pose[0])/.02,
                             abs(pose[1]-current_pose[1])/.5, .3) + .5
        scores[i] = p_la + (.02 if seen else .08)*p_detect - .002*nominal_motion
    return scores


def public_targetlike_block(frames):
    candidates = [f for f in frames if f["visible"] and f["identity"] >= .75]
    if not candidates:
        return None
    return np.mean([output_z(f["u"], f["angle"]) for f in candidates], axis=0)


def broyden_update(Jold, prev_z, now_z, prev_pose, now_pose):
    dq = command_q(now_pose) - command_q(prev_pose)
    if float(dq @ dq) < .02:
        return Jold
    measured_response = prev_z - now_z
    return Jold + np.outer(measured_response - Jold@dq, dq) / (float(dq@dq)+.01)


def run_episode(world, method, model, bank, cohort, world_id, seed, exact=False):
    # The method-specific choice function below sees public_c, belief, and pose only.
    public_c = world.public_c.copy()
    belief = Belief()
    current_pose = START
    online_j = np.eye(2)
    prev_targetlike = None
    prev_targetlike_pose = None
    visited = set()
    trace = []
    clock = 0.0
    motion_s = 0.0
    compute_s = 0.0
    first_public_detection = None
    first_independent_LA = None
    public_stop = False
    correct_public_stop = False
    false_public_stop = False
    for ordinal in range(6):
        start_compute = time.perf_counter()
        Jbase = response_for_method(method, public_c, model, online_j)
        scores = rank_actions(belief, public_c, current_pose, visited, Jbase,
                              prev_targetlike is not None)
        action_id = int(np.argmax(scores))
        planning_s = time.perf_counter() - start_compute
        compute_s += planning_s
        clock += planning_s
        if not math.isfinite(scores[action_id]):
            break
        actual_pose, movement = bank.reach(current_pose, action_id)
        if clock + movement + .5 > 60:
            break
        visited.add(action_id)
        clock += movement
        motion_s += movement
        frames = []
        events = []
        processing_s = 0.0
        for frame_id, offset in enumerate(FRAME_OFFSETS):
            public, private = render_frame(world, actual_pose, seed, world_id,
                                           ordinal, action_id, frame_id, exact)
            public["evidence_id"] = f"{cohort}/{world_id}/{ordinal}/{frame_id}"
            public["time_s"] = float(clock + offset)
            before = time.perf_counter()
            belief.update(public, actual_pose, Jbase)
            update_s = time.perf_counter()-before
            compute_s += update_s
            processing_s += update_s
            frames.append(public)
            events.append((public, private))
            if first_public_detection is None and public["visible"] and public["identity"] >= .75:
                first_public_detection = float(clock+offset)
        clock += .5 + processing_s
        block_public_stop = public_verifier(frames)
        block_audit_LA = independent_audit(events)
        if block_audit_LA and first_independent_LA is None:
            first_independent_LA = clock
        current_targetlike = public_targetlike_block(frames)
        if (method == "online_broyden_A" and prev_targetlike is not None
                and current_targetlike is not None):
            before = time.perf_counter()
            online_j = broyden_update(online_j, prev_targetlike, current_targetlike,
                                      prev_targetlike_pose, actual_pose)
            update_s = time.perf_counter()-before
            compute_s += update_s
            clock += update_s
        if current_targetlike is not None:
            prev_targetlike = current_targetlike
            prev_targetlike_pose = actual_pose
        trace.append(dict(action_id=action_id, commanded_pose=POSES[action_id],
            actual_DMP_endpoint=actual_pose, movement_s=movement, planning_s=planning_s,
            public_frames=frames, public_verifier=block_public_stop,
            independent_audit_only=block_audit_LA,
            private_sources_for_audit_only=[x[1]["source"] for x in events],
            posterior_ESS=belief.effective_sample_size(),
            posterior_model_mismatch_count=belief.mismatch_count,
            max_candidate_score=float(np.max(scores)), clock_s=clock))
        current_pose = actual_pose
        if block_public_stop:
            public_stop = True
            correct_public_stop = block_audit_LA
            false_public_stop = not block_audit_LA
            break
    return dict(cohort=cohort, world_id=world_id, method=method,
        success_ever_independent_LA=first_independent_LA is not None,
        first_independent_LA_s=first_independent_LA,
        RMT_60_s=first_independent_LA if first_independent_LA is not None else 60.0,
        first_public_detection_s=first_public_detection,
        public_stop=public_stop, correct_public_stop=correct_public_stop,
        false_public_stop=false_public_stop, no_find=first_independent_LA is None,
        actions=len(trace), frames=5*len(trace), motion_s=motion_s,
        compute_s=compute_s, elapsed_s=clock, trace=trace)


METHODS = tuple(CFG["policy"]["methods"])


def response_diagnostics(worlds, model, exact=False):
    rng = np.random.default_rng(931109 if exact else 931108)
    squared = {name: [] for name in ("analytic_geometry_A", "constant_jacobian_A",
                                      "trained_residual_A")}
    rank_change = {name: 0 for name in squared if name != "analytic_geometry_A"}
    score_max_change = {name: 0.0 for name in rank_change}
    for world in worlds:
        q0 = command_q(POSES[int(rng.integers(37))])
        q1 = command_q(POSES[int(rng.integers(37))])
        dq = q1-q0
        truth = response_j(world.public_c, world.response_E, exact) @ dq
        for method in squared:
            J = response_for_method(method, world.public_c, model, np.eye(2))
            squared[method].append(float(np.sum(np.square(J@dq-truth))))
        b = Belief()
        JA = np.eye(2)
        score_a = rank_actions(b, world.public_c, START, set(), JA, False)
        for method in rank_change:
            J = response_for_method(method, world.public_c, model, JA)
            score = rank_actions(b, world.public_c, START, set(), J, False)
            rank_change[method] += int(np.argmax(score) != np.argmax(score_a))
            score_max_change[method] = max(score_max_change[method],
                                          float(np.max(np.abs(score-score_a))))
    return dict(response_rmse={k:float(np.sqrt(np.mean(v)/2)) for k,v in squared.items()},
                root_action_changed_from_analytic=rank_change,
                max_root_candidate_score_difference=score_max_change)


def cohort_summary(rows):
    out = {}
    for method in METHODS:
        rs = [r for r in rows if r["method"] == method]
        n = len(rs)
        out[method] = dict(n=n,
            ever_independent_LA=sum(r["success_ever_independent_LA"] for r in rs),
            public_stops=sum(r["public_stop"] for r in rs),
            correct_public_stops=sum(r["correct_public_stop"] for r in rs),
            false_public_stops=sum(r["false_public_stop"] for r in rs),
            no_find=sum(r["no_find"] for r in rs),
            mean_RMT_60_s=float(np.mean([r["RMT_60_s"] for r in rs])),
            mean_first_public_detection_censored_60_s=float(np.mean([
                r["first_public_detection_s"] if r["first_public_detection_s"] is not None
                else 60.0 for r in rs])),
            mean_actions=float(np.mean([r["actions"] for r in rs])),
            mean_frames=float(np.mean([r["frames"] for r in rs])),
            mean_motion_s=float(np.mean([r["motion_s"] for r in rs])),
            mean_compute_s=float(np.mean([r["compute_s"] for r in rs])),
            mean_elapsed_s=float(np.mean([r["elapsed_s"] for r in rs])),
            mean_last_posterior_ESS=float(np.mean([
                r["trace"][-1]["posterior_ESS"] if r["trace"] else 80.0 for r in rs])),
            worlds_with_posterior_model_mismatch=sum(bool(r["trace"] and
                r["trace"][-1]["posterior_model_mismatch_count"]) for r in rs))
    return out


def pair_bootstrap(rows, first, second, metric, seed=931106):
    a = {r["world_id"]:r for r in rows if r["method"] == first}
    b = {r["world_id"]:r for r in rows if r["method"] == second}
    ids = sorted(set(a) & set(b))
    x = np.asarray([float(a[i][metric])-float(b[i][metric]) for i in ids])
    rng = np.random.default_rng(seed)
    boot = x[rng.integers(0, len(x), (2000, len(x)))].mean(1)
    return dict(n=len(x), mean=float(x.mean()), ci95=np.quantile(boot, [.025,.975]).tolist(),
                positive=int(np.sum(x>0)), zero=int(np.sum(x==0)), negative=int(np.sum(x<0)))


def evaluate_cohort(cohort, seed, n, model, shift=False, exact=False):
    worlds = sample_worlds(seed, n, shift=shift, exact=exact)
    bank = MotionBank()
    shared_bank_setup = bank.prewarm()
    rows = []
    for wi, world in enumerate(worlds):
        for method in METHODS:
            rows.append(run_episode(world, method, model, bank, cohort, wi, seed+5000, exact))
    comparisons = {}
    for other in ("analytic_geometry_A", "constant_jacobian_A", "online_broyden_A"):
        for metric in ("success_ever_independent_LA", "correct_public_stop", "false_public_stop", "RMT_60_s"):
            comparisons[f"trained_minus_{other}:{metric}"] = pair_bootstrap(
                rows, "trained_residual_A", other, metric)
    return dict(cohort=cohort, n=n, shared_generic_DMP_bank_setup_s=shared_bank_setup,
        DMP_bank_cache_entries=len(bank.cache),
        max_DMP_translation_endpoint_error_m=bank.max_translation_endpoint_error,
        max_DMP_yaw_endpoint_error_rad=bank.max_yaw_endpoint_error,
        max_DMP_translation_speed_m_s=bank.max_translation_speed,
        max_DMP_yaw_speed_rad_s=bank.max_yaw_speed,
        response_diagnostics=response_diagnostics(worlds, model, exact),
        summary=cohort_summary(rows), paired_comparisons=comparisons, rows=rows)


def model_file():
    return HERE / "MODEL.json"


def save_models():
    started = time.perf_counter()
    positive = train_public_response(931101, 512, False)
    exact = train_public_response(931107, 512, True)
    data = dict(protocol_sha256=PROTO_SHA,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_split="independent public source-world before/after frames, no hidden J labels",
        positive=positive, G_exact=exact,
        training_wall_s=time.perf_counter()-started)
    model_file().write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    return data


def load_models():
    data = json.loads(model_file().read_text())
    if data["protocol_sha256"] != PROTO_SHA:
        raise RuntimeError("model trained under a different protocol")
    if data["source_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise RuntimeError("source changed since model training; retrain before evaluation")
    return data


def run_phase(phase):
    if phase == "train":
        data = save_models()
        print(json.dumps(dict(protocol_sha256=PROTO_SHA,
            positive_public_pairs=data["positive"]["public_pairs_used"],
            exact_public_pairs=data["G_exact"]["public_pairs_used"],
            training_wall_s=data["training_wall_s"]), indent=2))
        return
    model_data = load_models()
    if phase == "dev":
        specs = [("validation", 931102, 128, False, False, model_data["positive"])]
    elif phase == "test":
        approved = json.loads((HERE / "PARENT_APPROVAL.json").read_text())
        if approved["approved_protocol_sha256"] != PROTO_SHA:
            raise RuntimeError("parent approval does not match frozen protocol")
        specs = [("test", 931103, 256, False, False, model_data["positive"]),
                 ("shift", 931104, 128, True, False, model_data["positive"]),
                 ("G_exact_negative", 931105, 128, False, True, model_data["G_exact"])]
    else:
        raise ValueError(phase)
    out = dict(scope=CFG["scope"], protocol_sha256=PROTO_SHA,
        source_sha256=model_data["source_sha256"],
        model_sha256=hashlib.sha256(model_file().read_bytes()).hexdigest(),
        no_real_B_mode_QP_dIRD=True, cohorts={})
    for name, seed, n, shift, exact, model in specs:
        out["cohorts"][name] = evaluate_cohort(name, seed, n, model, shift, exact)
        print(json.dumps(dict(cohort=name,
            response=out["cohorts"][name]["response_diagnostics"],
            summary=out["cohorts"][name]["summary"]), indent=2))
    output = HERE / ("dev_results.json" if phase == "dev" else "results.json")
    output.write_text(json.dumps(out, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("train", "dev", "test"))
    run_phase(ap.parse_args().phase)
