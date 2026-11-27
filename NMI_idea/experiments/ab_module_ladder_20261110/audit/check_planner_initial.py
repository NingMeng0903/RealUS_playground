"""Read-only, development-only planner contract review.

The robot and clock substitutes below are unit-test fixtures, not fabricated
robot performance evidence. No held-out catalogue or native controller is run.
The initial reviewed source is loaded from the audit snapshot to preserve the
failure evidence while the source owner repairs the live planner.
"""
from pathlib import Path
import hashlib
import importlib.util
import inspect
import json
import sys
import types
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SNAP = HERE / 'planner_review_initial'
source = SNAP / 'system.py'
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('audit_initial_system', ROOT/'planning/system.py')
s = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = s
exec(compile(source.read_text(), str(source), 'exec'), s.__dict__)
s.CFG = json.loads((SNAP/'PROTOCOL.json').read_text())
env = s.env

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

result = dict(scope='audit unit contracts only; development seed; no heldout/native run',
              source_sha256=sha(source), protocol_sha256=sha(SNAP/'PROTOCOL.json'),
              environment_sha256=sha(ROOT/'observation/environment.py'), checks={})
checks = result['checks']

# Exact finite-posterior aggregation over all legal hypotheses, including five
# observations conditionally independent only GIVEN the persistent hypothesis.
particles = env.sample_prior(720501, 16)
w = np.arange(1,17,dtype=float); w /= w.sum()
poses = s.candidate_actions(np.zeros(3), [], particles, w, max_candidates=10)
pred = env.predict_batch(particles, poses)
ai = int(np.argmax(pred['p_visible'].mean(0)))
frames = env.sample_block_from_prediction(pred, 7, ai, ('audit-dev', 0))
post, evidence, ll = s.block_posterior(w, frames, pred, ai)
manual_ll = np.stack([env.log_likelihood_batch(f, particles, poses[ai]) for f in frames]).sum(0)
manual = np.exp(np.log(w) + manual_ll - logsumexp(np.log(w)+manual_ll))
np.testing.assert_allclose(post, manual, rtol=1e-11, atol=1e-13)
belief = env.Belief(particles, w, support='finite')
for f in frames:
    belief.update(f, poses[ai])
np.testing.assert_allclose(belief.weights, manual, rtol=1e-11, atol=1e-13)
before = belief.weights.copy()
belief.update(frames[0], poses[ai])
np.testing.assert_array_equal(before, belief.weights)
checks['all_latents_five_frame_bayes'] = dict(passed=True, n=16,
    max_abs_error=float(np.max(abs(post-manual))), evidence_log_density=float(evidence),
    history_count=len(belief.history), duplicate_does_not_change_weights=True)

# Posterior association: a public pass is not treated as a guaranteed true stop.
# For each public frame, sum target/decoy likelihoods before multiplying across
# frames. Only the numerator restricts all five sources to the target.
good = [dict(frame_id=f'forced-public-{j}', time_s=float(t), visible=True,
    contact=True, image_u_m=0., image_depth_m=.035, visible_length_m=.02,
    image_axis_rad=0., identity_score=.8) for j,t in enumerate(env.FRAME_OFFSETS)]
pc = s.correct_given_public(w, good, pred, ai)
_, _, geom = s.quality_arrays(pred)
lp_s = np.where(pred['p_source'][:,ai]>0,
    np.log(np.maximum(pred['p_source'][:,ai], 1e-300)), -np.inf)
den_terms, num_terms = [], []
for f in good:
    obs = np.asarray([f[field] for field in env.FIELDS])
    source_terms = lp_s + env._feature_logpdf(obs, pred['means'][:,ai])
    den_terms.append(logsumexp(source_terms,axis=-1))
    num_terms.append(source_terms[:,0])
den = logsumexp(np.log(w)+np.sum(den_terms,axis=0))
num = logsumexp(np.log(w)+np.where(geom[:,ai],np.sum(num_terms,axis=0),-np.inf))
manual_pc = float(np.exp(num-den))
np.testing.assert_allclose(pc, manual_pc, rtol=1e-10, atol=1e-13)
checks['public_stop_target_association'] = dict(passed=True,
    probability_true_given_public_stop=pc, max_abs_error=abs(pc-manual_pc),
    no_assumption_public_pass_is_true=True)

# Nontrivial association fixture: the two sources have identical public feature
# densities. Five source identities remain hidden; a public pass cannot reveal
# them. This is a probability-algebra test, not an invented performance case.
same_means=np.tile(np.array([0.,.035,.020,0.,.8]),(2,1,2,1))
ambiguous_pred=dict(means=same_means,p_source=np.array([[[.6,.2]],[[.2,.6]]]),
    p_visible=np.full((2,1),.8),contact=np.ones((2,1),bool),
    geometric_length=np.full((2,1,2),.02),angle_audit=np.zeros((2,1,2)))
ambiguous_w=np.array([.25,.75])
got=s.correct_given_public(ambiguous_w,good,ambiguous_pred,0)
expected=.25*(.6/.8)**5+.75*(.2/.8)**5
np.testing.assert_allclose(got,expected,rtol=1e-12,atol=1e-13)
checks['public_stop_target_association_ambiguous_fixture']=dict(passed=True,
    probability_true_given_public_stop=got,expected=expected,
    false_stop_loss_s=60*(1-got),fixture_not_robot_performance=True)

# Predictive quadrature is an approximation: a uniform 32-hypothesis prior is
# represented by eight deterministic indices, not all 32 worlds per backup.
# Only arithmetic is used here; the locked catalogue is not instantiated.
marks = (np.arange(8)+.5)/8
hs32 = np.searchsorted(np.cumsum(np.ones(32)/32), marks)
checks['predictive_quadrature_scope'] = dict(indices_for_uniform_32=hs32.tolist(),
    selected_count=len(np.unique(hs32)), latent_count=32,
    exact_full_bayes_posterior_is_not_exact_predictive_integration=True)

# Inputs demonstrably cannot convey the remaining episode budget to the planner.
checks['missing_budget_state'] = dict(
    decide_signature=str(inspect.signature(s.Planner.decide)),
    terminal_signature=str(inspect.signature(s.Planner.terminal)),
    remaining_action_argument=False, remaining_time_argument=False,
    status='REPRODUCED_SOURCE_CONTRACT_FAILURE')

class FixedPlanner:
    def __init__(self, *args, **kwargs):
        self.response = types.SimpleNamespace(geometry_queries=0)
    def decide(self, *args, **kwargs):
        return s.Decision(np.zeros(3), 0., {'kind':'unit_fixture'})

class FixedBelief:
    def __init__(self, particles, **kwargs):
        self.weights = np.ones(len(particles))/len(particles)
    def update(self, *args, **kwargs):
        pass

class FixedRobot:
    q_seed = np.zeros(8)
    vmax = np.ones(8)
    queries = 0
    def endpoint(self, *args, **kwargs):
        return dict(found=True, q_kin=np.zeros(8), proxy=0.)

def fake_render(world, pose, key, start_time=0.):
    fs=[dict(f,time_s=start_time+t) for f,t in zip(good,env.FRAME_OFFSETS)]
    audits=[dict(frame_id=f['frame_id'],time_s=f['time_s'],passed=True) for f in fs]
    return fs,audits

# Motion+.5s acquisition fits in the budget, but a 60s belief update is charged
# afterwards. The initial source reports success beyond the 60s deadline.
ticks = iter([0.,0.,0.,0.,0.,60.])
with patch.object(s,'Planner',FixedPlanner), patch.object(env,'Belief',FixedBelief), \
     patch.object(env,'render_block',fake_render), \
     patch.object(s.time,'perf_counter',lambda:next(ticks)):
    late = s.run_episode(object(),[object()],'A_myopic',FixedRobot(),('hidden',9381),max_actions=1)
assert late['correct_public_stop'] and late['rmt_s'] > 60
assert all('9381' not in f['frame_id'] for f in late['events'][0]['public'])
checks['deadline_overrun_success'] = dict(status='REPRODUCED_SOURCE_CONTRACT_FAILURE',
    correct_public_stop=late['correct_public_stop'], rmt_s=late['rmt_s'],
    elapsed_s=late['elapsed_s'], requested_deadline_s=60.,
    public_ids=[f['frame_id'] for f in late['events'][0]['public']])

# Explicit fixture isolates the ablation's declared scope. Removing future cap
# removes the joint travel estimate as well as the availability mask.
robot=s.Robot.__new__(s.Robot);robot.vmax=np.ones(8)
robot.endpoint=lambda pose,q:dict(found=True,q_kin=np.ones(8)*4.,proxy=0.)
with_cap,_=robot.costs(np.zeros(3),np.zeros((1,3)),np.zeros(8),use_cap=True)
without_cap,_=robot.costs(np.zeros(3),np.zeros((1,3)),np.zeros(8),use_cap=False)
checks['no_future_cap_cost_scope'] = dict(with_cap_s=float(with_cap[0]),
    without_cap_s=float(without_cap[0]), removes_joint_time_too=bool(with_cap[0]!=without_cap[0]),
    status='REPRODUCED_PROTOCOL_DESCRIPTION_MISMATCH',fixture_not_robot_evidence=True)

result['all_passed_contracts_and_reproduced_failures'] = True
out=HERE/'planner_initial_checks.json'
out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,indent=2,allow_nan=False))
