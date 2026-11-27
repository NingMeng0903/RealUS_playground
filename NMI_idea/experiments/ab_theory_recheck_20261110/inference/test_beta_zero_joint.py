"""Regression for independently found beta=0 times hard-zero log likelihood."""
import numpy as np
from inference.continuous_smc import _mcmc_move,adaptive_tempered_resample_move

def test_beta_zero_joint_ignores_new_block_support():
    particles=[-1.,1.]
    block=lambda xs:np.where(np.asarray(xs)<0,-np.inf,0.)
    zero=lambda xs:np.zeros(len(xs))
    results=[]
    for joint in (None,lambda xs:(zero(xs),block(xs))):
        with np.errstate(invalid='raise'):
            moved,acceptance,_=_mcmc_move(particles,0.,block,zero,lambda x:0.,
                lambda x,rng,scale:-x,np.random.default_rng(0),sweeps=1,
                scales=[1.],joint_loglike=joint)
        assert moved==[1.,-1.]
        assert acceptance==1.
        results.append(moved)
    assert results[0]==results[1]

def test_initial_low_ess_joint_and_separate_callbacks_match():
    block=lambda xs:np.where(np.asarray(xs)==0,-np.inf,0.)
    zero=lambda xs:np.zeros(len(xs))
    results=[]
    for joint in (None,lambda xs:(zero(xs),block(xs))):
        with np.errstate(invalid='raise'):
            result=adaptive_tempered_resample_move(list(range(4)),[.8,.1,.05,.05],
                block,zero,lambda x:0.,lambda x,rng,scale:int(rng.integers(4)),
                np.random.default_rng(0),mh_sweeps=2,joint_loglike=joint)
        assert result.committed
        results.append(result)
    assert results[0].particles==results[1].particles
    np.testing.assert_array_equal(results[0].weights,results[1].weights)
    assert results[0].diagnostics==results[1].diagnostics
