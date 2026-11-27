"""Reproducible algebra/numerical contracts, never acoustic validation."""
from pathlib import Path
import sys, json, hashlib, time
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor'))
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import e3nn
from e3nn import o3

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

def tensor_contract(device):
    torch.manual_seed(41105)
    inp = o3.Irreps('2x0e + 3x1o')
    out = o3.Irreps('2x0e + 2x1o + 1x2e')
    net = o3.FullyConnectedTensorProduct(inp, inp, out).to(device=device, dtype=torch.float64)
    x = torch.randn(32, inp.dim, device=device, requires_grad=True)
    # e3nn 0.6.0 constructs Wigner generator constants on CPU in this env.
    # Build the fixed representation matrices there, then test actual GPU net.
    R = torch.tensor(Rotation.from_rotvec([.8, -.3, .5]).as_matrix())
    Di, Do = inp.D_from_matrix(R).to(device), out.D_from_matrix(R).to(device)
    def energy(z):
        y = net(z, z)
        return (y.square().sum(-1) + .1 * y[:, 0]).sum()
    y = net(x, x)
    xr = (x.detach() @ Di.T).requires_grad_()
    yr = net(xr, xr)
    E, Er = energy(x), energy(xr)
    grad = torch.autograd.grad(E, x)[0]
    gradr = torch.autograd.grad(Er, xr)[0]
    rng = np.random.default_rng(41105)
    fd = []
    for i, j in zip(rng.integers(0, 32, 48), rng.integers(0, inp.dim, 48)):
        xp, xm = x.detach().clone(), x.detach().clone()
        xp[i,j] += 1e-6; xm[i,j] -= 1e-6
        numeric = (energy(xp) - energy(xm)) / 2e-6
        exact = grad[i,j]
        fd.append(float((abs(numeric-exact)/(1+abs(exact))).detach()))
    if device == 'cuda': torch.cuda.synchronize()
    result = {
        'device': device, 'dtype': str(x.dtype), 'seed': 41105,
        'input_irreps': str(inp), 'output_irreps': str(out),
        'equivariance_max_abs': float((yr-y@Do.T).abs().max()),
        'energy_invariance_abs': float(abs(E-Er)),
        'gradient_equivariance_max_abs': float((gradr-grad@Di.T).abs().max()),
        'finite_difference_max_normalized_error': max(fd),
        'trained': False,
        'scope': 'Actual e3nn tensor-product and autograd numerical contract; no learning benefit or nonrigid symmetry claim.'
    }
    assert result['equivariance_max_abs'] < 1e-8, result
    assert result['energy_invariance_abs'] < 1e-7, result
    assert result['gradient_equivariance_max_abs'] < 1e-7, result
    assert max(fd) < 1e-5, result
    return result

def canonical_contract():
    rng = np.random.default_rng(41106)
    R = Rotation.random(random_state=rng).as_matrix()
    Q = Rotation.random(random_state=rng).as_matrix()
    x = rng.normal(size=(32, 3))
    a = rng.normal(size=(32, 3))
    W = rng.normal(size=(6, 3))
    # Ordinary nonlinear network in an observed probe frame, same weights.
    def predict(xw, aw, frame):
        local = np.concatenate([xw@frame, aw@frame], axis=-1)
        return np.tanh(local@W)@frame.T
    y = predict(x, a, R)
    yr = predict(x@Q.T, a@Q.T, Q@R)
    err = float(np.max(np.abs(yr-y@Q.T)))
    assert err < 1e-12
    return {'ordinary_canonical_network_equivariance_max_abs': err,
            'interpretation': 'Unique calibrated probe frame is sufficient here. This is not a capacity-matched learned performance comparison with e3nn.'}

def risk_contract():
    # Correlated residual and Jacobian: dropping cross terms changes full risk.
    r = np.array([-1., 1.]); B = r.copy(); step = -1.
    X = np.stack([r, B], axis=1)
    Gamma = X.T @ X / 2
    v = np.array([1.,step])
    actual = float(np.mean((r+B*step)**2))
    matrix = float(v@Gamma@v)
    omitted_cross = float(np.mean(r*r)+step*step*np.mean(B*B))
    # Ignoring the existing task residual can falsely accept the final action.
    increment_only = .2**2
    composed = (1-.2)**2
    # A=+/-1 has mean zero but no nonzero common null direction.
    nominal_risk, posterior_risk = float(B.mean()**2), float(np.mean(B*B))
    # A binary .8/.2 observation gives posterior mean +/- .6 and U=.64.
    likelihood = np.array([[.8,.2],[.2,.8]])
    prior = np.array([.5,.5])
    posterior = prior[:,None]*likelihood
    py = posterior.sum(0); posterior /= py
    means = B@posterior
    second = (B*B)@posterior
    U_after = second-means**2
    assert actual == matrix == 0.
    assert omitted_cross == 2.
    assert increment_only < .1 < composed
    assert abs(float(py@second)-1.) < 1e-12
    assert abs(1-float(py@U_after)-float(py@(means**2))) < 1e-12
    return {'full_risk_direct':actual, 'full_risk_matrix':matrix,
            'drop_cross_terms_risk':omitted_cross,
            'increment_only_risk':increment_only, 'final_command_risk':composed,
            'nominal_null_risk':nominal_risk,'posterior_risk':posterior_risk,
            'G_before':1., 'expected_G_after':float(py@second),
            'U_before':1., 'expected_U_after':float(py@U_after),
            'scope':'Finite exact mathematical examples, not patient-model calibration or a physical chance certificate.'}

def nonrigid_contract():
    F = np.diag([2., .5, 1.])
    tangent = np.array([1.,1.,0.]); normal = np.array([1.,-1.,0.])
    v = F@tangent
    correct_n = np.linalg.inv(F).T@normal
    wrong_n = F@normal
    normdot = lambda a,b: float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
    result = {'correct_tangent_normal_cosine':normdot(v,correct_n),
              'treat_normal_as_vector_cosine':normdot(v,wrong_n)}
    assert abs(result['correct_tangent_normal_cosine']) < 1e-12
    assert abs(result['treat_normal_as_vector_cosine']) > .5
    return result

if __name__ == '__main__':
    start = time.perf_counter()
    results = {'evidence_level':'algebra and numerical implementation only',
               'versions':{'python':sys.version,'torch':torch.__version__,'e3nn':e3nn.__version__},
               'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'e3nn':[tensor_contract('cpu')], 'canonical':canonical_contract(),
               'risk':risk_contract(),'nonrigid':nonrigid_contract()}
    if torch.cuda.is_available(): results['e3nn'].append(tensor_contract('cuda'))
    results['wall_s'] = time.perf_counter()-start
    (ROOT/'contracts.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
