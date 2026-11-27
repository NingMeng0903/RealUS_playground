"""Numerical witnesses for the theory contracts, not empirical robotics claims.

Run with pytest -s to show the six diagnostics. All inputs are local toy models;
no anatomy, simulator, or calibration assets are required.
"""

from itertools import product

import numpy as np
from scipy.integrate import quad
from scipy.linalg import cholesky_banded, cho_solve_banded
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.special import ndtr


# A piecewise-linear metric tree with nonuniform edge lengths and two Y junctions.
_EDGES = np.array([[0, 1], [1, 2], [0, 3], [3, 4], [0, 5], [5, 6], [5, 7]])
_LENGTHS = np.array([1.0, 0.8, 1.3, 0.7, 0.9, 1.1, 1.2])


def _graph_matrices():
    n = int(_EDGES.max()) + 1
    incidence = coo_matrix(
        (np.tile([-1.0, 1.0], len(_EDGES)),
         (np.repeat(np.arange(len(_EDGES)), 2), _EDGES.ravel())),
        shape=(len(_EDGES), n),
    ).tocsc()
    mass = np.asarray(abs(incidence).T @ (_LENGTHS / 2.0)).ravel()
    stiffness = (incidence.T @ diags(1.0 / _LENGTHS) @ incidence).tocsc()
    return incidence, mass, stiffness


def test_kirchhoff_shared_vertices_and_weak_flux():
    incidence, mass, stiffness = _graph_matrices()
    junctions = np.flatnonzero(np.bincount(_EDGES.ravel()) > 2)
    # An arbitrary field has a single trace at each vertex by construction.
    draws = np.random.default_rng(16).standard_normal((500, mass.size))
    endpoints = draws[:, _EDGES]
    trace_error = max(
        np.max(np.ptp(endpoints[:, _EDGES == vertex], axis=1))
        for vertex in junctions
    )

    # A balanced Neumann load supported at leaves is compatible with the
    # constant nullspace. Pin one DOF to fix the additive gauge. Zero-load
    # vertex rows imply the sum of outgoing slopes is zero for this solution,
    # not for arbitrary samples of the discrete Gaussian field.
    load = np.array([0.0, 0.0, 2.0, 0.0, -0.5, 0.0, 1.0, -2.5])
    assert load.sum() == 0.0
    displacement = np.r_[0.0, spsolve(stiffness[1:, 1:], load[1:])]
    edge_slope = (displacement[_EDGES[:, 1]] - displacement[_EDGES[:, 0]]) / _LENGTHS
    flux = np.zeros_like(load)
    np.add.at(flux, _EDGES[:, 0], edge_slope)
    np.add.at(flux, _EDGES[:, 1], -edge_slope)
    flux_error = np.max(np.abs(flux[junctions]))
    residual = np.max(np.abs(incidence.T @ edge_slope - load))
    print(f"\nKirchhoff: trace error={trace_error:.3e}, junction flux sum={flux_error:.3e}, "
          f"Neumann load residual={residual:.3e}")
    assert np.max(np.abs(edge_slope)) > 0.1
    assert max(trace_error, flux_error, residual) < 1e-12


def test_common_material_gauge_and_first_order_pose_suppression():
    # Two nearby points at the same station, with their complete non-shared
    # material terms collected into a and b. Distinct world-frame transports
    # must be pulled back before taking a difference.
    a = np.array([1.0, 2.0, -0.5])
    b = np.array([1.0, 2.125, -0.5])
    drift = np.array([16.0, -8.0, 4.0])
    g_a = np.eye(3)
    g_b = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    w_a, w_b = g_a @ (a + drift), g_b @ (b + drift)
    recovered = g_a.T @ w_a - g_b.T @ w_b
    np.testing.assert_array_equal(recovered, a - b)
    world_drift = (w_a - w_b) - (g_a @ a - g_b @ b)
    assert np.linalg.norm(world_drift) > 0.0

    # Smooth pose map with nonconstant pose Jacobian J_q(x)=b0+A x at q=0.
    # Compare the relative first-order error to both absolute first-order errors.
    a_x = np.diag([0.5, 2.0, -0.5])
    b0 = np.array([12.0, 5.0, 4.0])
    curvature = np.array([0.5, -0.25, 0.5])

    def pose(x, q):
        return x + q * (b0 + a_x @ x) + q**2 * curvature

    eps = 1e-4
    j_a = (pose(a, eps) - pose(a, -eps)) / (2 * eps)
    j_b = (pose(b, eps) - pose(b, -eps)) / (2 * eps)
    relative_jac = a_x @ (a - b)
    np.testing.assert_allclose(j_a - j_b, relative_jac, rtol=0, atol=1e-11)
    relative_error = np.linalg.norm(j_a - j_b)
    assert 0.0 < relative_error <= np.linalg.norm(a_x, ord=2) * np.linalg.norm(a - b) + 1e-11
    gain = min(np.linalg.norm(j_a), np.linalg.norm(j_b)) / relative_error
    print(f"\nGauge: material error={np.linalg.norm(recovered - (a - b)):.3e}, "
          f"uncancelled world drift={np.linalg.norm(world_drift):.3f}, "
          f"first-order pose suppression={gain:.3f}x")
    assert gain > 10.0


def test_mass_lumped_banded_cholesky_matches_dense_gp_conditioning():
    _, mass, stiffness = _graph_matrices()
    kappa = 0.8
    a = stiffness + kappa**2 * diags(mass, format="csc")
    precision = (a.T @ diags(1.0 / mass, format="csc") @ a).tocsc()

    # Off-vertex interpolation plus one observation mixing two graph branches.
    h = np.array([[0, 0.25, 0.75, 0, 0, 0, 0, 0],
                  [0.2, 0, 0, 0, 0, 0.8, 0, 0],
                  [0, 0, 0, 0.3, 0, 0, 0, 0.7]])
    y = np.array([0.25, -0.4, 0.8])
    variances = np.array([0.03, 0.07, 0.15])
    h_sparse = coo_matrix(h).tocsc()
    post_precision = precision + h_sparse.T @ diags(1.0 / variances) @ h_sparse
    lower = post_precision.tocoo()
    keep = lower.row >= lower.col
    offsets = lower.row[keep] - lower.col[keep]
    band = np.zeros((int(offsets.max()) + 1, mass.size))
    band[offsets, lower.col[keep]] = lower.data[keep]
    chol = cholesky_banded(band, lower=True)
    mean = cho_solve_banded((chol, True), h.T @ (y / variances))
    cov = cho_solve_banded((chol, True), np.eye(mass.size))

    # Independent spectral construction of the GP on this *same* mass-lumped
    # discretization, then condition in observation space (no posterior precision).
    inv_root_mass = 1.0 / np.sqrt(mass)
    laplacian = inv_root_mass[:, None] * stiffness.toarray() * inv_root_mass[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    features = inv_root_mass[:, None] * eigenvectors / (kappa**2 + eigenvalues)
    kernel = features @ features.T
    cross = kernel @ h.T
    innovation = h @ cross + np.diag(variances)
    dense_mean = cross @ np.linalg.solve(innovation, y)
    dense_cov = kernel - cross @ np.linalg.solve(innovation, cross.T)
    mean_error = np.max(np.abs(mean - dense_mean))
    covariance_error = np.max(np.abs(cov - dense_cov))
    print(f"\nMass-lumped GP: prior nnz={precision.nnz}, band rows={band.shape[0]}, "
          f"posterior mean error={mean_error:.3e}, covariance error={covariance_error:.3e}")
    assert precision.nnz < mass.size**2
    assert max(mean_error, covariance_error) < 1e-10


def test_belief_ipe_monotone_multiscale_attenuation():
    mu = np.array([0.25, -0.125, 0.75])
    p = np.kron(2.0 ** np.arange(6)[:, None], np.eye(3))
    base_cov = np.diag([0.001, 0.002, 0.003])
    # Positive-semidefinite, correlated increase in tracking uncertainty.
    direction = np.array([1.0, 0.5, -0.75])
    inflation = 0.01 * np.outer(direction, direction)
    phase = p @ mu
    amplitudes = []
    for scale in (0.0, 0.5, 1.0, 2.0):
        sigma = base_cov + scale * inflation
        damping = np.exp(-0.5 * np.diag(p @ sigma @ p.T))
        features = damping * np.array([np.sin(phase), np.cos(phase)])
        amplitudes.append(np.linalg.norm(features, axis=0).reshape(6, 3))
    amplitudes = np.array(amplitudes)
    assert np.all(np.diff(amplitudes, axis=0) < 0.0)
    assert np.all(np.diff(amplitudes, axis=1) < 0.0)
    # Doubling the projected frequency quadruples the extra log attenuation.
    log_change = np.log(amplitudes[-1] / amplitudes[0])
    np.testing.assert_allclose(log_change[1:], 4.0 * log_change[:-1], rtol=1e-12, atol=1e-14)
    print(f"\nIPE amplitude, first coordinate: low band {amplitudes[0, 0, 0]:.6f} -> "
          f"{amplitudes[-1, 0, 0]:.6f}; high band {amplitudes[0, -1, 0]:.6f} -> "
          f"{amplitudes[-1, -1, 0]:.3e}")


def test_heldout_split_conformal_whole_segment_violation():
    alpha = 0.20
    n_cal, n_test, n_sites = 2047, 8192, 12
    cal_seed, test_seed = np.random.SeedSequence(8964).spawn(2)
    scales = np.tile([0.5, 1.0, 2.0], n_sites // 3)

    def standardized_residuals(n, seed):
        rng = np.random.default_rng(seed)
        # Shared underestimation plus occasional localized spikes. Probes in
        # a segment are dependent; only whole segments are independent units.
        base = rng.choice([0.25, 0.5], size=(n, 1))
        residuals = np.broadcast_to(base, (n, n_sites)).copy()
        hit = np.flatnonzero(rng.random(n) < 0.6)
        locations = rng.integers(n_sites, size=hit.size)
        residuals[hit, locations] = rng.choice([1.5, 3.5], size=hit.size, p=[0.75, 0.25])
        return residuals

    cal = standardized_residuals(n_cal, cal_seed)
    cal_scores = np.max(cal, axis=1)
    k = int(np.ceil((n_cal + 1) * (1.0 - alpha)))
    q = np.partition(cal_scores, k - 1)[k - 1]
    # Analytic split-conformal bound: the randomized rank of one new score
    # among the exchangeable n_cal+1 scores is uniform; strict exceedances
    # of the calibration threshold are no more likely than rank > k. This
    # is marginal over calibration and the new segment, not a high-probability
    # bound on this fitted certificate nor a conditional-on-selection bound.
    rank_bound = 1.0 - k / (n_cal + 1)
    assert rank_bound <= alpha

    # Numerical witness: predicted segment safety margins clear the calibrated
    # pointwise residual envelope exactly. Negative true margin anywhere is
    # a violation of the entire certified segment in this toy geometry.
    test = standardized_residuals(n_test, test_seed)
    true_margins = (q - test) * scales
    violation = np.mean(np.any(true_margins < 0.0, axis=1))
    # An oracle marginal threshold of 0.5 covers each site with probability
    # 1 - .6/12 = .95, yet some site fails whenever a spike occurs (prob .6).
    marginal_failures = test > 0.5
    pointwise_failure = np.max(marginal_failures.mean(axis=0))
    marginal_segment_failure = np.mean(marginal_failures.any(axis=1))
    print(f"\nSplit conformal: analytic rank bound={rank_bound:.4f}; q={q:.2f}, "
          f"held-out segment violation={violation:.4f}; oracle marginal threshold "
          f"worst-site failure={pointwise_failure:.4f}, segment failure={marginal_segment_failure:.4f}")
    assert 0.0 < violation <= alpha
    assert pointwise_failure <= alpha < marginal_segment_failure


def _binary_channel(p):
    # Rows are true signs {-1,+1}; columns are observations of those signs.
    return np.array([[p, 1.0 - p], [1.0 - p, p]])


def _completion_value(channel, allocation, continuation):
    # Independent randomization uses allocation[observation, mode] only;
    # all designated physical modes are established at the same deadline.
    return 0.5 * np.einsum("ty,ym,tm->", channel, allocation, continuation)


def test_v83_reflection_v84_product_and_irreversible_cut():
    match = np.eye(2)
    anti = match[:, ::-1]
    independent = np.full((2, 2), 0.5)
    contrast = match - anti
    errors = []
    for p, s, r in product((0.5, 0.65, 0.81, 0.95), (0.2, 0.65, 1.0), (0.0, 0.2, 0.45)):
        channel = _binary_channel(p)
        continuation = s * match + r * anti
        # Obtain all four cells from two fixed-mode interventions, not by
        # conditioning on null events under the deterministic match policy.
        cells = channel.T @ continuation
        gamma = np.sum(cells * contrast)
        best_aligned = _completion_value(channel, match, continuation)
        uninformed = _completion_value(channel, independent, continuation)
        misaligned = _completion_value(channel, anti, continuation)
        errors.extend((gamma - 2.0 * (2.0 * p - 1.0) * (s - r),
                       best_aligned - uninformed - gamma / 4.0,
                       best_aligned - misaligned - gamma / 2.0))
        for allocation in (match, independent, anti):
            np.testing.assert_allclose((0.5 * channel.sum(axis=0)) @ allocation, [0.5, 0.5])

    # V8.3 thm:matchedgap: budget T-t0 is strictly below the minimum mode
    # reconnection time d0. The same-mode kernel may reflect the postprocessed
    # statistic Y'= -Y, without any change to the physical exit or raw log.
    kappa, c0, length, v_c, v_r = 40.0, 45.0, 55.0, 1.0, 1.0
    t0 = (length - c0) / v_c
    d0 = 2.0 * (c0 - kappa) / v_c + 2.0 / v_r
    deadline = 12.0
    assert t0 <= deadline < t0 + d0 and deadline - t0 < d0
    sigma = 1.0
    p = ndtr(1.0 / sigma)
    channel = _binary_channel(p)
    # Total variation cannot increase under projection to the unchangeable M.
    projected_tv = 0.5 * np.abs((channel @ match - channel @ anti)[1]).sum()

    def density(x, center):
        return np.exp(-0.5 * ((x - center) / sigma)**2) / (sigma * np.sqrt(2.0 * np.pi))

    # Reflection puts the source on the target graph M=sign(Y'), so joint TV
    # equals the scalar TV. The density difference is positive for x>0.
    reflected_tv, _ = quad(lambda x: density(x, 1.0) - density(x, -1.0), 0.0, np.inf)
    assert projected_tv > 0.0
    # With mandatory raw Y preservation, the target and unreflected source
    # have disjoint joint support (Y,M) a.s.; with free relabeling, TV is zero.
    raw_log_tv = 0.5 * np.abs(channel[:, :, None] * (match - anti)[None, :, :]).sum(axis=(1, 2))
    relabelled = channel[:, :, None] * anti[None, :, ::-1]
    np.testing.assert_allclose(raw_log_tv, 1.0, atol=1e-15, rtol=0)
    np.testing.assert_array_equal(relabelled, channel[:, :, None] * match[None, :, :])
    aligned_value = _completion_value(channel, match, match)  # s=1, r=0
    antialigned_value = _completion_value(channel, anti, match)
    gamma = np.sum((channel.T @ match) * contrast)
    errors.extend((projected_tv - reflected_tv, projected_tv - (2.0 * p - 1.0),
                   aligned_value - antialigned_value - reflected_tv, gamma / 2.0 - reflected_tv))

    # V8.4 thm:cut: exhaust the decision policies in a finite model whose only
    # precommit observations are the given channels. No wrong branch can finish.
    allocations = [match[:, choices].T for choices in product(range(2), repeat=2)]

    def bayes_bound(prefix):
        tv = 0.5 * np.abs(prefix[0] - prefix[1]).sum()
        return (1.0 + tv) / 2.0

    exact_cut = max(_completion_value(channel, policy, match) for policy in allocations)
    errors.extend((aligned_value - exact_cut, exact_cut - bayes_bound(channel)))
    alpha = 0.15  # Conditional failure rate bound on the certified inner experiment.
    inner, outer_only = _binary_channel(0.73), _binary_channel(0.89)
    prefixes = ((inner, (1.0 - alpha) * match), (outer_only, 0.9 * match))
    cut_value = max(_completion_value(prefix, policy, completion)
                    for prefix, completion in prefixes for policy in allocations)
    lower = (1.0 - alpha) * bayes_bound(inner)
    upper = max(bayes_bound(prefix) for prefix, _ in prefixes)
    assert lower < cut_value < upper
    identity_error = np.max(np.abs(errors))
    print(f"\nV8.3/V8.4: max identity error={identity_error:.3e}, "
          f"reflection defect={reflected_tv:.6f}, contrast={gamma:.6f}; exact cut={exact_cut:.6f}, "
          f"separate cut witness={lower:.6f} < {cut_value:.6f} < {upper:.6f}")
    assert identity_error < 1e-10
