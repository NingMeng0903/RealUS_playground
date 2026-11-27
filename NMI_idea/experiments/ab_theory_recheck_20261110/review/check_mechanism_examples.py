"""Independent analytic checks; no production imports and no robot execution.

The examples test mathematical conditions. Their normalized costs are NOT
claimed to be measured ultrasound or robot costs.
"""
from pathlib import Path
import json
import math
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, ndtri


OUT = Path(__file__).resolve().parent


def terminal_risk(p, delta=0.0, cap="both"):
    # Wrong terminal declaration costs 1. Abort costs 1.
    risks = [1.0 - p + delta, 1.0]
    if cap == "both":
        risks.append(p)
    return min(risks)


def binary_branch_risk(p, r, delta=0.0, cap="both"):
    # Independent unnormalized mass enumeration: rows H=+/-; columns Y=+/-.
    joint = np.array([[p*r, p*(1-r)], [(1-p)*(1-r), (1-p)*r]])
    value = 0.0
    for y in range(2):
        plus_mass, minus_mass = joint[:, y]
        mu = plus_mass + minus_mass
        risks = [minus_mass + delta*mu, mu]
        if cap == "both":
            risks.append(plus_mass)
        value += min(risks)
    return value


def gaussian_risk(snr, delta=0.0):
    # H equiprobable +/-1, Y=snr*H+N(0,1); full continuous Y retained.
    if snr == 0 or delta >= 1:
        return 0.5
    threshold = math.log((1+delta)/(1-delta))/(2*snr)
    choose_plus_given_plus = ndtr(snr-threshold)
    choose_plus_given_minus = ndtr(-snr-threshold)
    error = 0.5*(1-choose_plus_given_plus+choose_plus_given_minus)
    return float(error + 0.5*delta*(choose_plus_given_plus+choose_plus_given_minus))


def gaussian_risk_quadrature(snr, delta):
    norm = 1/math.sqrt(2*math.pi)
    def integrand(y):
        jp = 0.5*norm*math.exp(-0.5*(y-snr)**2)
        jm = 0.5*norm*math.exp(-0.5*(y+snr)**2)
        return min(jm+delta*(jp+jm), jp, jp+jm)
    return quad(integrand, -12-snr, 12+snr, epsabs=1e-11, limit=300)[0]


def binary_gaussian_mi(snr):
    norm = 1/math.sqrt(2*math.pi)
    entropy = quad(lambda z: norm*math.exp(-z*z/2)*
                   np.logaddexp(0, -2*snr*(snr+z))/math.log(2),
                   -12, 12, epsabs=1e-11)[0]
    return float(1-entropy)


def check():
    results = {"scope": "analytic mechanism examples, not task experiments"}

    # A: exact task-relevant identifiability, not necessarily full theta.
    F = np.array([[1., 0.], [2., 0.]])
    G_ok, G_bad = np.array([[3., 0.]]), np.array([[0., 1.]])
    theta0, theta1 = np.array([1., .6]), np.array([1., 1.4])
    assert np.array_equal(F@theta0, F@theta1)
    assert np.array_equal(G_ok@theta0, G_ok@theta1)
    assert not np.array_equal(G_bad@theta0, G_bad@theta1)
    results["A_nullspace"] = {
        "history_prediction_equal": True,
        "task_relevant_prediction_equal": True,
        "unexcited_query_predictions": [float((G_bad@t)[0]) for t in (theta0, theta1)],
        "history_rank": int(np.linalg.matrix_rank(F)),
        "parameter_dimension": 2,
    }

    # Position-origin nuisance cancels in local response increments. A change
    # in the public executable action set can make a previously irrelevant
    # response direction relevant, without making the origin identifiable.
    F3 = np.array([[0.,1.,0.], [0.,2.,0.]])
    Gx, Gy = np.array([[0.,3.,0.]]), np.array([[0.,0.,1.]])
    params = (np.array([-20.,1.,.6]), np.array([40.,1.,1.4]))
    assert np.array_equal(F3@params[0], F3@params[1])
    assert np.array_equal(Gx@params[0], Gx@params[1])
    F_after_y = np.vstack([F3, Gy])
    assert np.linalg.matrix_rank(F_after_y) == 2
    assert np.array_equal(F_after_y@np.array([1.,0.,0.]), np.zeros(3))
    identifiable_projection = np.linalg.pinv(F_after_y)@F_after_y
    assert np.max(np.abs(np.vstack([Gx,Gy])@(np.eye(3)-identifiable_projection))) < 1e-12
    results["A_position_nuisance_and_public_action_set"] = {
        "theta_fields": ["position_origin_nuisance", "gain_x", "gain_y"],
        "two_parameters_with_identical_history": [x.tolist() for x in params],
        "same_x_query_response": float((Gx@params[0])[0]),
        "different_y_query_responses": [float((Gy@x)[0]) for x in params],
        "rank_before_y_probe": int(np.linalg.matrix_rank(F3)),
        "rank_after_y_probe": int(np.linalg.matrix_rank(F_after_y)),
        "parameter_dimension": 3,
        "all_xy_queries_identifiable_after_y_probe": True,
        "position_origin_still_unidentified": True,
        "scope": "Origin is a cancelling gauge; an actual registration offset that changes visibility or centering cannot be discarded this way.",
    }

    # A: g is continuous, e_{k+1}=e_k+g*u_k, u=-e/g0.
    b, g0, tau = 4.0, 1.0, 0.5
    gs = np.linspace(.4, 1.6, 1201)
    errors = np.full_like(gs, b)
    for _ in range(3):
        errors += gs*(-errors/g0)
    closed = b*(1-gs/g0)**3
    assert np.max(np.abs(errors-closed)) < 1e-12
    first_u = -b/g0
    first_e = b + gs*first_u
    ghat = (first_e-b)/first_u
    final_e_adaptive = first_e + gs*(-first_e/ghat)
    assert np.max(np.abs(final_e_adaptive)) < 1e-12
    results["A_gain"] = {
        "b_mm": b, "g0": g0, "tolerance_mm": tau,
        "one_step_pass_gain_interval": [.875, 1.125],
        "three_step_pass_gain_interval": [.5, 1.5],
        "uniform_gain_domain": [.4, 1.6],
        "uniform_one_step_pass_probability_exact": .25/1.2,
        "uniform_three_step_pass_probability_exact": 1/1.2,
        "adaptive_second_step_max_abs_error_mm": float(np.max(np.abs(final_e_adaptive))),
        "interpretation": "Strong RLS/Broyden uses the ordinary first move to identify g; extra probing is not necessary here.",
    }
    sigma, u_star, alpha = .2, 4., .05
    z = float(ndtri(1-alpha/2))
    S = (sigma*abs(u_star)*z/tau)**2
    coverage = 2*ndtr(tau*math.sqrt(S)/(sigma*abs(u_star)))-1
    assert abs(coverage-(1-alpha)) < 1e-12
    results["A_fixed_design_noise"] = {
        "sigma_mm": sigma, "query_u_mm": u_star, "alpha": alpha,
        "required_sum_u_squared_mm2": S, "two_balanced_offsets_mm": math.sqrt(S/2),
        "coverage_at_threshold": float(coverage),
        "conditions": "Known baseline, independent Gaussian noise, fixed exogenous design, correctly specified local response. Not an adaptive optional-stopping guarantee.",
    }

    # B: exact finite-signal thresholds on broad grid.
    max_error = 0.0
    for p in np.linspace(0, 1, 101):
        for r in np.linspace(.5, 1, 101):
            exact = binary_branch_risk(p, r)
            formula = min(p, 1-p, 1-r)
            max_error = max(max_error, abs(exact-formula))
    assert max_error < 1e-12
    for delta in np.linspace(0, 1.5, 101):
        for r in np.linspace(.5, 1, 101):
            voi = .5-binary_branch_risk(.5, r, delta)
            formula = max(0., r-.5-delta/2)
            assert abs(voi-formula) < 1e-12
    for p in np.linspace(0, 1, 31):
        for r in np.linspace(.5, 1, 31):
            # Only + is executable; abort costs the same as a wrong decision.
            assert abs(binary_branch_risk(p, r, cap="plus")-(1-p)) < 1e-12

    rows = []
    for delta in (0., .2, .4, .5, .6, 1.):
        base = terminal_risk(.5, delta)
        probe = .1+binary_branch_risk(.5, .8, delta)
        rows.append({"extra_plus_cost": delta, "direct_risk": base,
                     "probe_total_risk": probe, "net_benefit": base-probe,
                     "decision": "probe" if probe < base-1e-12 else ("tie" if abs(probe-base)<1e-12 else "direct")})
    results["B_binary"] = {"grid_formula_max_error": max_error,
        "p": .5, "accuracy": .8, "probe_cost": .1,
        "rows": rows, "state_cost_threshold": .4,
        "only_plus_voi": 0.0,
        "strong_threshold_probe_baseline": "Matches exact two-step B on this toy; no algorithmic superiority follows."}

    snr = float(ndtri(.8))
    snr_u = float(ndtri(.95))
    mi = [binary_gaussian_mi(s) for s in (snr, snr_u)]
    qs = []
    for d in (0., .2, .5, .8):
        closed = gaussian_risk(snr, d)
        q = gaussian_risk_quadrature(snr, d)
        assert abs(closed-q) < 2e-8
        qs.append({"extra_plus_cost": d, "risk_closed": closed,
                   "risk_quadrature": q, "absolute_difference": abs(q-closed)})
    threshold = float(ndtri(.5+.1))
    assert abs(.5-gaussian_risk(threshold)-.1) < 1e-12
    results["B_continuous"] = {
        "target_snr_for_accuracy_point8": snr,
        "nuisance_snr_for_accuracy_point95": snr_u,
        "continuous_channel_mi_bits": mi,
        "probe_cost_point1_target_snr_threshold": threshold,
        "equal_information_control": "Set nuisance snr equal to target snr; entropy ties, task VOI differs. A favorable entropy tie-break can match B.",
        "strict_entropy_pressure_control": "At larger nuisance SNR entropy selects U, which is independent of target H; final target risk remains .5.",
        "state_cost_checks": qs,
    }

    # Current Gamma alias, explicitly BINARY Y = (X > 0).
    alias = []
    for name, x, w in (("P", [-1,1],[.5,.5]),
                       ("Q", [-math.sqrt(3),0,math.sqrt(3)],[1/6,2/3,1/6])):
        x,w = np.array(x),np.array(w)
        posterior_mse=0.
        for y in (False,True):
            m=(x>0)==y
            mass=w[m].sum()
            mean=float((w[m]*x[m]).sum()/mass)
            posterior_mse += float((w[m]*(x[m]-mean)**2).sum())
        alias.append({"name":name,"mean":float(w@x),"second_moment":float(w@(x*x)),
                      "posterior_mse":posterior_mse,"cost_point8_total":.8+posterior_mse})
    assert abs(alias[0]["posterior_mse"]) < 1e-12
    assert abs(alias[1]["posterior_mse"]-.4) < 1e-12
    results["Gamma_alias"] = alias
    results["all_checks_passed"] = True
    (OUT/"mechanism_checks.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps({"all_checks_passed": True, "output": str(OUT/"mechanism_checks.json"),
                      "A_required_excitation": S, "B_continuous_information_bits": mi}))


if __name__ == "__main__":
    check()
