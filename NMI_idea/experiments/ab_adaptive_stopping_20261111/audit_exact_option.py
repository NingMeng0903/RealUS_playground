"""Independent analytic audit; no production model or experimental outcome imported.

Two same-pose, same-dwell attempts share a Gaussian response latent.  Conditional
on that latent Q is the noisy quality pass probability.  The two-attempt HMM
correct/false stopping probabilities are quadratic in Q.  This audit compares
the analytic moments to adaptive integration, and measures GH33/GH65 error.
It does not train, calibrate, select a policy, or produce research outcomes.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import ndtr, owens_t
from numpy.polynomial.hermite import hermgauss


def main():
    started = time.perf_counter()
    seed = 721506
    rng = np.random.default_rng(seed)
    transition = np.array([[.97, .03], [.70, .30]])
    nodes33, weights33 = hermgauss(33)
    nodes65, weights65 = hermgauss(65)
    maxima = {name: {"error": 0.0, "case": None} for name in (
        "second_moment_vs_adaptive_quad", "correct_option_vs_adaptive_quad",
        "false_option_vs_adaptive_quad", "gh33_second_moment_vs_exact",
        "gh65_second_moment_vs_exact")}
    cases = []
    for index in range(200):
        mean = rng.uniform(-.6, .6)
        variance = 10 ** rng.uniform(-7, -.5)
        noise_variance = 10 ** rng.uniform(-5, -2)
        geometric_quality = rng.uniform(.2, .7)
        threshold = .58
        z = (geometric_quality + mean - threshold) / math.sqrt(variance + noise_variance)
        mu = float(ndtr(z))
        mu2 = float(mu - 2 * owens_t(z, math.sqrt(noise_variance / (2 * variance + noise_variance))))

        def quality_pass(t):
            return ndtr((geometric_quality + mean + math.sqrt(variance) * t - threshold)
                        / math.sqrt(noise_variance))

        def normal_density(t):
            return math.exp(-t * t / 2) / math.sqrt(2 * math.pi)

        split = float(np.clip((threshold - geometric_quality - mean) / math.sqrt(variance), -12, 12))

        def integrate(function):
            return quad(lambda t: function(t) * normal_density(t), -12, 12,
                        points=[split], epsabs=1e-11, epsrel=1e-11, limit=300)[0]

        numeric_mu2 = integrate(lambda t: quality_pass(t) ** 2)
        gh33 = float(np.sum(weights33 * quality_pass(math.sqrt(2) * nodes33) ** 2) / math.sqrt(math.pi))
        gh65 = float(np.sum(weights65 * quality_pass(math.sqrt(2) * nodes65) ** 2) / math.sqrt(math.pi))

        # d: detection, u Q: target public acceptance, a Q: correct target
        # acceptance, b: clutter acceptance.  mode=0 is present; mode=1 lost.
        detection = np.array([.8, 0.0])
        feature_pass = rng.uniform(.3, 1)
        true_gate = int(rng.random() < .5)
        u = detection * feature_pass
        a = u * true_gate
        clutter_accept = rng.uniform(0, .1)
        b = (1 - detection) * clutter_accept
        initial_mode = int(rng.integers(2))
        p1 = transition[initial_mode]
        correct = float(p1 @ a * mu + (p1 * (1 - b)) @ transition @ a * mu
                        - (p1 * u) @ transition @ a * mu2)
        false = float(p1 @ b + p1 @ (u - a) * mu
                      + (p1 * (1 - b)) @ transition @ b
                      - (p1 * u) @ transition @ b * mu
                      + (p1 * (1 - b)) @ transition @ (u - a) * mu
                      - (p1 * u) @ transition @ (u - a) * mu2)

        def direct(t):
            q = quality_pass(t)
            success = a * q
            wrong = b + (u - a) * q
            reject = 1 - success - wrong
            return np.array([p1 @ success + (p1 * reject) @ transition @ success,
                             p1 @ wrong + (p1 * reject) @ transition @ wrong])

        numeric_correct = integrate(lambda t: direct(t)[0])
        numeric_false = integrate(lambda t: direct(t)[1])
        errors = {
            "second_moment_vs_adaptive_quad": abs(mu2 - numeric_mu2),
            "correct_option_vs_adaptive_quad": abs(correct - numeric_correct),
            "false_option_vs_adaptive_quad": abs(false - numeric_false),
            "gh33_second_moment_vs_exact": abs(gh33 - mu2),
            "gh65_second_moment_vs_exact": abs(gh65 - mu2),
        }
        row = {"index": index, "mean": float(mean), "variance": float(variance),
               "noise_variance": float(noise_variance), "geometric_quality": float(geometric_quality),
               "quality_threshold": threshold, "feature_pass": float(feature_pass),
               "true_geometry_gate": true_gate, "clutter_accept": float(clutter_accept),
               "initial_mode": initial_mode, "first_moment": mu, "second_moment": mu2,
               "correct_probability": correct, "false_probability": false, "errors": errors}
        cases.append(row)
        for name, error in errors.items():
            if error > maxima[name]["error"]:
                maxima[name] = {"error": float(error), "case": index}

    analytic_ok = all(maxima[name]["error"] < 1e-9 for name in (
        "second_moment_vs_adaptive_quad", "correct_option_vs_adaptive_quad",
        "false_option_vs_adaptive_quad"))
    output = {"kind": "independent_same_pose_same_dwell_probability_identity_audit",
              "research_outcome": False, "seed": seed, "cases": 200,
              "bounds": {"response_mean": [-.6, .6], "response_variance_log10": [-7, -.5],
                         "sensor_variance_log10": [-5, -2], "geometric_quality": [.2, .7]},
              "integration": {"normal_domain": [-12, 12], "epsabs": 1e-11,
                              "epsrel": 1e-11, "split_at_quality_transition": True},
              "formula": "E Q=Phi(z); E Q^2=Phi(z)-2 T(z,sqrt(R/(2v+R)))",
              "scope": "Same projected response, same geometric quality, same sensor variance; general unequal attempts require E(Q1 Q2) with the correct joint Gaussian covariance.",
              "hmm": transition.tolist(), "maxima": maxima,
              "analytic_pass_under_1e_9": analytic_ok,
              "gh_warning": "GH33/GH65 are comparisons, not converged reference truth over these stress parameters.",
              "numpy": np.__version__, "scipy": scipy.__version__,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "wall_s": time.perf_counter() - started, "case_details": cases}
    destination = Path(__file__).with_suffix(".json")
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "analytic_pass_under_1e_9": analytic_ok,
                      "maxima": maxima}, indent=2))
    if not analytic_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
