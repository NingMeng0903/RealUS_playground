"""Verify paired held-knee acquisition clocks and recompute physical RMSE."""
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent / "separation"
ARMS = ("burst_one_site", "burst_two_response", "separated_one_site", "separated_two_response")
FIELDS = ("w_error_mm", "z_common_error_mm", "z_start_smoothed_error_mm")
NAMES = ("w_rmse_mm", "z_end_rmse_mm", "z_stop_smoothed_rmse_mm", "future_0p5_rmse_mm", "future_1p0_rmse_mm")
Z90 = 1.6448536269514722


def run_errors(run):
    squared = [float(run[field])**2 for field in FIELDS]
    for horizon in ("0.5s", "1.0s"):
        future = run["forecast"][horizon]
        error = np.asarray(future["mean_xyz_mm"]) - np.asarray(future["truth_xyz_mm"])
        value = np.mean(np.sum(error**2, axis=-1))
        assert np.isclose(value, future["rmse_mm"]**2, atol=1.e-10)
        squared.append(float(value))
    return squared


def main():
    cases = [json.loads(p.read_text()) for p in sorted((ROOT / "raw").glob("case_*.json"))]
    assert [r["case_index"] for r in cases] == list(range(384, 512))
    assert len({r["case_seed"] for r in cases}) == len(cases)
    clock, travel, valid = [], [], []
    for r in cases:
        assert r["prefix"]["captures"] == 28
        t = r["schedule"]["common_final_t_s"]
        clock.append(t - r["truth"]["stop_t_s"])
        assert {len(r["trace"]["arms"][arm]["frames"]) for arm in ARMS} == {3}
        for arm in ARMS:
            actions = r["trace"]["arms"][arm]["actions"]
            assert np.isclose(actions[-1]["t"], t)
            if arm.endswith("two_response"):
                travel.append(sum(a["distance_mm"] for a in actions))
            for a in actions:
                valid.append(a["duration_s"] >= .21 + a["distance_mm"] / 90. - 1.e-10)
            for m in r["arms"][arm].values():
                assert np.isclose(m["final_t_s"], t)
                assert all(np.isclose(m["forecast"][h]["time_s"], t + dt)
                           for h, dt in (("0.5s", .5), ("1.0s", 1.)))
    assert all(valid)
    errors = {}
    metrics = {}
    for arm in ARMS:
        for method in cases[0]["arms"][arm]:
            key = f"{arm}/{method}"
            rs = [r["arms"][arm][method] for r in cases]
            errors[key] = np.array([run_errors(r) for r in rs])
            metrics[key] = dict(zip(NAMES, np.sqrt(errors[key].mean(axis=0)).tolist()))
            for field, sd, short in zip(FIELDS, ("w_sd_mm", "z_common_sd_mm", "z_start_smoothed_sd_mm"),
                                        ("w", "z_end", "z_stop")):
                metrics[key][f"{short}_90_coverage"] = float(np.mean(
                    [abs(r[field]) <= Z90 * r[sd] for r in rs]))
    rng = np.random.default_rng(6610083)
    draws = rng.integers(0, len(cases), size=(4000, len(cases)))
    pairs = {
        "late_middle_same_site_vs_early": ("separated_one_site/latent", "burst_one_site/latent"),
        "late_middle_two_sites_vs_early": ("separated_two_response/latent", "burst_two_response/latent"),
        "two_sites_vs_one_early": ("burst_two_response/latent", "burst_one_site/latent"),
        "two_sites_vs_one_late": ("separated_two_response/latent", "separated_one_site/latent"),
        "early_same_site_vs_no_post_stop_US": ("burst_one_site/latent", "burst_one_site/latent_no_post_stop_US"),
        "late_two_sites_vs_no_post_stop_US": ("separated_two_response/latent", "separated_two_response/latent_no_post_stop_US"),
        "early_same_site_lag_vs_no_lag": ("burst_one_site/latent", "burst_one_site/no_lag_plus_US"),
    }
    contrasts = {}
    for name, (a, b) in pairs.items():
        x, y = errors[a], errors[b]
        delta = np.sqrt(x.mean(axis=0)) - np.sqrt(y.mean(axis=0))
        bootstrap = np.sqrt(x[draws].mean(axis=1)) - np.sqrt(y[draws].mean(axis=1))
        ci = np.quantile(bootstrap, [.025, .975], axis=0).T
        contrasts[name] = {n: {"delta": float(d), "ci95": c.tolist()} for n, d, c in zip(NAMES, delta, ci)}
    output = {
        "cases": len(cases), "case_indices": [384, 511],
        "prefix_frames": 28, "additional_shared_captures": 2, "paired_captures": 3,
        "common_finish_time_after_stop_s": [float(min(clock)), float(max(clock))],
        "two_site_travel_mm": [float(min(travel)), float(max(travel))],
        "zero_z_end_rmse_mm": float(np.sqrt(np.mean([r["truth"]["z_final_mm"]**2 for r in cases]))),
        "zero_z_stop_rmse_mm": float(np.sqrt(np.mean([r["truth"]["z_start_mm"]**2 for r in cases]))),
        "pooled": metrics, "paired_rmse_contrasts": contrasts,
    }
    (ROOT / "VERIFIED_SUMMARY.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "pooled"}, indent=2))


if __name__ == "__main__":
    main()
