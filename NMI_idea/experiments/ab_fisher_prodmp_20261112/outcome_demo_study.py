"""A fixed-budget ProDMP teacher that optimizes terminal positioning success."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import fisher_demo_study as study
from sensing import PRECISION, SD, capture_probabilities, detect_probability


HERE = Path(__file__).resolve().parent
ORIGINAL_CHOICES = study.teacher_choices
PATTERNS = np.array([[False, False], [True, False], [False, True], [True, True]])
ANGLES = (np.arange(256) + .5) * np.pi / 256.
RADIUS_MM = 3.


def success_probability(covariance, radius_mm=RADIUS_MM):
    """Probability that a zero-mean 2D Gaussian is within a radial tolerance."""
    eigenvalues = np.linalg.eigvalsh(covariance)
    angular_variance = (eigenvalues[..., 0, None] * np.cos(ANGLES)**2
                        + eigenvalues[..., 1, None] * np.sin(ANGLES)**2)
    return np.mean(-np.expm1(-.5 * radius_mm**2 / angular_variance), axis=-1)


def goal_scores(geometry, pairs, directional_probability):
    """Average 3 mm success over the four possible retained-frame patterns."""
    H = geometry.H
    information = ((H.swapaxes(-2, -1) / SD**2) @ H)[pairs]
    precisions = PRECISION + np.einsum("mi,pijk->pmjk", PATTERNS, information)
    task_covariances = np.linalg.inv(precisions)[..., :2, :2]
    # Estimate FOV survival with prior-averaged marginal probabilities. The
    # ranking model treats patterns as independent of the latent state.
    probability = capture_probabilities(geometry.nominal, H, directional_probability)
    hit = probability[pairs][:, None, :]
    weights = np.where(PATTERNS, hit, 1. - hit).prod(axis=2)
    return np.sum(weights * success_probability(task_covariances), axis=1)


def choose(geometry, preferred, matched_travel):
    pairs, choices, risks = ORIGINAL_CHOICES(geometry, preferred, matched_travel)
    direction = detect_probability(geometry.beam_tilt, preferred)
    choices["goal"] = int(np.argmax(goal_scores(geometry, pairs, direction)))
    return pairs, choices, risks


def paired_success(arms, left, right):
    a = {(r["site"], r["public_echo_center_deg"]): r for r in arms[left]}
    b = {(r["site"], r["public_echo_center_deg"]): r for r in arms[right]}
    if a.keys() != b.keys():
        raise ValueError("comparison requires the same held-out sites and echo conditions")
    sites = sorted({k[0] for k in a})
    by_site = np.asarray([
        np.mean([a[k]["within_3mm"] - b[k]["within_3mm"] for k in a if k[0] == s])
        for s in sites
    ])
    generator = np.random.default_rng(44115)
    draws = generator.choice(by_site, size=(6000, len(by_site))).mean(axis=1)
    return {
        "percentage_point_delta": float(100. * by_site.mean()),
        "site_bootstrap_95_pp": (100. * np.percentile(draws, [2.5, 97.5])).tolist(),
        "same_capture_pairs": sum(a[k]["chosen_deg"] == b[k]["chosen_deg"] for k in a),
        "contexts": len(a),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "outcome_teacher")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    study.teacher_choices = choose
    study.TEACHERS = (*study.TEACHERS, "goal")
    path = HERE / "TILT_BANK.npz"
    geometry = study.get_geometry(path)
    output = {
        "goal_tolerance_mm": RADIUS_MM,
        "demo_count": study.DEMO_COUNT,
        "subjects_per_context": study.SUBJECTS_PER_CONTEXT,
        "n_basis": 12,
        "source_sha256": {
            p.name: study.sha(p) for p in [Path(__file__), path, HERE / "fisher_demo_study.py",
                                          HERE / "sensing.py", HERE / "demo_fit.py"]
        },
        "design": "Same old simulation conditions, frozen heldout sites, and trial draws. "
                  "The additional teacher uses public analytic hit probabilities and target "
                  "Gaussian threshold success, never evaluation draws.",
    }
    for name, matched in (("equal_36deg", True), ("common_48deg_cap", False)):
        report, example = study.evaluate(geometry, matched, 12)
        report["paired_goal_success"] = {
            sensor: {
                f"{role}_vs_{other}": paired_success(arms, role + "_goal", role + "_" + other)
                for role in ("teacher", "prodmp") for other in ("fisher", "visibility", "conditional")
            }
            for sensor, arms in report["raw_by_context"].items()
        }
        output[name] = report
        if matched:
            render_example(example, report, args.out / "GOAL_TEACHER_3D.html")
        for sensor, arms in report["statistics"].items():
            values = {arm: round(100. * stats["within_3mm"], 3) for arm, stats in arms.items()
                      if arm.endswith(("goal", "fisher", "visibility", "conditional"))}
            print(name, sensor, values, flush=True)
    (args.out / "SUMMARY.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


def render_example(example, report, output):
    from plot_demo_study import make_figure
    import plot_demo_study as plotting

    plotting.METHODS = {
        "goal": ("3 mm 通过率优先", "#3b69a4"),
        "visibility": ("可见性优先", "#c36a25"),
        "fisher": ("Fisher", "#007e9d"),
    }
    geometry = example["geometry"]
    data = {
        "site": geometry.site, "pose_deg": geometry.pose,
        "preferred": example["public_echo_center_deg"], "contacts": geometry.contact_xyz_m,
        "frames": geometry.frames, "vessel": geometry.vessel_xyz_m, "angles": geometry.angles,
    }
    data.update({f"{name}_{key}": value for name, row in example["shown_paths"].items()
                 for key, value in row.items()})
    figure = make_figure(data, report, 12)
    figure.layout.title.text = (
        "按任务通过率生成 ProDMP 示范 · 等预算 36°"
        f"<br><sup>留出站点 {geometry.site} · 屈膝 {geometry.pose:g}°"
        f" · 合成回波角 {data['preferred']:+g}° · 3 秒 / 2 次采集</sup>"
    )
    shifted = report["statistics"]["shifted"]
    scores = "　".join(f"{name} {shifted['prodmp_' + key]['within_3mm']:.2%}"
                     for key, (name, _) in plotting.METHODS.items())
    figure.layout.annotations[-1].text = (
        "实线为学习轨迹　点线为示范　空心圆为回放所用的候选采集角<br>"
        f"留出定位 ≤3 mm（回波偏差与噪声变化）　{scores}"
    )
    figure.write_html(output, include_plotlyjs=True, auto_open=False)


if __name__ == "__main__":
    main()
