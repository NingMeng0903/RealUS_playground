#!/usr/bin/env python3
"""Audit dynamic posterior queries on the recorded B-case replay.

This is a read-only audit of public replay logs.  It calls the existing
``render_search_replay.get_chapters`` and ``belief_timeline`` functions, takes
one selected posterior from the recorded B case, and evaluates that *same*
mean/covariance at the calibrated 11, 15 and 19 degree pose knots.  No truth
draw, patient simulation, heart model, or new observation is generated.

The reported lateral variance is expanded as

    Var[d_i(theta)] = C_ii + 2 g_i(theta) C_i,slip
                       + g_i(theta)^2 C_slip,slip,

which makes the cross-covariance contribution explicit.  The output directory
contains ``SUMMARY.json`` and a small self-contained Plotly-HTML 3-D view.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.special import ndtri


HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "dynamic_query_audit_20261112"
POSE_DEGREES = (11, 15, 19)
Z90 = float(ndtri(0.95))


def _load_replay_module():
    import render_search_replay  # type: ignore  # noqa: E402

    return render_search_replay


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _site_indices(scan: Any, requested: Sequence[int] | None) -> list[int]:
    if requested:
        sites = sorted(set(int(i) for i in requested))
    else:
        # Six points span the material trunk while retaining the distal end.
        sites = [0, 6, 12, 18, 24, int(scan.n - 1)]
    if not sites or min(sites) < 0 or max(sites) >= int(scan.n):
        raise ValueError(f"material sites must lie in [0, {int(scan.n) - 1}]")
    return sites


def _query_point(scan: Any, belief: Any, pose: int, site: int) -> dict[str, Any]:
    """Evaluate one trunk material node in world coordinates."""

    mean = np.asarray(belief.mean, dtype=float)
    covariance = np.asarray(belief.covariance, dtype=float)
    slip_index = mean.size - 1
    row = np.asarray(scan.row(pose, site, 0, True), dtype=float)
    gain = float(row[slip_index])
    node_var = float(covariance[site, site])
    cross_var = float(2.0 * gain * covariance[site, slip_index])
    slip_var = float(gain * gain * covariance[slip_index, slip_index])
    variance = float(node_var + cross_var + slip_var)
    if variance < -1.0e-8:
        raise ValueError(f"negative posterior variance at pose={pose}, site={site}: {variance}")
    variance = max(variance, 0.0)
    signed_offset = float(row @ mean)
    sd = float(np.sqrt(variance))
    lateral = np.asarray(scan.frames[pose, site, 1], dtype=float)
    lateral_norm = float(np.linalg.norm(lateral))
    if lateral_norm <= 1.0e-12:
        raise ValueError(f"zero lateral transport axis at pose={pose}, site={site}")
    lateral = lateral / lateral_norm
    nominal = np.asarray(scan.trunk[pose, site], dtype=float)
    posterior = nominal + 1.0e-3 * signed_offset * lateral
    lower = nominal + 1.0e-3 * (signed_offset - Z90 * sd) * lateral
    upper = nominal + 1.0e-3 * (signed_offset + Z90 * sd) * lateral
    return {
        "pose_index": int(pose),
        "angle_deg": float(scan.angles[pose]),
        "site_index": int(site),
        "s_mm": float(scan.s[site]),
        "slip_gain": gain,
        "signed_offset_mm": signed_offset,
        "sd_mm": sd,
        "half_width90_mm": float(Z90 * sd),
        "variance_mm2": variance,
        "variance_components_mm2": {
            "node": node_var,
            "cross_covariance": cross_var,
            "slip": slip_var,
            "total": variance,
        },
        "nominal_world_m": nominal,
        "posterior_mean_world_m": posterior,
        "lower90_world_m": lower,
        "upper90_world_m": upper,
        "lateral_unit_world": lateral,
    }


def _all_pose_points(scan: Any, belief: Any, pose: int) -> list[dict[str, Any]]:
    return [_query_point(scan, belief, pose, int(site)) for site in range(int(scan.n))]


def _recompute_pose_learning_audit(dynamic_root: Path) -> dict[str, Any]:
    """Independently aggregate the sealed 48-run pose-learning audit."""

    source = dynamic_root / "pose_learning_audit.py"
    result = dynamic_root / "POSE_LEARNING_AUDIT.json"
    report = json.loads(result.read_text(encoding="utf-8"))
    runs = report.get("runs", [])
    if len(runs) != 48:
        raise AssertionError(f"expected 48 sealed pose-learning runs, found {len(runs)}")
    source_text = source.read_text(encoding="utf-8")
    varied_pattern = bool(re.search(r'"varied_train"\s*,\s*\(11\s*,\s*15\)', source_text))
    fixed_pattern = bool(re.search(r'"single_pose_train"\s*,\s*\(15\s*,\s*15\)', source_text))
    if not varied_pattern or not fixed_pattern:
        raise AssertionError("sealed pose-learning source does not state 11/15 and fixed-15 protocols")

    recomputed: dict[str, dict[str, float]] = {}
    for protocol in ("varied_train", "single_pose_train"):
        rows = [run["protocols"][protocol] for run in runs]
        values = {
            "holdout_19deg_rmse_mm": float(np.sqrt(np.mean([row["heldout_19deg_mse"] for row in rows]))),
            "holdout_19deg_zero_slip_rmse_mm": float(
                np.sqrt(np.mean([row["heldout_19deg_frozen_mse"] for row in rows]))
            ),
            "mean_slip_sd_mm": float(np.mean([row["slip_sd_mm"] for row in rows])),
            "mean_captures": float(np.mean([row["captures"] for row in rows])),
        }
        recomputed[protocol] = values
        for key, value in values.items():
            expected = float(report["summary"][protocol][key])
            if not np.isclose(value, expected, atol=1.0e-12, rtol=0.0):
                raise AssertionError(f"sealed pose audit mismatch {protocol}.{key}: {value} != {expected}")

    fixed_rows = [run["protocols"]["single_pose_train"] for run in runs]
    varied_rows = [run["protocols"]["varied_train"] for run in runs]
    fixed_slips = np.asarray([row["fitted_slip_mm"] for row in fixed_rows], dtype=float)
    fixed_sds = np.asarray([row["slip_sd_mm"] for row in fixed_rows], dtype=float)
    varied_sds = np.asarray([row["slip_sd_mm"] for row in varied_rows], dtype=float)
    fixed_identifiable = bool(np.allclose(fixed_slips, 0.0, atol=1.0e-12, rtol=0.0))
    fixed_prior_sd = float(np.median(fixed_sds))
    varied_improves = bool(
        recomputed["varied_train"]["holdout_19deg_rmse_mm"]
        < recomputed["single_pose_train"]["holdout_19deg_rmse_mm"]
    )
    return {
        "source": str(source),
        "source_sha256": _sha256(source),
        "result": str(result),
        "result_sha256": _sha256(result),
        "runs": int(len(runs)),
        "heldout_pose_deg": int(report["heldout_pose_deg"]),
        "training_protocols": {
            "varied_train": [11, 15],
            "single_pose_train": [15],
            "heldout": [19],
        },
        "recomputed_summary": recomputed,
        "matches_sealed_summary": True,
        "fixed_angle_identifiability": {
            "fixed_pose_fitted_slip_all_zero": fixed_identifiable,
            "fixed_pose_median_slip_sd_mm": fixed_prior_sd,
            "varied_pose_mean_slip_sd_mm": float(varied_sds.mean()),
            "varied_pose_slip_sd_less_than_fixed": bool(varied_sds.mean() < fixed_prior_sd),
            "varied_pose_19deg_rmse_less_than_fixed": varied_improves,
        },
    }


def _write_plot(path: Path, *, pose_points: Mapping[int, Sequence[Mapping[str, Any]]], selected_sites: Sequence[int]) -> None:
    """Write an offline 3-D view of the posterior and its lateral intervals."""

    traces: list[dict[str, Any]] = []
    colors = {11: "#2d6cdf", 15: "#c47a20", 19: "#a44262"}
    for angle in POSE_DEGREES:
        points = list(pose_points[angle])
        color = colors[angle]
        nominal = np.asarray([point["nominal_world_m"] for point in points], dtype=float)
        posterior = np.asarray([point["posterior_mean_world_m"] for point in points], dtype=float)
        lower = np.asarray([point["lower90_world_m"] for point in points], dtype=float)
        upper = np.asarray([point["upper90_world_m"] for point in points], dtype=float)
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "name": f"{angle}° nominal transported centerline",
                "x": nominal[:, 0].tolist(),
                "y": nominal[:, 1].tolist(),
                "z": nominal[:, 2].tolist(),
                "line": {"color": "#91a4ad", "width": 3, "dash": "dash"},
                "showlegend": False,
                "legendgroup": f"pose{angle}",
            }
        )
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines+markers",
                "name": f"{angle}°",
                "x": posterior[:, 0].tolist(),
                "y": posterior[:, 1].tolist(),
                "z": posterior[:, 2].tolist(),
                "line": {"color": color, "width": 6},
                "marker": {"color": color, "size": 3},
                "legendgroup": f"pose{angle}",
            }
        )
        i: list[int] = []
        j: list[int] = []
        k: list[int] = []
        # Vertex order is all upper points followed by all lower points.
        for index in range(len(points) - 1):
            i.extend((index, index + 1))
            j.extend((index + 1, len(points) + index + 1))
            k.extend((len(points) + index + 1, len(points) + index))
        vertices = np.vstack((upper, lower))
        traces.append(
            {
                "type": "mesh3d",
                "name": f"{angle}° 90% lateral ribbon",
                "x": vertices[:, 0].tolist(),
                "y": vertices[:, 1].tolist(),
                "z": vertices[:, 2].tolist(),
                "i": i,
                "j": j,
                "k": k,
                "color": color,
                "opacity": 0.28,
                "showlegend": False,
                "legendgroup": f"pose{angle}",
            }
        )
        chosen = np.asarray([points[site] for site in selected_sites], dtype=object)
        selected = np.asarray([points[site]["posterior_mean_world_m"] for site in selected_sites], dtype=float)
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": f"{angle}° queried addresses",
                "x": selected[:, 0].tolist(),
                "y": selected[:, 1].tolist(),
                "z": selected[:, 2].tolist(),
                "text": [
                    f"Arc length {point['s_mm']:.1f} mm<br>90% half-width {point['half_width90_mm']:.2f} mm"
                    for point in chosen
                ],
                "hovertemplate": "%{text}<extra></extra>",
                "marker": {"color": color, "size": 5, "symbol": "diamond"},
                "showlegend": False,
                "legendgroup": f"pose{angle}",
            }
        )
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        column_widths=[.60, .40],
        subplot_titles=("Posterior centerlines in world coordinates", "90% lateral interval half-width"),
    )
    for trace in traces:
        fig.add_trace(trace, row=1, col=1)
    for angle in POSE_DEGREES:
        points = pose_points[angle]
        fig.add_trace(go.Scatter(
            x=[point["s_mm"] for point in points],
            y=[point["half_width90_mm"] for point in points],
            name=f"{angle}°",
            legendgroup=f"pose{angle}",
            showlegend=False,
            line={"color": colors[angle], "width": 3},
        ), row=1, col=2)
    fig.update_xaxes(title_text="Material arc length (mm)", row=1, col=2)
    fig.update_yaxes(title_text="Half-width (mm)", rangemode="tozero", row=1, col=2)
    fig.update_layout(
        template="plotly_white",
        title="Same anatomy posterior at three knee angles<br><sup>Dashed lines show the nominal atlas</sup>",
        scene={
            "xaxis_title": "World X (m)",
            "yaxis_title": "World Y (m)",
            "zaxis_title": "World Z (m)",
            "aspectmode": "data",
            "camera": {
                "eye": {"x": 2.6, "y": 1.12, "z": 1.38},
            },
        },
        height=720,
        margin={"l": 25, "r": 30, "b": 85, "t": 110},
        legend={"orientation": "h", "y": -.12},
    )
    fig.write_html(path, include_plotlyjs=True)


def audit(*, out_dir: Path = DEFAULT_OUT, sites: Sequence[int] | None = None) -> dict[str, Any]:
    render = _load_replay_module()
    chapters = render.get_chapters()
    if len(chapters) < 3 or chapters[2].title != "Observation-guided B":
        raise AssertionError("render_search_replay.get_chapters() did not return the expected B case")
    chapter = chapters[2]
    geometry_path = Path(render.GEOMETRY)
    with np.load(geometry_path, allow_pickle=False) as geometry:
        scan = render.DynamicScan(geometry)
    selected_sites = _site_indices(scan, sites)

    # Reproduce the public posterior exactly as the replay renderer does.
    model = render.rollout.study.Recheck(scan, edge=None)
    beliefs, widths = render.belief_timeline(scan, chapter, model)
    if not beliefs:
        raise AssertionError("belief_timeline returned no posterior")
    selected_belief = beliefs[-1]
    covariance = np.asarray(selected_belief.covariance, dtype=float)
    if not np.allclose(covariance, covariance.T, atol=1.0e-10, rtol=0.0):
        raise AssertionError("selected posterior covariance is not symmetric")
    if float(np.linalg.eigvalsh(covariance).min()) < -1.0e-8:
        raise AssertionError("selected posterior covariance is not positive semidefinite")

    pose_by_degree = {int(round(float(angle))): int(index) for index, angle in enumerate(scan.angles)}
    if any(angle not in pose_by_degree for angle in POSE_DEGREES):
        raise AssertionError(f"calibrated pose knots do not contain {POSE_DEGREES}")
    all_points: dict[int, list[dict[str, Any]]] = {}
    selected_queries: list[dict[str, Any]] = []
    for angle in POSE_DEGREES:
        pose = pose_by_degree[angle]
        all_points[angle] = _all_pose_points(scan, selected_belief, pose)
        selected_queries.extend(
            [{**_query_point(scan, selected_belief, pose, site), "selected_address": True} for site in selected_sites]
        )

    # Existing sealed independent checks are re-aggregated here; no audit
    # script is executed and no synthetic truth is drawn.
    dynamic_root = geometry_path.parent
    pose_learning = _recompute_pose_learning_audit(dynamic_root)
    posterior_replay = dynamic_root.parent / "ab_fisher_continuation_20261112" / "POSTERIOR_REPLAY.json"
    posterior_replay_info: dict[str, Any] = {}
    if posterior_replay.is_file():
        sealed = json.loads(posterior_replay.read_text(encoding="utf-8"))
        posterior_replay_info = {
            "path": str(posterior_replay),
            "sha256": _sha256(posterior_replay),
            "scope": sealed.get("scope"),
            "seed": sealed.get("seed"),
            "zero_cross_covariance_control": sealed.get("zero_cross_covariance_control"),
        }

    cross_terms = np.asarray(
        [point["variance_components_mm2"]["cross_covariance"] for point in selected_queries], dtype=float
    )
    sd_by_angle = {
        str(angle): {
            "mean_sd_mm": float(np.mean([point["sd_mm"] for point in all_points[angle]])),
            "max_sd_mm": float(np.max([point["sd_mm"] for point in all_points[angle]])),
            "mean_half_width90_mm": float(np.mean([point["half_width90_mm"] for point in all_points[angle]])),
            "max_abs_cross_covariance_term_mm2": float(
                np.max(np.abs([point["variance_components_mm2"]["cross_covariance"] for point in all_points[angle]]))
            ),
        }
        for angle in POSE_DEGREES
    }
    summary: dict[str, Any] = {
        "format_version": 1,
        "scope": "same public B-case posterior queried at calibrated pose knots; no simulation",
        "source": {
            "render_search_replay": str(HERE / "render_search_replay.py"),
            "render_search_replay_sha256": _sha256(HERE / "render_search_replay.py"),
            "geometry": str(geometry_path),
            "geometry_sha256": _sha256(geometry_path),
            "chapter_title": chapter.title,
            "chapter_seed": int(chapter.seed),
            "prefix_steps": int(len(chapter.prefix)),
            "suffix_steps": int(len(chapter.suffix["steps"])),
            "public_log_fields": ["prefix_steps", "steps", "observation_mm", "centroids_mm", "evidence_gains"],
        },
        "selected_belief": {
            "selection": "final B-case public posterior from belief_timeline",
            "timeline_records": int(len(beliefs)),
            "ready_targets": int(selected_belief.ready_targets),
            "half_width_median_mm": float(selected_belief.half_width_mm),
            "timeline_widths_mm": _jsonable(widths),
            "mean_sha256": hashlib.sha256(np.asarray(selected_belief.mean, dtype=np.float64).tobytes()).hexdigest(),
            "covariance_sha256": hashlib.sha256(covariance.astype(np.float64).tobytes()).hexdigest(),
        },
        "query": {
            "pose_degrees": list(POSE_DEGREES),
            "material_sites": selected_sites,
            "z90": Z90,
            "formula": "Var[d_i(theta)] = C_ii + 2*g_i(theta)*C_i,slip + g_i(theta)^2*C_slip,slip",
            "signed_offset_convention": "positive along scan.frames[pose, site, 1]",
            "rows": selected_queries,
            "sd_summary_by_pose": sd_by_angle,
            "cross_covariance": {
                "max_abs_term_mm2": float(np.max(np.abs(cross_terms))),
                "mean_abs_term_mm2": float(np.mean(np.abs(cross_terms))),
                "nonzero_terms": int(np.count_nonzero(np.abs(cross_terms) > 1.0e-12)),
            },
        },
        "independent_checks": {
            "pose_learning_audit": pose_learning,
            "posterior_replay_sealed": posterior_replay_info,
            "belief_timeline_replay": True,
            "covariance_symmetric_psd": True,
            "no_truth_or_patient_simulation": True,
        },
        "interpretation": {
            "dynamic_confidence": "The same posterior produces pose-dependent world centerlines and SDs through the signed slip gain and cross-covariance; at 15° the calibrated slip gain is zero, while 11°/19° expose the learned pose mode.",
            "cheap_execution": "This audit performs logged posterior replay plus matrix/vector queries and writes one static HTML payload; it does not run a rollout or generate observations.",
        },
    }
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "SUMMARY.json").write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_plot(
        out_dir / "POSE_QUERY_3D.html",
        pose_points=all_points,
        selected_sites=selected_sites,
    )
    return summary


def _parse_sites(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sites", default=None, help="comma-separated material site indices")
    args = parser.parse_args(argv)
    summary = audit(out_dir=args.out, sites=_parse_sites(args.sites))
    print(json.dumps(_jsonable({"out": str(args.out.resolve()), "poses": POSE_DEGREES, "sites": summary["query"]["material_sites"]}), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke
    raise SystemExit(main())
