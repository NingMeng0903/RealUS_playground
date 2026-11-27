"""Generate independent gallbladder worlds and replay three paired controllers."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from gallbladder_belief import Noise, ORGAN_LIMIT, SENSOR
from gallbladder_geometry import ATLAS_PATH, Body, Reference
from gallbladder_mpc import ARMS, CONTACT, FRAME_DT, FRAME_LIMIT, Planner, probe_geometry
from model import SMPLX, SOURCE


HERE = Path(__file__).resolve().parent
DEFAULT = HERE / "results" / "gallbladder_prescreen"
SEEDS = {"pilot": 58_640_000, "dev": 58_641_000, "holdout": 58_642_000}
SIZES = {"pilot": 6, "dev": 48, "holdout": 192}
REFERENCE = None


@dataclass(frozen=True)
class Case:
    id: int
    beta: np.ndarray
    organ: np.ndarray
    rib: np.ndarray
    start: np.ndarray
    attempts: int


def initialize():
    global REFERENCE
    REFERENCE = Reference()


def generate(seed):
    rng = np.random.default_rng(seed)
    # Across-case morphology stays independent of window class and outcome.
    beta = rng.uniform([-1.55, -.84], [1.55, .84])
    body = Body(REFERENCE, beta)
    start = np.clip(np.array([rng.uniform(-207., -176.), rng.uniform(-7.5, 7.5),
                              rng.choice([-1., 1.]) * rng.uniform(46., 72.), rng.uniform(-11., 11.)]), *CONTACT)
    for attempt in range(1, 181):
        organ = rng.uniform(-1., 1., len(ORGAN_LIMIT)) * ORGAN_LIMIT
        rib = rng.uniform([-3.2, -2.2], [3.2, 2.2])
        anatomy = body.anatomy(organ)
        if body.valid(anatomy, rib):
            return Case(seed, beta, organ, rib, start, attempt), body, anatomy
    raise RuntimeError(f"no geometrically valid draw for shape {beta.tolist()} in {seed}")


def finite(array):
    return [float(x) if np.isfinite(x) else None for x in np.asarray(array).ravel()]


def run_root(seed, frames):
    case, body, anatomy = generate(seed)
    output = {
        "seed": seed, "beta_delta": case.beta.tolist(), "organ_params": case.organ.tolist(),
        "rib_shift_mm": case.rib.tolist(), "construction_attempts": case.attempts,
        "start_chart": case.start.tolist(), "start_contact_mm": body.surface(case.start).tolist(),
        "true_gallbladder_center_mm": anatomy.center.tolist(),
        "skin_deformation_rms_mm": body.skin_change_rms_mm,
        "skin_deformation_max_mm": body.skin_change_max_mm,
        "runs": {},
    }
    for copy, condition in enumerate(SENSOR):
        planner = Planner(body, case.start, case, condition)
        noise = Noise(seed, copy, condition, frames)
        root = planner.prescan(anatomy, case.start, noise, case.rib)
        if copy == 0:
            output["geometry"] = probe_geometry(body, planner, anatomy, case.rib)
            output["entry_aims"] = planner.charts.tolist()
            output["entry_family"] = planner.family.tolist()
        arms = {arm: planner.run(anatomy, noise, case.rib, root, arm, frames=frames, trace=True) for arm in ARMS}
        output["runs"][condition] = {
            "public": planner.public_diagnostic(case.start, root),
            "prescan": [{"time_s": frame.tick * FRAME_DT, "observations": finite(frame.readout),
                         "posterior": frame.posterior.tolist(), "geometrically_good": frame.geometry_pass}
                        for frame in root],
            "arms": arms,
        }
    return output


def compare(runs, reference, treatment):
    base = np.asarray([r[reference]["success"] for r in runs], dtype=int)
    candidate = np.asarray([r[treatment]["success"] for r in runs], dtype=int)
    delta = candidate - base
    rng = np.random.default_rng(713)
    draws = rng.integers(len(delta), size=(4096, len(delta)))
    bounds = np.percentile(delta[draws].mean(axis=1), [2.5, 97.5]) * 100.
    return {"delta_pp": float(100. * delta.mean()), "interval_pp": bounds.tolist(),
            "rescues": int((delta > 0).sum()), "harms": int((delta < 0).sum())}


def arm_summary(rows):
    travelled = [r["travel_mm"] for r in rows]
    first_time = [r["first_quality_time_s"] for r in rows if r["acquired"]]
    first_distance = [r["first_quality_travel_mm"] for r in rows if r["acquired"]]
    return {"success": sum(r["success"] for r in rows), "wrong": sum(r["wrong"] for r in rows),
            "acquired": sum(r["acquired"] for r in rows),
            "unconfirmed": sum(r["stopped_without_confirmation"] for r in rows),
            "requested_approach_available": sum(r["requested_approach_available"] for r in rows),
            "mean_travel_mm": float(np.mean(travelled)), "p90_travel_mm": float(np.percentile(travelled, 90)),
            "mean_turn_deg": float(np.mean([r["turn_deg"] for r in rows])),
            "mean_time_s": float(np.mean([r["time_s"] for r in rows])),
            "median_first_quality_time_s": float(np.median(first_time)) if first_time else None,
            "median_first_quality_travel_mm": float(np.median(first_distance)) if first_distance else None,
            "quality_frame_by_seconds": {str(s): sum(t <= s for t in first_time) for s in (2, 4, 6, 8, 12, 18, 24, 32)},
            "quality_frame_by_mm": {str(mm): sum(d <= mm for d in first_distance) for mm in (20, 40, 60, 90, 120, 180)},
            "max_join_position_error": max(r["join_position_error"] for r in rows),
            "max_join_velocity_error": max(r["join_velocity_error"] for r in rows)}


def pick_public(data):
    public = data["public"]
    return "subcostal" if public["subcostal_approach_score"] >= public["intercostal_approach_score"] else "intercostal"


def match_direct_forecast(data, arms):
    """Check for common support without looking at any branch's outcome."""
    start = np.array([r["public"]["direct_forecast"] for r in data])
    selected = np.array([pick_public(r) == "subcostal" for r in data])
    left, right = np.flatnonzero(selected), np.flatnonzero(~selected)
    candidates = sorted((abs(start[i] - start[j]), int(i), int(j)) for i in left for j in right
                        if abs(start[i] - start[j]) <= .08)
    used = set()
    pairs = []
    for _, i, j in candidates:
        if i in used or j in used:
            continue
        used.update((i, j))
        pairs.append((i, j))
    summary = {"pairs": len(pairs), "maximum_forecast_gap": .08}
    if pairs:
        l, r = np.asarray(pairs).T
        samples = [[arms[k] for k in group] for group in (l, r)]
        summary.update({"mean_forecast_difference": float((start[l] - start[r]).mean()),
                        "actual_direct_success": [sum(a["direct"]["success"] for a in sample) for sample in samples],
                        "window_differences": [compare(sample, "intercostal", "subcostal") for sample in samples]})
    return summary


def summarize(rows, best_fixed):
    trials = len(rows)
    geometry = np.array([[family["reachable_contacts"] for family in result["geometry"]] for result in rows])
    left, right = geometry[:, 1] > 0, geometry[:, 2] > 0
    changes = np.asarray([row["organ_params"] for row in rows])
    report = {
        "roots": trials,
        "conditions": {},
        "geometry": {"both": int((left & right).sum()), "subcostal_only": int((left & ~right).sum()),
                     "intercostal_only": int((~left & right).sum()), "neither": int((~left & ~right).sum()),
                     "midline_or_oblique_available": int((geometry[:, 0] > 0).sum()),
                     "median_valid_contacts": np.median(geometry, axis=0).tolist()},
        "generator": {"rejected_anatomy": sum(r["construction_attempts"] - 1 for r in rows),
                      "organ_params_min": changes.min(axis=0).tolist(), "organ_params_max": changes.max(axis=0).tolist(),
                      "skin_rms_range_mm": [min(r["skin_deformation_rms_mm"] for r in rows),
                                            max(r["skin_deformation_rms_mm"] for r in rows)]},
    }
    for condition in SENSOR:
        data = [row["runs"][condition] for row in rows]
        arms = [dict(r["arms"], frozen_public=r["arms"][pick_public(r)], best_fixed=r["arms"][best_fixed[condition]])
                for r in data]
        groups = {}
        for arm in ARMS[1:]:
            subset = [a for a, r in zip(arms, data) if pick_public(r) == arm]
            groups[arm] = {"n": len(subset)}
            if subset:
                groups[arm]["window_difference"] = compare(subset, "intercostal", "subcostal")
                groups[arm]["direct_success"] = sum(a["direct"]["success"] for a in subset)
        for_later = [a for a in arms if not a["direct"]["success"]]
        rescue = sum(a["subcostal"]["success"] or a["intercostal"]["success"] for a in for_later)
        report["conditions"][condition] = {
            "arms": {arm: arm_summary([a[arm] for a in arms]) for arm in (*ARMS, "best_fixed", "frozen_public")},
            "best_fixed_arm": best_fixed[condition], "public_subgroups": groups,
            "common_direct_forecast_support": match_direct_forecast(data, arms),
            "comparison_vs_direct": {arm: compare(arms, "direct", arm) for arm in (*ARMS[1:], "frozen_public")},
            "comparison_public_vs_best_fixed": compare(arms, "best_fixed", "frozen_public"),
            "potential_rescues_among_direct_failures": rescue,
            "direct_failures": len(for_later),
            "window_pair": compare(arms, "intercostal", "subcostal"),
        }
    return report


def hash_sources():
    inputs = sorted(HERE.glob("gallbladder_*.py"))
    inputs.extend((HERE / "GALLBLADDER_PRESCREEN_PROTOCOL.md",
                   ATLAS_PATH,
                   HERE.parent / "outputs" / "thorax_preview" / "generate.py",
                   HERE.parent / "outputs" / "thorax_continuous" / "model.py",
                   HERE.parent / "outputs" / "thorax_continuous" / "servo.py",
                   HERE.parent / "outputs" / "thorax_a4c_pilot" / "geometry.py", SMPLX, SOURCE))
    digests = {}
    for path in inputs:
        label = path.relative_to(HERE.parent) if path.is_relative_to(HERE.parent) else path
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digests[str(label)] = digest.hexdigest()
    return digests


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=SEEDS)
    parser.add_argument("--roots", type=int)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--frames", type=int, default=FRAME_LIMIT)
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    roots = args.roots or SIZES[args.stage]
    directory = args.out / args.stage
    directory.mkdir(parents=True, exist_ok=True)
    frozen = args.out / "frozen.json"
    source = hash_sources()
    if args.stage == "holdout":
        config = json.loads(frozen.read_text())
        if config["sources"] != source or config["frames"] != args.frames:
            raise RuntimeError("source or acquisition horizon changed since development freeze")
    else:
        config = {"sources": source, "frames": args.frames}
    seeds = range(SEEDS[args.stage], SEEDS[args.stage] + roots)
    results = []
    temporary = directory / "cases.jsonl.gz.part"
    with gzip.open(temporary, "wt") as stream, ProcessPoolExecutor(
            max_workers=args.workers, initializer=initialize) as executor:
        pending = [executor.submit(run_root, seed, args.frames) for seed in seeds]
        for finished in as_completed(pending):
            result = finished.result()
            stream.write(json.dumps(result, allow_nan=False, separators=(",", ":")) + "\n")
            # Trajectories remain in the archive. Keep only statistics in RAM.
            result.pop("entry_aims")
            for run in result["runs"].values():
                for arm in run["arms"].values():
                    arm.pop("trace")
            results.append(result)
            if len(results) == 1 or len(results) % 8 == 0 or len(results) == roots:
                print(f"{args.stage}: {len(results)}/{roots} paired anatomies", flush=True)
    temporary.replace(directory / "cases.jsonl.gz")
    results.sort(key=lambda row: row["seed"])
    if args.stage != "holdout":
        def fixed_rank(arm, condition):
            values = [r["runs"][condition]["arms"][arm] for r in results]
            return (sum(v["success"] for v in values), -sum(v["wrong"] for v in values),
                    -sum(v["time_s"] for v in values), -sum(v["travel_mm"] for v in values))
        config["best_fixed"] = {c: max(ARMS[1:], key=lambda a: fixed_rank(a, c)) for c in SENSOR}
        if args.stage == "dev":
            config["holdout_seed_range"] = [SEEDS["holdout"], SEEDS["holdout"] + SIZES["holdout"] - 1]
            dump(frozen, config)
    summary = summarize(results, config["best_fixed"])
    summary["settings"] = {"stage": args.stage, "frames": args.frames, "seeds": list(seeds),
                           "sources": source, "arms": ARMS}
    dump(directory / "summary.json", summary)


if __name__ == "__main__":
    main()
