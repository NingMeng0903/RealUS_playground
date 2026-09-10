"""Detached frozen v7/v8 finite-area comparison; no robot or IPC connections.

Run with the repository Python environment, PYTHONPATH including rm75_control:
  python analysis_artifacts/contact_qp_v8_r2/compare.py --freeze
  python analysis_artifacts/contact_qp_v8_r2/compare.py --workers 2
Configuration is frozen before execution; this is exploratory synthetic evidence.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import time
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from peirastic.apps import contact_qp_experiment as experiment
from peirastic.contact_qp.features import FeatureConfig
from peirastic.contact_qp.plant import PlantConfig
from peirastic.contact_qp.qp import QpConfig

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
POLICIES = ("legacy_v7", "differential_repair_v8")
SCENARIOS = ("healthy", "left_gap", "right_gap", "shadow", "delayed_execution")
SEEDS = (801, 802)


def settings(policy):
    plant = PlantConfig(scan_speed_m_s=.005, image_width=145, image_height=100)
    features = FeatureConfig(algorithm_version="randomwalk_welleweerd2020_v3",
        width=145, height=100, low_confidence_threshold=.8,
        effective_delay_s=plant.image_delay_s, calibration_version="synthetic_v1")
    qp = QpConfig(allocation_policy=policy, c_min=.8, angle_limit_rad=.35,
        lateral_windows=features.lateral_windows,
        quality_policy_version="synthetic_v3_cmin08_v8_comparison_unverified")
    return plant, features, qp


def protocol():
    return dict(seeds=SEEDS, scenarios=SCENARIOS, policies=POLICIES,
        regression_only=dict(seeds=(701,702),scenarios=('shadow',),purpose='prior failure regression, not independent acceptance'),
        configs={p:[experiment.plain(asdict(c)) for c in settings(p)] for p in POLICIES},
        energy="off: run_case has no energy input; joint full-port checks are separate",
        nominal="unchanged build_contact_nominal, 4 N, same FeatureExtractor/run_case(full)",
        limits="Exploratory synthetic runs, no acceptance tuning, hardware or real-data efficacy claim")


def run_one(job):
    scenario, seed, policy = job
    experiment.settings = lambda: settings(policy)
    start = time.perf_counter()
    destination = HERE / "runs" / scenario / str(seed)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        result, trace = experiment.run_case(scenario, seed, "full")
        np.savez_compressed(destination / (policy + ".npz"), **trace)
        force = trace["force_n"]
        scan = trace["phase"] == "scan"
        quality = np.column_stack([trace["quality_left"],trace["quality_center"],trace["quality_right"]])
        result.update(policy=policy, force_peak_n=float(force.max()) if force.size else None,
            physical_force_peak_n=float(trace['physical_force_n'].max()) if force.size else None,
            physical_force_rmse_n=float(np.sqrt(np.mean((trace['physical_force_n']-4)**2))) if force.size else None,
            force_above_4p5_s=float(np.sum(force > 4.5)*.005),
            force_above_6_s=float(np.sum(force > 6)*.005),
            scan_force_rmse_n=float(np.sqrt(np.mean((force[scan]-4)**2))) if scan.any() else None,
            quality_mean_lcr=quality[scan].mean(axis=0).tolist() if scan.any() else None,
            image_both_edges_ge_cmin_fraction=float(np.mean(np.min(quality[scan][:,[0,2]],axis=1)>=.8)) if scan.any() else None,
            scan_ticks=int(scan.sum()), energy_enabled=False)
    except Exception as exc:
        result=dict(scenario=scenario,seed=seed,policy=policy,status="aborted",valid=False,
                    reason="experiment_exception: "+repr(exc),wall_s=time.perf_counter()-start)
    (destination / (policy + ".json")).write_text(json.dumps(experiment.plain(result),indent=2))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--freeze",action="store_true")
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--regression",action="store_true")
    args=parser.parse_args()
    frozen=HERE/"protocol.json"
    current=json.loads(json.dumps(experiment.plain(protocol())))
    if args.freeze:
        if frozen.exists():raise RuntimeError("protocol already frozen")
        frozen.write_text(json.dumps(current,indent=2))
        print("Frozen 20 units: 5 scenarios x 2 seeds x 2 policies",flush=True)
        return
    if json.loads(frozen.read_text()) != current:raise RuntimeError("frozen configuration changed")
    prefix='regression_' if args.regression else ''
    if (HERE/(prefix+"source_hashes_start.json")).exists():raise RuntimeError("run already started")
    files=sorted(set([f for f in (ROOT/'peirastic').rglob('*.py') if 'tests' not in f.parts]
        +list((ROOT/'peirastic/configs').glob('*.yaml'))
        +list((ROOT/'rm75_control/rm75_control').rglob('*.py'))
        +[HERE/name for name in ('compare.py','energy_sweep.py','report.py')]))
    hashes=lambda:{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    initial=hashes()
    archive_path=HERE/(prefix+'frozen_source.zip')
    with ZipFile(archive_path,'x',compression=ZIP_DEFLATED) as archive:
        for file in files:
            relative=str(file.relative_to(ROOT));content=file.read_bytes()
            if hashlib.sha256(content).hexdigest()!=initial[relative]:
                raise RuntimeError('source changed during snapshot: '+relative)
            archive.writestr(relative,content)
        archive.writestr('protocol.json',frozen.read_bytes())
    (HERE/(prefix+"source_hashes_start.json")).write_text(json.dumps(initial,indent=2))
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[(s,k,p) for s in (('shadow',) if args.regression else SCENARIOS)
              for k in ((701,702) if args.regression else SEEDS) for p in POLICIES]
        for future in as_completed([pool.submit(run_one,j) for j in jobs]):
            result=future.result();results.append(result)
            (HERE/(prefix+"metrics.json")).write_text(json.dumps(experiment.plain(results),indent=2))
            print(f"{len(results)}/{len(jobs)} {result['scenario']} {result['seed']} {result['policy']}: {result['status']} {result.get('reason','')}",flush=True)
    (HERE/(prefix+"source_hashes_end.json")).write_text(json.dumps(hashes(),indent=2))


if __name__ == "__main__":main()
