"""Descriptive design-only reporting. Never imports or runs a controller/plant.

Read immutable experiment metrics/NPZ; write only MD report/artifacts. The
acceptance seeds are explicitly forbidden. Failed/missing runs remain visible.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "MD/contact_qp"
ACCEPTANCE = json.loads((BASE / "acceptance_v1.json").read_text())
SCENARIOS, VARIANTS = ACCEPTANCE["scenarios"], ACCEPTANCE["ablations"]
COMMON = ["baseline", "full", "matched_scan_speed", "matched_rocking"]
COLORS = dict(zip(COMMON, ["#777777", "#0072b2", "#e69f00", "#009e73"]))
LABELS = {"baseline": "Baseline", "full": "Full", "matched_scan_speed": "Speed matched",
          "matched_rocking": "Rocking matched", "image_only": "Image only",
          "no_aperture": "No aperture", "full_consistency": "Full + consistency"}
METRICS = ["rmse", "absolute_mean_bias", "peak_absolute_error", "bad_path_m",
           "valid_coverage_rate_m_s", "valid_coverage_fraction", "registered_image_coverage_m",
           "acoustic_coverage_fraction", "angular_travel_rad", "measured_forward_m", "elapsed_s",
           "force_active_fraction", "aperture_active_fraction", "omega_width_mechanical_mean",
           "omega_width_force_mean", "omega_width_complete_mean", "controller_p50_ms",
           "controller_p99_ms", "controller_max_ms", "simulated_deadline_overruns"]


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def finite(values):
    return np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)


def stats(values):
    a = finite(values)
    return dict(n=len(a), mean=float(a.mean()) if len(a) else None,
                median=float(np.median(a)) if len(a) else None,
                min=float(a.min()) if len(a) else None, max=float(a.max()) if len(a) else None)


def write_json(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(clean(v), ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else clean(v)
                             for k, v in row.items()})


def recovery_events(trace):
    """Join bad intervals until both edges stay >=.90 for .2 s; retain censoring."""
    t = np.asarray(trace["time_s"], dtype=float)
    good = np.minimum(trace["acoustic_left"], trace["acoustic_right"]) >= .90
    phase = trace["phase"] if "phase" in trace else np.full(len(t), "scan")
    events, onset, recovery = [], None, None
    for i in range(len(t)):
        if onset is None:
            if not good[i]: onset = i
            continue
        if not good[i]: recovery = None
        elif recovery is None: recovery = i
        if recovery is not None and t[i]-t[recovery] >= .2-1e-10:
            events.append(dict(onset_time_s=t[onset], onset_path_m=float(trace["path_m"][onset]),
                               left_censored=bool(onset == 0), recovered=True, right_censored=False,
                               recovery_start_s=t[recovery], confirmed_s=t[i],
                               duration_s=t[recovery]-t[onset],
                               confirmed_phase=str(phase[i]), observation_end_s=t[-1]))
            onset, recovery = None, None
    if onset is not None:
        events.append(dict(onset_time_s=t[onset], onset_path_m=float(trace["path_m"][onset]),
                           left_censored=bool(onset == 0), recovered=False, right_censored=True,
                           recovery_start_s=None, confirmed_s=None, duration_s=None,
                           observed_followup_s=t[-1]-t[onset], confirmed_phase=None,
                           observation_end_s=t[-1]))
    return events


def trace_metrics(trace, dt):
    n = len(trace["time_s"])
    for name in ("time_s", "path_m", "force_n", "omega_measured", "theta", "acoustic_left", "acoustic_right"):
        if len(trace[name]) != n or not np.isfinite(trace[name]).all():
            raise ValueError("invalid trace channel: " + name)
    if n == 0 or np.any(np.diff(trace["time_s"]) <= 0):
        raise ValueError("empty or non-increasing trace time")
    scan = trace["phase"] == "scan" if "phase" in trace else np.ones(n, dtype=bool)
    t = trace["time_s"]
    ds = np.diff(np.r_[trace["path_m"][0], trace["path_m"]]) if n else np.array([])
    events = recovery_events(trace)
    result = dict(trace_samples=n, scan_samples=int(scan.sum()), stop_tail_samples=int((~scan).sum()),
                  measured_positive_travel_m=float(np.maximum(ds, 0).sum()),
                  measured_reverse_travel_m=float(np.maximum(-ds, 0).sum()),
                  scan_angular_travel_rad=float(np.abs(trace["omega_measured"][scan]).sum()*dt),
                  scan_max_abs_angle_rad=float(np.abs(trace["theta"][scan]).max()) if scan.any() else None,
                  scan_end_s=float(t[scan][-1]) if scan.any() else None,
                  scan_controller_p95_ms=float(np.quantile(trace["controller_ms"][scan],.95)) if scan.any() else None,
                  scan_force_active_fraction=float(np.mean(trace["force_constraint_active"][scan])) if scan.any() else None,
                  scan_aperture_active_fraction=float(np.mean(trace["aperture_active"][scan])) if scan.any() else None,
                  scan_required_acoustic_bad_fraction=float(np.mean(np.minimum(trace["acoustic_left"][scan], trace["acoustic_right"][scan]) < .9)) if scan.any() else None,
                  recovery_events=len(events), recovered_events=sum(e["recovered"] for e in events),
                  unconfirmed_events=sum(not e["recovered"] for e in events), no_acoustic_deficit=not events,
                  recovery_confirmed_during_scan=sum(e["confirmed_phase"] == "scan" for e in events),
                  recovery_confirmed_during_stop_tail=sum(e["confirmed_phase"] == "stop_tail" for e in events))
    if "acoustic_center" in trace:
        result["scan_center_acoustic_bad_fraction"] = float(np.mean(trace["acoustic_center"][scan] < .9))
    for name in ("mechanical", "force", "complete"):
        values = trace["omega_width_" + name]
        valid = scan & np.isfinite(values) & (values >= 0)
        result["scan_omega_width_" + name + "_mean"] = float(values[valid].mean()) if valid.any() else None
        result["scan_omega_width_" + name + "_samples"] = int(valid.sum())
    return result, events


def collect(source, out):
    manifest = json.loads((source / "manifest.json").read_text())
    source_files_verified = 0
    if (source / "sources.zip").exists():
        with ZipFile(source / "sources.zip") as archive:
            if set(archive.namelist()) != set(manifest["hashes"]):
                raise ValueError("source snapshot file manifest differs")
            for name, expected in manifest["hashes"].items():
                if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                    raise ValueError("source snapshot hash differs: " + name)
                source_files_verified += 1
    seeds = ACCEPTANCE["design_seeds"]
    rows, events, input_hashes = [], [], {}
    metrics_checked, maximum_metric_error = 0, 0.
    files = sorted(source.glob("*/*/metrics.json"))
    for p in files:
        records = json.loads(p.read_text())
        for row in records:
            assert row["seed"] in seeds and row["scenario"] in SCENARIOS and row["variant"] in VARIANTS
            trace_path = p.parent / (row["variant"] + ".npz")
            row = dict(row, source_metrics=str(p.relative_to(ROOT)), trace_available=trace_path.exists())
            if trace_path.exists():
                with np.load(trace_path, allow_pickle=False) as trace:
                    extra, ev = trace_metrics(trace, manifest["plant"]["dt_s"])
                    error = trace["force_n"] - 4.
                    recalculated = np.array([np.sqrt(np.mean(error**2)), abs(np.mean(error)),
                                             np.max(abs(error)), np.sum(abs(trace["omega_measured"]))*manifest["plant"]["dt_s"]])
                    saved = np.array([row[k] for k in ("rmse","absolute_mean_bias","peak_absolute_error","angular_travel_rad")])
                    discrepancy = float(np.max(abs(recalculated-saved)))
                    if discrepancy > 1e-12:
                        raise ValueError("metric/trace mismatch: " + str(trace_path))
                    if abs(len(trace["time_s"])*manifest["plant"]["dt_s"]-row["elapsed_s"]) > 1e-12:
                        raise ValueError("elapsed/trace mismatch: " + str(trace_path))
                    maximum_metric_error = max(maximum_metric_error, discrepancy)
                    metrics_checked += 1
                row.update(extra)
                for e in ev:
                    events.append(dict(scenario=row["scenario"], seed=row["seed"], variant=row["variant"], **e))
            rows.append(row)
    expected = {(s,k,v) for s in SCENARIOS for k in seeds for v in VARIANTS}
    identities = [(r["scenario"],r["seed"],r["variant"]) for r in rows]
    if len(identities) != len(set(identities)): raise ValueError("duplicate run identities")
    failures = [{"file": p.name, "reason": p.read_text()} for p in sorted(source.glob("failure_*.txt"))]
    for p in sorted(source.rglob("*")):
        if p.is_file() and (p.suffix in (".json", ".npz", ".txt", ".zip")):
            input_hashes[str(p.relative_to(source))] = hashlib.sha256(p.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(input_hashes, sort_keys=True).encode()).hexdigest()
    means = []
    derived = sorted({k for r in rows for k in r if k.startswith("scan_") or k.startswith("measured_")})
    for s in SCENARIOS:
        for v in VARIANTS:
            subset = [r for r in rows if r["scenario"] == s and r["variant"] == v]
            record = dict(scenario=s, variant=v, recorded_n=len(subset), expected_n=len(seeds),
                          valid_n=sum(bool(r.get("valid")) for r in subset),
                          complete_n=sum(r["status"] == "complete" for r in subset),
                          gaps_n=sum(r["status"] == "complete_with_gaps" for r in subset),
                          aborted_n=sum(r["status"] == "aborted" for r in subset), missing_n=len(seeds)-len(subset))
            for m in dict.fromkeys(METRICS+derived):
                summary = stats(r.get(m) for r in subset)
                for k in ("mean", "min", "max", "n"): record[m + "__" + k] = summary[k]
            ev = [e for e in events if e["scenario"] == s and e["variant"] == v]
            record.update(recovery_events=len(ev), recovery_confirmed=sum(e["recovered"] for e in ev),
                          recovery_unconfirmed=sum(not e["recovered"] for e in ev),
                          recovery_no_deficit_runs=sum(r.get("no_acoustic_deficit", False) for r in subset),
                          recovery_median_s=stats(e["duration_s"] for e in ev)["median"])
            means.append(record)
    index = {key: r for key, r in zip(identities, rows)}
    comparisons, ratios = [], []
    for s in SCENARIOS:
        for v in COMMON:
            if v == "full": continue
            pairs = [(index[s,k,"full"],index[s,k,v]) for k in seeds if (s,k,"full") in index and (s,k,v) in index]
            c = dict(scenario=s, comparator=v, recorded_pairs=len(pairs), expected_pairs=len(seeds),
                     failed_pairs=sum(not a.get("valid",False) or not b.get("valid",False) for a,b in pairs),
                     interpretation="descriptive recorded-pair differences; no inferential acceptance")
            for m in METRICS:
                c["delta_"+m] = stats(a.get(m)-b.get(m) for a,b in pairs if a.get(m) is not None and b.get(m) is not None)["mean"]
            comparisons.append(c)
            if v.startswith("matched"):
                for a,b in pairs:
                    rr=dict(scenario=s,seed=a["seed"],comparator=v,full_valid=a.get("valid"),comparator_valid=b.get("valid"))
                    for name in ("angular_travel_rad","scan_angular_travel_rad","measured_positive_travel_m","measured_forward_m","elapsed_s"):
                        numerator,denominator=b.get(name),a.get(name)
                        rr[name+"_ratio_to_full"] = numerator/denominator if numerator is not None and denominator is not None and denominator > 1e-12 else None
                        rr[name+"_difference_from_full"] = numerator-denominator if numerator is not None and denominator is not None else None
                    ratios.append(rr)
    summary = dict(schema_version=1,source=str(source.relative_to(ROOT)),label=source.name,
                   scope="DESIGN_ONLY_NOT_HELD_OUT",source_has_final_metrics=(source/"metrics.json").exists(),
                   recorded_runs=len(rows),expected_runs=len(expected),valid_runs=sum(bool(r.get("valid")) for r in rows),
                   states=dict(Counter(r["status"] for r in rows)),
                   missing=[dict(scenario=s,seed=k,variant=v) for s,k,v in sorted(expected-set(identities))],
                   pair_exceptions=failures,manifest_sha256=input_hashes.get("manifest.json"),data_sha256=digest,
                   source_snapshot_sha256=input_hashes.get("sources.zip"),
                   verification=dict(source_files_verified=source_files_verified,run_metrics_verified=metrics_checked,
                                     maximum_metric_absolute_error=maximum_metric_error),
                   analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   model_hash=manifest["hashes"].get("peirastic/contact_qp/plant.py"),
                   qp_hash=manifest["hashes"].get("peirastic/contact_qp/qp.py"),
                   runner_hash=manifest["hashes"].get("peirastic/apps/contact_qp_experiment.py"),
                   settings={k:manifest[k] for k in ("plant","features","qp","library_versions")},
                   recovery=dict(events=len(events),confirmed=sum(e["recovered"] for e in events),
                                 unconfirmed=sum(not e["recovered"] for e in events)),
                   scenario_means=means,paired_descriptive=comparisons)
    out.mkdir(parents=True,exist_ok=True)
    (out/"report_design_snapshot.py.txt").write_bytes(Path(__file__).read_bytes())
    write_json(out/"summary.json",summary)
    write_json(out/"input_hashes.json",input_hashes)
    write_csv(out/"runs.csv",rows)
    write_csv(out/"scenario_ablation.csv",means)
    write_csv(out/"paired_descriptive.csv",comparisons)
    write_csv(out/"matched_motion.csv",ratios)
    write_csv(out/"recovery_events.csv",events)
    return summary,rows,ratios


def plots(source,out,summary,rows,ratios):
    lookup={(r["scenario"],r["variant"]):r for r in summary["scenario_means"]}
    fig,axes=plt.subplots(2,3,figsize=(15,10),constrained_layout=True)
    panels=[("rmse",1,"Force RMSE (N)"),("absolute_mean_bias",1,"Absolute force bias (N)"),
            ("peak_absolute_error",1,"Peak force error (N)"),("bad_path_m",1000,"Mechanical missing path (mm)"),
            ("valid_coverage_rate_m_s",1000,"Valid mechanical rate (mm/s)"),
            ("registered_image_coverage_m",1000,"Registered acoustic image coverage (mm)")]
    for ax,(m,factor,title) in zip(axes.flat,panels):
        a=np.array([[lookup[s,v].get(m+"__mean") or 0 if lookup[s,v].get(m+"__mean") is not None else np.nan for v in COMMON] for s in SCENARIOS])*factor
        im=ax.imshow(np.ma.masked_invalid(a),aspect="auto",cmap="viridis")
        for i,s in enumerate(SCENARIOS):
            for j,v in enumerate(COMMON):
                r=lookup[s,v]
                if np.isfinite(a[i,j]):
                    value=f"{a[i,j]:.3f}" if factor==1 else f"{a[i,j]:.1f}"
                    if r["aborted_n"] or r["missing_n"]: value+="*"
                    ax.text(j,i,value,ha="center",va="center",fontsize=8,color="white" if im.norm(a[i,j])<.6 else "black")
        ax.set_xticks(range(4),[LABELS[v] for v in COMMON],rotation=25,ha="right")
        ax.set_yticks(range(12),SCENARIOS);ax.set_title(title)
        fig.colorbar(im,ax=ax,shrink=.75)
    fig.suptitle(f"{source.name}: means of logged runs; * aborted or missing runs; design only, no confidence intervals",fontsize=12)
    fig.savefig(out/"common_comparators.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
    x=np.arange(len(SCENARIOS))
    for j,(m,label) in enumerate([( "scan_omega_width_mechanical_mean","Mechanical"),("scan_omega_width_force_mean","+ Force priority"),("scan_omega_width_complete_mean","+ Aperture")]):
        y=[lookup[s,"full"].get(m+"__mean") for s in SCENARIOS]
        axes[0].plot(x,[np.nan if z is None else z for z in y],marker="o",label=label)
    axes[0].set(ylabel="Mean feasible angular width (rad/s)",title="Full: scan-only sampled feasible sets",ylim=(0,.033));axes[0].legend(fontsize=8)
    for m,label in [("scan_force_active_fraction","Force rows"),("scan_aperture_active_fraction","Aperture rows")]:
        axes[1].plot(x,[lookup[s,"full"].get(m+"__mean") for s in SCENARIOS],marker="o",label=label)
    axes[1].set(ylabel="Fraction of scan ticks",title="Full: active hard constraints");axes[1].legend(fontsize=8)
    for v in ("full","baseline","matched_rocking"):
        axes[2].plot(x,[lookup[s,v].get("angular_travel_rad__mean") for s in SCENARIOS],marker="o",label=LABELS[v])
    axes[2].set(ylabel="Measured total angular travel (rad)",title="Includes bounded stopping tail");axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x,SCENARIOS,rotation=70,ha="right");ax.grid(alpha=.2)
    fig.suptitle(f"{source.name}: descriptive constraint and measured-motion diagnostics")
    fig.savefig(out/"constraints_and_rotation.png",dpi=170);plt.close(fig)
    for scenario in ("left_gap","both_edges","shadow","delayed_execution"):
        fig,axes=plt.subplots(2,2,figsize=(10.5,6.5),constrained_layout=True)
        count=0
        for v in COMMON:
            p=source/scenario/"0"/(v+".npz")
            if not p.exists():continue
            count+=1
            with np.load(p,allow_pickle=False) as t:
                time=t["time_s"];color=COLORS[v];label=LABELS[v]
                axes[0,0].plot(time,t["force_n"],color=color,label=label,lw=1)
                axes[0,1].plot(time,t["path_m"]*1000,color=color,label=label,lw=1)
                axes[1,0].plot(time,np.minimum(t["acoustic_left"],t["acoustic_right"]),color=color,label=label,lw=1)
                axes[1,1].plot(time,t["omega_measured"],color=color,label=label,lw=1)
                if v=="full":
                    axes[1,0].plot(time,np.minimum(t["coupling_left"],t["coupling_right"]),color="black",ls="--",lw=.8,label="Full mechanical")
                    scan=t["phase"]=="scan"
                    for ax in axes.flat:ax.axvspan(time[scan][-1],time[-1],color=color,alpha=.07)
        if not count:plt.close(fig);continue
        axes[0,0].axhline(4,color="black",ls=":",lw=.8)
        axes[0,1].axhline(60,color="black",ls=":",lw=.8)
        axes[1,0].axhline(.9,color="black",ls=":",lw=.8)
        axes[0,0].set(ylabel="Current force (N)",title="Includes scan transients and stopping tail")
        axes[0,1].set(ylabel="Measured path (mm)",title="No reference-as-coverage substitution")
        axes[1,0].set(ylabel="Worse required acoustic fraction",ylim=(-.05,1.05),title="Latent truth; confidence not used as truth")
        axes[1,1].set(ylabel="Measured rocking (rad/s)",title="Command acceptance does not imply execution")
        for ax in axes.flat:ax.set_xlabel("Active elapsed time (s)");ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=8)
        fig.suptitle(f"{source.name}: preselected {scenario}, design seed 0; shaded = Full stopping tail")
        fig.savefig(out/("case_"+scenario+".png"),dpi=160);plt.close(fig)


def f(value,precision=3):return "—" if value is None else f"{value:.{precision}f}"


def report(source,out,summary,rows,ratios,previous=None):
    rel=out.relative_to(BASE)
    lookup={(r["scenario"],r["variant"]):r for r in summary["scenario_means"]}
    lines=["# Stage 2 合成设计集实验报告", "",
           f"当前主表来自 `{source.name}`；**仅为设计集描述，未使用 acceptance seeds 1000–1019，不宣称统计验收通过**。本次读到 {summary['recorded_runs']}/{summary['expected_runs']} 个运行指标，valid={summary['valid_runs']}；状态 {summary['states']}，pair 异常文件 {len(summary['pair_exceptions'])} 个。",
           "",
           "均值采用每运行等权；中止运行的有限日志值仍保留在表内，其观察时长可能更短，不能作为成功证据。缺失运行明确计数，不补零，也不把未齐全的配对结果用于验收。图中 * 标记有中止或缺失的单元。", ""]
    if source.name == "design_v1":
        lines += ["**v1 是已知存在求解/执行子空间错误的预备设计版本，不能作为最终 Stage 2 结论；主比较待完整 v2 替换。**", ""]
    if not summary["source_has_final_metrics"]:
        lines += ["**预备快照：源目录尚无最终 metrics.json，实验可能仍在运行。此版用于检查字段/作图流程，主比较待最终设计版本替换。**", ""]
    if previous:
        lines += [f"`{previous['label']}` 保留为预备版本：{previous['recorded_runs']}/{previous['expected_runs']} 指标，{len(previous['missing'])} 项缺失，状态 {previous['states']}，{len(previous['pair_exceptions'])} 个 pair 异常。不能将两个版本挑选有利种子拼成一组。", ""]
    if source.name != "design_v1" and not summary["missing"]:
        full=[r for r in rows if r["variant"]=="full"]
        repair=[r for r in full if r["scenario"] in ACCEPTANCE["repairable_scenarios"]]
        pooled={}
        for comparator in ("baseline","matched_scan_speed","matched_rocking"):
            values=[r for r in summary["paired_descriptive"] if r["comparator"]==comparator and r["scenario"] in ACCEPTANCE["repairable_scenarios"]]
            pooled[comparator]={m:stats(r[m] for r in values)["mean"] for m in ("delta_bad_path_m","delta_valid_coverage_rate_m_s")}
        p=pooled["baseline"];matched=pooled["matched_scan_speed"]
        lines += [f"**行为目标尚未达到。** Full 有 {sum(r.get('valid',False) for r in full)}/{len(full)} 个有效完成运行，六个预列可修复场景中为 {sum(r.get('valid',False) for r in repair)}/{len(repair)}。完整记录和实现测试通过不等于效果通过。六场景等权的 full−baseline 缺失路径变化为 {p['delta_bad_path_m']*1000:+.4f} mm，有效覆盖速度变化为 {p['delta_valid_coverage_rate_m_s']*1000:+.4f} mm/s；与 matched_scan_speed 比较分别为 {matched['delta_bad_path_m']*1000:+.5f} mm 和 {matched['delta_valid_coverage_rate_m_s']*1000:+.5f} mm/s。这里只是保存运行的描述性均值，不能用中止记录的有限指标绕过失败门槛。", ""]
        shadow_base,shadow_full=lookup["shadow","baseline"],lookup["shadow","full"]
        lines += [f"shadow 的 baseline/full 机械缺失路径均为 0，独立 acoustic 与注册图像覆盖均为 0。Full 的平均耗时 {shadow_full['elapsed_s__mean']:.2f} s，对比 baseline {shadow_base['elapsed_s__mean']:.2f} s；力 RMSE {shadow_full['rmse__mean']:.5f} N 对比 {shadow_base['rmse__mean']:.5f} N。该固定遮挡没有被额外动作修复。局部力优先不等式不能替代端到端力误差比较。", ""]
        lines += ["## 各消融的完成状态与可修复场景均值", "",
                  "完成计数覆盖全部 120 个场景/种子运行；后四列只汇总六个可修复场景，各场景等权。中止数据仍保留，因此这些均值本身不代表成功。", "",
                  "| 算法 | complete / gaps / aborted | 可修复场景 RMSE(N) | missing(mm) | valid rate(mm/s) | measured angular travel(rad) |",
                  "|---|---:|---:|---:|---:|---:|"]
        for variant in VARIANTS:
            subset=[r for r in rows if r["variant"]==variant]
            counts=Counter(r["status"] for r in subset)
            ss=[lookup[s,variant] for s in ACCEPTANCE["repairable_scenarios"]]
            cells=[f(stats(r.get(m+"__mean") for r in ss)["mean"]*scale) for m,scale in [("rmse",1),("bad_path_m",1000),("valid_coverage_rate_m_s",1000),("angular_travel_rad",1)]]
            lines.append(f"| {variant} | {counts['complete']} / {counts['complete_with_gaps']} / {counts['aborted']} | "+" | ".join(cells)+" |")
        lines.append("")
    deltas={r["scenario"]:r for r in summary["paired_descriptive"] if r["comparator"]=="baseline"}
    compact=[]
    for s in ("left_gap","right_gap","delayed_execution"):
        c=deltas[s]
        if c["delta_bad_path_m"] is not None:
            compact.append(f"{s}：missing Δ={c['delta_bad_path_m']*1000:+.3f} mm、valid-rate Δ={c['delta_valid_coverage_rate_m_s']*1000:+.3f} mm/s（{c['recorded_pairs']}/10 pairs，{c['failed_pairs']} 失败对）")
    if compact:
        lines += ["主线与 baseline 的直接描述性比较："+"；".join(compact)+"。registered 图像覆盖的变化还会受到帧采样位置及扫描时序影响，不能单独证明机械修复有效。", ""]
    lines += ["## 统计口径与合成边界", "",
              "固定目标 4 N；完整生产 nominal 力/力矩律，60 mm 路径、20 mm/s 标称扫描。外环观测来自合成 B-mode 像素经过真实随机游走提取，控制器不接收独立机械或 acoustic truth。LEFT/RIGHT 为要求窗口，CENTER 只诊断。", "",
              "本设计闭环检查合成 plant、Cartesian QP 与 nominal/reference 的组合，不代表 native 最终发布协议、非原子设备事务或能量账本的 Stage 3 验收已完成。没有连接或驱动真实机器人。", "",
              "力主指标使用当前带既定噪声的 control Fz；RMSE、绝对平均偏差、峰值逐运行计算。源日志指标包含扫描后的约 0.25 s 停止尾段，valid rate 分母也包含它。本报告原样报告该口径，同时将约束活跃比例和角速度可行区间另按 phase='scan' 重算，避免尾段复制的诊断标志偏置。", "",
              "controller_ms 是离线运行的主机耗时，含 nominal/求解及安排的区间诊断；超过 5 ms 的计数是处理耗时诊断，不等于模拟真实执行故障。其 scan-only p95 和原始 p50/p99/max 保留在 CSV；并行运行机器上的计时不能作为硬实时上界。", "",
              "机械 missing path 是源实现的 L−有效前向路径并集；registered image coverage 是按实际拍摄位置/有效相位及独立 acoustic availability 重建的图像并集。机械全耦合也可能因图像采样起止边界留下很小的 registered 缺口，因此 complete_with_gaps 不能一概解释为机械失接触。反向/重复行程另在 runs.csv 保留，不能以 reference endpoint 代替实际覆盖。", "",
              "当前 both_edges 是瞬态跟踪家族，非持续双边静态缺口；[独立静态检查](ACCEPTANCE_METHOD.md) 已在设计种子下发现 θ=0、4 N 的两侧全耦合构形。shadow 的合成声学遮挡可在机械耦合良好时仍令边缘 acoustic availability=0，这两种真值必须分开。任何未修复或正视觉 slack 都不是人体不可修复性证明。", "",
              f"![常见比较器]({rel}/common_comparators.png)", "",
              "## Full 与 baseline 的逐场景原始运行均值", "",
              "括号为 complete / gaps / aborted / missing 计数。力单位 N，missing 和 registered 单位 mm，rate 单位 mm/s；都来自完整保存的运行口径。", "",
              "| 场景 | 算法（C/G/A/M） | RMSE | abs bias | peak | missing | valid rate | registered | acoustic fraction | angular rad |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in SCENARIOS:
        for v in ("baseline","full"):
            r=lookup[s,v];counts=f"{r['complete_n']}/{r['gaps_n']}/{r['aborted_n']}/{r['missing_n']}"
            vals=[]
            for m,scale in [("rmse",1),("absolute_mean_bias",1),("peak_absolute_error",1),("bad_path_m",1000),("valid_coverage_rate_m_s",1000),("registered_image_coverage_m",1000),("acoustic_coverage_fraction",1),("angular_travel_rad",1)]:
                val=r.get(m+"__mean");vals.append(f(None if val is None else val*scale))
            lines.append(f"| {s} | {v} ({counts}) | "+" | ".join(vals)+" |")
    lines += ["",f"全部 12×7 场景/消融单元及其样本数、范围、停止状态见 [scenario_ablation.csv]({rel}/scenario_ablation.csv)；逐运行值见 [runs.csv]({rel}/runs.csv)。不对设计集给出显著性、非劣性或总体成功概率结论。", "",
              "## 摇摆限制与 matched 控制的实际运动", "",
              f"![约束与实际转动]({rel}/constraints_and_rotation.png)", "",
              "可行 ω 宽度依次表示机械约束、加入力优先约束、再加入孔径预算后的区间投影；它不是实际摇摆幅度。可行区间仅在安排了诊断的 scan 帧计算，−1 哨兵不参与均值；baseline/matched 没有 QP 区间诊断，标为未提供而非零宽。活跃行比例使用 scan tick；硬限制未贴边也可通过目标代价改变动作。", "",
              "| 场景 | mechanical / force / complete ω 宽度(rad/s) | force / aperture 活跃比例 | matched speed 实际路程比 / 转动比 | matched rocking 实际路程比 / 转动比 |",
              "|---|---|---|---|---|"]
    for s in SCENARIOS:
        r=lookup[s,"full"]
        widths=" / ".join(f(r.get("scan_omega_width_"+m+"_mean__mean"),4) for m in ("mechanical","force","complete"))
        active=" / ".join(f(r.get("scan_"+m+"_active_fraction__mean")) for m in ("force","aperture"))
        cells=[]
        for v in ("matched_scan_speed","matched_rocking"):
            group=[x for x in ratios if x["scenario"]==s and x["comparator"]==v]
            cells.append(" / ".join(f(stats(x.get(m+"_ratio_to_full") for x in group)["mean"]) for m in ("measured_positive_travel_m","angular_travel_rad")))
        lines.append(f"| {s} | {widths} | {active} | {cells[0]} | {cells[1]} |")
    lines += ["", "运动比是每个完整配对中 comparator/full 的描述性比值，再在场景内等权平均；full 分母接近零时保留未定义，不把 0/0 设成 1。比值包括失败日志时同样受短运行影响，逐对 valid 标志、分母差值、elapsed 比见 CSV。matched 只规定回放/截幅规则，并不保证最终实际转动或总耗时恰好相等，尤其停止尾段和延迟执行必须检查。",
              f"完整明细：[matched_motion.csv]({rel}/matched_motion.csv)。", "",
              "## 恢复时间：描述性补充，不是新验收门槛", "",
              "事件从 required latent acoustic min(L,R)<0.90 的首次记录开始，直到两侧连续 ≥0.90 保持至少 0.20 s 才确认恢复；报告恢复开始相对事件起点的时间，并记录确认时刻。短于 0.20 s 的好转后再次变坏仍属于同一个事件。记录起始已坏标为左删失；记录结束仍未确认的事件保留右删失和可观察时长，不删掉，也不把它们记成零恢复时间。无缺口运行单列。stop_tail 才确认的恢复另有字段，不能当作扫描期间修复成功。", "",
              "| 场景 | 算法 | 事件数 | 确认恢复 | 未确认/删失 | 无缺口运行 | 已确认事件恢复时间中位(s) |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for s in SCENARIOS:
        for v in ("baseline","full"):
            r=lookup[s,v]
            lines.append(f"| {s} | {v} | {r['recovery_events']} | {r['recovery_confirmed']} | {r['recovery_unconfirmed']} | {r['recovery_no_deficit_runs']} | {f(r['recovery_median_s'])} |")
    lines += ["",f"逐事件边界、删失与确认 phase：[recovery_events.csv]({rel}/recovery_events.csv)。中位数只描述确认恢复的事件，必须与未恢复数量一起读；没有把它解释为全部事件的恢复分布。", "",
              "## 固定代表案例与失败保留", "",
              "以下案例预先固定使用 design seed 0 的 left_gap、both_edges、shadow、delayed_execution；没有按效果最好的种子挑选。蓝色阴影仅标 full 的停止尾段，各算法按自身时间轴显示。"]
    for s in ("left_gap","both_edges","shadow","delayed_execution"):
        if (out/("case_"+s+".png")).exists():lines += ["",f"![{s}, seed 0]({rel}/case_{s}.png)"]
    center=lookup["central_gap","full"].get("scan_center_acoustic_bad_fraction__mean")
    if center is not None:
        lines += ["",f"central_gap 的 Full 中，CENTER acoustic fraction<0.90 的 scan-tick 比例均值为 {center:.3%}，而要求的左右 acoustic 恢复事件数为 {lookup['central_gap','full']['recovery_events']}。中心诊断没有被混入边缘缺口统计。"]
    reasons=Counter(r.get("reason") or "unspecified" for r in rows if r["status"]=="aborted")
    failures=[r for r in rows if r["status"]=="aborted"]
    numerical=Counter(str((r.get("qp_failure")or{}).get("solver_status","not_recorded")) for r in failures)
    violations=sum(len(r.get("mechanical_violations",[])) for r in rows)
    lines += ["",f"已记录中止原因计数：`{dict(reasons)}`。pair 级异常为 `{summary['pair_exceptions']}`。异常前已保存 NPZ 但未写成 metrics.json 的文件也留在输入哈希索引中；不自行补造这些运行的完成状态。", "",
              f"失败记录的求解器原始诊断：`{dict(numerical)}`；日志中的实际机械包络越界事件合计 {violations}。`mechanical_infeasible` 是当时 QP/运动子空间 admission 的结果，不是静态接触几何不可修复的证明。若原始诊断为 MAX_ITER 而后续分类为 mechanical_rows_infeasible，两个事实均保留；不能改写成“所有数值失败已消失”。baseline/matched clip 后离开 H 或反向/超过允许进度也按中止保留，不放宽容差隐藏。", "",
              "## 哈希与复现", "",
              f"- 输入 manifest SHA-256：`{summary['manifest_sha256']}`",
              f"- 数据文件索引 SHA-256：`{summary['data_sha256']}`",
              f"- manifest 中 plant / QP / runner 源码 SHA-256：`{summary['model_hash']}` / `{summary['qp_hash']}` / `{summary['runner_hash']}`",
              f"- sources.zip SHA-256：`{summary['source_snapshot_sha256']}`；缺失时不能声称具有当次完整源码快照。",
              f"- 分析脚本 SHA-256：`{summary['analysis_sha256']}`",
              f"- 已核对 sources.zip 中 {summary['verification']['source_files_verified']} 个源码文件；{summary['verification']['run_metrics_verified']} 份 NPZ 的力/转动/耗时指标重算，最大绝对差 {summary['verification']['maximum_metric_absolute_error']:.3g}。",
              f"- 本次分析脚本快照：[report_design_snapshot.py.txt]({rel}/report_design_snapshot.py.txt)。",
              f"- 配置/依赖版本与所有描述性结果：[summary.json]({rel}/summary.json)；逐文件指纹：[input_hashes.json]({rel}/input_hashes.json)。", "",
              f"复现：`/media/camp/EXT_DRIVE/envs/genesis/bin/python MD/contact_qp/analysis_artifacts/report_design.py --input {source.relative_to(ROOT)}`。分析只读设计结果，不执行 controller/plant，不读取留出种子。"]
    (BASE/"EXPERIMENT_DESIGN.md").write_text("\n".join(lines)+"\n")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",required=True,type=Path)
    parser.add_argument("--previous",type=Path)
    args=parser.parse_args()
    source=args.input.resolve()
    if not source.name.startswith("design_"):raise ValueError("Design result directory required; acceptance data forbidden")
    out=BASE/"analysis_artifacts"/source.name
    summary,rows,ratios=collect(source,out)
    plots(source,out,summary,rows,ratios)
    previous=None
    if args.previous:
        p=BASE/"analysis_artifacts"/args.previous.name/"summary.json"
        if p.exists():previous=json.loads(p.read_text())
    report(source,out,summary,rows,ratios,previous)
    print(json.dumps({k:summary[k] for k in ("label","recorded_runs","expected_runs","valid_runs","states","recovery","data_sha256")},indent=2))


if __name__=="__main__":main()
