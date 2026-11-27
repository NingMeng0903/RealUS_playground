"""Development-only audits and paired acceptance summaries."""

from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from environment import Arm, ScanEnvironment, TRAIN_SEEDS, DEV_SEED, SMOKE_SEED, TEST_SEED
from environment import run_cases, test_cases, spec_for
import motion
from scan_kernel import DEPTH_MAX
from storage import digest, read_json, read_lines, shards


def verify_topology(pack):
    directory = pack.parent
    with np.load(pack, allow_pickle=False) as data:
        low, high = data["ct_bounds"]
        assert high - low == 16
        assert np.all(data["status"][3, low:high + 1, 0] == 1)
        assert np.count_nonzero(data["edge_status"][3, low:high, 0] == 3) == 1
        assert np.array_equal(data["links"], data["edge_status"] > 0)
        assert data["centers"].shape == (7, 4, 161, 6, 2)
        assert np.isfinite(data["centers"][:, 3, low:high + 1, 0]).all()
        ct_bounds = data["neutral_grid"][[low, high]].tolist()
        status = [np.bincount(edge.reshape(-1), minlength=4).tolist() for edge in data["edge_status"]]
    native = directory / "medseg_ct_route.npz"
    with np.load(native, allow_pickle=False) as data:
        assert np.array_equal(data["label_id"], [39] * 13 + [23] * 18)
        assert np.array_equal(data["native_layer"], np.arange(31))
        assert data["shared_voxel_faces"][12] == 233 and data["shared_voxel_faces"].min() > 0
        native_span = np.ptp(data["native_ras_mm"][:, 2])
    return {"CT_stations": 17, "CT_neutral_endpoints_mm": ct_bounds, "native_CT_slices": 31,
            "native_CT_span_mm": native_span, "native_CT_subjects": 1,
            "native_CT_contact_faces_at_23_39": 233, "linked_ct_transfers": 7,
            "adjacent_edge_counts_by_source_and_provenance": status,
            "inputs_sha256": {p.name: digest(p) for p in (pack, native, directory / "ct_reference_topology.json.gz")}}


def verify_split(target, dev_count):
    groups = [list(run_cases(target, "training", s)) for s in TRAIN_SEEDS]
    groups.extend([list(run_cases(dev_count, "development", DEV_SEED)),
                   list(run_cases(dev_count, "calibration", DEV_SEED + 1)), list(run_cases(64, "smoke", SMOKE_SEED)),
                   list(test_cases())])
    seeds = set()
    for group in groups:
        for case in group:
            for seed in (case.world_seed, case.tape_seed, case.action_seed):
                if seed in seeds:
                    raise RuntimeError("Random stream seed overlap")
                seeds.add(seed)
    return {"unique_world_tape_and_action_keys": len(seeds), "seeds_checked": len(groups)}


def quality_intervention():
    trial = motion.Trial(2, 744901, "stress")
    center = np.array([[0., 40.], [DEPTH_MAX * 4., 40.]])
    first, second = np.array([0., 0.]), np.array([0., 1.])
    for tick in range(1, 48):
        old = trial.observe(center, first, tick)
        new = trial.observe(center, second, tick)
        for a, b in zip(old, new):
            np.testing.assert_array_equal(a, b)
    return {"same_public_frames": 47, "offscreen_gate_changed": 1., "maximum_quality_change": 0.}


def integrated_interventions(pack, actor, threshold):
    from environment import rollout
    from synthetic_anatomy import LayoutFactory

    class Relabel(motion.Trial):
        def observe(self, *args, **kwargs):
            z, labels, q = super().observe(*args, **kwargs)
            return z, np.where(labels >= 0, (labels + 1) % 6, labels), q

    class ShiftEcho(motion.Trial):
        def observe(self, *args, **kwargs):
            z, labels, q = super().observe(*args, **kwargs)
            z[:, 0] = np.clip(z[:, 0] + 5., -24.99, 24.99)
            return z, labels, q

    case = next(c for c in run_cases(16, "smoke", SMOKE_SEED) if c.source == 3 and c.profile == "ordinary")
    scanner = motion.Scanner.training(pack, case.shape)
    env = ScanEnvironment(scanner, LayoutFactory(pack, scanner, case.shape))
    arm = Arm("invariance", sampled=True)
    base = rollout(env, case, arm, threshold, actor)
    alternative = rollout(env, case, arm, threshold, actor, trial_type=Relabel)
    for r, a in zip(base.steps, alternative.steps, strict=True):
        assert r["skill"] == a["skill"]
        np.testing.assert_array_equal(r["features"], a["features"])
        np.testing.assert_array_equal(r["controls"], a["controls"])
    assert len(base.frames) == len(alternative.frames)
    for r, a in zip(base.frames, alternative.frames):
        assert r[:2] == a[:2]
        np.testing.assert_array_equal(r[2], a[2])

    spec = spec_for(env.scanner, case)
    node, trial, limits = env.start(case, spec)
    for _ in range(4):
        env.route.advance(node, spec, trial, limits)
    shifted = ShiftEcho(case.source, case.tape_seed, case.profile, world_seed=case.world_seed, split=case.split)
    shifted.layout, shifted.window = trial.layout, trial.window
    caps = limits.probe(node, spec)
    turn = .48 if node.sector < 12 else -.48
    args = np.array([turn, .53, -.64])
    plan = env.plan(node, spec, 2, args, caps, arm)
    change = env.plan(node, spec, 2, np.array([turn, .53, .64]), caps, arm)
    assert plan is not None and change is not None
    outputs, stop_positions = [], []

    def stop_here(in_motion, info):
        stop_positions.append([in_motion.station, in_motion.sector, info["elapsed_ticks"], info["return_reserve_ticks"]])
        return True

    def execute(probe, tape, feedback, stop=None):
        return env.material.run_probe(node.fork(), spec, tape, probe, *caps,
                                      feedback=feedback, stop=stop)

    for tape, mode, control in [(trial, False, plan), (shifted, False, plan), (trial, True, plan),
                                (shifted, True, plan), (trial, False, change)]:
        outputs.append(execute(control, tape, mode))
    first, first_shift, closed, closed_shift, changed = outputs
    early = execute(plan, trial, True, stop_here)
    assert all(item.legal for item in outputs) and early.legal and early.stopped
    assert np.array_equal(first.trajectory, first_shift.trajectory)
    assert not np.array_equal(closed.trajectory, closed_shift.trajectory)
    assert not np.array_equal(first.trajectory, changed.trajectory)
    assert early.node.station == node.station and early.node.sector == node.sector
    assert early.ticks <= first.ticks
    assert stop_positions and not np.array_equal(stop_positions[0][:2], [node.station, node.sector])
    resumed = early.node
    env.route.advance(resumed, spec, trial, limits)
    assert resumed.station == node.station + 1
    assert len(resumed.reader.frames) == resumed.tick
    return {"offscreen_quality": quality_intervention(), "label_permutation_actions_equal": len(base.steps),
            "causal_echo_perturbation": True, "open_loop_invariant": True,
            "control_shape_trajectory_changed": True, "legal_stop_and_resume": True,
            "full_probe_ticks": first.ticks, "early_probe_ticks": early.ticks,
            "exit_opportunity": stop_positions[0], "material": env.material.dird.metadata}


def verify_generated_layouts(pack, rows, teaching):
    """Check static anatomy, native support and acquired echo differences before production."""
    from synthetic_anatomy import LayoutFactory

    by_case = defaultdict(list)
    for case, _arm, item in rows:
        by_case[case].append(item)
    deltas, relative_layouts, changed_echoes, comparison_ticks = [], defaultdict(set), 0, 0
    windows, regular = [], 0
    junction_counts = Counter()
    checked_junctions, moved_sfj = 0, 0
    scanners = {}
    factories = {}
    with np.load(pack, allow_pickle=False) as data:
        original = data["centers"]
    for case, items in by_case.items():
        samples = [item.metrics["layout_id"] for item in items]
        assert len(set(samples)) == 1 and samples[0]
        for item in items[1:]:
            np.testing.assert_array_equal(items[0].layout_centers, item.layout_centers)
        sample = items[0]
        before, after = original[case.shape, case.source], sample.layout_centers
        np.testing.assert_array_equal(np.isfinite(before), np.isfinite(after))
        relative = after[:, [1, 2]] - after[:, [0]]
        nominal = before[:, [1, 2]] - before[:, [0]]
        change = np.linalg.norm(relative - nominal, axis=-1)
        deltas.append(float(np.nanmax(change, initial=0.)))
        relative_layouts[(case.source, case.shape)].add(samples[0])
        metadata = sample.metrics["layout"]
        checks = metadata["accepted_checks"]
        assert metadata["source_index"] == case.source and checks["support_preserved"] and checks["links_preserved"]
        assert checks["new_outside_skin"] == checks["new_severe_bone_conflicts"] == checks["new_crossings"] == 0
        regular += int(not np.any(np.isfinite(before) & (before != after)))
        if metadata.get("junctions"):
            junction_counts[str(case.source)] += len(metadata["junctions"])
        if sample.metrics.get("recoverable_window"):
            challenge = sample.metrics["recoverable_window"]
            assert challenge["alternate_resolved"] > challenge["reference_resolved"]
            assert challenge["alternate_mixed"] < challenge["reference_mixed"]
            windows.append(challenge)
        if deltas[-1] < 1.e-4:
            continue
        if case.shape not in scanners:
            scanners[case.shape] = motion.Scanner.training(pack, case.shape)
        scanner = scanners[case.shape]
        if metadata.get("junctions"):
            if case.shape not in factories:
                factories[case.shape] = LayoutFactory(pack, scanner, case.shape)
            layout = factories[case.shape].build(case.source, case.world_seed, case.profile, case.split)
            assert layout.identifier == sample.metrics["layout_id"] and case.source == 3
            for junction in layout.metadata["junctions"]:
                roles = junction["roles"]
                position = junction["moved_station_index"]
                center = np.asarray(junction["shared_xyz_mm"])[[0, 2]]
                assert scanner.ct_bounds[0] <= position <= scanner.ct_bounds[1]
                assert all(edge["synthetic_shared_vertex"] == junction["junction_id"]
                           for edge in junction["incident_edges"])
                if len(roles) > 1:
                    np.testing.assert_allclose(layout.at(position)[roles], np.tile(center, (len(roles), 1)), atol=1.e-8)
                checked_junctions += 1
                moved_sfj += int(junction["labels"] == [23, 27] and abs(junction["axial_shift_mm"]) > .01)
        spec = spec_for(scanner, case)
        trial = motion.Trial(case.source, case.tape_seed, case.profile, world_seed=case.world_seed, split=case.split)
        stations = np.unique(np.linspace(spec[1], spec[2], 4).astype(int))
        for station in stations:
            for j, sector in enumerate(np.flatnonzero(scanner.valid[station])[::5]):
                pair = []
                for center in (before, after):
                    trial.layout = SimpleNamespace(centers=center)
                    points, gates = scanner._anatomy.training_truth(scanner.surface, trial, int(station), int(sector))
                    echoes, _, quality = trial.observe(points, gates, 3 + j, station=station, sector=sector)
                    pair.append((echoes, quality))
                assert pair[0][1] == pair[1][1]
                comparison_ticks += 1
                changed_echoes += int(not np.array_equal(pair[0][0], pair[1][0]))
    assert max(map(len, relative_layouts.values())) >= 2
    assert max(deltas) > 1. and changed_echoes > 0 and windows
    assert regular > 0
    assert checked_junctions > 0 and moved_sfj > 0
    assert all(item.get("fixed_layout_id") for item in teaching)
    return {"paired_smoke_cases": len(by_case), "distinct_layouts": len({items[0].metrics["layout_id"] for items in by_case.values()}),
            "source_reference_cases": regular, "relative_changes_over_one_mm": sum(v > 1. for v in deltas),
            "maximum_relative_displacement_mm": max(deltas), "same_pose_comparisons": comparison_ticks,
            "same_pose_changed_public_echoes": changed_echoes, "identical_finite_annotation_support": True,
            "paired_teacher_prefixes_fixed_layout": len(teaching), "recoverable_windows": len(windows),
            "alternate_window_resolved_frame_gain": int(sum(w["alternate_resolved"] - w["reference_resolved"] for w in windows)),
            "synthesized_junction_records_by_source": dict(junction_counts),
            "verified_shared_junction_vertices": checked_junctions,
            "shifted_saphenofemoral_junctions": moved_sfj,
            "layout_trials": [{"source": key[0], "shape": key[1], "distinct_layouts": len(value)}
                              for key, value in sorted(relative_layouts.items())]}


def same_track_report(rows):
    regions = [r for item in rows for r in item.audit]
    ranks = [r["true_rank"] for r in regions if r["true_rank"] is not None]
    old_submitted = [r for r in regions if r["old_commit"]]
    new_ranked = sorted((r["best"] for r in regions if r["best"] is not None), reverse=True)
    new_equal = new_ranked[:len(old_submitted)]
    equal_case = []
    for item in rows:
        count = sum(x["old_commit"] for x in item.audit)
        picks = sorted((r["best"] for r in item.audit if r["best"] is not None), reverse=True)
        equal_case.extend(picks[:count])
    grid = np.unique(np.r_[np.linspace(.32, .82, 26), .55])
    curve = []
    for threshold in grid:
        selected = [r["best"][1] for r in regions if r["best"] is not None and r["best"][0] >= threshold]
        correct = sum(selected)
        wrong = len(selected) - correct
        curve.append({"threshold": float(threshold), "submitted": len(selected), "correct": correct, "wrong": wrong,
                      "omitted": len(regions) - len(selected), "risk": wrong / max(1, len(selected)),
                      "coverage": len(selected) / len(regions), "calibration_score": correct - 1.12 * wrong})
    selected = max(curve, key=lambda row: (row["calibration_score"], row["correct"], row["threshold"]))
    return {"cases": len(rows), "regions": len(regions), "true_seen": len(ranks),
            "near_best_true_rank_median": None if not ranks else np.median(ranks),
            "best_true_when_seen": sum(rank == 1 for rank in ranks),
            "no_candidate": sum(r["candidates"] == 0 for r in regions),
            "candidate_slots": sum(r["candidates"] for r in regions),
            "equal_global_submissions": len(new_equal),
            "old_correct_at_equal": sum(r["old"][1] for r in old_submitted),
            "new_correct_at_equal_global": sum(p[1] for p in new_equal),
            "new_correct_at_equal_per_case": sum(p[1] for p in equal_case),
            "new_submissions_equal_per_case": len(equal_case), "risk_coverage": curve,
            "chosen": selected}


METRICS = ("pass", "wrong_delivery", "correct_regions", "wrong_regions", "no_result_regions",
           "correct_mm", "capture_mm", "captured_regions", "physical_regions", "strict_physical",
           "correct_physical_mm", "capture_physical_mm", "physically_accessible_mm", "raw_target_mm",
           "target_inside_bone_mm", "time_s", "travel_mm", "frames", "probes", "proposed", "early_probes",
           "early_stops", "stop_opportunities", "probe_ticks", "probe_mm", "feedback_updates", "dird_probes",
           "reward")


def summarize(rows):
    rows = list(rows)
    if not rows:
        return {"episodes": 0}
    count = {name: sum(r[name] for r in rows) for name in METRICS if name in rows[0]}
    attempts = np.asarray([r["attempted"] for r in rows]).sum(axis=0)
    done = np.asarray([r["succeeded"] for r in rows]).sum(axis=0)
    return {"episodes": len(rows), "complete": sum(r["complete"] for r in rows),
            "totals": count, "means": {k: v / len(rows) for k, v in count.items()},
            "attempted_by_skill": attempts, "executed_by_skill": done,
            "executable_fraction_by_skill": done / np.maximum(1, attempts),
            "before_region_end_fraction": count["early_probes"] / max(1, count["probes"]),
            "active_scan_count": sum(r["probes"] > 0 for r in rows)}


def training_summary(directory):
    all_rows = []
    pre, positive, negative, gains, student, tails = 0, 0, 0, [], [], []
    for path in shards(directory):
        all_rows.extend(read_lines(path / "episodes.jsonl.gz"))
        for record in read_lines(path / "teacher.jsonl.gz"):
            pre += int(record["eligible_prefix"])
            for branch in record["branches"]:
                if branch["skill"] and branch["feasible"]:
                    g = branch["paired_gain"]
                    positive += int(min(g) > 0.)
                    negative += int(max(g) < 0.)
                    gains.append(g)
                if branch["origin"] == "student" and branch["feasible"]:
                    student.extend(branch["paired_gain"])
            if record.get("full_tail") is not None:
                tails.extend(record["full_tail"]["sign_conflict"])
    result = summarize(all_rows)
    result["teacher"] = {"prefixes": pre, "positive_both_tapes": positive, "negative_both_tapes": negative,
                         "feasible_probe_branches": len(gains), "mixed_or_zero": len(gains) - positive - negative,
                         "branch_mean_gain": float(np.mean(gains)) if gains else None,
                         "observed_student_branch_mean_gain": float(np.mean(student)) if student else None,
                         "full_tail_comparisons": len(tails), "full_tail_sign_conflicts": sum(tails)}
    result["source_counts"] = dict(Counter(str(r["source"]) for r in all_rows))
    result["anatomy"] = {"distinct_layouts": len({r["layout_id"] for r in all_rows if r.get("layout_id")}),
                         "verified_recoverable_windows": sum(bool(r.get("recoverable_window")) for r in all_rows),
                         "ambiguous_window_traversed_at_least_once": sum(bool(r.get("acquired_in_ambiguous_window")) for r in all_rows)}
    return result


def paired(rows, arms, comparisons, seed=9438):
    grouped = {arm: {} for arm in arms}
    for row in rows:
        grouped[row["arm"]][row["identifier"]] = row
    res = {"arms": {arm: summarize(store.values()) for arm, store in grouped.items()}, "pairs": {}}
    rng = np.random.default_rng(seed)
    for alt, reference in comparisons:
        store, base = grouped[alt], grouped[reference]
        if set(store) != set(base):
            raise ValueError(f"Pair case sets differ for {alt} and {reference}")
        keys = sorted(store)
        table = defaultdict(list)
        strata = defaultdict(list)
        for key in keys:
            a, b = store[key], base[key]
            table[a["band"]].append((a, b))
            strata[(a["task"], a["profile"])].append((a, b))
        deltas = {}
        for metric in ("pass", "wrong_delivery", "correct_regions", "correct_mm", "capture_mm",
                       "correct_physical_mm", "strict_physical", "time_s", "travel_mm", "reward"):
            clusters = np.array([np.mean([a[metric] - b[metric] for a, b in values])
                                 for _, values in sorted(table.items())])
            draw = rng.integers(len(clusters), size=(4000, len(clusters)))
            interval = np.quantile(clusters[draw].mean(axis=1), [.025, .975])
            deltas[metric] = {"mean": clusters.mean(), "band_cluster_bootstrap_95": interval}
        ai = np.asarray([store[key]["region_outcomes"] for key in keys])
        bi = np.asarray([base[key]["region_outcomes"] for key in keys])
        res["pairs"][f"{alt}_vs_{reference}"] = {
            "pairs": len(keys), "random_band_clusters": len(table), "difference": deltas,
            "rescued_regions": int(((ai == 1) & (bi != 1)).sum()),
            "lost_regions": int(((ai != 1) & (bi == 1)).sum()),
            "new_wrong_regions": int(((ai == 0) & (bi != 0)).sum()),
            "removed_wrong_regions": int(((ai != 0) & (bi == 0)).sum()),
            "rescued_scans": sum(store[k]["pass"] > base[k]["pass"] for k in keys),
            "lost_scans": sum(store[k]["pass"] < base[k]["pass"] for k in keys),
            "task_profile_means": [{"task": task, "profile": profile, "pairs": len(values),
                                    **{m: float(np.mean([a[m] - b[m] for a, b in values]))
                                       for m in ("pass", "wrong_delivery", "correct_regions", "correct_mm", "time_s")}}
                                   for (task, profile), values in sorted(strata.items())],
        }
    return res


def render_report(path, freeze, acceptance):
    base = acceptance["wide"]
    lines = [
        "右腿超声闭环训练与独立配对验收",
        "",
        "实验在注册后的二维体表截面质心仿真中运行。静态解剖输入为三份图谱和一例 MedSeg，"
        "MedSeg 七种形体来自同一例标注。Blender 验收使用一份解剖，每个任务和压力等级各 16 条新扰动带。",
        "",
        "GPU 及训练计数",
        f"设备 {freeze['gpu']['device']}，PyTorch {freeze['gpu']['torch']}，CUDA {freeze['gpu']['torch_cuda_build']}。",
    ]
    for key, run in freeze["runs"].items():
        lines.append(f"{key} 完整学生扫描 {run['recount']['complete_unique']:,}，"
                     f"实际布局 {run['recount']['distinct_generated_layouts']:,}，"
                     f"帧数 {run['recount']['frames']:,}，剔除 {run['recount']['excluded']}。")
        t = run["summary"]["teacher"]
        if t["prefixes"]:
            lines.append(f"教师前缀 {t['prefixes']:,}，双带同向为正 {t['positive_both_tapes']:,}，"
                         f"双带同向为负 {t['negative_both_tapes']:,}，局部与全尾符号相反 "
                         f"{t['full_tail_sign_conflicts']}/{t['full_tail_comparisons']}。"
                         f"学生动作在匹配前缀的平均分支收益 {t['observed_student_branch_mean_gain']:.4f}。")
    lines.extend(["", f"训练域冻结读图阈值 {freeze['threshold']:.3f}。完整交付要求扫描到达末站且六区选择均正确。",
                  "正确覆盖长度由逐站实际候选的提交结果计算。物理可达子集以原始目标位置、合法扫描接触和精确骨遮挡判定。",
                  "", "Blender 宽预算冻结验收",
                  "实验臂 | 完整交付 | 至少一次错交 | 正确区间 | 平均正确覆盖 mm | 平均采集时间 s"])
    for arm, results in base["arms"].items():
        n, totals, m = results["episodes"], results["totals"], results["means"]
        lines.append(f"{arm} | {totals['pass']}/{n} | {totals['wrong_delivery']}/{n} | "
                     f"{totals['correct_regions']}/{6*n} | {m['correct_mm']:.2f} | {m['time_s']:.2f}")
    lines.extend(["", "同任务和同扰动配对变化"])
    for key, data in base["pairs"].items():
        d = data["difference"]["correct_regions"]
        lo, hi = d["band_cluster_bootstrap_95"]
        lines.append(f"{key} 每扫描正确区间变化 {d['mean']:+.3f}，95% 随机带成组区间 "
                     f"[{lo:+.3f}, {hi:+.3f}]，救回 {data['rescued_regions']} 区，损失 {data['lost_regions']} 区。")
    lines.extend(["", "详细指标、风险–覆盖曲线、物理可达子集、消融、随机种子和输入哈希见同目录的 JSON 文件。"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
