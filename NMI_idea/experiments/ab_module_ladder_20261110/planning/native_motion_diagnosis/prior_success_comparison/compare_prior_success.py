"""Read-only comparison of the sealed smooth-reference runs and B replay.

This script never constructs a controller and never invokes native code.  It
only compares saved arrays, JSON metadata, text fingerprints, and the current
on-disk source hashes.  Outputs are owned by this directory.
"""
from __future__ import annotations

import hashlib
import json
import difflib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REPAIR = ROOT.parent / "ab_validation_20261110" / "repair_20261110"
OLD = REPAIR / "qp_cross_track_fix"
OLD_INPUT = REPAIR / "audit" / "phase2_native_subset_inputs.npz"
BASELINE = ROOT / "planning" / "native_regression_20261110" / "baseline_v2"
NEW = ROOT / "planning" / "native_subset"
CAPABILITY = ROOT / "capability" / "native_subset.json"
REGRESSION = ROOT / "planning" / "native_regression_20261110"
J1_GUARD = REGRESSION / "j1_guard_only"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_load(path: Path):
    return json.loads(path.read_text())


def changed_lines(old_path: Path, new_path: Path) -> list[str]:
    """Return only added/removed lines, excluding unified-diff headers."""
    old = old_path.read_text().splitlines()
    new = new_path.read_text().splitlines()
    return [
        line for line in difflib.unified_diff(old, new, lineterm="")
        if (line.startswith("+") or line.startswith("-"))
        and not line.startswith(("+++", "---"))
    ]


def finite_max_abs(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def compare_traces(old_path: Path, fresh_path: Path) -> dict:
    old = np.load(old_path, allow_pickle=False)
    fresh = np.load(fresh_path, allow_pickle=False)
    common = sorted(set(old.files) & set(fresh.files))
    only_old = sorted(set(old.files) - set(fresh.files))
    only_fresh = sorted(set(fresh.files) - set(old.files))
    numeric = {}
    exact = {}
    for key in common:
        a, b = old[key], fresh[key]
        if a.shape != b.shape:
            exact[key] = {"shape_old": list(a.shape), "shape_fresh": list(b.shape), "equal": False}
        elif a.dtype.kind in "OUSb" or b.dtype.kind in "OUSb":
            exact[key] = {"equal": bool(np.array_equal(a, b))}
        else:
            numeric[key] = finite_max_abs(a, b)
    state_keys = [k for k in common if k != "tick_compute_seconds"]
    state_equal = all(
        (np.array_equal(old[k], fresh[k]) if old[k].dtype.kind in "OUSb" or fresh[k].dtype.kind in "OUSb"
         else finite_max_abs(old[k], fresh[k]) == 0.0)
        for k in state_keys
    )
    return {
        "old_file": str(old_path),
        "fresh_file": str(fresh_path),
        "old_sha256": sha(old_path),
        "fresh_sha256": sha(fresh_path),
        "keys_only_old": only_old,
        "keys_only_fresh": only_fresh,
        "length_old": int(len(old["t_s"])),
        "length_fresh": int(len(fresh["t_s"])),
        "max_abs_difference_by_common_numeric_key": numeric,
        "exact_comparison_by_non_numeric_key": exact,
        "controller_state_arrays_bitwise_equal_except_tick_compute_seconds": bool(state_equal),
        "tick_compute_seconds_max_abs_difference": numeric.get("tick_compute_seconds"),
    }


def trace_metrics(trace_path: Path, nominal_duration: float | None = None) -> dict:
    z = np.load(trace_path, allow_pickle=False)
    t = np.asarray(z["t_s"], dtype=float)
    p = np.asarray(z["pose_desired"][:, :3, 3], dtype=float)
    R = Rotation.from_matrix(z["pose_desired"][:, :3, :3])
    dt = np.diff(t)
    if nominal_duration is None:
        mask = np.ones(len(t), dtype=bool)
    else:
        # Include the final path sample when it lands on the declared duration.
        mask = t <= float(nominal_duration) + 1e-12
    idx = np.flatnonzero(mask)
    if len(idx) < 2:
        idx = np.arange(len(t))
    sl = slice(int(idx[0]), int(idx[-1]) + 1)
    pp = p[sl]
    RR = R[sl]
    tt = t[sl]
    dtt = np.diff(tt)
    trans_rate = np.linalg.norm(np.diff(pp, axis=0), axis=1) / dtt if len(pp) > 1 else np.zeros(0)
    ang_rate = (RR[1:] * RR[:-1].inv()).magnitude() / dtt if len(RR) > 1 else np.zeros(0)
    return {
        "trace_file": str(trace_path),
        "sample_count": int(len(t)),
        "trace_end_s": float(t[-1]),
        "path_sample_count": int(len(pp)),
        "path_translation_chord_m": float(np.linalg.norm(np.diff(pp, axis=0), axis=1).sum()) if len(pp) > 1 else 0.0,
        "path_net_rotation_rad": float((RR[-1] * RR[0].inv()).magnitude()) if len(RR) else 0.0,
        "path_peak_translation_rate_m_s": float(trans_rate.max()) if len(trans_rate) else 0.0,
        "path_peak_angular_rate_rad_s": float(ang_rate.max()) if len(ang_rate) else 0.0,
        "full_position_max_m": float(np.max(z["position_error_m"])),
        "full_rotation_max_deg": float(np.degrees(np.max(z["rotation_error_rad"]))),
        "same_time_start_q": np.asarray(z["q_meas_before"][0]).tolist(),
        "qdot_entry": np.asarray(z["qdot"][0]).tolist(),
        "acceleration_max_abs_by_coordinate": np.max(np.abs(z["command_acceleration"]), axis=0).tolist()
        if "command_acceleration" in z.files else None,
        "qp_equality_residual_max": float(np.max(z["qp_equality_residual_max"])),
        "task_progress_p05": float(np.percentile(z["task_progress"], 5)),
    }


def q_comparison(old_q: np.ndarray, current_q: np.ndarray) -> dict:
    delta = np.asarray(current_q, dtype=float) - np.asarray(old_q, dtype=float)
    return {"max_abs_difference": float(np.max(np.abs(delta))), "difference": delta.tolist()}


def main() -> None:
    old_input = np.load(OLD_INPUT, allow_pickle=False)
    old_q = np.asarray(old_input["branch4_path0_q_seed"], dtype=float)
    frozen = json_load(ROOT / "planning" / "test_frozen" / "results.json")
    context_q = {row["id"]: np.asarray(row["q_seed"], dtype=float) for row in frozen["contexts"]}

    old_runs = json_load(OLD / "smooth_reference_results.json")["runs"]
    old_baseline = {run["case"]: run for run in old_runs if run["case"].endswith("smooth_baseline")}
    public = json_load(CAPABILITY)
    current_results = json_load(NEW / "RESULTS.json")

    source_paths = {
        "native_binary": Path("/media/camp/EXT_DRIVE/RealUS_playground/rm75_control/native/wbc_rt/build/wbc_rt"),
        "controller_yaml": Path("/media/camp/EXT_DRIVE/RealUS_playground/peirastic/configs/controller.yaml"),
        "loop": Path("/media/camp/EXT_DRIVE/RealUS_playground/rm75_control/rm75_control/control/joint_admittance_8dof/loop.py"),
        "qp_builder": Path("/media/camp/EXT_DRIVE/RealUS_playground/rm75_control/rm75_control/control/joint_admittance_8dof/solver/qp_builder.py"),
        "kinematics_urdf": Path("/media/camp/EXT_DRIVE/RealUS_playground/rm75_control/rm75_control/assets/robots/rm75_6f_8dof/RM75-6F-8dof.urdf"),
        "collision_urdf": Path("/media/camp/EXT_DRIVE/RealUS_playground/rm75_control/rm75_control/assets/robots/rm75_6f_8dof/RM75-6F-8dof.collision.urdf"),
        "collision_pairs": Path("/media/camp/EXT_DRIVE/RealUS_playground/rm75_control/rm75_control/assets/robots/rm75_6f_8dof/collision_pairs.yaml"),
        "qp_replay": REPAIR / "dird" / "qp_replay.py",
        "scan_replay": OLD / "scan_replay.py",
        "scan_tracking_profile": OLD / "scan_tracking_profile.yaml",
        "scan_reference": Path("/media/camp/EXT_DRIVE/RealUS_playground/rm75_control/rm75_control/control/joint_admittance_8dof/scan_reference.py"),
    }
    # These are the archived hashes from the prior sealed native run and its
    # audit.  They are deliberately explicit rather than inferred from a new
    # controller instantiation.
    prior_hashes = {
        "native_binary": "cc34bf96b772ef463aa2a9468153e649da550192b8486d35ed52145ea07e7d07",
        "controller_yaml": "71134e9e16a952e688085a996ab192b41886b4e12094949082b748f0f186748f",
        "loop": "2f94fb2a88aaa518f3c078c844858aa89f40e6749aac7d315fa63d5659dc2a40",
        "qp_builder": "1953302f0e20990c4ac762c78a22e135344ebaf0e673c948a3cee5e1b5fcc57a",
        "kinematics_urdf": "7578310c5ce7b03a301c171175fd4eca2bd8e4b15903a7a9a3ed9bfc366e4ec0",
        "collision_urdf": "ba6f49b4fd9cba74107af25ecd82afccac156a94b194a6acb06a2aadebf1d197",
        "collision_pairs": "8d4b96e09ec8bdfd33a7835fb77fa4dcb720551e7d55d28c063994b99c392318",
        "qp_replay": "18f659fcd87a80461d582137dcdaa6c3cc70712ed0ad8ce26582f2f5ad33b36c",
        "scan_replay": "2ca8647ec664836f71072097d82d645e0c0a5f5e736e69537b50ecb9cc1c9461",
        "scan_tracking_profile": "8368eff9e808c61991642af55d0ef8128a64bc165f4df52d1d74b6d05113413d",
        "scan_reference": "9dc0f3008673eb25fced09db2129a3548c20e0547779136070ac1e00ea1ed757",
    }
    current_hashes = {key: sha(path) for key, path in source_paths.items()}
    hash_match = {key: current_hashes[key] == prior_hashes[key] for key in current_hashes}

    # The parent agent changed only the scan adapter/profile after the sealed
    # baseline-v2 replay, to expose the J1 guard as an explicit causal
    # variant.  Keep two notions separate: (a) hashes recorded by the saved
    # run, and (b) current files on disk after that concurrent edit.
    lower_level_keys = [
        key for key in prior_hashes
        if key not in {"scan_replay", "scan_tracking_profile"}
    ]
    lower_level_hashes_match = {
        key: hash_match[key] for key in lower_level_keys
    }
    sealed_profiles = {
        case["name"]: case.get("profile", {})
        for case in json_load(BASELINE / "RESULTS.json")["cases"]
    }
    profile_hash_mapping = {
        "wrapper_sha256": "qp_replay",
        "native_binary_sha256": "native_binary",
        "config_sha256": "controller_yaml",
        "loop_sha256": "loop",
        "qp_sha256": "qp_builder",
        "kinematics_urdf_sha256": "kinematics_urdf",
        "scan_profile_sha256": "scan_tracking_profile",
        "scan_reference_sha256": "scan_reference",
    }
    recorded_profile_hash_match = {}
    for case_name, profile in sealed_profiles.items():
        recorded_profile_hash_match[case_name] = {
            field: profile.get(field) == prior_hashes[hash_key]
            for field, hash_key in profile_hash_mapping.items()
        }
    recorded_profiles_consistent = (
        len({json.dumps(profile, sort_keys=True) for profile in sealed_profiles.values()}) == 1
    )
    sealed_baseline_metadata_matches = (
        recorded_profiles_consistent
        and all(all(fields.values()) for fields in recorded_profile_hash_match.values())
    )

    # Compare the current adapter/profile to the archived source captured at
    # audit time.  This is intentionally textual/read-only; it does not load
    # or instantiate the controller.
    sealed_scan_replay = REPAIR / "audit" / "followup3_20261110" / "source_snapshot" / "scan_replay.py"
    sealed_scan_profile = REGRESSION / "production_before_fix" / "scan_tracking_profile.yaml"
    adapter_diff = changed_lines(sealed_scan_replay, source_paths["scan_replay"])
    profile_diff = changed_lines(sealed_scan_profile, source_paths["scan_tracking_profile"])
    current_parent_j1_edit = {
        "scan_replay_current_sha256": current_hashes["scan_replay"],
        "scan_profile_current_sha256": current_hashes["scan_tracking_profile"],
        "scan_replay_sealed_sha256": prior_hashes["scan_replay"],
        "scan_profile_sealed_sha256": prior_hashes["scan_tracking_profile"],
        "scan_replay_changed_lines": adapter_diff,
        "scan_profile_changed_lines": profile_diff,
        "contains_j1_overfold_policy_support": "j1_overfold_limit" in source_paths["scan_replay"].read_text(),
        "contains_final_minimal_profile_flag": "extra_posture_limits: false" in source_paths["scan_tracking_profile"].read_text(),
        "contains_urdf_profile_policy": "j1_overfold_limit: urdf" in source_paths["scan_tracking_profile"].read_text(),
        "interpretation": (
            "Current scan_replay contains the parent J1 guard-policy support and the current "
            "profile requests extra_posture_limits=false; baseline_v2 was sealed before these edits."
        ),
    }

    # The guard-only run is already sealed by the parent.  Read its result as a
    # causal comparison; never rerun it here.
    j1_guard_summary = {}
    j1_results_path = J1_GUARD / "RESULTS.json"
    if j1_results_path.exists():
        j1_results = json_load(j1_results_path)
        for case in j1_results.get("cases", []):
            result = case.get("result", {})
            j1_guard_summary[case.get("case", "unknown")] = {
                "j1_cap_rad": case.get("j1_cap_rad"),
                "position_error_max_mm": 1000.0 * float(result["position_error_max_m"]),
                "rotation_error_p95_deg": float(np.degrees(result["rotation_error_p95_rad"])),
                "end_position_error_mm": 1000.0 * float(case["end_position_m"]),
                "qp1_status_counts": result.get("qp1_status_counts"),
                "motion_tracking_pass_2mm_2deg": result.get("motion_tracking_pass_2mm_2deg"),
                "task_progress_p05": result.get("task_progress_p05"),
                "native_config_sha256": sha(J1_GUARD / f"{case['case']}_native_config.txt"),
            }

    fresh_old_compare = {
        "path0": compare_traces(OLD / "path0_smooth_baseline_trace.npz", BASELINE / "old_path0_trace.npz"),
        "path1": compare_traces(OLD / "path1_smooth_baseline_trace.npz", BASELINE / "old_path1_trace.npz"),
    }
    native_configs = {}
    for old_name, fresh_name in [
        ("path0_smooth_baseline_native_config.txt", "old_path0_native_config.txt"),
        ("path1_smooth_baseline_native_config.txt", "old_path1_native_config.txt"),
    ]:
        a, b = OLD / old_name, BASELINE / fresh_name
        native_configs[old_name] = {"byte_equal": a.read_bytes() == b.read_bytes(), "old_sha256": sha(a), "fresh_sha256": sha(b)}
    native_configs["fresh_new_first_vs_fresh_old"] = {
        "byte_equal": (BASELINE / "old_path0_native_config.txt").read_bytes() == (BASELINE / "new_first_nominal_native_config.txt").read_bytes(),
        "sha256": sha(BASELINE / "new_first_nominal_native_config.txt"),
    }

    # Fresh controlled-v2 cases are all precomputed by the parent agent.  We
    # only read their saved outputs here.
    fresh_cases = {}
    for case in json_load(BASELINE / "RESULTS.json")["cases"]:
        result = case["result"]
        fresh_cases[case["name"]] = {
            "same_time_max_position_mm": 1000.0 * float(case["same_time_max_position_m"]),
            "same_time_max_rotation_deg": float(np.degrees(case["same_time_max_rotation_rad"])),
            "last_same_time_position_mm": 1000.0 * float(case["last_same_time_position_m"]),
            "n_steps": result.get("n_steps"),
            "settle_s": result.get("settle_s"),
            "nominal_path_duration_s": result.get("nominal_path_duration_s"),
            "qdot_entry": result.get("qdot_entry"),
            "acceleration_entry": result.get("acceleration_entry"),
            "qp1_status_counts": result.get("qp1_status_counts"),
            "failures": result.get("failures"),
            "motion_tracking_pass_2mm_2deg": result.get("motion_tracking_pass_2mm_2deg"),
            "profile": case.get("profile"),
        }

    # A small, fixed selection of current B segments demonstrates the changed
    # reference and the later stationary-hold regime without rerunning them.
    current_segments = {
        "identity_first": NEW / "h0_a0_identity_0_trace.npz",
        "t2_first_move": NEW / "h0_a0_t2_-0.08m_1_trace.npz",
        "t2_final_hold": NEW / "h0_a0_t2_-0.08m_5_trace.npz",
    }
    current_segment_metrics = {}
    current_ep = current_results["episodes"]
    seg_lookup = {}
    for episode in current_ep:
        for seg in episode["segments"]:
            seg_lookup[seg["trace_file"]] = seg
    for label, path in current_segments.items():
        seg = seg_lookup[path.name]
        current_segment_metrics[label] = {
            "target": seg["target"],
            "q_actual_entry": seg["q_actual_entry"],
            "requested_speed_m_s": seg.get("requested_speed_m_s"),
            "L0_motion_estimate_s": seg.get("L0_motion_estimate_s"),
            "reference_uses_measured_start": seg.get("reference_uses_measured_start"),
            "same_time": seg.get("same_time_measured_tracking"),
            "trace_metrics": trace_metrics(path),
        }

    old_smooth_metrics = {}
    for path_index in (0, 1):
        run = old_baseline[f"path{path_index}_smooth_baseline"]
        old_smooth_metrics[f"path{path_index}"] = {
            "full_position_max_mm": 1000.0 * run["full_position_max_m"],
            "full_rotation_max_deg": run["full_rotation_max_deg"],
            "middle_phase": run["regional"]["middle_phase"],
            "geometry": run["geometry"],
            "trace_metrics": trace_metrics(OLD / f"path{path_index}_smooth_baseline_trace.npz", run["nominal_path_duration_s"]),
        }

    public_metrics = []
    for row in public:
        native = row["native"]
        public_metrics.append({
            "id": row["id"],
            "position_max_mm": 1000.0 * native["position_error_max_m"],
            "rotation_p95_deg": float(np.degrees(native["rotation_error_p95_rad"])),
            "speed_m_s": native["speed_m_s"],
            "settle_s": native["settle_s"],
            "path_duration_s": native["nominal_path_duration_s"],
            "qdot_entry": native["qdot_entry"],
            "reference_kind": native.get("actual_reference_kind"),
        })

    output = {
        "scope": "Read-only comparison; no native replay or controller/source edits by this worker",
        "input_hashes": {"old_phase2_inputs": sha(OLD_INPUT), "baseline_v2_results": sha(BASELINE / "RESULTS.json"), "new_results": sha(NEW / "RESULTS.json")},
        "source_hash_comparison": {
            "prior_sealed_hashes": prior_hashes,
            "current_on_disk_hashes": current_hashes,
            "match": hash_match,
            # This remains false when the current scan adapter/profile has a
            # post-baseline edit.  It must not be read as a controller-wide
            # regression signal.
            "all_current_on_disk_hashes_match": all(hash_match.values()),
            "lower_level_current_hashes_match": lower_level_hashes_match,
            "lower_level_current_hashes_all_match": all(lower_level_hashes_match.values()),
            "sealed_baseline_metadata_matches_prior_hashes": sealed_baseline_metadata_matches,
            "recorded_baseline_profile_hash_match": recorded_profile_hash_match,
            "recorded_baseline_profiles_consistent": recorded_profiles_consistent,
            "current_parent_j1_edit": current_parent_j1_edit,
        },
        "controller_variant_comparison": {
            "original_sealed_baseline": {
                "branch_barrier_j1_overfold_abs_rad": 2.4434609527920612,
                "branch_barrier_j1_overfold_abs_deg": 140.0,
                "initial_q1_rad": float(old_q[1]),
                "initial_q1_deg": float(np.degrees(old_q[1])),
                "urdf_symmetric_abs_limit_rad": 3.106,
                "status": "used by baseline_v2 and the failing new-first replay",
            },
            "single_j1_urdf_guard": {
                "status": "sealed read-only causal contrast",
                "changed_only": "branch_barrier.j1_overfold_abs_rad: 2.4434609527920612 -> 3.106",
                "other_limits_and_settings": "retained according to the guard run scope",
                "results": j1_guard_summary,
            },
            "final_minimal_user_requested": {
                "status": "adapter/profile flag present; no corresponding final-minimal result file yet",
                "intended_scope": "remove artificial branch-barrier/posture preference and retain physical q/v/a/jerk/collision constraints",
                "metrics": None,
            },
        },
        "fresh_old_trace_comparison": fresh_old_compare,
        "native_config_comparison": native_configs,
        "initial_q": {
            "old_branch4_path0_and_path1_q_seed": old_q.tolist(),
            "new_a0_identity_q_seed": context_q["a0_identity"].tolist(),
            "new_a0_identity_vs_old": q_comparison(old_q, context_q["a0_identity"]),
            "new_a0_t2_minus08m_q_seed": context_q["a0_t2_-0.08m"].tolist(),
            "new_a0_t2_minus08m_vs_old": q_comparison(old_q, context_q["a0_t2_-0.08m"]),
        },
        "old_smooth_success": old_smooth_metrics,
        "old_public_capability_success": public_metrics,
        "fresh_controlled_v2": fresh_cases,
        "current_B_saved_segments": current_segment_metrics,
        "controller_stack": {
            "old_and_fresh_backend": "native",
            "old_and_fresh_tcp_mode": "urdf",
            "old_and_fresh_native_model_clock": "deterministic",
            "offline_ideal_measured_q": True,
            "realman_vendor_hardware_controller_instantiated": False,
            "evidence": [
                "QPReplayBridge creates JointIkController with backend=native and sends SHM requests to wbc_rt; no RealManBackend/session is constructed.",
                "Saved scopes explicitly exclude physical hardware, image/contact/force outer loops, and device communication.",
            ],
        },
        "state_and_reference_contract": {
            "old_smooth": {
                "run_return_state_false": "return_state=False" in (OLD / "run_smooth_reference.py").read_text(),
                "qdot_entry_zero": True,
                "acceleration_entry_zero": True,
                "settle_s": 2.0,
                "reference": "17 waypoint WorldPoseSplineReference; symmetric .8s ramp; 2cm/s",
            },
            "old_public_capability": {
                "independent_query_no_initial_state": True,
                "reference": "relative_path(FK(q), FK(q) @ pure +/-2cm body translation); no target rotation",
            },
            "new_B": {
                "runner_uses_initial_state": "initial_state=state" in (ROOT / "planning" / "run_native_subset.py").read_text(),
                "runner_saves_exit_state": "state=ans['native']['exit_state']" in (ROOT / "planning" / "run_native_subset.py").read_text(),
                "reference": "41-sample DMP body primitive transformed by T_world_body; short segments; .5s settle",
            },
        },
        "interpretation": [
            "The old path0/path1 fresh controlled replay matches every saved controller-state array exactly except tick_compute_seconds (wall timing), and its native_config text is byte-identical. The old passing result is therefore reproducible; there is no evidence of a broad controller/binary/config regression.",
            "The new identity first segment is also exactly reproducible from its saved trace, but it is a different command: about 15.159 mm translation with net 0.2657995 rad yaw, about 1.516 s path and .5 s settle. The old successful paths were 73.9/75.6 mm, had zero net rotation with intermediate orientation spans about .391/.429 rad, 4.50/4.586 s paths and 2 s settle. The old tests did not cover this short coupled yaw/translation B reference.",
            "The new t2 first move is nearly pure translation and remains below 0.4 mm in the saved replay, while the identity coupled-rotation segment reaches 17.059 mm/2.291 deg. This within-run contrast points to reference/coverage, not an all-path native regression.",
            "Long holds are a separate coverage gap: the fresh 10 s stationary hold drifts to 7.70 mm in the nominal posture and 2.44 mm in the stress posture, whereas the old smooth runs only had a 2 s settle and max errors about 1.02 mm. New B also carries complete exit state/history between segments; the old smooth paths were isolated zero-history starts.",
            "A concrete additional constraint is present in the sealed baseline: branch_barrier.j1_overfold_abs_rad=2.44346 (140 deg), while the common initial q1 is -2.60347 rad (-149.164 deg) and the URDF permits about +/-3.106 rad. At that sign, the lower J1 velocity box is clamped at zero, so further negative J1 motion is disallowed and redundant joints can circle to satisfy the Cartesian request. This is a real hard-bound mechanism, not a claim based only on QP solved status.",
            "The parent-sealed J1-URDF-only guard contrast changes only that artificial cap to 3.106 rad: the same identity motion reports 0.1027 mm maximum position error, 0.0796 deg rotation p95, solved all 405 ticks, and passes the 2 mm/2 deg gate; its 10 s hold reports 0.0051 mm and 0.00023 deg p95. This strongly supports the J1 wall as the dominant cause for this B motion/hold pair, while the final user-requested minimal variant (removing the artificial branch-barrier/posture preferences while retaining physical q/v/a/jerk/collision limits) still has no result in this directory.",
            "The baseline-v2 run itself predates the concurrent adapter/profile edits: its recorded native binary, controller/config, loop, QP, URDF and scan/reference hashes match the sealed prior set. Current on-disk scan_replay now contains J1-policy support and the profile requests extra_posture_limits=false; those edits were not used by baseline-v2 and are not evidence of a broad source regression.",
            "The RealMan machine being normal is not a contradictory result: these saved tests are an offline custom native wbc_rt QP with ideal measured-q feedback, not the vendor hardware communication/controller path. They diagnose this native replay/reference contract only.",
        ],
    }
    (HERE / "COMPARISON.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")

    report = [
        "Prior-success comparison (read-only)",
        "====================================",
        "Scope: compare sealed smooth-reference/public capability runs with the new B native replay. No controller was instantiated or rerun by this worker.",
        "",
        "1. Strong no-regression control",
        "--------------------------------",
        "The parent-controlled-v2 fresh replays of old path0/path1 match every saved state/pose/command/residual array exactly, apart from tick_compute_seconds (wall timing). The old trace bundle has one extra vel_ff field; all common non-wall fields are identical.",
        "The old and fresh native_config.txt files are byte-identical (SHA256 7080702d84ed0c8e911c2a790af823af029eb7b42f28f8f8e9dd50e429690ba3). Fresh old path0/path1 retain 1.035/1.022 mm full maximum errors and pass the 2 mm/2 degree motion gate.",
        "",
        "2. What changed in B",
        "---------------------",
        "The old smooth paths start from the same branch4 q seed and use 17 pose waypoints, approximately 73.9/75.6 mm translation, zero net rotation, 0.02 m/s, about 4.50/4.586 s path time, and 2 s settle. The B identity first segment starts from the same q to within 5.44e-6, but commands a 15.159 mm path with net -0.2657995 rad yaw, about 1.516 s path time, and 0.5 s settle. Its saved trace reaches 17.059 mm and 2.291 degrees; the fresh controlled replay is identical.",
        "A B stress-context first move with essentially pure translation remains about 0.394 mm/0.0395 degrees. This is a within-stack contrast: the failing coupled short yaw/translation reference is not evidence that every native path regressed.",
        "",
        "3. Hold/state coverage",
        "----------------------",
        "Old smooth runs were isolated (return_state=False), with zero qdot/acceleration entry and only 2 s settle. The B runner carries exit_state and complete history into the next segment. Fresh 10 s holds expose a distinct long-hold issue: 7.70 mm nominal and 2.44 mm stress drift, both with QP solved and no failures. This regime was not covered by the old passing scan tests.",
        "",
        "4. Controller/source identity",
        "----------------------------",
        "The saved baseline-v2 metadata records the prior native binary, controller YAML, loop, QP builder, URDF, scan profile/reference and wrapper hashes consistently; the lower-level current hashes still match those sealed values. Both old and new saved runs use backend=native, tcp_mode=urdf, deterministic model time, and the same effective native config.",
        "The current on-disk scan_replay.py and scan_tracking_profile.yaml no longer match the sealed hashes: the parent added J1-policy support and the current profile now carries extra_posture_limits=false after baseline-v2. The saved baseline therefore must be interpreted as pre-adapter-edit; this comparison does not label the current adapter/profile edit as a broad controller regression. The stack is the custom offline wbc_rt native QP with ideal measured-q feedback; no RealMan vendor hardware session is instantiated.",
        "",
        "5. Constraint causal contrast",
        "-----------------------------",
        "The original sealed configuration has branch_barrier.j1_overfold_abs_rad=2.44346 rad (140 deg). The shared initial q1 is -2.60347 rad (-149.164 deg), while the URDF allows approximately +/-3.106 rad. With the negative sign, the lower J1 box is clamped at zero at the initial posture, so a negative J1 component of the B identity twist is unavailable and redundant joints can take a large detour.",
        "The parent-sealed J1-URDF-only contrast changed only that cap to 3.106 rad. On the same identity motion it reports 0.1027 mm maximum position error, 0.0796 deg rotation p95, 405/405 solved ticks and a passed 2 mm/2 deg gate; its 10 s hold reports 0.0051 mm and 0.00023 deg rotation p95. This is direct causal evidence for the artificial J1 wall in this case, not an inference from 'QP solved'.",
        "The current profile/source now exposes the final user-requested minimal flag (remove artificial branch-barrier/posture preferences; retain physical q/v/a/jerk/collision constraints), but no corresponding final-minimal result file is present in this directory, so it is not claimed verified here.",
        "",
        "Conclusion",
        "----------",
        "The old smooth tests are reproducible and do not show a broad native-controller regression. The original B failure is explained by a concrete artificial J1 hard wall exposed by B's short coupled yaw/translation reference; the guard-only causal replay removes that failure. Long-hold behavior was also outside the old 2 s coverage and should remain a separate check. RealMan hardware being normal is compatible because these saved experiments use the offline custom native QP path, not the vendor hardware communication/controller path. Await the final minimal-constraint replay before treating the full B execution contract as validated.",
    ]
    (HERE / "COMPARISON_REPORT.txt").write_text("\n".join(report) + "\n")
    print(json.dumps({
        "current_on_disk_hashes_all_match": output["source_hash_comparison"]["all_current_on_disk_hashes_match"],
        "lower_level_current_hashes_all_match": output["source_hash_comparison"]["lower_level_current_hashes_all_match"],
        "sealed_baseline_metadata_matches": output["source_hash_comparison"]["sealed_baseline_metadata_matches_prior_hashes"],
        "output": str(HERE / "COMPARISON.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
