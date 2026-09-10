"""Detached, reproducible contact-QP experiments. This executable has no robot I/O.

Design: python -m peirastic.apps.contact_qp_experiment --split design --output ...
Freeze: add --freeze-only --manifest ... after design and execution-stage review.
Acceptance requires that exact manifest, all prescribed seeds and all variants.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from peirastic.contact_qp.evaluation import force_metrics, paired_report
from peirastic.contact_qp.features import FeatureConfig, FeatureExtractor, LatestObservation, registration_revision
from peirastic.contact_qp.coverage import PathCoverage
from peirastic.contact_qp.geometry import accepted_alpha, motion_basis
from peirastic.contact_qp.history import MotionHistory, ResponseConsistency, ResponseConfig
from peirastic.contact_qp.plant import FiniteAreaPlant, PlantConfig, SCENARIOS
from peirastic.contact_qp.qp import ContactQp, QpConfig, QpInput
from peirastic.contact_qp.types import ProbeGeometry, TwistConstraints
from peirastic.realman8dof.force.contact_nominal import build_contact_nominal, build_contact_position, icra_contact_payload
from peirastic.realman8dof.force.tff import compose_tff
from rm75_control.control.admittance_common.reference import MotionReference
from peirastic.scan_path import ForearmReference, make_spec

ROOT = Path(__file__).resolve().parents[2]
ACCEPTANCE_PATH = ROOT / "MD/contact_qp/acceptance_v1.json"
VARIANTS = ("baseline", "image_only", "no_aperture", "full", "full_consistency",
            "matched_scan_speed", "matched_rocking")


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def settings():
    plant = PlantConfig()
    features = FeatureConfig(width=plant.image_width, height=plant.image_height,
                             effective_delay_s=plant.image_delay_s, calibration_version="synthetic_v1")
    qp = QpConfig(angle_limit_rad=.35, lateral_windows=features.lateral_windows)
    return plant, features, qp


def frozen_manifest():
    p, f, q = settings()
    source_files = list((ROOT / "peirastic").rglob("*.py"))
    source_files += list((ROOT / "peirastic/configs").glob("*.yaml"))
    source_files += list((ROOT / "rm75_control/rm75_control").rglob("*.py"))
    for suffix in ("*.cpp", "*.hpp", "CMakeLists.txt"):
        for directory in ("src", "include"):
            source_files += list((ROOT / "rm75_control/native/wbc_rt" / directory).rglob(suffix))
    source_files += [ACCEPTANCE_PATH, ROOT / "rm75_control/native/wbc_rt/CMakeLists.txt"]
    from importlib.metadata import version, PackageNotFoundError
    versions = {}
    for package in ("numpy", "scipy", "proxsuite", "pin", "PyYAML"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "unavailable"
    return dict(schema_version=1, algorithm="contact_qp_synthetic_v1",
                plant=plain(asdict(p)), features=plain(asdict(f)), qp=plain(asdict(q)),
                response=asdict(ResponseConfig()), nominal_payload=icra_contact_payload(), library_versions=versions,
                hashes={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in sorted(set(source_files))},
                hardware_geometry_verified=False, hardware_energy_certified=False)


def observation_matches(obs, feature_config, *, registration_version):
    return (obs is not None and obs.window_version == feature_config.window_version
            and obs.calibration_version == feature_config.calibration_version
            and obs.registration_version == registration_version)


def save_source_snapshot(destination, manifest):
    from zipfile import ZipFile, ZIP_DEFLATED
    with ZipFile(destination/"sources.zip", "w", compression=ZIP_DEFLATED) as archive:
        for path, expected in manifest["hashes"].items():
            content = (ROOT/path).read_bytes()
            if hashlib.sha256(content).hexdigest() != expected:
                raise RuntimeError(f"source changed during experiment freeze: {path}")
            archive.writestr(path, content)


class FiniteIntervalScanReference:
    """Shared production ramp, sampled over one bounded candidate interval.

    Position feedback targets the accepted clock position. Feedforward is the
    candidate displacement divided by the complete execution hold duration. The caller owns the
    accepted clock; sampling alone never advances it. These synthetic L paths
    have constant desired orientation, so their angular feedforward is zero.
    """
    def __init__(self, length_m, speed_m_s, dt_s):
        self.length_m, self.speed_m_s, self.dt_s = length_m, speed_m_s, dt_s
        self.set_origin(np.zeros(6))

    def set_origin(self, pose0, *, t_s=None):
        start = np.asarray(pose0, dtype=float).reshape(6).copy()
        end = start.copy(); end[1] += self.length_m
        self.reference = ForearmReference(make_spec(start, end, "L", "DtP", 0, speed=self.speed_m_s))
        self.duration_s = self.reference.duration_s

    def candidate_duration(self, t_s):
        return min(self.dt_s, max(0., self.duration_s-float(t_s)))

    def exhaustion_reason(self, t_s):
        if t_s >= self.duration_s:
            return "reference_clock_end"
        # The final polynomial ramp can round to its exact endpoint pose before
        # its scalar clock runs out. This is completion evidence, not progress.
        if (t_s >= self.duration_s-self.reference.ramp and
                np.array_equal(self.reference.sample(t_s).pose_d,
                               self.reference.sample(self.duration_s).pose_d)):
            return "reference_geometry_roundoff"
        return ""

    def commit_time(self, t_s, alpha):
        accepted = float(np.clip(alpha, 0., 1.))
        if accepted == 0.:
            return float(t_s)
        advanced = min(self.duration_s, t_s+accepted*self.candidate_duration(t_s))
        # Floating-point clock exhaustion only, after positive publication.
        # The separate measured endpoint-distance condition is unchanged.
        tolerance = 8*np.finfo(float).eps*max(self.duration_s, 1.)
        return self.duration_s if self.duration_s-advanced <= tolerance else advanced

    def sample(self, t_s):
        start = float(np.clip(t_s, 0., self.duration_s))
        interval = self.candidate_duration(start)
        first = self.reference.sample(start)
        last = self.reference.sample(min(self.duration_s, start+interval))
        velocity = np.zeros(6)
        if interval > 0.:
            velocity[:3] = (last.pose_d[:3]-first.pose_d[:3])/self.dt_s
        return MotionReference(first.pose_d.copy(), velocity, start)


class MatchedMotion:
    """Ordered path replay preserves alpha-zero dwell instead of deadlocking."""
    def __init__(self, trace, dt):
        # Mechanical stopping is independently executed after this replay.
        # It is not a task dwell and must not freeze a comparator's scan clock.
        scan = np.asarray(trace.get("phase", ["scan"]*len(trace["path_m"]))) == "scan"
        self.path = np.asarray(trace["path_m"])[scan]
        self.alpha = np.asarray(trace["alpha"])[scan]
        self.omega = np.asarray(trace["omega_measured"])[scan]
        self.dt, self.index, self.dwell = dt, 0, 0.
        self.rotation_budget = float(np.sum(abs(self.omega))*dt)

    def speed(self, path):
        while self.index < len(self.path)-1:
            if self.alpha[self.index] <= 1e-10 and path >= self.path[self.index]-1e-6:
                self.dwell += self.dt
                if self.dwell >= self.dt:
                    self.index += 1
                    self.dwell = 0.
                return 0.
            if path >= self.path[self.index+1]-1e-10:
                self.index += 1
                continue
            break
        return float(self.alpha[self.index])

    def rocking(self, path, used):
        if not self.path.size:
            return 0.
        i = int(np.clip(np.searchsorted(self.path, path, side="right")-1, 0, len(self.path)-1))
        envelope = float(abs(self.omega[i]))
        return min(envelope, max(0., self.rotation_budget-used)/self.dt)



def image_trace_fields(observation, now_s):
    return dict(image_frame_seq=-1 if observation is None else observation.frame_seq,
                image_effective_time_s=-1. if observation is None else observation.effective_time_s,
                image_received_time_s=-1. if observation is None else observation.received_time_s,
                image_age_s=-1. if observation is None else now_s-observation.effective_time_s,
                quality_version="" if observation is None else json.dumps(observation.version))


def run_case(scenario, seed, variant, source_trace=None):
    pc, fc, base_qc = settings()
    plant = FiniteAreaPlant(scenario, seed, pc)
    geometry = ProbeGeometry.synthetic(pc.half_length_m)
    law, _ = build_contact_nominal(pc.dt_s)
    law.reset(pose=plant.pose(), f_ext=plant.control_wrench())
    reference = FiniteIntervalScanReference(pc.path_length_m, pc.scan_speed_m_s, pc.dt_s)
    position = build_contact_position(reference)
    selection = position.cfg.track_axes.copy()
    qc = replace(base_qc, enable_aperture=variant not in ("image_only", "no_aperture"),
                 enable_force_priority=variant != "image_only", enable_consistency=variant == "full_consistency",
                 enable_progress_loss=variant != "image_only")
    solver = ContactQp(qc)
    use_qp = variant in ("full", "full_consistency", "image_only", "no_aperture")
    matcher = MatchedMotion(source_trace, pc.dt_s) if source_trace else None
    history, response = MotionHistory(geometry), ResponseConsistency()
    extractor, latest, images = FeatureExtractor(fc), LatestObservation(), deque()
    expected_registration = registration_revision(fc, source_id="synthetic-camera")
    previous = np.zeros(6)
    image_stride = max(1, round(pc.image_period_s/pc.dt_s))
    active_tick = 0
    source_seq = 0
    t_ref = 0.
    mechanical_coverage, acoustic_coverage = [PathCoverage(pc.path_length_m, max_gap_s=.02) for _ in range(2)]
    image_coverage = PathCoverage(pc.path_length_m, max_gap_s=.08)
    actual_rotation = 0.
    reason = ""
    status = "aborted"
    total_timeout = 4*pc.path_length_m/pc.scan_speed_m_s+12.
    record = {name: [] for name in ("time_s", "path_m", "reference_s", "force_n", "physical_force_n",
                "alpha", "vn", "omega", "omega_measured", "theta", "quality_left", "quality_center", "quality_right",
                "coupling_left", "coupling_center", "coupling_right", "acoustic_left", "acoustic_center", "acoustic_right", "gamma_left",
                "gamma_right", "slack_left", "slack_right", "force_constraint_active", "aperture_active",
                "omega_width_mechanical", "omega_width_force", "omega_width_complete", "controller_ms",
                "nominal_twist", "qp_twist", "sent_twist", "measured_twist", "phase", "committed", "h_ref_s", "h_exec_s", "reference_exhaustion_reason",
                "image_frame_seq", "image_effective_time_s", "image_received_time_s", "image_age_s", "quality_version")}
    force_desired = np.array([0, 0, 4., 0, 0, 0])
    initial_contact = False
    begin_wall = time.perf_counter()
    steps = round((pc.warmup_s+total_timeout)/pc.dt_s)
    qp_failure = None
    previous_measured = plant.velocity.copy()
    mechanical_violations = []
    for tick in range(steps):
        now = 1.+plant.t
        active = plant.t >= pc.warmup_s-1e-9
        command_sent = False
        finished = False
        history.add(now, plant.velocity, tick)
        if tick >= round((pc.warmup_s-.3)/pc.dt_s) and tick % image_stride == 0:
            obs, _ = extractor.extract(plant.render(source_seq), frame_seq=source_seq, source_id="synthetic-camera",
                       capture_time_s=now+pc.image_delay_s,
                       received_time_s=now+pc.image_delay_s+pc.feature_processing_delay_s)
            source_seq += 1
            truth = bool(np.min(plant.window_acoustic_coupling(fc)[[0, 2]]) >= .9)
            images.append((obs, float(plant.path), truth, active))
        while images and images[0][0].received_time_s <= now+1e-12:
            delivered, captured_path, truth, scan_frame = images.popleft()
            if latest.accept(delivered) and scan_frame and observation_matches(delivered, fc, registration_version=expected_registration):
                image_coverage.observe(captured_path, truth and bool(delivered.valid[[0, 2]].all()), delivered.effective_time_s)
        obs = latest.observation
        if not observation_matches(obs, fc, registration_version=expected_registration):
            obs = None
        gamma = response.update(obs, history, now_s=now, max_age_s=qc.max_image_age_s)
        wrench = plant.control_wrench()
        if active and active_tick == 0:
            position.set_origin(plant.pose(), t_s=0.)
        path = position.sample(t_ref, plant.pose(), wrench)*selection if active else np.zeros(6)
        force_feedforward = position.last_path_twist*selection if active else np.zeros(6)
        kwargs = dict(dt_s=pc.dt_s, dt_actual=pc.dt_s, pose=plant.pose(), f_ext=wrench,
                      f_ext_raw=wrench, f_des=force_desired, path_twist=force_feedforward,
                      sensor_age_s=.001, feedback_age_s=.001, v_tcp_z_actual=plant.velocity[2], slack_norm=0.)
        if not active:
            out = law.update(**kwargs)
            command = out.v_force.copy()
        else:
            initial_contact |= bool(law.controller.contact_present)
            measured_acc = (plant.velocity-previous_measured)/pc.dt_s
            if (abs(plant.theta) > base_qc.angle_limit_rad+1e-8
                    or np.any(abs(plant.velocity) > base_qc.max_velocity+1e-8)
                    or np.any(abs(measured_acc) > base_qc.max_acceleration+1e-7)):
                mechanical_violations.append(dict(time_s=plant.scan_time, theta=plant.theta,
                                              measured_twist=plant.velocity.tolist(), acceleration=measured_acc.tolist()))
                reason = "actual_mechanical_limit"; break
            started = time.perf_counter()
            candidate_h = reference.candidate_duration(t_ref)
            out = law.prepare(measurement_id=tick, **kwargs)
            nominal = compose_tff(path, out.v_force, selection)
            alpha, slacks, diagnostics = 1., np.zeros(2), {}
            limit_v, limit_a = base_qc.max_velocity, base_qc.max_acceleration
            lo, hi = np.maximum(-limit_v, previous-limit_a*pc.dt_s), np.minimum(limit_v, previous+limit_a*pc.dt_s)
            lo[4] = max(lo[4], (-base_qc.angle_limit_rad-plant.theta)/pc.dt_s)
            hi[4] = min(hi[4], (base_qc.angle_limit_rad-plant.theta)/pc.dt_s)
            if np.any(lo > hi):
                law.abort(); reason = "mechanical_bounds_infeasible"; break
            mechanics = TwistConstraints(np.eye(6), lo, hi, valid_until_s=now+.01)
            if use_qp:
                result = solver.solve(QpInput(geometry, nominal, path, float(wrench[2]), pc.dt_s, now,
                              observation=obs, mechanical=mechanics, previous_twist=previous,
                              measured_angle=plant.theta, gamma=gamma), interval_diagnostics=tick % image_stride == 0)
                diagnostics = result.diagnostics
                if result.qp_twist is None:
                    qp_failure = plain(diagnostics)
                    law.abort(); reason = result.status.value; break
                command, slacks = result.qp_twist.copy(), result.slack
                if result.hard_constraints.violation(command) > 1e-7:
                    law.abort(); reason = "final_constraint_recheck"; break
                alpha = accepted_alpha(motion_basis(path), result.alpha, command, command, qc.subspace_tolerance)
            else:
                command = nominal.copy()
                if variant == "matched_scan_speed" and matcher:
                    command += (matcher.speed(plant.path)-1.)*path
                command = np.clip(command, lo, hi)
                if variant == "matched_rocking" and matcher:
                    cap = matcher.rocking(plant.path, actual_rotation)
                    command[4] = np.clip(command[4], -cap, cap)
                if mechanics.violation(command) > 1e-8:
                    law.abort(); reason = "matched_motion_mechanically_infeasible"; break
                try:
                    alpha = accepted_alpha(motion_basis(path), 1., command, command, qc.subspace_tolerance)
                except ValueError as exc:
                    # A clipped comparator may leave the admitted path subspace.
                    # Preserve its failure and delayed stop tail in the paired
                    # record instead of dropping all seven variants in this unit.
                    law.abort(); reason = "progress_confirmation_rejected: "+str(exc); break
            final_force = command.copy(); final_force[[0, 1, 3, 5]] = 0.
            # Queue acceptance precedes software-state commit. Delayed physical
            # execution remains independently visible in plant.velocity.
            plant.command(command)
            command_sent = True
            law.commit_applied(final_force, final_full_twist=command, accepted_normal_z=command[2])
            t_ref = reference.commit_time(t_ref, alpha)
            duration_ms = (time.perf_counter()-started)*1000.
            mech_q, acoustic_q = plant.window_coupling(fc), plant.window_acoustic_coupling(fc)
            good, acoustic_good = bool(mech_q[[0, 2]].min() >= .9), bool(acoustic_q[[0, 2]].min() >= .9)
            current = float(np.clip(plant.path, 0., pc.path_length_m))
            mechanical_coverage.observe(current, good, now)
            acoustic_coverage.observe(current, acoustic_good, now)
            actual_rotation += abs(plant.velocity[4])*pc.dt_s
            def width(name):
                interval = diagnostics.get(name)
                return float(interval[1]-interval[0]) if interval is not None and len(interval) == 2 else -1.
            active_rows = diagnostics.get("active_hard_rows", ())
            values = dict(time_s=plant.scan_time, path_m=plant.path, reference_s=t_ref, force_n=wrench[2],
                physical_force_n=-plant.wrench_environment[2], alpha=alpha, vn=command[2], omega=command[4],
                omega_measured=plant.velocity[4], theta=plant.theta,
                quality_left=float(obs.quality[0]) if obs else -1., quality_right=float(obs.quality[2]) if obs else -1.,
                quality_center=float(obs.quality[1]) if obs else -1., coupling_center=mech_q[1], acoustic_center=acoustic_q[1],
                coupling_left=mech_q[0], coupling_right=mech_q[2], acoustic_left=acoustic_q[0], acoustic_right=acoustic_q[2],
                gamma_left=gamma[0], gamma_right=gamma[1], slack_left=slacks[0], slack_right=slacks[1],
                force_constraint_active=any("force" in str(x) for x in active_rows),
                aperture_active=any("aperture" in str(x) for x in active_rows),
                omega_width_mechanical=width("omega_interval_mechanical"),
                omega_width_force=width("omega_interval_force_priority"),
                omega_width_complete=width("omega_interval_complete"), controller_ms=duration_ms,
                nominal_twist=nominal.copy(), qp_twist=command.copy(), sent_twist=command.copy(),
                measured_twist=plant.velocity.copy(), phase="scan", committed=True,
                h_ref_s=candidate_h, h_exec_s=pc.dt_s, reference_exhaustion_reason=reference.exhaustion_reason(t_ref),
                **image_trace_fields(obs, now))
            for key in record:
                record[key].append(values[key])
            active_tick += 1
            if reference.exhaustion_reason(t_ref) and plant.path >= pc.path_length_m-1e-4:
                status = "complete"  # Final acquisition label is assigned after delivered-image drain.
                finished = True
        if not command_sent:
            plant.command(command)
        previous = command.copy()
        previous_measured = plant.velocity.copy()
        plant.step()
        if finished:
            final_acc = (plant.velocity-previous_measured)/pc.dt_s
            if (abs(plant.theta) > base_qc.angle_limit_rad+1e-8
                    or np.any(abs(plant.velocity) > base_qc.max_velocity+1e-8)
                    or np.any(abs(final_acc) > base_qc.max_acceleration+1e-7)):
                mechanical_violations.append(dict(time_s=plant.scan_time, theta=plant.theta,
                                                   acceleration=final_acc.tolist()))
                status, reason = "aborted", "actual_mechanical_limit"
            break
    if record["time_s"]:
        # An aborted logical proposal does not erase a delayed held command.
        # Record a bounded-deceleration stop through confirmed plant decay.
        stop_command = previous.copy()
        for _ in range(round(.25/pc.dt_s)):
            stop_command += np.clip(-stop_command, -base_qc.max_acceleration*pc.dt_s,
                                    base_qc.max_acceleration*pc.dt_s)
            tail_previous_velocity = plant.velocity.copy()
            plant.command(stop_command); plant.step()
            tail_acceleration = (plant.velocity-tail_previous_velocity)/pc.dt_s
            if record["time_s"]:
                values = {key: record[key][-1] for key in record}
                values.update(time_s=plant.scan_time, path_m=plant.path, alpha=0., vn=stop_command[2], omega=stop_command[4],
                              omega_measured=plant.velocity[4], theta=plant.theta,
                              force_n=plant.control_wrench()[2], physical_force_n=-plant.wrench_environment[2],
                              nominal_twist=np.zeros(6), qp_twist=np.zeros(6), sent_twist=stop_command.copy(),
                              measured_twist=plant.velocity.copy(), committed=False, phase="stop_tail", controller_ms=0.,
                              h_ref_s=0., h_exec_s=pc.dt_s)
                # No new scan frame is captured in braking; advance the age of
                # the exact last recorded observation instead of copying freshness.
                values["image_age_s"] = (plant.t+1.-values["image_effective_time_s"]
                                          if values["image_frame_seq"] >= 0 else -1.)
                mech_q, acoustic_q = plant.window_coupling(fc), plant.window_acoustic_coupling(fc)
                values.update(coupling_left=mech_q[0], coupling_right=mech_q[2], acoustic_left=acoustic_q[0],
                              acoustic_right=acoustic_q[2], coupling_center=mech_q[1], acoustic_center=acoustic_q[1])
                for key in record:
                    record[key].append(values[key])
                active_tick += 1
            actual_rotation += abs(plant.velocity[4])*pc.dt_s
            mechanical_coverage.observe(plant.path, plant.window_coupling(fc)[[0, 2]].min() >= .9, plant.t+1)
            acoustic_coverage.observe(plant.path, plant.window_acoustic_coupling(fc)[[0, 2]].min() >= .9, plant.t+1)
            if (abs(plant.theta) > base_qc.angle_limit_rad+1e-8
                    or np.any(abs(plant.velocity) > base_qc.max_velocity+1e-8)
                    or np.any(abs(tail_acceleration) > base_qc.max_acceleration+1e-7)):
                mechanical_violations.append(dict(time_s=plant.scan_time, theta=plant.theta,
                                                  acceleration=tail_acceleration.tolist()))
            while images and images[0][0].received_time_s <= plant.t+1.+1e-12:
                delivered, captured_path, truth, scan_frame = images.popleft()
                if latest.accept(delivered) and scan_frame and observation_matches(delivered, fc, registration_version=expected_registration):
                    image_coverage.observe(captured_path, truth and bool(delivered.valid[[0, 2]].all()), delivered.effective_time_s)
    elapsed = active_tick*pc.dt_s
    if status == "aborted" and not reason:
        reason = "timeout"
    if not initial_contact:
        reason, status = "initial_contact_not_confirmed", "aborted"
    if status != "aborted":
        status = "complete" if image_coverage.valid_length_m >= pc.path_length_m-1.01e-4 else "complete_with_gaps"
    if mechanical_violations:
        reason, status = "actual_mechanical_limit", "aborted"
    if np.any(abs(plant.velocity) > np.array([1e-6]*3+[1e-5]*3)):
        reason, status = "stop_not_confirmed", "aborted"
    metrics = force_metrics(record["force_n"]) if record["force_n"] else {k: float("nan") for k in
                ("rmse", "absolute_mean_bias", "peak_absolute_error")}
    widths = {}
    for name in ("mechanical", "force", "complete"):
        a = np.asarray(record["omega_width_"+name]); a = a[a >= 0]
        widths["omega_width_"+name+"_mean"] = float(a.mean()) if a.size else None
    times = np.asarray(record["controller_ms"])
    valid_path, acoustic_valid_path = mechanical_coverage.valid_length_m, acoustic_coverage.valid_length_m
    result = dict(scenario=scenario, seed=int(seed), variant=variant, status=status, reason=reason,
                  valid=status != "aborted" and elapsed < total_timeout, elapsed_s=elapsed,
                  reference_s=t_ref, measured_forward_m=mechanical_coverage.maximum_path_m,
                  bad_path_m=pc.path_length_m-valid_path,
                  valid_coverage_fraction=valid_path/pc.path_length_m,
                  valid_coverage_rate_m_s=valid_path/max(elapsed, pc.dt_s),
                  acoustic_bad_path_m=pc.path_length_m-acoustic_valid_path,
                  acoustic_coverage_fraction=acoustic_valid_path/pc.path_length_m,
                  registered_image_coverage_m=image_coverage.valid_length_m,
                  invalid_forward_travel_m=mechanical_coverage.invalid_forward_travel_m,
                  qp_failure=qp_failure,
                  mechanical_violations=mechanical_violations,
                  angular_travel_rad=actual_rotation,
                  central_acoustic_gap_fraction=float(np.mean(np.asarray(record["acoustic_center"]) < .9)) if active_tick else None,
                  force_active_fraction=float(np.mean(record["force_constraint_active"])) if active_tick else 0.,
                  aperture_active_fraction=float(np.mean(record["aperture_active"])) if active_tick else 0.,
                  controller_p50_ms=float(np.median(times)) if times.size else None,
                  controller_p99_ms=float(np.quantile(times, .99)) if times.size else None,
                  controller_max_ms=float(times.max()) if times.size else None,
                  simulated_deadline_overruns=int(np.sum(times > 5.)),
                  wall_s=time.perf_counter()-begin_wall, **metrics, **widths)
    return result, {k: np.asarray(v) for k, v in record.items()}


def run_pair(args):
    scenario, seed, destination, save_all = args
    destination = Path(destination)/scenario/str(seed)
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    trace = None
    for variant in ("full", *(v for v in VARIANTS if v != "full")):
        try:
            if variant.startswith("matched") and trace is None:
                raise RuntimeError("matching controller trace unavailable")
            result, other_trace = run_case(scenario, seed, variant, trace if variant.startswith("matched") else None)
            if variant == "full":
                trace = other_trace
            if save_all or variant == "full":
                np.savez_compressed(destination/f"{variant}.npz", **other_trace)
        except Exception as exc:
            # Unexpected errors are explicit failed experimental units. They
            # cannot make either the completeness or effectiveness test pass.
            result = dict(scenario=scenario, seed=int(seed), variant=variant,
                          status="aborted", valid=False, reason="experiment_exception: "+repr(exc))
            (destination/f"failure_{variant}.txt").write_text(repr(exc))
        results.append(result)
        (destination/"metrics.json").write_text(json.dumps(plain(results), indent=2, allow_nan=True))
    (destination/"metrics.json").write_text(json.dumps(plain(results), indent=2, allow_nan=True))
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=("smoke", "design", "acceptance"), default="smoke")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--freeze-only", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--save-all-traces", action="store_true")
    args = p.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = frozen_manifest()
    if args.freeze_only:
        if args.manifest is None or args.manifest.exists():
            p.error("freeze requires a new --manifest path")
        args.manifest.write_text(json.dumps(manifest, indent=2, allow_nan=False))
        return
    acceptance = json.loads(ACCEPTANCE_PATH.read_text())
    if args.split == "acceptance":
        if args.manifest is None or json.loads(args.manifest.read_text()) != manifest:
            p.error("acceptance requires an unchanged frozen source/configuration manifest")
        seeds, scenarios = acceptance["acceptance_seeds"], SCENARIOS
    elif args.split == "design":
        seeds, scenarios = acceptance["design_seeds"], SCENARIOS
    else:
        seeds, scenarios = [0], ("healthy", "left_gap", "shadow")
    if (args.output/"manifest.json").exists():
        p.error("experiment output already exists; use a new versioned directory")
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    save_source_snapshot(args.output, manifest)
    jobs = [(s, seed, str(args.output), args.save_all_traces) for s in scenarios for seed in seeds]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run_pair, job): job for job in jobs}
        for i, future in enumerate(as_completed(pending), 1):
            job = pending[future]
            try:
                results.extend(future.result())
            except Exception as exc:
                (args.output/f"failure_{job[0]}_{job[1]}.txt").write_text(repr(exc))
                print(f"FAILED {job[0]} seed {job[1]}: {exc}", flush=True)
            print(f"{i}/{len(jobs)} pairs completed", flush=True)
    (args.output/"metrics.json").write_text(json.dumps(plain(results), indent=2, allow_nan=True))
    if args.split != "smoke":
        report = paired_report(results, acceptance, require_acceptance_seeds=args.split == "acceptance")
        report["split"] = args.split
        report["effectiveness_claim_permitted"] = args.split == "acceptance" and report["accepted"]
        (args.output/"acceptance.json").write_text(json.dumps(plain(report), indent=2, allow_nan=False))
    print(f"Saved {len(results)} runs to {args.output}", flush=True)


if __name__ == "__main__":
    main()
