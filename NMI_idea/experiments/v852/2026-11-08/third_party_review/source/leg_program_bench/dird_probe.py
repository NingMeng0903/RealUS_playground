"""Offline, CPU-only audit of the existing RM75 dIRD field.

Inputs MUST be physical probe45 TCP poses, not the DMP image coordinate frame.
``base_world`` is T_world_railbase. A virtual placement is permitted and is
always labelled as such; this module never registers or commands a robot.
The installed checkpoint is old, so using it requires allow_stale_audit=True.
Its scores are model outputs, not collision certificates or execution costs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path("/media/camp/EXT_DRIVE/RealUS_playground")
CHECKPOINT = REPO / "ird_playground/data/checkpoints/rm4d_signed/selected.pt"
ROBOT_SPEC = REPO / "ird_playground/configs/robot_probe45.yaml"
PROJECT_PYTHON = Path("/media/camp/EXT_DRIVE/envs/genesis/bin/python")


def _prepare_imports() -> None:
    for path in (REPO / "ird_playground", REPO / "rm75_control"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _transforms(value, name: str, *, trajectory: bool = False) -> np.ndarray:
    out = np.asarray(value, dtype=float)
    if trajectory:
        if out.ndim != 3 or out.shape[1:] != (4, 4) or len(out) == 0:
            raise ValueError(f"{name} must be a nonempty (W,4,4) trajectory")
    elif out.shape != (4, 4):
        raise ValueError(f"{name} must be (4,4)")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains nonfinite values")
    if not np.allclose(out[..., 3, :], [0, 0, 0, 1], atol=1e-8):
        raise ValueError(f"{name} has invalid homogeneous bottom row")
    r = out[..., :3, :3]
    if (not np.allclose(np.swapaxes(r, -1, -2) @ r, np.eye(3), atol=1e-6)
            or not np.allclose(np.linalg.det(r), 1.0, atol=1e-6)):
        raise ValueError(f"{name} rotations must lie in SO(3); reflections are invalid")
    return out


def convert_image_poses(T_image_world, T_image_tcp) -> np.ndarray:
    """Explicit caller-supplied frame bridge; no guessed axis flip or offset.

    T_image_tcp expresses the physical TCP in the image frame. This function
    composes it but does not establish that it is experimentally calibrated.
    """
    return (_transforms(T_image_world, "T_image_world", trajectory=True)
            @ _transforms(T_image_tcp, "T_image_tcp"))


def dependency_audit() -> dict:
    """Test imports and actual model construction, not only module names."""
    _prepare_imports()
    report = {}
    for name in ("torch", "pinocchio", "hppfcl", "coal", "proxsuite", "ruckig"):
        try:
            module = importlib.import_module(name)
            report[name] = {"available": True, "version": str(getattr(module, "__version__", "unknown"))}
        except Exception as exc:
            report[name] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    try:
        from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
        from rm75_control.control.joint_admittance_8dof.collision_model import CollisionModel
        kin = RobotKinematics()
        collision = CollisionModel(kin.model)
        report["robot_model_construction"] = {
            "available": True, "nq": kin.nq,
            "collision_pairs": len(collision.geom_model.collisionPairs),
        }
    except Exception as exc:
        report["robot_model_construction"] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    report["sqp_available"] = report["proxsuite"]["available"]
    return report


class DirdProbe:
    """Actual frozen field scoring. Failures never produce fabricated scores."""

    def __init__(self, checkpoint=CHECKPOINT, robot_spec=ROBOT_SPEC, *,
                 allow_stale_audit: bool = False, placement_label: str = "virtual_unregistered"):
        _prepare_imports()
        import torch
        from ird_playground.ird.robot_model import load_robot_model_spec
        from ird_playground.neural.signed_field import ReachabilitySDF
        self.torch = torch
        self.checkpoint = Path(checkpoint).resolve()
        self.spec = load_robot_model_spec(robot_spec)
        self.field = ReachabilitySDF.load(
            self.checkpoint, device="cpu", expected_robot=self.spec,
            allow_stale=bool(allow_stale_audit),
        )
        self.current_contract = self.spec.to_manifest()
        self.recorded_contract = (self.field.meta.get("robot_contract")
                                  or (self.field.meta.get("dataset_meta") or {}).get("robot_contract") or {})
        self.contract_differences = {
            key: {"current": value, "checkpoint": self.recorded_contract.get(key)}
            for key, value in self.current_contract.items()
            if self.recorded_contract.get(key) != value
        }
        self.allow_stale_audit = bool(allow_stale_audit)
        self.placement_label = str(placement_label)
        self.rail_limits = self.spec.rail_limits_m()
        self.axis_zero = self.spec.root_to_j1_axis().copy()
        # root_to_j1_axis includes the spec's locked rail; remove it before
        # base_from_rail_torch applies the raw URDF coordinate a second time.
        self.axis_zero[1, 3] -= float(self.spec.rail_locked_at_m)
        self._kin = None

    def metadata(self) -> dict:
        return {
            "status": "score_only", "physical_execution": False,
            "ready_for_execution": False, "device": "cpu",
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
            "allow_stale_audit": self.allow_stale_audit,
            "checkpoint_artifact_schema": self.field.meta.get("artifact_schema"),
            "checkpoint_calibration_schema": self.field.meta.get("calibration_schema"),
            "robot_contract_matches": not bool(self.contract_differences),
            "robot_contract_differences": self.contract_differences,
            "placement_label": self.placement_label,
            "physical_registration_verified": False,
            "input_tcp_frame": "physical_probe45_tcp",
            "base_world_convention": "T_world_railbase",
            "rail_convention": "raw_URDF_rail_y_m",
            "rail_limits_m": list(self.rail_limits),
            "T_railbase_j1_axis_at_zero": self.axis_zero.tolist(),
            "T_flange_tcp_current_URDF": self.current_contract["T_flange_tcp"],
            "T_flange_tcp_used_by_field": self.field.model.T_flange_tcp.detach().cpu().numpy().tolist(),
            "score_units": "uncalibrated_checkpoint_output_not_metres",
            "score_is_success_probability": False,
            "limitations": ["No environment collision certificate", "No actual-entry transition plan",
                            "No contact or image acceptance", "No measured execution duration"],
        }

    def rank_modes(self, T_tcp_world, base_world, rails) -> dict:
        """Rank scalar rail candidates or (K,W) rail schedules by min field score.

        ``rails`` shaped (K,) means K CONSTANT rail candidates, including when
        K equals W. Use (1,W) for one varying schedule. Poses/orientations remain
        identical across modes. Invalid rails are rejected, never clipped.
        """
        from ird_playground.region.operator import base_from_rail_torch
        from ird_playground.ird.canonical import canonical_flange_from_world_torch
        tcp = _transforms(T_tcp_world, "T_tcp_world", trajectory=True)
        base = _transforms(base_world, "base_world")
        r = np.asarray(rails, dtype=float)
        if r.ndim == 0:
            r = r.reshape(1)
        if r.ndim == 1:
            r = np.broadcast_to(r[:, None], (len(r), len(tcp))).copy()
        if r.ndim != 2 or r.shape[1] != len(tcp) or len(r) == 0:
            raise ValueError("rails must be nonempty (K,) constants or (K,W) schedules")
        modes, rejected = [], []
        torch = self.torch
        tcp_t = torch.as_tensor(tcp, dtype=torch.float32)
        base_t = torch.as_tensor(base, dtype=torch.float32)
        axis_t = torch.as_tensor(self.axis_zero, dtype=torch.float32)
        for index, path in enumerate(r):
            mode = {"mode_id": f"rail_{index}", "input_index": index,
                    "rail_m": [float(x) if np.isfinite(x) else None for x in path]}
            if not np.isfinite(path).all() or np.any(path < self.rail_limits[0]) or np.any(path > self.rail_limits[1]):
                mode.update(status="rejected", reason="raw rail outside URDF bounds or nonfinite")
                rejected.append(mode)
                continue
            with torch.no_grad():
                axis = base_from_rail_torch(torch.as_tensor(path, dtype=torch.float32), base_t, axis_t, axis=1)
                scores = self.field.score_world(tcp_t, axis).detach().cpu().numpy()
                canonical = canonical_flange_from_world_torch(tcp_t, axis, self.field.model.T_flange_tcp)
                support = self.field.model.support_distance(canonical).detach().cpu().numpy()
            if not np.isfinite(scores).all():
                raise RuntimeError(f"dIRD produced nonfinite scores for {mode['mode_id']}")
            mode.update(status="score_only", min_score=float(scores.min()), mean_score=float(scores.mean()),
                        waypoint_scores=scores.tolist(), worst_waypoint_index=int(np.argmin(scores)),
                        max_training_support_distance=float(support.max()),
                        all_waypoints_positive=bool(np.all(scores > 0)),
                        validated_reachable=False, ready_for_execution=False)
            modes.append(mode)
        modes.sort(key=lambda m: (-m["min_score"], -m["mean_score"], m["input_index"]))
        for rank, mode in enumerate(modes, 1):
            mode["rank"] = rank
        return {"metadata": self.metadata(), "ranking_rule": "descending minimum guarded field score, then mean",
                "base_world": base.tolist(), "waypoint_count": len(tcp), "modes": modes, "rejected": rejected}

    def kinematics(self):
        if self._kin is None:
            from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
            self._kin = RobotKinematics(urdf_path=self.spec.kinematics_urdf, tcp_frame=self.spec.tcp_frame)
        return self._kin

    def audit_ik(self, T_tcp_world, base_world, rail, q_seed, *, scan_speed_m_s: float = 0.02) -> dict:
        """Optional SRS+FK+dense self-collision audit, with no QP or hardware.

        Does not solve a path from q_seed to q_ref[0]. Retiming is only a nominal
        model prediction and the robot/world/contact checks remain incomplete.
        """
        from scipy.spatial.transform import Rotation
        tcp_w = _transforms(T_tcp_world, "T_tcp_world", trajectory=True)
        base = _transforms(base_world, "base_world")
        if len(tcp_w) < 3:
            raise ValueError("IK/retiming audit needs at least three waypoints")
        rails = np.broadcast_to(np.asarray(rail, dtype=float), (len(tcp_w),)).copy()
        if not np.isfinite(rails).all() or np.any(rails < self.rail_limits[0]) or np.any(rails > self.rail_limits[1]):
            raise ValueError("rail outside raw URDF bounds")
        seed = np.asarray(q_seed, dtype=float)
        if seed.shape != (8,) or not np.isfinite(seed).all():
            raise ValueError("q_seed must be a finite 8-vector")
        if not 0 < scan_speed_m_s <= 0.02:
            raise ValueError("scan_speed_m_s must lie in (0,0.02]")
        result = {"status": "score_only", "physical_execution": False, "ready_for_execution": False,
                  "world_collision_checked": False, "transition_from_entry_checked": False,
                  "contact_checked": False, "retiming_is_measured_duration": False,
                  "retiming_passed": False,
                  "q_seed": seed.tolist(), "q_seed_source": "caller_supplied_actual_or_virtual_entry"}
        try:
            from ird_playground.optimization.srs_trajectory_dp import solve_srs_trajectory_dp
            from ird_playground.optimization.trajectory_sqp import retime_trajectory
            from rm75_control.control.joint_admittance_8dof.collision_model import CollisionModel
            kin = self.kinematics()
            if np.any(seed < kin.q_lower) or np.any(seed > kin.q_upper):
                raise ValueError("q_seed outside current URDF joint limits")
            tcp = np.linalg.inv(base)[None, :, :] @ tcp_w
            lift = solve_srs_trajectory_dp(tcp, rails, seed, kin=kin)
            result.update(lift_valid=bool(lift.lift_valid), branch=int(lift.branch), failure=lift.failure)
            if not lift.lift_valid:
                return result
            q = np.asarray(lift.q_ref, dtype=float)
            actual = []
            for qi in q:
                pose = kin.fk_placement(qi)
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = pose.rotation, pose.translation
                actual.append(T)
            actual = np.asarray(actual)
            p_error = np.linalg.norm(tcp[:, :3, 3] - actual[:, :3, 3], axis=1)
            r_error = Rotation.from_matrix(np.swapaxes(tcp[:, :3, :3], 1, 2) @ actual[:, :3, :3]).magnitude()
            limit_ok = bool(np.all(q >= kin.q_lower - 1e-8) and np.all(q <= kin.q_upper + 1e-8))
            collision = CollisionModel(kin.model, collision_urdf=self.spec.collision_urdf,
                                       pair_config=self.spec.collision_pairs)
            minimum, samples = float("inf"), 0
            for a, b in zip(q[:-1], q[1:]):
                steps = max(1, int(np.ceil(max(abs(b[0] - a[0]) / .002,
                                             np.max(abs(b[1:] - a[1:])) / np.deg2rad(1)))))
                for alpha in np.linspace(0, 1, steps + 1):
                    collision.update((1-alpha)*a + alpha*b)
                    minimum = min(minimum, collision.min_distance())
                    samples += 1
            if not np.isfinite(minimum):
                raise RuntimeError("no finite exact self-collision distances available")
            audit_ok = bool(limit_ok and p_error.max() <= .0005 and r_error.max() <= np.deg2rad(1)
                            and minimum >= self.spec.collision_security_margin_m)
            result.update(status="kinematic_audit_only", kinematic_audit_passed=audit_ok,
                          q_ref=q.tolist(), max_fk_position_error_m=float(p_error.max()),
                          max_fk_rotation_error_rad=float(r_error.max()), joint_limits_passed=limit_ok,
                          minimum_self_collision_distance_m=float(minimum), collision_samples=samples,
                          self_collision_scope="configured_arm_body_pairs_only",
                          entry_to_first_q_delta=(q[0]-seed).tolist())
            if audit_ok:
                t, qdot, _ = retime_trajectory(q, actual, kin.v_max, scan_speed_m_s=scan_speed_m_s)
                result.update(nominal_timestamps_s=t.tolist(), nominal_duration_s=float(t[-1]),
                              nominal_qdot_ff=qdot.tolist(), retiming_passed=True)
            else:
                result["failure"] = "FK, joint-limit, or sampled arm self-collision audit failed"
        except Exception as exc:
            result.update(failure=f"{type(exc).__name__}: {exc}")
        return result

    def audit_ik_paths(self, T_tcp_world, base_world, rails, q_seed, *, top_k: int = 2,
                       scan_speed_m_s: float = 0.02) -> dict:
        """Rank supplied visible-history paths, then independently audit modes.

        The caller supplies the entry q_seed, including its raw rail coordinate.
        No initial joint pose is inferred from hidden vessel geometry. Each mode
        uses that same seed branch, and an entry mismatch is reported, not solved.
        """
        if top_k < 1:
            raise ValueError("top_k must be positive")
        ranking = self.rank_modes(T_tcp_world, base_world, rails)
        audits = []
        for mode in ranking["modes"][:top_k]:
            audits.append({"mode_id": mode["mode_id"], "input_index": mode["input_index"],
                           "rail_m": mode["rail_m"], "dird_min_score": mode["min_score"],
                           "audit": self.audit_ik(T_tcp_world, base_world, mode["rail_m"], q_seed,
                                                  scan_speed_m_s=scan_speed_m_s)})
        return {"ranking": ranking, "mode_audits": audits, "ready_for_execution": False,
                "q_seed_policy": "same caller-supplied entry and SRS branch for every mode"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="NPZ with T_tcp_world, base_world, rails")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-stale-audit", action="store_true")
    args = parser.parse_args()
    probe = DirdProbe(allow_stale_audit=args.allow_stale_audit)
    report = {"metadata": probe.metadata(), "dependencies": dependency_audit()}
    if args.input:
        with np.load(args.input, allow_pickle=False) as inputs:
            report["ranking"] = probe.rank_modes(inputs["T_tcp_world"], inputs["base_world"], inputs["rails"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "status": "score_only",
                      "contract_matches": report["metadata"]["robot_contract_matches"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
