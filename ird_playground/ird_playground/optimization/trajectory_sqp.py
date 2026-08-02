"""Fail-closed local SQP for rail + seven-joint scan trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
from scipy.spatial.transform import Rotation


NQ = 8
TASK_DIM = 6


class Kinematics8Dof(Protocol):
    q_lower: np.ndarray
    q_upper: np.ndarray
    v_max: np.ndarray

    def fk_matrix(self, q: np.ndarray) -> np.ndarray: ...
    def jacobian(self, q: np.ndarray) -> np.ndarray: ...
    def collision_rows(
        self, q: np.ndarray, *, d_activate: float, max_pairs: int
    ) -> list[tuple[float, np.ndarray, str]]: ...


@dataclass(frozen=True)
class WorldConstraint:
    """World-frame signed margin; non-negative means feasible."""

    name: str
    value: Callable[[np.ndarray, np.ndarray], np.ndarray | float]
    jacobian: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
    allowlisted_contact: bool = False


@dataclass(frozen=True)
class TrajectoryOptimizationConfig:
    max_iterations: int = 40
    trust_q: float = 0.12
    trust_rail_m: float = 0.02
    trust_task_position_m: float = 0.005
    trust_task_rotation_rad: float = 0.08
    task_offset_weight: float = 30.0
    q_step_weight: float = 1.0
    first_difference_weight: float = 1.0
    second_difference_weight: float = 4.0
    collision_safe_m: float = 0.01
    collision_activate_m: float = 0.04
    collision_max_pairs: int = 8
    finite_difference_eps: float = 1.0e-6
    kkt_tolerance: float = 1.0e-3
    fk_position_tolerance_m: float = 5.0e-4
    fk_rotation_tolerance_rad: float = np.deg2rad(1.0)
    segment_rail_step_m: float = 0.002
    segment_joint_step_rad: float = np.deg2rad(1.0)
    acceleration_limit: float | np.ndarray = 20.0
    jerk_limit: float | np.ndarray = 100.0


@dataclass(frozen=True)
class TrajectoryOptimizationProblem:
    s: np.ndarray
    T_tcp_nominal: np.ndarray
    q_seeds: tuple[np.ndarray, ...]
    position_tolerance_m: float | np.ndarray = 0.003
    rotation_tolerance_rad: float | np.ndarray = np.deg2rad(5.0)
    world_constraints: tuple[WorldConstraint, ...] = ()
    contact_allowlist: frozenset[str] = frozenset()
    contact_normals: np.ndarray | None = None
    # Data-driven priority: larger where the IRD margin changes sharply or
    # the nominal path is unreachable.  It relaxes the nominal pose penalty
    # there so the explicit projection stage can move to the nearest feasible
    # TCP without introducing another hand-tuned objective weight.
    reachability_priority: np.ndarray | None = None
    scan_speed_m_s: float = 0.02

    def validate(self) -> None:
        s = np.asarray(self.s, dtype=float)
        tcp = np.asarray(self.T_tcp_nominal, dtype=float)
        if s.ndim != 1 or len(s) < 2 or not np.all(np.diff(s) > 0.0):
            raise ValueError("s must be a strictly increasing 1-D path grid")
        if tcp.shape != (len(s), 4, 4):
            raise ValueError("T_tcp_nominal must have shape (W, 4, 4)")
        if not self.q_seeds:
            raise ValueError("at least one complete 8-DOF seed trajectory is required")
        for seed in self.q_seeds:
            if np.asarray(seed).shape != (len(s), NQ) or not np.isfinite(seed).all():
                raise ValueError("every q seed must be finite with shape (W, 8)")
        if self.reachability_priority is not None:
            priority = np.asarray(self.reachability_priority, dtype=float)
            if priority.shape != (len(s),) or not np.isfinite(priority).all() or np.any(priority < 0.0):
                raise ValueError("reachability_priority must be finite, non-negative and shape (W,)")
        if not 0.0 < float(self.scan_speed_m_s) <= 0.02 + 1.0e-12:
            raise ValueError("scan_speed_m_s must lie in (0, 0.02]")


@dataclass
class TrajectoryOptimizationResult:
    valid: bool
    s: np.ndarray
    timestamps: np.ndarray
    T_tcp_ref: np.ndarray
    cartesian_velocity_ff: np.ndarray
    q_ref: np.ndarray
    qdot_ff: np.ndarray
    rail_ref: np.ndarray
    contact_normals: np.ndarray
    task_offset: np.ndarray
    objective: float
    kkt_residual: float
    validation: dict[str, float | bool | str]
    seed_index: int = -1


def generate_ird_rail_warm_starts(
    field,
    trajectory_operator,
    problem: TrajectoryOptimizationProblem,
    base_seed: np.ndarray,
    rail_candidates_m: np.ndarray,
    T_world_rail: np.ndarray,
    T_rail_axis0: np.ndarray,
    *,
    top_k: int = 8,
    device: str = "cpu",
) -> tuple[np.ndarray, ...]:
    """Rank constant-rail task-space starts with the differentiable IRD operator."""
    import torch

    from ird_playground.region.operator import base_from_rail_torch

    rails = torch.as_tensor(rail_candidates_m, dtype=torch.float32, device=device)
    tcp = torch.as_tensor(problem.T_tcp_nominal, dtype=torch.float32, device=device)
    world_rail = torch.as_tensor(T_world_rail, dtype=torch.float32, device=device)
    rail_axis0 = torch.as_tensor(T_rail_axis0, dtype=torch.float32, device=device)
    scores = []
    with torch.no_grad():
        for rail in rails:
            axis = base_from_rail_torch(rail, world_rail, rail_axis0)
            result = trajectory_operator(field, tcp, axis)
            scores.append(float(result.trajectory_clearance.detach().cpu()))
    order = np.argsort(np.asarray(scores))[::-1][: max(1, int(top_k))]
    starts = []
    for index in order:
        seed = np.asarray(base_seed, dtype=float).copy()
        seed[:, 0] = float(np.asarray(rail_candidates_m)[index])
        starts.append(seed)
    return tuple(starts)


def _pose_error(T_target: np.ndarray, T_actual: np.ndarray) -> np.ndarray:
    error = np.empty(TASK_DIM, dtype=float)
    error[:3] = T_actual[:3, 3] - T_target[:3, 3]
    error[3:] = Rotation.from_matrix(
        T_target[:3, :3].T @ T_actual[:3, :3]
    ).as_rotvec()
    return error


def _offset_pose(T_nominal: np.ndarray, offset: np.ndarray) -> np.ndarray:
    target = np.asarray(T_nominal, dtype=float).copy()
    target[:3, 3] += offset[:3]
    target[:3, :3] = Rotation.from_rotvec(offset[3:]).as_matrix() @ target[:3, :3]
    return target


def _difference_matrix(n_waypoints: int, order: int) -> np.ndarray:
    base = np.eye(n_waypoints)
    return np.diff(base, n=order, axis=0)


def _world_value_and_jacobian(
    constraint: WorldConstraint,
    kin: Kinematics8Dof,
    q: np.ndarray,
    T: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    value = np.atleast_1d(np.asarray(constraint.value(q, T), dtype=float))
    if constraint.jacobian is not None:
        jac = np.atleast_2d(np.asarray(constraint.jacobian(q, T), dtype=float))
    else:
        jac = np.empty((len(value), NQ), dtype=float)
        for j in range(NQ):
            qp, qm = q.copy(), q.copy()
            qp[j] += eps
            qm[j] -= eps
            vp = np.atleast_1d(constraint.value(qp, kin.fk_matrix(qp)))
            vm = np.atleast_1d(constraint.value(qm, kin.fk_matrix(qm)))
            jac[:, j] = (vp - vm) / (2.0 * eps)
    if jac.shape != (len(value), NQ):
        raise ValueError(f"world constraint {constraint.name!r} returned invalid Jacobian")
    return value, jac


class Pinocchio8DofAdapter:
    """Adapter using the installed RM75 Pinocchio and HPP-FCL models."""

    def __init__(self, kinematics=None, collision=None) -> None:
        if kinematics is None:
            from rm75_control.control.joint_admittance_8dof.model import RobotKinematics

            kinematics = RobotKinematics()
        self.kin = kinematics
        if collision is None:
            from rm75_control.control.joint_admittance_8dof.collision_model import CollisionModel

            collision = CollisionModel(self.kin.model)
        self.collision = collision
        self.q_lower = np.asarray(self.kin.q_lower, dtype=float)
        self.q_upper = np.asarray(self.kin.q_upper, dtype=float)
        self.v_max = np.asarray(self.kin.v_max, dtype=float)

    def fk_matrix(self, q: np.ndarray) -> np.ndarray:
        placement = self.kin.fk_placement(q)
        T = np.eye(4)
        T[:3, :3] = placement.rotation
        T[:3, 3] = placement.translation
        return T

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        return np.asarray(self.kin.jacobian(q), dtype=float)

    def collision_rows(
        self, q: np.ndarray, *, d_activate: float, max_pairs: int
    ) -> list[tuple[float, np.ndarray, str]]:
        import pinocchio as pin

        self.collision.update(q)
        pairs = self.collision.active_pairs(d_activate)[:max_pairs]
        if not pairs:
            return []
        data = self.collision._kin_data  # noqa: SLF001
        pin.computeJointJacobians(self.collision.model, data, q)
        pin.updateFramePlacements(self.collision.model, data)
        rows = []
        for pair in pairs:
            point_jacs = []
            for geom_index, point in (
                (pair.geom_a, pair.point_a),
                (pair.geom_b, pair.point_b),
            ):
                geom = self.collision.geom_model.geometryObjects[geom_index]
                frame_id = int(geom.parentFrame)
                J = np.asarray(
                    pin.getFrameJacobian(
                        self.collision.model,
                        data,
                        frame_id,
                        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                    ),
                    dtype=float,
                )
                origin = np.asarray(data.oMf[frame_id].translation, dtype=float)
                r = np.asarray(point, dtype=float) - origin
                skew = np.array(
                    [[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]]
                )
                point_jacs.append(J[:3] - skew @ J[3:])
            gradient = pair.normal @ (point_jacs[0] - point_jacs[1])
            rows.append((pair.distance, gradient, f"{pair.name_a}:{pair.name_b}"))
        return rows


def _build_qp(
    problem: TrajectoryOptimizationProblem,
    cfg: TrajectoryOptimizationConfig,
    kin: Kinematics8Dof,
    q: np.ndarray,
    z: np.ndarray,
):
    import proxsuite

    w = len(q)
    nq_vars = w * NQ
    n = nq_vars + w * TASK_DIM
    H = np.eye(n) * cfg.q_step_weight
    g = np.zeros(n)
    z_slice = slice(nq_vars, n)
    priority = (
        np.zeros(w, dtype=float)
        if problem.reachability_priority is None
        else np.asarray(problem.reachability_priority, dtype=float)
    )
    # High priority means a reachability cliff: reduce the nominal tracking
    # penalty there and let the hard FK/collision constraints choose projection.
    z_weights = cfg.task_offset_weight / (1.0 + priority)
    for i, weight in enumerate(z_weights):
        sl = slice(i * TASK_DIM, (i + 1) * TASK_DIM)
        H[z_slice, z_slice][sl, sl] += np.eye(TASK_DIM) * weight
        g[z_slice][sl] += weight * z[i]

    q_flat = q.reshape(-1)
    for order, weight in (
        (1, cfg.first_difference_weight),
        (2, cfg.second_difference_weight),
    ):
        D = np.kron(_difference_matrix(w, order), np.eye(NQ))
        H[:nq_vars, :nq_vars] += weight * (D.T @ D)
        g[:nq_vars] += weight * (D.T @ (D @ q_flat))

    A = np.zeros((w * TASK_DIM, n))
    b = np.zeros(w * TASK_DIM)
    fk = []
    for i in range(w):
        T_actual = kin.fk_matrix(q[i])
        T_target = _offset_pose(problem.T_tcp_nominal[i], z[i])
        fk.append(T_actual)
        rows = slice(i * TASK_DIM, (i + 1) * TASK_DIM)
        A[rows, i * NQ : (i + 1) * NQ] = kin.jacobian(q[i])
        A[rows, nq_vars + i * TASK_DIM : nq_vars + (i + 1) * TASK_DIM] = -np.eye(TASK_DIM)
        b[rows] = -_pose_error(T_target, T_actual)

    c_rows, lower, upper = [], [], []
    for i, T_actual in enumerate(fk):
        for distance, gradient, _ in kin.collision_rows(
            q[i], d_activate=cfg.collision_activate_m, max_pairs=cfg.collision_max_pairs
        ):
            row = np.zeros(n)
            row[i * NQ : (i + 1) * NQ] = gradient
            c_rows.append(row)
            lower.append(cfg.collision_safe_m - distance)
            upper.append(np.inf)
        for constraint in problem.world_constraints:
            if constraint.allowlisted_contact or constraint.name in problem.contact_allowlist:
                continue
            values, jac = _world_value_and_jacobian(
                constraint, kin, q[i], T_actual, cfg.finite_difference_eps
            )
            for value, jac_row in zip(values, jac, strict=True):
                row = np.zeros(n)
                row[i * NQ : (i + 1) * NQ] = jac_row
                c_rows.append(row)
                lower.append(-value)
                upper.append(np.inf)
    C = np.asarray(c_rows, dtype=float).reshape(-1, n)
    l = np.asarray(lower, dtype=float)
    u = np.asarray(upper, dtype=float)

    pos_tol = np.broadcast_to(np.asarray(problem.position_tolerance_m, dtype=float), (3,))
    rot_tol = np.broadcast_to(np.asarray(problem.rotation_tolerance_rad, dtype=float), (3,))
    z_tol = np.tile(np.concatenate((pos_tol, rot_tol)), w)
    trust_z = np.tile(
        np.array([cfg.trust_task_position_m] * 3 + [cfg.trust_task_rotation_rad] * 3), w
    )
    trust_q = np.tile(np.array([cfg.trust_rail_m] + [cfg.trust_q] * 7), w)
    lower_q = np.tile(np.asarray(kin.q_lower), w) - q_flat
    upper_q = np.tile(np.asarray(kin.q_upper), w) - q_flat
    lower_z = -z_tol - z.reshape(-1)
    upper_z = z_tol - z.reshape(-1)
    l_box = np.concatenate((np.maximum(lower_q, -trust_q), np.maximum(lower_z, -trust_z)))
    u_box = np.concatenate((np.minimum(upper_q, trust_q), np.minimum(upper_z, trust_z)))

    solver = proxsuite.proxqp.dense.QP(n, len(b), len(l), box_constraints=True)
    solver.settings.eps_abs = 1.0e-7
    solver.settings.max_iter = 500
    solver.init(H, g, A, b, C, l, u, l_box=l_box, u_box=u_box)
    solver.solve()
    return np.asarray(solver.results.x), solver.results.info


def _objective(q: np.ndarray, z: np.ndarray, cfg: TrajectoryOptimizationConfig) -> float:
    value = cfg.task_offset_weight * float(np.sum(z * z))
    value += cfg.first_difference_weight * float(np.sum(np.diff(q, axis=0) ** 2))
    if len(q) > 2:
        value += cfg.second_difference_weight * float(np.sum(np.diff(q, n=2, axis=0) ** 2))
    return value


def retime_trajectory(
    q: np.ndarray,
    tcp: np.ndarray,
    v_max: np.ndarray,
    *,
    scan_speed_m_s: float = 0.02,
    acceleration_limit: float | np.ndarray = 20.0,
    jerk_limit: float | np.ndarray = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retime each segment with Ruckig, never faster than the TCP speed cap."""
    import ruckig

    q = np.asarray(q, dtype=float)
    positions = np.asarray(tcp, dtype=float)[:, :3, 3]
    arc_dt = np.linalg.norm(np.diff(positions, axis=0), axis=1) / float(scan_speed_m_s)
    v_lim = np.asarray(v_max, dtype=float)
    a_lim = np.broadcast_to(np.asarray(acceleration_limit, dtype=float), (NQ,))
    j_lim = np.broadcast_to(np.asarray(jerk_limit, dtype=float), (NQ,))
    segment_dt = np.maximum(arc_dt, 1.0e-3)
    for _ in range(2):
        timestamps = np.concatenate(([0.0], np.cumsum(segment_dt)))
        qdot = np.gradient(q, timestamps, axis=0, edge_order=1)
        qdot = np.clip(qdot, -v_lim, v_lim)
        for i in range(len(segment_dt)):
            inp = ruckig.InputParameter(NQ)
            inp.current_position = q[i].tolist()
            inp.current_velocity = qdot[i].tolist()
            inp.current_acceleration = [0.0] * NQ
            inp.target_position = q[i + 1].tolist()
            inp.target_velocity = qdot[i + 1].tolist()
            inp.target_acceleration = [0.0] * NQ
            inp.max_velocity = v_lim.tolist()
            inp.max_acceleration = a_lim.tolist()
            inp.max_jerk = j_lim.tolist()
            trajectory = ruckig.Trajectory(NQ)
            result = ruckig.Ruckig(NQ).calculate(inp, trajectory)
            if result.value < 0:
                raise RuntimeError(f"Ruckig failed on segment {i}: {result}")
            segment_dt[i] = max(segment_dt[i], float(trajectory.duration), arc_dt[i])
    timestamps = np.concatenate(([0.0], np.cumsum(segment_dt)))
    qdot = np.gradient(q, timestamps, axis=0, edge_order=1)
    qddot = np.gradient(qdot, timestamps, axis=0, edge_order=1)
    scale = max(
        1.0,
        float(np.max(np.abs(qdot) / np.maximum(v_lim, 1.0e-12))),
        float(np.sqrt(np.max(np.abs(qddot) / np.maximum(a_lim, 1.0e-12)))),
    )
    if scale > 1.0:
        segment_dt *= scale * (1.0 + 1.0e-9)
        timestamps = np.concatenate(([0.0], np.cumsum(segment_dt)))
        qdot = np.gradient(q, timestamps, axis=0, edge_order=1)
    cartesian = np.zeros((len(q), TASK_DIM), dtype=float)
    cartesian[:, :3] = np.gradient(positions, timestamps, axis=0, edge_order=1)
    for i in range(1, len(q) - 1):
        dt = timestamps[i + 1] - timestamps[i - 1]
        cartesian[i, 3:] = Rotation.from_matrix(
            tcp[i - 1, :3, :3].T @ tcp[i + 1, :3, :3]
        ).as_rotvec() / dt
    return timestamps, qdot, cartesian


def validate_trajectory(
    problem: TrajectoryOptimizationProblem,
    cfg: TrajectoryOptimizationConfig,
    kin: Kinematics8Dof,
    q: np.ndarray,
    z: np.ndarray,
    *,
    kkt_residual: float,
) -> dict[str, float | bool | str]:
    max_pos = max_rot = 0.0
    min_collision = float("inf")
    min_world = float("inf")
    limit_violation = float(
        max(np.max(kin.q_lower - q), np.max(q - kin.q_upper), 0.0)
    )
    for i in range(len(q)):
        error = _pose_error(_offset_pose(problem.T_tcp_nominal[i], z[i]), kin.fk_matrix(q[i]))
        max_pos = max(max_pos, float(np.linalg.norm(error[:3])))
        max_rot = max(max_rot, float(np.linalg.norm(error[3:])))

    for i in range(len(q) - 1):
        rail_steps = abs(q[i + 1, 0] - q[i, 0]) / cfg.segment_rail_step_m
        joint_steps = np.max(np.abs(q[i + 1, 1:] - q[i, 1:])) / cfg.segment_joint_step_rad
        n_steps = max(1, int(np.ceil(max(rail_steps, joint_steps))))
        for alpha in np.linspace(0.0, 1.0, n_steps + 1):
            qi = (1.0 - alpha) * q[i] + alpha * q[i + 1]
            Ti = kin.fk_matrix(qi)
            rows = kin.collision_rows(
                qi, d_activate=float("inf"), max_pairs=cfg.collision_max_pairs
            )
            if rows:
                min_collision = min(min_collision, min(row[0] for row in rows))
            for constraint in problem.world_constraints:
                values = np.atleast_1d(constraint.value(qi, Ti))
                if not constraint.allowlisted_contact and constraint.name not in problem.contact_allowlist:
                    min_world = min(min_world, float(np.min(values)))
    valid = bool(
        max_pos <= cfg.fk_position_tolerance_m
        and max_rot <= cfg.fk_rotation_tolerance_rad
        and limit_violation <= 0.0
        and min_collision >= cfg.collision_safe_m
        and min_world >= 0.0
        and kkt_residual <= cfg.kkt_tolerance
    )
    return {
        "valid": valid,
        "max_fk_position_m": max_pos,
        "max_fk_rotation_rad": max_rot,
        "joint_limit_violation": limit_violation,
        "min_collision_m": min_collision,
        "min_world_margin": min_world,
        "kkt_residual": float(kkt_residual),
        "failure": "" if valid else "one or more hard validators failed",
    }


def optimize_trajectory(
    problem: TrajectoryOptimizationProblem,
    *,
    config: TrajectoryOptimizationConfig | None = None,
    kinematics: Kinematics8Dof | None = None,
) -> TrajectoryOptimizationResult:
    """Return the lowest-cost fully validated local solution across all seeds."""
    problem.validate()
    cfg = config or TrajectoryOptimizationConfig()
    kin = kinematics or Pinocchio8DofAdapter()
    candidates = []
    for seed_index, seed in enumerate(problem.q_seeds):
        q = np.asarray(seed, dtype=float).copy()
        z = np.zeros((len(q), TASK_DIM), dtype=float)
        kkt = float("inf")
        try:
            for _ in range(cfg.max_iterations):
                step, info = _build_qp(problem, cfg, kin, q, z)
                dq = step[: q.size].reshape(q.shape)
                dz = step[q.size :].reshape(z.shape)
                q += dq
                z += dz
                kkt = max(float(info.pri_res), float(info.dua_res))
                if max(np.max(np.abs(dq)), np.max(np.abs(dz))) < 1.0e-7:
                    break
            validation = validate_trajectory(problem, cfg, kin, q, z, kkt_residual=kkt)
        except Exception as exc:
            validation = {"valid": False, "failure": str(exc), "kkt_residual": kkt}
        if validation.get("valid"):
            candidates.append((_objective(q, z, cfg), seed_index, q, z, validation, kkt))

    if not candidates:
        w = len(problem.s)
        normals = (
            np.asarray(problem.contact_normals, dtype=float)
            if problem.contact_normals is not None
            else np.asarray(problem.T_tcp_nominal)[:, :3, 2]
        )
        return TrajectoryOptimizationResult(
            valid=False,
            s=np.asarray(problem.s, dtype=float),
            timestamps=np.zeros(w),
            T_tcp_ref=np.asarray(problem.T_tcp_nominal, dtype=float).copy(),
            cartesian_velocity_ff=np.zeros((w, TASK_DIM)),
            q_ref=np.full((w, NQ), np.nan),
            qdot_ff=np.full((w, NQ), np.nan),
            rail_ref=np.full(w, np.nan),
            contact_normals=normals,
            task_offset=np.full((w, TASK_DIM), np.nan),
            objective=float("inf"),
            kkt_residual=float("inf"),
            validation={"valid": False, "failure": "no seed passed every hard validator"},
        )

    objective, seed_index, q, z, validation, kkt = min(candidates, key=lambda item: item[0])
    tcp = np.stack([kin.fk_matrix(qi) for qi in q])
    try:
        timestamps, qdot, cartesian = retime_trajectory(
            q,
            tcp,
            kin.v_max,
            scan_speed_m_s=problem.scan_speed_m_s,
            acceleration_limit=cfg.acceleration_limit,
            jerk_limit=cfg.jerk_limit,
        )
    except Exception as exc:
        validation = dict(validation)
        validation.update(valid=False, failure=f"retiming failed: {exc}")
        return TrajectoryOptimizationResult(
            valid=False,
            s=np.asarray(problem.s, dtype=float),
            timestamps=np.zeros(len(q)),
            T_tcp_ref=tcp,
            cartesian_velocity_ff=np.zeros((len(q), TASK_DIM)),
            q_ref=q,
            qdot_ff=np.full_like(q, np.nan),
            rail_ref=q[:, 0],
            contact_normals=tcp[:, :3, 2],
            task_offset=z,
            objective=objective,
            kkt_residual=kkt,
            validation=validation,
            seed_index=seed_index,
        )
    qddot = np.gradient(qdot, timestamps, axis=0, edge_order=1)
    vmax = np.asarray(kin.v_max, dtype=float)
    amax = np.broadcast_to(np.asarray(cfg.acceleration_limit, dtype=float), (NQ,))
    speed = np.linalg.norm(np.diff(tcp[:, :3, 3], axis=0), axis=1) / np.diff(timestamps)
    timing_valid = bool(
        np.all(np.abs(qdot) <= vmax + 1.0e-8)
        and np.all(np.abs(qddot) <= amax + 1.0e-6)
        and np.all(speed <= problem.scan_speed_m_s + 1.0e-9)
    )
    validation = dict(validation)
    validation.update(
        timing_valid=timing_valid,
        max_tcp_speed_m_s=float(speed.max(initial=0.0)),
        max_joint_velocity_ratio=float(np.max(np.abs(qdot) / np.maximum(vmax, 1.0e-12))),
        max_joint_acceleration_ratio=float(np.max(np.abs(qddot) / np.maximum(amax, 1.0e-12))),
    )
    if not timing_valid:
        validation.update(valid=False, failure="retimed path violates velocity or acceleration limits")
    normals = (
        np.asarray(problem.contact_normals, dtype=float)
        if problem.contact_normals is not None
        else tcp[:, :3, 2]
    )
    return TrajectoryOptimizationResult(
        valid=timing_valid,
        s=np.asarray(problem.s, dtype=float),
        timestamps=timestamps,
        T_tcp_ref=tcp,
        cartesian_velocity_ff=cartesian,
        q_ref=q,
        qdot_ff=qdot,
        rail_ref=q[:, 0],
        contact_normals=normals,
        task_offset=z,
        objective=objective,
        kkt_residual=kkt,
        validation=validation,
        seed_index=seed_index,
    )


__all__ = [
    "Pinocchio8DofAdapter",
    "TrajectoryOptimizationConfig",
    "TrajectoryOptimizationProblem",
    "TrajectoryOptimizationResult",
    "WorldConstraint",
    "generate_ird_rail_warm_starts",
    "optimize_trajectory",
    "retime_trajectory",
    "validate_trajectory",
]
