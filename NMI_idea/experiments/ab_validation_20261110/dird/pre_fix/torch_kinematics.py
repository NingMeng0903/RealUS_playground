"""Batched differentiable FK and DLS IK for the locked-rail RM75 7R chain."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


def _skew_z(q: "torch.Tensor") -> "torch.Tensor":
    c, s = torch.cos(q), torch.sin(q)
    z, o = torch.zeros_like(q), torch.ones_like(q)
    return torch.stack(
        [c, -s, z, s, c, z, z, z, o], dim=-1
    ).reshape(*q.shape, 3, 3)


def so3_log(R: "torch.Tensor") -> "torch.Tensor":
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)
    vee = torch.stack(
        [R[..., 2, 1] - R[..., 1, 2],
         R[..., 0, 2] - R[..., 2, 0],
         R[..., 1, 0] - R[..., 0, 1]],
        dim=-1,
    )
    sin_theta = torch.sin(theta)
    factor = theta / (2.0 * sin_theta.clamp_min(1e-7))
    factor = torch.where(theta < 1e-4, 0.5 + theta * theta / 12.0, factor)
    return factor.unsqueeze(-1) * vee


def so3_exp(rotvec: "torch.Tensor") -> "torch.Tensor":
    """Batch Rodrigues exponential for world-frame rotation vectors."""
    theta = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
    axis = rotvec / theta.clamp_min(1e-9)
    x, y, z = axis.unbind(dim=-1)
    zero = torch.zeros_like(x)
    K = torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero], dim=-1
    ).reshape(*rotvec.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device).expand(
        *rotvec.shape[:-1], 3, 3
    )
    theta_m = theta.unsqueeze(-1)
    R = eye + torch.sin(theta_m) * K + (1.0 - torch.cos(theta_m)) * (K @ K)
    return torch.where((theta < 1e-8).unsqueeze(-1), eye, R)


@dataclass
class BatchIkResult:
    q: "torch.Tensor"
    ok: "torch.Tensor"
    pos_error_m: "torch.Tensor"
    rot_error_rad: "torch.Tensor"
    iterations: int


@dataclass
class CollisionCheckedIkResult:
    """One selected, collision-free IK candidate per target."""

    q: "torch.Tensor"
    reachable: "torch.Tensor"
    seed_index: "torch.Tensor"
    candidate_collision_free: "torch.Tensor"
    pos_error_m: "torch.Tensor"
    rot_error_rad: "torch.Tensor"


def collision_free_mask(
    q: "torch.Tensor | np.ndarray",
    collision_filter,
    *,
    chunk_size: int = 8192,
    device: "torch.device | str | None" = None,
) -> "torch.Tensor":
    """Evaluate the probe-aware Pinocchio collision model for a q batch."""
    if torch is None:
        raise ImportError("torch is required")
    if torch.is_tensor(q):
        out_device = torch.device(device or q.device)
        q_np = q.detach().cpu().numpy()
    else:
        out_device = torch.device(device or "cpu")
        q_np = np.asarray(q)
    shape = q_np.shape[:-1]
    flat = np.asarray(q_np, dtype=np.float64).reshape(-1, 7)
    free = np.empty(flat.shape[0], dtype=bool)
    for start in range(0, flat.shape[0], int(chunk_size)):
        stop = min(flat.shape[0], start + int(chunk_size))
        free[start:stop] = collision_filter.free_mask(flat[start:stop])
    return torch.as_tensor(free.reshape(shape), dtype=torch.bool, device=out_device)


def select_collision_free_ik(
    result: BatchIkResult,
    collision_filter,
    *,
    tol_pos_m: float = 2.0e-4,
    tol_rot_rad: float = 1.0e-3,
    chunk_size: int = 8192,
) -> CollisionCheckedIkResult:
    """Select the best converged, collision-free seed for every target.

    The penultimate dimension is treated as the seed dimension. A target is
    reachable only when at least one candidate satisfies both IK tolerances and
    the configured robot+probe self-collision model.
    """
    if result.q.ndim < 2:
        raise ValueError("IK candidates must include a seed dimension")
    score = result.pos_error_m / max(float(tol_pos_m), 1e-12)
    score = score + result.rot_error_rad / max(float(tol_rot_rad), 1e-12)
    score = torch.where(result.ok, score, torch.full_like(score, float("inf")))
    order = score.argsort(dim=-1)
    target_shape = result.ok.shape[:-1]
    n_seed = result.ok.shape[-1]
    flat_q = result.q.reshape(-1, n_seed, 7)
    flat_ok = result.ok.reshape(-1, n_seed)
    flat_order = order.reshape(-1, n_seed)
    n_target = flat_q.shape[0]
    flat_reachable = torch.zeros(n_target, dtype=torch.bool, device=result.q.device)
    flat_seed_index = torch.zeros(n_target, dtype=torch.int64, device=result.q.device)
    candidate_free = torch.zeros_like(flat_ok, dtype=torch.bool)
    # Check the lowest-error candidate first. Only colliding targets proceed to
    # the next seed, which avoids collision-testing every converged branch.
    for rank in range(n_seed):
        unresolved = ~flat_reachable
        idx = flat_order[:, rank]
        candidate_ok = flat_ok.gather(1, idx[:, None]).squeeze(1)
        check = unresolved & candidate_ok
        if not bool(check.any()):
            continue
        rows = torch.nonzero(check, as_tuple=False).flatten()
        seeds = idx[rows]
        q_check = flat_q[rows, seeds]
        free = collision_free_mask(
            q_check,
            collision_filter,
            chunk_size=chunk_size,
            device=result.q.device,
        )
        candidate_free[rows, seeds] = free
        accepted_rows = rows[free]
        flat_reachable[accepted_rows] = True
        flat_seed_index[accepted_rows] = seeds[free]
    reachable = flat_reachable.reshape(target_shape)
    seed_index = flat_seed_index.reshape(target_shape)
    candidate_free = candidate_free.reshape(result.ok.shape)
    gather_q = seed_index[..., None, None].expand(*seed_index.shape, 1, 7)
    q_best = result.q.gather(-2, gather_q).squeeze(-2)
    pos_best = result.pos_error_m.gather(-1, seed_index[..., None]).squeeze(-1)
    rot_best = result.rot_error_rad.gather(-1, seed_index[..., None]).squeeze(-1)
    # Avoid presenting an arbitrary failed candidate as a usable IK solution.
    q_best = torch.where(reachable[..., None], q_best, torch.full_like(q_best, float("nan")))
    return CollisionCheckedIkResult(
        q=q_best,
        reachable=reachable,
        seed_index=seed_index,
        candidate_collision_free=candidate_free,
        pos_error_m=pos_best,
        rot_error_rad=rot_best,
    )


class TorchRM75Kinematics:
    """Torch kinematics built from Pinocchio's fixed joint placements."""

    def __init__(
        self,
        joint_R: np.ndarray,
        joint_t: np.ndarray,
        tcp_R: np.ndarray,
        tcp_t: np.ndarray,
        q_lower: np.ndarray,
        q_upper: np.ndarray,
        *,
        device: str | "torch.device" = "cpu",
        dtype: "torch.dtype | None" = None,
    ) -> None:
        if torch is None:
            raise ImportError("torch is required")
        dtype = dtype or torch.float32
        self.device = torch.device(device)
        self.dtype = dtype
        self.joint_R = torch.as_tensor(joint_R, dtype=dtype, device=self.device)
        self.joint_t = torch.as_tensor(joint_t, dtype=dtype, device=self.device)
        self.tcp_R = torch.as_tensor(tcp_R, dtype=dtype, device=self.device)
        self.tcp_t = torch.as_tensor(tcp_t, dtype=dtype, device=self.device)
        self.q_lower = torch.as_tensor(q_lower, dtype=dtype, device=self.device)
        self.q_upper = torch.as_tensor(q_upper, dtype=dtype, device=self.device)
        if self.joint_R.shape != (7, 3, 3) or self.joint_t.shape != (7, 3):
            raise ValueError("expected seven serial revolute joint placements")

    @classmethod
    def from_locked_model(
        cls,
        lm,
        *,
        device: str | "torch.device" = "cpu",
        dtype: "torch.dtype | None" = None,
    ) -> "TorchRM75Kinematics":
        model = lm.model
        joint_R = np.stack(
            [np.asarray(model.jointPlacements[i].rotation) for i in range(1, 8)]
        )
        joint_t = np.stack(
            [np.asarray(model.jointPlacements[i].translation) for i in range(1, 8)]
        )
        frame = model.frames[lm.tcp_id]
        if int(frame.parentJoint) != 7:
            raise ValueError("TCP frame must be attached to joint 7")
        return cls(
            joint_R,
            joint_t,
            np.asarray(frame.placement.rotation),
            np.asarray(frame.placement.translation),
            lm.q_lower,
            lm.q_upper,
            device=device,
            dtype=dtype,
        )

    def fk(
        self, q: "torch.Tensor", *, return_jacobian: bool = False
    ) -> tuple["torch.Tensor", "torch.Tensor"] | tuple[
        "torch.Tensor", "torch.Tensor", "torch.Tensor"
    ]:
        q = q.to(device=self.device, dtype=self.dtype)
        if q.shape[-1] != 7:
            raise ValueError(f"q must end in 7, got {tuple(q.shape)}")
        shape = q.shape[:-1]
        R = torch.eye(3, dtype=self.dtype, device=self.device).expand(*shape, 3, 3).clone()
        p = torch.zeros(*shape, 3, dtype=self.dtype, device=self.device)
        origins, axes = [], []
        z_local = torch.tensor([0.0, 0.0, 1.0], dtype=self.dtype, device=self.device)
        for j in range(7):
            p = p + (R @ self.joint_t[j].expand(*shape, 3).unsqueeze(-1)).squeeze(-1)
            R = R @ self.joint_R[j]
            origins.append(p)
            axes.append((R @ z_local.expand(*shape, 3).unsqueeze(-1)).squeeze(-1))
            R = R @ _skew_z(q[..., j])
        p_tcp = p + (R @ self.tcp_t.expand(*shape, 3).unsqueeze(-1)).squeeze(-1)
        R_tcp = R @ self.tcp_R
        if not return_jacobian:
            return p_tcp, R_tcp
        origin = torch.stack(origins, dim=-2)
        axis = torch.stack(axes, dim=-2)
        Jv = torch.cross(axis, p_tcp.unsqueeze(-2) - origin, dim=-1).transpose(-1, -2)
        Jw = axis.transpose(-1, -2)
        return p_tcp, R_tcp, torch.cat([Jv, Jw], dim=-2)

    @torch.no_grad()
    def ik_dls(
        self,
        target_p: "torch.Tensor",
        target_R: "torch.Tensor",
        q0: "torch.Tensor",
        *,
        max_iter: int = 80,
        damping: float = 2.0e-3,
        step_size: float = 0.8,
        max_step_rad: float = 0.25,
        tol_pos_m: float = 2.0e-4,
        tol_rot_rad: float = 1.0e-3,
    ) -> BatchIkResult:
        """Solve a broadcast batch. ``q0`` may include a seed dimension."""
        target_p = target_p.to(device=self.device, dtype=self.dtype)
        target_R = target_R.to(device=self.device, dtype=self.dtype)
        q = q0.to(device=self.device, dtype=self.dtype).clone()
        batch_shape = q.shape[:-1]
        while target_p.ndim < q.ndim:
            target_p = target_p.unsqueeze(-2)
        while target_R.ndim < q.ndim + 1:
            target_R = target_R.unsqueeze(-3)
        target_p = torch.broadcast_to(target_p, (*batch_shape, 3))
        target_R = torch.broadcast_to(target_R, (*batch_shape, 3, 3))
        eye6 = torch.eye(6, dtype=self.dtype, device=self.device).expand(*batch_shape, 6, 6)
        iterations = 0
        for iterations in range(1, int(max_iter) + 1):
            p, R, J = self.fk(q, return_jacobian=True)
            e_pos = target_p - p
            e_rot = so3_log(target_R @ R.transpose(-1, -2))
            pos_norm = torch.linalg.vector_norm(e_pos, dim=-1)
            rot_norm = torch.linalg.vector_norm(e_rot, dim=-1)
            active = (pos_norm > tol_pos_m) | (rot_norm > tol_rot_rad)
            if not bool(active.any()):
                break
            error = torch.cat([e_pos, e_rot], dim=-1)
            JJt = J @ J.transpose(-1, -2)
            solve = torch.linalg.solve(JJt + float(damping) ** 2 * eye6, error.unsqueeze(-1))
            dq = (J.transpose(-1, -2) @ solve).squeeze(-1)
            scale = (float(max_step_rad) / dq.abs().amax(dim=-1).clamp_min(float(max_step_rad))).clamp_max(1.0)
            dq = dq * scale.unsqueeze(-1)
            q_next = (q + float(step_size) * dq).clamp(self.q_lower, self.q_upper)
            q = torch.where(active.unsqueeze(-1), q_next, q)
        p, R = self.fk(q)
        pos_error = torch.linalg.vector_norm(target_p - p, dim=-1)
        rot_error = torch.linalg.vector_norm(so3_log(target_R @ R.transpose(-1, -2)), dim=-1)
        ok = (pos_error <= tol_pos_m) & (rot_error <= tol_rot_rad)
        return BatchIkResult(q, ok, pos_error, rot_error, iterations)
