"""IRD field as the rail inverse-reachability operator for QPIK.

The signed field in ``ird_playground`` is a reachability-margin proxy on the
probe45 flange chart (Vahrenkamp IRM / RM4D).  It is the right tool for
choosing the coordinated offset ``d* = y_tcp − y_rail`` and for a local
``∂IRD/∂rail`` at fixed TCP.  It is not a per-joint certificate and not a
control-thread planner: one-shot ``query_d_star`` at scan start only.

Hardware may be running gripper2.  Queries always rebuild the field's
probe45 TCP from ``link_7`` so the live tool offset cannot poison the chart.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rm75_control.control.joint_admittance_8dof.model import RobotKinematics


_REPO_ROOT = Path(__file__).resolve().parents[5]
_DEFAULT_CKPT = (
    _REPO_ROOT / "ird_playground" / "data" / "checkpoints" / "rm4d_signed" / "selected.pt"
)
_DEFAULT_ROBOT = _REPO_ROOT / "ird_playground" / "configs" / "robot_probe45.yaml"


@dataclass
class IrdConfig:
    enabled: bool = False
    checkpoint: str = str(_DEFAULT_CKPT)
    robot_spec: str = str(_DEFAULT_ROBOT)
    device: str = "cpu"
    # Production selected.pt predates artifact_schema v2; allow until rebuilt.
    allow_stale: bool = True
    goodness_period_ticks: int = 10


class IrdFieldHandle:
    """Loaded ReachabilitySDF + J1-axis map.  None of this runs SRS IK."""

    def __init__(self, cfg: IrdConfig | None = None) -> None:
        self.cfg = cfg or IrdConfig()
        self.field = None
        self._torch = None
        self.T_flange_tcp = np.eye(4, dtype=np.float64)
        self.T_world_rail = np.eye(4, dtype=np.float64)
        self.T_j1_axis0 = np.eye(4, dtype=np.float64)
        self.last_clearance: float = float("nan")
        self.last_d_star_clearance: float = float("nan")
        self._load()

    @property
    def available(self) -> bool:
        return self.field is not None and self._torch is not None

    def _load(self) -> None:
        ird_root = _REPO_ROOT / "ird_playground"
        if ird_root.is_dir():
            import sys

            root_s = str(ird_root)
            if root_s not in sys.path:
                sys.path.insert(0, root_s)
        ckpt = Path(self.cfg.checkpoint)
        spec_path = Path(self.cfg.robot_spec)
        if not ckpt.is_absolute():
            ckpt = (_REPO_ROOT / ckpt).resolve()
            if not ckpt.is_file():
                ckpt = (Path.cwd() / self.cfg.checkpoint).resolve()
        if not spec_path.is_absolute():
            spec_path = (_REPO_ROOT / spec_path).resolve()
            if not spec_path.is_file():
                spec_path = (Path.cwd() / self.cfg.robot_spec).resolve()
        if not ckpt.is_file() or not spec_path.is_file():
            return
        try:
            import torch
            from ird_playground.ird.robot_model import load_robot_model_spec
            from ird_playground.neural.signed_field import ReachabilitySDF
        except ImportError:
            return
        spec = load_robot_model_spec(spec_path)
        field = ReachabilitySDF.load(
            ckpt,
            device=str(self.cfg.device),
            expected_robot=spec,
            allow_stale=bool(self.cfg.allow_stale),
        )
        self._torch = torch
        self.field = field
        self.T_flange_tcp = np.asarray(
            field.model.T_flange_tcp.detach().cpu().numpy(), dtype=np.float64
        )
        self.T_j1_axis0 = np.asarray(spec.root_to_j1_axis(), dtype=np.float64)
        self.T_world_rail = np.eye(4, dtype=np.float64)

    def tcp_ird_from_flange(self, T_flange_world: np.ndarray) -> np.ndarray:
        """probe45 TCP the field was trained on, from live ``link_7``."""
        return np.asarray(T_flange_world, dtype=np.float64) @ self.T_flange_tcp

    def tcp_ird_from_q(self, kin: RobotKinematics, q_rad: np.ndarray) -> np.ndarray:
        M = kin.frame_placement(np.asarray(q_rad, dtype=float), "link_7")
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.asarray(M.rotation, dtype=np.float64)
        T[:3, 3] = np.asarray(M.translation, dtype=np.float64)
        return self.tcp_ird_from_flange(T)

    def _axis_from_rail(self, rail: "object") -> "object":
        from ird_playground.region.operator import base_from_rail_torch

        torch = self._torch
        device = self.field.device
        return base_from_rail_torch(
            rail,
            torch.as_tensor(self.T_world_rail, dtype=torch.float32, device=device),
            torch.as_tensor(self.T_j1_axis0, dtype=torch.float32, device=device),
            axis=1,
        )

    def score_tcp_rail(self, T_tcp_ird: np.ndarray, rail_m: float) -> float:
        torch = self._torch
        device = self.field.device
        tcp = torch.as_tensor(T_tcp_ird, dtype=torch.float32, device=device)
        rail = torch.as_tensor(float(rail_m), dtype=torch.float32, device=device)
        with torch.no_grad():
            val = self.field.score_world(tcp, self._axis_from_rail(rail))
        return float(val.detach().cpu().reshape(-1)[0])

    def g(self, kin: RobotKinematics, q_rad: np.ndarray) -> float:
        q = np.asarray(q_rad, dtype=float)
        val = self.score_tcp_rail(self.tcp_ird_from_q(kin, q), float(q[0]))
        self.last_clearance = val
        return val

    def dg_dy_rail(self, kin: RobotKinematics, q_rad: np.ndarray) -> float:
        """∂IRD/∂rail at the current TCP (true inverse-reachability slope)."""
        torch = self._torch
        device = self.field.device
        T = self.tcp_ird_from_q(kin, np.asarray(q_rad, dtype=float))
        tcp = torch.as_tensor(T, dtype=torch.float32, device=device)
        rail = torch.tensor(
            float(q_rad[0]), dtype=torch.float32, device=device, requires_grad=True
        )
        clearance = self.field.score_world(tcp, self._axis_from_rail(rail))
        (grad,) = torch.autograd.grad(clearance.reshape(()), rail, retain_graph=False)
        return float(grad.detach().cpu())

    def query_d_star(
        self,
        T_tcp_ird0: np.ndarray,
        *,
        y_tcp0_m: float,
        y_samples_m: np.ndarray,
        d_samples_m: np.ndarray,
        rail_lo: float,
        rail_hi: float,
        tau: float = 0.15,
    ) -> float | None:
        """Softmin IRD over a Y-stroke for each coordinated offset ``d = y − rail``.

        ``y_samples_m`` / ``y_tcp0_m`` are the *live* tool TCP world-Y values
        (same convention as ``d*``).  The IRD TCP is the probe45 pose that
        shares that world-Y increment so a gripper2 mount cannot shift the
        chart.
        """
        from ird_playground.region.operator import normalized_softmin

        torch = self._torch
        device = self.field.device
        y = np.asarray(y_samples_m, dtype=np.float64).reshape(-1)
        d = np.asarray(d_samples_m, dtype=np.float64).reshape(-1)
        if y.size == 0 or d.size == 0:
            return None
        # y − d ∈ [rail_lo, rail_hi] for every waypoint, not a fixed-rail IRM.
        d_lo = float(y.max()) - float(rail_hi)
        d_hi = float(y.min()) - float(rail_lo)
        if d_lo > d_hi + 1.0e-9:
            return None
        d = np.unique(np.clip(d, d_lo, d_hi))
        rails = y[None, :] - d[:, None]
        T0 = np.asarray(T_tcp_ird0, dtype=np.float64)
        n_d, n_y = rails.shape
        T = np.broadcast_to(T0, (n_d, n_y, 4, 4)).copy()
        T[..., 1, 3] = T0[1, 3] + (y[None, :] - float(y_tcp0_m))
        tcp = torch.as_tensor(T, dtype=torch.float32, device=device)
        rail_t = torch.as_tensor(rails, dtype=torch.float32, device=device)
        with torch.no_grad():
            scores = self.field.score_world(tcp, self._axis_from_rail(rail_t))
            clearance = normalized_softmin(scores, float(tau), dim=-1)
        idx = int(torch.argmax(clearance).detach().cpu())
        self.last_d_star_clearance = float(clearance[idx].detach().cpu())
        return float(d[idx])


class IrdRailGoodness:
    """``RailGoodness`` backed by the signed IRD field.

    Do not install this on the 5 ms thread: ``dg_dy_rail`` uses autograd
    and has measured 127 ms hitches.  Production goodness is σ_min.
    """

    def __init__(self, kin: RobotKinematics, handle: IrdFieldHandle) -> None:
        self.kin = kin
        self.handle = handle

    def g(self, q_rad: np.ndarray) -> float:
        return float(self.handle.g(self.kin, q_rad))

    def dg_dy_rail(self, q_rad: np.ndarray) -> float:
        return float(self.handle.dg_dy_rail(self.kin, q_rad))


def try_load_ird(cfg: IrdConfig | None = None) -> IrdFieldHandle | None:
    handle = IrdFieldHandle(cfg)
    return handle if handle.available else None


__all__ = [
    "IrdConfig",
    "IrdFieldHandle",
    "IrdRailGoodness",
    "try_load_ird",
]
