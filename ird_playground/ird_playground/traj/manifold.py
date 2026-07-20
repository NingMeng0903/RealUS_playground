"""GT vessel → skin manifold: λ ↦ T_tcp (deterministic; not NLP free vars).

Optimization variables are only (λ, r). Probe pose is produced by this map.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


def _complete_frame_n_t(n: np.ndarray, t: np.ndarray) -> np.ndarray:
    """R = [b, t_hat, n_hat] with n ≈ skin normal (probe +Z), t ≈ vessel tangent."""
    n = np.asarray(n, dtype=np.float64).reshape(3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    n = n / (np.linalg.norm(n) + 1e-12)
    t = t - n * float(np.dot(t, n))
    tn = np.linalg.norm(t)
    if tn < 1e-9:
        a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        t = np.cross(a, n)
        t = t / (np.linalg.norm(t) + 1e-12)
    else:
        t = t / tn
    b = np.cross(t, n)
    b = b / (np.linalg.norm(b) + 1e-12)
    t = np.cross(n, b)
    return np.stack([b, t, n], axis=1)


@dataclass
class VesselSkinSample:
    p: np.ndarray  # (3,) skin contact
    n: np.ndarray  # (3,) skin outward normal → probe +Z
    t: np.ndarray  # (3,) vessel tangent (in-plane)
    R: np.ndarray  # (3,3)
    T: np.ndarray  # (4,4)


class VesselSkinManifold:
    """Abstract λ ∈ ℝ → T_tcp. Concrete maps may use LBS / fiber / lookup tables."""

    def sample(self, lam: float) -> VesselSkinSample:
        raise NotImplementedError

    def sample_batch(self, lams: np.ndarray) -> list[VesselSkinSample]:
        return [self.sample(float(x)) for x in np.asarray(lams, dtype=np.float64).reshape(-1)]

    def T_tcp(self, lam: float) -> np.ndarray:
        return self.sample(lam).T

    def T_tcp_batch(self, lams: np.ndarray) -> np.ndarray:
        return np.stack([s.T for s in self.sample_batch(lams)], axis=0)


class SyntheticVesselSkinManifold(VesselSkinManifold):
    """Smooth synthetic skin curve for AD / P1 tests (no patient mesh required).

    λ is arc-length-like in meters along a planar arc in the xz plane, with a
    mild y undulation. Normal points roughly +Y (patient facing robot).
    """

    def __init__(
        self,
        *,
        center: tuple[float, float, float] = (0.35, 0.0, 0.25),
        radius_m: float = 0.12,
        length_m: float = 0.40,
        y_amp_m: float = 0.02,
    ) -> None:
        self.center = np.asarray(center, dtype=np.float64)
        self.radius_m = float(radius_m)
        self.length_m = float(length_m)
        self.y_amp_m = float(y_amp_m)

    def sample(self, lam: float) -> VesselSkinSample:
        # Map λ∈ℝ → angle along arc; clamp softly via tanh for stability
        s = float(lam)
        # centerline in xz, bulge in −y (skin facing robot at +y? use +y normal)
        ang = (s / max(self.length_m, 1e-6)) * np.pi  # ~0…π over length
        cx, cy, cz = self.center
        p = np.array(
            [
                cx + self.radius_m * np.sin(ang),
                cy + self.y_amp_m * np.sin(2.0 * ang),
                cz + self.radius_m * (1.0 - np.cos(ang)),
            ],
            dtype=np.float64,
        )
        # analytic tangent ds
        dang = np.pi / max(self.length_m, 1e-6)
        t = np.array(
            [
                self.radius_m * np.cos(ang) * dang,
                self.y_amp_m * 2.0 * np.cos(2.0 * ang) * dang,
                self.radius_m * np.sin(ang) * dang,
            ],
            dtype=np.float64,
        )
        # outward normal approx: from arc center toward point in xz, + small y
        n = np.array(
            [np.sin(ang), 0.85, 1.0 - np.cos(ang)],
            dtype=np.float64,
        )
        n = n / (np.linalg.norm(n) + 1e-12)
        R = _complete_frame_n_t(n, t)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = p
        return VesselSkinSample(p=p, n=n, t=t / (np.linalg.norm(t) + 1e-12), R=R, T=T)

    def sample_torch(
        self,
        lam: "torch.Tensor",
        *,
        dtype=None,
        device=None,
    ) -> "torch.Tensor":
        """Differentiable λ → T_tcp (4,4) or (N,4,4) for the synthetic map."""
        if torch is None:
            raise ImportError("torch required")
        dtype = dtype or torch.float32
        device = device or (lam.device if hasattr(lam, "device") else "cpu")
        lam = lam.to(device=device, dtype=dtype)
        single = lam.ndim == 0
        if single:
            lam = lam.reshape(1)
        L = max(self.length_m, 1e-6)
        ang = (lam / L) * np.pi
        cx, cy, cz = [float(x) for x in self.center]
        r = self.radius_m
        ya = self.y_amp_m
        px = cx + r * torch.sin(ang)
        py = cy + ya * torch.sin(2.0 * ang)
        pz = cz + r * (1.0 - torch.cos(ang))
        p = torch.stack([px, py, pz], dim=-1)  # (N,3)

        dang = np.pi / L
        tx = r * torch.cos(ang) * dang
        ty = ya * 2.0 * torch.cos(2.0 * ang) * dang
        tz = r * torch.sin(ang) * dang
        t = torch.stack([tx, ty, tz], dim=-1)
        t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        nx = torch.sin(ang)
        ny = torch.full_like(ang, 0.85)
        nz = 1.0 - torch.cos(ang)
        n = torch.stack([nx, ny, nz], dim=-1)
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        # Gram–Schmidt: R = [b, t_hat, n]
        t = t - n * (t * n).sum(dim=-1, keepdim=True)
        t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        b = torch.cross(t, n, dim=-1)
        b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        t = torch.cross(n, b, dim=-1)
        R = torch.stack([b, t, n], dim=-1)  # (N,3,3)

        N = lam.shape[0]
        T = torch.eye(4, dtype=dtype, device=device).expand(N, 4, 4).clone()
        T[:, :3, :3] = R
        T[:, :3, 3] = p
        return T[0] if single else T
