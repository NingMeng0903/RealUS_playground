"""Native male SMPL-X transport for the frozen shaft ROI charts.

The outer chart keeps its stored source-face/barycentric lineage after the
production CSR surface stencil.  Interior points use a declared radial
extension from a fixed-LBS core; this is a material-correspondence experiment,
not a biological tissue model.  No asset is written.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix


KNEE = {"left": 4, "right": 5}
LINEAGE_TOLERANCE_M = 2.0e-6


def _copy(a: Any, dtype: Any = None) -> np.ndarray:
    b = np.array(a, dtype=dtype, copy=True)
    b.setflags(write=False)
    return b


@dataclass
class SideAssets:
    """Frozen v2 chart arrays and the retained ownership attachment."""

    side: str
    vertices: np.ndarray
    tets: np.ndarray
    h: np.ndarray
    theta: np.ndarray
    depth: np.ndarray
    grid_shape: tuple[int, int, int]
    skin_faces: np.ndarray
    radial_fractions: np.ndarray
    core_points: np.ndarray
    core_lbs_weights: np.ndarray
    source_surface_face_indices: np.ndarray
    source_surface_barycentric: np.ndarray
    source_original_skin_vertex_indices: np.ndarray
    owned_vertices: np.ndarray
    owned_skin_faces: np.ndarray
    stencil_source_ids: np.ndarray
    stencil: csr_matrix
    lineage: dict[str, Any]


class ShapeTransport:
    """Transport both existing shaft ROIs through real male SMPL-X beta/pose."""

    def __init__(self, among_us: Path, repair: Path):
        among_us, repair = Path(among_us).resolve(), Path(repair).resolve()
        production = among_us / "dataset/processed/anatomy_retarget/leg_volume_coordinates/production"
        roi, ownership = repair / "harmonic/shaft_roi_v2", repair / "harmonic/ownership_smooth"
        self.source_paths: dict[str, Path] = {
            "smplx_male": among_us / "ref_code_library/InteractVLM/data/body_models/smplx/SMPLX_MALE.npz",
            "recovered_betas": among_us / "outputs/anatomy_retarget/recovered_smplx_20261108/recovered_atlas_betas.npy",
            "production": production,
            "shaft_roi_v2": roi,
            "ownership_smooth": ownership,
        }
        for side in KNEE:
            self.source_paths.update({
                f"atlas_{side}": roi / f"atlas_{side}.npz",
                f"attachments_{side}": roi / f"attachments_{side}.npz",
                f"owned_skin_{side}": ownership / f"owned_smooth_{side}_skin.npz",
                f"surface_stencil_{side}": production / f"material/surface_stencil_{side}.npz",
            })

        with np.load(self.source_paths["smplx_male"], allow_pickle=False) as m:
            self._template = np.asarray(m["v_template"], float)
            self._shapedirs = np.asarray(m["shapedirs"], float)[..., :10]
            self._regressor = np.asarray(m["J_regressor"], float)
            self._weights = np.asarray(m["weights"], float)
            self._parents = np.asarray(m["kintree_table"][0], np.int64)
            self._pose_dirs = {
                "left": np.array(m["posedirs"][..., 27:36], float, copy=True),
                "right": np.array(m["posedirs"][..., 36:45], float, copy=True),
            }
        self._parents[0] = -1
        if self._template.shape != (10475, 3) or self._weights.shape != (10475, 55):
            raise ValueError("Expected the original 10475-vertex, 55-joint SMPL-X model")
        self.beta = _copy(np.load(self.source_paths["recovered_betas"], allow_pickle=False), float)
        if self.beta.shape != (10,):
            raise ValueError(f"Recovered beta must have shape (10,), got {self.beta.shape}")
        self.sides = {side: self._load_side(side) for side in KNEE}

    def _load_side(self, side: str) -> SideAssets:
        with np.load(self.source_paths[f"atlas_{side}"], allow_pickle=False) as a:
            vertices = np.asarray(a["harmonic_vertices"], float)
            tets = np.asarray(a["harmonic_tets"], np.int32)
            h, theta, depth = (np.asarray(a[k], float) for k in ("harmonic_h", "harmonic_theta", "harmonic_d"))
            frozen_skin = np.asarray(a["skin_vertices"], float)
            skin_faces = np.asarray(a["skin_faces"], np.int32)
            core = np.asarray(a["core_points"], float)
        with np.load(self.source_paths[f"attachments_{side}"], allow_pickle=False) as a:
            grid = tuple(int(x) for x in a["grid_shape"])
            radial = np.asarray(a["radial_fractions"], float)
            core_w = np.asarray(a["core_lbs_weights"], float)
            face_ids = np.asarray(a["source_surface_face_indices"], np.int64)
            bary = np.asarray(a["source_surface_barycentric"], float)
            retained = np.asarray(a["source_original_skin_vertex_indices"], np.int64)
        with np.load(self.source_paths[f"owned_skin_{side}"], allow_pickle=False) as a:
            owned_vertices = np.asarray(a["skin_vertices"], float)
            owned_faces = np.asarray(a["skin_faces"], np.int32)
        with np.load(self.source_paths[f"surface_stencil_{side}"], allow_pickle=False) as a:
            source_ids = np.asarray(a["source_full_vertex_indices"], np.int64)
            stencil = csr_matrix((a["stencil_weights"], a["stencil_indices"], a["stencil_indptr"]),
                                 shape=(len(a["surface_full_vertex_indices"]), len(source_ids)))

        if grid != (72, 8, 108) or vertices.shape != (int(np.prod(grid)), 3):
            raise ValueError(f"Unexpected {side} chart grid/vertex shape: {grid}, {vertices.shape}")
        if len(frozen_skin) != grid[0] * grid[2] or len(face_ids) != len(frozen_skin):
            raise ValueError(f"Invalid {side} chart outer attachment shape")
        if len(retained) != len(owned_vertices) or len(radial) != grid[1] or radial[-1] != 1:
            raise ValueError(f"Invalid {side} retained-skin/radial attachment shape")
        if np.any(face_ids < 0) or np.any(face_ids >= len(owned_faces)):
            raise ValueError(f"Invalid {side} source face IDs")
        if np.any(np.abs(bary.sum(axis=1) - 1) > 1e-8) or np.any(bary < -1e-8):
            raise ValueError(f"Invalid {side} source barycentric coordinates")

        asset = SideAssets(side, _copy(vertices), _copy(tets, np.int32), _copy(h), _copy(theta),
                           _copy(depth), grid, _copy(skin_faces, np.int32), _copy(radial),
                           _copy(core), _copy(core_w), _copy(face_ids, np.int64), _copy(bary),
                           _copy(retained, np.int64), _copy(owned_vertices), _copy(owned_faces, np.int32),
                           _copy(source_ids, np.int64), stencil, {})
        state = self._native(side, np.zeros(2), 0.0)
        surface = asset.stencil @ state["native"][asset.stencil_source_ids]
        owned = surface[asset.source_original_skin_vertex_indices]
        chart = np.einsum("ni,nij->nj", asset.source_surface_barycentric,
                          owned[asset.owned_skin_faces[asset.source_surface_face_indices]])
        chart_error = np.linalg.norm(chart - frozen_skin, axis=1)
        owned_error = np.linalg.norm(owned - asset.owned_vertices, axis=1)
        if max(float(chart_error.max()), float(owned_error.max())) > LINEAGE_TOLERANCE_M:
            raise ValueError(f"{side} neutral source/face lineage exceeds 2 um")
        asset.lineage = {
            "side": side,
            "tolerance_m": LINEAGE_TOLERANCE_M,
            "neutral_source_face_barycentric_max_m": float(chart_error.max()),
            "neutral_retained_vertex_max_m": float(owned_error.max()),
            "fresh_ray_intersection": False,
        }
        return asset

    @staticmethod
    def _rotation(knee_deg: float) -> np.ndarray:
        c, s = np.cos(np.deg2rad(knee_deg)), np.sin(np.deg2rad(knee_deg))
        return np.array(((1., 0., 0.), (0., c, -s), (0., s, c)))

    def _native(self, side: str, delta: np.ndarray, knee_deg: float) -> dict[str, np.ndarray]:
        beta = np.array(self.beta, copy=True)
        beta[:2] += delta
        rest = self._template + np.einsum("vij,j->vi", self._shapedirs, beta)
        joints = self._regressor @ rest
        rot = self._rotation(knee_deg)
        global_tf, transforms = np.zeros((55, 4, 4)), np.zeros((55, 4, 4))
        for j, parent in enumerate(self._parents):
            local = np.eye(4)
            if j == KNEE[side]:
                local[:3, :3] = rot
            local[:3, 3] = joints[j] if parent < 0 else joints[j] - joints[parent]
            global_tf[j] = local if parent < 0 else global_tf[parent] @ local
            bind = np.eye(4); bind[:3, 3] = -joints[j]
            transforms[j] = global_tf[j] @ bind
        corrective = np.einsum("vij,j->vi", self._pose_dirs[side], (rot - np.eye(3)).reshape(9))
        blended = np.einsum("vj,jab->vab", self._weights, transforms)
        native = np.einsum("vab,vb->va", blended[:, :3, :3], rest + corrective) + blended[:, :3, 3]
        return {"native": native, "transforms": transforms, "joints": global_tf[:, :3, 3]}

    @staticmethod
    def _blend(points: np.ndarray, weights: np.ndarray, transforms: np.ndarray) -> np.ndarray:
        b = np.einsum("nj,jab->nab", weights, transforms)
        return np.einsum("nab,nb->na", b[:, :3, :3], points) + b[:, :3, 3]

    def deform(self, side: str, delta_beta: Any, knee_deg: Any):
        """Return chart vertices, owned native skin, owned faces, and 55 joints."""
        if side not in self.sides:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        delta = np.asarray(delta_beta, float)
        if delta.shape != (2,) or not np.all(np.isfinite(delta)):
            raise ValueError("delta_beta must be two finite values for beta0/beta1")
        knee = float(knee_deg)
        if not np.isfinite(knee) or not 0 <= knee <= 90:
            raise ValueError("knee_deg must lie in [0, 90]")
        asset = self.sides[side]
        state = self._native(side, delta, knee)
        surface = asset.stencil @ state["native"][asset.stencil_source_ids]
        owned = surface[asset.source_original_skin_vertex_indices]
        outer = np.einsum("ni,nij->nj", asset.source_surface_barycentric,
                          owned[asset.owned_skin_faces[asset.source_surface_face_indices]])
        rest_state = self._native(side, delta, 0.0)
        rest_surface = asset.stencil @ rest_state["native"][asset.stencil_source_ids]
        rest_owned = rest_surface[asset.source_original_skin_vertex_indices]
        rest_outer = np.einsum("ni,nij->nj", asset.source_surface_barycentric,
                               rest_owned[asset.owned_skin_faces[asset.source_surface_face_indices]])
        frozen_outer = asset.vertices.reshape(asset.grid_shape + (3,))[:, -1]
        core = asset.core_points + (rest_outer.reshape(asset.grid_shape[0], asset.grid_shape[2], 3)
                                    - frozen_outer).mean(axis=1)
        core = self._blend(core, asset.core_lbs_weights, state["transforms"])
        outer = outer.reshape(asset.grid_shape[0], asset.grid_shape[2], 3)
        volume = core[:, None, None] + asset.radial_fractions[None, :, None, None] * (outer[:, None, :, :] - core[:, None, None])
        return volume.reshape(-1, 3), np.array(owned), np.array(asset.owned_skin_faces), np.array(state["joints"])


__all__ = ["ShapeTransport", "SideAssets"]
