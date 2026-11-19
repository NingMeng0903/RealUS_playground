"""Parametric hidden-vessel worlds for the topology-discovery screen.

The public action graph is deliberately fixed: :func:`load` exposes the
checked 30-node bank and its costs.  Only the observation kernel varies by
case.  Latent families are generated from smooth 3-D centre lines and are
projected through the frozen atlas frames, rather than being encoded in a
public topology label or action set.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import lzma
from pathlib import Path
import sys

import numpy as np
from scipy.special import ndtr


WIDTH = 25.0
DEPTH = (3.0, 52.0)
IMAGE_SD = np.array([1.0, 0.68])
OUTLIER_SD = np.array([3.0, 0.68])
OUTLIER = 0.075
X_BINS = np.array([-8.0, -4.0, -1.5, 1.5, 4.0, 8.0])
Z_BINS = np.array([-5.0, -2.0, -0.7, 0.7, 2.0, 5.0])
N_FINE = 49
N_CLOCK = 16
FAMILY_NAMES = ("Y", "single", "parallel", "crossing")
_SPLIT_CODE = {"train": 11, "development": 23, "confirmation": 37,
               "eval": 41, "ood": 53, "stress": 67}


@dataclass(frozen=True)
class CaseBatch:
    """Private truth and public physical kernels for one case cohort."""

    labels: np.ndarray
    families: np.ndarray
    fork_mm: np.ndarray
    detect: np.ndarray
    cdf: np.ndarray
    centers: np.ndarray


@dataclass(frozen=True)
class Geometry:
    """Checked public graph plus frozen atlas projection data."""

    sites: np.ndarray
    views: np.ndarray
    cost: np.ndarray
    tip_xyz_mm: np.ndarray
    material_s: np.ndarray
    clock_s: np.ndarray
    start: int
    world: object
    nominal_px: np.ndarray
    trunk_xyz_mm: np.ndarray
    frames: np.ndarray
    nodes: tuple[dict, ...]


def _read(path: Path) -> dict:
    with lzma.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _bank_path(repo: Path, relative: str) -> Path:
    path = Path(repo).resolve() / relative
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _merge_edges(cost: np.ndarray, bank: dict, limit: int = 30) -> None:
    for edge in bank["edges"]:
        a, b = int(edge["start"]), int(edge["goal"])
        if a >= limit or b >= limit:
            continue
        row = np.array([edge["travel_mm"], edge["turn_deg"], edge["rail_mm"]], dtype=float)
        if not np.isfinite(cost[a, b, 0]) or row.sum() < cost[a, b].sum():
            cost[a, b] = row


def _legacy_world(repo: Path):
    root = Path(repo).resolve() / "NMI_idea" / "experiments" / "ab_full_b_rollout_20261113"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    acoustics = importlib.import_module("acoustics")
    return acoustics.setup_world(repo)


def load(repo: Path) -> Geometry:
    """Load public nodes and the union of already checked old-ID edges."""
    repo = Path(repo)
    base = _read(_bank_path(
        repo, "NMI_idea/experiments/ab_full_b_rollout_20261113/motion/BANK_MAIN.json.xz"))
    nodes = tuple(sorted(base["nodes"], key=lambda row: int(row["id"])))
    if len(nodes) != 30 or [int(row["id"]) for row in nodes] != list(range(30)):
        raise ValueError("BANK_MAIN must contain node IDs 0..29")
    cost = np.full((30, 30, 3), np.inf, dtype=float)
    _merge_edges(cost, base)
    inverse = repo / "NMI_idea/experiments/ab_inverse_acquisition_b_20261113/motion/BANK.json.xz"
    if inverse.exists():
        _merge_edges(cost, _read(inverse))
    base_world = _legacy_world(repo)
    material_s = np.asarray(base_world.scan.s, dtype=float).copy()
    trunk = 1000.0 * np.asarray(base_world._blend(base_world.scan.trunk, 15.0), dtype=float)
    frames = np.asarray(base_world._blend(base_world.scan.frames, 15.0), dtype=float)
    nominal, tip = [], []
    for row in nodes:
        site, view = int(row["site"]), int(row["view"])
        nominal.append(np.asarray(base_world.projection(15.0, site, view)[0][0], dtype=float))
        tip.append(np.asarray(row["tip_xyz_mm"], dtype=float))
    start = next(int(row["id"]) for row in nodes
                 if int(row["site"]) == 7 and int(row["view"]) == 3
                 and str(row["realization"]) == "dird_top")
    # Index 0 is the common prescan exit.  The .1 s allotment is included in
    # each public step so all controllers share a physical acquisition clock.
    return Geometry(
        sites=np.asarray([int(row["site"]) for row in nodes], dtype=np.int64),
        views=np.asarray([int(row["view"]) for row in nodes], dtype=np.int64),
        cost=cost,
        tip_xyz_mm=np.asarray(tip, dtype=float),
        material_s=material_s,
        clock_s=np.arange(N_CLOCK, dtype=float) * 1.60,
        start=start,
        world=base_world,
        nominal_px=np.asarray(nominal, dtype=float),
        trunk_xyz_mm=trunk,
        frames=frames,
        nodes=nodes,
    )


def _rng(seed: int, split: str) -> np.random.Generator:
    code = _SPLIT_CODE.get(str(split))
    if code is None:
        code = int.from_bytes(hashlib.sha256(str(split).encode()).digest()[:4], "little")
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(code), 78231]))


def _cdf(mu: np.ndarray, nominal: np.ndarray) -> np.ndarray:
    """Return aperture-truncated cumulative 7x7 bin masses.

    ``mu`` is [T,N,H,3,2] in probe/image coordinates.  The returned last
    axis is x-major 7x7 order, matching ``return_sets.build_tables``.
    """
    xedge = np.concatenate((np.full((len(nominal), 1), -WIDTH),
                            nominal[:, 0, None] + X_BINS[None],
                            np.full((len(nominal), 1), WIDTH)), axis=1)
    zedge = np.concatenate((np.full((len(nominal), 1), DEPTH[0]),
                            nominal[:, 1, None] + Z_BINS[None],
                            np.full((len(nominal), 1), DEPTH[1])), axis=1)
    combined = np.zeros((*mu.shape[:-1], 7, 7), dtype=np.float64)
    for sd, weight in ((IMAGE_SD, 1.0 - OUTLIER), (OUTLIER_SD, OUTLIER)):
        px = np.diff(ndtr((xedge[None, :, None, None, :] - mu[..., 0, None]) / sd[0]), axis=-1)
        pz = np.diff(ndtr((zedge[None, :, None, None, :] - mu[..., 1, None]) / sd[1]), axis=-1)
        combined += weight * px[..., :, None] * pz[..., None, :]
    flat = combined.reshape(*mu.shape[:-1], N_FINE)
    total = flat.sum(axis=-1, keepdims=True)
    flat = np.divide(flat, total, out=np.full_like(flat, 1.0 / N_FINE), where=total > 1.e-15)
    return np.cumsum(flat, axis=-1).astype(np.float32)


def _aperture(mu: np.ndarray) -> np.ndarray:
    lo, hi = np.asarray([-WIDTH, DEPTH[0]]), np.asarray([WIDTH, DEPTH[1]])
    out = np.zeros(mu.shape[:-1], dtype=float)
    for sd, weight in ((IMAGE_SD, 1.0 - OUTLIER), (OUTLIER_SD, OUTLIER)):
        out += weight * np.prod(ndtr((hi - mu) / sd) - ndtr((lo - mu) / sd), axis=-1)
    return np.clip(out, 0.0, 1.0)


def cases(world: Geometry, seed: int, count: int, split: str = "train") -> CaseBatch:
    """Generate a paired cohort with four hidden graph families.

    ``centers`` are image centroids relative to the nominal trunk projection.
    ``detect`` and ``cdf`` are the public kernel; labels/families/forks are
    retained only for evaluator scoring.
    """
    if count < 1:
        raise ValueError("count must be positive")
    rng = _rng(seed, split)
    h, n, t = int(count), len(world.sites), len(world.clock_s)
    # The atlas chart has 28 material samples, whereas the public bank has 30
    # pose nodes (six views at each checked site).  Generate hidden centreline
    # geometry on the chart and select the bank sites only at projection time.
    s = world.material_s
    site_index = np.asarray(world.sites, dtype=np.int64)
    s_nodes = s[site_index]
    frames = world.frames
    # Scan frames store basis vectors in the last axis (the same convention as
    # ``MapWorld.projection``), so columns—not matrix rows—are world directions.
    lateral, depth, tangent = frames[..., 1], frames[..., 2], frames[..., 0]
    # Half the prior carries a Y daughter (the positive endpoint); the other
    # half is split evenly across the three negative/distractor families.
    n_y = (h + 1) // 2
    n_other = h - n_y
    family = np.concatenate((
        np.zeros(n_y, dtype=np.int64),
        np.tile(np.arange(1, 4, dtype=np.int64), (n_other + 2) // 3)[:n_other],
    ))
    rng.shuffle(family)
    j = rng.integers(6, 8, size=h)
    # Public side convention: at the canonical view-3 center plane, +1 is
    # positive image-x along the local scan-frame lateral basis; -1 is left.
    side = np.where(rng.integers(0, 2, size=h) == 0, -1.0, 1.0)
    fork_path = s[j - 1] + rng.uniform(.18, .82, size=h) * (s[j] - s[j - 1])
    fork = np.full(h, np.nan, dtype=float)
    y_rows = family == 0
    fork[y_rows] = fork_path[y_rows]
    labels = np.zeros(h, dtype=np.int64)
    labels[y_rows] = 1 + 2 * (j[y_rows] - 6) + (side[y_rows] > 0)

    held_shape = str(split).lower() == "shape_holdout"
    ood = str(split).lower() in {"ood", "stress"} or held_shape
    # The daughter extends toward decreasing material-s (the j-2/j-3 distal
    # targets).  Keep its support local; a full-chart span would make the two
    # sides project almost identically at the checked sites.
    span = np.clip(fork_path - s[np.maximum(j - 3, 0)], 30.0, 50.0)
    prof = np.exp(-0.5 * ((s[None] - s[j - 1, None]) / 85.0) ** 2)
    # Coherent nuisance/deformation parameters are drawn once per case.
    shift_l = rng.normal(0., 1.5, h)
    shift_d = rng.normal(0., 1.0, h)
    curve = rng.normal(0., 2.0 if not ood else 5.0, h)
    phase = rng.uniform(0., 2 * np.pi, h)
    amp = rng.uniform(.25, 1.0 if not ood else 1.8, h)
    trunk_static = world.trunk_xyz_mm[None] + (
        lateral[None] * shift_l[:, None, None] * prof[:, :, None]
        + depth[None] * shift_d[:, None, None] * prof[:, :, None]
        + tangent[None] * curve[:, None, None]
        * (((s[None] - s[j - 1, None]) / np.maximum(span[:, None], 1.))
           * prof)[:, :, None]
    )
    temporal = amp[:, None] * np.sin(phase[:, None] + world.clock_s[None] / 3.7)
    motion = temporal[:, :, None, None] * depth[None, None, :, :]
    trunk = trunk_static[:, None] + motion

    # A Y branch joins the trunk at a continuous fork between s[j-1] and s[j].
    fork_for_path = fork_path
    u = np.clip((fork_for_path[:, None] - s[None]) / span[:, None], 0., 1.)
    smooth = u * u * (3. - 2. * u)
    sep = rng.uniform(5.5, 10.5, h)
    # Preserve the sampled left/right convention at the distal endpoint even
    # for the stronger OOD curvature draws.
    bend = np.maximum(rng.normal(0., 3.0 if not ood else 7.0, h), -.85 * sep)
    branch_lat = side[:, None] * (sep[:, None] * smooth + bend[:, None] * smooth * smooth)
    branch_depth = rng.normal(0., 2.0, h)[:, None] * smooth
    if held_shape:
        # A smooth S-bend is absent from the nominal polynomial family.
        # Multiplication by a positive factor preserves the distal side.
        branch_lat *= 1. + .28*np.sin(2.*np.pi*u)
        branch_depth += 1.5*smooth*np.sin(2.*np.pi*u)
    branch = trunk + lateral[None, None] * branch_lat[:, None, :, None] + depth[None, None] * branch_depth[:, None, :, None]
    parallel = family == 2
    crossing = family == 3
    parallel_lat = side[:, None] * (sep[:, None] + bend[:, None] * u)
    parallel_depth = rng.choice([-1., 1.], h)[:, None] * rng.uniform(5., 10., h)[:, None]
    cross_lat = side[:, None] * sep[:, None] * np.tanh((fork_for_path[:, None] - s[None]) / (18. + 5. * ood))
    cross_depth = rng.choice([-1., 1.], h)[:, None] * rng.uniform(5., 11., h)[:, None]
    parallel_pts = trunk + lateral[None, None] * parallel_lat[:, None, :, None] + depth[None, None] * parallel_depth[:, None, :, None]
    crossing_pts = trunk + lateral[None, None] * cross_lat[:, None, :, None] + depth[None, None] * cross_depth[:, None, :, None]
    side_pts = np.where(parallel[:, None, None, None], parallel_pts, branch)
    side_pts = np.where(crossing[:, None, None, None], crossing_pts, side_pts)

    # One persistent, curved ghost per case; visibility may dip in time/view,
    # but its geometry is not independently redrawn at every acquisition.
    gsite = rng.integers(3, min(18, n - 2), h)
    gw = rng.uniform(18., 46., h)
    gp = np.exp(-0.5 * ((s[None] - s[gsite, None]) / gw[:, None]) ** 2)
    gl = rng.normal(0., 7.0 if not ood else 11.0, h)[:, None] * gp
    gd = rng.normal(4.5, 2.0 if not ood else 4.0, h)[:, None] * gp
    if held_shape:
        gl += 2.5*gp*np.sin((s[None]-s[gsite,None])/12.)
        gd += 1.5*gp*np.cos((s[None]-s[gsite,None])/12.)
    ghost_static = trunk_static + lateral[None] * gl[:, :, None] + depth[None] * gd[:, :, None]
    ghost_static += tangent[None] * rng.normal(0., 3., h)[:, None, None] * gp[:, :, None]
    ghost = ghost_static[:, None] + motion

    points = np.stack((side_pts, trunk, ghost), axis=3)  # [H,T,N,3,3]
    # Near-fork overlap is intentionally represented by the independent
    # centroid kernels; 7x7 quantization and return_sets' unordered packet
    # expansion merge indistinguishable returns without inventing a label.
    # Project only the 30 public bank nodes; the hidden geometry remains on the
    # full chart so that family parameters are continuous between samples.
    delta = points[:, :, site_index, :, :] - world.trunk_xyz_mm[site_index][None, None, :, None, :]
    frame_rows = np.empty((n, 2, 3), dtype=float)
    nominal = np.asarray(world.nominal_px, dtype=float)
    for k, (site, view) in enumerate(zip(world.sites, world.views)):
        pose = world.world._blend(world.world.scan.view_frames[:, int(site), int(view)], 15.0)
        frame_rows[k] = pose[:, 1:3].T
    projected = np.einsum("htnkc,nac->htnka", delta, frame_rows).transpose(1, 2, 0, 3, 4).astype(np.float32)

    side_present = np.ones((h, n), dtype=bool)
    side_present[family == 1] = False
    side_present[y_rows] &= s_nodes[None] <= fork[y_rows, None]
    side_present = np.broadcast_to(side_present.T[None], (t, n, h))
    mu = projected + nominal[None, :, None, None, :]
    detect = np.empty((t, n, h, 3), dtype=np.float64)
    detect[..., 1] = .84
    detect[..., 0] = .78 * side_present
    ghost_view = np.where(world.views == 3, 1.0, np.where(world.views == 7, .78, .60))
    detect[..., 2] = (.16 + .34 * gp[:, site_index].T[None]) * ghost_view[None, :, None]
    # Persistent case-level dropout field, shared across all emitters.
    d0 = rng.uniform(6., 16., h)
    d1 = d0 + rng.uniform(.6, 2.2, h)
    outage = ((world.clock_s[:, None] >= d0[None]) & (world.clock_s[:, None] <= d1[None]))
    detect *= np.where(outage[:, None, :, None], .55, 1.0)
    detect *= _aperture(mu)
    return CaseBatch(labels, family, fork, detect.astype(np.float32), _cdf(mu, nominal), projected)


__all__ = ["CaseBatch", "Geometry", "FAMILY_NAMES", "cases", "load"]
