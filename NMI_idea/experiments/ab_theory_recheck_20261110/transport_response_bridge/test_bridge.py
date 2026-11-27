#!/usr/bin/env python3
"""Independent contract checks for the transport-response bridge."""

from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT.parent / "harmonic_mechanism"
ASSET = AUDIT.parent.parent / "ab_validation_20261110/repair_20261110/harmonic/shaft_roi_v2"
sys.path.insert(0, "/media/camp/EXT_DRIVE/Among_US/src")
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import load_leg_volume_atlas  # noqa: E402


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def mats(v):
    return np.moveaxis(v[:, 1:] - v[:, :1], 1, 2)


def check_pair(side, degree, cell_index):
    d = dict(np.load(ROOT / "RAW_RESULTS.npz"))
    h = dict(np.load(AUDIT / "RAW_RESULTS.npz"))
    a = load_leg_volume_atlas(ASSET / f"atlas_{side}.npz")
    ids = h[f"{side}_cell_ids"]
    b = h[f"{side}_barycentric"]
    cells = a.harmonic_tets[ids]
    v0 = a.harmonic_vertices[cells].astype(float)
    x0 = mats(v0)
    lab = np.stack((a.harmonic_theta, a.harmonic_h, a.harmonic_d), -1)[cells].astype(float)
    th = lab[:, :, 0]
    ss = np.sum(b * np.sin(th), axis=1)
    cc = np.sum(b * np.cos(th), axis=1)
    tb = (cc[:, None] * (np.sin(th[:, 1:]) - np.sin(th[:, :1])) - ss[:, None] * (np.cos(th[:, 1:]) - np.cos(th[:, :1]))) / (ss * ss + cc * cc)[:, None]
    kb = np.stack((tb, lab[:, 1:, 1] - lab[:, :1, 1], lab[:, 1:, 2] - lab[:, :1, 2]), axis=1)
    j0 = x0 @ np.linalg.inv(kb)
    t0 = unit(j0[:, :, 1])
    n0 = -unit(np.linalg.inv(j0)[:, 2, :])
    f = h[f"{side}_{degree}_deformation"].astype(float)
    t = unit(np.einsum("nij,nj->ni", f, t0))
    n = unit(np.linalg.solve(np.swapaxes(f, 1, 2), n0[..., None])[..., 0])
    rx, rz = t, -n
    ry = unit(np.cross(rz, rx))
    r = np.stack((rx, ry, rz), axis=2)
    i = cell_index
    assert np.max(np.abs(d[f"{side}_J0"][i] - j0[i])) < 1e-10
    assert np.max(np.abs(d[f"{side}_t0"][i] - t0[i])) < 1e-10
    assert np.max(np.abs(d[f"{side}_n0"][i] - n0[i])) < 1e-10
    assert np.max(np.abs(d[f"{side}_{degree}_t"][i] - t[i])) < 1e-10
    assert np.max(np.abs(d[f"{side}_{degree}_n"][i] - n[i])) < 1e-10
    assert np.max(np.abs(d[f"{side}_{degree}_R"][i] - r[i])) < 1e-10
    assert abs(float(np.dot(t0[i], n0[i]))) < 1e-10
    assert abs(float(np.dot(t[i], n[i]))) < 1e-10
    assert np.max(np.abs(r[i].T @ r[i] - np.eye(3))) < 1e-10
    assert float(np.linalg.det(r[i])) > 0.999999999

    pose = np.load(ASSET / f"{side}_pose_{degree:02d}.npz")
    vp = pose["vertices"].astype(float)[cells]
    p_bary = np.einsum("ni,nij->nj", b, vp)
    p_edges = vp[:, 0] + np.einsum("nij,nj->ni", mats(vp), b[:, 1:])
    assert float(np.max(np.linalg.norm(p_bary - p_edges, axis=1))) < 1e-12


def main():
    r = json.loads((ROOT / "RESULTS.json").read_text())
    assert r["status"] == "complete"
    assert len(r["metrics"]) == 2 * 7
    for row in r["metrics"]:
        for name in ("e3nn_correct", "e3nn_frozen_world", "e3nn_wrong_F_normal", "local_dyadic_correct"):
            m = row[name]
            assert np.isfinite(m["response_rmse_mm"])
            assert 0.0 <= m["success_rate_2mm"] <= 1.0
    # Two independent cells from two side/pose pairs, not a call into the run script.
    check_pair("right", 15, 13)
    check_pair("left", 75, 191)
    print("transport bridge independent checks: PASS")


if __name__ == "__main__":
    main()
