"""Posed-only 3D figure of the latest vessel plan (full body, equal scale)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rsc

from perception.vessel_skin_plan import (
    _load_posed_mesh,
    _load_tpose_mesh,
    _project_label,
    bind_tpose_to_posed,
    latest_plan_path,
    load_plan_file,
    repo_root,
)

# standoff lives with the scan app; keep this script import-light.
def _standoff(contact: np.ndarray, dz: float) -> np.ndarray:
    R = Rsc.from_euler("xyz", np.asarray(contact, float).reshape(6)[3:6]).as_matrix()
    out = np.asarray(contact, float).reshape(6).copy()
    out[:3] = out[:3] + R @ np.array([0.0, 0.0, -float(dz)])
    return out


def _equal_limits(ax, pts: np.ndarray) -> None:
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    c = 0.5 * (lo + hi)
    r = 0.5 * float(np.max(hi - lo)) + 0.04
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def _triad(ax, p, R, scale: float) -> None:
    colors = ("#d62728", "#2ca02c", "#1f77b4")
    labels = ("+X ⊥path", "+Y path", "+Z into")
    for i, (c, lab) in enumerate(zip(colors, labels)):
        q = p + scale * R[:, i]
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color=c, lw=2.4)
        ax.text(q[0], q[1], q[2], lab, color=c, fontsize=8)


def render(plan_path: Path, out_path: Path | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    plan = load_plan_file(plan_path)
    run_dir = plan_path.parent
    posed, faces = _load_posed_mesh(run_dir / "moment_0000" / "smplx_result.npz")
    root = repo_root()
    tpose, tfaces = _load_tpose_mesh(root / "outputs/anatomy_retarget/latest_canonical")
    tpose_pts, _h = _project_label(plan.label, repo=root)
    full_world, _, _ = bind_tpose_to_posed(
        tpose_pts, tpose, posed, faces if faces.size else tfaces
    )
    xyz = np.asarray(plan.world_xyz, float)
    contact = np.asarray(plan.contact_pose, float)
    standoff = _standoff(contact, plan.standoff_m)
    Rc = Rsc.from_euler("xyz", contact[3:6]).as_matrix()
    tris = posed[np.asarray(faces, np.int32)]

    fig = plt.figure(figsize=(13.2, 10.8), dpi=140)
    views = ((18, -70), (78, -90), (8, 8), (22, -145))
    titles = ("from robot", "top", "along +X", "oblique")
    for i, ((elev, azim), title) in enumerate(zip(views, titles), 1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        coll = Poly3DCollection(
            tris,
            facecolor="#e39b6a",
            edgecolor="#b56a3d",
            linewidths=0.04,
            alpha=0.55,
        )
        ax.add_collection3d(coll)
        ax.plot(full_world[:, 0], full_world[:, 1], full_world[:, 2], color="#5a3418", lw=2.0)
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="#ff7f0e", lw=3.4)
        ax.scatter([contact[0]], [contact[1]], [contact[2]], c="#d62728", s=36, depthshade=False)
        ax.scatter([standoff[0]], [standoff[1]], [standoff[2]], c="#17becf", s=36, depthshade=False)
        _triad(ax, contact[:3], Rc, 0.12)
        _equal_limits(ax, posed)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title, fontsize=10)
    fig.suptitle(
        f"{plan.run_name}  posed capture  {plan.label}  {plan.window_m*100:.1f}cm\n"
        f"red +X ⊥path   green +Y along path   blue +Z into skin   "
        f"cyan = 5cm −Z standoff",
        fontsize=11,
    )
    fig.tight_layout()
    dest = out_path or (run_dir / "vessel_plan_posed_tcp_3d.png")
    fig.savefig(dest, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, default=None)
    args = ap.parse_args()
    path = args.plan or latest_plan_path()
    if path is None:
        raise SystemExit("no vessel_plan.json")
    out = render(path)
    print(out)


if __name__ == "__main__":
    main()
