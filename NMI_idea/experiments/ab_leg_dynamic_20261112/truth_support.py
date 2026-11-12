"""Check that simulated displaced vessel nodes stay in the posed baked volume."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent/"ab_leg_bifurcation_20261112"
sys.path.insert(0,str(PREVIOUS))
from rebind_knee import TetCoverage


def build_support(scan, data):
    rebake = PREVIOUS/"knee_rebake"
    with np.load(rebake/"atlas_right.npz") as a, np.load(rebake/"attachments_right.npz") as b:
        vertices = a["harmonic_vertices"]
        tets = a["harmonic_tets"]
        weights = b["volume_lbs_weights"]
    transform = data["joint_transforms"]
    homogeneous = np.column_stack((vertices,np.ones(len(vertices))))
    locations = [(i,l) for i,l in scan.targets if l<2]
    rows, support = [], []
    for p,T in enumerate(transform):
        blend = np.einsum("vj,jkl->vkl",weights,T)
        warped = np.einsum("vkl,vl->vk",blend,homogeneous)[:,:3]
        mesh = TetCoverage(warped,tets)
        for i,label in locations:
            point = scan.trunk[p,i] if label==0 else scan.daughter[p,i]
            tangent = scan.frames[p,i,:,1]
            clip = mesh.covered_intervals(point-.05*tangent,point+.05*tangent)
            intervals = [(100.*a-50.,100.*b-50.) for a,b in clip]
            interval = next((u for u in intervals if u[0]<=0.<=u[1]),None)
            if interval is None:
                raise ValueError(f"posed nominal point outside native tetrahedra: {(p,i,label)}")
            rows.append(scan.row(p,i,label,p in scan.moving))
            support.append(interval)
    return np.stack(rows),np.asarray(support)


def draw_truth(scan, seed, support=None):
    """Draw a coherent patient shape once, before any policy runs."""
    generator = np.random.default_rng(seed)
    if support is None:
        state = generator.multivariate_normal(np.zeros(scan.dim),scan.prior)
        state[-1] = np.clip(generator.normal(0.,1.8),-3.5,3.5)
        return state,0
    H,limits = support
    for rejects in range(512):
        state = generator.multivariate_normal(np.zeros(scan.dim),scan.prior)
        state[-1] = np.clip(generator.normal(0.,1.8),-3.5,3.5)
        moved = H@state
        stable = H[:,:-1]@state[:-1]
        if np.all((moved>=limits[:,0]) & (moved<=limits[:,1]) &
                  (stable>=limits[:,0]) & (stable<=limits[:,1])):
            return state,rejects
    raise RuntimeError("failed to sample a coherent physically supported synthetic vessel")


if __name__=="__main__":
    from dynamic_scan import DynamicScan, SEED
    with np.load(HERE/"DYNAMIC_GEOMETRY_FROZEN.npz") as g:
        scan = DynamicScan(g)
        H,limits = build_support(scan,g)
    reports = []
    for k in range(16):
        seed = int(np.random.SeedSequence([SEED,k]).generate_state(1)[0])
        old,_ = draw_truth(scan,seed)
        state,rejects = draw_truth(scan,seed,(H,limits))
        x = H@old
        reports.append({"seed":seed,"unbounded_pose_nodes_outside":int(((x<limits[:,0]) | (x>limits[:,1])).sum()),
                        "rejects":rejects,
                        "accepted_moving_clearance_mm":float(np.min(np.minimum(H@state-limits[:,0],limits[:,1]-H@state)))})
    path=HERE/"TRUTH_SUPPORT.json"
    path.write_text(json.dumps({"row_count":len(H),"min_safe_offsets_mm":limits.min(axis=0).tolist(),
                                "max_safe_offsets_mm":limits.max(axis=0).tolist(),"seeds":reports},indent=2)+"\n")
    print(path, "affected old draws",sum(r["unbounded_pose_nodes_outside"]>0 for r in reports))
