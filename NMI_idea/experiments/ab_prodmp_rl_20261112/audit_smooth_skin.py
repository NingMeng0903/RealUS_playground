"""Sample the patched references against the original articulated skin mesh."""
from __future__ import annotations

from collections import defaultdict
import json

import igl
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation

from audit_trajectory import DynamicScan, load_rows
from smooth_switch import SmoothAtlas, learned_shapes
from validate_smooth_switch import GEOMETRY, HERE, OUT, SEED, ReferenceRecorder


def build_pose_mesh(geometry, atlas):
    metadata = json.loads(str(geometry["metadata_json"]))
    sources = metadata["source_assets"]
    knee = metadata["pose_schedule"]["knee_joint_index"]
    with np.load(sources["contact_surface"]["path"], allow_pickle=False) as data:
        vertices = data["vertices"].astype(float)
        weights = data["weights"].astype(float)
        faces = data["faces"].astype(np.int32)
    with np.load(sources["smplx_body_weights"]["path"], allow_pickle=False) as data:
        parents, pivot = data["parents"], data["rest_joints"][knee].astype(float)
    descendants = {knee}
    for i, parent in enumerate(parents):
        if int(parent) in descendants:
            descendants.add(i)
    affected = np.array(sorted(descendants))
    share = (weights[:, affected].sum(axis=1) / weights.sum(axis=1))[:, None]
    transform = geometry["robot_from_atlas"]
    for angle, joints in zip(atlas.angles, geometry["joint_transforms"]):
        r = Rotation.from_rotvec([angle, 0., 0.]).as_matrix()
        t = np.eye(4)
        t[:3, :3], t[:3, 3] = r, pivot - r @ pivot
        expected = np.broadcast_to(np.eye(4), joints.shape).copy()
        expected[affected] = t
        np.testing.assert_allclose(joints, expected, atol=1.e-6, rtol=0.)

    def mesh_at(angle):
        r = Rotation.from_rotvec([angle, 0., 0.]).as_matrix()
        v = vertices + share * ((vertices - pivot) @ r.T + pivot - vertices)
        return np.ascontiguousarray(v @ transform[:3, :3].T + transform[:3, 3])

    return mesh_at, faces


def main():
    summary = json.loads((OUT / "SUMMARY.json").read_text())
    slot = summary["clock_slot_s"]
    case = load_rows([HERE / "control_24", HERE / "eval_q_24"], [])[SEED]
    prefix_end = case["rl"]["prefix_steps"][-1]
    with np.load(GEOMETRY, allow_pickle=False) as geometry:
        scan, atlas = DynamicScan(geometry), SmoothAtlas(geometry)
        mesh_at, faces = build_pose_mesh(geometry, atlas)
    shapes = learned_shapes(scan.primitive)
    offsets, ids = np.unique(atlas.offsets, return_index=True)
    linear = RegularGridInterpolator(
        (atlas.angles, atlas.stations, offsets), atlas.targets[..., :3, 3][:, :, ids],
    )
    elapsed = np.linspace(0., slot, 65)
    samples = defaultdict(list)
    for arm in ("rule", "rl"):
        record = ReferenceRecorder(atlas, shapes, scan.moving, slot)
        previous = (prefix_end["pose"], *prefix_end["action"])
        for step in case[arm]["steps"]:
            goal = (step["pose"], *step["action"])
            record.append(previous, goal, step["path_skill"])
            previous = goal
        for k, segment in enumerate(record.segments, 1):
            p = segment.world_jet(elapsed, acceleration=False)[0][:, :3]
            theta = record.clock.sample(segment.time_s + elapsed)[0]
            c = segment.material_jet(elapsed)[0]
            old = linear(np.column_stack((theta, c[:, :2])))
            for phase, angle in enumerate(theta):
                samples[round(float(angle), 12)].append((arm, k, phase, p[phase], old[phase]))
    distances = defaultdict(list)
    for angle, group in samples.items():
        mesh = mesh_at(angle)
        for label, idx in (("c2", 3), ("piecewise_same_coordinate", 4)):
            points = np.stack([s[idx] for s in group])
            squared, _, _ = igl.point_mesh_squared_distance(points, mesh, faces)
            for s, error in zip(group, np.sqrt(squared) * 1000.):
                distances[s[0], label].append((float(error), s[1], s[2]))
    report = {"angle_samples": len(samples), "samples_per_move": len(elapsed), "arms": {}}
    for (arm, label), errors in distances.items():
        worst, action, phase = max(errors)
        report["arms"].setdefault(arm, {})[label] = {
            "samples": len(errors),
            "distance_mm_p95": float(np.percentile(np.asarray(errors)[:, 0], 95)),
            "distance_mm_max": worst,
            "worst_move": action,
            "worst_phase_index": phase,
        }
    text = json.dumps(report, indent=2, allow_nan=False)
    (OUT / "SKIN_SUMMARY.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
