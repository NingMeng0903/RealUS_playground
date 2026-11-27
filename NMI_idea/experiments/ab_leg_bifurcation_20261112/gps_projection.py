"""Project the Y distribution to leg skin views with real dIRD and static IK."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import sys

import numpy as np

from bifurcation_model import FORK, JUNCTION, LABELS, LOWER, TRUNK, UPPER, acquisition_probability, target_points


HERE = Path(__file__).resolve().parent
CAPABILITY = HERE.parent / "ab_module_ladder_20261110" / "capability"
PLACEMENT = (CAPABILITY.parent / "planning" / "native_regression_20261110"
             / "public_prior_layout" / "SELECTED.json")
LATERAL_GRID_MM = np.arange(-32.0, 32.1, 4.0)


@dataclass(frozen=True)
class Section:
    label: int
    x: float

    @property
    def name(self):
        return f"{LABELS[self.label]}_{self.x:+g}"


TRUNK_SECTIONS = tuple(Section(TRUNK, x) for x in (-36., -28., -20., -12., -6.))
JUNCTION_SECTIONS = tuple(Section(JUNCTION, x) for x in (-2., 4.))
FORK_SECTION = Section(FORK, 10.)
LOWER_SECTIONS = tuple(Section(LOWER, x) for x in (14., 20., 26., 32., 36.))
UPPER_SECTIONS = tuple(Section(UPPER, x) for x in (14., 20., 26., 32., 36.))
SECTIONS = (*TRUNK_SECTIONS, *JUNCTION_SECTIONS, FORK_SECTION, *LOWER_SECTIONS, *UPPER_SECTIONS)
STATIONS = np.array(sorted({section.x for section in SECTIONS}))


def chart_tcp(chart, xy):
    """Surface TCP position and right-handed longitudinal/inward frame."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    frame = chart.frame(xy)
    first, inward = frame[:, :, 0], frame[:, :, 2]
    side = np.cross(inward, first)
    out = np.broadcast_to(np.eye(4), (len(xy), 4, 4)).copy()
    out[:, :3, :3] = np.stack((first, side, inward), axis=-1)
    out[:, :3, 3] = chart.world(np.column_stack((xy, np.zeros(len(xy))))) / 1000.0
    return out


def load_capability():
    sys.path.insert(0, str(CAPABILITY))
    from capability import CapabilityEvaluator

    return CapabilityEvaluator()


def build_robot_table(charts, output):
    """Freeze robot evidence on a public grid before drawing any anatomy."""
    cap = load_capability()
    placement = json.loads(PLACEMENT.read_text())
    q = np.asarray(placement["q_seed"])
    anchor = np.asarray(placement["T_world_body"])
    xy = np.stack(np.meshgrid(STATIONS, LATERAL_GRID_MM, indexing="ij"), axis=-1).reshape(-1, 2)
    rows = []
    for pose_deg, chart in charts.items():
        # Each static leg pose is placed at the same previously audited TCP
        # anchor. All positions/normals away from the anchor use the posed mesh.
        robot_from_atlas = anchor @ np.linalg.inv(chart_tcp(chart, [[0., 0.]])[0])
        poses = robot_from_atlas[None] @ chart_tcp(chart, xy)
        scores = cap.score(poses, q[0])
        candidates = []
        for i, (view, target) in enumerate(zip(xy, poses)):
            witness = cap._static_pose(target, float(q[0]), q)
            candidates.append({
                "xy_mm": view.tolist(), "pose_robot": target.tolist(),
                "dird_margin": scores["raw_calibrated_margin"][i],
                "dird_proxy_pass": scores["proxy_pass"][i],
                "static_found": witness["found"], "q_static": witness.get("q_solution"),
                "static_check": witness.get("check"),
            })
        rows.append({"pose_deg": pose_deg, "robot_from_atlas": robot_from_atlas.tolist(),
                     "candidates": candidates})
        print(f"pose {pose_deg}: {sum(r['static_found'] for r in candidates)}/{len(candidates)} static poses", flush=True)
    result = {"scope": "dIRD ranking and finite static endpoint IK with configured robot self-collision."
                        " No body collisions, native execution, or path feasibility claim.",
              "q_seed": q.tolist(), "placement_source": str(PLACEMENT),
              "source": cap.manifest(), "poses": rows}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


@dataclass(frozen=True)
class ViewBank:
    xy: np.ndarray
    capture: np.ndarray
    distance_mm: np.ndarray
    margin: np.ndarray

    def scores(self, weights):
        """Rank visible probability mass, nearby skin, and dIRD margin."""
        visible = self.capture @ weights
        distance = self.distance_mm @ weights
        dird_penalty = np.where(self.margin >= 0., 1. / (1. + np.maximum(self.margin, 0.)),
                                1. + np.minimum(np.maximum(-self.margin, 0.), 5.))
        return visible - 0.0025 * distance - 0.012 * dird_penalty, visible, distance


def make_banks(geometry, theta, robot_pose_row):
    """Cache public sensor/geometric projections, never posterior observations."""
    banks = {}
    for section in SECTIONS:
        rows = [r for r in robot_pose_row["candidates"]
                if r["static_found"] and r["xy_mm"][0] == section.x]
        if not rows:
            raise RuntimeError(f"no static witness found for {section.name}")
        xy = np.array([r["xy_mm"] for r in rows])
        probability = np.array([
            acquisition_probability(*geometry.predict(theta, view)[1:], section.label)
            for view in xy
        ])
        surface = geometry.skin_world(xy)
        internal = geometry.body_world(target_points(theta, section.x, section.label))
        distance = np.linalg.norm(surface[:, None] - internal[None], axis=-1)
        banks[section] = ViewBank(xy, probability, distance,
                                  np.array([r["dird_margin"] for r in rows]))
    return banks


def projected_gps(banks, weights, sections=SECTIONS):
    result = []
    for section in sections:
        bank = banks[section]
        scores, coverage, distance = bank.scores(weights)
        i = int(np.argmax(scores))
        result.append({"section": section.name, "xy_mm": bank.xy[i].tolist(),
                       "predicted_capture": float(coverage[i]),
                       "expected_skin_to_internal_distance_mm": float(distance[i]),
                       "dird_margin": float(bank.margin[i])})
    return result
