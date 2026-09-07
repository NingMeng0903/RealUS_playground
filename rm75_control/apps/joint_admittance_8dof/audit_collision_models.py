#!/usr/bin/env python3
"""Compare exact STL and capsule collision clearances at recorded poses."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionModel
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics


def audit(paths):
    kin = RobotKinematics()
    root = kin.urdf_path.parent
    models = {name: CollisionModel(kin.model, collision_urdf=root / filename)
              for name, filename in (("mesh", "RM75-6F-8dof.collision.urdf"),
                                     ("capsule", "RM75-6F-8dof.collision.capsule.urdf"))}
    results = []
    for path in paths:
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        indices = {0, len(rows) // 2, len(rows) - 1}
        for field in ("qpik_qp1_solve_ms", "qpik_qp2_solve_ms"):
            def cost(i):
                try:
                    value = float(rows[i][field])
                    return value if np.isfinite(value) else -1.
                except (ValueError, KeyError, TypeError):
                    return -1.
            indices.add(max(range(len(rows)), key=cost))
        for index in sorted(indices):
            q = np.array([float(rows[index][f"q_meas_{i}"]) for i in range(8)])
            if not np.isfinite(q).all():
                continue
            distances = {}
            for name, model in models.items():
                model.update(q)  # All pairs: audit must not approximate far distances.
                distances[name] = {f"{p.name_a}/{p.name_b}": p.distance
                                   for p in model.all_pairs()}
            extras = [{"pair": key, "mesh_mm": distances["mesh"][key] * 1000,
                       "capsule_mm": value * 1000}
                      for key, value in distances["capsule"].items()
                      if value <= .05 and distances["mesh"][key] > .05]
            results.append({"csv": str(path), "data_row": index + 1, "q_meas": q.tolist(),
                            "active_at_50mm": {name: sum(d <= .05 for d in values.values())
                                               for name, values in distances.items()},
                            "extra_capsule_rows": extras,
                            "nearest_pairs": {name: [{"pair": k, "distance_mm": v * 1000}
                                for k, v in sorted(values.items(), key=lambda item: item[1])[:8]]
                                for name, values in distances.items()}})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.csv), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
