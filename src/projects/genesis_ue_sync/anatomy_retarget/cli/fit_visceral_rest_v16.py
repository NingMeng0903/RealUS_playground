"""Fit the bounded shared visceral rest field on a 213328 T-pose.

The command writes a small measurement report and a V14 compiled package with
the final field already baked into ``target_rest``.  The supplied T-pose is
used for its real SMPL-X skin boundary; no pose data is used to fit the field.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ..consistent_runtime_v14 import load_compiled_subject
from ..shared_visceral_rest_fit_v16 import (
    DEFAULT_PAIRS,
    run_shared_rest_fit_v16,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_tpose(path: Path, base) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {"pose", "transl", "before_vertices", "skin_vertices", "skin_faces"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"T-pose geometry is missing keys: {sorted(missing)}")
        pose = np.asarray(data["pose"], dtype=np.float64)
        transl = np.asarray(data["transl"], dtype=np.float64)
        if pose.shape != (55, 3) or transl.shape != (3,):
            raise ValueError("T-pose pose/transl have invalid shapes")
        if np.max(np.abs(pose), initial=0.0) > 1.0e-7:
            raise ValueError("fit input must be an actual zero-pose T-pose")
        if np.max(np.abs(transl), initial=0.0) > 1.0e-7:
            raise ValueError("fit input must have zero translation in target-rest coordinates")
        before = np.asarray(data["before_vertices"], dtype=np.float64)
        if before.shape != np.asarray(base.target_rest).shape:
            raise ValueError("T-pose vertices do not match compiled source vertex count")
        mismatch = float(np.max(np.abs(before - np.asarray(base.target_rest, dtype=np.float64)), initial=0.0))
        if mismatch > 2.0e-6:
            raise ValueError(
                "T-pose vertices are not the latest V15 compiled replay "
                f"(max mismatch {mismatch:.3g} m)"
            )
        skin_vertices = np.asarray(data["skin_vertices"], dtype=np.float64)
        skin_faces = np.asarray(data["skin_faces"], dtype=np.int32)
    if skin_vertices.ndim != 2 or skin_vertices.shape[1] != 3 or len(skin_vertices) == 0:
        raise ValueError("T-pose skin_vertices must be a non-empty [N, 3] array")
    if skin_faces.ndim != 2 or skin_faces.shape[1] != 3 or len(skin_faces) == 0:
        raise ValueError("T-pose skin_faces must be a non-empty [M, 3] array")
    if np.any(skin_faces < 0) or np.any(skin_faces >= len(skin_vertices)):
        raise ValueError("T-pose skin_faces contain an out-of-range vertex")
    return skin_vertices, skin_faces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--tpose-geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--support-mm", type=float, default=50.0)
    parser.add_argument("--guard-neighbors", action="store_true",
                        help="Constrain neighbouring soft/bone clearance and cumulative rest edge strain")
    args = parser.parse_args()
    support_m = float(args.support_mm) / 1000.0
    if not 40.0 <= float(args.support_mm) <= 60.0:
        raise ValueError("support-mm must be between 40 and 60")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    print("loading latest V15 compiled package", flush=True)
    base = load_compiled_subject(args.compiled)
    skin_vertices, skin_faces = _load_tpose(args.tpose_geometry, base)
    print("fitting T-pose Diaphragm/L2 and Liver/Rib_10R shared field", flush=True)
    field_refiner = None
    if args.guard_neighbors:
        from ..guarded_visceral_field_v16 import fit_guarded_field
        field_refiner = fit_guarded_field
    result = run_shared_rest_fit_v16(
        base,
        skin_vertices=skin_vertices,
        skin_faces=skin_faces,
        pairs=DEFAULT_PAIRS,
        rounds=int(args.rounds),
        support_radius_m=support_m,
        field_refiner=field_refiner,
    )
    report = result["report"]
    field = result["field"]
    compiled = result["compiled"]
    # The final RBF object is only the last accepted increment.  The runtime
    # package uses the fully baked target_rest below; keep that total delta in
    # a separate file so replay tooling cannot mistake one increment for the
    # accumulated material correction.
    field_path = args.output / "field_last_increment.npz"
    field.save_npz(field_path)
    total_delta_path = args.output / "total_material_displacement.npz"
    vertex_ids = np.arange(len(base.target_rest), dtype=np.int32)
    total_delta = np.asarray(compiled.target_rest, dtype=np.float64) - np.asarray(base.target_rest, dtype=np.float64)
    np.savez_compressed(
        total_delta_path,
        vertex_ids=vertex_ids,
        displacement_m=total_delta,
        base_target_rest=np.asarray(base.target_rest, dtype=np.float64),
    )

    # Dataclasses.replace has already rebuilt the same V14 LBS evaluator.  A
    # saved reload must preserve this exact target-rest candidate and all
    # source identities before it is made available for rendering.
    compiled_path = args.output / "compiled"
    compiled.save(compiled_path)
    loaded = load_compiled_subject(compiled_path)
    if not np.array_equal(loaded.target_rest, compiled.target_rest):
        raise AssertionError("saved compiled target_rest is not bit exact")
    if not np.array_equal(base.indices, loaded.indices):
        raise AssertionError("saved package changed sparse driver indices")
    if not np.array_equal(base.weights, loaded.weights):
        raise AssertionError("saved package changed sparse driver weights")
    if not np.array_equal(base.source_asset.faces, loaded.source_asset.faces):
        raise AssertionError("saved package changed source topology")
    if not np.array_equal(base.target_bind, loaded.target_bind):
        raise AssertionError("saved package changed target bind")
    with np.load(args.tpose_geometry, allow_pickle=False) as data:
        replay = loaded.apply_pose(data["pose"], data["transl"])
    expected = np.asarray(compiled.target_rest, dtype=np.float32)
    if not np.array_equal(replay, expected):
        raise AssertionError("saved package T-pose replay differs from baked target_rest")
    report.update(
        {
            "base_compiled": str(args.compiled.resolve()),
            "tpose_geometry": str(args.tpose_geometry.resolve()),
            "tpose_geometry_sha256": _sha256(args.tpose_geometry),
            "subject_id": args.tpose_geometry.stem.split("_", 1)[0],
            "output_compiled": str(compiled_path.resolve()),
            "field_file": str(field_path.resolve()),
            "field_serialization": "last_increment_only",
            "total_material_displacement_file": str(total_delta_path.resolve()),
            "total_material_displacement_sha256": _sha256(total_delta_path),
            "field_sha256": _sha256(field_path),
            "compiled_manifest_sha256": _sha256(compiled_path / "manifest.json"),
            "original_weights_bit_exact": bool(np.array_equal(base.weights, loaded.weights)),
            "original_indices_bit_exact": bool(np.array_equal(base.indices, loaded.indices)),
            "original_faces_bit_exact": bool(np.array_equal(base.source_asset.faces, loaded.source_asset.faces)),
            "bind_bit_exact": bool(np.array_equal(base.target_bind, loaded.target_bind)),
            "target_rest_changed": bool(not np.array_equal(base.target_rest, loaded.target_rest)),
            "runtime_nearest_search": False,
            "runtime_optimization": False,
            "betas": np.asarray(base.betas, dtype=np.float64).tolist(),
        }
    )
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"completed {len(report['rounds'])} round(s)", flush=True)
    for row in report["rounds"]:
        pairs = row["pair_collision"]["pairs"]
        print(
            f"round {row['round']}: "
            f"Diaphragm/L2 org-in-bone {pairs['Diaphragm__L2']['organ_inside_bone_max_depth_mm']:.3f} mm, "
            f"Liver/Rib_10R org-in-bone {pairs['Liver__Rib_10R']['organ_inside_bone_max_depth_mm']:.3f} mm, "
            f"edge symmetric max {row['edge_strain']['symmetric_max_pct']:.2f}%",
            flush=True,
        )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
