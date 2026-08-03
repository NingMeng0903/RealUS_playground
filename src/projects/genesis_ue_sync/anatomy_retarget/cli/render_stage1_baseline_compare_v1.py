"""Stage-1 baseline compare packs: 31133af/142, node2_004, and V4 failure.

Read-only Genesis reviewer. Does not modify retarget solvers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    JOINT_SPECS,
    _measure_frames,
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import skin_vertices
from projects.genesis_ue_sync.anatomy_retarget.cli.render_alignment_truth_genesis_v1 import (
    COLORS as TRUTH_COLORS,
    _baseline_pose,
    _faces,
    _mask,
    _outside_faces,
    _render_layer,
)
from projects.genesis_ue_sync.anatomy_retarget.cli.render_chain_rest_fit_genesis_v1 import (
    _contact_sheet,
    _export,
    _sha256,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import (
    easymocap_fit_to_smplx55,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_map_v1 import (
    build_pose_map_v1,
    pose_whole_chain_vertices,
)
from projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    load_smplx_model_v7,
    require_frozen_smplx_male_v7,
    smplx_body_surface_v7,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
    materialize_subject,
)
from projects.genesis_ue_sync.anatomy_retarget.whole_chain_rest_fit_v1 import (
    load_whole_chain_rest_fit_v1,
)


COLORS = {
    **TRUTH_COLORS,
    "bones": (0.93, 0.88, 0.72, 1.0),
    "bones_translucent": (0.93, 0.88, 0.72, 0.35),
    "outside": (1.0, 0.05, 0.02, 1.0),
    "near": (1.0, 0.45, 0.05, 0.95),
    "organs": (0.72, 0.22, 0.20, 0.55),
    "heart": (0.86, 0.12, 0.18, 0.70),
    "connective": (0.34, 0.78, 0.58, 0.35),
}


def _camera_manifest(skin: np.ndarray, frames: np.ndarray) -> dict[str, dict[str, Any]]:
    lookup = {spec.name: index for index, spec in enumerate(JOINT_SPECS)}
    lower = np.min(skin, axis=0)
    upper = np.max(skin, axis=0)
    center = 0.5 * (lower + upper)
    extent = float(np.max(upper - lower))
    distance = max(2.2, 1.35 * extent)
    cameras: dict[str, dict[str, Any]] = {
        "whole_ap": {
            "pos": tuple((center + (0.0, 0.0, distance)).tolist()),
            "lookat": tuple(center.tolist()),
            "up": (0.0, 1.0, 0.0),
            "fov": 34.0,
        },
        "whole_pa": {
            "pos": tuple((center + (0.0, 0.0, -distance)).tolist()),
            "lookat": tuple(center.tolist()),
            "up": (0.0, 1.0, 0.0),
            "fov": 34.0,
        },
    }
    for side, sign in (("left", 1.0), ("right", -1.0)):
        specs = (
            ("hip", "ap", (0.0, 0.0, 0.42), (0.0, 1.0, 0.0)),
            ("knee", "ap", (0.0, 0.0, 0.42), (0.0, 1.0, 0.0)),
            ("knee", "lateral", (0.42 * sign, 0.0, 0.0), (0.0, 1.0, 0.0)),
            ("ankle", "ap", (0.0, 0.0, 0.42), (0.0, 1.0, 0.0)),
            ("ankle", "oblique", (0.28 * sign, 0.08, 0.32), (0.0, 1.0, 0.0)),
            ("shoulder", "ap", (0.0, 0.0, 0.42), (0.0, 1.0, 0.0)),
            ("elbow", "ap", (0.0, 0.0, 0.42), (0.0, 1.0, 0.0)),
            ("elbow", "lateral", (0.42 * sign, 0.0, 0.0), (0.0, 1.0, 0.0)),
            ("wrist", "ap", (0.0, 0.0, 0.36), (0.0, 1.0, 0.0)),
            ("wrist", "oblique", (0.24 * sign, 0.10, 0.28), (0.0, 1.0, 0.0)),
        )
        for joint, view, offset, up in specs:
            pivot = frames[lookup[f"{side}_{joint}"], :3, 3]
            cameras[f"{side}_{joint}_{view}"] = {
                "pos": tuple((pivot + np.asarray(offset, dtype=np.float64)).tolist()),
                "lookat": tuple(pivot.tolist()),
                "up": up,
                "fov": 30.0,
            }
        wrist = frames[lookup[f"{side}_wrist"], :3, 3]
        elbow = frames[lookup[f"{side}_elbow"], :3, 3]
        distal = wrist - elbow
        norm = float(np.linalg.norm(distal))
        hand = wrist + (0.08 * distal / norm if norm > 1e-8 else np.zeros(3))
        ankle = frames[lookup[f"{side}_ankle"], :3, 3]
        cameras[f"{side}_hand_oblique"] = {
            "pos": tuple((hand + (0.18 * sign, 0.12, 0.28)).tolist()),
            "lookat": tuple(hand.tolist()),
            "up": (0.0, 1.0, 0.0),
            "fov": 30.0,
        }
        cameras[f"{side}_foot_oblique"] = {
            "pos": tuple((ankle + (0.18 * sign, -0.05, 0.30)).tolist()),
            "lookat": tuple(ankle.tolist()),
            "up": (0.0, 1.0, 0.0),
            "fov": 30.0,
        }
    return cameras


def _pose_pack_a(asset: Any, pose: np.ndarray) -> np.ndarray:
    return np.asarray(skin_vertices(asset, pose), dtype=np.float32)


def _load_v4_subject(path: Path) -> Any:
    from projects.genesis_ue_sync.anatomy_retarget._quarantine_v4.dynamic_main_chain_retarget_v4 import (
        DynamicMainChainSubjectV4,
    )

    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("artifact_kind") != "DynamicMainChainSubjectV4":
        raise ValueError(f"expected DynamicMainChainSubjectV4 at {path}")
    npz_name = str(manifest.get("npz", "dynamic_main_chain_subject_v4.npz"))
    with np.load(path / npz_name, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]).copy() for key in data.files}
    provenance = dict(manifest.get("provenance", {}))
    build_report = dict(manifest.get("build_report", {}))
    return DynamicMainChainSubjectV4(
        source_operator_digest=str(
            provenance.get(
                "source_operator_digest",
                build_report.get("source_operator_digest", "render-only"),
            )
        ),
        calibration_digest=str(
            provenance.get(
                "calibration_digest",
                build_report.get("calibration_digest", "render-only"),
            )
        ),
        source_subject_digest=str(
            provenance.get(
                "source_subject_digest",
                build_report.get("source_subject_digest", "render-only"),
            )
        ),
        smplx_model_sha256=str(manifest["smplx_model_sha256"]),
        capture_sha256=str(
            provenance.get(
                "capture_sha256",
                (
                    provenance.get("capture_sha256s", {}) or {}
                ).get(str(manifest["subject_label"]), "")
                or build_report.get("capture_sha256", "render-only"),
            )
        ),
        subject_label=str(manifest["subject_label"]),
        betas=np.asarray(arrays["betas"], dtype=np.float64),
        vertices_prefit=np.asarray(arrays["vertices_prefit"], dtype=np.float32),
        vertices_final=np.asarray(arrays["vertices_final"], dtype=np.float32),
        faces=np.asarray(arrays["faces"], dtype=np.int32),
        bone_parents=np.asarray(arrays["bone_parents"], dtype=np.int32),
        B_prefit=np.asarray(arrays["B_prefit"], dtype=np.float64),
        B_final=np.asarray(arrays["B_final"], dtype=np.float64),
        C_bone=np.asarray(arrays["C_bone"], dtype=np.float64),
        target_local_bind=np.asarray(arrays["target_local_bind"], dtype=np.float64),
        inverse_bind=np.asarray(arrays["inverse_bind"], dtype=np.float64),
        prefit_anatomical_frames=np.asarray(
            arrays["prefit_anatomical_frames"], dtype=np.float64
        ),
        final_anatomical_frames=np.asarray(
            arrays["final_anatomical_frames"], dtype=np.float64
        ),
        smplx_joints_tpose=np.asarray(arrays["smplx_joints_tpose"], dtype=np.float64),
        station_frame_translation=np.asarray(
            arrays["station_frame_translation"], dtype=np.float64
        ),
        centerline_points=np.asarray(arrays["centerline_points"], dtype=np.float64),
        mesh_policy=np.asarray(arrays["mesh_policy"]).copy(),
        moved_vertex_ids=np.asarray(arrays["moved_vertex_ids"], dtype=np.int32),
        build_report=build_report,
        pelvis_cage_vertex_ids=np.asarray(
            arrays.get("pelvis_cage_vertex_ids", np.zeros(0, dtype=np.int32)),
            dtype=np.int32,
        ),
        pelvis_cage_displacements=np.asarray(
            arrays.get(
                "pelvis_cage_displacements", np.zeros((0, 3), dtype=np.float64)
            ),
            dtype=np.float64,
        ),
        C_total=np.asarray(arrays["C_total"], dtype=np.float64),
        target_anatomical_rest_frames=np.asarray(
            arrays["target_anatomical_rest_frames"], dtype=np.float64
        ),
        target_station_from_anatomical=np.asarray(
            arrays["target_station_from_anatomical"], dtype=np.float64
        ),
        controller_pivot_local=np.asarray(
            arrays["controller_pivot_local"], dtype=np.float64
        ),
        controller_axis_local=np.asarray(
            arrays["controller_axis_local"], dtype=np.float64
        ),
        channel_basis_controller_indices=np.asarray(
            arrays["channel_basis_controller_indices"], dtype=np.int64
        ),
        channel_basis_change=np.asarray(arrays["channel_basis_change"], dtype=np.float64),
        main_chain_controller_mask=np.asarray(
            arrays["main_chain_controller_mask"], dtype=bool
        ),
        validation_pose_labels=np.asarray(arrays["validation_pose_labels"]).copy(),
        validation_pose_axis_angle=np.asarray(
            arrays["validation_pose_axis_angle"], dtype=np.float64
        ),
    )


def _render_modes(
    *,
    output: Path,
    vertices: np.ndarray,
    asset: Any,
    skin: np.ndarray,
    skin_faces: np.ndarray,
    frames: np.ndarray,
    backend: str,
    camera_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    bone_mask = _mask(asset, {"bone"})
    vessel_mask = _mask(asset, {"vessel"})
    nerve_mask = _mask(asset, {"nerve"})
    organ_mask = _mask(asset, {"organ"})
    heart_mask = _mask(asset, {"heart"})
    connective_mask = _mask(asset, {"connective_tissue"})
    bone_faces = _faces(asset.faces, bone_mask)
    vessel_faces = _faces(asset.faces, vessel_mask)
    nerve_faces = _faces(asset.faces, nerve_mask)
    organ_faces = _faces(asset.faces, organ_mask)
    heart_faces = _faces(asset.faces, heart_mask)
    connective_faces = _faces(asset.faces, connective_mask)
    outside_faces = _outside_faces(
        vertices, asset.faces, bone_mask, skin, skin_faces
    )
    cameras = _camera_manifest(skin, frames)
    if camera_names is not None:
        missing = [name for name in camera_names if name not in cameras]
        if missing:
            raise KeyError(f"requested cameras missing from manifest: {missing}")
        cameras = {name: cameras[name] for name in camera_names}
    assets = output / "mesh_assets"
    skin_path = _export(assets / "skin.obj", skin, skin_faces)
    bones_path = _export(assets / "bones.obj", vertices, bone_faces)
    outside_path = _export(assets / "outside.obj", vertices, outside_faces)
    vessel_path = _export(assets / "vessels.obj", vertices, vessel_faces)
    nerve_path = _export(assets / "nerves.obj", vertices, nerve_faces)

    def _maybe_export(name: str, faces: np.ndarray) -> Path | None:
        if len(np.asarray(faces)) == 0:
            return None
        return _export(assets / f"{name}.obj", vertices, faces)

    organ_path = _maybe_export("organs", organ_faces)
    heart_path = _maybe_export("heart", heart_faces)
    connective_path = _maybe_export("connective", connective_faces)

    def _entities(*items: tuple[str, Path | None, tuple[float, ...]]) -> list:
        return [(name, path, color) for name, path, color in items if path is not None]

    layers = {
        "bones_only": [
            ("skin", skin_path, COLORS["skin"]),
            ("bones", bones_path, COLORS["bones"]),
        ],
        "outside_heatmap": [
            ("skin", skin_path, (0.90, 0.58, 0.43, 0.10)),
            ("bones", bones_path, COLORS["bones_translucent"]),
            ("outside", outside_path, COLORS["outside"]),
        ],
        "bones_tubes": [
            ("skin", skin_path, (0.90, 0.58, 0.43, 0.08)),
            ("bones", bones_path, COLORS["bones"]),
            ("vessels", vessel_path, COLORS["vessels"]),
            ("nerves", nerve_path, COLORS["nerves"]),
        ],
        "full_anatomy": _entities(
            ("skin", skin_path, (0.90, 0.58, 0.43, 0.06)),
            ("bones", bones_path, COLORS["bones"]),
            ("organs", organ_path, COLORS["organs"]),
            ("heart", heart_path, COLORS["heart"]),
            ("connective", connective_path, COLORS["connective"]),
            ("vessels", vessel_path, COLORS["vessels"]),
            ("nerves", nerve_path, COLORS["nerves"]),
        ),
    }
    report: dict[str, Any] = {
        "camera_manifest": cameras,
        "camera_source": "smplx_skin_bbox_and_frozen_142_validation_frames",
        "candidate_camera_read": False,
        "layers": {},
    }
    for layer_name, entities in layers.items():
        layer_out = output / layer_name
        report["layers"][layer_name] = _render_layer(
            layer_out,
            entities=entities,
            cameras=cameras,
            backend=backend,
        )
    sheet_paths = []
    sheet_labels = []
    for layer_name in (
        "bones_only",
        "outside_heatmap",
        "bones_tubes",
        "full_anatomy",
    ):
        sheet = Path(report["layers"][layer_name]["contact_sheet"])
        sheet_paths.append(sheet)
        sheet_labels.append(layer_name)
    summary = output / "three_layer_contact_sheet.png"
    _contact_sheet(sheet_paths, sheet_labels, summary, {})
    report["three_layer_contact_sheet"] = str(summary)
    report["three_layer_contact_sheet_sha256"] = _sha256(summary)
    report["review_layer_names"] = list(layers.keys())
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", choices=("A", "B", "C", "all"), default="all")
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="Full-main-chain calibration for Pack A/B (node1_005).",
    )
    parser.add_argument(
        "--calibration-v4",
        type=Path,
        default=None,
        help="Calibration used to build V4 (defaults to --calibration).",
    )
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--smplx-model", type=Path, required=True)
    parser.add_argument("--capture-213328", type=Path, required=True)
    parser.add_argument("--capture-213712", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--node2-004-root",
        type=Path,
        default=Path(
            "outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node2_004"
        ),
    )
    parser.add_argument(
        "--v4-root",
        type=Path,
        default=Path(
            "outputs/anatomy_retarget/v8_candidates/chain_retarget_v4_node2_031_root"
        ),
    )
    parser.add_argument(
        "--v4-debug",
        type=Path,
        default=Path(
            "outputs/anatomy_retarget/v8_candidates/chain_retarget_v4_debug_node4_001"
        ),
    )
    parser.add_argument("--backend", default="cuda")
    parser.add_argument(
        "--subjects",
        default="213328,213712",
        help="Comma-separated subject labels.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite compare root: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    operator = load_source_operator(args.operator.resolve(), mmap=True)
    calibration_ab = load_anatomical_calibration_v1(
        args.calibration.resolve(),
        operator=operator,
        # node1_005 may predate local domain-fraction edits; render-only load.
        require_complete=False,
        required_scope="full_main_chain",
    )
    calibration_v4 = load_anatomical_calibration_v1(
        (args.calibration_v4 or args.calibration).resolve(),
        operator=operator,
        require_complete=True,
        required_scope="full_main_chain",
    )
    model_path, model_sha = require_frozen_smplx_male_v7(args.smplx_model)
    model = load_smplx_model_v7(model_path)
    capture_paths = {
        "213328": args.capture_213328.resolve(),
        "213712": args.capture_213712.resolve(),
    }
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    packs = ["A", "B", "C"] if args.pack == "all" else [args.pack]
    root_manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Stage1BaselineCompareV1",
        "branch_baseline_commit": "31133afba2ced3f4de01df7328d487859c7f9b05",
        "anatomy_baseline_commit": "142ece5f0bc646978ae3e8c9add76deea71c26a2",
        "smplx_gender": "male",
        "smplx_model_sha256": model_sha,
        "publishable": False,
        "trusted_latest_updated": False,
        "vessel_repair_started": False,
        "candidate_camera_read": False,
        "packs": {},
    }

    poses_by_subject: dict[str, dict[str, np.ndarray]] = {}
    betas_by_subject: dict[str, np.ndarray] = {}
    for label in subjects:
        with np.load(capture_paths[label], allow_pickle=False) as data:
            betas_by_subject[label] = np.asarray(data["shapes"]).reshape(-1)[:10]
            poses_by_subject[label] = {
                "tpose": np.zeros((55, 3), dtype=np.float32),
                f"pose_{label}": easymocap_fit_to_smplx55(
                    data["Rh"], data["poses"], model_path=model_path
                ),
                # Cross pose for the other capture when both subjects requested.
            }
    # Attach cross poses when both captures available.
    if set(subjects) >= {"213328", "213712"}:
        for label in subjects:
            other = "213712" if label == "213328" else "213328"
            with np.load(capture_paths[other], allow_pickle=False) as data:
                poses_by_subject[label][f"pose_{other}"] = easymocap_fit_to_smplx55(
                    data["Rh"], data["poses"], model_path=model_path
                )

    for pack in packs:
        pack_dir = output / f"pack_{pack}"
        pack_dir.mkdir(parents=True, exist_ok=False)
        pack_report: dict[str, Any] = {
            "pack": pack,
            "subjects": {},
            "publishable": False,
        }
        if pack == "A":
            pack_report["identity"] = "31133af_branch_start_142_materialize_baseline"
        elif pack == "B":
            pack_report["identity"] = "chain_retarget_v1_node2_004_whole_chain"
        else:
            pack_report["identity"] = "chain_retarget_v4_node2_031_root_failure"
            if args.v4_debug.resolve().exists():
                linked = pack_dir / "reused_v4_debug_outside"
                if not linked.exists():
                    shutil.copytree(
                        args.v4_debug.resolve(),
                        linked,
                        dirs_exist_ok=False,
                    )
                pack_report["reused_v4_debug_outside"] = str(linked)

        for label in subjects:
            betas = betas_by_subject[label]
            asset = materialize_subject(
                operator, betas=betas, gender="male"
            ).rigged_asset
            capture_sha = hashlib.sha256(
                capture_paths[label].read_bytes()
            ).hexdigest()
            subject_dir = pack_dir / f"subject_{label}"
            subject_dir.mkdir(parents=True, exist_ok=False)
            subject_report: dict[str, Any] = {"poses": {}}

            value = None
            pose_map = None
            v4_value = None
            calibration = calibration_v4 if pack == "C" else calibration_ab
            if pack == "B":
                value = load_whole_chain_rest_fit_v1(
                    args.node2_004_root.resolve() / f"subject_{label}",
                    operator=operator,
                    calibration=calibration,
                    smplx_model=model,
                    smplx_model_sha256=model_sha,
                    recheck=False,
                )
                pose_map = build_pose_map_v1(
                    value,
                    asset=asset,
                    calibration=calibration,
                    oracle_path=args.oracle.resolve(),
                    source_operator_digest=operator.runtime_digest(validate=False),
                )
            elif pack == "C":
                from projects.genesis_ue_sync.anatomy_retarget._quarantine_v4.dynamic_main_chain_retarget_v4 import (
                    pose_dynamic_main_chain_vertices_v4,
                )

                v4_value = _load_v4_subject(
                    args.v4_root.resolve() / f"subject_{label}"
                )

            for pose_name, pose in poses_by_subject[label].items():
                if pack == "A":
                    vertices = _pose_pack_a(asset, pose)
                    # Frames from posed baseline bones for camera look-ats.
                    frame_source = vertices
                elif pack == "B":
                    assert value is not None and pose_map is not None
                    vertices, _ = pose_whole_chain_vertices(
                        value,
                        pose_map,
                        source_asset=asset,
                        pose_axis_angle=pose,
                    )
                    frame_source = _baseline_pose(value, asset, pose)
                else:
                    assert v4_value is not None
                    vertices, _ = pose_dynamic_main_chain_vertices_v4(
                        v4_value,
                        asset=asset,
                        calibration=calibration,
                        smplx_model=model,
                        pose_axis_angle=pose,
                    )
                    frame_source = vertices

                skin, skin_faces = smplx_body_surface_v7(
                    model, betas=betas, pose_axis_angle=pose
                )
                frames, _widths, _details = _measure_frames(
                    frame_source,
                    calibration.domains,
                    calibration.joint_domain_bases,
                    partition="validation",
                )
                pose_out = subject_dir / pose_name
                subject_report["poses"][pose_name] = _render_modes(
                    output=pose_out,
                    vertices=np.asarray(vertices, dtype=np.float32),
                    asset=asset,
                    skin=skin,
                    skin_faces=skin_faces,
                    frames=frames,
                    backend=args.backend,
                )
                subject_report["poses"][pose_name]["capture_sha256"] = capture_sha
            pack_report["subjects"][label] = subject_report
        pack_path = pack_dir / "manifest.json"
        pack_path.write_text(
            json.dumps(pack_report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        root_manifest["packs"][pack] = {
            "path": str(pack_dir),
            "manifest_sha256": _sha256(pack_path),
            "identity": pack_report["identity"],
        }

    root_manifest["elapsed_seconds"] = float(time.perf_counter() - started)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(root_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Stage1BaselineCompareV1 -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
