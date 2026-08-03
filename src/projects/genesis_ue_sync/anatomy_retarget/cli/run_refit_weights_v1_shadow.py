"""Shadow V10 weight-refit: inherit Blender 235 linkage, refit knee LBS to SMPL-X.

Candidate-only. Does not update trusted/latest. Renders slim multipose review.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    _measure_frames,
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.chain_containment_v1 import (
    _signed_distance,
)
from projects.genesis_ue_sync.anatomy_retarget.cli.render_stage1_baseline_compare_v1 import (
    _render_modes,
)
from projects.genesis_ue_sync.anatomy_retarget.cli.run_amass_bedlam_retarget_matrix_v6 import (
    _pick_frame,
    _resolve_bedlam_file,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import easymocap_fit_to_smplx55
from projects.genesis_ue_sync.anatomy_retarget.pose_map_v1 import (
    build_pose_map_v1,
    pose_whole_chain_vertices,
)
from projects.genesis_ue_sync.anatomy_retarget.refit_weights_v1 import (
    build_weight_refit_v1,
    save_weight_refit_v1,
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


CAMS = (
    "left_knee_ap",
    "left_knee_lateral",
    "right_knee_ap",
    "left_elbow_ap",
    "whole_ap",
)
LAYERS = ("bones_only", "outside_heatmap", "full_anatomy")
KNEE_MESHES = ("Femur_L", "Patella_L", "Femur_R", "Patella_R")


def _mesh_ids(asset, mesh_name: str) -> np.ndarray:
    names = [str(x) for x in asset.source_mesh_names]
    start, stop = np.asarray(asset.source_vertex_ranges, dtype=np.int64)[names.index(mesh_name)]
    return np.arange(int(start), int(stop), dtype=np.int64)


def _femur_patella_outside(
    verts: np.ndarray,
    asset,
    skin: np.ndarray,
    faces: np.ndarray,
) -> dict[str, float]:
    signed = _signed_distance(verts, skin, faces)
    out: dict[str, float] = {}
    worst = 0.0
    for mesh in KNEE_MESHES:
        ids = _mesh_ids(asset, mesh)
        max_out = float(max(0.0, float(np.max(signed[ids]))))
        out[mesh] = max_out
        worst = max(worst, max_out)
    out["femur_patella_max_outside_m"] = worst
    return out


def _load_poses(model_path: Path, root: Path) -> dict[str, np.ndarray]:
    cap328 = root / "smplx_outputs/20260713_213328/moment_0000/smplx_result.npz"
    cap712 = root / "smplx_outputs/20260713_213712/moment_0000/smplx_result.npz"
    with np.load(cap328) as data:
        pose328 = easymocap_fit_to_smplx55(
            data["Rh"], data["poses"], model_path=model_path
        ).astype(np.float32)
        betas = np.asarray(data["shapes"]).reshape(-1)[:10]
    with np.load(cap712) as data:
        pose712 = easymocap_fit_to_smplx55(
            data["Rh"], data["poses"], model_path=model_path
        ).astype(np.float32)
    poses: dict[str, np.ndarray] = {
        "tpose": np.zeros((55, 3), np.float32),
        "pose_213328": pose328,
        "pose_213712": pose712,
    }
    bedlam_root = Path("/media/camp/EXT_DRIVE/Among_US/dataset/raw/humans/bedlam2/motions")
    amass_root = Path("/media/camp/EXT_DRIVE/Among_US/dataset/raw/humans/amass_hf/raw")
    for name, kind in (("it_4051_3XL_2304.npz", "lower"), ("it_4011_XL_2114.npz", "lower")):
        path = _resolve_bedlam_file(bedlam_root, name)
        with np.load(path, allow_pickle=True) as data:
            arr = data["poses"] if "poses" in data.files else data[data.files[0]]
        fi, pose55 = _pick_frame(arr, kind=kind)
        poses[f"bedlam_{Path(name).stem}_{kind}_f{fi}"] = pose55.astype(np.float32)
    walk = amass_root / "ACCAD/Female1Walking_c3d/B12 - walk turn right (90)_poses.npz"
    with np.load(walk, allow_pickle=True) as data:
        arr = data["poses"] if "poses" in data.files else data[data.files[0]]
    fi, pose55 = _pick_frame(arr, kind="full")
    poses[f"amass_walk_f{fi}"] = pose55.astype(np.float32)
    return poses, betas


def _render_slim(
    *,
    keep_dir: Path,
    label: str,
    pose_id: str,
    verts: np.ndarray,
    asset,
    skin: np.ndarray,
    skin_faces: np.ndarray,
    frames,
    tmp_root: Path,
) -> dict[str, dict[str, str]]:
    out = tmp_root / f"{label}_{pose_id}_{os.getpid()}"
    if out.exists():
        shutil.rmtree(out)
    _render_modes(
        output=out,
        vertices=np.asarray(verts, dtype=np.float32),
        asset=asset,
        skin=skin,
        skin_faces=skin_faces,
        frames=frames,
        backend="cuda",
        camera_names=CAMS,
    )
    skin_obj = out / "mesh_assets" / "skin.obj"
    if not skin_obj.is_file():
        raise FileNotFoundError(f"render assets missing: {skin_obj}")
    slim = keep_dir / label / pose_id
    kept: dict[str, dict[str, str]] = {}
    for layer in LAYERS:
        kept[layer] = {}
        for cam in CAMS:
            src = out / layer / "rgb" / f"{cam}.png"
            if not src.is_file():
                continue
            dst = slim / layer / f"{cam}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            kept[layer][cam] = str(dst)
        sheet = out / layer / "contact_sheet.png"
        if sheet.is_file():
            dst = slim / f"{layer}_contact_sheet.png"
            shutil.copy2(sheet, dst)
            kept[layer]["contact_sheet"] = str(dst)
    shutil.rmtree(out)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--operator",
        type=Path,
        default=Path("outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8"),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path(
            "outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node1_006/anatomical_calibration_v1"
        ),
    )
    parser.add_argument(
        "--oracle",
        type=Path,
        default=Path(
            "outputs/anatomy_retarget/v7_candidates/blender_link_oracle_v7_full_001/blender_link_oracle_v7.npz"
        ),
    )
    parser.add_argument(
        "--whole-chain",
        type=Path,
        default=Path(
            "outputs/anatomy_retarget/v8_candidates/chain_retarget_v7_node2_001/subject_213328"
        ),
        help="Bind authority (default V7). Weight refit does not re-solve rest.",
    )
    parser.add_argument(
        "--smplx-model",
        type=Path,
        default=Path("ref_code_library/EasyMocap/data/smplx/smplx/SMPLX_MALE.pkl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/anatomy_retarget/v9_candidates/weight_refit_v1_shadow_213328"
        ),
    )
    parser.add_argument("--prior-strength", type=float, default=0.35)
    parser.add_argument("--temperature-m", type=float, default=0.025)
    parser.add_argument("--core-axis-frac", type=float, default=0.55)
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()

    root = args.repo_root.resolve()
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output
    if output.exists():
        raise SystemExit(f"output exists (refuse overwrite): {output}")
    output.mkdir(parents=True, exist_ok=False)
    tmp = root / "outputs/anatomy_retarget/v9_candidates/_weight_refit_v1_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    op = load_source_operator((root / args.operator).resolve(), mmap=True)
    cal = load_anatomical_calibration_v1(
        (root / args.calibration).resolve(),
        operator=op,
        required_scope="full_main_chain",
    )
    model_path, model_sha = require_frozen_smplx_male_v7((root / args.smplx_model).resolve())
    model = load_smplx_model_v7(model_path)
    poses, betas = _load_poses(model_path, root)
    asset = materialize_subject(op, betas=betas, gender="male").rigged_asset
    value = load_whole_chain_rest_fit_v1(
        (root / args.whole_chain).resolve(),
        operator=op,
        calibration=cal,
        smplx_model=model,
        smplx_model_sha256=model_sha,
        recheck=False,
    )
    # Refit on body-shape rest (fitted anatomy), not template rest.
    refit = build_weight_refit_v1(
        asset,
        subject_label="213328",
        rest_vertices=np.asarray(value.vertices_final, dtype=np.float64),
        prior_strength=float(args.prior_strength),
        temperature_m=float(args.temperature_m),
        core_axis_frac=float(args.core_axis_frac),
    )
    save_weight_refit_v1(output / "refit", refit)
    asset_refit = refit.apply_to_asset(asset)

    oracle = (root / args.oracle).resolve()
    pm_base = build_pose_map_v1(
        value,
        asset=asset,
        calibration=cal,
        oracle_path=oracle,
        source_operator_digest=op.runtime_digest(validate=False),
    )
    # Same bind/pose_map; only weights change.
    pm_refit = pm_base

    audit = {
        "artifact_kind": "WeightRefitV1ShadowAudit",
        "publishable": False,
        "bind_authority": str(args.whole_chain),
        "prior_strength": float(args.prior_strength),
        "refit_report": refit.build_report,
        "cells": [],
    }
    render_manifest = {
        "artifact_kind": "WeightRefitV1SlimRenders",
        "publishable": False,
        "cameras": list(CAMS),
        "layers": list(LAYERS),
        "poses": {},
    }

    for pose_id, pose in poses.items():
        skin, faces = smplx_body_surface_v7(model, betas=betas, pose_axis_angle=pose)
        verts_v7, _ = pose_whole_chain_vertices(
            value, pm_base, source_asset=asset, pose_axis_angle=pose
        )
        verts_rf, _ = pose_whole_chain_vertices(
            value, pm_refit, source_asset=asset_refit, pose_axis_angle=pose
        )
        out_v7 = _femur_patella_outside(verts_v7, asset, skin, faces)
        out_rf = _femur_patella_outside(verts_rf, asset_refit, skin, faces)
        # Seat gaps (left medial/lateral) using calibration frames on refit verts.
        frames, _, _ = _measure_frames(
            verts_rf, cal.domains, cal.joint_domain_bases, partition="validation"
        )
        cell = {
            "pose_id": pose_id,
            "v7_femur_patella_max_outside_m": out_v7["femur_patella_max_outside_m"],
            "refit_femur_patella_max_outside_m": out_rf["femur_patella_max_outside_m"],
            "improved_vs_v7": bool(
                out_rf["femur_patella_max_outside_m"]
                <= out_v7["femur_patella_max_outside_m"] + 1.0e-6
            ),
            "meshes_v7": {k: out_v7[k] for k in KNEE_MESHES},
            "meshes_refit": {k: out_rf[k] for k in KNEE_MESHES},
        }
        audit["cells"].append(cell)
        print(
            f"{pose_id}: v7={out_v7['femur_patella_max_outside_m']*1000:.1f}mm "
            f"refit={out_rf['femur_patella_max_outside_m']*1000:.1f}mm "
            f"improved={cell['improved_vs_v7']}",
            flush=True,
        )

        if not args.skip_render:
            kept_rf = _render_slim(
                keep_dir=output / "renders",
                label="refit",
                pose_id=pose_id,
                verts=verts_rf,
                asset=asset_refit,
                skin=skin,
                skin_faces=faces,
                frames=frames,
                tmp_root=tmp,
            )
            render_manifest["poses"][pose_id] = {"refit": kept_rf}
            if pose_id in {"tpose", "pose_213328", "pose_213712"}:
                frames_v7, _, _ = _measure_frames(
                    verts_v7, cal.domains, cal.joint_domain_bases, partition="validation"
                )
                kept_v7 = _render_slim(
                    keep_dir=output / "renders",
                    label="v7",
                    pose_id=pose_id,
                    verts=verts_v7,
                    asset=asset,
                    skin=skin,
                    skin_faces=faces,
                    frames=frames_v7,
                    tmp_root=tmp,
                )
                render_manifest["poses"][pose_id]["v7"] = kept_v7

    n_improved = sum(1 for c in audit["cells"] if c["improved_vs_v7"])
    audit["n_improved_vs_v7"] = int(n_improved)
    audit["n_total"] = len(audit["cells"])
    # Visual authority: do not claim pass from numbers alone.
    audit["passed"] = False
    audit["decision"] = "shadow_only_await_visual_review"
    (output / "multipose_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if not args.skip_render:
        (output / "renders" / "manifest.json").write_text(
            json.dumps(render_manifest, indent=2) + "\n"
        )
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"OUTPUT {output}", flush=True)
    print(
        f"improved_vs_v7 {n_improved}/{len(audit['cells'])} (numbers only; await red-image review)",
        flush=True,
    )


if __name__ == "__main__":
    main()
