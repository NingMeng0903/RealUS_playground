"""Shadow V10 body-fit: weight refit + multipose rest inset into SMPL-X.

Candidate-only. Does not update trusted/latest.
Acceptability target (soft): knee-chain max outside < 5 mm on training poses.
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
from projects.genesis_ue_sync.anatomy_retarget.pose_map_v1 import (
    build_pose_map_v1,
    pose_whole_chain_vertices,
)
from projects.genesis_ue_sync.anatomy_retarget.refit_body_fit_v1 import (
    KNEE_MESHES,
    build_body_fit_pose_catalog_v1,
    build_body_fit_v1,
    save_body_fit_v1,
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
ACCEPT_OUTSIDE_M = 0.005


def _mesh_ids(asset, mesh_name: str) -> np.ndarray:
    names = [str(x) for x in asset.source_mesh_names]
    start, stop = np.asarray(asset.source_vertex_ranges, dtype=np.int64)[names.index(mesh_name)]
    return np.arange(int(start), int(stop), dtype=np.int64)


def _knee_outside(verts, asset, skin, faces) -> dict[str, float]:
    signed = _signed_distance(verts, skin, faces)
    out: dict[str, float] = {}
    worst = 0.0
    for mesh in KNEE_MESHES:
        ids = _mesh_ids(asset, mesh)
        max_out = float(max(0.0, float(np.max(signed[ids]))))
        out[mesh] = max_out
        worst = max(worst, max_out)
    out["knee_max_outside_m"] = worst
    return out


def _load_poses(model_path: Path, root: Path):
    with np.load(root / "smplx_outputs/20260713_213328/moment_0000/smplx_result.npz") as data:
        betas = np.asarray(data["shapes"]).reshape(-1)[:10]
    poses = build_body_fit_pose_catalog_v1(repo_root=root, model_path=model_path)
    return poses, betas


def _render_slim(*, keep_dir, label, pose_id, verts, asset, skin, skin_faces, frames, tmp_root):
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
    slim = keep_dir / label / pose_id
    kept = {}
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
            "outputs/anatomy_retarget/v9_candidates/body_fit_v1_shadow_213328"
        ),
    )
    parser.add_argument("--ls-iterations", type=int, default=16)
    parser.add_argument("--inset-iterations", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()

    root = args.repo_root.resolve()
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output
    if output.exists():
        raise SystemExit(f"output exists (refuse overwrite): {output}")
    output.mkdir(parents=True, exist_ok=False)
    tmp = root / "outputs/anatomy_retarget/v9_candidates/_body_fit_v1_tmp"
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
    oracle = (root / args.oracle).resolve()
    pm = build_pose_map_v1(
        value,
        asset=asset,
        calibration=cal,
        oracle_path=oracle,
        source_operator_digest=op.runtime_digest(validate=False),
    )

    print(f"train poses: {len(poses)}", flush=True)
    ls_iters = int(args.ls_iterations if args.inset_iterations is None else args.inset_iterations)
    body = build_body_fit_v1(
        asset,
        value,
        pm,
        subject_label="213328",
        model=model,
        betas=betas,
        poses=poses,
        ls_iterations=ls_iters,
    )
    save_body_fit_v1(output / "body_fit", body)
    asset_fit = body.apply_to_asset(asset)
    value_fit = body.apply_to_subject(value)

    audit = {
        "artifact_kind": "BodyFitRefitV1ShadowAudit",
        "publishable": False,
        "accept_outside_m": ACCEPT_OUTSIDE_M,
        "body_fit_report": {
            k: body.build_report[k]
            for k in (
                "worst_final_outside_m",
                "max_rest_displacement_m",
                "n_affected_vertices",
                "inset_iterations_ran",
                "final_poses",
            )
            if k in body.build_report
        },
        "cells": [],
    }
    render_manifest = {
        "artifact_kind": "BodyFitRefitV1SlimRenders",
        "publishable": False,
        "cameras": list(CAMS),
        "layers": list(LAYERS),
        "poses": {},
    }

    n_accept = 0
    for pose_id, pose in poses.items():
        skin, faces = smplx_body_surface_v7(model, betas=betas, pose_axis_angle=pose)
        verts_v7, _ = pose_whole_chain_vertices(
            value, pm, source_asset=asset, pose_axis_angle=pose
        )
        verts_fit, _ = pose_whole_chain_vertices(
            value_fit, pm, source_asset=asset_fit, pose_axis_angle=pose
        )
        out_v7 = _knee_outside(verts_v7, asset, skin, faces)
        out_fit = _knee_outside(verts_fit, asset_fit, skin, faces)
        accepted = bool(out_fit["knee_max_outside_m"] <= ACCEPT_OUTSIDE_M)
        n_accept += int(accepted)
        cell = {
            "pose_id": pose_id,
            "v7_knee_max_outside_m": out_v7["knee_max_outside_m"],
            "fit_knee_max_outside_m": out_fit["knee_max_outside_m"],
            "accepted_5mm": accepted,
            "meshes_v7": {k: out_v7[k] for k in KNEE_MESHES},
            "meshes_fit": {k: out_fit[k] for k in KNEE_MESHES},
        }
        audit["cells"].append(cell)
        print(
            f"{pose_id}: v7={out_v7['knee_max_outside_m']*1000:.1f}mm "
            f"fit={out_fit['knee_max_outside_m']*1000:.1f}mm "
            f"accept5mm={accepted}",
            flush=True,
        )
        if not args.skip_render:
            frames, _, _ = _measure_frames(
                verts_fit, cal.domains, cal.joint_domain_bases, partition="validation"
            )
            kept = _render_slim(
                keep_dir=output / "renders",
                label="body_fit",
                pose_id=pose_id,
                verts=verts_fit,
                asset=asset_fit,
                skin=skin,
                skin_faces=faces,
                frames=frames,
                tmp_root=tmp,
            )
            render_manifest["poses"][pose_id] = {"body_fit": kept}
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

    audit["n_accept_5mm"] = int(n_accept)
    audit["n_total"] = len(audit["cells"])
    audit["passed"] = bool(n_accept == len(audit["cells"]))
    audit["decision"] = (
        "shadow_accept_pending_visual"
        if audit["passed"]
        else "shadow_reject_outside_residual"
    )
    (output / "multipose_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if not args.skip_render:
        (output / "renders" / "manifest.json").write_text(
            json.dumps(render_manifest, indent=2) + "\n"
        )
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"OUTPUT {output}", flush=True)
    print(
        f"accept5mm {n_accept}/{len(audit['cells'])} decision={audit['decision']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
