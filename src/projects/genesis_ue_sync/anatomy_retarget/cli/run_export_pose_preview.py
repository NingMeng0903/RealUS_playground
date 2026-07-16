"""Offline preview exporter: skin the retargeted anatomy with a terminal-8 SMPL-X fit.

Outputs OBJ files for Blender-side inspection:
  - anatomy_tpose.obj      retargeted anatomy in the subject canonical T-pose
  - smpl_tpose.obj         subject canonical SMPL-X T-pose (overlay reference)
  - anatomy_posed.obj      anatomy skinned with the captured Rh/Th/poses
  - smpl_fit_posed.obj     fitted orange SMPL-X mesh from the capture (overlay reference)
  - smpl_drive_check.obj   canonical SMPL-X mesh re-skinned through the anatomy LBS path
  - preview_report.json    drive-check RMS vs fitted vertices + spans

The drive-check RMS validates the whole pose pipeline (pose adapter + LBS + pivot
compensation): a small value (< ~0.02 m) means the anatomy receives exactly the same
motion as the fitted orange mesh.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import joint_global_transforms, skin_vertices
from projects.genesis_ue_sync.anatomy_retarget.obj_io import read_obj_vertices, write_obj
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import (
    easymocap_drive_translation,
    easymocap_fit_to_smplx55,
)
from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import load_rigged_asset
from projects.genesis_ue_sync.anatomy_retarget.viz_overlay import (
    draw_preview_overlay,
    draw_regional_preview_overlay,
    sparse_leg_bone_vertices,
)

DEFAULT_ASSET = Path("outputs/anatomy_retarget/latest_asset/anatomy_rigged.npz")
DEFAULT_CANONICAL = Path("outputs/anatomy_retarget/latest_canonical")
DEFAULT_OUTPUT = Path("outputs/anatomy_retarget/preview")

LEG_BONE_VIZ_MESHES = frozenset(
    {"Femur_L", "Femur_R", "Tibia_L", "Tibia_R", "Fibula_L", "Fibula_R", "Patella_L", "Patella_R"}
)


def _regional_vertices(asset, vertices: np.ndarray, region: str) -> np.ndarray:
    indices: list[np.ndarray] = []
    for name, (start, stop), tissue in zip(
        asset.source_mesh_names, asset.source_vertex_ranges, asset.source_tissues
    ):
        lower = str(name).lower()
        side = "left" if lower.endswith("_l") else "right" if lower.endswith("_r") else ""
        include = False
        if region == "head":
            include = any(token in lower for token in ("skull", "brain", "cerebr", "cerebell", "lobe", "amygdala", "thalam"))
        elif region == "pelvis":
            include = str(tissue) == "bone" and any(token in lower for token in ("ilium", "sacrum", "ischium", "pubis", "pelvis"))
        elif region in {"left_hand", "right_hand"}:
            include = side == region.split("_", 1)[0] and str(tissue) == "bone" and any(
                token in lower for token in ("metacarpal", "phalanx_hand", "phalanges_hand")
            )
        elif region in {"left_arm_hand", "right_arm_hand"}:
            include = side == region.split("_", 1)[0] and str(tissue) == "bone" and any(
                token in lower
                for token in (
                    "humerus", "radius", "ulna", "scaphoid", "lunate", "triquetrum",
                    "pisiform", "trapezium", "trapezoid", "capitate", "hamate",
                    "metacarpal", "phalanx_hand", "phalanges_hand",
                )
            )
        elif region in {"left_foot", "right_foot"}:
            include = side == region.split("_", 1)[0] and str(tissue) == "bone" and any(
                token in lower
                for token in ("calcaneus", "talus", "navicular", "cuboid", "cuneiform", "metatarsal", "phalanx_foot", "phalanges_foot")
            )
        if include:
            indices.append(np.arange(int(start), int(stop), dtype=np.int64))
    if not indices:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(vertices, dtype=np.float32)[np.concatenate(indices)]


def _load_root_align_offset(motion_npz: Path, data: "np.lib.npyio.NpzFile") -> np.ndarray:
    if "root_align_offset" in data.files:
        return np.asarray(data["root_align_offset"], dtype=np.float32).reshape(3)
    moment_json = motion_npz.parent / "moment.json"
    if moment_json.is_file():
        info = json.loads(moment_json.read_text(encoding="utf-8")).get("smpl_root_alignment") or {}
        if info.get("applied") and info.get("offset_m") is not None:
            return np.asarray(info["offset_m"], dtype=np.float32).reshape(3)
    return np.zeros(3, dtype=np.float32)


def _span(vertices: np.ndarray) -> list[float]:
    return [float(v) for v in np.ptp(np.asarray(vertices, dtype=np.float32).reshape(-1, 3), axis=0)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--asset-npz", type=Path, default=DEFAULT_ASSET)
    p.add_argument("--motion-npz", type=Path, required=True, help="smplx_result.npz from a terminal-8 capture")
    p.add_argument("--canonical-dir", type=Path, default=DEFAULT_CANONICAL)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--apply-root-align",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add the capture root_align_offset (matches the published Genesis position)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    asset = load_rigged_asset(args.asset_npz)
    pelvis = np.asarray(asset.rest_joints, dtype=np.float32).reshape(-1, 3)[0]

    data = np.load(args.motion_npz)
    Rh = np.asarray(data["Rh"], dtype=np.float32).reshape(3)
    Th = np.asarray(data["Th"], dtype=np.float32).reshape(3)
    poses = np.asarray(data["poses"], dtype=np.float32).reshape(-1)
    fit_vertices = np.asarray(data["vertices"], dtype=np.float32).reshape(-1, 3)
    fit_faces = np.asarray(data["faces"], dtype=np.int32).reshape(-1, 3)
    offset = _load_root_align_offset(Path(args.motion_npz), data)
    extra = offset if args.apply_root_align else np.zeros(3, dtype=np.float32)

    pose55 = easymocap_fit_to_smplx55(Rh, poses).reshape(-1)
    th_eff = easymocap_drive_translation(Rh, Th, pelvis) + extra

    report: dict[str, object] = {
        "asset_npz": str(args.asset_npz),
        "motion_npz": str(args.motion_npz),
        "root_align_offset_m": [float(v) for v in offset],
        "root_align_applied": bool(args.apply_root_align),
    }

    write_obj(out / "anatomy_tpose.obj", asset.vertices_rest, asset.faces, comment="anatomy canonical T-pose")
    report["anatomy_tpose_span_m"] = _span(asset.vertices_rest)

    canonical_tpose_obj = Path(args.canonical_dir) / "smpl_canonical_tpose.obj"
    smpl_tpose_vertices: np.ndarray | None = None
    if canonical_tpose_obj.is_file():
        smpl_tpose_vertices = read_obj_vertices(canonical_tpose_obj)
        weights_npz = Path(args.canonical_dir) / "smpl_canonical_weights.npz"
        canon = np.load(weights_npz, allow_pickle=True)
        write_obj(out / "smpl_tpose.obj", smpl_tpose_vertices, canon["faces"], comment="subject canonical T-pose")

        # Drive-check: re-skin the canonical SMPL mesh through the anatomy LBS path
        # and compare against the fitted vertices from the capture.
        global_tf = joint_global_transforms(
            pose_axis_angle=pose55, rest_joints=canon["rest_joints"], parents=canon["parents"]
        )
        transforms = np.matmul(global_tf, np.asarray(canon["inverse_bind"], dtype=np.float32))
        blended = np.matmul(
            np.asarray(canon["lbs_weights"], dtype=np.float32), transforms.reshape(transforms.shape[0], 16)
        ).reshape(-1, 4, 4)
        homo = np.concatenate([smpl_tpose_vertices, np.ones((smpl_tpose_vertices.shape[0], 1), np.float32)], axis=1)
        canon_pelvis = np.asarray(canon["rest_joints"], dtype=np.float32).reshape(-1, 3)[0]
        check = np.matmul(blended, homo[:, :, None])[:, :3, 0]
        check = check + easymocap_drive_translation(Rh, Th, canon_pelvis) + extra
        write_obj(out / "smpl_drive_check.obj", check, canon["faces"], comment="canonical mesh via anatomy LBS path")
        target = fit_vertices + extra
        if check.shape == target.shape:
            rms = float(np.sqrt(np.mean(np.sum((check - target) ** 2, axis=1))))
            report["drive_check_rms_m"] = rms
            print(f"INFO drive-check rms vs fitted vertices: {rms:.4f} m")
        else:
            report["drive_check_rms_m"] = None
            print("WARN drive-check skipped: vertex count mismatch")

    posed = skin_vertices(asset, pose55, transl=th_eff)
    write_obj(out / "anatomy_posed.obj", posed, asset.faces, comment="anatomy skinned with capture pose (live LBS)")
    report["anatomy_posed_span_m"] = _span(posed)
    report["anatomy_posed_center_m"] = [float(v) for v in posed.mean(axis=0)]

    if asset.pose_cache_vertices is not None and asset.pose_cache_vertices.size:
        cache_verts = np.asarray(asset.pose_cache_vertices, dtype=np.float32).reshape(-1, 3)
        write_obj(out / "anatomy_pose_cache.obj", cache_verts, asset.faces, comment="offline pose cache vertices")
        if cache_verts.shape == posed.shape:
            delta = np.linalg.norm(cache_verts - posed, axis=1)
            report["pose_cache_vs_lbs_max_m"] = float(np.max(delta))
            report["pose_cache_vs_lbs_mean_m"] = float(np.mean(delta))
            report["pose_cache_vs_lbs_p999_m"] = float(np.quantile(delta, 0.999))
            print(
                "INFO pose cache vs live LBS: "
                f"max={report['pose_cache_vs_lbs_max_m']:.4f} m mean={report['pose_cache_vs_lbs_mean_m']:.4f} m"
            )

    write_obj(out / "smpl_fit_posed.obj", fit_vertices + extra, fit_faces, comment="fitted SMPL-X mesh from capture")
    report["smpl_fit_center_m"] = [float(v) for v in (fit_vertices + extra).mean(axis=0)]

    raw = np.load(args.asset_npz, allow_pickle=True)
    if smpl_tpose_vertices is not None:
        bone_tpose = sparse_leg_bone_vertices(asset.vertices_rest, raw, LEG_BONE_VIZ_MESHES)
        bone_posed = sparse_leg_bone_vertices(posed, raw, LEG_BONE_VIZ_MESHES)
        draw_preview_overlay(
            out / "preview_overlay.png",
            smpl_tpose=smpl_tpose_vertices,
            anatomy_tpose=np.asarray(asset.vertices_rest, dtype=np.float32),
            smpl_posed=fit_vertices + extra,
            anatomy_posed=posed,
        )
        draw_preview_overlay(
            out / "preview_overlay_with_leg_bones.png",
            smpl_tpose=smpl_tpose_vertices,
            anatomy_tpose=np.asarray(asset.vertices_rest, dtype=np.float32),
            smpl_posed=fit_vertices + extra,
            anatomy_posed=posed,
            leg_bones_tpose=bone_tpose,
            leg_bones_posed=bone_posed,
        )
        for region in (
            "head", "pelvis", "left_arm_hand", "right_arm_hand", "left_hand",
            "right_hand", "left_foot", "right_foot",
        ):
            rest_region = _regional_vertices(asset, asset.vertices_rest, region)
            posed_region = _regional_vertices(asset, posed, region)
            if not len(rest_region):
                continue
            draw_regional_preview_overlay(
                out / f"overlay_{region}.png",
                title=region.replace("_", " ").title(),
                smpl_tpose=smpl_tpose_vertices,
                anatomy_tpose=rest_region,
                smpl_posed=fit_vertices + extra,
                anatomy_posed=posed_region,
            )
        report["leg_bone_marker_points"] = {
            "tpose": int(bone_tpose.shape[0]),
            "posed": int(bone_posed.shape[0]),
            "meshes": sorted(LEG_BONE_VIZ_MESHES),
        }

    (out / "preview_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"INFO preview exported -> {out}")
    for name in (
        "anatomy_tpose.obj",
        "smpl_tpose.obj",
        "anatomy_posed.obj",
        "anatomy_pose_cache.obj",
        "smpl_fit_posed.obj",
        "smpl_drive_check.obj",
        "preview_overlay.png",
        "preview_overlay_with_leg_bones.png",
        "overlay_head.png",
        "overlay_pelvis.png",
        "overlay_left_hand.png",
        "overlay_right_hand.png",
        "overlay_left_foot.png",
        "overlay_right_foot.png",
        "overlay_left_arm_hand.png",
        "overlay_right_arm_hand.png",
    ):
        if (out / name).is_file():
            print(f"  {out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
