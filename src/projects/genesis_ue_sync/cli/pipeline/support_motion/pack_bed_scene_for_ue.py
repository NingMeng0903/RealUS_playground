#!/usr/bin/env python3
"""Build SyncSceneSpec JSON (bed2x1 m + human anchor + motion + capsule URDF metadata) for UE/Gensis."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
SRC_ROOT = next(parent for parent in (_THIS_FILE.parent, *_THIS_FILE.parents) if parent.name == "src")
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.project import project_paths
from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import HumanMotionSequence, load_amass_sequence
from projects.genesis_ue_sync.sim_platform.embodiments.smpl2urdf import resolve_smpl_proxy_urdf
from projects.genesis_ue_sync.sim_platform.scenes import compute_human_scene_placement, load_sync_scene_spec
from projects.genesis_ue_sync.sim_platform.scenes.robot_assets import resolve_robot_model_payload
from projects.genesis_ue_sync.sim_platform.human_refit.human_ue_calibration import (
    build_human_ue_calibration_dict,
    write_human_ue_calibration,
)


def _rel_to_repo(path: Path) -> str:
    path = path.resolve()
    root = project_paths(__file__).root.resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--amass-npz", type=Path, help="Source AMASS npz (SMPL)")
    p.add_argument("--manifest-row", type=Path, help="JSON file with one manifest row (from extract_babel_bed_subset)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--scene-name", type=str, default="babel_bed_scene")
    p.add_argument("--source-id", type=str, default="")
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--frame-limit", type=int, default=0, help="Max frames written to sequence npz (0=all)")
    p.add_argument("--bed-size", type=float, nargs=3, default=(2.0, 1.0, 0.36), metavar=("LX", "LY", "LZ"))
    p.add_argument("--bed-pos", type=float, nargs=3, default=(0.55, 0.0, 0.18), metavar=("X", "Y", "Z"))
    p.add_argument("--human-anchor-xy", type=float, nargs=2, default=(0.55, 0.0), metavar=("AX", "AY"))
    p.add_argument("--support-margin-m", type=float, default=0.015)
    p.add_argument("--collision-only-urdf", action="store_true")
    p.add_argument(
        "--skip-proxy-urdf",
        action="store_true",
        help="Skip SMPL capsule URDF (no torch/smpl body model in environment).",
    )
    p.add_argument("--extra-cam", action="store_true", help="Add third overhead camera")
    p.add_argument("--validate", action="store_true", help="Load written spec with load_sync_scene_spec")
    p.add_argument("--bed-fit", action="store_true", help="Run SMPL bed fit and write human_scene_placement.json")
    p.add_argument("--human-placement-out", type=Path, default=None, help="Output path for HumanScenePlacement JSON")
    p.add_argument("--robot-model", type=str, default="panda_urdf", help="Robot model_id from assets/robots/<id>/robot.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.amass_npz and args.manifest_row:
        raise SystemExit("Use only one of --amass-npz or --manifest-row")
    if not args.amass_npz and not args.manifest_row:
        raise SystemExit("Provide --amass-npz or --manifest-row")

    npz_src: Path
    babel_sid = ""
    label_text = ""
    if args.manifest_row:
        row = json.loads(Path(args.manifest_row).read_text(encoding="utf-8"))
        rnpz = row.get("resolved_npz")
        if not rnpz:
            raise SystemExit("manifest row missing resolved_npz")
        npz_src = Path(rnpz)
        babel_sid = str(row.get("babel_sid", ""))
        label_text = str(row.get("label_text", ""))
    else:
        npz_src = Path(args.amass_npz)

    if not npz_src.is_file():
        raise FileNotFoundError(npz_src)

    seq = load_amass_sequence(npz_src)
    if args.fps is not None:
        seq = HumanMotionSequence(
            source_dataset=seq.source_dataset,
            sequence_name=seq.sequence_name,
            source_path=seq.source_path,
            model_type=seq.model_type,
            fps=float(args.fps),
            gender=seq.gender,
            betas=seq.betas,
            poses=seq.poses,
            trans=seq.trans,
            image_names=list(seq.image_names),
            cam_int=seq.cam_int,
            cam_ext=seq.cam_ext,
            metadata=dict(seq.metadata),
        )
    if args.frame_limit and seq.frame_count > int(args.frame_limit):
        n = int(args.frame_limit)
        seq = HumanMotionSequence(
            source_dataset=seq.source_dataset,
            sequence_name=seq.sequence_name + f"_f{n}",
            source_path=seq.source_path,
            model_type=seq.model_type,
            fps=seq.fps,
            gender=seq.gender,
            betas=seq.betas,
            poses=seq.poses[:n],
            trans=seq.trans[:n],
            image_names=list(seq.image_names[:n]) if seq.image_names else [],
            cam_int=seq.cam_int[:n] if seq.cam_int is not None else None,
            cam_ext=seq.cam_ext[:n] if seq.cam_ext is not None else None,
            metadata=dict(seq.metadata),
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seq_out = args.out_dir / "sequence.npz"
    seq.save(seq_out)

    urdf_path: Path | None = None
    proxy_geom = None
    urdf_rel = ""
    shape_key = ""
    if not args.skip_proxy_urdf:
        urdf_dir = args.out_dir / "urdf_cache"
        try:
            urdf_path, proxy_geom = resolve_smpl_proxy_urdf(
                seq,
                cache_dir=urdf_dir,
                device="cpu",
                collision_only=bool(args.collision_only_urdf),
                force_rewrite=True,
            )
            urdf_rel = _rel_to_repo(urdf_path)
            shape_key = proxy_geom.shape_key
            shutil.copy2(urdf_path, args.out_dir / urdf_path.name)
        except (ModuleNotFoundError, FileNotFoundError, RuntimeError) as exc:
            msg = str(exc).lower()
            if isinstance(exc, ModuleNotFoundError) and "torch" not in msg:
                raise
            urdf_path = None
            urdf_rel = ""
            shape_key = ""
            print(f"[pack_bed_scene_for_ue] proxy_urdf_skipped: {exc}", flush=True)

    lx, ly, lz = (float(x) for x in args.bed_size)
    bx, by, bz = (float(x) for x in args.bed_pos)
    ax, ay = (float(x) for x in args.human_anchor_xy)
    top_z = bz + 0.5 * lz
    look_z = top_z + 0.195

    source_id = args.source_id or (f"babel_{babel_sid}" if babel_sid else seq.sequence_name)

    cameras: list[dict] = [
        {
            "name": "cam_left",
            "res": [640, 352],
            "pos": [bx, -2.55, 1.32],
            "lookat": [bx, 0.0, look_z],
            "up": [0.0, 0.0, 1.0],
            "fov": 62.0,
        },
        {
            "name": "cam_right",
            "res": [640, 352],
            "pos": [bx, 2.55, 1.32],
            "lookat": [bx, 0.0, look_z],
            "up": [0.0, 0.0, 1.0],
            "fov": 62.0,
        },
    ]
    if args.extra_cam:
        cameras.append(
            {
                "name": "cam_top",
                "res": [640, 352],
                "pos": [bx, 0.0, top_z + 2.49],
                "lookat": [bx, 0.0, look_z],
                "up": [1.0, 0.0, 0.0],
                "fov": 55.0,
            }
        )

    payload: dict = {
        "name": args.scene_name,
        "environment": {
            "ue_map": "/Game/Bedlam/IBLMap",
            "ue_hdri_name": "abandoned_hopper_terminal_03",
            "ground_plane_color": [0.92, 0.92, 0.92, 1.0],
        },
        "support_surface": {
            "name": "bed_surface",
            "pos": [bx, by, bz],
            "size": [lx, ly, lz],
            "color": [0.75, 0.75, 0.78, 1.0],
            "semantic_role": "bed",
            "spawn_in_genesis": True,
            "spawn_in_ue": True,
        },
        "robot": resolve_robot_model_payload(
            {
                "model_id": str(args.robot_model).strip(),
                "name": "robot_main",
                "base_pos": [0.0, -0.35, bz + 0.5 * lz],
                "joint_positions": [0.0, 0.35, 0.0, -1.2, 0.0, 1.1, 0.0],
            }
        ),
        "human": {
            "anchor_pos": [ax, ay, 0.0],
            "support_margin_m": float(args.support_margin_m),
            "support_reference": "support_surface_top",
            "align_floor": True,
        },
        "motion": {
            "source_id": source_id,
            "source_path": _rel_to_repo(npz_src),
            "sequence_npz_path": _rel_to_repo(seq_out),
            "mesh_manifest_path": "",
            "fps": float(seq.fps),
            "frame_count": int(seq.frame_count),
            "start_frame": 0,
            "frame_step": 1,
        },
        "render": {
            "fps": 8.0,
            "frame_limit": min(120, int(seq.frame_count)),
            "genesis_backend": "cuda",
            "ue_frame_count": min(240, int(seq.frame_count)),
            "ue_frame_step": 1,
            "ue_render_now": False,
        },
        "ue_avatar": {
            "body_mode": "retargeted_overlay",
            "body_name": "it_4375_M_2400",
            "texture_body": "Male_Skin_Preset_0027",
            "texture_clothing": None,
            "texture_clothing_overlay": "gr_ben_004_M_texture_01",
            "skeletal_mesh_path": "/Engine/PS/Bedlam/SMPLX_LH_animations/it_4375_M/it_4375_M_2400",
            "animation_asset_root": "/Game/Bedlam/Generated/RetargetedAnimations",
            "imported_fbx_root": "/Game/Bedlam/Generated/ImportedSMPLMotion",
            "fallback_animation_path": "",
            "hidden_material_path": "/Engine/PS/Bedlam/Core/Materials/M_SMPLX_Hidden.M_SMPLX_Hidden",
            "fbx_global_scale": 100.0,
        },
        "cameras": cameras,
        "metadata": {
            "babel_sid": babel_sid,
            "label_text": label_text,
            "human_collision_urdf": urdf_rel,
            "proxy_shape_key": shape_key,
            "bed_top_z_m": top_z,
            "notes": "Generated by pack_bed_scene_for_ue.py. Units meters.",
        },
    }

    scene_path = args.out_dir / "sync_scene.json"
    scene_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.bed_fit:
        spec_loaded = load_sync_scene_spec(scene_path)
        placement_path = args.human_placement_out or (args.out_dir / "human_scene_placement.json")
        placement = compute_human_scene_placement(
            seq,
            scene_spec=spec_loaded,
            sequence_npz_path=_rel_to_repo(seq_out),
            device="cpu",
        )
        placement.save(placement_path)
        payload["metadata"]["human_scene_placement"] = _rel_to_repo(placement_path)
        payload["metadata"]["scene_fit_revision"] = placement.scene_fit_revision
        scene_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[pack_bed_scene_for_ue] human_placement={placement_path}", flush=True)

    try:
        betas_list: list[float] = []
        if seq.betas is not None:
            import numpy as _np

            betas_list = [float(x) for x in _np.asarray(seq.betas, dtype=_np.float64).reshape(-1).tolist()]
    except Exception:
        betas_list = []
    calib = build_human_ue_calibration_dict(
        ue_avatar=payload["ue_avatar"],
        human_block=payload["human"],
        motion_block=payload["motion"],
        human_scene_placement_rel=str(payload["metadata"].get("human_scene_placement") or ""),
        scene_fit_revision=str(payload["metadata"].get("scene_fit_revision") or ""),
        betas=betas_list,
    )
    calib_path = args.out_dir / "human_ue_calibration.json"
    write_human_ue_calibration(calib_path, calib)
    payload["metadata"]["human_ue_calibration"] = _rel_to_repo(calib_path)
    scene_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[pack_bed_scene_for_ue] human_ue_calibration={calib_path}", flush=True)

    print(f"[pack_bed_scene_for_ue] scene={scene_path}", flush=True)
    print(f"[pack_bed_scene_for_ue] sequence={seq_out}", flush=True)
    print(f"[pack_bed_scene_for_ue] urdf={urdf_path}", flush=True)

    if args.validate:
        spec = load_sync_scene_spec(scene_path)
        assert spec.motion.resolved_sequence_npz_path is not None
        assert spec.motion.resolved_sequence_npz_path.is_file()
        print("[pack_bed_scene_for_ue] validate_ok", spec.name, spec.motion.frame_count, flush=True)


if __name__ == "__main__":
    main()
