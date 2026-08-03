"""Genesis renders for AMASS/BEDLAM beta×pose matrix cells (full_anatomy included)."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    _measure_frames,
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.cli.render_stage1_baseline_compare_v1 import (
    _render_modes,
)
from projects.genesis_ue_sync.anatomy_retarget.cli.run_amass_bedlam_retarget_matrix_v6 import (
    COUPLED_RBF_SUPPORT_RAD,
    FROZEN_AMASS_MOTIONS,
    FROZEN_BEDLAM2_MOTIONS,
    _load_capture_betas,
    _pick_frame,
    _resolve_bedlam_file,
    _sample_bedlam_betas,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import easymocap_fit_to_smplx55
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
    build_whole_chain_rest_fit_v1,
    load_whole_chain_rest_fit_v1,
)


MATRIX_CAMERAS = (
    "whole_ap",
    "whole_pa",
    "left_knee_ap",
    "left_knee_lateral",
    "right_knee_ap",
    "left_elbow_ap",
    "left_hand_oblique",
    "left_wrist_ap",
    "left_ankle_ap",
)

FOCUS_CAMERAS = (
    "left_knee_ap",
    "left_knee_lateral",
    "left_elbow_ap",
    "whole_ap",
    "whole_pa",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _focus_sheet(paths: list[Path], labels: list[str], output: Path) -> Path:
    images = [Image.open(path).convert("RGB") for path in paths]
    thumb = (480, 360)
    cols = min(3, max(1, len(images)))
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb[0], rows * 400), (18, 20, 24))
    draw = ImageDraw.Draw(sheet)
    for index, (image, label) in enumerate(zip(images, labels)):
        image.thumbnail(thumb)
        x = (index % cols) * thumb[0]
        y = (index // cols) * 400
        sheet.paste(image, (x, y))
        draw.text((x + 8, y + 368), label, fill=(235, 235, 235))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--smplx-model", type=Path, required=True)
    parser.add_argument("--capture-213328", type=Path, required=True)
    parser.add_argument("--capture-213712", type=Path, required=True)
    parser.add_argument(
        "--shadow",
        type=Path,
        required=True,
        help="chain_retarget_v6/v7 shadow root with subject_* folders",
    )
    parser.add_argument(
        "--matrix-manifest",
        type=Path,
        default=None,
        help="Optional existing matrix manifest; otherwise rebuild motion catalog.",
    )
    parser.add_argument(
        "--bedlam2-motions",
        type=Path,
        default=Path(
            "/media/camp/EXT_DRIVE/Among_US/dataset/raw/humans/bedlam2/motions"
        ),
    )
    parser.add_argument(
        "--amass-hf-root",
        type=Path,
        default=Path("/media/camp/EXT_DRIVE/Among_US/dataset/raw/humans/amass_hf/raw"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--extra-bedlam-betas", type=int, default=2)
    parser.add_argument(
        "--tag",
        default="matrix_genesis",
        help="Artifact kind suffix, e.g. v6_before or v7",
    )
    return parser


def _resolve_motions(
    *,
    bedlam_root: Path,
    amass_root: Path,
    captures: dict[str, Path],
    model_path: Path,
    matrix_manifest: Path | None,
) -> list[dict[str, Any]]:
    if matrix_manifest is not None and matrix_manifest.is_file():
        catalog = json.loads(
            (matrix_manifest.parent / "motion_catalog.json").read_text(encoding="utf-8")
        )
        # Re-pick poses from catalog paths for exact pose55.
        cells = []
        for entry in catalog["motions"]:
            pose_id = entry["pose_id"]
            if pose_id == "tpose":
                cells.append(
                    {
                        "pose_id": "tpose",
                        "pose55": np.zeros((55, 3), dtype=np.float32),
                        "source": "synthetic",
                    }
                )
                continue
            if pose_id.startswith("pose_"):
                label = pose_id.split("_", 1)[1]
                with np.load(captures[label], allow_pickle=False) as data:
                    pose55 = easymocap_fit_to_smplx55(
                        data["Rh"], data["poses"], model_path=model_path
                    )
                cells.append(
                    {"pose_id": pose_id, "pose55": pose55, "source": "capture"}
                )
                continue
            path = Path(entry["path"])
            frame_index = int(entry["frame_index"])
            kind = str(entry.get("kind", "full"))
            with np.load(path, allow_pickle=False) as data:
                poses = np.asarray(data["poses"], dtype=np.float64)
                if poses.ndim == 1:
                    _, pose55 = _pick_frame(poses, kind=kind)
                else:
                    # Exact catalog frame, then enforce RBF support clamp.
                    row = poses[frame_index]
                    _, pose55 = _pick_frame(row, kind=kind)
            cells.append(
                {
                    "pose_id": pose_id,
                    "pose55": pose55,
                    "source": entry.get("source", "motion"),
                    "path": str(path),
                    "frame_index": frame_index,
                    "kind": kind,
                }
            )
        return cells

    motion_cells: list[dict[str, Any]] = [
        {
            "pose_id": "tpose",
            "pose55": np.zeros((55, 3), dtype=np.float32),
            "source": "synthetic",
        }
    ]
    for label, path in captures.items():
        with np.load(path, allow_pickle=False) as data:
            pose55 = easymocap_fit_to_smplx55(
                data["Rh"], data["poses"], model_path=model_path
            )
        motion_cells.append(
            {"pose_id": f"pose_{label}", "pose55": pose55, "source": "capture"}
        )
    for name, kind in zip(FROZEN_BEDLAM2_MOTIONS, ("upper", "upper", "lower", "lower")):
        path = _resolve_bedlam_file(bedlam_root, name)
        with np.load(path, allow_pickle=False) as data:
            frame_index, pose55 = _pick_frame(data["poses"], kind=kind)
        motion_cells.append(
            {
                "pose_id": f"bedlam2_{path.stem}_{kind}_f{frame_index}",
                "pose55": pose55,
                "source": "bedlam2",
                "path": str(path),
                "frame_index": frame_index,
                "kind": kind,
            }
        )
    for rel in FROZEN_AMASS_MOTIONS:
        path = (amass_root / rel).resolve()
        if not path.is_file():
            matches = sorted(amass_root.rglob(Path(rel).name))
            if not matches:
                raise FileNotFoundError(f"AMASS motion missing: {rel}")
            path = matches[0]
        with np.load(path, allow_pickle=False) as data:
            frame_index, pose55 = _pick_frame(data["poses"], kind="full")
        motion_cells.append(
            {
                "pose_id": f"amass_{path.stem}_f{frame_index}",
                "pose55": pose55,
                "source": "amass_hf",
                "path": str(path),
                "frame_index": frame_index,
                "kind": "full",
            }
        )
    return motion_cells


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    operator = load_source_operator(args.operator.resolve(), mmap=True)
    calibration = load_anatomical_calibration_v1(
        args.calibration.resolve(),
        operator=operator,
        required_scope="full_main_chain",
    )
    model_path, model_sha = require_frozen_smplx_male_v7(args.smplx_model)
    model = load_smplx_model_v7(model_path)
    captures = {
        "213328": args.capture_213328.resolve(),
        "213712": args.capture_213712.resolve(),
    }
    shadow = args.shadow.resolve()
    bedlam_root = args.bedlam2_motions.resolve()
    amass_root = args.amass_hf_root.resolve()
    motions = _resolve_motions(
        bedlam_root=bedlam_root,
        amass_root=amass_root,
        captures=captures,
        model_path=model_path,
        matrix_manifest=(
            args.matrix_manifest.resolve() if args.matrix_manifest else None
        ),
    )

    beta_specs: list[dict[str, Any]] = []
    exclude: list[np.ndarray] = []
    for label, path in captures.items():
        betas = _load_capture_betas(path)
        subject_dir = shadow / f"subject_{label}"
        beta_specs.append(
            {
                "label": label,
                "betas": betas,
                "subject_dir": subject_dir if subject_dir.is_dir() else None,
            }
        )
        exclude.append(betas)
    for stem, betas in _sample_bedlam_betas(
        bedlam_root, args.extra_bedlam_betas, exclude=exclude
    ):
        beta_specs.append(
            {
                "label": f"bedlam_{stem}",
                "betas": betas,
                "subject_dir": None,
            }
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": f"AmassBedlamMatrixGenesis_{args.tag}",
        "shadow": str(shadow),
        "smplx_model_sha256": model_sha,
        "coupled_rbf_support_rad": COUPLED_RBF_SUPPORT_RAD,
        "cameras": list(MATRIX_CAMERAS),
        "publishable": False,
        "vessel_repair_started": False,
        "betas": {},
        "elapsed_seconds": 0.0,
    }

    for beta_spec in beta_specs:
        label = str(beta_spec["label"])
        betas = np.asarray(beta_spec["betas"], dtype=np.float64)
        subject_dir = beta_spec["subject_dir"]
        if subject_dir is not None:
            value = load_whole_chain_rest_fit_v1(
                subject_dir,
                operator=operator,
                calibration=calibration,
                smplx_model=model,
                smplx_model_sha256=model_sha,
                recheck=False,
            )
        else:
            value = build_whole_chain_rest_fit_v1(
                operator,
                calibration,
                betas=betas,
                subject_label=label,
                capture_sha256="matrix_render_ephemeral",
                smplx_model=model,
                smplx_model_sha256=model_sha,
            )
        asset = materialize_subject(operator, betas=betas, gender="male").rigged_asset
        pose_map = build_pose_map_v1(
            value,
            asset=asset,
            calibration=calibration,
            oracle_path=args.oracle.resolve(),
            source_operator_digest=operator.runtime_digest(validate=False),
        )
        beta_out = output / label
        beta_out.mkdir(parents=True, exist_ok=False)
        pose_reports: dict[str, Any] = {}
        focus_paths: list[Path] = []
        focus_labels: list[str] = []
        for motion in motions:
            pose_id = str(motion["pose_id"])
            pose55 = np.asarray(motion["pose55"], dtype=np.float32).reshape(55, 3)
            vertices, _ = pose_whole_chain_vertices(
                value,
                pose_map,
                source_asset=asset,
                pose_axis_angle=pose55,
            )
            skin, skin_faces = smplx_body_surface_v7(
                model, betas=betas, pose_axis_angle=pose55
            )
            frames, _widths, _details = _measure_frames(
                vertices,
                calibration.domains,
                calibration.joint_domain_bases,
                partition="validation",
            )
            pose_out = beta_out / pose_id
            pose_out.mkdir(parents=True, exist_ok=False)
            cell = _render_modes(
                output=pose_out,
                vertices=np.asarray(vertices, dtype=np.float32),
                asset=asset,
                skin=skin,
                skin_faces=skin_faces,
                frames=frames,
                backend=args.backend,
                camera_names=MATRIX_CAMERAS,
            )
            focused = {}
            for layer in ("bones_only", "outside_heatmap", "full_anatomy"):
                rgb_dir = pose_out / layer / "rgb"
                for cam in FOCUS_CAMERAS:
                    rgb = rgb_dir / f"{cam}.png"
                    if rgb.is_file():
                        focused[f"{layer}/{cam}"] = {
                            "path": str(rgb),
                            "sha256": _sha256(rgb),
                        }
            pose_reports[pose_id] = {
                "layers": list(cell.get("layers", {}).keys()),
                "focused": focused,
                "three_layer_contact_sheet": cell.get("three_layer_contact_sheet"),
            }
            # Collect knee/whole focus for beta sheet from capture + first lower motion.
            if pose_id in {"pose_213328", "tpose"} or "lower" in pose_id or pose_id.startswith(
                "amass_"
            ):
                for cam in ("left_knee_ap", "whole_ap"):
                    key = f"full_anatomy/{cam}"
                    if key in focused:
                        focus_paths.append(Path(focused[key]["path"]))
                        focus_labels.append(f"{pose_id}:{cam}")
        sheet = _focus_sheet(
            focus_paths[:12],
            focus_labels[:12],
            beta_out / "knee_full_anatomy_contact_sheet.png",
        )
        report["betas"][label] = {
            "poses": pose_reports,
            "knee_full_anatomy_contact_sheet": str(sheet),
            "knee_full_anatomy_contact_sheet_sha256": _sha256(sheet),
            "n_poses": len(pose_reports),
        }

    report["elapsed_seconds"] = float(time.perf_counter() - started)
    _write_json(output / "manifest.json", report)
    print(
        f"AmassBedlamMatrixGenesis tag={args.tag} betas={len(report['betas'])} "
        f"seconds={report['elapsed_seconds']:.1f} -> {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
