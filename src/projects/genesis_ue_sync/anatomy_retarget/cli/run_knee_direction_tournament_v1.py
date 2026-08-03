"""Run multi-direction knee tournament on 213328 + Genesis slim review."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    _measure_frames,
    load_anatomical_calibration_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.cli.render_stage1_baseline_compare_v1 import (
    _render_modes,
)
from projects.genesis_ue_sync.anatomy_retarget.joint_contact_v7 import (
    FrozenJointMaterialDomainsV7,
)
from projects.genesis_ue_sync.anatomy_retarget.knee_direction_tournament_v1 import (
    _eval_gates,
    build_delta_j_centerline,
    build_inward_shared_t,
    build_patella_only,
    save_tournament_subject,
)
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import easymocap_fit_to_smplx55
from projects.genesis_ue_sync.anatomy_retarget.pose_map_v1 import (
    build_pose_map_v1,
    pose_whole_chain_vertices,
)
from projects.genesis_ue_sync.anatomy_retarget.refit_body_fit_v1 import (
    build_body_fit_pose_catalog_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.refit_weights_v1 import build_weight_refit_v1
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


KEEP_CAMERAS = ("left_knee_ap", "left_knee_lateral", "right_knee_ap", "whole_ap")
KEEP_LAYERS = ("bones_only", "outside_heatmap", "full_anatomy")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pick_poses(catalog: dict[str, np.ndarray], cap_pose: np.ndarray) -> dict:
    keys = ["tpose"]
    walks = sorted(k for k in catalog if k.startswith("amass_") and "walk" in k.lower())
    if walks:
        keys.append(walks[0])
    out = {k: catalog[k] for k in keys if k in catalog}
    out["pose_213328"] = cap_pose
    return out


def _render_slim(
    *,
    label: str,
    value: Any,
    asset: Any,
    output: Path,
    operator: Any,
    calibration: Any,
    model: Any,
    oracle: Path,
    betas: np.ndarray,
    pose: np.ndarray,
    backend: str,
    custom_asset: Any | None = None,
) -> dict[str, Any]:
    use_asset = custom_asset if custom_asset is not None else asset
    pose_map = build_pose_map_v1(
        value,
        asset=use_asset,
        calibration=calibration,
        oracle_path=oracle,
        source_operator_digest=operator.runtime_digest(validate=False),
    )
    vertices, _ = pose_whole_chain_vertices(
        value, pose_map, source_asset=use_asset, pose_axis_angle=pose
    )
    skin, skin_faces = smplx_body_surface_v7(
        model, betas=betas, pose_axis_angle=pose
    )
    frames, _w, _d = _measure_frames(
        vertices,
        calibration.domains,
        calibration.joint_domain_bases,
        partition="validation",
    )
    shadow_out = output / "full" / label
    if shadow_out.exists():
        shutil.rmtree(shadow_out)
    shadow_out.mkdir(parents=True, exist_ok=False)
    cell = _render_modes(
        output=shadow_out,
        vertices=np.asarray(vertices, dtype=np.float32),
        asset=use_asset,
        skin=skin,
        skin_faces=skin_faces,
        frames=frames,
        backend=backend,
    )
    slim: dict[str, Any] = {"layers": {}}
    slim_root = output / "slim" / label
    slim_root.mkdir(parents=True, exist_ok=True)
    for layer in KEEP_LAYERS:
        layer_dir = shadow_out / layer / "rgb"
        kept = {}
        for cam in KEEP_CAMERAS:
            src = layer_dir / f"{cam}.png"
            if not src.is_file():
                continue
            dst = slim_root / layer / f"{cam}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            kept[cam] = {"path": str(dst), "sha256": _sha256(dst)}
        slim["layers"][layer] = kept
    # Drop full render to save disk.
    shutil.rmtree(shadow_out, ignore_errors=True)
    return {"label": label, "render_meta": cell, "slim": slim}


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
        "--smplx-model",
        type=Path,
        default=Path("ref_code_library/EasyMocap/data/smplx/smplx/SMPLX_MALE.pkl"),
    )
    parser.add_argument(
        "--v7-shadow",
        type=Path,
        default=Path("outputs/anatomy_retarget/v8_candidates/chain_retarget_v7_node2_001"),
    )
    parser.add_argument(
        "--v8-shadow",
        type=Path,
        default=Path("outputs/anatomy_retarget/v8_candidates/chain_retarget_v8_node2_001"),
    )
    parser.add_argument(
        "--capture",
        type=Path,
        default=Path("smplx_outputs/20260713_213328/moment_0000/smplx_result.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/anatomy_retarget/v9_candidates/knee_direction_tournament_v1"),
    )
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--max-nfev", type=int, default=36)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output
    if output.exists():
        raise SystemExit(f"output exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    op = load_source_operator((root / args.operator).resolve(), mmap=True)
    cal = load_anatomical_calibration_v1(
        (root / args.calibration).resolve(),
        operator=op,
        required_scope="full_main_chain",
    )
    model_path, model_sha = require_frozen_smplx_male_v7((root / args.smplx_model).resolve())
    model = load_smplx_model_v7(model_path)
    oracle = (root / args.oracle).resolve()
    domains = FrozenJointMaterialDomainsV7.freeze(
        source_bind_vertices=op.template_asset.vertices_rest,
        faces=op.template_asset.faces,
        domains=op.fixed_material_domains,
    )
    catalog = build_body_fit_pose_catalog_v1(repo_root=root, model_path=model_path)
    capture = (root / args.capture).resolve()
    with np.load(capture) as data:
        betas = np.asarray(data["shapes"], dtype=np.float64).reshape(-1)[:10]
        cap_pose = easymocap_fit_to_smplx55(
            data["Rh"], data["poses"], model_path=model_path
        )
    poses = _pick_poses(catalog, cap_pose)
    flex_key = "pose_213328"

    v7 = load_whole_chain_rest_fit_v1(
        (root / args.v7_shadow).resolve() / "subject_213328",
        operator=op,
        calibration=cal,
        smplx_model=model,
        smplx_model_sha256=model_sha,
        recheck=False,
    )
    v7 = replace(v7, subject_label="213328")
    asset = materialize_subject(op, betas=np.asarray(v7.betas), gender="male").rigged_asset

    matrix: dict[str, Any] = {
        "artifact_kind": "KneeDirectionTournamentMatrixV1",
        "publishable": False,
        "authority": "chain_retarget_v7_node2_001",
        "subject": "213328",
        "flex_pose": flex_key,
        "banned": [
            "coupled_knee_se3",
            "femur_axial_scale_main",
            "skel_limb_length_hard_anchor",
            "pca_corrective_net",
        ],
        "directions": [],
        "ranking": [],
    }

    # Collect (name, value, asset_override, build_report)
    candidates: list[tuple[str, Any, Any | None, dict]] = []

    # 1) baseline V7
    print("=== baseline_v7 ===", flush=True)
    candidates.append(("baseline_v7", v7, None, {"direction": "baseline_v7"}))

    # 2) existing V8
    v8_path = (root / args.v8_shadow).resolve() / "subject_213328"
    if v8_path.is_dir():
        print("=== v8_existing ===", flush=True)
        v8 = load_whole_chain_rest_fit_v1(
            v8_path,
            operator=op,
            calibration=cal,
            smplx_model=model,
            smplx_model_sha256=model_sha,
            recheck=False,
        )
        candidates.append(("v8_existing", v8, None, {"direction": "v8_existing"}))
    else:
        print("skip v8_existing (missing)", flush=True)

    # 3) inward shared translation
    print("=== inward_shared_t ===", flush=True)
    val, rep = build_inward_shared_t(
        v7,
        asset=asset,
        domains=domains,
        model=model,
        pose_bundle=poses,
        flex_key=flex_key,
        calibration=cal,
        max_nfev=int(args.max_nfev),
    )
    candidates.append(("inward_shared_t", val, None, rep))

    # 4) patella only
    print("=== patella_only ===", flush=True)
    val, rep = build_patella_only(
        v7,
        asset=asset,
        domains=domains,
        model=model,
        pose_bundle=poses,
        flex_key=flex_key,
        calibration=cal,
        max_nfev=int(args.max_nfev),
    )
    candidates.append(("patella_only", val, None, rep))

    # 5) delta_j centerline
    print("=== delta_j_centerline ===", flush=True)
    val, rep = build_delta_j_centerline(
        v7,
        asset=asset,
        domains=domains,
        model=model,
        pose_bundle=poses,
        flex_key=flex_key,
        calibration=cal,
        max_nfev=int(args.max_nfev),
    )
    candidates.append(("delta_j_centerline", val, None, rep))

    # 6) weight refit (same bind, different LBS)
    print("=== weight_refit ===", flush=True)
    refit = build_weight_refit_v1(
        asset,
        subject_label="213328",
        rest_vertices=np.asarray(v7.vertices_final, dtype=np.float64),
    )
    asset_refit = refit.apply_to_asset(asset)
    candidates.append(
        (
            "weight_refit",
            v7,
            asset_refit,
            {"direction": "weight_refit", "refit_report": refit.build_report},
        )
    )

    render_report: dict[str, Any] = {}
    for name, value, asset_override, build_rep in candidates:
        print(f"--- gate {name} ---", flush=True)
        use_asset = asset_override if asset_override is not None else asset
        # Weight-refit keeps V7 rest/bind; gate uses custom asset for posing.
        gates = _eval_gates(
            value=value,
            v7=v7,
            asset=use_asset,
            calibration=cal,
            domains=domains,
            oracle_path=oracle,
            operator=op,
            flex_pose=poses[flex_key],
            flex_key=flex_key,
            model=model,
            baseline_asset=asset,
        )
        # Strip non-JSON blobs
        gate_json = {
            k: v
            for k, v in gates.items()
            if k
            not in {
                "pose_map",
                "flex_vertices",
                "flex_v7_vertices",
            }
        }
        cell = {
            "direction": name,
            **gate_json,
            "build": {
                k: v
                for k, v in build_rep.items()
                if k not in {"fit"} or True
            },
        }
        # compact build fit
        if "fit" in build_rep:
            cell["build"] = {
                **{k: v for k, v in build_rep.items() if k != "fit"},
                "fit": build_rep["fit"],
            }
        matrix["directions"].append(cell)

        sub = output / "candidates" / name
        if name == "weight_refit":
            # Save V7 geometry + note weights changed at pose time only.
            save_tournament_subject(
                sub,
                value,
                report={**cell, "weights_override": True},
            )
            np.savez_compressed(
                sub / "driver_weights_override.npz",
                driver_indices=np.asarray(use_asset.driver_indices),
                driver_weights=np.asarray(use_asset.driver_weights),
            )
        elif name in {"baseline_v7", "v8_existing"}:
            save_tournament_subject(sub, value, report=cell)
        else:
            save_tournament_subject(sub, value, report=cell)

        if not args.skip_render:
            print(f"--- render {name} ---", flush=True)
            render_report[name] = _render_slim(
                label=name,
                value=value,
                asset=asset,
                output=output,
                operator=op,
                calibration=cal,
                model=model,
                oracle=oracle,
                betas=betas,
                pose=poses[flex_key],
                backend=args.backend,
                custom_asset=asset_override,
            )

    # Ranking: numeric_feasible first, then outside improvement magnitude.
    ranked = sorted(
        matrix["directions"],
        key=lambda c: (
            0 if c.get("numeric_feasible") else 1,
            0 if c.get("contact_passed") else 1,
            float(c.get("flex_femur_outside_cand_m", 1.0)),
        ),
    )
    matrix["ranking"] = [
        {
            "rank": i + 1,
            "direction": c["direction"],
            "numeric_feasible": c.get("numeric_feasible"),
            "contact_passed": c.get("contact_passed"),
            "outside_v7_mm": float(c["flex_femur_outside_v7_m"]) * 1000.0,
            "outside_cand_mm": float(c["flex_femur_outside_cand_m"]) * 1000.0,
            "delta_mm": (
                float(c["flex_femur_outside_cand_m"]) - float(c["flex_femur_outside_v7_m"])
            )
            * 1000.0,
        }
        for i, c in enumerate(ranked)
    ]
    # Feasible = numeric gate only; visual still human/genesis slim.
    feasible = [r for r in matrix["ranking"] if r["numeric_feasible"] and r["direction"] != "baseline_v7"]
    matrix["feasible_numeric"] = [r["direction"] for r in feasible]
    matrix["verdict"] = {
        "authority_unchanged": True,
        "trusted_latest_updated": False,
        "note": (
            "numeric_feasible ≠ image-pass. Inspect slim/*/outside_heatmap/left_knee_ap.png. "
            "Only promote after visual red clears."
        ),
        "best_numeric": matrix["ranking"][0]["direction"] if matrix["ranking"] else None,
    }
    matrix["elapsed_s"] = float(time.perf_counter() - started)
    matrix["renders"] = {
        k: {"slim_root": str(output / "slim" / k)} for k in render_report
    }

    (output / "matrix_manifest.json").write_text(json.dumps(matrix, indent=2) + "\n")
    (output / "render_manifest.json").write_text(json.dumps(render_report, indent=2) + "\n")
    print(json.dumps(matrix["ranking"], indent=2), flush=True)
    print(f"OUTPUT {output}", flush=True)
    print(f"slim -> {output / 'slim'}", flush=True)


if __name__ == "__main__":
    main()
