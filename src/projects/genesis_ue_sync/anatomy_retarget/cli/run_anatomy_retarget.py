#!/usr/bin/env python3
"""Run the offline Blender anatomy retarget step and optionally publish it to Genesis."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from common.project import project_paths
from projects.genesis_ue_sync.anatomy_retarget.asset_align import normalize_rigged_asset_file
from projects.genesis_ue_sync.anatomy_retarget.blender_retarget_runner import run_retarget
from projects.genesis_ue_sync.anatomy_retarget.canonical_registration import refine_canonical_arap
from projects.genesis_ue_sync.anatomy_retarget.containment import load_body_surface, repair_containment
from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import skin_vertices
from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import (
    easymocap_drive_translation,
    load_easymocap_smplx_fit_drive,
    smplx_pose_hash,
    smplx_shape_hash,
)
from projects.genesis_ue_sync.anatomy_retarget.quality_gate import evaluate_asset_quality, write_quality_report
from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import load_rigged_asset, save_rigged_asset
from projects.genesis_ue_sync.anatomy_retarget.shape_volume import apply_subject_beta_shape
from projects.genesis_ue_sync.anatomy_retarget.leg_material import compute_leg_material_coordinates
from projects.genesis_ue_sync.anatomy_retarget.diagnostics import write_mesh_diagnostics
from projects.genesis_ue_sync.anatomy_retarget.bone_segment_diagnostics import write_bone_segment_diagnostics
from projects.genesis_ue_sync.anatomy_retarget.segment_coupling import (
    bake_segment_coupling,
    refresh_segment_coupling,
    segment_coupling_roundtrip_error,
)
from projects.genesis_ue_sync.anatomy_retarget.source_rebind import source_bind_roundtrip
from projects.genesis_ue_sync.multiview_realtime.track_stream import (
    DEFAULT_ANATOMY_ASSET_PUB_BIND,
    TOPIC_ANATOMY_ASSET_V1,
    anatomy_asset_control_to_dict,
)


def _load_config(path: Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return dict(json.loads(text))
    try:
        import yaml  # type: ignore

        return dict(yaml.safe_load(text) or {})
    except Exception:
        return dict(json.loads(text))


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_key(*paths: Path, extra: str = "") -> str:
    digest = hashlib.sha256(extra.encode("utf-8"))
    for path in paths:
        digest.update(str(Path(path).resolve()).encode("utf-8"))
        digest.update(_file_digest(Path(path)).encode("ascii"))
    return digest.hexdigest()[:24]


def parse_args() -> argparse.Namespace:
    paths = project_paths(__file__)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=paths.configs_root / "anatomy" / "anatomy_retarget.yaml")
    p.add_argument("--canonical-dir", type=Path, default=paths.outputs_root / "anatomy_retarget" / "latest_canonical")
    p.add_argument("--output-dir", type=Path, default=paths.outputs_root / "anatomy_retarget" / "latest_asset")
    p.add_argument("--blend", type=Path, default=None)
    p.add_argument("--force-source-rebake", action="store_true", help="Ignore source/shape retarget caches.")
    p.add_argument("--profile-first-frame", action="store_true", help="Write source/shape/pose/publish timing report.")
    p.add_argument("--motion-npz", type=Path, default=None, help="Exact saved SMPL-X fit for final-pose containment/cache")
    p.add_argument("--timeout-s", type=float, default=900.0)
    p.add_argument("--publish-genesis", action="store_true")
    p.add_argument("--publish-bind", type=str, default=DEFAULT_ANATOMY_ASSET_PUB_BIND)
    p.add_argument("--publish-duration-s", type=float, default=2.0)
    p.add_argument("--publish-rate-hz", type=float, default=10.0)
    p.add_argument("--model-id", type=str, default="patient_anatomy")
    p.add_argument("--color-rgba", type=str, default="0.8,0.05,0.05,0.85")
    p.add_argument(
        "--enforce-quality-gate",
        action="store_true",
        help="Fail the offline bake when a diagnostic quality threshold is exceeded.",
    )
    p.add_argument(
        "--no-quality-gate",
        action="store_true",
        help="Deprecated compatibility flag; quality is diagnostic by default.",
    )
    return p.parse_args()


def _parse_rgba(raw: str) -> tuple[float, float, float, float]:
    vals = [float(v.strip()) for v in str(raw).split(",") if v.strip()]
    if len(vals) != 4:
        raise ValueError(f"Expected color as r,g,b,a, got {raw!r}")
    return tuple(max(0.0, min(1.0, v)) for v in vals)  # type: ignore[return-value]


def _publish_upsert(
    *,
    bind: str,
    model_id: str,
    asset_npz: Path,
    color_rgba: tuple[float, float, float, float],
    duration_s: float,
    rate_hz: float,
) -> int:
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.bind(str(bind))
    payload = anatomy_asset_control_to_dict(
        action="upsert",
        model_id=str(model_id),
        asset_npz=str(asset_npz.resolve()),
        color_rgba=color_rgba,
        timestamp_ns=time.time_ns(),
    )
    topic = TOPIC_ANATOMY_ASSET_V1.encode("utf-8")
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    end = time.time() + max(0.1, float(duration_s))
    interval = 1.0 / max(1.0, float(rate_hz))
    sent = 0
    time.sleep(0.2)
    while time.time() < end:
        sock.send_multipart([topic, body])
        sent += 1
        time.sleep(interval)
    sock.close(0)
    return int(sent)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    started_at = time.perf_counter()
    profile: dict[str, float] = {}
    cfg = _load_config(args.config)
    blend = args.blend or Path(str(cfg.get("blend_path", "")))
    if not blend:
        raise ValueError("Missing anatomy blend path; pass --blend or set blend_path in config.")
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.staging-", dir=str(out_dir.parent)))
    def _preserve_uncommitted_stage() -> None:
        if not stage_dir.exists():
            return
        failed_dir = out_dir.parent / f"{out_dir.name}.failed-{time.strftime('%Y%m%d-%H%M%S')}"
        suffix = 1
        while failed_dir.exists():
            failed_dir = out_dir.parent / f"{out_dir.name}.failed-{time.strftime('%Y%m%d-%H%M%S')}-{suffix}"
            suffix += 1
        try:
            os.replace(stage_dir, failed_dir)
            logging.error("uncommitted anatomy bake preserved at %s", failed_dir)
        except Exception:
            logging.exception("could not preserve failed anatomy staging directory %s", stage_dir)

    atexit.register(_preserve_uncommitted_stage)
    output_npz = stage_dir / "anatomy_rigged.npz"
    output_glb = stage_dir / "anatomy_rigged.glb"
    report_json = stage_dir / "retarget_report.json"
    manifest_path = Path(args.canonical_dir) / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    betas = manifest.get("betas", [])
    gender = str(manifest.get("gender", "male"))
    cache_root = out_dir.parent / "cache_v2"
    source_key = _cache_key(
        Path(blend), Path(args.config), Path(args.canonical_dir) / "smpl_canonical_tpose_neutral.obj",
        Path(__file__).resolve().parents[1] / "blender_scripts" / "blender_retarget_script.py",
        Path(__file__).resolve().parents[1] / "canonical_registration.py",
        Path(__file__).resolve().parents[1] / "segment_coupling.py",
        extra="source-template-v4-segment-coupling-v1",
    )
    shape_hash = smplx_shape_hash(betas, gender=gender) if betas else "neutral"
    source_cache = cache_root / "source_template_v4" / f"{source_key}.npz"
    shape_key = _cache_key(
        Path(args.canonical_dir) / "smpl_canonical_tpose.obj",
        Path(__file__).resolve().parents[1] / "shape_volume.py",
        Path(__file__).resolve().parents[1] / "leg_material.py",
        extra=f"{source_key}:{shape_hash}:subject-shape-v2",
    )
    shape_cache = cache_root / "shape" / f"{shape_key}.npz"
    source_cache_hit = source_cache.is_file() and not args.force_source_rebake
    shape_cache_hit = shape_cache.is_file() and not args.force_source_rebake
    containment_reports: list[dict[str, Any]] = []
    registration_report: dict[str, Any] = {}
    blender_report: dict[str, Any]
    if source_cache_hit:
        asset = load_rigged_asset(source_cache, validate=True)
        cached_meta = dict(asset.metadata or {})
        registration_report = dict(cached_meta.get("registration_report") or {})
        blender_report = dict(cached_meta.get("source_blender_report") or {})
        containment_reports.extend(list(cached_meta.get("source_containment_reports") or []))
        logging.info("source-rig cache hit key=%s", source_key)
    else:
        result = run_retarget(
            blend_path=blend, canonical_dir=args.canonical_dir, mapping_path=args.config,
            output_npz=output_npz, output_glb=output_glb, report_json=report_json,
            timeout_s=float(args.timeout_s),
        )
        if not result.ok:
            logging.error("Blender retarget failed returncode=%s log=%s", result.returncode, result.log_path)
            return int(result.returncode or 1)
        normalize_rigged_asset_file(output_npz, config=cfg, force=False)
        asset = load_rigged_asset(output_npz, validate=True)
        asset, registration_report = refine_canonical_arap(asset)
        if str(cfg.get("canonical_rest_space", "neutral")).lower() == "neutral":
            neutral_surface = load_body_surface(Path(args.canonical_dir) / "smpl_canonical_tpose_neutral.obj")
            asset, neutral_containment = repair_containment(
                asset, surface_vertices=neutral_surface[0], surface_faces=neutral_surface[1],
                stage="neutral_canonical", strict=False,
            )
            containment_reports.append(neutral_containment)
        coupling, coupling_report = bake_segment_coupling(asset)
        coupling_report["roundtrip_error_m"] = segment_coupling_roundtrip_error(asset, coupling)
        asset = type(asset)(**{**asset.__dict__, "source_segment_coupling": coupling})
        blender_report = json.loads(report_json.read_text(encoding="utf-8"))
        source_meta = dict(asset.metadata or {})
        source_meta.update({
            "registration_report": registration_report,
            "source_blender_report": blender_report,
            "source_containment_reports": containment_reports,
            "source_cache_key": source_key,
            "segment_coupling_report": coupling_report,
        })
        asset = type(asset)(**{**asset.__dict__, "metadata": source_meta})
        source_cache.parent.mkdir(parents=True, exist_ok=True)
        save_rigged_asset(source_cache, asset)
        logging.info("source_template_v4 stored key=%s", source_key)
    profile["source_template_s"] = time.perf_counter() - started_at
    bind_roundtrip = source_bind_roundtrip(asset)
    zero_pose_vertices = skin_vertices(asset, np.zeros((55, 3), dtype=np.float32))
    bind_roundtrip["zero_pose_vertex_error_m"] = float(
        np.max(np.linalg.norm(zero_pose_vertices - np.asarray(asset.vertices_rest, dtype=np.float32), axis=1))
    )
    bind_roundtrip["pass"] = bool(bind_roundtrip.get("pass", True) and bind_roundtrip["zero_pose_vertex_error_m"] <= 1.0e-4)
    if not bool(bind_roundtrip.get("pass", True)):
        raise RuntimeError(f"source bind round-trip failed: {bind_roundtrip}")

    source_vertices = (
        np.asarray(asset.registration_reference, dtype=np.float32).copy()
        if asset.registration_reference is not None else asset.vertices_rest.copy()
    )
    subject_surface = load_body_surface(Path(args.canonical_dir) / "smpl_canonical_tpose.obj")
    shape_report: dict[str, Any] = {"backend": "subject_bind_direct"}
    if shape_cache_hit:
        asset = load_rigged_asset(shape_cache, validate=True)
        cached_meta = dict(asset.metadata or {})
        shape_report = dict(cached_meta.get("shape_report") or shape_report)
        containment_reports.extend(list(cached_meta.get("shape_containment_reports") or []))
        logging.info("subject-shape cache hit shape_hash=%s", shape_hash)
    else:
        if str(cfg.get("canonical_rest_space", "neutral")).lower() == "neutral":
            asset, shape_report = apply_subject_beta_shape(asset, canonical_dir=args.canonical_dir)
        asset, subject_containment = repair_containment(
            asset, surface_vertices=subject_surface[0], surface_faces=subject_surface[1],
            stage="subject_beta", strict=False,
        )
        containment_reports.append(subject_containment)
        asset, leg_material_report = compute_leg_material_coordinates(
            asset, skin_vertices=subject_surface[0]
        )
        shape_meta = dict(asset.metadata or {})
        shape_meta.update({
            "shape_report": shape_report,
            "shape_containment_reports": [subject_containment],
            "leg_material_report": leg_material_report,
        })
        asset = type(asset)(**{**asset.__dict__, "metadata": shape_meta})
        shape_cache.parent.mkdir(parents=True, exist_ok=True)
        save_rigged_asset(shape_cache, asset)
        logging.info("subject-shape cache stored shape_hash=%s", shape_hash)
    asset, coupling_report = refresh_segment_coupling(asset)
    meta_coupling = dict(asset.metadata or {})
    meta_coupling["segment_coupling_report"] = coupling_report
    asset = type(asset)(**{**asset.__dict__, "metadata": meta_coupling})
    profile["subject_shape_s"] = time.perf_counter() - started_at - profile["source_template_s"]
    pose_report: dict[str, Any] | None = None
    if args.motion_npz is not None:
        motion_path = Path(args.motion_npz).expanduser().resolve()
        motion = np.load(motion_path)
        motion_betas = np.asarray(motion["shapes"], dtype=np.float32).reshape(-1)[:10]
        motion_shape_hash = smplx_shape_hash(motion_betas, gender=gender)
        expected_shape_hash = smplx_shape_hash(betas, gender=gender) if betas else ""
        if expected_shape_hash and motion_shape_hash != expected_shape_hash:
            raise ValueError(
                f"motion/canonical shape mismatch: motion={motion_shape_hash} canonical={expected_shape_hash}"
            )
        pose55, raw_transl = load_easymocap_smplx_fit_drive(motion_path, gender=gender)
        effective_transl = easymocap_drive_translation(pose55[:3], raw_transl, asset.rest_joints[0])
        cache_hash = smplx_pose_hash(pose55, effective_transl)
        pose_cache = cache_root / "pose" / f"{shape_key}-{cache_hash}.npz"
        if pose_cache.is_file() and not args.force_source_rebake:
            cached_pose = load_rigged_asset(pose_cache, validate=True)
            pose_report = dict((cached_pose.metadata or {}).get("pose_cache_report") or {})
            asset = type(asset)(
                **{
                    **asset.__dict__,
                    "pose_cache_vertices": cached_pose.pose_cache_vertices,
                    "pose_cache_hash": cached_pose.pose_cache_hash,
                }
            )
            logging.info("pose cache hit pose_hash=%s", cache_hash)
        else:
            posed_vertices = skin_vertices(asset, pose55, transl=effective_transl)
            if "vertices" not in motion.files or "faces" not in motion.files:
                raise ValueError(f"{motion_path} must include official posed SMPL-X vertices/faces")
            posed_asset = type(asset)(**{**asset.__dict__, "vertices_rest": posed_vertices})
            repaired_pose, pose_report = repair_containment(
                posed_asset,
                surface_vertices=np.asarray(motion["vertices"], dtype=np.float64).reshape(-1, 3),
                surface_faces=np.asarray(motion["faces"], dtype=np.int32).reshape(-1, 3),
                stage="final_pose",
                # Offline diagnostic refinement; never runs in the online viewer.
                max_iterations=12,
                strict=False,
            )
            asset = type(asset)(
                **{
                    **asset.__dict__,
                    "pose_cache_vertices": np.asarray(repaired_pose.vertices_rest, dtype=np.float32),
                    "pose_cache_hash": cache_hash,
                }
            )
            pose_meta = dict(asset.metadata or {})
            pose_meta["pose_cache_report"] = pose_report
            pose_asset = type(asset)(**{**asset.__dict__, "metadata": pose_meta})
            pose_cache.parent.mkdir(parents=True, exist_ok=True)
            save_rigged_asset(pose_cache, pose_asset)
            logging.info("pose cache stored pose_hash=%s", cache_hash)
        if pose_report is not None:
            containment_reports.append(pose_report)

    meta = dict(asset.metadata or {})
    meta.update(
        {
            "schema_version": 2,
            "gender": gender,
            "betas": betas,
            "shape_hash": smplx_shape_hash(betas, gender=gender) if betas else "",
            "canonical_source": str(manifest.get("source", "")),
            "derived_drivers": cfg.get("derived_drivers", {}),
            "registration_report": registration_report,
            "shape_report": shape_report,
            "containment_reports": containment_reports,
            "pose_cache_report": pose_report,
            "leg_material_report": dict(meta.get("leg_material_report") or {}),
            "segment_coupling_report": dict(meta.get("segment_coupling_report") or {}),
            "source_template_version": 4,
            "source_bind_roundtrip": bind_roundtrip,
        }
    )
    asset = type(asset)(**{**asset.__dict__, "metadata": meta})
    save_rigged_asset(output_npz, asset)
    mesh_diagnostics = write_mesh_diagnostics(
        asset,
        surface_vertices=subject_surface[0],
        surface_faces=subject_surface[1],
        output_path=stage_dir / "anatomy_mesh_diagnostics.json",
    )
    bone_segment_report: dict[str, Any] | None = None
    if args.motion_npz is not None:
        motion_path = Path(args.motion_npz).expanduser().resolve()
        pose55, raw_transl = load_easymocap_smplx_fit_drive(motion_path, gender=gender)
        effective_transl = easymocap_drive_translation(pose55[:3], raw_transl, asset.rest_joints[0])
        bone_segment_report = write_bone_segment_diagnostics(
            asset,
            pose_axis_angle=pose55,
            transl=effective_transl,
            output_path=stage_dir / "bone_segment_diagnostics.json",
            mesh_diagnostics=mesh_diagnostics,
        )
    meta = dict(asset.metadata or {})
    meta["bone_segment_diagnostics"] = bone_segment_report
    asset = type(asset)(**{**asset.__dict__, "metadata": meta})
    profile["pose_and_diagnostics_s"] = time.perf_counter() - started_at - profile["source_template_s"] - profile["subject_shape_s"]
    profile["total_pre_publish_s"] = time.perf_counter() - started_at
    if args.profile_first_frame:
        (stage_dir / "first_frame_profile.json").write_text(
            json.dumps({"seconds": profile, "source_cache_hit": source_cache_hit, "shape_cache_hit": shape_cache_hit}, indent=2),
            encoding="utf-8",
        )
        logging.info("first-frame profile %s", {key: round(value, 3) for key, value in profile.items()})
    tri_edges = np.concatenate(
        (asset.faces[:, [0, 1]], asset.faces[:, [1, 2]], asset.faces[:, [2, 0]]), axis=0
    )
    before_len = np.linalg.norm(
        source_vertices[tri_edges[:, 0]] - source_vertices[tri_edges[:, 1]], axis=1
    )
    after_len = np.linalg.norm(
        asset.vertices_rest[tri_edges[:, 0]] - asset.vertices_rest[tri_edges[:, 1]], axis=1
    )
    valid_edges = before_len > 1.0e-8
    post_ratio = after_len[valid_edges] / before_len[valid_edges]
    blender_report.setdefault("edge_stretch", {}).update(
        {
            "source_to_final_max": float(np.max(post_ratio)),
            "source_to_final_p999": float(np.quantile(post_ratio, 0.999)),
        }
    )
    if asset.pose_cache_vertices is not None:
        cached_len = np.linalg.norm(
            asset.pose_cache_vertices[tri_edges[:, 0]] - asset.pose_cache_vertices[tri_edges[:, 1]], axis=1
        )
        cached_ratio = cached_len[valid_edges] / before_len[valid_edges]
        blender_report["edge_stretch"].update(
            {
                "source_to_pose_cache_max": float(np.max(cached_ratio)),
                "source_to_pose_cache_p999": float(np.quantile(cached_ratio, 0.999)),
            }
        )
    quality = evaluate_asset_quality(
        asset,
        canonical_dir=args.canonical_dir,
        blender_report=blender_report,
        limits=dict(cfg.get("quality_gate", {}) or {}),
    )
    write_quality_report(stage_dir / "quality_report.json", quality)
    if not quality["passed"] and args.enforce_quality_gate:
        failed_dir = out_dir.parent / f"{out_dir.name}.failed-{time.strftime('%Y%m%d-%H%M%S')}"
        os.replace(stage_dir, failed_dir)
        logging.error("quality gate rejected anatomy asset; previous asset remains unchanged")
        for failure in quality["failures"]:
            logging.error("quality: %s", failure)
        logging.error("failed bake diagnostics preserved at %s", failed_dir)
        return 2
    if not quality["passed"]:
        logging.warning(
            "anatomy quality diagnostics have %d warning(s); asset remains publishable. "
            "Use --enforce-quality-gate for offline regression enforcement.",
            len(quality["failures"]),
        )
        for failure in quality["failures"]:
            logging.warning("quality diagnostic: %s", failure)

    previous_dir: Path | None = None
    if out_dir.exists():
        previous_dir = out_dir.parent / f"{out_dir.name}.previous-{time.strftime('%Y%m%d-%H%M%S')}"
        os.replace(out_dir, previous_dir)
    try:
        os.replace(stage_dir, out_dir)
    except Exception:
        if previous_dir is not None and previous_dir.exists() and not out_dir.exists():
            os.replace(previous_dir, out_dir)
        raise
    output_npz = out_dir / "anatomy_rigged.npz"
    logging.info(
        "retarget ok vertices=%s faces=%s joints=%s output=%s",
        asset.vertices_rest.shape[0],
        asset.faces.shape[0],
        len(asset.joint_names),
        output_npz,
    )
    if args.publish_genesis:
        sent = _publish_upsert(
            bind=str(args.publish_bind),
            model_id=str(args.model_id),
            asset_npz=output_npz,
            color_rgba=_parse_rgba(str(args.color_rgba)),
            duration_s=float(args.publish_duration_s),
            rate_hz=float(args.publish_rate_hz),
        )
        logging.info("published anatomy upsert sent=%s bind=%s", sent, args.publish_bind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
