"""Single-frame SMPL FK vs Genesis capsule link audit (kinematic, zero gravity recommended)."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as Rsci

from bridge.adapters.genesis import xyzw_from_genesis_quat_wxyz
from projects.genesis_ue_sync.sim_platform.datasets import (
    HumanMotionSequence,
    evaluate_smpl_sequence,
    load_amass_sequence,
)
from projects.genesis_ue_sync.sim_platform.embodiments.capsule_drive_canonical import (
    capsule_packed_q_from_smpl_axis_angle,
)
from projects.genesis_ue_sync.sim_platform.embodiments.smpl_capsule_runtime import (
    build_smpl_capsule_embodiment,
    prepare_smpl_capsule_runtime_asset,
)
from projects.genesis_ue_sync.sim_platform.embodiments.smpl_mjcf_retarget import (
    capsule_packed_q_from_smpl_mjcf,
    fk_smpl24_rot_mats,
)
from projects.genesis_ue_sync.sim_platform.embodiments.smpl2urdf import SMPL_PROXY_BODY_NAMES
from projects.genesis_ue_sync.sim_platform.scenes import load_sync_scene_spec
from projects.genesis_ue_sync.sim_platform.scenes.human_bed_fit import fit_human_sequence_to_bed
from projects.genesis_ue_sync.sim_platform.simulation.runtime import (
    GenesisPlatformRuntime,
    GenesisRuntimeConfig,
)


_DEBUG_SESSION_ID = "a6d3ad"
_DEBUG_LOG_PATH = Path("/media/camp/EXT_DRIVE/Among_US/.cursor/debug-a6d3ad.log")


def _debug_capsule_audit_ndjson(
    *,
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any],
    run_id: str = "pre-fix",
) -> None:
    if os.environ.get("AMONGUS_DEBUG_CAPSULE_AUDIT", "").strip() != "1":
        return
    # region agent log
    payload = {
        "sessionId": _DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        _DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except OSError:
        pass
    # endregion agent log


def _urdf_continuous_joint_count(urdf_path: Path) -> int:
    tree = ET.parse(urdf_path)
    return sum(
        1
        for j in tree.getroot().findall("joint")
        if j.attrib.get("type") in ("continuous", "revolute")
    )


@dataclass(frozen=True)
class RootTranslationAudit:
    world_offset_m: list[float]
    npz_trans_plus_offset_m: list[float]
    smpl_joint0_plus_offset_m: list[float]
    delta_trans_minus_joint0_m: list[float]
    delta_norm_trans_minus_joint0_m: float
    packed_q_translation_m: list[float]
    delta_pack_minus_joint0_m: list[float]
    delta_norm_pack_minus_joint0_m: float


def subset_sequence_one_frame(seq: HumanMotionSequence, frame_idx: int) -> HumanMotionSequence:
    fi = int(max(0, min(frame_idx, seq.frame_count - 1)))
    return HumanMotionSequence(
        source_dataset=seq.source_dataset,
        sequence_name=f"{seq.sequence_name}_audit_{fi}",
        source_path=seq.source_path,
        model_type=seq.model_type,
        fps=seq.fps,
        gender=seq.gender,
        betas=np.asarray(seq.betas, dtype=np.float32).copy(),
        poses=np.asarray(seq.poses[fi : fi + 1], dtype=np.float32).copy(),
        trans=np.asarray(seq.trans[fi : fi + 1], dtype=np.float32).copy(),
        image_names=[seq.image_names[fi]] if seq.image_names and fi < len(seq.image_names) else [],
        cam_int=np.asarray(seq.cam_int[fi : fi + 1], dtype=np.float32) if seq.cam_int is not None else None,
        cam_ext=np.asarray(seq.cam_ext[fi : fi + 1], dtype=np.float32) if seq.cam_ext is not None else None,
        metadata=dict(seq.metadata),
    )


def audit_root_sources(
    *,
    seq: HumanMotionSequence,
    frame_idx: int,
    world_off: np.ndarray,
    capsule_root_world_row: np.ndarray,
    q_packed: np.ndarray,
) -> RootTranslationAudit:
    wo = np.asarray(world_off, dtype=np.float64).reshape(3)
    fi = int(max(0, min(frame_idx, seq.frame_count - 1)))
    trans_off = np.asarray(seq.trans[fi, :3], dtype=np.float64) + wo
    qh = np.asarray(q_packed, dtype=np.float64).reshape(-1)
    j0 = np.zeros(3, dtype=np.float64)
    try:
        one = subset_sequence_one_frame(seq, fi)
        _vj, sj = evaluate_smpl_sequence(one, device="cpu", include_vertices=False, include_joints=True)
        del _vj
        if sj is not None:
            j0 = np.asarray(sj[0, 0, :3], dtype=np.float64).reshape(3) + wo
    except Exception:
        pass
    cap_root = np.asarray(capsule_root_world_row, dtype=np.float64).reshape(3)
    d_tr_j0 = trans_off - j0
    d_pk_j0 = qh[:3] - j0
    return RootTranslationAudit(
        world_offset_m=wo.tolist(),
        npz_trans_plus_offset_m=trans_off.tolist(),
        smpl_joint0_plus_offset_m=j0.tolist(),
        delta_trans_minus_joint0_m=d_tr_j0.tolist(),
        delta_norm_trans_minus_joint0_m=float(np.linalg.norm(d_tr_j0)),
        packed_q_translation_m=qh[:3].tolist(),
        delta_pack_minus_joint0_m=d_pk_j0.tolist(),
        delta_norm_pack_minus_joint0_m=float(np.linalg.norm(d_pk_j0)),
    )


def _genesis_dof_probe(robot: Any) -> dict[str, Any]:
    probe: dict[str, Any] = {}
    for attr in ("get_dofs_name", "get_dof_names"):
        fn = getattr(robot, attr, None)
        if callable(fn):
            try:
                val = fn()
                probe[attr] = list(val) if isinstance(val, (list, tuple)) else str(val)
            except Exception as ex:
                probe[attr] = f"<error: {ex}>"
    probe["dof_like_members"] = sorted(x for x in dir(robot) if "dof" in x.lower())
    return probe


def run_capsule_frame_audit(
    *,
    npz_path: Path,
    scene_spec_path: Path | None,
    frame_idx: int,
    cache_dir: Path,
    human_name: str = "patient_audit",
    backend: str = "cpu",
    capsule_force_rewrite: bool = False,
    fit_samples: int = 7,
    support_band_m: float = 0.03,
    center_margin_m: float = 0.05,
    human_center_mode: str = "bed_center",
    use_smpl_joint0_root: bool = True,
    genesis_proxy: str = "urdf",
) -> dict[str, Any]:
    """Build capsule from sequence shape, optionally fit placement to bed, compare SMPL joints to Genesis links."""

    seq = load_amass_sequence(Path(npz_path))
    gp = str(genesis_proxy).strip().lower()
    capsule_asset = prepare_smpl_capsule_runtime_asset(
        seq,
        cache_dir=Path(cache_dir),
        device="cpu",
        force_rewrite=bool(capsule_force_rewrite),
        genesis_proxy=gp,  # type: ignore[arg-type]
    )
    urdf_cont = _urdf_continuous_joint_count(Path(capsule_asset.urdf_path)) if gp == "urdf" else 0
    world_off = np.zeros(3, dtype=np.float32)
    scene_spec = None
    if scene_spec_path is not None and Path(scene_spec_path).is_file():
        scene_spec = load_sync_scene_spec(Path(scene_spec_path))
        target_center_xy = (
            np.asarray(scene_spec.support_surface.pos[:2], dtype=np.float32)
            if human_center_mode == "bed_center" and scene_spec.support_surface is not None
            else np.asarray(scene_spec.resolved_human_anchor()[:2], dtype=np.float32)
        )
        placement = fit_human_sequence_to_bed(
            seq,
            scene_spec=scene_spec,
            proxy_geometry=capsule_asset.proxy_geometry,
            device="cpu",
            sample_count=int(fit_samples),
            support_band_m=float(support_band_m),
            center_margin_m=float(center_margin_m),
            target_center_xy=target_center_xy,
        )
        world_off = np.asarray(placement.world_offset, dtype=np.float32).reshape(3)

    fi = int(max(0, min(frame_idx, seq.frame_count - 1)))
    trans = np.asarray(seq.trans, dtype=np.float32) + world_off
    capsule_root_world = np.asarray(trans, dtype=np.float32).copy()
    capsule_root_source = "npz_trans_plus_world_off"
    if use_smpl_joint0_root:
        try:
            _vj_dbg, joints_all = evaluate_smpl_sequence(
                seq,
                device="cpu",
                include_vertices=False,
                include_joints=True,
            )
            del _vj_dbg
            if (
                joints_all is not None
                and joints_all.ndim == 3
                and int(joints_all.shape[0]) == int(seq.frame_count)
                and int(joints_all.shape[1]) > 0
            ):
                capsule_root_world = np.asarray(joints_all[:, 0, :3], dtype=np.float32) + world_off
                capsule_root_source = "smpl_joint0_plus_world_off"
        except Exception:
            pass

    human_emb = build_smpl_capsule_embodiment(
        name=str(human_name), asset=capsule_asset, fixed_base=False, genesis_proxy=gp  # type: ignore[arg-type]
    )
    n_body = len(human_emb.robot.joint_names)
    free_n = 6
    if gp == "mjcf":
        if capsule_asset.mjcf_dof_layout_path is None:
            raise RuntimeError("genesis_proxy=mjcf requires mjcf_dof_layout_path on asset")
        q = capsule_packed_q_from_smpl_mjcf(
            pose_axis_angle_row=seq.poses[fi],
            root_translation_world_m=capsule_root_world[fi],
            layout_path=capsule_asset.mjcf_dof_layout_path,
        )
    else:
        q = capsule_packed_q_from_smpl_axis_angle(
            pose_axis_angle_row=seq.poses[fi],
            root_translation_world_m=capsule_root_world[fi],
            body_euler_count=n_body,
        )
    root_audit = audit_root_sources(
        seq=seq,
        frame_idx=fi,
        world_off=world_off,
        capsule_root_world_row=capsule_root_world[fi],
        q_packed=q,
    )

    runtime = GenesisPlatformRuntime(
        GenesisRuntimeConfig(
            backend=backend,
            show_viewer=False,
            show_fps=False,
            gravity=(0.0, 0.0, 0.0),
            enable_collision=False,
            enable_self_collision=False,
        )
    )
    runtime.initialize()
    runtime.add_articulated_entity(human_emb, name=str(human_name), pos=(0.0, 0.0, 0.0))
    runtime.build()
    runtime.reset()

    runtime.set_robot_joint_positions(str(human_name), q)
    read0 = np.asarray(runtime.get_robot_joint_positions(str(human_name)), dtype=np.float64).reshape(-1)
    qh = np.asarray(q, dtype=np.float64).reshape(-1)
    err_after_set = float(np.max(np.abs(read0 - qh[: read0.size]))) if read0.size <= qh.size else -1.0

    entity = runtime.entities[str(human_name)]
    dof_probe = _genesis_dof_probe(entity)
    n_dofs = int(getattr(entity, "n_dofs", -1))
    n_qs = int(getattr(entity, "n_qs", -1))

    runtime.step(n=1)
    read1 = np.asarray(runtime.get_robot_joint_positions(str(human_name)), dtype=np.float64).reshape(-1)
    err_after_step = float(np.max(np.abs(read1 - qh[: read1.size]))) if read1.size <= qh.size else -1.0
    err_body_after_step = (
        float(np.max(np.abs(read1[free_n:] - qh[free_n:])))
        if read1.size > free_n and qh.size > free_n and read1.size == qh.size
        else -1.0
    )

    pose_row = np.asarray(seq.poses[fi], dtype=np.float64).reshape(-1)
    Rg = fk_smpl24_rot_mats(pose_row)
    _vj2, joints_eval = evaluate_smpl_sequence(
        subset_sequence_one_frame(seq, fi),
        device="cpu",
        include_vertices=False,
        include_joints=True,
    )
    del _vj2
    link_rows: list[dict[str, Any]] = []
    n_skel = len(SMPL_PROXY_BODY_NAMES)
    if joints_eval is not None:
        jn = int(joints_eval.shape[1])
        lim = min(jn, n_skel)
        if jn < n_skel:
            link_rows.append({"warning": "smpl_joint_count_below_24", "joint_count": jn})
        for ji in range(lim):
            name = SMPL_PROXY_BODY_NAMES[ji]
            smpl_p = (np.asarray(joints_eval[0, ji, :3], dtype=np.float64) + world_off.astype(np.float64)).reshape(3)
            try:
                pose = runtime.get_link_pose(str(human_name), name)
                gpos = np.asarray(pose[:3], dtype=np.float64).reshape(3)
                dist = float(np.linalg.norm(gpos - smpl_p))
                qu_w = np.asarray(pose[3:7], dtype=np.float64).reshape(4)
                qu_xyzw = np.asarray(xyzw_from_genesis_quat_wxyz(tuple(float(x) for x in qu_w)), dtype=np.float64)
                Rgen = np.asarray(Rsci.from_quat(qu_xyzw).as_matrix(), dtype=np.float64)
                Rsm = np.asarray(Rg[int(ji)], dtype=np.float64).reshape(3, 3)
                rot_err_deg = float(
                    np.degrees(np.linalg.norm(Rsci.from_matrix(Rsm.T @ Rgen).as_rotvec()))
                )
            except Exception as ex:
                gpos = None
                dist = -1.0
                rot_err_deg = -1.0
                err = str(ex)
            row: dict[str, Any] = {
                "smpl_joint_name": name,
                "smpl_joint_index": ji,
                "smpl_position_world_m": smpl_p.tolist(),
                "genesis_link_position_world_m": None if gpos is None else gpos.tolist(),
                "position_err_m": dist,
                "orientation_err_deg": rot_err_deg,
            }
            if gpos is None:
                row["genesis_error"] = err
            link_rows.append(row)
    runtime.close()

    expected_packed_len = int(qh.size)
    if gp == "urdf":
        implied_free = int(n_dofs) - int(urdf_cont) if n_dofs >= 0 else None
    else:
        implied_free = int(n_dofs) - max(0, int(n_body) - 6) if n_dofs >= 0 else None
    floating_dof_layout_check = {
        "genesis_proxy": gp,
        "urdf_continuous_joint_count": urdf_cont,
        "n_revolute_from_embodiment_joint_names": int(n_body),
        "expected_packed_len_6_plus_urdf_continuous": int(6 + urdf_cont) if gp == "urdf" else None,
        "expected_packed_len_6_plus_embodiment_revolute": int(6 + n_body) if gp == "urdf" else int(n_body),
        "packed_q_len": int(qh.size),
        "packed_len_matches_6_plus_urdf_continuous": bool(qh.size == 6 + urdf_cont) if gp == "urdf" else None,
        "genesis_n_dofs": n_dofs,
        "packed_len_matches_genesis_n_dofs": bool(int(n_dofs) == qh.size) if n_dofs >= 0 else False,
        "genesis_implied_free_dof_count": implied_free,
    }
    out = {
        "frame_index": fi,
        "genesis_proxy": gp,
        "capsule_runtime_urdf": str(capsule_asset.urdf_path),
        "capsule_mjcf_path": str(capsule_asset.mjcf_path) if capsule_asset.mjcf_path is not None else None,
        "capsule_mjcf_dof_layout_path": str(capsule_asset.mjcf_dof_layout_path)
        if capsule_asset.mjcf_dof_layout_path is not None
        else None,
        "capsule_root_world_source": capsule_root_source,
        "n_revolute_joint_names_from_urdf": n_body,
        "packed_q_len": int(qh.size),
        "expected_packed_len_xyz_plus_euler6_and_body": expected_packed_len,
        "packed_len_matches_expected": bool(qh.size == expected_packed_len),
        "genesis_n_dofs": n_dofs,
        "genesis_n_qs": n_qs,
        "genesis_readback_len_after_set": int(read0.size),
        "max_abs_err_readback_vs_packed_after_set": err_after_set,
        "max_abs_err_readback_vs_packed_after_one_step": err_after_step,
        "max_abs_err_body_only_after_step": err_body_after_step,
        "genesis_dof_probe": dof_probe,
        "root_translation_audit": root_audit.__dict__,
        "link_vs_smpl_joint_positions": link_rows,
        "joint_names_head": list(human_emb.robot.joint_names[:8]),
        "joint_names_tail": list(human_emb.robot.joint_names[-8:]),
        "floating_dof_layout_check": floating_dof_layout_check,
    }
    pelvis_err = None
    for row in link_rows:
        if isinstance(row, dict) and row.get("smpl_joint_name") == "Pelvis" and "position_err_m" in row:
            pelvis_err = row.get("position_err_m")
            break
    _debug_capsule_audit_ndjson(
        hypothesis_id="H1",
        location="capsule_frame_audit.py:run_capsule_frame_audit",
        message="floating_base_dof_layout_vs_packed_q",
        data={
            "packed_q_len": int(qh.size),
            "genesis_n_dofs": n_dofs,
            "urdf_continuous_joint_count": urdf_cont,
            "genesis_implied_free_dof_count": implied_free,
            "packed_len_matches_genesis_n_dofs": floating_dof_layout_check.get("packed_len_matches_genesis_n_dofs"),
            "packed_len_matches_6_plus_urdf_continuous": floating_dof_layout_check.get("packed_len_matches_6_plus_urdf_continuous"),
            "note_if_free7": implied_free == 7,
        },
    )
    _debug_capsule_audit_ndjson(
        hypothesis_id="H2",
        location="capsule_frame_audit.py:run_capsule_frame_audit",
        message="genesis_readback_vs_packed_q",
        data={
            "max_abs_err_after_set": err_after_set,
            "max_abs_err_after_one_step": err_after_step,
            "max_abs_err_body_only_after_step": err_body_after_step,
            "readback_len": int(read0.size),
        },
    )
    _debug_capsule_audit_ndjson(
        hypothesis_id="H3",
        location="capsule_frame_audit.py:run_capsule_frame_audit",
        message="root_translation_pack_vs_smpl_joint0",
        data=dict(root_audit.__dict__),
    )
    _debug_capsule_audit_ndjson(
        hypothesis_id="H4",
        location="capsule_frame_audit.py:run_capsule_frame_audit",
        message="pelvis_link_vs_smpl_fk_position_err_m",
        data={"pelvis_position_err_m": pelvis_err},
    )
    return out
