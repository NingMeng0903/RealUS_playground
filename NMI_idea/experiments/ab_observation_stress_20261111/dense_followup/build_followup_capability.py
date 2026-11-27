"""Build the coarse/dense static capability fixtures for the resolution follow-up.

The coarse arm is a two-context projection of the frozen ``CAPABILITY_STRESS``
fixture.  The dense arm keeps those endpoint witnesses byte-for-byte as far as
JSON permits and calls the frozen capability evaluator only for newly added
poses.  A failed finite search is recorded as such; it is never an
infeasibility claim.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
STRESS = HERE.parent
PARENT_CAPABILITY = STRESS / "CAPABILITY_STRESS.json"
PARENT_BUILDER = STRESS / "build_capability.py"
LADDER = STRESS.parent / "ab_module_ladder_20261110"
LAYOUT = LADDER / "planning/native_regression_20261110/public_prior_layout/SELECTED.json"
CAPABILITY_DIR = LADDER / "capability"
if str(CAPABILITY_DIR) not in sys.path:
    sys.path.insert(0, str(CAPABILITY_DIR))
from capability import CapabilityEvaluator  # noqa: E402


CONTEXT_INDICES = (0, 2)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _world_poses(local: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    transforms = np.broadcast_to(np.eye(4), (len(local), 4, 4)).copy()
    transforms[:, :2, 3] = local[:, :2]
    transforms[:, :3, :3] = Rotation.from_euler("z", local[:, 2]).as_matrix()
    return anchor @ transforms


def _coarse_poses(parent: dict) -> np.ndarray:
    return np.asarray(parent["poses"], dtype=float)


def _dense_poses(parent: dict) -> np.ndarray:
    y = np.arange(-0.032, 0.032 + 0.001, 0.002, dtype=float)
    yaw = np.arange(-0.68, 0.68 + 0.001, 0.17, dtype=float)
    lattice = np.asarray([(0.0, yy, aa) for yy in y for aa in yaw], dtype=float)
    combined = np.vstack((lattice, np.asarray(parent["poses"], dtype=float)))
    return np.unique(np.round(combined, 9), axis=0)


def _context_copy(parent_context: dict, parent_pose_count: int) -> dict:
    """Copy one original witness context without recomputing any endpoint."""
    context = copy.deepcopy(parent_context)
    context["available"] = list(context["available"])
    context["actual_available_mask"] = list(context["available"])
    context["witness_origin"] = "parent CAPABILITY_STRESS.json; no endpoint recomputation"
    # ``joint_time_s`` is deliberately retained from the original fixture.
    if len(context["available"]) != parent_pose_count:
        raise ValueError("parent witness context has unexpected pose count")
    return context


def _make_context(
    parent_context: dict,
    parent_poses: np.ndarray,
    dense_poses: np.ndarray,
    evaluator: CapabilityEvaluator,
    q_start: np.ndarray,
    vmax: np.ndarray,
    parent_index: int,
) -> dict:
    """Build one dense context, reusing all parent pose witnesses."""
    parent_pose_keys = {
        tuple(np.round(pose, 9)): index for index, pose in enumerate(parent_poses)
    }
    anchor = np.asarray(parent_context["T_world_body"], dtype=float)
    initial_index = int(np.flatnonzero(np.all(np.isclose(dense_poses, [0., 0., 0.], atol=0.), axis=1))[0])
    parent_initial = int(parent_context["initial_index"])
    center_entry = parent_context["endpoint_records"][parent_initial]
    if not center_entry.get("found", False):
        raise ValueError(f"parent context {parent_context['id']} lacks a center witness")
    center_q = np.asarray(center_entry["q_solution"], dtype=float)
    if not np.array_equal(center_q, q_start):
        raise ValueError(f"parent context {parent_context['id']} changed its q start")

    entries = []
    reused = []
    targets = _world_poses(dense_poses, anchor)
    for index, pose in enumerate(dense_poses):
        key = tuple(np.round(pose, 9))
        if key in parent_pose_keys:
            parent_pose_index = parent_pose_keys[key]
            entry = copy.deepcopy(parent_context["endpoint_records"][parent_pose_index])
            entry["witness_origin"] = "parent CAPABILITY_STRESS.json"
            entry["parent_pose_index"] = int(parent_pose_index)
            reused.append(True)
        else:
            # This is the exact static FK/URDF/limits/configured self-collision
            # search used by build_capability.py.  A miss carries no-path_claim
            # false in the evaluator output and is retained as evidence only.
            entry = evaluator._static_pose(targets[index], float(q_start[0]), center_q)
            entry["witness_origin"] = "dense follow-up _static_pose"
            entry["parent_pose_index"] = None
            if not entry["found"]:
                entry["failed_search_is_infeasibility_certificate"] = False
                entry["failure_not_inexistence_proof"] = True
            reused.append(False)
        entries.append(entry)

    # Keep the serialized q witness contract honest: unavailable endpoints
    # remain ``None`` just as in the parent fixture.  A center-q placeholder is
    # used only in the separate joint-time calculation, never as a witness.
    q_kin = [entry.get("q_solution") if entry.get("found", False) else None for entry in entries]
    q_rows = np.asarray([
        row if row is not None else center_q for row in q_kin
    ], dtype=float)
    available = np.asarray([entry.get("found", False) for entry in entries], dtype=bool)
    joint_time = np.max(
        np.abs(q_rows[:, None] - q_rows[None, :]) / vmax[None, None, :], axis=-1
    )
    # The zero pose is common and must remain the original start exactly.
    if not np.array_equal(q_rows[initial_index], q_start):
        raise ValueError(f"dense context {parent_context['id']} changed center q")
    context = {
        "id": parent_context["id"],
        "available": available.tolist(),
        "actual_available_mask": available.tolist(),
        "initial_index": initial_index,
        "start_found": bool(available[initial_index]),
        "T_world_body": anchor.tolist(),
        "fixed_rail_m": float(q_start[0]),
        "q_start": q_start.tolist(),
        "q_kin": q_kin,
        "joint_time_s": joint_time.tolist(),
        "endpoint_records": entries,
        "witness_origin": {
            "parent_context_index": int(parent_index),
            "common_pose_count": int(sum(reused)),
            "new_pose_count": int(len(reused) - sum(reused)),
            "common_pose_witnesses_reused": True,
            "new_pose_search": "CapabilityEvaluator._static_pose with center q nearest seed",
            "failed_search_is_infeasibility_certificate": False,
        },
    }
    return context


def _input_hashes(arm: Path, evaluator: CapabilityEvaluator) -> dict[str, str]:
    paths = [
        PARENT_CAPABILITY,
        PARENT_BUILDER,
        LAYOUT,
        CAPABILITY_DIR / "capability.py",
        STRESS / "PROTOCOL.json",
    ]
    arm_protocol = arm / "PROTOCOL.json"
    if arm_protocol.exists():
        paths.append(arm_protocol)
    # Keep the path keys explicit: they are useful for replay audits even when
    # one path is a durable symlink to the frozen parent source.
    hashes = {str(path): _sha(path) for path in paths}
    hashes["capability_evaluator_manifest"] = hashlib.sha256(
        json.dumps(evaluator.manifest(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return hashes


def build_arm(arm_name: str) -> Path:
    if arm_name not in {"coarse", "dense"}:
        raise ValueError(arm_name)
    arm = HERE / arm_name
    parent = json.loads(PARENT_CAPABILITY.read_text())
    parent_poses = _coarse_poses(parent)
    dense_poses = _dense_poses(parent)
    if len(parent_poses) != 31 or len(dense_poses) != 305:
        raise ValueError(f"unexpected pose counts: coarse={len(parent_poses)} dense={len(dense_poses)}")
    layout = json.loads(LAYOUT.read_text())
    layout_q_seed = np.asarray(layout["q_seed"], dtype=float)
    evaluator = CapabilityEvaluator()
    vmax = np.asarray(getattr(evaluator.kin, "q_velocity_limits", [.4] + [2.512] * 7), dtype=float)
    if layout_q_seed.shape != (8,) or not np.all(np.isfinite(layout_q_seed)):
        raise ValueError("layout q_seed must be a finite length-8 start")
    selected_contexts = []
    q_start_by_context = {}
    for index in CONTEXT_INDICES:
        parent_context = parent["contexts"][index]
        context_initial = int(parent_context["initial_index"])
        q_start = np.asarray(
            parent_context["endpoint_records"][context_initial]["q_solution"],
            dtype=float,
        )
        if q_start.shape != (8,) or not np.all(np.isfinite(q_start)):
            raise ValueError(f"parent context {parent_context['id']} lacks a finite center q start")
        q_start_by_context[parent_context["id"]] = q_start.tolist()
        if arm_name == "coarse":
            context = _context_copy(parent_context, len(parent_poses))
            context["parent_context_index"] = int(index)
            context["fixed_rail_m"] = float(q_start[0])
            context["q_start"] = q_start.tolist()
            context["actual_available_mask"] = list(context["available"])
            selected_contexts.append(context)
        else:
            selected_contexts.append(_make_context(
                parent_context, parent_poses, dense_poses, evaluator, q_start, vmax, index,
            ))

    poses = parent_poses if arm_name == "coarse" else dense_poses
    out = {
        "schema": "ab_capability_resolution_followup_v1",
        "scope": "Public finite FK/URDF/limits/configured self-collision endpoint witnesses; finite IK search is evidence, not an infeasibility proof.",
        "resolution_arm": arm_name,
        "poses": poses.tolist(),
        "contexts": selected_contexts,
        "joint_velocity_limits": vmax.tolist(),
        "fixed_rail_m": float(layout_q_seed[0]),
        "q_start_by_context": q_start_by_context,
        "common_pose_witness_source": str(PARENT_CAPABILITY),
        "new_pose_source": None if arm_name == "coarse" else "CapabilityEvaluator._static_pose",
        "robot": evaluator.manifest(),
        "input_sha256": _input_hashes(arm, evaluator),
        "pose_construction": {
            "coarse": "direct projection of parent CAPABILITY_STRESS poses and endpoint records",
            "dense": "x=0, y=-.032:.002:.032, yaw=-.68:.17:.68 plus parent 31; round(9), unique",
            "dense_count": 305,
            "coarse_count": 31,
        },
    }
    output = arm / "CAPABILITY.json"
    _write(output, out)
    _write(arm / "CAPABILITY_BUILD_SUMMARY.json", {
        "resolution_arm": arm_name,
        "pose_count": len(poses),
        "contexts": [
            {
                "id": context["id"],
                "parent_context_index": (
                    context.get("parent_context_index")
                    if "parent_context_index" in context
                    else context["witness_origin"].get("parent_context_index")
                    if isinstance(context.get("witness_origin"), dict)
                    else None
                ),
                "available_count": int(sum(context["available"])),
                "actual_available_mask": context["actual_available_mask"],
                "common_pose_witnesses_reused": context.get("witness_origin", "").get("common_pose_witnesses_reused", arm_name == "coarse") if isinstance(context.get("witness_origin"), dict) else True,
                "new_pose_count": context.get("witness_origin", {}).get("new_pose_count", 0) if isinstance(context.get("witness_origin"), dict) else 0,
                "failed_new_search_count": sum(
                    bool(e.get("failure_not_inexistence_proof", False))
                    for e in context["endpoint_records"]
                ),
            }
            for context in selected_contexts
        ],
        "fixed_rail_m": float(layout_q_seed[0]),
        "q_start_by_context": q_start_by_context,
        "input_sha256": out["input_sha256"],
    })
    return output


def main() -> None:
    for arm in ("coarse", "dense"):
        output = build_arm(arm)
        print(f"built {output}", flush=True)


if __name__ == "__main__":
    main()
