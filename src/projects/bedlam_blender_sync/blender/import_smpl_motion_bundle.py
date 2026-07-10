"""Import a SMPL motion bundle into Blender and optionally export animation FBX."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Euler, Quaternion, Vector

_THIS_FILE = Path(__file__).resolve()
SRC_ROOT = next(parent for parent in (_THIS_FILE.parent, *_THIS_FILE.parents) if parent.name == "src")
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bridge.adapters.blender_bedlam import smpl_yup_to_blender as bridge_smpl_yup_to_blender

DEFAULT_ARMATURE_EULER_FIX_DEG = (math.degrees(math.pi / 2), 0.0, 0.0)
DEFAULT_ARMATURE_NAME = "SMPL24"

# SMPL-X_LH dressed skeletal meshes in UE do not include separate SMPL-24 left_hand/right_hand bones.
# Merge hand local rotation into wrist (chain: R_wrist @ R_hand) and export 22 body joints only.
SMPL_LEFT_WRIST_IDX = 20
SMPL_RIGHT_WRIST_IDX = 21
SMPL_LEFT_HAND_IDX = 22
SMPL_RIGHT_HAND_IDX = 23
UE_SMPL_BODY_JOINT_COUNT = 22
# Keep in sync with scripts/visualization/unreal/ue_bedlam_dual_cam_batch.py (fbx_export_sidecar validation).
FBX_EXPORT_PROFILE = "gs_no_apply_unit_v1"

# SMPL neutral zero-pose joint locations (24, 3), meters — from SMPL_NEUTRAL.pkl pelvis frame.
_REST_J = np.array(
    [
        [-1.79505953e-03, -2.23333446e-01, 2.82191255e-02],
        [6.77246757e-02, -3.14739671e-01, 2.14037877e-02],
        [-6.94655406e-02, -3.13855126e-01, 2.38993038e-02],
        [-4.32792313e-03, -1.14370215e-01, 1.52281192e-03],
        [1.02001221e-01, -6.89938274e-01, 1.69079858e-02],
        [-1.07755594e-01, -6.96424140e-01, 1.50492738e-02],
        [1.15910534e-03, 2.08102144e-02, 2.61528404e-03],
        [8.84055199e-02, -1.08789863e00, -2.67853442e-02],
        [-9.19818258e-02, -1.09483879e00, -2.72625243e-02],
        [2.61610388e-03, 7.37324481e-02, 2.80398521e-02],
        [1.14763659e-01, -1.14368952e00, 9.25030544e-02],
        [-1.17353574e-01, -1.14298274e00, 9.60854266e-02],
        [-1.62284535e-04, 2.87602804e-01, -1.48171829e-02],
        [8.14608431e-02, 1.95481750e-01, -6.04975478e-03],
        [-7.91430834e-02, 1.92565283e-01, -1.05754332e-02],
        [4.98955543e-03, 3.52572414e-01, 3.65317875e-02],
        [1.72437770e-01, 2.25950646e-01, -1.49179062e-02],
        [-1.75155461e-01, 2.25116450e-01, -1.97185045e-02],
        [4.32050017e-01, 2.13178586e-01, -4.23743412e-02],
        [-4.28897421e-01, 2.11787231e-01, -4.11194829e-02],
        [6.81283645e-01, 2.22164620e-01, -4.35452575e-02],
        [-6.84195501e-01, 2.19559526e-01, -4.66786778e-02],
        [7.65325829e-01, 2.14003084e-01, -5.84906248e-02],
        [-7.68817426e-01, 2.13442268e-01, -5.69937621e-02],
    ],
    dtype=np.float32,
)


def _smpl_yup_to_blender(vec: np.ndarray) -> Vector:
    """Map SMPL-ish Y-up positions to Blender Z-up (rotate +90 deg about X)."""
    return Vector(tuple(float(v) for v in bridge_smpl_yup_to_blender(vec).tolist()))


def _safe_name(name: str) -> str:
    s = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    return s or "bone"


# BEDLAM dressed body skeleton uses this root bone name; FBX animation must include its track.
UE_BEDLAM_ROOT_BONE_NAME = "SMPLX-neutral"


def _bone_names_for_joints(joint_names: list[str]) -> list[str]:
    out: list[str] = []
    for i, jn in enumerate(joint_names):
        out.append(UE_BEDLAM_ROOT_BONE_NAME if i == 0 else _safe_name(jn))
    return out


def _should_merge_smpl_hands(names: list[str], parents: list[int]) -> bool:
    if len(names) < SMPL_RIGHT_HAND_IDX + 1 or len(parents) <= SMPL_RIGHT_HAND_IDX:
        return False
    if parents[SMPL_LEFT_HAND_IDX] != SMPL_LEFT_WRIST_IDX or parents[SMPL_RIGHT_HAND_IDX] != SMPL_RIGHT_WRIST_IDX:
        return False
    return True


def _smpl_body_chain_for_ue(
    names: list[str],
    parents: list[int],
    qwxyz: np.ndarray,
) -> tuple[list[str], list[int], np.ndarray, np.ndarray]:
    """Return (body_names, body_parents, body_qwxyz, body_rest_j) for UE SMPLX_LH skeleton import."""
    if not _should_merge_smpl_hands(names, parents):
        rest_n = min(len(names), _REST_J.shape[0])
        return names[:rest_n], parents[:rest_n], qwxyz[:, :rest_n, :].copy(), _REST_J[:rest_n]

    body_names = names[:UE_SMPL_BODY_JOINT_COUNT]
    body_parents = parents[:UE_SMPL_BODY_JOINT_COUNT]
    rest_j = _REST_J[:UE_SMPL_BODY_JOINT_COUNT]
    t_len = qwxyz.shape[0]
    body_q = qwxyz[:, :UE_SMPL_BODY_JOINT_COUNT, :].copy()
    for t in range(t_len):
        qlw = Quaternion(tuple(float(x) for x in qwxyz[t, SMPL_LEFT_WRIST_IDX]))
        qlh = Quaternion(tuple(float(x) for x in qwxyz[t, SMPL_LEFT_HAND_IDX]))
        qc = (qlw @ qlh).normalized()
        body_q[t, SMPL_LEFT_WRIST_IDX] = np.array([qc.w, qc.x, qc.y, qc.z], dtype=np.float32)

        qrw = Quaternion(tuple(float(x) for x in qwxyz[t, SMPL_RIGHT_WRIST_IDX]))
        qrh = Quaternion(tuple(float(x) for x in qwxyz[t, SMPL_RIGHT_HAND_IDX]))
        qcr = (qrw @ qrh).normalized()
        body_q[t, SMPL_RIGHT_WRIST_IDX] = np.array([qcr.w, qcr.x, qcr.y, qcr.z], dtype=np.float32)

    return body_names, body_parents, body_q, rest_j


def _load_bundle(bundle_dir: Path):
    bundle_dir = Path(bundle_dir)
    js = bundle_dir / "smpl_motion_bundle.json"
    nz = bundle_dir / "smpl_motion_bundle.npz"
    if not js.is_file():
        raise FileNotFoundError(f"Missing {js}")
    if not nz.is_file():
        raise FileNotFoundError(f"Missing {nz}")
    meta = json.loads(js.read_text(encoding="utf-8"))
    data = np.load(nz, allow_pickle=True)
    return meta, data


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import SMPL motion bundle and optionally export FBX.")
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--export-fbx", type=Path)
    parser.add_argument("--armature-name", default=DEFAULT_ARMATURE_NAME)
    parser.add_argument(
        "--armature-euler-fix-deg",
        type=float,
        nargs=3,
        default=DEFAULT_ARMATURE_EULER_FIX_DEG,
    )
    parser.add_argument("--clear-existing", action="store_true")
    parser.add_argument(
        "--fbx-global-scale",
        type=float,
        default=100.0,
        help="FBX export scale (100 = meters to Unreal centimeters).",
    )
    return parser.parse_args(argv)


def _build_armature(
    joint_names: list[str],
    parents: list[int],
    rest_j: np.ndarray,
    name: str,
) -> bpy.types.Object:
    arm_data = bpy.data.armatures.new(name + "_data")
    arm_obj = bpy.data.objects.new(name, arm_data)
    bpy.context.collection.objects.link(arm_obj)

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")

    eb = arm_data.edit_bones
    name_map = _bone_names_for_joints(joint_names)
    rest_bl = np.stack([_smpl_yup_to_blender(rest_j[i]) for i in range(len(joint_names))])

    for i, bn in enumerate(name_map):
        bone = eb.new(bn)
        p = parents[i]
        if p < 0:
            head = Vector(rest_bl[i])
            # tail toward spine1 (idx 3) if available
            tail = Vector(rest_bl[3])
            if (tail - head).length < 1e-4:
                tail = head + Vector((0.0, 0.1, 0.0))
        else:
            head = Vector(rest_bl[p])
            tail = Vector(rest_bl[i])
            if (tail - head).length < 1e-4:
                tail = head + Vector((0.0, 0.05, 0.0))
        bone.head = head
        bone.tail = tail

    for i, bn in enumerate(name_map):
        p = parents[i]
        if p >= 0:
            eb[bn].parent = eb[name_map[p]]

    bpy.ops.object.mode_set(mode="OBJECT")
    return arm_obj


def import_smpl_bundle(
    bundle_dir: Path,
    *,
    clear: bool = True,
    armature_name: str = DEFAULT_ARMATURE_NAME,
    armature_euler_fix_deg: tuple[float, float, float] = DEFAULT_ARMATURE_EULER_FIX_DEG,
) -> bpy.types.Object:
    meta, data = _load_bundle(bundle_dir)
    names = list(meta["joint_names"])
    parents = [int(x) for x in meta["joint_parents"]]

    qwxyz = np.asarray(data["joint_quaternions_wxyz"], dtype=np.float32)
    trans = np.asarray(data["root_translation_world"], dtype=np.float32)
    fps = float(meta.get("fps") or 30.0)

    body_names, body_parents, body_qwxyz, rest_j = _smpl_body_chain_for_ue(names, parents, qwxyz)

    if clear:
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()

    arm_obj = _build_armature(body_names, body_parents, rest_j, armature_name)
    arm_obj.rotation_mode = "XYZ"
    arm_obj.rotation_euler = Euler(armature_euler_fix_deg, "XYZ")

    scene = bpy.context.scene
    scene.render.fps = int(round(fps))
    t_len = body_qwxyz.shape[0]
    scene.frame_start = 1
    export_frame_count = max(int(t_len), 2)
    scene.frame_end = export_frame_count

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="POSE")

    name_map = _bone_names_for_joints(body_names)

    for bone_name in name_map:
        pb = arm_obj.pose.bones[bone_name]
        pb.rotation_mode = "QUATERNION"

    for frame in range(1, export_frame_count + 1):
        t = min(frame - 1, t_len - 1)
        bpy.context.scene.frame_set(frame)
        tr = _smpl_yup_to_blender(trans[t])
        for j, bn in enumerate(name_map):
            q = body_qwxyz[t, j]
            quat = Quaternion((float(q[0]), float(q[1]), float(q[2]), float(q[3])))
            pb = arm_obj.pose.bones[bn]
            if j == 0:
                pb.location = tr
            else:
                pb.location = Vector((0.0, 0.0, 0.0))
            pb.rotation_quaternion = quat
            pb.keyframe_insert(data_path="location", frame=frame)
            pb.keyframe_insert(data_path="rotation_quaternion", frame=frame)

    bpy.ops.object.mode_set(mode="OBJECT")
    return arm_obj


def export_armature_fbx(
    armature_object: bpy.types.Object,
    output_path: Path,
    *,
    global_scale: float = 100.0,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    armature_object.select_set(True)
    bpy.context.view_layer.objects.active = armature_object
    gs = float(global_scale)
    fbx_kw = dict(
        filepath=str(output_path),
        use_selection=True,
        object_types={"ARMATURE"},
        add_leaf_bones=False,
        global_scale=gs,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_force_startend_keying=True,
        bake_anim_simplify_factor=0.0,
    )
    bpy.ops.export_scene.fbx(**fbx_kw)


def main() -> None:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    args = _parse_args(argv)
    bdir = args.bundle_dir.resolve()
    armature = import_smpl_bundle(
        bdir,
        clear=bool(args.clear_existing),
        armature_name=str(args.armature_name),
        armature_euler_fix_deg=tuple(float(v) for v in args.armature_euler_fix_deg),
    )
    if args.export_fbx is not None:
        gs = float(args.fbx_global_scale)
        export_armature_fbx(
            armature,
            args.export_fbx,
            global_scale=gs,
        )
        (bdir / "fbx_export_sidecar.json").write_text(
            json.dumps({"fbx_global_scale": gs, "export_profile": FBX_EXPORT_PROFILE}, indent=2),
            encoding="utf-8",
        )
        print(f"Exported SMPL FBX to {args.export_fbx.resolve()}")
    else:
        print(f"Imported SMPL motion from {args.bundle_dir.resolve()}")


if __name__ == "__main__":
    main()
