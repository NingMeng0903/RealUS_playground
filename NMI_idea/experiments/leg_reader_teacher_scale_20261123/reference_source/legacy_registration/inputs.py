"""Read the archived SMPL-X retarget and the native named reference geometries."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "active_observation_benchmark"
DRIVE = Path("/media/camp/PEI_T7")
PREVIOUS = DRIVE / "SMPLX_Vascular_Retarget_Audit"
CLOUD = DRIVE / "SMPLX_Vascular_Probability_Cloud"
INDEPENDENT = DRIVE / "SMPLX_Independent_Vascular_Comparison"
for folder in (
    BENCHMARK, CLOUD / "reproduction", INDEPENDENT / "reproduction", PREVIOUS / "reproduction",
):
    sys.path.insert(0, str(folder))

from functools import lru_cache
import json

import numpy as np
import trimesh

from align_cloud_sources import SMPL_DIR, map_to_body, read_mesh_pack, section_outline, target_model
from geometry_audit import Slicer
from register_sources import read_line_groups, shape_controls

SKIN = {
    k: trimesh.load_mesh(SMPL_DIR / f"smpl_canonical_tpose{suffix}.obj", process=False)
    for k, suffix in (("subject", ""), ("neutral", "_neutral"))
}
for surface in SKIN.values():
    surface.vertices *= 1000.
skeleton = json.loads((SMPL_DIR / "smpl_canonical_skeleton.json").read_text())
MORPH = map_to_body(shape_controls(skeleton, "subject", SKIN["subject"]),
                    shape_controls(skeleton, "neutral", SKIN["neutral"]),
                    section_outline(SKIN["subject"]), section_outline(SKIN["neutral"]))

DEFAULT_OUT = ROOT / "outputs" / "leg_reference_transfer"
SCAN_OUT = Path(os.environ.get("LEG_TRANSFER_OUT",DEFAULT_OUT))
ATLAS_PACK = Path(os.environ.get("LEG_TRANSFER_MESHES",CLOUD/"aligned_vessel_surfaces.npz"))
ARCHIVED_ATLAS_PACK = CLOUD / "reproduction" / "inputs" / "aligned_vessel_surfaces_pre_containment.npz"
PRE_CONTAINMENT_PACK = ARCHIVED_ATLAS_PACK if ARCHIVED_ATLAS_PACK.is_file() else CLOUD / "aligned_vessel_surfaces.npz"


@lru_cache(maxsize=2)
def meshes(pack=ATLAS_PACK):
    return {
        name: trimesh.Trimesh(v, f, process=False)
        for name, (v, f) in read_mesh_pack(pack).items()
        if name.split("|")[0] in {"BodyParts3D", "Open3DModel"}
    }


@lru_cache(maxsize=1)
def blender_lines():
    return read_line_groups(SMPL_DIR / "vessel_centerlines_rest.obj")


def neutral_lines():
    return {k: MORPH.transform(p.copy()) for k, p in blender_lines().items()}


def side_mesh(mesh, side):
    sign = -1.0 if side == "R" else 1.0
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    if any(sign * v[:, 0] < -15.) and any(sign * v[:, 0] > 15.):
        keep = np.median(sign * v[f, 0], axis=1) > 0.
        f = f[keep]
        ii, inverse = np.unique(f, return_inverse=True)
        return trimesh.Trimesh(v[ii], inverse.reshape(-1, 3), process=False)
    return mesh


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


AR_NAMES = {
    'EIA': '髂外动脉', 'CFA': '股总动脉', 'FA': '股动脉主干',
    'DFA': '股深动脉候选', 'PA': '腘动脉',
    'CALF_M': '小腿内侧候选', 'CALF_L': '小腿外侧候选',
    'ATA': '胫前动脉', 'PTA': '胫后动脉', 'FIBA': '腓动脉',
}
def artery_references(side):
    low = 'left' if side == 'L' else 'right'
    return {
        'EIA': [f'BodyParts3D|{low} external iliac artery', 'Open3DModel|External_iliac_artery.r' if side=='R' else None],
        'CFA': [f'BodyParts3D|{low} common femoral artery', 'Open3DModel|Common_femoral_artery.r' if side=='R' else None],
        'FA': [f'BodyParts3D|{low} femoral artery','Open3DModel|Femoral_artery.r' if side=='R' else None],
        'DFA': [f'BodyParts3D|{low} deep femoral artery','Open3DModel|Deep_artery_of_the_thigh.r' if side=='R' else None],
        'PA': [f'BodyParts3D|{low} popliteal artery',
               'Open3DModel|Popliteal_artery.r' if side=='R' else None],
        'ATA': [f'BodyParts3D|{low} anterior tibial artery','Open3DModel|Anterior_tibial_artery.r' if side=='R' else None],
        'PTA': [f'BodyParts3D|{low} posterior tibial artery','Open3DModel|Posterior_tibial_artery.r' if side=='R' else None],
        'FIBA': ['Open3DModel|Fibular_artery.r'] if side=='R' else [],
    }

