"""Prepare CT-labelled anatomy and atlas templates on actual SMPL-X rest shapes."""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from geometry import (ANGLES, ATLAS_PACK, BP, CLOUD, NORMALS, OPEN, YS,
                      Slicer, digest, hits, profiles, read_mesh_pack, segments, training_meshes)
from inputs import SMPL_DIR

SMPL = SMPL_DIR.parents[2] / "ref_code_library/InteractVLM/data/body_models/smplx/SMPLX_MALE.npz"
SOURCE_NAMES = ("BodyParts3D right", "BodyParts3D mirrored left", "Open3DModel right", "MedSeg CT right")
CT = 3
# Neutral plus matched broad, slender and mixed-length bodies. These are fixed
# synthetic beta settings of one male SMPL-X model, not additional CT subjects.
BETAS = (
    (0., 0., 0.), (1.7, 0., 0.), (-1.7, 0., 0.),
    (0., 1.7, .4), (0., -1.7, -.4), (1.2, -1.1, .7),
    (-1.2, 1.1, -.7),
)
CT_KEYS = (
    ("23 Right common femoral vein","39 Right femoral vein"),
    ("21 Right common femoral artery","35 Right femoral artery"),
    ("27 Right great saphenous vein",),
    ("29 Right deep femoral vein",),
    ("19 Right superficial epigastric vein(s)",),
    ("36 Right deep femoral artery",),
)


def guides(vertices, joints):
    pair = lambda j,k: joints[[j,k]].copy()
    foot = pair(11,10)
    foot[:,1] = vertices[:,1].min()+2.
    top = pair(2,1)
    top[:,1] = joints[0,1]+24.
    return np.stack([foot,pair(8,7),pair(5,4),pair(2,1),top])


def fixed_profiles(vertices, faces, outline_guides, heights):
    mesh = trimesh.Trimesh(vertices,faces,process=False)
    outline = segments(mesh)
    slicer = Slicer(vertices,faces)
    center = np.column_stack([
        np.interp(heights,outline_guides[:,0,1],outline_guides[:,0,k]) for k in (0,2)
    ])
    radius = np.empty((len(heights),len(ANGLES)))
    for i,y in enumerate(heights):
        cuts = outline(y)
        regions = [r for r in slicer.at(y) if r.xyz[0] < -20. and r.area > 500.]
        if regions:
            c = max(regions,key=lambda r:r.area).xyz[[0,2]]
            cuts = cuts[cuts[:,:,0].mean(axis=1)<0.]
        else:
            c = center[i]
        center[i] = c
        radius[i] = hits(cuts,np.repeat(c[None],len(ANGLES),axis=0),NORMALS).min(axis=1)
    if not np.isfinite(radius).all() or (radius <= 0.).any():
        raise ValueError("A beta body changed the feasible skin chart")
    return center,radius


def move_section(points, c0,rad0,c1,rad1, station):
    local=points-c0[station]
    theta=np.mod(np.arctan2(local[...,1],local[...,0]),2*np.pi)
    u=theta*len(ANGLES)/(2*np.pi)
    a=np.minimum(np.floor(np.nan_to_num(u)).astype(int),len(ANGLES)-1)
    w=u-a
    b=(a+1)%len(ANGLES)
    r0=(1.-w)*rad0[station,a]+w*rad0[station,b]
    r1=(1.-w)*rad1[station,a]+w*rad1[station,b]
    return c1[station]+local*(r1/r0)[...,None]


def generate(out):
    out.mkdir(parents=True,exist_ok=True)
    pack = read_mesh_pack(ATLAS_PACK)
    medseg = {key:trimesh.Trimesh(*pack[f"MedSeg|{key}"],process=False)
              for group in CT_KEYS for key in group}
    refs = training_meshes()
    original = np.stack([profiles(mesh,terms,"mesh")[0] for mesh,terms in
                         zip((*refs,medseg),(BP,BP,OPEN,CT_KEYS),strict=True)])
    with np.load(SMPL,allow_pickle=False) as f:
        base = f["v_template"].astype(float)*1000
        dirs = f["shapedirs"][...,:3].astype(float)*1000
        reg = f["J_regressor"]
        faces = f["f"]
    neutral = trimesh.load_mesh(SMPL_DIR/"smpl_canonical_tpose_neutral.obj",process=False)
    assert np.allclose(base,neutral.vertices*1000.,atol=.0002,rtol=0.)
    g0=guides(base,reg@base)
    c0,r0=fixed_profiles(base,faces,g0,YS)
    all_contacts,all_centers,all_ys,all_origins,all_radii=[],[],[],[],[]
    scale_summary=[]
    for beta in BETAS:
        body=base+np.einsum("vij,j->vi",dirs,beta)
        g1=guides(body,reg@body)
        y=np.interp(YS,g0[:,0,1],g1[:,0,1])
        c,r=fixed_profiles(body,faces,g1,y)
        centers=np.stack([move_section(original[:,s],c0,r0,c,r,s) for s in range(len(YS))],axis=1)
        assert np.array_equal(np.isfinite(centers),np.isfinite(original))
        skin=c[:,None]+r[...,None]*NORMALS[None]
        all_ys.append(y); all_centers.append(centers); all_contacts.append(skin)
        all_origins.append(c); all_radii.append(r)
        scale_summary.append({"beta":beta,"station_span_mm":float(y[0]-y[-1]),
                              "radius_ratio_p05_p50_p95":np.quantile(r/r0,[.05,.5,.95]).tolist()})
    np.savez_compressed(out/"ct_smplx_shapes.npz",ct_derived_centers=np.asarray(all_centers),
                        skin_contacts=np.asarray(all_contacts),body_stations_mm=np.asarray(all_ys),
                        skin_origins=np.asarray(all_origins),skin_radii=np.asarray(all_radii),
                        reference_origins=c0,reference_radii=r0,neutral_stations_mm=YS)
    result={"inputs":{"SMPL-X":str(SMPL),"aligned_vessels":str(ATLAS_PACK)},
            "sources":SOURCE_NAMES,"actual_labelled_CT":"MedSeg proximal right thigh, one subject",
            "shape_setting_count":len(BETAS),"shapes":scale_summary,
            "transport":"matched axial landmarks with skin-normalized polar radius per right-leg slice",
            "blender_loaded":False,
            "file_sha256":digest(out/"ct_smplx_shapes.npz"),
            "native_ct_station_counts":np.isfinite(original[CT]).all(axis=-1).sum(axis=0).tolist()}
    (out/"ct_smplx_shapes.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    generate(parser.parse_args().out)
