"""The three reference geometries and a held-out registered Blender leg."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import trimesh
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inputs import ATLAS_PACK, CLOUD, MORPH, PREVIOUS, SKIN, Slicer, meshes, neutral_lines, read_mesh_pack, target_model
from geometry_audit import y_centers
from align_cloud_sources import section_outline

J = CLOUD / "right_leg_joint"
NEW_OPEN = J / "shared_field_test/right_leg_open3dmodel_shared_field.npz"
OCCUPANCY = J / "bonehub_right_leg_occupancy.npz"
BLENDER_BONES = PREVIOUS / "evidence/geometry_stages.npz"
SOURCES = ("BodyParts3D-right", "BodyParts3D-left-mirrored", "Open3DModel-right", "Blender-right")
TRAIN = slice(0,3)
TEST = 3
YS = np.arange(-510.,-1281.,-5.)
ANGLES = np.arange(24)*2*np.pi/24
NORMALS = np.column_stack((np.cos(ANGLES), np.sin(ANGLES)))
TANGENTS = NORMALS[:,::-1] * [-1.,1.]
LABELS = ("deep_target", "arterial_competitor", "great_saphenous", "deep_femoral_or_fibular", "small_saphenous_or_other", "unmodeled_blender_vessel")
TASKS = {
    "saphenofemoral": (-525, -590, (0,1,2,3)),
    "femoral": (-595, -790, (0,1,2,3)),
    "popliteal": (-835, -875, (0,1,2,3)),
    "calf": (-980, -1255, (0,1,2,3)),
    "groin_to_distal_calf": (-525, -1255, (0,1,2,3,4)),
}
# Entries are candidate identities in the anatomy prior. Axial position is free
# to shift, while incidence and arterial/venous identity remain fixed.
GRAPH = {
    "deep_target": ["common femoral vein", "femoral vein", "popliteal vein", "posterior tibial vein"],
    "known_venous_incidence": [["great saphenous vein", "deep_target", "saphenofemoral"],
                              ["deep femoral vein", "deep_target", "proximal thigh"]],
    "calf_relation": "fibular vein is separate in the observed slice; no new edge emitted",
}
BP = (
    ("common femoral vein", "femoral vein", "popliteal vein", "posterior tibial vein"),
    ("common femoral artery", "femoral artery", "popliteal artery", "posterior tibial artery"),
    ("great saphenous vein",),
    ("deep femoral vein", "fibular vein"),
    ("small saphenous vein", "anterior tibial vein"), (),
)
OPEN = (
    ("Common_femoral_vein.r", "Femoral_vein.r", "Popliteal_vein.r", "Posterior_tibial_vein.r"),
    ("Common_femoral_artery.r","Femoral_artery.r","Popliteal_artery.r","Posterior_tibial_artery.r"),
    ("Great_saphenous_vein.r",),
    ("Deep_femoral_vein.r","Fibular_vein.r"),
    ("Small_saphenous_vein.r","Anterior_tibial_vein.r"), (),
)
BLENDER = (
    ("R_COM_FEM_V","R_SUPFEMV","R_POPV","R_POST_TIB_V"),
    ("R_CFA","R_FA","R_PA","R_CALF_M"),
    ("R_SAPH_V",),
    ("R_DEEP_FEM_V","R_PERONEAL_V"),
    (), ("R_CALF_L",),
)


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest() if hasattr(hashlib,"file_digest") else hashlib.sha256(f.read()).hexdigest()


def segments(surface):
    cut = section_outline(surface)
    def at(y):
        try:
            return cut(y)[:,[0,2]].reshape(-1,2,2)
        except ValueError:
            return np.empty((0,2,2))
    return at


def cross(a,b):
    return a[...,0]*b[...,1]-a[...,1]*b[...,0]


def hits(edges, origins, directions):
    # Positive ray lengths intersecting any segment.
    a = edges[None,:,0] - origins[:,None]
    e = np.diff(edges,axis=1)[None,:,0]
    d = directions[:,None]
    den = cross(d,e)
    safe = np.where(abs(den)>1.e-9,den,1.)
    t,u = cross(a,e)/safe,cross(a,d)/safe
    return np.where((abs(den)>1.e-9)&(t>1.e-5)&(u>=0.)&(u<1.),t,np.inf)


def profiles(surface, source, mode, stations=YS):
    arrays = np.full((len(stations),len(source),2),np.nan)
    present = {}
    for j, group in enumerate(source):
        # A route has anatomically specified continuation order. For
        # overlapping file extents, select the available later member at its
        # first support rather than fit a new branch to the overlap.
        for key in group:
            if key not in surface:
                continue
            if mode == "line":
                cuts = y_centers(surface[key],stations)
                candidate = {y:np.median(p[:,[0,2]],axis=0) for y,p in cuts.items()}
            else:
                mesh = surface[key]
                slicer = Slicer(np.asarray(mesh.vertices),np.asarray(mesh.faces))
                candidate = {}
                for y in stations:
                    ring=max((r for r in slicer.at(y,min_area=.6) if r.xyz[0] < 0),
                             key=lambda r:r.area,default=None)
                    if ring is not None:
                        candidate[y] = ring.xyz[[0,2]]
            present[key] = len(candidate)
            for i,y in enumerate(stations):
                if y in candidate:
                    arrays[i,j] = candidate[y]
    return arrays,present


def bone_profile(stations=YS):
    with np.load(BLENDER_BONES) as d:
        models = {}
        for bone in ("Femur","Tibia","Fibula","Patella"):
            key=f"{bone}_R"
            geom=trimesh.Trimesh(d[f"{key}_retarget_mm"],d[f"{key}_faces"],process=True)
            if not geom.is_watertight:
                raise ValueError(f"Unclosed Blender bone {key}")
            models[key]=segments(trimesh.Trimesh(MORPH.transform(geom.vertices.copy()),geom.faces,process=False))
    return [np.concatenate([m(y) for m in models.values()],axis=0) for y in stations]


def surface_profiles():
    skin, guides = target_model()
    center = np.column_stack([np.interp(YS,guides[:,0,1],guides[:,0,j]) for j in (0,2)])
    skin_sections = segments(skin)
    slicer = Slicer(np.asarray(skin.vertices), np.asarray(skin.faces))
    contacts = np.zeros((len(YS),len(ANGLES),2))
    for i,(y,c) in enumerate(zip(YS,center)):
        contour = skin_sections(y)
        right = [r for r in slicer.at(y) if r.xyz[0] < -20 and r.area > 500]
        if right:
            contour = contour[np.mean(contour[:,:,0],axis=1)<0]
            c = max(right,key=lambda r:r.area).xyz[[0,2]]
        t=hits(contour,np.broadcast_to(c,(len(ANGLES),2)),NORMALS).min(axis=1)
        if not np.isfinite(t).all():
            raise ValueError(f"Skin contact failed at {y}")
        contacts[i]=c+t[:,None]*NORMALS
    return contacts


def bone_prior():
    with np.load(OCCUPANCY) as a:
        return RegularGridInterpolator((a["x_mm"],a["y_mm"],a["z_mm"]),
                                       a["count"]/int(a["cohort_size"]),
                                       bounds_error=False,fill_value=np.nan)


def training_meshes():
    archive=meshes()
    right = {term:archive[f"BodyParts3D|right {term}"] for group in BP for term in group
             if f"BodyParts3D|right {term}" in archive}
    left = {}
    for group in BP:
        for term in group:
            key=f"BodyParts3D|left {term}"
            if key in archive:
                geo=archive[key]
                left[term]=trimesh.Trimesh(np.asarray(geo.vertices)*[-1.,1.,1.],geo.faces[:,::-1],process=False)
    other={k:trimesh.Trimesh(v,f,process=False) for k,(v,f) in read_mesh_pack(NEW_OPEN).items()}
    return right,left,other


def assemble(out):
    right,left,other=training_meshes()
    arterial_file=PREVIOUS/"evidence/arterial_paths.json"
    artery_paths=json.loads(arterial_file.read_text())
    blender=neutral_lines() | {k:np.asarray(v["neutral"]) for k,v in artery_paths.items() if k.startswith("R_")}
    layers=[]; counts={}
    for label,sources,terms,mode in zip(SOURCES,(right,left,other,blender),(BP,BP,OPEN,BLENDER),
                                        ("mesh","mesh","mesh","line"),strict=True):
        data,rows=profiles(sources,terms,mode)
        layers.append(data); counts[label]=rows
    contacts=surface_profiles()
    bones=bone_profile()
    bdist=np.full((len(YS),len(ANGLES),len(LABELS)),np.inf)
    inside=np.zeros((len(YS),len(LABELS)),dtype=bool)
    for i,(origin,edge,pos) in enumerate(zip(contacts,bones,layers[TEST])):
        if not len(edge):
            continue
        have=np.isfinite(pos).all(axis=1)
        if not have.any():
            continue
        good=np.flatnonzero(have)
        points=pos[have]
        ray=hits(edge,points,np.broadcast_to([.994,.109],points.shape))
        inside[i,good]=np.isfinite(ray).sum(axis=1)%2==1
        # Linear array rays originate along the width of the transducer.
        lat=(points[None]-origin[:,None])*TANGENTS[:,None]
        along=lat.sum(axis=-1)
        local_origins=origin[:,None]+along[...,None]*TANGENTS[:,None]
        directions=np.broadcast_to(-NORMALS[:,None],local_origins.shape)
        distance=hits(edge,local_origins.reshape(-1,2),directions.reshape(-1,2)).min(axis=1)
        bdist[i][:,good]=distance.reshape(len(ANGLES),len(points))
    paths=[ATLAS_PACK,NEW_OPEN,OCCUPANCY,BLENDER_BONES,arterial_file]
    joint_counts=np.asarray([np.isfinite(layer).all(axis=-1).sum(axis=0) for layer in layers],dtype=int)
    manifest={"sources":list(SOURCES),"training_geometries":list(SOURCES[:3]),"holdout_geometry":SOURCES[3],
              "input_sha256":{str(f):digest(f) for f in paths},
              "window_width_mm":50,"depth_range_mm":[3,80],"station_spacing_mm":5,
              "cross_section_counts":counts,"coverage_by_role":joint_counts.tolist(),
              "blender_center_inside_replayed_bone_by_role":inside.sum(axis=0).tolist(),
              "tasks":TASKS,"graph":GRAPH}
    out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/"geometry.npz",centers=np.stack(layers),contacts=contacts,bone_distance=bdist,inside_bone=inside,ys=YS,valid_contact=contacts[:,:,0]<-1)
    (out/"geometry_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    return manifest


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    opts=parser.parse_args()
    print(assemble(opts.out))
