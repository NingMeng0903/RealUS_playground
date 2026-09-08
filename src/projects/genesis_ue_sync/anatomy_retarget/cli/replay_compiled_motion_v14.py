"""Replay a saved subject on a frozen motion clip and render actual Genesis RGB.

No compilation, fitting, Blender access or nearest-point rebinding occurs.
The optional signed-distance queries are diagnostics and never change output.
Unsupported frames produce explicit title cards, not clamped geometry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from ..consistent_runtime_v14 import load_compiled_subject, pose
from ..smplx_body_surface_v7 import load_smplx_model_v7, require_frozen_smplx_male_v7
from .run_material_matrix_v13 import _pose_joints_and_skin
from .export_capture_joint_review_v13 import _tissue_codes
from .render_alignment_truth_genesis_v1 import _render_layer, COLORS
from .render_chain_rest_fit_genesis_v1 import _export
from .render_material_genesis_v13 import _cameras


ROOT = Path(__file__).resolve().parents[5]
MODEL = ROOT/'ref_code_library/EasyMocap/data/smplx/smplx/SMPLX_MALE.pkl'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode_video(directory, fps, ffmpeg_executable):
    directory=Path(directory);destination=directory/'genesis_motion.mp4'
    subprocess.run([ffmpeg_executable,'-hide_banner','-loglevel','error','-framerate',str(fps),
                    '-i',str(directory/'video_frames'/'%05d.png'),'-c:v','libx264',
                    '-pix_fmt','yuv420p',str(destination)],check=True)
    return dict(path=str(destination.resolve()),sha256=sha(destination))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiled',type=Path,required=True)
    parser.add_argument('--motion',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smplx-model',type=Path,default=MODEL)
    parser.add_argument('--views',nargs='+',default=['whole_ap','left_elbow_lateral'])
    parser.add_argument('--backend',default='cpu')
    parser.add_argument('--metrics-only',action='store_true')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    # Genesis isolates portions of sys.path during initialization. Resolve the
    # already-installed encoder before that isolation, not after rendering.
    ffmpeg_executable=None
    if not args.metrics_only:
        import imageio_ffmpeg
        ffmpeg_executable=imageio_ffmpeg.get_ffmpeg_exe()
    compiled=load_compiled_subject(args.compiled)
    model_path,model_sha=require_frozen_smplx_male_v7(args.smplx_model)
    model=load_smplx_model_v7(model_path)
    asset=compiled.source_asset; labels=_tissue_codes(asset)
    with np.load(args.motion,allow_pickle=False) as data:
        poses=data['poses'].copy(); translations=data['transl'].copy()
        frame_ids=data['frame_ids'].copy(); fps=float(data['output_fps'])
    if poses.ndim!=3 or poses.shape[1:]!=(55,3) or translations.shape!=(len(poses),3):
        raise ValueError('motion must contain poses[N,55,3] and transl[N,3]')
    if not np.isfinite(poses).all() or not np.isfinite(translations).all() or not np.isfinite(fps) or fps<=0:
        raise ValueError('motion contains invalid values')
    shutil.copy2(args.motion,args.output/'motion.npz')
    manifest=dict(compiled=str(args.compiled.resolve()),compiled_manifest_sha256=sha(args.compiled/'manifest.json'),
                  motion_sha256=sha(args.motion),smplx_model_sha256=model_sha,fps=fps,
                  publishable=False,anatomical_passed=False,used_for_fit=False,
                  runtime_optimization=False,runtime_blender=False,
                  display='root orientation and translation removed equally from skin and anatomy for upright review',
                  internal_material_alpha=1.0,frames=[])
    bone=np.flatnonzero(labels==0)
    vessels=np.flatnonzero(labels==1)
    tissue_faces={name:asset.faces[np.all(labels[asset.faces]==code,axis=1)]
                  for name,code in [('bones',0),('vessels',1),('nerves',2)]}
    left_hand=set()
    for i in range(235):
        parent=i
        while parent>=0 and parent!=135: parent=int(asset.source_bone_parents[parent])
        if parent==135: left_hand.add(i)
    arm=[]
    for n,t,c,(start,stop) in zip(asset.source_mesh_names,asset.source_tissues,
                                 asset.source_mesh_controller_bones,asset.source_vertex_ranges):
        if n in ['Humerus_L','Radius_L','Ulna_L'] or (t=='bone' and int(c) in left_hand):
            arm.extend(range(start,stop))
    arm=np.asarray(arm,dtype=np.int64)
    output_frames=args.output/'video_frames';output_frames.mkdir()
    for index,(p,translation,frame_id) in enumerate(zip(poses,translations,frame_ids)):
        record=dict(index=index,source_frame=int(frame_id),time_s=index/fps)
        start=time.perf_counter()
        try:
            vertices=pose(compiled,p,translation)
        except ValueError as exc:
            record.update(status='unsupported_or_evaluation_error',error=str(exc),error_type=type(exc).__name__)
            if not args.metrics_only:
                card=Image.new('RGB',(720*len(args.views),570),(35,8,8));draw=ImageDraw.Draw(card)
                draw.text((20,20),f'Frame {frame_id}: unsupported / evaluation error',fill='white')
                draw.text((20,55),str(exc)[:160],fill='white');card.save(output_frames/f'{index:05d}.png')
        else:
            import igl
            pose_seconds=time.perf_counter()-start
            skin,skin_faces,joints=_pose_joints_and_skin(model,betas=compiled.betas,pose=p)
            skin=skin+translation;joints=joints+translation
            record.update(status='evaluated',vertices_sha256=hashlib.sha256(vertices.tobytes()).hexdigest(),
                          pose_seconds=pose_seconds)
            distances=igl.signed_distance(np.asarray(vertices,dtype=np.float64),
                np.asarray(skin,dtype=np.float64),np.asarray(skin_faces,dtype=np.int64))[0]
            record['skin']={name:dict(max_outside_m=float(np.maximum(distances[ids],0).max()),
                                      outside_over_1mm_count=int(np.count_nonzero(distances[ids]>.001)))
                            for name,ids in [('all_bones',bone),('left_arm_bones',arm),('vessels',vessels)]}
            if not args.metrics_only:
                pivot=joints[0]; inverse=Rotation.from_rotvec(p[0]).inv()
                def upright(v):return inverse.apply(v-pivot)
                skin_u=upright(skin);joints_u=upright(joints);anatomy_u=upright(vertices)
                available=_cameras(skin_u,joints_u)
                unknown=set(args.views)-available.keys()
                if unknown: raise ValueError(f'unknown cameras {unknown}')
                cameras={name:available[name] for name in args.views}
                # Temporary OBJs only transport exact evaluated vertices to
                # Genesis; the saved package + frozen motion reproduces them.
                with tempfile.TemporaryDirectory(prefix='anatomy_v14_frame_') as temporary:
                    temporary=Path(temporary)
                    skin_obj=_export(temporary/'skin.obj',skin_u,skin_faces)
                    entities=[('skin',skin_obj,COLORS['skin'])]
                    for name,color in [('bones',COLORS['candidate']),('vessels',(*COLORS['vessels'][:3],1.0)),('nerves',(*COLORS['nerves'][:3],1.0))]:
                        obj=_export(temporary/f'{name}.obj',anatomy_u,tissue_faces[name])
                        entities.append((name,obj,color))
                    render_dir=args.output/'genesis'/f'{index:05d}'
                    _render_layer(render_dir,entities=entities,cameras=cameras,backend=args.backend)
                sheet=Image.new('RGB',(720*len(args.views),570),(25,25,25));draw=ImageDraw.Draw(sheet)
                for col,name in enumerate(args.views):
                    with Image.open(render_dir/'rgb'/f'{name}.png') as im:sheet.paste(im.convert('RGB'),(720*col,30))
                    draw.text((720*col+8,8),f'{name} | frame {frame_id} | diagnostic candidate',fill='white')
                sheet.save(output_frames/f'{index:05d}.png')
        record['elapsed_s']=time.perf_counter()-start;manifest['frames'].append(record)
        (args.output/'report.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
        print(index,int(frame_id),record['status'],flush=True)
    manifest['evaluated_frames']=sum(f['status']=='evaluated' for f in manifest['frames'])
    manifest['unsupported_or_error_frames']=len(poses)-manifest['evaluated_frames']
    if not args.metrics_only:
        manifest['video']=encode_video(args.output,fps,ffmpeg_executable)
    (args.output/'report.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
