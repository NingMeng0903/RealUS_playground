"""Read-only H5 ultrasound exports. No crop, mirror or intensity adjustment."""
from pathlib import Path
import json, math, subprocess
import h5py, cv2, numpy as np
import imageio_ffmpeg
ROOT=Path('/media/camp/EXT_DRIVE/ICRA_2027/icra 2027_contact/real_characterization')
OUT=Path(__file__).resolve().parent
FFMPEG=imageio_ffmpeg.get_ffmpeg_exe()
FPS=30
SOURCES={'baseline':'baseline_probe50/002','shadow':'shadow_probe50/001','active':'active_probe50/002'}
class Clip:
    def __init__(self,label,rel):
        self.label=label;self.path=ROOT/rel/'RH_Per_L_DtP.h5'
        with h5py.File(self.path,'r') as f:
            stamps=f['ultrasound/timestamp_ns'][:].astype(np.int64)
            if np.any(np.diff(stamps)<=0):raise ValueError('nonmonotone image clock')
            self.times=(stamps-stamps[0])/1e9
            self.frames=[cv2.imdecode(np.asarray(blob,np.uint8),cv2.IMREAD_GRAYSCALE) for blob in f['ultrasound/jpeg']]
            self.metadata=json.loads(f['ultrasound/metadata_json'][0])
        assert all(im is not None and im.shape==self.frames[0].shape for im in self.frames)
        self.h,self.w=self.frames[0].shape;self.duration=float(self.times[-1])
    def at(self,t):
        index=max(0,min(len(self.frames)-1,int(np.searchsorted(self.times,t,side='right')-1)))
        return self.frames[index],index,float(self.times[index])
    def facts(self):
        return dict(input=str(self.path),source_frames=len(self.frames),height=self.h,width=self.w,
                    source_duration_s=self.duration,source_clock='ultrasound/timestamp_ns',
                    original_crop_box=self.metadata.get('crop_box'),original_hflip=self.metadata.get('hflip'),
                    export_crop=None,export_flip=False,intensity_adjustment=False)
class Writer:
    def __init__(self,path,w,h):
        self.path=path;self.log=open(path.with_suffix('.encode.log'),'w')
        self.process=subprocess.Popen([FFMPEG,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','gray',
            '-s',f'{w}x{h}','-r',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18',
            '-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE,stderr=self.log)
        self.n=0
    def write(self,im):self.process.stdin.write(im.tobytes());self.n+=1
    def close(self):
        self.process.stdin.close();code=self.process.wait();self.log.close()
        if code:raise RuntimeError(f'encoder failed {self.path}')
        validation=subprocess.run([FFMPEG,'-hide_banner','-i',str(self.path),'-f','null','-'],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
        self.path.with_suffix('.decode.log').write_text(validation.stderr)
        if validation.returncode:raise RuntimeError(f'decode failed {self.path}')
        return dict(file=self.path.name,frames=self.n,fps=FPS,duration_s=self.n/FPS,bytes=self.path.stat().st_size,
                    codec='H.264 / yuv420p',validation='bundled ffmpeg complete decode, exit 0')
def text(img,msg,x,y,scale=.6):
    cv2.putText(img,msg,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,255,1,cv2.LINE_AA)
def panel(clip,t,ended=False):
    out=np.zeros((clip.h+90,clip.w),np.uint8)
    text(out,clip.label.upper(),12,24,.7)
    if ended:
        text(out,f'END at {clip.duration:.3f} s',12,53)
        text(out,'No further recorded frames',12,76,.5)
        text(out,'END',clip.w//2-38,clip.h//2+70,1.)
    else:
        image,index,actual=clip.at(t);out[90:]=image
        text(out,f'source t={actual:.3f}s  frame {index+1}/{len(clip.frames)}',12,53,.55)
        text(out,f'clip duration={clip.duration:.3f}s',12,76,.5)
    return out
clips=[Clip(k,v) for k,v in SOURCES.items()]
assert len({(c.h,c.w) for c in clips})==1
manifest={'resampling':'30 fps previous-source-frame hold using original H5 image timestamps; no delay adjustment',
          'limits':'Independent re-teaching; not identical paths or spatial/pixel registration. H5s already contact-segment cropped.',
          'single_playback':'1x actual elapsed time; MP4 final quantization <=1/30 s',
          'normalized_playback':'Each clip 0-100% elapsed. Individual columns time-warped, not 1x.',
          'sources':{c.label:c.facts() for c in clips},'outputs':[],'ffmpeg':FFMPEG,'ffprobe':'not installed; complete ffmpeg decode used instead'}
for c in clips:
    writer=Writer(OUT/f'{c.label}_raw_gray_1x.mp4',c.w,c.h)
    for i in range(math.ceil(c.duration*FPS)):writer.write(c.at(i/FPS)[0])
    manifest['outputs'].append(writer.close());print(c.label,'done',flush=True)
longest=max(c.duration for c in clips);n=math.ceil(longest*FPS);h=clips[0].h+90+62;w=clips[0].w*3
for mode in ['normalized_elapsed','elapsed_1x']:
    writer=Writer(OUT/f'comparison_{mode}.mp4',w,h)
    for i in range(n):
        elapsed=i/FPS;fraction=i/(n-1)
        tiles=[panel(c,fraction*c.duration if mode=='normalized_elapsed' else elapsed,
                     ended=(mode=='elapsed_1x' and elapsed>c.duration)) for c in clips]
        canvas=np.zeros((h,w),np.uint8);canvas[62:]=np.concatenate(tiles,axis=1)
        title=(f'Each clip 0-100% elapsed: {100*fraction:.1f}% (columns time-warped)' if mode=='normalized_elapsed'
               else f'Actual elapsed time 1x: {elapsed:.3f}s; END = no further frames')
        text(canvas,title,15,24,.65);text(canvas,'Independent re-teaching / different paths; NOT spatial registration. Original gray pixels; no re-flip.',15,49,.6)
        writer.write(canvas)
    manifest['outputs'].append(writer.close());print(mode,'done',flush=True)
# Contact sheet uses complete source frames; only preview is uniformly downscaled.
rows=[]
for fraction in [0,.25,.5,.75,1.]:
    row=np.concatenate([panel(c,fraction*c.duration) for c in clips],axis=1)
    row=cv2.resize(row,(996,408),interpolation=cv2.INTER_AREA);rows.append(row)
sheet=np.zeros((len(rows)*408+75,996),np.uint8)
text(sheet,'Baseline | Shadow | Active: 0%, 25%, 50%, 75%, 100% elapsed',12,25,.58)
text(sheet,'Independent re-teaching; no spatial registration. Preview only is resized.',12,52,.53)
sheet[75:]=np.concatenate(rows,axis=0)
cv2.imwrite(str(OUT/'contact_sheet.png'),sheet)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('ALL DONE',flush=True)
