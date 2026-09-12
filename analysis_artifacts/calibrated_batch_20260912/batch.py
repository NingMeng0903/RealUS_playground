"""Batch existing offline crop/alignment, preserving source and prior outputs."""
from pathlib import Path
import argparse, collections, hashlib, json, os, shutil, sys, time
import h5py
import numpy as np
sys.path.insert(0,'/media/camp/EXT_DRIVE/ICRA_YM/script')
import align_to_ultrasound as align
from postprocess import detect_contact_episodes, crop_hdf5_file, _validate_stream_layout
from scan_contact import contact_policy
BASE=Path('/media/camp/yameng/icra 2027')
SRC=BASE/'uncalibrated'; DST=BASE/'calibrated'
WORK=Path(__file__).resolve().parent
CAL=align.calibration(BASE/'delay_cal/002/delay_report.json')
RUN='batch_20260912'

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def write_json(p,data):
 p.parent.mkdir(parents=True,exist_ok=True)
 temp=p.with_suffix(p.suffix+'.tmp')
 temp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');os.replace(temp,p)

def inventory():
 out=[]
 for p in sorted(SRC.rglob('*')):
  if not p.is_file():continue
  st=p.stat();r=dict(relative=str(p.relative_to(SRC)),size=st.st_size,mtime_ns=st.st_mtime_ns)
  if p.suffix.lower() in ('.h5','.hdf5'):
   with h5py.File(p,'r') as h:
    r.update(kind='contact_segment' if h.attrs.get('postprocess_schema') else 'raw',complete=bool(h.attrs.get('complete',False)),schema=str(h.attrs.get('schema','')))
  else:r['kind']='metadata'
  out.append(r)
 return out

def validate(source,target):
 with h5py.File(source,'r') as a,h5py.File(target,'r') as b:
  assert b.attrs.get('alignment_schema')==align.SCHEMA and b.attrs['complete']
  assert b['provenance'].attrs['calibration_report_sha256']==CAL['report_sha256']
  us=align.timestamps(a['ultrasound/timestamp_ns'][:]);off=align.read_us_offset(a['ultrasound'])
  shifts={'tcp':CAL['ultrasound_minus_tcp_ns']+off-CAL['calibration_us_offset_ns']}
  shifts['force']=shifts['tcp']-CAL['force_minus_tcp_ns']
  keep=np.ones(len(us),bool);plans={};limit=align.seconds_ns(16.666667/1000)
  for name,shift in shifts.items():
   ts=align.timestamps(a[name+'/timestamp_ns'][:]);corrected=ts+shift
   index,delta=align.nearest_indices(corrected,us)
   keep &= (us>=corrected[0]) & (us<=corrected[-1]) & (abs(delta)<=limit)
   plans[name]=(index,delta,ts)
  ii=np.flatnonzero(keep);grid=us[ii]
  np.testing.assert_array_equal(b['alignment/image_source_index'][:],ii)
  np.testing.assert_array_equal(b['timestamp_ns'][:],grid)
  for name in ('tcp','force','ultrasound'):
   np.testing.assert_array_equal(b[name+'/timestamp_ns'][:],grid)
   ix=ii if name=='ultrasound' else plans[name][0][keep]
   if name!='ultrasound':
    np.testing.assert_array_equal(b['alignment/'+name+'/source_index'][:],ix)
    np.testing.assert_array_equal(b['alignment/'+name+'/source_timestamp_ns'][:],plans[name][2][ix])
    np.testing.assert_array_equal(b['alignment/'+name+'/match_error_ns'][:],plans[name][1][keep])
   def fields(g,pre=''):
    for key,item in g.items():
     path=pre+key
     if isinstance(item,h5py.Group):yield from fields(item,path+'/')
     elif item.ndim:yield path,item
   for key,item in fields(a[name]):
    if key in ('timestamp_ns','timestamp_mono_ns','header/stamp/sec','header/stamp/nanosec'):continue
    dest=b[name+'/'+key]
    assert len(dest)==len(ix),(name,key)
    if h5py.check_dtype(vlen=item.dtype) is not None:
     for j,k in enumerate(ix):np.testing.assert_array_equal(dest[j],item[int(k)])
    else:
     unique,inverse=np.unique(ix,return_inverse=True)
     np.testing.assert_equal(dest[:],item[unique][inverse])
  rep=json.loads(b.attrs['alignment_report_json'])
  assert rep['output_frames']==len(grid) and rep['input_frames']==len(us)
  return dict(input_frames=len(us),output_frames=len(grid),dropped_frames=len(us)-len(grid),streams=rep['streams'])

def postprocess_raw(p,cropdir):
 cropdir.mkdir(parents=True,exist_ok=True)
 with h5py.File(p,'r') as h:
  if not bool(h.attrs.get('complete',False)):raise ValueError('Incomplete raw input')
  _validate_stream_layout(h,require_all=True)
  f=h['force']
  result=detect_contact_episodes(f['timestamp_ns'][:],f['contact_force_n'][:],f['force_control_active'][:],f['mode_valid'][:],**contact_policy())
  ranges={n:h[n+'/timestamp_ns'][:] for n in ('tcp','force','ultrasound')}
 report=dict(input_path=str(p),config=result.config,accepted=[],rejected=list(result.rejected),outputs=[])
 for episode in result:
  lo,hi=episode['start_time_ns'],episode['end_time_ns']
  if any(not np.any((ts>=lo)&(ts<=hi)) for ts in ranges.values()):
   report['rejected'].append(dict(episode,reason='empty_stream'));continue
  dest=cropdir/f'raw_contact_{len(report["outputs"])+1:03d}.hdf5'
  actual=crop_hdf5_file(p,dest,lo,hi,episode=episode)
  report['accepted'].append(dict(episode,output_path=str(actual)));report['outputs'].append(str(actual))
 write_json(cropdir/'raw_contact_report.json',report)
 return report

def run(plan):
 if hasattr(os,'sched_getaffinity'):
  cpus=os.sched_getaffinity(0)-{2,3,4,5}
  if cpus:os.sched_setaffinity(0,cpus)
 os.nice(10)
 report=dict(source=str(SRC),destination=str(DST),calibration={k:v for k,v in CAL.items() if k!='report'},contact_policy=contact_policy(),files=[],metadata=[],raw_postprocess=[],errors=[])
 def aligned(source,target,role,status=None):
  existed=target.exists()
  if not existed:align.align_file(source,target,CAL)
  checked=validate(source,target)
  report['files'].append(dict(source=str(source),output=str(target),role=role,attempt_status=status,action='verified_existing' if existed else 'created',**checked))
  print('[OK]',str(target.relative_to(DST)),checked['output_frames'],'frames',flush=True)
 for row in plan:
  p=SRC/row['relative'];target=DST/row['relative']
  try:
   assert (p.stat().st_size,p.stat().st_mtime_ns)==(row['size'],row['mtime_ns']),'Source changed since plan'
   if row['kind']=='metadata':continue
   status='not_recorded'
   for f in ('failure.json','result.json'):
    if (p.parent/f).exists():status=json.loads((p.parent/f).read_text()).get('status',f.split('.')[0]);break
   aligned(p,target,row['kind'],status)
   if row['kind']=='raw':
    cropdir=WORK/'crops'/p.relative_to(SRC).parent
    rp=cropdir/'raw_contact_report.json'
    if rp.exists():result=json.loads(rp.read_text())
    else:result=postprocess_raw(p,cropdir)
    report['raw_postprocess'].append(dict(source=str(p),attempt_status=status,**result))
    for crop in result['outputs']:
     crop=Path(crop)
     aligned(crop,target.parent/'contact_segments'/crop.name,'auxiliary_contact_segment',status)
   assert (p.stat().st_size,p.stat().st_mtime_ns)==(row['size'],row['mtime_ns']),'Source changed during processing'
  except Exception as e:
   report['errors'].append(dict(source=str(p),error=f'{type(e).__name__}: {e}'));print('[ERROR]',p,e,flush=True)
  write_json(WORK/'progress.json',report)
 for row in plan:
  if row['kind']!='metadata':continue
  p=SRC/row['relative'];target=DST/row['relative']
  try:
   assert (p.stat().st_size,p.stat().st_mtime_ns)==(row['size'],row['mtime_ns']),'Source changed since plan'
   sha=digest(p)
   if target.exists() and digest(target)!=sha:target=DST/('_'+RUN)/'source_metadata'/row['relative']
   target.parent.mkdir(parents=True,exist_ok=True)
   if not target.exists():
    with p.open('rb') as a,target.open('xb') as b:shutil.copyfileobj(a,b,4*1024*1024)
   assert digest(target)==sha
   assert (p.stat().st_size,p.stat().st_mtime_ns)==(row['size'],row['mtime_ns'])
   report['metadata'].append(dict(source=str(p),output=str(target),sha256=sha))
  except Exception as e:report['errors'].append(dict(source=str(p),error=str(e)))
 report['summary']=dict(input_hdf5=sum(x['kind']!='metadata' for x in plan),original_aligned=sum(x['role']!='auxiliary_contact_segment' for x in report['files']),auxiliary_segments=sum(x['role']=='auxiliary_contact_segment' for x in report['files']),created=sum(x['action']=='created' for x in report['files']),verified_existing=sum(x['action']=='verified_existing' for x in report['files']),metadata_copied_verified=len(report['metadata']),errors=len(report['errors']))
 for p in (WORK/'report.json',DST/('_'+RUN)/'report.json'):write_json(p,report)
 for subject in sorted({Path(r['relative']).parts[0] for r in plan}):
  selected=[r for r in report['files'] if Path(r['output']).relative_to(DST).parts[0]==subject]
  write_json(DST/subject/(RUN+'_report.json'),dict(files=selected))
 print('[DONE]',json.dumps(report['summary']),flush=True)
 return bool(report['errors'])

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');args=ap.parse_args()
 if args.run:raise SystemExit(run(json.loads((WORK/'plan.json').read_text())))
 plan=inventory();write_json(WORK/'plan.json',plan)
 print(dict(collections.Counter(r['kind'] for r in plan)))
 print('calibration',CAL['ultrasound_minus_tcp_ns'],CAL['force_minus_tcp_ns'])
