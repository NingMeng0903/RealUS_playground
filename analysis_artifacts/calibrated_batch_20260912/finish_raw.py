from pathlib import Path
import json,collections,os
import batch
if hasattr(os,'sched_getaffinity'):
 cpus=os.sched_getaffinity(0)-{2,3,4,5}
 if cpus:os.sched_setaffinity(0,cpus)
os.nice(10)
report=json.loads((batch.WORK/'finishing_progress.json').read_text())
plan=json.loads((batch.WORK/'plan.json').read_text())
report['errors']=[e for e in report['errors'] if "unexpected keyword argument 'exit_absolute'" not in e['error']]
assert len([r for r in report['files'] if r['role']!='auxiliary_contact_segment'])==168
for row in plan:
 if row['kind']!='raw' or any(x['source']==str(batch.SRC/row['relative']) for x in report['raw_postprocess']):continue
 p=batch.SRC/row['relative'];target=batch.DST/row['relative'];status='not_recorded'
 try:
  for name in ('failure.json','result.json'):
   f=p.parent/name
   if f.exists():status=json.loads(f.read_text()).get('status',name.split('.')[0]);break
  cropdir=batch.WORK/'crops'/p.relative_to(batch.SRC).parent;rp=cropdir/'raw_contact_report.json'
  result=json.loads(rp.read_text()) if rp.exists() else batch.postprocess_raw(p,cropdir)
  report['raw_postprocess'].append(dict(source=str(p),attempt_status=status,**result))
  for crop in result['outputs']:
   crop=Path(crop);dest=target.parent/'contact_segments'/crop.name
   existed=dest.exists()
   if not existed:batch.align.align_file(crop,dest,batch.CAL)
   info=batch.validate(crop,dest)
   report['files'].append(dict(source=str(crop),raw_source=str(p),output=str(dest),role='auxiliary_contact_segment',attempt_status=status,action='verified_existing' if existed else 'created',**info))
  print('[POSTPROCESSED]',row['relative'],'segments',len(result['outputs']),'rejections',len(result['rejected']),flush=True)
 except Exception as e:
  report['errors'].append(dict(source=str(p),error=str(e)));print('[ERROR]',p,e,flush=True)
 batch.write_json(batch.WORK/'finishing_progress.json',report)
report['source_changes_after_processing']=[]
for row in plan:
 p=batch.SRC/row['relative']
 if not p.exists():
  report['source_changes_after_processing'].append(dict(source=str(p),status='missing_at_final_recheck_after_output_validation'));continue
 st=p.stat()
 if (st.st_size,st.st_mtime_ns)!=(row['size'],row['mtime_ns']):report['source_changes_after_processing'].append(dict(source=str(p),status='changed_since_inventory'))
original=[r for r in report['files'] if r['role']!='auxiliary_contact_segment']
formal=[r for r in original if r['role']=='contact_segment']
report['summary']=dict(input_hdf5=168,original_aligned=len(original),formal_scans=len(formal),raw_recordings=52,raw_postprocessed=len(report['raw_postprocess']),auxiliary_segments=len(report['files'])-len(original),created=sum(r['action']=='created' for r in report['files']),verified_existing=sum(r['action']=='verified_existing' for r in report['files']),metadata_copied_verified=len(report['metadata']),formal_input_frames=sum(r['input_frames'] for r in formal),formal_output_frames=sum(r['output_frames'] for r in formal),formal_dropped_frames=sum(r['dropped_frames'] for r in formal),errors=len(report['errors']))
for p in (batch.WORK/'report.json',batch.DST/('_'+batch.RUN)/'report.json'):batch.write_json(p,report)
for subject in sorted({Path(r['relative']).parts[0] for r in plan}):
 selected=[r for r in report['files'] if Path(r['output']).relative_to(batch.DST).parts[0]==subject]
 batch.write_json(batch.DST/subject/(batch.RUN+'_report.json'),dict(files=selected))
text=f'''# 时间戳对齐与后处理：2026-09-12

正式数据：11 个受试者/采集目录下的 116 个 LH/RH_Per_*.h5。这些输入已完成接触裁剪，本轮不重复裁剪。
辅助数据：对应 attempts 中的 52 个 raw.h5 已全长对齐；另提取 {report['summary']['auxiliary_segments']} 个接触段到各 attempt/contact_segments。
失败尝试保持在 attempts；接触段可满足力阈值条件但不代表整条扫描成功。请依据原 failure/result/session 元数据及本次 report 中的 attempt_status 选择训练/评价数据，勿把辅助段与正式数据重复计数。

采用 delay_cal/002/delay_report.jso