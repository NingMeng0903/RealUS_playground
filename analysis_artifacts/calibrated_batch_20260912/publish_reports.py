from pathlib import Path
import json,shutil
import batch
r=json.loads((batch.WORK/'report.json').read_text());assert not r['errors']
formal=[x for x in r['files'] if x['role']=='contact_segment']
assert len(formal)==116 and len(r['raw_postprocess'])==52
files=[]
import h5py
for f in formal:
 with h5py.File(f['output'],'r') as h:files.append(json.loads(h.attrs['alignment_report_json']))
report=dict(schema=batch.align.SCHEMA,calibration=r['calibration'],files=files,errors=[],input_files=len(files),output_files=len(files),input_frames=sum(f['input_frames'] for f in files),output_frames=sum(f['output_frames'] for f in files),batch_report='_batch_20260912/report.json',auxiliary_note='Full aligned raw and auxiliary contact segments remain under attempts; excluded from formal file counts.')
def publish(p,obj):
 if p.exists():
  backup=batch.DST/('_'+batch.RUN)/'previous_reports'/p.relative_to(batch.DST)
  backup.parent.mkdir(parents=True,exist_ok=True)
  if not backup.exists():shutil.copy2(p,backup)
 batch.write_json(p,obj)
publish(batch.DST/'alignment_report.json',report)
for person in sorted({Path(x['output']).parent.name for x in files}):
 publish(batch.DST/person/'alignment_report.json',dict(person=person,files=[x for x in files if Path(x['output']).parent.name==person]))
p=batch.DST/'README_batch_20260912.md';s=p.read_text();s=s.replace('根目录旧 alignment_report.json 属于历史批次，本轮总览以新报告为准。','根目录 alignment_report.json 和各目录同名报告已更新为116个正式扫描的完整索引；旧报告备份在 _batch_20260912/previous_reports。')
p.write_text(s)
print(json.dumps(r['summary']))
print('formal_max_residual_ms', {g:max(f['streams'][g]['residual']['max_ms'] for f in files) for g in ('tcp','force')})
print('subjects',len({Path(x['output']).parent.name for x in files}))
