from pathlib import Path
import hashlib,json,datetime
ROOT=Path(__file__).resolve().parent.parent
rows=[]
for p in sorted(ROOT.rglob('*')):
    if not p.is_file() or p.name=='MANIFEST.json' or '__pycache__' in p.parts or '.pytest_cache' in p.parts or 'vendor' in p.parts:
        continue
    rows.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
out={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'offline validation; no physical ultrasound','files':rows,
     'vendor':'e3nn0.6.0,opt_einsum_fx0.1.4,opt_einsum3.4.0 installed only in theory/vendor; install.log records packages',
     'production_change':'ird_playground/ird_playground/ird/torch_kinematics.py so3_exp only; module manifest records source hash and patch'}
(ROOT/'MANIFEST.json').write_text(json.dumps(out,indent=2)+'\n')
print(f'Manifest: {len(rows)} files')
