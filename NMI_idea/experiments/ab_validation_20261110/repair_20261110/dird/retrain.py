"""Re-run the actual production recipe into a NEW owned output directory."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml
from common import HERE, SOURCE, save


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=HERE/'candidate_reproduced')
    a=p.parse_args(); out=a.output.resolve()
    if not out.is_relative_to(HERE) or out.exists():
        raise ValueError('Choose a new, nonexistent directory inside this repair/dird directory; never overwrite a checkpoint')
    out.mkdir()
    cfg=yaml.safe_load((HERE/'candidate/train.yaml').read_text())
    cfg['io']={k:str(out/n) for k,n in [('checkpoint','selected.pt'),('report','train_report.json'),('conformal','conformal.json'),('eval_report','eval_report.json')]}
    path=out/'train.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    commands=[['-m','ird_playground.cli.train_signed','--config',str(path)],
        ['-m','ird_playground.cli.calibrate_conformal','--config',str(path),'--output',str(out/'conformal.json')],
        ['-m','ird_playground.cli.eval_signed','--config',str(path)]]
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    env['PYTHONPATH']=str(SOURCE)+':'+str(SOURCE.parent/'rm75_control')+':'+env.get('PYTHONPATH','')
    log=[]
    for i,cmd in enumerate(commands):
        started=time.perf_counter()
        with (out/f'command_{i}.log').open('w') as f:
            code=subprocess.run([sys.executable,'-u',*cmd],env=env,stdout=f,stderr=subprocess.STDOUT).returncode
        log.append({'argv':[sys.executable,'-u',*cmd],'exit_code':code,'elapsed_seconds':time.perf_counter()-started})
        (out/'commands.json').write_text(json.dumps(log,indent=2))
        if code and i<2: raise RuntimeError(f'actual production command {i} failed: {code}')
    print(json.dumps(log,indent=2))
    print('Evaluation exit1 means release criteria were not met; the report is retained, not relabeled or tuned away.')


if __name__=='__main__':main()
