"""Verify the evidence manifest and unpack gzip without external dependencies."""
import argparse,gzip,hashlib,json,shutil
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 root=Path(__file__).resolve().parent
 if a.out.exists() and any(a.out.iterdir()):raise SystemExit('Use a new empty output directory')
 a.out.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((root/'MANIFEST.json').read_text())
 for entry in manifest['files']:
  rel=Path(entry['path'])
  if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe path')
  data=(root/rel).read_bytes()
  if hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('Upload checksum mismatch: '+str(rel))
  if entry['encoding']=='gzip':
   data=gzip.decompress(data);rel=rel.with_suffix('')
  if hashlib.sha256(data).hexdigest()!=entry['original_sha256']:raise ValueError('Decoded checksum mismatch: '+str(rel))
  dst=a.out/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(data)
 shutil.copy2(root/'MANIFEST.json',a.out/'MANIFEST.json')
 print('Verified and extracted',len(manifest['files']),'files to',a.out)
if __name__=='__main__':main()
