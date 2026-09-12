from pathlib import Path
import hashlib, os, tempfile
base=Path('/media/camp/EXT_DRIVE/RealUS_playground/analysis_artifacts/contact_qp_session005_20260911/failures')
target=Path('/media/camp/EXT_DRIVE/ICRA_YM/script/scan_robot.py')
staged=base/'scan_robot.py.staged'
old=target.read_bytes(); new=staged.read_bytes()
assert hashlib.sha256(old).hexdigest()=='8cbfdc1c0f32d9b3e90d147e540a61c2ecfff82a78be9fab196b67778a07821b', 'target changed; do not overwrite'
assert hashlib.sha256(new).hexdigest()=='475651e872615123a75624d9bfe37d876ec01dddbe23843c41ca3a149d597b0b', 'reviewed patch changed'
compile(new,str(target),'exec')
(base/'scan_robot.py.before_apply').write_bytes(old)
fd,tmp=tempfile.mkstemp(prefix='.scan_robot.py.',dir=target.parent)
try:
    os.fchmod(fd,target.stat().st_mode & 0o777)
    with os.fdopen(fd,'wb') as stream:
        stream.write(new)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp,target)
finally:
    if os.path.exists(tmp): os.unlink(tmp)
print('Applied reviewed scan completion patch; sha256='+hashlib.sha256(target.read_bytes()).hexdigest())
