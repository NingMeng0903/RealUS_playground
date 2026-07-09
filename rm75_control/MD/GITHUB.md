# rm75_control GitHub Notes

Repository root: `/media/camp/EXT_DRIVE/rm75_control`  
Remote: `https://github.com/NingMeng0903/rm75_control.git`

## Active layout

- `rm75_control/`: installable Python package (session, motion, force scan, backend)
- `configs/`: robot YAML defaults (`rm75f_default.yaml`)
- `apps/`: application entry points
- `data/`: force-ID logs and runtime artifacts
- `tests/`: offline tests
- `MD/`: operator and Git notes

External runtime dependency (not in this repo):

- [RM_API2](https://github.com/RealManRobot/RM_API2) — set `RM_API2_PYTHON` in `env.sh`

## First push scope

Commit:

- `rm75_control/`
- `configs/`
- `apps/`
- `tests/`
- `MD/`
- `README.md`
- `env.sh`, `requirements.txt`, `setup.py`, `pyproject.toml`

Do not commit:

- local conda / venv trees
- `*.egg-info/` build artifacts
- machine-specific secrets or IPs if you prefer to keep them local
- large logs under `data/` or system `/tmp/` unless explicitly needed

## Git bootstrap

First time on a new machine:

```bash
cd /media/camp/EXT_DRIVE/rm75_control
git init
git add rm75_control/ configs/ apps/ tests/ MD/ README.md env.sh requirements.txt setup.py pyproject.toml
git status
git commit -m "Initial commit: RM75 control wrapper and force-scan demos."
git branch -M main
git remote add origin https://github.com/NingMeng0903/rm75_control.git
git push -u origin main
```

## Routine upload (after local changes)

```bash
cd /media/camp/EXT_DRIVE/rm75_control
git status
git add .
git commit -m "不弹，力不快"
git push origin main
```


If the remote is ahead:

```bash
cd /media/camp/EXT_DRIVE/rm75_control
git pull origin main --rebase
git push origin main
```～

## Current command entry points

Activate env before any script:

```bash
source /media/camp/EXT_DRIVE/rm75_control/env.sh
```

### 8-DOF joint admittance scan (current)

```bash
source /media/camp/EXT_DRIVE/rm75_control/env.sh
python apps/joint_admittance_8dof/d_sin_tool_y.py \
  --config configs/joint_admittance_8dof.yaml \
  --enable-force --desired-z 2.0 --scan-duration 3000 \
  --log-csv /tmp/scan.csv
```

### Recover after planned/stream force conflict

```bash
source /media/camp/EXT_DRIVE/rm75_control/env.sh
python apps/force_compensation/recover_force_stream.py
```

## Notes

- `README.md` is the primary GitHub landing page.
- RM_API2 stays outside this repo; document its path in `env.sh` on each machine.
- Do not force-push `main` unless you intentionally rewrite remote history.
