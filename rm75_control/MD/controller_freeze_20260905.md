# Controller freeze 2026-09-05

Recorded before residual-QP1 / publish / fault-SM work. Do not treat this as a git commit.

## Repositories

| Tree | HEAD | Notes |
|---|---|---|
| RealUS_playground | `13f1c27f01b2d05f6a3874167e0940f666c9d91a` | dirty: directional HQP + peirastic DOF + rail worker |
| ICRA_2027 | `ebe77072e1a48b93d41aa9b42f43bf4b9ad2238f` | dirty FORCE_TEST; imports playground via `sys.path` |

## Python import roots

- `peirastic` → `/media/camp/EXT_DRIVE/RealUS_playground/peirastic`
- `rm75_control` → `/media/camp/EXT_DRIVE/RealUS_playground/rm75_control/rm75_control`
- ICRA scripts prepend those two roots; they do not ship a peirastic copy.

## Native

- Source: `rm75_control/native/wbc_rt/`
- Binary: `rm75_control/native/wbc_rt/build/wbc_rt`
- Protocol at freeze: v7, `WbcIn=616`, `WbcOut=1440`
- Startup now refuses a binary whose embedded source/protocol hashes disagree with the tree (see `wbc_rt/build_id`).
- Post-rebuild `wbc_rt --hash` / `combined_hash()`: `0956b62e92727a821b5b46a9857f15cf3b6bac1ad488a0a07de268dcc23888c7`

## Config

- Window A yaml remains the existing peirastic / joint_admittance_8dof config path.
- No d*, rail macro, force, gamepad-filter, hardware box, or watchdog-threshold edits in this change set.
