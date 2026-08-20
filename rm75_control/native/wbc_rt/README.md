# wbc_rt

Separate-process C++ inner loop (Pinocchio + ProxQP + coal). Python keeps the
`JointIkController.step / enable / stop` facade and talks over named SHM.

Default yaml is still `inner.backend: python`. Do not switch production to
`native` until `tests/test_wbc_rt_offline_ab.py` passes on this machine.

## Build

```bash
./native/wbc_rt/build.sh
```

Requires the rm75 cmeel prefix (Pinocchio / ProxQP / coal). The script downloads
header-only [simde](https://github.com/simd-everywhere/simde) into
`third_party/simde` (gitignored).

## Run

```yaml
inner:
  backend: native          # or python
  native_bin: /path/to/wbc_rt   # optional; else WBC_RT_BIN / build/wbc_rt
  native_shm_prefix: rm75_wbc
```
