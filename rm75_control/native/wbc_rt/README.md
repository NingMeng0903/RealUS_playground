# wbc_rt

Separate-process C++ inner loop (Pinocchio + ProxQP + coal). Python keeps the
`JointIkController.step / enable / stop` facade and talks over named SHM.
The owning Python client also passes a socket with `--notify-fd`: requests
and completed replies wake the other process directly, without sleep polling
or a Python busy wait. Closing the owner socket stops the native worker.

The default YAML selects `inner.backend: native`. The source hash includes
the notification transport, so rebuild the binary after updating this code.
The Python client retains its 20 ms response deadline. An expired request
is not published, and a late reply cannot clear the fault; an acknowledged
reset is required before further steps.

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
