"""Make libcuda.so visible to Taichi/Genesis (dlopen libcuda.so, not libcuda.so.1)."""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path

from rm75_control.control.joint_admittance_8dof.param_model.paths import CUDA_SHIM_DIR

_REEXEC_FLAG = "RM75_GENESIS_CUDA_SHIM"


def _cuda_driver_candidates() -> list[Path]:
    return [
        Path("/lib/x86_64-linux-gnu/libcuda.so.1"),
        Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1"),
    ]


def _library_prefixes(shim_dir: Path) -> list[str]:
    prefixes = [str(shim_dir)]
    try:
        import torch

        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        if torch_lib.is_dir():
            prefixes.append(str(torch_lib))
    except Exception:
        pass
    for sp in site.getsitepackages():
        nvidia_root = Path(sp) / "nvidia"
        if nvidia_root.is_dir():
            for lib_dir in nvidia_root.rglob("lib"):
                if lib_dir.is_dir():
                    prefixes.append(str(lib_dir.resolve()))
    return prefixes


def ensure_cuda_driver_for_taichi(*, require_gpu: bool = True) -> None:
    """Re-exec once with LD_LIBRARY_PATH so Taichi can load libcuda.so.

    Ubuntu/Debian often provide only ``libcuda.so.1``; Quadrants/Taichi asks for
    ``libcuda.so`` and then reports ``Arch.cuda is not supported``.
    """
    if os.environ.get(_REEXEC_FLAG) == "1":
        return

    driver = next((p for p in _cuda_driver_candidates() if p.exists()), None)
    if driver is None:
        if require_gpu:
            raise RuntimeError(
                "NVIDIA driver library not found (libcuda.so.1). "
                "Install the proprietary driver and verify with nvidia-smi."
            )
        return

    shim_dir = CUDA_SHIM_DIR
    shim_dir.mkdir(exist_ok=True)
    link = shim_dir / "libcuda.so"
    try:
        if link.is_symlink() and link.resolve() == driver.resolve():
            pass
        else:
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(driver)
    except OSError as exc:
        if require_gpu:
            raise RuntimeError(f"failed to create {link} -> {driver}: {exc}") from exc
        return

    existing = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    for prefix in reversed(_library_prefixes(shim_dir)):
        if prefix not in existing:
            existing.insert(0, prefix)

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(existing)
    env[_REEXEC_FLAG] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)
