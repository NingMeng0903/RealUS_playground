"""Ensure ONNX Runtime CUDA/TensorRT providers can load bundled NVIDIA libs."""

from __future__ import annotations

import ctypes
import importlib
import logging
import os
import site
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_TRT_PRELOADED = False


def _tensorrt_lib_dir() -> Path | None:
    """Resolve pip ``tensorrt_libs`` directory (libnvinfer.so.10)."""
    for mod_name in ("tensorrt_libs", "tensorrt_cu12_libs"):
        try:
            mod = importlib.import_module(mod_name)
            root = Path(mod.__file__).resolve().parent
            if (root / "libnvinfer.so.10").is_file():
                return root
        except Exception:
            continue
    for entry in sys.path:
        root = Path(entry) / "tensorrt_libs"
        if (root / "libnvinfer.so.10").is_file():
            return root
    for sp in site.getsitepackages():
        root = Path(sp) / "tensorrt_libs"
        if (root / "libnvinfer.so.10").is_file():
            return root
    return None


def _preload_tensorrt_shared_libs(lib_dir: Path) -> None:
    """dlopen TensorRT .so before ORT TensorRT EP (LD_LIBRARY_PATH alone is unreliable)."""
    global _TRT_PRELOADED
    if _TRT_PRELOADED:
        return
    names = (
        "libnvinfer.so.10",
        "libnvinfer_plugin.so.10",
        "libnvonnxparser.so.10",
    )
    for name in names:
        path = lib_dir / name
        if not path.is_file():
            continue
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            logger.warning("Failed to preload %s: %s", path, exc)
    _TRT_PRELOADED = True


def ensure_ort_cuda_library_path() -> None:
    prefixes: list[str] = []
    try:
        torch = importlib.import_module("torch")
        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        if torch_lib.is_dir():
            prefixes.append(str(torch_lib))
    except Exception:
        pass

    trt_dir = _tensorrt_lib_dir()
    if trt_dir is not None:
        prefixes.append(str(trt_dir.resolve()))

    for sp in site.getsitepackages():
        sp_path = Path(sp)
        nvidia_root = sp_path / "nvidia"
        if nvidia_root.is_dir():
            for lib_dir in nvidia_root.rglob("lib"):
                if lib_dir.is_dir():
                    prefixes.append(str(lib_dir.resolve()))

    existing = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    for prefix in reversed(prefixes):
        if prefix not in existing:
            existing.insert(0, prefix)
    os.environ["LD_LIBRARY_PATH"] = ":".join(existing)

    if trt_dir is not None:
        _preload_tensorrt_shared_libs(trt_dir)


def ensure_ort_tensorrt_ready() -> bool:
    """Call before ``import onnxruntime`` when using TensorRT EP."""
    ensure_ort_cuda_library_path()
    trt_dir = _tensorrt_lib_dir()
    if trt_dir is None:
        logger.warning(
            "tensorrt_libs not found; install: pip install 'tensorrt-cu12==10.*'"
        )
        return False
    _preload_tensorrt_shared_libs(trt_dir)
    return True


__all__ = ["ensure_ort_cuda_library_path", "ensure_ort_tensorrt_ready"]
