"""Patch fixed-batch YOLOX ONNX exports for dynamic batch inference.

DWPose YOLOX ONNX files use reshape constants ``[1, 85, -1]`` that collapse
multiview batches. Replacing the leading ``1`` with ``0`` (copy input dim 0)
yields correct ``[N, 8400, 85]`` outputs while keeping batch=1 numerically identical.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference

_RESHAPE_INIT_NAMES: tuple[str, ...] = ("857", "865", "873")


def _mark_batch_dynamic(model: onnx.ModelProto) -> None:
    for tensor in (model.graph.input[0], model.graph.output[0]):
        tensor.type.tensor_type.shape.dim[0].dim_param = "batch"
        tensor.type.tensor_type.shape.dim[0].ClearField("dim_value")


def patch_yolox_dynbatch_onnx(src: Path, dst: Path) -> Path:
    """Write a dynamic-batch YOLOX ONNX next to the source export."""
    src = Path(src)
    dst = Path(dst)
    if not src.is_file():
        raise FileNotFoundError(f"YOLOX ONNX not found: {src}")

    model = onnx.load(str(src))
    _mark_batch_dynamic(model)

    patched = 0
    for init in model.graph.initializer:
        if init.name not in _RESHAPE_INIT_NAMES:
            continue
        arr = numpy_helper.to_array(init).astype(np.int64).copy()
        if arr.size < 3 or int(arr[0]) != 1:
            continue
        arr[0] = 0
        init.CopyFrom(numpy_helper.from_array(arr, init.name))
        patched += 1

    if patched != len(_RESHAPE_INIT_NAMES):
        raise RuntimeError(
            f"Expected to patch {len(_RESHAPE_INIT_NAMES)} reshape constants, got {patched} in {src}"
        )

    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass
    _mark_batch_dynamic(model)

    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst))
    return dst


def default_dynbatch_path(detector_onnx_path: Path) -> Path:
    detector_onnx_path = Path(detector_onnx_path)
    return detector_onnx_path.with_name(f"{detector_onnx_path.stem}_dynbatch{detector_onnx_path.suffix}")


def _is_yolox_dynbatch_patched(model_path: Path) -> bool:
    try:
        model = onnx.load(str(model_path))
    except Exception:
        return False
    try:
        output_dim0 = model.graph.output[0].type.tensor_type.shape.dim[0]
        if output_dim0.HasField("dim_value") and int(output_dim0.dim_value) == 1:
            return False
        input_dim0 = model.graph.input[0].type.tensor_type.shape.dim[0]
        if input_dim0.HasField("dim_value") and int(input_dim0.dim_value) == 1:
            return False
        init_by_name = {init.name: init for init in model.graph.initializer}
        for name in _RESHAPE_INIT_NAMES:
            init = init_by_name.get(name)
            if init is None:
                return False
            arr = numpy_helper.to_array(init).astype(np.int64)
            if arr.size < 3 or int(arr[0]) != 0:
                return False
    except Exception:
        return False
    return True


def ensure_yolox_dynbatch_onnx(detector_onnx_path: Path, *, force: bool = False) -> Path:
    """Return dynamic-batch ONNX path, creating it from the fixed-batch export if needed."""
    detector_onnx_path = Path(detector_onnx_path)
    dst = default_dynbatch_path(detector_onnx_path)
    if dst.is_file() and not force and _is_yolox_dynbatch_patched(dst):
        return dst
    return patch_yolox_dynbatch_onnx(detector_onnx_path, dst)


__all__ = [
    "default_dynbatch_path",
    "ensure_yolox_dynbatch_onnx",
    "patch_yolox_dynbatch_onnx",
]
