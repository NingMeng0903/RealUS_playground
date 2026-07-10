"""SceneCapture JPEG axis correction (UE-safe: no tracking package imports)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

_AGENT_DEBUG_LOG = "/home/camp/.cursor/debug-logs/debug-05706c.log"


def _agent_debug_log(*, hypothesis_id: str, location: str, message: str, data: dict, run_id: str = "post-fix") -> None:
    payload = {
        "sessionId": "05706c",
        "runId": str(run_id),
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(_AGENT_DEBUG_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass


@dataclass(frozen=True)
class CameraImageCorrection:
    flip_u: bool = False
    flip_v: bool = False
    reason: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None, *, reason: str = "") -> "CameraImageCorrection":
        payload = dict(payload or {})
        if bool(payload.get("rotate_180", False)):
            return cls(flip_u=True, flip_v=True, reason=reason or "rotate_180")
        return cls(
            flip_u=bool(payload.get("flip_u", payload.get("flip_x", False))),
            flip_v=bool(payload.get("flip_v", payload.get("flip_y", False))),
            reason=reason,
        )

    def apply_uv(self, uv: np.ndarray, *, width: int, height: int) -> np.ndarray:
        out = np.asarray(uv, dtype=np.float64).copy()
        w = max(int(width), 1)
        h = max(int(height), 1)
        if self.flip_u:
            out[..., 0] = float(w - 1) - out[..., 0]
        if self.flip_v:
            out[..., 1] = float(h - 1) - out[..., 1]
        return out

    def as_dict(self) -> dict[str, bool]:
        return {"flip_u": bool(self.flip_u), "flip_v": bool(self.flip_v)}

    @property
    def is_identity(self) -> bool:
        return not (self.flip_u or self.flip_v)


def apply_correction_to_rgb(rgb: np.ndarray, correction: CameraImageCorrection) -> np.ndarray:
    if correction.is_identity:
        return np.asarray(rgb, dtype=np.uint8)
    out = np.asarray(rgb, dtype=np.uint8)
    if correction.flip_u:
        out = np.flip(out, axis=1)
    if correction.flip_v:
        out = np.flip(out, axis=0)
    return np.ascontiguousarray(out)


def _normalize3(vec: Any) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v
    return v / n


def metadata_has_scene_capture_flip(meta: dict[str, Any]) -> bool:
    return any(
        k in meta
        for k in (
            "scene_capture_flip_u",
            "scene_capture_flip_v",
            "flip_u",
            "flip_v",
            "flip_x",
            "flip_y",
            "rotate_180",
            "image_flip_u",
            "image_flip_v",
        )
    )


def correction_from_metadata(camera_id: str, meta: dict[str, Any]) -> CameraImageCorrection:
    return CameraImageCorrection.from_dict(
        {
            "flip_u": meta.get(
                "scene_capture_flip_u",
                meta.get("flip_u", meta.get("flip_x", meta.get("image_flip_u", False))),
            ),
            "flip_v": meta.get(
                "scene_capture_flip_v",
                meta.get("flip_v", meta.get("flip_y", meta.get("image_flip_v", False))),
            ),
            "rotate_180": meta.get("rotate_180", False),
        },
        reason=f"{camera_id}: camera metadata",
    )


def derive_scene_capture_image_correction_from_spec(camera_spec) -> CameraImageCorrection:
    """Resolve SceneCapture JPEG axis correction at UE spawn.

    All cameras share the same SceneCapture2D spawn path. Side cameras (cam_left/cam_right)
    use identity flip and match Genesis. Near-nadir cam_top may still need scene_capture_flip_u/v
    after a well-conditioned up vector: actor rotator matches OpenCV, but SceneCapture2D render-target
    readback can differ by pi roll; flip metadata is applied once at JPEG encode in UE spawn.
    """
    name = str(getattr(camera_spec, "name", "camera"))
    meta = dict(getattr(camera_spec, "metadata", None) or {})
    if metadata_has_scene_capture_flip(meta):
        correction = CameraImageCorrection.from_dict(
            {
                "flip_u": meta.get(
                    "scene_capture_flip_u",
                    meta.get("flip_u", meta.get("flip_x", meta.get("image_flip_u", False))),
                ),
                "flip_v": meta.get(
                    "scene_capture_flip_v",
                    meta.get("flip_v", meta.get("flip_y", meta.get("image_flip_v", False))),
                ),
                "rotate_180": meta.get("rotate_180", False),
            },
            reason=f"{name}: scene camera metadata",
        )
    else:
        correction = CameraImageCorrection(reason=f"{name}: identity (same spawn as side cameras)")
    # #region agent log
    if name == "cam_top":
        _agent_debug_log(
            hypothesis_id="B",
            location="bridge/core/scene_capture_image_correction.py:derive_scene_capture_image_correction_from_spec",
            message="cam_top spawn image correction resolved",
            data={
                "correction": correction.as_dict(),
                "reason": str(correction.reason),
                "metadata_keys": sorted(str(k) for k in meta.keys()),
                "metadata_has_scene_capture_flip": bool(metadata_has_scene_capture_flip(meta)),
            },
        )
    # #endregion
    return correction
