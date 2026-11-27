"""Applied-state, strict multi-camera UE benchmark recorder.

The normal UE preview path is intentionally latest-only.  This recorder uses
the same TCP mux wire format but keeps every frame grouped by the explicit
``capture_group_id`` emitted by a one-shot UE capture request.  A group is
accepted only after the UE ACK and after all requested cameras have arrived.
"""

from __future__ import annotations

import json
import logging
import re
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


LOG = logging.getLogger(__name__)
DEFAULT_TOPIC = "amongus_camera_frame_v1"
DEFAULT_RUNTIME_ACK = ""
DEFAULT_ACK_CONNECT = "tcp://127.0.0.1:17367"
_SAFE = re.compile(r"[^A-Za-z0-9_.=-]+")


def _parse_endpoint(raw: str, *, default_scheme: str) -> tuple[str, int]:
    text = str(raw or "").strip()
    if "://" in text:
        scheme, text = text.split("://", 1)
        if scheme and scheme != default_scheme:
            raise ValueError(f"Expected {default_scheme} endpoint, got {raw!r}")
    if ":" not in text:
        raise ValueError(f"Endpoint must include host:port: {raw!r}")
    host, port = text.rsplit(":", 1)
    return host or "127.0.0.1", int(port)


def capture_group_id(session_id: str, loop_id: str, frame_index: int) -> str:
    """Return a filesystem-safe deterministic identity for one applied frame."""
    return "__".join(
        (
            _SAFE.sub("_", str(session_id).strip() or "session"),
            _SAFE.sub("_", str(loop_id).strip() or "loop"),
            f"f{int(frame_index):08d}",
        )
    )


def _matrix3(value: Any) -> list[list[float]] | None:
    try:
        rows = [list(row) for row in value]
        if len(rows) != 3 or any(len(row) != 3 for row in rows):
            return None
        out = [[float(v) for v in row] for row in rows]
        if not all(abs(v) < float("inf") for row in out for v in row):
            return None
        return out
    except Exception:
        return None


def validate_capture_metadata(
    metadata: dict[str, Any],
    *,
    capture_group: str,
    frame_index: int,
    session_id: str = "",
    loop_id: str = "",
    sim_time_ns: int | None = None,
    motion_frame_index: int | None = None,
    state_hash: str = "",
) -> list[str]:
    """Validate the frame-level identity and calibrated synthetic image contract."""
    errors: list[str] = []
    if str(metadata.get("capture_group_id") or "") != str(capture_group):
        errors.append("capture_group_id_mismatch")
    if session_id and str(metadata.get("session_id") or "") != str(session_id):
        errors.append("session_id_mismatch")
    if loop_id and str(metadata.get("loop_id") or "") != str(loop_id):
        errors.append("loop_id_mismatch")
    for field in ("frame_index", "applied_canonical_frame_index", "requested_frame_index"):
        if field in metadata:
            try:
                if int(metadata[field]) != int(frame_index):
                    errors.append(f"{field}_mismatch")
            except (TypeError, ValueError):
                errors.append(f"{field}_invalid")
    if str(metadata.get("source_time_domain") or "") != "genesis_sim":
        errors.append("source_time_domain_not_genesis_sim")
    if sim_time_ns is not None and "applied_canonical_sim_time_ns" in metadata:
        try:
            if int(metadata["applied_canonical_sim_time_ns"]) != int(sim_time_ns):
                errors.append("applied_canonical_sim_time_ns_mismatch")
        except (TypeError, ValueError):
            errors.append("applied_canonical_sim_time_ns_invalid")
    if sim_time_ns is not None and "requested_sim_time_ns" in metadata:
        try:
            if int(metadata["requested_sim_time_ns"]) != int(sim_time_ns):
                errors.append("requested_sim_time_ns_mismatch")
        except (TypeError, ValueError):
            errors.append("requested_sim_time_ns_invalid")
    if motion_frame_index is not None and "applied_motion_frame_index" in metadata:
        try:
            if int(metadata["applied_motion_frame_index"]) != int(motion_frame_index):
                errors.append("applied_motion_frame_index_mismatch")
        except (TypeError, ValueError):
            errors.append("applied_motion_frame_index_invalid")
    if state_hash and str(metadata.get("canonical_state_hash") or "") != str(state_hash):
        errors.append("canonical_state_hash_mismatch")
    intrinsics = metadata.get("intrinsics")
    K = intrinsics.get("K") if isinstance(intrinsics, dict) else None
    if _matrix3(K) is None:
        errors.append("intrinsics_K_missing_or_invalid")
    else:
        try:
            if float(K[0][0]) <= 0.0 or float(K[1][1]) <= 0.0 or float(K[2][2]) <= 0.0:
                errors.append("intrinsics_K_nonpositive")
        except Exception:
            errors.append("intrinsics_K_invalid")
    geometry = metadata.get("image_geometry")
    if not isinstance(geometry, dict) or not bool(geometry.get("undistorted")):
        errors.append("image_geometry_not_undistorted")
    if not isinstance(geometry, dict) or str(geometry.get("projection_distortion_model") or "") != "zero":
        errors.append("image_geometry_distortion_not_zero")
    distortion = intrinsics.get("distortion") if isinstance(intrinsics, dict) else None
    try:
        if distortion is None or any(abs(float(value)) > 1.0e-9 for value in distortion):
            errors.append("distortion_not_zero")
    except Exception:
        errors.append("distortion_invalid")
    return errors


@dataclass(frozen=True)
class CameraFramePacket:
    metadata: dict[str, Any]
    image_bytes: bytes
    received_monotonic_ns: int


class StrictCameraFrameCollector:
    """Collect explicit capture groups from the TCP mux without latest-drop behavior."""

    def __init__(
        self,
        connect: str,
        *,
        camera_names: Iterable[str],
        topic: str = DEFAULT_TOPIC,
        recv_timeout_ms: int = 100,
        rcv_hwm: int = 512,
    ) -> None:
        self.connect_endpoint = str(connect)
        self.camera_names = tuple(str(name) for name in camera_names)
        self.topic = str(topic)
        self.recv_timeout_ms = max(int(recv_timeout_ms), 1)
        self.rcv_hwm = max(int(rcv_hwm), 4)
        self._ctx = None
        self._sock = None
        self._groups: dict[str, dict[str, CameraFramePacket]] = {}
        self._quarantined = 0
        self._duplicate = 0

    @property
    def diagnostics(self) -> dict[str, int]:
        return {"quarantined": int(self._quarantined), "duplicate": int(self._duplicate)}

    def connect(self) -> None:
        try:
            import zmq
        except ImportError as exc:
            raise RuntimeError("pyzmq is required for UE benchmark recording") from exc
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVTIMEO, self.recv_timeout_ms)
        self._sock.setsockopt(zmq.RCVHWM, self.rcv_hwm)
        self._sock.setsockopt(zmq.SUBSCRIBE, self.topic.encode("utf-8"))
        self._sock.connect(self.connect_endpoint)

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close(0)
            except Exception:
                pass
        self._sock = None

    def poll_once(self) -> bool:
        if self._sock is None:
            raise RuntimeError("StrictCameraFrameCollector.connect() was not called")
        try:
            import zmq

            parts = self._sock.recv_multipart()
        except zmq.Again:
            return False
        if len(parts) < 3:
            self._quarantined += 1
            return True
        try:
            metadata = json.loads(parts[1].decode("utf-8"))
            if not isinstance(metadata, dict):
                raise TypeError("metadata is not an object")
            group = str(metadata.get("capture_group_id") or "")
            camera = str(metadata.get("camera_name") or metadata.get("camera_frame_id") or "")
            if not group or camera not in self.camera_names:
                self._quarantined += 1
                return True
            bucket = self._groups.setdefault(group, {})
            if camera in bucket:
                self._duplicate += 1
                return True
            bucket[camera] = CameraFramePacket(
                metadata=metadata,
                image_bytes=bytes(parts[2]),
                received_monotonic_ns=time.monotonic_ns(),
            )
        except Exception:
            self._quarantined += 1
        return True

    def collect_group(
        self,
        capture_group: str,
        *,
        frame_index: int,
        session_id: str = "",
        loop_id: str = "",
        sim_time_ns: int | None = None,
        motion_frame_index: int | None = None,
        state_hash: str = "",
        timeout_s: float = 10.0,
    ) -> dict[str, CameraFramePacket]:
        deadline = time.monotonic() + max(float(timeout_s), 0.1)
        expected = set(self.camera_names)
        while time.monotonic() < deadline:
            bucket = self._groups.get(str(capture_group), {})
            if expected.issubset(bucket):
                selected = {name: bucket[name] for name in self.camera_names}
                errors: list[str] = []
                for camera, packet in selected.items():
                    errors.extend(
                        f"{camera}:{error}"
                        for error in validate_capture_metadata(
                            packet.metadata,
                            capture_group=capture_group,
                            frame_index=frame_index,
                            session_id=session_id,
                            loop_id=loop_id,
                            sim_time_ns=sim_time_ns,
                            motion_frame_index=motion_frame_index,
                            state_hash=state_hash,
                        )
                    )
                if errors:
                    raise ValueError("invalid UE capture metadata: " + ",".join(errors))
                self._groups.pop(str(capture_group), None)
                return selected
            self.poll_once()
        have = sorted(self._groups.get(str(capture_group), {}))
        raise TimeoutError(f"capture group {capture_group!r} timed out; have={have} want={self.camera_names}")


def request_ue_capture(
    endpoint: str,
    request: dict[str, Any],
    *,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    host, port = _parse_endpoint(endpoint, default_scheme="tcp")
    with socket.create_connection((host, port), timeout=max(float(timeout_s), 1.0)) as conn:
        conn.settimeout(max(float(timeout_s), 1.0))
        conn.sendall((json.dumps(request, ensure_ascii=True, separators=(",", ":")) + "\n").encode("utf-8"))
        chunks: list[bytes] = []
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
    raw = b"".join(chunks).split(b"\n", 1)[0].strip()
    if not raw:
        raise RuntimeError("UE benchmark ACK endpoint returned an empty response")
    response = json.loads(raw.decode("utf-8"))
    if not isinstance(response, dict):
        raise TypeError("UE benchmark ACK must be a JSON object")
    return response


def _canonical_state_from_record(record: dict[str, Any]) -> dict[str, Any]:
    state = record.get("canonical_state")
    if isinstance(state, dict):
        return dict(state)
    payload = record.get("payload")
    if isinstance(payload, dict) and isinstance(payload.get("canonical_state"), dict):
        return dict(payload["canonical_state"])
    return dict(record)


def iter_canonical_jsonl(path: Path, *, frame_start: int | None, frame_end: int | None) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid canonical JSONL line {line_no}: {exc}") from exc
            if not isinstance(record, dict):
                continue
            state = _canonical_state_from_record(record)
            try:
                frame = int(state.get("frame_index", state.get("sim_step_index")))
            except (TypeError, ValueError):
                raise ValueError(f"Canonical line {line_no} has no frame_index")
            if frame_start is not None and frame < int(frame_start):
                continue
            if frame_end is not None and frame > int(frame_end):
                continue
            state["frame_index"] = frame
            yield state


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


class UEBenchmarkRecorder:
    def __init__(
        self,
        *,
        canonical_jsonl: Path,
        output_root: Path,
        session_id: str,
        loop_id: str,
        camera_connect: str,
        ack_connect: str,
        camera_names: Iterable[str],
        topic: str = DEFAULT_TOPIC,
        runtime_ack_connect: str | None = DEFAULT_RUNTIME_ACK,
        frame_start: int | None = None,
        frame_end: int | None = None,
        max_frames: int | None = None,
        timeout_s: float = 30.0,
        capture_timeout_s: float = 10.0,
        deduplicate_motion_frames: bool = True,
    ) -> None:
        self.canonical_jsonl = Path(canonical_jsonl).expanduser().resolve()
        self.output_root = Path(output_root).expanduser().resolve()
        self.session_id = str(session_id)
        self.loop_id = str(loop_id)
        self.camera_names = tuple(str(name) for name in camera_names)
        if len(set(self.camera_names)) != len(self.camera_names) or not self.camera_names:
            raise ValueError("camera_names must be unique and nonempty")
        self.camera_connect = str(camera_connect)
        self.ack_connect = str(ack_connect)
        self.topic = str(topic)
        self.runtime_ack_connect = runtime_ack_connect
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.max_frames = max_frames
        self.timeout_s = float(timeout_s)
        self.capture_timeout_s = float(capture_timeout_s)
        self.deduplicate_motion_frames = bool(deduplicate_motion_frames)

    def _runtime_ack(self, payload: dict[str, Any]) -> None:
        if not self.runtime_ack_connect:
            return
        host, port = _parse_endpoint(str(self.runtime_ack_connect), default_scheme="udp")
        message = {"topic": "ue_benchmark_capture_ack_v1", "payload": payload}
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendto(json.dumps(message, ensure_ascii=True, separators=(",", ":")).encode("utf-8"), (host, port))

    def _save_group(
        self,
        *,
        state: dict[str, Any],
        loop_id: str,
        group_id: str,
        ack: dict[str, Any],
        packets: dict[str, CameraFramePacket],
    ) -> dict[str, Any]:
        frame_index = int(state["frame_index"])
        group_dir = self.output_root / "frames" / group_id
        group_dir.mkdir(parents=True, exist_ok=False)
        metadata_by_camera: dict[str, Any] = {}
        image_paths: dict[str, str] = {}
        for camera, packet in packets.items():
            image_path = group_dir / f"{_SAFE.sub('_', camera)}.jpg"
            image_path.write_bytes(packet.image_bytes)
            image_paths[camera] = str(image_path.relative_to(self.output_root))
            metadata_by_camera[camera] = packet.metadata
        _write_json(group_dir / "metadata.json", {"by_camera": metadata_by_camera})
        _write_json(group_dir / "canonical.json", state)
        _write_json(group_dir / "gt.json", ack.get("gt") or {})
        record = {
            "status": "ok",
            "session_id": self.session_id,
            "loop_id": str(loop_id),
            "capture_group_id": group_id,
            "frame_index": frame_index,
            "sim_time_ns": state.get("sim_time_ns"),
            "image_paths": image_paths,
            "metadata_path": str((group_dir / "metadata.json").relative_to(self.output_root)),
            "canonical_path": str((group_dir / "canonical.json").relative_to(self.output_root)),
            "gt_path": str((group_dir / "gt.json").relative_to(self.output_root)),
            "ue_ack": ack,
        }
        return record

    def run(self) -> dict[str, Any]:
        if not self.canonical_jsonl.is_file():
            raise FileNotFoundError(self.canonical_jsonl)
        self.output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.output_root / "manifest.jsonl"
        collector = StrictCameraFrameCollector(
            self.camera_connect,
            camera_names=self.camera_names,
            topic=self.topic,
        )
        records: list[dict[str, Any]] = []
        seen_motion_loop: set[tuple[str, int]] = set()
        try:
            collector.connect()
            for state in iter_canonical_jsonl(
                self.canonical_jsonl,
                frame_start=self.frame_start,
                frame_end=self.frame_end,
            ):
                if self.max_frames is not None and len(records) >= int(self.max_frames):
                    break
                frame_index = int(state["frame_index"])
                # Preserve an explicitly supplied zero clock.  The first
                # canonical frame is valid at t=0, and replacing it from a
                # guessed playback rate would make the recorded identity
                # differ from the applied UE clock.  Only synthesize a clock
                # when the source row omits it entirely.
                sim_time_value = state.get("sim_time_ns")
                if sim_time_value is None:
                    sim_time_ns = int(round(frame_index * 1_000_000_000 / 30.0))
                    state["sim_time_ns"] = sim_time_ns
                else:
                    sim_time_ns = int(sim_time_value)
                human = state.get("human") if isinstance(state.get("human"), dict) else {}
                loop_index = human.get("loop_index", state.get("loop_index", 0))
                try:
                    loop_index_int = int(loop_index)
                except (TypeError, ValueError):
                    loop_index_int = 0
                loop_id = str(human.get("loop_id") or state.get("loop_id") or f"{self.loop_id}:l{loop_index_int}")
                motion_value = human.get("motion_frame_index", frame_index)
                try:
                    motion_frame_index = int(motion_value)
                except (TypeError, ValueError):
                    motion_frame_index = frame_index
                motion_key = (loop_id, motion_frame_index)
                if self.deduplicate_motion_frames and motion_key in seen_motion_loop:
                    continue
                seen_motion_loop.add(motion_key)
                group_id = capture_group_id(self.session_id, loop_id, frame_index)
                request = {
                    "command_type": "capture_canonical_group",
                    "request_id": f"{self.session_id}:{loop_id}:{frame_index}",
                    "session_id": self.session_id,
                    "loop_id": loop_id,
                    "capture_group_id": group_id,
                    "canonical_frame_index": frame_index,
                    "canonical_sim_time_ns": sim_time_ns,
                    "camera_names": list(self.camera_names),
                    "canonical_state": state,
                }
                ack = request_ue_capture(self.ack_connect, request, timeout_s=self.timeout_s)
                if not bool(ack.get("success")):
                    raise RuntimeError(f"UE capture failed for {group_id}: {ack}")
                if str(ack.get("capture_group_id") or "") != group_id:
                    raise RuntimeError(f"UE ACK group mismatch: expected={group_id} got={ack}")
                expected_motion_frame = int(motion_frame_index)
                for field, expected in (
                    ("requested_frame_index", frame_index),
                    ("applied_frame_index", frame_index),
                    ("requested_sim_time_ns", sim_time_ns),
                    ("applied_sim_time_ns", sim_time_ns),
                    ("applied_motion_frame_index", expected_motion_frame),
                ):
                    if field not in ack:
                        raise RuntimeError(f"UE ACK missing {field} for {group_id}: {ack}")
                    try:
                        actual = int(ack[field])
                    except (TypeError, ValueError) as exc:
                        raise RuntimeError(f"UE ACK invalid {field} for {group_id}: {ack}") from exc
                    if actual != int(expected):
                        raise RuntimeError(
                            f"UE ACK {field} mismatch for {group_id}: expected={expected} got={actual}"
                        )
                packets = collector.collect_group(
                    group_id,
                    frame_index=frame_index,
                    session_id=self.session_id,
                    loop_id=loop_id,
                    sim_time_ns=int(ack.get("applied_sim_time_ns", sim_time_ns)),
                    motion_frame_index=int(
                        ack.get(
                            "applied_motion_frame_index",
                            (human.get("motion_frame_index", frame_index) if isinstance(human, dict) else frame_index),
                        )
                    ),
                    state_hash=str(ack.get("applied_state_hash") or ""),
                    timeout_s=self.capture_timeout_s,
                )
                record = self._save_group(
                    state=state,
                    loop_id=loop_id,
                    group_id=group_id,
                    ack=ack,
                    packets=packets,
                )
                with manifest_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                records.append(record)
                self._runtime_ack({
                    "success": True,
                    "session_id": self.session_id,
                    "loop_id": loop_id,
                    "capture_group_id": group_id,
                    "frame_index": frame_index,
                    "sim_time_ns": sim_time_ns,
                    "camera_names": list(self.camera_names),
                    "manifest_record": record,
                    "ue_ack": ack,
                })
        finally:
            collector.close()
        summary = {
            "schema_version": 1,
            "source": "ue_strict_benchmark_recorder",
            "session_id": self.session_id,
            "loop_id": self.loop_id,
            "camera_names": list(self.camera_names),
            "frame_count": len(records),
            "manifest_jsonl": str(manifest_path),
            "collector": collector.diagnostics,
        }
        _write_json(self.output_root / "manifest.json", summary)
        return summary


__all__ = [
    "CameraFramePacket",
    "StrictCameraFrameCollector",
    "UEBenchmarkRecorder",
    "capture_group_id",
    "iter_canonical_jsonl",
    "request_ue_capture",
    "validate_capture_metadata",
]
