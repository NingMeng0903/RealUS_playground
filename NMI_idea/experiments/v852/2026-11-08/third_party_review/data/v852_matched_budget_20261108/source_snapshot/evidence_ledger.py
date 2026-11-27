"""Auditable evidence visibility and freshness ledger.

The ledger is deliberately independent of the scan simulator.  It separates a
packet's immutable acquisition facts from the release gate that determines when
an upper-level policy may read it.  A packet can therefore be released earlier
or later without changing its capture time or payload.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import copy
import math
from numbers import Real
from typing import Any, Iterator


class FrozenDict(Mapping):
    """Small read-only mapping used for private payload storage."""

    __slots__ = ("_data",)

    def __init__(self, values: Mapping[Any, Any]) -> None:
        self._data = dict(values)

    def __getitem__(self, key: Any) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"FrozenDict({self._data!r})"


def _freeze(value: Any) -> Any:
    """Copy common containers into recursively immutable values."""

    if isinstance(value, FrozenDict):
        return value
    if isinstance(value, Mapping):
        return FrozenDict({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    try:
        return copy.deepcopy(value)
    except Exception:
        # An opaque payload is still never returned directly; see _thaw.
        return value


def _thaw(value: Any) -> Any:
    """Return a detached, caller-owned payload copy."""

    if isinstance(value, FrozenDict):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, frozenset):
        return {_thaw(item) for item in value}
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _valid_time(value: Any, *, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, init=False)
class EvidenceRecord:
    """Immutable acquisition facts for one packet.

    ``payload`` is exposed as a detached copy.  Mutating the value returned by
    the property can therefore never mutate either this record or its ledger.
    """

    packet_id: str
    object_id: str
    region_id: str
    capture_time: float
    received_time: float
    release_time: float
    _payload: Any = field(repr=False)
    kind: str = "historical"

    def __init__(
        self,
        packet_id: str,
        object_id: str,
        region_id: str,
        capture_time: Real,
        received_time: Real,
        release_time: Real,
        payload: Any,
        kind: str = "historical",
    ) -> None:
        for name, value in (
            ("packet_id", packet_id),
            ("object_id", object_id),
            ("region_id", region_id),
        ):
            if not isinstance(value, str) or not value:
                raise TypeError(f"{name} must be a non-empty string")
        if not isinstance(kind, str) or not kind:
            raise TypeError("kind must be a non-empty string")
        object.__setattr__(self, "packet_id", packet_id)
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "region_id", region_id)
        object.__setattr__(self, "capture_time", _valid_time(capture_time, name="capture_time"))
        object.__setattr__(self, "received_time", _valid_time(received_time, name="received_time"))
        object.__setattr__(self, "release_time", _valid_time(release_time, name="release_time"))
        object.__setattr__(self, "_payload", _freeze(payload))
        object.__setattr__(self, "kind", kind)

    @property
    def payload(self) -> Any:
        return _thaw(self._payload)

    def content_key(self) -> tuple[Any, ...]:
        """Canonical key used for duplicate packet checks."""

        return (
            self.packet_id,
            self.object_id,
            self.region_id,
            self.capture_time,
            self.received_time,
            self.release_time,
            self._payload,
            self.kind,
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            "EvidenceRecord("
            f"packet_id={self.packet_id!r}, object_id={self.object_id!r}, "
            f"region_id={self.region_id!r}, capture_time={self.capture_time!r}, "
            f"received_time={self.received_time!r}, release_time={self.release_time!r}, "
            f"payload={self.payload!r}, kind={self.kind!r})"
        )


class EvidenceLedger:
    """Store packets and apply auditable visibility gates."""

    def __init__(self) -> None:
        self._records: dict[str, EvidenceRecord] = {}
        self._release_overrides: dict[str, float] = {}
        self._hidden_ids: set[str] = set()
        self._events: list[dict[str, Any]] = []

    @property
    def hidden_ids(self) -> frozenset[str]:
        return frozenset(self._hidden_ids)

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        """Detached audit events; changing the returned dicts is harmless."""

        return tuple(copy.deepcopy(event) for event in self._events)

    def _event(self, event_type: str, **fields: Any) -> None:
        self._events.append({"type": event_type, **fields})

    def add(self, record: EvidenceRecord) -> EvidenceRecord:
        if not isinstance(record, EvidenceRecord):
            raise TypeError("record must be an EvidenceRecord")
        previous = self._records.get(record.packet_id)
        if previous is None:
            self._records[record.packet_id] = record
            self._event("add", packet_id=record.packet_id)
            return record
        if previous.content_key() != record.content_key():
            self._event("reject_duplicate", packet_id=record.packet_id)
            raise ValueError(f"packet_id already exists with different content: {record.packet_id}")
        self._event("add_idempotent", packet_id=record.packet_id)
        return previous

    def _require(self, packet_id: str) -> EvidenceRecord:
        try:
            return self._records[packet_id]
        except KeyError as exc:
            raise KeyError(f"unknown packet_id: {packet_id}") from exc

    def effective_release_time(self, packet_id: str) -> float:
        record = self._require(packet_id)
        return self._release_overrides.get(packet_id, record.release_time)

    def reschedule_release(self, packet_id: str, release_time: Real) -> None:
        self._require(packet_id)
        gate = _valid_time(release_time, name="release_time")
        self._release_overrides[packet_id] = gate
        self._event("reschedule_release", packet_id=packet_id, release_time=gate)

    def set_hidden(self, packet_id: str, hidden: bool = True) -> None:
        self._require(packet_id)
        if hidden:
            self._hidden_ids.add(packet_id)
        else:
            self._hidden_ids.discard(packet_id)
        self._event("set_hidden", packet_id=packet_id, hidden=bool(hidden))

    def visible(
        self,
        now: Real,
        object_id: str,
        region_id: str | None = None,
        fresh_after: Real | None = None,
        include_historical: bool = True,
    ) -> tuple[EvidenceRecord, ...]:
        """Return only packets legally visible at ``now``.

        Freshness is based on immutable capture time, never receipt or release
        time.  A late packet cannot become a new observation merely because it
        was read after the request.
        """

        current = _valid_time(now, name="now")
        if not isinstance(object_id, str) or not object_id:
            raise TypeError("object_id must be a non-empty string")
        threshold = None if fresh_after is None else _valid_time(fresh_after, name="fresh_after")
        result = []
        for record in self._records.values():
            if record.packet_id in self._hidden_ids or record.object_id != object_id:
                continue
            if region_id is not None and record.region_id != region_id:
                continue
            if not include_historical and record.kind == "historical":
                continue
            if record.received_time > current or self.effective_release_time(record.packet_id) > current:
                continue
            if threshold is not None and record.capture_time < threshold:
                continue
            result.append(record)
        result.sort(key=lambda item: (item.capture_time, item.received_time, item.packet_id))
        return tuple(result)

    def latest_feature(
        self,
        now: Real,
        object_id: str,
        region_id: str | None = None,
        fresh_after: Real | None = None,
        include_historical: bool = True,
    ) -> EvidenceRecord | None:
        visible = self.visible(
            now,
            object_id,
            region_id=region_id,
            fresh_after=fresh_after,
            include_historical=include_historical,
        )
        return visible[-1] if visible else None

    def clone(self) -> "EvidenceLedger":
        cloned = EvidenceLedger()
        cloned._records = dict(self._records)
        cloned._release_overrides = dict(self._release_overrides)
        cloned._hidden_ids = set(self._hidden_ids)
        cloned._events = copy.deepcopy(self._events)
        return cloned

    def summary(self) -> dict[str, Any]:
        return {
            "n_records": len(self._records),
            "n_hidden": len(self._hidden_ids),
            "n_release_overrides": len(self._release_overrides),
            "n_events": len(self._events),
            "object_ids": tuple(sorted({record.object_id for record in self._records.values()})),
            "region_ids": tuple(sorted({record.region_id for record in self._records.values()})),
            "hidden_ids": tuple(sorted(self._hidden_ids)),
        }

