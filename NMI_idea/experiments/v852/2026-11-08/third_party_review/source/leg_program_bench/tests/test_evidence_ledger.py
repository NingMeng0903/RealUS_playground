from __future__ import annotations

import pytest

from leg_program_bench.evidence_ledger import EvidenceLedger, EvidenceRecord


def packet(packet_id="p1", *, capture=1.0, received=1.2, release=1.5, region="sa", kind="historical", payload=None):
    return EvidenceRecord(
        packet_id,
        "obj-a",
        region,
        capture,
        received,
        release,
        {"feature": payload if payload is not None else [1, 2], "packet": packet_id},
        kind=kind,
    )


def test_duplicate_packet_is_idempotent_but_changed_content_is_rejected():
    ledger = EvidenceLedger()
    record = packet()
    assert ledger.add(record) is record
    assert ledger.add(packet()) is record
    with pytest.raises(ValueError, match="different content"):
        ledger.add(packet(payload=[9]))
    assert ledger.summary()["n_records"] == 1


def test_release_gate_reschedule_keeps_capture_and_payload_unchanged():
    ledger = EvidenceLedger()
    record = packet()
    ledger.add(record)
    before = record.payload
    ledger.reschedule_release("p1", 3.0)
    assert record.capture_time == 1.0
    assert record.payload == before
    assert ledger.visible(2.0, "obj-a") == ()
    assert ledger.visible(3.0, "obj-a")[0].packet_id == "p1"
    assert ledger.effective_release_time("p1") == 3.0


def test_fresh_filter_uses_capture_time_not_receive_time():
    ledger = EvidenceLedger()
    ledger.add(packet(capture=1.0, received=4.0, release=4.0))
    assert ledger.visible(4.0, "obj-a", fresh_after=2.0) == ()
    assert ledger.visible(4.0, "obj-a", fresh_after=1.0)[0].packet_id == "p1"


def test_identity_region_and_late_packet_are_isolated():
    ledger = EvidenceLedger()
    ledger.add(packet("a", region="sa", received=2.0, release=2.0))
    ledger.add(EvidenceRecord("b", "obj-b", "sa", 1.0, 1.0, 1.0, {"x": 1}, kind="fresh"))
    ledger.add(packet("c", region="la", received=1.0, release=1.0))
    assert [r.packet_id for r in ledger.visible(1.5, "obj-a", region_id="sa")] == []
    assert [r.packet_id for r in ledger.visible(2.0, "obj-a", region_id="sa")] == ["a"]
    assert [r.packet_id for r in ledger.visible(2.0, "obj-a", region_id="la")] == ["c"]
    assert [r.packet_id for r in ledger.visible(2.0, "obj-b")] == ["b"]


def test_hidden_packet_is_absent_from_visible_and_latest_feature():
    ledger = EvidenceLedger()
    ledger.add(packet("p1", capture=1.0, received=1.0, release=1.0))
    ledger.add(packet("p2", capture=2.0, received=2.0, release=2.0))
    ledger.set_hidden("p2", True)
    assert [r.packet_id for r in ledger.visible(3.0, "obj-a")] == ["p1"]
    assert ledger.latest_feature(3.0, "obj-a").packet_id == "p1"
    ledger.set_hidden("p2", False)
    assert ledger.latest_feature(3.0, "obj-a").packet_id == "p2"


def test_payload_and_audit_views_do_not_leak_mutable_references():
    ledger = EvidenceLedger()
    original = {"nested": {"values": [1, 2]}}
    ledger.add(EvidenceRecord("p1", "obj-a", "sa", 1.0, 1.0, 1.0, original))
    result = ledger.visible(2.0, "obj-a")[0]
    result.payload["nested"]["values"].append(99)
    original["nested"]["values"].append(100)
    assert ledger.latest_feature(2.0, "obj-a").payload == {"nested": {"values": [1, 2]}}
    events = ledger.events
    events[0]["type"] = "tampered"
    assert ledger.events[0]["type"] == "add"


def test_repeated_read_is_not_a_new_frame_and_clone_is_independent():
    ledger = EvidenceLedger()
    ledger.add(packet())
    first = ledger.visible(2.0, "obj-a")
    second = ledger.visible(2.0, "obj-a")
    assert [r.packet_id for r in first] == [r.packet_id for r in second] == ["p1"]
    assert ledger.summary()["n_records"] == 1
    clone = ledger.clone()
    clone.set_hidden("p1", True)
    assert ledger.latest_feature(2.0, "obj-a") is not None
    assert clone.latest_feature(2.0, "obj-a") is None


def test_historical_filter_and_invalid_controls():
    ledger = EvidenceLedger()
    ledger.add(packet("old", kind="historical"))
    ledger.add(packet("new", kind="fresh"))
    assert [r.packet_id for r in ledger.visible(2.0, "obj-a", include_historical=False)] == ["new"]
    with pytest.raises(KeyError):
        ledger.set_hidden("missing")
    with pytest.raises(KeyError):
        ledger.reschedule_release("missing", 1.0)

