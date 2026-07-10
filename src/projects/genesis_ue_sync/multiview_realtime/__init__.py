"""Realtime multi-view human tracking and Genesis visualization (camera-agnostic ingress)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from projects.genesis_ue_sync.multiview_realtime.config import MultiviewRealtimeConfig

if TYPE_CHECKING:
    from projects.genesis_ue_sync.multiview_realtime.pipeline import MultiviewRealtimeTracker

__all__ = ["MultiviewRealtimeConfig", "MultiviewRealtimeTracker"]


def __getattr__(name: str):
    if name == "MultiviewRealtimeTracker":
        from projects.genesis_ue_sync.multiview_realtime.pipeline import MultiviewRealtimeTracker

        return MultiviewRealtimeTracker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
