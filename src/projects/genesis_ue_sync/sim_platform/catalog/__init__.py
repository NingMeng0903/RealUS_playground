"""BEDLAM avatar catalog."""

from projects.genesis_ue_sync.sim_platform.catalog.bedlam_avatar_index import (
    BedlamAvatarIndex,
    BedlamAvatarRecord,
    default_avatar_index_path,
    load_bedlam_avatar_index,
)

__all__ = [
    "BedlamAvatarIndex",
    "BedlamAvatarRecord",
    "default_avatar_index_path",
    "load_bedlam_avatar_index",
]
