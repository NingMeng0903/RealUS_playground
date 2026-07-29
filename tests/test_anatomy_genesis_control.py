from types import SimpleNamespace

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.genesis_control import (
    AnatomyAssetRegistry,
)


class _Drawer:
    def __init__(self) -> None:
        self.asset = SimpleNamespace(metadata={"shape_hash": "asset-shape"})
        self.draw_calls = 0
        self.clear_calls = 0

    def draw(self, pose_axis_angle, *, transl=None):
        self.draw_calls += 1
        return True

    def clear_node(self) -> None:
        self.clear_calls += 1


def test_shape_hash_mismatch_is_diagnostic_only() -> None:
    registry = AnatomyAssetRegistry.__new__(AnatomyAssetRegistry)
    drawer = _Drawer()
    registry._drawers = {"patient": drawer}

    drawn = registry.draw_all(
        np.zeros((55, 3), dtype=np.float32),
        shape_hash="different-live-shape",
    )

    assert drawn is True
    assert drawer.draw_calls == 1
    assert drawer.clear_calls == 0
