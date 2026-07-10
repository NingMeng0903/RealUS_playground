from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class ViTFeatureSnapshot:
    token_features: Any
    spatial_features: Any
    patch_shape: tuple[int, int]

    def cpu(self) -> "ViTFeatureSnapshot":
        return ViTFeatureSnapshot(
            token_features=self.token_features.detach().cpu(),
            spatial_features=self.spatial_features.detach().cpu(),
            patch_shape=self.patch_shape,
        )


class ViTFeatureTap:
    """Cache ViT patch-token features from ``last_norm`` or a specific transformer block."""

    def __init__(
        self,
        backbone: Any,
        *,
        hook_target: Literal["last_norm", "block"] = "last_norm",
        block_index: int | None = None,
    ) -> None:
        if not hasattr(backbone, "last_norm") or not hasattr(backbone, "patch_embed"):
            raise TypeError("ViTFeatureTap requires a ViT-style backbone with last_norm and patch_embed.")
        self.backbone = backbone
        self._hook_target = hook_target
        self._block_index = int(block_index) if block_index is not None else None
        self._handle = None
        self._latest: ViTFeatureSnapshot | None = None
        if hook_target == "block" and self._block_index is None:
            raise ValueError("hook_target='block' requires block_index.")
        if hook_target not in {"last_norm", "block"}:
            raise ValueError(f"Unsupported hook_target: {hook_target!r}")

    def _hook(self, module, inputs, output) -> None:  # noqa: ANN001
        tokens = output.detach().clone()
        if tokens.ndim != 3:
            raise ValueError(f"Expected ViT token tensor with shape (B, N, C), got {tuple(tokens.shape)}")
        patch_shape = tuple(int(v) for v in getattr(self.backbone.patch_embed, "patch_shape", (0, 0)))
        hp, wp = patch_shape
        if hp <= 0 or wp <= 0 or hp * wp != int(tokens.shape[1]):
            raise ValueError(
                f"Invalid patch shape {patch_shape} for token count {tokens.shape[1]}; "
                "check the input crop and ViT patch geometry."
            )
        spatial = tokens.permute(0, 2, 1).reshape(tokens.shape[0], tokens.shape[2], hp, wp).contiguous()
        self._latest = ViTFeatureSnapshot(
            token_features=tokens,
            spatial_features=spatial,
            patch_shape=(hp, wp),
        )

    def attach(self) -> "ViTFeatureTap":
        if self._handle is None:
            if self._hook_target == "last_norm":
                module = self.backbone.last_norm
            else:
                blocks = getattr(self.backbone, "blocks", None)
                if blocks is None:
                    raise TypeError("Backbone has no 'blocks' attribute; cannot hook a mid block.")
                idx = int(self._block_index)  # type: ignore[arg-type]
                n = len(blocks)
                if idx < 0 or idx >= n:
                    raise IndexError(f"block_index {idx} out of range for {n} blocks.")
                module = blocks[idx]
            self._handle = module.register_forward_hook(self._hook)
        return self

    def clear(self) -> None:
        self._latest = None

    def close(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._latest = None

    def latest(self, *, clone: bool = True) -> ViTFeatureSnapshot:
        if self._latest is None:
            raise RuntimeError("No ViT features captured yet. Run a forward pass after attaching the hook.")
        if not clone:
            return self._latest
        return ViTFeatureSnapshot(
            token_features=self._latest.token_features.clone(),
            spatial_features=self._latest.spatial_features.clone(),
            patch_shape=self._latest.patch_shape,
        )

    def latest_cpu(self) -> ViTFeatureSnapshot:
        return self.latest(clone=True).cpu()

    def __enter__(self) -> "ViTFeatureTap":
        return self.attach()

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()


__all__ = ["ViTFeatureSnapshot", "ViTFeatureTap"]
