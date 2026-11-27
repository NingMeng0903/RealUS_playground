"""Cached, vertex-only SMPL-X LBS for the online pose refinement path.

The module has no CUDA work at import time.  ``CachedSMPLXVertexForward``
expects the isolated official HybrIK SMPL-X package to have been loaded by
``load_official_layer()`` before construction, so it can reuse the official
``blend_shapes`` and ``vertices2joints`` implementations.
"""
from __future__ import annotations

from typing import Any


def _tree_levels(parents: Any) -> list[tuple[list[int], list[int]]]:
    """Build a topological level schedule from a 55-joint parent array."""
    parent_values = [int(value) for value in parents]
    if len(parent_values) != 55 or parent_values[0] != -1:
        raise ValueError(f"Expected a native SMPL-X 55-joint tree, got {len(parent_values)} joints")
    depth = [0] * len(parent_values)
    for child in range(1, len(parent_values)):
        parent = parent_values[child]
        if parent < 0 or parent >= child:
            raise ValueError(f"Parent index must precede child {child}: {parent}")
        depth[child] = depth[parent] + 1
    levels = []
    for level in range(1, max(depth) + 1):
        children = [index for index, value in enumerate(depth) if value == level]
        if children:
            levels.append((children, [parent_values[index] for index in children]))
    if sorted(index for children, _ in levels for index in children) != list(range(1, 55)):
        raise ValueError("The 55-joint tree did not produce one level entry per non-root joint")
    return levels


class CachedSMPLXVertexForward:
    """Evaluate fixed-shape SMPL-X vertices from a full 55-joint pose.

    ``model`` is an official HybrIK ``SMPLXLayer`` (or a test double with the
    same tensors), and ``betas``/``expression`` are frozen at construction.
    The output has shape ``[B, 10475, 3]`` for the official model.  The
    forward pass keeps the official LBS math and root alignment while omitting
    landmarks, face contour, joint selection, and unused output assembly.
    """

    def __init__(self, model: Any, betas: Any, expression: Any):
        import torch
        import torch.nn.functional as F

        try:
            from _realus_hybrik_smplx.lbs import blend_shapes, vertices2joints
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Call load_official_layer() before constructing CachedSMPLXVertexForward"
            ) from error

        self._torch = torch
        self._F = F
        self.device = betas.device
        self.dtype = betas.dtype
        self.model = model
        self.joint_count = int(model.parents.numel())
        if self.joint_count != 55:
            raise ValueError(f"Expected native SMPL-X 55-joint tensors, got {self.joint_count}")

        self.parents = model.parents.detach().to(device=self.device, dtype=torch.long)
        self._root_index = torch.zeros(1, dtype=torch.long, device=self.device)
        self._levels = []
        for children, parent_indices in _tree_levels(self.parents.detach().cpu().tolist()):
            self._levels.append(
                (
                    torch.tensor(children, dtype=torch.long, device=self.device),
                    torch.tensor(parent_indices, dtype=torch.long, device=self.device),
                )
            )

        # Shape/expression are intentionally frozen.  The detach/no_grad pair
        # prevents beta or expression gradients from accumulating in a loop.
        with torch.no_grad():
            shape_dirs = torch.cat([model.shapedirs, model.expr_dirs], dim=-1)
            shape_components = torch.cat([betas, expression], dim=-1)
            self.v_shaped = (
                model.v_template + blend_shapes(shape_components, shape_dirs)
            ).detach()
            self.rest_joints = vertices2joints(model.J_regressor, self.v_shaped).detach()
            self.relative_joints = torch.cat(
                [
                    self.rest_joints[:, :1],
                    self.rest_joints[:, 1:]
                    - self.rest_joints[:, self.parents[1:]],
                ],
                dim=1,
            ).detach()

        self.posedirs = model.posedirs.detach()
        self.lbs_weights = model.lbs_weights.detach()
        self._identity = torch.eye(3, dtype=self.dtype, device=self.device)

    def forward(
        self,
        full_pose: Any,
        *,
        root_align: bool = True,
        return_joints: bool = False,
    ) -> Any:
        """Return posed vertices (and optionally native FK joints).

        ``full_pose`` is shaped ``[B,55,3,3]``.  With ``return_joints=True``
        the return value is ``(vertices, joints)``, where ``joints`` are the
        native 55-joint positions produced by the same forward-kinematics
        transforms used for skinning.  Both outputs use the same root
        alignment convention.
        """
        torch = self._torch
        F = self._F
        if full_pose.ndim != 4 or tuple(full_pose.shape[1:]) != (55, 3, 3):
            raise ValueError(f"Expected full_pose [B,55,3,3], got {tuple(full_pose.shape)}")
        batch_size = int(full_pose.shape[0])

        pose_feature = (full_pose[:, 1:] - self._identity).reshape(batch_size, -1)
        pose_offsets = torch.matmul(pose_feature, self.posedirs).view(batch_size, -1, 3)
        v_posed = pose_offsets + self.v_shaped.expand(batch_size, -1, -1)

        rest_joints = self.rest_joints.expand(batch_size, -1, -1)
        relative_joints = self.relative_joints.expand(batch_size, -1, -1)
        rotation = full_pose.reshape(-1, 3, 3)
        translation = relative_joints.reshape(-1, 3, 1)
        transforms = torch.cat(
            [F.pad(rotation, [0, 0, 0, 1]), F.pad(translation, [0, 0, 0, 1], value=1)],
            dim=2,
        ).view(batch_size, 55, 4, 4)

        # The loop is over kinematic depth levels, not joints.  All parent and
        # child indices are resident tensors, so no CUDA scalar reaches Python.
        global_transforms = torch.zeros_like(transforms).index_copy(
            1, self._root_index, transforms.index_select(1, self._root_index)
        )
        for children, parents in self._levels:
            parent_global = global_transforms.index_select(1, parents)
            child_relative = transforms.index_select(1, children)
            child_global = torch.matmul(parent_global, child_relative)
            global_transforms = global_transforms.index_copy(1, children, child_global)

        posed_joints = global_transforms[:, :, :3, 3]
        joints_homogeneous = F.pad(rest_joints.unsqueeze(-1), [0, 0, 0, 1])
        relative_transforms = global_transforms - F.pad(
            torch.matmul(global_transforms, joints_homogeneous),
            [3, 0, 0, 0, 0, 0, 0, 0],
        )
        weights = self.lbs_weights.unsqueeze(0).expand(batch_size, -1, -1)
        skinning = torch.matmul(
            weights, relative_transforms.view(batch_size, 55, 16)
        ).view(batch_size, -1, 4, 4)
        homogeneous_vertices = torch.cat(
            [v_posed, torch.ones_like(v_posed[:, :, :1])], dim=2
        ).unsqueeze(-1)
        vertices = torch.matmul(skinning, homogeneous_vertices)[:, :, :3, 0]
        joints = posed_joints
        if root_align:
            # Keep both outputs differentiable and avoid mutating tensors that
            # may still be needed by autograd or by a caller's graph.
            root = posed_joints[:, :1]
            vertices = vertices - root
            joints = joints - root
        if return_joints:
            return vertices, joints
        return vertices

    __call__ = forward
