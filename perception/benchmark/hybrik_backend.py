"""Official HybrIK-X SMPL-X layer driven by measured body/hand landmarks.

By default shape and twist come from a static RGB initialization. Benchmarks
can additionally supply cached official RGB twist or local rotation priors;
this hybrid estimator is distinct from the pretrained paper's full pipeline.
Previous-pose tracking directly refines pose/root without rerunning analytic
IK, and causally filters measured xyz using the source sample clock.
"""
from __future__ import annotations

import contextlib
import importlib
import io
import json
from pathlib import Path
import sys
import types
import numpy as np

from perception.benchmark.orientation_constraints import (
    ORIENTATION_GROUPS, build_orientation_targets, orientation_loss, orientation_policy,
)


ROOT = Path(__file__).resolve().parents[2]
BODY_MAP = {0: 8, 1: 12, 2: 9, 4: 13, 5: 10, 7: 14, 8: 11,
            10: 19, 11: 22, 12: 1, 15: 0, 16: 5, 17: 2,
            18: 6, 19: 3, 20: 7, 21: 4}
NOMINAL_FRAME_DT_S = 1.0 / 30.0


def validate_rgb_pose_hint(rotations, agreement):
    values = np.asarray(rotations, dtype=np.float32)
    weights = np.asarray(agreement, dtype=np.float32)
    if values.shape != (55, 3, 3) or weights.shape != (55,):
        raise ValueError("RGB pose requires rotations [55,3,3] and agreement [55]")
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError("RGB pose hints must be finite")
    if np.any((weights < 0) | (weights > 1)):
        raise ValueError("RGB pose agreement must be in [0,1]")
    if np.any(weights[[0, 22, 23, 24]] != 0):
        raise ValueError("RGB local pose hint must exclude root and fixed face")
    if (not np.allclose(values @ values.transpose(0, 2, 1), np.eye(3), atol=1e-3)
            or not np.allclose(np.linalg.det(values), 1, atol=1e-3)):
        raise ValueError("RGB pose hints must be proper SO(3) rotations")
    return values, weights


def temporal_dt(timestamp_ns: int | None, previous_timestamp_ns: int | None,
                *, nominal_dt_s: float = NOMINAL_FRAME_DT_S) -> tuple[float, float]:
    """Return ``(dt_seconds, nominal_dt / dt)`` for a consecutive sample.

    The fallback is one nominal 30 Hz interval when either timestamp is absent.
    A non-increasing pair is rejected instead of silently turning a timestamp
    reset into an extremely strong temporal prior.
    """
    nominal_dt_s = float(nominal_dt_s)
    if not np.isfinite(nominal_dt_s) or nominal_dt_s <= 0:
        raise ValueError("nominal_dt_s must be positive and finite")
    if timestamp_ns is None or previous_timestamp_ns is None:
        return nominal_dt_s, 1.0
    delta_s = (int(timestamp_ns) - int(previous_timestamp_ns)) * 1e-9
    if not np.isfinite(delta_s) or delta_s <= 0:
        raise ValueError("timestamp_ns must increase strictly between accepted poses")
    return float(delta_s), float(nominal_dt_s / delta_s)


def so3_chordal_loss(current, reference, *, scale: float = 1.0):
    """Compute a rotation-matrix chordal loss, optionally normalized by time.

    Matrix subtraction is invariant to the 6D parameterization used by the
    optimizer: equivalent 6D representations that decode to the same SO(3)
    matrix produce exactly the same value. ``scale`` is the ratio between the
    nominal frame interval and the measured interval, so a constant angular
    velocity has the same penalty at 30, 10, and 2 Hz.
    """
    shape = tuple(current.shape)
    if shape != tuple(reference.shape) or len(shape) < 3 or shape[-2:] != (3, 3):
        raise ValueError(f"Expected matching rotation matrices [...,3,3], got {shape}")
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Temporal rotation scale must be positive and finite")
    return ((current - reference) * scale).square().mean()


def load_official_layer():
    """Import the official IK layer without importing obsolete RGB backbones."""
    path = ROOT / "ref_code_library/HybrIK/hybrik/models/layers/smplx"
    rotations = ROOT / "ref_code_library/pytorch3d"
    if not (path / "body_models.py").is_file() or not (rotations / "pytorch3d/transforms/rotation_conversions.py").is_file():
        raise FileNotFoundError("Clone official HybrIK and pytorch3d under ref_code_library first")
    if str(rotations) not in sys.path:
        sys.path.insert(0, str(rotations))
    name = "_realus_hybrik_smplx"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    return importlib.import_module(name + ".body_models").SMPLXLayer


def point_mapping() -> dict[int, int]:
    """SMPL-X native/leaf joint -> Body25 + left/right hand21 landmark."""
    mapping = dict(BODY_MAP)
    for native_base, point_base, leaf_base in ((25, 25, 60), (40, 46, 65)):
        for finger, mcp in enumerate((5, 9, 17, 13, 1)):
            for segment in range(3):
                mapping[native_base + finger * 3 + segment] = point_base + mcp + segment
            mapping[leaf_base + finger] = point_base + mcp + 3
    mapping[55], mapping[56] = 19, 22  # toe leaves
    return mapping


def sparse_landmark_regressor(path: Path) -> np.ndarray:
    with path.open() as handle:
        shape = [int(x) for x in handle.readline().lstrip("# ").split()]
    values = np.loadtxt(path)
    regressor = np.zeros(shape, dtype=np.float32)
    regressor[values[:, 0].astype(int), values[:, 1].astype(int)] = values[:, 2]
    return regressor[:67]


def refinement_landmarks(regressor, vertices, native_joints, *, convention: str):
    """Match measured shoulder semantics without changing model geometry.

    DWPose's COCO shoulders are constrained against native FK shoulders.
    The EasyMocap vertex regressor's shoulder rows are different locations;
    using them for this correspondence biased our supine upper-arm fits.
    Other body/hand rows retain their existing regression convention.
    """
    import torch
    landmarks = torch.einsum("jv,bvc->bjc", regressor, vertices)
    if convention == "dwpose":
        landmarks = landmarks.clone()
        landmarks[:, [5, 2]] = native_joints[:, [16, 17]]
    elif convention != "easymocap":
        raise ValueError("landmark convention must be 'dwpose' or 'easymocap'")
    return landmarks


def refinement_confidence(points: np.ndarray, valid: np.ndarray, *, convention: str) -> np.ndarray:
    confidence = np.where(valid, points[:, 3], 0).clip(0, 1)
    if convention == "dwpose":
        # DWPose synthesizes Neck by averaging 2D shoulders. Its triangulation
        # is neither an independent observation nor the model's neck landmark.
        confidence[1] = 0
    elif convention != "easymocap":
        raise ValueError("landmark convention must be 'dwpose' or 'easymocap'")
    return confidence


def pelvis_from_observed_hips(points: np.ndarray, valid: np.ndarray,
                              initial_landmarks: np.ndarray, initial_pelvis: np.ndarray,
                              rotation_from_initial: np.ndarray) -> np.ndarray:
    """Recover a missing MidHip from two measured hips and fitted proportions.

    The midpoint is a geometric measurement, not a predicted missing joint.
    Reject one-sided observations and implausible hip spans. The optimizer
    still sees the original observations, without an invented MidHip target.
    """
    if not (valid[9] and valid[12]):
        raise ValueError("Pelvis unavailable: reliable MidHip or both triangulated hips required")
    initial_pair = np.asarray(initial_landmarks)[[9, 12], :3]
    observed_pair = np.asarray(points)[[9, 12], :3]
    expected = float(np.linalg.norm(initial_pair[0] - initial_pair[1]))
    observed = float(np.linalg.norm(observed_pair[0] - observed_pair[1]))
    if (not np.isfinite(expected + observed) or expected < 0.05 or
            abs(observed - expected) > max(0.07, 0.35 * expected)):
        raise ValueError("Pelvis unavailable: triangulated hip span disagrees with fitted body")
    offset = np.asarray(initial_pelvis) - initial_pair.mean(axis=0)
    return observed_pair.mean(axis=0) + np.asarray(rotation_from_initial) @ offset


class InitializedHybrIKX:
    def __init__(self, init_npz: Path, *, device: str = "cuda", gender: str = "male", min_confidence: float = 0.05,
                 refine_steps: int = 0, pose_initialization: str = "hybrik",
                 temporal_weight: float = 0.0, orientation_weight: float = 0.0,
                 rgb_pose_weight: float = 0.0, observation_geometry_filter: bool = False,
                 occlusion_rotation_weight: float = 0.0, rgb_root_fallback: bool = False,
                 stationary_tolerance_m: float = 0.002,
                 observation_min_cutoff_hz: float = 1.0, observation_speed_beta: float = 20.0,
                 landmark_convention: str | None = None,
                 body_landmark_weight: float = 2.0, hand_landmark_weight: float = 2.0):
        import torch
        from scipy.spatial.transform import Rotation
        from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import easymocap_fit_to_smplx55

        self.torch = torch
        self.device = torch.device(device)
        gender = str(gender).strip().lower()
        if gender not in {"male", "female", "neutral"}:
            raise ValueError(f"Unsupported SMPL-X gender: {gender!r}")
        self.gender = gender
        self.min_confidence = min_confidence
        self.observation_geometry_filter = bool(observation_geometry_filter)
        self.rgb_root_fallback = bool(rgb_root_fallback)
        self.occlusion_rotation_weight = float(occlusion_rotation_weight)
        if not np.isfinite(self.occlusion_rotation_weight) or self.occlusion_rotation_weight < 0:
            raise ValueError("occlusion_rotation_weight must be finite and nonnegative")
        self.refine_steps = int(refine_steps)
        if self.refine_steps < 0:
            raise ValueError("refine_steps must be nonnegative")
        self.pose_initialization = str(pose_initialization).strip().lower()
        if self.pose_initialization not in {"hybrik", "previous"}:
            raise ValueError("pose_initialization must be 'hybrik' or 'previous'")
        if self.pose_initialization == "previous" and self.refine_steps <= 0:
            raise ValueError("pose_initialization='previous' requires refine_steps > 0")
        self.landmark_convention = (
            "dwpose" if self.pose_initialization == "previous" else "easymocap"
        ) if landmark_convention is None else str(landmark_convention).strip().lower()
        if self.landmark_convention not in {"dwpose", "easymocap"}:
            raise ValueError("landmark_convention must be 'dwpose' or 'easymocap'")
        self.body_landmark_weight = float(body_landmark_weight)
        self.hand_landmark_weight = float(hand_landmark_weight)
        if any(not np.isfinite(w) or w <= 0 for w in (self.body_landmark_weight, self.hand_landmark_weight)):
            raise ValueError("body/hand landmark weights must be positive and finite")
        if self.occlusion_rotation_weight and self.pose_initialization != "previous":
            raise ValueError("occlusion_rotation_weight requires previous pose refinement")
        self.temporal_weight = float(temporal_weight)
        if not np.isfinite(self.temporal_weight) or self.temporal_weight < 0:
            raise ValueError("temporal_weight must be finite and nonnegative")
        self.orientation_weight = float(orientation_weight)
        if not np.isfinite(self.orientation_weight) or self.orientation_weight < 0:
            raise ValueError("orientation_weight must be finite and nonnegative")
        if self.orientation_weight and self.refine_steps <= 0:
            raise ValueError("orientation_weight requires refine_steps > 0")
        self.rgb_pose_weight = float(rgb_pose_weight)
        if not np.isfinite(self.rgb_pose_weight) or self.rgb_pose_weight < 0:
            raise ValueError("rgb_pose_weight must be finite and nonnegative")
        if self.rgb_pose_weight and self.refine_steps <= 0:
            raise ValueError("rgb_pose_weight requires refine_steps > 0")
        if self.rgb_root_fallback and (not self.rgb_pose_weight or self.pose_initialization != "previous"):
            raise ValueError("rgb_root_fallback requires RGB pose priors and previous pose refinement")
        self.stationary_tolerance_m = float(stationary_tolerance_m)
        if not np.isfinite(self.stationary_tolerance_m) or self.stationary_tolerance_m < 0:
            raise ValueError("stationary_tolerance_m must be finite and nonnegative")
        self.stationary_gate = None
        self._stationary_result = None
        self.observation_min_cutoff_hz = float(observation_min_cutoff_hz)
        self.observation_speed_beta = float(observation_speed_beta)
        if not np.isfinite(self.observation_min_cutoff_hz) or self.observation_min_cutoff_hz < 0:
            raise ValueError("observation_min_cutoff_hz must be finite and nonnegative")
        if not np.isfinite(self.observation_speed_beta) or self.observation_speed_beta < 0:
            raise ValueError("observation_speed_beta must be finite and nonnegative")
        self.observation_filter = None
        if self.observation_min_cutoff_hz and self.pose_initialization == "previous":
            from perception.benchmark.adaptive_observation import AdaptiveObservationFilter
            self.observation_filter = AdaptiveObservationFilter(
                min_cutoff_hz=self.observation_min_cutoff_hz,
                speed_beta=self.observation_speed_beta, min_confidence=self.min_confidence)
        if self.stationary_tolerance_m and self.pose_initialization == "previous":
            from perception.benchmark.stationary_observation import StationaryObservationGate
            self.stationary_gate = StationaryObservationGate(
                position_tolerance_m=self.stationary_tolerance_m,
                min_confidence=self.min_confidence,
            )
        asset = ROOT / f"ref_code_library/EasyMocap/data/smplx/smplx/SMPLX_{gender.upper()}.pkl"
        self.model = load_official_layer()(str(asset.resolve()), gender=gender, ext="pkl", num_betas=10,
                                           use_pca=False, flat_hand_mean=True).eval().to(self.device)
        # Official LBS/IK traverses Python lists with these immutable indices.
        # CUDA scalar indices synchronize the host at every tree edge. CPU
        # indices also support the CUDA tensor gathers used by that code.
        for name in ("parents", "extended_parents", "children_map"):
            setattr(self.model, name, getattr(self.model, name).cpu())
        with np.load(init_npz) as seed:
            if "gender" in seed and str(np.asarray(seed["gender"]).reshape(-1)[0]).lower() != gender:
                raise ValueError("Initialization gender does not match the requested SMPL-X model")
            pose = easymocap_fit_to_smplx55(seed["Rh"], seed["poses"], gender=gender)
            self.betas_np = np.array(seed["shapes"], dtype=np.float32).reshape(-1)
            if self.betas_np.size != 10 or not np.isfinite(self.betas_np).all():
                raise ValueError("HybrIK-X initialization requires ten finite fitted SMPL-X betas")
            if "pelvis_world_m" not in seed:
                raise ValueError("Initialization must contain measured-model pelvis_world_m")
            self.pelvis = np.array(seed["pelvis_world_m"], dtype=np.float32).reshape(3)
        self.betas = torch.as_tensor(self.betas_np[None], device=self.device)
        self.expression = torch.zeros((1, 10), device=self.device)
        self.pose = torch.as_tensor(Rotation.from_rotvec(pose).as_matrix()[None], dtype=torch.float32, device=self.device)
        self.fixed_face = self.pose[:, 22:25].clone()
        self.last_timestamp_ns: int | None = None
        if self.refine_steps:
            from perception.benchmark.fast_smplx import CachedSMPLXVertexForward
            self._refinement_vertices = CachedSMPLXVertexForward(self.model, self.betas, self.expression)
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            initialized = self.model.forward_simple(betas=self.betas, expression=self.expression, full_pose=self.pose)
            twist = self.model.forward_get_twist(betas=self.betas, expression=self.expression, full_pose=self.pose)
            self.phis = torch.cat((torch.cos(twist), torch.sin(twist)), dim=-1)
            self.prior = self.model.get_extended_joints(initialized.joints_55, initialized.vertices)[0].cpu().numpy()
            regressor = sparse_landmark_regressor(ROOT / "ref_code_library/EasyMocap/data/smplx/J_regressor_body25_smplx.txt")
            self.landmark_regressor = torch.as_tensor(regressor, device=self.device)
            self.initial_landmarks = regressor @ initialized.vertices[0].cpu().numpy()
            self.rest_landmarks = None
            if self.observation_geometry_filter:
                rest_pose = torch.eye(3, device=self.device).repeat(1, 55, 1, 1)
                rest = self.model.forward_simple(
                    betas=self.betas, expression=self.expression, full_pose=rest_pose)
                self.rest_landmarks = regressor @ rest.vertices[0].cpu().numpy()
            self.offsets = {native: self.prior[native] - self.initial_landmarks[point]
                            for native, point in point_mapping().items()}
            shaped = self.model.v_template + torch.einsum("bl,vcl->bvc", self.betas, self.model.shapedirs)
            self.canonical_pelvis = torch.einsum("jv,bvc->bjc", self.model.J_regressor, shaped)[0, 0].cpu().numpy()
        if not np.isfinite(self.phis.cpu().numpy()).all():
            raise ValueError("Initialization produced invalid twist angles")
        self.initial_torso = self.prior[[1, 2, 16, 17]].copy()
        self.initial_prior = self.prior.copy()
        self.initial_root_rotation = self.pose[0, 0].detach().cpu().numpy().copy()
        # Compile/load CUDA and optimizer kernels during static initialization,
        # before measuring tracking. This uses only the RGB-fitted seed model.
        import time
        warmup_started = time.perf_counter()
        saved_pose = self.pose.clone()
        saved_prior = self.prior.copy()
        saved_pelvis = self.pelvis.copy()
        saved_timestamp_ns = self.last_timestamp_ns
        saved_stationary = None if self.stationary_gate is None else self.stationary_gate.snapshot()
        saved_stationary_result = self._stationary_result
        saved_filter = None if self.observation_filter is None else self.observation_filter.snapshot()
        warmup_points = np.column_stack((self.initial_landmarks + self.pelvis, np.ones(67))).astype(np.float32)
        if self.landmark_convention == "dwpose":
            warmup_points[[5, 2], :3] = self.prior[[16, 17]] + self.pelvis
        try:
            self.step(warmup_points)
        finally:
            self.pose = saved_pose
            self.prior = saved_prior
            self.pelvis = saved_pelvis
            self.last_timestamp_ns = saved_timestamp_ns
            if self.stationary_gate is not None:
                self.stationary_gate.restore(saved_stationary)
            self._stationary_result = saved_stationary_result
            if self.observation_filter is not None:
                self.observation_filter.restore(saved_filter)
        self.warmup_elapsed_s = time.perf_counter() - warmup_started

    def _refine(self, points: np.ndarray, pelvis: np.ndarray, valid: np.ndarray,
                *, temporal_reference=None, temporal_scale: float = 1.0,
                orientation_targets=None, rgb_rotations=None, rgb_agreement=None,
                occlusion_weights=None, temporal_dt_s: float = NOMINAL_FRAME_DT_S,
                root_observation=None) -> np.ndarray:
        """Fit only pose/root to observed 67 landmarks; shape and face stay fixed.

        ``temporal_reference`` is an accepted previous pose.  The temporal
        regularizer is deliberately computed from decoded rotation matrices,
        rather than the optimizer's 6D coordinates, so it does not depend on
        the redundant 6D representation.
        """
        torch = self.torch
        from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix

        active = list(range(22)) + list(range(25, 55))
        initial = self.pose.detach().clone()
        # Inference tensors must be cloned outside inference_mode before autograd.
        parameters = matrix_to_rotation_6d(initial[:, active]).clone().requires_grad_(True)
        root = torch.as_tensor(pelvis[None], device=self.device).clone().requires_grad_(True)
        target = torch.as_tensor(np.nan_to_num(points[:, :3])[None], device=self.device)
        convention = getattr(self, "landmark_convention", "easymocap")
        confidence = refinement_confidence(points, valid, convention=convention)
        if self.orientation_weight and orientation_targets is None:
            orientation_targets = build_orientation_targets(
                target, torch.as_tensor(confidence[None], device=self.device)
            )
        weights = torch.as_tensor(confidence[None, :, None], device=self.device)
        # Preserve measured confidence and the existing body/hand balance.
        # Increasing all body weights amplified unreliable leg observations
        # in the long-sequence audit, so it is an explicit experimental option.
        weights[:, :25] *= getattr(self, "body_landmark_weight", 2.)
        weights[:, 25:] *= getattr(self, "hand_landmark_weight", 2.)
        optimizer = torch.optim.Adam([parameters, root], lr=0.025)
        seed = parameters.detach().clone()
        best_loss = torch.full((), float("inf"), device=self.device)
        best_parameters, best_root = seed, root.detach().clone()
        previous_leg_world = None
        if occlusion_weights is not None:
            from perception.benchmark.occlusion_rotation import leg_world_rotations, occlusion_rotation_loss
            previous_leg_world = leg_world_rotations(temporal_reference).detach()
        for _ in range(self.refine_steps):
            optimizer.zero_grad(set_to_none=True)
            matrices = initial.clone()
            matrices[:, active] = rotation_6d_to_matrix(parameters)
            if convention == "dwpose":
                vertices, native_joints = self._refinement_vertices(matrices, return_joints=True)
            else:
                vertices, native_joints = self._refinement_vertices(matrices), None
            landmarks = refinement_landmarks(
                self.landmark_regressor, vertices, native_joints, convention=convention) + root[:, None]
            residual = landmarks - target
            robust = torch.nn.functional.huber_loss(residual, torch.zeros_like(residual), delta=0.03, reduction="none")
            loss = (robust * weights).sum() / weights.sum().clamp_min(1)
            if root_observation is not None:
                # A separately triangulated native pelvis is a real RGB
                # observation, used only when DWPose's pelvis/hip pair is absent.
                root_error = torch.nn.functional.huber_loss(
                    root, root_observation, delta=0.03, reduction="sum")
                loss = loss + 4.0 * root_error / weights.sum().clamp_min(1)
            loss = loss + 1e-5 * (parameters - seed).square().mean()
            if self.orientation_weight:
                loss = loss + self.orientation_weight * orientation_loss(landmarks, orientation_targets)
            if self.rgb_pose_weight and rgb_rotations is not None:
                # Local rotations share the SMPL-X rest frame across views.
                # Root and face are excluded by the hint validator. Agreement
                # is a cross-view consistency weight, not model confidence.
                error = (matrices - rgb_rotations).square().mean(dim=(-1, -2))
                loss = loss + self.rgb_pose_weight * (
                    (error * rgb_agreement).sum() / (rgb_agreement > 0).sum().clamp_min(1)
                )
            if self.temporal_weight and temporal_reference is not None:
                loss = loss + self.temporal_weight * so3_chordal_loss(
                    matrices[:, active], temporal_reference[:, active], scale=temporal_scale
                )
            if previous_leg_world is not None:
                loss = loss + self.occlusion_rotation_weight * occlusion_rotation_loss(
                    matrices, previous_leg_world, occlusion_weights, dt_s=temporal_dt_s)
            # Preserve strict best-loss selection of the pre-update state,
            # without a CUDA scalar download at every optimizer iteration.
            value = loss.detach()
            improved = value < best_loss
            best_loss = torch.where(improved, value, best_loss)
            best_parameters = torch.where(improved, parameters.detach(), best_parameters)
            best_root = torch.where(improved, root.detach(), best_root)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            self.pose = initial.clone()
            self.pose[:, active] = rotation_6d_to_matrix(best_parameters)
        return best_root[0].cpu().numpy()

    def _step_impl(self, points: np.ndarray, *, timestamp_ns: int | None = None,
                   rgb_phis=None, rgb_rotations=None, rgb_agreement=None,
                   rgb_pelvis_world_m=None) -> dict:
        import time
        from scipy.spatial.transform import Rotation

        step_started = time.perf_counter()
        points = np.asarray(points, dtype=np.float32)
        if points.shape != (67, 4):
            raise ValueError(f"Expected Body25 + hand21 + hand21 (67,4), got {points.shape}")
        geometry_metadata = None
        geometry_elapsed_s = 0.0
        if getattr(self, "observation_geometry_filter", False):
            from perception.benchmark.observation_geometry import filter_hand_observations
            geometry_started = time.perf_counter()
            points, geometry_metadata = filter_hand_observations(
                points, self.rest_landmarks, min_confidence=self.min_confidence)
            geometry_elapsed_s = time.perf_counter() - geometry_started
        # Safety-check the raw observations before smoothing, so an outlier
        # cannot gain confidence by moving inside the valid hand geometry.
        raw_points = points.copy()
        temporal_dt_s, temporal_scale = temporal_dt(
            timestamp_ns, getattr(self, "last_timestamp_ns", None))
        observation_filter = getattr(self, "observation_filter", None)
        filter_elapsed_s = 0.0
        filter_residual_m = 0.0
        if observation_filter is not None:
            filter_started = time.perf_counter()
            points = observation_filter.apply(points, timestamp_ns)
            if getattr(self, "observation_geometry_filter", False):
                points, _ = filter_hand_observations(
                    points, self.rest_landmarks, min_confidence=self.min_confidence)
            measured = np.isfinite(points).all(axis=1) & (points[:, 3] > self.min_confidence)
            if np.any(measured):
                filter_residual_m = float(np.max(np.linalg.norm(
                    points[measured, :3] - raw_points[measured, :3], axis=1)))
            filter_elapsed_s = time.perf_counter() - filter_started
        valid = np.isfinite(points).all(axis=1) & (points[:, 3] > self.min_confidence)
        torch = self.torch
        rgb_pelvis = None
        if rgb_pelvis_world_m is not None:
            if not getattr(self, "rgb_root_fallback", False):
                raise ValueError("RGB pelvis supplied without enabling rgb_root_fallback")
            rgb_pelvis = np.asarray(rgb_pelvis_world_m, dtype=np.float32)
            if rgb_pelvis.shape != (3,) or not np.isfinite(rgb_pelvis).all():
                raise ValueError("RGB pelvis must be finite world xyz [3]")
        phis = self.phis
        if rgb_phis is not None:
            if self.pose_initialization != "hybrik":
                raise ValueError("RGB twist requires pose_initialization='hybrik'")
            values = np.asarray(rgb_phis, dtype=np.float32)
            if values.shape != (54, 2) or not np.isfinite(values).all():
                raise ValueError("RGB twist must be finite [54,2]")
            norm = np.linalg.norm(values, axis=-1, keepdims=True)
            if np.any(norm < 1e-8):
                raise ValueError("RGB twist must have nonzero norm")
            phis = torch.as_tensor((values / norm)[None], device=self.device)
        rotation_hint, agreement_hint = None, None
        if rgb_rotations is not None or rgb_agreement is not None:
            values, agreement = validate_rgb_pose_hint(rgb_rotations, rgb_agreement)
            if not self.rgb_pose_weight:
                raise ValueError("RGB local pose requires positive rgb_pose_weight")
            rotation_hint = torch.as_tensor(values[None], device=self.device)
            agreement_hint = torch.as_tensor(agreement[None], device=self.device)
        stationary_gate = getattr(self, "stationary_gate", None)
        # Compare against the last fitted observation, never the last held
        # sample: slow real motion must accumulate until it crosses the bound.
        # RGB local rotations also release the hold for axial motion which
        # leaves landmark positions almost unchanged. An unused RGB pelvis is
        # excluded so crop noise cannot force a needless root refit.
        stationary_hints = dict(
            rgb_rotations=None if rotation_hint is None else values,
            rgb_agreement=None if agreement_hint is None else agreement,
            rgb_pelvis_world_m=(rgb_pelvis if not valid[8] and not (valid[9] and valid[12]) else None),
        )
        if (stationary_gate is not None and self._stationary_result is not None
                and filter_residual_m <= getattr(self, "stationary_tolerance_m", .002)
                and stationary_gate.matches(raw_points, **stationary_hints)):
            result = {key: value.copy() if isinstance(value, np.ndarray) else value
                      for key, value in self._stationary_result.items()}
            self.last_timestamp_ns = None if timestamp_ns is None else int(timestamp_ns)
            result.update(
                backend_elapsed_s=time.perf_counter() - step_started,
                stationary_held=True, temporal_dt_s=np.float32(temporal_dt_s),
                temporal_scale=np.float32(temporal_scale),
                timestamp_ns=np.int64(-1 if timestamp_ns is None else timestamp_ns),
                observation_geometry_metadata=json.dumps(geometry_metadata),
                observation_filter_residual_m=filter_residual_m,
                observation_filter_elapsed_s=filter_elapsed_s,
            )
            return result
        previous_pose = self.pose.detach().clone()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        mapping = point_mapping()
        # Use the four observed torso anchors to move unlabelled spine/collar
        # pivots. Their proportions come only from the RGB initialization.
        torso = [native for native in (1, 2, 16, 17) if valid[mapping[native]]]
        torso_transform = None
        if len(torso) >= 3:
            A = self.initial_prior[torso]
            B = np.array([points[mapping[j], :3] + self.offsets[j] for j in torso])
            ac, bc = A.mean(0), B.mean(0)
            U, _, Vt = np.linalg.svd((A - ac).T @ (B - bc))
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1] *= -1
                R = Vt.T @ U.T
            torso_transform = (R, ac, bc)
        if valid[8]:
            pelvis = points[8, :3] + self.offsets[0]
            root_source = "triangulated_midhip"
        elif not (valid[9] and valid[12]) and rgb_pelvis is not None:
            pelvis = rgb_pelvis.copy()
            root_source = "triangulated_rgb_pelvis"
        else:
            R = (torso_transform[0] if torso_transform is not None else
                 self.pose[0, 0].detach().cpu().numpy() @ self.initial_root_rotation.T)
            pelvis = pelvis_from_observed_hips(points, valid, self.initial_landmarks,
                                               self.initial_prior[0], R)
            root_source = "triangulated_hip_pair"
        target = self.prior.copy() + pelvis
        if torso_transform is not None:
            R, ac, bc = torso_transform
            for joint in (3, 6, 9, 13, 14):
                target[joint] = R @ (self.initial_prior[joint] - ac) + bc
        for native, point in mapping.items():
            if valid[point]:
                target[native] = points[point, :3] + self.offsets[native]
        target[0] = pelvis
        if self.pose_initialization == "hybrik":
            with torch.inference_mode():
                solved = self.model.hybrik(betas=self.betas, expression=self.expression,
                                          pose_skeleton=torch.as_tensor((target - pelvis)[None], device=self.device),
                                          phis=phis, root_align=True)
                self.pose = solved.rot_mats.clone()
                self.pose[:, 22:25] = self.fixed_face
        else:
            # The accepted previous pose is the initialization.  In
            # particular, do not call analytical IK here: it would reapply the
            # static initialization phis and recreate the frame-to-frame roll.
            self.pose = previous_pose
        orientation_targets = None
        orientation_group_mask = np.zeros(len(ORIENTATION_GROUPS), dtype=bool)
        if self.orientation_weight:
            orientation_targets = build_orientation_targets(
                torch.as_tensor(points[None, :, :3], device=self.device),
                torch.as_tensor(np.where(valid, points[:, 3], 0).clip(0, 1)[None], device=self.device),
            )
            orientation_group_mask = (orientation_targets.weights[0] > 0).cpu().numpy()
        if self.refine_steps:
            occlusion_weights = None
            if getattr(self, "occlusion_rotation_weight", 0):
                from perception.benchmark.occlusion_rotation import leg_occlusion_weights
                leg_weights = leg_occlusion_weights(points, self.min_confidence)
                if np.any(leg_weights):
                    occlusion_weights = torch.as_tensor(leg_weights[None], device=self.device)
            pelvis = self._refine(
                points, pelvis, valid, temporal_reference=previous_pose,
                temporal_scale=temporal_scale, orientation_targets=orientation_targets,
                rgb_rotations=rotation_hint, rgb_agreement=agreement_hint,
                occlusion_weights=occlusion_weights, temporal_dt_s=temporal_dt_s,
                root_observation=(torch.as_tensor(rgb_pelvis[None], device=self.device)
                                  if root_source == "triangulated_rgb_pelvis" else None),
            )
        with torch.inference_mode():
            applied = self.model.forward_simple(betas=self.betas, expression=self.expression, full_pose=self.pose)
            self.prior = self.model.get_extended_joints(applied.joints_55, applied.vertices)[0].cpu().numpy()
            rotations = self.pose[0].cpu().numpy()
            vertices = applied.vertices[0].cpu().numpy() + pelvis
            joints = applied.joints_55[0].cpu().numpy() + pelvis
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started + geometry_elapsed_s + filter_elapsed_s
        if not np.isfinite(vertices).all() or not np.isfinite(rotations).all():
            raise ValueError("HybrIK-X produced nonfinite geometry")
        pose = Rotation.from_matrix(rotations).as_rotvec().astype(np.float32)
        self.pelvis = pelvis
        self.last_timestamp_ns = None if timestamp_ns is None else int(timestamp_ns)
        result = dict(poses=pose.reshape(1, 165), Rh=pose[0].reshape(1, 3),
                    Th=(pelvis - rotations[0] @ self.canonical_pelvis).reshape(1, 3),
                    shapes=self.betas_np.reshape(1, -1), expression=np.zeros((1, 10), np.float32),
                    pelvis_world_m=pelvis, vertices=vertices, faces=self.model.faces,
                    joints_world_m=joints, backend_elapsed_s=elapsed, observed_points=int(valid.sum()),
                    refine_steps=self.refine_steps, root_source=root_source,
                    rgb_root_used=(root_source == "triangulated_rgb_pelvis"),
                    pose_initialization=self.pose_initialization,
                    landmark_convention=getattr(self, "landmark_convention", "easymocap"),
                    body_landmark_weight=float(getattr(self, "body_landmark_weight", 2.)),
                    hand_landmark_weight=float(getattr(self, "hand_landmark_weight", 2.)),
                    temporal_weight=np.float32(self.temporal_weight),
                    rgb_pose_weight=np.float32(self.rgb_pose_weight),
                    observation_geometry_filter=bool(getattr(self, "observation_geometry_filter", False)),
                    observation_geometry_metadata=json.dumps(geometry_metadata),
                    occlusion_rotation_weight=float(getattr(self, "occlusion_rotation_weight", 0)),
                    leg_occlusion_weights=(leg_weights if getattr(self, "occlusion_rotation_weight", 0)
                                           else np.zeros(4, np.float32)),
                    rgb_pose_agreement=(np.zeros(55, np.float32) if agreement_hint is None
                                        else agreement_hint[0].cpu().numpy()),
                    rgb_twist_supplied=bool(rgb_phis is not None),
                    orientation_weight=np.float32(self.orientation_weight),
                    orientation_observed_group_mask=orientation_group_mask,
                    orientation_group_names=np.asarray([group.name for group in ORIENTATION_GROUPS]),
                    orientation_policy=json.dumps(orientation_policy(), sort_keys=True),
                    temporal_dt_s=np.float32(temporal_dt_s),
                    temporal_scale=np.float32(temporal_scale),
                    stationary_held=False,
                    stationary_tolerance_m=float(getattr(self, "stationary_tolerance_m", 0)),
                    observation_min_cutoff_hz=float(getattr(self, "observation_min_cutoff_hz", 0)),
                    observation_speed_beta=float(getattr(self, "observation_speed_beta", 0)),
                    observation_filter_residual_m=filter_residual_m,
                    observation_filter_elapsed_s=filter_elapsed_s,
                    timestamp_ns=np.int64(-1 if timestamp_ns is None else timestamp_ns))
        if stationary_gate is not None:
            if stationary_gate.commit(raw_points, **stationary_hints):
                self._stationary_result = {
                    key: value.copy() if isinstance(value, np.ndarray) else value
                    for key, value in result.items()
                }
            else:
                stationary_gate.restore(None)
                self._stationary_result = None
        return result

    def step(self, points: np.ndarray, *, timestamp_ns: int | None = None,
             rgb_phis=None, rgb_rotations=None, rgb_agreement=None,
             rgb_pelvis_world_m=None) -> dict:
        """Track one landmark sample transactionally.

        A failed model/refinement evaluation must not turn a rejected sample
        into the next frame's pose or temporal reference.  The snapshot is
        cheap (one 55x3x3 tensor and two small arrays) compared with the
        model evaluation and also covers failures after pose refinement.
        """
        saved_pose = self.pose.clone()
        saved_prior = self.prior.copy()
        saved_pelvis = self.pelvis.copy()
        saved_timestamp_ns = getattr(self, "last_timestamp_ns", None)
        stationary_gate = getattr(self, "stationary_gate", None)
        saved_stationary = None if stationary_gate is None else stationary_gate.snapshot()
        saved_stationary_result = getattr(self, "_stationary_result", None)
        observation_filter = getattr(self, "observation_filter", None)
        saved_filter = None if observation_filter is None else observation_filter.snapshot()
        try:
            hints = {key: value for key, value in dict(rgb_phis=rgb_phis,
                     rgb_rotations=rgb_rotations, rgb_agreement=rgb_agreement,
                     rgb_pelvis_world_m=rgb_pelvis_world_m).items()
                     if value is not None}
            return self._step_impl(points, timestamp_ns=timestamp_ns, **hints)
        except Exception:
            self.pose = saved_pose
            self.prior = saved_prior
            self.pelvis = saved_pelvis
            self.last_timestamp_ns = saved_timestamp_ns
            if stationary_gate is not None:
                stationary_gate.restore(saved_stationary)
            self._stationary_result = saved_stationary_result
            if observation_filter is not None:
                observation_filter.restore(saved_filter)
            raise
