"""A fixed coupled hip/knee/ankle response, fitted offline on rule poses.

Unlike the earlier sum of independent axis responses, every kernel evaluates
all nine input angles together. No capture identifier, recorded motion, target
beta lookup table, nearest-surface query, or optimization is used at runtime.
This is a bounded engineering approximation, not an anatomical certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

SCALE_DEG = np.array([60., 45., 45., 60., 45., 45., 35., 35., 35.])
LEGACY_LOWER_DEG = np.array([-120., -60., -60., -15., -100., -100., -60., -60., -60.])
LEGACY_UPPER_DEG = np.array([30., 60., 60., 120., 100., 100., 60., 60., 60.])
LOWER_DEG = np.array([-120., -60., -60., -30., -100., -100., -60., -60., -60.])
UPPER_DEG = np.array([60., 60., 60., 120., 100., 100., 60., 60., 60.])


def _features(pose55):
    pose = np.asarray(pose55, dtype=np.float64)
    if pose.shape != (55, 3) or not np.isfinite(pose).all():
        raise ValueError('55 finite axis-angle rotations required')
    rotations = Rotation.from_rotvec(pose[[1, 4, 7, 2, 5, 8]]).as_rotvec()
    return np.rad2deg(rotations.reshape(2, 9))


def _source_supported(row):
    # These are the frozen source response balls, not angle clipping.
    return (np.linalg.norm(row[3:6]) <= 130. + 1e-8 and
            np.linalg.norm(row[6:9]) <= 75. + 1e-8)


def fixed_coupled_leg_samples_v17():
    """Deterministic protocol shared by every beta; no recorded poses."""
    rows = [np.zeros(9)]
    rejected = 0
    for hip in (-120., -90., -60., -30., 0., 30., 60.):
        for knee in (-30., -15., 0., 30., 60., 90., 120.):
            base = np.zeros(9); base[[0, 3]] = [hip, knee]
            candidates = [base]
            for axis in (1, 2, 4, 5, 6, 7, 8):
                knots = (-45., 45.) if axis < 6 else (-30., 30.)
                if axis in (4, 5):
                    knots = (-90., -45., 45., 90.)
                for angle in knots:
                    row = base.copy(); row[axis] = angle
                    candidates.append(row)
            for row in candidates:
                if _source_supported(row):
                    rows.append(row)
                else:
                    rejected += 1
    # Simultaneous off-axis rotations constrain interactions that single-axis
    # samples cannot identify. The seed and bounds are fixed for all subjects.
    lower = np.array([-105., -40., -40., -10., -45., -45., -35., -35., -35.])
    upper = np.array([25., 40., 40., 115., 45., 45., 35., 35., 35.])
    for row in qmc.scale(qmc.Sobol(9, scramble=True, seed=1701).random_base2(8), lower, upper):
        if _source_supported(row):
            rows.append(row)
        else:
            rejected += 1
    unique = np.unique(np.asarray(rows), axis=0)
    # Neutral first makes partial checkpoint reports easy to inspect.
    neutral = np.flatnonzero(np.all(unique == 0., axis=1))[0]
    unique[[0, neutral]] = unique[[neutral, 0]]
    return unique, rejected


def _kernel(query, centers, inverse_width=1.):
    # A global Gaussian RBF has coefficients fixed at compile time. Evaluation
    # performs no neighbor rebinding and no per-pose matrix solve.
    squared = np.sum((query[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
    return np.exp(-0.5 * inverse_width ** 2 * squared)


@dataclass
class CoupledLegArticulationV17:
    centers: np.ndarray
    coefficients: np.ndarray
    affine: np.ndarray
    neutral_offset: np.ndarray
    head_centers_local: np.ndarray
    report: dict
    kernel_inverse_width: float = 1.

    def __post_init__(self):
        if not np.isfinite(self.kernel_inverse_width) or self.kernel_inverse_width <= 0:
            raise ValueError('invalid coupled articulation kernel width')
        shapes = {'coefficients': (len(self.centers), 12), 'affine': (10, 12),
                  'neutral_offset': (12,), 'head_centers_local': (2, 3)}
        for name in ('centers', *shapes):
            value = np.asarray(getattr(self, name), dtype=np.float64).copy()
            if not np.isfinite(value).all():
                raise ValueError('nonfinite coupled articulation ' + name)
            if name == 'centers':
                if value.ndim != 2 or value.shape[1] != 9 or len(value) < 10:
                    raise ValueError('invalid coupled articulation centers')
            elif value.shape != shapes[name]:
                raise ValueError('invalid coupled articulation ' + name)
            value.setflags(write=False); setattr(self, name, value)

    def evaluate(self, pose55):
        angles = _features(pose55)
        lower=np.asarray(self.report.get('lower_support_deg',LEGACY_LOWER_DEG))
        upper=np.asarray(self.report.get('upper_support_deg',LEGACY_UPPER_DEG))
        if (np.any(angles < lower - 1e-6) or np.any(angles > upper + 1e-6)
                or not all(_source_supported(row) for row in angles)):
            raise ValueError('pose outside coupled leg response support; angles were not clipped')
        features = angles / SCALE_DEG
        distance = np.linalg.norm(features[:, None, :] - self.centers[None, :, :], axis=2)
        if np.any(distance.min(axis=1) > 2.5):
            raise ValueError('pose too far from coupled leg calibration samples')
        values = (_kernel(features, self.centers, self.kernel_inverse_width) @ self.coefficients +
                  np.c_[np.ones(2), features] @ self.affine - self.neutral_offset)
        result = np.stack([values[0, :6].reshape(2, 3), values[1, 6:].reshape(2, 3)])
        if np.any(np.linalg.norm(result, axis=2) > np.deg2rad(25.)):
            raise ValueError('coupled leg correction exceeds 25 degree norm support')
        return result

    def save(self, path):
        np.savez_compressed(path, **{key: getattr(self, key) for key in
            ('centers', 'coefficients', 'affine', 'neutral_offset', 'head_centers_local')},
            kernel_inverse_width=np.asarray(self.kernel_inverse_width),
            metadata_json=np.asarray(json.dumps(dict(schema='CoupledLegArticulationV17',
                                                       report=self.report), allow_nan=False)))

    @classmethod
    def load(cls, path):
        fields = {'centers', 'coefficients', 'affine', 'neutral_offset', 'head_centers_local'}
        with np.load(Path(path), allow_pickle=False) as data:
            if set(data.files) not in (fields | {'metadata_json'},fields | {'metadata_json','kernel_inverse_width'}):
                raise ValueError('invalid coupled articulation archive fields')
            meta = json.loads(str(data['metadata_json']))
            if meta['schema'] != 'CoupledLegArticulationV17':
                raise ValueError('invalid coupled articulation schema')
            return cls(**{key: data[key].copy() for key in fields}, report=meta['report'],
                kernel_inverse_width=float(data['kernel_inverse_width']) if 'kernel_inverse_width' in data.files else 1.)


def interpolate_coupled_leg_values_v17(angles, values, centers_local, report):
    """Freeze numerically conditioned interpolation from offline fit values.

    Width 2 was selected from the shared 916-sample geometry: its KKT condition
    was about 3.7e4, versus 1.4e8 for the discarded width-1 prototype. The same
    constant is retained for the expanded protocol and every beta; it is never
    selected using recorded motion. Reconstruction error is reported per bake.
    """
    features = np.asarray(angles,dtype=np.float64) / SCALE_DEG
    values = np.asarray(values,dtype=np.float64)
    if values.shape != (len(features),12) or not np.isfinite(values).all():
        raise ValueError('invalid offline coupled fit values')
    width=2.
    kernel = _kernel(features, features, width)
    polynomial = np.c_[np.ones(len(features)), features]
    matrix = np.block([[kernel + np.eye(len(kernel)) * 1e-8, polynomial],
                       [polynomial.T, np.zeros((10, 10))]])
    solution = np.linalg.solve(matrix, np.r_[values, np.zeros((10, 12))])
    coefficients, affine = solution[:-10], solution[-10:]
    neutral_offset = _kernel(np.zeros((1, 9)), features, width)[0] @ coefficients + affine[0]
    reconstructed = kernel @ coefficients + polynomial @ affine - neutral_offset
    error = np.rad2deg(reconstructed - values)
    report = dict(report,kernel_inverse_width=width,kernel_regularization=1e-8,
        training_reconstruction_max_deg=float(np.max(np.abs(error))),
        training_reconstruction_p95_deg=float(np.percentile(np.abs(error),95)))
    return CoupledLegArticulationV17(features,coefficients,affine,neutral_offset,centers_local,report,width)


def bake_coupled_leg_articulation_v17(subject, calibration, model, progress=None,
                                     checkpoint_path=None,reuse_saved_fits=False):
    from .lower_chain_pose_fit_v17 import fit_leg_pose_v17, head_centers_local_v17
    from .smplx_body_surface_v7 import _smplx_joint_kinematics_v7, smplx_body_surface_v7
    prior=subject.leg_articulation
    if prior is not None and not reuse_saved_fits:
        raise ValueError('subject already has a baked leg response')
    centers_local = head_centers_local_v17(subject.runtime, calibration)
    cached={}
    if reuse_saved_fits:
        if not isinstance(prior,CoupledLegArticulationV17):
            raise ValueError('coupled response required to extend its fixed sampling protocol')
        if not np.allclose(prior.head_centers_local,centers_local,atol=1e-12,rtol=0):
            raise ValueError('saved articulation and current rest head centers disagree')
        for record in prior.report['fits']:
            fits=record['fits']
            if [row['side'] for row in fits]!=['L','R'] or not all(row.get('skin_geometry_constraint') for row in fits):
                raise ValueError('cached fits lack the fixed bone-skin constraint')
            key=tuple(np.round(prior.centers[record['sample']]*SCALE_DEG,8))
            cached[key]=fits
    angles, excluded = fixed_coupled_leg_samples_v17()
    values = np.zeros((len(angles), 12)); records = []; reused=0
    for index, row in enumerate(angles):
        pose = np.zeros((55, 3))
        pose[[1, 2]] = np.deg2rad(row[:3])
        pose[[4, 5]] = np.deg2rad(row[3:6])
        pose[[7, 8]] = np.deg2rad(row[6:])
        if np.any(row):
            fits=cached.get(tuple(np.round(row,8)))
            if fits is None:
                _, _, global_ = _smplx_joint_kinematics_v7(model, betas=subject.target_betas,
                                                        pose_axis_angle=pose)
                skin = smplx_body_surface_v7(model, betas=subject.target_betas, pose_axis_angle=pose)
                twists, fits = fit_leg_pose_v17(subject.runtime, pose, global_, centers_local,
                                                skin_geometry=skin)
            else:
                twists=np.deg2rad(np.asarray([fit['angles_deg'] for fit in fits])).reshape(2,2,3)
                reused+=1
            values[index] = twists.reshape(12)
            records.append(dict(sample=index, fits=fits))
        if (index + 1) % 25 == 0 or index + 1 == len(angles):
            if progress:
                progress(dict(coupled_samples_done=index + 1, total=len(angles)))
            if checkpoint_path is not None:
                np.savez_compressed(checkpoint_path, angles_deg=angles[:index + 1],
                                    corrections=values[:index + 1])
    report = dict(method='coupled_gaussian_rbf_leg_articulation_v17', runtime_fit=False,
        runtime_surface_queries=False, runtime_matrix_solve=False,
        recorded_poses_used_for_fit=False, beta_specific_rules=False,
        training_pose_count=len(angles), source_domain_excluded_count=excluded,
        reused_rule_pose_fits=reused,new_rule_pose_fits=len(angles)-1-reused,
        feature_order=['hip_x','hip_y','hip_z','knee_x','knee_y','knee_z','ankle_x','ankle_y','ankle_z'],
        lower_support_deg=LOWER_DEG.tolist(), upper_support_deg=UPPER_DEG.tolist(),
        source_knee_norm_limit_deg=130., source_ankle_norm_limit_deg=75.,
        fits=records, anatomical_passed=False)
    return interpolate_coupled_leg_values_v17(angles,values,centers_local,report)
