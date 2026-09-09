"""One cap-preserving rest/bind map for both legs and all authored receivers."""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from .axial_caps_v14 import AxialCapsFieldV14, extract_proper_rotation_v14
from .consistent_runtime_v14 import driver_rotation_maps_v14
from .segment_similarity_rest_v10 import _segment_controller_sets, _descendants, _rotation_align


class LowerChainRestMapV17:
    """Twelve metre offsets: L knee, L ankle, R knee, R ankle.

    Hip sockets stay anchored to the pelvis. Only shaft axial length changes;
    cap domains and foot subtrees receive rigid transforms. Original weights
    transport every tissue, including vertices in long multi-controller tubes.
    """
    def __init__(self, base, calibration):
        if base.corrector is not None:
            raise ValueError('a changed rest map requires an offline pose-corrector refit')
        self.base = base
        self.asset = base.source_asset
        self.names = list(self.asset.source_bone_names)
        self.parents = np.asarray(base.parents, dtype=np.int64)
        self.rest = np.asarray(base.target_rest, dtype=np.float64)
        self.bind = np.asarray(base.target_bind, dtype=np.float64)
        self.domains = calibration.domains
        if len(self.names) != 235:
            raise ValueError('expected all 235 authored controllers')
        if calibration.source_operator_digest != base.source_pack.operator_runtime_digest:
            raise ValueError('calibration and source operator differ')
        groups = _segment_controller_sets(self.asset)
        self.segments = []
        self.terminal = []
        for side_index, (side, suffix) in enumerate((('left','L'), ('right','R'))):
            hip, knee, ankle = [self.names.index(n) for n in
                (f'Femur_Rot_{suffix}', f'Knee_Rotate_{suffix}', f'Ankle_Rot_{suffix}')]
            h, k, a = self.bind[[hip,knee,ankle], :3, 3]
            foot = np.asarray(sorted(_descendants(self.parents, ankle)), dtype=np.int64)
            definitions = [
                ('femur', h, k,
                 [f'{side}/femoral_head'],
                 [f'{side}/femoral_condyle_medial',f'{side}/femoral_condyle_lateral']),
                ('shank', k, a,
                 [f'{side}/tibial_plateau_medial',f'{side}/tibial_plateau_lateral'],
                 [f'ankle/{side}/tibia',f'ankle/{side}/fibula'])]
            for kind, p, q, proximal, distal in definitions:
                axis = (q-p)/np.linalg.norm(q-p)
                caps = []; cap_ids = []
                for keys in (proximal, distal):
                    ids = self._domain_ids(keys)
                    cap_ids.append(ids)
                    coordinates = (self.rest[ids]-p)@axis
                    caps.append((float(coordinates.min()),float(coordinates.max())))
                members = np.asarray(sorted(groups[f'{side}_{kind}']-set(foot.tolist())),dtype=np.int64)
                self.segments.append(dict(side_index=side_index, kind=kind, p=p.copy(),q=q.copy(),
                    caps=caps,cap_ids=cap_ids,controllers=members, hip=h.copy(), knee=k.copy(), ankle=a.copy()))
            self.terminal.append(foot)
        all_groups = [x['controllers'] for x in self.segments]+self.terminal
        combined = np.concatenate(all_groups)
        if len(combined) != len(np.unique(combined)):
            raise ValueError('overlapping lower controller groups')
        self.affected_controllers = np.sort(combined)
        self.controller_groups = tuple(all_groups)
        self._full_mass = [np.sum(base.weights*np.isin(base.indices,g),axis=1) for g in all_groups]

    def _domain_ids(self, bases):
        chunks = []
        for key in bases:
            candidates = [key+'.fit', key+'.validation']
            available = [self.domains[k] for k in candidates if k in self.domains]
            if not available and key in self.domains:
                available = [self.domains[key]]
            if not available:
                raise ValueError('missing anatomical cap: '+key)
            chunks.extend(available)
        ids = np.unique(np.concatenate(chunks)).astype(np.int64)
        if not len(ids) or ids.min()<0 or ids.max()>=len(self.rest):
            raise ValueError('invalid anatomical cap indices')
        return ids

    def _maps(self, parameters_m):
        values = np.asarray(parameters_m, dtype=np.float64)
        if values.shape != (12,) or not np.isfinite(values).all():
            raise ValueError('twelve finite station offsets in metres required')
        offsets = values.reshape(2,2,3)
        maps, ends, metadata = [], {}, []
        for segment in self.segments:
            side = segment['side_index']; p,q = segment['p'],segment['q']
            knee = segment['knee']+offsets[side,0]
            ankle = segment['ankle']+offsets[side,1]
            p1,q1 = (segment['hip'],knee) if segment['kind']=='femur' else (knee,ankle)
            length = float(np.linalg.norm(q-p)); target_length = float(np.linalg.norm(q1-p1))
            scale = target_length/length
            field = AxialCapsFieldV14(p,(q-p)/length,length,scale,*segment['caps'],
                                     proximal_cap_ids=segment['cap_ids'][0],distal_cap_ids=segment['cap_ids'][1])
            rotation = _rotation_align(q-p,q1-p1)
            translation = p1-rotation@p
            maps.append((field,rotation,translation))
            if segment['kind']=='shank':
                ends[side] = (None,rotation,translation+rotation@((q-p)/length*field.delta_m))
            metadata.append(dict(side=('left','right')[side],segment=segment['kind'],
                source_pivot_length_m=length,target_pivot_length_m=target_length,
                total_length_scale=scale,section_scale=1.0,
                minimum_axial_jacobian=float(field.analytic_jacobian_minimum),
                proximal_cap_ids=segment['cap_ids'][0].tolist(),
                distal_cap_ids=segment['cap_ids'][1].tolist(),
                cap_bounds_m=[list(x) for x in segment['caps']]))
        return maps+[ends[0],ends[1]], metadata

    @staticmethod
    def _map_points(points, spec):
        field, rotation, translation = spec
        mapped = points if field is None else field.map_points(points)
        return mapped@rotation.T+translation

    def sample(self, parameters_m, vertex_ids=None):
        ids = np.arange(len(self.rest)) if vertex_ids is None else np.asarray(vertex_ids,dtype=np.int64)
        if ids.ndim!=1 or np.any(ids<0) or np.any(ids>=len(self.rest)):
            raise ValueError('invalid vertex subset')
        specs, metadata = self._maps(parameters_m)
        points = self.rest[ids]
        moved = points.copy()
        bind = self.bind.copy()
        jacobians = np.tile(np.eye(3),(len(bind),1,1))
        for group, mass, spec in zip(self.controller_groups,self._full_mass,specs):
            selected = np.flatnonzero(mass[ids]>0)
            if len(selected):
                original = points[selected]
                moved[selected] += mass[ids[selected],None]*(self._map_points(original,spec)-original)
            field, rotation, _translation = spec
            origins = self.bind[group,:3,3]
            jac = np.broadcast_to(rotation,(len(group),3,3)).copy()
            if field is not None:
                jac = rotation[None]@field.jacobian(origins)
            jacobians[group] = jac
            bind[group,:3,3] = self._map_points(origins,spec)
            for index, j in zip(group,jac):
                bind[index,:3,:3] = extract_proper_rotation_v14(j)@self.bind[index,:3,:3]
        world_old = np.linalg.solve(self.bind[:,:3,:3].swapaxes(1,2),self.base.translation_maps)
        translations = bind[:,:3,:3].swapaxes(1,2)@jacobians@world_old
        rotations = driver_rotation_maps_v14(self.base.reference_bind,bind)
        # Exact identity preserves the source arrays instead of accumulating
        # matrix inversion roundoff on unchanged controllers or neutral maps.
        unaffected = np.ones(len(bind),dtype=bool); unaffected[self.affected_controllers]=False
        translations[unaffected] = self.base.translation_maps[unaffected]
        rotations[unaffected] = self.base.rotation_maps[unaffected]
        if not np.any(parameters_m):
            moved = points.copy(); bind=self.bind.copy()
            translations=self.base.translation_maps.copy(); rotations=self.base.rotation_maps.copy()
        return dict(vertices_rest=moved,target_bind=bind,translation_maps=translations,
                    rotation_maps=rotations,metadata=metadata)

    def compile(self, parameters_m):
        sample = self.sample(parameters_m)
        provenance = dict(self.base.provenance)
        provenance['lower_chain_rest_v17'] = dict(parameters_m=np.asarray(parameters_m).tolist(),
            segments=sample['metadata'],shared_original_weights=True,
            entire_ankle_subtrees=True,hip_sockets_anchored=True,anatomical_passed=False)
        return replace(self.base,target_rest=sample['vertices_rest'],target_bind=sample['target_bind'],
                       translation_maps=sample['translation_maps'],rotation_maps=sample['rotation_maps'],
                       provenance=provenance)
