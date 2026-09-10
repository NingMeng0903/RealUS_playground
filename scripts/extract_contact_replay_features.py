"""Read-only same-frame confidence extraction on the frozen 68-scan manifest.

v1 is reused from its frozen audit CSV, v2/v3 consume each original JPEG.
Outputs describe detection decisions without contact labels or accuracy claims.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from peirastic.contact_qp.features import FeatureConfig, FeatureExtractor, load_feature_config


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def classify(quality,valid,threshold):
    if not (valid[0] and valid[2]):return 'unknown_required_window'
    left,right=quality[0]<threshold,quality[2]<threshold
    return 'both_low' if left and right else 'left_low' if left else 'right_low' if right else 'both_above_threshold'


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temp,path)


def extract_scan(job):
    import cv2
    import h5py
    cv2.setNumThreads(1)
    relative=Path(job['scan']['file']);source=Path(job['source'])/relative
    csv_path=Path(job['audit'])/relative.with_suffix('')/'features.csv'
    destination=Path(job['output'])/'per_scan'/relative.with_suffix('.jsonl')
    meta_path=destination.with_suffix('.meta.json')
    signature=dict(config_hash=job['config_hash'],source_size=source.stat().st_size,
        source_mtime_ns=source.stat().st_mtime_ns,v1_csv_sha256=digest(csv_path),
        expected_frames=job['scan']['frames'])
    if meta_path.exists() and destination.exists():
        meta=json.loads(meta_path.read_text())
        if meta.get('signature')!=signature:raise ValueError(f'refuse stale cache overwrite: {relative}')
        if meta.get('output_sha256')!=digest(destination):raise ValueError(f'cache checksum mismatch: {relative}')
        return dict(file=str(relative),frames=meta['frames'],cached=True,path=str(destination))
    with csv_path.open(newline='') as stream:old=list(csv.DictReader(stream))
    expected=int(job['scan']['frames'])
    if len(old)!=expected or [int(row['frame']) for row in old]!=list(range(expected)):
        raise ValueError(f'v1 count/frame sequence mismatch: {relative}')
    if any(row['window_version']!='fecfe5c107ef36f7ce31' for row in old):
        raise ValueError('unexpected v1 window revision')
    extractors={name:FeatureExtractor(FeatureConfig(**job['configs'][name])) for name in ('v2','v3')}
    destination.parent.mkdir(parents=True,exist_ok=True);temporary=destination.with_name(destination.name+'.tmp')
    timings={name:[] for name in extractors};started=time.perf_counter()
    with h5py.File(source,'r') as handle, temporary.open('w') as output:
        if handle.attrs.get('schema')!='realus_icra_v1' or not bool(handle.attrs.get('complete',False)):
            raise ValueError(f'incomplete or non-raw H5: {relative}')
        images=handle['ultrasound/jpeg'];ut=np.asarray(handle['ultrasound/timestamp_ns'],dtype=float)*1e-9
        if len(images)!=expected or len(ut)!=expected:raise ValueError(f'raw frame count mismatch: {relative}')
        for index,previous in enumerate(old):
            jpeg=np.asarray(images[index],dtype=np.uint8)
            image=cv2.imdecode(jpeg,cv2.IMREAD_GRAYSCALE)
            if image is None:raise ValueError(f'JPEG decode failure: {relative}:{index}')
            metadata=json.loads(handle['ultrasound/metadata_json'][index]) if 'ultrasound/metadata_json' in handle else {}
            capture=1.+float(ut[index]-ut[0])
            profiles={'v1':dict(quality=[float(previous[k]) for k in ('left','center','right')],
                valid=[previous['valid'].lower()=='true']*3,window_version=previous['window_version'],
                registration_version=previous['registration_version'],calibration_version='unverified')}
            summaries={}
            for name,extractor in extractors.items():
                tick=time.perf_counter()
                obs,_=extractor.extract(image,frame_seq=index,source_id=str(relative),capture_time_s=capture,
                    received_time_s=capture,crop_box=metadata.get('crop_box'),hflip=metadata.get('hflip',False))
                timings[name].append(time.perf_counter()-tick)
                profiles[name]=dict(quality=obs.quality.tolist(),valid=obs.valid.tolist(),
                    window_version=obs.window_version,registration_version=obs.registration_version,
                    calibration_version=obs.calibration_version)
                features=extractor.last_features
                if features is not None:summaries[name]=dict(frame_status=features['frame_status'],
                    unknown_column_count=int(np.count_nonzero(features['unknown_columns'])),
                    low_confidence_column_count=int(np.count_nonzero(features['low_confidence_columns'])),
                    roi_mean=features['roi_mean'],paper_eq3_fullarea_mean=features['paper_eq3_fullarea_mean'])
            # Grey-value comparison only; not a mechanical/acoustic contact label.
            dark=[]
            for lo,hi in extractors['v3'].config.lateral_windows:
                band=image[:,int(lo*image.shape[1]):max(int(lo*image.shape[1])+1,int(hi*image.shape[1]))]
                dark.append(float(np.mean(band<8)))
            row=dict(source_file=str(relative),frame_id=f'{relative}:{index}',frame_index=index,
                jpeg_sha256=hashlib.sha256(jpeg.tobytes()).hexdigest(),
                force_n=float(previous['force_n']) if previous['force_n'] else None,
                historical_force_alignment_valid=previous['aligned_covered'].lower()=='true',
                historical_image_time_s=float(previous['image_time_s']),
                profiles=profiles,feature_summary=summaries,dark_fraction_lcr=dark,
                contact_ground_truth_available=False)
            output.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n')
    os.replace(temporary,destination)
    meta=dict(signature=signature,frames=expected,output_sha256=digest(destination),elapsed_s=time.perf_counter()-started,
        extraction_ms={name:dict(mean=float(np.mean(t))*1000,p95=float(np.quantile(t,.95))*1000) for name,t in timings.items()})
    atomic_json(meta_path,meta)
    return dict(file=str(relative),frames=expected,cached=False,path=str(destination),elapsed_s=meta['elapsed_s'])


def aggregate(output,scans,config_hash,expected):
    groups={name:dict(count=0,classes={},quality=[],valid=[]) for name in ('v1_t05','v2_t05','v3_t08','v2_t08')}
    by_scan={};total=0;target=output/'features.jsonl';temporary=target.with_name(target.name+'.tmp')
    with temporary.open('w') as combined:
        for scan in scans:
            path=output/'per_scan'/Path(scan['file']).with_suffix('.jsonl');count=0;scene={}
            with path.open() as stream:
                for line in stream:
                    row=json.loads(line);combined.write(line);count+=1;total+=1
                    for group,profile,threshold in (('v1_t05','v1',.5),('v2_t05','v2',.5),('v3_t08','v3',.8),('v2_t08','v2',.8)):
                        obs=row['profiles'][profile];label=classify(obs['quality'],obs['valid'],threshold)
                        stats=groups[group];stats['count']+=1;stats['classes'][label]=stats['classes'].get(label,0)+1
                        stats['quality'].append(obs['quality']);stats['valid'].append(obs['valid'])
                        entry=scene.setdefault(group,{})
                        entry[label]=entry.get(label,0)+1
            if count!=scan['frames']:raise ValueError('per-scan aggregate length mismatch')
            by_scan[scan['file']]=dict(frame_count=count,classification_counts=scene)
    if total!=expected:raise ValueError(f'expected {expected} frames, got {total}')
    os.replace(temporary,target)
    for stats in groups.values():
        quality=np.asarray(stats.pop('quality'));valid=np.asarray(stats.pop('valid'),dtype=bool)
        stats['quality_quantiles_lcr']={str(p):np.quantile(quality,p,axis=0).tolist() for p in (0,.05,.25,.5,.75,.95,1)}
        stats['unknown_window_counts_lcr']=(~valid).sum(axis=0).tolist()
        stats['classification_fraction']={key:value/stats['count'] for key,value in stats['classes'].items()}
    report=dict(schema=1,config_hash=config_hash,scans=len(scans),frames=total,features_sha256=digest(target),
        groups=groups,per_scan=by_scan,
        evidence_limits=['No annotated contact labels; classification proportions are not accuracy or repair success.',
            'v1/v2 thresholds 0.5 and v3 threshold 0.8 are deployment policies; v2_t08 isolates the threshold choice.',
            'Left/right windows are required. Center is recorded but does not determine class.',
            'Grey-value dark fraction is an analysis comparator, not contact ground truth.',
            'Archived force is audit-only; no new command execution or closed-loop counterfactual is inferred.'])
    atomic_json(output/'detection_summary.json',report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('/media/camp/EXT_DRIVE/ICRA_2027/icra 2027_contact/uncalibrated'))
    parser.add_argument('--audit',type=Path,default=Path('MD/contact_qp/human_audit'))
    parser.add_argument('--output',type=Path,default=Path('MD/contact_qp/replay_v5'))
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--max-scans',type=int,default=0,help='smoke only; summary explicitly marks subset')
    args=parser.parse_args(argv)
    audit=json.loads((args.audit/'summary.json').read_text())
    if audit['files']!=68 or audit['processed_frames']!=12588 or audit['window_version']!='fecfe5c107ef36f7ce31':
        raise ValueError('unexpected frozen baseline manifest')
    if sum(scan['frames'] for scan in audit['scans'])!=12588:raise ValueError('manifest frame total mismatch')
    configs=dict(v1=asdict(FeatureConfig(**audit['config'])),v2=asdict(FeatureConfig()),
        v3=asdict(load_feature_config('peirastic/config/contact_qp/welleweerd2020_features.json')))
    if FeatureConfig(**configs['v1']).window_version!=audit['window_version']:raise ValueError('v1 feature hash no longer reproducible')
    signature=dict(configs=configs,audit_summary_sha256=digest(args.audit/'summary.json'),
        feature_source_sha256=digest('peirastic/contact_qp/features.py'),extractor_source_sha256=digest(__file__))
    config_hash=canonical(signature);args.output.mkdir(parents=True,exist_ok=True)
    identity_path=args.output/'extraction_identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())['config_hash']!=config_hash:
        raise ValueError('refuse to overwrite output from a different extraction revision')
    atomic_json(identity_path,dict(config_hash=config_hash,**signature))
    atomic_json(args.output/'feature_configs.json',configs)
    scans=audit['scans'][:args.max_scans] if args.max_scans else audit['scans']
    started=time.perf_counter();completed=[]
    with ProcessPoolExecutor(max_workers=min(6,max(1,args.workers))) as pool:
        jobs=[pool.submit(extract_scan,dict(scan=scan,source=str(args.source),audit=str(args.audit),
            output=str(args.output),configs=configs,config_hash=config_hash)) for scan in scans]
        for future in as_completed(jobs):
            result=future.result();completed.append(result)
            print(json.dumps(dict(completed=len(completed),total_scans=len(scans),**result)),flush=True)
    report=aggregate(args.output,scans,config_hash,sum(scan['frames'] for scan in scans))
    report.update(elapsed_s=time.perf_counter()-started,complete_frozen_manifest=len(scans)==68,
                  worker_limit=min(6,max(1,args.workers)))
    atomic_json(args.output/'detection_summary.json',report)
    print(json.dumps(dict(frames=report['frames'],scans=report['scans'],elapsed_s=report['elapsed_s'])),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
