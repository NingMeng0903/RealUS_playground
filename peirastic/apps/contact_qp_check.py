"""Read-only confidence preflight. Never opens robot/controller command IPC."""
from __future__ import annotations

import argparse
import json
import time

from peirastic.contact_qp.features import FeatureConfig
from peirastic.contact_qp.runtime_config import load_study_config, validate_study_config
from peirastic.realman8dof.force.contact_observer import ConfidenceSubscriber


def observation_problem(observation, config, *, now_s):
    feature = config['feature']
    expected = (feature['registration_version'], feature['window_version'],
                FeatureConfig(**feature['config']).calibration_version)
    if observation is None:
        return 'no confidence messages; start the confidence worker and ultrasound publisher'
    if observation.version != expected:
        return f'version mismatch: expected={expected}, received={observation.version}; check source/crop/hflip and worker config'
    from peirastic.contact_qp.confidence_fusion import POLICIES, FEATURE_VERSION
    if (config.get('qp') or {}).get('allocation_policy')=='delay_kf_cop_v1':
        import numpy as np
        from peirastic.contact_qp.types import REGION_FEATURE_VERSION
        fc=FeatureConfig(**feature['config'])
        if (observation.region_feature_version!=REGION_FEATURE_VERSION
                or observation.region_layout_version!=fc.region_layout_version
                or not np.array_equal(observation.region_edges,fc.region_edges)
                or observation.timestamp_semantics!='effective_image_time'):
            return 'N-region confidence layout/version or effective image timestamp mismatch; restart the configured worker'
    if ((config.get('qp') or {}).get('allocation_policy') in POLICIES
            and observation.confidence_feature_version!=FEATURE_VERSION):
        return 'confidence centroid worker version mismatch; restart worker with the experimental config'
    max_age = float((config.get('qp') or {}).get('max_image_age_s', .30))
    # Preflight runs before teaching/contact. Unknown or low-quality free-air
    # frames still prove transport/configuration; they do not authorize repair.
    if not (observation.received_time_s<=now_s and 0<=now_s-observation.effective_time_s<=max_age):
        return f'confidence stale/future: age={now_s-observation.effective_time_s:.3f}s'
    return None


def wait_for_features(config, *, timeout_s=8., frames=3, subscriber=None):
    owned = subscriber is None
    subscriber = subscriber or ConfidenceSubscriber(config['feature_endpoint'])
    deadline = time.monotonic() + timeout_s
    last_identity = None
    source = None
    good = 0
    problem = 'no confidence messages'
    try:
        while time.monotonic() < deadline:
            observation = subscriber.snapshot()
            now = time.monotonic()
            problem = observation_problem(observation, config, now_s=now)
            if problem is None:
                identity = (observation.source_id, observation.version, observation.frame_seq)
                if source != observation.source_id:
                    source = observation.source_id
                    good = 0
                if identity != last_identity:
                    good += 1
                    last_identity = identity
                if good >= frames:
                    return dict(status='ready', distinct_frames=good,
                        feature_endpoint=config['feature_endpoint'],
                        quality=observation.quality.tolist(), valid=observation.valid.tolist(),
                        confidence_lr=None if observation.confidence_lr is None else observation.confidence_lr.tolist(),
                        weakside_feature_version=observation.weakside_feature_version,
                        region_confidence=None if observation.region_confidence is None else observation.region_confidence.tolist(),
                        region_valid=None if observation.region_valid is None else observation.region_valid.tolist(),
                        region_edges=None if observation.region_edges is None else observation.region_edges.tolist(),
                        region_layout_version=observation.region_layout_version,
                        timestamp_semantics=observation.timestamp_semantics,
                        age_s=now-observation.effective_time_s,
                        registration_version=observation.registration_version,
                        window_version=observation.window_version,
                        source_id=observation.source_id)
            else:
                good = 0
            time.sleep(.01)
        raise RuntimeError(f'confidence preflight failed: {problem or "not enough distinct fresh frames"}; '
                           f'receiver={subscriber.last_error or "no transport error"}')
    finally:
        if owned:
            subscriber.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--timeout-s', type=float, default=8.)
    parser.add_argument('--frames', type=int, default=3)
    args = parser.parse_args(argv)
    if not 0 < args.timeout_s <= 60 or args.frames < 1:
        parser.error('timeout must be in (0,60] seconds and frames must be positive')
    try:
        config = load_study_config(args.config)
        validate_study_config(config)
        result = wait_for_features(config, timeout_s=args.timeout_s, frames=args.frames)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'[CONFIDENCE FAILED] {exc}', flush=True)
        return 1
    print('[CONFIDENCE READY] ' + json.dumps(result), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
