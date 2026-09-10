"""Latest-only confidence worker. Does not attach to or command a robot.

python -m peirastic.apps.contact_qp_features --help
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import logging
import time

import numpy as np

from peirastic.contact_qp.features import FeatureConfig, FeatureExtractor, LatestObservation, load_feature_config

LOG = logging.getLogger("contact_qp_features")
OUTPUT_TOPIC = b"contact_qp_features_v1"


def process_parts(parts, extractor, received_s):
    if len(parts) != 3:
        raise ValueError("expected topic/metadata/JPEG multipart")
    metadata = json.loads(parts[1])
    if not isinstance(metadata,dict):raise ValueError('camera metadata must be an object')
    identities = [metadata.get(key) for key in ("source_id", "publisher_instance_id")]
    if any(not isinstance(value, str) or not value.strip() for value in identities):
        raise ValueError("source_id and publisher_instance_id are required")
    capture_ns = metadata.get("capture_monotonic_ns")
    index=metadata.get('frame_index')
    if type(index) is not int or index<0:raise ValueError('frame_index must be a nonnegative integer')
    domain=metadata.get("clock_domain")
    # The real publisher merges SharedClock envelope fields after constructing
    # camera metadata. That overwrites clock_domain, but never converts the
    # separately named raw capture_monotonic_ns field or applies its offset.
    camera_contract=(metadata.get('schema_version')==1 and metadata.get('encoding')=='jpeg'
                     and metadata.get('timestamp_source')=='host_frame_read_complete')
    shared_contract=False
    if domain=='realus_shared' and camera_contract and metadata.get('source_id')=='realus.us_framegrab':
        header=metadata.get('header')
        if not isinstance(header,dict):raise ValueError('shared header must be an object')
        stamp=header.get('stamp')
        if not isinstance(stamp,dict):raise ValueError('shared stamp must be an object')
        sec,nanosec=stamp.get('sec'),stamp.get('nanosec')
        index=metadata.get('frame_index')
        if (type(sec) is int and type(nanosec) is int and 0<=nanosec<1_000_000_000
                and type(index) is int and index>=0):
            shared_contract=sec*1_000_000_000+nanosec==metadata.get('timestamp_ns')
    capture_domain=metadata.get('capture_clock_domain')
    domain_valid=(domain=='host_monotonic' or
                  (domain=='realus_shared' and shared_contract and
                   isinstance(metadata.get('clock_id'),str) and bool(metadata['clock_id'])))
    if (isinstance(capture_ns,bool) or not isinstance(capture_ns,int) or capture_ns<=0
            or not domain_valid or capture_domain not in (None,'host_monotonic')):
        raise ValueError("monotonic capture timestamp missing; cannot align control history")
    import cv2
    if not parts[2]:raise ValueError('empty JPEG payload')
    try:
        im = cv2.imdecode(np.frombuffer(parts[2], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    except cv2.error as exc:
        raise ValueError('invalid JPEG payload') from exc
    if im is None:
        raise ValueError("invalid JPEG")
    source = ":".join(identities)
    obs, _ = extractor.extract(im, frame_seq=int(metadata["frame_index"]), source_id=source,
                                capture_time_s=float(capture_ns)*1e-9, received_time_s=received_s,
                                crop_box=metadata.get("crop_box"), hflip=metadata.get("hflip", False))
    return obs


def encode_feature_payload(observation, features, processing_s):
    """Bounded control snapshot; column arrays/regions belong in preview JSON."""
    payload=observation.to_dict()
    if features is not None:
        keys=('top_roi_rows','roi_mean','paper_eq3_fullarea_mean',
              'barycenter_row_col_1based','threshold','frame_status')
        summary={key:features[key] for key in keys if key in features}
        summary['low_confidence_column_count']=int(np.count_nonzero(features.get('low_confidence_columns',[])))
        summary['unknown_column_count']=int(np.count_nonzero(features.get('unknown_columns',[])))
        payload['confidence_features']=summary
    payload['processing_s']=float(processing_s)
    blob=json.dumps(payload,allow_nan=False,separators=(',',':')).encode()
    if len(blob)>8192:raise ValueError('feature snapshot exceeds receiver envelope')
    return blob


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="tcp://127.0.0.1:17359")
    parser.add_argument("--output", default="tcp://127.0.0.1:17361")
    parser.add_argument("--input-topic", default="amongus_camera_frame_v1")
    parser.add_argument("--feature-config", help="JSON/YAML bare FeatureConfig or controller feature.config")
    parser.add_argument("--delay-s", type=float, default=None)
    parser.add_argument("--calibration-version", default=None)
    parser.add_argument("--image-x-sign", type=int, choices=(-1, 1), default=None)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args(argv)
    import zmq
    logging.basicConfig(level=logging.INFO)
    config = load_feature_config(args.feature_config) if args.feature_config else FeatureConfig()
    overrides = {k: v for k, v in dict(effective_delay_s=args.delay_s,
                 calibration_version=args.calibration_version, image_x_sign=args.image_x_sign).items() if v is not None}
    extractor = FeatureExtractor(replace(config, **overrides))
    LOG.info("feature window_version=%s", extractor.config.window_version)
    latest = LatestObservation()
    context = zmq.Context()
    sub = context.socket(zmq.SUB); pub = context.socket(zmq.PUB)
    sub.setsockopt(zmq.RCVHWM, 2); sub.setsockopt(zmq.SUBSCRIBE, args.input_topic.encode())
    pub.setsockopt(zmq.SNDHWM, 2)
    sub.setsockopt(zmq.LINGER, 0); pub.setsockopt(zmq.LINGER, 0)
    sub.connect(args.input); pub.bind(args.output)
    count = 0
    try:
        while not args.max_frames or count < args.max_frames:
            if not sub.poll(100):
                continue
            parts = sub.recv_multipart()
            # CONFLATE is incompatible with multipart; explicitly drain queued frames.
            for _ in range(64):
                if not sub.poll(0):
                    break
                parts = sub.recv_multipart()
            received = time.monotonic()
            start = time.perf_counter()
            try:
                obs = process_parts(parts, extractor, received)
                if not latest.accept(obs):
                    continue
                blob=encode_feature_payload(obs,extractor.last_features,time.perf_counter()-start)
                pub.send_multipart([OUTPUT_TOPIC, blob], flags=zmq.NOBLOCK)
                count += 1
            except (ValueError, RuntimeError, KeyError, zmq.Again) as exc:
                LOG.warning("feature frame dropped: %s", exc)
    except KeyboardInterrupt:
        pass
    finally:
        sub.close(); pub.close(); context.term()


if __name__ == "__main__":
    main()
