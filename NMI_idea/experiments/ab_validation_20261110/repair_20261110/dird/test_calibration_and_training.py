"""Real-checkpoint regression for raw-vs-centered calibrated threshold."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from common import HERE, BASELINE, SOURCE, CapabilityBridge
from ird_playground.neural.train_signed import load_signed_train_config


def test_optional_initialization_seed_keeps_original_split_seed(tmp_path):
    cfgpath = HERE / 'candidate/train.yaml'
    cfg = load_signed_train_config(cfgpath, root=SOURCE)
    assert cfg.seed == 20261128 and cfg.initialization_seed == 20261110
    raw = yaml.safe_load(cfgpath.read_text())
    raw['training'].pop('initialization_seed')
    p = tmp_path / 'legacy.yaml'
    p.write_text(yaml.safe_dump(raw))
    legacy = load_signed_train_config(p, root=SOURCE)
    assert legacy.seed == cfg.seed and legacy.initialization_seed is None


def test_real_field_raw_threshold_uses_zero_plus_centered_threshold():
    b = CapabilityBridge()
    nominal = json.loads((BASELINE / 'dird/nominal_context.json').read_text())
    T = np.asarray(nominal['T_tcp_railbase'])
    q = np.asarray(nominal['q_seed'])
    conf = b.calibration
    assert conf['m_safe'] == pytest.approx(conf['zero_bias'] + conf['safety_threshold'])
    # Actual neural field, not a substituted capability oracle. Find a score
    # strictly between the old incorrect and corrected raw thresholds.
    target = .5 * (conf['m_safe'] + conf['safety_threshold'])
    low, high = 0., 1.
    def query(dx):
        x = T.copy()
        x[0, 3] += dx
        return b.query_path(x[None], np.asarray([q[0]]), q, exact=False)
    assert query(low)['min_raw_score'] > target > query(high)['min_raw_score']
    for _ in range(40):
        mid = .5 * (low + high)
        if query(mid)['min_raw_score'] > target:
            low = mid
        else:
            high = mid
    out = query(.5 * (low + high))
    raw = out['min_raw_score']
    assert conf['m_safe'] < raw < conf['safety_threshold']
    assert out['proxy_threshold'] == pytest.approx(conf['m_safe'])
    assert out['proxy_pass'] is True
    assert (raw - conf['zero_bias'] >= conf['safety_threshold']) is True
    assert out['score_is_safety_certificate'] is False


def test_actual_retraining_preserves_original_five_way_groups():
    old = json.loads((SOURCE / 'data/checkpoints/rm4d_signed_current_robot_20261108/train_report.json').read_text())
    new = json.loads((HERE / 'candidate/train_report.json').read_text())
    assert new['split_fingerprint'] == old['split_fingerprint']
    assert new['split_counts'] == old['split_counts']
    assert new['dataset_sha256'] == old['dataset_sha256']
