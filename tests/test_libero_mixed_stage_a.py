import itertools
from unittest.mock import patch

import numpy as np
import torch

from src.datasets.libero_mixed_stage_a import LiberoMixedStageA
from scripts.train_edar_lite import _features_for_batch


class FakeSuite:
    def __init__(self, dataset_name, **kwargs):
        assert kwargs['strict_future_horizon']
        assert kwargs['load_future_image']
        assert not kwargs['frame_ids_only']
        self.action_min = np.full(6, -int(dataset_name))
        self.action_max = np.full(6, int(dataset_name))

    def __iter__(self):
        yield {'frame_id': '0', 'future_frame_id': '8'}


def test_balanced_restart_shared_scale_and_namespaced_ids():
    with patch('src.datasets.libero_mixed_stage_a.LiberoAct', FakeSuite), patch('os.path.isdir', return_value=True):
        dataset = LiberoMixedStageA({'task_suite_names': ['1', '2', '3', '4'], 'data_root': '/unused'})
    for child in dataset.datasets:
        np.testing.assert_array_equal(child.action_min, np.full(6, -4))
        np.testing.assert_array_equal(child.action_max, np.full(6, 4))
    samples = list(itertools.islice(dataset, 24))
    for offset in range(0, 24, 4):
        assert {s['suite_name'] for s in samples[offset:offset+4]} == {'1', '2', '3', '4'}
    assert all(s['frame_id'] == s['suite_name'] + '/0' for s in samples)
    assert all(s['future_frame_id'] == s['suite_name'] + '/8' for s in samples)


def test_online_inference_features_support_backward():
    @torch.inference_mode()
    def extractor(images):
        return torch.ones(images.shape[0], 64, 8)
    current, future = _features_for_batch({'image': torch.zeros(2, 3, 8, 8), 'future_image': torch.zeros(2, 3, 8, 8)}, None, extractor, torch.device('cpu'))
    assert not current.is_inference()
    layer = torch.nn.Linear(8, 4)
    layer(current).square().mean().backward()
    assert layer.weight.grad is not None
    assert future.shape == (2, 64, 8)
