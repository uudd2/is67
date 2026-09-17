"""Balanced online Stage-A mixture with one action normalization space."""
import os
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset

from src.datasets.libero_act import LiberoAct


class LiberoMixedStageA(IterableDataset):
    def __init__(self, data_config, action_horizon=8):
        super().__init__()
        self.suites = tuple(data_config['task_suite_names'])
        if len(self.suites) < 2 or len(set(self.suites)) != len(self.suites):
            raise ValueError('Expected at least two distinct LIBERO suites.')
        if data_config.get('feature_cache'):
            raise ValueError('Mixed Stage-A uses online DINO, not a feature cache.')
        if data_config.get('normalization_mode', 'min_max') != 'min_max':
            raise ValueError('Mixed Stage-A requires shared min/max normalization.')
        self.datasets = []
        for suite in self.suites:
            path = os.path.join(data_config['data_root'], suite, '1.0.0')
            if not os.path.isdir(path):
                raise FileNotFoundError(path)
            self.datasets.append(LiberoAct(
                data_path=path, dataset_name=suite, history_len=1,
                future_len=action_horizon, full_sequence=True,
                input_modality='image', view_mode='single',
                load_future_image=True, future_image_mode='horizon',
                strict_future_horizon=True, frame_ids_only=False,
                buffer_size=int(data_config.get('buffer_size', 256)),
                normalization_mode='min_max',
            ))
        # Union of existing suite bounds, not newly estimated dataset statistics.
        self.action_min = np.min([d.action_min for d in self.datasets], axis=0)
        self.action_max = np.max([d.action_max for d in self.datasets], axis=0)
        for dataset in self.datasets:
            dataset.action_min = self.action_min.copy()
            dataset.action_max = self.action_max.copy()

    def __iter__(self):
        rng = random.Random(torch.initial_seed())
        streams = [iter(dataset) for dataset in self.datasets]
        while True:
            order = list(range(len(streams)))
            rng.shuffle(order)
            for index in order:
                try:
                    sample = next(streams[index])
                except StopIteration:
                    streams[index] = iter(self.datasets[index])
                    try:
                        sample = next(streams[index])
                    except StopIteration as error:
                        raise RuntimeError(f'No valid samples in {self.suites[index]} worker shard') from error
                sample = dict(sample)
                sample['suite_name'] = self.suites[index]
                for key in ('frame_id', 'future_frame_id'):
                    sample[key] = f'{self.suites[index]}/{sample[key]}'
                yield sample
