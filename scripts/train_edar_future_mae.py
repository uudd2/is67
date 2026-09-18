"""Independent LIBERO Future-MAE Stage A; reads current DINO cache only."""
import argparse
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
import yaml

from src.datasets.libero_act import LiberoAct
from src.models.edar_lite import EDARFeatureCache
from src.models.edar_future_mae import EDARFutureMAE, prepare_rgb

SCHEMA = 'edar_future_mae_stage_a_v1'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FutureMAESamples(IterableDataset):
    """Thin wrapper: original episode/window alignment and raw RGB are unchanged."""
    def __init__(self, data, normalization=None, suite=None):
        super().__init__()
        self.suites = list(data['task_suite_names'])
        if not self.suites or len(set(self.suites)) != len(self.suites):
            raise ValueError('Specify distinct LIBERO task suites.')
        self.children = []
        for name in self.suites:
            path = Path(data['data_root']).expanduser() / name / '1.0.0'
            if not path.is_dir():
                raise FileNotFoundError(path)
            self.children.append(LiberoAct(
                data_path=str(path), dataset_name=name, history_len=1, future_len=8,
                full_sequence=True, input_modality='image', view_mode='single',
                load_future_image=True, future_image_mode='horizon',
                strict_future_horizon=True, frame_ids_only=False,
                normalization_mode='min_max', buffer_size=int(data.get('buffer_size', 128)),
            ))
        minimum = np.min([child.action_min for child in self.children], axis=0)
        maximum = np.max([child.action_max for child in self.children], axis=0)
        self.normalization = normalization or {
            'mode': 'min_max', 'scope': 'shared_union_of_suite_bounds',
            'task_suite_names': self.suites.copy(), 'action_min': minimum.tolist(), 'action_max': maximum.tolist(),
        }
        if self.normalization['mode'] != 'min_max':
            raise ValueError('This experiment uses shared min/max actions.')
        for child in self.children:
            child.action_min = np.asarray(self.normalization['action_min'])
            child.action_max = np.asarray(self.normalization['action_max'])
        self.namespace = bool(data.get('cache_namespace_by_suite', True))
        if suite:
            if suite not in self.suites:
                raise ValueError(f'Unknown suite: {suite}')
            self.children = [self.children[self.suites.index(suite)]]
            self.suites = [suite]

    def __iter__(self):
        rng = random.Random(torch.initial_seed())
        streams = [iter(child) for child in self.children]
        while True:
            order = list(range(len(streams)))
            rng.shuffle(order)
            for index in order:
                try:
                    sample = next(streams[index])
                except StopIteration:
                    streams[index] = iter(self.children[index])
                    try:
                        sample = next(streams[index])
                    except StopIteration as error:
                        raise RuntimeError(f'No valid samples for {self.suites[index]}') from error
                frame, future_frame = sample['frame_id'], sample['future_frame_id']
                current_episode, current_time = frame.rsplit('_frame_', 1)
                future_episode, future_time = future_frame.rsplit('_frame_', 1)
                if current_episode != future_episode or int(future_time)-int(current_time) != 8:
                    raise ValueError('Expected same-episode t -> t+8 alignment.')
                prefix = self.suites[index] + '/' if self.namespace else ''
                yield {'frame_id': prefix+frame, 'future_frame_id': prefix+future_frame,
                       'suite_name': self.suites[index], 'actions': sample['future_actions'][:8],
                       'current_rgb': sample['image'], 'future_rgb': sample['future_image']}


def load_current_cache(config):
    path = Path(config['data']['feature_cache']).expanduser()
    metadata = EDARFeatureCache.read_metadata(path)
    if int(metadata['hidden_size']) != int(config['model'].get('visual_dim', 1024)):
        raise ValueError('DINO cache dimension does not match model.')
    if metadata['output_grid'] != 8 or metadata['image_size'] != 256:
        raise ValueError('Expected the existing 256px, 8x8 current DINO cache.')
    if os.path.realpath(metadata['backbone']) != os.path.realpath(config['data']['dino_model']):
        raise ValueError('DINO cache backbone mismatch.')
    return EDARFeatureCache(path, metadata), metadata


def prepare_batch(batch, cache, device):
    # Never read future_frame_id from the feature cache: target is raw future RGB.
    current_visual = cache.get_many(batch['frame_id']).float().to(device)
    current_rgb = prepare_rgb(batch['current_rgb'].to(device))
    future_rgb = prepare_rgb(batch['future_rgb'].to(device))
    return batch['actions'].float().to(device), current_visual, current_rgb, future_rgb


def save_checkpoint(path, model, optimizer, scheduler, scaler, step, config, metadata, normalization):
    payload = {'schema': SCHEMA, 'step': step, 'config': config, 'dino_metadata': metadata,
               'action_normalization': normalization, 'model_state_dict': model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
               'scaler_state_dict': scaler.state_dict()}
    temporary = Path(str(path)+'.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--micro-batch-size', type=int)
    parser.add_argument('--accumulation', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--buffer-size', type=int)
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    train, data, project = config['train'], config['data'], config['project']
    for key, value in (('steps', args.steps), ('per_device_batch_size', args.micro_batch_size),
                       ('gradient_accumulation_steps', args.accumulation)):
        if value is not None:
            train[key] = value
    if args.micro_batch_size is not None or args.accumulation is not None:
        train['batch_size'] = train['per_device_batch_size'] * train['gradient_accumulation_steps']
    if args.workers is not None:
        data['num_workers'] = args.workers
    if args.buffer_size is not None:
        data['buffer_size'] = args.buffer_size
    micro, accumulation, steps = int(train['per_device_batch_size']), int(train['gradient_accumulation_steps']), int(train['steps'])
    if min(micro, accumulation, steps) < 1 or micro*accumulation != int(train['batch_size']):
        raise ValueError('Require positive steps and microbatch * accumulation == batch_size.')
    set_seed(int(train.get('seed', 2026)))
    device = torch.device(train.get('device', 'cuda'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable.')
    precision = train.get('precision', 'bf16')
    if precision not in ('bf16', 'fp16', 'fp32'):
        raise ValueError('precision must be bf16, fp16, or fp32')
    dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
    use_autocast = device.type == 'cuda' and precision != 'fp32'
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and precision == 'fp16')
    cache, metadata = load_current_cache(config)
    model = EDARFutureMAE(**config['model']).to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(train['learning_rate']),
                                 betas=tuple(train.get('optimizer_betas', [0.9, 0.99])),
                                 weight_decay=float(train.get('weight_decay', 0.01)))
    warmup = int(train.get('warmup_steps', 1000))
    floor = float(train.get('min_learning_rate', 1e-5))/float(train['learning_rate'])
    def schedule(step):
        if step < warmup:
            return (step+1)/max(1, warmup)
        fraction = min(1.0, max(0.0, (step-warmup)/max(1, steps-warmup)))
        return floor + (1-floor)*0.5*(1+math.cos(math.pi*fraction))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    start, normalization = 0, None
    if train.get('resume_path'):
        previous = torch.load(train['resume_path'], map_location='cpu', weights_only=False)
        if previous.get('schema') != SCHEMA or previous['config']['model'] != config['model']:
            raise ValueError('Resume must use a matching Future-MAE checkpoint.')
        model.load_state_dict(previous['model_state_dict'], strict=True)
        optimizer.load_state_dict(previous['optimizer_state_dict'])
        scheduler.load_state_dict(previous['scheduler_state_dict'])
        scaler.load_state_dict(previous['scaler_state_dict'])
        start, normalization = int(previous['step']), previous['action_normalization']
        del previous
    if start >= steps:
        raise ValueError('Checkpoint is already at or beyond requested steps.')
    dataset = FutureMAESamples(data, normalization)
    workers = int(data.get('num_workers', 1))
    loader = DataLoader(dataset, batch_size=micro, num_workers=workers,
                        pin_memory=device.type == 'cuda', persistent_workers=workers > 0,
                        **({'multiprocessing_context': 'spawn'} if workers else {}))
    output = Path(args.output_dir or str(Path(project['output_dir']) / project['name'])).expanduser()
    if not train.get('resume_path') and output.exists() and any(output.glob('*.pt')):
        raise FileExistsError(f'Output already contains checkpoints: {output}')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    print(f'Future-MAE: {start} -> {steps}; effective batch={micro*accumulation}; output={output}', flush=True)
    model.train()
    iterator = iter(loader)
    with (output / 'metrics.jsonl').open('a', buffering=1) as log:
        for step_index in range(start, steps):
            optimizer.zero_grad(set_to_none=True)
            sums = {}
            for _ in range(accumulation):
                actions, visual, current, future = prepare_batch(next(iterator), cache, device)
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_autocast):
                    outputs = model(actions, visual, current)
                    loss, metrics = model.representation_loss(outputs, actions, current, future, config.get('rgb_weight', 0.2))
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite Future-MAE loss')
                scaler.scale(loss / accumulation).backward()
                for key, value in metrics.items():
                    sums[key] = sums.get(key, 0.0) + value.item()/accumulation
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, float(train.get('max_grad_norm', 1.0)), error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step = step_index+1
            if step % int(project.get('log_interval', 20)) == 0 or step == steps:
                line = json.dumps({'step': step, **sums})
                print(line, flush=True)
                log.write(line+'\n')
            if step % int(project.get('save_interval', 5000)) == 0:
                save_checkpoint(output / f'checkpoint_{step}.pt', model, optimizer, scheduler, scaler,
                                step, config, metadata, dataset.normalization)
        save_checkpoint(output / 'checkpoint_final.pt', model, optimizer, scheduler, scaler,
                        steps, config, metadata, dataset.normalization)
    print('Future-MAE training complete.', flush=True)


if __name__ == '__main__':
    main()
