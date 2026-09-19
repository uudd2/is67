"""PNG-only diagnostics for independent Future-MAE checkpoints."""
import argparse
import gc
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.edar_future_mae import EDARFutureMAE
from scripts.train_edar_future_mae import SCHEMA, FutureMAESamples, load_current_cache, prepare_batch, set_seed


def save_panel(current, target, prediction, metrics, destination):
    images = [tensor[0].float().cpu().permute(1, 2, 0).numpy() for tensor in (current, target, prediction)]
    current_image, target_image, prediction_image = images
    gt_change = np.abs(target_image-current_image).mean(axis=-1)
    pred_change = np.abs(prediction_image-current_image).mean(axis=-1)
    error = np.abs(prediction_image-target_image).mean(axis=-1)
    shared_max = max(float(gt_change.max()), float(pred_change.max()), 1e-6)
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for axis, image, title in zip(axes[0], images, ('Current RGB', 'Future GT (t+8)', 'Future Pred (display clipped)')):
        axis.imshow(np.clip(image, 0, 1))
        axis.set_title(title)
        axis.axis('off')
    for axis, values, title, vmax in zip(axes[1], (gt_change, pred_change, error),
                                        ('GT Change', 'Pred Change', 'Abs Error'),
                                        (shared_max, shared_max, max(float(error.max()), 1e-6))):
        shown = axis.imshow(values, cmap='magma', vmin=0, vmax=vmax)
        axis.set_title(title+' (mean |RGB difference|)')
        axis.axis('off')
        figure.colorbar(shown, ax=axis, fraction=0.046)
    figure.suptitle(f"Action MSE: {metrics['action_mse']:.5f} | RGB MSE: {metrics['loss_rgb']:.5f}\n"
                   f"Copy MSE: {metrics['copy_mse']:.5f} | MAE Gain: {metrics['mae_gain']:+.5f} | PSNR: {metrics['psnr']:.2f} dB")
    figure.savefig(destination, dpi=140)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', '--output_dir', default='outputs/edar_future_mae_vis')
    parser.add_argument('--num-vis', '--num_vis', type=int, default=20)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--suite', default='')
    parser.add_argument('--cache-dir', default='')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    if args.num_vis < 1 or args.stride < 1 or args.start_index < 0:
        parser.error('Require positive num_vis/stride and nonnegative start-index.')
    set_seed(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint.get('schema') != SCHEMA:
        raise ValueError('Expected an independent Future-MAE checkpoint.')
    config = checkpoint['config']
    if args.cache_dir:
        config['data']['feature_cache'] = args.cache_dir
    config['data']['buffer_size'] = 1
    cache, metadata = load_current_cache(config)
    if metadata != checkpoint['dino_metadata']:
        raise ValueError('Current DINO cache metadata differs from checkpoint.')
    dataset = FutureMAESamples(config['data'], checkpoint['action_normalization'], args.suite)
    model = EDARFutureMAE(**config['model'])
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model = model.to(device).eval()
    del checkpoint
    gc.collect()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    requested = {args.start_index+i*args.stride for i in range(args.num_vis)}
    rows = []
    with torch.inference_mode():
        for index, batch in enumerate(DataLoader(dataset, batch_size=1, num_workers=0)):
            if index not in requested:
                continue
            actions, visual, current, future = prepare_batch(batch, cache, device)
            outputs = model(actions, visual, current)
            _, metrics = model.representation_loss(outputs, actions, current, future, config.get('rgb_weight', 0.2))
            values = {name: value.item() for name, value in metrics.items()}
            destination = output / f'sample_{len(rows):03d}.png'
            save_panel(current, future, outputs['pred_future_rgb'], values, destination)
            rows.append(values)
            print(f'Saved {destination} | {batch["frame_id"][0]}', flush=True)
            if len(rows) >= args.num_vis:
                break
    summary = {name: float(np.mean([row[name] for row in rows])) for name in rows[0]}
    print(json.dumps({'samples': len(rows), **summary}, indent=2), flush=True)


if __name__ == '__main__':
    main()
