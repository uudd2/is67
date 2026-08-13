#!/usr/bin/env python3
import argparse
import csv
import gc
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.datasets.libero_act import LiberoAct
from src.datasets.av_aloha_multitask_act import AVAlohaMultitaskAct
from src.models.edar_lite import EDARFeatureCache, SingleViewEDARLite


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize EDAR-lite future DINO feature predictions."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--task", default="", help="Optional AV-ALOHA task name.")
    return parser.parse_args()


def build_model(checkpoint, device):
    config = checkpoint["config"]
    architecture = config["model"]["action_representation"]
    dino_metadata = checkpoint["dino_metadata"]
    model = SingleViewEDARLite(
        action_dim=int(architecture.get("action_dim", config["model"].get("action_dim", 7))),
        action_horizon=int(architecture.get("action_horizon", 8)),
        visual_dim=int(dino_metadata["hidden_size"]),
        visual_grid=int(architecture.get("visual_grid", 8)),
        model_dim=int(architecture.get("model_dim", 512)),
        latent_tokens=int(architecture.get("latent_tokens", 4)),
        latent_token_dim=int(architecture.get("latent_token_dim", 256)),
        num_layers=int(architecture.get("layers", 4)),
        num_heads=int(architecture.get("heads", 8)),
        mlp_ratio=float(architecture.get("mlp_ratio", 4.0)),
    )
    model.load_state_dict(checkpoint["edar_state_dict"], strict=True)
    return model.to(device).eval(), config


def shared_pca_maps(current, target, prediction, grid_size):
    features = torch.cat([current, target, prediction], dim=0).float()
    features = F.normalize(features, dim=-1)
    centered = features - features.mean(dim=0, keepdim=True)
    _, _, basis = torch.pca_lowrank(centered, q=3, center=False, niter=4)
    colors = centered @ basis
    low = torch.quantile(colors, 0.02, dim=0)
    high = torch.quantile(colors, 0.98, dim=0)
    colors = ((colors - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)
    maps = colors.reshape(3, grid_size, grid_size, 3).cpu().numpy()
    return maps[0], maps[1], maps[2]


def add_token_grid(axis, grid_size):
    axis.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.45, alpha=0.75)
    axis.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)


def show_rgb(axis, image, title):
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 1.0)
    axis.imshow(image)
    axis.set_title(title, fontsize=10)
    axis.axis("off")


def show_grid(axis, values, title, cmap=None, value_range=None):
    kwargs = {"interpolation": "nearest"}
    if cmap is not None:
        kwargs["cmap"] = cmap
    if value_range is not None:
        kwargs["vmin"], kwargs["vmax"] = value_range
    image = axis.imshow(values, **kwargs)
    axis.set_title(title, fontsize=10)
    add_token_grid(axis, values.shape[0])
    return image


def visualize_sample(model, cache, sample, sample_index, output_dir, device):
    actions = sample["future_actions"][: model.encoder.action_horizon].unsqueeze(0).to(device).float()
    current = cache.get(sample["frame_id"]).unsqueeze(0).to(device).float()
    target = cache.get(sample["future_frame_id"]).unsqueeze(0).to(device).float()

    with torch.inference_mode():
        outputs = model(actions, current)
    prediction = outputs["predicted_future_visual"].float()
    decoded_actions = outputs["decoded_actions"].float()

    current_norm = F.normalize(current, dim=-1)
    target_norm = F.normalize(target, dim=-1)
    prediction_norm = F.normalize(prediction, dim=-1)
    patch_cosine = (prediction_norm * target_norm).sum(dim=-1)[0]
    copy_cosine = (current_norm * target_norm).sum(dim=-1)[0]
    error_map = (1.0 - patch_cosine).clamp_min(0.0)
    motion_map = (1.0 - copy_cosine).clamp_min(0.0)
    grid_size = int(round(current.shape[1] ** 0.5))
    current_pca, target_pca, prediction_pca = shared_pca_maps(
        current[0], target[0], prediction[0], grid_size
    )

    pred_cosine = patch_cosine.mean().item()
    baseline_cosine = copy_cosine.mean().item()
    action_mse = F.mse_loss(decoded_actions, actions).item()
    feature_mse = F.mse_loss(prediction_norm, target_norm).item()
    improved_fraction = (patch_cosine > copy_cosine).float().mean().item()
    metrics = {
        "sample_index": sample_index,
        "frame_id": sample["frame_id"],
        "future_frame_id": sample["future_frame_id"],
        "prediction_cosine": pred_cosine,
        "copy_baseline_cosine": baseline_cosine,
        "cosine_gain": pred_cosine - baseline_cosine,
        "improved_patch_fraction": improved_fraction,
        "feature_mse": feature_mse,
        "action_mse": action_mse,
    }

    figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    show_rgb(axes[0, 0], sample["image"], "Current RGB (t)")
    show_rgb(
        axes[0, 1],
        sample["future_image"],
        f"Future RGB (horizon={model.encoder.action_horizon})",
    )
    show_grid(axes[0, 2], current_pca, "Current DINO PCA")
    show_grid(axes[0, 3], target_pca, "True future DINO PCA")
    show_grid(axes[1, 0], prediction_pca, "EDAR predicted DINO PCA")
    motion_image = show_grid(
        axes[1, 1],
        motion_map.reshape(grid_size, grid_size).cpu().numpy(),
        "True feature motion (1-cos)",
        cmap="magma",
        value_range=(0.0, max(float(motion_map.max().item()), 1e-6)),
    )
    error_image = show_grid(
        axes[1, 2],
        error_map.reshape(grid_size, grid_size).cpu().numpy(),
        "Prediction error (1-cos)",
        cmap="inferno",
        value_range=(0.0, max(float(error_map.max().item()), 1e-6)),
    )
    figure.colorbar(motion_image, ax=axes[1, 1], fraction=0.046)
    figure.colorbar(error_image, ax=axes[1, 2], fraction=0.046)

    action_axis = axes[1, 3]
    true_actions = actions[0].cpu().numpy()
    predicted_actions = decoded_actions[0].cpu().numpy()
    for dimension in range(true_actions.shape[-1]):
        action_axis.plot(
            true_actions[:, dimension],
            linewidth=1.3,
            label=f"gt a{dimension}" if dimension < 2 else None,
        )
        action_axis.plot(
            predicted_actions[:, dimension],
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            label=f"pred a{dimension}" if dimension < 2 else None,
        )
    action_axis.set_title(f"Action reconstruction (MSE={action_mse:.4f})", fontsize=10)
    action_axis.set_xlabel("Chunk step")
    action_axis.grid(alpha=0.25)
    action_axis.legend(fontsize=7, ncol=2)

    figure.suptitle(
        f"sample={sample_index} | pred cos={pred_cosine:.4f} | "
        f"copy cos={baseline_cosine:.4f} | gain={pred_cosine - baseline_cosine:+.4f} | "
        f"better patches={improved_fraction:.1%}",
        fontsize=12,
    )
    destination = output_dir / f"sample_{sample_index:06d}.png"
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return metrics


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint).expanduser()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "single_view_edar_lite_stage_a_v1":
        raise ValueError("Checkpoint is not an EDAR-lite Stage A checkpoint.")
    model, config = build_model(checkpoint, device)
    cache_dir = args.cache_dir or config["data"]["feature_cache"]
    cache_metadata = EDARFeatureCache.read_metadata(cache_dir)
    cache = EDARFeatureCache(cache_dir, cache_metadata)
    del checkpoint
    gc.collect()

    data_config = config["data"]
    horizon = model.encoder.action_horizon
    if data_config.get("dataset_name") == "av_aloha":
        tasks = data_config["av_aloha_tasks"]
        if args.task and args.task not in tasks:
                raise ValueError(f"Unknown AV-ALOHA task: {args.task}")
        dataset = AVAlohaMultitaskAct(
            data_root=os.path.expanduser(data_config["data_root"]),
            tasks=tasks,
            history_len=1,
            future_len=horizon,
            action_stride=int(data_config.get("action_stride", 1)),
            full_sequence=True,
            input_modality="image",
            view_mode="single",
            load_future_image=True,
            future_image_mode="horizon",
            buffer_size=1,
            main_camera=data_config.get(
                "main_camera", "observation.images.zed_cam_left"
            ),
            action_normalization=data_config.get("action_normalization", "min_max"),
            state_normalization=data_config.get("state_normalization", "identity"),
            balance_tasks=False,
        )
        # Preserve the six-task normalization statistics used in training, then
        # restrict iteration to the requested task.
        if args.task:
            dataset.task_info = [
                info for info in dataset.task_info if info["name"] == args.task
            ]
    else:
        dataset = LiberoAct(
            data_path=os.path.join(
                os.path.expanduser(data_config["data_root"]),
                data_config["task_suite_name"],
                "1.0.0",
            ),
            dataset_name=data_config["task_suite_name"],
            history_len=1,
            future_len=horizon,
            full_sequence=True,
            input_modality="image",
            view_mode="single",
            load_future_image=True,
            future_image_mode="horizon",
            strict_future_horizon=True,
            frame_ids_only=False,
            buffer_size=1,
            normalization_mode=data_config.get("normalization_mode", "min_max"),
            normalization_stats_path=data_config.get("normalization_stats_path"),
        )

    requested = {
        args.start_index + offset * args.stride
        for offset in range(args.num_samples)
    }
    rows = []
    for index, sample in enumerate(dataset):
        if index not in requested:
            continue
        rows.append(
            visualize_sample(model, cache, sample, index, output_dir, device)
        )
        print(f"Saved sample {index} ({len(rows)}/{len(requested)})")
        if len(rows) == len(requested):
            break
    if len(rows) != len(requested):
        raise RuntimeError(
            f"Dataset ended after producing {len(rows)}/{len(requested)} requested samples."
        )

    summary_path = output_dir / "summary.tsv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    numeric_keys = [
        "prediction_cosine",
        "copy_baseline_cosine",
        "cosine_gain",
        "improved_patch_fraction",
        "feature_mse",
        "action_mse",
    ]
    print(f"Saved visualization to {output_dir}")
    for key in numeric_keys:
        print(f"{key}: {np.mean([float(row[key]) for row in rows]):.6f}")


if __name__ == "__main__":
    main()
