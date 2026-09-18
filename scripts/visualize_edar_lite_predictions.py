#!/usr/bin/env python3
import argparse
import csv
import gc
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.datasets.libero_act import LiberoAct
from src.datasets.av_aloha_multitask_act import AVAlohaMultitaskAct
from src.models.edar_lite import EDARFeatureCache, FrozenDINOFeatureExtractor, SingleViewEDARLite


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize EDAR-lite future DINO feature predictions."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", "--output_dir", default="outputs/stage_a_vis")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", "--num-vis", "--num_vis", type=int, default=20)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--task", default="", help="Optional AV-ALOHA task name.")
    parser.add_argument("--suite", default="", help="Optional LIBERO suite from the checkpoint.")
    parser.add_argument("--selection", choices=("sequential", "random"), default="sequential")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.num_samples < 1 or args.stride < 1 or args.start_index < 0:
        parser.error("num_vis/stride must be positive and start-index nonnegative")
    return args


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


def add_token_grid(axis, grid_size):
    axis.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.45, alpha=0.75)
    axis.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)


def show_rgb(axis, image, title):
    if image is None:
        axis.text(0.5, 0.5, "RGB unavailable", ha="center", va="center")
        axis.set_title(title)
        axis.axis("off")
        return
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = image.transpose(1, 2, 0)
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


@torch.inference_mode()
def evaluate_sample(model, cache, extractor, sample, sample_index, device):
    actions = sample["future_actions"][: model.encoder.action_horizon].unsqueeze(0).to(device).float()
    if cache is not None:
        current = cache.get(sample["frame_id"]).unsqueeze(0).to(device).float()
        target = cache.get(sample["future_frame_id"]).unsqueeze(0).to(device).float()
    else:
        images = torch.stack([torch.as_tensor(sample["image"]), torch.as_tensor(sample["future_image"])])
        current, target = extractor(images.to(device)).float().chunk(2)
    outputs = model(actions, current)
    prediction = outputs["predicted_future_visual"].float()
    decoded_actions = outputs["decoded_actions"].float()

    current_norm = F.normalize(current, dim=-1)
    target_norm = F.normalize(target, dim=-1)
    prediction_norm = F.normalize(prediction, dim=-1)
    patch_cosine = (prediction_norm * target_norm).sum(dim=-1)[0]
    copy_cosine = (current_norm * target_norm).sum(dim=-1)[0]
    error_map = 1.0 - patch_cosine
    motion_map = 1.0 - copy_cosine
    predicted_motion = 1.0 - (current_norm * prediction_norm).sum(dim=-1)[0]
    grid_size = int(round(current.shape[1] ** 0.5))
    if grid_size != 8 or current.shape[1] != 64:
        raise ValueError("Quick evaluation expects the existing 8x8 patch grid.")

    pred_cosine = patch_cosine.mean().item()
    baseline_cosine = copy_cosine.mean().item()
    action_mse = F.mse_loss(decoded_actions, actions).item()
    metrics = {
        "sample_index": sample_index,
        "frame_id": sample["frame_id"],
        "future_frame_id": sample["future_frame_id"],
        "pred_cosine": pred_cosine,
        "copy_cosine": baseline_cosine,
        "cosine_gain": pred_cosine - baseline_cosine,
        "action_mse": action_mse,
        "copy_error": motion_map.mean().item(),
        "shuffle_gap": float("nan"),
    }
    return {"metrics": metrics, "sample": sample, "actions": actions.cpu(),
            "decoded": decoded_actions.cpu(), "current": current.cpu(), "target": target.cpu(),
            "prediction": prediction.cpu(), "latent": outputs["action_latent"].cpu(),
            "gt_effect": motion_map.reshape(8, 8).cpu().numpy(),
            "pred_effect": predicted_motion.reshape(8, 8).cpu().numpy(),
            "error": error_map.reshape(8, 8).cpu().numpy()}


@torch.inference_mode()
def add_shuffle_metrics(model, records, device):
    if len(records) < 2:
        return  # A single sample cannot provide a non-identity batch shuffle.
    latents = torch.cat([record["latent"] for record in records]).to(device).roll(1, dims=0)
    for index, record in enumerate(records):
        current, target = record["current"].to(device), record["target"].to(device)
        _, shuffled = model.decoder(latents[index:index+1], current)
        shuffled_loss, _ = model.change_weighted_effect_loss(shuffled, current, target)
        correct_loss, _ = model.change_weighted_effect_loss(record["prediction"].to(device), current, target)
        record["metrics"]["shuffle_gap"] = (shuffled_loss - correct_loss).item()


def visualize_sample(record, number, output_dir):
    metrics, sample = record["metrics"], record["sample"]
    figure = plt.figure(figsize=(14, 8), constrained_layout=True)
    layout = figure.add_gridspec(2, 3)
    show_rgb(figure.add_subplot(layout[0, 0]), sample.get("image"), "Current RGB (t)")
    show_rgb(figure.add_subplot(layout[0, 1]), sample.get("future_image"), "Future RGB (t+horizon)")
    action_layout = layout[0, 2].subgridspec(2, 1, height_ratios=[2, 1])
    action_axis = figure.add_subplot(action_layout[0])
    gripper_axis = figure.add_subplot(action_layout[1])
    true_actions = record["actions"][0].numpy()
    predicted_actions = record["decoded"][0].numpy()
    grippers = [6] if true_actions.shape[-1] == 7 else ([6, 13] if true_actions.shape[-1] == 14 else [])
    for dimension in range(true_actions.shape[-1]):
        axis = gripper_axis if dimension in grippers else action_axis
        color = plt.get_cmap("tab20")(dimension % 20)
        axis.plot(true_actions[:, dimension], color=color, lw=1.4, label=f"a{dimension}")
        axis.plot(predicted_actions[:, dimension], color=color, ls="--", lw=1.2)
    action_axis.set_title("Action curves: GT solid / Pred dashed", fontsize=10)
    gripper_axis.set_title("Gripper" if grippers else "No gripper dimension specified", fontsize=9)
    for axis in (action_axis, gripper_axis):
        axis.set_xticks(range(true_actions.shape[0]))
        axis.set_ylabel("Normalized")
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend(fontsize=7, ncol=3)
    gripper_axis.set_xlabel("Step")
    shared_max = max(float(record["gt_effect"].max()), float(record["pred_effect"].max()), 1e-6)
    for column, key, title in ((0, "gt_effect", "GT Effect Map (1 - cos)"),
                               (1, "pred_effect", "Pred Effect Map (1 - cos)"),
                               (2, "error", "Feature Error (1 - cos)")):
        axis = figure.add_subplot(layout[1, column])
        vmax = shared_max if key != "error" else max(float(record[key].max()), 1e-6)
        heatmap = show_grid(axis, record[key], title, cmap="magma" if key != "error" else "inferno", value_range=(0, vmax))
        figure.colorbar(heatmap, ax=axis, fraction=0.046)
    gap = metrics["shuffle_gap"]
    gap_text = f"{gap:+.4f}" if np.isfinite(gap) else "N/A (one sample)"
    figure.suptitle(
        f"Sample {number:03d} | Dataset index {metrics['sample_index']}\n"
        f"Action MSE: {metrics['action_mse']:.4f} | Pred Cos: {metrics['pred_cosine']:.4f} | "
        f"Copy Cos: {metrics['copy_cosine']:.4f} | Cos Gain: {metrics['cosine_gain']:+.4f} | Shuffle Gap: {gap_text}",
        fontsize=11,
    )
    destination = output_dir / f"sample_{number:03d}.png"
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
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
    normalization = checkpoint.get("action_normalization", {})
    expected_metadata = checkpoint["dino_metadata"]
    cache_dir = args.cache_dir or config["data"].get("feature_cache", "")
    cache, extractor = None, None
    if cache_dir:
        cache_metadata = EDARFeatureCache.read_metadata(cache_dir)
        for key in ("hidden_size", "image_size", "output_grid", "patch_size", "image_mean", "image_std"):
            if cache_metadata.get(key) != expected_metadata.get(key):
                raise ValueError(f"Cache/checkpoint DINO metadata mismatch: {key}")
        if os.path.realpath(cache_metadata["backbone"]) != os.path.realpath(expected_metadata["backbone"]):
            raise ValueError("Cache/checkpoint DINO backbone mismatch")
        cache = EDARFeatureCache(cache_dir, cache_metadata)
    else:
        extractor = FrozenDINOFeatureExtractor(config["model"]["dino_model"]).to(device).eval()
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
        suites = data_config.get("task_suite_names") or [data_config["task_suite_name"]]
        if args.suite:
            if args.suite not in suites:
                raise ValueError(f"Suite not present in checkpoint: {args.suite}")
            suites = [args.suite]
        datasets = []
        for suite in suites:
            child = LiberoAct(
                data_path=os.path.join(os.path.expanduser(data_config["data_root"]), suite, "1.0.0"),
                dataset_name=suite, history_len=1, future_len=horizon,
                full_sequence=True, input_modality="image", view_mode="single",
                load_future_image=True, future_image_mode="horizon",
                strict_future_horizon=True, frame_ids_only=False, buffer_size=1,
                normalization_mode=data_config.get("normalization_mode", "min_max"),
                normalization_stats_path=data_config.get("normalization_stats_path"),
            )
            if normalization.get("mode") == "min_max":
                child.action_min = np.asarray(normalization["action_min"])
                child.action_max = np.asarray(normalization["action_max"])
            datasets.append(child)

        def samples():
            # Reuse existing datasets; rotate suites without changing their internals.
            streams = [(suite, iter(child)) for suite, child in zip(suites, datasets)]
            while streams:
                active = []
                for suite, stream in streams:
                    try:
                        sample = dict(next(stream))
                    except StopIteration:
                        continue
                    if data_config.get("task_suite_names"):
                        for key in ("frame_id", "future_frame_id"):
                            sample[key] = f"{suite}/{sample[key]}"
                    yield sample
                    active.append((suite, stream))
                streams = active
        dataset = samples()

    requested = {
        args.start_index + offset * args.stride
        for offset in range(args.num_samples)
    }
    if args.selection == "random":
        requested = set(random.Random(args.seed).sample(
            range(args.start_index, args.start_index + args.num_samples * args.stride),
            args.num_samples,
        ))
    records = []
    for index, sample in enumerate(dataset):
        if index not in requested:
            continue
        records.append(
            evaluate_sample(model, cache, extractor, sample, index, device)
        )
        print(f"Evaluated sample {index} ({len(records)}/{len(requested)})", flush=True)
        if len(records) == len(requested):
            break
    if len(records) != len(requested):
        raise RuntimeError(
            f"Dataset ended after producing {len(records)}/{len(requested)} requested samples."
        )
    add_shuffle_metrics(model, records, device)
    for number, record in enumerate(records):
        visualize_sample(record, number, output_dir)
    rows = [record["metrics"] for record in records]

    summary_path = output_dir / "summary.tsv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved visualization to {output_dir}")
    print("========== Stage-A Quick Evaluation ==========")
    print(f"samples        : {len(rows)}")
    for key in ("action_mse", "pred_cosine", "copy_cosine", "cosine_gain", "shuffle_gap", "copy_error"):
        print(f"{key}: {np.mean([float(row[key]) for row in rows]):.6f}")
    print("==============================================")


if __name__ == "__main__":
    main()
