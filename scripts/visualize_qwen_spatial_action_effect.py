#!/usr/bin/env python3
"""Visualize the legacy QwenSpatial64 action-effect teacher predictions."""

import argparse
import csv
import gc
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.visualize_edar_lite_predictions import (
    shared_pca_maps,
    show_grid,
    show_rgb,
)
from src.datasets.libero_act import LiberoAct
from src.models.vita_latent_flow import (
    ActionDecoder,
    ActionEncoder,
    VisualDeltaDecoder,
    spatial_pool_qwen_visual_tokens,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize the old QwenSpatial64 action-effect module and compare "
            "it with the copy-current-feature baseline."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--edar-summary",
        default="",
        help="Optional EDAR summary.tsv produced on the same sample indices.",
    )
    return parser.parse_args()


def _substate(state_dict, prefix):
    result = {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not result:
        raise KeyError(f"Checkpoint has no parameters under {prefix!r}.")
    return result


class QwenSpatialActionEffect(torch.nn.Module):
    def __init__(self, vision, action_encoder, action_decoder, visual_decoder, grid_size):
        super().__init__()
        self.vision = vision
        self.action_encoder = action_encoder
        self.action_decoder = action_decoder
        self.visual_decoder = visual_decoder
        self.grid_size = int(grid_size)

    def encode_images(self, pixel_values, image_grid_thw):
        output = self.vision(pixel_values, grid_thw=image_grid_thw)
        merged_tokens = output.pooler_output
        merge_size = int(self.vision.spatial_merge_size)
        split_sizes = [
            int(grid.prod().item()) // (merge_size * merge_size)
            for grid in image_grid_thw
        ]
        image_tokens = merged_tokens.split(split_sizes, dim=0)
        return torch.stack(
            [
                spatial_pool_qwen_visual_tokens(
                    tokens,
                    grid,
                    spatial_merge_size=merge_size,
                    output_grid_size=self.grid_size,
                )
                for tokens, grid in zip(image_tokens, image_grid_thw)
            ],
            dim=0,
        )

    def forward(self, actions, pixel_values, image_grid_thw):
        visual_tokens = self.encode_images(pixel_values, image_grid_thw)
        current, target = visual_tokens.chunk(2, dim=0)
        action_latent = self.action_encoder(actions)
        predicted_delta = self.visual_decoder(action_latent, current)
        return {
            "current": current,
            "target": target,
            "prediction": current + predicted_delta,
            "decoded_actions": self.action_decoder(action_latent),
        }


def build_model(checkpoint, device):
    config = checkpoint["config"]
    model_config = config["model"]
    if model_config.get("vita_action_effect_visual_teacher") != "qwen_spatial":
        raise ValueError("Checkpoint is not a QwenSpatial action-effect experiment.")

    lmm_path = os.path.expanduser(model_config["lmm_path"])
    qwen_config = AutoConfig.from_pretrained(
        lmm_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    vision = Qwen3_5VisionModel(qwen_config.vision_config)

    action_dim = int(model_config.get("action_dim", 7))
    horizon = int(config["data"].get("future_len", 8))
    latent_dim = int(model_config.get("vita_latent_dim", 1024))
    hidden_dim = int(model_config.get("vita_hidden_dim", 1024))
    action_layers = int(model_config.get("vita_action_ae_layers", 6))
    decoder_layers = int(model_config.get("vita_action_effect_decoder_layers", 2))
    dropout = float(model_config.get("vita_dropout", 0.0))
    grid_size = int(model_config.get("vita_action_effect_spatial_grid_size", 8))
    visual_dim = int(qwen_config.vision_config.out_hidden_size)

    action_encoder = ActionEncoder(
        action_dim,
        horizon,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=action_layers,
        dropout=dropout,
    )
    action_decoder = ActionDecoder(
        action_dim,
        horizon,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=action_layers,
        dropout=dropout,
    )
    visual_decoder = VisualDeltaDecoder(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=decoder_layers,
        dropout=dropout,
        visual_dim=visual_dim,
    )

    state = checkpoint["model_state_dict"]
    vision.load_state_dict(_substate(state, "lmm.model.visual."), strict=True)
    action_encoder.load_state_dict(
        _substate(state, "vita_action_generator.action_encoder."), strict=True
    )
    action_decoder.load_state_dict(
        _substate(state, "vita_action_generator.action_decoder."), strict=True
    )
    visual_decoder.load_state_dict(
        _substate(state, "vita_action_generator.visual_delta_decoder."), strict=True
    )
    model = QwenSpatialActionEffect(
        vision,
        action_encoder,
        action_decoder,
        visual_decoder,
        grid_size,
    )
    processor = AutoProcessor.from_pretrained(
        lmm_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    return model.to(device).eval(), processor, config


def _prepare_images(processor, current_image, future_image, device, dtype):
    processed = processor.image_processor(
        images=[np.asarray(current_image), np.asarray(future_image)],
        return_tensors="pt",
    )
    return (
        processed["pixel_values"].to(device=device, dtype=dtype),
        processed["image_grid_thw"].to(device=device),
    )


def visualize_sample(model, processor, sample, sample_index, output_dir, device):
    actions = sample["future_actions"][:8].unsqueeze(0).to(device).float()
    vision_parameter = next(model.vision.parameters())
    pixel_values, image_grid_thw = _prepare_images(
        processor,
        sample["image"],
        sample["future_image"],
        device,
        vision_parameter.dtype,
    )
    with torch.inference_mode():
        outputs = model(actions, pixel_values, image_grid_thw)

    current = outputs["current"].float()
    target = outputs["target"].float()
    prediction = outputs["prediction"].float()
    decoded_actions = outputs["decoded_actions"].float()
    current_norm = F.normalize(current, dim=-1)
    target_norm = F.normalize(target, dim=-1)
    prediction_norm = F.normalize(prediction, dim=-1)
    patch_cosine = (prediction_norm * target_norm).sum(dim=-1)[0]
    copy_cosine = (current_norm * target_norm).sum(dim=-1)[0]
    error_map = (1.0 - patch_cosine).clamp_min(0.0)
    copy_error_map = (1.0 - copy_cosine).clamp_min(0.0)
    grid_size = model.grid_size
    current_pca, target_pca, prediction_pca = shared_pca_maps(
        current[0], target[0], prediction[0], grid_size
    )

    pred_cosine = patch_cosine.mean().item()
    baseline_cosine = copy_cosine.mean().item()
    pred_error = max(1.0 - pred_cosine, 0.0)
    baseline_error = max(1.0 - baseline_cosine, 0.0)
    error_reduction = 1.0 - pred_error / max(baseline_error, 1e-8)
    target_delta = target - current
    predicted_delta = prediction - current
    delta_huber = F.smooth_l1_loss(predicted_delta, target_delta).item()
    zero_delta_huber = F.smooth_l1_loss(
        torch.zeros_like(target_delta), target_delta
    ).item()
    delta_huber_reduction = 1.0 - delta_huber / max(zero_delta_huber, 1e-12)
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
        "feature_error_reduction": error_reduction,
        "delta_huber": delta_huber,
        "zero_delta_huber": zero_delta_huber,
        "delta_huber_reduction": delta_huber_reduction,
        "improved_patch_fraction": improved_fraction,
        "feature_mse": feature_mse,
        "action_mse": action_mse,
    }

    figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    show_rgb(axes[0, 0], sample["image"], "Current RGB (t)")
    show_rgb(axes[0, 1], sample["future_image"], "Future RGB (t+8)")
    show_grid(axes[0, 2], current_pca, "Current Qwen PCA")
    show_grid(axes[0, 3], target_pca, "True future Qwen PCA")
    show_grid(axes[1, 0], prediction_pca, "Old teacher prediction PCA")
    motion_image = show_grid(
        axes[1, 1],
        copy_error_map.reshape(grid_size, grid_size).cpu().numpy(),
        "True feature motion (1-cos)",
        cmap="magma",
        value_range=(0.0, max(float(copy_error_map.max().item()), 1e-6)),
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
        action_axis.plot(true_actions[:, dimension], linewidth=1.2)
        action_axis.plot(
            predicted_actions[:, dimension], linestyle="--", linewidth=1.0, alpha=0.8
        )
    action_axis.set_title(f"Action reconstruction (MSE={action_mse:.4f})", fontsize=10)
    action_axis.set_xlabel("Chunk step")
    action_axis.grid(alpha=0.25)

    figure.suptitle(
        f"QwenSpatial64 sample={sample_index} | pred cos={pred_cosine:.4f} | "
        f"copy cos={baseline_cosine:.4f} | gain={pred_cosine - baseline_cosine:+.4f} | "
        f"error reduction={error_reduction:+.1%}",
        fontsize=12,
    )
    destination = output_dir / f"sample_{sample_index:06d}.png"
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return metrics


def _write_summary(rows, destination):
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _compare_with_edar(rows, edar_summary, output_dir):
    with Path(edar_summary).expanduser().open(encoding="utf-8", newline="") as handle:
        edar_rows = {
            int(row["sample_index"]): row
            for row in csv.DictReader(handle, delimiter="\t")
        }
    comparison = []
    for row in rows:
        index = int(row["sample_index"])
        if index not in edar_rows:
            continue
        edar = edar_rows[index]
        edar_pred = float(edar["prediction_cosine"])
        edar_copy = float(edar["copy_baseline_cosine"])
        edar_reduction = 1.0 - (1.0 - edar_pred) / max(1.0 - edar_copy, 1e-8)
        comparison.append(
            {
                "sample_index": index,
                "qwen_cosine_gain": row["cosine_gain"],
                "qwen_feature_error_reduction": row["feature_error_reduction"],
                "qwen_improved_patch_fraction": row["improved_patch_fraction"],
                "edar_cosine_gain": float(edar["cosine_gain"]),
                "edar_feature_error_reduction": edar_reduction,
                "edar_improved_patch_fraction": float(edar["improved_patch_fraction"]),
            }
        )
    if comparison:
        _write_summary(comparison, output_dir / "teacher_comparison.tsv")
        print(f"Compared {len(comparison)} matching samples with EDAR.")


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        Path(args.checkpoint).expanduser(),
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    model, processor, config = build_model(checkpoint, device)
    del checkpoint
    gc.collect()

    data_config = config["data"]
    dataset = LiberoAct(
        data_path=os.path.join(
            os.path.expanduser(data_config["data_root"]),
            data_config["task_suite_name"],
            "1.0.0",
        ),
        dataset_name=data_config["task_suite_name"],
        history_len=1,
        future_len=8,
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
        args.start_index + offset * args.stride for offset in range(args.num_samples)
    }
    rows = []
    for index, sample in enumerate(dataset):
        if index not in requested:
            continue
        rows.append(
            visualize_sample(model, processor, sample, index, output_dir, device)
        )
        print(f"Saved sample {index} ({len(rows)}/{len(requested)})")
        if len(rows) == len(requested):
            break
    if len(rows) != len(requested):
        raise RuntimeError(
            f"Dataset ended after producing {len(rows)}/{len(requested)} requested samples."
        )

    _write_summary(rows, output_dir / "summary.tsv")
    if args.edar_summary:
        _compare_with_edar(rows, args.edar_summary, output_dir)
    print(f"Saved visualization to {output_dir}")
    for key in (
        "prediction_cosine",
        "copy_baseline_cosine",
        "cosine_gain",
        "feature_error_reduction",
        "delta_huber",
        "zero_delta_huber",
        "delta_huber_reduction",
        "improved_patch_fraction",
        "feature_mse",
        "action_mse",
    ):
        print(f"{key}: {np.mean([float(row[key]) for row in rows]):.6f}")


if __name__ == "__main__":
    main()
