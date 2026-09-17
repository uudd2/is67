import argparse
import json
import math
import os
import random

import numpy as np
import torch
import wandb
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.libero_act import LiberoAct
from src.datasets.av_aloha_multitask_act import AVAlohaMultitaskAct
from src.models.edar_lite import (
    EDARFeatureCache,
    FrozenDINOFeatureExtractor,
    SingleViewEDARLite,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage A training for single-view EDAR-lite.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_scheduler(optimizer, total_steps, warmup_steps, min_lr_ratio):
    def schedule(step):
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1.0 / warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, schedule)


def _normalization_metadata(dataset, config):
    if config["data"].get("dataset_name") == "av_aloha":
        mode = config["data"].get("action_normalization", "min_max")
        stats = dataset.stats["action"]
        return {
            "mode": mode,
            "stats": {name: value.tolist() for name, value in stats.items()},
        }
    metadata = {"mode": config["data"].get("normalization_mode", "min_max")}
    if config["data"].get("task_suite_names"):
        metadata["scope"] = "shared_union_of_suite_bounds"
        metadata["task_suite_names"] = list(config["data"]["task_suite_names"])
    if metadata["mode"] == "min_max":
        metadata["action_min"] = dataset.action_min.tolist()
        metadata["action_max"] = dataset.action_max.tolist()
    elif metadata["mode"] == "mean_std":
        metadata["stats_path"] = config["data"]["normalization_stats_path"]
        metadata["stats"] = {
            key: {name: value.tolist() for name, value in values.items()}
            for key, values in dataset.normalization_stats.items()
        }
    return metadata


def _load_visual_source(config, device):
    cache_dir = os.path.expanduser(config["data"].get("feature_cache", ""))
    dino_path = os.path.realpath(os.path.expanduser(config["model"]["dino_model"]))
    if cache_dir and os.path.exists(os.path.join(cache_dir, EDARFeatureCache.METADATA_FILE)):
        cache_metadata = EDARFeatureCache.read_metadata(cache_dir)
        cached_backbone = os.path.realpath(
            os.path.expanduser(cache_metadata["backbone"])
        )
        if cached_backbone != dino_path:
            raise ValueError("Configured DINO path does not match EDAR cache metadata.")
        if (
            int(cache_metadata["image_size"]) != 256
            or int(cache_metadata["output_grid"]) != 8
        ):
            raise ValueError("EDAR cache must use 256px input and an 8x8 output grid.")
        checkpoint_metadata = dict(cache_metadata)
        checkpoint_metadata["backbone"] = dino_path
        return (
            EDARFeatureCache(cache_dir, cache_metadata),
            None,
            checkpoint_metadata,
        )
    extractor = FrozenDINOFeatureExtractor(dino_path).to(device)
    return None, extractor, extractor.metadata()


def _features_for_batch(batch, cache, extractor, device):
    if cache is not None:
        current = cache.get_many(batch["frame_id"])
        future = cache.get_many(batch["future_frame_id"])
        return current.to(device), future.to(device)
    images = torch.cat([batch["image"], batch["future_image"]], dim=0).to(device)
    features = extractor(images)
    # The frozen extractor uses inference_mode; trainable linear layers must
    # receive ordinary tensors so they can save their inputs for backward.
    features = features.clone()
    current, future = features.chunk(2, dim=0)
    return current, future


def save_checkpoint(path, model, optimizer, scheduler, step, config, dino_metadata, normalization):
    checkpoint = {
        "schema": "single_view_edar_lite_stage_a_v1",
        "step": int(step),
        "edar_state_dict": model.state_dict(),
        "encoder_state_dict": model.encoder.state_dict(),
        "decoder_state_dict": model.decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config,
        "dino_metadata": dino_metadata,
        "action_normalization": normalization,
    }
    temporary = f"{path}.tmp"
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    seed = int(config["train"].get("seed", 2026))
    set_seed(seed)
    device = torch.device(config["train"].get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    cache, extractor, dino_metadata = _load_visual_source(config, device)
    data_config = config["data"]
    architecture = config["model"].get("action_representation", {})
    action_horizon = int(architecture.get("action_horizon", 8))
    action_dim = int(architecture.get("action_dim", config["model"].get("action_dim", 7)))
    if data_config.get("dataset_name") == "av_aloha":
        dataset = AVAlohaMultitaskAct(
            data_root=os.path.expanduser(data_config["data_root"]),
            tasks=data_config["av_aloha_tasks"],
            history_len=1,
            future_len=action_horizon,
            action_stride=int(data_config.get("action_stride", 1)),
            full_sequence=True,
            input_modality="image",
            view_mode="single",
            load_future_image=cache is None,
            frame_ids_only=cache is not None,
            future_image_mode="horizon",
            buffer_size=int(data_config.get("buffer_size", 1000)),
            main_camera=data_config.get(
                "main_camera", "observation.images.zed_cam_left"
            ),
            action_normalization=data_config.get("action_normalization", "min_max"),
            state_normalization=data_config.get("state_normalization", "identity"),
            balance_tasks=bool(data_config.get("balance_tasks", True)),
        )
    elif data_config.get("task_suite_names"):
        from src.datasets.libero_mixed_stage_a import LiberoMixedStageA

        dataset = LiberoMixedStageA(data_config, action_horizon)
    else:
        dataset = LiberoAct(
            data_path=os.path.join(
                os.path.expanduser(data_config["data_root"]),
                data_config["task_suite_name"],
                "1.0.0",
            ),
            dataset_name=data_config["task_suite_name"],
            history_len=1,
            future_len=action_horizon,
            full_sequence=True,
            input_modality="image",
            view_mode="single",
            load_future_image=cache is None,
            future_image_mode="horizon",
            strict_future_horizon=True,
            frame_ids_only=cache is not None,
            buffer_size=int(data_config.get("buffer_size", 1000)),
            normalization_mode=data_config.get("normalization_mode", "min_max"),
            normalization_stats_path=data_config.get("normalization_stats_path"),
        )
    micro_batch = int(config["train"].get("per_device_batch_size", 8))
    loader = DataLoader(
        dataset,
        batch_size=micro_batch,
        num_workers=int(config["data"].get("num_workers", 1)),
        pin_memory=device.type == "cuda",
        persistent_workers=int(config["data"].get("num_workers", 1)) > 0,
        **({"multiprocessing_context": data_config["multiprocessing_context"]}
           if int(data_config.get("num_workers", 1)) > 0
           and data_config.get("multiprocessing_context") else {}),
    )
    visual_dim = int(dino_metadata["hidden_size"])
    model = SingleViewEDARLite(
        action_dim=action_dim,
        action_horizon=action_horizon,
        visual_dim=visual_dim,
        visual_grid=8,
        model_dim=int(architecture.get("model_dim", 512)),
        latent_tokens=int(architecture.get("latent_tokens", 4)),
        latent_token_dim=int(architecture.get("latent_token_dim", 256)),
        num_layers=int(architecture.get("layers", 4)),
        num_heads=int(architecture.get("heads", 8)),
        mlp_ratio=float(architecture.get("mlp_ratio", 4.0)),
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        betas=tuple(config["train"].get("optimizer_betas", [0.9, 0.99])),
        weight_decay=float(config["train"].get("weight_decay", 0.01)),
    )
    total_steps = int(config["train"].get("steps", 100000))
    scheduler = build_scheduler(
        optimizer,
        total_steps,
        int(config["train"].get("warmup_steps", 1000)),
        float(config["train"].get("min_learning_rate", 1e-5))
        / float(config["train"]["learning_rate"]),
    )
    accumulation_steps = int(config["train"].get("gradient_accumulation_steps", 8))
    if micro_batch * accumulation_steps != int(config["train"].get("batch_size", 64)):
        raise ValueError("per_device_batch_size * gradient_accumulation_steps must equal batch_size.")

    output_dir = os.path.join(
        os.path.expanduser(config["project"]["output_dir"]),
        config["project"]["name"],
    )
    os.makedirs(output_dir, exist_ok=True)
    normalization = _normalization_metadata(dataset, config)
    start_step = 0
    resume_path = config["train"].get("resume_path", "")
    if resume_path:
        checkpoint = torch.load(os.path.expanduser(resume_path), map_location="cpu")
        if checkpoint.get("schema") != "single_view_edar_lite_stage_a_v1":
            raise ValueError("Resume checkpoint is not an EDAR-lite Stage A checkpoint.")
        model.load_state_dict(checkpoint["edar_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = int(checkpoint["step"])

    use_wandb = bool(config["project"].get("use_wandb", True))
    if use_wandb:
        wandb.init(
            project=config["project"].get("wandb_project", "VLANeXt_vita_hiermq54"),
            name=config["project"]["name"],
            config=config,
        )
    model.train()
    iterator = iter(loader)
    if cache is not None and bool(config["data"].get("preload_feature_cache", False)):
        cache.preload()
    optimizer.zero_grad(set_to_none=True)
    print(json.dumps({"event": "training_start", "start_step": start_step,
                      "total_steps": total_steps, "output_dir": output_dir,
                      "online_dino": cache is None,
                      "action_normalization": normalization}), flush=True)
    progress = tqdm(range(start_step, total_steps), initial=start_step, total=total_steps, desc="EDAR Stage A")
    effect_weight = float(architecture.get("lambda_effect", 0.2))
    precision = str(config["train"].get("precision", "bf16"))
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    log_interval = int(config["project"].get("log_interval", 20))
    save_interval = int(config["project"].get("save_interval", 20000))
    suite_counts = {}

    for step_index in progress:
        total_loss_sum = None
        metric_sums = {}
        last_metrics = None
        for _ in range(accumulation_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            actions = batch["future_actions"][:, :action_horizon].to(
                device, non_blocking=True
            )
            for suite in batch.get("suite_name", []):
                suite_counts[suite] = suite_counts.get(suite, 0) + 1
            current_visual, future_visual = _features_for_batch(
                batch,
                cache,
                extractor,
                device,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=device.type == "cuda",
            ):
                outputs = model(actions, current_visual)
                total_loss, metrics = model.representation_loss(
                    outputs,
                    actions,
                    future_visual,
                    effect_weight=effect_weight,
                )
                scaled_loss = total_loss / accumulation_steps
            scaled_loss.backward()
            detached_loss = total_loss.detach()
            total_loss_sum = (
                detached_loss
                if total_loss_sum is None
                else total_loss_sum + detached_loss
            )
            for name, value in metrics.items():
                metric_sums[name] = metric_sums.get(name, 0.0) + value
            last_metrics = metrics

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(config["train"].get("max_grad_norm", 1.0)),
            error_if_nonfinite=True,
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        total_loss_mean = total_loss_sum / accumulation_steps
        metrics = {
            name: value / accumulation_steps
            for name, value in metric_sums.items()
        }

        step = step_index + 1
        if step % log_interval == 0:
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=device.type == "cuda",
            ):
                shuffled_latent = outputs["action_latent"].roll(1, dims=0)
                _, shuffled_visual = model.decoder(
                    shuffled_latent,
                    current_visual,
                )
                shuffled_loss, _ = model.change_weighted_effect_loss(
                    shuffled_visual,
                    current_visual,
                    future_visual,
                )
                shuffle_gap = shuffled_loss - last_metrics["loss_effect"]
                latent = outputs["action_latent"].float()
                log_values = {
                    "train/loss_action": metrics["loss_action"].item(),
                    "train/loss_effect": metrics["loss_effect"].item(),
                    "train/total_loss": total_loss_mean.item(),
                    "train/z_mean": latent.mean().item(),
                    "train/z_std": latent.std().item(),
                    "train/z_norm": latent.norm(dim=-1).mean().item(),
                    "train/visual_cosine": metrics["visual_cosine"].item(),
                    "train/shuffle_gap": shuffle_gap.item(),
                    "train/grad_norm": float(grad_norm),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/step": step,
                }
            progress.set_postfix(loss=f"{total_loss_mean.item():.4f}")
            print(json.dumps({**log_values, "suite_samples": suite_counts}), flush=True)
            if use_wandb:
                wandb.log(log_values, step=step)
        if step % save_interval == 0:
            save_checkpoint(
                os.path.join(output_dir, f"checkpoint_{step}.pt"),
                model,
                optimizer,
                scheduler,
                step,
                config,
                dino_metadata,
                normalization,
            )

    save_checkpoint(
        os.path.join(output_dir, "checkpoint_final.pt"),
        model,
        optimizer,
        scheduler,
        total_steps,
        config,
        dino_metadata,
        normalization,
    )
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
