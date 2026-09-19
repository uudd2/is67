"""Stage-A training for real Piper LeRobot data using the top camera only."""
from __future__ import annotations

import argparse
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

CHECKPOINT_SCHEMA = "edar_lite_real_piper_top_stage_a_v1"


def parse_args():
    parser = argparse.ArgumentParser(description="Real Piper top-view EDAR-lite Stage-A training.")
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


def cosine_diagnostics(predicted, current, future):
    prediction = F.normalize(predicted.float(), dim=-1)
    current = F.normalize(current.float(), dim=-1)
    future = F.normalize(future.float(), dim=-1)
    pred_cosine = (prediction * future).sum(dim=-1).mean()
    copy_cosine = (current * future).sum(dim=-1).mean()
    return {
        "pred_cosine": pred_cosine,
        "copy_cosine": copy_cosine,
        "cosine_gain": pred_cosine - copy_cosine,
    }


def _json_stats(stats):
    return {
        key: (int(value) if key == "count" else np.asarray(value).tolist())
        for key, value in stats.items()
    }


def _online_features(batch, extractor, device):
    images = torch.cat([batch["image"], batch["future_image"]], dim=0).to(
        device, non_blocking=True
    )
    features = extractor(images).clone()
    return features.chunk(2, dim=0)


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    step,
    config,
    dino_metadata,
    action_stats,
    split_manifest,
):
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "step": int(step),
        "edar_state_dict": model.state_dict(),
        "encoder_state_dict": model.encoder.state_dict(),
        "decoder_state_dict": model.decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config,
        "dino_metadata": dino_metadata,
        "action_normalization": {
            "mode": config["data"].get("action_normalization", "mean_std"),
            "stats": _json_stats(action_stats),
            "source": "train_episodes_only",
        },
        "episode_split": split_manifest,
    }
    temporary = f"{path}.tmp"
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def validate(model, loader, extractor, device, effect_weight, autocast_dtype, max_batches):
    model.eval()
    sums = {}
    count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        actions = batch["future_actions"].to(device, non_blocking=True)
        current_visual, future_visual = _online_features(batch, extractor, device)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=device.type == "cuda",
        ):
            outputs = model(actions, current_visual)
            total_loss, metrics = model.representation_loss(
                outputs, actions, future_visual, effect_weight=effect_weight
            )
        diagnostics = cosine_diagnostics(
            outputs["predicted_future_visual"], current_visual, future_visual
        )
        values = {
            "loss_action": metrics["loss_action"],
            "loss_effect": metrics["loss_effect"],
            "total_loss": total_loss,
            **diagnostics,
        }
        if actions.shape[0] > 1:
            shuffled_latent = outputs["action_latent"].roll(1, dims=0)
            _, shuffled_visual = model.decoder(shuffled_latent, current_visual)
            shuffled_loss, _ = model.change_weighted_effect_loss(
                shuffled_visual, current_visual, future_visual
            )
            values["shuffle_gap"] = shuffled_loss - metrics["loss_effect"]
        for name, value in values.items():
            sums[name] = sums.get(name, 0.0) + float(value.detach().float().cpu())
        count += 1
    model.train()
    if count == 0:
        raise RuntimeError("Validation loader produced no batches")
    return {f"val/{name}": value / count for name, value in sums.items()}


def main():
    from src.datasets.lerobot_piper_stage_a import (
        PiperLeRobotStageA,
        compute_action_stats,
        make_episode_split_manifest,
    )
    from src.models.edar_lite import FrozenDINOFeatureExtractor, SingleViewEDARLite

    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    data_config = config["data"]
    architecture = config["model"].get("action_representation", {})
    batches = data_config["batches"]
    if not batches:
        raise ValueError("data.batches must contain at least one LeRobot batch")
    if not data_config.get("main_camera"):
        raise ValueError("data.main_camera must be set to the verified top-camera key")

    seed = int(config["train"].get("seed", 2026))
    set_seed(seed)
    device = torch.device(config["train"].get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    action_dim = int(architecture.get("action_dim", 14))
    if action_dim != 14:
        raise ValueError(f"Real Piper Stage-A requires action_dim=14, got {action_dim}")
    action_horizon = int(architecture.get("action_horizon", 8))
    action_stride = int(data_config.get("action_stride", 1))
    video_backend = str(data_config.get("video_backend", "pyav"))

    split_manifest = make_episode_split_manifest(
        batches,
        val_ratio=float(data_config.get("val_ratio", 0.1)),
        seed=int(data_config.get("split_seed", seed)),
    )
    action_stats = compute_action_stats(
        batches,
        split_manifest["train"],
        action_dim=action_dim,
        video_backend=video_backend,
    )
    dataset_kwargs = dict(
        batches=batches,
        action_stats=action_stats,
        main_camera=data_config["main_camera"],
        action_dim=action_dim,
        action_horizon=action_horizon,
        action_stride=action_stride,
        action_normalization=data_config.get("action_normalization", "mean_std"),
        video_backend=video_backend,
        buffer_size=int(data_config.get("buffer_size", 256)),
    )
    train_dataset = PiperLeRobotStageA(
        episode_ids_by_batch=split_manifest["train"], **dataset_kwargs
    )
    val_dataset = PiperLeRobotStageA(
        episode_ids_by_batch=split_manifest["val"],
        **{**dataset_kwargs, "buffer_size": 1},
    )

    num_workers = int(data_config.get("num_workers", 1))
    micro_batch = int(config["train"].get("per_device_batch_size", 8))
    loader_kwargs = dict(
        batch_size=micro_batch,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    val_loader = DataLoader(val_dataset, **loader_kwargs)

    dino_path = os.path.realpath(os.path.expanduser(config["model"]["dino_model"]))
    extractor = FrozenDINOFeatureExtractor(dino_path).to(device)
    dino_metadata = extractor.metadata()
    model = SingleViewEDARLite(
        action_dim=action_dim,
        action_horizon=action_horizon,
        visual_dim=int(dino_metadata["hidden_size"]),
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
    accumulation_steps = int(config["train"].get("gradient_accumulation_steps", 1))
    if micro_batch * accumulation_steps != int(config["train"].get("batch_size", micro_batch)):
        raise ValueError("per_device_batch_size * gradient_accumulation_steps must equal batch_size")

    output_dir = os.path.join(
        os.path.expanduser(config["project"]["output_dir"]), config["project"]["name"]
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "episode_split.json"), "w", encoding="utf-8") as handle:
        json.dump(split_manifest, handle, indent=2, sort_keys=True)
    with open(os.path.join(output_dir, "action_stats.json"), "w", encoding="utf-8") as handle:
        json.dump(_json_stats(action_stats), handle, indent=2, sort_keys=True)

    start_step = 0
    resume_path = config["train"].get("resume_path", "")
    if resume_path:
        checkpoint = torch.load(os.path.expanduser(resume_path), map_location="cpu")
        if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("Resume checkpoint is not a real Piper top-view Stage-A checkpoint")
        if checkpoint.get("episode_split") != split_manifest:
            raise ValueError("Resume checkpoint episode split does not match current config")
        model.load_state_dict(checkpoint["edar_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = int(checkpoint["step"])

    use_wandb = bool(config["project"].get("use_wandb", True))
    if use_wandb:
        import wandb

        wandb.init(
            project=config["project"].get("wandb_project", "VLANeXt_vita_hiermq54"),
            name=config["project"]["name"],
            config=config,
        )
    else:
        wandb = None

    effect_weight = float(architecture.get("lambda_effect", 0.2))
    precision = str(config["train"].get("precision", "bf16"))
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    log_interval = int(config["project"].get("log_interval", 20))
    val_interval = int(config["project"].get("val_interval", 1000))
    save_interval = int(config["project"].get("save_interval", 20000))
    max_val_batches = int(data_config.get("max_val_batches", 100))

    model.train()
    iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(
        range(start_step, total_steps), initial=start_step, total=total_steps, desc="Real EDAR Stage A"
    )
    for step_index in progress:
        total_loss_sum = 0.0
        metric_sums = {}
        for _ in range(accumulation_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            actions = batch["future_actions"].to(device, non_blocking=True)
            current_visual, future_visual = _online_features(batch, extractor, device)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=device.type == "cuda",
            ):
                outputs = model(actions, current_visual)
                total_loss, metrics = model.representation_loss(
                    outputs, actions, future_visual, effect_weight=effect_weight
                )
                (total_loss / accumulation_steps).backward()
            total_loss_sum += float(total_loss.detach())
            for name, value in metrics.items():
                metric_sums[name] = metric_sums.get(name, 0.0) + float(value.detach())

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(config["train"].get("max_grad_norm", 1.0)),
            error_if_nonfinite=True,
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step = step_index + 1

        if step % log_interval == 0:
            values = {
                "train/total_loss": total_loss_sum / accumulation_steps,
                "train/loss_action": metric_sums["loss_action"] / accumulation_steps,
                "train/loss_effect": metric_sums["loss_effect"] / accumulation_steps,
                "train/visual_cosine": metric_sums["visual_cosine"] / accumulation_steps,
                "train/grad_norm": float(grad_norm),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/step": step,
            }
            progress.set_postfix(loss=f"{values['train/total_loss']:.4f}")
            print(json.dumps(values), flush=True)
            if wandb is not None:
                wandb.log(values, step=step)

        if val_interval > 0 and step % val_interval == 0:
            values = validate(
                model,
                val_loader,
                extractor,
                device,
                effect_weight,
                autocast_dtype,
                max_val_batches,
            )
            values["val/step"] = step
            print(json.dumps(values), flush=True)
            if wandb is not None:
                wandb.log(values, step=step)

        if step % save_interval == 0:
            save_checkpoint(
                os.path.join(output_dir, f"checkpoint_{step}.pt"),
                model,
                optimizer,
                scheduler,
                step,
                config,
                dino_metadata,
                action_stats,
                split_manifest,
            )

    save_checkpoint(
        os.path.join(output_dir, "checkpoint_final.pt"),
        model,
        optimizer,
        scheduler,
        total_steps,
        config,
        dino_metadata,
        action_stats,
        split_manifest,
    )
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
