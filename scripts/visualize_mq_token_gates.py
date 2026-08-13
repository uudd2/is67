import argparse
import csv
import os
import sys
from itertools import islice
from types import SimpleNamespace

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from scripts.train import DataCollatorForVLANeXt
from src.datasets.libero_act import LiberoAct
from src.evaluation.libero_bench.VLANeXt_utils import get_processor, get_vla


def _build_token_labels(counts, num_tokens):
    labels = []
    for name in ("local", "cross", "global", "state"):
        for idx in range(int(counts.get(name, 0))):
            labels.append(f"{name}_{idx:02d}")
    if len(labels) < num_tokens:
        labels.extend(f"tok_{idx:02d}" for idx in range(len(labels), num_tokens))
    return labels[:num_tokens]


def _save_sample_images(sample, output_dir):
    if "images" in sample:
        images = sample["images"]
    elif "image_wrist" in sample:
        images = [sample["image"], sample["image_wrist"]]
    else:
        images = [sample["image"]]
    for idx, image in enumerate(images):
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        plt.imsave(os.path.join(output_dir, f"sample_view{idx}.png"), arr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="mq_gate_vis")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--diffusion-steps", type=int, default=6)
    parser.add_argument("--task-suite", default=None)
    parser.add_argument("--dummy", action="store_true", help="Use zero images instead of reading LIBERO.")
    parser.add_argument("--image", default=None, help="Path to a main-view image.")
    parser.add_argument("--wrist-image", default=None, help="Path to a wrist-view image. Defaults to --image.")
    parser.add_argument("--libero-first-frame", action="store_true", help="Read the first frame from a LIBERO TFDS trajectory.")
    parser.add_argument(
        "--instruction",
        default="put both the alphabet soup and the tomato sauce in the basket",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = SimpleNamespace(
        eval=SimpleNamespace(finetuned_checkpoint=args.checkpoint),
        model=SimpleNamespace(diffusion_steps=args.diffusion_steps),
    )
    print("[vis] loading model...", flush=True)
    model = get_vla(cfg)
    print("[vis] loading processor...", flush=True)
    processor = get_processor(cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    train_cfg = checkpoint["config"]
    data_cfg = train_cfg["data"]
    model_cfg = train_cfg["model"]

    dataset_name = data_cfg.get("dataset_name", "libero")
    if dataset_name != "libero":
        raise ValueError("This visualization script currently supports LiberoAct samples only.")

    if args.dummy or args.image:
        print("[vis] building image-file/dummy sample...", flush=True)
        history_len = int(data_cfg.get("history_len", 8))
        future_len = int(data_cfg.get("future_len", 8))
        if args.image:
            image = np.asarray(Image.open(args.image).convert("RGB").resize((256, 256)))
            wrist_path = args.wrist_image or args.image
            wrist_image = np.asarray(Image.open(wrist_path).convert("RGB").resize((256, 256)))
        else:
            image = np.zeros((256, 256, 3), dtype=np.uint8)
            wrist_image = image.copy()
        sample = {
            "instruction": args.instruction,
            "image": image,
            "image_wrist": wrist_image,
            "images": [image, wrist_image],
            "proprioception": torch.zeros(history_len, 7),
            "history_actions": torch.zeros(history_len, 7),
            "future_actions": torch.zeros(future_len, 7),
        }
    elif args.libero_first_frame:
        print("[vis] reading first LIBERO trajectory frame directly...", flush=True)
        import tensorflow as tf
        import tensorflow_datasets as tfds

        tf.config.set_visible_devices([], "GPU")
        task_suite = args.task_suite or data_cfg["task_suite_name"]
        data_path = os.path.join(data_cfg["data_root"], task_suite, "1.0.0")
        builder = tfds.builder_from_directory(builder_dir=data_path)
        read_config = tfds.ReadConfig(shuffle_seed=42, shuffle_reshuffle_each_iteration=False)
        ds = builder.as_dataset(split="train", shuffle_files=False, read_config=read_config)

        traj_batch = None
        traj_id = None
        for idx, traj_data in enumerate(ds):
            batch = next(iter(traj_data["steps"].batch(2000)))
            if batch["reward"][-1].numpy() == 1:
                traj_batch = batch
                traj_id = idx
                break
        if traj_batch is None:
            raise RuntimeError(f"No successful trajectory found in {data_path}")

        obs = traj_batch["observation"]
        image = obs["image"][0].numpy()
        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        if "wrist_image" in obs:
            wrist_image = obs["wrist_image"][0].numpy()
            if wrist_image.dtype != np.uint8:
                wrist_image = (wrist_image * 255).clip(0, 255).astype(np.uint8)
        else:
            wrist_image = image.copy()

        instruction = traj_batch["language_instruction"][0].numpy().decode("utf-8")
        history_len = int(data_cfg.get("history_len", 8))
        future_len = int(data_cfg.get("future_len", 8))
        sample = {
            "instruction": instruction,
            "image": image,
            "image_wrist": wrist_image,
            "images": [image, wrist_image],
            "proprioception": torch.zeros(history_len, 7),
            "history_actions": torch.zeros(history_len, 7),
            "future_actions": torch.zeros(future_len, 7),
        }
        print(f"[vis] using traj_id={traj_id}, instruction={instruction!r}", flush=True)
    else:
        task_suite = args.task_suite or data_cfg["task_suite_name"]
        data_path = os.path.join(data_cfg["data_root"], task_suite, "1.0.0")
        print(f"[vis] reading LIBERO sample {args.sample_index} from {data_path}...", flush=True)
        dataset = LiberoAct(
            data_path=data_path,
            dataset_name=task_suite,
            history_len=data_cfg.get("history_len", 8),
            future_len=data_cfg.get("future_len", 8),
            full_sequence=data_cfg.get("full_sequence", True),
            input_modality=data_cfg.get("input_modality", "image"),
            view_mode=data_cfg.get("view_mode", "single"),
            load_future_image=False,
            buffer_size=1,
        )
        sample = next(islice(iter(dataset), args.sample_index, None))
    _save_sample_images(sample, args.output_dir)

    collator = DataCollatorForVLANeXt(
        processor=getattr(model, "processor", processor),
        use_proprio_input_vlm=model_cfg.get("use_proprio_input_vlm", False),
        use_action_input_policy=model_cfg.get("use_action_input_policy", False),
        input_modality=data_cfg.get("input_modality", "image"),
        view_mode=data_cfg.get("view_mode", "single"),
        fps=float(data_cfg.get("fps", 20.0)),
        augmentation={"enabled": False},
        include_proprio=(
            model_cfg.get("action_generation_mode") == "vita_latent_flow"
            and model_cfg.get("vita_use_proprio", True)
        ),
    )
    print("[vis] collating sample...", flush=True)
    inputs, _, proprio, hist_actions, _ = collator([sample])

    device = next(model.parameters()).device
    model_inputs = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    for key in ("pixel_values", "pixel_values_videos"):
        if key in model_inputs:
            model_inputs[key] = model_inputs[key].to(dtype=torch.bfloat16)
    if proprio is not None:
        proprio = proprio.to(device, dtype=torch.bfloat16)
    if hist_actions is not None:
        hist_actions = hist_actions.to(device, dtype=torch.bfloat16)

    valid_keys = {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "token_type_ids",
        "mm_token_type_ids",
    }
    forward_args = {key: value for key, value in model_inputs.items() if key in valid_keys}

    flow = model.vita_action_generator.flow
    flow.set_gate_capture(True)
    print("[vis] running predict_action and capturing gates...", flush=True)
    with torch.no_grad():
        _ = model.predict_action(
            proprioception=proprio,
            history_actions=hist_actions,
            **forward_args,
        )
    gate_records = flow.get_gate_records()
    flow.set_gate_capture(False)
    print("[vis] writing heatmap/csv...", flush=True)

    if not gate_records or not gate_records[0]:
        raise RuntimeError("No MQ token gate records captured. Check whether token value gating is enabled.")

    rows = []
    row_labels = []
    num_tokens = gate_records[0][0].shape[1]
    token_labels = _build_token_labels(flow.condition_token_type_counts, num_tokens)
    for layer_idx, layer_records in enumerate(gate_records):
        for step_idx, gate in enumerate(layer_records):
            gate = gate[0]
            if gate.ndim == 2:
                gate = gate.mean(dim=-1)
            rows.append(gate.numpy())
            row_labels.append(f"step{step_idx:02d}_flow{layer_idx:02d}")
    matrix = np.stack(rows, axis=0)

    csv_path = os.path.join(args.output_dir, "gate_token_values.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row"] + token_labels)
        for label, values in zip(row_labels, matrix):
            writer.writerow([label] + [f"{float(v):.6f}" for v in values])

    plt.figure(figsize=(max(10, num_tokens * 0.28), max(6, len(row_labels) * 0.16)))
    im = plt.imshow(matrix, aspect="auto", cmap="viridis")
    plt.colorbar(im, label="token gate, head-mean")
    plt.yticks(np.arange(len(row_labels)), row_labels, fontsize=6)
    plt.xticks(np.arange(num_tokens), token_labels, rotation=90, fontsize=7)
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, num_tokens, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.35, alpha=0.75)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.xlabel("MQ condition token")
    plt.ylabel("sampling step / flow layer")
    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, "gate_token_heatmap.png")
    plt.savefig(fig_path, dpi=180)
    plt.close()

    plt.figure(figsize=(max(16, num_tokens * 0.45), max(10, len(row_labels) * 0.24)))
    im = plt.imshow(matrix, aspect="auto", cmap="viridis")
    plt.colorbar(im, label="token gate, head-mean")
    plt.yticks(np.arange(len(row_labels)), row_labels, fontsize=7)
    plt.xticks(np.arange(num_tokens), token_labels, rotation=90, fontsize=8)
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, num_tokens, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.45, alpha=0.85)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.xlabel("MQ condition token")
    plt.ylabel("sampling step / flow layer")
    plt.tight_layout()
    large_fig_path = os.path.join(args.output_dir, "gate_token_heatmap_large.png")
    plt.savefig(large_fig_path, dpi=220)
    plt.close()
    print(f"saved: {fig_path}")
    print(f"saved: {large_fig_path}")
    print(f"saved: {csv_path}")
    print(f"saved sample images under: {args.output_dir}")


if __name__ == "__main__":
    main()
