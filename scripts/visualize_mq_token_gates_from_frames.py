import argparse
import csv
import os
import re
import sys
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
from scripts.visualize_mq_token_gates import _build_token_labels, _save_sample_images
from src.evaluation.libero_bench.VLANeXt_utils import get_processor, get_vla


def _frame_sort_key(path):
    match = re.search(r"frame_(\d+)$", os.path.basename(path))
    return int(match.group(1)) if match else os.path.basename(path)


def _write_gate_outputs(model, sample, collator, model_cfg, data_cfg, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    _save_sample_images(sample, output_dir)

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
    with torch.no_grad():
        _ = model.predict_action(
            proprioception=proprio,
            history_actions=hist_actions,
            **forward_args,
        )
    gate_records = flow.get_gate_records()
    flow.set_gate_capture(False)
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
            rows.append(gate.float().detach().cpu().numpy())
            row_labels.append(f"step{step_idx:02d}_flow{layer_idx:02d}")
    matrix = np.stack(rows, axis=0)

    csv_path = os.path.join(output_dir, "gate_token_values.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row"] + token_labels)
        for label, values in zip(row_labels, matrix):
            writer.writerow([label] + [f"{float(v):.6f}" for v in values])

    for suffix, dpi, fig_scale in (
        ("gate_token_heatmap.png", 180, (0.28, 0.16, 10, 6, 7)),
        ("gate_token_heatmap_large.png", 220, (0.45, 0.24, 16, 10, 8)),
    ):
        x_scale, y_scale, min_w, min_h, xtick_size = fig_scale
        plt.figure(figsize=(max(min_w, num_tokens * x_scale), max(min_h, len(row_labels) * y_scale)))
        im = plt.imshow(matrix, aspect="auto", cmap="viridis")
        plt.colorbar(im, label="token gate, head-mean")
        plt.yticks(np.arange(len(row_labels)), row_labels, fontsize=max(6, xtick_size - 1))
        plt.xticks(np.arange(num_tokens), token_labels, rotation=90, fontsize=xtick_size)
        ax = plt.gca()
        ax.set_xticks(np.arange(-0.5, num_tokens, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.4, alpha=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        plt.xlabel("MQ condition token")
        plt.ylabel("sampling step / flow layer")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, suffix), dpi=dpi)
        plt.close()

    return csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--diffusion-steps", type=int, default=6)
    parser.add_argument(
        "--instruction",
        default="put both the alphabet soup and the tomato sauce in the basket",
    )
    parser.add_argument("--instruction-file", default=None)
    parser.add_argument("--view0-name", default="sample_view0.png")
    parser.add_argument("--view1-name", default="sample_view1.png")
    args = parser.parse_args()
    if args.instruction_file:
        with open(args.instruction_file, "r") as f:
            args.instruction = f.read().strip()

    cfg = SimpleNamespace(
        eval=SimpleNamespace(finetuned_checkpoint=args.checkpoint),
        model=SimpleNamespace(diffusion_steps=args.diffusion_steps),
    )
    print("[frames-vis] loading model...", flush=True)
    model = get_vla(cfg)
    print("[frames-vis] loading processor...", flush=True)
    processor = get_processor(cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    train_cfg = checkpoint["config"]
    data_cfg = train_cfg["data"]
    model_cfg = train_cfg["model"]

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

    frame_dirs = [
        os.path.join(args.frames_dir, name)
        for name in os.listdir(args.frames_dir)
        if os.path.isdir(os.path.join(args.frames_dir, name)) and name.startswith("frame_")
    ]
    frame_dirs.sort(key=_frame_sort_key)
    if not frame_dirs:
        raise FileNotFoundError(f"No frame_* directories found in {args.frames_dir}")

    history_len = int(data_cfg.get("history_len", 8))
    future_len = int(data_cfg.get("future_len", 8))
    os.makedirs(args.output_dir, exist_ok=True)
    for frame_dir in frame_dirs:
        frame_name = os.path.basename(frame_dir)
        view0_path = os.path.join(frame_dir, args.view0_name)
        view1_path = os.path.join(frame_dir, args.view1_name)
        if not os.path.exists(view0_path):
            print(f"[frames-vis] skip {frame_name}: missing {view0_path}", flush=True)
            continue
        if not os.path.exists(view1_path):
            view1_path = view0_path

        image = np.asarray(Image.open(view0_path).convert("RGB").resize((256, 256)))
        wrist_image = np.asarray(Image.open(view1_path).convert("RGB").resize((256, 256)))
        sample = {
            "instruction": args.instruction,
            "image": image,
            "image_wrist": wrist_image,
            "images": [image, wrist_image],
            "proprioception": torch.zeros(history_len, 7),
            "history_actions": torch.zeros(history_len, 7),
            "future_actions": torch.zeros(future_len, 7),
        }
        out_dir = os.path.join(args.output_dir, frame_name)
        print(f"[frames-vis] {frame_name} -> {out_dir}", flush=True)
        csv_path = _write_gate_outputs(model, sample, collator, model_cfg, data_cfg, out_dir)
        print(f"[frames-vis] saved {csv_path}", flush=True)

    print(f"[frames-vis] done: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
