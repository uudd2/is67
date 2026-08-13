#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.models.vita_latent_flow import VisualTokenResampler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="pretrained/Qwen3.5-0.8B")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="visual_pooling_diagnostic")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def feature_metrics(tokens, pooled):
    x = tokens.float()
    p = pooled.float()
    cosine = F.normalize(x, dim=-1) @ F.normalize(p, dim=-1).T

    centered_x = x - x.mean(dim=0, keepdim=True)
    centered_p = p - x.mean(dim=0, keepdim=True)
    nonzero = centered_p.norm(dim=-1) > 1e-8
    if nonzero.any():
        basis = torch.linalg.qr(centered_p[nonzero].T, mode="reduced").Q
        projected = centered_x @ basis @ basis.T
        variance_retained = (
            projected.square().sum() / centered_x.square().sum().clamp_min(1e-12)
        ).item()
    else:
        variance_retained = 0.0

    return {
        "num_tokens": int(p.shape[0]),
        "mean_best_cosine_coverage": cosine.max(dim=1).values.mean().item(),
        "min_best_cosine_coverage": cosine.max(dim=1).values.min().item(),
        "centered_subspace_variance_retained": variance_retained,
    }


def save_heatmap(array, path, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(array, cmap="viridis")
    ax.set_title(title)
    ax.set_xticks(np.arange(array.shape[1]))
    ax.set_yticks(np.arange(array.shape[0]))
    ax.set_xticks(np.arange(-0.5, array.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, array.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    source_image = Image.open(args.image).convert("RGB")
    source_image.save(output_dir / "sample_frame.png")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": source_image},
        {"type": "text", "text": "Describe the robot manipulation scene."},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[source_image], return_tensors="pt")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device).eval()

    pixels = inputs["pixel_values"].to(device=device, dtype=dtype)
    grid = inputs["image_grid_thw"].to(device=device)
    with torch.no_grad():
        vision_output = model.model.get_image_features(pixels, grid, return_dict=True)
        tokens = tuple(vision_output.pooler_output)[0]

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    prefix = "vita_action_generator.visual_resampler."
    resampler_state = {
        key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)
    }
    if not resampler_state:
        raise KeyError(f"No visual resampler parameters found under prefix {prefix}")

    resampler = VisualTokenResampler(dim=tokens.shape[-1], num_queries=4, num_heads=8)
    resampler.load_state_dict(resampler_state)
    resampler = resampler.to(device=device, dtype=dtype).eval()

    with torch.no_grad():
        context = resampler.norm(tokens.unsqueeze(0))
        queries = resampler.query.expand(1, -1, -1).to(device=device, dtype=dtype)
        learned, attention = resampler.attn(
            queries,
            context,
            context,
            need_weights=True,
            average_attn_weights=False,
        )
        learned = learned[0]
        attention = attention.float().mean(dim=1)[0]

    merged_h = int(grid[0, 1].item()) // 2
    merged_w = int(grid[0, 2].item()) // 2
    if merged_h * merged_w != tokens.shape[0]:
        raise ValueError(
            f"Expected a {merged_h}x{merged_w} token grid, got {tokens.shape[0]} tokens"
        )

    token_map = tokens.float().reshape(merged_h, merged_w, -1).permute(2, 0, 1).unsqueeze(0)
    spatial_2x2 = F.adaptive_avg_pool2d(token_map, (2, 2))
    spatial_tokens = spatial_2x2[0].permute(1, 2, 0).reshape(4, -1)
    mean_token = tokens.float().mean(dim=0, keepdim=True)
    learned_attention_pool = attention @ tokens.float()

    metrics = {
        "source_image": str(Path(args.image).resolve()),
        "original_token_shape": list(tokens.shape),
        "token_grid": [merged_h, merged_w],
        "mean_pool_1": feature_metrics(tokens, mean_token),
        "spatial_pool_2x2": feature_metrics(tokens, spatial_tokens),
        "learned_attention_pool_4": feature_metrics(tokens, learned_attention_pool),
        "learned_resampler_output_4": feature_metrics(tokens, learned),
        "note": (
            "Metrics are feature-space proxies, not downstream task-information accuracy. "
            "The learned_attention_pool metric applies learned attention weights directly to raw "
            "tokens for a fair comparison. The native Resampler output uses learned V/O projections "
            "and therefore is not in the original token coordinate system."
        ),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    token_norm = tokens.float().norm(dim=-1).reshape(merged_h, merged_w).cpu().numpy()
    save_heatmap(token_norm, output_dir / "original_token_norm.png", "Original visual-token norm")
    pooled_norm = spatial_2x2[0].norm(dim=0).cpu().numpy()
    save_heatmap(pooled_norm, output_dir / "spatial_pool_2x2_norm.png", "Direct spatial pool (2x2)")
    for index, weights in enumerate(attention):
        heatmap = weights.reshape(merged_h, merged_w).cpu().numpy()
        save_heatmap(
            heatmap,
            output_dir / f"resampler_query_{index}_attention.png",
            f"Learned Resampler query {index}",
        )

    print(json.dumps(metrics, indent=2))
    print(f"Saved diagnostics to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
