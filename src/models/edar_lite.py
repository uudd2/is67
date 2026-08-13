import json
import math
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def deterministic_dino_preprocess(images, image_size, image_mean, image_std):
    if not torch.is_tensor(images):
        images = torch.as_tensor(images)
    if images.ndim != 4:
        raise ValueError("DINO images must have shape [B,C,H,W] or [B,H,W,C].")
    if images.shape[-1] == 3 and images.shape[1] != 3:
        images = images.permute(0, 3, 1, 2)
    if images.shape[1] != 3:
        raise ValueError("DINO images must contain three RGB channels.")
    images = images.float()
    if images.max().item() > 1.5:
        images = images / 255.0
    height, width = images.shape[-2:]
    scale = int(image_size) / min(height, width)
    resized_height = max(int(image_size), int(round(height * scale)))
    resized_width = max(int(image_size), int(round(width * scale)))
    images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    top = (resized_height - int(image_size)) // 2
    left = (resized_width - int(image_size)) // 2
    images = images[:, :, top : top + int(image_size), left : left + int(image_size)]
    mean = torch.as_tensor(image_mean, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.as_tensor(image_std, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean) / std


class DeterministicDINOImageProcessor:
    def __init__(self, image_mean, image_std, image_size=256):
        self.image_mean = list(image_mean)
        self.image_std = list(image_std)
        self.image_size = int(image_size)

    def __call__(self, images, return_tensors="pt"):
        if return_tensors != "pt":
            raise ValueError("EDAR DINO preprocessing only supports PyTorch tensors.")
        tensors = []
        for image in images:
            if hasattr(image, "__array__"):
                image = torch.as_tensor(np.asarray(image))
            elif not torch.is_tensor(image):
                image = torch.as_tensor(image)
            tensors.append(image)
        return {
            "pixel_values": deterministic_dino_preprocess(
                torch.stack(tensors),
                self.image_size,
                self.image_mean,
                self.image_std,
            )
        }


def _sincos_1d(positions, dim):
    if dim % 2 != 0:
        raise ValueError("Sine-cosine embedding dimensions must be even.")
    omega = torch.arange(dim // 2, dtype=torch.float32, device=positions.device)
    omega = 1.0 / (10000 ** (omega / max(dim // 2, 1)))
    phase = positions.float().unsqueeze(-1) * omega.unsqueeze(0)
    return torch.cat([phase.sin(), phase.cos()], dim=-1)


def build_2d_sincos_position_embedding(grid_size, dim):
    if dim % 4 != 0:
        raise ValueError("2D sine-cosine embedding dimension must be divisible by 4.")
    coords = torch.arange(grid_size, dtype=torch.float32)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    return torch.cat(
        [_sincos_1d(y.flatten(), dim // 2), _sincos_1d(x.flatten(), dim // 2)],
        dim=-1,
    )


class _PreNormEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm_attn = nn.RMSNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm_ffn = nn.RMSNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, tokens):
        normalized = self.norm_attn(tokens)
        update = self.attn(normalized, normalized, normalized, need_weights=False)[0]
        tokens = tokens + update
        return tokens + self.ffn(self.norm_ffn(tokens))


class SingleViewEDARLiteEncoder(nn.Module):
    def __init__(
        self,
        action_dim=7,
        action_horizon=8,
        visual_dim=768,
        visual_grid=8,
        model_dim=512,
        latent_tokens=4,
        latent_token_dim=256,
        num_layers=4,
        num_heads=8,
        mlp_ratio=4.0,
    ):
        super().__init__()
        if latent_tokens * latent_token_dim != 1024:
            raise ValueError("EDAR-lite must preserve a 1024D flattened action latent.")
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.visual_dim = int(visual_dim)
        self.visual_grid = int(visual_grid)
        self.model_dim = int(model_dim)
        self.latent_tokens = int(latent_tokens)
        self.latent_token_dim = int(latent_token_dim)

        self.action_projection = nn.Linear(self.action_dim, self.model_dim)
        self.visual_projection = nn.Linear(self.visual_dim, self.model_dim)
        self.action_positions = nn.Parameter(
            torch.zeros(1, self.action_horizon, self.model_dim)
        )
        self.action_type = nn.Parameter(torch.zeros(1, 1, self.model_dim))
        self.visual_type = nn.Parameter(torch.zeros(1, 1, self.model_dim))
        self.register_type = nn.Parameter(torch.zeros(1, 1, self.model_dim))
        self.register_tokens = nn.Parameter(
            torch.empty(1, self.latent_tokens, self.model_dim)
        )
        self.register_norm = nn.RMSNorm(self.model_dim)
        self.register_projection = nn.Linear(self.model_dim, self.latent_token_dim)
        self.latent_norm = nn.LayerNorm(self.latent_tokens * self.latent_token_dim)
        self.blocks = nn.ModuleList(
            [
                _PreNormEncoderBlock(self.model_dim, num_heads, mlp_ratio)
                for _ in range(int(num_layers))
            ]
        )
        self.register_buffer(
            "visual_positions",
            build_2d_sincos_position_embedding(self.visual_grid, self.model_dim).unsqueeze(0),
            persistent=False,
        )
        nn.init.normal_(self.register_tokens, std=0.02)
        nn.init.normal_(self.action_positions, std=0.02)

    @property
    def num_visual_tokens(self):
        return self.visual_grid * self.visual_grid

    def forward(self, actions, current_visual_tokens, return_token_latent=False):
        expected_actions = (self.action_horizon, self.action_dim)
        if tuple(actions.shape[-2:]) != expected_actions:
            raise ValueError(
                f"Expected actions [B,{expected_actions[0]},{expected_actions[1]}], "
                f"got {tuple(actions.shape)}."
            )
        expected_visual = (self.num_visual_tokens, self.visual_dim)
        if tuple(current_visual_tokens.shape[-2:]) != expected_visual:
            raise ValueError(
                f"Expected visual tokens [B,{expected_visual[0]},{expected_visual[1]}], "
                f"got {tuple(current_visual_tokens.shape)}."
            )
        batch_size = actions.shape[0]
        action_tokens = (
            self.action_projection(actions)
            + self.action_positions
            + self.action_type
        )
        registers = self.register_tokens.expand(batch_size, -1, -1) + self.register_type
        visual_tokens = (
            self.visual_projection(current_visual_tokens)
            + self.visual_positions.to(dtype=current_visual_tokens.dtype)
            + self.visual_type
        )
        tokens = torch.cat([action_tokens, registers, visual_tokens], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        register_start = self.action_horizon
        register_states = tokens[:, register_start : register_start + self.latent_tokens]
        token_latent = self.register_projection(self.register_norm(register_states))
        action_latent = self.latent_norm(token_latent.flatten(1))
        if return_token_latent:
            return action_latent, token_latent
        return action_latent


class _SharedAttentionDualFFNBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm_attn = nn.RMSNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        hidden_dim = int(dim * mlp_ratio)
        self.norm_act = nn.RMSNorm(dim)
        self.norm_vis = nn.RMSNorm(dim)
        self.ffn_act = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.ffn_vis = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, tokens, attention_mask, action_prefix_length):
        normalized = self.norm_attn(tokens)
        update = self.attn(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            need_weights=False,
        )[0]
        tokens = tokens + update
        action_tokens = tokens[:, :action_prefix_length]
        action_tokens = action_tokens + self.ffn_act(self.norm_act(action_tokens))
        if tokens.shape[1] == action_prefix_length:
            return action_tokens
        visual_tokens = tokens[:, action_prefix_length:]
        visual_tokens = visual_tokens + self.ffn_vis(self.norm_vis(visual_tokens))
        return torch.cat([action_tokens, visual_tokens], dim=1)


class SingleViewEDARLiteDecoder(nn.Module):
    def __init__(
        self,
        action_dim=7,
        action_horizon=8,
        visual_dim=768,
        visual_grid=8,
        model_dim=512,
        latent_tokens=4,
        latent_token_dim=256,
        num_layers=4,
        num_heads=8,
        mlp_ratio=4.0,
    ):
        super().__init__()
        if latent_tokens * latent_token_dim != 1024:
            raise ValueError("EDAR-lite must preserve a 1024D flattened action latent.")
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.visual_dim = int(visual_dim)
        self.visual_grid = int(visual_grid)
        self.model_dim = int(model_dim)
        self.latent_tokens = int(latent_tokens)
        self.latent_token_dim = int(latent_token_dim)
        self.action_prefix_length = self.latent_tokens + self.action_horizon

        self.latent_projection = nn.Linear(self.latent_token_dim, self.model_dim)
        self.action_queries = nn.Parameter(
            torch.empty(1, self.action_horizon, self.model_dim)
        )
        self.action_positions = nn.Parameter(
            torch.zeros(1, self.action_horizon, self.model_dim)
        )
        self.visual_projection = nn.Linear(self.visual_dim, self.model_dim)
        self.blocks = nn.ModuleList(
            [
                _SharedAttentionDualFFNBlock(self.model_dim, num_heads, mlp_ratio)
                for _ in range(int(num_layers))
            ]
        )
        self.action_norm = nn.RMSNorm(self.model_dim)
        self.visual_norm = nn.RMSNorm(self.model_dim)
        self.action_head = nn.Linear(self.model_dim, self.action_dim)
        self.visual_head = nn.Linear(self.model_dim, self.visual_dim)
        self.visual_delta_scale = nn.Parameter(torch.tensor(0.1))
        self.register_buffer(
            "visual_positions",
            build_2d_sincos_position_embedding(self.visual_grid, self.model_dim).unsqueeze(0),
            persistent=False,
        )
        nn.init.normal_(self.action_queries, std=0.02)
        nn.init.normal_(self.action_positions, std=0.02)

    @property
    def num_visual_tokens(self):
        return self.visual_grid * self.visual_grid

    def _build_attention_mask(self, include_visual, device):
        total_length = self.action_prefix_length
        if include_visual:
            total_length += self.num_visual_tokens
        mask = torch.ones(total_length, total_length, dtype=torch.bool, device=device)
        mask[: self.latent_tokens, : self.latent_tokens] = False
        mask[
            self.latent_tokens : self.action_prefix_length,
            : self.action_prefix_length,
        ] = False
        if include_visual:
            mask[self.action_prefix_length :, :] = False
        return mask

    def _decode_tokens(self, action_latent, current_visual_tokens=None):
        if tuple(action_latent.shape[-1:]) != (self.latent_tokens * self.latent_token_dim,):
            raise ValueError(f"Expected action latent [B,1024], got {tuple(action_latent.shape)}.")
        batch_size = action_latent.shape[0]
        latent_tokens = action_latent.reshape(
            batch_size,
            self.latent_tokens,
            self.latent_token_dim,
        )
        latent_tokens = self.latent_projection(latent_tokens)
        action_tokens = (
            self.action_queries.expand(batch_size, -1, -1)
            + self.action_positions
        )
        tokens = torch.cat([latent_tokens, action_tokens], dim=1)
        include_visual = current_visual_tokens is not None
        if include_visual:
            expected_visual = (self.num_visual_tokens, self.visual_dim)
            if tuple(current_visual_tokens.shape[-2:]) != expected_visual:
                raise ValueError(
                    f"Expected visual tokens [B,{expected_visual[0]},{expected_visual[1]}], "
                    f"got {tuple(current_visual_tokens.shape)}."
                )
            visual_tokens = (
                self.visual_projection(current_visual_tokens)
                + self.visual_positions.to(dtype=current_visual_tokens.dtype)
            )
            tokens = torch.cat([tokens, visual_tokens], dim=1)
        attention_mask = self._build_attention_mask(include_visual, tokens.device)
        for block in self.blocks:
            tokens = block(tokens, attention_mask, self.action_prefix_length)
        return tokens

    def decode_actions(self, action_latent):
        tokens = self._decode_tokens(action_latent)
        action_states = tokens[:, self.latent_tokens : self.action_prefix_length]
        return self.action_head(self.action_norm(action_states))

    def forward(self, action_latent, current_visual_tokens):
        tokens = self._decode_tokens(action_latent, current_visual_tokens)
        action_states = tokens[:, self.latent_tokens : self.action_prefix_length]
        visual_states = tokens[:, self.action_prefix_length :]
        actions = self.action_head(self.action_norm(action_states))
        visual_delta = self.visual_head(self.visual_norm(visual_states))
        predicted_future = F.normalize(
            current_visual_tokens + self.visual_delta_scale * visual_delta,
            dim=-1,
        )
        return actions, predicted_future


class SingleViewEDARLite(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = SingleViewEDARLiteEncoder(**kwargs)
        self.decoder = SingleViewEDARLiteDecoder(**kwargs)

    def encode(self, actions, current_visual_tokens, return_token_latent=False):
        return self.encoder(actions, current_visual_tokens, return_token_latent)

    def decode_actions(self, action_latent):
        return self.decoder.decode_actions(action_latent)

    def forward(self, actions, current_visual_tokens):
        action_latent = self.encode(actions, current_visual_tokens)
        decoded_actions, predicted_future = self.decoder(
            action_latent,
            current_visual_tokens,
        )
        return {
            "action_latent": action_latent,
            "decoded_actions": decoded_actions,
            "predicted_future_visual": predicted_future,
        }

    @staticmethod
    def representation_loss(outputs, actions, future_visual_tokens, effect_weight=0.2):
        loss_action = F.mse_loss(outputs["decoded_actions"].float(), actions.float())
        target = F.normalize(future_visual_tokens.detach().float(), dim=-1)
        prediction = F.normalize(outputs["predicted_future_visual"].float(), dim=-1)
        loss_effect = (1.0 - (prediction * target).sum(dim=-1)).mean()
        return loss_action + float(effect_weight) * loss_effect, {
            "loss_action": loss_action.detach(),
            "loss_effect": loss_effect.detach(),
            "visual_cosine": (prediction * target).sum(dim=-1).mean().detach(),
        }


class FrozenDINOFeatureExtractor(nn.Module):
    """Frozen DINOv2/v3 patch extractor with deterministic preprocessing."""

    def __init__(
        self,
        model_name_or_path,
        image_size=256,
        output_grid=8,
        backbone=None,
        image_mean=None,
        image_std=None,
    ):
        super().__init__()
        requested_path = Path(model_name_or_path).expanduser()
        self.model_name_or_path = (
            str(requested_path.resolve())
            if requested_path.is_dir()
            else str(model_name_or_path)
        )
        self.image_size = int(image_size)
        self.output_grid = int(output_grid)
        if backbone is None:
            model_path = Path(self.model_name_or_path).expanduser()
            has_config = (model_path / "config.json").exists()
            has_weights = any(model_path.glob("*.safetensors")) or any(
                model_path.glob("pytorch_model*.bin")
            )
            if not model_path.is_dir() or not has_config or not has_weights:
                raise FileNotFoundError(
                    "Complete DINOv3 weights must be available locally before training: "
                    f"{self.model_name_or_path}"
                )
            from transformers import AutoImageProcessor, AutoModel

            processor = AutoImageProcessor.from_pretrained(
                self.model_name_or_path,
                local_files_only=True,
            )
            backbone = AutoModel.from_pretrained(
                self.model_name_or_path,
                local_files_only=True,
            )
            image_mean = processor.image_mean
            image_std = processor.image_std
        self.backbone = backbone
        config = self.backbone.config
        model_type = str(getattr(config, "model_type", ""))
        if "dinov2" not in model_type.lower() and "dinov3" not in model_type.lower():
            raise ValueError(
                f"EDAR-lite requires DINOv2 or DINOv3, but model_type is '{model_type}'."
            )
        self.visual_dim = int(config.hidden_size)
        patch_size = getattr(config, "patch_size", 16)
        self.patch_size = int(patch_size[0] if isinstance(patch_size, (tuple, list)) else patch_size)
        self.num_register_tokens = int(getattr(config, "num_register_tokens", 0))
        if image_mean is None or image_std is None:
            raise ValueError("DINO official image_mean and image_std are required.")
        self.register_buffer(
            "image_mean",
            torch.tensor(image_mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(image_std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode=True):
        super().train(False)
        self.backbone.eval()
        return self

    def preprocess(self, images):
        return deterministic_dino_preprocess(
            images,
            self.image_size,
            self.image_mean.flatten(),
            self.image_std.flatten(),
        )

    @torch.inference_mode()
    def forward(self, images):
        self.backbone.eval()
        pixel_values = self.preprocess(images)
        try:
            backbone_dtype = next(self.backbone.parameters()).dtype
        except StopIteration:
            backbone_dtype = pixel_values.dtype
        output = self.backbone(pixel_values=pixel_values.to(dtype=backbone_dtype))
        hidden = output.last_hidden_state
        prefix_tokens = 1 + self.num_register_tokens
        patches = hidden[:, prefix_tokens:]
        # ViT patch embedding uses a non-overlapping convolution. For DINOv2-L/14,
        # a 256px crop intentionally produces an 18x18 grid via floor division.
        patch_grid = self.image_size // self.patch_size
        expected_patches = patch_grid * patch_grid
        if patches.shape[1] != expected_patches:
            raise RuntimeError(
                f"Expected {expected_patches} DINO patch tokens, got {patches.shape[1]}."
            )
        patches = patches.reshape(
            patches.shape[0],
            patch_grid,
            patch_grid,
            self.visual_dim,
        ).permute(0, 3, 1, 2)
        patches = F.adaptive_avg_pool2d(patches, (self.output_grid, self.output_grid))
        return patches.flatten(2).transpose(1, 2).detach()

    def metadata(self):
        return {
            "schema_version": 1,
            "backbone": self.model_name_or_path,
            "model_type": str(getattr(self.backbone.config, "model_type", "")),
            "hidden_size": self.visual_dim,
            "patch_size": self.patch_size,
            "num_register_tokens": self.num_register_tokens,
            "image_size": self.image_size,
            "output_grid": self.output_grid,
            "image_mean": self.image_mean.flatten().cpu().tolist(),
            "image_std": self.image_std.flatten().cpu().tolist(),
        }


class EDARFeatureCache:
    METADATA_FILE = "metadata.json"

    def __init__(self, root, expected_metadata, create=False):
        self.root = Path(root).expanduser()
        self.features_dir = self.root / "features"
        self._preloaded_features = None
        self._preloaded_indices = None
        metadata_path = self.root / self.METADATA_FILE
        normalized_metadata = json.loads(json.dumps(expected_metadata, sort_keys=True))
        if create:
            self.features_dir.mkdir(parents=True, exist_ok=True)
            if metadata_path.exists():
                existing = json.loads(metadata_path.read_text(encoding="utf-8"))
                if existing != normalized_metadata:
                    raise ValueError("EDAR cache metadata does not match requested DINO setup.")
            else:
                metadata_path.write_text(
                    json.dumps(normalized_metadata, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
        else:
            if not metadata_path.exists():
                raise FileNotFoundError(f"Missing EDAR cache metadata: {metadata_path}")
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if existing != normalized_metadata:
                raise ValueError("EDAR cache metadata does not match requested DINO setup.")
        self.metadata = normalized_metadata

    @classmethod
    def read_metadata(cls, root):
        metadata_path = Path(root).expanduser() / cls.METADATA_FILE
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing EDAR cache metadata: {metadata_path}")
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    @staticmethod
    def _safe_frame_id(frame_id):
        value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(frame_id)).strip("._")
        if not value:
            raise ValueError("frame_id must contain at least one safe character.")
        return value

    def path_for(self, frame_id):
        return self.features_dir / f"{self._safe_frame_id(frame_id)}.pt"

    def put(self, frame_id, features):
        # A per-frame tensor is commonly a view into a full batch. Clone it so
        # torch.save does not serialize the entire batch storage for every frame.
        features = features.detach().cpu().clone().contiguous()
        if features.ndim != 2:
            raise ValueError("Cached EDAR frame features must have shape [N,D].")
        if features.dtype not in {torch.float16, torch.bfloat16}:
            features = features.to(torch.float16)
        destination = self.path_for(frame_id)
        temporary = destination.with_suffix(".tmp")
        torch.save(features, temporary)
        os.replace(temporary, destination)

    def get(self, frame_id, map_location="cpu"):
        safe_frame_id = self._safe_frame_id(frame_id)
        if self._preloaded_features is not None:
            try:
                return self._preloaded_features[
                    self._preloaded_indices[safe_frame_id]
                ]
            except KeyError as error:
                raise KeyError(f"EDAR feature is not cached: {frame_id}") from error
        path = self.path_for(frame_id)
        if not path.exists():
            raise KeyError(f"EDAR feature is not cached: {frame_id}")
        return torch.load(path, map_location=map_location, weights_only=True)

    def get_many(self, frame_ids):
        safe_ids = [self._safe_frame_id(frame_id) for frame_id in frame_ids]
        if self._preloaded_features is not None:
            try:
                indices = torch.tensor(
                    [self._preloaded_indices[frame_id] for frame_id in safe_ids],
                    dtype=torch.long,
                )
            except KeyError as error:
                raise KeyError(f"EDAR feature is not cached: {error.args[0]}") from error
            return self._preloaded_features.index_select(0, indices)
        return torch.stack([self.get(frame_id) for frame_id in safe_ids])

    def preload(self, log_interval=10000):
        if self._preloaded_features is not None:
            return
        paths = sorted(self.features_dir.glob("*.pt"))
        if not paths:
            raise FileNotFoundError(f"EDAR cache contains no features: {self.features_dir}")
        first = torch.load(paths[0], map_location="cpu", weights_only=True)
        storage = torch.empty(
            (len(paths), *first.shape),
            dtype=first.dtype,
            device="cpu",
        )
        indices = {}
        for index, path in enumerate(paths):
            features = (
                first
                if index == 0
                else torch.load(path, map_location="cpu", weights_only=True)
            )
            if features.shape != first.shape or features.dtype != first.dtype:
                raise ValueError(f"Inconsistent EDAR cache feature: {path}")
            storage[index].copy_(features)
            indices[path.stem] = index
            if log_interval and (index + 1) % int(log_interval) == 0:
                print(f"Preloaded {index + 1}/{len(paths)} EDAR feature frames")
        self._preloaded_features = storage
        self._preloaded_indices = indices
        gib = storage.numel() * storage.element_size() / (1024 ** 3)
        print(f"Preloaded {len(paths)} EDAR frames into RAM ({gib:.2f} GiB)")
