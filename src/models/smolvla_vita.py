import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel

from .edar_lite import SingleViewEDARLiteDecoder, SingleViewEDARLiteEncoder
from .smolvla_expert import SmolVLMWithExpertModel
from .vita_latent_flow import (
    ActionDecoder,
    ActionEncoder,
    VisualDeltaDecoder,
    VitaLatentActionGenerator,
)


def _normalize_action_effect_visual_teacher(teacher_type):
    teacher_type = str(teacher_type).lower()
    aliases = {
        "smolvlm_spatial": "smolvlm",
        "policy_smolvlm": "smolvlm",
    }
    teacher_type = aliases.get(teacher_type, teacher_type)
    if teacher_type not in {"dinov2", "smolvlm"}:
        raise ValueError(
            "Action-effect visual teacher must be 'dinov2' or 'smolvlm'."
        )
    return teacher_type


def _encode_smolvlm_spatial_pair(
    backbone,
    current_pixels,
    future_pixels,
    batch_size,
):
    if current_pixels is None or future_pixels is None:
        raise ValueError(
            "Action-effect loss requires current and future main-view images."
        )

    def flatten_single_view(pixels):
        if pixels.ndim == 5 and pixels.shape[1] == 1:
            pixels = pixels[:, 0]
        if pixels.ndim != 4:
            raise ValueError(
                "SmolVLM teacher pixels must have shape [B,C,H,W] or [B,1,C,H,W]."
            )
        return pixels

    current_pixels = flatten_single_view(current_pixels)
    future_pixels = flatten_single_view(future_pixels)
    if (
        current_pixels.shape[0] != batch_size
        or future_pixels.shape[0] != batch_size
    ):
        raise ValueError("SmolVLM teacher image batch size does not match actions.")

    vision_model = backbone.get_vlm_model().vision_model
    vision_parameter = next(vision_model.parameters())
    pixels = torch.cat([current_pixels, future_pixels], dim=0).to(
        device=vision_parameter.device,
        dtype=vision_parameter.dtype,
    )
    vision_model.eval()
    backbone.get_vlm_model().connector.eval()
    with torch.no_grad():
        spatial_tokens = backbone.embed_image(pixels)
    if spatial_tokens.ndim != 3 or spatial_tokens.shape[1] != 64:
        raise RuntimeError(
            "Expected the SmolVLM connector to produce [B,64,D] spatial tokens, "
            f"got {tuple(spatial_tokens.shape)}."
        )
    return (
        spatial_tokens[:batch_size].detach(),
        spatial_tokens[batch_size:].detach(),
    )


def _sinusoidal_time_embedding(time, dim, min_period=4e-3, max_period=4.0):
    if dim % 2:
        raise ValueError(f"Expert hidden size must be even, got {dim}.")
    fraction = torch.linspace(0.0, 1.0, dim // 2, device=time.device, dtype=torch.float32)
    period = min_period * (max_period / min_period) ** fraction
    phase = (2.0 * math.pi / period)[None, :] * time.float()[:, None]
    return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)


def _make_attention_mask(pad_mask, block_mask):
    """Build the prefix-LM mask used by the native SmolVLA expert."""
    cumulative = torch.cumsum(block_mask, dim=1)
    attention = cumulative[:, None, :] <= cumulative[:, :, None]
    valid = pad_mask[:, None, :] & pad_mask[:, :, None]
    return attention & valid


class LatentTokenExpander(nn.Module):
    """Expand one VITA chunk latent into SmolVLA's per-action suffix tokens."""

    def __init__(self, latent_dim, expert_dim, num_tokens):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.expert_dim = int(expert_dim)
        self.norm = nn.LayerNorm(latent_dim)
        self.proj = nn.Linear(latent_dim, self.num_tokens * self.expert_dim)
        self.token_bias = nn.Parameter(torch.zeros(1, self.num_tokens, self.expert_dim))
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.token_bias, std=0.02)

    def forward(self, latent):
        if latent.ndim != 3 or latent.shape[1] != 1:
            raise ValueError(f"Expected latent [B,1,D], got {tuple(latent.shape)}.")
        tokens = self.proj(self.norm(latent[:, 0]))
        tokens = tokens.view(latent.shape[0], self.num_tokens, self.expert_dim)
        return tokens + self.token_bias.to(dtype=tokens.dtype)


class LatentTokenReducer(nn.Module):
    """Reduce SmolVLA suffix outputs back to one 1024-D flow velocity."""

    def __init__(self, expert_dim, latent_dim, num_tokens):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.expert_dim = int(expert_dim)
        self.norm = nn.LayerNorm(expert_dim)
        self.proj = nn.Linear(self.num_tokens * self.expert_dim, latent_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[1] != self.num_tokens:
            raise ValueError(
                f"Expected {self.num_tokens} expert tokens, got {tuple(tokens.shape)}."
            )
        tokens = self.norm(tokens).flatten(1)
        return self.proj(tokens).unsqueeze(1)


class SmolVLAVitaLatentPolicy(nn.Module):
    """SmolVLA prefix/action expert with a one-token VITA action latent.

    There is deliberately no observation latent, MQ bank, or MQ gate. Images,
    language, and robot state form the native SmolVLA prefix. The action expert
    predicts a noise-to-action velocity in the VITA latent space, and the
    original VITA decoder maps the final [B,1,1024] latent to an action chunk.
    """

    def __init__(
        self,
        lmm_path="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        action_dim=7,
        num_actions=8,
        num_history=8,
        latent_dim=1024,
        action_hidden_dim=1024,
        action_ae_layers=6,
        action_ae_dropout=0.0,
        num_inference_timesteps=6,
        num_vlm_layers=16,
        num_expert_layers=16,
        expert_width_multiplier=0.75,
        self_attn_every_n_layers=2,
        attention_mode="cross_attn",
        load_vlm_weights=True,
        freeze_vlm=True,
        freeze_vision_encoder=True,
        freeze_visual_connector=False,
        state_dim=None,
        enc_recon_weight=0.2,
        consistency_weight=0.0,
        flow_recon_weight=0.0,
        num_sampling_steps=None,
        rollout_gradient_checkpointing=True,
        action_effect_enabled=True,
        action_effect_visual_teacher_type="dinov2",
        action_effect_visual_teacher_path=None,
        load_action_effect_visual_teacher=True,
        action_effect_visual_ae_weight=0.01,
        action_effect_decoder_layers=2,
        action_representation_type="legacy",
        edar_stage_a_checkpoint=None,
        edar_train_decoder=False,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.num_actions = int(num_actions)
        self.num_history = int(num_history)
        self.latent_dim = int(latent_dim)
        self.num_inference_timesteps = int(num_inference_timesteps)
        self.enc_recon_weight = float(enc_recon_weight)
        self.consistency_weight = float(consistency_weight)
        self.flow_recon_weight = float(flow_recon_weight)
        self.num_sampling_steps = int(
            num_sampling_steps or self.num_inference_timesteps
        )
        self.rollout_gradient_checkpointing = bool(
            rollout_gradient_checkpointing
        )
        self.action_effect_visual_ae_weight = float(action_effect_visual_ae_weight)
        self.action_effect_enabled = bool(action_effect_enabled)
        self.action_representation_type = str(action_representation_type).lower()
        self.use_edar_lite = self.action_representation_type == "single_view_edar_lite"
        if self.action_representation_type not in {"legacy", "single_view_edar_lite"}:
            raise ValueError(
                "SmolVLA action representation must be 'legacy' or "
                "'single_view_edar_lite'."
            )
        self.edar_stage_a_checkpoint = (
            os.path.expanduser(str(edar_stage_a_checkpoint))
            if edar_stage_a_checkpoint
            else None
        )
        self.edar_train_decoder = bool(edar_train_decoder)
        self.edar_dino_metadata = None
        self.action_effect_visual_teacher_type = (
            _normalize_action_effect_visual_teacher(
                action_effect_visual_teacher_type
            )
        )
        self.use_proprio_input_vlm = True
        self.use_action_input_policy = False
        self.use_vita_latent_flow = False
        self.vita_use_proprio = True
        self.model_family = "smolvlm"

        if self.latent_dim != 1024:
            raise ValueError("The first SmolVLA-VITA experiment fixes latent_dim=1024.")
        if self.num_inference_timesteps <= 0:
            raise ValueError("num_inference_timesteps must be positive.")
        if self.num_sampling_steps <= 0:
            raise ValueError("num_sampling_steps must be positive.")
        if self.consistency_weight < 0 or self.flow_recon_weight < 0:
            raise ValueError("VITA rollout loss weights must be non-negative.")

        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=lmm_path,
            load_vlm_weights=bool(load_vlm_weights),
            train_expert_only=bool(freeze_vlm),
            freeze_vision_encoder=bool(freeze_vision_encoder),
            freeze_visual_connector=bool(freeze_visual_connector),
            attention_mode=attention_mode,
            num_expert_layers=int(num_expert_layers),
            num_vlm_layers=int(num_vlm_layers),
            self_attn_every_n_layers=int(self_attn_every_n_layers),
            expert_width_multiplier=float(expert_width_multiplier),
        )
        self.hidden_size = int(self.vlm_with_expert.config.text_config.hidden_size)
        self.expert_hidden_size = int(self.vlm_with_expert.expert_hidden_size)

        state_dim = int(state_dim or max(1, self.num_history) * self.action_dim)
        self.state_dim = state_dim
        self.state_proj = nn.Linear(state_dim, self.hidden_size)

        if self.use_edar_lite:
            if not self.edar_stage_a_checkpoint:
                raise ValueError(
                    "SmolVLA EDAR training requires vita_edar_stage_a_checkpoint."
                )
            checkpoint = torch.load(
                self.edar_stage_a_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            if checkpoint.get("schema") != "single_view_edar_lite_stage_a_v1":
                raise ValueError("EDAR Stage A checkpoint has an incompatible schema.")
            architecture = checkpoint["config"]["model"]["action_representation"]
            if int(architecture.get("action_horizon", 8)) != self.num_actions:
                raise ValueError("EDAR action horizon does not match SmolVLA.")
            self.edar_dino_metadata = dict(checkpoint["dino_metadata"])
            edar_kwargs = {
                "action_dim": self.action_dim,
                "action_horizon": self.num_actions,
                "visual_dim": int(self.edar_dino_metadata["hidden_size"]),
                "visual_grid": int(architecture.get("visual_grid", 8)),
                "model_dim": int(architecture.get("model_dim", 512)),
                "latent_tokens": int(architecture.get("latent_tokens", 4)),
                "latent_token_dim": int(architecture.get("latent_token_dim", 256)),
                "num_layers": int(architecture.get("layers", 4)),
                "num_heads": int(architecture.get("heads", 8)),
                "mlp_ratio": float(architecture.get("mlp_ratio", 4.0)),
            }
            self.action_encoder = SingleViewEDARLiteEncoder(**edar_kwargs)
            self.action_decoder = SingleViewEDARLiteDecoder(**edar_kwargs)
            self.action_encoder.load_state_dict(
                checkpoint["encoder_state_dict"], strict=True
            )
            self.action_decoder.load_state_dict(
                checkpoint["decoder_state_dict"], strict=True
            )
            self.action_encoder.requires_grad_(False).eval()
            self.action_decoder.requires_grad_(self.edar_train_decoder)
            self.action_decoder.train(self.edar_train_decoder)
            del checkpoint
        else:
            self.action_encoder = ActionEncoder(
                action_dim=self.action_dim,
                horizon=self.num_actions,
                latent_dim=self.latent_dim,
                hidden_dim=int(action_hidden_dim),
                num_layers=int(action_ae_layers),
                dropout=float(action_ae_dropout),
            )
            self.action_decoder = ActionDecoder(
                action_dim=self.action_dim,
                horizon=self.num_actions,
                latent_dim=self.latent_dim,
                hidden_dim=int(action_hidden_dim),
                num_layers=int(action_ae_layers),
                dropout=float(action_ae_dropout),
            )

        self.latent_expander = LatentTokenExpander(
            self.latent_dim,
            self.expert_hidden_size,
            self.num_actions,
        )
        self.action_time_mlp_in = nn.Linear(
            self.expert_hidden_size * 2,
            self.expert_hidden_size,
        )
        self.action_time_mlp_out = nn.Linear(
            self.expert_hidden_size,
            self.expert_hidden_size,
        )
        self.latent_reducer = LatentTokenReducer(
            self.expert_hidden_size,
            self.latent_dim,
            self.num_actions,
        )

        self.action_effect_visual_teacher = None
        self.visual_delta_decoder = None
        if self.use_edar_lite:
            if self.action_effect_visual_teacher_type != "dinov2":
                raise ValueError("SmolVLA EDAR currently requires a DINOv2 teacher.")
            if load_action_effect_visual_teacher:
                if not action_effect_visual_teacher_path:
                    raise ValueError("SmolVLA EDAR training requires the DINOv2 path.")
                configured_teacher = os.path.realpath(
                    os.path.expanduser(action_effect_visual_teacher_path)
                )
                checkpoint_teacher = os.path.realpath(
                    os.path.expanduser(self.edar_dino_metadata["backbone"])
                )
                if configured_teacher != checkpoint_teacher:
                    raise ValueError(
                        "Configured DINO path does not match EDAR Stage A metadata."
                    )
                self.action_effect_visual_teacher = AutoModel.from_pretrained(
                    configured_teacher,
                    local_files_only=True,
                    dtype=torch.bfloat16,
                )
                self.action_effect_visual_teacher.requires_grad_(False).eval()
        elif self.action_effect_enabled:
            visual_dim = self.latent_dim
            if self.action_effect_visual_teacher_type == "smolvlm":
                visual_dim = self.hidden_size
                visual_modules = (
                    self.vlm_with_expert.get_vlm_model().vision_model,
                    self.vlm_with_expert.get_vlm_model().connector,
                )
                if any(
                    parameter.requires_grad
                    for module in visual_modules
                    for parameter in module.parameters()
                ):
                    raise ValueError(
                        "The shared SmolVLM spatial teacher requires both the vision "
                        "encoder and visual connector to be frozen."
                    )
            elif load_action_effect_visual_teacher:
                if not action_effect_visual_teacher_path:
                    raise ValueError(
                        "Action-effect training requires a local DINOv2 teacher path."
                    )
                self.action_effect_visual_teacher = AutoModel.from_pretrained(
                    action_effect_visual_teacher_path,
                    local_files_only=True,
                    dtype=torch.bfloat16,
                )
                teacher_dim = int(self.action_effect_visual_teacher.config.hidden_size)
                if teacher_dim != self.latent_dim:
                    raise ValueError(
                        f"DINOv2 hidden size {teacher_dim} must equal latent_dim {self.latent_dim}."
                    )
                self.action_effect_visual_teacher.requires_grad_(False)
                self.action_effect_visual_teacher.eval()
            self.visual_delta_decoder = VisualDeltaDecoder(
                latent_dim=self.latent_dim,
                hidden_dim=int(action_hidden_dim),
                num_layers=int(action_effect_decoder_layers),
                dropout=float(action_ae_dropout),
                visual_dim=visual_dim,
            )

    @property
    def lmm(self):
        return self.vlm_with_expert.vlm

    @property
    def processor(self):
        return self.vlm_with_expert.processor

    def train(self, mode=True):
        super().train(mode)
        self.vlm_with_expert.train(mode)
        if self.use_edar_lite:
            self.action_encoder.eval()
            self.action_decoder.eval()
        if self.action_effect_visual_teacher is not None:
            self.action_effect_visual_teacher.eval()
        if self.use_edar_lite:
            self.action_encoder.eval()
            self.action_decoder.train(mode and self.edar_train_decoder)
        return self

    def _decode_actions(self, action_latent):
        if self.use_edar_lite:
            return self.action_decoder.decode_actions(action_latent)
        return self.action_decoder(action_latent)

    def _encode_edar_current_visual(self, current_pixels, batch_size):
        teacher = self.action_effect_visual_teacher
        if teacher is None:
            raise RuntimeError(
                "The frozen DINOv2 teacher is required to construct EDAR targets."
            )
        if current_pixels is None or current_pixels.shape[0] != batch_size:
            raise ValueError("EDAR current-image batch must match the action batch.")
        teacher_parameter = next(teacher.parameters())
        pixels = current_pixels.to(
            device=teacher_parameter.device,
            dtype=teacher_parameter.dtype,
        )
        teacher.eval()
        with torch.no_grad():
            output = teacher(pixel_values=pixels, return_dict=True)
            prefix_tokens = 1 + int(
                self.edar_dino_metadata.get("num_register_tokens", 0)
            )
            patches = output.last_hidden_state[:, prefix_tokens:]
            image_size = int(self.edar_dino_metadata.get("image_size", 256))
            patch_size = int(self.edar_dino_metadata.get("patch_size", 14))
            patch_grid = image_size // patch_size
            expected_tokens = patch_grid * patch_grid
            if patches.shape[1] != expected_tokens:
                raise RuntimeError(
                    f"Expected {expected_tokens} DINO patches, got {patches.shape[1]}."
                )
            patches = patches.reshape(
                batch_size,
                patch_grid,
                patch_grid,
                patches.shape[-1],
            ).permute(0, 3, 1, 2)
            output_grid = int(self.edar_dino_metadata.get("output_grid", 8))
            patches = F.adaptive_avg_pool2d(
                patches.float(),
                (output_grid, output_grid),
            )
            patches = patches.flatten(2).transpose(1, 2)
        return patches.to(dtype=next(self.action_encoder.parameters()).dtype).detach()

    def _prepare_state(self, proprioception, batch_size, device, dtype):
        if proprioception is None:
            return torch.zeros(batch_size, self.state_dim, device=device, dtype=dtype)
        state = proprioception.flatten(1)
        if state.shape[1] != self.state_dim:
            raise ValueError(
                f"Expected flattened robot state dim {self.state_dim}, got {state.shape[1]}."
            )
        return state.to(device=device, dtype=dtype)

    def _embed_prefix(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        pixel_attention_mask,
        proprioception,
    ):
        if input_ids is None or attention_mask is None or pixel_values is None:
            raise ValueError("SmolVLA prefix requires text, attention mask, and images.")
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(1)
        if pixel_values.ndim != 5:
            raise ValueError(
                f"Expected pixel_values [B,V,C,H,W], got {tuple(pixel_values.shape)}."
            )

        embeddings = []
        pad_masks = []
        block_masks = []
        batch_size, num_views = pixel_values.shape[:2]
        vision_model = self.vlm_with_expert.get_vlm_model().vision_model
        for view_index in range(num_views):
            image = pixel_values[:, view_index].to(dtype=vision_model.dtype)
            patch_mask = None
            if pixel_attention_mask is not None:
                patch_mask = pixel_attention_mask[:, view_index]
                if patch_mask.shape[-2:] == image.shape[-2:]:
                    patch_size = int(vision_model.patch_size)
                    patch_mask = patch_mask.unfold(1, patch_size, patch_size).unfold(
                        2, patch_size, patch_size
                    )
                    patch_mask = patch_mask.any(dim=-1).any(dim=-1)
            image_tokens = self.vlm_with_expert.embed_image(image, patch_mask)
            image_tokens = image_tokens * math.sqrt(image_tokens.shape[-1])
            embeddings.append(image_tokens)
            pad_masks.append(
                torch.ones(
                    batch_size,
                    image_tokens.shape[1],
                    device=image_tokens.device,
                    dtype=torch.bool,
                )
            )
            block_masks.append(
                torch.zeros(
                    batch_size,
                    image_tokens.shape[1],
                    device=image_tokens.device,
                    dtype=torch.bool,
                )
            )

        language = self.vlm_with_expert.embed_language_tokens(input_ids)
        language = language * math.sqrt(language.shape[-1])
        embeddings.append(language)
        pad_masks.append(attention_mask.bool())
        block_masks.append(torch.zeros_like(attention_mask, dtype=torch.bool))

        state = self._prepare_state(
            proprioception,
            batch_size,
            language.device,
            self.state_proj.weight.dtype,
        )
        state_token = self.state_proj(state).unsqueeze(1)
        embeddings.append(state_token)
        pad_masks.append(
            torch.ones(batch_size, 1, device=state_token.device, dtype=torch.bool)
        )
        block_masks.append(
            torch.ones(batch_size, 1, device=state_token.device, dtype=torch.bool)
        )

        prefix = torch.cat(embeddings, dim=1)
        pad_mask = torch.cat(pad_masks, dim=1)
        block_mask = torch.cat(block_masks, dim=1)
        return prefix, pad_mask, block_mask

    def _embed_suffix(self, noisy_latent, timestep):
        latent_tokens = self.latent_expander(noisy_latent)
        time_tokens = _sinusoidal_time_embedding(
            timestep,
            self.expert_hidden_size,
        ).to(dtype=latent_tokens.dtype)
        time_tokens = time_tokens[:, None, :].expand_as(latent_tokens)
        suffix = torch.cat([latent_tokens, time_tokens], dim=-1)
        suffix = self.action_time_mlp_in(suffix)
        suffix = F.silu(suffix)
        suffix = self.action_time_mlp_out(suffix)
        pad_mask = torch.ones(
            suffix.shape[:2],
            device=suffix.device,
            dtype=torch.bool,
        )
        block_mask = torch.ones_like(pad_mask)
        return suffix, pad_mask, block_mask

    def _predict_velocity_train(
        self,
        noisy_latent,
        timestep,
        prefix,
        prefix_pad_mask,
        prefix_block_mask,
    ):
        suffix, suffix_pad_mask, suffix_block_mask = self._embed_suffix(
            noisy_latent,
            timestep,
        )
        pad_mask = torch.cat([prefix_pad_mask, suffix_pad_mask], dim=1)
        block_mask = torch.cat([prefix_block_mask, suffix_block_mask], dim=1)
        attention_mask = _make_attention_mask(pad_mask, block_mask)
        position_ids = torch.cumsum(pad_mask, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix, suffix],
            use_cache=False,
            fill_kv_cache=False,
        )
        return self.latent_reducer(suffix_out[:, -self.num_actions :]).float()

    def _encode_action_effect_pair(
        self,
        current_pixels,
        future_pixels,
        batch_size,
    ):
        if self.action_effect_visual_teacher_type == "smolvlm":
            return _encode_smolvlm_spatial_pair(
                self.vlm_with_expert,
                current_pixels,
                future_pixels,
                batch_size,
            )
        teacher = self.action_effect_visual_teacher
        if teacher is None:
            raise RuntimeError("DINOv2 teacher is unavailable.")
        if current_pixels is None or future_pixels is None:
            raise ValueError("Action-effect loss requires current and future main-view images.")
        if current_pixels.shape[0] != batch_size or future_pixels.shape[0] != batch_size:
            raise ValueError("DINOv2 image batch must match the action batch.")
        teacher_param = next(teacher.parameters())
        pixels = torch.cat([current_pixels, future_pixels], dim=0).to(
            device=teacher_param.device,
            dtype=teacher_param.dtype,
        )
        teacher.eval()
        with torch.no_grad():
            cls = teacher(pixel_values=pixels, return_dict=True).last_hidden_state[:, :1]
        return cls[:batch_size].detach(), cls[batch_size:].detach()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        pixel_attention_mask=None,
        actions=None,
        proprioception=None,
        return_loss_dict=False,
        action_effect_teacher_current_pixel_values=None,
        action_effect_teacher_future_pixel_values=None,
        **_,
    ):
        if actions is None:
            raise ValueError("Training requires actions.")
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        action_dtype = next(self.action_encoder.parameters()).dtype
        actions = actions.to(dtype=action_dtype)
        prefix, prefix_pad_mask, prefix_block_mask = self._embed_prefix(
            input_ids,
            attention_mask,
            pixel_values,
            pixel_attention_mask,
            proprioception,
        )

        if self.use_edar_lite:
            current_visual = self._encode_edar_current_visual(
                action_effect_teacher_current_pixel_values,
                actions.shape[0],
            )
            action_latent = self.action_encoder(actions, current_visual).unsqueeze(1)
        else:
            action_latent = self.action_encoder(actions).unsqueeze(1)
        noise = torch.randn_like(action_latent)
        beta = torch.distributions.Beta(1.5, 1.0)
        timestep = beta.sample((actions.shape[0],)).to(actions.device, dtype=torch.float32)
        timestep = timestep * 0.999 + 0.001
        t = timestep.to(dtype=action_latent.dtype).view(-1, 1, 1)
        noisy_latent = t * noise + (1.0 - t) * action_latent
        target_velocity = noise - action_latent
        need_rollout = self.consistency_weight > 0 or self.flow_recon_weight > 0
        prefix_cache = None
        if need_rollout:
            prefix_cache = self._cache_prefix(
                prefix,
                prefix_pad_mask,
                prefix_block_mask,
            )
            predicted_velocity = self._predict_velocity_cached_train(
                noisy_latent,
                timestep,
                prefix_pad_mask,
                prefix_cache,
            )
        else:
            predicted_velocity = self._predict_velocity_train(
                noisy_latent,
                timestep,
                prefix,
                prefix_pad_mask,
                prefix_block_mask,
            )
        loss_flow = F.mse_loss(predicted_velocity, target_velocity.float())

        reconstructed_actions = self._decode_actions(action_latent[:, 0])
        loss_action_recon = F.l1_loss(reconstructed_actions.float(), actions.float())
        loss = loss_flow + self.enc_recon_weight * loss_action_recon

        loss_consistency = None
        loss_flow_recon = None
        if need_rollout:
            predicted_action_latent = self._sample_latent_train(
                noise,
                prefix_pad_mask,
                prefix_cache,
            )
            loss_consistency = F.mse_loss(
                predicted_action_latent.float(),
                action_latent.float(),
            )
            loss = loss + self.consistency_weight * loss_consistency

            predicted_actions = self._decode_actions(predicted_action_latent[:, 0])
            loss_flow_recon = F.l1_loss(
                predicted_actions.float(),
                actions.float(),
            )
            loss = loss + self.flow_recon_weight * loss_flow_recon

        loss_visual = None
        if self.action_effect_enabled and not self.use_edar_lite:
            current_visual, future_visual = self._encode_action_effect_pair(
                action_effect_teacher_current_pixel_values,
                action_effect_teacher_future_pixel_values,
                actions.shape[0],
            )
            visual_target = future_visual - current_visual
            visual_prediction = self.visual_delta_decoder(
                action_latent[:, 0],
                current_visual,
            )
            loss_visual = F.smooth_l1_loss(
                visual_prediction.float(),
                visual_target.float(),
            )
            loss = loss + self.action_effect_visual_ae_weight * loss_visual

        if not return_loss_dict:
            return loss
        output = {
            "loss": loss,
            "loss_latent_flow": loss_flow.detach(),
            "loss_action_recon": loss_action_recon.detach(),
            "loss_action_recon_weighted": (
                self.enc_recon_weight * loss_action_recon
            ).detach(),
        }
        if loss_consistency is not None:
            output["loss_latent_consistency"] = loss_consistency.detach()
            output["loss_latent_consistency_weighted"] = (
                self.consistency_weight * loss_consistency
            ).detach()
        if loss_flow_recon is not None:
            output["loss_flow_action_recon"] = loss_flow_recon.detach()
            output["loss_flow_action_recon_weighted"] = (
                self.flow_recon_weight * loss_flow_recon
            ).detach()
        if loss_visual is not None:
            output["loss_action_effect_visual_ae"] = loss_visual.detach()
            output["loss_action_effect_visual_ae_weighted"] = (
                self.action_effect_visual_ae_weight * loss_visual
            ).detach()
        return output

    def _cache_prefix(self, prefix, pad_mask, block_mask):
        attention_mask = _make_attention_mask(pad_mask, block_mask)
        position_ids = torch.cumsum(pad_mask, dim=1) - 1
        _, cache = self.vlm_with_expert(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        return cache

    def _predict_velocity_cached(self, latent, timestep, prefix_pad_mask, cache):
        suffix, suffix_pad_mask, suffix_block_mask = self._embed_suffix(latent, timestep)
        batch_size, suffix_len = suffix_pad_mask.shape
        prefix_len = prefix_pad_mask.shape[1]
        prefix_attention = prefix_pad_mask[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )
        suffix_attention = _make_attention_mask(suffix_pad_mask, suffix_block_mask)
        attention_mask = torch.cat([prefix_attention, suffix_attention], dim=-1)
        prefix_offsets = prefix_pad_mask.sum(dim=-1, keepdim=True)
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_mask, dim=1) - 1
        outputs, _ = self.vlm_with_expert(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            inputs_embeds=[None, suffix],
            use_cache=True,
            fill_kv_cache=False,
        )
        return self.latent_reducer(outputs[1][:, -self.num_actions :]).float()

    def _predict_velocity_cached_train(
        self,
        latent,
        timestep,
        prefix_pad_mask,
        cache,
    ):
        if not self.rollout_gradient_checkpointing:
            return self._predict_velocity_cached(
                latent,
                timestep,
                prefix_pad_mask,
                cache,
            )

        def predict(latent_input, timestep_input):
            return self._predict_velocity_cached(
                latent_input,
                timestep_input,
                prefix_pad_mask,
                cache,
            )

        return checkpoint(
            predict,
            latent,
            timestep,
            use_reentrant=False,
        )

    def _sample_latent_train(self, noise, prefix_pad_mask, cache):
        latent = noise
        step_size = -1.0 / float(self.num_sampling_steps)
        for step in range(self.num_sampling_steps):
            timestep = torch.full(
                (noise.shape[0],),
                1.0 + step * step_size,
                device=noise.device,
                dtype=torch.float32,
            )
            velocity = self._predict_velocity_cached_train(
                latent,
                timestep,
                prefix_pad_mask,
                cache,
            )
            latent = latent + step_size * velocity.to(dtype=latent.dtype)
        return latent

    @torch.no_grad()
    def predict_action(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        pixel_attention_mask=None,
        proprioception=None,
        **_,
    ):
        prefix, prefix_pad_mask, prefix_block_mask = self._embed_prefix(
            input_ids,
            attention_mask,
            pixel_values,
            pixel_attention_mask,
            proprioception,
        )
        cache = self._cache_prefix(prefix, prefix_pad_mask, prefix_block_mask)
        latent = torch.randn(
            prefix.shape[0],
            1,
            self.latent_dim,
            device=prefix.device,
            dtype=next(self.action_decoder.parameters()).dtype,
        )
        step_size = -1.0 / float(self.num_inference_timesteps)
        for step in range(self.num_inference_timesteps):
            timestep = torch.full(
                (prefix.shape[0],),
                1.0 + step * step_size,
                device=prefix.device,
                dtype=torch.float32,
            )
            velocity = self._predict_velocity_cached(
                latent,
                timestep,
                prefix_pad_mask,
                cache,
            )
            latent = latent + step_size * velocity.to(dtype=latent.dtype)
        return self._decode_actions(latent[:, 0]).to(dtype=self.lmm.dtype)


class SmolVLAMQ54VitaPolicy(nn.Module):
    """SmolVLM prefix encoder with the v3.2 HierMQ54 VITA action expert."""

    def __init__(
        self,
        lmm_path,
        action_dim=7,
        num_actions=8,
        num_history=8,
        latent_dim=1024,
        hidden_dim=1024,
        action_ae_layers=6,
        flow_layers=8,
        flow_mlp_ratio=4.0,
        dropout=0.0,
        num_inference_timesteps=6,
        num_sampling_steps=6,
        num_vlm_layers=12,
        load_vlm_weights=True,
        freeze_vlm=False,
        freeze_vision_encoder=False,
        freeze_visual_connector=False,
        layer_local_queries_per_layer_list=(8, 4, 4, 4, 4, 4),
        layer_local_token_source_modes=(
            "visual",
            "text",
            "all",
            "all",
            "all",
            "all",
        ),
        hidden_layer_indices=(0, 0, 1, 4, 8, 12),
        global_queries=24,
        state_num_tokens=2,
        state_token_dropout=0.1,
        flow_cross_attention_heads=8,
        fixed_flow_cross_attention_scale=0.1,
        mq_token_gate_lambda=0.8,
        enc_recon_weight=0.2,
        consistency_weight=1.0,
        flow_recon_weight=0.2,
        log_diversity_loss=True,
        action_effect_enabled=True,
        action_effect_visual_teacher_type="dinov2",
        action_effect_visual_teacher_path=None,
        load_action_effect_visual_teacher=True,
        action_effect_visual_ae_weight=0.01,
        action_effect_decoder_layers=2,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.num_actions = int(num_actions)
        self.num_history = int(num_history)
        self.latent_dim = int(latent_dim)
        self.state_dim = self.num_history * self.action_dim
        self.num_inference_timesteps = int(num_inference_timesteps)
        self.enc_recon_weight = float(enc_recon_weight)
        self.consistency_weight = float(consistency_weight)
        self.flow_recon_weight = float(flow_recon_weight)
        self.log_diversity_loss = bool(log_diversity_loss)
        self.action_effect_enabled = bool(action_effect_enabled)
        self.action_effect_visual_ae_weight = float(action_effect_visual_ae_weight)
        self.action_effect_visual_teacher_type = (
            _normalize_action_effect_visual_teacher(
                action_effect_visual_teacher_type
            )
        )

        self.use_proprio_input_vlm = False
        self.use_action_input_policy = False
        self.use_vita_latent_flow = True
        self.vita_use_proprio = True
        self.model_family = "smolvlm"

        if self.latent_dim != 1024:
            raise ValueError("The v3.2-compatible MQ54 expert requires latent_dim=1024.")
        if int(num_vlm_layers) < max(int(i) for i in hidden_layer_indices):
            raise ValueError(
                "num_vlm_layers must cover every requested MQ hidden-state index."
            )
        local_query_count = sum(int(v) for v in layer_local_queries_per_layer_list)
        if local_query_count != 28 or int(global_queries) != 24:
            raise ValueError(
                "The MQ54 v3.2 layout requires 28 local and 24 global queries."
            )

        self.vlm_backbone = SmolVLMWithExpertModel(
            model_id=lmm_path,
            load_vlm_weights=bool(load_vlm_weights),
            train_expert_only=bool(freeze_vlm),
            freeze_vision_encoder=bool(freeze_vision_encoder),
            freeze_visual_connector=bool(freeze_visual_connector),
            attention_mode="self_attn",
            num_vlm_layers=int(num_vlm_layers),
            build_expert=False,
        )
        self.hidden_size = int(self.vlm_backbone.config.text_config.hidden_size)
        action_effect_visual_dim = self.latent_dim
        if self.action_effect_visual_teacher_type == "smolvlm":
            action_effect_visual_dim = self.hidden_size
            visual_modules = (
                self.vlm_backbone.get_vlm_model().vision_model,
                self.vlm_backbone.get_vlm_model().connector,
            )
            if any(
                parameter.requires_grad
                for module in visual_modules
                for parameter in module.parameters()
            ):
                raise ValueError(
                    "The shared SmolVLM spatial teacher requires both the vision "
                    "encoder and visual connector to be frozen."
                )

        self.vita_action_generator = VitaLatentActionGenerator(
            action_dim=self.action_dim,
            horizon=self.num_actions,
            vlm_hidden_dim=self.hidden_size,
            num_queries=64,
            latent_dim=self.latent_dim,
            hidden_dim=int(hidden_dim),
            action_ae_layers=int(action_ae_layers),
            flow_layers=int(flow_layers),
            flow_mlp_ratio=float(flow_mlp_ratio),
            dropout=float(dropout),
            num_sampling_steps=int(num_sampling_steps),
            proprio_dim=self.state_dim,
            condition_source="hidden_layer",
            hidden_layer_index=int(hidden_layer_indices[-1]),
            hidden_pooling="attention",
            pooling_num_heads=8,
            pooling_num_queries=52,
            gated_weighted_pooling=True,
            hierarchical_query_pooling=True,
            layer_local_queries_per_layer_list=[
                int(v) for v in layer_local_queries_per_layer_list
            ],
            layer_local_token_source_modes=list(layer_local_token_source_modes),
            cross_layer_queries=0,
            global_queries=int(global_queries),
            blockwise_layer_conditioning=False,
            blockwise_hidden_layer_indices=[int(v) for v in hidden_layer_indices],
            state_token_conditioning=True,
            state_num_tokens=int(state_num_tokens),
            state_token_dropout=float(state_token_dropout),
            flow_cross_attention=True,
            flow_cross_attention_heads=int(flow_cross_attention_heads),
            gated_flow_cross_attention=True,
            mq_token_value_gating=True,
            mq_token_gate_lambda=float(mq_token_gate_lambda),
            mq_token_gate_use_action_latent=True,
            fixed_flow_cross_attention_scale=float(
                fixed_flow_cross_attention_scale
            ),
            action_effect_enabled=self.action_effect_enabled,
            action_effect_condition_action_encoder=False,
            action_effect_visual_queries=1,
            action_effect_visual_tokens_prepooled=True,
            action_effect_num_heads=8,
            action_effect_encoder_layers=2,
            action_effect_decoder_layers=int(action_effect_decoder_layers),
            action_effect_visual_dim=action_effect_visual_dim,
        )

        self.action_effect_visual_teacher = None
        if (
            self.action_effect_enabled
            and self.action_effect_visual_teacher_type == "dinov2"
            and load_action_effect_visual_teacher
        ):
            if not action_effect_visual_teacher_path:
                raise ValueError(
                    "DINOv2 action-effect training requires a local teacher path."
                )
            self.action_effect_visual_teacher = AutoModel.from_pretrained(
                action_effect_visual_teacher_path,
                local_files_only=True,
                dtype=torch.bfloat16,
            )
            teacher_dim = int(self.action_effect_visual_teacher.config.hidden_size)
            if teacher_dim != self.latent_dim:
                raise ValueError(
                    f"DINOv2 hidden size {teacher_dim} must equal latent_dim "
                    f"{self.latent_dim}."
                )
            self.action_effect_visual_teacher.requires_grad_(False)
            self.action_effect_visual_teacher.eval()

    @classmethod
    def from_config(
        cls,
        model_config,
        data_config,
        load_action_effect_visual_teacher=True,
        num_inference_timesteps=None,
    ):
        return cls(
            lmm_path=model_config["lmm_path"],
            action_dim=model_config["action_dim"],
            num_actions=data_config["future_len"],
            num_history=data_config["history_len"],
            latent_dim=model_config.get("vita_latent_dim", 1024),
            hidden_dim=model_config.get("vita_hidden_dim", 1024),
            action_ae_layers=model_config.get("vita_action_ae_layers", 6),
            flow_layers=model_config.get("vita_flow_layers", 8),
            flow_mlp_ratio=model_config.get("vita_flow_mlp_ratio", 4.0),
            dropout=model_config.get("vita_dropout", 0.0),
            num_inference_timesteps=(
                num_inference_timesteps
                if num_inference_timesteps is not None
                else model_config.get("num_inference_timesteps", 6)
            ),
            num_sampling_steps=model_config.get("vita_num_sampling_steps", 6),
            num_vlm_layers=model_config.get("smolvla_num_vlm_layers", 12),
            load_vlm_weights=model_config.get("smolvla_load_vlm_weights", True),
            freeze_vlm=model_config.get("smolvla_freeze_vlm", False),
            freeze_vision_encoder=model_config.get(
                "smolvla_freeze_vision_encoder", False
            ),
            freeze_visual_connector=model_config.get(
                "smolvla_freeze_visual_connector", False
            ),
            layer_local_queries_per_layer_list=model_config.get(
                "vita_layer_local_queries_per_layer_list",
                [8, 4, 4, 4, 4, 4],
            ),
            layer_local_token_source_modes=model_config.get(
                "vita_layer_local_token_source_modes",
                ["visual", "text", "all", "all", "all", "all"],
            ),
            hidden_layer_indices=model_config.get(
                "vita_blockwise_hidden_layer_indices",
                [0, 0, 1, 4, 8, 12],
            ),
            global_queries=model_config.get("vita_global_queries", 24),
            state_num_tokens=model_config.get("vita_state_num_tokens", 2),
            state_token_dropout=model_config.get(
                "vita_state_token_dropout", 0.1
            ),
            flow_cross_attention_heads=model_config.get(
                "vita_flow_cross_attention_heads", 8
            ),
            fixed_flow_cross_attention_scale=model_config.get(
                "vita_flow_cross_attention_fixed_scale", 0.1
            ),
            mq_token_gate_lambda=model_config.get(
                "vita_mq_token_gate_lambda", 0.8
            ),
            enc_recon_weight=model_config.get("vita_enc_recon_weight", 0.2),
            consistency_weight=model_config.get(
                "vita_consistency_weight", 1.0
            ),
            flow_recon_weight=model_config.get("vita_flow_recon_weight", 0.2),
            log_diversity_loss=model_config.get(
                "vita_log_diversity_loss", True
            ),
            action_effect_enabled=model_config.get(
                "vita_action_effect_enabled", True
            ),
            action_effect_visual_teacher_type=model_config.get(
                "vita_action_effect_visual_teacher", "dinov2"
            ),
            action_effect_visual_teacher_path=model_config.get(
                "vita_action_effect_visual_teacher_path"
            ),
            load_action_effect_visual_teacher=load_action_effect_visual_teacher,
            action_effect_visual_ae_weight=model_config.get(
                "vita_action_effect_visual_ae_weight", 0.01
            ),
            action_effect_decoder_layers=model_config.get(
                "vita_action_effect_decoder_layers", 2
            ),
        )

    @property
    def lmm(self):
        return self.vlm_backbone.vlm

    @property
    def processor(self):
        return self.vlm_backbone.processor

    def train(self, mode=True):
        super().train(mode)
        self.vlm_backbone.train(mode)
        if self.action_effect_visual_teacher is not None:
            self.action_effect_visual_teacher.eval()
        return self

    def _prepare_proprioception(self, proprioception, batch_size, device, dtype):
        if proprioception is None:
            return torch.zeros(
                batch_size,
                self.num_history,
                self.action_dim,
                device=device,
                dtype=dtype,
            )
        state = proprioception
        if state.ndim == 2 and state.shape[1] == self.state_dim:
            state = state.view(batch_size, self.num_history, self.action_dim)
        if tuple(state.shape[1:]) != (self.num_history, self.action_dim):
            raise ValueError(
                "Expected robot state "
                f"[B,{self.num_history},{self.action_dim}], got {tuple(state.shape)}."
            )
        return state.to(device=device, dtype=dtype)

    def _embed_prefix_with_sources(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        pixel_attention_mask,
    ):
        if input_ids is None or attention_mask is None or pixel_values is None:
            raise ValueError("MQ54 extraction requires text, attention mask, and images.")
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(1)
        if pixel_values.ndim != 5:
            raise ValueError(
                f"Expected pixel_values [B,V,C,H,W], got {tuple(pixel_values.shape)}."
            )

        embeddings = []
        pad_masks = []
        source_ids = []
        batch_size, num_views = pixel_values.shape[:2]
        vision_model = self.vlm_backbone.get_vlm_model().vision_model

        for view_index in range(num_views):
            image = pixel_values[:, view_index].to(dtype=vision_model.dtype)
            patch_mask = None
            if pixel_attention_mask is not None:
                patch_mask = pixel_attention_mask[:, view_index]
                if patch_mask.shape[-2:] == image.shape[-2:]:
                    patch_size = int(vision_model.patch_size)
                    patch_mask = patch_mask.unfold(
                        1, patch_size, patch_size
                    ).unfold(2, patch_size, patch_size)
                    patch_mask = patch_mask.any(dim=-1).any(dim=-1)
            image_tokens = self.vlm_backbone.embed_image(image, patch_mask)
            image_tokens = image_tokens * math.sqrt(image_tokens.shape[-1])
            image_mask = torch.ones(
                batch_size,
                image_tokens.shape[1],
                device=image_tokens.device,
                dtype=torch.bool,
            )
            embeddings.append(image_tokens)
            pad_masks.append(image_mask)
            source_ids.append(
                torch.ones_like(image_mask, dtype=torch.long)
            )

        language = self.vlm_backbone.embed_language_tokens(input_ids)
        language = language * math.sqrt(language.shape[-1])
        language_mask = attention_mask.bool()
        embeddings.append(language)
        pad_masks.append(language_mask)
        source_ids.append(
            torch.where(
                language_mask,
                torch.zeros_like(language_mask, dtype=torch.long),
                torch.full_like(language_mask, -1, dtype=torch.long),
            )
        )

        prefix = torch.cat(embeddings, dim=1)
        pad_mask = torch.cat(pad_masks, dim=1)
        token_source_ids = torch.cat(source_ids, dim=1)
        block_mask = torch.zeros_like(pad_mask)
        prefix_attention = _make_attention_mask(pad_mask, block_mask)
        position_ids = torch.cumsum(pad_mask, dim=1) - 1
        hidden_states = self.vlm_backbone.forward_vlm_hidden_states(
            prefix,
            prefix_attention,
            position_ids,
        )
        return hidden_states, token_source_ids

    def _encode_observation(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        pixel_attention_mask,
        proprioception,
    ):
        hidden_states, token_source_ids = self._embed_prefix_with_sources(
            input_ids,
            attention_mask,
            pixel_values,
            pixel_attention_mask,
        )
        state = self._prepare_proprioception(
            proprioception,
            input_ids.shape[0],
            hidden_states[0].device,
            hidden_states[0].dtype,
        )
        observation_latent, condition_tokens = (
            self.vita_action_generator.encode_observation(
                hidden_states=hidden_states,
                hidden_token_type_ids=token_source_ids,
                proprioception=state,
                return_condition_tokens=True,
            )
        )
        if condition_tokens.shape[1] != 54:
            raise RuntimeError(
                f"Expected MQ54 condition bank, got {condition_tokens.shape[1]} tokens."
            )
        return observation_latent, condition_tokens

    def _encode_action_effect_pair(
        self,
        current_pixels,
        future_pixels,
        batch_size,
    ):
        if self.action_effect_visual_teacher_type == "smolvlm":
            return _encode_smolvlm_spatial_pair(
                self.vlm_backbone,
                current_pixels,
                future_pixels,
                batch_size,
            )
        teacher = self.action_effect_visual_teacher
        if teacher is None:
            raise RuntimeError("DINOv2 teacher is unavailable.")
        if current_pixels is None or future_pixels is None:
            raise ValueError(
                "Action-effect loss requires current and future main-view images."
            )
        teacher_param = next(teacher.parameters())
        pixels = torch.cat([current_pixels, future_pixels], dim=0).to(
            device=teacher_param.device,
            dtype=teacher_param.dtype,
        )
        teacher.eval()
        with torch.no_grad():
            cls = teacher(
                pixel_values=pixels,
                return_dict=True,
            ).last_hidden_state[:, :1]
        return cls[:batch_size].detach(), cls[batch_size:].detach()

    @staticmethod
    def _diversity_loss(condition_tokens):
        normalized = F.normalize(condition_tokens.float(), dim=-1)
        similarity = normalized @ normalized.transpose(1, 2)
        identity = torch.eye(
            condition_tokens.shape[1],
            device=condition_tokens.device,
            dtype=similarity.dtype,
        ).unsqueeze(0)
        return F.mse_loss(similarity, identity.expand_as(similarity))

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        pixel_attention_mask=None,
        actions=None,
        proprioception=None,
        return_loss_dict=False,
        action_effect_teacher_current_pixel_values=None,
        action_effect_teacher_future_pixel_values=None,
        **_,
    ):
        if actions is None:
            raise ValueError("Training requires actions.")
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        action_encoder = self.vita_action_generator.action_encoder
        actions = actions.to(dtype=action_encoder.input_proj.weight.dtype)

        observation_latent, condition_tokens = self._encode_observation(
            input_ids,
            attention_mask,
            pixel_values,
            pixel_attention_mask,
            proprioception,
        )
        action_latent = self.vita_action_generator.encode_action(actions)
        loss_flow = self.vita_action_generator.flow_matching_loss(
            observation_latent,
            action_latent,
            condition_tokens=condition_tokens,
        )

        encoded_actions = self.vita_action_generator.decode(action_latent)
        loss_action_recon = F.l1_loss(
            encoded_actions.float(),
            actions.float(),
        )
        loss = loss_flow + self.enc_recon_weight * loss_action_recon

        loss_visual = None
        if self.action_effect_enabled:
            current_visual, future_visual = self._encode_action_effect_pair(
                action_effect_teacher_current_pixel_values,
                action_effect_teacher_future_pixel_values,
                actions.shape[0],
            )
            current_visual = self.vita_action_generator.resample_visual_tokens(
                current_visual
            )
            future_visual = self.vita_action_generator.resample_visual_tokens(
                future_visual
            )
            visual_target = (future_visual - current_visual).detach()
            visual_prediction = self.vita_action_generator.decode_visual_delta(
                action_latent,
                current_visual,
            )
            loss_visual = F.smooth_l1_loss(
                visual_prediction.float(),
                visual_target.float(),
            )
            loss = loss + self.action_effect_visual_ae_weight * loss_visual

        loss_consistency = None
        loss_flow_recon = None
        if self.consistency_weight > 0 or self.flow_recon_weight > 0:
            predicted_action_latent = self.vita_action_generator.sample_latent(
                observation_latent,
                condition_tokens=condition_tokens,
            )
            loss_consistency = F.mse_loss(
                predicted_action_latent.float(),
                action_latent.float(),
            )
            loss = loss + self.consistency_weight * loss_consistency

            predicted_actions = self.vita_action_generator.decode(
                predicted_action_latent
            )
            loss_flow_recon = F.l1_loss(
                predicted_actions.float(),
                actions.float(),
            )
            loss = loss + self.flow_recon_weight * loss_flow_recon

        loss_diversity = (
            self._diversity_loss(condition_tokens)
            if self.log_diversity_loss
            else None
        )

        if not return_loss_dict:
            return loss
        output = {
            "loss": loss,
            "loss_flow": loss_flow.detach(),
            "loss_enc_action_recon": loss_action_recon.detach(),
            "loss_enc_action_recon_weighted": (
                self.enc_recon_weight * loss_action_recon
            ).detach(),
        }
        if loss_consistency is not None:
            output["loss_latent_consistency"] = loss_consistency.detach()
            output["loss_latent_consistency_weighted"] = (
                self.consistency_weight * loss_consistency
            ).detach()
        if loss_flow_recon is not None:
            output["loss_flow_action_recon"] = loss_flow_recon.detach()
            output["loss_flow_action_recon_weighted"] = (
                self.flow_recon_weight * loss_flow_recon
            ).detach()
        if loss_visual is not None:
            output["loss_action_effect_visual_ae"] = loss_visual.detach()
            output["loss_action_effect_visual_ae_weighted"] = (
                self.action_effect_visual_ae_weight * loss_visual
            ).detach()
        if loss_diversity is not None:
            output["loss_mq_diversity"] = loss_diversity.detach()
        return output

    @torch.no_grad()
    def predict_action(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        pixel_attention_mask=None,
        proprioception=None,
        **_,
    ):
        observation_latent, condition_tokens = self._encode_observation(
            input_ids,
            attention_mask,
            pixel_values,
            pixel_attention_mask,
            proprioception,
        )
        action_latent = self.vita_action_generator.sample_latent(
            observation_latent,
            num_steps=self.num_inference_timesteps,
            condition_tokens=condition_tokens,
        )
        actions = self.vita_action_generator.decode(action_latent)
        return actions.to(dtype=next(self.lmm.parameters()).dtype)
