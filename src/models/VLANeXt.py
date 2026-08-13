import sys
import os
from contextlib import nullcontext

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, AutoProcessor, AutoTokenizer, AutoModelForImageTextToText,
    SiglipVisionModel, SiglipImageProcessor, LlamaForCausalLM, 
    PaliGemmaForConditionalGeneration, 
    Qwen3VLForConditionalGeneration
)
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from .policies import (
    ActionDiffusionTransformerMetaquery, ActionDiffusionTransformerMoE,
    DualFlowActionDiffusionTransformerMoE,
    ActionRegressionTransformerMetaquery, ActionRegressionTransformerMoE,
    ActionClassificationTransformerMetaquery, ActionClassificationTransformerMoE, ActionVQVAE
)
from .generator import ImageGeneratorTransformer
from .encoder import ActionTransformerProjector
from .connector import ConnectorTransformer
from .progress_film import ProgressHead, ProgressFiLM
from .vlm_attention import LastLayerControlledHiddenAttention
from .latent_bridge import GatedCrossAttentionLatentBridge
from .vita_latent_flow import (
    ActionDecoder as VitaActionDecoder,
    ActionEncoder as VitaActionEncoder,
    VitaLatentActionGenerator,
    action_effect_ranking_loss,
    motion_weighted_dense_huber_loss,
    masked_action_reconstruction_loss,
    get_flow_latent_reconstruction_weights,
    resolve_flow_reconstruction_weights,
    recover_one_step_action_latent,
    spatial_pool_qwen_visual_tokens,
    stage_b_one_step_loss_weights,
)

try:
    from .Emu3_5_VisionTokenizer.modeling_emu3p5visionvq import Emu3p5VisionVQModel
except ImportError:
    # Fallback for directory with dot in name (Emu3.5_VisionTokenizer) which is not a valid package name
    sys.path.append(os.path.join(os.path.dirname(__file__), "Emu3.5_VisionTokenizer"))
    from modeling_emu3p5visionvq import Emu3p5VisionVQModel



class LlamaProcessorWrapper:
    def __init__(self, tokenizer, image_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor


class VLANeXtLatentActionEncoder(nn.Module):
    def __init__(self, action_dim, latent_dim, depth=4, num_heads=16, mlp_ratio=4.0, max_len=256, dropout=0.0):
        super().__init__()
        self.input_proj = nn.Linear(action_dim, latent_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, latent_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=int(latent_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(latent_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    def forward(self, actions):
        actions = actions.to(dtype=self.input_proj.weight.dtype)
        x = self.input_proj(actions)
        x = x + self.pos_embed[:, :x.shape[1], :].to(dtype=x.dtype)
        x = self.blocks(x)
        return self.norm(x)


class VLANeXtLatentActionDecoder(nn.Module):
    def __init__(self, action_dim, latent_dim, depth=4, num_heads=16, mlp_ratio=4.0, max_len=256, dropout=0.0):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, latent_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=int(latent_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(latent_dim)
        self.output_proj = nn.Linear(latent_dim, action_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, latent):
        latent = latent.to(dtype=self.output_proj.weight.dtype)
        x = latent + self.pos_embed[:, :latent.shape[1], :].to(dtype=latent.dtype)
        x = self.blocks(x)
        x = self.norm(x)
        return self.output_proj(x)


class VLANeXtObservationLatentProjector(nn.Module):
    def __init__(self, vlm_hidden_dim, latent_dim, num_actions, num_heads=16, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.query_embed = nn.Parameter(torch.zeros(1, num_actions, latent_dim))
        self.hidden_proj = (
            nn.Identity()
            if vlm_hidden_dim == latent_dim
            else nn.Linear(vlm_hidden_dim, latent_dim)
        )
        self.attn = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, int(latent_dim * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(int(latent_dim * mlp_ratio), latent_dim),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.query_embed, std=0.02)
        if isinstance(self.hidden_proj, nn.Linear):
            nn.init.xavier_uniform_(self.hidden_proj.weight)
            nn.init.zeros_(self.hidden_proj.bias)

    def forward(self, hidden_state):
        hidden_state = hidden_state.to(dtype=self.query_embed.dtype)
        kv = self.hidden_proj(hidden_state)
        queries = self.query_embed.expand(kv.shape[0], -1, -1).to(dtype=kv.dtype)
        attn_out, _ = self.attn(queries, kv, kv, need_weights=False)
        x = self.norm1(queries + attn_out)
        x = self.norm2(x + self.mlp(x))
        return x


class VLANeXt(nn.Module):
    def __init__(
        self, 
        lmm_path="Qwen/Qwen3-VL-2B-Instruct",
        vision_encoder_path="google/siglip2-base-patch16-256",
        action_dim=7,
        num_actions=1,
        num_queries=16,
        num_history=0,
        loss_type="diffusion", # Options: "diffusion", "regression", "classification"
        future_image_loss_weight=0.0,
        num_train_timesteps=1000,
        num_inference_timesteps=10,
        scheduler_type="ddim", # Options: "ddim", "flow_match"
        condition_type="loose", # Options: "loose", "tight", "soft"
        policy_hidden_size=1024,
        policy_depth=24,
        policy_num_heads=16,
        policy_mlp_ratio=4.0,
        use_proprio_input_vlm=True,
        use_action_input_policy=False,
        use_transformer_proprio_projector=True,
        projector_depth=2,
        projector_num_heads=4,
        use_transformer_connector=True,
        connector_depth=2,
        connector_num_heads=4,
        backbone_mode="finetune", # Options: "frozen", "finetune"
        gradient_checkpointing=True,
        num_bins=256,
        action_vqvae=None,

        generator_hidden_size=768,
        generator_depth=12,
        generator_num_heads=12,
        generator_mlp_ratio=4.0,
        attn_implementation="flash_attention_2",
        dct_loss_weight=0.1,
        dct_low_freq_weight=1.0,
        dct_high_freq_weight=3.0,
        dct_freq_split=0.5,
        dct_similarity_type="mse",  # Options: "mse", "mae", "cosine"
        use_progress_head=False,
        use_progress_film=False,
        progress_loss_weight=0.0,
        progress_detach=True,
        progress_noise=0.0,
        use_vlm_layer_attention=False,
        vlm_layer_attention_alpha=0.0,
        vlm_layer_indices=None,
        vlm_truncate_layers=None,
        use_latent_bridge=False,
        latent_bridge_num_heads=8,
        latent_bridge_max_views=4,
        latent_bridge_visual_trainable=True,
        latent_bridge_vlm_uses_proprio=False,
        action_generation_mode="direct_flow",
        vita_latent_dim=512,
        vita_hidden_dim=512,
        vita_action_ae_layers=4,
        vita_action_encoder_type="mlp",
        vita_action_cnn_layers=4,
        vita_action_cnn_kernel_size=5,
        vita_flow_layers=4,
        vita_flow_mlp_ratio=4.0,
        vita_dropout=0.0,
        vita_num_sampling_steps=6,
        vita_enc_recon_weight=0.5,
        vita_flow_recon_weight=0.5,
        vita_consistency_weight=1.0,
        vita_diversity_loss_weight=0.0,
        vita_log_diversity_loss=False,
        vita_action_progress_loss_weight=0.0,
        vita_goal_distance_loss_weight=0.0,
        vita_goal_metric_dim=256,
        vita_goal_distance_loss_type="huber",
        vita_goal_distance_detach_goal=False,
        vita_recon_loss_type="l1",
        vita_use_proprio=True,
        vita_zobs_mode="normal",
        vita_zobs_noise_scale=1.0,
        vita_condition_source="auto",
        vita_hidden_layer_index=-1,
        vita_secondary_hidden_layer_index=None,
        vita_extra_hidden_layer_indices=None,
        vita_hidden_pooling="mean",
        vita_pooling_num_heads=8,
        vita_pooling_num_queries=1,
        vita_gated_weighted_pooling=False,
        vita_obs_pool_token_indices=None,
        vita_hierarchical_query_pooling=False,
        vita_layer_local_queries_per_layer=2,
        vita_layer_local_queries_per_layer_list=None,
        vita_layer_local_token_source_modes=None,
        hier_mq_separate_views=False,
        vita_cross_layer_queries=18,
        vita_global_queries=8,
        vita_adaptive_local_mq=False,
        vita_adaptive_mq_router_dim=256,
        vita_adaptive_mq_num_slots=4,
        vita_adaptive_mq_reserve_source_positions=None,
        vita_adaptive_mq_temperature=1.0,
        vita_adaptive_mq_route_warmup_steps=0,
        vita_dynamic_topk_local_mq=False,
        vita_dynamic_topk_candidate_source_positions=None,
        vita_dynamic_topk_text_source_position=1,
        vita_dynamic_topk_candidates_per_source=8,
        vita_dynamic_topk_min_keep_per_source=2,
        vita_dynamic_topk_total_keep=24,
        vita_dynamic_topk_train_random_exploration=False,
        vita_condition_type_embeddings=False,
        vita_dynamic_extra_layer_gates=False,
        vita_extra_layer_gate_scales=None,
        vita_extra_layer_gate_hidden_dim=256,
        vita_extra_layer_gate_dropout=0.0,
        vita_blockwise_layer_conditioning=False,
        vita_blockwise_hidden_layer_indices=None,
        vita_state_token_conditioning=False,
        vita_state_num_tokens=4,
        vita_state_token_dropout=0.0,
        vita_state_broadcast_to_mq=True,
        vita_mq_local_overlap_loss_weight=0.0,
        vita_mq_cross_overlap_loss_weight=0.0,
        vita_mq_global_overlap_loss_weight=0.0,
        vita_mq_competitive_local_attention=False,
        vita_mq_competitive_attention_tau=1.0,
        vita_mq_competitive_attention_gamma=1.0,
        vita_mq_local_balance_loss_weight=0.0,
        vita_hierarchical_global_residual_block=False,
        vita_hierarchical_global_residual_scale=0.1,
        vita_hierarchical_global_ffn_ratio=4.0,
        vita_layer_aligned_mq=False,
        vita_layer_aligned_hidden_layer_indices=None,
        vita_layer_aligned_queries_per_layer=4,
        vita_layer_aligned_token_source_modes=None,
        vita_layer_aligned_global_conditioning=False,
        vita_layer_aligned_obs_pooling="last",
        vita_layer_aligned_encoder_obs_conditioning=True,
        vita_layer_aligned_flow_windows=None,
        vita_flow_memory_tokens=0,
        vita_flow_memory_update_scale=0.1,
        vita_flow_memory_mlp_ratio=4.0,
        vita_flow_memory_update_type="attention",
        vita_flow_memory_shared_update=False,
        vita_flow_memory_bottleneck_dim=256,
        vita_flow_layer_logits_init=None,
        vita_flow_cross_attention=False,
        vita_flow_cross_attention_heads=8,
        vita_flow_cross_attention_dim=None,
        vita_gated_flow_cross_attention=False,
        vita_typed_flow_cross_attention=False,
        vita_dynamic_typed_flow_gates=False,
        vita_mq_token_value_gating=False,
        vita_mq_headwise_token_value_gating=False,
        vita_mq_token_gate_lambda=0.1,
        vita_mq_token_gate_use_action_latent=False,
        vita_mq_token_gate_centered_residual=False,
        vita_mq_competitive_value_gating=False,
        vita_mq_global_relative_context_gating=False,
        vita_mq_global_relative_gate_score_dim=128,
        vita_mq_global_relative_gate_temperature=1.0,
        vita_flow_static_condition_cache=False,
        vita_flow_cross_attention_fixed_scale=None,
        vita_action_representation_type="legacy",
        vita_edar_stage_a_checkpoint=None,
        vita_edar_train_decoder=False,
        vita_edar_token_flow=False,
        vita_mq_type="hier_mq54",
        vita_enable_one_step_action_loss=False,
        vita_flow_loss_weight_final=0.3,
        vita_one_step_action_loss_weight_final=1.0,
        vita_stage_b_max_train_steps=1,
        dynamic_flow_reconstruction_weight=False,
        dynamic_flow_latent_reconstruction_weight=False,
        vita_action_effect_enabled=False,
        vita_action_effect_condition_action_encoder=True,
        vita_action_effect_visual_queries=4,
        vita_action_effect_visual_teacher="qwen",
        vita_action_effect_spatial_grid_size=8,
        vita_action_effect_visual_teacher_path=None,
        vita_action_effect_load_visual_teacher=True,
        vita_action_effect_num_heads=8,
        vita_action_effect_encoder_layers=2,
        vita_action_effect_decoder_layers=2,
        vita_action_effect_projection_dim=None,
        vita_action_effect_state_conditioning=False,
        vita_action_effect_visual_ae_weight=0.05,
        vita_action_effect_visual_flow_weight=0.05,
        vita_action_effect_rank_weight=0.0,
        vita_action_effect_rank_margin=0.1,
        vita_action_effect_rank_min_delta_norm=0.0,
        vita_action_effect_rank_distance_mode="global_cosine",
        vita_action_effect_rank_huber_weight=0.0,
        vita_action_effect_rank_scale_floor=1e-3,
        vita_action_effect_negative_queue_size=0,
        vita_action_effect_num_negatives=1,
        vita_action_effect_hard_negatives=1,
        vita_action_effect_negative_min_action_distance=0.0,
        vita_action_effect_main_view_only=False,
        vita_action_effect_horizons=None,
        vita_action_effect_horizon_loss_weights=None,
        vita_action_effect_action_token_dim=256,
        vita_action_effect_multihorizon_layers=2,
        vita_action_effect_motion_weight_floor=0.25,
        vita_action_effect_motion_weight_eps=1e-6,
        vita_action_effect_detach_visual_teacher=False,
        vlanext_dual_flow=False,
        vlanext_visual_flow_loss_weight=0.0,
    ):
        super().__init__()
        
        print(f"Initializing VLM {lmm_path} with attn_implementation: {attn_implementation}")
        if "paligemma" in lmm_path.lower():
            self.model_family = "paligemma"
            self.lmm = PaliGemmaForConditionalGeneration.from_pretrained(
                lmm_path, dtype=torch.bfloat16, _attn_implementation=attn_implementation
            )
            self.processor = AutoProcessor.from_pretrained(lmm_path, trust_remote_code=True)
            if hasattr(self.lmm.config, "text_config"):
                self.hidden_size = self.lmm.config.text_config.hidden_size
            else:
                self.hidden_size = self.lmm.config.hidden_size
        elif "llama" in lmm_path.lower():
            self.model_family = "llama"
            self.lmm = LlamaForCausalLM.from_pretrained(
                lmm_path, dtype=torch.bfloat16, attn_implementation=attn_implementation
            )
            self.vision_encoder = SiglipVisionModel.from_pretrained(
                vision_encoder_path, dtype=torch.bfloat16, attn_implementation=attn_implementation
            )
            tokenizer = AutoTokenizer.from_pretrained(lmm_path)
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
            image_processor = SiglipImageProcessor.from_pretrained(vision_encoder_path)
            self.processor = LlamaProcessorWrapper(tokenizer, image_processor)
            self.hidden_size = self.lmm.config.hidden_size
            self.vision_projector = nn.Sequential(
                nn.Linear(self.vision_encoder.config.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.SiLU(), 
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.SiLU(), 
                nn.Linear(self.hidden_size, self.hidden_size)
            )
        elif "qwen" in lmm_path.lower():
            self.model_family = "qwen"
            if "qwen3.5" in lmm_path.lower() or "qwen3_5" in lmm_path.lower():
                self.lmm = AutoModelForImageTextToText.from_pretrained(
                    lmm_path,
                    dtype=torch.bfloat16,
                    attn_implementation=attn_implementation,
                    trust_remote_code=True,
                )
            else:
                self.lmm = Qwen3VLForConditionalGeneration.from_pretrained(
                    lmm_path, dtype=torch.bfloat16, _attn_implementation=attn_implementation
                )
            self.processor = AutoProcessor.from_pretrained(lmm_path, trust_remote_code=True)
            if hasattr(self.lmm.config, "text_config"):
                self.hidden_size = self.lmm.config.text_config.hidden_size
            else:
                self.hidden_size = self.lmm.config.hidden_size
        
        if backbone_mode == "frozen":
            self.lmm.requires_grad_(False)
            if self.model_family == "llama":
                self.vision_encoder.requires_grad_(False)
        elif backbone_mode == "finetune":
            self.lmm.requires_grad_(True)
            if self.model_family == "llama":
                self.vision_encoder.requires_grad_(True)
        else:
            raise ValueError(f"Unknown backbone_mode: {backbone_mode}")

        if gradient_checkpointing:
            model_to_configure = self.lmm
            if hasattr(model_to_configure, "gradient_checkpointing_enable"):
                model_to_configure.gradient_checkpointing_enable()
            if hasattr(self.lmm, "enable_input_require_grads"):
                self.lmm.enable_input_require_grads()
            config = self.lmm.config
            if hasattr(config, "use_cache"):
                config.use_cache = False
            if self.model_family == "llama":
                 if hasattr(self.vision_encoder, "gradient_checkpointing_enable"):
                    self.vision_encoder.gradient_checkpointing_enable()

        self.num_queries = num_queries
        self.loss_type = loss_type
        self.scheduler_type = scheduler_type
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_inference_timesteps
        self.action_dim = action_dim
        self.num_actions = num_actions
        self.num_history = num_history
        self.num_bins = num_bins
        self.condition_type = condition_type
        self.use_proprio_input_vlm = use_proprio_input_vlm
        self.use_action_input_policy = use_action_input_policy
        self.future_image_loss_weight = future_image_loss_weight
        self.enable_future_image_loss = (future_image_loss_weight > 0)
        self.dct_loss_weight = dct_loss_weight
        self.dct_low_freq_weight = dct_low_freq_weight
        self.dct_high_freq_weight = dct_high_freq_weight
        self.dct_freq_split = dct_freq_split
        self.dct_similarity_type = dct_similarity_type
        self.use_progress_head = use_progress_head or use_progress_film
        self.use_progress_film = use_progress_film
        self.progress_loss_weight = float(progress_loss_weight)
        self.progress_detach = progress_detach
        self.progress_noise = float(progress_noise)
        self.use_vlm_layer_attention = bool(use_vlm_layer_attention)
        self.vlm_layer_indices = tuple(vlm_layer_indices) if vlm_layer_indices is not None else None
        self.vlm_truncate_layers = int(vlm_truncate_layers) if vlm_truncate_layers is not None else None
        self.policy_depth = int(policy_depth)
        self.use_latent_bridge = bool(use_latent_bridge)
        self.latent_bridge_vlm_uses_proprio = bool(latent_bridge_vlm_uses_proprio)
        self.action_generation_mode = str(action_generation_mode)
        self.use_vita_latent_flow = self.action_generation_mode == "vita_latent_flow"
        self.use_vlanext_latent_flow = self.action_generation_mode == "vlanext_latent_flow"
        self.use_vlanext_dual_flow = bool(vlanext_dual_flow)
        self.vlanext_visual_flow_loss_weight = float(vlanext_visual_flow_loss_weight)
        self.vlanext_latent_dim = int(vita_latent_dim)
        self.vita_enc_recon_weight = float(vita_enc_recon_weight)
        self.vita_flow_recon_weight = float(vita_flow_recon_weight)
        self.vita_consistency_weight = float(vita_consistency_weight)
        self.vita_diversity_loss_weight = float(vita_diversity_loss_weight)
        self.vita_log_diversity_loss = bool(vita_log_diversity_loss)
        self.vita_action_progress_loss_weight = float(vita_action_progress_loss_weight)
        self.vita_goal_distance_loss_weight = float(vita_goal_distance_loss_weight)
        self.vita_goal_distance_loss_type = str(vita_goal_distance_loss_type)
        self.vita_goal_distance_detach_goal = bool(vita_goal_distance_detach_goal)
        self.vita_action_effect_enabled = bool(vita_action_effect_enabled)
        self.vita_action_representation_type = str(
            vita_action_representation_type
        ).lower()
        self.vita_edar_stage_a_checkpoint = (
            os.path.expanduser(vita_edar_stage_a_checkpoint)
            if vita_edar_stage_a_checkpoint
            else None
        )
        self.vita_edar_train_decoder = bool(vita_edar_train_decoder)
        self.vita_edar_token_flow = bool(vita_edar_token_flow)
        self.vita_mq_type = str(vita_mq_type).lower()
        self.vita_enable_one_step_action_loss = bool(
            vita_enable_one_step_action_loss
        )
        self.vita_flow_loss_weight_final = float(vita_flow_loss_weight_final)
        self.vita_one_step_action_loss_weight_final = float(
            vita_one_step_action_loss_weight_final
        )
        self.vita_stage_b_max_train_steps = max(
            int(vita_stage_b_max_train_steps), 1
        )
        self.dynamic_flow_reconstruction_weight = bool(
            dynamic_flow_reconstruction_weight
        )
        self.dynamic_flow_latent_reconstruction_weight = bool(
            dynamic_flow_latent_reconstruction_weight
        )
        self.vita_stage_b_global_step = 0
        self.vita_action_effect_visual_teacher_type = str(
            vita_action_effect_visual_teacher
        ).lower()
        self.vita_action_effect_spatial_grid_size = int(
            vita_action_effect_spatial_grid_size
        )
        self.vita_action_effect_visual_teacher_path = (
            os.path.expanduser(vita_action_effect_visual_teacher_path)
            if vita_action_effect_visual_teacher_path
            else None
        )
        self.vita_action_effect_load_visual_teacher = bool(
            vita_action_effect_load_visual_teacher
        )
        self.vita_action_effect_visual_ae_weight = float(vita_action_effect_visual_ae_weight)
        self.vita_action_effect_visual_flow_weight = float(vita_action_effect_visual_flow_weight)
        self.vita_action_effect_rank_weight = float(vita_action_effect_rank_weight)
        self.vita_action_effect_rank_margin = float(vita_action_effect_rank_margin)
        self.vita_action_effect_rank_min_delta_norm = float(
            vita_action_effect_rank_min_delta_norm
        )
        self.vita_action_effect_rank_distance_mode = str(
            vita_action_effect_rank_distance_mode
        )
        self.vita_action_effect_rank_huber_weight = float(
            vita_action_effect_rank_huber_weight
        )
        self.vita_action_effect_rank_scale_floor = float(
            vita_action_effect_rank_scale_floor
        )
        self.vita_action_effect_negative_queue_size = int(
            vita_action_effect_negative_queue_size
        )
        self.vita_action_effect_num_negatives = int(vita_action_effect_num_negatives)
        self.vita_action_effect_hard_negatives = int(vita_action_effect_hard_negatives)
        self.vita_action_effect_negative_min_action_distance = float(
            vita_action_effect_negative_min_action_distance
        )
        self.vita_action_effect_main_view_only = bool(vita_action_effect_main_view_only)
        self.vita_action_effect_horizons = tuple(
            int(value) for value in (vita_action_effect_horizons or ())
        )
        self.vita_action_effect_horizon_loss_weights = tuple(
            float(value)
            for value in (
                vita_action_effect_horizon_loss_weights
                if vita_action_effect_horizon_loss_weights is not None
                else [1.0] * len(self.vita_action_effect_horizons)
            )
        )
        self.vita_action_effect_motion_weight_floor = float(
            vita_action_effect_motion_weight_floor
        )
        self.vita_action_effect_motion_weight_eps = float(
            vita_action_effect_motion_weight_eps
        )
        self.vita_action_effect_detach_visual_teacher = bool(
            vita_action_effect_detach_visual_teacher
        )
        if self.vita_action_effect_visual_teacher_type not in {
            "qwen",
            "qwen_spatial",
            "dinov2",
            "dinov3",
        }:
            raise ValueError(
                "vita_action_effect_visual_teacher must be 'qwen', "
                "'qwen_spatial', 'dinov2', or 'dinov3'."
            )
        if (
            self.vita_action_representation_type == "single_view_edar_lite"
            and self.vita_action_effect_visual_teacher_type not in {"dinov2", "dinov3"}
        ):
            raise ValueError("EDAR-lite Stage B requires a DINOv2 or DINOv3 visual teacher.")
        if (
            self.vita_action_representation_type == "single_view_edar_lite"
            and not self.vita_action_effect_enabled
        ):
            raise ValueError("EDAR-lite Stage B requires current-view teacher features.")
        if self.vita_action_effect_spatial_grid_size <= 0:
            raise ValueError("vita_action_effect_spatial_grid_size must be positive.")
        if self.vita_action_effect_horizons:
            if not self.vita_action_effect_enabled:
                raise ValueError(
                    "vita_action_effect_horizons requires "
                    "vita_action_effect_enabled=true."
                )
            if tuple(sorted(set(self.vita_action_effect_horizons))) != (
                self.vita_action_effect_horizons
            ):
                raise ValueError(
                    "vita_action_effect_horizons must be unique and sorted."
                )
            if (
                self.vita_action_effect_horizons[0] <= 0
                or self.vita_action_effect_horizons[-1] > int(num_actions)
            ):
                raise ValueError(
                    "vita_action_effect_horizons must lie within the action chunk."
                )
            if len(self.vita_action_effect_horizon_loss_weights) != len(
                self.vita_action_effect_horizons
            ):
                raise ValueError(
                    "vita_action_effect_horizon_loss_weights must match horizons."
                )
            if (
                any(value < 0 for value in self.vita_action_effect_horizon_loss_weights)
                or not any(
                    value > 0
                    for value in self.vita_action_effect_horizon_loss_weights
                )
            ):
                raise ValueError(
                    "Horizon loss weights must be non-negative with a positive sum."
                )
            if self.vita_action_effect_visual_teacher_type != "qwen_spatial":
                raise ValueError(
                    "Multi-horizon action-effect training requires qwen_spatial."
                )
            if not self.vita_action_effect_main_view_only:
                raise ValueError(
                    "Multi-horizon action-effect training requires the main view only."
                )
            if vita_action_effect_condition_action_encoder:
                raise ValueError(
                    "Multi-horizon prediction preserves the original ActionEncoder; "
                    "set vita_action_effect_condition_action_encoder=false."
                )
            if self.vita_action_effect_rank_weight > 0:
                raise ValueError(
                    "Action-effect ranking is not supported by the multi-horizon branch."
                )
            if self.vita_action_effect_visual_flow_weight > 0:
                raise ValueError(
                    "Visual Flow loss is not supported by the multi-horizon branch."
                )
        if not 0.0 <= self.vita_action_effect_motion_weight_floor <= 1.0:
            raise ValueError(
                "vita_action_effect_motion_weight_floor must be in [0, 1]."
            )
        if self.vita_action_effect_motion_weight_eps <= 0:
            raise ValueError("vita_action_effect_motion_weight_eps must be positive.")
        if int(vita_action_effect_action_token_dim) <= 0:
            raise ValueError("vita_action_effect_action_token_dim must be positive.")
        if int(vita_action_effect_multihorizon_layers) <= 0:
            raise ValueError(
                "vita_action_effect_multihorizon_layers must be positive."
            )
        if (
            self.vita_action_effect_visual_teacher_type == "qwen_spatial"
            and not self.vita_action_effect_main_view_only
        ):
            raise ValueError(
                "The Qwen spatial action-effect teacher currently requires "
                "vita_action_effect_main_view_only=true."
            )
        if (
            self.vita_action_effect_enabled
            and self.vita_action_effect_visual_teacher_type in {"dinov2", "dinov3"}
            and self.vita_action_effect_load_visual_teacher
            and not self.vita_action_effect_visual_teacher_path
        ):
            raise ValueError("DINO action-effect training requires a local teacher path.")
        if self.vita_action_effect_rank_weight < 0:
            raise ValueError("vita_action_effect_rank_weight must be non-negative.")
        if self.vita_action_effect_rank_margin < 0:
            raise ValueError("vita_action_effect_rank_margin must be non-negative.")
        if self.vita_action_effect_rank_min_delta_norm < 0:
            raise ValueError("vita_action_effect_rank_min_delta_norm must be non-negative.")
        if self.vita_action_effect_rank_distance_mode not in {
            "global_cosine",
            "tokenwise_cosine",
            "tokenwise_mixed",
        }:
            raise ValueError(
                "vita_action_effect_rank_distance_mode must be global_cosine, "
                "tokenwise_cosine, or tokenwise_mixed."
            )
        if self.vita_action_effect_rank_huber_weight < 0:
            raise ValueError("vita_action_effect_rank_huber_weight must be non-negative.")
        if self.vita_action_effect_rank_scale_floor <= 0:
            raise ValueError("vita_action_effect_rank_scale_floor must be positive.")
        if self.vita_action_effect_negative_queue_size < 0:
            raise ValueError("vita_action_effect_negative_queue_size must be non-negative.")
        if self.vita_action_effect_num_negatives <= 0:
            raise ValueError("vita_action_effect_num_negatives must be positive.")
        if self.vita_action_effect_hard_negatives <= 0:
            raise ValueError("vita_action_effect_hard_negatives must be positive.")
        if self.vita_action_effect_hard_negatives > self.vita_action_effect_num_negatives:
            raise ValueError(
                "vita_action_effect_hard_negatives cannot exceed "
                "vita_action_effect_num_negatives."
            )
        if self.vita_action_effect_negative_min_action_distance < 0:
            raise ValueError(
                "vita_action_effect_negative_min_action_distance must be non-negative."
            )
        self.register_buffer(
            "_action_effect_raw_action_queue",
            torch.zeros(
                self.vita_action_effect_negative_queue_size,
                num_actions,
                action_dim,
            ),
            persistent=False,
        )
        self._action_effect_queue_count = 0
        self._action_effect_queue_write_index = 0
        self.vita_recon_loss_type = str(vita_recon_loss_type)
        self.vita_use_proprio = bool(vita_use_proprio)
        self.vita_zobs_mode = str(vita_zobs_mode)
        self.vita_zobs_noise_scale = float(vita_zobs_noise_scale)
        if self.vita_zobs_mode not in {"normal", "noise"}:
            raise ValueError("vita_zobs_mode must be 'normal' or 'noise'.")

        if self.action_generation_mode not in {"direct_flow", "vita_latent_flow", "vlanext_latent_flow"}:
            raise ValueError(
                f"Unknown action_generation_mode: {self.action_generation_mode}. "
                "Expected 'direct_flow', 'vita_latent_flow', or 'vlanext_latent_flow'."
            )
        if (self.use_vita_latent_flow or self.use_vlanext_latent_flow) and (
            loss_type != "diffusion" or scheduler_type != "flow_match"
        ):
            raise ValueError(
                "Latent flow action modes require loss_type='diffusion' and scheduler_type='flow_match'."
            )
        
        self.action_vqvae_config = action_vqvae
        if self.action_vqvae_config.get('enabled', False):
            self.action_vqvae = ActionVQVAE(
                action_dim=action_dim,
                latent_codes_per_step=3, 
                codebook_size=self.action_vqvae_config.get('codebook_size', 1024),
                hidden_size=self.action_vqvae_config.get('hidden_size', 256),
                depth=self.action_vqvae_config.get('depth', 2),
                num_heads=self.action_vqvae_config.get('num_heads', 4)
            )
        else:
            self.action_vqvae = None


        if self.enable_future_image_loss:
            print("Initializing Future Image Generator Components...")
            self.vq_model = Emu3p5VisionVQModel.from_pretrained("BAAI/Emu3.5-VisionTokenizer", trust_remote_code=True)
            self.vq_model.requires_grad_(False)
            self.vq_codebook_size = self.vq_model.config.codebook_size
            
            self.generator = ImageGeneratorTransformer(
                vocab_size=self.vq_codebook_size,
                vlm_hidden_size=self.hidden_size,
                hidden_size=generator_hidden_size,
                depth=generator_depth,
                num_heads=generator_num_heads,
                mlp_ratio=generator_mlp_ratio
            )
        else:
            self.vq_model = None
            self.generator = None

        if self.use_proprio_input_vlm:
            projector_input_dim = action_dim
            if use_transformer_proprio_projector:
                self.action_projector = ActionTransformerProjector(
                    action_dim=projector_input_dim,
                    hidden_size=self.hidden_size,
                    depth=projector_depth,
                    num_heads=projector_num_heads
                )
            else:
                self.action_projector = nn.Linear(projector_input_dim, self.hidden_size)
        else:
            self.action_projector = None
        
        self.meta_queries = nn.Parameter(
            torch.randn(num_queries, self.hidden_size)
        )
        if self.condition_type == "loose":
            if use_transformer_connector:
                self.connector = ConnectorTransformer(
                    input_dim=self.hidden_size,
                    output_dim=self.hidden_size,
                    depth=connector_depth,
                    num_heads=connector_num_heads
                )
            else:
                self.connector = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.SiLU(),
                    nn.Linear(self.hidden_size, self.hidden_size) # Project to diffusion cond dim
                )
        else:
            self.connector = None

        if self.use_progress_head:
            self.progress_head = ProgressHead(self.hidden_size)
            self.progress_film = ProgressFiLM(self.hidden_size) if self.use_progress_film else None
        else:
            self.progress_head = None
            self.progress_film = None

        self.vlm_layer_attention = (
            LastLayerControlledHiddenAttention(self.hidden_size, init_alpha=vlm_layer_attention_alpha)
            if self.use_vlm_layer_attention
            else None
        )
        self.latent_bridge = (
            GatedCrossAttentionLatentBridge(
                hidden_size=self.hidden_size,
                action_dim=action_dim,
                num_history=num_history,
                num_heads=latent_bridge_num_heads,
                max_views=latent_bridge_max_views,
                visual_trainable=latent_bridge_visual_trainable,
            )
            if self.use_latent_bridge
            else None
        )
        self.vita_action_generator = (
            VitaLatentActionGenerator(
                action_dim=action_dim,
                horizon=num_actions,
                vlm_hidden_dim=self.hidden_size,
                num_queries=num_queries,
                latent_dim=vita_latent_dim,
                hidden_dim=vita_hidden_dim,
                action_ae_layers=vita_action_ae_layers,
                action_encoder_type=vita_action_encoder_type,
                action_cnn_layers=vita_action_cnn_layers,
                action_cnn_kernel_size=vita_action_cnn_kernel_size,
                flow_layers=vita_flow_layers,
                flow_mlp_ratio=vita_flow_mlp_ratio,
                dropout=vita_dropout,
                num_sampling_steps=vita_num_sampling_steps,
                proprio_dim=(num_history * action_dim) if self.vita_use_proprio else None,
                condition_source=vita_condition_source,
                hidden_layer_index=vita_hidden_layer_index,
                secondary_hidden_layer_index=vita_secondary_hidden_layer_index,
                extra_hidden_layer_indices=vita_extra_hidden_layer_indices,
                hidden_pooling=vita_hidden_pooling,
                pooling_num_heads=vita_pooling_num_heads,
                pooling_num_queries=vita_pooling_num_queries,
                gated_weighted_pooling=vita_gated_weighted_pooling,
                obs_pool_token_indices=vita_obs_pool_token_indices,
                hierarchical_query_pooling=vita_hierarchical_query_pooling,
                layer_local_queries_per_layer=vita_layer_local_queries_per_layer,
                layer_local_queries_per_layer_list=vita_layer_local_queries_per_layer_list,
                layer_local_token_source_modes=vita_layer_local_token_source_modes,
                hier_mq_separate_views=hier_mq_separate_views,
                cross_layer_queries=vita_cross_layer_queries,
                global_queries=vita_global_queries,
                adaptive_local_mq=vita_adaptive_local_mq,
                adaptive_mq_router_dim=vita_adaptive_mq_router_dim,
                adaptive_mq_num_slots=vita_adaptive_mq_num_slots,
                adaptive_mq_reserve_source_positions=vita_adaptive_mq_reserve_source_positions,
                adaptive_mq_temperature=vita_adaptive_mq_temperature,
                adaptive_mq_route_warmup_steps=vita_adaptive_mq_route_warmup_steps,
                dynamic_topk_local_mq=vita_dynamic_topk_local_mq,
                dynamic_topk_candidate_source_positions=vita_dynamic_topk_candidate_source_positions,
                dynamic_topk_text_source_position=vita_dynamic_topk_text_source_position,
                dynamic_topk_candidates_per_source=vita_dynamic_topk_candidates_per_source,
                dynamic_topk_min_keep_per_source=vita_dynamic_topk_min_keep_per_source,
                dynamic_topk_total_keep=vita_dynamic_topk_total_keep,
                dynamic_topk_train_random_exploration=vita_dynamic_topk_train_random_exploration,
                condition_type_embeddings=vita_condition_type_embeddings,
                dynamic_extra_layer_gates=vita_dynamic_extra_layer_gates,
                extra_layer_gate_scales=vita_extra_layer_gate_scales,
                extra_layer_gate_hidden_dim=vita_extra_layer_gate_hidden_dim,
                extra_layer_gate_dropout=vita_extra_layer_gate_dropout,
                blockwise_layer_conditioning=vita_blockwise_layer_conditioning,
                blockwise_hidden_layer_indices=vita_blockwise_hidden_layer_indices,
                state_token_conditioning=vita_state_token_conditioning,
                state_num_tokens=vita_state_num_tokens,
                state_token_dropout=vita_state_token_dropout,
                state_broadcast_to_mq=vita_state_broadcast_to_mq,
                mq_local_overlap_loss_weight=vita_mq_local_overlap_loss_weight,
                mq_cross_overlap_loss_weight=vita_mq_cross_overlap_loss_weight,
                mq_global_overlap_loss_weight=vita_mq_global_overlap_loss_weight,
                mq_competitive_local_attention=vita_mq_competitive_local_attention,
                mq_competitive_attention_tau=vita_mq_competitive_attention_tau,
                mq_competitive_attention_gamma=vita_mq_competitive_attention_gamma,
                mq_local_balance_loss_weight=vita_mq_local_balance_loss_weight,
                hierarchical_global_residual_block=vita_hierarchical_global_residual_block,
                hierarchical_global_residual_scale=vita_hierarchical_global_residual_scale,
                hierarchical_global_ffn_ratio=vita_hierarchical_global_ffn_ratio,
                layer_aligned_query_pooling=vita_layer_aligned_mq,
                layer_aligned_hidden_layer_indices=vita_layer_aligned_hidden_layer_indices,
                layer_aligned_queries_per_layer=vita_layer_aligned_queries_per_layer,
                layer_aligned_token_source_modes=vita_layer_aligned_token_source_modes,
                layer_aligned_global_conditioning=vita_layer_aligned_global_conditioning,
                layer_aligned_obs_pooling=vita_layer_aligned_obs_pooling,
                layer_aligned_encoder_obs_conditioning=vita_layer_aligned_encoder_obs_conditioning,
                layer_aligned_flow_windows=vita_layer_aligned_flow_windows,
                flow_memory_tokens=vita_flow_memory_tokens,
                flow_memory_update_scale=vita_flow_memory_update_scale,
                flow_memory_mlp_ratio=vita_flow_memory_mlp_ratio,
                flow_memory_update_type=vita_flow_memory_update_type,
                flow_memory_shared_update=vita_flow_memory_shared_update,
                flow_memory_bottleneck_dim=vita_flow_memory_bottleneck_dim,
                flow_cross_attention=vita_flow_cross_attention,
                flow_cross_attention_heads=vita_flow_cross_attention_heads,
                flow_cross_attention_dim=vita_flow_cross_attention_dim,
                gated_flow_cross_attention=vita_gated_flow_cross_attention,
                typed_flow_cross_attention=vita_typed_flow_cross_attention,
                dynamic_typed_flow_gates=vita_dynamic_typed_flow_gates,
                mq_token_value_gating=vita_mq_token_value_gating,
                mq_headwise_token_value_gating=vita_mq_headwise_token_value_gating,
                mq_token_gate_lambda=vita_mq_token_gate_lambda,
                mq_token_gate_use_action_latent=vita_mq_token_gate_use_action_latent,
                mq_token_gate_centered_residual=vita_mq_token_gate_centered_residual,
                mq_competitive_value_gating=vita_mq_competitive_value_gating,
                mq_global_relative_context_gating=(
                    vita_mq_global_relative_context_gating
                ),
                mq_global_relative_gate_score_dim=(
                    vita_mq_global_relative_gate_score_dim
                ),
                mq_global_relative_gate_temperature=(
                    vita_mq_global_relative_gate_temperature
                ),
                flow_static_condition_cache=(
                    vita_flow_static_condition_cache
                ),
                fixed_flow_cross_attention_scale=vita_flow_cross_attention_fixed_scale,
                flow_layer_logits_init=vita_flow_layer_logits_init,
                action_representation_type=self.vita_action_representation_type,
                edar_stage_a_checkpoint=self.vita_edar_stage_a_checkpoint,
                edar_train_decoder=self.vita_edar_train_decoder,
                edar_token_flow=self.vita_edar_token_flow,
                mq_type=self.vita_mq_type,
                action_effect_enabled=self.vita_action_effect_enabled,
                action_effect_condition_action_encoder=vita_action_effect_condition_action_encoder,
                action_effect_visual_queries=vita_action_effect_visual_queries,
                action_effect_visual_tokens_prepooled=(
                    self.vita_action_effect_visual_teacher_type
                    in {"dinov2", "dinov3", "qwen_spatial"}
                ),
                action_effect_num_heads=vita_action_effect_num_heads,
                action_effect_encoder_layers=vita_action_effect_encoder_layers,
                action_effect_decoder_layers=vita_action_effect_decoder_layers,
                action_effect_projection_dim=vita_action_effect_projection_dim,
                action_effect_state_conditioning=vita_action_effect_state_conditioning,
                action_effect_horizons=self.vita_action_effect_horizons,
                action_effect_action_token_dim=vita_action_effect_action_token_dim,
                action_effect_multihorizon_layers=(
                    vita_action_effect_multihorizon_layers
                ),
            )
            if self.use_vita_latent_flow
            else None
        )
        self.action_effect_visual_teacher = None
        if (
            self.use_vita_latent_flow
            and self.vita_action_effect_enabled
            and self.vita_action_effect_visual_teacher_type in {"dinov2", "dinov3"}
            and self.vita_action_effect_load_visual_teacher
        ):
            print(
                "Initializing frozen DINO action-effect teacher "
                f"{self.vita_action_effect_visual_teacher_path}"
            )
            self.action_effect_visual_teacher = AutoModel.from_pretrained(
                self.vita_action_effect_visual_teacher_path,
                local_files_only=True,
                dtype=torch.bfloat16,
            )
            teacher_dim = int(self.action_effect_visual_teacher.config.hidden_size)
            if (
                self.vita_action_representation_type != "single_view_edar_lite"
                and teacher_dim != int(vita_latent_dim)
            ):
                raise ValueError(
                    f"DINOv2 teacher hidden size {teacher_dim} must match "
                    f"vita_latent_dim {vita_latent_dim}."
                )
            self.action_effect_visual_teacher.requires_grad_(False)
            self.action_effect_visual_teacher.eval()
        self.vita_action_progress_head = (
            nn.Sequential(
                nn.LayerNorm(vita_latent_dim),
                nn.Linear(vita_latent_dim, vita_hidden_dim),
                nn.SiLU(),
                nn.Linear(vita_hidden_dim, 1),
            )
            if self.use_vita_latent_flow and self.vita_action_progress_loss_weight > 0
            else None
        )
        self.vita_goal_metric_head = (
            nn.Sequential(
                nn.LayerNorm(vita_latent_dim),
                nn.Linear(vita_latent_dim, vita_hidden_dim),
                nn.SiLU(),
                nn.Linear(vita_hidden_dim, int(vita_goal_metric_dim)),
            )
            if self.use_vita_latent_flow and self.vita_goal_distance_loss_weight > 0
            else None
        )
        if self.use_vlanext_latent_flow:
            self.vlanext_action_encoder = VitaActionEncoder(
                action_dim=action_dim,
                horizon=num_actions,
                latent_dim=self.vlanext_latent_dim,
                hidden_dim=vita_hidden_dim,
                num_layers=vita_action_ae_layers,
                dropout=vita_dropout,
            )
            self.vlanext_action_decoder = VitaActionDecoder(
                action_dim=action_dim,
                horizon=num_actions,
                latent_dim=self.vlanext_latent_dim,
                hidden_dim=vita_hidden_dim,
                num_layers=vita_action_ae_layers,
                dropout=vita_dropout,
            )
            self.vlanext_obs_projector = VLANeXtObservationLatentProjector(
                vlm_hidden_dim=self.hidden_size,
                latent_dim=self.vlanext_latent_dim,
                num_actions=1,
                num_heads=policy_num_heads,
                mlp_ratio=vita_flow_mlp_ratio,
                dropout=vita_dropout,
            )
        else:
            self.vlanext_action_encoder = None
            self.vlanext_action_decoder = None
            self.vlanext_obs_projector = None

        gen_hidden_dim = generator_hidden_size if self.enable_future_image_loss else None
        if self.use_vita_latent_flow:
            self.action_head = None
        elif self.use_vlanext_latent_flow:
            if condition_type not in ["tight", "soft"]:
                raise ValueError("vlanext_latent_flow currently requires condition_type='tight' or 'soft'.")
            head_cls = DualFlowActionDiffusionTransformerMoE if self.use_vlanext_dual_flow else ActionDiffusionTransformerMoE
            self.action_head = head_cls(
                action_dim=self.vlanext_latent_dim,
                vlm_hidden_size=self.hidden_size,
                hidden_size=policy_hidden_size,
                depth=policy_depth,
                num_heads=policy_num_heads,
                mlp_ratio=policy_mlp_ratio,
                gen_hidden_size=gen_hidden_dim,
                vlm_layer_indices=self.vlm_layer_indices,
            )
        elif loss_type == "regression":
            if condition_type in ["tight", "soft"]:
                self.action_head = ActionRegressionTransformerMoE(
                    action_dim=action_dim,
                    vlm_hidden_size=self.hidden_size,
                    num_actions=num_actions,
                    hidden_size=policy_hidden_size,
                    depth=policy_depth,
                    num_heads=policy_num_heads,
                    mlp_ratio=policy_mlp_ratio,
                    gen_hidden_size=gen_hidden_dim,
                    vlm_layer_indices=self.vlm_layer_indices,
                )
            elif condition_type == "loose":
                self.action_head = ActionRegressionTransformerMetaquery(
                    action_dim=action_dim,
                    condition_dim=self.hidden_size,
                    num_actions=num_actions,
                    hidden_size=policy_hidden_size,
                    depth=policy_depth,
                    num_heads=policy_num_heads,
                    mlp_ratio=policy_mlp_ratio
                )
            else:
                raise ValueError(f"Unknown condition type for regression: {condition_type}")
            self.noise_scheduler = None
        elif loss_type == "classification":
            is_vqvae = (self.action_vqvae is not None)

            if condition_type == "loose":
                if is_vqvae:
                    self.action_head = ActionClassificationTransformerMetaquery(
                        action_dim=action_dim,
                        condition_dim=self.hidden_size,
                        num_actions=num_actions,
                        hidden_size=policy_hidden_size,
                        depth=policy_depth,
                        num_heads=policy_num_heads,
                        mlp_ratio=policy_mlp_ratio,
                        vqvae_mode=True,
                        vq_codebook_size=self.action_vqvae.codebook_size,
                        vq_latent_codes=self.action_vqvae.latent_codes
                    )
                else:
                    self.action_head = ActionClassificationTransformerMetaquery(
                        action_dim=action_dim,
                        condition_dim=self.hidden_size,
                        num_actions=num_actions,
                        num_bins=num_bins,
                        hidden_size=policy_hidden_size,
                        depth=policy_depth,
                        num_heads=policy_num_heads,
                        mlp_ratio=policy_mlp_ratio,
                        vqvae_mode=False
                    )
            elif condition_type in ["tight", "soft"]:
                if is_vqvae:
                    self.action_head = ActionClassificationTransformerMoE(
                        action_dim=action_dim,
                        vlm_hidden_size=self.hidden_size,
                        num_actions=num_actions,
                        hidden_size=policy_hidden_size,
                        depth=policy_depth,
                        num_heads=policy_num_heads,
                        mlp_ratio=policy_mlp_ratio,
                        vqvae_mode=True,
                        vq_codebook_size=self.action_vqvae.codebook_size,
                        vq_latent_codes=self.action_vqvae.latent_codes,
                        gen_hidden_size=gen_hidden_dim,
                        vlm_layer_indices=self.vlm_layer_indices,
                    )
                else:
                    self.action_head = ActionClassificationTransformerMoE(
                        action_dim=action_dim,
                        vlm_hidden_size=self.hidden_size,
                        num_actions=num_actions,
                        num_bins=num_bins,
                        hidden_size=policy_hidden_size,
                        depth=policy_depth,
                        num_heads=policy_num_heads,
                        mlp_ratio=policy_mlp_ratio,
                        vqvae_mode=False,
                        gen_hidden_size=gen_hidden_dim,
                        vlm_layer_indices=self.vlm_layer_indices,
                    )
            else:
                raise NotImplementedError(f"Classification policy does not support {condition_type}.")
            self.noise_scheduler = None
        elif loss_type == "diffusion":
            if condition_type in ["tight", "soft"]:
                self.action_head = ActionDiffusionTransformerMoE(
                    action_dim=action_dim,
                    vlm_hidden_size=self.hidden_size,
                    hidden_size=policy_hidden_size,
                    depth=policy_depth,
                    num_heads=policy_num_heads,
                    mlp_ratio=policy_mlp_ratio,
                    gen_hidden_size=gen_hidden_dim,
                    vlm_layer_indices=self.vlm_layer_indices,
                )
            elif condition_type == "loose":
                self.action_head = ActionDiffusionTransformerMetaquery(
                    action_dim=action_dim,
                    condition_dim=self.hidden_size,
                    hidden_size=policy_hidden_size,
                    depth=policy_depth,
                    num_heads=policy_num_heads,
                    mlp_ratio=policy_mlp_ratio
                )
            else:
                raise ValueError(f"Unknown condition type for diffusion: {condition_type}")
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        if loss_type == "diffusion" and not self.use_vita_latent_flow and not self.use_vlanext_latent_flow:
            if scheduler_type == "ddim":
                self.noise_scheduler = DDIMScheduler(
                    num_train_timesteps=num_train_timesteps,
                    clip_sample=False,
                    prediction_type="epsilon"
                )
            elif scheduler_type == "flow_match": 
                self.noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=num_train_timesteps)
            else:
                raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        elif self.use_vita_latent_flow or self.use_vlanext_latent_flow:
            self.noise_scheduler = None

    def train(self, mode=True):
        super().train(mode)
        if self.action_effect_visual_teacher is not None:
            self.action_effect_visual_teacher.eval()
        return self

    def forward_action_vqvae_pretrain(self, actions):
        if self.action_vqvae is None:
            raise RuntimeError("Action VQ-VAE not initialized.")
            
        actions = actions.to(dtype=self.action_vqvae.in_proj.weight.dtype)
        loss = self.action_vqvae(actions)
        return loss

    def get_vlm_condition(self, input_ids, attention_mask, proprioception=None, proprio_attention_mask=None, pixel_values=None, pixel_values_videos=None, image_grid_thw=None, video_grid_thw=None, token_type_ids=None, mm_token_type_ids=None, return_token_type_ids=False, precomputed_image_embeds=None):
        if self.model_family == "paligemma":
            connector_out, hidden_states = self._get_vlm_condition_paligemma(input_ids, attention_mask, proprioception, proprio_attention_mask, pixel_values, token_type_ids=token_type_ids)
            return (connector_out, hidden_states, None) if return_token_type_ids else (connector_out, hidden_states)
        elif self.model_family == "llama":
            connector_out, hidden_states = self._get_vlm_condition_llama(input_ids, attention_mask, pixel_values, proprioception, proprio_attention_mask)
            return (connector_out, hidden_states, None) if return_token_type_ids else (connector_out, hidden_states)
        elif self.model_family == "qwen":
            return self._get_vlm_condition_qwen(input_ids, attention_mask, proprioception, proprio_attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, mm_token_type_ids=mm_token_type_ids, return_token_type_ids=return_token_type_ids, precomputed_image_embeds=precomputed_image_embeds)

    def _apply_vlm_layer_attention(self, hidden_states):
        if self.vlm_layer_attention is None:
            return hidden_states
        return self.vlm_layer_attention(hidden_states)

    def _encode_parallel_qwen_visual_tokens(
        self,
        pixel_values,
        image_grid_thw,
        future_pixel_values,
        future_image_grid_thw,
        batch_size,
    ):
        if self.model_family != "qwen":
            raise ValueError("Action-effect visual targets currently require a Qwen VLM.")
        if any(value is None for value in (
            pixel_values,
            image_grid_thw,
            future_pixel_values,
            future_image_grid_thw,
        )):
            raise ValueError("Action-effect training requires current and future image tensors.")

        current_image_count = int(image_grid_thw.shape[0])
        all_pixels = torch.cat([pixel_values, future_pixel_values], dim=0)
        all_grids = torch.cat([image_grid_thw, future_image_grid_thw], dim=0)
        vision_output = self.lmm.model.get_image_features(
            all_pixels,
            all_grids,
            return_dict=True,
        )
        all_embeds = tuple(vision_output.pooler_output)
        current_embeds = all_embeds[:current_image_count]
        future_embeds = all_embeds[current_image_count:]
        current_grids = image_grid_thw[:current_image_count]
        future_grids = future_image_grid_thw
        use_spatial_pooling = (
            self.vita_action_effect_visual_teacher_type == "qwen_spatial"
        )
        spatial_merge_size = int(self.lmm.model.visual.spatial_merge_size)

        def group_by_sample(
            image_embeds,
            image_grids,
            preserve_image_axis=False,
        ):
            if len(image_embeds) % batch_size != 0:
                raise ValueError(
                    f"Expected a fixed image count per sample, got {len(image_embeds)} images "
                    f"for batch size {batch_size}."
                )
            if len(image_embeds) != len(image_grids):
                raise ValueError("Qwen visual embeddings and image grids must align.")
            views = len(image_embeds) // batch_size
            if (
                preserve_image_axis
                and views != len(self.vita_action_effect_horizons)
            ):
                raise ValueError(
                    f"Expected {len(self.vita_action_effect_horizons)} future "
                    f"horizon images per sample, got {views}."
                )
            grouped = []
            for index in range(batch_size):
                sample_embeds = image_embeds[index * views : (index + 1) * views]
                sample_grids = image_grids[index * views : (index + 1) * views]
                if preserve_image_axis:
                    horizon_tokens = []
                    for sample_embed, sample_grid in zip(
                        sample_embeds,
                        sample_grids,
                    ):
                        horizon_tokens.append(
                            spatial_pool_qwen_visual_tokens(
                                sample_embed,
                                sample_grid,
                                spatial_merge_size=spatial_merge_size,
                                output_grid_size=(
                                    self.vita_action_effect_spatial_grid_size
                                ),
                            )
                        )
                    grouped.append(torch.stack(horizon_tokens, dim=0))
                    continue
                if self.vita_action_effect_main_view_only:
                    sample_embeds = sample_embeds[:1]
                    sample_grids = sample_grids[:1]
                if use_spatial_pooling:
                    if len(sample_embeds) != 1:
                        raise ValueError(
                            "Qwen spatial action-effect pooling requires one main view."
                        )
                    grouped.append(
                        spatial_pool_qwen_visual_tokens(
                            sample_embeds[0],
                            sample_grids[0],
                            spatial_merge_size=spatial_merge_size,
                            output_grid_size=self.vita_action_effect_spatial_grid_size,
                        )
                    )
                else:
                    grouped.append(torch.cat(sample_embeds, dim=0))
            if len({tokens.shape[0] for tokens in grouped}) != 1:
                raise ValueError("Action-effect visual resampling requires equal token counts per sample.")
            return torch.stack(grouped, dim=0)

        return (
            current_embeds,
            group_by_sample(current_embeds, current_grids),
            group_by_sample(
                future_embeds,
                future_grids,
                preserve_image_axis=bool(self.vita_action_effect_horizons),
            ),
        )

    def _encode_parallel_dinov2_visual_tokens(
        self,
        current_pixel_values,
        future_pixel_values,
        batch_size,
    ):
        teacher = self.action_effect_visual_teacher
        if teacher is None:
            raise RuntimeError(
                "The frozen DINOv2 teacher is not loaded. It is required only for "
                "action-effect training, not policy inference."
            )
        if current_pixel_values is None:
            raise ValueError("DINO action-effect training requires current images.")
        if current_pixel_values.shape[0] != batch_size:
            raise ValueError(
                "DINO action-effect image batch size must match the policy batch size."
            )
        if future_pixel_values is not None and future_pixel_values.shape[0] != batch_size:
            raise ValueError("DINO future image batch size must match the policy batch size.")

        teacher.eval()
        teacher_param = next(teacher.parameters())
        all_pixels = (
            torch.cat([current_pixel_values, future_pixel_values], dim=0)
            if future_pixel_values is not None
            else current_pixel_values
        ).to(
            device=teacher_param.device,
            dtype=teacher_param.dtype,
        )
        with torch.no_grad():
            output = teacher(pixel_values=all_pixels, return_dict=True)
            if self.vita_action_representation_type == "single_view_edar_lite":
                prefix_tokens = 1 + int(
                    getattr(teacher.config, "num_register_tokens", 0)
                )
                patch_tokens = output.last_hidden_state[:, prefix_tokens:]
                patch_size = getattr(teacher.config, "patch_size", 16)
                patch_size = int(
                    patch_size[0]
                    if isinstance(patch_size, (tuple, list))
                    else patch_size
                )
                patch_grid = 256 // patch_size
                if patch_tokens.shape[1] != patch_grid * patch_grid:
                    raise RuntimeError(
                        "EDAR teacher patch count does not match the deterministic "
                        f"256px preprocessing: expected {patch_grid * patch_grid}, "
                        f"got {patch_tokens.shape[1]}."
                    )
                visual_tokens = patch_tokens.reshape(
                    patch_tokens.shape[0],
                    patch_grid,
                    patch_grid,
                    patch_tokens.shape[-1],
                ).permute(0, 3, 1, 2)
                visual_tokens = F.adaptive_avg_pool2d(
                    visual_tokens,
                    (self.vita_action_effect_spatial_grid_size,) * 2,
                ).flatten(2).transpose(1, 2).detach()
            else:
                visual_tokens = output.last_hidden_state[:, :1].detach()
        current_tokens = visual_tokens[:batch_size]
        future_tokens = (
            visual_tokens[batch_size:]
            if future_pixel_values is not None
            else None
        )
        return current_tokens, future_tokens

    def _apply_latent_bridge(self, hidden_states, bridge_pixel_values=None, proprioception=None):
        if self.latent_bridge is None:
            return hidden_states
        return self.latent_bridge(
            hidden_states,
            bridge_images=bridge_pixel_values,
            proprioception=proprioception,
            policy_depth=self.policy_depth,
            layer_indices=self.vlm_layer_indices,
        )

    def _vlm_proprio(self, proprioception):
        if self.use_latent_bridge and not self.latent_bridge_vlm_uses_proprio:
            return None
        return proprioception

    def _expand_for_bridge_sequence(
        self,
        connector_out,
        hidden_states,
        hidden_token_type_ids,
        actions,
        proprioception,
        history_actions,
        bridge_pixel_values,
        progress_labels,
        action_progress_labels,
        goal_connector_out,
        goal_hidden_states,
        goal_hidden_token_type_ids,
        goal_distance_labels,
        future_images,
    ):
        if actions is None or actions.ndim != 4:
            return connector_out, hidden_states, hidden_token_type_ids, actions, proprioception, history_actions, bridge_pixel_values, progress_labels, action_progress_labels, goal_connector_out, goal_hidden_states, goal_hidden_token_type_ids, goal_distance_labels, future_images

        if future_images is not None:
            raise ValueError("Latent bridge sequence training does not currently support future image loss.")

        bsz, seq_len = actions.shape[:2]

        def expand_batch(tensor):
            if tensor is None:
                return None
            return tensor.unsqueeze(1).expand(bsz, seq_len, *tensor.shape[1:]).reshape(bsz * seq_len, *tensor.shape[1:])

        connector_out = expand_batch(connector_out)
        hidden_token_type_ids = expand_batch(hidden_token_type_ids)
        if isinstance(hidden_states, tuple):
            hidden_states = tuple(expand_batch(h) for h in hidden_states)
        else:
            hidden_states = expand_batch(hidden_states)
        goal_connector_out = expand_batch(goal_connector_out)
        goal_hidden_token_type_ids = expand_batch(goal_hidden_token_type_ids)
        if isinstance(goal_hidden_states, tuple):
            goal_hidden_states = tuple(expand_batch(h) for h in goal_hidden_states)
        else:
            goal_hidden_states = expand_batch(goal_hidden_states)

        actions = actions.reshape(bsz * seq_len, *actions.shape[2:])
        if proprioception is not None:
            proprioception = proprioception.reshape(bsz * seq_len, *proprioception.shape[2:])
        if history_actions is not None:
            history_actions = history_actions.reshape(bsz * seq_len, *history_actions.shape[2:])
        if bridge_pixel_values is not None:
            bridge_pixel_values = bridge_pixel_values.reshape(bsz * seq_len, *bridge_pixel_values.shape[2:])
        if progress_labels is not None:
            progress_labels = progress_labels.reshape(bsz * seq_len)
        if action_progress_labels is not None:
            action_progress_labels = action_progress_labels.reshape(bsz * seq_len)
        if goal_distance_labels is not None:
            goal_distance_labels = goal_distance_labels.reshape(bsz * seq_len)

        return connector_out, hidden_states, hidden_token_type_ids, actions, proprioception, history_actions, bridge_pixel_values, progress_labels, action_progress_labels, goal_connector_out, goal_hidden_states, goal_hidden_token_type_ids, goal_distance_labels, future_images

    def _get_vlm_condition_qwen(self, input_ids, attention_mask, proprioception, proprio_attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, mm_token_type_ids=None, return_token_type_ids=False, precomputed_image_embeds=None):
        B = input_ids.shape[0]
        proprio_token_count = int(proprioception.shape[1]) if (self.use_proprio_input_vlm and proprioception is not None) else 0
        
        backbone = self.lmm.model
        lmm_config = self.lmm.config
        pad_token_id = getattr(lmm_config, "pad_token_id", None)
        pad_token_id = pad_token_id if pad_token_id is not None else 0
        inputs_embeds = backbone.get_input_embeddings()(input_ids)
        
        if self.use_proprio_input_vlm and proprioception is not None:
            proprio_embeds = self.action_projector(proprioception.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype))
            inputs_embeds = torch.cat([proprio_embeds, inputs_embeds], dim=1)
            if attention_mask is not None:
                if proprio_attention_mask is not None:
                    proprio_mask = proprio_attention_mask.to(device=attention_mask.device, dtype=attention_mask.dtype)
                else:
                    proprio_mask = torch.ones(B, proprioception.shape[1], device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([proprio_mask, attention_mask], dim=1)
            proprio_ids = torch.full((B, proprioception.shape[1]), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
            input_ids = torch.cat([proprio_ids, input_ids], dim=1)
            if mm_token_type_ids is not None:
                proprio_type_ids = torch.zeros(B, proprioception.shape[1], device=mm_token_type_ids.device, dtype=mm_token_type_ids.dtype)
                mm_token_type_ids = torch.cat([proprio_type_ids, mm_token_type_ids], dim=1)

        if self.condition_type != "tight":
            queries_embeds = self.meta_queries.unsqueeze(0).expand(B, -1, -1).to(inputs_embeds.dtype)
            inputs_embeds = torch.cat([inputs_embeds, queries_embeds], dim=1)
            if attention_mask is not None:
                queries_mask = torch.ones(B, self.num_queries, device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([attention_mask, queries_mask], dim=1)
            queries_ids = torch.full((B, self.num_queries), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
            extended_input_ids = torch.cat([input_ids, queries_ids], dim=1)
            if mm_token_type_ids is not None:
                queries_type_ids = torch.zeros(B, self.num_queries, device=mm_token_type_ids.device, dtype=mm_token_type_ids.dtype)
                mm_token_type_ids = torch.cat([mm_token_type_ids, queries_type_ids], dim=1)
        else:
            extended_input_ids = input_ids

        forward_pixel_values = pixel_values
        if precomputed_image_embeds is not None:
            image_embeds = torch.cat(tuple(precomputed_image_embeds), dim=0).to(
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            image_mask, _ = backbone.get_placeholder_mask(
                extended_input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            forward_pixel_values = None

        if mm_token_type_ids is None and (image_grid_thw is not None or video_grid_thw is not None):
            mm_token_type_ids = torch.zeros_like(extended_input_ids)
            image_token_id = getattr(lmm_config, "image_token_id", None)
            video_token_id = getattr(lmm_config, "video_token_id", None)
            if image_token_id is not None:
                mm_token_type_ids = mm_token_type_ids.masked_fill(extended_input_ids == image_token_id, 1)
            if video_token_id is not None:
                mm_token_type_ids = mm_token_type_ids.masked_fill(extended_input_ids == video_token_id, 2)

        rope_kwargs = {
            "input_ids": extended_input_ids,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "attention_mask": attention_mask
        }
        if mm_token_type_ids is not None:
            rope_kwargs["mm_token_type_ids"] = mm_token_type_ids

        position_ids, _ = backbone.get_rope_index(**rope_kwargs)

        output_hidden_states_flag = (self.enable_future_image_loss or self.condition_type in ["tight", "soft"])
        forward_kwargs = {
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "pixel_values": forward_pixel_values,
            "pixel_values_videos": pixel_values_videos,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "output_hidden_states": output_hidden_states_flag,
        }
        outputs = self._run_qwen_backbone(backbone, forward_kwargs)
        hidden_states = outputs.hidden_states if output_hidden_states_flag else None
        hidden_states = self._apply_vlm_layer_attention(hidden_states)
        hidden_token_type_ids = None
        if return_token_type_ids:
            if mm_token_type_ids is not None:
                hidden_token_type_ids = mm_token_type_ids.clone()
            else:
                hidden_token_type_ids = torch.zeros_like(extended_input_ids)
            if proprio_token_count > 0:
                hidden_token_type_ids[:, :proprio_token_count] = 4
            if self.condition_type != "tight":
                hidden_token_type_ids[:, -self.num_queries:] = 3
            if attention_mask is not None:
                hidden_token_type_ids = hidden_token_type_ids.masked_fill(
                    attention_mask.to(device=hidden_token_type_ids.device) == 0,
                    -1,
                )
        connector_out = None
        if self.condition_type == "loose" and self.connector is not None:
            query_outputs = outputs.last_hidden_state[:, -self.num_queries:, :]
            connector_out = self.connector(query_outputs)
            
        if return_token_type_ids:
            return connector_out, hidden_states, hidden_token_type_ids
        return connector_out, hidden_states

    def _run_qwen_backbone(self, backbone, forward_kwargs):
        if self.vlm_truncate_layers is None:
            return backbone(**forward_kwargs)
        language_model = getattr(backbone, "language_model", None)
        layers = getattr(language_model, "layers", None)
        if language_model is None or layers is None:
            return backbone(**forward_kwargs)
        num_layers = len(layers)
        truncate_layers = int(self.vlm_truncate_layers)
        if truncate_layers <= 0 or truncate_layers > num_layers:
            raise ValueError(
                f"vlm_truncate_layers must be in [1, {num_layers}], got {truncate_layers}."
            )
        if truncate_layers == num_layers:
            return backbone(**forward_kwargs)

        original_layers = language_model.layers
        language_model.layers = nn.ModuleList(list(original_layers[:truncate_layers]))
        try:
            return backbone(**forward_kwargs)
        finally:
            language_model.layers = original_layers

    def _get_vlm_condition_llama(self, input_ids, attention_mask, pixel_values, proprioception, proprio_attention_mask):
        B = input_ids.shape[0]
        pixel_values = pixel_values.to(dtype=self.vision_encoder.dtype)
        
        vision_outputs = self.vision_encoder(pixel_values, output_hidden_states=True)
        image_feats = vision_outputs.last_hidden_state
        image_embeds = self.vision_projector(image_feats)

        if image_embeds.shape[0] != B:
            num_views = image_embeds.shape[0] // B
            image_embeds = image_embeds.view(B, num_views, -1, image_embeds.shape[-1])
            image_embeds = image_embeds.flatten(1, 2)
        
        text_embeds = self.lmm.model.embed_tokens(input_ids)

        proprio_embeds = None
        if self.use_proprio_input_vlm and proprioception is not None:
             proprio_embeds = self.action_projector(proprioception.to(device=text_embeds.device, dtype=text_embeds.dtype))

        embeds_list = [image_embeds]
        image_mask = torch.ones(B, image_embeds.shape[1], device=attention_mask.device, dtype=attention_mask.dtype)
        mask_list = [image_mask]

        if proprio_embeds is not None:
            embeds_list.append(proprio_embeds)
            if proprio_attention_mask is not None:
                mask_list.append(proprio_attention_mask.to(attention_mask.device))
            else:
                p_mask = torch.ones(B, proprio_embeds.shape[1], device=attention_mask.device, dtype=attention_mask.dtype)
                mask_list.append(p_mask)
        
        embeds_list.append(text_embeds)
        mask_list.append(attention_mask)

        if self.condition_type != "tight":
            queries_embeds = self.meta_queries.unsqueeze(0).expand(B, -1, -1).to(text_embeds.dtype)
            embeds_list.append(queries_embeds)
            queries_mask = torch.ones(B, self.num_queries, device=attention_mask.device, dtype=attention_mask.dtype)
            mask_list.append(queries_mask)

        inputs_embeds = torch.cat(embeds_list, dim=1)
        combined_attention_mask = torch.cat(mask_list, dim=1)

        output_hidden_states_flag = (self.enable_future_image_loss or self.condition_type in ["tight", "soft"])
        outputs = self.lmm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
            output_hidden_states=output_hidden_states_flag
        )
        hidden_states = outputs.hidden_states if output_hidden_states_flag else None
        hidden_states = self._apply_vlm_layer_attention(hidden_states)
        connector_out = None
        if self.condition_type == "loose" and self.connector is not None:
            query_outputs = outputs.last_hidden_state[:, -self.num_queries:, :]
            connector_out = self.connector(query_outputs)

        return connector_out, hidden_states

    def _get_vlm_condition_paligemma(self, input_ids, attention_mask, proprioception, proprio_attention_mask, pixel_values, token_type_ids=None):
        from transformers.models.paligemma.modeling_paligemma import create_causal_mask_mapping

        B = input_ids.shape[0]

        backbone = self.lmm.model

        inputs_embeds = backbone.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_outputs = backbone.get_image_features(pixel_values)
            image_features = image_outputs.pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = backbone.get_placeholder_mask(input_ids, inputs_embeds, image_features)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        if self.use_proprio_input_vlm and proprioception is not None:
            proprio_embeds = self.action_projector(proprioception.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype))
            inputs_embeds = torch.cat([proprio_embeds, inputs_embeds], dim=1)
            if attention_mask is not None:
                if proprio_attention_mask is not None:
                    proprio_mask = proprio_attention_mask.to(device=attention_mask.device, dtype=attention_mask.dtype)
                else:
                    proprio_mask = torch.ones(B, proprioception.shape[1], device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([proprio_mask, attention_mask], dim=1)
            # Proprio tokens are prefix context — token_type_ids=0 (bidirectional)
            if token_type_ids is not None:
                proprio_type_ids = torch.zeros(B, proprioception.shape[1], device=token_type_ids.device, dtype=token_type_ids.dtype)
                token_type_ids = torch.cat([proprio_type_ids, token_type_ids], dim=1)

        if self.condition_type != "tight":
            queries_embeds = self.meta_queries.unsqueeze(0).expand(B, -1, -1).to(inputs_embeds.dtype)
            inputs_embeds = torch.cat([inputs_embeds, queries_embeds], dim=1)
            if attention_mask is not None:
                queries_mask = torch.ones(B, self.num_queries, device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([attention_mask, queries_mask], dim=1)
            # Query tokens are suffix — token_type_ids=1 (causal)
            if token_type_ids is not None:
                queries_type_ids = torch.ones(B, self.num_queries, device=token_type_ids.device, dtype=token_type_ids.dtype)
                token_type_ids = torch.cat([token_type_ids, queries_type_ids], dim=1)

        # Build the proper PaliGemma causal mask with bidirectional attention on prefix/image tokens
        cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0) + 1  # PaliGemma positions are 1-indexed
        causal_mask_mapping = create_causal_mask_mapping(
            backbone.config,
            inputs_embeds,
            attention_mask,
            cache_position,
            past_key_values=None,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            is_training=self.training,
        )

        output_hidden_states_flag = (self.enable_future_image_loss or self.condition_type in ["tight", "soft"] )
        outputs = backbone.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states_flag,
        )
        hidden_states = outputs.hidden_states if output_hidden_states_flag else None
        hidden_states = self._apply_vlm_layer_attention(hidden_states)
        connector_out = None
        if self.condition_type == "loose" and self.connector is not None:
            query_outputs = outputs.last_hidden_state[:, -self.num_queries:, :]
            connector_out = self.connector(query_outputs)

        return connector_out, hidden_states

    def _compute_gen_loss_and_feats(self, future_images, vlm_hidden_states):
        with torch.no_grad():
            future_images = future_images.to(device=self.vq_model.device, dtype=self.vq_model.dtype)
            _, _, (_, _, token_ids) = self.vq_model.encode(future_images)
            B = future_images.shape[0]
            token_ids = token_ids.view(B, -1)
        
        sos_token = torch.zeros((B, 1), dtype=token_ids.dtype, device=token_ids.device)
        gen_input = torch.cat([sos_token, token_ids[:, :-1]], dim=1)
        
        gen_logits, gen_hidden_states = self.generator(gen_input, vlm_hidden_states)
        loss_img = F.cross_entropy(gen_logits.reshape(-1, self.vq_codebook_size), token_ids.reshape(-1))
        
        return loss_img, gen_hidden_states

    def _compute_dct_loss(self, pred, target):
        B, T, D = pred.shape

        if not hasattr(self, '_dct_matrix') or self._dct_matrix.shape[0] != T or self._dct_matrix.device != pred.device:
            n = torch.arange(T, device=pred.device).float()
            k = torch.arange(T, device=pred.device).float()
            dct_m = torch.cos((np.pi / T) * (n + 0.5).unsqueeze(0) * k.unsqueeze(1))
            
            dct_m[0, :] *= 1.0 / np.sqrt(T)
            dct_m[1:, :] *= np.sqrt(2.0 / T)
            
            self._dct_matrix = dct_m

        split_idx = max(1, int(T * self.dct_freq_split))
        freq_weights = torch.ones(T, device=pred.device, dtype=pred.dtype)
        freq_weights[:split_idx] = self.dct_low_freq_weight
        freq_weights[split_idx:] = self.dct_high_freq_weight
        freq_weights = freq_weights.view(1, T, 1)

        pred_perm = pred.permute(0, 2, 1)
        pred_dct = torch.matmul(pred_perm, self._dct_matrix.t())
        pred_dct = pred_dct.permute(0, 2, 1)

        target_perm = target.permute(0, 2, 1)
        target_dct = torch.matmul(target_perm, self._dct_matrix.t())
        target_dct = target_dct.permute(0, 2, 1)

        sim_type = self.dct_similarity_type
        if sim_type == "mse":
            diff = (pred_dct - target_dct) ** 2
            return (diff * freq_weights).mean()
        elif sim_type == "mae":
            diff = (pred_dct - target_dct).abs()
            return (diff * freq_weights).mean()
        elif sim_type == "cosine":
            pred_norm = torch.nn.functional.normalize(pred_dct, dim=-1)
            target_norm = torch.nn.functional.normalize(target_dct, dim=-1)
            cos_sim = (pred_norm * target_norm).sum(dim=-1, keepdim=True)
            cos_dist = 1.0 - cos_sim
            return (cos_dist * freq_weights).mean()
        else:
            raise ValueError(f"Unknown dct_similarity_type: {sim_type!r}. "
                             f"Options are: 'mse', 'mae', 'cosine'.")

    def _progress_condition_features(self, connector_out, hidden_states):
        if self.condition_type in ["tight", "soft"]:
            return hidden_states
        if self.condition_type == "loose":
            return connector_out.mean(dim=1)
        raise ValueError(f"Unknown condition type: {self.condition_type}")

    def _apply_progress(self, features, progress_labels=None):
        if not self.use_progress_head:
            return features, None

        progress_features = features[-1] if isinstance(features, tuple) else features
        progress_pred = self.progress_head(progress_features).float()
        progress_loss = None
        if progress_labels is not None and self.progress_loss_weight > 0:
            target = progress_labels.float()
            if self.training and self.progress_noise > 0:
                noise = torch.empty_like(target).uniform_(-self.progress_noise, self.progress_noise)
                target = torch.clamp(target + noise, 0.0, 1.0)
            progress_loss = F.mse_loss(progress_pred.float(), target)

        if self.use_progress_film:
            film_progress = progress_pred.detach() if self.progress_detach else progress_pred
            if isinstance(features, tuple):
                filmed_last = self.progress_film(features[-1], film_progress)
                features = tuple(list(features[:-1]) + [filmed_last])
            else:
                features = self.progress_film(features, film_progress)
        return features, progress_loss

    def _add_progress_loss(self, loss, progress_loss):
        if progress_loss is not None and self.progress_loss_weight > 0:
            return loss + self.progress_loss_weight * progress_loss.to(dtype=loss.dtype)
        return loss

    def _vita_reconstruction_loss(self, pred, target, action_mask=None):
        return masked_action_reconstruction_loss(
            pred,
            target,
            loss_type=self.vita_recon_loss_type,
            mask=action_mask,
        )

    @torch.no_grad()
    def _sample_action_effect_negative_actions(self, actions):
        """Sample dissimilar raw action chunks from this batch and the FIFO queue."""
        if actions.ndim != 3:
            raise ValueError("Action-effect queue expects actions with shape [B, H, D].")

        batch_actions = actions.detach()
        pool_parts = [batch_actions]
        if self._action_effect_queue_count > 0:
            pool_parts.append(
                self._action_effect_raw_action_queue[
                    :self._action_effect_queue_count
                ].to(device=actions.device, dtype=actions.dtype)
            )
        action_pool = torch.cat(pool_parts, dim=0)

        current_flat = batch_actions.float().flatten(1)
        pool_flat = action_pool.float().flatten(1)
        action_distance = (
            (current_flat[:, None, :] - pool_flat[None, :, :])
            .square()
            .mean(dim=-1)
            .sqrt()
        )
        valid = action_distance > self.vita_action_effect_negative_min_action_distance
        batch_indices = torch.arange(actions.shape[0], device=actions.device)
        valid[batch_indices, batch_indices] = False

        selected_count = min(
            self.vita_action_effect_num_negatives,
            action_pool.shape[0],
        )
        random_scores = torch.rand_like(action_distance)
        random_scores.masked_fill_(~valid, -1.0)
        selected_indices = random_scores.topk(selected_count, dim=1).indices
        selected_valid = valid.gather(1, selected_indices)
        selected_distance = action_distance.gather(1, selected_indices)
        selected_actions = action_pool[selected_indices]

        queue_capacity = self.vita_action_effect_negative_queue_size
        queue_fill_ratio = (
            float(self._action_effect_queue_count) / float(queue_capacity)
            if queue_capacity > 0
            else 0.0
        )
        metrics = {
            "candidate_negatives_per_sample": valid.float().sum(dim=1).mean(),
            "queue_fill_ratio": action_distance.new_tensor(queue_fill_ratio),
        }
        return selected_actions, selected_valid, selected_distance, metrics

    @torch.no_grad()
    def _enqueue_action_effect_actions(self, actions):
        capacity = self.vita_action_effect_negative_queue_size
        if capacity <= 0:
            return
        values = actions.detach().to(
            device=self._action_effect_raw_action_queue.device,
            dtype=self._action_effect_raw_action_queue.dtype,
        )
        if values.shape[1:] != self._action_effect_raw_action_queue.shape[1:]:
            raise ValueError(
                "Queued action shape does not match configured action horizon/dimension."
            )
        if values.shape[0] >= capacity:
            self._action_effect_raw_action_queue.copy_(values[-capacity:])
            self._action_effect_queue_count = capacity
            self._action_effect_queue_write_index = 0
            return

        write_index = self._action_effect_queue_write_index
        first_count = min(values.shape[0], capacity - write_index)
        self._action_effect_raw_action_queue[
            write_index:write_index + first_count
        ].copy_(values[:first_count])
        remaining = values.shape[0] - first_count
        if remaining > 0:
            self._action_effect_raw_action_queue[:remaining].copy_(
                values[first_count:]
            )
        self._action_effect_queue_write_index = (write_index + values.shape[0]) % capacity
        self._action_effect_queue_count = min(
            capacity,
            self._action_effect_queue_count + values.shape[0],
        )

    def _apply_vita_zobs_mode(self, observation_latent):
        if self.vita_zobs_mode == "normal":
            return observation_latent
        if self.vita_zobs_mode == "noise":
            return torch.randn_like(observation_latent) * self.vita_zobs_noise_scale
        raise ValueError(f"Unknown vita_zobs_mode: {self.vita_zobs_mode}.")

    def _vita_goal_distance_loss(
        self,
        observation_latent,
        goal_connector_out=None,
        goal_hidden_states=None,
        goal_hidden_token_type_ids=None,
        goal_distance_labels=None,
    ):
        if self.vita_goal_metric_head is None or goal_distance_labels is None:
            return None, None

        if self.condition_type in ["tight", "soft"]:
            if goal_hidden_states is None:
                raise ValueError("Goal-distance regression requires goal hidden states.")
            goal_latent = self.vita_action_generator.encode_observation(
                hidden_states=goal_hidden_states,
                hidden_token_type_ids=goal_hidden_token_type_ids,
                proprioception=None,
                return_condition_tokens=False,
            )
        elif self.condition_type == "loose":
            if goal_connector_out is None:
                raise ValueError("Goal-distance regression requires goal connector output.")
            goal_latent = self.vita_action_generator.encode_observation(
                connector_out=goal_connector_out,
                proprioception=None,
                return_condition_tokens=False,
            )
        else:
            raise ValueError(f"Unknown condition type: {self.condition_type}")

        observation_metric_latent = self.vita_action_generator.flatten_action_latent(
            observation_latent
        )
        goal_metric_latent = self.vita_action_generator.flatten_action_latent(goal_latent)
        obs_metric = F.normalize(
            self.vita_goal_metric_head(observation_metric_latent).float(), dim=-1
        )
        goal_metric = F.normalize(
            self.vita_goal_metric_head(goal_metric_latent).float(), dim=-1
        )
        pred_distance = torch.linalg.vector_norm(obs_metric - goal_metric, ord=2, dim=-1)
        target_distance = goal_distance_labels.float().detach()
        if self.vita_goal_distance_loss_type == "huber":
            loss = F.smooth_l1_loss(pred_distance, target_distance)
        elif self.vita_goal_distance_loss_type == "mse":
            loss = F.mse_loss(pred_distance, target_distance)
        else:
            raise ValueError(
                f"Unknown vita_goal_distance_loss_type: {self.vita_goal_distance_loss_type}. "
                "Expected 'huber' or 'mse'."
            )
        return loss, pred_distance

    def set_vita_adaptive_mq_step(self, step):
        if self.vita_action_generator is not None:
            self.vita_action_generator.set_adaptive_mq_step(step)
        self.vita_stage_b_global_step = max(int(step), 0)

    def forward(self, input_ids=None, attention_mask=None, actions=None, action_mask=None, proprioception=None, history_actions=None, proprio_attention_mask=None, pixel_values=None, pixel_values_videos=None, image_grid_thw=None, video_grid_thw=None, future_images=None, progress_labels=None, action_progress_labels=None, goal_distance_labels=None, return_loss_dict=False, task=None, token_type_ids=None, mm_token_type_ids=None, bridge_pixel_values=None, goal_input_ids=None, goal_attention_mask=None, goal_pixel_values=None, goal_pixel_values_videos=None, goal_image_grid_thw=None, goal_video_grid_thw=None, goal_token_type_ids=None, goal_mm_token_type_ids=None, future_input_ids=None, future_attention_mask=None, future_pixel_values=None, future_pixel_values_videos=None, future_image_grid_thw=None, future_video_grid_thw=None, future_token_type_ids=None, future_mm_token_type_ids=None, action_effect_teacher_current_pixel_values=None, action_effect_teacher_future_pixel_values=None):
        if task == "action_vqvae_pretrain":
            return self.forward_action_vqvae_pretrain(actions)
            
        if self.loss_type == "regression":
            return self._forward_regression(
                input_ids, attention_mask, actions, proprioception, history_actions, proprio_attention_mask,
                pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, future_images, progress_labels=progress_labels, token_type_ids=token_type_ids, mm_token_type_ids=mm_token_type_ids, bridge_pixel_values=bridge_pixel_values
            )
        elif self.loss_type == "classification":
            return self._forward_classification(
                input_ids, attention_mask, actions, proprioception, history_actions, proprio_attention_mask,
                pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, future_images, progress_labels=progress_labels, token_type_ids=token_type_ids, mm_token_type_ids=mm_token_type_ids, bridge_pixel_values=bridge_pixel_values
            )
        elif self.loss_type == "diffusion":
            return self._forward_diffusion(
                input_ids, attention_mask, actions, proprioception, history_actions, proprio_attention_mask,
                pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, future_images, progress_labels=progress_labels, action_progress_labels=action_progress_labels, goal_distance_labels=goal_distance_labels, return_loss_dict=return_loss_dict, token_type_ids=token_type_ids, mm_token_type_ids=mm_token_type_ids, bridge_pixel_values=bridge_pixel_values, goal_input_ids=goal_input_ids, goal_attention_mask=goal_attention_mask, goal_pixel_values=goal_pixel_values, goal_pixel_values_videos=goal_pixel_values_videos, goal_image_grid_thw=goal_image_grid_thw, goal_video_grid_thw=goal_video_grid_thw, goal_token_type_ids=goal_token_type_ids, goal_mm_token_type_ids=goal_mm_token_type_ids, future_input_ids=future_input_ids, future_attention_mask=future_attention_mask, future_pixel_values=future_pixel_values, future_pixel_values_videos=future_pixel_values_videos, future_image_grid_thw=future_image_grid_thw, future_video_grid_thw=future_video_grid_thw, future_token_type_ids=future_token_type_ids, future_mm_token_type_ids=future_mm_token_type_ids, action_effect_teacher_current_pixel_values=action_effect_teacher_current_pixel_values, action_effect_teacher_future_pixel_values=action_effect_teacher_future_pixel_values, action_mask=action_mask
            )

    def _forward_classification(self, input_ids, attention_mask, actions, proprioception, history_actions, proprio_attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, future_images=None, progress_labels=None, token_type_ids=None, mm_token_type_ids=None, bridge_pixel_values=None):
        connector_out, hidden_states = self.get_vlm_condition(
            input_ids, attention_mask, proprioception=self._vlm_proprio(proprioception), proprio_attention_mask=proprio_attention_mask,
            pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
            token_type_ids=token_type_ids, mm_token_type_ids=mm_token_type_ids
        )
        
        loss_img = 0.0
        gen_hidden_states = None
        if self.enable_future_image_loss and future_images is not None:
             loss_img, gen_hidden_states = self._compute_gen_loss_and_feats(future_images, hidden_states)

        policy_history = history_actions if self.use_action_input_policy else None
        progress_loss = None
        
        if self.condition_type in ["tight", "soft"]:
            hidden_states = self._apply_latent_bridge(hidden_states, bridge_pixel_values, proprioception)
            policy_features, progress_loss = self._apply_progress(hidden_states, progress_labels)
            if self.enable_future_image_loss:
                pred_logits = self.action_head(policy_features, history_actions=policy_history, gen_hidden_states=gen_hidden_states)
            else:
                pred_logits = self.action_head(policy_features, history_actions=policy_history)
        elif self.condition_type == "loose":
            cond_input = connector_out.mean(dim=1)
            cond_input, progress_loss = self._apply_progress(cond_input, progress_labels)
            pred_logits = self.action_head(cond_input, history_actions=policy_history)
        else:
            raise ValueError(f"Unknown condition type: {self.condition_type}")
        
        if actions.ndim == 2: actions = actions.unsqueeze(1)

        pred_action_continuous = None
        loss = 0.0

        if self.action_vqvae is not None:
             with torch.no_grad():
                 self.action_vqvae.eval()
                 actions_input = actions.to(dtype=self.action_vqvae.in_proj.weight.dtype)
                 _, indices, _ = self.action_vqvae.encode(actions_input)
             
             loss = F.cross_entropy(
                 pred_logits.reshape(-1, self.action_vqvae.codebook_size), 
                 indices.reshape(-1)
             )

             if self.dct_loss_weight > 0:
                 probs = F.softmax(pred_logits, dim=-1)
                 pred_action_continuous = self.action_vqvae.decode_probs(probs)
        else:
            logits = pred_logits
            pose_logits = logits[:, :, :self.action_dim - 1, :]
            gripper_logits = logits[:, :, -1:, :2]
            
            gt_pose = torch.clamp(actions[:, :, :6], -1, 1)
            gt_pose_idx = ((gt_pose + 1) / 2 * (self.num_bins - 1)).round().long()
            
            gt_gripper = torch.clamp(actions[:, :, 6:7], -1, 1)
            gt_gripper_idx = ((gt_gripper + 1) / 2).round().long() # 0 or 1

            loss_pose = F.cross_entropy(pose_logits.reshape(-1, self.num_bins), gt_pose_idx.reshape(-1))
            loss_gripper = F.cross_entropy(gripper_logits.reshape(-1, 2), gt_gripper_idx.reshape(-1))
            
            loss = (loss_pose + loss_gripper) / 2.0

            if self.dct_loss_weight > 0:
                pose_probs = F.softmax(pose_logits, dim=-1)
                bin_centers = torch.linspace(-1, 1, self.num_bins, device=actions.device, dtype=pose_probs.dtype)
                pred_pose = torch.sum(pose_probs * bin_centers, dim=-1)

                gripper_probs = F.softmax(gripper_logits, dim=-1)
                p1 = gripper_probs[..., 1]
                pred_gripper = -1.0 + 2.0 * p1
                
                pred_action_continuous = torch.cat([pred_pose, pred_gripper], dim=-1)

        if self.dct_loss_weight > 0 and pred_action_continuous is not None:
             loss_dct = self._compute_dct_loss(pred_action_continuous.float(), actions.float())
             loss = loss + self.dct_loss_weight * loss_dct
        
        if self.future_image_loss_weight > 0:
            loss = loss + self.future_image_loss_weight * loss_img
        loss = self._add_progress_loss(loss, progress_loss)
        return loss

    def _forward_regression(self, input_ids, attention_mask, actions, proprioception, history_actions, proprio_attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, future_images=None, progress_labels=None, token_type_ids=None, mm_token_type_ids=None, bridge_pixel_values=None):
        connector_out, hidden_states = self.get_vlm_condition(
            input_ids, attention_mask, proprioception=self._vlm_proprio(proprioception), proprio_attention_mask=proprio_attention_mask,
            pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
            token_type_ids=token_type_ids, mm_token_type_ids=mm_token_type_ids
        )

        loss_img = 0.0
        gen_hidden_states = None
        if self.enable_future_image_loss and future_images is not None:
             loss_img, gen_hidden_states = self._compute_gen_loss_and_feats(future_images, hidden_states)
        
        policy_history = history_actions if self.use_action_input_policy else None
        progress_loss = None
        
        if self.condition_type in ["tight", "soft"]:
             hidden_states = self._apply_latent_bridge(hidden_states, bridge_pixel_values, proprioception)
             policy_features, progress_loss = self._apply_progress(hidden_states, progress_labels)
             if self.enable_future_image_loss:
                 pred_actions = self.action_head(policy_features, history_actions=policy_history, gen_hidden_states=gen_hidden_states)
             else:
                 pred_actions = self.action_head(policy_features, history_actions=policy_history)
        elif self.condition_type == "loose":
             cond_input = connector_out.mean(dim=1)
             cond_input, progress_loss = self._apply_progress(cond_input, progress_labels)
             pred_actions = self.action_head(cond_input, history_actions=policy_history)
        else:
             raise ValueError(f"Unknown condition type: {self.condition_type}")

        if actions.ndim == 2: actions = actions.unsqueeze(1)
        loss = F.mse_loss(pred_actions, actions)

        if self.dct_loss_weight > 0:
            loss_dct = self._compute_dct_loss(pred_actions.float(), actions.float())
            loss = loss + self.dct_loss_weight * loss_dct
            
        if self.future_image_loss_weight > 0:
            loss = loss + self.future_image_loss_weight * loss_img
        loss = self._add_progress_loss(loss, progress_loss)
        return loss

    def _forward_diffusion(self, input_ids, attention_mask, actions, proprioception, history_actions, proprio_attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, future_images=None, progress_labels=None, action_progress_labels=None, goal_distance_labels=None, return_loss_dict=False, token_type_ids=None, mm_token_type_ids=None, bridge_pixel_values=None, goal_input_ids=None, goal_attention_mask=None, goal_pixel_values=None, goal_pixel_values_videos=None, goal_image_grid_thw=None, goal_video_grid_thw=None, goal_token_type_ids=None, goal_mm_token_type_ids=None, future_input_ids=None, future_attention_mask=None, future_pixel_values=None, future_pixel_values_videos=None, future_image_grid_thw=None, future_video_grid_thw=None, future_token_type_ids=None, future_mm_token_type_ids=None, action_effect_teacher_current_pixel_values=None, action_effect_teacher_future_pixel_values=None, action_mask=None):
        precomputed_image_embeds = None
        current_visual_tokens = None
        future_visual_tokens = None
        if self.use_vita_latent_flow and self.vita_action_effect_enabled:
            if pixel_values_videos is not None or future_pixel_values_videos is not None:
                raise ValueError("Action-effect training currently supports image inputs only.")
            if self.vita_action_effect_visual_teacher_type in {
                "qwen",
                "qwen_spatial",
            }:
                precomputed_image_embeds, current_visual_tokens, future_visual_tokens = (
                    self._encode_parallel_qwen_visual_tokens(
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        future_pixel_values=future_pixel_values,
                        future_image_grid_thw=future_image_grid_thw,
                        batch_size=input_ids.shape[0],
                    )
                )
            else:
                current_visual_tokens, future_visual_tokens = (
                    self._encode_parallel_dinov2_visual_tokens(
                        current_pixel_values=action_effect_teacher_current_pixel_values,
                        future_pixel_values=action_effect_teacher_future_pixel_values,
                        batch_size=input_ids.shape[0],
                    )
                )
        if self.use_vita_latent_flow:
            connector_out, hidden_states, hidden_token_type_ids = self.get_vlm_condition(
                input_ids, attention_mask, proprioception=self._vlm_proprio(proprioception), proprio_attention_mask=proprio_attention_mask,
                pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
                token_type_ids=token_type_ids, mm_token_type_ids=mm_token_type_ids,
                return_token_type_ids=True,
                precomputed_image_embeds=precomputed_image_embeds,
            )
        else:
            connector_out, hidden_states = self.get_vlm_condition(
                input_ids, attention_mask, proprioception=self._vlm_proprio(proprioception), proprio_attention_mask=proprio_attention_mask,
                pixel_values=pixel_values, pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
                token_type_ids=token_type_ids, mm_token_type_ids=mm_token_type_ids
            )
            hidden_token_type_ids = None
        goal_connector_out = None
        goal_hidden_states = None
        goal_hidden_token_type_ids = None
        if self.vita_goal_metric_head is not None and goal_distance_labels is not None:
            if goal_input_ids is None or goal_attention_mask is None:
                raise ValueError("Goal-distance regression requires goal VLM inputs.")
            goal_context = torch.no_grad() if self.vita_goal_distance_detach_goal else nullcontext()
            with goal_context:
                goal_connector_out, goal_hidden_states, goal_hidden_token_type_ids = self.get_vlm_condition(
                    goal_input_ids,
                    goal_attention_mask,
                    proprioception=None,
                    proprio_attention_mask=None,
                    pixel_values=goal_pixel_values,
                    pixel_values_videos=goal_pixel_values_videos,
                    image_grid_thw=goal_image_grid_thw,
                    video_grid_thw=goal_video_grid_thw,
                    token_type_ids=goal_token_type_ids,
                    mm_token_type_ids=goal_mm_token_type_ids,
                    return_token_type_ids=True,
                )

        loss_img = 0.0
        gen_hidden_states = None
        if self.enable_future_image_loss and future_images is not None:
             loss_img, gen_hidden_states = self._compute_gen_loss_and_feats(future_images, hidden_states)

        connector_out, hidden_states, hidden_token_type_ids, actions, proprioception, history_actions, bridge_pixel_values, progress_labels, action_progress_labels, goal_connector_out, goal_hidden_states, goal_hidden_token_type_ids, goal_distance_labels, future_images = self._expand_for_bridge_sequence(
            connector_out,
            hidden_states,
            hidden_token_type_ids,
            actions,
            proprioception,
            history_actions,
            bridge_pixel_values,
            progress_labels,
            action_progress_labels,
            goal_connector_out,
            goal_hidden_states,
            goal_hidden_token_type_ids,
            goal_distance_labels,
            future_images,
        )

        if self.use_vita_latent_flow:
            return self._forward_vita_latent_flow(
                connector_out=connector_out,
                hidden_states=hidden_states,
                hidden_token_type_ids=hidden_token_type_ids,
                actions=actions,
                proprioception=proprioception,
                bridge_pixel_values=bridge_pixel_values,
                progress_labels=progress_labels,
                action_progress_labels=action_progress_labels,
                goal_connector_out=goal_connector_out,
                goal_hidden_states=goal_hidden_states,
                goal_hidden_token_type_ids=goal_hidden_token_type_ids,
                goal_distance_labels=goal_distance_labels,
                loss_img=loss_img,
                return_loss_dict=return_loss_dict,
                current_visual_tokens=current_visual_tokens,
                future_visual_tokens=future_visual_tokens,
                image_grid_thw=image_grid_thw,
                action_mask=action_mask,
            )
        if self.use_vlanext_latent_flow:
            return self._forward_vlanext_latent_flow(
                hidden_states=hidden_states,
                actions=actions,
                bridge_pixel_values=bridge_pixel_values,
                proprioception=proprioception,
                loss_img=loss_img,
                return_loss_dict=return_loss_dict,
                future_input_ids=future_input_ids,
                future_attention_mask=future_attention_mask,
                future_pixel_values=future_pixel_values,
                future_pixel_values_videos=future_pixel_values_videos,
                future_image_grid_thw=future_image_grid_thw,
                future_video_grid_thw=future_video_grid_thw,
                future_token_type_ids=future_token_type_ids,
                future_mm_token_type_ids=future_mm_token_type_ids,
            )
        
        if actions.ndim == 2: actions = actions.unsqueeze(1)
        noise = torch.randn_like(actions)
        B = actions.shape[0]
        
        if self.scheduler_type == "flow_match":
            sigmas = torch.rand((B,), device=actions.device)
            sigmas_expanded = sigmas.view(B, *([1] * (actions.ndim - 1)))
            noisy_actions = (1.0 - sigmas_expanded) * actions + sigmas_expanded * noise
            noisy_actions = noisy_actions.to(dtype=actions.dtype)
            timesteps = sigmas * self.noise_scheduler.config.num_train_timesteps
            target = noise - actions
        else:
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=actions.device).long()
            noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
            target = noise
            
        policy_history = history_actions if self.use_action_input_policy else None
        progress_loss = None

        if self.condition_type in ["tight", "soft"]:
            hidden_states = self._apply_latent_bridge(hidden_states, bridge_pixel_values, proprioception)
            policy_features, progress_loss = self._apply_progress(hidden_states, progress_labels)
            if self.enable_future_image_loss:
                pred = self.action_head(noisy_actions, timesteps, policy_features, history_actions=policy_history, gen_hidden_states=gen_hidden_states)
            else:
                pred = self.action_head(noisy_actions, timesteps, policy_features, history_actions=policy_history)
        elif self.condition_type == "loose":
            cond_input = connector_out.mean(dim=1)
            cond_input, progress_loss = self._apply_progress(cond_input, progress_labels)
            pred = self.action_head(noisy_actions, timesteps, cond_input, history_actions=policy_history)
        else:
             raise ValueError(f"Unknown condition type: {self.condition_type}")
        
        loss_action = F.mse_loss(pred, target)
        loss = loss_action
        loss_dct = None

        if self.dct_loss_weight > 0:
            pred_x_start = None
            if self.scheduler_type == "flow_match":
                 pred_x_start = noisy_actions - sigmas_expanded * pred
            elif self.scheduler_type == "ddim":
                 def view_right(t):
                    while t.ndim < pred.ndim:
                        t = t.unsqueeze(-1)
                    return t
                 alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=pred.device, dtype=pred.dtype)
                 alpha_prod_t = alphas_cumprod[timesteps]
                 pred_x_start = (noisy_actions - view_right((1 - alpha_prod_t).sqrt()) * pred) / view_right(alpha_prod_t.sqrt())

            if pred_x_start is not None:
                 loss_dct = self._compute_dct_loss(pred_x_start.float(), actions.float())
                 loss = loss + self.dct_loss_weight * loss_dct
            
        if self.future_image_loss_weight > 0:
            loss = loss + self.future_image_loss_weight * loss_img
        loss = self._add_progress_loss(loss, progress_loss)
        if return_loss_dict:
            loss_dict = {
                "loss": loss,
                "loss_action": loss_action.detach(),
            }
            if loss_dct is not None:
                loss_dict["loss_dct"] = loss_dct.detach()
                loss_dict["loss_dct_weighted"] = (self.dct_loss_weight * loss_dct).detach()
            if progress_loss is not None:
                loss_dict["loss_progress"] = progress_loss.detach()
                loss_dict["loss_progress_weighted"] = (self.progress_loss_weight * progress_loss).detach()
            if self.future_image_loss_weight > 0:
                if not torch.is_tensor(loss_img):
                    loss_img = torch.tensor(loss_img, device=loss.device, dtype=loss.dtype)
                loss_dict["loss_future_image"] = loss_img.detach()
                loss_dict["loss_future_image_weighted"] = (self.future_image_loss_weight * loss_img).detach()
            return loss_dict
        return loss

    def _forward_vlanext_latent_flow(
        self,
        hidden_states,
        actions,
        bridge_pixel_values,
        proprioception,
        loss_img,
        return_loss_dict,
        future_input_ids=None,
        future_attention_mask=None,
        future_pixel_values=None,
        future_pixel_values_videos=None,
        future_image_grid_thw=None,
        future_video_grid_thw=None,
        future_token_type_ids=None,
        future_mm_token_type_ids=None,
    ):
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        if self.condition_type not in ["tight", "soft"]:
            raise ValueError("vlanext_latent_flow currently requires tight/soft hidden states.")

        hidden_states = self._apply_latent_bridge(
            hidden_states,
            bridge_pixel_values,
            proprioception,
        )
        z_obs = self.vlanext_obs_projector(hidden_states[-1])
        z_act = self.vlanext_action_encoder(actions).unsqueeze(1)

        batch_size = actions.shape[0]
        tau = torch.rand((batch_size,), device=actions.device)
        tau_view = tau.view(batch_size, *([1] * (z_obs.ndim - 1))).to(dtype=z_obs.dtype)
        z_tau = (1.0 - tau_view) * z_obs + tau_view * z_act
        target_velocity = z_act - z_obs
        timesteps = tau.float() * float(self.num_train_timesteps)

        loss_visual_flow = None
        if self.use_vlanext_dual_flow:
            if self.vlanext_visual_flow_loss_weight > 0 and (
                future_input_ids is None or future_attention_mask is None
            ):
                raise ValueError("Dual-flow visual loss requires future VLM inputs from the data collator.")
            if self.vlanext_visual_flow_loss_weight > 0:
                with torch.no_grad():
                    _, future_hidden_states = self.get_vlm_condition(
                        future_input_ids,
                        future_attention_mask,
                        proprioception=None,
                        proprio_attention_mask=None,
                        pixel_values=future_pixel_values,
                        pixel_values_videos=future_pixel_values_videos,
                        image_grid_thw=future_image_grid_thw,
                        video_grid_thw=future_video_grid_thw,
                        token_type_ids=future_token_type_ids,
                        mm_token_type_ids=future_mm_token_type_ids,
                    )
                z_vis_target = self.vlanext_obs_projector(future_hidden_states[-1]).detach()
            else:
                z_vis_target = z_obs.detach()
            z_vis_tau = (1.0 - tau_view) * z_obs + tau_view * z_vis_target
            target_visual_velocity = z_vis_target - z_obs
            pred_velocity, pred_visual_velocity = self.action_head(
                z_tau,
                z_vis_tau,
                timesteps,
                hidden_states,
            )
            if self.vlanext_visual_flow_loss_weight > 0:
                loss_visual_flow = F.mse_loss(
                    pred_visual_velocity.float(),
                    target_visual_velocity.float(),
                )
        else:
            pred_velocity = self.action_head(z_tau, timesteps, hidden_states)
        loss_flow = F.mse_loss(pred_velocity.float(), target_velocity.float())

        pred_action_latent = z_tau + (1.0 - tau_view) * pred_velocity
        pred_actions = self.vlanext_action_decoder(pred_action_latent.squeeze(1))
        loss_recon = self._vita_reconstruction_loss(pred_actions, actions)

        loss = loss_flow + self.vita_enc_recon_weight * loss_recon
        if loss_visual_flow is not None:
            loss = loss + self.vlanext_visual_flow_loss_weight * loss_visual_flow
        loss_dct = None
        if self.dct_loss_weight > 0:
            loss_dct = self._compute_dct_loss(pred_actions.float(), actions.float())
            loss = loss + self.dct_loss_weight * loss_dct

        if self.future_image_loss_weight > 0:
            loss = loss + self.future_image_loss_weight * loss_img

        if not return_loss_dict:
            return loss

        loss_dict = {
            "loss": loss,
            "loss_latent_flow": loss_flow.detach(),
            "loss_action_recon": loss_recon.detach(),
            "loss_action_recon_weighted": (
                self.vita_enc_recon_weight * loss_recon
            ).detach(),
        }
        if loss_dct is not None:
            loss_dict["loss_dct"] = loss_dct.detach()
            loss_dict["loss_dct_weighted"] = (
                self.dct_loss_weight * loss_dct
            ).detach()
        if loss_visual_flow is not None:
            loss_dict["loss_visual_flow"] = loss_visual_flow.detach()
            loss_dict["loss_visual_flow_weighted"] = (
                self.vlanext_visual_flow_loss_weight * loss_visual_flow
            ).detach()
        if self.future_image_loss_weight > 0:
            if not torch.is_tensor(loss_img):
                loss_img = torch.tensor(loss_img, device=loss.device, dtype=loss.dtype)
            loss_dict["loss_future_image"] = loss_img.detach()
            loss_dict["loss_future_image_weighted"] = (
                self.future_image_loss_weight * loss_img
            ).detach()
        return loss_dict

    def _forward_vita_latent_flow(
        self,
        connector_out,
        hidden_states,
        hidden_token_type_ids,
        actions,
        proprioception,
        bridge_pixel_values,
        progress_labels,
        action_progress_labels,
        goal_connector_out,
        goal_hidden_states,
        goal_hidden_token_type_ids,
        goal_distance_labels,
        loss_img,
        return_loss_dict,
        current_visual_tokens=None,
        future_visual_tokens=None,
        image_grid_thw=None,
        action_mask=None,
    ):
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)

        progress_loss = None
        if self.condition_type in ["tight", "soft"]:
            hidden_states = self._apply_latent_bridge(
                hidden_states,
                bridge_pixel_values,
                proprioception,
            )
            hidden_states, progress_loss = self._apply_progress(
                hidden_states,
                progress_labels,
            )
            observation_latent, condition_tokens = self.vita_action_generator.encode_observation(
                hidden_states=hidden_states,
                hidden_token_type_ids=hidden_token_type_ids,
                proprioception=proprioception if self.vita_use_proprio else None,
                return_condition_tokens=True,
                image_grid_thw=image_grid_thw,
                spatial_merge_size=int(self.lmm.model.visual.spatial_merge_size),
            )
        elif self.condition_type == "loose":
            connector_out, progress_loss = self._apply_progress(
                connector_out,
                progress_labels,
            )
            observation_latent, condition_tokens = self.vita_action_generator.encode_observation(
                connector_out=connector_out,
                proprioception=proprioception if self.vita_use_proprio else None,
                return_condition_tokens=True,
            )
        else:
            raise ValueError(f"Unknown condition type: {self.condition_type}")

        mq_overlap_loss = getattr(
            self.vita_action_generator.observation_encoder,
            "last_mq_overlap_loss",
            None,
        )
        mq_overlap_losses = dict(
            getattr(
                self.vita_action_generator.observation_encoder,
                "last_mq_overlap_losses",
                {},
            )
        )
        mq_balance_loss = getattr(
            self.vita_action_generator.observation_encoder,
            "last_mq_balance_loss",
            None,
        )
        mq_balance_losses = dict(
            getattr(
                self.vita_action_generator.observation_encoder,
                "last_mq_balance_losses",
                {},
            )
        )
        observation_latent = self._apply_vita_zobs_mode(observation_latent)

        current_visual_resampled = None
        future_visual_resampled = None
        visual_delta_target = None
        multi_horizon_visual_target = None
        if self.vita_action_effect_enabled:
            if current_visual_tokens is None:
                raise ValueError("Action representation training requires current visual tokens.")
            current_visual_resampled = self.vita_action_generator.resample_visual_tokens(
                current_visual_tokens
            )
            if self.vita_action_representation_type == "single_view_edar_lite":
                future_visual_resampled = None
            else:
                if future_visual_tokens is None:
                    raise ValueError(
                        "Legacy action-effect training requires future visual tokens."
                    )
                future_visual_resampled = self.vita_action_generator.resample_visual_tokens(
                    future_visual_tokens
                )
            if (
                self.vita_action_representation_type != "single_view_edar_lite"
                and self.vita_action_effect_horizons
            ):
                if future_visual_resampled.ndim != 4:
                    raise ValueError(
                        "Multi-horizon visual targets must have shape [B,H,N,D]."
                    )
                if future_visual_resampled.shape[1] != len(
                    self.vita_action_effect_horizons
                ):
                    raise ValueError(
                        "Future visual target count does not match configured horizons."
                    )
                if self.vita_action_effect_detach_visual_teacher:
                    current_visual_resampled = current_visual_resampled.detach()
                multi_horizon_visual_target = future_visual_resampled.detach()
            elif self.vita_action_representation_type != "single_view_edar_lite":
                visual_delta_target = (
                    future_visual_resampled - current_visual_resampled
                ).detach()

        action_latent = self.vita_action_generator.encode_action(
            actions,
            current_visual_tokens=current_visual_resampled,
        )
        loss_action_progress = None
        if self.vita_action_progress_head is not None and action_progress_labels is not None:
            progress_latent = self.vita_action_generator.flatten_action_latent(action_latent)
            progress_pred = self.vita_action_progress_head(progress_latent).squeeze(-1).float()
            progress_target = action_progress_labels.float().detach()
            loss_action_progress = F.smooth_l1_loss(progress_pred, progress_target)

        flow_condition_cache = None
        if self.vita_action_generator.flow.can_cache_condition(
            condition_tokens
        ):
            flow_condition_cache = (
                self.vita_action_generator.prepare_flow_condition_cache(
                    condition_tokens
                )
            )
        flow_step = None
        if self.vita_enable_one_step_action_loss:
            flow_step = self.vita_action_generator.flow_matching_step(
                observation_latent,
                action_latent,
                condition_tokens=condition_tokens,
                condition_cache=flow_condition_cache,
            )
            loss_flow = flow_step["loss"]
        else:
            loss_flow = self.vita_action_generator.flow_matching_loss(
                observation_latent,
                action_latent,
                condition_tokens=condition_tokens,
                condition_cache=flow_condition_cache,
            )

        if self.vita_enable_one_step_action_loss:
            predicted_action_latent = recover_one_step_action_latent(
                flow_step["interpolated"],
                flow_step["predicted_velocity"],
                flow_step["timestep"],
            )
            if predicted_action_latent.shape[1:] != (4, 256):
                raise ValueError("One-step EDAR latent must have shape [B,4,256].")
            predicted_actions = self.vita_action_generator.decode(
                predicted_action_latent
            )
            loss_action_one_step = self._vita_reconstruction_loss(
                predicted_actions,
                actions,
                action_mask=action_mask,
            )
            progress = (
                self.vita_stage_b_global_step
                / float(self.vita_stage_b_max_train_steps)
            )
            flow_weight, action_weight = stage_b_one_step_loss_weights(
                progress,
                flow_final=self.vita_flow_loss_weight_final,
                action_final=self.vita_one_step_action_loss_weight_final,
            )
            loss = flow_weight * loss_flow + action_weight * loss_action_one_step
            if not return_loss_dict:
                return loss
            loss_dict = {
                "loss": loss,
                "flow_loss_raw": loss_flow.detach(),
                "action_1step_loss_raw": loss_action_one_step.detach(),
                "flow_loss_weighted": (flow_weight * loss_flow).detach(),
                "action_1step_loss_weighted": (
                    action_weight * loss_action_one_step
                ).detach(),
                "flow_loss_weight": loss.new_tensor(flow_weight),
                "action_1step_loss_weight": loss.new_tensor(action_weight),
                "pred_z_act_norm": predicted_action_latent.detach().float().norm(dim=-1).mean(),
                "target_z_act_norm": action_latent.detach().float().norm(dim=-1).mean(),
            }
            if actions.shape[-1] >= 7:
                loss_dict["action_position_loss"] = self._vita_reconstruction_loss(
                    predicted_actions[..., :3], actions[..., :3], action_mask
                ).detach()
                loss_dict["action_rotation_loss"] = self._vita_reconstruction_loss(
                    predicted_actions[..., 3:6], actions[..., 3:6], action_mask
                ).detach()
                loss_dict["action_gripper_loss"] = self._vita_reconstruction_loss(
                    predicted_actions[..., 6:], actions[..., 6:], action_mask
                ).detach()
            return loss_dict

        if self.dynamic_flow_latent_reconstruction_weight:
            (
                flow_loss_weight,
                latent_consistency_weight,
                action_reconstruction_weight,
            ) = get_flow_latent_reconstruction_weights(
                self.vita_stage_b_global_step,
                self.vita_stage_b_max_train_steps,
            )
        else:
            flow_loss_weight, action_reconstruction_weight = (
                resolve_flow_reconstruction_weights(
                    enabled=self.dynamic_flow_reconstruction_weight,
                    global_step=self.vita_stage_b_global_step,
                    max_train_steps=self.vita_stage_b_max_train_steps,
                    fixed_flow_weight=1.0,
                    fixed_reconstruction_weight=self.vita_flow_recon_weight,
                )
            )
            latent_consistency_weight = self.vita_consistency_weight
        loss = flow_loss_weight * loss_flow
        if mq_overlap_loss is not None:
            loss = loss + mq_overlap_loss.to(dtype=loss.dtype)
        if mq_balance_loss is not None:
            loss = loss + mq_balance_loss.to(dtype=loss.dtype)
        if loss_action_progress is not None:
            loss = loss + self.vita_action_progress_loss_weight * loss_action_progress

        loss_goal_distance = None
        goal_distance_pred = None
        if self.vita_goal_metric_head is not None and goal_distance_labels is not None:
            loss_goal_distance, goal_distance_pred = self._vita_goal_distance_loss(
                observation_latent,
                goal_connector_out=goal_connector_out,
                goal_hidden_states=goal_hidden_states,
                goal_hidden_token_type_ids=goal_hidden_token_type_ids,
                goal_distance_labels=goal_distance_labels,
            )
            loss = loss + self.vita_goal_distance_loss_weight * loss_goal_distance

        loss_diversity = None
        if self.vita_diversity_loss_weight > 0 or self.vita_log_diversity_loss:
            diversity_tokens = condition_tokens
            if diversity_tokens.ndim == 4:
                # Blockwise conditioning returns [B, num_layers, num_queries, D].
                # Keep the regularizer within each layer instead of forcing
                # different VLM layers to be mutually orthogonal.
                num_queries = diversity_tokens.shape[-2]
                if num_queries > 1:
                    normalized_tokens = F.normalize(diversity_tokens.float(), dim=-1)
                    similarity = normalized_tokens @ normalized_tokens.transpose(-1, -2)
                    identity = torch.eye(
                        num_queries,
                        device=similarity.device,
                        dtype=similarity.dtype,
                    ).view(1, 1, num_queries, num_queries)
                    loss_diversity = F.mse_loss(similarity, identity.expand_as(similarity))
            elif diversity_tokens.shape[1] > 1:
                normalized_tokens = F.normalize(diversity_tokens.float(), dim=-1)
                similarity = normalized_tokens @ normalized_tokens.transpose(1, 2)
                identity = torch.eye(
                    similarity.shape[-1],
                    device=similarity.device,
                    dtype=similarity.dtype,
                ).unsqueeze(0)
                loss_diversity = F.mse_loss(similarity, identity.expand_as(similarity))
            if loss_diversity is not None:
                loss = loss + self.vita_diversity_loss_weight * loss_diversity

        encoded_actions = self.vita_action_generator.decode(action_latent)
        loss_enc_recon = self._vita_reconstruction_loss(encoded_actions, actions)
        if self.vita_action_representation_type != "single_view_edar_lite":
            loss = loss + self.vita_enc_recon_weight * loss_enc_recon
        loss_visual_ae = None
        loss_action_effect_rank = None
        action_effect_rank_metrics = {}
        action_effect_horizon_metrics = {}
        action_effect_state = None
        if (
            self.vita_action_effect_enabled
            and self.vita_action_representation_type != "single_view_edar_lite"
            and self.vita_action_effect_horizons
        ):
            predicted_visual_future = (
                self.vita_action_generator.predict_visual_horizons(
                    actions=actions,
                    action_latent=action_latent,
                    current_visual_tokens=current_visual_resampled,
                    proprioception=proprioception,
                )
            )
            (
                loss_visual_ae,
                action_effect_horizon_metrics,
            ) = motion_weighted_dense_huber_loss(
                prediction=predicted_visual_future,
                target=multi_horizon_visual_target,
                current=current_visual_resampled,
                horizon_weights=self.vita_action_effect_horizon_loss_weights,
                motion_weight_floor=(
                    self.vita_action_effect_motion_weight_floor
                ),
                eps=self.vita_action_effect_motion_weight_eps,
            )
            loss = loss + self.vita_action_effect_visual_ae_weight * loss_visual_ae
        elif (
            self.vita_action_effect_enabled
            and self.vita_action_representation_type != "single_view_edar_lite"
        ):
            action_effect_state = self.vita_action_generator.encode_action_effect_state(
                proprioception
            )
            effect_latent = self.vita_action_generator.project_action_effect(
                action_latent,
                action_effect_state,
            )
            predicted_visual_delta = self.vita_action_generator.decode_projected_visual_delta(
                effect_latent,
                current_visual_resampled,
                action_effect_state,
            )
            loss_visual_ae = F.smooth_l1_loss(
                predicted_visual_delta.float(),
                visual_delta_target.float(),
            )
            loss = loss + self.vita_action_effect_visual_ae_weight * loss_visual_ae
            if self.vita_action_effect_rank_weight > 0:
                if self.vita_action_effect_negative_queue_size > 0:
                    (
                        negative_actions,
                        negative_mask,
                        negative_action_distance,
                        queue_metrics,
                    ) = self._sample_action_effect_negative_actions(actions)
                    batch_size, negative_count = negative_actions.shape[:2]
                    flat_negative_actions = negative_actions.flatten(0, 1)
                    negative_current_visual = current_visual_resampled.unsqueeze(1).expand(
                        batch_size,
                        negative_count,
                        *current_visual_resampled.shape[1:],
                    ).flatten(0, 1)
                    with torch.no_grad():
                        negative_action_latent = self.vita_action_generator.encode_action(
                            flat_negative_actions,
                            current_visual_tokens=negative_current_visual,
                        )
                    negative_state = (
                        action_effect_state.unsqueeze(1).expand(
                            batch_size,
                            negative_count,
                            action_effect_state.shape[-1],
                        ).flatten(0, 1)
                        if action_effect_state is not None
                        else None
                    )
                    negative_effect_latent = self.vita_action_generator.project_action_effect(
                        negative_action_latent,
                        negative_state,
                    )
                    negative_visual_delta = (
                        self.vita_action_generator.decode_projected_visual_delta(
                            negative_effect_latent,
                            negative_current_visual,
                            negative_state,
                        ).unflatten(0, (batch_size, negative_count))
                    )
                    loss_action_effect_rank, action_effect_rank_metrics = (
                        action_effect_ranking_loss(
                            predicted_visual_delta,
                            negative_visual_delta,
                            visual_delta_target,
                            margin=self.vita_action_effect_rank_margin,
                            min_target_norm=self.vita_action_effect_rank_min_delta_norm,
                            distance_mode=self.vita_action_effect_rank_distance_mode,
                            huber_weight=self.vita_action_effect_rank_huber_weight,
                            scale_floor=self.vita_action_effect_rank_scale_floor,
                            negative_mask=negative_mask,
                            hard_negative_count=self.vita_action_effect_hard_negatives,
                            negative_action_distance=negative_action_distance,
                        )
                    )
                    action_effect_rank_metrics.update(queue_metrics)
                    if self.training:
                        self._enqueue_action_effect_actions(actions)
                elif effect_latent.shape[0] > 1:
                    shift = int(torch.randint(1, effect_latent.shape[0], ()).item())
                    shuffled_action_latent = torch.roll(action_latent, shifts=shift, dims=0)
                    shuffled_effect_latent = self.vita_action_generator.project_action_effect(
                        shuffled_action_latent,
                        action_effect_state,
                    )
                    shuffled_visual_delta = (
                        self.vita_action_generator.decode_projected_visual_delta(
                            shuffled_effect_latent,
                            current_visual_resampled,
                            action_effect_state,
                        )
                    )
                    loss_action_effect_rank, action_effect_rank_metrics = (
                        action_effect_ranking_loss(
                            predicted_visual_delta,
                            shuffled_visual_delta,
                            visual_delta_target,
                            margin=self.vita_action_effect_rank_margin,
                            min_target_norm=self.vita_action_effect_rank_min_delta_norm,
                            distance_mode=self.vita_action_effect_rank_distance_mode,
                            huber_weight=self.vita_action_effect_rank_huber_weight,
                            scale_floor=self.vita_action_effect_rank_scale_floor,
                        )
                    )
                if loss_action_effect_rank is not None:
                    loss = (
                        loss
                        + self.vita_action_effect_rank_weight * loss_action_effect_rank
                    )

        loss_consistency = None
        loss_flow_recon = None
        loss_dct = None
        loss_visual_flow = None
        need_flow_decode = (
            self.dynamic_flow_reconstruction_weight
            or self.dynamic_flow_latent_reconstruction_weight
            or self.vita_consistency_weight > 0
            or self.vita_flow_recon_weight > 0
            or self.dct_loss_weight > 0
            or (
                self.vita_action_effect_enabled
                and self.vita_action_representation_type != "single_view_edar_lite"
                and self.vita_action_effect_visual_flow_weight > 0
            )
        )
        if need_flow_decode:
            predicted_action_latent = self.vita_action_generator.sample_latent(
                observation_latent,
                condition_tokens=condition_tokens,
                condition_cache=flow_condition_cache,
            )
            loss_consistency = F.mse_loss(
                predicted_action_latent.float(),
                action_latent.float(),
            )
            loss = loss + latent_consistency_weight * loss_consistency

            predicted_actions = self.vita_action_generator.decode(
                predicted_action_latent
            )
            loss_flow_recon = self._vita_reconstruction_loss(
                predicted_actions,
                actions,
            )
            loss = loss + action_reconstruction_weight * loss_flow_recon

            if self.dct_loss_weight > 0:
                loss_dct = self._compute_dct_loss(
                    predicted_actions.float(),
                    actions.float(),
                )
                loss = loss + self.dct_loss_weight * loss_dct

            if (
                self.vita_action_effect_enabled
                and self.vita_action_representation_type != "single_view_edar_lite"
                and self.vita_action_effect_visual_flow_weight > 0
            ):
                predicted_flow_visual_delta = self.vita_action_generator.decode_visual_delta(
                    predicted_action_latent,
                    current_visual_resampled,
                    action_effect_state,
                )
                loss_visual_flow = F.smooth_l1_loss(
                    predicted_flow_visual_delta.float(),
                    visual_delta_target.float(),
                )
                loss = loss + self.vita_action_effect_visual_flow_weight * loss_visual_flow

        if self.future_image_loss_weight > 0:
            loss = loss + self.future_image_loss_weight * loss_img
        loss = self._add_progress_loss(loss, progress_loss)

        if not return_loss_dict:
            return loss

        loss_dict = {
            "loss": loss,
            "loss_flow": loss_flow.detach(),
            "flow_loss_raw": loss_flow.detach(),
            "flow_loss_weight": loss.new_tensor(flow_loss_weight),
            "flow_loss_weighted": (flow_loss_weight * loss_flow).detach(),
            "total_loss": loss.detach(),
            "loss_enc_action_recon": loss_enc_recon.detach(),
            "loss_enc_action_recon_weighted": (
                (
                    0.0
                    if self.vita_action_representation_type == "single_view_edar_lite"
                    else self.vita_enc_recon_weight
                )
                * loss_enc_recon
            ).detach(),
        }
        adaptive_mq_router_metrics = getattr(
            self.vita_action_generator.observation_encoder,
            "last_adaptive_mq_router_metrics",
            {},
        )
        for name, value in adaptive_mq_router_metrics.items():
            loss_dict[name] = value.detach()
        dynamic_topk_metrics = getattr(
            self.vita_action_generator.observation_encoder,
            "last_dynamic_topk_metrics",
            {},
        )
        for name, value in dynamic_topk_metrics.items():
            loss_dict[name] = value.detach()
        token_gate_metrics = getattr(
            self.vita_action_generator.flow,
            "last_token_gate_metrics",
            {},
        )
        for name, value in token_gate_metrics.items():
            loss_dict[name] = value.detach()
        global_token_metrics = getattr(
            self.vita_action_generator.observation_encoder,
            "last_global_token_metrics",
            {},
        )
        for name, value in global_token_metrics.items():
            loss_dict[name] = value.detach()
        if mq_overlap_losses:
            for name, value in mq_overlap_losses.items():
                loss_dict[f"loss_mq_attention_overlap_{name}"] = value.detach()
            if mq_overlap_loss is not None:
                loss_dict["loss_mq_attention_overlap_weighted"] = mq_overlap_loss.detach()
            else:
                loss_dict["loss_mq_attention_overlap_weighted"] = torch.zeros(
                    (), device=loss.device, dtype=loss.dtype
                )
        if mq_balance_losses:
            for name, value in mq_balance_losses.items():
                loss_dict[f"loss_mq_competitive_balance_{name}"] = value.detach()
            if mq_balance_loss is not None:
                loss_dict["loss_mq_competitive_balance_weighted"] = mq_balance_loss.detach()
            else:
                loss_dict["loss_mq_competitive_balance_weighted"] = torch.zeros(
                    (), device=loss.device, dtype=loss.dtype
                )
        if loss_diversity is not None:
            loss_dict["loss_mq_diversity"] = loss_diversity.detach()
            loss_dict["loss_mq_diversity_weighted"] = (
                self.vita_diversity_loss_weight * loss_diversity
            ).detach()
        if loss_consistency is not None:
            loss_dict["loss_latent_consistency"] = loss_consistency.detach()
            loss_dict["loss_latent_consistency_weighted"] = (
                latent_consistency_weight * loss_consistency
            ).detach()
            loss_dict["latent_consistency_weight"] = loss.new_tensor(
                latent_consistency_weight
            )
        if loss_flow_recon is not None:
            loss_dict["loss_flow_action_recon"] = loss_flow_recon.detach()
            loss_dict["loss_flow_action_recon_weighted"] = (
                action_reconstruction_weight * loss_flow_recon
            ).detach()
            loss_dict["action_reconstruction_loss_raw"] = (
                loss_flow_recon.detach()
            )
            loss_dict["action_reconstruction_weight"] = loss.new_tensor(
                action_reconstruction_weight
            )
            loss_dict["action_reconstruction_loss_weighted"] = (
                action_reconstruction_weight * loss_flow_recon
            ).detach()
        if loss_dct is not None:
            loss_dict["loss_dct"] = loss_dct.detach()
            loss_dict["loss_dct_weighted"] = (
                self.dct_loss_weight * loss_dct
            ).detach()
        if loss_visual_ae is not None:
            loss_dict["loss_action_effect_visual_ae"] = loss_visual_ae.detach()
            loss_dict["loss_action_effect_visual_ae_weighted"] = (
                self.vita_action_effect_visual_ae_weight * loss_visual_ae
            ).detach()
            if action_effect_horizon_metrics:
                for index, horizon in enumerate(
                    self.vita_action_effect_horizons
                ):
                    loss_dict[
                        f"loss_action_effect_visual_h{horizon}"
                    ] = action_effect_horizon_metrics["per_horizon"][index]
                    loss_dict[
                        f"action_effect_motion_h{horizon}"
                    ] = action_effect_horizon_metrics["mean_motion"][index]
                    loss_dict[
                        f"action_effect_max_patch_weight_h{horizon}"
                    ] = action_effect_horizon_metrics["max_patch_weight"][index]
        if loss_action_effect_rank is not None:
            loss_dict["loss_action_effect_rank"] = loss_action_effect_rank.detach()
            loss_dict["loss_action_effect_rank_weighted"] = (
                self.vita_action_effect_rank_weight * loss_action_effect_rank
            ).detach()
            for name, value in action_effect_rank_metrics.items():
                loss_dict[f"action_effect_{name}"] = value.detach()
        if loss_visual_flow is not None:
            loss_dict["loss_action_effect_visual_flow"] = loss_visual_flow.detach()
            loss_dict["loss_action_effect_visual_flow_weighted"] = (
                self.vita_action_effect_visual_flow_weight * loss_visual_flow
            ).detach()
        if progress_loss is not None:
            loss_dict["loss_progress"] = progress_loss.detach()
            loss_dict["loss_progress_weighted"] = (
                self.progress_loss_weight * progress_loss
            ).detach()
        if loss_action_progress is not None:
            loss_dict["loss_action_progress"] = loss_action_progress.detach()
            loss_dict["loss_action_progress_weighted"] = (
                self.vita_action_progress_loss_weight * loss_action_progress
            ).detach()
        if loss_goal_distance is not None:
            loss_dict["loss_goal_distance"] = loss_goal_distance.detach()
            loss_dict["loss_goal_distance_weighted"] = (
                self.vita_goal_distance_loss_weight * loss_goal_distance
            ).detach()
            loss_dict["goal_distance_pred_mean"] = goal_distance_pred.detach().mean()
            loss_dict["goal_distance_target_mean"] = goal_distance_labels.detach().float().mean()
        return loss_dict

    @torch.no_grad()
    def predict_action(self, input_ids, attention_mask, proprioception=None, history_actions=None, proprio_attention_mask=None, pixel_values=None, pixel_values_videos=None, image_grid_thw=None, video_grid_thw=None, token_type_ids=None, mm_token_type_ids=None, bridge_pixel_values=None, cached_vlm_condition=None):
        if cached_vlm_condition is None:
            if self.use_vita_latent_flow:
                connector_out, hidden_states, hidden_token_type_ids = self.get_vlm_condition(
                    input_ids, attention_mask,
                    proprioception=self._vlm_proprio(proprioception),
                    proprio_attention_mask=proprio_attention_mask,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    token_type_ids=token_type_ids,
                    mm_token_type_ids=mm_token_type_ids,
                    return_token_type_ids=True,
                )
            else:
                connector_out, hidden_states = self.get_vlm_condition(
                    input_ids, attention_mask,
                    proprioception=self._vlm_proprio(proprioception),
                    proprio_attention_mask=proprio_attention_mask,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    token_type_ids=token_type_ids,
                    mm_token_type_ids=mm_token_type_ids
                )
                hidden_token_type_ids = None
        else:
            if len(cached_vlm_condition) == 3:
                connector_out, hidden_states, hidden_token_type_ids = cached_vlm_condition
            else:
                connector_out, hidden_states = cached_vlm_condition
                hidden_token_type_ids = None
        if hidden_states is not None:
            B = hidden_states[-1].shape[0] if isinstance(hidden_states, (tuple, list)) else hidden_states.shape[0]
        elif connector_out is not None:
            B = connector_out.shape[0]
        else:
            B = input_ids.shape[0]
        
        policy_history = history_actions if self.use_action_input_policy else None
        gen_hidden_states = None
        if self.enable_future_image_loss and self.condition_type in ["tight", "soft"]:
             num_img_tokens = 256 
             curr_ids = torch.zeros((B, 1), dtype=torch.long, device=input_ids.device)
             gen_context = hidden_states
             for _ in range(num_img_tokens):
                 logits, _ = self.generator(curr_ids, gen_context)
                 next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                 curr_ids = torch.cat([curr_ids, next_token], dim=1)
             gen_input = curr_ids[:, :-1]
             _, gen_hidden_states = self.generator(gen_input, gen_context)

        if self.loss_type == "regression":
            if self.condition_type in ["tight", "soft"]:
                hidden_states = self._apply_latent_bridge(hidden_states, bridge_pixel_values, proprioception)
                policy_features, _ = self._apply_progress(hidden_states)
                if self.enable_future_image_loss:
                    action = self.action_head(policy_features, history_actions=policy_history, gen_hidden_states=gen_hidden_states)
                else:
                    action = self.action_head(policy_features, history_actions=policy_history)
            elif self.condition_type == "loose":
                cond_input = connector_out.mean(dim=1)
                cond_input, _ = self._apply_progress(cond_input)
                action = self.action_head(cond_input, history_actions=policy_history)
            if action.ndim == 2 and self.num_actions > 1:
                action = action.view(action.shape[0], self.num_actions, self.action_dim)
            
            return action.to(dtype=self.lmm.dtype)

        elif self.loss_type == "classification":
            if self.condition_type in ["tight", "soft"]:
                hidden_states = self._apply_latent_bridge(hidden_states, bridge_pixel_values, proprioception)
                policy_features, _ = self._apply_progress(hidden_states)
                if self.enable_future_image_loss:
                    logits = self.action_head(policy_features, history_actions=policy_history, gen_hidden_states=gen_hidden_states)
                else:
                    logits = self.action_head(policy_features, history_actions=policy_history)
            else:
                cond_input = connector_out.mean(dim=1)
                cond_input, _ = self._apply_progress(cond_input)
                logits = self.action_head(cond_input, history_actions=policy_history)

            if self.action_vqvae is not None:
                indices = torch.argmax(logits, dim=-1)  # (B, T, Latent_Codes)
                action = self.action_vqvae.decode_indices(indices)
                return action.to(dtype=self.lmm.dtype)

            else:
                pose_logits = logits[:, :, :self.action_dim - 1, :]
                gripper_logits = logits[:, :, -1:, :2]
                pose_idx = torch.argmax(pose_logits, dim=-1)
                gripper_idx = torch.argmax(gripper_logits, dim=-1)
                pose_pred = (pose_idx.float() / (self.num_bins - 1)) * 2 - 1
                gripper_pred = gripper_idx.float() * 2 - 1
                action = torch.cat([pose_pred, gripper_pred], dim=-1).to(dtype=self.lmm.dtype)
                return action

        elif self.loss_type == "diffusion":
            if self.use_vlanext_latent_flow:
                if self.condition_type not in ["tight", "soft"]:
                    raise ValueError("vlanext_latent_flow currently requires tight/soft hidden states.")
                hidden_states = self._apply_latent_bridge(
                    hidden_states,
                    bridge_pixel_values,
                    proprioception,
                )
                action_latent = self.vlanext_obs_projector(hidden_states[-1])
                visual_latent = action_latent.clone()
                num_steps = max(1, int(self.num_inference_timesteps))
                step_size = 1.0 / float(num_steps)
                for step_idx in range(num_steps):
                    tau = (float(step_idx) + 0.5) / float(num_steps)
                    timesteps = torch.full(
                        (B,),
                        tau * float(self.num_train_timesteps),
                        device=input_ids.device,
                    )
                    if self.use_vlanext_dual_flow:
                        velocity, visual_velocity = self.action_head(
                            action_latent,
                            visual_latent,
                            timesteps,
                            hidden_states,
                        )
                        visual_latent = visual_latent + step_size * visual_velocity
                    else:
                        velocity = self.action_head(action_latent, timesteps, hidden_states)
                    action_latent = action_latent + step_size * velocity
                action = self.vlanext_action_decoder(action_latent.squeeze(1))
                return action.to(dtype=self.lmm.dtype)

            if self.use_vita_latent_flow:
                if self.condition_type in ["tight", "soft"]:
                    hidden_states = self._apply_latent_bridge(
                        hidden_states,
                        bridge_pixel_values,
                        proprioception,
                    )
                    hidden_states, _ = self._apply_progress(hidden_states)
                    observation_latent, condition_tokens = self.vita_action_generator.encode_observation(
                        hidden_states=hidden_states,
                        hidden_token_type_ids=hidden_token_type_ids,
                        proprioception=proprioception if self.vita_use_proprio else None,
                        return_condition_tokens=True,
                        image_grid_thw=image_grid_thw,
                        spatial_merge_size=int(self.lmm.model.visual.spatial_merge_size),
                    )
                elif self.condition_type == "loose":
                    connector_out, _ = self._apply_progress(connector_out)
                    observation_latent, condition_tokens = self.vita_action_generator.encode_observation(
                        connector_out=connector_out,
                        proprioception=proprioception if self.vita_use_proprio else None,
                        return_condition_tokens=True,
                    )
                else:
                    raise ValueError(f"Unknown condition type: {self.condition_type}")
                observation_latent = self._apply_vita_zobs_mode(observation_latent)

                action_latent = self.vita_action_generator.sample_latent(
                    observation_latent,
                    num_steps=self.num_inference_timesteps,
                    condition_tokens=condition_tokens,
                )
                action = self.vita_action_generator.decode(action_latent)
                return action.to(dtype=self.lmm.dtype)

            action = torch.randn(B, self.num_actions, self.action_dim, device=input_ids.device).to(self.lmm.dtype)
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps)
            if self.condition_type in ["tight", "soft"]:
                hidden_states = self._apply_latent_bridge(hidden_states, bridge_pixel_values, proprioception)
            
            for t in self.noise_scheduler.timesteps:
                timesteps = torch.full((B,), t, device=input_ids.device)
                if self.scheduler_type != "flow_match": timesteps = timesteps.long()
                if self.condition_type in ["tight", "soft"]:
                    policy_features, _ = self._apply_progress(hidden_states)
                    if self.enable_future_image_loss:
                        output = self.action_head(action, timesteps, policy_features, history_actions=policy_history, gen_hidden_states=gen_hidden_states)
                    else:
                        output = self.action_head(action, timesteps, policy_features, history_actions=policy_history)
                else:
                    cond_input = connector_out.mean(dim=1)
                    cond_input, _ = self._apply_progress(cond_input)
                    output = self.action_head(action, timesteps, cond_input, history_actions=policy_history)
                
                action = self.noise_scheduler.step(output, t, action).prev_sample
                action = action.to(dtype=self.lmm.dtype)
            
            return action
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    @torch.no_grad()
    def predict_image(self, input_ids, attention_mask, proprioception=None, history_actions=None, proprio_attention_mask=None, pixel_values=None, pixel_values_videos=None, image_grid_thw=None, video_grid_thw=None, max_new_tokens=1024, token_type_ids=None, mm_token_type_ids=None):
        _, hidden_states = self.get_vlm_condition(
            input_ids, attention_mask,
            proprioception=self._vlm_proprio(proprioception),
            proprio_attention_mask=proprio_attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            token_type_ids=token_type_ids,
            mm_token_type_ids=mm_token_type_ids
        )
        gen_vlm_ctx = hidden_states
        
        curr_ids = torch.zeros((input_ids.shape[0], 1), dtype=torch.long, device=input_ids.device)
        
        for _ in range(max_new_tokens):
            logits, _ = self.generator(curr_ids, gen_vlm_ctx)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
        generated_tokens = curr_ids[:, 1:]
        H_latent = int(generated_tokens.shape[1]**0.5)
        decoded_images = self.vq_model.decode_code(generated_tokens, shape=(input_ids.shape[0], H_latent, H_latent))
        return decoded_images

if __name__ == "__main__":
    print("Testing VLANeXt Model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    # Initialize Model (Minimal Config)
    model = VLANeXt(
        lmm_path="Qwen/Qwen3-VL-2B-Instruct",
        action_dim=7, num_actions=4, num_history=2,
        backbone_mode="finetune", gradient_checkpointing=False
    ).to(device, dtype)
    processor = model.processor

    def run_test(modality="image"):
        print(f"\n=== Testing {modality.capitalize()} ===")
        B = 2
        # Dummy Data
        img = Image.new('RGB', (64, 64), color='red')
        media = [img] * B if modality == "image" else [[img]*8] * B
        content_key = "image" if modality == "image" else "video"
        
        # Process
        msgs = [[{"role": "user", "content": [{"type": content_key, content_key: m}, {"type": "text", "text": "Task."}]}] for m in media]
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        inputs = processor(text=texts, **{f"{modality}s": media}, padding=True, return_tensors="pt")
        
        # Move to device & cast
        inputs = {k: v.to(device) for k, v in inputs.items()}
        for k in ["pixel_values", "pixel_values_videos"]:
            if k in inputs: inputs[k] = inputs[k].to(dtype)
            
        # Filter valid args for forward
        valid_keys = {"input_ids", "attention_mask", "pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "mm_token_type_ids"}
        fwd_args = {k: v for k, v in inputs.items() if k in valid_keys}

        # Tensors
        act_gt = torch.randn(B, 4, 7, device=device, dtype=dtype)
        proprio = torch.randn(B, 2, 7, device=device, dtype=dtype)
        hist_act = torch.randn(B, 2, 7, device=device, dtype=dtype)

        # Tests
        print(f"Action Gen Loss: {model(actions=act_gt, proprioception=proprio, history_actions=hist_act, **fwd_args).item():.4f}")
        print(f"Action Pred Shape: {model.predict_action(proprioception=proprio, history_actions=hist_act, **fwd_args).shape}")

    run_test("image")
    run_test("video")
    print("\nTest Passed!")
