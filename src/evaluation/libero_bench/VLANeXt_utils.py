import torch
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TVF
from transformers import AutoProcessor, AutoTokenizer, SiglipImageProcessor
from src.models.VLANeXt import VLANeXt, LlamaProcessorWrapper
from src.models.rt2_like_baseline import RT2LikeBaseline
from src.models.smolvla_expert import SmolVLMProcessorWrapper
from src.models.smolvla_vita import (
    SmolVLAMQ54VitaPolicy,
    SmolVLAVitaLatentPolicy,
)
from src.datasets.libero_act import load_libero_mean_std_stats

def _get_checkpoint_path(cfg) -> str:
    if hasattr(cfg, "eval") and hasattr(cfg.eval, "finetuned_checkpoint"):
        return cfg.eval.finetuned_checkpoint
    raise ValueError("cfg.eval.finetuned_checkpoint is required")


def _reshape_legacy_layer_local_query_state(state_dict, model):
    """Compat for old hierarchical-MQ checkpoints saved as [layers, queries, dim]."""
    key = "vita_action_generator.observation_encoder.layer_local_query"
    if key not in state_dict:
        return False
    model_tensor = model.state_dict().get(key)
    ckpt_tensor = state_dict[key]
    if model_tensor is None or tuple(ckpt_tensor.shape) == tuple(model_tensor.shape):
        return False
    if ckpt_tensor.numel() != model_tensor.numel():
        return False
    state_dict[key] = ckpt_tensor.reshape(tuple(model_tensor.shape))
    return True


def get_vla(cfg):
    checkpoint_path = _get_checkpoint_path(cfg)
    print(f"Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_config = checkpoint['config']
    model_config = train_config['model']
    data_config = train_config['data']
    checkpoint_state_dict = checkpoint['model_state_dict']
    edar_token_flow = model_config.get('vita_edar_token_flow')
    if edar_token_flow is None:
        edar_token_flow = any(
            key.removeprefix('module.')
            == 'vita_action_generator.flow.condition_projection.weight'
            for key in checkpoint_state_dict
        )
    
    model_type = model_config.get('model_type', 'vlanext')
    ckpt_backbone_mode = model_config.get('backbone_mode', 'finetune')
    print(f"Model type: {model_type}, backbone_mode: {ckpt_backbone_mode}")

    attn_implementation = model_config.get('attn_implementation', 'flash_attention_2')

    if model_type == 'rt2_baseline':
        model = RT2LikeBaseline(
            lmm_path=model_config['lmm_path'],
            vision_encoder_path=model_config.get('vision_encoder_path', "google/siglip2-base-patch16-256"),
            action_dim=model_config['action_dim'],
            num_actions=data_config['future_len'],
            num_history=data_config['history_len'],
            use_proprio_input_vlm=model_config.get('use_proprio_input_vlm', True),
            use_transformer_projector=model_config.get('use_transformer_proprio_projector', True),
            projector_depth=model_config['projector_depth'],
            projector_num_heads=model_config['projector_num_heads'],
            backbone_mode=ckpt_backbone_mode,
            gradient_checkpointing=False,
            num_bins=model_config.get('num_bins', 256),
            attn_implementation=attn_implementation,
        )
    elif model_type == 'smolvla_mq54_vita':
        num_inference_timesteps = model_config.get('num_inference_timesteps', 6)
        if hasattr(cfg, "model") and hasattr(cfg.model, "diffusion_steps"):
            num_inference_timesteps = cfg.model.diffusion_steps
            print(f"Overriding inference diffusion steps to: {num_inference_timesteps}")
        model = SmolVLAMQ54VitaPolicy.from_config(
            model_config,
            data_config,
            load_action_effect_visual_teacher=False,
            num_inference_timesteps=num_inference_timesteps,
        )
    elif model_type == 'smolvla_vita_latent':
        num_inference_timesteps = model_config.get('num_inference_timesteps', 6)
        if hasattr(cfg, "model") and hasattr(cfg.model, "diffusion_steps"):
            num_inference_timesteps = cfg.model.diffusion_steps
            print(f"Overriding inference diffusion steps to: {num_inference_timesteps}")
        model = SmolVLAVitaLatentPolicy(
            lmm_path=model_config['lmm_path'],
            action_dim=model_config['action_dim'],
            num_actions=data_config['future_len'],
            num_history=data_config['history_len'],
            latent_dim=model_config.get('vita_latent_dim', 1024),
            action_hidden_dim=model_config.get('vita_hidden_dim', 1024),
            action_ae_layers=model_config.get('vita_action_ae_layers', 6),
            action_ae_dropout=model_config.get('vita_dropout', 0.0),
            num_inference_timesteps=num_inference_timesteps,
            num_vlm_layers=model_config.get('smolvla_num_vlm_layers', 16),
            num_expert_layers=model_config.get('smolvla_num_expert_layers', 16),
            expert_width_multiplier=model_config.get(
                'smolvla_expert_width_multiplier', 0.75
            ),
            self_attn_every_n_layers=model_config.get(
                'smolvla_self_attn_every_n_layers', 2
            ),
            attention_mode=model_config.get('smolvla_attention_mode', 'cross_attn'),
            load_vlm_weights=model_config.get('smolvla_load_vlm_weights', True),
            freeze_vlm=True,
            freeze_vision_encoder=True,
            freeze_visual_connector=model_config.get(
                'smolvla_freeze_visual_connector', False
            ),
            state_dim=data_config['history_len'] * model_config['action_dim'],
            enc_recon_weight=model_config.get('vita_enc_recon_weight', 0.2),
            consistency_weight=model_config.get('vita_consistency_weight', 0.0),
            flow_recon_weight=model_config.get('vita_flow_recon_weight', 0.0),
            num_sampling_steps=model_config.get(
                'vita_num_sampling_steps',
                num_inference_timesteps,
            ),
            rollout_gradient_checkpointing=model_config.get(
                'vita_rollout_gradient_checkpointing', True
            ),
            action_effect_enabled=model_config.get('vita_action_effect_enabled', True),
            action_effect_visual_teacher_type=model_config.get(
                'vita_action_effect_visual_teacher', 'dinov2'
            ),
            action_effect_visual_teacher_path=model_config.get(
                'vita_action_effect_visual_teacher_path', None
            ),
            load_action_effect_visual_teacher=False,
            action_effect_visual_ae_weight=model_config.get(
                'vita_action_effect_visual_ae_weight', 0.01
            ),
            action_effect_decoder_layers=model_config.get(
                'vita_action_effect_decoder_layers', 2
            ),
            action_representation_type=model_config.get(
                'vita_action_representation_type', 'legacy'
            ),
            edar_stage_a_checkpoint=model_config.get(
                'vita_edar_stage_a_checkpoint', None
            ),
            edar_train_decoder=model_config.get(
                'vita_edar_train_decoder', False
            ),
        )
    else:
        num_train_timesteps = model_config.get('num_train_timesteps', model_config.get('diffusion_steps', 1000))
        ckpt_inf_steps = model_config.get('num_inference_timesteps', model_config.get('diffusion_steps', 10))
        
        num_inference_timesteps = ckpt_inf_steps
        if hasattr(cfg, "model") and hasattr(cfg.model, "diffusion_steps"):
            num_inference_timesteps = cfg.model.diffusion_steps
            print(f"Overriding inference diffusion steps to: {num_inference_timesteps}")

        scheduler_type = model_config['scheduler_type']
        if hasattr(cfg, "model"):
            scheduler_type = getattr(cfg.model, "scheduler_type", scheduler_type)

        use_action_input_policy = model_config.get('use_action_input_policy', False)
        use_transformer_proprio_projector = model_config.get('use_transformer_proprio_projector', False)

        model = VLANeXt(
            lmm_path=model_config['lmm_path'],
            vision_encoder_path=model_config.get('vision_encoder_path', "google/siglip2-base-patch16-256"),
            action_dim=model_config['action_dim'],
            num_actions=data_config['future_len'],
            num_queries=model_config['num_queries'],
            num_history=data_config['history_len'],
            loss_type=model_config.get('loss_type', 'diffusion'),
            future_image_loss_weight=float(model_config.get('future_image_loss_weight', 0.0)),
            num_train_timesteps=num_train_timesteps,
            num_inference_timesteps=num_inference_timesteps,
            scheduler_type=scheduler_type,
            condition_type=model_config.get('condition_type', 'loose'),
            policy_hidden_size=model_config.get('policy_hidden_size', 1024),
            policy_depth=model_config.get('policy_depth', 29),
            policy_num_heads=model_config.get('policy_num_heads', 12),
            policy_mlp_ratio=model_config.get('policy_mlp_ratio', 4.0),
            use_proprio_input_vlm=model_config.get('use_proprio_input_vlm', True),
            use_action_input_policy=use_action_input_policy,
            use_transformer_proprio_projector=use_transformer_proprio_projector,
            projector_depth=model_config['projector_depth'],
            projector_num_heads=model_config['projector_num_heads'],
            use_transformer_connector=model_config['use_transformer_connector'],
            connector_depth=model_config['connector_depth'],
            connector_num_heads=model_config['connector_num_heads'],
            backbone_mode=ckpt_backbone_mode,
            gradient_checkpointing=False,
            num_bins=model_config.get('num_bins', 256),
            action_vqvae=model_config.get('action_vqvae', None),

            generator_hidden_size=model_config.get('generator_hidden_size', 768),
            generator_depth=model_config.get('generator_depth', 12),
            generator_num_heads=model_config.get('generator_num_heads', 12),
            generator_mlp_ratio=model_config.get('generator_mlp_ratio', 4.0),
            attn_implementation=attn_implementation,
            dct_loss_weight=model_config.get('dct_loss_weight', 0.1),
            dct_low_freq_weight=model_config.get('dct_low_freq_weight', 1.0),
            dct_high_freq_weight=model_config.get('dct_high_freq_weight', 1.0),
            dct_freq_split=model_config.get('dct_freq_split', 0.125),
            dct_similarity_type=model_config.get('dct_similarity_type', 'mae'),
            use_progress_head=model_config.get('use_progress_head', False) or model_config.get('use_progress_film', False),
            use_progress_film=model_config.get('use_progress_film', False),
            progress_loss_weight=model_config.get('progress_loss_weight', 0.0),
            progress_detach=model_config.get('progress_detach', True),
            progress_noise=0.0,
            use_vlm_layer_attention=model_config.get('use_vlm_layer_attention', False),
            vlm_layer_attention_alpha=model_config.get('vlm_layer_attention_alpha', 0.0),
            vlm_layer_indices=model_config.get('vlm_layer_indices', None),
            vlm_truncate_layers=model_config.get('vlm_truncate_layers', None),
            use_latent_bridge=model_config.get('use_latent_bridge', False),
            latent_bridge_num_heads=model_config.get('latent_bridge_num_heads', 8),
            latent_bridge_max_views=model_config.get('latent_bridge_max_views', 4),
            latent_bridge_visual_trainable=model_config.get('latent_bridge_visual_trainable', True),
            latent_bridge_vlm_uses_proprio=model_config.get('latent_bridge_vlm_uses_proprio', False),
            action_generation_mode=model_config.get('action_generation_mode', 'direct_flow'),
            vita_latent_dim=model_config.get('vita_latent_dim', 512),
            vita_hidden_dim=model_config.get('vita_hidden_dim', 512),
            vita_action_ae_layers=model_config.get('vita_action_ae_layers', 4),
            vita_action_encoder_type=model_config.get(
                'vita_action_encoder_type', 'mlp'
            ),
            vita_action_cnn_layers=model_config.get(
                'vita_action_cnn_layers', 4
            ),
            vita_action_cnn_kernel_size=model_config.get(
                'vita_action_cnn_kernel_size', 5
            ),
            vita_flow_layers=model_config.get('vita_flow_layers', 4),
            vita_flow_mlp_ratio=model_config.get('vita_flow_mlp_ratio', 4.0),
            vita_dropout=model_config.get('vita_dropout', 0.0),
            vita_num_sampling_steps=model_config.get('vita_num_sampling_steps', 6),
            vita_enc_recon_weight=model_config.get('vita_enc_recon_weight', 0.5),
            vita_flow_recon_weight=model_config.get('vita_flow_recon_weight', 0.5),
            vita_consistency_weight=model_config.get('vita_consistency_weight', 1.0),
            vita_diversity_loss_weight=model_config.get('vita_diversity_loss_weight', 0.0),
            vita_log_diversity_loss=model_config.get('vita_log_diversity_loss', False),
            vita_action_progress_loss_weight=model_config.get('vita_action_progress_loss_weight', 0.0),
            vita_goal_distance_loss_weight=model_config.get('vita_goal_distance_loss_weight', 0.0),
            vita_goal_metric_dim=model_config.get('vita_goal_metric_dim', 256),
            vita_goal_distance_loss_type=model_config.get('vita_goal_distance_loss_type', 'huber'),
            vita_goal_distance_detach_goal=model_config.get('vita_goal_distance_detach_goal', False),
            vita_recon_loss_type=model_config.get('vita_recon_loss_type', 'l1'),
            vita_use_proprio=model_config.get('vita_use_proprio', True),
            vita_zobs_mode=model_config.get('vita_zobs_mode', 'normal'),
            vita_zobs_noise_scale=model_config.get('vita_zobs_noise_scale', 1.0),
            vita_condition_source=model_config.get('vita_condition_source', 'auto'),
            vita_hidden_layer_index=model_config.get('vita_hidden_layer_index', -1),
            vita_secondary_hidden_layer_index=model_config.get('vita_secondary_hidden_layer_index', None),
            vita_extra_hidden_layer_indices=model_config.get('vita_extra_hidden_layer_indices', None),
            vita_hidden_pooling=model_config.get('vita_hidden_pooling', 'mean'),
            vita_pooling_num_heads=model_config.get('vita_pooling_num_heads', 8),
            vita_pooling_num_queries=model_config.get('vita_pooling_num_queries', 1),
            vita_gated_weighted_pooling=model_config.get('vita_gated_weighted_pooling', False),
            vita_obs_pool_token_indices=model_config.get('vita_obs_pool_token_indices', None),
            vita_hierarchical_query_pooling=model_config.get('vita_hierarchical_query_pooling', False),
            vita_layer_local_queries_per_layer=model_config.get('vita_layer_local_queries_per_layer', 2),
            vita_layer_local_queries_per_layer_list=model_config.get('vita_layer_local_queries_per_layer_list', None),
            vita_layer_local_token_source_modes=model_config.get('vita_layer_local_token_source_modes', None),
            hier_mq_separate_views=model_config.get('hier_mq_separate_views', False),
            vita_cross_layer_queries=model_config.get('vita_cross_layer_queries', 18),
            vita_global_queries=model_config.get('vita_global_queries', 8),
            vita_adaptive_local_mq=model_config.get('vita_adaptive_local_mq', False),
            vita_adaptive_mq_router_dim=model_config.get('vita_adaptive_mq_router_dim', 256),
            vita_adaptive_mq_num_slots=model_config.get('vita_adaptive_mq_num_slots', 4),
            vita_adaptive_mq_reserve_source_positions=model_config.get(
                'vita_adaptive_mq_reserve_source_positions',
                None,
            ),
            vita_adaptive_mq_temperature=model_config.get('vita_adaptive_mq_temperature', 1.0),
            vita_adaptive_mq_route_warmup_steps=model_config.get(
                'vita_adaptive_mq_route_warmup_steps',
                0,
            ),
            vita_dynamic_topk_local_mq=model_config.get(
                'vita_dynamic_topk_local_mq',
                False,
            ),
            vita_dynamic_topk_candidate_source_positions=model_config.get(
                'vita_dynamic_topk_candidate_source_positions',
                None,
            ),
            vita_dynamic_topk_text_source_position=model_config.get(
                'vita_dynamic_topk_text_source_position',
                1,
            ),
            vita_dynamic_topk_candidates_per_source=model_config.get(
                'vita_dynamic_topk_candidates_per_source',
                8,
            ),
            vita_dynamic_topk_min_keep_per_source=model_config.get(
                'vita_dynamic_topk_min_keep_per_source',
                2,
            ),
            vita_dynamic_topk_total_keep=model_config.get(
                'vita_dynamic_topk_total_keep',
                24,
            ),
            vita_dynamic_topk_train_random_exploration=model_config.get(
                'vita_dynamic_topk_train_random_exploration',
                False,
            ),
            vita_condition_type_embeddings=model_config.get('vita_condition_type_embeddings', False),
            vita_dynamic_extra_layer_gates=model_config.get('vita_dynamic_extra_layer_gates', False),
            vita_extra_layer_gate_scales=model_config.get('vita_extra_layer_gate_scales', None),
            vita_extra_layer_gate_hidden_dim=model_config.get('vita_extra_layer_gate_hidden_dim', 256),
            vita_extra_layer_gate_dropout=model_config.get('vita_extra_layer_gate_dropout', 0.0),
            vita_blockwise_layer_conditioning=model_config.get('vita_blockwise_layer_conditioning', False),
            vita_blockwise_hidden_layer_indices=model_config.get('vita_blockwise_hidden_layer_indices', None),
            vita_state_token_conditioning=model_config.get('vita_state_token_conditioning', False),
            vita_state_num_tokens=model_config.get('vita_state_num_tokens', 4),
            vita_state_token_dropout=model_config.get('vita_state_token_dropout', 0.0),
            vita_state_broadcast_to_mq=model_config.get('vita_state_broadcast_to_mq', True),
            vita_mq_local_overlap_loss_weight=model_config.get('vita_mq_local_overlap_loss_weight', 0.0),
            vita_mq_cross_overlap_loss_weight=model_config.get('vita_mq_cross_overlap_loss_weight', 0.0),
            vita_mq_global_overlap_loss_weight=model_config.get('vita_mq_global_overlap_loss_weight', 0.0),
            vita_mq_competitive_local_attention=model_config.get('vita_mq_competitive_local_attention', False),
            vita_mq_competitive_attention_tau=model_config.get('vita_mq_competitive_attention_tau', 1.0),
            vita_mq_competitive_attention_gamma=model_config.get('vita_mq_competitive_attention_gamma', 1.0),
            vita_mq_local_balance_loss_weight=model_config.get('vita_mq_local_balance_loss_weight', 0.0),
            vita_hierarchical_global_residual_block=model_config.get('vita_hierarchical_global_residual_block', False),
            vita_hierarchical_global_residual_scale=model_config.get('vita_hierarchical_global_residual_scale', 0.1),
            vita_hierarchical_global_ffn_ratio=model_config.get('vita_hierarchical_global_ffn_ratio', 4.0),
            vita_layer_aligned_mq=model_config.get('vita_layer_aligned_mq', False),
            vita_layer_aligned_hidden_layer_indices=model_config.get('vita_layer_aligned_hidden_layer_indices', None),
            vita_layer_aligned_queries_per_layer=model_config.get('vita_layer_aligned_queries_per_layer', 4),
            vita_layer_aligned_token_source_modes=model_config.get('vita_layer_aligned_token_source_modes', None),
            vita_layer_aligned_global_conditioning=model_config.get('vita_layer_aligned_global_conditioning', False),
            vita_layer_aligned_obs_pooling=model_config.get('vita_layer_aligned_obs_pooling', 'last'),
            vita_layer_aligned_encoder_obs_conditioning=model_config.get('vita_layer_aligned_encoder_obs_conditioning', True),
            vita_layer_aligned_flow_windows=model_config.get('vita_layer_aligned_flow_windows', None),
            vita_flow_memory_tokens=model_config.get('vita_flow_memory_tokens', 0),
            vita_flow_memory_update_scale=model_config.get('vita_flow_memory_update_scale', 0.1),
            vita_flow_memory_mlp_ratio=model_config.get('vita_flow_memory_mlp_ratio', 4.0),
            vita_flow_memory_update_type=model_config.get('vita_flow_memory_update_type', 'attention'),
            vita_flow_memory_shared_update=model_config.get('vita_flow_memory_shared_update', False),
            vita_flow_memory_bottleneck_dim=model_config.get('vita_flow_memory_bottleneck_dim', 256),
            vita_flow_layer_logits_init=model_config.get('vita_flow_layer_logits_init', None),
            vita_flow_cross_attention=model_config.get('vita_flow_cross_attention', False),
            vita_flow_cross_attention_heads=model_config.get('vita_flow_cross_attention_heads', 8),
            vita_flow_cross_attention_dim=model_config.get('vita_flow_cross_attention_dim', None),
            vita_gated_flow_cross_attention=model_config.get('vita_gated_flow_cross_attention', False),
            vita_typed_flow_cross_attention=model_config.get('vita_typed_flow_cross_attention', False),
            vita_dynamic_typed_flow_gates=model_config.get('vita_dynamic_typed_flow_gates', False),
            vita_mq_token_value_gating=model_config.get('vita_mq_token_value_gating', False),
            vita_mq_headwise_token_value_gating=model_config.get('vita_mq_headwise_token_value_gating', False),
            vita_mq_token_gate_lambda=model_config.get('vita_mq_token_gate_lambda', 0.1),
            vita_mq_token_gate_use_action_latent=model_config.get('vita_mq_token_gate_use_action_latent', False),
            vita_mq_token_gate_centered_residual=model_config.get('vita_mq_token_gate_centered_residual', False),
            vita_mq_competitive_value_gating=model_config.get('vita_mq_competitive_value_gating', False),
            vita_mq_global_relative_context_gating=model_config.get(
                'vita_mq_global_relative_context_gating',
                False,
            ),
            vita_mq_global_relative_gate_score_dim=model_config.get(
                'vita_mq_global_relative_gate_score_dim',
                128,
            ),
            vita_mq_global_relative_gate_temperature=model_config.get(
                'vita_mq_global_relative_gate_temperature',
                1.0,
            ),
            vita_flow_static_condition_cache=model_config.get(
                'vita_flow_static_condition_cache',
                False,
            ),
            vita_flow_cross_attention_fixed_scale=model_config.get('vita_flow_cross_attention_fixed_scale', None),
            vita_action_representation_type=model_config.get(
                'vita_action_representation_type',
                'legacy',
            ),
            vita_edar_stage_a_checkpoint=model_config.get(
                'vita_edar_stage_a_checkpoint',
                None,
            ),
            vita_edar_train_decoder=model_config.get(
                'vita_edar_train_decoder', False
            ),
            vita_edar_token_flow=edar_token_flow,
            vita_mq_type=model_config.get('vita_mq_type', 'hier_mq54'),
            vita_enable_one_step_action_loss=model_config.get(
                'enable_one_step_action_loss', False
            ),
            vita_flow_loss_weight_final=model_config.get(
                'flow_loss_weight_final', 0.3
            ),
            vita_one_step_action_loss_weight_final=model_config.get(
                'one_step_action_loss_weight_final', 1.0
            ),
            vita_stage_b_max_train_steps=data_config.get('max_steps', 1),
            dynamic_flow_reconstruction_weight=model_config.get(
                'dynamic_flow_reconstruction_weight', False
            ),
            dynamic_flow_latent_reconstruction_weight=model_config.get(
                'dynamic_flow_latent_reconstruction_weight', False
            ),
            vita_action_effect_enabled=model_config.get('vita_action_effect_enabled', False),
            vita_action_effect_condition_action_encoder=model_config.get(
                'vita_action_effect_condition_action_encoder', True
            ),
            vita_action_effect_visual_queries=model_config.get('vita_action_effect_visual_queries', 4),
            vita_action_effect_visual_teacher=model_config.get(
                'vita_action_effect_visual_teacher', 'qwen'
            ),
            vita_action_effect_spatial_grid_size=model_config.get(
                'vita_action_effect_spatial_grid_size', 8
            ),
            vita_action_effect_visual_teacher_path=model_config.get(
                'vita_action_effect_visual_teacher_path', None
            ),
            vita_action_effect_load_visual_teacher=False,
            vita_action_effect_num_heads=model_config.get('vita_action_effect_num_heads', 8),
            vita_action_effect_encoder_layers=model_config.get('vita_action_effect_encoder_layers', 2),
            vita_action_effect_decoder_layers=model_config.get('vita_action_effect_decoder_layers', 2),
            vita_action_effect_projection_dim=model_config.get('vita_action_effect_projection_dim', None),
            vita_action_effect_state_conditioning=model_config.get(
                'vita_action_effect_state_conditioning', False
            ),
            vita_action_effect_visual_ae_weight=model_config.get('vita_action_effect_visual_ae_weight', 0.05),
            vita_action_effect_visual_flow_weight=model_config.get('vita_action_effect_visual_flow_weight', 0.05),
            vita_action_effect_rank_weight=model_config.get('vita_action_effect_rank_weight', 0.0),
            vita_action_effect_rank_margin=model_config.get('vita_action_effect_rank_margin', 0.1),
            vita_action_effect_rank_min_delta_norm=model_config.get(
                'vita_action_effect_rank_min_delta_norm', 0.0
            ),
            vita_action_effect_rank_distance_mode=model_config.get(
                'vita_action_effect_rank_distance_mode', 'global_cosine'
            ),
            vita_action_effect_rank_huber_weight=model_config.get(
                'vita_action_effect_rank_huber_weight', 0.0
            ),
            vita_action_effect_rank_scale_floor=model_config.get(
                'vita_action_effect_rank_scale_floor', 1e-3
            ),
            vita_action_effect_negative_queue_size=model_config.get(
                'vita_action_effect_negative_queue_size', 0
            ),
            vita_action_effect_num_negatives=model_config.get(
                'vita_action_effect_num_negatives', 1
            ),
            vita_action_effect_hard_negatives=model_config.get(
                'vita_action_effect_hard_negatives', 1
            ),
            vita_action_effect_negative_min_action_distance=model_config.get(
                'vita_action_effect_negative_min_action_distance', 0.0
            ),
            vita_action_effect_main_view_only=model_config.get('vita_action_effect_main_view_only', False),
            vita_action_effect_horizons=model_config.get(
                'vita_action_effect_horizons', None
            ),
            vita_action_effect_horizon_loss_weights=model_config.get(
                'vita_action_effect_horizon_loss_weights', None
            ),
            vita_action_effect_action_token_dim=model_config.get(
                'vita_action_effect_action_token_dim', 256
            ),
            vita_action_effect_multihorizon_layers=model_config.get(
                'vita_action_effect_multihorizon_layers', 2
            ),
            vita_action_effect_motion_weight_floor=model_config.get(
                'vita_action_effect_motion_weight_floor', 0.25
            ),
            vita_action_effect_motion_weight_eps=model_config.get(
                'vita_action_effect_motion_weight_eps', 1e-6
            ),
            vita_action_effect_detach_visual_teacher=model_config.get(
                'vita_action_effect_detach_visual_teacher', False
            ),
            vlanext_dual_flow=model_config.get('vlanext_dual_flow', False),
            vlanext_visual_flow_loss_weight=model_config.get('vlanext_visual_flow_loss_weight', 0.0),
        )
    
    
    state_dict = checkpoint_state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    if _reshape_legacy_layer_local_query_state(state_dict, model):
        print("Reshaped legacy layer_local_query checkpoint tensor for current model.")
        
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded state dict. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device, dtype=torch.bfloat16)
    model.eval()
    
    model.train_config = train_config
    model.libero_normalization_stats = None
    if (
        data_config.get("dataset_name", "libero") == "libero"
        and data_config.get("normalization_mode", "min_max") == "mean_std"
    ):
        model.libero_normalization_stats = load_libero_mean_std_stats(
            data_config.get("normalization_stats_path")
        )
    
    return model

def get_processor(cfg):
    checkpoint_path = _get_checkpoint_path(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    lmm_path = config["model"]["lmm_path"]
    model_type = config["model"].get("model_type", "vlanext")

    if model_type in {"smolvla_vita_latent", "smolvla_mq54_vita"}:
        return SmolVLMProcessorWrapper.from_pretrained(lmm_path)

    if "llama" in lmm_path.lower():
        vision_encoder_path = config["model"].get("vision_encoder_path", "google/siglip2-base-patch16-256")
        tokenizer = AutoTokenizer.from_pretrained(lmm_path)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        image_processor = SiglipImageProcessor.from_pretrained(vision_encoder_path)
        return LlamaProcessorWrapper(tokenizer, image_processor)

    return AutoProcessor.from_pretrained(lmm_path, trust_remote_code=True)

def get_vla_action(cfg, model, processor, obs, task_label):
    data_cfg = model.train_config["data"]
    input_modality = data_cfg.get("input_modality", "image")
    view_mode = data_cfg.get("view_mode", "single")
    fps = float(data_cfg.get("fps", 20.0))
    history_len = getattr(model, "num_history", 0)
    device = next(model.parameters()).device
    effective_processor = getattr(model, "processor", processor)
    use_latent_bridge = bool(getattr(model, "use_latent_bridge", False))
    bridge_image_size = int(model.train_config["model"].get("latent_bridge_image_size", 224))
    train_refresh_interval = model.train_config["model"].get("latent_bridge_refresh_interval", 1)
    eval_refresh_interval = getattr(getattr(cfg, "eval", None), "latent_bridge_refresh_interval", train_refresh_interval)
    refresh_interval = max(1, int(eval_refresh_interval))
    bridge_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    bridge_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    is_paligemma = "PaliGemma" in effective_processor.__class__.__name__
    is_qwen = "Qwen" in effective_processor.__class__.__name__
    is_smol = "SmolVLM" in effective_processor.__class__.__name__
    is_llama = "Llama" in effective_processor.__class__.__name__

    def _take_last(history_list, fallback, n: int):
        if not history_list:
            history_list = [fallback]
        if n > 0:
            xs = history_list[-n:]
            if len(xs) < n:
                xs = ([xs[0]] * (n - len(xs))) + xs
            return xs
        return [fallback]

    all_images = obs.get("image_history", [obs["full_image"]])
    images_np = _take_last(all_images, obs["full_image"], history_len)

    if view_mode == "multi":
        all_wrist = obs.get("image_history_wrist", [obs["full_image_wrist"]])
        wrist_np = _take_last(all_wrist, obs["full_image_wrist"], history_len)

    pil_images = [Image.fromarray(img) for img in images_np]
    if view_mode == "multi":
        pil_wrist = [Image.fromarray(img) for img in wrist_np]

    proprioception = None
    if model.use_proprio_input_vlm or (
        getattr(model, "use_vita_latent_flow", False)
        and getattr(model, "vita_use_proprio", True)
    ):
        all_states = obs.get("state_history", [])
        if history_len > 0:
            states = all_states[-history_len:]
            if len(states) < history_len:
                states = ([states[0]] * (history_len - len(states))) + states if states else [np.zeros(model.action_dim)] * history_len
        else:
            states = []
        if len(states) > 0:
            states_np = np.stack(states).astype(np.float32, copy=False)
            normalization_stats = getattr(model, "libero_normalization_stats", None)
            if normalization_stats is not None:
                state_stats = normalization_stats["state"]
                states_np = (
                    states_np - state_stats["mean"]
                ) / state_stats["std"]
            proprioception = torch.tensor(states_np, dtype=torch.bfloat16).unsqueeze(0).to(device)

    history_actions = None
    if getattr(model, 'use_action_input_policy', False):
        all_actions = obs.get("action_history", [])
        if history_len > 0:
            actions = all_actions[-history_len:]
            if len(actions) < history_len:
                actions = ([np.zeros(model.action_dim)] * (history_len - len(actions))) + actions
        else:
            actions = []
        if len(actions) > 0:
            history_actions = torch.tensor(np.stack(actions), dtype=torch.bfloat16).unsqueeze(0).to(device)

    inputs = {}
    bridge_images = None

    def _bridge_tensor(img):
        img = TVF.resize(img, [bridge_image_size, bridge_image_size])
        x = torch.from_numpy(np.asarray(img, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0
        return (x - bridge_mean) / bridge_std
    
    if input_modality == "video":
        if is_paligemma or is_llama:
            raise ValueError(f"{proc_cls_name} implementation in VLANeXt currently supports 'image' modality only.")
        content = []
        if view_mode == "multi":
            content.extend([{"type": "video", "video": pil_images}, {"type": "video", "video": pil_wrist}])
            videos = [pil_images, pil_wrist]
        else:
            content.append({"type": "video", "video": pil_images})
            videos = [pil_images]

        content.append({"type": "text", "text": task_label})
        messages = [{"role": "user", "content": content}]
        text = effective_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = effective_processor(
            text=[text],
            videos=videos,
            videos_kwargs={"fps": fps, "return_metadata": True},
            padding=True,
            return_tensors="pt",
        )

    elif input_modality == "image":
        if view_mode == "multi":
            images = [pil_images[-1], pil_wrist[-1]]
        else:
            images = [pil_images[-1]]
        if use_latent_bridge:
            bridge_images = torch.stack([_bridge_tensor(img) for img in images], dim=0).unsqueeze(0)

        if is_llama:
            text = [task_label]
            inputs = effective_processor.tokenizer(text, padding=True, return_tensors="pt")
            image_inputs = effective_processor.image_processor(images, return_tensors="pt")
            inputs["pixel_values"] = image_inputs["pixel_values"]

        elif is_paligemma:
            text = "<image>" * len(images) + task_label
            inputs = effective_processor(
                text=[text],
                images=images,
                padding=True,
                return_tensors="pt",
            )
        elif is_qwen:
            content = []
            for img in images:
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": task_label})
            
            messages = [{"role": "user", "content": content}]
            text = effective_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = effective_processor(
                text=[text],
                images=images,
                padding=True,
                return_tensors="pt",
            )
        elif is_smol:
            inputs = effective_processor.tokenizer(
                [task_label.rstrip() + "\n"],
                padding=True,
                max_length=48,
                truncation=True,
                return_tensors="pt",
            )
            image_inputs = effective_processor.image_processor(
                [images],
                return_tensors="pt",
            )
            inputs["pixel_values"] = image_inputs["pixel_values"]
            if "pixel_attention_mask" in image_inputs:
                inputs["pixel_attention_mask"] = image_inputs[
                    "pixel_attention_mask"
                ]
    else:
        raise ValueError(f"Unknown input_modality: {input_modality} for model type")

    valid_keys = {
        "input_ids", "attention_mask", "pixel_values", "pixel_attention_mask",
        "pixel_values_videos", "image_grid_thw", "video_grid_thw",
        "token_type_ids", "mm_token_type_ids",
    }
    inputs = {k: v.to(device) for k, v in inputs.items() if k in valid_keys}

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    if "pixel_values_videos" in inputs:
        inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(torch.bfloat16)
    if bridge_images is not None:
        inputs["bridge_pixel_values"] = bridge_images.to(device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        cached_vlm_condition = None
        if use_latent_bridge and refresh_interval > 1:
            cache_step = int(getattr(model, "_latent_bridge_cache_step", 0))
            if (
                not hasattr(model, "_latent_bridge_vlm_cache")
                or model._latent_bridge_vlm_cache is None
                or cache_step % refresh_interval == 0
            ):
                condition_keys = {
                    "input_ids",
                    "attention_mask",
                    "pixel_values",
                    "pixel_values_videos",
                    "image_grid_thw",
                    "video_grid_thw",
                    "token_type_ids",
                    "mm_token_type_ids",
                }
                condition_inputs = {k: v for k, v in inputs.items() if k in condition_keys}
                model._latent_bridge_vlm_cache = model.get_vlm_condition(
                    proprioception=model._vlm_proprio(proprioception),
                    proprio_attention_mask=None,
                    **condition_inputs,
                )
            cached_vlm_condition = model._latent_bridge_vlm_cache
            model._latent_bridge_cache_step = cache_step + 1
        action_pred = model.predict_action(
            proprioception=proprioception,
            history_actions=history_actions,
            cached_vlm_condition=cached_vlm_condition,
            **inputs,
        )

    return action_pred[0].float().cpu().numpy()
