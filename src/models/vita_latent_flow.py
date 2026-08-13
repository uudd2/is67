import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.edar_lite import (
    SingleViewEDARLiteDecoder,
    SingleViewEDARLiteEncoder,
)


def _mlp(dim, hidden_dim, dropout):
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.GELU(approximate="tanh"),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
    )


def action_effect_ranking_loss(
    positive_prediction,
    negative_prediction,
    target,
    margin=0.1,
    min_target_norm=0.0,
    negative_mask=None,
    hard_negative_count=None,
    negative_action_distance=None,
    distance_mode="global_cosine",
    huber_weight=0.0,
    scale_floor=1e-3,
):
    """Rank the matched action effect ahead of one or more mismatched actions."""
    target = target.detach().float()
    positive_prediction = positive_prediction.float()
    if target.shape != positive_prediction.shape:
        raise ValueError("Positive prediction and target must share feature shape.")
    if negative_prediction.ndim == positive_prediction.ndim:
        negative_prediction = negative_prediction.unsqueeze(1)
    elif negative_prediction.ndim != positive_prediction.ndim + 1:
        raise ValueError(
            "negative_prediction must have shape [B, ...] or [B, K, ...]."
        )
    negative_prediction = negative_prediction.float()
    target_flat = target.flatten(1)
    positive_flat = positive_prediction.flatten(1)
    negative_flat = negative_prediction.flatten(2)
    if negative_flat.shape[0] != positive_flat.shape[0]:
        raise ValueError("Positive and negative predictions must share batch size.")
    if negative_flat.shape[-1] != positive_flat.shape[-1]:
        raise ValueError("Positive and negative predictions must share feature shape.")

    target_norm = target_flat.norm(dim=-1)
    distance_mode = str(distance_mode)
    valid_distance_modes = {"global_cosine", "tokenwise_cosine", "tokenwise_mixed"}
    if distance_mode not in valid_distance_modes:
        raise ValueError(
            f"Unknown action-effect distance mode: {distance_mode}. "
            f"Expected one of {sorted(valid_distance_modes)}."
        )
    if float(huber_weight) < 0:
        raise ValueError("Action-effect Huber weight must be non-negative.")
    if float(scale_floor) <= 0:
        raise ValueError("Action-effect scale floor must be positive.")

    if distance_mode == "global_cosine":
        target_unit = F.normalize(target_flat, dim=-1)
        positive_unit = F.normalize(positive_flat, dim=-1)
        negative_unit = F.normalize(negative_flat, dim=-1)
        positive_direction_distance = 1.0 - (
            positive_unit * target_unit
        ).sum(dim=-1)
        negative_direction_distance = 1.0 - (
            negative_unit * target_unit.unsqueeze(1)
        ).sum(dim=-1)
    else:
        if positive_prediction.ndim == 2:
            target_tokens = target.unsqueeze(1)
            positive_tokens = positive_prediction.unsqueeze(1)
            negative_tokens = negative_prediction.unsqueeze(2)
        else:
            target_tokens = target.flatten(2)
            positive_tokens = positive_prediction.flatten(2)
            negative_tokens = negative_prediction.flatten(3)

        token_norm = target_tokens.norm(dim=-1)
        valid_tokens = token_norm > torch.finfo(torch.float32).eps
        valid_token_count = valid_tokens.float().sum(dim=1).clamp_min(1.0)
        target_token_unit = F.normalize(target_tokens, dim=-1)
        positive_token_unit = F.normalize(positive_tokens, dim=-1)
        negative_token_unit = F.normalize(negative_tokens, dim=-1)
        positive_token_distance = 1.0 - (
            positive_token_unit * target_token_unit
        ).sum(dim=-1)
        negative_token_distance = 1.0 - (
            negative_token_unit * target_token_unit.unsqueeze(1)
        ).sum(dim=-1)
        positive_direction_distance = (
            positive_token_distance * valid_tokens.float()
        ).sum(dim=1) / valid_token_count
        negative_direction_distance = (
            negative_token_distance * valid_tokens.unsqueeze(1).float()
        ).sum(dim=2) / valid_token_count.unsqueeze(1)

    target_rms = target_flat.square().mean(dim=-1).sqrt().detach()
    if distance_mode == "tokenwise_mixed":
        effect_scale = target_rms.clamp_min(float(scale_floor))
        normalized_target = target_flat / effect_scale.unsqueeze(-1)
        normalized_positive = positive_flat / effect_scale.unsqueeze(-1)
        normalized_negative = negative_flat / effect_scale[:, None, None]
        positive_magnitude_distance = F.smooth_l1_loss(
            normalized_positive,
            normalized_target,
            reduction="none",
        ).mean(dim=-1)
        negative_magnitude_distance = F.smooth_l1_loss(
            normalized_negative,
            normalized_target.unsqueeze(1),
            reduction="none",
        ).mean(dim=-1)
    else:
        positive_magnitude_distance = torch.zeros_like(positive_direction_distance)
        negative_magnitude_distance = torch.zeros_like(negative_direction_distance)

    positive_distance = (
        positive_direction_distance
        + float(huber_weight) * positive_magnitude_distance
    )
    negative_distance = (
        negative_direction_distance
        + float(huber_weight) * negative_magnitude_distance
    )

    threshold = max(float(min_target_norm), torch.finfo(torch.float32).eps)
    valid_positive = target_norm > threshold
    if negative_mask is None:
        negative_mask = torch.ones_like(negative_distance, dtype=torch.bool)
    else:
        negative_mask = negative_mask.to(
            device=negative_distance.device,
            dtype=torch.bool,
        )
        if negative_mask.shape != negative_distance.shape:
            raise ValueError(
                "negative_mask must have shape [B, K] matching negative predictions."
            )
    valid_pairs = valid_positive.unsqueeze(1) & negative_mask
    rank_per_pair = F.relu(
        float(margin) + positive_distance.unsqueeze(1) - negative_distance
    )

    num_negatives = negative_distance.shape[1]
    hard_count = (
        num_negatives
        if hard_negative_count is None
        else max(1, min(int(hard_negative_count), num_negatives))
    )
    hard_scores = rank_per_pair.masked_fill(~valid_pairs, float("-inf"))
    hard_values, hard_indices = hard_scores.topk(hard_count, dim=1)
    hard_valid = valid_pairs.gather(1, hard_indices)
    hard_values = torch.where(
        hard_valid,
        hard_values,
        torch.zeros_like(hard_values),
    )
    selected_negative_distance = negative_distance.gather(1, hard_indices)
    selected_positive_distance = positive_distance.unsqueeze(1).expand_as(
        selected_negative_distance
    )
    hard_valid_float = hard_valid.float()
    selected_pair_count = hard_valid_float.sum().clamp_min(1.0)
    rank_loss = hard_values.sum() / selected_pair_count
    positive_mean = (
        selected_positive_distance * hard_valid_float
    ).sum() / selected_pair_count
    negative_mean = (
        selected_negative_distance * hard_valid_float
    ).sum() / selected_pair_count
    selected_positive_direction = positive_direction_distance.unsqueeze(1).expand_as(
        selected_negative_distance
    )
    selected_negative_direction = negative_direction_distance.gather(1, hard_indices)
    positive_direction_mean = (
        selected_positive_direction * hard_valid_float
    ).sum() / selected_pair_count
    negative_direction_mean = (
        selected_negative_direction * hard_valid_float
    ).sum() / selected_pair_count
    selected_positive_magnitude = positive_magnitude_distance.unsqueeze(1).expand_as(
        selected_negative_distance
    )
    selected_negative_magnitude = negative_magnitude_distance.gather(1, hard_indices)
    positive_magnitude_mean = (
        selected_positive_magnitude * hard_valid_float
    ).sum() / selected_pair_count
    negative_magnitude_mean = (
        selected_negative_magnitude * hard_valid_float
    ).sum() / selected_pair_count
    selected_gap = selected_negative_distance - selected_positive_distance
    shuffle_gap = (selected_gap * hard_valid_float).sum() / selected_pair_count
    rank_accuracy = (
        (
            selected_negative_distance
            >= selected_positive_distance + float(margin)
        ).float()
        * hard_valid_float
    ).sum() / selected_pair_count

    has_valid_negative = valid_pairs.any(dim=1)
    hardest_negative_distance = negative_distance.masked_fill(
        ~valid_pairs,
        float("inf"),
    ).min(dim=1).values
    hardest_gap = torch.where(
        has_valid_negative,
        hardest_negative_distance - positive_distance,
        torch.zeros_like(positive_distance),
    )
    hardest_shuffle_gap = hardest_gap.sum() / has_valid_negative.float().sum().clamp_min(1.0)

    if negative_action_distance is not None:
        negative_action_distance = negative_action_distance.to(
            device=negative_distance.device,
            dtype=torch.float32,
        )
        if negative_action_distance.shape != negative_distance.shape:
            raise ValueError(
                "negative_action_distance must have shape [B, K]."
            )
        selected_action_distance = negative_action_distance.gather(1, hard_indices)
        negative_action_distance_mean = (
            selected_action_distance * hard_valid_float
        ).sum() / selected_pair_count
    else:
        negative_action_distance_mean = rank_per_pair.sum() * 0.0

    return rank_loss, {
        "positive_distance": positive_mean,
        "negative_distance": negative_mean,
        "positive_direction_distance": positive_direction_mean,
        "negative_direction_distance": negative_direction_mean,
        "positive_magnitude_distance": positive_magnitude_mean,
        "negative_magnitude_distance": negative_magnitude_mean,
        "shuffle_gap": shuffle_gap,
        "hardest_shuffle_gap": hardest_shuffle_gap,
        "rank_accuracy": rank_accuracy,
        "valid_negatives_per_sample": negative_mask.float().sum(dim=1).mean(),
        "negative_action_distance": negative_action_distance_mean,
        "target_norm": target_norm.mean(),
        "target_rms": target_rms.mean(),
        "valid_positive_ratio": valid_positive.float().mean(),
    }


class ActionEncoder(nn.Module):
    """Deterministic action-chunk encoder used by the VITA-style latent flow."""

    def __init__(self, action_dim, horizon, latent_dim=512, hidden_dim=512, num_layers=4, dropout=0.0):
        super().__init__()
        self.input_proj = nn.Linear(action_dim * horizon, hidden_dim)
        self.blocks = nn.ModuleList(
            [_mlp(hidden_dim, hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, actions):
        x = self.input_proj(actions.flatten(1))
        for block in self.blocks:
            x = x + block(x)
        return self.output_proj(self.norm(x))


class TemporalCNNActionEncoder(nn.Module):
    """Encode an action chunk while preserving its temporal ordering."""

    def __init__(
        self,
        action_dim,
        horizon,
        latent_dim=512,
        hidden_dim=512,
        num_layers=4,
        kernel_size=5,
    ):
        super().__init__()
        if int(num_layers) <= 0:
            raise ValueError("Temporal CNN action encoder requires at least one layer.")
        if int(kernel_size) <= 0 or int(kernel_size) % 2 == 0:
            raise ValueError("Temporal CNN kernel size must be a positive odd number.")
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        layers = []
        in_channels = self.action_dim
        padding = int(kernel_size) // 2
        for _ in range(int(num_layers)):
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        int(hidden_dim),
                        kernel_size=int(kernel_size),
                        stride=2,
                        padding=padding,
                    ),
                    nn.GELU(),
                ]
            )
            in_channels = int(hidden_dim)
        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.output_proj = nn.Linear(int(hidden_dim), int(latent_dim))

    def forward(self, actions):
        if actions.ndim != 3 or actions.shape[1:] != (
            self.horizon,
            self.action_dim,
        ):
            raise ValueError(
                f"Expected actions [B,{self.horizon},{self.action_dim}], "
                f"got {tuple(actions.shape)}."
            )
        x = self.encoder(actions.transpose(1, 2))
        x = self.pool(x).squeeze(-1)
        return self.output_proj(self.norm(x))


class ActionDecoder(nn.Module):
    """Decode one action latent into a complete continuous action chunk."""

    def __init__(self, action_dim, horizon, latent_dim=512, hidden_dim=512, num_layers=4, dropout=0.0):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [_mlp(hidden_dim, hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, action_dim * horizon)

    def forward(self, latent):
        x = self.input_proj(latent)
        for block in self.blocks:
            x = x + block(x)
        actions = self.output_proj(self.norm(x))
        return actions.view(latent.shape[0], self.horizon, self.action_dim)


class VisualTokenResampler(nn.Module):
    """Compress layer-0 visual tokens into a fixed-size action-effect context."""

    def __init__(self, dim, num_queries=4, num_heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, int(num_queries), dim))
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, int(num_heads), batch_first=True)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, visual_tokens):
        context = self.norm(visual_tokens)
        query = self.query.expand(context.shape[0], -1, -1).to(dtype=context.dtype)
        pooled, _ = self.attn(query, context, context, need_weights=False)
        return pooled


def spatial_pool_qwen_visual_tokens(
    visual_tokens,
    grid_thw,
    spatial_merge_size,
    output_grid_size=8,
):
    """Pool one Qwen image token grid into a fixed row-major spatial grid."""
    if visual_tokens.ndim != 2:
        raise ValueError(
            f"Expected one image's visual tokens [N,D], got {tuple(visual_tokens.shape)}."
        )
    if grid_thw.numel() != 3:
        raise ValueError(f"Expected image_grid_thw [3], got {tuple(grid_thw.shape)}.")

    temporal, grid_height, grid_width = (
        int(value) for value in grid_thw.detach().cpu().tolist()
    )
    spatial_merge_size = int(spatial_merge_size)
    output_grid_size = int(output_grid_size)
    if spatial_merge_size <= 0 or output_grid_size <= 0:
        raise ValueError("Spatial merge and output grid sizes must be positive.")
    if (
        grid_height % spatial_merge_size != 0
        or grid_width % spatial_merge_size != 0
    ):
        raise ValueError(
            f"Qwen grid {(grid_height, grid_width)} is not divisible by "
            f"spatial_merge_size={spatial_merge_size}."
        )

    merged_height = grid_height // spatial_merge_size
    merged_width = grid_width // spatial_merge_size
    expected_tokens = temporal * merged_height * merged_width
    if visual_tokens.shape[0] != expected_tokens:
        raise ValueError(
            f"Qwen visual token count {visual_tokens.shape[0]} does not match "
            f"grid-derived count {expected_tokens}."
        )
    if merged_height < output_grid_size or merged_width < output_grid_size:
        raise ValueError(
            f"Qwen merged grid {(merged_height, merged_width)} is smaller than "
            f"the requested {output_grid_size}x{output_grid_size} teacher grid."
        )

    token_grid = visual_tokens.reshape(
        temporal,
        merged_height,
        merged_width,
        visual_tokens.shape[-1],
    ).mean(dim=0)
    original_dtype = token_grid.dtype
    token_grid = token_grid.permute(2, 0, 1).unsqueeze(0).float()
    pooled = F.adaptive_avg_pool2d(
        token_grid,
        (output_grid_size, output_grid_size),
    )
    return (
        pooled.squeeze(0)
        .permute(1, 2, 0)
        .reshape(output_grid_size * output_grid_size, visual_tokens.shape[-1])
        .to(dtype=original_dtype)
    )


class StructuredMQ54(nn.Module):
    """Build 52 structured MQ tokens plus two externally generated state tokens."""

    HIDDEN_LAYERS = (0, 1, 4, 8, 12)
    SPATIAL_LAYERS = (0, 1, 4, 8)
    SEMANTIC_LAYERS = (8, 12)
    TEXT_LAYERS = (8, 12)

    def __init__(self, hidden_dim=1024, num_heads=8):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.spatial_norms = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim) for _ in self.SPATIAL_LAYERS]
        )
        self.spatial_layer_logits = nn.Parameter(torch.zeros(4))
        self.semantic_norms = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim) for _ in self.SEMANTIC_LAYERS]
        )
        self.semantic_layer_logits = nn.Parameter(torch.zeros(2))
        self.text_norms = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim) for _ in self.TEXT_LAYERS]
        )
        self.text_layer_logits = nn.Parameter(torch.zeros(2))
        self.text_to_semantic_query = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.semantic_slot_embedding = nn.Parameter(
            torch.zeros(6, self.hidden_dim)
        )
        nn.init.normal_(self.semantic_slot_embedding, std=0.02)
        self.semantic_attention = nn.MultiheadAttention(
            self.hidden_dim, int(num_heads), batch_first=True
        )
        self.global_norm = nn.LayerNorm(self.hidden_dim)
        self.global_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.semantic_output_norm = nn.LayerNorm(self.hidden_dim)
        self.last_semantic_kv_lengths = ()
        self.register_buffer(
            "source_ids",
            torch.tensor([0] * 32 + [1] * 12 + [2] * 8, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "camera_ids",
            torch.tensor(
                [0] * 16 + [1] * 16 + [0] * 6 + [1] * 6 + [2] * 8,
                dtype=torch.long,
            ),
            persistent=False,
        )

    @staticmethod
    def _normalize_grids(image_grid_thw, batch_size):
        if image_grid_thw is None:
            raise ValueError("StructuredMQ54 requires image_grid_thw metadata.")
        grids = image_grid_thw
        if grids.ndim == 2:
            if grids.shape[0] != batch_size * 2 or grids.shape[1] != 3:
                raise ValueError("StructuredMQ54 requires exactly two image grids per sample.")
            grids = grids.reshape(batch_size, 2, 3)
        if grids.ndim != 3 or grids.shape != (batch_size, 2, 3):
            raise ValueError("image_grid_thw must have shape [B*2,3] or [B,2,3].")
        return grids

    @staticmethod
    def _grid_shape(grid, spatial_merge_size):
        temporal, height, width = [int(v) for v in grid.detach().cpu().tolist()]
        merge = int(spatial_merge_size)
        if temporal != 1 or height % merge or width % merge:
            raise ValueError("StructuredMQ54 requires divisible single-frame image grids.")
        return height // merge, width // merge

    @staticmethod
    def _token_positions(token_type_ids, token_type):
        if token_type == "visual":
            return [torch.where((row == 1) | (row == 2))[0] for row in token_type_ids]
        return [torch.where(row == 0)[0] for row in token_type_ids]

    def _view_tokens(
        self, hidden, visual_positions, grids, view_index, spatial_merge_size
    ):
        outputs = []
        for batch_index, positions in enumerate(visual_positions):
            counts = [
                math.prod(self._grid_shape(grids[batch_index, camera], spatial_merge_size))
                for camera in range(2)
            ]
            if positions.numel() != sum(counts):
                raise ValueError(
                    "Visual token mask count does not match the two image grids: "
                    f"{positions.numel()} != {sum(counts)}."
                )
            offset = sum(counts[:view_index])
            selected = positions[offset : offset + counts[view_index]]
            outputs.append(hidden[batch_index].index_select(0, selected))
        return outputs

    @staticmethod
    def _pad_sequences(sequences):
        max_len = max(sequence.shape[0] for sequence in sequences)
        padded = sequences[0].new_zeros(
            len(sequences), max_len, sequences[0].shape[-1]
        )
        mask = torch.ones(
            len(sequences), max_len, device=padded.device, dtype=torch.bool
        )
        for index, sequence in enumerate(sequences):
            padded[index, : sequence.shape[0]] = sequence
            mask[index, : sequence.shape[0]] = False
        return padded, mask

    def _spatial_tokens(self, hidden_states, visual_positions, grids, merge, view):
        weights = self.spatial_layer_logits.softmax(dim=0)
        layer_views = [
            self._view_tokens(hidden_states[layer], visual_positions, grids, view, merge)
            for layer in self.SPATIAL_LAYERS
        ]
        outputs = []
        for batch_index in range(len(visual_positions)):
            fused = sum(
                weights[layer_pos]
                * self.spatial_norms[layer_pos](layer_views[layer_pos][batch_index])
                for layer_pos in range(4)
            )
            height, width = self._grid_shape(grids[batch_index, view], merge)
            grid = fused.reshape(height, width, self.hidden_dim).permute(2, 0, 1)
            pooled = F.adaptive_avg_pool2d(grid.float(), (4, 4)).to(fused.dtype)
            outputs.append(pooled.permute(1, 2, 0).reshape(16, self.hidden_dim))
        return torch.stack(outputs)

    def _semantic_tokens(
        self, hidden_states, visual_positions, text_positions, grids, merge, view
    ):
        weights = self.semantic_layer_logits.softmax(dim=0)
        layer_views = [
            self._view_tokens(hidden_states[layer], visual_positions, grids, view, merge)
            for layer in self.SEMANTIC_LAYERS
        ]
        visual_sequences = [
            sum(
                weights[layer_pos]
                * self.semantic_norms[layer_pos](layer_views[layer_pos][batch_index])
                for layer_pos in range(2)
            )
            for batch_index in range(len(visual_positions))
        ]
        visual, padding_mask = self._pad_sequences(visual_sequences)
        text_anchor = torch.stack(
            [
                hidden_states[12][batch_index].index_select(0, positions).mean(dim=0)
                for batch_index, positions in enumerate(text_positions)
            ]
        )
        queries = self.text_to_semantic_query(text_anchor).unsqueeze(1)
        queries = queries + self.semantic_slot_embedding.unsqueeze(0)
        attended, _ = self.semantic_attention(
            queries, visual, visual, key_padding_mask=padding_mask, need_weights=False
        )
        view_global = torch.stack(
            [tokens.mean(dim=0) for tokens in layer_views[1]]
        )
        view_global = self.global_proj(self.global_norm(view_global)).unsqueeze(1)
        self.last_semantic_kv_lengths = tuple(
            sequence.shape[0] for sequence in visual_sequences
        )
        return self.semantic_output_norm(queries + attended + view_global)

    def _text_tokens(self, hidden_states, text_positions):
        weights = self.text_layer_logits.softmax(dim=0)
        outputs = []
        for batch_index, positions in enumerate(text_positions):
            if positions.numel() == 0:
                raise ValueError("StructuredMQ54 requires at least one valid text token.")
            fused = sum(
                weights[layer_pos]
                * self.text_norms[layer_pos](
                    hidden_states[layer][batch_index].index_select(0, positions)
                )
                for layer_pos, layer in enumerate(self.TEXT_LAYERS)
            )
            pooled = F.adaptive_avg_pool1d(
                fused.transpose(0, 1).unsqueeze(0).float(), 8
            ).to(fused.dtype)
            outputs.append(pooled.squeeze(0).transpose(0, 1))
        return torch.stack(outputs)

    def forward(
        self,
        hidden_states,
        hidden_token_type_ids,
        image_grid_thw,
        spatial_merge_size=1,
        state_tokens=None,
    ):
        if not isinstance(hidden_states, (tuple, list)) or len(hidden_states) <= 12:
            raise ValueError("StructuredMQ54 requires hidden states H0 through H12.")
        if hidden_token_type_ids is None:
            raise ValueError("StructuredMQ54 requires visual/text token type IDs.")
        batch_size = hidden_states[0].shape[0]
        grids = self._normalize_grids(image_grid_thw, batch_size)
        visual_positions = self._token_positions(hidden_token_type_ids, "visual")
        text_positions = self._token_positions(hidden_token_type_ids, "text")
        spatial = [
            self._spatial_tokens(
                hidden_states, visual_positions, grids, spatial_merge_size, view
            )
            for view in range(2)
        ]
        semantic = [
            self._semantic_tokens(
                hidden_states,
                visual_positions,
                text_positions,
                grids,
                spatial_merge_size,
                view,
            )
            for view in range(2)
        ]
        text = self._text_tokens(hidden_states, text_positions)
        mq_tokens = torch.cat([spatial[0], spatial[1], semantic[0], semantic[1], text], dim=1)
        if mq_tokens.shape[1:] != (52, self.hidden_dim):
            raise RuntimeError("StructuredMQ54 must produce [B,52,D] MQ tokens.")
        if state_tokens is None or state_tokens.shape != (batch_size, 2, self.hidden_dim):
            raise ValueError("StructuredMQ54 requires state_tokens shaped [B,2,D].")
        return torch.cat([mq_tokens, state_tokens], dim=1)


def motion_weighted_dense_huber_loss(
    prediction,
    target,
    current,
    horizon_weights=None,
    motion_weight_floor=0.25,
    eps=1e-6,
):
    """Dense future-token loss with larger weights on moving spatial regions."""
    if prediction.ndim != 4 or target.ndim != 4:
        raise ValueError("Prediction and target must have shape [B,H,N,D].")
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target must share shape.")
    if current.ndim != 3 or current.shape != target[:, 0].shape:
        raise ValueError("Current visual tokens must have shape [B,N,D].")
    motion_weight_floor = float(motion_weight_floor)
    if not 0.0 <= motion_weight_floor <= 1.0:
        raise ValueError("motion_weight_floor must be in [0, 1].")
    if float(eps) <= 0:
        raise ValueError("eps must be positive.")

    target = target.detach()
    motion = (
        target.float() - current.detach().float().unsqueeze(1)
    ).norm(dim=-1)
    mean_motion = motion.mean(dim=-1, keepdim=True)
    relative_motion = motion / mean_motion.clamp_min(float(eps))
    patch_weights = (
        motion_weight_floor
        + (1.0 - motion_weight_floor) * relative_motion
    )
    no_motion = mean_motion <= float(eps)
    patch_weights = torch.where(
        no_motion,
        torch.ones_like(patch_weights),
        patch_weights,
    )
    patch_weights = patch_weights / patch_weights.mean(
        dim=-1, keepdim=True
    ).clamp_min(float(eps))

    patch_loss = F.smooth_l1_loss(
        prediction.float(),
        target.float(),
        reduction="none",
    ).mean(dim=-1)
    per_horizon = (patch_loss * patch_weights).mean(dim=(0, 2))

    if horizon_weights is None:
        alpha = torch.ones_like(per_horizon)
    else:
        alpha = torch.as_tensor(
            horizon_weights,
            device=per_horizon.device,
            dtype=per_horizon.dtype,
        )
        if alpha.ndim != 1 or alpha.numel() != per_horizon.numel():
            raise ValueError("horizon_weights must match the number of horizons.")
        if torch.any(alpha < 0) or not bool(torch.any(alpha > 0)):
            raise ValueError("horizon_weights must be non-negative with a positive sum.")

    loss = (per_horizon * alpha).sum() / alpha.sum().clamp_min(float(eps))
    diagnostics = {
        "per_horizon": per_horizon.detach(),
        "mean_motion": mean_motion.mean(dim=(0, 2)).detach(),
        "max_patch_weight": patch_weights.amax(dim=(0, 2)).detach(),
    }
    return loss, diagnostics


class CausalActionTokenEncoder(nn.Module):
    """Encode one token per action step without exposing later actions."""

    def __init__(
        self,
        action_dim,
        horizon,
        token_dim=256,
        num_layers=2,
        num_heads=8,
        dropout=0.0,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.input_proj = nn.Linear(action_dim, token_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.horizon, token_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(token_dim)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, actions):
        if actions.ndim != 3:
            raise ValueError("Actions must have shape [B,T,A].")
        if actions.shape[1] > self.horizon:
            raise ValueError(
                f"Received {actions.shape[1]} actions for horizon {self.horizon}."
            )
        actions = actions.to(dtype=self.input_proj.weight.dtype)
        x = self.input_proj(actions)
        x = x + self.pos_embed[:, : x.shape[1]].to(dtype=x.dtype)
        causal_mask = torch.ones(
            x.shape[1],
            x.shape[1],
            device=x.device,
            dtype=torch.bool,
        ).triu(diagonal=1)
        return self.norm(self.blocks(x, mask=causal_mask))


class MultiHorizonVisualPredictor(nn.Module):
    """Predict spatial future features from strictly horizon-limited actions."""

    def __init__(
        self,
        action_dim,
        action_horizon,
        action_latent_dim,
        visual_dim,
        visual_tokens,
        horizons,
        token_dim=256,
        num_layers=2,
        num_heads=8,
        dropout=0.0,
    ):
        super().__init__()
        self.horizons = tuple(int(value) for value in horizons)
        if not self.horizons:
            raise ValueError("At least one action-effect horizon is required.")
        if any(value <= 0 or value > int(action_horizon) for value in self.horizons):
            raise ValueError("Action-effect horizons must lie within the action chunk.")
        if tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("Action-effect horizons must be unique and sorted.")

        self.action_tokens = CausalActionTokenEncoder(
            action_dim=action_dim,
            horizon=action_horizon,
            token_dim=token_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.visual_norm = nn.LayerNorm(visual_dim)
        self.visual_proj = nn.Linear(visual_dim, token_dim)
        self.prefix_latent_norm = nn.LayerNorm(action_latent_dim)
        self.prefix_latent_proj = nn.Linear(action_latent_dim, token_dim)
        self.state_norm = nn.LayerNorm(action_dim)
        self.state_proj = nn.Linear(action_dim, token_dim)
        self.horizon_embed = nn.Parameter(
            torch.zeros(len(self.horizons), token_dim)
        )
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, int(visual_tokens), token_dim)
        )
        self.cross_norms = nn.ModuleList(
            [nn.LayerNorm(token_dim) for _ in range(num_layers)]
        )
        self.cross_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    token_dim,
                    num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.ffn_norms = nn.ModuleList(
            [nn.LayerNorm(token_dim) for _ in range(num_layers)]
        )
        self.ffns = nn.ModuleList(
            [_mlp(token_dim, token_dim * 4, dropout) for _ in range(num_layers)]
        )
        self.output_norm = nn.LayerNorm(token_dim)
        self.output_proj = nn.Linear(token_dim, visual_dim)
        nn.init.normal_(self.horizon_embed, std=0.02)
        nn.init.normal_(self.spatial_pos_embed, std=0.02)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        actions,
        prefix_latents,
        current_visual_tokens,
        proprioception,
    ):
        if prefix_latents.ndim != 3:
            raise ValueError("Prefix latents must have shape [B,H,D].")
        if prefix_latents.shape[1] != len(self.horizons):
            raise ValueError("Prefix latent count must match configured horizons.")
        if current_visual_tokens.ndim != 3:
            raise ValueError("Current visual tokens must have shape [B,N,D].")
        if current_visual_tokens.shape[1] != self.spatial_pos_embed.shape[1]:
            raise ValueError("Current visual token count does not match the predictor.")
        if proprioception is None:
            raise ValueError("Multi-horizon prediction requires robot state.")

        action_tokens = self.action_tokens(actions)
        if proprioception.ndim == 3:
            current_state = proprioception[:, -1]
        elif proprioception.ndim == 2:
            current_state = proprioception
        else:
            raise ValueError("Robot state must have shape [B,T,A] or [B,A].")
        current_state = current_state.to(dtype=self.state_proj.weight.dtype)
        state_token = self.state_proj(self.state_norm(current_state)).unsqueeze(1)

        visual = self.visual_proj(
            self.visual_norm(current_visual_tokens).to(
                dtype=self.visual_proj.weight.dtype
            )
        )
        spatial_position = self.spatial_pos_embed.to(dtype=visual.dtype)
        predictions = []
        for index, horizon in enumerate(self.horizons):
            prefix_token = self.prefix_latent_proj(
                self.prefix_latent_norm(prefix_latents[:, index]).to(
                    dtype=self.prefix_latent_proj.weight.dtype
                )
            ).unsqueeze(1)
            memory = torch.cat(
                [state_token, action_tokens[:, :horizon], prefix_token],
                dim=1,
            )
            x = (
                visual
                + spatial_position
                + self.horizon_embed[index].to(dtype=visual.dtype).view(1, 1, -1)
            )
            for cross_norm, cross_attn, ffn_norm, ffn in zip(
                self.cross_norms,
                self.cross_attn,
                self.ffn_norms,
                self.ffns,
            ):
                query = cross_norm(x)
                delta, _ = cross_attn(
                    query,
                    memory,
                    memory,
                    need_weights=False,
                )
                x = x + delta
                x = x + ffn(ffn_norm(x))
            visual_delta = self.output_proj(self.output_norm(x))
            predictions.append(current_visual_tokens + visual_delta)
        return torch.stack(predictions, dim=1)


class ActionEffectStateEncoder(nn.Module):
    """Encode robot-state history for the auxiliary action-effect branch."""

    def __init__(self, proprio_dim, latent_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.proprio_dim = int(proprio_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.proprio_dim),
            nn.Linear(self.proprio_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, proprioception):
        state = proprioception.flatten(1)
        if state.shape[1] != self.proprio_dim:
            raise ValueError(
                f"Expected flattened action-effect proprioception dimension "
                f"{self.proprio_dim}, got {state.shape[1]}."
            )
        return self.net(state)


class VisualConditionedActionEncoder(nn.Module):
    """Preserve the action AE latent and add a visual-conditioned action residual."""

    def __init__(
        self,
        base_encoder,
        action_dim,
        horizon,
        latent_dim,
        hidden_dim,
        num_layers=2,
        num_heads=8,
        dropout=0.0,
        visual_dim=None,
    ):
        super().__init__()
        self.base_encoder = base_encoder
        visual_dim = int(visual_dim or latent_dim)
        self.visual_proj = (
            nn.Identity()
            if visual_dim == latent_dim
            else nn.Linear(visual_dim, latent_dim)
        )
        self.action_proj = nn.Linear(action_dim, latent_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, horizon, latent_dim))
        self.cross_norms = nn.ModuleList([nn.LayerNorm(latent_dim) for _ in range(num_layers)])
        self.cross_attn = nn.ModuleList(
            [nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True) for _ in range(num_layers)]
        )
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(latent_dim) for _ in range(num_layers)])
        self.ffns = nn.ModuleList(
            [_mlp(latent_dim, hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.output_norm = nn.LayerNorm(latent_dim)
        self.output_proj = nn.Linear(latent_dim, latent_dim)
        self.visual_residual_gate = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, actions, visual_tokens):
        base_latent = self.base_encoder(actions)
        visual_tokens = self.visual_proj(visual_tokens)
        x = self.action_proj(actions) + self.pos_embed[:, : actions.shape[1]].to(dtype=actions.dtype)
        for cross_norm, cross_attn, ffn_norm, ffn in zip(
            self.cross_norms, self.cross_attn, self.ffn_norms, self.ffns
        ):
            query = cross_norm(x)
            delta, _ = cross_attn(query, visual_tokens, visual_tokens, need_weights=False)
            x = x + delta
            x = x + ffn(ffn_norm(x))
        visual_delta = self.output_proj(self.output_norm(x).mean(dim=1))
        return base_latent + torch.tanh(self.visual_residual_gate) * visual_delta


class VisualDeltaDecoder(nn.Module):
    """Predict future-minus-current resampled visual tokens from an action latent."""

    def __init__(
        self,
        latent_dim,
        hidden_dim,
        num_layers=2,
        dropout=0.0,
        effect_dim=None,
        state_dim=None,
        visual_dim=None,
    ):
        super().__init__()
        effect_dim = int(effect_dim or latent_dim)
        visual_dim = int(visual_dim or latent_dim)
        self.visual_dim = visual_dim
        self.latent_proj = nn.Linear(effect_dim, visual_dim)
        self.state_norm = nn.LayerNorm(state_dim) if state_dim is not None else None
        self.state_proj = (
            nn.Linear(state_dim, visual_dim) if state_dim is not None else None
        )
        self.blocks = nn.ModuleList(
            [_mlp(visual_dim, hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(visual_dim)
        self.output_proj = nn.Linear(visual_dim, visual_dim)

    def forward(self, latent, current_visual_tokens, state_embedding=None):
        x = current_visual_tokens + self.latent_proj(latent).unsqueeze(1)
        if self.state_proj is not None:
            if state_embedding is None:
                raise ValueError("State-conditioned visual decoding requires state_embedding.")
            state_delta = self.state_proj(self.state_norm(state_embedding)).unsqueeze(1)
            x = x + state_delta
        for block in self.blocks:
            x = x + block(x)
        return self.output_proj(self.norm(x))


class ActionEffectProjector(nn.Module):
    """Select an action-latent subspace for visual-effect prediction."""

    def __init__(self, latent_dim, effect_dim, state_dim=None):
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim)
        self.state_norm = nn.LayerNorm(state_dim) if state_dim is not None else None
        input_dim = latent_dim + (int(state_dim) if state_dim is not None else 0)
        self.proj = nn.Linear(input_dim, effect_dim)

    def forward(self, action_latent, state_embedding=None):
        features = self.norm(action_latent)
        if self.state_norm is not None:
            if state_embedding is None:
                raise ValueError("State-conditioned action-effect projection requires state_embedding.")
            features = torch.cat(
                [features, self.state_norm(state_embedding)],
                dim=-1,
            )
        return self.proj(features)


class SlotwiseDepthRouter(nn.Module):
    """Mix aligned reserve MQ slots across a fixed set of VLM depths."""

    def __init__(
        self,
        hidden_dim,
        router_dim=256,
        num_slots=4,
        num_sources=5,
        temperature=1.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.router_dim = int(router_dim)
        self.num_slots = int(num_slots)
        self.num_sources = int(num_sources)
        self.temperature = float(temperature)
        if self.router_dim < 1:
            raise ValueError("adaptive MQ router_dim must be positive.")
        if self.num_slots < 1:
            raise ValueError("adaptive MQ num_slots must be positive.")
        if self.num_sources < 2:
            raise ValueError("adaptive MQ routing requires at least two depth sources.")
        if self.temperature <= 0:
            raise ValueError("adaptive MQ temperature must be positive.")

        self.router_norm = nn.LayerNorm(self.hidden_dim)
        self.source_embedding = nn.Parameter(
            torch.zeros(self.num_sources, self.hidden_dim)
        )
        self.key_proj = nn.Linear(self.hidden_dim, self.router_dim, bias=False)
        self.selector_queries = nn.Parameter(
            torch.empty(self.num_slots, self.router_dim)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.source_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.selector_queries, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.key_proj.weight)

    def forward(self, reserve_tokens):
        if reserve_tokens.ndim != 4:
            raise ValueError(
                "reserve_tokens must have shape [B, num_slots, num_sources, hidden_dim]."
            )
        _, num_slots, num_sources, hidden_dim = reserve_tokens.shape
        if num_slots != self.num_slots:
            raise ValueError(
                f"Expected {self.num_slots} adaptive MQ slots, got {num_slots}."
            )
        if num_sources != self.num_sources:
            raise ValueError(
                f"Expected {self.num_sources} adaptive MQ sources, got {num_sources}."
            )
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Expected adaptive MQ hidden dim {self.hidden_dim}, got {hidden_dim}."
            )

        features = self.router_norm(reserve_tokens)
        features = features + self.source_embedding[None, None, :, :]
        keys = self.key_proj(features)
        logits = torch.einsum(
            "bjsr,jr->bjs",
            keys,
            self.selector_queries,
        )
        logits = logits / math.sqrt(self.router_dim)
        route_weights = torch.softmax(logits / self.temperature, dim=-1)
        dynamic_tokens = torch.einsum(
            "bjs,bjsd->bjd",
            route_weights,
            reserve_tokens,
        )
        return dynamic_tokens, route_weights


def attention_concentration_score(attention_weights, key_padding_mask=None):
    """Score each MQ by normalized per-head attention concentration."""
    if attention_weights.ndim != 4:
        raise ValueError("attention_weights must have shape [B, H, Q, N].")

    probabilities = attention_weights.float()
    if key_padding_mask is not None:
        if key_padding_mask.shape != (
            probabilities.shape[0],
            probabilities.shape[-1],
        ):
            raise ValueError("key_padding_mask must have shape [B, N].")
        valid_mask = (~key_padding_mask).to(
            device=probabilities.device,
            dtype=probabilities.dtype,
        )
        probabilities = probabilities * valid_mask[:, None, None, :]
        probabilities = probabilities / probabilities.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-8)
        num_valid = valid_mask.sum(dim=-1).clamp_min(2.0)
    else:
        num_valid = probabilities.new_full(
            (probabilities.shape[0],),
            max(probabilities.shape[-1], 2),
        )

    safe_probabilities = probabilities.clamp_min(1e-8)
    entropy = -(probabilities * safe_probabilities.log()).sum(dim=-1)
    max_entropy = num_valid.log()[:, None, None].clamp_min(1e-8)
    concentration = (1.0 - entropy / max_entropy).clamp(0.0, 1.0)
    return concentration.mean(dim=1).detach()


def select_dynamic_mq(
    candidate_tokens,
    candidate_scores,
    min_keep_per_group=2,
    total_keep=24,
    random_exploration=False,
):
    """Select fixed-size MQ tokens while retaining minimum source coverage."""
    if candidate_tokens.ndim != 4:
        raise ValueError("candidate_tokens must have shape [B, G, Q, D].")
    if candidate_scores.shape != candidate_tokens.shape[:3]:
        raise ValueError("candidate_scores must have shape [B, G, Q].")

    batch_size, num_groups, candidates_per_group, hidden_dim = (
        candidate_tokens.shape
    )
    min_keep_per_group = int(min_keep_per_group)
    total_keep = int(total_keep)
    guaranteed_keep = num_groups * min_keep_per_group
    total_candidates = num_groups * candidates_per_group
    if min_keep_per_group < 1 or min_keep_per_group > candidates_per_group:
        raise ValueError("min_keep_per_group must be in [1, Q].")
    if total_keep < guaranteed_keep or total_keep > total_candidates:
        raise ValueError(
            "total_keep must be between guaranteed source coverage and "
            "the total candidate count."
        )
    if random_exploration and min_keep_per_group != 2:
        raise ValueError(
            "Random MQ exploration currently requires min_keep_per_group=2."
        )

    selected_mask = torch.zeros(
        batch_size,
        num_groups,
        candidates_per_group,
        dtype=torch.bool,
        device=candidate_tokens.device,
    )
    random_selected_mask = torch.zeros_like(selected_mask)
    if random_exploration:
        top1_indices = candidate_scores.argmax(
            dim=-1,
            keepdim=True,
        )
        selected_mask.scatter_(dim=-1, index=top1_indices, value=True)
        random_scores = torch.rand_like(candidate_scores).masked_fill(
            selected_mask,
            float("-inf"),
        )
        random_indices = random_scores.argmax(
            dim=-1,
            keepdim=True,
        )
        selected_mask.scatter_(dim=-1, index=random_indices, value=True)
        random_selected_mask.scatter_(
            dim=-1,
            index=random_indices,
            value=True,
        )
    else:
        guaranteed_indices = torch.topk(
            candidate_scores,
            k=min_keep_per_group,
            dim=-1,
        ).indices
        selected_mask.scatter_(
            dim=-1,
            index=guaranteed_indices,
            value=True,
        )

    extra_keep = total_keep - guaranteed_keep
    flat_mask = selected_mask.flatten(1)
    if extra_keep:
        remaining_scores = candidate_scores.flatten(1).masked_fill(
            flat_mask,
            float("-inf"),
        )
        extra_indices = torch.topk(
            remaining_scores,
            k=extra_keep,
            dim=-1,
        ).indices
        flat_mask.scatter_(dim=-1, index=extra_indices, value=True)

    flat_order = torch.arange(
        total_candidates,
        device=candidate_tokens.device,
    ).unsqueeze(0).expand(batch_size, -1)
    selected_indices = flat_order.masked_fill(
        ~flat_mask,
        total_candidates,
    ).sort(dim=-1).values[:, :total_keep]
    flat_tokens = candidate_tokens.reshape(
        batch_size,
        total_candidates,
        hidden_dim,
    )
    selected_tokens = flat_tokens.gather(
        dim=1,
        index=selected_indices.unsqueeze(-1).expand(-1, -1, hidden_dim),
    )
    selected_counts = selected_mask.sum(dim=-1)
    score_selected_mask = selected_mask & ~random_selected_mask
    return (
        selected_tokens,
        selected_counts,
        selected_mask,
        selected_indices,
        score_selected_mask,
        random_selected_mask,
    )


class ObservationLatentEncoder(nn.Module):
    """Project the VLM summary tokens into the action latent space."""

    def __init__(
        self,
        vlm_hidden_dim,
        latent_dim=512,
        hidden_dim=512,
        num_queries=16,
        proprio_dim=None,
        condition_source="auto",
        hidden_layer_index=-1,
        secondary_hidden_layer_index=None,
        extra_hidden_layer_indices=None,
        hidden_pooling="mean",
        pooling_num_heads=8,
        pooling_num_queries=1,
        gated_weighted_pooling=False,
        obs_pool_token_indices=None,
        hierarchical_query_pooling=False,
        layer_local_queries_per_layer=2,
        layer_local_queries_per_layer_list=None,
        layer_local_token_source_modes=None,
        hier_mq_separate_views=False,
        cross_layer_queries=18,
        global_queries=8,
        adaptive_local_mq=False,
        adaptive_mq_router_dim=256,
        adaptive_mq_num_slots=4,
        adaptive_mq_reserve_source_positions=None,
        adaptive_mq_temperature=1.0,
        adaptive_mq_route_warmup_steps=0,
        dynamic_topk_local_mq=False,
        dynamic_topk_candidate_source_positions=None,
        dynamic_topk_text_source_position=1,
        dynamic_topk_candidates_per_source=8,
        dynamic_topk_min_keep_per_source=2,
        dynamic_topk_total_keep=24,
        dynamic_topk_train_random_exploration=False,
        condition_type_embeddings=False,
        dynamic_extra_layer_gates=False,
        extra_layer_gate_scales=None,
        extra_layer_gate_hidden_dim=256,
        extra_layer_gate_dropout=0.0,
        blockwise_layer_conditioning=False,
        blockwise_hidden_layer_indices=None,
        state_token_conditioning=False,
        state_num_tokens=4,
        state_token_dropout=0.0,
        state_broadcast_to_mq=True,
        mq_local_overlap_loss_weight=0.0,
        mq_cross_overlap_loss_weight=0.0,
        mq_global_overlap_loss_weight=0.0,
        mq_competitive_local_attention=False,
        mq_competitive_attention_tau=1.0,
        mq_competitive_attention_gamma=1.0,
        mq_local_balance_loss_weight=0.0,
        hierarchical_global_residual_block=False,
        hierarchical_global_residual_scale=0.1,
        hierarchical_global_ffn_ratio=4.0,
        layer_aligned_query_pooling=False,
        layer_aligned_hidden_layer_indices=None,
        layer_aligned_queries_per_layer=4,
        layer_aligned_token_source_modes=None,
        layer_aligned_global_conditioning=False,
        layer_aligned_obs_pooling="last",
        layer_aligned_encoder_obs_conditioning=True,
        layer_aligned_flow_windows=None,
        mq_type="hier_mq54",
    ):
        super().__init__()
        self.num_queries = int(num_queries)
        self.mq_type = str(mq_type).lower()
        if self.mq_type not in {"hier_mq54", "structured_mq54"}:
            raise ValueError("mq_type must be 'hier_mq54' or 'structured_mq54'.")
        self.condition_source = str(condition_source)
        self.hidden_layer_index = int(hidden_layer_index)
        self.secondary_hidden_layer_index = (
            int(secondary_hidden_layer_index)
            if secondary_hidden_layer_index is not None
            else None
        )
        self.extra_hidden_layer_indices = [
            int(i) for i in (extra_hidden_layer_indices or [])
        ]
        if self.secondary_hidden_layer_index is not None:
            self.extra_hidden_layer_indices.insert(0, self.secondary_hidden_layer_index)
        self.hidden_pooling = str(hidden_pooling)
        self.pooling_num_queries = int(pooling_num_queries)
        self.gated_weighted_pooling = bool(gated_weighted_pooling)
        self.obs_pool_token_indices = (
            [int(i) for i in obs_pool_token_indices]
            if obs_pool_token_indices is not None
            else []
        )
        self.hierarchical_query_pooling = bool(hierarchical_query_pooling)
        self.layer_local_queries_per_layer = int(layer_local_queries_per_layer)
        self.layer_local_queries_per_layer_list = (
            [int(v) for v in layer_local_queries_per_layer_list]
            if layer_local_queries_per_layer_list is not None
            else None
        )
        self.layer_local_token_source_modes = (
            [str(v).lower() for v in layer_local_token_source_modes]
            if layer_local_token_source_modes is not None
            else None
        )
        self.hier_mq_separate_views = bool(hier_mq_separate_views)
        self.cross_layer_queries = int(cross_layer_queries)
        self.global_queries = int(global_queries)
        self.adaptive_local_mq = bool(adaptive_local_mq)
        self.adaptive_mq_router_dim = int(adaptive_mq_router_dim)
        self.adaptive_mq_num_slots = int(adaptive_mq_num_slots)
        self.adaptive_mq_reserve_source_positions = [
            int(position)
            for position in (adaptive_mq_reserve_source_positions or [])
        ]
        self.adaptive_mq_temperature = float(adaptive_mq_temperature)
        self.adaptive_mq_route_warmup_steps = int(
            adaptive_mq_route_warmup_steps
        )
        self._adaptive_mq_step = None
        self.dynamic_topk_local_mq = bool(dynamic_topk_local_mq)
        self.dynamic_topk_candidate_source_positions = [
            int(position)
            for position in (dynamic_topk_candidate_source_positions or [])
        ]
        self.dynamic_topk_text_source_position = int(
            dynamic_topk_text_source_position
        )
        self.dynamic_topk_candidates_per_source = int(
            dynamic_topk_candidates_per_source
        )
        self.dynamic_topk_min_keep_per_source = int(
            dynamic_topk_min_keep_per_source
        )
        self.dynamic_topk_total_keep = int(dynamic_topk_total_keep)
        self.dynamic_topk_train_random_exploration = bool(
            dynamic_topk_train_random_exploration
        )
        self.condition_type_embeddings = bool(condition_type_embeddings)
        self.dynamic_extra_layer_gates = bool(dynamic_extra_layer_gates)
        self.extra_layer_gate_dropout = nn.Dropout(float(extra_layer_gate_dropout))
        self.blockwise_layer_conditioning = bool(blockwise_layer_conditioning)
        self.blockwise_hidden_layer_indices = [
            int(i) for i in (blockwise_hidden_layer_indices or [])
        ]
        self.state_token_conditioning = bool(state_token_conditioning)
        self.state_num_tokens = int(state_num_tokens)
        self.state_broadcast_to_mq = bool(state_broadcast_to_mq)
        self.mq_local_overlap_loss_weight = float(mq_local_overlap_loss_weight)
        self.mq_cross_overlap_loss_weight = float(mq_cross_overlap_loss_weight)
        self.mq_global_overlap_loss_weight = float(mq_global_overlap_loss_weight)
        self.use_mq_overlap_loss = (
            self.mq_local_overlap_loss_weight > 0
            or self.mq_cross_overlap_loss_weight > 0
            or self.mq_global_overlap_loss_weight > 0
        )
        self.mq_competitive_local_attention = bool(mq_competitive_local_attention)
        self.mq_competitive_attention_tau = float(mq_competitive_attention_tau)
        self.mq_competitive_attention_gamma = float(mq_competitive_attention_gamma)
        self.mq_local_balance_loss_weight = float(mq_local_balance_loss_weight)
        self.hierarchical_global_residual_block = bool(
            hierarchical_global_residual_block
        )
        self.hierarchical_global_residual_scale = float(
            hierarchical_global_residual_scale
        )
        self.hierarchical_global_ffn_ratio = float(
            hierarchical_global_ffn_ratio
        )
        self.layer_aligned_query_pooling = bool(layer_aligned_query_pooling)
        self.layer_aligned_hidden_layer_indices = [
            int(i) for i in (layer_aligned_hidden_layer_indices or [])
        ]
        self.layer_aligned_queries_per_layer = int(layer_aligned_queries_per_layer)
        self.layer_aligned_token_source_modes = (
            [str(v).lower() for v in layer_aligned_token_source_modes]
            if layer_aligned_token_source_modes is not None
            else None
        )
        self.layer_aligned_global_conditioning = bool(layer_aligned_global_conditioning)
        self.layer_aligned_obs_pooling = str(layer_aligned_obs_pooling).lower()
        self.layer_aligned_encoder_obs_conditioning = bool(layer_aligned_encoder_obs_conditioning)
        self.layer_aligned_flow_windows = (
            [[int(layer_pos) for layer_pos in window] for window in layer_aligned_flow_windows]
            if layer_aligned_flow_windows is not None
            else None
        )
        self.last_mq_overlap_loss = None
        self.last_mq_overlap_losses = {}
        self.last_mq_balance_loss = None
        self.last_mq_balance_losses = {}
        self.last_adaptive_mq_route_weights = None
        self.last_adaptive_mq_router_metrics = {}
        self.last_dynamic_topk_selected_mask = None
        self.last_dynamic_topk_selected_indices = None
        self.last_dynamic_topk_score_selected_mask = None
        self.last_dynamic_topk_random_selected_mask = None
        self.last_dynamic_topk_metrics = {}
        self.last_global_token_metrics = {}
        self.structured_mq = (
            StructuredMQ54(vlm_hidden_dim, pooling_num_heads)
            if self.mq_type == "structured_mq54"
            else None
        )
        if self.pooling_num_queries < 1:
            raise ValueError("pooling_num_queries must be at least 1.")
        if any(i < 0 for i in self.obs_pool_token_indices):
            raise ValueError("obs_pool_token_indices must be non-negative.")
        if self.condition_source not in {"auto", "connector", "hidden_layer"}:
            raise ValueError(
                "condition_source must be 'auto', 'connector', or 'hidden_layer'."
            )
        if self.hidden_pooling not in {"mean", "attention"}:
            raise ValueError("hidden_pooling must be 'mean' or 'attention'.")
        if self.hierarchical_query_pooling and self.hidden_pooling != "attention":
            raise ValueError("Hierarchical query pooling requires hidden_pooling='attention'.")
        if self.hierarchical_query_pooling and not self.blockwise_hidden_layer_indices:
            raise ValueError("Hierarchical query pooling requires blockwise_hidden_layer_indices.")
        if self.hier_mq_separate_views and not self.hierarchical_query_pooling:
            raise ValueError("Separate-view HierMQ requires hierarchical_query_pooling=True.")
        if self.hierarchical_global_residual_block and not self.hierarchical_query_pooling:
            raise ValueError(
                "The global residual block requires hierarchical_query_pooling=True."
            )
        if self.hierarchical_global_residual_scale < 0:
            raise ValueError("hierarchical_global_residual_scale must be non-negative.")
        if self.hierarchical_global_ffn_ratio <= 0:
            raise ValueError("hierarchical_global_ffn_ratio must be positive.")
        if self.layer_aligned_query_pooling and self.hidden_pooling != "attention":
            raise ValueError("Layer-aligned MQ pooling requires hidden_pooling='attention'.")
        if self.layer_aligned_query_pooling and not self.layer_aligned_hidden_layer_indices:
            raise ValueError("Layer-aligned MQ pooling requires layer_aligned_hidden_layer_indices.")
        if self.layer_aligned_query_pooling and self.layer_aligned_queries_per_layer < 1:
            raise ValueError("layer_aligned_queries_per_layer must be positive.")
        if self.layer_aligned_obs_pooling not in {"last", "attn_obs_global", "attn_local_global"}:
            raise ValueError(
                "layer_aligned_obs_pooling must be 'last', 'attn_obs_global', or 'attn_local_global'."
            )
        if (
            self.layer_aligned_query_pooling
            and not self.layer_aligned_encoder_obs_conditioning
            and self.layer_aligned_obs_pooling in {"last", "attn_obs_global"}
        ):
            raise ValueError(
                "Disabling encoder obs tokens requires layer_aligned_obs_pooling='attn_local_global'."
            )
        if self.layer_aligned_global_conditioning and self.global_queries < 1:
            raise ValueError("layer_aligned_global_conditioning requires global_queries >= 1.")
        num_aligned_layers = len(self.layer_aligned_hidden_layer_indices)
        if self.layer_aligned_flow_windows is not None:
            if not self.layer_aligned_flow_windows:
                raise ValueError("layer_aligned_flow_windows must not be empty.")
            window_sizes = {len(window) for window in self.layer_aligned_flow_windows}
            if 0 in window_sizes:
                raise ValueError("layer_aligned_flow_windows cannot contain empty windows.")
            if len(window_sizes) != 1:
                raise ValueError(
                    "All layer_aligned_flow_windows must have the same size so they can be stacked."
                )
            invalid_positions = [
                layer_pos
                for window in self.layer_aligned_flow_windows
                for layer_pos in window
                if layer_pos < 0 or layer_pos >= num_aligned_layers
            ]
            if invalid_positions:
                raise ValueError(
                    "layer_aligned_flow_windows contains invalid selected-layer positions: "
                    f"{invalid_positions}"
                )
        if self.layer_aligned_token_source_modes is not None:
            if len(self.layer_aligned_token_source_modes) != num_aligned_layers:
                raise ValueError(
                    "layer_aligned_token_source_modes must have the same length as "
                    "layer_aligned_hidden_layer_indices."
                )
        if self.layer_aligned_token_source_modes is None:
            self.layer_aligned_token_source_modes = ["all"] * num_aligned_layers
        num_hier_layers = len(self.blockwise_hidden_layer_indices)
        if self.layer_local_queries_per_layer_list is not None:
            if len(self.layer_local_queries_per_layer_list) != num_hier_layers:
                raise ValueError(
                    "layer_local_queries_per_layer_list must have the same length as "
                    "blockwise_hidden_layer_indices."
                )
            if any(v < 1 for v in self.layer_local_queries_per_layer_list):
                raise ValueError("layer_local_queries_per_layer_list values must be positive.")
        if self.layer_local_token_source_modes is not None:
            if len(self.layer_local_token_source_modes) != num_hier_layers:
                raise ValueError(
                    "layer_local_token_source_modes must have the same length as "
                    "blockwise_hidden_layer_indices."
                )
        self.layer_local_query_counts = (
            self.layer_local_queries_per_layer_list
            if self.layer_local_queries_per_layer_list is not None
            else [self.layer_local_queries_per_layer] * num_hier_layers
        )
        if self.layer_local_token_source_modes is None:
            self.layer_local_token_source_modes = ["all"] * num_hier_layers
        if self.hier_mq_separate_views:
            non_text_counts = [
                count
                for count, mode in zip(
                    self.layer_local_query_counts,
                    self.layer_local_token_source_modes,
                )
                if mode != "text"
            ]
            if any(count % 2 for count in non_text_counts):
                raise ValueError(
                    "Separate-view HierMQ requires an even query count for every "
                    "non-text local source."
                )
        if self.adaptive_local_mq:
            if not self.hierarchical_query_pooling:
                raise ValueError(
                    "Adaptive local MQ routing requires hierarchical_query_pooling=True."
                )
            if self.adaptive_mq_route_warmup_steps < 0:
                raise ValueError("adaptive MQ route_warmup_steps must be non-negative.")
            if not self.adaptive_mq_reserve_source_positions:
                raise ValueError(
                    "Adaptive local MQ routing requires reserve source positions."
                )
            if self.adaptive_mq_reserve_source_positions[0] != 0:
                raise ValueError(
                    "The first adaptive MQ reserve source must be H0 visual at position 0."
                )
            if len(set(self.adaptive_mq_reserve_source_positions)) != len(
                self.adaptive_mq_reserve_source_positions
            ):
                raise ValueError("Adaptive MQ reserve source positions must be unique.")
            invalid_positions = [
                position
                for position in self.adaptive_mq_reserve_source_positions
                if position < 0 or position >= num_hier_layers
            ]
            if invalid_positions:
                raise ValueError(
                    "Adaptive MQ reserve source positions are out of range: "
                    f"{invalid_positions}."
                )
            slots = self.adaptive_mq_num_slots
            if self.layer_local_query_counts[0] != 2 * slots:
                raise ValueError(
                    "Adaptive MQ expects the H0 visual group to contain exactly "
                    f"2 * num_slots={2 * slots} queries."
                )
            non_anchor_counts = self.layer_local_query_counts[1:]
            if any(count != slots for count in non_anchor_counts):
                raise ValueError(
                    "Adaptive MQ expects every non-H0 local group to contain "
                    f"exactly num_slots={slots} anchor queries."
                )
            if self.layer_local_token_source_modes[0] not in {
                "visual",
                "vision",
                "image_video",
            }:
                raise ValueError(
                    "Adaptive MQ source position 0 must use visual tokens."
                )
        if self.dynamic_topk_local_mq:
            if self.adaptive_local_mq:
                raise ValueError(
                    "Dynamic TopK MQ and adaptive Router MQ are mutually exclusive."
                )
            if not self.hierarchical_query_pooling:
                raise ValueError(
                    "Dynamic TopK MQ requires hierarchical_query_pooling=True."
                )
            if self.mq_competitive_local_attention:
                raise ValueError(
                    "Dynamic TopK MQ does not support competitive local attention."
                )
            candidate_positions = self.dynamic_topk_candidate_source_positions
            if not candidate_positions:
                raise ValueError(
                    "Dynamic TopK MQ requires candidate source positions."
                )
            if len(set(candidate_positions)) != len(candidate_positions):
                raise ValueError(
                    "Dynamic TopK MQ candidate source positions must be unique."
                )
            invalid_positions = [
                position
                for position in [
                    *candidate_positions,
                    self.dynamic_topk_text_source_position,
                ]
                if position < 0 or position >= num_hier_layers
            ]
            if invalid_positions:
                raise ValueError(
                    "Dynamic TopK MQ source positions are out of range: "
                    f"{invalid_positions}."
                )
            if self.dynamic_topk_text_source_position in candidate_positions:
                raise ValueError(
                    "The fixed text source cannot also be a TopK candidate source."
                )
            expected_positions = set(range(num_hier_layers))
            configured_positions = set(candidate_positions) | {
                self.dynamic_topk_text_source_position
            }
            if configured_positions != expected_positions:
                raise ValueError(
                    "Dynamic TopK MQ requires every hierarchical source to be "
                    "either a candidate source or the fixed text source."
                )
            if self.dynamic_topk_candidates_per_source < 1:
                raise ValueError(
                    "dynamic_topk_candidates_per_source must be positive."
                )
            candidate_counts = [
                self.layer_local_query_counts[position]
                for position in candidate_positions
            ]
            if any(
                count != self.dynamic_topk_candidates_per_source
                for count in candidate_counts
            ):
                raise ValueError(
                    "Every Dynamic TopK MQ candidate source must contain exactly "
                    f"{self.dynamic_topk_candidates_per_source} queries."
                )
            if (
                self.dynamic_topk_min_keep_per_source < 1
                or self.dynamic_topk_min_keep_per_source
                > self.dynamic_topk_candidates_per_source
            ):
                raise ValueError(
                    "dynamic_topk_min_keep_per_source must be in [1, candidates]."
                )
            if (
                self.dynamic_topk_train_random_exploration
                and self.dynamic_topk_min_keep_per_source != 2
            ):
                raise ValueError(
                    "Dynamic TopK random exploration requires exactly two "
                    "guaranteed candidates per source."
                )
            guaranteed_keep = (
                len(candidate_positions)
                * self.dynamic_topk_min_keep_per_source
            )
            total_candidates = (
                len(candidate_positions)
                * self.dynamic_topk_candidates_per_source
            )
            if not (
                guaranteed_keep
                <= self.dynamic_topk_total_keep
                <= total_candidates
            ):
                raise ValueError(
                    "dynamic_topk_total_keep must preserve minimum source "
                    "coverage and cannot exceed the candidate count."
                )
            text_mode = self.layer_local_token_source_modes[
                self.dynamic_topk_text_source_position
            ]
            if text_mode != "text":
                raise ValueError(
                    "Dynamic TopK MQ fixed text source must use text tokens."
                )
            first_candidate_mode = self.layer_local_token_source_modes[
                candidate_positions[0]
            ]
            if first_candidate_mode not in {
                "visual",
                "vision",
                "image_video",
            }:
                raise ValueError(
                    "The first Dynamic TopK MQ candidate source must use visual tokens."
                )
        self.norm = nn.LayerNorm(vlm_hidden_dim)
        self.pool_query = (
            nn.Parameter(torch.zeros(1, self.pooling_num_queries, vlm_hidden_dim))
            if self.hidden_pooling == "attention" and self.structured_mq is None
            else None
        )
        self.pool_attention = (
            nn.MultiheadAttention(vlm_hidden_dim, pooling_num_heads, batch_first=True)
            if self.hidden_pooling == "attention" and self.structured_mq is None
            else None
        )
        if self.pool_query is not None:
            nn.init.normal_(self.pool_query, std=0.02)
        self.layer_local_query = (
            nn.Parameter(torch.zeros(sum(self.layer_local_query_counts), vlm_hidden_dim))
            if self.hierarchical_query_pooling
            else None
        )
        self.cross_layer_query = (
            nn.Parameter(torch.zeros(1, self.cross_layer_queries, vlm_hidden_dim))
            if self.hierarchical_query_pooling
            else None
        )
        self.global_query = (
            nn.Parameter(torch.zeros(1, self.global_queries, latent_dim))
            if self.hierarchical_query_pooling
            else None
        )
        self.layer_embedding = (
            nn.Parameter(torch.zeros(num_hier_layers, 1, vlm_hidden_dim))
            if self.hierarchical_query_pooling
            else None
        )
        self.global_attention = (
            nn.MultiheadAttention(latent_dim, pooling_num_heads, batch_first=True)
            if self.hierarchical_query_pooling
            else None
        )
        self.global_query_norm = (
            nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)
            if self.hierarchical_global_residual_block
            else None
        )
        self.global_context_norm = (
            nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)
            if self.hierarchical_global_residual_block
            else None
        )
        self.global_residual_ffn_norm = (
            nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)
            if self.hierarchical_global_residual_block
            else None
        )
        self.global_residual_ffn = (
            nn.Sequential(
                nn.Linear(
                    latent_dim,
                    int(latent_dim * self.hierarchical_global_ffn_ratio),
                ),
                nn.GELU(approximate="tanh"),
                nn.Linear(
                    int(latent_dim * self.hierarchical_global_ffn_ratio),
                    latent_dim,
                ),
            )
            if self.hierarchical_global_residual_block
            else None
        )
        adaptive_reserve_source_count = max(
            len(self.adaptive_mq_reserve_source_positions) - 1,
            0,
        )
        self.adaptive_reserve_query = (
            nn.Parameter(
                torch.zeros(
                    adaptive_reserve_source_count * self.adaptive_mq_num_slots,
                    vlm_hidden_dim,
                )
            )
            if self.adaptive_local_mq
            else None
        )
        self.adaptive_depth_router = (
            SlotwiseDepthRouter(
                hidden_dim=latent_dim,
                router_dim=self.adaptive_mq_router_dim,
                num_slots=self.adaptive_mq_num_slots,
                num_sources=len(self.adaptive_mq_reserve_source_positions),
                temperature=self.adaptive_mq_temperature,
            )
            if self.adaptive_local_mq
            else None
        )
        if self.hierarchical_query_pooling:
            nn.init.normal_(self.layer_local_query, std=0.02)
            nn.init.normal_(self.cross_layer_query, std=0.02)
            nn.init.normal_(self.global_query, std=0.02)
            nn.init.normal_(self.layer_embedding, std=0.02)
        if self.dynamic_topk_local_mq:
            with torch.no_grad():
                query_offsets = [0]
                for count in self.layer_local_query_counts:
                    query_offsets.append(query_offsets[-1] + count)
                for source_index, position in enumerate(
                    self.dynamic_topk_candidate_source_positions
                ):
                    start = query_offsets[position]
                    count = self.layer_local_query_counts[position]
                    old_count = min(count, 8 if source_index == 0 else 4)
                    extra_count = count - old_count
                    if extra_count <= 0:
                        continue
                    base_queries = self.layer_local_query[
                        start : start + old_count
                    ].clone()
                    repeated = base_queries.repeat(
                        math.ceil(extra_count / old_count),
                        1,
                    )[:extra_count]
                    self.layer_local_query[
                        start + old_count : start + count
                    ].copy_(
                        repeated + torch.randn_like(repeated) * 1e-3
                    )
        if self.adaptive_reserve_query is not None:
            with torch.no_grad():
                base_reserve = self.layer_local_query[
                    self.adaptive_mq_num_slots : 2 * self.adaptive_mq_num_slots
                ]
                repeated = base_reserve.repeat(adaptive_reserve_source_count, 1)
                self.adaptive_reserve_query.copy_(
                    repeated + torch.randn_like(repeated) * 1e-3
                )
        self.layer_aligned_query = (
            nn.Parameter(
                torch.zeros(
                    num_aligned_layers,
                    self.layer_aligned_queries_per_layer,
                    vlm_hidden_dim,
                )
            )
            if self.layer_aligned_query_pooling
            else None
        )
        self.layer_aligned_obs_init = (
            nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(hidden_dim, latent_dim),
            )
            if self.layer_aligned_query_pooling and self.layer_aligned_encoder_obs_conditioning
            else None
        )
        self.layer_aligned_obs_q_pool = (
            nn.Linear(latent_dim, latent_dim)
            if self.layer_aligned_query_pooling and self.layer_aligned_encoder_obs_conditioning
            else None
        )
        self.layer_aligned_obs_q_prev = (
            nn.Linear(latent_dim, latent_dim)
            if self.layer_aligned_query_pooling and self.layer_aligned_encoder_obs_conditioning
            else None
        )
        self.layer_aligned_obs_attention = (
            nn.MultiheadAttention(latent_dim, pooling_num_heads, batch_first=True)
            if self.layer_aligned_query_pooling and self.layer_aligned_encoder_obs_conditioning
            else None
        )
        self.layer_aligned_global_query = (
            nn.Parameter(torch.zeros(1, self.global_queries, latent_dim))
            if self.layer_aligned_query_pooling and self.layer_aligned_global_conditioning
            else None
        )
        self.layer_aligned_global_attention = (
            nn.MultiheadAttention(latent_dim, pooling_num_heads, batch_first=True)
            if self.layer_aligned_query_pooling and self.layer_aligned_global_conditioning
            else None
        )
        self.layer_aligned_obs_pool_query = (
            nn.Parameter(torch.zeros(1, 1, latent_dim))
            if self.layer_aligned_query_pooling
            and self.layer_aligned_obs_pooling in {"attn_obs_global", "attn_local_global"}
            else None
        )
        self.layer_aligned_obs_pool_attention = (
            nn.MultiheadAttention(latent_dim, pooling_num_heads, batch_first=True)
            if self.layer_aligned_query_pooling
            and self.layer_aligned_obs_pooling in {"attn_obs_global", "attn_local_global"}
            else None
        )
        self.layer_aligned_obs_head = (
            nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(hidden_dim, latent_dim),
            )
            if self.layer_aligned_query_pooling
            else None
        )
        if self.layer_aligned_query_pooling:
            nn.init.normal_(self.layer_aligned_query, std=0.02)
            if self.layer_aligned_global_query is not None:
                nn.init.normal_(self.layer_aligned_global_query, std=0.02)
            if self.layer_aligned_obs_pool_query is not None:
                nn.init.normal_(self.layer_aligned_obs_pool_query, std=0.02)
        self.local_type_embedding = (
            nn.Parameter(torch.zeros(1, 1, latent_dim))
            if self.condition_type_embeddings
            else None
        )
        self.cross_type_embedding = (
            nn.Parameter(torch.zeros(1, 1, latent_dim))
            if self.condition_type_embeddings
            else None
        )
        self.global_type_embedding = (
            nn.Parameter(torch.zeros(1, 1, latent_dim))
            if self.condition_type_embeddings
            else None
        )
        self.state_type_embedding = (
            nn.Parameter(torch.zeros(1, 1, latent_dim))
            if self.condition_type_embeddings
            else None
        )
        if self.condition_type_embeddings:
            nn.init.normal_(self.local_type_embedding, std=0.02)
            nn.init.normal_(self.cross_type_embedding, std=0.02)
            nn.init.normal_(self.global_type_embedding, std=0.02)
            nn.init.normal_(self.state_type_embedding, std=0.02)
        self.proj = nn.Sequential(
            nn.Linear(vlm_hidden_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.proprio_dim = int(proprio_dim) if proprio_dim else None
        self.proprio_proj = (
            nn.Sequential(
                nn.LayerNorm(self.proprio_dim),
                nn.Linear(self.proprio_dim, hidden_dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(hidden_dim, latent_dim),
            )
            if self.proprio_dim is not None
            else None
        )
        self.state_token_proj = (
            nn.Sequential(
                nn.LayerNorm(self.proprio_dim),
                nn.Linear(self.proprio_dim, hidden_dim),
                nn.GELU(approximate="tanh"),
                nn.Dropout(float(state_token_dropout)),
                nn.Linear(hidden_dim, self.state_num_tokens * latent_dim),
            )
            if self.proprio_dim is not None and self.state_token_conditioning
            else None
        )
        self.fusion_norm = nn.LayerNorm(latent_dim)
        self.extra_layer_alphas = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, self.pooling_num_queries, 1))
                for _ in self.extra_hidden_layer_indices
            ]
        )
        scales = list(extra_layer_gate_scales or [])
        if len(scales) < len(self.extra_hidden_layer_indices):
            scales.extend([1.0] * (len(self.extra_hidden_layer_indices) - len(scales)))
        self.extra_layer_gate_scales = [float(s) for s in scales[: len(self.extra_hidden_layer_indices)]]
        self.extra_layer_gate_mlps = nn.ModuleList()
        if self.dynamic_extra_layer_gates:
            gate_hidden = int(extra_layer_gate_hidden_dim)
            for _ in self.extra_hidden_layer_indices:
                gate_mlp = nn.Sequential(
                    nn.LayerNorm(latent_dim),
                    nn.Linear(latent_dim, gate_hidden),
                    nn.SiLU(),
                    nn.Linear(gate_hidden, 1),
                )
                nn.init.zeros_(gate_mlp[-1].weight)
                nn.init.zeros_(gate_mlp[-1].bias)
                self.extra_layer_gate_mlps.append(gate_mlp)
        self.pool_score = (
            nn.Linear(latent_dim, 1)
            if self.gated_weighted_pooling
            else None
        )
        self.pool_gate = (
            nn.Linear(latent_dim, 1)
            if self.gated_weighted_pooling
            else None
        )
        if self.gated_weighted_pooling:
            nn.init.zeros_(self.pool_score.weight)
            nn.init.zeros_(self.pool_score.bias)
            nn.init.zeros_(self.pool_gate.weight)
            nn.init.zeros_(self.pool_gate.bias)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        reserve_key = prefix + "adaptive_reserve_query"
        local_key = prefix + "layer_local_query"
        if (
            self.dynamic_topk_local_mq
            and local_key in state_dict
            and state_dict[local_key].shape != self.layer_local_query.shape
        ):
            old_local_query = state_dict[local_key]
            text_position = self.dynamic_topk_text_source_position
            text_count = self.layer_local_query_counts[text_position]
            candidate_total = old_local_query.shape[0] - text_count
            uniform_old_count = (
                candidate_total
                // len(self.dynamic_topk_candidate_source_positions)
                if candidate_total >= 0
                and candidate_total
                % len(self.dynamic_topk_candidate_source_positions)
                == 0
                else None
            )
            if (
                uniform_old_count is not None
                and uniform_old_count > 0
                and uniform_old_count
                <= self.dynamic_topk_candidates_per_source
            ):
                old_query_counts = [
                    (
                        text_count
                        if position == text_position
                        else uniform_old_count
                    )
                    for position in range(len(self.layer_local_query_counts))
                ]
            else:
                candidate_position_to_source = {
                    position: source_index
                    for source_index, position in enumerate(
                        self.dynamic_topk_candidate_source_positions
                    )
                }
                old_query_counts = []
                for position, current_count in enumerate(
                    self.layer_local_query_counts
                ):
                    source_index = candidate_position_to_source.get(position)
                    if source_index is None:
                        old_query_counts.append(current_count)
                    else:
                        old_query_counts.append(
                            min(
                                current_count,
                                8 if source_index == 0 else 4,
                            )
                        )
            if sum(old_query_counts) == old_local_query.shape[0]:
                expanded_groups = []
                old_offset = 0
                for old_count, current_count in (
                    zip(old_query_counts, self.layer_local_query_counts)
                ):
                    old_group = old_local_query[
                        old_offset : old_offset + old_count
                    ]
                    old_offset += old_count
                    kept_group = old_group[:current_count]
                    expanded_groups.append(kept_group)
                    extra_count = current_count - kept_group.shape[0]
                    if extra_count > 0:
                        repeated = old_group.repeat(
                            math.ceil(extra_count / old_count),
                            1,
                        )[:extra_count]
                        expanded_groups.append(
                            repeated
                            + torch.randn_like(repeated) * 1e-3
                        )
                state_dict[local_key] = torch.cat(expanded_groups, dim=0)
        if (
            self.adaptive_reserve_query is not None
            and reserve_key not in state_dict
            and local_key in state_dict
        ):
            old_local_query = state_dict[local_key]
            slots = self.adaptive_mq_num_slots
            if old_local_query.shape[0] >= 2 * slots:
                h0_reserve = old_local_query[slots : 2 * slots]
                source_count = len(self.adaptive_mq_reserve_source_positions) - 1
                repeated = h0_reserve.repeat(source_count, 1)
                state_dict[reserve_key] = (
                    repeated + torch.randn_like(repeated) * 1e-3
                )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def set_adaptive_mq_step(self, step):
        self._adaptive_mq_step = None if step is None else int(step)

    def _get_adaptive_mq_route_alpha(self):
        if not self.adaptive_local_mq:
            return 1.0
        if not self.training or self.adaptive_mq_route_warmup_steps == 0:
            return 1.0
        if self._adaptive_mq_step is None:
            return 0.0
        return min(
            max(self._adaptive_mq_step, 0)
            / float(self.adaptive_mq_route_warmup_steps),
            1.0,
        )

    def _set_adaptive_mq_router_metrics(self, route_weights, route_alpha):
        self.last_adaptive_mq_route_weights = route_weights
        entropy = -(
            route_weights
            * route_weights.clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        max_weight = route_weights.max(dim=-1).values.mean()
        source_mass = route_weights.sum(dim=1).mean(dim=0)
        metrics = {
            "router/entropy": entropy,
            "router/max_weight": max_weight,
            "router/route_alpha": route_weights.new_tensor(route_alpha),
        }
        for source_index, layer_position in enumerate(
            self.adaptive_mq_reserve_source_positions
        ):
            layer_index = self.blockwise_hidden_layer_indices[layer_position]
            source_mode = self.layer_local_token_source_modes[layer_position]
            source_name = (
                f"h{layer_index}_visual"
                if source_mode in {"visual", "vision", "image_video"}
                else f"h{layer_index}"
            )
            metrics[f"router/{source_name}_mass"] = source_mass[source_index]
        self.last_adaptive_mq_router_metrics = metrics

    def _dynamic_topk_source_name(self, layer_position):
        layer_index = self.blockwise_hidden_layer_indices[layer_position]
        source_mode = self.layer_local_token_source_modes[layer_position]
        if source_mode in {"visual", "vision", "image_video"}:
            return f"h{layer_index}_visual"
        return f"h{layer_index}"

    def _set_dynamic_topk_metrics(
        self,
        candidate_scores,
        selected_counts,
        selected_mask,
        selected_indices,
        score_selected_mask,
        random_selected_mask,
    ):
        self.last_dynamic_topk_selected_mask = selected_mask.detach()
        self.last_dynamic_topk_selected_indices = selected_indices.detach()
        self.last_dynamic_topk_score_selected_mask = (
            score_selected_mask.detach()
        )
        self.last_dynamic_topk_random_selected_mask = (
            random_selected_mask.detach()
        )
        metrics = {}
        for source_index, layer_position in enumerate(
            self.dynamic_topk_candidate_source_positions
        ):
            source_name = self._dynamic_topk_source_name(layer_position)
            source_counts = selected_counts[:, source_index].float()
            metrics[
                f"dynamic_select/{source_name}_selected_count_mean"
            ] = source_counts.mean()
        if selected_mask.shape[0] > 1:
            change_rate = (
                selected_mask[1:] != selected_mask[:-1]
            ).float().mean()
        else:
            change_rate = selected_mask.new_zeros((), dtype=torch.float32)
        metrics["dynamic_select/change_rate"] = change_rate
        self.last_dynamic_topk_metrics = metrics

    def _pool_project_tokens(self, tokens, query=None):
        tokens = self.norm(tokens)
        if self.hidden_pooling == "attention":
            if query is None:
                query = self.pool_query.expand(tokens.shape[0], -1, -1)
            pooled, _ = self.pool_attention(query=query, key=tokens, value=tokens)
        else:
            pooled = tokens.mean(dim=1, keepdim=True)
        return self.proj(pooled)

    def _add_type_embedding(self, tokens, token_type):
        if not self.condition_type_embeddings:
            return tokens
        if token_type == "local":
            return tokens + self.local_type_embedding
        if token_type == "cross":
            return tokens + self.cross_type_embedding
        if token_type == "global":
            return tokens + self.global_type_embedding
        if token_type == "state":
            return tokens + self.state_type_embedding
        raise ValueError(f"Unknown condition token type: {token_type}")

    @staticmethod
    def _attention_overlap_loss(attn_weights):
        if attn_weights is None or attn_weights.shape[1] <= 1:
            return None
        attn = attn_weights.float()
        sim = attn @ attn.transpose(1, 2)
        num_queries = sim.shape[-1]
        offdiag = ~torch.eye(num_queries, device=sim.device, dtype=torch.bool)
        return sim[:, offdiag].mean()

    def _set_mq_overlap_losses(self, local_losses, cross_loss, global_loss):
        self.last_mq_overlap_loss = None
        self.last_mq_overlap_losses = {}
        weighted_losses = []

        if local_losses:
            local_loss = torch.stack(local_losses).mean()
            self.last_mq_overlap_losses["local"] = local_loss
            if self.mq_local_overlap_loss_weight > 0:
                weighted_losses.append(self.mq_local_overlap_loss_weight * local_loss)
        if cross_loss is not None:
            self.last_mq_overlap_losses["cross"] = cross_loss
            if self.mq_cross_overlap_loss_weight > 0:
                weighted_losses.append(self.mq_cross_overlap_loss_weight * cross_loss)
        if global_loss is not None:
            self.last_mq_overlap_losses["global"] = global_loss
            if self.mq_global_overlap_loss_weight > 0:
                weighted_losses.append(self.mq_global_overlap_loss_weight * global_loss)

        if weighted_losses:
            self.last_mq_overlap_loss = torch.stack(weighted_losses).sum()

    def _set_mq_balance_losses(self, local_losses):
        self.last_mq_balance_loss = None
        self.last_mq_balance_losses = {}
        if not local_losses:
            return
        local_loss = torch.stack(local_losses).mean()
        self.last_mq_balance_losses["local"] = local_loss
        if self.mq_local_balance_loss_weight > 0:
            self.last_mq_balance_loss = self.mq_local_balance_loss_weight * local_loss

    def _multihead_competitive_pool(self, query, key_value, key_padding_mask=None):
        mha = self.pool_attention
        embed_dim = query.shape[-1]
        num_heads = mha.num_heads
        head_dim = embed_dim // num_heads
        if head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads for competitive MQ pooling.")

        w_q, w_k, w_v = mha.in_proj_weight.chunk(3, dim=0)
        b_q, b_k, b_v = (
            mha.in_proj_bias.chunk(3, dim=0)
            if mha.in_proj_bias is not None
            else (None, None, None)
        )
        q = F.linear(query, w_q, b_q)
        k = F.linear(key_value, w_k, b_k)
        v = F.linear(key_value, w_v, b_v)

        bsz, num_queries = q.shape[:2]
        num_tokens = k.shape[1]
        q = q.view(bsz, num_queries, num_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, num_tokens, num_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, num_tokens, num_heads, head_dim).transpose(1, 2)

        logits = (q @ k.transpose(-1, -2)) / math.sqrt(head_dim)
        masked_logits = logits
        if key_padding_mask is not None:
            masked_logits = logits.masked_fill(key_padding_mask[:, None, None, :], torch.finfo(logits.dtype).min)

        tau = max(self.mq_competitive_attention_tau, 1e-6)
        responsibility = torch.softmax(logits / tau, dim=2)
        comp_logits = masked_logits + self.mq_competitive_attention_gamma * torch.log(
            responsibility.clamp_min(1e-6)
        )
        attn = torch.softmax(comp_logits, dim=-1)
        pooled = attn @ v
        pooled = pooled.transpose(1, 2).contiguous().view(bsz, num_queries, embed_dim)
        pooled = mha.out_proj(pooled)

        if key_padding_mask is None:
            usage = responsibility.mean(dim=-1)
        else:
            valid = (~key_padding_mask).to(dtype=responsibility.dtype)[:, None, None, :]
            usage = (responsibility * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0)
        target = torch.full_like(usage, 1.0 / max(num_queries, 1))
        balance_loss = F.mse_loss(usage.float(), target.float())
        avg_attn = attn.mean(dim=1)
        return pooled, avg_attn, balance_loss

    def _source_key_padding_mask(self, token_source_ids, mode):
        if token_source_ids is None:
            return None
        mode = str(mode).lower()
        if mode in {"all", "*"}:
            allowed = token_source_ids >= 0
        elif mode in {"non_meta", "all_no_meta", "no_meta"}:
            allowed = (token_source_ids >= 0) & (token_source_ids != 3)
        elif mode in {"visual", "vision", "image_video"}:
            allowed = (token_source_ids == 1) | (token_source_ids == 2)
        elif mode == "image":
            allowed = token_source_ids == 1
        elif mode == "video":
            allowed = token_source_ids == 2
        elif mode == "text":
            allowed = token_source_ids == 0
        elif mode in {"visual_text", "text_visual"}:
            allowed = (token_source_ids == 0) | (token_source_ids == 1) | (token_source_ids == 2)
        elif mode == "meta":
            allowed = token_source_ids == 3
        elif mode == "proprio":
            allowed = token_source_ids == 4
        else:
            raise ValueError(f"Unknown layer-local token source mode: {mode}")

        key_padding_mask = ~allowed
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked] = False
        return key_padding_mask

    @staticmethod
    def _make_shared_view_queries(queries):
        if queries.shape[-2] % 2:
            raise ValueError("Separate-view query count must be even.")
        per_view = queries.shape[-2] // 2
        return 0.5 * (queries[..., :per_view, :] + queries[..., per_view:, :])

    @staticmethod
    def _view_visual_masks(
        hidden_token_type_ids,
        image_grid_thw,
        spatial_merge_size,
    ):
        if hidden_token_type_ids is None:
            raise ValueError("Separate-view HierMQ requires hidden token type IDs.")
        if image_grid_thw is None:
            raise ValueError("Separate-view HierMQ requires image_grid_thw metadata.")
        batch_size, sequence_length = hidden_token_type_ids.shape
        grids = image_grid_thw
        if grids.ndim == 2:
            if grids.shape != (batch_size * 2, 3):
                raise ValueError(
                    "Separate-view HierMQ requires exactly two image grids per sample."
                )
            grids = grids.reshape(batch_size, 2, 3)
        if grids.ndim != 3 or grids.shape != (batch_size, 2, 3):
            raise ValueError("image_grid_thw must have shape [B*2,3] or [B,2,3].")

        merge = int(spatial_merge_size)
        if merge < 1:
            raise ValueError("spatial_merge_size must be positive.")
        view_masks = [
            torch.zeros(
                batch_size,
                sequence_length,
                device=hidden_token_type_ids.device,
                dtype=torch.bool,
            )
            for _ in range(2)
        ]
        for batch_index in range(batch_size):
            visual_positions = torch.where(
                (hidden_token_type_ids[batch_index] == 1)
                | (hidden_token_type_ids[batch_index] == 2)
            )[0]
            counts = []
            for view_index in range(2):
                temporal, height, width = [
                    int(value)
                    for value in grids[batch_index, view_index].detach().cpu().tolist()
                ]
                if height % merge or width % merge:
                    raise ValueError(
                        "Image grid dimensions must be divisible by spatial_merge_size."
                    )
                counts.append(temporal * (height // merge) * (width // merge))
            if visual_positions.numel() != sum(counts):
                raise ValueError(
                    "Visual token count does not match the two image grids: "
                    f"{visual_positions.numel()} != {sum(counts)}."
                )
            offset = 0
            for view_index, count in enumerate(counts):
                selected = visual_positions[offset : offset + count]
                view_masks[view_index][batch_index, selected] = True
                offset += count
        return tuple(view_masks)

    @staticmethod
    def _split_view_context_masks(original_key_padding_mask, view_visual_masks):
        all_visual_mask = view_visual_masks[0] | view_visual_masks[1]
        if original_key_padding_mask is None:
            original_allowed = torch.ones_like(all_visual_mask)
        else:
            original_allowed = ~original_key_padding_mask
        return tuple(
            ~(
                original_allowed
                & (~all_visual_mask | view_visual_mask)
            )
            for view_visual_mask in view_visual_masks
        )

    def _hierarchical_pool_project_tokens(
        self,
        hidden_states,
        hidden_token_type_ids=None,
        image_grid_thw=None,
        spatial_merge_size=1,
    ):
        if not isinstance(hidden_states, (tuple, list)):
            raise ValueError("Hierarchical query pooling requires all VLM hidden states.")
        local_tokens = []
        anchor_tokens = []
        reserve_tokens_by_position = {}
        dynamic_candidate_tokens = {}
        dynamic_candidate_scores = {}
        dynamic_fixed_text_tokens = None
        cross_sources = []
        cross_masks = []
        local_overlap_losses = []
        local_balance_losses = []
        self.last_adaptive_mq_route_weights = None
        self.last_adaptive_mq_router_metrics = {}
        self.last_dynamic_topk_selected_mask = None
        self.last_dynamic_topk_selected_indices = None
        self.last_dynamic_topk_score_selected_mask = None
        self.last_dynamic_topk_random_selected_mask = None
        self.last_dynamic_topk_metrics = {}
        batch_size = hidden_states[0].shape[0]
        view_visual_masks = None
        if self.hier_mq_separate_views:
            view_visual_masks = self._view_visual_masks(
                hidden_token_type_ids,
                image_grid_thw,
                spatial_merge_size,
            )
        query_offset = 0
        adaptive_query_offsets = {
            layer_position: reserve_index * self.adaptive_mq_num_slots
            for reserve_index, layer_position in enumerate(
                self.adaptive_mq_reserve_source_positions[1:]
            )
        }
        for layer_pos, layer_index in enumerate(self.blockwise_hidden_layer_indices):
            layer_tokens = self.norm(hidden_states[layer_index])
            source_mode = self.layer_local_token_source_modes[layer_pos]
            key_padding_mask = self._source_key_padding_mask(hidden_token_type_ids, source_mode)
            query_count = self.layer_local_query_counts[layer_pos]
            local_query = self.layer_local_query[
                query_offset : query_offset + query_count
            ].unsqueeze(0).expand(batch_size, -1, -1)
            query_offset += query_count
            separate_source_views = (
                self.hier_mq_separate_views and source_mode != "text"
            )
            if separate_source_views:
                shared_query = self._make_shared_view_queries(local_query)
                view_context_masks = self._split_view_context_masks(
                    key_padding_mask,
                    view_visual_masks,
                )
                pooled_views = []
                attention_views = []
                balance_views = []
                for view_context_mask in view_context_masks:
                    if self.mq_competitive_local_attention and shared_query.shape[1] > 1:
                        view_pooled, view_attn, view_balance = self._multihead_competitive_pool(
                            shared_query,
                            layer_tokens,
                            key_padding_mask=view_context_mask,
                        )
                        balance_views.append(view_balance)
                    else:
                        view_pooled, view_attn = self.pool_attention(
                            query=shared_query,
                            key=layer_tokens,
                            value=layer_tokens,
                            key_padding_mask=view_context_mask,
                            need_weights=self.use_mq_overlap_loss,
                            average_attn_weights=True,
                        )
                    pooled_views.append(view_pooled)
                    attention_views.append(view_attn)
                local_pooled = torch.cat(pooled_views, dim=1)
                local_attn = (
                    torch.cat(attention_views, dim=1)
                    if attention_views[0] is not None
                    else None
                )
                if balance_views:
                    local_balance_losses.append(torch.stack(balance_views).mean())
            elif self.mq_competitive_local_attention and query_count > 1:
                local_pooled, local_attn, local_balance = self._multihead_competitive_pool(
                    local_query,
                    layer_tokens,
                    key_padding_mask=key_padding_mask,
                )
                local_balance_losses.append(local_balance)
            else:
                is_dynamic_candidate = (
                    self.dynamic_topk_local_mq
                    and layer_pos
                    in self.dynamic_topk_candidate_source_positions
                )
                local_pooled, local_attn = self.pool_attention(
                    query=local_query,
                    key=layer_tokens,
                    value=layer_tokens,
                    key_padding_mask=key_padding_mask,
                    need_weights=(
                        self.use_mq_overlap_loss or is_dynamic_candidate
                    ),
                    average_attn_weights=not is_dynamic_candidate,
                )
            if self.use_mq_overlap_loss:
                overlap_attention = (
                    local_attn.mean(dim=1)
                    if local_attn is not None and local_attn.ndim == 4
                    else local_attn
                )
                local_overlap = self._attention_overlap_loss(
                    overlap_attention
                )
                if local_overlap is not None:
                    local_overlap_losses.append(local_overlap)
            projected_local = self._add_type_embedding(
                self.proj(local_pooled),
                "local",
            )
            if self.dynamic_topk_local_mq:
                if layer_pos in self.dynamic_topk_candidate_source_positions:
                    dynamic_candidate_tokens[layer_pos] = projected_local
                    dynamic_candidate_scores[
                        layer_pos
                    ] = attention_concentration_score(
                        local_attn,
                        key_padding_mask=key_padding_mask,
                    )
                elif layer_pos == self.dynamic_topk_text_source_position:
                    dynamic_fixed_text_tokens = projected_local
            elif self.adaptive_local_mq:
                slots = self.adaptive_mq_num_slots
                if layer_pos == 0:
                    anchor_tokens.append(projected_local[:, :slots])
                    reserve_tokens_by_position[layer_pos] = projected_local[:, slots:]
                else:
                    anchor_tokens.append(projected_local)
                    if layer_pos in adaptive_query_offsets:
                        reserve_offset = adaptive_query_offsets[layer_pos]
                        reserve_query = self.adaptive_reserve_query[
                            reserve_offset : reserve_offset + slots
                        ].unsqueeze(0).expand(batch_size, -1, -1)
                        if self.mq_competitive_local_attention and slots > 1:
                            reserve_pooled, _, _ = self._multihead_competitive_pool(
                                reserve_query,
                                layer_tokens,
                                key_padding_mask=key_padding_mask,
                            )
                        else:
                            reserve_pooled, _ = self.pool_attention(
                                query=reserve_query,
                                key=layer_tokens,
                                value=layer_tokens,
                                key_padding_mask=key_padding_mask,
                                need_weights=False,
                            )
                        reserve_tokens_by_position[layer_pos] = self._add_type_embedding(
                            self.proj(reserve_pooled),
                            "local",
                        )
            else:
                local_tokens.append(projected_local)
            cross_sources.append(layer_tokens + self.layer_embedding[layer_pos].unsqueeze(0))
            if key_padding_mask is not None:
                cross_masks.append(key_padding_mask)

        if self.dynamic_topk_local_mq:
            missing_positions = [
                position
                for position in self.dynamic_topk_candidate_source_positions
                if position not in dynamic_candidate_tokens
            ]
            if missing_positions or dynamic_fixed_text_tokens is None:
                raise RuntimeError(
                    "Dynamic TopK MQ did not produce all configured candidate "
                    f"sources or fixed text tokens; missing={missing_positions}."
                )
            candidate_tokens = torch.stack(
                [
                    dynamic_candidate_tokens[position]
                    for position in self.dynamic_topk_candidate_source_positions
                ],
                dim=1,
            )
            candidate_scores = torch.stack(
                [
                    dynamic_candidate_scores[position]
                    for position in self.dynamic_topk_candidate_source_positions
                ],
                dim=1,
            )
            (
                selected_tokens,
                selected_counts,
                selected_mask,
                selected_indices,
                score_selected_mask,
                random_selected_mask,
            ) = select_dynamic_mq(
                candidate_tokens,
                candidate_scores,
                min_keep_per_group=self.dynamic_topk_min_keep_per_source,
                total_keep=self.dynamic_topk_total_keep,
                random_exploration=(
                    self.training
                    and self.dynamic_topk_train_random_exploration
                ),
            )
            self._set_dynamic_topk_metrics(
                candidate_scores,
                selected_counts,
                selected_mask,
                selected_indices,
                score_selected_mask,
                random_selected_mask,
            )
            local_tokens = [selected_tokens, dynamic_fixed_text_tokens]
        elif self.adaptive_local_mq:
            missing_positions = [
                position
                for position in self.adaptive_mq_reserve_source_positions
                if position not in reserve_tokens_by_position
            ]
            if missing_positions:
                raise RuntimeError(
                    "Adaptive MQ reserve tokens were not produced for positions "
                    f"{missing_positions}."
                )
            reserve_tokens = torch.stack(
                [
                    reserve_tokens_by_position[position]
                    for position in self.adaptive_mq_reserve_source_positions
                ],
                dim=2,
            )
            routed_dynamic, route_weights = self.adaptive_depth_router(
                reserve_tokens
            )
            route_alpha = self._get_adaptive_mq_route_alpha()
            baseline_dynamic = reserve_tokens[:, :, 0]
            dynamic_tokens = baseline_dynamic + route_alpha * (
                routed_dynamic - baseline_dynamic
            )
            self._set_adaptive_mq_router_metrics(
                route_weights,
                route_alpha,
            )
            local_tokens = [
                anchor_tokens[0],
                dynamic_tokens,
                *anchor_tokens[1:],
            ]

        cross_source = torch.cat(cross_sources, dim=1)
        cross_query = self.cross_layer_query.expand(batch_size, -1, -1)
        cross_tokens = None
        cross_overlap = None
        if cross_query.shape[1] > 0:
            cross_key_padding_mask = torch.cat(cross_masks, dim=1) if cross_masks else None
            cross_pooled, cross_attn = self.pool_attention(
                query=cross_query,
                key=cross_source,
                value=cross_source,
                key_padding_mask=cross_key_padding_mask,
                need_weights=self.use_mq_overlap_loss,
                average_attn_weights=True,
            )
            cross_overlap = (
                self._attention_overlap_loss(cross_attn)
                if self.use_mq_overlap_loss
                else None
            )
            cross_tokens = self._add_type_embedding(self.proj(cross_pooled), "cross")

        mid_token_parts = local_tokens + ([cross_tokens] if cross_tokens is not None else [])
        mid_tokens = torch.cat(mid_token_parts, dim=1)
        global_query = self.global_query.expand(batch_size, -1, -1)
        if self.hierarchical_global_residual_block:
            normalized_global_query = self.global_query_norm(global_query)
            normalized_mid_tokens = self.global_context_norm(mid_tokens)
            global_attn_output, global_attn = self.global_attention(
                query=normalized_global_query,
                key=normalized_mid_tokens,
                value=normalized_mid_tokens,
                need_weights=self.use_mq_overlap_loss,
                average_attn_weights=True,
            )
            global_after_attention = (
                global_query
                + self.hierarchical_global_residual_scale * global_attn_output
            )
            global_tokens = (
                global_after_attention
                + self.global_residual_ffn(
                    self.global_residual_ffn_norm(global_after_attention)
                )
            )
            self.last_global_token_metrics = {
                "global_pool/query_token_std": global_query.detach().float().std(
                    dim=1,
                    unbiased=False,
                ).mean(),
                "global_pool/attention_output_token_std": (
                    global_attn_output.detach().float().std(
                        dim=1,
                        unbiased=False,
                    ).mean()
                ),
                "global_pool/after_attention_residual_token_std": (
                    global_after_attention.detach().float().std(
                        dim=1,
                        unbiased=False,
                    ).mean()
                ),
                "global_pool/after_ffn_token_std": (
                    global_tokens.detach().float().std(
                        dim=1,
                        unbiased=False,
                    ).mean()
                ),
            }
        else:
            global_tokens, global_attn = self.global_attention(
                query=global_query,
                key=mid_tokens,
                value=mid_tokens,
                need_weights=self.use_mq_overlap_loss,
                average_attn_weights=True,
            )
            self.last_global_token_metrics = {}
        global_overlap = (
            self._attention_overlap_loss(global_attn)
            if self.use_mq_overlap_loss
            else None
        )
        self._set_mq_overlap_losses(local_overlap_losses, cross_overlap, global_overlap)
        self._set_mq_balance_losses(local_balance_losses)
        global_tokens = self._add_type_embedding(global_tokens, "global")
        return torch.cat([mid_tokens, global_tokens], dim=1)

    def _layer_aligned_pool_project_tokens(self, hidden_states, hidden_token_type_ids=None):
        if not isinstance(hidden_states, (tuple, list)):
            raise ValueError("Layer-aligned MQ pooling requires all VLM hidden states.")
        self.last_mq_overlap_loss = None
        self.last_mq_overlap_losses = {}
        self.last_mq_balance_loss = None
        self.last_mq_balance_losses = {}

        batch_size = hidden_states[0].shape[0]
        layer_memories = []
        all_local_tokens = []
        obs_condition_tokens = []
        obs_token = None

        if not self.layer_aligned_encoder_obs_conditioning:
            selected_tokens = torch.stack(
                [self.norm(hidden_states[layer_index]) for layer_index in self.layer_aligned_hidden_layer_indices],
                dim=1,
            )
            num_layers, sequence_length, hidden_dim = selected_tokens.shape[1:]
            flat_tokens = selected_tokens.reshape(batch_size * num_layers, sequence_length, hidden_dim)
            flat_queries = (
                self.layer_aligned_query.unsqueeze(0)
                .expand(batch_size, -1, -1, -1)
                .reshape(batch_size * num_layers, self.layer_aligned_queries_per_layer, hidden_dim)
            )
            layer_masks = [
                self._source_key_padding_mask(hidden_token_type_ids, source_mode)
                for source_mode in self.layer_aligned_token_source_modes
            ]
            flat_mask = None
            if any(mask is not None for mask in layer_masks):
                mask_template = next(mask for mask in layer_masks if mask is not None)
                layer_masks = [
                    mask if mask is not None else torch.zeros_like(mask_template)
                    for mask in layer_masks
                ]
                flat_mask = (
                    torch.stack(layer_masks, dim=1)
                    .reshape(batch_size * num_layers, sequence_length)
                )
            local_pooled, _ = self.pool_attention(
                query=flat_queries,
                key=flat_tokens,
                value=flat_tokens,
                key_padding_mask=flat_mask,
                need_weights=False,
            )
            local_pooled = local_pooled.reshape(
                batch_size,
                num_layers,
                self.layer_aligned_queries_per_layer,
                hidden_dim,
            )
            projected_local = self._add_type_embedding(self.proj(local_pooled), "local")
            all_local_tokens = list(projected_local.unbind(dim=1))
            layer_memories = list(all_local_tokens)
        else:
            for layer_pos, layer_index in enumerate(self.layer_aligned_hidden_layer_indices):
                layer_tokens = self.norm(hidden_states[layer_index])
                source_mode = self.layer_aligned_token_source_modes[layer_pos]
                key_padding_mask = self._source_key_padding_mask(hidden_token_type_ids, source_mode)
                query = self.layer_aligned_query[layer_pos].unsqueeze(0).expand(batch_size, -1, -1)
                local_pooled, _ = self.pool_attention(
                    query=query,
                    key=layer_tokens,
                    value=layer_tokens,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )
                local_tokens = self._add_type_embedding(self.proj(local_pooled), "local")
                all_local_tokens.append(local_tokens)
                pooled_summary = local_tokens.mean(dim=1)
                if obs_token is None:
                    obs_token = self.layer_aligned_obs_init(pooled_summary).unsqueeze(1)
                else:
                    obs_query = (
                        self.layer_aligned_obs_q_pool(pooled_summary)
                        + self.layer_aligned_obs_q_prev(obs_token[:, 0])
                    ).unsqueeze(1)
                    obs_memory = torch.cat([local_tokens, obs_token], dim=1)
                    obs_token, _ = self.layer_aligned_obs_attention(
                        query=obs_query,
                        key=obs_memory,
                        value=obs_memory,
                        need_weights=False,
                    )
                obs_condition_token = self._add_type_embedding(obs_token, "global")
                obs_condition_tokens.append(obs_condition_token)
                layer_memories.append(torch.cat([local_tokens, obs_condition_token], dim=1))

        stacked_obs_tokens = torch.cat(obs_condition_tokens, dim=1) if obs_condition_tokens else None
        global_tokens = None
        if self.layer_aligned_global_conditioning:
            global_sources = list(all_local_tokens)
            if stacked_obs_tokens is not None:
                global_sources.append(stacked_obs_tokens)
            global_source = torch.cat(global_sources, dim=1)
            global_query = self.layer_aligned_global_query.expand(batch_size, -1, -1)
            global_tokens, _ = self.layer_aligned_global_attention(
                query=global_query,
                key=global_source,
                value=global_source,
                need_weights=False,
            )
            global_tokens = self._add_type_embedding(global_tokens, "global")
            if self.layer_aligned_flow_windows is None:
                layer_memories = [
                    torch.cat([layer_memory, global_tokens], dim=1)
                    for layer_memory in layer_memories
                ]

        if self.layer_aligned_flow_windows is not None:
            if self.layer_aligned_encoder_obs_conditioning:
                raise ValueError(
                    "layer_aligned_flow_windows requires layer_aligned_encoder_obs_conditioning=False."
                )
            layer_memories = []
            for window in self.layer_aligned_flow_windows:
                window_tokens = [all_local_tokens[layer_pos] for layer_pos in window]
                if global_tokens is not None:
                    window_tokens.append(global_tokens)
                layer_memories.append(torch.cat(window_tokens, dim=1))

        blockwise_condition_tokens = torch.stack(layer_memories, dim=1)
        if self.layer_aligned_obs_pooling in {"attn_obs_global", "attn_local_global"}:
            if self.layer_aligned_obs_pooling == "attn_obs_global":
                obs_pool_tokens = stacked_obs_tokens
            else:
                obs_pool_tokens = torch.cat(all_local_tokens, dim=1)
            if global_tokens is not None:
                obs_pool_tokens = torch.cat([obs_pool_tokens, global_tokens], dim=1)
            obs_query = self.layer_aligned_obs_pool_query.expand(batch_size, -1, -1)
            pooled_obs, _ = self.layer_aligned_obs_pool_attention(
                query=obs_query,
                key=obs_pool_tokens,
                value=obs_pool_tokens,
                need_weights=False,
            )
            observation_latent = self.layer_aligned_obs_head(pooled_obs[:, 0])
        else:
            observation_latent = self.layer_aligned_obs_head(obs_token[:, 0])
        return observation_latent, blockwise_condition_tokens

    def _select_tokens(self, connector_out, hidden_states):
        if self.condition_source == "auto":
            if connector_out is not None:
                return connector_out
            if hidden_states is None:
                raise ValueError("VITA conditioning requires connector or hidden states.")
            tokens = hidden_states[-1] if isinstance(hidden_states, (tuple, list)) else hidden_states
            if self.num_queries > 0 and tokens.shape[1] >= self.num_queries:
                tokens = tokens[:, -self.num_queries :]
            return tokens
        if self.condition_source == "connector":
            if connector_out is None:
                raise ValueError("VITA connector conditioning requires connector_out.")
            return connector_out
        if hidden_states is None:
            raise ValueError("VITA hidden-layer conditioning requires VLM hidden states.")
        if isinstance(hidden_states, (tuple, list)):
            return hidden_states[self.hidden_layer_index]
        if self.hidden_layer_index not in {-1, 0}:
            raise ValueError("A single hidden-state tensor only supports index -1 or 0.")
        return hidden_states

    def _select_observation_pool_tokens(self, condition_tokens):
        if not self.obs_pool_token_indices:
            return condition_tokens
        max_index = max(self.obs_pool_token_indices)
        if max_index >= condition_tokens.shape[1]:
            raise ValueError(
                "obs_pool_token_indices contains index "
                f"{max_index}, but condition_tokens only has "
                f"{condition_tokens.shape[1]} tokens."
            )
        indices = torch.as_tensor(
            self.obs_pool_token_indices,
            device=condition_tokens.device,
            dtype=torch.long,
        )
        return condition_tokens.index_select(1, indices)

    def _pool_observation_tokens(self, tokens):
        if self.gated_weighted_pooling and tokens.shape[1] > 1:
            gate = torch.sigmoid(self.pool_gate(tokens))
            logits = self.pool_score(tokens) + torch.log(gate.clamp_min(1e-6))
            weights = torch.softmax(logits, dim=1)
            return (weights * tokens).sum(dim=1)
        return tokens.mean(dim=1)

    def forward(
        self,
        connector_out=None,
        hidden_states=None,
        hidden_token_type_ids=None,
        proprioception=None,
        return_condition_tokens=False,
        image_grid_thw=None,
        spatial_merge_size=1,
    ):
        if self.structured_mq is not None:
            if proprioception is None or self.state_token_proj is None:
                raise ValueError("StructuredMQ54 requires configured proprioception state tokens.")
            proprio = proprioception.flatten(1)
            state_tokens = self.state_token_proj(proprio).view(
                proprio.shape[0], self.state_num_tokens, -1
            )
            condition_tokens = self.structured_mq(
                hidden_states=hidden_states,
                hidden_token_type_ids=hidden_token_type_ids,
                image_grid_thw=image_grid_thw,
                spatial_merge_size=spatial_merge_size,
                state_tokens=state_tokens,
            )
            condition_tokens = self.fusion_norm(condition_tokens)
            observation_latent = self._pool_observation_tokens(
                self._select_observation_pool_tokens(condition_tokens)
            )
            if return_condition_tokens:
                return observation_latent, condition_tokens
            return observation_latent
        if self.layer_aligned_query_pooling:
            observation_latent, blockwise_condition_tokens = self._layer_aligned_pool_project_tokens(
                hidden_states,
                hidden_token_type_ids=hidden_token_type_ids,
            )
            if proprioception is not None:
                if self.proprio_proj is None:
                    raise ValueError("Received proprioception but no proprioception encoder was configured.")
                proprio = proprioception.flatten(1)
                if proprio.shape[1] != self.proprio_dim:
                    raise ValueError(
                        f"Expected flattened proprioception dimension {self.proprio_dim}, "
                        f"got {proprio.shape[1]}."
                    )
                state_latent = self.proprio_proj(proprio)
                observation_latent = observation_latent + state_latent
                if self.state_token_proj is not None:
                    state_tokens = self.state_token_proj(proprio).view(
                        proprio.shape[0],
                        self.state_num_tokens,
                        -1,
                    )
                    state_tokens = self._add_type_embedding(state_tokens, "state")
                    state_tokens = state_tokens.unsqueeze(1).expand(
                        -1,
                        blockwise_condition_tokens.shape[1],
                        -1,
                        -1,
                    )
                    blockwise_condition_tokens = torch.cat(
                        [blockwise_condition_tokens, state_tokens],
                        dim=2,
                    )
            observation_latent = self.fusion_norm(observation_latent)
            blockwise_condition_tokens = self.fusion_norm(blockwise_condition_tokens)
            if return_condition_tokens:
                return observation_latent, blockwise_condition_tokens
            return observation_latent

        if self.hierarchical_query_pooling:
            condition_tokens = self._hierarchical_pool_project_tokens(
                hidden_states,
                hidden_token_type_ids=hidden_token_type_ids,
                image_grid_thw=image_grid_thw,
                spatial_merge_size=spatial_merge_size,
            )
            blockwise_condition_tokens = None
        else:
            self.last_mq_overlap_loss = None
            self.last_mq_overlap_losses = {}
            self.last_mq_balance_loss = None
            self.last_mq_balance_losses = {}
            tokens = self._select_tokens(connector_out, hidden_states)
            query = None
            if self.hidden_pooling == "attention":
                query = self.pool_query.expand(tokens.shape[0], -1, -1)
            condition_tokens = self._pool_project_tokens(tokens, query=query)

            blockwise_condition_tokens = None
            if self.blockwise_layer_conditioning:
                if not isinstance(hidden_states, (tuple, list)):
                    raise ValueError("Block-wise layer conditioning requires all VLM hidden states.")
                layer_tokens = []
                for layer_index in self.blockwise_hidden_layer_indices:
                    layer_tokens.append(
                        self._pool_project_tokens(hidden_states[layer_index], query=query)
                    )
                blockwise_condition_tokens = torch.stack(layer_tokens, dim=1)

        query = None

        if self.extra_hidden_layer_indices:
            if not isinstance(hidden_states, (tuple, list)):
                raise ValueError("Extra hidden-layer conditioning requires all VLM hidden states.")
            for idx, layer_index in enumerate(self.extra_hidden_layer_indices):
                extra_tokens = self._pool_project_tokens(hidden_states[layer_index], query=query)
                if self.dynamic_extra_layer_gates:
                    gate = torch.tanh(self.extra_layer_gate_mlps[idx](condition_tokens))
                    gate = self.extra_layer_gate_dropout(gate)
                else:
                    gate = torch.tanh(self.extra_layer_alphas[idx])
                condition_tokens = condition_tokens + self.extra_layer_gate_scales[idx] * gate * extra_tokens
        if proprioception is not None:
            if self.proprio_proj is None:
                raise ValueError("Received proprioception but no proprioception encoder was configured.")
            proprio = proprioception.flatten(1)
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(
                    f"Expected flattened proprioception dimension {self.proprio_dim}, "
                    f"got {proprio.shape[1]}."
                )
            if self.state_broadcast_to_mq:
                state_latent = self.proprio_proj(proprio).unsqueeze(1)
                condition_tokens = condition_tokens + state_latent
            if self.state_token_proj is not None:
                state_tokens = self.state_token_proj(proprio).view(
                    proprio.shape[0],
                    self.state_num_tokens,
                    -1,
                )
                state_tokens = self._add_type_embedding(state_tokens, "state")
                if blockwise_condition_tokens is not None:
                    state_tokens = state_tokens.unsqueeze(1).expand(
                        -1,
                        blockwise_condition_tokens.shape[1],
                        -1,
                        -1,
                    )
                    blockwise_condition_tokens = torch.cat(
                        [blockwise_condition_tokens, state_tokens],
                        dim=2,
                    )
                else:
                    condition_tokens = torch.cat([condition_tokens, state_tokens], dim=1)
        condition_tokens = self.fusion_norm(condition_tokens)
        observation_tokens = self._select_observation_pool_tokens(condition_tokens)
        observation_latent = self._pool_observation_tokens(observation_tokens)
        if return_condition_tokens:
            return observation_latent, blockwise_condition_tokens if blockwise_condition_tokens is not None else condition_tokens
        return observation_latent


class TimestepEmbedding(nn.Module):
    def __init__(self, output_dim, frequency_dim=256):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, output_dim * 4),
            nn.Mish(),
            nn.Linear(output_dim * 4, output_dim),
        )

    def forward(self, timestep):
        half = self.frequency_dim // 2
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=timestep.device, dtype=torch.float32)
            / max(half, 1)
        )
        angles = timestep.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([angles.cos(), angles.sin()], dim=-1)
        if self.frequency_dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return self.mlp(embedding.to(dtype=self.mlp[0].weight.dtype))


def stage_b_one_step_loss_weights(
    progress,
    flow_final=0.3,
    action_final=1.0,
):
    """Piecewise-linear FM/action supervision schedule for Stage-B."""
    progress = min(max(float(progress), 0.0), 1.0)
    flow_final = float(flow_final)
    action_final = float(action_final)
    if progress <= 0.1:
        return 1.0, 0.0
    if progress <= 0.35:
        ratio = (progress - 0.1) / 0.25
        return 1.0 - 0.4 * ratio, 0.5 * ratio
    if progress <= 0.7:
        ratio = (progress - 0.35) / 0.35
        return 0.6 + (flow_final - 0.6) * ratio, 0.5 + (action_final - 0.5) * ratio
    return flow_final, action_final


def get_flow_reconstruction_weights(
    global_step: int,
    max_train_steps: int,
) -> tuple[float, float]:
    progress = min(
        max(global_step / max(max_train_steps, 1), 0.0),
        1.0,
    )

    def lerp(start, end, ratio):
        return start + (end - start) * ratio

    if progress < 0.10:
        return 1.0, 0.10

    if progress < 0.40:
        ratio = (progress - 0.10) / 0.30
        return (
            lerp(1.0, 0.60, ratio),
            lerp(0.10, 0.50, ratio),
        )

    if progress < 0.70:
        ratio = (progress - 0.40) / 0.30
        return (
            lerp(0.60, 0.35, ratio),
            lerp(0.50, 1.00, ratio),
        )

    return 0.35, 1.00


def resolve_flow_reconstruction_weights(
    enabled: bool,
    global_step: int,
    max_train_steps: int,
    fixed_flow_weight: float,
    fixed_reconstruction_weight: float,
) -> tuple[float, float]:
    if enabled:
        return get_flow_reconstruction_weights(global_step, max_train_steps)
    return float(fixed_flow_weight), float(fixed_reconstruction_weight)


def get_flow_latent_reconstruction_weights(
    global_step: int,
    max_train_steps: int,
) -> tuple[float, float, float]:
    progress = min(
        max(global_step / max(max_train_steps, 1), 0.0),
        1.0,
    )

    def lerp(start, end, ratio):
        return start + (end - start) * ratio

    if progress < 0.10:
        return 1.0, 1.0, 0.20

    if progress < 0.40:
        ratio = (progress - 0.10) / 0.30
        return (
            lerp(1.0, 0.60, ratio),
            lerp(1.0, 0.60, ratio),
            lerp(0.20, 0.50, ratio),
        )

    if progress < 0.70:
        ratio = (progress - 0.40) / 0.30
        return (
            lerp(0.60, 0.35, ratio),
            lerp(0.60, 0.30, ratio),
            lerp(0.50, 1.00, ratio),
        )

    return 0.35, 0.25, 1.00


def recover_one_step_action_latent(interpolated, predicted_velocity, timestep):
    """Recover the t=1 action endpoint from the current straight Flow path."""
    t = timestep.to(device=interpolated.device, dtype=interpolated.dtype)
    while t.ndim < interpolated.ndim:
        t = t.unsqueeze(-1)
    return interpolated + (1.0 - t) * predicted_velocity


def masked_action_reconstruction_loss(prediction, target, loss_type="l1", mask=None):
    """Apply the existing action loss in one coordinate system with optional padding mask."""
    if prediction.shape != target.shape:
        raise ValueError("Predicted and target actions must have identical shapes.")
    if loss_type == "l1":
        element_loss = F.l1_loss(prediction.float(), target.float(), reduction="none")
    elif loss_type == "l2":
        element_loss = F.mse_loss(prediction.float(), target.float(), reduction="none")
    else:
        raise ValueError("loss_type must be 'l1' or 'l2'.")
    if mask is None:
        return element_loss.mean()
    mask = mask.to(device=element_loss.device, dtype=element_loss.dtype)
    while mask.ndim < element_loss.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(element_loss)
    return (element_loss * mask).sum() / mask.sum().clamp_min(1.0)


class GlobalRelativeContextGate(nn.Module):
    """Context-dependent value gates from within-group global-token differences."""

    def __init__(
        self,
        token_dim,
        context_dim,
        score_dim=128,
        gate_scale=0.4,
        temperature=1.0,
    ):
        super().__init__()
        score_dim = int(score_dim)
        if score_dim <= 0:
            raise ValueError("Global relative gate score_dim must be positive.")
        if float(temperature) <= 0:
            raise ValueError("Global relative gate temperature must be positive.")

        self.score_dim = score_dim
        self.gate_scale = float(gate_scale)
        self.temperature = float(temperature)
        self.score_scale = score_dim ** -0.5
        self.token_norm = nn.LayerNorm(token_dim)
        self.relative_norm = nn.LayerNorm(token_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.key_proj = nn.Linear(token_dim, score_dim, bias=False)
        self.query_proj = nn.Linear(context_dim, score_dim, bias=False)

        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.zeros_(self.query_proj.weight)

    def prepare_token_features(self, global_token_features):
        if global_token_features.ndim != 3:
            raise ValueError(
                "Global relative gate tokens must have shape [B,N,D]."
            )

        token_features = self.token_norm(global_token_features)
        relative_features = token_features - token_features.mean(
            dim=1,
            keepdim=True,
        )
        relative_features = self.relative_norm(relative_features)
        keys = self.key_proj(relative_features)
        return {
            "relative_features": relative_features,
            "keys": keys,
        }

    def score_prepared(self, prepared_tokens, global_context):
        if global_context.ndim != 2:
            raise ValueError(
                "Global relative gate context must have shape [B,D]."
            )
        keys = prepared_tokens["keys"]
        if keys.shape[0] != global_context.shape[0]:
            raise ValueError(
                "Global relative gate token/context batch sizes must match."
            )

        query = self.query_proj(self.context_norm(global_context))
        raw_logits = (
            keys * query.unsqueeze(1)
        ).sum(dim=-1) * self.score_scale
        centered_logits = raw_logits - raw_logits.mean(dim=1, keepdim=True)
        scaled_logits = centered_logits / self.temperature
        responses = torch.tanh(scaled_logits)
        centered_responses = responses - responses.mean(dim=1, keepdim=True)
        gates = 1.0 + self.gate_scale * centered_responses
        return {
            "gates": gates,
            "raw_logits": raw_logits,
            "centered_logits": centered_logits,
            "scaled_logits": scaled_logits,
            "responses": responses,
            "centered_responses": centered_responses,
            "relative_features": prepared_tokens["relative_features"],
            "keys": keys,
            "query": query,
        }

    def forward(self, global_token_features, global_context):
        return self.score_prepared(
            self.prepare_token_features(global_token_features),
            global_context,
        )


class FlowBlock(nn.Module):
    CONDITION_TYPES = ("local", "cross", "global", "state")

    def __init__(
        self,
        hidden_dim,
        mlp_ratio=4.0,
        dropout=0.0,
        cross_attention=False,
        num_heads=8,
        gated_cross_attention=False,
        typed_cross_attention=False,
        dynamic_typed_cross_attention_gates=False,
        token_value_gating=False,
        headwise_token_value_gating=False,
        cross_attention_dim=None,
        condition_token_type_counts=None,
        condition_token_source_counts=None,
        token_gate_lambda=0.1,
        token_gate_use_action_latent=False,
        token_gate_centered_residual=False,
        competitive_value_gating=False,
        global_relative_context_gating=False,
        global_relative_gate_score_dim=128,
        global_relative_gate_temperature=1.0,
        fixed_cross_attention_scale=None,
    ):
        super().__init__()
        self.cross_attention = bool(cross_attention)
        self.gated_cross_attention = bool(gated_cross_attention)
        self.typed_cross_attention = bool(typed_cross_attention)
        self.dynamic_typed_cross_attention_gates = bool(dynamic_typed_cross_attention_gates)
        self.token_value_gating = bool(token_value_gating)
        self.headwise_token_value_gating = bool(headwise_token_value_gating)
        self.token_gate_lambda = float(token_gate_lambda)
        self.token_gate_use_action_latent = bool(token_gate_use_action_latent)
        self.token_gate_centered_residual = bool(token_gate_centered_residual)
        self.competitive_value_gating = bool(competitive_value_gating)
        self.global_relative_context_gating = bool(
            global_relative_context_gating
        )
        self.fixed_cross_attention_scale = (
            float(fixed_cross_attention_scale)
            if fixed_cross_attention_scale is not None
            else None
        )
        self.capture_token_value_gates = False
        self.captured_token_value_gates = []
        self.last_token_value_gate_metrics = {}
        self.num_heads = int(num_heads)
        self.attn_dim = int(cross_attention_dim) if cross_attention_dim is not None else hidden_dim
        self.head_dim = self.attn_dim // self.num_heads
        if self.token_value_gating and not self.cross_attention:
            raise ValueError("MQ token value gating requires flow_cross_attention=True.")
        if self.cross_attention and self.attn_dim % self.num_heads != 0:
            raise ValueError("flow cross_attention_dim must be divisible by num_heads.")
        if self.headwise_token_value_gating and not self.token_value_gating:
            raise ValueError("Head-wise MQ token value gating requires token_value_gating=True.")
        if self.competitive_value_gating and not self.token_value_gating:
            raise ValueError("Competitive Value Gate requires token_value_gating=True.")
        if self.global_relative_context_gating and not self.competitive_value_gating:
            raise ValueError(
                "Global relative context gating requires competitive_value_gating=True."
            )
        if self.competitive_value_gating and self.headwise_token_value_gating:
            raise ValueError("Competitive Value Gate currently supports token-wise gates only.")
        if self.competitive_value_gating and self.token_gate_centered_residual:
            raise ValueError(
                "Competitive Value Gate and centered residual gates are mutually exclusive."
            )
        if self.competitive_value_gating and not 0.0 <= self.token_gate_lambda <= 0.5:
            raise ValueError(
                "Competitive Value Gate scale must be in [0, 0.5] to keep gates non-negative."
            )
        source_counts = tuple(
            (str(name), int(count))
            for name, count in (condition_token_source_counts or [])
            if int(count) > 0
        )
        if self.competitive_value_gating and not source_counts:
            raise ValueError(
                "Competitive Value Gate requires ordered condition token source counts."
            )
        self.gate_source_names = tuple(name for name, _ in source_counts)
        num_gate_sources = (
            len(source_counts)
            if self.competitive_value_gating
            else len(self.CONDITION_TYPES)
        )
        self.cross_norm = (
            nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
            if self.cross_attention
            else None
        )
        self.condition_norm = (
            nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
            if self.cross_attention and not self.typed_cross_attention and not self.token_value_gating
            else None
        )
        self.cross_attn = (
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            if self.cross_attention and not self.typed_cross_attention and not self.token_value_gating
            else None
        )
        self.cross_attention_alpha = (
            nn.Parameter(torch.tensor(math.atanh(0.1), dtype=torch.float32))
            if (
                self.cross_attention
                and self.gated_cross_attention
                and not self.typed_cross_attention
                and not self.token_value_gating
                and self.fixed_cross_attention_scale is None
            )
            else None
        )
        self.value_gate_condition_norm = (
            nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
            if self.cross_attention and self.token_value_gating
            else None
        )
        self.value_gate_q_proj = (
            nn.Linear(hidden_dim, self.attn_dim)
            if self.cross_attention and self.token_value_gating
            else None
        )
        self.value_gate_k_proj = (
            nn.Linear(hidden_dim, self.attn_dim)
            if self.cross_attention and self.token_value_gating
            else None
        )
        self.value_gate_v_proj = (
            nn.Linear(hidden_dim, self.attn_dim)
            if self.cross_attention and self.token_value_gating
            else None
        )
        self.value_gate_out_proj = (
            nn.Linear(self.attn_dim, hidden_dim)
            if self.cross_attention and self.token_value_gating
            else None
        )
        self.value_gate_attn_dropout = (
            nn.Dropout(dropout)
            if self.cross_attention and self.token_value_gating
            else None
        )
        self.value_gate_source_embedding = (
            nn.Embedding(num_gate_sources, hidden_dim)
            if self.cross_attention and self.token_value_gating
            else None
        )
        self.value_gate_token_mlp = (
            nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if self.cross_attention and self.token_value_gating
            else None
        )
        value_gate_global_dim = hidden_dim * (4 if self.token_gate_use_action_latent else 3)
        self.value_gate_global_mlp = (
            nn.Sequential(
                nn.LayerNorm(value_gate_global_dim),
                nn.Linear(value_gate_global_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if self.cross_attention and self.token_value_gating
            else None
        )
        value_gate_input_dim = hidden_dim * (
            3 if self.headwise_token_value_gating and self.token_gate_use_action_latent else 2
        )
        value_gate_output_dim = self.num_heads if self.headwise_token_value_gating else 1
        self.value_gate_output_dim = int(value_gate_output_dim)
        self.value_gate_mlp = (
            nn.Sequential(
                nn.LayerNorm(value_gate_input_dim),
                nn.Linear(value_gate_input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, value_gate_output_dim),
            )
            if self.cross_attention and self.token_value_gating and not self.token_gate_centered_residual
            else None
        )
        self.global_relative_context_gate = (
            GlobalRelativeContextGate(
                token_dim=hidden_dim,
                context_dim=hidden_dim,
                score_dim=global_relative_gate_score_dim,
                gate_scale=self.token_gate_lambda,
                temperature=global_relative_gate_temperature,
            )
            if (
                self.cross_attention
                and self.token_value_gating
                and self.competitive_value_gating
                and self.global_relative_context_gating
            )
            else None
        )
        self.value_gate_shared_mlp = (
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, len(self.CONDITION_TYPES) * value_gate_output_dim),
            )
            if self.cross_attention and self.token_value_gating and self.token_gate_centered_residual
            else None
        )
        self.value_gate_residual_mlp = (
            nn.Sequential(
                nn.LayerNorm(hidden_dim * 3),
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, value_gate_output_dim),
            )
            if self.cross_attention and self.token_value_gating and self.token_gate_centered_residual
            else None
        )
        self.value_gate_branch_alpha = (
            nn.Parameter(torch.zeros((), dtype=torch.float32))
            if (
                self.cross_attention
                and self.gated_cross_attention
                and self.token_value_gating
                and self.fixed_cross_attention_scale is None
            )
            else None
        )
        if self.value_gate_source_embedding is not None:
            nn.init.normal_(self.value_gate_source_embedding.weight, std=0.02)
        if self.value_gate_mlp is not None:
            nn.init.zeros_(self.value_gate_mlp[-1].weight)
            nn.init.zeros_(self.value_gate_mlp[-1].bias)
        if self.value_gate_shared_mlp is not None:
            nn.init.zeros_(self.value_gate_shared_mlp[-1].weight)
            nn.init.zeros_(self.value_gate_shared_mlp[-1].bias)
        if self.value_gate_residual_mlp is not None:
            nn.init.zeros_(self.value_gate_residual_mlp[-1].weight)
            nn.init.zeros_(self.value_gate_residual_mlp[-1].bias)
        token_type_ids = []
        condition_type_slices = []
        token_start = 0
        for type_idx, name in enumerate(self.CONDITION_TYPES):
            count = int((condition_token_type_counts or {}).get(name, 0))
            token_type_ids.extend([type_idx] * count)
            if count > 0:
                condition_type_slices.append(slice(token_start, token_start + count))
            else:
                condition_type_slices.append(None)
            token_start += count
        self.condition_type_slices = tuple(condition_type_slices)
        self.state_type_idx = self.CONDITION_TYPES.index("state")
        gate_source_ids = []
        gate_source_slices = []
        source_start = 0
        if self.competitive_value_gating:
            for source_idx, (_, count) in enumerate(source_counts):
                gate_source_ids.extend([source_idx] * count)
                gate_source_slices.append(
                    slice(source_start, source_start + count)
                )
                source_start += count
            if source_start != token_start:
                raise ValueError(
                    "Condition token source counts must sum to the total "
                    f"condition count ({source_start} != {token_start})."
                )
        else:
            gate_source_ids = list(token_type_ids)
            gate_source_slices = list(condition_type_slices)
            self.gate_source_names = self.CONDITION_TYPES
        self.gate_source_slices = tuple(gate_source_slices)
        if self.token_value_gating:
            self.register_buffer(
                "condition_type_ids",
                torch.tensor(token_type_ids, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "gate_source_ids",
                torch.tensor(gate_source_ids, dtype=torch.long),
                persistent=False,
            )
        else:
            self.condition_type_ids = None
            self.gate_source_ids = None
        self.typed_condition_norms = (
            nn.ModuleDict(
                {
                    name: nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
                    for name in self.CONDITION_TYPES
                }
            )
            if self.cross_attention and self.typed_cross_attention and not self.token_value_gating
            else None
        )
        self.typed_cross_attns = (
            nn.ModuleDict(
                {
                    name: nn.MultiheadAttention(
                        hidden_dim,
                        num_heads,
                        dropout=dropout,
                        batch_first=True,
                    )
                    for name in self.CONDITION_TYPES
                }
            )
            if self.cross_attention and self.typed_cross_attention and not self.token_value_gating
            else None
        )
        self.typed_cross_attention_alphas = (
            nn.ParameterDict(
                {
                    name: nn.Parameter(torch.tensor(math.atanh(0.1), dtype=torch.float32))
                    for name in self.CONDITION_TYPES
                }
            )
            if (
                self.cross_attention
                and self.gated_cross_attention
                and self.typed_cross_attention
                and not self.token_value_gating
                and not self.dynamic_typed_cross_attention_gates
            )
            else None
        )
        self.typed_gate_mlp = (
            nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, len(self.CONDITION_TYPES)),
            )
            if (
                self.cross_attention
                and self.gated_cross_attention
                and self.typed_cross_attention
                and not self.token_value_gating
                and self.dynamic_typed_cross_attention_gates
            )
            else None
        )
        if self.typed_gate_mlp is not None:
            nn.init.zeros_(self.typed_gate_mlp[-1].weight)
            nn.init.constant_(self.typed_gate_mlp[-1].bias, math.atanh(0.1))
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = _mlp(hidden_dim, int(hidden_dim * mlp_ratio), dropout)
        self.time_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )
        nn.init.zeros_(self.time_modulation[-1].weight)
        nn.init.zeros_(self.time_modulation[-1].bias)

    def prepare_value_gate_condition(self, context):
        if context.ndim != 3:
            raise ValueError("MQ token value gating expects condition tokens [B,Q,D].")
        if self.condition_type_ids is None or self.condition_type_ids.numel() != context.shape[1]:
            expected = 0 if self.condition_type_ids is None else int(self.condition_type_ids.numel())
            raise ValueError(
                f"MQ token value gate type count mismatch: expected {expected}, got {context.shape[1]}."
            )
        if self.gate_source_ids is None or self.gate_source_ids.numel() != context.shape[1]:
            expected = 0 if self.gate_source_ids is None else int(self.gate_source_ids.numel())
            raise ValueError(
                f"MQ token value gate source count mismatch: expected {expected}, got {context.shape[1]}."
            )
        batch_size, num_tokens, _ = context.shape
        context = self.value_gate_condition_norm(context)
        source_ids = self.gate_source_ids.to(device=context.device)
        source_emb = self.value_gate_source_embedding(source_ids).unsqueeze(0).expand(batch_size, -1, -1)

        token_features = self.value_gate_token_mlp(torch.cat([context, source_emb], dim=-1))
        pooled_features = token_features.mean(dim=1)
        state_slice = self.condition_type_slices[self.state_type_idx]
        if state_slice is not None:
            state_emb = context[:, state_slice].mean(dim=1)
        else:
            state_emb = torch.zeros_like(pooled_features)
        k = self.value_gate_k_proj(context)
        v = self.value_gate_v_proj(context)
        k = k.view(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        v = v.view(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        global_relative_tokens = None
        if self.global_relative_context_gate is not None:
            global_slice = self.condition_type_slices[
                self.CONDITION_TYPES.index("global")
            ]
            if global_slice is None:
                raise ValueError(
                    "Global relative context gating requires global tokens."
                )
            global_relative_tokens = (
                self.global_relative_context_gate.prepare_token_features(
                    token_features[:, global_slice]
                )
            )
        return {
            "context": context,
            "source_emb": source_emb,
            "token_features": token_features,
            "pooled_features": pooled_features,
            "state_emb": state_emb,
            "k": k,
            "v": v,
            "global_relative_tokens": global_relative_tokens,
        }

    def _value_gated_cross_attention(
        self,
        x,
        query,
        context,
        time_embedding,
        condition_cache=None,
    ):
        if condition_cache is None:
            if context is None:
                raise ValueError(
                    "MQ token value gating requires condition tokens or a cache."
                )
            condition_cache = self.prepare_value_gate_condition(context)
        context = condition_cache["context"]
        batch_size, num_tokens, _ = context.shape
        if batch_size != query.shape[0]:
            raise ValueError(
                "MQ token value gate cache batch size does not match the query."
            )
        source_emb = condition_cache["source_emb"]
        token_features = condition_cache["token_features"]
        pooled_features = condition_cache["pooled_features"]
        state_emb = condition_cache["state_emb"]
        global_inputs = [pooled_features, time_embedding, state_emb]
        if self.token_gate_use_action_latent:
            global_inputs.append(query[:, 0])
        global_context = self.value_gate_global_mlp(torch.cat(global_inputs, dim=-1))
        global_gate_result = None
        if self.global_relative_context_gate is not None:
            local_slice = self.condition_type_slices[
                self.CONDITION_TYPES.index("local")
            ]
            global_slice = self.condition_type_slices[
                self.CONDITION_TYPES.index("global")
            ]
            if local_slice is None or global_slice is None:
                raise ValueError(
                    "Global relative context gating requires local and global tokens."
                )

            local_features = token_features[:, local_slice]
            local_gate_context = global_context.unsqueeze(1).expand(
                -1,
                local_features.shape[1],
                -1,
            )
            local_gate_logits = self.value_gate_mlp(
                torch.cat([local_features, local_gate_context], dim=-1)
            )
            local_relevance = torch.tanh(local_gate_logits)
            local_gates = 1.0 + self.token_gate_lambda * (
                local_relevance
                - local_relevance.mean(dim=1, keepdim=True)
            )

            global_gate_result = self.global_relative_context_gate.score_prepared(
                condition_cache["global_relative_tokens"],
                global_context,
            )
            gate_logits = token_features.new_zeros(
                batch_size,
                num_tokens,
                self.value_gate_output_dim,
            )
            value_gate = token_features.new_ones(
                batch_size,
                num_tokens,
                self.value_gate_output_dim,
            )
            gate_logits = self._replace_token_slice(
                gate_logits,
                local_slice,
                local_gate_logits,
            )
            gate_logits = self._replace_token_slice(
                gate_logits,
                global_slice,
                global_gate_result["centered_logits"].unsqueeze(-1),
            )
            value_gate = self._replace_token_slice(
                value_gate,
                local_slice,
                local_gates,
            )
            value_gate = self._replace_token_slice(
                value_gate,
                global_slice,
                global_gate_result["gates"].unsqueeze(-1),
            )
        elif self.token_gate_centered_residual:
            centered_context = self._center_residual_groups(context)
            residual_inputs = torch.cat([centered_context, context, source_emb], dim=-1)
            residual_logits = self.value_gate_residual_mlp(residual_inputs)
            residual_logits = self._center_residual_groups(residual_logits)
            shared_logits_by_type = self.value_gate_shared_mlp(global_context).view(
                batch_size,
                len(self.CONDITION_TYPES),
                self.value_gate_output_dim,
            )
            shared_logits = self._expand_type_logits(shared_logits_by_type)
            gate_logits = shared_logits + residual_logits
        else:
            gate_context = global_context.unsqueeze(1).expand(-1, num_tokens, -1)
            gate_inputs = [token_features, gate_context]
            if self.headwise_token_value_gating and self.token_gate_use_action_latent:
                gate_inputs.append(query[:, :1].expand(-1, num_tokens, -1))
            gate_logits = self.value_gate_mlp(torch.cat(gate_inputs, dim=-1))
        if self.global_relative_context_gate is None:
            if self.competitive_value_gating:
                value_gate = self._competitive_value_gate(gate_logits)
            else:
                value_gate = 1.0 + self.token_gate_lambda * torch.tanh(
                    gate_logits
                )
        self._set_token_value_gate_metrics(
            value_gate,
            gate_logits=gate_logits,
            global_gate_result=global_gate_result,
        )
        if self.capture_token_value_gates:
            self.captured_token_value_gates.append(value_gate.detach().float().cpu())

        q = self.value_gate_q_proj(query)
        k = condition_cache["k"]
        v = condition_cache["v"]

        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        if self.headwise_token_value_gating:
            v = v * value_gate.transpose(1, 2).unsqueeze(-1)
        else:
            v = v * value_gate.unsqueeze(1)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn.float(), dim=-1).to(dtype=q.dtype)
        attn = self.value_gate_attn_dropout(attn)
        cross_out = torch.matmul(attn, v)
        cross_out = cross_out.transpose(1, 2).contiguous().view(batch_size, 1, self.attn_dim)
        cross_out = self.value_gate_out_proj(cross_out)
        if self.value_gate_branch_alpha is not None:
            cross_out = torch.tanh(self.value_gate_branch_alpha) * cross_out
        elif self.fixed_cross_attention_scale is not None:
            cross_out = self.fixed_cross_attention_scale * cross_out
        return cross_out

    def _competitive_value_gate(self, gate_logits):
        relevance = torch.tanh(gate_logits)
        centered = torch.zeros_like(relevance)
        for type_name in ("local", "global"):
            type_idx = self.CONDITION_TYPES.index(type_name)
            token_slice = self.condition_type_slices[type_idx]
            if token_slice is None:
                continue
            group_relevance = relevance[:, token_slice]
            centered[:, token_slice] = (
                group_relevance
                - group_relevance.mean(dim=1, keepdim=True)
            )
        return 1.0 + self.token_gate_lambda * centered

    @staticmethod
    def _replace_token_slice(values, token_slice, replacement):
        return torch.cat(
            [
                values[:, : token_slice.start],
                replacement,
                values[:, token_slice.stop :],
            ],
            dim=1,
        )

    def _set_token_value_gate_metrics(
        self,
        value_gate,
        gate_logits=None,
        global_gate_result=None,
    ):
        metrics = {}
        detached_gate = value_gate.detach().float()
        for source_name, token_slice in zip(
            self.gate_source_names,
            self.gate_source_slices,
        ):
            if source_name == "state":
                continue
            metrics[f"gate/{source_name}_mean"] = detached_gate[
                :, token_slice
            ].mean()
        for type_name in ("local", "global"):
            type_idx = self.CONDITION_TYPES.index(type_name)
            token_slice = self.condition_type_slices[type_idx]
            if token_slice is not None:
                metrics[f"gate/{type_name}_std"] = detached_gate[
                    :, token_slice
                ].std(unbiased=False)
                if gate_logits is not None:
                    metrics[f"gate/{type_name}_logit_std"] = (
                        gate_logits.detach().float()[:, token_slice].std(
                            unbiased=False
                        )
                    )
        if global_gate_result is not None:
            detached_result = {
                name: value.detach().float()
                for name, value in global_gate_result.items()
            }
            global_gates = detached_result["gates"]
            centered_logits = detached_result["centered_logits"]
            scaled_logits = detached_result["scaled_logits"]
            metrics["gate/global_gate_token_std"] = global_gates.std(
                dim=1,
                unbiased=False,
            ).mean()
            metrics["gate/global_logit_token_std"] = centered_logits.std(
                dim=1,
                unbiased=False,
            ).mean()
            metrics["gate/global_gate_mean"] = global_gates.mean()
            metrics["gate/global_gate_min"] = global_gates.min()
            metrics["gate/global_gate_max"] = global_gates.max()
            metrics["gate/global_tanh_grad_mean"] = (
                1.0 - torch.tanh(scaled_logits).square()
            ).mean()
            metrics["gate/global_relative_feature_std"] = detached_result[
                "relative_features"
            ].std(dim=1, unbiased=False).mean()
            metrics["gate/global_query_norm"] = detached_result["query"].norm(
                dim=-1
            ).mean()
            metrics["gate/global_key_norm"] = detached_result["keys"].norm(
                dim=-1
            ).mean()
        self.last_token_value_gate_metrics = metrics

    def _center_residual_groups(self, values):
        parts = []
        for type_idx, token_slice in enumerate(self.condition_type_slices):
            if token_slice is None:
                continue
            group_values = values[:, token_slice]
            if type_idx == self.state_type_idx:
                parts.append(torch.zeros_like(group_values))
            else:
                parts.append(group_values - group_values.mean(dim=1, keepdim=True))
        return torch.cat(parts, dim=1)

    def _expand_type_logits(self, shared_logits_by_type):
        parts = []
        for type_idx, token_slice in enumerate(self.condition_type_slices):
            if token_slice is None:
                continue
            count = token_slice.stop - token_slice.start
            parts.append(shared_logits_by_type[:, type_idx : type_idx + 1].expand(-1, count, -1))
        return torch.cat(parts, dim=1)

    def forward(
        self,
        x,
        time_embedding,
        condition_tokens=None,
        condition_cache=None,
    ):
        if self.cross_attention:
            if condition_tokens is None and condition_cache is None:
                raise ValueError(
                    "Flow cross-attention requires condition tokens or a cache."
                )
            query = self.cross_norm(x).unsqueeze(1)
            if self.token_value_gating:
                cross_out = self._value_gated_cross_attention(
                    x=x,
                    query=query,
                    context=condition_tokens,
                    time_embedding=time_embedding,
                    condition_cache=condition_cache,
                )
                x = x + cross_out[:, 0]
            elif self.typed_cross_attention:
                if not isinstance(condition_tokens, dict):
                    raise ValueError("Typed Flow cross-attention requires a dict of condition token groups.")
                dynamic_gates = None
                if self.typed_gate_mlp is not None:
                    dynamic_gates = self.typed_gate_mlp(torch.cat([x, time_embedding], dim=-1))
                cross_residual = None
                for type_idx, name in enumerate(self.CONDITION_TYPES):
                    tokens = condition_tokens.get(name)
                    if tokens is None or tokens.shape[1] == 0:
                        continue
                    context = self.typed_condition_norms[name](tokens)
                    cross_out, _ = self.typed_cross_attns[name](
                        query=query,
                        key=context,
                        value=context,
                    )
                    if dynamic_gates is not None:
                        cross_out = torch.tanh(dynamic_gates[:, type_idx]).view(-1, 1, 1) * cross_out
                    elif self.typed_cross_attention_alphas is not None:
                        cross_out = torch.tanh(self.typed_cross_attention_alphas[name]) * cross_out
                    cross_residual = cross_out if cross_residual is None else cross_residual + cross_out
                if cross_residual is not None:
                    x = x + cross_residual[:, 0]
            else:
                context = self.condition_norm(condition_tokens)
                cross_out, _ = self.cross_attn(query=query, key=context, value=context)
                if self.cross_attention_alpha is not None:
                    cross_out = torch.tanh(self.cross_attention_alpha) * cross_out
                elif self.fixed_cross_attention_scale is not None:
                    cross_out = self.fixed_cross_attention_scale * cross_out
                x = x + cross_out[:, 0]
        gate, scale, shift = self.time_modulation(time_embedding).chunk(3, dim=-1)
        normalized = self.norm(x) * (1 + scale) + shift
        return x + gate * self.mlp(normalized)


class FlowMemoryUpdate(nn.Module):
    """Update flow-internal memory tokens from the current action latent."""

    def __init__(self, hidden_dim, num_heads=8, dropout=0.0, init_scale=0.1, mlp_ratio=4.0):
        super().__init__()
        self.memory_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.context_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.mlp = _mlp(hidden_dim, int(hidden_dim * float(mlp_ratio)), dropout)
        self.alpha = nn.Parameter(torch.tensor(math.atanh(float(init_scale)), dtype=torch.float32))

    def forward(self, memory, x):
        context = torch.cat([memory, x.unsqueeze(1)], dim=1)
        attn_out, _ = self.attn(
            query=self.memory_norm(memory),
            key=self.context_norm(context),
            value=self.context_norm(context),
            need_weights=False,
        )
        memory = memory + torch.tanh(self.alpha) * attn_out
        memory = memory + torch.tanh(self.alpha) * self.mlp(self.memory_norm(memory))
        return memory


class PooledBottleneckFlowMemoryUpdate(nn.Module):
    """Lightweight flow memory update from pooled condition context and flow delta."""

    def __init__(
        self,
        hidden_dim,
        num_layers=1,
        dropout=0.0,
        init_scale=0.1,
        bottleneck_dim=256,
        mlp_ratio=1.0,
    ):
        super().__init__()
        self.memory_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.signal_norm = nn.LayerNorm(hidden_dim * 6, elementwise_affine=False, eps=1e-6)
        self.layer_embedding = nn.Embedding(max(int(num_layers), 1), hidden_dim)
        signal_hidden = max(int(bottleneck_dim * float(mlp_ratio)), int(bottleneck_dim))
        self.signal_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 6, signal_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(signal_hidden, hidden_dim),
        )
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.down = nn.Linear(hidden_dim, int(bottleneck_dim))
        self.up = nn.Linear(int(bottleneck_dim), hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(math.atanh(float(init_scale)), dtype=torch.float32))
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, memory, x, previous_x, context_summary, time_embedding, layer_index):
        layer_ids = torch.full(
            (x.shape[0],),
            int(layer_index),
            dtype=torch.long,
            device=x.device,
        ).clamp_max(self.layer_embedding.num_embeddings - 1)
        layer_emb = self.layer_embedding(layer_ids)
        memory_summary = memory.mean(dim=1)
        delta_x = x - previous_x
        signal_input = torch.cat(
            [memory_summary, x, delta_x, context_summary, time_embedding, layer_emb],
            dim=-1,
        )
        signal = self.signal_mlp(self.signal_norm(signal_input))
        gate = torch.sigmoid(self.gate(signal)).unsqueeze(1)
        memory_input = self.memory_norm(memory) + signal.unsqueeze(1)
        delta_memory = self.up(F.gelu(self.down(memory_input), approximate="tanh"))
        return memory + torch.tanh(self.alpha) * gate * delta_memory


class SourceBiasedCrossAttention(nn.Module):
    """Cross-attention with timestep-conditioned bias over MQ source groups."""

    def __init__(
        self,
        dim=256,
        num_heads=8,
        num_sources=7,
        num_slots=4,
        source_counts=None,
        use_slot_source_bias=False,
        use_count_correction=False,
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError("Cross-attention dim must be divisible by num_heads.")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.num_sources = int(num_sources)
        self.num_slots = int(num_slots)
        self.query_norm = nn.RMSNorm(self.dim)
        self.output_norm = nn.RMSNorm(self.dim)
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.source_bias = nn.Linear(self.dim, self.num_heads * self.num_sources)
        self.slot_source_bias = (
            nn.Parameter(torch.zeros(self.num_slots, self.num_sources))
            if use_slot_source_bias
            else None
        )
        self.use_count_correction = bool(use_count_correction)
        counts = torch.ones(self.num_sources, dtype=torch.float32)
        if source_counts is not None:
            counts = torch.as_tensor(source_counts, dtype=torch.float32)
            if counts.shape != (self.num_sources,) or torch.any(counts <= 0):
                raise ValueError("source_counts must contain one positive count per source.")
        self.register_buffer("log_source_counts", counts.log(), persistent=False)
        self.residual_gate = nn.Linear(self.dim * 3, 1)
        nn.init.zeros_(self.residual_gate.weight)
        nn.init.constant_(self.residual_gate.bias, -4.0)
        self.last_gate = None
        self.last_source_attention_mass = {}

    def forward(self, x, mq_context, source_ids, time_context, state_context):
        batch_size, num_slots, _ = x.shape
        num_tokens = mq_context.shape[1]
        q = self.q_proj(self.query_norm(x)).view(
            batch_size, num_slots, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(mq_context).view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(mq_context).view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )
        source_bias = self.source_bias(F.silu(time_context)).view(
            batch_size, self.num_heads, self.num_sources
        )
        logits = logits + source_bias.index_select(2, source_ids).unsqueeze(2).float()
        if self.slot_source_bias is not None:
            logits = logits + self.slot_source_bias.index_select(1, source_ids).unsqueeze(0).unsqueeze(0)
        if self.use_count_correction:
            logits = logits - self.log_source_counts.index_select(0, source_ids).view(1, 1, 1, -1)
        attention = torch.softmax(logits, dim=-1).to(dtype=v.dtype)
        cross_out = torch.matmul(attention, v).transpose(1, 2).reshape(
            batch_size, num_slots, self.dim
        )
        cross_out = self.out_proj(cross_out)
        expanded_time = time_context.unsqueeze(1).expand(-1, num_slots, -1)
        expanded_state = state_context.unsqueeze(1).expand(-1, num_slots, -1)
        gate = torch.sigmoid(
            self.residual_gate(
                torch.cat([self.query_norm(x), expanded_time, expanded_state], dim=-1)
            )
        )
        self.last_gate = gate.detach()
        self.last_source_attention_mass = {
            source_index: attention[..., source_ids == source_index].sum(dim=-1).mean().detach()
            for source_index in range(self.num_sources)
        }
        return x + gate * self.output_norm(cross_out)


class EDARTokenFlowBlock(nn.Module):
    """AdaLN token Flow block with optional source-aware MQ injection."""

    def __init__(
        self,
        dim=256,
        num_heads=8,
        mlp_ratio=4.0,
        cross_attention=False,
        num_sources=7,
        num_slots=4,
        source_counts=None,
        use_slot_source_bias=False,
        use_count_correction=False,
    ):
        super().__init__()
        self.self_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ffn = _mlp(dim, int(dim * mlp_ratio), 0.0)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        self.cross_attention = (
            SourceBiasedCrossAttention(
                dim=dim,
                num_heads=num_heads,
                num_sources=num_sources,
                num_slots=num_slots,
                source_counts=source_counts,
                use_slot_source_bias=use_slot_source_bias,
                use_count_correction=use_count_correction,
            )
            if cross_attention
            else None
        )

    @staticmethod
    def _modulate(x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x, mq_context, source_ids, time_context, state_context):
        shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn = (
            self.modulation(time_context + state_context).chunk(6, dim=-1)
        )
        normalized = self._modulate(self.self_norm(x), shift_msa, scale_msa)
        self_out, _ = self.self_attn(normalized, normalized, normalized, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * self_out
        if self.cross_attention is not None:
            x = self.cross_attention(
                x, mq_context, source_ids, time_context, state_context
            )
        normalized = self._modulate(self.ffn_norm(x), shift_ffn, scale_ffn)
        return x + gate_ffn.unsqueeze(1) * self.ffn(normalized)


class EDARSourceAwareTokenFlow(nn.Module):
    """EDAR Flow that preserves four 256-dimensional latent slots."""

    INJECTION_BLOCKS = (1, 3, 5, 7)

    def __init__(self, source_counts, state_tokens=2, num_layers=8, camera_ids=None):
        super().__init__()
        self.dim = 256
        self.num_slots = 4
        self.state_tokens = int(state_tokens)
        self.source_counts = tuple((str(name), int(count)) for name, count in source_counts)
        self.num_sources = len(self.source_counts)
        structured_sources = camera_ids is not None
        if sum(count for _, count in self.source_counts) != 52:
            raise ValueError("EDAR Flow requires exactly 52 MQ tokens before state tokens.")
        if self.state_tokens != 2:
            raise ValueError("EDAR Flow requires exactly two trailing state tokens.")
        source_ids = torch.cat(
            [torch.full((count,), index, dtype=torch.long) for index, (_, count) in enumerate(self.source_counts)]
        )
        self.register_buffer("source_ids", source_ids, persistent=False)
        if camera_ids is not None:
            camera_ids = torch.as_tensor(camera_ids, dtype=torch.long)
            if camera_ids.shape != (52,) or camera_ids.min() < 0 or camera_ids.max() > 2:
                raise ValueError("camera_ids must contain 52 IDs in [0,2].")
            self.register_buffer("camera_ids", camera_ids, persistent=False)
            self.camera_embedding = nn.Embedding(3, self.dim)
        else:
            self.camera_ids = None
            self.camera_embedding = None
        self.condition_projection = nn.Linear(1024, self.dim)
        self.condition_norm = nn.RMSNorm(self.dim)
        self.source_embedding = nn.Embedding(self.num_sources, self.dim)
        self.state_projection = nn.Linear(1024, self.dim)
        self.time_embedding = TimestepEmbedding(self.dim)
        self.blocks = nn.ModuleList(
            [
                EDARTokenFlowBlock(
                    dim=self.dim,
                    num_heads=8,
                    mlp_ratio=4.0,
                    cross_attention=index in self.INJECTION_BLOCKS,
                    num_sources=self.num_sources,
                    num_slots=self.num_slots,
                    source_counts=[count for _, count in self.source_counts],
                    use_slot_source_bias=structured_sources,
                    use_count_correction=structured_sources,
                )
                for index in range(int(num_layers))
            ]
        )
        if len(self.blocks) != 8:
            raise ValueError("EDAR Flow requires exactly eight blocks.")
        self.final_norm = nn.LayerNorm(self.dim)
        self.final_projection = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.final_projection.weight)
        nn.init.zeros_(self.final_projection.bias)
        self.last_token_gate_metrics = {}

    def can_cache_condition(self, condition_tokens):
        return False

    def prepare_condition_cache(self, condition_tokens):
        raise ValueError("EDAR source-aware Flow does not support condition caches.")

    def forward(self, x, timestep, condition_tokens=None, condition_cache=None):
        if condition_cache is not None:
            raise ValueError("EDAR source-aware Flow does not accept condition_cache.")
        if x.ndim != 3 or x.shape[1:] != (self.num_slots, self.dim):
            raise ValueError("EDAR Flow latent must have shape [B,4,256].")
        if condition_tokens is None or condition_tokens.ndim != 3:
            raise ValueError("EDAR Flow requires condition tokens shaped [B,54,1024].")
        if condition_tokens.shape[1:] != (52 + self.state_tokens, 1024):
            raise ValueError("EDAR Flow condition tokens must have shape [B,54,1024].")
        mq_tokens = condition_tokens[:, :52]
        state_tokens = condition_tokens[:, 52:]
        mq_context = self.condition_norm(self.condition_projection(mq_tokens))
        mq_context = mq_context + self.source_embedding(self.source_ids).unsqueeze(0)
        if self.camera_embedding is not None:
            mq_context = mq_context + self.camera_embedding(self.camera_ids).unsqueeze(0)
        state_context = self.state_projection(state_tokens.mean(dim=1))
        time_context = self.time_embedding(timestep.float()).to(dtype=x.dtype)
        mq_context = mq_context.to(dtype=x.dtype)
        state_context = state_context.to(dtype=x.dtype)
        metrics = {}
        for index, block in enumerate(self.blocks):
            x = block(x, mq_context, self.source_ids, time_context, state_context)
            if block.cross_attention is not None:
                metrics[f"block_{index}/gate_mean"] = block.cross_attention.last_gate.mean()
                for source_index, mass in block.cross_attention.last_source_attention_mass.items():
                    source_name = self.source_counts[source_index][0]
                    metrics[f"block_{index}/attention_mass/{source_name}"] = mass
        self.last_token_gate_metrics = metrics
        return self.final_projection(self.final_norm(x))


class LatentFlowNetwork(nn.Module):
    """VITA-style vector field operating on a single action latent."""

    def __init__(
        self,
        latent_dim=512,
        hidden_dim=512,
        num_layers=4,
        mlp_ratio=4.0,
        dropout=0.0,
        cross_attention=False,
        cross_attention_heads=8,
        gated_cross_attention=False,
        typed_cross_attention=False,
        dynamic_typed_cross_attention_gates=False,
        token_value_gating=False,
        headwise_token_value_gating=False,
        flow_cross_attention_dim=None,
        token_gate_lambda=0.1,
        token_gate_use_action_latent=False,
        token_gate_centered_residual=False,
        competitive_value_gating=False,
        global_relative_context_gating=False,
        global_relative_gate_score_dim=128,
        global_relative_gate_temperature=1.0,
        static_condition_cache=False,
        fixed_cross_attention_scale=None,
        condition_token_type_counts=None,
        condition_token_source_counts=None,
        num_condition_layers=0,
        layer_logits_init=None,
        aligned_layer_conditioning=False,
        flow_memory_tokens=0,
        flow_memory_update_scale=0.1,
        flow_memory_mlp_ratio=4.0,
        flow_memory_update_type="attention",
        flow_memory_shared_update=False,
        flow_memory_bottleneck_dim=256,
    ):
        super().__init__()
        self.cross_attention = bool(cross_attention)
        self.typed_cross_attention = bool(typed_cross_attention)
        self.dynamic_typed_cross_attention_gates = bool(dynamic_typed_cross_attention_gates)
        self.token_value_gating = bool(token_value_gating)
        self.headwise_token_value_gating = bool(headwise_token_value_gating)
        self.competitive_value_gating = bool(competitive_value_gating)
        self.global_relative_context_gating = bool(
            global_relative_context_gating
        )
        self.static_condition_cache = bool(static_condition_cache)
        self.last_token_gate_metrics = {}
        if self.typed_cross_attention and not self.cross_attention:
            raise ValueError("Typed Flow cross-attention requires flow_cross_attention=True.")
        if self.token_value_gating and not self.cross_attention:
            raise ValueError("MQ token value gating requires flow_cross_attention=True.")
        if self.headwise_token_value_gating and not self.token_value_gating:
            raise ValueError("Head-wise MQ token value gating requires MQ token value gating.")
        self.num_condition_layers = int(num_condition_layers)
        self.aligned_layer_conditioning = bool(aligned_layer_conditioning)
        self.flow_memory_tokens = int(flow_memory_tokens)
        self.flow_memory_update_type = str(flow_memory_update_type).lower()
        self.flow_memory_shared_update = bool(flow_memory_shared_update)
        self.condition_token_type_counts = {
            name: int((condition_token_type_counts or {}).get(name, 0))
            for name in FlowBlock.CONDITION_TYPES
        }
        if self.flow_memory_tokens < 0:
            raise ValueError("flow_memory_tokens must be non-negative.")
        if self.flow_memory_update_type not in {"attention", "pooled_bottleneck"}:
            raise ValueError("flow_memory_update_type must be 'attention' or 'pooled_bottleneck'.")
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.condition_proj = (
            nn.Linear(latent_dim, hidden_dim)
            if self.cross_attention
            else None
        )
        self.time_embedding = TimestepEmbedding(hidden_dim)
        self.blocks = nn.ModuleList(
            [
                FlowBlock(
                    hidden_dim,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    cross_attention=self.cross_attention,
                    num_heads=cross_attention_heads,
                    gated_cross_attention=gated_cross_attention,
                    typed_cross_attention=self.typed_cross_attention,
                    dynamic_typed_cross_attention_gates=self.dynamic_typed_cross_attention_gates,
                    token_value_gating=self.token_value_gating,
                    headwise_token_value_gating=self.headwise_token_value_gating,
                    cross_attention_dim=flow_cross_attention_dim,
                    condition_token_type_counts=self.condition_token_type_counts,
                    condition_token_source_counts=condition_token_source_counts,
                    token_gate_lambda=token_gate_lambda,
                    token_gate_use_action_latent=token_gate_use_action_latent,
                    token_gate_centered_residual=token_gate_centered_residual,
                    competitive_value_gating=self.competitive_value_gating,
                    global_relative_context_gating=(
                        self.global_relative_context_gating
                    ),
                    global_relative_gate_score_dim=(
                        global_relative_gate_score_dim
                    ),
                    global_relative_gate_temperature=(
                        global_relative_gate_temperature
                    ),
                    fixed_cross_attention_scale=fixed_cross_attention_scale,
                )
                for _ in range(num_layers)
            ]
        )
        self.flow_memory_init = (
            nn.Parameter(torch.zeros(1, self.flow_memory_tokens, hidden_dim))
            if self.flow_memory_tokens > 0
            else None
        )
        self.flow_memory_update = None
        self.flow_memory_updates = None
        if self.flow_memory_tokens > 0:
            update_cls = (
                PooledBottleneckFlowMemoryUpdate
                if self.flow_memory_update_type == "pooled_bottleneck"
                else FlowMemoryUpdate
            )

            def make_update():
                if self.flow_memory_update_type == "pooled_bottleneck":
                    return update_cls(
                        hidden_dim,
                        num_layers=num_layers,
                        dropout=dropout,
                        init_scale=flow_memory_update_scale,
                        bottleneck_dim=flow_memory_bottleneck_dim,
                        mlp_ratio=flow_memory_mlp_ratio,
                    )
                return update_cls(
                    hidden_dim,
                    num_heads=cross_attention_heads,
                    dropout=dropout,
                    init_scale=flow_memory_update_scale,
                    mlp_ratio=flow_memory_mlp_ratio,
                )

            if self.flow_memory_shared_update:
                self.flow_memory_update = make_update()
            else:
                self.flow_memory_updates = nn.ModuleList([make_update() for _ in range(num_layers)])
        if self.flow_memory_init is not None:
            nn.init.normal_(self.flow_memory_init, std=0.02)
        self.layer_logits = None
        if (
            self.cross_attention
            and self.num_condition_layers > 0
            and not self.aligned_layer_conditioning
        ):
            self.layer_logits = nn.Parameter(torch.zeros(num_layers, self.num_condition_layers))
            if layer_logits_init is not None:
                init = torch.tensor(layer_logits_init, dtype=torch.float32)
                if init.shape != self.layer_logits.shape:
                    raise ValueError(
                        f"layer_logits_init shape {tuple(init.shape)} does not match "
                        f"{tuple(self.layer_logits.shape)}."
                    )
                with torch.no_grad():
                    self.layer_logits.copy_(init)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def clear_gate_records(self):
        for block in self.blocks:
            block.captured_token_value_gates.clear()

    def set_gate_capture(self, enabled=True):
        enabled = bool(enabled)
        for block in self.blocks:
            block.capture_token_value_gates = enabled
            if enabled:
                block.captured_token_value_gates.clear()

    def get_gate_records(self):
        return [list(block.captured_token_value_gates) for block in self.blocks]

    def can_cache_condition(self, condition_tokens):
        return (
            self.static_condition_cache
            and self.cross_attention
            and self.token_value_gating
            and not self.aligned_layer_conditioning
            and self.layer_logits is None
            and self.flow_memory_tokens == 0
            and condition_tokens is not None
            and condition_tokens.ndim == 3
        )

    def prepare_condition_cache(self, condition_tokens):
        """Build a graph-preserving cache valid for this batch's Flow calls."""
        if not self.can_cache_condition(condition_tokens):
            raise ValueError(
                "Static Flow condition caching requires fixed [B,Q,D] token-gated "
                "conditions without layer routing or Flow memory."
            )
        context = self.condition_proj(condition_tokens)
        return {
            "context": context,
            "block_caches": tuple(
                block.prepare_value_gate_condition(context)
                for block in self.blocks
            ),
        }

    def _split_condition_tokens(self, context):
        if context.ndim != 3:
            raise ValueError("Typed Flow cross-attention expects condition tokens [B,Q,D].")
        offset = 0
        groups = {}
        for name in FlowBlock.CONDITION_TYPES:
            count = self.condition_token_type_counts.get(name, 0)
            if count > 0:
                groups[name] = context[:, offset : offset + count]
            offset += count
        if offset != context.shape[1]:
            raise ValueError(
                f"Typed condition token count mismatch: expected {offset}, got {context.shape[1]}."
            )
        return groups

    def _insert_flow_memory(self, context, memory):
        if memory is None:
            return context
        if context.ndim != 3:
            raise ValueError("Flow memory insertion expects condition tokens [B,Q,D].")
        local_count = self.condition_token_type_counts.get("local", 0)
        cross_count = self.condition_token_type_counts.get("cross", 0)
        global_count = self.condition_token_type_counts.get("global", 0)
        state_count = self.condition_token_type_counts.get("state", 0)
        base_global_count = global_count - self.flow_memory_tokens
        if base_global_count < 0:
            raise ValueError("Global token count is smaller than flow_memory_tokens.")
        insert_at = local_count + cross_count + base_global_count
        expected_without_memory = local_count + cross_count + base_global_count + state_count
        if context.shape[1] != expected_without_memory:
            raise ValueError(
                "Flow memory condition token count mismatch: "
                f"expected {expected_without_memory} before memory insertion, got {context.shape[1]}."
            )
        return torch.cat([context[:, :insert_at], memory, context[:, insert_at:]], dim=1)

    def forward(
        self,
        latent,
        timestep,
        condition_tokens=None,
        condition_cache=None,
    ):
        x = self.input_proj(latent)
        context = None
        block_condition_caches = None
        if self.cross_attention:
            if condition_cache is not None:
                context = condition_cache["context"]
                block_condition_caches = condition_cache["block_caches"]
                if len(block_condition_caches) != len(self.blocks):
                    raise ValueError(
                        "Static Flow condition cache has the wrong block count."
                    )
                if context.shape[0] != latent.shape[0]:
                    raise ValueError(
                        "Static Flow condition cache batch size does not match latent."
                    )
            else:
                if condition_tokens is None:
                    raise ValueError(
                        "Flow cross-attention requires condition tokens or a cache."
                    )
                context = self.condition_proj(condition_tokens)
        time_embedding = self.time_embedding(timestep)
        memory = (
            self.flow_memory_init.expand(latent.shape[0], -1, -1)
            if self.flow_memory_init is not None
            else None
        )
        block_gate_metrics = []
        for block_idx, block in enumerate(self.blocks):
            block_context = context
            if self.aligned_layer_conditioning and context is not None and context.ndim == 4:
                block_context = context[:, min(block_idx, context.shape[1] - 1)]
            elif self.layer_logits is not None:
                if self.typed_cross_attention:
                    raise ValueError("Typed Flow cross-attention does not support layer_logits.")
                if context.ndim != 4:
                    raise ValueError("Block-wise layer conditioning expects condition tokens [B,L,Q,D].")
                weights = torch.softmax(self.layer_logits[block_idx], dim=0)
                block_context = (context * weights.view(1, -1, 1, 1)).sum(dim=1)
            base_block_context = block_context
            if self.flow_memory_tokens > 0:
                block_context = self._insert_flow_memory(block_context, memory)
            if self.typed_cross_attention and not self.token_value_gating:
                block_context = self._split_condition_tokens(block_context)
            previous_x = x
            block_condition_cache = (
                block_condition_caches[block_idx]
                if block_condition_caches is not None
                else None
            )
            x = block(
                x,
                time_embedding,
                condition_tokens=block_context,
                condition_cache=block_condition_cache,
            )
            if block.last_token_value_gate_metrics:
                block_gate_metrics.append(block.last_token_value_gate_metrics)
            if self.flow_memory_update is not None or self.flow_memory_updates is not None:
                updater = self.flow_memory_update or self.flow_memory_updates[block_idx]
                if self.flow_memory_update_type == "pooled_bottleneck":
                    if base_block_context is None:
                        raise ValueError("Pooled bottleneck flow memory requires condition tokens.")
                    context_summary = base_block_context.mean(dim=1)
                    memory = updater(
                        memory,
                        x,
                        previous_x,
                        context_summary,
                        time_embedding,
                        block_idx,
                    )
                else:
                    memory = updater(memory, x)
        if block_gate_metrics:
            shared_keys = set.intersection(
                *(set(metrics) for metrics in block_gate_metrics)
            )
            self.last_token_gate_metrics = {
                name: torch.stack(
                    [metrics[name] for metrics in block_gate_metrics]
                ).mean()
                for name in sorted(shared_keys)
            }
        else:
            self.last_token_gate_metrics = {}
        return self.output_proj(self.norm(x))


class VitaLatentActionGenerator(nn.Module):
    """Action autoencoder plus observation-to-action conditional flow matching."""

    def __init__(
        self,
        action_dim,
        horizon,
        vlm_hidden_dim,
        num_queries=16,
        latent_dim=512,
        hidden_dim=512,
        action_ae_layers=4,
        action_encoder_type="mlp",
        action_cnn_layers=4,
        action_cnn_kernel_size=5,
        flow_layers=4,
        flow_mlp_ratio=4.0,
        dropout=0.0,
        num_sampling_steps=6,
        proprio_dim=None,
        condition_source="auto",
        hidden_layer_index=-1,
        secondary_hidden_layer_index=None,
        extra_hidden_layer_indices=None,
        hidden_pooling="mean",
        pooling_num_heads=8,
        pooling_num_queries=1,
        gated_weighted_pooling=False,
        obs_pool_token_indices=None,
        hierarchical_query_pooling=False,
        layer_local_queries_per_layer=2,
        layer_local_queries_per_layer_list=None,
        layer_local_token_source_modes=None,
        hier_mq_separate_views=False,
        cross_layer_queries=18,
        global_queries=8,
        adaptive_local_mq=False,
        adaptive_mq_router_dim=256,
        adaptive_mq_num_slots=4,
        adaptive_mq_reserve_source_positions=None,
        adaptive_mq_temperature=1.0,
        adaptive_mq_route_warmup_steps=0,
        dynamic_topk_local_mq=False,
        dynamic_topk_candidate_source_positions=None,
        dynamic_topk_text_source_position=1,
        dynamic_topk_candidates_per_source=8,
        dynamic_topk_min_keep_per_source=2,
        dynamic_topk_total_keep=24,
        dynamic_topk_train_random_exploration=False,
        condition_type_embeddings=False,
        dynamic_extra_layer_gates=False,
        extra_layer_gate_scales=None,
        extra_layer_gate_hidden_dim=256,
        extra_layer_gate_dropout=0.0,
        blockwise_layer_conditioning=False,
        blockwise_hidden_layer_indices=None,
        state_token_conditioning=False,
        state_num_tokens=4,
        state_token_dropout=0.0,
        state_broadcast_to_mq=True,
        mq_local_overlap_loss_weight=0.0,
        mq_cross_overlap_loss_weight=0.0,
        mq_global_overlap_loss_weight=0.0,
        mq_competitive_local_attention=False,
        mq_competitive_attention_tau=1.0,
        mq_competitive_attention_gamma=1.0,
        mq_local_balance_loss_weight=0.0,
        hierarchical_global_residual_block=False,
        hierarchical_global_residual_scale=0.1,
        hierarchical_global_ffn_ratio=4.0,
        layer_aligned_query_pooling=False,
        layer_aligned_hidden_layer_indices=None,
        layer_aligned_queries_per_layer=4,
        layer_aligned_token_source_modes=None,
        layer_aligned_global_conditioning=False,
        layer_aligned_obs_pooling="last",
        layer_aligned_encoder_obs_conditioning=True,
        layer_aligned_flow_windows=None,
        flow_memory_tokens=0,
        flow_memory_update_scale=0.1,
        flow_memory_mlp_ratio=4.0,
        flow_memory_update_type="attention",
        flow_memory_shared_update=False,
        flow_memory_bottleneck_dim=256,
        flow_cross_attention=False,
        flow_cross_attention_heads=8,
        flow_cross_attention_dim=None,
        gated_flow_cross_attention=False,
        typed_flow_cross_attention=False,
        dynamic_typed_flow_gates=False,
        mq_token_value_gating=False,
        mq_headwise_token_value_gating=False,
        mq_token_gate_lambda=0.1,
        mq_token_gate_use_action_latent=False,
        mq_token_gate_centered_residual=False,
        mq_competitive_value_gating=False,
        mq_global_relative_context_gating=False,
        mq_global_relative_gate_score_dim=128,
        mq_global_relative_gate_temperature=1.0,
        flow_static_condition_cache=False,
        fixed_flow_cross_attention_scale=None,
        flow_layer_logits_init=None,
        action_representation_type="legacy",
        edar_stage_a_checkpoint=None,
        edar_train_decoder=False,
        edar_token_flow=False,
        mq_type="hier_mq54",
        action_effect_enabled=False,
        action_effect_condition_action_encoder=True,
        action_effect_visual_queries=4,
        action_effect_visual_tokens_prepooled=False,
        action_effect_num_heads=8,
        action_effect_encoder_layers=2,
        action_effect_decoder_layers=2,
        action_effect_projection_dim=None,
        action_effect_state_conditioning=False,
        action_effect_visual_dim=None,
        action_effect_horizons=None,
        action_effect_action_token_dim=256,
        action_effect_multihorizon_layers=2,
    ):
        super().__init__()
        self.num_sampling_steps = int(num_sampling_steps)
        self.action_representation_type = str(action_representation_type).lower()
        if self.action_representation_type not in {"legacy", "single_view_edar_lite"}:
            raise ValueError(
                "action_representation_type must be 'legacy' or "
                "'single_view_edar_lite'."
            )
        self.edar_stage_a_checkpoint = (
            str(edar_stage_a_checkpoint) if edar_stage_a_checkpoint else None
        )
        self.edar_train_decoder = bool(edar_train_decoder)
        self.edar_token_flow = bool(edar_token_flow)
        self.mq_type = str(mq_type).lower()
        if self.edar_token_flow and self.action_representation_type != "single_view_edar_lite":
            raise ValueError("edar_token_flow requires single_view_edar_lite.")
        typed_flow_cross_attention = bool(typed_flow_cross_attention)
        mq_token_value_gating = bool(mq_token_value_gating)
        mq_headwise_token_value_gating = bool(mq_headwise_token_value_gating)
        mq_competitive_value_gating = bool(mq_competitive_value_gating)
        layer_aligned_query_pooling = bool(layer_aligned_query_pooling)
        if mq_headwise_token_value_gating and not mq_token_value_gating:
            raise ValueError("mq_headwise_token_value_gating requires mq_token_value_gating=True.")
        if mq_competitive_value_gating and not mq_token_value_gating:
            raise ValueError("mq_competitive_value_gating requires mq_token_value_gating=True.")
        if mq_competitive_value_gating and mq_headwise_token_value_gating:
            raise ValueError(
                "mq_competitive_value_gating currently supports token-wise gates only."
            )
        if (
            typed_flow_cross_attention or mq_token_value_gating
        ) and not (
            hierarchical_query_pooling
            or layer_aligned_query_pooling
            or self.mq_type == "structured_mq54"
        ):
            raise ValueError(
                "Typed/token-gated Flow conditioning requires hierarchical_query_pooling "
                "or layer_aligned_query_pooling=True, or structured_mq54."
            )
        condition_token_type_counts = None
        condition_token_source_counts = None
        aligned_layer_count = (
            len(layer_aligned_flow_windows)
            if layer_aligned_flow_windows is not None
            else len(layer_aligned_hidden_layer_indices or [])
        )
        aligned_query_count = int(layer_aligned_queries_per_layer)
        if layer_aligned_query_pooling:
            local_query_count = aligned_query_count * (
                len(layer_aligned_flow_windows[0])
                if layer_aligned_flow_windows is not None
                else 1
            )
            cross_query_count = 0
            encoder_obs_count = 1 if layer_aligned_encoder_obs_conditioning else 0
            global_query_count = (
                encoder_obs_count
                + (int(global_queries) if layer_aligned_global_conditioning else 0)
                + int(flow_memory_tokens)
            )
        else:
            if dynamic_topk_local_mq:
                if layer_local_queries_per_layer_list is None:
                    raise ValueError(
                        "Dynamic TopK MQ requires explicit per-source query counts."
                    )
                text_position = int(dynamic_topk_text_source_position)
                local_query_count = int(
                    dynamic_topk_total_keep
                ) + int(
                    layer_local_queries_per_layer_list[text_position]
                )
            else:
                local_query_count = (
                    sum(int(v) for v in layer_local_queries_per_layer_list)
                    if layer_local_queries_per_layer_list is not None
                    else len(blockwise_hidden_layer_indices or []) * int(layer_local_queries_per_layer)
                )
            cross_query_count = int(cross_layer_queries)
            global_query_count = int(global_queries)
        edar_source_counts = None
        edar_camera_ids = None
        if self.edar_token_flow:
            if self.mq_type == "structured_mq54":
                edar_source_counts = [
                    ("spatial", 32),
                    ("semantic", 12),
                    ("text", 8),
                ]
                edar_camera_ids = [0] * 16 + [1] * 16 + [0] * 6 + [1] * 6 + [2] * 8
                expected_state_tokens = (
                    int(state_num_tokens)
                    if proprio_dim is not None and state_token_conditioning
                    else 0
                )
                if expected_state_tokens != 2:
                    raise ValueError("Structured EDAR Flow requires two state tokens.")
            else:
                source_query_counts = (
                [int(value) for value in layer_local_queries_per_layer_list]
                if layer_local_queries_per_layer_list is not None
                else [
                    int(layer_local_queries_per_layer)
                    for _ in (blockwise_hidden_layer_indices or [])
                ]
            )
                source_modes = [
                str(value).lower()
                for value in (
                    layer_local_token_source_modes
                    or ["all"] * len(source_query_counts)
                )
            ]
                hidden_indices = [
                int(value) for value in (blockwise_hidden_layer_indices or [])
            ]
                if not (
                len(source_query_counts) == len(source_modes) == len(hidden_indices)
                ):
                    raise ValueError("EDAR MQ source metadata must have matching lengths.")
                edar_source_counts = []
                for hidden_index, source_mode, query_count in zip(
                hidden_indices, source_modes, source_query_counts
                ):
                    if source_mode in {"visual", "vision", "image_video"}:
                        source_name = f"h{hidden_index}_visual"
                    elif source_mode == "text":
                        source_name = f"h{hidden_index}_text"
                    else:
                        source_name = f"h{hidden_index}"
                    edar_source_counts.append((source_name, query_count))
                if global_query_count:
                    edar_source_counts.append(("global", global_query_count))
                expected_sources = [
                ("h0_visual", 8),
                ("h0_text", 4),
                ("h1", 4),
                ("h4", 4),
                ("h8", 4),
                ("h12", 4),
                ("global", 24),
                ]
                if edar_source_counts != expected_sources:
                    raise ValueError(
                    "EDAR source-aware Flow requires MQ sources "
                    f"{expected_sources}, got {edar_source_counts}."
                    )
                expected_state_tokens = (
                int(state_num_tokens)
                if proprio_dim is not None and state_token_conditioning
                else 0
                )
                if expected_state_tokens != 2:
                    raise ValueError("EDAR source-aware Flow requires two state tokens.")
        if typed_flow_cross_attention or mq_token_value_gating or int(flow_memory_tokens) > 0:
            condition_token_type_counts = {
                "local": local_query_count,
                "cross": cross_query_count,
                "global": global_query_count,
                "state": int(state_num_tokens)
                if proprio_dim is not None and state_token_conditioning
                else 0,
            }
        if mq_competitive_value_gating:
            if layer_aligned_query_pooling:
                raise ValueError(
                    "Competitive Value Gate currently requires hierarchical MQ pooling."
                )
            if dynamic_topk_local_mq or adaptive_local_mq:
                raise ValueError(
                    "Competitive Value Gate keeps the complete static MQ set and "
                    "cannot be combined with dynamic/adaptive MQ selection."
                )
            if cross_query_count:
                raise ValueError(
                    "Competitive Value Gate v1 requires cross_layer_queries=0."
                )
            source_query_counts = (
                [int(value) for value in layer_local_queries_per_layer_list]
                if layer_local_queries_per_layer_list is not None
                else [
                    int(layer_local_queries_per_layer)
                    for _ in (blockwise_hidden_layer_indices or [])
                ]
            )
            source_modes = [
                str(value).lower()
                for value in (
                    layer_local_token_source_modes
                    or ["all"] * len(source_query_counts)
                )
            ]
            hidden_indices = [
                int(value)
                for value in (blockwise_hidden_layer_indices or [])
            ]
            if not (
                len(source_query_counts)
                == len(source_modes)
                == len(hidden_indices)
            ):
                raise ValueError(
                    "Competitive Value Gate source metadata must align with "
                    "the hierarchical hidden-layer configuration."
                )
            condition_token_source_counts = []
            for hidden_index, source_mode, query_count in zip(
                hidden_indices,
                source_modes,
                source_query_counts,
            ):
                if source_mode in {"visual", "vision", "image_video"}:
                    source_name = f"h{hidden_index}_visual"
                elif source_mode == "text":
                    source_name = f"h{hidden_index}_text"
                else:
                    source_name = f"h{hidden_index}"
                condition_token_source_counts.append(
                    (source_name, query_count)
                )
            if global_query_count:
                condition_token_source_counts.append(
                    ("global", global_query_count)
                )
            state_count = condition_token_type_counts["state"]
            if state_count:
                condition_token_source_counts.append(("state", state_count))
        self.observation_encoder = ObservationLatentEncoder(
            vlm_hidden_dim=vlm_hidden_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            proprio_dim=proprio_dim,
            condition_source=condition_source,
            hidden_layer_index=hidden_layer_index,
            secondary_hidden_layer_index=secondary_hidden_layer_index,
            extra_hidden_layer_indices=extra_hidden_layer_indices,
            hidden_pooling=hidden_pooling,
            pooling_num_heads=pooling_num_heads,
            pooling_num_queries=pooling_num_queries,
            gated_weighted_pooling=gated_weighted_pooling,
            obs_pool_token_indices=obs_pool_token_indices,
            hierarchical_query_pooling=hierarchical_query_pooling,
            layer_local_queries_per_layer=layer_local_queries_per_layer,
            layer_local_queries_per_layer_list=layer_local_queries_per_layer_list,
            layer_local_token_source_modes=layer_local_token_source_modes,
            hier_mq_separate_views=hier_mq_separate_views,
            cross_layer_queries=cross_layer_queries,
            global_queries=global_queries,
            adaptive_local_mq=adaptive_local_mq,
            adaptive_mq_router_dim=adaptive_mq_router_dim,
            adaptive_mq_num_slots=adaptive_mq_num_slots,
            adaptive_mq_reserve_source_positions=adaptive_mq_reserve_source_positions,
            adaptive_mq_temperature=adaptive_mq_temperature,
            adaptive_mq_route_warmup_steps=adaptive_mq_route_warmup_steps,
            dynamic_topk_local_mq=dynamic_topk_local_mq,
            dynamic_topk_candidate_source_positions=dynamic_topk_candidate_source_positions,
            dynamic_topk_text_source_position=dynamic_topk_text_source_position,
            dynamic_topk_candidates_per_source=dynamic_topk_candidates_per_source,
            dynamic_topk_min_keep_per_source=dynamic_topk_min_keep_per_source,
            dynamic_topk_total_keep=dynamic_topk_total_keep,
            dynamic_topk_train_random_exploration=dynamic_topk_train_random_exploration,
            condition_type_embeddings=condition_type_embeddings,
            dynamic_extra_layer_gates=dynamic_extra_layer_gates,
            extra_layer_gate_scales=extra_layer_gate_scales,
            extra_layer_gate_hidden_dim=extra_layer_gate_hidden_dim,
            extra_layer_gate_dropout=extra_layer_gate_dropout,
            blockwise_layer_conditioning=blockwise_layer_conditioning,
            blockwise_hidden_layer_indices=blockwise_hidden_layer_indices,
            state_token_conditioning=state_token_conditioning,
            state_num_tokens=state_num_tokens,
            state_token_dropout=state_token_dropout,
            state_broadcast_to_mq=state_broadcast_to_mq,
            mq_local_overlap_loss_weight=mq_local_overlap_loss_weight,
            mq_cross_overlap_loss_weight=mq_cross_overlap_loss_weight,
            mq_global_overlap_loss_weight=mq_global_overlap_loss_weight,
            mq_competitive_local_attention=mq_competitive_local_attention,
            mq_competitive_attention_tau=mq_competitive_attention_tau,
            mq_competitive_attention_gamma=mq_competitive_attention_gamma,
            mq_local_balance_loss_weight=mq_local_balance_loss_weight,
            hierarchical_global_residual_block=hierarchical_global_residual_block,
            hierarchical_global_residual_scale=hierarchical_global_residual_scale,
            hierarchical_global_ffn_ratio=hierarchical_global_ffn_ratio,
            layer_aligned_query_pooling=layer_aligned_query_pooling,
            layer_aligned_hidden_layer_indices=layer_aligned_hidden_layer_indices,
            layer_aligned_queries_per_layer=layer_aligned_queries_per_layer,
            layer_aligned_token_source_modes=layer_aligned_token_source_modes,
            layer_aligned_global_conditioning=layer_aligned_global_conditioning,
            layer_aligned_obs_pooling=layer_aligned_obs_pooling,
            layer_aligned_encoder_obs_conditioning=layer_aligned_encoder_obs_conditioning,
            layer_aligned_flow_windows=layer_aligned_flow_windows,
            mq_type=self.mq_type,
        )
        action_encoder_type = str(action_encoder_type).lower()
        if action_encoder_type not in {"mlp", "temporal_cnn"}:
            raise ValueError(
                "action_encoder_type must be 'mlp' or 'temporal_cnn'."
            )
        if self.action_representation_type == "single_view_edar_lite":
            base_action_encoder = None
        elif action_encoder_type == "temporal_cnn":
            base_action_encoder = TemporalCNNActionEncoder(
                action_dim=action_dim,
                horizon=horizon,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_layers=action_cnn_layers,
                kernel_size=action_cnn_kernel_size,
            )
        else:
            base_action_encoder = ActionEncoder(
                action_dim=action_dim,
                horizon=horizon,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_layers=action_ae_layers,
                dropout=dropout,
            )
        self.action_effect_enabled = bool(action_effect_enabled)
        self.action_effect_visual_queries = int(action_effect_visual_queries)
        if self.action_effect_visual_queries <= 0:
            raise ValueError("action_effect_visual_queries must be positive.")
        self.action_effect_visual_tokens_prepooled = bool(
            action_effect_visual_tokens_prepooled
        )
        self.action_effect_condition_action_encoder = bool(
            action_effect_condition_action_encoder
        )
        self.action_effect_state_conditioning = bool(action_effect_state_conditioning)
        self.action_effect_projection_dim = (
            int(action_effect_projection_dim)
            if action_effect_projection_dim is not None
            else None
        )
        self.action_effect_visual_dim = int(
            action_effect_visual_dim or latent_dim
        )
        self.action_effect_horizons = tuple(
            int(value) for value in (action_effect_horizons or ())
        )
        if self.action_effect_projection_dim is not None and self.action_effect_projection_dim <= 0:
            raise ValueError("action_effect_projection_dim must be positive or None.")
        if self.action_effect_state_conditioning and proprio_dim is None:
            raise ValueError("Action-effect state conditioning requires proprio_dim.")
        if self.action_effect_horizons and self.action_effect_condition_action_encoder:
            raise ValueError(
                "Multi-horizon action-effect prediction requires the original "
                "unconditioned ActionEncoder."
            )
        self.edar_encoder = None
        self.edar_decoder = None
        self.edar_dino_metadata = None
        self.edar_observation_adapter = None
        if self.action_representation_type == "single_view_edar_lite":
            if int(latent_dim) != 1024:
                raise ValueError("EDAR-lite Stage B requires vita_latent_dim=1024.")
            if not self.edar_stage_a_checkpoint:
                raise ValueError("EDAR-lite Stage B requires edar_stage_a_checkpoint.")
            checkpoint = torch.load(
                self.edar_stage_a_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            if checkpoint.get("schema") != "single_view_edar_lite_stage_a_v1":
                raise ValueError("EDAR Stage A checkpoint has an incompatible schema.")
            architecture = checkpoint["config"]["model"]["action_representation"]
            self.edar_dino_metadata = dict(checkpoint["dino_metadata"])
            visual_dim = int(self.edar_dino_metadata["hidden_size"])
            if int(self.edar_dino_metadata.get("image_size", 256)) != 256:
                raise ValueError("EDAR Stage A checkpoint must use 256px DINO inputs.")
            if int(self.edar_dino_metadata.get("output_grid", 8)) != 8:
                raise ValueError("EDAR Stage A checkpoint must use an 8x8 DINO grid.")
            edar_kwargs = {
                "action_dim": int(action_dim),
                "action_horizon": int(horizon),
                "visual_dim": visual_dim,
                "visual_grid": int(architecture.get("visual_grid", 8)),
                "model_dim": int(architecture.get("model_dim", 512)),
                "latent_tokens": int(architecture.get("latent_tokens", 4)),
                "latent_token_dim": int(architecture.get("latent_token_dim", 256)),
                "num_layers": int(architecture.get("layers", 4)),
                "num_heads": int(architecture.get("heads", 8)),
                "mlp_ratio": float(architecture.get("mlp_ratio", 4.0)),
            }
            self.edar_encoder = SingleViewEDARLiteEncoder(**edar_kwargs)
            self.edar_decoder = SingleViewEDARLiteDecoder(**edar_kwargs)
            self.edar_latent_tokens = int(edar_kwargs["latent_tokens"])
            self.edar_latent_token_dim = int(edar_kwargs["latent_token_dim"])
            if (self.edar_latent_tokens, self.edar_latent_token_dim) != (4, 256):
                raise ValueError(
                    "EDAR source-aware Flow requires Stage-A latents shaped [B,4,256]."
                )
            if self.edar_token_flow:
                self.edar_observation_adapter = nn.Linear(1024, 1024)
                with torch.no_grad():
                    self.edar_observation_adapter.weight.copy_(torch.eye(1024))
                    self.edar_observation_adapter.bias.zero_()
            self.edar_encoder.load_state_dict(
                checkpoint["encoder_state_dict"],
                strict=True,
            )
            self.edar_decoder.load_state_dict(
                checkpoint["decoder_state_dict"],
                strict=True,
            )
            self.edar_encoder.requires_grad_(False).eval()
            self.edar_decoder.requires_grad_(self.edar_train_decoder)
            self.edar_decoder.train(self.edar_train_decoder)
            self.action_effect_visual_dim = visual_dim
            self.action_encoder = None
            self.visual_resampler = None
            self.action_effect_state_encoder = None
            self.action_effect_projector = None
            self.visual_delta_decoder = None
            self.multi_horizon_visual_predictor = None
            del checkpoint
        elif self.action_effect_enabled:
            self.visual_resampler = (
                None
                if self.action_effect_visual_tokens_prepooled
                else VisualTokenResampler(
                    dim=self.action_effect_visual_dim,
                    num_queries=action_effect_visual_queries,
                    num_heads=action_effect_num_heads,
                )
            )
            if self.action_effect_condition_action_encoder:
                self.action_encoder = VisualConditionedActionEncoder(
                    base_encoder=base_action_encoder,
                    action_dim=action_dim,
                    horizon=horizon,
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    num_layers=action_effect_encoder_layers,
                    num_heads=action_effect_num_heads,
                    dropout=dropout,
                    visual_dim=self.action_effect_visual_dim,
                )
            else:
                self.action_encoder = base_action_encoder
            if self.action_effect_horizons:
                self.action_effect_state_encoder = None
                self.action_effect_projector = None
                self.visual_delta_decoder = None
                self.multi_horizon_visual_predictor = MultiHorizonVisualPredictor(
                    action_dim=action_dim,
                    action_horizon=horizon,
                    action_latent_dim=latent_dim,
                    visual_dim=self.action_effect_visual_dim,
                    visual_tokens=self.action_effect_visual_queries,
                    horizons=self.action_effect_horizons,
                    token_dim=int(action_effect_action_token_dim),
                    num_layers=int(action_effect_multihorizon_layers),
                    num_heads=action_effect_num_heads,
                    dropout=dropout,
                )
            else:
                self.action_effect_state_encoder = (
                    ActionEffectStateEncoder(
                        proprio_dim=proprio_dim,
                        latent_dim=latent_dim,
                        hidden_dim=hidden_dim,
                        dropout=dropout,
                    )
                    if self.action_effect_state_conditioning
                    else None
                )
                action_effect_state_dim = (
                    latent_dim if self.action_effect_state_conditioning else None
                )
                self.action_effect_projector = (
                    ActionEffectProjector(
                        latent_dim,
                        self.action_effect_projection_dim or latent_dim,
                        state_dim=action_effect_state_dim,
                    )
                    if self.action_effect_projection_dim is not None
                    or self.action_effect_state_conditioning
                    else nn.Identity()
                )
                self.visual_delta_decoder = VisualDeltaDecoder(
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    num_layers=action_effect_decoder_layers,
                    dropout=dropout,
                    effect_dim=self.action_effect_projection_dim,
                    state_dim=action_effect_state_dim,
                    visual_dim=self.action_effect_visual_dim,
                )
                self.multi_horizon_visual_predictor = None
        else:
            self.action_encoder = base_action_encoder
            self.visual_resampler = None
            self.action_effect_state_encoder = None
            self.action_effect_projector = None
            self.visual_delta_decoder = None
            self.multi_horizon_visual_predictor = None
        self.action_decoder = (
            None
            if self.action_representation_type == "single_view_edar_lite"
            else ActionDecoder(
                action_dim=action_dim,
                horizon=horizon,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_layers=action_ae_layers,
                dropout=dropout,
            )
        )
        if self.edar_token_flow:
            self.flow = EDARSourceAwareTokenFlow(
                source_counts=edar_source_counts,
                state_tokens=int(state_num_tokens),
                num_layers=flow_layers,
                camera_ids=edar_camera_ids,
            )
        else:
            self.flow = LatentFlowNetwork(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=flow_layers,
            mlp_ratio=flow_mlp_ratio,
            dropout=dropout,
            cross_attention=flow_cross_attention,
            cross_attention_heads=flow_cross_attention_heads,
            flow_cross_attention_dim=flow_cross_attention_dim,
            gated_cross_attention=gated_flow_cross_attention,
            typed_cross_attention=typed_flow_cross_attention,
            dynamic_typed_cross_attention_gates=dynamic_typed_flow_gates,
            token_value_gating=mq_token_value_gating,
            headwise_token_value_gating=mq_headwise_token_value_gating,
            token_gate_lambda=mq_token_gate_lambda,
            token_gate_use_action_latent=mq_token_gate_use_action_latent,
            token_gate_centered_residual=mq_token_gate_centered_residual,
            competitive_value_gating=mq_competitive_value_gating,
            global_relative_context_gating=mq_global_relative_context_gating,
            global_relative_gate_score_dim=mq_global_relative_gate_score_dim,
            global_relative_gate_temperature=mq_global_relative_gate_temperature,
            static_condition_cache=flow_static_condition_cache,
            fixed_cross_attention_scale=fixed_flow_cross_attention_scale,
            condition_token_type_counts=condition_token_type_counts,
            condition_token_source_counts=condition_token_source_counts,
            num_condition_layers=(
                aligned_layer_count
                if layer_aligned_query_pooling
                else len(blockwise_hidden_layer_indices or []) if blockwise_layer_conditioning else 0
            ),
            layer_logits_init=flow_layer_logits_init,
            aligned_layer_conditioning=layer_aligned_query_pooling,
            flow_memory_tokens=flow_memory_tokens,
            flow_memory_update_scale=flow_memory_update_scale,
            flow_memory_mlp_ratio=flow_memory_mlp_ratio,
            flow_memory_update_type=flow_memory_update_type,
            flow_memory_shared_update=flow_memory_shared_update,
            flow_memory_bottleneck_dim=flow_memory_bottleneck_dim,
            )

    def set_adaptive_mq_step(self, step):
        self.observation_encoder.set_adaptive_mq_step(step)

    def encode_observation(
        self,
        connector_out=None,
        hidden_states=None,
        hidden_token_type_ids=None,
        proprioception=None,
        return_condition_tokens=False,
        image_grid_thw=None,
        spatial_merge_size=1,
    ):
        encoded = self.observation_encoder(
            connector_out=connector_out,
            hidden_states=hidden_states,
            hidden_token_type_ids=hidden_token_type_ids,
            proprioception=proprioception,
            return_condition_tokens=return_condition_tokens,
            image_grid_thw=image_grid_thw,
            spatial_merge_size=spatial_merge_size,
        )
        if not self.edar_token_flow:
            return encoded
        if return_condition_tokens:
            observation_latent, condition_tokens = encoded
        else:
            observation_latent = encoded
            condition_tokens = None
        observation_latent = self.edar_observation_adapter(observation_latent)
        observation_latent = observation_latent.reshape(
            observation_latent.shape[0], self.edar_latent_tokens, self.edar_latent_token_dim
        )
        if return_condition_tokens:
            return observation_latent, condition_tokens
        return observation_latent

    def flow_matching_loss(
        self,
        observation_latent,
        action_latent,
        condition_tokens=None,
        condition_cache=None,
    ):
        details = self.flow_matching_step(
            observation_latent,
            action_latent,
            condition_tokens=condition_tokens,
            condition_cache=condition_cache,
        )
        return details["loss"]

    def flow_matching_step(
        self,
        observation_latent,
        action_latent,
        condition_tokens=None,
        condition_cache=None,
    ):
        timestep = torch.rand(
            observation_latent.shape[0],
            device=observation_latent.device,
            dtype=torch.float32,
        ).clamp_max(0.999)
        t = timestep.to(dtype=observation_latent.dtype)
        while t.ndim < observation_latent.ndim:
            t = t.unsqueeze(-1)
        interpolated = (1 - t) * observation_latent + t * action_latent
        target_velocity = action_latent - observation_latent
        predicted_velocity = self.flow(
            interpolated,
            timestep,
            condition_tokens=condition_tokens,
            condition_cache=condition_cache,
        )
        return {
            "loss": F.mse_loss(predicted_velocity.float(), target_velocity.float()),
            "timestep": timestep,
            "interpolated": interpolated,
            "target_velocity": target_velocity,
            "predicted_velocity": predicted_velocity,
        }

    def prepare_flow_condition_cache(self, condition_tokens):
        return self.flow.prepare_condition_cache(condition_tokens)

    def sample_latent(
        self,
        observation_latent,
        num_steps=None,
        condition_tokens=None,
        condition_cache=None,
    ):
        steps = int(num_steps or self.num_sampling_steps)
        if (
            condition_cache is None
            and self.flow.can_cache_condition(condition_tokens)
        ):
            condition_cache = self.prepare_flow_condition_cache(
                condition_tokens
            )
        latent = observation_latent
        dt = 1.0 / steps
        for step in range(steps):
            timestep = torch.full(
                (latent.shape[0],),
                step / steps,
                device=latent.device,
                dtype=torch.float32,
            )
            latent = latent + self.flow(
                latent,
                timestep,
                condition_tokens=condition_tokens,
                condition_cache=condition_cache,
            ) * dt
        return latent

    def train(self, mode=True):
        super().train(mode)
        if self.edar_encoder is not None:
            self.edar_encoder.eval()
            self.edar_decoder.train(mode and self.edar_train_decoder)
        return self

    def decode(self, action_latent):
        if self.action_representation_type == "single_view_edar_lite":
            if self.edar_token_flow:
                action_latent = self.flatten_action_latent(action_latent)
            return self.edar_decoder.decode_actions(action_latent)
        return self.action_decoder(action_latent)

    def flatten_action_latent(self, action_latent):
        if not self.edar_token_flow:
            return action_latent
        if action_latent.ndim != 3 or tuple(action_latent.shape[1:]) != (4, 256):
            raise ValueError("EDAR action latent must have shape [B,4,256].")
        return action_latent.flatten(1)

    def resample_visual_tokens(self, visual_tokens):
        if self.action_representation_type == "single_view_edar_lite":
            expected_tokens = self.edar_encoder.num_visual_tokens
            if visual_tokens.ndim != 3 or visual_tokens.shape[1] != expected_tokens:
                raise ValueError(
                    f"EDAR-lite expects [B,{expected_tokens},D] current visual tokens."
                )
            return visual_tokens
        if self.action_effect_enabled and self.action_effect_visual_tokens_prepooled:
            if visual_tokens.ndim not in {3, 4}:
                raise ValueError(
                    "Pre-pooled visual tokens must have shape [B,Q,D] "
                    "or [B,H,Q,D]."
                )
            if visual_tokens.shape[-2] != self.action_effect_visual_queries:
                raise ValueError(
                    f"Expected {self.action_effect_visual_queries} pre-pooled visual "
                    f"tokens, got {visual_tokens.shape[-2]}."
                )
            return visual_tokens
        if self.visual_resampler is None:
            raise RuntimeError("Visual resampling requires action_effect_enabled=True.")
        return self.visual_resampler(visual_tokens)

    def encode_action(self, actions, current_visual_tokens=None):
        if self.action_representation_type == "single_view_edar_lite":
            if current_visual_tokens is None:
                raise ValueError("EDAR-lite encoding requires current DINOv3 tokens.")
            if self.edar_token_flow:
                _, token_latent = self.edar_encoder(
                    actions,
                    current_visual_tokens,
                    return_token_latent=True,
                )
                if tuple(token_latent.shape[1:]) != (4, 256):
                    raise ValueError("EDAR encoder must return token latents shaped [B,4,256].")
                return token_latent
            return self.edar_encoder(actions, current_visual_tokens)
        if self.action_effect_enabled and self.action_effect_condition_action_encoder:
            if current_visual_tokens is None:
                raise ValueError("Action-effect encoding requires current visual tokens.")
            return self.action_encoder(actions, current_visual_tokens)
        return self.action_encoder(actions)

    def encode_action_prefix_latents(self, actions, full_action_latent):
        if self.multi_horizon_visual_predictor is None:
            raise RuntimeError("Action prefix latents require multi-horizon prediction.")
        truncated_horizons = [
            horizon
            for horizon in self.action_effect_horizons
            if horizon < actions.shape[1]
        ]
        encoded_truncated = {}
        if truncated_horizons:
            prefix_batches = []
            for horizon in truncated_horizons:
                prefix_actions = actions.clone()
                prefix_actions[:, horizon:] = 0
                prefix_batches.append(prefix_actions)
            batch_size = actions.shape[0]
            stacked_prefixes = torch.stack(prefix_batches, dim=1)
            flat_prefixes = stacked_prefixes.flatten(0, 1)
            flat_latents = self.action_encoder(flat_prefixes)
            batched_latents = flat_latents.reshape(
                batch_size,
                len(truncated_horizons),
                -1,
            )
            encoded_truncated = {
                horizon: batched_latents[:, index]
                for index, horizon in enumerate(truncated_horizons)
            }

        prefix_latents = []
        for horizon in self.action_effect_horizons:
            if horizon >= actions.shape[1]:
                prefix_latents.append(full_action_latent)
            else:
                prefix_latents.append(encoded_truncated[horizon])
        return torch.stack(prefix_latents, dim=1)

    def predict_visual_horizons(
        self,
        actions,
        action_latent,
        current_visual_tokens,
        proprioception,
    ):
        if self.multi_horizon_visual_predictor is None:
            raise RuntimeError("Multi-horizon visual predictor is not enabled.")
        prefix_latents = self.encode_action_prefix_latents(
            actions,
            action_latent,
        )
        return self.multi_horizon_visual_predictor(
            actions=actions,
            prefix_latents=prefix_latents,
            current_visual_tokens=current_visual_tokens,
            proprioception=proprioception,
        )

    def encode_action_effect_state(self, proprioception):
        if self.action_effect_state_encoder is None:
            return None
        if proprioception is None:
            raise ValueError("Action-effect state conditioning requires proprioception.")
        return self.action_effect_state_encoder(proprioception)

    def decode_visual_delta(
        self,
        action_latent,
        current_visual_tokens,
        state_embedding=None,
    ):
        if self.action_effect_projector is None or self.visual_delta_decoder is None:
            raise RuntimeError("Visual-delta decoding requires action_effect_enabled=True.")
        effect_latent = self.project_action_effect(action_latent, state_embedding)
        return self.decode_projected_visual_delta(
            effect_latent,
            current_visual_tokens,
            state_embedding,
        )

    def project_action_effect(self, action_latent, state_embedding=None):
        if self.action_effect_projector is None:
            raise RuntimeError("Action-effect projection requires action_effect_enabled=True.")
        if isinstance(self.action_effect_projector, nn.Identity):
            return self.action_effect_projector(action_latent)
        return self.action_effect_projector(action_latent, state_embedding)

    def decode_projected_visual_delta(
        self,
        effect_latent,
        current_visual_tokens,
        state_embedding=None,
    ):
        if self.visual_delta_decoder is None:
            raise RuntimeError("Visual-delta decoding requires action_effect_enabled=True.")
        return self.visual_delta_decoder(
            effect_latent,
            current_visual_tokens,
            state_embedding,
        )
