import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18PatchEncoder(nn.Module):
    def __init__(self, out_dim, max_views=4, trainable=True):
        super().__init__()
        backbone = resnet18(weights=None)
        self.stem = nn.Sequential(*list(backbone.children())[:-2])
        self.camera_embed = nn.Parameter(torch.zeros(max_views, 512))
        self.proj = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, out_dim),
        )
        self.max_views = int(max_views)
        if not trainable:
            self.requires_grad_(False)

    def forward(self, images):
        if images is None:
            return None
        if images.ndim == 4:
            images = images.unsqueeze(1)
        if images.ndim != 5:
            raise ValueError(f"Expected bridge images with shape [B,V,3,H,W], got {tuple(images.shape)}")

        bsz, num_views, channels, height, width = images.shape
        if channels != 3:
            raise ValueError(f"Expected RGB bridge images, got {channels} channels")
        if num_views > self.max_views:
            raise ValueError(f"Got {num_views} views, but max_views={self.max_views}")

        dtype = next(self.stem.parameters()).dtype
        images = images.to(dtype=dtype)
        feats = self.stem(images.reshape(bsz * num_views, channels, height, width))
        feats = feats.flatten(2).transpose(1, 2)
        feats = feats.reshape(bsz, num_views, feats.shape[1], feats.shape[2])
        cam = self.camera_embed[:num_views].to(device=feats.device, dtype=feats.dtype)
        feats = feats + cam.view(1, num_views, 1, -1)
        feats = feats.flatten(1, 2)
        return self.proj(feats)


class GatedCrossAttentionLatentBridge(nn.Module):
    def __init__(
        self,
        hidden_size,
        action_dim,
        num_history,
        num_heads=8,
        max_views=4,
        visual_trainable=True,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_history = int(num_history)
        self.visual_encoder = ResNet18PatchEncoder(
            out_dim=hidden_size,
            max_views=max_views,
            trainable=visual_trainable,
        )
        self.query_norm = nn.LayerNorm(hidden_size)
        self.visual_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        state_in = max(1, self.num_history) * action_dim
        self.state_mlp = nn.Sequential(
            nn.LayerNorm(state_in),
            nn.Linear(state_in, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        self.visual_gate = nn.Parameter(torch.tensor(0.0))
        self.state_gate = nn.Parameter(torch.tensor(0.0))

    def _selected_indices(self, num_hidden_states, policy_depth, layer_indices):
        if num_hidden_states <= 0:
            return []
        if layer_indices is not None:
            return [int(i) for i in layer_indices]
        depth = min(int(policy_depth), num_hidden_states)
        return list(range(num_hidden_states - depth, num_hidden_states))

    def forward(self, hidden_states, bridge_images=None, proprioception=None, policy_depth=24, layer_indices=None):
        if bridge_images is None or hidden_states is None:
            return hidden_states

        is_tuple = isinstance(hidden_states, (tuple, list))
        layers = list(hidden_states) if is_tuple else [hidden_states]
        if not layers:
            return hidden_states

        z_ref = layers[-1]
        z_norm = self.query_norm(z_ref)
        visual_tokens = self.visual_encoder(bridge_images)
        visual_tokens = self.visual_norm(visual_tokens).to(dtype=z_norm.dtype)
        delta_visual, _ = self.cross_attn(query=z_norm, key=visual_tokens, value=visual_tokens)

        delta_state = torch.zeros_like(delta_visual)
        if proprioception is not None:
            state = proprioception.to(device=z_ref.device, dtype=z_ref.dtype)
            if state.ndim == 2:
                state = state.unsqueeze(1)
            if state.shape[1] < self.num_history:
                pad = state[:, :1].expand(-1, self.num_history - state.shape[1], -1)
                state = torch.cat([pad, state], dim=1)
            state = state[:, -self.num_history :, :].flatten(1)
            gamma, beta = self.state_mlp(state).chunk(2, dim=-1)
            delta_state = gamma.unsqueeze(1) * z_norm + beta.unsqueeze(1)

        delta = torch.tanh(self.visual_gate).to(dtype=z_ref.dtype) * delta_visual
        delta = delta + torch.tanh(self.state_gate).to(dtype=z_ref.dtype) * delta_state

        indices = self._selected_indices(len(layers), policy_depth, layer_indices)
        for idx in indices:
            layers[idx] = layers[idx] + delta.to(dtype=layers[idx].dtype)

        return tuple(layers) if is_tuple else layers[0]
