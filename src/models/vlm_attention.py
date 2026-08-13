import math

import torch
import torch.nn as nn


class LastLayerControlledHiddenAttention(nn.Module):
    """Reweight VLM layer hidden states using the final layer as controller.

    The module returns the same tuple structure as HF hidden_states. Its initial
    behavior is identity because alpha starts at 0.
    """

    def __init__(self, hidden_size, init_alpha=0.0):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, hidden_states):
        if hidden_states is None:
            return None
        if not isinstance(hidden_states, (tuple, list)):
            return hidden_states
        if len(hidden_states) <= 1:
            return hidden_states

        layers = list(hidden_states)
        controller = layers[-1].mean(dim=1)
        layer_summaries = torch.stack([h.mean(dim=1) for h in layers], dim=1)

        q = self.query(controller).unsqueeze(1)
        k = self.key(layer_summaries)
        scores = (q * k).sum(dim=-1) / math.sqrt(k.shape[-1])
        weights = torch.softmax(scores.float(), dim=-1).to(dtype=layers[-1].dtype)

        # Uniform attention should preserve the original scale.
        num_layers = len(layers)
        scales = 1.0 + self.alpha.to(dtype=weights.dtype) * (weights * num_layers - 1.0)
        return tuple(h * scales[:, i].view(-1, 1, 1).to(dtype=h.dtype) for i, h in enumerate(layers))
