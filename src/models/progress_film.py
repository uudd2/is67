import torch
import torch.nn as nn


class ProgressHead(nn.Module):
    def __init__(self, hidden_size, mlp_ratio=0.25):
        super().__init__()
        inner_size = max(32, int(hidden_size * mlp_ratio))
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, inner_size),
            nn.SiLU(),
            nn.Linear(inner_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, features):
        if features.ndim == 3:
            features = features.mean(dim=1)
        return self.net(features).squeeze(-1)


class ProgressFiLM(nn.Module):
    def __init__(self, hidden_size, mlp_hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, mlp_hidden_size),
            nn.SiLU(),
            nn.Linear(mlp_hidden_size, hidden_size * 2),
        )
        self._init_identity()

    def _init_identity(self):
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, features, progress):
        progress = progress.to(dtype=features.dtype)
        if progress.ndim == 1:
            progress = progress.unsqueeze(-1)
        gamma, beta = self.net(progress).chunk(2, dim=-1)
        if features.ndim == 3:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return (1.0 + gamma) * features + beta
