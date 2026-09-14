"""Flat MLP denoiser over the whole trajectory (NoMaD-adjacent, cheapest option)."""

import torch
import torch.nn as nn

from visnavkit.models.layers.embeddings import SinusoidalTimeEmbedding
from visnavkit.models.layers.mlp import build_mlp

__all__ = ["MLPDenoiser"]


class MLPDenoiser(nn.Module):
    def __init__(
        self,
        action_dim: int,
        num_pts: int,
        cond_dim: int,
        hidden: int = 512,
        layers: int = 3,
        time_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.action_dim, self.num_pts = action_dim, num_pts
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.mlp = build_mlp(
            num_pts * action_dim + cond_dim + time_dim, hidden, num_pts * action_dim, layers=layers, dropout=dropout
        )

    def forward(self, x_t, t, cond, tokens=None):
        out = self.mlp(torch.cat([x_t.flatten(1), cond, self.time_embed(t)], dim=-1))
        return out.reshape(x_t.shape[0], self.num_pts, self.action_dim)
