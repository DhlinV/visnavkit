"""Diffusion Transformer denoiser: adaLN-Zero self-attention over anchor tokens, cross-attention to context."""

import torch
import torch.nn as nn

from visnavkit.models.layers.embeddings import SinusoidalTimeEmbedding

__all__ = ["DiTDenoiser"]


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class DiTBlock(nn.Module):
    def __init__(self, hidden: int, num_heads: int, cond_dim: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.cross = nn.MultiheadAttention(
            hidden, num_heads, dropout=dropout, batch_first=True, kdim=cond_dim, vdim=cond_dim
        )
        self.norm3 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_ratio * hidden), nn.GELU(approximate="tanh"), nn.Linear(mlp_ratio * hidden, hidden)
        )
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 9 * hidden))
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

    def forward(self, x, c, tokens):
        s1, sc1, g1, s2, sc2, g2, s3, sc3, g3 = self.adaln(c)[:, None].chunk(9, dim=-1)
        h = modulate(self.norm1(x), s1, sc1)
        x = x + g1 * self.attn(h, h, h, need_weights=False)[0]
        h = modulate(self.norm2(x), s2, sc2)
        x = x + g2 * self.cross(h, tokens, tokens, need_weights=False)[0]
        h = modulate(self.norm3(x), s3, sc3)
        return x + g3 * self.mlp(h)


class DiTDenoiser(nn.Module):
    """Each of the ``T`` anchor steps is a token; time and pooled conditioning drive adaLN."""

    def __init__(
        self,
        action_dim: int,
        num_pts: int,
        cond_dim: int,
        hidden: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        time_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.action_dim, self.num_pts = action_dim, num_pts
        self.in_proj = nn.Linear(action_dim, hidden)
        self.pos = nn.Parameter(torch.zeros(1, num_pts, hidden))
        nn.init.normal_(self.pos, std=0.02)
        self.time_embed = SinusoidalTimeEmbedding(time_dim, hidden)
        self.cond_proj = nn.Linear(cond_dim, hidden)
        self.blocks = nn.ModuleList([DiTBlock(hidden, num_heads, cond_dim, dropout=dropout) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.final_adaln = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        self.out_proj = nn.Linear(hidden, action_dim)
        nn.init.zeros_(self.final_adaln[-1].weight)
        nn.init.zeros_(self.final_adaln[-1].bias)

    def forward(self, x_t, t, cond, tokens=None):
        if tokens is None:
            tokens = cond[:, None]
        c = self.time_embed(t) + self.cond_proj(cond)
        x = self.in_proj(x_t) + self.pos
        for block in self.blocks:
            x = block(x, c, tokens)
        shift, scale = self.final_adaln(c)[:, None].chunk(2, dim=-1)
        return self.out_proj(modulate(self.final_norm(x), shift, scale))
