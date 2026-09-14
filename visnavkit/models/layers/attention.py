import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """Pool ``(N, L, D)`` tokens into ``(N, D)`` with one learned query (export-friendly cross attention)."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        query = self.query.repeat(tokens.shape[0], 1, 1)
        pooled, _ = self.attn(query, tokens, tokens, need_weights=False)
        return self.norm(pooled[:, 0] + query[:, 0])
