"""Shared transformer construction and temporal reductions."""

import torch
import torch.nn as nn

__all__ = ["BaseTemporalEncoder"]


class BaseTemporalEncoder(nn.Module):
    """Encode ``[batch, frames, features]`` and optionally reduce the frame axis.

    A single transformer layer stays unwrapped to preserve checkpoint parameter
    names. ``num_layers=0`` preserves the legacy per-frame encoder behavior.
    """

    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        ff_dim=2048,
        dropout=0.1,
        seq_len=10,
        reduction="last",
        mask_p=0.5,
        route_embed_dim=0,
        num_layers=1,
    ):
        super().__init__()
        if seq_len < 1:
            raise ValueError("seq_len must be positive")
        if num_layers < 0:
            raise ValueError("num_layers must be nonnegative")
        if reduction not in {"last", "avg", "sum", "none"}:
            raise ValueError(f"Unknown reduction value {reduction=}")
        if not 0 <= mask_p <= 1:
            raise ValueError("mask_p must be between 0 and 1")
        if embed_dim < 1 or route_embed_dim < 0:
            raise ValueError("embed_dim must be positive and route_embed_dim nonnegative")

        self.seq_len = seq_len
        self.reduction = reduction
        self.mask_p = mask_p
        self.model_dim = embed_dim + route_embed_dim

        if num_layers == 0:
            self.tformer = None
            return

        if num_heads < 1 or self.model_dim % num_heads:
            raise ValueError("embed_dim + route_embed_dim must be divisible by a positive num_heads")

        self.pos_embedding = nn.Embedding(seq_len, self.model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.tformer = (
            layer if num_layers == 1 else nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        )

    def _attention_mask(self, seq_len, device):
        return None

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, frames, features], got shape {tuple(x.shape)}")
        batch, frames, features = x.shape
        if frames == 0:
            raise ValueError("Temporal encoder requires at least one frame")
        if features != self.model_dim:
            raise ValueError(f"Expected {self.model_dim} features, got {features}")

        if self.tformer is not None:
            if frames > self.seq_len:
                raise ValueError(f"Sequence length {frames} exceeds positional embedding capacity {self.seq_len}")
            positions = self.pos_embedding(torch.arange(frames, device=x.device))[None, :, :]
            x = x + positions.expand(batch, frames, features)
            # Both a layer and a stack accept the attention mask as their second argument.
            x = self.tformer(x, self._attention_mask(frames, x.device))

        if self.reduction == "last":
            return x[:, -1, :]
        if self.reduction == "avg":
            return x.mean(dim=1)
        if self.reduction == "sum":
            return x.sum(dim=1)
        return x
