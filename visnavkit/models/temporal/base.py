"""Shared transformer construction and temporal reductions over ``(B, F, K, D)`` tokens."""

import torch
import torch.nn as nn

__all__ = ["BaseTemporalEncoder"]

REDUCTIONS = ("last", "avg", "sum", "none")


class BaseTemporalEncoder(nn.Module):
    """Mix per-frame tokens across time and optionally reduce the frame axis.

    Input ``(B, F, K, D)``: F frames of K tokens each. Output ``(B, F', K, D)`` with F' = F for
    ``reduction=none`` and F' = 1 otherwise. Attention runs over the F*K token sequence; frame
    position embeddings are shared by the K tokens of a frame and the frame-level attention
    mask is expanded per token. ``num_layers=0`` applies only the reduction.
    """

    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        ff_dim=2048,
        dropout=0.1,
        seq_len=10,
        reduction="last",
        mask_p=0.0,
        num_layers=1,
    ):
        super().__init__()
        if seq_len < 1:
            raise ValueError("seq_len must be positive")
        if num_layers < 0:
            raise ValueError("num_layers must be nonnegative")
        if reduction not in REDUCTIONS:
            raise ValueError(f"Unknown reduction value {reduction=}")
        if not 0 <= mask_p <= 1:
            raise ValueError("mask_p must be between 0 and 1")
        if embed_dim < 1:
            raise ValueError("embed_dim must be positive")

        self.seq_len = seq_len
        self.reduction = reduction
        self.mask_p = mask_p
        self.embed_dim = embed_dim
        if num_layers == 0:
            self.tformer = None
            return
        if num_heads < 1 or embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by a positive num_heads")
        self.pos_embedding = nn.Embedding(seq_len, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
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

    def _attention_mask(self, seq_len: int, device) -> torch.Tensor | None:
        """Frame-level ``(F, F)`` additive mask or None for unrestricted attention."""
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [batch, frames, tokens, features], got shape {tuple(x.shape)}")
        batch, frames, tokens, features = x.shape
        if frames == 0:
            raise ValueError("Temporal encoder requires at least one frame")
        if features != self.embed_dim:
            raise ValueError(f"Expected {self.embed_dim} features, got {features}")

        if self.tformer is not None:
            if frames > self.seq_len:
                raise ValueError(f"Sequence length {frames} exceeds positional embedding capacity {self.seq_len}")
            positions = self.pos_embedding(torch.arange(frames, device=x.device))[None, :, None, :]
            x = (x + positions).reshape(batch, frames * tokens, features)
            mask = self._attention_mask(frames, x.device)
            if mask is not None and tokens > 1:
                # expand (not repeat_interleave): traced shapes must not become CPU index tensors
                mask = (
                    mask[:, None, :, None]
                    .expand(frames, tokens, frames, tokens)
                    .reshape(frames * tokens, frames * tokens)
                )
            x = self.tformer(x, mask).reshape(batch, frames, tokens, features)

        if self.reduction == "last":
            return x[:, -1:]
        if self.reduction == "avg":
            return x.mean(dim=1, keepdim=True)
        if self.reduction == "sum":
            return x.sum(dim=1, keepdim=True)
        return x
