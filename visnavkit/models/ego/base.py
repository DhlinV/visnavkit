"""Ego-status encoders map the robot's own state to per-frame tokens."""

import torch
import torch.nn as nn

__all__ = ["BaseEgoEncoder"]


class BaseEgoEncoder(nn.Module):
    """Shared ego-token contract, learned null token, and per-sample dropout.

    The ego status is a free-form ``(B, F, E)`` vector: the dataset decides what goes in it
    (speed, yaw rate, past odometry, ...) and the encoder only has to know its width. Tokens are
    per frame, so they join the vision tokens of the same frame before the temporal encoder.

    Args:
        feat_size: token width ``D``.
        num_tokens: ``G`` tokens emitted per frame (0 for policies without ego status).
        p_drop: training-time probability of replacing a sample's ego tokens with the null token,
            so the policy also works when the ego status is missing at deployment.
    """

    def __init__(self, feat_size: int, num_tokens: int = 1, p_drop: float = 0.0):
        super().__init__()
        if num_tokens < 0 or feat_size < 1:
            raise ValueError("num_tokens must be nonnegative and feat_size positive")
        if not 0 <= p_drop <= 1:
            raise ValueError("p_drop must be between 0 and 1")
        self.feat_size = feat_size
        self.num_tokens = num_tokens
        self.p_drop = p_drop
        if num_tokens:
            self.null_token = nn.Parameter(torch.zeros(1, 1, num_tokens, feat_size))
            nn.init.normal_(self.null_token, std=0.02)

    def encode(self, ego: torch.Tensor) -> torch.Tensor:
        """``(B, F, E)`` ego status -> ``(B, F, G, D)`` tokens."""
        raise NotImplementedError

    def example_input(self, batch_size: int, frames: int = 1, device=None) -> torch.Tensor | None:
        """Synthetic ego batch for shape checks and export tracing (None when no tokens)."""
        return None

    def forward(self, ego: torch.Tensor | None = None, batch_size: int | None = None, frames: int = 1) -> torch.Tensor:
        if self.num_tokens == 0:
            n = batch_size if ego is None else ego.shape[0]
            device = None if ego is None else ego.device
            return torch.zeros(n, frames, 0, self.feat_size, device=device)
        if ego is None:
            if batch_size is None:
                raise ValueError("batch_size is required when no ego status is supplied")
            return self.null_token.repeat(batch_size, frames, 1, 1)  # a copy, not a parameter view
        if ego.ndim != 3:
            raise ValueError(f"Ego status must be (B, F, E), got {tuple(ego.shape)}")
        tokens = self.encode(ego)
        if self.training and self.p_drop > 0:
            keep = torch.rand(tokens.shape[0], 1, 1, 1, device=tokens.device) >= self.p_drop
            tokens = torch.where(keep, tokens, self.null_token.to(tokens.dtype))
        return tokens
