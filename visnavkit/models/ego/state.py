import torch
import torch.nn as nn

from visnavkit.models.layers.mlp import build_mlp

from .base import BaseEgoEncoder

__all__ = ["EgoStateEncoder"]


class EgoStateEncoder(BaseEgoEncoder):
    """Free-form ego state ``(B, F, in_dim)`` projected to ``num_tokens`` tokens per frame.

    ``in_dim`` has to match what the dataset packs into its ``ego`` array (speed, yaw rate,
    past odometry, ...); the encoder makes no assumption about the channel meanings.
    """

    def __init__(
        self,
        feat_size: int,
        in_dim: int = 1,
        hidden: int = 128,
        num_tokens: int = 1,
        p_drop: float = 0.0,
        **kwargs,
    ):
        super().__init__(feat_size, num_tokens=num_tokens, p_drop=p_drop, **kwargs)
        if in_dim < 1:
            raise ValueError("in_dim must be positive")
        self.in_dim = in_dim
        self.mlp = build_mlp(in_dim, hidden, num_tokens * feat_size, layers=2)
        self.norm = nn.LayerNorm(feat_size)

    def encode(self, ego):
        if ego.shape[-1] != self.in_dim:
            raise ValueError(f"Ego status must be (B, F, {self.in_dim}), got {tuple(ego.shape)}")
        b, f = ego.shape[:2]
        return self.norm(self.mlp(ego.float()).reshape(b, f, self.num_tokens, self.feat_size))

    def example_input(self, batch_size, frames=1, device=None):
        return torch.zeros(batch_size, frames, self.in_dim, device=device)
