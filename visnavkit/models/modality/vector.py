import torch
import torch.nn as nn

from visnavkit.models.layers.mlp import build_mlp

from .base import BaseModalityEncoder

__all__ = ["VectorEncoder"]


class VectorEncoder(BaseModalityEncoder):
    """A free-form per-frame vector ``(B, F, in_dim)`` projected to tokens.

    The default reads the ``ego`` batch key, but ``key`` points it at any per-frame vector the
    dataset provides (IMU window, wheel odometry, a learned embedding). The encoder makes no
    assumption about what the channels mean; only the width has to match.
    """

    def __init__(
        self,
        feat_size: int,
        key: str = "ego",
        in_dim: int = 1,
        hidden: int = 128,
        num_tokens: int = 1,
        p_drop: float = 0.0,
        **kwargs,
    ):
        super().__init__(feat_size, num_tokens=num_tokens, p_drop=p_drop, **kwargs)
        if in_dim < 1:
            raise ValueError("in_dim must be positive")
        self.input_names = (key,)
        self.in_dim = in_dim
        self.mlp = build_mlp(in_dim, hidden, num_tokens * feat_size, layers=2)
        self.norm = nn.LayerNorm(feat_size)

    def encode(self, values, *, image_hw=None):
        if values.ndim != 3 or values.shape[-1] != self.in_dim:
            raise ValueError(f"{self.input_names[0]} must be (B, F, {self.in_dim}), got {tuple(values.shape)}")
        b, f = values.shape[:2]
        return self.norm(self.mlp(values.float()).reshape(b, f, self.num_tokens, self.feat_size))

    def example_inputs(self, batch_size, frames=1, image_hw=(64, 64), device=None):
        return (torch.zeros(batch_size, frames, self.in_dim, device=device),)
