import torch
import torch.nn as nn

from visnavkit.models.layers.mlp import build_mlp

from .base import BaseGoalEncoder

__all__ = ["InstructionGoalEncoder"]


class InstructionGoalEncoder(BaseGoalEncoder):
    """Language instruction given as a precomputed text embedding ``(N, E)``.

    Text encoders (CLIP, T5, ...) run offline, so the policy has no tokenizer dependency; the
    embedding is projected to ``num_tokens`` goal tokens.
    """

    goal_type = "instruction"

    def __init__(
        self,
        feat_size: int,
        embed_dim: int = 512,
        hidden: int = 512,
        num_tokens: int = 1,
        p_drop: float = 0.0,
        **kwargs,
    ):
        super().__init__(feat_size, num_tokens=num_tokens, p_drop=p_drop, **kwargs)
        self.embed_dim = embed_dim
        self.mlp = build_mlp(embed_dim, hidden, num_tokens * feat_size, layers=2)
        self.norm = nn.LayerNorm(feat_size)

    def encode(self, goal, observation=None):
        if goal.ndim != 2 or goal.shape[1] != self.embed_dim:
            raise ValueError(f"Instruction embeddings must have shape (N, {self.embed_dim}), got {tuple(goal.shape)}")
        return self.norm(self.mlp(goal.float()).reshape(goal.shape[0], self.num_tokens, self.feat_size))

    def example_input(self, batch_size, device=None, image_hw=(64, 64)):
        return torch.randn(batch_size, self.embed_dim, device=device)
