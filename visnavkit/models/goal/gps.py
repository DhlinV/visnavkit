import torch
import torch.nn as nn

from visnavkit.models.layers.mlp import build_mlp
from visnavkit.models.normalization import Normalizer

from .base import BaseGoalEncoder

__all__ = ["GpsGoalEncoder"]


class GpsGoalEncoder(BaseGoalEncoder):
    """Local goal offset ``(N, in_dim)`` in metres in the current ego frame, as a GPS waypoint gives it.

    ``point`` carries the same information as (distance, cos, sin); this keeps the raw Cartesian
    offset, which is what LogoNav and FlowPilot condition on. ``normalizer`` decides how those
    metres are scaled — ``scale`` needs no corpus, ``meanstd`` uses one once it exists.
    """

    goal_type = "gps"
    per_frame = True

    def __init__(
        self,
        feat_size: int,
        in_dim: int = 2,
        hidden: int = 256,
        normalizer: Normalizer | None = None,
        p_drop: float = 0.0,
        **kwargs,
    ):
        super().__init__(feat_size, num_tokens=1, p_drop=p_drop, **kwargs)
        if in_dim < 2:
            raise ValueError("in_dim must be at least 2 (x, y)")
        self.in_dim = in_dim
        self.normalizer = normalizer or Normalizer(mode="scale", scale=20.0)
        self.mlp = build_mlp(in_dim, hidden, feat_size, layers=2)
        self.norm = nn.LayerNorm(feat_size)

    def encode(self, goal, observation=None):
        if goal.ndim != 2 or goal.shape[1] != self.in_dim:
            raise ValueError(f"GPS goals must have shape (N, {self.in_dim}), got {tuple(goal.shape)}")
        return self.norm(self.mlp(self.normalizer.normalize(goal.float())))[:, None]

    def example_input(self, batch_size, device=None, image_hw=(64, 64)):
        goal = torch.zeros(batch_size, self.in_dim, device=device)
        goal[:, 0] = 5.0
        return goal
