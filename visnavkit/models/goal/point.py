import torch
import torch.nn as nn

from visnavkit.models.layers.mlp import build_mlp

from .base import BaseGoalEncoder

__all__ = ["PointGoalEncoder"]


class PointGoalEncoder(BaseGoalEncoder):
    """Point goal ``(N, 3) = (distance_m, cos(bearing), sin(bearing))`` in the current ego frame.

    Distance is clipped to ``max_distance`` and scaled to [0, 1] before the MLP (S2E-style).
    """

    goal_type = "point"
    per_frame = True

    def __init__(self, feat_size: int, hidden: int = 256, max_distance: float = 20.0, p_drop: float = 0.0, **kwargs):
        super().__init__(feat_size, num_tokens=1, p_drop=p_drop, **kwargs)
        if max_distance <= 0:
            raise ValueError("max_distance must be positive")
        self.max_distance = max_distance
        self.mlp = build_mlp(3, hidden, feat_size, layers=2)
        self.norm = nn.LayerNorm(feat_size)

    def encode(self, goal, observation=None):
        if goal.ndim != 2 or goal.shape[1] != 3:
            raise ValueError(f"Point goals must have shape (N, 3), got {tuple(goal.shape)}")
        goal = goal.float()
        features = torch.cat([(goal[:, :1] / self.max_distance).clamp(0, 1), goal[:, 1:]], dim=1)
        return self.norm(self.mlp(features))[:, None]

    def example_input(self, batch_size, device=None, image_hw=(64, 64)):
        return torch.tensor([[5.0, 1.0, 0.0]], device=device).expand(batch_size, -1).clone()
