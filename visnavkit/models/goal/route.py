import torch
import torch.nn as nn

from .base import BaseGoalEncoder

__all__ = ["RouteImageGoalEncoder"]


class RouteImageGoalEncoder(BaseGoalEncoder):
    """Rendered route / map patch ``(N, C, H, W)`` centred on the ego pose, encoded by a small CNN.

    A blank patch maps to the encoder's own "no route" code, matching deploy-time absence.
    """

    goal_type = "route_image"

    def __init__(self, feat_size: int, in_chans: int = 3, channels=(32, 64, 128, 256), p_drop: float = 0.0, **kwargs):
        super().__init__(feat_size, num_tokens=1, p_drop=p_drop, **kwargs)
        layers = []
        width = in_chans
        for out in channels:
            layers += [nn.Conv2d(width, out, 3, stride=2, padding=1), nn.BatchNorm2d(out), nn.ReLU(inplace=True)]
            width = out
        self.in_chans = in_chans
        self.cnn = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.proj = nn.Sequential(nn.Linear(width, feat_size), nn.LayerNorm(feat_size))

    def encode(self, goal, observation=None):
        if goal.ndim != 4 or goal.shape[1] != self.in_chans:
            raise ValueError(f"Route images must have shape (N, {self.in_chans}, H, W), got {tuple(goal.shape)}")
        x = goal.float().div(255.0) if not goal.is_floating_point() else goal
        return self.proj(self.cnn(x))[:, None]

    def example_input(self, batch_size, device=None, image_hw=(64, 64)):
        return torch.rand(batch_size, self.in_chans, *image_hw, device=device)
