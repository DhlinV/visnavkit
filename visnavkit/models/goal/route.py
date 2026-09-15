import torch
import torch.nn as nn

from .base import BaseGoalEncoder

__all__ = ["RouteImageGoalEncoder", "load_route_encoder", "route_cnn"]


def route_cnn(in_chans: int, channels) -> nn.Sequential:
    """Stride-2 conv stack pooled to ``(N, channels[-1])``; ``RouteAutoencoder.encoder`` is this same module."""
    layers, width = [], in_chans
    for out in channels:
        layers += [nn.Conv2d(width, out, 3, stride=2, padding=1), nn.BatchNorm2d(out), nn.ReLU(inplace=True)]
        width = out
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())


def load_route_encoder(weights) -> dict:
    """The encoder of a ``visnavkit-train-route`` checkpoint (or plain state dict), keyed like ``route_cnn``."""
    state = torch.load(weights, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    prefix = "model.encoder."
    encoder = {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}
    if not encoder:
        raise ValueError(f"{weights} holds no '{prefix}*' tensors; expected a route autoencoder checkpoint")
    return encoder


class RouteImageGoalEncoder(BaseGoalEncoder):
    """Rendered route / map patch ``(N, C, H, W)`` centred on the ego pose, encoded by a small CNN.

    A blank patch maps to the encoder's own "no route" code, matching deploy-time absence.
    ``weights`` initialises the CNN from a route autoencoder trained with ``visnavkit-train-route``.
    """

    goal_type = "route_image"

    def __init__(
        self,
        feat_size: int,
        in_chans: int = 3,
        channels=(32, 64, 128, 256),
        p_drop: float = 0.0,
        weights=None,
        **kwargs,
    ):
        super().__init__(feat_size, num_tokens=1, p_drop=p_drop, **kwargs)
        self.in_chans = in_chans
        self.cnn = route_cnn(in_chans, channels)
        self.proj = nn.Sequential(nn.Linear(channels[-1], feat_size), nn.LayerNorm(feat_size))
        if weights is not None:
            self.cnn.load_state_dict(load_route_encoder(weights))

    def encode(self, goal, observation=None):
        if goal.ndim != 4 or goal.shape[1] != self.in_chans:
            raise ValueError(f"Route images must have shape (N, {self.in_chans}, H, W), got {tuple(goal.shape)}")
        x = goal.float().div(255.0) if not goal.is_floating_point() else goal
        return self.proj(self.cnn(x))[:, None]

    def example_input(self, batch_size, device=None, image_hw=(64, 64)):
        return torch.rand(batch_size, self.in_chans, *image_hw, device=device)
