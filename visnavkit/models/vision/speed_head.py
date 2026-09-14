import torch.nn as nn

from visnavkit.models.layers.res_block import FusableResBlock

__all__ = ["SpeedHead"]


class SpeedHead(nn.Module):
    """Optional per-frame speed regression (SmoothL1).

    Only some recipes supervise speed from the image; it is an auxiliary training signal, not
    part of the policy contract, so recipes opt in with ``vision_encoder.speed_head=true``.
    """

    def __init__(self, feat_size=512):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(feat_size, 32), FusableResBlock(32, 32), nn.Linear(32, 1))
        self.criterion = nn.SmoothL1Loss()

    def forward(self, x):
        return self.head(x)

    def get_losses(self, pred, gt):
        return self.criterion(pred, gt.reshape(-1, 1))
