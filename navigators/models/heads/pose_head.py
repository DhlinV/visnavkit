import torch
import torch.nn as nn

from navigators.models.layers.res_block import FusableResBlock


class PoseHead(nn.Module):
    """Per-frame speed regression head (SmoothL1), maskable for frames without a previous image."""

    def __init__(self, feat_size=512):
        super().__init__()
        self.speed_head = nn.Sequential(
            nn.Linear(feat_size, 32),
            FusableResBlock(32, 32),
            nn.Linear(32, 1),
        )
        self.loss_speed = nn.SmoothL1Loss(reduction="none")

    def forward(self, x):
        return self.speed_head(x)

    def get_losses(self, pred, gt, mask=None):
        mask = mask.reshape(-1, 1) if mask is not None else torch.ones_like(pred)
        loss = self.loss_speed(pred, gt.reshape(-1, 1)) * mask
        return loss.mean()
