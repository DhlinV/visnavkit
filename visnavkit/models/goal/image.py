import timm
import torch
import torch.nn as nn
import torchvision

from .base import BaseGoalEncoder

__all__ = ["ImageGoalEncoder"]


class ImageGoalEncoder(BaseGoalEncoder):
    """Goal image ``(N, 3, H, W)`` (uint8 or [0, 1] float) encoded by its own timm backbone.

    ``stack_observation=True`` concatenates the current observation with the goal image
    (6-channel input) as in GNM / ViNT / NoMaD's goal encoders.
    """

    goal_type = "image"

    def __init__(
        self,
        feat_size: int,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        stack_observation: bool = False,
        freeze_backbone: bool = False,
        p_drop: float = 0.0,
        **kwargs,
    ):
        super().__init__(feat_size, num_tokens=1, p_drop=p_drop, **kwargs)
        self.stack_observation = stack_observation
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, in_chans=6 if stack_observation else 3
        )
        data_config = timm.data.resolve_model_data_config(self.backbone)
        self.normalize = torchvision.transforms.Normalize(data_config["mean"], data_config["std"])
        self.proj = nn.Sequential(nn.Linear(self.backbone.num_features, feat_size), nn.LayerNorm(feat_size))
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _to_float(self, image):
        return image.float().div(255.0) if not image.is_floating_point() else image

    def encode(self, goal, observation=None):
        if goal.ndim != 4 or goal.shape[1] != 3:
            raise ValueError(f"Goal images must have shape (N, 3, H, W), got {tuple(goal.shape)}")
        x = self.normalize(self._to_float(goal))
        if self.stack_observation:
            if observation is None:
                raise ValueError("stack_observation=True requires the current observation frame")
            x = torch.cat([self.normalize(self._to_float(observation[:, 3:6])), x], dim=1)
        return self.proj(self.backbone(x))[:, None]

    def example_input(self, batch_size, device=None, image_hw=(64, 64)):
        return torch.rand(batch_size, 3, *image_hw, device=device)
