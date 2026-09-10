"""Shared single-frame DINO backbones for previous/current RGB pairs."""

from collections.abc import Mapping
from copy import deepcopy

import timm
import torch
from timm.layers import resample_abs_pos_embed

from .base import BaseVisionEncoder


class DinoEncoder(BaseVisionEncoder):
    """Encode each frame with shared DINO weights, then concatenate pooled embeddings.

    DINO keeps its pretrained three-channel patch embedding. Inputs are padded to a
    patch multiple; spatial supervision heads require a feature-pyramid encoder.
    """

    def __init__(
        self,
        backbone_name="vit_small_patch16_dinov3",
        in_chans=6,
        img_embed_size=1024,
        img_embed_drop=0,
        p_drop_prev_img=0.1,
        pretrained=True,
        freeze_backbone=True,
        neck_cfg: Mapping | None = None,
        heads: Mapping | None = None,
        weights: str | None = None,
        loss_pose_weight=1,
    ):
        if in_chans != 6:
            raise ValueError(f"DinoEncoder expects a stacked RGB pair (in_chans=6), got {in_chans}")
        if heads:
            raise ValueError("DinoEncoder does not support extra heads (no multi-scale features)")
        backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=True,
            dynamic_img_pad=True,
        )
        super().__init__(
            backbone,
            2 * backbone.num_features,
            img_embed_size=img_embed_size,
            img_embed_drop=img_embed_drop,
            p_drop_prev_img=p_drop_prev_img,
            neck_cfg=neck_cfg,
            heads=None,
            loss_pose_weight=loss_pose_weight,
        )
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
        self._load_weights(weights)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    @torch.no_grad()
    def prepare_for_export(self, image_size: tuple[int, int]):
        """Return a fixed-resolution eval copy with DINOv2 positions precomputed.

        Use timm's original antialiased interpolation before ONNX tracing, where
        that operation is unsupported. The training encoder and its checkpoint
        remain unchanged. DINOv3 uses rotary positions and needs no adaptation.
        """
        if len(image_size) != 2 or any(not isinstance(size, int) or size <= 0 for size in image_size):
            raise ValueError("image_size must contain positive integer height and width")
        encoder = deepcopy(self).eval()
        backbone = encoder.backbone
        if getattr(backbone, "pos_embed", None) is None:
            return encoder
        patch_embed = backbone.patch_embed
        grid_size = patch_embed.dynamic_feat_size(image_size)
        positions = resample_abs_pos_embed(
            backbone.pos_embed,
            new_size=grid_size,
            old_size=patch_embed.grid_size,
            num_prefix_tokens=0 if backbone.no_embed_class else backbone.num_prefix_tokens,
        )
        backbone.pos_embed = torch.nn.Parameter(positions, requires_grad=backbone.pos_embed.requires_grad)
        patch_embed.img_size = tuple(image_size)
        patch_embed.grid_size = grid_size
        patch_embed.num_patches = grid_size[0] * grid_size[1]
        patch_embed.strict_img_size = True
        patch_embed.flatten = True
        backbone.dynamic_img_size = False
        return encoder

    def _encode_frames(self, x):
        batch_size = x.shape[0]
        features = self.backbone(torch.cat((x[:, :3], x[:, 3:]), dim=0))
        return torch.cat((features[:batch_size], features[batch_size:]), dim=1), []
