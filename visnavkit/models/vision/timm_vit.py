"""Vision transformers from timm: DINOv2, DINOv3, ViT/DeiT, EVA-02, SigLIP, CLIP, ..."""

from copy import deepcopy

import timm
import torch
from timm.layers import resample_abs_pos_embed

from visnavkit.models.vision.base import BaseVisionEncoder


class TimmViTEncoder(BaseVisionEncoder):
    """Any timm ViT with ``forward_features``: pooled/CLS token as the frame embedding, patch tokens as the map.

    Inputs are padded to a patch multiple, so the pretrained three-channel patch embedding is
    used unchanged.
    """

    def __init__(
        self,
        backbone_name: str = "vit_small_patch14_dinov2",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        heads=None,
        **kwargs,
    ):
        if heads:
            raise ValueError("TimmViTEncoder does not support extra spatial heads (no feature pyramid)")
        backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=3,
            dynamic_img_size=True,
            dynamic_img_pad=True,
        )
        super().__init__(backbone, backbone.num_features, freeze_backbone=freeze_backbone, **kwargs)
        self.backbone_name = backbone_name

    @torch.no_grad()
    def prepare_for_export(self, image_size: tuple[int, int]):
        """Return a fixed-resolution eval copy with absolute position embeddings precomputed.

        timm's antialiased interpolation is unsupported by ONNX tracing; rotary models need nothing.
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

    def _encode(self, x):
        features = self.backbone.forward_features(x)
        embedding = self.backbone.forward_head(features, pre_logits=True)
        patches = features[:, self.backbone.num_prefix_tokens :]
        gh, gw = self.backbone.patch_embed.dynamic_feat_size(tuple(x.shape[-2:]))
        patches = patches.transpose(1, 2).reshape(x.shape[0], -1, gh, gw)
        return embedding, patches, []
