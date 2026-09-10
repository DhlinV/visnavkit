"""DINOv2 vision encoder."""

from .vit_dino import DinoEncoder


class DINOv2Encoder(DinoEncoder):
    """Frozen DINOv2 ViT-S/14 by default, with automatic patch padding."""

    def __init__(self, backbone_name="vit_small_patch14_dinov2", **kwargs):
        super().__init__(backbone_name=backbone_name, **kwargs)
