"""DINOv3 vision encoder."""

from .vit_dino import DinoEncoder


class DINOv3Encoder(DinoEncoder):
    """Frozen DINOv3 ViT-S/16 by default."""

    def __init__(self, backbone_name="vit_small_patch16_dinov3", **kwargs):
        super().__init__(backbone_name=backbone_name, **kwargs)
