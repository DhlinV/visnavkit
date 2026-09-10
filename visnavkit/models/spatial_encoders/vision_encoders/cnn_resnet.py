"""ResNet vision encoder."""

from .timm import TimmVisionEncoder


class ResNetEncoder(TimmVisionEncoder):
    """ResNet-18 with native activations and stride 8, 16, and 32 feature stages."""

    def __init__(self, backbone_name="resnet18", out_indices=(2, 3, 4), act_layer=None, **kwargs):
        super().__init__(backbone_name=backbone_name, out_indices=out_indices, act_layer=act_layer, **kwargs)
