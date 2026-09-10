"""FastViT vision encoder."""

from .timm import TimmVisionEncoder


class FastViTEncoder(TimmVisionEncoder):
    """FastViT-T12 feature pyramid with the export-friendly GELU activation."""

    def __init__(self, backbone_name="fastvit_t12", **kwargs):
        super().__init__(backbone_name=backbone_name, **kwargs)
