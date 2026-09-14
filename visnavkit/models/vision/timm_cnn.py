"""Convolutional and hybrid timm backbones exposed as feature pyramids."""

from functools import partial

import timm
import torch.nn as nn
from hydra.utils import instantiate

from visnavkit.models.vision.base import BaseVisionEncoder


class TimmCNNEncoder(BaseVisionEncoder):
    """Any ``features_only`` timm backbone: ResNet, EfficientNet, MobileNet, ConvNeXt, RegNet, FastViT, ...

    The last pyramid stage provides the patch map; its global average pool is the frame embedding.

    Args:
        backbone_name: timm model name.
        out_indices: pyramid stages to expose (negative indices count from the last stage).
        act_layer: ``gelu_tanh`` swaps activations for an export-friendly GELU; ``None`` keeps the
            backbone's native activations (required for pretrained CNN weights).
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        out_indices: tuple[int, ...] = (-3, -2, -1),
        act_layer: str | None = None,
        heads=None,
        **kwargs,
    ):
        out_indices = tuple(out_indices)
        if not out_indices:
            raise ValueError("out_indices must select at least one backbone feature stage")
        backbone_kwargs = dict(
            pretrained=pretrained,
            in_chans=3,
            num_classes=0,
            features_only=True,
            out_indices=out_indices,
        )
        if act_layer == "gelu_tanh":
            backbone_kwargs["act_layer"] = partial(nn.GELU, approximate="tanh")
        elif act_layer is not None:
            raise ValueError(f"Unsupported {act_layer=}")
        backbone = timm.create_model(backbone_name, **backbone_kwargs)
        embed_dims = backbone.feature_info.channels()
        super().__init__(
            backbone,
            embed_dims[-1],
            heads={name: instantiate(cfg, in_chs=embed_dims) for name, cfg in (heads or {}).items()},
            **kwargs,
        )
        self.backbone_name = backbone_name
        self.embed_dims = embed_dims

    def _encode(self, x):
        pyramid = self.backbone(x)
        last = pyramid[-1]
        return last.mean(dim=(2, 3)), last, pyramid
