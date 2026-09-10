"""Feature-pyramid adapter for timm backbones."""

from collections.abc import Mapping
from functools import partial

import timm
import torch.nn as nn
from hydra.utils import instantiate

from .base import BaseVisionEncoder


class TimmVisionEncoder(BaseVisionEncoder):
    """Encode stacked RGB pairs with a timm feature pyramid and optional spatial heads."""

    def __init__(
        self,
        backbone_name="fastvit_t12",
        in_chans=6,
        img_embed_size=2048,
        img_embed_drop=0,
        p_drop_prev_img=0.1,
        pretrained=True,
        neck_cfg: Mapping | None = None,
        heads: Mapping | None = None,
        weights: str | None = None,
        loss_pose_weight=1,
        out_indices=(1, 2, 3),
        act_layer: str | None = "gelu_tanh",
    ):
        if in_chans != 6:
            raise ValueError(f"TimmVisionEncoder expects a stacked RGB pair (in_chans=6), got {in_chans}")
        out_indices = tuple(out_indices)
        if not out_indices:
            raise ValueError("out_indices must select at least one backbone feature stage")
        backbone_kwargs = dict(
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
            global_pool="",
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
            img_embed_size=img_embed_size,
            img_embed_drop=img_embed_drop,
            p_drop_prev_img=p_drop_prev_img,
            neck_cfg=neck_cfg,
            heads={name: instantiate(cfg, in_chs=embed_dims) for name, cfg in (heads or {}).items()},
            loss_pose_weight=loss_pose_weight,
        )
        self.embed_dims = embed_dims
        self.gap = nn.AdaptiveAvgPool2d(1)
        self._load_weights(weights)

    def _encode_frames(self, x):
        features = self.backbone(x)
        return self.gap(features[-1]).flatten(1), features
