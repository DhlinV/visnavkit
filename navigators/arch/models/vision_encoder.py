import logging
from functools import partial

import timm
import torch
import torch.nn as nn
import torchvision
from hydra.utils import instantiate
from omegaconf import DictConfig

from navigators.arch.modules.plan_head import PoseHead
from navigators.arch.modules.res_block import FusableResBlock
from navigators.utils.logger import get_logger

logger = get_logger(__name__)
logging.getLogger("timm").setLevel(logging.ERROR)


def _neck(in_dim: int, cfg: DictConfig) -> nn.Sequential:
    """Dropout -> Linear -> ResBlocks -> Linear -> BN projection from image embedding to feat dim."""
    return nn.Sequential(
        nn.Dropout(p=cfg.dropout),
        nn.Linear(in_dim, cfg.dim),
        *[FusableResBlock(cfg.dim, 2 * cfg.dim) for _ in range(cfg.n_res_blocks)],
        nn.Linear(cfg.dim, cfg.dim),
        nn.BatchNorm1d(cfg.dim),
    )


class VisionEncoder(nn.Module):
    """Per-frame encoder: timm backbone over a stacked prev+cur RGB pair -> ``feat_out`` token
    for the action decoder, a pose (speed) head, and optional extra supervision heads."""

    def __init__(
        self,
        backbone_name="fastvit_t12",
        in_chans=6,
        img_embed_size=2048,
        img_embed_drop=0,
        p_drop_prev_img=0.1,
        pretrained=True,
        neck_cfg: DictConfig | None = None,
        heads: DictConfig = DictConfig({}),
        weights: str | None = None,
        loss_pose_weight=1,
    ):
        super().__init__()

        self.loss_pose_weight = loss_pose_weight

        # augs
        self.p_drop_prev_img = p_drop_prev_img

        # backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
            global_pool="",
            act_layer=partial(nn.GELU, approximate="tanh"),
            features_only=True,
            out_indices=(1, 2, 3),
        )
        data_config = timm.data.resolve_model_data_config(self.backbone)
        self.normalize_frame_transform = torchvision.transforms.Normalize(
            mean=data_config["mean"], std=data_config["std"]
        )
        self.embed_dims = self.backbone.feature_info.channels()  # ty:ignore[call-non-callable]
        self.heads = nn.ModuleDict({name: instantiate(cfg, in_chs=self.embed_dims) for name, cfg in heads.items()})

        # non multi-scale image embedding
        self.img_embed_size = img_embed_size
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.final_linear = nn.Linear(self.embed_dims[-1], self.img_embed_size)
        self.embed_norm = nn.BatchNorm1d(self.img_embed_size)
        self.dropout = nn.Dropout(img_embed_drop)

        # two necks off the shared embedding: one feeds the action decoder, one the pose head
        self.action_neck = _neck(self.img_embed_size, neck_cfg)
        self.feat_norm = nn.LayerNorm(neck_cfg.dim)
        self.pose_neck = _neck(self.img_embed_size, neck_cfg)
        self.pose_head = PoseHead(feat_size=neck_cfg.dim)

        if weights is not None:
            logger.info(f"Loading vision weights from {weights}")
            state_dict = torch.load(weights)
            self.load_state_dict(state_dict)

    def forward(self, x, export_heads: list[str] | tuple[str, ...] | None = None):
        input_hw = x.shape[-2:]
        x = x.clone()  # this is needed for avoiding in-place normalization

        # randomly drop previous image to discourage shortcut learning
        bs = x.shape[0]
        prev_img_mask = torch.ones(bs, device=x.device) == 1
        if self.training:
            should_mask = torch.rand(bs, device=x.device) < self.p_drop_prev_img
            x[should_mask, 0:3] = 0
            prev_img_mask = ~should_mask  # true for images that HAVE a previous image, false for black

        x[:, 0:3] = self.normalize_frame_transform(x[:, 0:3])
        x[:, 3:6] = self.normalize_frame_transform(x[:, 3:6])

        p3, p4, p5 = self.backbone(x)
        backbone_feats = [p3, p4, p5]

        # non multi-scale features
        x = self.gap(p5).flatten(1)
        x = self.final_linear(x)
        x = self.embed_norm(x)
        x = self.dropout(x)

        feat_out = self.feat_norm(self.action_neck(x))
        pose = self.pose_head(self.pose_neck(x))

        res = dict(
            pose=pose,
            feat_out=feat_out,
            prev_img_mask=prev_img_mask,
        )

        # additional heads, typically used just for training supervision or introspection
        heads = self.heads.keys() if export_heads is None else export_heads
        for name in heads:
            res.update(self.heads[name](backbone_feats, out_size=input_hw, export=export_heads is not None))

        return res

    def get_head_output_names(self, head_names: list[str] | tuple[str, ...]) -> list[str]:
        return [output_name for name in head_names for output_name in self.heads[name].output_names]

    def get_losses(self, preds, targets):
        loss_dict = dict()
        pose_loss = self.pose_head.get_losses(preds["pose"], targets["frame_speeds"], preds["prev_img_mask"])
        loss_dict["pose"] = pose_loss.detach()
        total_loss = self.loss_pose_weight * pose_loss
        for name, head in self.heads.items():
            loss = head.get_losses(preds, targets)
            total_loss = total_loss + head.loss_weight * loss
            loss_dict[name] = loss.detach()
        loss_dict["total"] = total_loss
        return loss_dict


if __name__ == "__main__":
    # forward smoke test: model/base.yaml vision_encoder kwargs, downscaled frame size
    neck_cfg = DictConfig({"n_res_blocks": 2, "dim": 256, "dropout": 0.1})
    model = VisionEncoder(
        backbone_name="fastvit_t8", img_embed_size=1024, img_embed_drop=0.1, pretrained=False, neck_cfg=neck_cfg
    ).eval()
    x = torch.rand(2, 6, 240, 320)  # (B, prev+cur RGB, H, W)
    with torch.no_grad():
        out = model(x)
    for name, val in out.items():
        print(f"{name}: {tuple(val.shape)}")
