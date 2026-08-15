import logging

import timm
import torch
import torch.nn as nn
import torchvision
from omegaconf import DictConfig

from navigators.models.encoders.vision_encoder import _neck
from navigators.models.heads.pose_head import PoseHead
from navigators.utils.logger import get_logger

logger = get_logger(__name__)
logging.getLogger("timm").setLevel(logging.ERROR)


class DinoEncoder(nn.Module):
    """DINOv2/DINOv3 ViT encoder. The prev and cur frames of the stacked pair are encoded
    separately by the shared backbone (preserving the 3-channel pretrained patch embed),
    pooled embeddings are concatenated, then the same neck/pose tail as VisionEncoder.

    Select the version via ``backbone_name``, e.g. ``vit_small_patch14_dinov2`` or
    ``vit_small_patch16_dinov3`` (any timm dinov2/dinov3 variant works; patch14 inputs are
    padded to a patch multiple automatically). Extra supervision heads are unsupported —
    a ViT exposes no multi-scale feature pyramid.
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
        neck_cfg: DictConfig | None = None,
        heads: DictConfig = DictConfig({}),
        weights: str | None = None,
        loss_pose_weight=1,
    ):
        super().__init__()
        if in_chans != 6:
            raise ValueError(f"DinoEncoder expects a stacked prev+cur RGB pair (in_chans=6), got {in_chans}")
        if heads:
            raise ValueError("DinoEncoder does not support extra heads (no multi-scale features)")

        self.loss_pose_weight = loss_pose_weight
        self.p_drop_prev_img = p_drop_prev_img
        self.freeze_backbone = freeze_backbone

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=True,
            dynamic_img_pad=True,
        )
        data_config = timm.data.resolve_model_data_config(self.backbone)
        self.normalize_frame_transform = torchvision.transforms.Normalize(
            mean=data_config["mean"], std=data_config["std"]
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.heads = nn.ModuleDict({})

        # prev + cur pooled embeddings, concatenated
        self.img_embed_size = img_embed_size
        self.final_linear = nn.Linear(2 * self.backbone.num_features, self.img_embed_size)
        self.embed_norm = nn.BatchNorm1d(self.img_embed_size)
        self.dropout = nn.Dropout(img_embed_drop)

        self.action_neck = _neck(self.img_embed_size, neck_cfg)
        self.feat_norm = nn.LayerNorm(neck_cfg.dim)
        self.pose_neck = _neck(self.img_embed_size, neck_cfg)
        self.pose_head = PoseHead(feat_size=neck_cfg.dim)

        if weights is not None:
            logger.info(f"Loading vision weights from {weights}")
            state_dict = torch.load(weights)
            self.load_state_dict(state_dict)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x, export_heads: list[str] | tuple[str, ...] | None = None):
        x = x.clone()  # this is needed for avoiding in-place normalization

        # randomly drop previous image to discourage shortcut learning
        bs = x.shape[0]
        prev_img_mask = torch.ones(bs, device=x.device) == 1
        if self.training:
            should_mask = torch.rand(bs, device=x.device) < self.p_drop_prev_img
            x[should_mask, 0:3] = 0
            prev_img_mask = ~should_mask  # true for images that HAVE a previous image, false for black

        prev = self.normalize_frame_transform(x[:, 0:3])
        cur = self.normalize_frame_transform(x[:, 3:6])
        feats = self.backbone(torch.cat([prev, cur], dim=0))  # one pass, shared weights: (2B, D)
        x = torch.cat([feats[:bs], feats[bs:]], dim=1)  # (B, 2D)

        x = self.final_linear(x)
        x = self.embed_norm(x)
        x = self.dropout(x)

        feat_out = self.feat_norm(self.action_neck(x))
        pose = self.pose_head(self.pose_neck(x))

        return dict(pose=pose, feat_out=feat_out, prev_img_mask=prev_img_mask)

    def get_head_output_names(self, head_names: list[str] | tuple[str, ...]) -> list[str]:
        return [output_name for name in head_names for output_name in self.heads[name].output_names]

    def get_losses(self, preds, targets):
        pose_loss = self.pose_head.get_losses(preds["pose"], targets["frame_speeds"], preds["prev_img_mask"])
        return dict(pose=pose_loss.detach(), total=self.loss_pose_weight * pose_loss)


if __name__ == "__main__":
    # forward smoke test at the downscaled frame size (patch14 variants get padded automatically)
    neck_cfg = DictConfig({"n_res_blocks": 2, "dim": 256, "dropout": 0.1})
    for name in ("vit_small_patch14_dinov2", "vit_small_patch16_dinov3"):
        model = DinoEncoder(backbone_name=name, pretrained=False, neck_cfg=neck_cfg).eval()
        x = torch.rand(2, 6, 240, 320)
        with torch.no_grad():
            out = model(x)
        print(name, {k: tuple(v.shape) for k, v in out.items() if v.dim() > 1})
