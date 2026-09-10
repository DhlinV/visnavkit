"""Shared frame preprocessing, projections, supervision, and export contract."""

from collections.abc import Mapping

import timm
import torch
import torch.nn as nn
import torchvision

from visnavkit.models.heads.pose_head import PoseHead
from visnavkit.models.layers.res_block import FusableResBlock
from visnavkit.utils.logger import get_logger

logger = get_logger(__name__)


def build_neck(in_dim: int, cfg: Mapping) -> nn.Sequential:
    """Project an image embedding through residual blocks to the decoder feature size."""
    return nn.Sequential(
        nn.Dropout(p=cfg["dropout"]),
        nn.Linear(in_dim, cfg["dim"]),
        *[FusableResBlock(cfg["dim"], 2 * cfg["dim"]) for _ in range(cfg["n_res_blocks"])],
        nn.Linear(cfg["dim"], cfg["dim"]),
        nn.BatchNorm1d(cfg["dim"]),
    )


class BaseVisionEncoder(nn.Module):
    """Encode ``(B, 6, H, W)`` previous/current RGB pairs into pose and feature tokens.

    Subclasses implement ``_encode_frames`` for normalized pairs. Shared modules stay
    directly on the encoder so existing checkpoint parameter names remain valid.
    """

    def __init__(
        self,
        backbone: nn.Module,
        in_features: int,
        *,
        img_embed_size: int,
        img_embed_drop: float,
        p_drop_prev_img: float,
        neck_cfg: Mapping | None,
        heads: Mapping[str, nn.Module] | None,
        loss_pose_weight: float,
    ):
        super().__init__()
        if not 0 <= p_drop_prev_img <= 1:
            raise ValueError("p_drop_prev_img must be between 0 and 1")
        neck_cfg = {"dim": 256, "n_res_blocks": 2, "dropout": 0.1, **(neck_cfg or {})}
        if not isinstance(neck_cfg["dim"], int) or neck_cfg["dim"] <= 0:
            raise ValueError("neck_cfg.dim must be a positive integer")
        if not isinstance(neck_cfg["n_res_blocks"], int) or neck_cfg["n_res_blocks"] < 0:
            raise ValueError("neck_cfg.n_res_blocks must be a nonnegative integer")

        self.backbone = backbone
        data_config = timm.data.resolve_model_data_config(backbone)
        self.normalize_frame_transform = torchvision.transforms.Normalize(
            mean=data_config["mean"], std=data_config["std"]
        )
        self.p_drop_prev_img = p_drop_prev_img
        self.loss_pose_weight = loss_pose_weight
        self.heads = nn.ModuleDict(heads or {})
        self.img_embed_size = img_embed_size
        self.final_linear = nn.Linear(in_features, img_embed_size)
        self.embed_norm = nn.BatchNorm1d(img_embed_size)
        self.dropout = nn.Dropout(img_embed_drop)
        self.action_neck = build_neck(img_embed_size, neck_cfg)
        self.feat_norm = nn.LayerNorm(neck_cfg["dim"])
        self.pose_neck = build_neck(img_embed_size, neck_cfg)
        self.pose_head = PoseHead(feat_size=neck_cfg["dim"])

    def _load_weights(self, weights: str | None):
        if weights is not None:
            logger.info(f"Loading vision weights from {weights}")
            self.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))

    def prepare_for_export(self, image_size: tuple[int, int]):
        """Return an encoder ready for export at the given ``(height, width)``."""
        return self

    def _prepare_frames(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != 6:
            raise ValueError(f"Expected a (B, 6, H, W) previous/current RGB pair, got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError("Frame pairs must be floating-point tensors in the [0, 1] range")
        prev_img_mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        prev = x[:, :3]
        if self.training and self.p_drop_prev_img > 0:
            prev_img_mask = torch.rand(x.shape[0], device=x.device) >= self.p_drop_prev_img
            prev = torch.where(prev_img_mask[:, None, None, None], prev, torch.zeros_like(prev))
        return torch.cat(
            (self.normalize_frame_transform(prev), self.normalize_frame_transform(x[:, 3:])), dim=1
        ), prev_img_mask

    def _encode_frames(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, export_heads: list[str] | tuple[str, ...] | None = None):
        input_hw = x.shape[-2:]
        x, prev_img_mask = self._prepare_frames(x)
        x, backbone_feats = self._encode_frames(x)
        x = self.dropout(self.embed_norm(self.final_linear(x)))
        feat_out = self.feat_norm(self.action_neck(x))
        pose = self.pose_head(self.pose_neck(x))
        result = {
            "pose": pose,
            "feat_out": feat_out,
            "prev_img_mask": prev_img_mask,
        }
        for name in self.heads if export_heads is None else export_heads:
            result.update(self.heads[name](backbone_feats, out_size=input_hw, export=export_heads is not None))
        return result

    def get_head_output_names(self, head_names: list[str] | tuple[str, ...]) -> list[str]:
        return [output_name for name in head_names for output_name in self.heads[name].output_names]

    def get_losses(self, preds, targets):
        pose_loss = self.pose_head.get_losses(preds["pose"], targets["frame_speeds"], preds["prev_img_mask"])
        loss_dict = {"pose": pose_loss.detach()}
        total_loss = self.loss_pose_weight * pose_loss
        for name, head in self.heads.items():
            loss = head.get_losses(preds, targets)
            total_loss = total_loss + head.loss_weight * loss
            loss_dict[name] = loss.detach()
        loss_dict["total"] = total_loss
        return loss_dict
