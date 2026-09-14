"""Shared frame-pair preparation, token projection, and auxiliary supervision for vision encoders."""

from collections.abc import Mapping

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from visnavkit.models.layers.res_block import FusableResBlock
from visnavkit.models.outputs import VisionOutput
from visnavkit.models.vision.pose_head import PoseHead
from visnavkit.utils.logger import get_logger

logger = get_logger(__name__)

TOKEN_MODES = ("global", "patch", "fused")
PAIR_MODES = ("early", "late")


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
    """Encode ``(N, 6, H, W)`` previous/current RGB pairs into ``(N, K, feat_size)`` tokens plus speed.

    Subclasses wrap a backbone and implement ``_encode`` (global embedding, patch map, pyramid).

    Args:
        backbone: the image backbone module.
        embed_dim: channel width of the backbone's global embedding and patch map.
        feat_size: token width consumed by the temporal encoder and action decoder.
        token_mode: ``global`` (K=1), ``patch`` (K=gh*gw pooled patches), or ``fused`` (1 + gh*gw).
        patch_grid: ``(gh, gw)`` adaptive-pooling grid for patch tokens.
        pair_mode: ``early`` feeds the 6-channel stack to the backbone; ``late`` runs a shared
            3-channel backbone on both frames and concatenates their embeddings.
        freeze_backbone: keep backbone weights fixed and in eval mode.
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        *,
        feat_size: int = 256,
        token_mode: str = "global",
        patch_grid: tuple[int, int] = (4, 4),
        pair_mode: str = "early",
        img_embed_size: int = 1024,
        img_embed_drop: float = 0.1,
        p_drop_prev_img: float = 0.1,
        neck_cfg: Mapping | None = None,
        heads: Mapping[str, nn.Module] | None = None,
        loss_pose_weight: float = 1.0,
        freeze_backbone: bool = False,
        weights: str | None = None,
    ):
        super().__init__()
        if token_mode not in TOKEN_MODES:
            raise ValueError(f"token_mode must be one of {TOKEN_MODES}, got {token_mode!r}")
        if pair_mode not in PAIR_MODES:
            raise ValueError(f"pair_mode must be one of {PAIR_MODES}, got {pair_mode!r}")
        if not 0 <= p_drop_prev_img <= 1:
            raise ValueError("p_drop_prev_img must be between 0 and 1")
        patch_grid = tuple(int(v) for v in patch_grid)
        if len(patch_grid) != 2 or min(patch_grid) < 1:
            raise ValueError("patch_grid must be two positive integers")
        neck_cfg = {"n_res_blocks": 2, "dropout": 0.1, **(neck_cfg or {}), "dim": feat_size}
        if not isinstance(neck_cfg["n_res_blocks"], int) or neck_cfg["n_res_blocks"] < 0:
            raise ValueError("neck_cfg.n_res_blocks must be a nonnegative integer")

        self.backbone = backbone
        self.feat_size = feat_size
        self.token_mode = token_mode
        self.patch_grid = patch_grid
        self.pair_mode = pair_mode
        self.p_drop_prev_img = p_drop_prev_img
        self.loss_pose_weight = loss_pose_weight
        self.freeze_backbone = freeze_backbone
        data_config = timm.data.resolve_model_data_config(backbone)
        self.normalize_frame_transform = torchvision.transforms.Normalize(data_config["mean"], data_config["std"])

        pair_dim = embed_dim * (2 if pair_mode == "late" else 1)
        self.final_linear = nn.Linear(pair_dim, img_embed_size)
        self.embed_norm = nn.BatchNorm1d(img_embed_size)
        self.dropout = nn.Dropout(img_embed_drop)
        self.action_neck = build_neck(img_embed_size, neck_cfg)
        self.feat_norm = nn.LayerNorm(feat_size)
        self.pose_neck = build_neck(img_embed_size, neck_cfg)
        self.pose_head = PoseHead(feat_size=feat_size)
        self.heads = nn.ModuleDict(heads or {})
        if token_mode != "global":
            self.patch_proj = nn.Sequential(nn.Linear(pair_dim, feat_size), nn.LayerNorm(feat_size))
            self.patch_pos = nn.Parameter(torch.zeros(1, patch_grid[0] * patch_grid[1], feat_size))
            nn.init.normal_(self.patch_pos, std=0.02)
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
        if weights is not None:
            logger.info(f"Loading vision weights from {weights}")
            self.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))

    @property
    def num_tokens(self) -> int:
        patches = self.patch_grid[0] * self.patch_grid[1]
        return {"global": 1, "patch": patches, "fused": 1 + patches}[self.token_mode]

    @property
    def backbone_in_chans(self) -> int:
        return 6 if self.pair_mode == "early" else 3

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def prepare_for_export(self, image_size: tuple[int, int]):
        """Return an encoder ready for tracing at ``(height, width)``; the default needs no changes."""
        return self

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        """Normalized frames ``(N, C, H, W)`` -> global ``(N, E)``, patch map ``(N, E, h, w)`` or None, pyramid."""
        raise NotImplementedError

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
        pair = torch.cat((self.normalize_frame_transform(prev), self.normalize_frame_transform(x[:, 3:])), dim=1)
        return pair, prev_img_mask

    def _pool_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """Average-pool the patch map to ``patch_grid``; resample when the grid does not divide it (ONNX-safe)."""
        h, w = patches.shape[-2:]
        if h % self.patch_grid[0] == 0 and w % self.patch_grid[1] == 0:
            return F.adaptive_avg_pool2d(patches, self.patch_grid)
        return F.interpolate(patches, size=self.patch_grid, mode="bilinear", align_corners=False)

    def _run_backbone(self, pair: torch.Tensor):
        if self.pair_mode == "early":
            return self._encode(pair)
        n = pair.shape[0]
        embedding, patches, pyramid = self._encode(torch.cat((pair[:, :3], pair[:, 3:]), dim=0))
        embedding = torch.cat((embedding[:n], embedding[n:]), dim=1)
        patches = None if patches is None else torch.cat((patches[:n], patches[n:]), dim=1)
        return embedding, patches, pyramid

    def forward(self, x: torch.Tensor, export_heads: list[str] | tuple[str, ...] | None = None) -> VisionOutput:
        input_hw = x.shape[-2:]
        pair, prev_img_mask = self._prepare_frames(x)
        embedding, patches, pyramid = self._run_backbone(pair)
        embedding = self.dropout(self.embed_norm(self.final_linear(embedding)))
        global_token = self.feat_norm(self.action_neck(embedding))[:, None]
        pose = self.pose_head(self.pose_neck(embedding))
        if self.token_mode == "global":
            tokens = global_token
        else:
            if patches is None:
                raise ValueError(f"{type(self).__name__} provides no patch map; use token_mode=global")
            grid = self._pool_patches(patches).flatten(2).transpose(1, 2)
            patch_tokens = self.patch_proj(grid) + self.patch_pos
            tokens = patch_tokens if self.token_mode == "patch" else torch.cat((global_token, patch_tokens), dim=1)
        head_outputs = {}
        for name in self.heads if export_heads is None else export_heads:
            head_outputs.update(self.heads[name](pyramid, out_size=input_hw, export=export_heads is not None))
        return VisionOutput(tokens=tokens, pose=pose, prev_img_mask=prev_img_mask, heads=head_outputs)

    def get_head_output_names(self, head_names: list[str] | tuple[str, ...]) -> list[str]:
        return [output_name for name in head_names for output_name in self.heads[name].output_names]

    def get_losses(self, preds: VisionOutput, targets):
        pose_loss = self.pose_head.get_losses(preds.pose, targets["frame_speeds"], preds.prev_img_mask)
        loss_dict = {"pose": pose_loss.detach()}
        total_loss = self.loss_pose_weight * pose_loss
        for name, head in self.heads.items():
            loss = head.get_losses(preds.heads, targets)
            total_loss = total_loss + head.loss_weight * loss
            loss_dict[name] = loss.detach()
        loss_dict["total"] = total_loss
        return loss_dict
