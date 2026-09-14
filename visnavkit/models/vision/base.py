"""Shared frame preparation, token projection, and optional auxiliary heads for vision encoders."""

from collections.abc import Mapping

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from visnavkit.models.layers.res_block import FusableResBlock
from visnavkit.models.outputs import VisionOutput
from visnavkit.models.vision.speed_head import SpeedHead
from visnavkit.utils.logger import get_logger

logger = get_logger(__name__)

TOKEN_MODES = ("global", "patch", "fused")


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
    """Encode one ``(N, 3, H, W)`` RGB frame into ``(N, K, feat_size)`` tokens.

    Subclasses wrap a backbone and implement ``_encode`` (global embedding, patch map, pyramid).
    Motion is recovered by the temporal encoder across frames, not by stacking frames here.

    Args:
        backbone: the image backbone module.
        embed_dim: channel width of the backbone's global embedding and patch map.
        feat_size: token width consumed by the temporal encoder and action decoder.
        token_mode: ``global`` (K=1), ``patch`` (K=gh*gw pooled patches), or ``fused`` (1 + gh*gw).
        patch_grid: ``(gh, gw)`` adaptive-pooling grid for patch tokens.
        speed_head: add the auxiliary per-frame speed regression some recipes train with; it is
            a model-specific loss, so it is off unless a recipe asks for it.
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
        img_embed_size: int = 1024,
        img_embed_drop: float = 0.1,
        neck_cfg: Mapping | None = None,
        heads: Mapping[str, nn.Module] | None = None,
        speed_head: bool = False,
        loss_speed_weight: float = 1.0,
        freeze_backbone: bool = False,
        weights: str | None = None,
    ):
        super().__init__()
        if token_mode not in TOKEN_MODES:
            raise ValueError(f"token_mode must be one of {TOKEN_MODES}, got {token_mode!r}")
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
        self.loss_speed_weight = loss_speed_weight
        self.freeze_backbone = freeze_backbone
        data_config = timm.data.resolve_model_data_config(backbone)
        self.normalize_frame_transform = torchvision.transforms.Normalize(data_config["mean"], data_config["std"])

        self.final_linear = nn.Linear(embed_dim, img_embed_size)
        self.embed_norm = nn.BatchNorm1d(img_embed_size)
        self.dropout = nn.Dropout(img_embed_drop)
        self.action_neck = build_neck(img_embed_size, neck_cfg)
        self.feat_norm = nn.LayerNorm(feat_size)
        self.speed_neck = build_neck(img_embed_size, neck_cfg) if speed_head else None
        self.speed_head = SpeedHead(feat_size=feat_size) if speed_head else None
        self.heads = nn.ModuleDict(heads or {})
        if token_mode != "global":
            self.patch_proj = nn.Sequential(nn.Linear(embed_dim, feat_size), nn.LayerNorm(feat_size))
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
    def has_speed_head(self) -> bool:
        return self.speed_head is not None

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def prepare_for_export(self, image_size: tuple[int, int]):
        """Return an encoder ready for tracing at ``(height, width)``; the default needs no changes."""
        return self

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        """Normalized frames ``(N, 3, H, W)`` -> global ``(N, E)``, patch map ``(N, E, h, w)`` or None, pyramid."""
        raise NotImplementedError

    def _prepare_frames(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected a (N, 3, H, W) RGB frame, got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError("Frames must be floating-point tensors in the [0, 1] range")
        return self.normalize_frame_transform(x)

    def _pool_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """Average-pool the patch map to ``patch_grid``; resample when the grid does not divide it (ONNX-safe)."""
        h, w = patches.shape[-2:]
        if h % self.patch_grid[0] == 0 and w % self.patch_grid[1] == 0:
            return F.adaptive_avg_pool2d(patches, self.patch_grid)
        return F.interpolate(patches, size=self.patch_grid, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor, export_heads: list[str] | tuple[str, ...] | None = None) -> VisionOutput:
        input_hw = x.shape[-2:]
        embedding, patches, pyramid = self._encode(self._prepare_frames(x))
        embedding = self.dropout(self.embed_norm(self.final_linear(embedding)))
        global_token = self.feat_norm(self.action_neck(embedding))[:, None]
        speed = self.speed_head(self.speed_neck(embedding)) if self.has_speed_head else None
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
        return VisionOutput(tokens=tokens, speed=speed, heads=head_outputs)

    def get_head_output_names(self, head_names: list[str] | tuple[str, ...]) -> list[str]:
        return [output_name for name in head_names for output_name in self.heads[name].output_names]

    def get_losses(self, preds: VisionOutput, targets: Mapping) -> dict:
        """Auxiliary losses only; an encoder with no enabled head contributes nothing."""
        loss_dict, total = {}, None
        if self.has_speed_head:
            speed_loss = self.speed_head.get_losses(preds.speed, targets["frame_speeds"])
            loss_dict["speed"] = speed_loss.detach()
            total = self.loss_speed_weight * speed_loss
        for name, head in self.heads.items():
            loss = head.get_losses(preds.heads, targets)
            total = head.loss_weight * loss if total is None else total + head.loss_weight * loss
            loss_dict[name] = loss.detach()
        if total is None:
            return {}
        loss_dict["total"] = total
        return loss_dict
