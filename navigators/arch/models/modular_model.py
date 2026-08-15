from typing import Dict, List

import torch
import torch.nn as nn
from omegaconf import DictConfig

from navigators.utils.logger import get_logger

logger = get_logger(__name__)


class ModularModel(nn.Module):
    """Policy wrapped with vision. Pass ``modules`` as ``{vision, policy}`` (Hydra config).
    An optional ``route_encoder`` module conditions the policy on a goal/route patch."""

    def __init__(
        self,
        modules: Dict[str, nn.Module],
        feat_size,
        loss_cfg: DictConfig,
        export_cfg: DictConfig,
        frozen_modules: List[str] | None = None,
        trainable_modules: List[str] | None = None,
        route_drop_p: float = 0.0,
    ):
        super().__init__()
        self.loss_cfg = loss_cfg
        self.feat_size = feat_size
        self.vision_model = modules["vision"]
        self.policy_model = modules["policy"]
        self.route_encoder = modules.get("route_encoder")
        self.route_drop_p = route_drop_p

        step = export_cfg.seq_step
        seq_len = export_cfg.seq_len
        self.feature_idxs = torch.tensor(range(-(seq_len - 1) * step, 0, step))
        self.export_heads = []
        self._trainable_modules = trainable_modules or []

        self._configure_trainable_modules(frozen_modules or [])
        if self._trainable_modules:
            self.train(True)

    def _configure_trainable_modules(self, frozen_modules: List[str]) -> None:
        if self._trainable_modules:
            logger.warning(
                f"Modules {self._trainable_modules} were set to be trainable. Freezing all other modules \
                           regardless of explicitly set frozen modules: {frozen_modules}"
            )
            for name, p in self.named_parameters():
                # fuzzy name matching (eg vision_model.head.x matches vision_model.head.x.fpn, etc)
                if any(trainable in name for trainable in self._trainable_modules):
                    p.requires_grad = True
                else:
                    p.requires_grad = False

        else:
            # Backwards compatibility: freeze any layers that need freezing
            for frozen_module in frozen_modules:
                mod = getattr(self, frozen_module, None)
                if mod is None:
                    logger.warning(f"Module {frozen_module} not found. It will not be frozen.")
                    continue
                logger.info(f"Freezing {frozen_module}")
                for p in mod.parameters():
                    p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._trainable_modules:
            for module in self.modules():
                module.eval()
            for name, module in self.named_modules():
                if any(trainable in name for trainable in self._trainable_modules):
                    module.train(True)
        return self

    def _append_route_embeddings(self, policy_input: torch.Tensor, route_patch: torch.Tensor | None) -> torch.Tensor:
        if self.route_encoder is None:
            return policy_input
        if route_patch is None:
            raise ValueError("route_patch is required when model.modules.route_encoder is configured.")

        if self.training and self.route_drop_p > 0:
            # per-sample (whole clip), not per-frame: partial zeroing would leak the route back
            # in through temporal fusion. Zero the PATCH, not the embedding: the encoder maps a
            # blank (all-background) patch to its own "no route" code, matching deploy-time absence
            keep = torch.rand(route_patch.shape[0], 1, 1, 1, device=route_patch.device) >= self.route_drop_p
            route_patch = route_patch * keep
        route_embeddings = self.route_encoder(route_patch)
        return torch.cat([policy_input, route_embeddings], dim=-1)

    def forward(self, x, fb=None, route_patch=None):
        # Training: full sequence of frames
        if fb is None:
            B, F, C, H, W = x.shape
            x = x.view((B * F, C, H, W))
            vision_outputs = self.vision_model(x)
            vision_feats = vision_outputs["feat_out"]
            policy_input = vision_feats.view((B, F, self.feat_size))
            policy_input = self._append_route_embeddings(policy_input, route_patch)
            policy_outputs = self.policy_model(policy_input)
            return dict(vision=vision_outputs, policy=policy_outputs)

        # Export: single frame + feature buffer
        B, C, H, W = x.shape
        vision_outputs = self.vision_model(x, export_heads=self.export_heads)
        gathered_fb = fb[:, self.feature_idxs, :]
        vision_feats = vision_outputs["feat_out"]
        current_token = vision_feats.reshape(B, 1, self.feat_size)
        policy_input = torch.cat([gathered_fb, current_token], dim=1)
        policy_input = self._append_route_embeddings(policy_input, route_patch)
        plan_output = self.policy_model(policy_input)["plan"]["plans"]
        head_outputs = tuple(
            vision_outputs[name] for name in self.vision_model.get_head_output_names(self.export_heads)
        )
        return (plan_output, vision_outputs["pose"], current_token.flatten(1), *head_outputs)

    def get_export_output_names(self) -> list[str]:
        output_names = ["plan", "pose", "feat_out"]
        output_names.extend(self.vision_model.get_head_output_names(self.export_heads))
        return output_names

    def get_losses(self, preds, targets):
        vision_loss_dict = self.vision_model.get_losses(preds["vision"], targets["vision"])
        policy_loss_dict, policy_loss_debug = self.policy_model.get_losses(preds["policy"], targets["policy"])
        total_loss = (
            self.loss_cfg.vision_weight * vision_loss_dict["total"]
            + self.loss_cfg.policy_weight * policy_loss_dict["total"]
        )
        loss_dict = dict(loss=total_loss)
        maybe_detach = lambda v: v.detach() if isinstance(v, torch.Tensor) else v  # noqa: E731
        loss_dict.update({f"vision_{k}": maybe_detach(v) for k, v in vision_loss_dict.items()})
        loss_dict.update({f"policy_{k}": maybe_detach(v) for k, v in policy_loss_dict.items()})
        loss_debug = dict(policy_loss_debug=policy_loss_debug)
        return loss_dict, loss_debug
