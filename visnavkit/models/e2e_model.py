import torch
import torch.nn as nn
from omegaconf import DictConfig

from visnavkit.utils.logger import get_logger

logger = get_logger(__name__)


def _detach(v):
    return v.detach() if isinstance(v, torch.Tensor) else v


class E2EModel(nn.Module):
    """Vision encoder + temporal action decoder. Pass ``modules`` as
    ``{vision_encoder, action_decoder}`` (Hydra config). An optional ``route_encoder``
    module conditions the action decoder on a goal/route patch."""

    def __init__(
        self,
        modules: dict[str, nn.Module],
        feat_size: int,
        loss_cfg: DictConfig,
        export_cfg: DictConfig,
        frozen_modules: list[str] | None = None,
        trainable_modules: list[str] | None = None,
        route_drop_p: float = 0.0,
    ):
        super().__init__()
        self.loss_cfg = loss_cfg
        self.feat_size = feat_size
        self.vision_encoder = modules["vision_encoder"]
        self.action_decoder = modules["action_decoder"]
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

    def _configure_trainable_modules(self, frozen_modules: list[str]) -> None:
        if self._trainable_modules:
            logger.warning(
                f"Modules {self._trainable_modules} were set to be trainable. Freezing all other modules "
                f"regardless of explicitly set frozen modules: {frozen_modules}"
            )
            for name, p in self.named_parameters():
                # fuzzy name matching (eg vision_encoder.head.x matches vision_encoder.head.x.fpn, etc)
                p.requires_grad = any(trainable in name for trainable in self._trainable_modules)
        else:
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

    def _append_route_embeddings(self, decoder_input: torch.Tensor, route_patch: torch.Tensor | None) -> torch.Tensor:
        if self.route_encoder is None:
            return decoder_input
        if route_patch is None:
            raise ValueError("route_patch is required when model.modules.route_encoder is configured.")

        if self.training and self.route_drop_p > 0:
            # per-sample (whole clip), not per-frame: partial zeroing would leak the route back
            # in through temporal fusion. Zero the PATCH, not the embedding: the encoder maps a
            # blank (all-background) patch to its own "no route" code, matching deploy-time absence
            keep = torch.rand(route_patch.shape[0], 1, 1, 1, device=route_patch.device) >= self.route_drop_p
            route_patch = route_patch * keep
        route_embeddings = self.route_encoder(route_patch)
        return torch.cat([decoder_input, route_embeddings], dim=-1)

    def forward(self, x, fb=None, route_patch=None):
        # Training: full sequence of frames
        if fb is None:
            B, F, C, H, W = x.shape
            vision_outputs = self.vision_encoder(x.view(B * F, C, H, W))
            decoder_input = vision_outputs["feat_out"].view(B, F, self.feat_size)
            decoder_input = self._append_route_embeddings(decoder_input, route_patch)
            action_outputs = self.action_decoder(decoder_input)
            return dict(vision=vision_outputs, action=action_outputs)

        # Export: single frame + feature buffer
        B = x.shape[0]
        vision_outputs = self.vision_encoder(x, export_heads=self.export_heads)
        current_token = vision_outputs["feat_out"].reshape(B, 1, self.feat_size)
        decoder_input = torch.cat([fb[:, self.feature_idxs, :], current_token], dim=1)
        decoder_input = self._append_route_embeddings(decoder_input, route_patch)
        plan_output = self.action_decoder(decoder_input)["plan"]["plans"]
        head_outputs = tuple(
            vision_outputs[name] for name in self.vision_encoder.get_head_output_names(self.export_heads)
        )
        return (plan_output, vision_outputs["pose"], current_token.flatten(1), *head_outputs)

    def get_export_output_names(self) -> list[str]:
        output_names = ["plan", "pose", "feat_out"]
        output_names.extend(self.vision_encoder.get_head_output_names(self.export_heads))
        return output_names

    def get_losses(self, preds, targets):
        vision_loss_dict = self.vision_encoder.get_losses(preds["vision"], targets["vision"])
        action_loss_dict, action_loss_debug = self.action_decoder.get_losses(preds["action"], targets["action"])
        total_loss = (
            self.loss_cfg.vision_weight * vision_loss_dict["total"]
            + self.loss_cfg.action_weight * action_loss_dict["total"]
        )
        loss_dict = dict(loss=total_loss)
        loss_dict.update({f"vision_{k}": _detach(v) for k, v in vision_loss_dict.items()})
        loss_dict.update({f"action_{k}": _detach(v) for k, v in action_loss_dict.items()})
        loss_debug = dict(action_loss_debug=action_loss_debug)
        return loss_dict, loss_debug
