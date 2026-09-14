"""NavigationPolicy: frames -> vision tokens -> temporal context -> goal tokens -> action decoder."""

import torch
import torch.nn as nn
from omegaconf import DictConfig

from visnavkit.models.outputs import PolicyOutput, VisionOutput
from visnavkit.utils.logger import get_logger

logger = get_logger(__name__)


def _detach(v):
    return v.detach() if isinstance(v, torch.Tensor) else v


def _matches(name: str, prefixes: list[str]) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


class NavigationPolicy(nn.Module):
    """Compose the four stages; every stage exchanges ``feat_size``-wide tokens.

    Training: ``forward(frames, goal=None, noise=None)`` with frames ``(B, F, 6, H, W)``.
    Deployment: ``predict(frame, feature_buffer, goal=None, noise=None)`` encodes one frame and
    reuses past frame tokens from the buffer ``(B, history, K * D)``.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        temporal_encoder: nn.Module,
        goal_encoder: nn.Module,
        action_decoder: nn.Module,
        feat_size: int,
        loss_cfg: DictConfig,
        export_cfg: DictConfig,
        frozen_modules: list[str] | None = None,
        trainable_modules: list[str] | None = None,
    ):
        super().__init__()
        for name, module in (
            ("vision_encoder", vision_encoder),
            ("goal_encoder", goal_encoder),
            ("action_decoder", action_decoder),
        ):
            if getattr(module, "feat_size", feat_size) != feat_size:
                raise ValueError(f"{name}.feat_size must equal model.feat_size={feat_size}")
        if temporal_encoder.embed_dim != feat_size:
            raise ValueError(f"temporal_encoder.embed_dim must equal model.feat_size={feat_size}")
        self.vision_encoder = vision_encoder
        self.temporal_encoder = temporal_encoder
        self.goal_encoder = goal_encoder
        self.action_decoder = action_decoder
        self.feat_size = feat_size
        self.loss_cfg = loss_cfg

        step, seq_len = export_cfg.seq_step, export_cfg.seq_len
        if step < 1 or seq_len < 1:
            raise ValueError("export_cfg.seq_step and seq_len must be positive")
        self.register_buffer(
            "feature_idxs", torch.arange(-(seq_len - 1) * step, 0, step, dtype=torch.long), persistent=False
        )
        self.export_heads: list[str] = []
        self._trainable_modules = list(trainable_modules or [])
        self._configure_trainable_modules(list(frozen_modules or []))
        if self._trainable_modules:
            self.train(True)

    # ---- freezing --------------------------------------------------------------------------
    def _configure_trainable_modules(self, frozen_modules: list[str]) -> None:
        if self._trainable_modules:
            logger.warning(f"Only {self._trainable_modules} train; frozen_modules={frozen_modules} is ignored")
            for name, p in self.named_parameters():
                p.requires_grad = _matches(name, self._trainable_modules)
            return
        for frozen in frozen_modules:
            module = self.get_submodule(frozen) if frozen in dict(self.named_modules()) else None
            if module is None:
                logger.warning(f"Module {frozen} not found. It will not be frozen.")
                continue
            logger.info(f"Freezing {frozen}")
            for p in module.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._trainable_modules:
            for module in self.children():
                module.eval()
            for name, module in self.named_modules():
                if _matches(name, self._trainable_modules):
                    module.train(True)
        return self

    # ---- stages ----------------------------------------------------------------------------
    @property
    def num_tokens(self) -> int:
        return self.vision_encoder.num_tokens

    @property
    def token_dim(self) -> int:
        """Width of one frame in the deployment feature buffer."""
        return self.num_tokens * self.feat_size

    def encode_frames(self, frames: torch.Tensor) -> VisionOutput:
        b, f = frames.shape[:2]
        return self.vision_encoder(frames.reshape(b * f, *frames.shape[2:]))

    def _goal_tokens(self, goal, batch: int, frames: int, decisions: int, observation=None):
        encoder = self.goal_encoder
        n = batch * decisions
        if encoder.num_tokens == 0:
            return None
        if goal is None:
            return encoder(None, batch_size=n)
        if encoder.per_frame:
            if goal.shape[:2] != (batch, frames):
                raise ValueError(
                    f"{type(encoder).__name__} expects per-frame goals (B, F, ...), got {tuple(goal.shape)}"
                )
            goal = goal[:, -1] if decisions == 1 else goal.flatten(0, 1)
        else:
            if goal.shape[0] != batch:
                raise ValueError(
                    f"{type(encoder).__name__} expects one goal per window (B, ...), got {tuple(goal.shape)}"
                )
            if decisions > 1:
                goal = goal.repeat_interleave(decisions, dim=0)
        return encoder(goal, observation=observation)

    def forward(
        self, frames: torch.Tensor, goal: torch.Tensor | None = None, noise: torch.Tensor | None = None
    ) -> PolicyOutput:
        b, f = frames.shape[:2]
        vision = self.encode_frames(frames)
        context = self.temporal_encoder(vision.tokens.reshape(b, f, self.num_tokens, self.feat_size))
        decisions = context.shape[1]
        observation = frames.flatten(0, 1) if decisions > 1 else frames[:, -1]
        goal_tokens = self._goal_tokens(goal, b, f, decisions, observation)
        plan = self.action_decoder(context.reshape(b * decisions, self.num_tokens, self.feat_size), goal_tokens, noise)
        return PolicyOutput(vision=vision, plan=plan, goal_tokens=goal_tokens)

    def predict(
        self,
        frame: torch.Tensor,
        feature_buffer: torch.Tensor,
        goal: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ):
        """One decision from the newest frame pair plus buffered past-frame tokens (export graph)."""
        b = frame.shape[0]
        vision = self.vision_encoder(frame, export_heads=self.export_heads)
        current = vision.tokens.reshape(b, 1, self.token_dim)
        window = torch.cat([feature_buffer[:, self.feature_idxs], current], dim=1)
        context = self.temporal_encoder(window.reshape(b, -1, self.num_tokens, self.feat_size))[:, -1]
        goal_tokens = (
            None if self.goal_encoder.num_tokens == 0 else self.goal_encoder(goal, batch_size=b, observation=frame)
        )
        plan = self.action_decoder(context, goal_tokens, noise).plans
        heads = tuple(vision.heads[name] for name in self.vision_encoder.get_head_output_names(self.export_heads))
        return (plan, vision.pose, current.flatten(1), *heads)

    # ---- export contract -------------------------------------------------------------------
    def export_input_names(self) -> list[str]:
        names = ["input", "feature_buffer"]
        if self.goal_encoder.num_tokens:
            names.append("goal")
        if self.action_decoder.uses_noise:
            names.append("noise")
        return names

    def export_output_names(self) -> list[str]:
        return ["plan", "pose", "feat_out", *self.vision_encoder.get_head_output_names(self.export_heads)]

    def example_inputs(self, batch_size: int, image_hw: tuple[int, int], device=None) -> tuple:
        frame = torch.rand(batch_size, 6, *image_hw, device=device)
        buffer = (
            torch.randn(
                batch_size, int(-self.feature_idxs[0]) if len(self.feature_idxs) else 0, self.token_dim, device=device
            )
            * 0.1
        )
        inputs = [frame, buffer]
        if self.goal_encoder.num_tokens:
            inputs.append(self.goal_encoder.example_input(batch_size, device, image_hw=tuple(image_hw)))
        if self.action_decoder.uses_noise:
            inputs.append(self.action_decoder.example_noise(batch_size, device))
        return tuple(inputs)

    # ---- losses ---------------------------------------------------------------------------
    def get_losses(self, preds: PolicyOutput, targets):
        vision_loss_dict = self.vision_encoder.get_losses(preds.vision, targets["vision"])
        action_loss_dict, action_loss_debug = self.action_decoder.get_losses(preds.plan, targets["action"])
        total_loss = (
            self.loss_cfg.vision_weight * vision_loss_dict["total"]
            + self.loss_cfg.action_weight * action_loss_dict["total"]
        )
        loss_dict = dict(loss=total_loss)
        loss_dict.update({f"vision_{k}": _detach(v) for k, v in vision_loss_dict.items()})
        loss_dict.update({f"action_{k}": _detach(v) for k, v in action_loss_dict.items()})
        return loss_dict, dict(action_loss_debug=action_loss_debug)
