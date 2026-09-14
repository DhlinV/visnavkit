"""NavigationPolicy: [vision, goal, ego status, calibration] -> context -> action decoder."""

from collections.abc import Sequence

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


def _add_frame_axis(value: torch.Tensor | None) -> torch.Tensor | None:
    """Deployment passes the newest frame's side input without a frame axis."""
    return value if value is None else value[:, None]


def _as_list(value) -> list:
    """A single module/tensor or any sequence of them (Hydra hands lists back as ListConfig)."""
    if value is None:
        return []
    if isinstance(value, nn.Module) or torch.is_tensor(value):
        return [value]
    return list(value)


class NavigationPolicy(nn.Module):
    """Compose the encoders and the decoder; every stage exchanges ``feat_size``-wide tokens.

    The policy takes the three raw inputs and owns the wiring between them:

    - ``vision`` ``(B, F, 3, H, W)`` RGB frames in [0, 1] -> per-frame vision tokens.
    - ``ego`` ``(B, F, E)`` free-form ego status -> per-frame ego tokens, concatenated with the
      vision tokens of the same frame, so the temporal encoder mixes both across time.
    - ``intrinsics`` ``(B, F, 3, 3)`` and ``extrinsics`` ``(B, F, 4, 4)`` -> per-frame camera
      tokens, joined the same way, so one policy can serve several camera rigs.
    - ``goal`` one goal per goal encoder -> goal tokens for the action decoder. ``goal_encoder``
      may be a list, in which case ``goal`` is the matching list and the tokens are concatenated.

    Training: ``forward(vision, goal=None, ego=None, intrinsics=None, extrinsics=None, noise=None)``.
    Deployment: ``predict(frame, feature_buffer, ...)`` encodes one frame and reuses past frame
    tokens from the buffer ``(B, history, K * D)``.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        temporal_encoder: nn.Module,
        action_decoder: nn.Module,
        goal_encoder: nn.Module | Sequence[nn.Module] | None = None,
        ego_encoder: nn.Module | None = None,
        camera_encoder: nn.Module | None = None,
        feat_size: int = 256,
        loss_cfg: DictConfig | None = None,
        export_cfg: DictConfig | None = None,
        frozen_modules: list[str] | None = None,
        trainable_modules: list[str] | None = None,
    ):
        super().__init__()
        goal_encoders = _as_list(goal_encoder)
        modules = [("vision_encoder", vision_encoder), ("action_decoder", action_decoder)]
        modules += [(f"goal_encoders.{i}", encoder) for i, encoder in enumerate(goal_encoders)]
        if ego_encoder is not None:
            modules.append(("ego_encoder", ego_encoder))
        if camera_encoder is not None:
            modules.append(("camera_encoder", camera_encoder))
        for name, module in modules:
            if getattr(module, "feat_size", feat_size) != feat_size:
                raise ValueError(f"{name}.feat_size must equal model.feat_size={feat_size}")
        if temporal_encoder.embed_dim != feat_size:
            raise ValueError(f"temporal_encoder.embed_dim must equal model.feat_size={feat_size}")
        self.vision_encoder = vision_encoder
        self.temporal_encoder = temporal_encoder
        self.goal_encoders = nn.ModuleList(goal_encoders)
        self.ego_encoder = ego_encoder
        self.camera_encoder = camera_encoder
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

    # ---- token budget ----------------------------------------------------------------------
    @property
    def vision_tokens(self) -> int:
        return self.vision_encoder.num_tokens

    @property
    def ego_tokens(self) -> int:
        return 0 if self.ego_encoder is None else self.ego_encoder.num_tokens

    @property
    def camera_tokens(self) -> int:
        return 0 if self.camera_encoder is None else self.camera_encoder.num_tokens

    @property
    def goal_tokens(self) -> int:
        return sum(encoder.num_tokens for encoder in self.goal_encoders)

    @property
    def num_tokens(self) -> int:
        """Tokens per frame entering the temporal encoder."""
        return self.vision_tokens + self.ego_tokens + self.camera_tokens

    @property
    def token_dim(self) -> int:
        """Width of one frame in the deployment feature buffer."""
        return self.num_tokens * self.feat_size

    # ---- stages ----------------------------------------------------------------------------
    def encode_frames(self, vision: torch.Tensor) -> VisionOutput:
        b, f = vision.shape[:2]
        return self.vision_encoder(vision.reshape(b * f, *vision.shape[2:]))

    def encode_ego(self, ego: torch.Tensor | None, batch: int, frames: int) -> torch.Tensor | None:
        if self.ego_tokens == 0:
            return None
        tokens = self.ego_encoder(ego, batch_size=batch, frames=frames)
        if tokens.shape[:2] != (batch, frames):
            raise ValueError(f"Ego status must be (B, F, E) = ({batch}, {frames}, E), got {tuple(tokens.shape[:2])}")
        return tokens

    def encode_camera(self, intrinsics, extrinsics, batch: int, frames: int, image_hw) -> torch.Tensor | None:
        if self.camera_tokens == 0:
            return None
        tokens = self.camera_encoder(intrinsics, extrinsics, batch_size=batch, frames=frames, image_hw=image_hw)
        if tokens.shape[:2] != (batch, frames):
            raise ValueError(f"Calibration must cover ({batch}, {frames}) frames, got {tuple(tokens.shape[:2])}")
        return tokens

    def _per_frame_tokens(self, vision_tokens, ego_tokens, camera_tokens, dim: int):
        extra = [t for t in (ego_tokens, camera_tokens) if t is not None]
        return torch.cat([vision_tokens, *extra], dim=dim) if extra else vision_tokens

    def _encode_goals(self, goals: list, batch: int, observation=None) -> torch.Tensor | None:
        """Per-encoder goals (already shaped for the decision batch) -> concatenated tokens."""
        if self.goal_tokens == 0:
            return None
        tokens = [
            encoder(goal, batch_size=batch, observation=observation)
            for encoder, goal in zip(self.goal_encoders, goals)
            if encoder.num_tokens
        ]
        return torch.cat(tokens, dim=1)

    def null_goal_tokens(self, batch_size: int) -> torch.Tensor | None:
        """Goal tokens for goal-free inference: each encoder's learned null token."""
        return self._encode_goals([None] * len(self.goal_encoders), batch_size)

    def _match_goals(self, goal) -> list:
        goals = [None] * len(self.goal_encoders) if goal is None else _as_list(goal)
        if len(goals) != len(self.goal_encoders):
            raise ValueError(f"{len(self.goal_encoders)} goal encoders expect {len(self.goal_encoders)} goals")
        return goals

    def _window_goals(self, goal, batch: int, frames: int, decisions: int) -> list:
        """Reshape each goal from its dataset layout to one entry per decision."""
        prepared = []
        for encoder, value in zip(self.goal_encoders, self._match_goals(goal)):
            if value is None or encoder.num_tokens == 0:
                prepared.append(None)
            elif encoder.per_frame:
                if value.shape[:2] != (batch, frames):
                    raise ValueError(
                        f"{type(encoder).__name__} expects per-frame goals (B, F, ...), got {tuple(value.shape)}"
                    )
                prepared.append(value[:, -1] if decisions == 1 else value.flatten(0, 1))
            else:
                if value.shape[0] != batch:
                    raise ValueError(
                        f"{type(encoder).__name__} expects one goal per window (B, ...), got {tuple(value.shape)}"
                    )
                prepared.append(value.repeat_interleave(decisions, dim=0) if decisions > 1 else value)
        return prepared

    def forward(
        self,
        vision: torch.Tensor,
        goal=None,
        ego: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        extrinsics: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> PolicyOutput:
        b, f = vision.shape[:2]
        vision_out = self.encode_frames(vision)
        ego_tokens = self.encode_ego(ego, b, f)
        camera_tokens = self.encode_camera(intrinsics, extrinsics, b, f, vision.shape[-2:])
        tokens = self._per_frame_tokens(
            vision_out.tokens.reshape(b, f, self.vision_tokens, self.feat_size), ego_tokens, camera_tokens, dim=2
        )
        context = self.temporal_encoder(tokens)
        decisions = context.shape[1]
        observation = vision.flatten(0, 1) if decisions > 1 else vision[:, -1]
        goal_tokens = self._encode_goals(self._window_goals(goal, b, f, decisions), b * decisions, observation)
        plan = self.action_decoder(context.reshape(b * decisions, self.num_tokens, self.feat_size), goal_tokens, noise)
        return PolicyOutput(
            vision=vision_out,
            plan=plan,
            goal_tokens=goal_tokens,
            ego_tokens=ego_tokens,
            camera_tokens=camera_tokens,
        )

    def predict(
        self,
        frame: torch.Tensor,
        feature_buffer: torch.Tensor,
        goal=None,
        ego: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        extrinsics: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ):
        """One decision from the newest frame plus buffered past-frame tokens (export graph).

        The newest frame's side inputs are given without a frame axis: ``ego (B, E)``,
        ``intrinsics (B, 3, 3)``, ``extrinsics (B, 4, 4)``.
        """
        b = frame.shape[0]
        vision = self.vision_encoder(frame, export_heads=self.export_heads)
        ego_tokens = self.encode_ego(_add_frame_axis(ego), b, 1)
        camera_tokens = self.encode_camera(
            _add_frame_axis(intrinsics), _add_frame_axis(extrinsics), b, 1, frame.shape[-2:]
        )
        tokens = self._per_frame_tokens(
            vision.tokens,
            None if ego_tokens is None else ego_tokens[:, 0],
            None if camera_tokens is None else camera_tokens[:, 0],
            dim=1,
        )
        current = tokens.reshape(b, 1, self.token_dim)
        window = torch.cat([feature_buffer[:, self.feature_idxs], current], dim=1)
        context = self.temporal_encoder(window.reshape(b, -1, self.num_tokens, self.feat_size))[:, -1]
        goal_tokens = self._encode_goals(self._match_goals(goal), b, frame)
        plan = self.action_decoder(context, goal_tokens, noise).plans
        heads = tuple(vision.heads[name] for name in self.vision_encoder.get_head_output_names(self.export_heads))
        speed = (vision.speed,) if self.vision_encoder.has_speed_head else ()
        return (plan, current.flatten(1), *speed, *heads)

    # ---- export contract -------------------------------------------------------------------
    def goal_input_names(self) -> list[str]:
        names = [f"goal_{i}" for i, encoder in enumerate(self.goal_encoders) if encoder.num_tokens]
        return ["goal"] if len(names) == 1 else names

    def export_input_names(self) -> list[str]:
        names = ["vision", "feature_buffer", *self.goal_input_names()]
        if self.ego_tokens:
            names.append("ego")
        if self.camera_tokens:
            names += ["intrinsics", "extrinsics"]
        if self.action_decoder.uses_noise:
            names.append("noise")
        return names

    def export_output_names(self) -> list[str]:
        names = ["plan", "feat_out"]
        if self.vision_encoder.has_speed_head:
            names.append("speed")
        return [*names, *self.vision_encoder.get_head_output_names(self.export_heads)]

    def example_inputs(self, batch_size: int, image_hw: tuple[int, int], device=None) -> tuple:
        frame = torch.rand(batch_size, 3, *image_hw, device=device)
        history = int(-self.feature_idxs[0]) if len(self.feature_idxs) else 0
        buffer = torch.randn(batch_size, history, self.token_dim, device=device) * 0.1
        inputs = [frame, buffer]
        inputs += [
            encoder.example_input(batch_size, device, image_hw=tuple(image_hw))
            for encoder in self.goal_encoders
            if encoder.num_tokens
        ]
        if self.ego_tokens:
            inputs.append(self.ego_encoder.example_input(batch_size, device=device).squeeze(1))
        if self.camera_tokens:
            calibration = self.camera_encoder.example_input(batch_size, 1, tuple(image_hw), device)
            inputs += [value.squeeze(1) for value in calibration]
        if self.action_decoder.uses_noise:
            inputs.append(self.action_decoder.example_noise(batch_size, device))
        return tuple(inputs)

    def example_batch(self, batch_size: int, frames: int, image_hw: tuple[int, int], device=None):
        """Synthetic ``(vision, goal, ego, intrinsics, extrinsics)`` inputs for shape checks."""
        vision = torch.rand(batch_size, frames, 3, *image_hw, device=device)
        goals = []
        for encoder in self.goal_encoders:
            if encoder.num_tokens == 0:
                goals.append(None)
                continue
            count = batch_size * frames if encoder.per_frame else batch_size
            goal = encoder.example_input(count, device, image_hw=tuple(image_hw))
            goals.append(goal.reshape(batch_size, frames, -1) if encoder.per_frame else goal)
        if len(self.goal_encoders) > 1:
            goal = goals
        else:
            goal = goals[0] if goals else None
        ego = None if self.ego_tokens == 0 else self.ego_encoder.example_input(batch_size, frames, device)
        camera = (None, None)
        if self.camera_tokens:
            camera = self.camera_encoder.example_input(batch_size, frames, tuple(image_hw), device)
        return vision, goal, ego, *camera

    # ---- losses ---------------------------------------------------------------------------
    def get_losses(self, preds: PolicyOutput, targets):
        vision_loss_dict = self.vision_encoder.get_losses(preds.vision, targets.get("vision", {}))
        action_loss_dict, action_loss_debug = self.action_decoder.get_losses(preds.plan, targets["action"])
        total_loss = self.loss_cfg.action_weight * action_loss_dict["total"]
        if vision_loss_dict:
            total_loss = total_loss + self.loss_cfg.vision_weight * vision_loss_dict["total"]
        loss_dict = dict(loss=total_loss)
        loss_dict.update({f"vision_{k}": _detach(v) for k, v in vision_loss_dict.items()})
        loss_dict.update({f"action_{k}": _detach(v) for k, v in action_loss_dict.items()})
        return loss_dict, dict(action_loss_debug=action_loss_debug)
