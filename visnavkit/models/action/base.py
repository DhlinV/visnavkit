"""Action decoder contract: context tokens (+ goal tokens) -> trajectories in the shared flat layout."""

import torch
import torch.nn as nn

from visnavkit.models.action.normalizer import ActionNormalizer
from visnavkit.models.action.outputs import parse_plan_output
from visnavkit.models.action.spaces import ActionSpace
from visnavkit.models.layers.attention import AttentionPool
from visnavkit.models.outputs import PlanOutput

__all__ = ["BaseActionDecoder"]


class BaseActionDecoder(nn.Module):
    """Shared conditioning, packing, and loss plumbing for every trajectory decoder.

    ``forward(context, goal=None, noise=None)`` takes ``(N, T_ctx, D)`` context tokens and
    ``(N, G, D)`` goal tokens. Subclasses implement :meth:`decode` on the pooled conditioning
    vector (and the raw tokens for cross-attention denoisers) and :meth:`loss` on normalized
    action-space targets. :meth:`pack` converts normalized actions to the flat pose-space
    layout ``[mu, log_scale, confidence_logit]`` per mode.

    Args:
        action_space: what is predicted per anchor (waypoint / velocity) and the anchor grid.
        normalizer: affine action normalization (identity by default).
        pooling: ``attention`` (learned query) or ``mean`` over the concatenated tokens; a
            single context token with no goal is passed through unchanged.
    """

    uses_noise: bool = False

    def __init__(
        self,
        action_space: ActionSpace,
        normalizer: ActionNormalizer | None = None,
        feat_size: int = 256,
        num_modes: int = 1,
        pooling: str = "attention",
        pool_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if pooling not in ("attention", "mean"):
            raise ValueError("pooling must be 'attention' or 'mean'")
        if num_modes < 1:
            raise ValueError("num_modes must be positive")
        self.action_space = action_space
        self.normalizer = normalizer or ActionNormalizer()
        self.feat_size = feat_size
        self.num_modes = num_modes
        self.dropout = dropout
        self.type_embedding = nn.Embedding(2, feat_size)
        nn.init.normal_(self.type_embedding.weight, std=0.02)
        self.pool = AttentionPool(feat_size, pool_heads, dropout) if pooling == "attention" else None

    # ---- shapes -------------------------------------------------------------------------
    @property
    def num_pts(self) -> int:
        return self.action_space.num_pts

    @property
    def action_dim(self) -> int:
        return self.action_space.action_dim

    @property
    def pose_size(self) -> int:
        return self.action_space.pose_size

    @property
    def flat_size(self) -> int:
        return self.num_modes * (2 * self.num_pts * self.pose_size + 1)

    # ---- conditioning ----------------------------------------------------------------------
    def condition(self, context: torch.Tensor, goal: torch.Tensor | None = None):
        if context.ndim != 3 or context.shape[-1] != self.feat_size:
            raise ValueError(f"context must be (N, T, {self.feat_size}), got {tuple(context.shape)}")
        if goal is not None and goal.shape[1] > 0:
            tokens = torch.cat([context + self.type_embedding.weight[0], goal + self.type_embedding.weight[1]], dim=1)
        else:
            tokens = context
        if tokens.shape[1] == 1:
            cond = tokens[:, 0]
        elif self.pool is None:
            cond = tokens.mean(dim=1)
        else:
            cond = self.pool(tokens)
        return tokens, cond

    def forward(
        self, context: torch.Tensor, goal: torch.Tensor | None = None, noise: torch.Tensor | None = None
    ) -> PlanOutput:
        tokens, cond = self.condition(context, goal)
        return self.decode(cond, tokens, noise)

    def decode(self, cond: torch.Tensor, tokens: torch.Tensor, noise: torch.Tensor | None) -> PlanOutput:
        raise NotImplementedError

    def example_noise(self, batch_size: int, device=None) -> torch.Tensor | None:
        return None

    # ---- packing ------------------------------------------------------------------------
    def pack(self, mu: torch.Tensor, log_b: torch.Tensor | None = None, logits: torch.Tensor | None = None):
        """Normalized ``(N, M, T, A)`` means (+ log-scales, logits) -> flat pose-space layout."""
        n, m = mu.shape[:2]
        poses = self.action_space.to_poses(self.normalizer.unnormalize(mu))
        if log_b is not None and self.action_space.kind == "waypoint":
            scales = log_b + torch.log(self.normalizer.scale.to(mu.dtype)) if not self.normalizer.identity else log_b
        else:
            scales = torch.zeros_like(poses)
        conf = logits if logits is not None else mu.new_zeros(n, m)
        return torch.cat([poses.flatten(2), scales.flatten(2), conf[..., None]], dim=-1).flatten(1)

    def targets(self, targets: dict) -> torch.Tensor:
        """Dataset targets -> normalized ``(N, T, A)`` actions."""
        return self.normalizer.normalize(self.action_space.targets_from_poses(targets["future_poses"]))

    # ---- losses -------------------------------------------------------------------------
    def loss(self, preds: PlanOutput, gt: torch.Tensor, targets: dict):
        """Return ``(dict(total, reg, cls), debug)`` from normalized targets ``gt``."""
        raise NotImplementedError

    def get_losses(self, preds: PlanOutput, targets: dict):
        loss_dict, debug = self.loss(preds, self.targets(targets), targets)
        for key in ("reg", "cls"):
            loss_dict[key] = loss_dict[key].detach()
        return loss_dict, debug

    @torch.no_grad()
    def parse_output(self, plans: torch.Tensor) -> dict:
        return parse_plan_output(plans, num_modes=self.num_modes, num_pts=self.num_pts, pose_size=self.pose_size)


def regression_loss(kind: str):
    if kind == "l1":
        return nn.L1Loss(reduction="none")
    if kind == "mse":
        return nn.MSELoss(reduction="none")
    raise ValueError(f"loss must be 'l1' or 'mse', got {kind!r}")
