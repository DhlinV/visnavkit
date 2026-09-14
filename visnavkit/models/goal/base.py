"""Goal encoders map a goal specification to ``(N, G, D)`` tokens for the action decoder."""

import torch
import torch.nn as nn

__all__ = ["BaseGoalEncoder"]


class BaseGoalEncoder(nn.Module):
    """Shared goal-token contract, learned null token, and per-sample goal dropout.

    Args:
        feat_size: token width ``D``.
        num_tokens: ``G`` tokens emitted per goal (0 for goal-free policies).
        p_drop: training-time probability of replacing a sample's goal with the null token, so
            the policy also learns goal-free behaviour (NoMaD-style goal masking).
        per_frame: True when the dataset provides one goal per observed frame ``(B, F, ...)``;
            False when one goal describes the whole window ``(B, ...)``.
        goal_type: dataset-facing name of the goal specification (``none``, ``point``, ``image``,
            ``route_image``, ``instruction``); datasets read it from the model config.
    """

    goal_type: str = "none"
    per_frame: bool = False

    def __init__(self, feat_size: int, num_tokens: int = 1, p_drop: float = 0.0, goal_type: str | None = None):
        super().__init__()
        if num_tokens < 0 or feat_size < 1:
            raise ValueError("num_tokens must be nonnegative and feat_size positive")
        if not 0 <= p_drop <= 1:
            raise ValueError("p_drop must be between 0 and 1")
        self.feat_size = feat_size
        self.num_tokens = num_tokens
        self.p_drop = p_drop
        if goal_type is not None:
            self.goal_type = goal_type
        if num_tokens:
            self.null_token = nn.Parameter(torch.zeros(1, num_tokens, feat_size))
            nn.init.normal_(self.null_token, std=0.02)

    def encode(self, goal: torch.Tensor, observation: torch.Tensor | None = None) -> torch.Tensor:
        """Goal batch -> ``(N, G, D)`` tokens."""
        raise NotImplementedError

    def example_input(self, batch_size: int, device=None, image_hw=(64, 64)) -> torch.Tensor | None:
        """Synthetic goal batch for shape checks and export tracing (None for goal-free encoders).

        ``image_hw`` sizes image-like goals; datasets crop goal images like the frames.
        """
        return None

    def forward(
        self, goal: torch.Tensor | None = None, batch_size: int | None = None, observation: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.num_tokens == 0:
            n = batch_size if goal is None else goal.shape[0]
            device = None if goal is None else goal.device
            return torch.zeros(n, 0, self.feat_size, device=device)
        if goal is None:
            if batch_size is None:
                raise ValueError("batch_size is required when no goal is supplied")
            return self.null_token.repeat(batch_size, 1, 1)  # a copy, not a parameter view (tracing/FLOP counting)
        tokens = self.encode(goal, observation)
        if self.training and self.p_drop > 0:
            keep = torch.rand(tokens.shape[0], 1, 1, device=tokens.device) >= self.p_drop
            tokens = torch.where(keep, tokens, self.null_token.to(tokens.dtype))
        return tokens
