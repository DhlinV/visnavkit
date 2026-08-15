import math

import torch
import torch.nn as nn


class LaplaceNLLLoss(nn.Module):
    """
    The Laplacian NLL loss is defined as:

      log(2b) + abs(target - μ) / b

    This is the L1 equivalent of NLL, which uses a guassian distribution.
    Generally, this has better handling for outliers.

    Laplacian NLL expects a chunked tensor of shape [μ0, ..., μN, log_b0, ..., log_bN].
    It expects outputs the b parameter to be log(b) values rather than b directly.
    This removes the need to enforce positivity at the network layer.
    """

    def __init__(
        self,
        log_b_min: float | None = -5.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.log_b_min = log_b_min
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loc, log_b = pred.chunk(2, dim=-1)
        if self.log_b_min is not None:
            log_b = torch.clamp(log_b, min=self.log_b_min)

        nll = math.log(2.0) + log_b + torch.abs(target - loc) / torch.exp(log_b)
        if self.reduction == "mean":
            return nll.mean()
        if self.reduction == "sum":
            return nll.sum()
        if self.reduction == "none":
            return nll
        raise ValueError(f"{self.reduction} is not a valid value for reduction")
