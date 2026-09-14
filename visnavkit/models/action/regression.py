"""Deterministic single-trajectory regression (GNM / ViNT / CityWalker style)."""

import torch

from visnavkit.models.action.base import BaseActionDecoder, regression_loss
from visnavkit.models.layers.mlp import build_mlp
from visnavkit.models.outputs import PlanOutput

__all__ = ["RegressionDecoder"]


class RegressionDecoder(BaseActionDecoder):
    def __init__(
        self, action_space, normalizer=None, feat_size=256, hidden=256, layers=2, loss="l1", dropout=0.0, **kwargs
    ):
        super().__init__(action_space, normalizer, feat_size=feat_size, num_modes=1, dropout=dropout, **kwargs)
        self.mlp = build_mlp(feat_size, hidden, self.action_space.flat_dim, layers=layers, dropout=dropout)
        self.criterion = regression_loss(loss)

    def decode(self, cond, tokens, noise=None):
        mu = self.mlp(cond).reshape(cond.shape[0], 1, self.num_pts, self.action_dim)
        return PlanOutput(plans=self.pack(mu), mu=mu, cond=cond)

    def loss(self, preds, gt, targets):
        per_sample = self.criterion(preds.mu[:, 0], gt).mean(dim=(1, 2))
        reg = per_sample.mean()
        return dict(total=reg, reg=reg, cls=torch.zeros_like(reg)), dict(imitation_loss_per_sample=per_sample.detach())
