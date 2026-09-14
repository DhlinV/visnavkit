"""Anchor classification with per-anchor residual refinement (MIMIC / Hydra-MDP style)."""

import torch
import torch.nn.functional as F

from visnavkit.models.action.anchors import AnchorSet
from visnavkit.models.action.base import BaseActionDecoder, regression_loss
from visnavkit.models.layers.mlp import build_mlp
from visnavkit.models.outputs import PlanOutput

__all__ = ["AnchorDecoder"]


class AnchorDecoder(BaseActionDecoder):
    """Score ``K`` anchor trajectories and regress a residual for each; ``num_modes = K``."""

    def __init__(
        self,
        action_space,
        normalizer=None,
        feat_size=256,
        anchors: AnchorSet | None = None,
        hidden=256,
        loss="l1",
        loss_cls_alpha=1.0,
        dropout=0.0,
        **kwargs,
    ):
        anchors = (anchors or AnchorSet()).build(action_space.t_anchors.tolist(), action_space.pose_size)
        super().__init__(
            action_space, normalizer, feat_size=feat_size, num_modes=anchors.num_anchors, dropout=dropout, **kwargs
        )
        self.anchors = anchors
        self.loss_cls_alpha = loss_cls_alpha
        self.criterion = regression_loss(loss)
        self.classifier = build_mlp(feat_size, hidden, self.num_modes, layers=1, dropout=dropout)
        self.residual = build_mlp(
            feat_size, hidden, self.num_modes * self.action_space.flat_dim, layers=2, dropout=dropout
        )

    def anchor_actions(self) -> torch.Tensor:
        """``(K, T, A)`` anchors in normalized action space."""
        return self.normalizer.normalize(self.action_space.targets_from_poses(self.anchors.poses))

    def decode(self, cond, tokens, noise=None):
        n = cond.shape[0]
        residual = self.residual(cond).reshape(n, self.num_modes, self.num_pts, self.action_dim)
        mu = self.anchor_actions()[None].to(residual.dtype) + residual
        logits = self.classifier(cond)
        return PlanOutput(plans=self.pack(mu, None, logits), mu=mu, logits=logits, cond=cond)

    def loss(self, preds, gt, targets):
        label = self.anchors.nearest(targets["future_poses"])
        rows = torch.arange(gt.shape[0], device=gt.device)
        per_sample = self.criterion(preds.mu[rows, label], gt).mean(dim=(1, 2))
        reg = per_sample.mean()
        cls = F.cross_entropy(preds.logits, label)
        return dict(total=reg + self.loss_cls_alpha * cls, reg=reg, cls=cls), dict(
            imitation_loss_per_sample=per_sample.detach()
        )
