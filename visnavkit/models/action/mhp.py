"""Multi-hypothesis prediction: per-mode Laplace regression plus mode classification."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from visnavkit.models.action.base import BaseActionDecoder
from visnavkit.models.layers.res_block import FusableResBlock
from visnavkit.models.losses.laplace_nll_loss import LaplaceNLLLoss
from visnavkit.models.outputs import PlanOutput

__all__ = ["MHPDecoder"]

MODE_SELECTIONS = ("ade", "angle", "angle-with-tie-break", "cosine-similarity")


class MHPDecoder(BaseActionDecoder):
    """``num_modes`` trajectories with Laplace NLL on the mode closest to the target and CE on its index.

    Mode selection compares poses (not normalized actions): ``ade`` (closest xy path), ``angle``
    (final-point heading), ``angle-with-tie-break`` (heading, then ADE within a tolerance), or
    ``cosine-similarity`` (whole-path direction).
    """

    def __init__(
        self,
        action_space,
        normalizer=None,
        feat_size=256,
        num_modes=5,
        hidden=128,
        dropout=0.0,
        loss_cls_alpha=1.0,
        mode_selection="angle",
        angle_deg_tiebreak_threshold=5.0,
        log_b_min=-1.609,
        **kwargs,
    ):
        super().__init__(action_space, normalizer, feat_size=feat_size, num_modes=num_modes, dropout=dropout, **kwargs)
        if mode_selection not in MODE_SELECTIONS:
            raise ValueError(f"mode_selection must be one of {MODE_SELECTIONS}")
        self.loss_cls_alpha = loss_cls_alpha
        self.mode_selection = mode_selection
        self.angle_deg_tiebreak_threshold = angle_deg_tiebreak_threshold
        self.loss_reg = LaplaceNLLLoss(reduction="none", log_b_min=log_b_min)
        per_mode = 2 * self.action_space.flat_dim + 1
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            FusableResBlock(feat_size, 2 * feat_size),
            FusableResBlock(feat_size, 2 * feat_size),
            nn.Linear(feat_size, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            FusableResBlock(hidden, hidden),
            nn.Linear(hidden, num_modes * per_mode),
        )

    def decode(self, cond, tokens, noise=None):
        n, d = cond.shape[0], self.action_space.flat_dim
        out = self.head(cond).reshape(n, self.num_modes, 2 * d + 1)
        mu = out[..., :d].reshape(n, self.num_modes, self.num_pts, self.action_dim)
        log_b = out[..., d : 2 * d].reshape(n, self.num_modes, self.num_pts, self.action_dim)
        logits = out[..., -1]
        return PlanOutput(plans=self.pack(mu, log_b, logits), mu=mu, log_b=log_b, logits=logits, cond=cond)

    def select_mode(self, pred_poses: torch.Tensor, gt_poses: torch.Tensor) -> torch.Tensor:
        """``(N, M, T, P)`` predictions vs ``(N, T, P)`` targets -> ``(N,)`` mode index."""
        target = gt_poses[:, None]
        if self.mode_selection == "ade":
            return torch.norm(target[..., :2] - pred_poses[..., :2], dim=-1).sum(-1).argmin(dim=-1)
        if self.mode_selection == "cosine-similarity":
            similarity = F.cosine_similarity(
                pred_poses[..., :2].flatten(-2), target[..., :2].flatten(-2), dim=-1, eps=1e-6
            )
            return similarity.argmax(dim=-1)
        gt_angles = torch.arctan2(target[..., -1, 1], target[..., -1, 0])
        pred_angles = torch.arctan2(pred_poses[..., -1, 1], pred_poses[..., -1, 0])
        angles = (gt_angles - pred_angles).abs()
        angles = torch.minimum(angles, 2 * math.pi - angles)
        if self.mode_selection == "angle":
            return angles.argmin(dim=-1)
        distances = torch.norm(target[..., :2] - pred_poses[..., :2], dim=-1).sum(-1)
        valid = angles <= math.radians(self.angle_deg_tiebreak_threshold)
        return torch.where(
            valid.any(dim=-1),
            torch.where(valid, distances, torch.full_like(distances, float("inf"))).argmin(dim=-1),
            distances.argmin(dim=-1),
        )

    def loss(self, preds, gt, targets):
        with torch.no_grad():
            pred_poses = self.action_space.to_poses(self.normalizer.unnormalize(preds.mu))
            best_mode = self.select_mode(pred_poses, targets["future_poses"])
        rows = torch.arange(gt.shape[0], device=gt.device)
        best = torch.cat([preds.mu[rows, best_mode].flatten(1), preds.log_b[rows, best_mode].flatten(1)], dim=-1)
        per_sample = self.loss_reg(best, gt.flatten(1)).mean(-1)
        reg = per_sample.mean()
        cls = F.cross_entropy(preds.logits, best_mode)
        total = reg + self.loss_cls_alpha * cls
        return dict(total=total, reg=reg, cls=cls), dict(imitation_loss_per_sample=per_sample.detach())
