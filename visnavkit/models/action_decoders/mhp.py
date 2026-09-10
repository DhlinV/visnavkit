"""Multi-hypothesis trajectory regression and mode classification."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from visnavkit.models.action_decoders.outputs import parse_plan_output
from visnavkit.models.layers.res_block import FusableResBlock
from visnavkit.models.losses.laplace_nll_loss import LaplaceNLLLoss
from visnavkit.utils.logger import get_logger

__all__ = ["PlanHead"]

logger = get_logger(__name__)


class PlanHead(nn.Module):
    """Multi-hypothesis (MHP) plan head: per-mode trajectory regression + mode classification."""

    def __init__(
        self,
        num_modes=3,
        num_pts=10,
        feat_size=512,
        pose_size=2,
        loss_cls_alpha=1,
        hidden=128,
        weights: str | None = None,
        dropout=0,
        mode_selection="angle",
        angle_deg_tiebreak_threshold=5.0,
        log_b_min=-1.609,
    ):
        super().__init__()

        self.pose_size = pose_size
        self.flat_size = num_modes * (2 * num_pts * self.pose_size + 1)  # (mu_std * pose_width + conf) * num_modes
        self.num_modes = num_modes
        self.num_pts = num_pts
        self.loss_cls_alpha = loss_cls_alpha
        self.pretrained = False
        self.dropout = dropout
        self.mode_selection = mode_selection
        self.angle_deg_tiebreak_threshold = angle_deg_tiebreak_threshold
        self.loss_reg = LaplaceNLLLoss(reduction="none", log_b_min=log_b_min)

        self.plan_head = nn.Sequential(
            nn.Dropout(dropout),
            FusableResBlock(feat_size, 2 * feat_size),
            FusableResBlock(feat_size, 2 * feat_size),
            nn.Linear(feat_size, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            FusableResBlock(hidden, hidden),
            nn.Linear(hidden, self.flat_size),
        )

        self.scale = nn.Parameter(torch.ones(self.flat_size))

        if weights is not None:
            self.pretrained = True
            logger.info(f"Loading plan head weights from {weights}")
            state_dict = torch.load(weights)
            self.load_state_dict(state_dict)

    def forward(self, x):
        plans = self.plan_head(x) * self.scale
        return dict(plans=plans)

    def get_losses(self, preds, gt):
        """MHP loss with Laplacian NLL for regression. ``preds`` is this head's forward output dict."""
        pred = preds["plans"]
        target = gt[:, None, :, :]  # add batch dim

        # reshape
        preds = pred.reshape(-1, self.num_modes, 2 * self.num_pts * self.pose_size + 1)
        pred_conf = preds[..., -1]
        pred_trajectories = preds[..., :-1].reshape(-1, self.num_modes, 2, self.num_pts, self.pose_size)
        pred_mu = pred_trajectories[:, :, 0]

        if self.mode_selection == "ade":
            distances = torch.norm(target[..., :2] - pred_mu[..., :2], dim=-1).sum(-1)
            best_mode = torch.argmin(distances, dim=-1)
        elif self.mode_selection == "angle":
            gt_angles = torch.arctan2(target[..., -1, 1], target[..., -1, 0])
            pred_angles = torch.arctan2(pred_mu[..., -1, 1], pred_mu[..., -1, 0])
            angles = (gt_angles - pred_angles).abs()
            angles = torch.minimum(angles, 2 * np.pi - angles)
            best_mode = torch.argmin(angles, dim=-1)
        elif self.mode_selection == "angle-with-tie-break":
            gt_angles = torch.arctan2(target[..., -1, 1], target[..., -1, 0])
            pred_angles = torch.arctan2(pred_mu[..., -1, 1], pred_mu[..., -1, 0])
            angles = (gt_angles - pred_angles).abs()
            angles = torch.minimum(angles, 2 * np.pi - angles)

            # tie break
            distances = torch.norm(target[..., :2] - pred_mu[..., :2], dim=-1).sum(-1)
            valid = angles <= np.deg2rad(self.angle_deg_tiebreak_threshold)
            best_mode = torch.where(
                valid.any(dim=-1),
                torch.where(valid, distances, float("inf")).argmin(dim=-1),
                distances.argmin(dim=-1),
            )
        elif self.mode_selection == "cosine-similarity":
            # Select the mode whose full xy path is most cosine-similar to the GT path, flattened
            # over T*xy into one vector per mode. Cosine is scale-invariant, so it matches trajectory
            # SHAPE/DIRECTION rather than closest distance (ade/fde) or only the final-point heading
            # (angle) -- which cuts the near-tie churn 'angle' suffers on low-speed / near-straight clips.
            gt_flat = target[..., :2].flatten(start_dim=-2)  # (B, 1, T*2)
            pred_flat = pred_mu[..., :2].flatten(start_dim=-2)  # (B, M, T*2)
            cos_sim = F.cosine_similarity(pred_flat, gt_flat, dim=-1, eps=1e-6)  # (B, M), broadcast over M
            best_mode = torch.argmax(cos_sim, dim=-1)  # highest similarity = best
        else:
            raise ValueError(f"Unsupported mode selection {self.mode_selection}")

        best_traj = pred_trajectories[torch.arange(pred_trajectories.shape[0]), best_mode]

        # cls loss (per-sample and mean)
        ego_cls_loss = F.cross_entropy(pred_conf, best_mode.detach())
        ego_cls_per_sample = F.cross_entropy(pred_conf, best_mode.detach(), reduction="none")

        # reg loss (per-sample and mean)
        B = best_traj.shape[0]
        # NOTE: trajectory outputs don't follow the torch "chunk" layout (inherited from autopilot),
        # so mu/log_b are indexed and concatenated here rather than best_traj.chunk(2, dim=-1)
        reg_raw = self.loss_reg(torch.cat([best_traj[:, 0], best_traj[:, 1]], dim=-1), gt)
        reg_per_sample = reg_raw.reshape(B, -1).mean(-1)
        ego_reg_nll_loss = reg_per_sample.mean()

        loss_imitation_per_sample = self.loss_cls_alpha * ego_cls_per_sample + reg_per_sample
        loss_imitation = loss_imitation_per_sample.mean()

        loss_dict = dict(
            total=loss_imitation,
            cls=ego_cls_loss.detach(),
            reg=ego_reg_nll_loss.detach(),
        )
        loss_debug = dict(imitation_loss_per_sample=loss_imitation_per_sample.detach())
        return loss_dict, loss_debug

    @torch.no_grad()
    def parse_output(self, output):
        return parse_plan_output(output, num_modes=self.num_modes, num_pts=self.num_pts, pose_size=self.pose_size)
