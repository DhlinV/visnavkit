import torch
import torch.nn as nn
import torch.nn.functional as F

from navigators.models.heads.plan_head import parse_plan_output


class WaypointHead(nn.Module):
    """Deterministic single-mode waypoint regression head (GNM / ViNT / CityWalker style).

    An MLP regresses the flattened future trajectory directly with a plain L1/MSE loss
    instead of a multi-hypothesis NLL. Emits the same flat MHP layout as PlanHead with
    num_modes = 1 (log_b = 0, conf = 0) so metrics, export, and parse_output are unchanged.
    Extra PlanHead-only config keys are accepted and ignored so recipes can swap heads
    via _target_ alone.
    """

    def __init__(
        self,
        num_pts=10,
        feat_size=512,
        pose_size=3,
        hidden=256,
        loss="l1",
        dropout=0,
        num_modes=1,
        **_ignored,
    ):
        super().__init__()
        assert num_modes == 1, "WaypointHead is single-mode; use PlanHead/DiffusionPlanHead for multi-modal plans"
        self.num_modes = 1
        self.num_pts = num_pts
        self.pose_size = pose_size
        self.traj_dim = num_pts * pose_size
        self.flat_size = 2 * self.traj_dim + 1
        self.loss = loss
        self.pretrained = False

        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_size, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.traj_dim),
        )

    def forward(self, x):
        mu = self.mlp(x)
        # flat MHP layout, single mode: [mu, log_b=0, conf=0]
        plans = torch.cat([mu, torch.zeros_like(mu), mu.new_zeros(mu.shape[0], 1)], dim=-1)
        return dict(plans=plans)

    def get_losses(self, preds, gt):
        """Plain regression loss (reported as reg; cls is zero — there is a single mode)."""
        mu = preds["plans"][:, : self.traj_dim]
        target = gt.reshape(gt.shape[0], -1)
        loss_fn = F.l1_loss if self.loss == "l1" else F.mse_loss
        per_sample = loss_fn(mu, target, reduction="none").mean(-1)
        reg = per_sample.mean()
        loss_dict = dict(total=reg, reg=reg.detach(), cls=torch.zeros_like(reg).detach())
        return loss_dict, dict(imitation_loss_per_sample=per_sample.detach())

    @torch.no_grad()
    def parse_output(self, output):
        return parse_plan_output(output, num_modes=1, num_pts=self.num_pts, pose_size=self.pose_size)
