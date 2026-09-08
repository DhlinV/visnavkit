import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from visnavkit.models.heads.plan_head import parse_plan_output


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding, (N,) int timesteps -> (N, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / (half - 1))
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class DiffusionPlanHead(nn.Module):
    """Conditional trajectory diffusion head (DDPM training, DDIM sampling) in the style of
    diffusion policy / NoMaD: an MLP denoiser predicts the noise added to the flattened
    GT trajectory, conditioned on the action-decoder feature and a timestep embedding.

    Emits the same flat MHP layout as PlanHead — per mode [mu, log_b, conf] with log_b = 0
    and uniform confidences — so metrics, export, and parse_output are unchanged. The
    num_modes hypotheses come from independent noise samples. Extra PlanHead-only config
    keys are accepted and ignored so configs can swap heads via _target_ alone.
    """

    def __init__(
        self,
        num_modes=5,
        num_pts=10,
        feat_size=512,
        pose_size=3,
        hidden=512,
        time_embed_dim=64,
        train_timesteps=100,
        sample_steps=8,
        traj_scale=10.0,
        dropout=0,
        **_ignored,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.num_pts = num_pts
        self.pose_size = pose_size
        self.flat_size = num_modes * (2 * num_pts * pose_size + 1)
        self.traj_dim = num_pts * pose_size
        self.time_embed_dim = time_embed_dim
        self.train_timesteps = train_timesteps
        self.sample_steps = sample_steps
        self.traj_scale = traj_scale
        self.pretrained = False

        betas = torch.linspace(1e-4, 0.02, train_timesteps)  # DDPM linear schedule
        self.register_buffer("alphas_cumprod", torch.cumprod(1.0 - betas, dim=0))

        self.denoiser = nn.Sequential(
            nn.Linear(self.traj_dim + feat_size + time_embed_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.traj_dim),
        )

    def _eps(self, x_t, t, cond):
        temb = timestep_embedding(t, self.time_embed_dim)
        return self.denoiser(torch.cat([x_t, cond, temb], dim=-1))

    def _sample(self, cond, noise=None):
        """DDIM (eta=0) over sample_steps; num_modes independent noise draws per sample."""
        B = cond.shape[0]
        cond = cond.repeat_interleave(self.num_modes, dim=0)  # (B*M, D)
        x_t = noise if noise is not None else torch.randn(cond.shape[0], self.traj_dim, device=cond.device, dtype=cond.dtype)
        ts = torch.linspace(self.train_timesteps - 1, 0, self.sample_steps, device=cond.device).long()
        x0 = x_t
        for i in range(self.sample_steps):
            ab_t = self.alphas_cumprod[ts[i]]
            eps = self._eps(x_t, ts[i].expand(cond.shape[0]), cond)
            x0 = ((x_t - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()).clamp(-4, 4)
            ab_prev = self.alphas_cumprod[ts[i + 1]] if i + 1 < self.sample_steps else torch.ones_like(ab_t)
            x_t = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps

        mu = (x0 * self.traj_scale).reshape(B, self.num_modes, self.traj_dim)
        # flat MHP layout per mode: [mu, log_b=0, conf=0] (uniform confidences after softmax)
        per_mode = torch.cat([mu, torch.zeros_like(mu), mu.new_zeros(B, self.num_modes, 1)], dim=-1)
        return per_mode.reshape(B, self.flat_size)

    def forward(self, x):
        if self.training:
            # loss only needs cond; skip the expensive sampling and keep the output contract
            plans = x.new_zeros(x.shape[0], self.flat_size)
        else:
            plans = self._sample(x)
        return dict(plans=plans, cond=x)

    def get_losses(self, preds, gt):
        """Noise-prediction MSE (reported as reg; cls is zero — modes are noise draws, not logits)."""
        cond = preds["cond"]
        x0 = gt.reshape(gt.shape[0], -1) / self.traj_scale
        t = torch.randint(0, self.train_timesteps, (x0.shape[0],), device=x0.device)
        ab = self.alphas_cumprod[t][:, None]
        noise = torch.randn_like(x0)
        x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        per_sample = F.mse_loss(self._eps(x_t, t, cond), noise, reduction="none").mean(-1)
        reg = per_sample.mean()
        loss_dict = dict(total=reg, reg=reg.detach(), cls=torch.zeros_like(reg).detach())
        return loss_dict, dict(imitation_loss_per_sample=per_sample.detach())

    @torch.no_grad()
    def parse_output(self, output):
        return parse_plan_output(output, num_modes=self.num_modes, num_pts=self.num_pts, pose_size=self.pose_size)
