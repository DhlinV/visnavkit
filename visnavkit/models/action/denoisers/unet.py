"""1D conditional U-Net denoiser (diffusion policy / NoMaD ``ConditionalUnet1D``) with FiLM conditioning."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from visnavkit.models.layers.embeddings import SinusoidalTimeEmbedding

__all__ = ["UNet1DDenoiser"]


class Conv1dBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, n_groups):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2), nn.GroupNorm(n_groups, out_ch), nn.Mish()
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, kernel_size, n_groups):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Conv1dBlock(in_ch, out_ch, kernel_size, n_groups), Conv1dBlock(out_ch, out_ch, kernel_size, n_groups)]
        )
        self.film = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, 2 * out_ch))
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x)
        scale, bias = self.film(cond)[:, :, None].chunk(2, dim=1)
        out = self.blocks[1](out * (1 + scale) + bias)
        return out + self.residual(x)


class UNet1DDenoiser(nn.Module):
    """Time is the sequence axis; the trajectory is padded to a multiple of ``2 ** (levels - 1)``."""

    def __init__(
        self,
        action_dim: int,
        num_pts: int,
        cond_dim: int,
        down_dims=(64, 128, 256),
        kernel_size: int = 5,
        n_groups: int = 8,
        time_dim: int = 64,
    ):
        super().__init__()
        self.action_dim, self.num_pts = action_dim, num_pts
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        global_dim = time_dim + cond_dim
        dims = [action_dim, *down_dims]
        self.multiple = 2 ** (len(down_dims) - 1)
        self.down = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            last = i == len(down_dims) - 1
            self.down.append(
                nn.ModuleList(
                    [
                        ConditionalResBlock1D(d_in, d_out, global_dim, kernel_size, n_groups),
                        ConditionalResBlock1D(d_out, d_out, global_dim, kernel_size, n_groups),
                        nn.Identity() if last else nn.Conv1d(d_out, d_out, 3, stride=2, padding=1),
                    ]
                )
            )
        mid = dims[-1]
        self.mid = nn.ModuleList([ConditionalResBlock1D(mid, mid, global_dim, kernel_size, n_groups) for _ in range(2)])
        self.up = nn.ModuleList()
        for d_in, d_out in reversed(list(zip(dims[1:-1], dims[2:]))):
            self.up.append(
                nn.ModuleList(
                    [
                        ConditionalResBlock1D(2 * d_out, d_in, global_dim, kernel_size, n_groups),
                        ConditionalResBlock1D(d_in, d_in, global_dim, kernel_size, n_groups),
                        nn.ConvTranspose1d(d_in, d_in, 4, stride=2, padding=1),
                    ]
                )
            )
        self.final = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size, n_groups), nn.Conv1d(down_dims[0], action_dim, 1)
        )

    def forward(self, x_t, t, cond, tokens=None):
        x = x_t.transpose(1, 2)  # (N, A, T)
        pad = (-self.num_pts) % self.multiple
        if pad:
            x = F.pad(x, (0, pad))
        c = torch.cat([self.time_embed(t), cond], dim=-1)
        skips = []
        for res1, res2, down in self.down:
            x = res2(res1(x, c), c)
            skips.append(x)
            x = down(x)
        for block in self.mid:
            x = block(x, c)
        for res1, res2, up in self.up:
            x = torch.cat([x, skips.pop()], dim=1)
            x = res2(res1(x, c), c)
            x = up(x)
        x = self.final(x)[:, :, : self.num_pts]
        return x.transpose(1, 2)
