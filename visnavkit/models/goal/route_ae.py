"""Route-patch autoencoders: a compact route code whose encoder seeds ``RouteImageGoalEncoder``."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .route import route_cnn

__all__ = ["RouteAutoencoder"]


class RouteAutoencoder(nn.Module):
    """AE, or VAE with ``variational=True``, over ``(N, C, H, W)`` route patches in [0, 1].

    ``encoder`` is exactly ``RouteImageGoalEncoder.cnn`` (same layers, same parameter names), so a
    checkpoint loads into the goal encoder through its ``weights`` option. The decoder mirrors the
    stride-2 stack from a ``(channels[-1], H / 2^L, W / 2^L)`` map, which pins ``image_hw`` at
    build time. Losses are per-sample sums (the ELBO convention), so ``beta=1`` is the plain VAE.
    """

    def __init__(
        self,
        in_chans: int = 3,
        channels=(32, 64, 128, 256),
        latent_dim: int = 128,
        variational: bool = False,
        image_hw=(64, 64),
        beta: float = 1.0,
    ):
        super().__init__()
        scale = 2 ** len(channels)
        height, width = image_hw
        if height % scale or width % scale:
            raise ValueError(
                f"image_hw must be multiples of {scale} for {len(channels)} stride-2 stages, got {image_hw}"
            )
        self.in_chans, self.image_hw, self.variational, self.beta = in_chans, (height, width), variational, beta
        self.encoder = route_cnn(in_chans, channels)
        self.to_latent = nn.Linear(channels[-1], latent_dim * (2 if variational else 1))
        self.grid = (channels[-1], height // scale, width // scale)
        self.from_latent = nn.Linear(latent_dim, math.prod(self.grid))
        layers, width = [], channels[-1]
        for out in reversed(channels[:-1]):
            layers += [
                nn.ConvTranspose2d(width, out, 4, stride=2, padding=1),
                nn.BatchNorm2d(out),
                nn.ReLU(inplace=True),
            ]
            width = out
        self.decoder = nn.Sequential(*layers, nn.ConvTranspose2d(width, in_chans, 4, stride=2, padding=1))

    def encode(self, x: torch.Tensor):
        """Patches -> ``(z, mu, logvar)``; the AE returns ``(z, None, None)``, the VAE samples only in training."""
        latent = self.to_latent(self.encoder(x))
        if not self.variational:
            return latent, None, None
        mu, logvar = latent.chunk(2, dim=1)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if self.training else mu
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decoder(self.from_latent(z).reshape(-1, *self.grid)))

    def forward(self, x: torch.Tensor) -> dict:
        if x.ndim != 4 or tuple(x.shape[1:]) != (self.in_chans, *self.image_hw):
            raise ValueError(
                f"Expected route patches (N, {self.in_chans}, {self.image_hw[0]}, {self.image_hw[1]}), got {tuple(x.shape)}"
            )
        z, mu, logvar = self.encode(x)
        return dict(recon=self.decode(z), z=z, mu=mu, logvar=logvar)

    def loss(self, x: torch.Tensor, out: dict) -> dict:
        recon = F.mse_loss(out["recon"], x, reduction="sum") / x.shape[0]
        if out["logvar"] is None:
            kl = torch.zeros((), device=x.device)
        else:
            kl = -0.5 * (1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp()).sum(dim=1).mean()
        return dict(loss=recon + self.beta * kl, recon=recon, kl=kl)
