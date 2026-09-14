"""Noise schedulers on continuous time ``t in [0, 1]`` (``t=1`` pure noise, ``t=0`` clean)."""

import torch
import torch.nn as nn

__all__ = ["BaseScheduler"]


class BaseScheduler(nn.Module):
    """Training corrupts ``x0`` at a sampled ``t``; sampling walks ``step_times`` from 1 to 0."""

    def sample_t(self, n: int, device=None) -> torch.Tensor:
        raise NotImplementedError

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def target(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """What the denoiser regresses (noise for DDPM, velocity for flow matching)."""
        raise NotImplementedError

    def step_times(self, num_steps: int) -> list[float]:
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        return torch.linspace(1.0, 0.0, num_steps + 1).tolist()

    def step(
        self, model_output: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _broadcast(value: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        return value.reshape(-1, *([1] * (like.ndim - 1))).to(like.dtype)
