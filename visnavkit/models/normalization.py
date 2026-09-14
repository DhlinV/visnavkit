"""Affine normalization shared by supervision targets and input signals.

Supervision and inputs rarely want the same treatment — action targets are usually fit to a
corpus, a goal distance is a hand-set scale, an ego vector mixes m/s with rad/s — so every
stage that needs normalization owns its own :class:`Normalizer` instead of sharing one global
convention. Statistics live in buffers, so they travel inside checkpoints and ONNX graphs and
training, export and deployment cannot drift apart.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

__all__ = ["MODES", "Normalizer"]

MODES = ("none", "meanstd", "minmax", "scale")


class Normalizer(nn.Module):
    """Affine map ``(x - offset) / scale`` over the last one or two axes.

    Args:
        mode: ``none`` is the identity. ``meanstd`` and ``minmax`` (to [-1, 1]) are fit with
            :meth:`fit` or read from ``stats_path`` (NPZ with ``mean``/``std`` or ``min``/``max``).
            ``scale`` takes the divisor straight from ``scale``, for quantities with a known
            range that needs no corpus (a goal distance in metres, a normalized image size).
        stats_path: NPZ of statistics, shaped ``(T, A)`` or ``(A,)``.
        scale: divisor for ``mode=scale``; a scalar or one value per channel.
        offset: subtracted before scaling in ``mode=scale`` (default 0).
    """

    def __init__(
        self,
        mode: str = "none",
        stats_path: str | None = None,
        scale: float | list[float] | None = None,
        offset: float | list[float] | None = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.eps = eps
        self.register_buffer("offset", torch.zeros(1, 1))
        self.register_buffer("scale", torch.ones(1, 1))
        if mode == "scale":
            if scale is None:
                raise ValueError("mode=scale requires scale")
            values = np.atleast_1d(np.asarray(scale, dtype=np.float32))
            self._set(np.zeros_like(values) if offset is None else np.broadcast_to(offset, values.shape), values)
        elif stats_path is not None:
            if mode == "none":
                raise ValueError("stats_path requires mode meanstd or minmax")
            self.load_stats(stats_path)
        elif scale is not None or offset is not None:
            raise ValueError("scale/offset are only used with mode=scale")

    @property
    def identity(self) -> bool:
        return self.mode == "none"

    def _set(self, offset, scale):
        offset = torch.as_tensor(np.asarray(offset), dtype=torch.float32)
        scale = torch.as_tensor(np.asarray(scale), dtype=torch.float32).clamp_min(self.eps)
        if offset.ndim == 1:
            offset, scale = offset[None], scale[None]
        if offset.ndim != 2 or offset.shape != scale.shape:
            raise ValueError("normalization statistics must be shaped (T, A) or (A,)")
        self.offset = offset
        self.scale = scale

    def load_stats(self, path):
        stats = np.load(Path(path))
        if self.mode == "meanstd":
            self._set(stats["mean"], stats["std"])
        else:
            low, high = np.asarray(stats["min"]), np.asarray(stats["max"])
            self._set((high + low) / 2, (high - low) / 2)

    @torch.no_grad()
    def fit(self, values: torch.Tensor, per_step: bool = True):
        """Set statistics from ``(N, T, A)`` values; ``per_step=False`` shares them across steps."""
        if self.mode in ("none", "scale"):
            raise ValueError(f"mode={self.mode} has nothing to fit")
        dims = (0,) if per_step else (0, 1)
        if self.mode == "meanstd":
            self._set(values.mean(dims), values.std(dims))
        else:
            low, high = values.amin(dims), values.amax(dims)
            self._set((high + low) / 2, (high - low) / 2)
        return self

    def save_stats(self, path):
        if self.mode == "meanstd":
            np.savez(path, mean=self.offset.cpu().numpy(), std=self.scale.cpu().numpy())
        elif self.mode == "minmax":
            offset, scale = self.offset.cpu().numpy(), self.scale.cpu().numpy()
            np.savez(path, min=offset - scale, max=offset + scale)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        if self.identity:
            return values
        return (values - self.offset.to(values.dtype)) / self.scale.to(values.dtype)

    def unnormalize(self, values: torch.Tensor) -> torch.Tensor:
        if self.identity:
            return values
        return values * self.scale.to(values.dtype) + self.offset.to(values.dtype)
