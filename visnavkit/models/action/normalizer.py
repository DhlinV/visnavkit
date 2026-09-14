"""Action normalization with buffers, so statistics travel inside checkpoints and ONNX graphs."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

__all__ = ["ActionNormalizer"]

MODES = ("none", "meanstd", "minmax")


class ActionNormalizer(nn.Module):
    """Affine map ``(x - offset) / scale`` over ``(..., T, A)`` actions.

    ``mode=none`` is the identity. ``meanstd`` and ``minmax`` (to [-1, 1]) read ``stats_path``
    (NPZ with ``mean``/``std`` or ``min``/``max``, shaped ``(T, A)`` or ``(A,)``) or are fit
    with :meth:`fit`. Generative decoders expect roughly unit-scale targets.
    """

    def __init__(self, mode: str = "none", stats_path: str | None = None, eps: float = 1e-6):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.eps = eps
        self.register_buffer("offset", torch.zeros(1, 1))
        self.register_buffer("scale", torch.ones(1, 1))
        if stats_path is not None:
            if mode == "none":
                raise ValueError("stats_path requires mode meanstd or minmax")
            self.load_stats(stats_path)

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
    def fit(self, actions: torch.Tensor, per_step: bool = True):
        """Set statistics from ``(N, T, A)`` actions; ``per_step=False`` shares them across anchors."""
        dims = (0,) if per_step else (0, 1)
        if self.mode == "meanstd":
            self._set(actions.mean(dims), actions.std(dims))
        elif self.mode == "minmax":
            low, high = actions.amin(dims), actions.amax(dims)
            self._set((high + low) / 2, (high - low) / 2)
        return self

    def save_stats(self, path):
        if self.mode == "meanstd":
            np.savez(path, mean=self.offset.cpu().numpy(), std=self.scale.cpu().numpy())
        elif self.mode == "minmax":
            offset, scale = self.offset.cpu().numpy(), self.scale.cpu().numpy()
            np.savez(path, min=offset - scale, max=offset + scale)

    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        if self.identity:
            return actions
        return (actions - self.offset.to(actions.dtype)) / self.scale.to(actions.dtype)

    def unnormalize(self, actions: torch.Tensor) -> torch.Tensor:
        if self.identity:
            return actions
        return actions * self.scale.to(actions.dtype) + self.offset.to(actions.dtype)
