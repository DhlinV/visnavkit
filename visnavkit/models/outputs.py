"""Typed component outputs (a small dataclass counterpart of diffusers' ``BaseOutput``)."""

from dataclasses import dataclass, fields

import torch

__all__ = ["BaseOutput", "PlanOutput", "PolicyOutput", "VisionOutput"]


@dataclass
class BaseOutput:
    """Attribute access for readers, ``out["name"]`` and ``dict(out)`` for generic code."""

    def __getitem__(self, key: str):
        return getattr(self, key)

    def keys(self):
        return [f.name for f in fields(self) if getattr(self, f.name) is not None]

    def items(self):
        return [(name, getattr(self, name)) for name in self.keys()]


@dataclass
class VisionOutput(BaseOutput):
    """Per-frame vision features.

    Args:
        tokens: ``(N, K, D)`` decoder-ready tokens (``K`` = 1 global, patch grid, or 1 + grid).
        pose: ``(N, 1)`` auxiliary speed regression.
        prev_img_mask: ``(N,)`` False where the previous frame was dropped by augmentation.
        heads: extra spatial-head outputs keyed by their export names.
    """

    tokens: torch.Tensor
    pose: torch.Tensor
    prev_img_mask: torch.Tensor
    heads: dict | None = None


@dataclass
class PlanOutput(BaseOutput):
    """Action decoder output.

    Args:
        plans: ``(N, M * (2 * T * P + 1))`` flat per-mode ``[mu, log_scale, confidence_logit]`` in pose
            space; every decoder emits this layout so metrics and export never change.
        mu: ``(N, M, T, A)`` normalized action-space means (None while generative decoders train).
        log_b: ``(N, M, T, A)`` normalized Laplace log-scales (MHP only).
        logits: ``(N, M)`` mode logits (MHP and anchor decoders).
        cond: ``(N, D)`` pooled conditioning vector, kept for generative losses.
        tokens: ``(N, L, D)`` context plus goal tokens, kept for cross-attention denoisers.
    """

    plans: torch.Tensor
    mu: torch.Tensor | None = None
    log_b: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    cond: torch.Tensor | None = None
    tokens: torch.Tensor | None = None


@dataclass
class PolicyOutput(BaseOutput):
    vision: VisionOutput
    plan: PlanOutput
    goal_tokens: torch.Tensor | None = None
