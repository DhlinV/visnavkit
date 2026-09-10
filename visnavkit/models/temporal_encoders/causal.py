"""Causal temporal attention with optional training-time history masking."""

import torch

from .base import BaseTemporalEncoder

__all__ = ["CausalTemporalEncoder", "generate_causal_mask"]


def generate_causal_mask(seq_len: int, mask_p: float = 0.0, device=None) -> torch.Tensor:
    """Mask future frames and randomly drop past-frame attention during training.

    The diagonal is always visible. History dropout uses one mask for the batch,
    preserving the original encoder's attention behavior.
    """
    if seq_len < 1:
        raise ValueError("seq_len must be positive")
    if not 0 <= mask_p <= 1:
        raise ValueError("mask_p must be between 0 and 1")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float("-inf"), diagonal=1)
    if mask_p > 0.0:
        rand_mask = torch.rand(seq_len, seq_len, device=device) < mask_p
        lower_triangle = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=-1)
        random_mask = rand_mask * lower_triangle
        causal_mask = causal_mask.masked_fill(random_mask.bool(), float("-inf"))
    return causal_mask


class CausalTemporalEncoder(BaseTemporalEncoder):
    """Attend to the current frame and past frames only."""

    def _attention_mask(self, seq_len, device):
        return generate_causal_mask(seq_len, mask_p=self.mask_p if self.training else 0, device=device)
