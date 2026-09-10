"""Compatibility imports for the original causal temporal encoder."""

from .causal import CausalTemporalEncoder, generate_causal_mask

TemporalEncoder = CausalTemporalEncoder

__all__ = ["TemporalEncoder", "generate_causal_mask"]
