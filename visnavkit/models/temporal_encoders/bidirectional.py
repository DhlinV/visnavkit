"""Temporal attention over the complete observed frame window."""

from .base import BaseTemporalEncoder

__all__ = ["BidirectionalTemporalEncoder"]


class BidirectionalTemporalEncoder(BaseTemporalEncoder):
    """Attend to every frame in the input window without an attention mask.

    ``mask_p`` is accepted for configuration compatibility; history masking only
    applies to the causal encoder.
    """
