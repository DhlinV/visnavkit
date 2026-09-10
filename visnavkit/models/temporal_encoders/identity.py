"""Per-frame features with an optional temporal reduction."""

from .base import BaseTemporalEncoder

__all__ = ["IdentityTemporalEncoder"]


class IdentityTemporalEncoder(BaseTemporalEncoder):
    """Apply only the configured reduction, without learned temporal mixing."""

    def __init__(self, num_layers=0, **kwargs):
        if num_layers != 0:
            raise ValueError("IdentityTemporalEncoder requires num_layers=0")
        super().__init__(num_layers=0, **kwargs)
