"""Per-frame tokens with an optional temporal reduction and no learned mixing."""

from .base import BaseTemporalEncoder

__all__ = ["IdentityTemporalEncoder"]


class IdentityTemporalEncoder(BaseTemporalEncoder):
    """Single-frame policies (GNM-style) or plain averaging over the window."""

    def __init__(self, num_layers=0, reduction="none", **kwargs):
        if num_layers != 0:
            raise ValueError("IdentityTemporalEncoder requires num_layers=0")
        super().__init__(num_layers=0, reduction=reduction, **kwargs)
