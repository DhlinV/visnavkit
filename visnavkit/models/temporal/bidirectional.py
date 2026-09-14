"""Temporal attention over the complete observed frame window."""

from .base import BaseTemporalEncoder

__all__ = ["BidirectionalTemporalEncoder"]


class BidirectionalTemporalEncoder(BaseTemporalEncoder):
    """Attend to every frame in the window. Defaults to one decision per window.

    ``reduction=none`` exposes earlier frames' decisions to later observations, so it is refused
    unless ``allow_offline=True`` is set for an intentional offline sequence task.
    """

    def __init__(self, *args, allow_offline: bool = False, mask_p: float = 0.0, reduction="last", **kwargs):
        if reduction == "none" and not allow_offline:
            raise ValueError(
                "Bidirectional reduction=none leaks future frames into per-frame targets; set allow_offline=true"
            )
        super().__init__(*args, mask_p=mask_p, reduction=reduction, **kwargs)
