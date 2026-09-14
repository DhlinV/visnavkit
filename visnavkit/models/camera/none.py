from .base import BaseCameraEncoder

__all__ = ["NoCameraEncoder"]


class NoCameraEncoder(BaseCameraEncoder):
    """Calibration-agnostic policy: emits zero camera tokens."""

    def __init__(self, feat_size: int, **_ignored):
        super().__init__(feat_size, num_tokens=0)
