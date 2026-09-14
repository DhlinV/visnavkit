from .base import BaseEgoEncoder

__all__ = ["NoEgoEncoder"]


class NoEgoEncoder(BaseEgoEncoder):
    """Vision-only policy: emits zero ego tokens."""

    def __init__(self, feat_size: int, **_ignored):
        super().__init__(feat_size, num_tokens=0)
