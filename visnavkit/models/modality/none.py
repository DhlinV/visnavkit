from .base import BaseModalityEncoder

__all__ = ["NoModalityEncoder"]


class NoModalityEncoder(BaseModalityEncoder):
    """Placeholder that reads nothing and emits no tokens, so a slot can be switched off."""

    def __init__(self, feat_size: int, **_ignored):
        super().__init__(feat_size, num_tokens=0)
