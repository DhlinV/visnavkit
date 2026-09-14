from .base import BaseGoalEncoder

__all__ = ["NoGoalEncoder"]


class NoGoalEncoder(BaseGoalEncoder):
    """Goal-free policy: emits zero goal tokens, so the decoder sees context only."""

    goal_type = "none"

    def __init__(self, feat_size: int, **_ignored):
        super().__init__(feat_size, num_tokens=0)
