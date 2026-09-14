"""Goal encoders: none, point, image, route image, instruction -> goal tokens."""

from .base import BaseGoalEncoder
from .image import ImageGoalEncoder
from .instruction import InstructionGoalEncoder
from .none import NoGoalEncoder
from .point import PointGoalEncoder
from .route import RouteImageGoalEncoder

__all__ = [
    "BaseGoalEncoder",
    "ImageGoalEncoder",
    "InstructionGoalEncoder",
    "NoGoalEncoder",
    "PointGoalEncoder",
    "RouteImageGoalEncoder",
]
