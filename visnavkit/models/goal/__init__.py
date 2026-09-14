"""Goal encoders: none, point, gps, image, route image, instruction -> goal tokens."""

from .base import BaseGoalEncoder
from .gps import GpsGoalEncoder
from .image import ImageGoalEncoder
from .instruction import InstructionGoalEncoder
from .none import NoGoalEncoder
from .point import PointGoalEncoder
from .route import RouteImageGoalEncoder

__all__ = [
    "BaseGoalEncoder",
    "GpsGoalEncoder",
    "ImageGoalEncoder",
    "InstructionGoalEncoder",
    "NoGoalEncoder",
    "PointGoalEncoder",
    "RouteImageGoalEncoder",
]
