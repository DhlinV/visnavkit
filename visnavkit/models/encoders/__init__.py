"""Legacy encoder imports; new code uses spatial_encoders.vision_encoders."""

from .dino_encoder import DinoEncoder
from .vision_encoder import VisionEncoder

__all__ = ["DinoEncoder", "VisionEncoder"]
