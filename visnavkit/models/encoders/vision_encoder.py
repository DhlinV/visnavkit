"""Compatibility imports for stored configs; use spatial_encoders.vision_encoders."""

from visnavkit.models.spatial_encoders.vision_encoders.base import build_neck as _neck
from visnavkit.models.spatial_encoders.vision_encoders.timm import TimmVisionEncoder as VisionEncoder

__all__ = ["VisionEncoder", "_neck"]
