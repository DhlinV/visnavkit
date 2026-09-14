"""Ego-status encoders: the robot's own state -> per-frame tokens."""

from .base import BaseEgoEncoder
from .none import NoEgoEncoder
from .state import EgoStateEncoder

__all__ = ["BaseEgoEncoder", "EgoStateEncoder", "NoEgoEncoder"]
