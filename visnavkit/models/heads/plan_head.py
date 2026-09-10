"""Compatibility imports for the multi-hypothesis action head."""

from visnavkit.models.action_decoders.mhp import PlanHead
from visnavkit.models.action_decoders.outputs import parse_plan_output

__all__ = ["PlanHead", "parse_plan_output"]
